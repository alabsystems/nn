// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for end-to-end attention → decoder pipeline.
//!
//! Phase 27 of #1729: connects the deep attention stack (Phase 25) to the
//! Kokoro decoder (Phase 26) via a context-vector bridge.
//!
//! Architecture:
//! ```text
//!   Input: hidden [T_dec, D_model] (Variable)
//!   ┌───────────────────────────────────────────────────────────────────────┐
//!   │ Deep Attention Stack (N layers)                                      │
//!   │   Layer i: Q_i @ K_i^T / √d_k + mask → Softmax → W_i              │
//!   │           Context_i = W_i @ V → FFN → Residual                     │
//!   │   Final layer → attention weights [H, T_dec, T_enc]                 │
//!   └──────────────────────────────┬────────────────────────────────────────┘
//!                                  │
//!   ┌──────────────────────────────▼────────────────────────────────────────┐
//!   │ Context Bridge                                                       │
//!   │   Context = W_final @ V_final:  [H, T_dec, d_k]                    │
//!   │   Reshape → [T_dec, D_model] → Linear → [T_dec, decoder_channels]  │
//!   │   Transpose → [decoder_channels, T_dec]  (Kokoro decoder input)    │
//!   └──────────────────────────────┬────────────────────────────────────────┘
//!                                  │
//!   ┌──────────────────────────────▼────────────────────────────────────────┐
//!   │ Kokoro Decoder (simplified)                                          │
//!   │   Conv1d → LeakyReLU → ConvTranspose1d → ResBlock → LeakyReLU      │
//!   │   → Conv1d → Exp → magnitude spectrum [OUT_CH, TIME_UP]             │
//!   └───────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! The key insight: we build ONE unified graph (not composition via
//! `compose_sequential`) so NY propagates bounds through the
//! full chain in a single pass. The context bridge is inline.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 27.

// Helpers shared across test binaries; not all functions used by all binaries.
#![allow(dead_code, clippy::duplicated_attributes)]

use nn_dsl::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNodeId};
use nn_verify::{BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// Shared decoder dimensions — delegated to super::common::decoder_common (Part of #1970).
pub(super) use super::common::decoder_common::{
    D_K, D_MODEL, FFN_DIM, NUM_HEADS, T_DEC, T_ENC, UPSAMPLE_KERNEL, UPSAMPLE_PADDING,
    UPSAMPLE_STRIDE, WEIGHT_MAG,
};

// File-specific constants for the attention→decoder pipeline.

/// Decoder channels (Kokoro decoder input channels).
/// Must equal T_DEC for the reshape bridge to work with small dimensions.
pub(super) const DECODER_CHANNELS: usize = 4;

/// Output channels from Kokoro decoder.
pub(super) const OUT_CHANNELS: usize = 4;

/// Upsampled channels after ConvTranspose1d.
const UPSAMPLED_CHANNELS: usize = 4;

/// Output time after upsampling.
/// conv_transpose1d: (in-1)*stride + kernel - 2*padding
pub(super) const TIME_UP: usize =
    (T_DEC - 1) * UPSAMPLE_STRIDE + UPSAMPLE_KERNEL - 2 * UPSAMPLE_PADDING;

// Shared mask helpers — delegated to super::common (Part of #1970).
use super::common::bounds_min_max;
use super::common::build_strict_causal_mask;

// Sinusoidal PE — delegated to super::common (Part of #1970).
use super::common::sinusoidal_pe_interleaved as sinusoidal_pe;

/// ResBlock conv kernel size.
const RESBLOCK_KERNEL: usize = 3;

/// ResBlock conv padding.
const RESBLOCK_PADDING: usize = 1;

// Weight constructors — delegated to super::common::weights (Part of #1938).
use super::common::weights;

// ---------------------------------------------------------------------------
// Per-layer attention inputs
// ---------------------------------------------------------------------------

struct AttentionLayerInputs {
    w_q: TensorNodeId,
    w_k: TensorNodeId,
    w_v: TensorNodeId,
    w_o: TensorNodeId,
    mask: TensorNodeId,
    ln_weight: TensorNodeId,
    ln_bias: TensorNodeId,
    ln_eps: TensorNodeId,
    ffn_up: TensorNodeId,
    ffn_down: TensorNodeId,
}

// ---------------------------------------------------------------------------
// End-to-end pipeline builder
// ---------------------------------------------------------------------------

/// Build the full attention → context bridge → decoder pipeline.
///
/// `num_attn_layers`: Number of attention layers (≥2). Layers 0..N-2 produce
/// context+residual+FFN. Layer N-1 produces attention weights AND context
/// for the decoder.
///
/// Returns `(TensorKernelDef, output_shape)` where output is the decoder's
/// magnitude spectrum `[OUT_CHANNELS, TIME_UP]`.
pub(super) fn build_attention_decoder_pipeline(
    num_attn_layers: usize,
) -> (TensorKernelDef, [usize; 2]) {
    assert!(
        num_attn_layers >= 2,
        "pipeline needs at least 2 attention layers"
    );

    let scale = 1.0 / (D_K as f32).sqrt();
    let scores_shape = [NUM_HEADS, T_DEC, T_ENC];
    let ctx_shape = [NUM_HEADS, T_DEC, D_K];

    let mut b = TensorBlockBuilder::new("attention_decoder_pipeline");

    // === Global attention inputs ===
    let hidden = b.add_input("hidden", &[T_DEC, D_MODEL]);
    let dec_pe = b.add_input("dec_pe", &[T_DEC, D_MODEL]);
    let enc_k_input = b.add_input("enc_k", &[T_ENC, D_MODEL]);
    let enc_v_input = b.add_input("enc_v", &[T_ENC, D_MODEL]);

    // === Per-layer attention inputs ===
    let mut layer_inputs = Vec::with_capacity(num_attn_layers);
    for i in 0..num_attn_layers {
        let suffix = format!("_L{i}");
        layer_inputs.push(AttentionLayerInputs {
            w_q: b.add_input(&format!("w_q{suffix}"), &[D_MODEL, D_MODEL]),
            w_k: b.add_input(&format!("w_k{suffix}"), &[D_MODEL, D_MODEL]),
            w_v: b.add_input(&format!("w_v{suffix}"), &[D_MODEL, D_MODEL]),
            w_o: b.add_input(&format!("w_o{suffix}"), &[D_MODEL, D_MODEL]),
            mask: b.add_input(&format!("mask{suffix}"), &[T_DEC, T_ENC]),
            ln_weight: b.add_input(&format!("ln_w{suffix}"), &[D_MODEL]),
            ln_bias: b.add_input(&format!("ln_b{suffix}"), &[D_MODEL]),
            ln_eps: b.add_input(&format!("ln_eps{suffix}"), &[1]),
            ffn_up: b.add_input(&format!("ffn_up{suffix}"), &[FFN_DIM, D_MODEL]),
            ffn_down: b.add_input(&format!("ffn_down{suffix}"), &[D_MODEL, FFN_DIM]),
        });
    }

    // === Context bridge inputs ===
    let bridge_proj_w = b.add_input("bridge_proj_w", &[D_MODEL, DECODER_CHANNELS]);

    // === Decoder inputs ===
    let dec_eps = b.add_input("dec_eps", &[1]);
    let conv_pre_w = b.add_input("conv_pre_w", &[DECODER_CHANNELS, DECODER_CHANNELS, 7]);
    let upsample_w = b.add_input(
        "upsample_w",
        &[DECODER_CHANNELS, UPSAMPLED_CHANNELS, UPSAMPLE_KERNEL],
    );
    let style_gamma = b.add_input("res_style_gamma", &[UPSAMPLED_CHANNELS]);
    let style_beta = b.add_input("res_style_beta", &[UPSAMPLED_CHANNELS]);
    let res_alpha = b.add_input("res_alpha", &[1]);
    let res_conv_w = b.add_input(
        "res_conv_w",
        &[UPSAMPLED_CHANNELS, UPSAMPLED_CHANNELS, RESBLOCK_KERNEL],
    );
    let conv_post_w = b.add_input("conv_post_w", &[OUT_CHANNELS, UPSAMPLED_CHANNELS, 7]);

    // ===================================================================
    // Stage 1: Deep Attention Stack
    // ===================================================================

    // Layer 0: add PE to hidden
    let mut prev = b.add_binary_add(hidden, dec_pe, &[T_DEC, D_MODEL]);

    // Track final-layer context for the bridge
    let mut final_ctx_flat: Option<TensorNodeId> = None;

    for (layer_idx, li) in layer_inputs.iter().enumerate() {
        let is_last = layer_idx == num_attn_layers - 1;

        // Q = prev @ W_q, K = enc_k @ W_k
        let q = b.add_matmul(prev, li.w_q, false, None, &[T_DEC, D_MODEL]);
        let k = b.add_matmul(enc_k_input, li.w_k, false, None, &[T_ENC, D_MODEL]);

        // Multi-head reshape + transpose
        let q_r = b.add_reshape(q, &[T_DEC, NUM_HEADS, D_K]);
        let k_r = b.add_reshape(k, &[T_ENC, NUM_HEADS, D_K]);
        let q_t = b.add_transpose(q_r, &[1, 0, 2], &[NUM_HEADS, T_DEC, D_K]);
        let k_t = b.add_transpose(k_r, &[1, 0, 2], &[NUM_HEADS, T_ENC, D_K]);

        // Scores = Q @ K^T / √d_k + mask → Softmax
        let scores = b.add_matmul(q_t, k_t, true, Some(scale), &scores_shape);
        let mask_bc = b.add_broadcast(li.mask, &scores_shape);
        let masked = b.add_binary_add(scores, mask_bc, &scores_shape);
        let weights = b.add_softmax(masked, -1, &scores_shape);

        // V = enc_v @ W_v (always compute V — we need context for both
        // intermediate layers and the final layer's bridge)
        let v = b.add_matmul(enc_v_input, li.w_v, false, None, &[T_ENC, D_MODEL]);
        let v_r = b.add_reshape(v, &[T_ENC, NUM_HEADS, D_K]);
        let v_t = b.add_transpose(v_r, &[1, 0, 2], &[NUM_HEADS, T_ENC, D_K]);

        // Context = W @ V: [H, T_dec, T_enc] @ [H, T_enc, d_k] → [H, T_dec, d_k]
        let ctx = b.add_matmul(weights, v_t, false, None, &ctx_shape);

        // Transpose back: [H, T_dec, d_k] → [T_dec, H, d_k]
        let ctx_t = b.add_transpose(ctx, &[1, 0, 2], &[T_DEC, NUM_HEADS, D_K]);
        let ctx_flat = b.add_reshape(ctx_t, &[T_DEC, D_MODEL]);

        if is_last {
            // Save final context for the bridge
            final_ctx_flat = Some(ctx_flat);
            // Don't do residual/FFN on the last layer — go straight to decoder
            break;
        }

        // Output projection + Residual + LayerNorm + FFN
        let attn_out = b.add_matmul(ctx_flat, li.w_o, false, None, &[T_DEC, D_MODEL]);
        let res = b.add_binary_add(prev, attn_out, &[T_DEC, D_MODEL]);
        let normed = b.add_layer_norm(
            res,
            li.ln_eps,
            1,
            li.ln_weight,
            li.ln_bias,
            &[T_DEC, D_MODEL],
        );
        let ffn1 = b.add_linear(normed, li.ffn_up, None, &[T_DEC, FFN_DIM]);
        let act = b.add_gelu(ffn1, &[T_DEC, FFN_DIM]);
        let ffn2 = b.add_linear(act, li.ffn_down, None, &[T_DEC, D_MODEL]);
        let ffn_res = b.add_binary_add(res, ffn2, &[T_DEC, D_MODEL]);

        prev = ffn_res;
    }

    let ctx_flat = final_ctx_flat.expect("at least 2 layers");

    // ===================================================================
    // Stage 2: Context Bridge (attention output → decoder input)
    // ===================================================================

    // Project context from D_MODEL to DECODER_CHANNELS:
    // [T_DEC, D_MODEL] @ [D_MODEL, DECODER_CHANNELS] → [T_DEC, DECODER_CHANNELS]
    let bridge_out = b.add_matmul(
        ctx_flat,
        bridge_proj_w,
        false,
        None,
        &[T_DEC, DECODER_CHANNELS],
    );

    // Transpose to [DECODER_CHANNELS, T_DEC] (Kokoro decoder expects [C, T])
    let decoder_input = b.add_transpose(bridge_out, &[1, 0], &[DECODER_CHANNELS, T_DEC]);

    // ===================================================================
    // Stage 3: Kokoro Decoder (simplified with LeakyReLU)
    // ===================================================================

    let up_shape = [UPSAMPLED_CHANNELS, TIME_UP];

    // Conv pre: [DECODER_CHANNELS, T_DEC] → [DECODER_CHANNELS, T_DEC]
    let x = b.add_conv1d(
        decoder_input,
        conv_pre_w,
        None,
        1,
        3,
        &[DECODER_CHANNELS, T_DEC],
    );

    // LeakyReLU(0.1) before upsample
    let x_act = b.add_leaky_relu(x, 0.1, &[DECODER_CHANNELS, T_DEC]);

    // ConvTranspose1d upsample: [DECODER_CHANNELS, T_DEC] → [UPSAMPLED_CHANNELS, TIME_UP]
    let x_up = b.add_conv_transpose_1d(
        x_act,
        upsample_w,
        None,
        UPSAMPLE_STRIDE,
        UPSAMPLE_PADDING,
        1,
        1,
        0, // output_padding
        &up_shape,
    );

    // ResBlock: InstanceNorm + Snake + Conv1d + residual
    let normed = b.add_instance_norm(
        x_up,
        dec_eps,
        1,
        Some(style_gamma),
        Some(style_beta),
        &up_shape,
    );

    let alpha_bc = b.add_broadcast(res_alpha, &up_shape);
    let snake_kernel = build_snake_scalar_kernel().expect("snake kernel");
    let snake_out = b.add_elementwise(snake_kernel, &[normed, alpha_bc], &up_shape);

    let sublayer_out = b.add_conv1d(snake_out, res_conv_w, None, 1, RESBLOCK_PADDING, &up_shape);

    // Residual
    let res_out = b.add_binary_add(x_up, sublayer_out, &up_shape);

    // LeakyReLU(0.01) before conv_post
    let res_act = b.add_leaky_relu(res_out, 0.01, &up_shape);

    // Conv post: [UPSAMPLED_CHANNELS, TIME_UP] → [OUT_CHANNELS, TIME_UP]
    let x_post = b.add_conv1d(res_act, conv_post_w, None, 1, 3, &[OUT_CHANNELS, TIME_UP]);

    // Exp (log-magnitude → magnitude)
    let output = b.add_exp(x_post, &[OUT_CHANNELS, TIME_UP]);

    let out_shape = [OUT_CHANNELS, TIME_UP];
    (
        b.build(output)
            .expect("valid attention decoder pipeline graph"),
        out_shape,
    )
}

// ---------------------------------------------------------------------------
// Binding constructors
// ---------------------------------------------------------------------------

/// Build bindings for the full attention → decoder pipeline.
pub(super) fn pipeline_bindings(
    num_attn_layers: usize,
    pe_scale: f32,
    w_perturbation: f32,
) -> Vec<TensorParamBinding> {
    let dec_pe = {
        let mut pe = sinusoidal_pe(T_DEC, D_MODEL, NUM_HEADS);
        pe.mapv_inplace(|v| v * pe_scale);
        pe
    };
    let enc_k = weights::encoder_k(T_ENC, D_MODEL);
    let enc_v = weights::encoder_k(T_ENC, D_MODEL);
    let w_proj = weights::near_identity(D_MODEL, w_perturbation);
    let mask = build_strict_causal_mask(T_DEC, T_ENC);
    let ln_w = weights::norm_weight(D_MODEL);
    let ln_b = weights::norm_bias(D_MODEL);
    let ffn_up_w = weights::ffn_weight(FFN_DIM, D_MODEL, 0.1);
    let ffn_down_w = weights::ffn_weight(D_MODEL, FFN_DIM, 0.1);

    let mut bindings = vec![
        TensorParamBinding::Variable,               // hidden
        TensorParamBinding::ConstantTensor(dec_pe), // dec_pe
        TensorParamBinding::ConstantTensor(enc_k),  // enc_k
        TensorParamBinding::ConstantTensor(enc_v),  // enc_v
    ];

    // Per-layer attention bindings
    for _ in 0..num_attn_layers {
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone())); // w_q
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone())); // w_k
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone())); // w_v
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone())); // w_o
        bindings.push(TensorParamBinding::ConstantTensor(mask.clone())); // mask
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // ln_weight
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // ln_bias
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ln_eps
        bindings.push(TensorParamBinding::ConstantTensor(ffn_up_w.clone())); // ffn_up
        bindings.push(TensorParamBinding::ConstantTensor(ffn_down_w.clone())); // ffn_down
    }

    // Context bridge: projection [D_MODEL, DECODER_CHANNELS]
    let bridge_w = weights::ffn_weight(D_MODEL, DECODER_CHANNELS, 0.1);
    bindings.push(TensorParamBinding::ConstantTensor(bridge_w));

    // Decoder inputs
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // dec_eps

    // conv_pre weight [DECODER_CHANNELS, DECODER_CHANNELS, 7]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[DECODER_CHANNELS, DECODER_CHANNELS, 7]),
        WEIGHT_MAG,
    )));

    // upsample weight [DECODER_CHANNELS, UPSAMPLED_CHANNELS, UPSAMPLE_KERNEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[DECODER_CHANNELS, UPSAMPLED_CHANNELS, UPSAMPLE_KERNEL]),
        WEIGHT_MAG,
    )));

    // style_gamma [UPSAMPLED_CHANNELS]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[UPSAMPLED_CHANNELS]),
        1.0f32,
    )));

    // style_beta [UPSAMPLED_CHANNELS]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[UPSAMPLED_CHANNELS]),
        0.0f32,
    )));

    // res_alpha (scalar)
    bindings.push(TensorParamBinding::ConstantScalar(1.0));

    // res_conv weight [UPSAMPLED_CHANNELS, UPSAMPLED_CHANNELS, RESBLOCK_KERNEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[UPSAMPLED_CHANNELS, UPSAMPLED_CHANNELS, RESBLOCK_KERNEL]),
        WEIGHT_MAG,
    )));

    // conv_post weight [OUT_CHANNELS, UPSAMPLED_CHANNELS, 7]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[OUT_CHANNELS, UPSAMPLED_CHANNELS, 7]),
        WEIGHT_MAG,
    )));

    bindings
}

// ---------------------------------------------------------------------------
// Pipeline result analysis
// ---------------------------------------------------------------------------

/// Result from analyzing the end-to-end pipeline output.
#[derive(Debug)]
pub(super) struct PipelineResult {
    pub(super) num_attn_layers: usize,
    pub(super) graph_nodes: usize,
    pub(super) output_shape: Vec<usize>,
    pub(super) min_output_lo: f32,
    pub(super) max_output_hi: f32,
    pub(super) avg_bound_width: f32,
    pub(super) all_positive: bool,
    pub(super) all_finite: bool,
}

/// Analyze pipeline output bounds.
pub(super) fn analyze_pipeline_output(
    output: &BoundedTensor,
    num_attn_layers: usize,
    graph_nodes: usize,
) -> PipelineResult {
    let (lo, hi) = output.lower_upper();
    let shape: Vec<usize> = lo.shape().to_vec();
    let flat_lo: Vec<f32> = lo.iter().copied().collect();
    let flat_hi: Vec<f32> = hi.iter().copied().collect();

    let (min_lo, max_hi) = bounds_min_max(output);

    let avg_width: f32 = flat_lo
        .iter()
        .zip(flat_hi.iter())
        .map(|(&l, &h)| h - l)
        .sum::<f32>()
        / flat_lo.len() as f32;

    let all_positive = flat_lo.iter().all(|&v| v >= 0.0);
    let all_finite = flat_lo.iter().chain(flat_hi.iter()).all(|v| v.is_finite());

    PipelineResult {
        num_attn_layers,
        graph_nodes,
        output_shape: shape,
        min_output_lo: min_lo,
        max_output_hi: max_hi,
        avg_bound_width: avg_width,
        all_positive,
        all_finite,
    }
}
