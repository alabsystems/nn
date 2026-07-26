// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Whisper cross-attention NY composition.
//!
//! Verifies bounds propagation through the Whisper decoder's cross-attention
//! mechanism, where Q comes from the decoder hidden state and K/V come from
//! the encoder output (bound as constant).
//!
//! This file decomposes cross-attention into its individual sub-components:
//!
//! 1. **Q projection**: Linear projection of decoder hidden state to query space.
//! 2. **K/V projection**: Linear projection of encoder output to key/value space
//!    (constant binding -- encoder certified separately).
//! 3. **Full cross-attention**: Q projection + K/V projection + scaled dot-product
//!    attention + output projection. The output shape follows Q (decoder seq len).
//! 4. **Full decoder block**: SelfAttn(causal) + CrossAttn(encoder) + FFN with
//!    residual connections and pre-norm LayerNorms.
//! 5. **2-block decoder stack**: Two sequential decoder blocks with shared
//!    encoder conditioning, testing bounds stability through depth.
//!
//! Architecture (Radford et al. 2023, "Robust Speech Recognition via Large-Scale
//! Weak Supervision"):
//! - Cross-attention: Q from decoder, K/V from frozen encoder output
//! - Standard (bidirectional) mask: decoder can attend to ALL encoder positions
//! - Pre-norm: LayerNorm before each sub-block
//! - FFN: Linear(D, FFN_DIM) -> GELU -> Linear(FFN_DIM, D)
//!
//! GELU requires CROWN linearization. LayerNorm requires heuristic linearization
//! (IbpValidated mode). Softmax in attention uses piecewise CROWN approximation.
//!
//! Part of #3572: Whisper cross-attention compose verification tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding, VerificationSoundnessMode};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- per issue #3572 specification
// ---------------------------------------------------------------------------

/// Decoder sequence length (number of tokens).
const DECODER_SEQ: usize = 4;
/// Encoder output sequence length (e.g., mel frames after conv stems).
const ENCODER_SEQ: usize = 8;
/// Embedding / model dimension.
const EMBED_DIM: usize = 32;
/// Number of attention heads (head_dim = EMBED_DIM / NUM_HEADS = 8).
const NUM_HEADS: usize = 4;
/// Head dimension (EMBED_DIM / NUM_HEADS).
const HEAD_DIM: usize = EMBED_DIM / NUM_HEADS;
/// FFN intermediate dimension.
const FFN_DIM: usize = 128;
/// Small weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Builder helpers: Q projection (decoder side)
// ---------------------------------------------------------------------------

/// Build a Q projection kernel: Linear(decoder_hidden -> query space).
///
/// Input: `[DECODER_SEQ, EMBED_DIM]` (Variable -- decoder hidden state).
/// Output: `[DECODER_SEQ, EMBED_DIM]` (Q projected, before reshape to heads).
///
/// This isolates the decoder-side query projection, the first step of
/// cross-attention where only decoder state contributes.
fn build_q_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_xattn_q_proj");

    let decoder_hidden = b.add_input("decoder_hidden", &[DECODER_SEQ, EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);

    let q_proj = b.add_linear(decoder_hidden, q_w, None, &[DECODER_SEQ, EMBED_DIM]);

    b.build(q_proj).expect("valid Q projection kernel")
}

/// Bindings for Q projection: decoder_hidden=Variable, q_weight=Constant.
fn q_projection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // decoder_hidden [DECODER_SEQ, EMBED_DIM]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, EMBED_DIM]),
            WEIGHT_MAG,
        )), // q_weight [EMBED_DIM, EMBED_DIM]
    ]
}

// ---------------------------------------------------------------------------
// Builder helpers: K/V projection (encoder side)
// ---------------------------------------------------------------------------

/// Build a K/V projection kernel: Linear(encoder_output -> key space) + Linear(encoder_output -> value space).
///
/// Input: `[ENCODER_SEQ, EMBED_DIM]` (Variable -- encoder output for testing).
/// Output: `[ENCODER_SEQ, EMBED_DIM * 2]` (K and V concatenated).
///
/// In production, encoder output is a constant. Here we make it Variable to
/// verify bounds propagation through the K/V branch independently. The full
/// cross-attention test binds encoder output as Constant.
fn build_kv_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_xattn_kv_proj");

    let encoder_output = b.add_input("encoder_output", &[ENCODER_SEQ, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);

    // K projection: [ENCODER_SEQ, EMBED_DIM] -> [ENCODER_SEQ, EMBED_DIM]
    let k_proj = b.add_linear(encoder_output, k_w, None, &[ENCODER_SEQ, EMBED_DIM]);

    // V projection: [ENCODER_SEQ, EMBED_DIM] -> [ENCODER_SEQ, EMBED_DIM]
    let v_proj = b.add_linear(encoder_output, v_w, None, &[ENCODER_SEQ, EMBED_DIM]);

    // Concatenate K and V along the last axis for a single output
    let out = b.add_concat(&[k_proj, v_proj], 1, &[ENCODER_SEQ, EMBED_DIM * 2]);

    b.build(out).expect("valid K/V projection kernel")
}

/// Bindings for K/V projection: encoder_output=Variable, weights=Constant.
fn kv_projection_bindings() -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[EMBED_DIM, EMBED_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable, // encoder_output [ENCODER_SEQ, EMBED_DIM]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight
        TensorParamBinding::ConstantTensor(w_proj), // v_weight
    ]
}

// ---------------------------------------------------------------------------
// Builder helpers: Full cross-attention
// ---------------------------------------------------------------------------

/// Build a full cross-attention kernel.
///
/// Q input: `[DECODER_SEQ, EMBED_DIM]` (Variable -- decoder hidden state).
/// KV input: `[ENCODER_SEQ, EMBED_DIM]` (ConstantTensor -- encoder output).
/// Output: `[DECODER_SEQ, EMBED_DIM]`.
///
/// Uses `add_multi_head_cross_attention` which handles Q/K/V projections,
/// multi-head reshape, scaled dot-product attention, and output projection.
/// Standard (bidirectional) mask: decoder can attend to all encoder positions.
fn build_full_cross_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_xattn_full");

    let decoder_hidden = b.add_input("decoder_hidden", &[DECODER_SEQ, EMBED_DIM]);
    let encoder_output = b.add_input("encoder_output", &[ENCODER_SEQ, EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);
    let out_w = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);

    let attn = b
        .add_multi_head_cross_attention(
            decoder_hidden,
            encoder_output,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[DECODER_SEQ, EMBED_DIM],
        )
        .expect("valid full cross-attention");

    b.build(attn).expect("valid full cross-attention kernel")
}

/// Bindings for full cross-attention.
fn full_cross_attention_bindings() -> Vec<TensorParamBinding> {
    let kv_const = ArrayD::from_elem(IxDyn(&[ENCODER_SEQ, EMBED_DIM]), 0.1f32);
    let w_proj = ArrayD::from_elem(IxDyn(&[EMBED_DIM, EMBED_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable, // decoder_hidden [D_SEQ, D]
        TensorParamBinding::ConstantTensor(kv_const), // encoder_output [E_SEQ, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj), // out_weight [D, D]
    ]
}

// ---------------------------------------------------------------------------
// Builder helpers: Decoder block (self-attn + cross-attn + FFN)
// ---------------------------------------------------------------------------

/// Build a full Whisper decoder block with cross-attention.
///
/// Input: `[DECODER_SEQ, EMBED_DIM]` (Variable -- decoder hidden state).
/// Encoder output: `[ENCODER_SEQ, EMBED_DIM]` (Constant).
/// Output: `[DECODER_SEQ, EMBED_DIM]`.
///
/// Pre-norm structure with 3 residual connections:
/// 1. LN -> MHA(causal self-attn) -> + residual
/// 2. LN -> CrossMHA(encoder_output) -> + residual
/// 3. LN -> Linear(D, FFN_DIM) -> GELU -> Linear(FFN_DIM, D) -> + residual
fn build_decoder_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_xattn_dec_block");

    // Inputs
    let decoder_hidden = b.add_input("decoder_hidden", &[DECODER_SEQ, EMBED_DIM]);
    let encoder_output = b.add_input("encoder_output", &[ENCODER_SEQ, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);

    // Self-attention weights
    let sa_ln_w = b.add_input("sa_ln_weight", &[EMBED_DIM]);
    let sa_ln_b = b.add_input("sa_ln_bias", &[EMBED_DIM]);
    let sa_q_w = b.add_input("sa_q_weight", &[EMBED_DIM, EMBED_DIM]);
    let sa_k_w = b.add_input("sa_k_weight", &[EMBED_DIM, EMBED_DIM]);
    let sa_v_w = b.add_input("sa_v_weight", &[EMBED_DIM, EMBED_DIM]);
    let sa_out_w = b.add_input("sa_out_weight", &[EMBED_DIM, EMBED_DIM]);

    // Cross-attention weights
    let ca_ln_w = b.add_input("ca_ln_weight", &[EMBED_DIM]);
    let ca_ln_b = b.add_input("ca_ln_bias", &[EMBED_DIM]);
    let ca_q_w = b.add_input("ca_q_weight", &[EMBED_DIM, EMBED_DIM]);
    let ca_k_w = b.add_input("ca_k_weight", &[EMBED_DIM, EMBED_DIM]);
    let ca_v_w = b.add_input("ca_v_weight", &[EMBED_DIM, EMBED_DIM]);
    let ca_out_w = b.add_input("ca_out_weight", &[EMBED_DIM, EMBED_DIM]);

    // FFN weights
    let ffn_ln_w = b.add_input("ffn_ln_weight", &[EMBED_DIM]);
    let ffn_ln_b = b.add_input("ffn_ln_bias", &[EMBED_DIM]);
    let ffn1_w = b.add_input("ffn1_weight", &[FFN_DIM, EMBED_DIM]);
    let ffn2_w = b.add_input("ffn2_weight", &[EMBED_DIM, FFN_DIM]);

    let shape = [DECODER_SEQ, EMBED_DIM];
    let ffn_shape = [DECODER_SEQ, FFN_DIM];

    // --- Sub-block 1: Causal self-attention ---
    let sa_normed = b.add_layer_norm(decoder_hidden, eps, 1, sa_ln_w, sa_ln_b, &shape);
    let sa_out = b
        .add_multi_head_attention(
            sa_normed,
            sa_q_w,
            sa_k_w,
            sa_v_w,
            sa_out_w,
            NUM_HEADS,
            AttentionMask::Causal,
            &shape,
        )
        .expect("valid causal self-attention");
    let residual1 = b.add_binary_add(decoder_hidden, sa_out, &shape);

    // --- Sub-block 2: Cross-attention with encoder output ---
    let ca_normed = b.add_layer_norm(residual1, eps, 1, ca_ln_w, ca_ln_b, &shape);
    let ca_out = b
        .add_multi_head_cross_attention(
            ca_normed,
            encoder_output,
            ca_q_w,
            ca_k_w,
            ca_v_w,
            ca_out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid cross-attention");
    let residual2 = b.add_binary_add(residual1, ca_out, &shape);

    // --- Sub-block 3: FFN ---
    let ffn_normed = b.add_layer_norm(residual2, eps, 1, ffn_ln_w, ffn_ln_b, &shape);
    let ffn1 = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);
    let out = b.add_binary_add(residual2, ffn2, &shape);

    b.build(out).expect("valid decoder block kernel")
}

/// Bindings for decoder block.
fn decoder_block_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let kv_const = ArrayD::from_elem(IxDyn(&[ENCODER_SEQ, d]), 0.1f32);
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);

    vec![
        TensorParamBinding::Variable, // decoder_hidden [D_SEQ, D]
        TensorParamBinding::ConstantTensor(kv_const), // encoder_output [E_SEQ, D]
        TensorParamBinding::ConstantScalar(1e-5), // eps
        // Self-attention weights
        TensorParamBinding::ConstantTensor(ln_w.clone()), // sa_ln_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // sa_ln_bias
        TensorParamBinding::ConstantTensor(w_proj.clone()), // sa_q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // sa_k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // sa_v_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // sa_out_weight
        // Cross-attention weights
        TensorParamBinding::ConstantTensor(ln_w.clone()), // ca_ln_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // ca_ln_bias
        TensorParamBinding::ConstantTensor(w_proj.clone()), // ca_q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // ca_k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // ca_v_weight
        TensorParamBinding::ConstantTensor(w_proj),       // ca_out_weight
        // FFN weights
        TensorParamBinding::ConstantTensor(ln_w), // ffn_ln_weight
        TensorParamBinding::ConstantTensor(ln_b), // ffn_ln_bias
        TensorParamBinding::ConstantTensor(w_ffn1), // ffn1_weight
        TensorParamBinding::ConstantTensor(w_ffn2), // ffn2_weight
    ]
}

// ---------------------------------------------------------------------------
// Builder helpers: 2-block decoder stack
// ---------------------------------------------------------------------------

/// Build a 2-block decoder stack with shared encoder conditioning.
///
/// Input: `[DECODER_SEQ, EMBED_DIM]` (Variable -- decoder hidden state).
/// Encoder output: `[ENCODER_SEQ, EMBED_DIM]` (Constant -- shared across blocks).
/// Output: `[DECODER_SEQ, EMBED_DIM]`.
///
/// Architecture:
///   Block 0: LN -> SelfAttn(causal) -> + res -> LN -> CrossAttn(enc) -> + res -> LN -> FFN -> + res
///   Block 1: LN -> SelfAttn(causal) -> + res -> LN -> CrossAttn(enc) -> + res -> LN -> FFN -> + res
///
/// The same encoder output feeds both blocks' cross-attention layers.
/// This tests bounds stability through depth (2 chained decoder blocks).
fn build_2_block_decoder_stack_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_xattn_2block_stack");

    let decoder_hidden = b.add_input("decoder_hidden", &[DECODER_SEQ, EMBED_DIM]);
    let encoder_output = b.add_input("encoder_output", &[ENCODER_SEQ, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);

    let shape = [DECODER_SEQ, EMBED_DIM];
    let ffn_shape = [DECODER_SEQ, FFN_DIM];

    let mut current = decoder_hidden;

    for block_idx in 0..2 {
        let pfx = format!("b{block_idx}");

        // Self-attention weights
        let sa_ln_w = b.add_input(&format!("{pfx}_sa_ln_w"), &[EMBED_DIM]);
        let sa_ln_b = b.add_input(&format!("{pfx}_sa_ln_b"), &[EMBED_DIM]);
        let sa_q_w = b.add_input(&format!("{pfx}_sa_qw"), &[EMBED_DIM, EMBED_DIM]);
        let sa_k_w = b.add_input(&format!("{pfx}_sa_kw"), &[EMBED_DIM, EMBED_DIM]);
        let sa_v_w = b.add_input(&format!("{pfx}_sa_vw"), &[EMBED_DIM, EMBED_DIM]);
        let sa_out_w = b.add_input(&format!("{pfx}_sa_ow"), &[EMBED_DIM, EMBED_DIM]);

        // Cross-attention weights
        let ca_ln_w = b.add_input(&format!("{pfx}_ca_ln_w"), &[EMBED_DIM]);
        let ca_ln_b = b.add_input(&format!("{pfx}_ca_ln_b"), &[EMBED_DIM]);
        let ca_q_w = b.add_input(&format!("{pfx}_ca_qw"), &[EMBED_DIM, EMBED_DIM]);
        let ca_k_w = b.add_input(&format!("{pfx}_ca_kw"), &[EMBED_DIM, EMBED_DIM]);
        let ca_v_w = b.add_input(&format!("{pfx}_ca_vw"), &[EMBED_DIM, EMBED_DIM]);
        let ca_out_w = b.add_input(&format!("{pfx}_ca_ow"), &[EMBED_DIM, EMBED_DIM]);

        // FFN weights
        let ffn_ln_w = b.add_input(&format!("{pfx}_ffn_ln_w"), &[EMBED_DIM]);
        let ffn_ln_b = b.add_input(&format!("{pfx}_ffn_ln_b"), &[EMBED_DIM]);
        let ffn1_w = b.add_input(&format!("{pfx}_ffn1w"), &[FFN_DIM, EMBED_DIM]);
        let ffn2_w = b.add_input(&format!("{pfx}_ffn2w"), &[EMBED_DIM, FFN_DIM]);

        // Sub-block 1: Causal self-attention
        let sa_normed = b.add_layer_norm(current, eps, 1, sa_ln_w, sa_ln_b, &shape);
        let sa_out = b
            .add_multi_head_attention(
                sa_normed,
                sa_q_w,
                sa_k_w,
                sa_v_w,
                sa_out_w,
                NUM_HEADS,
                AttentionMask::Causal,
                &shape,
            )
            .expect("valid causal self-attention");
        let residual1 = b.add_binary_add(current, sa_out, &shape);

        // Sub-block 2: Cross-attention with encoder output
        let ca_normed = b.add_layer_norm(residual1, eps, 1, ca_ln_w, ca_ln_b, &shape);
        let ca_out = b
            .add_multi_head_cross_attention(
                ca_normed,
                encoder_output,
                ca_q_w,
                ca_k_w,
                ca_v_w,
                ca_out_w,
                NUM_HEADS,
                AttentionMask::Standard,
                &shape,
            )
            .expect("valid cross-attention");
        let residual2 = b.add_binary_add(residual1, ca_out, &shape);

        // Sub-block 3: FFN
        let ffn_normed = b.add_layer_norm(residual2, eps, 1, ffn_ln_w, ffn_ln_b, &shape);
        let ffn1 = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
        let act = b.add_gelu(ffn1, &ffn_shape);
        let ffn2 = b.add_linear(act, ffn2_w, None, &shape);
        current = b.add_binary_add(residual2, ffn2, &shape);
    }

    b.build(current)
        .expect("valid 2-block decoder stack kernel")
}

/// Bindings for 2-block decoder stack.
fn two_block_decoder_stack_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let kv_const = ArrayD::from_elem(IxDyn(&[ENCODER_SEQ, d]), 0.1f32);
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);

    let mut bindings = vec![
        TensorParamBinding::Variable, // decoder_hidden [D_SEQ, D]
        TensorParamBinding::ConstantTensor(kv_const), // encoder_output [E_SEQ, D]
        TensorParamBinding::ConstantScalar(1e-5), // eps
    ];

    // 2 blocks, each with: sa(ln_w, ln_b, q, k, v, out) + ca(ln_w, ln_b, q, k, v, out) + ffn(ln_w, ln_b, ffn1, ffn2)
    for _ in 0..2 {
        // Self-attention: ln_w, ln_b, q_w, k_w, v_w, out_w
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));

        // Cross-attention: ln_w, ln_b, q_w, k_w, v_w, out_w
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));

        // FFN: ln_w, ln_b, ffn1_w, ffn2_w
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn1.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn2.clone()));
    }

    bindings
}

// ===========================================================================
// Tests: Q projection
// ===========================================================================

/// Q projection TensorKernelDef validates.
#[test]
fn test_whisper_xattn_q_proj_def_validates() {
    let def = build_q_projection_kernel();
    def.validate().expect("Q projection kernel should validate");
}

/// IBP bounds propagate through Q projection.
///
/// Q projection is a linear operation (matmul with constant weight),
/// so IBP produces exact bounds.
#[test]
fn test_whisper_xattn_q_proj_ibp_propagates() {
    let def = build_q_projection_kernel();
    let bindings = q_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DECODER_SEQ, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Q projection");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[DECODER_SEQ, EMBED_DIM],
        "Q projection output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper cross-attn Q projection IBP: bounds=[{lo_min}, {hi_max}]");

    // Linear projection with small weights (0.02) on [-1, 1] input.
    // Output range: each element sums EMBED_DIM products of weight * input.
    // Max magnitude: EMBED_DIM * WEIGHT_MAG * 1.0 = 32 * 0.02 = 0.64.
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

// ===========================================================================
// Tests: K/V projection
// ===========================================================================

/// K/V projection TensorKernelDef validates.
#[test]
fn test_whisper_xattn_kv_proj_def_validates() {
    let def = build_kv_projection_kernel();
    def.validate()
        .expect("K/V projection kernel should validate");
}

/// IBP bounds propagate through K/V projection.
///
/// Two linear projections from the same input, concatenated. Both are
/// linear, so IBP produces exact bounds.
#[test]
fn test_whisper_xattn_kv_proj_ibp_propagates() {
    let def = build_kv_projection_kernel();
    let bindings = kv_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENCODER_SEQ, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through K/V projection");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[ENCODER_SEQ, EMBED_DIM * 2],
        "K/V projection output shape mismatch (K and V concatenated)"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper cross-attn K/V projection IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// Tests: Full cross-attention
// ===========================================================================

/// Full cross-attention TensorKernelDef validates.
#[test]
fn test_whisper_xattn_full_def_validates() {
    let def = build_full_cross_attention_kernel();
    def.validate()
        .expect("full cross-attention kernel should validate");
}

/// Full cross-attention graph builds with sufficient nodes.
#[test]
fn test_whisper_xattn_full_graph_builds() {
    let def = build_full_cross_attention_kernel();
    let bindings = full_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("full cross-attention graph should translate");

    // Q/K/V projections + reshape + transpose + attention +
    // transpose + reshape + output projection = many nodes.
    assert!(
        graph.num_nodes() >= 5,
        "full cross-attention graph should have >= 5 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through full cross-attention.
///
/// Key structural property: Q comes from the Variable decoder hidden state,
/// while K/V come from the Constant encoder output. The output shape
/// follows Q: [DECODER_SEQ, EMBED_DIM], not [ENCODER_SEQ, ...].
#[test]
fn test_whisper_xattn_full_ibp_propagates() {
    let def = build_full_cross_attention_kernel();
    let bindings = full_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DECODER_SEQ, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full cross-attention");

    // Output shape matches Q (decoder) sequence length, not KV (encoder).
    assert_eq!(
        output.lower_upper().0.shape(),
        &[DECODER_SEQ, EMBED_DIM],
        "output shape must be [DECODER_SEQ, EMBED_DIM], not [ENCODER_SEQ, EMBED_DIM]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper full cross-attention IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through full cross-attention.
///
/// Cross-attention with constant K/V should allow CROWN to produce tighter
/// bounds than IBP since the K/V branch has zero perturbation radius.
/// No LayerNorm here, so CROWN linearization is through softmax + GELU only.
#[test]
fn test_whisper_xattn_full_crown_propagation() {
    let def = build_full_cross_attention_kernel();
    let bindings = full_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DECODER_SEQ, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[DECODER_SEQ, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper full cross-attention: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Full cross-attention bounds width stays reasonable.
///
/// With small weights (0.02) and [-1, 1] input, bounds should not blow up.
/// Cross-attention with constant encoder output should produce manageable
/// bounds because K/V perturbation is zero.
#[test]
fn test_whisper_xattn_full_bounds_width() {
    let def = build_full_cross_attention_kernel();
    let bindings = full_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DECODER_SEQ, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full cross-attention");
    let (lo, hi) = output.lower_upper();

    let max_width = lo
        .iter()
        .zip(hi.iter())
        .map(|(l, u)| (u - l).abs())
        .fold(0.0f32, f32::max);

    // Small weights and bounded input should keep bounds manageable.
    // head_dim = 8, so attention dot products are modest.
    assert!(
        max_width < 500.0,
        "cross-attention IBP bounds max width {max_width} should be < 500.0 \
         (DECODER_SEQ={DECODER_SEQ}, EMBED_DIM={EMBED_DIM}, NUM_HEADS={NUM_HEADS}, HEAD_DIM={HEAD_DIM})"
    );
}

/// Verify and record full cross-attention under status key.
#[test]
fn test_whisper_xattn_full_verify_and_record() {
    let def = build_full_cross_attention_kernel();
    let bindings = full_cross_attention_bindings();
    let input = uniform_bounds(&[DECODER_SEQ, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_cross_attention_full");
    assert_eq!(
        result.num_variables, 1,
        "single Variable input (decoder_hidden)"
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[DECODER_SEQ, EMBED_DIM]);

    // No LayerNorm in pure cross-attention, so soundness depends on
    // whether softmax linearization is classified as sound or heuristic.
    eprintln!(
        "Full cross-attention soundness mode: {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Tests: Decoder block (self-attn + cross-attn + FFN)
// ===========================================================================

/// Decoder block TensorKernelDef validates.
#[test]
fn test_whisper_xattn_dec_block_def_validates() {
    let def = build_decoder_block_kernel();
    def.validate()
        .expect("decoder block kernel should validate");
}

/// Decoder block graph builds with sufficient complexity.
#[test]
fn test_whisper_xattn_dec_block_graph_builds() {
    let def = build_decoder_block_kernel();
    let bindings = decoder_block_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("decoder block graph should translate");

    // Self-attn: LN + MHA + residual (~12 nodes)
    // Cross-attn: LN + CrossMHA + residual (~12 nodes)
    // FFN: LN + Linear + GELU + Linear + residual (~6 nodes)
    // Total: at least 25 nodes
    assert!(
        graph.num_nodes() >= 25,
        "decoder block should have >= 25 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the decoder block.
#[test]
fn test_whisper_xattn_dec_block_ibp_propagates() {
    let def = build_decoder_block_kernel();
    let bindings = decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DECODER_SEQ, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder block");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[DECODER_SEQ, EMBED_DIM],
        "decoder block output shape must be [DECODER_SEQ, EMBED_DIM]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper decoder block IBP: bounds=[{lo_min}, {hi_max}]");

    // Bounds may be wide due to 3 chained LayerNorms with residuals.
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through the decoder block.
///
/// Contains 3 LayerNorms, so CROWN uses heuristic linearization
/// (IbpValidated mode). CROWN may fall back to IBP due to the depth
/// of normalization layers.
#[test]
fn test_whisper_xattn_dec_block_crown_propagation() {
    let def = build_decoder_block_kernel();
    let bindings = decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DECODER_SEQ, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[DECODER_SEQ, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper decoder block: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record decoder block under status key.
#[test]
fn test_whisper_xattn_dec_block_verify_and_record() {
    let def = build_decoder_block_kernel();
    let bindings = decoder_block_bindings();
    let input = uniform_bounds(&[DECODER_SEQ, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_cross_attention_dec_block");
    assert_eq!(
        result.num_variables, 1,
        "single Variable input (decoder_hidden)"
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[DECODER_SEQ, EMBED_DIM]);

    // 3 LayerNorms use heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "Decoder block with LayerNorms should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Tests: 2-block decoder stack
// ===========================================================================

/// 2-block decoder stack TensorKernelDef validates.
#[test]
fn test_whisper_xattn_2block_stack_def_validates() {
    let def = build_2_block_decoder_stack_kernel();
    def.validate()
        .expect("2-block decoder stack kernel should validate");
}

/// 2-block decoder stack graph builds with substantial node count.
#[test]
fn test_whisper_xattn_2block_stack_graph_builds() {
    let def = build_2_block_decoder_stack_kernel();
    let bindings = two_block_decoder_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("2-block decoder stack graph should translate");

    // 2 blocks x (self-attn + cross-attn + FFN + LayerNorms + residuals) >= 50 nodes
    assert!(
        graph.num_nodes() >= 50,
        "2-block decoder stack should have >= 50 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the 2-block decoder stack.
///
/// Tests bounds stability through depth: 2 sequential decoder blocks
/// with shared encoder conditioning. Bounds should remain finite even
/// though 6 LayerNorms + 2 attention layers are chained.
#[test]
fn test_whisper_xattn_2block_stack_ibp_propagates() {
    let def = build_2_block_decoder_stack_kernel();
    let bindings = two_block_decoder_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DECODER_SEQ, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-block decoder stack");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[DECODER_SEQ, EMBED_DIM],
        "2-block stack output shape must be [DECODER_SEQ, EMBED_DIM]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper 2-block decoder stack IBP: bounds=[{lo_min}, {hi_max}]");

    // 6 LayerNorms + residuals may widen bounds significantly.
    // Primary invariant: finiteness and non-degeneracy.
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through the 2-block decoder stack.
///
/// 6 LayerNorms across 2 blocks use heuristic CROWN linearization.
/// CROWN may fall back to IBP due to the accumulated normalization depth.
#[test]
fn test_whisper_xattn_2block_stack_crown_propagation() {
    let def = build_2_block_decoder_stack_kernel();
    let bindings = two_block_decoder_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DECODER_SEQ, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[DECODER_SEQ, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper 2-block decoder stack: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record 2-block decoder stack under status key.
#[test]
fn test_whisper_xattn_2block_stack_verify_and_record() {
    let def = build_2_block_decoder_stack_kernel();
    let bindings = two_block_decoder_stack_bindings();
    let input = uniform_bounds(&[DECODER_SEQ, EMBED_DIM], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "whisper_cross_attention_2block_stack",
    );
    assert_eq!(
        result.num_variables, 1,
        "single Variable input (decoder_hidden)"
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[DECODER_SEQ, EMBED_DIM]);

    // 6 LayerNorms use heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "2-block stack with LayerNorms should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}
