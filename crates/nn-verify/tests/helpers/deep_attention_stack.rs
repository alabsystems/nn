// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for deep (N-layer) attention + FFN stacking.
//!
//! Phase 25 of #1729: extends Phase 24's 2-layer pipeline to N layers,
//! testing whether attention monotonicity degrades proportionally (graceful)
//! or catastrophically (collapse) with depth.
//!
//! Architecture per layer:
//! ```text
//! Layer i: Q_i = (prev + PE) @ W_qi,  K_i = enc_k @ W_ki
//!          Scores_i = Q_i @ K_i^T / √d_k + mask → Softmax → W_i [H, T_dec, T_enc]
//!          Ctx_i = W_i @ V_i → Transpose → Reshape → O_proj → [T_dec, D]
//!          Res_i = prev + Ctx_i
//!          FFN_i: LayerNorm(Res_i) → Linear(up) → GELU → Linear(down)
//!          Out_i = Res_i + FFN_i(Res_i)
//! ```
//!
//! The final layer outputs attention weights (not context) so we can extract
//! a certificate. Intermediate layers output context + residual + FFN.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 25.

// Helpers shared across test binaries; not all functions used by all binaries.
#![allow(dead_code, clippy::duplicated_attributes)]

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_verify::{BoundedTensor, TensorParamBinding};

// Causal mask — delegated to super::common (Part of #1970).
pub(super) use super::common::build_strict_causal_mask;

// Sinusoidal PE — delegated to super::common (Part of #1970).
use super::common::sinusoidal_pe_interleaved as sinusoidal_pe;

// Weight constructors — delegated to super::common::weights (Part of #1938).
use super::common::weights;

// ---------------------------------------------------------------------------
// Deep stack graph builder
// ---------------------------------------------------------------------------

/// Per-layer input node set for the deep stack builder.
struct LayerInputs {
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

/// Build an N-layer attention + FFN stack.
///
/// Layers 0..N-2 compute full attention (Q→K→V→context→output→residual→FFN).
/// Layer N-1 computes attention weights only (for certificate extraction).
///
/// Returns the TensorKernelDef where the output is the final layer's
/// attention weights `[H, T_dec, T_enc]`.
pub(super) fn build_deep_attention_stack(
    num_layers: usize,
    t_dec: usize,
    t_enc: usize,
    d_model: usize,
    num_heads: usize,
    ffn_dim: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    assert!(num_layers >= 2, "deep stack needs at least 2 layers");

    let d_k = d_model / num_heads;
    let scale = 1.0 / (d_k as f32).sqrt();
    let scores_shape = [num_heads, t_dec, t_enc];
    let ctx_shape = [num_heads, t_dec, d_k];

    let mut b = TensorBlockBuilder::new("deep_attention_stack");

    // Global inputs
    let hidden = b.add_input("hidden", &[t_dec, d_model]);
    let dec_pe = b.add_input("dec_pe", &[t_dec, d_model]);
    let enc_k_input = b.add_input("enc_k", &[t_enc, d_model]);
    let enc_v_input = b.add_input("enc_v", &[t_enc, d_model]);

    // Per-layer inputs
    let mut layer_inputs = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let suffix = format!("_L{i}");
        layer_inputs.push(LayerInputs {
            w_q: b.add_input(&format!("w_q{suffix}"), &[d_model, d_model]),
            w_k: b.add_input(&format!("w_k{suffix}"), &[d_model, d_model]),
            w_v: b.add_input(&format!("w_v{suffix}"), &[d_model, d_model]),
            w_o: b.add_input(&format!("w_o{suffix}"), &[d_model, d_model]),
            mask: b.add_input(&format!("mask{suffix}"), &[t_dec, t_enc]),
            ln_weight: b.add_input(&format!("ln_w{suffix}"), &[d_model]),
            ln_bias: b.add_input(&format!("ln_b{suffix}"), &[d_model]),
            ln_eps: b.add_input(&format!("ln_eps{suffix}"), &[1]),
            ffn_up: b.add_input(&format!("ffn_up{suffix}"), &[ffn_dim, d_model]),
            ffn_down: b.add_input(&format!("ffn_down{suffix}"), &[d_model, ffn_dim]),
        });
    }

    // Layer 0: add PE to hidden
    let mut prev = b.add_binary_add(hidden, dec_pe, &[t_dec, d_model]);

    for (layer_idx, li) in layer_inputs.iter().enumerate().take(num_layers) {
        let is_last = layer_idx == num_layers - 1;

        // Q = prev @ W_q
        let q = b.add_matmul(prev, li.w_q, false, None, &[t_dec, d_model]);
        // K = enc_k @ W_k
        let k = b.add_matmul(enc_k_input, li.w_k, false, None, &[t_enc, d_model]);

        // Multi-head reshape + transpose
        let q_r = b.add_reshape(q, &[t_dec, num_heads, d_k]);
        let k_r = b.add_reshape(k, &[t_enc, num_heads, d_k]);
        let q_t = b.add_transpose(q_r, &[1, 0, 2], &[num_heads, t_dec, d_k]);
        let k_t = b.add_transpose(k_r, &[1, 0, 2], &[num_heads, t_enc, d_k]);

        // Scores = Q @ K^T / √d_k + mask → Softmax
        let scores = b.add_matmul(q_t, k_t, true, Some(scale), &scores_shape);
        let mask_bc = b.add_broadcast(li.mask, &scores_shape);
        let masked = b.add_binary_add(scores, mask_bc, &scores_shape);
        let weights = b.add_softmax(masked, -1, &scores_shape);

        if is_last {
            // Final layer: output attention weights for certificate extraction
            return b.build(weights).expect("valid deep attention stack graph");
        }

        // V = enc_v @ W_v
        let v = b.add_matmul(enc_v_input, li.w_v, false, None, &[t_enc, d_model]);
        let v_r = b.add_reshape(v, &[t_enc, num_heads, d_k]);
        let v_t = b.add_transpose(v_r, &[1, 0, 2], &[num_heads, t_enc, d_k]);

        // Context = W @ V: [H, T_dec, T_enc] @ [H, T_enc, d_k] → [H, T_dec, d_k]
        let ctx = b.add_matmul(weights, v_t, false, None, &ctx_shape);

        // Transpose back: [H, T_dec, d_k] → [T_dec, H, d_k]
        let ctx_t = b.add_transpose(ctx, &[1, 0, 2], &[t_dec, num_heads, d_k]);
        let ctx_flat = b.add_reshape(ctx_t, &[t_dec, d_model]);

        // Output projection: O = ctx_flat @ W_o
        let attn_out = b.add_matmul(ctx_flat, li.w_o, false, None, &[t_dec, d_model]);

        // Residual + LayerNorm + FFN
        let res = b.add_binary_add(prev, attn_out, &[t_dec, d_model]);
        let normed = b.add_layer_norm(
            res,
            li.ln_eps,
            1,
            li.ln_weight,
            li.ln_bias,
            &[t_dec, d_model],
        );
        let ffn1 = b.add_linear(normed, li.ffn_up, None, &[t_dec, ffn_dim]);
        let act = b.add_gelu(ffn1, &[t_dec, ffn_dim]);
        let ffn2 = b.add_linear(act, li.ffn_down, None, &[t_dec, d_model]);
        let ffn_res = b.add_binary_add(res, ffn2, &[t_dec, d_model]);

        prev = ffn_res;
    }

    unreachable!("loop exits via `return` on last layer");
}

// ---------------------------------------------------------------------------
// Binding constructors
// ---------------------------------------------------------------------------

/// Build bindings for an N-layer deep stack.
pub(super) fn deep_stack_bindings(
    num_layers: usize,
    t_dec: usize,
    t_enc: usize,
    d_model: usize,
    num_heads: usize,
    ffn_dim: usize,
    pe_scale: f32,
    w_perturbation: f32,
) -> Vec<TensorParamBinding> {
    let dec_pe = {
        let mut pe = sinusoidal_pe(t_dec, d_model, num_heads);
        pe.mapv_inplace(|v| v * pe_scale);
        pe
    };
    let enc_k = weights::encoder_k(t_enc, d_model);
    let enc_v = weights::encoder_k(t_enc, d_model); // Same structure as K
    let w_proj = weights::near_identity(d_model, w_perturbation);
    let mask = build_strict_causal_mask(t_dec, t_enc);
    let ln_w = weights::norm_weight(d_model);
    let ln_b = weights::norm_bias(d_model);
    let ffn_up_w = weights::ffn_weight(ffn_dim, d_model, 0.1);
    let ffn_down_w = weights::ffn_weight(d_model, ffn_dim, 0.1);

    let mut bindings = vec![
        TensorParamBinding::Variable,               // hidden
        TensorParamBinding::ConstantTensor(dec_pe), // dec_pe
        TensorParamBinding::ConstantTensor(enc_k),  // enc_k
        TensorParamBinding::ConstantTensor(enc_v),  // enc_v
    ];

    // Per-layer bindings: w_q, w_k, w_v, w_o, mask, ln_w, ln_b, ln_eps, ffn_up, ffn_down
    for _ in 0..num_layers {
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

    bindings
}

// ---------------------------------------------------------------------------
// Depth degradation analysis
// ---------------------------------------------------------------------------

/// Margin result for one depth configuration.
#[derive(Debug)]
pub(super) struct DepthResult {
    pub(super) num_layers: usize,
    pub(super) per_head_min_margin: Vec<f64>,
    pub(super) min_margin: f64,
    pub(super) is_proven: bool,
    pub(super) proven_heads: usize,
    pub(super) per_head_target_weight_lo: Vec<f64>,
    pub(super) graph_nodes: usize,
}

/// Extract a depth result from output bounds.
pub(super) fn extract_depth_result(
    output: &BoundedTensor,
    num_layers: usize,
    num_heads: usize,
    t_dec: usize,
    t_enc: usize,
    graph_nodes: usize,
    alignment_fn: impl Fn(usize) -> usize,
) -> DepthResult {
    let (lo, hi) = output.lower_upper();
    let flat_lo: Vec<f32> = lo.iter().copied().collect();
    let flat_hi: Vec<f32> = hi.iter().copied().collect();

    let head_stride = t_dec * t_enc;
    let mut per_head_min_margin = Vec::with_capacity(num_heads);
    let mut per_head_target_weight_lo = Vec::with_capacity(num_heads);
    let mut proven_heads = 0;

    for h in 0..num_heads {
        let head_offset = h * head_stride;
        let mut head_min_margin = f64::INFINITY;
        let mut head_min_target_lo = f64::INFINITY;

        for t in 0..t_dec {
            let target = alignment_fn(t);
            if target >= t_enc {
                continue;
            }

            let max_visible = target;
            if t_enc > 1 && target == t_enc - 1 && max_visible == t_enc - 1 {
                continue;
            }

            let target_lo = f64::from(flat_lo[head_offset + t * t_enc + target]);
            if target_lo < head_min_target_lo {
                head_min_target_lo = target_lo;
            }

            let mut max_other_hi = f64::NEG_INFINITY;
            for j in 0..=max_visible {
                if j != target {
                    let upper = f64::from(flat_hi[head_offset + t * t_enc + j]);
                    if upper > max_other_hi {
                        max_other_hi = upper;
                    }
                }
            }

            let margin = if max_visible == 0 {
                f64::INFINITY
            } else {
                target_lo - max_other_hi
            };

            if margin < head_min_margin {
                head_min_margin = margin;
            }
        }

        if head_min_margin > 0.0 {
            proven_heads += 1;
        }
        per_head_min_margin.push(head_min_margin);
        per_head_target_weight_lo.push(head_min_target_lo);
    }

    let min_margin = per_head_min_margin
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    DepthResult {
        num_layers,
        per_head_min_margin,
        min_margin,
        is_proven: min_margin > 0.0,
        proven_heads,
        per_head_target_weight_lo,
        graph_nodes,
    }
}
