// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for multi-layer attention + FFN composition.
//!
//! Phase 24 of #1729: proves that attention monotonicity survives through
//! a full transformer block (attention → value projection → FFN + residual
//! → LayerNorm → next attention layer).
//!
//! Architecture:
//! ```text
//! Layer 1:  Q @ K^T / √d_k + mask → Softmax → W × V → O_proj → Attn_out [T_dec, D]
//! Residual: Attn_out + hidden → R1 [T_dec, D]
//! FFN:      LayerNorm(R1) → Linear(up) → GELU → Linear(down) → FFN_out [T_dec, D]
//! Residual: FFN_out + R1 → R2 [T_dec, D]
//! Layer 2:  R2 as Q → R2 @ K^T / √d_k + mask → Softmax → W2 [H, T_dec, T_enc]
//! ```
//!
//! Key insight: the residual connection is critical — it preserves position
//! information from the input through the FFN, even if the FFN introduces
//! some mixing. The LayerNorm re-centers but preserves relative differences.
//! If the original positional encoding is strong enough, position information
//! survives the FFN+residual+LN path and Layer 2 attention remains monotonic.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 24.

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{BoundedTensor, TensorParamBinding};

// Causal mask + propagation — delegated to super::common (Part of #1970).
pub(super) use super::common::{
    build_strict_causal_mask, graph_propagate, strict_causal_alignment,
};

// Sinusoidal PE — delegated to super::common (Part of #1970).
use super::common::sinusoidal_pe_interleaved as sinusoidal_pe;

// Weight constructors — delegated to super::common::weights (Part of #1938).
use super::common::weights;

// ---------------------------------------------------------------------------
// Graph builders — full attention + FFN + attention pipeline
// ---------------------------------------------------------------------------

/// Build a 2-layer attention pipeline with FFN between layers.
///
/// The full pipeline:
/// 1. **Attention Layer 1**: Q @ K^T / √d_k + mask → Softmax → W × V → O_proj
/// 2. **Residual + LayerNorm → FFN (Linear → GELU → Linear) + Residual**
/// 3. **Attention Layer 2**: result as Q → new scores → Softmax → W2
///
/// Inputs (in binding order):
/// - `hidden` (Variable): `[T_dec, D]`
/// - `dec_pe`: `[T_dec, D]`
/// - `enc_k`: `[T_enc, D]` (key embeddings for both layers)
/// - `enc_v`: `[T_enc, D]` (value embeddings for Layer 1)
/// - `w_q1`, `w_k1`, `w_v1`, `w_o1`: `[D, D]` (Layer 1 projections)
/// - `mask`: `[T_dec, T_enc]`
/// - `ln_weight`, `ln_bias`: `[D]`
/// - `ln_eps`: `[1]`
/// - `ffn_up`: `[ffn_dim, D]`
/// - `ffn_down`: `[D, ffn_dim]`
/// - `w_q2`, `w_k2`: `[D, D]` (Layer 2 Q/K projections)
/// - `mask2`: `[T_dec, T_enc]`
///
/// Output: Layer 2 attention weights `[H, T_dec, T_enc]`.
pub(super) fn build_two_layer_attention_ffn(
    t_dec: usize,
    t_enc: usize,
    d_model: usize,
    num_heads: usize,
    ffn_dim: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let d_k = d_model / num_heads;
    let mut b = TensorBlockBuilder::new("two_layer_attn_ffn");

    // --- Inputs ---
    let hidden = b.add_input("hidden", &[t_dec, d_model]);
    let dec_pe = b.add_input("dec_pe", &[t_dec, d_model]);
    let enc_k = b.add_input("enc_k", &[t_enc, d_model]);
    let enc_v = b.add_input("enc_v", &[t_enc, d_model]);
    let w_q1 = b.add_input("w_q1", &[d_model, d_model]);
    let w_k1 = b.add_input("w_k1", &[d_model, d_model]);
    let w_v1 = b.add_input("w_v1", &[d_model, d_model]);
    let w_o1 = b.add_input("w_o1", &[d_model, d_model]);
    let mask1 = b.add_input("mask1", &[t_dec, t_enc]);
    let ln_weight = b.add_input("ln_weight", &[d_model]);
    let ln_bias = b.add_input("ln_bias", &[d_model]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ffn_up = b.add_input("ffn_up", &[ffn_dim, d_model]);
    let ffn_down = b.add_input("ffn_down", &[d_model, ffn_dim]);
    let w_q2 = b.add_input("w_q2", &[d_model, d_model]);
    let w_k2 = b.add_input("w_k2", &[d_model, d_model]);
    let mask2 = b.add_input("mask2", &[t_dec, t_enc]);

    // === LAYER 1: Full attention with value projection ===

    // Q1 = (hidden + PE) @ W_q1
    let q1_in = b.add_binary_add(hidden, dec_pe, &[t_dec, d_model]);
    let q1 = b.add_matmul(q1_in, w_q1, false, None, &[t_dec, d_model]);

    // K1 = enc_k @ W_k1
    let k1 = b.add_matmul(enc_k, w_k1, false, None, &[t_enc, d_model]);

    // V1 = enc_v @ W_v1
    let v1 = b.add_matmul(enc_v, w_v1, false, None, &[t_enc, d_model]);

    // Multi-head reshape + transpose
    let q1_r = b.add_reshape(q1, &[t_dec, num_heads, d_k]);
    let k1_r = b.add_reshape(k1, &[t_enc, num_heads, d_k]);
    let v1_r = b.add_reshape(v1, &[t_enc, num_heads, d_k]);
    let q1_t = b.add_transpose(q1_r, &[1, 0, 2], &[num_heads, t_dec, d_k]);
    let k1_t = b.add_transpose(k1_r, &[1, 0, 2], &[num_heads, t_enc, d_k]);
    let v1_t = b.add_transpose(v1_r, &[1, 0, 2], &[num_heads, t_enc, d_k]);

    // Scores1 = Q1 @ K1^T / √d_k + mask
    let scale = 1.0 / (d_k as f32).sqrt();
    let scores_shape = [num_heads, t_dec, t_enc];
    let scores1 = b.add_matmul(q1_t, k1_t, true, Some(scale), &scores_shape);
    let mask1_bc = b.add_broadcast(mask1, &scores_shape);
    let masked1 = b.add_binary_add(scores1, mask1_bc, &scores_shape);

    // W1 = softmax(Scores1)
    let w1 = b.add_softmax(masked1, -1, &scores_shape);

    // Context1 = W1 @ V1: [H, T_dec, T_enc] @ [H, T_enc, d_k] → [H, T_dec, d_k]
    let ctx_shape = [num_heads, t_dec, d_k];
    let ctx1 = b.add_matmul(w1, v1_t, false, None, &ctx_shape);

    // Transpose back: [H, T_dec, d_k] → [T_dec, H, d_k]
    let ctx1_t = b.add_transpose(ctx1, &[1, 0, 2], &[t_dec, num_heads, d_k]);
    // Reshape: [T_dec, H, d_k] → [T_dec, D]
    let ctx1_flat = b.add_reshape(ctx1_t, &[t_dec, d_model]);

    // Output projection: O = ctx1_flat @ W_o1
    let attn_out = b.add_matmul(ctx1_flat, w_o1, false, None, &[t_dec, d_model]);

    // === RESIDUAL + LAYERNORM + FFN ===

    // Residual 1: hidden + attn_out
    let res1 = b.add_binary_add(hidden, attn_out, &[t_dec, d_model]);

    // LayerNorm(res1)
    let normed = b.add_layer_norm(res1, ln_eps, 1, ln_weight, ln_bias, &[t_dec, d_model]);

    // FFN: Linear(up) → GELU → Linear(down)
    let ffn1 = b.add_linear(normed, ffn_up, None, &[t_dec, ffn_dim]);
    let act = b.add_gelu(ffn1, &[t_dec, ffn_dim]);
    let ffn2 = b.add_linear(act, ffn_down, None, &[t_dec, d_model]);

    // Residual 2: res1 + ffn2
    let res2 = b.add_binary_add(res1, ffn2, &[t_dec, d_model]);

    // === LAYER 2: Attention on FFN output ===
    // Q2 = res2 @ W_q2 (no PE addition — positional info is in res2)
    let q2 = b.add_matmul(res2, w_q2, false, None, &[t_dec, d_model]);

    // K2 = enc_k @ W_k2 (re-uses same encoder keys)
    let k2 = b.add_matmul(enc_k, w_k2, false, None, &[t_enc, d_model]);

    // Multi-head reshape + transpose
    let q2_r = b.add_reshape(q2, &[t_dec, num_heads, d_k]);
    let k2_r = b.add_reshape(k2, &[t_enc, num_heads, d_k]);
    let q2_t = b.add_transpose(q2_r, &[1, 0, 2], &[num_heads, t_dec, d_k]);
    let k2_t = b.add_transpose(k2_r, &[1, 0, 2], &[num_heads, t_enc, d_k]);

    // Scores2 = Q2 @ K2^T / √d_k + mask2
    let scores2 = b.add_matmul(q2_t, k2_t, true, Some(scale), &scores_shape);
    let mask2_bc = b.add_broadcast(mask2, &scores_shape);
    let masked2 = b.add_binary_add(scores2, mask2_bc, &scores_shape);

    // W2 = softmax(Scores2) — Layer 2 attention weights
    let w2 = b.add_softmax(masked2, -1, &scores_shape);

    b.build(w2).expect("valid two-layer attention FFN graph")
}

/// Build a simplified FFN-only pipeline: input → LayerNorm → FFN → residual → attention.
///
/// Simpler version that isolates the FFN survival property without the
/// first attention layer's complexity. Tests whether position info in
/// the input survives FFN+residual to produce monotonic attention.
///
/// Inputs:
/// - `hidden` (Variable): `[T_dec, D]`
/// - `enc_k`: `[T_enc, D]`
/// - `ln_weight`, `ln_bias`: `[D]`, `ln_eps`: `[1]`
/// - `ffn_up`: `[ffn_dim, D]`, `ffn_down`: `[D, ffn_dim]`
/// - `w_q`, `w_k`: `[D, D]`
/// - `mask`: `[T_dec, T_enc]`
///
/// Output: attention weights `[H, T_dec, T_enc]`.
pub(super) fn build_ffn_to_attention(
    t_dec: usize,
    t_enc: usize,
    d_model: usize,
    num_heads: usize,
    ffn_dim: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let d_k = d_model / num_heads;
    let mut b = TensorBlockBuilder::new("ffn_to_attention");

    let hidden = b.add_input("hidden", &[t_dec, d_model]);
    let enc_k = b.add_input("enc_k", &[t_enc, d_model]);
    let ln_weight = b.add_input("ln_weight", &[d_model]);
    let ln_bias = b.add_input("ln_bias", &[d_model]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ffn_up = b.add_input("ffn_up", &[ffn_dim, d_model]);
    let ffn_down = b.add_input("ffn_down", &[d_model, ffn_dim]);
    let w_q = b.add_input("w_q", &[d_model, d_model]);
    let w_k = b.add_input("w_k", &[d_model, d_model]);
    let mask = b.add_input("mask", &[t_dec, t_enc]);

    // LayerNorm(hidden) → FFN → residual
    let normed = b.add_layer_norm(hidden, ln_eps, 1, ln_weight, ln_bias, &[t_dec, d_model]);
    let ffn1 = b.add_linear(normed, ffn_up, None, &[t_dec, ffn_dim]);
    let act = b.add_gelu(ffn1, &[t_dec, ffn_dim]);
    let ffn2 = b.add_linear(act, ffn_down, None, &[t_dec, d_model]);
    let res = b.add_binary_add(hidden, ffn2, &[t_dec, d_model]);

    // Attention on FFN output
    let q = b.add_matmul(res, w_q, false, None, &[t_dec, d_model]);
    let k = b.add_matmul(enc_k, w_k, false, None, &[t_enc, d_model]);

    let q_r = b.add_reshape(q, &[t_dec, num_heads, d_k]);
    let k_r = b.add_reshape(k, &[t_enc, num_heads, d_k]);
    let q_t = b.add_transpose(q_r, &[1, 0, 2], &[num_heads, t_dec, d_k]);
    let k_t = b.add_transpose(k_r, &[1, 0, 2], &[num_heads, t_enc, d_k]);

    let scale = 1.0 / (d_k as f32).sqrt();
    let scores_shape = [num_heads, t_dec, t_enc];
    let scores = b.add_matmul(q_t, k_t, true, Some(scale), &scores_shape);
    let mask_bc = b.add_broadcast(mask, &scores_shape);
    let masked = b.add_binary_add(scores, mask_bc, &scores_shape);
    let weights = b.add_softmax(masked, -1, &scores_shape);

    b.build(weights).expect("valid FFN-to-attention graph")
}

// ---------------------------------------------------------------------------
// Binding constructors
// ---------------------------------------------------------------------------

/// Bindings for the full 2-layer attention + FFN pipeline.
pub(super) fn two_layer_bindings(
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
    let enc_v = weights::encoder_k(t_enc, d_model);
    let w_proj = weights::near_identity(d_model, w_perturbation);
    let mask = build_strict_causal_mask(t_dec, t_enc);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantTensor(dec_pe),         // dec_pe
        TensorParamBinding::ConstantTensor(enc_k),          // enc_k
        TensorParamBinding::ConstantTensor(enc_v),          // enc_v
        TensorParamBinding::ConstantTensor(w_proj.clone()), // w_q1
        TensorParamBinding::ConstantTensor(w_proj.clone()), // w_k1
        TensorParamBinding::ConstantTensor(w_proj.clone()), // w_v1
        TensorParamBinding::ConstantTensor(w_proj.clone()), // w_o1
        TensorParamBinding::ConstantTensor(mask.clone()),   // mask1
        TensorParamBinding::ConstantTensor(weights::norm_weight(d_model)), // ln_weight
        TensorParamBinding::ConstantTensor(weights::norm_bias(d_model)), // ln_bias
        TensorParamBinding::ConstantScalar(1e-5),           // ln_eps
        TensorParamBinding::ConstantTensor(weights::ffn_weight(ffn_dim, d_model, 0.1)), // ffn_up
        TensorParamBinding::ConstantTensor(weights::ffn_weight(d_model, ffn_dim, 0.1)), // ffn_down
        TensorParamBinding::ConstantTensor(w_proj.clone()), // w_q2
        TensorParamBinding::ConstantTensor(w_proj),         // w_k2
        TensorParamBinding::ConstantTensor(mask),           // mask2
    ]
}

/// Bindings for the FFN-to-attention pipeline.
pub(super) fn ffn_to_attention_bindings(
    t_enc: usize,
    d_model: usize,
    ffn_dim: usize,
    w_perturbation: f32,
) -> Vec<TensorParamBinding> {
    let enc_k = weights::encoder_k(t_enc, d_model);
    let w_proj = weights::near_identity(d_model, w_perturbation);
    let mask = build_strict_causal_mask(8, t_enc); // default T_dec=8

    vec![
        TensorParamBinding::Variable,              // hidden
        TensorParamBinding::ConstantTensor(enc_k), // enc_k
        TensorParamBinding::ConstantTensor(weights::norm_weight(d_model)), // ln_weight
        TensorParamBinding::ConstantTensor(weights::norm_bias(d_model)), // ln_bias
        TensorParamBinding::ConstantScalar(1e-5),  // ln_eps
        TensorParamBinding::ConstantTensor(weights::ffn_weight(ffn_dim, d_model, 0.1)), // ffn_up
        TensorParamBinding::ConstantTensor(weights::ffn_weight(d_model, ffn_dim, 0.1)), // ffn_down
        TensorParamBinding::ConstantTensor(w_proj.clone()), // w_q
        TensorParamBinding::ConstantTensor(w_proj), // w_k
        TensorParamBinding::ConstantTensor(mask),  // mask
    ]
}

// ---------------------------------------------------------------------------
// Certificate extraction
// ---------------------------------------------------------------------------

/// Certificate for Layer 2 attention monotonicity after FFN transformation.
#[derive(Debug)]
pub(super) struct ComposedAttentionCertificate {
    pub(super) num_heads: usize,
    pub(super) decoder_steps: usize,
    pub(super) encoder_positions: usize,
    /// Per-head minimum weight margin at Layer 2.
    pub(super) per_head_min_margin: Vec<f64>,
    /// Overall minimum margin across all heads.
    pub(super) min_margin: f64,
    /// Whether monotonicity is proven at Layer 2.
    pub(super) is_proven: bool,
    /// Number of heads with proven weight dominance.
    pub(super) proven_heads: usize,
    /// Per-head target weight lower bounds at Layer 2.
    pub(super) per_head_target_weight_lo: Vec<f64>,
    pub(super) input_bound: f64,
    pub(super) propagation_mode: String,
}

/// Extract a composed attention certificate from Layer 2 weight bounds.
///
/// Same logic as Phase 23's `extract_softmax_certificate` but wrapped
/// in a distinct type that makes it clear the bounds survived FFN.
pub(super) fn extract_composed_certificate(
    output: &BoundedTensor,
    num_heads: usize,
    t_dec: usize,
    t_enc: usize,
    input_bound: f64,
    mode: &str,
    alignment_fn: impl Fn(usize) -> usize,
) -> ComposedAttentionCertificate {
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

            // Skip saturated rows where target is last position.
            let max_visible = target;
            if t_enc > 1 && target == t_enc - 1 && max_visible == t_enc - 1 {
                continue;
            }

            let target_lo = f64::from(flat_lo[head_offset + t * t_enc + target]);
            if target_lo < head_min_target_lo {
                head_min_target_lo = target_lo;
            }

            // Max upper bound among other unmasked positions.
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

    ComposedAttentionCertificate {
        num_heads,
        decoder_steps: t_dec,
        encoder_positions: t_enc,
        per_head_min_margin,
        min_margin,
        is_proven: min_margin > 0.0,
        proven_heads,
        per_head_target_weight_lo,
        input_bound,
        propagation_mode: mode.to_string(),
    }
}
