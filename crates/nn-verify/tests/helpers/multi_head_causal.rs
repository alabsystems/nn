// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for multi-head causal cross-attention.
//!
//! Phase 22 of #1729: extends Phase 21's causal cross-attention to multi-head
//! attention, where H independent attention heads each compute their own
//! score matrix with a shared causal mask.
//!
//! Architecture:
//! ```text
//! Q: [T_dec, D] → W_q [D, D] → Reshape [T_dec, H, d_k] → Transpose [H, T_dec, d_k]
//! K: [T_enc, D] → W_k [D, D] → Reshape [T_enc, H, d_k] → Transpose [H, T_enc, d_k]
//! Scores = Q_proj @ K_proj^T / √d_k → [H, T_dec, T_enc]
//! Mask: [T_dec, T_enc] → Broadcast [H, T_dec, T_enc]
//! S_masked = Scores + Mask_broadcast
//! ```
//!
//! Key insight: each head independently preserves the causal alignment property.
//! Near-identity W_q and W_k maintain the PE diagonal dominance per-head.
//! The certificate checks per-head alignment dominance and reports the minimum
//! margin across all heads (worst-case head determines provability).
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 22.

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{BoundedTensor, TensorParamBinding};
use ndarray::ArrayD;

// Causal mask + propagation — delegated to super::common (Part of #1970).
pub(super) use super::common::{
    build_linear_causal_mask, build_strict_causal_mask, graph_propagate, linear_alignment,
    sinusoidal_pe_interleaved, strict_causal_alignment,
};

use super::common::weights;

// ---------------------------------------------------------------------------
// Graph builders
// ---------------------------------------------------------------------------

/// Build multi-head causal cross-attention scores (simple: no projections).
///
/// Q: Variable [T_dec, D], K: Const [T_enc, D], mask: Const [T_dec, T_enc]
/// Output: S_masked = (Q @ K^T / √d_k) + mask_broadcast → [H, T_dec, T_enc]
///
/// The "multi-head" here reshapes Q and K into [H, T, d_k] before scoring.
pub(super) fn build_mh_causal_simple(
    t_dec: usize,
    t_enc: usize,
    d_model: usize,
    num_heads: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let d_k = d_model / num_heads;
    let mut b = TensorBlockBuilder::new("mh_causal_simple");

    let q = b.add_input("decoder_hidden", &[t_dec, d_model]);
    let k = b.add_input("encoder_text", &[t_enc, d_model]);
    let mask = b.add_input("causal_mask", &[t_dec, t_enc]);

    // Reshape: [T, D] → [T, H, d_k]
    let q_r = b.add_reshape(q, &[t_dec, num_heads, d_k]);
    let k_r = b.add_reshape(k, &[t_enc, num_heads, d_k]);

    // Transpose: [T, H, d_k] → [H, T, d_k]
    let q_t = b.add_transpose(q_r, &[1, 0, 2], &[num_heads, t_dec, d_k]);
    let k_t = b.add_transpose(k_r, &[1, 0, 2], &[num_heads, t_enc, d_k]);

    // Scores: [H, T_dec, d_k] @ [H, d_k, T_enc] → [H, T_dec, T_enc]
    let scale = 1.0 / (d_k as f32).sqrt();
    let scores_shape = [num_heads, t_dec, t_enc];
    let scores = b.add_matmul(q_t, k_t, true, Some(scale), &scores_shape);

    // Broadcast mask: [T_dec, T_enc] → [H, T_dec, T_enc]
    let mask_bc = b.add_broadcast(mask, &scores_shape);
    let masked = b.add_binary_add(scores, mask_bc, &scores_shape);

    b.build(masked).expect("valid multi-head causal graph")
}

/// Build multi-head causal cross-attention with PE and projections.
///
/// Q = (hidden + dec_PE) @ W_q → Reshape → Transpose → [H, T_dec, d_k]
/// K = enc_PE @ W_k → Reshape → Transpose → [H, T_enc, d_k]
/// Scores = Q @ K^T / √d_k + mask_broadcast → [H, T_dec, T_enc]
///
/// W_q and W_k are near-identity to preserve PE structure per-head.
pub(super) fn build_mh_causal_projected(
    t_dec: usize,
    t_enc: usize,
    d_model: usize,
    num_heads: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let d_k = d_model / num_heads;
    let mut b = TensorBlockBuilder::new("mh_causal_projected");

    let hidden = b.add_input("decoder_hidden", &[t_dec, d_model]);
    let dec_pe = b.add_input("decoder_pe", &[t_dec, d_model]);
    let enc_k = b.add_input("encoder_k", &[t_enc, d_model]);
    let w_q = b.add_input("w_q", &[d_model, d_model]);
    let w_k = b.add_input("w_k", &[d_model, d_model]);
    let mask = b.add_input("causal_mask", &[t_dec, t_enc]);

    // Q = (hidden + dec_PE) @ W_q
    let q_combined = b.add_binary_add(hidden, dec_pe, &[t_dec, d_model]);
    let q_proj = b.add_matmul(q_combined, w_q, false, None, &[t_dec, d_model]);

    // K = enc_K @ W_k
    let k_proj = b.add_matmul(enc_k, w_k, false, None, &[t_enc, d_model]);

    // Reshape: [T, D] → [T, H, d_k]
    let q_r = b.add_reshape(q_proj, &[t_dec, num_heads, d_k]);
    let k_r = b.add_reshape(k_proj, &[t_enc, num_heads, d_k]);

    // Transpose: [T, H, d_k] → [H, T, d_k]
    let q_t = b.add_transpose(q_r, &[1, 0, 2], &[num_heads, t_dec, d_k]);
    let k_t = b.add_transpose(k_r, &[1, 0, 2], &[num_heads, t_enc, d_k]);

    // Scores: [H, T_dec, d_k] @ [H, d_k, T_enc] → [H, T_dec, T_enc]
    let scale = 1.0 / (d_k as f32).sqrt();
    let scores_shape = [num_heads, t_dec, t_enc];
    let scores = b.add_matmul(q_t, k_t, true, Some(scale), &scores_shape);

    // Broadcast mask: [T_dec, T_enc] → [H, T_dec, T_enc]
    let mask_bc = b.add_broadcast(mask, &scores_shape);
    let masked = b.add_binary_add(scores, mask_bc, &scores_shape);

    b.build(masked)
        .expect("valid projected multi-head causal graph")
}

// ---------------------------------------------------------------------------
// Binding constructors
// ---------------------------------------------------------------------------

/// Simple multi-head causal bindings: Q=Variable, K=Const, mask=Const.
pub(super) fn mh_simple_bindings(
    t_enc: usize,
    d_model: usize,
    mask: ArrayD<f32>,
) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weights::encoder_k(t_enc, d_model)),
        TensorParamBinding::ConstantTensor(mask),
    ]
}

/// Projected multi-head causal bindings with head-interleaved PE.
///
/// hidden=Variable, dec_PE=Const, enc_K=Const(PE), W_q=Const, W_k=Const, mask=Const.
///
/// Uses interleaved PE to distribute frequency diversity evenly across heads.
/// Without interleaving, contiguous head slicing assigns all high-frequency-base
/// dims (slow-varying, nearly identical across positions for small T) to the
/// last head, destroying diagonal dominance.
pub(super) fn mh_projected_bindings(
    t_dec: usize,
    t_enc: usize,
    d_model: usize,
    num_heads: usize,
    pe_scale: f32,
    w_perturbation: f32,
    mask: ArrayD<f32>,
) -> Vec<TensorParamBinding> {
    let mut dec_pe = sinusoidal_pe_interleaved(t_dec, d_model, num_heads);
    dec_pe.mapv_inplace(|v| v * pe_scale);
    let mut enc_pe = sinusoidal_pe_interleaved(t_enc, d_model, num_heads);
    enc_pe.mapv_inplace(|v| v * pe_scale);
    let w_q = weights::near_identity(d_model, w_perturbation);
    let w_k = weights::near_identity(d_model, w_perturbation);
    vec![
        TensorParamBinding::Variable,               // hidden
        TensorParamBinding::ConstantTensor(dec_pe), // dec_PE
        TensorParamBinding::ConstantTensor(enc_pe), // enc_K (PE-based)
        TensorParamBinding::ConstantTensor(w_q),    // W_q
        TensorParamBinding::ConstantTensor(w_k),    // W_k
        TensorParamBinding::ConstantTensor(mask),   // causal mask
    ]
}

// ---------------------------------------------------------------------------
// Multi-head causal alignment certificate
// ---------------------------------------------------------------------------

/// Certificate for multi-head causal attention alignment dominance.
///
/// For each head `h` and each decoder step `t`, checks that the alignment
/// target `S[h, t, f(t)]` dominates all other unmasked positions in row `t`:
///   `lower(S[h, t, f(t)]) > max_{j unmasked, j != f(t)}(upper(S[h, t, j]))`
///
/// Reports per-head margins and the overall minimum (worst-case head).
#[derive(Debug)]
pub(super) struct MultiHeadCausalCertificate {
    pub(super) num_heads: usize,
    pub(super) decoder_steps: usize,
    pub(super) encoder_positions: usize,
    /// Per-head minimum margins: `per_head_min_margin[h]` is the worst row
    /// margin for head `h`.
    pub(super) per_head_min_margin: Vec<f64>,
    /// Per-head row margins: `per_head_row_margins[h]` is the row margin
    /// vector for head `h`.
    pub(super) per_head_row_margins: Vec<Vec<f64>>,
    /// Overall minimum margin across all heads and rows.
    pub(super) min_margin: f64,
    /// Whether all heads have proven alignment dominance (min_margin > 0).
    pub(super) is_proven: bool,
    /// Number of heads with individually proven alignment dominance.
    pub(super) proven_heads: usize,
    pub(super) input_bound: f64,
    pub(super) propagation_mode: String,
}

/// Extract a multi-head causal alignment certificate from score bounds.
///
/// Output shape: `[H, T_dec, T_enc]`.
/// `alignment_fn(t)` returns the target encoder position for decoder step `t`.
pub(super) fn extract_mh_causal_certificate(
    output: &BoundedTensor,
    num_heads: usize,
    t_dec: usize,
    t_enc: usize,
    input_bound: f64,
    mode: &str,
    alignment_fn: impl Fn(usize) -> usize,
) -> MultiHeadCausalCertificate {
    let (lo, hi) = output.lower_upper();
    let flat_lo: Vec<f32> = lo.iter().copied().collect();
    let flat_hi: Vec<f32> = hi.iter().copied().collect();

    let head_stride = t_dec * t_enc;
    let mut per_head_min_margin = Vec::with_capacity(num_heads);
    let mut per_head_row_margins = Vec::with_capacity(num_heads);
    let mut proven_heads = 0;

    for h in 0..num_heads {
        let head_offset = h * head_stride;
        let mut row_margins = Vec::new();

        for t in 0..t_dec {
            let target = alignment_fn(t);
            if target >= t_enc {
                continue;
            }

            // Skip post-alignment rows (target saturated at last position).
            let max_visible = target;
            if t_enc > 1 && target == t_enc - 1 && max_visible == t_enc - 1 {
                continue;
            }

            let target_lo = f64::from(flat_lo[head_offset + t * t_enc + target]);

            // Find max upper bound among OTHER unmasked positions.
            let mut max_other_hi = f64::NEG_INFINITY;
            for j in 0..=max_visible {
                if j != target {
                    let upper = f64::from(flat_hi[head_offset + t * t_enc + j]);
                    if upper > max_other_hi {
                        max_other_hi = upper;
                    }
                }
            }

            // If target is the only visible position, trivially dominant.
            let margin = if max_visible == 0 {
                f64::INFINITY
            } else {
                target_lo - max_other_hi
            };
            row_margins.push(margin);
        }

        let head_min = row_margins.iter().copied().fold(f64::INFINITY, f64::min);
        if head_min > 0.0 {
            proven_heads += 1;
        }
        per_head_min_margin.push(head_min);
        per_head_row_margins.push(row_margins);
    }

    let min_margin = per_head_min_margin
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    MultiHeadCausalCertificate {
        num_heads,
        decoder_steps: t_dec,
        encoder_positions: t_enc,
        per_head_min_margin,
        per_head_row_margins,
        min_margin,
        is_proven: min_margin > 0.0,
        proven_heads,
        input_bound,
        propagation_mode: mode.to_string(),
    }
}

/// Count unmasked positions per row from a causal mask.
pub(super) fn count_unmasked_per_row(mask: &ArrayD<f32>, t_dec: usize, t_enc: usize) -> Vec<usize> {
    let data = mask.as_slice().expect("contiguous mask");
    (0..t_dec)
        .map(|t| (0..t_enc).filter(|&j| data[t * t_enc + j] == 0.0).count())
        .collect()
}
