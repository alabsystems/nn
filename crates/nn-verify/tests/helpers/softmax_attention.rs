// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for softmax-inclusive multi-head causal attention.
//!
//! Phase 23 of #1729: extends Phase 22's multi-head causal cross-attention
//! to include softmax, proving monotonicity of actual attention *weights*
//! (post-softmax), not just pre-softmax *scores*.
//!
//! Architecture:
//! ```text
//! Q: [T_dec, D] → W_q [D, D] → Reshape [T_dec, H, d_k] → Transpose [H, T_dec, d_k]
//! K: [T_enc, D] → W_k [D, D] → Reshape [T_enc, H, d_k] → Transpose [H, T_enc, d_k]
//! Scores = Q_proj @ K_proj^T / √d_k → [H, T_dec, T_enc]
//! Mask: [T_dec, T_enc] → Broadcast [H, T_dec, T_enc]
//! S_masked = Scores + Mask_broadcast → [H, T_dec, T_enc]
//! W = Softmax(S_masked, axis=-1) → [H, T_dec, T_enc]   ← NEW in Phase 23
//! ```
//!
//! Key insight: softmax is a monotone function — if pre-softmax score S[t,j*]
//! dominates all other S[t,j], then post-softmax weight W[t,j*] also dominates.
//! The formal proof propagates IBP bounds through NY's SoftmaxLayer.
//! The resulting bounds on W are in [0,1] with sum-to-one constraints, giving
//! tighter certificates than raw score bounds.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 23.

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::TensorParamBinding;
use ndarray::ArrayD;

// Causal mask + propagation — delegated to super::common (Part of #1970).
pub(super) use super::common::{
    build_linear_causal_mask, build_strict_causal_mask, graph_propagate, linear_alignment,
    sinusoidal_pe_interleaved, strict_causal_alignment,
};

use super::common::weights;

// ---------------------------------------------------------------------------
// Graph builders — softmax-inclusive
// ---------------------------------------------------------------------------

/// Build multi-head causal cross-attention with softmax (simple: no projections).
///
/// Q: Variable [T_dec, D], K: Const [T_enc, D], mask: Const [T_dec, T_enc]
/// Output: W = Softmax(Q @ K^T / √d_k + mask_broadcast, axis=-1) → [H, T_dec, T_enc]
///
/// The output is attention weights in [0, 1] that sum to 1 along the T_enc axis.
pub(super) fn build_mh_causal_softmax_simple(
    t_dec: usize,
    t_enc: usize,
    d_model: usize,
    num_heads: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let d_k = d_model / num_heads;
    let mut b = TensorBlockBuilder::new("mh_causal_softmax_simple");

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

    // Softmax along last axis (T_enc): [H, T_dec, T_enc] → [H, T_dec, T_enc]
    // Each row sums to 1.0, masked positions get near-zero weight.
    let weights = b.add_softmax(masked, -1, &scores_shape);

    b.build(weights)
        .expect("valid softmax multi-head causal graph")
}

/// Build multi-head causal cross-attention with PE, projections, and softmax.
///
/// Q = (hidden + dec_PE) @ W_q → Reshape → Transpose → [H, T_dec, d_k]
/// K = enc_PE @ W_k → Reshape → Transpose → [H, T_enc, d_k]
/// Scores = Q @ K^T / √d_k + mask_broadcast → [H, T_dec, T_enc]
/// W = Softmax(Scores, axis=-1) → [H, T_dec, T_enc]
pub(super) fn build_mh_causal_softmax_projected(
    t_dec: usize,
    t_enc: usize,
    d_model: usize,
    num_heads: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let d_k = d_model / num_heads;
    let mut b = TensorBlockBuilder::new("mh_causal_softmax_projected");

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

    // Broadcast mask + softmax
    let mask_bc = b.add_broadcast(mask, &scores_shape);
    let masked = b.add_binary_add(scores, mask_bc, &scores_shape);
    let weights = b.add_softmax(masked, -1, &scores_shape);

    b.build(weights)
        .expect("valid softmax projected multi-head causal graph")
}

/// Build projected multi-head causal scores WITHOUT softmax (for comparison).
///
/// Same graph as `build_mh_causal_softmax_projected` but stops at masked scores.
/// Used to compare score-space margins (Phase 22) against weight-space margins
/// (Phase 23) on the same architecture.
pub(super) fn build_scores_only_projected(
    t_dec: usize,
    t_enc: usize,
    d_model: usize,
    num_heads: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let d_k = d_model / num_heads;
    let mut b = TensorBlockBuilder::new("scores_only_projected");

    let hidden = b.add_input("decoder_hidden", &[t_dec, d_model]);
    let dec_pe = b.add_input("decoder_pe", &[t_dec, d_model]);
    let enc_k = b.add_input("encoder_k", &[t_enc, d_model]);
    let w_q = b.add_input("w_q", &[d_model, d_model]);
    let w_k = b.add_input("w_k", &[d_model, d_model]);
    let mask = b.add_input("causal_mask", &[t_dec, t_enc]);

    let q_combined = b.add_binary_add(hidden, dec_pe, &[t_dec, d_model]);
    let q_proj = b.add_matmul(q_combined, w_q, false, None, &[t_dec, d_model]);
    let k_proj = b.add_matmul(enc_k, w_k, false, None, &[t_enc, d_model]);

    let q_r = b.add_reshape(q_proj, &[t_dec, num_heads, d_k]);
    let k_r = b.add_reshape(k_proj, &[t_enc, num_heads, d_k]);
    let q_t = b.add_transpose(q_r, &[1, 0, 2], &[num_heads, t_dec, d_k]);
    let k_t = b.add_transpose(k_r, &[1, 0, 2], &[num_heads, t_enc, d_k]);

    let scale = 1.0 / (d_k as f32).sqrt();
    let scores_shape = [num_heads, t_dec, t_enc];
    let scores = b.add_matmul(q_t, k_t, true, Some(scale), &scores_shape);
    let mask_bc = b.add_broadcast(mask, &scores_shape);
    let masked = b.add_binary_add(scores, mask_bc, &scores_shape);

    b.build(masked).expect("valid scores-only projected graph")
}

// ---------------------------------------------------------------------------
// Binding constructors
// ---------------------------------------------------------------------------

/// Simple softmax multi-head causal bindings: Q=Variable, K=Const, mask=Const.
pub(super) fn mh_softmax_simple_bindings(
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

/// Projected softmax multi-head causal bindings with head-interleaved PE.
pub(super) fn mh_softmax_projected_bindings(
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
// Softmax-inclusive attention weight certificate
// ---------------------------------------------------------------------------

#[path = "softmax_attention_certificate.rs"]
mod certificate;
pub(super) use certificate::{
    assert_certificate_margins_valid, assert_weight_margins_bounded, extract_softmax_certificate,
};
