// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for asymmetric cross-attention (Q_SEQ != KV_SEQ).
//!
//! Provides graph builders and binding constructors for Phase 20 tests
//! where decoder sequence length differs from encoder sequence length,
//! modeling the realistic TTS cross-attention pattern.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 20.

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code)]

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Identity-like encoder K tensor
// ---------------------------------------------------------------------------

use super::common::weights;

// Sinusoidal PE — delegated to super::common (Part of #1970).
pub(super) use super::common::sinusoidal_pe;

// ---------------------------------------------------------------------------
// Graph builders
// ---------------------------------------------------------------------------

/// Build an asymmetric cross-attention score graph (simple, no projections).
///
/// Q: [t_dec, d] (Variable — decoder hidden states)
/// K: [t_enc, d] (ConstantTensor — encoder text embeddings)
/// Output: [t_dec, t_enc] pre-softmax scores.
pub(super) fn build_asymmetric_scores_simple(
    t_dec: usize,
    t_enc: usize,
    d: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("asym_scores_simple");
    let q = b.add_input("decoder_hidden", &[t_dec, d]);
    let k = b.add_input("encoder_text", &[t_enc, d]);
    let scale = 1.0 / (d as f32).sqrt();
    let scores = b.add_matmul(q, k, true, Some(scale), &[t_dec, t_enc]);
    b.build(scores).expect("valid asymmetric score graph")
}

/// Build asymmetric cross-attention with linear projections and multi-head.
///
/// Q: [t_dec, D] → W_q → reshape [t_dec, H, dk] → transpose [H, t_dec, dk]
/// K: [t_enc, D] → W_k → reshape [t_enc, H, dk] → transpose [H, t_enc, dk]
/// Scores: [H, t_dec, dk] @ [H, dk, t_enc] → [H, t_dec, t_enc]
pub(super) fn build_asymmetric_scores_projected(
    t_dec: usize,
    t_enc: usize,
    d: usize,
    num_heads: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let dk = d / num_heads;
    let mut b = TensorBlockBuilder::new("asym_scores_projected");

    let q_input = b.add_input("decoder_hidden", &[t_dec, d]);
    let k_input = b.add_input("encoder_text", &[t_enc, d]);
    let w_q = b.add_input("w_q", &[d, d]);
    let w_k = b.add_input("w_k", &[d, d]);

    let q_proj = b.add_matmul(q_input, w_q, false, None, &[t_dec, d]);
    let k_proj = b.add_matmul(k_input, w_k, false, None, &[t_enc, d]);

    let q_mh = b.add_reshape(q_proj, &[t_dec, num_heads, dk]);
    let k_mh = b.add_reshape(k_proj, &[t_enc, num_heads, dk]);

    let q_t = b.add_transpose(q_mh, &[1, 0, 2], &[num_heads, t_dec, dk]);
    let k_t = b.add_transpose(k_mh, &[1, 0, 2], &[num_heads, t_enc, dk]);

    let scale = 1.0 / (dk as f32).sqrt();
    let scores = b.add_matmul(q_t, k_t, true, Some(scale), &[num_heads, t_dec, t_enc]);

    b.build(scores)
        .expect("valid asymmetric projected score graph")
}

/// Build PE-aware asymmetric cross-attention.
///
/// Q = decoder_hidden + decoder_PE → [t_dec, D]
/// K = encoder_PE → [t_enc, D]
/// Scores = Q @ K^T / sqrt(D) → [t_dec, t_enc]
pub(super) fn build_asymmetric_scores_pe_aware(
    t_dec: usize,
    t_enc: usize,
    d: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("asym_scores_pe");
    let hidden = b.add_input("decoder_hidden", &[t_dec, d]);
    let dec_pe = b.add_input("decoder_pe", &[t_dec, d]);
    let enc_pe = b.add_input("encoder_pe", &[t_enc, d]);
    let q = b.add_binary_add(hidden, dec_pe, &[t_dec, d]);
    let scale = 1.0 / (d as f32).sqrt();
    let scores = b.add_matmul(q, enc_pe, true, Some(scale), &[t_dec, t_enc]);
    b.build(scores)
        .expect("valid asymmetric PE-aware score graph")
}

// ---------------------------------------------------------------------------
// Binding constructors
// ---------------------------------------------------------------------------

/// Simple bindings: Q=Variable, K=ConstantTensor (identity-like encoder).
pub(super) fn simple_bindings(t_enc: usize, d: usize) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weights::encoder_k(t_enc, d)),
    ]
}

/// Projected bindings with uniform small-weight projections.
pub(super) fn projected_bindings(t_enc: usize, d: usize, w_scale: f32) -> Vec<TensorParamBinding> {
    let k_tensor = weights::encoder_k(t_enc, d);
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), w_scale);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(k_tensor),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj),
    ]
}

/// PE-aware bindings: Q=Variable, decoder_PE=Constant, encoder_PE=Constant.
///
/// `pe_scale` controls the PE amplitude. Larger values increase the
/// constant diagonal-dominant signal relative to the Variable perturbation.
pub(super) fn pe_aware_bindings(
    t_dec: usize,
    t_enc: usize,
    d: usize,
    pe_scale: f32,
) -> Vec<TensorParamBinding> {
    let mut dec_pe = sinusoidal_pe(t_dec, d);
    let mut enc_pe = sinusoidal_pe(t_enc, d);
    dec_pe.mapv_inplace(|v| v * pe_scale);
    enc_pe.mapv_inplace(|v| v * pe_scale);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(dec_pe),
        TensorParamBinding::ConstantTensor(enc_pe),
    ]
}

// ---------------------------------------------------------------------------
// Propagation + certificate helpers
// ---------------------------------------------------------------------------

pub(super) use super::common::graph_propagate;

/// Extract a monotonicity certificate from flat score bounds.
pub(super) fn extract_certificate(
    output: &BoundedTensor,
    t_dec: usize,
    t_enc: usize,
    input_bound: f64,
    mode: &str,
) -> nn_tts_verify::monotonicity::AttentionMonotonicityCertificate {
    let (lo, hi) = output.lower_upper();
    let score_lower: Vec<f32> = lo.iter().copied().collect();
    let score_upper: Vec<f32> = hi.iter().copied().collect();
    nn_tts_verify::monotonicity::interpret_attention_monotonicity(
        &score_lower,
        &score_upper,
        t_dec,
        t_enc,
        input_bound,
        mode,
    )
    .expect("valid certificate")
}
