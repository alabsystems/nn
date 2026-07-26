// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for monolithic `TensorOpKind::Attention` → NY
//! `SelfAttentionLayer` translation.
//!
//! Tests standard (bidirectional) and causal attention at various dimensions,
//! including dvoice-scale Qwen3 GQA dimensions (16 heads, d=128).
//!
//! Part of #750, Part of #729.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Build a monolithic attention kernel: Attention(Q, K, V) → output.
fn build_attention_kernel(
    name: &str,
    seq_len: usize,
    d_k: usize,
    d_v: usize,
    mask: AttentionMask,
    scale: Option<f32>,
) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let q = b.add_input("query", &[seq_len, d_k]);
    let k = b.add_input("key", &[seq_len, d_k]);
    let v = b.add_input("value", &[seq_len, d_v]);
    let out = b.add_attention(q, k, v, mask, scale, &[seq_len, d_v]);
    b.build(out).expect("valid graph")
}

// ---------------------------------------------------------------------------
// Standard attention tests
// ---------------------------------------------------------------------------

#[test]
fn test_attention_standard_translation_builds_graph() {
    let kernel = build_attention_kernel("attn_std", 4, 8, 8, AttentionMask::Standard, None);
    let bindings = vec![
        TensorParamBinding::Variable, // Q
        TensorParamBinding::Variable, // K
        TensorParamBinding::Variable, // V
    ];
    let graph = tensor_kernel_to_graph(&kernel, &bindings)
        .expect("standard attention must build NY graph");
    // Should have: 3 input slices + 1 attention node = 4 minimum
    assert!(
        graph.num_nodes() >= 4,
        "attention graph too small: {} nodes",
        graph.num_nodes()
    );
}

#[test]
fn test_attention_standard_ibp_propagation() {
    let seq_len = 4;
    let d = 8;
    let kernel = build_attention_kernel("attn_ibp", seq_len, d, d, AttentionMask::Standard, None);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
    ];
    let graph = tensor_kernel_to_graph(&kernel, &bindings).expect("build graph");

    // Stacked input: [3, seq_len, d] (3 variables stacked along axis 0)
    let lower = ArrayD::from_elem(IxDyn(&[3, seq_len, d]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3, seq_len, d]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP propagation for standard attention must succeed");
    let (lo, hi) = output.lower_upper();

    // Bounds must be finite
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "output lower bounds must be finite: {lo:?}"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "output upper bounds must be finite: {hi:?}"
    );
    // Bounds soundness: lower <= upper for all elements.
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

#[test]
fn test_attention_standard_with_explicit_scale() {
    let seq_len = 4;
    let d = 64;
    let scale = 1.0 / (d as f32).sqrt(); // 1/sqrt(64) = 0.125
    let kernel = build_attention_kernel(
        "attn_scaled",
        seq_len,
        d,
        d,
        AttentionMask::Standard,
        Some(scale),
    );
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
    ];
    let graph = tensor_kernel_to_graph(&kernel, &bindings).expect("build scaled attention graph");

    let lower = ArrayD::from_elem(IxDyn(&[3, seq_len, d]), -0.5f32);
    let upper = ArrayD::from_elem(IxDyn(&[3, seq_len, d]), 0.5f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP for scaled attention");
    let (lo, hi) = output.lower_upper();

    assert!(
        lo.iter().all(|v| v.is_finite()),
        "scaled attention lower bounds finite: {lo:?}"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "scaled attention upper bounds finite: {hi:?}"
    );
}

// ---------------------------------------------------------------------------
// Causal attention tests
// ---------------------------------------------------------------------------

#[test]
fn test_attention_causal_translation_builds_graph() {
    let kernel = build_attention_kernel("attn_causal", 4, 8, 8, AttentionMask::Causal, None);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
    ];
    let graph = tensor_kernel_to_graph(&kernel, &bindings)
        .expect("causal attention must build NY graph");
    assert!(
        graph.num_nodes() >= 4,
        "causal attention graph too small: {} nodes",
        graph.num_nodes()
    );
}

#[test]
fn test_attention_causal_ibp_propagation() {
    let seq_len = 4;
    let d = 8;
    let kernel = build_attention_kernel(
        "attn_causal_ibp",
        seq_len,
        d,
        d,
        AttentionMask::Causal,
        None,
    );
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
    ];
    let graph = tensor_kernel_to_graph(&kernel, &bindings).expect("build causal graph");

    let lower = ArrayD::from_elem(IxDyn(&[3, seq_len, d]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3, seq_len, d]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP for causal attention must succeed");
    let (lo, hi) = output.lower_upper();

    assert!(
        lo.iter().all(|v| v.is_finite()),
        "causal output lower bounds finite: {lo:?}"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "causal output upper bounds finite: {hi:?}"
    );
}

// ---------------------------------------------------------------------------
// dvoice-scale: Qwen3 GQA dimensions
// ---------------------------------------------------------------------------

#[test]
fn test_attention_dvoice_qwen3_dimensions() {
    // Qwen3-TTS uses GQA with 16 heads, d_head=128.
    // Single-head verification: seq_len=32, d_k=128, d_v=128.
    let seq_len = 32;
    let d_k = 128;
    let d_v = 128;
    let scale = 1.0 / (d_k as f32).sqrt(); // 1/sqrt(128) ≈ 0.0884

    let kernel = build_attention_kernel(
        "qwen3_attn",
        seq_len,
        d_k,
        d_v,
        AttentionMask::Causal,
        Some(scale),
    );
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
    ];
    let graph = tensor_kernel_to_graph(&kernel, &bindings)
        .expect("Qwen3-scale attention must build NY graph");

    // Use realistic embedding range [-2, 2] (post-LayerNorm activations)
    let lower = ArrayD::from_elem(IxDyn(&[3, seq_len, d_k]), -2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3, seq_len, d_k]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP for Qwen3 attention must succeed");
    let (lo, hi) = output.lower_upper();

    assert!(
        lo.iter().all(|v| v.is_finite()),
        "Qwen3 attention lower bounds must be finite"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "Qwen3 attention upper bounds must be finite"
    );
}

// ---------------------------------------------------------------------------
// Validation tests
// ---------------------------------------------------------------------------

/// Build an Attention kernel manually (bypassing TensorBlockBuilder's
/// debug_assert in `build()`) so we can test invalid configurations.
fn manual_attention_kernel(
    q_shape: &[usize],
    k_shape: &[usize],
    v_shape: &[usize],
    out_shape: &[usize],
    mask: AttentionMask,
    scale: Option<f32>,
) -> TensorKernelDef {
    TensorKernelDef::new(
        "test_attn",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "query".into(),
                    shape: q_shape.to_vec(),
                },
                q_shape.to_vec(),
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "key".into(),
                    shape: k_shape.to_vec(),
                },
                k_shape.to_vec(),
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Input {
                    name: "value".into(),
                    shape: v_shape.to_vec(),
                },
                v_shape.to_vec(),
            ),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::Attention {
                    q: TensorNodeId::new(0),
                    k: TensorNodeId::new(1),
                    v: TensorNodeId::new(2),
                    mask,
                    scale,
                },
                out_shape.to_vec(),
            ),
        ],
        TensorNodeId::new(3),
    )
}

#[test]
fn test_attention_qk_dimension_mismatch_rejected() {
    // Q d_k=8, K d_k=16 — mismatch
    let kernel = manual_attention_kernel(
        &[4, 8],
        &[4, 16],
        &[4, 8],
        &[4, 8],
        AttentionMask::Standard,
        None,
    );
    let result = kernel.validate();
    assert!(
        result.is_err(),
        "Q/K dimension mismatch should fail validation"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("head dimension mismatch"),
        "error should mention head dimension: {err_msg}"
    );
}

#[test]
fn test_attention_kv_seq_mismatch_rejected() {
    // K T_kv=6, V T_kv=8 — mismatch
    let kernel = manual_attention_kernel(
        &[4, 8],
        &[6, 8],
        &[8, 8],
        &[4, 8],
        AttentionMask::Standard,
        None,
    );
    let result = kernel.validate();
    assert!(
        result.is_err(),
        "K/V seq length mismatch should fail validation"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("sequence length mismatch"),
        "error should mention seq length: {err_msg}"
    );
}

#[test]
fn test_attention_negative_scale_rejected() {
    let kernel = manual_attention_kernel(
        &[4, 8],
        &[4, 8],
        &[4, 8],
        &[4, 8],
        AttentionMask::Standard,
        Some(-1.0),
    );
    let result = kernel.validate();
    assert!(result.is_err(), "negative scale should fail validation");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("scale"),
        "error should mention scale: {err_msg}"
    );
}

#[test]
fn test_attention_rank_1_input_rejected() {
    // Q is rank 1 — needs >= 2
    let kernel = manual_attention_kernel(
        &[8],
        &[4, 8],
        &[4, 8],
        &[4, 8],
        AttentionMask::Standard,
        None,
    );
    let result = kernel.validate();
    assert!(result.is_err(), "rank-1 Q should fail validation");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("at least 2 dimensions"),
        "error should mention rank: {err_msg}"
    );
}
