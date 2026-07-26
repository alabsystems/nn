// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Qwen3 GQA (Grouped Query Attention) NY composition.
//!
//! Models the Grouped Query Attention mechanism where Q has more heads than K/V:
//!
//!   Q: [seq_len, num_heads * head_dim]     -- full head count
//!   K: [seq_len, num_kv_heads * head_dim]  -- fewer heads
//!   V: [seq_len, num_kv_heads * head_dim]  -- fewer heads
//!
//! Each KV head serves `num_heads / num_kv_heads` query heads. This is modeled
//! by repeating KV heads to match Q's head count before standard multi-head
//! attention (Ainslie et al., 2023 "GQA: Training Generalized Multi-Query
//! Transformer Models from Multi-Head Checkpoints").
//!
//! The GQA test verifies the full attention pipeline:
//!   Linear(Q) -> Linear(K) -> Linear(V) ->
//!   Reshape -> [K/V repeat] -> Transpose ->
//!   Attention(Q, K^T, V) -> Transpose -> Reshape -> Linear(out)
//!
//! Dimensions (small for fast verification):
//! - EMBED_DIM=64, NUM_HEADS=4, NUM_KV_HEADS=2, HEAD_DIM=16, SEQ_LEN=8
//!
//! Part of #3560: Qwen3 RoPE + GQA NY compose verification tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions — small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Embedding dimension (production: 4096 for Qwen3-8B).
const EMBED_DIM: usize = 64;
/// Number of query attention heads (production: 32 for Qwen3-8B).
const NUM_HEADS: usize = 4;
/// Number of key/value attention heads (production: 8 for Qwen3-8B GQA).
const NUM_KV_HEADS: usize = 2;
/// Per-head dimension: EMBED_DIM / NUM_HEADS.
const HEAD_DIM: usize = EMBED_DIM / NUM_HEADS; // 16
/// Number of query head groups per KV head.
const NUM_GROUPS: usize = NUM_HEADS / NUM_KV_HEADS; // 2
/// Sequence length.
const SEQ_LEN: usize = 8;
/// KV projection dimension: NUM_KV_HEADS * HEAD_DIM.
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 32

/// Weight magnitude for small-scale test weights.
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Build a GQA attention block as a TensorKernelDef.
///
/// Input: `[SEQ_LEN, EMBED_DIM]` (Variable — post-RMSNorm hidden states).
///
/// Architecture:
///   1. Q projection: [S, D] -> [S, D]          (full head count)
///   2. K projection: [S, D] -> [S, KV_DIM]     (fewer heads)
///   3. V projection: [S, D] -> [S, KV_DIM]     (fewer heads)
///   4. Reshape Q to [S, NUM_HEADS, HEAD_DIM]
///   5. Reshape K to [S, NUM_KV_HEADS, HEAD_DIM]
///   6. Reshape V to [S, NUM_KV_HEADS, HEAD_DIM]
///   7. Repeat K/V heads: [S, NUM_KV_HEADS, HD] -> [S, NUM_HEADS, HD]
///      via reshape [S, NUM_KV_HEADS, 1, HD] -> broadcast [S, NUM_KV_HEADS, GROUPS, HD]
///      -> reshape [S, NUM_HEADS, HD]
///   8. Transpose all to [NUM_HEADS, S, HEAD_DIM]
///   9. Attention: softmax(Q @ K^T / sqrt(d)) @ V
///  10. Transpose back to [S, NUM_HEADS, HEAD_DIM]
///  11. Reshape to [S, D]
///  12. Output projection: [S, D] -> [S, D]
fn build_gqa_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_gqa");

    // --- Inputs ---
    let input = b.add_input("hidden", &[SEQ_LEN, EMBED_DIM]);
    let q_weight = b.add_input("q_proj", &[EMBED_DIM, EMBED_DIM]);
    let k_weight = b.add_input("k_proj", &[KV_DIM, EMBED_DIM]);
    let v_weight = b.add_input("v_proj", &[KV_DIM, EMBED_DIM]);
    let o_weight = b.add_input("o_proj", &[EMBED_DIM, EMBED_DIM]);

    // --- Q/K/V projections ---
    let q = b.add_linear(input, q_weight, None, &[SEQ_LEN, EMBED_DIM]);
    let k = b.add_linear(input, k_weight, None, &[SEQ_LEN, KV_DIM]);
    let v = b.add_linear(input, v_weight, None, &[SEQ_LEN, KV_DIM]);

    // --- Reshape to multi-head layout ---
    // Q: [S, D] -> [S, NUM_HEADS, HEAD_DIM]
    let q = b.add_reshape(q, &[SEQ_LEN, NUM_HEADS, HEAD_DIM]);
    // K: [S, KV_DIM] -> [S, NUM_KV_HEADS, HEAD_DIM]
    let k = b.add_reshape(k, &[SEQ_LEN, NUM_KV_HEADS, HEAD_DIM]);
    // V: [S, KV_DIM] -> [S, NUM_KV_HEADS, HEAD_DIM]
    let v = b.add_reshape(v, &[SEQ_LEN, NUM_KV_HEADS, HEAD_DIM]);

    // --- Repeat KV heads to match Q head count ---
    // K: [S, NUM_KV_HEADS, HEAD_DIM] -> [S, NUM_KV_HEADS, 1, HEAD_DIM]
    let k = b.add_reshape(k, &[SEQ_LEN, NUM_KV_HEADS, 1, HEAD_DIM]);
    // Broadcast: [S, NUM_KV_HEADS, 1, HEAD_DIM] -> [S, NUM_KV_HEADS, NUM_GROUPS, HEAD_DIM]
    let k = b.add_broadcast(k, &[SEQ_LEN, NUM_KV_HEADS, NUM_GROUPS, HEAD_DIM]);
    // Reshape: [S, NUM_KV_HEADS, NUM_GROUPS, HEAD_DIM] -> [S, NUM_HEADS, HEAD_DIM]
    let k = b.add_reshape(k, &[SEQ_LEN, NUM_HEADS, HEAD_DIM]);

    // V: same repeat pattern
    let v = b.add_reshape(v, &[SEQ_LEN, NUM_KV_HEADS, 1, HEAD_DIM]);
    let v = b.add_broadcast(v, &[SEQ_LEN, NUM_KV_HEADS, NUM_GROUPS, HEAD_DIM]);
    let v = b.add_reshape(v, &[SEQ_LEN, NUM_HEADS, HEAD_DIM]);

    // --- Transpose to [NUM_HEADS, S, HEAD_DIM] for per-head attention ---
    let q = b.add_transpose(q, &[1, 0, 2], &[NUM_HEADS, SEQ_LEN, HEAD_DIM]);
    let k = b.add_transpose(k, &[1, 0, 2], &[NUM_HEADS, SEQ_LEN, HEAD_DIM]);
    let v = b.add_transpose(v, &[1, 0, 2], &[NUM_HEADS, SEQ_LEN, HEAD_DIM]);

    // --- Attention with causal mask ---
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn_out = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[NUM_HEADS, SEQ_LEN, HEAD_DIM],
    );

    // --- Transpose back to [S, NUM_HEADS, HEAD_DIM] ---
    let attn_out = b.add_transpose(attn_out, &[1, 0, 2], &[SEQ_LEN, NUM_HEADS, HEAD_DIM]);

    // --- Reshape to [S, D] ---
    let attn_out = b.add_reshape(attn_out, &[SEQ_LEN, EMBED_DIM]);

    // --- Output projection ---
    let output = b.add_linear(attn_out, o_weight, None, &[SEQ_LEN, EMBED_DIM]);

    b.build(output).expect("valid GQA kernel")
}

/// Build parameter bindings for the GQA kernel.
///
/// hidden = Variable, all weights = ConstantTensor.
fn gqa_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // hidden [SEQ_LEN, EMBED_DIM]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, EMBED_DIM]),
            WEIGHT_MAG,
        )), // q_proj
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[KV_DIM, EMBED_DIM]),
            WEIGHT_MAG,
        )), // k_proj
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[KV_DIM, EMBED_DIM]),
            WEIGHT_MAG,
        )), // v_proj
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, EMBED_DIM]),
            WEIGHT_MAG,
        )), // o_proj
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// GQA TensorKernelDef validates.
#[test]
fn test_qwen3_gqa_def_validates() {
    let def = build_gqa_kernel();
    def.validate().expect("GQA kernel should validate");
}

/// GQA translates to NY GraphNetwork.
#[test]
fn test_qwen3_gqa_graph_builds() {
    let def = build_gqa_kernel();
    let bindings = gqa_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("GQA graph should translate");

    // Linear(3) + Reshape(6) + Broadcast(2) + Transpose(3+1) + Attention(1) +
    // Transpose(1) + Reshape(1) + Linear(1) = ~19+ nodes
    assert!(
        graph.num_nodes() >= 15,
        "GQA graph should have >= 15 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through GQA.
///
/// With small weights (0.02) and [-1, 1] input, the attention softmax
/// normalizes scores to sum to 1, so output magnitudes stay bounded.
#[test]
fn test_qwen3_gqa_ibp_propagates() {
    let def = build_gqa_kernel();
    let bindings = gqa_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through GQA");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 GQA IBP: bounds=[{lo_min}, {hi_max}]");

    // With small weights, IBP bounds should remain tractable.
    // The attention softmax constrains the output range.
    assert!(
        lo_min.is_finite(),
        "IBP lower bound must be finite, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "IBP upper bound must be finite, got {hi_max}"
    );
    assert!(
        lo_min.abs() < 1e8,
        "IBP lower bound magnitude should be < 1e8, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1e8,
        "IBP upper bound magnitude should be < 1e8, got {hi_max}"
    );
}

/// CROWN bounds propagate through GQA.
///
/// GQA includes softmax (nonlinear) and bilinear matmul (Q @ K^T),
/// so CROWN may fall back to IBP. The test verifies structural validity
/// regardless of propagation method.
#[test]
fn test_qwen3_gqa_crown_propagation() {
    let def = build_gqa_kernel();
    let bindings = gqa_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 GQA: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

/// GQA verify and record under "qwen3_gqa" key.
#[test]
fn test_qwen3_gqa_verify_and_record() {
    let def = build_gqa_kernel();
    let bindings = gqa_bindings();
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_gqa");
    assert_eq!(result.num_variables, 1, "single Variable input (hidden)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, EMBED_DIM]);
}
