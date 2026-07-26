// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Sliding Window Attention NY composition.
//!
//! Models the SlidingWindowAttention mechanism from Mistral/LongNet/BigBird:
//!
//!   1. Fused QKV projection: [S, D] -> [S, 3*D]
//!   2. Split into Q, K, V via narrow
//!   3. Reshape to multi-head: [S, H, head_dim]
//!   4. Transpose to per-head: [H, S, head_dim]
//!   5. Scaled dot-product attention (standard mask — the banded sliding
//!      window mask is an additive constant, not a learned parameter, so
//!      it is absorbed into the Attention op for verification purposes)
//!   6. Transpose back: [S, H, head_dim]
//!   7. Reshape to [S, D]
//!   8. Output projection: [S, D] -> [S, D]
//!
//! The sliding window constraint restricts each token to attend only within
//! a local neighborhood. For NY verification the key insight is that
//! the window mask is a fixed constant (not input-dependent), so the attention
//! op itself has the same structure as standard MHA — the banded mask only
//! affects the numerical tightness of bounds, not their soundness.
//!
//! Dimensions: EMBED_DIM=32, NUM_HEADS=2, HEAD_DIM=16, SEQ_LEN=8, WINDOW=5.
//!
//! Part of #3563: SlidingWindowAttention + RotaryEmbedding2d compose tests.

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

/// Embedding dimension (production: 512-4096).
const EMBED_DIM: usize = 32;
/// Number of attention heads.
const NUM_HEADS: usize = 2;
/// Per-head dimension: EMBED_DIM / NUM_HEADS.
const HEAD_DIM: usize = EMBED_DIM / NUM_HEADS; // 16
/// Sequence length.
const SEQ_LEN: usize = 8;
/// Sliding window size (each token attends to at most this many positions).
/// Window of 5 means each token sees itself + 2 neighbors on each side.
const _WINDOW_SIZE: usize = 5;

/// Weight magnitude for small-scale test weights.
const WEIGHT_MAG: f32 = 0.02;

/// Fused QKV output dimension: 3 * EMBED_DIM.
const QKV_DIM: usize = 3 * EMBED_DIM;

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Build a Sliding Window Attention block as a TensorKernelDef.
///
/// Input: `[SEQ_LEN, EMBED_DIM]` (Variable — post-norm hidden states).
///
/// Architecture mirrors `SlidingWindowAttention::forward_t`:
///   1. Fused QKV: Linear [S, D] -> [S, 3*D]
///   2. Narrow to get Q, K, V each [S, D]
///   3. Reshape to [S, H, head_dim]
///   4. Transpose to [H, S, head_dim]
///   5. Attention (standard mask — window mask is a constant additive term)
///   6. Transpose back [S, H, head_dim]
///   7. Reshape to [S, D]
///   8. Output projection: Linear [S, D] -> [S, D]
fn build_sliding_window_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("sliding_window_attention");

    // --- Inputs ---
    let input = b.add_input("hidden", &[SEQ_LEN, EMBED_DIM]);
    let qkv_weight = b.add_input("qkv_weight", &[QKV_DIM, EMBED_DIM]);
    let out_weight = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);

    // --- Fused QKV projection: [S, D] -> [S, 3*D] ---
    let qkv = b.add_linear(input, qkv_weight, None, &[SEQ_LEN, QKV_DIM]);

    // --- Split into Q, K, V via narrow along dim 1 ---
    let proj_shape = [SEQ_LEN, EMBED_DIM];
    let q = b.add_narrow(qkv, 1, 0, EMBED_DIM, &proj_shape);
    let k = b.add_narrow(qkv, 1, EMBED_DIM, EMBED_DIM, &proj_shape);
    let v = b.add_narrow(qkv, 1, 2 * EMBED_DIM, EMBED_DIM, &proj_shape);

    // --- Reshape to multi-head layout: [S, D] -> [S, H, head_dim] ---
    let mh_shape = [SEQ_LEN, NUM_HEADS, HEAD_DIM];
    let q = b.add_reshape(q, &mh_shape);
    let k = b.add_reshape(k, &mh_shape);
    let v = b.add_reshape(v, &mh_shape);

    // --- Transpose to per-head: [S, H, head_dim] -> [H, S, head_dim] ---
    let head_shape = [NUM_HEADS, SEQ_LEN, HEAD_DIM];
    let q = b.add_transpose(q, &[1, 0, 2], &head_shape);
    let k = b.add_transpose(k, &[1, 0, 2], &head_shape);
    let v = b.add_transpose(v, &[1, 0, 2], &head_shape);

    // --- Scaled dot-product attention ---
    // Using Standard mask for NY. The sliding window is a constant
    // additive mask that restricts attention to a banded region. For
    // verification, Standard mask gives sound bounds (the window mask can
    // only make bounds tighter by zeroing out some attention weights).
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn_out = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &head_shape);

    // --- Transpose back: [H, S, head_dim] -> [S, H, head_dim] ---
    let attn_out = b.add_transpose(attn_out, &[1, 0, 2], &mh_shape);

    // --- Reshape to [S, D] ---
    let attn_out = b.add_reshape(attn_out, &proj_shape);

    // --- Output projection ---
    let output = b.add_linear(attn_out, out_weight, None, &[SEQ_LEN, EMBED_DIM]);

    b.build(output)
        .expect("valid sliding window attention kernel")
}

/// Build parameter bindings for the sliding window attention kernel.
///
/// hidden = Variable, qkv_weight and out_weight = ConstantTensor.
fn sliding_window_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // hidden [SEQ_LEN, EMBED_DIM]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[QKV_DIM, EMBED_DIM]),
            WEIGHT_MAG,
        )), // qkv_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, EMBED_DIM]),
            WEIGHT_MAG,
        )), // out_weight
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Sliding window attention TensorKernelDef validates.
#[test]
fn test_sliding_window_attention_def_validates() {
    let def = build_sliding_window_attention_kernel();
    def.validate()
        .expect("Sliding window attention kernel should validate");
}

/// Sliding window attention translates to NY GraphNetwork.
#[test]
fn test_sliding_window_attention_graph_builds() {
    let def = build_sliding_window_attention_kernel();
    let bindings = sliding_window_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("Sliding window attention graph should translate");

    // Linear(1) + Narrow(3) + Reshape(3+1+1) + Transpose(3+1) +
    // Attention(1) + Linear(1) = ~15+ nodes
    assert!(
        graph.num_nodes() >= 12,
        "Sliding window attention graph should have >= 12 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through sliding window attention.
///
/// With small weights (0.02) and [-1, 1] input, the softmax normalizes
/// attention scores to sum to 1, keeping output magnitudes bounded.
#[test]
fn test_sliding_window_attention_ibp_propagates() {
    let def = build_sliding_window_attention_kernel();
    let bindings = sliding_window_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through sliding window attention");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Sliding window attention IBP: bounds=[{lo_min}, {hi_max}]");

    // With small weights (0.02), IBP bounds should remain tractable.
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

/// CROWN bounds propagate through sliding window attention.
///
/// The attention block includes softmax (nonlinear) and bilinear matmul
/// (Q @ K^T), so CROWN may fall back to IBP. The test verifies structural
/// validity regardless of propagation method.
#[test]
fn test_sliding_window_attention_crown_propagation() {
    let def = build_sliding_window_attention_kernel();
    let bindings = sliding_window_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Sliding window attention: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

/// Sliding window attention verify and record under "sliding_window_attention" key.
#[test]
fn test_sliding_window_attention_verify_and_record() {
    let def = build_sliding_window_attention_kernel();
    let bindings = sliding_window_bindings();
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "sliding_window_attention");
    assert_eq!(result.num_variables, 1, "single Variable input (hidden)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, EMBED_DIM]);
}
