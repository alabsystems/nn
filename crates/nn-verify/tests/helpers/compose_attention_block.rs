// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: tensor-level composition of scaled dot-product attention.
//!
//! Attention = Softmax(Q @ K^T / sqrt(d_k)) @ V
//!
//! Validates that the three-op pipeline (MatMul -> Softmax -> MatMul) composes
//! correctly through `tensor_kernel_to_graph` and produces a single NY
//! `GraphNetwork` where IBP bounds propagate end-to-end.
//!
//! This is the core compute pattern of all dvoice transformer models:
//! Qwen3-TTS, CosyVoice3, DiT, Demucs, CAM++.
//!
//! Part of #737, Part of #741, Part of #729.

use super::common;
use super::common::assert_crown_tighter_when_not_fallback;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Attention block builder
// ---------------------------------------------------------------------------

/// Build a scaled dot-product attention block as a multi-op TensorKernelDef.
///
/// Attention(Q, K, V) = Softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// Q shape: [seq_len, d_k]
/// K shape: [seq_len, d_k]
/// V shape: [seq_len, d_v]
/// Output shape: [seq_len, d_v]
fn build_attention_block(
    name: &str,
    seq_len: usize,
    d_k: usize,
    d_v: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);

    // Inputs: Q, K, V (all bounded variables for verification)
    let q = b.add_input("query", &[seq_len, d_k]);
    let k = b.add_input("key", &[seq_len, d_k]);
    let v = b.add_input("value", &[seq_len, d_v]);

    // scores = Q @ K^T / sqrt(d_k)
    let scale = 1.0 / (d_k as f32).sqrt();
    let scores = b.add_matmul(q, k, true, Some(scale), &[seq_len, seq_len]);

    // attn_weights = Softmax(scores, axis=-1)
    let attn_weights = b.add_softmax(scores, -1, &[seq_len, seq_len]);

    // output = attn_weights @ V
    let output = b.add_matmul(attn_weights, v, false, None, &[seq_len, d_v]);

    b.build(output).expect("valid graph")
}

/// Build attention without scale (simpler case for testing).
fn build_attention_unscaled(
    name: &str,
    seq_len: usize,
    d_k: usize,
    d_v: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);

    let q = b.add_input("query", &[seq_len, d_k]);
    let k = b.add_input("key", &[seq_len, d_k]);
    let v = b.add_input("value", &[seq_len, d_v]);

    let scores = b.add_matmul(q, k, true, None, &[seq_len, seq_len]);
    let attn_weights = b.add_softmax(scores, -1, &[seq_len, seq_len]);
    let output = b.add_matmul(attn_weights, v, false, None, &[seq_len, d_v]);

    b.build(output).expect("valid graph")
}

// ---------------------------------------------------------------------------
// Graph construction tests
// ---------------------------------------------------------------------------

/// Attention block translates into a valid NY GraphNetwork.
#[test]
fn test_attention_block_graph_builds() {
    let def = build_attention_block("attn_basic", 4, 3, 3);
    assert_eq!(def.nodes.len(), 6, "3 inputs + scores + softmax + output");

    let bindings = vec![
        TensorParamBinding::Variable, // Q
        TensorParamBinding::Variable, // K
        TensorParamBinding::Variable, // V
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("attention graph must build");
    assert!(
        graph.num_nodes() >= 3,
        "need at least MatMul, Softmax, MatMul nodes"
    );
}

/// Unscaled attention variant builds.
#[test]
fn test_attention_unscaled_graph_builds() {
    let def = build_attention_unscaled("attn_unscaled", 4, 3, 3);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("unscaled attention graph");
    assert!(graph.num_nodes() >= 3);
}

// ---------------------------------------------------------------------------
// IBP bounds propagation tests
// ---------------------------------------------------------------------------

/// IBP bounds propagate through the full attention pipeline.
///
/// With 3 variable inputs (Q, K, V), bounds are stacked along a leading axis:
/// bounds shape = [3, seq_len, d]. Output retains the leading dimension: [1, seq_len, d].
#[test]
fn test_attention_ibp_bounds_propagate() {
    let (seq_len, d) = (4, 3);
    let def = build_attention_block("attn_ibp", seq_len, d, d);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("attention graph");

    // Multi-variable bounds: [3, seq_len, d] where axis 0 selects Q/K/V.
    let mut lower = ArrayD::from_elem(IxDyn(&[3, seq_len, d]), -1.0f32);
    let mut upper = ArrayD::from_elem(IxDyn(&[3, seq_len, d]), 1.0f32);
    // Q: [-1, 1], K: [-1, 1], V: [-0.5, 0.5] (narrower range for V)
    for i in 0..seq_len {
        for j in 0..d {
            lower[[2, i, j]] = -0.5;
            upper[[2, i, j]] = 0.5;
        }
    }

    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through attention block");
    let (lo, _hi) = output.lower_upper();

    // Each variable enters its subgraph at its TRUE declared rank (#358 flat
    // per-variable Slice+Reshape harness), so the output keeps its natural
    // [seq_len, d_v] shape with no leading stacking dimension.
    assert_eq!(
        lo.shape(),
        &[seq_len, d],
        "output shape [seq_len, d_v] at natural rank"
    );

    common::assert_bounds_valid(&output);
}

/// IBP bounds for small attention: seq_len=2, d_k=d_v=2.
///
/// With tiny dimensions we can reason about expected bounds more concretely.
/// Softmax output is in [0, 1] and sums to 1 per row. The final MatMul with V
/// produces a convex combination of V rows, so output must be within V's bounds.
#[test]
fn test_attention_ibp_small() {
    let (seq_len, d) = (2, 2);
    let def = build_attention_block("attn_small", seq_len, d, d);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("small attention graph");

    // Q, K: [-2, 2], V: [0, 1]
    let mut lower = ArrayD::from_elem(IxDyn(&[3, seq_len, d]), 0.0f32);
    let mut upper = ArrayD::from_elem(IxDyn(&[3, seq_len, d]), 0.0f32);
    for i in 0..seq_len {
        for j in 0..d {
            lower[[0, i, j]] = -2.0;
            upper[[0, i, j]] = 2.0;
            lower[[1, i, j]] = -2.0;
            upper[[1, i, j]] = 2.0;
            lower[[2, i, j]] = 0.0;
            upper[[2, i, j]] = 1.0;
        }
    }

    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through small attention");
    let (lo, _hi) = output.lower_upper();

    // Output keeps its natural [seq_len, d_v] rank (#358 flat per-variable harness).
    assert_eq!(lo.shape(), &[seq_len, d]);

    common::assert_bounds_valid(&output);

    // Note: IBP through MatMul+Softmax+MatMul will produce wide bounds
    // (IBP is known to be loose for multi-op chains, see #697). We verify
    // structural correctness (finite, valid) rather than tight numerical bounds.
}

// ---------------------------------------------------------------------------
// Dvoice-scale tests
// ---------------------------------------------------------------------------

/// Dvoice attention scale: seq_len=16, d_k=8.
///
/// Simulates one head of Demucs attention at a small sequence length.
/// Verifies that the pipeline handles realistic dimensions.
#[test]
fn test_attention_dvoice_scale() {
    let (seq_len, d_k) = (16, 8);
    let def = build_attention_block("attn_dvoice", seq_len, d_k, d_k);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dvoice attention graph");

    // Q, K, V all in [-1, 1] (typical after LayerNorm)
    let input = common::uniform_bounds(&[3, seq_len, d_k], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dvoice attention");
    let (lo, _hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[seq_len, d_k], "output shape");

    common::assert_bounds_valid(&output);
}

/// Attention with larger model dimension: seq_len=8, d=16.
///
/// Tests that attention scales to larger feature dimensions typical of
/// smaller transformer heads.
#[test]
fn test_attention_larger_dimension() {
    let (seq_len, d) = (8, 16);
    let def = build_attention_block("attn_large_d", seq_len, d, d);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("larger attention graph");

    // Q, K, V in [-0.5, 0.5] (post-LayerNorm typical range)
    let input = common::uniform_bounds(&[3, seq_len, d], 0.5);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through larger attention");
    let (lo, _hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[seq_len, d], "output shape");

    common::assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// Structural tests
// ---------------------------------------------------------------------------

/// Verify the attention TensorKernelDef structure.
#[test]
fn test_attention_kernel_def_structure() {
    let def = build_attention_block("attn_struct", 4, 3, 3);
    assert_eq!(def.name, "attn_struct");
    assert_eq!(def.nodes.len(), 6, "3 inputs + 3 ops");

    // Verify shapes through the pipeline.
    assert_eq!(def.nodes[0].shape, vec![4, 3], "Q shape");
    assert_eq!(def.nodes[1].shape, vec![4, 3], "K shape");
    assert_eq!(def.nodes[2].shape, vec![4, 3], "V shape");
    assert_eq!(def.nodes[3].shape, vec![4, 4], "scores shape [seq, seq]");
    assert_eq!(
        def.nodes[4].shape,
        vec![4, 4],
        "attn_weights shape [seq, seq]"
    );
    assert_eq!(def.nodes[5].shape, vec![4, 3], "output shape [seq, d_v]");
}

// ---------------------------------------------------------------------------
// CROWN propagation tests
// ---------------------------------------------------------------------------

/// CROWN produces tighter-or-equal bounds than IBP on attention block.
#[test]
fn test_attention_crown_tighter_than_ibp() {
    let (seq_len, d) = (2, 2);
    let def = build_attention_block("attn_crown", seq_len, d, d);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("attention graph");

    // Q, K, V all in [-1, 1]
    let input = common::uniform_bounds(&[3, seq_len, d], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("Attention: method={method:?}, fallback={fallback_reason:?}");
    common::assert_bounds_valid(&output);
}
