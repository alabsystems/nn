// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: cross-attention MHA → NY.
//!
//! Validates that `add_multi_head_cross_attention()` translates to NY
//! and propagates IBP bounds correctly.
//!
//! Part of #779 Phase D.

#![allow(dead_code)]

use super::common;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Equal-sequence cross-attention: Q and KV have the same sequence length.
const SEQ_LEN: usize = 4;
const MODEL_DIM: usize = 8;
const NUM_HEADS: usize = 2;

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Build a cross-MHA kernel: Q is Variable, KV is a separate input.
fn build_cross_mha_kernel(name: &str) -> TensorKernelDef {
    let d = MODEL_DIM;
    let mut b = TensorBlockBuilder::new(name);

    let q_input = b.add_input("q_input", &[SEQ_LEN, d]);
    let kv_input = b.add_input("kv_input", &[SEQ_LEN, d]);
    let q_w = b.add_input("q_weight", &[d, d]);
    let k_w = b.add_input("k_weight", &[d, d]);
    let v_w = b.add_input("v_weight", &[d, d]);
    let out_w = b.add_input("out_weight", &[d, d]);

    let out = b
        .add_multi_head_cross_attention(
            q_input,
            kv_input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ_LEN, d],
        )
        .expect("valid cross-MHA");
    b.build(out).expect("valid kernel")
}

/// Bindings: q_input=Variable, kv_input=ConstantTensor, weights=ConstantTensor.
fn cross_mha_bindings() -> Vec<TensorParamBinding> {
    let d = MODEL_DIM;
    let w_small = 0.02f32;

    let kv_const = ArrayD::from_elem(IxDyn(&[SEQ_LEN, d]), 0.1f32);
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), w_small);

    vec![
        TensorParamBinding::Variable,                       // q_input
        TensorParamBinding::ConstantTensor(kv_const),       // kv_input
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight
        TensorParamBinding::ConstantTensor(w_proj),         // out_weight
    ]
}

/// Input bounds for Q Variable [SEQ, D] in [-1, 1].
fn cross_mha_input_bounds() -> BoundedTensor {
    common::uniform_bounds(&[SEQ_LEN, MODEL_DIM], 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Cross-MHA translates to a valid NY GraphNetwork.
#[test]
fn test_cross_mha_graph_builds() {
    let def = build_cross_mha_kernel("xmha_build");
    let bindings = cross_mha_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("cross-MHA graph must build");

    assert!(
        graph.num_nodes() >= 5,
        "cross-MHA should produce multiple nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through cross-MHA.
#[test]
fn test_cross_mha_ibp_propagates() {
    let def = build_cross_mha_kernel("xmha_ibp");
    let bindings = cross_mha_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("cross-MHA graph");

    let input = cross_mha_input_bounds();
    let output = graph.propagate_ibp(&input).expect("IBP through cross-MHA");
    let (lo, _hi) = output.lower_upper();

    // Output shape: [SEQ_LEN, D]
    assert_eq!(lo.shape(), &[SEQ_LEN, MODEL_DIM], "output shape [S, D]");

    common::assert_bounds_valid(&output);
}

/// Cross-MHA IBP bounds have reasonable width.
#[test]
fn test_cross_mha_bounds_width() {
    let def = build_cross_mha_kernel("xmha_width");
    let bindings = cross_mha_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("cross-MHA graph");

    let input = cross_mha_input_bounds();
    let output = graph.propagate_ibp(&input).expect("IBP through cross-MHA");
    let (lo, hi) = output.lower_upper();

    let max_width = lo
        .iter()
        .zip(hi.iter())
        .map(|(l, u)| (u - l).abs())
        .fold(0.0f32, f32::max);

    // Small weights (0.02) and [-1, 1] input should keep bounds tight.
    assert!(
        max_width < 100.0,
        "cross-MHA IBP bounds should be tight, got max width {max_width}"
    );
}

/// Cross-MHA output preserves sequence length.
#[test]
fn test_cross_mha_output_shape() {
    let def = build_cross_mha_kernel("xmha_shape");
    let bindings = cross_mha_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = cross_mha_input_bounds();
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, _hi) = output.lower_upper();

    // Output must have same sequence length and model dimension as input
    assert_eq!(lo.shape()[0], SEQ_LEN, "output seq len must match input");
    assert_eq!(lo.shape()[1], MODEL_DIM, "output dim must match input");
}

// ---------------------------------------------------------------------------
// CROWN propagation tests
// ---------------------------------------------------------------------------

/// CROWN produces tighter-or-equal bounds than IBP on cross-MHA.
#[test]
fn test_cross_mha_crown_tighter_than_ibp() {
    let def = build_cross_mha_kernel("xmha_crown");
    let bindings = cross_mha_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("cross-MHA graph");

    let input = cross_mha_input_bounds();
    let (method, output, fallback_reason) =
        common::assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("Cross-MHA: method={method:?}, fallback={fallback_reason:?}");
    common::assert_bounds_valid(&output);
}
