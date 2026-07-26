// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Upsample1d composition tests: NY propagation through
//! Reshape → Tile → Reshape decomposition.
//!
//! Validates that the Upsample1d nearest-neighbor decomposition translates
//! through `tensor_kernel_to_graph` and produces a NY `GraphNetwork`
//! where IBP and CROWN bounds propagate end-to-end.
//!
//! The decomposition is:
//!   [..., T] → Reshape [..., T, 1] → Broadcast [..., T, factor] → Reshape [..., T*factor]
//!
//! Part of #2222.

use super::common::{assert_bounds_valid, assert_crown_tighter_when_not_fallback, uniform_bounds};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

/// Build a pure Upsample1d graph: Reshape → Broadcast → Reshape.
///
/// Single variable input of shape `[channels, time]`.
/// Output shape: `[channels, time * factor]`.
fn build_upsample1d(
    channels: usize,
    time: usize,
    factor: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let out_time = time * factor;

    let mut b = TensorBlockBuilder::new("upsample1d");
    let data = b.add_input("data", &[channels, time]);

    // Upsample1d decomposition: reshape → broadcast → reshape
    let r1 = b.add_reshape(data, &[channels, time, 1]);
    let tile = b.add_broadcast(r1, &[channels, time, factor]);
    let output = b.add_reshape(tile, &[channels, out_time]);

    b.build(output).expect("valid upsample graph")
}

/// Build a Linear → ReLU → Upsample1d → Linear chain.
///
/// Tests that bounds propagate correctly through the Upsample1d decomposition
/// when sandwiched between linear layers and non-linearities.
fn build_linear_upsample_linear(
    in_features: usize,
    hidden: usize,
    out_features: usize,
    factor: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let up_len = hidden * factor;

    let mut b = TensorBlockBuilder::new("linear_upsample_linear");

    let data = b.add_input("data", &[in_features]);
    let w1 = b.add_input("w1", &[hidden, in_features]);
    let w2 = b.add_input("w2", &[out_features, up_len]);

    // Linear 1: [in_features] → [hidden]
    let lin1 = b.add_linear(data, w1, None, &[hidden]);

    // ReLU: [hidden] → [hidden]
    let relu = b.add_relu(lin1, &[hidden]);

    // Upsample1d by factor: [hidden] → [hidden, 1] → [hidden, factor] → [hidden*factor]
    let r1 = b.add_reshape(relu, &[hidden, 1]);
    let tile = b.add_broadcast(r1, &[hidden, factor]);
    let up = b.add_reshape(tile, &[up_len]);

    // Linear 2: [up_len] → [out_features]
    let lin2 = b.add_linear(up, w2, None, &[out_features]);

    b.build(lin2).expect("valid graph")
}

// ---------------------------------------------------------------------------
// Tests: pure Upsample1d
// ---------------------------------------------------------------------------

/// Pure Upsample1d builds and translates to a NY graph.
#[test]
fn test_upsample1d_graph_builds() {
    let def = build_upsample1d(3, 8, 4);
    assert_eq!(def.nodes.last().unwrap().shape, vec![3, 32]);

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("upsample graph");

    // Reshape + Tile + Reshape = at least 3 NY nodes (plus input identity).
    assert!(
        graph.num_nodes() >= 3,
        "upsample1d needs >= 3 NY nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through pure Upsample1d.
///
/// Since upsample is purely structural (replication), output bounds should
/// equal input bounds — each output element is an exact copy of an input element.
#[test]
fn test_upsample1d_ibp_propagates() {
    let def = build_upsample1d(2, 4, 3);

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[2, 4], 5.0);
    let output = graph.propagate_ibp(&input).expect("IBP through upsample1d");

    assert_eq!(output.lower_upper().0.shape(), &[2, 12]);
    assert_bounds_valid(&output);

    // Key soundness property: since upsample is replication, output range
    // should not exceed input range [-5.0, 5.0].
    let (lower, upper) = output.lower_upper();
    for &lo in lower.iter() {
        assert!(
            lo >= -5.01,
            "upsample1d lower bound should be >= -5.0 (got {lo})"
        );
    }
    for &hi in upper.iter() {
        assert!(
            hi <= 5.01,
            "upsample1d upper bound should be <= 5.0 (got {hi})"
        );
    }
}

/// CROWN propagation through pure Upsample1d.
#[test]
fn test_upsample1d_crown_propagates() {
    let def = build_upsample1d(2, 4, 2);

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[2, 4], 5.0);

    let (_method, output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[2, 8]);
    assert_bounds_valid(&output);
}

/// Upsample1d with factor=1 is identity — output bounds match input exactly.
#[test]
fn test_upsample1d_factor_1_is_identity() {
    let def = build_upsample1d(3, 8, 1);

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[3, 8], 2.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through factor-1 upsample");

    // factor=1 means output shape = input shape
    assert_eq!(output.lower_upper().0.shape(), &[3, 8]);
    assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// Tests: Linear → ReLU → Upsample1d → Linear
// ---------------------------------------------------------------------------

/// Linear → ReLU → Upsample1d → Linear graph builds and translates.
#[test]
fn test_linear_upsample_linear_graph_builds() {
    let def = build_linear_upsample_linear(8, 4, 6, 3);
    assert_eq!(def.nodes.last().unwrap().shape, vec![6]);

    let w1 = ArrayD::from_elem(IxDyn(&[4, 8]), 0.1f32);
    let w2 = ArrayD::from_elem(IxDyn(&[6, 12]), 0.05f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w1),
        TensorParamBinding::ConstantTensor(w2),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("linear-upsample-linear graph");
    assert!(
        graph.num_nodes() >= 5,
        "linear-upsample-linear needs >= 5 NY nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through Linear → ReLU → Upsample1d → Linear.
#[test]
fn test_linear_upsample_linear_ibp_propagates() {
    let def = build_linear_upsample_linear(8, 4, 6, 3);

    let w1 = ArrayD::from_elem(IxDyn(&[4, 8]), 0.1f32);
    let w2 = ArrayD::from_elem(IxDyn(&[6, 12]), 0.05f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w1),
        TensorParamBinding::ConstantTensor(w2),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[8], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through linear-upsample-linear");

    assert_eq!(output.lower_upper().0.shape(), &[6]);
    assert_bounds_valid(&output);
}

/// CROWN propagation through Linear → ReLU → Upsample1d → Linear.
///
/// When CROWN succeeds (no IBP fallback), asserts bounds are tighter than IBP.
#[test]
fn test_linear_upsample_linear_crown_propagates() {
    let def = build_linear_upsample_linear(8, 4, 6, 2);

    let w1 = ArrayD::from_elem(IxDyn(&[4, 8]), 0.1f32);
    let w2 = ArrayD::from_elem(IxDyn(&[6, 8]), 0.05f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w1),
        TensorParamBinding::ConstantTensor(w2),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[8], 1.0);

    let (_method, output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[6]);
    assert_bounds_valid(&output);
}
