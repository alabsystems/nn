// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test: `TensorOpKind::Sigmoid` → NY `SigmoidLayer`.
//!
//! Part of #645 AC5: sigmoid graph translation produces correct bounds.

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a minimal tensor kernel: input → sigmoid → output.
fn sigmoid_tensor_kernel(shape: &[usize]) -> TensorKernelDef {
    TensorKernelDef::new(
        "sigmoid_test",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: shape.to_vec(),
                },
                shape.to_vec(),
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Sigmoid {
                    input: TensorNodeId::new(0),
                },
                shape.to_vec(),
            ),
        ],
        TensorNodeId::new(1),
    )
}

#[test]
fn test_sigmoid_tensor_builds_graph() {
    let def = sigmoid_tensor_kernel(&[4, 32]);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("sigmoid tensor graph should build");
    // SigmoidLayer is a single node (no decomposition).
    assert_eq!(graph.num_nodes(), 1, "sigmoid graph should have 1 node");
}

#[test]
fn test_sigmoid_tensor_ibp_bounds_correct() {
    // Sigmoid is monotonically increasing: sigmoid(-2) ≈ 0.119, sigmoid(2) ≈ 0.881.
    // IBP should produce bounds [sigmoid(-2), sigmoid(2)].
    let def = sigmoid_tensor_kernel(&[1]);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("sigmoid graph should build");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -2.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 2.0f32),
    )
    .expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();

    let expected_lo = 1.0 / (1.0 + (2.0f32).exp()); // sigmoid(-2)
    let expected_hi = 1.0 / (1.0 + (-2.0f32).exp()); // sigmoid(2)

    assert!(
        (lo[[0]] - expected_lo).abs() < 1e-4,
        "lower bound {:.6} should be near sigmoid(-2) = {:.6}",
        lo[[0]],
        expected_lo
    );
    assert!(
        (hi[[0]] - expected_hi).abs() < 1e-4,
        "upper bound {:.6} should be near sigmoid(2) = {:.6}",
        hi[[0]],
        expected_hi
    );

    // Sigmoid output is always in (0, 1)
    assert!(lo[[0]] > 0.0, "sigmoid lower bound must be > 0");
    assert!(hi[[0]] < 1.0, "sigmoid upper bound must be < 1");
}

#[test]
fn test_sigmoid_tensor_constant_fold() {
    // When input is constant, sigmoid should constant-fold.
    let def = sigmoid_tensor_kernel(&[1]);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(0.0)])
        .expect("sigmoid constant graph should build");

    // Constant-folded sigmoid(0) = 0.5 — the graph should produce tight bounds.
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 0.0f32),
    )
    .expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();

    // Constant-folded: output should be exactly 0.5 (within float tolerance).
    assert!(
        (lo[[0]] - 0.5).abs() < 1e-5,
        "constant sigmoid(0) lower = {}, expected 0.5",
        lo[[0]]
    );
    assert!(
        (hi[[0]] - 0.5).abs() < 1e-5,
        "constant sigmoid(0) upper = {}, expected 0.5",
        hi[[0]]
    );
}

#[test]
fn test_sigmoid_tensor_block_builder() {
    use nn_dsl::tensor_block_builder::TensorBlockBuilder;

    let mut b = TensorBlockBuilder::new("sigmoid_block");
    let x = b.add_input("x", &[8, 16]);
    let sig = b.add_sigmoid(x, &[8, 16]);
    let def = b.build(sig).expect("valid graph");

    assert_eq!(def.name, "sigmoid_block");
    assert_eq!(def.nodes.len(), 2);
    assert_eq!(def.output, TensorNodeId::new(1));
    assert_eq!(def.nodes[1].shape, vec![8, 16]);
    assert!(
        matches!(&def.nodes[1].kind, TensorOpKind::Sigmoid { input } if *input == TensorNodeId::new(0))
    );
}
