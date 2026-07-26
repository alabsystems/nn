// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test: `TensorOpKind::Softmax` → NY `SoftmaxLayer`.
//!
//! Part of #738: Softmax tensor op for attention layer support.

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a minimal tensor kernel: input → softmax(axis) → output.
fn softmax_tensor_kernel(shape: &[usize], axis: i32) -> TensorKernelDef {
    TensorKernelDef::new(
        "softmax_test",
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
                TensorOpKind::Softmax {
                    input: TensorNodeId::new(0),
                    axis,
                },
                shape.to_vec(),
            ),
        ],
        TensorNodeId::new(1),
    )
}

#[test]
fn test_softmax_tensor_builds_graph() {
    let def = softmax_tensor_kernel(&[4, 8], -1);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("softmax tensor graph should build");
    // SoftmaxLayer is a single node (no decomposition).
    assert_eq!(graph.num_nodes(), 1, "softmax graph should have 1 node");
}

#[test]
fn test_softmax_tensor_positive_axis() {
    // axis=1 on shape [4, 8] normalizes along the last dimension.
    let def = softmax_tensor_kernel(&[4, 8], 1);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("softmax with positive axis should build");
    assert_eq!(graph.num_nodes(), 1);
}

#[test]
fn test_softmax_tensor_negative_axis() {
    // axis=-1 means last axis, same as axis=1 for a rank-2 tensor.
    let def = softmax_tensor_kernel(&[4, 8], -1);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("softmax with negative axis should build");
    assert_eq!(graph.num_nodes(), 1);
}

#[test]
fn test_softmax_tensor_ibp_bounds_in_unit_interval() {
    // Softmax output is always in [0, 1] per element, summing to 1 along axis.
    // IBP should produce bounds within [0, 1].
    let def = softmax_tensor_kernel(&[4], -1);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("softmax graph should build");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[4]), -2.0f32),
        ArrayD::from_elem(IxDyn(&[4]), 2.0f32),
    )
    .expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();

    for i in 0..4 {
        assert!(
            lo[[i]] >= -1e-6,
            "softmax lower bound [{i}] = {} should be >= 0",
            lo[[i]]
        );
        assert!(
            hi[[i]] <= 1.0 + 1e-6,
            "softmax upper bound [{i}] = {} should be <= 1",
            hi[[i]]
        );
    }
}

#[test]
fn test_softmax_tensor_constant_fold() {
    // Constant scalar: softmax of a single scalar is 1.0.
    // The constant-folded graph outputs AddConstant(1.0), so IBP input must
    // be 0.0 for the result to equal the folded value (0.0 + 1.0 = 1.0).
    let def = softmax_tensor_kernel(&[1], -1);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(0.0)])
        .expect("softmax constant graph should build");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 0.0f32),
    )
    .expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();

    // Constant-folded: softmax of single element = 1.0.
    assert!(
        (lo[[0]] - 1.0).abs() < 1e-5,
        "constant softmax lower = {}, expected 1.0",
        lo[[0]]
    );
    assert!(
        (hi[[0]] - 1.0).abs() < 1e-5,
        "constant softmax upper = {}, expected 1.0",
        hi[[0]]
    );
}

#[test]
fn test_softmax_tensor_block_builder() {
    use nn_dsl::tensor_block_builder::TensorBlockBuilder;

    let mut b = TensorBlockBuilder::new("softmax_block");
    let x = b.add_input("logits", &[8, 64]);
    let sm = b.add_softmax(x, -1, &[8, 64]);
    let def = b.build(sm).expect("valid graph");

    assert_eq!(def.name, "softmax_block");
    assert_eq!(def.nodes.len(), 2);
    assert_eq!(def.output, TensorNodeId::new(1));
    assert_eq!(def.nodes[1].shape, vec![8, 64]);
    assert!(matches!(
        &def.nodes[1].kind,
        TensorOpKind::Softmax { input, axis } if *input == TensorNodeId::new(0) && *axis == -1
    ));
}

#[test]
fn test_softmax_tensor_weight_rejected() {
    let def = softmax_tensor_kernel(&[4], -1);
    let weights = ArrayD::from_elem(IxDyn(&[4]), 1.0f32);
    let result = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantTensor(weights)]);
    assert!(
        result.is_err(),
        "softmax on WeightTensor should be rejected"
    );
}

#[test]
fn test_softmax_tensor_3d_shape() {
    // 3D tensor: batch × sequence × features. Softmax on last axis (features).
    let def = softmax_tensor_kernel(&[2, 4, 8], -1);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("3D softmax should build");
    assert_eq!(graph.num_nodes(), 1);
}

#[test]
fn test_softmax_validation_rejects_scalar() {
    // Softmax on a scalar (rank 0) should fail validation.
    let def = TensorKernelDef::new(
        "bad_softmax",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![1],
                },
                vec![1],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Softmax {
                    input: TensorNodeId::new(0),
                    axis: 5, // out of bounds for rank 1
                },
                vec![1],
            ),
        ],
        TensorNodeId::new(1),
    );
    let result = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]);
    assert!(result.is_err(), "softmax axis=5 on rank-1 should fail");
}
