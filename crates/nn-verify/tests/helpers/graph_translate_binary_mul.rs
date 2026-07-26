// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `TensorOpKind::BinaryMul` → NY `MulBinaryLayer`.
//!
//! Tests:
//! - Two-variable IBP bounds propagation: McCormick envelope for bounds(a*b)
//! - Constant-folding: BinaryMul of two constants
//! - Mixed: variable * constant
//! - Validation rejects shape mismatch

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a simple binary_mul kernel: two inputs of the same shape, output = left * right.
fn binary_mul_kernel(name: &str, shape: &[usize]) -> TensorKernelDef {
    TensorKernelDef::new(
        name,
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "left".into(),
                    shape: shape.to_vec(),
                },
                shape.to_vec(),
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "right".into(),
                    shape: shape.to_vec(),
                },
                shape.to_vec(),
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::BinaryMul {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(1),
                },
                shape.to_vec(),
            ),
        ],
        TensorNodeId::new(2),
    )
}

#[test]
fn test_binary_mul_two_variables_builds_graph() {
    let def = binary_mul_kernel("mul_test", &[4, 32]);
    let graph = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    )
    .expect("binary mul graph should build");
    // 2 variables → multi-variable setup adds SliceLayer nodes + binary MulBinaryLayer
    assert!(
        graph.num_nodes() >= 1,
        "graph should have at least the mul node"
    );
}

#[test]
fn test_binary_mul_ibp_bounds_correct() {
    // IBP bounds for a * b (McCormick envelope):
    //   lower = min(a_lo*b_lo, a_lo*b_up, a_up*b_lo, a_up*b_up)
    //   upper = max(a_lo*b_lo, a_lo*b_up, a_up*b_lo, a_up*b_up)
    let shape = &[2, 4];
    let def = binary_mul_kernel("ibp_mul", shape);
    let graph = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    )
    .expect("build mul graph");

    // Multi-variable: inputs are stacked along axis 0 → shape [2, 2, 4]
    let mut lower = ArrayD::from_elem(IxDyn(&[2, 2, 4]), 0.0f32);
    let mut upper = ArrayD::from_elem(IxDyn(&[2, 2, 4]), 0.0f32);
    // left input (slice 0): bounds [-1, 3]
    for i in 0..2 {
        for j in 0..4 {
            lower[[0, i, j]] = -1.0;
            upper[[0, i, j]] = 3.0;
        }
    }
    // right input (slice 1): bounds [2, 5]
    for i in 0..2 {
        for j in 0..4 {
            lower[[1, i, j]] = 2.0;
            upper[[1, i, j]] = 5.0;
        }
    }
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");

    // Products: (-1)*2=-2, (-1)*5=-5, 3*2=6, 3*5=15
    // Expected: lower = min(-2,-5,6,15) = -5, upper = max(-2,-5,6,15) = 15
    let out_lower = output.lower();
    let out_upper = output.upper();
    for &v in out_lower.iter() {
        assert!((v - (-5.0)).abs() < 1e-4, "expected lower ~-5.0, got {v}");
    }
    for &v in out_upper.iter() {
        assert!((v - 15.0).abs() < 1e-4, "expected upper ~15.0, got {v}");
    }
}

#[test]
fn test_binary_mul_constant_fold() {
    // BinaryMul of two constant scalars should fold to a constant output.
    let shape = &[2, 4];
    let def = binary_mul_kernel("const_mul", shape);
    let graph = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::ConstantScalar(3.0),
            TensorParamBinding::ConstantScalar(7.0),
        ],
    )
    .expect("constant-fold binary mul should succeed");

    // Constant output → graph wraps it in AddConstant(21.0) identity
    let lower = ArrayD::from_elem(IxDyn(&[2, 4]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 4]), 0.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    for &v in output.lower().iter() {
        assert!(
            (v - 21.0).abs() < 1e-5,
            "expected constant fold result 21.0, got {v}"
        );
    }
}

#[test]
fn test_binary_mul_variable_times_constant() {
    // One variable * one constant: output = var * 5.0
    let shape = &[2, 4];
    let def = binary_mul_kernel("mixed_mul", shape);
    let graph = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantScalar(5.0),
        ],
    )
    .expect("mixed binary mul should succeed");

    let lower = ArrayD::from_elem(IxDyn(&[2, 4]), -3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 4]), 7.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    // Expected: lower = min(-3*5, 7*5) = -15, upper = max(-3*5, 7*5) = 35
    for &v in output.lower().iter() {
        assert!((v - (-15.0)).abs() < 1e-4, "expected lower ~-15.0, got {v}");
    }
    for &v in output.upper().iter() {
        assert!((v - 35.0).abs() < 1e-4, "expected upper ~35.0, got {v}");
    }
}

#[test]
fn test_binary_mul_validation_rejects_shape_mismatch() {
    let def = TensorKernelDef::new(
        "bad_mul",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".into(),
                    shape: vec![2, 4],
                },
                vec![2, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".into(),
                    shape: vec![3, 4],
                },
                vec![3, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::BinaryMul {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(1),
                },
                vec![2, 4],
            ),
        ],
        TensorNodeId::new(2),
    );
    let result = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    );
    assert!(
        result.is_err(),
        "BinaryMul with shape mismatch should fail validation"
    );
}
