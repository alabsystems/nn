// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `TensorOpKind::Narrow` → NY `SliceLayer`.
//!
//! Tests:
//! - Single variable IBP: bounds preserved through slice
//! - Constant input passes through unchanged
//! - Validation rejects out-of-bounds narrow parameters

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a simple Narrow kernel: input [C, T] → narrow(axis=1, start, length) → [C, length].
fn narrow_kernel(channels: usize, time: usize, start: usize, length: usize) -> TensorKernelDef {
    let in_shape = vec![channels, time];
    let out_shape = vec![channels, length];
    TensorKernelDef::new(
        "narrow_test",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: in_shape.clone(),
                },
                in_shape,
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Narrow {
                    input: TensorNodeId::new(0),
                    axis: 1,
                    start,
                    length,
                },
                out_shape,
            ),
        ],
        TensorNodeId::new(1),
    )
}

/// Narrow of a variable input builds a valid NY graph.
#[test]
fn test_narrow_variable_builds_graph() {
    let def = narrow_kernel(4, 16, 2, 8);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("narrow graph should build");
    assert!(
        graph.num_nodes() >= 1,
        "graph should have at least the slice node"
    );
}

/// IBP bounds propagate correctly through a Narrow operation.
///
/// SliceLayer preserves bounds element-wise for the selected range.
/// If input bounds are [-1, 3] uniformly, the narrow output should
/// also have bounds [-1, 3].
#[test]
fn test_narrow_ibp_bounds_preserved() {
    let ch = 2;
    let time = 8;
    let start = 2;
    let length = 4;
    let def = narrow_kernel(ch, time, start, length);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[ch, time]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[ch, time]), 3.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP through narrow");

    let out_lower = output.lower();
    let out_upper = output.upper();

    // Output shape: [ch, length] = [2, 4]
    assert_eq!(
        out_lower.len(),
        ch * length,
        "output should have ch*length = {} elements, shape={:?}",
        ch * length,
        out_lower.shape()
    );

    for &v in out_lower.iter() {
        assert!((v - (-1.0)).abs() < 1e-5, "expected lower ~-1.0, got {v}");
    }
    for &v in out_upper.iter() {
        assert!((v - 3.0).abs() < 1e-5, "expected upper ~3.0, got {v}");
    }
}

/// Narrow of a constant input passes the constant through unchanged.
#[test]
fn test_narrow_constant_passthrough() {
    let def = narrow_kernel(2, 8, 0, 4);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(7.0)])
        .expect("constant narrow should succeed");

    let lower = ArrayD::from_elem(IxDyn(&[2, 4]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 4]), 0.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");

    for &v in output.lower().iter() {
        assert!(
            (v - 7.0).abs() < 1e-5,
            "expected constant fold result 7.0, got {v}"
        );
    }
}

/// Validation rejects narrow with start+length exceeding dimension.
#[test]
fn test_narrow_validation_rejects_out_of_bounds() {
    // start=6, length=5, dim=8: 6+5=11 > 8
    let in_shape = vec![2, 8];
    let def = TensorKernelDef::new(
        "bad_narrow",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: in_shape.clone(),
                },
                in_shape,
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Narrow {
                    input: TensorNodeId::new(0),
                    axis: 1,
                    start: 6,
                    length: 5,
                },
                vec![2, 5],
            ),
        ],
        TensorNodeId::new(1),
    );
    let result = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]);
    assert!(
        result.is_err(),
        "Narrow with start+length > dim should fail validation"
    );
}

/// Validation rejects narrow with zero length.
#[test]
fn test_narrow_validation_rejects_zero_length() {
    let in_shape = vec![2, 8];
    let def = TensorKernelDef::new(
        "zero_narrow",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: in_shape.clone(),
                },
                in_shape,
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Narrow {
                    input: TensorNodeId::new(0),
                    axis: 1,
                    start: 0,
                    length: 0,
                },
                vec![2, 0],
            ),
        ],
        TensorNodeId::new(1),
    );
    let result = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]);
    assert!(
        result.is_err(),
        "Narrow with length=0 should fail validation"
    );
}
