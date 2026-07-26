// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for tensor-level IR core validation: graph structure,
//! reduce ops, and elementwise ops.
//!
//! Structural tests (forward ref, composition, pretty-print) extracted to
//! `tensor_ir_validation_structural.rs` (#557, 500-line limit).

use nn_dsl::test_kernels::{square_kernel, sub_kernel};
use nn_dsl::{ReduceOp, TensorIRError, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};

#[test]
fn test_simple_reduce_mean_validates() {
    let def = TensorKernelDef::new(
        "mean_test",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![4, 32, 128],
                },
                vec![4, 32, 128],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reduce {
                    op: ReduceOp::Mean,
                    input: TensorNodeId::new(0),
                    axis: 2,
                    keepdim: false,
                },
                vec![4, 32],
            ),
        ],
        TensorNodeId::new(1),
    );

    def.validate().expect("simple reduce_mean should validate");
}

#[test]
fn test_reduce_axis_out_of_bounds() {
    let def = TensorKernelDef::new(
        "bad_axis",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![4, 32],
                },
                vec![4, 32],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reduce {
                    op: ReduceOp::Sum,
                    input: TensorNodeId::new(0),
                    axis: 5,
                    keepdim: false,
                },
                vec![4],
            ),
        ],
        TensorNodeId::new(1),
    );

    let err = def.validate().expect_err("axis 5 on 2D shape should fail");
    assert!(
        matches!(err, TensorIRError::ReduceAxisOutOfBounds { axis: 5, .. }),
        "expected ReduceAxisOutOfBounds, got: {err:?}"
    );
}

#[test]
fn test_elementwise_param_mismatch() {
    let def = TensorKernelDef::new(
        "mismatch",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".to_string(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Elementwise {
                    kernel: square_kernel(),
                    inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                },
                vec![4],
            ),
        ],
        TensorNodeId::new(2),
    );

    let err = def
        .validate()
        .expect_err("param count mismatch should fail");
    assert!(
        matches!(
            err,
            TensorIRError::ElementwiseParamMismatch {
                expected: 1,
                got: 2
            }
        ),
        "expected ElementwiseParamMismatch, got: {err:?}"
    );
}

#[test]
fn test_elementwise_shape_mismatch_rejected() {
    // Two inputs with different shapes: [4] vs [8]. sub_kernel expects 2 params.
    let def = TensorKernelDef::new(
        "ew_shape_mismatch",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".to_string(),
                    shape: vec![8],
                },
                vec![8],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Elementwise {
                    kernel: sub_kernel(),
                    inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                },
                vec![4],
            ),
        ],
        TensorNodeId::new(2),
    );

    let err = def
        .validate()
        .expect_err("elementwise with mismatched input shapes should fail");
    assert!(
        matches!(
            err,
            TensorIRError::ElementwiseShapeMismatch {
                ref expected,
                ref found,
                index: 1,
            } if *expected == vec![4] && *found == vec![8]
        ),
        "expected ElementwiseShapeMismatch([4] vs [8] at index 1), got: {err:?}"
    );
}

#[test]
fn test_elementwise_rank_mismatch_rejected() {
    // Two inputs with different ranks: [4, 32] vs [4, 32, 128].
    let def = TensorKernelDef::new(
        "ew_rank_mismatch",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![4, 32],
                },
                vec![4, 32],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".to_string(),
                    shape: vec![4, 32, 128],
                },
                vec![4, 32, 128],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Elementwise {
                    kernel: sub_kernel(),
                    inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                },
                vec![4, 32],
            ),
        ],
        TensorNodeId::new(2),
    );

    let err = def
        .validate()
        .expect_err("elementwise with different-rank inputs should fail");
    assert!(
        matches!(err, TensorIRError::ElementwiseShapeMismatch { .. }),
        "expected ElementwiseShapeMismatch, got: {err:?}"
    );
}

#[test]
fn test_elementwise_matching_shapes_validates() {
    // Two inputs with identical shapes: both [4, 32]. Should pass.
    let def = TensorKernelDef::new(
        "ew_same_shape",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![4, 32],
                },
                vec![4, 32],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".to_string(),
                    shape: vec![4, 32],
                },
                vec![4, 32],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Elementwise {
                    kernel: sub_kernel(),
                    inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                },
                vec![4, 32],
            ),
        ],
        TensorNodeId::new(2),
    );

    def.validate()
        .expect("elementwise with matching input shapes should validate");
}

// Structural validation tests (forward ref, mismatched ID, empty graph,
// composition, dimension validation, pretty-print) extracted to
// tensor_ir_validation_structural.rs (#557, 500-line limit).
