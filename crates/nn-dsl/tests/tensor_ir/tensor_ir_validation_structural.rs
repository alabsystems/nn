// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for tensor-level IR structural validation: forward refs,
//! mismatched IDs, empty graphs, composition, dimension validation, and
//! pretty-printing.
//!
//! Extracted from `tensor_ir_validation_core.rs` (#557, 500-line limit).

use nn_dsl::test_kernels::{square_kernel, sub_kernel};
use nn_dsl::{
    tensor_ir_pretty_print, ReduceOp, TensorIRError, TensorKernelDef, TensorNode, TensorNodeId,
    TensorOpKind,
};

#[test]
fn test_forward_ref_rejected() {
    let def = TensorKernelDef::new(
        "fwd_ref",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Reduce {
                    op: ReduceOp::Mean,
                    input: TensorNodeId::new(1),
                    axis: 0,
                    keepdim: false,
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![4, 8],
                },
                vec![4, 8],
            ),
        ],
        TensorNodeId::new(1),
    );

    let err = def.validate().expect_err("forward ref should fail");
    assert!(
        matches!(err, TensorIRError::ForwardRef(..)),
        "expected ForwardRef, got: {err:?}"
    );
}

#[test]
fn test_mismatched_node_id() {
    let def = TensorKernelDef::new(
        "bad_id",
        vec![TensorNode::new(
            TensorNodeId::new(5),
            TensorOpKind::Input {
                name: "x".to_string(),
                shape: vec![4],
            },
            vec![4],
        )],
        TensorNodeId::new(5),
    );

    let err = def.validate().expect_err("mismatched ID should fail");
    assert!(
        matches!(err, TensorIRError::MismatchedNodeId { .. }),
        "expected MismatchedNodeId, got: {err:?}"
    );
}

#[test]
fn test_empty_graph_rejected() {
    let def = TensorKernelDef::new("empty", vec![], TensorNodeId::new(0));

    let err = def.validate().expect_err("empty graph should fail");
    assert!(
        matches!(err, TensorIRError::EmptyGraph),
        "expected EmptyGraph, got: {err:?}"
    );
}

#[test]
fn test_variance_composition() {
    // var(x) = mean(x^2) - mean(x)^2, the canonical design doc pattern.
    let def = TensorKernelDef::new(
        "variance",
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
                TensorOpKind::Elementwise {
                    kernel: square_kernel(),
                    inputs: vec![TensorNodeId::new(0)],
                },
                vec![4, 32, 128],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Reduce {
                    op: ReduceOp::Mean,
                    input: TensorNodeId::new(0),
                    axis: 2,
                    keepdim: false,
                },
                vec![4, 32],
            ),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::Reduce {
                    op: ReduceOp::Mean,
                    input: TensorNodeId::new(1),
                    axis: 2,
                    keepdim: false,
                },
                vec![4, 32],
            ),
            TensorNode::new(
                TensorNodeId::new(4),
                TensorOpKind::Elementwise {
                    kernel: square_kernel(),
                    inputs: vec![TensorNodeId::new(2)],
                },
                vec![4, 32],
            ),
            TensorNode::new(
                TensorNodeId::new(5),
                TensorOpKind::Elementwise {
                    kernel: sub_kernel(),
                    inputs: vec![TensorNodeId::new(3), TensorNodeId::new(4)],
                },
                vec![4, 32],
            ),
        ],
        TensorNodeId::new(5),
    );

    def.validate()
        .expect("variance composition should validate");
}

#[test]
fn test_reduce_1d_produces_scalar_shape() {
    let def = TensorKernelDef::new(
        "reduce_1d",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![128],
                },
                vec![128],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reduce {
                    op: ReduceOp::Sum,
                    input: TensorNodeId::new(0),
                    axis: 0,
                    keepdim: false,
                },
                vec![1],
            ),
        ],
        TensorNodeId::new(1),
    );

    def.validate().expect("1D reduce to scalar should validate");
}

#[test]
fn test_empty_dimension_rejected() {
    let def = TensorKernelDef::new(
        "zero_dim",
        vec![TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "x".to_string(),
                shape: vec![4, 0, 128],
            },
            vec![4, 0, 128],
        )],
        TensorNodeId::new(0),
    );

    let err = def.validate().expect_err("zero dimension should fail");
    assert!(
        matches!(err, TensorIRError::EmptyDimension(..)),
        "expected EmptyDimension, got: {err:?}"
    );
}

#[test]
fn test_pretty_print_format() {
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

    let output = tensor_ir_pretty_print(&def);
    assert!(output.contains("tensor_kernel mean_test"));
    assert!(output.contains("input(\"x\", [4, 32, 128])"));
    assert!(output.contains("reduce_mean(%0, axis=2)"));
    assert!(
        !output.contains("keepdim"),
        "keepdim=false should not appear in pretty print"
    );
    assert!(output.contains("return %1"));
}

#[test]
fn test_pretty_print_reduce_keepdim() {
    let def = TensorKernelDef::new(
        "keepdim_test",
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
                    op: ReduceOp::Sum,
                    input: TensorNodeId::new(0),
                    axis: 2,
                    keepdim: true,
                },
                vec![4, 32, 1],
            ),
        ],
        TensorNodeId::new(1),
    );

    let output = tensor_ir_pretty_print(&def);
    assert!(
        output.contains("reduce_sum(%0, axis=2, keepdim=true)"),
        "keepdim=true should appear in pretty print. Got: {output}"
    );
}
