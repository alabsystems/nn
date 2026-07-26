// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for tensor IR validation and output shape computation.
//!
//! Covers broadcast validation, output shape inference, topological ordering,
//! and graph-level invariants. Conv1d-specific error variants are covered in
//! the integration test file `tests/tensor_ir_validation_conv1d.rs`.
//!
//! Split into submodules per #631:
//! - This file: core validation, topological ordering, broadcast, reduce
//! - `tensor_ir_validate_tests_structural.rs`: elementwise, reshape, axis_select, stack
//! - `tensor_ir_validate_tests_conv.rs`: conv1d shape tests
//!
//! See #595.

use super::*;
use crate::ir::{KernelDef, Param, ScalarType};
use crate::test_kernels::identity_kernel;

#[path = "tensor_ir_validate_tests_structural.rs"]
mod structural;

#[path = "tensor_ir_validate_tests_conv.rs"]
mod conv;

#[path = "tensor_ir_validate_tests_layers.rs"]
mod layers;

// ---------------------------------------------------------------------------
// Helper: build a minimal graph with specified nodes
// ---------------------------------------------------------------------------

fn input_node(id: usize, name: &str, shape: Vec<usize>) -> TensorNode {
    TensorNode::new(
        TensorNodeId::new(id),
        TensorOpKind::Input {
            name: name.to_string(),
            shape: shape.clone(),
        },
        shape,
    )
}

/// 2-param add kernel for multi-input elementwise tests.
fn add_kernel() -> KernelDef {
    use crate::ir::{BinOpKind, IRNode, IRNodeKind, NodeId};
    KernelDef::new(
        "add",
        vec![
            Param::new("a", ScalarType::F32),
            Param::new("b", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    )
}

// ===========================================================================
// Graph-level invariant tests
// ===========================================================================

#[test]
fn test_empty_graph() {
    let def = TensorKernelDef::new("empty", vec![], TensorNodeId::new(0));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(err, TensorIRError::EmptyGraph),
        "expected EmptyGraph, got: {err}"
    );
}

#[test]
fn test_mismatched_node_id() {
    // Node at index 0 has id=1 — violates the invariant.
    let node = TensorNode::new(
        TensorNodeId::new(1),
        TensorOpKind::Input {
            name: "x".to_string(),
            shape: vec![4],
        },
        vec![4],
    );
    let def = TensorKernelDef::new("bad_id", vec![node], TensorNodeId::new(0));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::MismatchedNodeId {
                found,
                expected_index: 0
            } if found == TensorNodeId::new(1)
        ),
        "expected MismatchedNodeId, got: {err}"
    );
}

#[test]
fn test_empty_dimension_rejected() {
    let def = TensorKernelDef::new(
        "zero_dim",
        vec![input_node(0, "x", vec![4, 0, 8])],
        TensorNodeId::new(0),
    );
    let err = def.validate().unwrap_err();
    assert!(
        matches!(err, TensorIRError::EmptyDimension(_)),
        "expected EmptyDimension, got: {err}"
    );
}

#[test]
fn test_output_ref_out_of_bounds() {
    let def = TensorKernelDef::new(
        "bad_output",
        vec![input_node(0, "x", vec![4])],
        TensorNodeId::new(5), // only 1 node
    );
    let err = def.validate().unwrap_err();
    assert!(
        matches!(err, TensorIRError::InvalidNodeRef(id) if id == TensorNodeId::new(5)),
        "expected InvalidNodeRef, got: {err}"
    );
}

// ===========================================================================
// Topological ordering / reference tests
// ===========================================================================

#[test]
fn test_forward_reference_rejected() {
    // Reduce at node 1 references node 2 (forward ref).
    let nodes = vec![
        input_node(0, "x", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Reduce {
                op: ReduceOp::Sum,
                input: TensorNodeId::new(2), // forward ref
                axis: 1,
                keepdim: false,
            },
            vec![4],
        ),
        input_node(2, "y", vec![4, 8]),
    ];
    let def = TensorKernelDef::new("fwd_ref", nodes, TensorNodeId::new(1));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::ForwardRef(a, b) if a == TensorNodeId::new(1) && b == TensorNodeId::new(2)
        ),
        "expected ForwardRef, got: {err}"
    );
}

#[test]
fn test_self_reference_rejected() {
    // Reduce at node 1 references itself.
    let nodes = vec![
        input_node(0, "x", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Reduce {
                op: ReduceOp::Mean,
                input: TensorNodeId::new(1), // self ref
                axis: 0,
                keepdim: false,
            },
            vec![8],
        ),
    ];
    let def = TensorKernelDef::new("self_ref", nodes, TensorNodeId::new(1));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::ForwardRef(a, b) if a == TensorNodeId::new(1) && b == TensorNodeId::new(1)
        ),
        "expected ForwardRef (self-ref), got: {err}"
    );
}

#[test]
fn test_invalid_node_ref() {
    // Reduce references node 99 which doesn't exist.
    let nodes = vec![
        input_node(0, "x", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Reduce {
                op: ReduceOp::Sum,
                input: TensorNodeId::new(99),
                axis: 0,
                keepdim: false,
            },
            vec![8],
        ),
    ];
    let def = TensorKernelDef::new("bad_ref", nodes, TensorNodeId::new(1));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(err, TensorIRError::InvalidNodeRef(id) if id == TensorNodeId::new(99)),
        "expected InvalidNodeRef, got: {err}"
    );
}

// ===========================================================================
// Broadcast validation tests
// ===========================================================================

#[test]
fn test_broadcast_valid_right_alignment() {
    // [8] broadcast to [4, 8] with right alignment — valid.
    let nodes = vec![
        input_node(0, "x", vec![8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Broadcast {
                input: TensorNodeId::new(0),
                target_shape: vec![4, 8],
                alignment: BroadcastAlignment::Right,
            },
            vec![4, 8],
        ),
    ];
    let def = TensorKernelDef::new("bcast_right", nodes, TensorNodeId::new(1));
    assert!(
        def.validate().is_ok(),
        "valid right-aligned broadcast should pass"
    );
}

#[test]
fn test_broadcast_valid_left_alignment() {
    // [4] broadcast to [4, 8] with left alignment — valid.
    let nodes = vec![
        input_node(0, "x", vec![4]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Broadcast {
                input: TensorNodeId::new(0),
                target_shape: vec![4, 8],
                alignment: BroadcastAlignment::Left,
            },
            vec![4, 8],
        ),
    ];
    let def = TensorKernelDef::new("bcast_left", nodes, TensorNodeId::new(1));
    assert!(
        def.validate().is_ok(),
        "valid left-aligned broadcast should pass"
    );
}

#[test]
fn test_broadcast_incompatible_shapes() {
    // [3] broadcast to [4, 8] with right alignment — incompatible (3 != 8).
    let nodes = vec![
        input_node(0, "x", vec![3]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Broadcast {
                input: TensorNodeId::new(0),
                target_shape: vec![4, 8],
                alignment: BroadcastAlignment::Right,
            },
            vec![4, 8],
        ),
    ];
    let def = TensorKernelDef::new("bcast_bad", nodes, TensorNodeId::new(1));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(err, TensorIRError::IncompatibleBroadcast { .. }),
        "expected IncompatibleBroadcast, got: {err}"
    );
}

#[test]
fn test_broadcast_input_rank_exceeds_target() {
    // [4, 8] broadcast to [8] — input rank > target rank.
    let nodes = vec![
        input_node(0, "x", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Broadcast {
                input: TensorNodeId::new(0),
                target_shape: vec![8],
                alignment: BroadcastAlignment::Right,
            },
            vec![8],
        ),
    ];
    let def = TensorKernelDef::new("bcast_rank", nodes, TensorNodeId::new(1));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(err, TensorIRError::IncompatibleBroadcast { .. }),
        "expected IncompatibleBroadcast for rank mismatch, got: {err}"
    );
}

#[test]
fn test_broadcast_scalar_to_any() {
    // [1] broadcast to [4, 8, 16] with right alignment — scalar broadcasts to anything.
    let nodes = vec![
        input_node(0, "eps", vec![1]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Broadcast {
                input: TensorNodeId::new(0),
                target_shape: vec![4, 8, 16],
                alignment: BroadcastAlignment::Right,
            },
            vec![4, 8, 16],
        ),
    ];
    let def = TensorKernelDef::new("bcast_scalar", nodes, TensorNodeId::new(1));
    assert!(def.validate().is_ok(), "scalar broadcast should pass");
}

#[test]
fn test_broadcast_same_rank_compatible() {
    // [4, 1, 16] broadcast to [4, 8, 16] with left alignment — valid (dim 1 broadcasts).
    let nodes = vec![
        input_node(0, "x", vec![4, 1, 16]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Broadcast {
                input: TensorNodeId::new(0),
                target_shape: vec![4, 8, 16],
                alignment: BroadcastAlignment::Left,
            },
            vec![4, 8, 16],
        ),
    ];
    let def = TensorKernelDef::new("bcast_same", nodes, TensorNodeId::new(1));
    assert!(
        def.validate().is_ok(),
        "same-rank broadcast with dim=1 should pass"
    );
}

// ===========================================================================
// Output shape computation tests — reduce
// ===========================================================================

#[test]
fn test_output_shape_reduce_removes_axis() {
    // input [4, 8], reduce axis=1 → [4]
    let nodes = vec![
        input_node(0, "x", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Reduce {
                op: ReduceOp::Sum,
                input: TensorNodeId::new(0),
                axis: 1,
                keepdim: false,
            },
            vec![4],
        ),
    ];
    let def = TensorKernelDef::new("reduce_shape", nodes, TensorNodeId::new(1));
    assert!(def.validate().is_ok(), "reduce should produce [4]");
}

#[test]
fn test_output_shape_reduce_1d_to_scalar() {
    // input [8], reduce axis=0 → [1] (scalar representation)
    let nodes = vec![
        input_node(0, "x", vec![8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Reduce {
                op: ReduceOp::Mean,
                input: TensorNodeId::new(0),
                axis: 0,
                keepdim: false,
            },
            vec![1],
        ),
    ];
    let def = TensorKernelDef::new("reduce_scalar", nodes, TensorNodeId::new(1));
    assert!(def.validate().is_ok(), "reducing 1-D should produce [1]");
}

#[test]
fn test_reduce_axis_out_of_bounds() {
    // input [4, 8], reduce axis=2 → out of bounds
    let nodes = vec![
        input_node(0, "x", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Reduce {
                op: ReduceOp::Sum,
                input: TensorNodeId::new(0),
                axis: 2,
                keepdim: false,
            },
            vec![4],
        ),
    ];
    let def = TensorKernelDef::new("reduce_oob", nodes, TensorNodeId::new(1));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(err, TensorIRError::ReduceAxisOutOfBounds { axis: 2, .. }),
        "expected ReduceAxisOutOfBounds, got: {err}"
    );
}

#[test]
fn test_output_shape_reduce_keepdim_retains_axis() {
    // input [4, 8], reduce axis=1 keepdim=true → [4, 1]
    let nodes = vec![
        input_node(0, "x", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Reduce {
                op: ReduceOp::Sum,
                input: TensorNodeId::new(0),
                axis: 1,
                keepdim: true,
            },
            vec![4, 1],
        ),
    ];
    let def = TensorKernelDef::new("reduce_keepdim", nodes, TensorNodeId::new(1));
    assert!(
        def.validate().is_ok(),
        "reduce keepdim should produce [4, 1]"
    );
}

#[test]
fn test_output_shape_reduce_keepdim_3d() {
    // input [3, 4, 8], reduce axis=1 keepdim=true → [3, 1, 8]
    let nodes = vec![
        input_node(0, "x", vec![3, 4, 8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Reduce {
                op: ReduceOp::Mean,
                input: TensorNodeId::new(0),
                axis: 1,
                keepdim: true,
            },
            vec![3, 1, 8],
        ),
    ];
    let def = TensorKernelDef::new("reduce_keepdim_3d", nodes, TensorNodeId::new(1));
    assert!(
        def.validate().is_ok(),
        "reduce keepdim on 3D should produce [3, 1, 8]"
    );
}

#[test]
fn test_output_shape_reduce_keepdim_wrong_shape_rejected() {
    // input [4, 8], reduce axis=1 keepdim=true, but node claims shape [4] (wrong)
    let nodes = vec![
        input_node(0, "x", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Reduce {
                op: ReduceOp::Sum,
                input: TensorNodeId::new(0),
                axis: 1,
                keepdim: true,
            },
            vec![4], // wrong: should be [4, 1]
        ),
    ];
    let def = TensorKernelDef::new("reduce_keepdim_bad", nodes, TensorNodeId::new(1));
    assert!(
        def.validate().is_err(),
        "mismatched keepdim shape should fail"
    );
}
