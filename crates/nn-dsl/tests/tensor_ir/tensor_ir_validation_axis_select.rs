// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for tensor-level IR validation of AxisSelect operations.

use nn_dsl::{TensorIRError, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};

#[test]
fn test_axis_select_axis_zero_rejected() {
    let def = TensorKernelDef::new(
        "axis_select_axis0",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![2, 4, 8],
                },
                vec![2, 4, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::AxisSelect {
                    input: TensorNodeId::new(0),
                    axis: 0,
                    index: 1,
                },
                vec![4, 8],
            ),
        ],
        TensorNodeId::new(1),
    );

    let err = def
        .validate()
        .expect_err("axis_select must reject axis 0 in verification path");
    assert!(
        matches!(err, TensorIRError::AxisZeroReserved { op: "axis_select" }),
        "expected AxisZeroReserved(axis_select), got: {err:?}"
    );
}

#[test]
fn test_axis_select_reduces_rank_and_keeps_other_dims() {
    let def = TensorKernelDef::new(
        "axis_select_rank",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![3, 5, 7],
                },
                vec![3, 5, 7],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::AxisSelect {
                    input: TensorNodeId::new(0),
                    axis: 2,
                    index: 4,
                },
                vec![3, 5],
            ),
        ],
        TensorNodeId::new(1),
    );

    def.validate()
        .expect("axis_select should remove the selected axis from output shape");
}

// --- AxisSelect axis-0 edge cases and missing rejection paths ---

#[test]
fn test_axis_select_axis_zero_on_1d_input() {
    // axis=0 on a 1D tensor: axis is in-bounds (0 < 1), but must still yield
    // AxisZeroReserved (not AxisSelectOutOfBounds).
    let def = TensorKernelDef::new(
        "axis_select_1d_axis0",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![8],
                },
                vec![8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::AxisSelect {
                    input: TensorNodeId::new(0),
                    axis: 0,
                    index: 0,
                },
                vec![1],
            ),
        ],
        TensorNodeId::new(1),
    );

    let err = def
        .validate()
        .expect_err("axis_select axis=0 on 1D must be rejected");
    assert!(
        matches!(err, TensorIRError::AxisZeroReserved { op: "axis_select" }),
        "expected AxisZeroReserved, got: {err:?}"
    );
}

#[test]
fn test_axis_select_axis_zero_index_zero() {
    // axis=0 with index=0: valid index should NOT bypass axis-0 guard.
    let def = TensorKernelDef::new(
        "axis_select_axis0_idx0",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![2, 4, 8],
                },
                vec![2, 4, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::AxisSelect {
                    input: TensorNodeId::new(0),
                    axis: 0,
                    index: 0,
                },
                vec![4, 8],
            ),
        ],
        TensorNodeId::new(1),
    );

    let err = def
        .validate()
        .expect_err("axis=0 with index=0 must still be rejected");
    assert!(
        matches!(err, TensorIRError::AxisZeroReserved { op: "axis_select" }),
        "expected AxisZeroReserved, got: {err:?}"
    );
}

#[test]
fn test_axis_select_axis_zero_large_index() {
    // axis=0, index=99: must yield AxisZeroReserved, not AxisSelectIndexOutOfBounds.
    // Verifies check ordering: axis-0 guard fires before index validation.
    let def = TensorKernelDef::new(
        "axis_select_axis0_large_idx",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![2, 4],
                },
                vec![2, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::AxisSelect {
                    input: TensorNodeId::new(0),
                    axis: 0,
                    index: 99,
                },
                vec![4],
            ),
        ],
        TensorNodeId::new(1),
    );

    let err = def
        .validate()
        .expect_err("axis=0 with out-of-bounds index must yield AxisZeroReserved");
    assert!(
        matches!(err, TensorIRError::AxisZeroReserved { op: "axis_select" }),
        "expected AxisZeroReserved (not AxisSelectIndexOutOfBounds), got: {err:?}"
    );
}

#[test]
fn test_axis_select_axis_out_of_bounds() {
    let def = TensorKernelDef::new(
        "axis_select_oob",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![4, 8],
                },
                vec![4, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::AxisSelect {
                    input: TensorNodeId::new(0),
                    axis: 5,
                    index: 0,
                },
                vec![4],
            ),
        ],
        TensorNodeId::new(1),
    );

    let err = def
        .validate()
        .expect_err("axis 5 on rank-2 input must be rejected");
    assert!(
        matches!(err, TensorIRError::AxisSelectOutOfBounds { axis: 5, .. }),
        "expected AxisSelectOutOfBounds, got: {err:?}"
    );
}

#[test]
fn test_axis_select_index_out_of_bounds() {
    let def = TensorKernelDef::new(
        "axis_select_idx_oob",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![3, 5, 7],
                },
                vec![3, 5, 7],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::AxisSelect {
                    input: TensorNodeId::new(0),
                    axis: 1,
                    index: 5,
                },
                vec![3, 7],
            ),
        ],
        TensorNodeId::new(1),
    );

    let err = def
        .validate()
        .expect_err("index 5 at dim=5 (axis=1) must be rejected");
    assert!(
        matches!(
            err,
            TensorIRError::AxisSelectIndexOutOfBounds {
                index: 5,
                dim: 5,
                axis: 1
            }
        ),
        "expected AxisSelectIndexOutOfBounds, got: {err:?}"
    );
}

#[test]
fn test_axis_select_rank2_last_axis_produces_rank1() {
    // [3, 5] axis=1 index=2 -> [3]
    let def = TensorKernelDef::new(
        "axis_select_last",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![3, 5],
                },
                vec![3, 5],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::AxisSelect {
                    input: TensorNodeId::new(0),
                    axis: 1,
                    index: 2,
                },
                vec![3],
            ),
        ],
        TensorNodeId::new(1),
    );

    def.validate()
        .expect("selecting last axis of rank-2 tensor should produce rank-1 output");
}
