// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv1d validation tests for tensor IR output shape computation.
//!
//! Split from `tensor_ir_validate_tests.rs` per #631.

use super::*;

// ===========================================================================
// Conv1d output shape computation tests
// ===========================================================================

#[test]
fn test_output_shape_conv1d_computation() {
    // input [4, 8], weight [2, 4, 3], stride=1, pad=0, dil=1, groups=1
    // output_len = (8 + 0 - 3) / 1 + 1 = 6 → [2, 6]
    let nodes = vec![
        input_node(0, "input", vec![4, 8]),
        input_node(1, "weight", vec![2, 4, 3]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Conv1d {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: None,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: 1,
            },
            vec![2, 6],
        ),
    ];
    let def = TensorKernelDef::new("conv1d_shape", nodes, TensorNodeId::new(2));
    assert!(
        def.validate().is_ok(),
        "Conv1d output shape [2,6] should match computation"
    );
}

#[test]
fn test_output_shape_conv1d_with_stride() {
    // input [4, 16], weight [8, 4, 3], stride=2, pad=0, dil=1, groups=1
    // output_len = (16 + 0 - 3) / 2 + 1 = 7 → [8, 7]
    let nodes = vec![
        input_node(0, "input", vec![4, 16]),
        input_node(1, "weight", vec![8, 4, 3]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Conv1d {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: None,
                stride: 2,
                padding: 0,
                dilation: 1,
                groups: 1,
            },
            vec![8, 7],
        ),
    ];
    let def = TensorKernelDef::new("conv1d_stride", nodes, TensorNodeId::new(2));
    assert!(
        def.validate().is_ok(),
        "Conv1d with stride=2 output shape [8,7] should match"
    );
}

#[test]
fn test_output_shape_conv1d_wrong_shape_detected() {
    // Same as above but node claims wrong output shape [8, 99].
    let nodes = vec![
        input_node(0, "input", vec![4, 16]),
        input_node(1, "weight", vec![8, 4, 3]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Conv1d {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: None,
                stride: 2,
                padding: 0,
                dilation: 1,
                groups: 1,
            },
            vec![8, 99], // wrong
        ),
    ];
    let def = TensorKernelDef::new("conv1d_bad_shape", nodes, TensorNodeId::new(2));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(err, TensorIRError::IncompatibleBroadcast { .. }),
        "expected shape mismatch from compute_output_shape check, got: {err}"
    );
}
