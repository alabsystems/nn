// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Conv1d tensor IR validation.
//!
//! Covers all 12 Conv1d error variants in `validate_conv1d` plus the overflow
//! paths in `compute_output_shape`. Tests construct graphs manually (bypassing
//! the builder) to exercise the validator paths directly.
//!
//! See #595, #604.

use nn_dsl::{
    TensorIRConvError, TensorIRError, TensorIRLayerError, TensorKernelDef, TensorNode,
    TensorNodeId, TensorOpKind,
};

/// Build a Conv1d graph with customizable parameters.
/// Defaults: input [4, 8], weight [2, 4, 3], stride=1, padding=0, dilation=1, groups=1.
fn conv1d_graph(
    input_shape: Vec<usize>,
    weight_shape: Vec<usize>,
    bias: Option<(TensorNodeId, Vec<usize>)>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    output_shape: Vec<usize>,
) -> TensorKernelDef {
    let mut nodes = vec![
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "input".to_string(),
                shape: input_shape.clone(),
            },
            input_shape,
        ),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Input {
                name: "weight".to_string(),
                shape: weight_shape.clone(),
            },
            weight_shape,
        ),
    ];

    let bias_id = if let Some((_, ref bias_shape)) = bias {
        nodes.push(TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Input {
                name: "bias".to_string(),
                shape: bias_shape.clone(),
            },
            bias_shape.clone(),
        ));
        Some(TensorNodeId::new(2))
    } else {
        None
    };

    let conv_idx = nodes.len();
    nodes.push(TensorNode::new(
        TensorNodeId::new(conv_idx),
        TensorOpKind::Conv1d {
            input: TensorNodeId::new(0),
            weight: TensorNodeId::new(1),
            bias: bias_id,
            stride,
            padding,
            dilation,
            groups,
        },
        output_shape,
    ));

    TensorKernelDef::new("test_conv1d", nodes, TensorNodeId::new(conv_idx))
}

// --- Happy path ---

#[test]
fn test_conv1d_valid_basic() {
    // input [4, 8], weight [2, 4, 3], stride=1, pad=0, dil=1, groups=1
    // output_len = (8 + 0 - 3) / 1 + 1 = 6, output_shape = [2, 6]
    let def = conv1d_graph(vec![4, 8], vec![2, 4, 3], None, 1, 0, 1, 1, vec![2, 6]);
    assert!(def.validate().is_ok(), "valid Conv1d should pass");
}

#[test]
fn test_conv1d_valid_with_padding() {
    // input [4, 8], weight [2, 4, 3], stride=1, pad=1, dil=1, groups=1
    // output_len = (8 + 2 - 3) / 1 + 1 = 8
    let def = conv1d_graph(vec![4, 8], vec![2, 4, 3], None, 1, 1, 1, 1, vec![2, 8]);
    assert!(def.validate().is_ok(), "padded Conv1d should pass");
}

#[test]
fn test_conv1d_valid_with_bias() {
    // input [4, 8], weight [2, 4, 3], bias [2], stride=1, pad=0
    // output = [2, 6]
    let def = conv1d_graph(
        vec![4, 8],
        vec![2, 4, 3],
        Some((TensorNodeId::new(2), vec![2])),
        1,
        0,
        1,
        1,
        vec![2, 6],
    );
    assert!(def.validate().is_ok(), "Conv1d with bias should pass");
}

// --- Error variant: Conv1dZeroDilation ---

#[test]
fn test_conv1d_zero_dilation() {
    let def = conv1d_graph(vec![4, 8], vec![2, 4, 3], None, 1, 0, 0, 1, vec![2, 6]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dZeroDilation
            ))
        ),
        "expected Conv1dZeroDilation, got: {err}"
    );
}

// --- Error variant: Conv1dZeroGroups ---

#[test]
fn test_conv1d_zero_groups() {
    let def = conv1d_graph(vec![4, 8], vec![2, 4, 3], None, 1, 0, 1, 0, vec![2, 6]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dZeroGroups
            ))
        ),
        "expected Conv1dZeroGroups, got: {err}"
    );
}

// --- Error variant: Conv1dInputRankTooLow ---

#[test]
fn test_conv1d_input_rank_too_low() {
    // 1-D input: shape [8] — needs at least 2 dims
    let def = conv1d_graph(vec![8], vec![2, 4, 3], None, 1, 0, 1, 1, vec![2, 6]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dInputRankTooLow { rank: 1 }
            ))
        ),
        "expected Conv1dInputRankTooLow, got: {err}"
    );
}

// --- Error variant: Conv1dWeightShape ---

#[test]
fn test_conv1d_weight_wrong_rank() {
    // Weight must be 3D [out_ch, in_ch/groups, kernel_size]. Give it 2D.
    let def = conv1d_graph(vec![4, 8], vec![2, 3], None, 1, 0, 1, 1, vec![2, 6]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dWeightShape { .. }
            ))
        ),
        "expected Conv1dWeightShape, got: {err}"
    );
}

// --- Error variant: Conv1dGroupsChannelMismatch ---

#[test]
fn test_conv1d_groups_channel_mismatch() {
    // input [5, 8] — 5 in_channels not divisible by groups=2
    // weight [4, 2, 3] — 4 out_ch, in_ch/groups=2 (if groups divides evenly)
    let def = conv1d_graph(vec![5, 8], vec![4, 2, 3], None, 1, 0, 1, 2, vec![4, 6]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dGroupsChannelMismatch {
                    in_channels: 5,
                    groups: 2
                }
            ))
        ),
        "expected Conv1dGroupsChannelMismatch, got: {err}"
    );
}

// --- Error variant: Conv1dGroupsOutputMismatch ---

#[test]
fn test_conv1d_groups_output_mismatch() {
    // input [4, 8], 4 in_channels divisible by 2 groups
    // weight [3, 2, 3] — 3 out_channels not divisible by 2 groups
    let def = conv1d_graph(vec![4, 8], vec![3, 2, 3], None, 1, 0, 1, 2, vec![3, 6]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dGroupsOutputMismatch {
                    out_channels: 3,
                    groups: 2
                }
            ))
        ),
        "expected Conv1dGroupsOutputMismatch, got: {err}"
    );
}

// --- Error variant: Conv1dGroupsWeightMismatch ---

#[test]
fn test_conv1d_groups_weight_mismatch() {
    // input [4, 8], groups=2. Expected weight_in = 4/2 = 2.
    // weight [4, 3, 3] — weight_in=3 != expected 2
    let def = conv1d_graph(vec![4, 8], vec![4, 3, 3], None, 1, 0, 1, 2, vec![4, 6]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dGroupsWeightMismatch {
                    weight_in_channels: 3,
                    expected: 2
                }
            ))
        ),
        "expected Conv1dGroupsWeightMismatch, got: {err}"
    );
}

// --- Error variant: Conv1dZeroStride ---

#[test]
fn test_conv1d_zero_stride() {
    let def = conv1d_graph(vec![4, 8], vec![2, 4, 3], None, 0, 0, 1, 1, vec![2, 6]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dZeroStride
            ))
        ),
        "expected Conv1dZeroStride, got: {err}"
    );
}

// --- Error variant: Conv1dZeroKernelSize ---
// Note: `Conv1dZeroKernelSize` is unreachable via `validate()` on a complete
// graph because the weight node's Input shape validation (`validate_shape`)
// rejects zero-dimensional shapes first. The variant exists as defense-in-depth
// for callers that invoke `validate_conv1d` directly with pre-validated nodes.

#[test]
fn test_conv1d_zero_kernel_size_caught_by_shape_validation() {
    // weight [2, 4, 0] — kernel_size=0. The Input node's shape validation
    // fires EmptyDimension before Conv1d-specific checks run.
    let def = conv1d_graph(vec![4, 8], vec![2, 4, 0], None, 1, 0, 1, 1, vec![2, 9]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(err, TensorIRError::EmptyDimension(_)),
        "expected EmptyDimension for zero-kernel weight shape, got: {err}"
    );
}

// --- Error variant: Conv1dKernelTooLarge ---

#[test]
fn test_conv1d_kernel_too_large() {
    // input [4, 2], weight [2, 4, 5], pad=0 -> padded=2, effective_kernel=5
    // padded < effective_kernel
    let def = conv1d_graph(vec![4, 2], vec![2, 4, 5], None, 1, 0, 1, 1, vec![2, 1]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dKernelTooLarge { .. }
            ))
        ),
        "expected Conv1dKernelTooLarge, got: {err}"
    );
}

// --- Error variant: Conv1dBiasShape ---

#[test]
fn test_conv1d_bias_wrong_shape() {
    // weight [2, 4, 3] -> out_channels=2, bias should be [2] but we give [3]
    let def = conv1d_graph(
        vec![4, 8],
        vec![2, 4, 3],
        Some((TensorNodeId::new(2), vec![3])),
        1,
        0,
        1,
        1,
        vec![2, 6],
    );
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dBiasShape {
                    expected: 2,
                    got_shape: _
                }
            ))
        ),
        "expected Conv1dBiasShape, got: {err}"
    );
}

// --- Overflow: effective_kernel (existing test #604) ---

#[test]
fn test_validate_conv1d_effective_kernel_overflow() {
    // dilation=usize::MAX, kernel_size=3 -> effective_kernel = MAX * 2 + 1 overflows
    let def = conv1d_graph(
        vec![4, 8],
        vec![2, 4, 3],
        None,
        1,
        0,
        usize::MAX,
        1,
        vec![2, 1],
    );

    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dArithmeticOverflow { .. }
            ))
        ),
        "expected arithmetic overflow from validator, got: {err}"
    );
}

// --- Overflow: padded (existing test #604) ---

#[test]
fn test_validate_conv1d_padded_overflow() {
    // padding=usize::MAX -> padded = in_len + 2 * MAX overflows
    let def = conv1d_graph(
        vec![4, 8],
        vec![2, 4, 3],
        None,
        1,
        usize::MAX,
        1,
        1,
        vec![2, 1],
    );

    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dArithmeticOverflow { .. }
            ))
        ),
        "expected padded overflow from validator, got: {err}"
    );
}
