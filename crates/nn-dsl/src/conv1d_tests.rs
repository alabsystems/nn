// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Conv1d tensor kernel builder (extracted from conv1d.rs, 500-line limit).

use super::*;
use crate::tensor_ir::{TensorIRConvError, TensorIRLayerError};

#[test]
fn test_build_conv1d_basic() {
    let def = build_conv1d("conv1d_basic", 4, 2, 3, 8, 1, 0, false).expect("build");
    assert_eq!(def.name, "conv1d_basic");
    assert_eq!(def.nodes.len(), 3);
    assert_eq!(def.nodes[2].shape, vec![2, 6]);
    def.validate().expect("basic conv1d should validate");
}

#[test]
fn test_build_conv1d_with_bias() {
    let def = build_conv1d("conv1d_bias", 4, 2, 3, 8, 1, 0, true).expect("build");
    assert_eq!(def.nodes.len(), 4);
    assert_eq!(def.nodes[3].shape, vec![2, 6]);
    def.validate().expect("conv1d with bias should validate");
}

#[test]
fn test_build_conv1d_stride_padding() {
    let def = build_conv1d("conv1d_ds", 1, 48, 8, 16000, 4, 2, false).expect("build");
    assert_eq!(def.nodes[2].shape, vec![48, 4000]);
    def.validate()
        .expect("conv1d with stride/padding should validate");
}

#[test]
fn test_build_conv1d_kernel_size_1() {
    let def = build_conv1d("conv1d_1x1", 48, 96, 1, 4000, 1, 0, true).expect("build");
    assert_eq!(def.nodes[3].shape, vec![96, 4000]);
    def.validate().expect("1x1 conv1d should validate");
}

#[test]
fn test_build_conv1d_dilation() {
    // dilation=2, kernel_size=3 -> effective_kernel = 2*(3-1)+1 = 5
    // out_len = (16 + 0 - 5) / 1 + 1 = 12
    let def = build_conv1d_full("conv1d_dil", 4, 2, 3, 16, 1, 0, 2, 1, false).expect("build");
    assert_eq!(def.nodes[2].shape, vec![2, 12]);
    def.validate().expect("dilated conv1d should validate");
}

#[test]
fn test_build_conv1d_groups() {
    // groups=2, in_channels=4 -> weight_in_channels = 4/2 = 2
    let def = build_conv1d_full("conv1d_grp", 4, 4, 3, 16, 1, 0, 1, 2, false).expect("build");
    assert_eq!(def.nodes[1].shape, vec![4, 2, 3]);
    assert_eq!(def.nodes[2].shape, vec![4, 14]);
    def.validate().expect("grouped conv1d should validate");
}

#[test]
fn test_build_conv1d_dilation_and_groups() {
    // dilation=2, groups=2, kernel_size=3
    // effective_kernel = 2*(3-1)+1 = 5
    // out_len = (16 + 2 - 5) / 2 + 1 = 13/2 + 1 = 6 + 1 = 7
    let def = build_conv1d_full("conv1d_dg", 4, 4, 3, 16, 2, 1, 2, 2, true).expect("build");
    assert_eq!(def.nodes[1].shape, vec![4, 2, 3]);
    assert_eq!(def.nodes[3].shape, vec![4, 7]);
    def.validate()
        .expect("dilated+grouped conv1d should validate");
}

#[test]
fn test_conv1d_validation_channel_mismatch() {
    use crate::tensor_ir::TensorIRError;

    let nodes = vec![
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "data".into(),
                shape: vec![4, 8],
            },
            vec![4, 8],
        ),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Input {
                name: "weight".into(),
                shape: vec![2, 3, 3],
            },
            vec![2, 3, 3],
        ),
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
    let def = TensorKernelDef::new("bad_conv", nodes, TensorNodeId::new(2));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dGroupsWeightMismatch {
                    weight_in_channels: 3,
                    expected: 4
                }
            ))
        ),
        "expected weight mismatch, got: {err}"
    );
}

#[test]
fn test_conv1d_validation_zero_stride() {
    use crate::tensor_ir::TensorIRError;

    let nodes = vec![
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "data".into(),
                shape: vec![4, 8],
            },
            vec![4, 8],
        ),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Input {
                name: "weight".into(),
                shape: vec![2, 4, 3],
            },
            vec![2, 4, 3],
        ),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Conv1d {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: None,
                stride: 0,
                padding: 0,
                dilation: 1,
                groups: 1,
            },
            vec![2, 6],
        ),
    ];
    let def = TensorKernelDef::new("zero_stride", nodes, TensorNodeId::new(2));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dZeroStride
            ))
        ),
        "expected zero stride, got: {err}"
    );
}

#[test]
fn test_conv1d_validation_zero_dilation() {
    use crate::tensor_ir::TensorIRError;

    let nodes = vec![
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "data".into(),
                shape: vec![4, 8],
            },
            vec![4, 8],
        ),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Input {
                name: "weight".into(),
                shape: vec![2, 4, 3],
            },
            vec![2, 4, 3],
        ),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Conv1d {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: None,
                stride: 1,
                padding: 0,
                dilation: 0,
                groups: 1,
            },
            vec![2, 6],
        ),
    ];
    let def = TensorKernelDef::new("zero_dilation", nodes, TensorNodeId::new(2));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dZeroDilation
            ))
        ),
        "expected zero dilation, got: {err}"
    );
}

#[test]
fn test_conv1d_validation_zero_groups() {
    use crate::tensor_ir::TensorIRError;

    let nodes = vec![
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "data".into(),
                shape: vec![4, 8],
            },
            vec![4, 8],
        ),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Input {
                name: "weight".into(),
                shape: vec![2, 4, 3],
            },
            vec![2, 4, 3],
        ),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Conv1d {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: None,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: 0,
            },
            vec![2, 6],
        ),
    ];
    let def = TensorKernelDef::new("zero_groups", nodes, TensorNodeId::new(2));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dZeroGroups
            ))
        ),
        "expected zero groups, got: {err}"
    );
}

#[test]
fn test_conv1d_validation_groups_channel_mismatch() {
    use crate::tensor_ir::TensorIRError;

    let nodes = vec![
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "data".into(),
                shape: vec![5, 8],
            },
            vec![5, 8],
        ),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Input {
                name: "weight".into(),
                shape: vec![2, 2, 3],
            },
            vec![2, 2, 3],
        ),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Conv1d {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: None,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: 2,
            },
            vec![2, 6],
        ),
    ];
    let def = TensorKernelDef::new("bad_groups", nodes, TensorNodeId::new(2));
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
        "expected groups channel mismatch, got: {err}"
    );
}

/// `kernel_size = 0` returns `Err(Conv1dZeroKernelSize)` instead of panicking
/// from `(kernel_size - 1)` usize underflow. See #599.
#[test]
fn test_build_conv1d_kernel_size_zero() {
    use crate::tensor_ir::TensorIRError;
    let err = build_conv1d("conv1d_ks0", 4, 2, 0, 8, 1, 0, false).unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dZeroKernelSize
            ))
        ),
        "expected Conv1dZeroKernelSize, got: {err}"
    );
}

/// Large dilation × kernel_size overflows usize; checked arithmetic catches it. See #599.
#[test]
fn test_build_conv1d_arithmetic_overflow() {
    use crate::tensor_ir::TensorIRError;
    let err = build_conv1d_full(
        "conv1d_overflow",
        4,
        2,
        3,
        16,
        1,
        0,
        usize::MAX, // dilation * (kernel_size - 1) overflows checked_mul
        1,
        false,
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::Conv1dArithmeticOverflow { .. }
            ))
        ),
        "expected Conv1dArithmeticOverflow, got: {err}"
    );
}

/// Verifies that the builder stores correct op parameters (stride, padding,
/// dilation, groups) in the Conv1d node, not just the output shape.
/// A bug that computes the right shape but stores wrong parameters would
/// silently break codegen and verification. See P1 iter 118 reflection.
#[test]
fn test_build_conv1d_stores_correct_op_params() {
    // dilation=3, groups=2, stride=2, padding=1
    let def = build_conv1d_full("conv1d_params", 4, 4, 3, 16, 2, 1, 3, 2, true).expect("build");

    // Find the Conv1d op node
    let conv_node = def
        .nodes
        .iter()
        .find(|n| matches!(n.kind, TensorOpKind::Conv1d { .. }))
        .expect("Conv1d node must exist");

    match &conv_node.kind {
        TensorOpKind::Conv1d {
            stride,
            padding,
            dilation,
            groups,
            ..
        } => {
            assert_eq!(*stride, 2, "stride must be stored as 2");
            assert_eq!(*padding, 1, "padding must be stored as 1");
            assert_eq!(*dilation, 3, "dilation must be stored as 3");
            assert_eq!(*groups, 2, "groups must be stored as 2");
        }
        other => panic!("expected Conv1d op, got: {other:?}"),
    }
}
