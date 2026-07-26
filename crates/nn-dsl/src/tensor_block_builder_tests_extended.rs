// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for `TensorBlockBuilder` — broadcast, reduce, pad, conv2d ops.
//!
//! Extracted from `tensor_block_builder_tests.rs` (#175 pattern, Part of #824).

use super::*;

// ===========================================================================
// add_broadcast_left tests (#820 AC1)
// ===========================================================================

/// Left-aligned broadcast: [C] → [C, T] aligns channel dim to leftmost axis.
#[test]
fn test_builder_broadcast_left_basic() {
    let mut b = TensorBlockBuilder::new("bc_left");
    let x = b.add_input("gamma", &[4]);

    let bc = b.add_broadcast_left(x, &[4, 16]);
    let def = b.build(bc).expect("valid graph");

    // 1 input + 1 broadcast = 2 nodes
    assert_eq!(def.nodes.len(), 2);
    assert_eq!(def.nodes.last().unwrap().shape, vec![4, 16]);

    match &def.nodes[1].kind {
        TensorOpKind::Broadcast {
            input,
            target_shape,
            alignment,
        } => {
            assert_eq!(*input, TensorNodeId::new(0));
            assert_eq!(target_shape, &[4, 16]);
            assert_eq!(*alignment, BroadcastAlignment::Left);
        }
        other => panic!("expected Broadcast, got {other:?}"),
    }
}

/// Left-aligned broadcast into 3D: [C] → [C, H, W].
#[test]
fn test_builder_broadcast_left_3d() {
    let mut b = TensorBlockBuilder::new("bc_left_3d");
    let gamma = b.add_input("gamma", &[16]);

    let bc = b.add_broadcast_left(gamma, &[16, 8, 8]);
    let def = b.build(bc).expect("valid graph");

    assert_eq!(def.nodes.len(), 2);
    assert_eq!(def.nodes.last().unwrap().shape, vec![16, 8, 8]);
    assert!(matches!(
        &def.nodes[1].kind,
        TensorOpKind::Broadcast { alignment, .. }
            if *alignment == BroadcastAlignment::Left
    ));
}

// ===========================================================================
// add_reduce tests (#820 AC2)
// ===========================================================================

/// Reduce Mean along axis 1: [4, 8] → [4] (axis removed).
#[test]
fn test_builder_reduce_mean() {
    let mut b = TensorBlockBuilder::new("reduce_mean");
    let x = b.add_input("x", &[4, 8]);

    // Reduce removes the axis entirely: [4, 8] → [4]
    let reduced = b.add_reduce(x, ReduceOp::Mean, 1, false, &[4]);
    let def = b.build(reduced).expect("valid graph");

    assert_eq!(def.nodes.len(), 2);
    assert_eq!(def.nodes.last().unwrap().shape, vec![4]);

    match &def.nodes[1].kind {
        TensorOpKind::Reduce {
            op,
            input,
            axis,
            keepdim,
        } => {
            assert_eq!(*op, ReduceOp::Mean);
            assert_eq!(*input, TensorNodeId::new(0));
            assert_eq!(*axis, 1);
            assert!(!keepdim);
        }
        other => panic!("expected Reduce, got {other:?}"),
    }
}

/// Reduce Sum along axis 0: [4, 8] → [8] (axis removed).
#[test]
fn test_builder_reduce_sum() {
    let mut b = TensorBlockBuilder::new("reduce_sum");
    let x = b.add_input("x", &[4, 8]);

    // Reduce removes axis 0: [4, 8] → [8]
    let reduced = b.add_reduce(x, ReduceOp::Sum, 0, false, &[8]);
    let def = b.build(reduced).expect("valid graph");

    assert_eq!(def.nodes.len(), 2);
    assert_eq!(def.nodes.last().unwrap().shape, vec![8]);

    match &def.nodes[1].kind {
        TensorOpKind::Reduce {
            op,
            input,
            axis,
            keepdim,
        } => {
            assert_eq!(*op, ReduceOp::Sum);
            assert_eq!(*input, TensorNodeId::new(0));
            assert_eq!(*axis, 0);
            assert!(!keepdim);
        }
        other => panic!("expected Reduce, got {other:?}"),
    }
}

// ===========================================================================
// add_zero_pad_1d tests (#820 AC3)
// ===========================================================================

/// ZeroPad1d: left-only causal padding on [C, T] tensor.
#[test]
fn test_builder_zero_pad_1d_left_only() {
    // Causal Conv1d pattern: pad_left=kernel_size-1, pad_right=0.
    let mut b = TensorBlockBuilder::new("causal_pad");
    let x = b.add_input("x", &[48, 16]);

    // kernel_size=8 → pad_left=7, output T: 16+7 = 23
    let padded = b.add_zero_pad_1d(x, 7, 0, &[48, 23]);
    let def = b.build(padded).expect("valid graph");

    assert_eq!(def.nodes.len(), 2);
    assert_eq!(def.nodes.last().unwrap().shape, vec![48, 23]);

    match &def.nodes[1].kind {
        TensorOpKind::ZeroPad1d {
            input,
            pad_left,
            pad_right,
        } => {
            assert_eq!(*input, TensorNodeId::new(0));
            assert_eq!(*pad_left, 7);
            assert_eq!(*pad_right, 0);
        }
        other => panic!("expected ZeroPad1d, got {other:?}"),
    }
}

/// ZeroPad1d with symmetric padding.
#[test]
fn test_builder_zero_pad_1d_symmetric() {
    let mut b = TensorBlockBuilder::new("sym_pad");
    let x = b.add_input("x", &[8, 32]);

    // Symmetric: pad_left=2, pad_right=2, output T: 32+4 = 36
    let padded = b.add_zero_pad_1d(x, 2, 2, &[8, 36]);
    let def = b.build(padded).expect("valid graph");

    assert_eq!(def.nodes.len(), 2);
    assert_eq!(def.nodes.last().unwrap().shape, vec![8, 36]);

    match &def.nodes[1].kind {
        TensorOpKind::ZeroPad1d {
            pad_left,
            pad_right,
            ..
        } => {
            assert_eq!(*pad_left, 2);
            assert_eq!(*pad_right, 2);
        }
        other => panic!("expected ZeroPad1d, got {other:?}"),
    }
}

// ===========================================================================
// add_conv2d_full tests (#820 AC4)
// ===========================================================================

/// Conv2d with full parameters: stride, padding, dilation, groups.
#[test]
fn test_builder_conv2d_full_basic() {
    let mut b = TensorBlockBuilder::new("conv2d_full");
    let data = b.add_input("data", &[3, 32, 32]);
    let weight = b.add_input("weight", &[16, 3, 3, 3]);

    // stride=1, padding=1, dilation=1, groups=1 → output [16, 32, 32]
    let conv = b.add_conv2d_full(data, weight, None, 1, 1, 1, 1, 1, 1, 1, &[16, 32, 32]);
    let def = b.build(conv).expect("valid graph");

    assert_eq!(def.nodes.len(), 3);
    assert_eq!(def.nodes.last().unwrap().shape, vec![16, 32, 32]);

    match &def.nodes[2].kind {
        TensorOpKind::Conv2d {
            input,
            weight,
            bias,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
        } => {
            assert_eq!(*input, TensorNodeId::new(0));
            assert_eq!(*weight, TensorNodeId::new(1));
            assert!(bias.is_none());
            assert_eq!(*stride_h, 1);
            assert_eq!(*stride_w, 1);
            assert_eq!(*padding_h, 1);
            assert_eq!(*padding_w, 1);
            assert_eq!(*dilation_h, 1);
            assert_eq!(*dilation_w, 1);
            assert_eq!(*groups, 1);
        }
        other => panic!("expected Conv2d, got {other:?}"),
    }
}

/// Conv2d with bias, stride=2, and dilation=2.
#[test]
fn test_builder_conv2d_full_with_bias_and_dilation() {
    let mut b = TensorBlockBuilder::new("conv2d_dilated");
    let data = b.add_input("data", &[3, 64, 64]);
    let weight = b.add_input("weight", &[32, 3, 3, 3]);
    let bias = b.add_input("bias", &[32]);

    // stride=(2,2), padding=(2,2), dilation=(2,2), groups=1
    // eff_k = dilation*(k-1)+1 = 2*(3-1)+1 = 5
    // out = (64 + 2*2 - 5)/2 + 1 = 63/2 + 1 = 32
    let conv = b.add_conv2d_full(data, weight, Some(bias), 2, 2, 2, 2, 2, 2, 1, &[32, 32, 32]);
    let def = b.build(conv).expect("valid graph");

    assert_eq!(def.nodes.len(), 4);
    assert_eq!(def.nodes.last().unwrap().shape, vec![32, 32, 32]);

    match &def.nodes[3].kind {
        TensorOpKind::Conv2d {
            bias,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            ..
        } => {
            assert!(bias.is_some(), "bias should be present");
            assert_eq!(*stride_h, 2);
            assert_eq!(*stride_w, 2);
            assert_eq!(*dilation_h, 2);
            assert_eq!(*dilation_w, 2);
        }
        other => panic!("expected Conv2d, got {other:?}"),
    }
}
