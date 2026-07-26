// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `build_conv2d` and `build_conv2d_full`.

use super::{build_conv2d, build_conv2d_full};
use crate::tensor_ir::{TensorIRConvError, TensorIRError, TensorIRLayerError};

#[test]
fn test_conv2d_basic_3x3() {
    // Conv2d: [4, 8, 8] @ [2, 4, 3, 3] -> [2, 6, 6], stride=1, pad=0
    let def = build_conv2d("conv2d_basic", 4, 2, 3, 3, 8, 8, 1, 1, 0, 0, false).expect("build");
    let output = def.nodes.last().unwrap();
    assert_eq!(output.shape, vec![2, 6, 6]);
}

#[test]
fn test_conv2d_with_bias() {
    let def = build_conv2d("conv2d_bias", 4, 2, 3, 3, 8, 8, 1, 1, 0, 0, true).expect("build");
    // Should have 4 nodes: data, weight, bias, conv2d
    assert_eq!(def.nodes.len(), 4);
    let output = def.nodes.last().unwrap();
    assert_eq!(output.shape, vec![2, 6, 6]);
}

#[test]
fn test_conv2d_stride_padding() {
    // Demucs spectral pattern: 3×3 kernel, stride=1, pad=1 (same padding)
    // in: [48, 16, 16] -> out: [96, 16, 16]
    let def =
        build_conv2d("conv2d_demucs", 48, 96, 3, 3, 16, 16, 1, 1, 1, 1, false).expect("build");
    let output = def.nodes.last().unwrap();
    assert_eq!(output.shape, vec![96, 16, 16]);
}

#[test]
fn test_conv2d_asymmetric_kernel() {
    // 1×3 kernel (width-only convolution)
    let def = build_conv2d("conv2d_1x3", 4, 8, 1, 3, 10, 10, 1, 1, 0, 0, false).expect("build");
    let output = def.nodes.last().unwrap();
    // out_h = (10 - 1)/1 + 1 = 10, out_w = (10 - 3)/1 + 1 = 8
    assert_eq!(output.shape, vec![8, 10, 8]);
}

#[test]
fn test_conv2d_stride_2() {
    // Downsampling: stride=2, kernel=3, padding=1
    // out_h = (8 + 2 - 3)/2 + 1 = 4, out_w = (8 + 2 - 3)/2 + 1 = 4
    let def = build_conv2d("conv2d_ds", 4, 8, 3, 3, 8, 8, 2, 2, 1, 1, false).expect("build");
    let output = def.nodes.last().unwrap();
    assert_eq!(output.shape, vec![8, 4, 4]);
}

#[test]
fn test_conv2d_1x1_pointwise() {
    // Pointwise: 1×1 kernel
    let def = build_conv2d("conv2d_1x1", 48, 96, 1, 1, 16, 16, 1, 1, 0, 0, true).expect("build");
    let output = def.nodes.last().unwrap();
    assert_eq!(output.shape, vec![96, 16, 16]);
}

#[test]
fn test_conv2d_dilation() {
    // dilation=2, kernel=3: effective kernel = 2*(3-1)+1 = 5
    // out_h = (8 - 5)/1 + 1 = 4, out_w = (8 - 5)/1 + 1 = 4
    let def = build_conv2d_full(
        "conv2d_dilated",
        4,
        2,
        3,
        3,
        8,
        8,
        1,
        1,
        0,
        0,
        2,
        2,
        1,
        false,
    )
    .expect("build");
    let output = def.nodes.last().unwrap();
    assert_eq!(output.shape, vec![2, 4, 4]);
}

#[test]
fn test_conv2d_zero_stride_error() {
    let err = build_conv2d("bad", 4, 2, 3, 3, 8, 8, 0, 1, 0, 0, false).unwrap_err();
    assert!(matches!(
        err,
        TensorIRError::Layer(TensorIRLayerError::Conv(
            TensorIRConvError::Conv2dZeroStride { .. }
        ))
    ));
}

#[test]
fn test_conv2d_zero_kernel_error() {
    let err = build_conv2d("bad", 4, 2, 0, 3, 8, 8, 1, 1, 0, 0, false).unwrap_err();
    assert!(matches!(
        err,
        TensorIRError::Layer(TensorIRLayerError::Conv(
            TensorIRConvError::Conv2dZeroKernelSize { .. }
        ))
    ));
}

#[test]
fn test_conv2d_zero_dilation_error() {
    let err = build_conv2d_full("bad", 4, 2, 3, 3, 8, 8, 1, 1, 0, 0, 0, 1, 1, false).unwrap_err();
    assert!(matches!(
        err,
        TensorIRError::Layer(TensorIRLayerError::Conv(
            TensorIRConvError::Conv2dZeroDilation { .. }
        ))
    ));
}

#[test]
fn test_conv2d_zero_groups_error() {
    let err = build_conv2d_full("bad", 4, 2, 3, 3, 8, 8, 1, 1, 0, 0, 1, 1, 0, false).unwrap_err();
    assert!(matches!(
        err,
        TensorIRError::Layer(TensorIRLayerError::Conv(
            TensorIRConvError::Conv2dZeroGroups
        ))
    ));
}

#[test]
fn test_conv2d_groups_channel_mismatch_error() {
    // 4 in_channels not divisible by 3 groups
    let err = build_conv2d_full("bad", 4, 6, 3, 3, 8, 8, 1, 1, 0, 0, 1, 1, 3, false).unwrap_err();
    assert!(matches!(
        err,
        TensorIRError::Layer(TensorIRLayerError::Conv(
            TensorIRConvError::Conv2dGroupsChannelMismatch { .. }
        ))
    ));
}

#[test]
fn test_conv2d_kernel_too_large_error() {
    // kernel 5×5 on 3×3 input with no padding: padded_h=3 < eff_kh=5
    let err = build_conv2d("bad", 4, 2, 5, 5, 3, 3, 1, 1, 0, 0, false).unwrap_err();
    assert!(matches!(
        err,
        TensorIRError::Layer(TensorIRLayerError::Conv(
            TensorIRConvError::Conv2dArithmeticOverflow { .. }
        ))
    ));
}

#[test]
fn test_conv2d_node_count_no_bias() {
    let def = build_conv2d("t", 4, 2, 3, 3, 8, 8, 1, 1, 0, 0, false).expect("build");
    // data + weight + conv2d = 3 nodes
    assert_eq!(def.nodes.len(), 3);
}

#[test]
fn test_conv2d_node_count_with_bias() {
    let def = build_conv2d("t", 4, 2, 3, 3, 8, 8, 1, 1, 0, 0, true).expect("build");
    // data + weight + bias + conv2d = 4 nodes
    assert_eq!(def.nodes.len(), 4);
}

#[test]
fn test_conv2d_weight_shape() {
    let def = build_conv2d("t", 4, 8, 3, 5, 10, 10, 1, 1, 0, 0, false).expect("build");
    // Weight node should be [out_ch=8, in_ch=4, kH=3, kW=5]
    assert_eq!(def.nodes[1].shape, vec![8, 4, 3, 5]);
}
