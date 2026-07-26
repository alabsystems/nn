#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for layers::ConvTranspose2d layer and DynTensor::conv_transpose2d op.

use crate::dyn_tensor::DynTensor;
use crate::layers::{ConvTranspose2d, ConvTranspose2dConfig, Module};
use crate::Device;

fn make_tensor(data: &[f32], shape: &[usize]) -> DynTensor {
    DynTensor::new(data, shape, &Device::Cpu).unwrap()
}

// -- DynTensor conv_transpose2d op tests --------------------------------------

#[test]
fn test_conv_transpose2d_basic_identity() {
    // 1 batch, 1 in_channel, 2x2 input, 1 out_channel, 1x1 kernel (weight=1)
    // ConvTranspose2d with 1x1 kernel, stride=1 should be identity
    let input = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let kernel = make_tensor(&[1.0], &[1, 1, 1, 1]);

    let output = input
        .conv_transpose2d(&kernel, [0, 0], [0, 0], [1, 1], [1, 1], 1)
        .unwrap();

    assert_eq!(output.dims(), &[1, 1, 2, 2]);
    let vals = output.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_conv_transpose2d_stride2_upsample() {
    // ConvTranspose2d with stride=2, 1x1 kernel acts as upsampling with zeros
    // Input: [1, 1, 2, 2], kernel: [1, 1, 1, 1] weight=1, stride=2
    // Output: (2-1)*2 + 1 = 3 per dim → [1, 1, 3, 3]
    let input = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let kernel = make_tensor(&[1.0], &[1, 1, 1, 1]);

    let output = input
        .conv_transpose2d(&kernel, [0, 0], [0, 0], [2, 2], [1, 1], 1)
        .unwrap();

    assert_eq!(output.dims(), &[1, 1, 3, 3]);
    let vals = output.to_flat_vec::<f32>().unwrap();
    // Stride-2 places input values at even positions, zeros elsewhere
    assert_eq!(vals, vec![1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0,]);
}

#[test]
fn test_conv_transpose2d_2x2_kernel() {
    // 1 batch, 1 channel, 2x2 input, 2x2 kernel (all ones), stride=1
    // Output: (2-1)*1 + 1*(2-1) + 1 = 3 per dim → [1, 1, 3, 3]
    let input = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let kernel = make_tensor(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]);

    let output = input
        .conv_transpose2d(&kernel, [0, 0], [0, 0], [1, 1], [1, 1], 1)
        .unwrap();

    assert_eq!(output.dims(), &[1, 1, 3, 3]);
    let vals = output.to_flat_vec::<f32>().unwrap();
    // Each input pixel spreads to a 2x2 area; overlapping regions sum.
    // Position (0,0): 1
    // Position (0,1): 1+2 = 3
    // Position (0,2): 2
    // Position (1,0): 1+3 = 4
    // Position (1,1): 1+2+3+4 = 10
    // Position (1,2): 2+4 = 6
    // Position (2,0): 3
    // Position (2,1): 3+4 = 7
    // Position (2,2): 4
    assert_eq!(vals, vec![1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0,]);
}

#[test]
fn test_conv_transpose2d_with_padding() {
    // Padding reduces the output size.
    // Output: (2-1)*1 + 1*(2-1) + 1 - 2*1 = 1 per dim → [1, 1, 1, 1]
    let input = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let kernel = make_tensor(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]);

    let output = input
        .conv_transpose2d(&kernel, [1, 1], [0, 0], [1, 1], [1, 1], 1)
        .unwrap();

    assert_eq!(output.dims(), &[1, 1, 1, 1]);
    let vals = output.to_flat_vec::<f32>().unwrap();
    // With padding=1, only the center position survives: 1+2+3+4 = 10
    assert_eq!(vals, vec![10.0]);
}

#[test]
fn test_conv_transpose2d_multiple_channels() {
    // 1 batch, 2 in_channels → 3 out_channels, 1x1 kernel
    // Kernel: [2, 3, 1, 1] — in_ch=2 maps to out_ch=3
    let input = make_tensor(
        &[
            1.0, 2.0, // ic0: 1x2
            3.0, 4.0, // ic1: 1x2
        ],
        &[1, 2, 1, 2],
    );
    // Kernel: ic0→[oc0=1, oc1=0, oc2=1], ic1→[oc0=0, oc1=1, oc2=1]
    let kernel = make_tensor(
        &[
            1.0, 0.0, 1.0, // ic0 → oc0, oc1, oc2
            0.0, 1.0, 1.0, // ic1 → oc0, oc1, oc2
        ],
        &[2, 3, 1, 1],
    );

    let output = input
        .conv_transpose2d(&kernel, [0, 0], [0, 0], [1, 1], [1, 1], 1)
        .unwrap();

    assert_eq!(output.dims(), &[1, 3, 1, 2]);
    let vals = output.to_flat_vec::<f32>().unwrap();
    // oc0: 1*ic0 + 0*ic1 = [1,2]
    // oc1: 0*ic0 + 1*ic1 = [3,4]
    // oc2: 1*ic0 + 1*ic1 = [4,6]
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 4.0, 6.0]);
}

#[test]
fn test_conv_transpose2d_batch() {
    // 2 batches, 1 channel, 1x1 input, 1x1 kernel (weight=3)
    let input = make_tensor(&[2.0, 5.0], &[2, 1, 1, 1]);
    let kernel = make_tensor(&[3.0], &[1, 1, 1, 1]);

    let output = input
        .conv_transpose2d(&kernel, [0, 0], [0, 0], [1, 1], [1, 1], 1)
        .unwrap();

    assert_eq!(output.dims(), &[2, 1, 1, 1]);
    let vals = output.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![6.0, 15.0]);
}

#[test]
fn test_conv_transpose2d_with_bias_nn_layer() {
    let input = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let kernel = make_tensor(&[1.0], &[1, 1, 1, 1]);
    let bias = make_tensor(&[10.0], &[1]);

    let conv = ConvTranspose2d::new(kernel, Some(bias), ConvTranspose2dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 1, 2, 2]);
    let vals = output.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn test_conv_transpose2d_stride2_nn_layer() {
    let input = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let kernel = make_tensor(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]);

    let config = ConvTranspose2dConfig {
        stride: [2, 2],
        ..Default::default()
    };
    let conv = ConvTranspose2d::new(kernel, None, config).unwrap();
    let output = conv.forward(&input).unwrap();

    // Output: (2-1)*2 + 1*(2-1) + 1 = 4 per dim → [1, 1, 4, 4]
    assert_eq!(output.dims(), &[1, 1, 4, 4]);
}

// -- Validation tests ---------------------------------------------------------

#[test]
fn test_conv_transpose2d_invalid_rank() {
    let weight = make_tensor(&[1.0, 2.0, 3.0], &[3]);
    let result = ConvTranspose2d::new(weight, None, ConvTranspose2dConfig::default());
    assert!(result.is_err());
}

#[test]
fn test_conv_transpose2d_zero_groups() {
    let weight = make_tensor(&[1.0], &[1, 1, 1, 1]);
    let config = ConvTranspose2dConfig {
        groups: 0,
        ..Default::default()
    };
    let result = ConvTranspose2d::new(weight, None, config);
    assert!(result.is_err());
}

#[test]
fn test_conv_transpose2d_zero_stride() {
    let input = make_tensor(&[1.0], &[1, 1, 1, 1]);
    let kernel = make_tensor(&[1.0], &[1, 1, 1, 1]);
    let result = input.conv_transpose2d(&kernel, [0, 0], [0, 0], [0, 0], [1, 1], 1);
    assert!(result.is_err());
}

#[test]
fn test_conv_transpose2d_zero_dilation() {
    let input = make_tensor(&[1.0], &[1, 1, 1, 1]);
    let kernel = make_tensor(&[1.0], &[1, 1, 1, 1]);
    let result = input.conv_transpose2d(&kernel, [0, 0], [0, 0], [1, 1], [0, 0], 1);
    assert!(result.is_err());
}

#[test]
fn test_conv_transpose2d_channel_mismatch() {
    // Input has 2 in_channels, kernel expects 3
    let input = make_tensor(&[1.0; 8], &[1, 2, 2, 2]);
    let kernel = make_tensor(&[1.0; 3], &[3, 1, 1, 1]);
    let result = input.conv_transpose2d(&kernel, [0, 0], [0, 0], [1, 1], [1, 1], 1);
    assert!(result.is_err());
}

#[test]
fn test_conv_transpose2d_output_padding_ge_stride() {
    // output_padding must be < stride
    let input = make_tensor(&[1.0], &[1, 1, 1, 1]);
    let kernel = make_tensor(&[1.0], &[1, 1, 1, 1]);
    let result = input.conv_transpose2d(&kernel, [0, 0], [2, 2], [2, 2], [1, 1], 1);
    assert!(result.is_err());
}

// -- Config defaults ----------------------------------------------------------

#[test]
fn test_conv_transpose2d_config_default() {
    let config = ConvTranspose2dConfig::default();
    assert_eq!(config.padding, [0, 0]);
    assert_eq!(config.output_padding, [0, 0]);
    assert_eq!(config.stride, [1, 1]);
    assert_eq!(config.dilation, [1, 1]);
    assert_eq!(config.groups, 1);
}

#[test]
fn test_conv_transpose2d_accessors() {
    let weight = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let bias = make_tensor(&[5.0], &[1]);
    let config = ConvTranspose2dConfig {
        stride: [2, 2],
        padding: [1, 1],
        ..Default::default()
    };
    let conv = ConvTranspose2d::new(weight, Some(bias), config).unwrap();

    assert_eq!(conv.weight().dims(), &[1, 1, 2, 2]);
    assert!(conv.bias().is_some());
    assert_eq!(conv.config().stride, [2, 2]);
    assert_eq!(conv.config().padding, [1, 1]);
}

// -- Groups support -----------------------------------------------------------

#[test]
fn test_conv_transpose2d_groups() {
    // 1 batch, 2 in_channels, 1x1 input, 2 out_channels, 1x1 kernel, groups=2
    // Group 0: ic0 → oc0, Group 1: ic1 → oc1
    let input = make_tensor(&[3.0, 7.0], &[1, 2, 1, 1]);
    // Kernel: [2, 1, 1, 1] — each group has 1 in_ch → 1 out_ch
    let kernel = make_tensor(&[2.0, 5.0], &[2, 1, 1, 1]);

    let output = input
        .conv_transpose2d(&kernel, [0, 0], [0, 0], [1, 1], [1, 1], 2)
        .unwrap();

    assert_eq!(output.dims(), &[1, 2, 1, 1]);
    let vals = output.to_flat_vec::<f32>().unwrap();
    // oc0: 3*2 = 6, oc1: 7*5 = 35
    assert_eq!(vals, vec![6.0, 35.0]);
}

// -- Dilation support ---------------------------------------------------------

#[test]
fn test_conv_transpose2d_dilation() {
    // 1 batch, 1 channel, 2x2 input, 2x2 kernel, dilation=2
    // Output per dim: (2-1)*1 + 2*(2-1) + 1 = 4 → [1, 1, 4, 4]
    let input = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let kernel = make_tensor(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]);

    let output = input
        .conv_transpose2d(&kernel, [0, 0], [0, 0], [1, 1], [2, 2], 1)
        .unwrap();

    assert_eq!(output.dims(), &[1, 1, 4, 4]);
    let vals = output.to_flat_vec::<f32>().unwrap();
    // With dilation=2, each input pixel spreads to positions offset by 2
    // (0,0)=1 contributes to (0,0),(0,2),(2,0),(2,2)
    // (0,1)=2 contributes to (0,1),(0,3),(2,1),(2,3)
    // (1,0)=3 contributes to (1,0),(1,2),(3,0),(3,2)
    // (1,1)=4 contributes to (1,1),(1,3),(3,1),(3,3)
    #[rustfmt::skip]
    let expected = vec![
        1.0, 2.0, 1.0, 2.0,
        3.0, 4.0, 3.0, 4.0,
        1.0, 2.0, 1.0, 2.0,
        3.0, 4.0, 3.0, 4.0,
    ];
    assert_eq!(vals, expected);
}
