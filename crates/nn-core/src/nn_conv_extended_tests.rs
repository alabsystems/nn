// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for convolution layers: Conv1d, Conv2d, Conv3d,
//! ConvTranspose1d, ConvTranspose2d, and output length formulas.
//!
//! Part of #4186.

use crate::dyn_tensor::DynTensor;
use crate::layers::{
    Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Conv3d, Conv3dConfig, ConvTranspose2d,
    ConvTranspose2dConfig, Module,
};
use crate::test_prng::rand_f32_vec;
use crate::{
    conv1d_out_len, conv2d_out_len, conv3d_out_len, conv_transpose1d_out_len,
    conv_transpose2d_out_len, Device,
};

/// Helper: create a DynTensor with deterministic random data.
fn rand_tensor(seed: u64, dims: &[usize]) -> DynTensor {
    let numel: usize = dims.iter().product();
    let data = rand_f32_vec(seed, numel, -1.0, 1.0);
    DynTensor::from_vec(data, dims, &Device::Cpu).unwrap()
}

#[test]
fn test_conv1d_output_length() {
    // Formula: (input + 2*pad - dilation*(kernel-1) - 1) / stride + 1
    // input=16, kernel=3, pad=1, stride=1, dilation=1 => (16+2-2-1)/1+1 = 16
    let out = conv1d_out_len(16, 3, 1, 1, 1).unwrap();
    assert_eq!(out, 16);

    // input=16, kernel=3, pad=0, stride=2, dilation=1 => (16+0-2-1)/2+1 = 7
    let out = conv1d_out_len(16, 3, 0, 2, 1).unwrap();
    assert_eq!(out, 7);

    // input=32, kernel=5, pad=2, stride=1, dilation=1 => (32+4-4-1)/1+1 = 32
    let out = conv1d_out_len(32, 5, 2, 1, 1).unwrap();
    assert_eq!(out, 32);

    // Dilated: input=16, kernel=3, pad=0, stride=1, dilation=2
    // effective_kernel = 2*(3-1)+1 = 5
    // => (16+0-5)/1+1 = 12
    let out = conv1d_out_len(16, 3, 0, 1, 2).unwrap();
    assert_eq!(out, 12);
}

#[test]
fn test_conv2d_output_shape() {
    let n = 2;
    let c_in = 3;
    let c_out = 8;
    let h_in = 32;
    let w_in = 32;
    let kernel = 3;
    let padding = 1;
    let stride = 2;

    // Expected output spatial: (32 + 2*1 - 2 - 1)/2 + 1 = 16
    let h_out = conv2d_out_len(h_in, kernel, padding, stride, 1).unwrap();
    let w_out = conv2d_out_len(w_in, kernel, padding, stride, 1).unwrap();
    assert_eq!(h_out, 16);
    assert_eq!(w_out, 16);

    // Build Conv2d layer with zeros weight + no bias
    let weight = DynTensor::zeros(
        &[c_out, c_in, kernel, kernel],
        crate::DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let config = Conv2dConfig::new(padding, stride, 1);
    let conv = Conv2d::new(weight, None, config).unwrap();

    let input = rand_tensor(100, &[n, c_in, h_in, w_in]);
    let output = conv.forward(&input).unwrap();
    assert_eq!(output.dims(), &[n, c_out, h_out, w_out]);
}

#[test]
fn test_conv3d_output_shape() {
    let n = 1;
    let c_in = 3;
    let c_out = 16;
    let d_in = 8;
    let h_in = 16;
    let w_in = 16;
    let kernel = 3;
    let padding = 1;
    let stride = 2;

    let d_out = conv3d_out_len(d_in, kernel, padding, stride, 1).unwrap();
    let h_out = conv3d_out_len(h_in, kernel, padding, stride, 1).unwrap();
    let w_out = conv3d_out_len(w_in, kernel, padding, stride, 1).unwrap();
    assert_eq!(d_out, 4);
    assert_eq!(h_out, 8);
    assert_eq!(w_out, 8);

    let weight = DynTensor::zeros(
        &[c_out, c_in, kernel, kernel, kernel],
        crate::DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let config = Conv3dConfig::new(padding, stride, 1);
    let conv = Conv3d::new(weight, None, config).unwrap();

    let input = rand_tensor(200, &[n, c_in, d_in, h_in, w_in]);
    let output = conv.forward(&input).unwrap();
    assert_eq!(output.dims(), &[n, c_out, d_out, h_out, w_out]);
}

#[test]
fn test_conv_transpose1d_output_length() {
    // Formula: (input - 1) * stride - 2*padding + dilation*(kernel-1) + output_padding + 1
    // input=8, kernel=4, pad=1, output_pad=0, stride=2, dilation=1
    // => (8-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 14 - 2 + 3 + 1 = 16
    let out = conv_transpose1d_out_len(8, 4, 1, 0, 2, 1).unwrap();
    assert_eq!(out, 16);

    // input=4, kernel=4, pad=0, output_pad=0, stride=4, dilation=1
    // => (4-1)*4 - 0 + 3 + 0 + 1 = 12 + 4 = 16
    let out = conv_transpose1d_out_len(4, 4, 0, 0, 4, 1).unwrap();
    assert_eq!(out, 16);
}

#[test]
fn test_conv_transpose2d_output_shape() {
    let n = 1;
    let c_in = 16;
    let c_out = 8;
    let h_in = 8;
    let w_in = 8;
    let kernel = 4;
    let padding = 1;
    let stride = 2;

    // Expected: (8-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 16
    let h_out = conv_transpose2d_out_len(h_in, kernel, padding, 0, stride, 1).unwrap();
    let w_out = conv_transpose2d_out_len(w_in, kernel, padding, 0, stride, 1).unwrap();
    assert_eq!(h_out, 16);
    assert_eq!(w_out, 16);

    // Weight for ConvTranspose2d: [in_channels, out_channels/groups, kH, kW]
    let weight = DynTensor::zeros(
        &[c_in, c_out, kernel, kernel],
        crate::DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let config = ConvTranspose2dConfig::new(padding, stride, 1);
    let deconv = ConvTranspose2d::new(weight, None, config).unwrap();

    let input = rand_tensor(300, &[n, c_in, h_in, w_in]);
    let output = deconv.forward(&input).unwrap();
    assert_eq!(output.dims(), &[n, c_out, h_out, w_out]);
}

#[test]
fn test_conv1d_no_bias() {
    let n = 1;
    let c_in = 2;
    let c_out = 4;
    let kernel = 3;
    let length = 16;

    // Create Conv1d with no bias
    let weight = DynTensor::zeros(&[c_out, c_in, kernel], crate::DType::F32, &Device::Cpu).unwrap();
    let config = Conv1dConfig::new(1, 1, 1);
    let conv = Conv1d::new(weight, None, config).unwrap();
    assert!(conv.bias().is_none());

    let input = rand_tensor(400, &[n, c_in, length]);
    // With zero weights and no bias, output should be all zeros
    let output = conv.forward(&input).unwrap();
    let out_data = output.to_f32_array().unwrap();
    for &val in out_data.iter() {
        assert!(
            val.abs() < 1e-6,
            "Conv1d with zero weight and no bias should produce zeros, got {val}"
        );
    }
}

#[test]
fn test_conv2d_no_bias() {
    let n = 1;
    let c_in = 1;
    let c_out = 1;
    let kernel = 3;
    let h = 8;
    let w = 8;

    let weight = DynTensor::zeros(
        &[c_out, c_in, kernel, kernel],
        crate::DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let config = Conv2dConfig::new(1, 1, 1);
    let conv = Conv2d::new(weight, None, config).unwrap();
    assert!(conv.bias().is_none());

    let input = rand_tensor(500, &[n, c_in, h, w]);
    let output = conv.forward(&input).unwrap();
    let out_data = output.to_f32_array().unwrap();
    for &val in out_data.iter() {
        assert!(
            val.abs() < 1e-6,
            "Conv2d with zero weight and no bias should produce zeros, got {val}"
        );
    }
}

#[test]
fn test_conv1d_identity_kernel() {
    // A Conv1d with kernel_size=1, stride=1, padding=0, and identity-like weight
    // should be a passthrough (per-channel linear projection).
    // Weight [out_ch, in_ch, 1] = identity matrix reshaped.
    let n = 1;
    let channels = 2;
    let length = 8;

    // Build identity weight: weight[i][j][0] = 1 if i==j, else 0
    let mut weight_data = vec![0.0f32; channels * channels];
    for i in 0..channels {
        weight_data[i * channels + i] = 1.0;
    }
    let weight = DynTensor::from_vec(weight_data, &[channels, channels, 1], &Device::Cpu).unwrap();
    let config = Conv1dConfig::new(0, 1, 1);
    let conv = Conv1d::new(weight, None, config).unwrap();

    let input = rand_tensor(600, &[n, channels, length]);
    let output = conv.forward(&input).unwrap();

    // Output should equal input
    assert_eq!(output.dims(), input.dims());
    let in_data = input.to_f32_array().unwrap();
    let out_data = output.to_f32_array().unwrap();
    for (a, b) in in_data.iter().zip(out_data.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "Identity conv1d should be passthrough: {a} vs {b}"
        );
    }
}
