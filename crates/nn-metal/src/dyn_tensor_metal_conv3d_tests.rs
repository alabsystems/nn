// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU Conv3d tests: verify Metal dispatch produces correct results vs CPU reference.
//!
//! Part of #3866.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_close, init};

/// 1x1x1 identity-like kernel with stride 1 — output should match input channels.
#[test]
fn test_conv3d_gpu_identity_kernel() {
    init();
    // Input: [1, 2, 3, 3, 3] (batch=1, channels=2, depth=3, height=3, width=3)
    let input_data: Vec<f32> = (0..54).map(|i| i as f32 * 0.1).collect();
    let cpu_input = DynTensor::new(&input_data, &[1, 2, 3, 3, 3], &Device::Cpu).unwrap();

    // Kernel: [4, 2, 1, 1, 1] (out_ch=4, in_ch=2, kD=1, kH=1, kW=1)
    let kernel_data: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) * 0.1).collect();
    let cpu_kernel = DynTensor::new(&kernel_data, &[4, 2, 1, 1, 1], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    let padding = [0, 0, 0];
    let stride = [1, 1, 1];
    let dilation = [1, 1, 1];

    let gpu_result = gpu_input
        .conv3d(&gpu_kernel, padding, stride, dilation, 1)
        .unwrap();
    let cpu_result = cpu_input
        .conv3d(&cpu_kernel, padding, stride, dilation, 1)
        .unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.dims(), &[1, 4, 3, 3, 3]);

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "conv3d_identity_kernel");
}

/// Random input, compare GPU vs CPU output for correctness.
#[test]
fn test_conv3d_gpu_matches_cpu() {
    init();
    // Input: [1, 3, 4, 4, 4]
    let input_data: Vec<f32> = (0..192).map(|i| ((i * 7 + 3) % 100) as f32 * 0.01).collect();
    let cpu_input = DynTensor::new(&input_data, &[1, 3, 4, 4, 4], &Device::Cpu).unwrap();

    // Kernel: [2, 3, 3, 3, 3] (out_ch=2, in_ch=3, 3x3x3 kernel)
    let kernel_data: Vec<f32> = (0..162).map(|i| ((i * 13 + 5) % 100) as f32 * 0.01 - 0.5).collect();
    let cpu_kernel = DynTensor::new(&kernel_data, &[2, 3, 3, 3, 3], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    let padding = [0, 0, 0];
    let stride = [1, 1, 1];
    let dilation = [1, 1, 1];

    let gpu_result = gpu_input
        .conv3d(&gpu_kernel, padding, stride, dilation, 1)
        .unwrap();
    let cpu_result = cpu_input
        .conv3d(&cpu_kernel, padding, stride, dilation, 1)
        .unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "conv3d_matches_cpu");
}

/// Qwen3-VL patch embedding: [1, 3, 2, 14, 14] input with [2, 14, 14] kernel.
///
/// This is the exact shape used by Qwen3-VL vision encoder for 3D patch embedding
/// where the temporal dimension (depth=2) and spatial dimensions (14x14) are the
/// kernel size, producing a single output point per patch.
#[test]
fn test_conv3d_gpu_qwen3vl_patch() {
    init();
    // Input: [1, 3, 2, 14, 14] (batch=1, rgb=3, temporal=2, h=14, w=14)
    let n_in = 1 * 3 * 2 * 14 * 14;
    let input_data: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.001) - 0.5).collect();
    let cpu_input = DynTensor::new(&input_data, &[1, 3, 2, 14, 14], &Device::Cpu).unwrap();

    // Kernel: [1152, 3, 2, 14, 14] (out_ch=1152, in_ch=3, kD=2, kH=14, kW=14)
    // This is the full Qwen3-VL patch embedding kernel.
    // Use a smaller out_ch for test speed: out_ch=8.
    let out_ch = 8;
    let n_ker = out_ch * 3 * 2 * 14 * 14;
    let kernel_data: Vec<f32> = (0..n_ker).map(|i| ((i * 17 + 3) % 200) as f32 * 0.001 - 0.1).collect();
    let cpu_kernel = DynTensor::new(&kernel_data, &[out_ch, 3, 2, 14, 14], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    let padding = [0, 0, 0];
    let stride = [2, 14, 14];
    let dilation = [1, 1, 1];

    let gpu_result = gpu_input
        .conv3d(&gpu_kernel, padding, stride, dilation, 1)
        .unwrap();
    let cpu_result = cpu_input
        .conv3d(&cpu_kernel, padding, stride, dilation, 1)
        .unwrap();

    // Output: [1, 8, 1, 1, 1] — one output per patch.
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.dims(), &[1, out_ch, 1, 1, 1]);

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-3, "conv3d_qwen3vl_patch");
}

/// Conv3d with padding.
#[test]
fn test_conv3d_gpu_with_padding() {
    init();
    let input_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let cpu_input = DynTensor::new(&input_data, &[1, 1, 2, 3, 4], &Device::Cpu).unwrap();

    let kernel_data: Vec<f32> = vec![1.0; 27]; // [1, 1, 3, 3, 3]
    let cpu_kernel = DynTensor::new(&kernel_data, &[1, 1, 3, 3, 3], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    let padding = [1, 1, 1];
    let stride = [1, 1, 1];
    let dilation = [1, 1, 1];

    let gpu_result = gpu_input
        .conv3d(&gpu_kernel, padding, stride, dilation, 1)
        .unwrap();
    let cpu_result = cpu_input
        .conv3d(&cpu_kernel, padding, stride, dilation, 1)
        .unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "conv3d_with_padding");
}

/// Conv3d with stride > 1.
#[test]
fn test_conv3d_gpu_with_stride() {
    init();
    let input_data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let cpu_input = DynTensor::new(&input_data, &[1, 1, 4, 4, 4], &Device::Cpu).unwrap();

    let kernel_data: Vec<f32> = vec![0.5; 16]; // [2, 1, 2, 2, 2] = 16 elements
    let cpu_kernel = DynTensor::new(&kernel_data, &[2, 1, 2, 2, 2], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    let padding = [0, 0, 0];
    let stride = [2, 2, 2];
    let dilation = [1, 1, 1];

    let gpu_result = gpu_input
        .conv3d(&gpu_kernel, padding, stride, dilation, 1)
        .unwrap();
    let cpu_result = cpu_input
        .conv3d(&cpu_kernel, padding, stride, dilation, 1)
        .unwrap();

    // Stride 2 on 4x4x4 with 2x2x2 kernel → 2x2x2 output per channel.
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.dims(), &[1, 2, 2, 2, 2]);

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "conv3d_with_stride");
}

/// Conv3d with batch > 1.
#[test]
fn test_conv3d_gpu_batched() {
    init();
    // Input: [2, 2, 3, 3, 3] (batch=2)
    let input_data: Vec<f32> = (0..108).map(|i| i as f32 * 0.05).collect();
    let cpu_input = DynTensor::new(&input_data, &[2, 2, 3, 3, 3], &Device::Cpu).unwrap();

    // Kernel: [2, 2, 2, 2, 2] = 32 elements
    let kernel_data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let cpu_kernel = DynTensor::new(&kernel_data, &[2, 2, 2, 2, 2], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    let padding = [0, 0, 0];
    let stride = [1, 1, 1];
    let dilation = [1, 1, 1];

    let gpu_result = gpu_input
        .conv3d(&gpu_kernel, padding, stride, dilation, 1)
        .unwrap();
    let cpu_result = cpu_input
        .conv3d(&cpu_kernel, padding, stride, dilation, 1)
        .unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.dims()[0], 2, "output batch must be 2");

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "conv3d_batched");
}

/// Verify output dimensions match the conv3d formula.
#[test]
fn test_conv3d_gpu_output_shape() {
    init();
    // Input: [1, 3, 8, 16, 16]
    let n_in = 1 * 3 * 8 * 16 * 16;
    let input_data: Vec<f32> = (0..n_in).map(|i| i as f32 * 0.001).collect();
    let cpu_input = DynTensor::new(&input_data, &[1, 3, 8, 16, 16], &Device::Cpu).unwrap();

    // Kernel: [6, 3, 3, 5, 5] with stride [2, 3, 3] and padding [1, 2, 2]
    let n_ker = 6 * 3 * 3 * 5 * 5;
    let kernel_data: Vec<f32> = (0..n_ker).map(|i| ((i * 11 + 7) % 100) as f32 * 0.01 - 0.5).collect();
    let cpu_kernel = DynTensor::new(&kernel_data, &[6, 3, 3, 5, 5], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    let padding = [1, 2, 2];
    let stride = [2, 3, 3];
    let dilation = [1, 1, 1];

    let gpu_result = gpu_input
        .conv3d(&gpu_kernel, padding, stride, dilation, 1)
        .unwrap();

    // Expected output dims (formula: (L + 2*P - D*(K-1) - 1) / S + 1):
    // D: (8 + 2*1 - 1*(3-1) - 1) / 2 + 1 = 7/2 + 1 = 4
    // H: (16 + 2*2 - 1*(5-1) - 1) / 3 + 1 = 15/3 + 1 = 6
    // W: (16 + 2*2 - 1*(5-1) - 1) / 3 + 1 = 15/3 + 1 = 6
    assert_eq!(gpu_result.dims(), &[1, 6, 4, 6, 6]);
}
