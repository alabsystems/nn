#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ConvTranspose1d GPU tests for dilation>1 and groups>1 (#1364).
//!
//! Extracted from `dyn_tensor_metal_shape_ops_conv_tests.rs` for file-size
//! compliance (#1402).

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_close, init};

// -- ConvTranspose1d dilation > 1 tests (#1364) --------------------------------

#[test]
fn test_gpu_conv_transpose1d_dilation2() {
    init();
    // Input: [1, 2, 4] (batch=1, in_ch=2, length=4)
    let input_data: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let cpu_input = DynTensor::new(&input_data, &[1, 2, 4], &Device::Cpu).unwrap();

    // Kernel: [2, 3, 3] (in_ch=2, out_ch=3, kernel_size=3)
    let kernel_data: Vec<f32> = (0..18).map(|i| (i as f32 - 9.0) * 0.05).collect();
    let cpu_kernel = DynTensor::new(&kernel_data, &[2, 3, 3], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    // stride=1, padding=0, output_padding=0, dilation=2, groups=1
    // out = (4-1)*1 + 2*(3-1) + 1 = 8
    let gpu_result = gpu_input
        .conv_transpose1d(&gpu_kernel, 0, 0, 1, 2, 1)
        .unwrap();
    let cpu_result = cpu_input
        .conv_transpose1d(&cpu_kernel, 0, 0, 1, 2, 1)
        .unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.device(), Device::metal());

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "conv_transpose1d_dilation2");
}

#[test]
fn test_gpu_conv_transpose1d_dilation3_stride2() {
    init();
    // Input: [1, 1, 6] (batch=1, in_ch=1, length=6)
    let input_data: Vec<f32> = vec![1.0, -0.5, 0.3, 0.8, -0.2, 0.6];
    let cpu_input = DynTensor::new(&input_data, &[1, 1, 6], &Device::Cpu).unwrap();

    // Kernel: [1, 2, 3] (in_ch=1, out_ch=2, kernel_size=3)
    let kernel_data: Vec<f32> = vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.2];
    let cpu_kernel = DynTensor::new(&kernel_data, &[1, 2, 3], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    // stride=2, padding=1, output_padding=0, dilation=3, groups=1
    // out = (6-1)*2 + 3*(3-1) + 1 - 2*1 = 10 + 6 + 1 - 2 = 15
    let gpu_result = gpu_input
        .conv_transpose1d(&gpu_kernel, 1, 0, 2, 3, 1)
        .unwrap();
    let cpu_result = cpu_input
        .conv_transpose1d(&cpu_kernel, 1, 0, 2, 3, 1)
        .unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(
        &gpu_vals,
        &cpu_vals,
        1e-4,
        "conv_transpose1d_dilation3_stride2",
    );
}

// -- ConvTranspose1d groups > 1 tests (#1364) --------------------------------

#[test]
fn test_gpu_conv_transpose1d_groups2() {
    init();
    // Input: [1, 4, 6] (batch=1, in_ch=4, length=6)
    let input_data: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.05).collect();
    let cpu_input = DynTensor::new(&input_data, &[1, 4, 6], &Device::Cpu).unwrap();

    // groups=2: in_ch_per_group=2, out_ch_per_group=3
    // Kernel: [4, 3, 3] (in_ch=4, out_ch_per_group=3, kernel_size=3)
    let kernel_data: Vec<f32> = (0..36).map(|i| (i as f32 - 18.0) * 0.02).collect();
    let cpu_kernel = DynTensor::new(&kernel_data, &[4, 3, 3], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    // stride=2, padding=1, output_padding=0, dilation=1, groups=2
    // out_ch = 3 * 2 = 6
    // out = (6-1)*2 + 1*(3-1) + 1 - 2*1 = 10 + 2 + 1 - 2 = 11
    let gpu_result = gpu_input
        .conv_transpose1d(&gpu_kernel, 1, 0, 2, 1, 2)
        .unwrap();
    let cpu_result = cpu_input
        .conv_transpose1d(&cpu_kernel, 1, 0, 2, 1, 2)
        .unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    // out_ch = 3 * 2 = 6
    assert_eq!(
        gpu_result.dims()[1],
        6,
        "output channels = 6 (3 per group * 2 groups)"
    );

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "conv_transpose1d_groups2");
}

#[test]
fn test_gpu_conv_transpose1d_depthwise() {
    init();
    // Depthwise: groups=in_ch (each channel is its own group)
    // Input: [1, 3, 8] (batch=1, in_ch=3, length=8)
    let input_data: Vec<f32> = (0..24).map(|i| (i as f32).sin() * 0.3).collect();
    let cpu_input = DynTensor::new(&input_data, &[1, 3, 8], &Device::Cpu).unwrap();

    // groups=3: in_ch_per_group=1, out_ch_per_group=1
    // Kernel: [3, 1, 4] (in_ch=3, out_ch_per_group=1, kernel_size=4)
    let kernel_data: Vec<f32> = vec![
        0.2, -0.1, 0.3, 0.1, -0.2, 0.4, -0.3, 0.1, 0.5, -0.2, 0.1, 0.3,
    ];
    let cpu_kernel = DynTensor::new(&kernel_data, &[3, 1, 4], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    // stride=2, padding=1, output_padding=0, dilation=1, groups=3
    // out_ch = 1 * 3 = 3
    // out = (8-1)*2 + 1*(4-1) + 1 - 2*1 = 14 + 3 + 1 - 2 = 16
    let gpu_result = gpu_input
        .conv_transpose1d(&gpu_kernel, 1, 0, 2, 1, 3)
        .unwrap();
    let cpu_result = cpu_input
        .conv_transpose1d(&cpu_kernel, 1, 0, 2, 1, 3)
        .unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.dims()[1], 3, "depthwise: out_ch = in_ch");

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "conv_transpose1d_depthwise");
}

#[test]
fn test_gpu_conv_transpose1d_dilation2_groups2() {
    init();
    // Combined: dilation=2 AND groups=2
    // Input: [1, 4, 4] (batch=1, in_ch=4, length=4)
    let input_data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
    let cpu_input = DynTensor::new(&input_data, &[1, 4, 4], &Device::Cpu).unwrap();

    // groups=2: in_ch_per_group=2, out_ch_per_group=2
    // Kernel: [4, 2, 3] (in_ch=4, out_ch_per_group=2, kernel_size=3)
    let kernel_data: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.03).collect();
    let cpu_kernel = DynTensor::new(&kernel_data, &[4, 2, 3], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    // stride=1, padding=0, output_padding=0, dilation=2, groups=2
    // out_ch = 2 * 2 = 4
    // out = (4-1)*1 + 2*(3-1) + 1 - 0 = 3 + 4 + 1 = 8
    let gpu_result = gpu_input
        .conv_transpose1d(&gpu_kernel, 0, 0, 1, 2, 2)
        .unwrap();
    let cpu_result = cpu_input
        .conv_transpose1d(&cpu_kernel, 0, 0, 1, 2, 2)
        .unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.dims()[1], 4, "out_ch = 2 per group * 2 groups");

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(
        &gpu_vals,
        &cpu_vals,
        1e-4,
        "conv_transpose1d_dilation2_groups2",
    );
}

// -- ConvTranspose1d output_padding > 0 tests (#1957) -------------------------

#[test]
fn test_gpu_conv_transpose1d_output_padding1() {
    init();
    // Input: [1, 2, 4] (batch=1, in_ch=2, length=4)
    let input_data: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let cpu_input = DynTensor::new(&input_data, &[1, 2, 4], &Device::Cpu).unwrap();

    // Kernel: [2, 3, 3] (in_ch=2, out_ch=3, kernel_size=3)
    let kernel_data: Vec<f32> = (0..18).map(|i| (i as f32 - 9.0) * 0.05).collect();
    let cpu_kernel = DynTensor::new(&kernel_data, &[2, 3, 3], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    // stride=2, padding=1, output_padding=1, dilation=1, groups=1
    // out = (4-1)*2 + 1*(3-1) + 1 - 2*1 + 1 = 6 + 2 + 1 - 2 + 1 = 8
    let gpu_result = gpu_input
        .conv_transpose1d(&gpu_kernel, 1, 1, 2, 1, 1)
        .unwrap();
    let cpu_result = cpu_input
        .conv_transpose1d(&cpu_kernel, 1, 1, 2, 1, 1)
        .unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.device(), Device::metal());
    assert_eq!(
        gpu_result.dims()[2],
        8,
        "output length with output_padding=1"
    );

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(
        &gpu_vals,
        &cpu_vals,
        1e-4,
        "conv_transpose1d_output_padding1",
    );
}
