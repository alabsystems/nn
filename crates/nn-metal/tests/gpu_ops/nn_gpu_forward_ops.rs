// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU forward tests for activation, shape, and convolution ops.
//!
//! Split from nn_gpu_forward.rs for 500-line limit. Each test constructs a
//! tensor on GPU, runs an operation, and compares against CPU reference output.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

const TOL: f32 = 1e-4;

fn init() {
    gpu_init();
}

fn assert_close(gpu_result: &DynTensor, cpu_result: &DynTensor, label: &str) {
    assert_gpu_cpu_close(gpu_result, cpu_result, TOL, label);
}

// -- AC4: GPU sigmoid, gelu, tanh ---------------------------------------------

#[test]
fn test_sigmoid_gpu() {
    init();
    let x_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

    // CPU reference
    let x_cpu = DynTensor::new(&x_data, &[5], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.sigmoid().unwrap();

    // GPU
    let x_gpu = DynTensor::new(&x_data, &[5], &Device::metal()).unwrap();
    let y_gpu = x_gpu.sigmoid().unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_close(&y_gpu, &y_cpu, "sigmoid");
}

#[test]
fn test_gelu_gpu() {
    init();
    let x_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

    // CPU reference
    let x_cpu = DynTensor::new(&x_data, &[5], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.gelu().unwrap();

    // GPU
    let x_gpu = DynTensor::new(&x_data, &[5], &Device::metal()).unwrap();
    let y_gpu = x_gpu.gelu().unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_close(&y_gpu, &y_cpu, "gelu");
}

#[test]
fn test_tanh_gpu() {
    init();
    let x_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

    // CPU reference
    let x_cpu = DynTensor::new(&x_data, &[5], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.tanh().unwrap();

    // GPU
    let x_gpu = DynTensor::new(&x_data, &[5], &Device::metal()).unwrap();
    let y_gpu = x_gpu.tanh().unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_close(&y_gpu, &y_cpu, "tanh");
}

#[test]
fn test_sigmoid_2d_gpu() {
    init();
    // 2D tensor to test multi-dim dispatch
    let x_data: Vec<f32> = (-6..6).map(|i| i as f32 * 0.5).collect();

    // CPU reference
    let x_cpu = DynTensor::new(&x_data, &[3, 4], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.sigmoid().unwrap();

    // GPU
    let x_gpu = DynTensor::new(&x_data, &[3, 4], &Device::metal()).unwrap();
    let y_gpu = x_gpu.sigmoid().unwrap();

    assert_eq!(y_gpu.dims(), &[3, 4]);
    assert_close(&y_gpu, &y_cpu, "sigmoid_2d");
}

// -- #1057: Verify GPU roundtrip ops work on Metal tensors --------------------

#[test]
fn test_elu_gpu() {
    init();
    let x_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let alpha = 1.0;

    let x_cpu = DynTensor::new(&x_data, &[5], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.elu(alpha).unwrap();

    let x_gpu = DynTensor::new(&x_data, &[5], &Device::metal()).unwrap();
    let y_gpu = x_gpu.elu(alpha).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_close(&y_gpu, &y_cpu, "elu");
}

#[test]
fn test_clamp_gpu() {
    init();
    let x_data = vec![-3.0, -1.0, 0.5, 2.0, 5.0];

    let x_cpu = DynTensor::new(&x_data, &[5], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.clamp(-1.0, 3.0).unwrap();

    let x_gpu = DynTensor::new(&x_data, &[5], &Device::metal()).unwrap();
    let y_gpu = x_gpu.clamp(-1.0, 3.0).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_close(&y_gpu, &y_cpu, "clamp");
}

#[test]
fn test_leaky_relu_gpu() {
    init();
    let x_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

    let x_cpu = DynTensor::new(&x_data, &[5], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.leaky_relu(0.1).unwrap();

    let x_gpu = DynTensor::new(&x_data, &[5], &Device::metal()).unwrap();
    let y_gpu = x_gpu.leaky_relu(0.1).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_close(&y_gpu, &y_cpu, "leaky_relu");
}

#[test]
fn test_softmax_dim_gpu() {
    init();
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let x_cpu = DynTensor::new(&x_data, &[2, 3], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.softmax(1).unwrap();

    let x_gpu = DynTensor::new(&x_data, &[2, 3], &Device::metal()).unwrap();
    let y_gpu = x_gpu.softmax(1).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[2, 3]);
    assert_close(&y_gpu, &y_cpu, "softmax_dim");
}

#[test]
fn test_log_softmax_gpu() {
    init();
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let x_cpu = DynTensor::new(&x_data, &[2, 3], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.log_softmax(1).unwrap();

    let x_gpu = DynTensor::new(&x_data, &[2, 3], &Device::metal()).unwrap();
    let y_gpu = x_gpu.log_softmax(1).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_close(&y_gpu, &y_cpu, "log_softmax");
}

#[test]
fn test_clamp_min_gpu() {
    init();
    let x_data = vec![-3.0, -1.0, 0.0, 2.0, 5.0];

    let x_cpu = DynTensor::new(&x_data, &[5], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.clamp_min(0.0).unwrap();

    let x_gpu = DynTensor::new(&x_data, &[5], &Device::metal()).unwrap();
    let y_gpu = x_gpu.clamp_min(0.0).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_close(&y_gpu, &y_cpu, "clamp_min");
}

#[test]
fn test_clamp_max_gpu() {
    init();
    let x_data = vec![-3.0, -1.0, 0.0, 2.0, 5.0];

    let x_cpu = DynTensor::new(&x_data, &[5], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.clamp_max(2.0).unwrap();

    let x_gpu = DynTensor::new(&x_data, &[5], &Device::metal()).unwrap();
    let y_gpu = x_gpu.clamp_max(2.0).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_close(&y_gpu, &y_cpu, "clamp_max");
}

#[test]
fn test_powf_gpu() {
    init();
    let x_data = vec![1.0, 2.0, 3.0, 4.0];

    let x_cpu = DynTensor::new(&x_data, &[4], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.powf(2.0).unwrap();

    let x_gpu = DynTensor::new(&x_data, &[4], &Device::metal()).unwrap();
    let y_gpu = x_gpu.powf(2.0).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_close(&y_gpu, &y_cpu, "powf");
}

// -- #1023: Verify shape ops work on GPU tensors ---------------------------------

#[test]
fn test_transpose_gpu() {
    init();
    let x_data: Vec<f32> = (0..12).map(|i| i as f32).collect();

    let x_cpu = DynTensor::new(&x_data, &[3, 4], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.transpose(0, 1).unwrap();

    let x_gpu = DynTensor::new(&x_data, &[3, 4], &Device::metal()).unwrap();
    let y_gpu = x_gpu.transpose(0, 1).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[4, 3]);
    assert_close(&y_gpu, &y_cpu, "transpose");
}

#[test]
fn test_narrow_gpu() {
    init();
    let x_data: Vec<f32> = (0..20).map(|i| i as f32).collect();

    let x_cpu = DynTensor::new(&x_data, &[4, 5], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.narrow(1, 1, 3).unwrap();

    let x_gpu = DynTensor::new(&x_data, &[4, 5], &Device::metal()).unwrap();
    let y_gpu = x_gpu.narrow(1, 1, 3).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[4, 3]);
    assert_close(&y_gpu, &y_cpu, "narrow");
}

#[test]
fn test_cat_gpu() {
    init();
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0];

    let a_cpu = DynTensor::new(&a_data, &[2, 2], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[2, 3], &Device::Cpu).unwrap();
    let y_cpu = DynTensor::cat(&[&a_cpu, &b_cpu], 1).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[2, 2], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[2, 3], &Device::metal()).unwrap();
    let y_gpu = DynTensor::cat(&[&a_gpu, &b_gpu], 1).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[2, 5]);
    assert_close(&y_gpu, &y_cpu, "cat");
}

#[test]
fn test_permute_gpu() {
    init();
    let x_data: Vec<f32> = (0..24).map(|i| i as f32).collect();

    let x_cpu = DynTensor::new(&x_data, &[2, 3, 4], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.permute([2, 0, 1]).unwrap();

    let x_gpu = DynTensor::new(&x_data, &[2, 3, 4], &Device::metal()).unwrap();
    let y_gpu = x_gpu.permute([2, 0, 1]).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[4, 2, 3]);
    assert_close(&y_gpu, &y_cpu, "permute");
}

#[test]
fn test_stack_gpu() {
    init();
    let a_data = vec![1.0f32, 2.0, 3.0];
    let b_data = vec![4.0f32, 5.0, 6.0];

    let a_cpu = DynTensor::new(&a_data, &[3], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[3], &Device::Cpu).unwrap();
    let y_cpu = DynTensor::stack(&[&a_cpu, &b_cpu], 0).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[3], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[3], &Device::metal()).unwrap();
    let y_gpu = DynTensor::stack(&[&a_gpu, &b_gpu], 0).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[2, 3]);
    assert_close(&y_gpu, &y_cpu, "stack");
}

// -- #1030: Verify conv ops work on GPU tensors ----------------------------------

#[test]
fn test_conv1d_gpu() {
    init();
    // Input: [1, 2, 8], Kernel: [3, 2, 3], padding=1, stride=1, dilation=1, groups=1
    let x_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..18).map(|i| (i as f32) * 0.01).collect();

    let x_cpu = DynTensor::new(&x_data, &[1, 2, 8], &Device::Cpu).unwrap();
    let k_cpu = DynTensor::new(&k_data, &[3, 2, 3], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.conv1d(&k_cpu, 1, 1, 1, 1).unwrap();

    let x_gpu = DynTensor::new(&x_data, &[1, 2, 8], &Device::metal()).unwrap();
    let k_gpu = DynTensor::new(&k_data, &[3, 2, 3], &Device::metal()).unwrap();
    let y_gpu = x_gpu.conv1d(&k_gpu, 1, 1, 1, 1).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[1, 3, 8]);
    assert_close(&y_gpu, &y_cpu, "conv1d");
}

#[test]
fn test_conv_transpose1d_gpu() {
    init();
    // Input: [1, 2, 4], Kernel: [2, 3, 3], padding=1, output_padding=0, stride=2, dilation=1, groups=1
    let x_data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..18).map(|i| (i as f32) * 0.01).collect();

    let x_cpu = DynTensor::new(&x_data, &[1, 2, 4], &Device::Cpu).unwrap();
    let k_cpu = DynTensor::new(&k_data, &[2, 3, 3], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.conv_transpose1d(&k_cpu, 1, 0, 2, 1, 1).unwrap();

    let x_gpu = DynTensor::new(&x_data, &[1, 2, 4], &Device::metal()).unwrap();
    let k_gpu = DynTensor::new(&k_data, &[2, 3, 3], &Device::metal()).unwrap();
    let y_gpu = x_gpu.conv_transpose1d(&k_gpu, 1, 0, 2, 1, 1).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), y_cpu.dims());
    assert_close(&y_gpu, &y_cpu, "conv_transpose1d");
}

#[test]
fn test_pad1d_gpu() {
    init();
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    let x_cpu = DynTensor::new(&x_data, &[2, 3], &Device::Cpu).unwrap();
    let y_cpu = x_cpu.pad1d(2, 1).unwrap();

    let x_gpu = DynTensor::new(&x_data, &[2, 3], &Device::metal()).unwrap();
    let y_gpu = x_gpu.pad1d(2, 1).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[2, 6]);
    assert_close(&y_gpu, &y_cpu, "pad1d");
}

// GPU division semantics tests extracted to nn_gpu_forward_div.rs for 500-line limit.
