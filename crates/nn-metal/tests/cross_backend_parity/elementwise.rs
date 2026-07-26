// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Elementwise operation parity tests: CPU vs Metal.
//!
//! Each test runs the same operation on both CPU and Metal backends,
//! then asserts element-wise equality within tolerance.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

const TOL: f32 = 1e-5;

fn init() {
    gpu_init();
}

// -- Relu ------------------------------------------------------------------

#[test]
fn test_parity_relu() {
    init();
    let data = rand_f32_vec(42, 256, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.relu().unwrap();
    let gpu_out = gpu.relu().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "relu");
}

// -- Gelu ------------------------------------------------------------------

#[test]
fn test_parity_gelu() {
    init();
    let data = rand_f32_vec(43, 256, -3.0, 3.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.gelu().unwrap();
    let gpu_out = gpu.gelu().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 1e-4, "gelu");
}

// -- Silu ------------------------------------------------------------------

#[test]
fn test_parity_silu() {
    init();
    let data = rand_f32_vec(44, 256, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.silu().unwrap();
    let gpu_out = gpu.silu().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "silu");
}

// -- Sigmoid ---------------------------------------------------------------

#[test]
fn test_parity_sigmoid() {
    init();
    let data = rand_f32_vec(45, 256, -6.0, 6.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.sigmoid().unwrap();
    let gpu_out = gpu.sigmoid().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "sigmoid");
}

// -- Tanh ------------------------------------------------------------------

#[test]
fn test_parity_tanh() {
    init();
    let data = rand_f32_vec(46, 256, -4.0, 4.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.tanh().unwrap();
    let gpu_out = gpu.tanh().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "tanh");
}

// -- Softmax ---------------------------------------------------------------

#[test]
fn test_parity_softmax() {
    init();
    let data = rand_f32_vec(47, 120, -3.0, 3.0);

    let cpu = DynTensor::new(&data, &[10, 12], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[10, 12], &Device::metal()).unwrap();

    // Softmax over last dim (dim 1)
    let cpu_out = cpu.softmax(1).unwrap();
    let gpu_out = gpu.softmax(1).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 1e-4, "softmax");
}

// -- Exp -------------------------------------------------------------------

#[test]
fn test_parity_exp() {
    init();
    // Keep range small to avoid huge values
    let data = rand_f32_vec(48, 256, -3.0, 3.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.exp().unwrap();
    let gpu_out = gpu.exp().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 1e-4, "exp");
}

// -- Log -------------------------------------------------------------------

#[test]
fn test_parity_log() {
    init();
    // Positive values only for log
    let data = rand_f32_vec(49, 256, 0.01, 10.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.log().unwrap();
    let gpu_out = gpu.log().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "log");
}

// -- Batched elementwise: 3D input -----------------------------------------

#[test]
fn test_parity_relu_3d() {
    init();
    let data = rand_f32_vec(50, 480, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[2, 10, 24], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[2, 10, 24], &Device::metal()).unwrap();

    let cpu_out = cpu.relu().unwrap();
    let gpu_out = gpu.relu().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[2, 10, 24]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "relu_3d");
}

// -- Log softmax -----------------------------------------------------------

#[test]
fn test_parity_log_softmax() {
    init();
    let data = rand_f32_vec(51, 120, -3.0, 3.0);

    let cpu = DynTensor::new(&data, &[10, 12], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[10, 12], &Device::metal()).unwrap();

    let cpu_out = cpu.log_softmax(1).unwrap();
    let gpu_out = gpu.log_softmax(1).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 1e-4, "log_softmax");
}
