// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compare and where_cond parity tests: CPU vs Metal.
//!
//! Tests scalar and tensor comparisons (eq, ne, ge, gt, lt, le) and
//! conditional select (where_cond) on both backends.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

const TOL: f32 = 1e-6;

fn init() {
    gpu_init();
}

// -- Scalar compare: gt ----------------------------------------------------

#[test]
fn test_parity_compare_gt() {
    init();
    let data = rand_f32_vec(600, 256, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.gt(0.0).unwrap();
    let gpu_out = gpu.gt(0.0).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "compare_gt");
}

// -- Scalar compare: le ----------------------------------------------------

#[test]
fn test_parity_compare_le() {
    init();
    let data = rand_f32_vec(601, 256, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.le(2.0).unwrap();
    let gpu_out = gpu.le(2.0).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "compare_le");
}

// -- Scalar compare: eq ----------------------------------------------------

#[test]
fn test_parity_compare_eq() {
    init();
    // Use integers cast to float so eq comparison is meaningful
    let data: Vec<f32> = (0..64).map(|i| (i % 4) as f32).collect();

    let cpu = DynTensor::new(&data, &[8, 8], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[8, 8], &Device::metal()).unwrap();

    let cpu_out = cpu.eq(2.0).unwrap();
    let gpu_out = gpu.eq(2.0).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "compare_eq");
}

// -- Tensor compare: broadcast_gt ------------------------------------------

#[test]
fn test_parity_compare_tensor_gt() {
    init();
    let a_data = rand_f32_vec(602, 256, -5.0, 5.0);
    let b_data = rand_f32_vec(603, 256, -5.0, 5.0);

    let a_cpu = DynTensor::new(&a_data, &[16, 16], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[16, 16], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.broadcast_gt(&b_cpu).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[16, 16], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[16, 16], &Device::metal()).unwrap();
    let gpu_out = a_gpu.broadcast_gt(&b_gpu).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "compare_tensor_gt");
}

// -- where_cond with f32 mask ---------------------------------------------

#[test]
fn test_parity_where_cond() {
    init();
    let mask_data = rand_f32_vec(604, 256, -3.0, 3.0);
    let a_data = rand_f32_vec(605, 256, -5.0, 5.0);
    let b_data = rand_f32_vec(606, 256, -5.0, 5.0);

    // Create mask via gt(0.0) on CPU
    let mask_cpu = DynTensor::new(&mask_data, &[16, 16], &Device::Cpu)
        .unwrap()
        .gt(0.0)
        .unwrap();
    let a_cpu = DynTensor::new(&a_data, &[16, 16], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[16, 16], &Device::Cpu).unwrap();
    let cpu_out = mask_cpu.where_cond(&a_cpu, &b_cpu).unwrap();

    // Same mask via gt(0.0) on GPU
    let mask_gpu = DynTensor::new(&mask_data, &[16, 16], &Device::metal())
        .unwrap()
        .gt(0.0)
        .unwrap();
    let a_gpu = DynTensor::new(&a_data, &[16, 16], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[16, 16], &Device::metal()).unwrap();
    let gpu_out = mask_gpu.where_cond(&a_gpu, &b_gpu).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "where_cond");
}
