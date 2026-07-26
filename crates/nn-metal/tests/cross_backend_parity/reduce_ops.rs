// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Reduce operation parity tests: CPU vs Metal.
//!
//! Tests sum, mean, max, min reductions with keepdim on GPU,
//! including last-axis and non-last-axis (transpose path).

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

const TOL: f32 = 1e-4;

fn init() {
    gpu_init();
}

// -- Sum keepdim last axis -------------------------------------------------

#[test]
fn test_parity_sum_keepdim_last() {
    init();
    let data = rand_f32_vec(500, 120, -3.0, 3.0);

    let cpu = DynTensor::new(&data, &[10, 12], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[10, 12], &Device::metal()).unwrap();

    let cpu_out = cpu.sum_keepdim(1).unwrap();
    let gpu_out = gpu.sum_keepdim(1).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[10, 1]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "sum_keepdim_last");
}

// -- Sum keepdim first axis (transpose path) -------------------------------

#[test]
fn test_parity_sum_keepdim_first() {
    init();
    let data = rand_f32_vec(501, 120, -3.0, 3.0);

    let cpu = DynTensor::new(&data, &[10, 12], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[10, 12], &Device::metal()).unwrap();

    let cpu_out = cpu.sum_keepdim(0).unwrap();
    let gpu_out = gpu.sum_keepdim(0).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[1, 12]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "sum_keepdim_first");
}

// -- Mean keepdim ----------------------------------------------------------

#[test]
fn test_parity_mean_keepdim() {
    init();
    let data = rand_f32_vec(502, 2 * 8 * 16, -2.0, 2.0);

    let cpu = DynTensor::new(&data, &[2, 8, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[2, 8, 16], &Device::metal()).unwrap();

    // Mean over last dim
    let cpu_out = cpu.mean_keepdim(2).unwrap();
    let gpu_out = gpu.mean_keepdim(2).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[2, 8, 1]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "mean_keepdim");
}

// -- Max keepdim -----------------------------------------------------------

#[test]
fn test_parity_max_keepdim() {
    init();
    let data = rand_f32_vec(503, 120, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[10, 12], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[10, 12], &Device::metal()).unwrap();

    let cpu_out = cpu.max_keepdim(1).unwrap();
    let gpu_out = gpu.max_keepdim(1).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[10, 1]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "max_keepdim");
}

// -- Min keepdim -----------------------------------------------------------

#[test]
fn test_parity_min_keepdim() {
    init();
    let data = rand_f32_vec(504, 120, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[10, 12], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[10, 12], &Device::metal()).unwrap();

    let cpu_out = cpu.min_keepdim(1).unwrap();
    let gpu_out = gpu.min_keepdim(1).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[10, 1]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "min_keepdim");
}

// -- Sum all ---------------------------------------------------------------

#[test]
fn test_parity_sum_all() {
    init();
    let data = rand_f32_vec(505, 256, -2.0, 2.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.sum_all().unwrap();
    let gpu_out = gpu.sum_all().unwrap();

    assert_eq!(gpu_out.dims(), cpu_out.dims());
    // Wider tolerance for full reduction (accumulation error)
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 5e-3, "sum_all");
}

// -- Mean over middle axis (3D, transpose path) ----------------------------

#[test]
fn test_parity_mean_keepdim_middle() {
    init();
    let data = rand_f32_vec(506, 2 * 8 * 16, -2.0, 2.0);

    let cpu = DynTensor::new(&data, &[2, 8, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[2, 8, 16], &Device::metal()).unwrap();

    // Mean over middle dim (axis 1) — exercises transpose-reduce-transpose path
    let cpu_out = cpu.mean_keepdim(1).unwrap();
    let gpu_out = gpu.mean_keepdim(1).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[2, 1, 16]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "mean_keepdim_middle");
}
