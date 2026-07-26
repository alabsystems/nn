// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Clamp operation parity tests: CPU vs Metal.
//!
//! Tests clamp, clamp_min (relu-like), and clamp_max on both backends.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

const TOL: f32 = 1e-6;

fn init() {
    gpu_init();
}

// -- Clamp (both bounds) ---------------------------------------------------

#[test]
fn test_parity_clamp() {
    init();
    let data = rand_f32_vec(700, 256, -10.0, 10.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.clamp(-2.0, 3.0).unwrap();
    let gpu_out = gpu.clamp(-2.0, 3.0).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "clamp");
}

// -- Clamp with tight bounds -----------------------------------------------

#[test]
fn test_parity_clamp_tight() {
    init();
    let data = rand_f32_vec(701, 256, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    // Tight clamp: most values should be clamped
    let cpu_out = cpu.clamp(-0.1, 0.1).unwrap();
    let gpu_out = gpu.clamp(-0.1, 0.1).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "clamp_tight");
}

// -- Clamp 3D --------------------------------------------------------------

#[test]
fn test_parity_clamp_3d() {
    init();
    let data = rand_f32_vec(702, 2 * 8 * 16, -10.0, 10.0);

    let cpu = DynTensor::new(&data, &[2, 8, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[2, 8, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.clamp(0.0, 6.0).unwrap();
    let gpu_out = gpu.clamp(0.0, 6.0).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[2, 8, 16]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "clamp_3d");
}
