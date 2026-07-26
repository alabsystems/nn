// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Binary operation parity tests: CPU vs Metal.
//!
//! Tests add, sub, mul, div, maximum, minimum on both backends,
//! including broadcast variants and scalar binary ops.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

const TOL: f32 = 1e-5;

fn init() {
    gpu_init();
}

// -- Add -------------------------------------------------------------------

#[test]
fn test_parity_add() {
    init();
    let a_data = rand_f32_vec(300, 256, -5.0, 5.0);
    let b_data = rand_f32_vec(301, 256, -5.0, 5.0);

    let a_cpu = DynTensor::new(&a_data, &[16, 16], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[16, 16], &Device::Cpu).unwrap();
    let cpu_out = (&a_cpu + &b_cpu).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[16, 16], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[16, 16], &Device::metal()).unwrap();
    let gpu_out = (&a_gpu + &b_gpu).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "add");
}

// -- Sub -------------------------------------------------------------------

#[test]
fn test_parity_sub() {
    init();
    let a_data = rand_f32_vec(302, 256, -5.0, 5.0);
    let b_data = rand_f32_vec(303, 256, -5.0, 5.0);

    let a_cpu = DynTensor::new(&a_data, &[16, 16], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[16, 16], &Device::Cpu).unwrap();
    let cpu_out = (&a_cpu - &b_cpu).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[16, 16], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[16, 16], &Device::metal()).unwrap();
    let gpu_out = (&a_gpu - &b_gpu).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "sub");
}

// -- Mul -------------------------------------------------------------------

#[test]
fn test_parity_mul() {
    init();
    let a_data = rand_f32_vec(304, 256, -3.0, 3.0);
    let b_data = rand_f32_vec(305, 256, -3.0, 3.0);

    let a_cpu = DynTensor::new(&a_data, &[16, 16], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[16, 16], &Device::Cpu).unwrap();
    let cpu_out = (&a_cpu * &b_cpu).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[16, 16], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[16, 16], &Device::metal()).unwrap();
    let gpu_out = (&a_gpu * &b_gpu).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "mul");
}

// -- Div -------------------------------------------------------------------

#[test]
fn test_parity_div() {
    init();
    let a_data = rand_f32_vec(306, 256, -5.0, 5.0);
    // Avoid values near zero for division
    let b_data = rand_f32_vec(307, 256, 0.5, 3.0);

    let a_cpu = DynTensor::new(&a_data, &[16, 16], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[16, 16], &Device::Cpu).unwrap();
    let cpu_out = (&a_cpu / &b_cpu).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[16, 16], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[16, 16], &Device::metal()).unwrap();
    let gpu_out = (&a_gpu / &b_gpu).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "div");
}

// -- Maximum ---------------------------------------------------------------

#[test]
fn test_parity_maximum() {
    init();
    let a_data = rand_f32_vec(308, 256, -5.0, 5.0);
    let b_data = rand_f32_vec(309, 256, -5.0, 5.0);

    let a_cpu = DynTensor::new(&a_data, &[16, 16], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[16, 16], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.maximum(&b_cpu).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[16, 16], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[16, 16], &Device::metal()).unwrap();
    let gpu_out = a_gpu.maximum(&b_gpu).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "maximum");
}

// -- Minimum ---------------------------------------------------------------

#[test]
fn test_parity_minimum() {
    init();
    let a_data = rand_f32_vec(310, 256, -5.0, 5.0);
    let b_data = rand_f32_vec(311, 256, -5.0, 5.0);

    let a_cpu = DynTensor::new(&a_data, &[16, 16], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[16, 16], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.minimum(&b_cpu).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[16, 16], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[16, 16], &Device::metal()).unwrap();
    let gpu_out = a_gpu.minimum(&b_gpu).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "minimum");
}

// -- Broadcast add (3D + 1D) -----------------------------------------------

#[test]
fn test_parity_broadcast_add() {
    init();
    let a_data = rand_f32_vec(312, 2 * 4 * 8, -2.0, 2.0);
    let b_data = rand_f32_vec(313, 8, -1.0, 1.0);

    let a_cpu = DynTensor::new(&a_data, &[2, 4, 8], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[8], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.broadcast_add(&b_cpu).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[2, 4, 8], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[8], &Device::metal()).unwrap();
    let gpu_out = a_gpu.broadcast_add(&b_gpu).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[2, 4, 8]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "broadcast_add");
}

// -- Scalar add/mul --------------------------------------------------------

#[test]
fn test_parity_scalar_add() {
    init();
    let data = rand_f32_vec(314, 256, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let cpu_out = (&cpu + 3.14f64).unwrap();

    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();
    let gpu_out = (&gpu + 3.14f64).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "scalar_add");
}

#[test]
fn test_parity_scalar_mul() {
    init();
    let data = rand_f32_vec(315, 256, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let cpu_out = (&cpu * 2.5f64).unwrap();

    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();
    let gpu_out = (&gpu * 2.5f64).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "scalar_mul");
}
