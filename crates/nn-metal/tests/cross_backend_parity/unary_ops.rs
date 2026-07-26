// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended unary operation parity tests: CPU vs Metal.
//!
//! Tests unary ops not covered by elementwise.rs: sqrt, abs, recip, sin, cos,
//! neg, sqr, floor, round, fract, gelu_erf.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

const TOL: f32 = 1e-5;

fn init() {
    gpu_init();
}

// -- Sqrt ------------------------------------------------------------------

#[test]
fn test_parity_sqrt() {
    init();
    // Positive values only for sqrt
    let data = rand_f32_vec(400, 256, 0.01, 10.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.sqrt().unwrap();
    let gpu_out = gpu.sqrt().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "sqrt");
}

// -- Abs -------------------------------------------------------------------

#[test]
fn test_parity_abs() {
    init();
    let data = rand_f32_vec(401, 256, -10.0, 10.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.abs().unwrap();
    let gpu_out = gpu.abs().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "abs");
}

// -- Recip -----------------------------------------------------------------

#[test]
fn test_parity_recip() {
    init();
    // Avoid zero for recip
    let data = rand_f32_vec(402, 256, 0.5, 5.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.recip().unwrap();
    let gpu_out = gpu.recip().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "recip");
}

// -- Sin -------------------------------------------------------------------

#[test]
fn test_parity_sin() {
    init();
    let data = rand_f32_vec(403, 256, -6.28, 6.28);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.sin().unwrap();
    let gpu_out = gpu.sin().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "sin");
}

// -- Cos -------------------------------------------------------------------

#[test]
fn test_parity_cos() {
    init();
    let data = rand_f32_vec(404, 256, -6.28, 6.28);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.cos().unwrap();
    let gpu_out = gpu.cos().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "cos");
}

// -- Neg -------------------------------------------------------------------

#[test]
fn test_parity_neg() {
    init();
    let data = rand_f32_vec(405, 256, -10.0, 10.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.neg().unwrap();
    let gpu_out = gpu.neg().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "neg");
}

// -- Sqr -------------------------------------------------------------------

#[test]
fn test_parity_sqr() {
    init();
    let data = rand_f32_vec(406, 256, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.sqr().unwrap();
    let gpu_out = gpu.sqr().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "sqr");
}

// -- Floor -----------------------------------------------------------------

#[test]
fn test_parity_floor() {
    init();
    let data = rand_f32_vec(407, 256, -10.0, 10.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.floor().unwrap();
    let gpu_out = gpu.floor().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "floor");
}

// -- Round -----------------------------------------------------------------

#[test]
fn test_parity_round() {
    init();
    let data = rand_f32_vec(408, 256, -10.0, 10.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.round().unwrap();
    let gpu_out = gpu.round().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "round");
}

// -- Fract -----------------------------------------------------------------

#[test]
fn test_parity_fract() {
    init();
    let data = rand_f32_vec(409, 256, -10.0, 10.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.fract().unwrap();
    let gpu_out = gpu.fract().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "fract");
}

// -- Gelu Erf --------------------------------------------------------------

#[test]
fn test_parity_gelu_erf() {
    init();
    let data = rand_f32_vec(410, 256, -3.0, 3.0);

    let cpu = DynTensor::new(&data, &[16, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[16, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.gelu_erf().unwrap();
    let gpu_out = gpu.gelu_erf().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 1e-4, "gelu_erf");
}
