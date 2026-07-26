// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Linear algebra parity tests: CPU vs Metal.
//!
//! Tests matmul, linear layer, batched matmul, transpose, and concatenation.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

const TOL: f32 = 1e-4;

fn init() {
    gpu_init();
}

// -- MatMul ----------------------------------------------------------------

#[test]
fn test_parity_matmul() {
    init();
    let a_data = rand_f32_vec(60, 32, -1.0, 1.0);
    let b_data = rand_f32_vec(61, 32, -1.0, 1.0);

    // A: [4, 8], B: [8, 4] -> C: [4, 4]
    let a_cpu = DynTensor::new(&a_data, &[4, 8], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[8, 4], &Device::Cpu).unwrap();
    let c_cpu = a_cpu.matmul(&b_cpu).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[4, 8], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[8, 4], &Device::metal()).unwrap();
    let c_gpu = a_gpu.matmul(&b_gpu).unwrap();

    assert_eq!(c_gpu.device(), Device::metal());
    assert_eq!(c_gpu.dims(), &[4, 4]);
    assert_eq!(c_gpu.dims(), c_cpu.dims());
    assert_gpu_cpu_close(&c_gpu, &c_cpu, TOL, "matmul");
}

// -- MatMul larger ----------------------------------------------------------

#[test]
fn test_parity_matmul_larger() {
    init();
    let a_data = rand_f32_vec(62, 128 * 64, -1.0, 1.0);
    let b_data = rand_f32_vec(63, 64 * 32, -1.0, 1.0);

    // A: [128, 64], B: [64, 32] -> C: [128, 32]
    let a_cpu = DynTensor::new(&a_data, &[128, 64], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[64, 32], &Device::Cpu).unwrap();
    let c_cpu = a_cpu.matmul(&b_cpu).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[128, 64], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[64, 32], &Device::metal()).unwrap();
    let c_gpu = a_gpu.matmul(&b_gpu).unwrap();

    assert_eq!(c_gpu.dims(), &[128, 32]);
    assert_gpu_cpu_close(&c_gpu, &c_cpu, TOL, "matmul_larger");
}

// -- Linear layer (no bias) ------------------------------------------------

#[test]
fn test_parity_linear_no_bias() {
    init();
    let w_data = rand_f32_vec(64, 32 * 16, -0.5, 0.5);
    let x_data = rand_f32_vec(65, 4 * 16, -1.0, 1.0);

    // Weight: [32, 16], Input: [4, 16] -> Output: [4, 32]
    let w_cpu = DynTensor::new(&w_data, &[32, 16], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[4, 16], &Device::Cpu).unwrap();
    let linear_cpu = Linear::new(w_cpu, None).unwrap();
    let y_cpu = linear_cpu.forward(&x_cpu).unwrap();

    let w_gpu = DynTensor::new(&w_data, &[32, 16], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[4, 16], &Device::metal()).unwrap();
    let linear_gpu = Linear::new(w_gpu, None).unwrap();
    let y_gpu = linear_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[4, 32]);
    assert_gpu_cpu_close(&y_gpu, &y_cpu, TOL, "linear_no_bias");
}

// -- Linear layer (with bias) ----------------------------------------------

#[test]
fn test_parity_linear_with_bias() {
    init();
    let w_data = rand_f32_vec(66, 32 * 16, -0.5, 0.5);
    let b_data = rand_f32_vec(67, 32, -0.1, 0.1);
    let x_data = rand_f32_vec(68, 4 * 16, -1.0, 1.0);

    let w_cpu = DynTensor::new(&w_data, &[32, 16], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[32], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[4, 16], &Device::Cpu).unwrap();
    let linear_cpu = Linear::new(w_cpu, Some(b_cpu)).unwrap();
    let y_cpu = linear_cpu.forward(&x_cpu).unwrap();

    let w_gpu = DynTensor::new(&w_data, &[32, 16], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[32], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[4, 16], &Device::metal()).unwrap();
    let linear_gpu = Linear::new(w_gpu, Some(b_gpu)).unwrap();
    let y_gpu = linear_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[4, 32]);
    assert_gpu_cpu_close(&y_gpu, &y_cpu, TOL, "linear_with_bias");
}

// -- Batched matmul (3D) ---------------------------------------------------

#[test]
fn test_parity_bmm() {
    init();
    // Batched: [2, 4, 8] @ [2, 8, 6] -> [2, 4, 6]
    let a_data = rand_f32_vec(69, 2 * 4 * 8, -1.0, 1.0);
    let b_data = rand_f32_vec(70, 2 * 8 * 6, -1.0, 1.0);

    let a_cpu = DynTensor::new(&a_data, &[2, 4, 8], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[2, 8, 6], &Device::Cpu).unwrap();
    let c_cpu = a_cpu.matmul(&b_cpu).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[2, 4, 8], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[2, 8, 6], &Device::metal()).unwrap();
    let c_gpu = a_gpu.matmul(&b_gpu).unwrap();

    assert_eq!(c_gpu.device(), Device::metal());
    assert_eq!(c_gpu.dims(), &[2, 4, 6]);
    assert_eq!(c_gpu.dims(), c_cpu.dims());
    assert_gpu_cpu_close(&c_gpu, &c_cpu, TOL, "bmm");
}

// -- Transpose -------------------------------------------------------------

#[test]
fn test_parity_transpose() {
    init();
    let data = rand_f32_vec(71, 24, -2.0, 2.0);

    let cpu = DynTensor::new(&data, &[4, 6], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[4, 6], &Device::metal()).unwrap();

    let cpu_out = cpu.t().unwrap();
    let gpu_out = gpu.t().unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[6, 4]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 1e-6, "transpose");
}

// -- Cat -------------------------------------------------------------------

#[test]
fn test_parity_cat() {
    init();
    let a_data = rand_f32_vec(72, 24, -1.0, 1.0);
    let b_data = rand_f32_vec(73, 24, -1.0, 1.0);

    // Cat along dim 0: [3, 8] + [3, 8] -> [6, 8]
    let a_cpu = DynTensor::new(&a_data, &[3, 8], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[3, 8], &Device::Cpu).unwrap();
    let cpu_out = DynTensor::cat(&[&a_cpu, &b_cpu], 0).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[3, 8], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[3, 8], &Device::metal()).unwrap();
    let gpu_out = DynTensor::cat(&[&a_gpu, &b_gpu], 0).unwrap();

    assert_eq!(gpu_out.dims(), &[6, 8]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 1e-6, "cat_dim0");
}

// -- Cat along dim 1 -------------------------------------------------------

#[test]
fn test_parity_cat_dim1() {
    init();
    let a_data = rand_f32_vec(74, 12, -1.0, 1.0);
    let b_data = rand_f32_vec(75, 18, -1.0, 1.0);

    // Cat along dim 1: [3, 4] + [3, 6] -> [3, 10]
    let a_cpu = DynTensor::new(&a_data, &[3, 4], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[3, 6], &Device::Cpu).unwrap();
    let cpu_out = DynTensor::cat(&[&a_cpu, &b_cpu], 1).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[3, 4], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[3, 6], &Device::metal()).unwrap();
    let gpu_out = DynTensor::cat(&[&a_gpu, &b_gpu], 1).unwrap();

    assert_eq!(gpu_out.dims(), &[3, 10]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 1e-6, "cat_dim1");
}
