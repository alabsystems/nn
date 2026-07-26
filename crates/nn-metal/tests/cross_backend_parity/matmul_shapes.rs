// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended matmul shape parity tests: CPU vs Metal.
//!
//! Tests various M,K,N combinations including non-aligned shapes,
//! tall-skinny, wide-short, square, and large matrices to exercise
//! both naive and simdgroup GEMM paths.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

const TOL: f32 = 1e-3;

fn init() {
    gpu_init();
}

fn run_matmul_parity(seed: u64, m: usize, k: usize, n: usize, label: &str) {
    let a_data = rand_f32_vec(seed, m * k, -1.0, 1.0);
    let b_data = rand_f32_vec(seed + 1, k * n, -1.0, 1.0);

    let a_cpu = DynTensor::new(&a_data, &[m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[k, n], &Device::Cpu).unwrap();
    let c_cpu = a_cpu.matmul(&b_cpu).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[m, k], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[k, n], &Device::metal()).unwrap();
    let c_gpu = a_gpu.matmul(&b_gpu).unwrap();

    assert_eq!(c_gpu.dims(), &[m, n]);
    assert_eq!(c_gpu.dims(), c_cpu.dims());
    assert_gpu_cpu_close(&c_gpu, &c_cpu, TOL, label);
}

// -- Small square ----------------------------------------------------------

#[test]
fn test_parity_matmul_small_square() {
    init();
    run_matmul_parity(1000, 8, 8, 8, "matmul_8x8x8");
}

// -- Tall skinny -----------------------------------------------------------

#[test]
fn test_parity_matmul_tall_skinny() {
    init();
    run_matmul_parity(1001, 256, 4, 4, "matmul_256x4x4");
}

// -- Wide short ------------------------------------------------------------

#[test]
fn test_parity_matmul_wide_short() {
    init();
    run_matmul_parity(1002, 4, 4, 256, "matmul_4x4x256");
}

// -- Non-aligned (odd dimensions) ------------------------------------------

#[test]
fn test_parity_matmul_non_aligned() {
    init();
    run_matmul_parity(1003, 13, 17, 11, "matmul_13x17x11");
}

// -- Large (triggers simdgroup path) ---------------------------------------

#[test]
fn test_parity_matmul_large() {
    init();
    // 256x256x256: aligned, large enough for simdgroup
    run_matmul_parity(1004, 256, 256, 256, "matmul_256x256x256");
}

// -- Batch matmul 4D (multi-head attention shape) --------------------------

#[test]
fn test_parity_bmm_4d() {
    init();
    // [batch, heads, seq, d_head] @ [batch, heads, d_head, seq]
    let batch = 2;
    let heads = 4;
    let seq = 16;
    let d_head = 8;

    let a_data = rand_f32_vec(1005, batch * heads * seq * d_head, -1.0, 1.0);
    let b_data = rand_f32_vec(1006, batch * heads * d_head * seq, -1.0, 1.0);

    let a_cpu = DynTensor::new(&a_data, &[batch, heads, seq, d_head], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[batch, heads, d_head, seq], &Device::Cpu).unwrap();
    let c_cpu = a_cpu.matmul(&b_cpu).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[batch, heads, seq, d_head], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[batch, heads, d_head, seq], &Device::metal()).unwrap();
    let c_gpu = a_gpu.matmul(&b_gpu).unwrap();

    assert_eq!(c_gpu.dims(), &[batch, heads, seq, seq]);
    assert_eq!(c_gpu.dims(), c_cpu.dims());
    assert_gpu_cpu_close(&c_gpu, &c_cpu, TOL, "bmm_4d");
}

// -- Rectangular large (K >> M, N) -----------------------------------------

#[test]
fn test_parity_matmul_large_k() {
    init();
    run_matmul_parity(1007, 32, 512, 32, "matmul_32x512x32");
}
