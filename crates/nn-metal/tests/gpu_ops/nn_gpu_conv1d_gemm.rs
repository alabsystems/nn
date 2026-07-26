// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Differential tests for Conv1d im2col + simdgroup GEMM path (#3002).
//!
//! These tests use shapes large enough to trigger the GEMM routing
//! (`should_use_conv1d_gemm` threshold: M*K*N >= 2M FLOPs) and compare
//! GPU results against CPU reference output.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use super::test_utils::{assert_gpu_cpu_close, gpu_init, rand_f32_vec};

const GEMM_TOL: f32 = 1e-4;

/// Helper: run conv1d on both GPU and CPU, assert results match within tolerance.
fn assert_conv1d_gemm_parity(
    batch: usize,
    c_in: usize,
    c_out: usize,
    k_size: usize,
    l_in: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    tol: f32,
    label: &str,
) {
    let in_elems = batch * c_in * l_in;
    let k_elems = c_out * (c_in / groups) * k_size;
    let x_data = rand_f32_vec(42, in_elems, -1.0, 1.0);
    let k_data = rand_f32_vec(99, k_elems, -0.5, 0.5);

    let x_cpu = DynTensor::new(&x_data, &[batch, c_in, l_in], &Device::Cpu).unwrap();
    let k_cpu = DynTensor::new(&k_data, &[c_out, c_in / groups, k_size], &Device::Cpu).unwrap();
    let y_cpu = x_cpu
        .conv1d(&k_cpu, padding, stride, dilation, groups)
        .unwrap();

    let x_gpu = DynTensor::new(&x_data, &[batch, c_in, l_in], &Device::metal()).unwrap();
    let k_gpu = DynTensor::new(&k_data, &[c_out, c_in / groups, k_size], &Device::metal()).unwrap();
    let y_gpu = x_gpu
        .conv1d(&k_gpu, padding, stride, dilation, groups)
        .unwrap();

    assert_eq!(y_gpu.dims(), y_cpu.dims(), "{label}: shape mismatch");
    assert_gpu_cpu_close(&y_gpu, &y_cpu, tol, label);
}

/// Kokoro conv_pre shape: Conv1d(512, 512, 7, padding=3).
/// M=512, K=3584, N=120 → 220M FLOPs — well above GEMM threshold.
#[test]
fn test_conv1d_gemm_kokoro_conv_pre() {
    gpu_init();
    assert_conv1d_gemm_parity(
        1,
        512,
        512,
        7,
        126,
        1,
        3,
        1,
        1,
        GEMM_TOL,
        "kokoro_conv_pre_512x512x7",
    );
}

/// Kokoro ResBlock conv shape: Conv1d(256, 256, 3, padding=1).
/// M=256, K=768, N=120 → 23.6M FLOPs.
#[test]
fn test_conv1d_gemm_kokoro_resblock() {
    gpu_init();
    assert_conv1d_gemm_parity(
        1,
        256,
        256,
        3,
        120,
        1,
        1,
        1,
        1,
        GEMM_TOL,
        "kokoro_resblock_256x256x3",
    );
}

/// Moderate shape just above GEMM threshold (boundary test).
/// M=128, K=192, N=128 → 3.1M FLOPs (just above 2M threshold).
#[test]
fn test_conv1d_gemm_threshold_boundary() {
    gpu_init();
    assert_conv1d_gemm_parity(
        1,
        64,
        128,
        3,
        128,
        1,
        1,
        1,
        1,
        GEMM_TOL,
        "boundary_128x64x3",
    );
}

/// Kokoro TextEncoder conv shape: Conv1d(512, 512, 5, padding=2).
/// M=512, K=2560, N=128 → 167M FLOPs.
#[test]
fn test_conv1d_gemm_kokoro_text_encoder() {
    gpu_init();
    assert_conv1d_gemm_parity(
        1,
        512,
        512,
        5,
        128,
        1,
        2,
        1,
        1,
        GEMM_TOL,
        "kokoro_text_enc_512x512x5",
    );
}

/// Batched Conv1d — ensures per-batch im2col + GEMM loop works.
#[test]
fn test_conv1d_gemm_batched() {
    gpu_init();
    assert_conv1d_gemm_parity(
        4,
        128,
        128,
        3,
        256,
        1,
        1,
        1,
        1,
        GEMM_TOL,
        "batched_4x128x128x3",
    );
}

/// Conv1d with stride=2 (downsampling) — GEMM with non-unit stride.
#[test]
fn test_conv1d_gemm_stride2() {
    gpu_init();
    assert_conv1d_gemm_parity(
        1,
        256,
        256,
        3,
        240,
        2,
        1,
        1,
        1,
        GEMM_TOL,
        "stride2_256x256x3",
    );
}

/// Conv1d with dilation=2 — dilated GEMM path.
#[test]
fn test_conv1d_gemm_dilated() {
    gpu_init();
    assert_conv1d_gemm_parity(
        1,
        256,
        256,
        3,
        240,
        1,
        2,
        2,
        1,
        GEMM_TOL,
        "dilated2_256x256x3",
    );
}
