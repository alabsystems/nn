// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU scalar op tests for BF16/F16 dtypes (#3230 Gap 2).
//!
//! Verifies that `add_scalar`, `mul_scalar`, `div_scalar`, `affine`,
//! `clamp`, `clamp_min`, `clamp_max` dispatch to GPU for BF16/F16
//! tensors instead of falling back to scalar_like + CPU allocation.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

// BF16/F16 accumulate in float but read/write in half — wider tolerance.
const TOL: f32 = 2e-3;

fn init() {
    gpu_init();
}

fn assert_close(gpu: &DynTensor, cpu: &DynTensor, label: &str) {
    assert_gpu_cpu_close(gpu, cpu, TOL, label);
}

/// Create a GPU tensor in the given dtype from f32 data.
fn gpu_tensor(data: &[f32], shape: &[usize], dtype: DType) -> DynTensor {
    let t = DynTensor::new(data, shape, &Device::metal()).unwrap();
    t.to_dtype(dtype).unwrap()
}

/// Create a CPU tensor in the given dtype from f32 data.
fn cpu_tensor(data: &[f32], shape: &[usize], dtype: DType) -> DynTensor {
    let t = DynTensor::new(data, shape, &Device::Cpu).unwrap();
    t.to_dtype(dtype).unwrap()
}

// -- add_scalar BF16 ---------------------------------------------------------

#[test]
fn test_add_scalar_bf16() {
    init();
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];

    let y_cpu = cpu_tensor(&data, &[2, 3], DType::BF16)
        .add_scalar(1.5)
        .unwrap();
    let y_gpu = gpu_tensor(&data, &[2, 3], DType::BF16)
        .add_scalar(1.5)
        .unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "add_scalar_bf16");
}

// -- mul_scalar BF16 ---------------------------------------------------------

#[test]
fn test_mul_scalar_bf16() {
    init();
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];

    let y_cpu = cpu_tensor(&data, &[6], DType::BF16)
        .mul_scalar(0.5)
        .unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::BF16)
        .mul_scalar(0.5)
        .unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "mul_scalar_bf16");
}

// -- div_scalar BF16 ---------------------------------------------------------

#[test]
fn test_div_scalar_bf16() {
    init();
    let data = vec![2.0, 4.0, 6.0, 8.0];

    let y_cpu = cpu_tensor(&data, &[4], DType::BF16)
        .div_scalar(2.0)
        .unwrap();
    let y_gpu = gpu_tensor(&data, &[4], DType::BF16)
        .div_scalar(2.0)
        .unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "div_scalar_bf16");
}

// -- affine BF16 -------------------------------------------------------------

#[test]
fn test_affine_bf16() {
    init();
    let data = vec![1.0, 2.0, 3.0, 4.0];

    let y_cpu = cpu_tensor(&data, &[4], DType::BF16)
        .affine(2.0, -1.0)
        .unwrap();
    let y_gpu = gpu_tensor(&data, &[4], DType::BF16)
        .affine(2.0, -1.0)
        .unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "affine_bf16");
}

// -- clamp BF16 --------------------------------------------------------------

#[test]
fn test_clamp_bf16() {
    init();
    let data = vec![-3.0, -1.0, 0.5, 2.0, 5.0];

    let y_cpu = cpu_tensor(&data, &[5], DType::BF16)
        .clamp(-1.0, 3.0)
        .unwrap();
    let y_gpu = gpu_tensor(&data, &[5], DType::BF16)
        .clamp(-1.0, 3.0)
        .unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "clamp_bf16");
}

// -- clamp_min BF16 ----------------------------------------------------------

#[test]
fn test_clamp_min_bf16() {
    init();
    let data = vec![-3.0, -1.0, 0.0, 2.0, 5.0];

    let y_cpu = cpu_tensor(&data, &[5], DType::BF16).clamp_min(0.0).unwrap();
    let y_gpu = gpu_tensor(&data, &[5], DType::BF16).clamp_min(0.0).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "clamp_min_bf16");
}

// -- clamp_max BF16 ----------------------------------------------------------

#[test]
fn test_clamp_max_bf16() {
    init();
    let data = vec![-3.0, -1.0, 0.0, 2.0, 5.0];

    let y_cpu = cpu_tensor(&data, &[5], DType::BF16).clamp_max(1.0).unwrap();
    let y_gpu = gpu_tensor(&data, &[5], DType::BF16).clamp_max(1.0).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "clamp_max_bf16");
}

// -- F16 scalar ops ----------------------------------------------------------

#[test]
fn test_add_scalar_f16() {
    init();
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];

    let y_cpu = cpu_tensor(&data, &[2, 3], DType::F16)
        .add_scalar(1.5)
        .unwrap();
    let y_gpu = gpu_tensor(&data, &[2, 3], DType::F16)
        .add_scalar(1.5)
        .unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "add_scalar_f16");
}

#[test]
fn test_mul_scalar_f16() {
    init();
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];

    let y_cpu = cpu_tensor(&data, &[6], DType::F16).mul_scalar(0.5).unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::F16).mul_scalar(0.5).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "mul_scalar_f16");
}

#[test]
fn test_div_scalar_f16() {
    init();
    let data = vec![2.0, 4.0, 6.0, 8.0];

    let y_cpu = cpu_tensor(&data, &[4], DType::F16).div_scalar(2.0).unwrap();
    let y_gpu = gpu_tensor(&data, &[4], DType::F16).div_scalar(2.0).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "div_scalar_f16");
}

#[test]
fn test_affine_f16() {
    init();
    let data = vec![1.0, 2.0, 3.0, 4.0];

    let y_cpu = cpu_tensor(&data, &[4], DType::F16)
        .affine(2.0, -1.0)
        .unwrap();
    let y_gpu = gpu_tensor(&data, &[4], DType::F16)
        .affine(2.0, -1.0)
        .unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "affine_f16");
}

#[test]
fn test_clamp_f16() {
    init();
    let data = vec![-3.0, -1.0, 0.5, 2.0, 5.0];

    let y_cpu = cpu_tensor(&data, &[5], DType::F16)
        .clamp(-1.0, 3.0)
        .unwrap();
    let y_gpu = gpu_tensor(&data, &[5], DType::F16)
        .clamp(-1.0, 3.0)
        .unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "clamp_f16");
}

#[test]
fn test_clamp_min_f16() {
    init();
    let data = vec![-3.0, -1.0, 0.0, 2.0, 5.0];

    let y_cpu = cpu_tensor(&data, &[5], DType::F16).clamp_min(0.0).unwrap();
    let y_gpu = gpu_tensor(&data, &[5], DType::F16).clamp_min(0.0).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "clamp_min_f16");
}

#[test]
fn test_clamp_max_f16() {
    init();
    let data = vec![-3.0, -1.0, 0.0, 2.0, 5.0];

    let y_cpu = cpu_tensor(&data, &[5], DType::F16).clamp_max(1.0).unwrap();
    let y_gpu = gpu_tensor(&data, &[5], DType::F16).clamp_max(1.0).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "clamp_max_f16");
}
