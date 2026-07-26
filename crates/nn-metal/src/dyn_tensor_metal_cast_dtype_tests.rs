// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for GPU-native dtype conversion kernels.
//!
//! Validates F32↔F16/BF16 conversion stays on GPU (no CPU round-trip)
//! and produces values matching CPU conversion within half-precision tolerance.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use crate::test_common::{assert_close, init};

/// Create a deterministic F32 GPU tensor.
fn gpu_f32(data: &[f32], shape: &[usize]) -> DynTensor {
    DynTensor::new(data, shape, &Device::Cpu)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap()
}

/// Round-trip: read GPU tensor back as f32 values.
fn read_f32(t: &DynTensor) -> Vec<f32> {
    let cpu = t.to_device(&Device::Cpu).unwrap();
    cpu.to_f32_array().unwrap().iter().copied().collect()
}

// -- F32 → F16 ---------------------------------------------------------------

#[test]
fn test_gpu_cast_f32_to_f16() {
    init();
    let data = vec![1.0f32, -2.5, 0.0, 3.14, 65504.0, -65504.0];
    let gpu = gpu_f32(&data, &[2, 3]);

    let f16 = gpu.to_dtype(DType::F16).unwrap();
    assert_eq!(f16.dtype(), DType::F16);
    assert!(f16.device().is_gpu(), "result should stay on GPU");
    assert_eq!(f16.dims(), &[2, 3]);

    // Read back as f32 and compare — half precision has ~3 decimal digits.
    let vals = read_f32(&f16);
    assert_close(&vals, &data, 1e-2, "f32_to_f16");
}

#[test]
fn test_gpu_cast_f32_to_f16_roundtrip() {
    init();
    let data = vec![0.1, -0.5, 1.0, 100.0, 0.001, -1000.0];
    let gpu = gpu_f32(&data, &[6]);

    // F32 → F16 → F32 round-trip
    let f16 = gpu.to_dtype(DType::F16).unwrap();
    let back = f16.to_dtype(DType::F32).unwrap();
    assert_eq!(back.dtype(), DType::F32);
    assert!(back.device().is_gpu());

    let vals = read_f32(&back);
    // Half precision: values should match within ~0.1% for normal range.
    assert_close(&vals, &data, 1e-1, "f32_f16_f32_roundtrip");
}

// -- F32 → BF16 --------------------------------------------------------------

#[test]
fn test_gpu_cast_f32_to_bf16() {
    init();
    let data = vec![1.0f32, -2.0, 0.5, 128.0];
    let gpu = gpu_f32(&data, &[4]);

    let bf16 = gpu.to_dtype(DType::BF16).unwrap();
    assert_eq!(bf16.dtype(), DType::BF16);
    assert!(bf16.device().is_gpu());

    let vals = read_f32(&bf16);
    // BF16 has same exponent range as F32 but only ~3 sig digits.
    assert_close(&vals, &data, 1e-1, "f32_to_bf16");
}

// -- F16 → F32 ---------------------------------------------------------------

#[test]
fn test_gpu_cast_f16_to_f32() {
    init();
    // Create F16 GPU tensor via CPU path first.
    let data = vec![1.0f32, -3.0, 0.0, 42.0];
    let cpu_f32 = DynTensor::new(&data, &[4], &Device::Cpu).unwrap();
    let cpu_f16 = cpu_f32.to_dtype(DType::F16).unwrap();
    let gpu_f16 = cpu_f16.to_device(&Device::metal()).unwrap();
    assert_eq!(gpu_f16.dtype(), DType::F16);

    let f32_result = gpu_f16.to_dtype(DType::F32).unwrap();
    assert_eq!(f32_result.dtype(), DType::F32);
    assert!(f32_result.device().is_gpu());

    let vals = read_f32(&f32_result);
    assert_close(&vals, &data, 1e-2, "f16_to_f32");
}

// -- BF16 → F32 ---------------------------------------------------------------

#[test]
fn test_gpu_cast_bf16_to_f32() {
    init();
    let data = vec![1.0f32, -0.5, 256.0, 0.125];
    let cpu_f32 = DynTensor::new(&data, &[2, 2], &Device::Cpu).unwrap();
    let cpu_bf16 = cpu_f32.to_dtype(DType::BF16).unwrap();
    let gpu_bf16 = cpu_bf16.to_device(&Device::metal()).unwrap();

    let f32_result = gpu_bf16.to_dtype(DType::F32).unwrap();
    assert_eq!(f32_result.dtype(), DType::F32);
    assert!(f32_result.device().is_gpu());
    assert_eq!(f32_result.dims(), &[2, 2]);

    let vals = read_f32(&f32_result);
    assert_close(&vals, &data, 1e-1, "bf16_to_f32");
}

// -- Edge cases ---------------------------------------------------------------

#[test]
fn test_gpu_cast_empty_tensor() {
    init();
    let empty = DynTensor::zeros(&[0, 3], DType::F32, &Device::metal()).unwrap();
    let f16 = empty.to_dtype(DType::F16).unwrap();
    assert_eq!(f16.dtype(), DType::F16);
    assert_eq!(f16.dims(), &[0, 3]);
}

#[test]
fn test_gpu_cast_scalar() {
    init();
    let scalar = DynTensor::full(&[], 3.14, DType::F32, &Device::metal()).unwrap();
    let f16 = scalar.to_dtype(DType::F16).unwrap();
    assert_eq!(f16.dtype(), DType::F16);
    assert_eq!(f16.dims(), &[] as &[usize]);

    let back = f16.to_dtype(DType::F32).unwrap();
    let vals = read_f32(&back);
    assert!((vals[0] - 3.14).abs() < 0.01, "scalar cast: {}", vals[0]);
}

#[test]
fn test_gpu_cast_large_tensor() {
    init();
    // 256K elements — exercises threadgroup dispatch with many threads.
    let n = 256 * 1024;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 1280.0).collect();
    let gpu = gpu_f32(&data, &[n]);

    let f16 = gpu.to_dtype(DType::F16).unwrap();
    let back = f16.to_dtype(DType::F32).unwrap();
    let vals = read_f32(&back);

    // Check a sample of values (half precision loses precision for large values).
    for &idx in &[0, 100, 1000, 10000, n / 2, n - 1] {
        let expected = data[idx];
        let actual = vals[idx];
        let tol = expected.abs() * 0.01 + 1.0; // 1% relative + 1.0 absolute
        assert!(
            (actual - expected).abs() < tol,
            "idx {idx}: expected {expected}, got {actual}"
        );
    }
}

// -- Same dtype (no-op) -------------------------------------------------------

#[test]
fn test_gpu_cast_same_dtype_is_noop() {
    init();
    let gpu = gpu_f32(&[1.0, 2.0], &[2]);
    let same = gpu.to_dtype(DType::F32).unwrap();
    assert_eq!(same.dtype(), DType::F32);
    let vals = read_f32(&same);
    assert_close(&vals, &[1.0, 2.0], 0.0, "same_dtype_noop");
}
