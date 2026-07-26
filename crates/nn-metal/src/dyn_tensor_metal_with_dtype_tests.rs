#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU tests for WithDType generic extraction methods on DynTensor.
//!
//! Tests `to_vec1`, `to_vec2`, `to_vec3`, and `to_scalar` on Metal GPU
//! tensors, exercising the GPU-to-CPU transfer path that is untested by
//! the CPU-only tests in `with_dtype_tests.rs`.
//!
//! Covers: #1151

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use crate::test_common::init;

// -- AC1: to_vec1::<f32>() on GPU tensor --------------------------------------

#[test]
fn test_to_vec1_f32_gpu() {
    init();
    let data = vec![1.0f32, 2.5, -3.0, 4.75];
    let gpu = DynTensor::new(&data, &[4], &Device::metal()).unwrap();
    assert_eq!(gpu.device(), Device::metal());

    let result = gpu.to_vec1::<f32>().unwrap();
    assert_eq!(result, data);
}

// -- AC2: to_vec2::<f32>() on GPU tensor --------------------------------------

#[test]
fn test_to_vec2_f32_gpu() {
    init();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gpu = DynTensor::new(&data, &[2, 3], &Device::metal()).unwrap();
    assert_eq!(gpu.device(), Device::metal());

    let result = gpu.to_vec2::<f32>().unwrap();
    assert_eq!(result, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
}

// -- to_vec3::<f32>() on GPU tensor -------------------------------------------

#[test]
fn test_to_vec3_f32_gpu() {
    init();
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let gpu = DynTensor::new(&data, &[2, 3, 4], &Device::metal()).unwrap();
    assert_eq!(gpu.device(), Device::metal());

    let result = gpu.to_vec3::<f32>().unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].len(), 3);
    assert_eq!(result[0][0].len(), 4);
    assert_eq!(result[0][0], vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(result[1][2], vec![21.0, 22.0, 23.0, 24.0]);
}

// -- AC3: to_scalar::<f32>() on GPU tensor ------------------------------------

#[test]
fn test_to_scalar_f32_gpu() {
    init();
    let gpu = DynTensor::new(&[42.0f32], &[1], &Device::metal()).unwrap();
    assert_eq!(gpu.device(), Device::metal());

    let result = gpu.to_scalar::<f32>().unwrap();
    assert_eq!(result, 42.0);
}

#[test]
fn test_to_scalar_f32_gpu_0d() {
    init();
    // 0-D (scalar) tensor: shape [] with 1 element
    let gpu = DynTensor::full(&[], 7.5, DType::F32, &Device::metal()).unwrap();
    assert_eq!(gpu.device(), Device::metal());
    assert_eq!(gpu.numel(), 1);

    let result = gpu.to_scalar::<f32>().unwrap();
    assert_eq!(result, 7.5);
}

// -- AC4: to_vec1 on non-f32 dtype GPU tensor ---------------------------------
// Metal backend stores all data as f32 buffers. U8 tensors roundtrip via f32
// promotion in cpu_to_gpu and u8 reconstruction in gpu_to_cpu.
// U32 tensors store as native u32 in Metal buffers — no f32 intermediate.

#[test]
fn test_to_vec1_u8_gpu() {
    init();
    let data = vec![10u8, 20, 30, 255];
    let gpu = DynTensor::from_vec_u8(data.clone(), &[4], &Device::metal()).unwrap();
    assert_eq!(gpu.device(), Device::metal());
    assert_eq!(gpu.dtype(), DType::U8);

    let result = gpu.to_vec1::<u8>().unwrap();
    assert_eq!(result, data);
}

// -- Regression: U32 values >= 2^24 round-trip correctly (fixes #1322) --------
// Before the fix, multi-dim U32 tensors went through U32→F32→u32 conversion,
// which loses precision for values >= 16,777,216 (2^24) due to f32 mantissa.

#[test]
fn test_u32_large_values_roundtrip_1d() {
    init();
    // Values near and above the F32 precision boundary (2^24 = 16_777_216).
    let data = vec![0u32, 1, 16_777_215, 16_777_216, 16_777_217, u32::MAX];
    let gpu = DynTensor::from_vec_u32(data.clone(), &[6], &Device::Cpu)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    assert_eq!(gpu.device(), Device::metal());
    assert_eq!(gpu.dtype(), DType::U32);

    let result = gpu
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec1::<u32>()
        .unwrap();
    assert_eq!(result, data);
}

#[test]
fn test_u32_large_values_roundtrip_2d() {
    init();
    // Multi-dim U32 tensor — this was the buggy path before #1322 fix.
    let data = vec![16_777_215u32, 16_777_216, 16_777_217, 100_000_000];
    let gpu = DynTensor::from_vec_u32(data.clone(), &[2, 2], &Device::Cpu)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    assert_eq!(gpu.device(), Device::metal());
    assert_eq!(gpu.dtype(), DType::U32);
    assert_eq!(gpu.dims(), &[2, 2]);

    let result = gpu
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    assert_eq!(result, data);
}

#[test]
fn test_u32_large_values_roundtrip_3d() {
    init();
    // 3D tensor to exercise the general multi-dim path.
    let data = vec![
        0u32,
        u32::MAX,
        16_777_216,
        16_777_217,
        999_999_999,
        1,
        2,
        42,
    ];
    let gpu = DynTensor::from_vec_u32(data.clone(), &[2, 2, 2], &Device::Cpu)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    assert_eq!(gpu.device(), Device::metal());

    let result = gpu
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    assert_eq!(result, data);
}
