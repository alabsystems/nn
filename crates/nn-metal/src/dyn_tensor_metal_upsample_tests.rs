#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU parity tests for upsample operations.
//!
//! Validates that `upsample_nearest_1d`, `upsample_nearest_2d`, and
//! `upsample_bilinear_2d` produce identical results on GPU as on CPU,
//! and that overflow guards work on the GPU path.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_gpu_vals, init};

#[test]
fn test_gpu_upsample_nearest_2d_basic() {
    init();
    // [1, 1, 2, 2] upsampled 2x → [1, 1, 4, 4]
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let cpu = DynTensor::new(&data, &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let cpu_result = cpu.upsample_nearest_2d(2, 2).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 2, 2], &Device::metal()).unwrap();
    let gpu_result = gpu.upsample_nearest_2d(2, 2).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "upsample_nearest_2d 2x");
}

#[test]
fn test_gpu_upsample_nearest_2d_asymmetric() {
    init();
    // [1, 1, 2, 3] upsampled 3x height, 2x width → [1, 1, 6, 6]
    let data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 2, 3], &Device::Cpu).unwrap();
    let cpu_result = cpu.upsample_nearest_2d(3, 2).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 2, 3], &Device::metal()).unwrap();
    let gpu_result = gpu.upsample_nearest_2d(3, 2).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "upsample_nearest_2d asym");
}

#[test]
fn test_gpu_upsample_bilinear_2d_basic() {
    init();
    // [1, 1, 2, 2] upsampled 2x → [1, 1, 4, 4]
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let cpu = DynTensor::new(&data, &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let cpu_result = cpu.upsample_bilinear_2d(2.0, 2.0, false).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 2, 2], &Device::metal()).unwrap();
    let gpu_result = gpu.upsample_bilinear_2d(2.0, 2.0, false).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_gpu_vals(&gpu_result, &expected, 1e-5, "upsample_bilinear_2d 2x");
}

#[test]
fn test_gpu_upsample_bilinear_2d_align_corners() {
    init();
    // [1, 1, 3, 3] upsampled 2x with align_corners=true
    let data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 3, 3], &Device::Cpu).unwrap();
    let cpu_result = cpu.upsample_bilinear_2d(2.0, 2.0, true).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 3, 3], &Device::metal()).unwrap();
    let gpu_result = gpu.upsample_bilinear_2d(2.0, 2.0, true).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_gpu_vals(
        &gpu_result,
        &expected,
        1e-5,
        "upsample_bilinear_2d align_corners",
    );
}

#[test]
fn test_gpu_upsample_nearest_2d_batched() {
    init();
    // [2, 3, 2, 2] — batch=2, channels=3, upsampled 2x
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let cpu = DynTensor::new(&data, &[2, 3, 2, 2], &Device::Cpu).unwrap();
    let cpu_result = cpu.upsample_nearest_2d(2, 2).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[2, 3, 2, 2], &Device::metal()).unwrap();
    let gpu_result = gpu.upsample_nearest_2d(2, 2).unwrap();
    assert_eq!(gpu_result.dims(), &[2, 3, 4, 4]);
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "upsample_nearest_2d batched");
}

#[test]
fn test_gpu_upsample_nearest_2d_scale_h_only() {
    init();
    // [1, 1, 2, 3] upsampled 3x height, 1x width → [1, 1, 6, 3]
    let data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 2, 3], &Device::Cpu).unwrap();
    let cpu_result = cpu.upsample_nearest_2d(3, 1).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 2, 3], &Device::metal()).unwrap();
    let gpu_result = gpu.upsample_nearest_2d(3, 1).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 1, 6, 3]);
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "upsample_nearest_2d h_only");
}

#[test]
fn test_gpu_upsample_nearest_2d_overflow_returns_error() {
    init();
    // GPU path: spatial_upsample.rs checked_mul guards overflow.
    let gpu = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::metal()).unwrap();
    let result = gpu.upsample_nearest_2d(usize::MAX, 2);
    assert!(
        result.is_err(),
        "GPU upsample_nearest_2d should return error on overflow"
    );
}

// -- 1D upsample GPU tests ----------------------------------------------------

#[test]
fn test_gpu_upsample_nearest_1d_basic() {
    init();
    // [3] upsampled 2x → [6]
    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let cpu = DynTensor::new(&data, &[3], &Device::Cpu).unwrap();
    let cpu_result = cpu.upsample_nearest_1d(2).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[3], &Device::metal()).unwrap();
    let gpu_result = gpu.upsample_nearest_1d(2).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "upsample_nearest_1d 2x");
}

#[test]
fn test_gpu_upsample_nearest_1d_overflow_returns_error() {
    init();
    // GPU path: spatial_upsample.rs:39 uses checked_mul (#1360 AC1).
    // factor * t overflows usize — must return DimensionOverflow, not panic.
    let gpu = DynTensor::new(&[1.0, 2.0], &[2], &Device::metal()).unwrap();
    let result = gpu.upsample_nearest_1d(usize::MAX);
    assert!(
        result.is_err(),
        "GPU upsample should return DimensionOverflow on overflow"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("overflow") || msg.contains("Overflow"),
        "error should mention overflow: {msg}"
    );
}
