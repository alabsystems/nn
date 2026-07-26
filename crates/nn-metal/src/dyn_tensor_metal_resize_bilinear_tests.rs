// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU parity tests for `resize_bilinear`.
//!
//! Validates that `resize_bilinear` produces identical results on GPU as on CPU
//! for various input shapes, upscale/downscale ratios, and batch configurations.
//!
//! Part of #3535.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_gpu_vals, init};

#[test]
fn test_gpu_resize_bilinear_upsample_basic() {
    init();
    // [1, 1, 2, 2] → [1, 1, 4, 4] (2x upscale)
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let cpu = DynTensor::new(&data, &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let cpu_result = cpu.resize_bilinear(4, 4).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 2, 2], &Device::metal()).unwrap();
    let gpu_result = gpu.resize_bilinear(4, 4).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_gpu_vals(&gpu_result, &expected, 1e-5, "resize_bilinear 2x upsample");
}

#[test]
fn test_gpu_resize_bilinear_downsample() {
    init();
    // [1, 1, 4, 4] → [1, 1, 2, 2] (2x downscale)
    let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu.resize_bilinear(2, 2).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::metal()).unwrap();
    let gpu_result = gpu.resize_bilinear(2, 2).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_gpu_vals(&gpu_result, &expected, 1e-5, "resize_bilinear 2x downsample");
}

#[test]
fn test_gpu_resize_bilinear_asymmetric() {
    init();
    // [1, 1, 3, 4] → [1, 1, 6, 2] (different scale factors per axis)
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 3, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu.resize_bilinear(6, 2).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 3, 4], &Device::metal()).unwrap();
    let gpu_result = gpu.resize_bilinear(6, 2).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_gpu_vals(
        &gpu_result,
        &expected,
        1e-5,
        "resize_bilinear asymmetric",
    );
}

#[test]
fn test_gpu_resize_bilinear_batched() {
    init();
    // [2, 3, 4, 4] → [2, 3, 8, 8] (batch=2, channels=3)
    let data: Vec<f32> = (1..=96).map(|x| x as f32 * 0.1).collect();
    let cpu = DynTensor::new(&data, &[2, 3, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu.resize_bilinear(8, 8).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[2, 3, 4, 4], &Device::metal()).unwrap();
    let gpu_result = gpu.resize_bilinear(8, 8).unwrap();
    assert_eq!(gpu_result.dims(), &[2, 3, 8, 8]);
    assert_gpu_vals(
        &gpu_result,
        &expected,
        1e-5,
        "resize_bilinear batched",
    );
}

#[test]
fn test_gpu_resize_bilinear_rank3() {
    init();
    // [3, 4, 4] → [3, 2, 2] (rank 3 input, no batch dim)
    let data: Vec<f32> = (1..=48).map(|x| x as f32 * 0.1).collect();
    let cpu = DynTensor::new(&data, &[3, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu.resize_bilinear(2, 2).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[3, 4, 4], &Device::metal()).unwrap();
    let gpu_result = gpu.resize_bilinear(2, 2).unwrap();
    assert_eq!(gpu_result.dims(), &[3, 2, 2]);
    assert_gpu_vals(&gpu_result, &expected, 1e-5, "resize_bilinear rank3");
}

#[test]
fn test_gpu_resize_bilinear_identity() {
    init();
    // [1, 1, 3, 3] → [1, 1, 3, 3] (identity — handled by DynTensor before GPU)
    let data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 3, 3], &Device::Cpu).unwrap();
    let cpu_result = cpu.resize_bilinear(3, 3).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 3, 3], &Device::metal()).unwrap();
    let gpu_result = gpu.resize_bilinear(3, 3).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "resize_bilinear identity");
}

#[test]
fn test_gpu_resize_bilinear_large() {
    init();
    // [1, 3, 32, 32] → [1, 3, 224, 224] (typical image preprocessing)
    let n = 1 * 3 * 32 * 32;
    let data: Vec<f32> = (0..n).map(|x| (x as f32) / (n as f32)).collect();
    let cpu = DynTensor::new(&data, &[1, 3, 32, 32], &Device::Cpu).unwrap();
    let cpu_result = cpu.resize_bilinear(224, 224).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 3, 32, 32], &Device::metal()).unwrap();
    let gpu_result = gpu.resize_bilinear(224, 224).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 3, 224, 224]);
    assert_gpu_vals(
        &gpu_result,
        &expected,
        1e-4,
        "resize_bilinear 32→224",
    );
}

#[test]
fn test_gpu_resize_bilinear_1x1() {
    init();
    // [1, 1, 4, 4] → [1, 1, 1, 1] (extreme downscale)
    let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu.resize_bilinear(1, 1).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::metal()).unwrap();
    let gpu_result = gpu.resize_bilinear(1, 1).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 1, 1, 1]);
    assert_gpu_vals(
        &gpu_result,
        &expected,
        1e-5,
        "resize_bilinear extreme downsample 1x1",
    );
}
