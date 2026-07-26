#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Metal GPU-native shape ops (#1084 D4).
//!
//! Each test creates tensors on Metal GPU, runs the shape operation
//! (which now dispatches via MetalDynBackend instead of CPU round-trip),
//! then transfers the result to CPU and compares against the CPU reference.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_close, init};

// -- Narrow tests -------------------------------------------------------------

#[test]
fn test_gpu_narrow() {
    init();
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[3, 4], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // Narrow dim=1, start=1, len=2: pick columns 1..3
    let gpu_result = gpu.narrow(1, 1, 2).unwrap();
    assert_eq!(gpu_result.dims(), &[3, 2]);
    assert_eq!(gpu_result.device(), Device::metal());

    let cpu_result = cpu.narrow(1, 1, 2).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "narrow");
}

#[test]
fn test_gpu_narrow_dim0() {
    init();
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[4, 6], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // Narrow dim=0, start=1, len=2: pick rows 1..3
    let gpu_result = gpu.narrow(0, 1, 2).unwrap();
    assert_eq!(gpu_result.dims(), &[2, 6]);

    let cpu_result = cpu.narrow(0, 1, 2).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "narrow_dim0");
}

/// Dim-0 narrow on a 3D tensor — the LSTM use case where
/// `input.narrow(0, t, 1)` extracts a single timestep from
/// `[seq_len, batch, hidden_size]`.
#[test]
fn test_gpu_narrow_dim0_3d_single_slice() {
    init();
    // Simulate [seq_len=4, batch=2, hidden=3]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[4, 2, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // Narrow dim=0, start=2, len=1: extract timestep 2
    let gpu_result = gpu.narrow(0, 2, 1).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 2, 3]);

    let cpu_result = cpu.narrow(0, 2, 1).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "narrow_dim0_3d_single");
}

/// Dim-0 narrow starting at offset 0 (beginning of buffer).
#[test]
fn test_gpu_narrow_dim0_start_zero() {
    init();
    let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[5, 4], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let gpu_result = gpu.narrow(0, 0, 3).unwrap();
    assert_eq!(gpu_result.dims(), &[3, 4]);

    let cpu_result = cpu.narrow(0, 0, 3).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "narrow_dim0_start_zero");
}

/// Dim-0 narrow on 1D tensor (single dimension).
#[test]
fn test_gpu_narrow_dim0_1d() {
    init();
    let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[10], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let gpu_result = gpu.narrow(0, 3, 4).unwrap();
    assert_eq!(gpu_result.dims(), &[4]);

    let cpu_result = cpu.narrow(0, 3, 4).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "narrow_dim0_1d");
}

// -- Transpose tests ----------------------------------------------------------

#[test]
fn test_gpu_transpose() {
    init();
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[2, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let gpu_result = gpu.transpose(0, 1).unwrap();
    assert_eq!(gpu_result.dims(), &[3, 2]);
    assert_eq!(gpu_result.device(), Device::metal());

    let cpu_result = cpu.transpose(0, 1).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "transpose");
}

#[test]
fn test_gpu_transpose_3d() {
    init();
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[2, 3, 4], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // Swap dims 0 and 2: [2,3,4] -> [4,3,2]
    let gpu_result = gpu.transpose(0, 2).unwrap();
    assert_eq!(gpu_result.dims(), &[4, 3, 2]);

    let cpu_result = cpu.transpose(0, 2).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "transpose_3d");
}

// -- Permute tests ------------------------------------------------------------

#[test]
fn test_gpu_permute() {
    init();
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[2, 3, 4], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // Permute [2,3,4] -> [4,2,3] via axes [2,0,1]
    let gpu_result = gpu.permute([2, 0, 1]).unwrap();
    assert_eq!(gpu_result.dims(), &[4, 2, 3]);
    assert_eq!(gpu_result.device(), Device::metal());

    let cpu_result = cpu.permute([2, 0, 1]).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "permute");
}

// -- Cat tests ----------------------------------------------------------------

#[test]
fn test_gpu_cat_dim0() {
    init();
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::metal()).unwrap();
    let b = DynTensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2], &Device::metal()).unwrap();

    let gpu_result = DynTensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(gpu_result.dims(), &[4, 2]);
    assert_eq!(gpu_result.device(), Device::metal());

    let a_cpu = a.to_device(&Device::Cpu).unwrap();
    let b_cpu = b.to_device(&Device::Cpu).unwrap();
    let cpu_result = DynTensor::cat(&[&a_cpu, &b_cpu], 0).unwrap();

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "cat_dim0");
}

#[test]
fn test_gpu_cat_dim1() {
    init();
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::metal()).unwrap();
    let b = DynTensor::new(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 3], &Device::metal()).unwrap();

    let gpu_result = DynTensor::cat(&[&a, &b], 1).unwrap();
    assert_eq!(gpu_result.dims(), &[2, 5]);

    let a_cpu = a.to_device(&Device::Cpu).unwrap();
    let b_cpu = b.to_device(&Device::Cpu).unwrap();
    let cpu_result = DynTensor::cat(&[&a_cpu, &b_cpu], 1).unwrap();

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "cat_dim1");
}

// -- Expand tests -------------------------------------------------------------

#[test]
fn test_gpu_expand_basic() {
    init();
    // [1, 4] -> [3, 4]: expand dim 0 from 1 to 3
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let cpu = DynTensor::new(&data, &[1, 4], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let gpu_result = gpu.expand([3, 4]).unwrap();
    assert_eq!(gpu_result.dims(), &[3, 4]);
    assert_eq!(gpu_result.device(), Device::metal());

    let cpu_result = cpu.expand([3, 4]).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "expand_basic");
}

#[test]
fn test_gpu_expand_multi_dim() {
    init();
    // [1, 3, 1] -> [2, 3, 4]: expand dims 0 and 2
    let data = vec![1.0f32, 2.0, 3.0];
    let cpu = DynTensor::new(&data, &[1, 3, 1], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let gpu_result = gpu.expand([2, 3, 4]).unwrap();
    assert_eq!(gpu_result.dims(), &[2, 3, 4]);

    let cpu_result = cpu.expand([2, 3, 4]).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "expand_multi_dim");
}

// Slice-set, single-tensor cat, and packed cat tests extracted to reduce file size.
#[path = "dyn_tensor_metal_shape_ops_slice_set_tests.rs"]
mod slice_set_tests;
