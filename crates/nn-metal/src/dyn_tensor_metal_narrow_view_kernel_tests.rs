#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Narrow-view regression tests for GPU kernel dispatch (#1964).
//!
//! Each test creates a full tensor on GPU, narrows dim-0 to create a view
//! with `byte_offset > 0`, then runs a kernel operation on the view.
//! Results are compared against the equivalent CPU narrow-view operation.
//!
//! These tests exercise the `dispatch_buffers_with_offsets` and
//! `set_buffer_with_offset` paths wired in #1964 to ensure byte_offset
//! is correctly propagated through scatter_add, index_add, cumsum,
//! topk, argmax, and argmin GPU kernels.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use crate::test_common::{assert_close, init};

// ===== scatter_add =====

/// scatter_add where the base tensor `x` is a narrow view.
#[test]
fn test_narrow_view_scatter_add() {
    init();
    // Full tensor [4, 3], narrow to rows 1..3 -> [2, 3]
    let full_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let cpu_full = DynTensor::new(&full_data, &[4, 3], &Device::Cpu).unwrap();
    let gpu_full = cpu_full.to_device(&Device::metal()).unwrap();

    let cpu_view = cpu_full.narrow(0, 1, 2).unwrap(); // rows 1..3
    let gpu_view = gpu_full.narrow(0, 1, 2).unwrap();
    assert_eq!(gpu_view.dims(), &[2, 3]);

    // src and index have same shape as view
    let src_data = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0];
    let src_cpu = DynTensor::new(&src_data, &[2, 3], &Device::Cpu).unwrap();
    let src_gpu = src_cpu.to_device(&Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![1, 0, 1, 0, 1, 0], &[2, 3], &Device::Cpu).unwrap();

    let cpu_result = cpu_view.scatter_add(0, &ids, &src_cpu).unwrap();
    let gpu_result = gpu_view.scatter_add(0, &ids, &src_gpu).unwrap();

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "narrow_view_scatter_add");
}

/// scatter_add where the src tensor is a narrow view.
#[test]
fn test_narrow_view_scatter_add_src_view() {
    init();
    let full_src: Vec<f32> = (0..15).map(|i| (i + 1) as f32).collect();
    let cpu_full_src = DynTensor::new(&full_src, &[5, 3], &Device::Cpu).unwrap();
    let gpu_full_src = cpu_full_src.to_device(&Device::metal()).unwrap();

    let cpu_src = cpu_full_src.narrow(0, 2, 2).unwrap(); // rows 2..4
    let gpu_src = gpu_full_src.narrow(0, 2, 2).unwrap();

    let base = DynTensor::zeros(&[3, 3], DType::F32, &Device::Cpu).unwrap();
    let base_gpu = base.to_device(&Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 1, 2, 2, 0, 1], &[2, 3], &Device::Cpu).unwrap();

    let cpu_result = base.scatter_add(0, &ids, &cpu_src).unwrap();
    let gpu_result = base_gpu.scatter_add(0, &ids, &gpu_src).unwrap();

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "narrow_view_scatter_add_src");
}

// ===== index_add =====

/// index_add where the base tensor `x` is a narrow view.
#[test]
fn test_narrow_view_index_add() {
    init();
    let full_data: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let cpu_full = DynTensor::new(&full_data, &[5, 4], &Device::Cpu).unwrap();
    let gpu_full = cpu_full.to_device(&Device::metal()).unwrap();

    let cpu_view = cpu_full.narrow(0, 2, 3).unwrap(); // rows 2..5 -> [3, 4]
    let gpu_view = gpu_full.narrow(0, 2, 3).unwrap();

    let src_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let src_cpu = DynTensor::new(&src_data, &[2, 4], &Device::Cpu).unwrap();
    let src_gpu = src_cpu.to_device(&Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 2], &[2], &Device::Cpu).unwrap();

    let cpu_result = cpu_view.index_add(0, &ids, &src_cpu).unwrap();
    let gpu_result = gpu_view.index_add(0, &ids, &src_gpu).unwrap();

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "narrow_view_index_add");
}

/// index_add where the src tensor is a narrow view.
#[test]
fn test_narrow_view_index_add_src_view() {
    init();
    let full_src: Vec<f32> = (0..24).map(|i| (i + 1) as f32).collect();
    let cpu_full_src = DynTensor::new(&full_src, &[6, 4], &Device::Cpu).unwrap();
    let gpu_full_src = cpu_full_src.to_device(&Device::metal()).unwrap();

    let cpu_src = cpu_full_src.narrow(0, 1, 3).unwrap(); // rows 1..4
    let gpu_src = gpu_full_src.narrow(0, 1, 3).unwrap();

    let base = DynTensor::zeros(&[5, 4], DType::F32, &Device::Cpu).unwrap();
    let base_gpu = base.to_device(&Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![4, 0, 2], &[3], &Device::Cpu).unwrap();

    let cpu_result = base.index_add(0, &ids, &cpu_src).unwrap();
    let gpu_result = base_gpu.index_add(0, &ids, &gpu_src).unwrap();

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "narrow_view_index_add_src");
}

// ===== cumsum =====

/// cumsum on a narrow view (single-pass: axis_size <= 256).
#[test]
fn test_narrow_view_cumsum() {
    init();
    let full_data: Vec<f32> = (0..24).map(|i| (i + 1) as f32).collect();
    let cpu_full = DynTensor::new(&full_data, &[6, 4], &Device::Cpu).unwrap();
    let gpu_full = cpu_full.to_device(&Device::metal()).unwrap();

    let cpu_view = cpu_full.narrow(0, 1, 4).unwrap(); // [4, 4]
    let gpu_view = gpu_full.narrow(0, 1, 4).unwrap();

    // cumsum along dim 0
    let cpu_result = cpu_view.cumsum(0).unwrap();
    let gpu_result = gpu_view.cumsum(0).unwrap();

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "narrow_view_cumsum_dim0");
}

/// cumsum along dim 1 on a narrow view.
#[test]
fn test_narrow_view_cumsum_dim1() {
    init();
    let full_data: Vec<f32> = (0..30).map(|i| (i + 1) as f32).collect();
    let cpu_full = DynTensor::new(&full_data, &[6, 5], &Device::Cpu).unwrap();
    let gpu_full = cpu_full.to_device(&Device::metal()).unwrap();

    let cpu_view = cpu_full.narrow(0, 2, 3).unwrap(); // [3, 5]
    let gpu_view = gpu_full.narrow(0, 2, 3).unwrap();

    let cpu_result = cpu_view.cumsum(1).unwrap();
    let gpu_result = gpu_view.cumsum(1).unwrap();

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "narrow_view_cumsum_dim1");
}

// ===== topk =====

/// topk on a narrow view.
#[test]
fn test_narrow_view_topk() {
    init();
    let full_data: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let cpu_full = DynTensor::new(&full_data, &[4, 5], &Device::Cpu).unwrap();
    let gpu_full = cpu_full.to_device(&Device::metal()).unwrap();

    let cpu_view = cpu_full.narrow(0, 1, 2).unwrap(); // [2, 5]
    let gpu_view = gpu_full.narrow(0, 1, 2).unwrap();

    let (cpu_vals, cpu_idx) = cpu_view.topk(1, 3).unwrap(); // top-3 along last dim
    let (gpu_vals, gpu_idx) = gpu_view.topk(1, 3).unwrap();

    let cpu_v = cpu_vals.to_flat_vec::<f32>().unwrap();
    let gpu_v = gpu_vals
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_v, &cpu_v, 1e-5, "narrow_view_topk_vals");

    // Verify indices match (U32 tensors).
    let cpu_i = cpu_idx.to_flat_vec::<u32>().unwrap();
    let gpu_i = gpu_idx
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    assert_eq!(gpu_i, cpu_i, "narrow_view_topk_indices mismatch");
}

// ===== argmax / argmin =====

/// argmax on a narrow view.
#[test]
fn test_narrow_view_argmax() {
    init();
    let full_data: Vec<f32> = (0..18).map(|i| i as f32).collect();
    let cpu_full = DynTensor::new(&full_data, &[6, 3], &Device::Cpu).unwrap();
    let gpu_full = cpu_full.to_device(&Device::metal()).unwrap();

    let cpu_view = cpu_full.narrow(0, 2, 3).unwrap(); // [3, 3]
    let gpu_view = gpu_full.narrow(0, 2, 3).unwrap();

    let cpu_result = cpu_view.argmax(1).unwrap();
    let gpu_result = gpu_view.argmax(1).unwrap();

    let cpu_vals = cpu_result.to_flat_vec::<u32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    assert_eq!(gpu_vals, cpu_vals, "narrow_view_argmax mismatch");
}

/// argmin on a narrow view.
#[test]
fn test_narrow_view_argmin() {
    init();
    // Use values where the minimum differs per row to exercise offset correctness.
    let full_data: Vec<f32> = vec![
        9.0, 1.0, 5.0, // row 0
        3.0, 7.0, 2.0, // row 1
        8.0, 4.0, 6.0, // row 2
        1.0, 9.0, 3.0, // row 3
        5.0, 2.0, 8.0, // row 4
    ];
    let cpu_full = DynTensor::new(&full_data, &[5, 3], &Device::Cpu).unwrap();
    let gpu_full = cpu_full.to_device(&Device::metal()).unwrap();

    let cpu_view = cpu_full.narrow(0, 1, 3).unwrap(); // rows 1..4
    let gpu_view = gpu_full.narrow(0, 1, 3).unwrap();

    let cpu_result = cpu_view.argmin(1).unwrap();
    let gpu_result = gpu_view.argmin(1).unwrap();

    let cpu_vals = cpu_result.to_flat_vec::<u32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    assert_eq!(gpu_vals, cpu_vals, "narrow_view_argmin mismatch");
}
