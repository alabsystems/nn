#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-native index_select integration tests.
//!
//! Extracted from `dyn_tensor_metal_shape_ops_tests.rs` to stay under 500-line limit.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_close, init};

#[test]
fn test_gpu_index_select_dim0_2d() {
    init();
    // Weight table: 4 rows x 3 cols (like a small embedding table)
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[4, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // Select rows [2, 0, 3]
    let ids = DynTensor::from_vec_u32(vec![2, 0, 3], &[3], &Device::Cpu).unwrap();

    let gpu_result = gpu.index_select(&ids, 0).unwrap();
    assert_eq!(gpu_result.dims(), &[3, 3]);
    assert_eq!(gpu_result.device(), Device::metal());

    let cpu_result = cpu.index_select(&ids, 0).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "index_select_dim0_2d");
}

#[test]
fn test_gpu_index_select_dim0_3d() {
    init();
    // 3D tensor: [4, 2, 3] — select along first axis
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[4, 2, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let ids = DynTensor::from_vec_u32(vec![1, 3], &[2], &Device::Cpu).unwrap();

    let gpu_result = gpu.index_select(&ids, 0).unwrap();
    assert_eq!(gpu_result.dims(), &[2, 2, 3]);
    assert_eq!(gpu_result.device(), Device::metal());

    let cpu_result = cpu.index_select(&ids, 0).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "index_select_dim0_3d");
}

#[test]
fn test_gpu_index_select_repeated_indices() {
    init();
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[4, 2], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // Repeat index 1 three times
    let ids = DynTensor::from_vec_u32(vec![1, 1, 1], &[3], &Device::Cpu).unwrap();

    let gpu_result = gpu.index_select(&ids, 0).unwrap();
    assert_eq!(gpu_result.dims(), &[3, 2]);

    let cpu_result = cpu.index_select(&ids, 0).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "index_select_repeated");
}

#[test]
fn test_gpu_index_select_single_row() {
    init();
    // Embedding-like: large vocab, select one token
    let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[8, 4], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let ids = DynTensor::from_vec_u32(vec![5], &[1], &Device::Cpu).unwrap();

    let gpu_result = gpu.index_select(&ids, 0).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 4]);

    let expected = vec![20.0, 21.0, 22.0, 23.0]; // row 5
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &expected, 0.0, "index_select_single_row");
}

#[test]
fn test_gpu_index_select_i64_indices() {
    init();
    // I64 is the standard dtype for token IDs (per #1207).
    // Verify GPU embedding dispatch handles I64 indices natively.
    let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[5, 4], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let ids = DynTensor::from_vec_i64(vec![3, 1, 4], &[3], &Device::Cpu).unwrap();

    let gpu_result = gpu.index_select(&ids, 0).unwrap();
    assert_eq!(gpu_result.dims(), &[3, 4]);
    assert_eq!(gpu_result.device(), Device::metal());

    let cpu_result = cpu.index_select(&ids, 0).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "index_select_i64_indices");
}

/// GPU index_select along dim=1 (native GPU dispatch, no CPU fallback).
///
/// Before #1997, dim!=0 fell back to CPU. Now the generalized kernel
/// handles arbitrary dim on Metal natively.
#[test]
fn test_gpu_index_select_dim1_native() {
    init();
    // [3, 4] tensor, select columns 0 and 2 along dim=1
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[3, 4], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let ids = DynTensor::from_vec_u32(vec![0, 2], &[2], &Device::Cpu).unwrap();

    let gpu_result = gpu.index_select(&ids, 1).unwrap();
    assert_eq!(gpu_result.dims(), &[3, 2]);
    assert_eq!(gpu_result.device(), Device::metal());

    let cpu_result = cpu.index_select(&ids, 1).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "index_select_dim1_native");
}

/// GPU index_select along dim=1 on a 3D tensor.
///
/// Exercises the outer/inner decomposition: outer_size=2, inner_size=4.
#[test]
fn test_gpu_index_select_dim1_3d() {
    init();
    // [2, 5, 4] tensor, select 3 slices along dim=1
    let data: Vec<f32> = (0..40).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[2, 5, 4], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let ids = DynTensor::from_vec_u32(vec![4, 1, 0], &[3], &Device::Cpu).unwrap();

    let gpu_result = gpu.index_select(&ids, 1).unwrap();
    assert_eq!(gpu_result.dims(), &[2, 3, 4]);
    assert_eq!(gpu_result.device(), Device::metal());

    let cpu_result = cpu.index_select(&ids, 1).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "index_select_dim1_3d");
}

/// GPU index_select along dim=2 (last dim of a 3D tensor).
///
/// Exercises inner_size=1 with outer_size>1.
#[test]
fn test_gpu_index_select_dim2_3d() {
    init();
    // [3, 4, 5] tensor, select 2 elements along last dim
    let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[3, 4, 5], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let ids = DynTensor::from_vec_u32(vec![3, 0], &[2], &Device::Cpu).unwrap();

    let gpu_result = gpu.index_select(&ids, 2).unwrap();
    assert_eq!(gpu_result.dims(), &[3, 4, 2]);
    assert_eq!(gpu_result.device(), Device::metal());

    let cpu_result = cpu.index_select(&ids, 2).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "index_select_dim2_3d");
}

/// GPU index_select on rank-1 tensor (dim=0, inner_size=1).
///
/// Before #1997, rank < 2 fell back to CPU. Now rank-1 is supported.
#[test]
fn test_gpu_index_select_rank1() {
    init();
    let data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let cpu = DynTensor::new(&data, &[5], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let ids = DynTensor::from_vec_u32(vec![4, 2, 0], &[3], &Device::Cpu).unwrap();

    let gpu_result = gpu.index_select(&ids, 0).unwrap();
    assert_eq!(gpu_result.dims(), &[3]);
    assert_eq!(gpu_result.device(), Device::metal());

    let cpu_result = cpu.index_select(&ids, 0).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "index_select_rank1");
}

/// Regression test: index_select with index > 2^24 (16,777,216).
/// IEEE 754 f32 has 24-bit mantissa, so integers > 2^24 lose precision.
/// Index 16,777,217 stored as f32 becomes 16,777,216.0 — wrong row.
/// This test verifies native u32 index buffers preserve precision.
///
/// Issue: #1490
#[test]
fn test_gpu_index_select_index_beyond_f32_precision() {
    init();
    // We need a weight table with > 2^24 rows. Use 2^24 + 2 rows × 1 col.
    // At 4 bytes each = ~67 MB. Same approach as the gather regression test.
    let n: usize = (1 << 24) + 2; // 16_777_218
    let target_idx: u32 = (1 << 24) + 1; // 16_777_217

    // Create weight table: zeros everywhere except sentinel values.
    let mut data = vec![0.0f32; n];
    data[(1 << 24) as usize] = 42.0; // row 16_777_216
    data[target_idx as usize] = 99.0; // row 16_777_217

    let gpu = DynTensor::new(&data, &[n, 1], &Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![target_idx], &[1], &Device::Cpu).unwrap();

    let result = gpu.index_select(&ids, 0).unwrap();
    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // If indices were stored as f32, target_idx (16_777_217) would round to
    // 16_777_216, returning 42.0 instead of the correct 99.0.
    assert_eq!(
        vals[0], 99.0,
        "index_select at row 16_777_217 must return 99.0 (not 42.0 from f32 rounding)"
    );
}

/// GPU index_select with an OOB index must return an error, not silently clamp.
///
/// Before #1597, the MSL kernel clamped OOB indices to the last row, producing
/// wrong results instead of errors. Host-side pre-validation now catches this.
#[test]
fn test_gpu_index_select_oob_returns_error() {
    init();
    // Weight table: 4 rows x 2 cols
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let gpu = DynTensor::new(&data, &[4, 2], &Device::metal()).unwrap();

    // Index 4 is out of bounds (valid range: 0..3)
    let ids = DynTensor::from_vec_u32(vec![0, 4], &[2], &Device::Cpu).unwrap();
    let result = gpu.index_select(&ids, 0);

    assert!(result.is_err(), "OOB index_select must return Err");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("out of bounds") || msg.contains("out of range"),
        "error must mention out of bounds/range, got: {msg}"
    );
}

/// GPU index_select with all-valid indices must still succeed.
#[test]
fn test_gpu_index_select_max_valid_index() {
    init();
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let gpu = DynTensor::new(&data, &[4, 2], &Device::metal()).unwrap();

    // Index 3 is the last valid row
    let ids = DynTensor::from_vec_u32(vec![3], &[1], &Device::Cpu).unwrap();
    let result = gpu.index_select(&ids, 0).unwrap();
    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&vals, &[6.0, 7.0], 1e-6, "index_select_max_valid");
}

// -- index_select_unchecked tests (Part of #2653) ----------------------------

/// Unchecked index_select with GPU U32 indices — basic correctness.
#[test]
fn test_gpu_index_select_unchecked_u32() {
    init();
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let gpu = DynTensor::new(&data, &[4, 3], &Device::metal()).unwrap();

    let ids = DynTensor::from_vec_u32(vec![2, 0, 3], &[3], &Device::metal()).unwrap();
    let result = gpu.index_select_unchecked(&ids, 0).unwrap();
    assert_eq!(result.dims(), &[3, 3]);
    assert_eq!(result.device(), Device::metal());

    let cpu_data = DynTensor::new(&data, &[4, 3], &Device::Cpu).unwrap();
    let cpu_ids = DynTensor::from_vec_u32(vec![2, 0, 3], &[3], &Device::Cpu).unwrap();
    let expected = cpu_data.index_select(&cpu_ids, 0).unwrap();

    let got = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let exp = expected.to_flat_vec::<f32>().unwrap();
    assert_close(&got, &exp, 0.0, "unchecked_u32");
}

/// Unchecked index_select with GPU F32 indices — the main perf use case.
/// F32 indices get cast to uint inline in MSL without CPU readback.
#[test]
fn test_gpu_index_select_unchecked_f32() {
    init();
    let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let gpu = DynTensor::new(&data, &[5, 4], &Device::metal()).unwrap();

    // F32 indices on GPU: [3.0, 1.0, 4.0] → rows 3, 1, 4.
    let ids_data: Vec<f32> = vec![3.0, 1.0, 4.0];
    let ids = DynTensor::new(&ids_data, &[3], &Device::metal()).unwrap();
    let result = gpu.index_select_unchecked(&ids, 0).unwrap();
    assert_eq!(result.dims(), &[3, 4]);

    let cpu_data = DynTensor::new(&data, &[5, 4], &Device::Cpu).unwrap();
    let cpu_ids = DynTensor::from_vec_u32(vec![3, 1, 4], &[3], &Device::Cpu).unwrap();
    let expected = cpu_data.index_select(&cpu_ids, 0).unwrap();

    let got = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let exp = expected.to_flat_vec::<f32>().unwrap();
    assert_close(&got, &exp, 0.0, "unchecked_f32");
}

/// Unchecked index_select along dim=1 with F32 indices.
#[test]
fn test_gpu_index_select_unchecked_dim1_f32() {
    init();
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let gpu = DynTensor::new(&data, &[3, 4], &Device::metal()).unwrap();

    // Select columns 0 and 2 with F32 indices.
    let ids = DynTensor::new(&[0.0_f32, 2.0], &[2], &Device::metal()).unwrap();
    let result = gpu.index_select_unchecked(&ids, 1).unwrap();
    assert_eq!(result.dims(), &[3, 2]);

    let cpu_data = DynTensor::new(&data, &[3, 4], &Device::Cpu).unwrap();
    let cpu_ids = DynTensor::from_vec_u32(vec![0, 2], &[2], &Device::Cpu).unwrap();
    let expected = cpu_data.index_select(&cpu_ids, 1).unwrap();

    let got = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let exp = expected.to_flat_vec::<f32>().unwrap();
    assert_close(&got, &exp, 0.0, "unchecked_dim1_f32");
}

/// Unchecked output matches checked output exactly (U32 path).
#[test]
fn test_gpu_index_select_unchecked_matches_checked() {
    init();
    let data: Vec<f32> = (0..40).map(|i| i as f32).collect();
    let gpu = DynTensor::new(&data, &[2, 5, 4], &Device::metal()).unwrap();

    let ids = DynTensor::from_vec_u32(vec![4, 1, 0], &[3], &Device::metal()).unwrap();

    let checked = gpu.index_select(&ids, 1).unwrap();
    let unchecked = gpu.index_select_unchecked(&ids, 1).unwrap();

    assert_eq!(checked.dims(), unchecked.dims());

    let c_vals = checked
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let u_vals = unchecked
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&c_vals, &u_vals, 0.0, "unchecked_matches_checked");
}
