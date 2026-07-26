#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU reduce tests — last-axis, non-last-axis, 4D, var_keepdim (#1147, #2069).
//!
//! The MSL reduce kernel only supports last-axis reductions. For non-last-axis
//! reductions, `gpu_reduce_via_transpose` transposes the target dim to the last
//! position, runs the GPU reduce, and transposes the result back.
//!
//! Also covers last-axis reduce (direct `gpu_reduce` path), 4D tensors,
//! single-element reduces, and `var_keepdim` (decomposed on GPU).

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_gpu_vals, init};

// -- non-last-axis sum --------------------------------------------------------

#[test]
fn test_gpu_sum_dim0_2d() {
    init();
    // [2,3] sum(dim=0) → [3]
    // [[1,2,3],[4,5,6]] → [5,7,9]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let r = t.sum(0).unwrap();
    assert_eq!(r.device(), Device::metal(), "result must stay on GPU");
    assert_eq!(r.dims(), &[3]);
    assert_gpu_vals(&r, &[5.0, 7.0, 9.0], 1e-4, "sum dim0 2d");
}

#[test]
fn test_gpu_sum_dim0_2d_keepdim() {
    init();
    // [2,3] sum_keepdim(dim=0) → [1,3]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let r = t.sum_keepdim(0).unwrap();
    assert_eq!(r.device(), Device::metal(), "result must stay on GPU");
    assert_eq!(r.dims(), &[1, 3]);
    assert_gpu_vals(&r, &[5.0, 7.0, 9.0], 1e-4, "sum_keepdim dim0 2d");
}

#[test]
fn test_gpu_sum_dim0_3d() {
    init();
    // [2,2,3] sum(dim=0) → [2,3]
    // [[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]] → [[8,10,12],[14,16,18]]
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[2, 2, 3], &Device::metal()).unwrap();
    let r = t.sum(0).unwrap();
    assert_eq!(r.device(), Device::metal());
    assert_eq!(r.dims(), &[2, 3]);
    assert_gpu_vals(
        &r,
        &[8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
        1e-4,
        "sum dim0 3d",
    );
}

#[test]
fn test_gpu_sum_dim1_3d() {
    init();
    // [2,3,2] sum(dim=1) → [2,2]
    // [[[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]]] → [[9,12],[27,30]]
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 2], &Device::metal()).unwrap();
    let r = t.sum(1).unwrap();
    assert_eq!(r.device(), Device::metal());
    assert_eq!(r.dims(), &[2, 2]);
    assert_gpu_vals(&r, &[9.0, 12.0, 27.0, 30.0], 1e-4, "sum dim1 3d");
}

// -- non-last-axis mean -------------------------------------------------------

#[test]
fn test_gpu_mean_dim0_2d() {
    init();
    // [2,3] mean(dim=0) → [3]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let r = t.mean(0).unwrap();
    assert_eq!(r.device(), Device::metal());
    assert_eq!(r.dims(), &[3]);
    assert_gpu_vals(&r, &[2.5, 3.5, 4.5], 1e-4, "mean dim0 2d");
}

#[test]
fn test_gpu_mean_keepdim_dim0_3d() {
    init();
    // [2,2,3] mean_keepdim(dim=0) → [1,2,3]
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[2, 2, 3], &Device::metal()).unwrap();
    let r = t.mean_keepdim(0).unwrap();
    assert_eq!(r.device(), Device::metal());
    assert_eq!(r.dims(), &[1, 2, 3]);
    assert_gpu_vals(
        &r,
        &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        1e-4,
        "mean_keepdim dim0 3d",
    );
}

// -- non-last-axis max --------------------------------------------------------

#[test]
fn test_gpu_max_dim0_2d() {
    init();
    // [2,3] max(dim=0) → [3]
    let t = DynTensor::new(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let r = t.max(0).unwrap();
    assert_eq!(r.device(), Device::metal());
    assert_eq!(r.dims(), &[3]);
    assert_gpu_vals(&r, &[4.0, 5.0, 6.0], 1e-4, "max dim0 2d");
}

// -- non-last-axis min --------------------------------------------------------

#[test]
fn test_gpu_min_dim0_2d() {
    init();
    // [2,3] min(dim=0) → [3]
    let t = DynTensor::new(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let r = t.min(0).unwrap();
    assert_eq!(r.device(), Device::metal());
    assert_eq!(r.dims(), &[3]);
    assert_gpu_vals(&r, &[1.0, 2.0, 3.0], 1e-4, "min dim0 2d");
}

// -- non-last-axis reduces stay on GPU for follow-up ops ----------------------

#[test]
fn test_gpu_reduce_dim0_usable_in_followup() {
    init();
    // sum(dim=0) result can be used in subsequent GPU operations without
    // "mixed device" errors — the primary motivation for this change.
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let s = t.sum_keepdim(0).unwrap(); // [1,3]
    assert_eq!(s.device(), Device::metal());
    // broadcast_sub: [2,3] - [1,3] → centering
    let centered = t.broadcast_sub(&s).unwrap();
    assert_eq!(centered.device(), Device::metal());
    assert_gpu_vals(
        &centered,
        &[-4.0, -5.0, -6.0, -1.0, -2.0, -3.0],
        1e-4,
        "sum_keepdim(0)→sub",
    );
}

// -- parity with CPU ----------------------------------------------------------

#[test]
fn test_gpu_reduce_dim0_parity_with_cpu() {
    init();
    // Verify GPU non-last-axis reduce matches CPU results exactly.
    let data: Vec<f32> = (0..24).map(|x| x as f32 * 0.1).collect();
    let gpu = DynTensor::new(&data, &[2, 3, 4], &Device::metal()).unwrap();
    let cpu = DynTensor::new(&data, &[2, 3, 4], &Device::Cpu).unwrap();

    // sum(dim=0)
    let gpu_sum = gpu.sum(0).unwrap();
    let cpu_sum = cpu.sum(0).unwrap();
    assert_eq!(gpu_sum.dims(), cpu_sum.dims());
    let gpu_vals = gpu_sum
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_sum.to_flat_vec::<f32>().unwrap();
    for (i, (g, c)) in gpu_vals.iter().zip(&cpu_vals).enumerate() {
        assert!(
            (g - c).abs() < 1e-4,
            "sum dim0 parity[{i}]: gpu={g}, cpu={c}"
        );
    }

    // mean_keepdim(dim=1)
    let gpu_mean = gpu.mean_keepdim(1).unwrap();
    let cpu_mean = cpu.mean_keepdim(1).unwrap();
    assert_eq!(gpu_mean.dims(), cpu_mean.dims());
    let gpu_vals = gpu_mean
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_mean.to_flat_vec::<f32>().unwrap();
    for (i, (g, c)) in gpu_vals.iter().zip(&cpu_vals).enumerate() {
        assert!(
            (g - c).abs() < 1e-4,
            "mean_keepdim dim1 parity[{i}]: gpu={g}, cpu={c}"
        );
    }
}

// -- zero-length dim guards (#1855) -------------------------------------------
// Metal cannot allocate zero-byte buffers, so zero-length dim GPU tensors only
// arise from upstream GPU operations (e.g., narrow to length 0).  The guards in
// gpu_reduce (line 30) and gpu_reduce_compensated (line 144) are defense-in-depth.
// These tests verify the CPU path returns ZeroLengthDimension — the same error
// the GPU guard produces — ensuring CPU/GPU behavioral parity.

#[test]
fn test_mean_zero_length_dim_returns_error() {
    init();
    // [2, 0, 3] mean(dim=1) must return ZeroLengthDimension, not NaN.
    let t = DynTensor::new(&[], &[2, 0, 3], &Device::Cpu).unwrap();
    let err = t.mean(1).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("ero") && msg.contains("ength"),
        "expected ZeroLengthDimension error, got: {msg}"
    );
}

#[test]
fn test_mean_keepdim_zero_length_dim_returns_error() {
    init();
    // mean_keepdim on zero-length axis must also return error.
    let t = DynTensor::new(&[], &[2, 0, 3], &Device::Cpu).unwrap();
    let err = t.mean_keepdim(1).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("ero") && msg.contains("ength"),
        "expected ZeroLengthDimension error, got: {msg}"
    );
}

// -- Sum/Max/Min zero-length identity values (#2058) --------------------------
// Prior to the fix, GPU Sum/Max/Min over zero-length axes would dispatch a
// kernel that reads 0 elements, leaving the output buffer with uninitialized
// memory contents. The fix returns correct identity values matching CPU:
// Sum → 0.0, Max → NEG_INFINITY, Min → INFINITY.

#[test]
fn test_sum_zero_length_dim_returns_zero() {
    init();
    // [2, 0, 3] sum(dim=1) should return [2, 3] filled with 0.0
    let t = DynTensor::new(&[], &[2, 0, 3], &Device::Cpu).unwrap();
    let result = t.sum(1).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|&v| v == 0.0),
        "sum of empty axis should be 0.0, got: {vals:?}"
    );
}

#[test]
fn test_sum_keepdim_zero_length_dim_returns_zero() {
    init();
    let t = DynTensor::new(&[], &[2, 0, 3], &Device::Cpu).unwrap();
    let result = t.sum_keepdim(1).unwrap();
    assert_eq!(result.dims(), &[2, 1, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|&v| v == 0.0),
        "sum_keepdim of empty axis should be 0.0"
    );
}

#[test]
fn test_max_zero_length_dim_returns_neg_inf() {
    init();
    let t = DynTensor::new(&[], &[2, 0, 3], &Device::Cpu).unwrap();
    let result = t.max_keepdim(1).unwrap();
    assert_eq!(result.dims(), &[2, 1, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|&v| v == f32::NEG_INFINITY),
        "max of empty axis should be NEG_INFINITY, got: {vals:?}"
    );
}

#[test]
fn test_min_zero_length_dim_returns_inf() {
    init();
    let t = DynTensor::new(&[], &[2, 0, 3], &Device::Cpu).unwrap();
    let result = t.min_keepdim(1).unwrap();
    assert_eq!(result.dims(), &[2, 1, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|&v| v == f32::INFINITY),
        "min of empty axis should be INFINITY, got: {vals:?}"
    );
}
