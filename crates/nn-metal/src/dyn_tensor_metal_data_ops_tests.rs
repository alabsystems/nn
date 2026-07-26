#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for GPU data-dependent ops: gather, repeat_interleave, scatter_add.
//!
//! Cumsum tests extracted to `dyn_tensor_metal_cumsum_tests.rs`.
//! Argmax/argmin tests extracted to `dyn_tensor_metal_argreduce_tests.rs`.
//!
//! Each test validates CPU/GPU parity by running the same operation on both
//! devices and comparing results within tolerance.
//!
//! Issue: #1178

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_gpu_vals, init};

// =============================================================================
// gather tests
// =============================================================================

#[test]
fn test_gpu_gather_dim0() {
    init();
    // data: [[1,2],[3,4],[5,6]] shape [3,2], gather dim=0, indices=[[0],[2]]
    let data = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 2], &[2, 1], &Device::Cpu).unwrap();

    let cpu_data = data.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_data.gather(&ids, 0).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = data.gather(&ids, 0).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-6, "gather_dim0");
}

#[test]
fn test_gpu_gather_dim1() {
    init();
    // data: [[10,20,30],[40,50,60]] shape [2,3], gather dim=1, indices=[[1,0],[2,1]]
    let data = DynTensor::new(
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        &[2, 3],
        &Device::metal(),
    )
    .unwrap();
    let ids = DynTensor::from_vec_u32(vec![1, 0, 2, 1], &[2, 2], &Device::Cpu).unwrap();

    let cpu_data = data.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_data.gather(&ids, 1).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = data.gather(&ids, 1).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-6, "gather_dim1");
}

#[test]
fn test_gpu_gather_single_element() {
    init();
    let data = DynTensor::new(&[42.0], &[1], &Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0], &[1], &Device::Cpu).unwrap();

    let cpu_data = data.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_data.gather(&ids, 0).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = data.gather(&ids, 0).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-6, "gather_single");
}

#[test]
fn test_gpu_gather_preserves_device() {
    init();
    let data = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![2, 0, 1], &[3], &Device::Cpu).unwrap();
    let result = data.gather(&ids, 0).unwrap();
    assert_eq!(
        result.device(),
        Device::metal(),
        "gather must preserve GPU device"
    );
}

/// Regression test: gather with index > 2^24 (16,777,216).
/// IEEE 754 f32 has 24-bit mantissa, so integers > 2^24 lose precision.
/// Index 16,777,217 stored as f32 becomes 16,777,216.0 — wrong element.
/// This test verifies native u32 index buffers preserve precision.
///
/// Issue: #1490
#[test]
fn test_gpu_gather_index_beyond_f32_precision() {
    init();
    // We need a data tensor with > 2^24 elements along the gather dimension.
    // Use 2^24 + 2 = 16,777,218 elements. At 4 bytes each = ~67 MB.
    let n: usize = (1 << 24) + 2; // 16_777_218
    let target_idx: u32 = (1 << 24) + 1; // 16_777_217

    // Create data: zeros everywhere except known sentinel values.
    let mut data = vec![0.0f32; n];
    data[(1 << 24) as usize] = 42.0; // index 16_777_216
    data[target_idx as usize] = 99.0; // index 16_777_217

    let gpu_data = DynTensor::new(&data, &[n], &Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![target_idx], &[1], &Device::Cpu).unwrap();

    let result = gpu_data.gather(&ids, 0).unwrap();
    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // If indices were stored as f32, target_idx (16_777_217) would round to
    // 16_777_216, returning 42.0 instead of the correct 99.0.
    assert_eq!(
        vals[0], 99.0,
        "gather at index 16_777_217 must return 99.0 (not 42.0 from f32 rounding)"
    );
}

/// Regression test: scatter_add with index > 2^24.
/// Same precision concern as gather — u32 indices must not pass through f32.
///
/// Issue: #1490
#[test]
fn test_gpu_scatter_add_index_beyond_f32_precision() {
    init();
    let n: usize = (1 << 24) + 2; // 16_777_218
    let target_idx: u32 = (1 << 24) + 1; // 16_777_217

    let base = DynTensor::zeros(&[n], nn_core::DType::F32, &Device::metal()).unwrap();
    let src = DynTensor::new(&[77.0], &[1], &Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![target_idx], &[1], &Device::Cpu).unwrap();

    let result = base.scatter_add(0, &ids, &src).unwrap();
    let cpu_result = result.to_device(&Device::Cpu).unwrap();
    let vals = cpu_result.to_flat_vec::<f32>().unwrap();

    // If indices were stored as f32, scatter_add would write 77.0 at index
    // 16_777_216 instead of 16_777_217.
    assert_eq!(
        vals[target_idx as usize], 77.0,
        "scatter_add at index 16_777_217 must land at the correct position"
    );
    assert_eq!(
        vals[(1 << 24) as usize],
        0.0,
        "scatter_add must not corrupt adjacent index 16_777_216"
    );
}

// =============================================================================
// repeat_interleave tests
// =============================================================================

/// Helper: create a 1-D f32 DynTensor of repeat counts on CPU.
fn counts_tensor(counts: &[f32]) -> DynTensor {
    DynTensor::new(counts, &[counts.len()], &Device::Cpu).unwrap()
}

#[test]
fn test_gpu_repeat_interleave_1d() {
    init();
    // [10, 20, 30] repeat_interleave(dim=0, counts=[2, 1, 3])
    // => [10, 10, 20, 30, 30, 30]
    let t = DynTensor::new(&[10.0, 20.0, 30.0], &[3], &Device::metal()).unwrap();
    let repeats = counts_tensor(&[2.0, 1.0, 3.0]);

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.repeat_interleave(0, &repeats).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.repeat_interleave(0, &repeats).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-6, "repeat_interleave_1d");
}

#[test]
fn test_gpu_repeat_interleave_2d_dim0() {
    init();
    // [[1,2],[3,4]] repeat_interleave(dim=0, counts=[3, 1])
    // => [[1,2],[1,2],[1,2],[3,4]]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::metal()).unwrap();
    let repeats = counts_tensor(&[3.0, 1.0]);

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.repeat_interleave(0, &repeats).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.repeat_interleave(0, &repeats).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-6, "repeat_interleave_2d_dim0");
    assert_eq!(gpu_result.dims(), &[4, 2]);
}

#[test]
fn test_gpu_repeat_interleave_2d_dim1() {
    init();
    // [[1,2,3],[4,5,6]] repeat_interleave(dim=1, counts=[1, 2, 1])
    // => [[1,2,2,3],[4,5,5,6]]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let repeats = counts_tensor(&[1.0, 2.0, 1.0]);

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.repeat_interleave(1, &repeats).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.repeat_interleave(1, &repeats).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-6, "repeat_interleave_2d_dim1");
    assert_eq!(gpu_result.dims(), &[2, 4]);
}

#[test]
fn test_gpu_repeat_interleave_3d() {
    init();
    // shape [2, 3, 2], repeat_interleave(dim=1, counts=[2, 1, 3])
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 2], &Device::metal()).unwrap();
    let repeats = counts_tensor(&[2.0, 1.0, 3.0]);

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.repeat_interleave(1, &repeats).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.repeat_interleave(1, &repeats).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-6, "repeat_interleave_3d");
    assert_eq!(gpu_result.dims(), &[2, 6, 2]);
}

#[test]
fn test_gpu_repeat_interleave_preserves_device() {
    init();
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::metal()).unwrap();
    let repeats = counts_tensor(&[1.0, 1.0, 1.0]);
    let result = t.repeat_interleave(0, &repeats).unwrap();
    assert_eq!(
        result.device(),
        Device::metal(),
        "repeat_interleave must preserve GPU device"
    );
}

#[test]
fn test_gpu_repeat_interleave_single_element() {
    init();
    let t = DynTensor::new(&[42.0], &[1], &Device::metal()).unwrap();
    let repeats = counts_tensor(&[5.0]);
    let result = t.repeat_interleave(0, &repeats).unwrap();
    assert_gpu_vals(
        &result,
        &[42.0, 42.0, 42.0, 42.0, 42.0],
        1e-6,
        "repeat_single",
    );
    assert_eq!(result.dims(), &[5]);
}

// =============================================================================
// repeat_interleave GPU-native path tests (counts on GPU)
//
// These tests put the counts tensor on Device::metal() to trigger the
// gpu_repeat_interleave_from_gpu path (Blelloch prefix sum + scatter).
// Verifies CPU/GPU parity for the full GPU-native pipeline. (#2616)
// =============================================================================

/// Helper: create a 1-D f32 DynTensor of repeat counts on GPU.
fn gpu_counts_tensor(counts: &[f32]) -> DynTensor {
    DynTensor::new(counts, &[counts.len()], &Device::metal()).unwrap()
}

#[test]
fn test_gpu_repeat_interleave_gpu_counts_1d() {
    init();
    // [10, 20, 30] repeat_interleave(dim=0, counts=[2, 1, 3]) => [10, 10, 20, 30, 30, 30]
    let t = DynTensor::new(&[10.0, 20.0, 30.0], &[3], &Device::metal()).unwrap();
    let repeats = gpu_counts_tensor(&[2.0, 1.0, 3.0]);

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_repeats = repeats.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.repeat_interleave(0, &cpu_repeats).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.repeat_interleave(0, &repeats).unwrap();
    assert_gpu_vals(
        &gpu_result,
        &cpu_vals,
        1e-6,
        "repeat_interleave_gpu_counts_1d",
    );
}

#[test]
fn test_gpu_repeat_interleave_gpu_counts_2d_dim0() {
    init();
    // [[1,2],[3,4]] repeat_interleave(dim=0, counts=[3, 1])
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::metal()).unwrap();
    let repeats = gpu_counts_tensor(&[3.0, 1.0]);

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_repeats = repeats.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.repeat_interleave(0, &cpu_repeats).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.repeat_interleave(0, &repeats).unwrap();
    assert_gpu_vals(
        &gpu_result,
        &cpu_vals,
        1e-6,
        "repeat_interleave_gpu_counts_2d_dim0",
    );
    assert_eq!(gpu_result.dims(), &[4, 2]);
}

#[test]
fn test_gpu_repeat_interleave_gpu_counts_2d_dim1() {
    init();
    // [[1,2,3],[4,5,6]] repeat_interleave(dim=1, counts=[1, 2, 1])
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let repeats = gpu_counts_tensor(&[1.0, 2.0, 1.0]);

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_repeats = repeats.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.repeat_interleave(1, &cpu_repeats).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.repeat_interleave(1, &repeats).unwrap();
    assert_gpu_vals(
        &gpu_result,
        &cpu_vals,
        1e-6,
        "repeat_interleave_gpu_counts_2d_dim1",
    );
    assert_eq!(gpu_result.dims(), &[2, 4]);
}

#[test]
fn test_gpu_repeat_interleave_gpu_counts_3d() {
    init();
    // shape [2, 3, 2], repeat_interleave(dim=1, counts=[2, 1, 3])
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 2], &Device::metal()).unwrap();
    let repeats = gpu_counts_tensor(&[2.0, 1.0, 3.0]);

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_repeats = repeats.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.repeat_interleave(1, &cpu_repeats).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.repeat_interleave(1, &repeats).unwrap();
    assert_gpu_vals(
        &gpu_result,
        &cpu_vals,
        1e-6,
        "repeat_interleave_gpu_counts_3d",
    );
    assert_eq!(gpu_result.dims(), &[2, 6, 2]);
}

#[test]
fn test_gpu_repeat_interleave_gpu_counts_with_zeros() {
    init();
    // Some counts are zero: [a, b, c] with counts=[0, 3, 0] => [b, b, b]
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::metal()).unwrap();
    let repeats = gpu_counts_tensor(&[0.0, 3.0, 0.0]);

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_repeats = repeats.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.repeat_interleave(0, &cpu_repeats).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.repeat_interleave(0, &repeats).unwrap();
    assert_gpu_vals(
        &gpu_result,
        &cpu_vals,
        1e-6,
        "repeat_interleave_gpu_counts_zeros",
    );
    assert_eq!(gpu_result.dims(), &[3]);
}

#[test]
fn test_gpu_repeat_interleave_gpu_counts_preserves_device() {
    init();
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::metal()).unwrap();
    let repeats = gpu_counts_tensor(&[1.0, 1.0, 1.0]);
    let result = t.repeat_interleave(0, &repeats).unwrap();
    assert_eq!(
        result.device(),
        Device::metal(),
        "GPU-native repeat_interleave must preserve GPU device"
    );
}

#[test]
fn test_gpu_repeat_interleave_gpu_counts_single_element() {
    init();
    let t = DynTensor::new(&[42.0], &[1], &Device::metal()).unwrap();
    let repeats = gpu_counts_tensor(&[5.0]);
    let result = t.repeat_interleave(0, &repeats).unwrap();
    assert_gpu_vals(
        &result,
        &[42.0, 42.0, 42.0, 42.0, 42.0],
        1e-6,
        "repeat_interleave_gpu_single",
    );
    assert_eq!(result.dims(), &[5]);
}

#[test]
fn test_gpu_prefix_sum_submit_sync_reads_total_without_flush() {
    init();
    crate::flush().unwrap();
    crate::reset_counters();

    let counts = gpu_counts_tensor(&[2.0, 1.0, 3.0]);
    let offsets_buf = crate::dyn_tensor_metal::dispatch_prefix_sum_only(&counts, 3).unwrap();

    let stats_after_dispatch = crate::dispatch_stats();
    assert_eq!(stats_after_dispatch.flushes, 0, "dispatch should stay lazy");
    assert_eq!(
        stats_after_dispatch.submits, 0,
        "dispatch should not submit"
    );

    crate::submit().unwrap();
    let stats_after_submit = crate::dispatch_stats();
    assert_eq!(
        stats_after_submit.flushes, 0,
        "submit path must not increment flushes"
    );
    assert_eq!(
        stats_after_submit.submits, 1,
        "submit path should record exactly one submit"
    );

    crate::sync().unwrap();
    let total = crate::dyn_tensor_metal::read_prefix_sum_total(&offsets_buf, 3).unwrap();
    assert_eq!(total, 6, "prefix sum total should match repeat count sum");
}

// =============================================================================
// repeat_interleave GPU counts — edge cases
// =============================================================================

/// GPU-native repeat_interleave silently treats NaN counts as 0 (via MSL
/// `floor(NaN)` → 0 clamp), while the CPU path returns an error.
///
/// This test documents the CPU/GPU behavioral divergence for NaN counts.
/// The CPU path (`repeat_interleave_validate_counts`) checks `!v.is_finite()`
/// and returns `Err`. The GPU-native path (`ri_prefix_sum_u32` MSL kernel)
/// converts `floor(NaN)` → NaN, then `NaN > 0.0f` → false, so `val = 0u`.
///
/// Finding: #2218 strategic audit (Prover P10). Kokoro duration predictor
/// outputs pass through this path — NaN in durations would silently produce
/// zero-length output segments instead of propagating an error.
#[test]
fn test_gpu_repeat_interleave_gpu_counts_nan_divergence() {
    init();
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::metal()).unwrap();

    // CPU path: NaN count should return error
    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let nan_counts_cpu = DynTensor::new(&[1.0, f32::NAN, 1.0], &[3], &Device::Cpu).unwrap();
    let cpu_result = cpu_t.repeat_interleave(0, &nan_counts_cpu);
    assert!(
        cpu_result.is_err(),
        "CPU repeat_interleave must reject NaN counts"
    );

    // GPU-native path now validates counts and rejects NaN (matches CPU).
    // Fixed in #2218 F1: gpu_repeat_interleave_from_gpu validates counts
    // before GPU dispatch.
    let nan_counts_gpu = DynTensor::new(&[1.0, f32::NAN, 1.0], &[3], &Device::metal()).unwrap();
    let gpu_result = t.repeat_interleave(0, &nan_counts_gpu);
    assert!(
        gpu_result.is_err(),
        "GPU repeat_interleave must reject NaN counts (matches CPU)"
    );
}

/// GPU-native repeat_interleave now rejects negative counts (matches CPU).
/// Fixed in #2218 F1.
#[test]
fn test_gpu_repeat_interleave_gpu_counts_negative_divergence() {
    init();
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::metal()).unwrap();

    // CPU path: negative count should return error
    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let neg_counts_cpu = DynTensor::new(&[1.0, -1.0, 1.0], &[3], &Device::Cpu).unwrap();
    let cpu_result = cpu_t.repeat_interleave(0, &neg_counts_cpu);
    assert!(
        cpu_result.is_err(),
        "CPU repeat_interleave must reject negative counts"
    );

    // GPU-native path now validates counts and rejects negatives (matches CPU).
    let neg_counts_gpu = DynTensor::new(&[1.0, -1.0, 1.0], &[3], &Device::metal()).unwrap();
    let gpu_result = t.repeat_interleave(0, &neg_counts_gpu);
    assert!(
        gpu_result.is_err(),
        "GPU repeat_interleave must reject negative counts (matches CPU)"
    );
}

// =============================================================================
// scatter_add tests
// =============================================================================

#[test]
fn test_gpu_scatter_add_basic() {
    init();
    // base: [0, 0, 0, 0, 0] shape [5]
    // scatter_add(dim=0, index=[1, 3, 1], src=[10, 20, 30])
    // => [0, 40, 0, 20, 0]  (index 1 gets 10+30=40, index 3 gets 20)
    let base = DynTensor::zeros(&[5], nn_core::DType::F32, &Device::metal()).unwrap();
    let src = DynTensor::new(&[10.0, 20.0, 30.0], &[3], &Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![1, 3, 1], &[3], &Device::Cpu).unwrap();

    let cpu_base = base.to_device(&Device::Cpu).unwrap();
    let cpu_src = src.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_base.scatter_add(0, &ids, &cpu_src).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = base.scatter_add(0, &ids, &src).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-5, "scatter_add_basic");
}

#[test]
fn test_gpu_scatter_add_2d_dim0() {
    init();
    // base: [[0,0],[0,0],[0,0]] shape [3,2]
    // src:  [[1,2],[3,4]] shape [2,2]
    // index: [[2,0],[1,2]] (U32)
    // Result: row 0 gets src[0][1]=2 at col 1; row 1 gets src[1][0]=3 at col 0;
    //         row 2 gets src[0][0]=1 at col 0 and src[1][1]=4 at col 1
    let base = DynTensor::zeros(&[3, 2], nn_core::DType::F32, &Device::metal()).unwrap();
    let src = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![2, 0, 1, 2], &[2, 2], &Device::Cpu).unwrap();

    let cpu_base = base.to_device(&Device::Cpu).unwrap();
    let cpu_src = src.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_base.scatter_add(0, &ids, &cpu_src).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = base.scatter_add(0, &ids, &src).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-5, "scatter_add_2d_dim0");
}

#[test]
fn test_gpu_scatter_add_2d_dim1() {
    init();
    // base: [[0,0,0],[0,0,0]] shape [2,3]
    // src:  [[10,20],[30,40]] shape [2,2]
    // index: [[0,2],[1,0]] (scatter along dim=1)
    let base = DynTensor::zeros(&[2, 3], nn_core::DType::F32, &Device::metal()).unwrap();
    let src = DynTensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2], &Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 2, 1, 0], &[2, 2], &Device::Cpu).unwrap();

    let cpu_base = base.to_device(&Device::Cpu).unwrap();
    let cpu_src = src.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_base.scatter_add(1, &ids, &cpu_src).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = base.scatter_add(1, &ids, &src).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-5, "scatter_add_2d_dim1");
}

#[test]
fn test_gpu_scatter_add_accumulate() {
    init();
    // Test that multiple sources accumulate correctly at the same index.
    // base: [100, 200, 300] shape [3]
    // src:  [1, 2, 3, 4] shape [4]
    // index: [0, 0, 0, 2] — index 0 gets 1+2+3=6 added, index 2 gets 4
    // Result: [106, 200, 304]
    let base = DynTensor::new(&[100.0, 200.0, 300.0], &[3], &Device::metal()).unwrap();
    let src = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[4], &Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 0, 0, 2], &[4], &Device::Cpu).unwrap();

    let cpu_base = base.to_device(&Device::Cpu).unwrap();
    let cpu_src = src.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_base.scatter_add(0, &ids, &cpu_src).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = base.scatter_add(0, &ids, &src).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-5, "scatter_add_accumulate");
}

#[test]
fn test_gpu_scatter_add_3d() {
    init();
    // shape: base [2, 4, 3], src [2, 2, 3], scatter along dim=1
    let base_data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let base = DynTensor::new(&base_data, &[2, 4, 3], &Device::metal()).unwrap();
    let src_data: Vec<f32> = (1..=12).map(|x| x as f32 * 10.0).collect();
    let src = DynTensor::new(&src_data, &[2, 2, 3], &Device::metal()).unwrap();
    // scatter indices: each element maps to a row in the dim=1 axis
    let ids = DynTensor::from_vec_u32(
        vec![0, 0, 0, 3, 3, 3, 1, 1, 1, 2, 2, 2],
        &[2, 2, 3],
        &Device::Cpu,
    )
    .unwrap();

    let cpu_base = base.to_device(&Device::Cpu).unwrap();
    let cpu_src = src.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_base.scatter_add(1, &ids, &cpu_src).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = base.scatter_add(1, &ids, &src).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-5, "scatter_add_3d");
    assert_eq!(gpu_result.dims(), &[2, 4, 3]);
}

#[test]
fn test_gpu_scatter_add_preserves_device() {
    init();
    let base = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::metal()).unwrap();
    let src = DynTensor::new(&[10.0, 20.0, 30.0], &[3], &Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![2, 0, 1], &[3], &Device::Cpu).unwrap();
    let result = base.scatter_add(0, &ids, &src).unwrap();
    assert_eq!(
        result.device(),
        Device::metal(),
        "scatter_add must preserve GPU device"
    );
}

/// GPU scatter_add with an OOB index must return an error, not silently skip.
///
/// Before #1597, the MSL kernel silently skipped writes with OOB indices,
/// producing wrong results. Host-side pre-validation now catches this.
#[test]
fn test_gpu_scatter_add_oob_returns_error() {
    init();
    let base = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::metal()).unwrap();
    let src = DynTensor::new(&[10.0, 20.0, 30.0], &[3], &Device::metal()).unwrap();
    // Index 3 is out of bounds for a dim-0 size of 3 (valid: 0..2)
    let ids = DynTensor::from_vec_u32(vec![0, 1, 3], &[3], &Device::Cpu).unwrap();
    let result = base.scatter_add(0, &ids, &src);

    assert!(result.is_err(), "OOB scatter_add must return Err");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("out of bounds"),
        "error must mention 'out of bounds', got: {msg}"
    );
}

/// GPU scatter_add with max valid index must still succeed.
#[test]
fn test_gpu_scatter_add_max_valid_index() {
    init();
    let base = DynTensor::new(&[0.0, 0.0, 0.0], &[3], &Device::metal()).unwrap();
    let src = DynTensor::new(&[10.0, 20.0], &[2], &Device::metal()).unwrap();
    // Index 2 is the last valid position
    let ids = DynTensor::from_vec_u32(vec![0, 2], &[2], &Device::Cpu).unwrap();
    let result = base.scatter_add(0, &ids, &src).unwrap();
    assert_gpu_vals(&result, &[10.0, 0.0, 20.0], 1e-6, "scatter_add_max_valid");
}

/// Regression test for #2028: scatter_add with empty src on a GPU-produced tensor.
///
/// Before the fix, the empty-src path called `clone_buffer_range` without
/// `flush()`, reading stale/zeroed data when the input was produced by a
/// prior lazy-batched GPU op that hadn't been committed yet.
#[test]
fn test_gpu_scatter_add_empty_src_after_gpu_op() {
    init();
    // Create a GPU tensor via a GPU arithmetic op (add), producing a tensor
    // whose data lives in a lazy-batch buffer.
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[4], &Device::metal()).unwrap();
    let b = DynTensor::new(&[10.0, 20.0, 30.0, 40.0], &[4], &Device::metal()).unwrap();
    let gpu_result = (&a + &b).unwrap(); // [11, 22, 33, 44] — produced by GPU dispatch

    // scatter_add with empty src should return a clone of gpu_result, not stale zeros.
    // Empty tensors use CPU device — Metal does not allow zero-size buffer creation.
    let empty_src = DynTensor::zeros(&[0], nn_core::DType::F32, &Device::Cpu).unwrap();
    let empty_idx = DynTensor::from_vec_u32(vec![], &[0], &Device::Cpu).unwrap();
    let cloned = gpu_result.scatter_add(0, &empty_idx, &empty_src).unwrap();

    assert_gpu_vals(
        &cloned,
        &[11.0, 22.0, 33.0, 44.0],
        1e-6,
        "scatter_add_empty_src_flush",
    );
}

/// Regression test for #2028: index_add with empty src on a GPU-produced tensor.
#[test]
fn test_gpu_index_add_empty_src_after_gpu_op() {
    init();
    let a = DynTensor::new(&[5.0, 6.0, 7.0], &[1, 3], &Device::metal()).unwrap();
    let b = DynTensor::new(&[0.5, 0.5, 0.5], &[1, 3], &Device::metal()).unwrap();
    let gpu_result = (&a * &b).unwrap(); // [2.5, 3.0, 3.5] — produced by GPU dispatch

    // Empty tensors use CPU device — Metal does not allow zero-size buffer creation.
    let empty_src = DynTensor::zeros(&[0, 3], nn_core::DType::F32, &Device::Cpu).unwrap();
    let empty_idx = DynTensor::from_vec_u32(vec![], &[0], &Device::Cpu).unwrap();
    let cloned = gpu_result.index_add(0, &empty_idx, &empty_src).unwrap();

    assert_gpu_vals(&cloned, &[2.5, 3.0, 3.5], 1e-6, "index_add_empty_src_flush");
}
