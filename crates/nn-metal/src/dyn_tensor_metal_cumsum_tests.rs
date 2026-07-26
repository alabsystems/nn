#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for GPU cumsum (prefix scan) dispatch.
//!
//! Extracted from `dyn_tensor_metal_data_ops_tests.rs` for file-size compliance.
//! Issue: #1178

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_gpu_vals, init};

/// Assert GPU tensor values match expected with combined absolute + relative
/// tolerance (like numpy's `allclose`).
///
/// Checks `|gpu - cpu| <= atol + rtol * |cpu|` per element. This prevents
/// overly loose absolute tolerances on large-magnitude cumsum outputs while
/// still catching bugs on small-magnitude early elements.
fn assert_gpu_vals_allclose(t: &DynTensor, expected: &[f32], atol: f32, rtol: f32, label: &str) {
    assert_eq!(t.device(), Device::metal(), "{label}: must stay on GPU");
    let vals = t
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals.len(), expected.len(), "{label}: length mismatch");
    for (i, (&g, &e)) in vals.iter().zip(expected.iter()).enumerate() {
        let tol = atol + rtol * e.abs();
        assert!(
            (g - e).abs() <= tol,
            "{label}[{i}]: gpu={g}, cpu={e}, diff={}, tol={tol} (atol={atol}, rtol={rtol})",
            (g - e).abs()
        );
    }
}

// =============================================================================
// cumsum single-pass tests (axis <= 256)
// =============================================================================

#[test]
fn test_gpu_cumsum_1d() {
    init();
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[4], &Device::metal()).unwrap();

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.cumsum(0).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.cumsum(0).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-4, "cumsum_1d");
}

#[test]
fn test_gpu_cumsum_2d_dim0() {
    init();
    // [[1,2],[3,4],[5,6]] cumsum(dim=0) => [[1,2],[4,6],[9,12]]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &Device::metal()).unwrap();

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.cumsum(0).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.cumsum(0).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-4, "cumsum_2d_dim0");
}

#[test]
fn test_gpu_cumsum_2d_dim1() {
    init();
    // [[1,2,3],[4,5,6]] cumsum(dim=1) => [[1,3,6],[4,9,15]]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::metal()).unwrap();

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.cumsum(1).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.cumsum(1).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-4, "cumsum_2d_dim1");
}

#[test]
fn test_gpu_cumsum_preserves_device() {
    init();
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::metal()).unwrap();
    let result = t.cumsum(0).unwrap();
    assert_eq!(
        result.device(),
        Device::metal(),
        "cumsum must preserve GPU device"
    );
}

#[test]
fn test_gpu_cumsum_single_element() {
    init();
    let t = DynTensor::new(&[7.0], &[1], &Device::metal()).unwrap();
    let result = t.cumsum(0).unwrap();
    assert_gpu_vals(&result, &[7.0], 1e-6, "cumsum_single");
}

// -- Boundary condition tests (P1-140 algorithm audit) ------------------------

#[test]
fn test_gpu_cumsum_256_exact_boundary() {
    init();
    // Exactly 256 elements: the maximum single-pass case.
    // If the single-pass/multi-pass decision boundary is off-by-one
    // (e.g., `< 256` instead of `<= 256`), this test would fail because
    // 256 would incorrectly use the multi-pass path.
    let data: Vec<f32> = (1..=256).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[256], &Device::metal()).unwrap();

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.cumsum(0).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.cumsum(0).unwrap();
    // Last element: sum(1..=256) = 32896. f32 can represent this exactly.
    assert_gpu_vals_allclose(&gpu_result, &cpu_vals, 1e-4, 1e-5, "cumsum_256_boundary");
}

#[test]
fn test_gpu_cumsum_all_zeros() {
    init();
    // All-zeros cumsum should produce all-zeros. Exercises Blelloch scan
    // initialization: shared memory is zero-filled for padding threads.
    // If initialization is wrong, zeros input is the only way to catch it
    // (non-zero data masks initialization bugs).
    let data = vec![0.0_f32; 300];
    let t = DynTensor::new(&data, &[300], &Device::metal()).unwrap();
    let gpu_result = t.cumsum(0).unwrap();
    let expected = vec![0.0_f32; 300];
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "cumsum_all_zeros");
}

#[test]
fn test_gpu_cumsum_negative_values() {
    init();
    // Mixed positive/negative values exercise cancellation paths.
    let data: Vec<f32> = (0..300)
        .map(|x| if x % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let t = DynTensor::new(&data, &[300], &Device::metal()).unwrap();

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.cumsum(0).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.cumsum(0).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-4, "cumsum_negative");
}

// =============================================================================
// cumsum multi-pass tests (axis > 256)
// =============================================================================

#[test]
fn test_gpu_cumsum_512() {
    init();
    // 512-element axis — requires multi-pass (2 blocks of 256)
    let data: Vec<f32> = (1..=512).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[512], &Device::metal()).unwrap();

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.cumsum(0).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.cumsum(0).unwrap();
    // Multi-pass cumsum: use relative tolerance to catch bugs at both small
    // and large magnitudes. atol=1e-4 catches small-value errors; rtol=1e-5
    // scales with magnitude for large accumulated values.
    assert_gpu_vals_allclose(&gpu_result, &cpu_vals, 1e-4, 1e-5, "cumsum_512");
}

#[test]
fn test_gpu_cumsum_1000() {
    init();
    // 1000-element axis — 4 blocks (3 full + 1 partial)
    let data: Vec<f32> = (1..=1000).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[1000], &Device::metal()).unwrap();

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.cumsum(0).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.cumsum(0).unwrap();
    // Multi-pass cumsum: last element ~500500. Previous absolute tolerance of
    // 1.0 allowed 100% error on small values. Use relative tolerance instead.
    assert_gpu_vals_allclose(&gpu_result, &cpu_vals, 1e-4, 1e-5, "cumsum_1000");
}

#[test]
fn test_gpu_cumsum_2d_large_dim1() {
    init();
    // [4, 512]: cumsum along dim=1 (large axis, multi-pass)
    let data: Vec<f32> = (1..=2048).map(|x| x as f32 * 0.01).collect();
    let t = DynTensor::new(&data, &[4, 512], &Device::metal()).unwrap();

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.cumsum(1).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.cumsum(1).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-2, "cumsum_2d_large_dim1");
}

#[test]
fn test_gpu_cumsum_2d_large_dim0() {
    init();
    // [512, 4]: cumsum along dim=0 (large axis, multi-pass, non-last-axis)
    let data: Vec<f32> = (1..=2048).map(|x| x as f32 * 0.01).collect();
    let t = DynTensor::new(&data, &[512, 4], &Device::metal()).unwrap();

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.cumsum(0).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.cumsum(0).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-2, "cumsum_2d_large_dim0");
}

#[test]
fn test_gpu_cumsum_257() {
    init();
    // 257 elements — minimal multi-pass case (2 blocks: 256 + 1)
    let data: Vec<f32> = (1..=257).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[257], &Device::metal()).unwrap();

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.cumsum(0).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.cumsum(0).unwrap();
    assert_gpu_vals_allclose(&gpu_result, &cpu_vals, 1e-4, 1e-5, "cumsum_257");
}

#[test]
fn test_gpu_cumsum_4096() {
    init();
    // 4096 elements — larger multi-pass (16 blocks)
    let data: Vec<f32> = (0..4096).map(|x| (x as f32 + 1.0) * 0.001).collect();
    let t = DynTensor::new(&data, &[4096], &Device::metal()).unwrap();

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.cumsum(0).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.cumsum(0).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-2, "cumsum_4096");
}

#[test]
fn test_gpu_cumsum_3d_large_axis() {
    init();
    // [2, 300, 3]: cumsum along dim=1 (300 elements, 2 blocks, multi-pass)
    let data: Vec<f32> = (0..1800).map(|x| (x as f32 + 1.0) * 0.01).collect();
    let t = DynTensor::new(&data, &[2, 300, 3], &Device::metal()).unwrap();

    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.cumsum(1).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.cumsum(1).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-2, "cumsum_3d_large_dim1");
}

// =============================================================================
// Kahan-compensated cumsum tests (#2909)
// =============================================================================

/// Kahan GPU cumsum matches CPU f64 cumsum for SineGen-sized inputs.
///
/// Shape [1, 126, 9] with values in [0, 0.5] — the exact shape and value range
/// used by SineGen's frame-rate rad_frames tensor. Acceptance criterion:
/// atol=1e-4 (Kahan O(nε) ≈ 126*2^-24 ≈ 7.5e-6 per element, well within 1e-4).
#[test]
fn test_gpu_cumsum_kahan_sinegen_shape() {
    init();
    let n = 126 * 9;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.003) % 0.5).collect();
    let gpu_t = DynTensor::new(&data, &[1, 126, 9], &Device::metal()).unwrap();

    let cpu_t = gpu_t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.cumsum_kahan(1).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = gpu_t.cumsum_kahan(1).unwrap();
    assert_gpu_vals_allclose(&gpu_result, &cpu_vals, 1e-4, 1e-5, "kahan_sinegen");
}

#[test]
fn test_gpu_cumsum_kahan_preserves_device() {
    init();
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::metal()).unwrap();
    let result = t.cumsum_kahan(0).unwrap();
    assert_eq!(
        result.device(),
        Device::metal(),
        "cumsum_kahan must preserve GPU device"
    );
}

#[test]
fn test_gpu_cumsum_kahan_1d() {
    init();
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[4], &Device::metal()).unwrap();
    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.cumsum_kahan(0).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.cumsum_kahan(0).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-6, "kahan_1d");
}

#[test]
fn test_gpu_cumsum_kahan_2d_dim1() {
    init();
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let cpu_t = t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.cumsum_kahan(1).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = t.cumsum_kahan(1).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-6, "kahan_2d_dim1");
}

/// Kahan should be more precise than naive f32 for cancellation-heavy inputs.
#[test]
fn test_gpu_cumsum_kahan_precision() {
    init();
    // Alternating +1/-1 stress-tests cancellation. After 200 elements, naive
    // f32 can drift; Kahan should stay within 1e-5 of f64 reference.
    let data: Vec<f32> = (0..200)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let gpu_t = DynTensor::new(&data, &[1, 200, 1], &Device::metal()).unwrap();
    let cpu_t = gpu_t.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_t.cumsum_kahan(1).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = gpu_t.cumsum_kahan(1).unwrap();
    assert_gpu_vals_allclose(&gpu_result, &cpu_vals, 1e-5, 0.0, "kahan_precision");
}
