#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! BF16/F16 normalization and softmax dtype-preservation tests.
//!
//! Extracted from `tests_bf16_f16_ext.rs` to keep files under 500 lines.
//! Covers: LayerNorm, RmsNorm, GroupNorm, softmax, log_softmax.

use crate::dyn_tensor::test_helpers::cpu;
use crate::{DType, DynTensor};
use half::bf16;
use half::f16;
use ndarray::{ArrayD, IxDyn};

// -- Helpers (duplicated from parent for test module independence) -------------

fn bf16_tensor(data: &[f32], dims: &[usize]) -> DynTensor {
    let arr = ArrayD::from_shape_vec(
        IxDyn(dims),
        data.iter().map(|&v| bf16::from_f32(v)).collect(),
    )
    .unwrap();
    DynTensor::from_cpu_bf16(arr).unwrap()
}

fn f16_tensor(data: &[f32], dims: &[usize]) -> DynTensor {
    let arr = ArrayD::from_shape_vec(
        IxDyn(dims),
        data.iter().map(|&v| f16::from_f32(v)).collect(),
    )
    .unwrap();
    DynTensor::from_cpu_f16(arr).unwrap()
}

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

// -- BF16 normalization ops (#1671, #1672) ------------------------------------
//
// GPU norm ops (layer_norm, rms_norm, group_norm) fall back to CPU for bf16
// tensors because make_eps_buffer creates f32 buffers (#1672) and gpu_slice_set
// hardcodes f32 buffer access (#1671). These tests verify the CPU fallback path
// produces correct results with bf16 inputs.

#[test]
fn test_bf16_layer_norm_preserves_dtype() {
    use crate::layers::{LayerNorm, Module};
    // Input: [1, 4] — normalize over last dim (4 elements)
    let x = bf16_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let weight = bf16_tensor(&[1.0, 1.0, 1.0, 1.0], &[4]);
    let bias = bf16_tensor(&[0.0, 0.0, 0.0, 0.0], &[4]);
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();
    let result = ln.forward(&x).unwrap();
    assert_eq!(
        result.dtype(),
        DType::BF16,
        "LayerNorm should preserve BF16"
    );
    assert_eq!(result.dims(), &[1, 4]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // Mean = 2.5, Std ≈ 1.118. Normalized: [-1.342, -0.447, 0.447, 1.342]
    assert!(approx(vals[0], -1.342, 0.15));
    assert!(approx(vals[3], 1.342, 0.15));
}

#[test]
fn test_bf16_rms_norm_preserves_dtype() {
    use crate::layers::{Module, RmsNorm};
    // Input: [1, 3] — RMS normalize over last dim
    let x = bf16_tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let weight = bf16_tensor(&[1.0, 1.0, 1.0], &[3]);
    let rn = RmsNorm::new(weight, 1e-5).unwrap();
    let result = rn.forward(&x).unwrap();
    assert_eq!(result.dtype(), DType::BF16, "RmsNorm should preserve BF16");
    assert_eq!(result.dims(), &[1, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // RMS = sqrt((1+4+9)/3) ≈ 2.160. Normalized: [0.463, 0.926, 1.389]
    assert!(approx(vals[0], 0.463, 0.1));
    assert!(approx(vals[2], 1.389, 0.1));
}

#[test]
fn test_bf16_group_norm_preserves_dtype() {
    use crate::layers::{GroupNorm, Module};
    // Input: [batch=1, channels=2, length=3], num_groups=2
    let x = bf16_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    let weight = bf16_tensor(&[1.0, 1.0], &[2]);
    let bias = bf16_tensor(&[0.0, 0.0], &[2]);
    let gn = GroupNorm::new(2, 2, weight, bias, 1e-5).unwrap();
    let result = gn.forward(&x).unwrap();
    assert_eq!(
        result.dtype(),
        DType::BF16,
        "GroupNorm should preserve BF16"
    );
    assert_eq!(result.dims(), &[1, 2, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // Group 0 (ch=0): [1,2,3] → mean=2, std≈0.816 → [-1.22, 0, 1.22]
    assert!(approx(vals[0], -1.22, 0.15));
    assert!(approx(vals[1], 0.0, 0.15));
    assert!(approx(vals[2], 1.22, 0.15));
}

// -- BF16/F16 softmax dtype preservation and parity (#1691) --------------------

#[test]
fn test_bf16_softmax_preserves_dtype() {
    let a = bf16_tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let result = a.softmax(1).unwrap();
    assert_eq!(result.dtype(), DType::BF16, "softmax should preserve BF16");
    assert_eq!(result.dims(), &[1, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]
    assert!(approx(vals[0], 0.0900, 0.02));
    assert!(approx(vals[1], 0.2447, 0.02));
    assert!(approx(vals[2], 0.6652, 0.02));
    // Sum should be ~1.0.
    let sum: f32 = vals.iter().sum();
    assert!(approx(sum, 1.0, 0.05));
}

/// Regression test for #1691 AC5: bf16 softmax matches f32 softmax within
/// tolerance. Before the fix, f32::MAX constant overflowed bf16 range (~65504),
/// causing errors or wrong results.
#[test]
fn test_bf16_softmax_matches_f32_softmax() {
    let data = &[1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 0.0, 2.5];
    // Compute f32 reference
    let f32_input = DynTensor::from_vec(data.to_vec(), &[2, 4], &cpu()).unwrap();
    let f32_result = f32_input.softmax(1).unwrap();
    let f32_vals = f32_result.to_flat_vec::<f32>().unwrap();
    // Compute bf16 result
    let bf16_input = bf16_tensor(data, &[2, 4]);
    let bf16_result = bf16_input.softmax(1).unwrap();
    assert_eq!(bf16_result.dtype(), DType::BF16);
    let bf16_vals = bf16_result.to_flat_vec::<f32>().unwrap();
    // Compare element-wise: bf16 has ~3 decimal digits of precision
    for (i, (&f32_v, &bf16_v)) in f32_vals.iter().zip(bf16_vals.iter()).enumerate() {
        assert!(
            approx(bf16_v, f32_v, 0.01),
            "bf16 softmax[{i}] = {bf16_v}, f32 = {f32_v}, diff = {}",
            (bf16_v - f32_v).abs()
        );
    }
    // Both rows must sum to ~1.0
    let bf16_sum_r0: f32 = bf16_vals[..4].iter().sum();
    let bf16_sum_r1: f32 = bf16_vals[4..].iter().sum();
    assert!(approx(bf16_sum_r0, 1.0, 0.02), "row 0 sum: {bf16_sum_r0}");
    assert!(approx(bf16_sum_r1, 1.0, 0.02), "row 1 sum: {bf16_sum_r1}");
}

/// F16 softmax matches f32 softmax within tolerance (#1691).
#[test]
fn test_f16_softmax_matches_f32_softmax() {
    let data = &[1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 0.0, 2.5];
    let f32_input = DynTensor::from_vec(data.to_vec(), &[2, 4], &cpu()).unwrap();
    let f32_result = f32_input.softmax(1).unwrap();
    let f32_vals = f32_result.to_flat_vec::<f32>().unwrap();
    let f16_input = f16_tensor(data, &[2, 4]);
    let f16_result = f16_input.softmax(1).unwrap();
    assert_eq!(f16_result.dtype(), DType::F16);
    let f16_vals = f16_result.to_flat_vec::<f32>().unwrap();
    for (i, (&f32_v, &f16_v)) in f32_vals.iter().zip(f16_vals.iter()).enumerate() {
        assert!(
            approx(f16_v, f32_v, 0.005),
            "f16 softmax[{i}] = {f16_v}, f32 = {f32_v}, diff = {}",
            (f16_v - f32_v).abs()
        );
    }
}

/// BF16 log_softmax matches f32 log_softmax within tolerance (#1691).
#[test]
fn test_bf16_log_softmax_matches_f32_log_softmax() {
    let data = &[1.0, 2.0, 3.0, 0.5, -1.0, 2.5];
    let f32_input = DynTensor::from_vec(data.to_vec(), &[2, 3], &cpu()).unwrap();
    let f32_result = f32_input.log_softmax(1).unwrap();
    let f32_vals = f32_result.to_flat_vec::<f32>().unwrap();
    let bf16_input = bf16_tensor(data, &[2, 3]);
    let bf16_result = bf16_input.log_softmax(1).unwrap();
    assert_eq!(bf16_result.dtype(), DType::BF16);
    let bf16_vals = bf16_result.to_flat_vec::<f32>().unwrap();
    for (i, (&f32_v, &bf16_v)) in f32_vals.iter().zip(bf16_vals.iter()).enumerate() {
        assert!(
            approx(bf16_v, f32_v, 0.05),
            "bf16 log_softmax[{i}] = {bf16_v}, f32 = {f32_v}, diff = {}",
            (bf16_v - f32_v).abs()
        );
    }
}

/// BF16 softmax with large values stays within bf16 range (#1691).
/// Before the fix, f32::MAX constant overflowed bf16 range.
#[test]
fn test_bf16_softmax_large_values_no_overflow() {
    // Values near bf16 max (~65504). Before #1691 fix, clamping to f32::MAX
    // overflowed bf16, producing errors.
    let data = &[100.0, 200.0, 50.0];
    let input = bf16_tensor(data, &[1, 3]);
    let result = input.softmax(1).unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // Largest value (200) should dominate
    assert!(vals[1] > 0.99, "largest value should dominate: {}", vals[1]);
    let sum: f32 = vals.iter().sum();
    assert!(approx(sum, 1.0, 0.02), "sum should be ~1.0: {sum}");
}
