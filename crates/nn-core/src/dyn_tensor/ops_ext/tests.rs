#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for extended DynTensor operations: non-keepdim reductions, softmax,
//! log-softmax, argmax/argmin, clamp variants, flatten.
//!
//! Additional tests (cumsum, repeat_interleave, powf, topk, NaN boundary)
//! are in `dyn_tensor_ops_ext_tests_extra.rs`.

use crate::dyn_tensor::test_helpers::{assert_close, cpu, tnd};
use crate::dyn_tensor::{DynTensor, D};

// -- Non-keepdim reductions ---------------------------------------------------

#[test]
fn test_sum_removes_dim() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let s = x.sum(1).expect("sum");
    assert_eq!(s.dims(), &[2]);
    assert_eq!(s.to_flat_vec::<f32>().unwrap(), vec![6.0, 15.0]);
}

#[test]
fn test_sum_dim0() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let s = x.sum(0).expect("sum");
    assert_eq!(s.dims(), &[3]);
    assert_eq!(s.to_flat_vec::<f32>().unwrap(), vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_mean_removes_dim() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let m = x.mean(1).expect("mean");
    assert_eq!(m.dims(), &[2]);
    assert_close(&m.to_flat_vec::<f32>().unwrap(), &[2.0, 5.0], 1e-6);
}

#[test]
fn test_max_removes_dim() {
    let x = tnd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let m = x.max(1).expect("max");
    assert_eq!(m.dims(), &[2]);
    assert_eq!(m.to_flat_vec::<f32>().unwrap(), vec![5.0, 6.0]);
}

#[test]
fn test_min_removes_dim() {
    let x = tnd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let m = x.min(1).expect("min");
    assert_eq!(m.dims(), &[2]);
    assert_eq!(m.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn test_sum_3d() {
    let x = tnd(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 2, 3],
    );
    let s = x.sum(1).expect("sum");
    assert_eq!(s.dims(), &[2, 3]);
    assert_eq!(
        s.to_flat_vec::<f32>().unwrap(),
        vec![5.0, 7.0, 9.0, 17.0, 19.0, 21.0]
    );
}

// -- Argmax / Argmin ----------------------------------------------------------

#[test]
fn test_argmax_basic() {
    let x = tnd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let idx = x.argmax(1).expect("argmax");
    assert_eq!(idx.dims(), &[2]);
    assert_eq!(idx.dtype(), crate::DType::U32);
    assert_eq!(idx.to_vec1::<u32>().unwrap(), vec![1, 2]);
}

#[test]
fn test_argmin_basic() {
    let x = tnd(&[3.0, 1.0, 5.0, 6.0, 4.0, 2.0], &[2, 3]);
    let idx = x.argmin(1).expect("argmin");
    assert_eq!(idx.dims(), &[2]);
    assert_eq!(idx.dtype(), crate::DType::U32);
    assert_eq!(idx.to_vec1::<u32>().unwrap(), vec![1, 2]);
}

#[test]
fn test_argmax_ties_returns_first() {
    // When multiple elements share the max, argmax returns the lowest index.
    let x = tnd(&[5.0, 5.0, 5.0, 1.0, 3.0, 3.0], &[2, 3]);
    let idx = x.argmax(1).expect("argmax ties");
    assert_eq!(idx.dims(), &[2]);
    // Row 0: [5,5,5] all tied → 0; Row 1: [1,3,3] max=3 first at 1
    assert_eq!(idx.to_vec1::<u32>().unwrap(), vec![0, 1]);
}

#[test]
fn test_argmin_ties_returns_first() {
    let x = tnd(&[2.0, 2.0, 5.0, 3.0, 1.0, 1.0], &[2, 3]);
    let idx = x.argmin(1).expect("argmin ties");
    assert_eq!(idx.dims(), &[2]);
    // Row 0: [2,2,5] min=2 first at 0; Row 1: [3,1,1] min=1 first at 1
    assert_eq!(idx.to_vec1::<u32>().unwrap(), vec![0, 1]);
}

#[test]
fn test_argmax_dim0() {
    let x = tnd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let idx = x.argmax(0).expect("argmax dim 0");
    assert_eq!(idx.dims(), &[3]);
    assert_eq!(idx.to_vec1::<u32>().unwrap(), vec![1, 0, 1]);
}

// -- Clamp variants -----------------------------------------------------------

#[test]
fn test_clamp_min() {
    let x = tnd(&[-1.0, 0.5, 2.0, -3.0], &[4]);
    let y = x.clamp_min(0.0).expect("clamp_min");
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![0.0, 0.5, 2.0, 0.0]);
}

#[test]
fn test_clamp_max() {
    let x = tnd(&[-1.0, 0.5, 2.0, 3.0], &[4]);
    let y = x.clamp_max(1.0).expect("clamp_max");
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![-1.0, 0.5, 1.0, 1.0]);
}

// -- Generalized softmax ------------------------------------------------------

#[test]
fn test_softmax_last_dim_matches() {
    let x = tnd(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
    let s1 = crate::dyn_tensor::softmax_last_dim(&x).expect("softmax_last_dim");
    let s2 = x.softmax(1).expect("softmax(1)");
    assert_close(
        &s1.to_flat_vec::<f32>().unwrap(),
        &s2.to_flat_vec::<f32>().unwrap(),
        1e-6,
    );
}

#[test]
fn test_softmax_dim0() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let s = x.softmax(0).expect("softmax dim 0");
    assert_eq!(s.dims(), &[2, 3]);
    let data = s.to_flat_vec::<f32>().unwrap();
    for col in 0..3 {
        let sum = data[col] + data[col + 3];
        assert!((sum - 1.0).abs() < 1e-5, "column {col} sum = {sum}");
    }
}

#[test]
fn test_softmax_sums_to_one() {
    let x = tnd(&[0.5, 1.5, 2.5], &[3]);
    let s = x.softmax(0).expect("softmax");
    let sum: f32 = s.to_flat_vec::<f32>().unwrap().iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "softmax sum = {sum}");
}

#[test]
fn test_softmax_uniform_produces_uniform() {
    // Equal inputs should produce equal probabilities.
    let x = tnd(&[1.0, 1.0, 1.0, 1.0], &[4]);
    let s = x.softmax(0).expect("softmax");
    let data = s.to_flat_vec::<f32>().unwrap();
    for &v in &data {
        assert!((v - 0.25).abs() < 1e-6, "expected 0.25, got {v}");
    }
}

// -- Log-softmax --------------------------------------------------------------

#[test]
fn test_log_softmax_large_values_stable() {
    // Numerically stable: large values shouldn't overflow.
    // exp(100) overflows f32 alone, but max-subtraction handles it.
    let x = tnd(&[100.0, 200.0, 300.0], &[3]);
    let ls = x.log_softmax(0).expect("log_softmax");
    let data = ls.to_flat_vec::<f32>().unwrap();
    // All log_softmax values must be <= 0 (log of probability)
    for &v in &data {
        assert!(v <= 0.0, "log_softmax should be <= 0, got {v}");
        assert!(v.is_finite(), "log_softmax should be finite, got {v}");
    }
    // exp(log_softmax) must sum to ~1
    let sum: f32 = data.iter().map(|v| v.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-5, "exp(log_softmax) sum = {sum}");
}

#[test]
fn test_log_softmax_basic() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    let ls = x.log_softmax(0).expect("log_softmax");
    let s = x.softmax(0).expect("softmax");
    let softmax_data = s.to_flat_vec::<f32>().unwrap();
    let log_softmax_data = ls.to_flat_vec::<f32>().unwrap();
    let expected: Vec<f32> = softmax_data.iter().map(|v| v.ln()).collect();
    assert_close(&log_softmax_data, &expected, 1e-5);
}

#[test]
fn test_log_softmax_sums_property() {
    let x = tnd(&[0.1, 0.5, 0.9, 0.2, 0.6, 1.0], &[2, 3]);
    let ls = x.log_softmax(1).expect("log_softmax");
    let data = ls.to_flat_vec::<f32>().unwrap();
    for row in data.chunks(3) {
        let sum: f32 = row.iter().map(|v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5, "exp(log_softmax) sum = {sum}");
    }
}

// -- Flatten ------------------------------------------------------------------

#[test]
fn test_flatten_full_range() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let y = x.flatten(0, 1).expect("flatten");
    assert_eq!(y.dims(), &[6]);
}

#[test]
fn test_flatten_partial() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let x = tnd(&data, &[2, 3, 4]);
    let y = x.flatten(1, 2).expect("flatten");
    assert_eq!(y.dims(), &[2, 12]);
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_flatten_single_dim_is_identity() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let y = x.flatten(0, 0).expect("flatten same dim");
    assert_eq!(y.dims(), &[2, 3]);
}

#[test]
fn test_flatten_4d() {
    let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
    let x = tnd(&data, &[2, 3, 4, 5]);
    let y = x.flatten(1, 3).expect("flatten");
    assert_eq!(y.dims(), &[2, 60]);
}

// -- Error cases --------------------------------------------------------------

#[test]
fn test_sum_invalid_dim() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    assert!(x.sum(1).is_err());
}

#[test]
fn test_argmax_invalid_dim() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    assert!(x.argmax(2).is_err());
}

#[test]
fn test_softmax_invalid_dim() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    assert!(x.softmax(1).is_err());
}

#[test]
fn test_flatten_invalid_range() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert!(x.flatten(1, 0).is_err());
    assert!(x.flatten(0, 2).is_err());
}

// -- arg_sort -----------------------------------------------------------------

fn u32_vec(tensor: &DynTensor) -> Vec<u32> {
    tensor.as_cpu_u32().unwrap().iter().copied().collect()
}

#[test]
fn test_arg_sort_ascending() {
    // [3, 1, 2] → ascending indices [1, 2, 0]
    let x = tnd(&[3.0, 1.0, 2.0], &[3]);
    let result = x.arg_sort(D::Minus1, true).unwrap();
    assert_eq!(result.dims(), &[3]);
    assert_eq!(u32_vec(&result), vec![1, 2, 0]);
}

#[test]
fn test_arg_sort_descending() {
    // [3, 1, 2] → descending indices [0, 2, 1]
    let x = tnd(&[3.0, 1.0, 2.0], &[3]);
    let result = x.arg_sort(D::Minus1, false).unwrap();
    assert_eq!(result.dims(), &[3]);
    assert_eq!(u32_vec(&result), vec![0, 2, 1]);
}

#[test]
fn test_arg_sort_2d() {
    // [[3, 1, 2], [6, 4, 5]] — each row sorted independently
    let x = tnd(&[3.0, 1.0, 2.0, 6.0, 4.0, 5.0], &[2, 3]);
    let result = x.arg_sort(D::Minus1, false).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    // Row 0: [3,1,2] desc → [0,2,1]
    // Row 1: [6,4,5] desc → [0,2,1]
    assert_eq!(u32_vec(&result), vec![0, 2, 1, 0, 2, 1]);
}

#[test]
fn test_arg_sort_nan_rejects() {
    let x = tnd(&[1.0, f32::NAN, 3.0], &[3]);
    assert!(x.arg_sort(D::Minus1, true).is_err());
}

#[test]
fn test_arg_sort_single_element() {
    let x = tnd(&[42.0], &[1]);
    let result = x.arg_sort(D::Minus1, true).unwrap();
    assert_eq!(result.dims(), &[1]);
    assert_eq!(u32_vec(&result), vec![0]);
}

#[test]
fn test_arg_sort_inverse_roundtrip() {
    // Sorting descending then arg_sort on the result gives the inverse perm.
    // This is the dvoice pattern: inv_indices = sorted_indices.arg_sort(D::Minus1, true)
    let x = tnd(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]);
    let sorted_idx = x.arg_sort(D::Minus1, false).unwrap();
    let s = u32_vec(&sorted_idx);
    // Convert U32 indices to F32 to apply arg_sort for inverse.
    let sorted_f32 = DynTensor::from_vec(
        s.iter().map(|&v| v as f32).collect(),
        sorted_idx.dims(),
        &cpu(),
    )
    .unwrap();
    let inv_idx = sorted_f32.arg_sort(D::Minus1, true).unwrap();
    let inv = u32_vec(&inv_idx);
    // Verify: for each position i, inv[sorted_idx[i]] == i
    for i in 0..5 {
        assert_eq!(inv[s[i] as usize], i as u32, "roundtrip at position {i}");
    }
}

#[test]
fn test_arg_sort_infinity_sorts_correctly() {
    // +Inf and -Inf are valid floats with well-defined ordering.
    // -Inf < all finite < +Inf.
    let x = tnd(&[1.0, f32::INFINITY, -1.0, f32::NEG_INFINITY, 0.0], &[5]);
    let result = x.arg_sort(D::Minus1, true).unwrap();
    // Expected ascending order: -Inf(3), -1.0(2), 0.0(4), 1.0(0), +Inf(1)
    assert_eq!(u32_vec(&result), vec![3, 2, 4, 0, 1]);
}

#[test]
fn test_arg_sort_equal_elements() {
    // Equal elements: sort must produce valid permutation indices
    let x = tnd(&[5.0, 5.0, 5.0], &[3]);
    let result = x.arg_sort(D::Minus1, true).unwrap();
    let indices = u32_vec(&result);
    // All indices must appear exactly once (valid permutation)
    let mut sorted_indices = indices;
    sorted_indices.sort_unstable();
    assert_eq!(sorted_indices, vec![0, 1, 2]);
}

#[test]
fn test_arg_sort_empty_last_dim() {
    // Shape [2, 0] — zero-length last dimension. Should either return
    // a valid empty tensor or a meaningful error, not panic.
    let x = DynTensor::from_vec(vec![], &[2, 0], &cpu()).unwrap();
    // chunks_exact(0) panics — this test catches the bug.
    let result = x.arg_sort(D::Minus1, true);
    // Accept either a valid empty-shaped result or an error, but not a panic.
    if let Ok(t) = result {
        assert_eq!(t.dims(), &[2, 0]);
    }
}
