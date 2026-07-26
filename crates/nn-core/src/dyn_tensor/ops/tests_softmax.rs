#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Softmax and NaN/Inf edge-case tests for DynTensor ops.
//! Extracted from dyn_tensor_ops_tests.rs to keep it under 500 lines.

use super::*;
use crate::dyn_tensor::test_helpers::{approx_eq, cpu, t1d, t2d};
use crate::DynTensor;

// -- Softmax ------------------------------------------------------------------

#[test]
fn test_softmax_1d() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let s = softmax_last_dim(&a).unwrap();
    let v = s.to_vec1::<f32>().unwrap();
    let sum: f32 = v.iter().sum();
    assert!(approx_eq(sum, 1.0, 1e-5));
    assert!(v[0] < v[1] && v[1] < v[2]); // monotonic
}

#[test]
fn test_softmax_2d() {
    // Softmax should be independent per row
    let a = t2d(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], 2, 3);
    let s = softmax_last_dim(&a).unwrap();
    assert_eq!(s.dims(), &[2, 3]);
    let flat = s.to_flat_vec::<f32>().unwrap();
    // Rows should be identical and sum to 1
    assert!(approx_eq(flat[0] + flat[1] + flat[2], 1.0, 1e-5));
    assert!(approx_eq(flat[3] + flat[4] + flat[5], 1.0, 1e-5));
    assert!(approx_eq(flat[0], flat[3], 1e-6));
}

#[test]
fn test_softmax_last_dim_nan_returns_error() {
    let a = DynTensor::from_vec(vec![1.0, f32::NAN, 3.0], &[1, 3], &cpu()).unwrap();
    let result = softmax_last_dim(&a);
    assert!(
        result.is_err(),
        "softmax_last_dim with NaN input should return Err"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Non-finite"),
        "error should mention Non-finite: {err}"
    );
}

/// softmax_last_dim allows -inf (needed for attention masks).
#[test]
fn test_softmax_last_dim_neg_inf_allowed() {
    // Mixed -inf and finite: -inf positions become 0, finite positions get probability.
    let a = DynTensor::from_vec(vec![f32::NEG_INFINITY, 1.0, 2.0], &[1, 3], &cpu()).unwrap();
    let result = softmax_last_dim(&a);
    assert!(result.is_ok(), "softmax_last_dim should accept -inf input");
    let vals = result.unwrap().to_flat_vec::<f32>().unwrap();
    // First element (masked with -inf) should be ~0.0
    assert!(
        vals[0] < 1e-6,
        "masked position should be near zero: {}",
        vals[0]
    );
    // Remaining should sum to ~1.0
    let sum: f32 = vals.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1: {sum}");
}

// -- NaN/Inf edge-case tests --------------------------------------------------

/// mean_all on an empty tensor returns an error (not NaN via 0.0/0.0).
#[test]
fn test_mean_all_empty_tensor_returns_error() {
    let a = DynTensor::from_vec(vec![], &[0], &cpu()).unwrap();
    let result = a.mean_all();
    assert!(
        result.is_err(),
        "mean_all on empty tensor should return Err"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("non-empty"),
        "error should mention non-empty: {err}"
    );
}

/// softmax on a tensor where last dim is 0 returns an error.
#[test]
fn test_softmax_empty_last_dim_returns_error() {
    let a = DynTensor::from_vec(vec![], &[1, 0], &cpu()).unwrap();
    let result = a.softmax(1);
    assert!(result.is_err(), "softmax on empty dim should return Err");
}

/// softmax where all values are -inf produces zeros (#1310).
/// -inf is intentionally allowed because attention masks use -inf for masked positions.
/// All-neg-inf is degenerate: guard zeros the lane instead of NaN from -inf - (-inf).
#[test]
fn test_softmax_all_neg_inf_produces_zeros() {
    let a = DynTensor::from_vec(
        vec![f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
        &[1, 3],
        &cpu(),
    )
    .unwrap();
    let s = a.softmax(1).unwrap();
    let vals = s.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| *v == 0.0),
        "all-neg-inf softmax should produce zeros, got: {vals:?}"
    );
}

/// softmax with mixed all-neg-inf and normal rows (#1310).
/// Only the all-neg-inf row should be zeroed; the normal row is unaffected.
#[test]
fn test_softmax_mixed_neg_inf_and_normal_rows() {
    let a = DynTensor::from_vec(
        vec![
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY, // row 0: all -inf
            1.0,
            2.0,
            3.0, // row 1: normal
        ],
        &[2, 3],
        &cpu(),
    )
    .unwrap();
    let s = a.softmax(1).unwrap();
    let vals = s.to_flat_vec::<f32>().unwrap();
    // Row 0: all zeros
    assert!(
        vals[0] == 0.0 && vals[1] == 0.0 && vals[2] == 0.0,
        "all-neg-inf row should be zeros: {:?}",
        &vals[..3]
    );
    // Row 1: valid probability distribution
    let row1_sum: f32 = vals[3..6].iter().sum();
    assert!(
        (row1_sum - 1.0).abs() < 1e-5,
        "normal row should sum to 1: {row1_sum}"
    );
    assert!(vals[3] < vals[4] && vals[4] < vals[5]); // monotonic
}

/// softmax with NaN input returns NonFiniteData error.
#[test]
fn test_softmax_nan_input_returns_error() {
    let a = DynTensor::from_vec(vec![1.0, f32::NAN, 3.0], &[1, 3], &cpu()).unwrap();
    let result = a.softmax(1);
    assert!(result.is_err(), "softmax with NaN input should return Err");
    let err = result.unwrap_err().to_string();
    assert!(err.contains("NaN"), "error should mention NaN: {err}");
}

// -- +Inf edge-case tests -----------------------------------------------------

/// softmax with +inf input: the +inf position dominates with probability 1.
/// Previously produced silent all-NaN output (IEEE 754: inf - inf = NaN).
#[test]
fn test_softmax_pos_inf_gives_one_at_inf_position() {
    let a = DynTensor::from_vec(vec![1.0, f32::INFINITY, 3.0], &[1, 3], &cpu()).unwrap();
    let s = a.softmax(1).unwrap();
    let vals = s.to_flat_vec::<f32>().unwrap();
    // +inf position should get probability 1.0
    assert!(
        approx_eq(vals[1], 1.0, 1e-6),
        "+inf position should have prob 1.0, got: {}",
        vals[1]
    );
    // Finite positions should get probability 0.0
    assert!(
        vals[0] == 0.0 && vals[2] == 0.0,
        "finite positions should be 0: [{}, {}]",
        vals[0],
        vals[2]
    );
    // Sum should be 1.0
    let sum: f32 = vals.iter().sum();
    assert!(approx_eq(sum, 1.0, 1e-6), "softmax should sum to 1: {sum}");
}

/// softmax with multiple +inf values: probability shared equally.
#[test]
fn test_softmax_multiple_pos_inf_shared_equally() {
    let a = DynTensor::from_vec(vec![f32::INFINITY, 1.0, f32::INFINITY], &[1, 3], &cpu()).unwrap();
    let s = a.softmax(1).unwrap();
    let vals = s.to_flat_vec::<f32>().unwrap();
    assert!(
        approx_eq(vals[0], 0.5, 1e-6),
        "first +inf should get 0.5, got: {}",
        vals[0]
    );
    assert!(
        vals[1] == 0.0,
        "finite position should be 0, got: {}",
        vals[1]
    );
    assert!(
        approx_eq(vals[2], 0.5, 1e-6),
        "second +inf should get 0.5, got: {}",
        vals[2]
    );
}

/// softmax with all +inf: uniform distribution (1/N each).
#[test]
fn test_softmax_all_pos_inf_uniform() {
    let a = DynTensor::from_vec(
        vec![f32::INFINITY, f32::INFINITY, f32::INFINITY],
        &[1, 3],
        &cpu(),
    )
    .unwrap();
    let s = a.softmax(1).unwrap();
    let vals = s.to_flat_vec::<f32>().unwrap();
    for (i, v) in vals.iter().enumerate() {
        assert!(
            approx_eq(*v, 1.0 / 3.0, 1e-6),
            "all-inf softmax should be uniform, position {i} = {v}"
        );
    }
}

/// softmax_last_dim with +inf: same behavior via free function path.
#[test]
fn test_softmax_last_dim_pos_inf_gives_one() {
    let a = DynTensor::from_vec(vec![1.0, f32::INFINITY, 3.0], &[1, 3], &cpu()).unwrap();
    let s = softmax_last_dim(&a).unwrap();
    let vals = s.to_flat_vec::<f32>().unwrap();
    assert!(
        approx_eq(vals[1], 1.0, 1e-6),
        "+inf via softmax_last_dim should get 1.0, got: {}",
        vals[1]
    );
    assert!(vals[0] == 0.0 && vals[2] == 0.0);
}

/// log_softmax with +inf: log(1) = 0 at +inf position, log(0) = -inf elsewhere.
#[test]
fn test_log_softmax_pos_inf_gives_zero_at_inf_position() {
    let a = DynTensor::from_vec(vec![1.0, f32::INFINITY, 3.0], &[1, 3], &cpu()).unwrap();
    let s = a.log_softmax(1).unwrap();
    let vals = s.to_flat_vec::<f32>().unwrap();
    // log(1.0) = 0.0 at the single +inf position
    assert!(
        approx_eq(vals[1], 0.0, 1e-6),
        "log_softmax at +inf should be 0.0, got: {}",
        vals[1]
    );
    // log(0.0) = -inf at finite positions
    assert!(
        vals[0] == f32::NEG_INFINITY,
        "finite position should be -inf, got: {}",
        vals[0]
    );
    assert!(
        vals[2] == f32::NEG_INFINITY,
        "finite position should be -inf, got: {}",
        vals[2]
    );
}

/// log_softmax with multiple +inf: log(1/2) ≈ -0.693 at +inf positions.
#[test]
fn test_log_softmax_multiple_pos_inf() {
    let a = DynTensor::from_vec(vec![f32::INFINITY, 1.0, f32::INFINITY], &[1, 3], &cpu()).unwrap();
    let s = a.log_softmax(1).unwrap();
    let vals = s.to_flat_vec::<f32>().unwrap();
    let expected_log = -(2.0_f32).ln(); // log(1/2) ≈ -0.693
    assert!(
        approx_eq(vals[0], expected_log, 1e-5),
        "+inf log_softmax should be {expected_log}, got: {}",
        vals[0]
    );
    assert!(
        vals[1] == f32::NEG_INFINITY,
        "finite should be -inf, got: {}",
        vals[1]
    );
    assert!(
        approx_eq(vals[2], expected_log, 1e-5),
        "+inf log_softmax should be {expected_log}, got: {}",
        vals[2]
    );
}

// -- Log-softmax all-neg-inf tests (#1326) ------------------------------------

/// log_softmax where all values are -inf produces -inf (#1326).
/// log(softmax([-inf, -inf, ...])) = log([0, 0, ...]) = [-inf, -inf, ...].
#[test]
fn test_log_softmax_all_neg_inf_produces_neg_inf() {
    let a = DynTensor::from_vec(
        vec![f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
        &[1, 3],
        &cpu(),
    )
    .unwrap();
    let s = a.log_softmax(1).unwrap();
    let vals = s.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| *v == f32::NEG_INFINITY),
        "all-neg-inf log_softmax should produce -inf, got: {vals:?}"
    );
}

/// log_softmax with mixed all-neg-inf and normal rows (#1326).
#[test]
fn test_log_softmax_mixed_neg_inf_and_normal_rows() {
    let a = DynTensor::from_vec(
        vec![
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY, // row 0: all -inf
            1.0,
            2.0,
            3.0, // row 1: normal
        ],
        &[2, 3],
        &cpu(),
    )
    .unwrap();
    let s = a.log_softmax(1).unwrap();
    let vals = s.to_flat_vec::<f32>().unwrap();
    // Row 0: all -inf
    assert!(
        vals[0] == f32::NEG_INFINITY
            && vals[1] == f32::NEG_INFINITY
            && vals[2] == f32::NEG_INFINITY,
        "all-neg-inf row should be -inf: {:?}",
        &vals[..3]
    );
    // Row 1: valid log-probabilities (all negative, logsumexp is valid)
    assert!(
        vals[3..6].iter().all(|v| v.is_finite() && *v < 0.0),
        "normal row should have finite negative log-probs: {:?}",
        &vals[3..6]
    );
    // log_softmax values should sum to approximately -inf via exp?
    // Actually, exp(log_softmax) should sum to 1.
    let exp_sum: f32 = vals[3..6].iter().map(|v| v.exp()).sum();
    assert!(
        (exp_sum - 1.0).abs() < 1e-5,
        "exp(log_softmax) should sum to 1: {exp_sum}"
    );
}

/// softmax_last_dim with all-neg-inf produces zeros (not NaN) (#1326).
/// Tests the ops/softmax.rs path specifically.
#[test]
fn test_softmax_last_dim_all_neg_inf_produces_zeros() {
    let a = DynTensor::from_vec(
        vec![
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ],
        &[2, 2],
        &cpu(),
    )
    .unwrap();
    let s = softmax_last_dim(&a).unwrap();
    let vals = s.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| *v == 0.0),
        "all-neg-inf softmax_last_dim should produce zeros, got: {vals:?}"
    );
}
