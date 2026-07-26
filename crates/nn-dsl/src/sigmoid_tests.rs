// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `sigmoid.rs`. Extracted per inline-test-extraction pattern (#175).

use super::*;

#[test]
fn test_sigmoid_scalar_zero() {
    // sigmoid(0) = 0.5
    let result = sigmoid_scalar(0.0).unwrap();
    assert!(
        (result - 0.5).abs() < 1e-6,
        "sigmoid(0) should be 0.5, got {result}"
    );
}

#[test]
fn test_sigmoid_scalar_positive() {
    // sigmoid(1) ≈ 0.7311
    let result = sigmoid_scalar(1.0).unwrap();
    assert!(
        (result - 0.7311).abs() < 0.01,
        "sigmoid(1) should be ~0.731, got {result}"
    );
}

#[test]
fn test_sigmoid_scalar_negative() {
    // sigmoid(-1) ≈ 0.2689
    let result = sigmoid_scalar(-1.0).unwrap();
    assert!(
        (result - 0.2689).abs() < 0.01,
        "sigmoid(-1) should be ~0.269, got {result}"
    );
}

#[test]
fn test_sigmoid_scalar_large_positive() {
    // sigmoid(x) → 1 for large positive x
    let result = sigmoid_scalar(10.0).unwrap();
    assert!(
        (result - 1.0).abs() < 1e-4,
        "sigmoid(10) should be ~1.0, got {result}"
    );
}

#[test]
fn test_sigmoid_scalar_large_negative() {
    // sigmoid(x) → 0 for large negative x
    let result = sigmoid_scalar(-10.0).unwrap();
    assert!(result < 1e-4, "sigmoid(-10) should be ~0.0, got {result}");
}

#[test]
fn test_sigmoid_scalar_symmetry() {
    // sigmoid(-x) = 1 - sigmoid(x)
    let a = sigmoid_scalar(2.0).unwrap();
    let b = sigmoid_scalar(-2.0).unwrap();
    assert!(
        (a + b - 1.0).abs() < 1e-6,
        "sigmoid(x) + sigmoid(-x) should be 1.0, got {a} + {b}"
    );
}

#[test]
fn test_sigmoid_scalar_nan_rejected() {
    assert!(sigmoid_scalar(f32::NAN).is_err());
}

#[test]
fn test_sigmoid_scalar_inf_rejected() {
    assert!(sigmoid_scalar(f32::INFINITY).is_err());
}

#[test]
fn test_sigmoid_bounds_monotone() {
    let (lo, hi) = sigmoid_scalar_bounds(-2.0, 2.0).unwrap();
    let expected_lo = sigmoid_scalar(-2.0).unwrap();
    let expected_hi = sigmoid_scalar(2.0).unwrap();
    assert!((lo - expected_lo).abs() < 1e-5);
    assert!((hi - expected_hi).abs() < 1e-5);
    assert!(lo < hi, "bounds should be ordered: {lo} < {hi}");
}

#[test]
fn test_sigmoid_bounds_positive_range() {
    let (lo, hi) = sigmoid_scalar_bounds(1.0, 3.0).unwrap();
    assert!(lo > 0.5, "sigmoid(1) should be > 0.5, got {lo}");
    assert!(hi < 1.0, "sigmoid(3) should be < 1.0, got {hi}");
    assert!(lo < hi);
}

#[test]
fn test_sigmoid_bounds_nan_rejected() {
    assert!(sigmoid_scalar_bounds(f32::NAN, 1.0).is_err());
}

#[test]
fn test_sigmoid_bounds_inverted_rejected() {
    assert!(sigmoid_scalar_bounds(1.0, -1.0).is_err());
}

#[test]
fn test_sigmoid_ref_basic() {
    let input = vec![0.0, 1.0, -1.0];
    let output = sigmoid_ref(&input).unwrap();
    assert_eq!(output.len(), 3);
    assert!((output[0] - 0.5).abs() < 1e-6);
    assert!((output[1] - 0.7311).abs() < 0.01);
    assert!((output[2] - 0.2689).abs() < 0.01);
}

#[test]
fn test_sigmoid_ref_empty_rejected() {
    assert!(sigmoid_ref(&[]).is_err());
}

#[test]
fn test_build_sigmoid_kernel() {
    let kernel = build_sigmoid_kernel().expect("should build sigmoid kernel");
    assert_eq!(kernel.name, "sigmoid");
    assert_eq!(kernel.params.len(), 1);
}
