// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for shared utility functions in `util.rs`.

use super::*;
use ny_api::BoundedTensor;
use ndarray::{ArrayD, IxDyn};

// --- finite_or tests ---

#[test]
fn test_finite_or_returns_finite_value() {
    assert_eq!(finite_or(1.5, 0.0), 1.5);
    assert_eq!(finite_or(-3.0, 0.0), -3.0);
    assert_eq!(finite_or(0.0, 99.0), 0.0);
}

#[test]
fn test_finite_or_replaces_nan() {
    assert_eq!(finite_or(f32::NAN, 0.0), 0.0);
    assert_eq!(finite_or(f32::NAN, -1.0), -1.0);
}

#[test]
fn test_finite_or_replaces_infinity() {
    assert_eq!(finite_or(f32::INFINITY, 0.0), 0.0);
    assert_eq!(finite_or(f32::NEG_INFINITY, 0.0), 0.0);
}

#[test]
fn test_finite_or_boundary_values() {
    assert_eq!(finite_or(f32::MAX, 0.0), f32::MAX);
    assert_eq!(finite_or(f32::MIN, 0.0), f32::MIN);
    assert_eq!(finite_or(f32::MIN_POSITIVE, 0.0), f32::MIN_POSITIVE);
    // Negative zero is finite
    assert_eq!(finite_or(-0.0, 1.0), -0.0);
}

// --- sanitize_tensor_bounds tests ---

#[test]
fn test_sanitize_tensor_bounds_all_finite() {
    let input = [1.0, -2.0, 3.5, 0.0];
    let result = sanitize_tensor_bounds(&input);
    assert_eq!(result, vec![1.0, -2.0, 3.5, 0.0]);
}

#[test]
fn test_sanitize_tensor_bounds_replaces_non_finite() {
    let input = [1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -2.0];
    let result = sanitize_tensor_bounds(&input);
    assert_eq!(result, vec![1.0, 0.0, 0.0, 0.0, -2.0]);
}

#[test]
fn test_sanitize_tensor_bounds_empty() {
    let result = sanitize_tensor_bounds(&[]);
    assert!(result.is_empty());
}

// --- get_value tests ---

#[test]
fn test_get_value_valid_index() {
    let values = vec![10, 20, 30];
    assert_eq!(*get_value(&values, 0, "test").unwrap(), 10);
    assert_eq!(*get_value(&values, 2, "test").unwrap(), 30);
}

#[test]
fn test_get_value_out_of_bounds() {
    let values = vec![10, 20];
    let err = get_value(&values, 5, "test_ctx").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("test_ctx") && msg.contains("5") && msg.contains("2"),
        "error should contain context, index, and length, got: {msg}"
    );
}

#[test]
fn test_get_value_empty_slice() {
    let values: Vec<i32> = vec![];
    let err = get_value(&values, 0, "empty").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("0") && msg.contains("len 0"),
        "error should mention len 0, got: {msg}"
    );
}

// --- bounds_min_max tests ---

#[test]
fn test_bounds_min_max_single_element() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap();
    let bt = BoundedTensor::new(lower, upper).unwrap();
    let (lo, hi) = bounds_min_max(&bt);
    assert_eq!(lo, -1.0);
    assert_eq!(hi, 1.0);
}

#[test]
fn test_bounds_min_max_multiple_elements() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-5.0, 0.0, 2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 10.0, 3.0]).unwrap();
    let bt = BoundedTensor::new(lower, upper).unwrap();
    let (lo, hi) = bounds_min_max(&bt);
    assert_eq!(lo, -5.0);
    assert_eq!(hi, 10.0);
}

#[test]
fn test_bounds_min_max_conservative_bounds() {
    // BoundedTensor::new_conservative creates [-inf, +inf] bounds
    let bt = BoundedTensor::new_conservative(&[2]);
    let (lo, hi) = bounds_min_max(&bt);
    // Infinity should propagate through the fold
    assert_eq!(lo, f32::NEG_INFINITY);
    assert_eq!(hi, f32::INFINITY);
}
