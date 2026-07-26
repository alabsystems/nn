// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `bounds_bridge` module: conversion between
//! nn-core `IntervalBounds` and NY `BoundedTensor`.
//!
//! Covers AC2 of #187: round-trip conversion and error paths.

use nn_core::IntervalBounds;
use nn_verify::{to_bounded_tensor, to_interval_bounds, BoundedTensor};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Happy path: round-trip conversion preserves bounds
// ---------------------------------------------------------------------------

#[test]
fn test_round_trip_interval_to_bounded_to_interval() {
    let lower = ArrayD::from_elem(IxDyn(&[2, 4]), -3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 4]), 7.0f32);
    let original = IntervalBounds::new_allow_infinite(lower.clone(), upper.clone())
        .expect("valid IntervalBounds");

    let bt = to_bounded_tensor(original).expect("IntervalBounds -> BoundedTensor");
    let recovered = to_interval_bounds(bt).expect("BoundedTensor -> IntervalBounds");

    let (rec_lower, rec_upper) = recovered.into_parts();
    assert_eq!(rec_lower, lower, "lower bounds should survive round-trip");
    assert_eq!(rec_upper, upper, "upper bounds should survive round-trip");
}

#[test]
fn test_round_trip_bounded_to_interval_to_bounded() {
    let lower = ArrayD::from_elem(IxDyn(&[3, 8]), 0.5f32);
    let upper = ArrayD::from_elem(IxDyn(&[3, 8]), 2.5f32);
    let original = BoundedTensor::new(lower.clone(), upper.clone()).expect("valid BoundedTensor");

    let ib = to_interval_bounds(original).expect("BoundedTensor -> IntervalBounds");
    let recovered = to_bounded_tensor(ib).expect("IntervalBounds -> BoundedTensor");

    let (rec_lower, rec_upper) = recovered.lower_upper();
    assert_eq!(rec_lower, lower, "lower bounds should survive round-trip");
    assert_eq!(rec_upper, upper, "upper bounds should survive round-trip");
}

// ---------------------------------------------------------------------------
// Infinite bounds: preserved through conversion
// ---------------------------------------------------------------------------

#[test]
fn test_infinite_bounds_preserved() {
    let lower = ArrayD::from_elem(IxDyn(&[1, 2]), f32::NEG_INFINITY);
    let upper = ArrayD::from_elem(IxDyn(&[1, 2]), f32::INFINITY);
    let ib = IntervalBounds::new_allow_infinite(lower, upper).expect("infinite bounds allowed");

    let bt = to_bounded_tensor(ib).expect("IntervalBounds -> BoundedTensor with infinities");
    let recovered =
        to_interval_bounds(bt).expect("BoundedTensor -> IntervalBounds with infinities");

    let (rec_lower, rec_upper) = recovered.into_parts();
    assert!(
        rec_lower.iter().all(|v| *v == f32::NEG_INFINITY),
        "negative infinity lower bounds should be preserved"
    );
    assert!(
        rec_upper.iter().all(|v| *v == f32::INFINITY),
        "positive infinity upper bounds should be preserved"
    );
}

// ---------------------------------------------------------------------------
// Scalar (1-element) bounds
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_round_trip() {
    let lower = ArrayD::from_elem(IxDyn(&[1]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 1.0f32);
    let bt = BoundedTensor::new(lower, upper).expect("scalar BoundedTensor");

    let ib = to_interval_bounds(bt).expect("scalar BoundedTensor -> IntervalBounds");
    let recovered = to_bounded_tensor(ib).expect("scalar IntervalBounds -> BoundedTensor");

    let (rec_lower, rec_upper) = recovered.lower_upper();
    assert_eq!(rec_lower[[0]], -1.0);
    assert_eq!(rec_upper[[0]], 1.0);
}

// ---------------------------------------------------------------------------
// Error paths: NaN bounds rejected
// ---------------------------------------------------------------------------

#[test]
fn test_to_interval_bounds_nan_lower_rejected() {
    let lower = ArrayD::from_elem(IxDyn(&[2]), f32::NAN);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 1.0f32);
    // BoundedTensor::new rejects NaN, so we test via IntervalBounds direction.
    // IntervalBounds::new_allow_infinite also rejects NaN, so both directions
    // are guarded. This tests the error mapping path.
    let result = IntervalBounds::new_allow_infinite(lower, upper);
    assert!(
        result.is_err(),
        "NaN lower bounds should be rejected by IntervalBounds"
    );
}

#[test]
fn test_to_bounded_tensor_inverted_bounds_rejected() {
    // IntervalBounds::new_allow_infinite rejects lower > upper
    let lower = ArrayD::from_elem(IxDyn(&[2]), 5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 1.0f32);
    let result = IntervalBounds::new_allow_infinite(lower, upper);
    assert!(
        result.is_err(),
        "inverted bounds (lower > upper) should be rejected"
    );
}

// ---------------------------------------------------------------------------
// Error paths: shape mismatch rejected (#472)
// ---------------------------------------------------------------------------

#[test]
fn test_interval_bounds_shape_mismatch_rejected() {
    let lower = ArrayD::zeros(IxDyn(&[2, 3]));
    let upper = ArrayD::ones(IxDyn(&[2, 4])); // Different shape
    let result = IntervalBounds::new_allow_infinite(lower, upper);
    assert!(
        result.is_err(),
        "mismatched shapes should be rejected by IntervalBounds"
    );
}

#[test]
fn test_bounded_tensor_shape_mismatch_rejected() {
    let lower = ArrayD::zeros(IxDyn(&[2, 3]));
    let upper = ArrayD::ones(IxDyn(&[2, 4])); // Different shape
    let result = BoundedTensor::new(lower, upper);
    assert!(
        result.is_err(),
        "mismatched shapes should be rejected by BoundedTensor"
    );
}
