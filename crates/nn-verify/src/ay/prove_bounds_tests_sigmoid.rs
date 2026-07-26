// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for sigmoid_output_bounds: error paths (#394) and numerical correctness (#425).
//! Part of #659 AC1.

use super::*;

// ======================== sigmoid_output_bounds error paths ========================

#[test]
fn test_sigmoid_bounds_nan_x_lower_returns_error() {
    assert_bounds_error!(
        sigmoid_output_bounds(f64::NAN, 1.0),
        "non-finite output bounds"
    );
}

#[test]
fn test_sigmoid_bounds_inf_x_upper_returns_error() {
    assert_bounds_error!(
        sigmoid_output_bounds(-1.0, f64::INFINITY),
        "non-finite output bounds"
    );
}

#[test]
fn test_sigmoid_bounds_neg_inf_x_lower_returns_error() {
    assert_bounds_error!(
        sigmoid_output_bounds(f64::NEG_INFINITY, 1.0),
        "non-finite output bounds"
    );
}

#[test]
fn test_sigmoid_inverted_bounds_rejected() {
    assert_bounds_error!(sigmoid_output_bounds(5.0, -5.0), "inverted bounds");
}

// ======================== sigmoid_output_bounds correctness ========================

/// sigmoid(0) = 0.5 exactly.
#[test]
fn test_sigmoid_bounds_at_zero() {
    let (lo, hi) = sigmoid_output_bounds(0.0, 0.0).expect("finite inputs");
    assert!(
        (lo - 0.5).abs() < 1e-10,
        "sigmoid(0) lower should be 0.5, got {lo}"
    );
    assert!(
        (hi - 0.5).abs() < 1e-10,
        "sigmoid(0) upper should be 0.5, got {hi}"
    );
}

/// For positive range: sigmoid(1) ≈ 0.7311, sigmoid(3) ≈ 0.9526.
#[test]
fn test_sigmoid_bounds_positive_range() {
    let (lo, hi) = sigmoid_output_bounds(1.0, 3.0).expect("finite inputs");
    assert!(lo > 0.73 && lo < 0.74, "sigmoid(1.0) ≈ 0.7311, got lo={lo}");
    assert!(hi > 0.95 && hi < 0.96, "sigmoid(3.0) ≈ 0.9526, got hi={hi}");
}

/// Symmetric range: sigmoid(-x) + sigmoid(x) = 1.
#[test]
fn test_sigmoid_bounds_symmetric_range() {
    let (lo, hi) = sigmoid_output_bounds(-5.0, 5.0).expect("finite inputs");
    assert!(
        lo > 0.006 && lo < 0.007,
        "sigmoid(-5) ≈ 0.00669, got lo={lo}"
    );
    assert!(
        hi > 0.993 && hi < 0.994,
        "sigmoid(5) ≈ 0.99331, got hi={hi}"
    );
    assert!(
        ((lo + hi) - 1.0).abs() < 1e-6,
        "sigmoid(-x) + sigmoid(x) should equal 1.0, got {lo} + {hi} = {}",
        lo + hi
    );
}

/// Negative range: sigmoid(-5) ≈ 0.00669, sigmoid(-1) ≈ 0.2689.
#[test]
fn test_sigmoid_bounds_negative_range() {
    let (lo, hi) = sigmoid_output_bounds(-5.0, -1.0).expect("finite inputs");
    assert!(
        lo > 0.006 && lo < 0.007,
        "sigmoid(-5) ≈ 0.00669, got lo={lo}"
    );
    assert!(
        hi > 0.268 && hi < 0.270,
        "sigmoid(-1) ≈ 0.2689, got hi={hi}"
    );
}

/// Point range: lo == hi should return sigmoid(x) for both bounds.
#[test]
fn test_sigmoid_bounds_point_range() {
    let (lo, hi) = sigmoid_output_bounds(1.0, 1.0).expect("finite inputs");
    assert!((lo - hi).abs() < 1e-10, "point range should give lo == hi");
    assert!(lo > 0.731 && lo < 0.732, "sigmoid(1.0) ≈ 0.7311, got {lo}");
}

/// Wide range: sigmoid is always in (0, 1).
#[test]
fn test_sigmoid_bounds_wide_range_within_01() {
    let (lo, hi) = sigmoid_output_bounds(-100.0, 100.0).expect("finite inputs");
    assert!(lo > 0.0, "sigmoid output must be > 0");
    assert!(hi < 1.0, "sigmoid output must be strictly < 1.0");
    assert!(
        hi >= 1.0 - 1e-10,
        "sigmoid(100) bound should be very close to 1.0, got {hi}"
    );
    assert!(
        lo < 1e-40,
        "sigmoid(-100) should be extremely close to 0, got {lo}"
    );
}

/// Even wider range: sigmoid(±1000) must still satisfy (0, 1) invariant.
#[test]
fn test_sigmoid_bounds_extreme_range_within_01() {
    let (lo, hi) = sigmoid_output_bounds(-1000.0, 1000.0).expect("finite inputs");
    assert!(lo > 0.0, "sigmoid(-1000) must be > 0, got {lo}");
    assert!(hi < 1.0, "sigmoid(1000) must be < 1, got {hi}");
}
