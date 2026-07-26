// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for relu_output_bounds: error paths (#394) and numerical correctness (#425).
//! Part of #857 AC1+AC2.

use super::*;

// ======================== relu_output_bounds error paths ========================

#[test]
fn test_relu_bounds_nan_x_lower_returns_error() {
    assert_bounds_error!(
        relu_output_bounds(f64::NAN, 1.0),
        "non-finite output bounds"
    );
}

#[test]
fn test_relu_bounds_inf_x_upper_returns_error() {
    assert_bounds_error!(
        relu_output_bounds(-1.0, f64::INFINITY),
        "non-finite output bounds"
    );
}

#[test]
fn test_relu_bounds_neg_inf_x_lower_returns_error() {
    assert_bounds_error!(
        relu_output_bounds(f64::NEG_INFINITY, 1.0),
        "non-finite output bounds"
    );
}

#[test]
fn test_relu_inverted_bounds_rejected() {
    assert_bounds_error!(relu_output_bounds(5.0, -5.0), "inverted bounds");
}

// ======================== relu_output_bounds correctness ========================

/// relu(0) = 0 exactly.
#[test]
fn test_relu_bounds_at_zero() {
    let (lo, hi) = relu_output_bounds(0.0, 0.0).expect("finite inputs");
    assert!(lo.abs() < 1e-15, "relu(0) lower should be 0.0, got {lo}");
    assert!(hi.abs() < 1e-15, "relu(0) upper should be 0.0, got {hi}");
}

/// Positive range: relu passes through unchanged.
#[test]
fn test_relu_bounds_positive_range() {
    let (lo, hi) = relu_output_bounds(2.0, 7.0).expect("finite inputs");
    assert!(
        (lo - 2.0).abs() < 1e-15,
        "relu lower should be 2.0, got {lo}"
    );
    assert!(
        (hi - 7.0).abs() < 1e-15,
        "relu upper should be 7.0, got {hi}"
    );
}

/// Negative range: relu clamps both to 0.
#[test]
fn test_relu_bounds_negative_range() {
    let (lo, hi) = relu_output_bounds(-5.0, -1.0).expect("finite inputs");
    assert!(lo.abs() < 1e-15, "relu(-5) lower should be 0.0, got {lo}");
    assert!(hi.abs() < 1e-15, "relu(-1) upper should be 0.0, got {hi}");
}

/// Zero-crossing range: lower clamps to 0, upper passes through.
#[test]
fn test_relu_bounds_zero_crossing() {
    let (lo, hi) = relu_output_bounds(-3.0, 5.0).expect("finite inputs");
    assert!(
        lo.abs() < 1e-15,
        "relu lower should be 0.0 for negative input, got {lo}"
    );
    assert!(
        (hi - 5.0).abs() < 1e-15,
        "relu upper should be 5.0, got {hi}"
    );
}

/// Point range: lo == hi.
#[test]
fn test_relu_bounds_point_range_positive() {
    let (lo, hi) = relu_output_bounds(3.0, 3.0).expect("finite inputs");
    assert!((lo - hi).abs() < 1e-15, "point range should give lo == hi");
    assert!(
        (lo - 3.0).abs() < 1e-15,
        "relu(3.0) should be 3.0, got {lo}"
    );
}

/// Point range at negative value: both clamp to 0.
#[test]
fn test_relu_bounds_point_range_negative() {
    let (lo, hi) = relu_output_bounds(-2.0, -2.0).expect("finite inputs");
    assert!(lo.abs() < 1e-15, "relu(-2) should be 0.0, got {lo}");
    assert!(hi.abs() < 1e-15, "relu(-2) should be 0.0, got {hi}");
}

/// Wide range: output lower is non-negative, upper matches input upper.
#[test]
fn test_relu_bounds_wide_range() {
    let (lo, hi) = relu_output_bounds(-1000.0, 1000.0).expect("finite inputs");
    assert!(lo >= 0.0, "relu output lower must be >= 0, got {lo}");
    assert!(
        (hi - 1000.0).abs() < 1e-10,
        "relu upper should be 1000.0, got {hi}"
    );
}

/// Invariant: relu output lower is always >= 0.
#[test]
fn test_relu_bounds_output_always_nonnegative() {
    for x_lo in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0] {
        for x_hi in [x_lo, x_lo + 1.0, x_lo + 50.0, x_lo + 200.0] {
            if x_hi < x_lo {
                continue;
            }
            let (lo, hi) = relu_output_bounds(x_lo, x_hi).expect("finite inputs");
            assert!(
                lo >= 0.0,
                "relu output lo must be >= 0 for [{x_lo}, {x_hi}], got {lo}"
            );
            assert!(
                hi >= lo,
                "relu output hi must be >= lo for [{x_lo}, {x_hi}], got lo={lo}, hi={hi}"
            );
        }
    }
}
