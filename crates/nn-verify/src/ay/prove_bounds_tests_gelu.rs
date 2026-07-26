// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for gelu_output_bounds: error paths (#394) and numerical correctness (#425).
//! Part of #639.

use super::*;

// ======================== gelu_output_bounds error paths ========================

#[test]
fn test_gelu_bounds_nan_x_lower_returns_error() {
    assert_bounds_error!(
        gelu_output_bounds(f64::NAN, 1.0),
        "non-finite output bounds"
    );
}

#[test]
fn test_gelu_bounds_inf_x_upper_returns_error() {
    assert_bounds_error!(
        gelu_output_bounds(-1.0, f64::INFINITY),
        "non-finite output bounds"
    );
}

#[test]
fn test_gelu_bounds_neg_inf_x_lower_returns_error() {
    assert_bounds_error!(
        gelu_output_bounds(f64::NEG_INFINITY, 1.0),
        "non-finite output bounds"
    );
}

#[test]
fn test_gelu_inverted_bounds_rejected() {
    assert_bounds_error!(gelu_output_bounds(5.0, -5.0), "inverted bounds");
}

// ======================== gelu_output_bounds correctness ========================

/// GELU(0) = 0 exactly.
#[test]
fn test_gelu_bounds_at_zero() {
    let (lo, hi) = gelu_output_bounds(0.0, 0.0).expect("finite inputs");
    assert!(
        (lo - 0.0).abs() < 1e-10,
        "gelu(0) lower should be 0, got {lo}"
    );
    assert!(
        (hi - 0.0).abs() < 1e-10,
        "gelu(0) upper should be 0, got {hi}"
    );
}

/// For large positive x, gelu(x) ≈ x.
#[test]
fn test_gelu_bounds_positive_range() {
    let (lo, hi) = gelu_output_bounds(1.0, 3.0).expect("finite inputs");
    assert!(lo > 0.8 && lo < 0.9, "gelu(1.0) ≈ 0.841, got lo={lo}");
    assert!(hi > 2.99 && hi < 3.01, "gelu(3.0) ≈ 2.996, got hi={hi}");
}

/// Range spanning the GELU minimum at x ≈ -0.752.
#[test]
fn test_gelu_bounds_spanning_minimum() {
    let (lo, hi) = gelu_output_bounds(-2.0, 2.0).expect("finite inputs");
    assert!(
        lo < -0.16 && lo > -0.18,
        "gelu minimum ≈ -0.170, got lo={lo}"
    );
    assert!(hi > 1.95 && hi < 1.96, "gelu(2.0) ≈ 1.955, got hi={hi}");
}

/// Negative range not spanning the minimum.
#[test]
fn test_gelu_bounds_negative_range() {
    let (lo, hi) = gelu_output_bounds(-5.0, -3.0).expect("finite inputs");
    assert!(lo < 0.0, "lower should be negative, got lo={lo}");
    assert!(
        hi > -0.01,
        "upper ≈ 0 for deeply negative input, got hi={hi}"
    );
}

/// Point range: lo == hi should return gelu(x) for both bounds.
#[test]
fn test_gelu_bounds_point_range() {
    let (lo, hi) = gelu_output_bounds(1.0, 1.0).expect("finite inputs");
    assert!((lo - hi).abs() < 1e-10, "point range should give lo == hi");
    assert!(lo > 0.84 && lo < 0.85, "gelu(1.0) ≈ 0.841, got {lo}");
}
