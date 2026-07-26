// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for tanh_output_bounds: error paths (#394) and numerical correctness (#425).
//! Part of #857 AC3+AC4.

use super::*;

// ======================== tanh_output_bounds error paths ========================

#[test]
fn test_tanh_bounds_nan_x_lower_returns_error() {
    assert_bounds_error!(
        tanh_output_bounds(f64::NAN, 1.0),
        "non-finite output bounds"
    );
}

#[test]
fn test_tanh_bounds_inf_x_upper_returns_error() {
    assert_bounds_error!(
        tanh_output_bounds(-1.0, f64::INFINITY),
        "non-finite output bounds"
    );
}

#[test]
fn test_tanh_bounds_neg_inf_x_lower_returns_error() {
    assert_bounds_error!(
        tanh_output_bounds(f64::NEG_INFINITY, 1.0),
        "non-finite output bounds"
    );
}

#[test]
fn test_tanh_inverted_bounds_rejected() {
    assert_bounds_error!(tanh_output_bounds(5.0, -5.0), "inverted bounds");
}

// ======================== tanh_output_bounds correctness ========================

/// tanh(0) = 0 exactly.
#[test]
fn test_tanh_bounds_at_zero() {
    let (lo, hi) = tanh_output_bounds(0.0, 0.0).expect("finite inputs");
    assert!(lo.abs() < 1e-15, "tanh(0) lower should be 0.0, got {lo}");
    assert!(hi.abs() < 1e-15, "tanh(0) upper should be 0.0, got {hi}");
}

/// Positive range: tanh(1) ≈ 0.7616, tanh(3) ≈ 0.9951.
#[test]
fn test_tanh_bounds_positive_range() {
    let (lo, hi) = tanh_output_bounds(1.0, 3.0).expect("finite inputs");
    assert!(lo > 0.761 && lo < 0.762, "tanh(1.0) ≈ 0.7616, got lo={lo}");
    assert!(hi > 0.995 && hi < 0.996, "tanh(3.0) ≈ 0.9951, got hi={hi}");
}

/// Symmetric range: tanh(-x) = -tanh(x).
#[test]
fn test_tanh_bounds_symmetric_range() {
    let (lo, hi) = tanh_output_bounds(-5.0, 5.0).expect("finite inputs");
    assert!(lo < -0.999 && lo > -1.0, "tanh(-5) ≈ -0.9999, got lo={lo}");
    assert!(hi > 0.999 && hi < 1.0, "tanh(5) ≈ 0.9999, got hi={hi}");
    assert!(
        (lo + hi).abs() < 1e-10,
        "tanh(-x) + tanh(x) should equal 0.0, got {lo} + {hi} = {}",
        lo + hi
    );
}

/// Negative range: tanh(-3) ≈ -0.9951, tanh(-1) ≈ -0.7616.
#[test]
fn test_tanh_bounds_negative_range() {
    let (lo, hi) = tanh_output_bounds(-3.0, -1.0).expect("finite inputs");
    assert!(
        lo < -0.995 && lo > -0.996,
        "tanh(-3) ≈ -0.9951, got lo={lo}"
    );
    assert!(
        hi < -0.761 && hi > -0.762,
        "tanh(-1) ≈ -0.7616, got hi={hi}"
    );
}

/// Point range: lo == hi should return tanh(x) for both bounds.
#[test]
fn test_tanh_bounds_point_range() {
    let (lo, hi) = tanh_output_bounds(1.0, 1.0).expect("finite inputs");
    assert!((lo - hi).abs() < 1e-15, "point range should give lo == hi");
    assert!(lo > 0.761 && lo < 0.762, "tanh(1.0) ≈ 0.7616, got {lo}");
}

/// Wide range: f64::tanh saturates to ±1.0 for |x| > ~19.
/// In IEEE 754 f64, tanh(±100) == ±1.0 exactly (not open interval).
#[test]
fn test_tanh_bounds_wide_range_saturates() {
    let (lo, hi) = tanh_output_bounds(-100.0, 100.0).expect("finite inputs");
    assert!(lo >= -1.0, "tanh output must be >= -1.0, got {lo}");
    assert!(hi <= 1.0, "tanh output must be <= 1.0, got {hi}");
    // f64::tanh(-100) saturates to exactly -1.0
    assert!(
        (lo - (-1.0)).abs() < 1e-15,
        "tanh(-100) should saturate to -1.0 in f64, got {lo}"
    );
    assert!(
        (hi - 1.0).abs() < 1e-15,
        "tanh(100) should saturate to 1.0 in f64, got {hi}"
    );
}

/// Saturation region: f64::tanh saturates for large inputs.
/// tanh(10) ≈ 0.9999999958776927, tanh(20) = 1.0 (f64 saturation).
#[test]
fn test_tanh_bounds_saturation_region() {
    let (lo, hi) = tanh_output_bounds(10.0, 20.0).expect("finite inputs");
    assert!(lo > 0.0, "tanh(10) must be positive, got {lo}");
    assert!(hi <= 1.0, "tanh(20) must be <= 1.0, got {hi}");
    assert!(
        (hi - 1.0).abs() < 1e-10,
        "tanh(20) should be at or very close to 1.0, got {hi}"
    );
    assert!(
        lo > 1.0 - 1e-7,
        "tanh(10) should be very close to 1.0, got {lo}"
    );
}

/// Extreme range: f64::tanh(±1000) saturates to exactly ±1.0.
/// In IEEE 754 f64, tanh saturates around |x| > 19.
#[test]
fn test_tanh_bounds_extreme_range() {
    let (lo, hi) = tanh_output_bounds(-1000.0, 1000.0).expect("finite inputs");
    assert!(lo >= -1.0, "tanh(-1000) must be >= -1, got {lo}");
    assert!(hi <= 1.0, "tanh(1000) must be <= 1, got {hi}");
    // f64 saturates: tanh(±1000) == ±1.0 exactly
    assert!(
        (lo - (-1.0)).abs() < 1e-15,
        "tanh(-1000) should be exactly -1.0 in f64, got {lo}"
    );
    assert!(
        (hi - 1.0).abs() < 1e-15,
        "tanh(1000) should be exactly 1.0 in f64, got {hi}"
    );
}

/// Invariant: tanh output always satisfies -1 <= lo <= hi <= 1 (closed in f64).
/// Mathematical tanh is in open interval (-1, 1), but f64 saturates at ±1.0.
#[test]
fn test_tanh_bounds_output_invariant() {
    for x_lo in [-50.0, -5.0, -1.0, 0.0, 1.0, 5.0] {
        for x_hi in [x_lo, x_lo + 1.0, x_lo + 10.0, x_lo + 100.0] {
            if x_hi < x_lo {
                continue;
            }
            let (lo, hi) = tanh_output_bounds(x_lo, x_hi).expect("finite inputs");
            assert!(
                lo >= -1.0,
                "tanh lo must be >= -1 for [{x_lo}, {x_hi}], got {lo}"
            );
            assert!(
                hi <= 1.0,
                "tanh hi must be <= 1 for [{x_lo}, {x_hi}], got {hi}"
            );
            assert!(
                hi >= lo,
                "tanh hi must be >= lo for [{x_lo}, {x_hi}], got lo={lo}, hi={hi}"
            );
        }
    }
}
