// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for RoPE scalar bounds (extracted from rope_bounds.rs).

use super::*;
use std::f32::consts::{FRAC_PI_2, PI, TAU};

// --- cos_range tests ---

#[test]
fn cos_range_single_point_zero() {
    let (lo, hi) = cos_range(0.0, 0.0);
    assert!((lo - 1.0).abs() < 1e-6, "cos(0)=1, got lo={lo}");
    assert!((hi - 1.0).abs() < 1e-6, "cos(0)=1, got hi={hi}");
}

#[test]
fn cos_range_single_point_pi() {
    let (lo, hi) = cos_range(PI, PI);
    assert!((lo - (-1.0)).abs() < 1e-6, "cos(π)=-1, got lo={lo}");
    assert!((hi - (-1.0)).abs() < 1e-6, "cos(π)=-1, got hi={hi}");
}

#[test]
fn cos_range_full_period() {
    let (lo, hi) = cos_range(0.0, TAU);
    assert_eq!(lo, -1.0);
    assert_eq!(hi, 1.0);
}

#[test]
fn cos_range_straddles_pi() {
    // [π-0.1, π+0.1] contains cos=-1 trough
    let (lo, hi) = cos_range(PI - 0.1, PI + 0.1);
    assert_eq!(lo, -1.0, "interval contains π, cos trough");
    assert!(hi > -1.0);
}

#[test]
fn cos_range_straddles_two_pi() {
    // [2π-0.1, 2π+0.1] contains cos=1 peak
    let (lo, hi) = cos_range(TAU - 0.1, TAU + 0.1);
    assert_eq!(hi, 1.0, "interval contains 2π, cos peak");
    assert!(lo < 1.0);
}

#[test]
fn cos_range_first_quadrant() {
    // [0, π/2]: cos decreasing from 1 to 0
    let (lo, hi) = cos_range(0.0, FRAC_PI_2);
    assert!(lo >= -1e-6, "cos(π/2) ≈ 0");
    assert!((hi - 1.0).abs() < 1e-6, "cos(0) = 1");
}

#[test]
fn cos_range_negative_interval() {
    // [-π, 0]: contains cos=-1 at -π, cos=1 at 0
    let (lo, hi) = cos_range(-PI, 0.0);
    assert_eq!(lo, -1.0);
    assert!((hi - 1.0).abs() < 1e-6);
}

// --- sin_range tests ---

#[test]
fn sin_range_single_point_zero() {
    let (lo, hi) = sin_range(0.0, 0.0);
    assert!(lo.abs() < 1e-5, "sin(0)≈0, got lo={lo}");
    assert!(hi.abs() < 1e-5, "sin(0)≈0, got hi={hi}");
}

#[test]
fn sin_range_single_point_pi_half() {
    let (lo, hi) = sin_range(FRAC_PI_2, FRAC_PI_2);
    assert!((lo - 1.0).abs() < 1e-5, "sin(π/2)=1, got lo={lo}");
    assert!((hi - 1.0).abs() < 1e-5, "sin(π/2)=1, got hi={hi}");
}

#[test]
fn sin_range_full_period() {
    let (lo, hi) = sin_range(0.0, TAU);
    assert_eq!(lo, -1.0);
    assert_eq!(hi, 1.0);
}

// --- interval_mul tests ---

#[test]
fn interval_mul_positive_ranges() {
    let (lo, hi) = interval_mul(1.0, 3.0, 2.0, 4.0);
    assert!((lo - 2.0).abs() < 1e-6);
    assert!((hi - 12.0).abs() < 1e-6);
}

#[test]
fn interval_mul_mixed_signs() {
    // [-2, 3] × [-1, 4]: corners = (2, -8, -3, 12) → [-8, 12]
    let (lo, hi) = interval_mul(-2.0, 3.0, -1.0, 4.0);
    assert!((lo - (-8.0)).abs() < 1e-6);
    assert!((hi - 12.0).abs() < 1e-6);
}

#[test]
fn interval_mul_zero_in_range() {
    let (lo, hi) = interval_mul(0.0, 1.0, -1.0, 1.0);
    assert!((lo - (-1.0)).abs() < 1e-6);
    assert!((hi - 1.0).abs() < 1e-6);
}

// --- nan_propagating_min/max tests ---

#[test]
fn nan_propagating_min_normal() {
    assert_eq!(nan_propagating_min_f32(1.0, 2.0), 1.0);
    assert_eq!(nan_propagating_min_f32(2.0, 1.0), 1.0);
}

#[test]
fn nan_propagating_min_nan() {
    assert!(nan_propagating_min_f32(f32::NAN, 1.0).is_nan());
    assert!(nan_propagating_min_f32(1.0, f32::NAN).is_nan());
}

#[test]
fn nan_propagating_max_normal() {
    assert_eq!(nan_propagating_max_f32(1.0, 2.0), 2.0);
}

#[test]
fn nan_propagating_max_nan() {
    assert!(nan_propagating_max_f32(f32::NAN, 1.0).is_nan());
}

// --- rope bounds error path tests ---

#[test]
fn rope_cos_bounds_rejects_nan_input() {
    let result = rope_cos_scalar_bounds(f32::NAN, 1.0, -1.0, 1.0, 0.0, 1.0);
    assert!(result.is_err(), "NaN input should be rejected");
}

#[test]
fn rope_cos_bounds_rejects_inf_input() {
    let result = rope_cos_scalar_bounds(f32::INFINITY, 1.0, -1.0, 1.0, 0.0, 1.0);
    assert!(result.is_err(), "Inf input should be rejected");
}

#[test]
fn rope_cos_bounds_rejects_inverted_pair() {
    let result = rope_cos_scalar_bounds(5.0, -5.0, -1.0, 1.0, 0.0, 1.0);
    assert!(result.is_err(), "inverted x0 bounds should be rejected");
}

#[test]
fn rope_sin_bounds_rejects_nan_input() {
    let result = rope_sin_scalar_bounds(-1.0, 1.0, f32::NAN, 1.0, 0.0, 1.0);
    assert!(result.is_err(), "NaN input should be rejected");
}

// --- Numerical correctness tests ---

#[test]
fn rope_cos_bounds_x0_only_at_zero_freq() {
    // freq=0: cos(0)=1, sin(0)=0 → rope_cos = x0
    let (lo, hi) = rope_cos_scalar_bounds(-3.0, 5.0, -2.0, 4.0, 0.0, 0.0).unwrap();
    assert!(lo <= -3.0 + 1e-5, "lower should be ≤ -3.0, got {lo}");
    assert!(hi >= 5.0 - 1e-5, "upper should be ≥ 5.0, got {hi}");
}

#[test]
fn rope_sin_bounds_x1_only_at_zero_freq() {
    // freq=0: sin(0)=0, cos(0)=1 → rope_sin = x1
    let (lo, hi) = rope_sin_scalar_bounds(-2.0, 3.0, -4.0, 7.0, 0.0, 0.0).unwrap();
    assert!(lo <= -4.0 + 1e-5, "lower should be ≤ -4.0, got {lo}");
    assert!(hi >= 7.0 - 1e-5, "upper should be ≥ 7.0, got {hi}");
}

#[test]
fn rope_cos_bounds_unit_inputs_full_freq() {
    // x0=x1=[-1,1], freq=[0,2π]: max is √2 at f=-π/4, min is -√2
    let (lo, hi) = rope_cos_scalar_bounds(-1.0, 1.0, -1.0, 1.0, 0.0, TAU).unwrap();
    let sqrt2 = std::f32::consts::SQRT_2;
    assert!(lo <= -sqrt2 + 0.01, "should reach ≈-√2, got lo={lo}");
    assert!(hi >= sqrt2 - 0.01, "should reach ≈√2, got hi={hi}");
}

#[test]
fn rope_bounds_output_always_ordered() {
    let configs = [
        (-5.0, 5.0, -3.0, 7.0, -0.5, 0.5),
        (-2.0, 3.0, -4.0, 1.0, 2.5, 3.8),
        (-1.0, 1.0, -1.0, 1.0, 5.5, 7.0),
        (0.0, 1.0, 0.0, 1.0, 0.0, PI),
    ];
    for (x0l, x0h, x1l, x1h, fl, fh) in configs {
        let (clo, chi) = rope_cos_scalar_bounds(x0l, x0h, x1l, x1h, fl, fh).unwrap();
        assert!(
            clo <= chi,
            "cos inverted for ({x0l},{x0h},{x1l},{x1h},{fl},{fh})"
        );
        let (slo, shi) = rope_sin_scalar_bounds(x0l, x0h, x1l, x1h, fl, fh).unwrap();
        assert!(
            slo <= shi,
            "sin inverted for ({x0l},{x0h},{x1l},{x1h},{fl},{fh})"
        );
    }
}
