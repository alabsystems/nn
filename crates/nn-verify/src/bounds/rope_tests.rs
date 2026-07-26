// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Numerical correctness tests for RoPE bounds functions.
//!
//! Per design doc line ~283: "Every `prove_bounds` analytical function needs both
//! error-path AND numerical correctness tests." These tests verify that
//! `rope_output_bounds` computes correct values via the DSL scalar bounds functions
//! for both rope_cos and rope_sin.

use super::*;
use nn_dsl::{rope_cos_scalar_bounds, rope_sin_scalar_bounds};

const TOL: f64 = 1e-3; // f32 path — wider tolerance

fn assert_close(actual: f64, expected: f64, msg: &str) {
    crate::bounds::assert_close_f64(actual, expected, TOL, msg);
}

// ============================================================================
// RoPE-cos output bounds (AC10a)
// ============================================================================
// rope_cos(x0, x1, freq) = x0 * cos(freq) - x1 * sin(freq)
// When x0 is the variable and x1, freq are constant point intervals,
// output is linear in x0: output = x0 * cos(freq) - x1 * sin(freq)

#[test]
fn test_rope_cos_zero_freq() {
    // freq = 0 → cos(0) = 1, sin(0) = 0
    // rope_cos = x0 * 1 - x1 * 0 = x0
    // x1 = 3.0, x0 in [-2, 5]
    // Expected: output in [-2, 5]
    let (lo, hi) = rope_output_bounds(3.0, 0.0, -2.0, 5.0, rope_cos_scalar_bounds).unwrap();
    assert_close(lo, -2.0, "rope_cos lo freq=0");
    assert_close(hi, 5.0, "rope_cos hi freq=0");
}

#[test]
fn test_rope_cos_pi_half_freq() {
    // freq = pi/2 → cos(pi/2) ~ 0, sin(pi/2) = 1
    // rope_cos = x0 * 0 - x1 * 1 = -x1
    // x1 = 2.0, x0 in [-1, 1]
    // Expected: output ~ -2.0 (constant, independent of x0)
    let freq = std::f32::consts::FRAC_PI_2;
    let (lo, hi) = rope_output_bounds(2.0, freq, -1.0, 1.0, rope_cos_scalar_bounds).unwrap();
    // cos(pi/2) is numerically ~0 but not exactly 0 in f32
    // Output should be approximately -2.0 for both bounds
    assert!(
        (lo - (-2.0)).abs() < 0.1,
        "rope_cos lo freq=pi/2: expected ~-2.0, got {lo}"
    );
    assert!(
        (hi - (-2.0)).abs() < 0.1,
        "rope_cos hi freq=pi/2: expected ~-2.0, got {hi}"
    );
}

#[test]
fn test_rope_cos_known_freq() {
    // freq = 1.0, x1 = 0.5, x0 in [1, 3]
    // cos(1) ~ 0.5403, sin(1) ~ 0.8415
    // rope_cos = x0 * 0.5403 - 0.5 * 0.8415
    //          = x0 * 0.5403 - 0.4207
    // At x0=1: 0.5403 - 0.4207 = 0.1196
    // At x0=3: 1.6209 - 0.4207 = 1.2002
    let (lo, hi) = rope_output_bounds(0.5, 1.0, 1.0, 3.0, rope_cos_scalar_bounds).unwrap();
    let cos_1 = 1.0_f64.cos();
    let sin_1 = 1.0_f64.sin();
    let expected_lo = 1.0 * cos_1 - 0.5 * sin_1;
    let expected_hi = 3.0 * cos_1 - 0.5 * sin_1;
    // Soundness: bounds must contain the true point evaluations
    assert!(
        lo <= expected_lo + TOL,
        "rope_cos lo: {lo} <= {expected_lo}"
    );
    assert!(
        hi >= expected_hi - TOL,
        "rope_cos hi: {hi} >= {expected_hi}"
    );
    // Tightness: bounds must not be vacuously wide.
    // RoPE-cos is linear in x0, so expected width = (3-1) * |cos(1)| ≈ 1.08.
    // Allow 2x slack for interval arithmetic widening.
    let width = hi - lo;
    let expected_width = expected_hi - expected_lo;
    assert!(
        width < expected_width * 2.0,
        "rope_cos bounds too wide: width={width}, expected~{expected_width}"
    );
}

// ============================================================================
// RoPE-sin output bounds (AC10b)
// ============================================================================
// rope_sin(x0, x1, freq) = x0 * sin(freq) + x1 * cos(freq)

#[test]
fn test_rope_sin_zero_freq() {
    // freq = 0 → sin(0) = 0, cos(0) = 1
    // rope_sin = x0 * 0 + x1 * 1 = x1
    // x1 = 4.0, x0 in [-3, 3]
    // Expected: output ~ 4.0 (constant)
    let (lo, hi) = rope_output_bounds(4.0, 0.0, -3.0, 3.0, rope_sin_scalar_bounds).unwrap();
    assert_close(lo, 4.0, "rope_sin lo freq=0");
    assert_close(hi, 4.0, "rope_sin hi freq=0");
}

#[test]
fn test_rope_sin_pi_half_freq() {
    // freq = pi/2 → sin(pi/2) = 1, cos(pi/2) ~ 0
    // rope_sin = x0 * 1 + x1 * 0 = x0
    // x1 = 2.0, x0 in [-1, 5]
    // Expected: output ~ [-1, 5]
    let freq = std::f32::consts::FRAC_PI_2;
    let (lo, hi) = rope_output_bounds(2.0, freq, -1.0, 5.0, rope_sin_scalar_bounds).unwrap();
    assert!(
        (lo - (-1.0)).abs() < 0.1,
        "rope_sin lo freq=pi/2: expected ~-1.0, got {lo}"
    );
    assert!(
        (hi - 5.0).abs() < 0.1,
        "rope_sin hi freq=pi/2: expected ~5.0, got {hi}"
    );
}

#[test]
fn test_rope_sin_known_freq() {
    // freq = 1.0, x1 = 0.5, x0 in [1, 3]
    // sin(1) ~ 0.8415, cos(1) ~ 0.5403
    // rope_sin = x0 * 0.8415 + 0.5 * 0.5403
    //          = x0 * 0.8415 + 0.2701
    // At x0=1: 0.8415 + 0.2701 = 1.1116
    // At x0=3: 2.5244 + 0.2701 = 2.7945
    let (lo, hi) = rope_output_bounds(0.5, 1.0, 1.0, 3.0, rope_sin_scalar_bounds).unwrap();
    let sin_1 = 1.0_f64.sin();
    let cos_1 = 1.0_f64.cos();
    let expected_lo = 1.0 * sin_1 + 0.5 * cos_1;
    let expected_hi = 3.0 * sin_1 + 0.5 * cos_1;
    assert!(
        lo <= expected_lo + TOL,
        "rope_sin lo: {lo} <= {expected_lo}"
    );
    assert!(
        hi >= expected_hi - TOL,
        "rope_sin hi: {hi} >= {expected_hi}"
    );
    // Tightness: bounds must not be vacuously wide.
    // RoPE-sin is linear in x0, so expected width = (3-1) * |sin(1)| ≈ 1.68.
    // Allow 2x slack for interval arithmetic widening.
    let width = hi - lo;
    let expected_width = expected_hi - expected_lo;
    assert!(
        width < expected_width * 2.0,
        "rope_sin bounds too wide: width={width}, expected~{expected_width}"
    );
}

// ============================================================================
// Error-path tests: NaN/Inf/inverted bounds rejection
// ============================================================================

#[test]
fn test_rope_nan_x1_const_rejected() {
    // x1_const (first arg) is NaN → NonFiniteConstantParam { index: 1 }
    assert!(rope_output_bounds(f64::NAN as f32, 1.0, -1.0, 1.0, rope_cos_scalar_bounds).is_err());
}

#[test]
fn test_rope_inf_x1_const_rejected() {
    assert!(rope_output_bounds(f32::INFINITY, 1.0, -1.0, 1.0, rope_cos_scalar_bounds).is_err());
}

#[test]
fn test_rope_nan_freq_const_rejected() {
    // freq_const (second arg) is NaN → NonFiniteConstantParam { index: 2 }
    assert!(rope_output_bounds(1.0, f32::NAN, -1.0, 1.0, rope_cos_scalar_bounds).is_err());
}

#[test]
fn test_rope_inf_freq_const_rejected() {
    assert!(rope_output_bounds(1.0, f32::INFINITY, -1.0, 1.0, rope_sin_scalar_bounds).is_err());
}

#[test]
fn test_rope_nan_lower_rejected() {
    // x0_lower is NaN → NonFiniteBound
    assert!(rope_output_bounds(1.0, 1.0, f32::NAN, 1.0, rope_cos_scalar_bounds).is_err());
}

#[test]
fn test_rope_inf_upper_rejected() {
    // x0_upper is +Inf → NonFiniteBound
    assert!(rope_output_bounds(1.0, 1.0, -1.0, f32::INFINITY, rope_sin_scalar_bounds).is_err());
}

#[test]
fn test_rope_inverted_bounds_rejected() {
    // x0_lower > x0_upper → InvertedBounds
    assert!(rope_output_bounds(1.0, 1.0, 5.0, -5.0, rope_cos_scalar_bounds).is_err());
}
