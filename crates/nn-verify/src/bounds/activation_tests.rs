// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Numerical correctness AND error-path tests for activation bounds functions.
//!
//! Per design doc line ~283: "Every `prove_bounds` analytical function needs both
//! error-path AND numerical correctness tests." Dispatch-level tests live in
//! `dispatch_tests.rs` and `dispatch_error_tests.rs`. This file tests both
//! computed output values against hand-calculated expected results AND
//! NaN/Inf/inverted-bounds rejection at the analytical function level.

use super::*;

const TOL: f64 = 1e-6;

fn assert_close(actual: f64, expected: f64, msg: &str) {
    crate::bounds::assert_close_f64(actual, expected, TOL, msg);
}

// ============================================================================
// SiLU-Mul output bounds (AC1)
// ============================================================================

// Reference: silu(x) = x / (1 + exp(-x))
fn silu_ref(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

#[test]
fn test_silu_mul_positive_range_unit_up() {
    // x in [0, 5], up = 1.0
    // silu is monotonically increasing for x > SILU_ARGMIN ~ -1.278
    // silu(0) = 0.0, silu(5) ~ 4.9665
    let (lo, hi) = silu_mul_output_bounds(1.0, 0.0, 5.0).unwrap();
    assert_close(lo, silu_ref(0.0) * 1.0, "silu_mul lo [0,5] up=1");
    assert_close(hi, silu_ref(5.0) * 1.0, "silu_mul hi [0,5] up=1");
}

#[test]
fn test_silu_mul_spanning_argmin() {
    // x in [-3, 3], up = 1.0
    // Range spans SILU_ARGMIN ~ -1.278, so minimum is at argmin
    // silu(-1.278464) ~ -0.27846
    let (lo, hi) = silu_mul_output_bounds(1.0, -3.0, 3.0).unwrap();
    let silu_at_argmin = silu_ref(-1.278_464_5);
    // Soundness: lo must be at or below the true minimum
    assert!(
        lo <= silu_at_argmin + TOL,
        "lower bound must include argmin value: lo={lo}, argmin_val={silu_at_argmin}"
    );
    assert_close(hi, silu_ref(3.0), "silu_mul hi [-3,3] up=1");
    // Tightness: bounds must not be vacuously wide.
    // Expected width = silu(3) - silu_at_argmin ≈ 2.9587 - (-0.2785) ≈ 3.237.
    // Allow 2x slack for interval arithmetic widening.
    let width = hi - lo;
    let expected_width = silu_ref(3.0) - silu_at_argmin;
    assert!(
        width < expected_width * 2.0,
        "silu_mul bounds too wide: width={width}, expected~{expected_width}"
    );
}

#[test]
fn test_silu_mul_below_argmin() {
    // x in [-5, -2], up = 1.0
    // Both endpoints below ARGMIN ~ -1.278. silu is NOT monotonic here.
    // silu(-5) ~ -0.03346, silu(-2) ~ -0.23841
    // Lower = silu(-2) (most negative), upper = silu(-5) (closer to 0)
    let (lo, hi) = silu_mul_output_bounds(1.0, -5.0, -2.0).unwrap();
    assert_close(lo, silu_ref(-2.0), "silu_mul lo [-5,-2] up=1");
    assert_close(hi, silu_ref(-5.0), "silu_mul hi [-5,-2] up=1");
}

#[test]
fn test_silu_mul_negative_up() {
    // x in [0, 2], up = -3.0
    // silu(0) = 0, silu(2) ~ 1.7616
    // output = silu(x) * (-3): flips sign → lo = silu(2)*(-3) ~ -5.2847, hi = silu(0)*(-3) = 0
    let (lo, hi) = silu_mul_output_bounds(-3.0, 0.0, 2.0).unwrap();
    assert_close(lo, silu_ref(2.0) * -3.0, "silu_mul lo [0,2] up=-3");
    assert_close(hi, silu_ref(0.0) * -3.0, "silu_mul hi [0,2] up=-3");
}

#[test]
fn test_silu_mul_zero_up() {
    // up = 0: output always 0 regardless of x
    let (lo, hi) = silu_mul_output_bounds(0.0, -10.0, 10.0).unwrap();
    assert_eq!(lo, 0.0, "silu_mul lo up=0 should be 0");
    assert_eq!(hi, 0.0, "silu_mul hi up=0 should be 0");
}

// ============================================================================
// Sigmoid output bounds (AC2)
// ============================================================================

#[test]
fn test_sigmoid_near_zero() {
    // x in [-1, 1]
    // sigmoid(-1) ~ 0.26894, sigmoid(1) ~ 0.73106
    let (lo, hi) = sigmoid_output_bounds(-1.0, 1.0).unwrap();
    assert_close(lo, 1.0 / (1.0 + 1.0_f64.exp()), "sigmoid lo [-1,1]");
    assert_close(hi, 1.0 / (1.0 + (-1.0_f64).exp()), "sigmoid hi [-1,1]");
}

#[test]
fn test_sigmoid_large_positive() {
    // x in [10, 20]
    // sigmoid(10) ~ 0.99995, sigmoid(20) ~ 1.0 - 2e-9
    // Both very close to 1 but clamped below 1 - EPSILON
    let (lo, hi) = sigmoid_output_bounds(10.0, 20.0).unwrap();
    let expected_lo = 1.0 / (1.0 + (-10.0_f64).exp()); // sigmoid(10) ~ 0.99995
    let expected_hi = 1.0 / (1.0 + (-20.0_f64).exp()); // sigmoid(20) ~ 1.0 - 2e-9
                                                       // Soundness: bounds contain the true values
    assert!(lo > 0.999, "sigmoid(10) should be > 0.999, got {lo}");
    assert!(hi < 1.0, "sigmoid should be < 1.0, got {hi}");
    assert!(lo <= hi, "sigmoid is monotonic: lo <= hi");
    // Tightness: bounds should be close to true sigmoid values.
    // sigmoid is monotonic, so expected width = sigmoid(20) - sigmoid(10) ≈ 5e-5.
    // Allow 2x slack for interval arithmetic widening.
    let width = hi - lo;
    let expected_width = expected_hi - expected_lo;
    assert!(
        width < expected_width * 2.0 + 1e-4,
        "sigmoid bounds too wide: width={width}, expected~{expected_width}"
    );
}

#[test]
fn test_sigmoid_large_negative() {
    // x in [-20, -10]
    // sigmoid(-10) ~ 4.54e-5, sigmoid(-20) ~ 2.06e-9
    // Both very close to 0, clamped above MIN_POSITIVE by activation.rs:118.
    // Exercises the lower saturation region where (-x).exp() overflows.
    let (lo, hi) = sigmoid_output_bounds(-20.0, -10.0).unwrap();
    let expected_lo = 1.0 / (1.0 + 20.0_f64.exp()); // sigmoid(-20) ~ 2.06e-9
    let expected_hi = 1.0 / (1.0 + 10.0_f64.exp()); // sigmoid(-10) ~ 4.54e-5
                                                    // Soundness: bounds must contain the true values
    assert!(lo > 0.0, "sigmoid is always > 0, got {lo}");
    assert!(hi < 0.001, "sigmoid(-10) should be < 0.001, got {hi}");
    assert!(lo <= hi, "sigmoid is monotonic: lo <= hi");
    // Tightness: bounds should be close to true sigmoid values.
    let width = hi - lo;
    let expected_width = expected_hi - expected_lo;
    assert!(
        width < expected_width * 2.0 + 1e-4,
        "sigmoid large-negative bounds too wide: width={width}, expected~{expected_width}"
    );
}

#[test]
fn test_sigmoid_at_zero() {
    // x in [0, 0] — point interval
    // sigmoid(0) = 0.5
    let (lo, hi) = sigmoid_output_bounds(0.0, 0.0).unwrap();
    assert_close(lo, 0.5, "sigmoid lo at 0");
    assert_close(hi, 0.5, "sigmoid hi at 0");
}

// ============================================================================
// ReLU output bounds (AC3)
// ============================================================================

#[test]
fn test_relu_negative_only() {
    // x in [-5, -1]
    // relu(x) = max(x, 0) → both 0
    let (lo, hi) = relu_output_bounds(-5.0, -1.0).unwrap();
    assert_eq!(lo, 0.0, "relu lo [-5,-1]");
    assert_eq!(hi, 0.0, "relu hi [-5,-1]");
}

#[test]
fn test_relu_mixed() {
    // x in [-3, 7]
    // relu(-3) = 0, relu(7) = 7
    let (lo, hi) = relu_output_bounds(-3.0, 7.0).unwrap();
    assert_eq!(lo, 0.0, "relu lo [-3,7]");
    assert_eq!(hi, 7.0, "relu hi [-3,7]");
}

#[test]
fn test_relu_positive_only() {
    // x in [2, 10]
    // relu(2) = 2, relu(10) = 10
    let (lo, hi) = relu_output_bounds(2.0, 10.0).unwrap();
    assert_eq!(lo, 2.0, "relu lo [2,10]");
    assert_eq!(hi, 10.0, "relu hi [2,10]");
}

// ============================================================================
// Tanh output bounds (AC4)
// ============================================================================

#[test]
fn test_tanh_unit_interval() {
    // x in [-1, 1]
    // tanh(-1) ~ -0.76159, tanh(1) ~ 0.76159
    let (lo, hi) = tanh_output_bounds(-1.0, 1.0).unwrap();
    assert_close(lo, (-1.0_f64).tanh(), "tanh lo [-1,1]");
    assert_close(hi, 1.0_f64.tanh(), "tanh hi [-1,1]");
}

#[test]
fn test_tanh_wide_range() {
    // x in [-10, 10]
    // tanh(-10) ~ -1.0 + 4e-9, tanh(10) ~ 1.0 - 4e-9
    let (lo, hi) = tanh_output_bounds(-10.0, 10.0).unwrap();
    assert!(lo > -1.0, "tanh should be > -1.0, got {lo}");
    assert!(hi < 1.0, "tanh should be < 1.0, got {hi}");
    assert!(lo < -0.999, "tanh(-10) should be near -1, got {lo}");
    assert!(hi > 0.999, "tanh(10) should be near 1, got {hi}");
}

#[test]
fn test_tanh_at_zero() {
    // tanh(0) = 0
    let (lo, hi) = tanh_output_bounds(0.0, 0.0).unwrap();
    assert_close(lo, 0.0, "tanh lo at 0");
    assert_close(hi, 0.0, "tanh hi at 0");
}

// ============================================================================
// GELU output bounds (AC5)
// ============================================================================

// Reference: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn gelu_ref(x: f64) -> f64 {
    let k: f64 = 0.797_884_560_802_865_4; // sqrt(2/pi)
    let inner = k * (x + 0.044715 * x * x * x);
    let e2 = (2.0 * inner).exp();
    0.5 * x * (2.0 - 2.0 / (e2 + 1.0))
}

#[test]
fn test_gelu_positive_range() {
    // x in [1, 5] — entirely above GELU_ARGMIN ~ -0.7523
    // gelu(1) ~ 0.8412, gelu(5) ~ 5.0
    let (lo, hi) = gelu_output_bounds(1.0, 5.0).unwrap();
    assert_close(lo, gelu_ref(1.0), "gelu lo [1,5]");
    assert_close(hi, gelu_ref(5.0), "gelu hi [1,5]");
}

#[test]
fn test_gelu_spanning_argmin() {
    // x in [-2, 2] — spans GELU_ARGMIN ~ -0.7523
    // gelu at argmin ~ -0.170, gelu(-2) ~ -0.0454, gelu(2) ~ 1.9546
    // Lower should include argmin value.
    let (lo, hi) = gelu_output_bounds(-2.0, 2.0).unwrap();
    let gelu_at_argmin = gelu_ref(-0.752_252_6);
    // Soundness: lo must be at or below the true minimum
    assert!(
        lo <= gelu_at_argmin + TOL,
        "lower bound must include argmin value: lo={lo}, argmin_val={gelu_at_argmin}"
    );
    assert_close(hi, gelu_ref(2.0), "gelu hi [-2,2]");
    // Tightness: bounds must not be vacuously wide.
    // Expected width = gelu(2) - gelu_at_argmin ≈ 1.9546 - (-0.170) ≈ 2.125.
    // Allow 2x slack for interval arithmetic widening.
    let width = hi - lo;
    let expected_width = gelu_ref(2.0) - gelu_at_argmin;
    assert!(
        width < expected_width * 2.0,
        "gelu bounds too wide: width={width}, expected~{expected_width}"
    );
}

#[test]
fn test_gelu_below_argmin() {
    // x in [-5, -2] — entirely below GELU_ARGMIN
    // gelu(-5) ~ -1.7e-6, gelu(-2) ~ -0.0454
    // gelu is increasing again below argmin (approaching 0 from negative side)
    let (lo, hi) = gelu_output_bounds(-5.0, -2.0).unwrap();
    // gelu(-2) is the more negative value (closer to argmin trough)
    assert_close(lo, gelu_ref(-2.0), "gelu lo [-5,-2]");
    // gelu(-5) is closer to 0
    assert_close(hi, gelu_ref(-5.0), "gelu hi [-5,-2]");
}

#[test]
fn test_gelu_at_zero() {
    // gelu(0) = 0
    let (lo, hi) = gelu_output_bounds(0.0, 0.0).unwrap();
    assert_close(lo, 0.0, "gelu lo at 0");
    assert_close(hi, 0.0, "gelu hi at 0");
}

// ============================================================================
// Error-path tests: NaN/Inf/inverted bounds rejection
// ============================================================================

// --- SiLU-Mul error paths ---

#[test]
fn test_silu_mul_nan_lower_rejected() {
    assert!(silu_mul_output_bounds(1.0, f64::NAN, 1.0).is_err());
}

#[test]
fn test_silu_mul_nan_upper_rejected() {
    assert!(silu_mul_output_bounds(1.0, -1.0, f64::NAN).is_err());
}

#[test]
fn test_silu_mul_inf_lower_rejected() {
    assert!(silu_mul_output_bounds(1.0, f64::INFINITY, 1.0).is_err());
}

#[test]
fn test_silu_mul_inf_upper_rejected() {
    assert!(silu_mul_output_bounds(1.0, -1.0, f64::NEG_INFINITY).is_err());
}

#[test]
fn test_silu_mul_inverted_bounds_rejected() {
    assert!(silu_mul_output_bounds(1.0, 5.0, -5.0).is_err());
}

#[test]
fn test_silu_mul_nan_up_const_rejected() {
    assert!(silu_mul_output_bounds(f64::NAN, -1.0, 1.0).is_err());
}

#[test]
fn test_silu_mul_inf_up_const_rejected() {
    assert!(silu_mul_output_bounds(f64::INFINITY, -1.0, 1.0).is_err());
}

// --- Sigmoid error paths ---

#[test]
fn test_sigmoid_nan_lower_rejected() {
    assert!(sigmoid_output_bounds(f64::NAN, 1.0).is_err());
}

#[test]
fn test_sigmoid_inf_upper_rejected() {
    assert!(sigmoid_output_bounds(-1.0, f64::INFINITY).is_err());
}

#[test]
fn test_sigmoid_inverted_bounds_rejected() {
    assert!(sigmoid_output_bounds(5.0, -5.0).is_err());
}

// --- ReLU error paths ---

#[test]
fn test_relu_nan_lower_rejected() {
    assert!(relu_output_bounds(f64::NAN, 1.0).is_err());
}

#[test]
fn test_relu_inf_upper_rejected() {
    assert!(relu_output_bounds(-1.0, f64::INFINITY).is_err());
}

#[test]
fn test_relu_inverted_bounds_rejected() {
    assert!(relu_output_bounds(5.0, -5.0).is_err());
}

// --- Tanh error paths ---

#[test]
fn test_tanh_nan_lower_rejected() {
    assert!(tanh_output_bounds(f64::NAN, 1.0).is_err());
}

#[test]
fn test_tanh_inf_upper_rejected() {
    assert!(tanh_output_bounds(-1.0, f64::INFINITY).is_err());
}

#[test]
fn test_tanh_inverted_bounds_rejected() {
    assert!(tanh_output_bounds(5.0, -5.0).is_err());
}

// --- GELU error paths ---

#[test]
fn test_gelu_nan_lower_rejected() {
    assert!(gelu_output_bounds(f64::NAN, 1.0).is_err());
}

#[test]
fn test_gelu_inf_upper_rejected() {
    assert!(gelu_output_bounds(-1.0, f64::INFINITY).is_err());
}

#[test]
fn test_gelu_inverted_bounds_rejected() {
    assert!(gelu_output_bounds(5.0, -5.0).is_err());
}

// ============================================================================
// LeakyReLU output bounds (AC6 — #2165)
// ============================================================================

#[test]
fn test_leaky_relu_positive_only() {
    // x in [2, 10], slope = 0.01
    // leaky_relu(2) = 2, leaky_relu(10) = 10
    let (lo, hi) = leaky_relu_output_bounds(0.01, 2.0, 10.0).unwrap();
    assert_close(lo, 2.0, "leaky_relu lo [2,10]");
    assert_close(hi, 10.0, "leaky_relu hi [2,10]");
}

#[test]
fn test_leaky_relu_negative_only() {
    // x in [-5, -1], slope = 0.01
    // leaky_relu(-5, 0.01) = -0.05, leaky_relu(-1, 0.01) = -0.01
    let (lo, hi) = leaky_relu_output_bounds(0.01, -5.0, -1.0).unwrap();
    assert_close(lo, -0.05, "leaky_relu lo [-5,-1]");
    assert_close(hi, -0.01, "leaky_relu hi [-5,-1]");
}

#[test]
fn test_leaky_relu_mixed_range() {
    // x in [-3, 7], slope = 0.1
    // leaky_relu(-3, 0.1) = -0.3, leaky_relu(7, 0.1) = 7
    let (lo, hi) = leaky_relu_output_bounds(0.1, -3.0, 7.0).unwrap();
    assert_close(lo, -0.3, "leaky_relu lo [-3,7]");
    assert_close(hi, 7.0, "leaky_relu hi [-3,7]");
}

#[test]
fn test_leaky_relu_slope_one_is_identity() {
    // slope = 1.0: leaky_relu(x, 1) = x for all x
    let (lo, hi) = leaky_relu_output_bounds(1.0, -5.0, 5.0).unwrap();
    assert_close(lo, -5.0, "leaky_relu slope=1 lo");
    assert_close(hi, 5.0, "leaky_relu slope=1 hi");
}

#[test]
fn test_leaky_relu_negative_slope() {
    // slope = -0.5, x in [-2, 4]
    // leaky_relu(-2, -0.5) = 1.0 (negative slope * negative x = positive)
    // leaky_relu(0) = 0
    // leaky_relu(4, -0.5) = 4
    // With negative slope, kink at x=0 is a local minimum (value 0).
    let (lo, hi) = leaky_relu_output_bounds(-0.5, -2.0, 4.0).unwrap();
    assert_close(lo, 0.0, "leaky_relu neg slope lo (kink at 0)");
    assert_close(hi, 4.0, "leaky_relu neg slope hi");
}

// --- LeakyReLU error paths ---

#[test]
fn test_leaky_relu_nan_lower_rejected() {
    assert!(leaky_relu_output_bounds(0.01, f64::NAN, 1.0).is_err());
}

#[test]
fn test_leaky_relu_inf_upper_rejected() {
    assert!(leaky_relu_output_bounds(0.01, -1.0, f64::INFINITY).is_err());
}

#[test]
fn test_leaky_relu_inverted_bounds_rejected() {
    assert!(leaky_relu_output_bounds(0.01, 5.0, -5.0).is_err());
}

#[test]
fn test_leaky_relu_nan_alpha_rejected() {
    assert!(leaky_relu_output_bounds(f64::NAN, -1.0, 1.0).is_err());
}

#[test]
fn test_leaky_relu_inf_alpha_rejected() {
    assert!(leaky_relu_output_bounds(f64::INFINITY, -1.0, 1.0).is_err());
}

// ============================================================================
// Exp output bounds (AC7 — #2165)
// ============================================================================

#[test]
fn test_exp_unit_interval() {
    // x in [-1, 1]
    // exp(-1) ~ 0.3679, exp(1) ~ 2.7183
    let (lo, hi) = exp_output_bounds(-1.0, 1.0).unwrap();
    assert_close(lo, (-1.0_f64).exp(), "exp lo [-1,1]");
    assert_close(hi, 1.0_f64.exp(), "exp hi [-1,1]");
}

#[test]
fn test_exp_at_zero() {
    // exp(0) = 1
    let (lo, hi) = exp_output_bounds(0.0, 0.0).unwrap();
    assert_close(lo, 1.0, "exp lo at 0");
    assert_close(hi, 1.0, "exp hi at 0");
}

#[test]
fn test_exp_negative_range() {
    // x in [-10, -5]
    // exp(-10) ~ 4.54e-5, exp(-5) ~ 6.74e-3
    let (lo, hi) = exp_output_bounds(-10.0, -5.0).unwrap();
    assert!(lo > 0.0, "exp is always positive");
    assert_close(lo, (-10.0_f64).exp(), "exp lo [-10,-5]");
    assert_close(hi, (-5.0_f64).exp(), "exp hi [-10,-5]");
}

#[test]
fn test_exp_large_positive_overflow() {
    // exp(1000) = +inf — should error on output finiteness
    assert!(exp_output_bounds(0.0, 1000.0).is_err());
}

// --- Exp error paths ---

#[test]
fn test_exp_nan_lower_rejected() {
    assert!(exp_output_bounds(f64::NAN, 1.0).is_err());
}

#[test]
fn test_exp_inf_upper_rejected() {
    assert!(exp_output_bounds(-1.0, f64::INFINITY).is_err());
}

#[test]
fn test_exp_inverted_bounds_rejected() {
    assert!(exp_output_bounds(5.0, -5.0).is_err());
}

// ============================================================================
// Softplus output bounds (AC8 — #2165)
// ============================================================================

#[test]
fn test_softplus_unit_interval() {
    // x in [-1, 1]
    // softplus(-1) = ln(1 + exp(-1)) ~ 0.3133, softplus(1) = ln(1 + exp(1)) ~ 1.3133
    let (lo, hi) = softplus_output_bounds(-1.0, 1.0).unwrap();
    let expected_lo = (-1.0_f64).exp().ln_1p();
    let expected_hi = 1.0_f64.exp().ln_1p();
    assert_close(lo, expected_lo, "softplus lo [-1,1]");
    assert_close(hi, expected_hi, "softplus hi [-1,1]");
}

#[test]
fn test_softplus_at_zero() {
    // softplus(0) = ln(2) ~ 0.6931
    let (lo, hi) = softplus_output_bounds(0.0, 0.0).unwrap();
    assert_close(lo, 2.0_f64.ln(), "softplus lo at 0");
    assert_close(hi, 2.0_f64.ln(), "softplus hi at 0");
}

#[test]
fn test_softplus_large_positive() {
    // softplus(x) ≈ x for large x
    let (lo, hi) = softplus_output_bounds(10.0, 20.0).unwrap();
    assert!((lo - 10.0).abs() < 0.001, "softplus(10) ≈ 10, got {lo}");
    assert!((hi - 20.0).abs() < 0.001, "softplus(20) ≈ 20, got {hi}");
}

#[test]
fn test_softplus_large_negative() {
    // softplus(x) ≈ exp(x) ≈ 0 for large negative x
    let (lo, hi) = softplus_output_bounds(-20.0, -10.0).unwrap();
    assert!(lo > 0.0, "softplus is always positive");
    assert!(lo < 1e-8, "softplus(-20) near 0, got {lo}");
    assert!(hi < 0.001, "softplus(-10) near 0, got {hi}");
}

// --- Softplus error paths ---

#[test]
fn test_softplus_nan_lower_rejected() {
    assert!(softplus_output_bounds(f64::NAN, 1.0).is_err());
}

#[test]
fn test_softplus_inf_upper_rejected() {
    assert!(softplus_output_bounds(-1.0, f64::INFINITY).is_err());
}

#[test]
fn test_softplus_inverted_bounds_rejected() {
    assert!(softplus_output_bounds(5.0, -5.0).is_err());
}
