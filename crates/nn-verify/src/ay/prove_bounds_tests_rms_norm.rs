// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for rms_norm_scalar_output_bounds and norm_affine_output_bounds.
//! Split from prove_bounds_tests.rs (#453).

use super::*;

// ======================== rms_norm_scalar_output_bounds error paths ========================

#[test]
fn test_rms_norm_bounds_nan_rms_inv_returns_error() {
    assert_bounds_error!(
        rms_norm_scalar_output_bounds(f32::NAN, 1.0, -1.0, 1.0),
        "non-finite constant parameter",
        1
    );
}

#[test]
fn test_rms_norm_bounds_inf_weight_returns_error() {
    assert_bounds_error!(
        rms_norm_scalar_output_bounds(1.0, f32::INFINITY, -1.0, 1.0),
        "non-finite constant parameter",
        2
    );
}

#[test]
fn test_rms_norm_bounds_nan_x_lower_returns_error() {
    assert_bounds_error!(
        rms_norm_scalar_output_bounds(1.0, 1.0, f64::NAN, 1.0),
        "non-finite output bounds"
    );
}

#[test]
fn test_rms_norm_bounds_inf_x_upper_returns_error() {
    assert_bounds_error!(
        rms_norm_scalar_output_bounds(1.0, 1.0, -1.0, f64::INFINITY),
        "non-finite output bounds"
    );
}

#[test]
fn test_rms_norm_inverted_bounds_rejected() {
    assert_bounds_error!(
        rms_norm_scalar_output_bounds(1.0, 1.0, 5.0, -5.0),
        "inverted bounds"
    );
}

// ======================== rms_norm_scalar_output_bounds correctness ========================

#[test]
fn test_rms_norm_bounds_basic_positive_coeff() {
    // coeff = rms_inv * weight = 2.0 * 3.0 = 6.0, x in [-1.0, 1.0]
    let (lo, hi) = rms_norm_scalar_output_bounds(2.0, 3.0, -1.0, 1.0).unwrap();
    assert!(
        (lo - -6.0).abs() < 1e-10,
        "lower bound should be -6.0, got {lo}"
    );
    assert!(
        (hi - 6.0).abs() < 1e-10,
        "upper bound should be 6.0, got {hi}"
    );
}

#[test]
fn test_rms_norm_bounds_negative_coeff_flips_order() {
    // coeff = 2.0 * -3.0 = -6.0 (negative)
    let (lo, hi) = rms_norm_scalar_output_bounds(2.0, -3.0, -1.0, 1.0).unwrap();
    assert!(
        (lo - -6.0).abs() < 1e-10,
        "negative coeff: lower should be -6.0, got {lo}"
    );
    assert!(
        (hi - 6.0).abs() < 1e-10,
        "negative coeff: upper should be 6.0, got {hi}"
    );
}

#[test]
fn test_rms_norm_bounds_zero_coeff() {
    let (lo, hi) = rms_norm_scalar_output_bounds(0.0, 5.0, -100.0, 100.0).unwrap();
    assert!(
        lo.abs() < 1e-10,
        "zero coeff: lower should be 0.0, got {lo}"
    );
    assert!(
        hi.abs() < 1e-10,
        "zero coeff: upper should be 0.0, got {hi}"
    );
}

#[test]
fn test_rms_norm_bounds_overflow_returns_error() {
    assert_bounds_error!(
        rms_norm_scalar_output_bounds(f32::MAX, f32::MAX, -1e300, 1e300),
        "non-finite output bounds"
    );
}

// ======================== norm_affine_output_bounds error paths ========================

#[test]
fn test_norm_affine_bounds_nan_mean_returns_error() {
    assert_bounds_error!(
        norm_affine_output_bounds(f32::NAN, 1.0, 1e-5, 1.0, 0.0, -1.0, 1.0),
        "non-finite constant parameter",
        1
    );
}

#[test]
fn test_norm_affine_bounds_nan_gamma_returns_error() {
    assert_bounds_error!(
        norm_affine_output_bounds(0.0, 1.0, 1e-5, f32::NAN, 0.0, -1.0, 1.0),
        "non-finite constant parameter",
        4
    );
}

#[test]
fn test_norm_affine_bounds_inf_var_returns_error() {
    assert_bounds_error!(
        norm_affine_output_bounds(0.0, f32::INFINITY, 1e-5, 1.0, 0.0, -1.0, 1.0),
        "non-finite constant parameter",
        2
    );
}

#[test]
fn test_norm_affine_bounds_nan_x_lower_returns_error() {
    assert_bounds_error!(
        norm_affine_output_bounds(0.0, 1.0, 1e-5, 1.0, 0.0, f64::NAN, 1.0),
        "non-finite output bounds"
    );
}

#[test]
fn test_norm_affine_inverted_x_bounds_returns_error() {
    assert_bounds_error!(
        norm_affine_output_bounds(0.0, 1.0, 1e-5, 1.0, 0.0, 5.0, 1.0),
        "inverted bounds"
    );
}

#[test]
fn test_norm_affine_bounds_negative_denom_returns_error() {
    // var + eps = -1.0 + 0.5 = -0.5 <= 0
    assert_bounds_error!(
        norm_affine_output_bounds(0.0, -1.0, 0.5, 1.0, 0.0, -1.0, 1.0),
        "non-finite"
    );
}

// ======================== norm_affine_output_bounds correctness ========================

#[test]
fn test_norm_affine_bounds_basic_positive_coeff() {
    // mean=0.0, var=1.0, eps=1e-5, gamma=3.0, beta=0.5, x in [-1.0, 2.0]
    let inv_std = 1.0 / (1.0_f64 + 1e-5).sqrt();
    let a = (-1.0 - 0.0) * inv_std * 3.0 + 0.5;
    let b = (2.0 - 0.0) * inv_std * 3.0 + 0.5;
    let (lo, hi) = norm_affine_output_bounds(0.0, 1.0, 1e-5, 3.0, 0.5, -1.0, 2.0).unwrap();
    assert!((lo - a).abs() < 1e-6, "lower bound should be {a}, got {lo}");
    assert!((hi - b).abs() < 1e-6, "upper bound should be {b}, got {hi}");
}

#[test]
fn test_norm_affine_bounds_negative_gamma_flips() {
    // mean=3.0, var=4.0, eps=0.0, gamma=-2.0, beta=0.0, x in [0.0, 1.0]
    let (lo, hi) = norm_affine_output_bounds(3.0, 4.0, 0.0, -2.0, 0.0, 0.0, 1.0).unwrap();
    assert!(
        (lo - 2.0).abs() < 1e-10,
        "negative gamma: lower should be 2.0, got {lo}"
    );
    assert!(
        (hi - 3.0).abs() < 1e-10,
        "negative gamma: upper should be 3.0, got {hi}"
    );
}

#[test]
fn test_norm_affine_bounds_zero_gamma() {
    // gamma=0.0, output = beta = 7.0 for all x
    let (lo, hi) = norm_affine_output_bounds(5.0, 2.0, 0.1, 0.0, 7.0, -100.0, 100.0).unwrap();
    assert!(
        (lo - 7.0).abs() < 1e-10,
        "zero gamma: lower should be 7.0, got {lo}"
    );
    assert!(
        (hi - 7.0).abs() < 1e-10,
        "zero gamma: upper should be 7.0, got {hi}"
    );
}
