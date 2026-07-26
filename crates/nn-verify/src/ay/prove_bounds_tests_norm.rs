// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for instance_norm_output_bounds and adain_output_bounds.
//! Split from prove_bounds_tests.rs (#453).

use super::*;

// ======================== instance_norm_output_bounds error paths ========================

#[test]
fn test_instance_norm_bounds_nan_mean_returns_error() {
    assert_bounds_error!(
        instance_norm_output_bounds(f32::NAN, 1.0, 1e-5, -1.0, 1.0),
        "non-finite constant parameter",
        1
    );
}

#[test]
fn test_instance_norm_bounds_inf_var_returns_error() {
    assert_bounds_error!(
        instance_norm_output_bounds(0.0, f32::INFINITY, 1e-5, -1.0, 1.0),
        "non-finite constant parameter",
        2
    );
}

#[test]
fn test_instance_norm_bounds_nan_eps_returns_error() {
    assert_bounds_error!(
        instance_norm_output_bounds(0.0, 1.0, f32::NAN, -1.0, 1.0),
        "non-finite constant parameter",
        3
    );
}

#[test]
fn test_instance_norm_bounds_inf_x_lower_returns_error() {
    assert_bounds_error!(
        instance_norm_output_bounds(0.0, 1.0, 1e-5, f64::INFINITY, 1.0),
        "non-finite output bounds"
    );
}

#[test]
fn test_instance_norm_bounds_nan_x_upper_returns_error() {
    assert_bounds_error!(
        instance_norm_output_bounds(0.0, 1.0, 1e-5, -1.0, f64::NAN),
        "non-finite output bounds"
    );
}

#[test]
fn test_instance_norm_inverted_x_bounds_returns_error() {
    assert_bounds_error!(
        instance_norm_output_bounds(0.0, 1.0, 1e-5, 0.5, 0.01),
        "inverted bounds"
    );
}

#[test]
fn test_instance_norm_bounds_negative_denom_returns_error() {
    // var + eps = -1.0 + 0.5 = -0.5 <= 0
    assert_bounds_error!(
        instance_norm_output_bounds(0.0, -1.0, 0.5, -1.0, 1.0),
        "non-finite"
    );
}

// ======================== instance_norm_output_bounds correctness ========================

#[test]
fn test_instance_norm_bounds_basic() {
    // mean=1.0, var=1.0, eps=0.0, inv_std = 1.0, x in [-1.0, 3.0]
    let (lo, hi) = instance_norm_output_bounds(1.0, 1.0, 0.0, -1.0, 3.0).unwrap();
    assert!((lo - -2.0).abs() < 1e-10, "lower should be -2.0, got {lo}");
    assert!((hi - 2.0).abs() < 1e-10, "upper should be 2.0, got {hi}");
}

#[test]
fn test_instance_norm_bounds_high_variance() {
    // mean=0.0, var=4.0, eps=0.0, inv_std = 0.5, x in [2.0, 6.0]
    let (lo, hi) = instance_norm_output_bounds(0.0, 4.0, 0.0, 2.0, 6.0).unwrap();
    assert!((lo - 1.0).abs() < 1e-10, "lower should be 1.0, got {lo}");
    assert!((hi - 3.0).abs() < 1e-10, "upper should be 3.0, got {hi}");
}

#[test]
fn test_instance_norm_bounds_x_equals_mean() {
    // x in [5.0, 5.0], mean=5.0, output = 0.0
    let (lo, hi) = instance_norm_output_bounds(5.0, 1.0, 1e-5, 5.0, 5.0).unwrap();
    assert!(lo.abs() < 1e-10, "x=mean: lower should be 0.0, got {lo}");
    assert!(hi.abs() < 1e-10, "x=mean: upper should be 0.0, got {hi}");
}

// ======================== adain_output_bounds error paths ========================

#[test]
fn test_adain_bounds_nan_mu_returns_error() {
    assert_bounds_error!(
        adain_output_bounds(f32::NAN, 1.0, 1.0, 0.0, 1e-5, -1.0, 1.0),
        "non-finite constant parameter",
        1
    );
}

#[test]
fn test_adain_bounds_inf_var_returns_error() {
    assert_bounds_error!(
        adain_output_bounds(0.0, f32::INFINITY, 1.0, 0.0, 1e-5, -1.0, 1.0),
        "non-finite constant parameter",
        2
    );
}

#[test]
fn test_adain_bounds_nan_gamma_returns_error() {
    assert_bounds_error!(
        adain_output_bounds(0.0, 1.0, f32::NAN, 0.0, 1e-5, -1.0, 1.0),
        "non-finite constant parameter",
        3
    );
}

#[test]
fn test_adain_bounds_inf_beta_returns_error() {
    assert_bounds_error!(
        adain_output_bounds(0.0, 1.0, 1.0, f32::INFINITY, 1e-5, -1.0, 1.0),
        "non-finite constant parameter",
        4
    );
}

#[test]
fn test_adain_bounds_nan_eps_returns_error() {
    assert_bounds_error!(
        adain_output_bounds(0.0, 1.0, 1.0, 0.0, f32::NAN, -1.0, 1.0),
        "non-finite constant parameter",
        5
    );
}

#[test]
fn test_adain_bounds_nan_x_lower_returns_error() {
    assert_bounds_error!(
        adain_output_bounds(0.0, 1.0, 1.0, 0.0, 1e-5, f64::NAN, 1.0),
        "non-finite output bounds"
    );
}

#[test]
fn test_adain_bounds_inf_x_upper_returns_error() {
    assert_bounds_error!(
        adain_output_bounds(0.0, 1.0, 1.0, 0.0, 1e-5, -1.0, f64::INFINITY),
        "non-finite output bounds"
    );
}

#[test]
fn test_adain_inverted_x_bounds_returns_error() {
    assert_bounds_error!(
        adain_output_bounds(0.0, 1.0, 1.0, 0.0, 1e-5, 1.0, 0.01),
        "inverted bounds"
    );
}

#[test]
fn test_adain_bounds_negative_denom_returns_error() {
    // var + eps = -1.0 + 0.5 = -0.5 <= 0
    assert_bounds_error!(
        adain_output_bounds(0.0, -1.0, 1.0, 0.0, 0.5, -1.0, 1.0),
        "non-finite"
    );
}

// ======================== adain_output_bounds correctness ========================

#[test]
fn test_adain_bounds_basic_positive_gamma() {
    // mu=1.0, var=4.0, gamma=2.0, beta=0.5, eps=0.0, x in [-1.0, 3.0]
    let (lo, hi) = adain_output_bounds(1.0, 4.0, 2.0, 0.5, 0.0, -1.0, 3.0).unwrap();
    assert!((lo - -1.5).abs() < 1e-10, "lower should be -1.5, got {lo}");
    assert!((hi - 2.5).abs() < 1e-10, "upper should be 2.5, got {hi}");
}

#[test]
fn test_adain_bounds_negative_gamma_flips() {
    // mu=0.0, var=1.0, gamma=-3.0, beta=0.0, eps=0.0, x in [1.0, 2.0]
    let (lo, hi) = adain_output_bounds(0.0, 1.0, -3.0, 0.0, 0.0, 1.0, 2.0).unwrap();
    assert!(
        (lo - -6.0).abs() < 1e-10,
        "negative gamma: lower should be -6.0, got {lo}"
    );
    assert!(
        (hi - -3.0).abs() < 1e-10,
        "negative gamma: upper should be -3.0, got {hi}"
    );
}

#[test]
fn test_adain_bounds_x_at_mu() {
    // x in [5.0, 5.0], mu=5.0, output = beta = 3.0
    let (lo, hi) = adain_output_bounds(5.0, 1.0, 2.0, 3.0, 1e-5, 5.0, 5.0).unwrap();
    assert!(
        (lo - 3.0).abs() < 1e-10,
        "x=mu: lower should be 3.0, got {lo}"
    );
    assert!(
        (hi - 3.0).abs() < 1e-10,
        "x=mu: upper should be 3.0, got {hi}"
    );
}

#[test]
fn test_adain_bounds_with_nonzero_eps() {
    // mu=0.0, var=1.0, gamma=1.0, beta=0.0, eps=3.0
    // inv_std = 1/sqrt(4.0) = 0.5, x in [0.0, 4.0]
    let (lo, hi) = adain_output_bounds(0.0, 1.0, 1.0, 0.0, 3.0, 0.0, 4.0).unwrap();
    assert!((lo - 0.0).abs() < 1e-10, "lower should be 0.0, got {lo}");
    assert!((hi - 2.0).abs() < 1e-10, "upper should be 2.0, got {hi}");
}
