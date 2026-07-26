// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for bounds_adain_snake dispatch function.
//! This kernel composes adain + snake: snake(adain(x, mu, var, gamma, beta, eps), alpha).
//!
//! Parameter ordering: cp[0]=mu, cp[1]=var, cp[2]=gamma, cp[3]=beta, cp[4]=alpha, cp[5]=eps.
//! Per #425: both error-path AND numerical correctness tests required.

use super::*;

// ======================== bounds_adain_snake error paths ========================

#[test]
fn test_adain_snake_too_few_params_returns_error() {
    // Requires 6 params, provide only 5.
    let cp = [0.0_f32, 1.0, 1.0, 0.0, 1.0];
    let err = bounds_adain_snake(&cp, -1.0, 1.0).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("bounds_adain_snake"),
        "error should mention function name, got: {msg}"
    );
}

#[test]
fn test_adain_snake_nan_mu_returns_error() {
    // cp[0]=mu is NaN — adain_output_bounds should reject.
    let cp = [f32::NAN, 1.0, 1.0, 0.0, 1.0, 1e-5];
    assert_bounds_error!(
        bounds_adain_snake(&cp, -1.0, 1.0),
        "non-finite constant parameter",
        1
    );
}

#[test]
fn test_adain_snake_inf_var_returns_error() {
    let cp = [0.0, f32::INFINITY, 1.0, 0.0, 1.0, 1e-5];
    assert_bounds_error!(
        bounds_adain_snake(&cp, -1.0, 1.0),
        "non-finite constant parameter",
        2
    );
}

#[test]
fn test_adain_snake_nan_gamma_returns_error() {
    let cp = [0.0, 1.0, f32::NAN, 0.0, 1.0, 1e-5];
    assert_bounds_error!(
        bounds_adain_snake(&cp, -1.0, 1.0),
        "non-finite constant parameter",
        3
    );
}

#[test]
fn test_adain_snake_inf_beta_returns_error() {
    let cp = [0.0, 1.0, 1.0, f32::INFINITY, 1.0, 1e-5];
    assert_bounds_error!(
        bounds_adain_snake(&cp, -1.0, 1.0),
        "non-finite constant parameter",
        4
    );
}

#[test]
fn test_adain_snake_nan_eps_returns_error() {
    // cp[5]=eps is NaN — note: eps is at index 5, maps to adain param 5.
    let cp = [0.0, 1.0, 1.0, 0.0, 1.0, f32::NAN];
    assert_bounds_error!(
        bounds_adain_snake(&cp, -1.0, 1.0),
        "non-finite constant parameter",
        5
    );
}

#[test]
fn test_adain_snake_inverted_x_bounds_returns_error() {
    let cp = [0.0, 1.0, 1.0, 0.0, 1.0, 1e-5];
    assert_bounds_error!(bounds_adain_snake(&cp, 1.0, -1.0), "inverted bounds");
}

#[test]
fn test_adain_snake_negative_denom_returns_error() {
    // var + eps = -1.0 + 0.5 = -0.5 <= 0 → sqrt undefined
    let cp = [0.0, -1.0, 1.0, 0.0, 1.0, 0.5];
    let result = bounds_adain_snake(&cp, -1.0, 1.0);
    assert!(result.is_err(), "negative denominator should cause error");
}

// ======================== bounds_adain_snake numerical correctness ========================

#[test]
fn test_adain_snake_identity_composition() {
    // AdaIN with mu=0, var=1, gamma=1, beta=0, eps=0 → identity (output = input).
    // Snake with alpha=1 on [-1, 1]:
    //   snake_lo = -1.0, snake_hi = 1.0 + 1/1 = 2.0
    let cp = [0.0_f32, 1.0, 1.0, 0.0, 1.0, 0.0];
    let (lo, hi) = bounds_adain_snake(&cp, -1.0, 1.0)
        .expect("identity composition should succeed")
        .expect("identity composition should return Some bounds");
    assert!(
        (lo - -1.0).abs() < 1e-10,
        "identity adain + snake(alpha=1): lower should be -1.0, got {lo}"
    );
    assert!(
        (hi - 2.0).abs() < 1e-10,
        "identity adain + snake(alpha=1): upper should be 2.0, got {hi}"
    );
}

#[test]
fn test_adain_snake_with_gamma_and_beta() {
    // AdaIN: mu=1.0, var=4.0, gamma=2.0, beta=0.5, eps=0.0, x in [-1.0, 3.0]
    //   inv_std = 1/sqrt(4.0) = 0.5
    //   adain(x) = gamma * (x - mu) * inv_std + beta = 2.0 * (x-1.0) * 0.5 + 0.5
    //   For x=-1.0: 2.0 * (-2.0) * 0.5 + 0.5 = -1.5
    //   For x=3.0:  2.0 * (2.0) * 0.5 + 0.5 = 2.5
    //   adain bounds: [-1.5, 2.5]
    //
    // Snake with alpha=2.0 on [-1.5, 2.5]:
    //   snake_lo = -1.5, snake_hi = 2.5 + 1/2 = 3.0
    let cp = [1.0_f32, 4.0, 2.0, 0.5, 2.0, 0.0];
    let (lo, hi) = bounds_adain_snake(&cp, -1.0, 3.0)
        .expect("gamma+beta composition should succeed")
        .expect("gamma+beta composition should return Some bounds");
    assert!(
        (lo - -1.5).abs() < 1e-10,
        "adain+snake: lower should be -1.5, got {lo}"
    );
    assert!(
        (hi - 3.0).abs() < 1e-10,
        "adain+snake: upper should be 3.0, got {hi}"
    );
}

#[test]
fn test_adain_snake_negative_gamma_flips() {
    // AdaIN: mu=0.0, var=1.0, gamma=-3.0, beta=0.0, eps=0.0, x in [1.0, 2.0]
    //   adain(x) = -3.0 * x + 0.0 → for x=1.0: -3.0, x=2.0: -6.0
    //   Since gamma is negative, adain flips: bounds = [-6.0, -3.0]
    //
    // Snake with alpha=1.0 on [-6.0, -3.0]:
    //   snake_lo = -6.0, snake_hi = -3.0 + 1/1 = -2.0
    let cp = [0.0_f32, 1.0, -3.0, 0.0, 1.0, 0.0];
    let (lo, hi) = bounds_adain_snake(&cp, 1.0, 2.0)
        .expect("negative gamma composition should succeed")
        .expect("negative gamma composition should return Some bounds");
    assert!(
        (lo - -6.0).abs() < 1e-10,
        "negative gamma + snake: lower should be -6.0, got {lo}"
    );
    assert!(
        (hi - -2.0).abs() < 1e-10,
        "negative gamma + snake: upper should be -2.0, got {hi}"
    );
}

#[test]
fn test_adain_snake_x_equals_mu() {
    // x in [5.0, 5.0], mu=5.0 → adain output = beta = 3.0 (point)
    // Snake alpha=4 on [3.0, 3.0]:
    //   snake_lo = 3.0, snake_hi = 3.0 + 1/4 = 3.25
    let cp = [5.0_f32, 1.0, 2.0, 3.0, 4.0, 1e-5];
    let (lo, hi) = bounds_adain_snake(&cp, 5.0, 5.0)
        .expect("x=mu composition should succeed")
        .expect("x=mu composition should return Some bounds");
    assert!(
        (lo - 3.0).abs() < 1e-4,
        "x=mu: lower should be ~3.0, got {lo}"
    );
    assert!(
        (hi - 3.25).abs() < 1e-4,
        "x=mu: upper should be ~3.25, got {hi}"
    );
}

#[test]
fn test_adain_snake_large_alpha_tight_snake() {
    // Large alpha → 1/alpha is small → snake barely changes bounds.
    // AdaIN identity: mu=0, var=1, gamma=1, beta=0, eps=0
    // x in [0.0, 1.0], alpha=100.0
    //   adain bounds: [0.0, 1.0] (identity)
    //   snake bounds: [0.0, 1.0 + 1/100] = [0.0, 1.01]
    let cp = [0.0_f32, 1.0, 1.0, 0.0, 100.0, 0.0];
    let (lo, hi) = bounds_adain_snake(&cp, 0.0, 1.0)
        .expect("large alpha composition should succeed")
        .expect("large alpha composition should return Some bounds");
    assert!(
        (lo - 0.0).abs() < 1e-10,
        "large alpha: lower should be 0.0, got {lo}"
    );
    assert!(
        (hi - 1.01).abs() < 1e-10,
        "large alpha: upper should be 1.01, got {hi}"
    );
}

#[test]
fn test_adain_snake_with_nonzero_eps() {
    // mu=0.0, var=1.0, gamma=1.0, beta=0.0, eps=3.0
    //   inv_std = 1/sqrt(1.0+3.0) = 0.5
    //   adain bounds for x in [0.0, 4.0]: [0.0, 2.0]
    // Snake alpha=1.0 on [0.0, 2.0]:
    //   snake_lo = 0.0, snake_hi = 2.0 + 1.0 = 3.0
    let cp = [0.0_f32, 1.0, 1.0, 0.0, 1.0, 3.0];
    let (lo, hi) = bounds_adain_snake(&cp, 0.0, 4.0)
        .expect("nonzero eps composition should succeed")
        .expect("nonzero eps composition should return Some bounds");
    assert!(
        (lo - 0.0).abs() < 1e-10,
        "nonzero eps: lower should be 0.0, got {lo}"
    );
    assert!(
        (hi - 3.0).abs() < 1e-10,
        "nonzero eps: upper should be 3.0, got {hi}"
    );
}
