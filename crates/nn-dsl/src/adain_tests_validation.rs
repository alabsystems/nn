// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! AdaIN input validation and error rejection tests.
//!
//! Tests for non-finite input rejection (NaN, Inf), negative denominator
//! rejection, and extreme magnitude output guards. Extracted from
//! `adain_tests.rs` (#1420).

use super::*;

// ── AC2: adain_scalar rejects var_val + eps <= 0 ──

#[test]
fn test_adain_scalar_rejects_negative_denominator() {
    // var_val + eps = -1.0 + 0.0 = -1.0 < 0 → sqrt would be NaN
    let result = adain_scalar(1.0, 0.0, -1.0, 1.0, 0.0, 0.0);
    assert!(result.is_err(), "should reject var_val + eps = -1.0");

    // var_val + eps = 0.0 + 0.0 = 0.0 → recip of sqrt(0) = Inf
    let result = adain_scalar(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    assert!(result.is_err(), "should reject var_val + eps = 0.0");

    // var_val + eps = -0.5 + 0.1 = -0.4 < 0
    let result = adain_scalar(1.0, 0.0, -0.5, 1.0, 0.0, 0.1);
    assert!(result.is_err(), "should reject var_val + eps = -0.4");

    // NaN eps → not finite
    let result = adain_scalar(1.0, 0.0, 1.0, 1.0, 0.0, f32::NAN);
    assert!(result.is_err(), "should reject NaN eps");

    // Infinity eps → not finite
    let result = adain_scalar(1.0, 0.0, 1.0, 1.0, 0.0, f32::INFINITY);
    assert!(result.is_err(), "should reject Inf eps");
}

// ── AC2: fused ref also rejects negative denominator ──

#[test]
fn test_adain_snake_fused_scalar_rejects_negative_denominator() {
    let result = adain_snake_fused_scalar(1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 0.0);
    assert!(result.is_err(), "fused should reject var_val + eps = -1.0");
}

// ── AC1/AC3: adain_scalar rejects non-finite inputs ──

#[test]
fn test_adain_scalar_rejects_nan_x() {
    let err = adain_scalar(f32::NAN, 0.0, 1.0, 1.0, 0.0, 1e-5).expect_err("NaN x should fail");
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "x", .. }),
        "expected NonFiniteInput for x, got {err:?}"
    );
}

#[test]
fn test_adain_scalar_rejects_nan_mu() {
    let err = adain_scalar(1.0, f32::NAN, 1.0, 1.0, 0.0, 1e-5).expect_err("NaN mu should fail");
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "mu", .. }),
        "expected NonFiniteInput for mu, got {err:?}"
    );
}

#[test]
fn test_adain_scalar_rejects_inf_gamma() {
    let err =
        adain_scalar(1.0, 0.0, 1.0, f32::INFINITY, 0.0, 1e-5).expect_err("Inf gamma should fail");
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "gamma", .. }),
        "expected NonFiniteInput for gamma, got {err:?}"
    );
}

#[test]
fn test_adain_scalar_rejects_neg_inf_beta() {
    let err = adain_scalar(1.0, 0.0, 1.0, 1.0, f32::NEG_INFINITY, 1e-5)
        .expect_err("-Inf beta should fail");
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "beta", .. }),
        "expected NonFiniteInput for beta, got {err:?}"
    );
}

#[test]
fn test_adain_scalar_rejects_nan_var_val() {
    let err =
        adain_scalar(1.0, 0.0, f32::NAN, 1.0, 0.0, 1e-5).expect_err("NaN var_val should fail");
    assert!(
        matches!(
            err,
            KernelError::NonFiniteInput {
                name: "var_val",
                ..
            }
        ),
        "expected NonFiniteInput for var_val, got {err:?}"
    );
}

// ── AC2: adain_snake_fused_scalar rejects non-finite alpha ──

#[test]
fn test_adain_snake_fused_scalar_rejects_nan_alpha() {
    let err = adain_snake_fused_scalar(1.0, 0.0, 1.0, 1.0, 0.0, f32::NAN, 1e-5)
        .expect_err("NaN alpha should fail");
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "alpha", .. }),
        "expected NonFiniteInput for alpha, got {err:?}"
    );
}

#[test]
fn test_adain_snake_fused_scalar_rejects_inf_alpha() {
    let err = adain_snake_fused_scalar(1.0, 0.0, 1.0, 1.0, 0.0, f32::INFINITY, 1e-5)
        .expect_err("Inf alpha should fail");
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "alpha", .. }),
        "expected NonFiniteInput for alpha, got {err:?}"
    );
}

// ── AC2: fused propagates non-finite input errors from adain_scalar ──

#[test]
fn test_adain_snake_fused_scalar_rejects_nan_x() {
    let result = adain_snake_fused_scalar(f32::NAN, 0.0, 1.0, 1.0, 0.0, 1.0, 1e-5);
    assert!(result.is_err(), "fused should reject NaN x");
}

// ── AC4: output guard catches non-finite results from extreme magnitudes ──

#[test]
fn test_adain_scalar_rejects_extreme_magnitude_output() {
    // gamma * (x - mu) can overflow to Inf with extreme values
    let result = adain_scalar(f32::MAX, f32::MIN, 1e-8, f32::MAX, 0.0, 1e-5);
    assert!(
        result.is_err(),
        "extreme magnitudes producing Inf output should fail"
    );
}
