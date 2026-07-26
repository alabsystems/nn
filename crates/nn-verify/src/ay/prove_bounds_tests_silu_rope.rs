// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for silu_mul_output_bounds and rope_output_bounds.
//! Split from prove_bounds_tests.rs (#453).

use super::*;

// ======================== silu_mul_output_bounds error paths ========================

#[test]
fn test_silu_mul_bounds_nan_up_const_returns_error() {
    assert_bounds_error!(
        silu_mul_output_bounds(f64::NAN, -1.0, 1.0),
        "non-finite constant parameter",
        1
    );
}

#[test]
fn test_silu_mul_bounds_nan_x_lower_returns_error() {
    assert_bounds_error!(
        silu_mul_output_bounds(1.0, f64::NAN, 1.0),
        "non-finite output bounds"
    );
}

#[test]
fn test_silu_mul_bounds_inf_x_upper_returns_error() {
    assert_bounds_error!(
        silu_mul_output_bounds(1.0, -1.0, f64::INFINITY),
        "non-finite output bounds"
    );
}

#[test]
fn test_silu_mul_inverted_bounds_rejected() {
    assert_bounds_error!(silu_mul_output_bounds(1.0, 5.0, -5.0), "inverted bounds");
}

// ======================== silu_mul_output_bounds correctness ========================

#[test]
fn test_silu_mul_bounds_x_range_positive_up() {
    // silu_mul(x, up) = silu(x) * up, x in [-1.0, 2.0], up=3.0
    let silu_neg1 = -1.0_f64 / (1.0 + 1.0_f64.exp());
    let silu_2 = 2.0_f64 / (1.0 + (-2.0_f64).exp());
    let (lo, hi) = silu_mul_output_bounds(3.0, -1.0, 2.0).unwrap();
    assert!(
        (lo - silu_neg1 * 3.0).abs() < 1e-10,
        "lower should be {}, got {lo}",
        silu_neg1 * 3.0
    );
    assert!(
        (hi - silu_2 * 3.0).abs() < 1e-10,
        "upper should be {}, got {hi}",
        silu_2 * 3.0
    );
}

#[test]
fn test_silu_mul_bounds_negative_up_flips_order() {
    let silu_1 = 1.0_f64 / (1.0 + (-1.0_f64).exp());
    let silu_3 = 3.0_f64 / (1.0 + (-3.0_f64).exp());
    let (lo, hi) = silu_mul_output_bounds(-2.0, 1.0, 3.0).unwrap();
    assert!(
        (lo - silu_3 * -2.0).abs() < 1e-10,
        "negative up: lower should be {}, got {lo}",
        silu_3 * -2.0
    );
    assert!(
        (hi - silu_1 * -2.0).abs() < 1e-10,
        "negative up: upper should be {}, got {hi}",
        silu_1 * -2.0
    );
}

#[test]
fn test_silu_mul_bounds_zero_up() {
    let (lo, hi) = silu_mul_output_bounds(0.0, -100.0, 100.0).unwrap();
    assert!(lo.abs() < 1e-10, "zero up: lower should be 0.0, got {lo}");
    assert!(hi.abs() < 1e-10, "zero up: upper should be 0.0, got {hi}");
}

#[test]
fn test_silu_mul_bounds_point_interval() {
    let silu_2 = 2.0_f64 / (1.0 + (-2.0_f64).exp());
    let (lo, hi) = silu_mul_output_bounds(1.0, 2.0, 2.0).unwrap();
    assert!(
        (lo - silu_2).abs() < 1e-10,
        "point: lower should be {silu_2}, got {lo}"
    );
    assert!(
        (hi - silu_2).abs() < 1e-10,
        "point: upper should be {silu_2}, got {hi}"
    );
}

#[test]
fn test_silu_mul_bounds_spanning_argmin_includes_global_minimum() {
    fn silu_f64(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }
    const SILU_ARGMIN: f64 = -1.278_464_5;

    let silu_at_argmin = silu_f64(SILU_ARGMIN);
    let silu_at_hi = silu_f64(1.0);

    let (lo, hi) = silu_mul_output_bounds(2.0, -3.0, 1.0).unwrap();
    assert!(
        (lo - silu_at_argmin * 2.0).abs() < 1e-10,
        "argmin-spanning: lower should be {}, got {lo}",
        silu_at_argmin * 2.0
    );
    assert!(
        (hi - silu_at_hi * 2.0).abs() < 1e-10,
        "argmin-spanning: upper should be {}, got {hi}",
        silu_at_hi * 2.0
    );
}

#[test]
fn test_silu_mul_bounds_spanning_argmin_negative_up() {
    fn silu_f64(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }
    const SILU_ARGMIN: f64 = -1.278_464_5;

    let silu_at_argmin = silu_f64(SILU_ARGMIN);
    let silu_at_hi = silu_f64(1.0);

    let (lo, hi) = silu_mul_output_bounds(-1.5, -3.0, 1.0).unwrap();
    let expected_lo = (silu_at_hi * -1.5).min(silu_at_argmin * -1.5);
    let expected_hi = (silu_at_hi * -1.5).max(silu_at_argmin * -1.5);
    assert!(
        (lo - expected_lo).abs() < 1e-10,
        "argmin-spanning neg up: lower should be {expected_lo}, got {lo}"
    );
    assert!(
        (hi - expected_hi).abs() < 1e-10,
        "argmin-spanning neg up: upper should be {expected_hi}, got {hi}"
    );
}

// ======================== rope_output_bounds error paths ========================

#[test]
fn test_rope_bounds_nan_x1_const_returns_error() {
    assert_bounds_error!(
        rope_output_bounds(f32::NAN, 1.0, 0.0, 1.0, nn_dsl::rope_cos_scalar_bounds),
        "non-finite constant parameter",
        1
    );
}

#[test]
fn test_rope_bounds_nan_freq_const_returns_error() {
    assert_bounds_error!(
        rope_output_bounds(1.0, f32::NAN, 0.0, 1.0, nn_dsl::rope_cos_scalar_bounds),
        "non-finite constant parameter",
        2
    );
}

#[test]
fn test_rope_bounds_inf_x0_lower_returns_error() {
    assert_bounds_error!(
        rope_output_bounds(
            1.0,
            1.0,
            f32::INFINITY,
            1.0,
            nn_dsl::rope_cos_scalar_bounds
        ),
        "non-finite output bounds"
    );
}

#[test]
fn test_rope_bounds_nan_x0_upper_returns_error() {
    assert_bounds_error!(
        rope_output_bounds(1.0, 1.0, 0.0, f32::NAN, nn_dsl::rope_sin_scalar_bounds),
        "non-finite output bounds"
    );
}

#[test]
fn test_rope_inverted_x0_bounds_returns_error() {
    assert_bounds_error!(
        rope_output_bounds(1.0, 1.0, 1.0, 0.0, nn_dsl::rope_cos_scalar_bounds),
        "inverted bounds"
    );
}

// ======================== rope_output_bounds correctness ========================

#[test]
fn test_rope_bounds_cos_point_freq_zero() {
    let (lo, hi) = rope_output_bounds(3.0, 0.0, 2.0, 5.0, nn_dsl::rope_cos_scalar_bounds).unwrap();
    assert!(
        (lo - 2.0).abs() < 1e-4,
        "freq=0 rope_cos: lower should be 2.0, got {lo}"
    );
    assert!(
        (hi - 5.0).abs() < 1e-4,
        "freq=0 rope_cos: upper should be 5.0, got {hi}"
    );
}

#[test]
fn test_rope_bounds_sin_point_freq_zero() {
    let (lo, hi) = rope_output_bounds(3.0, 0.0, 2.0, 5.0, nn_dsl::rope_sin_scalar_bounds).unwrap();
    assert!(
        (lo - 3.0).abs() < 1e-4,
        "freq=0 rope_sin: lower should be 3.0, got {lo}"
    );
    assert!(
        (hi - 3.0).abs() < 1e-4,
        "freq=0 rope_sin: upper should be 3.0, got {hi}"
    );
}

#[test]
fn test_rope_bounds_cos_point_x0_exercises_both_terms() {
    let freq = std::f32::consts::FRAC_PI_4;
    let (lo, hi) = rope_output_bounds(3.0, freq, 2.0, 2.0, nn_dsl::rope_cos_scalar_bounds)
        .expect("rope_cos at pi/4");
    let expected = 2.0_f64 * f64::from(freq).cos() - 3.0_f64 * f64::from(freq).sin();
    assert!(
        (lo - expected).abs() < 1e-3,
        "freq=pi/4 rope_cos: lower should be {expected:.6}, got {lo}"
    );
    assert!(
        (hi - expected).abs() < 1e-3,
        "freq=pi/4 rope_cos: upper should be {expected:.6}, got {hi}"
    );
}
