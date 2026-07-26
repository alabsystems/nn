// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-verification tests: nn-dsl f32 bounds vs nn-verify f64 bounds (#463).
//!
//! Every kernel analytical bounds function exists in two independent
//! implementations at different precisions:
//!
//! - **nn-dsl (f32):** runtime bounds checking (`*_scalar_bounds`)
//! - **nn-verify (f64):** SMT output bounds (`*_output_bounds`)
//!
//! These tests verify that the f64 bounds contain the f32 bounds for
//! representative inputs, catching formula drift between the two copies.

use nn_dsl::{
    rope_cos_scalar_bounds, rope_sin_scalar_bounds, silu_mul_scalar_bounds, snake_scalar_bounds,
};

use super::prove_dispatch::{rope_output_bounds, silu_mul_output_bounds};
use crate::ay::snake_uf::snake_output_bounds;

/// Assert that f64 bounds (from nn-verify) contain f32 bounds (from nn-dsl).
///
/// The f64 version may be strictly wider due to higher precision in the
/// computation. The invariant is: `f64_lo <= f32_lo` and `f64_hi >= f32_hi`.
fn assert_f64_contains_f32(
    kernel: &str,
    f32_bounds: (f32, f32),
    f64_bounds: (f64, f64),
    tolerance: f64,
) {
    let (f32_lo, f32_hi) = f32_bounds;
    let (f64_lo, f64_hi) = f64_bounds;
    assert!(
        f64_lo <= f64::from(f32_lo) + tolerance,
        "{kernel}: f64 lower ({f64_lo}) must be <= f32 lower ({f32_lo}) + tol ({tolerance})"
    );
    assert!(
        f64_hi >= f64::from(f32_hi) - tolerance,
        "{kernel}: f64 upper ({f64_hi}) must be >= f32 upper ({f32_hi}) - tol ({tolerance})"
    );
}

// --- silu_mul cross-verification ---

/// Test cases for silu_mul: (x_lo, x_hi, up_const).
/// The f32 version takes (x_lo, x_hi, up_lo, up_hi) with up as a point
/// interval; the f64 version takes (up_const, x_lower, x_upper).
const SILU_MUL_CASES: &[(f32, f32, f32)] = &[
    // Positive x range, positive up
    (0.5, 2.0, 1.0),
    // Negative x range (spans silu minimum at ~-1.278)
    (-3.0, -0.5, 1.0),
    // Range spanning zero and silu minimum
    (-2.0, 2.0, 1.0),
    // Negative up (flips bounds)
    (-2.0, 2.0, -1.5),
    // Zero x bounds (point input)
    (0.0, 0.0, 1.0),
    // Large positive x
    (5.0, 10.0, 2.0),
    // Small positive up near zero
    (-1.0, 1.0, 0.01),
];

#[test]
fn test_silu_mul_f32_f64_agreement() {
    for &(x_lo, x_hi, up) in SILU_MUL_CASES {
        // f32 bounds: (x_lo, x_hi, up_lo, up_hi) with up as point interval
        let f32_result = silu_mul_scalar_bounds(x_lo, x_hi, up, up)
            .unwrap_or_else(|e| panic!("f32 silu_mul failed for x=[{x_lo},{x_hi}] up={up}: {e}"));

        // f64 bounds: (up_const, x_lower, x_upper)
        let f64_result = silu_mul_output_bounds(f64::from(up), f64::from(x_lo), f64::from(x_hi))
            .unwrap_or_else(|e| panic!("f64 silu_mul failed for x=[{x_lo},{x_hi}] up={up}: {e}"));

        assert_f64_contains_f32("silu_mul", f32_result, f64_result, 1e-6);
    }
}

// --- snake cross-verification ---

/// Test cases for snake: (x_lo, x_hi, alpha).
/// The f32 version takes (x_lo, x_hi, alpha_lo, alpha_hi) with alpha as
/// a point interval; the f64 version takes (x_lo, x_hi, alpha).
const SNAKE_CASES: &[(f32, f32, f32)] = &[
    // Standard case
    (0.0, 1.0, 1.0),
    // Negative x range
    (-2.0, -0.5, 1.0),
    // Range spanning zero
    (-1.0, 1.0, 2.0),
    // Large alpha (tight sin^2 envelope)
    (-1.0, 1.0, 10.0),
    // Small alpha (wide sin^2 envelope)
    (-1.0, 1.0, 0.1),
    // Large x range
    (-10.0, 10.0, 1.0),
    // Point input
    (0.0, 0.0, 1.0),
];

#[test]
fn test_snake_f32_f64_agreement() {
    for &(x_lo, x_hi, alpha) in SNAKE_CASES {
        // f32 bounds: (x_lo, x_hi, alpha_lo, alpha_hi) with alpha as point interval
        let f32_result = snake_scalar_bounds(x_lo, x_hi, alpha, alpha).unwrap_or_else(|e| {
            panic!("f32 snake failed for x=[{x_lo},{x_hi}] alpha={alpha}: {e}")
        });

        // f64 bounds: (x_lo, x_hi, alpha)
        let f64_result = snake_output_bounds(f64::from(x_lo), f64::from(x_hi), f64::from(alpha))
            .unwrap_or_else(|e| {
                panic!("f64 snake failed for x=[{x_lo},{x_hi}] alpha={alpha}: {e}")
            });

        assert_f64_contains_f32("snake", f32_result, f64_result, 1e-6);
    }
}

// --- rope_cos cross-verification ---

/// Test cases for rope_cos: (x0_lo, x0_hi, x1_const, freq_const).
/// The f32 version takes (x0_lo, x0_hi, x1_lo, x1_hi, freq_lo, freq_hi)
/// with x1 and freq as point intervals.
/// The f64 version takes (x1_const, freq_const, x0_lower, x0_upper, bounds_fn).
const ROPE_CASES: &[(f32, f32, f32, f32)] = &[
    // Standard case
    (0.0, 1.0, 0.5, 1.0),
    // Negative x0 range
    (-2.0, -0.5, 1.0, 0.5),
    // Range spanning zero
    (-1.0, 1.0, 0.0, 1.0),
    // Large freq (many rotations)
    (-1.0, 1.0, 1.0, 10.0),
    // Zero freq (no rotation: cos=1, sin=0)
    (-1.0, 1.0, 0.5, 0.0),
    // Negative x1
    (-1.0, 1.0, -2.0, 1.0),
    // Point x0 input
    (0.5, 0.5, 1.0, 1.0),
];

#[test]
fn test_rope_cos_f32_f64_agreement() {
    for &(x0_lo, x0_hi, x1, freq) in ROPE_CASES {
        // f32 bounds: (x0_lo, x0_hi, x1_lo, x1_hi, freq_lo, freq_hi)
        let f32_result =
            rope_cos_scalar_bounds(x0_lo, x0_hi, x1, x1, freq, freq).unwrap_or_else(|e| {
                panic!("f32 rope_cos failed for x0=[{x0_lo},{x0_hi}] x1={x1} freq={freq}: {e}")
            });

        // f64 bounds: (x1_const, freq_const, x0_lower, x0_upper, bounds_fn)
        let f64_result = rope_output_bounds(x1, freq, x0_lo, x0_hi, rope_cos_scalar_bounds)
            .unwrap_or_else(|e| {
                panic!("f64 rope_cos failed for x0=[{x0_lo},{x0_hi}] x1={x1} freq={freq}: {e}")
            });

        assert_f64_contains_f32("rope_cos", f32_result, f64_result, 1e-6);
    }
}

#[test]
fn test_rope_sin_f32_f64_agreement() {
    for &(x0_lo, x0_hi, x1, freq) in ROPE_CASES {
        // f32 bounds: (x0_lo, x0_hi, x1_lo, x1_hi, freq_lo, freq_hi)
        let f32_result =
            rope_sin_scalar_bounds(x0_lo, x0_hi, x1, x1, freq, freq).unwrap_or_else(|e| {
                panic!("f32 rope_sin failed for x0=[{x0_lo},{x0_hi}] x1={x1} freq={freq}: {e}")
            });

        // f64 bounds: (x1_const, freq_const, x0_lower, x0_upper, bounds_fn)
        let f64_result = rope_output_bounds(x1, freq, x0_lo, x0_hi, rope_sin_scalar_bounds)
            .unwrap_or_else(|e| {
                panic!("f64 rope_sin failed for x0=[{x0_lo},{x0_hi}] x1={x1} freq={freq}: {e}")
            });

        assert_f64_contains_f32("rope_sin", f32_result, f64_result, 1e-6);
    }
}
