// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for ULP rounding functions and round_for_soundness.
//!
//! Extracted from `kani_bounds.rs` to stay under the 500-line file limit.
//! These bit-manipulation functions are critical to soundness rounding
//! (IntervalBounds::round_for_soundness) and must preserve infeasible
//! sentinels (+inf, -inf) used by mark_infeasible_all().

use crate::bounds::{next_down_f32, next_up_f32};

/// next_down_f32 returns a value strictly less than x for all finite
/// positive inputs (normal and subnormal).
#[kani::unwind(1)]
#[kani::proof]
fn next_down_f32_strictly_less_for_positive_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x > 0.0);
    let result = next_down_f32(x);
    assert!(result.is_finite(), "result must be finite");
    assert!(result < x, "next_down must be strictly less than input");
}

/// next_up_f32 returns a value strictly greater than x for all finite
/// positive inputs.
#[kani::unwind(1)]
#[kani::proof]
fn next_up_f32_strictly_greater_for_positive_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x > 0.0);
    let result = next_up_f32(x);
    // f32::MAX maps to +inf, which is still > x
    assert!(result > x, "next_up must be strictly greater than input");
}

/// next_down_f32 returns a value strictly less than x for all finite
/// negative inputs.
#[kani::unwind(1)]
#[kani::proof]
fn next_down_f32_strictly_less_for_negative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x < 0.0);
    let result = next_down_f32(x);
    // f32::MIN maps to -inf, which is still < x
    assert!(result < x, "next_down must be strictly less than input");
}

/// next_up_f32 returns a value strictly greater than x for all finite
/// negative inputs.
#[kani::unwind(1)]
#[kani::proof]
fn next_up_f32_strictly_greater_for_negative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x < 0.0);
    let result = next_up_f32(x);
    assert!(
        result.is_finite(),
        "result must be finite for negative input"
    );
    assert!(result > x, "next_up must be strictly greater than input");
}

/// Neither next_down_f32 nor next_up_f32 produces NaN from finite input.
#[kani::unwind(1)]
#[kani::proof]
fn ulp_functions_never_produce_nan_from_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    let down = next_down_f32(x);
    let up = next_up_f32(x);
    assert!(
        !down.is_nan(),
        "next_down must not produce NaN from finite input"
    );
    assert!(
        !up.is_nan(),
        "next_up must not produce NaN from finite input"
    );
}

/// next_down_f32 at zero: covers the `x == 0.0` branch in bounds_repair.rs
/// that returns the smallest negative subnormal. Also verifies -0.0 gives
/// the same result as +0.0 (IEEE 754: +0.0 == -0.0). (#262)
#[kani::unwind(1)]
#[kani::proof]
fn next_down_f32_at_zero() {
    let result = next_down_f32(0.0);
    assert!(result.is_finite(), "next_down(0.0) must be finite");
    assert!(result < 0.0, "next_down(0.0) must be negative");
    let result_neg_zero = next_down_f32(-0.0);
    assert_eq!(
        result.to_bits(),
        result_neg_zero.to_bits(),
        "next_down(+0.0) and next_down(-0.0) must produce identical bits"
    );
}

/// next_up_f32 at zero: covers the `x == 0.0` branch in bounds_repair.rs
/// that returns the smallest positive subnormal. Also verifies -0.0 gives
/// the same result as +0.0. (#262)
#[kani::unwind(1)]
#[kani::proof]
fn next_up_f32_at_zero() {
    let result = next_up_f32(0.0);
    assert!(result.is_finite(), "next_up(0.0) must be finite");
    assert!(result > 0.0, "next_up(0.0) must be positive");
    let result_neg_zero = next_up_f32(-0.0);
    assert_eq!(
        result.to_bits(),
        result_neg_zero.to_bits(),
        "next_up(+0.0) and next_up(-0.0) must produce identical bits"
    );
}

/// Infinity preservation: both functions return their input unchanged
/// for infinity inputs. This ensures round_for_soundness does not
/// corrupt infeasible sentinels (+inf, -inf).
#[kani::unwind(1)]
#[kani::proof]
fn ulp_functions_preserve_infinity_sentinels() {
    assert_eq!(
        next_down_f32(f32::INFINITY),
        f32::INFINITY,
        "next_down(+inf) must preserve +inf sentinel"
    );
    assert_eq!(
        next_up_f32(f32::NEG_INFINITY),
        f32::NEG_INFINITY,
        "next_up(-inf) must preserve -inf sentinel"
    );
    assert_eq!(
        next_down_f32(f32::NEG_INFINITY),
        f32::NEG_INFINITY,
        "next_down(-inf) must preserve -inf"
    );
    assert_eq!(
        next_up_f32(f32::INFINITY),
        f32::INFINITY,
        "next_up(+inf) must preserve +inf"
    );
}

/// NaN input passes through both functions unchanged (NaN propagation).
#[kani::unwind(1)]
#[kani::proof]
fn ulp_functions_propagate_nan() {
    let x: f32 = kani::any();
    kani::assume(x.is_nan());
    let down = next_down_f32(x);
    let up = next_up_f32(x);
    assert!(down.is_nan(), "next_down(NaN) must be NaN");
    assert!(up.is_nan(), "next_up(NaN) must be NaN");
}

// round_for_soundness composition proofs and repair_nan_to_fallback harnesses
// extracted to stay under the 500-line limit (Part of #1575).
#[path = "ulp_round.rs"]
mod round;
