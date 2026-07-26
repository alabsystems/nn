// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `clamp_literal_for_type` precision guard: f32/f16 overflow,
//! underflow, boundary passthrough, zero preservation, and NaN/Inf handling.

use super::*;

/// Assert that non-finite values (NaN, ±Inf) pass through `clamp_literal_for_type`
/// unchanged for both F32 and F16 scalar types.
fn assert_non_finite_passthrough(v: f64) {
    let f32_result = clamp_literal_for_type(v, ScalarType::F32);
    let f16_result = clamp_literal_for_type(v, ScalarType::F16);
    if v.is_nan() {
        assert!(f32_result.is_nan(), "NaN must pass through for F32");
        assert!(f16_result.is_nan(), "NaN must pass through for F16");
    } else {
        assert_eq!(f32_result, v, "Inf must pass through for F32");
        assert_eq!(f16_result, v, "Inf must pass through for F16");
    }
}

#[test]
fn test_clamp_literal_f32_passthrough() {
    assert_eq!(clamp_literal_for_type(1e-8, ScalarType::F32), 1e-8);
    assert_eq!(clamp_literal_for_type(0.0, ScalarType::F32), 0.0);
    assert_eq!(clamp_literal_for_type(-1e-30, ScalarType::F32), -1e-30);
    // Values within f32 range pass through unchanged
    assert_eq!(clamp_literal_for_type(1.0, ScalarType::F32), 1.0);
    assert_eq!(clamp_literal_for_type(-42.5, ScalarType::F32), -42.5);
}

#[test]
fn test_clamp_literal_f32_overflow_clamped() {
    let f32_max = f64::from(f32::MAX);
    // Values exceeding f32 MAX should be clamped to f32 MAX
    assert_eq!(clamp_literal_for_type(1e300, ScalarType::F32), f32_max);
    assert_eq!(clamp_literal_for_type(1e40, ScalarType::F32), f32_max);
}

#[test]
fn test_clamp_literal_f32_negative_overflow_clamped() {
    let f32_max = f64::from(f32::MAX);
    // Negative values below -f32 MAX should be clamped to -f32 MAX
    assert_eq!(clamp_literal_for_type(-1e300, ScalarType::F32), -f32_max);
    assert_eq!(clamp_literal_for_type(-1e40, ScalarType::F32), -f32_max);
}

#[test]
fn test_clamp_literal_f32_max_boundary_passthrough() {
    let f32_max = f64::from(f32::MAX);
    // Exactly f32 MAX should pass through
    assert_eq!(clamp_literal_for_type(f32_max, ScalarType::F32), f32_max);
    assert_eq!(clamp_literal_for_type(-f32_max, ScalarType::F32), -f32_max);
}

#[test]
fn test_clamp_literal_f16_underflow_clamped() {
    let f16_min = f64::from(half::f16::MIN_POSITIVE);
    // 1e-8 underflows f16 (min positive normal ≈ 6.1e-5), must be clamped up
    let result = clamp_literal_for_type(1e-8, ScalarType::F16);
    assert_eq!(result, f16_min, "1e-8 should clamp to f16 MIN_POSITIVE");
}

#[test]
fn test_clamp_literal_f16_negative_underflow_clamped() {
    let f16_min = f64::from(half::f16::MIN_POSITIVE);
    let result = clamp_literal_for_type(-1e-8, ScalarType::F16);
    assert_eq!(
        result, -f16_min,
        "negative underflow should clamp to -MIN_POSITIVE"
    );
}

#[test]
fn test_clamp_literal_f16_zero_preserved() {
    assert_eq!(clamp_literal_for_type(0.0, ScalarType::F16), 0.0);
}

#[test]
fn test_clamp_literal_f16_representable_passthrough() {
    // 1.0 is perfectly representable in f16, should pass through unchanged
    assert_eq!(clamp_literal_for_type(1.0, ScalarType::F16), 1.0);
    assert_eq!(clamp_literal_for_type(-42.0, ScalarType::F16), -42.0);
}

#[test]
fn test_clamp_literal_f16_overflow_clamped() {
    let f16_max = f64::from(half::f16::MAX);
    // Values exceeding f16 MAX should be clamped to f16 MAX
    assert_eq!(clamp_literal_for_type(70000.0, ScalarType::F16), f16_max);
    assert_eq!(clamp_literal_for_type(1e10, ScalarType::F16), f16_max);
}

#[test]
fn test_clamp_literal_f16_negative_overflow_clamped() {
    let f16_max = f64::from(half::f16::MAX);
    // Negative values below -f16 MAX should be clamped to -f16 MAX
    assert_eq!(clamp_literal_for_type(-70000.0, ScalarType::F16), -f16_max);
    assert_eq!(clamp_literal_for_type(-1e10, ScalarType::F16), -f16_max);
}

#[test]
fn test_clamp_literal_f16_max_boundary_passthrough() {
    let f16_max = f64::from(half::f16::MAX);
    // Exactly f16 MAX should pass through
    assert_eq!(clamp_literal_for_type(f16_max, ScalarType::F16), f16_max);
    assert_eq!(clamp_literal_for_type(-f16_max, ScalarType::F16), -f16_max);
}

// --- Non-finite value tests (NaN, Inf) ---
// Before the P1-56 fix, NaN silently passed through all comparison branches
// (all return false), which was accidentally correct for format_float (which
// handles NaN separately) but violated clamp_literal_for_type's contract of
// making values "representable in the target scalar type."

#[test]
fn test_clamp_literal_nan_passthrough() {
    assert_non_finite_passthrough(f64::NAN);
}

#[test]
fn test_clamp_literal_positive_inf_passthrough() {
    assert_non_finite_passthrough(f64::INFINITY);
}

#[test]
fn test_clamp_literal_negative_inf_passthrough() {
    assert_non_finite_passthrough(f64::NEG_INFINITY);
}

#[test]
fn test_clamp_literal_negative_zero() {
    // -0.0 is finite and should pass through for both types.
    let f32_result = clamp_literal_for_type(-0.0, ScalarType::F32);
    assert!(
        f32_result.is_sign_negative(),
        "F32: -0.0 sign must be preserved"
    );
    assert_eq!(f32_result, 0.0);

    let f16_result = clamp_literal_for_type(-0.0, ScalarType::F16);
    assert!(
        f16_result.is_sign_negative(),
        "F16: -0.0 sign must be preserved"
    );
    assert_eq!(f16_result, 0.0);
}
