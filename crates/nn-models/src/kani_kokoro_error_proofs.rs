// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for kokoro_error validation functions.
//!
//! Proves that:
//! 1. validate_speed rejects NaN.
//! 2. validate_speed rejects positive infinity.
//! 3. validate_speed rejects negative infinity.
//! 4. validate_speed rejects zero.
//! 5. validate_speed rejects negative values.
//! 6. validate_speed accepts valid positive finite speeds.
//! 7. LOG_MAG_CLAMP_MAX prevents f32 overflow from exp().
//! 8. validate_speed is consistent: accept iff finite and positive.
//!
//! Part of #3793, #3351.

use crate::kokoro_error::{validate_speed, LOG_MAG_CLAMP_MAX};

/// Proof 1: validate_speed rejects NaN.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_validate_speed_rejects_nan() {
    let result = validate_speed(f32::NAN);
    assert!(result.is_err(), "NaN speed must be rejected");
}

/// Proof 2: validate_speed rejects positive infinity.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_validate_speed_rejects_pos_inf() {
    let result = validate_speed(f32::INFINITY);
    assert!(result.is_err(), "positive infinity must be rejected");
}

/// Proof 3: validate_speed rejects negative infinity.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_validate_speed_rejects_neg_inf() {
    let result = validate_speed(f32::NEG_INFINITY);
    assert!(result.is_err(), "negative infinity must be rejected");
}

/// Proof 4: validate_speed rejects zero.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_validate_speed_rejects_zero() {
    let result = validate_speed(0.0);
    assert!(result.is_err(), "zero speed must be rejected");
}

/// Proof 5: validate_speed rejects negative values.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_validate_speed_rejects_negative() {
    let speed: f32 = kani::any();
    kani::assume(speed.is_finite());
    kani::assume(speed < 0.0);
    let result = validate_speed(speed);
    assert!(result.is_err(), "negative speed must be rejected");
}

/// Proof 6: validate_speed accepts valid positive finite speeds.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_validate_speed_accepts_positive_finite() {
    let speed: f32 = kani::any();
    kani::assume(speed.is_finite());
    kani::assume(speed > 0.0);
    let result = validate_speed(speed);
    assert!(result.is_ok(), "positive finite speed must be accepted");
}

/// Proof 7: LOG_MAG_CLAMP_MAX is safe for f32 exp().
///
/// exp(88.0) ≈ 1.65e38, which is less than f32::MAX ≈ 3.4e38.
/// This proves the constant prevents f32 overflow.
fn ln_f64_stub(_x: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0);
    r
}

#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::ln, ln_f64_stub)]
fn proof_log_mag_clamp_max_prevents_overflow() {
    // LOG_MAG_CLAMP_MAX = 88.0
    assert!(LOG_MAG_CLAMP_MAX == 88.0);
    // exp(88.0) ≈ 1.65e38, which is less than f32::MAX ≈ 3.4e38.
    // ln(f32::MAX) ≈ 88.72, so LOG_MAG_CLAMP_MAX < ln(f32::MAX).
    // With transcendental stubs, we verify the constant directly:
    // 88.0 < 88.72 (pre-computed ln(3.4028235e38)).
    let max_val = LOG_MAG_CLAMP_MAX as f32;
    assert!(max_val.is_finite());
    assert!(max_val > 0.0);
    // Pre-computed: ln(f32::MAX) = 88.72283... > 88.0.
    let ln_f32_max: f64 = 88.722839;
    assert!(
        (max_val as f64) < ln_f32_max,
        "LOG_MAG_CLAMP_MAX must be below ln(f32::MAX)"
    );
}

/// Proof 8: validate_speed accepts iff finite and positive.
///
/// Proves the bidirectional equivalence:
/// validate_speed(x).is_ok() ⟺ x.is_finite() && x > 0.0
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_validate_speed_iff_finite_positive() {
    let speed: f32 = kani::any();
    let result = validate_speed(speed);
    let should_accept = speed.is_finite() && speed > 0.0;
    assert_eq!(
        result.is_ok(),
        should_accept,
        "validate_speed must accept iff speed is finite and positive"
    );
}
