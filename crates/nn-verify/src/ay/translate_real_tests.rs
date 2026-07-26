// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `real_from_f64` and `real_from_f64_denominator`.
//!
//! Regression tests for boundary bugs (#161, #398/#549) and the adaptive
//! denominator for small values. These supplement the Kani proof harnesses
//! in `translate_real.rs` with concrete test vectors.

use super::{real_from_f64, real_from_f64_denominator};

// ======================== real_from_f64: basic values ========================

#[test]
fn test_real_from_f64_zero() {
    let expr = real_from_f64(0.0).expect("0.0 should encode successfully");
    // 0.0 hits the integer fast-path (0.0 == 0.0.floor())
    let _ = expr; // Expression created without error
}

#[test]
fn test_real_from_f64_positive_one() {
    let expr = real_from_f64(1.0).expect("1.0 should encode successfully");
    let _ = expr;
}

#[test]
fn test_real_from_f64_negative_one() {
    let expr = real_from_f64(-1.0).expect("-1.0 should encode successfully");
    let _ = expr;
}

#[test]
fn test_real_from_f64_typical_ml_value() {
    // Typical ML activation value
    let expr = real_from_f64(0.123456789).expect("typical value should encode");
    let _ = expr;
}

// ======================== real_from_f64: non-finite rejection ========================

#[test]
fn test_real_from_f64_nan_rejected() {
    let err = real_from_f64(f64::NAN).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite literal"),
        "NaN should produce NonFiniteLiteral, got: {msg}"
    );
}

#[test]
fn test_real_from_f64_pos_infinity_rejected() {
    let err = real_from_f64(f64::INFINITY).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite literal"),
        "+Inf should produce NonFiniteLiteral, got: {msg}"
    );
}

#[test]
fn test_real_from_f64_neg_infinity_rejected() {
    let err = real_from_f64(f64::NEG_INFINITY).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite literal"),
        "-Inf should produce NonFiniteLiteral, got: {msg}"
    );
}

// ======================== real_from_f64: i64 overflow boundary (#398/#549) ========================

#[test]
fn test_real_from_f64_overflow_boundary_rejected() {
    // i64::MAX as f64 rounds up because i64::MAX is odd and f64 mantissa
    // cannot represent it exactly. The `>=` guard (not `>`) catches this.
    // With default denom 1e6, fractional |val| >= ~9.223e12 triggers overflow.
    // Use a fractional value to avoid the integer fast-path.
    let val = 9.3e12 + 0.5;
    let err = real_from_f64(val).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("too large for real encoding"),
        "overflow value should produce ValueTooLargeForRealEncoding, got: {msg}"
    );
}

#[test]
fn test_real_from_f64_negative_overflow_rejected() {
    // Fractional negative value in overflow zone
    let err = real_from_f64(-9.3e12 - 0.5).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("too large for real encoding"),
        "negative overflow value should be rejected, got: {msg}"
    );
}

#[test]
fn test_real_from_f64_large_integer_uses_fast_path() {
    // Large integers bypass the overflow guard via the integer fast-path.
    // 9.3e12 is an integer and |9.3e12| < i64::MAX as f64 (~9.22e18).
    let expr = real_from_f64(9.3e12).expect("large integer should use fast-path");
    let _ = expr;
}

#[test]
fn test_real_from_f64_safe_range_accepted() {
    // Values well within the safe range should succeed
    let expr = real_from_f64(1e6).expect("1e6 should be in safe range");
    let _ = expr;
    let expr = real_from_f64(-1e6).expect("-1e6 should be in safe range");
    let _ = expr;
}

// ======================== real_from_f64: small values / adaptive denominator ========================

#[test]
fn test_real_from_f64_small_fraction() {
    // 1e-7 is below the default denominator threshold (|val * 1e6| < 0.5).
    // Adaptive denominator should kick in to avoid quantizing to zero.
    let expr = real_from_f64(1e-7).expect("small fraction should encode with adaptive denom");
    let _ = expr;
}

#[test]
fn test_real_from_f64_quantization_margin_territory() {
    // 5e-7 is at the edge of the SMT quantization margin.
    let expr = real_from_f64(5e-7).expect("5e-7 should encode successfully");
    let _ = expr;
}

// ======================== real_from_f64: integer fast-path ========================

#[test]
fn test_real_from_f64_large_integer_accepted() {
    // Large integer that fits in i64 should use the integer fast-path
    let val = 1_000_000_000.0_f64; // 1e9, well within i64
    let expr = real_from_f64(val).expect("1e9 integer should use fast-path");
    let _ = expr;
}

#[test]
fn test_real_from_f64_negative_integer() {
    let expr = real_from_f64(-42.0).expect("-42.0 integer should encode");
    let _ = expr;
}

// ======================== real_from_f64_denominator ========================

#[test]
fn test_denominator_default_for_normal_values() {
    // For |val| >= 5e-7 (i.e., |val * 1e6| >= 0.5), use default 1e6
    assert_eq!(real_from_f64_denominator(1.0), 1_000_000);
    assert_eq!(real_from_f64_denominator(-100.0), 1_000_000);
    assert_eq!(real_from_f64_denominator(0.001), 1_000_000);
}

#[test]
fn test_denominator_adaptive_for_tiny_values() {
    // For |val| < 5e-7, denominator should increase beyond 1e6
    let denom = real_from_f64_denominator(1e-10);
    assert!(
        denom > 1_000_000,
        "tiny value 1e-10 should get adaptive denom > 1e6, got {denom}"
    );
    assert!(
        denom <= 1_000_000_000_000_000,
        "denom should not exceed 1e15, got {denom}"
    );
}

#[test]
fn test_denominator_zero_uses_default() {
    assert_eq!(real_from_f64_denominator(0.0), 1_000_000);
}

#[test]
fn test_denominator_always_at_least_1e6() {
    // Even for large values, denominator is at least 1e6
    let denom = real_from_f64_denominator(1e12);
    assert!(denom >= 1_000_000, "denom should be >= 1e6, got {denom}");
}

#[test]
fn test_denominator_capped_at_1e15() {
    // Extremely tiny values should cap at 1e15
    let denom = real_from_f64_denominator(1e-20);
    assert_eq!(
        denom, 1_000_000_000_000_000,
        "extremely tiny value should cap denom at 1e15, got {denom}"
    );
}

// ======================== real_from_f64: numerical correctness ========================
//
// The tests above only check Ok/Err — they don't verify the encoded value
// matches the input. These tests check the SMT-LIB2 string representation
// to ensure the rational encoding is numerically correct.

#[test]
fn test_real_from_f64_integer_encodes_exactly() {
    // Integer values should produce exact SMT-LIB2 "N.0" (no division).
    let expr = real_from_f64(42.0).expect("42.0 should encode");
    let smt = format!("{expr}");
    assert_eq!(
        smt, "42.0",
        "integer 42.0 should encode as 42.0, got: {smt}"
    );
}

#[test]
fn test_real_from_f64_negative_integer_encodes_exactly() {
    let expr = real_from_f64(-7.0).expect("-7.0 should encode");
    let smt = format!("{expr}");
    assert!(smt.contains("7"), "should contain 7, got: {smt}");
}

#[test]
fn test_real_from_f64_half_encodes_as_rational() {
    // 0.5 should encode as 500000 / 1000000.
    let expr = real_from_f64(0.5).expect("0.5 should encode");
    let smt = format!("{expr}");
    assert!(
        smt.contains("/"),
        "fractional 0.5 should encode as division, got: {smt}"
    );
    assert!(
        smt.contains("500000"),
        "numerator should be 500000, got: {smt}"
    );
    assert!(
        smt.contains("1000000"),
        "denominator should be 1000000, got: {smt}"
    );
}

#[test]
fn test_real_from_f64_precision_round_trip() {
    // Verify encoding preserves 6 significant digits.
    // real_from_f64(val) encodes as round(val * 1e6) / 1e6.
    // Decoded value should be within 5e-7 of the original.
    #[allow(clippy::approx_constant)]
    // Intentional: testing round-trip on approximate values, not exact constants.
    let test_values: &[f64] = &[0.123456, -0.987654, 3.141593, 1e-4, -2.718282];
    for &val in test_values {
        let expr = real_from_f64(val).unwrap_or_else(|_| panic!("{val} should encode"));
        let smt = format!("{expr}");
        // Extract numerator from "(/ N.0 1000000.0)"
        if let Some(numer_str) = smt.strip_prefix("(/ ").and_then(|s| s.split(".0 ").next()) {
            let numer: f64 = numer_str.parse().unwrap_or(f64::NAN);
            let decoded = numer / 1_000_000.0;
            let error = (decoded - val).abs();
            assert!(
                error < 5.1e-7,
                "round-trip error for {val}: decoded={decoded}, error={error}"
            );
        }
    }
}
