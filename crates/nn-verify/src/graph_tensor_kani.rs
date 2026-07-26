// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani verification harnesses for dimension/axis conversion functions.
//!
//! These functions are the trust boundary between nn's `usize` dimensions and
//! NY's `i32`/`i64` parameters. Incorrect conversion would silently
//! produce wrong verification graphs.

use super::*;

/// Prove: `dim_as_i64` succeeds for all values in [0, i64::MAX] and
/// returns a value equal to the input.
///
/// The critical property: the conversion preserves the numeric value.
/// Any usize in [0, i64::MAX] must round-trip exactly.
#[kani::unwind(1)]
#[kani::proof]
fn dim_as_i64_preserves_value() {
    let val: usize = kani::any();
    kani::assume(val <= i64::MAX as usize);

    let result = dim_as_i64(val, "kani");
    assert!(result.is_ok(), "must succeed for val <= i64::MAX");
    assert!(
        result.unwrap() as usize == val,
        "must preserve numeric value"
    );
}

/// Prove: `dim_as_i64` fails for values above i64::MAX.
///
/// On 64-bit platforms, usize::MAX > i64::MAX. The conversion must reject
/// these values. On 32-bit platforms, all usize values fit in i64, so this
/// harness is vacuously true (which is correct — no rejection needed).
#[kani::unwind(1)]
#[kani::proof]
fn dim_as_i64_rejects_overflow() {
    let val: usize = kani::any();
    kani::assume(val > i64::MAX as usize);

    let result = dim_as_i64(val, "kani");
    assert!(result.is_err(), "must fail for val > i64::MAX");
}

/// Prove: `axis_as_i32` succeeds for all values in [0, i32::MAX] and
/// preserves the numeric value.
#[kani::unwind(1)]
#[kani::proof]
fn axis_as_i32_preserves_value() {
    let val: usize = kani::any();
    kani::assume(val <= i32::MAX as usize);

    let result = axis_as_i32(val, "kani");
    assert!(result.is_ok(), "must succeed for val <= i32::MAX");
    assert!(
        result.unwrap() as usize == val,
        "must preserve numeric value"
    );
}

/// Prove: `axis_as_i32` fails for all values above i32::MAX.
///
/// This is the practical boundary: tensors with >2B dimensions on a single
/// axis would overflow i32 axis parameters. The conversion must reject these.
#[kani::unwind(1)]
#[kani::proof]
fn axis_as_i32_rejects_overflow() {
    let val: usize = kani::any();
    kani::assume(val > i32::MAX as usize);

    let result = axis_as_i32(val, "kani");
    assert!(result.is_err(), "must fail for val > i32::MAX");
}

// ---------------------------------------------------------------------------
// checked_i64_to_usize — inverse of dim_as_i64
// ---------------------------------------------------------------------------

/// Prove: `checked_i64_to_usize` succeeds for all non-negative i64 values
/// and preserves the numeric value.
///
/// Critical property: non-negative i64 round-trips exactly through usize.
/// This is the trust boundary for PyTorch shape values entering NY.
#[kani::unwind(1)]
#[kani::proof]
fn checked_i64_to_usize_preserves_value() {
    let val: i64 = kani::any();
    kani::assume(val >= 0);

    let result = checked_i64_to_usize(val, "kani");
    assert!(result.is_ok(), "must succeed for non-negative i64");
    assert!(
        result.unwrap() == val as usize,
        "must preserve numeric value"
    );
}

/// Prove: `checked_i64_to_usize` rejects all negative i64 values.
///
/// Negative shape dimensions from PyTorch traces are invalid. Without this
/// guard, `i64 as usize` wraps (e.g., -1_i64 as usize == usize::MAX),
/// producing enormous buffer allocations or wrong graph topology.
#[kani::unwind(1)]
#[kani::proof]
fn checked_i64_to_usize_rejects_negative() {
    let val: i64 = kani::any();
    kani::assume(val < 0);

    let result = checked_i64_to_usize(val, "kani");
    assert!(result.is_err(), "must fail for negative i64");
}

/// Prove: `checked_i64_to_usize` and `dim_as_i64` are inverse for valid values.
///
/// For any usize value that fits in i64, the round-trip must preserve the value:
/// checked_i64_to_usize(dim_as_i64(v)) == v.
#[kani::unwind(1)]
#[kani::proof]
fn checked_i64_to_usize_round_trip() {
    let val: usize = kani::any();
    kani::assume(val <= i64::MAX as usize);

    let as_i64 = dim_as_i64(val, "kani").unwrap();
    let back = checked_i64_to_usize(as_i64, "kani").unwrap();
    assert!(back == val, "round-trip must preserve value");
}

// ---------------------------------------------------------------------------
// checked_f64_to_usize — scale factors and threshold counts
// ---------------------------------------------------------------------------

/// Prove: `checked_f64_to_usize` rejects NaN.
///
/// NaN is non-finite, caught by the `is_finite()` guard. Without this,
/// Rust saturates `NaN as usize` to 0, silently wrong for scale factors.
#[kani::unwind(1)]
#[kani::proof]
fn checked_f64_to_usize_rejects_nan() {
    let result = checked_f64_to_usize(f64::NAN, "kani");
    assert!(result.is_err(), "must reject NaN");
}

/// Prove: `checked_f64_to_usize` rejects positive and negative infinity.
#[kani::unwind(1)]
#[kani::proof]
fn checked_f64_to_usize_rejects_infinity() {
    let result_pos = checked_f64_to_usize(f64::INFINITY, "kani");
    assert!(result_pos.is_err(), "must reject +Inf");

    let result_neg = checked_f64_to_usize(f64::NEG_INFINITY, "kani");
    assert!(result_neg.is_err(), "must reject -Inf");
}

/// Prove: `checked_f64_to_usize` rejects negative finite values.
///
/// Uses deterministic negative values. Symbolic f64 is impractical for CBMC
/// (2^64 bit patterns), so we cover representative negative values.
#[kani::unwind(1)]
#[kani::proof]
fn checked_f64_to_usize_rejects_negative() {
    let result = checked_f64_to_usize(-1.0, "kani");
    assert!(result.is_err(), "must reject -1.0");

    let result2 = checked_f64_to_usize(-0.001, "kani");
    assert!(result2.is_err(), "must reject -0.001");

    let result3 = checked_f64_to_usize(f64::MIN, "kani");
    assert!(result3.is_err(), "must reject f64::MIN");
}

/// Prove: `checked_f64_to_usize` rejects non-integral values.
///
/// Scale factors like 1.5 or 0.5 cannot be converted to usize dimensions
/// without silent truncation. The 1e-6 tolerance allows tiny floating-point
/// errors (e.g., 2.0000000001) but rejects clearly fractional values.
#[kani::unwind(1)]
#[kani::proof]
fn checked_f64_to_usize_rejects_non_integral() {
    let result = checked_f64_to_usize(3.7, "kani");
    assert!(result.is_err(), "must reject 3.7");

    let result2 = checked_f64_to_usize(0.5, "kani");
    assert!(result2.is_err(), "must reject 0.5");

    let result3 = checked_f64_to_usize(1.001, "kani");
    assert!(result3.is_err(), "must reject 1.001");
}

/// Prove: `checked_f64_to_usize` succeeds for small non-negative integers
/// and preserves the exact value.
///
/// Uses deterministic values to avoid CBMC f64 symbolic explosion.
/// Covers common scale factors (1, 2, 3, 4) and larger shape dims.
#[kani::unwind(1)]
#[kani::proof]
fn checked_f64_to_usize_accepts_integers() {
    let result = checked_f64_to_usize(0.0, "kani");
    assert!(result.is_ok() && result.unwrap() == 0, "must accept 0.0");

    let result = checked_f64_to_usize(1.0, "kani");
    assert!(result.is_ok() && result.unwrap() == 1, "must accept 1.0");

    let result = checked_f64_to_usize(4.0, "kani");
    assert!(result.is_ok() && result.unwrap() == 4, "must accept 4.0");

    let result = checked_f64_to_usize(1024.0, "kani");
    assert!(
        result.is_ok() && result.unwrap() == 1024,
        "must accept 1024.0"
    );

    let result = checked_f64_to_usize(65536.0, "kani");
    assert!(
        result.is_ok() && result.unwrap() == 65536,
        "must accept 65536.0"
    );
}

/// Prove: `checked_f64_to_usize` accepts near-integer values within tolerance.
///
/// IEEE 754 arithmetic can produce values like 2.0000000000001 from
/// `(4.0 / 2.0)`. The 1e-6 tolerance ensures these are accepted.
#[kani::unwind(1)]
#[kani::proof]
fn checked_f64_to_usize_tolerance_boundary() {
    // Within tolerance: accepted
    let result = checked_f64_to_usize(3.0 + 1e-7, "kani");
    assert!(
        result.is_ok(),
        "must accept 3.0 + 1e-7 (within 1e-6 tolerance)"
    );
    assert!(result.unwrap() == 3, "must round to 3");

    // Outside tolerance: rejected
    let result = checked_f64_to_usize(3.0 + 1e-5, "kani");
    assert!(
        result.is_err(),
        "must reject 3.0 + 1e-5 (outside tolerance)"
    );
}
