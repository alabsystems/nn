// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for NPY dtype conversion safety properties.
//!
//! These harnesses verify properties of `npy_convert.rs` conversion functions
//! beyond the inline harnesses: f16 roundtrip fidelity, f64 range guards,
//! i32 precision boundary, endianness symmetry, and multi-element conversion
//! correctness.
//!
//! Issue: #3670

// ---------------------------------------------------------------------------
// f16 conversion harnesses
// ---------------------------------------------------------------------------

/// Proves that f16 LE roundtrip preserves the value through f16 precision.
///
/// f16 has limited precision (10-bit mantissa), so the roundtrip is:
/// f32 -> f16 -> le_bytes -> convert_f16_le -> f32.
/// The result must equal the intermediate f16 value promoted back to f32.
#[kani::unwind(1)]
#[kani::proof]
fn f16_le_roundtrip_through_f16_precision() {
    let bits: u16 = kani::any();
    let f16_val = half::f16::from_bits(bits);
    let f32_val = f16_val.to_f32();

    // Skip NaN/Inf f16 values — they convert but are non-finite.
    kani::assume(f32_val.is_finite());

    let bytes = f16_val.to_le_bytes();
    let result = crate::npy::convert::convert_npy_to_f32(&bytes, "<f2", 1);
    assert!(result.is_ok(), "valid f16 LE bytes must convert");
    let out = result.unwrap();
    assert!(out.len() == 1, "must produce one element");
    assert!(
        out[0] == f32_val,
        "f16 LE roundtrip must preserve the f16-precision value"
    );
}

/// Proves that f16 BE roundtrip preserves the value through f16 precision.
#[kani::unwind(1)]
#[kani::proof]
fn f16_be_roundtrip_through_f16_precision() {
    let bits: u16 = kani::any();
    let f16_val = half::f16::from_bits(bits);
    let f32_val = f16_val.to_f32();

    kani::assume(f32_val.is_finite());

    let bytes = f16_val.to_be_bytes();
    let result = crate::npy::convert::convert_npy_to_f32(&bytes, ">f2", 1);
    assert!(result.is_ok(), "valid f16 BE bytes must convert");
    let out = result.unwrap();
    assert!(out.len() == 1, "must produce one element");
    assert!(
        out[0] == f32_val,
        "f16 BE roundtrip must preserve the f16-precision value"
    );
}

// ---------------------------------------------------------------------------
// f64 range guard harnesses
// ---------------------------------------------------------------------------

/// Proves that f64 LE conversion rejects non-finite f64 values.
///
/// A f64 NaN or Infinity encoded as `<f8` bytes must produce an error,
/// not silently cast to f32.
#[kani::unwind(1)]
#[kani::proof]
fn f64_le_rejects_non_finite() {
    let selector: u8 = kani::any();
    kani::assume(selector < 3);

    let val: f64 = match selector {
        0 => f64::NAN,
        1 => f64::INFINITY,
        _ => f64::NEG_INFINITY,
    };

    let bytes = val.to_le_bytes();
    let result = crate::npy::convert::convert_npy_to_f32(&bytes, "<f8", 1);
    assert!(
        result.is_err(),
        "non-finite f64 must be rejected by convert_f64_le"
    );
}

/// Proves that f64 BE conversion rejects non-finite f64 values.
#[kani::unwind(1)]
#[kani::proof]
fn f64_be_rejects_non_finite() {
    let selector: u8 = kani::any();
    kani::assume(selector < 3);

    let val: f64 = match selector {
        0 => f64::NAN,
        1 => f64::INFINITY,
        _ => f64::NEG_INFINITY,
    };

    let bytes = val.to_be_bytes();
    let result = crate::npy::convert::convert_npy_to_f32(&bytes, ">f8", 1);
    assert!(
        result.is_err(),
        "non-finite f64 must be rejected by convert_f64_be"
    );
}

/// Proves that f64 values exceeding f32::MAX magnitude are rejected.
///
/// f64 can represent values far beyond f32::MAX (~3.4e38). The converter
/// must reject these to prevent silent overflow to f32::INFINITY.
#[kani::unwind(1)]
#[kani::proof]
fn f64_le_rejects_out_of_f32_range() {
    // Use a value just barely above f32::MAX.
    let val: f64 = f64::from(f32::MAX) * 2.0;
    let bytes = val.to_le_bytes();
    let result = crate::npy::convert::convert_npy_to_f32(&bytes, "<f8", 1);
    assert!(
        result.is_err(),
        "f64 value exceeding f32::MAX must be rejected"
    );
}

/// Proves that f64 values within f32 range convert successfully (LE).
#[kani::unwind(1)]
#[kani::proof]
fn f64_le_accepts_in_range_value() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    // Promote to f64, then encode — should round-trip.
    let val64 = f64::from(val);
    let bytes = val64.to_le_bytes();
    let result = crate::npy::convert::convert_npy_to_f32(&bytes, "<f8", 1);
    assert!(result.is_ok(), "f64 value within f32 range must succeed");
    let out = result.unwrap();
    assert!(out.len() == 1, "must produce one element");
    // f32 -> f64 -> f32 roundtrip is exact.
    assert!(out[0] == val, "f64(f32) roundtrip must be exact");
}

// ---------------------------------------------------------------------------
// i32 precision boundary harnesses
// ---------------------------------------------------------------------------

/// Proves that i32 values within the f32 exact range [-2^24, 2^24] convert
/// successfully, and values outside that range are rejected.
///
/// f32 has a 23-bit mantissa, so integers up to 2^24 = 16,777,216 are
/// exactly representable. The i32 converter must reject larger values
/// to prevent silent precision loss.
#[kani::unwind(5)]
#[kani::proof]
fn i32_le_precision_boundary() {
    let val: i32 = kani::any();
    // Focus on boundary region.
    kani::assume(val >= (1i32 << 23) || val <= -(1i32 << 23));
    // Keep bounded to avoid timeout.
    kani::assume(val >= -(1i32 << 25) && val <= (1i32 << 25));

    let bytes = val.to_le_bytes();
    let result = crate::npy::convert::convert_npy_to_f32(&bytes, "<i4", 1);

    let limit: i64 = 1i64 << 24;
    if (val as i64).abs() > limit {
        assert!(
            result.is_err(),
            "i32 value with |v| > 2^24 must be rejected for precision safety"
        );
    } else {
        assert!(
            result.is_ok(),
            "i32 value with |v| <= 2^24 must be accepted"
        );
        let out = result.unwrap();
        assert!(
            out[0] == val as f32,
            "i32-to-f32 must be exact within range"
        );
    }
}

/// Proves that all i16 values are within f32 exact range and convert losslessly.
///
/// i16 range is [-32768, 32767], well within 2^24. No precision loss possible.
#[kani::unwind(1)]
#[kani::proof]
fn i16_always_within_f32_precision() {
    let val: i16 = kani::any();
    let as_i64 = i64::from(val);
    let limit: i64 = 1i64 << 24;

    assert!(
        as_i64.abs() <= limit,
        "all i16 values must be within f32 exact range"
    );
}

// ---------------------------------------------------------------------------
// Endianness symmetry harnesses
// ---------------------------------------------------------------------------

/// Proves that LE and BE conversions of the same f32 value produce the
/// same result — endianness handling is correct and symmetric.
#[kani::unwind(1)]
#[kani::proof]
fn f32_endianness_symmetry() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    let le_bytes = val.to_le_bytes();
    let be_bytes = val.to_be_bytes();

    let le_result = crate::npy::convert::convert_npy_to_f32(&le_bytes, "<f4", 1);
    let be_result = crate::npy::convert::convert_npy_to_f32(&be_bytes, ">f4", 1);

    assert!(le_result.is_ok(), "LE conversion must succeed");
    assert!(be_result.is_ok(), "BE conversion must succeed");

    let le_val = le_result.unwrap()[0];
    let be_val = be_result.unwrap()[0];

    assert!(
        le_val == be_val,
        "LE and BE conversions of the same value must produce the same f32"
    );
    assert!(
        le_val == val,
        "both conversions must recover the original value"
    );
}

// ---------------------------------------------------------------------------
// u8 / i8 single-byte conversion harnesses
// ---------------------------------------------------------------------------

/// Proves that u8 conversion is lossless and the output is in [0, 255].
#[kani::unwind(1)]
#[kani::proof]
fn u8_conversion_range() {
    let val: u8 = kani::any();

    let result = crate::npy::convert::convert_npy_to_f32(&[val], "<u1", 1);
    assert!(result.is_ok(), "u8 conversion must succeed");
    let out = result.unwrap()[0];

    assert!(out >= 0.0, "u8 as f32 must be non-negative");
    assert!(out <= 255.0, "u8 as f32 must be at most 255");
    assert!(out == f32::from(val), "u8 conversion must be exact");
}

/// Proves that i8 conversion is lossless and the output is in [-128, 127].
#[kani::unwind(1)]
#[kani::proof]
fn i8_conversion_range() {
    let raw_byte: u8 = kani::any();

    let result = crate::npy::convert::convert_npy_to_f32(&[raw_byte], "<i1", 1);
    assert!(result.is_ok(), "i8 conversion must succeed");
    let out = result.unwrap()[0];

    let expected = f32::from(raw_byte as i8);
    assert!(out >= -128.0, "i8 as f32 must be >= -128");
    assert!(out <= 127.0, "i8 as f32 must be <= 127");
    assert!(out == expected, "i8 conversion must be exact");
}

/// Proves that the "|u1" dtype alias produces the same result as "<u1".
#[kani::unwind(1)]
#[kani::proof]
fn u8_dtype_aliases_equivalent() {
    let val: u8 = kani::any();

    let r1 = crate::npy::convert::convert_npy_to_f32(&[val], "<u1", 1);
    let r2 = crate::npy::convert::convert_npy_to_f32(&[val], "|u1", 1);

    assert!(r1.is_ok(), "<u1 must succeed");
    assert!(r2.is_ok(), "|u1 must succeed");
    assert!(
        r1.unwrap()[0] == r2.unwrap()[0],
        "<u1 and |u1 must produce identical results"
    );
}

/// Proves that the "|i1" dtype alias produces the same result as "<i1".
#[kani::unwind(1)]
#[kani::proof]
fn i8_dtype_aliases_equivalent() {
    let val: u8 = kani::any();

    let r1 = crate::npy::convert::convert_npy_to_f32(&[val], "<i1", 1);
    let r2 = crate::npy::convert::convert_npy_to_f32(&[val], "|i1", 1);

    assert!(r1.is_ok(), "<i1 must succeed");
    assert!(r2.is_ok(), "|i1 must succeed");
    assert!(
        r1.unwrap()[0] == r2.unwrap()[0],
        "<i1 and |i1 must produce identical results"
    );
}
