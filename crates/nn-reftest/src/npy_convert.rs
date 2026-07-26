// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! NumPy dtype conversion functions for `.npy` file loading.
//!
//! Converts raw byte buffers to `Vec<f32>` for each supported NumPy dtype.
//! Extracted from `npy.rs` to keep files under the 400-line maintenance
//! threshold.

use crate::error::ReftestError;

/// Convert raw NPY bytes to f32 based on the dtype descriptor string.
///
/// Supported dtypes:
/// - `<f4` / `<f2` / `<f8`: little-endian float32/float16/float64
/// - `>f4` / `>f2` / `>f8`: big-endian float32/float16/float64
/// - `<i4` / `<i2` / `<i8` / `<i1`: little-endian signed integers
pub(crate) fn convert_npy_to_f32(
    raw: &[u8],
    dtype: &str,
    numel: usize,
) -> Result<Vec<f32>, ReftestError> {
    match dtype {
        "<f4" => convert_f32_le(raw, numel),
        ">f4" => convert_f32_be(raw, numel),
        "<f8" => convert_f64_le(raw, numel),
        ">f8" => convert_f64_be(raw, numel),
        "<f2" => convert_f16_le(raw, numel),
        ">f2" => convert_f16_be(raw, numel),
        "<i4" | "=i4" => convert_i32_le(raw, numel),
        "<i8" | "=i8" => convert_i64_le(raw, numel),
        "<i2" | "=i2" => convert_i16_le(raw, numel),
        "<i1" | "|i1" => convert_i8(raw, numel),
        "<u1" | "|u1" => convert_u8(raw, numel),
        other => Err(ReftestError::NpyUnsupportedDtype(other.to_string())),
    }
}

fn checked_byte_count(numel: usize, bytes_per_element: usize) -> Result<usize, ReftestError> {
    numel
        .checked_mul(bytes_per_element)
        .ok_or(ReftestError::ByteCountOverflow {
            numel,
            bytes_per_element,
        })
}

fn validate_raw_len(raw: &[u8], expected: usize) -> Result<(), ReftestError> {
    if raw.len() < expected {
        return Err(ReftestError::DataLengthMismatch {
            expected,
            actual: raw.len(),
        });
    }
    Ok(())
}

/// f32 can represent integers exactly up to 2^24 (16,777,216).
/// Beyond this, the mantissa cannot hold all significant bits.
const F32_INT_PRECISION_LIMIT: i64 = 1 << 24;

/// Guard against silent precision loss when casting integers to f32.
///
/// Returns an error if `|value| > 2^24`, where f32's 23-bit mantissa
/// can no longer represent the integer exactly.
fn check_int_precision(value: i64, index: usize) -> Result<(), ReftestError> {
    if value.abs() > F32_INT_PRECISION_LIMIT {
        return Err(ReftestError::IntPrecisionLoss { value, index });
    }
    Ok(())
}

fn convert_f32_le(raw: &[u8], numel: usize) -> Result<Vec<f32>, ReftestError> {
    let expected = checked_byte_count(numel, 4)?;
    validate_raw_len(raw, expected)?;
    Ok(raw[..expected]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn convert_f32_be(raw: &[u8], numel: usize) -> Result<Vec<f32>, ReftestError> {
    let expected = checked_byte_count(numel, 4)?;
    validate_raw_len(raw, expected)?;
    Ok(raw[..expected]
        .chunks_exact(4)
        .map(|b| f32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn convert_f64_le(raw: &[u8], numel: usize) -> Result<Vec<f32>, ReftestError> {
    let expected = checked_byte_count(numel, 8)?;
    validate_raw_len(raw, expected)?;
    raw[..expected]
        .chunks_exact(8)
        .enumerate()
        .map(|(i, b)| {
            let v = f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
            if !v.is_finite() || v.abs() > f64::from(f32::MAX) {
                return Err(ReftestError::F64OutOfF32Range { value: v, index: i });
            }
            Ok(v as f32)
        })
        .collect()
}

fn convert_f64_be(raw: &[u8], numel: usize) -> Result<Vec<f32>, ReftestError> {
    let expected = checked_byte_count(numel, 8)?;
    validate_raw_len(raw, expected)?;
    raw[..expected]
        .chunks_exact(8)
        .enumerate()
        .map(|(i, b)| {
            let v = f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
            if !v.is_finite() || v.abs() > f64::from(f32::MAX) {
                return Err(ReftestError::F64OutOfF32Range { value: v, index: i });
            }
            Ok(v as f32)
        })
        .collect()
}

fn convert_f16_le(raw: &[u8], numel: usize) -> Result<Vec<f32>, ReftestError> {
    let expected = checked_byte_count(numel, 2)?;
    validate_raw_len(raw, expected)?;
    Ok(raw[..expected]
        .chunks_exact(2)
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect())
}

fn convert_f16_be(raw: &[u8], numel: usize) -> Result<Vec<f32>, ReftestError> {
    let expected = checked_byte_count(numel, 2)?;
    validate_raw_len(raw, expected)?;
    Ok(raw[..expected]
        .chunks_exact(2)
        .map(|b| half::f16::from_be_bytes([b[0], b[1]]).to_f32())
        .collect())
}

fn convert_i32_le(raw: &[u8], numel: usize) -> Result<Vec<f32>, ReftestError> {
    let expected = checked_byte_count(numel, 4)?;
    validate_raw_len(raw, expected)?;
    raw[..expected]
        .chunks_exact(4)
        .enumerate()
        .map(|(i, b)| {
            let v = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            check_int_precision(i64::from(v), i)?;
            Ok(v as f32)
        })
        .collect()
}

fn convert_i64_le(raw: &[u8], numel: usize) -> Result<Vec<f32>, ReftestError> {
    let expected = checked_byte_count(numel, 8)?;
    validate_raw_len(raw, expected)?;
    raw[..expected]
        .chunks_exact(8)
        .enumerate()
        .map(|(i, b)| {
            let v = i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
            check_int_precision(v, i)?;
            Ok(v as f32)
        })
        .collect()
}

fn convert_i16_le(raw: &[u8], numel: usize) -> Result<Vec<f32>, ReftestError> {
    let expected = checked_byte_count(numel, 2)?;
    validate_raw_len(raw, expected)?;
    Ok(raw[..expected]
        .chunks_exact(2)
        .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])))
        .collect())
}

fn convert_i8(raw: &[u8], numel: usize) -> Result<Vec<f32>, ReftestError> {
    validate_raw_len(raw, numel)?;
    Ok(raw[..numel].iter().map(|&b| f32::from(b as i8)).collect())
}

fn convert_u8(raw: &[u8], numel: usize) -> Result<Vec<f32>, ReftestError> {
    validate_raw_len(raw, numel)?;
    Ok(raw[..numel].iter().map(|&b| f32::from(b)).collect())
}

// ---------------------------------------------------------------------------
// Kani proof harnesses for NPY dtype conversion safety (#3593)
// ---------------------------------------------------------------------------

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // -- Harness 1: checked_byte_count detects overflow --

    /// Proves that `checked_byte_count` returns `Err` when `numel * bytes_per_element`
    /// overflows `usize`.
    #[kani::unwind(1)]
    #[kani::proof]
    fn checked_byte_count_detects_overflow() {
        let numel: usize = kani::any();
        let bpe: usize = kani::any();

        // Only consider cases where multiplication would overflow.
        kani::assume(bpe > 0);
        kani::assume(numel > usize::MAX / bpe);

        let result = checked_byte_count(numel, bpe);
        assert!(
            result.is_err(),
            "checked_byte_count must return Err on overflow"
        );
    }

    // -- Harness 2: checked_byte_count correct for valid inputs --

    /// Proves that `checked_byte_count` returns the correct product when
    /// no overflow occurs.
    #[kani::unwind(5)]
    #[kani::proof]
    fn checked_byte_count_correct_for_valid() {
        let numel: usize = kani::any();
        let bpe: usize = kani::any();

        // Constrain to values that won't overflow.
        kani::assume(bpe > 0 && bpe <= 8);
        kani::assume(numel <= 1_000_000);

        let result = checked_byte_count(numel, bpe);
        let expected = numel * bpe;
        assert!(
            result.is_ok(),
            "checked_byte_count must succeed for small inputs"
        );
        assert!(
            result.unwrap() == expected,
            "checked_byte_count must return numel * bpe"
        );
    }

    // -- Harness 3: check_int_precision boundary at 2^24 --

    /// Proves that `check_int_precision` accepts values with |v| <= 2^24
    /// and rejects values with |v| > 2^24.
    ///
    /// f32 has a 23-bit mantissa, so integers up to 2^24 (16,777,216)
    /// are exactly representable. Beyond that, precision is lost.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int_precision_boundary_exact() {
        let value: i64 = kani::any();

        // Restrict to near the boundary to keep the search tractable.
        kani::assume(value >= -(1i64 << 25) && value <= (1i64 << 25));

        let result = check_int_precision(value, 0);
        let limit: i64 = 1 << 24;

        if value.abs() > limit {
            assert!(result.is_err(), "values with |v| > 2^24 must be rejected");
        } else {
            assert!(result.is_ok(), "values with |v| <= 2^24 must be accepted");
        }
    }

    // -- Harness 4: validate_raw_len rejects short buffers --

    /// Proves that `validate_raw_len` returns `Err` when the buffer is
    /// shorter than the expected byte count.
    #[kani::unwind(8)]
    #[kani::proof]
    fn validate_raw_len_rejects_short_buffer() {
        let buf_len: usize = kani::any();
        let expected: usize = kani::any();

        kani::assume(buf_len < 256);
        kani::assume(expected < 256);
        kani::assume(buf_len < expected);

        let buf = vec![0u8; buf_len];
        let result = validate_raw_len(&buf, expected);

        assert!(
            result.is_err(),
            "validate_raw_len must reject buffers shorter than expected"
        );
    }

    // -- Harness 5: validate_raw_len accepts sufficient buffers --

    /// Proves that `validate_raw_len` returns `Ok` when the buffer is
    /// at least as long as the expected byte count.
    #[kani::unwind(8)]
    #[kani::proof]
    fn validate_raw_len_accepts_sufficient_buffer() {
        let buf_len: usize = kani::any();
        let expected: usize = kani::any();

        kani::assume(buf_len <= 256);
        kani::assume(expected <= buf_len);

        let buf = vec![0u8; buf_len];
        let result = validate_raw_len(&buf, expected);

        assert!(
            result.is_ok(),
            "validate_raw_len must accept buffers >= expected length"
        );
    }
}
