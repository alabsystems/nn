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
const F32_INT_PRECISION_FLOOR: i64 = -F32_INT_PRECISION_LIMIT;

/// Guard against silent precision loss when casting integers to f32.
///
/// Returns an error if `|value| > 2^24`, where f32's 23-bit mantissa
/// can no longer represent the integer exactly. A range match, not `|value|`:
/// `i64::MIN` has no `i64` absolute value, and `abs` overflowed on it.
fn check_int_precision(value: i64, index: usize) -> Result<(), ReftestError> {
    match value {
        F32_INT_PRECISION_FLOOR..=F32_INT_PRECISION_LIMIT => Ok(()),
        _ => Err(ReftestError::IntPrecisionLoss { value, index }),
    }
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
