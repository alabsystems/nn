// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for compiled model execution helpers.
//!
//! Extracted from `compiled_model_execute_helpers.rs` to keep that file under
//! the 450-line production limit.

use super::*;

#[test]
fn test_validate_buffer_capacity_ok() {
    // 4 elements × 4 bytes = 16 bytes, buffer has 16
    assert!(validate_buffer_capacity(16, 0, &[2, 2], DType::F32, "test").is_ok());
}

#[test]
fn test_validate_buffer_capacity_with_offset() {
    // 4 elements × 4 bytes = 16 bytes, buffer is 32 with offset 16 → 16 available
    assert!(validate_buffer_capacity(32, 16, &[2, 2], DType::F32, "test").is_ok());
}

#[test]
fn test_validate_buffer_capacity_too_small() {
    // 4 elements × 4 bytes = 16 bytes, buffer has only 12
    let err = validate_buffer_capacity(12, 0, &[2, 2], DType::F32, "test");
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("buffer capacity 12 < required 16"),
        "got: {msg}"
    );
}

#[test]
fn test_validate_buffer_capacity_offset_reduces_available() {
    // 4 elements × 4 bytes = 16 bytes, buffer is 20 but offset 8 → 12 available < 16
    let err = validate_buffer_capacity(20, 8, &[2, 2], DType::F32, "test");
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("buffer capacity 12 < required 16"),
        "got: {msg}"
    );
}

#[test]
fn test_validate_buffer_capacity_f16_dtype() {
    // 6 elements × 2 bytes = 12 bytes
    assert!(validate_buffer_capacity(12, 0, &[2, 3], DType::F16, "test").is_ok());
    let err = validate_buffer_capacity(11, 0, &[2, 3], DType::F16, "test");
    assert!(err.is_err());
}

#[test]
fn test_validate_buffer_capacity_overflow() {
    // usize::MAX × 2 overflows
    let err = validate_buffer_capacity(100, 0, &[usize::MAX, 2], DType::F32, "test");
    let msg = format!("{}", err.unwrap_err());
    assert!(msg.contains("overflow"), "got: {msg}");
}

#[test]
fn test_validate_buffer_capacity_empty_shape() {
    // scalar: product = 1, needs 4 bytes for F32
    assert!(validate_buffer_capacity(4, 0, &[], DType::F32, "test").is_ok());
}
