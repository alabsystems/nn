// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for [`to_u32`] helper.

use crate::to_u32;

#[test]
fn test_to_u32_zero() {
    let result = to_u32(0, "test").unwrap();
    assert_eq!(result, 0);
}

#[test]
fn test_to_u32_max_valid() {
    let result = to_u32(u32::MAX as usize, "test").unwrap();
    assert_eq!(result, u32::MAX);
}

#[test]
fn test_to_u32_overflow_by_one() {
    let result = to_u32(u32::MAX as usize + 1, "test");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, nn_core::TensorError::ValueOutOfRange { .. }));
    let msg = err.to_string();
    assert!(msg.contains("u32::MAX"), "error: {msg}");
}

#[test]
fn test_to_u32_usize_max() {
    let result = to_u32(usize::MAX, "grid_size");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, nn_core::TensorError::ValueOutOfRange { .. }));
}
