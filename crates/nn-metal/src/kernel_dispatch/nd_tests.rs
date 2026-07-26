// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for multi-dimensional dispatch and validation helpers.

use super::{checked_output_bytes, validate_input_lengths};
use crate::error::MetalError;

#[test]
fn test_checked_output_bytes_normal() {
    assert_eq!(checked_output_bytes(100, 4).expect("normal"), 400);
    assert_eq!(checked_output_bytes(0, 4).expect("zero"), 0);
    assert_eq!(checked_output_bytes(1, 2).expect("small"), 2);
}

#[test]
fn test_checked_output_bytes_overflow_returns_error() {
    let err = checked_output_bytes(usize::MAX, 4).unwrap_err();
    match err {
        MetalError::BufferByteOverflow { elems, elem_size } => {
            assert_eq!(elems, usize::MAX);
            assert_eq!(elem_size, 4);
        }
        other => panic!("expected BufferByteOverflow, got: {other}"),
    }
}

/// Boundary: largest element count that fits x 4 bytes succeeds;
/// one more overflows.
#[test]
fn test_checked_output_bytes_boundary() {
    let max_elems = usize::MAX / 4;
    assert_eq!(
        checked_output_bytes(max_elems, 4).expect("boundary"),
        max_elems * 4
    );

    let err = checked_output_bytes(max_elems + 1, 4).unwrap_err();
    assert!(matches!(err, MetalError::BufferByteOverflow { .. }));
}

/// Overflow with f16 element size (2 bytes).
#[test]
fn test_checked_output_bytes_overflow_f16() {
    let max_elems = usize::MAX / 2;
    assert_eq!(
        checked_output_bytes(max_elems, 2).expect("f16 boundary"),
        max_elems * 2
    );

    let err = checked_output_bytes(max_elems + 1, 2).unwrap_err();
    assert!(matches!(err, MetalError::BufferByteOverflow { .. }));
}

// --- validate_input_lengths ---

#[test]
fn test_validate_input_lengths_all_sufficient() {
    let a = [1.0f32; 100];
    let b = [2.0f32; 200];
    assert!(validate_input_lengths(&[&a, &b], 100).is_ok());
}

#[test]
fn test_validate_input_lengths_exact_match() {
    let a = [1.0f32; 64];
    assert!(validate_input_lengths(&[&a], 64).is_ok());
}

#[test]
fn test_validate_input_lengths_empty_inputs_zero_min() {
    let inputs: &[&[f32]] = &[];
    assert!(validate_input_lengths(inputs, 0).is_ok());
}

#[test]
fn test_validate_input_lengths_short_first_input() {
    let a = [1.0f32; 10];
    let b = [2.0f32; 100];
    let err = validate_input_lengths(&[&a, &b], 64).unwrap_err();
    match err {
        MetalError::InputLenMismatch {
            expected,
            got,
            index,
        } => {
            assert_eq!(expected, 64);
            assert_eq!(got, 10);
            assert_eq!(index, 0);
        }
        other => panic!("expected InputLenMismatch, got: {other}"),
    }
}

#[test]
fn test_validate_input_lengths_short_second_input() {
    let a = [1.0f32; 100];
    let b = [2.0f32; 5];
    let err = validate_input_lengths(&[&a, &b], 64).unwrap_err();
    match err {
        MetalError::InputLenMismatch {
            expected,
            got,
            index,
        } => {
            assert_eq!(expected, 64);
            assert_eq!(got, 5);
            assert_eq!(index, 1);
        }
        other => panic!("expected InputLenMismatch, got: {other}"),
    }
}

#[test]
fn test_validate_input_lengths_zero_min_always_passes() {
    let a = [1.0f32; 0];
    assert!(validate_input_lengths(&[&a], 0).is_ok());
}
