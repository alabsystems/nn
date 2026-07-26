// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `validated_elem_range` — the GPU readback bounds validator.
//!
//! `validated_elem_range` is called 6 times in the GPU→CPU transfer path,
//! guarding F32/BF16/F16/U32 buffer reads against misalignment, overflow,
//! and out-of-bounds access. Zero prior test coverage.

use nn_core::TensorError;

use super::validated_elem_range;

// ---------------------------------------------------------------------------
// Happy path: valid range computation
// ---------------------------------------------------------------------------

#[test]
fn test_valid_range_from_zero_offset() {
    let (start, end) = validated_elem_range(0, 4, 10, 100).expect("valid");
    assert_eq!(start, 0);
    assert_eq!(end, 10);
}

#[test]
fn test_valid_range_with_byte_offset() {
    // byte_offset=16, elem_size=4 → start=4, numel=5 → end=9
    let (start, end) = validated_elem_range(16, 4, 5, 100).expect("valid");
    assert_eq!(start, 4);
    assert_eq!(end, 9);
}

#[test]
fn test_valid_range_bf16_element_size() {
    // BF16/F16: elem_size=2, byte_offset=8 → start=4, numel=3 → end=7
    let (start, end) = validated_elem_range(8, 2, 3, 100).expect("valid");
    assert_eq!(start, 4);
    assert_eq!(end, 7);
}

#[test]
fn test_valid_range_exact_fit() {
    // buf_len=10, requesting exactly [0..10)
    let (start, end) = validated_elem_range(0, 4, 10, 10).expect("valid");
    assert_eq!(start, 0);
    assert_eq!(end, 10);
}

#[test]
fn test_valid_range_exact_fit_with_offset() {
    // byte_offset=20, elem_size=4 → start=5, numel=5 → end=10, buf_len=10
    let (start, end) = validated_elem_range(20, 4, 5, 10).expect("valid");
    assert_eq!(start, 5);
    assert_eq!(end, 10);
}

#[test]
fn test_valid_range_zero_numel() {
    // Zero-element tensor: start == end is valid.
    let (start, end) = validated_elem_range(0, 4, 0, 100).expect("valid");
    assert_eq!(start, 0);
    assert_eq!(end, 0);
}

// ---------------------------------------------------------------------------
// Alignment validation: misaligned byte_offset
// ---------------------------------------------------------------------------

#[test]
fn test_misaligned_f32_offset_returns_error() {
    // byte_offset=3 not aligned to elem_size=4
    let result = validated_elem_range(3, 4, 10, 100);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, TensorError::ValueOutOfRange { .. }),
        "expected ValueOutOfRange for misaligned offset, got: {err:?}"
    );
}

#[test]
fn test_misaligned_bf16_offset_returns_error() {
    // byte_offset=3 not aligned to elem_size=2
    let result = validated_elem_range(3, 2, 10, 100);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        TensorError::ValueOutOfRange { .. }
    ));
}

#[test]
fn test_misaligned_by_one_byte_returns_error() {
    // byte_offset=1 not aligned to elem_size=4
    let result = validated_elem_range(1, 4, 10, 100);
    assert!(result.is_err());
}

#[test]
fn test_aligned_offsets_succeed() {
    // All valid alignments for F32 (elem_size=4)
    for off in (0..=20).step_by(4) {
        assert!(
            validated_elem_range(off, 4, 1, 100).is_ok(),
            "byte_offset={off} should be aligned to elem_size=4"
        );
    }
    // All valid alignments for BF16 (elem_size=2)
    for off in (0..=20).step_by(2) {
        assert!(
            validated_elem_range(off, 2, 1, 100).is_ok(),
            "byte_offset={off} should be aligned to elem_size=2"
        );
    }
}

// ---------------------------------------------------------------------------
// Overflow protection: start + numel overflow
// ---------------------------------------------------------------------------

#[test]
fn test_numel_overflow_returns_error() {
    // start=0, numel=usize::MAX → 0 + usize::MAX is fine (no overflow)
    // But end > buf_len, so it fails with DataLengthMismatch.
    let result = validated_elem_range(0, 4, usize::MAX, 100);
    assert!(result.is_err());
}

#[test]
fn test_start_plus_numel_overflow_returns_error() {
    // byte_offset=4, elem_size=4 → start=1. numel=usize::MAX → 1 + MAX overflows.
    let result = validated_elem_range(4, 4, usize::MAX, usize::MAX);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        TensorError::DimensionOverflow { .. }
    ));
}

// ---------------------------------------------------------------------------
// Bounds validation: end > buf_len
// ---------------------------------------------------------------------------

#[test]
fn test_end_exceeds_buf_len_returns_error() {
    // start=0, numel=11, buf_len=10 → end=11 > 10
    let result = validated_elem_range(0, 4, 11, 10);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        TensorError::DataLengthMismatch { .. }
    ));
}

#[test]
fn test_offset_pushes_end_past_buf_len() {
    // byte_offset=20 → start=5, numel=6 → end=11, buf_len=10
    let result = validated_elem_range(20, 4, 6, 10);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        TensorError::DataLengthMismatch { .. }
    ));
}

#[test]
fn test_one_past_end_returns_error() {
    // Exact fit is OK (end == buf_len), one past is not.
    assert!(validated_elem_range(0, 4, 10, 10).is_ok());
    assert!(validated_elem_range(0, 4, 11, 10).is_err());
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_single_element_at_start() {
    let (start, end) = validated_elem_range(0, 4, 1, 1).expect("valid");
    assert_eq!(start, 0);
    assert_eq!(end, 1);
}

#[test]
fn test_single_element_at_end() {
    // byte_offset=36, elem_size=4 → start=9, numel=1 → end=10, buf_len=10
    let (start, end) = validated_elem_range(36, 4, 1, 10).expect("valid");
    assert_eq!(start, 9);
    assert_eq!(end, 10);
}

#[test]
fn test_large_aligned_offset() {
    // 1 MB offset, 4-byte elements → start=262144
    let (start, end) = validated_elem_range(1024 * 1024, 4, 100, 262144 + 100).expect("valid");
    assert_eq!(start, 262144);
    assert_eq!(end, 262244);
}

// ---------------------------------------------------------------------------
// retype_kernel: ensure KernelDef params/return type are rewritten
// ---------------------------------------------------------------------------

#[test]
fn test_retype_kernel_f32_is_noop() {
    use nn_dsl::ir::{KernelDef, NodeId, Param, ScalarType};
    let def = KernelDef::new(
        "test_op",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("y", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![],
        NodeId::new(0),
    );
    let result = super::retype_kernel(def, ScalarType::F32);
    assert_eq!(result.params[0].ty, ScalarType::F32);
    assert_eq!(result.params[1].ty, ScalarType::F32);
    assert_eq!(result.return_type, ScalarType::F32);
}

#[test]
fn test_retype_kernel_to_f16() {
    use nn_dsl::ir::{KernelDef, NodeId, Param, ScalarType};
    let def = KernelDef::new(
        "test_op",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("y", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![],
        NodeId::new(0),
    );
    let result = super::retype_kernel(def, ScalarType::F16);
    assert_eq!(result.params[0].ty, ScalarType::F16);
    assert_eq!(result.params[1].ty, ScalarType::F16);
    assert_eq!(result.return_type, ScalarType::F16);
}

#[test]
fn test_retype_kernel_preserves_name() {
    use nn_dsl::ir::{KernelDef, NodeId, Param, ScalarType};
    let def = KernelDef::new(
        "nn_kernel",
        vec![Param::new("a", ScalarType::F32)],
        ScalarType::F32,
        vec![],
        NodeId::new(0),
    );
    let result = super::retype_kernel(def, ScalarType::F16);
    assert_eq!(result.name, "nn_kernel");
    assert_eq!(result.params[0].name, "a");
}
