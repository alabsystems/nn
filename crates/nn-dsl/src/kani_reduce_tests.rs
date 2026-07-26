// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for reduce_sum_ref and reduce_mean_ref reference implementations.
//!
//! Extracted from kani_reduce.rs inline tests for 500-line compliance.

use super::*;

#[test]
fn test_reduce_sum_ref_basic() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = reduce_sum_ref(&input, 3).expect("valid test input");
    assert_eq!(result.len(), 2);
    assert!((result[0] - 6.0).abs() < 1e-6);
    assert!((result[1] - 15.0).abs() < 1e-6);
}

#[test]
fn test_reduce_mean_ref_basic() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = reduce_mean_ref(&input, 3).expect("valid test input");
    assert_eq!(result.len(), 2);
    assert!((result[0] - 2.0).abs() < 1e-6);
    assert!((result[1] - 5.0).abs() < 1e-6);
}

#[test]
fn test_reduce_sum_ref_single_element() {
    let input = [42.0];
    let result = reduce_sum_ref(&input, 1).expect("valid test input");
    assert_eq!(result, vec![42.0]);
}

#[test]
fn test_reduce_mean_ref_single_element() {
    let input = [42.0];
    let result = reduce_mean_ref(&input, 1).expect("valid test input");
    assert_eq!(result, vec![42.0]);
}

#[test]
fn test_reduce_sum_ref_full_reduce() {
    let input = [1.0, 2.0, 3.0, 4.0];
    let result = reduce_sum_ref(&input, 4).expect("valid test input");
    assert_eq!(result.len(), 1);
    assert!((result[0] - 10.0).abs() < 1e-6);
}

#[test]
fn test_reduce_mean_ref_full_reduce() {
    let input = [1.0, 2.0, 3.0, 4.0];
    let result = reduce_mean_ref(&input, 4).expect("valid test input");
    assert_eq!(result.len(), 1);
    assert!((result[0] - 2.5).abs() < 1e-6);
}

#[test]
fn test_reduce_mean_ref_all_same() {
    let input = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
    let result = reduce_mean_ref(&input, 3).expect("valid test input");
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], 5.0);
    assert_eq!(result[1], 5.0);
}

#[test]
fn test_reduce_sum_ref_zero_dim_returns_err() {
    let err = reduce_sum_ref(&[1.0, 2.0], 0).expect_err("should reject dim=0");
    assert!(matches!(
        err,
        KernelError::InvalidDimension {
            name: "reduce_dim",
            value: 0
        }
    ),);
}

#[test]
fn test_reduce_mean_ref_zero_dim_returns_err() {
    let err = reduce_mean_ref(&[1.0, 2.0], 0).expect_err("should reject dim=0");
    assert!(matches!(
        err,
        KernelError::InvalidDimension {
            name: "reduce_dim",
            value: 0
        }
    ),);
}

#[test]
fn test_reduce_sum_ref_misaligned_returns_err() {
    let err = reduce_sum_ref(&[1.0, 2.0, 3.0], 2).expect_err("should reject misaligned input");
    assert!(matches!(err, KernelError::ShapeMismatch { .. }));
}

#[test]
fn test_reduce_sum_ref_multi_slice() {
    // 3 slices of 4 elements
    let input = [
        1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, -1.0, -2.0, -3.0, -4.0,
    ];
    let result = reduce_sum_ref(&input, 4).expect("valid test input");
    assert_eq!(result.len(), 3);
    assert!((result[0] - 10.0).abs() < 1e-6);
    assert!((result[1] - 100.0).abs() < 1e-6);
    assert!((result[2] - (-10.0)).abs() < 1e-6);
}

#[test]
fn test_reduce_mean_ref_negative_values() {
    let input = [-4.0, -2.0, 0.0, 2.0];
    let result = reduce_mean_ref(&input, 4).expect("valid test input");
    assert_eq!(result.len(), 1);
    assert!((result[0] - (-1.0)).abs() < 1e-6);
}

/// Denormal inputs: reduce_sum must produce a finite result for subnormal f32 values.
/// Subnormals are the smallest representable non-zero values; arithmetic on them
/// can trigger flush-to-zero on some hardware, but Rust's software implementation
/// preserves them.
#[test]
fn test_reduce_sum_ref_denormal_inputs() {
    let tiny = f32::from_bits(1); // smallest positive subnormal: ~1.4e-45
    let input = [tiny, tiny, tiny, tiny];
    let result = reduce_sum_ref(&input, 4).expect("valid test input");
    assert_eq!(result.len(), 1);
    assert!(
        result[0] > 0.0,
        "sum of 4 positive subnormals should be positive, got {}",
        result[0]
    );
    assert!(result[0].is_finite(), "sum of subnormals must be finite");
}

/// Cancellation test: sum of values that nearly cancel. The sum of
/// [M, -M, eps] should give eps, not zero (within f32 tolerance).
#[test]
fn test_reduce_sum_ref_cancellation() {
    let big = 1.0e7_f32;
    let eps = 1.0_f32;
    let input = [big, -big, eps];
    let result = reduce_sum_ref(&input, 3).expect("valid test input");
    assert_eq!(result.len(), 1);
    // f32 addition is left-to-right: (big + (-big)) + eps = 0 + eps = eps
    assert!(
        (result[0] - eps).abs() < 1e-3,
        "sum should recover eps after cancellation, got {}",
        result[0]
    );
}

/// Mean of identical values should return that value exactly.
#[test]
fn test_reduce_mean_ref_identical_large() {
    let val = 1.0e6_f32;
    let input = [val; 8];
    let result = reduce_mean_ref(&input, 8).expect("valid test input");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], val, "mean of identical values should be exact");
}

/// Regression: reduce_dim > 2^24 returns DimensionExceedsF32Precision
/// instead of silently losing precision in the `reduce_dim as f32` cast.
#[test]
fn test_reduce_mean_ref_dim_exceeds_f32_precision_returns_err() {
    let err = reduce_mean_ref(&[1.0], (1 << 24) + 1).expect_err("should reject dim > 2^24");
    assert!(matches!(
        err,
        KernelError::DimensionExceedsF32Precision {
            name: "reduce_dim",
            ..
        }
    ));
}
