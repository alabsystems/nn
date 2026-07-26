// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `count_non_finite()` — verifies IEEE 754 classification.

use super::count_non_finite;

#[test]
fn test_empty_slice_returns_zero() {
    assert_eq!(count_non_finite(&[]), 0);
}

#[test]
fn test_all_finite_returns_zero() {
    assert_eq!(
        count_non_finite(&[1.0, -2.5, 0.0, f32::MAX, f32::MIN_POSITIVE]),
        0
    );
}

#[test]
fn test_subnormals_are_finite() {
    // Subnormal values (smaller than MIN_POSITIVE) ARE finite
    let subnormal = f32::MIN_POSITIVE * 0.5;
    assert!(subnormal > 0.0 && subnormal < f32::MIN_POSITIVE); // confirm subnormal
    assert_eq!(count_non_finite(&[subnormal, -subnormal]), 0);
}

#[test]
fn test_nan_counted() {
    assert_eq!(count_non_finite(&[1.0, f32::NAN, 3.0]), 1);
}

#[test]
fn test_all_non_finite_variants() {
    // NaN, +Inf, -Inf all counted
    assert_eq!(
        count_non_finite(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]),
        3
    );
}

#[test]
fn test_mixed_finite_and_non_finite() {
    let data = [
        1.0,
        f32::NAN,
        2.0,
        f32::INFINITY,
        3.0,
        f32::NEG_INFINITY,
        4.0,
    ];
    assert_eq!(count_non_finite(&data), 3);
}
