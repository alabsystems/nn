// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! IEEE 754 subnormal value tests for IntervalBounds.
//!
//! Extracted from `tests_ieee754_edge.rs` for 500-line compliance.
//! Covers subnormal construction, ULP functions, and
//! round-for-soundness behavior with subnormal inputs.
//!
//! IBP arithmetic subnormal tests (mul, scale, add) removed in #2005 —
//! arithmetic is provided by `ny_tensor::BoundedTensor`.
//!
//! Part of #1685.

use super::*;
use ndarray::arr1;

// ---------------------------------------------------------------------------
// Subnormal values in constructors
// ---------------------------------------------------------------------------

#[test]
fn test_new_subnormal_bounds_accepted() {
    // Smallest positive subnormal: f32::from_bits(1) ≈ 1.4e-45.
    let tiny = f32::from_bits(1);
    assert!(tiny > 0.0 && tiny < f32::MIN_POSITIVE, "confirm subnormal");
    let bounds = IntervalBounds::new(arr1(&[tiny]).into_dyn(), arr1(&[tiny * 2.0]).into_dyn())
        .expect("subnormal bounds should be accepted");
    assert!(bounds.lower()[[0]] > 0.0);
    assert!(bounds.lower()[[0]] <= bounds.upper()[[0]]);
}

// ---------------------------------------------------------------------------
// ULP functions with negative zero and subnormal
// ---------------------------------------------------------------------------

#[test]
fn test_next_down_f32_zero_produces_negative_subnormal() {
    let result = next_down_f32(0.0);
    assert!(
        result < 0.0,
        "next_down(0.0) should be negative, got {result:e}"
    );
    // Should be the smallest negative subnormal: -f32::from_bits(1) = 0x80000001.
    assert_eq!(
        result.to_bits(),
        0x8000_0001,
        "next_down(0.0) should be -smallest_subnormal"
    );
}

#[test]
fn test_next_up_f32_zero_produces_positive_subnormal() {
    let result = next_up_f32(0.0);
    assert!(
        result > 0.0,
        "next_up(0.0) should be positive, got {result:e}"
    );
    assert_eq!(
        result.to_bits(),
        1,
        "next_up(0.0) should be smallest positive subnormal"
    );
}

#[test]
fn test_next_down_f32_negative_zero() {
    // next_down_f32 has `if x == 0.0` check. IEEE 754: -0.0 == 0.0 is true.
    // So next_down_f32(-0.0) should hit the same branch as next_down_f32(0.0).
    let result = next_down_f32(-0.0);
    assert!(
        result < 0.0,
        "next_down(-0.0) should be negative, got {result:e}"
    );
    assert_eq!(
        result.to_bits(),
        0x8000_0001,
        "next_down(-0.0) should be same as next_down(0.0)"
    );
}

#[test]
fn test_next_up_f32_negative_zero() {
    // next_up_f32(-0.0): -0.0 == 0.0 is true, so hits the zero branch.
    let result = next_up_f32(-0.0);
    assert!(
        result > 0.0,
        "next_up(-0.0) should be positive, got {result:e}"
    );
    assert_eq!(
        result.to_bits(),
        1,
        "next_up(-0.0) should be same as next_up(0.0)"
    );
}

#[test]
fn test_next_down_f32_smallest_positive_subnormal() {
    // next_down(smallest_positive_subnormal) should cross to zero or negative zero.
    let smallest = f32::from_bits(1);
    let result = next_down_f32(smallest);
    // bits(smallest) = 1, sign=positive, so bits-1 = 0 which is +0.0.
    assert_eq!(result, 0.0, "next_down(smallest subnormal) should be 0.0");
}

#[test]
fn test_next_up_f32_smallest_negative_subnormal() {
    // next_up(smallest negative subnormal = -f32::from_bits(1)) should be -0.0 or 0.0.
    let neg_smallest = f32::from_bits(0x8000_0001);
    let result = next_up_f32(neg_smallest);
    // sign=negative, bits=0x80000001, so bits-1=0x80000000 which is -0.0.
    assert!(
        result == 0.0 || result == -0.0,
        "next_up(smallest neg subnormal) should be ±0.0, got {result:e}"
    );
}

// ---------------------------------------------------------------------------
// Round-for-soundness with subnormal bounds
// ---------------------------------------------------------------------------

#[test]
fn test_round_for_soundness_subnormal_widens() {
    let tiny = f32::from_bits(100); // small subnormal
    let bounds = IntervalBounds::new(arr1(&[tiny]).into_dyn(), arr1(&[tiny * 2.0]).into_dyn())
        .expect("valid subnormal bounds");
    let rounded = bounds.round_for_soundness();
    assert!(
        rounded.lower()[[0]] < tiny,
        "lower should decrease below subnormal"
    );
    assert!(
        rounded.upper()[[0]] > tiny * 2.0,
        "upper should increase above subnormal"
    );
    assert!(
        rounded.lower()[[0]] <= rounded.upper()[[0]],
        "ordering preserved"
    );
}
