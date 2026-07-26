// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! IEEE 754 edge case tests for IntervalBounds.
//!
//! Covers negative zero in constructors, empty array bounds, and
//! from_epsilon with zero epsilon.
//!
//! Subnormal value tests extracted to `tests_ieee754_subnormal.rs`.
//!
//! IBP arithmetic edge case tests (scale, shift, mul with negative zero)
//! removed in #2005 — arithmetic is provided by `ny_tensor::BoundedTensor`.
//!
//! Part of #1685.

use super::*;
use ndarray::arr1;

// ---------------------------------------------------------------------------
// Negative zero in constructors
// ---------------------------------------------------------------------------

#[test]
fn test_new_negative_zero_lower_positive_zero_upper() {
    // IEEE 754: -0.0 == 0.0, so [-0.0, 0.0] is not inverted.
    let bounds = IntervalBounds::new(arr1(&[-0.0f32]).into_dyn(), arr1(&[0.0f32]).into_dyn())
        .expect("[-0.0, 0.0] should be accepted");
    // Both -0.0 and 0.0 compare equal, so lower <= upper.
    assert!(bounds.lower()[[0]] <= bounds.upper()[[0]]);
}

#[test]
fn test_new_positive_zero_lower_negative_zero_upper() {
    // [0.0, -0.0]: IEEE 754 says 0.0 == -0.0, and 0.0 > -0.0 is false.
    // So this should be accepted (not inverted).
    let bounds = IntervalBounds::new(arr1(&[0.0f32]).into_dyn(), arr1(&[-0.0f32]).into_dyn())
        .expect("[0.0, -0.0] should be accepted (IEEE 754: 0.0 == -0.0)");
    assert!(bounds.lower()[[0]] <= bounds.upper()[[0]]);
}

#[test]
fn test_concrete_negative_zero_accepted() {
    let bounds = IntervalBounds::concrete(arr1(&[-0.0f32]).into_dyn())
        .expect("concrete(-0.0) should be accepted");
    // Concrete means lower == upper. With -0.0, both are -0.0 but -0.0 == 0.0.
    assert_eq!(bounds.lower()[[0]], bounds.upper()[[0]]);
    assert!(bounds.max_width().abs() < f32::EPSILON);
}

#[test]
fn test_from_epsilon_negative_zero_center() {
    // -0.0 is finite and valid as a center value.
    let bounds = IntervalBounds::from_epsilon(arr1(&[-0.0f32]).into_dyn(), 1.0)
        .expect("from_epsilon with -0.0 center should be accepted");
    // -0.0 - 1.0 = -1.0, -0.0 + 1.0 = 1.0
    assert!((bounds.lower()[[0]] - (-1.0)).abs() < 1e-6);
    assert!((bounds.upper()[[0]] - 1.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Empty array bounds (AC4: 0-element bounds)
// ---------------------------------------------------------------------------

#[test]
fn test_new_empty_bounds_accepted() {
    use ndarray::ArrayD;
    let empty_lo = ArrayD::<f32>::zeros(ndarray::IxDyn(&[0]));
    let empty_hi = ArrayD::<f32>::zeros(ndarray::IxDyn(&[0]));
    let bounds = IntervalBounds::new(empty_lo, empty_hi).expect("empty bounds should be accepted");
    assert_eq!(bounds.lower().len(), 0);
    assert_eq!(bounds.upper().len(), 0);
}

#[test]
fn test_concrete_empty_accepted() {
    use ndarray::ArrayD;
    let empty = ArrayD::<f32>::zeros(ndarray::IxDyn(&[0]));
    let bounds = IntervalBounds::concrete(empty).expect("concrete empty should be accepted");
    assert_eq!(bounds.lower().len(), 0);
    assert_eq!(bounds.upper().len(), 0);
}

#[test]
fn test_from_epsilon_empty_accepted() {
    use ndarray::ArrayD;
    let empty = ArrayD::<f32>::zeros(ndarray::IxDyn(&[0]));
    let bounds =
        IntervalBounds::from_epsilon(empty, 1.0).expect("from_epsilon on empty should succeed");
    assert_eq!(bounds.lower().len(), 0);
    assert_eq!(bounds.upper().len(), 0);
}

#[test]
fn test_empty_bounds_max_width_is_zero() {
    use ndarray::ArrayD;
    let empty = ArrayD::<f32>::zeros(ndarray::IxDyn(&[0]));
    let bounds = IntervalBounds::new(empty.clone(), empty).expect("valid");
    // max_width fold on empty array returns initial value 0.0.
    assert_eq!(
        bounds.max_width(),
        0.0,
        "empty bounds max_width should be 0.0"
    );
}

// ---------------------------------------------------------------------------
// from_epsilon with zero epsilon (AC5)
// ---------------------------------------------------------------------------

#[test]
fn test_from_epsilon_zero_produces_concrete_bounds() {
    // from_epsilon(center, 0.0) should produce lower == upper == center.
    let center = arr1(&[3.0f32, -2.5, 0.0]).into_dyn();
    let bounds =
        IntervalBounds::from_epsilon(center.clone(), 0.0).expect("from_epsilon(_, 0.0) valid");
    for i in 0..3 {
        assert_eq!(
            bounds.lower()[[i]],
            center[[i]],
            "lower[{i}] should equal center"
        );
        assert_eq!(
            bounds.upper()[[i]],
            center[[i]],
            "upper[{i}] should equal center"
        );
    }
    assert!(
        bounds.max_width().abs() < f32::EPSILON,
        "zero epsilon should produce zero width, got {}",
        bounds.max_width()
    );
}

#[test]
fn test_from_epsilon_zero_various_centers() {
    // Positive center.
    let b1 = IntervalBounds::from_epsilon(arr1(&[42.0f32]).into_dyn(), 0.0)
        .expect("from_epsilon(42, 0.0) valid");
    assert_eq!(b1.lower()[[0]], 42.0);
    assert_eq!(b1.upper()[[0]], 42.0);

    // Negative center.
    let b2 = IntervalBounds::from_epsilon(arr1(&[-7.5f32]).into_dyn(), 0.0)
        .expect("from_epsilon(-7.5, 0.0) valid");
    assert_eq!(b2.lower()[[0]], -7.5);
    assert_eq!(b2.upper()[[0]], -7.5);
}
