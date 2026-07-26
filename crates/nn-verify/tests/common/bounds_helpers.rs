// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for bounds contract parity tests.
//!
//! Used by `bounds_contract_tests.rs` and `bounds_contract_boundary_tests.rs`.

use nn_core::IntervalBounds;
use nn_verify::{to_bounded_tensor, BoundedTensor};
use ndarray::arr1;

/// Create singleton (1-element) IntervalBounds.
pub(crate) fn ib_singleton(lower: f32, upper: f32) -> IntervalBounds {
    IntervalBounds::new(arr1(&[lower]).into_dyn(), arr1(&[upper]).into_dyn())
        .expect("valid singleton IntervalBounds")
}

/// Create singleton (1-element) BoundedTensor.
pub(crate) fn bt_singleton(lower: f32, upper: f32) -> BoundedTensor {
    BoundedTensor::new(arr1(&[lower]).into_dyn(), arr1(&[upper]).into_dyn())
        .expect("valid singleton BoundedTensor")
}

pub(crate) fn ib_singleton_allow_inf(lower: f32, upper: f32) -> IntervalBounds {
    IntervalBounds::new_allow_infinite(arr1(&[lower]).into_dyn(), arr1(&[upper]).into_dyn())
        .expect("valid singleton IntervalBounds (allow inf)")
}

pub(crate) fn bt_singleton_allow_inf(lower: f32, upper: f32) -> BoundedTensor {
    BoundedTensor::new_allow_infinite(arr1(&[lower]).into_dyn(), arr1(&[upper]).into_dyn())
        .expect("valid singleton BoundedTensor (allow inf)")
}

/// Extract (lower, upper) scalar from IntervalBounds.
pub(crate) fn ib_extract(b: &IntervalBounds) -> (f32, f32) {
    (b.lower()[[0]], b.upper()[[0]])
}

/// Extract (lower, upper) scalar from BoundedTensor.
pub(crate) fn bt_extract(b: &BoundedTensor) -> (f32, f32) {
    let (lo, hi) = b.lower_upper();
    (lo[[0]], hi[[0]])
}

/// Assert IntervalBounds converted through bridge matches BoundedTensor.
pub(crate) fn assert_bridge_parity(ib: &IntervalBounds, bt: &BoundedTensor, context: &str) {
    let converted = to_bounded_tensor(ib.clone()).expect("bridge conversion should succeed");
    let (conv_lo, conv_hi) = converted.lower_upper();
    let (bt_lo, bt_hi) = bt.lower_upper();
    assert_eq!(
        conv_lo, bt_lo,
        "{context}: lower bounds mismatch after bridge conversion"
    );
    assert_eq!(
        conv_hi, bt_hi,
        "{context}: upper bounds mismatch after bridge conversion"
    );
}

/// Standard finite test inputs.
pub(crate) const TEST_PAIRS: &[(f32, f32)] = &[
    (0.0, 0.0),
    (1.0, 2.0),
    (-3.0, -1.0),
    (-1.0, 1.0),
    (0.5, 100.0),
    (-100.0, 100.0),
    (1.0e4, 2.0e4),
    (-2.0e4, -1.0e4),
    (1.0e-10, 1.0e-9),
];

pub(crate) const TEST_SCALARS: &[f32] = &[0.0, 1.0, -1.0, 2.5, -0.5, 100.0, -100.0, 1.0e-7];
