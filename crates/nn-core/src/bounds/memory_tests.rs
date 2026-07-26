// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Memory verification tests for IntervalBounds.
//!
//! These tests verify:
//! 1. `into_parts` produces independently usable arrays (no aliasing).
//! 2. `clone` produces a deep, independent copy (modifying clone doesn't
//!    affect original).
//! 3. `concrete` creates non-aliasing lower/upper arrays.
//! 4. `mark_infeasible_all` + `repair_invalid_inplace` round-trip.
//!
//! IBP arithmetic tests (mul, add, scale, shift) removed in #2005 —
//! arithmetic is provided by `ny_tensor::BoundedTensor`.

use super::*;
use ndarray::{arr1, Array1};

// --- Ownership and aliasing ---

/// Verify `into_parts` returns independently usable arrays that don't alias.
#[test]
fn test_into_parts_no_aliasing() {
    let bounds = IntervalBounds::new(
        arr1(&[1.0f32, 2.0, 3.0]).into_dyn(),
        arr1(&[4.0f32, 5.0, 6.0]).into_dyn(),
    )
    .expect("valid");

    let (mut lower, mut upper) = bounds.into_parts();

    // Modifying lower should not affect upper (no shared backing).
    lower[[0]] = 999.0;
    assert_eq!(upper[[0]], 4.0, "upper must be independent of lower");

    upper[[2]] = -999.0;
    assert_eq!(lower[[2]], 3.0, "lower must be independent of upper");
}

/// Verify `clone` produces a deep copy — modifying the clone does not
/// affect the original.
#[test]
fn test_clone_deep_independence() {
    let original = IntervalBounds::new(
        arr1(&[1.0f32, 2.0]).into_dyn(),
        arr1(&[3.0f32, 4.0]).into_dyn(),
    )
    .expect("valid");

    let cloned = original.clone();

    // Verify cloned values match original.
    assert_eq!(cloned.lower()[[0]], 1.0, "cloned lower matches original");
    assert_eq!(cloned.upper()[[0]], 3.0, "cloned upper matches original");

    // Use mark_infeasible_all to mutate the clone (no arithmetic needed).
    let mut modified = cloned;
    modified.mark_infeasible_all();

    // Original must be unaffected.
    assert_eq!(original.lower()[[0]], 1.0, "original lower unchanged");
    assert_eq!(original.upper()[[0]], 3.0, "original upper unchanged");

    // Modified clone has infeasible sentinels.
    assert_eq!(modified.lower()[[0]], f32::INFINITY);
    assert_eq!(modified.upper()[[0]], f32::NEG_INFINITY);
}

/// Verify `concrete` creates independent lower and upper arrays.
/// The `concrete` constructor clones `values` for lower and moves
/// original into upper. They must not alias.
#[test]
fn test_concrete_no_aliasing() {
    let data = arr1(&[10.0f32, 20.0, 30.0]).into_dyn();
    let bounds = IntervalBounds::concrete(data).expect("valid");

    // Lower and upper should be identical but independent.
    assert_eq!(bounds.lower()[[0]], bounds.upper()[[0]]);
    assert_eq!(bounds.lower()[[1]], bounds.upper()[[1]]);
    assert_eq!(bounds.lower()[[2]], bounds.upper()[[2]]);

    // Verify they are equal to the original values.
    assert_eq!(bounds.lower()[[0]], 10.0);
    assert_eq!(bounds.upper()[[2]], 30.0);
}

/// Verify `mark_infeasible_all` + `repair_invalid_inplace` round-trip
/// produces valid bounds (not the original values, but [-inf, +inf]).
#[test]
fn test_mark_infeasible_repair_round_trip() {
    let n = 100;
    let mut bounds = IntervalBounds::new(
        Array1::from(vec![1.0f32; n]).into_dyn(),
        Array1::from(vec![2.0f32; n]).into_dyn(),
    )
    .expect("valid");

    bounds.mark_infeasible_all();

    // After marking infeasible, lower=+inf, upper=-inf (inverted).
    assert_eq!(bounds.lower()[[0]], f32::INFINITY);
    assert_eq!(bounds.upper()[[0]], f32::NEG_INFINITY);

    let repaired = bounds.repair_invalid_inplace();
    assert_eq!(repaired, n, "all elements should be repaired");

    // After repair, bounds should be [-inf, +inf].
    for i in 0..n {
        assert_eq!(bounds.lower()[[i]], f32::NEG_INFINITY);
        assert_eq!(bounds.upper()[[i]], f32::INFINITY);
    }
}
