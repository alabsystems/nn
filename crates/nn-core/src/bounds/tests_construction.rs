// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Construction and validation tests for IntervalBounds: new, from_epsilon,
//! concrete, mark_infeasible, enforce_bound_ordering, max_width, accessors.

use super::*;
use ndarray::arr1;

// --- mark_infeasible_all + repair_invalid_inplace ---

#[test]
fn test_mark_infeasible_and_repair() {
    let lower = arr1(&[1.0f32, 2.0]).into_dyn();
    let upper = arr1(&[3.0f32, 4.0]).into_dyn();
    let mut bounds = IntervalBounds::new(lower, upper).expect("valid");

    bounds.mark_infeasible_all();
    assert_eq!(bounds.lower()[[0]], f32::INFINITY);
    assert_eq!(bounds.upper()[[0]], f32::NEG_INFINITY);

    let repaired = bounds.repair_invalid_inplace();
    assert_eq!(repaired, 2, "both elements should be repaired");
    assert_eq!(bounds.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(bounds.upper()[[0]], f32::INFINITY);
    assert!(bounds.lower()[[0]] <= bounds.upper()[[0]]);
}

#[test]
fn test_mark_infeasible_single_element() {
    let mut bounds = IntervalBounds::new(arr1(&[5.0f32]).into_dyn(), arr1(&[10.0f32]).into_dyn())
        .expect("valid");
    bounds.mark_infeasible_all();
    assert_eq!(bounds.lower()[[0]], f32::INFINITY);
    assert_eq!(bounds.upper()[[0]], f32::NEG_INFINITY);
}

#[test]
fn test_repair_invalid_inplace_already_valid_returns_zero() {
    let mut bounds = IntervalBounds::new(
        arr1(&[1.0f32, 2.0]).into_dyn(),
        arr1(&[3.0f32, 4.0]).into_dyn(),
    )
    .expect("valid");
    let repaired = bounds.repair_invalid_inplace();
    assert_eq!(repaired, 0, "no elements need repair");
    assert_eq!(bounds.lower()[[0]], 1.0, "unchanged");
    assert_eq!(bounds.upper()[[1]], 4.0, "unchanged");
}

// --- from_epsilon error paths ---

#[test]
fn test_from_epsilon_nan_center_rejected() {
    let center = arr1(&[f32::NAN]).into_dyn();
    let err = IntervalBounds::from_epsilon(center, 0.1).expect_err("NaN center should be rejected");
    assert!(
        format!("{err}").contains("NaN or Inf in center"),
        "expected NaN/Inf center error, got: {err}"
    );
}

#[test]
fn test_from_epsilon_negative_epsilon_rejected() {
    let center = arr1(&[1.0f32]).into_dyn();
    let err = IntervalBounds::from_epsilon(center, -0.1)
        .expect_err("negative epsilon should be rejected");
    assert!(
        format!("{err}").contains("epsilon must be non-negative"),
        "expected epsilon validation error, got: {err}"
    );
}

#[test]
fn test_from_epsilon_inf_epsilon_rejected() {
    let center = arr1(&[1.0f32]).into_dyn();
    let err = IntervalBounds::from_epsilon(center, f32::INFINITY)
        .expect_err("infinite epsilon should be rejected");
    assert!(
        format!("{err}").contains("epsilon must be non-negative and finite"),
        "expected epsilon validation error, got: {err}"
    );
}

#[test]
fn test_from_epsilon_nan_epsilon_rejected() {
    let center = arr1(&[1.0f32]).into_dyn();
    let err =
        IntervalBounds::from_epsilon(center, f32::NAN).expect_err("NaN epsilon should be rejected");
    assert!(
        format!("{err}").contains("epsilon must be non-negative and finite"),
        "expected epsilon validation error, got: {err}"
    );
}

#[test]
fn test_from_epsilon_overflow_clamped() {
    let center = arr1(&[f32::MAX]).into_dyn();
    let bounds = IntervalBounds::from_epsilon(center, 1.0).expect("should clamp, not fail");
    assert_eq!(bounds.upper()[[0]], f32::MAX);
}

#[test]
fn test_from_epsilon_underflow_clamped() {
    let center = arr1(&[f32::MIN]).into_dyn();
    let bounds = IntervalBounds::from_epsilon(center, 1.0).expect("should clamp, not fail");
    assert_eq!(bounds.lower()[[0]], f32::MIN);
}

// --- concrete error paths ---

#[test]
fn test_concrete_nan_rejected() {
    let values = arr1(&[f32::NAN]).into_dyn();
    let err = IntervalBounds::concrete(values).expect_err("NaN should be rejected");
    assert!(
        format!("{err}").contains("NaN or Inf in concrete"),
        "expected concrete validation error, got: {err}"
    );
}

#[test]
fn test_concrete_inf_rejected() {
    let values = arr1(&[f32::INFINITY]).into_dyn();
    let err = IntervalBounds::concrete(values).expect_err("Inf should be rejected");
    assert!(
        format!("{err}").contains("NaN or Inf in concrete"),
        "expected concrete validation error, got: {err}"
    );
}

#[test]
fn test_concrete_neg_inf_rejected() {
    let values = arr1(&[f32::NEG_INFINITY]).into_dyn();
    let err = IntervalBounds::concrete(values).expect_err("-Inf should be rejected");
    assert!(
        format!("{err}").contains("NaN or Inf in concrete"),
        "expected concrete validation error, got: {err}"
    );
}

#[test]
fn test_concrete_zero_width() {
    let values = arr1(&[1.0f32, 2.0, 3.0]).into_dyn();
    let bounds = IntervalBounds::concrete(values).expect("valid");
    assert!(
        (bounds.max_width() - 0.0).abs() < f32::EPSILON,
        "concrete bounds should have zero width"
    );
}

// --- enforce_bound_ordering returns repair count (#210 AC2) ---

#[test]
fn test_enforce_bound_ordering_returns_repair_count() {
    let mut lower = arr1(&[1.0f32, f32::NAN, 5.0]).into_dyn();
    let mut upper = arr1(&[2.0f32, 3.0, 4.0]).into_dyn(); // elem 2: inverted (5 > 4)
    let count = enforce_bound_ordering(&mut lower, &mut upper);
    assert_eq!(
        count, 2,
        "NaN element and inverted element should be repaired"
    );
    // Element 0: untouched
    assert_eq!(lower[[0]], 1.0);
    assert_eq!(upper[[0]], 2.0);
    // Element 1: NaN lower → repaired
    assert_eq!(lower[[1]], -FALLBACK_BOUND);
    assert_eq!(upper[[1]], FALLBACK_BOUND);
    // Element 2: inverted → repaired
    assert_eq!(lower[[2]], -FALLBACK_BOUND);
    assert_eq!(upper[[2]], FALLBACK_BOUND);
}

#[test]
fn test_enforce_bound_ordering_zero_repairs_for_valid() {
    let mut lower = arr1(&[1.0f32, 2.0]).into_dyn();
    let mut upper = arr1(&[3.0f32, 4.0]).into_dyn();
    let count = enforce_bound_ordering(&mut lower, &mut upper);
    assert_eq!(count, 0, "valid bounds should need zero repairs");
}

// --- max_width edge cases ---

#[test]
fn test_max_width_uniform() {
    let bounds = IntervalBounds::new(
        arr1(&[0.0f32, 0.0, 0.0]).into_dyn(),
        arr1(&[1.0f32, 1.0, 1.0]).into_dyn(),
    )
    .expect("valid");
    assert!((bounds.max_width() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_max_width_single_element() {
    let bounds = IntervalBounds::new(arr1(&[5.0f32]).into_dyn(), arr1(&[15.0f32]).into_dyn())
        .expect("valid");
    assert!((bounds.max_width() - 10.0).abs() < f32::EPSILON);
}

#[test]
fn test_max_width_same_infinity_returns_inf() {
    let bounds = IntervalBounds::new_allow_infinite(
        arr1(&[f32::INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )
    .expect("same-inf bounds are valid");
    let w = bounds.max_width();
    assert!(
        w.is_infinite(),
        "max_width of [+Inf, +Inf] should be Inf (not 0.0), got {w}"
    );
}

#[test]
fn test_max_width_neg_infinity_both_returns_inf() {
    let bounds = IntervalBounds::new_allow_infinite(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
    )
    .expect("same -Inf bounds are valid");
    let w = bounds.max_width();
    assert!(
        w.is_infinite(),
        "max_width of [-Inf, -Inf] should be Inf (not 0.0), got {w}"
    );
}

#[test]
fn test_max_width_mixed_inf_returns_inf() {
    let bounds = IntervalBounds::new_allow_infinite(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )
    .expect("valid infinite range");
    let w = bounds.max_width();
    assert!(
        w.is_infinite(),
        "max_width of [-Inf, +Inf] should be Inf, got {w}"
    );
}

// --- Tests migrated from bounds.rs inline mod tests ---

#[test]
fn test_from_epsilon() {
    let center = arr1(&[1.0f32, 2.0, 3.0]).into_dyn();
    let bounds = IntervalBounds::from_epsilon(center, 0.1).expect("valid epsilon bounds");
    assert!((bounds.lower()[[0]] - 0.9).abs() < 1e-6);
    assert!((bounds.upper()[[0]] - 1.1).abs() < 1e-6);
}

#[test]
fn test_concrete_bounds() {
    let values = arr1(&[1.0f32, 2.0, 3.0]).into_dyn();
    let bounds = IntervalBounds::concrete(values).expect("valid concrete bounds");
    assert_eq!(bounds.lower()[[0]], 1.0);
    assert_eq!(bounds.upper()[[0]], 1.0);
    assert_eq!(bounds.lower()[[2]], 3.0);
    assert_eq!(bounds.upper()[[2]], 3.0);
}

#[test]
fn test_nan_rejected() {
    let lower = arr1(&[f32::NAN, 0.5]).into_dyn();
    let upper = arr1(&[1.0f32, 1.5]).into_dyn();
    let err = IntervalBounds::new(lower, upper).expect_err("NaN should be rejected");
    assert!(
        format!("{err}").contains("NaN or Inf"),
        "expected NaN/Inf error, got: {err}"
    );
}

#[test]
fn test_inverted_rejected() {
    let lower = arr1(&[2.0f32]).into_dyn();
    let upper = arr1(&[1.0f32]).into_dyn();
    let err = IntervalBounds::new(lower, upper).expect_err("inverted bounds should be rejected");
    assert!(
        format!("{err}").contains("inverted bounds"),
        "expected inverted bounds error, got: {err}"
    );
}

#[test]
fn test_inf_rejected_in_new() {
    let lower = arr1(&[0.0f32]).into_dyn();
    let upper = arr1(&[f32::INFINITY]).into_dyn();
    let err = IntervalBounds::new(lower, upper).expect_err("Inf should be rejected");
    assert!(
        format!("{err}").contains("NaN or Inf"),
        "expected NaN/Inf error, got: {err}"
    );
}

#[test]
fn test_inf_allowed_in_new_allow_infinite() {
    let lower = arr1(&[f32::NEG_INFINITY]).into_dyn();
    let upper = arr1(&[f32::INFINITY]).into_dyn();
    let bounds = IntervalBounds::new_allow_infinite(lower, upper)
        .expect("infinite bounds should be allowed");
    assert_eq!(bounds.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(bounds.upper()[[0]], f32::INFINITY);
}

#[test]
fn test_max_width() {
    let lower = arr1(&[0.0f32, 1.0, 2.0]).into_dyn();
    let upper = arr1(&[1.0f32, 3.0, 2.5]).into_dyn();
    let bounds = IntervalBounds::new(lower, upper).expect("valid bounds");
    assert!((bounds.max_width() - 2.0).abs() < 1e-6);
}

#[test]
fn test_shape_mismatch_rejected() {
    let lower = arr1(&[0.0f32, 1.0]).into_dyn();
    let upper = arr1(&[1.0f32]).into_dyn();
    let err = IntervalBounds::new(lower, upper).expect_err("shape mismatch should be rejected");
    assert!(
        format!("{err}").contains("shape mismatch"),
        "expected shape mismatch error, got: {err}"
    );
}

#[test]
fn test_into_parts() {
    let lower = arr1(&[0.0f32]).into_dyn();
    let upper = arr1(&[1.0f32]).into_dyn();
    let bounds = IntervalBounds::new(lower.clone(), upper.clone()).expect("valid");
    let (l, u) = bounds.into_parts();
    assert_eq!(l, lower);
    assert_eq!(u, upper);
}

#[test]
fn test_lower_upper_accessors() {
    let lower = arr1(&[0.0f32]).into_dyn();
    let upper = arr1(&[1.0f32]).into_dyn();
    let bounds = IntervalBounds::new(lower, upper).expect("valid");
    let (l, u) = bounds.lower_upper();
    assert_eq!(l[[0]], 0.0);
    assert_eq!(u[[0]], 1.0);
}

#[test]
fn test_new_rejects_nan_in_first_element() {
    let err = IntervalBounds::new(
        arr1(&[f32::NAN, 1.0, 2.0]).into_dyn(),
        arr1(&[1.0f32, 2.0, 3.0]).into_dyn(),
    )
    .expect_err("NaN rejected");
    assert!(format!("{err}").contains("NaN or Inf"));
}

#[test]
fn test_new_rejects_inverted_at_last_element() {
    let err = IntervalBounds::new(
        arr1(&[0.0f32, 0.0, 5.0]).into_dyn(),
        arr1(&[1.0f32, 1.0, 4.0]).into_dyn(),
    )
    .expect_err("inverted rejected");
    assert!(format!("{err}").contains("inverted"));
}
