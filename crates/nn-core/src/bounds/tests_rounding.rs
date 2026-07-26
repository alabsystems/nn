// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Rounding and ULP tests for IntervalBounds: round_for_soundness,
//! next_down_f32, next_up_f32, NaN repair, infeasible sentinel preservation.

use super::*;
use ndarray::arr1;

// --- round_for_soundness (exercises next_down_f32/next_up_f32) ---

#[test]
fn test_round_for_soundness_widens_by_ulp() {
    let bounds =
        IntervalBounds::new(arr1(&[1.0f32]).into_dyn(), arr1(&[2.0f32]).into_dyn()).expect("valid");
    let rounded = bounds.round_for_soundness();
    assert!(rounded.lower()[[0]] < 1.0, "lower should decrease");
    assert!(rounded.upper()[[0]] > 2.0, "upper should increase");
    assert!((rounded.lower()[[0]] - 1.0).abs() < 1e-6);
    assert!((rounded.upper()[[0]] - 2.0).abs() < 1e-6);
}

#[test]
fn test_round_for_soundness_negative_values() {
    let bounds = IntervalBounds::new(arr1(&[-3.0f32]).into_dyn(), arr1(&[-1.0f32]).into_dyn())
        .expect("valid");
    let rounded = bounds.round_for_soundness();
    assert!(
        rounded.lower()[[0]] < -3.0,
        "lower should decrease further negative"
    );
    assert!(
        rounded.upper()[[0]] > -1.0,
        "upper should increase toward zero"
    );
}

#[test]
fn test_round_for_soundness_zero_crossing() {
    let bounds =
        IntervalBounds::new(arr1(&[0.0f32]).into_dyn(), arr1(&[0.0f32]).into_dyn()).expect("valid");
    let rounded = bounds.round_for_soundness();
    assert!(
        rounded.lower()[[0]] < 0.0,
        "lower crosses to negative subnormal"
    );
    assert!(
        rounded.upper()[[0]] > 0.0,
        "upper crosses to positive subnormal"
    );
}

// --- round_for_soundness infeasible sentinel preservation (#171) ---

#[test]
fn test_round_for_soundness_preserves_infeasible_sentinel() {
    let lower = arr1(&[1.0f32, 2.0]).into_dyn();
    let upper = arr1(&[3.0f32, 4.0]).into_dyn();
    let mut bounds = IntervalBounds::new(lower, upper).expect("valid");

    bounds.mark_infeasible_all();
    assert_eq!(bounds.lower()[[0]], f32::INFINITY);
    assert_eq!(bounds.upper()[[0]], f32::NEG_INFINITY);

    let rounded = bounds.round_for_soundness();
    assert_eq!(
        rounded.lower()[[0]],
        f32::INFINITY,
        "next_down_f32(+inf) must return +inf to preserve infeasible sentinel"
    );
    assert_eq!(
        rounded.upper()[[0]],
        f32::NEG_INFINITY,
        "next_up_f32(-inf) must return -inf to preserve infeasible sentinel"
    );
    assert_eq!(
        rounded.lower()[[1]],
        f32::INFINITY,
        "all elements should preserve sentinel"
    );
    assert_eq!(
        rounded.upper()[[1]],
        f32::NEG_INFINITY,
        "all elements should preserve sentinel"
    );
}

#[test]
fn test_next_down_f32_preserves_positive_infinity() {
    let result = next_down_f32(f32::INFINITY);
    assert_eq!(
        result,
        f32::INFINITY,
        "next_down_f32(+inf) must return +inf, got {result}"
    );
}

#[test]
fn test_next_up_f32_preserves_negative_infinity() {
    let result = next_up_f32(f32::NEG_INFINITY);
    assert_eq!(
        result,
        f32::NEG_INFINITY,
        "next_up_f32(-inf) must return -inf, got {result}"
    );
}

#[test]
fn test_next_down_f32_neg_infinity_is_identity() {
    let result = next_down_f32(f32::NEG_INFINITY);
    assert_eq!(result, f32::NEG_INFINITY);
}

#[test]
fn test_next_up_f32_pos_infinity_is_identity() {
    let result = next_up_f32(f32::INFINITY);
    assert_eq!(result, f32::INFINITY);
}

// --- round_for_soundness extreme finite boundary (#234) ---

#[test]
fn test_round_for_soundness_f32_max_introduces_infinity() {
    let bounds = IntervalBounds::new(arr1(&[f32::MAX]).into_dyn(), arr1(&[f32::MAX]).into_dyn())
        .expect("concrete MAX bounds");
    let rounded = bounds.round_for_soundness();
    assert!(
        rounded.upper()[[0]].is_infinite(),
        "next_up_f32(MAX) = Inf, so upper becomes infinite, got {}",
        rounded.upper()[[0]]
    );
    assert!(
        rounded.lower()[[0]] < f32::MAX,
        "lower should be one ULP below MAX, got {}",
        rounded.lower()[[0]]
    );
    assert!(
        rounded.lower()[[0]].is_finite(),
        "lower should remain finite (one ULP below MAX)"
    );
    assert!(
        rounded.lower()[[0]] <= rounded.upper()[[0]],
        "ordering invariant: lower <= upper"
    );
}

#[test]
fn test_round_for_soundness_f32_min_introduces_neg_infinity() {
    let bounds = IntervalBounds::new(arr1(&[f32::MIN]).into_dyn(), arr1(&[f32::MIN]).into_dyn())
        .expect("concrete MIN bounds");
    let rounded = bounds.round_for_soundness();
    assert!(
        rounded.lower()[[0]].is_infinite() && rounded.lower()[[0]] < 0.0,
        "next_down_f32(MIN) = -Inf, so lower becomes -infinity, got {}",
        rounded.lower()[[0]]
    );
    assert!(
        rounded.upper()[[0]] > f32::MIN,
        "upper should be one ULP above MIN, got {}",
        rounded.upper()[[0]]
    );
    assert!(
        rounded.upper()[[0]].is_finite(),
        "upper should remain finite (one ULP above MIN)"
    );
    assert!(
        rounded.lower()[[0]] <= rounded.upper()[[0]],
        "ordering invariant: lower <= upper"
    );
}

#[test]
fn test_round_for_soundness_f32_max_to_min_range() {
    // The widest possible finite interval: [MIN, MAX]
    let bounds = IntervalBounds::new(arr1(&[f32::MIN]).into_dyn(), arr1(&[f32::MAX]).into_dyn())
        .expect("valid full finite range");
    let rounded = bounds.round_for_soundness();
    // Both endpoints should become infinite after ULP widening
    assert!(
        rounded.lower()[[0]] == f32::NEG_INFINITY,
        "next_down_f32(MIN) = -Inf, got {}",
        rounded.lower()[[0]]
    );
    assert!(
        rounded.upper()[[0]] == f32::INFINITY,
        "next_up_f32(MAX) = +Inf, got {}",
        rounded.upper()[[0]]
    );
    assert!(
        rounded.lower()[[0]] <= rounded.upper()[[0]],
        "ordering invariant: lower <= upper"
    );
}

// --- round_for_soundness NaN repair (#202) ---

#[test]
fn test_round_for_soundness_repairs_nan_to_fallback() {
    // Construct bounds with NaN directly (bypasses constructor validation).
    // In practice NaN could arrive via a bug in an intermediate arithmetic
    // operation. round_for_soundness must not propagate it.
    let bounds = IntervalBounds {
        lower: arr1(&[f32::NAN]).into_dyn(),
        upper: arr1(&[f32::NAN]).into_dyn(),
    };
    let rounded = bounds.round_for_soundness();
    assert!(
        !rounded.lower()[[0]].is_nan(),
        "NaN lower must be repaired, got {}",
        rounded.lower()[[0]]
    );
    assert!(
        !rounded.upper()[[0]].is_nan(),
        "NaN upper must be repaired, got {}",
        rounded.upper()[[0]]
    );
    assert_eq!(
        rounded.lower()[[0]],
        -FALLBACK_BOUND,
        "NaN lower should become -FALLBACK_BOUND"
    );
    assert_eq!(
        rounded.upper()[[0]],
        FALLBACK_BOUND,
        "NaN upper should become +FALLBACK_BOUND"
    );
    assert!(
        rounded.lower()[[0]] <= rounded.upper()[[0]],
        "repaired bounds must satisfy lower <= upper"
    );
}

#[test]
fn test_round_for_soundness_repairs_nan_lower_only() {
    // Only the lower bound is NaN — both should be repaired to fallback.
    let bounds = IntervalBounds {
        lower: arr1(&[f32::NAN]).into_dyn(),
        upper: arr1(&[2.0f32]).into_dyn(),
    };
    let rounded = bounds.round_for_soundness();
    assert_eq!(
        rounded.lower()[[0]],
        -FALLBACK_BOUND,
        "NaN lower should become -FALLBACK_BOUND"
    );
    assert_eq!(
        rounded.upper()[[0]],
        FALLBACK_BOUND,
        "when lower is NaN, upper is also widened to FALLBACK_BOUND"
    );
}

#[test]
fn test_round_for_soundness_repairs_nan_upper_only() {
    // Mirror of nan_lower_only: only the upper bound is NaN.
    // Both should be repaired to fallback since repair_nan_to_fallback
    // triggers on either endpoint being NaN.
    let bounds = IntervalBounds {
        lower: arr1(&[2.0f32]).into_dyn(),
        upper: arr1(&[f32::NAN]).into_dyn(),
    };
    let rounded = bounds.round_for_soundness();
    assert_eq!(
        rounded.lower()[[0]],
        -FALLBACK_BOUND,
        "when upper is NaN, lower should also widen to -FALLBACK_BOUND"
    );
    assert_eq!(
        rounded.upper()[[0]],
        FALLBACK_BOUND,
        "NaN upper should become +FALLBACK_BOUND"
    );
}

// --- repair_non_finite_lower/upper with infinity inputs (P1]71 audit) ---

#[test]
fn test_repair_non_finite_lower_on_positive_infinity() {
    assert_eq!(
        repair_non_finite_lower(f32::INFINITY),
        -FALLBACK_BOUND,
        "+Inf lower should be repaired to -FALLBACK_BOUND"
    );
}

#[test]
fn test_repair_non_finite_lower_on_negative_infinity() {
    assert_eq!(
        repair_non_finite_lower(f32::NEG_INFINITY),
        -FALLBACK_BOUND,
        "-Inf lower should be repaired to -FALLBACK_BOUND"
    );
}

#[test]
fn test_repair_non_finite_upper_on_positive_infinity() {
    assert_eq!(
        repair_non_finite_upper(f32::INFINITY),
        FALLBACK_BOUND,
        "+Inf upper should be repaired to +FALLBACK_BOUND"
    );
}

#[test]
fn test_repair_non_finite_upper_on_negative_infinity() {
    assert_eq!(
        repair_non_finite_upper(f32::NEG_INFINITY),
        FALLBACK_BOUND,
        "-Inf upper should be repaired to +FALLBACK_BOUND"
    );
}

#[test]
fn test_repair_non_finite_lower_preserves_finite() {
    assert_eq!(repair_non_finite_lower(42.0), 42.0);
    assert_eq!(repair_non_finite_lower(-42.0), -42.0);
    assert_eq!(repair_non_finite_lower(0.0), 0.0);
}

#[test]
fn test_repair_non_finite_upper_preserves_finite() {
    assert_eq!(repair_non_finite_upper(42.0), 42.0);
    assert_eq!(repair_non_finite_upper(-42.0), -42.0);
    assert_eq!(repair_non_finite_upper(0.0), 0.0);
}

// --- enforce_bound_ordering with infinite interval (P1]71 audit) ---

#[test]
fn test_enforce_bound_ordering_repairs_infinite_interval() {
    // [-inf, +inf] is valid ordering but non-finite → should be repaired.
    let mut lower = arr1(&[f32::NEG_INFINITY]).into_dyn();
    let mut upper = arr1(&[f32::INFINITY]).into_dyn();
    let count = enforce_bound_ordering(&mut lower, &mut upper);
    assert_eq!(count, 1, "infinite interval should be repaired");
    assert_eq!(lower[[0]], -FALLBACK_BOUND);
    assert_eq!(upper[[0]], FALLBACK_BOUND);
}

// --- Migrated soundness rounding test ---

#[test]
fn test_soundness_rounding() {
    let lower = arr1(&[1.0f32]).into_dyn();
    let upper = arr1(&[2.0f32]).into_dyn();
    let bounds = IntervalBounds::new(lower, upper).expect("valid bounds");
    let rounded = bounds.round_for_soundness();
    // Directed rounding widens by 1 ULP
    assert!(rounded.lower()[[0]] < 1.0);
    assert!(rounded.upper()[[0]] > 2.0);
}

// --- Cumulative ULP widening: round_for_soundness_n (#2707) ---

#[test]
fn test_round_for_soundness_n_depth_zero_is_clone() {
    let bounds =
        IntervalBounds::new(arr1(&[1.0f32]).into_dyn(), arr1(&[2.0f32]).into_dyn()).expect("valid");
    let rounded = bounds.round_for_soundness_n(0);
    assert_eq!(rounded.lower()[[0]], 1.0);
    assert_eq!(rounded.upper()[[0]], 2.0);
}

#[test]
fn test_round_for_soundness_n_depth_one_matches_single() {
    let bounds =
        IntervalBounds::new(arr1(&[1.0f32]).into_dyn(), arr1(&[2.0f32]).into_dyn()).expect("valid");
    let single = bounds.round_for_soundness();
    let n1 = bounds.round_for_soundness_n(1);
    assert_eq!(single.lower()[[0]], n1.lower()[[0]]);
    assert_eq!(single.upper()[[0]], n1.upper()[[0]]);
}

#[test]
fn test_round_for_soundness_n_58_layers_wider_than_1() {
    // AC2 (#2707): bounds after 58 layers must be measurably wider than after 1.
    let bounds =
        IntervalBounds::new(arr1(&[1.0f32]).into_dyn(), arr1(&[2.0f32]).into_dyn()).expect("valid");
    let r1 = bounds.round_for_soundness_n(1);
    let r58 = bounds.round_for_soundness_n(58);

    // Lower moves further down for depth=58
    assert!(
        r58.lower()[[0]] < r1.lower()[[0]],
        "58-layer lower={} must be below 1-layer lower={}",
        r58.lower()[[0]],
        r1.lower()[[0]]
    );
    // Upper moves further up for depth=58
    assert!(
        r58.upper()[[0]] > r1.upper()[[0]],
        "58-layer upper={} must be above 1-layer upper={}",
        r58.upper()[[0]],
        r1.upper()[[0]]
    );

    // Width difference should be approximately 57 ULPs on each side.
    let width_1 = r1.upper()[[0]] - r1.lower()[[0]];
    let width_58 = r58.upper()[[0]] - r58.lower()[[0]];
    assert!(
        width_58 > width_1,
        "58-layer width={width_58} must exceed 1-layer width={width_1}"
    );
}

#[test]
fn test_round_for_soundness_n_monotonic() {
    // Width must be monotonically non-decreasing with depth.
    let bounds =
        IntervalBounds::new(arr1(&[1.0f32]).into_dyn(), arr1(&[2.0f32]).into_dyn()).expect("valid");
    let mut prev_width = 0.0f32;
    for depth in 0..=100 {
        let r = bounds.round_for_soundness_n(depth);
        let width = r.upper()[[0]] - r.lower()[[0]];
        assert!(
            width >= prev_width,
            "depth={depth}: width={width} < prev_width={prev_width}"
        );
        prev_width = width;
    }
}

#[test]
fn test_round_for_soundness_n_preserves_infeasible_sentinel() {
    let lower = arr1(&[1.0f32]).into_dyn();
    let upper = arr1(&[2.0f32]).into_dyn();
    let mut bounds = IntervalBounds::new(lower, upper).expect("valid");
    bounds.mark_infeasible_all();

    let rounded = bounds.round_for_soundness_n(58);
    assert_eq!(
        rounded.lower()[[0]],
        f32::INFINITY,
        "infeasible +inf sentinel must be preserved"
    );
    assert_eq!(
        rounded.upper()[[0]],
        f32::NEG_INFINITY,
        "infeasible -inf sentinel must be preserved"
    );
}

#[test]
fn test_round_for_soundness_n_negative_values() {
    let bounds = IntervalBounds::new(arr1(&[-3.0f32]).into_dyn(), arr1(&[-1.0f32]).into_dyn())
        .expect("valid");
    let r10 = bounds.round_for_soundness_n(10);
    let r1 = bounds.round_for_soundness_n(1);
    assert!(r10.lower()[[0]] < r1.lower()[[0]]);
    assert!(r10.upper()[[0]] > r1.upper()[[0]]);
}

#[test]
fn test_round_for_soundness_n_zero_crossing() {
    let bounds =
        IntervalBounds::new(arr1(&[0.0f32]).into_dyn(), arr1(&[0.0f32]).into_dyn()).expect("valid");
    let r58 = bounds.round_for_soundness_n(58);
    assert!(r58.lower()[[0]] < 0.0, "lower must be negative");
    assert!(r58.upper()[[0]] > 0.0, "upper must be positive");
}

#[test]
fn test_next_down_n_f32_basic() {
    let x = 1.0f32;
    let d1 = next_down_f32(x);
    let d3 = next_down_n_f32(x, 3);
    let d3_manual = next_down_f32(next_down_f32(next_down_f32(x)));
    assert_eq!(
        d3.to_bits(),
        d3_manual.to_bits(),
        "next_down_n_f32(x, 3) must equal 3 consecutive next_down_f32 calls"
    );
    assert!(d3 < d1, "3 ULPs down must be below 1 ULP down");
}

#[test]
fn test_next_up_n_f32_basic() {
    let x = 1.0f32;
    let u1 = next_up_f32(x);
    let u3 = next_up_n_f32(x, 3);
    let u3_manual = next_up_f32(next_up_f32(next_up_f32(x)));
    assert_eq!(
        u3.to_bits(),
        u3_manual.to_bits(),
        "next_up_n_f32(x, 3) must equal 3 consecutive next_up_f32 calls"
    );
    assert!(u3 > u1, "3 ULPs up must be above 1 ULP up");
}

#[test]
fn test_next_down_n_f32_zero_depth() {
    assert_eq!(next_down_n_f32(1.0, 0), 1.0);
    assert_eq!(next_up_n_f32(1.0, 0), 1.0);
}

#[test]
fn test_next_down_n_f32_nan_propagation() {
    assert!(next_down_n_f32(f32::NAN, 10).is_nan());
    assert!(next_up_n_f32(f32::NAN, 10).is_nan());
}

#[test]
fn test_next_down_n_f32_inf_preservation() {
    assert_eq!(next_down_n_f32(f32::INFINITY, 10), f32::INFINITY);
    assert_eq!(next_up_n_f32(f32::INFINITY, 10), f32::INFINITY);
    assert_eq!(next_down_n_f32(f32::NEG_INFINITY, 10), f32::NEG_INFINITY);
    assert_eq!(next_up_n_f32(f32::NEG_INFINITY, 10), f32::NEG_INFINITY);
}

#[test]
fn test_next_down_n_f32_saturation_to_inf() {
    // Finite input that saturates to infinity after N steps.
    // next_down_f32(f32::MIN) = -inf (bits + 1 overflows to inf encoding).
    assert_eq!(
        next_down_n_f32(f32::MIN, 1),
        f32::NEG_INFINITY,
        "next_down from f32::MIN must saturate to -inf"
    );
    // Depth 2: already -inf after step 1, stays -inf.
    assert_eq!(
        next_down_n_f32(f32::MIN, 2),
        f32::NEG_INFINITY,
        "further steps past saturation must stay -inf"
    );
    // Mirror: next_up from f32::MAX saturates to +inf.
    assert_eq!(
        next_up_n_f32(f32::MAX, 1),
        f32::INFINITY,
        "next_up from f32::MAX must saturate to +inf"
    );
    assert_eq!(
        next_up_n_f32(f32::MAX, 2),
        f32::INFINITY,
        "further steps past saturation must stay +inf"
    );
}

#[test]
fn test_round_for_soundness_n_extreme_boundary() {
    // round_for_soundness_n at f32::MAX/MIN — after 1 ULP, both become infinite.
    let bounds = IntervalBounds::new(arr1(&[f32::MIN]).into_dyn(), arr1(&[f32::MAX]).into_dyn())
        .expect("valid full finite range");
    let r1 = bounds.round_for_soundness_n(1);
    assert_eq!(
        r1.lower()[[0]],
        f32::NEG_INFINITY,
        "lower at f32::MIN must become -inf after 1 ULP"
    );
    assert_eq!(
        r1.upper()[[0]],
        f32::INFINITY,
        "upper at f32::MAX must become +inf after 1 ULP"
    );
    assert!(
        r1.lower()[[0]] <= r1.upper()[[0]],
        "ordering invariant: -inf <= +inf"
    );
}
