// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Live parity contract tests between `nn_core::IntervalBounds` and
//! `ny_api::BoundedTensor`.
//!
//! These tests exercise both types with identical inputs and assert the
//! results are semantically equivalent. The bridge function
//! `to_bounded_tensor(ib)` is used to convert `IntervalBounds` results
//! for comparison against direct `BoundedTensor` results.
//!
//! Constructor, infeasible/repair, and round_for_soundness parity.
//! IBP arithmetic parity (add, mul, scale, shift) removed in #2005.
//!
//! Part of #1799 — bounds contract packet.

use super::common::bounds_helpers::{
    assert_bridge_parity, bt_extract, bt_singleton, bt_singleton_allow_inf, ib_extract,
    ib_singleton, ib_singleton_allow_inf, TEST_PAIRS,
};
use nn_core::IntervalBounds;
use nn_verify::BoundedTensor;

// ===========================================================================
// D1/D2: Constructor parity
// ===========================================================================

#[test]
fn parity_new_finite_ordered() {
    for &(lo, hi) in TEST_PAIRS {
        let ib = ib_singleton(lo, hi);
        let bt = bt_singleton(lo, hi);
        assert_eq!(
            ib_extract(&ib),
            bt_extract(&bt),
            "new() parity: [{lo}, {hi}]"
        );
        assert_bridge_parity(&ib, &bt, &format!("new [{lo}, {hi}]"));
    }
}

#[test]
fn parity_new_allow_infinite() {
    let cases: &[(f32, f32)] = &[
        (f32::NEG_INFINITY, f32::INFINITY),
        (f32::NEG_INFINITY, 0.0),
        (0.0, f32::INFINITY),
        (-1.0, 1.0),
    ];
    for &(lo, hi) in cases {
        let ib = ib_singleton_allow_inf(lo, hi);
        let bt = bt_singleton_allow_inf(lo, hi);
        let ib_vals = ib_extract(&ib);
        let bt_vals = bt_extract(&bt);
        assert_eq!(
            ib_vals.0.to_bits(),
            bt_vals.0.to_bits(),
            "new_allow_infinite lower parity: [{lo}, {hi}]"
        );
        assert_eq!(
            ib_vals.1.to_bits(),
            bt_vals.1.to_bits(),
            "new_allow_infinite upper parity: [{lo}, {hi}]"
        );
    }
}

#[test]
fn parity_new_rejects_nan() {
    use ndarray::arr1;
    let nan_lower = IntervalBounds::new(arr1(&[f32::NAN]).into_dyn(), arr1(&[1.0]).into_dyn());
    let nan_lower_bt = BoundedTensor::new(arr1(&[f32::NAN]).into_dyn(), arr1(&[1.0]).into_dyn());
    assert!(
        nan_lower.is_err(),
        "IntervalBounds::new should reject NaN lower"
    );
    assert!(
        nan_lower_bt.is_err(),
        "BoundedTensor::new should reject NaN lower"
    );

    let nan_upper = IntervalBounds::new(arr1(&[0.0]).into_dyn(), arr1(&[f32::NAN]).into_dyn());
    let nan_upper_bt = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[f32::NAN]).into_dyn());
    assert!(
        nan_upper.is_err(),
        "IntervalBounds::new should reject NaN upper"
    );
    assert!(
        nan_upper_bt.is_err(),
        "BoundedTensor::new should reject NaN upper"
    );
}

#[test]
fn parity_new_rejects_inverted() {
    use ndarray::arr1;
    let inverted_ib = IntervalBounds::new(arr1(&[5.0]).into_dyn(), arr1(&[1.0]).into_dyn());
    let inverted_bt = BoundedTensor::new(arr1(&[5.0]).into_dyn(), arr1(&[1.0]).into_dyn());
    assert!(
        inverted_ib.is_err(),
        "IntervalBounds::new should reject inverted bounds"
    );
    assert!(
        inverted_bt.is_err(),
        "BoundedTensor::new should reject inverted bounds"
    );
}

#[test]
fn parity_from_epsilon() {
    use ndarray::arr1;
    let test_vals: &[f32] = &[0.0, 1.0, -1.0, 100.0, -100.0, 1.0e-10];
    let test_eps: &[f32] = &[0.0, 1.0e-7, 1.0, 100.0];

    for &val in test_vals {
        for &eps in test_eps {
            let ib = IntervalBounds::from_epsilon(arr1(&[val]).into_dyn(), eps)
                .expect("from_epsilon should succeed");
            let bt = BoundedTensor::from_epsilon(arr1(&[val]).into_dyn(), eps)
                .expect("from_epsilon should succeed");
            let ib_vals = ib_extract(&ib);
            let bt_vals = bt_extract(&bt);
            assert_eq!(
                ib_vals, bt_vals,
                "from_epsilon parity: val={val}, eps={eps}"
            );
            assert_bridge_parity(&ib, &bt, &format!("from_epsilon val={val} eps={eps}"));
        }
    }
}

#[test]
fn parity_concrete() {
    use ndarray::arr1;
    let vals: &[f32] = &[0.0, 1.0, -1.0, 42.5, -100.0, 1.0e-10];
    for &val in vals {
        let ib = IntervalBounds::concrete(arr1(&[val]).into_dyn())
            .expect("concrete should succeed for finite val");
        let bt = BoundedTensor::concrete(arr1(&[val]).into_dyn())
            .expect("concrete should succeed for finite val");
        let ib_vals = ib_extract(&ib);
        let bt_vals = bt_extract(&bt);
        assert_eq!(ib_vals, bt_vals, "concrete parity: val={val}");
        assert_bridge_parity(&ib, &bt, &format!("concrete val={val}"));
    }
}

// ===========================================================================
// D1/D2: mark_infeasible_all + repair parity
// ===========================================================================

#[test]
fn parity_mark_infeasible_all() {
    let mut ib = ib_singleton(1.0, 2.0);
    ib.mark_infeasible_all();
    let ib_vals = ib_extract(&ib);

    let mut bt = bt_singleton(1.0, 2.0);
    bt.mark_infeasible_all();
    let bt_vals = bt_extract(&bt);

    // Both should set (+inf, -inf) sentinel.
    assert_eq!(
        ib_vals.0.to_bits(),
        bt_vals.0.to_bits(),
        "mark_infeasible_all lower parity"
    );
    assert_eq!(
        ib_vals.1.to_bits(),
        bt_vals.1.to_bits(),
        "mark_infeasible_all upper parity"
    );
}

// parity_repair_invalid_inplace deleted: BoundedTensor::repair_invalid_inplace()
// does not exist in NY at rev b37bd828. Re-add when upstream provides the method.

#[test]
fn parity_mark_infeasible_then_round_for_soundness() {
    // This sequence exercises the infeasible sentinel path through
    // round_for_soundness (#171).
    //
    // nn preserves exact (+inf, -inf) sentinels through rounding because
    // next_down_f32/next_up_f32 guard infinity inputs.
    //
    // NY converts (+inf, -inf) to (MAX, -MAX) because its
    // next_down_f32(+inf) → f32::MAX and next_up_f32(-inf) → -f32::MAX.
    // This is by design: NY tracks infeasibility via a separate
    // boolean mask (InvpropState.infeasible_mask), not via sentinel values
    // in the bounds. The key invariant is lower > upper (inverted = infeasible).
    let mut ib = ib_singleton(1.0, 2.0);
    ib.mark_infeasible_all();
    let ib_rounded = ib.round_for_soundness();
    let ib_vals = ib_extract(&ib_rounded);

    let mut bt = bt_singleton(1.0, 2.0);
    bt.mark_infeasible_all();
    let bt_rounded = bt.round_for_soundness();
    let bt_vals = bt_extract(&bt_rounded);

    // nn preserves exact (+inf, -inf) infeasible sentinels.
    assert_eq!(
        ib_vals,
        (f32::INFINITY, f32::NEG_INFINITY),
        "nn: infeasible sentinel preserved through round_for_soundness (#171)"
    );
    // NY also preserves inf sentinels after bump (previously clamped to MAX).
    assert_eq!(
        bt_vals,
        (f32::INFINITY, f32::NEG_INFINITY),
        "NY: round_for_soundness preserves inf infeasible sentinel"
    );
    // Key invariant: both results have lower > upper (still infeasible).
    assert!(ib_vals.0 > ib_vals.1, "nn: bounds still inverted");
    assert!(bt_vals.0 > bt_vals.1, "NY: bounds still inverted");
}

#[test]
fn parity_round_for_soundness_finite() {
    // round_for_soundness on finite bounds: both should widen by 1 ULP.
    for &(lo, hi) in TEST_PAIRS {
        let ib_rounded = ib_singleton(lo, hi).round_for_soundness();
        let bt_rounded = bt_singleton(lo, hi).round_for_soundness();
        let ib_vals = ib_extract(&ib_rounded);
        let bt_vals = bt_extract(&bt_rounded);

        // Both should produce the same ULP-widened result for finite inputs.
        assert_eq!(ib_vals, bt_vals, "round_for_soundness parity: [{lo}, {hi}]");
    }
}
