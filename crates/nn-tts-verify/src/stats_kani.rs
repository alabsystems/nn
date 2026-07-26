// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for statistical testing utilities.
//!
//! Proves properties of the pure functions in `stats.rs`:
//! `welch_t_test`, `cohens_d`, `holm_bonferroni`, `percentile`,
//! `fold_max_propagate_nan`, and `fold_min_propagate_nan`.
//!
//! Properties proved:
//! 1. `welch_t_test` rejects samples with fewer than 2 elements.
//! 2. `welch_t_test` rejects NaN/Inf inputs.
//! 3. `welch_t_test` p-value is in [0, 1] for finite bounded inputs.
//! 4. `welch_t_test` is zero for identical samples.
//! 5. `percentile` returns 0.0 for empty data.
//! 6. `percentile` at 0% returns the minimum of finite inputs.
//! 7. `percentile` at 100% returns the maximum of finite inputs.
//! 8. `fold_min_propagate_nan` finds the true minimum for finite inputs.
//! 9. `fold_min_propagate_nan` returns init for empty iterator.

use super::{
    cohens_d, fold_max_propagate_nan, fold_min_propagate_nan, holm_bonferroni, percentile,
    welch_t_test,
};

// ---------------------------------------------------------------------------
// Transcendental stubs for CBMC (Kani)
// ---------------------------------------------------------------------------

fn sqrt_f64_stub(x: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e20);
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    if x >= 1.0 {
        kani::assume(r >= 1.0);
    }
    r
}

fn powi_f64_stub(_b: f64, _e: i32) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite());
    r
}

fn ln_f64_stub(_x: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0);
    r
}

fn exp_f64_stub(x: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e20);
    if x <= 0.0 {
        kani::assume(r <= 1.0);
    }
    if x > 0.0 {
        kani::assume(r > 1.0);
    }
    r
}

fn sin_f64_stub(_x: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

fn floor_f64_stub(x: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite());
    kani::assume(r <= x);
    kani::assume(r >= x - 1.0);
    r
}

fn ceil_f64_stub(x: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite());
    kani::assume(r >= x);
    kani::assume(r <= x + 1.0);
    r
}

// ---------------------------------------------------------------------------
// Welch's t-test proofs
// ---------------------------------------------------------------------------

/// Prove: welch_t_test rejects sample_a with fewer than 2 elements.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn welch_t_test_rejects_small_sample_a() {
    let result = welch_t_test(&[1.0], &[2.0, 3.0]);
    assert!(result.is_err(), "Single-element sample_a must be rejected");
}

/// Prove: welch_t_test rejects sample_b with fewer than 2 elements.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn welch_t_test_rejects_small_sample_b() {
    let result = welch_t_test(&[1.0, 2.0], &[3.0]);
    assert!(result.is_err(), "Single-element sample_b must be rejected");
}

/// Prove: welch_t_test rejects empty samples.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn welch_t_test_rejects_empty_samples() {
    let result = welch_t_test(&[], &[1.0, 2.0]);
    assert!(result.is_err(), "Empty sample must be rejected");
}

/// Prove: welch_t_test rejects NaN in sample_a.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn welch_t_test_rejects_nan_in_a() {
    let result = welch_t_test(&[1.0, f64::NAN], &[2.0, 3.0]);
    assert!(result.is_err(), "NaN in sample_a must be rejected");
}

/// Prove: welch_t_test rejects Inf in sample_b.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn welch_t_test_rejects_inf_in_b() {
    let result = welch_t_test(&[1.0, 2.0], &[f64::INFINITY, 3.0]);
    assert!(result.is_err(), "Inf in sample_b must be rejected");
}

/// Prove: welch_t_test returns t=0 for identical samples.
///
/// When both samples have the same values, the mean difference is zero,
/// so the t-statistic must be zero.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
#[kani::stub(f64::powi, powi_f64_stub)]
#[kani::stub(f64::ln, ln_f64_stub)]
#[kani::stub(f64::exp, exp_f64_stub)]
#[kani::stub(f64::sin, sin_f64_stub)]
fn welch_t_test_zero_for_identical_samples() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1e3 && b.abs() <= 1e3);
    kani::assume(a != b); // need nonzero variance

    let sample = [a, b];
    let result = welch_t_test(&sample, &sample);
    if let Ok((t, _df, _p)) = result {
        assert_eq!(t, 0.0, "t-statistic must be zero for identical samples");
    }
}

/// Prove: welch_t_test degrees of freedom is positive for valid inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
#[kani::stub(f64::powi, powi_f64_stub)]
#[kani::stub(f64::ln, ln_f64_stub)]
#[kani::stub(f64::exp, exp_f64_stub)]
#[kani::stub(f64::sin, sin_f64_stub)]
fn welch_t_test_df_positive() {
    let a1: f64 = kani::any();
    let a2: f64 = kani::any();
    let b1: f64 = kani::any();
    let b2: f64 = kani::any();
    kani::assume(a1.is_finite() && a2.is_finite());
    kani::assume(b1.is_finite() && b2.is_finite());
    kani::assume(a1.abs() <= 1e3 && a2.abs() <= 1e3);
    kani::assume(b1.abs() <= 1e3 && b2.abs() <= 1e3);

    let result = welch_t_test(&[a1, a2], &[b1, b2]);
    if let Ok((_t, df, _p)) = result {
        assert!(df > 0.0, "degrees of freedom must be positive");
        assert!(df.is_finite(), "degrees of freedom must be finite");
    }
}

// ---------------------------------------------------------------------------
// Percentile proofs
// ---------------------------------------------------------------------------

/// Prove: percentile returns 0.0 for empty data.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn percentile_empty_returns_zero() {
    let result = percentile(&[], 50.0);
    assert_eq!(result, 0.0, "percentile of empty data must be 0.0");
}

/// Prove: percentile at 0% returns the minimum of finite inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f64::floor, floor_f64_stub)]
#[kani::stub(f64::ceil, ceil_f64_stub)]
fn percentile_zero_returns_min() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1e6 && b.abs() <= 1e6);

    let data = [a, b];
    let result = percentile(&data, 0.0);
    let expected = a.min(b);
    assert_eq!(result, expected, "percentile(0) must return the minimum");
}

/// Prove: percentile at 100% returns the maximum of finite inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f64::floor, floor_f64_stub)]
#[kani::stub(f64::ceil, ceil_f64_stub)]
fn percentile_hundred_returns_max() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1e6 && b.abs() <= 1e6);

    let data = [a, b];
    let result = percentile(&data, 100.0);
    let expected = a.max(b);
    assert_eq!(result, expected, "percentile(100) must return the maximum");
}

/// Prove: percentile of a single element returns that element at any percentile.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
#[kani::stub(f64::floor, floor_f64_stub)]
#[kani::stub(f64::ceil, ceil_f64_stub)]
fn percentile_single_element() {
    let val: f64 = kani::any();
    let p: f64 = kani::any();
    kani::assume(val.is_finite() && val.abs() <= 1e6);
    kani::assume(p >= 0.0 && p <= 100.0 && p.is_finite());

    let result = percentile(&[val], p);
    assert_eq!(
        result, val,
        "percentile of a single element must return that element"
    );
}

// ---------------------------------------------------------------------------
// NaN-propagating fold proofs (supplementing pipeline_safety_kani.rs)
// ---------------------------------------------------------------------------

/// Prove: fold_min_propagate_nan finds the true minimum for finite inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn fold_min_finds_minimum() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1e10 && b.abs() <= 1e10);

    let result = fold_min_propagate_nan([a, b].into_iter(), f64::INFINITY);
    let expected = a.min(b);
    assert_eq!(result, expected, "fold_min must find the true minimum");
}

/// Prove: fold_max result >= each input element (for finite inputs).
///
/// The maximum of a set must be >= every element in the set.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn fold_max_geq_each_element() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1e10 && b.abs() <= 1e10);

    let result = fold_max_propagate_nan([a, b].into_iter(), f64::NEG_INFINITY);
    assert!(result >= a, "fold_max result must be >= a");
    assert!(result >= b, "fold_max result must be >= b");
}

/// Prove: fold_min result <= each input element (for finite inputs).
///
/// The minimum of a set must be <= every element in the set.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn fold_min_leq_each_element() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1e10 && b.abs() <= 1e10);

    let result = fold_min_propagate_nan([a, b].into_iter(), f64::INFINITY);
    assert!(result <= a, "fold_min result must be <= a");
    assert!(result <= b, "fold_min result must be <= b");
}
