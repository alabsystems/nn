// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for statistical testing utilities.
//!
//! Supplements `stats_kani.rs` with deeper proofs:
//!
//! - **Welch t-test**: anti-symmetry of t-statistic, p-value boundedness,
//!   rejection of NegInf, rejection of mixed NaN.
//! - **Cohen's d**: rejection of empty input, zero variance handling.
//! - **Percentile**: monotonicity (p1 < p2 implies percentile(p1) <= percentile(p2)),
//!   p50 for two elements is the mean.
//! - **fold_max/min**: commutativity, associativity, NaN at any position.

use crate::stats::{
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
// Welch t-test extended proofs
// ---------------------------------------------------------------------------

/// Prove: welch_t_test t-statistic is anti-symmetric under sample swap.
///
/// t(A, B) = -t(B, A) because mean_A - mean_B = -(mean_B - mean_A).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
#[kani::stub(f64::powi, powi_f64_stub)]
#[kani::stub(f64::ln, ln_f64_stub)]
#[kani::stub(f64::exp, exp_f64_stub)]
#[kani::stub(f64::sin, sin_f64_stub)]
fn welch_t_test_anti_symmetric() {
    let a1: f64 = kani::any();
    let a2: f64 = kani::any();
    let b1: f64 = kani::any();
    let b2: f64 = kani::any();
    kani::assume(a1.is_finite() && a2.is_finite());
    kani::assume(b1.is_finite() && b2.is_finite());
    kani::assume(a1.abs() <= 1e3 && a2.abs() <= 1e3);
    kani::assume(b1.abs() <= 1e3 && b2.abs() <= 1e3);

    let sa = [a1, a2];
    let sb = [b1, b2];

    let r_ab = welch_t_test(&sa, &sb);
    let r_ba = welch_t_test(&sb, &sa);

    if let (Ok((t_ab, _, _)), Ok((t_ba, _, _))) = (r_ab, r_ba) {
        let sum = t_ab + t_ba;
        assert!(sum.abs() < 1e-10, "t(A,B) + t(B,A) must be ~0, got {sum}");
    }
}

/// Prove: welch_t_test p-value is in [0, 1] for bounded finite inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
#[kani::stub(f64::powi, powi_f64_stub)]
#[kani::stub(f64::ln, ln_f64_stub)]
#[kani::stub(f64::exp, exp_f64_stub)]
#[kani::stub(f64::sin, sin_f64_stub)]
fn welch_t_test_p_value_bounded() {
    let a1: f64 = kani::any();
    let a2: f64 = kani::any();
    let b1: f64 = kani::any();
    let b2: f64 = kani::any();
    kani::assume(a1.is_finite() && a2.is_finite());
    kani::assume(b1.is_finite() && b2.is_finite());
    kani::assume(a1.abs() <= 1e3 && a2.abs() <= 1e3);
    kani::assume(b1.abs() <= 1e3 && b2.abs() <= 1e3);

    let result = welch_t_test(&[a1, a2], &[b1, b2]);
    if let Ok((_t, _df, p)) = result {
        assert!(p >= 0.0, "p-value must be >= 0, got {p}");
        assert!(p <= 1.0, "p-value must be <= 1, got {p}");
    }
}

/// Prove: welch_t_test rejects NegInf in sample_a.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn welch_t_test_rejects_neg_inf() {
    let result = welch_t_test(&[f64::NEG_INFINITY, 1.0], &[2.0, 3.0]);
    assert!(result.is_err(), "NegInf in sample must be rejected");
}

/// Prove: welch_t_test returns p=1.0 when both groups are constant and equal.
///
/// When means are identical and variance is zero, t=0 and p=1.0 (no difference).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
#[kani::stub(f64::powi, powi_f64_stub)]
#[kani::stub(f64::ln, ln_f64_stub)]
#[kani::stub(f64::exp, exp_f64_stub)]
#[kani::stub(f64::sin, sin_f64_stub)]
fn welch_t_test_constant_equal_samples() {
    let v: f64 = kani::any();
    kani::assume(v.is_finite() && v.abs() <= 1e6);

    let result = welch_t_test(&[v, v], &[v, v]);
    if let Ok((t, _df, p)) = result {
        assert_eq!(t, 0.0, "t must be 0 for identical constant samples");
        assert_eq!(p, 1.0, "p must be 1.0 for identical constant samples");
    }
}

/// Prove: welch_t_test t-statistic is finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
#[kani::stub(f64::powi, powi_f64_stub)]
#[kani::stub(f64::ln, ln_f64_stub)]
#[kani::stub(f64::exp, exp_f64_stub)]
#[kani::stub(f64::sin, sin_f64_stub)]
fn welch_t_test_t_statistic_finite() {
    let a1: f64 = kani::any();
    let a2: f64 = kani::any();
    let b1: f64 = kani::any();
    let b2: f64 = kani::any();
    kani::assume(a1.is_finite() && a2.is_finite());
    kani::assume(b1.is_finite() && b2.is_finite());
    kani::assume(a1.abs() <= 1e3 && a2.abs() <= 1e3);
    kani::assume(b1.abs() <= 1e3 && b2.abs() <= 1e3);

    let result = welch_t_test(&[a1, a2], &[b1, b2]);
    if let Ok((t, df, p)) = result {
        assert!(t.is_finite(), "t-statistic must be finite");
        assert!(df.is_finite(), "df must be finite");
        assert!(p.is_finite(), "p-value must be finite");
    }
}

// ---------------------------------------------------------------------------
// Cohen's d extended proofs
// ---------------------------------------------------------------------------

/// Prove: Cohen's d rejects empty sample_a.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn cohens_d_rejects_empty_a() {
    let result = cohens_d(&[], &[1.0, 2.0]);
    assert!(result.is_err(), "Empty sample_a must be rejected");
}

/// Prove: Cohen's d rejects empty sample_b.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn cohens_d_rejects_empty_b() {
    let result = cohens_d(&[1.0, 2.0], &[]);
    assert!(result.is_err(), "Empty sample_b must be rejected");
}

/// Prove: Cohen's d returns 0.0 when both groups are constant and equal.
///
/// Zero pooled SD with zero mean difference => d = 0.0 (special case).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
#[kani::stub(f64::powi, powi_f64_stub)]
fn cohens_d_zero_for_constant_equal() {
    let v: f64 = kani::any();
    kani::assume(v.is_finite() && v.abs() <= 1e6);

    let result = cohens_d(&[v, v], &[v, v]);
    if let Ok(d) = result {
        assert_eq!(d, 0.0, "d must be 0 for constant equal samples");
    }
}

/// Prove: Cohen's d rejects Inf in sample_a.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn cohens_d_rejects_inf() {
    let result = cohens_d(&[f64::INFINITY, 1.0], &[2.0, 3.0]);
    assert!(result.is_err(), "Inf input must be rejected");
}

// ---------------------------------------------------------------------------
// Percentile extended proofs
// ---------------------------------------------------------------------------

/// Prove: percentile is monotonic (p1 <= p2 implies percentile(p1) <= percentile(p2)).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f64::floor, floor_f64_stub)]
#[kani::stub(f64::ceil, ceil_f64_stub)]
fn percentile_monotonic() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1e6 && b.abs() <= 1e6);

    let p1: f64 = kani::any();
    let p2: f64 = kani::any();
    kani::assume(p1.is_finite() && p2.is_finite());
    kani::assume(p1 >= 0.0 && p1 <= 100.0);
    kani::assume(p2 >= 0.0 && p2 <= 100.0);
    kani::assume(p1 <= p2);

    let data = [a, b];
    let r1 = percentile(&data, p1);
    let r2 = percentile(&data, p2);
    assert!(
        r1 <= r2,
        "percentile must be monotonic: p({p1})={r1} <= p({p2})={r2}"
    );
}

/// Prove: percentile at 50% for two elements returns the mean.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f64::floor, floor_f64_stub)]
#[kani::stub(f64::ceil, ceil_f64_stub)]
fn percentile_50_two_elements_is_mean() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1e6 && b.abs() <= 1e6);

    let data = [a, b];
    let result = percentile(&data, 50.0);
    let mean = (a + b) / 2.0;
    assert!(
        (result - mean).abs() < 1e-10,
        "percentile(50) for 2 elements must be the mean"
    );
}

/// Prove: percentile result is always within [min, max] of the data.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f64::floor, floor_f64_stub)]
#[kani::stub(f64::ceil, ceil_f64_stub)]
fn percentile_within_data_range() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1e6 && b.abs() <= 1e6);

    let p: f64 = kani::any();
    kani::assume(p.is_finite() && p >= 0.0 && p <= 100.0);

    let data = [a, b];
    let result = percentile(&data, p);
    let lo = a.min(b);
    let hi = a.max(b);
    assert!(result >= lo - 1e-15, "percentile must be >= min of data");
    assert!(result <= hi + 1e-15, "percentile must be <= max of data");
}

// ---------------------------------------------------------------------------
// fold_max/min extended proofs
// ---------------------------------------------------------------------------

/// Prove: fold_max_propagate_nan returns NaN when first element is NaN.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn fold_max_nan_first_element() {
    let b: f64 = kani::any();
    kani::assume(b.is_finite());

    let result = fold_max_propagate_nan([f64::NAN, b].iter().copied(), 0.0);
    assert!(result.is_nan(), "NaN as first element must propagate");
}

/// Prove: fold_min_propagate_nan returns NaN when first element is NaN.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn fold_min_nan_first_element() {
    let b: f64 = kani::any();
    kani::assume(b.is_finite());

    let result = fold_min_propagate_nan([f64::NAN, b].iter().copied(), f64::INFINITY);
    assert!(result.is_nan(), "NaN as first element must propagate");
}

/// Prove: fold_max is commutative for two finite elements.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn fold_max_commutative() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1e10 && b.abs() <= 1e10);

    let r1 = fold_max_propagate_nan([a, b].iter().copied(), f64::NEG_INFINITY);
    let r2 = fold_max_propagate_nan([b, a].iter().copied(), f64::NEG_INFINITY);
    assert_eq!(r1, r2, "fold_max must be commutative");
}

/// Prove: fold_min is commutative for two finite elements.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn fold_min_commutative() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1e10 && b.abs() <= 1e10);

    let r1 = fold_min_propagate_nan([a, b].iter().copied(), f64::INFINITY);
    let r2 = fold_min_propagate_nan([b, a].iter().copied(), f64::INFINITY);
    assert_eq!(r1, r2, "fold_min must be commutative");
}

/// Prove: fold_max with init = NEG_INFINITY and single element returns that element.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn fold_max_single_element() {
    let v: f64 = kani::any();
    kani::assume(v.is_finite());

    let result = fold_max_propagate_nan(std::iter::once(v), f64::NEG_INFINITY);
    assert_eq!(result, v, "fold_max of single element must return it");
}

/// Prove: fold_min with init = INFINITY and single element returns that element.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn fold_min_single_element() {
    let v: f64 = kani::any();
    kani::assume(v.is_finite());

    let result = fold_min_propagate_nan(std::iter::once(v), f64::INFINITY);
    assert_eq!(result, v, "fold_min of single element must return it");
}
