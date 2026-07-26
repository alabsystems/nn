// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for quantization certificate drift computation.
//!
//! Proves the mathematical properties of `compute_element_drift()` — the core
//! function that measures per-element Hausdorff distance between f32 and
//! quantized model output bounds. This drift feeds the Lipschitz composition
//! machinery to determine whether quantization preserves audio quality.
//!
//! Properties proved:
//! 1. Drift is non-negative for all finite inputs.
//! 2. Drift is zero when f32 and quantized bounds are identical.
//! 3. Max drift dominates mean drift (max >= mean).
//! 4. Drift output is finite for bounded inputs.
//!
//! All harnesses use the actual `compute_element_drift()` function —
//! no transcendental stubs needed (only abs, max, add, div).

use super::compute_element_drift;

/// Prove: element drift is non-negative for all finite inputs.
///
/// The drift formula `max(|ql - fl|, |qh - fh|)` uses absolute values,
/// so the result must be >= 0. The mean (sum of non-negatives / n) must
/// also be non-negative.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn drift_non_negative_single_element() {
    let fl: f32 = kani::any();
    let fh: f32 = kani::any();
    let ql: f32 = kani::any();
    let qh: f32 = kani::any();
    kani::assume(fl.is_finite() && fh.is_finite());
    kani::assume(ql.is_finite() && qh.is_finite());

    let result = compute_element_drift(&[fl], &[fh], &[ql], &[qh]);
    if let Ok((max_drift, mean_drift, n)) = result {
        assert!(max_drift >= 0.0, "max_drift must be non-negative");
        assert!(mean_drift >= 0.0, "mean_drift must be non-negative");
        assert_eq!(n, 1);
    }
}

/// Prove: drift is exactly zero when f32 and quantized bounds are identical.
///
/// If the quantized model has the same output bounds as the f32 model,
/// there is no quantization-induced drift. This is the identity case:
/// `max(|lo - lo|, |hi - hi|) = max(0, 0) = 0`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn drift_zero_when_bounds_identical() {
    let lo: f32 = kani::any();
    let hi: f32 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());

    let result = compute_element_drift(&[lo], &[hi], &[lo], &[hi]);
    if let Ok((max_drift, mean_drift, _)) = result {
        assert!(
            max_drift == 0.0,
            "drift must be 0 when bounds are identical"
        );
        assert!(
            mean_drift == 0.0,
            "mean drift must be 0 when bounds are identical"
        );
    }
}

/// Prove: max_drift >= mean_drift for multi-element bounds.
///
/// The max of a set of non-negative values is always >= their mean.
/// This is a fundamental property: the certificate's `max_output_drift`
/// is the conservative bound used in Lipschitz composition.
///
/// Uses 2-element arrays to exercise the aggregation path.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn max_drift_dominates_mean() {
    let fl: [f32; 2] = [kani::any(), kani::any()];
    let fh: [f32; 2] = [kani::any(), kani::any()];
    let ql: [f32; 2] = [kani::any(), kani::any()];
    let qh: [f32; 2] = [kani::any(), kani::any()];
    for i in 0..2 {
        kani::assume(fl[i].is_finite() && fh[i].is_finite());
        kani::assume(ql[i].is_finite() && qh[i].is_finite());
        // Bound inputs to prevent f64 overflow in sum.
        kani::assume(fl[i].abs() <= 1e4 && fh[i].abs() <= 1e4);
        kani::assume(ql[i].abs() <= 1e4 && qh[i].abs() <= 1e4);
    }

    let result = compute_element_drift(&fl, &fh, &ql, &qh);
    if let Ok((max_drift, mean_drift, n)) = result {
        assert_eq!(n, 2);
        assert!(
            max_drift >= mean_drift,
            "max_drift ({max_drift}) must be >= mean_drift ({mean_drift})"
        );
    }
}

/// Prove: drift output is finite when all inputs are finite and bounded.
///
/// With bounded f32 inputs (|x| <= 1e4), all intermediate f64 computations
/// (subtraction, absolute value, max, sum, division by n=1) must produce
/// finite results. This guards against accidental overflow or NaN
/// propagation in the drift formula.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(2)]
fn drift_finite_for_bounded_inputs() {
    let fl: f32 = kani::any();
    let fh: f32 = kani::any();
    let ql: f32 = kani::any();
    let qh: f32 = kani::any();
    kani::assume(fl.is_finite() && fh.is_finite());
    kani::assume(ql.is_finite() && qh.is_finite());
    kani::assume(fl.abs() <= 1e4 && fh.abs() <= 1e4);
    kani::assume(ql.abs() <= 1e4 && qh.abs() <= 1e4);

    let result = compute_element_drift(&[fl], &[fh], &[ql], &[qh]);
    if let Ok((max_drift, mean_drift, _)) = result {
        assert!(
            max_drift.is_finite(),
            "max_drift must be finite for bounded inputs"
        );
        assert!(
            mean_drift.is_finite(),
            "mean_drift must be finite for bounded inputs"
        );
    }
}
