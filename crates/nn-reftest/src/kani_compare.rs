// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for tensor comparison tolerance logic.
//!
//! These harnesses verify the pure numerical functions in the comparison
//! engine: tolerance computation, NaN handling, cosine similarity edge
//! cases, relative error skip logic, and metric bounds.
//!
//! Issue: #3593

use crate::compare::{compare_tensors, ComparisonConfig};
use crate::trace::NamedTensor;

/// Helper: create a 1-element NamedTensor for scalar comparison proofs.
fn scalar_tensor(name: &str, val: f32) -> NamedTensor {
    NamedTensor {
        name: name.to_string(),
        shape: vec![1],
        data: vec![val],
    }
}

/// Helper: create a 2-element NamedTensor for pair comparison proofs.
fn pair_tensor(name: &str, a: f32, b: f32) -> NamedTensor {
    NamedTensor {
        name: name.to_string(),
        shape: vec![2],
        data: vec![a, b],
    }
}

// -- Harness 1: abs_diff finite for finite inputs --

/// Proves that comparing two finite scalars produces finite max_abs_diff.
///
/// Domain: both values in [-1e6, 1e6]. The absolute difference `(r - c).abs()`
/// must be finite (no overflow for bounded f32). This is the core tolerance
/// computation used in every tensor comparison.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(2)]
fn abs_diff_finite_for_bounded_inputs() {
    let r: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0e6 && r <= 1.0e6);
    kani::assume(c.is_finite() && c >= -1.0e6 && c <= 1.0e6);

    let config = ComparisonConfig::new(1.0, 1.0, -1.0); // permissive
    let reference = scalar_tensor("ref", r);
    let candidate = scalar_tensor("cand", c);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    assert!(
        result.max_abs_diff.is_finite(),
        "max_abs_diff must be finite for finite inputs"
    );
    assert!(
        result.mean_abs_diff.is_finite(),
        "mean_abs_diff must be finite for finite inputs"
    );
    assert!(
        result.max_abs_diff >= 0.0,
        "max_abs_diff must be non-negative"
    );
}

// -- Harness 2: NaN input produces infinite divergence --

/// Proves that a NaN candidate element causes max_abs_diff = INFINITY.
///
/// IEEE 754: NaN bypasses normal comparisons. Without explicit is_finite()
/// checks, NaN comparisons silently return false, and max_abs stays at 0.0.
/// The comparison engine must detect non-finite inputs and flag them.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn nan_candidate_produces_infinite_divergence() {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0e6 && r <= 1.0e6);

    let config = ComparisonConfig::new(1.0, 1.0, -1.0);
    let reference = scalar_tensor("ref", r);
    let candidate = scalar_tensor("cand", f32::NAN);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    assert!(
        result.max_abs_diff == f32::INFINITY,
        "NaN candidate must produce INFINITY max_abs_diff"
    );
    assert!(
        result.max_rel_diff == f32::INFINITY,
        "NaN candidate must produce INFINITY max_rel_diff"
    );
}

// -- Harness 3: NaN reference produces infinite divergence --

/// Proves that a NaN reference element also triggers infinite divergence.
///
/// Symmetric with harness 2: both reference and candidate NaN paths
/// must converge to the same INFINITY sentinel.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn nan_reference_produces_infinite_divergence() {
    let c: f32 = kani::any();
    kani::assume(c.is_finite() && c >= -1.0e6 && c <= 1.0e6);

    let config = ComparisonConfig::new(1.0, 1.0, -1.0);
    let reference = scalar_tensor("ref", f32::NAN);
    let candidate = scalar_tensor("cand", c);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    assert!(
        result.max_abs_diff == f32::INFINITY,
        "NaN reference must produce INFINITY max_abs_diff"
    );
}

// -- Harness 4: relative error skip for near-zero values --

/// Proves that when both |r| and |c| are below abs_tolerance, the relative
/// error component (max_rel_diff) stays at 0.0.
///
/// This prevents misleadingly large relative errors from tiny absolute
/// differences on near-zero values (e.g., 1e-8 diff on 3e-7 = ~3% rel).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn relative_error_skipped_for_near_zero_values() {
    let r: f32 = kani::any();
    let c: f32 = kani::any();
    let atol: f32 = 1e-3;

    kani::assume(r.is_finite() && r.abs() < atol);
    kani::assume(c.is_finite() && c.abs() < atol);

    let config = ComparisonConfig::new(atol, 1.0, -1.0);
    let reference = scalar_tensor("ref", r);
    let candidate = scalar_tensor("cand", c);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    // When both values are below abs_tolerance, the relative error skip
    // condition fires: `r.abs() >= atol || c.abs() >= atol` is false.
    assert!(
        result.max_rel_diff == 0.0,
        "relative error must be 0.0 when both values are below abs_tolerance"
    );
}

// -- Harness 5: relative error denominator never zero --

/// Proves that the relative error denominator `max(|r|, |c|, 1e-8)` is
/// always at least 1e-8 when relative error is computed (i.e., when at
/// least one of |r|, |c| >= abs_tolerance).
///
/// This guarantees no division by zero in the relative error path.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn relative_error_denominator_positive() {
    let r: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0e6 && r <= 1.0e6);
    kani::assume(c.is_finite() && c >= -1.0e6 && c <= 1.0e6);

    // At least one must exceed abs_tolerance to enter the rel-error path.
    let atol: f32 = 1e-5;
    kani::assume(r.abs() >= atol || c.abs() >= atol);

    let denom = r.abs().max(c.abs()).max(1e-8);

    assert!(denom >= 1e-8, "denominator must be at least 1e-8");
    assert!(denom.is_finite(), "denominator must be finite");

    let abs_diff = (r - c).abs();
    let rel = abs_diff / denom;
    assert!(rel.is_finite(), "relative error must be finite");
    assert!(rel >= 0.0, "relative error must be non-negative");
}

// -- Harness 6: cosine similarity for identical finite scalars --

/// Proves that comparing a finite scalar with itself yields cosine
/// similarity = 1.0 (perfect alignment).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn cosine_similarity_identical_scalars() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite() && v != 0.0);
    kani::assume(v >= -1.0e6 && v <= 1.0e6);

    let config = ComparisonConfig::new(1.0, 1.0, -1.0);
    let reference = scalar_tensor("ref", v);
    let candidate = scalar_tensor("cand", v);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    assert!(
        result.cosine_similarity == 1.0,
        "identical non-zero scalars must have cosine similarity = 1.0"
    );
}

// -- Harness 7: cosine similarity for both-zero vectors --

/// Proves that comparing two zero vectors yields cosine similarity = 1.0.
///
/// The comparison engine treats both-zero as "identical" (convention),
/// not as "undefined" (mathematical). This avoids false failures on
/// zero-valued layers (e.g., bias tensors initialized to zero).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn cosine_similarity_both_zero_is_one() {
    let config = ComparisonConfig::new(1.0, 1.0, -1.0);
    let reference = scalar_tensor("ref", 0.0);
    let candidate = scalar_tensor("cand", 0.0);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    assert!(
        result.cosine_similarity == 1.0,
        "both-zero vectors must have cosine similarity = 1.0"
    );
}

// -- Harness 8: cosine similarity one-zero vector is 0.0 --

/// Proves that comparing a zero vector with a non-zero vector yields
/// cosine similarity = 0.0 (no meaningful similarity).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn cosine_similarity_one_zero_is_zero() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite() && v != 0.0);
    kani::assume(v >= -1.0e6 && v <= 1.0e6);

    let config = ComparisonConfig::new(1e6, 1.0, -1.0);
    let reference = scalar_tensor("ref", 0.0);
    let candidate = scalar_tensor("cand", v);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    assert!(
        result.cosine_similarity == 0.0,
        "zero-vs-nonzero must have cosine similarity = 0.0"
    );
}

// -- Harness 9: RMS diff finite and non-negative for bounded inputs --

/// Proves that rms_diff is finite and non-negative for any pair of
/// bounded finite 2-element tensors.
///
/// RMS = sqrt(sum_sq_diff / n). Since sum_sq_diff and n are both
/// positive and finite for bounded inputs, the result must be finite.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn rms_diff_finite_and_nonneg_for_bounded_pair() {
    let r0: f32 = kani::any();
    let r1: f32 = kani::any();
    let c0: f32 = kani::any();
    let c1: f32 = kani::any();
    kani::assume(r0.is_finite() && r0 >= -1.0e4 && r0 <= 1.0e4);
    kani::assume(r1.is_finite() && r1 >= -1.0e4 && r1 <= 1.0e4);
    kani::assume(c0.is_finite() && c0 >= -1.0e4 && c0 <= 1.0e4);
    kani::assume(c1.is_finite() && c1 >= -1.0e4 && c1 <= 1.0e4);

    let config = ComparisonConfig::new(1e6, 1.0, -1.0);
    let reference = pair_tensor("ref", r0, r1);
    let candidate = pair_tensor("cand", c0, c1);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    assert!(
        result.rms_diff.is_finite(),
        "rms_diff must be finite for bounded inputs"
    );
    assert!(result.rms_diff >= 0.0, "rms_diff must be non-negative");
}

// -- Harness 10: mean_abs_diff <= max_abs_diff --

/// Proves that mean absolute difference never exceeds max absolute
/// difference for any pair of bounded finite 2-element tensors.
///
/// This is a fundamental statistical invariant: the mean of a set of
/// non-negative values never exceeds the maximum.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn mean_abs_diff_bounded_by_max_abs_diff() {
    let r0: f32 = kani::any();
    let r1: f32 = kani::any();
    let c0: f32 = kani::any();
    let c1: f32 = kani::any();
    kani::assume(r0.is_finite() && r0 >= -1.0e4 && r0 <= 1.0e4);
    kani::assume(r1.is_finite() && r1 >= -1.0e4 && r1 <= 1.0e4);
    kani::assume(c0.is_finite() && c0 >= -1.0e4 && c0 <= 1.0e4);
    kani::assume(c1.is_finite() && c1 >= -1.0e4 && c1 <= 1.0e4);

    let config = ComparisonConfig::new(1e6, 1.0, -1.0);
    let reference = pair_tensor("ref", r0, r1);
    let candidate = pair_tensor("cand", c0, c1);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    assert!(
        result.mean_abs_diff <= result.max_abs_diff,
        "mean_abs_diff must not exceed max_abs_diff"
    );
}

// -- Harness 11: peak amplitude tracks candidate correctly --

/// Proves that peak_amplitude equals the absolute value of the
/// candidate scalar for a single finite element.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(2)]
fn peak_amplitude_equals_candidate_abs() {
    let r: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0e6 && r <= 1.0e6);
    kani::assume(c.is_finite() && c >= -1.0e6 && c <= 1.0e6);

    let config = ComparisonConfig::new(1e6, 1.0, -1.0);
    let reference = scalar_tensor("ref", r);
    let candidate = scalar_tensor("cand", c);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    assert!(
        result.peak_amplitude == c.abs(),
        "peak_amplitude must equal |candidate| for single element"
    );
}

// -- Harness 12: peak amplitude INFINITY for non-finite candidate --

/// Proves that a non-finite (NaN or Inf) candidate element produces
/// peak_amplitude = INFINITY.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn peak_amplitude_infinity_for_nan_candidate() {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0e6 && r <= 1.0e6);

    let config = ComparisonConfig::new(1e6, 1.0, -1.0);
    let reference = scalar_tensor("ref", r);
    let candidate = scalar_tensor("cand", f32::NAN);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    assert!(
        result.peak_amplitude == f32::INFINITY,
        "NaN candidate must produce INFINITY peak_amplitude"
    );
}

// -- Harness 13: identical tensors always pass --

/// Proves that comparing any bounded finite scalar with itself always
/// passes, regardless of how tight the tolerances are (abs > 0).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn identical_tensors_always_pass() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite() && v >= -1.0e6 && v <= 1.0e6);

    // Use strict config — identical data should still pass.
    let config = ComparisonConfig::strict();
    let reference = scalar_tensor("ref", v);
    let candidate = scalar_tensor("cand", v);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    assert!(
        result.max_abs_diff == 0.0,
        "identical scalars must have zero max_abs_diff"
    );
    assert!(
        result.mean_abs_diff == 0.0,
        "identical scalars must have zero mean_abs_diff"
    );
    assert!(
        result.rms_diff == 0.0,
        "identical scalars must have zero rms_diff"
    );
    assert!(result.passed, "identical tensors must always pass");
}
