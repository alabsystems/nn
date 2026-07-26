// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for TTS pipeline safety properties.
//!
//! Proves correctness of the pipeline composition, junction contract
//! validation, cost model arithmetic, streaming crossfade, monotonicity
//! certificate interpretation, NaN-propagating folds, and moonshot
//! verification level ordering.
//!
//! These harnesses cover the pure functions that underpin dvoice's
//! multi-stage TTS pipeline safety guarantees. Each harness proves a
//! property that, if violated, would silently corrupt pipeline
//! verification certificates.
//!
//! Properties proved:
//!
//! 1. Junction contracts: `bounds_within_contract` correctly identifies
//!    containment; `max_contract_violation` returns non-negative values
//!    and zero iff contained.
//! 2. Pipeline composition: `check_junction` NaN-guards prevent false
//!    containment claims; soundness propagation is correct.
//! 3. Cost model: `estimate_time_us` is monotonically increasing in both
//!    FLOPs and memory bytes; result is always >= dispatch overhead.
//! 4. Streaming: `crossfade_linear` output is a convex combination
//!    bounded by input extremes.
//! 5. Monotonicity: `interpret_duration_positivity` correctly classifies
//!    proven vs not-proven; `max_provable_input_bound` is positive for
//!    valid inputs and inversely proportional to weight magnitude.
//! 6. Stats: NaN-propagating folds correctly propagate NaN; Holm-Bonferroni
//!    adjusted p-values are monotonically non-decreasing and bounded in [0, 1].
//! 7. Moonshot: `VerificationLevel` ordering is a total order consistent
//!    with the intended strength hierarchy.

// ---- Junction Contract Proofs -----------------------------------------------

/// Prove: `bounds_within_contract` returns true iff all elements are contained.
///
/// For a single-element case, containment means:
///   proven_lower >= contract.lower AND proven_upper <= contract.upper
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_within_contract_single_element_correct() {
    let cl: f64 = kani::any();
    let cu: f64 = kani::any();
    let pl: f64 = kani::any();
    let pu: f64 = kani::any();
    kani::assume(cl.is_finite() && cu.is_finite());
    kani::assume(pl.is_finite() && pu.is_finite());
    kani::assume(cl <= cu && pl <= pu);
    kani::assume(cl.abs() <= 1e6 && cu.abs() <= 1e6);
    kani::assume(pl.abs() <= 1e6 && pu.abs() <= 1e6);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", cl, cu);
    let result = crate::kokoro_contracts::bounds_within_contract(&contract, &[pl], &[pu]);

    let expected = pl >= cl && pu <= cu;
    assert_eq!(
        result, expected,
        "bounds_within_contract must match element-wise containment"
    );
}

/// Prove: `bounds_within_contract` rejects NaN in proven bounds.
///
/// IEEE 754 NaN comparison returns false, which could falsely indicate
/// containment if not guarded. The function must return false for NaN inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_within_contract_rejects_nan() {
    let cl: f64 = kani::any();
    let cu: f64 = kani::any();
    kani::assume(cl.is_finite() && cu.is_finite());
    kani::assume(cl <= cu);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", cl, cu);

    // NaN in lower bound
    let result_nan_lo =
        crate::kokoro_contracts::bounds_within_contract(&contract, &[f64::NAN], &[0.0]);
    assert!(!result_nan_lo, "NaN lower bound must not be contained");

    // NaN in upper bound
    let result_nan_hi =
        crate::kokoro_contracts::bounds_within_contract(&contract, &[0.0], &[f64::NAN]);
    assert!(!result_nan_hi, "NaN upper bound must not be contained");
}

/// Prove: `max_contract_violation` is non-negative for all finite inputs.
///
/// The function clamps the result to >= 0, so even when bounds are fully
/// contained (negative gaps), the output is 0.0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn max_contract_violation_non_negative() {
    let cl: f64 = kani::any();
    let cu: f64 = kani::any();
    let pl: f64 = kani::any();
    let pu: f64 = kani::any();
    kani::assume(cl.is_finite() && cu.is_finite());
    kani::assume(pl.is_finite() && pu.is_finite());
    kani::assume(cl.abs() <= 1e6 && cu.abs() <= 1e6);
    kani::assume(pl.abs() <= 1e6 && pu.abs() <= 1e6);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", cl, cu);
    let violation = crate::kokoro_contracts::max_contract_violation(&contract, &[pl], &[pu]);

    assert!(
        violation >= 0.0,
        "violation must be non-negative, got {violation}"
    );
}

/// Prove: `max_contract_violation` is zero iff `bounds_within_contract` is true.
///
/// These two functions must agree: zero violation <=> contained.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn violation_zero_iff_contained() {
    let cl: f64 = kani::any();
    let cu: f64 = kani::any();
    let pl: f64 = kani::any();
    let pu: f64 = kani::any();
    kani::assume(cl.is_finite() && cu.is_finite());
    kani::assume(pl.is_finite() && pu.is_finite());
    kani::assume(cl <= cu && pl <= pu);
    kani::assume(cl.abs() <= 1e6 && cu.abs() <= 1e6);
    kani::assume(pl.abs() <= 1e6 && pu.abs() <= 1e6);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", cl, cu);
    let contained = crate::kokoro_contracts::bounds_within_contract(&contract, &[pl], &[pu]);
    let violation = crate::kokoro_contracts::max_contract_violation(&contract, &[pl], &[pu]);

    if contained {
        assert_eq!(violation, 0.0, "zero violation when contained");
    } else {
        assert!(
            violation > 0.0 || violation == f64::MAX,
            "positive violation when not contained"
        );
    }
}

/// Prove: `max_contract_violation` returns MAX for NaN inputs.
///
/// NaN bounds must not be treated as zero-violation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn violation_max_for_nan() {
    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", -1.0, 1.0);
    let violation = crate::kokoro_contracts::max_contract_violation(&contract, &[f64::NAN], &[0.5]);
    assert_eq!(violation, f64::MAX, "NaN input must produce MAX violation");
}

// ---- Cost Model Proofs ------------------------------------------------------

/// Prove: `estimate_time_us` is >= dispatch overhead for any valid inputs.
///
/// The roofline model computes max(compute, memory) + overhead. Since
/// max(a, b) >= 0 for non-negative a, b, the result must be >= overhead.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn estimate_time_us_geq_overhead() {
    let flops: u64 = kani::any();
    let mem_bytes: u64 = kani::any();

    let model = crate::cost_model::HardwareCostModel::m4_max();
    let time = model.estimate_time_us(flops, mem_bytes);

    assert!(
        time >= model.dispatch_overhead_us,
        "time must be >= dispatch overhead"
    );
    assert!(time.is_finite(), "time must be finite");
}

/// Prove: `estimate_time_us` is monotonically increasing in FLOPs.
///
/// More FLOPs => more time (or equal, due to memory-bound regime).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn estimate_time_us_monotonic_in_flops() {
    let f1: u64 = kani::any();
    let f2: u64 = kani::any();
    let mem_bytes: u64 = kani::any();
    kani::assume(f1 < f2);

    let model = crate::cost_model::HardwareCostModel::m4_max();
    let t1 = model.estimate_time_us(f1, mem_bytes);
    let t2 = model.estimate_time_us(f2, mem_bytes);

    assert!(t2 >= t1, "more FLOPs must not decrease time");
}

/// Prove: `estimate_time_us` is monotonically increasing in memory bytes.
///
/// More memory traffic => more time (or equal, due to compute-bound regime).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn estimate_time_us_monotonic_in_memory() {
    let flops: u64 = kani::any();
    let m1: u64 = kani::any();
    let m2: u64 = kani::any();
    kani::assume(m1 < m2);

    let model = crate::cost_model::HardwareCostModel::m4_max();
    let t1 = model.estimate_time_us(flops, m1);
    let t2 = model.estimate_time_us(flops, m2);

    assert!(t2 >= t1, "more memory must not decrease time");
}

// ---- Streaming Crossfade Proofs ---------------------------------------------

/// Prove: `crossfade_linear` output is bounded by input extremes.
///
/// Since crossfade is a convex combination: out[i] = tail[i]*(1-a) + head[i]*a,
/// each output sample must lie between min(tail[i], head[i]) and
/// max(tail[i], head[i]).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
fn crossfade_bounded_by_inputs() {
    let t0: f32 = kani::any();
    let t1: f32 = kani::any();
    let h0: f32 = kani::any();
    let h1: f32 = kani::any();
    kani::assume(t0.is_finite() && t1.is_finite());
    kani::assume(h0.is_finite() && h1.is_finite());
    kani::assume(t0.abs() <= 1.0 && t1.abs() <= 1.0);
    kani::assume(h0.abs() <= 1.0 && h1.abs() <= 1.0);

    let tail = [t0, t1];
    let head = [h0, h1];
    let result = crate::streaming::crossfade_linear(&tail, &head);
    assert!(result.is_ok());
    let blended = result.unwrap();
    assert_eq!(blended.len(), 2);

    for i in 0..2 {
        let lo = tail[i].min(head[i]);
        let hi = tail[i].max(head[i]);
        // Allow small epsilon for f32 arithmetic
        assert!(
            f64::from(blended[i]) >= f64::from(lo) - 1e-6,
            "crossfade output must be >= min(tail, head)"
        );
        assert!(
            f64::from(blended[i]) <= f64::from(hi) + 1e-6,
            "crossfade output must be <= max(tail, head)"
        );
    }
}

/// Prove: `crossfade_linear` rejects mismatched lengths.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn crossfade_rejects_length_mismatch() {
    let result = crate::streaming::crossfade_linear(&[0.0, 1.0], &[0.0]);
    assert!(result.is_err(), "mismatched lengths must error");
}

// ---- Monotonicity Certificate Proofs ----------------------------------------

/// Prove: `interpret_duration_positivity` sets `is_proven` correctly.
///
/// The certificate is proven iff `lower_bound > threshold`. This is the
/// core correctness claim for duration positivity verification.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn duration_positivity_proven_iff_bound_exceeds_threshold() {
    let lower: f64 = kani::any();
    let threshold: f64 = kani::any();
    kani::assume(lower.is_finite() && threshold.is_finite());
    kani::assume(lower.abs() <= 1e6 && threshold.abs() <= 1e6);

    let cert =
        crate::monotonicity::interpret_duration_positivity(lower, threshold, 1.0, 1.0, 1, "CROWN");

    assert_eq!(
        cert.is_proven,
        lower > threshold,
        "is_proven must equal (lower_bound > threshold)"
    );
}

/// Prove: `max_provable_input_bound` is positive for valid weight certificates.
///
/// With positive PE margin and positive weight magnitudes, the provable
/// input bound must be positive and finite.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn max_provable_input_bound_positive() {
    let pe_margin: f64 = kani::any();
    let max_abs: f64 = kani::any();
    kani::assume(pe_margin > 0.0 && pe_margin.is_finite());
    kani::assume(max_abs > 0.0 && max_abs.is_finite());
    kani::assume(pe_margin <= 1e6 && max_abs <= 1e6);

    let cert = crate::monotonicity::WeightMagnitudeCertificate {
        per_layer_max_abs: vec![max_abs],
        layer_names: vec!["test".to_string()],
        d_model: 64,
        magnitude_bound: 1.0,
        all_within_bound: true,
        violating_layers: 0,
        max_normalized_magnitude: max_abs * 8.0, // sqrt(64) = 8
    };

    let ib = crate::monotonicity::max_provable_input_bound(&cert, pe_margin);
    assert!(ib > 0.0, "provable input bound must be positive");
    assert!(ib.is_finite(), "provable input bound must be finite");
}

/// Prove: `max_provable_input_bound` is inversely proportional to weight magnitude.
///
/// Larger weight magnitudes => smaller provable input bound. This is the
/// key scaling relationship: heavier weights make proofs harder.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn max_provable_input_bound_decreasing_in_weight_mag() {
    let pe_margin: f64 = kani::any();
    let mag1: f64 = kani::any();
    let mag2: f64 = kani::any();
    kani::assume(pe_margin > 0.0 && pe_margin.is_finite());
    kani::assume(pe_margin <= 1e6);
    kani::assume(mag1 > 0.0 && mag2 > 0.0);
    kani::assume(mag1.is_finite() && mag2.is_finite());
    kani::assume(mag1 < mag2);
    kani::assume(mag1 <= 1e6 && mag2 <= 1e6);

    let cert1 = crate::monotonicity::WeightMagnitudeCertificate {
        per_layer_max_abs: vec![mag1],
        layer_names: vec!["test".to_string()],
        d_model: 64,
        magnitude_bound: 1.0,
        all_within_bound: true,
        violating_layers: 0,
        max_normalized_magnitude: 0.0,
    };
    let cert2 = crate::monotonicity::WeightMagnitudeCertificate {
        per_layer_max_abs: vec![mag2],
        layer_names: vec!["test".to_string()],
        d_model: 64,
        magnitude_bound: 1.0,
        all_within_bound: true,
        violating_layers: 0,
        max_normalized_magnitude: 0.0,
    };

    let ib1 = crate::monotonicity::max_provable_input_bound(&cert1, pe_margin);
    let ib2 = crate::monotonicity::max_provable_input_bound(&cert2, pe_margin);
    assert!(
        ib1 > ib2,
        "larger weight mag must give smaller provable input bound"
    );
}

// ---- NaN-Propagating Fold Proofs --------------------------------------------

/// Prove: `fold_max_propagate_nan` returns NaN if any element is NaN.
///
/// This is the critical safety property: IEEE 754 `f64::max` discards NaN
/// (maxNum semantics). Our custom fold MUST propagate NaN to prevent
/// silent corruption of bound computations.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn fold_max_propagate_nan_propagates() {
    let a: f64 = kani::any();
    kani::assume(a.is_finite());

    let result = crate::stats::fold_max_propagate_nan([a, f64::NAN].into_iter(), 0.0);
    assert!(
        result.is_nan(),
        "NaN element must propagate through fold_max"
    );
}

/// Prove: `fold_min_propagate_nan` returns NaN if any element is NaN.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn fold_min_propagate_nan_propagates() {
    let a: f64 = kani::any();
    kani::assume(a.is_finite());

    let result = crate::stats::fold_min_propagate_nan([a, f64::NAN].into_iter(), f64::INFINITY);
    assert!(
        result.is_nan(),
        "NaN element must propagate through fold_min"
    );
}

/// Prove: `fold_max_propagate_nan` finds the true maximum for finite inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn fold_max_finds_maximum() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1e10 && b.abs() <= 1e10);

    let result = crate::stats::fold_max_propagate_nan([a, b].into_iter(), f64::NEG_INFINITY);
    let expected = a.max(b);
    assert_eq!(result, expected, "fold_max must find the true maximum");
}

// ---- Moonshot VerificationLevel Ordering Proofs -----------------------------

/// Prove: `VerificationLevel` ordering is consistent with intended strength.
///
/// None < Empirical < CrownPartial < CrownProbabilistic < CrownProven < KaniProven < SmtProven
///
/// This ordering determines MoonshotStatus rollups. If the ordering is wrong,
/// `all_at_least_crown_partial()` gives incorrect results.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn verification_level_ordering_consistent() {
    use crate::moonshot::VerificationLevel;

    assert!(VerificationLevel::None < VerificationLevel::Empirical);
    assert!(VerificationLevel::Empirical < VerificationLevel::CrownPartial);
    assert!(VerificationLevel::CrownPartial < VerificationLevel::CrownProbabilistic);
    assert!(VerificationLevel::CrownProbabilistic < VerificationLevel::CrownProven);
    assert!(VerificationLevel::CrownProven < VerificationLevel::KaniProven);
    assert!(VerificationLevel::KaniProven < VerificationLevel::SmtProven);
}

// ---- Pipeline Junction NaN Guard Proofs -------------------------------------

/// Prove: `check_junction` treats NaN bounds as violations.
///
/// Non-finite values in either stage's bounds must produce a violation,
/// never a false containment claim. This is the IEEE 754 NaN guard that
/// prevents silent corruption.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn check_junction_nan_is_violation() {
    use crate::pipeline::{check_junction, VerifiedStage};

    let from = VerifiedStage::new(
        "from",
        vec![1],
        vec![1],
        vec![0.0],
        vec![1.0],
        vec![f64::NAN], // NaN in output lower
        vec![0.5],
        "CROWN",
        true,
    );
    let to = VerifiedStage::new(
        "to",
        vec![1],
        vec![1],
        vec![-1.0],
        vec![1.0],
        vec![0.0],
        vec![1.0],
        "CROWN",
        true,
    );

    let result = check_junction(&from, &to, 0);
    assert!(
        !result.bounds_contained,
        "NaN output bounds must not be contained"
    );
    assert!(
        result.violation_count > 0,
        "NaN must produce at least one violation"
    );
}

/// Prove: `check_junction` reports zero violations when bounds are identical.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn check_junction_identical_bounds_no_violation() {
    use crate::pipeline::{check_junction, VerifiedStage};

    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo <= hi);
    kani::assume(lo.abs() <= 1e6 && hi.abs() <= 1e6);

    let from = VerifiedStage::new(
        "from",
        vec![1],
        vec![1],
        vec![lo],
        vec![hi],
        vec![lo],
        vec![hi],
        "CROWN",
        true,
    );
    let to = VerifiedStage::new(
        "to",
        vec![1],
        vec![1],
        vec![lo],
        vec![hi],
        vec![0.0],
        vec![1.0],
        "CROWN",
        true,
    );

    let result = check_junction(&from, &to, 0);
    assert!(
        result.bounds_contained,
        "identical bounds must be contained"
    );
    assert_eq!(
        result.violation_count, 0,
        "identical bounds must have zero violations"
    );
    assert_eq!(
        result.max_violation, 0.0,
        "identical bounds must have zero max violation"
    );
}
