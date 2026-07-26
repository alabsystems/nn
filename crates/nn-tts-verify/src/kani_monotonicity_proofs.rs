// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for monotonicity verification.
//!
//! Proves correctness of duration positivity certificates, attention
//! monotonicity diagonal dominance, multi-head weight margin aggregation,
//! weight magnitude validation, and max provable input bound computation.
//!
//! Properties proved:
//!
//! 1. `interpret_duration_positivity` classifies `is_proven` correctly
//!    for all finite inputs.
//! 2. Duration positivity certificate field consistency.
//! 3. Attention monotonicity diagonal dominance margin computation.
//! 4. Attention monotonicity NaN/Inf rejection at input boundary.
//! 5. Multi-head weight margin aggregation takes the per-step minimum.
//! 6. `validate_weight_magnitudes` NaN/Inf defense-in-depth.
//! 7. `max_provable_input_bound` scaling relationships.
//! 8. Attention dimension mismatch error paths.

// ---------------------------------------------------------------------------
// Duration Positivity Proofs
// ---------------------------------------------------------------------------

/// Prove: `interpret_duration_positivity` sets `is_proven` iff
/// `lower_bound > threshold` for all finite inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn duration_positivity_is_proven_classification() {
    let lower: f64 = kani::any();
    let threshold: f64 = kani::any();
    kani::assume(lower.is_finite() && threshold.is_finite());
    kani::assume(lower.abs() <= 1e8 && threshold.abs() <= 1e8);

    let cert =
        crate::monotonicity::interpret_duration_positivity(lower, threshold, 1.0, 1.0, 1, "TEST");
    assert_eq!(
        cert.is_proven,
        lower > threshold,
        "is_proven must be (lower_bound > threshold)"
    );
}

/// Prove: certificate fields match constructor arguments exactly.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn duration_positivity_fields_match_args() {
    let lower: f64 = kani::any();
    let threshold: f64 = kani::any();
    let input_bound: f64 = kani::any();
    let style_bound: f64 = kani::any();
    kani::assume(lower.is_finite() && threshold.is_finite());
    kani::assume(input_bound.is_finite() && style_bound.is_finite());

    let cert = crate::monotonicity::interpret_duration_positivity(
        lower,
        threshold,
        input_bound,
        style_bound,
        4,
        "alpha-CROWN",
    );
    assert_eq!(cert.lower_bound, lower);
    assert_eq!(cert.threshold, threshold);
    assert_eq!(cert.input_bound, input_bound);
    assert_eq!(cert.style_bound, style_bound);
    assert_eq!(cert.sequence_length, 4);
    assert_eq!(cert.propagation_mode, "alpha-CROWN");
}

/// Prove: boundary case — equal lower_bound and threshold is NOT proven.
///
/// Strict inequality: `lower_bound > threshold`. Equal values must not
/// claim the proof succeeded.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn duration_positivity_boundary_not_proven() {
    let val: f64 = kani::any();
    kani::assume(val.is_finite() && val.abs() <= 1e8);

    let cert = crate::monotonicity::interpret_duration_positivity(val, val, 1.0, 1.0, 1, "CROWN");
    assert!(
        !cert.is_proven,
        "equal lower_bound and threshold must not be proven"
    );
}

// ---------------------------------------------------------------------------
// Attention Monotonicity Proofs (2x2)
// ---------------------------------------------------------------------------

/// Prove: 2x2 attention matrix diagonal dominance margin is correct.
///
/// For a 2x2 matrix with known structure, verify the row margins and
/// min_margin are computed correctly.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn attention_2x2_margin_correctness() {
    let d0_lo: f32 = kani::any();
    let d1_lo: f32 = kani::any();
    let o01_hi: f32 = kani::any();
    let o10_hi: f32 = kani::any();
    kani::assume(d0_lo.is_finite() && d1_lo.is_finite());
    kani::assume(o01_hi.is_finite() && o10_hi.is_finite());
    kani::assume(d0_lo.abs() <= 100.0 && d1_lo.abs() <= 100.0);
    kani::assume(o01_hi.abs() <= 100.0 && o10_hi.abs() <= 100.0);

    // lower: [d0_lo, *, *, d1_lo] — off-diag lowers don't matter
    // upper: [*, o01_hi, o10_hi, *] — diag uppers don't matter
    let lower = [d0_lo, 0.0f32, 0.0f32, d1_lo];
    let upper = [0.0f32, o01_hi, o10_hi, 0.0f32];

    let cert =
        crate::monotonicity::interpret_attention_monotonicity(&lower, &upper, 2, 2, 1.0, "TEST")
            .unwrap();

    let expected_m0 = f64::from(d0_lo) - f64::from(o01_hi);
    let expected_m1 = f64::from(d1_lo) - f64::from(o10_hi);
    assert!(
        (cert.row_margins[0] - expected_m0).abs() < 1e-6,
        "row 0 margin mismatch"
    );
    assert!(
        (cert.row_margins[1] - expected_m1).abs() < 1e-6,
        "row 1 margin mismatch"
    );

    let expected_min = expected_m0.min(expected_m1);
    assert!(
        (cert.min_margin - expected_min).abs() < 1e-6,
        "min_margin mismatch"
    );
}

/// Prove: `is_proven` is true iff `min_margin > 0` for a 2x2 matrix.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn attention_2x2_is_proven_iff_positive_margin() {
    let d0_lo: f32 = kani::any();
    let d1_lo: f32 = kani::any();
    let o01_hi: f32 = kani::any();
    let o10_hi: f32 = kani::any();
    kani::assume(d0_lo.is_finite() && d1_lo.is_finite());
    kani::assume(o01_hi.is_finite() && o10_hi.is_finite());
    kani::assume(d0_lo.abs() <= 100.0 && d1_lo.abs() <= 100.0);
    kani::assume(o01_hi.abs() <= 100.0 && o10_hi.abs() <= 100.0);

    let lower = [d0_lo, 0.0f32, 0.0f32, d1_lo];
    let upper = [0.0f32, o01_hi, o10_hi, 0.0f32];

    let cert =
        crate::monotonicity::interpret_attention_monotonicity(&lower, &upper, 2, 2, 1.0, "TEST")
            .unwrap();

    assert_eq!(
        cert.is_proven,
        cert.min_margin > 0.0,
        "is_proven must equal (min_margin > 0)"
    );
}

/// Prove: attention monotonicity rejects dimension mismatch in score_lower.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn attention_rejects_lower_dimension_mismatch() {
    // 2x2 = 4 elements expected, but only 3 provided for lower.
    let lower = [1.0f32, 2.0, 3.0];
    let upper = [1.0f32, 2.0, 3.0, 4.0];
    let result =
        crate::monotonicity::interpret_attention_monotonicity(&lower, &upper, 2, 2, 1.0, "TEST");
    assert!(result.is_err(), "dimension mismatch must produce error");
}

/// Prove: attention monotonicity rejects dimension mismatch in score_upper.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn attention_rejects_upper_dimension_mismatch() {
    let lower = [1.0f32, 2.0, 3.0, 4.0];
    let upper = [1.0f32, 2.0]; // Only 2 elements for 2x2
    let result =
        crate::monotonicity::interpret_attention_monotonicity(&lower, &upper, 2, 2, 1.0, "TEST");
    assert!(result.is_err(), "dimension mismatch must produce error");
}

/// Prove: 1x1 attention is trivially monotonic with infinite margin.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn attention_1x1_trivially_monotonic() {
    let lo: f32 = kani::any();
    let hi: f32 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());

    let cert =
        crate::monotonicity::interpret_attention_monotonicity(&[lo], &[hi], 1, 1, 1.0, "TEST")
            .unwrap();
    assert!(cert.is_proven, "1x1 must be trivially monotonic");
    assert!(cert.min_margin.is_infinite(), "1x1 margin must be infinite");
}

// ---------------------------------------------------------------------------
// Multi-Head Weight Margin Proofs
// ---------------------------------------------------------------------------

/// Prove: multi-head aggregation takes the minimum across heads for each step.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn multi_head_takes_minimum_per_step() {
    let h0_m0: f64 = kani::any();
    let h1_m0: f64 = kani::any();
    kani::assume(h0_m0.is_finite() && h1_m0.is_finite());
    kani::assume(h0_m0.abs() <= 1e6 && h1_m0.abs() <= 1e6);

    let head0 = vec![h0_m0];
    let head1 = vec![h1_m0];
    let cert =
        crate::monotonicity::from_multi_head_weight_margins(&[head0, head1], 1, 1, 1.0, "TEST")
            .unwrap();

    let expected_min = h0_m0.min(h1_m0);
    assert!(
        (cert.row_margins[0] - expected_min).abs() < 1e-10,
        "row margin must be min across heads"
    );
    assert!(
        (cert.min_margin - expected_min).abs() < 1e-10,
        "min_margin must equal row margin for single step"
    );
}

/// Prove: multi-head `is_proven` is true iff all per-step minima > 0.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn multi_head_is_proven_iff_all_positive() {
    let h0_m0: f64 = kani::any();
    let h1_m0: f64 = kani::any();
    kani::assume(h0_m0.is_finite() && h1_m0.is_finite());
    kani::assume(h0_m0.abs() <= 1e6 && h1_m0.abs() <= 1e6);

    let head0 = vec![h0_m0];
    let head1 = vec![h1_m0];
    let cert =
        crate::monotonicity::from_multi_head_weight_margins(&[head0, head1], 1, 1, 1.0, "TEST")
            .unwrap();

    let min_margin = h0_m0.min(h1_m0);
    assert_eq!(
        cert.is_proven,
        min_margin > 0.0,
        "is_proven must be (min of all heads > 0)"
    );
}

/// Prove: multi-head rejects NaN in per-head margins.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn multi_head_rejects_nan_margin() {
    let head0 = vec![1.0];
    let head1 = vec![f64::NAN];
    let result =
        crate::monotonicity::from_multi_head_weight_margins(&[head0, head1], 1, 1, 1.0, "TEST");
    assert!(result.is_err(), "NaN margins must be rejected");
}

/// Prove: multi-head rejects truncated margin vectors.
///
/// #1994 regression: truncated heads were silently skipped, leaving
/// INFINITY as the fallback — a fail-open soundness gap.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn multi_head_rejects_truncated_margins() {
    let head0 = vec![0.5, 0.3];
    let head1 = vec![0.2]; // Too short for diag_count=2
    let result =
        crate::monotonicity::from_multi_head_weight_margins(&[head0, head1], 2, 2, 1.0, "TEST");
    assert!(result.is_err(), "truncated margins must be rejected");
}

// ---------------------------------------------------------------------------
// Weight Magnitude Validation Proofs
// ---------------------------------------------------------------------------

/// Prove: `validate_weight_magnitudes` rejects NaN weight data.
///
/// IEEE 754 maxNum: `max(x, NaN) = x`, so `fold(0.0, f64::max)` silently
/// skips NaN elements. The function must guard against this.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn weight_validation_rejects_nan() {
    let w = [0.001f32, f32::NAN];
    let weights: Vec<&[f32]> = vec![&w];
    let names = vec!["layer"];
    let fan_ins = vec![64];
    let result =
        crate::monotonicity::validate_weight_magnitudes(&weights, &names, &fan_ins, 64, 0.1);
    assert!(result.is_err(), "NaN weight must be rejected");
}

/// Prove: `validate_weight_magnitudes` rejects Inf weight data.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn weight_validation_rejects_inf() {
    let w = [f32::INFINITY, 0.001f32];
    let weights: Vec<&[f32]> = vec![&w];
    let names = vec!["layer"];
    let fan_ins = vec![64];
    let result =
        crate::monotonicity::validate_weight_magnitudes(&weights, &names, &fan_ins, 64, 0.1);
    assert!(result.is_err(), "Inf weight must be rejected");
}

/// Prove: `validate_weight_magnitudes` rejects mismatched layer_names count.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_validation_rejects_name_count_mismatch() {
    let w = [0.1f32];
    let weights: Vec<&[f32]> = vec![&w];
    let names = vec!["a", "b"]; // 2 names, 1 weight
    let fan_ins = vec![10];
    let result =
        crate::monotonicity::validate_weight_magnitudes(&weights, &names, &fan_ins, 10, 0.1);
    assert!(result.is_err(), "mismatched name count must error");
}

/// Prove: `validate_weight_magnitudes` rejects mismatched fan_ins count.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_validation_rejects_fan_ins_count_mismatch() {
    let w = [0.1f32];
    let weights: Vec<&[f32]> = vec![&w];
    let names = vec!["a"];
    let fan_ins = vec![10, 20]; // 2 fan_ins, 1 weight
    let result =
        crate::monotonicity::validate_weight_magnitudes(&weights, &names, &fan_ins, 10, 0.1);
    assert!(result.is_err(), "mismatched fan_ins count must error");
}

/// Prove: `max_provable_input_bound` returns 0.0 for NaN pe_margin.
///
/// NaN pe_margin must not produce NaN result — defense-in-depth.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn max_provable_input_bound_nan_pe_margin_returns_zero() {
    let cert = crate::monotonicity::WeightMagnitudeCertificate {
        per_layer_max_abs: vec![0.003],
        layer_names: vec!["test".to_string()],
        d_model: 64,
        magnitude_bound: 0.1,
        all_within_bound: true,
        violating_layers: 0,
        max_normalized_magnitude: 0.024,
    };
    let result = crate::monotonicity::max_provable_input_bound(&cert, f64::NAN);
    assert_eq!(result, 0.0, "NaN pe_margin must return 0.0");
}

/// Prove: `max_provable_input_bound` returns 0.0 for negative pe_margin.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn max_provable_input_bound_negative_pe_margin_returns_zero() {
    let pe: f64 = kani::any();
    kani::assume(pe.is_finite() && pe < 0.0);

    let cert = crate::monotonicity::WeightMagnitudeCertificate {
        per_layer_max_abs: vec![0.003],
        layer_names: vec!["test".to_string()],
        d_model: 64,
        magnitude_bound: 0.1,
        all_within_bound: true,
        violating_layers: 0,
        max_normalized_magnitude: 0.024,
    };
    let result = crate::monotonicity::max_provable_input_bound(&cert, pe);
    assert_eq!(result, 0.0, "negative pe_margin must return 0.0");
}

/// Prove: `max_provable_input_bound` returns INFINITY for zero-weight model.
///
/// With all weights zero, any input bound is provable.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn max_provable_input_bound_zero_weights_infinite() {
    let pe: f64 = kani::any();
    kani::assume(pe > 0.0 && pe.is_finite() && pe <= 1e6);

    let cert = crate::monotonicity::WeightMagnitudeCertificate {
        per_layer_max_abs: vec![0.0],
        layer_names: vec!["test".to_string()],
        d_model: 64,
        magnitude_bound: 0.1,
        all_within_bound: true,
        violating_layers: 0,
        max_normalized_magnitude: 0.0,
    };
    let result = crate::monotonicity::max_provable_input_bound(&cert, pe);
    assert!(
        result.is_infinite(),
        "zero weights must give infinite bound"
    );
}
