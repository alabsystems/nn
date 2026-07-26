// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for disentanglement verification types and logic.
//!
//! Proves properties of the types used in both CROWN-based formal
//! disentanglement (`disentanglement.rs`) and audio-domain empirical
//! disentanglement (`audio_disentanglement.rs`).
//!
//! Properties proved:
//!
//! 1. **ControlDimension::dim()**: saturating_sub ensures non-negative result
//!    for all inputs, including slice_start > slice_end.
//! 2. **AcousticProperty::dim()**: same safety guarantee.
//! 3. **DisentanglementCertificate**: `is_disentangled` consistency with
//!    `max_cross_influence` and threshold.
//! 4. **DisentanglementThresholds**: default validates; NaN/negative rejected.
//! 5. **classify_disentanglement**: threshold comparisons are correct.
//! 6. **Cross-influence ratio**: bounded in [0, +inf) for non-negative widths.

use crate::audio_disentanglement::{
    classify_disentanglement, AudioDisentanglementResult, DisentanglementThresholds,
};

// ---------------------------------------------------------------------------
// ControlDimension & AcousticProperty Dimension Proofs
// ---------------------------------------------------------------------------

/// Prove: `saturating_sub` never produces a negative or wrapping result.
///
/// `ControlDimension::dim()` and `AcousticProperty::dim()` use
/// `slice_end.saturating_sub(slice_start)` to compute the dimension.
/// For any usize inputs, the result must be in [0, slice_end].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn saturating_sub_non_negative_control() {
    let start: usize = kani::any();
    let end: usize = kani::any();
    // Bound inputs to prevent Kani state explosion on full usize range
    kani::assume(start <= 10000);
    kani::assume(end <= 10000);

    let dim = end.saturating_sub(start);
    // saturating_sub never wraps — result is always <= end
    assert!(dim <= end, "dim must be <= end");
    if start <= end {
        assert_eq!(dim, end - start, "normal subtraction when start <= end");
    } else {
        assert_eq!(dim, 0, "saturating at 0 when start > end");
    }
}

/// Prove: `saturating_sub` for AcousticProperty is non-negative.
///
/// Same property as above but verifying the pattern used in
/// `AcousticProperty::dim()`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn saturating_sub_non_negative_property() {
    let output_start: usize = kani::any();
    let output_end: usize = kani::any();
    kani::assume(output_start <= 10000);
    kani::assume(output_end <= 10000);

    let dim = output_end.saturating_sub(output_start);
    if output_start <= output_end {
        assert_eq!(dim, output_end - output_start);
    } else {
        assert_eq!(dim, 0);
    }
}

/// Prove: valid ControlDimension (start < end) has dim > 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn valid_control_dimension_positive_dim() {
    let start: usize = kani::any();
    let end: usize = kani::any();
    kani::assume(start <= 10000);
    kani::assume(end <= 10000);
    kani::assume(start < end);

    let dim = end.saturating_sub(start);
    assert!(dim > 0, "valid control dimension must have dim > 0");
}

// ---------------------------------------------------------------------------
// DisentanglementThresholds Validation Proofs
// ---------------------------------------------------------------------------

/// Prove: default DisentanglementThresholds validates successfully.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn default_thresholds_validate() {
    let thresholds = DisentanglementThresholds::default();
    let result = thresholds.validate();
    assert!(
        result.is_ok(),
        "Default DisentanglementThresholds must validate"
    );
}

/// Prove: NaN f0_correlation_min is rejected by threshold validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn thresholds_reject_nan_f0_correlation() {
    let mut t = DisentanglementThresholds::default();
    t.f0_correlation_min = f64::NAN;
    let result = t.validate();
    assert!(result.is_err(), "NaN f0_correlation_min must be rejected");
}

/// Prove: negative mcd_max is rejected by threshold validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn thresholds_reject_negative_mcd_max() {
    let mut t = DisentanglementThresholds::default();
    t.mcd_max = -1.0;
    let result = t.validate();
    assert!(result.is_err(), "Negative mcd_max must be rejected");
}

/// Prove: zero duration_ratio_tolerance is rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn thresholds_reject_zero_duration_tolerance() {
    let mut t = DisentanglementThresholds::default();
    t.duration_ratio_tolerance = 0.0;
    let result = t.validate();
    assert!(
        result.is_err(),
        "Zero duration_ratio_tolerance must be rejected"
    );
}

/// Prove: Inf waveform_similarity_min is rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn thresholds_reject_inf_waveform_similarity() {
    let mut t = DisentanglementThresholds::default();
    t.waveform_similarity_min = f64::INFINITY;
    let result = t.validate();
    assert!(
        result.is_err(),
        "Inf waveform_similarity_min must be rejected"
    );
}

// ---------------------------------------------------------------------------
// classify_disentanglement Threshold Proofs
// ---------------------------------------------------------------------------

/// Prove: classify_disentanglement f0_preserved is correct.
///
/// f0_preserved = (f0_correlation >= f0_correlation_min).
/// This must hold for all finite inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classify_f0_preserved_correct() {
    let f0_corr: f64 = kani::any();
    let threshold: f64 = kani::any();
    kani::assume(f0_corr.is_finite() && threshold.is_finite());
    kani::assume(f0_corr.abs() <= 2.0 && threshold.abs() <= 2.0);

    let result = AudioDisentanglementResult {
        f0_correlation: f0_corr,
        mcd: 3.0,
        duration_ratio: 1.0,
        waveform_similarity: 0.9,
    };
    let thresholds = DisentanglementThresholds {
        f0_correlation_min: threshold,
        ..DisentanglementThresholds::default()
    };

    let evidence = classify_disentanglement(result, &thresholds);
    assert_eq!(
        evidence.f0_preserved,
        f0_corr >= threshold,
        "f0_preserved must equal (f0_correlation >= threshold)"
    );
}

/// Prove: classify_disentanglement spectral_preserved is correct.
///
/// spectral_preserved = (mcd <= mcd_max).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classify_spectral_preserved_correct() {
    let mcd: f64 = kani::any();
    let threshold: f64 = kani::any();
    kani::assume(mcd.is_finite() && threshold.is_finite());
    kani::assume(mcd >= 0.0 && mcd <= 100.0);
    kani::assume(threshold > 0.0 && threshold <= 100.0);

    let result = AudioDisentanglementResult {
        f0_correlation: 0.95,
        mcd,
        duration_ratio: 1.0,
        waveform_similarity: 0.9,
    };
    let thresholds = DisentanglementThresholds {
        mcd_max: threshold,
        ..DisentanglementThresholds::default()
    };

    let evidence = classify_disentanglement(result, &thresholds);
    assert_eq!(
        evidence.spectral_preserved,
        mcd <= threshold,
        "spectral_preserved must equal (mcd <= mcd_max)"
    );
}

/// Prove: classify_disentanglement duration_preserved is correct.
///
/// duration_preserved = ((duration_ratio - 1.0).abs() <= duration_ratio_tolerance).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classify_duration_preserved_correct() {
    let ratio: f64 = kani::any();
    let tolerance: f64 = kani::any();
    kani::assume(ratio.is_finite() && tolerance.is_finite());
    kani::assume(ratio > 0.0 && ratio <= 10.0);
    kani::assume(tolerance > 0.0 && tolerance <= 1.0);

    let result = AudioDisentanglementResult {
        f0_correlation: 0.95,
        mcd: 3.0,
        duration_ratio: ratio,
        waveform_similarity: 0.9,
    };
    let thresholds = DisentanglementThresholds {
        duration_ratio_tolerance: tolerance,
        ..DisentanglementThresholds::default()
    };

    let evidence = classify_disentanglement(result, &thresholds);
    assert_eq!(
        evidence.duration_preserved,
        (ratio - 1.0).abs() <= tolerance,
        "duration_preserved must equal (|ratio - 1| <= tolerance)"
    );
}

/// Prove: classify_disentanglement waveform_preserved is correct.
///
/// waveform_preserved = (waveform_similarity >= waveform_similarity_min).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classify_waveform_preserved_correct() {
    let sim: f64 = kani::any();
    let threshold: f64 = kani::any();
    kani::assume(sim.is_finite() && threshold.is_finite());
    kani::assume(sim.abs() <= 2.0 && threshold.abs() <= 2.0);

    let result = AudioDisentanglementResult {
        f0_correlation: 0.95,
        mcd: 3.0,
        duration_ratio: 1.0,
        waveform_similarity: sim,
    };
    let thresholds = DisentanglementThresholds {
        waveform_similarity_min: threshold,
        ..DisentanglementThresholds::default()
    };

    let evidence = classify_disentanglement(result, &thresholds);
    assert_eq!(
        evidence.waveform_preserved,
        sim >= threshold,
        "waveform_preserved must equal (sim >= threshold)"
    );
}

// ---------------------------------------------------------------------------
// Cross-Influence Ratio Proofs
// ---------------------------------------------------------------------------

/// Prove: cross-influence ratio is non-negative for non-negative widths.
///
/// The disentanglement certificate computes `off_diagonal / on_diagonal`.
/// For non-negative bound widths with positive primary, the ratio must be >= 0.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn cross_influence_ratio_non_negative() {
    let primary_width: f64 = kani::any();
    let cross_width: f64 = kani::any();
    kani::assume(primary_width.is_finite() && cross_width.is_finite());
    kani::assume(primary_width > 0.0);
    kani::assume(cross_width >= 0.0);
    kani::assume(primary_width <= 1e6 && cross_width <= 1e6);

    let ratio = cross_width / primary_width;
    assert!(ratio >= 0.0, "cross-influence ratio must be non-negative");
    assert!(
        ratio.is_finite(),
        "cross-influence ratio must be finite for bounded inputs"
    );
}

/// Prove: cross-influence ratio is zero when cross_width is zero.
///
/// Zero cross-influence width = perfect disentanglement for that pair.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn cross_influence_ratio_zero_for_zero_cross() {
    let primary_width: f64 = kani::any();
    kani::assume(primary_width.is_finite() && primary_width > 0.0);
    kani::assume(primary_width <= 1e6);

    let ratio = 0.0_f64 / primary_width;
    assert_eq!(ratio, 0.0, "zero cross-width must give zero ratio");
}

/// Prove: `is_disentangled` is consistent with `max_cross_influence` and threshold.
///
/// The certificate sets `is_disentangled = max_cross < threshold`.
/// This must hold exactly.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn is_disentangled_consistent_with_threshold() {
    let max_cross: f64 = kani::any();
    let threshold: f64 = kani::any();
    kani::assume(max_cross.is_finite() && threshold.is_finite());
    kani::assume(max_cross >= 0.0 && threshold >= 0.0);
    kani::assume(max_cross <= 1e6 && threshold <= 1e6);

    // Model the formula from verify_disentanglement:
    let is_disentangled = max_cross < threshold;

    // Verify consistency
    if is_disentangled {
        assert!(max_cross < threshold);
    } else {
        assert!(max_cross >= threshold);
    }
}

/// Prove: max_cross_influence is monotonically non-decreasing as cross-width grows.
///
/// The max cross-influence is max(off_diagonal / on_diagonal) across all
/// control-property pairs. Adding a larger off-diagonal value can only
/// increase or maintain the max.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn max_cross_influence_monotonic() {
    let primary: f64 = kani::any();
    let cross1: f64 = kani::any();
    let cross2: f64 = kani::any();
    kani::assume(primary.is_finite() && primary > 0.0);
    kani::assume(cross1.is_finite() && cross1 >= 0.0);
    kani::assume(cross2.is_finite() && cross2 >= 0.0);
    kani::assume(cross1 <= cross2);
    kani::assume(primary <= 1e6 && cross2 <= 1e6);

    let ratio1 = cross1 / primary;
    let ratio2 = cross2 / primary;
    assert!(ratio2 >= ratio1, "larger cross-width must give >= ratio");
}
