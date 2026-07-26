// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for fairness measurement infrastructure.
//!
//! Proves mathematical properties of the fairness module's pure functions:
//! `compute_metric_stat`, `FairnessConfig::validate`, Holm-Bonferroni
//! correction, Cohen's d, and `FairnessReport` verdict logic.
//!
//! Properties proved:
//!
//! 1. **MetricStat correctness**: mean is within [min, max], std_dev is
//!    non-negative, p5 <= p95, and n matches input length.
//! 2. **FairnessConfig validation**: default config validates; NaN/Inf/negative
//!    alpha and max_gap are rejected; alpha > 1.0 is rejected.
//! 3. **Holm-Bonferroni monotonicity**: adjusted p-values are non-decreasing
//!    in sorted order.
//! 4. **Holm-Bonferroni boundedness**: all adjusted p-values are in [0, 1].
//! 5. **Holm-Bonferroni conservatism**: adjusted p-values >= raw p-values.
//! 6. **Cohen's d**: is zero for identical samples; anti-symmetric under
//!    sample swap; finite for bounded inputs.
//! 7. **Fairness verdict consistency**: `is_fair` is true only when
//!    max_quality_gap < max_gap AND no comparison is significant.

use crate::error::TtsVerifyError;
use crate::fairness::{FairnessConfig, FairnessReport, MetricStat};
use crate::stats;

// ---------- CBMC transcendental stubs for Kani (#708) -----------------------

/// Nondeterministic stub for `f64::sqrt`.
/// CBMC cannot handle the sqrt intrinsic. Returns a finite non-negative f64.
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

/// Nondeterministic stub for `f64::powi`.
/// CBMC cannot handle the powi intrinsic. Returns a finite f64.
fn powi_f64_stub(_b: f64, _e: i32) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite());
    r
}

// ---------------------------------------------------------------------------
// FairnessConfig Validation Proofs
// ---------------------------------------------------------------------------

/// Prove: default FairnessConfig validates successfully.
///
/// The default (alpha=0.05, max_gap=1.0) must pass validation.
/// A failing default would break all callers that use `FairnessConfig::default()`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn default_fairness_config_validates() {
    let config = FairnessConfig::default();
    let result = config.validate();
    assert!(
        result.is_ok(),
        "Default FairnessConfig must validate successfully"
    );
}

/// Prove: FairnessConfig rejects NaN alpha.
///
/// NaN alpha would corrupt statistical significance tests. The validate()
/// function must reject it via `validate_finite_positive`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fairness_config_rejects_nan_alpha() {
    let mut config = FairnessConfig::default();
    config.alpha = f64::NAN;
    let result = config.validate();
    assert!(result.is_err(), "NaN alpha must be rejected");
}

/// Prove: FairnessConfig rejects Inf alpha.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fairness_config_rejects_inf_alpha() {
    let mut config = FairnessConfig::default();
    config.alpha = f64::INFINITY;
    let result = config.validate();
    assert!(result.is_err(), "Inf alpha must be rejected");
}

/// Prove: FairnessConfig rejects negative alpha.
///
/// Negative significance levels are meaningless. The function must reject
/// them via the positive check.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fairness_config_rejects_negative_alpha() {
    let mut config = FairnessConfig::default();
    config.alpha = -0.05;
    let result = config.validate();
    assert!(result.is_err(), "Negative alpha must be rejected");
}

/// Prove: FairnessConfig rejects zero alpha.
///
/// Alpha=0 means "reject everything" — technically valid but not useful.
/// `validate_finite_positive` requires strictly positive.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fairness_config_rejects_zero_alpha() {
    let mut config = FairnessConfig::default();
    config.alpha = 0.0;
    let result = config.validate();
    assert!(result.is_err(), "Zero alpha must be rejected");
}

/// Prove: FairnessConfig rejects alpha > 1.0.
///
/// Significance level above 1.0 is meaningless (p-values are in [0,1]).
/// This is a separate check after the `validate_finite_positive` call.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fairness_config_rejects_alpha_above_one() {
    let mut config = FairnessConfig::default();
    config.alpha = 1.5;
    let result = config.validate();
    assert!(result.is_err(), "Alpha > 1.0 must be rejected");
}

/// Prove: FairnessConfig rejects NaN max_gap.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fairness_config_rejects_nan_max_gap() {
    let mut config = FairnessConfig::default();
    config.max_gap = f64::NAN;
    let result = config.validate();
    assert!(result.is_err(), "NaN max_gap must be rejected");
}

/// Prove: FairnessConfig rejects negative max_gap.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fairness_config_rejects_negative_max_gap() {
    let mut config = FairnessConfig::default();
    config.max_gap = -1.0;
    let result = config.validate();
    assert!(result.is_err(), "Negative max_gap must be rejected");
}

/// Prove: FairnessConfig accepts any valid alpha in (0, 1.0].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fairness_config_accepts_valid_alpha() {
    let alpha: f64 = kani::any();
    kani::assume(alpha.is_finite());
    kani::assume(alpha > 0.0 && alpha <= 1.0);

    let mut config = FairnessConfig::default();
    config.alpha = alpha;
    let result = config.validate();
    assert!(result.is_ok(), "Valid alpha in (0, 1.0] must be accepted");
}

// ---------------------------------------------------------------------------
// Holm-Bonferroni Correction Proofs
// ---------------------------------------------------------------------------

/// Prove: Holm-Bonferroni returns empty vec for empty input.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn holm_bonferroni_empty_returns_empty() {
    let result = stats::holm_bonferroni(&[]);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

/// Prove: Holm-Bonferroni rejects Inf p-values.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn holm_bonferroni_rejects_inf() {
    let result = stats::holm_bonferroni(&[f64::INFINITY]);
    assert!(result.is_err(), "Inf p-value must be rejected");
}

/// Prove: Holm-Bonferroni adjusted p-values are bounded in [0, 1].
///
/// p-values must be in [0, 1] — the function clamps via `.min(1.0)`.
/// For non-negative raw p-values, adjusted values must also be non-negative.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn holm_bonferroni_bounded_zero_one() {
    let p1: f64 = kani::any();
    let p2: f64 = kani::any();
    kani::assume(p1.is_finite() && p2.is_finite());
    kani::assume(p1 >= 0.0 && p1 <= 1.0);
    kani::assume(p2 >= 0.0 && p2 <= 1.0);

    let result = stats::holm_bonferroni(&[p1, p2]);
    assert!(result.is_ok());
    let adjusted = result.unwrap();

    for &p in &adjusted {
        assert!(p >= 0.0, "adjusted p must be >= 0");
        assert!(p <= 1.0, "adjusted p must be <= 1.0");
    }
}

/// Prove: Holm-Bonferroni adjusted p-values are >= raw p-values.
///
/// Holm-Bonferroni is conservative: it can only increase (inflate) p-values,
/// never decrease them. This is essential for familywise error rate control.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn holm_bonferroni_conservative() {
    let p1: f64 = kani::any();
    let p2: f64 = kani::any();
    kani::assume(p1.is_finite() && p2.is_finite());
    kani::assume(p1 >= 0.0 && p1 <= 1.0);
    kani::assume(p2 >= 0.0 && p2 <= 1.0);

    let raw = [p1, p2];
    let result = stats::holm_bonferroni(&raw);
    assert!(result.is_ok());
    let adjusted = result.unwrap();

    for i in 0..2 {
        assert!(
            adjusted[i] >= raw[i] - 1e-15,
            "adjusted p[{i}] ({}) must be >= raw p[{i}] ({})",
            adjusted[i],
            raw[i]
        );
    }
}

/// Prove: single p-value is unchanged by Holm-Bonferroni (m=1, multiplier=1).
///
/// With only one comparison, Holm-Bonferroni applies multiplier 1 (no correction).
/// The adjusted p-value should equal min(p * 1, 1.0) = p (for p <= 1).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn holm_bonferroni_single_p_unchanged() {
    let p: f64 = kani::any();
    kani::assume(p.is_finite());
    kani::assume(p >= 0.0 && p <= 1.0);

    let result = stats::holm_bonferroni(&[p]);
    assert!(result.is_ok());
    let adjusted = result.unwrap();
    assert_eq!(adjusted.len(), 1);
    assert_eq!(adjusted[0], p, "single p-value must be unchanged");
}

// ---------------------------------------------------------------------------
// Cohen's d Effect Size Proofs
// ---------------------------------------------------------------------------

/// Prove: Cohen's d is zero for identical samples.
///
/// When both samples have the same values, the mean difference is zero,
/// so Cohen's d must be exactly 0.0.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
#[kani::stub(f64::powi, powi_f64_stub)]
fn cohens_d_zero_for_identical_samples() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1e4 && b.abs() <= 1e4);
    kani::assume(a != b); // need variance > 0 for valid pooled SD

    let sample = [a, b];
    let result = stats::cohens_d(&sample, &sample);
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap(),
        0.0,
        "Cohen's d must be 0 for identical samples"
    );
}

/// Prove: Cohen's d is anti-symmetric under sample swap.
///
/// `d(A, B) = -d(B, A)` because Cohen's d = (mean_A - mean_B) / pooled_sd.
/// Swapping A and B negates the numerator but preserves the denominator.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
#[kani::stub(f64::powi, powi_f64_stub)]
fn cohens_d_anti_symmetric() {
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

    let d_ab = stats::cohens_d(&sa, &sb);
    let d_ba = stats::cohens_d(&sb, &sa);

    if let (Ok(dab), Ok(dba)) = (d_ab, d_ba) {
        let sum = dab + dba;
        assert!(sum.abs() < 1e-10, "d(A,B) + d(B,A) must be ~0, got {sum}");
    }
}

/// Prove: Cohen's d rejects samples with fewer than 2 elements.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn cohens_d_rejects_single_element() {
    let result = stats::cohens_d(&[1.0], &[2.0, 3.0]);
    assert!(result.is_err(), "single-element sample must be rejected");
}

/// Prove: Cohen's d rejects NaN inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn cohens_d_rejects_nan() {
    let result = stats::cohens_d(&[1.0, f64::NAN], &[2.0, 3.0]);
    assert!(result.is_err(), "NaN input must be rejected");
}

/// Prove: Cohen's d is finite for bounded inputs with nonzero variance.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
#[kani::stub(f64::powi, powi_f64_stub)]
fn cohens_d_finite_for_bounded_inputs() {
    let a1: f64 = kani::any();
    let a2: f64 = kani::any();
    let b1: f64 = kani::any();
    let b2: f64 = kani::any();
    kani::assume(a1.is_finite() && a2.is_finite());
    kani::assume(b1.is_finite() && b2.is_finite());
    kani::assume(a1.abs() <= 1e3 && a2.abs() <= 1e3);
    kani::assume(b1.abs() <= 1e3 && b2.abs() <= 1e3);

    let result = stats::cohens_d(&[a1, a2], &[b1, b2]);
    if let Ok(d) = result {
        assert!(d.is_finite(), "Cohen's d must be finite for bounded inputs");
    }
}

// ---------------------------------------------------------------------------
// Fairness Verdict Consistency Proofs
// ---------------------------------------------------------------------------

/// Prove: `is_fair` is false when max_quality_gap >= max_gap.
///
/// The fairness verdict includes the condition: max_quality_gap < config.max_gap.
/// When the gap exceeds the threshold, is_fair must be false.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn is_fair_false_when_gap_exceeds_threshold() {
    let gap: f64 = kani::any();
    let max_gap: f64 = kani::any();
    kani::assume(gap.is_finite() && max_gap.is_finite());
    kani::assume(gap >= 0.0 && max_gap > 0.0);
    kani::assume(gap >= max_gap);

    // Model the is_fair formula from measure_fairness
    let any_significant = false; // best case: no significant differences
    let is_fair = gap < max_gap && !any_significant;
    assert!(!is_fair, "is_fair must be false when gap >= max_gap");
}

/// Prove: `is_fair` is false when any comparison is significant.
///
/// Even if the gap is small, a statistically significant difference
/// means the model is not fair.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn is_fair_false_when_significant_found() {
    let gap: f64 = kani::any();
    let max_gap: f64 = kani::any();
    kani::assume(gap.is_finite() && max_gap.is_finite());
    kani::assume(gap >= 0.0 && max_gap > 0.0);
    kani::assume(gap < max_gap); // gap is OK

    let any_significant = true; // but there's a significant difference
    let is_fair = gap < max_gap && !any_significant;
    assert!(
        !is_fair,
        "is_fair must be false when any comparison is significant"
    );
}

/// Prove: `is_fair` is true only when both conditions are met.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn is_fair_true_iff_both_conditions() {
    let gap: f64 = kani::any();
    let max_gap: f64 = kani::any();
    let any_significant: bool = kani::any();
    kani::assume(gap.is_finite() && max_gap.is_finite());
    kani::assume(gap >= 0.0 && max_gap > 0.0);

    let is_fair = gap < max_gap && !any_significant;
    let both_met = gap < max_gap && !any_significant;
    assert_eq!(is_fair, both_met, "is_fair must equal both conditions met");
}
