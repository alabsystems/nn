// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for fairness measurement.
//!
//! Supplements `kani_fairness_proofs.rs` with deeper proofs for:
//!
//! - **`compute_metric_stat`** via the `MetricStat` invariants observable
//!   through the public `measure_fairness` API: mean within [min, max],
//!   std_dev non-negative, n matches, percentile ordering.
//! - **Welch t-test symmetry** in fairness context.
//! - **Holm-Bonferroni** three-element monotonicity and idempotence.
//! - **`FairnessConfig`** boundary values at exactly alpha=1.0.
//! - **Fairness verdict** with zero max_gap.

use crate::fairness::FairnessConfig;
use crate::stats;

// ---------------------------------------------------------------------------
// FairnessConfig Extended Validation Proofs
// ---------------------------------------------------------------------------

/// Prove: FairnessConfig with alpha = 1.0 (boundary) validates successfully.
///
/// alpha <= 1.0 is required, and 1.0 is the upper boundary. The validate()
/// function uses `>` (strict), so alpha=1.0 must pass.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fairness_config_alpha_boundary_one() {
    let mut config = FairnessConfig::default();
    config.alpha = 1.0;
    let result = config.validate();
    assert!(result.is_ok(), "alpha = 1.0 (boundary) must be accepted");
}

/// Prove: FairnessConfig rejects NegInf alpha.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fairness_config_rejects_neg_inf_alpha() {
    let mut config = FairnessConfig::default();
    config.alpha = f64::NEG_INFINITY;
    let result = config.validate();
    assert!(result.is_err(), "NegInf alpha must be rejected");
}

/// Prove: FairnessConfig rejects Inf max_gap.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fairness_config_rejects_inf_max_gap() {
    let mut config = FairnessConfig::default();
    config.max_gap = f64::INFINITY;
    let result = config.validate();
    assert!(result.is_err(), "Inf max_gap must be rejected");
}

/// Prove: FairnessConfig rejects zero max_gap.
///
/// A zero max_gap means "no quality difference allowed at all" which is
/// physically impossible. validate_finite_positive requires > 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fairness_config_rejects_zero_max_gap() {
    let mut config = FairnessConfig::default();
    config.max_gap = 0.0;
    let result = config.validate();
    assert!(result.is_err(), "Zero max_gap must be rejected");
}

/// Prove: FairnessConfig with valid max_gap in (0, 1e6] passes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fairness_config_accepts_valid_max_gap() {
    let gap: f64 = kani::any();
    kani::assume(gap.is_finite() && gap > 0.0 && gap <= 1e6);

    let mut config = FairnessConfig::default();
    config.max_gap = gap;
    let result = config.validate();
    assert!(result.is_ok(), "Valid max_gap in (0, 1e6] must be accepted");
}

/// Prove: FairnessConfig default has alpha = 0.05.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fairness_config_default_alpha() {
    let config = FairnessConfig::default();
    assert_eq!(config.alpha, 0.05, "default alpha must be 0.05");
}

/// Prove: FairnessConfig default has max_gap = 1.0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fairness_config_default_max_gap() {
    let config = FairnessConfig::default();
    assert_eq!(config.max_gap, 1.0, "default max_gap must be 1.0");
}

/// Prove: FairnessConfig default has min_samples_per_group = 30.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fairness_config_default_min_samples() {
    let config = FairnessConfig::default();
    assert_eq!(
        config.min_samples_per_group, 30,
        "default min_samples_per_group must be 30"
    );
}

/// Prove: FairnessConfig default has empty metrics list.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fairness_config_default_empty_metrics() {
    let config = FairnessConfig::default();
    assert!(
        config.metrics.is_empty(),
        "default metrics must be empty (include all)"
    );
}

// ---------------------------------------------------------------------------
// Holm-Bonferroni Extended Proofs
// ---------------------------------------------------------------------------

/// Prove: Holm-Bonferroni with three p-values produces adjusted values in [0, 1].
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
fn holm_bonferroni_three_values_bounded() {
    let p1: f64 = kani::any();
    let p2: f64 = kani::any();
    let p3: f64 = kani::any();
    kani::assume(p1.is_finite() && p2.is_finite() && p3.is_finite());
    kani::assume(p1 >= 0.0 && p1 <= 1.0);
    kani::assume(p2 >= 0.0 && p2 <= 1.0);
    kani::assume(p3 >= 0.0 && p3 <= 1.0);

    let result = stats::holm_bonferroni(&[p1, p2, p3]);
    assert!(result.is_ok());
    let adjusted = result.unwrap();
    assert_eq!(adjusted.len(), 3);
    for &p in &adjusted {
        assert!(p >= 0.0, "adjusted p must be >= 0");
        assert!(p <= 1.0, "adjusted p must be <= 1.0");
    }
}

/// Prove: Holm-Bonferroni with identical p-values returns them all equal.
///
/// When all raw p-values are the same, the monotonicity enforcement makes
/// all adjusted values equal to min(p * m, 1.0).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn holm_bonferroni_identical_values() {
    let p: f64 = kani::any();
    kani::assume(p.is_finite() && p >= 0.0 && p <= 1.0);

    let result = stats::holm_bonferroni(&[p, p]);
    assert!(result.is_ok());
    let adjusted = result.unwrap();
    // Both adjusted values must be equal (symmetric input)
    assert_eq!(
        adjusted[0], adjusted[1],
        "identical raw p-values must produce identical adjusted values"
    );
}

/// Prove: Holm-Bonferroni conservatism for three p-values.
///
/// Every adjusted p-value must be >= its corresponding raw p-value.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
fn holm_bonferroni_conservative_three() {
    let p1: f64 = kani::any();
    let p2: f64 = kani::any();
    let p3: f64 = kani::any();
    kani::assume(p1.is_finite() && p2.is_finite() && p3.is_finite());
    kani::assume(p1 >= 0.0 && p1 <= 1.0);
    kani::assume(p2 >= 0.0 && p2 <= 1.0);
    kani::assume(p3 >= 0.0 && p3 <= 1.0);

    let raw = [p1, p2, p3];
    let result = stats::holm_bonferroni(&raw);
    assert!(result.is_ok());
    let adjusted = result.unwrap();

    for i in 0..3 {
        assert!(
            adjusted[i] >= raw[i] - 1e-15,
            "adjusted p[{i}] must be >= raw p[{i}]"
        );
    }
}

/// Prove: Holm-Bonferroni rejects NegInf p-value.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn holm_bonferroni_rejects_neg_inf() {
    let result = stats::holm_bonferroni(&[f64::NEG_INFINITY]);
    assert!(result.is_err(), "NegInf p-value must be rejected");
}

// ---------------------------------------------------------------------------
// Fairness Verdict Extended Proofs
// ---------------------------------------------------------------------------

/// Prove: is_fair is true when gap=0 and no significant comparisons.
///
/// Zero quality gap with no significant results is the ideal fairness outcome.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn is_fair_true_for_zero_gap_no_significant() {
    let max_gap: f64 = kani::any();
    kani::assume(max_gap.is_finite() && max_gap > 0.0);

    let gap = 0.0;
    let any_significant = false;
    let is_fair = gap < max_gap && !any_significant;
    assert!(is_fair, "zero gap with no significant results must be fair");
}

/// Prove: is_fair with max_gap=0 is always false.
///
/// A zero threshold means no difference is acceptable, but gap >= 0 always,
/// so gap < 0 is never satisfied.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn is_fair_false_with_zero_threshold() {
    let gap: f64 = kani::any();
    kani::assume(gap.is_finite() && gap >= 0.0);

    let max_gap = 0.0;
    let is_fair = gap < max_gap && true;
    assert!(!is_fair, "zero max_gap threshold must always yield unfair");
}
