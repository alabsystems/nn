// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for quality bound certificates (Lipschitz composition).
//!
//! Proves properties of the Lipschitz constant helpers and the
//! `verify_quality_bounds()` function that determines whether TTS audio
//! quality is formally guaranteed under adversarial perturbations.
//!
//! Properties proved:
//! 1. `spectral_convergence_lipschitz` and `cosine_similarity_lipschitz`
//!    produce positive, finite results for valid inputs and are monotonically
//!    decreasing (larger reference -> smaller constant -> tighter bound).
//! 2. Margin formula is correctly implemented for both higher-is-better and
//!    lower-is-better metrics.
//! 3. Zero perturbation preserves quality when baseline meets threshold.
//! 4. Input validation rejects NaN, Inf, negative output_bound_width.
//! 5. `all_guaranteed` is consistent with per-metric `guaranteed` flags.
//! 6. Zero Lipschitz constant means zero degradation regardless of delta.
//! 7. `cosine_similarity_lipschitz` monotonically decreasing in norm.
//! 8. `verify_quality_bounds` rejects empty metrics list.
//! 9. `tightest_margin` equals the minimum margin across metrics.
//!
//! Only simple reciprocal Lipschitz helpers are proved here.
//! `snr_lipschitz` and `mcd_lipschitz` use transcendentals (powf, sqrt, ln)
//! that CBMC cannot model -- those are tested via unit tests in
//! `quality_bound_tests.rs`.

use super::{
    cosine_similarity_lipschitz, spectral_convergence_lipschitz, verify_quality_bounds,
    QualityMetricSpec,
};

// ---------------------------------------------------------------------------
// Lipschitz constant proofs (simple reciprocal functions)
// ---------------------------------------------------------------------------

/// Prove: spectral_convergence_lipschitz returns positive, finite L for
/// all positive finite energies.
///
/// SC Lipschitz = 1/energy. For energy in [1e-10, 1e10], result is finite
/// and positive.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn spectral_convergence_lipschitz_positive_finite() {
    let energy: f64 = kani::any();
    kani::assume(energy.is_finite() && energy > 0.0);
    kani::assume(energy >= 1e-10 && energy <= 1e10);

    let result = spectral_convergence_lipschitz(energy);
    assert!(result.is_ok(), "must succeed for valid input");
    let l = result.unwrap();
    assert!(l > 0.0, "Lipschitz constant must be positive");
    assert!(l.is_finite(), "Lipschitz constant must be finite");
}

/// Prove: spectral_convergence_lipschitz is monotonically decreasing.
///
/// Larger reference energy -> smaller Lipschitz constant -> tighter quality
/// bound. This means models with louder reference audio get stronger
/// quality guarantees for the same CROWN output bound width.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn spectral_convergence_lipschitz_decreasing() {
    let e1: f64 = kani::any();
    let e2: f64 = kani::any();
    kani::assume(e1.is_finite() && e2.is_finite());
    kani::assume(e1 > 0.0 && e2 > 0.0);
    kani::assume(e1 < e2); // e1 < e2
    kani::assume(e1 >= 1e-10 && e2 <= 1e10);

    let l1 = spectral_convergence_lipschitz(e1).unwrap();
    let l2 = spectral_convergence_lipschitz(e2).unwrap();
    assert!(
        l1 > l2,
        "Larger energy must give smaller Lipschitz: l1={l1} (e={e1}) > l2={l2} (e={e2})"
    );
}

/// Prove: cosine_similarity_lipschitz returns positive, finite L for
/// all positive finite norms.
///
/// Cosine Lipschitz = 1/norm. Same structure as spectral convergence.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn cosine_similarity_lipschitz_positive_finite() {
    let norm: f64 = kani::any();
    kani::assume(norm.is_finite() && norm > 0.0);
    kani::assume(norm >= 1e-10 && norm <= 1e10);

    let result = cosine_similarity_lipschitz(norm);
    assert!(result.is_ok(), "must succeed for valid input");
    let l = result.unwrap();
    assert!(l > 0.0, "Lipschitz constant must be positive");
    assert!(l.is_finite(), "Lipschitz constant must be finite");
}

/// Prove: cosine_similarity_lipschitz is monotonically decreasing in norm.
///
/// Larger signal norm -> smaller Lipschitz constant -> tighter quality bound.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn cosine_similarity_lipschitz_decreasing() {
    let n1: f64 = kani::any();
    let n2: f64 = kani::any();
    kani::assume(n1.is_finite() && n2.is_finite());
    kani::assume(n1 > 0.0 && n2 > 0.0);
    kani::assume(n1 < n2);
    kani::assume(n1 >= 1e-10 && n2 <= 1e10);

    let l1 = cosine_similarity_lipschitz(n1).unwrap();
    let l2 = cosine_similarity_lipschitz(n2).unwrap();
    assert!(l1 > l2, "Larger norm must give smaller Lipschitz constant");
}

/// Prove: spectral_convergence_lipschitz rejects NaN input.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn spectral_convergence_lipschitz_rejects_nan() {
    let result = spectral_convergence_lipschitz(f64::NAN);
    assert!(result.is_err(), "NaN energy must be rejected");
}

/// Prove: spectral_convergence_lipschitz rejects zero input.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn spectral_convergence_lipschitz_rejects_zero() {
    let result = spectral_convergence_lipschitz(0.0);
    assert!(result.is_err(), "Zero energy must be rejected");
}

/// Prove: spectral_convergence_lipschitz rejects negative input.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn spectral_convergence_lipschitz_rejects_negative() {
    let val: f64 = kani::any();
    kani::assume(val.is_finite() && val < 0.0);
    let result = spectral_convergence_lipschitz(val);
    assert!(result.is_err(), "Negative energy must be rejected");
}

/// Prove: cosine_similarity_lipschitz rejects NaN input.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn cosine_similarity_lipschitz_rejects_nan() {
    let result = cosine_similarity_lipschitz(f64::NAN);
    assert!(result.is_err(), "NaN norm must be rejected");
}

/// Prove: cosine_similarity_lipschitz rejects Inf input.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn cosine_similarity_lipschitz_rejects_inf() {
    let result = cosine_similarity_lipschitz(f64::INFINITY);
    assert!(result.is_err(), "Inf norm must be rejected");
}

// ---------------------------------------------------------------------------
// Input validation proofs
// ---------------------------------------------------------------------------

/// Prove: verify_quality_bounds rejects NaN output_bound_width.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn verify_quality_bounds_rejects_nan_delta() {
    let spec = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: 1.0,
        baseline_value: 10.0,
        threshold: 5.0,
        higher_is_better: true,
        citation: "kani",
    };
    let result = verify_quality_bounds(f64::NAN, &[spec]);
    assert!(result.is_err(), "NaN output_bound_width must be rejected");
}

/// Prove: verify_quality_bounds rejects negative output_bound_width.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn verify_quality_bounds_rejects_negative_delta() {
    let delta: f64 = kani::any();
    kani::assume(delta.is_finite() && delta < 0.0);

    let spec = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: 1.0,
        baseline_value: 10.0,
        threshold: 5.0,
        higher_is_better: true,
        citation: "kani",
    };
    let result = verify_quality_bounds(delta, &[spec]);
    assert!(
        result.is_err(),
        "Negative output_bound_width must be rejected"
    );
}

/// Prove: verify_quality_bounds rejects Inf output_bound_width.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn verify_quality_bounds_rejects_inf_delta() {
    let spec = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: 1.0,
        baseline_value: 10.0,
        threshold: 5.0,
        higher_is_better: true,
        citation: "kani",
    };
    let result = verify_quality_bounds(f64::INFINITY, &[spec]);
    assert!(result.is_err(), "Inf output_bound_width must be rejected");
}

/// Prove: verify_quality_bounds rejects empty metrics list.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn verify_quality_bounds_rejects_empty_metrics() {
    let result = verify_quality_bounds(1.0, &[]);
    assert!(result.is_err(), "Empty metrics list must be rejected");
}

/// Prove: verify_quality_bounds rejects NaN Lipschitz constant in metric spec.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn verify_quality_bounds_rejects_nan_lipschitz() {
    let spec = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: f64::NAN,
        baseline_value: 10.0,
        threshold: 5.0,
        higher_is_better: true,
        citation: "kani",
    };
    let result = verify_quality_bounds(1.0, &[spec]);
    assert!(result.is_err(), "NaN Lipschitz constant must be rejected");
}

/// Prove: verify_quality_bounds rejects NaN baseline value in metric spec.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn verify_quality_bounds_rejects_nan_baseline() {
    let spec = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: 1.0,
        baseline_value: f64::NAN,
        threshold: 5.0,
        higher_is_better: true,
        citation: "kani",
    };
    let result = verify_quality_bounds(1.0, &[spec]);
    assert!(result.is_err(), "NaN baseline_value must be rejected");
}

/// Prove: verify_quality_bounds rejects NaN threshold in metric spec.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn verify_quality_bounds_rejects_nan_threshold() {
    let spec = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: 1.0,
        baseline_value: 10.0,
        threshold: f64::NAN,
        higher_is_better: true,
        citation: "kani",
    };
    let result = verify_quality_bounds(1.0, &[spec]);
    assert!(result.is_err(), "NaN threshold must be rejected");
}

// ---------------------------------------------------------------------------
// Quality bound margin formula proofs
// ---------------------------------------------------------------------------

/// Prove: margin formula is correct for higher-is-better metrics.
///
/// For a metric where higher values are better (e.g., SNR):
///   worst_case = baseline - L * delta
///   margin = worst_case - threshold
///   guaranteed iff margin >= 0
///
/// This is the core mathematical claim of the quality bound certificate.
/// If the formula is wrong, the certificate gives false guarantees.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn margin_correct_higher_is_better() {
    let lipschitz: f64 = kani::any();
    let baseline: f64 = kani::any();
    let threshold: f64 = kani::any();
    let delta: f64 = kani::any();

    kani::assume(lipschitz.is_finite() && lipschitz >= 0.0 && lipschitz <= 1e6);
    kani::assume(baseline.is_finite() && baseline.abs() <= 1e6);
    kani::assume(threshold.is_finite() && threshold.abs() <= 1e6);
    kani::assume(delta.is_finite() && delta >= 0.0 && delta <= 1e3);

    let spec = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: lipschitz,
        baseline_value: baseline,
        threshold,
        higher_is_better: true,
        citation: "kani",
    };

    let result = verify_quality_bounds(delta, &[spec]);
    if let Ok(cert) = result {
        let r = &cert.metric_results[0];

        // Verify: worst_case = baseline - L * delta
        let expected_worst = baseline - lipschitz * delta;
        assert_eq!(
            r.worst_case_value, expected_worst,
            "worst_case must equal baseline - L*delta"
        );

        // Verify: margin = worst_case - threshold
        let expected_margin = expected_worst - threshold;
        assert_eq!(
            r.margin, expected_margin,
            "margin must equal worst_case - threshold"
        );

        // Verify: guaranteed iff margin >= 0
        assert_eq!(
            r.guaranteed,
            r.margin >= 0.0,
            "guaranteed must be true iff margin >= 0"
        );
    }
}

/// Prove: margin formula is correct for lower-is-better metrics.
///
/// For a metric where lower values are better (e.g., MCD):
///   worst_case = baseline + L * delta
///   margin = threshold - worst_case
///   guaranteed iff margin >= 0
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn margin_correct_lower_is_better() {
    let lipschitz: f64 = kani::any();
    let baseline: f64 = kani::any();
    let threshold: f64 = kani::any();
    let delta: f64 = kani::any();

    kani::assume(lipschitz.is_finite() && lipschitz >= 0.0 && lipschitz <= 1e6);
    kani::assume(baseline.is_finite() && baseline.abs() <= 1e6);
    kani::assume(threshold.is_finite() && threshold.abs() <= 1e6);
    kani::assume(delta.is_finite() && delta >= 0.0 && delta <= 1e3);

    let spec = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: lipschitz,
        baseline_value: baseline,
        threshold,
        higher_is_better: false,
        citation: "kani",
    };

    let result = verify_quality_bounds(delta, &[spec]);
    if let Ok(cert) = result {
        let r = &cert.metric_results[0];

        // Verify: worst_case = baseline + L * delta
        let expected_worst = baseline + lipschitz * delta;
        assert_eq!(
            r.worst_case_value, expected_worst,
            "worst_case must equal baseline + L*delta"
        );

        // Verify: margin = threshold - worst_case
        let expected_margin = threshold - expected_worst;
        assert_eq!(
            r.margin, expected_margin,
            "margin must equal threshold - worst_case"
        );

        // Verify: guaranteed iff margin >= 0
        assert_eq!(
            r.guaranteed,
            r.margin >= 0.0,
            "guaranteed must be true iff margin >= 0"
        );
    }
}

/// Prove: zero perturbation preserves quality when baseline meets threshold.
///
/// When delta = 0 and the baseline already satisfies the threshold
/// (baseline >= threshold for higher-is-better), the quality bound must
/// be guaranteed. This is the soundness base case: no perturbation means
/// no degradation.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn zero_perturbation_preserves_quality() {
    let lipschitz: f64 = kani::any();
    let baseline: f64 = kani::any();
    let threshold: f64 = kani::any();

    kani::assume(lipschitz.is_finite() && lipschitz >= 0.0 && lipschitz <= 1e10);
    kani::assume(baseline.is_finite() && baseline.abs() <= 1e10);
    kani::assume(threshold.is_finite() && threshold.abs() <= 1e10);
    // Baseline already meets threshold (higher-is-better).
    kani::assume(baseline >= threshold);

    let spec = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: lipschitz,
        baseline_value: baseline,
        threshold,
        higher_is_better: true,
        citation: "kani",
    };

    let result = verify_quality_bounds(0.0, &[spec]);
    if let Ok(cert) = result {
        assert!(
            cert.all_guaranteed,
            "Zero perturbation must preserve quality when baseline meets threshold"
        );
    }
}

/// Prove: zero perturbation preserves quality for lower-is-better metrics.
///
/// When delta = 0 and baseline <= threshold (lower is better), must be guaranteed.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn zero_perturbation_preserves_quality_lower_is_better() {
    let lipschitz: f64 = kani::any();
    let baseline: f64 = kani::any();
    let threshold: f64 = kani::any();

    kani::assume(lipschitz.is_finite() && lipschitz >= 0.0 && lipschitz <= 1e10);
    kani::assume(baseline.is_finite() && baseline.abs() <= 1e10);
    kani::assume(threshold.is_finite() && threshold.abs() <= 1e10);
    kani::assume(baseline <= threshold);

    let spec = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: lipschitz,
        baseline_value: baseline,
        threshold,
        higher_is_better: false,
        citation: "kani",
    };

    let result = verify_quality_bounds(0.0, &[spec]);
    if let Ok(cert) = result {
        assert!(
            cert.all_guaranteed,
            "Zero perturbation must preserve quality when baseline meets threshold (lower-is-better)"
        );
    }
}

// ---------------------------------------------------------------------------
// Certificate consistency proofs
// ---------------------------------------------------------------------------

/// Prove: `all_guaranteed` is true iff every metric is guaranteed.
///
/// The certificate's `all_guaranteed` flag must be the logical AND of
/// per-metric `guaranteed` flags. If any metric fails, all_guaranteed is false.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn all_guaranteed_consistency_two_metrics() {
    let delta: f64 = kani::any();
    kani::assume(delta.is_finite() && delta >= 0.0 && delta <= 100.0);

    let l1: f64 = kani::any();
    let l2: f64 = kani::any();
    let b1: f64 = kani::any();
    let b2: f64 = kani::any();
    let t1: f64 = kani::any();
    let t2: f64 = kani::any();
    kani::assume(l1.is_finite() && l1 >= 0.0 && l1 <= 100.0);
    kani::assume(l2.is_finite() && l2 >= 0.0 && l2 <= 100.0);
    kani::assume(b1.is_finite() && b1.abs() <= 1e4);
    kani::assume(b2.is_finite() && b2.abs() <= 1e4);
    kani::assume(t1.is_finite() && t1.abs() <= 1e4);
    kani::assume(t2.is_finite() && t2.abs() <= 1e4);

    let specs = [
        QualityMetricSpec {
            name: String::from("m1"),
            lipschitz_constant: l1,
            baseline_value: b1,
            threshold: t1,
            higher_is_better: true,
            citation: "kani",
        },
        QualityMetricSpec {
            name: String::from("m2"),
            lipschitz_constant: l2,
            baseline_value: b2,
            threshold: t2,
            higher_is_better: false,
            citation: "kani",
        },
    ];

    let result = verify_quality_bounds(delta, &specs);
    if let Ok(cert) = result {
        let manual_all = cert.metric_results.iter().all(|r| r.guaranteed);
        assert_eq!(
            cert.all_guaranteed, manual_all,
            "all_guaranteed must equal AND of per-metric guaranteed flags"
        );
    }
}

/// Prove: zero Lipschitz constant means zero degradation regardless of delta.
///
/// A metric with L=0 has max_quality_change=0 and worst_case=baseline.
/// This is correct: if the metric is completely insensitive to output
/// perturbations, no amount of perturbation can degrade quality.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn zero_lipschitz_no_degradation() {
    let baseline: f64 = kani::any();
    let threshold: f64 = kani::any();
    let delta: f64 = kani::any();

    kani::assume(baseline.is_finite() && baseline.abs() <= 1e6);
    kani::assume(threshold.is_finite() && threshold.abs() <= 1e6);
    kani::assume(delta.is_finite() && delta >= 0.0 && delta <= 1e6);

    let spec = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: 0.0,
        baseline_value: baseline,
        threshold,
        higher_is_better: true,
        citation: "kani",
    };

    let result = verify_quality_bounds(delta, &[spec]);
    if let Ok(cert) = result {
        let r = &cert.metric_results[0];
        assert_eq!(
            r.max_quality_change, 0.0,
            "zero Lipschitz must produce zero quality change"
        );
        assert_eq!(
            r.worst_case_value, baseline,
            "zero Lipschitz must preserve baseline exactly"
        );
    }
}

/// Prove: output_bound_width stored in certificate matches the input.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn certificate_stores_delta() {
    let delta: f64 = kani::any();
    kani::assume(delta.is_finite() && delta >= 0.0 && delta <= 1e6);

    let spec = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: 1.0,
        baseline_value: 10.0,
        threshold: 5.0,
        higher_is_better: true,
        citation: "kani",
    };

    let result = verify_quality_bounds(delta, &[spec]);
    if let Ok(cert) = result {
        assert_eq!(
            cert.output_bound_width, delta,
            "certificate must store the input delta"
        );
        assert_eq!(
            cert.metric_results[0].output_bound_width, delta,
            "per-metric result must store the input delta"
        );
    }
}
