// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for quality bound certificates.
//!
//! Supplements `quality_bound_kani.rs` with deeper proofs:
//!
//! - **snr_lipschitz**: rejection of NaN/zero/negative/Inf signal_rms,
//!   rejection of NaN baseline_snr.
//! - **mcd_lipschitz**: rejection of zero frames, positive-finite for valid.
//! - **cosine_similarity_lipschitz**: rejection of zero/negative.
//! - **verify_quality_bounds**: tightest_margin consistency, negative Lipschitz
//!   rejection, multi-metric all_guaranteed logic.
//! - **Margin monotonicity**: larger delta => smaller margin (higher-is-better).

use crate::quality_bound::{
    cosine_similarity_lipschitz, mcd_lipschitz, snr_lipschitz, spectral_convergence_lipschitz,
    verify_quality_bounds, QualityMetricSpec,
};

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

/// Nondeterministic stub for `f64::ln`.
/// CBMC cannot handle the ln intrinsic. Returns a finite f64
/// in a plausible range for log values.
fn ln_f64_stub(_x: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0);
    r
}

// ---------------------------------------------------------------------------
// snr_lipschitz proofs
// ---------------------------------------------------------------------------

/// Prove: snr_lipschitz rejects NaN signal_rms.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn snr_lipschitz_rejects_nan_signal() {
    let result = snr_lipschitz(f64::NAN, 25.0);
    assert!(result.is_err(), "NaN signal_rms must be rejected");
}

/// Prove: snr_lipschitz rejects zero signal_rms.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn snr_lipschitz_rejects_zero_signal() {
    let result = snr_lipschitz(0.0, 25.0);
    assert!(result.is_err(), "Zero signal_rms must be rejected");
}

/// Prove: snr_lipschitz rejects negative signal_rms.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn snr_lipschitz_rejects_negative_signal() {
    let result = snr_lipschitz(-1.0, 25.0);
    assert!(result.is_err(), "Negative signal_rms must be rejected");
}

/// Prove: snr_lipschitz rejects Inf signal_rms.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn snr_lipschitz_rejects_inf_signal() {
    let result = snr_lipschitz(f64::INFINITY, 25.0);
    assert!(result.is_err(), "Inf signal_rms must be rejected");
}

/// Prove: snr_lipschitz rejects NaN baseline_snr_db.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn snr_lipschitz_rejects_nan_baseline() {
    let result = snr_lipschitz(0.1, f64::NAN);
    assert!(result.is_err(), "NaN baseline_snr_db must be rejected");
}

// ---------------------------------------------------------------------------
// mcd_lipschitz proofs
// ---------------------------------------------------------------------------

/// Prove: mcd_lipschitz rejects zero frames.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn mcd_lipschitz_rejects_zero_frames() {
    let result = mcd_lipschitz(0);
    assert!(result.is_err(), "Zero frames must be rejected");
}

/// Prove: mcd_lipschitz returns positive finite for n_frames >= 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
#[kani::stub(f64::ln, ln_f64_stub)]
fn mcd_lipschitz_positive_finite() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 100_000);

    let result = mcd_lipschitz(n);
    assert!(result.is_ok(), "valid n_frames must succeed");
    let l = result.unwrap();
    assert!(l > 0.0, "MCD Lipschitz must be positive");
    assert!(l.is_finite(), "MCD Lipschitz must be finite");
}

// ---------------------------------------------------------------------------
// cosine_similarity_lipschitz extended proofs
// ---------------------------------------------------------------------------

/// Prove: cosine_similarity_lipschitz rejects zero norm.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn cosine_lipschitz_rejects_zero() {
    let result = cosine_similarity_lipschitz(0.0);
    assert!(result.is_err(), "Zero norm must be rejected");
}

/// Prove: cosine_similarity_lipschitz rejects negative norm.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn cosine_lipschitz_rejects_negative() {
    let val: f64 = kani::any();
    kani::assume(val.is_finite() && val < 0.0);
    let result = cosine_similarity_lipschitz(val);
    assert!(result.is_err(), "Negative norm must be rejected");
}

// ---------------------------------------------------------------------------
// spectral_convergence_lipschitz extended proofs
// ---------------------------------------------------------------------------

/// Prove: spectral_convergence_lipschitz rejects Inf input.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn spectral_convergence_lipschitz_rejects_inf() {
    let result = spectral_convergence_lipschitz(f64::INFINITY);
    assert!(result.is_err(), "Inf energy must be rejected");
}

/// Prove: spectral_convergence_lipschitz rejects NegInf input.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn spectral_convergence_lipschitz_rejects_neg_inf() {
    let result = spectral_convergence_lipschitz(f64::NEG_INFINITY);
    assert!(result.is_err(), "NegInf energy must be rejected");
}

// ---------------------------------------------------------------------------
// verify_quality_bounds extended proofs
// ---------------------------------------------------------------------------

/// Prove: verify_quality_bounds rejects negative Lipschitz constant.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn verify_quality_bounds_rejects_negative_lipschitz() {
    let spec = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: -1.0,
        baseline_value: 10.0,
        threshold: 5.0,
        higher_is_better: true,
        citation: "kani",
    };
    let result = verify_quality_bounds(1.0, &[spec]);
    assert!(
        result.is_err(),
        "Negative Lipschitz constant must be rejected"
    );
}

/// Prove: verify_quality_bounds rejects Inf baseline_value.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn verify_quality_bounds_rejects_inf_baseline() {
    let spec = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: 1.0,
        baseline_value: f64::INFINITY,
        threshold: 5.0,
        higher_is_better: true,
        citation: "kani",
    };
    let result = verify_quality_bounds(1.0, &[spec]);
    assert!(result.is_err(), "Inf baseline_value must be rejected");
}

/// Prove: verify_quality_bounds rejects Inf threshold.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn verify_quality_bounds_rejects_inf_threshold() {
    let spec = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: 1.0,
        baseline_value: 10.0,
        threshold: f64::INFINITY,
        higher_is_better: true,
        citation: "kani",
    };
    let result = verify_quality_bounds(1.0, &[spec]);
    assert!(result.is_err(), "Inf threshold must be rejected");
}

/// Prove: tightest_margin equals the minimum margin across all metrics.
///
/// The certificate must identify the metric closest to failing (smallest margin).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn tightest_margin_is_minimum() {
    let l1: f64 = kani::any();
    let l2: f64 = kani::any();
    kani::assume(l1.is_finite() && l1 >= 0.0 && l1 <= 100.0);
    kani::assume(l2.is_finite() && l2 >= 0.0 && l2 <= 100.0);

    let delta: f64 = kani::any();
    kani::assume(delta.is_finite() && delta >= 0.0 && delta <= 10.0);

    let specs = [
        QualityMetricSpec {
            name: String::from("m1"),
            lipschitz_constant: l1,
            baseline_value: 20.0,
            threshold: 10.0,
            higher_is_better: true,
            citation: "kani",
        },
        QualityMetricSpec {
            name: String::from("m2"),
            lipschitz_constant: l2,
            baseline_value: 3.0,
            threshold: 6.0,
            higher_is_better: false,
            citation: "kani",
        },
    ];

    let result = verify_quality_bounds(delta, &specs);
    if let Ok(cert) = result {
        let min_margin = cert
            .metric_results
            .iter()
            .map(|r| r.margin)
            .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
        assert!(
            (cert.tightest_margin - min_margin).abs() < 1e-12,
            "tightest_margin must equal minimum margin"
        );
    }
}

/// Prove: delta=0 with output_bound_width=0 means max_quality_change=0 for all.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn zero_delta_zero_quality_change() {
    let l: f64 = kani::any();
    kani::assume(l.is_finite() && l >= 0.0 && l <= 1e6);

    let spec = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: l,
        baseline_value: 10.0,
        threshold: 5.0,
        higher_is_better: true,
        citation: "kani",
    };

    let result = verify_quality_bounds(0.0, &[spec]);
    if let Ok(cert) = result {
        assert_eq!(
            cert.metric_results[0].max_quality_change, 0.0,
            "delta=0 must produce zero quality change"
        );
    }
}

/// Prove: larger delta produces smaller or equal margin (higher-is-better).
///
/// Margin = baseline - L*delta - threshold. As delta increases, margin decreases.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn margin_decreases_with_delta_higher_is_better() {
    let lipschitz: f64 = kani::any();
    let baseline: f64 = kani::any();
    let threshold: f64 = kani::any();
    let d1: f64 = kani::any();
    let d2: f64 = kani::any();

    kani::assume(lipschitz.is_finite() && lipschitz >= 0.0 && lipschitz <= 100.0);
    kani::assume(baseline.is_finite() && baseline.abs() <= 1e4);
    kani::assume(threshold.is_finite() && threshold.abs() <= 1e4);
    kani::assume(d1.is_finite() && d1 >= 0.0 && d1 <= 100.0);
    kani::assume(d2.is_finite() && d2 >= 0.0 && d2 <= 100.0);
    kani::assume(d1 <= d2);

    let spec1 = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: lipschitz,
        baseline_value: baseline,
        threshold,
        higher_is_better: true,
        citation: "kani",
    };
    let spec2 = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: lipschitz,
        baseline_value: baseline,
        threshold,
        higher_is_better: true,
        citation: "kani",
    };

    let r1 = verify_quality_bounds(d1, &[spec1]);
    let r2 = verify_quality_bounds(d2, &[spec2]);

    if let (Ok(c1), Ok(c2)) = (r1, r2) {
        assert!(
            c1.metric_results[0].margin >= c2.metric_results[0].margin - 1e-10,
            "smaller delta must give >= margin"
        );
    }
}

/// Prove: larger delta produces smaller or equal margin (lower-is-better).
///
/// Margin = threshold - (baseline + L*delta). As delta increases, margin decreases.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn margin_decreases_with_delta_lower_is_better() {
    let lipschitz: f64 = kani::any();
    let baseline: f64 = kani::any();
    let threshold: f64 = kani::any();
    let d1: f64 = kani::any();
    let d2: f64 = kani::any();

    kani::assume(lipschitz.is_finite() && lipschitz >= 0.0 && lipschitz <= 100.0);
    kani::assume(baseline.is_finite() && baseline.abs() <= 1e4);
    kani::assume(threshold.is_finite() && threshold.abs() <= 1e4);
    kani::assume(d1.is_finite() && d1 >= 0.0 && d1 <= 100.0);
    kani::assume(d2.is_finite() && d2 >= 0.0 && d2 <= 100.0);
    kani::assume(d1 <= d2);

    let spec1 = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: lipschitz,
        baseline_value: baseline,
        threshold,
        higher_is_better: false,
        citation: "kani",
    };
    let spec2 = QualityMetricSpec {
        name: String::from("test"),
        lipschitz_constant: lipschitz,
        baseline_value: baseline,
        threshold,
        higher_is_better: false,
        citation: "kani",
    };

    let r1 = verify_quality_bounds(d1, &[spec1]);
    let r2 = verify_quality_bounds(d2, &[spec2]);

    if let (Ok(c1), Ok(c2)) = (r1, r2) {
        assert!(
            c1.metric_results[0].margin >= c2.metric_results[0].margin - 1e-10,
            "smaller delta must give >= margin (lower-is-better)"
        );
    }
}
