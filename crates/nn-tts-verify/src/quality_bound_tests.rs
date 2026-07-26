// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for quality bound certificates.

use super::*;

// ---------------------------------------------------------------------------
// Lipschitz constant computation tests
// ---------------------------------------------------------------------------

#[test]
fn test_snr_lipschitz_basic() {
    // signal_rms=0.1, baseline_snr=20dB → noise_rms = 0.1 * 10^(-20/20) = 0.01
    // L = 20 / (ln(10) * 0.01) ≈ 868.6
    let l = snr_lipschitz(0.1, 20.0).unwrap();
    let noise_rms = 0.1 * 10.0_f64.powf(-20.0 / 20.0);
    let expected = 20.0 / (10.0_f64.ln() * noise_rms);
    assert!(
        (l - expected).abs() < 0.01,
        "SNR Lipschitz = {l}, expected {expected}"
    );
}

#[test]
fn test_snr_lipschitz_higher_snr_gives_larger_constant() {
    // Higher baseline SNR → smaller noise → LARGER Lipschitz constant
    // (more sensitive to perturbation because noise floor is lower).
    let l_20db = snr_lipschitz(0.1, 20.0).unwrap();
    let l_40db = snr_lipschitz(0.1, 40.0).unwrap();
    assert!(
        l_40db > l_20db,
        "Higher SNR should give larger Lipschitz: 40dB={l_40db}, 20dB={l_20db}"
    );
}

#[test]
fn test_snr_lipschitz_rejects_zero_rms() {
    assert!(snr_lipschitz(0.0, 20.0).is_err());
}

#[test]
fn test_snr_lipschitz_rejects_negative_rms() {
    assert!(snr_lipschitz(-1.0, 20.0).is_err());
}

#[test]
fn test_snr_lipschitz_rejects_nan_rms() {
    assert!(snr_lipschitz(f64::NAN, 20.0).is_err());
}

#[test]
fn test_snr_lipschitz_rejects_nan_snr() {
    assert!(snr_lipschitz(0.1, f64::NAN).is_err());
}

/// Verify the Lipschitz bound is sound: L × δ must be ≥ actual SNR degradation.
///
/// This test demonstrates the bug that existed when signal_rms was used
/// as the denominator instead of noise_rms.
#[test]
fn test_snr_lipschitz_soundness() {
    let signal_rms = 0.15;
    let baseline_snr_db = 25.0;
    let delta = 0.01; // CROWN output perturbation magnitude

    let l = snr_lipschitz(signal_rms, baseline_snr_db).unwrap();
    let predicted_max_degradation = l * delta;

    // Compute actual worst-case degradation.
    let noise_rms = signal_rms * 10.0_f64.powf(-baseline_snr_db / 20.0);
    let worst_noise = noise_rms + delta;
    let worst_snr = 20.0 * (signal_rms / worst_noise).log10();
    let actual_degradation = baseline_snr_db - worst_snr;

    // Lipschitz bound MUST be >= actual degradation (conservative).
    assert!(
        predicted_max_degradation >= actual_degradation,
        "Lipschitz bound is UNSOUND: predicted max={predicted_max_degradation:.4} dB \
         < actual={actual_degradation:.4} dB \
         (L={l:.2}, δ={delta}, noise_rms={noise_rms:.6})"
    );
}

#[test]
fn test_spectral_convergence_lipschitz_basic() {
    // Energy = 10.0 → L = 1/10 = 0.1
    let l = spectral_convergence_lipschitz(10.0).unwrap();
    assert!((l - 0.1).abs() < 1e-10, "SC Lipschitz = {l}");
}

#[test]
fn test_spectral_convergence_lipschitz_rejects_zero() {
    assert!(spectral_convergence_lipschitz(0.0).is_err());
}

#[test]
fn test_mcd_lipschitz_basic() {
    // n_frames = 100 → L = (10√2/ln10) / √100 = 10√2/ln10 / 10
    let l = mcd_lipschitz(100).unwrap();
    let expected = 10.0 * 2.0_f64.sqrt() / 10.0_f64.ln() / 10.0;
    assert!(
        (l - expected).abs() < 1e-10,
        "MCD Lipschitz = {l}, expected {expected}"
    );
}

#[test]
fn test_mcd_lipschitz_more_frames_tighter() {
    // More frames → tighter (smaller) Lipschitz constant.
    let l_few = mcd_lipschitz(10).unwrap();
    let l_many = mcd_lipschitz(1000).unwrap();
    assert!(l_many < l_few, "More frames should give smaller Lipschitz");
}

#[test]
fn test_mcd_lipschitz_rejects_zero_frames() {
    assert!(mcd_lipschitz(0).is_err());
}

#[test]
fn test_cosine_similarity_lipschitz_basic() {
    // L2 norm = 5.0 → L = 1/5 = 0.2
    let l = cosine_similarity_lipschitz(5.0).unwrap();
    assert!((l - 0.2).abs() < 1e-10, "Cosine Lipschitz = {l}");
}

#[test]
fn test_cosine_similarity_lipschitz_rejects_zero() {
    assert!(cosine_similarity_lipschitz(0.0).is_err());
}

// ---------------------------------------------------------------------------
// Quality bound verification tests
// ---------------------------------------------------------------------------

#[test]
fn test_verify_quality_bounds_all_pass() {
    let metrics = vec![
        QualityMetricSpec {
            name: "SNR".into(),
            lipschitz_constant: 10.0,
            baseline_value: 30.0, // 30 dB SNR baseline
            threshold: 10.0,      // Must stay above 10 dB
            higher_is_better: true,
            citation: "test",
        },
        QualityMetricSpec {
            name: "MCD".into(),
            lipschitz_constant: 1.0,
            baseline_value: 3.0, // 3 dB MCD baseline
            threshold: 6.0,      // Must stay below 6 dB
            higher_is_better: false,
            citation: "test",
        },
    ];

    let cert = verify_quality_bounds(0.5, &metrics).unwrap();
    assert!(cert.all_guaranteed, "All metrics should be guaranteed");

    // SNR: worst = 30 - 10*0.5 = 25 dB, margin = 25 - 10 = 15
    let snr = &cert.metric_results[0];
    assert_eq!(snr.metric_name, "SNR");
    assert!((snr.max_quality_change - 5.0).abs() < 1e-10);
    assert!((snr.worst_case_value - 25.0).abs() < 1e-10);
    assert!((snr.margin - 15.0).abs() < 1e-10);
    assert!(snr.guaranteed);

    // MCD: worst = 3 + 1*0.5 = 3.5, margin = 6 - 3.5 = 2.5
    let mcd = &cert.metric_results[1];
    assert_eq!(mcd.metric_name, "MCD");
    assert!((mcd.max_quality_change - 0.5).abs() < 1e-10);
    assert!((mcd.worst_case_value - 3.5).abs() < 1e-10);
    assert!((mcd.margin - 2.5).abs() < 1e-10);
    assert!(mcd.guaranteed);
}

#[test]
fn test_verify_quality_bounds_one_fails() {
    let metrics = vec![QualityMetricSpec {
        name: "SNR".into(),
        lipschitz_constant: 100.0, // Very sensitive
        baseline_value: 15.0,
        threshold: 10.0,
        higher_is_better: true,
        citation: "test",
    }];

    // δ = 0.1, L*δ = 10.0, worst = 15 - 10 = 5 < 10 → fails
    let cert = verify_quality_bounds(0.1, &metrics).unwrap();
    assert!(!cert.all_guaranteed);
    assert!(!cert.metric_results[0].guaranteed);
    assert!((cert.metric_results[0].margin - (-5.0)).abs() < 1e-10);
}

#[test]
fn test_verify_quality_bounds_zero_width() {
    // Zero perturbation → all metrics trivially guaranteed.
    let metrics = vec![QualityMetricSpec {
        name: "SNR".into(),
        lipschitz_constant: 1000.0,
        baseline_value: 20.0,
        threshold: 10.0,
        higher_is_better: true,
        citation: "test",
    }];

    let cert = verify_quality_bounds(0.0, &metrics).unwrap();
    assert!(cert.all_guaranteed);
    assert!((cert.metric_results[0].worst_case_value - 20.0).abs() < 1e-10);
}

#[test]
fn test_verify_quality_bounds_tightest_metric() {
    let metrics = vec![
        QualityMetricSpec {
            name: "A".into(),
            lipschitz_constant: 1.0,
            baseline_value: 20.0,
            threshold: 10.0,
            higher_is_better: true,
            citation: "test",
        },
        QualityMetricSpec {
            name: "B".into(),
            lipschitz_constant: 1.0,
            baseline_value: 12.0, // Closer to threshold
            threshold: 10.0,
            higher_is_better: true,
            citation: "test",
        },
    ];

    let cert = verify_quality_bounds(1.0, &metrics).unwrap();
    assert_eq!(cert.tightest_metric, "B");
    // B: worst = 12 - 1 = 11, margin = 11 - 10 = 1
    assert!((cert.tightest_margin - 1.0).abs() < 1e-10);
}

#[test]
fn test_verify_quality_bounds_lower_is_better() {
    // MCD-like metric: lower is better.
    let metrics = vec![QualityMetricSpec {
        name: "MCD".into(),
        lipschitz_constant: 2.0,
        baseline_value: 4.0,
        threshold: 6.0, // Must stay below 6
        higher_is_better: false,
        citation: "test",
    }];

    // δ = 0.5, L*δ = 1.0, worst = 4 + 1 = 5 < 6 → passes
    let cert = verify_quality_bounds(0.5, &metrics).unwrap();
    assert!(cert.all_guaranteed);
    assert!((cert.metric_results[0].worst_case_value - 5.0).abs() < 1e-10);
    assert!((cert.metric_results[0].margin - 1.0).abs() < 1e-10);
}

#[test]
fn test_verify_quality_bounds_lower_is_better_fails() {
    let metrics = vec![QualityMetricSpec {
        name: "MCD".into(),
        lipschitz_constant: 10.0,
        baseline_value: 4.0,
        threshold: 6.0,
        higher_is_better: false,
        citation: "test",
    }];

    // δ = 0.5, L*δ = 5.0, worst = 4 + 5 = 9 > 6 → fails
    let cert = verify_quality_bounds(0.5, &metrics).unwrap();
    assert!(!cert.all_guaranteed);
    assert!((cert.metric_results[0].worst_case_value - 9.0).abs() < 1e-10);
    assert!((cert.metric_results[0].margin - (-3.0)).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Input validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_verify_quality_bounds_rejects_negative_width() {
    let metrics = vec![QualityMetricSpec {
        name: "A".into(),
        lipschitz_constant: 1.0,
        baseline_value: 20.0,
        threshold: 10.0,
        higher_is_better: true,
        citation: "test",
    }];
    assert!(verify_quality_bounds(-1.0, &metrics).is_err());
}

#[test]
fn test_verify_quality_bounds_rejects_nan_width() {
    let metrics = vec![QualityMetricSpec {
        name: "A".into(),
        lipschitz_constant: 1.0,
        baseline_value: 20.0,
        threshold: 10.0,
        higher_is_better: true,
        citation: "test",
    }];
    assert!(verify_quality_bounds(f64::NAN, &metrics).is_err());
}

#[test]
fn test_verify_quality_bounds_rejects_inf_width() {
    let metrics = vec![QualityMetricSpec {
        name: "A".into(),
        lipschitz_constant: 1.0,
        baseline_value: 20.0,
        threshold: 10.0,
        higher_is_better: true,
        citation: "test",
    }];
    assert!(verify_quality_bounds(f64::INFINITY, &metrics).is_err());
}

#[test]
fn test_verify_quality_bounds_rejects_empty_metrics() {
    assert!(verify_quality_bounds(1.0, &[]).is_err());
}

#[test]
fn test_verify_quality_bounds_rejects_nan_lipschitz() {
    let metrics = vec![QualityMetricSpec {
        name: "A".into(),
        lipschitz_constant: f64::NAN,
        baseline_value: 20.0,
        threshold: 10.0,
        higher_is_better: true,
        citation: "test",
    }];
    assert!(verify_quality_bounds(1.0, &metrics).is_err());
}

#[test]
fn test_verify_quality_bounds_rejects_negative_lipschitz() {
    let metrics = vec![QualityMetricSpec {
        name: "A".into(),
        lipschitz_constant: -1.0,
        baseline_value: 20.0,
        threshold: 10.0,
        higher_is_better: true,
        citation: "test",
    }];
    assert!(verify_quality_bounds(1.0, &metrics).is_err());
}

#[test]
fn test_verify_quality_bounds_rejects_nan_baseline() {
    let metrics = vec![QualityMetricSpec {
        name: "A".into(),
        lipschitz_constant: 1.0,
        baseline_value: f64::NAN,
        threshold: 10.0,
        higher_is_better: true,
        citation: "test",
    }];
    assert!(verify_quality_bounds(1.0, &metrics).is_err());
}

#[test]
fn test_verify_quality_bounds_rejects_nan_threshold() {
    let metrics = vec![QualityMetricSpec {
        name: "A".into(),
        lipschitz_constant: 1.0,
        baseline_value: 20.0,
        threshold: f64::NAN,
        higher_is_better: true,
        citation: "test",
    }];
    assert!(verify_quality_bounds(1.0, &metrics).is_err());
}

// ---------------------------------------------------------------------------
// Standard quality specs tests
// ---------------------------------------------------------------------------

#[test]
fn test_standard_quality_specs_construction() {
    let specs = standard_quality_specs(
        0.1,  // signal_rms
        1.0,  // signal_l2_norm
        10.0, // reference_spectral_energy
        100,  // n_frames
        30.0, // baseline_snr
        0.05, // baseline_sc
        3.5,  // baseline_mcd
        0.95, // baseline_cosine
    )
    .unwrap();

    assert_eq!(specs[0].name, "SNR");
    assert_eq!(specs[1].name, "spectral_convergence");
    assert_eq!(specs[2].name, "MCD");
    assert_eq!(specs[3].name, "cosine_similarity");

    // Verify higher_is_better settings.
    assert!(specs[0].higher_is_better); // SNR
    assert!(!specs[1].higher_is_better); // spectral_convergence (lower = better)
    assert!(!specs[2].higher_is_better); // MCD (lower = better)
    assert!(specs[3].higher_is_better); // cosine_similarity
}

#[test]
fn test_standard_quality_specs_with_verify() {
    let specs = standard_quality_specs(0.1, 1.0, 10.0, 100, 30.0, 0.05, 3.5, 0.95).unwrap();

    // Small perturbation: all should pass.
    let cert = verify_quality_bounds(0.001, &specs).unwrap();
    assert!(
        cert.all_guaranteed,
        "Small perturbation should guarantee all metrics: {:?}",
        cert.metric_results
            .iter()
            .map(|r| format!("{}: margin={:.4}", r.metric_name, r.margin))
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_standard_quality_specs_large_perturbation() {
    let specs = standard_quality_specs(0.1, 1.0, 10.0, 100, 30.0, 0.05, 3.5, 0.95).unwrap();

    // Large perturbation: some should fail.
    let cert = verify_quality_bounds(10.0, &specs).unwrap();
    assert!(
        !cert.all_guaranteed,
        "Large perturbation should violate at least one metric"
    );
}

// ---------------------------------------------------------------------------
// End-to-end: CROWN output width → quality certificate
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_crown_to_quality_certificate() {
    // Simulate what Phase 17 compose tests will do:
    // 1. CROWN verification gives output_bound_width = 0.005 (tight CROWN bound)
    // 2. Build quality specs from measured baselines
    // 3. Verify quality bounds hold
    //
    // Note: with the corrected SNR Lipschitz constant (using noise_rms),
    // δ must be small enough that L_snr × δ < baseline - threshold.
    // At 25 dB SNR with signal_rms=0.15: L_snr ≈ 1030, so δ < 0.0145.

    let output_bound_width = 0.005;

    let specs = vec![
        QualityMetricSpec {
            name: "SNR".into(),
            lipschitz_constant: snr_lipschitz(0.15, 25.0).unwrap(),
            baseline_value: 25.0,
            threshold: 10.0,
            higher_is_better: true,
            citation: "ITU-T P.56",
        },
        QualityMetricSpec {
            name: "spectral_convergence".into(),
            lipschitz_constant: spectral_convergence_lipschitz(5.0).unwrap(),
            baseline_value: 0.02,
            threshold: 0.5,
            higher_is_better: false,
            citation: "Arik et al. (2018)",
        },
        QualityMetricSpec {
            name: "MCD".into(),
            lipschitz_constant: mcd_lipschitz(50).unwrap(),
            baseline_value: 4.0,
            threshold: 6.0,
            higher_is_better: false,
            citation: "Kubichek (1993)",
        },
        QualityMetricSpec {
            name: "cosine_similarity".into(),
            lipschitz_constant: cosine_similarity_lipschitz(2.0).unwrap(),
            baseline_value: 0.95,
            threshold: 0.8,
            higher_is_better: true,
            citation: "Jia et al. (2018)",
        },
    ];

    let cert = verify_quality_bounds(output_bound_width, &specs).unwrap();

    // With δ = 0.005, all metrics should pass.
    for r in &cert.metric_results {
        assert!(
            r.guaranteed,
            "{} failed: worst={:.4}, threshold={:.4}, margin={:.4}",
            r.metric_name, r.worst_case_value, r.threshold, r.margin
        );
    }
    assert!(cert.all_guaranteed);
    assert!(cert.tightest_margin > 0.0);
}
