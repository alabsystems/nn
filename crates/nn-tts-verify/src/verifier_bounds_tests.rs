// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional tests for TtsVerifier, Certificate, HardBoundsConfig,
//! RejectionPolicy, CheckOverrides, and bound check functions.
//!
//! Part of #3819.

use crate::bounds::{
    check_duration, check_no_clipping, check_no_dc_offset, check_non_silence,
    check_spectral_coverage, check_tail_energy, HardBound, SpectralCoverageConfig,
};
use crate::certificate::Certificate;
use crate::config::{CheckOverrides, HardBoundsConfig, QualityConfig, RejectionPolicy};
use crate::cost_model::HardwareCostModel;
use crate::deterministic;
use crate::error::TtsVerifyError;
use crate::quality::QualityMetric;
use crate::test_audio_helpers::sine_wave_full;
use crate::verifier::TtsVerifier;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a synthetic speech-like signal with harmonics spanning multiple
/// frequency bands. Satisfies spectral coverage at 24 kHz / 8 bands.
fn rich_speech(sample_rate: u32, duration_sec: f64) -> Vec<f32> {
    let n = (f64::from(sample_rate) * duration_sec) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let pi2 = 2.0 * std::f32::consts::PI;
            let mut s = 0.0_f32;
            for k in 1..=50 {
                let freq = 200.0 * k as f32;
                if freq > sample_rate as f32 / 2.0 {
                    break;
                }
                s += (1.0 / k as f32) * (pi2 * freq * t).sin();
            }
            s * 0.15
        })
        .collect()
}

// ===========================================================================
// TtsVerifier builder edge cases
// ===========================================================================

#[test]
fn test_builder_chaining_sample_rate_and_quality() {
    let v = TtsVerifier::builder()
        .sample_rate(44100)
        .with_quality()
        .build()
        .expect("chained builder should succeed");
    assert_eq!(v.sample_rate, 44100);
    assert!(v.quality.is_some());
}

#[test]
fn test_builder_custom_quality_config() {
    let qc = QualityConfig {
        max_mcd_db: 8.0,
        min_hnr_db: 10.0,
        ..Default::default()
    };
    let v = TtsVerifier::builder()
        .quality(qc)
        .build()
        .expect("builder with custom quality config should succeed");
    let qv = v.quality.as_ref().expect("quality config should be set");
    assert!((qv.max_mcd_db - 8.0).abs() < 1e-10);
    assert!((qv.min_hnr_db - 10.0).abs() < 1e-10);
}

#[test]
fn test_builder_hard_bounds_with_rejection_policy() {
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Warn,
        ..Default::default()
    };
    let v = TtsVerifier::builder()
        .hard_bounds(hb)
        .build()
        .expect("builder should succeed");
    assert_eq!(v.hard_bounds.rejection_policy, RejectionPolicy::Warn);
}

// ===========================================================================
// TtsVerifier::verify input validation
// ===========================================================================

#[test]
fn test_verify_all_nan_input() {
    let v = TtsVerifier::builder().build().unwrap();
    let samples = vec![f32::NAN; 100];
    let err = v.verify(&samples).unwrap_err();
    match err {
        TtsVerifyError::NonFiniteInput { count } => assert_eq!(count, 100),
        other => panic!("expected NonFiniteInput, got: {other:?}"),
    }
}

#[test]
fn test_verify_inf_input() {
    let v = TtsVerifier::builder().build().unwrap();
    let samples = vec![0.1, f32::INFINITY, 0.2];
    let err = v.verify(&samples).unwrap_err();
    assert!(matches!(err, TtsVerifyError::NonFiniteInput { count: 1 }));
}

#[test]
fn test_verify_neg_inf_input() {
    let v = TtsVerifier::builder().build().unwrap();
    let samples = vec![f32::NEG_INFINITY, 0.0];
    let err = v.verify(&samples).unwrap_err();
    assert!(matches!(err, TtsVerifyError::NonFiniteInput { count: 1 }));
}

// ===========================================================================
// TtsVerifier::verify_with_reference edge cases
// ===========================================================================

#[test]
fn test_verify_with_reference_empty_reference() {
    let v = TtsVerifier::builder().build().unwrap();
    let cand = rich_speech(24000, 0.5);
    let err = v.verify_with_reference(&cand, &[]).unwrap_err();
    assert!(matches!(err, TtsVerifyError::EmptyInput));
}

#[test]
fn test_verify_with_reference_empty_candidate() {
    let v = TtsVerifier::builder().build().unwrap();
    let reference = rich_speech(24000, 0.5);
    let err = v.verify_with_reference(&[], &reference).unwrap_err();
    assert!(matches!(err, TtsVerifyError::EmptyInput));
}

#[test]
fn test_verify_with_reference_reject_policy_on_clipped() {
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Reject,
        ..Default::default()
    };
    let v = TtsVerifier::builder().hard_bounds(hb).build().unwrap();

    // Use rich_speech amplified to clip, so hard bounds fail but DSP succeeds.
    let mut signal = rich_speech(24000, 0.5);
    for s in &mut signal {
        *s *= 5.0;
    }
    let err = v.verify_with_reference(&signal, &signal).unwrap_err();
    match err {
        TtsVerifyError::VerificationRejected { cert } => {
            assert!(!cert.overall_passed);
        }
        other => panic!("expected VerificationRejected, got: {other:?}"),
    }
}

#[test]
fn test_verify_with_reference_warn_policy_on_clipped() {
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Warn,
        ..Default::default()
    };
    let v = TtsVerifier::builder().hard_bounds(hb).build().unwrap();

    // Use rich_speech amplified to clip, so hard bounds fail but DSP succeeds.
    let mut signal = rich_speech(24000, 0.5);
    for s in &mut signal {
        *s *= 5.0;
    }
    // Warn returns Ok even for failing hard bounds.
    let cert = v.verify_with_reference(&signal, &signal).unwrap();
    // Individual hard bounds still reflect truth (clipping fails).
    assert!(!cert.passes_hard_bounds());
    let clip_check = cert
        .hard_bounds
        .iter()
        .find(|b| b.name == "no_clipping")
        .expect("should have clipping check");
    assert!(
        !clip_check.passed,
        "amplified signal should fail clipping check"
    );
}

// ===========================================================================
// Certificate methods
// ===========================================================================

#[test]
fn test_certificate_passes_hard_bounds_all_pass() {
    let cert = Certificate {
        hard_bounds: vec![
            HardBound {
                name: "a",
                passed: true,
                value: 0.5,
                threshold: 0.1,
            },
            HardBound {
                name: "b",
                passed: true,
                value: 0.8,
                threshold: 0.5,
            },
        ],
        quality_metrics: vec![],
        phoneme_results: None,
        overall_passed: true,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };
    assert!(cert.passes_hard_bounds());
}

#[test]
fn test_certificate_passes_hard_bounds_one_fails() {
    let cert = Certificate {
        hard_bounds: vec![
            HardBound {
                name: "ok",
                passed: true,
                value: 0.5,
                threshold: 0.1,
            },
            HardBound {
                name: "fail",
                passed: false,
                value: 0.01,
                threshold: 0.1,
            },
        ],
        quality_metrics: vec![],
        phoneme_results: None,
        overall_passed: false,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };
    assert!(!cert.passes_hard_bounds());
}

#[test]
fn test_certificate_passes_hard_bounds_empty() {
    let cert = Certificate {
        hard_bounds: vec![],
        quality_metrics: vec![],
        phoneme_results: None,
        overall_passed: true,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };
    // Vacuously true when no hard bounds.
    assert!(cert.passes_hard_bounds());
}

#[test]
fn test_certificate_passes_quality_all_pass() {
    let cert = Certificate {
        hard_bounds: vec![],
        quality_metrics: vec![
            QualityMetric {
                name: "m1",
                value: 5.0,
                threshold: 6.0,
                passed: true,
                citation: "test",
            },
            QualityMetric {
                name: "m2",
                value: 20.0,
                threshold: 15.0,
                passed: true,
                citation: "test",
            },
        ],
        phoneme_results: None,
        overall_passed: true,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };
    assert!(cert.passes_quality());
}

#[test]
fn test_certificate_report_overall_failed() {
    let cert = Certificate {
        hard_bounds: vec![HardBound {
            name: "silence",
            passed: false,
            value: 0.001,
            threshold: 0.01,
        }],
        quality_metrics: vec![],
        phoneme_results: None,
        overall_passed: false,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };
    let report = cert.report();
    assert!(report.contains("FAILED"), "report should say FAILED");
    assert!(report.contains("FAIL"), "report should show FAIL status");
    assert!(report.contains("silence"), "report should name the check");
}

#[test]
fn test_certificate_report_overall_passed() {
    let cert = Certificate {
        hard_bounds: vec![HardBound {
            name: "clipping",
            passed: true,
            value: 0.5,
            threshold: 1.0,
        }],
        quality_metrics: vec![],
        phoneme_results: None,
        overall_passed: true,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };
    let report = cert.report();
    assert!(report.contains("PASSED"), "report should say PASSED");
}

#[test]
fn test_certificate_report_with_quality_section() {
    let cert = Certificate {
        hard_bounds: vec![],
        quality_metrics: vec![QualityMetric {
            name: "mcd",
            value: 3.5,
            threshold: 6.0,
            passed: true,
            citation: "Kubichek 1993",
        }],
        phoneme_results: None,
        overall_passed: true,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };
    let report = cert.report();
    assert!(
        report.contains("Quality Metrics"),
        "report should have quality section"
    );
    assert!(report.contains("mcd"), "report should mention metric name");
    assert!(
        report.contains("Kubichek 1993"),
        "report should show citation"
    );
}

// ===========================================================================
// HardBoundsConfig validation and effective thresholds
// ===========================================================================

#[test]
fn test_hard_bounds_config_inverted_duration_range() {
    let cfg = HardBoundsConfig {
        min_duration_sec: 10.0,
        max_duration_sec: 5.0,
        ..Default::default()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, TtsVerifyError::InvalidConfig(_)));
}

#[test]
fn test_hard_bounds_config_negative_min_rms() {
    let cfg = HardBoundsConfig {
        min_rms: -0.01,
        ..Default::default()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, TtsVerifyError::InvalidConfig(_)));
}

#[test]
fn test_hard_bounds_config_zero_tail_ms() {
    let cfg = HardBoundsConfig {
        tail_ms: 0.0,
        ..Default::default()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, TtsVerifyError::InvalidConfig(_)));
}

// ===========================================================================
// CheckOverrides
// ===========================================================================

#[test]
fn test_check_overrides_default_all_none() {
    let o = CheckOverrides::new();
    assert!(o.min_rms.is_none());
    assert!(o.max_amplitude.is_none());
    assert!(o.max_dc_offset.is_none());
    assert!(o.max_click_diff.is_none());
    assert!(o.min_duration_sec.is_none());
    assert!(o.max_duration_sec.is_none());
    assert!(o.max_tail_energy_ratio.is_none());
    o.validate().expect("default overrides should be valid");
}

#[test]
fn test_check_overrides_nan_min_rms_rejected() {
    let o = CheckOverrides {
        min_rms: Some(f64::NAN),
        ..Default::default()
    };
    assert!(o.validate().is_err());
}

#[test]
fn test_check_overrides_inverted_duration_rejected() {
    let o = CheckOverrides {
        min_duration_sec: Some(10.0),
        max_duration_sec: Some(5.0),
        ..Default::default()
    };
    assert!(o.validate().is_err());
}

#[test]
fn test_check_overrides_negative_max_amplitude_rejected() {
    let o = CheckOverrides {
        max_amplitude: Some(-1.0),
        ..Default::default()
    };
    assert!(o.validate().is_err());
}

#[test]
fn test_effective_thresholds_with_overrides() {
    let hb = HardBoundsConfig {
        min_rms: 0.01,
        max_amplitude: 1.0,
        max_dc_offset: 0.05,
        max_click_diff: 0.5,
        min_duration_sec: 0.1,
        max_duration_sec: 300.0,
        max_tail_energy_ratio: 3.0,
        overrides: CheckOverrides {
            min_rms: Some(0.005),
            max_amplitude: Some(0.9),
            max_dc_offset: Some(0.02),
            max_click_diff: Some(0.3),
            min_duration_sec: Some(0.05),
            max_duration_sec: Some(600.0),
            max_tail_energy_ratio: Some(2.0),
        },
        ..Default::default()
    };
    assert!((hb.effective_min_rms() - 0.005).abs() < 1e-10);
    assert!((hb.effective_max_amplitude() - 0.9).abs() < 1e-10);
    assert!((hb.effective_max_dc_offset() - 0.02).abs() < 1e-10);
    assert!((hb.effective_max_click_diff() - 0.3).abs() < 1e-10);
    assert!((hb.effective_min_duration_sec() - 0.05).abs() < 1e-10);
    assert!((hb.effective_max_duration_sec() - 600.0).abs() < 1e-10);
    assert!((hb.effective_max_tail_energy_ratio() - 2.0).abs() < 1e-10);
}

#[test]
fn test_effective_thresholds_without_overrides() {
    let hb = HardBoundsConfig::default();
    assert!((hb.effective_min_rms() - 0.01).abs() < 1e-10);
    assert!((hb.effective_max_amplitude() - 1.0).abs() < 1e-10);
    assert!((hb.effective_max_dc_offset() - 0.05).abs() < 1e-10);
    assert!((hb.effective_max_click_diff() - 0.5).abs() < 1e-10);
    assert!((hb.effective_min_duration_sec() - 0.1).abs() < 1e-10);
    assert!((hb.effective_max_duration_sec() - 300.0).abs() < 1e-10);
    assert!((hb.effective_max_tail_energy_ratio() - 3.0).abs() < 1e-10);
}

// ===========================================================================
// RejectionPolicy
// ===========================================================================

#[test]
fn test_rejection_policy_default_is_reject() {
    assert_eq!(RejectionPolicy::default(), RejectionPolicy::Reject);
}

#[test]
fn test_rejection_policy_equality() {
    assert_eq!(RejectionPolicy::Warn, RejectionPolicy::Warn);
    assert_ne!(RejectionPolicy::Reject, RejectionPolicy::Warn);
    assert_ne!(RejectionPolicy::Warn, RejectionPolicy::Remediate);
    assert_ne!(RejectionPolicy::Reject, RejectionPolicy::Remediate);
}

// ===========================================================================
// QualityConfig validation
// ===========================================================================

#[test]
fn test_quality_config_inverted_f0_range() {
    let cfg = QualityConfig {
        f0_range: (400.0, 80.0), // inverted
        ..Default::default()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, TtsVerifyError::InvalidConfig(_)));
}

#[test]
fn test_quality_config_nan_spectral_tilt() {
    let cfg = QualityConfig {
        spectral_tilt: (f64::NAN, -3.0),
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_quality_config_nan_optional_stoi() {
    let cfg = QualityConfig {
        min_stoi: Some(f64::NAN),
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_quality_config_nan_optional_pesq() {
    let cfg = QualityConfig {
        min_pesq: Some(f64::INFINITY),
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

// ===========================================================================
// Bound check functions — additional coverage
// ===========================================================================

#[test]
fn test_check_non_silence_very_quiet_audio() {
    // Audio with very low amplitude — should fail with default 0.01 threshold.
    let audio: Vec<f32> = (0..2400).map(|i| 0.001 * (i as f32 * 0.1).sin()).collect();
    let result = check_non_silence(&audio, 0.01);
    assert!(
        !result.passed,
        "very quiet audio should fail non-silence check"
    );
    assert!(result.value < 0.01);
}

#[test]
fn test_check_no_clipping_boundary_value() {
    // Samples at exactly the threshold should pass.
    let audio = vec![1.0_f32, -1.0, 0.5, -0.5];
    let result = check_no_clipping(&audio, 1.0);
    assert!(
        result.passed,
        "samples at exactly 1.0 should pass clipping at 1.0"
    );
}

#[test]
fn test_check_no_dc_offset_symmetric_square_wave() {
    // A symmetric square wave has zero DC offset.
    let audio: Vec<f32> = (0..2400)
        .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
        .collect();
    let result = check_no_dc_offset(&audio, 0.01);
    assert!(
        result.passed,
        "symmetric square wave should have zero DC offset"
    );
    assert!(result.value < 0.001);
}

#[test]
fn test_check_duration_exact_boundary() {
    // Duration exactly at min_sec should pass.
    let sr = 24000_u32;
    let n = (f64::from(sr) * 0.5) as usize; // exactly 0.5 sec
    let audio = vec![0.1_f32; n];
    let result = check_duration(&audio, sr, 0.5, 2.0);
    assert!(result.passed, "duration at exactly min_sec should pass");
}

#[test]
fn test_check_spectral_coverage_pure_tone_narrow() {
    // A single pure tone should cover only 1 band — likely fails 50% coverage.
    let audio = sine_wave_full(1000.0, 24000, 1.0, 0.5);
    let config = SpectralCoverageConfig::default();
    let result = check_spectral_coverage(&audio, 24000, &config).unwrap();
    // A single tone may or may not pass depending on band width, but coverage should be low.
    assert!(
        result.value < 0.5,
        "single tone should have low spectral coverage: {:.3}",
        result.value
    );
}

#[test]
fn test_check_tail_energy_very_short_signal() {
    // 5 samples — shorter than any tail/body window.
    let audio = vec![0.5_f32; 5];
    let result = check_tail_energy(&audio, 24000, 50.0, 500.0, 3.0);
    // Should not panic. Tail covers entire signal.
    assert_eq!(result.name, "tail_energy");
}

// ===========================================================================
// HardwareCostModel
// ===========================================================================

#[test]
fn test_hardware_cost_model_conservative_is_slower() {
    let standard = HardwareCostModel::m4_max();
    let conservative = HardwareCostModel::m4_max_conservative();
    // Conservative model should give longer time estimates.
    let flops = 1_000_000_000u64;
    let bytes = 10_000_000u64;
    let t_std = standard.estimate_time_us(flops, bytes);
    let t_con = conservative.estimate_time_us(flops, bytes);
    assert!(
        t_con > t_std,
        "conservative estimate ({t_con}) should exceed standard ({t_std})"
    );
}

#[test]
fn test_hardware_cost_model_zero_flops_zero_bytes() {
    let model = HardwareCostModel::m4_max();
    let time = model.estimate_time_us(0, 0);
    // Should equal exactly the dispatch overhead.
    assert!((time - model.dispatch_overhead_us).abs() < 1e-10);
}

// ===========================================================================
// Deterministic hash integration
// ===========================================================================

#[test]
fn test_verify_produces_deterministic_hash() {
    let v = TtsVerifier::builder().build().unwrap();
    let signal = rich_speech(24000, 0.5);
    let cert = v.verify(&signal).unwrap();
    assert!(
        cert.deterministic_hash.is_some(),
        "verify should produce a hash"
    );
    let hash = cert.deterministic_hash.as_ref().unwrap();
    assert_eq!(hash.len(), 64, "SHA-256 hex digest should be 64 chars");
    // Verify it matches the standalone function.
    assert_eq!(hash, &deterministic::pcm_sha256(&signal));
}

#[test]
fn test_verify_different_signals_different_hashes() {
    // Use Warn policy so spectral coverage failures don't cause VerificationRejected.
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Warn,
        ..Default::default()
    };
    let v = TtsVerifier::builder().hard_bounds(hb).build().unwrap();
    let sig_a = rich_speech(24000, 0.5);
    let sig_b = sine_wave_full(440.0, 24000, 0.5, 0.3);
    let cert_a = v.verify(&sig_a).unwrap();
    let cert_b = v.verify(&sig_b).unwrap();
    assert_ne!(
        cert_a.deterministic_hash, cert_b.deterministic_hash,
        "different signals should produce different hashes"
    );
}

// ===========================================================================
// Verifier with overrides modifying behavior
// ===========================================================================

#[test]
fn test_verify_override_loosens_silence_threshold() {
    // Default min_rms is 0.01. Very quiet audio fails normally.
    // Use Warn policy so the VerificationRejected error does not block us.
    let quiet: Vec<f32> = (0..24000).map(|i| 0.005 * (i as f32 * 0.1).sin()).collect();
    let hb_strict = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Warn,
        ..Default::default()
    };
    let v_strict = TtsVerifier::builder()
        .hard_bounds(hb_strict)
        .build()
        .unwrap();
    let cert_strict = v_strict.verify(&quiet).unwrap();
    let silence_strict = cert_strict
        .hard_bounds
        .iter()
        .find(|b| b.name == "non_silence")
        .expect("should have non_silence check");
    assert!(
        !silence_strict.passed,
        "quiet audio should fail non_silence with default threshold"
    );

    // Override to loosen the threshold.
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Warn,
        overrides: CheckOverrides {
            min_rms: Some(0.001),
            ..Default::default()
        },
        ..Default::default()
    };
    let v_loose = TtsVerifier::builder().hard_bounds(hb).build().unwrap();
    let cert_loose = v_loose.verify(&quiet).unwrap();
    let silence_check = cert_loose
        .hard_bounds
        .iter()
        .find(|b| b.name == "non_silence")
        .expect("should have non_silence check");
    assert!(
        silence_check.passed,
        "quiet audio should pass with loosened threshold"
    );
}

// ===========================================================================
// SpectralCoverageConfig validation
// ===========================================================================

#[test]
fn test_spectral_coverage_config_zero_bands_rejected() {
    let cfg = SpectralCoverageConfig {
        n_bands: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_spectral_coverage_config_nan_min_energy_rejected() {
    let cfg = SpectralCoverageConfig {
        min_energy_db: f64::NAN,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}
