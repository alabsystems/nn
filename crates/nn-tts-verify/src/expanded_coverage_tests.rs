// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Expanded test coverage for nn-tts-verify.
//!
//! Covers: HardBoundsConfig construction and validation, RejectionPolicy
//! behavior, individual bound checks, TtsVerifier pipeline, cost model
//! estimates, audio quality metrics, deterministic hashing, and edge cases
//! (silent audio, clipped audio, very short audio).
//!
//! Part of #4302.

use crate::bounds::{
    check_duration, check_no_clicks, check_no_clipping, check_no_dc_offset, check_non_silence,
    check_nyquist, check_spectral_coverage, check_tail_energy, HardBound, SpectralCoverageConfig,
};
use crate::certificate::Certificate;
use crate::config::{CheckOverrides, HardBoundsConfig, QualityConfig, RejectionPolicy};
use crate::cost_model::HardwareCostModel;
use crate::deterministic::{self, DeterministicCert, DeterministicMeta};
use crate::error::TtsVerifyError;
use crate::quality;
use crate::stats::{cohens_d, holm_bonferroni, welch_t_test};
use crate::test_audio_helpers::{sine_wave_full, sine_wave_samples};
use crate::verifier::TtsVerifier;

// ===========================================================================
// Helpers
// ===========================================================================

/// Rich harmonic signal that passes all hard bounds including spectral coverage.
fn rich_signal(sample_rate: u32, duration_sec: f64) -> Vec<f32> {
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

/// Build a verifier with Warn policy for tests that inspect certificate content.
fn warn_verifier(sample_rate: u32) -> TtsVerifier {
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Warn,
        ..Default::default()
    };
    TtsVerifier::builder()
        .sample_rate(sample_rate)
        .hard_bounds(hb)
        .build()
        .unwrap()
}

// ===========================================================================
// HardBoundsConfig construction and defaults
// ===========================================================================

#[test]
fn test_hard_bounds_config_default_values() {
    let cfg = HardBoundsConfig::default();
    assert!((cfg.min_rms - 0.01).abs() < f64::EPSILON);
    assert!((cfg.max_amplitude - 1.0).abs() < f64::EPSILON);
    assert!((cfg.max_dc_offset - 0.05).abs() < f64::EPSILON);
    assert!((cfg.max_click_diff - 0.5).abs() < f64::EPSILON);
    assert!((cfg.min_duration_sec - 0.1).abs() < f64::EPSILON);
    assert!((cfg.max_duration_sec - 300.0).abs() < f64::EPSILON);
    assert!((cfg.tail_ms - 50.0).abs() < f64::EPSILON);
    assert!((cfg.body_ms - 500.0).abs() < f64::EPSILON);
    assert!((cfg.max_tail_energy_ratio - 3.0).abs() < f64::EPSILON);
    assert_eq!(cfg.rejection_policy, RejectionPolicy::Reject);
}

#[test]
fn test_hard_bounds_config_custom_fields_validate() {
    let cfg = HardBoundsConfig {
        min_rms: 0.005,
        max_amplitude: 0.95,
        max_dc_offset: 0.03,
        max_click_diff: 0.3,
        min_duration_sec: 0.05,
        max_duration_sec: 600.0,
        tail_ms: 100.0,
        body_ms: 1000.0,
        max_tail_energy_ratio: 5.0,
        spectral: SpectralCoverageConfig::default(),
        rejection_policy: RejectionPolicy::Warn,
        overrides: CheckOverrides::default(),
    };
    cfg.validate().expect("custom config should be valid");
}

#[test]
fn test_hard_bounds_config_validate_nan_min_rms() {
    let cfg = HardBoundsConfig {
        min_rms: f64::NAN,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_hard_bounds_config_validate_zero_body_ms() {
    let cfg = HardBoundsConfig {
        body_ms: 0.0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_hard_bounds_config_validate_negative_max_click_diff() {
    let cfg = HardBoundsConfig {
        max_click_diff: -0.1,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_hard_bounds_config_validate_inf_max_tail_energy_ratio() {
    let cfg = HardBoundsConfig {
        max_tail_energy_ratio: f64::INFINITY,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_hard_bounds_config_validate_equal_duration_range() {
    let cfg = HardBoundsConfig {
        min_duration_sec: 5.0,
        max_duration_sec: 5.0,
        ..Default::default()
    };
    assert!(
        cfg.validate().is_err(),
        "equal min/max duration should fail"
    );
}

// ===========================================================================
// CheckOverrides validation edge cases
// ===========================================================================

#[test]
fn test_check_overrides_inf_max_click_diff_rejected() {
    let o = CheckOverrides {
        max_click_diff: Some(f64::INFINITY),
        ..Default::default()
    };
    assert!(o.validate().is_err());
}

#[test]
fn test_check_overrides_nan_max_dc_offset_rejected() {
    let o = CheckOverrides {
        max_dc_offset: Some(f64::NAN),
        ..Default::default()
    };
    assert!(o.validate().is_err());
}

#[test]
fn test_check_overrides_zero_max_amplitude_rejected() {
    let o = CheckOverrides {
        max_amplitude: Some(0.0),
        ..Default::default()
    };
    assert!(o.validate().is_err());
}

#[test]
fn test_check_overrides_valid_partial_overrides() {
    let o = CheckOverrides {
        min_rms: Some(0.005),
        max_duration_sec: Some(600.0),
        ..Default::default()
    };
    o.validate().expect("partial overrides should be valid");
}

#[test]
fn test_check_overrides_equal_duration_range_rejected() {
    let o = CheckOverrides {
        min_duration_sec: Some(5.0),
        max_duration_sec: Some(5.0),
        ..Default::default()
    };
    assert!(o.validate().is_err(), "equal min/max override should fail");
}

// ===========================================================================
// RejectionPolicy behavior
// ===========================================================================

#[test]
fn test_rejection_policy_default_is_reject() {
    assert_eq!(RejectionPolicy::default(), RejectionPolicy::Reject);
}

#[test]
fn test_rejection_policy_debug_output() {
    let debug_str = format!("{:?}", RejectionPolicy::Warn);
    assert!(debug_str.contains("Warn"));
}

#[test]
fn test_rejection_policy_clone() {
    let p = RejectionPolicy::Remediate;
    let p2 = p;
    assert_eq!(p, p2);
}

#[test]
fn test_reject_policy_silent_audio_returns_err() {
    let signal = vec![0.0_f32; 24000];
    let v = TtsVerifier::builder().build().unwrap();
    let result = v.verify(&signal);
    assert!(result.is_err());
    match result.unwrap_err() {
        TtsVerifyError::VerificationRejected { cert } => {
            assert!(!cert.overall_passed);
            let silence_check = cert
                .hard_bounds
                .iter()
                .find(|b| b.name == "non_silence")
                .unwrap();
            assert!(!silence_check.passed);
        }
        other => panic!("expected VerificationRejected, got: {other:?}"),
    }
}

#[test]
fn test_warn_policy_silent_audio_returns_ok_with_failures_visible() {
    let signal = vec![0.0_f32; 24000];
    let v = warn_verifier(24000);
    let cert = v.verify(&signal).unwrap();
    // Warn policy: overall_passed is true (hard bound failures masked).
    assert!(cert.overall_passed);
    // Individual hard bounds still show failure.
    assert!(!cert.passes_hard_bounds());
}

#[test]
fn test_remediate_policy_silent_audio_returns_ok() {
    let signal = vec![0.0_f32; 24000];
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Remediate,
        ..Default::default()
    };
    let v = TtsVerifier::builder().hard_bounds(hb).build().unwrap();
    let cert = v.verify(&signal).unwrap();
    assert!(
        cert.overall_passed,
        "Remediate should mask hard bound failures"
    );
}

#[test]
fn test_reject_policy_good_audio_returns_ok() {
    let signal = rich_signal(24000, 0.5);
    let v = TtsVerifier::builder().build().unwrap();
    let cert = v.verify(&signal).unwrap();
    assert!(cert.overall_passed);
    assert!(cert.passes_hard_bounds());
}

// ===========================================================================
// Individual bound checks - edge cases
// ===========================================================================

#[test]
fn test_non_silence_single_sample_above_threshold() {
    let audio = vec![0.5_f32];
    let result = check_non_silence(&audio, 0.01);
    assert!(result.passed, "single loud sample should pass");
    assert!((result.value - 0.5).abs() < 1e-6);
}

#[test]
fn test_non_silence_single_sample_below_threshold() {
    let audio = vec![0.001_f32];
    let result = check_non_silence(&audio, 0.01);
    assert!(!result.passed, "single quiet sample should fail");
}

#[test]
fn test_no_clipping_all_zeros() {
    let audio = vec![0.0_f32; 1000];
    let result = check_no_clipping(&audio, 1.0);
    assert!(result.passed, "zeros should not clip");
    assert!((result.value - 0.0).abs() < 1e-10);
}

#[test]
fn test_no_clipping_negative_peak() {
    let audio = vec![-1.5_f32, 0.0, 0.5];
    let result = check_no_clipping(&audio, 1.0);
    assert!(
        !result.passed,
        "negative peak at -1.5 should fail clipping check"
    );
    assert!((result.value - 1.5).abs() < 1e-6);
}

#[test]
fn test_dc_offset_all_positive() {
    let audio = vec![0.3_f32; 1000];
    let result = check_no_dc_offset(&audio, 0.1);
    assert!(
        !result.passed,
        "all-positive samples should have large DC offset"
    );
    assert!((result.value - 0.3).abs() < 1e-6);
}

#[test]
fn test_dc_offset_balanced_signal() {
    // Symmetric signal: DC offset near zero.
    let audio: Vec<f32> = (0..2400)
        .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
        .collect();
    let result = check_no_dc_offset(&audio, 0.01);
    assert!(result.passed, "balanced signal should have zero DC offset");
}

#[test]
fn test_clicks_smooth_ramp() {
    let audio: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
    let result = check_no_clicks(&audio, 0.01);
    // Max diff for a smooth ramp = 0.001.
    assert!(result.passed, "smooth ramp should have small diffs");
    assert!(result.value < 0.002);
}

#[test]
fn test_clicks_large_discontinuity() {
    let mut audio = vec![0.0_f32; 1000];
    audio[500] = 1.0;
    audio[501] = -1.0;
    let result = check_no_clicks(&audio, 0.5);
    assert!(!result.passed, "2.0 jump should exceed 0.5 threshold");
    assert!(result.value >= 2.0);
}

#[test]
fn test_duration_exact_at_max_boundary() {
    let sr = 24000_u32;
    let n = (f64::from(sr) * 2.0) as usize; // exactly 2.0 sec
    let audio = vec![0.1_f32; n];
    let result = check_duration(&audio, sr, 0.5, 2.0);
    assert!(result.passed, "duration at exactly max_sec should pass");
}

#[test]
fn test_duration_empty_audio() {
    let result = check_duration(&[], 24000, 0.1, 300.0);
    assert!(!result.passed, "empty audio should have zero duration");
    assert!((result.value - 0.0).abs() < 1e-10);
}

#[test]
fn test_nyquist_low_frequency_passes() {
    let audio = sine_wave_full(200.0, 24000, 1.0, 0.5);
    let result = check_nyquist(&audio, 24000).unwrap();
    assert!(result.passed, "200 Hz sine should have low nyquist energy");
    assert!(result.value < 0.05);
}

#[test]
fn test_spectral_coverage_single_band() {
    let config = SpectralCoverageConfig {
        n_bands: 1,
        min_energy_db: -80.0,
        min_coverage: 0.5,
    };
    let audio = sine_wave_full(440.0, 24000, 0.5, 0.5);
    let result = check_spectral_coverage(&audio, 24000, &config).unwrap();
    // With 1 band, coverage is either 0% or 100%.
    assert!(result.value == 0.0 || result.value == 1.0);
}

#[test]
fn test_tail_energy_all_zeros() {
    let audio = vec![0.0_f32; 24000];
    let result = check_tail_energy(&audio, 24000, 50.0, 500.0, 3.0);
    assert!(result.passed, "all-zero audio should have ratio 0.0");
    assert!(result.value < 1e-10);
}

#[test]
fn test_tail_energy_single_sample() {
    let audio = vec![0.5_f32];
    let result = check_tail_energy(&audio, 24000, 50.0, 500.0, 3.0);
    assert_eq!(result.name, "tail_energy");
    // Should not panic on single sample.
}

// ===========================================================================
// TtsVerifier pipeline
// ===========================================================================

#[test]
fn test_verifier_produces_all_8_hard_bounds() {
    let signal = rich_signal(24000, 0.5);
    let v = TtsVerifier::builder().build().unwrap();
    let cert = v.verify(&signal).unwrap();
    assert_eq!(
        cert.hard_bounds.len(),
        8,
        "should have exactly 8 hard bounds"
    );
    let names: Vec<&str> = cert.hard_bounds.iter().map(|b| b.name).collect();
    assert!(names.contains(&"non_silence"));
    assert!(names.contains(&"no_clipping"));
    assert!(names.contains(&"no_dc_offset"));
    assert!(names.contains(&"no_clicks"));
    assert!(names.contains(&"duration"));
    assert!(names.contains(&"tail_energy"));
    assert!(names.contains(&"spectral_coverage"));
    assert!(names.contains(&"nyquist"));
}

#[test]
fn test_verifier_no_quality_metrics_by_default() {
    let signal = rich_signal(24000, 0.5);
    let v = TtsVerifier::builder().build().unwrap();
    let cert = v.verify(&signal).unwrap();
    assert!(cert.quality_metrics.is_empty());
    assert!(cert.phoneme_results.is_none());
}

#[test]
fn test_verifier_with_quality_produces_3_metrics() {
    let signal = rich_signal(24000, 0.5);
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Warn,
        ..Default::default()
    };
    let v = TtsVerifier::builder()
        .hard_bounds(hb)
        .with_quality()
        .build()
        .unwrap();
    let cert = v.verify(&signal).unwrap();
    assert!(
        cert.quality_metrics.len() >= 3,
        "quality should produce HNR, F0, spectral tilt"
    );
    let names: Vec<&str> = cert.quality_metrics.iter().map(|m| m.name).collect();
    assert!(names.contains(&"hnr"));
    assert!(names.contains(&"f0_range"));
    assert!(names.contains(&"spectral_tilt"));
}

#[test]
fn test_verifier_verify_with_reference_produces_mcd() {
    let signal = rich_signal(24000, 0.5);
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Warn,
        ..Default::default()
    };
    let v = TtsVerifier::builder().hard_bounds(hb).build().unwrap();
    let cert = v.verify_with_reference(&signal, &signal).unwrap();
    let has_mcd = cert.quality_metrics.iter().any(|m| m.name == "mcd");
    assert!(has_mcd, "verify_with_reference should produce MCD metric");
}

#[test]
fn test_verifier_verify_with_reference_cosine_similarity() {
    let signal = rich_signal(24000, 0.5);
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Warn,
        ..Default::default()
    };
    let v = TtsVerifier::builder().hard_bounds(hb).build().unwrap();
    let cert = v.verify_with_reference(&signal, &signal).unwrap();
    let cos_sim = cert
        .quality_metrics
        .iter()
        .find(|m| m.name == "cosine_similarity")
        .expect("should have cosine_similarity");
    assert!(
        (cos_sim.value - 1.0).abs() < 1e-6,
        "identical signals should have cosine similarity 1.0, got {}",
        cos_sim.value
    );
    assert!(cos_sim.passed);
}

#[test]
fn test_verifier_verify_with_reference_snr_infinite_for_identical() {
    let signal = rich_signal(24000, 0.5);
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Warn,
        ..Default::default()
    };
    let v = TtsVerifier::builder().hard_bounds(hb).build().unwrap();
    let cert = v.verify_with_reference(&signal, &signal).unwrap();
    let snr = cert
        .quality_metrics
        .iter()
        .find(|m| m.name == "snr")
        .expect("should have snr");
    // SNR of identical signals is infinite.
    assert!(snr.value.is_infinite() || snr.value > 100.0);
    assert!(snr.passed);
}

#[test]
fn test_verifier_at_16khz_sample_rate() {
    let signal = rich_signal(16000, 0.5);
    let v = TtsVerifier::builder().sample_rate(16000).build().unwrap();
    let cert = v.verify(&signal).unwrap();
    assert!(cert.overall_passed);
}

#[test]
fn test_verifier_at_44100_sample_rate() {
    let signal = rich_signal(44100, 0.5);
    let v = TtsVerifier::builder().sample_rate(44100).build().unwrap();
    let cert = v.verify(&signal).unwrap();
    assert!(cert.overall_passed);
}

// ===========================================================================
// Cost model estimates
// ===========================================================================

#[test]
fn test_hardware_cost_model_m4_max_defaults() {
    let model = HardwareCostModel::m4_max();
    assert!(model.peak_tflops_f32 > 0.0);
    assert!(model.peak_bandwidth_gbs > 0.0);
    assert!(model.dispatch_overhead_us > 0.0);
    model.validate().unwrap();
}

#[test]
fn test_hardware_cost_model_m4_max_conservative_defaults() {
    let model = HardwareCostModel::m4_max_conservative();
    model.validate().unwrap();
    let standard = HardwareCostModel::m4_max();
    // Conservative should have lower peak throughput → higher time estimates.
    assert!(
        model.peak_tflops_f32 < standard.peak_tflops_f32
            || model.peak_bandwidth_gbs < standard.peak_bandwidth_gbs,
        "conservative should have lower peak throughput"
    );
}

#[test]
fn test_hardware_cost_model_estimate_time_compute_bound() {
    let model = HardwareCostModel::m4_max();
    // 1 TFLOP at ~56 TFLOPS → ~17.8 ms + overhead
    let time = model.estimate_time_us(1_000_000_000_000, 0);
    assert!(time > model.dispatch_overhead_us);
    assert!(time < 1_000_000.0, "1 TFLOP should take < 1s");
}

#[test]
fn test_hardware_cost_model_estimate_time_memory_bound() {
    let model = HardwareCostModel::m4_max();
    // 1 GB at ~400 GB/s → ~2.5ms + overhead
    let time = model.estimate_time_us(0, 1_000_000_000);
    assert!(time > model.dispatch_overhead_us);
    assert!(time < 100_000.0, "1 GB transfer should take < 100ms");
}

#[test]
fn test_hardware_cost_model_validate_nan() {
    let mut model = HardwareCostModel::m4_max();
    model.peak_tflops_f32 = f64::NAN;
    assert!(model.validate().is_err());
}

#[test]
fn test_hardware_cost_model_validate_zero_tflops() {
    let mut model = HardwareCostModel::m4_max();
    model.peak_tflops_f32 = 0.0;
    assert!(model.validate().is_err());
}

// ===========================================================================
// Audio quality metrics
// ===========================================================================

#[test]
fn test_compute_hnr_sine_wave() {
    // A pure sine wave is perfectly periodic → high HNR.
    let audio = sine_wave_full(440.0, 24000, 0.5, 0.5);
    let result = quality::compute_hnr(&audio, 24000, 10.0).unwrap();
    assert_eq!(result.name, "hnr");
    assert!(result.value > 0.0, "sine wave should have positive HNR");
}

#[test]
fn test_compute_hnr_noise() {
    // Pseudo-noise should have low HNR.
    let audio: Vec<f32> = (0..12000)
        .map(|i| {
            
            (i as f32 * 0.73).sin() * 0.3
                + (i as f32 * 1.37).cos() * 0.2
                + (i as f32 * 2.71).sin() * 0.15
        })
        .collect();
    let result = quality::compute_hnr(&audio, 24000, 10.0).unwrap();
    assert_eq!(result.name, "hnr");
}

#[test]
fn test_check_f0_range_all_in_range() {
    let f0 = vec![100.0, 150.0, 200.0, 250.0, 300.0];
    let result = quality::check_f0_range(&f0, 80.0, 400.0);
    assert_eq!(result.name, "f0_range");
    assert!(
        (result.value - 1.0).abs() < 1e-10,
        "all in range should be 100%"
    );
    assert!(result.passed);
}

#[test]
fn test_check_f0_range_none_in_range() {
    let f0 = vec![50.0, 60.0, 500.0, 600.0]; // all outside 80-400 Hz
    let result = quality::check_f0_range(&f0, 80.0, 400.0);
    assert!(
        (result.value - 0.0).abs() < 1e-10,
        "none in range should be 0%"
    );
    assert!(!result.passed);
}

#[test]
fn test_check_f0_range_empty_contour() {
    let result = quality::check_f0_range(&[], 80.0, 400.0);
    assert!(!result.passed, "empty contour should fail");
}

#[test]
fn test_check_f0_range_all_unvoiced() {
    let f0 = vec![0.0, 0.0, 0.0]; // all unvoiced
    let result = quality::check_f0_range(&f0, 80.0, 400.0);
    assert!(!result.passed, "all unvoiced should fail");
}

#[test]
fn test_compute_mcd_identical_signals() {
    let audio = sine_wave_full(440.0, 24000, 0.5, 0.5);
    let result = quality::compute_mcd(&audio, &audio, 24000, 6.0).unwrap();
    assert_eq!(result.name, "mcd");
    assert!(
        result.value < 0.01,
        "MCD of identical signals should be near 0"
    );
    assert!(result.passed);
}

#[test]
fn test_compute_mcd_length_mismatch() {
    let a = sine_wave_full(440.0, 24000, 0.5, 0.5);
    let b = sine_wave_full(440.0, 24000, 1.0, 0.5);
    let err = quality::compute_mcd(&a, &b, 24000, 6.0).unwrap_err();
    assert!(matches!(err, TtsVerifyError::LengthMismatch { .. }));
}

#[test]
fn test_compute_mcd_empty_input() {
    let err = quality::compute_mcd(&[], &[], 24000, 6.0).unwrap_err();
    assert!(matches!(err, TtsVerifyError::EmptyInput));
}

#[test]
fn test_compute_rms_sine_wave() {
    let audio = sine_wave_full(440.0, 24000, 0.5, 0.5);
    let result = quality::compute_rms(&audio, 0.1).unwrap();
    assert_eq!(result.name, "rms_energy");
    // RMS of a sine wave with amplitude A is A/sqrt(2) ≈ 0.354
    assert!((result.value - 0.354).abs() < 0.02);
    assert!(result.passed);
}

#[test]
fn test_compute_rms_empty_input() {
    let err = quality::compute_rms(&[], 0.01).unwrap_err();
    assert!(matches!(err, TtsVerifyError::EmptyInput));
}

#[test]
fn test_compute_cosine_similarity_identical() {
    let audio = sine_wave_full(440.0, 24000, 0.5, 0.5);
    let result = quality::compute_cosine_similarity(&audio, &audio, 0.85).unwrap();
    assert!((result.value - 1.0).abs() < 1e-6);
    assert!(result.passed);
}

#[test]
fn test_compute_cosine_similarity_negated() {
    let audio = sine_wave_full(440.0, 24000, 0.5, 0.5);
    let neg: Vec<f32> = audio.iter().map(|&x| -x).collect();
    let result = quality::compute_cosine_similarity(&audio, &neg, 0.85).unwrap();
    assert!(
        (result.value - (-1.0)).abs() < 1e-6,
        "negated signal should have cosine similarity -1.0"
    );
    assert!(!result.passed);
}

#[test]
fn test_compute_snr_identical_signals_infinite() {
    let audio = sine_wave_full(440.0, 24000, 0.5, 0.5);
    let result = quality::compute_snr(&audio, &audio, 10.0).unwrap();
    assert!(result.value.is_infinite() || result.value > 100.0);
    assert!(result.passed);
}

#[test]
fn test_compute_sdr_identical_signals_infinite() {
    let audio = sine_wave_full(440.0, 24000, 0.5, 0.5);
    let result = quality::compute_sdr(&audio, &audio, 5.0).unwrap();
    assert!(result.value.is_infinite() || result.value > 100.0);
    assert!(result.passed);
}

// ===========================================================================
// Deterministic hashing
// ===========================================================================

#[test]
fn test_pcm_sha256_deterministic() {
    let audio = vec![0.1_f32, -0.2, 0.3, 0.0, 1.0];
    let h1 = deterministic::pcm_sha256(&audio);
    let h2 = deterministic::pcm_sha256(&audio);
    assert_eq!(h1, h2, "same input should produce same hash");
    assert_eq!(h1.len(), 64, "SHA-256 should be 64 hex chars");
}

#[test]
fn test_pcm_sha256_order_matters() {
    let a = vec![0.1_f32, 0.2];
    let b = vec![0.2_f32, 0.1];
    assert_ne!(deterministic::pcm_sha256(&a), deterministic::pcm_sha256(&b));
}

#[test]
fn test_deterministic_cert_roundtrip() {
    let audio = rich_signal(24000, 0.5);
    let meta = DeterministicMeta {
        input_text: Some("Hello world".into()),
        voice_id: Some("speaker_01".into()),
        seed: Some(42),
    };
    let cert = DeterministicCert::from_audio(&audio, meta);
    assert!(cert.verify(&audio), "should verify against same audio");

    let mut modified = audio;
    modified[0] += 0.001;
    assert!(!cert.verify(&modified), "should reject modified audio");
}

#[test]
fn test_deterministic_cert_default_meta() {
    let meta = DeterministicMeta::default();
    assert!(meta.input_text.is_none());
    assert!(meta.voice_id.is_none());
    assert!(meta.seed.is_none());
}

// ===========================================================================
// Edge cases: silent audio
// ===========================================================================

#[test]
fn test_silent_audio_all_bound_results() {
    let signal = vec![0.0_f32; 24000];
    let v = warn_verifier(24000);
    let cert = v.verify(&signal).unwrap();
    // non_silence should fail.
    let non_silence = cert
        .hard_bounds
        .iter()
        .find(|b| b.name == "non_silence")
        .unwrap();
    assert!(!non_silence.passed);
    assert!(non_silence.value < 1e-10);
    // no_clipping should pass (no peaks).
    let clipping = cert
        .hard_bounds
        .iter()
        .find(|b| b.name == "no_clipping")
        .unwrap();
    assert!(clipping.passed);
    // no_dc_offset should pass (mean is 0).
    let dc = cert
        .hard_bounds
        .iter()
        .find(|b| b.name == "no_dc_offset")
        .unwrap();
    assert!(dc.passed);
    // duration should pass (1 sec > 0.1 min).
    let dur = cert
        .hard_bounds
        .iter()
        .find(|b| b.name == "duration")
        .unwrap();
    assert!(dur.passed);
}

// ===========================================================================
// Edge cases: clipped audio
// ===========================================================================

#[test]
fn test_clipped_audio_multiple_bounds_fail() {
    let mut signal = rich_signal(24000, 0.5);
    for s in &mut signal {
        *s *= 5.0; // amplify heavily
    }
    let v = warn_verifier(24000);
    let cert = v.verify(&signal).unwrap();
    let clipping = cert
        .hard_bounds
        .iter()
        .find(|b| b.name == "no_clipping")
        .unwrap();
    assert!(!clipping.passed, "amplified signal should fail clipping");
    assert!(clipping.value > 1.0);
}

// ===========================================================================
// Edge cases: very short audio
// ===========================================================================

#[test]
fn test_very_short_audio_5_samples() {
    let signal = vec![0.3_f32, -0.3, 0.3, -0.3, 0.3];
    let v = warn_verifier(24000);
    let cert = v.verify(&signal).unwrap();
    let dur = cert
        .hard_bounds
        .iter()
        .find(|b| b.name == "duration")
        .unwrap();
    assert!(!dur.passed, "5 samples at 24kHz = ~0.2ms, below min 100ms");
}

#[test]
fn test_minimum_duration_boundary() {
    // Exactly 100ms at 24kHz = 2400 samples.
    let signal = sine_wave_samples(440.0, 24000, 2400);
    let result = check_duration(&signal, 24000, 0.1, 300.0);
    assert!(result.passed, "exactly 100ms should pass 0.1s minimum");
}

#[test]
fn test_just_below_minimum_duration() {
    // 99ms at 24kHz = 2376 samples.
    let signal = sine_wave_samples(440.0, 24000, 2376);
    let result = check_duration(&signal, 24000, 0.1, 300.0);
    assert!(!result.passed, "99ms should fail 0.1s minimum");
}

// ===========================================================================
// Certificate methods
// ===========================================================================

#[test]
fn test_certificate_empty_hard_bounds_vacuously_passes() {
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
    assert!(cert.passes_hard_bounds());
    assert!(cert.passes_quality());
}

#[test]
fn test_certificate_report_no_quality_section_when_empty() {
    let cert = Certificate {
        hard_bounds: vec![HardBound {
            name: "test",
            passed: true,
            value: 0.5,
            threshold: 0.1,
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
    assert!(!report.contains("Quality Metrics"));
    assert!(report.contains("Hard Bounds"));
}

#[test]
fn test_certificate_report_no_phoneme_section_when_none() {
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
    let report = cert.report();
    assert!(!report.contains("Per-Phoneme Verification"));
}

#[test]
fn test_certificate_report_no_hash_section_when_none() {
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
    let report = cert.report();
    assert!(!report.contains("Deterministic Hash"));
}

// ===========================================================================
// Statistics utilities
// ===========================================================================

#[test]
fn test_welch_t_test_identical_samples() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (t, _df, p) = welch_t_test(&a, &b).unwrap();
    assert!(t.abs() < 1e-10, "identical samples should have t=0");
    assert!(
        (p - 1.0).abs() < 0.01,
        "identical samples should have p~1.0"
    );
}

#[test]
fn test_welch_t_test_very_different_samples() {
    let a = vec![1.0, 1.1, 0.9, 1.0, 1.05];
    let b = vec![100.0, 100.1, 99.9, 100.0, 100.05];
    let (t, _df, p) = welch_t_test(&a, &b).unwrap();
    assert!(
        t.abs() > 10.0,
        "very different samples should have large |t|"
    );
    assert!(p < 0.001, "very different samples should have small p");
}

#[test]
fn test_welch_t_test_insufficient_samples() {
    let err = welch_t_test(&[1.0], &[2.0, 3.0]).unwrap_err();
    assert!(matches!(err, TtsVerifyError::Dsp(_)));
}

#[test]
fn test_welch_t_test_nan_input() {
    let err = welch_t_test(&[1.0, f64::NAN], &[2.0, 3.0]).unwrap_err();
    assert!(matches!(err, TtsVerifyError::NonFiniteInput { .. }));
}

#[test]
fn test_cohens_d_no_effect() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let d = cohens_d(&a, &b).unwrap();
    assert!(d.abs() < 1e-10, "identical samples should have d=0");
}

#[test]
fn test_cohens_d_large_effect() {
    let a = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let b = vec![10.0, 10.0, 10.0, 10.0, 10.0];
    let d = cohens_d(&a, &b).unwrap();
    // Zero variance in both groups → pooled sd is 0 → d is 0.
    assert!(d.abs() < 1e-10, "zero-variance groups should have d=0");
}

#[test]
fn test_cohens_d_nan_input() {
    let err = cohens_d(&[1.0, f64::NAN], &[2.0, 3.0]).unwrap_err();
    assert!(matches!(err, TtsVerifyError::NonFiniteInput { .. }));
}

#[test]
fn test_holm_bonferroni_empty() {
    let result = holm_bonferroni(&[]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_holm_bonferroni_single_value() {
    let result = holm_bonferroni(&[0.03]).unwrap();
    assert_eq!(result.len(), 1);
    assert!((result[0] - 0.03).abs() < 1e-10);
}

#[test]
fn test_holm_bonferroni_multiple_values() {
    let p_values = vec![0.01, 0.04, 0.03, 0.005];
    let adjusted = holm_bonferroni(&p_values).unwrap();
    assert_eq!(adjusted.len(), 4);
    // All adjusted p-values should be >= original.
    for (orig, adj) in p_values.iter().zip(adjusted.iter()) {
        assert!(
            *adj >= *orig - 1e-10,
            "adjusted {adj} should be >= original {orig}"
        );
    }
    // All adjusted p-values should be <= 1.0.
    for adj in &adjusted {
        assert!(*adj <= 1.0 + 1e-10);
    }
}

#[test]
fn test_holm_bonferroni_nan_rejected() {
    let err = holm_bonferroni(&[0.01, f64::NAN, 0.05]).unwrap_err();
    assert!(matches!(err, TtsVerifyError::NonFiniteInput { .. }));
}

// ===========================================================================
// QualityConfig validation
// ===========================================================================

#[test]
fn test_quality_config_default_valid() {
    QualityConfig::default().validate().unwrap();
}

#[test]
fn test_quality_config_nan_min_cosine_similarity() {
    let cfg = QualityConfig {
        min_cosine_similarity: f64::NAN,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_quality_config_nan_min_sdr_db() {
    let cfg = QualityConfig {
        min_sdr_db: f64::INFINITY,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_quality_config_nan_f0_contour_correlation() {
    let cfg = QualityConfig {
        min_f0_contour_correlation: Some(f64::NAN),
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

// ===========================================================================
// Effective threshold combinations
// ===========================================================================

#[test]
fn test_effective_threshold_override_takes_precedence() {
    let hb = HardBoundsConfig {
        min_rms: 0.01, // default
        overrides: CheckOverrides {
            min_rms: Some(0.001), // override
            ..Default::default()
        },
        ..Default::default()
    };
    assert!((hb.effective_min_rms() - 0.001).abs() < 1e-10);
}

#[test]
fn test_effective_threshold_falls_back_to_default() {
    let hb = HardBoundsConfig::default();
    assert!((hb.effective_min_rms() - 0.01).abs() < 1e-10);
    assert!((hb.effective_max_amplitude() - 1.0).abs() < 1e-10);
}

#[test]
fn test_override_loosens_clipping_threshold_behavior() {
    let mut signal = rich_signal(24000, 0.5);
    // Amplify slightly: peak around 0.75
    for s in &mut signal {
        *s *= 2.5;
    }

    // Strict: max_amplitude=0.5 should fail.
    let strict = HardBoundsConfig {
        max_amplitude: 0.5,
        rejection_policy: RejectionPolicy::Warn,
        ..Default::default()
    };
    let v_strict = TtsVerifier::builder().hard_bounds(strict).build().unwrap();
    let cert_strict = v_strict.verify(&signal).unwrap();
    let clip_strict = cert_strict
        .hard_bounds
        .iter()
        .find(|b| b.name == "no_clipping")
        .unwrap();
    assert!(
        !clip_strict.passed,
        "0.5 threshold should fail on amplified signal"
    );

    // Override loosens to 2.0 → should pass.
    let loose = HardBoundsConfig {
        max_amplitude: 0.5,
        rejection_policy: RejectionPolicy::Warn,
        overrides: CheckOverrides {
            max_amplitude: Some(2.0),
            ..Default::default()
        },
        ..Default::default()
    };
    let v_loose = TtsVerifier::builder().hard_bounds(loose).build().unwrap();
    let cert_loose = v_loose.verify(&signal).unwrap();
    let clip_loose = cert_loose
        .hard_bounds
        .iter()
        .find(|b| b.name == "no_clipping")
        .unwrap();
    assert!(
        clip_loose.passed,
        "2.0 threshold should pass on amplified signal"
    );
}
