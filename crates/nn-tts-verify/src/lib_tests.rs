// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for TtsVerifier builder API and Certificate integration.

use crate::*;

/// Generate a synthetic speech-like signal with broad spectral coverage.
///
/// Includes harmonics from 220 Hz to 7040 Hz (6 octaves), matching the
/// spectral spread of natural speech (fundamental + formants + fricatives).
fn synthetic_speech(sample_rate: u32, duration_sec: f64) -> Vec<f32> {
    let n = (f64::from(sample_rate) * duration_sec) as usize;
    let mut signal = vec![0.0_f32; n];
    // Harmonics spanning 220-7040 Hz (6 octaves, covers 8 spectral bands at 24 kHz).
    let freqs = [220.0, 440.0, 880.0, 1760.0, 3520.0, 5280.0, 7040.0];
    let amps = [0.25, 0.20, 0.15, 0.10, 0.06, 0.03, 0.02];
    for (i, sample) in signal.iter_mut().enumerate() {
        let t = i as f64 / f64::from(sample_rate);
        for (&f, &a) in freqs.iter().zip(amps.iter()) {
            *sample += (a * (2.0 * std::f64::consts::PI * f * t).sin()) as f32;
        }
    }
    signal
}

use crate::test_audio_helpers::sine_wave_full;

// -- Builder tests -----------------------------------------------------------

#[test]
fn test_builder_default_sample_rate() {
    let v = TtsVerifier::builder().build().unwrap();
    assert_eq!(v.sample_rate, 24000);
}

#[test]
fn test_builder_zero_sample_rate_rejected() {
    let err = TtsVerifier::builder().sample_rate(0).build().unwrap_err();
    assert!(matches!(err, TtsVerifyError::InvalidSampleRate(0)));
}

#[test]
fn test_builder_custom_sample_rate() {
    let v = TtsVerifier::builder().sample_rate(16000).build().unwrap();
    assert_eq!(v.sample_rate, 16000);
}

#[test]
fn test_builder_with_quality() {
    let v = TtsVerifier::builder().with_quality().build().unwrap();
    assert!(v.quality.is_some());
}

#[test]
fn test_builder_custom_hard_bounds() {
    let config = HardBoundsConfig {
        min_rms: 0.001,
        max_amplitude: 0.95,
        ..Default::default()
    };
    let v = TtsVerifier::builder().hard_bounds(config).build().unwrap();
    assert!((v.hard_bounds.min_rms - 0.001).abs() < 1e-10);
    assert!((v.hard_bounds.max_amplitude - 0.95).abs() < 1e-10);
}

// -- Verify (hard bounds only) tests -----------------------------------------

#[test]
fn test_verify_synthetic_speech_passes() {
    let signal = synthetic_speech(24000, 0.5);
    let v = TtsVerifier::builder().build().unwrap();
    let cert = v.verify(&signal).unwrap();
    assert!(
        cert.passes_hard_bounds(),
        "Hard bounds should pass on synthetic speech"
    );
    assert!(
        cert.quality_metrics.is_empty(),
        "No quality metrics without reference"
    );
    assert!(cert.overall_passed);
}

#[test]
fn test_verify_empty_input() {
    let v = TtsVerifier::builder().build().unwrap();
    let err = v.verify(&[]).unwrap_err();
    assert!(matches!(err, TtsVerifyError::EmptyInput));
}

#[test]
fn test_verify_non_finite_input() {
    let v = TtsVerifier::builder().build().unwrap();
    let samples = vec![0.1, f32::NAN, 0.2];
    let err = v.verify(&samples).unwrap_err();
    assert!(matches!(err, TtsVerifyError::NonFiniteInput { count: 1 }));
}

#[test]
fn test_verify_silent_fails() {
    let signal = vec![0.0_f32; 24000]; // 1 second of silence.
                                       // Use Warn policy so we get Ok(cert) even when hard bounds fail,
                                       // allowing us to inspect which individual checks failed.
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Warn,
        ..Default::default()
    };
    let v = TtsVerifier::builder().hard_bounds(hb).build().unwrap();
    let cert = v.verify(&signal).unwrap();
    assert!(
        !cert.passes_hard_bounds(),
        "Silence should fail non_silence check"
    );
    let silence_check = cert
        .hard_bounds
        .iter()
        .find(|b| b.name == "non_silence")
        .unwrap();
    assert!(!silence_check.passed);
}

#[test]
fn test_verify_clipped_fails() {
    let mut signal = synthetic_speech(24000, 0.5);
    // Amplify to cause clipping.
    for s in &mut signal {
        *s *= 5.0;
    }
    // Use Warn policy so we get Ok(cert) for inspection.
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Warn,
        ..Default::default()
    };
    let v = TtsVerifier::builder().hard_bounds(hb).build().unwrap();
    let cert = v.verify(&signal).unwrap();
    let clip_check = cert
        .hard_bounds
        .iter()
        .find(|b| b.name == "no_clipping")
        .unwrap();
    assert!(
        !clip_check.passed,
        "Amplified signal should fail clipping check"
    );
}

#[test]
fn test_verify_dc_offset_fails() {
    let mut signal = synthetic_speech(24000, 0.5);
    // Add large DC offset.
    for s in &mut signal {
        *s += 0.2;
    }
    // Use Warn policy so we get Ok(cert) for inspection.
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Warn,
        ..Default::default()
    };
    let v = TtsVerifier::builder().hard_bounds(hb).build().unwrap();
    let cert = v.verify(&signal).unwrap();
    let dc_check = cert
        .hard_bounds
        .iter()
        .find(|b| b.name == "no_dc_offset")
        .unwrap();
    assert!(!dc_check.passed, "DC offset should fail");
}

#[test]
fn test_verify_too_short_fails() {
    let signal = sine_wave_full(440.0, 24000, 0.01, 0.3); // 10ms — below 100ms min.
                                                          // Use Warn policy so we get Ok(cert) for inspection.
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Warn,
        ..Default::default()
    };
    let v = TtsVerifier::builder().hard_bounds(hb).build().unwrap();
    let cert = v.verify(&signal).unwrap();
    let dur_check = cert
        .hard_bounds
        .iter()
        .find(|b| b.name == "duration")
        .unwrap();
    assert!(!dur_check.passed, "10ms signal should fail duration check");
}

// -- Verify with reference tests ---------------------------------------------

#[test]
fn test_verify_with_reference_identical() {
    let signal = synthetic_speech(24000, 0.5);
    // Use Warn policy: spectral_tilt quality metric may fail on synthetic
    // harmonics, but we want to inspect the certificate regardless.
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Warn,
        ..Default::default()
    };
    let v = TtsVerifier::builder().hard_bounds(hb).build().unwrap();
    let cert = v.verify_with_reference(&signal, &signal).unwrap();
    assert!(cert.passes_hard_bounds());
    // MCD of identical signals should be near zero.
    let mcd = cert
        .quality_metrics
        .iter()
        .find(|m| m.name == "mcd")
        .unwrap();
    assert!(
        mcd.value < 1.0,
        "MCD of identical signals should be near 0, got {}",
        mcd.value
    );
    assert!(mcd.passed);
}

#[test]
fn test_verify_with_reference_length_mismatch() {
    let a = synthetic_speech(24000, 0.5);
    let b = synthetic_speech(24000, 1.0);
    let v = TtsVerifier::builder().build().unwrap();
    let err = v.verify_with_reference(&a, &b).unwrap_err();
    assert!(matches!(err, TtsVerifyError::LengthMismatch { .. }));
}

// -- Certificate tests -------------------------------------------------------

#[test]
fn test_certificate_report_contains_status() {
    let signal = synthetic_speech(24000, 0.5);
    let v = TtsVerifier::builder().build().unwrap();
    let cert = v.verify(&signal).unwrap();
    let report = cert.report();
    assert!(report.contains("TTS Verification Certificate"));
    assert!(report.contains("Hard Bounds"));
    assert!(report.contains("PASS") || report.contains("FAIL"));
}

#[test]
fn test_certificate_passes_quality_vacuous() {
    let signal = synthetic_speech(24000, 0.5);
    let v = TtsVerifier::builder().build().unwrap();
    let cert = v.verify(&signal).unwrap();
    assert!(
        cert.passes_quality(),
        "Vacuously true when no quality metrics"
    );
}

#[test]
fn test_verify_with_quality_enabled() {
    let signal = synthetic_speech(24000, 0.5);
    // Use Warn policy: synthetic speech may fail spectral_tilt quality
    // metric, and we want to inspect the certificate not test rejection.
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
        !cert.quality_metrics.is_empty(),
        "Should have quality metrics when enabled"
    );
    // Should have HNR, F0 range, spectral tilt.
    assert!(cert.quality_metrics.len() >= 3);
}

// -- Certificate::passes_quality failure path --------------------------------

#[test]
fn test_certificate_passes_quality_with_failing_metric() {
    use crate::quality::QualityMetric;

    let cert = Certificate {
        hard_bounds: vec![],
        quality_metrics: vec![
            QualityMetric {
                name: "good_metric",
                value: 0.9,
                threshold: 0.5,
                passed: true,
                citation: "test",
            },
            QualityMetric {
                name: "bad_metric",
                value: 0.3,
                threshold: 0.5,
                passed: false,
                citation: "test",
            },
        ],
        phoneme_results: None,
        overall_passed: false,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };
    assert!(
        !cert.passes_quality(),
        "passes_quality() must return false when any quality metric fails"
    );
}

// -- Certificate::report() branch coverage -----------------------------------

#[test]
fn test_certificate_report_with_phoneme_results() {
    use crate::phoneme::PhonemeResult;

    let cert = Certificate {
        hard_bounds: vec![],
        quality_metrics: vec![],
        phoneme_results: Some(vec![
            PhonemeResult {
                label: "aa".into(),
                duration_ms: 50.0,
                metrics: vec![],
                passed: true,
            },
            PhonemeResult {
                label: "k".into(),
                duration_ms: 30.0,
                metrics: vec![],
                passed: false,
            },
        ]),
        overall_passed: false,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };
    let report = cert.report();
    assert!(
        report.contains("Per-Phoneme Verification"),
        "report must include phoneme section"
    );
    assert!(
        report.contains("1/2 phonemes passed"),
        "report must show phoneme pass count"
    );
    assert!(report.contains("/aa/"), "report must list phoneme labels");
    assert!(report.contains("/k/"), "report must list phoneme labels");
}

#[test]
fn test_certificate_report_with_deterministic_hash() {
    let cert = Certificate {
        hard_bounds: vec![],
        quality_metrics: vec![],
        phoneme_results: None,
        overall_passed: true,
        deterministic_hash: Some("abc123def456".into()),
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };
    let report = cert.report();
    assert!(
        report.contains("Deterministic Hash"),
        "report must include hash section"
    );
    assert!(
        report.contains("abc123def456"),
        "report must include the actual hash value"
    );
}
