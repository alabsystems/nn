// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test exercising `nn::tts_verify` from the top-level crate.
//!
//! Validates that dvoice (and other consumers) can use TTS verification
//! via `nn = { features = ["tts-verify"] }` without depending on
//! `nn-tts-verify` directly.
//!
//! Part of #1356.

use nn::tts_verify::{Certificate, HardBoundsConfig, TtsVerifier, TtsVerifyError};

/// Generate a synthetic speech-like signal with broad spectral coverage.
///
/// Includes harmonics from 220 Hz to 7040 Hz (6 octaves), matching the
/// spectral spread of natural speech (fundamental + formants + fricatives).
fn synthetic_speech(sample_rate: u32, duration_sec: f64) -> Vec<f32> {
    let n = (f64::from(sample_rate) * duration_sec) as usize;
    let mut signal = vec![0.0_f32; n];
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

#[test]
fn test_verifier_builder_default() {
    let verifier = TtsVerifier::builder().build().unwrap();
    let samples = synthetic_speech(24000, 0.5);
    let cert = verifier.verify(&samples).expect("verify should succeed");
    assert!(
        cert.passes_hard_bounds(),
        "synthetic speech should pass hard bounds: {}",
        cert.report()
    );
}

#[test]
fn test_verifier_custom_sample_rate() {
    let verifier = TtsVerifier::builder().sample_rate(16000).build().unwrap();
    let samples = synthetic_speech(16000, 0.5);
    let cert = verifier.verify(&samples).expect("verify should succeed");
    assert!(cert.passes_hard_bounds());
}

#[test]
fn test_verifier_empty_input_error() {
    let verifier = TtsVerifier::builder().build().unwrap();
    let err = verifier.verify(&[]).unwrap_err();
    assert!(matches!(err, TtsVerifyError::EmptyInput));
}

#[test]
fn test_verifier_non_finite_input_error() {
    let verifier = TtsVerifier::builder().build().unwrap();
    let samples = vec![1.0, f32::NAN, 0.5];
    let err = verifier.verify(&samples).unwrap_err();
    assert!(matches!(err, TtsVerifyError::NonFiniteInput { .. }));
}

#[test]
fn test_certificate_report_not_empty() {
    let verifier = TtsVerifier::builder().build().unwrap();
    let samples = synthetic_speech(24000, 0.5);
    let cert = verifier.verify(&samples).expect("verify");
    let report = cert.report();
    assert!(!report.is_empty(), "report should contain bound results");
}

#[test]
fn test_verifier_with_reference_quality() {
    let verifier = TtsVerifier::builder().with_quality().build().unwrap();
    let reference = synthetic_speech(24000, 0.5);
    let candidate = synthetic_speech(24000, 0.5);
    let cert = verifier
        .verify_with_reference(&candidate, &reference)
        .expect("verify_with_reference should not error");
    // Hard bounds should pass (synthetic speech has broad spectral coverage).
    assert!(cert.passes_hard_bounds(), "hard bounds: {}", cert.report());
    // Quality metrics are populated (even if spectral_tilt fails for synthetic
    // signals — the API exercised correctly is what matters for integration).
    assert!(
        !cert.quality_metrics.is_empty(),
        "quality metrics should be populated"
    );
}

#[test]
fn test_hard_bounds_config_default() {
    let config = HardBoundsConfig::default();
    assert!(config.max_amplitude > 0.0);
    assert!(config.min_duration_sec > 0.0);
}

#[test]
fn test_certificate_type_accessible() {
    let verifier = TtsVerifier::builder().build().unwrap();
    let samples = synthetic_speech(24000, 0.5);
    let cert: Certificate = verifier.verify(&samples).expect("verify");
    let _hard = cert.passes_hard_bounds();
    let _quality = cert.passes_quality();
    let _report = cert.report();
}
