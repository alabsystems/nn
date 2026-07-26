// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for STOI (Short-Time Objective Intelligibility) metric.

use super::*;
use crate::test_audio_helpers::sine_wave;

#[test]
fn test_identical_signals_high_stoi() {
    let signal = sine_wave(440.0, 16000, 1.0);
    let result = compute_stoi(&signal, &signal, 16000, 0.5).unwrap();
    // Identical signals should have very high STOI (close to 1.0).
    assert!(
        result.value > 0.9,
        "Identical signals should have STOI > 0.9, got {}",
        result.value
    );
    assert!(result.passed);
}

#[test]
fn test_noise_reduces_stoi() {
    let reference = sine_wave(440.0, 16000, 1.0);
    // Add significant noise.
    let noisy: Vec<f32> = reference
        .iter()
        .enumerate()
        .map(|(i, &x)| x + 0.5 * ((i as f32 * 17.3).sin()))
        .collect();
    let clean_result = compute_stoi(&reference, &reference, 16000, 0.5).unwrap();
    let noisy_result = compute_stoi(&reference, &noisy, 16000, 0.5).unwrap();
    assert!(
        noisy_result.value < clean_result.value,
        "Noisy signal ({}) should have lower STOI than clean ({})",
        noisy_result.value,
        clean_result.value
    );
}

#[test]
fn test_stoi_range_0_to_1() {
    let reference = sine_wave(200.0, 16000, 1.0);
    let degraded = sine_wave(2000.0, 16000, 1.0);
    let result = compute_stoi(&reference, &degraded, 16000, 0.0).unwrap();
    assert!(
        result.value >= 0.0 && result.value <= 1.0,
        "STOI should be in [0, 1], got {}",
        result.value
    );
}

#[test]
fn test_different_sample_rates() {
    // Test with 24 kHz (common TTS rate).
    let signal = sine_wave(440.0, 24000, 1.0);
    let result = compute_stoi(&signal, &signal, 24000, 0.5).unwrap();
    assert!(
        result.value > 0.8,
        "Identical signals at 24 kHz should have high STOI, got {}",
        result.value
    );
}

#[test]
fn test_10k_native_sample_rate() {
    // Test at STOI's native 10 kHz — no resampling needed.
    let signal = sine_wave(440.0, 10000, 1.0);
    let result = compute_stoi(&signal, &signal, 10000, 0.5).unwrap();
    assert!(
        result.value > 0.9,
        "Identical signals at 10 kHz should have STOI > 0.9, got {}",
        result.value
    );
}

#[test]
fn test_22050_non_integer_decimation() {
    // 22050 Hz → 10 kHz has ratio 2.205 (non-integer), exercises linear interpolation.
    let signal = sine_wave(440.0, 22050, 1.0);
    let result = compute_stoi(&signal, &signal, 22050, 0.5).unwrap();
    assert!(
        result.value > 0.8,
        "Identical signals at 22050 Hz should have STOI > 0.8, got {}",
        result.value
    );
}

#[test]
fn test_threshold_pass_fail() {
    let reference = sine_wave(440.0, 16000, 1.0);
    // Add heavy noise to reduce STOI score.
    let noisy: Vec<f32> = reference
        .iter()
        .enumerate()
        .map(|(i, &x)| x + 0.8 * ((i as f32 * 37.1).sin()))
        .collect();
    // Compute score with no threshold to get actual value.
    let result = compute_stoi(&reference, &noisy, 16000, 0.0).unwrap();
    assert!(result.passed, "Should pass with threshold 0.0");
    // Now set threshold above the actual score — should fail.
    let strict = compute_stoi(&reference, &noisy, 16000, result.value + 0.1).unwrap();
    assert!(
        !strict.passed,
        "Should fail with threshold above actual score (score={}, threshold={})",
        result.value,
        result.value + 0.1,
    );
}

#[test]
fn test_empty_input_error() {
    let result = compute_stoi(&[], &[1.0], 16000, 0.5);
    assert!(result.is_err());
}

#[test]
fn test_length_mismatch_error() {
    let a = sine_wave(440.0, 16000, 1.0);
    let b = sine_wave(440.0, 16000, 0.5);
    let result = compute_stoi(&a, &b, 16000, 0.5);
    assert!(result.is_err());
}

#[test]
fn test_zero_sample_rate_error() {
    let signal = sine_wave(440.0, 16000, 1.0);
    let result = compute_stoi(&signal, &signal, 0, 0.5);
    assert!(result.is_err());
}

#[test]
fn test_too_short_signal_error() {
    // Signal too short for STOI analysis (need >= 30 frames at 10 kHz).
    let short = vec![0.1_f32; 100];
    let result = compute_stoi(&short, &short, 16000, 0.5);
    assert!(result.is_err());
}

#[test]
fn test_metric_name_and_citation() {
    let signal = sine_wave(440.0, 16000, 1.0);
    let result = compute_stoi(&signal, &signal, 16000, 0.5).unwrap();
    assert_eq!(result.name, "stoi");
    assert!(result.citation.contains("Taal"));
}

#[test]
fn test_pearson_correlation_perfect() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let r = pearson_correlation(&a, &b);
    assert!((r - 1.0).abs() < 1e-10);
}

#[test]
fn test_pearson_correlation_negative() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let r = pearson_correlation(&a, &b);
    assert!((r - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_vector_norm() {
    let v = vec![3.0, 4.0];
    assert!((vector_norm(&v) - 5.0).abs() < 1e-10);
}

#[test]
fn test_third_octave_bands_cover_speech_range() {
    let weights = third_octave_band_weights(STOI_N_FFT, STOI_SAMPLE_RATE);
    assert_eq!(weights.len(), 15);
    // Each band should have at least one non-zero bin.
    for (i, band) in weights.iter().enumerate() {
        let active_bins: usize = band.iter().filter(|&&w| w > 0.0).count();
        assert!(
            active_bins > 0,
            "Band {} (center={} Hz) has no active bins",
            i,
            THIRD_OCTAVE_CENTERS[i]
        );
    }
}
