// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PESQ (Perceptual Evaluation of Speech Quality) metric.

use super::*;
use crate::test_audio_helpers::sine_wave;

#[test]
fn test_identical_signals_high_pesq() {
    let signal = sine_wave(440.0, 16000, 1.0);
    let result = compute_pesq(&signal, &signal, 16000, 2.0).unwrap();
    // Identical signals should produce high PESQ (close to 4.5).
    assert!(
        result.value > 3.5,
        "Identical signals should have PESQ > 3.5, got {}",
        result.value
    );
    assert!(result.passed);
}

#[test]
fn test_noisy_signal_lower_pesq() {
    let reference = sine_wave(440.0, 16000, 1.0);
    // Add significant noise.
    let noisy: Vec<f32> = reference
        .iter()
        .enumerate()
        .map(|(i, &x)| x + 0.5 * ((i as f32 * 17.3).sin()))
        .collect();
    let clean_result = compute_pesq(&reference, &reference, 16000, 0.0).unwrap();
    let noisy_result = compute_pesq(&reference, &noisy, 16000, 0.0).unwrap();
    assert!(
        noisy_result.value < clean_result.value,
        "Noisy signal ({}) should have lower PESQ than clean ({})",
        noisy_result.value,
        clean_result.value
    );
}

#[test]
fn test_pesq_range() {
    let reference = sine_wave(200.0, 16000, 1.0);
    let degraded = sine_wave(2000.0, 16000, 1.0);
    let result = compute_pesq(&reference, &degraded, 16000, 0.0).unwrap();
    assert!(
        result.value >= -0.5 && result.value <= 4.5,
        "PESQ should be in [-0.5, 4.5], got {}",
        result.value
    );
}

#[test]
fn test_pesq_8khz() {
    let signal = sine_wave(440.0, 8000, 1.0);
    let result = compute_pesq(&signal, &signal, 8000, 2.0).unwrap();
    assert!(
        result.value > 3.0,
        "Identical signals at 8 kHz should have PESQ > 3.0, got {}",
        result.value
    );
}

#[test]
fn test_pesq_24khz() {
    let signal = sine_wave(440.0, 24000, 1.0);
    let result = compute_pesq(&signal, &signal, 24000, 2.0).unwrap();
    assert!(
        result.value > 3.0,
        "Identical signals at 24 kHz should have PESQ > 3.0, got {}",
        result.value
    );
}

#[test]
fn test_scaled_signal() {
    let reference = sine_wave(440.0, 16000, 1.0);
    let scaled: Vec<f32> = reference.iter().map(|&x| x * 0.5).collect();
    // Level alignment should compensate for gain difference.
    let result = compute_pesq(&reference, &scaled, 16000, 0.0).unwrap();
    // After level alignment, scaled and original should be very similar.
    assert!(
        result.value > 3.0,
        "Level-aligned scaled signal should have PESQ > 3.0, got {}",
        result.value
    );
}

#[test]
fn test_empty_input_error() {
    let result = compute_pesq(&[], &[1.0], 16000, 2.0);
    assert!(result.is_err());
}

#[test]
fn test_length_mismatch_error() {
    let a = sine_wave(440.0, 16000, 1.0);
    let b = sine_wave(440.0, 16000, 0.5);
    let result = compute_pesq(&a, &b, 16000, 2.0);
    assert!(result.is_err());
}

#[test]
fn test_zero_sample_rate_error() {
    let signal = sine_wave(440.0, 16000, 1.0);
    let result = compute_pesq(&signal, &signal, 0, 2.0);
    assert!(result.is_err());
}

#[test]
fn test_too_short_signal_error() {
    let short = vec![0.1_f32; 100]; // < 0.5s at 16 kHz.
    let result = compute_pesq(&short, &short, 16000, 2.0);
    assert!(result.is_err());
}

#[test]
fn test_threshold_pass_fail() {
    let reference = sine_wave(440.0, 16000, 1.0);
    // Add heavy noise so PESQ is low.
    let noisy: Vec<f32> = reference
        .iter()
        .enumerate()
        .map(|(i, &x)| x + 0.8 * ((i as f32 * 37.1).sin()))
        .collect();
    let result = compute_pesq(&reference, &noisy, 16000, 0.0).unwrap();
    let strict_result = compute_pesq(&reference, &noisy, 16000, result.value + 1.0).unwrap();
    assert!(
        !strict_result.passed,
        "Noisy signal should fail threshold above its score (score={}, threshold={})",
        result.value,
        result.value + 1.0,
    );
}

#[test]
fn test_metric_name_and_citation() {
    let signal = sine_wave(440.0, 16000, 1.0);
    let result = compute_pesq(&signal, &signal, 16000, 0.0).unwrap();
    assert_eq!(result.name, "pesq");
    assert!(result.citation.contains("P.862"));
}

#[test]
fn test_level_align_preserves_shape() {
    let samples = vec![0.1_f32, 0.2, -0.3, 0.4, -0.5];
    let aligned = level_align(&samples);
    assert_eq!(aligned.len(), samples.len());
    // Verify sign preservation.
    for (orig, aligned_val) in samples.iter().zip(aligned.iter()) {
        assert_eq!(orig.signum(), aligned_val.signum() as f32);
    }
}

#[test]
fn test_level_align_silent_signal() {
    let silent = vec![0.0_f32; 100];
    let aligned = level_align(&silent);
    // Silent signal should remain silent.
    assert!(aligned.iter().all(|&x| x == 0.0));
}

#[test]
fn test_estimate_delay_no_delay() {
    let signal: Vec<f64> = (0..1000)
        .map(|i| (2.0 * std::f64::consts::PI * 440.0 * f64::from(i) / 16000.0).sin())
        .collect();
    let delay = estimate_delay(&signal, &signal);
    assert_eq!(delay, 0, "Identical signals should have zero delay");
}

#[test]
fn test_estimate_delay_positive() {
    // Use a chirp (frequency sweep) so cross-correlation has a unique peak.
    // A pure sine has periodic NCC peaks at every period multiple.
    let n = 4000_usize;
    let shift = 80_usize;
    let sr = 16000.0_f64;
    let reference: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / sr;
            // Chirp from 200 Hz to 2000 Hz over the signal duration.
            let freq = 200.0 + 1800.0 * t * sr / n as f64;
            (2.0 * std::f64::consts::PI * freq * t).sin()
        })
        .collect();
    // degraded[i] = reference[i - shift], so degraded is delayed by `shift` samples.
    let mut degraded = vec![0.0_f64; n];
    degraded[shift..n].copy_from_slice(&reference[..(n - shift)]);
    let delay = estimate_delay(&reference, &degraded);
    // Positive delay means degraded is delayed relative to reference.
    assert!(
        (delay - shift as i64).abs() <= 2,
        "Expected delay ~{shift}, got {delay}",
    );
}

#[test]
fn test_bark_band_weights_coverage() {
    let weights = bark_band_weights(PESQ_N_FFT, 16000);
    assert_eq!(weights.len(), N_BARK_BANDS);
    // Verify each band has at least one active bin.
    for (i, band) in weights.iter().enumerate() {
        let active = band.iter().filter(|&&w| w > 0.0).count();
        // Some bands at very high frequencies may have 0 bins for 16 kHz.
        if BARK_EDGES[i + 1] <= 8000.0 {
            assert!(active > 0, "Band {i} should have active bins");
        }
    }
}
