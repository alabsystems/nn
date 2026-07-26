// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for audio-domain disentanglement measurement.

use super::*;
use crate::test_audio_helpers::sine_wave_samples;

/// Generate a chirp (frequency sweep) from f0 to f1.
fn chirp(f0_hz: f64, f1_hz: f64, sample_rate: u32, n_samples: usize) -> Vec<f32> {
    (0..n_samples)
        .map(|i| {
            let t = i as f64 / f64::from(sample_rate);
            let duration = n_samples as f64 / f64::from(sample_rate);
            let freq = f0_hz + (f1_hz - f0_hz) * t / duration;
            (2.0 * std::f64::consts::PI * freq * t).sin() as f32
        })
        .collect()
}

// ---- Design doc test #9: F0-shifted pair ------------------------------------

#[test]
fn test_audio_disentanglement_f0_shift() {
    // Baseline: 200 Hz sine. Perturbed: 300 Hz sine. Same duration.
    // F0 shift should show LOW F0 correlation (different pitch) but
    // similar spectral envelope shape (both pure sinusoids) and same duration.
    let sr = 16000;
    let n = sr * 2; // 2 seconds
    let baseline = sine_wave_samples(200.0, sr as u32, n);
    let perturbed = sine_wave_samples(300.0, sr as u32, n);

    let result = measure_audio_disentanglement(&baseline, &perturbed, sr as u32).unwrap();

    // F0 is different (200 vs 300 Hz) → correlation should be low.
    // Note: pure sinusoids at different frequencies have near-zero correlation
    // of their F0 contours because YIN detects different constant frequencies.
    // The correlation of two constant-but-different values is undefined (both
    // have zero variance), so f0_pearson_correlation returns 0.0.
    assert!(
        result.f0_correlation < 0.5,
        "F0 correlation should be low for different pitches, got {}",
        result.f0_correlation
    );

    // Duration is identical.
    assert!(
        (result.duration_ratio - 1.0).abs() < 0.001,
        "Duration ratio should be ~1.0, got {}",
        result.duration_ratio
    );
}

// ---- Design doc test #10: Timbre-shifted pair -------------------------------

#[test]
fn test_audio_disentanglement_timbre_shift() {
    // Baseline: pure sine at 200 Hz.
    // Perturbed: chirp from 200 Hz to 800 Hz (spectrally very different).
    // The MCD should be high (different spectral content).
    let sr = 16000;
    let n = sr * 2; // 2 seconds
    let baseline = sine_wave_samples(200.0, sr as u32, n);
    let perturbed = chirp(200.0, 800.0, sr as u32, n);

    let result = measure_audio_disentanglement(&baseline, &perturbed, sr as u32).unwrap();

    // Spectral content is very different → MCD should be high.
    assert!(
        result.mcd > 2.0,
        "MCD should be high for spectrally different signals, got {}",
        result.mcd
    );

    // Duration is identical (same number of samples).
    assert!(
        (result.duration_ratio - 1.0).abs() < 0.001,
        "Duration ratio should be ~1.0, got {}",
        result.duration_ratio
    );
}

// ---- Additional coverage tests ----------------------------------------------

#[test]
fn test_identical_signals_are_fully_preserved() {
    let sr = 16000;
    let n = sr * 2;
    let signal = sine_wave_samples(200.0, sr as u32, n);

    let result = measure_audio_disentanglement(&signal, &signal, sr as u32).unwrap();

    // Identical signals: perfect F0 correlation, zero MCD, ratio = 1.0,
    // waveform similarity = 1.0.
    assert!(
        result.mcd < 0.01,
        "MCD should be ~0 for identical signals, got {}",
        result.mcd
    );
    assert!(
        (result.duration_ratio - 1.0).abs() < f64::EPSILON,
        "Duration ratio should be exactly 1.0, got {}",
        result.duration_ratio
    );
    assert!(
        result.waveform_similarity > 0.99,
        "Waveform similarity should be ~1.0, got {}",
        result.waveform_similarity
    );
}

#[test]
fn test_different_lengths_compute_duration_ratio() {
    let sr = 16000;
    let baseline = sine_wave_samples(200.0, sr as u32, sr * 2); // 2 seconds
    let perturbed = sine_wave_samples(200.0, sr as u32, sr * 3); // 3 seconds

    let result = measure_audio_disentanglement(&baseline, &perturbed, sr as u32).unwrap();

    // Duration ratio: 3s / 2s = 1.5
    assert!(
        (result.duration_ratio - 1.5).abs() < 0.001,
        "Duration ratio should be ~1.5, got {}",
        result.duration_ratio
    );
}

#[test]
fn test_classify_disentanglement_f0_preserved() {
    let result = AudioDisentanglementResult {
        f0_correlation: 0.95,
        mcd: 3.0,
        duration_ratio: 1.0,
        waveform_similarity: 0.9,
    };
    let thresholds = DisentanglementThresholds::default();
    let evidence = classify_disentanglement(result, &thresholds);

    assert!(evidence.f0_preserved, "F0 should be preserved at 0.95");
    assert!(
        evidence.spectral_preserved,
        "Spectral should be preserved at MCD 3.0"
    );
    assert!(evidence.duration_preserved, "Duration should be preserved");
    assert!(evidence.waveform_preserved, "Waveform should be preserved");
}

#[test]
fn test_classify_disentanglement_f0_changed() {
    let result = AudioDisentanglementResult {
        f0_correlation: 0.2,
        mcd: 3.0,
        duration_ratio: 1.0,
        waveform_similarity: 0.9,
    };
    let thresholds = DisentanglementThresholds::default();
    let evidence = classify_disentanglement(result, &thresholds);

    assert!(!evidence.f0_preserved, "F0 should NOT be preserved at 0.2");
    assert!(
        evidence.spectral_preserved,
        "Spectral should still be preserved"
    );
    assert!(evidence.duration_preserved, "Duration should be preserved");
}

#[test]
fn test_classify_disentanglement_spectral_changed() {
    let result = AudioDisentanglementResult {
        f0_correlation: 0.95,
        mcd: 10.0, // Very high MCD
        duration_ratio: 1.0,
        waveform_similarity: 0.3,
    };
    let thresholds = DisentanglementThresholds::default();
    let evidence = classify_disentanglement(result, &thresholds);

    assert!(evidence.f0_preserved, "F0 should be preserved");
    assert!(
        !evidence.spectral_preserved,
        "Spectral should NOT be preserved at MCD 10.0"
    );
    assert!(
        !evidence.waveform_preserved,
        "Waveform should NOT be preserved at 0.3"
    );
}

#[test]
fn test_classify_disentanglement_duration_changed() {
    let result = AudioDisentanglementResult {
        f0_correlation: 0.95,
        mcd: 3.0,
        duration_ratio: 1.5, // 50% longer
        waveform_similarity: 0.9,
    };
    let thresholds = DisentanglementThresholds::default();
    let evidence = classify_disentanglement(result, &thresholds);

    assert!(
        !evidence.duration_preserved,
        "Duration should NOT be preserved at ratio 1.5"
    );
}

#[test]
fn test_empty_baseline_returns_error() {
    let perturbed = vec![0.1_f32; 100];
    let result = measure_audio_disentanglement(&[], &perturbed, 16000);
    assert!(result.is_err());
}

#[test]
fn test_empty_perturbed_returns_error() {
    let baseline = vec![0.1_f32; 100];
    let result = measure_audio_disentanglement(&baseline, &[], 16000);
    assert!(result.is_err());
}

#[test]
fn test_zero_sample_rate_returns_error() {
    let signal = vec![0.1_f32; 100];
    let result = measure_audio_disentanglement(&signal, &signal, 0);
    assert!(result.is_err());
}

#[test]
fn test_custom_thresholds() {
    let thresholds = DisentanglementThresholds {
        f0_correlation_min: 0.99, // Very strict
        mcd_max: 2.0,             // Very strict
        duration_ratio_tolerance: 0.01,
        waveform_similarity_min: 0.95,
    };
    let result = AudioDisentanglementResult {
        f0_correlation: 0.95,
        mcd: 3.0,
        duration_ratio: 1.05,
        waveform_similarity: 0.9,
    };
    let evidence = classify_disentanglement(result, &thresholds);

    // With strict thresholds, nothing should be preserved.
    assert!(!evidence.f0_preserved);
    assert!(!evidence.spectral_preserved);
    assert!(!evidence.duration_preserved);
    assert!(!evidence.waveform_preserved);
}
