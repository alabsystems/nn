// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for spectral comparison module.
//!
//! AC6: Known-value spectral pairs (pure sine -> LSD ~ 0, phase shift -> low LSD).
//! AC7: Time-shifted identical waveforms fail element-wise but pass spectral.

use super::stft::make_window;
use super::*;
use std::f32::consts::PI;

#[path = "spectral_stft_tests.rs"]
mod stft_tests;

/// Generate a pure sine wave.
fn sine_wave(freq_hz: f32, sample_rate: f32, num_samples: usize, phase: f32) -> Vec<f32> {
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * PI * freq_hz * t + phase).sin()
        })
        .collect()
}

// -------------------------------------------------------------------------
// AC6: Known-value spectral pairs
// -------------------------------------------------------------------------

/// Two identical signals must have LSD = 0, SC = 0, phase coherence = 1.0.
#[test]
fn test_identical_signals_perfect_spectral_match() {
    let signal = sine_wave(440.0, 16000.0, 4096, 0.0);
    let config = SpectralConfig::default();
    let result = compare_spectral(&signal, &signal, &config).expect("comparison should succeed");

    assert!(
        result.log_spectral_distance_db < 1e-6,
        "LSD should be ~0 for identical signals, got {}",
        result.log_spectral_distance_db
    );
    assert!(
        result.spectral_convergence < 1e-6,
        "SC should be ~0 for identical signals, got {}",
        result.spectral_convergence
    );
    assert!(
        result.phase_coherence > 0.999,
        "phase coherence should be ~1.0 for identical signals, got {}",
        result.phase_coherence
    );
    assert!(
        result.max_magnitude_diff_db < 1e-6,
        "max magnitude diff should be ~0 for identical signals, got {}",
        result.max_magnitude_diff_db
    );
    assert!(result.passed, "identical signals should pass comparison");
}

/// Pure sine at same frequency with a 90-degree phase offset:
/// Magnitude spectrum is similar but NOT identical due to windowed STFT
/// (Hann window interacts differently with different phases at frame edges).
/// LSD is bounded but non-zero; phase coherence drops.
#[test]
fn test_phase_shifted_sine_bounded_lsd() {
    let reference = sine_wave(440.0, 16000.0, 4096, 0.0);
    let candidate = sine_wave(440.0, 16000.0, 4096, PI / 2.0);
    let config = SpectralConfig::new(5.0, 0.1, 0.0); // relax thresholds for windowing effects
    let result =
        compare_spectral(&reference, &candidate, &config).expect("comparison should succeed");

    // Same frequency with windowed STFT: LSD bounded (windowing effects cause
    // non-zero LSD), but much lower than different-frequency case.
    assert!(
        result.log_spectral_distance_db < 5.0,
        "LSD should be bounded for phase shift (same freq), got {}",
        result.log_spectral_distance_db
    );
    // SC should be moderate — same frequency structure.
    assert!(
        result.spectral_convergence < 0.1,
        "SC should be moderate for same-frequency phase shift, got {}",
        result.spectral_convergence
    );

    // Phase coherence should be < 1.0 due to constant phase offset.
    assert!(
        result.phase_coherence < 1.0,
        "phase coherence should be < 1.0 for phase-shifted signal, got {}",
        result.phase_coherence
    );
}

/// Two different frequencies should have high LSD (spectral mismatch).
#[test]
fn test_different_frequencies_high_lsd() {
    let reference = sine_wave(440.0, 16000.0, 4096, 0.0);
    let candidate = sine_wave(880.0, 16000.0, 4096, 0.0);
    let config = SpectralConfig::default();
    let result =
        compare_spectral(&reference, &candidate, &config).expect("comparison should succeed");

    // Different frequencies -> spectral energy in different bins -> high LSD.
    assert!(
        result.log_spectral_distance_db > 1.0,
        "LSD should be > 1.0 dB for different frequencies, got {}",
        result.log_spectral_distance_db
    );
    assert!(
        result.spectral_convergence > 0.01,
        "SC should be > 0.01 for different frequencies, got {}",
        result.spectral_convergence
    );
    assert!(
        !result.passed,
        "different frequencies should fail default thresholds"
    );
}

/// Silence vs signal: SC should be high (reference is all silence).
#[test]
fn test_silence_reference_vs_signal() {
    let reference = vec![0.0f32; 4096];
    let candidate = sine_wave(440.0, 16000.0, 4096, 0.0);
    let config = SpectralConfig::default();
    let result =
        compare_spectral(&reference, &candidate, &config).expect("comparison should succeed");

    // Silence reference -> SC = inf (or very large), LSD large.
    assert!(
        result.spectral_convergence > 0.01,
        "SC should be large for silence vs signal, got {}",
        result.spectral_convergence
    );
    assert!(!result.passed, "silence vs signal should fail");
}

// -------------------------------------------------------------------------
// AC7: Time-shifted identical waveforms
// -------------------------------------------------------------------------

/// A 1-sample time shift of a sine wave should have high element-wise error
/// but near-zero spectral difference (same frequency content).
#[test]
fn test_time_shifted_waveform_passes_spectral_fails_elementwise() {
    let sample_rate = 16000.0;
    let n = 4096;
    let freq = 440.0;

    let reference = sine_wave(freq, sample_rate, n, 0.0);
    // Shift by 1 sample: drop first sample, append one zero.
    let mut candidate = reference[1..].to_vec();
    candidate.push(0.0);

    // --- Element-wise comparison should FAIL ---
    // A 1-sample shift of a 440 Hz sine at 16 kHz creates significant peak error.
    let max_abs = reference
        .iter()
        .zip(candidate.iter())
        .map(|(r, c)| (r - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs > 0.01,
        "element-wise max_abs should be large for time-shifted signal, got {max_abs}"
    );

    // --- Spectral comparison should PASS ---
    // Same frequency content (440 Hz) -> magnitude spectrum is nearly identical.
    let config = SpectralConfig::new(
        2.0,  // LSD: slightly relaxed (edge effects from shift)
        0.05, // SC: slightly relaxed
        0.0,  // ignore phase (phase will differ due to shift)
    );
    let result =
        compare_spectral(&reference, &candidate, &config).expect("spectral comparison succeeded");

    assert!(
        result.log_spectral_distance_db < 2.0,
        "LSD should be low for time-shifted signal, got {} dB",
        result.log_spectral_distance_db
    );
    assert!(
        result.spectral_convergence < 0.05,
        "SC should be low for time-shifted signal, got {}",
        result.spectral_convergence
    );
    assert!(
        result.passed,
        "time-shifted signal should pass spectral comparison: {result}"
    );
}

/// Larger time shift (10 samples): still same frequency content.
#[test]
fn test_larger_time_shift_still_passes_spectral() {
    let sample_rate = 16000.0;
    let n = 8192;
    let freq = 1000.0;

    let reference = sine_wave(freq, sample_rate, n, 0.0);
    // Shift by 10 samples.
    let mut candidate = reference[10..].to_vec();
    candidate.extend_from_slice(&[0.0; 10]);

    // Element-wise error should be significant.
    let max_abs = reference
        .iter()
        .zip(candidate.iter())
        .map(|(r, c)| (r - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs > 0.01,
        "element-wise max_abs should be large for 10-sample shift, got {max_abs}"
    );

    // Spectral comparison should still pass (same frequency).
    let config = SpectralConfig::new(2.0, 0.05, 0.0);
    let result =
        compare_spectral(&reference, &candidate, &config).expect("spectral comparison succeeded");

    assert!(
        result.log_spectral_distance_db < 2.0,
        "LSD should be low for 10-sample shifted signal, got {} dB",
        result.log_spectral_distance_db
    );
    assert!(
        result.passed,
        "10-sample shifted signal should pass spectral"
    );
}

// -------------------------------------------------------------------------
// Window function tests
// -------------------------------------------------------------------------

#[test]
fn test_window_functions_basic_properties() {
    let n = 256;

    // Hann window: starts and ends at 0, peaks in middle.
    let hann = make_window(WindowFn::Hann, n);
    assert_eq!(hann.len(), n);
    assert!(hann[0].abs() < 1e-6, "Hann should start at ~0");
    assert!(hann[n / 2] > 0.9, "Hann should peak near middle");

    // Hamming window: similar shape but doesn't reach 0 at edges.
    let hamming = make_window(WindowFn::Hamming, n);
    assert_eq!(hamming.len(), n);
    assert!(hamming[0] > 0.07, "Hamming should not reach 0 at edges");
    assert!(hamming[n / 2] > 0.9, "Hamming should peak near middle");

    // Rectangular window: all 1.0.
    let rect = make_window(WindowFn::Rectangular, n);
    assert_eq!(rect.len(), n);
    for val in &rect {
        assert!(
            (*val - 1.0_f32).abs() < 1e-6,
            "Rectangular should be all 1.0"
        );
    }
}

// -------------------------------------------------------------------------
// SpectralConfig tests
// -------------------------------------------------------------------------

#[test]
fn test_spectral_config_defaults() {
    let config = SpectralConfig::default();
    assert_eq!(config.max_lsd_db, 1.0);
    assert_eq!(config.max_spectral_convergence, 0.01);
    assert_eq!(config.min_phase_coherence, 0.95);
}

#[test]
fn test_spectral_config_with_stft() {
    let stft = StftConfig {
        n_fft: 512,
        hop_length: 128,
        window: WindowFn::Hamming,
    };
    let config = SpectralConfig::default().with_stft(stft);
    assert_eq!(config.stft.n_fft, 512);
    assert_eq!(config.stft.window, WindowFn::Hamming);
}

// -------------------------------------------------------------------------
// SpectralComparison Display test
// -------------------------------------------------------------------------

#[test]
fn test_spectral_comparison_display_pass() {
    let cmp = SpectralComparison {
        log_spectral_distance_db: 0.5,
        spectral_convergence: 0.005,
        max_magnitude_diff_db: 1.2,
        mean_magnitude_diff_db: 0.3,
        phase_coherence: 0.98,
        passed: true,
    };
    let s = format!("{cmp}");
    assert!(s.contains("[PASS]"));
    assert!(s.contains("LSD="));
}

#[test]
fn test_spectral_comparison_display_fail() {
    let cmp = SpectralComparison {
        log_spectral_distance_db: 5.0,
        spectral_convergence: 0.5,
        max_magnitude_diff_db: 10.0,
        mean_magnitude_diff_db: 5.0,
        phase_coherence: 0.3,
        passed: false,
    };
    let s = format!("{cmp}");
    assert!(s.contains("[FAIL]"));
}

// -------------------------------------------------------------------------
// Edge cases: empty inputs to compare_spectral
// -------------------------------------------------------------------------

#[test]
fn test_compare_spectral_empty_reference() {
    let config = SpectralConfig::default();
    let result = compare_spectral(&[], &[1.0; 1024], &config);
    assert!(result.is_err());
}

#[test]
fn test_compare_spectral_empty_candidate() {
    let config = SpectralConfig::default();
    let result = compare_spectral(&[1.0; 1024], &[], &config);
    assert!(result.is_err());
}

// -------------------------------------------------------------------------
// Multi-frequency signal test
// -------------------------------------------------------------------------

/// Signal with two sine components: both reference and candidate have the same
/// frequency mix. LSD should be ~ 0.
#[test]
fn test_multi_frequency_identical() {
    let n = 8192;
    let sr = 16000.0;

    let reference: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            0.5 * (2.0 * PI * 440.0 * t).sin() + 0.3 * (2.0 * PI * 1000.0 * t).sin()
        })
        .collect();

    let config = SpectralConfig::default();
    let result =
        compare_spectral(&reference, &reference, &config).expect("comparison should succeed");

    assert!(result.log_spectral_distance_db < 1e-6);
    assert!(result.spectral_convergence < 1e-6);
    assert!(result.passed);
}
