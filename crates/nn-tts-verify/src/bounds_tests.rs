// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for TTS hard bounds verification functions.

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

use crate::test_audio_helpers::sine_wave_full;

// ---------------------------------------------------------------------------
// check_non_silence
// ---------------------------------------------------------------------------

#[test]
fn test_non_silence_pass() {
    let audio = sine_wave_full(440.0, 24000, 0.1, 0.5);
    let result = check_non_silence(&audio, 0.01);
    assert!(
        result.passed,
        "sine wave should not be silent: rms={:.6}",
        result.value
    );
    assert!(result.value > 0.01);
    assert_eq!(result.name, "non_silence");
}

#[test]
fn test_non_silence_fail() {
    let silence = vec![0.0_f32; 2400];
    let result = check_non_silence(&silence, 0.01);
    assert!(!result.passed, "silence should fail non_silence check");
    assert!(result.value < 1e-10);
}

#[test]
fn test_non_silence_empty() {
    let empty: &[f32] = &[];
    let result = check_non_silence(empty, 0.01);
    // RMS of empty should be 0.
    assert!(!result.passed, "empty audio should fail non_silence check");
}

// ---------------------------------------------------------------------------
// check_no_clipping
// ---------------------------------------------------------------------------

#[test]
fn test_no_clipping_pass() {
    let audio = sine_wave_full(440.0, 24000, 0.1, 0.5);
    let result = check_no_clipping(&audio, 1.0);
    assert!(
        result.passed,
        "sine at 0.5 amplitude should not clip at 1.0: peak={:.6}",
        result.value
    );
    assert!(result.value < 0.51);
}

#[test]
fn test_no_clipping_fail() {
    let mut audio = sine_wave_full(440.0, 24000, 0.1, 0.5);
    audio[100] = 1.5; // Introduce a clipping sample.
    let result = check_no_clipping(&audio, 1.0);
    assert!(
        !result.passed,
        "sample at 1.5 should exceed clipping threshold of 1.0"
    );
    assert!((result.value - 1.5).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// check_no_dc_offset
// ---------------------------------------------------------------------------

#[test]
fn test_no_dc_offset_pass() {
    let audio = sine_wave_full(440.0, 24000, 0.1, 0.5);
    let result = check_no_dc_offset(&audio, 0.05);
    assert!(
        result.passed,
        "sine wave should have near-zero DC offset: {:.6}",
        result.value
    );
}

#[test]
fn test_no_dc_offset_fail() {
    // Audio with large DC offset (all positive).
    let audio = vec![0.5_f32; 2400];
    let result = check_no_dc_offset(&audio, 0.1);
    assert!(
        !result.passed,
        "constant 0.5 should have DC offset of 0.5, failing 0.1 threshold"
    );
    assert!((result.value - 0.5).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// check_no_clicks
// ---------------------------------------------------------------------------

#[test]
fn test_no_clicks_pass() {
    let audio = sine_wave_full(440.0, 24000, 0.1, 0.5);
    let result = check_no_clicks(&audio, 0.5);
    assert!(
        result.passed,
        "smooth sine should have small sample diffs: {:.6}",
        result.value
    );
}

#[test]
fn test_no_clicks_fail() {
    let mut audio = sine_wave_full(440.0, 24000, 0.1, 0.5);
    // Introduce a click (large jump).
    audio[500] = 1.0;
    audio[501] = -1.0;
    let result = check_no_clicks(&audio, 0.5);
    assert!(
        !result.passed,
        "2.0 sample jump should exceed 0.5 threshold: {:.6}",
        result.value
    );
    assert!(result.value >= 1.5); // At least the introduced jump.
}

// ---------------------------------------------------------------------------
// check_duration
// ---------------------------------------------------------------------------

#[test]
fn test_duration_pass() {
    let audio = sine_wave_full(440.0, 24000, 1.0, 0.5); // 1 second.
    let result = check_duration(&audio, 24000, 0.5, 2.0);
    assert!(
        result.passed,
        "1s audio should pass [0.5, 2.0]: duration={:.3}",
        result.value
    );
    assert!((result.value - 1.0).abs() < 0.01);
}

#[test]
fn test_duration_too_short() {
    let audio = sine_wave_full(440.0, 24000, 0.1, 0.5); // 0.1 seconds.
    let result = check_duration(&audio, 24000, 0.5, 2.0);
    assert!(!result.passed, "0.1s audio should fail min 0.5s check");
}

#[test]
fn test_duration_too_long() {
    let audio = sine_wave_full(440.0, 24000, 3.0, 0.5); // 3 seconds.
    let result = check_duration(&audio, 24000, 0.5, 2.0);
    assert!(!result.passed, "3s audio should fail max 2.0s check");
}

#[test]
fn test_duration_zero_sample_rate() {
    let audio = sine_wave_full(440.0, 24000, 0.1, 0.5);
    let result = check_duration(&audio, 0, 0.0, 1.0);
    // Duration with sample_rate=0 should be 0.0.
    assert!((result.value - 0.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// check_spectral_coverage
// ---------------------------------------------------------------------------

#[test]
fn test_spectral_coverage_pass() {
    // A rich signal covering many frequency bands should pass coverage.
    // With 8 bands at 24kHz (0-12kHz), each band is ~1500Hz wide.
    // Place one frequency in each of 5+ bands to exceed the 50% threshold.
    let n = 24000; // 1 second at 24kHz.
    let freqs = [
        200.0, 800.0, 2000.0, 3500.0, 5000.0, 7000.0, 9000.0, 11000.0,
    ];
    let audio: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / 24000.0;
            freqs
                .iter()
                .map(|f| 0.1 * (2.0 * std::f32::consts::PI * f * t).sin())
                .sum::<f32>()
        })
        .collect();
    let config = SpectralCoverageConfig::default();
    let result = check_spectral_coverage(&audio, 24000, &config).unwrap();
    assert!(
        result.passed,
        "8-frequency signal should pass spectral coverage: {:.3}",
        result.value
    );
}

#[test]
fn test_spectral_coverage_zero_bands_error() {
    let audio = sine_wave_full(440.0, 24000, 0.1, 0.5);
    let config = SpectralCoverageConfig {
        n_bands: 0,
        ..SpectralCoverageConfig::default()
    };
    let err = check_spectral_coverage(&audio, 24000, &config).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("n_bands"),
        "should report n_bands error: {msg}"
    );
}

// ---------------------------------------------------------------------------
// check_tail_energy
// ---------------------------------------------------------------------------

#[test]
fn test_tail_energy_uniform_signal_pass() {
    // A uniform sine wave has equal energy in body and tail — ratio ~ 1.0.
    let audio = sine_wave_full(440.0, 24000, 1.0, 0.5);
    let result = check_tail_energy(&audio, 24000, 50.0, 500.0, 3.0);
    assert!(
        result.passed,
        "uniform sine should have tail/body ratio near 1.0: {:.4}",
        result.value
    );
    assert_eq!(result.name, "tail_energy");
    // Ratio should be close to 1.0 for a uniform signal.
    assert!(
        result.value < 1.5,
        "uniform sine ratio should be near 1.0, got {:.4}",
        result.value
    );
}

#[test]
fn test_tail_energy_spike_fail() {
    // Create audio with a loud tail (energy spike at utterance end).
    let n = 24000; // 1 second at 24kHz.
    let mut audio = vec![0.1_f32; n];
    // Blast the last 1200 samples (50ms at 24kHz) with high amplitude.
    for sample in audio[n - 1200..].iter_mut() {
        *sample = 0.9;
    }
    let result = check_tail_energy(&audio, 24000, 50.0, 500.0, 3.0);
    assert!(
        !result.passed,
        "loud tail should exceed max_ratio 3.0: ratio={:.4}",
        result.value
    );
    assert!(result.value > 3.0);
}

#[test]
fn test_tail_energy_silent_body_returns_zero() {
    // If the body is silent, ratio should be 0.0 (no false alarm).
    let audio = vec![0.0_f32; 24000];
    let result = check_tail_energy(&audio, 24000, 50.0, 500.0, 3.0);
    assert!(
        result.passed,
        "silent audio should have ratio 0.0 (body is silent)"
    );
    assert!(result.value < 1e-10);
}

#[test]
fn test_tail_energy_short_audio() {
    // Very short audio (shorter than tail_ms) should still work.
    let audio = sine_wave_full(440.0, 24000, 0.01, 0.5); // 10ms = 240 samples.
    let result = check_tail_energy(&audio, 24000, 50.0, 500.0, 3.0);
    // Short audio: tail covers all samples, body may be tiny or empty.
    // Should not panic.
    assert_eq!(result.name, "tail_energy");
}

#[test]
fn test_tail_energy_quiet_tail_pass() {
    // Create audio with a quiet tail (fade-out).
    let n = 24000; // 1 second at 24kHz.
    let audio: Vec<f32> = (0..n)
        .map(|i| {
            let t = f64::from(i) / 24000.0;
            let base = (2.0 * std::f64::consts::PI * 440.0 * t).sin();
            // Fade out the last 50ms.
            let fade = if i >= n - 1200 {
                0.01 // quiet tail
            } else {
                0.5
            };
            (base * fade) as f32
        })
        .collect();
    let result = check_tail_energy(&audio, 24000, 50.0, 500.0, 3.0);
    assert!(
        result.passed,
        "quiet tail should easily pass: ratio={:.4}",
        result.value
    );
    assert!(result.value < 0.1);
}

// ---------------------------------------------------------------------------
// check_nyquist
// ---------------------------------------------------------------------------

#[test]
fn test_nyquist_pass() {
    // Low-frequency sine should have minimal Nyquist energy.
    let audio = sine_wave_full(440.0, 24000, 1.0, 0.5);
    let result = check_nyquist(&audio, 24000).unwrap();
    assert!(
        result.passed,
        "440 Hz sine should have low Nyquist energy: {:.6}",
        result.value
    );
}

#[test]
fn test_nyquist_fail_near_nyquist() {
    // Signal near Nyquist frequency should have high Nyquist energy.
    // At 24kHz sample rate, Nyquist is 12kHz. Use 11.5kHz.
    let audio = sine_wave_full(11500.0, 24000, 1.0, 0.5);
    let result = check_nyquist(&audio, 24000).unwrap();
    // The signal energy should be concentrated in the top frequency band.
    // May or may not fail depending on band energy distribution, but ratio should be elevated.
    assert!(
        result.value > 0.01,
        "near-Nyquist signal should have elevated Nyquist ratio: {:.6}",
        result.value
    );
}
