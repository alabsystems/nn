// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for hard bound checks: non-silence, clipping, DC offset, clicks,
//! duration, spectral coverage, Nyquist.

use nn_tts_verify::bounds::{self, SpectralCoverageConfig};

fn sine_wave(freq: f64, sample_rate: u32, duration_sec: f64, amplitude: f32) -> Vec<f32> {
    let n = (f64::from(sample_rate) * duration_sec) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(sample_rate);
            amplitude * (2.0 * std::f64::consts::PI * freq * t).sin() as f32
        })
        .collect()
}

// -- Non-silence tests -------------------------------------------------------

#[test]
fn test_non_silence_pass() {
    let signal = sine_wave(440.0, 16000, 0.5, 0.3);
    let result = bounds::check_non_silence(&signal, 0.01);
    assert!(result.passed, "Audible tone should pass non-silence");
    assert!(
        result.value > 0.1,
        "RMS should be > 0.1, got {}",
        result.value
    );
}

#[test]
fn test_non_silence_fail() {
    let silence = vec![0.0_f32; 16000];
    let result = bounds::check_non_silence(&silence, 0.01);
    assert!(!result.passed, "Silence should fail non-silence check");
}

// -- Clipping tests ----------------------------------------------------------

#[test]
fn test_no_clipping_pass() {
    let signal = sine_wave(440.0, 16000, 0.5, 0.5);
    let result = bounds::check_no_clipping(&signal, 1.0);
    assert!(result.passed, "Signal within [-1,1] should pass clipping");
}

#[test]
fn test_no_clipping_fail() {
    let signal = sine_wave(440.0, 16000, 0.5, 1.5);
    let result = bounds::check_no_clipping(&signal, 1.0);
    assert!(
        !result.passed,
        "Signal with amplitude 1.5 should fail clipping"
    );
}

// -- DC offset tests ---------------------------------------------------------

#[test]
fn test_no_dc_offset_pass() {
    let signal = sine_wave(440.0, 16000, 0.5, 0.3);
    let result = bounds::check_no_dc_offset(&signal, 0.05);
    assert!(result.passed, "Zero-mean sine should pass DC offset check");
}

#[test]
fn test_no_dc_offset_fail() {
    let signal: Vec<f32> = (0..16000).map(|_| 0.1).collect();
    let result = bounds::check_no_dc_offset(&signal, 0.05);
    assert!(
        !result.passed,
        "Constant 0.1 signal should fail DC offset > 0.05"
    );
}

// -- Click tests -------------------------------------------------------------

#[test]
fn test_no_clicks_pass() {
    let signal = sine_wave(100.0, 16000, 0.5, 0.3);
    let result = bounds::check_no_clicks(&signal, 0.5);
    assert!(result.passed, "Smooth sine should have no clicks");
}

#[test]
fn test_no_clicks_fail() {
    let mut signal = vec![0.0_f32; 1000];
    signal[500] = 0.9; // Sharp transient.
    let result = bounds::check_no_clicks(&signal, 0.5);
    assert!(!result.passed, "Sharp transient should fail click check");
}

// -- Duration tests ----------------------------------------------------------

#[test]
fn test_duration_pass() {
    let signal = sine_wave(440.0, 16000, 1.0, 0.3);
    let result = bounds::check_duration(&signal, 16000, 0.1, 300.0);
    assert!(
        result.passed,
        "1-second signal should pass duration [0.1, 300]"
    );
    assert!((result.value - 1.0).abs() < 0.01);
}

#[test]
fn test_duration_too_short() {
    let signal = sine_wave(440.0, 16000, 0.01, 0.3);
    let result = bounds::check_duration(&signal, 16000, 0.1, 300.0);
    assert!(!result.passed, "10ms signal should fail min_duration=0.1");
}

#[test]
fn test_duration_too_long() {
    // Simulate very long (we can't create 300s of audio, test with low max).
    let signal = sine_wave(440.0, 16000, 2.0, 0.3);
    let result = bounds::check_duration(&signal, 16000, 0.1, 1.0);
    assert!(!result.passed, "2s signal should fail max_duration=1.0");
}

// -- Spectral coverage tests -------------------------------------------------

#[test]
fn test_spectral_coverage_broadband_pass() {
    // Multi-harmonic signal has energy across bands.
    let n = 16000;
    let mut signal = vec![0.0_f32; n];
    for (i, sample) in signal.iter_mut().enumerate() {
        let t = i as f64 / 16000.0;
        *sample = (0.3 * (2.0 * std::f64::consts::PI * 200.0 * t).sin()
            + 0.2 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin()
            + 0.1 * (2.0 * std::f64::consts::PI * 3000.0 * t).sin()
            + 0.05 * (2.0 * std::f64::consts::PI * 6000.0 * t).sin()) as f32;
    }
    let config = SpectralCoverageConfig::default();
    let result = bounds::check_spectral_coverage(&signal, 16000, &config).unwrap();
    assert!(
        result.passed,
        "Broadband signal should pass spectral coverage"
    );
}

#[test]
fn test_spectral_coverage_single_tone_may_fail() {
    // Very narrow-band signal — only 1 frequency.
    let signal = sine_wave(440.0, 16000, 0.5, 0.3);
    let mut config = SpectralCoverageConfig::default();
    config.n_bands = 8;
    config.min_energy_db = -40.0; // Strict threshold.
    config.min_coverage = 0.75; // Need 75% of bands.
    let result = bounds::check_spectral_coverage(&signal, 16000, &config).unwrap();
    // Pure tone concentrates energy in one band; likely fails 75% coverage.
    assert!(
        !result.passed,
        "Single tone should fail strict spectral coverage"
    );
}

// -- Spectral coverage n_bands==0 guard tests --------------------------------

#[test]
fn test_spectral_coverage_zero_bands_returns_error() {
    let signal = sine_wave(440.0, 16000, 0.5, 0.3);
    let mut config = SpectralCoverageConfig::default();
    config.n_bands = 0;
    let result = bounds::check_spectral_coverage(&signal, 16000, &config);
    assert!(result.is_err(), "n_bands=0 should return error");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("n_bands"),
        "error should mention n_bands: {msg}"
    );
}

// -- Nyquist tests -----------------------------------------------------------

#[test]
fn test_nyquist_low_frequency_pass() {
    let signal = sine_wave(440.0, 16000, 0.5, 0.3);
    let result = bounds::check_nyquist(&signal, 16000).unwrap();
    assert!(
        result.passed,
        "440 Hz tone should have negligible Nyquist energy"
    );
}
