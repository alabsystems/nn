// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the auto-EQ spectral analysis processor.

use super::*;

const TEST_SR: f32 = 24000.0;

fn sine_wave(freq: f32, sr: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (std::f32::consts::TAU * freq * i as f32 / sr).sin())
        .collect()
}

fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    (signal.iter().map(|s| s * s).sum::<f32>() / signal.len() as f32).sqrt()
}

#[test]
fn test_config_defaults_valid() {
    assert!(AutoEqConfig::new().validate().is_ok());
}

#[test]
fn test_config_rejects_bad_window() {
    assert!(AutoEqConfig::new()
        .with_analysis_window(100)
        .validate()
        .is_err());
    assert!(AutoEqConfig::new()
        .with_analysis_window(256)
        .validate()
        .is_err());
    assert!(AutoEqConfig::new()
        .with_analysis_window(8192)
        .validate()
        .is_err());
    assert!(AutoEqConfig::new()
        .with_analysis_window(512)
        .validate()
        .is_ok());
    assert!(AutoEqConfig::new()
        .with_analysis_window(4096)
        .validate()
        .is_ok());
}

#[test]
fn test_config_rejects_bad_strength() {
    assert!(AutoEqConfig::new()
        .with_correction_strength(-0.1)
        .validate()
        .is_err());
    assert!(AutoEqConfig::new()
        .with_correction_strength(1.1)
        .validate()
        .is_err());
    assert!(AutoEqConfig::new()
        .with_correction_strength(f32::NAN)
        .validate()
        .is_err());
}

#[test]
fn test_config_rejects_bad_boost_cut() {
    assert!(AutoEqConfig::new()
        .with_max_boost_db(-1.0)
        .validate()
        .is_err());
    assert!(AutoEqConfig::new()
        .with_max_boost_db(25.0)
        .validate()
        .is_err());
    assert!(AutoEqConfig::new()
        .with_max_cut_db(-1.0)
        .validate()
        .is_err());
    assert!(AutoEqConfig::new()
        .with_max_cut_db(49.0)
        .validate()
        .is_err());
}

#[test]
fn test_processor_creation() {
    let config = AutoEqConfig::new();
    let proc = AutoEqProcessor::new(&config, TEST_SR);
    assert!(proc.is_ok());
    let proc = proc.unwrap();
    // Nyquist = 12000 Hz, so bands up to 10000 Hz are included.
    assert!(!proc.band_frequencies().is_empty());
    assert!(proc.band_frequencies().last().unwrap() < &12000.0);
}

#[test]
fn test_processor_rejects_bad_sample_rate() {
    let config = AutoEqConfig::new();
    assert!(AutoEqProcessor::new(&config, 0.0).is_err());
    assert!(AutoEqProcessor::new(&config, -1.0).is_err());
    assert!(AutoEqProcessor::new(&config, f32::NAN).is_err());
}

#[test]
fn test_analyze_spectrum_detects_tone() {
    let config = AutoEqConfig::new().with_analysis_window(2048);
    let proc = AutoEqProcessor::new(&config, TEST_SR).unwrap();

    // Generate a 1 kHz sine wave.
    let audio = sine_wave(1000.0, TEST_SR, 4096);
    let band_db = proc.analyze_spectrum(&audio);

    // Find the 1 kHz band.
    let idx_1k = proc
        .band_frequencies()
        .iter()
        .position(|&f| (f - 1000.0).abs() < 1.0);
    assert!(idx_1k.is_some(), "1 kHz band should exist");
    let idx = idx_1k.unwrap();

    // The 1 kHz band should have significantly more energy than distant bands.
    let low_band_db = band_db[0]; // 20 Hz band
    assert!(
        band_db[idx] > low_band_db + 10.0,
        "1 kHz band ({} dB) should be >10 dB above 20 Hz band ({} dB)",
        band_db[idx],
        low_band_db,
    );
}

#[test]
fn test_analyze_and_correct_produces_finite_output() {
    let config = AutoEqConfig::new()
        .with_target_curve(TargetCurve::Speech)
        .with_correction_strength(1.0)
        .with_max_boost_db(6.0)
        .with_max_cut_db(12.0);
    let mut proc = AutoEqProcessor::new(&config, TEST_SR).unwrap();

    let mut audio = sine_wave(1000.0, TEST_SR, 4096);
    let original_rms = rms(&audio);
    proc.analyze_and_correct(&mut audio, TEST_SR);
    let corrected_rms = rms(&audio);

    assert!(corrected_rms.is_finite(), "output must be finite");
    let _ = original_rms; // informational
}

#[test]
fn test_disabled_is_passthrough() {
    let config = AutoEqConfig::new().with_enabled(false);
    let mut proc = AutoEqProcessor::new(&config, TEST_SR).unwrap();

    let mut audio = sine_wave(440.0, TEST_SR, 2048);
    let original = audio.clone();
    proc.analyze_and_correct(&mut audio, TEST_SR);

    assert_eq!(audio, original, "disabled auto-EQ should be passthrough");
}

#[test]
fn test_reset_clears_state() {
    let config = AutoEqConfig::new();
    let mut proc = AutoEqProcessor::new(&config, TEST_SR).unwrap();

    let mut audio = sine_wave(1000.0, TEST_SR, 4096);
    proc.analyze_and_correct(&mut audio, TEST_SR);

    proc.reset();
    assert!(proc.band_gains().iter().all(|&g| g == 0.0));
}

#[test]
fn test_target_curve_flat() {
    let curve = TargetCurve::Flat;
    let vals = curve.evaluate();
    assert!(vals.iter().all(|&v| v == 0.0));
}

#[test]
fn test_target_curve_speech_has_warmth() {
    let curve = TargetCurve::Speech;
    let vals = curve.evaluate();
    assert!(vals[8] > 0.0, "Speech curve should boost 125 Hz");
    assert!(vals[30] < 0.0, "Speech curve should roll off 20 kHz");
}

#[test]
fn test_target_curve_custom_interpolation() {
    let curve = TargetCurve::Custom(vec![(100.0, -3.0), (1000.0, 0.0), (10000.0, 3.0)]);
    let vals = curve.evaluate();
    let idx_100 = 7; // THIRD_OCTAVE_CENTERS[7] = 100.0
    assert!(
        (vals[idx_100] - (-3.0)).abs() < 0.01,
        "Custom at 100 Hz = {}, expected -3.0",
        vals[idx_100],
    );
}

#[test]
fn test_fft_roundtrip_basic() {
    let n = 512;
    let mut data: Vec<(f32, f32)> = (0..n).map(|i| ((i as f32 * 0.3).sin(), 0.0)).collect();
    let original: Vec<(f32, f32)> = data.clone();

    fft(&mut data);
    // Inverse via conjugate + FFT + conjugate + scale.
    for (_, im) in data.iter_mut() {
        *im = -*im;
    }
    fft(&mut data);
    let scale = 1.0 / n as f32;
    for (re, im) in data.iter_mut() {
        *re *= scale;
        *im = -*im * scale;
    }

    for (i, (&orig, &recovered)) in original.iter().zip(data.iter()).enumerate() {
        assert!(
            (orig.0 - recovered.0).abs() < 1e-4,
            "sample {i}: re mismatch: {} vs {}",
            orig.0,
            recovered.0,
        );
    }
}

#[test]
fn test_nan_input_handled() {
    let config = AutoEqConfig::new().with_correction_strength(1.0);
    let mut proc = AutoEqProcessor::new(&config, TEST_SR).unwrap();

    let mut audio = vec![f32::NAN; 2048];
    proc.analyze_and_correct(&mut audio, TEST_SR);

    assert!(
        audio.iter().all(|s| s.is_finite()),
        "NaN input must produce finite output"
    );
}
