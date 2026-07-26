// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for quality metrics: MCD, HNR, F0 extraction, spectral tilt.

use nn_tts_verify::quality;

fn sine_wave(freq: f64, sample_rate: u32, duration_sec: f64, amplitude: f32) -> Vec<f32> {
    let n = (f64::from(sample_rate) * duration_sec) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(sample_rate);
            amplitude * (2.0 * std::f64::consts::PI * freq * t).sin() as f32
        })
        .collect()
}

fn harmonic_signal(sample_rate: u32, duration_sec: f64) -> Vec<f32> {
    let n = (f64::from(sample_rate) * duration_sec) as usize;
    let sr = f64::from(sample_rate);
    let mut signal = vec![0.0_f32; n];
    for (i, sample) in signal.iter_mut().enumerate() {
        let t = i as f64 / sr;
        *sample = (0.4 * (2.0 * std::f64::consts::PI * 200.0 * t).sin()
            + 0.2 * (2.0 * std::f64::consts::PI * 400.0 * t).sin()
            + 0.1 * (2.0 * std::f64::consts::PI * 800.0 * t).sin()) as f32;
    }
    signal
}

// -- MCD tests ---------------------------------------------------------------

#[test]
fn test_mcd_identical_signals() {
    let signal = harmonic_signal(16000, 0.5);
    let result = quality::compute_mcd(&signal, &signal, 16000, 6.0).unwrap();
    assert!(
        result.value < 0.5,
        "MCD of identical signals should be near 0, got {}",
        result.value
    );
    assert!(result.passed);
    assert_eq!(result.citation, "Kubichek 1993, IEEE ICASSP");
}

#[test]
fn test_mcd_different_signals() {
    let a = sine_wave(200.0, 16000, 0.5, 0.3);
    let b = sine_wave(500.0, 16000, 0.5, 0.3);
    let result = quality::compute_mcd(&a, &b, 16000, 6.0).unwrap();
    assert!(
        result.value > 1.0,
        "Different signals should have significant MCD, got {}",
        result.value
    );
}

#[test]
fn test_mcd_empty_input() {
    let result = quality::compute_mcd(&[], &[], 16000, 6.0);
    assert!(result.is_err());
}

#[test]
fn test_mcd_length_mismatch() {
    let a = sine_wave(200.0, 16000, 0.5, 0.3);
    let b = sine_wave(200.0, 16000, 1.0, 0.3);
    let result = quality::compute_mcd(&a, &b, 16000, 6.0);
    assert!(result.is_err());
}

// -- HNR tests ---------------------------------------------------------------

#[test]
fn test_hnr_pure_tone_high() {
    let signal = sine_wave(200.0, 16000, 0.5, 0.5);
    let result = quality::compute_hnr(&signal, 16000, 15.0).unwrap();
    assert!(
        result.value > 10.0,
        "Pure tone should have high HNR, got {}",
        result.value
    );
    assert_eq!(result.citation, "Boersma 1993, IFA Proceedings");
}

#[test]
fn test_hnr_empty() {
    let result = quality::compute_hnr(&[], 16000, 15.0);
    assert!(result.is_err());
}

// -- F0 extraction tests -----------------------------------------------------

#[test]
fn test_f0_extraction_220hz() {
    let signal = sine_wave(220.0, 16000, 0.5, 0.5);
    let f0 = quality::extract_f0(&signal, 16000).unwrap();
    assert!(!f0.is_empty(), "Should extract F0 frames");
    let voiced: Vec<f64> = f0.iter().copied().filter(|&f| f > 0.0).collect();
    assert!(!voiced.is_empty(), "Should have voiced frames");
    let mean: f64 = voiced.iter().sum::<f64>() / voiced.len() as f64;
    assert!(
        (mean - 220.0).abs() < 30.0,
        "Mean F0 should be ~220 Hz, got {mean}"
    );
}

#[test]
fn test_f0_extraction_empty() {
    let result = quality::extract_f0(&[], 16000);
    assert!(result.is_err());
}

// -- F0 range tests ----------------------------------------------------------

#[test]
fn test_f0_range_in_range() {
    let f0 = vec![150.0, 160.0, 170.0, 180.0, 0.0]; // One unvoiced frame.
    let result = quality::check_f0_range(&f0, 80.0, 400.0);
    assert!(result.passed, "F0 150-180 Hz should pass [80, 400] range");
    assert_eq!(result.citation, "Titze 1994, Prentice Hall");
}

#[test]
fn test_f0_range_out_of_range() {
    let f0 = vec![50.0, 55.0, 60.0, 45.0]; // All below 80 Hz.
    let result = quality::check_f0_range(&f0, 80.0, 400.0);
    assert!(!result.passed, "F0 45-60 Hz should fail [80, 400] range");
}

#[test]
fn test_f0_range_all_unvoiced() {
    let f0 = vec![0.0, 0.0, 0.0];
    let result = quality::check_f0_range(&f0, 80.0, 400.0);
    assert!(!result.passed, "All unvoiced should fail F0 range check");
}

// -- Spectral tilt tests -----------------------------------------------------

#[test]
fn test_spectral_tilt_harmonic_signal() {
    let signal = harmonic_signal(16000, 1.0);
    let result = quality::compute_spectral_tilt(&signal, 16000, (-20.0, -1.0)).unwrap();
    assert!(
        result.value < 0.0,
        "Harmonic signal should have negative tilt, got {}",
        result.value
    );
    assert_eq!(result.citation, "Fant 1960, Mouton");
}

#[test]
fn test_spectral_tilt_empty() {
    let result = quality::compute_spectral_tilt(&[], 16000, (-12.0, -3.0));
    assert!(result.is_err());
}
