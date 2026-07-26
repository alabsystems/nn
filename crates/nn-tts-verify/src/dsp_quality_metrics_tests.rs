// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for audio quality metric functions.

use super::*;

#[test]
fn test_cosine_similarity_identical() {
    let a = [1.0f32, 2.0, 3.0];
    let sim = cosine_similarity(&a, &a).unwrap();
    assert!(
        (sim - 1.0).abs() < 1e-10,
        "identical vectors → 1.0, got {sim}"
    );
}

#[test]
fn test_cosine_similarity_opposite() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [-1.0f32, -2.0, -3.0];
    let sim = cosine_similarity(&a, &b).unwrap();
    assert!(
        (sim - (-1.0)).abs() < 1e-10,
        "opposite vectors → -1.0, got {sim}"
    );
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = [1.0f32, 0.0];
    let b = [0.0f32, 1.0];
    let sim = cosine_similarity(&a, &b).unwrap();
    assert!(sim.abs() < 1e-10, "orthogonal vectors → 0.0, got {sim}");
}

#[test]
fn test_cosine_similarity_zero_vector() {
    let a = [0.0f32, 0.0, 0.0];
    let b = [1.0f32, 2.0, 3.0];
    let sim = cosine_similarity(&a, &b).unwrap();
    assert_eq!(sim, 0.0, "zero vector → 0.0");
}

#[test]
fn test_cosine_similarity_empty() {
    let a: [f32; 0] = [];
    assert!(cosine_similarity(&a, &a).is_err());
}

#[test]
fn test_cosine_similarity_length_mismatch() {
    let a = [1.0f32, 2.0];
    let b = [1.0f32];
    assert!(cosine_similarity(&a, &b).is_err());
}

#[test]
fn test_snr_db_identical() {
    let a = [1.0f32, -0.5, 0.3];
    let snr = snr_db(&a, &a).unwrap();
    assert!(
        snr.is_infinite() && snr > 0.0,
        "identical → +Inf, got {snr}"
    );
}

#[test]
fn test_snr_db_noisy() {
    let reference = [1.0f32, 0.0, -1.0];
    let candidate = [1.1f32, 0.1, -0.9]; // 10% noise
    let snr = snr_db(&candidate, &reference).unwrap();
    assert!(snr > 0.0, "small noise → positive SNR, got {snr}");
    assert!(snr.is_finite(), "finite SNR");
}

#[test]
fn test_snr_db_silent_reference() {
    let reference = [0.0f32, 0.0, 0.0];
    let candidate = [1.0f32, 2.0, 3.0];
    let snr = snr_db(&candidate, &reference).unwrap();
    assert_eq!(snr, 0.0, "silent reference → 0 dB");
}

#[test]
fn test_sdr_db_identical() {
    let a = [1.0f32, -0.5, 0.3];
    let sdr = sdr_db(&a, &a).unwrap();
    assert!(
        sdr.is_infinite() && sdr > 0.0,
        "identical → +Inf, got {sdr}"
    );
}

#[test]
fn test_sdr_db_scaled() {
    // Scaled version has no distortion in BSS sense (same direction).
    let reference = [1.0f32, 2.0, 3.0];
    let candidate = [2.0f32, 4.0, 6.0]; // 2x scale
    let sdr = sdr_db(&candidate, &reference).unwrap();
    assert!(
        sdr.is_infinite() && sdr > 0.0,
        "scaled copy → +Inf SDR, got {sdr}"
    );
}

#[test]
fn test_sdr_db_noisy() {
    let reference = [1.0f32, 0.0, -1.0];
    let candidate = [1.1f32, 0.05, -0.95];
    let sdr = sdr_db(&candidate, &reference).unwrap();
    assert!(sdr > 0.0, "small distortion → positive SDR, got {sdr}");
    assert!(sdr.is_finite(), "finite SDR");
}

#[test]
fn test_sdr_db_empty() {
    let a: [f32; 0] = [];
    assert!(sdr_db(&a, &a).is_err());
}

#[test]
fn test_cosine_similarity_scalar_positive() {
    let sim = cosine_similarity_scalar(3.0, 4.0);
    assert!((sim - 1.0).abs() < 1e-10, "same sign → 1.0, got {sim}");
}

#[test]
fn test_cosine_similarity_scalar_negative() {
    let sim = cosine_similarity_scalar(3.0, -4.0);
    assert!(
        (sim - (-1.0)).abs() < 1e-10,
        "opposite sign → -1.0, got {sim}"
    );
}

#[test]
fn test_cosine_similarity_scalar_zero() {
    assert_eq!(cosine_similarity_scalar(0.0, 5.0), 0.0);
    assert_eq!(cosine_similarity_scalar(5.0, 0.0), 0.0);
    assert_eq!(cosine_similarity_scalar(0.0, 0.0), 0.0);
}

#[test]
fn test_snr_scalar_values() {
    let snr = snr_scalar(1.0, 0.1);
    // SNR = 10*log10(1.0/0.01) = 10*log10(100) = 20 dB
    assert!((snr - 20.0).abs() < 1e-6, "expected 20 dB, got {snr}");
}

#[test]
fn test_rms_scalar_non_negative() {
    assert!(rms_scalar(0.0) >= 0.0);
    assert!(rms_scalar(1.0) >= 0.0);
    assert!(rms_scalar(-1.0) >= 0.0);
    assert!((rms_scalar(3.0) - 3.0).abs() < 1e-10);
}

#[test]
fn test_power_to_db() {
    let db = power_to_db(100.0);
    assert!((db - 20.0).abs() < 1e-10, "10*log10(100) = 20, got {db}");
}

#[test]
fn test_hz_to_mel_monotonic() {
    let mel_0 = hz_to_mel(0.0);
    let mel_1k = hz_to_mel(1000.0);
    let mel_4k = hz_to_mel(4000.0);
    assert!(mel_0 < mel_1k, "hz_to_mel must be monotonic");
    assert!(mel_1k < mel_4k, "hz_to_mel must be monotonic");
}

#[test]
fn test_mel_roundtrip() {
    let f = 1000.0;
    let roundtrip = mel_to_hz(hz_to_mel(f));
    assert!(
        (roundtrip - f).abs() < 1e-6,
        "mel roundtrip: expected {f}, got {roundtrip}"
    );
}

#[test]
fn test_hz_to_mel_zero() {
    let mel = hz_to_mel(0.0);
    assert!(mel.abs() < 1e-10, "mel(0 Hz) should be ~0, got {mel}");
}
