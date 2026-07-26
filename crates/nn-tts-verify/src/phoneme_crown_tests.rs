// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::phoneme_crown`] — CROWN-verified per-phoneme acoustic features.

use super::*;

// Default F0 base frequency used in Kokoro
const F0_BASE: f64 = 200.0;

#[test]
fn test_basic_proven_certificate() {
    // T=2 phonemes, output = [f0_0, f0_1, energy_0, energy_1]
    let lower = [0.1_f32, 0.2, 0.5, 0.6];
    let upper = [0.3_f32, 0.4, 0.8, 0.9];
    let cert = interpret_phoneme_features(&lower, &upper, 2, 1.0, 1.0, F0_BASE, "CROWN").unwrap();

    assert!(cert.is_proven);
    assert_eq!(cert.sequence_length, 2);
    assert_eq!(cert.f0_lower_hz.len(), 2);
    assert_eq!(cert.f0_upper_hz.len(), 2);
    assert_eq!(cert.energy_lower.len(), 2);
    assert_eq!(cert.energy_upper.len(), 2);

    // F0 Hz = exp(logit) * 200
    // Use f64::from(f32) to match the conversion in interpret_phoneme_features
    let expected_f0_lo_0 = f64::from(0.1_f32).exp() * F0_BASE;
    let expected_f0_hi_0 = f64::from(0.3_f32).exp() * F0_BASE;
    assert!((cert.f0_lower_hz[0] - expected_f0_lo_0).abs() < 1e-6);
    assert!((cert.f0_upper_hz[0] - expected_f0_hi_0).abs() < 1e-6);

    // Energy bounds are identity (no denormalization)
    assert!((cert.energy_lower[0] - 0.5).abs() < 1e-6);
    assert!((cert.energy_upper[0] - 0.8).abs() < 1e-6);
}

#[test]
fn test_non_finite_bounds_not_proven() {
    // One F0 logit is +Inf → vacuous bound → not proven
    let lower = [0.1_f32, f32::NEG_INFINITY, 0.5, 0.6];
    let upper = [0.3_f32, f32::INFINITY, 0.8, 0.9];
    let cert = interpret_phoneme_features(&lower, &upper, 2, 1.0, 1.0, F0_BASE, "IBP").unwrap();

    assert!(!cert.is_proven);
    assert_eq!(cert.propagation_mode, "IBP");
}

#[test]
fn test_nan_bounds_not_proven() {
    let lower = [f32::NAN, 0.2, 0.5, 0.6];
    let upper = [0.3_f32, 0.4, 0.8, 0.9];
    let cert = interpret_phoneme_features(&lower, &upper, 2, 1.0, 1.0, F0_BASE, "CROWN").unwrap();

    assert!(!cert.is_proven);
}

#[test]
fn test_energy_non_finite_not_proven() {
    let lower = [0.1_f32, 0.2, f32::NEG_INFINITY, 0.6];
    let upper = [0.3_f32, 0.4, f32::INFINITY, 0.9];
    let cert = interpret_phoneme_features(&lower, &upper, 2, 1.0, 1.0, F0_BASE, "CROWN").unwrap();

    assert!(!cert.is_proven);
}

#[test]
fn test_zero_sequence_length_error() {
    let err = interpret_phoneme_features(&[], &[], 0, 1.0, 1.0, F0_BASE, "CROWN");
    assert!(err.is_err());
    let msg = format!("{}", err.unwrap_err());
    assert!(msg.contains("sequence_length"));
}

#[test]
fn test_length_mismatch_error() {
    let lower = [0.1_f32, 0.2]; // 2 elements, but seq_len=2 needs 4
    let upper = [0.3_f32, 0.4];
    let err = interpret_phoneme_features(&lower, &upper, 2, 1.0, 1.0, F0_BASE, "CROWN");
    assert!(err.is_err());
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("sequence_length"),
        "error should reference sequence_length, got: {msg}"
    );
}

#[test]
fn test_invalid_base_freq_error() {
    let lower = [0.1_f32, 0.5];
    let upper = [0.3_f32, 0.8];
    let err = interpret_phoneme_features(&lower, &upper, 1, 1.0, 1.0, 0.0, "CROWN");
    assert!(err.is_err());

    let err2 = interpret_phoneme_features(&lower, &upper, 1, 1.0, 1.0, f64::NAN, "CROWN");
    assert!(err2.is_err());

    let err3 = interpret_phoneme_features(&lower, &upper, 1, 1.0, 1.0, f64::NEG_INFINITY, "CROWN");
    assert!(err3.is_err());
}

#[test]
fn test_single_phoneme() {
    let lower = [0.0_f32, 1.0];
    let upper = [1.0_f32, 2.0];
    let cert = interpret_phoneme_features(&lower, &upper, 1, 0.5, 0.5, F0_BASE, "CROWN").unwrap();

    assert!(cert.is_proven);
    assert_eq!(cert.sequence_length, 1);
    assert_eq!(cert.input_bound, 0.5);
    assert_eq!(cert.style_bound, 0.5);

    // F0: exp(0)*200 = 200, exp(1)*200 ≈ 543.656
    assert!((cert.f0_lower_hz[0] - F0_BASE).abs() < 1e-6);
    assert!((cert.f0_upper_hz[0] - F0_BASE * 1.0_f64.exp()).abs() < 1e-3);
}

#[test]
fn test_f0_range_hz_helper() {
    let lower = [0.0_f32, 0.5, 1.0, 1.5];
    let upper = [1.0_f32, 1.5, 2.0, 2.5];
    let cert = interpret_phoneme_features(&lower, &upper, 2, 1.0, 1.0, F0_BASE, "CROWN").unwrap();

    let range_0 = f0_range_hz(&cert, 0).unwrap();
    let expected = (1.0_f64.exp() - 0.0_f64.exp()) * F0_BASE;
    assert!((range_0 - expected).abs() < 1e-3);

    // Out of range
    assert!(f0_range_hz(&cert, 2).is_none());
}

#[test]
fn test_energy_range_helper() {
    let lower = [0.0_f32, 0.5, 1.0, 1.5];
    let upper = [1.0_f32, 1.5, 2.0, 2.5];
    let cert = interpret_phoneme_features(&lower, &upper, 2, 1.0, 1.0, F0_BASE, "CROWN").unwrap();

    let range_0 = energy_range(&cert, 0).unwrap();
    assert!((range_0 - 1.0).abs() < 1e-6); // 2.0 - 1.0

    let range_1 = energy_range(&cert, 1).unwrap();
    assert!((range_1 - 1.0).abs() < 1e-6); // 2.5 - 1.5

    assert!(energy_range(&cert, 2).is_none());
}

#[test]
fn test_max_f0_range() {
    let lower = [0.0_f32, -1.0, 0.5, 0.5];
    let upper = [1.0_f32, 1.0, 1.5, 1.5];
    let cert = interpret_phoneme_features(&lower, &upper, 2, 1.0, 1.0, F0_BASE, "CROWN").unwrap();

    let max_range = max_f0_range_hz(&cert);
    // Phoneme 1: exp(1)*200 - exp(-1)*200 = (e - 1/e)*200 ≈ 470.47
    // Phoneme 0: exp(1)*200 - exp(0)*200 = (e - 1)*200 ≈ 343.66
    let range_1 = (1.0_f64.exp() - (-1.0_f64).exp()) * F0_BASE;
    assert!((max_range - range_1).abs() < 1e-3);
}

#[test]
fn test_max_energy_range() {
    let lower = [0.0_f32, 0.0, 0.5, 1.0];
    let upper = [1.0_f32, 1.0, 2.0, 2.5];
    let cert = interpret_phoneme_features(&lower, &upper, 2, 1.0, 1.0, F0_BASE, "CROWN").unwrap();

    let max_range = max_energy_range(&cert);
    // Phoneme 0: 2.0 - 0.5 = 1.5; Phoneme 1: 2.5 - 1.0 = 1.5
    assert!((max_range - 1.5).abs() < 1e-6);
}

#[test]
fn test_large_f0_logits_overflow() {
    // Very large logits → exp overflows f64 to Inf → not proven
    // exp(710) ≈ 2.23e308 → finite; exp(711) → Inf
    let lower = [710.0_f32, 0.5];
    let upper = [800.0_f32, 1.5];
    let cert = interpret_phoneme_features(&lower, &upper, 1, 1.0, 1.0, F0_BASE, "IBP").unwrap();

    assert!(!cert.is_proven); // exp(800)*200 overflows f64
}
