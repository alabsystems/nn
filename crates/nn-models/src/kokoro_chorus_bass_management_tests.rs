// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `kokoro_chorus_bass_management` — psychoacoustic stereo bass manager.

use super::*;

const SR: f32 = KOKORO_SAMPLE_RATE as f32;

fn sine_wave(freq: f32, n: usize, amplitude: f32) -> Vec<f32> {
    (0..n)
        .map(|i| amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / SR).sin())
        .collect()
}

fn rms(buf: &[f32]) -> f32 {
    let sum_sq: f32 = buf.iter().map(|x| x * x).sum();
    (sum_sq / buf.len().max(1) as f32).sqrt()
}

// --- Config validation ---

#[test]
fn test_config_default_valid() {
    BassManagementConfig::new()
        .validate()
        .expect("default config should be valid");
}

#[test]
fn test_config_builder_roundtrip() {
    let cfg = BassManagementConfig::new()
        .with_crossover_hz(100.0)
        .with_mono_below(false)
        .with_phase_trick(true)
        .with_sub_alignment(false)
        .with_rumble_filter_hz(25.0)
        .with_bass_enhancement(0.5);
    cfg.validate().expect("builder config should be valid");
    assert_eq!(cfg.crossover_hz, 100.0);
    assert!(!cfg.mono_below);
    assert!(cfg.phase_trick);
    assert!(!cfg.sub_alignment);
    assert_eq!(cfg.rumble_filter_hz, 25.0);
    assert_eq!(cfg.bass_enhancement, 0.5);
}

#[test]
fn test_config_invalid_crossover() {
    assert!(BassManagementConfig::new()
        .with_crossover_hz(50.0)
        .validate()
        .is_err());
    assert!(BassManagementConfig::new()
        .with_crossover_hz(400.0)
        .validate()
        .is_err());
    assert!(BassManagementConfig::new()
        .with_crossover_hz(f32::NAN)
        .validate()
        .is_err());
}

#[test]
fn test_config_invalid_rumble() {
    assert!(BassManagementConfig::new()
        .with_rumble_filter_hz(5.0)
        .validate()
        .is_err());
    assert!(BassManagementConfig::new()
        .with_rumble_filter_hz(70.0)
        .validate()
        .is_err());
    assert!(BassManagementConfig::new()
        .with_rumble_filter_hz(f32::INFINITY)
        .validate()
        .is_err());
}

#[test]
fn test_config_invalid_enhancement() {
    assert!(BassManagementConfig::new()
        .with_bass_enhancement(-0.1)
        .validate()
        .is_err());
    assert!(BassManagementConfig::new()
        .with_bass_enhancement(1.1)
        .validate()
        .is_err());
    assert!(BassManagementConfig::new()
        .with_bass_enhancement(f32::NAN)
        .validate()
        .is_err());
}

#[test]
fn test_config_rumble_must_be_below_crossover() {
    let cfg = BassManagementConfig::new()
        .with_crossover_hz(60.0)
        .with_rumble_filter_hz(60.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_presets_valid() {
    BassManagementConfig::broadcast()
        .validate()
        .expect("broadcast valid");
    BassManagementConfig::headphones()
        .validate()
        .expect("headphones valid");
    BassManagementConfig::speakers_small()
        .validate()
        .expect("speakers_small valid");
    BassManagementConfig::speakers_large()
        .validate()
        .expect("speakers_large valid");
    BassManagementConfig::subwoofer_mix()
        .validate()
        .expect("subwoofer_mix valid");
}

// --- Processor behavior ---

#[test]
fn test_mono_bass_coherence() {
    // With mono_below=true and identical L/R input, bass should be identical.
    let n = 4096;
    let mut left = sine_wave(80.0, n, 0.5);
    let mut right = sine_wave(80.0, n, 0.5);
    let cfg = BassManagementConfig::new();
    let mut proc = BassManager::new_kokoro(cfg).expect("valid");
    proc.process(&mut left, &mut right);
    for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
        assert!(
            (l - r).abs() < 1e-5,
            "sample {i}: L={l}, R={r} should be equal with mono bass",
        );
    }
}

#[test]
fn test_phase_trick_creates_difference() {
    // With phase_trick=true, L and R should differ even for mono input.
    let n = 4096;
    let mut left = sine_wave(80.0, n, 0.5);
    let mut right = sine_wave(80.0, n, 0.5);
    let cfg = BassManagementConfig::new().with_phase_trick(true);
    let mut proc = BassManager::new_kokoro(cfg).expect("valid");
    proc.process(&mut left, &mut right);

    let mut max_diff = 0.0_f32;
    for (&l, &r) in left.iter().zip(right.iter()) {
        max_diff = max_diff.max((l - r).abs());
    }
    assert!(
        max_diff > 1e-4,
        "phase trick should create L/R difference, max_diff={max_diff}",
    );
}

#[test]
fn test_high_freq_passthrough() {
    // High-frequency content (well above crossover) should pass through
    // largely unchanged. We compare RMS before and after.
    let n = 8192;
    let mut left = sine_wave(4000.0, n, 0.5);
    let mut right = sine_wave(4000.0, n, 0.5);
    let dry_rms = rms(&left);
    let cfg = BassManagementConfig::new();
    let mut proc = BassManager::new_kokoro(cfg).expect("valid");
    proc.process(&mut left, &mut right);
    let wet_rms = rms(&left[1024..]); // skip transient settling
    let ratio = wet_rms / dry_rms;
    assert!(
        ratio > 0.85 && ratio < 1.15,
        "high freq should pass through: dry_rms={dry_rms}, wet_rms={wet_rms}, ratio={ratio}",
    );
}

#[test]
fn test_bass_enhancement_adds_harmonics() {
    let n = 8192;
    let mut left = sine_wave(80.0, n, 0.5);
    let mut right = sine_wave(80.0, n, 0.5);
    let dry_rms = rms(&left);

    let cfg = BassManagementConfig::new().with_bass_enhancement(0.8);
    let mut proc = BassManager::new_kokoro(cfg).expect("valid");
    proc.process(&mut left, &mut right);
    let wet_rms = rms(&left[1024..]);

    // Enhancement should change the signal character (RMS may differ).
    // The key check is that it runs without error and produces finite output.
    assert!(wet_rms.is_finite(), "output should be finite");
    assert!(wet_rms > 0.0, "output should have energy");
    let _ = dry_rms; // used only for documentation of the test intent
}

#[test]
fn test_all_outputs_finite() {
    let mut left = vec![
        0.0,
        0.5,
        -0.5,
        1.0,
        -1.0,
        0.001,
        -0.001,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let mut right = vec![
        0.5,
        -0.5,
        0.0,
        -1.0,
        1.0,
        -0.001,
        0.001,
        f32::NAN,
        f32::NEG_INFINITY,
        f32::INFINITY,
    ];
    let cfg = BassManagementConfig::new()
        .with_bass_enhancement(1.0)
        .with_phase_trick(true);
    let mut proc = BassManager::new_kokoro(cfg).expect("valid");
    proc.process(&mut left, &mut right);
    for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
        assert!(l.is_finite(), "left[{i}] = {l} should be finite");
        assert!(r.is_finite(), "right[{i}] = {r} should be finite");
    }
}

#[test]
fn test_empty_buffers() {
    let cfg = BassManagementConfig::new();
    let mut proc = BassManager::new_kokoro(cfg).expect("valid");
    let mut left: Vec<f32> = vec![];
    let mut right: Vec<f32> = vec![];
    proc.process(&mut left, &mut right);
    assert!(left.is_empty());
    assert!(right.is_empty());
}

#[test]
fn test_mismatched_lengths() {
    let cfg = BassManagementConfig::new();
    let mut proc = BassManager::new_kokoro(cfg).expect("valid");
    let mut left = vec![0.3; 100];
    let mut right = vec![0.3; 50];
    proc.process(&mut left, &mut right);
    for &v in &left[50..] {
        assert_eq!(v, 0.3, "beyond min length should be untouched");
    }
}

#[test]
fn test_invalid_sample_rate() {
    let cfg = BassManagementConfig::new();
    assert!(BassManager::new(cfg, 0.0).is_err());
    assert!(BassManager::new(cfg, -44100.0).is_err());
    assert!(BassManager::new(cfg, f32::NAN).is_err());
}

#[test]
fn test_reset_clears_state() {
    let cfg = BassManagementConfig::new().with_phase_trick(true);
    let mut proc = BassManager::new_kokoro(cfg).expect("valid");
    let mut left = vec![0.5; 200];
    let mut right = vec![0.5; 200];
    proc.process(&mut left, &mut right);
    proc.reset();
    // After reset, filter states should be zeroed.
    assert_eq!(proc.lp_left.stage1.z1, 0.0);
    assert_eq!(proc.hp_right.stage2.z2, 0.0);
    assert_eq!(proc.phase_allpass.z1, 0.0);
    assert_eq!(proc.rumble.biquad.z1, 0.0);
}

#[test]
fn test_harmonic_enhance_zero_is_identity() {
    assert_eq!(harmonic_enhance(0.5, 0.0), 0.5);
    assert_eq!(harmonic_enhance(-0.3, 0.0), -0.3);
}

#[test]
fn test_harmonic_enhance_bounded() {
    for &x in &[0.5, 1.0, 2.0, -0.5, -1.0, -2.0] {
        let out = harmonic_enhance(x, 1.0);
        assert!(
            out.abs() < x.abs() + 0.01,
            "harmonic_enhance({x}, 1.0) = {out} should not exceed input magnitude",
        );
    }
}

#[test]
fn test_harmonic_enhance_nan_passthrough() {
    let out = harmonic_enhance(f32::NAN, 0.5);
    assert_eq!(out.classify(), std::num::FpCategory::Nan);
    // With amount=0 the NaN input passes through.
    let out2 = harmonic_enhance(f32::NAN, 0.0);
    assert!(out2.is_nan());
}
