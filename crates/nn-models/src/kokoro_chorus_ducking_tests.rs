// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for spectral ducking and sidechain compression.

use super::*;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sine_wave(freq_hz: f32, amplitude: f32, duration_sec: f32) -> Vec<f32> {
    let sr = KOKORO_SAMPLE_RATE as f32;
    let n = (duration_sec * sr) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            amplitude * (2.0 * std::f32::consts::PI * freq_hz * t).sin()
        })
        .collect()
}

fn rms(signal: &[f32], skip: usize) -> f32 {
    let s = &signal[skip..];
    if s.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = s.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    (sum_sq / s.len() as f64).sqrt() as f32
}

// ---------------------------------------------------------------------------
// DuckingConfig validation
// ---------------------------------------------------------------------------

#[test]
fn test_ducking_config_default_is_valid() {
    DuckingConfig::default()
        .validate()
        .expect("default should be valid");
}

#[test]
fn test_ducking_config_builder() {
    let cfg = DuckingConfig::new(1)
        .with_duck_amount_db(-12.0)
        .with_attack_ms(10.0)
        .with_release_ms(200.0)
        .with_threshold_db(-40.0)
        .with_frequency_aware(true)
        .with_n_bands(6);
    cfg.validate().expect("builder config should be valid");
    assert_eq!(cfg.lead_voice_index, 1);
    assert!((cfg.duck_amount_db - (-12.0)).abs() < 1e-6);
    assert!(cfg.frequency_aware);
    assert_eq!(cfg.n_bands, 6);
}

#[test]
fn test_ducking_config_reject_bad_duck_amount() {
    let cfg = DuckingConfig::default().with_duck_amount_db(-25.0);
    assert!(cfg.validate().is_err());
    let cfg = DuckingConfig::default().with_duck_amount_db(1.0);
    assert!(cfg.validate().is_err());
    let cfg = DuckingConfig::default().with_duck_amount_db(f32::NAN);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_ducking_config_reject_bad_attack() {
    let cfg = DuckingConfig::default().with_attack_ms(0.5);
    assert!(cfg.validate().is_err());
    let cfg = DuckingConfig::default().with_attack_ms(100.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_ducking_config_reject_bad_release() {
    let cfg = DuckingConfig::default().with_release_ms(10.0);
    assert!(cfg.validate().is_err());
    let cfg = DuckingConfig::default().with_release_ms(600.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_ducking_config_reject_bad_threshold() {
    let cfg = DuckingConfig::default().with_threshold_db(-70.0);
    assert!(cfg.validate().is_err());
    let cfg = DuckingConfig::default().with_threshold_db(1.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_ducking_config_reject_bad_n_bands() {
    let cfg = DuckingConfig::default().with_n_bands(0);
    assert!(cfg.validate().is_err());
    let cfg = DuckingConfig::default().with_n_bands(9);
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// Broadband ducking
// ---------------------------------------------------------------------------

#[test]
fn test_broadband_ducking_reduces_non_lead() {
    let duration = 0.5;
    let lead = sine_wave(440.0, 0.5, duration);
    let backing = sine_wave(440.0, 0.5, duration);

    let config = DuckingConfig::new(0)
        .with_duck_amount_db(-12.0)
        .with_threshold_db(-20.0)
        .with_attack_ms(2.0)
        .with_release_ms(100.0);
    let mut ducker = SpectralDucker::new(&config, KOKORO_SAMPLE_RATE as f32).unwrap();

    let backing_original_rms = rms(&backing, 0);
    let mut voices = vec![lead, backing];
    ducker.process(&mut voices, &config).unwrap();

    // Lead (voice 0) should be unchanged.
    let lead_rms_after = rms(&voices[0], 2400);
    let lead_expected = rms(&sine_wave(440.0, 0.5, duration), 2400);
    let lead_ratio = lead_rms_after / lead_expected;
    assert!(
        lead_ratio > 0.95 && lead_ratio < 1.05,
        "lead should be unchanged, ratio = {lead_ratio}",
    );

    // Backing voice should be reduced.
    let skip = (KOKORO_SAMPLE_RATE as f32 * 0.15) as usize;
    let backing_rms_after = rms(&voices[1], skip);
    let reduction_ratio = backing_rms_after / backing_original_rms;
    // -12 dB ~ 0.25 linear. With smooth envelope, expect significant reduction.
    assert!(
        reduction_ratio < 0.85,
        "backing should be reduced, ratio = {reduction_ratio:.4}",
    );
}

#[test]
fn test_ducking_no_effect_when_lead_below_threshold() {
    let duration = 0.3;
    // Lead is very quiet — below threshold.
    let lead = sine_wave(440.0, 0.001, duration);
    let backing = sine_wave(440.0, 0.5, duration);

    let config = DuckingConfig::new(0)
        .with_duck_amount_db(-12.0)
        .with_threshold_db(-20.0);
    let mut ducker = SpectralDucker::new(&config, KOKORO_SAMPLE_RATE as f32).unwrap();

    let backing_original_rms = rms(&backing, 0);
    let mut voices = vec![lead, backing];
    ducker.process(&mut voices, &config).unwrap();

    let backing_rms_after = rms(&voices[1], 0);
    let ratio = backing_rms_after / backing_original_rms;
    // Should be nearly unchanged since lead is below threshold.
    assert!(
        ratio > 0.95,
        "backing should be unchanged when lead is quiet, ratio = {ratio:.4}",
    );
}

#[test]
fn test_ducking_empty_voices_ok() {
    let config = DuckingConfig::default();
    let mut ducker = SpectralDucker::new(&config, KOKORO_SAMPLE_RATE as f32).unwrap();
    let mut voices: Vec<Vec<f32>> = vec![];
    assert!(ducker.process(&mut voices, &config).is_ok());
}

#[test]
fn test_ducking_lead_index_out_of_bounds() {
    let config = DuckingConfig::new(5);
    let mut ducker = SpectralDucker::new(&config, KOKORO_SAMPLE_RATE as f32).unwrap();
    let mut voices = vec![vec![0.0; 100], vec![0.0; 100]];
    assert!(ducker.process(&mut voices, &config).is_err());
}

#[test]
fn test_ducking_mismatched_lengths() {
    let config = DuckingConfig::default();
    let mut ducker = SpectralDucker::new(&config, KOKORO_SAMPLE_RATE as f32).unwrap();
    let mut voices = vec![vec![0.0; 100], vec![0.0; 200]];
    assert!(ducker.process(&mut voices, &config).is_err());
}

// ---------------------------------------------------------------------------
// Frequency-aware ducking
// ---------------------------------------------------------------------------

#[test]
fn test_frequency_aware_ducks_lead_band_only() {
    let duration = 0.5;
    // Lead is a low-frequency tone (200 Hz).
    let lead = sine_wave(200.0, 0.5, duration);
    // Backing has both low (200 Hz) and high (8000 Hz) content.
    let n = lead.len();
    let sr = KOKORO_SAMPLE_RATE as f32;
    let backing: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            0.3 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 8000.0 * t).sin()
        })
        .collect();

    let config = DuckingConfig::new(0)
        .with_duck_amount_db(-15.0)
        .with_threshold_db(-20.0)
        .with_attack_ms(2.0)
        .with_release_ms(80.0)
        .with_frequency_aware(true)
        .with_n_bands(4);
    let mut ducker = SpectralDucker::new(&config, sr).unwrap();

    let backing_original_rms = rms(&backing, 0);
    let mut voices = vec![lead, backing];
    ducker.process(&mut voices, &config).unwrap();

    // With frequency-aware ducking, high-frequency content should be
    // less affected than broadband ducking. The overall RMS should still
    // be reduced (low band is ducked), but less than broadband.
    let skip = (sr * 0.15) as usize;
    let backing_rms_after = rms(&voices[1], skip);
    let ratio = backing_rms_after / backing_original_rms;
    // Some reduction expected (low band is ducked) but high band preserved.
    assert!(
        ratio < 0.95,
        "frequency-aware ducking should reduce some energy, ratio = {ratio:.4}",
    );
    // But not as much as full broadband -15 dB (0.178).
    assert!(
        ratio > 0.20,
        "frequency-aware should preserve high-band content, ratio = {ratio:.4}",
    );
}

#[test]
fn test_ducking_reset_clears_state() {
    let config = DuckingConfig::new(0)
        .with_frequency_aware(true)
        .with_n_bands(3);
    let mut ducker = SpectralDucker::new(&config, KOKORO_SAMPLE_RATE as f32).unwrap();

    // Process some audio.
    let lead = sine_wave(440.0, 0.5, 0.1);
    let backing = sine_wave(440.0, 0.5, 0.1);
    let mut voices = vec![lead, backing];
    ducker.process(&mut voices, &config).unwrap();

    // Reset and verify gains are back to 1.0.
    ducker.reset();
    for &g in &ducker.gain_state {
        assert!(
            (g - 1.0).abs() < 1e-6,
            "gain should be 1.0 after reset, got {g}"
        );
    }
}

// ---------------------------------------------------------------------------
// Sidechain compression
// ---------------------------------------------------------------------------

#[test]
fn test_sidechain_config_default_is_valid() {
    SidechainConfig::default()
        .validate()
        .expect("default should be valid");
}

#[test]
fn test_sidechain_config_reject_bad_params() {
    let cfg = SidechainConfig {
        duck_amount_db: -25.0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());

    let cfg = SidechainConfig {
        attack_ms: 0.0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());

    let cfg = SidechainConfig {
        release_ms: 10.0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());

    let cfg = SidechainConfig {
        threshold_db: 5.0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_sidechain_reduces_audio_when_sc_active() {
    let duration = 0.5;
    let sr = KOKORO_SAMPLE_RATE as f32;

    // Audio: constant-amplitude tone.
    let mut audio = sine_wave(440.0, 0.5, duration);
    // Sidechain: loud tone that triggers ducking.
    let sidechain = sine_wave(440.0, 0.5, duration);

    let original_rms = rms(&audio, 0);

    let config = SidechainConfig {
        duck_amount_db: -12.0,
        attack_ms: 2.0,
        release_ms: 100.0,
        threshold_db: -20.0,
    };
    apply_sidechain(&mut audio, &sidechain, &config).unwrap();

    let skip = (sr * 0.1) as usize;
    let processed_rms = rms(&audio, skip);
    let ratio = processed_rms / original_rms;
    assert!(
        ratio < 0.85,
        "sidechain should reduce audio, ratio = {ratio:.4}",
    );
}

#[test]
fn test_sidechain_no_effect_when_sc_quiet() {
    let duration = 0.3;
    let mut audio = sine_wave(440.0, 0.5, duration);
    // Sidechain: very quiet, below threshold.
    let sidechain = sine_wave(440.0, 0.001, duration);

    let original_rms = rms(&audio, 0);
    let config = SidechainConfig::default();
    apply_sidechain(&mut audio, &sidechain, &config).unwrap();

    let processed_rms = rms(&audio, 0);
    let ratio = processed_rms / original_rms;
    assert!(
        ratio > 0.95,
        "audio should be unchanged when sidechain is quiet, ratio = {ratio:.4}",
    );
}

#[test]
fn test_sidechain_mismatched_lengths() {
    let mut audio = vec![0.0; 100];
    let sidechain = vec![0.0; 200];
    let config = SidechainConfig::default();
    assert!(apply_sidechain(&mut audio, &sidechain, &config).is_err());
}

#[test]
fn test_sidechain_empty_buffers() {
    let mut audio: Vec<f32> = vec![];
    let sidechain: Vec<f32> = vec![];
    let config = SidechainConfig::default();
    assert!(apply_sidechain(&mut audio, &sidechain, &config).is_ok());
}

#[test]
fn test_sidechain_handles_nan_input() {
    let mut audio = vec![0.5, f32::NAN, 0.3, 0.4];
    let sidechain = vec![0.5, 0.5, f32::NAN, 0.5];
    let config = SidechainConfig::default();
    apply_sidechain(&mut audio, &sidechain, &config).unwrap();
    // NaN samples should be zeroed.
    assert!(audio[1] == 0.0 || audio[1].is_finite());
}

// ---------------------------------------------------------------------------
// Envelope follower
// ---------------------------------------------------------------------------

#[test]
fn test_envelope_follower_tracks_level() {
    let mut env = EnvelopeFollower::new(5.0, 150.0);
    // Feed silence -- should stay at floor.
    for _ in 0..100 {
        let db = env.process(0.0);
        assert!(db <= -100.0, "silence should give very low dB, got {db}");
    }
    // Feed a loud signal -- envelope should rise.
    for _ in 0..2400 {
        env.process(0.5);
    }
    let db = env.process(0.5);
    assert!(
        db > -20.0,
        "after loud signal, envelope should be above -20 dB, got {db}"
    );

    // Feed silence -- envelope should decay. 48000 samples = 2 seconds
    // at 24kHz, well beyond the 150ms release time constant.
    for _ in 0..48000 {
        env.process(0.0);
    }
    let db = env.process(0.0);
    assert!(
        db < -40.0,
        "after silence, envelope should decay below -40 dB, got {db}"
    );
}

#[test]
fn test_envelope_follower_reset() {
    let mut env = EnvelopeFollower::new(5.0, 150.0);
    for _ in 0..2400 {
        env.process(0.8);
    }
    env.reset();
    let db = env.process(0.0);
    assert!(db <= -100.0, "after reset, envelope should be at floor");
}

// ---------------------------------------------------------------------------
// Multi-voice scenario
// ---------------------------------------------------------------------------

#[test]
fn test_three_voice_ducking_lead_in_middle() {
    let duration = 0.4;
    let lead = sine_wave(440.0, 0.5, duration);
    let v0 = sine_wave(440.0, 0.4, duration);
    let v2 = sine_wave(880.0, 0.4, duration);

    let config = DuckingConfig::new(1) // lead is voice index 1
        .with_duck_amount_db(-10.0)
        .with_threshold_db(-20.0)
        .with_attack_ms(3.0)
        .with_release_ms(100.0);
    let mut ducker = SpectralDucker::new(&config, KOKORO_SAMPLE_RATE as f32).unwrap();

    let v0_orig_rms = rms(&v0, 0);
    let v2_orig_rms = rms(&v2, 0);

    let mut voices = vec![v0, lead, v2];
    ducker.process(&mut voices, &config).unwrap();

    let skip = (KOKORO_SAMPLE_RATE as f32 * 0.1) as usize;
    let v0_rms = rms(&voices[0], skip);
    let v2_rms = rms(&voices[2], skip);

    // Both non-lead voices should be reduced.
    assert!(
        v0_rms / v0_orig_rms < 0.85,
        "voice 0 should be ducked, ratio = {:.4}",
        v0_rms / v0_orig_rms,
    );
    assert!(
        v2_rms / v2_orig_rms < 0.85,
        "voice 2 should be ducked, ratio = {:.4}",
        v2_rms / v2_orig_rms,
    );
}
