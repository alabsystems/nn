// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for breathing patterns and micro-timing humanization.

use super::*;

const SR: u32 = 24000;

// ---------------------------------------------------------------------------
// PRNG determinism
// ---------------------------------------------------------------------------

#[test]
fn test_lcg_deterministic_same_seed() {
    let mut a = Lcg::new(0, 42);
    let mut b = Lcg::new(0, 42);
    for _ in 0..100 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

#[test]
fn test_lcg_different_seeds_differ() {
    let mut a = Lcg::new(0, 42);
    let mut b = Lcg::new(1, 42);
    // At least one of the first 10 values should differ.
    let differs = (0..10).any(|_| a.next_u64() != b.next_u64());
    assert!(
        differs,
        "different seeds should produce different sequences"
    );
}

#[test]
fn test_lcg_f32_range() {
    let mut rng = Lcg::new(7, 99);
    for _ in 0..1000 {
        let v = rng.next_f32();
        assert!((0.0..1.0).contains(&v), "next_f32 out of range: {v}");
    }
}

#[test]
fn test_lcg_f32_range_custom() {
    let mut rng = Lcg::new(3, 77);
    for _ in 0..1000 {
        let v = rng.next_f32_range(2.0, 6.0);
        assert!(
            (2.0..6.0).contains(&v),
            "next_f32_range out of [2.0, 6.0): {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// BreathPattern validation
// ---------------------------------------------------------------------------

#[test]
fn test_breath_pattern_default_valid() {
    BreathPattern::default()
        .validate()
        .expect("default should be valid");
}

#[test]
fn test_breath_pattern_min_gt_max_rejected() {
    let bp = BreathPattern {
        min_interval_sec: 5.0,
        max_interval_sec: 3.0,
        ..BreathPattern::default()
    };
    assert!(bp.validate().is_err());
}

#[test]
fn test_breath_pattern_nan_rejected() {
    let bp = BreathPattern {
        breath_depth: f32::NAN,
        ..BreathPattern::default()
    };
    assert!(bp.validate().is_err());
}

#[test]
fn test_breath_pattern_out_of_range_rejected() {
    let bp = BreathPattern {
        breath_duration_sec: 0.01, // below 0.05 minimum
        ..BreathPattern::default()
    };
    assert!(bp.validate().is_err());
}

// ---------------------------------------------------------------------------
// MicroTiming validation
// ---------------------------------------------------------------------------

#[test]
fn test_micro_timing_default_valid() {
    MicroTiming::default()
        .validate()
        .expect("default should be valid");
}

#[test]
fn test_micro_timing_jitter_too_large() {
    let mt = MicroTiming {
        onset_jitter_sec: 0.05, // above 0.030 max
        ..MicroTiming::default()
    };
    assert!(mt.validate().is_err());
}

#[test]
fn test_micro_timing_drift_rate_too_low() {
    let mt = MicroTiming {
        drift_rate_hz: 0.01, // below 0.05 min
        ..MicroTiming::default()
    };
    assert!(mt.validate().is_err());
}

// ---------------------------------------------------------------------------
// AmplitudeEnvelope validation
// ---------------------------------------------------------------------------

#[test]
fn test_envelope_default_valid() {
    AmplitudeEnvelope::default()
        .validate()
        .expect("default should be valid");
}

#[test]
fn test_envelope_attack_too_small() {
    let env = AmplitudeEnvelope {
        attack_sec: 0.0001, // below 0.001 minimum
        ..AmplitudeEnvelope::default()
    };
    assert!(env.validate().is_err());
}

#[test]
fn test_envelope_release_too_large() {
    let env = AmplitudeEnvelope {
        release_sec: 1.0, // above 0.500 max
        ..AmplitudeEnvelope::default()
    };
    assert!(env.validate().is_err());
}

// ---------------------------------------------------------------------------
// HumanizeConfig validation
// ---------------------------------------------------------------------------

#[test]
fn test_humanize_config_default_valid() {
    HumanizeConfig::default()
        .validate()
        .expect("default should be valid");
}

#[test]
fn test_humanize_config_disabled_valid() {
    // Disabled config skips validation of sub-configs.
    HumanizeConfig::disabled()
        .validate()
        .expect("disabled should be valid");
}

// ---------------------------------------------------------------------------
// Deterministic output (same seed = same output)
// ---------------------------------------------------------------------------

#[test]
fn test_apply_humanize_deterministic() {
    let config = HumanizeConfig::default();
    let original = vec![0.5f32; SR as usize * 3]; // 3 seconds

    let mut a = original.clone();
    apply_humanize(&mut a, &config, 0, SR).expect("should succeed");

    let mut b = original;
    apply_humanize(&mut b, &config, 0, SR).expect("should succeed");

    assert_eq!(a, b, "same seed must produce identical output");
}

// ---------------------------------------------------------------------------
// Different voice indices produce different output
// ---------------------------------------------------------------------------

#[test]
fn test_different_voices_differ() {
    let config = HumanizeConfig::default();
    let original = vec![0.5f32; SR as usize * 3];

    let mut v0 = original.clone();
    apply_humanize(&mut v0, &config, 0, SR).expect("should succeed");

    let mut v1 = original;
    apply_humanize(&mut v1, &config, 1, SR).expect("should succeed");

    assert_ne!(
        v0, v1,
        "different voice indices should produce different output"
    );
}

// ---------------------------------------------------------------------------
// Amplitude envelope shape verification
// ---------------------------------------------------------------------------

#[test]
fn test_envelope_attack_ramps_up() {
    let env = AmplitudeEnvelope {
        attack_sec: 0.050,
        hold_sec: 0.0,
        decay_sec: 0.0,
        sustain_level: 1.0,
        release_sec: 0.005,
    };
    let config = HumanizeConfig {
        envelope: env,
        enable_breath: false,
        enable_timing: false,
        enable_envelope: true,
        ..HumanizeConfig::default()
    };

    // 200ms of constant 1.0 audio.
    let mut audio = vec![1.0f32; (SR as f32 * 0.2) as usize];
    apply_humanize(&mut audio, &config, 0, SR).expect("should succeed");

    // First sample should be near zero (start of attack).
    assert!(
        audio[0].abs() < 0.01,
        "first sample should be near zero during attack, got {}",
        audio[0],
    );

    // At 50ms (attack end), should be near 1.0.
    let attack_end = (0.050 * SR as f32).round() as usize;
    if attack_end < audio.len() {
        assert!(
            audio[attack_end] > 0.95,
            "sample at attack end should be near 1.0, got {}",
            audio[attack_end],
        );
    }
}

#[test]
fn test_envelope_release_ramps_down() {
    let env = AmplitudeEnvelope {
        attack_sec: 0.001,
        hold_sec: 0.0,
        decay_sec: 0.0,
        sustain_level: 1.0,
        release_sec: 0.100,
    };
    let config = HumanizeConfig {
        envelope: env,
        enable_breath: false,
        enable_timing: false,
        enable_envelope: true,
        ..HumanizeConfig::default()
    };

    let mut audio = vec![1.0f32; (SR as f32 * 0.5) as usize];
    apply_humanize(&mut audio, &config, 0, SR).expect("should succeed");

    // Last sample should be near zero (end of release).
    let last = *audio.last().unwrap();
    assert!(
        last.abs() < 0.05,
        "last sample should be near zero during release, got {last}",
    );

    // 50ms before end should still have significant amplitude.
    let mid_release = audio.len() - (0.050 * SR as f32).round() as usize;
    assert!(
        audio[mid_release] > 0.3,
        "mid-release should have significant amplitude, got {}",
        audio[mid_release],
    );
}

// ---------------------------------------------------------------------------
// Breath pattern regularity
// ---------------------------------------------------------------------------

#[test]
fn test_breath_dips_within_range() {
    let breath = BreathPattern {
        min_interval_sec: 2.0,
        max_interval_sec: 4.0,
        breath_duration_sec: 0.15,
        breath_depth: 0.5,
    };
    let config = HumanizeConfig {
        breath,
        enable_breath: true,
        enable_timing: false,
        enable_envelope: false,
        ..HumanizeConfig::default()
    };

    // 10 seconds of constant 1.0 audio.
    let len = SR as usize * 10;
    let mut audio = vec![1.0f32; len];
    apply_humanize(&mut audio, &config, 0, SR).expect("should succeed");

    // Count regions where amplitude dipped below 0.9 (breath dips).
    // We expect 2-5 breaths in 10 seconds with 2-4s intervals.
    let mut dip_count = 0;
    let mut in_dip = false;
    for &s in &audio {
        if s < 0.9 && !in_dip {
            dip_count += 1;
            in_dip = true;
        } else if s >= 0.95 {
            in_dip = false;
        }
    }

    assert!(
        (1..=6).contains(&dip_count),
        "expected 1-6 breath dips in 10s, got {dip_count}",
    );
}

// ---------------------------------------------------------------------------
// Micro-timing does not change audio energy drastically
// ---------------------------------------------------------------------------

#[test]
fn test_micro_timing_preserves_energy() {
    let config = HumanizeConfig {
        enable_breath: false,
        enable_timing: true,
        enable_envelope: false,
        ..HumanizeConfig::default()
    };

    // Sine wave at 440Hz, 1 second.
    let len = SR as usize;
    let mut audio: Vec<f32> = (0..len)
        .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / SR as f32).sin())
        .collect();

    let energy_before: f32 = audio.iter().map(|s| s * s).sum();
    apply_humanize(&mut audio, &config, 0, SR).expect("should succeed");
    let energy_after: f32 = audio.iter().map(|s| s * s).sum();

    // Energy should be within 5% -- timing drift shifts samples but
    // should not create or destroy energy significantly.
    let ratio = energy_after / energy_before;
    assert!(
        (0.90..=1.10).contains(&ratio),
        "energy ratio {ratio} outside [0.90, 1.10]: timing drift changed energy too much",
    );
}

// ---------------------------------------------------------------------------
// Empty audio is a no-op
// ---------------------------------------------------------------------------

#[test]
fn test_apply_humanize_empty_audio() {
    let config = HumanizeConfig::default();
    let mut audio: Vec<f32> = vec![];
    apply_humanize(&mut audio, &config, 0, SR).expect("empty audio should succeed");
    assert!(audio.is_empty());
}

// ---------------------------------------------------------------------------
// Disabled config is pass-through
// ---------------------------------------------------------------------------

#[test]
fn test_disabled_config_passthrough() {
    let config = HumanizeConfig::disabled();
    let original = vec![0.7f32; SR as usize];
    let mut audio = original.clone();
    apply_humanize(&mut audio, &config, 0, SR).expect("should succeed");
    assert_eq!(audio, original, "disabled config should not modify audio");
}

// ---------------------------------------------------------------------------
// Builder methods
// ---------------------------------------------------------------------------

#[test]
fn test_breath_only_config() {
    let config = HumanizeConfig::breath_only();
    assert!(config.enable_breath);
    assert!(!config.enable_timing);
    assert!(!config.enable_envelope);
    config.validate().expect("breath_only should be valid");
}

#[test]
fn test_with_custom_breath() {
    let bp = BreathPattern {
        min_interval_sec: 3.0,
        max_interval_sec: 5.0,
        breath_duration_sec: 0.2,
        breath_depth: 0.3,
    };
    let config = HumanizeConfig::default().with_breath(bp);
    assert!((config.breath.min_interval_sec - 3.0).abs() < 1e-6);
    config.validate().expect("custom breath should be valid");
}

#[test]
fn test_with_custom_timing() {
    let mt = MicroTiming {
        onset_jitter_sec: 0.010,
        tempo_drift_max: 0.01,
        drift_rate_hz: 0.5,
    };
    let config = HumanizeConfig::default().with_timing(mt);
    assert!((config.timing.onset_jitter_sec - 0.010).abs() < 1e-6);
    config.validate().expect("custom timing should be valid");
}

#[test]
fn test_with_custom_envelope() {
    let env = AmplitudeEnvelope {
        attack_sec: 0.050,
        hold_sec: 0.1,
        decay_sec: 0.1,
        sustain_level: 0.8,
        release_sec: 0.100,
    };
    let config = HumanizeConfig::default().with_envelope(env);
    assert!((config.envelope.sustain_level - 0.8).abs() < 1e-6);
    config.validate().expect("custom envelope should be valid");
}

// ---------------------------------------------------------------------------
// apply_jitter (public circular-shift API)
// ---------------------------------------------------------------------------

#[test]
fn test_apply_jitter_zero_shift_is_identity() {
    let audio: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
    let result = apply_jitter(&audio, 0, 0);
    assert_eq!(result, audio);
}

#[test]
fn test_apply_jitter_circular_shift() {
    let audio = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = apply_jitter(&audio, 2, 0);
    // Shift right by 2: last 2 elements move to front.
    assert_eq!(result, vec![4.0, 5.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_apply_jitter_full_rotation_is_identity() {
    let audio = vec![1.0, 2.0, 3.0, 4.0];
    let result = apply_jitter(&audio, 4, 0);
    // Shift by length = identity (modulo wraps to 0).
    assert_eq!(result, audio);
}

#[test]
fn test_apply_jitter_preserves_length() {
    let audio = vec![0.5f32; 1000];
    let result = apply_jitter(&audio, 37, 3);
    assert_eq!(result.len(), audio.len());
}

#[test]
fn test_apply_jitter_different_offsets_differ() {
    let audio: Vec<f32> = (0..200).map(|i| (i as f32 * 0.1).sin()).collect();
    let a = apply_jitter(&audio, 5, 0);
    let b = apply_jitter(&audio, 10, 1);
    assert_ne!(
        a, b,
        "different jitter offsets should produce different output"
    );
}

#[test]
fn test_apply_jitter_empty_audio() {
    let audio: Vec<f32> = vec![];
    let result = apply_jitter(&audio, 5, 0);
    assert!(result.is_empty());
}

// ---------------------------------------------------------------------------
// Onset jitter via apply_humanize (integrated path)
// ---------------------------------------------------------------------------

#[test]
fn test_onset_jitter_changes_audio() {
    let timing_with_jitter = MicroTiming {
        onset_jitter_sec: 0.020, // 20ms -- large enough to see effect
        tempo_drift_max: 0.0,    // disable drift to isolate jitter
        drift_rate_hz: 0.3,
    };
    let config = HumanizeConfig {
        timing: timing_with_jitter,
        enable_breath: false,
        enable_timing: true,
        enable_envelope: false,
        ..HumanizeConfig::default()
    };

    let original: Vec<f32> = (0..SR as usize * 2)
        .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / SR as f32).sin())
        .collect();

    let mut voice0 = original.clone();
    apply_humanize(&mut voice0, &config, 0, SR).expect("should succeed");

    let mut voice1 = original;
    apply_humanize(&mut voice1, &config, 1, SR).expect("should succeed");

    // Different voices should get different jitter offsets.
    assert_ne!(
        voice0, voice1,
        "different voice indices should produce different jitter shifts"
    );
}

#[test]
fn test_onset_jitter_deterministic() {
    let timing = MicroTiming {
        onset_jitter_sec: 0.015,
        tempo_drift_max: 0.0,
        drift_rate_hz: 0.3,
    };
    let config = HumanizeConfig {
        timing,
        enable_breath: false,
        enable_timing: true,
        enable_envelope: false,
        ..HumanizeConfig::default()
    };

    let original = vec![0.7f32; SR as usize];

    let mut a = original.clone();
    apply_humanize(&mut a, &config, 5, SR).expect("should succeed");

    let mut b = original;
    apply_humanize(&mut b, &config, 5, SR).expect("should succeed");

    assert_eq!(
        a, b,
        "same voice index must produce identical jitter output"
    );
}

#[test]
fn test_onset_jitter_preserves_energy() {
    let timing = MicroTiming {
        onset_jitter_sec: 0.010,
        tempo_drift_max: 0.0,
        drift_rate_hz: 0.3,
    };
    let config = HumanizeConfig {
        timing,
        enable_breath: false,
        enable_timing: true,
        enable_envelope: false,
        ..HumanizeConfig::default()
    };

    let original: Vec<f32> = (0..SR as usize)
        .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / SR as f32).sin())
        .collect();
    let energy_before: f32 = original.iter().map(|s| s * s).sum();

    let mut audio = original;
    apply_humanize(&mut audio, &config, 2, SR).expect("should succeed");
    let energy_after: f32 = audio.iter().map(|s| s * s).sum();

    // Circular shift preserves energy exactly.
    let ratio = energy_after / energy_before;
    assert!(
        (0.99..=1.01).contains(&ratio),
        "circular jitter should preserve energy, ratio = {ratio}",
    );
}
