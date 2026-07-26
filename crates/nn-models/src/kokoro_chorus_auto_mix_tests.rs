// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for kokoro_chorus_auto_mix.

use super::*;

fn sine_voice(freq_hz: f32, amplitude: f32, n_samples: usize, sr: f32) -> Vec<f32> {
    (0..n_samples)
        .map(|i| amplitude * (std::f32::consts::TAU * freq_hz * i as f32 / sr).sin())
        .collect()
}

#[test]
fn test_config_default_validates() {
    AutoMixConfig::default()
        .validate()
        .expect("default config should validate");
}

#[test]
fn test_config_presets_validate() {
    AutoMixConfig::balanced()
        .validate()
        .expect("balanced preset should validate");
    AutoMixConfig::lead_focused()
        .validate()
        .expect("lead_focused preset should validate");
    AutoMixConfig::blended()
        .validate()
        .expect("blended preset should validate");
    AutoMixConfig::broadcast()
        .validate()
        .expect("broadcast preset should validate");
}

#[test]
fn test_config_invalid_correction_speed() {
    let config = AutoMixConfig::new().with_correction_speed(1.5);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_invalid_analysis_window() {
    let config = AutoMixConfig::new().with_analysis_window_ms(5.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_invalid_max_gain() {
    let config = AutoMixConfig::new().with_max_gain_change_db(-1.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_invalid_sample_rate() {
    let config = AutoMixConfig::new().with_sample_rate(0.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_mixer_creation() {
    let config = AutoMixConfig::default();
    let mixer = AutoMixer::new(&config);
    assert!(mixer.is_ok());
}

#[test]
fn test_analyze_empty_voices() {
    let config = AutoMixConfig::default();
    let mut mixer = AutoMixer::new(&config).unwrap();
    let mut voices: Vec<Vec<f32>> = vec![];
    let analysis = mixer.analyze_and_adjust(&mut voices);
    assert!(analysis.applied_gains.is_empty());
}

#[test]
fn test_analyze_identical_voices_small_correction() {
    let config = AutoMixConfig::new()
        .with_target_balance(TargetBalance::Flat)
        .with_correction_speed(1.0);
    let mut mixer = AutoMixer::new(&config).unwrap();

    let n = 4096;
    let sr = 24000.0;
    let tone = sine_voice(440.0, 0.3, n, sr);
    let mut voices = vec![tone.clone(), tone];

    let analysis = mixer.analyze_and_adjust(&mut voices);
    assert_eq!(analysis.applied_gains.len(), 2);

    // With identical voices and flat target, corrections stay within max_gain_change_db.
    for &g in &analysis.applied_gains {
        assert!(
            g.abs() <= 6.0 + 0.01,
            "gain correction should be bounded by max_gain_change_db, got {g}"
        );
    }
}

#[test]
fn test_lead_voice_never_attenuated() {
    let config = AutoMixConfig::new()
        .with_lead_voice(Some(0))
        .with_correction_speed(1.0)
        .with_target_balance(TargetBalance::Flat);
    let mut mixer = AutoMixer::new(&config).unwrap();

    let n = 4096;
    let sr = 24000.0;
    // Lead voice is loud; backing voice is quiet.
    let lead = sine_voice(440.0, 0.8, n, sr);
    let backing = sine_voice(440.0, 0.1, n, sr);
    let mut voices = vec![lead, backing];

    let analysis = mixer.analyze_and_adjust(&mut voices);
    // Lead voice gain should be >= 0 (never cut).
    assert!(
        analysis.applied_gains[0] >= -0.01,
        "lead voice should not be attenuated, got {}",
        analysis.applied_gains[0],
    );
}

#[test]
fn test_max_gain_change_respected() {
    let max_db = 3.0;
    let config = AutoMixConfig::new()
        .with_max_gain_change_db(max_db)
        .with_correction_speed(1.0);
    let mut mixer = AutoMixer::new(&config).unwrap();

    let n = 4096;
    let sr = 24000.0;
    let loud = sine_voice(440.0, 0.8, n, sr);
    let quiet = sine_voice(4000.0, 0.01, n, sr);
    let mut voices = vec![loud, quiet];

    let analysis = mixer.analyze_and_adjust(&mut voices);
    for &g in &analysis.applied_gains {
        assert!(
            g.abs() <= max_db + 0.01,
            "gain should not exceed max_gain_change_db={max_db}, got {g}",
        );
    }
}

#[test]
fn test_target_balance_profiles() {
    let flat = TargetBalance::Flat.evaluate();
    assert!(flat.iter().all(|&v| v == 0.0));

    let speech = TargetBalance::Speech.evaluate();
    // Speech should have a mid-presence boost (band 3 or 4 > 0).
    assert!(speech[3] > 0.0 || speech[4] > 0.0);

    let singing = TargetBalance::Singing.evaluate();
    // Singing should have a mid scoop (band 3 < 0).
    assert!(singing[3] < 0.0);

    let broadcast = TargetBalance::Broadcast.evaluate();
    // Broadcast should have HF rolloff (band 7 < 0).
    assert!(broadcast[7] < 0.0);
}

#[test]
fn test_custom_target_balance() {
    let custom = TargetBalance::Custom(vec![(0.0, -3.0), (3.0, 2.0), (7.0, -5.0)]);
    let levels = custom.evaluate();
    assert!((levels[0] - (-3.0)).abs() < f32::EPSILON);
    assert!((levels[3] - 2.0).abs() < f32::EPSILON);
    assert!((levels[7] - (-5.0)).abs() < f32::EPSILON);
    // Unspecified bands should be 0.
    assert!((levels[1]).abs() < f32::EPSILON);
}

#[test]
fn test_reset_clears_gains() {
    let config = AutoMixConfig::new().with_correction_speed(1.0);
    let mut mixer = AutoMixer::new(&config).unwrap();

    let n = 4096;
    let sr = 24000.0;
    let mut voices = vec![
        sine_voice(440.0, 0.5, n, sr),
        sine_voice(2000.0, 0.5, n, sr),
    ];

    mixer.analyze_and_adjust(&mut voices);
    mixer.reset();

    assert!(
        mixer
            .current_gains()
            .iter()
            .all(|&g| g.abs() < f32::EPSILON),
        "reset should zero all gains"
    );
}

#[test]
fn test_nan_in_voice_handled() {
    let config = AutoMixConfig::default();
    let mut mixer = AutoMixer::new(&config).unwrap();
    let mut voices = vec![vec![0.1; 2048], vec![f32::NAN; 2048]];
    // Should not panic.
    let analysis = mixer.analyze_and_adjust(&mut voices);
    assert_eq!(analysis.applied_gains.len(), 2);
    for &g in &analysis.applied_gains {
        assert!(g.is_finite(), "gains must be finite, got {g}");
    }
}

#[test]
fn test_lead_clarity_metric() {
    let config = AutoMixConfig::new()
        .with_lead_voice(Some(0))
        .with_correction_speed(0.0); // No correction, just measure.
    let mut mixer = AutoMixer::new(&config).unwrap();

    let n = 4096;
    let sr = 24000.0;
    let lead = sine_voice(440.0, 0.8, n, sr);
    let backing = sine_voice(440.0, 0.05, n, sr);
    let mut voices = vec![lead, backing];

    let analysis = mixer.analyze_and_adjust(&mut voices);
    // Lead is much louder, so clarity should be high.
    assert!(
        analysis.lead_clarity > 0.5,
        "lead clarity should be high when lead is dominant, got {}",
        analysis.lead_clarity,
    );
}

#[test]
fn test_correction_speed_zero_no_adjustment() {
    let config = AutoMixConfig::new().with_correction_speed(0.0);
    let mut mixer = AutoMixer::new(&config).unwrap();

    let n = 4096;
    let sr = 24000.0;
    let mut voices = vec![
        sine_voice(440.0, 0.5, n, sr),
        sine_voice(2000.0, 0.5, n, sr),
    ];

    let analysis = mixer.analyze_and_adjust(&mut voices);
    // With speed=0, gains should remain at 0.
    for &g in &analysis.applied_gains {
        assert!(
            g.abs() < f32::EPSILON,
            "correction_speed=0 should produce no gain change, got {g}",
        );
    }
}

#[test]
fn test_mix_analysis_deviation_reported() {
    let config = AutoMixConfig::new()
        .with_target_balance(TargetBalance::Speech)
        .with_correction_speed(0.0); // Measure only.
    let mut mixer = AutoMixer::new(&config).unwrap();

    let n = 4096;
    let sr = 24000.0;
    let mut voices = vec![sine_voice(440.0, 0.5, n, sr)];

    let analysis = mixer.analyze_and_adjust(&mut voices);
    // At least some bands should have non-zero deviation from Speech target.
    let any_deviation = analysis.target_deviation.iter().any(|&d| d.abs() > 0.1);
    assert!(
        any_deviation,
        "single-frequency voice should deviate from speech target"
    );
}
