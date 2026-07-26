// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for adaptive dynamics processor with psychoacoustic masking.

use super::*;

#[test]
fn test_default_config_validates() {
    AdaptiveDynamicsConfig::default().validate().unwrap();
}

#[test]
fn test_preset_configs_validate() {
    AdaptiveDynamicsConfig::transparent().validate().unwrap();
    AdaptiveDynamicsConfig::musical().validate().unwrap();
    AdaptiveDynamicsConfig::broadcast().validate().unwrap();
    AdaptiveDynamicsConfig::aggressive().validate().unwrap();
}

#[test]
fn test_invalid_threshold_rejected() {
    let cfg = AdaptiveDynamicsConfig::new().with_threshold_db(1.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_invalid_ratio_rejected() {
    let cfg = AdaptiveDynamicsConfig::new().with_ratio(0.5);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_builder_chain() {
    let cfg = AdaptiveDynamicsConfig::new()
        .with_threshold_db(-18.0)
        .with_ratio(4.0)
        .with_attack_ms(3.0)
        .with_release_ms(40.0)
        .with_knee_db(4.0)
        .with_masking_model(MaskingModel::BarkScale)
        .with_lookahead_ms(3.0)
        .with_makeup_gain_db(2.0);
    cfg.validate().unwrap();
    assert_eq!(cfg.ratio, 4.0);
    assert_eq!(cfg.masking_model, MaskingModel::BarkScale);
}

#[test]
fn test_processor_creation() {
    let proc = AdaptiveDynamicsProcessor::with_defaults();
    assert!(proc.is_ok());
}

#[test]
fn test_silence_passthrough() {
    let mut proc = AdaptiveDynamicsProcessor::with_defaults().unwrap();
    let mut audio = vec![0.0f32; 1000];
    proc.process(&mut audio);
    assert!(audio.iter().all(|&s| s.abs() < 1e-6));
}

#[test]
fn test_empty_buffer_no_crash() {
    let mut proc = AdaptiveDynamicsProcessor::with_defaults().unwrap();
    let mut audio = Vec::new();
    proc.process(&mut audio);
    assert!(audio.is_empty());
}

#[test]
fn test_nan_sanitization() {
    let mut proc = AdaptiveDynamicsProcessor::with_defaults().unwrap();
    let mut audio = vec![0.5, f32::NAN, 0.3, f32::INFINITY, -0.2];
    proc.process(&mut audio);
    assert!(audio.iter().all(|s| s.is_finite()));
}

#[test]
fn test_gain_reduction_reports_nonzero_for_loud_signal() {
    // threshold_db close to 0 means compress when band power exceeds
    // masking threshold by just 1 dB. A loud broadband signal will
    // have bands well above the masking threshold from other bands.
    let cfg = AdaptiveDynamicsConfig::new()
        .with_threshold_db(-1.0) // compress 1 dB above masking
        .with_ratio(8.0)
        .with_attack_ms(0.5)
        .with_release_ms(10.0)
        .with_knee_db(0.0)
        .with_lookahead_ms(0.0);
    let mut proc = AdaptiveDynamicsProcessor::new(&cfg).unwrap();
    // Loud broadband signal: multiple harmonics at high amplitude
    let mut audio: Vec<f32> = (0..4800)
        .map(|i| {
            let t = i as f32 / 24000.0;
            let pi2 = 2.0 * std::f32::consts::PI;
            0.3 * (pi2 * 200.0 * t).sin()
                + 0.3 * (pi2 * 1000.0 * t).sin()
                + 0.3 * (pi2 * 4000.0 * t).sin()
        })
        .collect();
    proc.process(&mut audio);
    assert!(
        proc.get_gain_reduction_db() > 0.0,
        "Expected gain reduction > 0, got {}",
        proc.get_gain_reduction_db(),
    );
}

#[test]
fn test_reset_clears_state() {
    let mut proc = AdaptiveDynamicsProcessor::with_defaults().unwrap();
    let mut audio: Vec<f32> = (0..2400)
        .map(|i| {
            0.8 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 24000.0).sin()
        })
        .collect();
    proc.process(&mut audio);
    proc.reset();
    assert!(proc.get_gain_reduction_db().abs() < 1e-6);
}

#[test]
fn test_masking_threshold_simple() {
    // Band 3 at 60 dB. Upward masking to band 4: spread = 60 - 10 = 50,
    // masked = 50 - 10 = 40 >> absolute threshold of -80 dBFS for band 4.
    let power =
        [-120.0, -120.0, -120.0, 60.0, -120.0, -120.0, -120.0, -120.0];
    let thresh = compute_masking_thresholds(&power, MaskingModel::SimpleMasking);
    // Band 4: upward masking from band 3 (j=3 < i=4)
    assert!(
        thresh[4] > ABSOLUTE_THRESHOLD_DB[4],
        "Band 4 threshold {} should exceed absolute {}",
        thresh[4], ABSOLUTE_THRESHOLD_DB[4],
    );
    // Band 2: downward masking from band 3 (j=3 > i=2), steeper slope
    // spread = 60 - 27 = 33, masked = 33 - 10 = 23 > absolute -80
    assert!(
        thresh[2] > ABSOLUTE_THRESHOLD_DB[2],
        "Band 2 threshold {} should exceed absolute {}",
        thresh[2], ABSOLUTE_THRESHOLD_DB[2],
    );
}

#[test]
fn test_masking_threshold_bark() {
    // Same scenario with BarkScale model (power-summation).
    let power =
        [-120.0, -120.0, -120.0, 60.0, -120.0, -120.0, -120.0, -120.0];
    let thresh = compute_masking_thresholds(&power, MaskingModel::BarkScale);
    assert!(
        thresh[4] > ABSOLUTE_THRESHOLD_DB[4],
        "Band 4 threshold {} should exceed absolute {}",
        thresh[4], ABSOLUTE_THRESHOLD_DB[4],
    );
}

#[test]
fn test_all_presets_produce_valid_processors() {
    for cfg in [
        AdaptiveDynamicsConfig::transparent(),
        AdaptiveDynamicsConfig::musical(),
        AdaptiveDynamicsConfig::broadcast(),
        AdaptiveDynamicsConfig::aggressive(),
    ] {
        let proc = AdaptiveDynamicsProcessor::new(&cfg);
        assert!(proc.is_ok(), "Failed for preset: {:?}", cfg);
    }
}
