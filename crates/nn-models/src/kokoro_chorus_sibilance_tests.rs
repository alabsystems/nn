// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `kokoro_chorus_sibilance` — de-essing, enhancement, alignment.

use super::*;

#[test]
fn test_config_default_valid() {
    SibilanceConfig::new()
        .validate()
        .expect("default config should be valid");
}

#[test]
fn test_config_builder_roundtrip() {
    let cfg = SibilanceConfig::new()
        .with_mode(SibilanceMode::DeEss)
        .with_detection_freq_hz(7000.0)
        .with_threshold_db(-25.0)
        .with_reduction_db(8.0)
        .with_attack_ms(0.3)
        .with_release_ms(25.0)
        .with_enhancement_db(4.0)
        .with_air_freq_hz(14000.0)
        .with_air_boost_db(3.0)
        .with_stagger_ms(2.0)
        .with_align_sibilants(false);
    cfg.validate().expect("builder config should be valid");
    assert_eq!(cfg.mode, SibilanceMode::DeEss);
    assert_eq!(cfg.detection_freq_hz, 7000.0);
    assert!(!cfg.align_sibilants);
}

#[test]
fn test_config_invalid_freq() {
    let r = SibilanceConfig::new()
        .with_detection_freq_hz(500.0)
        .validate();
    assert!(r.is_err(), "freq below 1000 should be invalid");
    let r = SibilanceConfig::new()
        .with_detection_freq_hz(f32::NAN)
        .validate();
    assert!(r.is_err(), "NaN freq should be invalid");
}

#[test]
fn test_config_invalid_reduction() {
    let r = SibilanceConfig::new().with_reduction_db(30.0).validate();
    assert!(r.is_err(), "reduction > 24 should be invalid");
}

#[test]
fn test_presets_validate() {
    gentle_deess()
        .validate()
        .expect("gentle_deess should be valid");
    aggressive_deess()
        .validate()
        .expect("aggressive_deess should be valid");
    presence_enhance()
        .validate()
        .expect("presence_enhance should be valid");
    broadcast_balanced()
        .validate()
        .expect("broadcast_balanced should be valid");
}

#[test]
fn test_processor_deess_reduces_sibilance() {
    let sr = 24000.0;
    let n = 2048;
    // Generate a 7 kHz sibilant-like tone.
    let mut audio: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 7000.0 * i as f32 / sr).sin() * 0.8)
        .collect();
    let energy_before: f32 = audio.iter().map(|x| x * x).sum();

    let cfg = aggressive_deess();
    let mut proc = SibilanceProcessor::new(cfg, sr).expect("valid config");
    proc.process_voice(&mut audio);

    let energy_after: f32 = audio.iter().map(|x| x * x).sum();
    assert!(
        energy_after < energy_before,
        "de-essing should reduce sibilance energy: before={energy_before}, after={energy_after}",
    );
}

#[test]
fn test_processor_enhance_boosts() {
    let sr = 24000.0;
    let n = 2048;
    // Low-energy sibilant tone (below threshold).
    let mut audio: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 6500.0 * i as f32 / sr).sin() * 0.01)
        .collect();
    let energy_before: f32 = audio.iter().map(|x| x * x).sum();

    let cfg = presence_enhance();
    let mut proc = SibilanceProcessor::new(cfg, sr).expect("valid config");
    proc.process_voice(&mut audio);

    let energy_after: f32 = audio.iter().map(|x| x * x).sum();
    assert!(
        energy_after > energy_before * 0.9,
        "enhancement should not reduce low-level signal: before={energy_before}, after={energy_after}",
    );
}

#[test]
fn test_processor_balanced_finite_output() {
    let sr = 24000.0;
    let n = 1024;
    let mut audio: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            (2.0 * std::f32::consts::PI * 6500.0 * t).sin() * 0.5
                + (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3
        })
        .collect();

    let cfg = broadcast_balanced();
    let mut proc = SibilanceProcessor::new(cfg, sr).expect("valid config");
    proc.process_voice(&mut audio);

    for (i, &v) in audio.iter().enumerate() {
        assert!(v.is_finite(), "sample {i} is non-finite: {v}");
    }
}

#[test]
fn test_nan_input_clamped() {
    let mut audio = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5, -0.3];
    let cfg = SibilanceConfig::new();
    let mut proc = SibilanceProcessor::new(cfg, 24000.0).expect("valid");
    proc.process_voice(&mut audio);
    for (i, &v) in audio.iter().enumerate() {
        assert!(v.is_finite(), "sample {i} should be finite, got {v}");
    }
}

#[test]
fn test_reset_clears_state() {
    let cfg = SibilanceConfig::new();
    let mut proc = SibilanceProcessor::new(cfg, 24000.0).expect("valid");
    let mut buf = vec![0.5; 100];
    proc.process_voice(&mut buf);
    proc.reset();
    assert_eq!(proc.envelope, 0.0);
}

#[test]
fn test_align_sibilants_single_voice_noop() {
    let cfg = SibilanceConfig::new();
    let mut voices = vec![vec![0.1; 100]];
    let original = voices.clone();
    align_sibilants(&mut voices, &cfg, 24000.0);
    assert_eq!(voices, original, "single voice should be unchanged");
}

#[test]
fn test_align_sibilants_disabled() {
    let cfg = SibilanceConfig::new().with_align_sibilants(false);
    let mut voices = vec![vec![0.1; 100], vec![0.2; 100]];
    let original = voices.clone();
    align_sibilants(&mut voices, &cfg, 24000.0);
    assert_eq!(
        voices, original,
        "disabled alignment should not modify voices"
    );
}

#[test]
fn test_apply_delay_basic() {
    let mut buf = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    apply_delay(&mut buf, 2);
    assert_eq!(buf, vec![0.0, 0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_apply_delay_zero() {
    let mut buf = vec![1.0, 2.0, 3.0];
    let original = buf.clone();
    apply_delay(&mut buf, 0);
    assert_eq!(buf, original);
}

#[test]
fn test_biquad_bandpass_finite() {
    let mut f = BiquadFilter::bandpass(6500.0, 1.5, 24000.0);
    for i in 0..100 {
        let x = (i as f32 * 0.1).sin();
        let y = f.process(x);
        assert!(
            y.is_finite(),
            "bandpass output should be finite at sample {i}"
        );
    }
}

#[test]
fn test_detect_sibilant_onsets_empty() {
    let cfg = SibilanceConfig::new();
    let onsets = detect_sibilant_onsets(&[], &cfg, 24000.0);
    assert!(onsets.is_empty());
}
