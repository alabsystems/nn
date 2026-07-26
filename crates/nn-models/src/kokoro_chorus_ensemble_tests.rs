// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for kokoro_chorus_ensemble (classic modulated delay chorus effect).

use super::*;

fn sine_tone(freq_hz: f32, sr: f32, n_samples: usize) -> Vec<f32> {
    (0..n_samples)
        .map(|i| (std::f32::consts::TAU * freq_hz * i as f32 / sr).sin())
        .collect()
}

#[test]
fn test_default_config_validates() {
    EnsembleConfig::default().validate().unwrap();
}

#[test]
fn test_all_presets_validate() {
    EnsembleConfig::subtle_chorus().validate().unwrap();
    EnsembleConfig::rich_ensemble().validate().unwrap();
    EnsembleConfig::thick_flange().validate().unwrap();
    EnsembleConfig::string_machine().validate().unwrap();
    EnsembleConfig::juno_chorus_i().validate().unwrap();
    EnsembleConfig::juno_chorus_ii().validate().unwrap();
}

#[test]
fn test_validation_rejects_invalid() {
    assert!(EnsembleConfig::default()
        .with_n_voices(0)
        .validate()
        .is_err());
    assert!(EnsembleConfig::default()
        .with_n_voices(9)
        .validate()
        .is_err());
    assert!(EnsembleConfig::default()
        .with_rate_hz(0.0)
        .validate()
        .is_err());
    assert!(EnsembleConfig::default()
        .with_depth_ms(0.0)
        .validate()
        .is_err());
    assert!(EnsembleConfig::default()
        .with_feedback(0.8)
        .validate()
        .is_err());
    assert!(EnsembleConfig::default().with_mix(-0.1).validate().is_err());
    assert!(EnsembleConfig::default()
        .with_high_cut_hz(100.0)
        .validate()
        .is_err());
    assert!(EnsembleConfig::default()
        .with_rate_hz(f32::NAN)
        .validate()
        .is_err());
    assert!(EnsembleConfig::default()
        .with_mix(f32::INFINITY)
        .validate()
        .is_err());
}

#[test]
fn test_processor_creation() {
    let config = EnsembleConfig::default();
    assert!(EnsembleProcessor::new(&config, 24000.0).is_ok());
}

#[test]
fn test_mono_preserves_length() {
    let config = EnsembleConfig::default();
    let mut proc = EnsembleProcessor::new(&config, 24000.0).unwrap();
    let audio = sine_tone(440.0, 24000.0, 12000);
    let (l, r) = proc.process_mono(&audio);
    assert_eq!(l.len(), audio.len());
    assert_eq!(r.len(), audio.len());
}

#[test]
fn test_stereo_preserves_length() {
    let config = EnsembleConfig::default();
    let mut proc = EnsembleProcessor::new(&config, 24000.0).unwrap();
    let mut l = sine_tone(440.0, 24000.0, 6000);
    let mut r = sine_tone(440.0, 24000.0, 6000);
    proc.process_stereo(&mut l, &mut r).unwrap();
    assert_eq!(l.len(), 6000);
    assert_eq!(r.len(), 6000);
}

#[test]
fn test_stereo_length_mismatch_error() {
    let config = EnsembleConfig::default();
    let mut proc = EnsembleProcessor::new(&config, 24000.0).unwrap();
    let mut l = vec![0.0; 100];
    let mut r = vec![0.0; 200];
    assert!(proc.process_stereo(&mut l, &mut r).is_err());
}

#[test]
fn test_chorus_modifies_audio() {
    let config = EnsembleConfig::default();
    let mut proc = EnsembleProcessor::new(&config, 24000.0).unwrap();
    let audio = sine_tone(440.0, 24000.0, 12000);
    let (l, _r) = proc.process_mono(&audio);

    let diff: f32 = l
        .iter()
        .zip(audio.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / l.len() as f32;
    assert!(
        diff > 1e-4,
        "chorus should modify audio, mean diff = {diff}"
    );
}

#[test]
fn test_stereo_width() {
    let config = EnsembleConfig::default();
    let mut proc = EnsembleProcessor::new(&config, 24000.0).unwrap();
    let audio = sine_tone(440.0, 24000.0, 12000);
    let (l, r) = proc.process_mono(&audio);

    let lr_diff: f32 = l
        .iter()
        .zip(r.iter())
        .skip(2000)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / (l.len() - 2000) as f32;
    assert!(lr_diff > 1e-4, "L/R should differ, mean diff = {lr_diff}");
}

#[test]
fn test_no_nan_or_inf() {
    let config = EnsembleConfig::rich_ensemble();
    let mut proc = EnsembleProcessor::new(&config, 24000.0).unwrap();
    let audio = sine_tone(440.0, 24000.0, 24000);
    let (l, r) = proc.process_mono(&audio);
    for (i, (&sl, &sr)) in l.iter().zip(r.iter()).enumerate() {
        assert!(sl.is_finite(), "L[{i}] = {sl}");
        assert!(sr.is_finite(), "R[{i}] = {sr}");
    }
}

#[test]
fn test_reset_clears_state() {
    let config = EnsembleConfig::default();
    let mut proc = EnsembleProcessor::new(&config, 24000.0).unwrap();
    let _ = proc.process_mono(&sine_tone(440.0, 24000.0, 12000));
    proc.reset();
    for dl in &proc.delay_lines {
        let energy: f32 = dl.iter().map(|s| s * s).sum();
        assert!(energy < 1e-10, "delay line should be zeroed after reset");
    }
}

#[test]
fn test_flanger_mode() {
    let config = EnsembleConfig::thick_flange();
    let mut proc = EnsembleProcessor::new(&config, 24000.0).unwrap();
    let audio = sine_tone(440.0, 24000.0, 12000);
    let (l, r) = proc.process_mono(&audio);
    assert_eq!(l.len(), audio.len());
    assert_eq!(r.len(), audio.len());
    for &s in l.iter().chain(r.iter()) {
        assert!(s.is_finite());
    }
}

#[test]
fn test_vibrato_mode_fully_wet() {
    let config = EnsembleConfig::default().with_mode(EnsembleMode::Vibrato);
    let mut proc = EnsembleProcessor::new(&config, 24000.0).unwrap();

    // Vibrato of silence should yield silence.
    let (l, r) = proc.process_mono(&vec![0.0_f32; 6000]);
    let energy: f32 = l.iter().chain(r.iter()).map(|s| s * s).sum();
    assert!(energy < 1e-6, "vibrato of silence should be silent");

    // Vibrato of a tone should have energy.
    proc.reset();
    let (l2, _) = proc.process_mono(&sine_tone(440.0, 24000.0, 6000));
    let rms: f32 = l2.iter().map(|s| s * s).sum::<f32>() / l2.len() as f32;
    assert!(rms > 0.01, "vibrato of tone should have energy");
}

#[test]
fn test_string_ensemble_mode() {
    let config = EnsembleConfig::string_machine();
    let mut proc = EnsembleProcessor::new(&config, 24000.0).unwrap();
    let audio = sine_tone(440.0, 24000.0, 24000);
    let (l, r) = proc.process_mono(&audio);
    assert_eq!(l.len(), audio.len());
    for &s in l.iter().chain(r.iter()) {
        assert!(s.is_finite());
    }
}

#[test]
fn test_builder_chain() {
    let config = EnsembleConfig::default()
        .with_n_voices(4)
        .with_rate_hz(1.0)
        .with_depth_ms(8.0)
        .with_feedback(0.3)
        .with_mix(0.6)
        .with_stereo_spread(90.0)
        .with_mode(EnsembleMode::Flanger)
        .with_high_cut_hz(6000.0);
    config.validate().unwrap();
    assert_eq!(config.n_voices, 4);
    assert!((config.rate_hz - 1.0).abs() < 1e-6);
    assert_eq!(config.mode, EnsembleMode::Flanger);
}
