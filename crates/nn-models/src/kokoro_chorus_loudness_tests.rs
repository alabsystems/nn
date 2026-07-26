// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for kokoro_chorus_loudness (ITU-R BS.1770 LUFS, A-weighting, Bark).

use super::*;

#[test]
fn test_config_default_validates() {
    LoudnessConfig::default()
        .validate()
        .expect("default config should be valid");
}

#[test]
fn test_config_builder() {
    let cfg = LoudnessConfig::default()
        .with_target_lufs(-14.0)
        .with_true_peak_limit(-0.5)
        .with_gate_threshold(-60.0)
        .with_measurement_window_ms(200.0)
        .with_weighting(LoudnessWeighting::AWeighting);
    cfg.validate().expect("builder config should be valid");
    assert_eq!(cfg.target_lufs, -14.0);
    assert_eq!(cfg.weighting, LoudnessWeighting::AWeighting);
}

#[test]
fn test_config_invalid_target_lufs() {
    let cfg = LoudnessConfig::default().with_target_lufs(5.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_invalid_peak_limit() {
    let cfg = LoudnessConfig::default().with_true_peak_limit(-20.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_invalid_gate() {
    let cfg = LoudnessConfig::default().with_gate_threshold(0.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_invalid_window() {
    let cfg = LoudnessConfig::default().with_measurement_window_ms(10.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_silence_measurement() {
    let mut meter = LoudnessMeter::default_24k().unwrap();
    let silence = vec![0.0f32; 24000];
    let lufs = meter.measure_momentary(&silence);
    assert!(
        lufs <= SILENCE_DB + 1.0,
        "silence should measure near floor: {lufs}"
    );
}

#[test]
fn test_empty_measurement() {
    let mut meter = LoudnessMeter::default_24k().unwrap();
    assert!(meter.measure_momentary(&[]) <= SILENCE_DB + 1.0);
}

#[test]
fn test_sine_loudness_reasonable_range() {
    let mut meter = LoudnessMeter::default_24k().unwrap();
    let n = 24000;
    let sine: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 24000.0).sin() * 0.5)
        .collect();
    let lufs = meter.measure_momentary(&sine);
    assert!(lufs > -12.0, "sine LUFS too low: {lufs}");
    assert!(lufs < 0.0, "sine LUFS too high: {lufs}");
}

#[test]
fn test_integrated_loudness_gating() {
    let mut meter = LoudnessMeter::default_24k().unwrap();
    let silence = vec![0.0f32; 12000];
    let tone: Vec<f32> = (0..24000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin() * 0.3)
        .collect();
    meter.feed(&silence);
    meter.feed(&tone);
    let integrated = meter.integrated_loudness();
    assert!(
        integrated > SILENCE_DB + 10.0,
        "integrated too low: {integrated}"
    );
}

#[test]
fn test_true_peak_sine() {
    let sine: Vec<f32> = (0..4800)
        .map(|i| (2.0 * std::f32::consts::PI * 997.0 * i as f32 / 24000.0).sin() * 0.8)
        .collect();
    let tp = measure_true_peak(&sine, 24000.0);
    assert!(tp > -3.0, "true peak too low: {tp}");
    assert!(tp < 0.0, "true peak should be negative dBFS: {tp}");
}

#[test]
fn test_true_peak_empty() {
    assert!(measure_true_peak(&[], 24000.0) <= SILENCE_DB + 1.0);
}

#[test]
fn test_true_peak_dc() {
    let dc = vec![0.5f32; 100];
    let tp = measure_true_peak(&dc, 24000.0);
    let expected = 20.0 * (0.5f64).log10();
    assert!(
        (tp - expected as f32).abs() < 0.5,
        "DC true peak: {tp} vs {expected}"
    );
}

#[test]
fn test_normalize_to_target() {
    let mut meter =
        LoudnessMeter::new(&LoudnessConfig::default().with_target_lufs(-16.0), 24000.0).unwrap();
    let mut audio: Vec<f32> = (0..48000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin() * 0.1)
        .collect();
    let gain = meter.normalize_to_target(&mut audio);
    assert!(gain > 0.0, "expected positive gain, got {gain}");
}

#[test]
fn test_normalize_silence_noop() {
    let mut meter = LoudnessMeter::default_24k().unwrap();
    let mut audio = vec![0.0f32; 48000];
    let gain = meter.normalize_to_target(&mut audio);
    assert!(
        (gain).abs() < 0.01,
        "silence normalize should be noop: {gain}"
    );
}

#[test]
fn test_bark_band_loudness_basic() {
    let sine: Vec<f32> = (0..4096)
        .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 24000.0).sin())
        .collect();
    let bands = bark_band_loudness(&sine, 24000.0);
    assert_eq!(bands.len(), 24);
    assert!(
        bands[9] > bands[0],
        "1kHz band should be louder than DC band"
    );
}

#[test]
fn test_bark_band_empty() {
    let bands = bark_band_loudness(&[], 24000.0);
    assert_eq!(bands.len(), 24);
    assert!(bands.iter().all(|&b| b <= SILENCE_DB + 1.0));
}

#[test]
fn test_meter_reset() {
    let mut meter = LoudnessMeter::default_24k().unwrap();
    let tone: Vec<f32> = (0..24000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin() * 0.5)
        .collect();
    meter.feed(&tone);
    assert!(meter.integrated_loudness() > SILENCE_DB + 10.0);
    meter.reset();
    assert!(meter.integrated_loudness() <= SILENCE_DB + 1.0);
}

#[test]
fn test_a_weighting_mode() {
    let cfg = LoudnessConfig::default().with_weighting(LoudnessWeighting::AWeighting);
    let mut meter = LoudnessMeter::new(&cfg, 24000.0).unwrap();
    let tone: Vec<f32> = (0..24000)
        .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 24000.0).sin() * 0.5)
        .collect();
    let lufs = meter.measure_momentary(&tone);
    assert!(lufs > -15.0 && lufs < 0.0, "A-weighted 1kHz sine: {lufs}");
}

#[test]
fn test_flat_weighting_mode() {
    let cfg = LoudnessConfig::default().with_weighting(LoudnessWeighting::Flat);
    let mut meter = LoudnessMeter::new(&cfg, 24000.0).unwrap();
    let tone: Vec<f32> = (0..24000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin() * 0.5)
        .collect();
    let lufs = meter.measure_momentary(&tone);
    assert!(
        lufs > -15.0 && lufs < 0.0,
        "flat-weighted 440Hz sine: {lufs}"
    );
}

#[test]
fn test_nan_safety() {
    let mut meter = LoudnessMeter::default_24k().unwrap();
    let audio = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5, -0.5];
    let lufs = meter.measure_momentary(&audio);
    assert!(
        lufs.is_finite(),
        "NaN input should produce finite LUFS: {lufs}"
    );
    let tp = measure_true_peak(&audio, 24000.0);
    assert!(
        tp.is_finite(),
        "NaN input should produce finite true peak: {tp}"
    );
}

#[test]
fn test_48k_sample_rate() {
    let mut meter = LoudnessMeter::new(&LoudnessConfig::default(), 48000.0).unwrap();
    let tone: Vec<f32> = (0..48000)
        .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin() * 0.5)
        .collect();
    let lufs = meter.measure_momentary(&tone);
    assert!(lufs > -12.0 && lufs < 0.0, "48kHz 1kHz sine: {lufs}");
}

#[test]
fn test_22050_sample_rate() {
    let mut meter = LoudnessMeter::new(&LoudnessConfig::default(), 22050.0).unwrap();
    let tone: Vec<f32> = (0..22050)
        .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 22050.0).sin() * 0.5)
        .collect();
    let lufs = meter.measure_momentary(&tone);
    assert!(lufs > -12.0 && lufs < 0.0, "22050Hz 1kHz sine: {lufs}");
}

#[test]
fn test_invalid_sample_rate() {
    assert!(LoudnessMeter::new(&LoudnessConfig::default(), 0.0).is_err());
    assert!(LoudnessMeter::new(&LoudnessConfig::default(), -1.0).is_err());
    assert!(LoudnessMeter::new(&LoudnessConfig::default(), f32::NAN).is_err());
}

#[test]
fn test_measure_integrated_convenience() {
    let mut meter = LoudnessMeter::default_24k().unwrap();
    let tone: Vec<f32> = (0..24000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin() * 0.5)
        .collect();
    let lufs = meter.measure_integrated(&tone);
    assert!(lufs > SILENCE_DB + 10.0, "integrated convenience: {lufs}");
}

#[test]
fn test_config_accessors() {
    let meter = LoudnessMeter::default_24k().unwrap();
    assert_eq!(meter.sample_rate(), 24000.0);
    assert_eq!(meter.config().target_lufs, -16.0);
}

#[test]
fn test_bark_invalid_sample_rate() {
    let bands = bark_band_loudness(&[0.5; 100], -1.0);
    assert_eq!(bands.len(), 24);
    assert!(bands.iter().all(|&b| b <= SILENCE_DB + 1.0));
}
