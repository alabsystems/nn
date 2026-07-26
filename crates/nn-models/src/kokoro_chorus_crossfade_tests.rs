// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::kokoro_chorus_crossfade`].

use super::*;

#[test]
fn test_config_default_validates() {
    CrossfadeOptimizerConfig::default()
        .validate()
        .expect("default config should be valid");
}

#[test]
fn test_config_builder_roundtrip() {
    let cfg = CrossfadeOptimizerConfig::builder()
        .analysis_mode(CrossfadeAnalysis::PhaseAligned)
        .min_crossfade_samples(240)
        .max_crossfade_samples(4800)
        .target_snr_db(50.0)
        .zero_crossing_search(false)
        .spectral_match(true)
        .build()
        .expect("valid builder config");
    assert_eq!(cfg.analysis_mode, CrossfadeAnalysis::PhaseAligned);
    assert_eq!(cfg.min_crossfade_samples, 240);
    assert_eq!(cfg.max_crossfade_samples, 4800);
    assert!(!cfg.zero_crossing_search);
    assert!(cfg.spectral_match);
}

#[test]
fn test_config_rejects_invalid_min() {
    let res = CrossfadeOptimizerConfig::builder()
        .min_crossfade_samples(10)
        .build();
    assert!(res.is_err());
}

#[test]
fn test_config_rejects_max_less_than_min() {
    let res = CrossfadeOptimizerConfig::builder()
        .min_crossfade_samples(480)
        .max_crossfade_samples(100)
        .build();
    assert!(res.is_err());
}

#[test]
fn test_config_rejects_nan_snr() {
    let res = CrossfadeOptimizerConfig::builder()
        .target_snr_db(f32::NAN)
        .build();
    assert!(res.is_err());
}

#[test]
fn test_zero_crossings_sine() {
    // Simple sine wave should have crossings at expected positions.
    let n = 100;
    let audio: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * i as f32 / 50.0).sin())
        .collect();
    let zc = find_zero_crossings(&audio);
    // Expect crossings near samples 0, 25, 50, 75.
    assert!(!zc.is_empty());
    // At least 2 crossings per period, 2 periods = 4 crossings.
    assert!(zc.len() >= 3, "got {} crossings", zc.len());
}

#[test]
fn test_zero_crossings_empty() {
    assert!(find_zero_crossings(&[]).is_empty());
    assert!(find_zero_crossings(&[1.0]).is_empty());
}

#[test]
fn test_energy_envelope_constant() {
    let audio = vec![0.5_f32; 100];
    let env = compute_energy_envelope(&audio, 10);
    assert_eq!(env.len(), 100);
    // Steady-state energy should be 0.25.
    let last = *env.last().expect("non-empty");
    assert!((last - 0.25).abs() < 0.01, "expected ~0.25, got {last}");
}

#[test]
fn test_energy_envelope_empty() {
    let env = compute_energy_envelope(&[], 10);
    assert!(env.is_empty());
}

#[test]
fn test_adaptive_window_symmetric() {
    let (fo, fi) = generate_adaptive_window(100, 100);
    assert_eq!(fo.len(), 100);
    assert_eq!(fi.len(), 100);
    // fade_out starts near 1.0, ends near 0.0.
    assert!((fo[0] - 1.0).abs() < 0.01);
    assert!(fo[99] < 0.01);
    // fade_in starts near 0.0, ends near 1.0.
    assert!(fi[0] < 0.01);
    assert!((fi[99] - 1.0).abs() < 0.01);
}

#[test]
fn test_adaptive_window_empty() {
    let (fo, fi) = generate_adaptive_window(0, 0);
    assert!(fo.is_empty());
    assert!(fi.is_empty());
}

#[test]
fn test_optimizer_push_flush_roundtrip() {
    let cfg = CrossfadeOptimizerConfig::builder()
        .analysis_mode(CrossfadeAnalysis::Fixed)
        .min_crossfade_samples(48)
        .max_crossfade_samples(48)
        .zero_crossing_search(false)
        .build()
        .expect("valid config");
    let mut opt = CrossfadeOptimizer::new(cfg).expect("valid optimizer");

    // First chunk: 200 samples.
    let chunk1 = vec![0.5_f32; 200];
    let out1 = opt.push_chunk(&chunk1).expect("push_chunk 1");
    // First push returns prefix (200 - 48 = 152 samples).
    assert_eq!(out1.len(), 152);

    // Second chunk: 200 samples.
    let chunk2 = vec![0.3_f32; 200];
    let out2 = opt.push_chunk(&chunk2).expect("push_chunk 2");
    // Should get prev_tail crossfaded with new chunk.
    assert!(!out2.is_empty());

    // Flush: returns retained tail.
    let tail = opt.flush();
    assert!(tail.is_some());
    assert_eq!(tail.expect("tail").len(), 48);
}

#[test]
fn test_optimizer_reset() {
    let cfg = CrossfadeOptimizerConfig::default();
    let mut opt = CrossfadeOptimizer::new(cfg).expect("valid");
    let _ = opt.push_chunk(&vec![0.1; 600]);
    assert!(opt.prev_chunk_tail.is_some());
    opt.reset();
    assert!(opt.prev_chunk_tail.is_none());
}

#[test]
fn test_optimize_crossfade_output_bounded() {
    // With bounded input, output should remain bounded.
    let cfg = CrossfadeOptimizerConfig::builder()
        .analysis_mode(CrossfadeAnalysis::EnergyAdaptive)
        .min_crossfade_samples(48)
        .max_crossfade_samples(96)
        .zero_crossing_search(false)
        .build()
        .expect("valid config");
    let mut opt = CrossfadeOptimizer::new(cfg).expect("valid");

    let prev = vec![0.8_f32; 100];
    let next = vec![-0.3_f32; 100];
    let out = opt.optimize_and_crossfade(&prev, &next).expect("crossfade");
    for &s in &out {
        assert!(s.is_finite(), "non-finite sample in crossfade output: {s}");
        assert!(s.abs() <= 1.5, "output sample out of reasonable range: {s}");
    }
}

#[test]
fn test_phase_aligned_mode() {
    // Smoke test that phase-aligned mode doesn't crash.
    let cfg = CrossfadeOptimizerConfig::builder()
        .analysis_mode(CrossfadeAnalysis::PhaseAligned)
        .min_crossfade_samples(48)
        .max_crossfade_samples(240)
        .build()
        .expect("valid config");
    let mut opt = CrossfadeOptimizer::new(cfg).expect("valid");

    let sine: Vec<f32> = (0..500)
        .map(|i| (2.0 * std::f32::consts::PI * i as f32 / 100.0).sin())
        .collect();
    let _ = opt.push_chunk(&sine[..250]).expect("chunk 1");
    let out = opt.push_chunk(&sine[200..]).expect("chunk 2");
    assert!(!out.is_empty());
}

#[test]
fn test_spectral_match_mode() {
    let cfg = CrossfadeOptimizerConfig::builder()
        .analysis_mode(CrossfadeAnalysis::SpectralMatch)
        .min_crossfade_samples(48)
        .max_crossfade_samples(240)
        .spectral_match(true)
        .build()
        .expect("valid config");
    let mut opt = CrossfadeOptimizer::new(cfg).expect("valid");

    let chunk = vec![0.1_f32; 300];
    let _ = opt.push_chunk(&chunk).expect("chunk 1");
    let out = opt.push_chunk(&chunk).expect("chunk 2");
    assert!(!out.is_empty());
}
