// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `kokoro_chorus_stereo_analysis`.

use super::*;

const SR: f32 = 24000.0;

// -- Config tests --

#[test]
fn test_config_default_is_valid() {
    StereoAnalysisConfig::default()
        .validate()
        .expect("default config should be valid");
}

#[test]
fn test_config_builder_chain() {
    let cfg = StereoAnalysisConfig::new()
        .with_monitoring(true)
        .with_correction(true)
        .with_min_correlation(-0.5)
        .with_target_correlation(0.5)
        .with_correction_strength(0.8)
        .with_bass_mono_below_hz(100.0)
        .with_sample_rate(48000.0);
    cfg.validate().expect("builder config should be valid");
    assert!(cfg.enable_monitoring);
    assert!(cfg.enable_correction);
    assert!((cfg.min_correlation - (-0.5)).abs() < 1e-6);
}

#[test]
fn test_config_rejects_nan_correlation() {
    let cfg = StereoAnalysisConfig::new().with_min_correlation(f32::NAN);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rejects_out_of_range_strength() {
    let cfg = StereoAnalysisConfig::new().with_correction_strength(1.5);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rejects_min_above_target() {
    let cfg = StereoAnalysisConfig::new()
        .with_min_correlation(0.8)
        .with_target_correlation(0.3);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rejects_negative_bass_freq() {
    let cfg = StereoAnalysisConfig::new().with_bass_mono_below_hz(-10.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rejects_invalid_sample_rate() {
    let cfg = StereoAnalysisConfig::new().with_sample_rate(500.0);
    assert!(cfg.validate().is_err());
}

// -- Analyzer construction --

#[test]
fn test_analyzer_new_valid() {
    let cfg = StereoAnalysisConfig::default();
    StereoAnalyzer::new(cfg).expect("valid config");
}

#[test]
fn test_analyzer_new_rejects_invalid() {
    let cfg = StereoAnalysisConfig::new().with_sample_rate(0.0);
    assert!(StereoAnalyzer::new(cfg).is_err());
}

// -- Analysis: correlation --

#[test]
fn test_identical_channels_correlation_one() {
    let cfg = StereoAnalysisConfig::default();
    let analyzer = StereoAnalyzer::new(cfg).unwrap();

    let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
    let metrics = analyzer.analyze(&signal, &signal);

    assert!(
        (metrics.correlation - 1.0).abs() < 0.01,
        "identical channels should have correlation ~1.0, got {}",
        metrics.correlation,
    );
    assert!(metrics.mono_compatible);
}

#[test]
fn test_inverted_channels_correlation_neg_one() {
    let cfg = StereoAnalysisConfig::default();
    let analyzer = StereoAnalyzer::new(cfg).unwrap();

    let left: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
    let right: Vec<f32> = left.iter().map(|x| -x).collect();
    let metrics = analyzer.analyze(&left, &right);

    assert!(
        metrics.correlation < -0.9,
        "inverted channels should have correlation near -1.0, got {}",
        metrics.correlation,
    );
    assert!(!metrics.mono_compatible);
}

#[test]
fn test_uncorrelated_channels() {
    let cfg = StereoAnalysisConfig::default();
    let analyzer = StereoAnalyzer::new(cfg).unwrap();

    // Two sinusoids at unrelated frequencies.
    let n = 4800;
    let left: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / SR).sin())
        .collect();
    let right: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 1170.0 / SR).sin())
        .collect();
    let metrics = analyzer.analyze(&left, &right);

    assert!(
        metrics.correlation.abs() < 0.3,
        "unrelated frequencies should have low correlation, got {}",
        metrics.correlation,
    );
}

// -- Analysis: mid/side levels --

#[test]
fn test_mono_signal_has_no_side() {
    let cfg = StereoAnalysisConfig::default();
    let analyzer = StereoAnalyzer::new(cfg).unwrap();

    let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.05).sin()).collect();
    let metrics = analyzer.analyze(&signal, &signal);

    // Side should be silence (neg infinity dB).
    assert!(
        metrics.side_level_db < -80.0,
        "mono signal should have very low side level, got {} dB",
        metrics.side_level_db,
    );
    // Mid should be substantial.
    assert!(
        metrics.mid_level_db > -20.0,
        "mono signal should have substantial mid level, got {} dB",
        metrics.mid_level_db,
    );
}

// -- Analysis: phase offset --

#[test]
fn test_identical_channels_zero_phase() {
    let cfg = StereoAnalysisConfig::default();
    let analyzer = StereoAnalyzer::new(cfg).unwrap();

    let signal: Vec<f32> = (0..1024)
        .map(|i| (i as f32 * 0.1).sin().max(0.01)) // keep positive
        .collect();
    let metrics = analyzer.analyze(&signal, &signal);

    assert!(
        metrics.phase_offset_deg < 5.0,
        "identical channels should have near-zero phase offset, got {} deg",
        metrics.phase_offset_deg,
    );
}

// -- Analysis: empty/edge cases --

#[test]
fn test_analyze_empty_buffers() {
    let cfg = StereoAnalysisConfig::default();
    let analyzer = StereoAnalyzer::new(cfg).unwrap();
    let metrics = analyzer.analyze(&[], &[]);
    assert!((metrics.correlation - 1.0).abs() < 1e-6);
    assert!(metrics.mono_compatible);
}

#[test]
fn test_analyze_nan_input() {
    let cfg = StereoAnalysisConfig::default();
    let analyzer = StereoAnalyzer::new(cfg).unwrap();
    let left = vec![f32::NAN; 100];
    let right = vec![0.5; 100];
    let metrics = analyzer.analyze(&left, &right);
    // Should not panic; correlation defaults to 1.0 for all-nan.
    assert!(metrics.correlation.is_finite());
}

#[test]
fn test_monitoring_disabled_returns_default() {
    let cfg = StereoAnalysisConfig::new().with_monitoring(false);
    let analyzer = StereoAnalyzer::new(cfg).unwrap();
    let left = vec![0.5; 100];
    let right = vec![-0.5; 100];
    let metrics = analyzer.analyze(&left, &right);
    assert!((metrics.correlation - 1.0).abs() < 1e-6);
}

// -- Correction: mid/side narrowing --

#[test]
fn test_correction_narrows_anticorrelated() {
    let cfg = StereoAnalysisConfig::new()
        .with_correction(true)
        .with_min_correlation(0.0)
        .with_correction_strength(1.0)
        .with_bass_mono_below_hz(0.0); // Disable bass mono for this test.
    let mut analyzer = StereoAnalyzer::new(cfg).unwrap();

    // Anti-correlated signal.
    let n = 1024;
    let mut left: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut right: Vec<f32> = left.iter().map(|x| -x).collect();

    analyzer.correct(&mut left, &mut right);

    // With strength=1.0, side should be zeroed -> L == R.
    for i in 0..n {
        assert!(
            (left[i] - right[i]).abs() < 1e-6,
            "strength=1.0 should collapse to mono at sample {i}"
        );
    }
}

#[test]
fn test_correction_disabled_no_change() {
    let cfg = StereoAnalysisConfig::new().with_correction(false);
    let mut analyzer = StereoAnalyzer::new(cfg).unwrap();

    let orig = vec![0.5, -0.3, 0.7];
    let mut left = orig.clone();
    let mut right = vec![-0.5, 0.3, -0.7];
    let orig_right = right.clone();

    analyzer.correct(&mut left, &mut right);

    assert_eq!(left, orig);
    assert_eq!(right, orig_right);
}

// -- Correction: bass mono enforcement --

#[test]
fn test_bass_mono_enforcement() {
    let cfg = StereoAnalysisConfig::new()
        .with_correction(true)
        .with_min_correlation(-1.0) // Don't trigger broadband correction.
        .with_bass_mono_below_hz(200.0)
        .with_sample_rate(SR);
    let mut analyzer = StereoAnalyzer::new(cfg).unwrap();

    // Low-frequency stereo signal (100 Hz).
    let n = 4800;
    let freq = 100.0;
    let mut left: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / SR;
            (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect();
    let mut right: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / SR;
            (2.0 * std::f32::consts::PI * freq * t + 1.0).sin()
        })
        .collect();

    analyzer.correct(&mut left, &mut right);

    // After settling, bass L and R should be closer together.
    let settle = n * 3 / 4;
    let max_diff: f32 = (settle..n)
        .map(|i| (left[i] - right[i]).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 0.4,
        "bass mono enforcement should narrow 100Hz stereo, max diff = {max_diff}"
    );
}

// -- Process (analyze + correct) --

#[test]
fn test_process_returns_pre_correction_metrics() {
    let cfg = StereoAnalysisConfig::new()
        .with_monitoring(true)
        .with_correction(true)
        .with_min_correlation(0.0)
        .with_correction_strength(1.0)
        .with_bass_mono_below_hz(0.0);
    let mut analyzer = StereoAnalyzer::new(cfg).unwrap();

    let n = 1024;
    let mut left: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut right: Vec<f32> = left.iter().map(|x| -x).collect();

    let metrics = analyzer.process(&mut left, &mut right);

    // Metrics should reflect the pre-correction state (anti-correlated).
    assert!(
        metrics.correlation < -0.9,
        "process should return pre-correction metrics, got correlation {}",
        metrics.correlation,
    );
    // But the audio should now be corrected (mono).
    for i in 0..n {
        assert!(
            (left[i] - right[i]).abs() < 1e-6,
            "audio should be corrected after process"
        );
    }
}

// -- Reset --

#[test]
fn test_reset_clears_filter_state() {
    let cfg = StereoAnalysisConfig::new()
        .with_correction(true)
        .with_bass_mono_below_hz(200.0);
    let mut analyzer = StereoAnalyzer::new(cfg).unwrap();

    // Feed some signal to build filter state.
    let mut left = vec![1.0; 200];
    let mut right = vec![-1.0; 200];
    analyzer.correct(&mut left, &mut right);

    // Reset.
    analyzer.reset();

    // Process silence -> should produce silence.
    let mut left = vec![0.0; 200];
    let mut right = vec![0.0; 200];
    analyzer.correct(&mut left, &mut right);

    let max_val = left
        .iter()
        .chain(right.iter())
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_val < 1e-10,
        "after reset + silence, output should be silent, got max={max_val}"
    );
}

// -- rms_to_db --

#[test]
fn test_rms_to_db_unity() {
    let db = rms_to_db(1.0);
    assert!((db - 0.0).abs() < 1e-6, "rms=1.0 should be 0 dBFS");
}

#[test]
fn test_rms_to_db_zero() {
    let db = rms_to_db(0.0);
    assert!(db == f32::NEG_INFINITY);
}

#[test]
fn test_rms_to_db_nan() {
    let db = rms_to_db(f32::NAN);
    assert!(db == f32::NEG_INFINITY);
}
