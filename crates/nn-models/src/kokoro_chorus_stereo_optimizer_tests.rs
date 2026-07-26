// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

#[test]
fn test_config_default_validates() {
    StereoOptimizerConfig::default()
        .validate()
        .expect("default config should validate");
}

#[test]
fn test_config_natural_preset_validates() {
    StereoOptimizerConfig::natural()
        .validate()
        .expect("natural preset should validate");
}

#[test]
fn test_config_wide_preset_validates() {
    StereoOptimizerConfig::wide()
        .validate()
        .expect("wide preset should validate");
}

#[test]
fn test_config_broadcast_safe_preset_validates() {
    StereoOptimizerConfig::broadcast_safe()
        .validate()
        .expect("broadcast_safe should validate");
}

#[test]
fn test_config_headphone_preset_validates() {
    StereoOptimizerConfig::headphone()
        .validate()
        .expect("headphone preset should validate");
}

#[test]
fn test_config_invalid_target_width_nan() {
    let config = StereoOptimizerConfig::new().with_target_width(f32::NAN);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_invalid_target_width_out_of_range() {
    let config = StereoOptimizerConfig::new().with_target_width(1.5);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_invalid_min_correlation_inf() {
    let config = StereoOptimizerConfig::new().with_min_correlation(f32::INFINITY);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_invalid_bass_mono_freq_negative() {
    let config = StereoOptimizerConfig::new().with_bass_mono_freq_hz(-10.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_invalid_width_smoothing_nan() {
    let config = StereoOptimizerConfig::new().with_width_smoothing_ms(f32::NAN);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_invalid_mix_out_of_range() {
    let config = StereoOptimizerConfig::new().with_mix(2.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_builder_chain() {
    let config = StereoOptimizerConfig::new()
        .with_target_width(0.8)
        .with_min_correlation(0.2)
        .with_bass_mono_freq_hz(150.0)
        .with_width_smoothing_ms(40.0)
        .with_bass_mono(false)
        .with_mix(0.5);
    config.validate().expect("chained builder should validate");
    assert!((config.target_width - 0.8).abs() < 1e-6);
    assert!((config.min_correlation - 0.2).abs() < 1e-6);
    assert!((config.bass_mono_freq_hz - 150.0).abs() < 1e-6);
    assert!((config.width_smoothing_ms - 40.0).abs() < 1e-6);
    assert!(!config.enable_bass_mono);
    assert!((config.mix - 0.5).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Optimizer construction
// ---------------------------------------------------------------------------

#[test]
fn test_optimizer_new_valid() {
    let config = StereoOptimizerConfig::natural();
    StereoOptimizer::new(&config, 24000).expect("should construct with valid config");
}

#[test]
fn test_optimizer_new_invalid_sample_rate_low() {
    let config = StereoOptimizerConfig::natural();
    assert!(StereoOptimizer::new(&config, 100).is_err());
}

#[test]
fn test_optimizer_new_invalid_sample_rate_high() {
    let config = StereoOptimizerConfig::natural();
    assert!(StereoOptimizer::new(&config, 200_000).is_err());
}

// ---------------------------------------------------------------------------
// Correlation measurement
// ---------------------------------------------------------------------------

#[test]
fn test_correlation_identical_channels() {
    let config = StereoOptimizerConfig::natural();
    let mut opt = StereoOptimizer::new(&config, 24000).unwrap();

    // Identical L and R -> correlation should be ~1.0.
    let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut left = signal.clone();
    let mut right = signal;

    opt.process_stereo(&mut left, &mut right);
    assert!(
        opt.current_correlation() > 0.9,
        "identical channels should have high correlation, got {}",
        opt.current_correlation(),
    );
}

#[test]
fn test_correlation_inverted_channels() {
    let config = StereoOptimizerConfig::new()
        .with_target_width(1.0)
        .with_min_correlation(-1.0);
    let mut opt = StereoOptimizer::new(&config, 24000).unwrap();

    // Feed many blocks of anti-correlated content so the EMA converges.
    for _ in 0..50 {
        let signal: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.05).sin()).collect();
        let mut left = signal.clone();
        let mut right: Vec<f32> = signal.iter().map(|&s| -s).collect();
        opt.process_stereo(&mut left, &mut right);
    }

    assert!(
        opt.current_correlation() < 0.0,
        "inverted channels should have negative correlation, got {}",
        opt.current_correlation(),
    );
}

#[test]
fn test_correlation_tracking_initial_value() {
    let config = StereoOptimizerConfig::natural();
    let opt = StereoOptimizer::new(&config, 24000).unwrap();
    // Before any processing, correlation starts at 1.0.
    assert!((opt.current_correlation() - 1.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Width control
// ---------------------------------------------------------------------------

#[test]
fn test_zero_width_produces_mono() {
    let config = StereoOptimizerConfig::new()
        .with_target_width(0.0)
        .with_bass_mono(false);
    let mut opt = StereoOptimizer::new(&config, 24000).unwrap();

    let mut left = vec![1.0f32; 256];
    let mut right = vec![-1.0f32; 256];

    // Process enough blocks for width to converge (EMA).
    for _ in 0..20 {
        opt.process_stereo(&mut left, &mut right);
        left = vec![1.0f32; 256];
        right = vec![-1.0f32; 256];
    }

    opt.process_stereo(&mut left, &mut right);

    // With zero width, L and R should converge toward the mid signal.
    for i in 0..left.len() {
        let diff = (left[i] - right[i]).abs();
        assert!(
            diff < 0.15,
            "at sample {i}: L={}, R={}, diff={diff} — expected near-mono",
            left[i],
            right[i],
        );
    }
}

#[test]
fn test_full_width_preserves_stereo() {
    let config = StereoOptimizerConfig::new()
        .with_target_width(1.0)
        .with_min_correlation(-1.0) // Never narrow.
        .with_bass_mono(false);
    let mut opt = StereoOptimizer::new(&config, 24000).unwrap();

    let mut left: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut right: Vec<f32> = (0..512).map(|i| (i as f32 * 0.15).cos()).collect();
    let orig_left = left.clone();
    let orig_right = right.clone();

    opt.process_stereo(&mut left, &mut right);

    // Full width with no bass mono and mix=1 should approximately
    // preserve the original signal (width=1 means side*1.0).
    let max_diff: f32 = left
        .iter()
        .zip(orig_left.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 0.01,
        "full width should preserve left channel, max diff = {max_diff}",
    );
    let max_diff_r: f32 = right
        .iter()
        .zip(orig_right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff_r < 0.01,
        "full width should preserve right channel, max diff = {max_diff_r}",
    );
}

// ---------------------------------------------------------------------------
// Bass mono summing
// ---------------------------------------------------------------------------

#[test]
fn test_bass_mono_summing_reduces_bass_stereo() {
    let config = StereoOptimizerConfig::new()
        .with_target_width(1.0)
        .with_min_correlation(-1.0)
        .with_bass_mono(true)
        .with_bass_mono_freq_hz(200.0);
    let mut opt = StereoOptimizer::new(&config, 24000).unwrap();

    // Generate a low-frequency stereo signal (100 Hz < 200 Hz cutoff).
    let sr = 24000.0f32;
    let freq = 100.0f32;
    let n = 2400; // 100ms
    let mut left: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
        .collect();
    let mut right: Vec<f32> = (0..n)
        .map(|i| -(2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
        .collect();

    // Process multiple blocks to let the bass filter settle.
    for _ in 0..5 {
        opt.process_stereo(&mut left, &mut right);
    }

    // After bass mono summing, L and R should be closer in the low band.
    // Check the last quarter of samples (filter has settled).
    let quarter = n * 3 / 4;
    let avg_diff: f32 = left[quarter..]
        .iter()
        .zip(right[quarter..].iter())
        .map(|(l, r)| (l - r).abs())
        .sum::<f32>()
        / (n - quarter) as f32;
    // With bass mono, the difference should be reduced vs the original
    // (which was 2.0 peak-to-peak difference).
    assert!(
        avg_diff < 1.0,
        "bass mono summing should reduce LR difference, avg diff = {avg_diff}",
    );
}

#[test]
fn test_bass_mono_disabled_preserves_bass_stereo() {
    let config = StereoOptimizerConfig::new()
        .with_target_width(1.0)
        .with_min_correlation(-1.0)
        .with_bass_mono(false);
    let mut opt = StereoOptimizer::new(&config, 24000).unwrap();

    let n = 512;
    let mut left: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut right: Vec<f32> = (0..n).map(|i| -(i as f32 * 0.01).sin()).collect();
    let orig_left = left.clone();

    opt.process_stereo(&mut left, &mut right);

    // Without bass mono, channels should be preserved.
    let max_diff: f32 = left
        .iter()
        .zip(orig_left.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 0.01,
        "no bass mono should preserve signal, max diff = {max_diff}"
    );
}

// ---------------------------------------------------------------------------
// Width limiting on low correlation
// ---------------------------------------------------------------------------

#[test]
fn test_width_narrows_on_low_correlation() {
    let config = StereoOptimizerConfig::new()
        .with_target_width(0.9)
        .with_min_correlation(0.5)
        .with_bass_mono(false)
        .with_width_smoothing_ms(1.0); // Very fast response for test.
    let mut opt = StereoOptimizer::new(&config, 24000).unwrap();

    // Feed many blocks of anti-correlated content. We generate fresh
    // signal each iteration since process_stereo modifies the buffers.
    let n = 4096;
    for _ in 0..100 {
        let mut left: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut right: Vec<f32> = left.iter().map(|&s| -s).collect();
        opt.process_stereo(&mut left, &mut right);
    }

    // Width should have narrowed below the target.
    assert!(
        opt.effective_width() < config.target_width,
        "width should narrow on low correlation, got {} (target was {})",
        opt.effective_width(),
        config.target_width,
    );
}

// ---------------------------------------------------------------------------
// Mix (dry/wet)
// ---------------------------------------------------------------------------

#[test]
fn test_mix_zero_is_bypass() {
    let config = StereoOptimizerConfig::new()
        .with_target_width(0.0) // Would collapse to mono...
        .with_mix(0.0); // ...but mix=0 bypasses.
    let mut opt = StereoOptimizer::new(&config, 24000).unwrap();

    let mut left = vec![0.5f32; 128];
    let mut right = vec![-0.5f32; 128];
    let orig_left = left.clone();
    let orig_right = right.clone();

    opt.process_stereo(&mut left, &mut right);

    for i in 0..left.len() {
        assert!(
            (left[i] - orig_left[i]).abs() < 1e-6,
            "mix=0 should bypass: left[{i}] = {} vs {}",
            left[i],
            orig_left[i],
        );
        assert!(
            (right[i] - orig_right[i]).abs() < 1e-6,
            "mix=0 should bypass: right[{i}] = {} vs {}",
            right[i],
            orig_right[i],
        );
    }
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

#[test]
fn test_reset_restores_initial_state() {
    let config = StereoOptimizerConfig::natural();
    let mut opt = StereoOptimizer::new(&config, 24000).unwrap();

    // Process many blocks of varying data to change internal state.
    for _ in 0..50 {
        let mut left: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut right: Vec<f32> = left.iter().map(|&s| -s).collect();
        opt.process_stereo(&mut left, &mut right);
    }

    // Correlation EMA should have moved away from 1.0 after feeding
    // anti-correlated content.
    let corr_before_reset = opt.current_correlation();
    assert!(
        corr_before_reset < 0.99,
        "state should have changed, got correlation {corr_before_reset}",
    );

    opt.reset();

    assert!(
        (opt.current_correlation() - 1.0).abs() < 1e-6,
        "reset should restore correlation to 1.0",
    );
    assert!(
        (opt.effective_width() - config.target_width).abs() < 1e-6,
        "reset should restore effective width to target",
    );
}

// ---------------------------------------------------------------------------
// NaN / Inf safety
// ---------------------------------------------------------------------------

#[test]
fn test_nan_input_produces_zero() {
    let config = StereoOptimizerConfig::natural();
    let mut opt = StereoOptimizer::new(&config, 24000).unwrap();

    let mut left = vec![f32::NAN; 64];
    let mut right = vec![f32::NAN; 64];

    opt.process_stereo(&mut left, &mut right);

    for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
        assert!(l.is_finite(), "left[{i}] should be finite, got {l}");
        assert!(r.is_finite(), "right[{i}] should be finite, got {r}");
    }
}

#[test]
fn test_inf_input_produces_zero() {
    let config = StereoOptimizerConfig::natural();
    let mut opt = StereoOptimizer::new(&config, 24000).unwrap();

    let mut left = vec![f32::INFINITY; 64];
    let mut right = vec![f32::NEG_INFINITY; 64];

    opt.process_stereo(&mut left, &mut right);

    for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
        assert!(l.is_finite(), "left[{i}] should be finite, got {l}");
        assert!(r.is_finite(), "right[{i}] should be finite, got {r}");
    }
}

#[test]
fn test_mixed_nan_and_valid_samples() {
    let config = StereoOptimizerConfig::natural();
    let mut opt = StereoOptimizer::new(&config, 24000).unwrap();

    let mut left = vec![0.5, f32::NAN, 0.3, f32::INFINITY, 0.1];
    let mut right = vec![0.5, 0.3, f32::NAN, 0.1, f32::NEG_INFINITY];

    opt.process_stereo(&mut left, &mut right);

    for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
        assert!(l.is_finite(), "left[{i}] should be finite, got {l}");
        assert!(r.is_finite(), "right[{i}] should be finite, got {r}");
    }
}

// ---------------------------------------------------------------------------
// Empty / edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_empty_buffers_no_crash() {
    let config = StereoOptimizerConfig::natural();
    let mut opt = StereoOptimizer::new(&config, 24000).unwrap();

    let mut left: Vec<f32> = Vec::new();
    let mut right: Vec<f32> = Vec::new();
    opt.process_stereo(&mut left, &mut right);
    // Should not panic.
}

#[test]
fn test_silence_preserves_silence() {
    let config = StereoOptimizerConfig::natural();
    let mut opt = StereoOptimizer::new(&config, 24000).unwrap();

    let mut left = vec![0.0f32; 256];
    let mut right = vec![0.0f32; 256];

    opt.process_stereo(&mut left, &mut right);

    for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
        assert!(
            l.abs() < 1e-10,
            "silence should stay silent: left[{i}] = {l}",
        );
        assert!(
            r.abs() < 1e-10,
            "silence should stay silent: right[{i}] = {r}",
        );
    }
}

#[test]
fn test_single_sample() {
    let config = StereoOptimizerConfig::natural();
    let mut opt = StereoOptimizer::new(&config, 24000).unwrap();

    let mut left = vec![0.7f32];
    let mut right = vec![0.3f32];

    opt.process_stereo(&mut left, &mut right);

    assert!(left[0].is_finite());
    assert!(right[0].is_finite());
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

#[test]
fn test_accessors() {
    let config = StereoOptimizerConfig::wide();
    let opt = StereoOptimizer::new(&config, 48000).unwrap();

    assert!((opt.sample_rate() - 48000.0).abs() < 1e-3);
    assert!((opt.config().target_width - 1.0).abs() < 1e-6);
    assert!((opt.effective_width() - 1.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Preset differences
// ---------------------------------------------------------------------------

#[test]
fn test_presets_have_distinct_parameters() {
    let natural = StereoOptimizerConfig::natural();
    let wide = StereoOptimizerConfig::wide();
    let broadcast = StereoOptimizerConfig::broadcast_safe();
    let headphone = StereoOptimizerConfig::headphone();

    // Each preset should differ in at least one parameter.
    assert!(natural.target_width < wide.target_width);
    assert!(broadcast.min_correlation > natural.min_correlation);
    assert!(!headphone.enable_bass_mono);
    assert!(broadcast.bass_mono_freq_hz > natural.bass_mono_freq_hz);
}
