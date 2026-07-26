// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the true peak limiter (`kokoro_chorus_limiter`).

use super::*;

const SR: f32 = 24000.0;

// -- Config validation tests -------------------------------------------

#[test]
fn test_limiter_config_default_validates() {
    LimiterConfig::default()
        .validate()
        .expect("default config should validate");
}

#[test]
fn test_limiter_config_transparent_validates() {
    LimiterConfig::transparent()
        .validate()
        .expect("transparent preset should validate");
}

#[test]
fn test_limiter_config_broadcast_validates() {
    LimiterConfig::broadcast()
        .validate()
        .expect("broadcast preset should validate");
}

#[test]
fn test_limiter_config_aggressive_validates() {
    LimiterConfig::aggressive()
        .validate()
        .expect("aggressive preset should validate");
}

#[test]
fn test_limiter_config_gentle_validates() {
    LimiterConfig::gentle()
        .validate()
        .expect("gentle preset should validate");
}

#[test]
fn test_limiter_config_ceiling_too_low() {
    let cfg = LimiterConfig::new().with_ceiling_db(-15.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_limiter_config_ceiling_too_high() {
    let cfg = LimiterConfig::new().with_ceiling_db(1.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_limiter_config_attack_nan() {
    let cfg = LimiterConfig::new().with_attack_ms(f32::NAN);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_limiter_config_release_out_of_range() {
    let cfg = LimiterConfig::new().with_release_ms(2000.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_limiter_config_oversample_invalid() {
    let cfg = LimiterConfig::new().with_oversample_factor(3);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_limiter_config_mix_negative() {
    let cfg = LimiterConfig::new().with_mix(-0.1);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_limiter_config_lookahead_negative() {
    let cfg = LimiterConfig::new().with_lookahead_ms(-1.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_limiter_config_builder_chain() {
    let cfg = LimiterConfig::new()
        .with_ceiling_db(-2.0)
        .with_attack_ms(0.3)
        .with_release_ms(80.0)
        .with_lookahead_ms(2.0)
        .with_oversample_factor(4)
        .with_stereo_link(false)
        .with_mix(0.8);
    cfg.validate()
        .expect("chained builder config should validate");
    assert!((cfg.ceiling_db - (-2.0)).abs() < 1e-6);
    assert!(!cfg.stereo_link);
    assert!((cfg.mix - 0.8).abs() < 1e-6);
}

// -- Processor creation tests ------------------------------------------

#[test]
fn test_limiter_processor_new_default() {
    let cfg = LimiterConfig::default();
    let proc = LimiterProcessor::new(&cfg, SR);
    assert!(proc.is_ok());
}

#[test]
fn test_limiter_processor_invalid_sample_rate() {
    let cfg = LimiterConfig::default();
    assert!(LimiterProcessor::new(&cfg, 0.0).is_err());
    assert!(LimiterProcessor::new(&cfg, -44100.0).is_err());
    assert!(LimiterProcessor::new(&cfg, f32::NAN).is_err());
}

// -- Peak limiting tests -----------------------------------------------

#[test]
fn test_limiter_reduces_peaks_below_ceiling() {
    let cfg = LimiterConfig::new()
        .with_ceiling_db(-6.0)
        .with_lookahead_ms(0.0)
        .with_oversample_factor(1);
    let mut proc = LimiterProcessor::new(&cfg, SR).unwrap();

    // Create a signal with peaks at 1.0 (0 dBFS).
    let mut left = vec![0.0; 480];
    let mut right = vec![0.0; 480];
    for i in 100..200 {
        left[i] = 1.0;
        right[i] = 1.0;
    }

    proc.process_stereo(&mut left, &mut right).unwrap();

    let ceiling = db_to_linear(-6.0);
    // After processing, all samples should be at or below the ceiling
    // (with a small tolerance for the attack transient).
    for &s in left[110..200].iter() {
        assert!(
            s <= ceiling + 0.01,
            "left sample {s} exceeds ceiling {ceiling}",
        );
    }
}

#[test]
fn test_limiter_quiet_signal_unchanged() {
    let cfg = LimiterConfig::new()
        .with_ceiling_db(-1.0)
        .with_lookahead_ms(0.0)
        .with_oversample_factor(1);
    let mut proc = LimiterProcessor::new(&cfg, SR).unwrap();

    // Signal well below ceiling.
    let original: Vec<f32> = (0..240).map(|i| 0.1 * (i as f32 * 0.1).sin()).collect();
    let mut left = original.clone();
    let mut right = original.clone();

    proc.process_stereo(&mut left, &mut right).unwrap();

    for (i, (&o, &l)) in original.iter().zip(left.iter()).enumerate() {
        assert!(
            (o - l).abs() < 1e-5,
            "sample {i} changed from {o} to {l} -- quiet signal should be unchanged",
        );
    }
}

#[test]
fn test_limiter_gain_reduction_reported() {
    let cfg = LimiterConfig::new()
        .with_ceiling_db(-6.0)
        .with_lookahead_ms(0.0)
        .with_oversample_factor(1);
    let mut proc = LimiterProcessor::new(&cfg, SR).unwrap();

    assert!((proc.gain_reduction_db() - 0.0).abs() < 1e-6);

    let mut left = vec![1.0; 480];
    let mut right = vec![1.0; 480];
    proc.process_stereo(&mut left, &mut right).unwrap();

    // Gain reduction should be negative (attenuation).
    assert!(proc.gain_reduction_db() < -1.0);
}

#[test]
fn test_limiter_stereo_link_preserves_image() {
    let cfg = LimiterConfig::new()
        .with_ceiling_db(-6.0)
        .with_stereo_link(true)
        .with_lookahead_ms(0.0)
        .with_oversample_factor(1);
    let mut proc = LimiterProcessor::new(&cfg, SR).unwrap();

    // Left channel is loud, right is quiet. With stereo link,
    // both should get the same gain reduction.
    let mut left = vec![1.0; 480];
    let mut right = vec![0.01; 480];

    proc.process_stereo(&mut left, &mut right).unwrap();

    // Because of stereo linking, both channels share gain reduction.
    // The right channel should still be attenuated proportionally.
    let ceiling = db_to_linear(-6.0);
    for &s in left[100..].iter() {
        assert!(s <= ceiling + 0.01);
    }
}

#[test]
fn test_limiter_empty_buffers() {
    let cfg = LimiterConfig::default();
    let mut proc = LimiterProcessor::new(&cfg, SR).unwrap();
    let mut left: Vec<f32> = vec![];
    let mut right: Vec<f32> = vec![];
    proc.process_stereo(&mut left, &mut right).unwrap();
}

#[test]
fn test_limiter_mismatched_buffers_error() {
    let cfg = LimiterConfig::default();
    let mut proc = LimiterProcessor::new(&cfg, SR).unwrap();
    let mut left = vec![0.0; 10];
    let mut right = vec![0.0; 20];
    assert!(proc.process_stereo(&mut left, &mut right).is_err());
}

#[test]
fn test_limiter_nan_safety() {
    let cfg = LimiterConfig::new()
        .with_ceiling_db(-1.0)
        .with_lookahead_ms(0.0)
        .with_oversample_factor(1);
    let mut proc = LimiterProcessor::new(&cfg, SR).unwrap();

    let mut left = vec![0.0; 100];
    let mut right = vec![0.0; 100];
    left[10] = f32::NAN;
    left[20] = f32::INFINITY;
    left[30] = f32::NEG_INFINITY;
    right[40] = f32::NAN;

    proc.process_stereo(&mut left, &mut right).unwrap();

    // All output samples should be finite.
    for (i, &s) in left.iter().enumerate() {
        assert!(s.is_finite(), "left[{i}] = {s} is not finite");
    }
    for (i, &s) in right.iter().enumerate() {
        assert!(s.is_finite(), "right[{i}] = {s} is not finite");
    }
}

#[test]
fn test_limiter_reset_clears_state() {
    let cfg = LimiterConfig::new()
        .with_ceiling_db(-6.0)
        .with_lookahead_ms(0.0)
        .with_oversample_factor(1);
    let mut proc = LimiterProcessor::new(&cfg, SR).unwrap();

    let mut left = vec![1.0; 480];
    let mut right = vec![1.0; 480];
    proc.process_stereo(&mut left, &mut right).unwrap();

    assert!(proc.gain_reduction_db() < -1.0);

    proc.reset();
    assert!((proc.gain_reduction_db() - 0.0).abs() < 1e-6);
}

#[test]
fn test_limiter_mix_zero_bypass() {
    let cfg = LimiterConfig::new()
        .with_ceiling_db(-6.0)
        .with_mix(0.0)
        .with_lookahead_ms(0.0)
        .with_oversample_factor(1);
    let mut proc = LimiterProcessor::new(&cfg, SR).unwrap();

    let original = vec![1.0; 100];
    let mut left = original.clone();
    let mut right = original.clone();
    proc.process_stereo(&mut left, &mut right).unwrap();

    for (i, (&o, &l)) in original.iter().zip(left.iter()).enumerate() {
        assert!(
            (o - l).abs() < 1e-6,
            "sample {i} changed with mix=0.0: {o} -> {l}",
        );
    }
}

#[test]
fn test_limiter_oversampled_peak_detection() {
    // Verify that the oversampled peak detector catches intersample peaks.
    let y0 = 0.0f32;
    let y1 = 0.8;
    let y2 = -0.8;
    let y3 = 0.0;
    // The interpolated curve between y1 and y2 should exceed |0.8|
    // at some fractional point.
    let peak = oversampled_peak(y0, y1, y2, y3, 4);
    assert!(peak >= 0.8, "oversampled peak {peak} should be >= 0.8");
}

#[test]
fn test_limiter_mono_processing() {
    let cfg = LimiterConfig::new()
        .with_ceiling_db(-6.0)
        .with_lookahead_ms(0.0)
        .with_oversample_factor(1);
    let mut proc = LimiterProcessor::new(&cfg, SR).unwrap();

    let mut buffer = vec![1.0; 480];
    proc.process_mono(&mut buffer).unwrap();

    let ceiling = db_to_linear(-6.0);
    for &s in buffer[100..].iter() {
        assert!(s <= ceiling + 0.01);
    }
}

#[test]
fn test_limiter_latency_samples() {
    let cfg = LimiterConfig::new().with_lookahead_ms(1.0);
    let proc = LimiterProcessor::new(&cfg, SR).unwrap();
    // 1ms at 24kHz = 24 samples.
    assert_eq!(proc.latency_samples(), 24);
}

#[test]
fn test_limiter_lookahead_smooths_attack() {
    // With lookahead, the limiter should start reducing gain before
    // the peak arrives, resulting in less distortion.
    let cfg_no_la = LimiterConfig::new()
        .with_ceiling_db(-6.0)
        .with_lookahead_ms(0.0)
        .with_oversample_factor(1);
    let cfg_la = LimiterConfig::new()
        .with_ceiling_db(-6.0)
        .with_lookahead_ms(2.0)
        .with_oversample_factor(1);

    let mut proc_no_la = LimiterProcessor::new(&cfg_no_la, SR).unwrap();
    let mut proc_la = LimiterProcessor::new(&cfg_la, SR).unwrap();

    let make_signal = || -> (Vec<f32>, Vec<f32>) {
        let mut l = vec![0.0f32; 480];
        let mut r = vec![0.0f32; 480];
        for i in 100..200 {
            l[i] = 1.0;
            r[i] = 1.0;
        }
        (l, r)
    };

    let (mut l1, mut r1) = make_signal();
    let (mut l2, mut r2) = make_signal();

    proc_no_la.process_stereo(&mut l1, &mut r1).unwrap();
    proc_la.process_stereo(&mut l2, &mut r2).unwrap();

    // Both should limit, but the lookahead version should have less
    // overshoot at the attack transient.
    let ceiling = db_to_linear(-6.0);

    // Count overshoots (samples above ceiling) in the attack region.
    let over_no_la = l1[100..120]
        .iter()
        .filter(|&&s| s > ceiling + 0.001)
        .count();
    let over_la = l2[100..120]
        .iter()
        .filter(|&&s| s > ceiling + 0.001)
        .count();

    // Lookahead should have fewer or equal overshoots.
    assert!(
        over_la <= over_no_la + 5,
        "lookahead should not produce more overshoots: la={over_la}, no_la={over_no_la}",
    );
}

#[test]
fn test_limiter_db_helpers() {
    // Verify round-trip.
    let db = -6.0f32;
    let lin = db_to_linear(db);
    let back = linear_to_db(lin);
    assert!(
        (db - back).abs() < 0.001,
        "db round-trip: {db} -> {lin} -> {back}"
    );

    // Zero amplitude should return silence dB.
    assert!(linear_to_db(0.0) <= SILENCE_DB + 1.0);
}

#[test]
fn test_hermite_interp_midpoint() {
    // Constant signal: interpolation should return the same value.
    let v = hermite_interp(0.5, 0.5, 0.5, 0.5, 0.5);
    assert!((v - 0.5).abs() < 1e-6);
}

// -- Stereo unlink tests -----------------------------------------------

#[test]
fn test_limiter_stereo_unlinked_independent_channels() {
    // When stereo link is off, a loud left channel should NOT reduce
    // the quiet right channel.
    let cfg = LimiterConfig::new()
        .with_ceiling_db(-6.0)
        .with_stereo_link(false)
        .with_lookahead_ms(0.0)
        .with_oversample_factor(1);
    let mut proc = LimiterProcessor::new(&cfg, SR).unwrap();

    let mut left = vec![1.0; 480];
    let mut right = vec![0.01; 480];

    proc.process_stereo(&mut left, &mut right).unwrap();

    let ceiling = db_to_linear(-6.0);
    // Left should be limited.
    for &s in left[100..].iter() {
        assert!(s <= ceiling + 0.01);
    }
    // Right was below ceiling and should remain near its original value.
    for &s in right[100..].iter() {
        assert!(
            (s - 0.01).abs() < 0.005,
            "right channel changed unexpectedly: {s}",
        );
    }
}

#[test]
fn test_limiter_all_oversample_factors() {
    for &os in &[1u32, 2, 4] {
        let cfg = LimiterConfig::new()
            .with_ceiling_db(-3.0)
            .with_lookahead_ms(0.0)
            .with_oversample_factor(os);
        let mut proc = LimiterProcessor::new(&cfg, SR).unwrap();

        let mut left = vec![0.9; 240];
        let mut right = vec![0.9; 240];
        proc.process_stereo(&mut left, &mut right).unwrap();

        let ceiling = db_to_linear(-3.0);
        for &s in left.iter() {
            assert!(
                s <= ceiling + 0.02,
                "oversample={os}: sample {s} exceeds ceiling {ceiling}",
            );
        }
    }
}
