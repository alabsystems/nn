// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for chorus configuration and audio mixing.

use super::*;

// ---------------------------------------------------------------------------
// ChorusConfig
// ---------------------------------------------------------------------------

#[test]
fn test_equal_gain_8_voices() {
    let config = ChorusConfig::equal_gain(8).unwrap();
    assert_eq!(config.n_voices, 8);
    assert_eq!(config.gains.len(), 8);
    for &g in &config.gains {
        assert!((g - 0.125).abs() < 1e-7);
    }
    assert!(config.clip_output);
}

#[test]
fn test_equal_gain_1_voice() {
    let config = ChorusConfig::equal_gain(1).unwrap();
    assert_eq!(config.n_voices, 1);
    assert!((config.gains[0] - 1.0).abs() < 1e-7);
}

#[test]
fn test_equal_gain_zero_rejected() {
    assert!(ChorusConfig::equal_gain(0).is_err());
}

#[test]
fn test_equal_gain_33_rejected() {
    assert!(ChorusConfig::equal_gain(33).is_err());
}

#[test]
fn test_with_gains_custom() {
    let config = ChorusConfig::with_gains(vec![0.5, 0.3, 0.2]).unwrap();
    assert_eq!(config.n_voices, 3);
    assert_eq!(config.gains, vec![0.5, 0.3, 0.2]);
}

#[test]
fn test_with_gains_negative_rejected() {
    assert!(ChorusConfig::with_gains(vec![0.5, -0.1]).is_err());
}

#[test]
fn test_with_gains_above_one_rejected() {
    let result = ChorusConfig::with_gains(vec![0.5, 1.5]);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("gain[1]"),
        "error should identify the offending gain: {msg}"
    );
}

#[test]
fn test_with_gains_nan_rejected() {
    assert!(ChorusConfig::with_gains(vec![f32::NAN]).is_err());
}

#[test]
fn test_with_gains_inf_rejected() {
    assert!(ChorusConfig::with_gains(vec![f32::INFINITY]).is_err());
}

#[test]
fn test_validate_mismatch() {
    let mut config = ChorusConfig::equal_gain(4).unwrap();
    config.gains.push(0.1); // 5 gains for 4 voices
    assert!(config.validate().is_err());
}

#[test]
fn test_with_clip() {
    let config = ChorusConfig::equal_gain(2).unwrap().with_clip(false);
    assert!(!config.clip_output);
}

// ---------------------------------------------------------------------------
// mix_voices
// ---------------------------------------------------------------------------

#[test]
fn test_mix_single_voice_passthrough() {
    let audio = vec![vec![0.5, -0.3, 0.8]];
    let gains = vec![1.0];
    let mixed = mix_voices(&audio, &gains, false).unwrap();
    assert_eq!(mixed.len(), 3);
    for (i, (&m, &a)) in mixed.iter().zip(audio[0].iter()).enumerate() {
        assert!((m - a).abs() < 1e-7, "sample {i}: expected {a}, got {m}");
    }
}

#[test]
fn test_mix_two_equal_voices() {
    let v1 = vec![1.0, 0.0, -1.0];
    let v2 = vec![0.0, 1.0, -1.0];
    let gains = vec![0.5, 0.5];
    let mixed = mix_voices(&[v1, v2], &gains, false).unwrap();
    assert!((mixed[0] - 0.5).abs() < 1e-7); // 1.0*0.5 + 0.0*0.5
    assert!((mixed[1] - 0.5).abs() < 1e-7); // 0.0*0.5 + 1.0*0.5
    assert!((mixed[2] - (-1.0)).abs() < 1e-7); // -1.0*0.5 + -1.0*0.5
}

#[test]
fn test_mix_different_lengths_zero_pads() {
    let v1 = vec![1.0, 1.0, 1.0, 1.0];
    let v2 = vec![0.5, 0.5];
    let gains = vec![1.0, 1.0];
    let mixed = mix_voices(&[v1, v2], &gains, false).unwrap();
    assert_eq!(mixed.len(), 4);
    assert!((mixed[0] - 1.5).abs() < 1e-7); // 1.0 + 0.5
    assert!((mixed[1] - 1.5).abs() < 1e-7); // 1.0 + 0.5
    assert!((mixed[2] - 1.0).abs() < 1e-7); // 1.0 + 0 (zero-padded)
    assert!((mixed[3] - 1.0).abs() < 1e-7); // 1.0 + 0
}

#[test]
fn test_mix_clipping() {
    // Two voices at full volume should clip to 1.0.
    let v1 = vec![0.8];
    let v2 = vec![0.8];
    let gains = vec![1.0, 1.0];

    let mixed_unclipped = mix_voices(&[v1.clone(), v2.clone()], &gains, false).unwrap();
    assert!((mixed_unclipped[0] - 1.6).abs() < 1e-7);

    let mixed_clipped = mix_voices(&[v1, v2], &gains, true).unwrap();
    assert!((mixed_clipped[0] - 1.0).abs() < 1e-7);
}

#[test]
fn test_mix_negative_clipping() {
    let v1 = vec![-0.9];
    let v2 = vec![-0.9];
    let gains = vec![1.0, 1.0];
    let mixed = mix_voices(&[v1, v2], &gains, true).unwrap();
    assert!((mixed[0] - (-1.0)).abs() < 1e-7);
}

#[test]
fn test_mix_empty_voices() {
    let mixed = mix_voices(&[], &[], false).unwrap();
    assert!(mixed.is_empty());
}

#[test]
fn test_mix_all_empty_audio() {
    let mixed = mix_voices(&[vec![], vec![]], &[0.5, 0.5], false).unwrap();
    assert!(mixed.is_empty());
}

#[test]
fn test_mix_length_mismatch_rejected() {
    let result = mix_voices(&[vec![1.0]], &[0.5, 0.5], false);
    assert!(result.is_err());
}

#[test]
fn test_mix_gain_clamped_above_one() {
    // Gain > 1.0 is clamped to 1.0 inside mix_voices.
    let v1 = vec![0.5];
    let mixed = mix_voices(&[v1], &[2.0], false).unwrap();
    assert!((mixed[0] - 0.5).abs() < 1e-7); // 0.5 * clamp(2.0, 0, 1) = 0.5
}

// ---------------------------------------------------------------------------
// mix_voices_with_config
// ---------------------------------------------------------------------------

#[test]
fn test_mix_with_config_8_voices() {
    let config = ChorusConfig::equal_gain(8).unwrap();
    let voices: Vec<Vec<f32>> = (0..8).map(|_| vec![1.0; 100]).collect();
    let mixed = mix_voices_with_config(&voices, &config).unwrap();
    assert_eq!(mixed.len(), 100);
    // 8 voices at 0.125 gain each, all at 1.0 = sum = 1.0.
    for &s in &mixed {
        assert!((s - 1.0).abs() < 1e-5, "expected ~1.0, got {s}");
    }
}

#[test]
fn test_mix_with_config_voice_count_mismatch() {
    let config = ChorusConfig::equal_gain(4).unwrap();
    let voices: Vec<Vec<f32>> = (0..3).map(|_| vec![1.0]).collect();
    assert!(mix_voices_with_config(&voices, &config).is_err());
}

// ---------------------------------------------------------------------------
// Property: equal gain mixing preserves energy for uncorrelated signals
// ---------------------------------------------------------------------------

#[test]
fn test_equal_gain_no_overflow_uncorrelated() {
    // With equal gain = 1/N, N uncorrelated voices (alternating ±1.0)
    // should produce output in [-1.0, 1.0] without clipping.
    let n = 8;
    let config = ChorusConfig::equal_gain(n).unwrap();
    let len = 1000;
    let voices: Vec<Vec<f32>> = (0..n)
        .map(|v| {
            (0..len)
                .map(|i| if (i + v) % 2 == 0 { 1.0 } else { -1.0 })
                .collect()
        })
        .collect();
    let mixed = mix_voices_with_config(&voices, &config).unwrap();
    let max_abs = mixed.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs <= 1.0 + 1e-6,
        "equal-gain mixing overflow: max_abs = {max_abs}",
    );
}

// ---------------------------------------------------------------------------
// mix_voices_stereo
// ---------------------------------------------------------------------------

#[test]
fn test_stereo_center_pan_produces_equal_channels() {
    // Center pan (0.0) should produce identical L and R channels.
    // Equal-power: angle = π/4, cos = sin ≈ 0.707.
    let audio = vec![vec![1.0, -0.5, 0.3]];
    let params = vec![VoiceMix {
        gain: 1.0,
        pan: 0.0,
    }];
    let stereo = mix_voices_stereo(&audio, &params, false).unwrap();
    assert_eq!(stereo.len(), 6); // 3 samples * 2 channels
    for i in 0..3 {
        let left = stereo[i * 2];
        let right = stereo[i * 2 + 1];
        assert!(
            (left - right).abs() < 1e-6,
            "sample {i}: L={left} != R={right} at center pan",
        );
    }
}

#[test]
fn test_stereo_hard_left_pan() {
    // Pan = -1.0: angle = 0, cos(0) = 1.0, sin(0) = 0.0.
    let audio = vec![vec![0.8]];
    let params = vec![VoiceMix {
        gain: 1.0,
        pan: -1.0,
    }];
    let stereo = mix_voices_stereo(&audio, &params, false).unwrap();
    assert!((stereo[0] - 0.8).abs() < 1e-6, "left should be 0.8");
    assert!(stereo[1].abs() < 1e-6, "right should be ~0.0");
}

#[test]
fn test_stereo_hard_right_pan() {
    // Pan = 1.0: angle = π/2, cos(π/2) ≈ 0, sin(π/2) = 1.0.
    let audio = vec![vec![0.8]];
    let params = vec![VoiceMix {
        gain: 1.0,
        pan: 1.0,
    }];
    let stereo = mix_voices_stereo(&audio, &params, false).unwrap();
    assert!(stereo[0].abs() < 1e-6, "left should be ~0.0");
    assert!((stereo[1] - 0.8).abs() < 1e-6, "right should be 0.8");
}

#[test]
fn test_stereo_two_voices_opposite_pans() {
    // Voice 1 panned left, voice 2 panned right.
    let v1 = vec![1.0];
    let v2 = vec![1.0];
    let params = vec![
        VoiceMix {
            gain: 0.5,
            pan: -1.0,
        },
        VoiceMix {
            gain: 0.5,
            pan: 1.0,
        },
    ];
    let stereo = mix_voices_stereo(&[v1, v2], &params, false).unwrap();
    // Left channel: v1 at full left (0.5 * cos(0) * 1.0) + v2 at full right (0.5 * cos(π/2) * 1.0)
    assert!((stereo[0] - 0.5).abs() < 1e-6, "left should be ~0.5");
    assert!((stereo[1] - 0.5).abs() < 1e-6, "right should be ~0.5");
}

#[test]
fn test_stereo_clipping() {
    let v1 = vec![0.9];
    let v2 = vec![0.9];
    let params = vec![
        VoiceMix {
            gain: 1.0,
            pan: -1.0,
        },
        VoiceMix {
            gain: 1.0,
            pan: -1.0,
        },
    ];
    let stereo = mix_voices_stereo(&[v1, v2], &params, true).unwrap();
    assert!((stereo[0] - 1.0).abs() < 1e-6, "left should clip to 1.0");
}

#[test]
fn test_stereo_empty_voices() {
    let stereo = mix_voices_stereo(&[], &[], false).unwrap();
    assert!(stereo.is_empty());
}

#[test]
fn test_stereo_length_mismatch_rejected() {
    let result = mix_voices_stereo(
        &[vec![1.0]],
        &[
            VoiceMix {
                gain: 0.5,
                pan: 0.0,
            },
            VoiceMix {
                gain: 0.5,
                pan: 0.0,
            },
        ],
        false,
    );
    assert!(result.is_err());
}

#[test]
fn test_stereo_different_lengths_zero_pads() {
    let v1 = vec![1.0, 1.0, 1.0];
    let v2 = vec![0.5];
    let params = vec![
        VoiceMix {
            gain: 1.0,
            pan: 0.0,
        },
        VoiceMix {
            gain: 1.0,
            pan: 0.0,
        },
    ];
    let stereo = mix_voices_stereo(&[v1, v2], &params, false).unwrap();
    // 3 samples * 2 channels = 6
    assert_eq!(stereo.len(), 6);
    // Sample 2 (index 2): only v1 contributes (v2 zero-padded).
    let cos_pi_4 = (std::f32::consts::FRAC_PI_4).cos();
    let expected = 1.0 * cos_pi_4; // only v1
    assert!(
        (stereo[4] - expected).abs() < 1e-5,
        "sample 2 left: expected {expected}, got {}",
        stereo[4],
    );
}

// ---------------------------------------------------------------------------
// ChorusConfig::with_stereo_pan
// ---------------------------------------------------------------------------

#[test]
fn test_with_stereo_pan_valid() {
    let config = ChorusConfig::with_stereo_pan(vec![0.5, 0.5], vec![-0.5, 0.5]).unwrap();
    assert_eq!(config.n_voices, 2);
    assert!(config.pans.is_some());
    assert_eq!(config.pans.as_ref().unwrap(), &[-0.5, 0.5]);
}

#[test]
fn test_with_stereo_pan_length_mismatch() {
    let result = ChorusConfig::with_stereo_pan(vec![0.5, 0.5], vec![0.0]);
    assert!(result.is_err());
}

#[test]
fn test_with_stereo_pan_out_of_range() {
    let result = ChorusConfig::with_stereo_pan(vec![0.5], vec![1.5]);
    assert!(result.is_err());
}

#[test]
fn test_with_stereo_pan_nan_rejected() {
    let result = ChorusConfig::with_stereo_pan(vec![0.5], vec![f32::NAN]);
    assert!(result.is_err());
}

#[test]
fn test_mix_with_config_stereo_dispatches() {
    // When pans are set, mix_voices_with_config should produce stereo output.
    let config = ChorusConfig::with_stereo_pan(
        vec![1.0],
        vec![0.0], // center
    )
    .unwrap();
    let voices = vec![vec![1.0; 10]];
    let mixed = mix_voices_with_config(&voices, &config).unwrap();
    // Stereo: 10 samples * 2 channels = 20.
    assert_eq!(mixed.len(), 20);
    // Center pan: L == R.
    for i in 0..10 {
        assert!(
            (mixed[i * 2] - mixed[i * 2 + 1]).abs() < 1e-6,
            "sample {i}: L != R for center pan",
        );
    }
}

#[test]
fn test_validate_pans_mismatch() {
    let mut config = ChorusConfig::equal_gain(2).unwrap();
    config.pans = Some(vec![0.0, 0.0, 0.0]); // 3 pans for 2 voices
    assert!(config.validate().is_err());
}

// ---------------------------------------------------------------------------
// Pitch shift factor
// ---------------------------------------------------------------------------

#[test]
fn test_pitch_shift_factor_zero() {
    let f = pitch_shift_factor(0.0);
    assert!((f - 1.0).abs() < 1e-6, "0 semitones = no shift, got {f}");
}

#[test]
fn test_pitch_shift_factor_octave_up() {
    let f = pitch_shift_factor(12.0);
    assert!((f - 2.0).abs() < 1e-5, "+12 semitones = octave up, got {f}");
}

#[test]
fn test_pitch_shift_factor_octave_down() {
    let f = pitch_shift_factor(-12.0);
    assert!(
        (f - 0.5).abs() < 1e-5,
        "-12 semitones = octave down, got {f}"
    );
}

#[test]
fn test_pitch_shift_factor_5_cents() {
    // 5 cents = 0.05 semitones. Factor should be very close to 1.0.
    let f = pitch_shift_factor(0.05);
    assert!(f > 1.0 && f < 1.01, "5 cents should be near 1.0, got {f}");
}

// ---------------------------------------------------------------------------
// equal_power constructor
// ---------------------------------------------------------------------------

#[test]
fn test_equal_power_4_voices() {
    let config = ChorusConfig::equal_power(4).unwrap();
    assert_eq!(config.n_voices, 4);
    assert!(config.sqrt_gain_normalization);
    let expected_gain = 1.0 / (4.0f32).sqrt(); // 0.5
    for &g in &config.gains {
        assert!(
            (g - expected_gain).abs() < 1e-6,
            "expected {expected_gain}, got {g}"
        );
    }
}

#[test]
fn test_equal_power_1_voice() {
    let config = ChorusConfig::equal_power(1).unwrap();
    assert!((config.gains[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_equal_power_zero_rejected() {
    assert!(ChorusConfig::equal_power(0).is_err());
}

// ---------------------------------------------------------------------------
// rich_chorus preset
// ---------------------------------------------------------------------------

#[test]
fn test_rich_chorus_8_voices() {
    let config = ChorusConfig::rich_chorus(8).unwrap();
    assert_eq!(config.n_voices, 8);
    assert!(config.sqrt_gain_normalization);
    assert!(config.soft_limiter_drive.is_some());
    assert!(config.pitch_semitones.is_some());
    assert!(config.timing_offsets_sec.is_some());

    let pitches = config.pitch_semitones.as_ref().unwrap();
    assert_eq!(pitches.len(), 8);
    // First and last should be symmetric around 0.
    assert!(
        (pitches[0] + pitches[7]).abs() < 1e-6,
        "pitch spread should be symmetric"
    );
    // All within ±0.08 semitones.
    for &p in pitches {
        assert!(p.abs() <= 0.08 + 1e-6, "pitch {p} exceeds ±0.08 semitones");
    }

    let offsets = config.timing_offsets_sec.as_ref().unwrap();
    assert_eq!(offsets.len(), 8);
    // All within ±3ms.
    for &t in offsets {
        assert!(t.abs() <= 0.003 + 1e-6, "offset {t} exceeds ±3ms");
    }

    // Config should validate.
    config.validate().unwrap();
}

#[test]
fn test_rich_chorus_1_voice() {
    let config = ChorusConfig::rich_chorus(1).unwrap();
    assert_eq!(config.pitch_semitones.as_ref().unwrap(), &[0.0]);
    assert_eq!(config.timing_offsets_sec.as_ref().unwrap(), &[0.0]);
    config.validate().unwrap();
}

// ---------------------------------------------------------------------------
// Pitch semitones validation
// ---------------------------------------------------------------------------

#[test]
fn test_validate_pitch_semitones_length_mismatch() {
    let mut config = ChorusConfig::equal_gain(2).unwrap();
    config.pitch_semitones = Some(vec![0.0]); // 1 pitch for 2 voices
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_pitch_semitones_out_of_range() {
    let mut config = ChorusConfig::equal_gain(1).unwrap();
    config.pitch_semitones = Some(vec![13.0]); // > 12.0
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_pitch_semitones_nan_rejected() {
    let mut config = ChorusConfig::equal_gain(1).unwrap();
    config.pitch_semitones = Some(vec![f32::NAN]);
    assert!(config.validate().is_err());
}

// ---------------------------------------------------------------------------
// Timing offsets validation
// ---------------------------------------------------------------------------

#[test]
fn test_validate_timing_offsets_length_mismatch() {
    let mut config = ChorusConfig::equal_gain(2).unwrap();
    config.timing_offsets_sec = Some(vec![0.001]); // 1 offset for 2 voices
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_timing_offsets_out_of_range() {
    let mut config = ChorusConfig::equal_gain(1).unwrap();
    config.timing_offsets_sec = Some(vec![0.06]); // > 50ms
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_timing_offsets_nan_rejected() {
    let mut config = ChorusConfig::equal_gain(1).unwrap();
    config.timing_offsets_sec = Some(vec![f32::NAN]);
    assert!(config.validate().is_err());
}

// ---------------------------------------------------------------------------
// Stereo width validation
// ---------------------------------------------------------------------------

#[test]
fn test_validate_stereo_width_out_of_range() {
    let mut config = ChorusConfig::equal_gain(2).unwrap();
    config.stereo_width = 1.5;
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_stereo_width_negative() {
    let mut config = ChorusConfig::equal_gain(2).unwrap();
    config.stereo_width = -0.1;
    assert!(config.validate().is_err());
}

// ---------------------------------------------------------------------------
// Soft limiter validation
// ---------------------------------------------------------------------------

#[test]
fn test_validate_soft_limiter_zero_rejected() {
    let mut config = ChorusConfig::equal_gain(2).unwrap();
    config.soft_limiter_drive = Some(0.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_soft_limiter_negative_rejected() {
    let mut config = ChorusConfig::equal_gain(2).unwrap();
    config.soft_limiter_drive = Some(-1.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_soft_limiter_too_large_rejected() {
    let mut config = ChorusConfig::equal_gain(2).unwrap();
    config.soft_limiter_drive = Some(11.0);
    assert!(config.validate().is_err());
}

// ---------------------------------------------------------------------------
// Soft limiter mixing behavior
// ---------------------------------------------------------------------------

#[test]
fn test_soft_limiter_prevents_clipping() {
    // Two voices at full volume should produce values < 1.0 with soft limiter.
    let config = ChorusConfig::with_gains(vec![1.0, 1.0])
        .unwrap()
        .with_soft_limiter(1.5)
        .with_clip(false);
    let v1 = vec![0.8; 100];
    let v2 = vec![0.8; 100];
    let mixed = mix_voices_with_config(&[v1, v2], &config).unwrap();
    for &s in &mixed {
        assert!(
            s.abs() < 1.0,
            "soft limiter should keep output below 1.0, got {s}",
        );
        assert!(s > 0.0, "positive input should give positive output");
    }
}

#[test]
fn test_soft_limiter_preserves_small_signals() {
    // Small signals should pass through nearly unchanged.
    let config = ChorusConfig::with_gains(vec![1.0])
        .unwrap()
        .with_soft_limiter(1.0);
    let v1 = vec![0.1; 100];
    let mixed = mix_voices_with_config(&[v1], &config).unwrap();
    for &s in &mixed {
        // tanh(0.1) / 1.0 ≈ 0.0997 — very close to 0.1.
        assert!(
            (s - 0.1).abs() < 0.005,
            "small signal should be nearly unchanged, got {s}",
        );
    }
}

#[test]
fn test_soft_limiter_odd_symmetric() {
    // tanh is odd: f(-x) = -f(x).
    let config = ChorusConfig::with_gains(vec![1.0])
        .unwrap()
        .with_soft_limiter(2.0);
    let positive = vec![0.7];
    let negative = vec![-0.7];
    let mixed_pos = mix_voices_with_config(&[positive], &config).unwrap();
    let mixed_neg = mix_voices_with_config(&[negative], &config).unwrap();
    assert!(
        (mixed_pos[0] + mixed_neg[0]).abs() < 1e-6,
        "soft limiter should be odd-symmetric: {} vs {}",
        mixed_pos[0],
        mixed_neg[0],
    );
}

// ---------------------------------------------------------------------------
// Stereo width behavior
// ---------------------------------------------------------------------------

#[test]
fn test_stereo_width_zero_collapses_to_mono() {
    // With stereo_width = 0.0, all pans are scaled to 0.0 (center).
    let config = ChorusConfig::with_stereo_pan(vec![0.5, 0.5], vec![-1.0, 1.0])
        .unwrap()
        .with_stereo_width(0.0);
    let v1 = vec![1.0; 10];
    let v2 = vec![1.0; 10];
    let mixed = mix_voices_with_config(&[v1, v2], &config).unwrap();
    // With width=0 both voices are centered: L == R for every sample.
    for i in 0..10 {
        let left = mixed[i * 2];
        let right = mixed[i * 2 + 1];
        assert!(
            (left - right).abs() < 1e-6,
            "sample {i}: width=0 should give L={left} == R={right}",
        );
    }
}

#[test]
fn test_stereo_width_half_reduces_spread() {
    // With width=0.5, pan=-1.0 becomes effective pan=-0.5.
    let full_config = ChorusConfig::with_stereo_pan(vec![1.0], vec![-1.0])
        .unwrap()
        .with_stereo_width(1.0);
    let half_config = ChorusConfig::with_stereo_pan(vec![1.0], vec![-1.0])
        .unwrap()
        .with_stereo_width(0.5);
    let audio = vec![vec![0.8; 10]];
    let full_mixed = mix_voices_with_config(&audio, &full_config).unwrap();
    let half_mixed = mix_voices_with_config(&audio, &half_config).unwrap();
    // With full width and hard-left pan, right channel is ~0.
    // With half width, right channel should have more signal.
    assert!(
        half_mixed[1] > full_mixed[1] + 0.01,
        "half width should have more right signal: half={} vs full={}",
        half_mixed[1],
        full_mixed[1],
    );
}

// ---------------------------------------------------------------------------
// Timing offset behavior
// ---------------------------------------------------------------------------

#[test]
fn test_timing_offset_delays_signal() {
    // Positive offset should delay voice 1 relative to voice 0.
    let mut config = ChorusConfig::with_gains(vec![1.0, 1.0]).unwrap();
    // 10ms offset at 24kHz = 240 samples.
    config.timing_offsets_sec = Some(vec![0.0, 0.010]);
    config.clip_output = false;

    // Voice 0: impulse at sample 0.
    let mut v0 = vec![0.0f32; 1000];
    v0[0] = 1.0;
    // Voice 1: impulse at sample 0 (will be delayed by 240 samples).
    let mut v1 = vec![0.0f32; 1000];
    v1[0] = 1.0;

    let mixed = mix_voices_with_config(&[v0, v1], &config).unwrap();
    // Voice 0's impulse is at sample 0, voice 1's at sample 240.
    assert!(
        mixed[0].abs() > 0.5,
        "sample 0 should have voice 0's impulse, got {}",
        mixed[0],
    );
    assert!(
        mixed[240].abs() > 0.5,
        "sample 240 should have voice 1's delayed impulse, got {}",
        mixed[240],
    );
    // Between them should be ~0.
    assert!(
        mixed[120].abs() < 0.01,
        "sample 120 should be near zero, got {}",
        mixed[120],
    );
}

#[test]
fn test_timing_offset_negative_advances_signal() {
    let mut config = ChorusConfig::with_gains(vec![1.0]).unwrap();
    // -5ms offset = advance by 120 samples at 24kHz.
    config.timing_offsets_sec = Some(vec![-0.005]);
    config.clip_output = false;

    let mut v0 = vec![0.0f32; 1000];
    v0[200] = 1.0; // impulse at 200

    let mixed = mix_voices_with_config(&[v0], &config).unwrap();
    // Advanced by 120 samples: impulse should now be at 200 - 120 = 80.
    assert!(
        mixed[80].abs() > 0.5,
        "advanced impulse should be at ~80, got {} at 80",
        mixed[80],
    );
}

// ---------------------------------------------------------------------------
// Builder chaining
// ---------------------------------------------------------------------------

#[test]
fn test_builder_chaining_all_features() {
    let config = ChorusConfig::equal_gain(4)
        .unwrap()
        .with_pitch_semitones(vec![0.0, 0.05, -0.05, 0.1])
        .with_timing_offsets(vec![0.0, 0.002, -0.002, 0.003])
        .with_stereo_width(0.8)
        .with_soft_limiter(1.5)
        .with_sqrt_gain_normalization()
        .with_clip(false);

    assert!(config.pitch_semitones.is_some());
    assert!(config.timing_offsets_sec.is_some());
    assert!((config.stereo_width - 0.8).abs() < 1e-6);
    assert!(config.soft_limiter_drive.is_some());
    assert!(config.sqrt_gain_normalization);
    assert!(!config.clip_output);
    config.validate().unwrap();
}

// ---------------------------------------------------------------------------
// Property: equal_power mixing with N voices — max output < sqrt(N)/sqrt(N) = 1.0
// ---------------------------------------------------------------------------

#[test]
fn test_equal_power_no_overflow_correlated() {
    // With sqrt(N) gain, N fully correlated voices (all = 1.0) produce
    // sum = N * (1/sqrt(N)) = sqrt(N). That CAN exceed 1.0 for N > 1.
    // But with soft limiter, it stays bounded.
    let n = 8;
    let config = ChorusConfig::equal_power(n).unwrap().with_soft_limiter(1.0);
    let voices: Vec<Vec<f32>> = (0..n).map(|_| vec![1.0; 100]).collect();
    let mixed = mix_voices_with_config(&voices, &config).unwrap();
    for &s in &mixed {
        assert!(
            s.abs() < 1.0,
            "soft limiter should bound equal_power output, got {s}",
        );
    }
}

// ---------------------------------------------------------------------------
// mix_voices_from_refs with timing offsets
// ---------------------------------------------------------------------------

#[test]
fn test_mix_from_refs_timing_offsets() {
    let mut config = ChorusConfig::equal_gain(2).unwrap();
    config.timing_offsets_sec = Some(vec![0.0, 0.005]); // 5ms offset for voice 1

    let v0: Vec<f32> = (0..480).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    let v1: Vec<f32> = (0..480).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    let refs: Vec<&[f32]> = vec![v0.as_slice(), v1.as_slice()];

    let mixed = mix_voices_from_refs(&refs, &config).unwrap();
    // Voice 0 impulse at 0, voice 1 impulse delayed by 120 samples (5ms * 24kHz).
    let g = 0.5; // equal gain 1/2
    assert!(
        (mixed[0] - g).abs() < 0.01,
        "sample 0 should have voice 0 impulse * gain, got {}",
        mixed[0],
    );
    assert!(
        (mixed[120] - g).abs() < 0.01,
        "sample 120 should have voice 1 delayed impulse * gain, got {}",
        mixed[120],
    );
}

// ---------------------------------------------------------------------------
// Pitch shift via resampling
// ---------------------------------------------------------------------------

#[test]
fn test_pitch_shift_raises_frequency() {
    // A sine wave at 1kHz pitch-shifted up by 12 semitones (octave) should
    // read from input at 2x rate, so output[i] ≈ input[2i].
    let sample_rate = 24000.0f32;
    let freq = 1000.0;
    let len = 2400; // 100ms
    let sine: Vec<f32> = (0..len)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin())
        .collect();

    let mut config = ChorusConfig::with_gains(vec![1.0]).unwrap();
    config.pitch_semitones = Some(vec![12.0]);
    config.clip_output = false;
    let mixed = mix_voices_with_config(std::slice::from_ref(&sine), &config).unwrap();

    // Verify output[i] ≈ input[2*i] for the active portion (first half).
    let active_len = len / 2; // rate=2 means we exhaust input in half the samples
    for i in 0..active_len {
        let expected = sine[2 * i];
        assert!(
            (mixed[i] - expected).abs() < 1e-4,
            "sample {i}: expected input[{}]={expected}, got {}",
            2 * i,
            mixed[i],
        );
    }
    // Second half should be zero (past end of input).
    for i in (active_len + 10)..len {
        assert!(
            mixed[i].abs() < 1e-6,
            "sample {i} should be zero-padded, got {}",
            mixed[i],
        );
    }

    // Verify frequency doubling by counting zero crossings in active region only.
    let orig_crossings = count_zero_crossings(&sine[..active_len]);
    let shifted_crossings = count_zero_crossings(&mixed[..active_len]);
    let ratio = shifted_crossings as f32 / orig_crossings as f32;
    assert!(
        (ratio - 2.0).abs() < 0.15,
        "octave up should double zero-crossing rate in active region: ratio={ratio}",
    );
}

#[test]
fn test_pitch_shift_lowers_frequency() {
    let sample_rate = 24000.0f32;
    let freq = 2000.0;
    let len = 2400;
    let sine: Vec<f32> = (0..len)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin())
        .collect();

    let orig_crossings = count_zero_crossings(&sine);

    let mut config = ChorusConfig::with_gains(vec![1.0]).unwrap();
    config.pitch_semitones = Some(vec![-12.0]);
    config.clip_output = false;
    let mixed = mix_voices_with_config(&[sine], &config).unwrap();

    let shifted_crossings = count_zero_crossings(&mixed);
    let ratio = shifted_crossings as f32 / orig_crossings as f32;
    assert!(
        (ratio - 0.5).abs() < 0.15,
        "octave down should halve zero-crossing rate: ratio={ratio}, \
         orig={orig_crossings}, shifted={shifted_crossings}",
    );
}

#[test]
fn test_pitch_shift_zero_semitones_passthrough() {
    let audio = vec![0.1, 0.5, -0.3, 0.8, -0.9];
    let mut config = ChorusConfig::with_gains(vec![1.0]).unwrap();
    config.pitch_semitones = Some(vec![0.0]);
    config.clip_output = false;

    let mixed = mix_voices_with_config(std::slice::from_ref(&audio), &config).unwrap();
    for (i, (&orig, &mixed_s)) in audio.iter().zip(mixed.iter()).enumerate() {
        assert!(
            (orig - mixed_s).abs() < 1e-6,
            "sample {i}: 0 semitones should be passthrough, got {mixed_s} vs {orig}",
        );
    }
}

#[test]
fn test_pitch_shift_small_detuning_preserves_signal() {
    // ±5 cents (0.05 semitones) should barely change the signal.
    let audio: Vec<f32> = (0..1000).map(|i| (i as f32 / 100.0).sin()).collect();
    let mut config = ChorusConfig::with_gains(vec![1.0]).unwrap();
    config.pitch_semitones = Some(vec![0.05]); // +5 cents
    config.clip_output = false;

    let mixed = mix_voices_with_config(std::slice::from_ref(&audio), &config).unwrap();
    // RMS difference should be very small for tiny detuning.
    let rms_diff: f32 = audio
        .iter()
        .zip(mixed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / audio.len() as f32;
    let rms_diff = rms_diff.sqrt();
    assert!(
        rms_diff < 0.05,
        "5 cents detuning should barely change signal, RMS diff = {rms_diff}",
    );
}

#[test]
fn test_pitch_shift_with_timing_offset_combined() {
    // Both pitch shift and timing offset should compose correctly.
    let mut config = ChorusConfig::with_gains(vec![1.0, 1.0]).unwrap();
    config.pitch_semitones = Some(vec![0.0, 0.05]); // voice 1 slightly detuned
    config.timing_offsets_sec = Some(vec![0.0, 0.002]); // voice 1 delayed 2ms
    config.clip_output = false;

    let mut v0 = vec![0.0f32; 1000];
    v0[0] = 1.0; // impulse at sample 0
    let mut v1 = vec![0.0f32; 1000];
    v1[0] = 1.0; // impulse at sample 0

    let mixed = mix_voices_with_config(&[v0, v1], &config).unwrap();
    // Voice 0: impulse at sample 0, no shift.
    assert!(
        mixed[0].abs() > 0.4,
        "voice 0 impulse at 0, got {}",
        mixed[0]
    );
    // Voice 1: delayed by 48 samples (2ms * 24kHz).
    assert!(
        mixed[48].abs() > 0.4,
        "voice 1 delayed impulse at 48, got {}",
        mixed[48],
    );
}

// ---------------------------------------------------------------------------
// pitch_factors helper
// ---------------------------------------------------------------------------

#[test]
fn test_pitch_factors_none_when_no_pitch() {
    let config = ChorusConfig::equal_gain(4).unwrap();
    assert!(config.pitch_factors().is_none());
}

#[test]
fn test_pitch_factors_computes_correctly() {
    let config = ChorusConfig::equal_gain(3)
        .unwrap()
        .with_pitch_semitones(vec![0.0, 12.0, -12.0]);
    let factors = config.pitch_factors().unwrap();
    assert!((factors[0] - 1.0).abs() < 1e-6);
    assert!((factors[1] - 2.0).abs() < 1e-5);
    assert!((factors[2] - 0.5).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// with_normalized_gains
// ---------------------------------------------------------------------------

#[test]
fn test_with_normalized_gains_sqrt() {
    let config = ChorusConfig::with_gains(vec![0.1, 0.2, 0.3, 0.4])
        .unwrap()
        .with_sqrt_gain_normalization()
        .with_normalized_gains();
    let expected = 1.0 / (4.0f32).sqrt();
    for &g in &config.gains {
        assert!((g - expected).abs() < 1e-6, "expected {expected}, got {g}");
    }
}

#[test]
fn test_with_normalized_gains_linear() {
    let config = ChorusConfig::with_gains(vec![0.1, 0.9, 0.5])
        .unwrap()
        .with_normalized_gains();
    let expected = 1.0 / 3.0;
    for &g in &config.gains {
        assert!((g - expected).abs() < 1e-6, "expected {expected}, got {g}");
    }
}

// ---------------------------------------------------------------------------
// Pitch shift in stereo mode
// ---------------------------------------------------------------------------

#[test]
fn test_pitch_shift_stereo_produces_shifted_output() {
    let sample_rate = 24000.0f32;
    let freq = 1000.0;
    let len = 2400;
    let sine: Vec<f32> = (0..len)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin())
        .collect();

    let config = ChorusConfig::with_stereo_pan(vec![1.0], vec![0.0])
        .unwrap()
        .with_pitch_semitones(vec![12.0])
        .with_clip(false);

    let mixed = mix_voices_with_config(std::slice::from_ref(&sine), &config).unwrap();
    // Stereo: interleaved L/R. Extract left channel.
    let left: Vec<f32> = mixed.iter().step_by(2).copied().collect();

    // For octave up (rate=2), active region is first half.
    let active_len = len / 2;
    let orig_crossings = count_zero_crossings(&sine[..active_len]);
    let shifted_crossings = count_zero_crossings(&left[..active_len]);
    let ratio = shifted_crossings as f32 / orig_crossings as f32;
    assert!(
        (ratio - 2.0).abs() < 0.2,
        "stereo octave up in active region: ratio={ratio}",
    );
}

// ---------------------------------------------------------------------------
// Soft limiter with pitch shift and timing combined
// ---------------------------------------------------------------------------

#[test]
fn test_rich_chorus_full_pipeline() {
    // Verify the rich_chorus preset exercises all parameters together.
    let config = ChorusConfig::rich_chorus(4).unwrap();
    assert!(config.pitch_semitones.is_some());
    assert!(config.timing_offsets_sec.is_some());
    assert!(config.soft_limiter_drive.is_some());
    assert!(config.sqrt_gain_normalization);
    // rich_chorus now includes auto-spread stereo positions.
    assert!(config.pans.is_some());

    let voices: Vec<Vec<f32>> = (0..4)
        .map(|v| {
            (0..2400)
                .map(|i| ((i + v * 100) as f32 / 200.0).sin())
                .collect()
        })
        .collect();

    let mixed = mix_voices_with_config(&voices, &config).unwrap();
    // Stereo output: interleaved [L0, R0, L1, R1, ...] = 2 * 2400 = 4800.
    assert_eq!(mixed.len(), 4800);
    // All samples should be reasonably bounded. The soft limiter keeps the
    // dry mix under 1.0, but the reverb tail can add small amounts of energy
    // above 1.0. With default reverb_mix=0.15, the overshoot is minimal.
    for &s in &mixed {
        assert!(
            s.abs() < 1.5,
            "rich chorus output should be bounded, got {s}",
        );
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn count_zero_crossings(signal: &[f32]) -> usize {
    signal
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count()
}
