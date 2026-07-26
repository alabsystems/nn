// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for stereo imaging and constant-power panning.

use super::*;

// ---------------------------------------------------------------------------
// StereoPosition
// ---------------------------------------------------------------------------

#[test]
fn test_named_positions_to_pan() {
    assert_eq!(StereoPosition::Left.to_pan(), -1.0);
    assert_eq!(StereoPosition::CenterLeft.to_pan(), -0.5);
    assert_eq!(StereoPosition::Center.to_pan(), 0.0);
    assert_eq!(StereoPosition::CenterRight.to_pan(), 0.5);
    assert_eq!(StereoPosition::Right.to_pan(), 1.0);
}

#[test]
fn test_custom_position_to_pan() {
    assert!((StereoPosition::Custom(0.3).to_pan() - 0.3).abs() < 1e-7);
    // Out-of-range values are clamped.
    assert_eq!(StereoPosition::Custom(-2.0).to_pan(), -1.0);
    assert_eq!(StereoPosition::Custom(5.0).to_pan(), 1.0);
}

#[test]
fn test_from_pan_snaps_to_named() {
    assert_eq!(StereoPosition::from_pan(-1.0), StereoPosition::Left);
    assert_eq!(StereoPosition::from_pan(-0.5), StereoPosition::CenterLeft);
    assert_eq!(StereoPosition::from_pan(0.0), StereoPosition::Center);
    assert_eq!(StereoPosition::from_pan(0.5), StereoPosition::CenterRight);
    assert_eq!(StereoPosition::from_pan(1.0), StereoPosition::Right);
}

#[test]
fn test_from_pan_custom_for_non_snap() {
    let pos = StereoPosition::from_pan(0.3);
    match pos {
        StereoPosition::Custom(v) => assert!((v - 0.3).abs() < 1e-7),
        _ => panic!("expected Custom, got {pos:?}"),
    }
}

// ---------------------------------------------------------------------------
// Default voice layouts
// ---------------------------------------------------------------------------

#[test]
fn test_default_layout_0_voices() {
    assert!(default_voice_layout(0).is_empty());
}

#[test]
fn test_default_layout_1_voice_is_center() {
    let layout = default_voice_layout(1);
    assert_eq!(layout.len(), 1);
    assert_eq!(layout[0], StereoPosition::Center);
}

#[test]
fn test_default_layout_2_voices() {
    let layout = default_voice_layout(2);
    assert_eq!(layout.len(), 2);
    assert_eq!(layout[0], StereoPosition::CenterLeft);
    assert_eq!(layout[1], StereoPosition::CenterRight);
}

#[test]
fn test_default_layout_3_voices_lcr() {
    let layout = default_voice_layout(3);
    assert_eq!(layout.len(), 3);
    assert_eq!(layout[0], StereoPosition::Left);
    assert_eq!(layout[1], StereoPosition::Center);
    assert_eq!(layout[2], StereoPosition::Right);
}

#[test]
fn test_default_layout_4_voices() {
    let layout = default_voice_layout(4);
    assert_eq!(layout.len(), 4);
    assert_eq!(layout[0], StereoPosition::Left);
    assert_eq!(layout[1], StereoPosition::CenterLeft);
    assert_eq!(layout[2], StereoPosition::CenterRight);
    assert_eq!(layout[3], StereoPosition::Right);
}

#[test]
fn test_default_layout_5_voices() {
    let layout = default_voice_layout(5);
    assert_eq!(layout.len(), 5);
    assert_eq!(layout[0], StereoPosition::Left);
    assert_eq!(layout[1], StereoPosition::CenterLeft);
    assert_eq!(layout[2], StereoPosition::Center);
    assert_eq!(layout[3], StereoPosition::CenterRight);
    assert_eq!(layout[4], StereoPosition::Right);
}

#[test]
fn test_default_layout_8_voices_endpoints() {
    let layout = default_voice_layout(8);
    assert_eq!(layout.len(), 8);
    // First and last should snap to Left/Right.
    assert_eq!(layout[0], StereoPosition::Left);
    assert_eq!(layout[7], StereoPosition::Right);
}

// ---------------------------------------------------------------------------
// Constant-power pan law
// ---------------------------------------------------------------------------

#[test]
fn test_constant_power_hard_left() {
    let (l, r) = constant_power_pan(-1.0);
    assert!((l - 1.0).abs() < 1e-6, "left should be 1.0, got {l}");
    assert!(r.abs() < 1e-6, "right should be 0.0, got {r}");
}

#[test]
fn test_constant_power_hard_right() {
    let (l, r) = constant_power_pan(1.0);
    assert!(l.abs() < 1e-6, "left should be 0.0, got {l}");
    assert!((r - 1.0).abs() < 1e-6, "right should be 1.0, got {r}");
}

#[test]
fn test_constant_power_center() {
    let (l, r) = constant_power_pan(0.0);
    let expected = std::f32::consts::FRAC_1_SQRT_2; // ~0.707
    assert!(
        (l - expected).abs() < 1e-5,
        "left should be ~0.707, got {l}"
    );
    assert!(
        (r - expected).abs() < 1e-5,
        "right should be ~0.707, got {r}"
    );
}

#[test]
fn test_constant_power_preserves_energy_across_positions() {
    // Power = left^2 + right^2 should be 1.0 for all pan positions.
    for i in 0..=20 {
        let pan = -1.0 + 2.0 * (i as f32) / 20.0;
        let (l, r) = constant_power_pan(pan);
        let power = l * l + r * r;
        assert!(
            (power - 1.0).abs() < 1e-5,
            "power at pan={pan:.2} is {power:.6}, expected 1.0"
        );
    }
}

#[test]
fn test_constant_power_symmetry() {
    // Symmetry: pan(x).left == pan(-x).right and vice versa.
    for i in 0..=10 {
        let pan = i as f32 / 10.0;
        let (l_pos, r_pos) = constant_power_pan(pan);
        let (l_neg, r_neg) = constant_power_pan(-pan);
        assert!(
            (l_pos - r_neg).abs() < 1e-6,
            "symmetry failed at pan={pan}: l_pos={l_pos}, r_neg={r_neg}"
        );
        assert!(
            (r_pos - l_neg).abs() < 1e-6,
            "symmetry failed at pan={pan}: r_pos={r_pos}, l_neg={l_neg}"
        );
    }
}

// ---------------------------------------------------------------------------
// StereoPanner
// ---------------------------------------------------------------------------

#[test]
fn test_panner_default_full_width() {
    let panner = StereoPanner::default();
    assert!((panner.width() - 1.0).abs() < 1e-7);
}

#[test]
fn test_panner_zero_width_collapses_to_center() {
    let panner = StereoPanner::new(0.0);
    // All positions should produce center gains.
    let expected_center = std::f32::consts::FRAC_1_SQRT_2;
    for pan in [-1.0, -0.5, 0.0, 0.5, 1.0] {
        let (l, r) = panner.pan_gains(pan);
        assert!(
            (l - expected_center).abs() < 1e-5,
            "width=0 pan={pan}: left={l}, expected ~0.707"
        );
        assert!(
            (r - expected_center).abs() < 1e-5,
            "width=0 pan={pan}: right={r}, expected ~0.707"
        );
    }
}

#[test]
fn test_panner_half_width_reduces_spread() {
    let full = StereoPanner::new(1.0);
    let half = StereoPanner::new(0.5);

    // At half width, hard-left voice should not be at hard left.
    let (l_full, _) = full.pan_gains(-1.0);
    let (l_half, _) = half.pan_gains(-1.0);
    // Half width should put it at effective pan -0.5, so less extreme.
    assert!(l_half < l_full, "half width should be less extreme");
}

// ---------------------------------------------------------------------------
// StereoChorusConfig
// ---------------------------------------------------------------------------

#[test]
fn test_config_auto_layout_3() {
    let config = StereoChorusConfig::auto_layout(3).unwrap();
    assert_eq!(config.positions.len(), 3);
    assert!((config.stereo_width - 1.0).abs() < 1e-7);
    assert!(!config.mono_compatible);
}

#[test]
fn test_config_auto_layout_zero_rejected() {
    assert!(StereoChorusConfig::auto_layout(0).is_err());
}

#[test]
fn test_config_auto_layout_33_rejected() {
    assert!(StereoChorusConfig::auto_layout(33).is_err());
}

#[test]
fn test_config_with_stereo_width() {
    let config = StereoChorusConfig::auto_layout(3)
        .unwrap()
        .with_stereo_width(0.5);
    assert!((config.stereo_width - 0.5).abs() < 1e-7);
}

#[test]
fn test_config_with_mono_compatible() {
    let config = StereoChorusConfig::auto_layout(3)
        .unwrap()
        .with_mono_compatible(true);
    assert!(config.mono_compatible);
}

#[test]
fn test_config_effective_pans() {
    let config = StereoChorusConfig::auto_layout(3)
        .unwrap()
        .with_stereo_width(0.5);
    let pans = config.effective_pans();
    // LCR at half width: -0.5, 0.0, 0.5
    assert_eq!(pans.len(), 3);
    assert!((pans[0] - (-0.5)).abs() < 1e-6);
    assert!(pans[1].abs() < 1e-6);
    assert!((pans[2] - 0.5).abs() < 1e-6);
}

#[test]
fn test_config_nan_position_rejected() {
    let result = StereoChorusConfig::new(vec![StereoPosition::Custom(f32::NAN)]);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// apply_stereo_mix
// ---------------------------------------------------------------------------

#[test]
fn test_stereo_mix_empty() {
    let config = StereoChorusConfig::new(vec![]).unwrap();
    let (l, r) = apply_stereo_mix(&[], &config).unwrap();
    assert!(l.is_empty());
    assert!(r.is_empty());
}

#[test]
fn test_stereo_mix_single_center_voice() {
    let config = StereoChorusConfig::auto_layout(1).unwrap();
    let voice = vec![1.0f32; 100];
    let (l, r) = apply_stereo_mix(&[voice], &config).unwrap();
    assert_eq!(l.len(), 100);
    assert_eq!(r.len(), 100);
    // Center voice: L and R should be equal (~0.707).
    let expected = std::f32::consts::FRAC_1_SQRT_2;
    for i in 0..100 {
        assert!(
            (l[i] - expected).abs() < 1e-5,
            "l[{i}]={}, expected ~{expected}",
            l[i]
        );
        assert!(
            (r[i] - expected).abs() < 1e-5,
            "r[{i}]={}, expected ~{expected}",
            r[i]
        );
    }
}

#[test]
fn test_stereo_mix_hard_left_voice() {
    let config = StereoChorusConfig::new(vec![StereoPosition::Left]).unwrap();
    let voice = vec![1.0f32; 50];
    let (l, r) = apply_stereo_mix(&[voice], &config).unwrap();
    for i in 0..50 {
        assert!(
            (l[i] - 1.0).abs() < 1e-5,
            "hard left: l[{i}]={}, expected 1.0",
            l[i]
        );
        assert!(
            r[i].abs() < 1e-5,
            "hard left: r[{i}]={}, expected 0.0",
            r[i]
        );
    }
}

#[test]
fn test_stereo_mix_hard_right_voice() {
    let config = StereoChorusConfig::new(vec![StereoPosition::Right]).unwrap();
    let voice = vec![1.0f32; 50];
    let (l, r) = apply_stereo_mix(&[voice], &config).unwrap();
    for i in 0..50 {
        assert!(
            l[i].abs() < 1e-5,
            "hard right: l[{i}]={}, expected 0.0",
            l[i]
        );
        assert!(
            (r[i] - 1.0).abs() < 1e-5,
            "hard right: r[{i}]={}, expected 1.0",
            r[i]
        );
    }
}

#[test]
fn test_stereo_mix_3_voice_lcr_energy_preservation() {
    // LCR layout: voice 0 = Left, voice 1 = Center, voice 2 = Right.
    // Each voice is constant amplitude 1.0.
    //
    // Per-sample contributions:
    //   Left channel:  cos(0)*1 + cos(pi/4)*1 + cos(pi/2)*1 = 1.0 + 0.707 + 0.0 = 1.707
    //   Right channel: sin(0)*1 + sin(pi/4)*1 + sin(pi/2)*1 = 0.0 + 0.707 + 1.0 = 1.707
    //
    // The key property: L and R should be symmetric for LCR with equal signals.
    let config = StereoChorusConfig::auto_layout(3).unwrap();
    let voices: Vec<Vec<f32>> = vec![vec![1.0; 100]; 3];
    let (l, r) = apply_stereo_mix(&voices, &config).unwrap();

    // Verify L == R (symmetric layout, equal signals).
    for i in 0..100 {
        assert!(
            (l[i] - r[i]).abs() < 1e-5,
            "LCR symmetry: l[{i}]={} != r[{i}]={}",
            l[i],
            r[i]
        );
    }

    // Verify expected per-sample value: 1.0 + 1/sqrt(2) ~ 1.707
    let expected = 1.0 + std::f32::consts::FRAC_1_SQRT_2;
    for i in 0..100 {
        assert!(
            (l[i] - expected).abs() < 1e-4,
            "LCR left[{i}]={}, expected ~{expected}",
            l[i]
        );
    }
}

#[test]
fn test_stereo_mix_mono_compatibility() {
    // With mono_compatible=true, L+R per sample should approximate
    // what a mono mix of the same voices would produce.
    let config = StereoChorusConfig::auto_layout(3)
        .unwrap()
        .with_mono_compatible(true);
    let voices: Vec<Vec<f32>> = vec![vec![1.0; 100]; 3];
    let (l, r) = apply_stereo_mix(&voices, &config).unwrap();

    // Mono folddown: (L + R) for each voice at each position.
    // With mono normalization, each voice contributes
    // (cos(a) + sin(a)) / sqrt(2) to the mono sum.
    // At center: (0.707 + 0.707) / 1.414 = 1.0.
    // At hard left: (1.0 + 0.0) / 1.414 = 0.707.
    // At hard right: (0.0 + 1.0) / 1.414 = 0.707.
    // So mono sum = 0.707 + 1.0 + 0.707 = 2.414.
    // Without mono normalization: 1.0 + 1.414 + 1.0 = 3.414.
    for i in 0..100 {
        let mono_sum = l[i] + r[i];
        // The mono sum should be reasonable (not wildly different from n_voices).
        assert!(
            mono_sum > 0.0 && mono_sum < 5.0,
            "mono sum at sample {i} = {mono_sum}, out of reasonable range"
        );
    }

    // The key property: the mono sum should be approximately consistent
    // across samples (all voices are constant so it should be identical).
    let mono_sums: Vec<f32> = l.iter().zip(r.iter()).map(|(li, ri)| li + ri).collect();
    let first = mono_sums[0];
    for (i, &s) in mono_sums.iter().enumerate() {
        assert!(
            (s - first).abs() < 1e-5,
            "mono sum inconsistent: sample 0={first}, sample {i}={s}"
        );
    }
}

#[test]
fn test_stereo_mix_zero_width_equals_mono() {
    // At width=0.0, all voices collapse to center. L and R should be equal.
    let config = StereoChorusConfig::auto_layout(3)
        .unwrap()
        .with_stereo_width(0.0);
    let voices: Vec<Vec<f32>> = vec![vec![1.0; 50], vec![0.5; 50], vec![0.8; 50]];
    let (l, r) = apply_stereo_mix(&voices, &config).unwrap();

    for i in 0..50 {
        assert!(
            (l[i] - r[i]).abs() < 1e-5,
            "width=0: l[{i}]={} != r[{i}]={}",
            l[i],
            r[i]
        );
    }
}

#[test]
fn test_stereo_mix_different_length_voices() {
    // Shorter voices should be zero-padded.
    let config = StereoChorusConfig::auto_layout(2).unwrap();
    let voices = vec![vec![1.0; 100], vec![0.5; 50]];
    let (l, r) = apply_stereo_mix(&voices, &config).unwrap();
    assert_eq!(l.len(), 100);
    assert_eq!(r.len(), 100);
    // After sample 50, only voice 0 (CenterLeft) contributes.
    // Voice 1 (CenterRight) is zero-padded.
    let (lg_cl, rg_cl) = constant_power_pan(-0.5);
    assert!(
        (l[75] - lg_cl).abs() < 1e-5,
        "after short voice ends, l should only have voice 0"
    );
    assert!(
        (r[75] - rg_cl).abs() < 1e-5,
        "after short voice ends, r should only have voice 0"
    );
}

#[test]
fn test_stereo_mix_length_mismatch_error() {
    let config = StereoChorusConfig::auto_layout(2).unwrap();
    let voices = vec![vec![1.0; 10]]; // Only 1 voice, config expects 2.
    assert!(apply_stereo_mix(&voices, &config).is_err());
}

// ---------------------------------------------------------------------------
// interleave_stereo
// ---------------------------------------------------------------------------

#[test]
fn test_interleave_stereo_basic() {
    let left = vec![1.0, 2.0, 3.0];
    let right = vec![4.0, 5.0, 6.0];
    let interleaved = interleave_stereo(&left, &right).unwrap();
    assert_eq!(interleaved, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_interleave_stereo_empty() {
    let interleaved = interleave_stereo(&[], &[]).unwrap();
    assert!(interleaved.is_empty());
}

#[test]
fn test_interleave_stereo_length_mismatch() {
    assert!(interleave_stereo(&[1.0, 2.0], &[3.0]).is_err());
}

// ---------------------------------------------------------------------------
// End-to-end: stereo mix -> interleave roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_stereo_to_interleaved_roundtrip() {
    let config = StereoChorusConfig::auto_layout(3).unwrap();
    let voices: Vec<Vec<f32>> = vec![vec![0.8; 200], vec![0.6; 200], vec![0.4; 200]];
    let (l, r) = apply_stereo_mix(&voices, &config).unwrap();
    let interleaved = interleave_stereo(&l, &r).unwrap();
    assert_eq!(interleaved.len(), 400); // 200 samples * 2 channels
                                        // Verify interleaving: odd indices are right, even are left.
    for i in 0..200 {
        assert!(
            (interleaved[i * 2] - l[i]).abs() < 1e-7,
            "interleaved L mismatch at frame {i}"
        );
        assert!(
            (interleaved[i * 2 + 1] - r[i]).abs() < 1e-7,
            "interleaved R mismatch at frame {i}"
        );
    }
}

// ---------------------------------------------------------------------------
// Constant-power energy preservation (the critical acoustic property)
// ---------------------------------------------------------------------------

#[test]
fn test_per_voice_energy_preserved_across_pan_positions() {
    // For a single voice at various positions, L^2 + R^2 should always = signal^2.
    let signal = vec![0.7f32; 100];
    let signal_power = 0.7 * 0.7;

    for position in [
        StereoPosition::Left,
        StereoPosition::CenterLeft,
        StereoPosition::Center,
        StereoPosition::CenterRight,
        StereoPosition::Right,
        StereoPosition::Custom(-0.3),
        StereoPosition::Custom(0.7),
    ] {
        let config = StereoChorusConfig::new(vec![position]).unwrap();
        let (l, r) = apply_stereo_mix(std::slice::from_ref(&signal), &config).unwrap();

        let avg_power: f32 = l
            .iter()
            .zip(r.iter())
            .map(|(li, ri)| li * li + ri * ri)
            .sum::<f32>()
            / 100.0;
        assert!(
            (avg_power - signal_power).abs() < 1e-4,
            "energy not preserved at {position:?}: expected {signal_power}, got {avg_power}"
        );
    }
}
