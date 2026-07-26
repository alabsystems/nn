// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `kokoro_chorus_transient_align`.

use super::*;

const SR: f32 = 24000.0;

// -- Config tests -------------------------------------------------------

#[test]
fn test_config_default_valid() {
    TransientAlignConfig::new()
        .validate()
        .expect("default should be valid");
}

#[test]
fn test_config_tight_valid() {
    TransientAlignConfig::tight()
        .validate()
        .expect("tight preset should be valid");
}

#[test]
fn test_config_natural_valid() {
    TransientAlignConfig::natural()
        .validate()
        .expect("natural preset should be valid");
}

#[test]
fn test_config_loose_valid() {
    TransientAlignConfig::loose()
        .validate()
        .expect("loose preset should be valid");
}

#[test]
fn test_config_percussion_valid() {
    TransientAlignConfig::percussion()
        .validate()
        .expect("percussion preset should be valid");
}

#[test]
fn test_config_builder_chain() {
    let cfg = TransientAlignConfig::new()
        .with_detection_threshold_db(-24.0)
        .with_alignment_strength(0.8)
        .with_lookahead_ms(3.0)
        .with_attack_window_ms(8.0)
        .with_max_shift_ms(4.0)
        .with_mix(0.7);
    cfg.validate().expect("builder chain should be valid");
    assert_eq!(cfg.detection_threshold_db, -24.0);
    assert_eq!(cfg.alignment_strength, 0.8);
    assert_eq!(cfg.lookahead_ms, 3.0);
    assert_eq!(cfg.attack_window_ms, 8.0);
    assert_eq!(cfg.max_shift_ms, 4.0);
    assert!((cfg.mix - 0.7).abs() < 1e-6);
}

#[test]
fn test_config_invalid_threshold_too_high() {
    assert!(TransientAlignConfig::new()
        .with_detection_threshold_db(-5.0)
        .validate()
        .is_err());
}

#[test]
fn test_config_invalid_threshold_too_low() {
    assert!(TransientAlignConfig::new()
        .with_detection_threshold_db(-61.0)
        .validate()
        .is_err());
}

#[test]
fn test_config_invalid_threshold_nan() {
    assert!(TransientAlignConfig::new()
        .with_detection_threshold_db(f32::NAN)
        .validate()
        .is_err());
}

#[test]
fn test_config_invalid_strength_out_of_range() {
    assert!(TransientAlignConfig::new()
        .with_alignment_strength(1.1)
        .validate()
        .is_err());
    assert!(TransientAlignConfig::new()
        .with_alignment_strength(-0.1)
        .validate()
        .is_err());
}

#[test]
fn test_config_invalid_mix_nan() {
    assert!(TransientAlignConfig::new()
        .with_mix(f32::NAN)
        .validate()
        .is_err());
}

#[test]
fn test_config_invalid_mix_infinity() {
    assert!(TransientAlignConfig::new()
        .with_mix(f32::INFINITY)
        .validate()
        .is_err());
}

#[test]
fn test_config_boundary_values() {
    TransientAlignConfig::new()
        .with_detection_threshold_db(-60.0)
        .with_alignment_strength(0.0)
        .with_lookahead_ms(0.5)
        .with_attack_window_ms(1.0)
        .with_max_shift_ms(0.5)
        .with_mix(0.0)
        .validate()
        .expect("boundary min values should be valid");

    TransientAlignConfig::new()
        .with_detection_threshold_db(-6.0)
        .with_alignment_strength(1.0)
        .with_lookahead_ms(20.0)
        .with_attack_window_ms(50.0)
        .with_max_shift_ms(20.0)
        .with_mix(1.0)
        .validate()
        .expect("boundary max values should be valid");
}

// -- Aligner construction -----------------------------------------------

#[test]
fn test_aligner_new_valid() {
    let cfg = TransientAlignConfig::new();
    let aligner = TransientAligner::new(&cfg, 4, SR).expect("valid");
    assert_eq!(aligner.n_voices(), 4);
    assert_eq!(aligner.sample_rate(), SR);
}

#[test]
fn test_aligner_new_zero_voices_rejected() {
    let cfg = TransientAlignConfig::new();
    assert!(TransientAligner::new(&cfg, 0, SR).is_err());
}

#[test]
fn test_aligner_new_invalid_sample_rate() {
    let cfg = TransientAlignConfig::new();
    assert!(TransientAligner::new(&cfg, 2, 0.0).is_err());
    assert!(TransientAligner::new(&cfg, 2, -1.0).is_err());
    assert!(TransientAligner::new(&cfg, 2, f32::NAN).is_err());
    assert!(TransientAligner::new(&cfg, 2, f32::INFINITY).is_err());
}

// -- Onset detection ----------------------------------------------------

#[test]
fn test_detect_onsets_silence_returns_empty() {
    let silence = vec![0.0f32; 4800];
    let onsets = detect_onsets(&silence, 48, 1e-6, 48);
    assert!(onsets.is_empty(), "silence should have no onsets");
}

#[test]
fn test_detect_onsets_sharp_click() {
    // Silence then a sharp click.
    let mut audio = vec![0.0f32; 2400];
    // Insert a click at sample 1200.
    for i in 1200..1210 {
        audio[i] = if i % 2 == 0 { 0.9 } else { -0.9 };
    }
    let onsets = detect_onsets(&audio, 48, 1e-6, 24);
    assert!(
        !onsets.is_empty(),
        "should detect onset from sharp click, got 0 onsets",
    );
    // The detected onset should be near sample 1200.
    let first = onsets[0].sample;
    assert!(
        (first as i64 - 1200).unsigned_abs() < 100,
        "onset should be near sample 1200, got {first}",
    );
}

#[test]
fn test_detect_onsets_empty_audio() {
    let onsets = detect_onsets(&[], 48, 1e-6, 24);
    assert!(onsets.is_empty());
}

// -- Process voices -----------------------------------------------------

#[test]
fn test_process_voices_single_voice_noop() {
    let cfg = TransientAlignConfig::tight();
    let mut aligner = TransientAligner::new(&cfg, 1, SR).expect("valid");
    let mut voices = vec![vec![0.5f32; 2400]];
    // Single voice: process_voices should be a no-op (< 2 voices).
    aligner
        .process_voices(&mut voices)
        .expect("single voice should succeed");
}

#[test]
fn test_process_voices_identical_voices_minimal_change() {
    let cfg = TransientAlignConfig::natural();
    let mut aligner = TransientAligner::new(&cfg, 3, SR).expect("valid");

    // All three voices are identical.
    let voice: Vec<f32> = (0..4800)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SR).sin())
        .collect();
    let mut voices = vec![voice.clone(), voice.clone(), voice.clone()];
    aligner.process_voices(&mut voices).expect("should succeed");

    // Identical voices should have very little change since onsets
    // are already aligned.
    for (vi, v) in voices.iter().enumerate() {
        let max_diff: f32 = v
            .iter()
            .zip(voice.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.3,
            "identical voices should stay similar, voice {vi} max_diff={max_diff}",
        );
    }
}

#[test]
fn test_process_voices_shifted_click_gets_aligned() {
    let cfg = TransientAlignConfig::tight();
    let mut aligner = TransientAligner::new(&cfg, 2, SR).expect("valid");

    let len = 4800;
    let mut voice0 = vec![0.0f32; len];
    let mut voice1 = vec![0.0f32; len];

    // Voice 0: click at sample 1000.
    for i in 1000..1010 {
        voice0[i] = if i % 2 == 0 { 0.8 } else { -0.8 };
    }
    // Voice 1: same click at sample 1050 (shifted by ~2ms at 24kHz).
    for i in 1050..1060 {
        voice1[i] = if i % 2 == 0 { 0.8 } else { -0.8 };
    }

    let original_v1 = voice1.clone();
    let mut voices = vec![voice0, voice1];
    aligner.process_voices(&mut voices).expect("should succeed");

    // Voice 1 should have changed (shifted toward voice 0's onset).
    let changed: bool = voices[1]
        .iter()
        .zip(original_v1.iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-10);
    assert!(
        changed,
        "voice 1 should be modified by alignment when click is offset",
    );
}

#[test]
fn test_process_voices_output_all_finite() {
    let cfg = TransientAlignConfig::percussion();
    let mut aligner = TransientAligner::new(&cfg, 2, SR).expect("valid");

    let mut voice0 = vec![0.0f32; 2400];
    let mut voice1 = vec![0.0f32; 2400];
    // Insert transients with some NaN sprinkled in.
    for i in 500..520 {
        voice0[i] = 0.7;
    }
    voice0[510] = f32::NAN;
    for i in 520..540 {
        voice1[i] = 0.7;
    }
    voice1[530] = f32::INFINITY;

    let mut voices = vec![voice0, voice1];
    aligner
        .process_voices(&mut voices)
        .expect("should handle NaN/Inf");
    for (vi, v) in voices.iter().enumerate() {
        for (si, &s) in v.iter().enumerate() {
            assert!(s.is_finite(), "voice {vi} sample {si} non-finite: {s}");
        }
    }
}

#[test]
fn test_process_voices_mix_zero_is_noop() {
    let cfg = TransientAlignConfig::new().with_mix(0.0);
    let mut aligner = TransientAligner::new(&cfg, 2, SR).expect("valid");

    let voice: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.01).sin()).collect();
    let original = vec![voice.clone(), voice];
    let mut voices = original.clone();
    aligner
        .process_voices(&mut voices)
        .expect("mix=0 should succeed");
    assert_eq!(voices, original, "mix=0 should be identity");
}

#[test]
fn test_process_voices_wrong_voice_count_rejected() {
    let cfg = TransientAlignConfig::new();
    let mut aligner = TransientAligner::new(&cfg, 3, SR).expect("valid");
    let mut voices = vec![vec![0.0; 100], vec![0.0; 100]];
    assert!(
        aligner.process_voices(&mut voices).is_err(),
        "wrong voice count should be rejected",
    );
}

#[test]
fn test_process_voices_empty_buffers_ok() {
    let cfg = TransientAlignConfig::new();
    let mut aligner = TransientAligner::new(&cfg, 2, SR).expect("valid");
    let mut voices = vec![vec![], vec![]];
    aligner
        .process_voices(&mut voices)
        .expect("empty buffers should succeed");
}

// -- Reset --------------------------------------------------------------

#[test]
fn test_reset_clears_lookahead_buffers() {
    let cfg = TransientAlignConfig::new();
    let mut aligner = TransientAligner::new(&cfg, 2, SR).expect("valid");
    // Push some data through the look-ahead buffers.
    for buf in &mut aligner.lookahead_bufs {
        buf.push(0.5);
        buf.push(0.3);
    }
    aligner.reset();
    for buf in &aligner.lookahead_bufs {
        assert!(
            buf.buf.iter().all(|&s| s == 0.0),
            "look-ahead buffer should be zero after reset",
        );
    }
}

// -- Onset clustering ---------------------------------------------------

#[test]
fn test_cluster_onsets_groups_nearby() {
    let voice0 = vec![Onset {
        sample: 100,
        strength: 1.0,
    }];
    let voice1 = vec![Onset {
        sample: 110,
        strength: 0.8,
    }];
    let clusters = cluster_onsets(&[voice0, voice1], 20);
    assert_eq!(clusters.len(), 1, "nearby onsets should cluster");
    assert_eq!(clusters[0].len(), 2);
}

#[test]
fn test_cluster_onsets_separates_distant() {
    let voice0 = vec![
        Onset {
            sample: 100,
            strength: 1.0,
        },
        Onset {
            sample: 1000,
            strength: 1.0,
        },
    ];
    let voice1 = vec![
        Onset {
            sample: 105,
            strength: 0.8,
        },
        Onset {
            sample: 1010,
            strength: 0.8,
        },
    ];
    let clusters = cluster_onsets(&[voice0, voice1], 20);
    assert_eq!(
        clusters.len(),
        2,
        "distant onset groups should form separate clusters",
    );
}

#[test]
fn test_cluster_onsets_single_voice_no_cluster() {
    // A single onset per cluster needs >= 2 entries to form a cluster.
    let voice0 = vec![Onset {
        sample: 100,
        strength: 1.0,
    }];
    let voice1: Vec<Onset> = vec![];
    let clusters = cluster_onsets(&[voice0, voice1], 20);
    assert!(
        clusters.is_empty(),
        "single-voice onset should not form a cluster",
    );
}

// -- Energy preservation ------------------------------------------------

#[test]
fn test_energy_preservation_after_alignment() {
    let cfg = TransientAlignConfig::natural();
    let mut aligner = TransientAligner::new(&cfg, 2, SR).expect("valid");

    let len = 4800;
    let mut voice0 = vec![0.0f32; len];
    let mut voice1 = vec![0.0f32; len];
    // Clicks at different positions.
    for i in 1000..1020 {
        voice0[i] = 0.6 * (i as f32 * 0.5).sin();
    }
    for i in 1040..1060 {
        voice1[i] = 0.6 * (i as f32 * 0.5).sin();
    }

    let orig_energy: f64 = voice0
        .iter()
        .chain(voice1.iter())
        .map(|&s| f64::from(s) * f64::from(s))
        .sum();

    let mut voices = vec![voice0, voice1];
    aligner.process_voices(&mut voices).expect("should succeed");

    let new_energy: f64 = voices
        .iter()
        .flat_map(|v| v.iter())
        .map(|&s| f64::from(s) * f64::from(s))
        .sum();

    // Energy should be within 50% of original (the process only moves
    // small attack windows, not the whole signal).
    let ratio = new_energy / orig_energy.max(1e-30);
    assert!(
        ratio > 0.5 && ratio < 2.0,
        "energy should be roughly preserved, ratio={ratio}",
    );
}

// -- ms_to_samples helper -----------------------------------------------

#[test]
fn test_ms_to_samples_basic() {
    assert_eq!(ms_to_samples(1.0, 24000.0), 24);
    assert_eq!(ms_to_samples(5.0, 24000.0), 120);
    assert_eq!(ms_to_samples(0.01, 24000.0), 1); // clamped to 1
}
