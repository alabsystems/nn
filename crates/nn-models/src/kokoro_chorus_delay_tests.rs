// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `kokoro_chorus_delay` multi-tap delay and echo module.

use super::*;

#[test]
fn test_default_config_validates() {
    DelayConfig::new().expect("default config should validate");
}

#[test]
fn test_all_presets_validate() {
    for (name, cfg) in [
        ("slapback", DelayConfig::slapback()),
        ("ping_pong", DelayConfig::ping_pong()),
        ("rhythmic", DelayConfig::rhythmic()),
        ("ambient", DelayConfig::ambient()),
        ("haas_wide", DelayConfig::haas_wide()),
    ] {
        cfg.validate()
            .unwrap_or_else(|e| panic!("{name} preset failed validation: {e}"));
    }
}

#[test]
fn test_feedback_clamped() {
    let cfg = DelayConfig::default().with_feedback(0.96);
    assert!(cfg.validate().is_err(), "feedback > 0.95 should fail");
}

#[test]
fn test_empty_taps_rejected() {
    let cfg = DelayConfig::default().with_taps(vec![]);
    assert!(cfg.validate().is_err(), "empty taps should fail");
}

#[test]
fn test_slapback_no_repeats() {
    let cfg = DelayConfig::slapback();
    let mut proc = MultiTapDelay::new(&cfg).unwrap();
    // Feed an impulse
    let mut impulse = vec![0.0_f32; 4800]; // 200ms at 24kHz
    impulse[0] = 1.0;
    let mut right = impulse.clone();
    proc.process_stereo(&mut impulse, &mut right).unwrap();

    // With 0 feedback and 80ms delay, there should be exactly one echo
    // at sample ~1920 (80ms * 24kHz) and no further repeats
    let echo_idx = (80.0 / 1000.0 * 24000.0) as usize;
    // Check that the echo region has energy
    let echo_energy: f32 = impulse
        [echo_idx.saturating_sub(2)..=(echo_idx + 2).min(impulse.len() - 1)]
        .iter()
        .map(|x| x * x)
        .sum();
    assert!(
        echo_energy > 0.001,
        "should have echo energy near {echo_idx}"
    );

    // Check tail is silent (no feedback repeats)
    let tail_energy: f32 = impulse[echo_idx + 100..].iter().map(|x| x * x).sum();
    assert!(
        tail_energy < 1e-6,
        "slapback tail should be silent, got energy {tail_energy}"
    );
}

#[test]
fn test_ping_pong_stereo_separation() {
    let cfg = DelayConfig::ping_pong();
    let mut proc = MultiTapDelay::new(&cfg).unwrap();
    let mut impulse = vec![0.0_f32; 24000]; // 1 second
    impulse[0] = 1.0;
    let mut right = impulse.clone();
    proc.process_stereo(&mut impulse, &mut right).unwrap();

    // First tap at 250ms panned left -- left should have more energy near
    // sample 6000 than right
    let region = 5900..6100;
    let left_e: f32 = impulse[region.clone()].iter().map(|x| x * x).sum();
    let right_e: f32 = right[region].iter().map(|x| x * x).sum();
    assert!(left_e > right_e, "first tap should be panned left");
}

#[test]
fn test_tempo_sync_resolves_beats() {
    let cfg = DelayConfig::rhythmic().with_tempo_bpm(120.0);
    let tap = &cfg.taps[2]; // 1.0 beat at 120 BPM = 500ms
    let ms = tap.resolved_delay_ms(Some(120.0));
    assert!(
        (ms - 500.0).abs() < 0.01,
        "1 beat at 120 BPM = 500ms, got {ms}"
    );
}

#[test]
fn test_set_tempo() {
    let cfg = DelayConfig::rhythmic();
    let mut proc = MultiTapDelay::new(&cfg).unwrap();
    proc.set_tempo(90.0);
    assert_eq!(proc.config().tempo_bpm, Some(90.0));
}

#[test]
fn test_reset_clears_state() {
    let cfg = DelayConfig::slapback();
    let mut proc = MultiTapDelay::new(&cfg).unwrap();
    // Process some audio
    let mut left = vec![1.0_f32; 480];
    let mut right = left.clone();
    proc.process_stereo(&mut left, &mut right).unwrap();

    proc.reset();

    // After reset, processing silence should produce silence
    let mut silence_l = vec![0.0_f32; 480];
    let mut silence_r = vec![0.0_f32; 480];
    proc.process_stereo(&mut silence_l, &mut silence_r).unwrap();
    let energy: f32 = silence_l
        .iter()
        .chain(silence_r.iter())
        .map(|x| x * x)
        .sum();
    assert!(
        energy < 1e-10,
        "after reset, silence in should give silence out"
    );
}

#[test]
fn test_mono_processing() {
    let cfg = DelayConfig::haas_wide();
    let mut proc = MultiTapDelay::new(&cfg).unwrap();
    let audio = vec![0.5_f32; 960];
    let (left, right) = proc.process_mono(&audio);
    assert_eq!(left.len(), audio.len());
    assert_eq!(right.len(), audio.len());
    // All outputs should be finite
    assert!(left.iter().all(|x| x.is_finite()));
    assert!(right.iter().all(|x| x.is_finite()));
}

#[test]
fn test_nan_input_safety() {
    let cfg = DelayConfig::slapback();
    let mut proc = MultiTapDelay::new(&cfg).unwrap();
    let mut left = vec![f32::NAN; 240];
    let mut right = vec![f32::INFINITY; 240];
    proc.process_stereo(&mut left, &mut right).unwrap();
    // All outputs should be finite (NaN/Inf clamped to 0)
    assert!(
        left.iter().all(|x| x.is_finite()),
        "left should be all finite"
    );
    assert!(
        right.iter().all(|x| x.is_finite()),
        "right should be all finite"
    );
}

#[test]
fn test_ambient_preset_long_tail() {
    let cfg = DelayConfig::ambient();
    let mut proc = MultiTapDelay::new(&cfg).unwrap();
    let mut impulse = vec![0.0_f32; 48000]; // 2 seconds
    impulse[0] = 1.0;
    let mut right = impulse.clone();
    proc.process_stereo(&mut impulse, &mut right).unwrap();

    // Ambient has high feedback (0.55) and long taps -- should have energy
    // well past 1 second
    let late_energy: f32 = impulse[24000..36000].iter().map(|x| x * x).sum();
    assert!(late_energy > 1e-8, "ambient should have late tail energy");
}
