// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for mix automation and scene manager.

use super::*;

#[test]
fn test_crossfade_curve_linear_endpoints() {
    assert!((apply_curve(CrossfadeCurve::Linear, 0.0) - 0.0).abs() < 1e-6);
    assert!((apply_curve(CrossfadeCurve::Linear, 1.0) - 1.0).abs() < 1e-6);
    assert!((apply_curve(CrossfadeCurve::Linear, 0.5) - 0.5).abs() < 1e-6);
}

#[test]
fn test_crossfade_curve_scurve_endpoints() {
    assert!((apply_curve(CrossfadeCurve::SCurve, 0.0) - 0.0).abs() < 1e-6);
    assert!((apply_curve(CrossfadeCurve::SCurve, 1.0) - 1.0).abs() < 1e-6);
    // Midpoint should be exactly 0.5 for smoothstep.
    assert!((apply_curve(CrossfadeCurve::SCurve, 0.5) - 0.5).abs() < 1e-6);
}

#[test]
fn test_crossfade_curve_equal_power_endpoints() {
    assert!((apply_curve(CrossfadeCurve::EqualPower, 0.0) - 0.0).abs() < 1e-6);
    assert!((apply_curve(CrossfadeCurve::EqualPower, 1.0) - 1.0).abs() < 1e-6);
}

#[test]
fn test_crossfade_curve_clamp() {
    assert!((apply_curve(CrossfadeCurve::Linear, -1.0) - 0.0).abs() < 1e-6);
    assert!((apply_curve(CrossfadeCurve::Linear, 2.0) - 1.0).abs() < 1e-6);
}

#[test]
fn test_scene_snapshot_defaults() {
    let s = SceneSnapshot::new(4);
    assert_eq!(s.per_voice_gains.len(), 4);
    assert_eq!(s.per_voice_pans.len(), 4);
    s.validate().expect("default scene should be valid");
}

#[test]
fn test_scene_snapshot_validation_empty_voices() {
    let s = SceneSnapshot {
        per_voice_gains: vec![],
        per_voice_pans: vec![],
        master_gain: 1.0,
        stereo_width: 1.0,
        reverb_mix: 0.1,
        dynamics_threshold: -18.0,
        effect_enables: EffectEnables::NONE,
    };
    assert!(s.validate().is_err());
}

#[test]
fn test_scene_snapshot_validation_length_mismatch() {
    let s = SceneSnapshot {
        per_voice_gains: vec![1.0, 1.0],
        per_voice_pans: vec![0.0],
        master_gain: 1.0,
        stereo_width: 1.0,
        reverb_mix: 0.1,
        dynamics_threshold: -18.0,
        effect_enables: EffectEnables::NONE,
    };
    assert!(s.validate().is_err());
}

#[test]
fn test_default_pans_mono() {
    let p = default_pans(1);
    assert_eq!(p, vec![0.0]);
}

#[test]
fn test_default_pans_stereo() {
    let p = default_pans(2);
    assert!((p[0] - (-1.0)).abs() < 1e-6);
    assert!((p[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_default_pans_four_voices() {
    let p = default_pans(4);
    assert!((p[0] - (-1.0)).abs() < 1e-6);
    assert!((p[3] - 1.0).abs() < 1e-6);
    // Midpoints should be symmetric.
    assert!((p[1] + p[2]).abs() < 1e-6);
}

#[test]
fn test_mix_automator_no_transition() {
    let scene = SceneSnapshot::new(2);
    let auto = MixAutomator::new_kokoro(scene, CrossfadeCurve::Linear);
    let params = auto.get_current_params(0);
    assert!((params.master_gain - 1.0).abs() < 1e-6);
    assert!((params.transition_progress - 1.0).abs() < 1e-6);
    assert!(!auto.is_transitioning());
}

#[test]
fn test_mix_automator_transition() {
    let scene_a = SceneSnapshot::new(2);
    let mut scene_b = SceneSnapshot::new(2);
    scene_b.master_gain = 0.5;
    scene_b.reverb_mix = 0.5;

    let mut auto = MixAutomator::new_kokoro(scene_a, CrossfadeCurve::Linear);
    auto.set_scene(scene_b, 1000.0); // 1 second transition
    assert!(auto.is_transitioning());

    // At start: should be close to scene_a.
    let p0 = auto.get_current_params(0);
    assert!((p0.master_gain - 1.0).abs() < 0.01);

    // At midpoint: should be between A and B.
    let mid = KOKORO_SAMPLE_RATE / 2; // 0.5 seconds
    let p_mid = auto.get_current_params(mid);
    assert!((p_mid.master_gain - 0.75).abs() < 0.01);

    // After full transition: should be scene_b.
    let p_end = auto.get_current_params(KOKORO_SAMPLE_RATE);
    assert!((p_end.master_gain - 0.5).abs() < 0.01);
}

#[test]
fn test_mix_automator_advance() {
    let scene = SceneSnapshot::new(2);
    let mut auto = MixAutomator::new_kokoro(scene, CrossfadeCurve::SCurve);
    auto.advance(100);
    let mut target = SceneSnapshot::new(2);
    target.master_gain = 0.0;
    auto.set_scene(target, 100.0); // short transition
    assert!(auto.is_transitioning());
}

#[test]
fn test_mix_automator_reset() {
    let scene_a = SceneSnapshot::new(2);
    let mut scene_b = SceneSnapshot::new(2);
    scene_b.master_gain = 0.3;
    let mut auto = MixAutomator::new_kokoro(scene_a, CrossfadeCurve::Linear);
    auto.set_scene(scene_b.clone(), 1000.0);
    auto.reset(scene_b);
    assert!(!auto.is_transitioning());
}

#[test]
fn test_mix_automator_voice_count_transition() {
    // Transition from 2 voices to 4 voices.
    let scene_a = SceneSnapshot::new(2);
    let scene_b = SceneSnapshot::new(4);
    let mut auto = MixAutomator::new_kokoro(scene_a, CrossfadeCurve::Linear);
    auto.set_scene(scene_b, 500.0);
    let params = auto.get_current_params(0);
    // Should have 4 entries (max of 2, 4).
    assert_eq!(params.per_voice_gains.len(), 4);
    // Voices 2,3 fade from 0 to their target.
    assert!((params.per_voice_gains[2] - 0.0).abs() < 0.01);
}

#[test]
fn test_process_gains_applies_automation() {
    let mut scene_a = SceneSnapshot::new(2);
    scene_a.per_voice_gains = vec![0.5, 0.5];
    let auto = MixAutomator::new_kokoro(scene_a, CrossfadeCurve::Linear);

    let mut voices = vec![vec![1.0; 10], vec![1.0; 10]];
    auto.process_gains(&mut voices, 0, 10);
    // All samples should be scaled by 0.5.
    for v in &voices {
        for &s in v {
            assert!((s - 0.5).abs() < 1e-6);
        }
    }
}

#[test]
fn test_effect_enables_interpolate() {
    let a = EffectEnables(EffectEnables::EQ | EffectEnables::REVERB);
    let b = EffectEnables(EffectEnables::VIBRATO);
    assert_eq!(EffectEnables::interpolate(a, b, 0.3), a);
    assert_eq!(EffectEnables::interpolate(a, b, 0.7), b);
}

#[test]
fn test_automation_config_builder() {
    let cfg = AutomationConfig::new()
        .with_transition_ms(250.0)
        .with_curve(CrossfadeCurve::EqualPower)
        .with_scene("intro", SceneSnapshot::new(2));
    cfg.validate().expect("builder config should be valid");
    assert!(cfg.find_scene("intro").is_some());
    assert!(cfg.find_scene("missing").is_none());
}

#[test]
fn test_automation_timeline_ordering() {
    let mut tl = AutomationTimeline::new();
    tl.add_keyframe(1000, SceneSnapshot::new(2), 100.0);
    tl.add_keyframe(500, SceneSnapshot::new(2), 100.0);
    tl.add_keyframe(2000, SceneSnapshot::new(2), 100.0);
    assert_eq!(tl.keyframes[0].sample_position, 500);
    assert_eq!(tl.keyframes[1].sample_position, 1000);
    assert_eq!(tl.keyframes[2].sample_position, 2000);
    tl.validate().expect("timeline should be valid");
}

#[test]
fn test_build_to_chorus_preset() {
    let tl = build_to_chorus(4, 2000.0).expect("should build");
    assert_eq!(tl.keyframes.len(), 2);
    // First keyframe: only voice 0 active.
    assert!((tl.keyframes[0].scene.per_voice_gains[0] - 1.0).abs() < 1e-6);
    assert!((tl.keyframes[0].scene.per_voice_gains[1] - 0.0).abs() < 1e-6);
    // Last keyframe: all voices at 1.0.
    for &g in &tl.keyframes[1].scene.per_voice_gains {
        assert!((g - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_fade_to_intimate_preset() {
    let tl = fade_to_intimate(6, 1500.0).expect("should build");
    assert_eq!(tl.keyframes.len(), 2);
    // End: voices 2..5 faded out.
    let end = &tl.keyframes[1].scene;
    assert!((end.per_voice_gains[0] - 1.0).abs() < 1e-6);
    assert!((end.per_voice_gains[2] - 0.0).abs() < 1e-6);
    assert!((end.stereo_width - 0.4).abs() < 1e-6);
}

#[test]
fn test_dynamic_swell_preset() {
    let tl = dynamic_swell(4, 3000.0, 6000.0, 500.0).expect("should build");
    assert_eq!(tl.keyframes.len(), 3);
    // Peak should be louder.
    assert!(tl.keyframes[1].scene.master_gain > tl.keyframes[0].scene.master_gain);
    // End should match start.
    assert!((tl.keyframes[2].scene.master_gain - tl.keyframes[0].scene.master_gain).abs() < 1e-6);
}

#[test]
fn test_build_to_chorus_zero_voices_error() {
    assert!(build_to_chorus(0, 1000.0).is_err());
}

#[test]
fn test_fade_to_intimate_one_voice_error() {
    assert!(fade_to_intimate(1, 1000.0).is_err());
}

#[test]
fn test_dynamic_swell_bad_timing_error() {
    assert!(dynamic_swell(2, 0.0, 1000.0, 100.0).is_err());
    assert!(dynamic_swell(2, 2000.0, 1000.0, 100.0).is_err());
}

#[test]
fn test_ms_to_samples() {
    assert_eq!(ms_to_samples(1000.0, 24000), 24000);
    assert_eq!(ms_to_samples(500.0, 24000), 12000);
    assert_eq!(ms_to_samples(0.0, 24000), 0);
}

#[test]
fn test_scene_snapshot_nan_gain_rejected() {
    let mut s = SceneSnapshot::new(2);
    s.per_voice_gains[0] = f32::NAN;
    assert!(s.validate().is_err());
}

#[test]
fn test_scene_snapshot_inf_reverb_rejected() {
    let mut s = SceneSnapshot::new(2);
    s.reverb_mix = f32::INFINITY;
    assert!(s.validate().is_err());
}
