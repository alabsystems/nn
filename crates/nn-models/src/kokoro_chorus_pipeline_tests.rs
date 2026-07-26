// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Generate a sine wave at the given frequency and duration.
fn sine_voice(freq_hz: f32, duration_secs: f32) -> Vec<f32> {
    let n = (KOKORO_SAMPLE_RATE as f32 * duration_secs) as usize;
    (0..n)
        .map(|i| {
            (2.0 * std::f32::consts::PI * freq_hz * i as f32 / KOKORO_SAMPLE_RATE as f32).sin()
                * 0.5
        })
        .collect()
}

#[test]
fn test_output_length_matches_input() {
    let voices: Vec<Vec<f32>> = (0..4)
        .map(|i| sine_voice(220.0 + i as f32 * 10.0, 0.5))
        .collect();
    let config = ChorusMasterConfig::minimal(4).unwrap();
    let (left, right) = process_chorus(&voices, &config).unwrap();
    assert_eq!(left.len(), right.len(), "L and R must have equal length");
    assert_eq!(
        left.len(),
        voices[0].len(),
        "output length must match input"
    );
}

#[test]
fn test_stereo_channels_differ() {
    let voices: Vec<Vec<f32>> = (0..4)
        .map(|i| sine_voice(220.0 + i as f32 * 30.0, 0.3))
        .collect();
    let config = ChorusMasterConfig::standard(4).unwrap();
    let (left, right) = process_chorus(&voices, &config).unwrap();
    // With stereo panning, L and R should differ.
    let diff: f32 = left
        .iter()
        .zip(right.iter())
        .map(|(&l, &r)| (l - r).abs())
        .sum::<f32>()
        / left.len() as f32;
    assert!(
        diff > 1e-6,
        "stereo channels should differ, mean diff = {diff}"
    );
}

#[test]
fn test_mono_compatibility_single_voice() {
    let voice = sine_voice(440.0, 0.2);
    let config = ChorusMasterConfig::minimal(1).unwrap();
    let (left, right) = process_chorus(&[voice], &config).unwrap();
    // Single voice should produce identical L and R (centered).
    let max_diff = left
        .iter()
        .zip(right.iter())
        .map(|(&l, &r)| (l - r).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 0.01,
        "single voice should be nearly mono, max L/R diff = {max_diff}"
    );
}

#[test]
fn test_minimal_preset_smoke() {
    let voices: Vec<Vec<f32>> = (0..3).map(|_| sine_voice(300.0, 0.1)).collect();
    let config = ChorusMasterConfig::minimal(3).unwrap();
    let result = process_chorus(&voices, &config);
    assert!(result.is_ok(), "minimal preset should succeed");
}

#[test]
fn test_standard_preset_smoke() {
    let voices: Vec<Vec<f32>> = (0..4).map(|_| sine_voice(300.0, 0.1)).collect();
    let config = ChorusMasterConfig::standard(4).unwrap();
    let result = process_chorus(&voices, &config);
    assert!(result.is_ok(), "standard preset should succeed");
}

#[test]
fn test_full_preset_smoke() {
    let voices: Vec<Vec<f32>> = (0..4).map(|_| sine_voice(300.0, 0.1)).collect();
    let config = ChorusMasterConfig::full(4).unwrap();
    let result = process_chorus(&voices, &config);
    assert!(result.is_ok(), "full preset should succeed");
}

#[test]
fn test_validation_rejects_zero_voices() {
    let result = ChorusMasterConfig::new(0);
    assert!(result.is_err(), "0 voices should be rejected");
}

#[test]
fn test_validation_rejects_too_many_voices() {
    let result = ChorusMasterConfig::new(33);
    assert!(result.is_err(), "33 voices should be rejected");
}

#[test]
fn test_voice_count_mismatch_rejected() {
    let voices = vec![sine_voice(300.0, 0.1), sine_voice(400.0, 0.1)];
    let config = ChorusMasterConfig::minimal(3).unwrap();
    let result = process_chorus(&voices, &config);
    assert!(result.is_err(), "wrong voice count should be rejected");
}

#[test]
fn test_builder_chaining() {
    let config = ChorusMasterConfig::new(4)
        .unwrap()
        .with_eq(EqPreset::Natural.to_config())
        .with_deesser(DeEsserConfig::default())
        .with_detune(DetuneConfig::default())
        .with_humanize(HumanizeConfig::default())
        .with_blend(EnsembleBlendConfig::default())
        .with_stereo(StereoChorusConfig::auto_layout(4).unwrap())
        .with_dynamics(DynamicsPreset::Gentle.to_config())
        .with_reverb(ReverbConfig::default())
        .with_limiter(true);

    assert!(config.eq.is_some());
    assert!(config.deesser.is_some());
    assert!(config.detune.is_some());
    assert!(config.humanize.is_some());
    assert!(config.blend.is_some());
    assert!(config.stereo.is_some());
    assert!(config.dynamics.is_some());
    assert!(config.reverb.is_some());
    assert!(config.limiter_enabled);
}

#[test]
fn test_output_is_finite() {
    let voices: Vec<Vec<f32>> = (0..4)
        .map(|i| sine_voice(220.0 + i as f32 * 50.0, 0.2))
        .collect();
    let config = ChorusMasterConfig::full(4).unwrap();
    let (left, right) = process_chorus(&voices, &config).unwrap();
    for (i, &s) in left.iter().enumerate() {
        assert!(s.is_finite(), "left[{i}] is not finite: {s}");
    }
    for (i, &s) in right.iter().enumerate() {
        assert!(s.is_finite(), "right[{i}] is not finite: {s}");
    }
}

#[test]
fn test_limiter_bounds_output() {
    // Large amplitude input should be limited.
    let loud: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.05).sin() * 5.0).collect();
    let voices = vec![loud.clone(), loud.clone(), loud];
    let config = ChorusMasterConfig::new(3)
        .unwrap()
        .with_stereo(StereoChorusConfig::auto_layout(3).unwrap())
        .with_limiter(true);
    let (left, right) = process_chorus(&voices, &config).unwrap();
    let max_l = left.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    let max_r = right.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    // BusLimiter targets -0.1 dBFS ~ 0.989. Allow some headroom.
    assert!(max_l < 1.5, "limiter should constrain left, max = {max_l}");
    assert!(max_r < 1.5, "limiter should constrain right, max = {max_r}");
}

#[test]
fn test_pipeline_deterministic_across_instances() {
    // Two fresh pipelines with the same config should produce identical output.
    let voices: Vec<Vec<f32>> = (0..2).map(|_| sine_voice(440.0, 0.1)).collect();
    let config = ChorusMasterConfig::standard(2).unwrap();

    let mut pipeline1 = ChorusMasterPipeline::new(config.clone()).unwrap();
    let mut pipeline2 = ChorusMasterPipeline::new(config).unwrap();

    let (l1, r1) = pipeline1.process(&voices).unwrap();
    let (l2, r2) = pipeline2.process(&voices).unwrap();

    assert_eq!(l1.len(), l2.len());
    assert_eq!(r1.len(), r2.len());
    let max_diff_l = l1
        .iter()
        .zip(l2.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_diff_r = r1
        .iter()
        .zip(r2.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff_l < 1e-6,
        "two fresh pipelines should produce identical L, diff = {max_diff_l}"
    );
    assert!(
        max_diff_r < 1e-6,
        "two fresh pipelines should produce identical R, diff = {max_diff_r}"
    );
}

#[test]
fn test_silent_input_produces_near_silent_output() {
    let voices: Vec<Vec<f32>> = (0..3).map(|_| vec![0.0f32; 2400]).collect();
    let config = ChorusMasterConfig::full(3).unwrap();
    let (left, right) = process_chorus(&voices, &config).unwrap();
    let rms_l = (left.iter().map(|s| s * s).sum::<f32>() / left.len() as f32).sqrt();
    let rms_r = (right.iter().map(|s| s * s).sum::<f32>() / right.len() as f32).sqrt();
    assert!(
        rms_l < 0.01,
        "silent input should produce near-silent L, rms = {rms_l}"
    );
    assert!(
        rms_r < 0.01,
        "silent input should produce near-silent R, rms = {rms_r}"
    );
}

// ---------------------------------------------------------------------------
// Production preset tests
// ---------------------------------------------------------------------------

#[test]
fn test_singing_chorus_preset_smoke() {
    let voices: Vec<Vec<f32>> = (0..4)
        .map(|i| sine_voice(220.0 + i as f32 * 20.0, 0.3))
        .collect();
    let config = ChorusMasterConfig::singing_chorus(4).unwrap();
    let (left, right) = process_chorus(&voices, &config).unwrap();
    assert_eq!(left.len(), right.len());
    for (i, &s) in left.iter().enumerate() {
        assert!(s.is_finite(), "singing_chorus left[{i}] not finite: {s}");
    }
    for (i, &s) in right.iter().enumerate() {
        assert!(s.is_finite(), "singing_chorus right[{i}] not finite: {s}");
    }
}

#[test]
fn test_singing_chorus_has_vibrato() {
    let config = ChorusMasterConfig::singing_chorus(4).unwrap();
    assert!(
        config.vibrato.is_some(),
        "singing_chorus must enable vibrato"
    );
    let vibrato = config.vibrato.as_ref().unwrap();
    assert!(
        vibrato.depth_cents >= 25.0,
        "singing vibrato depth should be >= 25 cents"
    );
}

#[test]
fn test_singing_chorus_has_reverb() {
    let config = ChorusMasterConfig::singing_chorus(4).unwrap();
    assert!(config.reverb.is_some(), "singing_chorus must enable reverb");
}

#[test]
fn test_singing_chorus_wide_stereo() {
    let config = ChorusMasterConfig::singing_chorus(4).unwrap();
    let stereo = config
        .stereo
        .as_ref()
        .expect("singing_chorus must have stereo");
    assert!(
        stereo.stereo_width >= 0.7,
        "singing_chorus should have wide stereo, got {}",
        stereo.stereo_width,
    );
}

#[test]
fn test_singing_chorus_validation() {
    let config = ChorusMasterConfig::singing_chorus(8).unwrap();
    assert!(
        config.validate().is_ok(),
        "singing_chorus(8) should validate"
    );
}

#[test]
fn test_speaking_chorus_preset_smoke() {
    let voices: Vec<Vec<f32>> = (0..3).map(|_| sine_voice(180.0, 0.3)).collect();
    let config = ChorusMasterConfig::speaking_chorus(3).unwrap();
    let (left, right) = process_chorus(&voices, &config).unwrap();
    assert_eq!(left.len(), right.len());
    for (i, &s) in left.iter().enumerate() {
        assert!(s.is_finite(), "speaking_chorus left[{i}] not finite: {s}");
    }
}

#[test]
fn test_speaking_chorus_no_vibrato() {
    let config = ChorusMasterConfig::speaking_chorus(4).unwrap();
    assert!(
        config.vibrato.is_none(),
        "speaking_chorus must NOT have vibrato"
    );
}

#[test]
fn test_speaking_chorus_no_reverb() {
    let config = ChorusMasterConfig::speaking_chorus(4).unwrap();
    assert!(
        config.reverb.is_none(),
        "speaking_chorus must NOT have reverb"
    );
}

#[test]
fn test_speaking_chorus_narrow_stereo() {
    let config = ChorusMasterConfig::speaking_chorus(4).unwrap();
    let stereo = config
        .stereo
        .as_ref()
        .expect("speaking_chorus must have stereo");
    assert!(
        stereo.stereo_width <= 0.5,
        "speaking_chorus should have narrow stereo, got {}",
        stereo.stereo_width,
    );
    assert!(
        stereo.mono_compatible,
        "speaking_chorus should be mono-compatible",
    );
}

#[test]
fn test_speaking_chorus_strong_dynamics() {
    let config = ChorusMasterConfig::speaking_chorus(4).unwrap();
    assert!(
        config.dynamics.is_some(),
        "speaking_chorus must have dynamics"
    );
}

#[test]
fn test_speaking_chorus_validation() {
    let config = ChorusMasterConfig::speaking_chorus(6).unwrap();
    assert!(
        config.validate().is_ok(),
        "speaking_chorus(6) should validate"
    );
}

#[test]
fn test_intimate_preset_smoke() {
    let voices: Vec<Vec<f32>> = (0..2).map(|_| sine_voice(300.0, 0.3)).collect();
    let config = ChorusMasterConfig::intimate(2).unwrap();
    let (left, right) = process_chorus(&voices, &config).unwrap();
    assert_eq!(left.len(), right.len());
    for (i, &s) in left.iter().enumerate() {
        assert!(s.is_finite(), "intimate left[{i}] not finite: {s}");
    }
}

#[test]
fn test_intimate_no_reverb() {
    let config = ChorusMasterConfig::intimate(3).unwrap();
    assert!(config.reverb.is_none(), "intimate must NOT have reverb");
}

#[test]
fn test_intimate_no_vibrato() {
    let config = ChorusMasterConfig::intimate(3).unwrap();
    assert!(config.vibrato.is_none(), "intimate must NOT have vibrato");
}

#[test]
fn test_intimate_tight_stereo() {
    let config = ChorusMasterConfig::intimate(4).unwrap();
    let stereo = config.stereo.as_ref().expect("intimate must have stereo");
    assert!(
        stereo.stereo_width <= 0.4,
        "intimate should have tight stereo, got {}",
        stereo.stereo_width,
    );
}

#[test]
fn test_intimate_warm_eq() {
    let config = ChorusMasterConfig::intimate(3).unwrap();
    let eq = config.eq.as_ref().expect("intimate must have EQ");
    assert!(eq.low_gain_db > 0.0, "intimate EQ should have low boost");
    assert!(
        eq.high_gain_db < 0.0,
        "intimate EQ should have high roll-off"
    );
}

#[test]
fn test_intimate_validation() {
    let config = ChorusMasterConfig::intimate(4).unwrap();
    assert!(config.validate().is_ok(), "intimate(4) should validate");
}

#[test]
fn test_cathedral_preset_smoke() {
    let voices: Vec<Vec<f32>> = (0..6)
        .map(|i| sine_voice(200.0 + i as f32 * 15.0, 0.3))
        .collect();
    let config = ChorusMasterConfig::cathedral(6).unwrap();
    let (left, right) = process_chorus(&voices, &config).unwrap();
    assert_eq!(left.len(), right.len());
    for (i, &s) in left.iter().enumerate() {
        assert!(s.is_finite(), "cathedral left[{i}] not finite: {s}");
    }
}

#[test]
fn test_cathedral_has_reverb() {
    let config = ChorusMasterConfig::cathedral(4).unwrap();
    let reverb = config.reverb.as_ref().expect("cathedral must have reverb");
    assert!(
        reverb.reverb_mix >= 0.25,
        "cathedral reverb should be wet, got {}",
        reverb.reverb_mix
    );
    assert!(
        reverb.room_size >= 0.7,
        "cathedral room should be large, got {}",
        reverb.room_size
    );
}

#[test]
fn test_cathedral_has_vibrato() {
    let config = ChorusMasterConfig::cathedral(4).unwrap();
    let vibrato = config
        .vibrato
        .as_ref()
        .expect("cathedral must have vibrato");
    assert!(
        vibrato.depth_cents >= 40.0,
        "cathedral vibrato should be deep"
    );
}

#[test]
fn test_cathedral_full_stereo() {
    let config = ChorusMasterConfig::cathedral(4).unwrap();
    let stereo = config.stereo.as_ref().expect("cathedral must have stereo");
    assert!(
        stereo.stereo_width >= 0.9,
        "cathedral should have full stereo, got {}",
        stereo.stereo_width,
    );
}

#[test]
fn test_cathedral_validation() {
    let config = ChorusMasterConfig::cathedral(8).unwrap();
    assert!(config.validate().is_ok(), "cathedral(8) should validate");
}

#[test]
fn test_broadcast_preset_smoke() {
    let voices: Vec<Vec<f32>> = (0..3).map(|_| sine_voice(250.0, 0.3)).collect();
    let config = ChorusMasterConfig::broadcast(3).unwrap();
    let (left, right) = process_chorus(&voices, &config).unwrap();
    assert_eq!(left.len(), right.len());
    for (i, &s) in left.iter().enumerate() {
        assert!(s.is_finite(), "broadcast left[{i}] not finite: {s}");
    }
}

#[test]
fn test_broadcast_no_reverb() {
    let config = ChorusMasterConfig::broadcast(4).unwrap();
    assert!(config.reverb.is_none(), "broadcast must NOT have reverb");
}

#[test]
fn test_broadcast_no_vibrato() {
    let config = ChorusMasterConfig::broadcast(4).unwrap();
    assert!(config.vibrato.is_none(), "broadcast must NOT have vibrato");
}

#[test]
fn test_broadcast_mono_compatible() {
    let config = ChorusMasterConfig::broadcast(4).unwrap();
    let stereo = config.stereo.as_ref().expect("broadcast must have stereo");
    assert!(
        stereo.mono_compatible,
        "broadcast should be mono-compatible",
    );
}

#[test]
fn test_broadcast_aggressive_dynamics() {
    let config = ChorusMasterConfig::broadcast(4).unwrap();
    let dynamics = config
        .dynamics
        .as_ref()
        .expect("broadcast must have dynamics");
    // Aggressive dynamics has higher ratios than Gentle.
    assert!(
        dynamics.mid.ratio >= 4.0,
        "broadcast dynamics should be aggressive"
    );
}

#[test]
fn test_broadcast_has_deesser() {
    let config = ChorusMasterConfig::broadcast(4).unwrap();
    assert!(config.deesser.is_some(), "broadcast must have de-esser");
}

#[test]
fn test_broadcast_validation() {
    let config = ChorusMasterConfig::broadcast(6).unwrap();
    assert!(config.validate().is_ok(), "broadcast(6) should validate");
}

// ---------------------------------------------------------------------------
// Cross-preset property tests
// ---------------------------------------------------------------------------

#[test]
fn test_all_presets_valid_for_various_voice_counts() {
    for n in [1, 2, 3, 4, 6, 8, 16, 32] {
        // All presets must construct and validate for valid voice counts.
        let configs = [
            ("minimal", ChorusMasterConfig::minimal(n)),
            ("standard", ChorusMasterConfig::standard(n)),
            ("full", ChorusMasterConfig::full(n)),
            ("singing_chorus", ChorusMasterConfig::singing_chorus(n)),
            ("speaking_chorus", ChorusMasterConfig::speaking_chorus(n)),
            ("intimate", ChorusMasterConfig::intimate(n)),
            ("cathedral", ChorusMasterConfig::cathedral(n)),
            ("broadcast", ChorusMasterConfig::broadcast(n)),
        ];
        for (name, result) in &configs {
            let config = result.as_ref().unwrap_or_else(|e| {
                panic!("{name}({n}) failed to construct: {e}");
            });
            config.validate().unwrap_or_else(|e| {
                panic!("{name}({n}) failed validation: {e}");
            });
        }
    }
}

#[test]
fn test_all_presets_produce_finite_output() {
    let voice_counts = [2, 4, 6];
    let preset_fns: Vec<(
        &str,
        Box<dyn Fn(usize) -> Result<ChorusMasterConfig, KokoroError>>,
    )> = vec![
        (
            "singing_chorus",
            Box::new(ChorusMasterConfig::singing_chorus),
        ),
        (
            "speaking_chorus",
            Box::new(ChorusMasterConfig::speaking_chorus),
        ),
        ("intimate", Box::new(ChorusMasterConfig::intimate)),
        ("cathedral", Box::new(ChorusMasterConfig::cathedral)),
        ("broadcast", Box::new(ChorusMasterConfig::broadcast)),
    ];
    for &n in &voice_counts {
        let voices: Vec<Vec<f32>> = (0..n)
            .map(|i| sine_voice(200.0 + i as f32 * 25.0, 0.2))
            .collect();
        for (name, f) in &preset_fns {
            let config = f(n).unwrap_or_else(|e| panic!("{name}({n}) construct: {e}"));
            let (left, right) = process_chorus(&voices, &config)
                .unwrap_or_else(|e| panic!("{name}({n}) process: {e}"));
            for &s in &left {
                assert!(s.is_finite(), "{name}({n}) left has non-finite sample");
            }
            for &s in &right {
                assert!(s.is_finite(), "{name}({n}) right has non-finite sample");
            }
        }
    }
}

#[test]
fn test_singing_preset_stereo_wider_than_speaking() {
    let singing = ChorusMasterConfig::singing_chorus(4).unwrap();
    let speaking = ChorusMasterConfig::speaking_chorus(4).unwrap();
    let sing_width = singing.stereo.as_ref().unwrap().stereo_width;
    let speak_width = speaking.stereo.as_ref().unwrap().stereo_width;
    assert!(
        sing_width > speak_width,
        "singing ({sing_width}) should be wider than speaking ({speak_width})",
    );
}

#[test]
fn test_cathedral_reverb_wetter_than_singing() {
    let cathedral = ChorusMasterConfig::cathedral(4).unwrap();
    let singing = ChorusMasterConfig::singing_chorus(4).unwrap();
    let cath_mix = cathedral.reverb.as_ref().unwrap().reverb_mix;
    let sing_mix = singing.reverb.as_ref().unwrap().reverb_mix;
    assert!(
        cath_mix > sing_mix,
        "cathedral reverb ({cath_mix}) should be wetter than singing ({sing_mix})",
    );
}
