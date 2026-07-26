// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for spatial reverb and room modeling.

use super::*;

// -- ReverbConfig validation --

#[test]
fn test_reverb_config_default_valid() {
    let config = ReverbConfig::default();
    config.validate().unwrap();
    assert!((config.reverb_mix - 0.15).abs() < 1e-6);
    assert!((config.room_size - 0.3).abs() < 1e-6);
    assert!(config.early_reflections);
    assert!((config.damping - 0.5).abs() < 1e-6);
}

#[test]
fn test_reverb_config_builder_chaining() {
    let config = ReverbConfig::new()
        .with_reverb_mix(0.25)
        .with_room_size(0.6)
        .with_early_reflections(false)
        .with_damping(0.8);
    config.validate().unwrap();
    assert!((config.reverb_mix - 0.25).abs() < 1e-6);
    assert!((config.room_size - 0.6).abs() < 1e-6);
    assert!(!config.early_reflections);
    assert!((config.damping - 0.8).abs() < 1e-6);
}

#[test]
fn test_reverb_config_mix_out_of_range() {
    let config = ReverbConfig::new().with_reverb_mix(1.5);
    assert!(config.validate().is_err());
}

#[test]
fn test_reverb_config_mix_negative() {
    let config = ReverbConfig::new().with_reverb_mix(-0.1);
    assert!(config.validate().is_err());
}

#[test]
fn test_reverb_config_room_size_out_of_range() {
    let config = ReverbConfig::new().with_room_size(1.5);
    assert!(config.validate().is_err());
}

#[test]
fn test_reverb_config_damping_nan() {
    let config = ReverbConfig::new().with_damping(f32::NAN);
    assert!(config.validate().is_err());
}

// -- Comb filter properties --

#[test]
fn test_comb_filter_silence_in_silence_out() {
    let mut comb = CombFilter::new(100, 0.8, 0.5);
    for _ in 0..500 {
        let out = comb.process(0.0);
        assert!(
            out.abs() < 1e-6,
            "comb should output silence for silence input"
        );
    }
}

#[test]
fn test_comb_filter_impulse_response_delayed() {
    let delay = 100;
    let mut comb = CombFilter::new(delay, 0.5, 0.0);
    // Feed an impulse.
    let out0 = comb.process(1.0);
    assert!(out0.abs() < 1e-6, "no output before delay");
    // Feed silence for delay-1 samples.
    for i in 1..delay {
        let out = comb.process(0.0);
        assert!(out.abs() < 1e-6, "no output before delay at sample {i}");
    }
    // At exactly delay samples, should get the impulse.
    let out_delay = comb.process(0.0);
    assert!(
        (out_delay - 1.0).abs() < 1e-6,
        "impulse should appear at delay, got {out_delay}"
    );
}

#[test]
fn test_comb_filter_decays() {
    let delay = 50;
    let mut comb = CombFilter::new(delay, 0.7, 0.3);
    // Feed impulse then silence.
    comb.process(1.0);
    let mut all_output = Vec::with_capacity(2000);
    for _ in 0..2000 {
        all_output.push(comb.process(0.0));
    }
    // Find peak absolute values in each delay-length window.
    let mut window_peaks = Vec::new();
    for chunk in all_output.chunks(delay) {
        let peak = chunk.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if peak > 1e-6 {
            window_peaks.push(peak);
        }
    }
    // Should have multiple echoes that decay over time.
    assert!(
        window_peaks.len() >= 3,
        "should have at least 3 echo windows, got {}",
        window_peaks.len(),
    );
    // Overall decay: last window peak much smaller than first.
    let first = window_peaks[0];
    let last = *window_peaks.last().unwrap();
    assert!(
        last < first * 0.5,
        "comb should decay: first={first}, last={last}",
    );
}

// -- Allpass filter properties --

#[test]
fn test_allpass_bounded_output() {
    // The allpass filter should produce bounded output for bounded input
    // and should not diverge. With feedback < 1.0 it is stable.
    let mut allpass = AllpassFilter::new(50, ALLPASS_FEEDBACK);

    let mut max_abs = 0.0f32;
    for i in 0..5000 {
        let input = (i as f32 / 20.0).sin();
        let out = allpass.process(input);
        assert!(out.is_finite(), "output must be finite at sample {i}");
        max_abs = max_abs.max(out.abs());
    }
    // Run tail -- filter should drain.
    for _ in 0..500 {
        let out = allpass.process(0.0);
        assert!(out.is_finite(), "tail output must be finite");
    }
    // Output should be bounded (not diverging).
    assert!(
        max_abs < 5.0,
        "allpass output should be bounded, max_abs = {max_abs}"
    );
}

// -- Stereo reverb properties --

#[test]
fn test_stereo_reverb_silence_passthrough() {
    let config = ReverbConfig::default();
    let mut reverb = StereoReverb::new(&config);
    let mut buffer = vec![0.0f32; 200];
    reverb.process_stereo(&mut buffer);
    for &s in &buffer {
        assert!(s.abs() < 1e-6, "silence in = silence out");
    }
}

#[test]
fn test_stereo_reverb_adds_energy() {
    let config = ReverbConfig::new().with_reverb_mix(0.3).with_room_size(0.5);
    let mut reverb = StereoReverb::new(&config);

    // Create stereo impulse.
    let len = 4000;
    let mut buffer = vec![0.0f32; len * 2];
    buffer[0] = 1.0; // L
    buffer[1] = 1.0; // R

    reverb.process_stereo(&mut buffer);

    // After the impulse, there should be reverb tail energy.
    let tail_energy: f32 = buffer[200..].iter().map(|s| s * s).sum();
    assert!(
        tail_energy > 0.001,
        "reverb should add tail energy, got {tail_energy}"
    );
}

#[test]
fn test_stereo_reverb_dry_wet_extremes() {
    // Fully dry: output = input.
    let dry_config = ReverbConfig::new().with_reverb_mix(0.0);
    let mut dry_reverb = StereoReverb::new(&dry_config);
    let mut dry_buf = vec![0.5, 0.3, -0.2, 0.7];
    let original = dry_buf.clone();
    dry_reverb.process_stereo(&mut dry_buf);
    for (i, (&out, &orig)) in dry_buf.iter().zip(original.iter()).enumerate() {
        assert!(
            (out - orig).abs() < 1e-6,
            "dry reverb should pass through at sample {i}: {out} vs {orig}"
        );
    }
}

#[test]
fn test_stereo_reverb_bounded_output() {
    // With bounded input, output should remain finite.
    let config = ReverbConfig::new().with_reverb_mix(0.5).with_room_size(0.9);
    let mut reverb = StereoReverb::new(&config);

    let len = 10000;
    let mut buffer: Vec<f32> = (0..len * 2)
        .map(|i| ((i as f32) / 50.0).sin() * 0.5)
        .collect();

    reverb.process_stereo(&mut buffer);

    for (i, &s) in buffer.iter().enumerate() {
        assert!(
            s.is_finite(),
            "output must be finite at sample {i}, got {s}"
        );
        assert!(
            s.abs() < 5.0,
            "output should be bounded at sample {i}, got {s}"
        );
    }
}

#[test]
fn test_mono_reverb_works() {
    let config = ReverbConfig::new().with_reverb_mix(0.2);
    let mut reverb = StereoReverb::new(&config);

    let mut buffer = vec![0.0f32; 2000];
    buffer[0] = 1.0; // impulse

    reverb.process_mono(&mut buffer);

    // Should have reverb tail.
    let tail_energy: f32 = buffer[100..].iter().map(|s| s * s).sum();
    assert!(tail_energy > 0.0001, "mono reverb should add tail");
}

// -- Early reflections --

#[test]
fn test_early_reflections_add_energy() {
    let len = 2400; // 100ms at 24kHz
    let mut stereo = vec![0.0f32; len * 2];

    // One voice panned left with an impulse.
    let mut voice = vec![0.0f32; len];
    voice[0] = 1.0;

    let voices: Vec<&[f32]> = vec![voice.as_slice()];
    let pans = vec![-0.8];
    let gains = vec![1.0];

    apply_early_reflections(&mut stereo, &voices, &pans, &gains);

    // Should have added some energy beyond sample 0.
    let total_energy: f32 = stereo.iter().map(|s| s * s).sum();
    assert!(
        total_energy > 0.001,
        "early reflections should add energy, got {total_energy}"
    );

    // Energy should be in delayed positions (not at sample 0).
    let delayed_energy: f32 = stereo[2..].iter().map(|s| s * s).sum();
    assert!(
        delayed_energy > 0.0005,
        "reflections should appear after sample 0, got {delayed_energy}"
    );
}

#[test]
fn test_early_reflections_center_pan_symmetric() {
    let len = 2400;
    let mut stereo = vec![0.0f32; len * 2];

    let mut voice = vec![0.0f32; len];
    voice[0] = 1.0;

    let voices: Vec<&[f32]> = vec![voice.as_slice()];
    let pans = vec![0.0]; // center
    let gains = vec![1.0];

    apply_early_reflections(&mut stereo, &voices, &pans, &gains);

    // Center-panned voice: left and right reflection energy should be similar.
    let left_energy: f32 = (0..len).map(|i| stereo[i * 2] * stereo[i * 2]).sum();
    let right_energy: f32 = (0..len)
        .map(|i| stereo[i * 2 + 1] * stereo[i * 2 + 1])
        .sum();

    let ratio = if left_energy > 0.0 && right_energy > 0.0 {
        left_energy / right_energy
    } else {
        1.0
    };
    assert!(
        (ratio - 1.0).abs() < 0.3,
        "center pan should give roughly equal L/R reflections, ratio = {ratio}"
    );
}

#[test]
fn test_early_reflections_pan_asymmetry() {
    let len = 2400;

    // Voice panned hard left.
    let mut stereo_left = vec![0.0f32; len * 2];
    let mut voice = vec![0.0f32; len];
    voice[0] = 1.0;
    let voices: Vec<&[f32]> = vec![voice.as_slice()];
    apply_early_reflections(&mut stereo_left, &voices, &[-1.0], &[1.0]);

    // Voice panned hard right.
    let mut stereo_right = vec![0.0f32; len * 2];
    apply_early_reflections(&mut stereo_right, &voices, &[1.0], &[1.0]);

    // The reflection patterns should differ between left-panned and right-panned
    // voices. Check that the buffers are not identical sample-by-sample.
    let mut differs = false;
    for i in 0..stereo_left.len() {
        if (stereo_left[i] - stereo_right[i]).abs() > 1e-8 {
            differs = true;
            break;
        }
    }
    assert!(
        differs,
        "different pan positions should give different reflection patterns"
    );
}

// -- Full reverb pipeline --

#[test]
fn test_apply_reverb_stereo_full_pipeline() {
    let config = ReverbConfig::new()
        .with_reverb_mix(0.2)
        .with_room_size(0.4)
        .with_early_reflections(true)
        .with_damping(0.5);

    let len = 4800;
    let mut stereo: Vec<f32> = (0..len * 2)
        .map(|i| ((i as f32) / 80.0).sin() * 0.3)
        .collect();
    let original_energy: f32 = stereo.iter().map(|s| s * s).sum();

    let voice = vec![0.3f32; len];
    let voices: Vec<&[f32]> = vec![voice.as_slice()];
    let pans = vec![0.5];
    let gains = vec![0.5];

    apply_reverb(
        &mut stereo,
        &config,
        true,
        Some(&voices),
        Some(&pans),
        Some(&gains),
    )
    .unwrap();

    let final_energy: f32 = stereo.iter().map(|s| s * s).sum();
    // Reverb redistributes energy temporally. With mix=0.2, the dry
    // signal retains 80% amplitude but wet portion may spread energy
    // beyond the buffer. Total energy should not drop catastrophically.
    assert!(
        final_energy >= original_energy * 0.3,
        "reverb should not drastically reduce energy: {final_energy} vs {original_energy}"
    );

    // All samples should be finite.
    for (i, &s) in stereo.iter().enumerate() {
        assert!(s.is_finite(), "sample {i} is not finite: {s}");
    }
}

#[test]
fn test_apply_reverb_zero_mix_passthrough() {
    let config = ReverbConfig::new().with_reverb_mix(0.0);
    let mut buffer = vec![0.5, -0.3, 0.7, 0.1];
    let original = buffer.clone();
    apply_reverb(&mut buffer, &config, true, None, None, None).unwrap();
    for (i, (&out, &orig)) in buffer.iter().zip(original.iter()).enumerate() {
        assert!(
            (out - orig).abs() < 1e-6,
            "zero mix should pass through at {i}"
        );
    }
}

#[test]
fn test_apply_reverb_mono() {
    let config = ReverbConfig::new().with_reverb_mix(0.15);
    let mut buffer = vec![0.0f32; 3000];
    buffer[0] = 1.0;
    apply_reverb(&mut buffer, &config, false, None, None, None).unwrap();

    let tail_energy: f32 = buffer[100..].iter().map(|s| s * s).sum();
    assert!(tail_energy > 0.0001, "mono reverb tail should exist");
}

#[test]
fn test_apply_reverb_skips_early_reflections_when_disabled() {
    let config = ReverbConfig::new()
        .with_reverb_mix(0.2)
        .with_early_reflections(false);

    let len = 2400;
    let mut stereo = vec![0.0f32; len * 2];
    stereo[0] = 1.0;
    stereo[1] = 1.0;

    let voice = vec![1.0f32; len];
    let voices: Vec<&[f32]> = vec![voice.as_slice()];

    // Even with voice data, early reflections should not be applied.
    apply_reverb(
        &mut stereo,
        &config,
        true,
        Some(&voices),
        Some(&[0.5]),
        Some(&[1.0]),
    )
    .unwrap();

    // This is a smoke test: the output should be valid regardless.
    for &s in &stereo {
        assert!(s.is_finite(), "all samples must be finite");
    }
}

// -- Frequency response property --

#[test]
fn test_reverb_frequency_response_damping_effect() {
    // Higher damping should attenuate high frequencies more in the reverb tail.
    let len = 8000;

    // Generate a high-frequency test signal (6kHz sine at 24kHz SR).
    let high_freq: Vec<f32> = (0..len)
        .map(|i| (2.0 * std::f32::consts::PI * 6000.0 * i as f32 / 24000.0).sin() * 0.3)
        .collect();

    // Low damping.
    let low_damp_config = ReverbConfig::new()
        .with_reverb_mix(1.0)
        .with_room_size(0.5)
        .with_damping(0.1)
        .with_early_reflections(false);
    let mut low_damp_buf = high_freq.clone();
    apply_reverb(&mut low_damp_buf, &low_damp_config, false, None, None, None).unwrap();
    let low_damp_tail_energy: f32 = low_damp_buf[4000..].iter().map(|s| s * s).sum();

    // High damping.
    let high_damp_config = ReverbConfig::new()
        .with_reverb_mix(1.0)
        .with_room_size(0.5)
        .with_damping(0.9)
        .with_early_reflections(false);
    let mut high_damp_buf = high_freq;
    apply_reverb(
        &mut high_damp_buf,
        &high_damp_config,
        false,
        None,
        None,
        None,
    )
    .unwrap();
    let high_damp_tail_energy: f32 = high_damp_buf[4000..].iter().map(|s| s * s).sum();

    // Higher damping should produce less tail energy for high-frequency content.
    assert!(
        high_damp_tail_energy < low_damp_tail_energy,
        "high damping should attenuate HF tail more: high={high_damp_tail_energy} vs low={low_damp_tail_energy}"
    );
}

// -- Room size affects decay length --

#[test]
fn test_room_size_affects_decay_length() {
    let len = 12000; // 500ms at 24kHz

    // Small room.
    let small_config = ReverbConfig::new()
        .with_reverb_mix(1.0)
        .with_room_size(0.1)
        .with_damping(0.3)
        .with_early_reflections(false);
    let mut small_buf = vec![0.0f32; len];
    small_buf[0] = 1.0;
    apply_reverb(&mut small_buf, &small_config, false, None, None, None).unwrap();
    let small_late_energy: f32 = small_buf[6000..].iter().map(|s| s * s).sum();

    // Large room.
    let large_config = ReverbConfig::new()
        .with_reverb_mix(1.0)
        .with_room_size(0.9)
        .with_damping(0.3)
        .with_early_reflections(false);
    let mut large_buf = vec![0.0f32; len];
    large_buf[0] = 1.0;
    apply_reverb(&mut large_buf, &large_config, false, None, None, None).unwrap();
    let large_late_energy: f32 = large_buf[6000..].iter().map(|s| s * s).sum();

    // Larger room should have more late-tail energy.
    assert!(
        large_late_energy > small_late_energy,
        "large room should have more tail: large={large_late_energy} vs small={small_late_energy}"
    );
}
