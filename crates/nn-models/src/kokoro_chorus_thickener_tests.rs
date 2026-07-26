// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the kokoro_chorus_thickener module.

use super::*;

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

#[test]
fn test_default_config_validates() {
    let config = ThickenerConfig::default();
    config.validate().expect("default config should be valid");
}

#[test]
fn test_new_config_validates() {
    let config = ThickenerConfig::new().expect("new() should succeed");
    assert!((config.pitch_depth_cents - 8.0).abs() < f32::EPSILON);
    assert!((config.time_depth_ms - 1.5).abs() < f32::EPSILON);
    assert!((config.amplitude_depth_db - 1.0).abs() < f32::EPSILON);
    assert!((config.lfo_rate_hz - 0.5).abs() < f32::EPSILON);
    assert_eq!(config.lfo_shape, LfoShape::Sine);
    assert!((config.chorus_delay_ms - 15.0).abs() < f32::EPSILON);
    assert!((config.chorus_depth_ms - 5.0).abs() < f32::EPSILON);
    assert!((config.mix - 0.4).abs() < f32::EPSILON);
}

#[test]
fn test_config_pitch_depth_nan_rejected() {
    let config = ThickenerConfig::default().with_pitch_depth_cents(f32::NAN);
    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("pitch_depth_cents"));
}

#[test]
fn test_config_pitch_depth_out_of_range() {
    let config = ThickenerConfig::default().with_pitch_depth_cents(31.0);
    assert!(config.validate().is_err());
    let config = ThickenerConfig::default().with_pitch_depth_cents(-0.1);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_time_depth_out_of_range() {
    let config = ThickenerConfig::default().with_time_depth_ms(11.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_amplitude_depth_out_of_range() {
    let config = ThickenerConfig::default().with_amplitude_depth_db(7.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_lfo_rate_out_of_range() {
    let config = ThickenerConfig::default().with_lfo_rate_hz(0.005);
    assert!(config.validate().is_err());
    let config = ThickenerConfig::default().with_lfo_rate_hz(11.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_mix_out_of_range() {
    let config = ThickenerConfig::default().with_mix(-0.1);
    assert!(config.validate().is_err());
    let config = ThickenerConfig::default().with_mix(1.1);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_chorus_delay_out_of_range() {
    let config = ThickenerConfig::default().with_chorus_delay_ms(51.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_chorus_depth_out_of_range() {
    let config = ThickenerConfig::default().with_chorus_depth_ms(21.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_inf_rejected() {
    let config = ThickenerConfig::default().with_amplitude_depth_db(f32::INFINITY);
    assert!(config.validate().is_err());
}

// ---------------------------------------------------------------------------
// Presets validate
// ---------------------------------------------------------------------------

#[test]
fn test_preset_subtle_validates() {
    ThickenerConfig::subtle()
        .validate()
        .expect("subtle preset should be valid");
}

#[test]
fn test_preset_lush_validates() {
    ThickenerConfig::lush()
        .validate()
        .expect("lush preset should be valid");
}

#[test]
fn test_preset_dramatic_validates() {
    ThickenerConfig::dramatic()
        .validate()
        .expect("dramatic preset should be valid");
}

#[test]
fn test_preset_gentle_sway_validates() {
    ThickenerConfig::gentle_sway()
        .validate()
        .expect("gentle_sway preset should be valid");
}

// ---------------------------------------------------------------------------
// Processor construction
// ---------------------------------------------------------------------------

#[test]
fn test_processor_new_default() {
    let config = ThickenerConfig::default();
    let proc = ThickenerProcessor::new(&config, 4, 44100.0).expect("should succeed");
    assert_eq!(proc.n_voices(), 4);
    assert!((proc.sample_rate() - 44100.0).abs() < f32::EPSILON);
}

#[test]
fn test_processor_new_invalid_sample_rate() {
    let config = ThickenerConfig::default();
    let err = ThickenerProcessor::new(&config, 4, 500.0).unwrap_err();
    assert!(err.to_string().contains("sample_rate"));
}

#[test]
fn test_processor_new_zero_voices() {
    let config = ThickenerConfig::default();
    let err = ThickenerProcessor::new(&config, 0, 44100.0).unwrap_err();
    assert!(err.to_string().contains("n_voices"));
}

#[test]
fn test_processor_new_invalid_config_propagated() {
    let config = ThickenerConfig::default().with_mix(2.0);
    assert!(ThickenerProcessor::new(&config, 4, 44100.0).is_err());
}

// ---------------------------------------------------------------------------
// Processing behavior
// ---------------------------------------------------------------------------

#[test]
fn test_process_empty_voices() {
    let config = ThickenerConfig::default();
    let mut proc = ThickenerProcessor::new(&config, 4, 44100.0).unwrap();
    let mut voices: Vec<Vec<f32>> = vec![];
    proc.process_voices(&mut voices); // should not panic
}

#[test]
fn test_process_silence_stays_near_silent() {
    let config = ThickenerConfig::default();
    let mut proc = ThickenerProcessor::new(&config, 2, 44100.0).unwrap();
    let mut voices = vec![vec![0.0f32; 1024]; 2];
    proc.process_voices(&mut voices);
    // With zero input, output should be zero or very near zero.
    for voice in &voices {
        for &s in voice {
            assert!(
                s.abs() < 1e-6,
                "silence in should produce near-silence out, got {s}"
            );
        }
    }
}

#[test]
fn test_process_modifies_signal() {
    let config = ThickenerConfig::lush();
    let mut proc = ThickenerProcessor::new(&config, 2, 44100.0).unwrap();
    // Generate a simple sine tone.
    let len = 4096;
    let freq = 440.0;
    let sine: Vec<f32> = (0..len)
        .map(|i| (std::f32::consts::TAU * freq * i as f32 / 44100.0).sin())
        .collect();
    let mut voices = vec![sine.clone(); 2];
    proc.process_voices(&mut voices);
    // After processing, the signal should differ from the original.
    let mut diff_count = 0usize;
    for (orig, processed) in sine.iter().zip(voices[0].iter()) {
        if (orig - processed).abs() > 1e-6 {
            diff_count += 1;
        }
    }
    assert!(
        diff_count > len / 4,
        "expected significant signal modification, got {diff_count}/{len} changed samples"
    );
}

#[test]
fn test_process_output_is_finite() {
    let config = ThickenerConfig::dramatic();
    let mut proc = ThickenerProcessor::new(&config, 3, 48000.0).unwrap();
    let len = 2048;
    let mut voices: Vec<Vec<f32>> = (0..3)
        .map(|v| {
            (0..len)
                .map(|i| {
                    let f = 300.0 + v as f32 * 50.0;
                    (std::f32::consts::TAU * f * i as f32 / 48000.0).sin() * 0.5
                })
                .collect()
        })
        .collect();
    proc.process_voices(&mut voices);
    for (vi, voice) in voices.iter().enumerate() {
        for (si, &s) in voice.iter().enumerate() {
            assert!(
                s.is_finite(),
                "non-finite sample at voice {vi} sample {si}: {s}"
            );
        }
    }
}

#[test]
fn test_process_nan_input_sanitized() {
    let config = ThickenerConfig::subtle();
    let mut proc = ThickenerProcessor::new(&config, 1, 44100.0).unwrap();
    let mut voices = vec![vec![f32::NAN; 128]];
    proc.process_voices(&mut voices);
    for &s in &voices[0] {
        assert!(
            s.is_finite(),
            "NaN input should produce finite output, got {s}"
        );
    }
}

#[test]
fn test_process_different_voices_get_different_modulation() {
    let config = ThickenerConfig::lush();
    let mut proc = ThickenerProcessor::new(&config, 4, 44100.0).unwrap();
    // Give all voices the same input.
    let len = 2048;
    let tone: Vec<f32> = (0..len)
        .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / 44100.0).sin())
        .collect();
    let mut voices = vec![tone; 4];
    proc.process_voices(&mut voices);
    // Voices should differ from each other due to per-voice phase offsets.
    let mut any_differ = false;
    for i in 1..4 {
        let mut diffs = 0;
        for j in 0..len {
            if (voices[0][j] - voices[i][j]).abs() > 1e-6 {
                diffs += 1;
            }
        }
        if diffs > len / 10 {
            any_differ = true;
        }
    }
    assert!(
        any_differ,
        "voices with same input should receive different modulation"
    );
}

#[test]
fn test_reset_restores_initial_state() {
    let config = ThickenerConfig::default();
    let mut proc = ThickenerProcessor::new(&config, 2, 44100.0).unwrap();
    // Process some audio.
    let mut voices = vec![vec![0.5f32; 512]; 2];
    proc.process_voices(&mut voices);
    // Reset.
    proc.reset();
    // Process silence — should behave like freshly constructed.
    let mut silence = vec![vec![0.0f32; 256]; 2];
    proc.process_voices(&mut silence);
    for voice in &silence {
        for &s in voice {
            assert!(
                s.abs() < 1e-6,
                "after reset, silence input should produce near-silence"
            );
        }
    }
}

#[test]
fn test_process_single_voice() {
    let config = ThickenerConfig::subtle();
    let mut proc = ThickenerProcessor::new(&config, 1, 44100.0).unwrap();
    let mut voices = vec![vec![0.3f32; 256]];
    proc.process_voices(&mut voices);
    // Should not panic and output should be finite.
    for &s in &voices[0] {
        assert!(s.is_finite());
    }
}

// ---------------------------------------------------------------------------
// LFO shape coverage
// ---------------------------------------------------------------------------

#[test]
fn test_triangle_lfo_shape() {
    let config = ThickenerConfig::default().with_lfo_shape(LfoShape::Triangle);
    let mut proc = ThickenerProcessor::new(&config, 2, 44100.0).unwrap();
    let len = 1024;
    let mut voices: Vec<Vec<f32>> = vec![
        (0..len)
            .map(|i| (std::f32::consts::TAU * 300.0 * i as f32 / 44100.0).sin())
            .collect();
        2
    ];
    proc.process_voices(&mut voices);
    for voice in &voices {
        for &s in voice {
            assert!(s.is_finite());
        }
    }
}

// ---------------------------------------------------------------------------
// Builder chain
// ---------------------------------------------------------------------------

#[test]
fn test_builder_chain() {
    let config = ThickenerConfig::default()
        .with_pitch_depth_cents(12.0)
        .with_time_depth_ms(2.0)
        .with_amplitude_depth_db(1.5)
        .with_lfo_rate_hz(0.8)
        .with_lfo_shape(LfoShape::Triangle)
        .with_chorus_delay_ms(20.0)
        .with_chorus_depth_ms(7.0)
        .with_mix(0.5);
    config
        .validate()
        .expect("builder chain config should be valid");
    assert!((config.pitch_depth_cents - 12.0).abs() < f32::EPSILON);
    assert!((config.time_depth_ms - 2.0).abs() < f32::EPSILON);
    assert!((config.amplitude_depth_db - 1.5).abs() < f32::EPSILON);
    assert!((config.lfo_rate_hz - 0.8).abs() < f32::EPSILON);
    assert_eq!(config.lfo_shape, LfoShape::Triangle);
    assert!((config.chorus_delay_ms - 20.0).abs() < f32::EPSILON);
    assert!((config.chorus_depth_ms - 7.0).abs() < f32::EPSILON);
    assert!((config.mix - 0.5).abs() < f32::EPSILON);
}

// ---------------------------------------------------------------------------
// db_to_linear utility
// ---------------------------------------------------------------------------

#[test]
fn test_db_to_linear_zero() {
    let g = db_to_linear(0.0);
    assert!((g - 1.0).abs() < 1e-6, "0 dB should be unity gain, got {g}");
}

#[test]
fn test_db_to_linear_six_db() {
    let g = db_to_linear(6.0);
    assert!((g - 1.9953).abs() < 0.01, "+6 dB should be ~2.0, got {g}");
}

#[test]
fn test_db_to_linear_negative() {
    let g = db_to_linear(-6.0);
    assert!((g - 0.5012).abs() < 0.01, "-6 dB should be ~0.5, got {g}");
}

#[test]
fn test_db_to_linear_nan_returns_unity() {
    assert!((db_to_linear(f32::NAN) - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_db_to_linear_inf_returns_unity() {
    assert!((db_to_linear(f32::INFINITY) - 1.0).abs() < f32::EPSILON);
}

// ---------------------------------------------------------------------------
// Edge: more voices than configured
// ---------------------------------------------------------------------------

#[test]
fn test_process_more_voices_than_configured() {
    let config = ThickenerConfig::default();
    let mut proc = ThickenerProcessor::new(&config, 2, 44100.0).unwrap();
    // Pass 4 voices but processor configured for 2 — should process first 2.
    let mut voices = vec![vec![0.3f32; 128]; 4];
    proc.process_voices(&mut voices);
    // First 2 should be processed (finite).
    for &s in &voices[0] {
        assert!(s.is_finite());
    }
    for &s in &voices[1] {
        assert!(s.is_finite());
    }
    // Voices 2-3 should be unchanged (still 0.3).
    for &s in &voices[2] {
        assert!((s - 0.3).abs() < f32::EPSILON);
    }
}

// ---------------------------------------------------------------------------
// Accessor coverage
// ---------------------------------------------------------------------------

#[test]
fn test_config_accessor() {
    let config = ThickenerConfig::lush();
    let proc = ThickenerProcessor::new(&config, 3, 44100.0).unwrap();
    assert!((proc.config().pitch_depth_cents - 10.0).abs() < f32::EPSILON);
}
