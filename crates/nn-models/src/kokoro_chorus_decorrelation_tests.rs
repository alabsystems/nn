// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `kokoro_chorus_decorrelation` allpass diffusion de-correlation.

use super::*;

#[test]
fn test_default_config_validates() {
    let cfg = DecorrelationConfig::default();
    cfg.validate().expect("default config should be valid");
}

#[test]
fn test_presets_validate() {
    for (name, cfg) in [
        ("subtle", DecorrelationConfig::subtle()),
        ("wide", DecorrelationConfig::wide()),
        ("maximum", DecorrelationConfig::maximum()),
        ("bass_safe", DecorrelationConfig::bass_safe()),
    ] {
        cfg.validate()
            .unwrap_or_else(|e| panic!("preset '{name}' should validate: {e}"));
    }
}

#[test]
fn test_invalid_config_rejected() {
    // n_stages = 0
    assert!(DecorrelationConfig::new(0, 5.0, 0.7, true, 42, 0.5).is_err());
    // n_stages = 33
    assert!(DecorrelationConfig::new(33, 5.0, 0.7, true, 42, 0.5).is_err());
    // max_delay_ms NaN
    assert!(DecorrelationConfig::new(8, f32::NAN, 0.7, true, 42, 0.5).is_err());
    // diffusion > 1.0
    assert!(DecorrelationConfig::new(8, 5.0, 1.5, true, 42, 0.5).is_err());
    // mix < 0.0
    assert!(DecorrelationConfig::new(8, 5.0, 0.7, true, 42, -0.1).is_err());
}

#[test]
fn test_builder_pattern() {
    let cfg = DecorrelationConfig::default()
        .with_n_stages(4)
        .with_max_delay_ms(3.0)
        .with_diffusion(0.5)
        .with_frequency_dependent(false)
        .with_per_voice_seed(123)
        .with_mix(0.4);
    assert_eq!(cfg.n_stages, 4);
    assert!((cfg.max_delay_ms - 3.0).abs() < 1e-6);
    assert!((cfg.diffusion - 0.5).abs() < 1e-6);
    assert!(!cfg.frequency_dependent);
    assert_eq!(cfg.per_voice_seed, 123);
    assert!((cfg.mix - 0.4).abs() < 1e-6);
}

#[test]
fn test_processor_creation() {
    let cfg = DecorrelationConfig::default();
    let proc = DecorrelationProcessor::new(&cfg, 4, 24000.0);
    assert!(proc.is_ok());
    let proc = proc.unwrap();
    assert_eq!(proc.n_voices(), 4);
    assert!((proc.sample_rate() - 24000.0).abs() < 1e-6);
}

#[test]
fn test_processor_rejects_zero_voices() {
    let cfg = DecorrelationConfig::default();
    assert!(DecorrelationProcessor::new(&cfg, 0, 24000.0).is_err());
}

#[test]
fn test_processor_rejects_bad_sample_rate() {
    let cfg = DecorrelationConfig::default();
    assert!(DecorrelationProcessor::new(&cfg, 4, 0.0).is_err());
    assert!(DecorrelationProcessor::new(&cfg, 4, -1.0).is_err());
    assert!(DecorrelationProcessor::new(&cfg, 4, f32::NAN).is_err());
}

#[test]
fn test_broadband_energy_preservation() {
    // Allpass filters preserve energy. With mix=1.0, the output power
    // should be very close to input power.
    let cfg = DecorrelationConfig::default()
        .with_frequency_dependent(false)
        .with_mix(1.0);
    let mut proc = DecorrelationProcessor::new(&cfg, 2, 24000.0).unwrap();

    // Generate a test signal (sine wave).
    let n = 4096;
    let signal: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
        .collect();

    let input_power: f32 = signal.iter().map(|s| s * s).sum::<f32>() / n as f32;

    let mut voices = vec![signal.clone(), signal];
    proc.process_voices(&mut voices);

    for (v_idx, voice) in voices.iter().enumerate() {
        let output_power: f32 = voice.iter().map(|s| s * s).sum::<f32>() / n as f32;
        let ratio = output_power / input_power;
        // Allpass is unity gain, so ratio should be close to 1.0.
        // Allow some tolerance for filter startup transient.
        assert!(
            (0.8..=1.2).contains(&ratio),
            "voice {v_idx}: power ratio {ratio:.4} outside [0.8, 1.2]"
        );
    }
}

#[test]
fn test_voices_become_decorrelated() {
    // Two identical input signals should produce different outputs
    // after decorrelation (different allpass chains per voice).
    let cfg = DecorrelationConfig::default()
        .with_frequency_dependent(false)
        .with_mix(1.0);
    let mut proc = DecorrelationProcessor::new(&cfg, 2, 24000.0).unwrap();

    let n = 4096;
    let signal: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
        .collect();

    let mut voices = vec![signal.clone(), signal];
    proc.process_voices(&mut voices);

    // Compute cross-correlation at lag 0. If voices are identical,
    // correlation = 1.0. After decorrelation, it should be < 1.0.
    let mean0: f32 = voices[0].iter().sum::<f32>() / n as f32;
    let mean1: f32 = voices[1].iter().sum::<f32>() / n as f32;

    let mut cov = 0.0f64;
    let mut var0 = 0.0f64;
    let mut var1 = 0.0f64;
    for i in 0..n {
        let d0 = f64::from(voices[0][i] - mean0);
        let d1 = f64::from(voices[1][i] - mean1);
        cov += d0 * d1;
        var0 += d0 * d0;
        var1 += d1 * d1;
    }
    let denom = (var0 * var1).sqrt();
    let correlation = if denom > 1e-12 { cov / denom } else { 1.0 };

    // With decorrelation active, the correlation should be well below 1.
    assert!(
        correlation < 0.95,
        "voices should be decorrelated: correlation = {correlation:.4}"
    );
}

#[test]
fn test_zero_mix_is_passthrough() {
    let cfg = DecorrelationConfig::default().with_mix(0.0);
    let mut proc = DecorrelationProcessor::new(&cfg, 1, 24000.0).unwrap();

    let signal: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let original = signal.clone();
    let mut voices = vec![signal];
    proc.process_voices(&mut voices);

    for (i, (&out, &orig)) in voices[0].iter().zip(original.iter()).enumerate() {
        assert!(
            (out - orig).abs() < 1e-7,
            "sample {i}: expected {orig}, got {out}"
        );
    }
}

#[test]
fn test_deterministic_output() {
    let cfg = DecorrelationConfig::default().with_frequency_dependent(false);
    let signal: Vec<f32> = (0..1024)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
        .collect();

    // Run twice with same config.
    let mut proc1 = DecorrelationProcessor::new(&cfg, 2, 24000.0).unwrap();
    let mut voices1 = vec![signal.clone(), signal.clone()];
    proc1.process_voices(&mut voices1);

    let mut proc2 = DecorrelationProcessor::new(&cfg, 2, 24000.0).unwrap();
    let mut voices2 = vec![signal.clone(), signal];
    proc2.process_voices(&mut voices2);

    for v in 0..2 {
        for i in 0..1024 {
            assert!(
                (voices1[v][i] - voices2[v][i]).abs() < 1e-7,
                "non-deterministic output at voice {v} sample {i}"
            );
        }
    }
}

#[test]
fn test_reset_clears_state() {
    let cfg = DecorrelationConfig::default()
        .with_frequency_dependent(false)
        .with_mix(1.0);
    let mut proc = DecorrelationProcessor::new(&cfg, 1, 24000.0).unwrap();

    let signal: Vec<f32> = (0..512)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
        .collect();

    // Process once.
    let mut voices = vec![signal.clone()];
    proc.process_voices(&mut voices);
    let first_run = voices[0].clone();

    // Reset and process again.
    proc.reset();
    let mut voices = vec![signal];
    proc.process_voices(&mut voices);
    let second_run = voices[0].clone();

    for i in 0..512 {
        assert!(
            (first_run[i] - second_run[i]).abs() < 1e-7,
            "reset did not clear state: sample {i} differs"
        );
    }
}

#[test]
fn test_frequency_dependent_mode() {
    let cfg = DecorrelationConfig::default()
        .with_frequency_dependent(true)
        .with_mix(1.0);
    let mut proc = DecorrelationProcessor::new(&cfg, 2, 24000.0).unwrap();

    // Process a mixed signal with low and high frequency content.
    let n = 4096;
    let signal: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / 24000.0;
            // 200 Hz (low) + 5000 Hz (high)
            0.5 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
                + 0.5 * (2.0 * std::f32::consts::PI * 5000.0 * t).sin()
        })
        .collect();

    let mut voices = vec![signal.clone(), signal];
    proc.process_voices(&mut voices);

    // Basic sanity: output should be finite and non-zero.
    for (v_idx, voice) in voices.iter().enumerate() {
        let has_nonzero = voice.iter().any(|&s| s.abs() > 1e-6);
        assert!(has_nonzero, "voice {v_idx} is all zeros");
        let all_finite = voice.iter().all(|s| s.is_finite());
        assert!(all_finite, "voice {v_idx} has non-finite samples");
    }
}

#[test]
fn test_nan_inf_resilience() {
    let cfg = DecorrelationConfig::default()
        .with_frequency_dependent(false)
        .with_mix(1.0);
    let mut proc = DecorrelationProcessor::new(&cfg, 1, 24000.0).unwrap();

    let mut signal = vec![0.5f32; 128];
    signal[10] = f32::NAN;
    signal[20] = f32::INFINITY;
    signal[30] = f32::NEG_INFINITY;

    let mut voices = vec![signal];
    proc.process_voices(&mut voices);

    // All output samples should be finite.
    for (i, &s) in voices[0].iter().enumerate() {
        assert!(s.is_finite(), "sample {i} is non-finite: {s}");
    }
}

#[test]
fn test_mismatched_voice_count_robust() {
    // Processor configured for 4 voices, but only 2 provided.
    // Should process the 2 available without panic.
    let cfg = DecorrelationConfig::default().with_frequency_dependent(false);
    let mut proc = DecorrelationProcessor::new(&cfg, 4, 24000.0).unwrap();

    let signal = vec![0.5f32; 64];
    let mut voices = vec![signal.clone(), signal];
    proc.process_voices(&mut voices); // should not panic
    assert_eq!(voices.len(), 2);
}
