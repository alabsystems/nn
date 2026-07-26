// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for formant-preserving ensemble voice blending.

use super::*;

// ---------------------------------------------------------------------------
// Config validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_config_default_validates() {
    let config = EnsembleBlendConfig::default();
    config.validate().unwrap();
}

#[test]
fn test_config_disabled_validates() {
    let config = EnsembleBlendConfig::disabled();
    config.validate().unwrap();
    assert!(config.blend_strength < 1e-6);
    assert!(!config.formant_preservation);
    assert!(!config.harmonic_alignment);
}

#[test]
fn test_config_new_validates() {
    let config = EnsembleBlendConfig::new(0.75).unwrap();
    assert!((config.blend_strength - 0.75).abs() < 1e-6);
}

#[test]
fn test_config_invalid_blend_strength_nan() {
    let config = EnsembleBlendConfig {
        blend_strength: f32::NAN,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_invalid_blend_strength_inf() {
    let config = EnsembleBlendConfig {
        blend_strength: f32::INFINITY,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_invalid_blend_strength_negative() {
    let config = EnsembleBlendConfig {
        blend_strength: -0.1,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_invalid_blend_strength_over_one() {
    let config = EnsembleBlendConfig {
        blend_strength: 1.1,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_invalid_min_period_too_small() {
    let config = EnsembleBlendConfig {
        min_period: 5,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_invalid_max_period_less_than_min() {
    let config = EnsembleBlendConfig {
        min_period: 100,
        max_period: 50,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_with_formant_preservation() {
    let config = EnsembleBlendConfig::default().with_formant_preservation(false);
    assert!(!config.formant_preservation);
}

#[test]
fn test_config_with_harmonic_alignment() {
    let config = EnsembleBlendConfig::default().with_harmonic_alignment(false);
    assert!(!config.harmonic_alignment);
}

// ---------------------------------------------------------------------------
// FormantShift unit tests
// ---------------------------------------------------------------------------

/// Generate a simple sine wave at the given frequency.
fn sine_wave(freq_hz: f32, sample_rate: u32, n_samples: usize) -> Vec<f32> {
    (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (std::f32::consts::TAU * freq_hz * t).sin()
        })
        .collect()
}

/// Compute RMS of a buffer.
fn test_rms(audio: &[f32]) -> f32 {
    rms_energy(audio)
}

#[test]
fn test_formant_shift_identity() {
    // Pitch factor 1.0 should return identical audio.
    let audio = sine_wave(220.0, 24000, 4800);
    let shifter = FormantShift::new(30, 300);
    let shifted = shifter.shift(&audio, 1.0);
    assert_eq!(shifted.len(), audio.len());
    // Should be identical (no processing).
    for (a, b) in audio.iter().zip(shifted.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "Identity shift should not modify audio",
        );
    }
}

#[test]
fn test_formant_shift_empty_input() {
    let audio: Vec<f32> = Vec::new();
    let shifter = FormantShift::new(30, 300);
    let shifted = shifter.shift(&audio, 1.5);
    assert!(shifted.is_empty());
}

#[test]
fn test_formant_shift_preserves_length() {
    let audio = sine_wave(200.0, 24000, 9600);
    let shifter = FormantShift::new(30, 300);
    let shifted = shifter.shift(&audio, 1.2);
    assert_eq!(shifted.len(), audio.len());
}

#[test]
fn test_formant_shift_preserves_rms_energy() {
    // RMS energy should be roughly preserved after pitch shift.
    let audio = sine_wave(200.0, 24000, 9600);
    let original_rms = test_rms(&audio);
    let shifter = FormantShift::new(30, 300);

    for factor in &[0.8, 0.9, 1.1, 1.2] {
        let shifted = shifter.shift(&audio, *factor);
        let shifted_rms = test_rms(&shifted);
        let ratio = shifted_rms / original_rms;
        // Allow 40% tolerance -- PSOLA overlap-add can modulate amplitude.
        assert!(
            (0.6..=1.6).contains(&ratio),
            "RMS ratio {ratio} out of tolerance for factor {factor}",
        );
    }
}

#[test]
fn test_formant_shift_nan_factor_returns_copy() {
    let audio = sine_wave(200.0, 24000, 2400);
    let shifter = FormantShift::new(30, 300);
    let shifted = shifter.shift(&audio, f32::NAN);
    // NaN factor should be treated as identity.
    assert_eq!(shifted.len(), audio.len());
    for (a, b) in audio.iter().zip(shifted.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_formant_shift_output_is_finite() {
    // Ensure no NaN/Inf in output even with edge-case input.
    let mut audio = sine_wave(150.0, 24000, 4800);
    audio[100] = f32::NAN;
    audio[200] = f32::INFINITY;
    let shifter = FormantShift::new(30, 300);
    let shifted = shifter.shift(&audio, 1.1);
    for (i, &s) in shifted.iter().enumerate() {
        assert!(s.is_finite(), "Non-finite output at index {i}: {s}");
    }
}

#[test]
fn test_formant_shift_spectral_centroid_preserved() {
    // Formant-preserving shift should keep the spectral centroid (brightness)
    // roughly the same, unlike naive resampling which shifts formants too.
    let audio = sine_wave(200.0, 24000, 9600);
    let aligner = SpectralAlignment::new(2048);
    let original_centroid = aligner.estimate_centroid(&audio);

    let shifter = FormantShift::new(30, 300);
    let shifted = shifter.shift(&audio, 1.15); // 15% pitch up
    let shifted_centroid = aligner.estimate_centroid(&shifted);

    // The centroid should be within a reasonable tolerance.
    // For a pure sine wave the centroid proxy (ZCR) is tightly coupled to
    // pitch, so we allow wider tolerance here. For complex signals (speech),
    // formant preservation would show a stronger effect.
    let diff = (original_centroid - shifted_centroid).abs();
    assert!(
        diff < 0.25,
        "Centroid shifted too much: original={original_centroid}, shifted={shifted_centroid}, diff={diff}",
    );
}

// ---------------------------------------------------------------------------
// SpectralAlignment unit tests
// ---------------------------------------------------------------------------

#[test]
fn test_spectral_centroid_silence() {
    let audio = vec![0.0f32; 4800];
    let aligner = SpectralAlignment::new(2048);
    let centroid = aligner.estimate_centroid(&audio);
    assert!(
        (centroid - 0.5).abs() < 1e-6,
        "Silence should have neutral centroid"
    );
}

#[test]
fn test_spectral_centroid_low_frequency() {
    // A low-frequency signal should have a lower centroid than a high-frequency one.
    let low = sine_wave(100.0, 24000, 9600);
    let high = sine_wave(4000.0, 24000, 9600);
    let aligner = SpectralAlignment::new(2048);
    let c_low = aligner.estimate_centroid(&low);
    let c_high = aligner.estimate_centroid(&high);
    assert!(
        c_high > c_low,
        "High freq centroid ({c_high}) should exceed low freq ({c_low})",
    );
}

#[test]
fn test_spectral_centroid_in_range() {
    let audio = sine_wave(440.0, 24000, 4800);
    let aligner = SpectralAlignment::new(2048);
    let centroid = aligner.estimate_centroid(&audio);
    assert!(
        (0.0..=1.0).contains(&centroid),
        "Centroid {centroid} out of [0, 1] range",
    );
}

#[test]
fn test_align_voice_no_change_at_zero_strength() {
    let mut audio = sine_wave(300.0, 24000, 4800);
    let original = audio.clone();
    let aligner = SpectralAlignment::new(2048);
    aligner.align_voice(&mut audio, 0.3, 0.7, 0.0);
    for (a, b) in audio.iter().zip(original.iter()) {
        assert!(
            (a - b).abs() < 1e-7,
            "Zero strength should not modify audio"
        );
    }
}

#[test]
fn test_align_voice_modifies_at_full_strength() {
    let mut audio = sine_wave(300.0, 24000, 4800);
    let original = audio.clone();
    let aligner = SpectralAlignment::new(2048);
    // Big centroid difference + full strength should produce a change.
    aligner.align_voice(&mut audio, 0.1, 0.9, 1.0);
    let any_different = audio
        .iter()
        .zip(original.iter())
        .any(|(a, b)| (a - b).abs() > 1e-7);
    assert!(any_different, "Full-strength alignment should modify audio");
}

#[test]
fn test_align_voice_output_finite() {
    let mut audio = sine_wave(300.0, 24000, 4800);
    let aligner = SpectralAlignment::new(2048);
    aligner.align_voice(&mut audio, 0.2, 0.8, 0.5);
    for (i, &s) in audio.iter().enumerate() {
        assert!(s.is_finite(), "Non-finite at index {i}: {s}");
    }
}

// ---------------------------------------------------------------------------
// blend_voices integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_blend_voices_empty() {
    let mut voices: Vec<Vec<f32>> = Vec::new();
    let config = EnsembleBlendConfig::default();
    blend_voices(&mut voices, &config, 24000).unwrap();
}

#[test]
fn test_blend_voices_single_voice_passthrough() {
    let original = sine_wave(220.0, 24000, 4800);
    let mut voices = vec![original.clone()];
    let config = EnsembleBlendConfig::default();
    blend_voices(&mut voices, &config, 24000).unwrap();
    // Single voice should not be modified.
    for (a, b) in voices[0].iter().zip(original.iter()) {
        assert!((a - b).abs() < 1e-7, "Single voice should be passthrough");
    }
}

#[test]
fn test_blend_voices_zero_strength_passthrough() {
    let v1 = sine_wave(200.0, 24000, 4800);
    let v2 = sine_wave(300.0, 24000, 4800);
    let originals = vec![v1, v2];
    let mut voices = originals.clone();
    let config = EnsembleBlendConfig::new(0.0).unwrap();
    blend_voices(&mut voices, &config, 24000).unwrap();
    for (voice, orig) in voices.iter().zip(originals.iter()) {
        for (a, b) in voice.iter().zip(orig.iter()) {
            assert!(
                (a - b).abs() < 1e-7,
                "Zero blend_strength should be passthrough",
            );
        }
    }
}

#[test]
fn test_blend_voices_full_strength_modifies() {
    let v1 = sine_wave(200.0, 24000, 9600);
    let v2 = sine_wave(350.0, 24000, 9600);
    let originals = [v1.clone(), v2.clone()];
    let mut voices = vec![v1, v2];
    let config = EnsembleBlendConfig::new(1.0).unwrap();
    blend_voices(&mut voices, &config, 24000).unwrap();

    // At least one voice should have been modified.
    let any_modified = voices
        .iter()
        .zip(originals.iter())
        .any(|(v, o)| v.iter().zip(o.iter()).any(|(a, b)| (a - b).abs() > 1e-6));
    assert!(
        any_modified,
        "Full-strength blend should modify at least one voice"
    );
}

#[test]
fn test_blend_voices_preserves_rms() {
    let v1 = sine_wave(200.0, 24000, 9600);
    let v2 = sine_wave(250.0, 24000, 9600);
    let rms_before: Vec<f32> = vec![test_rms(&v1), test_rms(&v2)];
    let mut voices = vec![v1, v2];
    let config = EnsembleBlendConfig::new(0.5).unwrap();
    blend_voices(&mut voices, &config, 24000).unwrap();
    for (i, voice) in voices.iter().enumerate() {
        let rms_after = test_rms(voice);
        let ratio = rms_after / rms_before[i];
        // RMS should be within 50% of original.
        assert!(
            (0.5..=2.0).contains(&ratio),
            "Voice {i} RMS ratio {ratio} out of tolerance",
        );
    }
}

#[test]
fn test_blend_voices_output_all_finite() {
    let v1 = sine_wave(180.0, 24000, 4800);
    let v2 = sine_wave(220.0, 24000, 4800);
    let v3 = sine_wave(260.0, 24000, 4800);
    let mut voices = vec![v1, v2, v3];
    let config = EnsembleBlendConfig::default();
    blend_voices(&mut voices, &config, 24000).unwrap();
    for (vi, voice) in voices.iter().enumerate() {
        for (i, &s) in voice.iter().enumerate() {
            assert!(s.is_finite(), "Non-finite at voice {vi}, index {i}: {s}");
        }
    }
}

#[test]
fn test_blend_voices_invalid_config() {
    let mut voices = vec![vec![0.0; 100], vec![0.0; 100]];
    let config = EnsembleBlendConfig {
        blend_strength: -1.0,
        ..Default::default()
    };
    assert!(blend_voices(&mut voices, &config, 24000).is_err());
}

#[test]
fn test_blend_formant_only() {
    let v1 = sine_wave(200.0, 24000, 9600);
    let v2 = sine_wave(250.0, 24000, 9600);
    let mut voices = vec![v1, v2];
    let config = EnsembleBlendConfig::new(0.5)
        .unwrap()
        .with_harmonic_alignment(false);
    assert!(config.formant_preservation);
    assert!(!config.harmonic_alignment);
    blend_voices(&mut voices, &config, 24000).unwrap();
}

#[test]
fn test_blend_harmonic_only() {
    let v1 = sine_wave(200.0, 24000, 9600);
    let v2 = sine_wave(250.0, 24000, 9600);
    let mut voices = vec![v1, v2];
    let config = EnsembleBlendConfig::new(0.5)
        .unwrap()
        .with_formant_preservation(false);
    assert!(!config.formant_preservation);
    assert!(config.harmonic_alignment);
    blend_voices(&mut voices, &config, 24000).unwrap();
}

// ---------------------------------------------------------------------------
// Resample grain tests
// ---------------------------------------------------------------------------

#[test]
fn test_resample_grain_identity() {
    let grain: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
    let resampled = resample_grain(&grain, 100);
    assert_eq!(resampled.len(), 100);
    for (a, b) in grain.iter().zip(resampled.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_resample_grain_empty() {
    let grain: Vec<f32> = Vec::new();
    let resampled = resample_grain(&grain, 50);
    assert_eq!(resampled.len(), 50);
    for &s in &resampled {
        assert!((s - 0.0).abs() < 1e-6);
    }
}

#[test]
fn test_resample_grain_stretch() {
    let grain: Vec<f32> = (0..50).map(|i| (i as f32 * 0.1).sin()).collect();
    let resampled = resample_grain(&grain, 100);
    assert_eq!(resampled.len(), 100);
    for &s in &resampled {
        assert!(s.is_finite());
    }
}

#[test]
fn test_resample_grain_compress() {
    let grain: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
    let resampled = resample_grain(&grain, 50);
    assert_eq!(resampled.len(), 50);
    for &s in &resampled {
        assert!(s.is_finite());
    }
}
