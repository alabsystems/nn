// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for multi-resolution STFT loss metric.

use super::*;
use crate::test_audio_helpers::sine_wave;

#[test]
fn test_identical_signals_zero_loss() {
    let signal = sine_wave(440.0, 24000, 0.5);
    let config = MultiResStftConfig::default();
    let result = compute_multi_res_stft(&signal, &signal, 24000, &config).unwrap();
    assert!(
        result.value < 1e-6,
        "Identical signals should have near-zero loss, got {}",
        result.value
    );
    assert!(result.passed);
}

#[test]
fn test_different_frequencies_high_loss() {
    let sig_a = sine_wave(200.0, 24000, 0.5);
    let sig_b = sine_wave(2000.0, 24000, 0.5);
    let config = MultiResStftConfig::default();
    let result = compute_multi_res_stft(&sig_a, &sig_b, 24000, &config).unwrap();
    assert!(
        result.value > 0.1,
        "Different frequencies should have high loss, got {}",
        result.value
    );
}

#[test]
fn test_scaled_signal_moderate_loss() {
    let reference = sine_wave(440.0, 24000, 0.5);
    let candidate: Vec<f32> = reference.iter().map(|&x| x * 0.5).collect();
    let config = MultiResStftConfig::default();
    let result = compute_multi_res_stft(&candidate, &reference, 24000, &config).unwrap();
    // Halved amplitude = 6 dB difference, should be moderate loss.
    assert!(result.value > 0.01);
    assert!(result.value < 5.0);
}

#[test]
fn test_noisy_signal() {
    let reference = sine_wave(440.0, 24000, 0.5);
    let mut candidate = reference.clone();
    // Add small noise.
    for (i, sample) in candidate.iter_mut().enumerate() {
        *sample += 0.01 * ((i as f32 * 7.3).sin());
    }
    let config = MultiResStftConfig::default();
    let result = compute_multi_res_stft(&candidate, &reference, 24000, &config).unwrap();
    // Small noise should produce loss much less than very different signals.
    // Loss is (spectral_convergence + log_spectral_distance) / 2 averaged over
    // resolutions — LSD component can be > 1.0 even for small perturbations.
    assert!(
        result.value < 5.0,
        "Small noise should have moderate loss, got {}",
        result.value
    );
    // Should be less than the different-frequencies case.
    let sig_diff = sine_wave(2000.0, 24000, 0.5);
    let diff_result = compute_multi_res_stft(&sig_diff, &reference, 24000, &config).unwrap();
    assert!(
        result.value < diff_result.value,
        "Noisy signal loss ({}) should be less than different-frequency loss ({})",
        result.value,
        diff_result.value
    );
}

#[test]
fn test_custom_fft_sizes() {
    let signal = sine_wave(440.0, 24000, 0.5);
    let config = MultiResStftConfig {
        fft_sizes: vec![256, 512],
        max_loss: 2.0,
    };
    let result = compute_multi_res_stft(&signal, &signal, 24000, &config).unwrap();
    assert!(result.value < 1e-6);
}

#[test]
fn test_empty_input_error() {
    let config = MultiResStftConfig::default();
    let result = compute_multi_res_stft(&[], &[1.0], 24000, &config);
    assert!(result.is_err());
}

#[test]
fn test_length_mismatch_error() {
    let config = MultiResStftConfig::default();
    let result = compute_multi_res_stft(&[1.0; 4096], &[1.0; 2048], 24000, &config);
    assert!(result.is_err());
}

#[test]
fn test_non_power_of_two_fft_error() {
    let signal = sine_wave(440.0, 24000, 0.5);
    let config = MultiResStftConfig {
        fft_sizes: vec![500],
        max_loss: 1.0,
    };
    let result = compute_multi_res_stft(&signal, &signal, 24000, &config);
    assert!(result.is_err());
}

#[test]
fn test_zero_sample_rate_error() {
    let signal = sine_wave(440.0, 24000, 0.5);
    let config = MultiResStftConfig::default();
    let result = compute_multi_res_stft(&signal, &signal, 0, &config);
    assert!(result.is_err());
}

#[test]
fn test_threshold_pass_fail() {
    let reference = sine_wave(440.0, 24000, 0.5);
    let sig_b = sine_wave(880.0, 24000, 0.5);
    let strict_config = MultiResStftConfig {
        fft_sizes: DEFAULT_FFT_SIZES.to_vec(),
        max_loss: 0.001, // Very strict.
    };
    let result = compute_multi_res_stft(&sig_b, &reference, 24000, &strict_config).unwrap();
    assert!(
        !result.passed,
        "Different signals should fail strict threshold"
    );
}

#[test]
fn test_audio_shorter_than_smallest_fft() {
    let short_signal: Vec<f32> = vec![0.1; 100]; // Only 100 samples, less than 512.
    let config = MultiResStftConfig::default();
    let result = compute_multi_res_stft(&short_signal, &short_signal, 24000, &config);
    assert!(result.is_err());
}
