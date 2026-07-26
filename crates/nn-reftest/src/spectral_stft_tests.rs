// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! STFT unit tests: config, validation, output shape, and result accessors.

use super::*;

// -------------------------------------------------------------------------
// StftConfig tests
// -------------------------------------------------------------------------

#[test]
fn test_stft_config_n_freqs() {
    let config = StftConfig {
        n_fft: 1024,
        hop_length: 256,
        window: WindowFn::Hann,
    };
    assert_eq!(config.n_freqs(), 513);
}

#[test]
fn test_stft_config_default() {
    let config = StftConfig::default();
    assert_eq!(config.n_fft, 1024);
    assert_eq!(config.hop_length, 256);
    assert_eq!(config.window, WindowFn::Hann);
}

// -------------------------------------------------------------------------
// STFT validation tests
// -------------------------------------------------------------------------

#[test]
fn test_stft_empty_signal_returns_error() {
    let config = StftConfig::default();
    let result = stft_magnitude(&[], &config);
    assert!(result.is_err());
}

#[test]
fn test_stft_zero_n_fft_returns_error() {
    let config = StftConfig {
        n_fft: 0,
        hop_length: 256,
        window: WindowFn::Hann,
    };
    let result = stft_magnitude(&[1.0; 2048], &config);
    assert!(result.is_err());
}

#[test]
fn test_stft_non_power_of_two_returns_error() {
    let config = StftConfig {
        n_fft: 1000,
        hop_length: 256,
        window: WindowFn::Hann,
    };
    let result = stft_magnitude(&[1.0; 2048], &config);
    assert!(result.is_err());
}

#[test]
fn test_stft_zero_hop_returns_error() {
    let config = StftConfig {
        n_fft: 1024,
        hop_length: 0,
        window: WindowFn::Hann,
    };
    let result = stft_magnitude(&[1.0; 2048], &config);
    assert!(result.is_err());
}

// -------------------------------------------------------------------------
// STFT basic output shape tests
// -------------------------------------------------------------------------

#[test]
fn test_stft_output_shape() {
    let config = StftConfig {
        n_fft: 256,
        hop_length: 128,
        window: WindowFn::Hann,
    };
    let signal = vec![0.0f32; 1024];
    let result = stft_magnitude(&signal, &config).expect("stft should succeed");
    // n_freqs = 256/2 + 1 = 129
    assert_eq!(result.n_freqs, 129);
    // n_frames = (1024 - 256) / 128 + 1 = 7
    assert_eq!(result.n_frames, 7);
    assert_eq!(result.data.len(), 129 * 7);
}

#[test]
fn test_stft_short_signal_one_frame() {
    let config = StftConfig {
        n_fft: 1024,
        hop_length: 256,
        window: WindowFn::Hann,
    };
    // Signal shorter than n_fft -- should produce 1 zero-padded frame.
    let signal = vec![1.0f32; 100];
    let result = stft_magnitude(&signal, &config).expect("stft should succeed");
    assert_eq!(result.n_frames, 1);
    assert_eq!(result.n_freqs, 513);
}

// -------------------------------------------------------------------------
// STFT result accessor test
// -------------------------------------------------------------------------

#[test]
fn test_stft_result_get() {
    let config = StftConfig {
        n_fft: 256,
        hop_length: 128,
        window: WindowFn::Hann,
    };
    let signal = sine_wave(440.0, 16000.0, 1024, 0.0);
    let result = stft_magnitude(&signal, &config).expect("stft should succeed");
    // Verify the accessor returns non-negative values.
    for freq in 0..result.n_freqs {
        for frame in 0..result.n_frames {
            assert!(result.get(freq, frame) >= 0.0);
        }
    }
}
