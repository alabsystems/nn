// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Signal processing tests for STFT, iSTFT, and Kokoro-specific audio paths (#4186).
//!
//! Covers: STFT error paths, mel filterbank shape validation, StftParams/IstftParams
//! construction edge cases, Kokoro iSTFT error handling, power spectrum properties,
//! and windowing function numerical properties.

use std::f32::consts::PI;

use crate::istft::{IstftBasis, IstftError, IstftParams};
use crate::kokoro_istft::{kokoro_istft, KokoroIstftError, KokoroIstftParams};
use crate::kokoro_tts::{KOKORO_HOP_LENGTH, KOKORO_N_FFT};
use crate::stft::{compute_stft_magnitude, StftError, StftParams};

// ===========================================================================
// 1. StftParams construction and derived fields
// ===========================================================================

#[test]
fn test_stft_params_new_derives_n_freqs() {
    let params = StftParams::new(256, 128);
    assert_eq!(params.n_freqs, 129, "n_freqs = n_fft/2 + 1");
}

#[test]
fn test_stft_params_new_derives_pad_right() {
    let params = StftParams::new(256, 128);
    assert_eq!(params.pad_right, 64, "pad_right = n_fft/4");
}

#[test]
fn test_stft_params_default_is_silero_vad() {
    let params = StftParams::default();
    assert_eq!(params.n_fft, 256);
    assert_eq!(params.hop_length, 128);
    assert_eq!(params.n_freqs, 129);
    assert_eq!(params.pad_right, 64);
}

#[test]
fn test_stft_params_new_small_fft() {
    let params = StftParams::new(8, 4);
    assert_eq!(params.n_freqs, 5); // 8/2 + 1
    assert_eq!(params.pad_right, 2); // 8/4
}

// ===========================================================================
// 2. compute_stft_magnitude error handling
// ===========================================================================

#[test]
fn test_stft_basis_size_mismatch() {
    let params = StftParams::default(); // n_fft=256
    let audio = vec![0.0f32; 576];
    let bad_basis = vec![0.0f32; 100]; // wrong size
    let err = compute_stft_magnitude(&audio, &bad_basis, &params).unwrap_err();
    assert!(matches!(err, StftError::BasisSizeMismatch { .. }));
}

#[test]
fn test_stft_audio_too_short_for_padding() {
    let params = StftParams::new(256, 128); // pad_right = 64, needs audio >= 66
    let audio = vec![0.0f32; 2]; // way too short
    let basis = vec![0.0f32; (256 + 2) * 256];
    let err = compute_stft_magnitude(&audio, &basis, &params).unwrap_err();
    assert!(matches!(err, StftError::AudioTooShortForPadding { .. }));
}

#[test]
fn test_stft_n_freqs_mismatch() {
    // Construct params with inconsistent n_freqs.
    let params = StftParams {
        n_fft: 256,
        hop_length: 128,
        n_freqs: 100, // wrong: should be 129
        pad_right: 64,
    };
    let audio = vec![0.0f32; 576];
    let basis = vec![0.0f32; (256 + 2) * 256];
    let err = compute_stft_magnitude(&audio, &basis, &params).unwrap_err();
    assert!(matches!(err, StftError::FreqsMismatch { .. }));
}

#[test]
fn test_stft_zero_input_produces_zero_magnitude() {
    let params = StftParams::new(8, 4);
    let audio = vec![0.0f32; 100];
    let basis_len = (8 + 2) * 8;
    let basis = vec![0.0f32; basis_len];
    let mag = compute_stft_magnitude(&audio, &basis, &params).unwrap();
    for v in &mag {
        assert!(
            *v == 0.0 || v.abs() < 1e-10,
            "zero input should produce zero magnitude, got {v}"
        );
    }
}

// ===========================================================================
// 3. IstftParams validation
// ===========================================================================

#[test]
fn test_istft_params_various_valid_sizes() {
    for &(n_fft, hop) in &[(4, 2), (8, 2), (16, 4), (64, 16), (128, 32), (4096, 1024)] {
        let params = IstftParams::new(n_fft, hop, false, false);
        assert!(params.is_ok(), "n_fft={n_fft}, hop={hop} should be valid");
    }
}

#[test]
fn test_istft_params_rejects_all_odd_values() {
    for odd in [1, 3, 5, 7, 9, 11, 13, 15, 127, 255] {
        let result = IstftParams::new(odd, 4, false, false);
        assert!(result.is_err(), "odd n_fft={odd} should be rejected");
    }
}

// ===========================================================================
// 4. Hann window numerical properties
// ===========================================================================

#[test]
fn test_hann_window_sum_of_squares() {
    // Sum of squared Hann window values = n_fft * 3/8 (analytical result)
    let n_fft = 64;
    let params = IstftParams::new(n_fft, 16, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let window = basis.window();
    let sum_sq: f32 = window.iter().map(|w| w * w).sum();
    let expected = n_fft as f32 * 3.0 / 8.0;
    assert!(
        (sum_sq - expected).abs() < 0.1,
        "sum of squared Hann window = {sum_sq}, expected ~{expected}"
    );
}

#[test]
fn test_hann_window_non_negative() {
    for &n_fft in &[8, 16, 20, 32, 64, 128] {
        let params = IstftParams::new(n_fft, n_fft / 4, false, false).unwrap();
        let basis = IstftBasis::new(params).unwrap();
        for (k, &w) in basis.window().iter().enumerate() {
            assert!(w >= 0.0, "Hann window must be non-negative: w[{k}] = {w}");
        }
    }
}

#[test]
fn test_hann_window_length_matches_n_fft() {
    for &n_fft in &[8, 20, 64, 128] {
        let params = IstftParams::new(n_fft, n_fft / 4, false, false).unwrap();
        let basis = IstftBasis::new(params).unwrap();
        assert_eq!(basis.window().len(), n_fft);
    }
}

// ===========================================================================
// 5. IstftBasis DFT matrix properties
// ===========================================================================

#[test]
fn test_istft_basis_n_bins() {
    let params = IstftParams::new(20, 5, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    assert_eq!(basis.n_bins(), 11); // 20/2 + 1
}

#[test]
fn test_istft_basis_cos_sin_length() {
    let n_fft = 20;
    let params = IstftParams::new(n_fft, 5, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let n_bins = n_fft / 2 + 1;
    assert_eq!(basis.cos_basis().len(), n_bins * n_fft);
    assert_eq!(basis.sin_basis().len(), n_bins * n_fft);
}

#[test]
fn test_istft_basis_cos_dc_row_all_ones() {
    // cos(2*pi*0*k/N) = 1.0 for all k, so first row of cos_basis should be all 1.0
    let n_fft = 16;
    let params = IstftParams::new(n_fft, 4, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    for k in 0..n_fft {
        let val = basis.cos_basis()[k]; // f=0, k
        assert!(
            (val - 1.0).abs() < 1e-6,
            "cos_basis[0, {k}] = {val}, expected 1.0"
        );
    }
}

#[test]
fn test_istft_basis_sin_dc_row_all_zeros() {
    // sin(2*pi*0*k/N) = 0.0 for all k
    let n_fft = 16;
    let params = IstftParams::new(n_fft, 4, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    for k in 0..n_fft {
        let val = basis.sin_basis()[k]; // f=0, k
        assert!(val.abs() < 1e-6, "sin_basis[0, {k}] = {val}, expected 0.0");
    }
}

// ===========================================================================
// 6. Kokoro iSTFT error paths
// ===========================================================================

#[test]
fn test_kokoro_istft_mismatched_real_imag_length() {
    let params = KokoroIstftParams {
        n_fft: 20,
        hop_length: 5,
    };
    let n_bins = 11;
    let real = vec![0.0f32; n_bins * 2]; // 2 frames
    let imag = vec![0.0f32; n_bins * 3]; // 3 frames -- mismatch
    let err = kokoro_istft(&params, &real, &imag, 2, 30).unwrap_err();
    assert!(matches!(err, KokoroIstftError::ShapeMismatch { .. }));
}

#[test]
fn test_kokoro_istft_inf_input_rejected() {
    let params = KokoroIstftParams {
        n_fft: 20,
        hop_length: 5,
    };
    let n_bins = 11;
    let mut real = vec![0.0f32; n_bins];
    real[5] = f32::INFINITY;
    let imag = vec![0.0f32; n_bins];
    let err = kokoro_istft(&params, &real, &imag, 1, 20).unwrap_err();
    assert!(matches!(err, KokoroIstftError::NonFiniteInput));
}

#[test]
fn test_kokoro_istft_neg_inf_input_rejected() {
    let params = KokoroIstftParams {
        n_fft: 20,
        hop_length: 5,
    };
    let n_bins = 11;
    let mut real = vec![0.0f32; n_bins];
    real[3] = f32::NEG_INFINITY;
    let imag = vec![0.0f32; n_bins];
    let err = kokoro_istft(&params, &real, &imag, 1, 20).unwrap_err();
    assert!(matches!(err, KokoroIstftError::NonFiniteInput));
}

// ===========================================================================
// 7. STFT magnitude output shape
// ===========================================================================

#[test]
fn test_stft_magnitude_output_shape() {
    // For Silero VAD: audio=576, n_fft=256, hop=128, pad_right=64
    // padded_len = 576 + 64 = 640
    // n_frames = (640 - 256) / 128 + 1 = 384/128 + 1 = 4
    // output length = n_freqs * n_frames = 129 * 4 = 516
    let params = StftParams::default();
    let audio_len = 576;
    let padded_len = audio_len + params.pad_right;
    let n_frames = (padded_len - params.n_fft) / params.hop_length + 1;
    assert_eq!(n_frames, 4, "Silero VAD should produce 4 STFT frames");

    let expected_output_len = params.n_freqs * n_frames;
    assert_eq!(expected_output_len, 516);

    // Build a valid STFT basis and compute.
    let basis_len = (params.n_fft + 2) * params.n_fft;
    // Use identity-like basis: each filter is an impulse at its center.
    let mut basis = vec![0.0f32; basis_len];
    for f in 0..(params.n_fft + 2) {
        // Put a small value at position 0 of each filter so we get some output.
        basis[f * params.n_fft] = 1.0;
    }
    let audio: Vec<f32> = (0..audio_len).map(|i| (i as f32 * 0.01).sin()).collect();
    let mag = compute_stft_magnitude(&audio, &basis, &params).unwrap();
    assert_eq!(mag.len(), expected_output_len);
}

// ===========================================================================
// 8. Kokoro iSTFT constants match config
// ===========================================================================

#[test]
fn test_kokoro_istft_constants_match_config() {
    let cfg = crate::kokoro_tts::KokoroConfig::default();
    assert_eq!(KOKORO_N_FFT, cfg.n_fft);
    assert_eq!(KOKORO_HOP_LENGTH, cfg.n_fft / 4);
}

// ===========================================================================
// 9. Power spectrum properties
// ===========================================================================

#[test]
fn test_stft_magnitude_pure_tone_has_peak() {
    // A pure cosine at exactly bin k should produce a peak at bin k.
    let n_fft = 32;
    let hop = 8;
    let params = StftParams::new(n_fft, hop);

    // Build DFT basis for the STFT. Per compute_stft_magnitude's contract the
    // layout is [cos rows for bins 0..n_freqs][sin rows for bins 0..n_freqs],
    // NOT all-cosine rows: the first n_freqs rows are the real (cos) basis and
    // the next n_freqs rows are the imaginary (-sin) basis. Building every row
    // as a cosine made bin k's sin slot alias onto cos(2*pi*(n_fft-k)), leaking
    // a bin-5 tone into bin 10's magnitude.
    let n_filters = n_fft + 2;
    let n_freqs = n_fft / 2 + 1;
    let basis_len = n_filters * n_fft;
    let mut basis = vec![0.0f32; basis_len];
    for f in 0..n_freqs {
        for k in 0..n_fft {
            let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
            basis[f * n_fft + k] = angle.cos(); // real (cosine) basis
            basis[(n_freqs + f) * n_fft + k] = -angle.sin(); // imag (-sine) basis
        }
    }

    // Generate audio: cosine at bin 5.
    let signal_len = 200;
    let target_bin = 5usize;
    let audio: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * target_bin as f32 * i as f32 / n_fft as f32).cos())
        .collect();

    let mag = compute_stft_magnitude(&audio, &basis, &params).unwrap();
    let padded_len = signal_len + params.pad_right;
    let n_frames = (padded_len - n_fft) / hop + 1;

    // Check middle frame for peak at target_bin.
    let t = n_frames / 2;
    let mut max_mag = 0.0f32;
    let mut max_bin = 0usize;
    for f in 0..params.n_freqs {
        let val = mag[f * n_frames + t];
        if val > max_mag {
            max_mag = val;
            max_bin = f;
        }
    }
    assert_eq!(
        max_bin, target_bin,
        "peak should be at bin {target_bin}, got bin {max_bin}"
    );
}

// ===========================================================================
// 10. IstftBasis error paths
// ===========================================================================

#[test]
fn test_istft_basis_rejects_zero_nfft() {
    let params = IstftParams {
        n_fft: 0,
        hop_length: 4,
        normalized: false,
        center: false,
    };
    match IstftBasis::new(params) {
        Err(IstftError::OddNfft { n_fft: 0 }) => {} // expected
        Err(other) => panic!("expected OddNfft{{n_fft: 0}}, got {other}"),
        Ok(_) => panic!("expected error for n_fft=0, got Ok"),
    }
}

#[test]
fn test_istft_basis_rejects_zero_hop() {
    let params = IstftParams {
        n_fft: 16,
        hop_length: 0,
        normalized: false,
        center: false,
    };
    match IstftBasis::new(params) {
        Err(IstftError::ZeroHopLength) => {} // expected
        Err(other) => panic!("expected ZeroHopLength, got {other}"),
        Ok(_) => panic!("expected error for hop_length=0, got Ok"),
    }
}

#[test]
fn test_istft_non_finite_real_rejected() {
    let params = IstftParams::new(8, 2, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let n_bins = 5; // 8/2 + 1
    let mut real = vec![0.0f32; n_bins * 2];
    real[0] = f32::NAN;
    let imag = vec![0.0f32; n_bins * 2];
    let err = basis.istft(&real, &imag, 2, 10).unwrap_err();
    assert!(matches!(err, IstftError::NonFiniteInput));
}

#[test]
fn test_istft_length_mismatch_rejected() {
    let params = IstftParams::new(8, 2, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let real = vec![0.0f32; 10];
    let imag = vec![0.0f32; 15]; // different length
    let err = basis.istft(&real, &imag, 2, 10).unwrap_err();
    assert!(matches!(err, IstftError::LengthMismatch { .. }));
}
