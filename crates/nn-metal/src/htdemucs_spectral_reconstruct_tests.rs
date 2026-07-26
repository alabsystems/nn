// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for spectral_reconstruct: spectral decoder output → iSTFT → waveform.
//!
//! Part of #961 D2 (HTDemucs iSTFT integration).

use super::helpers::spectral_reconstruct;
use crate::istft::{IstftBasis, IstftParams};

/// Small test parameters for fast unit testing.
/// n_fft=8, hop=2, normalized=true, center=true.
/// This gives n_bins=5, and we can test with stft_t=4 frames.
const SMALL_N_FFT: usize = 8;
const SMALL_HOP: usize = 2;
const SMALL_STFT_F: usize = SMALL_N_FFT / 2 + 1; // 5
const SMALL_STFT_T: usize = 4;
const SMALL_AUDIO_T: usize = 16;

fn make_istft_params(
    n_fft: usize,
    hop_length: usize,
    normalized: bool,
    center: bool,
) -> IstftParams {
    IstftParams::new(n_fft, hop_length, normalized, center).expect("valid params")
}

fn small_basis() -> IstftBasis {
    IstftBasis::new(make_istft_params(SMALL_N_FFT, SMALL_HOP, true, true)).expect("valid params")
}

/// Number of output channels (4 sources × 2 audio channels).
const OUTPUT_CHANNELS: usize = 8;
/// Number of spectral decoder channels (4 sources × 2 audio × 2 real/imag).
const SPECTRAL_CHANNELS: usize = 16;

#[test]
fn test_spectral_reconstruct_zero_input_produces_zero_output() {
    let basis = small_basis();
    let spectral_decoded = vec![0.0f32; SPECTRAL_CHANNELS * SMALL_STFT_F * SMALL_STFT_T];
    let result = spectral_reconstruct(
        &spectral_decoded,
        &basis,
        SMALL_STFT_F,
        SMALL_STFT_T,
        SMALL_AUDIO_T,
        0.0, // mean
        1.0, // std_val (no denorm effect when mean=0, std=1)
    )
    .expect("zero input should succeed");

    assert_eq!(result.len(), OUTPUT_CHANNELS * SMALL_AUDIO_T);
    // All zeros in → all zeros out (iSTFT of zero complex input is zero waveform).
    for &v in &result {
        assert!(v.abs() < 1e-6, "expected near-zero output, got {v}");
    }
}

#[test]
fn test_spectral_reconstruct_output_shape() {
    let basis = small_basis();
    let spectral_decoded = vec![0.01f32; SPECTRAL_CHANNELS * SMALL_STFT_F * SMALL_STFT_T];
    let result = spectral_reconstruct(
        &spectral_decoded,
        &basis,
        SMALL_STFT_F,
        SMALL_STFT_T,
        SMALL_AUDIO_T,
        0.0,
        1.0,
    )
    .expect("valid input should succeed");

    assert_eq!(
        result.len(),
        OUTPUT_CHANNELS * SMALL_AUDIO_T,
        "output should be [OUTPUT_CHANNELS, audio_t] flattened"
    );
}

#[test]
fn test_spectral_reconstruct_denormalization_applied() {
    // With mean=0 and std=2, denormalized = v * 2 + 0 = 2*v.
    // So the spectral output with all 0.5 values becomes 1.0 after denorm.
    let basis = small_basis();
    let spectral_decoded = vec![0.5f32; SPECTRAL_CHANNELS * SMALL_STFT_F * SMALL_STFT_T];

    let result_no_denorm = spectral_reconstruct(
        &spectral_decoded,
        &basis,
        SMALL_STFT_F,
        SMALL_STFT_T,
        SMALL_AUDIO_T,
        0.0,
        1.0, // identity denorm
    )
    .expect("no denorm should succeed");

    let result_with_denorm = spectral_reconstruct(
        &spectral_decoded,
        &basis,
        SMALL_STFT_F,
        SMALL_STFT_T,
        SMALL_AUDIO_T,
        0.0,
        2.0, // 2x denorm
    )
    .expect("with denorm should succeed");

    // With std=2, all input values are doubled before iSTFT.
    // iSTFT is linear, so output should be doubled.
    let max_ratio_diff = result_no_denorm
        .iter()
        .zip(result_with_denorm.iter())
        .filter(|(&a, _)| a.abs() > 1e-8) // skip near-zero
        .map(|(&a, &b)| (b / a - 2.0).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_ratio_diff < 0.01,
        "denormalized output should be ~2x the identity output, max ratio diff: {max_ratio_diff}"
    );
}

#[test]
fn test_spectral_reconstruct_wrong_length_rejected() {
    let basis = small_basis();
    let wrong_len = vec![0.0f32; SPECTRAL_CHANNELS * SMALL_STFT_F * SMALL_STFT_T + 1];
    let result = spectral_reconstruct(
        &wrong_len,
        &basis,
        SMALL_STFT_F,
        SMALL_STFT_T,
        SMALL_AUDIO_T,
        0.0,
        1.0,
    );
    assert!(result.is_err(), "wrong length should be rejected");
}

#[test]
fn test_spectral_reconstruct_nan_rejected() {
    let basis = small_basis();
    let mut data = vec![0.0f32; SPECTRAL_CHANNELS * SMALL_STFT_F * SMALL_STFT_T];
    data[0] = f32::NAN;
    let result = spectral_reconstruct(
        &data,
        &basis,
        SMALL_STFT_F,
        SMALL_STFT_T,
        SMALL_AUDIO_T,
        0.0,
        1.0,
    );
    assert!(result.is_err(), "NaN input should be rejected");
}

#[test]
fn test_spectral_reconstruct_channels_are_independent() {
    // Use center=false to avoid short-signal edge effects.
    let basis = IstftBasis::new(make_istft_params(SMALL_N_FFT, SMALL_HOP, true, false))
        .expect("valid params");

    let ft = SMALL_STFT_F * SMALL_STFT_T;
    let mut spectral_decoded = vec![0.0f32; SPECTRAL_CHANNELS * ft];

    // Source 0, channel 0, real part = channel index 0.
    // Set DC bin (f=0) to 1.0 for each frame to produce a constant signal.
    for v in spectral_decoded.iter_mut().take(SMALL_STFT_T) {
        *v = 1.0; // DC bin of real part
    }

    let result = spectral_reconstruct(
        &spectral_decoded,
        &basis,
        SMALL_STFT_F,
        SMALL_STFT_T,
        SMALL_AUDIO_T,
        0.0,
        1.0,
    )
    .expect("valid input should succeed");

    // Source 0, channel 0 (output channel 0) should have non-zero values.
    let ch0_energy: f32 = result[..SMALL_AUDIO_T].iter().map(|v| v * v).sum();

    // Source 1, channel 0 (output channel 2) should have zero values.
    let ch2_start = 2 * SMALL_AUDIO_T;
    let ch2_energy: f32 = result[ch2_start..ch2_start + SMALL_AUDIO_T]
        .iter()
        .map(|v| v * v)
        .sum();

    assert!(
        ch0_energy > 1e-6,
        "channel 0 should have non-zero output, got energy={ch0_energy}"
    );
    assert!(
        ch2_energy < 1e-10,
        "channel 2 should be zero (only channel 0 has input), got energy={ch2_energy}"
    );
}

#[test]
fn test_spectral_reconstruct_all_channels_finite() {
    // Moderate-amplitude input across all channels.
    let basis = small_basis();
    let spectral_decoded: Vec<f32> = (0..SPECTRAL_CHANNELS * SMALL_STFT_F * SMALL_STFT_T)
        .map(|i| 0.01 * ((i % 7) as f32 - 3.0))
        .collect();

    let result = spectral_reconstruct(
        &spectral_decoded,
        &basis,
        SMALL_STFT_F,
        SMALL_STFT_T,
        SMALL_AUDIO_T,
        0.5, // non-zero mean
        1.5, // non-unity std
    )
    .expect("moderate input should succeed");

    assert_eq!(result.len(), OUTPUT_CHANNELS * SMALL_AUDIO_T);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] = {v} is not finite");
    }
}
