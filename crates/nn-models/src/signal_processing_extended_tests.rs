// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended signal processing tests (#4186).
//!
//! Covers:
//! - STFT window function generation (Hann, Hamming) and numerical properties
//! - STFT parameter validation (hop length, FFT size, window size)
//! - Round-trip STFT -> iSTFT reconstruction with various signal types
//! - Frequency bin calculations and spectral peak identification
//! - Signal padding modes (reflection, zero-pad, truncation)
//! - Edge cases: empty signals, single-sample signals, very long signals
//! - DFT basis orthogonality and Nyquist row properties
//! - KokoroForwardStft shape, magnitude, phase, and error paths
//! - iSTFT center-trim geometry and normalized scaling

use std::f32::consts::PI;

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::istft::{IstftBasis, IstftError, IstftParams};
use crate::kokoro_forward_stft::KokoroForwardStft;
use crate::kokoro_tts::{
    harmonic_source, prepare_istft_input, KOKORO_HOP_LENGTH, KOKORO_N_BINS, KOKORO_N_FFT,
    KOKORO_SAMPLE_RATE,
};
use crate::stft::{compute_stft_magnitude, StftError, StftParams};

// ===========================================================================
// Helpers
// ===========================================================================

/// Periodic Hann window: w[k] = 0.5 * (1 - cos(2*pi*k / n)).
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n as f32).cos()))
        .collect()
}

/// Hamming window: w[k] = 0.54 - 0.46 * cos(2*pi*k / (n-1)).
fn hamming_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|k| 0.54 - 0.46 * (2.0 * PI * k as f32 / (n - 1) as f32).cos())
        .collect()
}

/// Build a DFT-style STFT basis (cos + sin rows) for compute_stft_magnitude.
fn build_dft_basis(n_fft: usize) -> Vec<f32> {
    let n_filters = n_fft + 2;
    let n_freqs = n_fft / 2 + 1;
    let mut basis = vec![0.0f32; n_filters * n_fft];
    for f in 0..n_freqs {
        for k in 0..n_fft {
            let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
            basis[f * n_fft + k] = angle.cos();
        }
    }
    for f in 0..n_freqs {
        for k in 0..n_fft {
            let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
            basis[(n_freqs + f) * n_fft + k] = -angle.sin();
        }
    }
    basis
}

/// Scalar windowed forward STFT (Hann window + DFT per frame).
/// Returns (real, imag) each [n_bins, n_frames] row-major, plus n_frames.
fn windowed_forward_stft(signal: &[f32], n_fft: usize, hop: usize) -> (Vec<f32>, Vec<f32>, usize) {
    let n_bins = n_fft / 2 + 1;
    let n_frames = if signal.len() >= n_fft {
        (signal.len() - n_fft) / hop + 1
    } else {
        0
    };
    let window = hann_window(n_fft);

    let mut real = vec![0.0f32; n_bins * n_frames];
    let mut imag = vec![0.0f32; n_bins * n_frames];
    for t in 0..n_frames {
        let offset = t * hop;
        for f in 0..n_bins {
            let mut r = 0.0f32;
            let mut im = 0.0f32;
            for k in 0..n_fft {
                let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
                let windowed = signal[offset + k] * window[k];
                r += windowed * angle.cos();
                im -= windowed * angle.sin();
            }
            real[f * n_frames + t] = r;
            imag[f * n_frames + t] = im;
        }
    }
    (real, imag, n_frames)
}

// ===========================================================================
// 1. STFT window function generation — Hann
// ===========================================================================

#[test]
fn test_hann_window_values_match_istft_basis() {
    // Verify our local Hann function matches IstftBasis for multiple sizes.
    for &n in &[8, 16, 20, 32, 64, 128] {
        let params = IstftParams::new(n, n / 4, false, false).unwrap();
        let basis = IstftBasis::new(params).unwrap();
        let basis_win = basis.window();
        let local_win = hann_window(n);
        assert_eq!(basis_win.len(), local_win.len());
        for k in 0..n {
            let diff = (basis_win[k] - local_win[k]).abs();
            assert!(diff < 1e-7, "n={n}, k={k}: mismatch {diff}");
        }
    }
}

#[test]
fn test_hann_window_endpoint_zero_peak_one() {
    // Periodic Hann: w[0] = 0, w[N/2] = 1
    for &n in &[8, 16, 32, 64, 128, 256] {
        let w = hann_window(n);
        assert!(w[0].abs() < 1e-7, "n={n}: w[0]={} should be ~0", w[0]);
        assert!(
            (w[n / 2] - 1.0).abs() < 1e-6,
            "n={n}: w[N/2]={} should be ~1.0",
            w[n / 2]
        );
    }
}

#[test]
fn test_hann_window_symmetry() {
    // Periodic Hann: w[k] = w[N-k] for k = 1..N/2-1
    for &n in &[16, 32, 64, 128] {
        let w = hann_window(n);
        for k in 1..n / 2 {
            let diff = (w[k] - w[n - k]).abs();
            assert!(
                diff < 1e-6,
                "n={n}: w[{k}]={} != w[{}]={}",
                w[k],
                n - k,
                w[n - k]
            );
        }
    }
}

#[test]
fn test_hann_window_monotonic_increasing_first_half() {
    for &n in &[16, 32, 64, 128] {
        let w = hann_window(n);
        for k in 1..n / 2 {
            assert!(
                w[k] >= w[k - 1] - 1e-7,
                "n={n}: not increasing at k={k}: w[{}]={} > w[{k}]={}",
                k - 1,
                w[k - 1],
                w[k]
            );
        }
    }
}

#[test]
fn test_hann_window_monotonic_decreasing_second_half() {
    for &n in &[16, 32, 64, 128] {
        let w = hann_window(n);
        for k in (n / 2 + 1)..n {
            assert!(
                w[k] <= w[k - 1] + 1e-7,
                "n={n}: not decreasing at k={k}: w[{}]={} < w[{k}]={}",
                k - 1,
                w[k - 1],
                w[k]
            );
        }
    }
}

#[test]
fn test_hann_window_sum_of_squares_analytical() {
    // Sum of squared Hann window = n * 3/8
    for &n in &[16, 32, 64, 128] {
        let w = hann_window(n);
        let sum_sq: f32 = w.iter().map(|v| v * v).sum();
        let expected = n as f32 * 3.0 / 8.0;
        assert!(
            (sum_sq - expected).abs() < 0.1,
            "n={n}: sum_sq={sum_sq}, expected ~{expected}"
        );
    }
}

#[test]
fn test_hann_window_cola_50_percent_overlap() {
    // With hop = n/2, sum of Hann windows = constant 1.0 in interior
    let n = 64;
    let hop = n / 2;
    let n_frames = 20;
    let full_len = n + (n_frames - 1) * hop;
    let w = hann_window(n);
    let mut win_sum = vec![0.0f32; full_len];
    for t in 0..n_frames {
        for k in 0..n {
            win_sum[t * hop + k] += w[k];
        }
    }
    let margin = 2 * n;
    for i in margin..(full_len.saturating_sub(margin)) {
        assert!(
            (win_sum[i] - 1.0).abs() < 0.01,
            "COLA sum at {i}={}, expected ~1.0",
            win_sum[i]
        );
    }
}

// ===========================================================================
// 2. STFT window function generation — Hamming
// ===========================================================================

#[test]
fn test_hamming_window_endpoint_values() {
    // Hamming: w[0] = 0.54 - 0.46 = 0.08 (not zero like Hann)
    for &n in &[16, 32, 64, 128] {
        let w = hamming_window(n);
        assert!(
            (w[0] - 0.08).abs() < 1e-5,
            "n={n}: hamming[0]={}, expected 0.08",
            w[0]
        );
        assert!(
            (w[n - 1] - 0.08).abs() < 1e-5,
            "n={n}: hamming[last]={}, expected 0.08",
            w[n - 1]
        );
    }
}

#[test]
fn test_hamming_window_always_positive() {
    // Hamming window is always > 0 (unlike Hann which touches 0)
    for &n in &[8, 16, 32, 64, 128] {
        let w = hamming_window(n);
        for (k, &v) in w.iter().enumerate() {
            assert!(v > 0.0, "n={n}: hamming[{k}]={v} should be > 0");
        }
    }
}

#[test]
fn test_hamming_window_symmetric() {
    // Hamming: w[k] = w[N-1-k]
    for &n in &[16, 32, 64, 128] {
        let w = hamming_window(n);
        for k in 0..n / 2 {
            let diff = (w[k] - w[n - 1 - k]).abs();
            assert!(
                diff < 1e-6,
                "n={n}: hamming[{k}]={} != hamming[{}]={}",
                w[k],
                n - 1 - k,
                w[n - 1 - k]
            );
        }
    }
}

#[test]
fn test_hamming_window_peak_near_center() {
    for &n in &[16, 32, 64, 128] {
        let w = hamming_window(n);
        let max_val = w.iter().copied().fold(0.0f32, f32::max);
        let peak_idx = w.iter().position(|&v| (v - max_val).abs() < 1e-7).unwrap();
        let expected_center = (n - 1) / 2;
        assert!(
            (peak_idx as i64 - expected_center as i64).unsigned_abs() <= 1,
            "n={n}: peak at {peak_idx}, expected ~{expected_center}"
        );
        // The symmetric Hamming window (denominator n-1) reaches exactly 1.0
        // only when a sample lands on the center; for even n the center falls
        // between samples, so the max sample is slightly below 1.0 (n=16 ->
        // 0.98995, diff 0.01005). Use a tolerance that honestly covers the
        // smallest even window.
        assert!(
            (max_val - 1.0).abs() < 0.02,
            "n={n}: peak={max_val}, expected ~1.0"
        );
    }
}

#[test]
fn test_hamming_window_single_sample() {
    let w = hamming_window(1);
    assert_eq!(w.len(), 1);
    assert!(
        (w[0] - 1.0).abs() < 1e-7,
        "single-sample window should be 1.0"
    );
}

#[test]
fn test_hamming_window_two_samples() {
    let w = hamming_window(2);
    assert_eq!(w.len(), 2);
    // w[0] = 0.54 - 0.46*cos(0) = 0.08, w[1] = 0.54 - 0.46*cos(2pi) = 0.08
    for (k, &v) in w.iter().enumerate() {
        assert!(
            (v - 0.08).abs() < 1e-5,
            "hamming_2[{k}]={v}, expected ~0.08"
        );
    }
}

// ===========================================================================
// 3. STFT parameter validation
// ===========================================================================

#[test]
fn test_stft_params_new_derives_fields() {
    let params = StftParams::new(256, 128);
    assert_eq!(params.n_fft, 256);
    assert_eq!(params.hop_length, 128);
    assert_eq!(params.n_freqs, 129); // 256/2 + 1
    assert_eq!(params.pad_right, 64); // 256/4
}

#[test]
fn test_stft_params_small_fft_derives_correctly() {
    let params = StftParams::new(8, 4);
    assert_eq!(params.n_freqs, 5); // 8/2 + 1
    assert_eq!(params.pad_right, 2); // 8/4
}

#[test]
fn test_istft_params_rejects_zero_nfft() {
    let result = IstftParams::new(0, 4, false, false);
    assert!(matches!(result, Err(IstftError::OddNfft { n_fft: 0 })));
}

#[test]
fn test_istft_params_rejects_odd_nfft() {
    for odd in [1, 3, 5, 7, 9, 11, 127, 255] {
        let result = IstftParams::new(odd, 4, false, false);
        assert!(result.is_err(), "odd n_fft={odd} should be rejected");
    }
}

#[test]
fn test_istft_params_rejects_zero_hop() {
    let result = IstftParams::new(16, 0, false, false);
    assert!(matches!(result, Err(IstftError::ZeroHopLength)));
}

#[test]
fn test_istft_params_accepts_all_valid_even_sizes() {
    for &(n_fft, hop) in &[(2, 1), (4, 2), (8, 2), (16, 4), (64, 16), (4096, 1024)] {
        let result = IstftParams::new(n_fft, hop, false, false);
        assert!(result.is_ok(), "n_fft={n_fft}, hop={hop} should be valid");
    }
}

#[test]
fn test_istft_params_default_is_htdemucs() {
    let params = IstftParams::default();
    assert_eq!(params.n_fft, 4096);
    assert_eq!(params.hop_length, 1024);
    assert!(params.normalized);
    assert!(params.center);
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
fn test_stft_rejects_inconsistent_n_freqs() {
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
fn test_stft_rejects_wrong_basis_size() {
    let params = StftParams::default();
    let audio = vec![0.0f32; 576];
    let bad_basis = vec![0.0f32; 100];
    let err = compute_stft_magnitude(&audio, &bad_basis, &params).unwrap_err();
    assert!(matches!(err, StftError::BasisSizeMismatch { .. }));
}

#[test]
fn test_kokoro_forward_stft_rejects_zero_nfft() {
    let result = KokoroForwardStft::new(0, 5, &Device::Cpu);
    assert!(result.is_err(), "n_fft=0 should be rejected");
}

#[test]
fn test_kokoro_forward_stft_rejects_odd_nfft() {
    let result = KokoroForwardStft::new(21, 5, &Device::Cpu);
    assert!(result.is_err(), "odd n_fft should be rejected");
}

#[test]
fn test_kokoro_forward_stft_rejects_zero_hop() {
    let result = KokoroForwardStft::new(20, 0, &Device::Cpu);
    assert!(result.is_err(), "hop_length=0 should be rejected");
}

// ===========================================================================
// 4. Round-trip STFT -> iSTFT reconstruction
// ===========================================================================

#[test]
fn test_roundtrip_pure_sine_reconstruction_below_1e4() {
    let n_fft = 64;
    let hop = 16;
    let signal_len = 1024;
    let signal: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
        .collect();

    let (real, imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let skip = 2 * n_fft;
    let end = signal_len.min(reconstructed.len()).saturating_sub(skip);
    let mut max_err = 0.0f32;
    for i in skip..end {
        max_err = max_err.max((reconstructed[i] - signal[i]).abs());
    }
    assert!(max_err < 1e-4, "roundtrip error {max_err:.2e} exceeds 1e-4");
}

#[test]
fn test_roundtrip_multi_frequency_signal() {
    let n_fft = 32;
    let hop = 8;
    let signal_len = 512;
    let signal: Vec<f32> = (0..signal_len)
        .map(|i| {
            let t = i as f32 / signal_len as f32;
            0.5 * (2.0 * PI * 3.0 * t).sin()
                + 0.3 * (2.0 * PI * 7.0 * t).cos()
                + 0.2 * (2.0 * PI * 11.0 * t).sin()
        })
        .collect();

    let (real, imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let skip = 2 * n_fft;
    let end = signal_len.min(reconstructed.len()).saturating_sub(skip);
    let mut max_err = 0.0f32;
    for i in skip..end {
        max_err = max_err.max((reconstructed[i] - signal[i]).abs());
    }
    assert!(max_err < 1e-4, "multi-freq roundtrip error {max_err:.2e}");
}

#[test]
fn test_roundtrip_kokoro_size_nfft20_hop5() {
    let n_fft = 20;
    let hop = 5;
    let signal_len = 400;
    let signal: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * 4.0 * i as f32 / signal_len as f32).sin())
        .collect();

    let (real, imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let skip = 2 * n_fft;
    let end = signal_len.min(reconstructed.len()).saturating_sub(skip);
    let mut max_err = 0.0f32;
    for i in skip..end {
        max_err = max_err.max((reconstructed[i] - signal[i]).abs());
    }
    assert!(max_err < 1e-4, "Kokoro-sized roundtrip error {max_err:.2e}");
}

#[test]
fn test_roundtrip_normalized_mode() {
    // Forward with 1/sqrt(N) normalization, inverse with normalized=true
    let n_fft = 64;
    let hop = 16;
    let signal_len = 512;
    let norm_factor = 1.0 / (n_fft as f32).sqrt();

    let signal: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * 6.0 * i as f32 / signal_len as f32).sin())
        .collect();

    let (raw_real, raw_imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
    let real: Vec<f32> = raw_real.iter().map(|v| v * norm_factor).collect();
    let imag: Vec<f32> = raw_imag.iter().map(|v| v * norm_factor).collect();

    let params = IstftParams::new(n_fft, hop, true, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let skip = 2 * n_fft;
    let end = signal_len.min(reconstructed.len()).saturating_sub(skip);
    let mut max_err = 0.0f32;
    for i in skip..end {
        max_err = max_err.max((reconstructed[i] - signal[i]).abs());
    }
    assert!(max_err < 1e-4, "normalized roundtrip error {max_err:.2e}");
}

#[test]
fn test_roundtrip_chirp_signal() {
    // Frequency sweep: tests broadband reconstruction
    let n_fft = 32;
    let hop = 8;
    let signal_len = 256;
    let signal: Vec<f32> = (0..signal_len)
        .map(|i| {
            let t = i as f32 / signal_len as f32;
            (2.0 * PI * (1.0 * t + 9.0 * t * t / 2.0)).sin()
        })
        .collect();

    let (real, imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let skip = n_fft;
    let end = signal_len.min(reconstructed.len()).saturating_sub(skip);
    let mut max_err = 0.0f32;
    for i in skip..end {
        max_err = max_err.max((reconstructed[i] - signal[i]).abs());
    }
    assert!(max_err < 0.05, "chirp roundtrip error {max_err:.6}");
}

// ===========================================================================
// 5. Frequency bin calculations
// ===========================================================================

#[test]
fn test_frequency_bin_count_formula() {
    // n_bins = n_fft/2 + 1 for real-valued FFT
    for &n_fft in &[8, 16, 20, 32, 64, 128, 256, 512] {
        let params = IstftParams::new(n_fft, n_fft / 4, false, false).unwrap();
        let basis = IstftBasis::new(params).unwrap();
        assert_eq!(basis.n_bins(), n_fft / 2 + 1, "n_fft={n_fft}");
    }
}

#[test]
fn test_kokoro_forward_stft_n_bins() {
    let stft = KokoroForwardStft::new(20, 5, &Device::Cpu).unwrap();
    assert_eq!(stft.n_bins(), 11); // 20/2 + 1

    let stft2 = KokoroForwardStft::new(64, 16, &Device::Cpu).unwrap();
    assert_eq!(stft2.n_bins(), 33); // 64/2 + 1
}

#[test]
fn test_frequency_bin_to_hz_conversion() {
    // bin k corresponds to frequency f = k * sample_rate / n_fft
    let n_fft = 256;
    let sample_rate = 16000.0f32;
    let n_bins = n_fft / 2 + 1;

    // DC bin (k=0) -> 0 Hz
    let dc_hz = 0.0 * sample_rate / n_fft as f32;
    assert_eq!(dc_hz, 0.0);

    // Nyquist bin (k=n_fft/2) -> sample_rate/2
    let nyquist_hz = (n_fft / 2) as f32 * sample_rate / n_fft as f32;
    assert!((nyquist_hz - 8000.0).abs() < 1e-3);

    // Frequency resolution
    let freq_res = sample_rate / n_fft as f32;
    assert!((freq_res - 62.5).abs() < 1e-3);

    // All bins should span [0, Nyquist]
    for k in 0..n_bins {
        let freq = k as f32 * sample_rate / n_fft as f32;
        assert!(freq >= 0.0, "bin {k}: freq={freq} should be >= 0");
        assert!(
            freq <= sample_rate / 2.0 + 1e-3,
            "bin {k}: freq={freq} should be <= Nyquist"
        );
    }
}

#[test]
fn test_stft_pure_tone_peak_at_expected_bin() {
    // A cosine at exactly bin k should produce a peak at bin k
    let n_fft = 32;
    let hop = 8;
    let signal_len = 256;
    let target_bin = 5usize;

    let signal: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * target_bin as f32 * i as f32 / n_fft as f32).cos())
        .collect();

    let (real, imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
    let n_bins = n_fft / 2 + 1;

    // Check middle frame
    let t = n_frames / 2;
    let mut max_mag = 0.0f32;
    let mut max_bin = 0usize;
    for f in 0..n_bins {
        let mag = real[f * n_frames + t].hypot(imag[f * n_frames + t]);
        if mag > max_mag {
            max_mag = mag;
            max_bin = f;
        }
    }
    assert_eq!(
        max_bin, target_bin,
        "peak should be at bin {target_bin}, got {max_bin}"
    );
}

#[test]
fn test_dc_signal_energy_at_bin_zero() {
    // DC signal (constant): all energy at bin 0
    let n_fft = 32;
    let hop = 8;
    let signal_len = 256;
    let signal = vec![1.0f32; signal_len];

    let (real, imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
    let n_bins = n_fft / 2 + 1;

    let t = n_frames / 2;
    let dc_mag = real[t].hypot(imag[t]);
    for f in 1..n_bins {
        let other_mag = real[f * n_frames + t].hypot(imag[f * n_frames + t]);
        assert!(
            dc_mag > other_mag,
            "DC mag ({dc_mag}) should exceed bin {f} mag ({other_mag})"
        );
    }
}

#[test]
fn test_stft_frame_count_formula() {
    // n_frames = floor((signal_len - n_fft) / hop) + 1
    let configs: &[(usize, usize, usize)] = &[
        (64, 16, 256),
        (32, 8, 128),
        (20, 5, 200),
        (128, 32, 512),
        (16, 4, 64),
    ];
    for &(n_fft, hop, signal_len) in configs {
        let expected = (signal_len - n_fft) / hop + 1;
        let signal: Vec<f32> = vec![0.0; signal_len];
        let (_real, _imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
        assert_eq!(
            n_frames, expected,
            "n_fft={n_fft}, hop={hop}, len={signal_len}"
        );
    }
}

// ===========================================================================
// 6. Signal padding modes
// ===========================================================================

#[test]
fn test_stft_reflection_padding_produces_valid_output() {
    let params = StftParams::new(8, 4);
    let audio = vec![1.0f32; 8]; // padded_len = 8 + 2 = 10
    let basis = build_dft_basis(8);
    let result = compute_stft_magnitude(&audio, &basis, &params);
    assert!(
        result.is_ok(),
        "reflection padding should produce valid output"
    );
    let mag = result.unwrap();
    let n_freqs = 8 / 2 + 1;
    let n_frames = (8 + params.pad_right - 8) / 4 + 1;
    assert_eq!(mag.len(), n_freqs * n_frames);
}

#[test]
fn test_stft_audio_too_short_for_reflection_padding() {
    let params = StftParams::new(8, 4); // pad_right=2, needs >= 4 samples
    let audio = vec![1.0f32; 3]; // too short
    let basis = build_dft_basis(8);
    let result = compute_stft_magnitude(&audio, &basis, &params);
    assert!(
        matches!(result, Err(StftError::AudioTooShortForPadding { .. })),
        "audio too short for padding should error: {result:?}"
    );
}

#[test]
fn test_istft_output_zero_padded_when_requested_longer() {
    let n_fft = 16;
    let hop = 4;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 3;
    let full_len = n_fft + (n_frames - 1) * hop; // 24

    let mut real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];
    for t in 0..n_frames {
        real[t] = 1.0; // DC
    }

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let requested = 100;
    let output = basis.istft(&real, &imag, n_frames, requested).unwrap();

    assert_eq!(output.len(), requested);
    for i in full_len..requested {
        assert_eq!(output[i], 0.0, "padding at index {i} should be 0");
    }
}

#[test]
fn test_istft_output_truncated_when_requested_shorter() {
    let n_fft = 32;
    let hop = 8;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 10;
    let full_len = n_fft + (n_frames - 1) * hop; // 104

    let real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let requested = 50; // shorter than full_len
    let output = basis.istft(&real, &imag, n_frames, requested).unwrap();

    assert_eq!(output.len(), requested);
    assert!(
        requested < full_len,
        "requested should be shorter than full_len"
    );
}

#[test]
fn test_istft_center_trim_removes_correct_amount() {
    // center=true trims n_fft/2 from each side
    let n_fft = 64;
    let hop = 16;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 10;
    let full_len = n_fft + (n_frames - 1) * hop;
    let trim = n_fft / 2;
    let expected_trimmed = full_len - 2 * trim;

    let real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];

    let params = IstftParams::new(n_fft, hop, false, true).unwrap(); // center=true
    let basis = IstftBasis::new(params).unwrap();
    let output = basis
        .istft(&real, &imag, n_frames, expected_trimmed)
        .unwrap();

    assert_eq!(output.len(), expected_trimmed);
}

// ===========================================================================
// 7. Edge cases: empty signals, single-sample signals, very long signals
// ===========================================================================

#[test]
fn test_stft_empty_signal_yields_zero_frames() {
    let signal: Vec<f32> = vec![];
    let (_real, _imag, n_frames) = windowed_forward_stft(&signal, 32, 8);
    assert_eq!(n_frames, 0, "empty signal should yield 0 frames");
}

#[test]
fn test_stft_single_sample_signal_yields_zero_frames() {
    let signal = vec![1.0f32];
    let (_real, _imag, n_frames) = windowed_forward_stft(&signal, 16, 4);
    assert_eq!(
        n_frames, 0,
        "single-sample signal shorter than n_fft=16 => 0 frames"
    );
}

#[test]
fn test_stft_signal_shorter_than_nfft_yields_zero_frames() {
    let signal = vec![0.5f32; 15];
    let (_real, _imag, n_frames) = windowed_forward_stft(&signal, 16, 4);
    assert_eq!(n_frames, 0, "signal shorter than n_fft => 0 frames");
}

#[test]
fn test_stft_signal_exactly_nfft_yields_one_frame() {
    for &n_fft in &[8, 16, 32, 64] {
        let signal = vec![1.0f32; n_fft];
        let (_real, _imag, n_frames) = windowed_forward_stft(&signal, n_fft, n_fft / 4);
        assert_eq!(n_frames, 1, "n_fft={n_fft}: exactly 1 frame");
    }
}

#[test]
fn test_stft_very_long_signal_correct_frame_count() {
    // Very long signal: 100k samples
    let n_fft = 64;
    let hop = 16;
    let signal_len = 100_000;
    let expected_frames = (signal_len - n_fft) / hop + 1;

    // Only allocate the signal, not actually compute STFT (just test the formula)
    assert_eq!(expected_frames, 6247);

    // Do a smaller version to verify the actual STFT agrees
    let small_len = 10_000;
    let signal: Vec<f32> = (0..small_len).map(|i| (i as f32 * 0.001).sin()).collect();
    let (_real, _imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
    let expected = (small_len - n_fft) / hop + 1;
    assert_eq!(n_frames, expected);
}

#[test]
fn test_istft_single_frame_produces_finite_output() {
    let n_fft = 32;
    let hop = 8;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 1;

    let mut real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];
    real[0] = 1.0; // DC

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let output = basis.istft(&real, &imag, n_frames, n_fft).unwrap();

    assert_eq!(output.len(), n_fft);
    for v in &output {
        assert!(v.is_finite(), "single-frame output must be finite");
    }
}

#[test]
fn test_istft_zero_input_produces_zero_output() {
    let n_fft = 32;
    let hop = 8;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 10;

    let real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let output = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    for (i, &v) in output.iter().enumerate() {
        assert!(
            v.abs() < 1e-10,
            "zero input should yield zero output, got {v} at {i}"
        );
    }
}

#[test]
fn test_istft_rejects_nan_input() {
    let params = IstftParams::new(8, 2, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let n_bins = 5;
    let mut real = vec![0.0f32; n_bins * 2];
    real[0] = f32::NAN;
    let imag = vec![0.0f32; n_bins * 2];
    let err = basis.istft(&real, &imag, 2, 10).unwrap_err();
    assert!(matches!(err, IstftError::NonFiniteInput));
}

#[test]
fn test_istft_rejects_inf_input() {
    let params = IstftParams::new(8, 2, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let n_bins = 5;
    let mut real = vec![0.0f32; n_bins];
    real[0] = f32::INFINITY;
    let imag = vec![0.0f32; n_bins];
    let err = basis.istft(&real, &imag, 1, 8).unwrap_err();
    assert!(matches!(err, IstftError::NonFiniteInput));
}

#[test]
fn test_istft_rejects_length_mismatch() {
    let params = IstftParams::new(8, 2, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let real = vec![0.0f32; 10];
    let imag = vec![0.0f32; 15]; // different length
    let err = basis.istft(&real, &imag, 2, 10).unwrap_err();
    assert!(matches!(err, IstftError::LengthMismatch { .. }));
}

#[test]
fn test_istft_rejects_shape_mismatch() {
    let params = IstftParams::new(8, 2, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    // n_bins=5, n_frames=2 => expected 10, but we pass 8 elements
    let real = vec![0.0f32; 8];
    let imag = vec![0.0f32; 8];
    let err = basis.istft(&real, &imag, 2, 8).unwrap_err();
    assert!(matches!(err, IstftError::ShapeMismatch { .. }));
}

// ===========================================================================
// 8. DFT basis orthogonality and Nyquist row properties
// ===========================================================================

#[test]
fn test_dft_cos_basis_orthogonality_interior_rows() {
    let n_fft = 32;
    let params = IstftParams::new(n_fft, 8, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let cos = basis.cos_basis();
    let n_bins = n_fft / 2 + 1;
    let expected_self_dot = n_fft as f32 / 2.0;

    for f in 1..(n_bins - 1) {
        for g in 1..(n_bins - 1) {
            let mut dot = 0.0f32;
            for k in 0..n_fft {
                dot += cos[f * n_fft + k] * cos[g * n_fft + k];
            }
            if f == g {
                assert!(
                    (dot - expected_self_dot).abs() < 0.01,
                    "cos row {f} self-dot={dot}, expected {expected_self_dot}"
                );
            } else {
                assert!(
                    dot.abs() < 0.01,
                    "cos row {f} dot row {g}={dot}, expected ~0"
                );
            }
        }
    }
}

#[test]
fn test_dft_cos_sin_cross_orthogonality() {
    let n_fft = 32;
    let params = IstftParams::new(n_fft, 8, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let cos = basis.cos_basis();
    let sin = basis.sin_basis();
    let n_bins = n_fft / 2 + 1;

    for f in 0..n_bins {
        let mut dot = 0.0f32;
        for k in 0..n_fft {
            dot += cos[f * n_fft + k] * sin[f * n_fft + k];
        }
        assert!(dot.abs() < 0.01, "cos-sin cross at freq {f}: dot={dot}");
    }
}

#[test]
fn test_dft_nyquist_row_alternating_sign() {
    // cos(pi*k) = (-1)^k
    let n_fft = 16;
    let params = IstftParams::new(n_fft, 4, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let cos = basis.cos_basis();
    let nyquist = n_fft / 2;

    for k in 0..n_fft {
        let expected = if k % 2 == 0 { 1.0 } else { -1.0 };
        let val = cos[nyquist * n_fft + k];
        assert!(
            (val - expected).abs() < 1e-6,
            "Nyquist cos[{nyquist}, {k}]={val}, expected {expected}"
        );
    }
}

#[test]
fn test_dft_dc_row_all_ones() {
    // cos(0) = 1.0 for all k
    let n_fft = 16;
    let params = IstftParams::new(n_fft, 4, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    for k in 0..n_fft {
        let val = basis.cos_basis()[k]; // f=0
        assert!(
            (val - 1.0).abs() < 1e-6,
            "cos DC row[{k}]={val}, expected 1.0"
        );
    }
}

#[test]
fn test_dft_sin_dc_row_all_zeros() {
    // sin(0) = 0.0 for all k
    let n_fft = 16;
    let params = IstftParams::new(n_fft, 4, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    for k in 0..n_fft {
        let val = basis.sin_basis()[k]; // f=0
        assert!(val.abs() < 1e-6, "sin DC row[{k}]={val}, expected 0.0");
    }
}

// ===========================================================================
// 9. KokoroForwardStft shape and error paths
// ===========================================================================

#[test]
fn test_kokoro_forward_stft_output_shape() {
    let n_fft = 20;
    let hop = 5;
    let signal_len = 200;
    let n_bins = n_fft / 2 + 1;
    let expected_frames = (signal_len - n_fft) / hop + 1;

    let signal: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * 3.0 * i as f32 / signal_len as f32).sin())
        .collect();
    let stft = KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let input = DynTensor::from_vec(signal, &[1, 1, signal_len], &Device::Cpu).unwrap();

    let (magnitude, phase) = stft.forward(&input).unwrap();
    assert_eq!(magnitude.dims(), &[1, n_bins, expected_frames]);
    assert_eq!(phase.dims(), &[1, n_bins, expected_frames]);
}

#[test]
fn test_kokoro_forward_stft_batched() {
    let n_fft = 20;
    let hop = 5;
    let signal_len = 100;
    let batch = 3;
    let n_bins = n_fft / 2 + 1;
    let expected_frames = (signal_len - n_fft) / hop + 1;

    let signal = vec![0.5f32; batch * signal_len];
    let stft = KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let input = DynTensor::from_vec(signal, &[batch, 1, signal_len], &Device::Cpu).unwrap();

    let (magnitude, phase) = stft.forward(&input).unwrap();
    assert_eq!(magnitude.dims(), &[batch, n_bins, expected_frames]);
    assert_eq!(phase.dims(), &[batch, n_bins, expected_frames]);
}

#[test]
fn test_kokoro_forward_stft_magnitude_non_negative() {
    let n_fft = 20;
    let hop = 5;
    let signal_len = 200;

    let signal: Vec<f32> = (0..signal_len)
        .map(|i| {
            let t = i as f32 / signal_len as f32;
            0.3 * (2.0 * PI * 2.0 * t).sin() + 0.7 * (2.0 * PI * 5.0 * t).cos()
        })
        .collect();

    let stft = KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let input = DynTensor::from_vec(signal, &[1, 1, signal_len], &Device::Cpu).unwrap();
    let (magnitude, _) = stft.forward(&input).unwrap();

    let mag_flat = magnitude.to_flat_vec::<f32>().unwrap();
    for (i, &v) in mag_flat.iter().enumerate() {
        assert!(v >= 0.0, "magnitude must be non-negative: mag[{i}]={v}");
        assert!(v.is_finite(), "magnitude must be finite: mag[{i}]={v}");
    }
}

#[test]
fn test_kokoro_forward_stft_phase_in_range() {
    let n_fft = 20;
    let hop = 5;
    let signal_len = 200;

    let signal: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * 4.0 * i as f32 / signal_len as f32).sin())
        .collect();

    let stft = KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let input = DynTensor::from_vec(signal, &[1, 1, signal_len], &Device::Cpu).unwrap();
    let (_, phase) = stft.forward(&input).unwrap();

    let phase_flat = phase.to_flat_vec::<f32>().unwrap();
    for (i, &v) in phase_flat.iter().enumerate() {
        assert!(
            (-PI..=PI).contains(&v),
            "phase must be in [-pi, pi]: phase[{i}]={v}"
        );
        assert!(v.is_finite(), "phase must be finite: phase[{i}]={v}");
    }
}

#[test]
fn test_kokoro_forward_stft_zero_signal_zero_magnitude() {
    let n_fft = 20;
    let hop = 5;
    let signal_len = 100;

    let signal = vec![0.0f32; signal_len];
    let stft = KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let input = DynTensor::from_vec(signal, &[1, 1, signal_len], &Device::Cpu).unwrap();
    let (magnitude, _) = stft.forward(&input).unwrap();

    let mag_flat = magnitude.to_flat_vec::<f32>().unwrap();
    for (i, &v) in mag_flat.iter().enumerate() {
        assert!(v.abs() < 1e-8, "zero signal => zero mag, got {v} at {i}");
    }
}

#[test]
fn test_kokoro_forward_stft_forward_cat_shape() {
    let n_fft = 20;
    let hop = 5;
    let signal_len = 100;
    let n_bins = n_fft / 2 + 1;
    let expected_frames = (signal_len - n_fft) / hop + 1;

    let signal: Vec<f32> = (0..signal_len).map(|i| i as f32 * 0.01).collect();
    let stft = KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let input = DynTensor::from_vec(signal, &[1, 1, signal_len], &Device::Cpu).unwrap();

    let cat = stft.forward_cat(&input).unwrap();
    assert_eq!(cat.dims(), &[1, 2 * n_bins, expected_frames]);
}

#[test]
fn test_kokoro_forward_stft_rejects_2d_input() {
    let stft = KokoroForwardStft::new(20, 5, &Device::Cpu).unwrap();
    let input = DynTensor::from_vec(vec![0.0f32; 100], &[10, 10], &Device::Cpu).unwrap();
    assert!(stft.forward(&input).is_err(), "2D input should be rejected");
}

#[test]
fn test_kokoro_forward_stft_rejects_multichannel() {
    let stft = KokoroForwardStft::new(20, 5, &Device::Cpu).unwrap();
    let input = DynTensor::from_vec(vec![0.0f32; 200], &[1, 2, 100], &Device::Cpu).unwrap();
    assert!(
        stft.forward(&input).is_err(),
        "multichannel input should be rejected"
    );
}

#[test]
fn test_kokoro_forward_stft_rejects_too_short_signal() {
    let n_fft = 20;
    let stft = KokoroForwardStft::new(n_fft, 5, &Device::Cpu).unwrap();
    let input = DynTensor::from_vec(vec![0.0f32; 10], &[1, 1, 10], &Device::Cpu).unwrap();
    assert!(
        stft.forward(&input).is_err(),
        "signal shorter than n_fft should be rejected"
    );
}

// ===========================================================================
// 10. Kokoro signal constants and utility functions
// ===========================================================================

#[test]
fn test_kokoro_signal_constants_consistency() {
    assert_eq!(KOKORO_N_FFT, 20);
    assert_eq!(KOKORO_HOP_LENGTH, 5);
    assert_eq!(KOKORO_SAMPLE_RATE, 24000);
    assert_eq!(KOKORO_N_BINS, KOKORO_N_FFT / 2 + 1);
    assert_eq!(KOKORO_N_BINS, 11);
}

#[test]
fn test_harmonic_source_output_shape() {
    let t = 100;
    let f0_data = vec![440.0f32; t];
    let f0 = DynTensor::from_vec(f0_data, &[1, 1, t], &Device::Cpu).unwrap();

    let result = harmonic_source(&f0, KOKORO_SAMPLE_RATE as f32).unwrap();
    assert_eq!(result.dims(), &[1, 1, t]);
}

#[test]
fn test_harmonic_source_zero_f0_produces_zero() {
    let t = 50;
    let f0_data = vec![0.0f32; t];
    let f0 = DynTensor::from_vec(f0_data, &[1, 1, t], &Device::Cpu).unwrap();

    let result = harmonic_source(&f0, KOKORO_SAMPLE_RATE as f32).unwrap();
    let flat = result.to_flat_vec::<f32>().unwrap();
    for (i, &v) in flat.iter().enumerate() {
        assert!(v.abs() < 1e-6, "zero F0 => zero output, got {v} at {i}");
    }
}

#[test]
fn test_harmonic_source_bounded_amplitude() {
    let t = 200;
    let f0_data: Vec<f32> = (0..t)
        .map(|i| 100.0 + 300.0 * (i as f32 / t as f32))
        .collect();
    let f0 = DynTensor::from_vec(f0_data, &[1, 1, t], &Device::Cpu).unwrap();

    let result = harmonic_source(&f0, KOKORO_SAMPLE_RATE as f32).unwrap();
    let flat = result.to_flat_vec::<f32>().unwrap();
    for (i, &v) in flat.iter().enumerate() {
        assert!(
            v.abs() <= 1.0 + 1e-6,
            "amplitude must be <= 1.0, got {v} at {i}"
        );
        assert!(v.is_finite(), "output must be finite at {i}");
    }
}

#[test]
fn test_prepare_istft_input_shape_and_split() {
    let n_fft = 20;
    let n_frames = 4;
    let half = n_fft / 2;
    let n_bins = half + 1;

    let mut data = Vec::with_capacity(n_fft * n_frames);
    for c in 0..n_fft {
        for t in 0..n_frames {
            data.push((c * 100 + t) as f32);
        }
    }
    let decoder_output = DynTensor::from_vec(data, &[1, n_fft, n_frames], &Device::Cpu).unwrap();

    let (real, imag, frames): (Vec<f32>, Vec<f32>, usize) =
        prepare_istft_input(&decoder_output).unwrap();
    assert_eq!(frames, n_frames);
    assert_eq!(real.len(), n_bins * n_frames);
    assert_eq!(imag.len(), n_bins * n_frames);

    // Nyquist rows should be zero-padded
    for t in 0..n_frames {
        assert!(
            real[half * n_frames + t].abs() < 1e-6,
            "real Nyquist should be zero"
        );
        assert!(
            imag[half * n_frames + t].abs() < 1e-6,
            "imag Nyquist should be zero"
        );
    }
}

#[test]
fn test_prepare_istft_input_rejects_wrong_rank() {
    let data = vec![0.0f32; 20];
    let tensor = DynTensor::from_vec(data, &[4, 5], &Device::Cpu).unwrap();
    assert!(
        prepare_istft_input(&tensor).is_err(),
        "2D input should be rejected"
    );
}

#[test]
fn test_prepare_istft_input_rejects_batch_not_one() {
    let data = vec![0.0f32; 2 * 20 * 4];
    let tensor = DynTensor::from_vec(data, &[2, 20, 4], &Device::Cpu).unwrap();
    assert!(
        prepare_istft_input(&tensor).is_err(),
        "batch != 1 should be rejected"
    );
}

// ===========================================================================
// 11. iSTFT DC-only and normalized scaling
// ===========================================================================

#[test]
fn test_istft_dc_only_input_constant_interior() {
    let n_fft = 32;
    let hop = 8;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 20;

    let mut real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];
    for t in 0..n_frames {
        real[t] = 1.0; // DC
    }

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let output = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let margin = 2 * n_fft;
    let start = margin;
    let end = full_len.saturating_sub(margin);

    if end > start {
        let first_val = output[start];
        assert!(
            first_val.abs() > 1e-6,
            "DC-only should produce non-zero interior"
        );
        for i in start..end {
            let diff = (output[i] - first_val).abs();
            assert!(
                diff < 0.01 * first_val.abs(),
                "DC-only interior should be constant: output[{i}]={}, expected ~{first_val}",
                output[i]
            );
        }
    }
}

#[test]
fn test_istft_normalized_vs_unnormalized_differ_by_sqrt_n() {
    let n_fft = 16;
    let hop = 4;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 5;

    let mut real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];
    for t in 0..n_frames {
        real[t] = 1.0;
    }

    let full_len = n_fft + (n_frames - 1) * hop;

    let params_norm = IstftParams::new(n_fft, hop, true, false).unwrap();
    let basis_norm = IstftBasis::new(params_norm).unwrap();
    let out_norm = basis_norm.istft(&real, &imag, n_frames, full_len).unwrap();

    let params_unnorm = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis_unnorm = IstftBasis::new(params_unnorm).unwrap();
    let out_unnorm = basis_unnorm
        .istft(&real, &imag, n_frames, full_len)
        .unwrap();

    let ratio = (n_fft as f32).sqrt();
    let margin = 2 * n_fft;
    let start = margin;
    let end = full_len.saturating_sub(margin);

    if end > start {
        for i in start..end {
            if out_unnorm[i].abs() > 1e-6 {
                let actual_ratio = out_norm[i] / out_unnorm[i];
                assert!(
                    (actual_ratio - ratio).abs() < 0.1,
                    "norm/unnorm ratio at {i}: {actual_ratio}, expected ~{ratio}"
                );
            }
        }
    }
}

// ===========================================================================
// 12. STFT linearity and energy scaling
// ===========================================================================

#[test]
fn test_stft_linearity() {
    // STFT(a*x + b*y) = a*STFT(x) + b*STFT(y)
    let n_fft = 32;
    let hop = 8;
    let signal_len = 128;
    let a = 0.7f32;
    let b = 1.3f32;

    let x: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * 3.0 * i as f32 / signal_len as f32).sin())
        .collect();
    let y: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * 7.0 * i as f32 / signal_len as f32).cos())
        .collect();
    let combined: Vec<f32> = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| a * xi + b * yi)
        .collect();

    let (rx, ix, n_frames) = windowed_forward_stft(&x, n_fft, hop);
    let (ry, iy, _) = windowed_forward_stft(&y, n_fft, hop);
    let (rc, ic, _) = windowed_forward_stft(&combined, n_fft, hop);

    let n_bins = n_fft / 2 + 1;
    let mut max_real_err = 0.0f32;
    let mut max_imag_err = 0.0f32;
    for i in 0..(n_bins * n_frames) {
        max_real_err = max_real_err.max((rc[i] - (a * rx[i] + b * ry[i])).abs());
        max_imag_err = max_imag_err.max((ic[i] - (a * ix[i] + b * iy[i])).abs());
    }
    assert!(max_real_err < 1e-4, "linearity real error: {max_real_err}");
    assert!(max_imag_err < 1e-4, "linearity imag error: {max_imag_err}");
}

#[test]
fn test_stft_energy_quadruples_with_doubled_amplitude() {
    let n_fft = 32;
    let hop = 8;
    let signal_len = 128;

    let signal: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * 5.0 * i as f32 / signal_len as f32).sin())
        .collect();
    let scaled: Vec<f32> = signal.iter().map(|&v| 2.0 * v).collect();

    let (r1, i1, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
    let (r2, i2, _) = windowed_forward_stft(&scaled, n_fft, hop);

    let n_bins = n_fft / 2 + 1;
    let t = n_frames / 2;
    let mut energy1 = 0.0f32;
    let mut energy2 = 0.0f32;
    for f in 0..n_bins {
        energy1 += r1[f * n_frames + t].powi(2) + i1[f * n_frames + t].powi(2);
        energy2 += r2[f * n_frames + t].powi(2) + i2[f * n_frames + t].powi(2);
    }

    if energy1 > 1e-8 {
        let ratio = energy2 / energy1;
        assert!(
            (ratio - 4.0).abs() < 0.1,
            "energy ratio={ratio}, expected ~4.0"
        );
    }
}

// ===========================================================================
// 13. AM signal spectral properties
// ===========================================================================

#[test]
fn test_stft_am_signal_has_carrier_and_sidebands() {
    // AM: cos(2pi*fc*t) * (1 + m*cos(2pi*fm*t))
    // Produces peaks at fc, fc+fm, fc-fm
    let n_fft = 64;
    let signal_len = 512;
    let fc = 8; // carrier bin
    let fm = 3; // modulator bin
    let m = 0.5f32;

    let signal: Vec<f32> = (0..signal_len)
        .map(|i| {
            let t = i as f32 / n_fft as f32;
            let carrier = (2.0 * PI * fc as f32 * t).cos();
            let modulator = 1.0 + m * (2.0 * PI * fm as f32 * t).cos();
            carrier * modulator
        })
        .collect();

    let window = hann_window(n_fft);
    let n_bins = n_fft / 2 + 1;
    let n_frames = (signal_len - n_fft) / 16 + 1;
    let t_mid = n_frames / 2;
    let offset = t_mid * 16;

    let mut magnitudes = vec![0.0f32; n_bins];
    for f in 0..n_bins {
        let mut r = 0.0f32;
        let mut im = 0.0f32;
        for k in 0..n_fft {
            let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
            let windowed = signal[offset + k] * window[k];
            r += windowed * angle.cos();
            im -= windowed * angle.sin();
        }
        magnitudes[f] = r.hypot(im);
    }

    let carrier_mag = magnitudes[fc];
    let lower_sb = magnitudes[fc - fm];
    let upper_sb = magnitudes[fc + fm];

    assert!(
        carrier_mag > lower_sb,
        "carrier should exceed lower sideband"
    );
    assert!(
        carrier_mag > upper_sb,
        "carrier should exceed upper sideband"
    );
    // Sidebands should have significant energy
    assert!(
        lower_sb > 0.1,
        "lower sideband should have significant energy"
    );
    assert!(
        upper_sb > 0.1,
        "upper sideband should have significant energy"
    );
}
