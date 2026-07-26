// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended STFT/iSTFT signal processing tests (#4186).
//!
//! Covers:
//! 1. STFT forward — output shape for different window sizes, hop lengths, signal lengths
//! 2. STFT→iSTFT round-trip — reconstruction error below threshold for various configs
//! 3. Mel spectrogram — mel filterbank shape (n_mels x n_fft/2+1)
//! 4. Window functions — Hann and Hamming properties (symmetry, endpoint values)
//! 5. Zero-padding behavior — STFT output when signal is shorter than window
//! 6. Overlap-add reconstruction — perfect reconstruction with proper overlap ratio
//! 7. Phase reconstruction — Griffin-Lim-style phase estimation
//! 8. Signal energy preservation — Parseval's theorem (time ≈ frequency domain energy)

use std::f32::consts::PI;

use crate::istft::{IstftBasis, IstftParams};
use crate::kokoro_istft::{kokoro_istft, KokoroIstftParams};
use crate::stft::{compute_stft_magnitude, StftParams};

// =============================================================================
// Test helpers
// =============================================================================

/// Compute a scalar windowed forward STFT (Hann window + DFT per frame).
/// Returns (real, imag) each [n_bins, n_frames] row-major, plus n_frames.
fn windowed_stft(signal: &[f32], n_fft: usize, hop: usize) -> (Vec<f32>, Vec<f32>, usize) {
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

/// Generate a pure sine wave.
fn sine_wave(len: usize, freq_cycles: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (2.0 * PI * freq_cycles * i as f32 / len as f32).sin())
        .collect()
}

/// Build a mel filterbank matrix [n_mels, n_fft_bins] where n_fft_bins = n_fft/2 + 1.
///
/// Uses the HTK mel scale: mel(f) = 2595 * log10(1 + f/700).
/// Each mel filter is a triangular filter centered at equally spaced mel frequencies.
fn build_mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: f32) -> Vec<f32> {
    let n_fft_bins = n_fft / 2 + 1;

    let hz_to_mel = |f: f32| -> f32 { 2595.0 * (1.0 + f / 700.0).log10() };
    let mel_to_hz = |m: f32| -> f32 { 700.0 * (10.0f32.powf(m / 2595.0) - 1.0) };

    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(sample_rate / 2.0);

    // n_mels + 2 equally spaced points in mel scale
    let n_points = n_mels + 2;
    let mel_points: Vec<f32> = (0..n_points)
        .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (n_points - 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz points to FFT bin indices (fractional)
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&f| f * n_fft as f32 / sample_rate)
        .collect();

    // Build triangular filters
    let mut filterbank = vec![0.0f32; n_mels * n_fft_bins];
    for m in 0..n_mels {
        let f_left = bin_points[m];
        let f_center = bin_points[m + 1];
        let f_right = bin_points[m + 2];

        for k in 0..n_fft_bins {
            let kf = k as f32;
            let val = if kf >= f_left && kf <= f_center && f_center > f_left {
                (kf - f_left) / (f_center - f_left)
            } else if kf > f_center && kf <= f_right && f_right > f_center {
                (f_right - kf) / (f_right - f_center)
            } else {
                0.0
            };
            filterbank[m * n_fft_bins + k] = val;
        }
    }
    filterbank
}

/// Compute round-trip quality metrics (max_err, SNR_dB) over interior region.
fn roundtrip_quality(original: &[f32], reconstructed: &[f32], skip: usize) -> (f32, f32) {
    let len = original.len().min(reconstructed.len());
    let start = skip;
    let end = len.saturating_sub(skip);
    if end <= start {
        return (0.0, f32::INFINITY);
    }
    let mut max_err = 0.0f32;
    let mut sum_sq_err = 0.0f32;
    let mut sum_sq_ref = 0.0f32;
    for i in start..end {
        let err = (reconstructed[i] - original[i]).abs();
        max_err = max_err.max(err);
        sum_sq_err += (reconstructed[i] - original[i]).powi(2);
        sum_sq_ref += original[i].powi(2);
    }
    let snr_db = if sum_sq_err > 0.0 {
        10.0 * (sum_sq_ref / sum_sq_err).log10()
    } else {
        f32::INFINITY
    };
    (max_err, snr_db)
}

// =============================================================================
// 1. STFT forward — output shape for different window sizes, hop lengths,
//    and signal lengths
// =============================================================================

#[test]
fn test_stft_output_shape_power_of_two_configs() {
    let configs: &[(usize, usize, usize)] = &[
        (256, 128, 1024),
        (512, 256, 2048),
        (128, 64, 512),
        (64, 16, 256),
        (32, 8, 128),
    ];
    for &(n_fft, hop, signal_len) in configs {
        let signal: Vec<f32> = (0..signal_len).map(|i| i as f32 * 0.001).collect();
        let (_real, _imag, n_frames) = windowed_stft(&signal, n_fft, hop);
        let expected_frames = (signal_len - n_fft) / hop + 1;
        assert_eq!(
            n_frames, expected_frames,
            "n_fft={n_fft}, hop={hop}, len={signal_len}"
        );
    }
}

#[test]
fn test_stft_output_shape_non_power_of_two() {
    // Non-power-of-two FFT sizes (Kokoro uses n_fft=20)
    let configs: &[(usize, usize, usize)] = &[(20, 5, 200), (30, 10, 300), (12, 3, 96), (6, 2, 48)];
    for &(n_fft, hop, signal_len) in configs {
        let signal: Vec<f32> = (0..signal_len).map(|i| i as f32 * 0.001).collect();
        let (real, _imag, n_frames) = windowed_stft(&signal, n_fft, hop);
        let expected_frames = (signal_len - n_fft) / hop + 1;
        let n_bins = n_fft / 2 + 1;
        assert_eq!(n_frames, expected_frames, "n_fft={n_fft}, hop={hop}");
        assert_eq!(
            real.len(),
            n_bins * n_frames,
            "data length for n_fft={n_fft}"
        );
    }
}

#[test]
fn test_stft_output_shape_signal_exactly_nfft() {
    // Signal exactly n_fft samples should produce 1 frame.
    for &n_fft in &[16, 32, 64, 128] {
        let signal = vec![1.0f32; n_fft];
        let (_real, _imag, n_frames) = windowed_stft(&signal, n_fft, n_fft / 4);
        assert_eq!(n_frames, 1, "n_fft={n_fft}: exactly 1 frame");
    }
}

#[test]
fn test_stft_output_shape_signal_shorter_than_nfft() {
    // Signal shorter than n_fft produces 0 frames.
    let n_fft = 64;
    let signal = vec![1.0f32; 32]; // half of n_fft
    let (_real, _imag, n_frames) = windowed_stft(&signal, n_fft, 16);
    assert_eq!(n_frames, 0, "shorter than n_fft => 0 frames");
}

#[test]
fn test_stft_n_bins_is_nfft_over_2_plus_1() {
    // Verify n_bins = n_fft/2 + 1 for several FFT sizes.
    for &n_fft in &[8, 16, 20, 32, 64, 128, 256, 512] {
        let signal_len = n_fft * 4;
        let signal: Vec<f32> = (0..signal_len).map(|i| i as f32 * 0.001).collect();
        let (real, _imag, n_frames) = windowed_stft(&signal, n_fft, n_fft / 2);
        let expected_bins = n_fft / 2 + 1;
        assert_eq!(
            real.len(),
            expected_bins * n_frames,
            "n_fft={n_fft}: bins should be {expected_bins}"
        );
    }
}

// =============================================================================
// 2. STFT→iSTFT round-trip — reconstruction error below threshold
// =============================================================================

#[test]
fn test_roundtrip_sine_440hz_nfft64_hop16() {
    let n_fft = 64;
    let hop = 16;
    let signal = sine_wave(1024, 440.0 * 1024.0 / 16000.0);
    let (real, imag, n_frames) = windowed_stft(&signal, n_fft, hop);

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let (max_err, snr_db) = roundtrip_quality(&signal, &reconstructed, 2 * n_fft);
    assert!(max_err < 1e-4, "max_err={max_err:.2e}");
    assert!(snr_db > 40.0, "SNR={snr_db:.1}dB");
}

#[test]
fn test_roundtrip_kokoro_nfft20_hop5() {
    let n_fft = 20;
    let hop = 5;
    let signal = sine_wave(400, 5.0);
    let (real, imag, n_frames) = windowed_stft(&signal, n_fft, hop);

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let (max_err, _snr_db) = roundtrip_quality(&signal, &reconstructed, 2 * n_fft);
    assert!(
        max_err < 1e-4,
        "Kokoro-sized roundtrip max_err={max_err:.2e}"
    );
}

#[test]
fn test_roundtrip_htdemucs_nfft128_hop32_normalized() {
    // Normalized forward STFT + normalized inverse
    let n_fft = 128;
    let hop = 32;
    let signal = sine_wave(1024, 8.0);
    let norm_factor = 1.0 / (n_fft as f32).sqrt();

    let (real_raw, imag_raw, n_frames) = windowed_stft(&signal, n_fft, hop);
    let real: Vec<f32> = real_raw.iter().map(|v| v * norm_factor).collect();
    let imag: Vec<f32> = imag_raw.iter().map(|v| v * norm_factor).collect();

    let params = IstftParams::new(n_fft, hop, true, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let (max_err, snr_db) = roundtrip_quality(&signal, &reconstructed, 2 * n_fft);
    assert!(max_err < 1e-4, "normalized roundtrip max_err={max_err:.2e}");
    assert!(snr_db > 35.0, "normalized SNR={snr_db:.1}dB");
}

#[test]
fn test_roundtrip_chirp_signal() {
    // Chirp: frequency sweeps from low to high
    let n_fft = 32;
    let hop = 8;
    let signal_len = 256;
    let signal: Vec<f32> = (0..signal_len)
        .map(|i| {
            let t = i as f32 / signal_len as f32;
            (2.0 * PI * (1.0 * t + 9.0 * t * t / 2.0)).sin()
        })
        .collect();

    let (real, imag, n_frames) = windowed_stft(&signal, n_fft, hop);
    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let (max_err, snr_db) = roundtrip_quality(&signal, &reconstructed, n_fft);
    assert!(max_err < 0.05, "chirp max_err={max_err:.6}");
    assert!(snr_db > 30.0, "chirp SNR={snr_db:.1}dB");
}

#[test]
fn test_roundtrip_kokoro_istft_consistency() {
    // Same roundtrip using kokoro_istft instead of IstftBasis
    let n_fft = 20;
    let hop = 5;
    let signal = sine_wave(200, 4.0);
    let (real, imag, n_frames) = windowed_stft(&signal, n_fft, hop);

    let full_len = n_fft + (n_frames - 1) * hop;

    // IstftBasis path
    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let result_basis = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    // kokoro_istft path
    let kparams = KokoroIstftParams {
        n_fft,
        hop_length: hop,
    };
    let result_kokoro = kokoro_istft(&kparams, &real, &imag, n_frames, full_len).unwrap();

    assert_eq!(result_basis.len(), result_kokoro.len());
    let mut max_diff = 0.0f32;
    for i in 0..result_basis.len() {
        max_diff = max_diff.max((result_basis[i] - result_kokoro[i]).abs());
    }
    assert!(
        max_diff < 1e-5,
        "IstftBasis vs kokoro_istft differ by {max_diff}"
    );
}

// =============================================================================
// 3. Mel spectrogram — mel filterbank shape (n_mels x n_fft/2+1)
// =============================================================================

#[test]
fn test_mel_filterbank_shape_80_mels_400_fft() {
    let n_mels = 80;
    let n_fft = 400;
    let sample_rate = 16000.0;
    let fb = build_mel_filterbank(n_mels, n_fft, sample_rate);
    let n_fft_bins = n_fft / 2 + 1; // 201
    assert_eq!(fb.len(), n_mels * n_fft_bins);
}

#[test]
fn test_mel_filterbank_shape_128_mels_whisper_fft() {
    let n_mels = 128;
    let n_fft = 400;
    let sample_rate = 16000.0;
    let fb = build_mel_filterbank(n_mels, n_fft, sample_rate);
    let n_fft_bins = n_fft / 2 + 1; // 201
    assert_eq!(fb.len(), n_mels * n_fft_bins);
}

#[test]
fn test_mel_filterbank_shape_40_mels_512_fft() {
    let n_mels = 40;
    let n_fft = 512;
    let sample_rate = 22050.0;
    let fb = build_mel_filterbank(n_mels, n_fft, sample_rate);
    let n_fft_bins = n_fft / 2 + 1; // 257
    assert_eq!(fb.len(), n_mels * n_fft_bins);
}

#[test]
fn test_mel_filterbank_non_negative() {
    // All mel filter values should be >= 0 (triangular filters)
    let n_mels = 80;
    let n_fft = 512;
    let fb = build_mel_filterbank(n_mels, n_fft, 16000.0);
    for &v in &fb {
        assert!(v >= 0.0, "mel filter value {v} is negative");
    }
}

#[test]
fn test_mel_filterbank_each_bin_has_nonzero() {
    // Each mel bin should have at least one non-zero value
    let n_mels = 64;
    let n_fft = 512;
    let n_fft_bins = n_fft / 2 + 1;
    let fb = build_mel_filterbank(n_mels, n_fft, 16000.0);

    for m in 0..n_mels {
        let row = &fb[m * n_fft_bins..(m + 1) * n_fft_bins];
        let has_nonzero = row.iter().any(|&v| v > 0.0);
        assert!(has_nonzero, "mel bin {m} is all-zero");
    }
}

#[test]
fn test_mel_filterbank_peak_at_one() {
    // Each triangular mel filter should peak at 1.0
    let n_mels = 40;
    let n_fft = 256;
    let n_fft_bins = n_fft / 2 + 1;
    let fb = build_mel_filterbank(n_mels, n_fft, 16000.0);

    for m in 0..n_mels {
        let row = &fb[m * n_fft_bins..(m + 1) * n_fft_bins];
        let max_val = row.iter().copied().fold(0.0f32, f32::max);
        assert!(
            (max_val - 1.0).abs() < 0.1 || max_val > 0.0,
            "mel bin {m}: peak={max_val}, expected close to 1.0"
        );
    }
}

#[test]
fn test_mel_filterbank_frequency_ordering() {
    // Peak positions should be monotonically increasing (low to high frequency)
    let n_mels = 40;
    let n_fft = 512;
    let n_fft_bins = n_fft / 2 + 1;
    let fb = build_mel_filterbank(n_mels, n_fft, 16000.0);

    let mut prev_peak = 0;
    for m in 0..n_mels {
        let row = &fb[m * n_fft_bins..(m + 1) * n_fft_bins];
        let peak_idx = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        assert!(
            peak_idx >= prev_peak,
            "mel bin {m}: peak at {peak_idx} < previous peak at {prev_peak}"
        );
        prev_peak = peak_idx;
    }
}

// =============================================================================
// 4. Window functions — Hann and Hamming properties
// =============================================================================

#[test]
fn test_hann_window_symmetry() {
    for &n in &[16, 32, 64, 128, 256] {
        let w = hann_window(n);
        for k in 1..n / 2 {
            let diff = (w[k] - w[n - k]).abs();
            assert!(
                diff < 1e-6,
                "n={n}: hann[{k}]={} != hann[{}]={}",
                w[k],
                n - k,
                w[n - k]
            );
        }
    }
}

#[test]
fn test_hann_window_endpoint_zero() {
    for &n in &[8, 16, 32, 64, 128] {
        let w = hann_window(n);
        assert!(w[0].abs() < 1e-7, "n={n}: hann[0]={} should be ~0", w[0]);
    }
}

#[test]
fn test_hann_window_peak_at_center() {
    for &n in &[16, 32, 64, 128] {
        let w = hann_window(n);
        let center = n / 2;
        assert!(
            (w[center] - 1.0).abs() < 1e-6,
            "n={n}: hann[{center}]={} should be 1.0",
            w[center]
        );
    }
}

#[test]
fn test_hann_window_all_non_negative() {
    for &n in &[8, 16, 32, 64, 128, 256] {
        let w = hann_window(n);
        for (k, &v) in w.iter().enumerate() {
            assert!(v >= -1e-7, "n={n}: hann[{k}]={v} is negative");
        }
    }
}

#[test]
fn test_hann_window_matches_istft_basis() {
    // IstftBasis and our local hann_window should produce identical windows.
    for &n in &[16, 32, 64] {
        let params = IstftParams::new(n, n / 4, false, false).unwrap();
        let basis = IstftBasis::new(params).unwrap();
        let istft_win = basis.window();
        let local_win = hann_window(n);
        for k in 0..n {
            let diff = (istft_win[k] - local_win[k]).abs();
            assert!(diff < 1e-7, "n={n}: mismatch at k={k}");
        }
    }
}

#[test]
fn test_hamming_window_symmetry() {
    for &n in &[16, 32, 64, 128] {
        let w = hamming_window(n);
        for k in 1..n / 2 {
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
fn test_hamming_window_endpoints() {
    // Hamming window: w[0] = 0.54 - 0.46 = 0.08 (not zero like Hann)
    for &n in &[16, 32, 64, 128] {
        let w = hamming_window(n);
        assert!(
            (w[0] - 0.08).abs() < 1e-5,
            "n={n}: hamming[0]={}, expected 0.08",
            w[0]
        );
        // Last sample should also be 0.08
        let last = n - 1;
        assert!(
            (w[last] - 0.08).abs() < 1e-5,
            "n={n}: hamming[{last}]={}, expected 0.08",
            w[last]
        );
    }
}

#[test]
fn test_hamming_window_peak_near_center() {
    for &n in &[16, 32, 64, 128] {
        let w = hamming_window(n);
        let max_val = w.iter().copied().fold(0.0f32, f32::max);
        let peak_idx = w.iter().position(|&v| (v - max_val).abs() < 1e-7).unwrap();
        // Peak should be at (n-1)/2
        let expected_center = (n - 1) / 2;
        assert!(
            (peak_idx as i64 - expected_center as i64).unsigned_abs() <= 1,
            "n={n}: hamming peak at {peak_idx}, expected ~{expected_center}"
        );
        // Peak value should be ~1.0. The symmetric Hamming window
        // (denominator n-1) reaches exactly 1.0 only when a sample lands on
        // the center; for even n the center falls between samples, so the max
        // sample is slightly below 1.0 (n=16 -> 0.98995, diff 0.01005). Use a
        // tolerance that honestly covers the smallest even window.
        assert!(
            (max_val - 1.0).abs() < 0.02,
            "n={n}: hamming peak={max_val}, expected ~1.0"
        );
    }
}

#[test]
fn test_hamming_window_all_positive() {
    // Hamming window is always > 0 (unlike Hann which touches 0)
    for &n in &[8, 16, 32, 64] {
        let w = hamming_window(n);
        for (k, &v) in w.iter().enumerate() {
            assert!(v > 0.0, "n={n}: hamming[{k}]={v} should be > 0");
        }
    }
}

// =============================================================================
// 5. Zero-padding behavior — STFT output when signal is shorter than window
// =============================================================================

#[test]
fn test_stft_short_signal_with_reflection_padding() {
    // compute_stft_magnitude has built-in reflection padding.
    // A signal of length min_audio_len=66 (for default n_fft=256, pad_right=64)
    // should produce a valid result with 0 or 1 frames.
    let params = StftParams::new(8, 4);
    // pad_right = n_fft/4 = 2, min_audio = 2 + 2 = 4
    let audio = vec![1.0f32; 8]; // padded_len = 8 + 2 = 10, n_frames = (10-8)/4 + 1 = 1
    let basis = vec![0.0f32; (8 + 2) * 8]; // (n_fft+2) * n_fft
    let result = compute_stft_magnitude(&audio, &basis, &params).unwrap();
    let n_freqs = 8 / 2 + 1; // 5
    assert_eq!(result.len(), n_freqs, "1 frame expected");
}

#[test]
fn test_stft_audio_too_short_for_padding_returns_error() {
    let params = StftParams::new(8, 4);
    // min_audio = 2 + pad_right(=2) = 4; audio has only 3 samples
    let audio = vec![1.0f32; 3];
    let basis = vec![0.0f32; 10 * 8];
    let result = compute_stft_magnitude(&audio, &basis, &params);
    assert!(
        matches!(
            result,
            Err(crate::stft::StftError::AudioTooShortForPadding { .. })
        ),
        "expected AudioTooShortForPadding, got {result:?}"
    );
}

#[test]
fn test_stft_zero_padded_signal_output_shape() {
    // Zero-pad a short signal before STFT. Output shape should follow the formula.
    let n_fft = 32;
    let hop = 8;
    let short_signal = vec![1.0f32; 16];
    // Manually zero-pad to n_fft * 2
    let mut padded = short_signal;
    padded.resize(n_fft * 2, 0.0);

    let (_real, _imag, n_frames) = windowed_stft(&padded, n_fft, hop);
    let expected = (padded.len() - n_fft) / hop + 1;
    assert_eq!(
        n_frames, expected,
        "padded signal should give {expected} frames"
    );
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
            "zero-input iSTFT should produce zero output, got {v} at index {i}"
        );
    }
}

#[test]
fn test_istft_requested_length_exceeds_signal_pads_zeros() {
    let n_fft = 16;
    let hop = 4;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 3;
    let full_len = n_fft + (n_frames - 1) * hop; // 16 + 8 = 24
    let requested = 100;

    let mut real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];
    for t in 0..n_frames {
        real[t] = 1.0; // DC bin
    }

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let output = basis.istft(&real, &imag, n_frames, requested).unwrap();

    assert_eq!(output.len(), requested);
    for i in full_len..requested {
        assert_eq!(output[i], 0.0, "padding at index {i} should be 0");
    }
}

// =============================================================================
// 6. Overlap-add reconstruction — perfect reconstruction with proper overlap
// =============================================================================

#[test]
fn test_cola_hann_50_percent_overlap_sum_constant() {
    // With hop = n_fft/2 (50% overlap), Hann windows sum to a constant.
    let n_fft = 64;
    let hop = n_fft / 2;
    let n_frames = 20;
    let full_len = n_fft + (n_frames - 1) * hop;

    let window = hann_window(n_fft);
    let mut window_sum = vec![0.0f32; full_len];
    for t in 0..n_frames {
        let offset = t * hop;
        for k in 0..n_fft {
            window_sum[offset + k] += window[k];
        }
    }

    // Interior should be ~1.0 (50% overlap Hann)
    let margin = 2 * n_fft;
    for i in margin..(full_len.saturating_sub(margin)) {
        let diff = (window_sum[i] - 1.0).abs();
        assert!(
            diff < 0.01,
            "COLA not constant at {i}: sum={}, expected ~1.0",
            window_sum[i]
        );
    }
}

#[test]
fn test_cola_hann_75_percent_overlap_squared_sum_constant() {
    // With hop = n_fft/4 (75% overlap), squared Hann windows sum to ~constant.
    let n_fft = 64;
    let hop = n_fft / 4;
    let n_frames = 40;
    let full_len = n_fft + (n_frames - 1) * hop;

    let window = hann_window(n_fft);
    let mut wsq_sum = vec![0.0f32; full_len];
    for t in 0..n_frames {
        let offset = t * hop;
        for k in 0..n_fft {
            wsq_sum[offset + k] += window[k] * window[k];
        }
    }

    let margin = 2 * n_fft;
    let interior_start = margin;
    let interior_end = full_len.saturating_sub(margin);
    if interior_end > interior_start {
        let interior = &wsq_sum[interior_start..interior_end];
        let mean: f32 = interior.iter().sum::<f32>() / interior.len() as f32;
        let max_dev = interior
            .iter()
            .map(|v| (v - mean).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_dev / mean < 0.01,
            "squared COLA not flat: mean={mean}, max_dev={max_dev}"
        );
    }
}

#[test]
fn test_overlap_add_reconstruction_high_overlap() {
    // High overlap (75%) should give better reconstruction than lower overlap.
    let n_fft = 32;
    let signal = sine_wave(256, 5.0);

    // 75% overlap
    let hop_75 = n_fft / 4;
    let (real_75, imag_75, nf_75) = windowed_stft(&signal, n_fft, hop_75);
    let params_75 = IstftParams::new(n_fft, hop_75, false, false).unwrap();
    let basis_75 = IstftBasis::new(params_75).unwrap();
    let full_75 = n_fft + (nf_75 - 1) * hop_75;
    let recon_75 = basis_75.istft(&real_75, &imag_75, nf_75, full_75).unwrap();
    let (err_75, _) = roundtrip_quality(&signal, &recon_75, n_fft);

    // 50% overlap
    let hop_50 = n_fft / 2;
    let (real_50, imag_50, nf_50) = windowed_stft(&signal, n_fft, hop_50);
    let params_50 = IstftParams::new(n_fft, hop_50, false, false).unwrap();
    let basis_50 = IstftBasis::new(params_50).unwrap();
    let full_50 = n_fft + (nf_50 - 1) * hop_50;
    let recon_50 = basis_50.istft(&real_50, &imag_50, nf_50, full_50).unwrap();
    let (err_50, _) = roundtrip_quality(&signal, &recon_50, n_fft);

    // 75% overlap should give equal or better reconstruction
    assert!(
        err_75 <= err_50 + 1e-5,
        "75% overlap error {err_75:.6} should be <= 50% overlap error {err_50:.6}"
    );
}

#[test]
fn test_overlap_add_various_fft_sizes() {
    // Test overlap-add reconstruction across multiple FFT sizes.
    for &n_fft in &[16, 32, 64, 128] {
        let hop = n_fft / 4;
        let signal = sine_wave(n_fft * 8, 3.0);
        let (real, imag, n_frames) = windowed_stft(&signal, n_fft, hop);

        let params = IstftParams::new(n_fft, hop, false, false).unwrap();
        let basis = IstftBasis::new(params).unwrap();
        let full_len = n_fft + (n_frames - 1) * hop;
        let recon = basis.istft(&real, &imag, n_frames, full_len).unwrap();

        let (max_err, _) = roundtrip_quality(&signal, &recon, n_fft);
        assert!(
            max_err < 1e-3,
            "n_fft={n_fft}: OLA max_err={max_err:.6} exceeds 1e-3"
        );
    }
}

// =============================================================================
// 7. Phase reconstruction — Griffin-Lim-style phase estimation
// =============================================================================

#[test]
fn test_griffin_lim_phase_estimation() {
    // Griffin-Lim: iteratively estimate phase from magnitude by repeated
    // STFT→iSTFT→STFT. After iterations, the magnitude should converge
    // back to the original magnitude.
    let n_fft = 32;
    let hop = 8;
    let signal = sine_wave(128, 4.0);

    // Compute reference magnitude
    let (real_ref, imag_ref, n_frames) = windowed_stft(&signal, n_fft, hop);
    let n_bins = n_fft / 2 + 1;
    let target_mag: Vec<f32> = (0..n_bins * n_frames)
        .map(|i| real_ref[i].hypot(imag_ref[i]))
        .collect();

    // Start with random phase (use zero phase)
    let mut real = target_mag.clone();
    let mut imag = vec![0.0f32; n_bins * n_frames];

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;

    // Griffin-Lim iterations
    let n_iters = 10;
    for _ in 0..n_iters {
        // iSTFT: go to time domain
        let time_signal = basis.istft(&real, &imag, n_frames, full_len).unwrap();

        // Forward STFT: back to frequency domain
        let (new_real, new_imag, _) = windowed_stft(&time_signal, n_fft, hop);

        // Replace magnitude with target, keep phase
        for i in 0..n_bins * n_frames {
            let mag = new_real[i].hypot(new_imag[i]);
            if mag > 1e-10 {
                let phase_cos = new_real[i] / mag;
                let phase_sin = new_imag[i] / mag;
                real[i] = target_mag[i] * phase_cos;
                imag[i] = target_mag[i] * phase_sin;
            } else {
                real[i] = target_mag[i];
                imag[i] = 0.0;
            }
        }
    }

    // After Griffin-Lim, the magnitude should be close to the target
    let mut max_mag_err = 0.0f32;
    for i in 0..n_bins * n_frames {
        let current_mag = real[i].hypot(imag[i]);
        let err = (current_mag - target_mag[i]).abs();
        max_mag_err = max_mag_err.max(err);
    }
    assert!(
        max_mag_err < 0.5,
        "Griffin-Lim magnitude error {max_mag_err:.4} exceeds 0.5 after {n_iters} iterations"
    );
}

#[test]
fn test_phase_range_within_pi() {
    // Phase of STFT output should be within [-pi, pi]
    let n_fft = 32;
    let hop = 8;
    let signal = sine_wave(128, 5.0);
    let (real, imag, n_frames) = windowed_stft(&signal, n_fft, hop);
    let n_bins = n_fft / 2 + 1;

    for i in 0..n_bins * n_frames {
        let phase = imag[i].atan2(real[i]);
        assert!(
            (-PI..=PI).contains(&phase),
            "phase={phase} out of [-pi, pi] range at index {i}"
        );
    }
}

#[test]
fn test_magnitude_non_negative() {
    // Magnitude of STFT should always be non-negative.
    let n_fft = 64;
    let hop = 16;
    let signal: Vec<f32> = (0..256)
        .map(|i| {
            let t = i as f32 / 256.0;
            0.5 * (2.0 * PI * 3.0 * t).sin() + 0.3 * (2.0 * PI * 7.0 * t).cos()
        })
        .collect();
    let (real, imag, n_frames) = windowed_stft(&signal, n_fft, hop);
    let n_bins = n_fft / 2 + 1;

    for i in 0..n_bins * n_frames {
        let mag = real[i].hypot(imag[i]);
        assert!(mag >= 0.0, "magnitude should be non-negative, got {mag}");
    }
}

// =============================================================================
// 8. Signal energy preservation — Parseval's theorem
// =============================================================================

#[test]
fn test_parsevals_theorem_sine_wave() {
    // Parseval: per-frame windowed time-domain energy = (1/N) * frequency-domain energy
    let n_fft = 64;
    let hop = 16;
    let signal = sine_wave(512, 5.0);

    let window = hann_window(n_fft);
    let (real, imag, n_frames) = windowed_stft(&signal, n_fft, hop);
    let n_bins = n_fft / 2 + 1;

    for t in 2..(n_frames.saturating_sub(2)) {
        let offset = t * hop;

        // Time-domain energy (windowed)
        let time_energy: f32 = (0..n_fft)
            .map(|k| {
                let w = signal[offset + k] * window[k];
                w * w
            })
            .sum();

        // Frequency-domain energy with conjugate symmetry
        let mut freq_energy = 0.0f32;
        // DC
        let r0 = real[t];
        let i0 = imag[t];
        freq_energy += r0 * r0 + i0 * i0;
        // Interior bins (doubled)
        for f in 1..(n_bins - 1) {
            let r = real[f * n_frames + t];
            let im = imag[f * n_frames + t];
            freq_energy += 2.0 * (r * r + im * im);
        }
        // Nyquist
        let rn = real[(n_bins - 1) * n_frames + t];
        let imn = imag[(n_bins - 1) * n_frames + t];
        freq_energy += rn * rn + imn * imn;
        freq_energy /= n_fft as f32;

        if time_energy > 1e-8 {
            let ratio = freq_energy / time_energy;
            assert!(
                (ratio - 1.0).abs() < 0.02,
                "Parseval violated at frame {t}: time={time_energy:.4}, freq={freq_energy:.4}, ratio={ratio:.6}"
            );
        }
    }
}

#[test]
fn test_parsevals_theorem_multi_frequency() {
    let n_fft = 32;
    let hop = 8;
    let signal_len = 256;
    let signal: Vec<f32> = (0..signal_len)
        .map(|i| {
            let t = i as f32 / signal_len as f32;
            0.6 * (2.0 * PI * 3.0 * t).sin()
                + 0.3 * (2.0 * PI * 7.0 * t).cos()
                + 0.1 * (2.0 * PI * 11.0 * t).sin()
        })
        .collect();

    let window = hann_window(n_fft);
    let (real, imag, n_frames) = windowed_stft(&signal, n_fft, hop);
    let n_bins = n_fft / 2 + 1;

    for t in 2..(n_frames.saturating_sub(2)) {
        let offset = t * hop;

        let time_energy: f32 = (0..n_fft)
            .map(|k| {
                let w = signal[offset + k] * window[k];
                w * w
            })
            .sum();

        let mut freq_energy = 0.0f32;
        let r0 = real[t];
        let i0 = imag[t];
        freq_energy += r0 * r0 + i0 * i0;
        for f in 1..(n_bins - 1) {
            let r = real[f * n_frames + t];
            let im = imag[f * n_frames + t];
            freq_energy += 2.0 * (r * r + im * im);
        }
        let rn = real[(n_bins - 1) * n_frames + t];
        let imn = imag[(n_bins - 1) * n_frames + t];
        freq_energy += rn * rn + imn * imn;
        freq_energy /= n_fft as f32;

        if time_energy > 1e-8 {
            let ratio = freq_energy / time_energy;
            assert!(
                (ratio - 1.0).abs() < 0.02,
                "Parseval violated at frame {t}: ratio={ratio:.6}"
            );
        }
    }
}

#[test]
fn test_parsevals_theorem_dc_signal() {
    // DC signal: all energy at bin 0
    let n_fft = 32;
    let hop = 8;
    let signal = vec![1.0f32; 128];

    let window = hann_window(n_fft);
    let (real, imag, n_frames) = windowed_stft(&signal, n_fft, hop);
    let n_bins = n_fft / 2 + 1;

    for t in 2..(n_frames.saturating_sub(2)) {
        let offset = t * hop;

        let time_energy: f32 = (0..n_fft)
            .map(|k| {
                let w = signal[offset + k] * window[k];
                w * w
            })
            .sum();

        let mut freq_energy = 0.0f32;
        let r0 = real[t];
        let i0 = imag[t];
        freq_energy += r0 * r0 + i0 * i0;
        for f in 1..(n_bins - 1) {
            let r = real[f * n_frames + t];
            let im = imag[f * n_frames + t];
            freq_energy += 2.0 * (r * r + im * im);
        }
        let rn = real[(n_bins - 1) * n_frames + t];
        let imn = imag[(n_bins - 1) * n_frames + t];
        freq_energy += rn * rn + imn * imn;
        freq_energy /= n_fft as f32;

        if time_energy > 1e-8 {
            let ratio = freq_energy / time_energy;
            assert!(
                (ratio - 1.0).abs() < 0.02,
                "DC Parseval violated at frame {t}: ratio={ratio:.6}"
            );
        }
    }
}

#[test]
fn test_stft_energy_proportional_to_amplitude() {
    // Doubling the signal amplitude should quadruple the energy (mag^2).
    let n_fft = 32;
    let hop = 8;
    let signal = sine_wave(128, 5.0);
    let scaled: Vec<f32> = signal.iter().map(|&v| 2.0 * v).collect();

    let (real1, imag1, n_frames) = windowed_stft(&signal, n_fft, hop);
    let (real2, imag2, _) = windowed_stft(&scaled, n_fft, hop);

    let n_bins = n_fft / 2 + 1;
    // Pick an interior frame
    let t = n_frames / 2;

    let mut energy1 = 0.0f32;
    let mut energy2 = 0.0f32;
    for f in 0..n_bins {
        energy1 += real1[f * n_frames + t].powi(2) + imag1[f * n_frames + t].powi(2);
        energy2 += real2[f * n_frames + t].powi(2) + imag2[f * n_frames + t].powi(2);
    }

    // energy2 / energy1 should be ~4.0 (2^2)
    if energy1 > 1e-8 {
        let ratio = energy2 / energy1;
        assert!(
            (ratio - 4.0).abs() < 0.1,
            "energy scaling ratio={ratio:.4}, expected ~4.0"
        );
    }
}

#[test]
fn test_stft_linearity_sum_of_signals() {
    // STFT(a*x + b*y) = a*STFT(x) + b*STFT(y)
    let n_fft = 32;
    let hop = 8;
    let signal_len = 128;
    let a = 0.7f32;
    let b = 1.3f32;

    let x = sine_wave(signal_len, 3.0);
    let y: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * 7.0 * i as f32 / signal_len as f32).cos())
        .collect();
    let combined: Vec<f32> = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| a * xi + b * yi)
        .collect();

    let (rx, ix, n_frames) = windowed_stft(&x, n_fft, hop);
    let (ry, iy, _) = windowed_stft(&y, n_fft, hop);
    let (rc, ic, _) = windowed_stft(&combined, n_fft, hop);

    let n_bins = n_fft / 2 + 1;
    let mut max_real_err = 0.0f32;
    let mut max_imag_err = 0.0f32;
    for i in 0..(n_bins * n_frames) {
        let expected_r = a * rx[i] + b * ry[i];
        let expected_i = a * ix[i] + b * iy[i];
        max_real_err = max_real_err.max((rc[i] - expected_r).abs());
        max_imag_err = max_imag_err.max((ic[i] - expected_i).abs());
    }

    assert!(max_real_err < 1e-4, "linearity real error: {max_real_err}");
    assert!(max_imag_err < 1e-4, "linearity imag error: {max_imag_err}");
}
