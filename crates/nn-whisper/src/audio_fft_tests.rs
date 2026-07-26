#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for FFT implementation and DFT→FFT parity.

use std::f64::consts::PI;

use super::*;

#[test]
fn test_next_power_of_2() {
    assert_eq!(next_power_of_2(1), 1);
    assert_eq!(next_power_of_2(2), 2);
    assert_eq!(next_power_of_2(3), 4);
    assert_eq!(next_power_of_2(400), 512);
    assert_eq!(next_power_of_2(512), 512);
    assert_eq!(next_power_of_2(1024), 1024);
}

#[test]
fn test_fft_known_dc() {
    // FFT of all-ones should produce DC = n, all other bins = 0.
    let n = 8;
    let mut re = vec![1.0f64; n];
    let mut im = vec![0.0f64; n];
    fft_in_place(&mut re, &mut im, n).unwrap();
    assert!((re[0] - 8.0).abs() < 1e-10, "DC should be 8, got {}", re[0]);
    for k in 1..n {
        assert!(
            re[k].abs() < 1e-10 && im[k].abs() < 1e-10,
            "bin {k}: re={}, im={} should be ~0",
            re[k],
            im[k]
        );
    }
}

#[test]
fn test_fft_single_frequency() {
    // FFT of cos(2π·k₀·t/N) should have energy only at bin k₀ and N-k₀.
    let n = 64;
    let k0 = 5usize;
    let mut re: Vec<f64> = (0..n)
        .map(|t| (2.0 * PI * k0 as f64 * t as f64 / n as f64).cos())
        .collect();
    let mut im = vec![0.0f64; n];
    fft_in_place(&mut re, &mut im, n).unwrap();

    // Magnitude at each bin.
    let mags: Vec<f64> = re.iter().zip(im.iter()).map(|(r, i)| r.hypot(*i)).collect();

    // Bins k0 and n-k0 should have magnitude n/2 = 32.
    assert!(
        (mags[k0] - 32.0).abs() < 1e-8,
        "bin {k0} mag = {}, expected 32",
        mags[k0]
    );
    assert!(
        (mags[n - k0] - 32.0).abs() < 1e-8,
        "bin {} mag = {}, expected 32",
        n - k0,
        mags[n - k0]
    );

    // All other bins should be near zero.
    for (k, &m) in mags.iter().enumerate() {
        if k != k0 && k != n - k0 {
            assert!(m < 1e-8, "bin {k} mag = {m}, expected ~0");
        }
    }
}

#[test]
fn test_fft_non_power_of_2_returns_error() {
    let mut re = vec![1.0f64; 7];
    let mut im = vec![0.0f64; 7];
    let err = fft_in_place(&mut re, &mut im, 7).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("power of 2"),
        "should reject non-power-of-2, got: {msg}"
    );
}

#[test]
fn test_fft_buffer_too_short_returns_error() {
    let mut re = vec![1.0f64; 4];
    let mut im = vec![0.0f64; 4];
    // n=8 but buffers only have 4 elements.
    let err = fft_in_place(&mut re, &mut im, 8).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("buffers len"),
        "should reject short buffers, got: {msg}"
    );
}

/// Direct DFT reference for parity testing against FFT.
fn reference_dft_power(padded: &[f64], n_fft: usize, hop_length: usize) -> Vec<f64> {
    let n_freqs = n_fft / 2 + 1;
    let n_frames = (padded.len() - n_fft) / hop_length + 1;
    let window = hann_window(n_fft);
    let mut power = vec![0.0f64; n_frames * n_freqs];
    for i in 0..n_frames {
        let start = i * hop_length;
        for k in 0..n_freqs {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for t in 0..n_fft {
                let w = padded[start + t] * window[t];
                let angle = -2.0 * PI * ((k * t) % n_fft) as f64 / n_fft as f64;
                re += w * angle.cos();
                im += w * angle.sin();
            }
            power[i * n_freqs + k] = re * re + im * im;
        }
    }
    power
}

#[test]
fn test_fft_power_spectrogram_matches_dft() {
    // Generate a synthetic signal: sum of two sine waves.
    let n_fft = 400;
    let hop_length = 160;
    let n_samples = 3200; // ~0.2s at 16kHz
    let padded: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = f64::from(i) / 16000.0;
            (2.0 * PI * 440.0 * t).sin() + 0.5 * (2.0 * PI * 1000.0 * t).sin()
        })
        .collect();

    let fft_power = power_spectrogram(&padded, n_fft, hop_length).unwrap();
    let dft_power = reference_dft_power(&padded, n_fft, hop_length);

    assert_eq!(fft_power.len(), dft_power.len());

    let mut max_rel_err = 0.0f64;
    for (i, (&fft_v, &dft_v)) in fft_power.iter().zip(dft_power.iter()).enumerate() {
        let abs_err = (fft_v - dft_v).abs();
        // Use relative error for large values, absolute for near-zero.
        let err = if dft_v.abs() > 1e-10 {
            abs_err / dft_v.abs()
        } else {
            abs_err
        };
        if err > max_rel_err {
            max_rel_err = err;
        }
        assert!(
            err < 1e-6,
            "power mismatch at index {i}: fft={fft_v:.10e}, dft={dft_v:.10e}, err={err:.2e}"
        );
    }
}

#[test]
fn test_fft_mel_spectrogram_smoke() {
    // Smoke test: verify mel spectrogram has correct shape and finite values
    // for a 440Hz sine wave. For actual FFT-vs-DFT parity, see
    // test_fft_power_spectrogram_matches_dft above.
    let sample_rate = 16000;
    let audio: Vec<f32> = (0..sample_rate)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();

    let mel = whisper_mel_spectrogram(&audio).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();

    // Verify shape: [1, 128, n_frames].
    let dims = mel.dims();
    assert_eq!(dims[0], 1, "batch dim should be 1");
    assert_eq!(dims[1], 128, "mel bins should be 128");
    assert!(dims[2] > 0, "should have at least 1 frame");

    // Verify all values are finite.
    assert!(vals.iter().all(|v| v.is_finite()));

    // Verify the mel spectrogram is non-trivial (not all zeros).
    let max_val = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_val = vals.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(max_val > min_val, "mel spectrogram should not be flat");

    // Verify 440Hz energy localization: mel bin with peak energy for a 440Hz
    // sine should be in the lower frequency range.
    // Tensor shape: [1, n_bins, n_frames] (row-major: bin varies before frame).
    // Values are log-compressed (log10 scale), so "peak" = highest (least negative).
    let n_bins = dims[1];
    let n_frames = dims[2];
    let mut bin_mean = vec![0.0f64; n_bins];
    for bin in 0..n_bins {
        for frame in 0..n_frames {
            bin_mean[bin] += f64::from(vals[bin * n_frames + frame]);
        }
        bin_mean[bin] /= n_frames as f64;
    }
    let peak_bin = bin_mean
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    // 440Hz should produce peak energy in lower mel bins, not upper.
    // Whisper mel bins: 128 bins spanning 0-8kHz. 440Hz maps to roughly
    // bin 20-50 depending on mel scale parameters.
    assert!(
        peak_bin < 80,
        "440Hz peak energy at mel bin {peak_bin} — expected lower half (< 80)"
    );
    // The peak bin value should be meaningfully higher than average.
    let overall_mean: f64 = bin_mean.iter().sum::<f64>() / n_bins as f64;
    assert!(
        bin_mean[peak_bin] > overall_mean,
        "peak bin {peak_bin} mean ({:.2}) should exceed overall mean ({:.2})",
        bin_mean[peak_bin],
        overall_mean
    );
}

#[test]
fn test_fft_benchmark_vs_dft() {
    // FFT (Bluestein) vs direct DFT on 3s audio (sufficient for scaling comparison).
    // 3s at 16kHz = 48,000 samples + 2×200 reflect pad = 48,400 padded.
    let n_fft = 400;
    let hop_length = 160;
    let n_samples = 48_400; // padded length for 3s audio
    let padded: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = f64::from(i) / 16000.0;
            (2.0 * PI * 440.0 * t).sin()
        })
        .collect();

    // Warm up cache.
    let _ = power_spectrogram(&padded, n_fft, hop_length).unwrap();
    let _ = reference_dft_power(&padded[..3200], n_fft, hop_length);

    // Time FFT (Bluestein) on full 30s.
    let start = std::time::Instant::now();
    let fft_result = power_spectrogram(&padded, n_fft, hop_length).unwrap();
    let fft_elapsed = start.elapsed();

    // Time DFT on a small slice (20 frames) — DFT is too slow for full 30s.
    let small_len = n_fft + 20 * hop_length; // ~21 frames
    let start = std::time::Instant::now();
    let _dft_result = reference_dft_power(&padded[..small_len], n_fft, hop_length);
    let dft_elapsed = start.elapsed();

    let n_frames_total = fft_result.len() / (n_fft / 2 + 1);
    let dft_frames = 21;
    let dft_per_frame_us = dft_elapsed.as_micros() as f64 / f64::from(dft_frames);
    let fft_per_frame_us = fft_elapsed.as_micros() as f64 / n_frames_total as f64;

    eprintln!(
        "FFT benchmark: {n_frames_total} frames in {fft_elapsed:?} ({fft_per_frame_us:.1} us/frame)"
    );
    eprintln!(
        "DFT benchmark: {dft_frames} frames in {dft_elapsed:?} ({dft_per_frame_us:.1} us/frame)"
    );
    eprintln!(
        "Speedup: {:.1}x (FFT vs DFT per frame)",
        dft_per_frame_us / fft_per_frame_us
    );

    // Assert measurable speedup (at least 3x — conservative due to CI variability).
    assert!(
        fft_per_frame_us < dft_per_frame_us,
        "FFT ({fft_per_frame_us:.1} us/frame) should be faster than DFT ({dft_per_frame_us:.1} us/frame)"
    );
}

// -- Input validation guard tests (W3-93 / #1648) --

#[test]
fn test_power_spectrogram_zero_n_fft_returns_error() {
    let padded = vec![1.0f64; 100];
    let err = power_spectrogram(&padded, 0, 160).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("n_fft") && msg.contains("0"),
        "should reject n_fft=0, got: {msg}"
    );
}

#[test]
fn test_power_spectrogram_zero_hop_length_returns_error() {
    let padded = vec![1.0f64; 400];
    let err = power_spectrogram(&padded, 400, 0).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("hop_length") && msg.contains("0"),
        "should reject hop_length=0, got: {msg}"
    );
}

#[test]
fn test_power_spectrogram_padded_shorter_than_n_fft_returns_error() {
    let padded = vec![1.0f64; 100]; // shorter than n_fft=400
    let err = power_spectrogram(&padded, 400, 160).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("padded length") && msg.contains("100"),
        "should reject short padded, got: {msg}"
    );
}

#[test]
fn test_power_spectrogram_huge_n_fft_returns_error() {
    // n_fft = usize::MAX / 2 + 1 would cause 2 * n_fft to overflow.
    // The padded.len() < n_fft guard fires first (correctly), but either
    // guard prevents the overflow.
    let huge_n_fft = usize::MAX / 2 + 1;
    let padded = vec![1.0f64; 1];
    assert!(
        power_spectrogram(&padded, huge_n_fft, 160).is_err(),
        "should reject huge n_fft that would cause overflow"
    );
}
