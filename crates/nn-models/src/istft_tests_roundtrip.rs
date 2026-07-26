// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Round-trip, output-length, COLA, and model-parameter iSTFT tests.

use super::*;

/// Helper: compute forward STFT (real + imag) for round-trip testing.
/// Returns (real, imag) each of shape [n_bins, n_frames] row-major.
fn forward_stft(signal: &[f32], n_fft: usize, hop: usize) -> (Vec<f32>, Vec<f32>) {
    let n_bins = n_fft / 2 + 1;
    let n_frames = if signal.len() >= n_fft {
        (signal.len() - n_fft) / hop + 1
    } else {
        0
    };

    let mut real = vec![0.0f32; n_bins * n_frames];
    let mut imag = vec![0.0f32; n_bins * n_frames];

    // Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
        .collect();

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

    (real, imag)
}

#[test]
fn test_round_trip_sine_wave() {
    let n_fft = 64;
    let hop = 16;
    let freq = 3.0;
    let signal_len = 256;

    let signal: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * freq * i as f32 / signal_len as f32).sin())
        .collect();

    let (real, imag) = forward_stft(&signal, n_fft, hop);
    let n_frames = (signal_len - n_fft) / hop + 1;

    let params = IstftParams {
        n_fft,
        hop_length: hop,
        normalized: false,
        center: false,
    };
    let basis = IstftBasis::new(params).unwrap();

    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let margin = n_fft / 2;
    let mut max_err = 0.0f32;
    for i in margin..(full_len - margin) {
        if i < signal_len {
            let err = (reconstructed[i] - signal[i]).abs();
            max_err = max_err.max(err);
        }
    }

    assert!(
        max_err < 0.05,
        "round-trip max error in interior = {max_err}, expected < 0.05"
    );
}

#[test]
fn test_round_trip_dc_signal() {
    let n_fft = 16;
    let hop = 4;
    let signal = vec![1.0f32; 64];

    let (real, imag) = forward_stft(&signal, n_fft, hop);
    let n_frames = (64 - n_fft) / hop + 1;

    let params = IstftParams {
        n_fft,
        hop_length: hop,
        normalized: false,
        center: false,
    };
    let basis = IstftBasis::new(params).unwrap();

    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let margin = n_fft / 2;
    for (i, &val) in reconstructed
        .iter()
        .enumerate()
        .take((full_len - margin).min(signal.len()))
        .skip(margin)
    {
        assert!(
            (val - 1.0).abs() < 0.05,
            "DC round-trip: sample[{i}] = {val}, expected ~1.0",
        );
    }
}

#[test]
fn test_round_trip_normalized() {
    let n_fft = 32;
    let hop = 8;
    let signal_len = 128;

    let signal: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * 5.0 * i as f32 / signal_len as f32).sin())
        .collect();

    let n_bins = n_fft / 2 + 1;
    let n_frames = (signal_len - n_fft) / hop + 1;
    let norm_factor = 1.0 / (n_fft as f32).sqrt();

    let window: Vec<f32> = (0..n_fft)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
        .collect();

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
            real[f * n_frames + t] = r * norm_factor;
            imag[f * n_frames + t] = im * norm_factor;
        }
    }

    let params = IstftParams {
        n_fft,
        hop_length: hop,
        normalized: true,
        center: false,
    };
    let basis = IstftBasis::new(params).unwrap();

    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let margin = n_fft / 2;
    let mut max_err = 0.0f32;
    for i in margin..(full_len - margin).min(signal_len) {
        let err = (reconstructed[i] - signal[i]).abs();
        max_err = max_err.max(err);
    }

    assert!(
        max_err < 0.05,
        "normalized round-trip max error = {max_err}, expected < 0.05"
    );
}

// ---- Center trimming test ----

#[test]
fn test_center_trim() {
    let n_fft = 16;
    let hop = 4;
    let n_frames = 5;
    let n_bins = n_fft / 2 + 1;

    let params = IstftParams {
        n_fft,
        hop_length: hop,
        normalized: false,
        center: true,
    };
    let basis = IstftBasis::new(params).unwrap();

    let real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];

    let full_len = n_fft + (n_frames - 1) * hop;
    let trimmed_len = full_len - n_fft;
    let result = basis.istft(&real, &imag, n_frames, trimmed_len).unwrap();
    assert_eq!(result.len(), trimmed_len);
}

// ---- Output length trimming/padding test ----

#[test]
fn test_output_length_truncation() {
    let params = IstftParams {
        n_fft: 8,
        hop_length: 4,
        normalized: false,
        center: false,
    };
    let basis = IstftBasis::new(params).unwrap();
    let n_bins = basis.n_bins();
    let n_frames = 3;

    let real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];

    let result = basis.istft(&real, &imag, n_frames, 5).unwrap();
    assert_eq!(result.len(), 5);
}

#[test]
fn test_output_length_padding() {
    let params = IstftParams {
        n_fft: 8,
        hop_length: 4,
        normalized: false,
        center: false,
    };
    let basis = IstftBasis::new(params).unwrap();
    let n_bins = basis.n_bins();
    let n_frames = 2;

    let real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];

    let full_len = 8 + (n_frames - 1) * 4;
    let result = basis.istft(&real, &imag, n_frames, full_len + 10).unwrap();
    assert_eq!(result.len(), full_len + 10);

    for (i, &val) in result.iter().enumerate().skip(full_len).take(10) {
        assert_eq!(val, 0.0, "padded sample[{i}] should be 0.0");
    }
}

// ---- COLA condition test ----

#[test]
fn test_cola_condition_holds() {
    let n_fft = 64;
    let hop = 16;
    let n_frames = 10;
    let full_len = n_fft + (n_frames - 1) * hop;

    let window: Vec<f32> = (0..n_fft)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
        .collect();

    let mut window_sum = vec![0.0f32; full_len];
    for t in 0..n_frames {
        let offset = t * hop;
        for k in 0..n_fft {
            window_sum[offset + k] += window[k] * window[k];
        }
    }

    let margin = n_fft / 2;
    for (i, &ws) in window_sum
        .iter()
        .enumerate()
        .take(full_len - margin)
        .skip(margin)
    {
        assert!(
            ws > 1e-6,
            "COLA condition violated at position {i}: window_sum = {ws}",
        );
    }
}

// ---- Known-value test with simple impulse ----

#[test]
fn test_istft_single_frame_impulse() {
    let n_fft = 8;
    let hop = 4;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 1;

    let params = IstftParams {
        n_fft,
        hop_length: hop,
        normalized: false,
        center: false,
    };
    let basis = IstftBasis::new(params).unwrap();

    let mut real = vec![0.0f32; n_bins * n_frames];
    real[0] = 1.0;
    let imag = vec![0.0f32; n_bins * n_frames];

    let result = basis.istft(&real, &imag, n_frames, n_fft).unwrap();

    let expected_peak = 1.0 / n_fft as f32;
    assert!(
        (result[n_fft / 2] - expected_peak).abs() < 1e-6,
        "peak: expected {expected_peak}, got {}",
        result[n_fft / 2]
    );
}

// ---- HTDemucs parameters test ----

#[test]
fn test_htdemucs_params_construction() {
    let params = IstftParams {
        n_fft: 4096,
        hop_length: 1024,
        normalized: true,
        center: false,
    };
    let basis = IstftBasis::new(params).unwrap();
    assert_eq!(basis.n_bins(), 2049);

    let n_bins = basis.n_bins();
    let n_frames = 2;
    let real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];

    let full_len = 4096 + (n_frames - 1) * 1024;
    let result = basis.istft(&real, &imag, n_frames, full_len).unwrap();
    assert_eq!(result.len(), full_len);
}

// ---- Kokoro parameters test ----

#[test]
fn test_kokoro_params_construction() {
    // Kokoro does NOT use center trimming. center=false matches
    // the kokoro_istft implementation (which hardcodes no center trim).
    let params = IstftParams {
        n_fft: 20,
        hop_length: 5,
        normalized: false,
        center: false,
    };
    let basis = IstftBasis::new(params).unwrap();
    assert_eq!(basis.n_bins(), 11);

    let n_bins = basis.n_bins();
    let n_frames = 4;
    let real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];

    let full_len = 20 + 3 * 5;
    let result = basis.istft(&real, &imag, n_frames, full_len).unwrap();
    assert_eq!(result.len(), full_len);
}
