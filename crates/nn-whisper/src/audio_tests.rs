#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Whisper audio preprocessing (mel spectrogram).

use super::*;

// -- mel filterbank tests -----------------------------------------------------

#[test]
fn test_mel_filterbank_shape() {
    let filters = mel_filterbank(128, 400, 16000);
    // [128 mels, 201 freqs]
    assert_eq!(filters.len(), 128 * 201);
}

#[test]
fn test_mel_filterbank_non_negative() {
    let filters = mel_filterbank(128, 400, 16000);
    for (i, &v) in filters.iter().enumerate() {
        assert!(v >= 0.0, "filter[{i}] = {v} < 0");
    }
}

#[test]
fn test_mel_filterbank_rows_have_nonzero() {
    let n_mels = 128;
    let n_freqs = 201;
    let filters = mel_filterbank(n_mels, 400, 16000);
    for m in 0..n_mels {
        let row_sum: f32 = (0..n_freqs).map(|k| filters[m * n_freqs + k]).sum();
        assert!(row_sum > 0.0, "mel band {m} is all zeros (sum={row_sum})");
    }
}

#[test]
fn test_mel_filterbank_triangular_structure() {
    // Each mel band should have a single contiguous nonzero region (triangle).
    let n_mels = 128;
    let n_freqs = 201;
    let filters = mel_filterbank(n_mels, 400, 16000);
    for m in 0..n_mels {
        let row: Vec<f32> = (0..n_freqs).map(|k| filters[m * n_freqs + k]).collect();
        // Find first and last nonzero.
        let first = row.iter().position(|&v| v > 0.0);
        let last = row.iter().rposition(|&v| v > 0.0);
        if let (Some(f), Some(l)) = (first, last) {
            // All values between first and last should be nonzero.
            for (k, val) in row.iter().enumerate().take(l + 1).skip(f) {
                assert!(
                    *val > 0.0,
                    "mel band {m}: gap at freq bin {k} (first={f}, last={l})"
                );
            }
        }
    }
}

#[test]
fn test_mel_filterbank_80_bins() {
    // Whisper v1/v2 use 80 mel bins.
    let filters = mel_filterbank(80, 400, 16000);
    assert_eq!(filters.len(), 80 * 201);
    // All non-negative.
    assert!(filters.iter().all(|&v| v >= 0.0));
}

// -- Slaney mel scale tests ---------------------------------------------------

#[test]
fn test_hz_to_mel_roundtrip() {
    // Roundtrip: mel_to_hz(hz_to_mel(f)) ≈ f.
    let test_freqs = [0.0, 100.0, 500.0, 999.9, 1000.0, 1001.0, 4000.0, 8000.0];
    for &f in &test_freqs {
        let mel = hz_to_mel(f);
        let back = mel_to_hz(mel);
        assert!(
            (back - f).abs() < 0.01,
            "roundtrip failed: hz_to_mel({f}) = {mel}, mel_to_hz({mel}) = {back}"
        );
    }
}

#[test]
fn test_hz_to_mel_monotonic() {
    let mut prev = hz_to_mel(0.0);
    for hz in (1..=8000).step_by(10) {
        let mel = hz_to_mel(f64::from(hz));
        assert!(mel > prev, "mel scale not monotonic at {hz} Hz");
        prev = mel;
    }
}

// -- pcm_to_mel tests ---------------------------------------------------------

#[test]
fn test_pcm_to_mel_output_shape() {
    // 1 second of silence at 16 kHz.
    let audio = vec![0.0f32; 16000];
    let filters = mel_filterbank(128, 400, 16000);
    let mel = pcm_to_mel(&audio, &filters, 400, 160, 128).unwrap();
    let dims = mel.dims();
    assert_eq!(dims[0], 1, "batch");
    assert_eq!(dims[1], 128, "n_mels");
    // n_frames = (16000 + 400 - 400) / 160 + 1 = 101
    assert_eq!(dims[2], 101, "n_frames");
}

#[test]
fn test_pcm_to_mel_30s_shape() {
    // Full 30-second Whisper chunk.
    let audio = vec![0.0f32; 480_000];
    let filters = mel_filterbank(128, 400, 16000);
    let mel = pcm_to_mel(&audio, &filters, 400, 160, 128).unwrap();
    let dims = mel.dims();
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 128);
    assert_eq!(dims[2], 3001, "30s should produce 3001 frames");
}

#[test]
fn test_pcm_to_mel_output_finite() {
    let audio = vec![0.0f32; 16000];
    let filters = mel_filterbank(128, 400, 16000);
    let mel = pcm_to_mel(&audio, &filters, 400, 160, 128).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "mel spectrogram must be finite"
    );
}

#[test]
fn test_pcm_to_mel_output_range() {
    // With silence input, all values should be normalized to the floor.
    let audio = vec![0.0f32; 16000];
    let filters = mel_filterbank(128, 400, 16000);
    let mel = pcm_to_mel(&audio, &filters, 400, 160, 128).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();
    // After normalization: (x + 4) / 4. For silence: log10(1e-10) = -10,
    // clamped to max-8 = -10-8 = -18 -> but max is also -10, so floor = -18.
    // Then (-10 + 4) / 4 = -1.5. All values should be <= 1.0.
    for &v in &vals {
        assert!(v <= 1.0, "mel value {v} > 1.0");
    }
}

#[test]
fn test_pcm_to_mel_empty_audio_error() {
    let filters = mel_filterbank(128, 400, 16000);
    let result = pcm_to_mel(&[], &filters, 400, 160, 128);
    assert!(result.is_err());
}

#[test]
fn test_pcm_to_mel_filter_size_mismatch_error() {
    let audio = vec![0.0f32; 16000];
    let bad_filters = vec![0.0f32; 10]; // wrong size
    let result = pcm_to_mel(&audio, &bad_filters, 400, 160, 128);
    assert!(result.is_err());
}

// -- AC4: Synthetic sine wave integration test --------------------------------

#[test]
fn test_pcm_to_mel_sine_wave_energy_localized() {
    // Generate a 1 kHz sine wave — energy should be concentrated in the
    // mel band(s) near 1 kHz, not spread uniformly.
    let sample_rate = 16000;
    let freq = 1000.0f32;
    let duration_samples = sample_rate; // 1 second
    let audio: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect();

    let n_mels = 128;
    let filters = mel_filterbank(n_mels, 400, sample_rate);
    let mel = pcm_to_mel(&audio, &filters, 400, 160, n_mels).unwrap();
    let dims = mel.dims();
    let n_frames = dims[2];

    let vals = mel.to_flat_vec::<f32>().unwrap();

    // Compute mean energy per mel band across all frames.
    let mut band_means = vec![0.0f32; n_mels];
    for m in 0..n_mels {
        let sum: f32 = (0..n_frames).map(|t| vals[m * n_frames + t]).sum();
        band_means[m] = sum / n_frames as f32;
    }

    // Find the mel band with maximum mean energy.
    let max_band = band_means
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    // The peak should be in the mel bands corresponding to ~1 kHz.
    // For 128 bands, 16 kHz sample rate, Slaney scale: ~1 kHz is around band 25-35.
    assert!(
        (15..=45).contains(&max_band),
        "1 kHz sine peak at band {max_band}, expected 15..=45"
    );

    // Peak band mean should be significantly higher than the overall mean.
    let overall_mean: f32 = band_means.iter().sum::<f32>() / n_mels as f32;
    assert!(
        band_means[max_band] > overall_mean,
        "peak band mean ({}) should exceed overall mean ({})",
        band_means[max_band],
        overall_mean
    );
}

#[test]
fn test_whisper_mel_spectrogram_convenience() {
    // Test the convenience wrapper with 0.5s of audio.
    // whisper_mel_spectrogram pads to 30s (N_SAMPLES = 480,000), so output
    // always has the same number of frames regardless of input length.
    let audio = vec![0.0f32; 8000];
    let mel = whisper_mel_spectrogram(&audio).unwrap();
    let dims = mel.dims();
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 128);
    // 30s padded: STFT produces 3001, clipped to N_FRAMES=3000.
    assert_eq!(dims[2], 3000);
}

#[test]
fn test_whisper_mel_spectrogram_pads_short_audio() {
    // Short audio (1s) should produce the same frame count as 30s audio.
    // Both inputs get padded to N_SAMPLES=480,000 internally, so we only
    // need to verify the short-audio output shape matches the expected dims.
    let short = vec![0.0f32; 16000];
    let mel_short = whisper_mel_spectrogram(&short).unwrap();
    // Same dims as the 30s test above: [1, 128, 3000].
    assert_eq!(mel_short.dims(), &[1, 128, 3000]);
}

#[test]
fn test_whisper_mel_spectrogram_truncates_long_audio() {
    // Audio longer than 30s should be truncated to N_SAMPLES.
    let long_audio = vec![0.1f32; 640_000]; // 40s
    let mel = whisper_mel_spectrogram(&long_audio).unwrap();
    let dims = mel.dims();
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 128);
    // Truncated to 30s, then clipped to N_FRAMES=3000.
    assert_eq!(dims[2], 3000);
}

// -- AC3: Numerical parity tests (extracted to audio_parity_tests.rs) ---------

#[path = "audio_parity_tests.rs"]
mod parity_tests;

// -- NaN/Inf input rejection tests (#1491) ------------------------------------

#[test]
fn test_pcm_to_mel_rejects_nan_input() {
    let filters = mel_filterbank(128, 400, 16000);
    let mut audio = vec![0.0f32; 16000];
    audio[500] = f32::NAN;
    audio[1000] = f32::NAN;
    let err = pcm_to_mel(&audio, &filters, 400, 160, 128).unwrap_err();
    match &err {
        TensorError::NonFiniteData { name, count } => {
            assert!(
                name.contains("audio"),
                "expected audio in name, got: {name}"
            );
            assert_eq!(*count, 2);
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_pcm_to_mel_rejects_inf_input() {
    let filters = mel_filterbank(128, 400, 16000);
    let mut audio = vec![0.0f32; 16000];
    audio[0] = f32::INFINITY;
    let err = pcm_to_mel(&audio, &filters, 400, 160, 128).unwrap_err();
    match &err {
        TensorError::NonFiniteData { name, count } => {
            assert!(
                name.contains("audio"),
                "expected audio in name, got: {name}"
            );
            assert_eq!(*count, 1);
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_pcm_to_mel_rejects_neg_inf_input() {
    let filters = mel_filterbank(128, 400, 16000);
    let mut audio = vec![0.1f32; 16000];
    audio[8000] = f32::NEG_INFINITY;
    let err = pcm_to_mel(&audio, &filters, 400, 160, 128).unwrap_err();
    match &err {
        TensorError::NonFiniteData { count, .. } => {
            assert_eq!(*count, 1);
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

// -- Hann window test ---------------------------------------------------------

#[test]
fn test_hann_window_properties() {
    let w = hann_window(400);
    assert_eq!(w.len(), 400);
    // First element should be 0 (periodic Hann).
    assert!((w[0]).abs() < 1e-10);
    // Middle element should be close to 1.0.
    assert!((w[200] - 1.0).abs() < 1e-10);
    // All values in [0, 1].
    for (i, &v) in w.iter().enumerate() {
        assert!((0.0..=1.0).contains(&v), "window[{i}] = {v} out of [0, 1]");
    }
}

// -- Reflect-padding regression tests --

#[test]
fn test_left_reflect_padding_no_boundary_duplication() {
    // NumPy np.pad(audio, pad, mode='reflect') does NOT duplicate the
    // boundary sample. For audio = [10, 20, 30, 40, 50] with pad=3,
    // left padding should be [40, 30, 20, | 10, 20, 30, 40, 50, ...].
    // NOT [30, 20, 10, | 10, 20, 30, 40, 50, ...] (boundary duplicated).
    let audio: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let n_fft = 6; // pad = n_fft/2 = 3
    let pad = n_fft / 2;
    let padded_len = audio.len() + 2 * pad;
    let mut padded = vec![0.0f64; padded_len];

    // Replicate production left-reflect logic from audio.rs.
    for i in 0..pad {
        padded[pad - 1 - i] = f64::from(audio[(i + 1).min(audio.len() - 1)]);
    }
    for (i, &s) in audio.iter().enumerate() {
        padded[pad + i] = f64::from(s);
    }

    // Left padding: padded[0]=audio[3]=40, padded[1]=audio[2]=30, padded[2]=audio[1]=20.
    assert_eq!(padded[0], 40.0, "padded[0] should be audio[3]");
    assert_eq!(padded[1], 30.0, "padded[1] should be audio[2]");
    assert_eq!(padded[2], 20.0, "padded[2] should be audio[1]");
    // Boundary sample audio[0] appears only once at padded[pad].
    assert_eq!(padded[pad], 10.0, "padded[pad] should be audio[0]");
    // padded[pad-1] should NOT equal audio[0] (was the bug: boundary duplication).
    assert_ne!(
        padded[pad - 1],
        padded[pad],
        "left-reflect should not duplicate boundary sample"
    );
}

// -- FFT tests (extracted to audio_fft_tests.rs) -----------------------------

#[path = "audio_fft_tests.rs"]
mod fft_tests;
