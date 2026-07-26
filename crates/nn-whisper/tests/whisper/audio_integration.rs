// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Whisper audio preprocessing.
//!
//! Covers edge cases not tested by inline `audio_tests.rs`:
//! short audio, NaN/Inf propagation, determinism, silence normalization
//! values, and whisper_mel_spectrogram with non-trivial content.

use nn_whisper::audio::{mel_filterbank, pcm_to_mel, whisper_mel_spectrogram};

// ---------------------------------------------------------------------------
// Short audio edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_pcm_to_mel_single_sample() {
    // 1 sample: the shortest possible valid input.
    // reflect-pad uses clamped index i.min(audio.len()-1) = i.min(0) = 0 for all.
    let audio = [0.5f32];
    let filters = mel_filterbank(128, 400, 16000);
    let mel = pcm_to_mel(&audio, &filters, 400, 160, 128).unwrap();
    let dims = mel.dims();
    assert_eq!(dims[0], 1, "batch");
    assert_eq!(dims[1], 128, "n_mels");
    // n_frames = (1 + 400 - 400) / 160 + 1 = 1
    assert_eq!(dims[2], 1, "n_frames for 1 sample");
    let vals = mel.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "single sample output must be finite"
    );
}

#[test]
fn test_pcm_to_mel_two_samples() {
    // 2 samples: left reflect pad = 200 values, right reflect pad = 200 values.
    // With only 2 samples, most pad values come from clamped indices.
    let audio = [0.3f32, -0.7];
    let filters = mel_filterbank(128, 400, 16000);
    let mel = pcm_to_mel(&audio, &filters, 400, 160, 128).unwrap();
    let dims = mel.dims();
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 128);
    assert!(dims[2] >= 1, "should produce at least 1 frame");
    let vals = mel.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()));
}

#[test]
fn test_pcm_to_mel_shorter_than_half_nfft() {
    // audio.len() = 100 < n_fft/2 = 200.
    // Left reflect-pad clamps: audio[i.min(99)] for i in 0..200.
    let audio: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01).collect();
    let filters = mel_filterbank(128, 400, 16000);
    let mel = pcm_to_mel(&audio, &filters, 400, 160, 128).unwrap();
    let dims = mel.dims();
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 128);
    assert!(dims[2] >= 1);
    let vals = mel.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()));
}

#[test]
fn test_pcm_to_mel_exactly_nfft_samples() {
    // audio.len() = 400 = n_fft. After padding: 400 + 2*200 = 800.
    // n_frames = (800 - 400) / 160 + 1 = 3.
    let audio = vec![0.1f32; 400];
    let filters = mel_filterbank(128, 400, 16000);
    let mel = pcm_to_mel(&audio, &filters, 400, 160, 128).unwrap();
    let dims = mel.dims();
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 128);
    assert_eq!(dims[2], 3, "n_fft samples should produce 3 frames");
}

// ---------------------------------------------------------------------------
// NaN / Inf propagation
// ---------------------------------------------------------------------------

#[test]
fn test_pcm_to_mel_rejects_nan_input() {
    // pcm_to_mel rejects NaN input at entry (#1491). Previously, NaN was
    // silently absorbed by the log floor (NaN.max(1e-10) → 1e-10 in Rust).
    let mut audio = vec![0.0f32; 1600];
    audio[800] = f32::NAN;
    let filters = mel_filterbank(128, 400, 16000);
    let err = pcm_to_mel(&audio, &filters, 400, 160, 128).unwrap_err();
    let msg = format!("{err}").to_lowercase();
    assert!(
        msg.contains("non-finite"),
        "expected non-finite error, got: {err}"
    );
}

#[test]
fn test_pcm_to_mel_rejects_inf_input() {
    // pcm_to_mel rejects Inf input at entry (#1491). Previously, Inf was
    // converted to NaN by DFT arithmetic, then silently absorbed by the log floor.
    let mut audio = vec![0.0f32; 1600];
    audio[800] = f32::INFINITY;
    let filters = mel_filterbank(128, 400, 16000);
    let err = pcm_to_mel(&audio, &filters, 400, 160, 128).unwrap_err();
    let msg = format!("{err}").to_lowercase();
    assert!(
        msg.contains("non-finite"),
        "expected non-finite error, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Silence normalization exact values
// ---------------------------------------------------------------------------

#[test]
fn test_pcm_to_mel_silence_normalization_value() {
    // Silence: all power = 0 → log10(max(0, 1e-10)) = log10(1e-10) = -10.
    // max_val = -10, floor = -10 - 8 = -18. All values are -10 (not below floor).
    // After affine: (-10 + 4) / 4 = -1.5.
    let audio = vec![0.0f32; 16000];
    let filters = mel_filterbank(128, 400, 16000);
    let mel = pcm_to_mel(&audio, &filters, 400, 160, 128).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();

    // All values should be exactly -1.5 for silence.
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - (-1.5)).abs() < 1e-5,
            "silence mel[{i}] = {v}, expected -1.5"
        );
    }
}

// ---------------------------------------------------------------------------
// whisper_mel_spectrogram with content
// ---------------------------------------------------------------------------

#[test]
fn test_whisper_mel_spectrogram_sine_wave() {
    // 1 kHz sine wave through the convenience function.
    let sample_rate = 16000;
    let audio: Vec<f32> = (0..sample_rate)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
        })
        .collect();

    let mel = whisper_mel_spectrogram(&audio).unwrap();
    let dims = mel.dims();
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 128);
    // 30s padding (480000 samples) → (480000 + 400 - 400) / 160 + 1 = 3001, clipped to N_FRAMES = 3000
    assert_eq!(dims[2], 3000);

    let vals = mel.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()));

    // Peak band should NOT be in the highest-frequency bands (1 kHz is low).
    let n_frames = 3000;
    let band_means: Vec<f32> = (0..128)
        .map(|m| {
            let sum: f32 = (0..n_frames).map(|t| vals[m * n_frames + t]).sum();
            sum / n_frames as f32
        })
        .collect();
    let max_band = band_means
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert!(
        max_band < 80,
        "1 kHz peak at band {max_band}, expected < 80"
    );
}

#[test]
fn test_whisper_mel_spectrogram_tone_vs_silence() {
    // A tone should produce higher max mel values than silence.
    let silence = vec![0.0f32; 16000];
    let tone: Vec<f32> = (0..16000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
        .collect();

    let mel_silence = whisper_mel_spectrogram(&silence).unwrap();
    let mel_tone = whisper_mel_spectrogram(&tone).unwrap();

    let max_silence = mel_silence
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let max_tone = mel_tone
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    assert!(
        max_tone > max_silence,
        "tone max ({max_tone}) should exceed silence max ({max_silence})"
    );
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

#[test]
fn test_pcm_to_mel_deterministic() {
    // Same input should always produce the same output.
    let audio: Vec<f32> = (0..3200).map(|i| (i as f32 * 0.01).sin()).collect();
    let filters = mel_filterbank(128, 400, 16000);

    let mel1 = pcm_to_mel(&audio, &filters, 400, 160, 128).unwrap();
    let mel2 = pcm_to_mel(&audio, &filters, 400, 160, 128).unwrap();

    let v1 = mel1.to_flat_vec::<f32>().unwrap();
    let v2 = mel2.to_flat_vec::<f32>().unwrap();
    assert_eq!(v1.len(), v2.len());
    for (i, (&a, &b)) in v1.iter().zip(v2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "non-deterministic at index {i}: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// mel_filterbank edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_mel_filterbank_small_nfft() {
    // Very small n_fft = 8, n_freqs = 5.
    let filters = mel_filterbank(4, 8, 16000);
    assert_eq!(filters.len(), 4 * 5);
    assert!(filters.iter().all(|&v| v >= 0.0));
}

#[test]
fn test_mel_filterbank_single_mel_bin() {
    // 1 mel bin: the triangle spans the entire frequency range.
    let filters = mel_filterbank(1, 400, 16000);
    assert_eq!(filters.len(), 201);
    // Should have at least some nonzero values.
    let sum: f32 = filters.iter().sum();
    assert!(sum > 0.0, "single mel bin should have nonzero values");
}

// ---------------------------------------------------------------------------
// pcm_to_mel with non-standard parameters
// ---------------------------------------------------------------------------

#[test]
fn test_pcm_to_mel_small_nfft_and_hop() {
    // Smaller n_fft=32, hop=16, n_mels=8 — verifies the pipeline works
    // with non-Whisper parameters.
    let audio: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
    let n_fft = 32;
    let hop = 16;
    let n_mels = 8;
    let filters = mel_filterbank(n_mels, n_fft, 16000);
    let mel = pcm_to_mel(&audio, &filters, n_fft, hop, n_mels).unwrap();
    let dims = mel.dims();
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], n_mels);
    // n_frames = (256 + 32 - 32) / 16 + 1 = 17
    assert_eq!(dims[2], 17);
    let vals = mel.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()));
}

#[test]
fn test_pcm_to_mel_80_mels() {
    // Whisper v1/v2 use 80 mel bins.
    let audio = vec![0.0f32; 16000];
    let filters = mel_filterbank(80, 400, 16000);
    let mel = pcm_to_mel(&audio, &filters, 400, 160, 80).unwrap();
    let dims = mel.dims();
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 80);
    assert_eq!(dims[2], 101);
}
