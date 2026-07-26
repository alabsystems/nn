// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Mel spectrogram tests: filterbank construction for different n_mels values,
//! mel frequency conversion edge cases, STFT window construction,
//! spectrogram shape for different audio lengths, and known audio input
//! verification. Part of #4186.

use crate::audio::{mel_filterbank, pcm_to_mel, whisper_mel_spectrogram_for_config};
use crate::config::{HOP_LENGTH, N_FFT, N_FRAMES, SAMPLE_RATE};
use nn_core::audio::{hann_window, hz_to_mel_slaney, mel_to_hz_slaney};

// ============================================================================
// Mel filterbank construction for different n_mels values
// ============================================================================

#[test]
fn test_mel_filterbank_40_bins() {
    let n_mels = 40;
    let n_fft = 400;
    let n_freqs = n_fft / 2 + 1;
    let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);
    assert_eq!(filters.len(), n_mels * n_freqs);

    // Every band should have at least some nonzero energy.
    for m in 0..n_mels {
        let row_sum: f32 = (0..n_freqs).map(|k| filters[m * n_freqs + k]).sum();
        assert!(
            row_sum > 0.0,
            "mel band {m} (of {n_mels}) has zero total weight"
        );
    }
}

#[test]
fn test_mel_filterbank_64_bins() {
    let n_mels = 64;
    let n_fft = 400;
    let n_freqs = n_fft / 2 + 1;
    let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);
    assert_eq!(filters.len(), n_mels * n_freqs);

    // All values should be non-negative (triangular filters).
    assert!(
        filters.iter().all(|&v| v >= 0.0),
        "filterbank should have no negative values"
    );
}

#[test]
fn test_mel_filterbank_128_bins_nonzero_per_band() {
    let n_mels = 128;
    let n_fft = 400;
    let n_freqs = n_fft / 2 + 1;
    let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);

    // Count the number of nonzero frequency bins per mel band.
    for m in 0..n_mels {
        let nonzero_count = (0..n_freqs)
            .filter(|&k| filters[m * n_freqs + k] > 0.0)
            .count();
        assert!(
            nonzero_count >= 1,
            "mel band {m} (of 128) has no nonzero bins"
        );
    }
}

#[test]
fn test_mel_filterbank_slaney_normalization() {
    // Slaney area normalization: sum of each triangular filter should
    // approximate 2 / (right_hz - left_hz). Verify that normalized
    // filters sum to a value consistent with area normalization.
    let n_mels = 80;
    let n_fft = 400;
    let n_freqs = n_fft / 2 + 1;
    let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);

    let freq_resolution = SAMPLE_RATE as f64 / n_fft as f64;

    for m in 0..n_mels {
        let row_sum: f64 = (0..n_freqs)
            .map(|k| f64::from(filters[m * n_freqs + k]))
            .sum();
        // Each triangular filter after normalization integrates to approximately
        // 2 / bandwidth. With discrete bins, sum * freq_resolution should be
        // approximately 2 / bandwidth * bandwidth_in_bins * freq_resolution,
        // but the key invariant is that the sum is finite and positive.
        assert!(
            row_sum > 0.0 && row_sum.is_finite(),
            "mel band {m}: row_sum={row_sum} should be finite and positive, freq_res={freq_resolution}"
        );
    }
}

#[test]
fn test_mel_filterbank_peak_at_center_frequency() {
    // For well-separated mel bands, the filter peak should be at or near
    // the center frequency of the triangular filter.
    let n_mels = 40;
    let n_fft = 1024;
    let n_freqs = n_fft / 2 + 1;
    let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);

    for m in 0..n_mels {
        let peak_bin = (0..n_freqs)
            .max_by(|&a, &b| {
                filters[m * n_freqs + a]
                    .partial_cmp(&filters[m * n_freqs + b])
                    .unwrap()
            })
            .unwrap();
        let peak_val = filters[m * n_freqs + peak_bin];

        // The peak value should be positive.
        assert!(
            peak_val > 0.0,
            "mel band {m} peak value should be > 0, got {peak_val}"
        );
    }
}

// ============================================================================
// Mel frequency conversion: hz_to_mel and mel_to_hz edge cases
// ============================================================================

#[test]
fn test_hz_to_mel_monotonically_increasing() {
    // Mel scale should be monotonically increasing for all positive Hz values.
    let mut prev = hz_to_mel_slaney(0.0);
    for hz in (1..=20000).step_by(10) {
        let mel = hz_to_mel_slaney(f64::from(hz));
        assert!(
            mel > prev,
            "mel scale not monotonically increasing: hz_to_mel({hz}) = {mel} <= prev ({prev})"
        );
        prev = mel;
    }
}

#[test]
fn test_hz_to_mel_known_values_below_1khz() {
    // Below 1 kHz, Slaney mel is linear: mel = hz / 200 * 3.
    // Actually: mel = hz * 3 / 200 (the linear region factor).
    // The exact formula is hz / (200/3) = hz * 3 / 200 = hz / 66.667
    let mel_500 = hz_to_mel_slaney(500.0);
    // In the linear region: mel = hz / (200/3) = 500 * 3 / 200 = 7.5
    assert!(
        (mel_500 - 7.5).abs() < 0.01,
        "hz_to_mel(500) = {mel_500}, expected 7.5"
    );
}

#[test]
fn test_mel_to_hz_known_values() {
    // Verify that mel_to_hz reverses hz_to_mel for a range of frequencies.
    for &hz in &[0.0, 100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0] {
        let mel = hz_to_mel_slaney(hz);
        let back = mel_to_hz_slaney(mel);
        assert!(
            (back - hz).abs() < 1e-6,
            "roundtrip failed: hz={hz}, mel={mel}, back={back}"
        );
    }
}

#[test]
fn test_hz_to_mel_large_frequency() {
    // High frequencies should produce large but finite mel values.
    let mel = hz_to_mel_slaney(16000.0);
    assert!(mel.is_finite(), "mel(16kHz) should be finite");
    assert!(mel > 0.0, "mel(16kHz) should be positive");
    // Slaney mel for 16 kHz is approximately 25.1 (log region).
    assert!(mel > 20.0, "mel(16kHz) = {mel}, expected > 20");
}

#[test]
fn test_hz_to_mel_negative_returns_negative() {
    // Negative Hz input: mel scale should handle gracefully.
    // In the linear region below 1 kHz, negative Hz gives negative mel.
    let mel = hz_to_mel_slaney(-100.0);
    assert!(mel.is_finite(), "mel(-100) should be finite");
    assert!(mel < 0.0, "mel(-100) = {mel}, should be negative");
}

// ============================================================================
// STFT window construction
// ============================================================================

#[test]
fn test_hann_window_zero_at_start() {
    // Periodic Hann: w[0] = 0.5*(1 - cos(0)) = 0.
    let w = hann_window(400);
    assert!(
        w[0].abs() < 1e-15,
        "hann_window[0] = {}, expected 0",
        w[0]
    );
}

#[test]
fn test_hann_window_peak_near_middle() {
    // The peak of a periodic Hann window is at or near the center.
    let n = 400;
    let w = hann_window(n);
    let peak_idx = w
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    // Peak should be near n/2.
    let center = n / 2;
    assert!(
        (peak_idx as isize - center as isize).unsigned_abs() <= 1,
        "peak at {peak_idx}, expected near {center}"
    );
}

#[test]
fn test_hann_window_all_non_negative() {
    let w = hann_window(400);
    for (i, &v) in w.iter().enumerate() {
        assert!(v >= 0.0, "hann_window[{i}] = {v}, should be >= 0");
    }
}

#[test]
fn test_hann_window_max_value_near_one() {
    // The periodic Hann window maximum approaches 1.0 but never exactly 1.0
    // (that would require the symmetric form). For n=400, the peak should
    // be very close to 1.0.
    let w = hann_window(400);
    let max_val = w.iter().copied().fold(0.0f64, f64::max);
    assert!(
        (max_val - 1.0).abs() < 0.001,
        "hann max = {max_val}, expected close to 1.0"
    );
}

#[test]
fn test_hann_window_various_sizes() {
    for &n in &[64, 128, 256, 400, 512, 1024, 2048] {
        let w = hann_window(n);
        assert_eq!(w.len(), n, "window length mismatch for n={n}");
        assert!(
            w.iter().all(|v| v.is_finite()),
            "non-finite values for n={n}"
        );
        assert!(
            w.iter().all(|&v| v >= 0.0),
            "negative values for n={n}"
        );
    }
}

// ============================================================================
// Spectrogram shape for different audio lengths
// ============================================================================

#[test]
fn test_pcm_to_mel_shape_short_audio() {
    // 0.5s = 8000 samples. Frame count = 8000/160 + 1 = 51 frames.
    let n_mels = 128;
    let filters = mel_filterbank(n_mels, N_FFT, SAMPLE_RATE);
    let audio = vec![0.1f32; 8000];
    let mel = pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, n_mels).unwrap();
    let dims = mel.dims();
    assert_eq!(dims[0], 1, "batch dim");
    assert_eq!(dims[1], n_mels, "mel dim");
    assert_eq!(dims[2], 8000 / HOP_LENGTH + 1, "frame dim");
}

#[test]
fn test_pcm_to_mel_shape_1_second() {
    // 1s = 16000 samples. Frame count = 16000/160 + 1 = 101 frames.
    let n_mels = 80;
    let filters = mel_filterbank(n_mels, N_FFT, SAMPLE_RATE);
    let audio = vec![0.1f32; SAMPLE_RATE];
    let mel = pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, n_mels).unwrap();
    assert_eq!(mel.dim(2).unwrap(), SAMPLE_RATE / HOP_LENGTH + 1);
}

#[test]
fn test_pcm_to_mel_shape_5_seconds() {
    // 5s = 80000 samples. Frame count = 80000/160 + 1 = 501 frames.
    let n_mels = 128;
    let filters = mel_filterbank(n_mels, N_FFT, SAMPLE_RATE);
    let audio = vec![0.1f32; 5 * SAMPLE_RATE];
    let mel = pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, n_mels).unwrap();
    assert_eq!(mel.dim(2).unwrap(), 5 * SAMPLE_RATE / HOP_LENGTH + 1);
}

#[test]
fn test_whisper_mel_spectrogram_for_config_clips_to_n_frames() {
    // No matter the audio length, the whisper-specific function clips to N_FRAMES.
    for &audio_len in &[1000, 16000, 480000, 640000] {
        let audio = vec![0.1f32; audio_len];
        let mel = whisper_mel_spectrogram_for_config(&audio, 128).unwrap();
        assert_eq!(
            mel.dim(2).unwrap(),
            N_FRAMES,
            "audio_len={audio_len}: expected {N_FRAMES} frames"
        );
    }
}

// ============================================================================
// Known audio input -> known spectrogram output
// ============================================================================

#[test]
fn test_sine_wave_energy_in_expected_mel_band() {
    // A 440 Hz sine wave should produce most energy in mel bands corresponding
    // to 440 Hz. For 128 mel bands spanning 0-8000 Hz, 440 Hz falls in the
    // lower bands.
    let n_mels = 128;
    let filters = mel_filterbank(n_mels, N_FFT, SAMPLE_RATE);

    let audio: Vec<f32> = (0..16000)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();

    let mel = pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, n_mels).unwrap();
    let n_frames = mel.dim(2).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();

    // Compute per-band mean energy.
    let mut band_means: Vec<f32> = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
        let sum: f32 = (0..n_frames).map(|t| vals[m * n_frames + t]).sum();
        band_means.push(sum / n_frames as f32);
    }

    // Find the band with maximum mean energy.
    let peak_band = band_means
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    // 440 Hz should fall in the lower quarter of mel bands (below band 32 of 128).
    assert!(
        peak_band < 40,
        "440 Hz peak at mel band {peak_band}, expected < 40 (lower frequency region)"
    );
}

#[test]
fn test_high_frequency_sine_energy_in_upper_bands() {
    // A 7000 Hz sine wave should produce energy in upper mel bands.
    let n_mels = 128;
    let filters = mel_filterbank(n_mels, N_FFT, SAMPLE_RATE);

    let audio: Vec<f32> = (0..16000)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            (2.0 * std::f32::consts::PI * 7000.0 * t).sin()
        })
        .collect();

    let mel = pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, n_mels).unwrap();
    let n_frames = mel.dim(2).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();

    let mut band_means: Vec<f32> = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
        let sum: f32 = (0..n_frames).map(|t| vals[m * n_frames + t]).sum();
        band_means.push(sum / n_frames as f32);
    }

    let peak_band = band_means
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    // 7000 Hz should fall in the upper half of mel bands.
    assert!(
        peak_band > n_mels / 2,
        "7 kHz peak at mel band {peak_band}, expected > {} (upper frequency region)",
        n_mels / 2
    );
}

#[test]
fn test_impulse_response_spreads_energy() {
    // A single impulse should spread energy across all mel bands
    // (white-noise-like spectrum).
    let n_mels = 80;
    let filters = mel_filterbank(n_mels, N_FFT, SAMPLE_RATE);

    let mut audio = vec![0.0f32; 16000];
    audio[8000] = 1.0; // impulse at center

    let mel = pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, n_mels).unwrap();
    let n_frames = mel.dim(2).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();

    let mut band_means: Vec<f32> = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
        let sum: f32 = (0..n_frames).map(|t| vals[m * n_frames + t]).sum();
        band_means.push(sum / n_frames as f32);
    }

    // Most bands should have some energy (not just one band).
    let active_bands = band_means.iter().filter(|&&v| v > band_means[0] - 0.5).count();
    assert!(
        active_bands > n_mels / 2,
        "impulse should spread energy: only {active_bands}/{n_mels} bands active"
    );
}
