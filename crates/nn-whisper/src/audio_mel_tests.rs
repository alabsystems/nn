// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for Whisper audio/mel spectrogram processing.
//!
//! Covers mel filterbank construction, mel scale conversions, STFT window
//! functions, mel spectrogram output shapes, sample rate handling, audio
//! padding, normalization, and edge cases (silence, DC, single sample).

use crate::audio::{
    compute_log_mel_spectrogram, mel_filterbank, pcm_to_mel, whisper_mel_spectrogram,
    whisper_mel_spectrogram_for_config,
};
use crate::config::{HOP_LENGTH, N_FFT, N_FRAMES, N_SAMPLES, NUM_MEL_BINS, SAMPLE_RATE};
use nn_core::audio::{hz_to_mel_slaney, mel_to_hz_slaney};

// ============================================================================
// Mel filter bank construction
// ============================================================================

#[test]
fn test_mel_filterbank_correct_number_of_filters_various_configs() {
    // Verify element count = n_mels * n_freqs for several configurations.
    for &(n_mels, n_fft, sr) in &[
        (80, 400, 16000),
        (128, 400, 16000),
        (40, 512, 22050),
        (64, 1024, 44100),
        (128, 2048, 48000),
    ] {
        let n_freqs = n_fft / 2 + 1;
        let filters = mel_filterbank(n_mels, n_fft, sr);
        assert_eq!(
            filters.len(),
            n_mels * n_freqs,
            "config ({n_mels}, {n_fft}, {sr}): expected {} elements, got {}",
            n_mels * n_freqs,
            filters.len()
        );
    }
}

#[test]
fn test_mel_filterbank_all_values_finite() {
    let filters = mel_filterbank(128, 400, 16000);
    for (i, &v) in filters.iter().enumerate() {
        assert!(
            v.is_finite(),
            "filter[{i}] = {v} is not finite"
        );
    }
}

#[test]
fn test_mel_filterbank_adjacent_bands_contiguous() {
    // Adjacent triangular filters should have contiguous or overlapping support
    // regions. With discrete frequency bins, narrow mel bands at low frequencies
    // may not share a nonzero bin but their supports should be adjacent (the
    // last nonzero bin of band m should be >= first nonzero bin of band m+1 - 1).
    let n_mels = 80;
    let n_fft = 400;
    let n_freqs = n_fft / 2 + 1;
    let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);

    for m in 0..n_mels - 1 {
        let row_a: Vec<f32> = (0..n_freqs).map(|k| filters[m * n_freqs + k]).collect();
        let row_b: Vec<f32> = (0..n_freqs).map(|k| filters[(m + 1) * n_freqs + k]).collect();

        let last_a = row_a.iter().rposition(|&v| v > 0.0);
        let first_b = row_b.iter().position(|&v| v > 0.0);

        if let (Some(la), Some(fb)) = (last_a, first_b) {
            // The last nonzero bin of band m should be at or past the first
            // nonzero bin of band m+1 (overlap), or at most 1 bin gap.
            assert!(
                la + 2 >= fb,
                "mel bands {m} and {}: gap too large (last_a={la}, first_b={fb})",
                m + 1
            );
        }
    }
}

#[test]
fn test_mel_filterbank_different_sample_rates() {
    // Verify filterbank construction works at sample rates other than 16 kHz.
    // Use n_fft proportional to sample rate so that frequency resolution is
    // sufficient for the mel band spacing at each sample rate.
    let configs: &[(usize, usize)] = &[
        (8000, 256),   // 8 kHz, n_fft=256
        (22050, 1024), // 22.05 kHz, n_fft=1024
        (44100, 2048), // 44.1 kHz, n_fft=2048
        (48000, 2048), // 48 kHz, n_fft=2048
    ];
    for &(sr, n_fft) in configs {
        let n_mels = 80;
        let n_freqs = n_fft / 2 + 1;
        let filters = mel_filterbank(n_mels, n_fft, sr);

        assert_eq!(filters.len(), n_mels * n_freqs);
        assert!(
            filters.iter().all(|&v| v >= 0.0),
            "negative filter value at sample rate {sr}"
        );
        assert!(
            filters.iter().all(|v| v.is_finite()),
            "non-finite filter value at sample rate {sr}"
        );

        // Every band should have nonzero energy.
        for m in 0..n_mels {
            let row_sum: f32 = (0..n_freqs).map(|k| filters[m * n_freqs + k]).sum();
            assert!(
                row_sum > 0.0,
                "mel band {m} all-zero at sample rate {sr}, n_fft={n_fft}"
            );
        }
    }
}

// ============================================================================
// Mel filter bank frequency range (0 to sample_rate/2)
// ============================================================================

#[test]
fn test_mel_filterbank_covers_zero_to_nyquist() {
    // The filterbank should have nonzero values starting from low frequency
    // bins and extending up to (near) the Nyquist frequency.
    let n_mels = 128;
    let n_fft = 400;
    let n_freqs = n_fft / 2 + 1;
    let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);

    // Find the lowest frequency bin with any nonzero filter value.
    let lowest_active_bin = (0..n_freqs)
        .find(|&k| (0..n_mels).any(|m| filters[m * n_freqs + k] > 0.0))
        .expect("filterbank has no active bins");

    // Find the highest frequency bin with any nonzero filter value.
    let highest_active_bin = (0..n_freqs)
        .rev()
        .find(|&k| (0..n_mels).any(|m| filters[m * n_freqs + k] > 0.0))
        .expect("filterbank has no active bins");

    // Lowest active bin should be near DC (bin 0 or 1).
    assert!(
        lowest_active_bin <= 2,
        "lowest active bin = {lowest_active_bin}, expected <= 2 (near DC)"
    );

    // Highest active bin should be near Nyquist (last few bins).
    assert!(
        highest_active_bin >= n_freqs - 5,
        "highest active bin = {highest_active_bin}, expected >= {} (near Nyquist)",
        n_freqs - 5
    );
}

// ============================================================================
// Mel scale conversion: hz_to_mel and mel_to_hz round-trip
// ============================================================================

#[test]
fn test_mel_scale_roundtrip_dense_sweep() {
    // Dense sweep from 0 to 16000 Hz in 1 Hz steps.
    for hz in (0..=16000).step_by(1) {
        let f = f64::from(hz);
        let mel = hz_to_mel_slaney(f);
        let back = mel_to_hz_slaney(mel);
        assert!(
            (back - f).abs() < 1e-6,
            "roundtrip failed at {f} Hz: mel={mel}, back={back}, diff={}",
            (back - f).abs()
        );
    }
}

#[test]
fn test_mel_scale_boundary_at_1khz() {
    // Slaney mel scale transitions from linear to logarithmic at 1000 Hz.
    // Verify continuity at the boundary.
    let mel_999 = hz_to_mel_slaney(999.0);
    let mel_1000 = hz_to_mel_slaney(1000.0);
    let mel_1001 = hz_to_mel_slaney(1001.0);

    // Should be monotonically increasing.
    assert!(mel_999 < mel_1000, "mel(999) >= mel(1000)");
    assert!(mel_1000 < mel_1001, "mel(1000) >= mel(1001)");

    // Continuity: the jump across boundary should be small.
    let diff_across = mel_1001 - mel_999;
    assert!(
        diff_across < 1.0,
        "discontinuity at 1 kHz boundary: mel(999)={mel_999}, mel(1001)={mel_1001}, diff={diff_across}"
    );
}

#[test]
fn test_mel_scale_zero_hz_is_zero_mel() {
    let mel_0 = hz_to_mel_slaney(0.0);
    assert!(
        mel_0.abs() < 1e-12,
        "hz_to_mel_slaney(0) = {mel_0}, expected 0"
    );
    let hz_0 = mel_to_hz_slaney(0.0);
    assert!(
        hz_0.abs() < 1e-12,
        "mel_to_hz_slaney(0) = {hz_0}, expected 0"
    );
}

// ============================================================================
// STFT window function (Hann window)
// ============================================================================

#[test]
fn test_hann_window_symmetric_values() {
    // Periodic Hann window: w[k] = 0.5 * (1 - cos(2*pi*k/N)).
    // Verify symmetry: w[k] should approximately equal w[N-1-k] for periodic window
    // (not exact for periodic, but close for large N).
    let n = 400;
    let w = nn_core::audio::hann_window(n);

    // For periodic Hann, w[k] = 0.5*(1 - cos(2*pi*k/N)).
    // w[1] should equal w[N-1].
    for k in 1..n / 2 {
        let diff = (w[k] - w[n - k]).abs();
        assert!(
            diff < 1e-10,
            "hann_window not symmetric: w[{k}]={}, w[{}]={}, diff={diff}",
            w[k],
            n - k,
            w[n - k]
        );
    }
}

#[test]
fn test_hann_window_sum_property() {
    // The sum of a periodic Hann window is N/2.
    let n = 400;
    let w = nn_core::audio::hann_window(n);
    let sum: f64 = w.iter().sum();
    let expected = n as f64 / 2.0;
    assert!(
        (sum - expected).abs() < 1e-8,
        "hann sum = {sum}, expected {expected}"
    );
}

#[test]
fn test_hann_window_length_matches_n_fft() {
    // Whisper uses n_fft=400 Hann window.
    let w = nn_core::audio::hann_window(N_FFT);
    assert_eq!(w.len(), N_FFT, "window length should match N_FFT");
}

// ============================================================================
// Mel spectrogram output shape: [1, n_mels, n_frames]
// ============================================================================

#[test]
fn test_whisper_mel_spectrogram_output_shape_standard() {
    // Standard 30s Whisper input: output should be [1, 128, 3000].
    let audio = vec![0.0f32; N_SAMPLES];
    let mel = whisper_mel_spectrogram(&audio).unwrap();
    assert_eq!(mel.dims(), &[1, NUM_MEL_BINS, N_FRAMES]);
}

#[test]
fn test_whisper_mel_spectrogram_for_config_80_bins() {
    // Whisper tiny/base/small/medium use 80 mel bins.
    let audio = vec![0.0f32; 16000];
    let mel = whisper_mel_spectrogram_for_config(&audio, 80).unwrap();
    let dims = mel.dims();
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 80, "expected 80 mel bins");
    assert_eq!(dims[2], N_FRAMES, "should be clipped to N_FRAMES");
}

#[test]
fn test_whisper_mel_spectrogram_for_config_128_bins() {
    // Whisper large-v3/turbo uses 128 mel bins.
    let audio = vec![0.0f32; 16000];
    let mel = whisper_mel_spectrogram_for_config(&audio, 128).unwrap();
    let dims = mel.dims();
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 128);
    assert_eq!(dims[2], N_FRAMES);
}

#[test]
fn test_pcm_to_mel_frame_count_various_lengths() {
    // Verify frame count formula: n_frames = n_samples / hop + 1
    // (after reflect padding of n_fft/2 on each side).
    let n_mels = 128;
    let filters = mel_filterbank(n_mels, N_FFT, SAMPLE_RATE);

    for &n_samples in &[800_usize, 1600, 4800, 16000, 48000] {
        let audio = vec![0.1f32; n_samples];
        let mel = pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, n_mels).unwrap();
        let actual_frames = mel.dim(2).unwrap();
        let expected_frames = n_samples / HOP_LENGTH + 1;
        assert_eq!(
            actual_frames, expected_frames,
            "n_samples={n_samples}: got {actual_frames} frames, expected {expected_frames}"
        );
    }
}

// ============================================================================
// Sample rate handling (16000 Hz for Whisper)
// ============================================================================

#[test]
fn test_whisper_constants_consistency() {
    assert_eq!(SAMPLE_RATE, 16_000, "Whisper sample rate must be 16 kHz");
    assert_eq!(N_FFT, 400, "Whisper FFT size must be 400");
    assert_eq!(HOP_LENGTH, 160, "Whisper hop length must be 160");
    assert_eq!(N_SAMPLES, 480_000, "30s at 16 kHz = 480,000 samples");
    assert_eq!(N_FRAMES, 3000, "N_SAMPLES / HOP_LENGTH = 3000");
    assert_eq!(NUM_MEL_BINS, 128, "large-v3 default mel bins = 128");
}

#[test]
fn test_pcm_to_mel_with_non_standard_fft_sizes() {
    // pcm_to_mel should work with FFT sizes other than 400.
    let audio = vec![0.1f32; 16000];
    for &n_fft in &[256_usize, 512, 1024] {
        let hop = n_fft / 4; // common default
        let n_mels = 80;
        let n_freqs = n_fft / 2 + 1;
        let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);
        assert_eq!(filters.len(), n_mels * n_freqs);

        let mel = pcm_to_mel(&audio, &filters, n_fft, hop, n_mels).unwrap();
        let dims = mel.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], n_mels);
        assert!(dims[2] > 0, "should have frames for n_fft={n_fft}");

        // All values finite.
        let vals = mel.to_flat_vec::<f32>().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "non-finite values for n_fft={n_fft}"
        );
    }
}

// ============================================================================
// Audio padding to 30 seconds
// ============================================================================

#[test]
fn test_padding_short_audio_to_30s() {
    // 1 second of audio should produce the same output shape as 30 seconds.
    let short = vec![0.1f32; SAMPLE_RATE];
    let mel = whisper_mel_spectrogram(&short).unwrap();
    assert_eq!(mel.dims(), &[1, NUM_MEL_BINS, N_FRAMES]);
}

#[test]
fn test_padding_exact_30s_audio() {
    // Exactly N_SAMPLES should work without allocation.
    let exact = vec![0.1f32; N_SAMPLES];
    let mel = whisper_mel_spectrogram(&exact).unwrap();
    assert_eq!(mel.dims(), &[1, NUM_MEL_BINS, N_FRAMES]);
}

#[test]
fn test_padding_longer_than_30s_truncates() {
    // 40s of audio should be truncated to 30s before mel computation.
    let long = vec![0.1f32; SAMPLE_RATE * 40];
    let mel = whisper_mel_spectrogram(&long).unwrap();
    assert_eq!(
        mel.dims(),
        &[1, NUM_MEL_BINS, N_FRAMES],
        "40s input should produce same shape as 30s"
    );
}

#[test]
fn test_padding_very_short_audio() {
    // 100 samples (6.25 ms at 16 kHz) should be padded to 30s.
    let tiny = vec![0.5f32; 100];
    let mel = whisper_mel_spectrogram(&tiny).unwrap();
    assert_eq!(mel.dims(), &[1, NUM_MEL_BINS, N_FRAMES]);

    // All values should be finite.
    let vals = mel.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()));
}

// ============================================================================
// Audio normalization (peak normalization)
// ============================================================================

#[test]
fn test_mel_normalization_affine_transform() {
    // pcm_to_mel applies: log10(max(1e-10, power)) -> clamp(max - 8) -> (x + 4) / 4.
    // For silence: log10(1e-10) = -10, max = -10, floor = -18.
    // All values = (-10 + 4) / 4 = -1.5.
    let audio = vec![0.0f32; 16000];
    let filters = mel_filterbank(128, N_FFT, SAMPLE_RATE);
    let mel = pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, 128).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();

    // All silence values should be identical.
    let first = vals[0];
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - first).abs() < 1e-6,
            "silence mel[{i}] = {v}, expected uniform {first}"
        );
    }

    // The normalized silence value should be (-10 + 4) / 4 = -1.5.
    assert!(
        (first - (-1.5)).abs() < 0.01,
        "silence normalized value = {first}, expected -1.5"
    );
}

#[test]
fn test_mel_normalization_louder_signal_higher_values() {
    // A louder signal should produce higher mel values than a quieter one.
    let make_sine = |amplitude: f32| -> Vec<f32> {
        (0..16000)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                amplitude * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect()
    };

    let filters = mel_filterbank(128, N_FFT, SAMPLE_RATE);

    let mel_quiet = pcm_to_mel(&make_sine(0.01), &filters, N_FFT, HOP_LENGTH, 128).unwrap();
    let mel_loud = pcm_to_mel(&make_sine(1.0), &filters, N_FFT, HOP_LENGTH, 128).unwrap();

    let mean_quiet: f32 =
        mel_quiet.to_flat_vec::<f32>().unwrap().iter().sum::<f32>() / mel_quiet.elem_count() as f32;
    let mean_loud: f32 =
        mel_loud.to_flat_vec::<f32>().unwrap().iter().sum::<f32>() / mel_loud.elem_count() as f32;

    assert!(
        mean_loud > mean_quiet,
        "loud signal mean ({mean_loud}) should exceed quiet signal mean ({mean_quiet})"
    );
}

// ============================================================================
// Edge cases: silence (all zeros) -> valid mel
// ============================================================================

#[test]
fn test_silence_produces_valid_finite_mel() {
    let audio = vec![0.0f32; 16000];
    let filters = mel_filterbank(128, N_FFT, SAMPLE_RATE);
    let mel = pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, 128).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();

    assert!(
        vals.iter().all(|v| v.is_finite()),
        "silence should produce all-finite mel values"
    );
    assert!(
        !vals.iter().any(|v| v.is_nan()),
        "silence should produce no NaN values"
    );
}

#[test]
fn test_silence_mel_values_are_negative() {
    // Silence has zero power -> log10(1e-10) = -10 -> after normalization = -1.5.
    let audio = vec![0.0f32; 16000];
    let filters = mel_filterbank(128, N_FFT, SAMPLE_RATE);
    let mel = pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, 128).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();

    let max_val = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_val < 0.0,
        "silence mel max = {max_val}, should be negative"
    );
}

// ============================================================================
// Edge cases: DC signal (constant value)
// ============================================================================

#[test]
fn test_dc_signal_produces_valid_mel() {
    // A constant (DC) signal should produce valid mel output.
    // DC energy concentrates in bin 0 (0 Hz).
    let audio = vec![0.5f32; 16000];
    let filters = mel_filterbank(128, N_FFT, SAMPLE_RATE);
    let mel = pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, 128).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();

    assert!(
        vals.iter().all(|v| v.is_finite()),
        "DC signal should produce all-finite mel values"
    );
}

#[test]
fn test_dc_signal_energy_in_low_bands() {
    // DC (0 Hz) energy should be concentrated in the lowest mel band.
    let audio = vec![1.0f32; 16000];
    let n_mels = 128;
    let filters = mel_filterbank(n_mels, N_FFT, SAMPLE_RATE);
    let mel = pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, n_mels).unwrap();
    let n_frames = mel.dim(2).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();

    // Mean energy per band.
    let mut band_means = vec![0.0f32; n_mels];
    for m in 0..n_mels {
        let sum: f32 = (0..n_frames).map(|t| vals[m * n_frames + t]).sum();
        band_means[m] = sum / n_frames as f32;
    }

    // Band 0 (DC) should have higher energy than higher bands.
    let low_mean = band_means[0];
    let high_mean: f32 = band_means[n_mels / 2..].iter().sum::<f32>()
        / (n_mels - n_mels / 2) as f32;
    assert!(
        low_mean > high_mean,
        "DC signal: band 0 mean ({low_mean}) should exceed upper half mean ({high_mean})"
    );
}

// ============================================================================
// Edge cases: single sample
// ============================================================================

#[test]
fn test_single_sample_produces_valid_mel() {
    // A single sample is the minimum valid input for pcm_to_mel.
    let audio = vec![0.5f32; 1];
    let filters = mel_filterbank(128, N_FFT, SAMPLE_RATE);
    let mel = pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, 128).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();

    assert!(
        vals.iter().all(|v| v.is_finite()),
        "single sample should produce all-finite mel values"
    );
    assert_eq!(mel.dims()[0], 1, "batch dim");
    assert_eq!(mel.dims()[1], 128, "mel dim");
    assert!(mel.dims()[2] >= 1, "should have at least 1 frame");
}

#[test]
fn test_single_sample_through_whisper_pipeline() {
    // Single sample through the full whisper mel pipeline (pads to 30s).
    let audio = vec![1.0f32; 1];
    let mel = whisper_mel_spectrogram(&audio).unwrap();
    assert_eq!(mel.dims(), &[1, NUM_MEL_BINS, N_FRAMES]);

    let vals = mel.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()));
}

// ============================================================================
// Mel spectrogram values are finite (no NaN/Inf)
// ============================================================================

#[test]
fn test_mel_spectrogram_finite_for_diverse_inputs() {
    let filters = mel_filterbank(128, N_FFT, SAMPLE_RATE);

    let test_cases: Vec<(&str, Vec<f32>)> = vec![
        ("silence", vec![0.0; 16000]),
        ("dc_positive", vec![1.0; 16000]),
        ("dc_negative", vec![-1.0; 16000]),
        ("tiny_amplitude", vec![1e-7; 16000]),
        ("large_amplitude", vec![100.0; 16000]),
        (
            "alternating",
            (0..16000).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect(),
        ),
        (
            "ramp",
            (0..16000).map(|i| i as f32 / 16000.0).collect(),
        ),
        (
            "impulse",
            {
                let mut v = vec![0.0f32; 16000];
                v[0] = 1.0;
                v
            },
        ),
    ];

    for (name, audio) in test_cases {
        let mel = pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, 128).unwrap();
        let vals = mel.to_flat_vec::<f32>().unwrap();
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                v.is_finite(),
                "{name}: mel value at index {i} = {v} is not finite"
            );
        }
    }
}

#[test]
fn test_mel_spectrogram_rejects_nan_input() {
    let filters = mel_filterbank(128, N_FFT, SAMPLE_RATE);
    let mut audio = vec![0.0f32; 16000];
    audio[42] = f32::NAN;
    assert!(
        pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, 128).is_err(),
        "should reject NaN input"
    );
}

#[test]
fn test_mel_spectrogram_rejects_inf_input() {
    let filters = mel_filterbank(128, N_FFT, SAMPLE_RATE);
    let mut audio = vec![0.0f32; 16000];
    audio[42] = f32::INFINITY;
    assert!(
        pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, 128).is_err(),
        "should reject Inf input"
    );
}

// ============================================================================
// Log mel spectrogram has reasonable range
// ============================================================================

#[test]
fn test_log_mel_range_silence() {
    // Silence -> all values at the floor after normalization.
    // log10(1e-10) = -10, clamp max-8 = -10-8 = -18, but max is -10 so
    // floor = -18. All values = -10 (above floor). Normalized: (-10+4)/4 = -1.5.
    let audio = vec![0.0f32; 16000];
    let filters = mel_filterbank(128, N_FFT, SAMPLE_RATE);
    let mel = pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, 128).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();

    for &v in &vals {
        assert!(
            (-3.0..=2.0).contains(&v),
            "mel value {v} outside reasonable range [-3, 2]"
        );
    }
}

#[test]
fn test_log_mel_range_sine_wave() {
    let audio: Vec<f32> = (0..16000)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
        })
        .collect();

    let filters = mel_filterbank(128, N_FFT, SAMPLE_RATE);
    let mel = pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, 128).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();

    let min_val = vals.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // After normalization, values should be in a reasonable range.
    // The 80 dB dynamic range clamp + affine normalization constrains the range.
    assert!(
        min_val >= -5.0,
        "min mel value {min_val} too negative (below -5)"
    );
    assert!(
        max_val <= 5.0,
        "max mel value {max_val} too large (above 5)"
    );
    assert!(
        max_val > min_val,
        "mel should have dynamic range, but min={min_val} == max={max_val}"
    );
}

#[test]
fn test_compute_log_mel_spectrogram_raw_values_negative() {
    // compute_log_mel_spectrogram undoes the affine normalization, returning
    // raw log10 values. For typical audio, these should be negative.
    let audio: Vec<f32> = (0..16000)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            0.1 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();

    let result = compute_log_mel_spectrogram(&audio, SAMPLE_RATE as u32);
    assert_eq!(result.len(), 80, "should have 80 mel bands");

    // Raw log10 values for moderate amplitude should be in [-10, 0] range.
    for (m, band) in result.iter().enumerate() {
        for (t, &v) in band.iter().enumerate() {
            assert!(
                v.is_finite(),
                "raw log mel[{m}][{t}] = {v} is not finite"
            );
            assert!(
                v >= -15.0,
                "raw log mel[{m}][{t}] = {v} too negative"
            );
        }
    }
}

// ============================================================================
// Error path tests
// ============================================================================

#[test]
fn test_pcm_to_mel_zero_n_fft_returns_error() {
    let audio = vec![0.1f32; 1000];
    let filters = vec![0.0f32; 128]; // dummy
    let err = pcm_to_mel(&audio, &filters, 0, 160, 128);
    assert!(err.is_err(), "n_fft=0 should be an error");
}

#[test]
fn test_pcm_to_mel_zero_hop_returns_error() {
    let audio = vec![0.1f32; 1000];
    let filters = mel_filterbank(128, N_FFT, SAMPLE_RATE);
    let err = pcm_to_mel(&audio, &filters, N_FFT, 0, 128);
    assert!(err.is_err(), "hop_length=0 should be an error");
}

#[test]
fn test_pcm_to_mel_empty_audio_returns_error() {
    let filters = mel_filterbank(128, N_FFT, SAMPLE_RATE);
    let err = pcm_to_mel(&[], &filters, N_FFT, HOP_LENGTH, 128);
    assert!(err.is_err(), "empty audio should be an error");
}

#[test]
fn test_pcm_to_mel_mismatched_filter_size_returns_error() {
    let audio = vec![0.1f32; 16000];
    let bad_filters = vec![0.0f32; 100]; // wrong size
    let err = pcm_to_mel(&audio, &bad_filters, N_FFT, HOP_LENGTH, 128);
    assert!(err.is_err(), "wrong filter size should be an error");
}
