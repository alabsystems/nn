// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for audio preprocessing utilities: resampling, stereo-to-mono,
//! normalization, and 30-second chunk padding/trimming.

use super::*;
use crate::config::{N_SAMPLES, SAMPLE_RATE};

// ============================================================================
// Stereo-to-mono conversion
// ============================================================================

#[test]
fn test_stereo_to_mono_basic() {
    // L=1.0, R=0.0 -> mono=0.5
    let stereo = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0];
    let mono = stereo_to_mono(&stereo).unwrap();
    assert_eq!(mono.len(), 3);
    assert!((mono[0] - 0.5).abs() < 1e-6);
    assert!((mono[1] - 0.5).abs() < 1e-6);
    assert!((mono[2] - 0.5).abs() < 1e-6);
}

#[test]
fn test_stereo_to_mono_identical_channels() {
    let val = 0.75;
    let stereo: Vec<f32> = (0..100).flat_map(|_| [val, val]).collect();
    let mono = stereo_to_mono(&stereo).unwrap();
    assert_eq!(mono.len(), 100);
    for &s in &mono {
        assert!((s - val).abs() < 1e-6);
    }
}

#[test]
fn test_stereo_to_mono_empty() {
    let mono = stereo_to_mono(&[]).unwrap();
    assert!(mono.is_empty());
}

#[test]
fn test_stereo_to_mono_odd_length_errors() {
    let result = stereo_to_mono(&[1.0, 2.0, 3.0]);
    assert!(result.is_err(), "odd sample count should fail");
}

#[test]
fn test_stereo_to_mono_preserves_mean() {
    // For any stereo pair, mono = (L+R)/2.
    let stereo = vec![-1.0, 1.0, 0.3, -0.7, 0.0, 0.0];
    let mono = stereo_to_mono(&stereo).unwrap();
    assert!((mono[0] - 0.0).abs() < 1e-6);
    assert!((mono[1] - (-0.2)).abs() < 1e-6);
    assert!((mono[2] - 0.0).abs() < 1e-6);
}

#[test]
fn test_stereo_to_mono_negative_values() {
    let stereo = vec![-0.5, -0.3, -1.0, -1.0];
    let mono = stereo_to_mono(&stereo).unwrap();
    assert!((mono[0] - (-0.4)).abs() < 1e-6);
    assert!((mono[1] - (-1.0)).abs() < 1e-6);
}

// ============================================================================
// Resampling (any rate -> 16 kHz)
// ============================================================================

#[test]
fn test_resample_same_rate_passthrough() {
    let audio: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    let result = resample(&audio, 16000, 16000).unwrap();
    assert_eq!(result.len(), audio.len());
    for (a, b) in audio.iter().zip(result.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_resample_downsample_48k_to_16k() {
    // 48kHz to 16kHz: ratio = 3, output ~1/3 the samples.
    let n = 48000;
    let audio: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
    let result = resample(&audio, 48000, 16000).unwrap();
    // Expected: ceil(48000 / 3) = 16000.
    assert_eq!(result.len(), 16000);
}

#[test]
fn test_resample_upsample_8k_to_16k() {
    // 8kHz to 16kHz: ratio = 0.5, output ~2x the samples.
    let n = 8000;
    let audio: Vec<f32> = (0..n).map(|i| (i as f32 * 0.002).sin()).collect();
    let result = resample(&audio, 8000, 16000).unwrap();
    assert_eq!(result.len(), 16000);
}

#[test]
fn test_resample_44100_to_16000() {
    // Common CD-quality to Whisper rate.
    let n = 44100;
    let audio: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
    let result = resample(&audio, 44100, 16000).unwrap();
    let expected_len = (f64::from(n) / (44100.0 / 16000.0)).ceil() as usize;
    assert_eq!(result.len(), expected_len);
}

#[test]
fn test_resample_preserves_dc() {
    // A constant signal should remain constant after resampling.
    let audio = vec![0.5f32; 1000];
    let result = resample(&audio, 48000, 16000).unwrap();
    for &v in &result {
        assert!((v - 0.5).abs() < 1e-5, "DC not preserved: {v}");
    }
}

#[test]
fn test_resample_empty() {
    let result = resample(&[], 44100, 16000).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_resample_zero_source_rate_errors() {
    let result = resample(&[1.0], 0, 16000);
    assert!(result.is_err());
}

#[test]
fn test_resample_zero_target_rate_errors() {
    let result = resample(&[1.0], 16000, 0);
    assert!(result.is_err());
}

#[test]
fn test_resample_single_sample() {
    let result = resample(&[0.7], 48000, 16000).unwrap();
    assert!(!result.is_empty());
    // Single sample can only produce itself.
    assert!((result[0] - 0.7).abs() < 1e-6);
}

#[test]
fn test_resample_output_finite() {
    let audio: Vec<f32> = (0..10000).map(|i| (i as f32 * 0.01).sin()).collect();
    let result = resample(&audio, 22050, 16000).unwrap();
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "sample {i} is not finite: {v}");
    }
}

// ============================================================================
// Audio normalization
// ============================================================================

#[test]
fn test_normalize_audio_peak_to_one() {
    let audio = vec![0.0, 0.5, -0.5, 0.25, -0.25];
    let normalized = normalize_audio(&audio);
    let peak = normalized.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!((peak - 1.0).abs() < 1e-6, "peak should be 1.0, got {peak}");
}

#[test]
fn test_normalize_audio_already_normalized() {
    let audio = vec![0.0, 1.0, -1.0, 0.5];
    let normalized = normalize_audio(&audio);
    // Peak is already 1.0, so values shouldn't change.
    for (a, b) in audio.iter().zip(normalized.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_normalize_audio_silence() {
    let audio = vec![0.0; 100];
    let normalized = normalize_audio(&audio);
    // Silent audio should remain silent (no division by zero).
    for &v in &normalized {
        assert!((v - 0.0).abs() < 1e-10);
    }
}

#[test]
fn test_normalize_audio_empty() {
    let normalized = normalize_audio(&[]);
    assert!(normalized.is_empty());
}

#[test]
fn test_normalize_audio_large_values() {
    let audio = vec![0.0, 100.0, -50.0, 25.0];
    let normalized = normalize_audio(&audio);
    assert!((normalized[1] - 1.0).abs() < 1e-6);
    assert!((normalized[2] - (-0.5)).abs() < 1e-6);
    assert!((normalized[3] - 0.25).abs() < 1e-6);
}

#[test]
fn test_normalize_audio_preserves_sign() {
    let audio = vec![-0.3, 0.6, -0.9, 0.1];
    let normalized = normalize_audio(&audio);
    for (a, n) in audio.iter().zip(normalized.iter()) {
        assert_eq!(a.signum(), n.signum(), "sign flipped for {a}");
    }
}

#[test]
fn test_normalize_audio_tiny_values() {
    // Very small values below threshold should pass through unchanged.
    let audio = vec![1e-12, -1e-12, 0.0];
    let normalized = normalize_audio(&audio);
    for (a, n) in audio.iter().zip(normalized.iter()) {
        assert!((a - n).abs() < 1e-15);
    }
}

#[test]
fn test_normalize_audio_single_sample() {
    let normalized = normalize_audio(&[0.5]);
    assert!((normalized[0] - 1.0).abs() < 1e-6);
}

// ============================================================================
// 30-second chunk padding/trimming
// ============================================================================

#[test]
fn test_pad_or_trim_shorter_audio() {
    let audio = vec![1.0; 1000];
    let result = pad_or_trim(&audio);
    assert_eq!(result.len(), N_SAMPLES);
    // First 1000 samples preserved.
    for &v in &result[..1000] {
        assert!((v - 1.0).abs() < 1e-6);
    }
    // Rest is zero-padded.
    for &v in &result[1000..] {
        assert!((v - 0.0).abs() < 1e-6);
    }
}

#[test]
fn test_pad_or_trim_exact_length() {
    let audio = vec![0.5; N_SAMPLES];
    let result = pad_or_trim(&audio);
    assert_eq!(result.len(), N_SAMPLES);
    for &v in &result {
        assert!((v - 0.5).abs() < 1e-6);
    }
}

#[test]
fn test_pad_or_trim_longer_audio() {
    let audio = vec![0.7; N_SAMPLES + 10000];
    let result = pad_or_trim(&audio);
    assert_eq!(result.len(), N_SAMPLES);
    // All samples should be from the original (truncated, not padded).
    for &v in &result {
        assert!((v - 0.7).abs() < 1e-6);
    }
}

#[test]
fn test_pad_or_trim_empty() {
    let result = pad_or_trim(&[]);
    assert_eq!(result.len(), N_SAMPLES);
    for &v in &result {
        assert!((v - 0.0).abs() < 1e-6);
    }
}

#[test]
fn test_pad_or_trim_single_sample() {
    let result = pad_or_trim(&[0.99]);
    assert_eq!(result.len(), N_SAMPLES);
    assert!((result[0] - 0.99).abs() < 1e-6);
    assert!((result[1] - 0.0).abs() < 1e-6);
}

#[test]
fn test_pad_or_trim_n_samples_is_30_seconds_at_16khz() {
    assert_eq!(N_SAMPLES, SAMPLE_RATE * 30);
    assert_eq!(N_SAMPLES, 480_000);
}

// ============================================================================
// Full preprocessing pipeline
// ============================================================================

#[test]
fn test_preprocess_mono_16k() {
    // Mono audio at 16 kHz: only normalization and padding applied.
    let audio: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
    let result = preprocess_audio(&audio, 1, 16000).unwrap();
    assert_eq!(result.len(), N_SAMPLES);
    // Peak should be normalized to 1.0.
    let peak = result.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!((peak - 1.0).abs() < 1e-5, "peak after normalization: {peak}");
}

#[test]
fn test_preprocess_stereo_44100() {
    // Stereo at 44.1 kHz: stereo-to-mono + resample + normalize + pad.
    let n = 44100; // 1 second of stereo audio.
    let stereo: Vec<f32> = (0..n * 2)
        .map(|i| (i as f32 * 0.001).sin() * 0.3)
        .collect();
    let result = preprocess_audio(&stereo, 2, 44100).unwrap();
    assert_eq!(result.len(), N_SAMPLES);
    // All values should be finite.
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "sample {i} is not finite");
    }
}

#[test]
fn test_preprocess_invalid_channels_errors() {
    let result = preprocess_audio(&[1.0], 0, 16000);
    assert!(result.is_err());

    let result = preprocess_audio(&[1.0], 3, 16000);
    assert!(result.is_err());
}

#[test]
fn test_preprocess_zero_sample_rate_errors() {
    let result = preprocess_audio(&[1.0], 1, 0);
    assert!(result.is_err());
}

#[test]
fn test_preprocess_48k_mono() {
    // Mono at 48 kHz: resample to 16 kHz + normalize + pad.
    let audio: Vec<f32> = (0..48000).map(|i| (i as f32 * 0.002).sin()).collect();
    let result = preprocess_audio(&audio, 1, 48000).unwrap();
    assert_eq!(result.len(), N_SAMPLES);
}

#[test]
fn test_preprocess_8k_mono() {
    // Mono at 8 kHz: upsample to 16 kHz + normalize + pad.
    let audio: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.005).sin()).collect();
    let result = preprocess_audio(&audio, 1, 8000).unwrap();
    assert_eq!(result.len(), N_SAMPLES);
}

#[test]
fn test_preprocess_long_audio_truncated() {
    // 60 seconds of audio at 16 kHz should be truncated to 30 seconds.
    let audio: Vec<f32> = (0..960_000).map(|i| (i as f32 * 0.001).sin()).collect();
    let result = preprocess_audio(&audio, 1, 16000).unwrap();
    assert_eq!(result.len(), N_SAMPLES);
}

#[test]
fn test_preprocess_output_bounded() {
    // After normalization, all values should be in [-1, 1].
    let audio: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin() * 5.0).collect();
    let result = preprocess_audio(&audio, 1, 16000).unwrap();
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (-1.0..=1.0).contains(&v),
            "sample {i} = {v} out of [-1, 1]"
        );
    }
}
