// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Normalization and denormalization tests for HTDemucs.
//!
//! Extracted from `htdemucs_tests.rs` to keep both files under 500 lines.
//! Covers `normalize_audio()` and `denormalize_output()` edge cases:
//! zero input, constant input, roundtrip, single-sample, large values, NaN, Inf.
//!
//! Part of #779 and #931.

use super::*;

// ---------------------------------------------------------------------------
// Normalization tests
// ---------------------------------------------------------------------------

#[test]
fn test_normalize_audio_zero_input() {
    let t = 100;
    let audio = vec![0.0f32; AUDIO_CHANNELS * t];
    let (normalized, mean, std_val) = normalize_audio(&audio, t).expect("valid audio");

    assert_eq!(normalized.len(), audio.len());
    assert_eq!(mean, 0.0);
    assert_eq!(std_val, 1e-8); // epsilon floor
                               // All output should be 0.
    for v in &normalized {
        assert!((v.abs()) < 1e-6, "expected ~0, got {v}");
    }
}

#[test]
fn test_normalize_audio_constant_input() {
    let t = 100;
    let val = 3.0f32;
    let audio = vec![val; AUDIO_CHANNELS * t];
    let (normalized, mean, std_val) = normalize_audio(&audio, t).expect("valid audio");

    // Constant f32 input with f64 accumulation: mean is exact.
    assert_eq!(mean, val, "mean should be exactly {val}, got {mean}");
    assert_eq!(std_val, 1e-8); // zero variance → epsilon
                               // (val - val) / epsilon = 0.0 exactly (IEEE 754: x - x = 0 for finite x).
    for v in &normalized {
        assert!(v.abs() < 1e-6, "expected ~0 after normalize: {v}");
    }
}

#[test]
fn test_normalize_denormalize_roundtrip() {
    let t = 200;
    let audio: Vec<f32> = (0..AUDIO_CHANNELS * t)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let (normalized, mean, std_val) = normalize_audio(&audio, t).expect("valid audio");

    // denormalize_output expects OUTPUT_CHANNELS * t elements (4 sources × 2 ch × t).
    // Repeat the normalized AUDIO_CHANNELS data across NUM_SOURCES to fill the buffer.
    let mut output_buf = Vec::with_capacity(OUTPUT_CHANNELS * t);
    for _ in 0..NUM_SOURCES {
        output_buf.extend_from_slice(&normalized);
    }

    // Denormalize back — first AUDIO_CHANNELS * t elements should recover input.
    // The roundtrip is algebraically exact: (v - mean) / std * std + mean = v.
    // Only f32 rounding error remains (~1e-7 relative). Use 1e-5 as generous bound.
    let recovered = denormalize_output(&output_buf, t, mean, std_val).expect("valid data");
    for i in 0..audio.len() {
        let diff = (audio[i] - recovered[i]).abs();
        assert!(
            diff < 1e-5,
            "roundtrip mismatch at {i}: {:.8} vs {:.8} (diff={diff:.10})",
            audio[i],
            recovered[i]
        );
    }
}

/// Single-sample audio: t=1 triggers edge case in mean/std computation.
/// The mono average is just the single sample, and std should hit the
/// epsilon floor (zero variance from a single observation).
#[test]
fn test_normalize_audio_single_sample() {
    let t = 1;
    let audio = vec![0.5f32; AUDIO_CHANNELS * t]; // stereo, 1 sample
    let (normalized, mean, std_val) = normalize_audio(&audio, t).expect("valid audio");

    assert_eq!(normalized.len(), AUDIO_CHANNELS);
    assert!((mean - 0.5).abs() < 1e-6, "mean should be 0.5, got {mean}");
    assert_eq!(std_val, 1e-8, "single sample has zero variance → epsilon");
}

/// Large-magnitude audio: verify f64 accumulation prevents f32 overflow
/// during variance computation. With constant 1e6 values, std is epsilon,
/// so (1e6 - 1e6) / 1e-8 ≈ 0 — output stays finite.
#[test]
fn test_normalize_audio_large_values() {
    let t = 100;
    let val = 1e6_f32;
    let audio = vec![val; AUDIO_CHANNELS * t];
    let (normalized, mean, std_val) = normalize_audio(&audio, t).expect("valid audio");

    assert!(mean.is_finite(), "mean must be finite, got {mean}");
    assert!(std_val.is_finite(), "std must be finite, got {std_val}");
    // Constant input → zero variance → epsilon floor
    assert_eq!(std_val, 1e-8);
    for v in &normalized {
        assert!(v.is_finite(), "normalized output must be finite, got {v}");
    }
}

/// AC3 (#931): Large finite values with high variance remain finite thanks to
/// f64 accumulation. The NormalizeOverflow guard is defense-in-depth; with f64
/// accumulation and f32 inputs, overflow is not directly triggerable.
/// This test verifies that extreme-magnitude f32 values produce finite output.
#[test]
fn test_normalize_audio_extreme_magnitude_stays_finite() {
    let t = 100;
    // Alternating f32::MAX and -f32::MAX to maximize variance.
    let mut audio = vec![0.0f32; AUDIO_CHANNELS * t];
    for (i, val) in audio.iter_mut().enumerate() {
        *val = if i % 2 == 0 { f32::MAX } else { -f32::MAX };
    }
    let result = normalize_audio(&audio, t);
    // With f64 accumulation, this should succeed — output stays finite.
    assert!(
        result.is_ok(),
        "extreme-magnitude finite f32 values should normalize successfully, got: {:?}",
        result.err()
    );
    let (normalized, _, _) = result.unwrap();
    for v in &normalized {
        assert!(v.is_finite(), "normalized output must be finite, got {v}");
    }
}

/// NaN input: normalize_audio may return error (NaN propagates through mean,
/// producing NaN in output which triggers the finiteness check).
/// The caller (forward()) rejects NaN before calling normalize_audio,
/// so this is defense-in-depth.
#[test]
fn test_normalize_audio_nan_input() {
    let t = 100;
    let mut audio = vec![1.0f32; AUDIO_CHANNELS * t];
    audio[42] = f32::NAN;
    // NaN in mean causes NaN in (v - mean), which is non-finite → error.
    let result = normalize_audio(&audio, t);
    assert!(
        result.is_err(),
        "NaN input should produce NormalizeOverflow (NaN propagates through mean)"
    );
}

/// Infinity input: normalize_audio returns error (Inf propagates).
#[test]
fn test_normalize_audio_inf_input() {
    let t = 100;
    let mut audio = vec![1.0f32; AUDIO_CHANNELS * t];
    audio[0] = f32::INFINITY;
    let result = normalize_audio(&audio, t);
    assert!(
        result.is_err(),
        "Inf input should produce NormalizeOverflow (Inf propagates through normalization)"
    );
}

// ---------------------------------------------------------------------------
// Denormalization tests
// ---------------------------------------------------------------------------

/// AC4 (#931): Short data produces DenormalizeLengthMismatch error.
#[test]
fn test_denormalize_output_short_data_returns_error() {
    let t = 100;
    // OUTPUT_CHANNELS * t elements expected, provide fewer.
    let short_data = vec![0.0f32; OUTPUT_CHANNELS * t - 1];
    let result = denormalize_output(&short_data, t, 0.0, 1.0);
    assert!(
        result.is_err(),
        "expected DenormalizeLengthMismatch for short data"
    );
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("denormalize length mismatch"),
        "expected DenormalizeLengthMismatch, got: {err}"
    );
}

/// Exact-length data succeeds.
#[test]
fn test_denormalize_output_exact_length() {
    let t = 50;
    let data = vec![1.0f32; OUTPUT_CHANNELS * t];
    let result = denormalize_output(&data, t, 0.5, 2.0);
    assert!(result.is_ok(), "exact length should succeed");
    let out = result.unwrap();
    assert_eq!(out.len(), OUTPUT_CHANNELS * t);
    // Each value: 1.0 * 2.0 + 0.5 = 2.5
    for v in &out {
        assert!((v - 2.5).abs() < 1e-6, "expected 2.5, got {v}");
    }
}

/// Longer data succeeds (extra elements are truncated to OUTPUT_CHANNELS * t).
#[test]
fn test_denormalize_output_longer_data() {
    let t = 50;
    let data = vec![1.0f32; OUTPUT_CHANNELS * t + 100];
    let result = denormalize_output(&data, t, 0.0, 1.0);
    assert!(result.is_ok(), "longer data should succeed");
    assert_eq!(result.unwrap().len(), OUTPUT_CHANNELS * t);
}
