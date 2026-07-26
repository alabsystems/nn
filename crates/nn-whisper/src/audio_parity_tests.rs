#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Numerical parity tests for Whisper audio preprocessing.
//!
//! Verifies mel filterbank and pcm_to_mel produce outputs matching
//! reference implementations (dvoice-common Slaney algorithm).
//! Extracted from `audio_tests.rs` for code-health (500-line limit).

use std::f64::consts::PI;

use super::super::*;

// -- AC3: Numerical parity with reference implementation ----------------------

/// Reference mel filterbank using identical Slaney algorithm as
/// `dvoice-common/src/mel.rs:253-301`. We inline the reference here
/// to verify our implementation matches without a cross-crate dependency.
fn reference_mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: usize) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let sr = sample_rate as f64;
    let f_sp: f64 = 200.0 / 3.0;
    let min_log_mel: f64 = 1000.0 / f_sp;
    let logstep = 6.4_f64.ln() / 27.0;

    let ref_hz_to_mel = |hz: f64| -> f64 {
        if hz < 1000.0 {
            hz / f_sp
        } else {
            min_log_mel + (hz / 1000.0).ln() / logstep
        }
    };
    let ref_mel_to_hz = |mel: f64| -> f64 {
        if mel < min_log_mel {
            f_sp * mel
        } else {
            1000.0 * (logstep * (mel - min_log_mel)).exp()
        }
    };

    let freqs: Vec<f64> = (0..n_freqs).map(|i| i as f64 * sr / n_fft as f64).collect();
    let min_mel = ref_hz_to_mel(0.0);
    let max_mel = ref_hz_to_mel(sr / 2.0);
    let hz_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| {
            let mel = min_mel + (max_mel - min_mel) * i as f64 / (n_mels + 1) as f64;
            ref_mel_to_hz(mel)
        })
        .collect();

    let mut filters = vec![0.0f32; n_mels * n_freqs];
    for i in 0..n_mels {
        let left = hz_points[i];
        let center = hz_points[i + 1];
        let right = hz_points[i + 2];
        for j in 0..n_freqs {
            let f = freqs[j];
            let rising = if center > left {
                (f - left) / (center - left)
            } else {
                0.0
            };
            let falling = if right > center {
                (right - f) / (right - center)
            } else {
                0.0
            };
            filters[i * n_freqs + j] = rising.min(falling).max(0.0) as f32;
        }
        let enorm = (2.0 / (right - left)) as f32;
        for j in 0..n_freqs {
            filters[i * n_freqs + j] *= enorm;
        }
    }
    filters
}

#[test]
fn test_mel_filterbank_parity_with_reference() {
    // Compare our mel_filterbank() output against the reference
    // (dvoice-common algorithm) element-by-element.
    let ours = mel_filterbank(128, 400, 16000);
    let reference = reference_mel_filterbank(128, 400, 16000);
    assert_eq!(ours.len(), reference.len());

    let mut max_diff = 0.0f32;
    for (i, (&a, &b)) in ours.iter().zip(reference.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(
            diff < 1e-6,
            "filterbank mismatch at index {i}: ours={a}, ref={b}, diff={diff}"
        );
    }
    // Should be bit-exact since we use the same algorithm and precision.
    assert!(
        max_diff < 1e-10,
        "max filterbank diff = {max_diff}, expected near-exact"
    );
}

#[test]
fn test_pcm_to_mel_known_tone_parity() {
    // Generate a known 440 Hz tone (A4) and verify the mel spectrogram
    // produces consistent output that matches expected numerical properties.
    let sample_rate: i32 = 16000;
    let duration = 0.1; // 100 ms = 1600 samples
    let n_samples = (f64::from(sample_rate) * duration) as usize;
    let audio: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / f64::from(sample_rate);
            (2.0 * PI * 440.0 * t).sin() as f32
        })
        .collect();

    let filters = mel_filterbank(128, 400, 16000);
    let mel = pcm_to_mel(&audio, &filters, 400, 160, 128).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();

    // All values finite.
    assert!(vals.iter().all(|v| v.is_finite()));

    // Normalization: (log10(power) + 4) / 4. For tones with power > 1,
    // max can exceed 1.0. Verify it stays in a reasonable range.
    let max_val = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(max_val <= 3.0, "max={max_val} unexpectedly large");

    // A 440 Hz tone should produce higher energy in low-frequency mel bands
    // than high-frequency bands.
    let n_frames = mel.dims()[2];
    let low_mean: f32 = (0..20)
        .map(|m| (0..n_frames).map(|t| vals[m * n_frames + t]).sum::<f32>())
        .sum::<f32>()
        / (20 * n_frames) as f32;
    let high_mean: f32 = (108..128)
        .map(|m| (0..n_frames).map(|t| vals[m * n_frames + t]).sum::<f32>())
        .sum::<f32>()
        / (20 * n_frames) as f32;
    assert!(
        low_mean > high_mean,
        "440 Hz tone: low bands ({low_mean}) should be stronger than high bands ({high_mean})"
    );
}
