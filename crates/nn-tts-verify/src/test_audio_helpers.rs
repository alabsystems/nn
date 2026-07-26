// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared test audio generation helpers for nn-tts-verify.
//!
//! Consolidates the `sine_wave` function that was duplicated across 10 test
//! files into a single canonical implementation.

/// Generate a sine wave with the given frequency, sample rate, and duration.
///
/// Returns `(sample_rate * duration_sec).ceil()` samples at amplitude 1.0.
pub(crate) fn sine_wave(freq_hz: f64, sample_rate: u32, duration_sec: f64) -> Vec<f32> {
    sine_wave_full(freq_hz, sample_rate, duration_sec, 1.0)
}

/// Generate a sine wave with the given frequency, sample rate, and sample count.
///
/// Returns exactly `num_samples` samples at amplitude 1.0.
pub(crate) fn sine_wave_samples(freq_hz: f64, sample_rate: u32, num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| {
            let t = i as f64 / f64::from(sample_rate);
            (2.0 * std::f64::consts::PI * freq_hz * t).sin() as f32
        })
        .collect()
}

/// Generate a sine wave with explicit amplitude control.
///
/// Returns `(sample_rate * duration_sec).ceil()` samples scaled by `amplitude`.
pub(crate) fn sine_wave_full(
    freq_hz: f64,
    sample_rate: u32,
    duration_sec: f64,
    amplitude: f32,
) -> Vec<f32> {
    let n = (f64::from(sample_rate) * duration_sec).ceil() as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(sample_rate);
            amplitude * (2.0 * std::f64::consts::PI * freq_hz * t).sin() as f32
        })
        .collect()
}
