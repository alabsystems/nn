// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Minimal DSP utilities for TTS verification.
//!
//! Provides RMS, DC offset, click detection, spectral energy analysis,
//! mel filterbank, autocorrelation, and YIN F0 extraction.

use crate::error::{DspErrorKind, TtsVerifyError};

/// Compute root-mean-square power of a signal.
pub fn rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

/// Compute the DC offset (mean of the signal).
pub fn dc_offset(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&x| f64::from(x)).sum();
    sum / samples.len() as f64
}

/// Find the maximum absolute sample-to-sample difference (click detection).
pub fn max_sample_diff(samples: &[f32]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    crate::stats::fold_max_propagate_nan(
        samples
            .windows(2)
            .map(|w| (f64::from(w[1]) - f64::from(w[0])).abs()),
        0.0_f64,
    )
}

#[path = "dsp_spectral.rs"]
mod spectral;
pub use spectral::{autocorrelation, hnr, spectral_band_energy, spectral_tilt};

/// YIN fundamental frequency (F0) extraction.
///
/// Returns F0 estimates in Hz for each analysis frame.
///
/// Algorithm: de Cheveigné & Kawahara, "YIN, a fundamental frequency
/// estimator for speech and music", JASA 2002.
pub fn yin_f0(
    samples: &[f32],
    sample_rate: u32,
    frame_size: usize,
    hop_size: usize,
    threshold: f64,
) -> Result<Vec<f64>, TtsVerifyError> {
    if samples.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    if sample_rate == 0 {
        return Err(TtsVerifyError::InvalidSampleRate(sample_rate));
    }
    if frame_size == 0 || hop_size == 0 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
            param: "frame_size and hop_size must be > 0",
        }));
    }

    let half_w = frame_size / 2;
    let mut f0_values = Vec::new();

    let mut start = 0;
    while start + frame_size <= samples.len() {
        let frame = &samples[start..start + frame_size];

        // Step 1: Difference function d(tau).
        let mut diff = vec![0.0_f64; half_w];
        for tau in 1..half_w {
            let mut sum = 0.0_f64;
            for j in 0..half_w {
                let delta = f64::from(frame[j]) - f64::from(frame[j + tau]);
                sum += delta * delta;
            }
            diff[tau] = sum;
        }

        // Step 2: Cumulative mean normalized difference function d'(tau).
        let mut cmnd = vec![0.0_f64; half_w];
        cmnd[0] = 1.0;
        let mut running_sum = 0.0_f64;
        for tau in 1..half_w {
            running_sum += diff[tau];
            cmnd[tau] = if running_sum > 0.0 {
                diff[tau] * tau as f64 / running_sum
            } else {
                1.0
            };
        }

        // Step 3: Absolute threshold — find first tau where cmnd < threshold.
        let min_period = (f64::from(sample_rate) / 500.0).ceil() as usize; // max F0 = 500 Hz
        let max_period = (f64::from(sample_rate) / 50.0).floor() as usize; // min F0 = 50 Hz
        let search_end = half_w.min(max_period + 1);

        let mut best_tau = 0;
        for tau in min_period..search_end {
            if cmnd[tau] < threshold {
                // Find the local minimum after this point.
                best_tau = tau;
                while best_tau + 1 < search_end && cmnd[best_tau + 1] < cmnd[best_tau] {
                    best_tau += 1;
                }
                break;
            }
        }

        let f0 = if best_tau > 0 {
            // Step 4: Parabolic interpolation around the minimum.
            let tau_f = if best_tau > 0 && best_tau + 1 < half_w {
                let s0 = cmnd[best_tau - 1];
                let s1 = cmnd[best_tau];
                let s2 = cmnd[best_tau + 1];
                let denom = 2.0 * s1 - s2 - s0;
                if denom.abs() > 1e-10 {
                    best_tau as f64 + (s0 - s2) / (2.0 * denom)
                } else {
                    best_tau as f64
                }
            } else {
                best_tau as f64
            };
            f64::from(sample_rate) / tau_f
        } else {
            0.0 // Unvoiced frame.
        };

        f0_values.push(f0);
        start += hop_size;
    }

    Ok(f0_values)
}

/// Compute a single triangular mel filter weight at frequency bin `k`.
///
/// Given a triangular filter defined by `(f_left, f_center, f_right)`:
/// - Rising slope: `(k - f_left) / (f_center - f_left)` for `f_left <= k <= f_center`
/// - Falling slope: `(f_right - k) / (f_right - f_center)` for `f_center < k <= f_right`
/// - Zero outside the triangle
///
/// Returns a value in `[0.0, 1.0]` when `f_left < f_center < f_right`.
pub fn triangular_filter_weight(k: f64, f_left: f64, f_center: f64, f_right: f64) -> f64 {
    if k >= f_left && k <= f_center && f_center > f_left {
        (k - f_left) / (f_center - f_left)
    } else if k > f_center && k <= f_right && f_right > f_center {
        (f_right - k) / (f_right - f_center)
    } else {
        0.0
    }
}

/// Compute mel filterbank coefficients (triangular filters).
///
/// Returns `n_mels` triangular filter weights for `n_fft / 2 + 1` frequency bins.
pub fn mel_filterbank(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<Vec<f64>> {
    let n_bins = n_fft / 2 + 1;
    let f_max = f64::from(sample_rate) / 2.0;

    use nn_core::audio::{hz_to_mel_htk as hz_to_mel, mel_to_hz_htk as mel_to_hz};

    let mel_min = 0.0;
    let mel_max = hz_to_mel(f_max);

    // n_mels + 2 equally spaced points in mel scale.
    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz to FFT bin indices.
    let bin_points: Vec<f64> = hz_points
        .iter()
        .map(|&f| f * n_fft as f64 / f64::from(sample_rate))
        .collect();

    let mut filters = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
        let f_left = bin_points[m];
        let f_center = bin_points[m + 1];
        let f_right = bin_points[m + 2];

        let filter: Vec<f64> = (0..n_bins)
            .map(|k| triangular_filter_weight(k as f64, f_left, f_center, f_right))
            .collect();
        filters.push(filter);
    }

    filters
}

#[path = "dsp_quality_metrics.rs"]
mod quality_metrics;
pub use quality_metrics::{cosine_similarity, sdr_db, snr_db};

#[cfg(kani)]
#[path = "dsp_kani_proofs.rs"]
mod kani_proofs;
