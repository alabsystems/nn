// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Short-Time Objective Intelligibility (STOI).
//!
//! Measures speech intelligibility on a 0-1 scale using short-time temporal
//! envelope correlations in 1/3-octave bands. Score >0.7 is generally
//! considered intelligible.
//!
//! Citation: Taal et al. 2011, "An Algorithm for Intelligibility Prediction
//! of Time-Frequency Weighted Noisy Speech", IEEE TASLP.

use crate::error::{DspErrorKind, TtsVerifyError};
use crate::quality::QualityMetric;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// STOI analysis parameters (Taal et al. 2011, Table I).
const STOI_SAMPLE_RATE: u32 = 10_000;
const STOI_N_FFT: usize = 256; // 25.6 ms at 10 kHz
const STOI_HOP: usize = 128; // 50% overlap
const STOI_N_SEGMENTS: usize = 30; // 30-frame segments (~384 ms)

/// 1/3-octave band center frequencies (Hz) from 150 Hz to 4300 Hz.
/// These are the 15 bands specified in the STOI algorithm.
const THIRD_OCTAVE_CENTERS: [f64; 15] = [
    150.0, 188.0, 237.0, 298.0, 375.0, 473.0, 596.0, 750.0, 945.0, 1189.0, 1498.0, 1887.0, 2376.0,
    2993.0, 3770.0,
];

/// Compute STOI between a reference and degraded signal.
///
/// Both signals must have the same length. The algorithm internally resamples
/// to 10 kHz if needed (via simple decimation for common rates).
///
/// Returns a score in [0, 1] where higher = more intelligible.
pub fn compute_stoi(
    reference: &[f32],
    degraded: &[f32],
    sample_rate: u32,
    min_stoi: f64,
) -> Result<QualityMetric, TtsVerifyError> {
    validate_stoi_inputs(reference, degraded, sample_rate)?;

    // Resample to 10 kHz if needed.
    let (ref_10k, deg_10k) = resample_to_10k(reference, degraded, sample_rate)?;

    // Compute STFT magnitude for both signals.
    let ref_stft = stft_magnitudes(&ref_10k, STOI_N_FFT, STOI_HOP)?;
    let deg_stft = stft_magnitudes(&deg_10k, STOI_N_FFT, STOI_HOP)?;

    let n_frames = ref_stft.len().min(deg_stft.len());
    if n_frames < STOI_N_SEGMENTS {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InsufficientSamples {
            operation: "STOI requires at least 30 STFT frames at 10 kHz",
            needed: STOI_N_SEGMENTS,
            got: n_frames,
        }));
    }

    // Compute 1/3-octave band energies for each frame.
    let band_weights = third_octave_band_weights(STOI_N_FFT, STOI_SAMPLE_RATE);
    let ref_bands = compute_band_energies(&ref_stft[..n_frames], &band_weights);
    let deg_bands = compute_band_energies(&deg_stft[..n_frames], &band_weights);

    // Compute per-segment, per-band correlations.
    let n_bands = THIRD_OCTAVE_CENTERS.len();
    let n_segs = n_frames - STOI_N_SEGMENTS + 1;
    let mut total_corr = 0.0_f64;
    let mut count = 0_u64;

    for band in 0..n_bands {
        for seg_start in 0..n_segs {
            let seg_end = seg_start + STOI_N_SEGMENTS;

            // Extract segment vectors for this band.
            let ref_seg: Vec<f64> = (seg_start..seg_end)
                .map(|f| ref_bands[f * n_bands + band])
                .collect();
            let deg_seg: Vec<f64> = (seg_start..seg_end)
                .map(|f| deg_bands[f * n_bands + band])
                .collect();

            // Clipping: normalize degraded segment to have same norm as reference.
            let ref_norm = vector_norm(&ref_seg);
            if ref_norm < 1e-15 {
                continue; // Skip silent segments.
            }

            let deg_norm = vector_norm(&deg_seg);
            let deg_normalized: Vec<f64> = if deg_norm > 1e-15 {
                let scale = ref_norm / deg_norm;
                deg_seg.iter().map(|&x| x * scale).collect()
            } else {
                continue; // Skip silent segments.
            };

            // Correlation coefficient.
            let r = pearson_correlation(&ref_seg, &deg_normalized);
            if r.is_finite() {
                total_corr += r;
                count += 1;
            }
        }
    }

    let stoi_score = if count > 0 {
        (total_corr / count as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };

    Ok(QualityMetric {
        name: "stoi",
        value: stoi_score,
        threshold: min_stoi,
        passed: stoi_score >= min_stoi,
        citation: "Taal et al. 2011, IEEE TASLP",
    })
}

/// Resample signals to 10 kHz via simple decimation.
///
/// Supports common TTS sample rates (8000, 16000, 22050, 24000, 44100, 48000).
/// For rates not evenly divisible by 10000, uses linear interpolation.
fn resample_to_10k(
    reference: &[f32],
    degraded: &[f32],
    sample_rate: u32,
) -> Result<(Vec<f64>, Vec<f64>), TtsVerifyError> {
    if sample_rate == STOI_SAMPLE_RATE {
        let r: Vec<f64> = reference.iter().map(|&x| f64::from(x)).collect();
        let d: Vec<f64> = degraded.iter().map(|&x| f64::from(x)).collect();
        return Ok((r, d));
    }

    let ratio = f64::from(sample_rate) / f64::from(STOI_SAMPLE_RATE);
    let out_len = (reference.len() as f64 / ratio).floor() as usize;

    if out_len < STOI_N_FFT {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InsufficientSamples {
            operation: "STOI resampling to 10 kHz",
            needed: STOI_N_FFT,
            got: out_len,
        }));
    }

    let resample = |input: &[f32]| -> Vec<f64> {
        (0..out_len)
            .map(|i| {
                let src_pos = i as f64 * ratio;
                let idx = src_pos.floor() as usize;
                let frac = src_pos - idx as f64;
                if idx + 1 < input.len() {
                    f64::from(input[idx]) * (1.0 - frac) + f64::from(input[idx + 1]) * frac
                } else if idx < input.len() {
                    f64::from(input[idx])
                } else {
                    0.0
                }
            })
            .collect()
    };

    Ok((resample(reference), resample(degraded)))
}

/// Compute STFT magnitude frames.
fn stft_magnitudes(
    samples: &[f64],
    n_fft: usize,
    hop: usize,
) -> Result<Vec<Vec<f64>>, TtsVerifyError> {
    if samples.len() < n_fft {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InsufficientSamples {
            operation: "STFT magnitude computation",
            needed: n_fft,
            got: samples.len(),
        }));
    }

    let n_bins = n_fft / 2 + 1;
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut frames = Vec::new();
    let mut start = 0;

    while start + n_fft <= samples.len() {
        // Hann window.
        let mut buffer: Vec<Complex<f64>> = (0..n_fft)
            .map(|i| {
                let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos());
                Complex::new(samples[start + i] * w, 0.0)
            })
            .collect();

        fft.process(&mut buffer);

        let mag: Vec<f64> = buffer[..n_bins].iter().map(|c| c.norm()).collect();
        frames.push(mag);
        start += hop;
    }

    Ok(frames)
}

/// Compute 1/3-octave band weights for STFT bins.
///
/// Returns a matrix of [n_bands][n_bins] weights where each row sums to
/// the number of bins in that band.
fn third_octave_band_weights(n_fft: usize, sample_rate: u32) -> Vec<Vec<f64>> {
    let n_bins = n_fft / 2 + 1;
    let freq_resolution = f64::from(sample_rate) / n_fft as f64;
    let n_bands = THIRD_OCTAVE_CENTERS.len();

    // 1/3-octave factor: center * 2^(±1/6).
    let factor_lo = 2.0_f64.powf(-1.0 / 6.0);
    let factor_hi = 2.0_f64.powf(1.0 / 6.0);

    let mut weights = Vec::with_capacity(n_bands);
    for &center in &THIRD_OCTAVE_CENTERS {
        let lo = center * factor_lo;
        let hi = center * factor_hi;

        let mut band = vec![0.0_f64; n_bins];
        for (k, val) in band.iter_mut().enumerate() {
            let freq = k as f64 * freq_resolution;
            if freq >= lo && freq <= hi {
                *val = 1.0;
            }
        }
        weights.push(band);
    }

    weights
}

/// Compute band energies for all frames.
///
/// Returns a flat Vec of [n_frames * n_bands] where element [f*n_bands + b]
/// is the energy in band b of frame f.
fn compute_band_energies(stft_frames: &[Vec<f64>], band_weights: &[Vec<f64>]) -> Vec<f64> {
    let n_bands = band_weights.len();
    let mut energies = Vec::with_capacity(stft_frames.len() * n_bands);

    for frame in stft_frames {
        for band in band_weights {
            let energy: f64 = frame
                .iter()
                .zip(band.iter())
                .map(|(&mag, &w)| mag * mag * w)
                .sum();
            // Use sqrt(energy) as envelope amplitude (Taal et al. use power,
            // but amplitude correlation is equivalent and more numerically stable).
            energies.push(energy.sqrt());
        }
    }

    energies
}

/// Euclidean norm of a vector.
fn vector_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Pearson correlation coefficient between two equal-length vectors.
fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let mean_a: f64 = a.iter().sum::<f64>() / n;
    let mean_b: f64 = b.iter().sum::<f64>() / n;

    let mut cov = 0.0_f64;
    let mut var_a = 0.0_f64;
    let mut var_b = 0.0_f64;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let da = x - mean_a;
        let db = y - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-15 {
        return 0.0;
    }

    (cov / denom).clamp(-1.0, 1.0)
}

fn validate_stoi_inputs(
    reference: &[f32],
    degraded: &[f32],
    sample_rate: u32,
) -> Result<(), TtsVerifyError> {
    if reference.is_empty() || degraded.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    if sample_rate == 0 {
        return Err(TtsVerifyError::InvalidSampleRate(sample_rate));
    }
    if reference.len() != degraded.len() {
        return Err(TtsVerifyError::LengthMismatch {
            candidate: degraded.len(),
            reference: reference.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "stoi_tests.rs"]
mod tests;
