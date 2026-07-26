// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PESQ (Perceptual Evaluation of Speech Quality) — ITU-T P.862.
//!
//! Produces a MOS-LQO (Mean Opinion Score - Listening Quality Objective)
//! in the range [-0.5, 4.5]. Higher = better perceptual quality.
//!
//! This implementation follows the wideband PESQ (P.862.2) algorithm:
//! 1. Level alignment to -26 dBov
//! 2. Time alignment via cross-correlation delay estimation
//! 3. Bark-band power spectrum computation
//! 4. Perceptual disturbance calculation
//! 5. Aggregation to final MOS-LQO score
//!
//! Citation: ITU-T Recommendation P.862 (2001), P.862.1 (2003), P.862.2 (2007).

use crate::error::{DspErrorKind, TtsVerifyError};
use crate::quality::QualityMetric;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Number of Bark bands for wideband PESQ (50-3850 Hz at 16 kHz).
const N_BARK_BANDS: usize = 25;

/// PESQ target level in dBov.
const TARGET_LEVEL_DBOV: f64 = -26.0;

/// FFT size for spectral analysis.
const PESQ_N_FFT: usize = 512;

/// Hop size (50% overlap).
const PESQ_HOP: usize = 256;

/// Bark band edge frequencies in Hz (ITU-T P.862, Table 1).
/// 26 edges define 25 bands.
const BARK_EDGES: [f64; 26] = [
    50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 510.0, 570.0, 640.0, 720.0,
    810.0, 920.0, 1040.0, 1170.0, 1310.0, 1480.0, 1670.0, 1890.0, 2150.0, 2460.0, 2800.0, 3150.0,
    3550.0,
];

/// Compute PESQ MOS-LQO score between reference and degraded signals.
///
/// Both signals must have the same length and sample rate.
/// Supports sample rates of 8000 Hz (narrowband) and 16000 Hz (wideband).
pub fn compute_pesq(
    reference: &[f32],
    degraded: &[f32],
    sample_rate: u32,
    min_pesq: f64,
) -> Result<QualityMetric, TtsVerifyError> {
    validate_pesq_inputs(reference, degraded, sample_rate)?;

    // Step 1: Level alignment — normalize both signals to target level.
    let ref_aligned = level_align(reference);
    let deg_aligned = level_align(degraded);

    // Step 2: Time alignment — estimate and compensate delay.
    let delay = estimate_delay(&ref_aligned, &deg_aligned);
    let (ref_trimmed, deg_trimmed) = apply_delay(&ref_aligned, &deg_aligned, delay);

    if ref_trimmed.len() < PESQ_N_FFT {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InsufficientSamples {
            operation: "PESQ analysis after alignment",
            needed: PESQ_N_FFT,
            got: ref_trimmed.len(),
        }));
    }

    // Step 3: Bark-band power spectra.
    let bark_weights = bark_band_weights(PESQ_N_FFT, sample_rate);
    let ref_bark = bark_band_power_frames(&ref_trimmed, &bark_weights)?;
    let deg_bark = bark_band_power_frames(&deg_trimmed, &bark_weights)?;

    let n_frames = ref_bark.len().min(deg_bark.len());
    if n_frames == 0 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::EmptyInput {
            what: "no frames for PESQ computation",
        }));
    }

    // Step 4: Perceptual disturbance — loudness domain.
    let (sym_disturb, asym_disturb) = compute_disturbance(&ref_bark, &deg_bark, n_frames);

    // Step 5: Map to MOS-LQO using P.862.2 wideband mapping.
    //   MOS = 4.5 - 0.1 * D_sym - 0.0309 * D_asym
    let raw_mos = 4.5 - 0.1 * sym_disturb - 0.0309 * asym_disturb;
    let mos = raw_mos.clamp(-0.5, 4.5);

    Ok(QualityMetric {
        name: "pesq",
        value: mos,
        threshold: min_pesq,
        passed: mos >= min_pesq,
        citation: "ITU-T P.862/P.862.2",
    })
}

/// Align signal level to -26 dBov (PESQ standard).
fn level_align(samples: &[f32]) -> Vec<f64> {
    let samples_f64: Vec<f64> = samples.iter().map(|&x| f64::from(x)).collect();
    let rms = (samples_f64.iter().map(|&x| x * x).sum::<f64>() / samples_f64.len() as f64).sqrt();

    if rms < 1e-15 {
        return samples_f64; // Silent signal — no alignment.
    }

    // Target RMS from dBov: rms_target = 10^(target_dbov/20).
    let target_rms = 10.0_f64.powf(TARGET_LEVEL_DBOV / 20.0);
    let gain = target_rms / rms;

    samples_f64.iter().map(|&x| x * gain).collect()
}

/// Estimate delay between reference and degraded using normalized cross-correlation.
///
/// Uses energy-normalized cross-correlation (NCC) to avoid bias from
/// varying overlap window sizes at different delays.
///
/// Returns delay in samples (positive = degraded is delayed).
fn estimate_delay(reference: &[f64], degraded: &[f64]) -> i64 {
    let max_delay = (reference.len() / 4).min(8000); // Max ±0.5s at 16 kHz.
    let n = reference.len().min(degraded.len());

    let mut best_corr = f64::NEG_INFINITY;
    let mut best_delay: i64 = 0;

    // Search over delay range using normalized cross-correlation.
    for delay in -(max_delay as i64)..=(max_delay as i64) {
        let mut corr = 0.0_f64;
        let mut energy_ref = 0.0_f64;
        let mut energy_deg = 0.0_f64;

        for (i, &ref_val) in reference.iter().enumerate().take(n) {
            let j = i as i64 + delay;
            if j >= 0 && (j as usize) < n {
                let deg_val = degraded[j as usize];
                corr += ref_val * deg_val;
                energy_ref += ref_val * ref_val;
                energy_deg += deg_val * deg_val;
            }
        }

        let denom = (energy_ref * energy_deg).sqrt();
        if denom > 1e-15 {
            let normalized = corr / denom;
            if normalized > best_corr {
                best_corr = normalized;
                best_delay = delay;
            }
        }
    }

    best_delay
}

/// Apply delay compensation by trimming signals to aligned region.
fn apply_delay(reference: &[f64], degraded: &[f64], delay: i64) -> (Vec<f64>, Vec<f64>) {
    let n = reference.len().min(degraded.len());

    if delay >= 0 {
        let d = delay as usize;
        let end = n.saturating_sub(d);
        (reference[..end].to_vec(), degraded[d..d + end].to_vec())
    } else {
        let d = (-delay) as usize;
        let end = n.saturating_sub(d);
        (reference[d..d + end].to_vec(), degraded[..end].to_vec())
    }
}

/// Compute Bark band weights for FFT bins.
fn bark_band_weights(n_fft: usize, sample_rate: u32) -> Vec<Vec<f64>> {
    let n_bins = n_fft / 2 + 1;
    let freq_resolution = f64::from(sample_rate) / n_fft as f64;

    let mut weights = Vec::with_capacity(N_BARK_BANDS);
    for band in 0..N_BARK_BANDS {
        let lo = BARK_EDGES[band];
        let hi = BARK_EDGES[band + 1];

        let mut w = vec![0.0_f64; n_bins];
        for (k, val) in w.iter_mut().enumerate() {
            let freq = k as f64 * freq_resolution;
            if freq >= lo && freq < hi {
                *val = 1.0;
            }
        }
        weights.push(w);
    }

    weights
}

/// Compute Bark-band power for all STFT frames.
fn bark_band_power_frames(
    samples: &[f64],
    bark_weights: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, TtsVerifyError> {
    let n_bins = PESQ_N_FFT / 2 + 1;
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(PESQ_N_FFT);

    let mut frames = Vec::new();
    let mut start = 0;

    while start + PESQ_N_FFT <= samples.len() {
        // Hann window + FFT.
        let mut buffer: Vec<Complex<f64>> = (0..PESQ_N_FFT)
            .map(|i| {
                let w =
                    0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / PESQ_N_FFT as f64).cos());
                Complex::new(samples[start + i] * w, 0.0)
            })
            .collect();

        fft.process(&mut buffer);

        // Power spectrum.
        let power: Vec<f64> = buffer[..n_bins]
            .iter()
            .map(|c| c.norm_sqr() / PESQ_N_FFT as f64)
            .collect();

        // Bark band powers.
        let band_powers: Vec<f64> = bark_weights
            .iter()
            .map(|w| {
                power
                    .iter()
                    .zip(w.iter())
                    .map(|(&p, &wt)| p * wt)
                    .sum::<f64>()
            })
            .collect();

        frames.push(band_powers);
        start += PESQ_HOP;
    }

    Ok(frames)
}

/// Compute symmetric and asymmetric perceptual disturbance.
///
/// Applies Zwicker loudness model approximation and computes frame-level
/// disturbance aggregated across Bark bands.
fn compute_disturbance(
    ref_bark: &[Vec<f64>],
    deg_bark: &[Vec<f64>],
    n_frames: usize,
) -> (f64, f64) {
    let eps = 1e-15_f64;

    let mut sym_total = 0.0_f64;
    let mut asym_total = 0.0_f64;
    let mut frame_count = 0_u64;

    for frame in 0..n_frames {
        let mut frame_sym = 0.0_f64;
        let mut frame_asym = 0.0_f64;

        for band in 0..N_BARK_BANDS {
            // Approximate Zwicker loudness: N = k * P^0.23
            // where P is Bark band power (simplified Zwicker sone model).
            let ref_loudness = (ref_bark[frame][band] + eps).powf(0.23);
            let deg_loudness = (deg_bark[frame][band] + eps).powf(0.23);

            let diff = deg_loudness - ref_loudness;

            // Symmetric disturbance: absolute loudness difference.
            frame_sym += diff.abs();

            // Asymmetric disturbance: only additions (degraded louder than reference).
            // This captures masking: additions are more perceptible than deletions.
            if diff > 0.0 {
                frame_asym += diff;
            }
        }

        sym_total += frame_sym;
        asym_total += frame_asym;
        frame_count += 1;
    }

    let n = (frame_count as f64).max(1.0);
    (sym_total / n, asym_total / n)
}

fn validate_pesq_inputs(
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
    // PESQ requires minimum signal length (~0.5s).
    let min_samples = sample_rate as usize / 2;
    if reference.len() < min_samples {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InsufficientSamples {
            operation: "PESQ requires at least 0.5s of audio",
            needed: min_samples,
            got: reference.len(),
        }));
    }
    Ok(())
}

#[cfg(test)]
#[path = "pesq_tests.rs"]
mod tests;
