// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! FFT-based spectral analysis: band energy, tilt, HNR, autocorrelation.
//!
//! Extracted from `dsp.rs` (Part of #1970 D5).

use crate::error::{DspErrorKind, TtsVerifyError};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Compute per-band spectral energy via FFT.
///
/// Returns energy in dB for `n_bands` equal-width frequency bands.
/// Uses a single Hann-windowed FFT over the entire signal.
pub fn spectral_band_energy(
    samples: &[f32],
    sample_rate: u32,
    n_bands: usize,
) -> Result<Vec<f64>, TtsVerifyError> {
    if samples.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    if sample_rate == 0 {
        return Err(TtsVerifyError::InvalidSampleRate(0));
    }
    if n_bands == 0 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
            param: "n_bands must be > 0",
        }));
    }

    let n = samples.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    // Apply Hann window and convert to complex.
    let mut buffer: Vec<Complex<f64>> = samples
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos());
            Complex::new(f64::from(x) * w, 0.0)
        })
        .collect();

    fft.process(&mut buffer);

    // Power spectrum (one-sided: bins 0..n/2+1).
    let n_bins = n / 2 + 1;
    let powers: Vec<f64> = buffer[..n_bins]
        .iter()
        .map(|c| c.norm_sqr() / n as f64)
        .collect();

    // Divide bins into equal-width frequency bands.
    let bins_per_band = n_bins.max(n_bands) / n_bands.max(1);
    let mut band_energy = Vec::with_capacity(n_bands);
    for b in 0..n_bands {
        let start = b * bins_per_band;
        let end = ((b + 1) * bins_per_band).min(n_bins);
        if start >= n_bins {
            band_energy.push(-120.0); // Below noise floor.
            continue;
        }
        let avg: f64 = powers[start..end].iter().sum::<f64>() / (end - start).max(1) as f64;
        // Convert to dB, clamp at -120 dB.
        let db = if avg > 0.0 {
            10.0 * avg.log10()
        } else {
            -120.0
        };
        band_energy.push(db.max(-120.0));
    }

    Ok(band_energy)
}

/// Compute autocorrelation of a signal at a given lag.
pub fn autocorrelation(samples: &[f32], max_lag: usize) -> Vec<f64> {
    let n = samples.len();
    let mut result = Vec::with_capacity(max_lag + 1);
    for lag in 0..=max_lag.min(n.saturating_sub(1)) {
        let sum: f64 = (0..n - lag)
            .map(|i| f64::from(samples[i]) * f64::from(samples[i + lag]))
            .sum();
        result.push(sum);
    }
    result
}

/// Compute spectral tilt (dB/octave) via linear regression on log-power spectrum.
///
/// Returns the slope in dB per octave.
pub fn spectral_tilt(samples: &[f32], sample_rate: u32) -> Result<f64, TtsVerifyError> {
    if samples.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    if sample_rate == 0 {
        return Err(TtsVerifyError::InvalidSampleRate(sample_rate));
    }

    let n = samples.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex<f64>> = samples
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos());
            Complex::new(f64::from(x) * w, 0.0)
        })
        .collect();

    fft.process(&mut buffer);

    let n_bins = n / 2 + 1;
    let freq_resolution = f64::from(sample_rate) / n as f64;

    // Linear regression of log2(power) on log2(frequency).
    // Skip bin 0 (DC) and very low bins.
    let min_bin = (50.0 / freq_resolution).ceil() as usize; // Start at ~50 Hz.
    let max_bin = n_bins.min((8000.0 / freq_resolution).ceil() as usize); // Up to 8 kHz.

    if min_bin >= max_bin || max_bin <= min_bin + 2 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::Computation {
            what: "not enough frequency bins for tilt estimation",
        }));
    }

    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_xy = 0.0_f64;
    let mut sum_xx = 0.0_f64;
    let mut count = 0.0_f64;

    for (bin, buf_val) in buffer
        .iter()
        .enumerate()
        .skip(min_bin)
        .take(max_bin - min_bin)
    {
        let freq = bin as f64 * freq_resolution;
        let power = buf_val.norm_sqr() / n as f64;
        if power <= 0.0 || freq <= 0.0 {
            continue;
        }
        let x = freq.log2(); // log2(Hz)
        let y = 10.0 * power.log10(); // dB
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
        count += 1.0;
    }

    if count < 2.0 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::Computation {
            what: "not enough data points for tilt regression",
        }));
    }

    // Slope in dB per doubling of frequency = dB per octave.
    let slope = (count * sum_xy - sum_x * sum_y) / (count * sum_xx - sum_x.powi(2));
    Ok(slope)
}

/// Compute Harmonic-to-Noise Ratio (HNR) via autocorrelation.
///
/// Citation: Boersma 1993, IFA Proceedings.
/// Returns HNR in dB. Higher = more periodic (more tonal).
pub fn hnr(samples: &[f32], sample_rate: u32) -> Result<f64, TtsVerifyError> {
    if samples.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    if sample_rate == 0 {
        return Err(TtsVerifyError::InvalidSampleRate(sample_rate));
    }

    // Search for pitch period in 50-500 Hz range.
    let min_lag = (f64::from(sample_rate) / 500.0).ceil() as usize;
    let max_lag = (f64::from(sample_rate) / 50.0).floor() as usize;

    if max_lag >= samples.len() || min_lag >= max_lag {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InsufficientSamples {
            operation: "HNR estimation",
            needed: max_lag + 1,
            got: samples.len(),
        }));
    }

    let ac = autocorrelation(samples, max_lag);
    let energy = ac[0]; // Autocorrelation at lag 0 = total energy.

    if energy <= 0.0 {
        return Ok(-120.0); // Silent signal.
    }

    // Find maximum autocorrelation in the pitch range.
    let max_ac = crate::stats::fold_max_propagate_nan(
        ac[min_lag..=max_lag].iter().copied(),
        f64::NEG_INFINITY,
    );

    // HNR = 10 * log10(r_max / (1 - r_max)) where r_max = max_ac / energy.
    let r = (max_ac / energy).clamp(-0.999, 0.999);
    if r <= 0.0 {
        return Ok(-120.0); // No periodicity detected.
    }
    Ok(10.0 * (r / (1.0 - r)).log10())
}
