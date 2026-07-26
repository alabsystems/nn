// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-resolution STFT loss as a verification metric.
//!
//! Computes spectral convergence + log spectral distance at multiple FFT sizes
//! (512, 1024, 2048), then averages across resolutions. This catches
//! frequency-domain artifacts that time-domain metrics miss.
//!
//! Citation: Yamamoto et al. 2020, "Parallel WaveGAN: A fast waveform
//! generation model based on generative adversarial networks with
//! multi-resolution spectrogram", ICASSP.

use crate::error::{validate_finite_positive, DspErrorKind, InvalidConfigKind, TtsVerifyError};
use crate::quality::QualityMetric;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Default FFT sizes for multi-resolution analysis.
const DEFAULT_FFT_SIZES: [usize; 3] = [512, 1024, 2048];

/// Configuration for multi-resolution STFT comparison.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MultiResStftConfig {
    /// FFT sizes for multi-resolution analysis. Default: [512, 1024, 2048].
    pub fft_sizes: Vec<usize>,
    /// Maximum acceptable average loss. Default: 1.0.
    pub max_loss: f64,
}

impl MultiResStftConfig {
    /// Validate that all f64 fields are finite and positive.
    pub fn validate(&self) -> Result<(), TtsVerifyError> {
        validate_finite_positive(self.max_loss, "max_loss")?;
        if self.fft_sizes.is_empty() {
            return Err(TtsVerifyError::InvalidConfig(
                InvalidConfigKind::Constraint {
                    what: "fft_sizes must not be empty",
                },
            ));
        }
        Ok(())
    }
}

impl Default for MultiResStftConfig {
    fn default() -> Self {
        Self {
            fft_sizes: DEFAULT_FFT_SIZES.to_vec(),
            max_loss: 1.0,
        }
    }
}

/// Compute multi-resolution STFT loss between candidate and reference.
///
/// For each FFT size, computes:
/// - **Spectral convergence**: Frobenius norm of STFT difference / Frobenius norm of reference STFT
/// - **Log spectral distance**: L1 norm of log magnitude difference
///
/// Returns the average of (spectral_convergence + log_spectral_distance) across
/// all resolutions. Lower is better.
pub fn compute_multi_res_stft(
    candidate: &[f32],
    reference: &[f32],
    sample_rate: u32,
    config: &MultiResStftConfig,
) -> Result<QualityMetric, TtsVerifyError> {
    validate_stft_inputs(candidate, reference, sample_rate, config)?;

    let mut total_loss = 0.0_f64;
    let mut count = 0;

    for &n_fft in &config.fft_sizes {
        let hop = n_fft / 4;
        let (sc, lsd) = single_resolution_stft_loss(candidate, reference, n_fft, hop)?;
        total_loss += sc + lsd;
        count += 1;
    }

    let avg_loss = if count > 0 {
        total_loss / f64::from(count)
    } else {
        0.0
    };

    Ok(QualityMetric {
        name: "multi_res_stft_loss",
        value: avg_loss,
        threshold: config.max_loss,
        passed: avg_loss <= config.max_loss,
        citation: "Yamamoto et al. 2020, ICASSP",
    })
}

/// Compute spectral convergence and log spectral distance for a single FFT size.
///
/// Returns (spectral_convergence, log_spectral_distance).
fn single_resolution_stft_loss(
    candidate: &[f32],
    reference: &[f32],
    n_fft: usize,
    hop: usize,
) -> Result<(f64, f64), TtsVerifyError> {
    let cand_mag = stft_magnitude(candidate, n_fft, hop)?;
    let ref_mag = stft_magnitude(reference, n_fft, hop)?;

    let n_frames = cand_mag.len().min(ref_mag.len());
    if n_frames == 0 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::Computation {
            what: "audio too short for STFT at this resolution",
        }));
    }

    let n_bins = n_fft / 2 + 1;

    // Spectral convergence: ||S_ref - S_cand||_F / ||S_ref||_F
    let mut diff_sq_sum = 0.0_f64;
    let mut ref_sq_sum = 0.0_f64;

    // Log spectral distance: mean(|log(S_ref + eps) - log(S_cand + eps)|)
    let eps = 1e-7_f64;
    let mut log_diff_sum = 0.0_f64;
    let mut element_count = 0_u64;

    for frame in 0..n_frames {
        for bin in 0..n_bins {
            let r = ref_mag[frame][bin];
            let c = cand_mag[frame][bin];
            let d = r - c;

            diff_sq_sum += d * d;
            ref_sq_sum += r * r;

            log_diff_sum += ((r + eps).ln() - (c + eps).ln()).abs();
            element_count += 1;
        }
    }

    let sc = if ref_sq_sum > 0.0 {
        (diff_sq_sum / ref_sq_sum).sqrt()
    } else {
        0.0
    };

    let lsd = if element_count > 0 {
        log_diff_sum / element_count as f64
    } else {
        0.0
    };

    Ok((sc, lsd))
}

/// Compute STFT magnitude spectrogram.
///
/// Returns a Vec of frames, each frame is a Vec of magnitude values for n_fft/2+1 bins.
pub(crate) fn stft_magnitude(
    samples: &[f32],
    n_fft: usize,
    hop: usize,
) -> Result<Vec<Vec<f64>>, TtsVerifyError> {
    if samples.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    if n_fft == 0 || hop == 0 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
            param: "n_fft and hop must be > 0",
        }));
    }

    let n_bins = n_fft / 2 + 1;
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut frames = Vec::new();
    let mut start = 0;

    while start + n_fft <= samples.len() {
        // Apply Hann window.
        let mut buffer: Vec<Complex<f64>> = (0..n_fft)
            .map(|i| {
                let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos());
                Complex::new(f64::from(samples[start + i]) * w, 0.0)
            })
            .collect();

        fft.process(&mut buffer);

        // Magnitude spectrum (not power — magnitude for STFT loss).
        let mag: Vec<f64> = buffer[..n_bins].iter().map(|c| c.norm()).collect();
        frames.push(mag);

        start += hop;
    }

    Ok(frames)
}

fn validate_stft_inputs(
    candidate: &[f32],
    reference: &[f32],
    sample_rate: u32,
    config: &MultiResStftConfig,
) -> Result<(), TtsVerifyError> {
    if candidate.is_empty() || reference.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    if sample_rate == 0 {
        return Err(TtsVerifyError::InvalidSampleRate(sample_rate));
    }
    if candidate.len() != reference.len() {
        return Err(TtsVerifyError::LengthMismatch {
            candidate: candidate.len(),
            reference: reference.len(),
        });
    }
    if config.fft_sizes.is_empty() {
        return Err(TtsVerifyError::Dsp(DspErrorKind::EmptyInput {
            what: "fft_sizes must not be empty",
        }));
    }
    for &size in &config.fft_sizes {
        if size == 0 || size & (size - 1) != 0 {
            return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
                param: "FFT size must be a power of 2",
            }));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "multi_res_stft_tests.rs"]
mod tests;
