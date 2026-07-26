// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Quality metrics for TTS audio verification.
//!
//! Each metric returns a `QualityMetric` with value, threshold, and citation.
//! Thresholds are from published literature on speech quality assessment.

use crate::dsp;
use crate::error::{DspErrorKind, TtsVerifyError};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Result of a single quality metric evaluation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct QualityMetric {
    /// Human-readable name of the metric.
    pub name: &'static str,
    /// Measured value.
    pub value: f64,
    /// Threshold for pass/fail.
    pub threshold: f64,
    /// Whether the metric passed.
    pub passed: bool,
    /// Academic citation for the threshold.
    pub citation: &'static str,
}

/// Mel Cepstral Distortion — requires reference PCM.
///
/// MCD measures spectral distance between candidate and reference.
/// Lower is better. Threshold: < 6.0 dB indicates acceptable quality.
///
/// Citation: Kubichek 1993, "Mel-cepstral distance measure for objective
/// speech quality assessment", IEEE ICASSP.
pub fn compute_mcd(
    candidate: &[f32],
    reference: &[f32],
    sample_rate: u32,
    max_mcd_db: f64,
) -> Result<QualityMetric, TtsVerifyError> {
    validate_pair(candidate, reference, sample_rate)?;

    let n_fft = 1024;
    let hop = 256;
    let n_mels = 80;

    let cand_mfcc = compute_mfcc(candidate, sample_rate, n_fft, hop, n_mels, 13)?;
    let ref_mfcc = compute_mfcc(reference, sample_rate, n_fft, hop, n_mels, 13)?;

    let n_frames = cand_mfcc.len().min(ref_mfcc.len());
    if n_frames == 0 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::Computation {
            what: "no MFCC frames to compare",
        }));
    }

    // MCD = (10/ln(10)) * sqrt(2) * mean_over_frames(sqrt(sum_k(c_k - r_k)^2))
    // where k=1..n_coeffs (skip coefficient 0 = energy).
    let scale = 10.0 / 10.0_f64.ln() * 2.0_f64.sqrt();
    let mut total = 0.0_f64;
    for i in 0..n_frames {
        let n_coeffs = cand_mfcc[i].len().min(ref_mfcc[i].len());
        let sq_sum: f64 = (1..n_coeffs) // Skip c0.
            .map(|k| {
                let d = cand_mfcc[i][k] - ref_mfcc[i][k];
                d * d
            })
            .sum();
        total += sq_sum.sqrt();
    }
    let mcd = scale * total / n_frames as f64;

    Ok(QualityMetric {
        name: "mcd",
        value: mcd,
        threshold: max_mcd_db,
        passed: mcd <= max_mcd_db,
        citation: "Kubichek 1993, IEEE ICASSP",
    })
}

/// Harmonic-to-Noise Ratio via autocorrelation.
///
/// HNR measures vocal quality. Higher = more periodic (cleaner voice).
/// Threshold: > 15 dB for normal speech quality.
///
/// Citation: Boersma 1993, "Accurate short-term analysis of the
/// fundamental frequency and the harmonics-to-noise ratio of a sampled
/// sound", IFA Proceedings 17.
pub fn compute_hnr(
    samples: &[f32],
    sample_rate: u32,
    min_hnr_db: f64,
) -> Result<QualityMetric, TtsVerifyError> {
    let value = dsp::hnr(samples, sample_rate)?;
    Ok(QualityMetric {
        name: "hnr",
        value,
        threshold: min_hnr_db,
        passed: value >= min_hnr_db,
        citation: "Boersma 1993, IFA Proceedings",
    })
}

/// Extract fundamental frequency (F0) contour using YIN algorithm.
///
/// Returns F0 values in Hz for each analysis frame. Unvoiced frames = 0.0.
///
/// Citation: de Cheveigné & Kawahara, "YIN, a fundamental frequency
/// estimator for speech and music", JASA 2002.
pub fn extract_f0(samples: &[f32], sample_rate: u32) -> Result<Vec<f64>, TtsVerifyError> {
    if samples.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    if sample_rate == 0 {
        return Err(TtsVerifyError::InvalidSampleRate(sample_rate));
    }
    // Frame size ~40ms, hop ~10ms.
    let frame_size = (f64::from(sample_rate) * 0.04) as usize;
    let hop_size = (f64::from(sample_rate) * 0.01) as usize;
    dsp::yin_f0(samples, sample_rate, frame_size, hop_size, 0.15)
}

/// Check that F0 falls within the expected speech range.
///
/// Normal speech F0: 80-400 Hz (covers male bass to female soprano).
///
/// Citation: Titze 1994, "Principles of Voice Production",
/// Prentice Hall.
pub fn check_f0_range(f0_contour: &[f64], min_hz: f64, max_hz: f64) -> QualityMetric {
    // Only check voiced frames (F0 > 0).
    let voiced: Vec<f64> = f0_contour.iter().copied().filter(|&f| f > 0.0).collect();
    if voiced.is_empty() {
        return QualityMetric {
            name: "f0_range",
            value: 0.0,
            threshold: min_hz,
            passed: false,
            citation: "Titze 1994, Prentice Hall",
        };
    }
    let in_range = voiced
        .iter()
        .filter(|&&f| f >= min_hz && f <= max_hz)
        .count();
    let ratio = in_range as f64 / voiced.len() as f64;
    // Require 80% of voiced frames in range.
    let threshold = 0.8;
    QualityMetric {
        name: "f0_range",
        value: ratio,
        threshold,
        passed: ratio >= threshold,
        citation: "Titze 1994, Prentice Hall",
    }
}

/// Compute spectral tilt (dB/octave).
///
/// Speech has a characteristic spectral tilt of -3 to -12 dB/octave.
/// Flat tilt suggests noise; steep tilt suggests muffled/low-pass output.
///
/// Citation: Fant 1960, "Acoustic Theory of Speech Production", Mouton.
pub fn compute_spectral_tilt(
    samples: &[f32],
    sample_rate: u32,
    range: (f64, f64),
) -> Result<QualityMetric, TtsVerifyError> {
    let tilt = dsp::spectral_tilt(samples, sample_rate)?;
    let (min_tilt, max_tilt) = range;
    Ok(QualityMetric {
        name: "spectral_tilt",
        value: tilt,
        threshold: min_tilt, // Reports lower bound as threshold.
        passed: tilt >= min_tilt && tilt <= max_tilt,
        citation: "Fant 1960, Mouton",
    })
}

/// Compute cosine similarity between candidate and reference.
///
/// Used for voice cloning discrimination (dvoice V1 gate #4, V2-4).
/// Higher is better. Threshold: > 0.85 indicates acceptable voice similarity.
pub fn compute_cosine_similarity(
    candidate: &[f32],
    reference: &[f32],
    min_similarity: f64,
) -> Result<QualityMetric, TtsVerifyError> {
    let value = dsp::cosine_similarity(candidate, reference)?;
    Ok(QualityMetric {
        name: "cosine_similarity",
        value,
        threshold: min_similarity,
        passed: value >= min_similarity,
        citation: "dvoice V1/V2 gate condition",
    })
}

/// Compute Signal-to-Noise Ratio between candidate and reference.
///
/// Used for Demucs stem extraction quality (dvoice V1 gate #5).
/// Higher is better. Threshold: > 10 dB indicates acceptable quality.
pub fn compute_snr(
    candidate: &[f32],
    reference: &[f32],
    min_snr_db: f64,
) -> Result<QualityMetric, TtsVerifyError> {
    let value = dsp::snr_db(candidate, reference)?;
    Ok(QualityMetric {
        name: "snr",
        value,
        threshold: min_snr_db,
        passed: value >= min_snr_db,
        citation: "ITU-T P.56",
    })
}

/// Compute Signal-to-Distortion Ratio between candidate and reference.
///
/// SDR measures source separation quality using the BSS_EVAL methodology.
/// Higher is better. Threshold: > 5 dB indicates acceptable separation.
///
/// Citation: Vincent et al. 2006, "Performance measurement in blind audio
/// source separation", IEEE TASLP.
pub fn compute_sdr(
    candidate: &[f32],
    reference: &[f32],
    min_sdr_db: f64,
) -> Result<QualityMetric, TtsVerifyError> {
    let value = dsp::sdr_db(candidate, reference)?;
    Ok(QualityMetric {
        name: "sdr",
        value,
        threshold: min_sdr_db,
        passed: value >= min_sdr_db,
        citation: "Vincent et al. 2006, IEEE TASLP",
    })
}

/// Compute RMS energy of a signal.
///
/// Used across all dvoice gate tests for non-silence and level verification.
/// Returns RMS in linear scale. Threshold is a minimum level.
pub fn compute_rms(samples: &[f32], min_rms: f64) -> Result<QualityMetric, TtsVerifyError> {
    if samples.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    let value = dsp::rms(samples);
    Ok(QualityMetric {
        name: "rms_energy",
        value,
        threshold: min_rms,
        passed: value >= min_rms,
        citation: "ITU-R BS.1770-4",
    })
}

// -- Internal helpers --------------------------------------------------------

fn validate_pair(
    candidate: &[f32],
    reference: &[f32],
    sample_rate: u32,
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
    Ok(())
}

/// Compute MFCCs for a signal: STFT → mel filterbank → log → DCT.
fn compute_mfcc(
    samples: &[f32],
    sample_rate: u32,
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    n_coeffs: usize,
) -> Result<Vec<Vec<f64>>, TtsVerifyError> {
    let mel_fb = dsp::mel_filterbank(sample_rate, n_fft, n_mels);

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let n_bins = n_fft / 2 + 1;
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

        // Power spectrum.
        let power: Vec<f64> = buffer[..n_bins]
            .iter()
            .map(|c| c.norm_sqr() / n_fft as f64)
            .collect();

        // Apply mel filterbank → log.
        let mel_spec: Vec<f64> = mel_fb
            .iter()
            .map(|filter| {
                let energy: f64 = filter.iter().zip(power.iter()).map(|(&f, &p)| f * p).sum();
                (energy.max(1e-10)).ln()
            })
            .collect();

        // Type-II DCT to get MFCCs.
        let mfcc: Vec<f64> = (0..n_coeffs)
            .map(|k| {
                mel_spec
                    .iter()
                    .enumerate()
                    .map(|(n, &val)| {
                        val * (std::f64::consts::PI * k as f64 * (n as f64 + 0.5) / n_mels as f64)
                            .cos()
                    })
                    .sum::<f64>()
            })
            .collect();

        frames.push(mfcc);
        start += hop;
    }

    if frames.is_empty() {
        return Err(TtsVerifyError::Dsp(DspErrorKind::Computation {
            what: "audio too short for MFCC computation",
        }));
    }

    Ok(frames)
}

#[cfg(kani)]
#[path = "quality_kani.rs"]
mod kani_proofs;
