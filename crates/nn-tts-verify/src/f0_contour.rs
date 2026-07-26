// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! F0 contour correlation for prosody verification.
//!
//! Compares fundamental frequency trajectories between candidate and reference
//! using Pearson correlation on voiced frames. Catches prosody regressions that
//! mean F0 range checks miss: two utterances can have identical mean F0 but
//! completely different intonation patterns.

use crate::error::TtsVerifyError;
use crate::quality::{self, QualityMetric};

/// Compute F0 contour correlation between candidate and reference.
///
/// Extracts F0 contours using YIN, aligns voiced frames, and computes
/// Pearson correlation. Returns a value in [-1.0, 1.0] where 1.0 means
/// identical prosody.
///
/// Frames where either signal is unvoiced (F0 = 0.0) are excluded from
/// the correlation computation.
pub fn compute_f0_contour_correlation(
    candidate: &[f32],
    reference: &[f32],
    sample_rate: u32,
    min_correlation: f64,
) -> Result<QualityMetric, TtsVerifyError> {
    if candidate.is_empty() || reference.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    if sample_rate == 0 {
        return Err(TtsVerifyError::InvalidSampleRate(sample_rate));
    }

    let cand_f0 = quality::extract_f0(candidate, sample_rate)?;
    let ref_f0 = quality::extract_f0(reference, sample_rate)?;

    let correlation = f0_pearson_correlation(&cand_f0, &ref_f0)?;

    Ok(QualityMetric {
        name: "f0_contour_correlation",
        value: correlation,
        threshold: min_correlation,
        passed: correlation >= min_correlation,
        citation: "Prosody assessment via F0 contour correlation",
    })
}

/// Pearson correlation between two F0 contours.
///
/// Only considers frames where both signals are voiced (F0 > 0.0).
/// Returns 0.0 if fewer than 2 co-voiced frames exist.
pub(crate) fn f0_pearson_correlation(a: &[f64], b: &[f64]) -> Result<f64, TtsVerifyError> {
    let n_frames = a.len().min(b.len());

    // Collect co-voiced frames (both F0 > 0).
    let pairs: Vec<(f64, f64)> = (0..n_frames)
        .filter_map(|i| {
            if a[i] > 0.0 && b[i] > 0.0 {
                Some((a[i], b[i]))
            } else {
                None
            }
        })
        .collect();

    if pairs.len() < 2 {
        return Ok(0.0); // Not enough co-voiced frames.
    }

    let n = pairs.len() as f64;
    let mean_a: f64 = pairs.iter().map(|(x, _)| x).sum::<f64>() / n;
    let mean_b: f64 = pairs.iter().map(|(_, y)| y).sum::<f64>() / n;

    let mut cov = 0.0_f64;
    let mut var_a = 0.0_f64;
    let mut var_b = 0.0_f64;

    for &(x, y) in &pairs {
        let da = x - mean_a;
        let db = y - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-15 {
        return Ok(0.0); // Constant F0 in one or both signals.
    }

    Ok((cov / denom).clamp(-1.0, 1.0))
}

#[cfg(test)]
#[path = "f0_contour_tests.rs"]
mod tests;
