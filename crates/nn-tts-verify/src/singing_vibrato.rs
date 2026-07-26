// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Vibrato extraction and verification for singing voice synthesis.
//!
//! Extracts vibrato parameters (rate, depth, onset) from sustained notes
//! using autocorrelation-based period detection on the detrended F0 contour.
//! Verifies parameters against configurable quality standards.

use crate::dsp;
use crate::error::TtsVerifyError;
use crate::quality::QualityMetric;
use crate::singing::{hz_to_cents, MusicalScore};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Vibrato parameters for a single sustained note.
#[derive(Debug, Clone)]
pub struct VibratoParams {
    /// Vibrato rate in Hz (expected: 4-8 Hz for natural singing).
    pub rate_hz: f64,
    /// Vibrato depth in cents (expected: 20-100 cents).
    pub depth_cents: f64,
    /// Vibrato onset time (seconds after note onset).
    pub onset_sec: f64,
    /// Is vibrato present? (rate > 0 and depth > threshold).
    pub present: bool,
}

/// Configuration for vibrato verification.
#[derive(Debug, Clone)]
pub struct VibratoConfig {
    /// Acceptable vibrato rate range in Hz.
    pub rate_range_hz: (f64, f64),
    /// Acceptable vibrato depth range in cents.
    pub depth_range_cents: (f64, f64),
    /// Minimum vibrato onset delay in seconds.
    pub min_onset_sec: f64,
    /// Minimum note duration to analyze vibrato.
    pub min_note_duration_sec: f64,
}

impl Default for VibratoConfig {
    fn default() -> Self {
        Self {
            rate_range_hz: (4.0, 8.0),
            depth_range_cents: (20.0, 100.0),
            min_onset_sec: 0.15,
            min_note_duration_sec: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// YIN constants (same as singing_pitch.rs)
// ---------------------------------------------------------------------------

const YIN_FRAME_SIZE: usize = 2048;
const YIN_HOP_SIZE: usize = 256;
const YIN_THRESHOLD: f64 = 0.15;

/// Minimum depth in cents to consider vibrato present.
const VIBRATO_MIN_DEPTH_CENTS: f64 = 10.0;

// ---------------------------------------------------------------------------
// Vibrato extraction
// ---------------------------------------------------------------------------

/// Extract vibrato parameters from a sustained note's F0 contour.
///
/// Methodology:
/// 1. Detrend F0 contour (remove linear drift)
/// 2. Compute autocorrelation to find vibrato period
/// 3. Rate = 1 / period
/// 4. Depth = std(detrended) * 2.83 (≈ peak-to-peak for sinusoidal)
/// 5. Convert depth to cents relative to mean F0
pub fn extract_vibrato(f0_contour: &[f64], hop_size_sec: f64) -> VibratoParams {
    let absent = VibratoParams {
        rate_hz: 0.0,
        depth_cents: 0.0,
        onset_sec: 0.0,
        present: false,
    };

    // Need at least ~4 vibrato cycles to measure reliably.
    if f0_contour.len() < 20 {
        return absent;
    }

    // Filter out unvoiced frames (f0 == 0).
    let voiced: Vec<(usize, f64)> = f0_contour
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .map(|(i, &v)| (i, v))
        .collect();

    if voiced.len() < 20 {
        return absent;
    }

    let mean_f0 = voiced.iter().map(|(_, v)| v).sum::<f64>() / voiced.len() as f64;
    if mean_f0 <= 0.0 {
        return absent;
    }

    let detrended = detrend_voiced(&voiced);

    let (rate_hz, depth_cents) = match measure_rate_and_depth(&detrended, hop_size_sec, mean_f0) {
        Some(rd) => rd,
        None => return absent,
    };

    let onset_sec = detect_vibrato_onset(&detrended, hop_size_sec, mean_f0);
    let present = depth_cents >= VIBRATO_MIN_DEPTH_CENTS;

    VibratoParams {
        rate_hz,
        depth_cents,
        onset_sec,
        present,
    }
}

/// Measure vibrato rate and depth from detrended F0 via autocorrelation.
///
/// Returns `Some((rate_hz, depth_cents))` if a vibrato period is detected,
/// `None` if no significant periodicity is found.
fn measure_rate_and_depth(
    detrended: &[f64],
    hop_size_sec: f64,
    mean_f0: f64,
) -> Option<(f64, f64)> {
    let detrended_f32: Vec<f32> = detrended.iter().map(|&v| v as f32).collect();

    // Search range: 3 Hz to 12 Hz vibrato.
    let min_lag = (1.0 / 12.0 / hop_size_sec).ceil() as usize;
    let max_lag = (1.0 / 3.0 / hop_size_sec).floor() as usize;
    let max_lag = max_lag.min(detrended_f32.len() / 2);

    if min_lag >= max_lag || max_lag == 0 {
        return None;
    }

    let acf = dsp::autocorrelation(&detrended_f32, max_lag);
    let acf_at_0 = if acf[0] > 0.0 { acf[0] } else { return None };
    let normalized: Vec<f64> = acf.iter().map(|&v| v / acf_at_0).collect();

    let lag = find_first_peak(&normalized, min_lag, max_lag)?;
    let rate = 1.0 / (lag as f64 * hop_size_sec);

    // Depth: std of detrended * 2.83 ≈ peak-to-peak for sinusoidal.
    let std_dev = compute_std(detrended);
    let depth_hz = std_dev * 2.83;
    let depth = hz_to_cents(mean_f0 + depth_hz / 2.0, mean_f0 - depth_hz / 2.0).abs();

    Some((rate, depth))
}

/// Verify vibrato parameters against singing quality standards.
pub fn verify_vibrato(params: &VibratoParams, config: &VibratoConfig) -> QualityMetric {
    let (lo_rate, hi_rate) = config.rate_range_hz;
    let (lo_depth, hi_depth) = config.depth_range_cents;

    let rate_ok = !params.present || (params.rate_hz >= lo_rate && params.rate_hz <= hi_rate);
    let depth_ok =
        !params.present || (params.depth_cents >= lo_depth && params.depth_cents <= hi_depth);
    let onset_ok = !params.present || params.onset_sec >= config.min_onset_sec;

    let passed = rate_ok && depth_ok && onset_ok;

    QualityMetric {
        name: "vibrato_quality",
        value: params.rate_hz,
        threshold: hi_rate,
        passed,
        citation: "Sundberg 1994, The Science of the Singing Voice",
    }
}

/// Verify vibrato for all sustained notes in a score.
pub fn verify_score_vibrato(
    samples: &[f32],
    score: &MusicalScore,
    config: &VibratoConfig,
    sample_rate: u32,
) -> Result<Vec<(usize, VibratoParams, QualityMetric)>, TtsVerifyError> {
    if samples.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    if sample_rate == 0 {
        return Err(TtsVerifyError::InvalidSampleRate(sample_rate));
    }
    score.validate()?;

    let f0 = dsp::yin_f0(
        samples,
        sample_rate,
        YIN_FRAME_SIZE,
        YIN_HOP_SIZE,
        YIN_THRESHOLD,
    )?;
    let hop_sec = YIN_HOP_SIZE as f64 / f64::from(sample_rate);
    let mut results = Vec::new();

    for (i, note) in score.notes.iter().enumerate() {
        if note.is_rest || note.duration_sec < config.min_note_duration_sec {
            continue;
        }

        let start_frame = (note.onset_sec / hop_sec).floor() as usize;
        let end_frame = ((note.onset_sec + note.duration_sec) / hop_sec).ceil() as usize;
        let end_frame = end_frame.min(f0.len());

        if start_frame >= end_frame {
            continue;
        }

        let note_f0 = &f0[start_frame..end_frame];
        let params = extract_vibrato(note_f0, hop_sec);
        let metric = verify_vibrato(&params, config);
        results.push((i, params, metric));
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Remove linear trend from voiced F0 values.
fn detrend_voiced(voiced: &[(usize, f64)]) -> Vec<f64> {
    let n = voiced.len() as f64;
    let mean_x = voiced.iter().map(|(i, _)| *i as f64).sum::<f64>() / n;
    let mean_y = voiced.iter().map(|(_, v)| v).sum::<f64>() / n;

    let mut cov_xy = 0.0;
    let mut var_x = 0.0;
    for &(i, v) in voiced {
        let dx = i as f64 - mean_x;
        cov_xy += dx * (v - mean_y);
        var_x += dx * dx;
    }

    let slope = if var_x > 0.0 { cov_xy / var_x } else { 0.0 };
    let intercept = mean_y - slope * mean_x;

    voiced
        .iter()
        .map(|&(i, v)| v - (slope * i as f64 + intercept))
        .collect()
}

/// Compute standard deviation of a slice.
fn compute_std(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

/// Find the first peak in a normalized autocorrelation within a lag range.
fn find_first_peak(acf: &[f64], min_lag: usize, max_lag: usize) -> Option<usize> {
    let max_lag = max_lag.min(acf.len().saturating_sub(1));
    if min_lag >= max_lag {
        return None;
    }

    // Look for the first local maximum above 0.3 (significant correlation).
    ((min_lag + 1)..max_lag)
        .find(|&lag| acf[lag] > acf[lag - 1] && acf[lag] >= acf[lag + 1] && acf[lag] > 0.3)
}

/// Detect vibrato onset: first point where local F0 variation exceeds threshold.
fn detect_vibrato_onset(detrended: &[f64], hop_sec: f64, mean_f0: f64) -> f64 {
    if detrended.len() < 10 || mean_f0 <= 0.0 {
        return 0.0;
    }

    // Sliding window of ~50 ms to detect onset of oscillation.
    let window = (0.050 / hop_sec).ceil() as usize;
    let window = window.max(3).min(detrended.len());

    // Threshold: local std must represent > 5 cents of variation.
    let threshold_hz = mean_f0 * (2.0_f64.powf(5.0 / 1200.0) - 1.0);

    for start in 0..detrended.len().saturating_sub(window) {
        let local = &detrended[start..start + window];
        let local_std = compute_std(local);
        if local_std > threshold_hz {
            return start as f64 * hop_sec;
        }
    }

    detrended.len() as f64 * hop_sec
}
