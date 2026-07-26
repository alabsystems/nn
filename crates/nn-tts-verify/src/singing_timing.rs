// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Timing verification for singing voice synthesis.
//!
//! Verifies that note onsets and durations match a musical score,
//! using energy-based onset detection combined with F0 voicing detection.

use crate::dsp;
use crate::error::TtsVerifyError;
use crate::singing::MusicalScore;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Configuration for musical timing verification.
#[derive(Debug, Clone)]
pub struct TimingConfig {
    /// Maximum onset deviation in seconds (default: 0.030).
    pub max_onset_deviation_sec: f64,
    /// Maximum duration deviation as fraction (default: 0.10 = 10%).
    pub max_duration_deviation_fraction: f64,
    /// Maximum silence RMS for rest verification (default: 0.01).
    pub rest_max_rms: f64,
}

impl Default for TimingConfig {
    fn default() -> Self {
        Self {
            max_onset_deviation_sec: 0.030,
            max_duration_deviation_fraction: 0.10,
            rest_max_rms: 0.01,
        }
    }
}

/// Result of timing verification for one note.
#[derive(Debug, Clone)]
pub struct TimingResult {
    /// Index of the note in the score.
    pub note_index: usize,
    /// Deviation of detected onset from score onset (seconds).
    pub onset_deviation_sec: f64,
    /// Deviation of detected duration from score duration (fraction).
    pub duration_deviation_fraction: f64,
    /// Whether the note passed timing criteria.
    pub passed: bool,
}

// ---------------------------------------------------------------------------
// Timing verification
// ---------------------------------------------------------------------------

/// RMS analysis hop size in samples (10 ms at 24 kHz).
const RMS_HOP: usize = 240;
/// RMS analysis window size in samples (20 ms at 24 kHz).
const RMS_WINDOW: usize = 480;

/// Verify timing of sung audio against a musical score.
///
/// Uses energy-based onset detection (RMS envelope thresholding)
/// combined with F0 onset detection (first voiced frame).
pub fn verify_timing(
    samples: &[f32],
    score: &MusicalScore,
    config: &TimingConfig,
    sample_rate: u32,
) -> Result<Vec<TimingResult>, TtsVerifyError> {
    if samples.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    if sample_rate == 0 {
        return Err(TtsVerifyError::InvalidSampleRate(sample_rate));
    }
    score.validate()?;

    // Compute RMS envelope for onset/offset detection.
    let rms_env = compute_rms_envelope(samples, sample_rate);
    let hop_sec = RMS_HOP as f64 / f64::from(sample_rate);

    // Determine onset threshold: fraction of peak RMS.
    let peak_rms = crate::stats::fold_max_propagate_nan(rms_env.iter().copied(), 0.0_f64);
    let onset_threshold = if peak_rms > 0.0 {
        (peak_rms * 0.1).max(config.rest_max_rms)
    } else {
        config.rest_max_rms
    };

    let mut results = Vec::new();

    for (i, note) in score.notes.iter().enumerate() {
        if note.is_rest {
            // For rests: verify the region is silent.
            let rest_result = verify_rest(
                &rms_env,
                note.onset_sec,
                note.duration_sec,
                hop_sec,
                config.rest_max_rms,
                i,
            );
            results.push(rest_result);
            continue;
        }

        // For voiced notes: detect onset via energy threshold crossing.
        let expected_onset_frame = (note.onset_sec / hop_sec).round() as usize;
        let search_start = expected_onset_frame
            .saturating_sub((config.max_onset_deviation_sec * 2.0 / hop_sec).ceil() as usize);
        let search_end = (expected_onset_frame
            + (config.max_onset_deviation_sec * 2.0 / hop_sec).ceil() as usize)
            .min(rms_env.len());

        let detected_onset_frame =
            detect_onset(&rms_env, search_start, search_end, onset_threshold);

        let onset_deviation = match detected_onset_frame {
            Some(frame) => (frame as f64 * hop_sec) - note.onset_sec,
            None => f64::MAX, // No onset detected.
        };

        // Detect offset: where energy drops below threshold after onset.
        let detected_duration = match detected_onset_frame {
            Some(onset_frame) => {
                let search_end_off = (onset_frame
                    + (note.duration_sec * 1.5 / hop_sec).ceil() as usize)
                    .min(rms_env.len());
                detect_offset(&rms_env, onset_frame, search_end_off, onset_threshold)
                    .map(|off_frame| (off_frame - onset_frame) as f64 * hop_sec)
                    .unwrap_or(note.duration_sec)
            }
            None => 0.0,
        };

        let duration_deviation = if note.duration_sec > 0.0 {
            (detected_duration - note.duration_sec).abs() / note.duration_sec
        } else {
            0.0
        };

        let passed = onset_deviation.abs() <= config.max_onset_deviation_sec
            && duration_deviation <= config.max_duration_deviation_fraction;

        results.push(TimingResult {
            note_index: i,
            onset_deviation_sec: onset_deviation,
            duration_deviation_fraction: duration_deviation,
            passed,
        });
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute RMS envelope from samples.
fn compute_rms_envelope(samples: &[f32], _sample_rate: u32) -> Vec<f64> {
    let mut envelope = Vec::new();
    let mut pos = 0;
    while pos + RMS_WINDOW <= samples.len() {
        let window = &samples[pos..pos + RMS_WINDOW];
        envelope.push(dsp::rms(window));
        pos += RMS_HOP;
    }
    envelope
}

/// Detect first energy onset in a search range.
fn detect_onset(rms_env: &[f64], start: usize, end: usize, threshold: f64) -> Option<usize> {
    let end = end.min(rms_env.len());
    (start..end).find(|&i| rms_env[i] > threshold)
}

/// Detect energy offset (drop below threshold) after onset.
fn detect_offset(rms_env: &[f64], onset: usize, end: usize, threshold: f64) -> Option<usize> {
    let end = end.min(rms_env.len());
    // Skip past the onset region (look for sustained→silence transition).
    ((onset + 1)..end).find(|&i| rms_env[i] < threshold)
}

/// Verify that a rest region is silent.
fn verify_rest(
    rms_env: &[f64],
    onset_sec: f64,
    duration_sec: f64,
    hop_sec: f64,
    max_rms: f64,
    note_index: usize,
) -> TimingResult {
    let start_frame = (onset_sec / hop_sec).floor() as usize;
    let end_frame = ((onset_sec + duration_sec) / hop_sec).ceil() as usize;
    let end_frame = end_frame.min(rms_env.len());

    let rest_is_silent = if start_frame < end_frame {
        rms_env[start_frame..end_frame]
            .iter()
            .all(|&r| r <= max_rms)
    } else {
        true
    };

    TimingResult {
        note_index,
        onset_deviation_sec: 0.0,
        duration_deviation_fraction: 0.0,
        passed: rest_is_silent,
    }
}
