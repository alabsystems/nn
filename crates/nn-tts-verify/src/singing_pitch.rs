// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pitch accuracy verification for singing voice synthesis.
//!
//! Compares per-frame F0 from YIN extraction against expected MIDI note
//! frequencies from a musical score. Reports per-note deviation in cents,
//! in-tune fraction, and onset latency.

use crate::dsp;
use crate::error::{DspErrorKind, TtsVerifyError};
use crate::singing::{hz_to_cents, midi_to_hz, MusicalScore};

// ---------------------------------------------------------------------------
// Configuration & result types
// ---------------------------------------------------------------------------

/// Configuration for pitch accuracy verification.
#[derive(Debug, Clone)]
pub struct PitchAccuracyConfig {
    /// Maximum allowed pitch deviation in cents (default: 50 = quarter-tone).
    pub max_deviation_cents: f64,
    /// Fraction of note that must be within tolerance (default: 0.80).
    pub in_tune_fraction: f64,
    /// Pitch onset tolerance in seconds (default: 0.030).
    /// How quickly pitch must reach target after note onset.
    pub onset_tolerance_sec: f64,
    /// Minimum note duration to verify (skip very short notes).
    pub min_note_duration_sec: f64,
}

impl Default for PitchAccuracyConfig {
    fn default() -> Self {
        Self {
            max_deviation_cents: 50.0,
            in_tune_fraction: 0.80,
            onset_tolerance_sec: 0.030,
            min_note_duration_sec: 0.05,
        }
    }
}

/// Result of pitch accuracy verification for one note.
#[derive(Debug, Clone)]
pub struct NoteAccuracyResult {
    /// MIDI note number that was tested.
    pub midi_note: u8,
    /// Target frequency in Hz.
    pub target_hz: f64,
    /// Mean F0 during the stable portion of the note.
    pub mean_f0_hz: f64,
    /// Deviation from target in cents.
    pub deviation_cents: f64,
    /// Fraction of frames within tolerance.
    pub in_tune_fraction: f64,
    /// Time to reach target pitch (onset latency).
    pub onset_latency_sec: f64,
    /// Whether the note passed all criteria.
    pub passed: bool,
}

// ---------------------------------------------------------------------------
// YIN configuration for singing (wider range than speech)
// ---------------------------------------------------------------------------

/// Frame size for F0 extraction (2048 at 24 kHz ≈ 85 ms).
const YIN_FRAME_SIZE: usize = 2048;
/// Hop size (256 at 24 kHz ≈ 10.7 ms).
const YIN_HOP_SIZE: usize = 256;
/// YIN correlation threshold for singing voice.
const YIN_THRESHOLD: f64 = 0.15;

// ---------------------------------------------------------------------------
// Pitch accuracy verification
// ---------------------------------------------------------------------------

/// Verify pitch accuracy of sung audio against a musical score.
///
/// Uses YIN F0 extraction, then compares per-frame F0 against expected
/// MIDI note frequencies. Each non-rest note is evaluated for in-tune
/// fraction and onset latency.
///
/// # Arguments
///
/// * `samples` — PCM audio (mono, f32).
/// * `score` — Musical score with notes and tempo.
/// * `config` — Pitch accuracy thresholds.
/// * `sample_rate` — Audio sample rate in Hz.
pub fn verify_pitch_accuracy(
    samples: &[f32],
    score: &MusicalScore,
    config: &PitchAccuracyConfig,
    sample_rate: u32,
) -> Result<Vec<NoteAccuracyResult>, TtsVerifyError> {
    if samples.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    if sample_rate == 0 {
        return Err(TtsVerifyError::InvalidSampleRate(sample_rate));
    }
    score.validate()?;

    // Extract F0 contour using YIN.
    let f0 = dsp::yin_f0(
        samples,
        sample_rate,
        YIN_FRAME_SIZE,
        YIN_HOP_SIZE,
        YIN_THRESHOLD,
    )?;
    if f0.is_empty() {
        return Err(TtsVerifyError::Dsp(DspErrorKind::EmptyInput {
            what: "F0 contour is empty",
        }));
    }

    let hop_sec = YIN_HOP_SIZE as f64 / f64::from(sample_rate);
    let mut results = Vec::new();

    for note in &score.notes {
        // Skip rests and very short notes.
        if note.is_rest || note.duration_sec < config.min_note_duration_sec {
            continue;
        }

        let target_hz = midi_to_hz(note.midi_note);

        // Map note time range to F0 frame indices.
        let start_frame = (note.onset_sec / hop_sec).floor() as usize;
        let end_frame = ((note.onset_sec + note.duration_sec) / hop_sec).ceil() as usize;
        let end_frame = end_frame.min(f0.len());

        if start_frame >= end_frame {
            continue;
        }

        // Skip onset frames for stable pitch evaluation.
        let onset_frames = (config.onset_tolerance_sec / hop_sec).ceil() as usize;
        let stable_start = (start_frame + onset_frames).min(end_frame);

        // Compute per-frame deviations over the stable portion.
        let stable_f0 = &f0[stable_start..end_frame];
        let voiced: Vec<f64> = stable_f0.iter().copied().filter(|&v| v > 0.0).collect();

        let (mean_f0, deviation, in_tune_frac) = if voiced.is_empty() {
            (0.0, f64::MAX, 0.0)
        } else {
            let mean = voiced.iter().sum::<f64>() / voiced.len() as f64;
            let dev = hz_to_cents(mean, target_hz).abs();
            let in_tune = voiced
                .iter()
                .filter(|&&v| hz_to_cents(v, target_hz).abs() < config.max_deviation_cents)
                .count() as f64
                / voiced.len() as f64;
            (mean, dev, in_tune)
        };

        // Compute onset latency: time from note onset to first in-tune frame.
        let onset_latency = compute_onset_latency(
            &f0[start_frame..end_frame],
            target_hz,
            config.max_deviation_cents,
            hop_sec,
        );

        let passed = in_tune_frac >= config.in_tune_fraction;

        results.push(NoteAccuracyResult {
            midi_note: note.midi_note,
            target_hz,
            mean_f0_hz: mean_f0,
            deviation_cents: deviation,
            in_tune_fraction: in_tune_frac,
            onset_latency_sec: onset_latency,
            passed,
        });
    }

    Ok(results)
}

/// Compute onset latency: time from note start to first in-tune frame.
fn compute_onset_latency(
    f0_segment: &[f64],
    target_hz: f64,
    tolerance_cents: f64,
    hop_sec: f64,
) -> f64 {
    for (i, &f0) in f0_segment.iter().enumerate() {
        if f0 > 0.0 && hz_to_cents(f0, target_hz).abs() < tolerance_cents {
            return i as f64 * hop_sec;
        }
    }
    // Never reached target pitch.
    f0_segment.len() as f64 * hop_sec
}
