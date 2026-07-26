// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Singing voice synthesis verification — core types and conversions.
//!
//! Provides musical score representation and pitch conversion utilities
//! for verifying singing voice synthesis quality (pitch accuracy, vibrato,
//! timing) against a reference musical score.

use crate::error::{DspErrorKind, InvalidConfigKind, TtsVerifyError};

#[path = "singing_pitch.rs"]
pub mod pitch;
#[path = "singing_timing.rs"]
pub mod timing;
#[path = "singing_vibrato.rs"]
pub mod vibrato;

// ---------------------------------------------------------------------------
// Musical score types
// ---------------------------------------------------------------------------

/// A single note in a musical score.
#[derive(Debug, Clone)]
pub struct ScoreNote {
    /// MIDI note number (60 = C4, 69 = A4 = 440 Hz).
    pub midi_note: u8,
    /// Note onset time in seconds.
    pub onset_sec: f64,
    /// Note duration in seconds.
    pub duration_sec: f64,
    /// Is this a rest? (no expected pitch during this interval).
    pub is_rest: bool,
}

/// Complete musical score for one voice part.
#[derive(Debug, Clone)]
pub struct MusicalScore {
    /// Sequence of notes (including rests) in temporal order.
    pub notes: Vec<ScoreNote>,
    /// Tempo in BPM (used for timing tolerance calculation).
    pub tempo_bpm: f64,
}

impl MusicalScore {
    /// Validate the score for use in verification.
    pub fn validate(&self) -> Result<(), TtsVerifyError> {
        if self.notes.is_empty() {
            return Err(TtsVerifyError::EmptyInput);
        }
        if !self.tempo_bpm.is_finite() || self.tempo_bpm <= 0.0 {
            return Err(TtsVerifyError::InvalidConfig(
                InvalidConfigKind::NonPositive { param: "tempo_bpm" },
            ));
        }
        for note in &self.notes {
            if !note.onset_sec.is_finite() || note.onset_sec < 0.0 {
                return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
                    param: "onset_sec must be finite and non-negative",
                }));
            }
            if !note.duration_sec.is_finite() || note.duration_sec <= 0.0 {
                return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
                    param: "duration_sec must be finite and positive",
                }));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pitch conversion utilities
// ---------------------------------------------------------------------------

/// Convert MIDI note number to frequency in Hz.
///
/// A4 (MIDI 69) = 440 Hz. Each semitone is a factor of 2^(1/12).
///
/// # Examples
///
/// ```
/// # use nn_tts_verify::singing::midi_to_hz;
/// assert!((midi_to_hz(69) - 440.0).abs() < 0.01);  // A4
/// assert!((midi_to_hz(60) - 261.63).abs() < 0.01);  // C4
/// ```
pub fn midi_to_hz(midi_note: u8) -> f64 {
    440.0 * 2.0_f64.powf((f64::from(midi_note) - 69.0) / 12.0)
}

/// Convert the ratio between two frequencies to cents.
///
/// 100 cents = 1 semitone. 1200 cents = 1 octave.
/// Positive means `f1` is higher than `f2`.
///
/// Returns 0.0 if either frequency is non-positive.
pub fn hz_to_cents(f1: f64, f2: f64) -> f64 {
    if f1 <= 0.0 || f2 <= 0.0 {
        return 0.0;
    }
    1200.0 * (f1 / f2).log2()
}

#[cfg(test)]
#[path = "singing_tests.rs"]
mod tests;
