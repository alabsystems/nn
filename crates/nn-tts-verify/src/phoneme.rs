// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Phoneme types for per-phoneme pronunciation verification.
//!
//! Phoneme boundaries come from external forced alignment (e.g., Montreal
//! Forced Aligner, wav2vec2-CTC). nn-tts-verify verifies audio quality
//! given known boundaries — it does not perform alignment inference.

use crate::error::{validate_finite, validate_finite_positive, InvalidConfigKind, TtsVerifyError};
use crate::quality::QualityMetric;

/// A single phoneme boundary from external forced alignment.
#[derive(Debug, Clone)]
pub struct PhonemeSpan {
    /// IPA or ARPAbet label (e.g., "θ", "TH", "AH0").
    pub label: String,
    /// Start sample index (inclusive).
    pub start: usize,
    /// End sample index (exclusive).
    pub end: usize,
}

/// Complete alignment for an utterance.
#[derive(Debug, Clone)]
pub struct PhonemeAlignment {
    /// Ordered sequence of phoneme spans covering the utterance.
    pub phonemes: Vec<PhonemeSpan>,
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Total number of samples in the utterance.
    pub total_samples: usize,
}

impl PhonemeAlignment {
    /// Validate that all spans are within bounds and non-empty.
    pub fn validate(&self) -> Result<(), TtsVerifyError> {
        if self.phonemes.is_empty() {
            return Err(TtsVerifyError::InvalidConfig(
                InvalidConfigKind::Constraint {
                    what: "PhonemeAlignment must have at least one phoneme",
                },
            ));
        }
        if self.sample_rate == 0 {
            return Err(TtsVerifyError::InvalidSampleRate(0));
        }
        for span in &self.phonemes {
            if span.start >= span.end {
                return Err(TtsVerifyError::InvalidConfig(
                    InvalidConfigKind::Constraint {
                        what: "phoneme span start must be < end",
                    },
                ));
            }
            if span.end > self.total_samples {
                return Err(TtsVerifyError::InvalidConfig(
                    InvalidConfigKind::Constraint {
                        what: "phoneme span end must not exceed total_samples",
                    },
                ));
            }
        }
        Ok(())
    }
}

/// Quality results for a single phoneme segment.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PhonemeResult {
    /// IPA or ARPAbet label.
    pub label: String,
    /// Duration of this phoneme in milliseconds.
    pub duration_ms: f64,
    /// Quality metrics computed for this segment.
    pub metrics: Vec<QualityMetric>,
    /// Whether all metrics passed for this phoneme.
    pub passed: bool,
}

/// Per-phoneme verification configuration.
///
/// Default thresholds are from Crystal (2003) for duration and
/// Titze (1994) for F0 ranges.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PhonemeVerifyConfig {
    /// Minimum phoneme duration in ms (below = deletion suspect).
    /// Default: 20.0 (Crystal 2003).
    pub min_duration_ms: f64,
    /// Maximum phoneme duration in ms (above = insertion/stall).
    /// Default: 500.0.
    pub max_duration_ms: f64,
    /// MCD threshold per segment in dB (higher = more tolerant).
    /// Default: 8.0.
    pub max_mcd_db: f64,
    /// Minimum HNR for voiced phonemes in dB.
    /// Default: 10.0.
    pub min_voiced_hnr_db: f64,
    /// F0 range for voiced phonemes in Hz.
    /// Default: (60.0, 500.0) (Titze 1994).
    pub f0_range_hz: (f64, f64),
    /// Minimum energy ratio vs utterance mean (below = too quiet).
    /// Default: 0.05.
    pub min_energy_ratio: f64,
}

impl PhonemeVerifyConfig {
    /// Validate that all f64 fields are finite and within sensible ranges.
    pub fn validate(&self) -> Result<(), TtsVerifyError> {
        validate_finite_positive(self.min_duration_ms, "min_duration_ms")?;
        validate_finite_positive(self.max_duration_ms, "max_duration_ms")?;
        validate_finite_positive(self.max_mcd_db, "max_mcd_db")?;
        validate_finite(self.min_voiced_hnr_db, "min_voiced_hnr_db")?;
        validate_finite(self.f0_range_hz.0, "f0_range_hz.0")?;
        validate_finite(self.f0_range_hz.1, "f0_range_hz.1")?;
        validate_finite_positive(self.min_energy_ratio, "min_energy_ratio")?;
        if self.min_duration_ms >= self.max_duration_ms {
            return Err(TtsVerifyError::InvalidConfig(
                InvalidConfigKind::RangeInverted {
                    param: "duration_ms",
                },
            ));
        }
        if self.f0_range_hz.0 >= self.f0_range_hz.1 {
            return Err(TtsVerifyError::InvalidConfig(
                InvalidConfigKind::RangeInverted {
                    param: "f0_range_hz",
                },
            ));
        }
        Ok(())
    }
}

impl Default for PhonemeVerifyConfig {
    fn default() -> Self {
        Self {
            min_duration_ms: 20.0,
            max_duration_ms: 500.0,
            max_mcd_db: 8.0,
            min_voiced_hnr_db: 10.0,
            f0_range_hz: (60.0, 500.0),
            min_energy_ratio: 0.05,
        }
    }
}
