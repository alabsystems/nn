// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pronunciation defect detection from per-phoneme quality results.
//!
//! Given per-phoneme quality results from [`verify_phonemes`](crate::phoneme_verify::verify_phonemes),
//! classifies each failure into a pronunciation defect type. This enables
//! targeted feedback: "phoneme /θ/ was devoiced" rather than "quality check
//! failed."

use crate::phoneme::{PhonemeResult, PhonemeVerifyConfig};

/// Types of pronunciation defects detectable from acoustic analysis.
///
/// Each variant carries the phoneme label and the measured value that
/// triggered the detection.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PronunciationDefect {
    /// Phoneme too short — likely deleted or under-articulated.
    /// Duration below `min_duration_ms`.
    Deletion { label: String, duration_ms: f64 },

    /// Phoneme too long — likely stalled or repeated.
    /// Duration above `max_duration_ms`.
    Insertion { label: String, duration_ms: f64 },

    /// Voiced phoneme has low HNR — likely devoiced (e.g., /ð/ -> /θ/).
    Devoicing { label: String, hnr_db: f64 },

    /// High MCD vs reference — likely substituted for a different phoneme.
    Substitution { label: String, mcd_db: f64 },

    /// Energy too low relative to context — likely under-articulated.
    WeakArticulation { label: String, energy_ratio: f64 },
}

/// Detect pronunciation defects from per-phoneme verification results.
///
/// Examines each `PhonemeResult` that failed and classifies the failure
/// based on which metric failed and by how much.
///
/// Returns an empty vector if all phonemes passed.
pub fn detect_defects(
    results: &[PhonemeResult],
    config: &PhonemeVerifyConfig,
) -> Vec<PronunciationDefect> {
    let mut defects = Vec::new();

    for result in results {
        if result.passed {
            continue;
        }

        // Check duration defects first (highest diagnostic value).
        if result.duration_ms < config.min_duration_ms {
            defects.push(PronunciationDefect::Deletion {
                label: result.label.clone(),
                duration_ms: result.duration_ms,
            });
            continue;
        }
        if result.duration_ms > config.max_duration_ms {
            defects.push(PronunciationDefect::Insertion {
                label: result.label.clone(),
                duration_ms: result.duration_ms,
            });
            continue;
        }

        // Check individual metric failures.
        for metric in &result.metrics {
            if metric.passed {
                continue;
            }

            match metric.name {
                "hnr" => {
                    defects.push(PronunciationDefect::Devoicing {
                        label: result.label.clone(),
                        hnr_db: metric.value,
                    });
                }
                "mcd" => {
                    defects.push(PronunciationDefect::Substitution {
                        label: result.label.clone(),
                        mcd_db: metric.value,
                    });
                }
                "energy_ratio" => {
                    defects.push(PronunciationDefect::WeakArticulation {
                        label: result.label.clone(),
                        energy_ratio: metric.value,
                    });
                }
                _ => {
                    // Other metric failures (e.g., f0_range) don't map
                    // directly to a pronunciation defect category.
                }
            }
        }
    }

    defects
}
