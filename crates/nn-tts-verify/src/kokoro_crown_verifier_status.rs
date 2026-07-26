// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Status file parsing and segment extraction for [`KokoroCrownVerifier`].
//!
//! Extracted from `kokoro_crown_verifier.rs` for the 500-line limit.
//! Part of #3874.

use serde::Deserialize;

use crate::kokoro_contracts::{
    all_contracts, bounds_within_contract, JunctionContract, VerifiedJunctionContract, J2_F0_LOWER,
    J2_F0_UPPER,
};
use crate::monotonicity::propagation_mode_is_sound_crown_family;

use super::{SegmentBounds, SegmentId, VerifierError};

// ============================================================================
// Status file serde models
// ============================================================================

/// Minimal serde model for the status file.
#[derive(Debug, Deserialize)]
pub(super) struct StatusFile {
    pub(super) kernels: std::collections::HashMap<String, StatusEntry>,
}

/// A single entry in the status file.
#[derive(Debug, Deserialize)]
pub(super) struct StatusEntry {
    #[allow(dead_code)]
    pub(super) status: Option<String>,
    pub(super) method: Option<String>,
    pub(super) soundness_mode: Option<String>,
    pub(super) proof_strength: Option<String>,
    pub(super) output_width: Option<f64>,
    pub(super) output_bounds: Option<StatusOutputBounds>,
    pub(super) input_bounds: Option<StatusInputBounds>,
    #[serde(default)]
    pub(super) stale: bool,
}

#[derive(Debug, Deserialize)]
pub(super) struct StatusOutputBounds {
    pub(super) lower: Option<f64>,
    pub(super) upper: Option<f64>,
    pub(super) tensor_lower: Option<Vec<f64>>,
    pub(super) tensor_upper: Option<Vec<f64>>,
    pub(super) shape: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
pub(super) struct StatusInputBounds {
    pub(super) input_shape: Option<Vec<usize>>,
    pub(super) input_range: Option<Vec<f64>>,
}

// ============================================================================
// Segment extraction
// ============================================================================

/// Extract the best bounds for a segment, preferring sound CROWN-family proofs.
///
/// Tries each status key prefix for the segment in priority order.
/// Within a prefix, entries are ranked:
/// sound > unsound, sound CROWN-family > IBP, tighter > wider.
pub(super) fn extract_best_bounds(
    status: &StatusFile,
    seg: SegmentId,
) -> Result<SegmentBounds, VerifierError> {
    let prefixes = seg.status_key_prefixes();

    // Collect non-stale entries matching any of the segment's prefixes.
    let mut candidates: Vec<(&String, &StatusEntry)> = Vec::new();
    for prefix in prefixes {
        let entries: Vec<(&String, &StatusEntry)> = status
            .kernels
            .iter()
            .filter(|(key, entry)| key.starts_with(prefix) && !entry.stale)
            .collect();
        candidates.extend(entries);
    }

    if candidates.is_empty() {
        let tried = prefixes
            .iter()
            .map(|p| format!("'{p}'"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(VerifierError::MissingSegment {
            segment: seg.name().to_string(),
            prefix: tried,
        });
    }

    // Rank candidates: prefer sound, then sound CROWN-family over IBP, then tightest width.
    candidates.sort_by(|a, b| {
        let a_sound = a.1.soundness_mode.as_deref() == Some("sound");
        let b_sound = b.1.soundness_mode.as_deref() == Some("sound");
        let a_crown =
            a.1.method
                .as_deref()
                .is_some_and(propagation_mode_is_sound_crown_family);
        let b_crown =
            b.1.method
                .as_deref()
                .is_some_and(propagation_mode_is_sound_crown_family);
        let a_width = a.1.output_width.unwrap_or(f64::MAX);
        let b_width = b.1.output_width.unwrap_or(f64::MAX);

        // Sound first, then sound CROWN-family, then tightest.
        b_sound.cmp(&a_sound).then(b_crown.cmp(&a_crown)).then(
            a_width
                .partial_cmp(&b_width)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });

    let (key, entry) = candidates[0];
    entry_to_segment_bounds(seg, key, entry)
}

/// Convert a status entry to `SegmentBounds`.
fn entry_to_segment_bounds(
    seg: SegmentId,
    key: &str,
    entry: &StatusEntry,
) -> Result<SegmentBounds, VerifierError> {
    let ob = entry
        .output_bounds
        .as_ref()
        .ok_or_else(|| VerifierError::CertificateValidation {
            reason: format!("entry '{key}' has no output_bounds"),
        })?;

    let shape = ob.shape.clone().unwrap_or_else(|| vec![1]);
    let elements: usize = shape.iter().product();

    // Use tensor_lower/tensor_upper if available, else broadcast scalar.
    let output_lower = ob.tensor_lower.clone().unwrap_or_else(|| {
        let scalar = ob.lower.unwrap_or(0.0);
        vec![scalar; elements]
    });
    let output_upper = ob.tensor_upper.clone().unwrap_or_else(|| {
        let scalar = ob.upper.unwrap_or(0.0);
        vec![scalar; elements]
    });

    // Input bounds from status entry.
    let (input_shape, input_lower, input_upper) = if let Some(ib) = &entry.input_bounds {
        let in_shape = ib.input_shape.clone().unwrap_or_else(|| vec![1]);
        let in_elements: usize = in_shape.iter().product();
        let range = ib.input_range.clone().unwrap_or_else(|| vec![-1.0, 1.0]);
        let lo = range.first().copied().unwrap_or(-1.0);
        let hi = range.last().copied().unwrap_or(1.0);
        (in_shape, vec![lo; in_elements], vec![hi; in_elements])
    } else {
        // Default: same shape as output, range [-1, 1].
        (shape.clone(), vec![-1.0; elements], vec![1.0; elements])
    };

    Ok(SegmentBounds {
        segment: seg,
        status_key: key.to_string(),
        method: entry.method.clone().unwrap_or_default(),
        is_sound: entry.soundness_mode.as_deref() == Some("sound"),
        proof_strength: entry.proof_strength.clone().unwrap_or_default(),
        output_shape: shape,
        output_lower,
        output_upper,
        output_width: entry.output_width.unwrap_or(0.0),
        input_lower,
        input_upper,
        input_shape,
    })
}

// ============================================================================
// Segment property checking
// ============================================================================

/// Check whether a segment's bounds prove its specific property.
pub(super) fn check_segment_property(bounds: &SegmentBounds) -> (bool, String) {
    match bounds.segment {
        SegmentId::BertEncoder => {
            let bounded = bounds.output_lower.iter().all(|v| v.is_finite())
                && bounds.output_upper.iter().all(|v| v.is_finite());
            let width = bounds.output_width;
            (
                bounded && bounds.is_sound,
                format!(
                    "Hidden state bounded: width={width:.4}, sound={}",
                    bounds.is_sound
                ),
            )
        }
        SegmentId::TextEncoder => {
            let bounded = bounds.output_lower.iter().all(|v| v.is_finite())
                && bounds.output_upper.iter().all(|v| v.is_finite());
            let width = bounds.output_width;
            (
                bounded && bounds.is_sound,
                format!(
                    "Encoded repr bounded: width={width:.4}, sound={}",
                    bounds.is_sound
                ),
            )
        }
        SegmentId::ProsodyPredictor => {
            let bounded = bounds.output_lower.iter().all(|v| v.is_finite())
                && bounds.output_upper.iter().all(|v| v.is_finite());
            let width = bounds.output_width;
            (
                bounded && bounds.is_sound,
                format!(
                    "Prosody bounded: width={width:.4}, sound={}",
                    bounds.is_sound
                ),
            )
        }
        SegmentId::F0EnergyPredictor => {
            let f0_in_range = bounds.proves_f0_range();
            (
                f0_in_range && bounds.is_sound,
                format!(
                    "F0 in [{}, {}] Hz: {}, sound={}",
                    J2_F0_LOWER, J2_F0_UPPER, f0_in_range, bounds.is_sound
                ),
            )
        }
        SegmentId::Generator => {
            let pcm_ok = bounds.proves_pcm_range();
            (
                pcm_ok && bounds.is_sound,
                format!("PCM in [-1.0, 1.0]: {}, sound={}", pcm_ok, bounds.is_sound),
            )
        }
    }
}

// ============================================================================
// Junction contract checking
// ============================================================================

/// Check all junction contracts against segment bounds.
pub(super) fn check_junction_contracts(
    segments: &[SegmentBounds],
) -> Vec<VerifiedJunctionContract> {
    let contracts = all_contracts();
    contracts
        .into_iter()
        .map(|contract| {
            let verified = check_contract_against_segments(&contract, segments);
            if verified {
                VerifiedJunctionContract {
                    contract,
                    composition_proof_lean4: None,
                    composition_theorem_name: None,
                    bounds_verified: true,
                }
            } else {
                VerifiedJunctionContract::new(contract)
            }
        })
        .collect()
}

/// Check a single junction contract against the segment bounds.
fn check_contract_against_segments(
    contract: &JunctionContract,
    segments: &[SegmentBounds],
) -> bool {
    // Map junction names to the relevant segment's output bounds.
    let (lower, upper) = match contract.name {
        "J2_F0" | "J2_ENERGY" => {
            if let Some(seg) = segments
                .iter()
                .find(|s| s.segment == SegmentId::F0EnergyPredictor)
            {
                (&seg.output_lower, &seg.output_upper)
            } else {
                return false;
            }
        }
        "J3_MAGNITUDE" | "J3B_PHASE" => {
            if let Some(seg) = segments.iter().find(|s| s.segment == SegmentId::Generator) {
                (&seg.output_lower, &seg.output_upper)
            } else {
                return false;
            }
        }
        "J4_BF16" => {
            if let Some(seg) = segments
                .iter()
                .find(|s| s.segment == SegmentId::ProsodyPredictor)
            {
                (&seg.output_lower, &seg.output_upper)
            } else {
                return false;
            }
        }
        "J5_AUDIO" => {
            if let Some(seg) = segments.iter().find(|s| s.segment == SegmentId::Generator) {
                (&seg.output_lower, &seg.output_upper)
            } else {
                return false;
            }
        }
        _ => return false,
    };

    bounds_within_contract(contract, lower, upper)
}
