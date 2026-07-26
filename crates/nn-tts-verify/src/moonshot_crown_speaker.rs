// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Property 4 (speaker consistency) verification via ECAPA-TDNN embedding bounds.
//!
//! Verifies that TTS output, when passed through the ECAPA-TDNN speaker encoder,
//! produces an embedding within distance ε of a reference speaker embedding.
//!
//! The worst-case L2 distance is computed from CROWN bounds on the embedding:
//!
//! ```text
//! d_worst² = Σ max(|ref_i - lower_i|, |ref_i - upper_i|)²
//! ```

use super::{MoonshotPropertyResult, PROPERTY_NAMES};
use crate::moonshot::VerificationLevel;

/// Speaker consistency evidence for Property 4 verification.
///
/// Captures CROWN bounds through the ECAPA-TDNN speaker encoder, producing
/// bounded embedding distance from a reference speaker embedding.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpeakerConsistencyEvidence {
    /// ECAPA-TDNN embedding dimension (typically 192).
    pub embed_dim: usize,
    /// CROWN-proven lower bounds on the speaker embedding (per element).
    pub embedding_lower: Vec<f64>,
    /// CROWN-proven upper bounds on the speaker embedding (per element).
    pub embedding_upper: Vec<f64>,
    /// Reference speaker embedding (per element).
    pub reference_embedding: Vec<f64>,
    /// Maximum distance threshold (embedding distance must be < ε).
    pub distance_threshold: f64,
    /// Whether the CROWN bounds are sound (not IBP fallback).
    pub is_sound: bool,
}

impl SpeakerConsistencyEvidence {
    /// Create new speaker consistency evidence.
    pub fn new(
        embed_dim: usize,
        embedding_lower: Vec<f64>,
        embedding_upper: Vec<f64>,
        reference_embedding: Vec<f64>,
        distance_threshold: f64,
        is_sound: bool,
    ) -> Self {
        Self {
            embed_dim,
            embedding_lower,
            embedding_upper,
            reference_embedding,
            distance_threshold,
            is_sound,
        }
    }
}

/// Check Property 4 (speaker consistency) against ECAPA-TDNN embedding bounds.
///
/// Speaker consistency requires that the TTS output, when passed through the
/// ECAPA-TDNN speaker encoder, produces an embedding within distance ε of the
/// reference speaker embedding.
///
/// Given CROWN bounds `[lower_i, upper_i]` on each embedding dimension and a
/// reference embedding `ref_i`, the worst-case L2 distance is:
///
/// ```text
/// d_worst² = Σ max(|ref_i - lower_i|, |ref_i - upper_i|)²
/// ```
///
/// The check: `d_worst < distance_threshold`.
pub fn check_speaker_consistency(evidence: &SpeakerConsistencyEvidence) -> MoonshotPropertyResult {
    let dim = evidence.embed_dim;

    if evidence.embedding_lower.len() != dim
        || evidence.embedding_upper.len() != dim
        || evidence.reference_embedding.len() != dim
    {
        return MoonshotPropertyResult {
            property_index: 3,
            property_name: PROPERTY_NAMES[3],
            proven: false,
            level: VerificationLevel::None,
            bound_value: f64::INFINITY,
            threshold: evidence.distance_threshold,
            is_sound: false,
            explanation: "dimension mismatch in speaker consistency evidence".to_string(),
        };
    }

    // Compute worst-case L2 distance: for each dimension, take the maximum
    // absolute difference from the reference to either bound endpoint.
    //
    // SOUNDNESS: f64::max(NaN, x) returns x (IEEE 754-2008 maxNum semantics),
    // silently discarding NaN. We must check finiteness of each bound element
    // before the max() call to prevent NaN-contaminated bounds from producing
    // a false "proven" result. See P1-234 audit.
    let d_worst_sq: f64 = (0..dim)
        .map(|i| {
            let lo = evidence.embedding_lower[i];
            let hi = evidence.embedding_upper[i];
            let rf = evidence.reference_embedding[i];
            if !lo.is_finite() || !hi.is_finite() || !rf.is_finite() {
                return f64::INFINITY;
            }
            let d_lo = (rf - lo).abs();
            let d_hi = (rf - hi).abs();
            let d_max = d_lo.max(d_hi);
            d_max * d_max
        })
        .sum();
    let d_worst = d_worst_sq.sqrt();

    let finite = d_worst.is_finite();
    let within_threshold = d_worst < evidence.distance_threshold;
    let proven = finite && within_threshold;

    let level = if proven && evidence.is_sound {
        VerificationLevel::CrownProven
    } else if proven {
        VerificationLevel::CrownPartial
    } else {
        VerificationLevel::Empirical
    };

    MoonshotPropertyResult {
        property_index: 3,
        property_name: PROPERTY_NAMES[3],
        proven,
        level,
        bound_value: d_worst,
        threshold: evidence.distance_threshold,
        is_sound: evidence.is_sound,
        explanation: format!(
            "worst-case L2 distance = {d_worst:.6}, threshold = {:.6}: {}",
            evidence.distance_threshold,
            if proven { "PROVEN" } else { "NOT PROVEN" }
        ),
    }
}
