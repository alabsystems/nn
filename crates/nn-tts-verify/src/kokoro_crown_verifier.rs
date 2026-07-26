// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro end-to-end CROWN certificate verifier.
//!
//! [`KokoroCrownVerifier`] is the unified API that dvoice calls to obtain
//! a CROWN proof certificate for the Kokoro TTS pipeline. It:
//!
//! 1. Loads per-segment verification results from `nn_verify_status_kokoro.json`
//! 2. Maps the 5 Kokoro pipeline segments to their proven bounds
//! 3. Composes segments via [`verify_pipeline`] to prove end-to-end properties
//! 4. Checks junction contracts (J2-J5) at zone crossings
//! 5. Produces a [`KokoroCrownCertificate`] with:
//!    - PCM output in [-1.0, 1.0] proof (P2: non-clipping)
//!    - F0 bounds (from junction contract J2)
//!    - Soundness provenance per segment
//! 6. Supports save/load for persistent caching
//!
//! # The 5 Kokoro Segments
//!
//! | Segment | Status Key(s) | Property to Prove |
//! |---------|---------------|-------------------|
//! | 0: BertEncoder | `kokoro_production_bert_encoder*` | Hidden state bounded |
//! | 1: TextEncoder | `kokoro_production_text_encoder*` | Encoded repr bounded |
//! | 2: ProsodyPredictor | `kokoro_production_prosody_predictor*` | Durations bounded |
//! | 3: F0+EnergyPredictor | `kokoro_production_f0_predictor*` | F0 in [50, 800] Hz |
//! | 4: Generator (decoder) | `kokoro_production_generator`, `*_istft` | PCM in [-1, 1] |
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_tts_verify::kokoro_crown_verifier::KokoroCrownVerifier;
//! use std::path::Path;
//!
//! // Build from the verification status file.
//! let verifier = KokoroCrownVerifier::from_status_file(
//!     Path::new("nn_verify_status_kokoro.json"),
//! )?;
//!
//! // Verify all segments and produce a certificate.
//! let cert = verifier.verify_all()?;
//! assert!(cert.pipeline_is_sound);
//!
//! // Save certificate for deployment.
//! verifier.save(Path::new("kokoro_certificate.json"))?;
//!
//! // Load cached certificate.
//! let loaded = KokoroCrownVerifier::load(Path::new("kokoro_certificate.json"))?;
//! ```
//!
//! Part of #3874.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::TtsVerifyError;
use crate::kokoro_contracts::{
    VerifiedJunctionContract, J2_F0_LOWER, J2_F0_UPPER, J5_AUDIO_LOWER, J5_AUDIO_UPPER,
};
use crate::kokoro_crown_certificate::KokoroCrownCertificate;
use crate::moonshot_crown::{verify_properties_from_pipeline, MoonshotCrownBundle};
use crate::pipeline::{verify_pipeline, PipelineCertificate, VerifiedStage};

#[path = "kokoro_crown_verifier_status.rs"]
mod status;
use status::{check_junction_contracts, check_segment_property, extract_best_bounds, StatusFile};

/// Identifies one of the 5 Kokoro pipeline segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SegmentId {
    /// Segment 0: PlBert + BertEncoder — input tokens to hidden states.
    BertEncoder,
    /// Segment 1: TextEncoder — hidden states to encoded representation.
    TextEncoder,
    /// Segment 2: ProsodyPredictor — encoded repr to duration/prosody.
    ProsodyPredictor,
    /// Segment 3: F0 + Energy Predictor — prosody to F0 and energy features.
    F0EnergyPredictor,
    /// Segment 4: Generator (decoder + iSTFT) — features to PCM audio.
    Generator,
}

impl SegmentId {
    /// All 5 segments in pipeline order.
    pub const ALL: [Self; 5] = [
        Self::BertEncoder,
        Self::TextEncoder,
        Self::ProsodyPredictor,
        Self::F0EnergyPredictor,
        Self::Generator,
    ];

    /// Human-readable name for the segment.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::BertEncoder => "BertEncoder",
            Self::TextEncoder => "TextEncoder",
            Self::ProsodyPredictor => "ProsodyPredictor",
            Self::F0EnergyPredictor => "F0EnergyPredictor",
            Self::Generator => "Generator",
        }
    }

    /// Status file key prefixes for this segment, in priority order.
    ///
    /// The verifier tries each prefix in order, using the first that has
    /// non-stale entries. Production prefixes are tried first; fallback
    /// prefixes cover cases where production entries are marked stale.
    #[must_use]
    pub fn status_key_prefixes(self) -> &'static [&'static str] {
        match self {
            Self::BertEncoder => &[
                "kokoro_production_bert_encoder",
                "kokoro_encoder_style_chain",
            ],
            Self::TextEncoder => &["kokoro_production_text_encoder", "kokoro_text_encoder"],
            Self::ProsodyPredictor => &[
                "kokoro_production_prosody_predictor",
                "kokoro_prosody",
                "kokoro_duration_branch",
            ],
            Self::F0EnergyPredictor => {
                &["kokoro_production_f0_predictor", "kokoro_f0_adain_resblk"]
            }
            Self::Generator => &[
                "kokoro_production_generator",
                "kokoro_production_istft",
                "kokoro_generator",
                "kokoro_vocoder",
            ],
        }
    }

    /// The property this segment must prove, as described in issue #3874.
    #[must_use]
    pub fn property_description(self) -> &'static str {
        match self {
            Self::BertEncoder => "Hidden state bounded (norm_inf <= B0)",
            Self::TextEncoder => "Encoded representation bounded",
            Self::ProsodyPredictor => "Durations non-negative, bounded",
            Self::F0EnergyPredictor => "F0 in [50, 800] Hz, energy >= 0",
            Self::Generator => "PCM in [-1.0, 1.0] (no clipping)",
        }
    }
}

/// Verified bounds for a single pipeline segment, extracted from the status file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentBounds {
    /// Which segment this covers.
    pub segment: SegmentId,
    /// Status file entry key used.
    pub status_key: String,
    /// Verification method (e.g., "IBP", "CROWN", "AlphaCrown", "BetaCrown").
    pub method: String,
    /// Whether the verification is sound.
    pub is_sound: bool,
    /// Proof strength classification.
    pub proof_strength: String,
    /// Output shape from the status entry.
    pub output_shape: Vec<usize>,
    /// Per-element output lower bounds.
    pub output_lower: Vec<f64>,
    /// Per-element output upper bounds.
    pub output_upper: Vec<f64>,
    /// Scalar output width (max - min across all elements).
    pub output_width: f64,
    /// Input lower bounds (uniform across elements).
    pub input_lower: Vec<f64>,
    /// Input upper bounds (uniform across elements).
    pub input_upper: Vec<f64>,
    /// Input shape.
    pub input_shape: Vec<usize>,
}

impl SegmentBounds {
    /// Whether the output bounds prove PCM in [-1, 1].
    #[must_use]
    pub fn proves_pcm_range(&self) -> bool {
        self.output_lower.iter().all(|&v| v >= J5_AUDIO_LOWER)
            && self.output_upper.iter().all(|&v| v <= J5_AUDIO_UPPER)
    }

    /// Whether the output bounds prove F0 in [50, 800] Hz.
    #[must_use]
    pub fn proves_f0_range(&self) -> bool {
        self.output_lower.iter().all(|&v| v >= J2_F0_LOWER)
            && self.output_upper.iter().all(|&v| v <= J2_F0_UPPER)
    }

    /// Convert to a `VerifiedStage` for pipeline composition.
    #[must_use]
    pub fn to_verified_stage(&self) -> VerifiedStage {
        VerifiedStage::new(
            self.segment.name(),
            self.input_shape.clone(),
            self.output_shape.clone(),
            self.input_lower.clone(),
            self.input_upper.clone(),
            self.output_lower.clone(),
            self.output_upper.clone(),
            &self.method,
            self.is_sound,
        )
    }
}

/// Result of verifying a single segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentVerifyResult {
    /// Segment identifier.
    pub segment: SegmentId,
    /// The verified bounds.
    pub bounds: SegmentBounds,
    /// Whether the segment-specific property is proven.
    pub property_proven: bool,
    /// Human-readable explanation.
    pub explanation: String,
}

/// Error type for `KokoroCrownVerifier` operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VerifierError {
    /// Status file could not be read.
    #[error("failed to read status file: {0}")]
    Io(#[from] std::io::Error),

    /// Status file JSON is malformed.
    #[error("failed to parse status file: {0}")]
    Json(#[from] serde_json::Error),

    /// A required segment is missing from the status file.
    #[error("segment {segment} not found in status file (looked for key prefix '{prefix}')")]
    MissingSegment {
        /// Segment that was not found.
        segment: String,
        /// Status key prefix that was searched.
        prefix: String,
    },

    /// Pipeline composition failed.
    #[error("pipeline verification failed: {0}")]
    Pipeline(#[from] TtsVerifyError),

    /// Certificate validation failed.
    #[error("certificate validation failed: {reason}")]
    CertificateValidation {
        /// Description of the validation failure.
        reason: String,
    },
}

/// Unified Kokoro CROWN certificate verifier.
///
/// Loads verification results for the 5 Kokoro pipeline segments from the
/// status file, composes them into an end-to-end pipeline certificate, and
/// checks junction contracts at zone crossings.
///
/// This is the entry point for dvoice to obtain a CROWN certificate.
#[derive(Debug, Clone)]
pub struct KokoroCrownVerifier {
    /// Per-segment verified bounds, in pipeline order.
    segments: Vec<SegmentBounds>,
    /// The model name for the certificate.
    model_name: String,
}

impl KokoroCrownVerifier {
    /// Build a verifier from a verification status file.
    ///
    /// Reads `nn_verify_status_kokoro.json` and extracts the best verified
    /// bounds for each of the 5 Kokoro segments, preferring sound
    /// CROWN-family methods over IBP when both are present.
    ///
    /// # Errors
    ///
    /// Returns [`VerifierError::Io`] if the file cannot be read,
    /// [`VerifierError::Json`] if parsing fails, or
    /// [`VerifierError::MissingSegment`] if a required segment has no entry.
    pub fn from_status_file(path: &Path) -> Result<Self, VerifierError> {
        Self::from_status_file_with_name(path, "dvoice-kokoro-v1")
    }

    /// Build a verifier with a custom model name.
    pub fn from_status_file_with_name(
        path: &Path,
        model_name: &str,
    ) -> Result<Self, VerifierError> {
        let content = std::fs::read_to_string(path)?;
        let status: StatusFile = serde_json::from_str(&content)?;

        let mut segments = Vec::with_capacity(5);
        for seg_id in SegmentId::ALL {
            let bounds = extract_best_bounds(&status, seg_id)?;
            segments.push(bounds);
        }

        Ok(Self {
            segments,
            model_name: model_name.to_string(),
        })
    }

    /// Build a verifier from pre-computed segment bounds.
    ///
    /// Used when bounds are already available (e.g., from a live CROWN run)
    /// rather than loaded from the status file.
    #[must_use]
    pub fn from_segments(segments: Vec<SegmentBounds>, model_name: &str) -> Self {
        Self {
            segments,
            model_name: model_name.to_string(),
        }
    }

    /// Number of segments loaded.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Get the bounds for a specific segment.
    #[must_use]
    pub fn segment_bounds(&self, seg: SegmentId) -> Option<&SegmentBounds> {
        self.segments.iter().find(|s| s.segment == seg)
    }

    /// Whether all segments have sound verification.
    #[must_use]
    pub fn all_sound(&self) -> bool {
        self.segments.iter().all(|s| s.is_sound)
    }

    /// Verify a single segment and return its result.
    #[must_use]
    pub fn verify_segment(&self, seg: SegmentId) -> Option<SegmentVerifyResult> {
        let bounds = self.segment_bounds(seg)?;
        let (property_proven, explanation) = check_segment_property(bounds);
        Some(SegmentVerifyResult {
            segment: seg,
            bounds: bounds.clone(),
            property_proven,
            explanation,
        })
    }

    /// Verify all segments and produce a [`KokoroCrownCertificate`].
    ///
    /// Pipeline:
    /// 1. Convert each segment to a `VerifiedStage`
    /// 2. Compose via `verify_pipeline()` for junction compatibility
    /// 3. Check moonshot properties (P1-P3, P6) against the pipeline
    /// 4. Check junction contracts (J2-J5)
    /// 5. Package into a `KokoroCrownCertificate`
    ///
    /// # Errors
    ///
    /// Returns [`VerifierError::Pipeline`] if pipeline composition fails.
    pub fn verify_all(&self) -> Result<VerifyAllResult, VerifierError> {
        // 1. Build pipeline stages.
        let stages: Vec<VerifiedStage> = self
            .segments
            .iter()
            .map(SegmentBounds::to_verified_stage)
            .collect();

        // 2. Compose pipeline (requires >= 2 stages).
        let pipeline_cert = if stages.len() >= 2 {
            verify_pipeline(&stages)?
        } else {
            // Single-stage pipeline: construct a degenerate certificate.
            let stage = &stages[0];
            PipelineCertificate {
                stages: stages.clone(),
                junctions: vec![],
                e2e_input_lower: stage.input_lower.clone(),
                e2e_input_upper: stage.input_upper.clone(),
                e2e_output_lower: stage.output_lower.clone(),
                e2e_output_upper: stage.output_upper.clone(),
                is_valid: true,
                is_sound: stage.is_sound,
            }
        };

        // 3. Check moonshot properties against the pipeline.
        let dim = self
            .segments
            .last()
            .map(|s| {
                let elements: usize = s.output_shape.iter().product();
                elements
            })
            .unwrap_or(256);
        let crown_bundle = verify_properties_from_pipeline(&pipeline_cert, dim);

        // 4. Check junction contracts.
        let verified_junctions = check_junction_contracts(&self.segments);

        // 5. Per-segment verification results.
        let segment_results: Vec<SegmentVerifyResult> = SegmentId::ALL
            .iter()
            .filter_map(|&seg| self.verify_segment(seg))
            .collect();

        // 6. Build the certificate.
        let certificate = KokoroCrownCertificate::from_components(
            &self.model_name,
            &crown_bundle,
            &verified_junctions,
        );

        // 7. Validate structural integrity.
        if let Err(e) = certificate.validate() {
            return Err(VerifierError::CertificateValidation {
                reason: format!("{e}"),
            });
        }

        Ok(VerifyAllResult {
            certificate,
            pipeline_cert,
            crown_bundle,
            segment_results,
            verified_junctions,
        })
    }

    /// Save a certificate to a JSON file.
    ///
    /// # Errors
    ///
    /// Returns [`VerifierError::Io`] on write failure or
    /// [`VerifierError::Json`] on serialization failure.
    pub fn save(result: &VerifyAllResult, path: &Path) -> Result<(), VerifierError> {
        let json =
            result
                .certificate
                .to_json()
                .map_err(|e| VerifierError::CertificateValidation {
                    reason: format!("serialization failed: {e}"),
                })?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a certificate from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns [`VerifierError::Io`] on read failure,
    /// [`VerifierError::Json`] on parse failure, or
    /// [`VerifierError::CertificateValidation`] if the loaded certificate
    /// fails structural validation.
    pub fn load(path: &Path) -> Result<KokoroCrownCertificate, VerifierError> {
        let json = std::fs::read_to_string(path)?;
        let cert: KokoroCrownCertificate =
            serde_json::from_str(&json)?;
        cert.validate()
            .map_err(|e| VerifierError::CertificateValidation {
                reason: format!("{e}"),
            })?;
        Ok(cert)
    }

    /// Check whether the last stage proves PCM output in [-1, 1].
    ///
    /// This is an O(1) check suitable for runtime use. Returns `true` if the
    /// Generator segment's verified bounds satisfy the PCM range contract.
    #[must_use]
    pub fn input_in_pcm_domain(&self) -> bool {
        self.segment_bounds(SegmentId::Generator)
            .map_or(false, SegmentBounds::proves_pcm_range)
    }
}

/// Full result of `verify_all()`.
#[derive(Debug, Clone)]
pub struct VerifyAllResult {
    /// The deployable certificate.
    pub certificate: KokoroCrownCertificate,
    /// Pipeline composition certificate with junction compatibility.
    pub pipeline_cert: PipelineCertificate,
    /// Moonshot property results (P1-P3, P6).
    pub crown_bundle: MoonshotCrownBundle,
    /// Per-segment verification results.
    pub segment_results: Vec<SegmentVerifyResult>,
    /// Junction contract verification (J2-J5).
    pub verified_junctions: Vec<VerifiedJunctionContract>,
}

impl VerifyAllResult {
    /// Whether the certificate proves PCM output in [-1, 1] for the token domain.
    #[must_use]
    pub fn proves_pcm_range(&self) -> bool {
        self.certificate
            .properties
            .iter()
            .any(|p| p.property_index == 1 && p.proven)
    }

    /// Whether the pipeline is fully sound (no heuristic/vacuous stages).
    #[must_use]
    pub fn is_sound(&self) -> bool {
        self.certificate.pipeline_is_sound
    }

    /// Whether all junction contracts are satisfied.
    #[must_use]
    pub fn all_junctions_verified(&self) -> bool {
        self.certificate.all_junctions_verified
    }

    /// Human-readable summary of the verification result.
    #[must_use]
    pub fn summary(&self) -> String {
        self.certificate.summary()
    }
}

#[cfg(test)]
#[path = "kokoro_crown_verifier_tests.rs"]
mod tests;
