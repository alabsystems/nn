// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Certificate format for verified weight edits.
//!
//! An [`EditCertificate`] captures the full provenance of a weight edit:
//! which weights were modified, how they changed, and what bounds the
//! verifier proved on the edited model. This enables offline auditing
//! of weight surgery operations (ROME rank-1 updates, LoRA overlays,
//! gradient-based steering) without re-running the verifier.
//!
//! Always available (no `NY` feature gate required) — this is
//! a data format, not a verification engine.
//!
//! # Example
//!
//! ```rust
//! use nn_verify::edit_certificate::{EditCertificate, EditedWeight, EditType};
//! use nn_verify::PropMethod;
//!
//! let cert = EditCertificate::new(
//!     "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".into(),
//!     "f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5".into(),
//!     PropMethod::Ibp,
//! )
//! .with_edited_weight(EditedWeight {
//!     layer_name: "transformer.h.4.mlp.c_proj".to_string(),
//!     edit_type: EditType::Rank1Update,
//!     delta_norm: 0.042,
//!     delta_rank: Some(1),
//! });
//!
//! assert_eq!(cert.edited_weights.len(), 1);
//! ```

use serde::{Deserialize, Serialize};

use crate::certificate::now_iso8601;
use crate::certificate_types::{validate_sha256_hex, KaniProofRecord};
use crate::error::VerifyError;
use crate::soundness_compat::{default_soundness_mode, VerificationSoundnessMode};
use crate::status::OutputBoundsRecord;
use crate::verify_types::PropMethod;

/// Certificate proving a weight edit is safe and achieves its target.
///
/// Records the full provenance chain: original model hash, edited model hash,
/// which weight matrices were modified (with Frobenius norm and rank of the
/// delta), and what bounds the verifier proved on the edited model.
///
/// Two categories of bounds:
/// - **Target bounds**: what the edit should achieve (e.g., output for the
///   target prompt is within a desired range).
/// - **Preservation bounds**: what the edit should NOT break (e.g., outputs
///   for unrelated prompts remain within the original model's bounds).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EditCertificate {
    /// SHA-256 hex digest of the original model weights (before edit).
    pub original_model_hash: String,
    /// SHA-256 hex digest of the edited model weights (after edit).
    pub edited_model_hash: String,
    /// Which weight matrices were modified and how.
    pub edited_weights: Vec<EditedWeight>,
    /// Target behavior bounds (what the edit should achieve).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub target_bounds: Option<OutputBoundsRecord>,
    /// Preservation bounds (what the edit should NOT break).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub preservation_bounds: Option<OutputBoundsRecord>,
    /// Kani status for numerical safety of the edit operation.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub kani_status: Option<KaniProofRecord>,
    /// ISO 8601 timestamp of verification.
    pub verified_at: String,
    /// Propagation method used (IBP or CROWN).
    pub prop_method: PropMethod,
    /// Soundness classification: `Sound` if no heuristics were used.
    #[serde(default = "default_soundness_mode")]
    pub soundness_mode: VerificationSoundnessMode,

    // --- Integrity fields (same as ProofCertificate v4) ---
    /// SHA-256 hex digest of the canonical certificate content (all fields
    /// except `content_hash` and `hmac_signature`).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub content_hash: Option<String>,
    /// HMAC-SHA256 hex digest over the `content_hash`, keyed with a shared
    /// secret. Prevents forgery of edit provenance.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub hmac_signature: Option<String>,
}

/// Description of a single modified weight matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EditedWeight {
    /// Fully-qualified layer name (e.g., `"transformer.h.4.mlp.c_proj"`).
    pub layer_name: String,
    /// Type of edit applied to this weight matrix.
    pub edit_type: EditType,
    /// Frobenius norm of the weight delta: `||W_new - W_old||_F`.
    pub delta_norm: f32,
    /// Rank of the weight delta, if applicable (e.g., 1 for ROME rank-1 updates).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub delta_rank: Option<usize>,
}

/// Type of weight edit applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EditType {
    /// ROME-style rank-1 update: `W' = W + u * v^T`.
    Rank1Update,
    /// LoRA adapter overlay: `W' = W + alpha * B * A`.
    LoraOverlay,
    /// Full weight replacement (arbitrary delta).
    DirectWrite,
    /// Gradient-based steering: `W' = W - lr * grad`.
    GradientStep,
}

impl EditCertificate {
    /// Create a new edit certificate with the given model hashes and method.
    ///
    /// Use builder methods (`with_*`) to attach bounds and Kani status.
    #[must_use]
    pub fn new(
        original_model_hash: String,
        edited_model_hash: String,
        prop_method: PropMethod,
    ) -> Self {
        Self {
            original_model_hash,
            edited_model_hash,
            edited_weights: Vec::new(),
            target_bounds: None,
            preservation_bounds: None,
            kani_status: None,
            verified_at: now_iso8601(),
            prop_method,
            soundness_mode: VerificationSoundnessMode::Heuristic,
            content_hash: None,
            hmac_signature: None,
        }
    }

    /// Add an edited weight entry.
    #[must_use]
    pub fn with_edited_weight(mut self, weight: EditedWeight) -> Self {
        self.edited_weights.push(weight);
        self
    }

    /// Set all edited weight entries at once.
    #[must_use]
    pub fn with_edited_weights(mut self, weights: Vec<EditedWeight>) -> Self {
        self.edited_weights = weights;
        self
    }

    /// Attach target behavior bounds.
    #[must_use]
    pub fn with_target_bounds(mut self, bounds: OutputBoundsRecord) -> Self {
        self.target_bounds = Some(bounds);
        self
    }

    /// Attach preservation bounds.
    #[must_use]
    pub fn with_preservation_bounds(mut self, bounds: OutputBoundsRecord) -> Self {
        self.preservation_bounds = Some(bounds);
        self
    }

    /// Attach Kani proof status.
    #[must_use]
    pub fn with_kani_status(mut self, status: KaniProofRecord) -> Self {
        self.kani_status = Some(status);
        self
    }

    /// Set the soundness mode.
    #[must_use]
    pub fn with_soundness_mode(mut self, mode: VerificationSoundnessMode) -> Self {
        self.soundness_mode = mode;
        self
    }

    /// Validate structural self-consistency of the certificate.
    ///
    /// Checks:
    /// - Model hashes are valid SHA-256 hex digests (64 chars).
    /// - At least one edited weight is recorded.
    /// - All delta norms are finite and non-negative.
    /// - Bounds (if present) have `lower <= upper`.
    pub fn validate(&self) -> Result<(), VerifyError> {
        validate_sha256_hex(&self.original_model_hash).map_err(|()| {
            VerifyError::InvalidInput("original_model_hash is not valid SHA-256 hex".into())
        })?;
        validate_sha256_hex(&self.edited_model_hash).map_err(|()| {
            VerifyError::InvalidInput("edited_model_hash is not valid SHA-256 hex".into())
        })?;
        if self.edited_weights.is_empty() {
            return Err(VerifyError::InvalidInput("edited_weights is empty".into()));
        }
        for w in &self.edited_weights {
            if !w.delta_norm.is_finite() || w.delta_norm < 0.0 {
                return Err(VerifyError::InvalidInput(format!(
                    "delta_norm for '{}' is not finite non-negative: {}",
                    w.layer_name, w.delta_norm
                )));
            }
        }
        if let Some(ref b) = self.target_bounds {
            // IEEE 754: NaN > NaN returns false, bypassing inverted-bounds check.
            // Guard with is_finite() first (same pattern as ProofCertificate::validate).
            if !b.lower.is_finite() || !b.upper.is_finite() {
                return Err(VerifyError::InvalidInput(format!(
                    "target_bounds has non-finite values: lower={}, upper={}",
                    b.lower, b.upper
                )));
            }
            if b.lower > b.upper {
                return Err(VerifyError::InvalidInput(
                    "target_bounds: lower > upper".into(),
                ));
            }
        }
        if let Some(ref b) = self.preservation_bounds {
            if !b.lower.is_finite() || !b.upper.is_finite() {
                return Err(VerifyError::InvalidInput(format!(
                    "preservation_bounds has non-finite values: lower={}, upper={}",
                    b.lower, b.upper
                )));
            }
            if b.lower > b.upper {
                return Err(VerifyError::InvalidInput(
                    "preservation_bounds: lower > upper".into(),
                ));
            }
        }
        Ok(())
    }

    /// Serialize to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, VerifyError> {
        serde_json::to_string_pretty(self).map_err(VerifyError::from)
    }
}

impl std::fmt::Display for EditType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rank1Update => write!(f, "rank1_update"),
            Self::LoraOverlay => write!(f, "lora_overlay"),
            Self::DirectWrite => write!(f, "direct_write"),
            Self::GradientStep => write!(f, "gradient_step"),
        }
    }
}

#[cfg(test)]
#[path = "edit_certificate_tests.rs"]
mod tests;
