// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Proof certificate format for verified ML kernels.
//!
//! A `ProofCertificate` captures the evidence produced by NY verification
//! (input specification, output bounds, verification method, soundness mode) in a
//! format that enables offline auditing without re-running the verifier.
//!
//! ## Status JSON vs. Proof Certificate
//!
//! `nn_verify_status.json` records *that* verification passed. A `ProofCertificate`
//! records *why* — the intermediate bounds, input specification, and soundness
//! provenance that an auditor needs to check the result independently.
//!
//! ## Version History
//!
//! - **v1:** Input spec, output bounds, method, soundness mode, SMT outcome.
//! - **v2:** Per-layer bound traces, Kani proof records, weight/source SHA-256
//!   fingerprints, verifier version. All v2 fields are `Option` so v1 JSON
//!   deserializes into v2 structs without error.
//! - **v3:** SMT proof artifacts (Alethe proof text, proof verdict). Enables
//!   machine-checkable proofs: an auditor can validate the SMT UNSAT proof
//!   independently of the solver. All v3 fields are `Option` for backward compat.
//! - **v4:** HMAC-SHA256 integrity (`content_hash`, `hmac_signature`). Enables
//!   tamper detection: content hash covers all fields except itself and the
//!   signature. All v4 fields are `Option` for backward compat.
//! - **v5:** Precision model metadata (`precision_model`). Records whether
//!   verification modeled F16/BF16 precision loss at dtype cast points (#3023).
//!   `Option` for backward compat; `None` treated as `F32Only`.
//! - **v6:** Constructive proof certificate (`constructive_proof`). Contains
//!   machine-checkable proof data from NY's constructive proof pipeline
//!   (#4315). `Option` for backward compat; `None` means no constructive proof.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use std::path::Path;
//! use nn_verify::certificate::{certificate_from_pipeline, CertificateBundle};
//!
//! // After verification (given gamma_crown result and input metadata):
//! // let cert = certificate_from_pipeline(&gamma_crown, &variable_inputs, &constants, None);
//! // let bundle = CertificateBundle::new("nn_model").with_certificate(cert);
//! // bundle.save(Path::new("model.proof.json"))?;
//! ```

use serde::{Deserialize, Serialize};

use crate::soundness_compat::VerificationSoundnessMode;

use crate::certificate_types::{validate_sha256_hex, ConstructiveProofData};
use crate::error::VerifyError;
use crate::status::{InputBoundsRecord, OutputBoundsRecord, SmtProofVerdict};
use crate::verify_types::{KernelVerification, OutputTensorBounds, PropMethod};

#[path = "certificate_bundle.rs"]
pub mod bundle;
pub use bundle::{CertificateBundle, CertificateError};

// Re-export v2 types and fingerprinting functions so callers use `certificate::` path.
pub use crate::certificate_types::{
    compute_bytes_hash, compute_file_hash, KaniOutcome, KaniProofRecord, LayerBoundRecord,
    PrecisionModel,
};

/// Proof certificate for a single verified kernel.
///
/// Contains the full verification evidence: input specification, output bounds,
/// method, and soundness provenance. An auditor can check:
/// 1. Input bounds match the deployment specification
/// 2. Output bounds satisfy the required safety property
/// 3. Soundness mode is `Sound` (no heuristic approximations)
/// 4. The verification method (IBP/CROWN) is appropriate for the precision needed
/// 5. (v2) Per-layer bound traces enable independent re-derivation
/// 6. (v2) Weight/source hashes bind the certificate to a specific binary
/// 7. (v3) SMT proof artifact is machine-checkable (Alethe format)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProofCertificate {
    /// Certificate format version (for forward compatibility).
    pub version: u32,
    /// Name of the verified kernel.
    pub kernel_name: String,
    /// Input specification: what bounds were assumed on each input variable.
    pub input_spec: InputBoundsRecord,
    /// Proved output bounds: the verification guarantee.
    pub output_bounds: OutputBoundsRecord,
    /// Per-element output bounds from NY (full tensor when available).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub output_tensor: Option<OutputTensorBounds>,
    /// Width of the output interval (upper - lower).
    pub output_width: f32,
    /// Whether the output bounds are provably finite.
    pub is_finite: bool,
    /// Propagation method that produced the bounds.
    pub method: PropMethod,
    /// Soundness classification: `Sound` if no heuristics were used.
    pub soundness_mode: VerificationSoundnessMode,
    /// If CROWN was attempted but failed, the fallback reason.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub crown_fallback_reason: Option<String>,
    /// Optional ay SMT cross-verification outcome.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub smt_outcome: Option<String>,
    /// ISO 8601 timestamp of certificate generation.
    pub generated_at: String,

    // --- v2 additions (all Option for backward compatibility) ---
    /// Per-layer bound trace enabling independent IBP re-checking.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub layer_bounds: Option<Vec<LayerBoundRecord>>,
    /// Kani formal verification status for this kernel.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub kani_status: Option<KaniProofRecord>,
    /// SHA-256 hex digest of the model weights used during verification.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub weight_hash: Option<String>,
    /// SHA-256 hex digest of the Rust source file containing the kernel.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub source_hash: Option<String>,
    /// Version string of the NY verifier used.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub verifier_version: Option<String>,
    /// Fraction of layers verified with CROWN (vs IBP fallback).
    /// Populated from layer_bounds when available. None for legacy certificates.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub crown_coverage: Option<f32>,
    /// Number of layers that fell back to IBP.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ibp_fallback_count: Option<usize>,

    // --- v3 additions (all Option for backward compatibility) ---
    /// Alethe proof text from ay SMT solver (machine-checkable UNSAT proof).
    ///
    /// When present, an independent Alethe proof checker can validate the SMT
    /// result without trusting the solver. This is the raw proof artifact
    /// captured from ay's proof output during verification.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub smt_proof_alethe: Option<String>,
    /// Verdict from independent validation of the SMT proof artifact.
    ///
    /// `Some(Verified)` means the proof was checked and is valid.
    /// `Some(Unchecked)` means a proof was produced but not yet validated.
    /// `None` means no proof artifact was generated (pre-v3 certificates).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub smt_proof_verdict: Option<SmtProofVerdict>,

    // --- v4 additions (all Option for backward compatibility) ---
    /// SHA-256 hex digest of the canonical certificate content (all fields
    /// except `content_hash` and `hmac_signature`). Enables integrity
    /// verification without a shared secret.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub content_hash: Option<String>,
    /// HMAC-SHA256 hex digest over the `content_hash`, keyed with a shared
    /// secret. Prevents forgery: an attacker cannot recompute a valid
    /// signature without the key, even if they maintain internal consistency.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub hmac_signature: Option<String>,

    // --- v5 additions (all Option for backward compatibility) ---
    /// How F16/BF16 precision loss was modeled during verification (#3023).
    ///
    /// `None` for pre-v5 certificates (treated as `F32Only`).
    /// When `Some(F16Aware { .. })`, the certificate proves the F16 execution
    /// path, not just the F32 algorithm.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub precision_model: Option<PrecisionModel>,

    // --- v6 additions (all Option for backward compatibility) ---
    /// Constructive proof certificate from NY (#4315).
    ///
    /// Contains machine-checkable proof data: IBP recomputation data,
    /// verified output bounds, and optional Lean4 export text. An auditor
    /// can verify the certificate independently of NY by
    /// recomputing the interval arithmetic or checking the Lean4 proof.
    ///
    /// `None` for pre-v6 certificates or when constructive proof generation
    /// was not attempted (e.g., non-linear-ReLU networks).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub constructive_proof: Option<ConstructiveProofData>,
}

/// Current certificate format version.
pub const CERTIFICATE_VERSION: u32 = 6;

impl ProofCertificate {
    /// Generate a proof certificate from a `KernelVerification` result.
    ///
    /// `input_spec` describes the input bounds used for verification.
    /// v2 fields default to `None`; use builder methods to populate them.
    #[must_use]
    pub fn from_verification(result: &KernelVerification, input_spec: InputBoundsRecord) -> Self {
        Self {
            version: CERTIFICATE_VERSION,
            kernel_name: result.kernel_name.clone(),
            input_spec,
            output_bounds: OutputBoundsRecord::from_verification(result),
            output_tensor: result.output_tensor.clone(),
            output_width: result.output_width,
            is_finite: result.is_finite,
            method: result.method,
            soundness_mode: result.soundness_mode,
            crown_fallback_reason: result.crown_fallback_reason.clone(),
            smt_outcome: None,
            generated_at: now_iso8601(),
            layer_bounds: None,
            kani_status: None,
            weight_hash: None,
            source_hash: None,
            verifier_version: None,
            crown_coverage: None,
            ibp_fallback_count: None,
            smt_proof_alethe: None,
            smt_proof_verdict: None,
            content_hash: None,
            hmac_signature: None,
            precision_model: None,
            constructive_proof: None,
        }
    }

    /// Attach an SMT cross-verification outcome to the certificate.
    #[must_use]
    pub fn with_smt_outcome(mut self, outcome: &str) -> Self {
        self.smt_outcome = Some(outcome.to_string());
        self
    }

    /// Attach per-layer bound traces for independent checking.
    ///
    /// Also computes and populates `crown_coverage` and `ibp_fallback_count`
    /// from the per-layer propagation method provenance.
    #[must_use]
    pub fn with_layer_bounds(mut self, bounds: Vec<LayerBoundRecord>) -> Self {
        let total = bounds.len();
        if total > 0 {
            let crown = bounds.iter().filter(|r| r.method.is_tight()).count();
            self.crown_coverage = Some(crown as f32 / total as f32);
            self.ibp_fallback_count = Some(total - crown);
        }
        self.layer_bounds = Some(bounds);
        self
    }

    /// Attach Kani proof status from `kani_status.json`.
    #[must_use]
    pub fn with_kani_status(mut self, status: KaniProofRecord) -> Self {
        self.kani_status = Some(status);
        self
    }

    /// Attach a SHA-256 hash of the model weights file.
    #[must_use]
    pub fn with_weight_hash(mut self, hash: String) -> Self {
        self.weight_hash = Some(hash);
        self
    }

    /// Attach a SHA-256 hash of the kernel source file.
    #[must_use]
    pub fn with_source_hash(mut self, hash: String) -> Self {
        self.source_hash = Some(hash);
        self
    }

    /// Attach the NY verifier version string.
    #[must_use]
    pub fn with_verifier_version(mut self, version: String) -> Self {
        self.verifier_version = Some(version);
        self
    }

    /// Attach an SMT proof artifact and its validation verdict.
    ///
    /// `proof_text` is the raw Alethe proof from ay. `verdict` records
    /// whether the proof was independently validated. When both
    /// `smt_outcome == "Proven"` and `verdict == Verified`, the certificate
    /// contains a machine-checkable proof — not just a solver assertion.
    #[must_use]
    pub fn with_smt_proof(mut self, proof_text: String, verdict: SmtProofVerdict) -> Self {
        self.smt_proof_alethe = Some(proof_text);
        self.smt_proof_verdict = Some(verdict);
        self
    }

    /// Attach F16/BF16 precision model metadata (#3023).
    ///
    /// Records whether the verification modeled F16 precision loss at dtype
    /// cast points, enabling consumers to distinguish "proved for F32 algorithm"
    /// from "proved for F16 execution."
    #[must_use]
    pub fn with_precision_model(mut self, model: PrecisionModel) -> Self {
        self.precision_model = Some(model);
        self
    }

    /// Attach a constructive proof certificate from NY (#4315).
    ///
    /// Contains machine-checkable proof data: IBP recomputation data and/or
    /// Lean4 export text. An auditor can verify independently of NY.
    #[must_use]
    pub fn with_constructive_proof(mut self, proof: ConstructiveProofData) -> Self {
        self.constructive_proof = Some(proof);
        self
    }

    /// Whether this certificate has a constructive proof attached.
    #[must_use]
    pub fn has_constructive_proof(&self) -> bool {
        self.constructive_proof.is_some()
    }

    /// Validate structural self-consistency of the certificate.
    ///
    /// Checks v1 fields (version, name, bounds consistency) and v2 fields
    /// (layer index ordering, hash format) when present.
    pub fn validate(&self) -> Result<(), CertificateError> {
        if self.version == 0 || self.version > CERTIFICATE_VERSION {
            return Err(CertificateError::UnsupportedVersion {
                version: self.version,
                max_supported: CERTIFICATE_VERSION,
            });
        }
        if self.kernel_name.is_empty() {
            return Err(CertificateError::EmptyKernelName);
        }
        if self.is_finite {
            if !self.output_bounds.lower.is_finite() || !self.output_bounds.upper.is_finite() {
                return Err(CertificateError::FiniteFlagMismatch {
                    lower: self.output_bounds.lower,
                    upper: self.output_bounds.upper,
                });
            }
        }
        // Inverted bounds are always invalid, regardless of is_finite flag.
        // Use is_finite() guards to avoid IEEE 754 NaN comparison pitfalls
        // (NaN > NaN returns false, bypassing the check).
        if self.output_bounds.lower.is_finite()
            && self.output_bounds.upper.is_finite()
            && self.output_bounds.lower > self.output_bounds.upper
        {
            return Err(CertificateError::InvertedBounds {
                lower: self.output_bounds.lower,
                upper: self.output_bounds.upper,
            });
        }
        self.validate_output_width()?;

        // v2 validations
        if let Some(ref bounds) = self.layer_bounds {
            if bounds.is_empty() {
                return Err(CertificateError::EmptyLayerBounds);
            }
            for (i, record) in bounds.iter().enumerate() {
                if record.layer_index != i {
                    return Err(CertificateError::LayerIndexMismatch {
                        expected: i,
                        actual: record.layer_index,
                    });
                }
            }
        }
        if let Some(ref hash) = self.weight_hash {
            validate_sha256_hex(hash).map_err(|()| CertificateError::InvalidHash {
                field: "weight_hash".to_string(),
                value: hash.clone(),
            })?;
        }
        if let Some(ref hash) = self.source_hash {
            validate_sha256_hex(hash).map_err(|()| CertificateError::InvalidHash {
                field: "source_hash".to_string(),
                value: hash.clone(),
            })?;
        }
        // v4 validations
        if let Some(ref hash) = self.content_hash {
            validate_sha256_hex(hash).map_err(|()| CertificateError::InvalidHash {
                field: "content_hash".to_string(),
                value: hash.clone(),
            })?;
        }
        if let Some(ref sig) = self.hmac_signature {
            validate_sha256_hex(sig).map_err(|()| CertificateError::InvalidHash {
                field: "hmac_signature".to_string(),
                value: sig.clone(),
            })?;
        }
        Ok(())
    }

    /// Validate output_width consistency with output bounds.
    ///
    /// When both bounds are finite, the width must be finite and match
    /// `upper - lower` within floating-point tolerance.
    fn validate_output_width(&self) -> Result<(), CertificateError> {
        if !self.output_bounds.lower.is_finite() || !self.output_bounds.upper.is_finite() {
            return Ok(());
        }
        if !self.output_width.is_finite() {
            return Err(CertificateError::NonFiniteOutputWidth {
                width: self.output_width,
                lower: self.output_bounds.lower,
                upper: self.output_bounds.upper,
            });
        }
        let expected = self.output_bounds.upper - self.output_bounds.lower;
        // IEEE 754 overflow guard: finite bounds whose difference overflows f32::MAX.
        if !expected.is_finite() {
            return Err(CertificateError::OutputWidthMismatch {
                width: self.output_width,
                expected,
                lower: self.output_bounds.lower,
                upper: self.output_bounds.upper,
            });
        }
        let abs_diff = (self.output_width - expected).abs();
        let rel_threshold = 1e-5 * expected.abs().max(self.output_width.abs()).max(1.0);
        if abs_diff > 1e-6 && abs_diff > rel_threshold {
            return Err(CertificateError::OutputWidthMismatch {
                width: self.output_width,
                expected,
                lower: self.output_bounds.lower,
                upper: self.output_bounds.upper,
            });
        }
        Ok(())
    }

    /// Serialize to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, VerifyError> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

// Pipeline factory functions extracted to certificate_pipeline.rs (500-line limit).
#[path = "certificate_pipeline.rs"]
mod pipeline;
pub(crate) use pipeline::now_iso8601;
pub use pipeline::{
    certificate_from_pipeline, certificate_from_pipeline_enriched, CertificateEnrichment,
};

// Cryptographic integrity (HMAC-SHA256 signing + content hashing).
#[path = "certificate_integrity.rs"]
pub mod integrity;
pub use integrity::{
    compute_content_hash, sign_bundle, sign_certificate, verify_bundle_signatures,
    verify_bundle_signatures_strict, verify_content_hash, verify_signature, BundleIntegrityError,
    IntegrityError,
};

// Re-export so test modules can access via `use super::*`.
pub use crate::status::ParamInputRecord;

#[cfg(kani)]
#[path = "kani_certificate.rs"]
mod kani_certificate;

#[cfg(test)]
#[path = "certificate_test_helpers.rs"]
mod certificate_test_helpers;
#[cfg(test)]
#[path = "certificate_v2_tests.rs"]
mod certificate_v2_tests;
#[cfg(test)]
#[path = "certificate_v2_tests_enriched.rs"]
mod certificate_v2_tests_enriched;
#[cfg(test)]
#[path = "certificate_tests.rs"]
mod tests;
