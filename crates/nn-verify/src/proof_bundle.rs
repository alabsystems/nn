// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Proof certificate serialization — `ProofBundle` for shipping models with
//! machine-checkable proof certificates.
//!
//! A `ProofBundle` aggregates all verification evidence for a model:
//! bound certificates (from NY IBP/CROWN/AlphaCROWN), Kani harness
//! summaries, and CROWN verification summaries. The bundle serializes to JSON
//! via serde, enabling auditors to check the certificate without re-running
//! the verifier.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use nn_verify::proof_bundle::{ProofBundleBuilder, BoundCertificate, VerificationMethod};
//!
//! let bundle = ProofBundleBuilder::new("kokoro_v1", "abc123...def456...")
//!     .add_bound_certificate(BoundCertificate {
//!         kernel_name: "snake".to_string(),
//!         input_bounds: (-1.0, 1.0),
//!         output_bounds: (-3.87, 3.87),
//!         method: VerificationMethod::Crown,
//!         is_tight: true,
//!     })
//!     .set_kani_summary(754, 754, 0, 0)
//!     .set_crown_summary(51, 26, 14, 11)
//!     .build()
//!     .expect("valid bundle");
//!
//! let json = bundle.to_json().expect("serialize");
//! ```
//!
//! Part of #3561 (Proof certificate serialization).

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::certificate::now_iso8601;
use crate::certificate_types::validate_sha256_hex;
use crate::error::VerifyError;

/// Verification method used to produce a bound certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum VerificationMethod {
    /// Interval Bound Propagation — fast, sound, may be loose.
    Ibp,
    /// CROWN linear relaxation — tighter than IBP.
    Crown,
    /// Alpha-CROWN: optimized linear relaxation with learnable slopes.
    AlphaCrown,
    /// Beta-CROWN: branch-and-bound with CROWN bounds.
    BetaCrown,
    /// Analytical: closed-form bounds from mathematical analysis.
    Analytical,
}

/// A single bound certificate for a verified kernel.
///
/// Records the input/output bounds and the verification method used.
/// `is_tight` indicates whether the method produces CROWN-quality (tight)
/// bounds, as opposed to loose IBP bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundCertificate {
    /// Name of the verified kernel (e.g., "snake", "instance_norm").
    pub kernel_name: String,
    /// Input bounds used during verification: (lower, upper).
    pub input_bounds: (f32, f32),
    /// Proved output bounds: (lower, upper).
    pub output_bounds: (f32, f32),
    /// Verification method that produced these bounds.
    pub method: VerificationMethod,
    /// Whether the bounds are tight (CROWN-quality or better).
    pub is_tight: bool,
}

/// Summary of Kani formal verification for the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KaniSummary {
    /// Total number of Kani proof harnesses.
    pub total_harnesses: usize,
    /// Number of harnesses that passed verification.
    pub passed: usize,
    /// Number of harnesses that failed verification.
    pub failed: usize,
    /// Number of harnesses that timed out.
    pub timeout: usize,
}

/// Summary of NY CROWN verification for the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrownSummary {
    /// Total number of verification entries.
    pub total_entries: usize,
    /// Entries with sound verification (no heuristic approximations).
    pub sound: usize,
    /// Entries using heuristic approximations.
    pub heuristic: usize,
    /// Entries with vacuous (trivially true) bounds.
    pub vacuous: usize,
}

/// A complete proof bundle for a verified model.
///
/// Contains all verification evidence needed for offline auditing:
/// model identity (hash + name), bound certificates, Kani and CROWN
/// summaries, creation metadata. Serializes to JSON for deployment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProofBundle {
    /// SHA-256 hex digest of the model weights file.
    pub model_hash: String,
    /// Human-readable model name (e.g., "kokoro_v1").
    pub model_name: String,
    /// Per-kernel bound certificates from NY verification.
    pub bound_certificates: Vec<BoundCertificate>,
    /// Summary of Kani formal verification harnesses.
    pub kani_summary: Option<KaniSummary>,
    /// Summary of NY CROWN verification entries.
    pub gamma_crown_summary: Option<CrownSummary>,
    /// ISO 8601 timestamp of bundle creation.
    pub created_at: String,
    /// Version of the nn framework that produced this bundle.
    pub nn_version: String,
}

impl ProofBundle {
    /// Validate structural self-consistency of the bundle.
    ///
    /// Checks:
    /// - `model_hash` is a valid SHA-256 hex digest (64 hex chars)
    /// - `model_name` is non-empty
    /// - At least one bound certificate exists
    /// - No NaN values in any bound certificate's input/output bounds
    /// - Input bounds are not inverted (lower <= upper)
    /// - Output bounds are not inverted (lower <= upper)
    /// - Kani summary counts are consistent (passed + failed + timeout <= total)
    /// - CROWN summary counts are consistent (sound + heuristic + vacuous <= total)
    pub fn validate(&self) -> Result<(), ProofBundleError> {
        // Model hash must be valid SHA-256
        validate_sha256_hex(&self.model_hash).map_err(|()| ProofBundleError::InvalidModelHash {
            hash: self.model_hash.clone(),
        })?;

        // Model name must be non-empty
        if self.model_name.is_empty() {
            return Err(ProofBundleError::EmptyModelName);
        }

        // At least one certificate required
        if self.bound_certificates.is_empty() {
            return Err(ProofBundleError::NoCertificates);
        }

        // Validate each bound certificate
        for (i, cert) in self.bound_certificates.iter().enumerate() {
            // Check for NaN — use is_finite() before comparison per IEEE 754 rule
            if !cert.input_bounds.0.is_finite() || !cert.input_bounds.1.is_finite() {
                return Err(ProofBundleError::NonFiniteBounds {
                    certificate_index: i,
                    kernel_name: cert.kernel_name.clone(),
                    field: "input_bounds".to_string(),
                    lower: cert.input_bounds.0,
                    upper: cert.input_bounds.1,
                });
            }
            if !cert.output_bounds.0.is_finite() || !cert.output_bounds.1.is_finite() {
                return Err(ProofBundleError::NonFiniteBounds {
                    certificate_index: i,
                    kernel_name: cert.kernel_name.clone(),
                    field: "output_bounds".to_string(),
                    lower: cert.output_bounds.0,
                    upper: cert.output_bounds.1,
                });
            }

            // Inverted bounds check (safe — we already verified finiteness)
            if cert.input_bounds.0 > cert.input_bounds.1 {
                return Err(ProofBundleError::InvertedBounds {
                    certificate_index: i,
                    kernel_name: cert.kernel_name.clone(),
                    field: "input_bounds".to_string(),
                    lower: cert.input_bounds.0,
                    upper: cert.input_bounds.1,
                });
            }
            if cert.output_bounds.0 > cert.output_bounds.1 {
                return Err(ProofBundleError::InvertedBounds {
                    certificate_index: i,
                    kernel_name: cert.kernel_name.clone(),
                    field: "output_bounds".to_string(),
                    lower: cert.output_bounds.0,
                    upper: cert.output_bounds.1,
                });
            }

            // Kernel name must be non-empty
            if cert.kernel_name.is_empty() {
                return Err(ProofBundleError::EmptyKernelName {
                    certificate_index: i,
                });
            }
        }

        // Validate Kani summary consistency
        if let Some(ref kani) = self.kani_summary {
            let sum = kani.passed + kani.failed + kani.timeout;
            if sum > kani.total_harnesses {
                return Err(ProofBundleError::InconsistentKaniSummary {
                    total: kani.total_harnesses,
                    sum,
                });
            }
        }

        // Validate CROWN summary consistency
        if let Some(ref crown) = self.gamma_crown_summary {
            let sum = crown.sound + crown.heuristic + crown.vacuous;
            if sum > crown.total_entries {
                return Err(ProofBundleError::InconsistentCrownSummary {
                    total: crown.total_entries,
                    sum,
                });
            }
        }

        Ok(())
    }

    /// Serialize to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, VerifyError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, VerifyError> {
        Ok(serde_json::from_str(json)?)
    }

    /// Save the bundle to a JSON file using atomic write semantics.
    ///
    /// Writes to a `.tmp` file first, then renames to the target path.
    /// This prevents partial writes from corrupting the certificate file.
    pub fn save(&self, path: &Path) -> Result<(), VerifyError> {
        use std::io::Write;

        let json = serde_json::to_string_pretty(self)?;
        let dir = path.parent().unwrap_or(Path::new("."));

        let tmp_path = {
            let mut s = path.as_os_str().to_owned();
            s.push(".tmp");
            std::path::PathBuf::from(s)
        };
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp_path)?;
        file.write_all(json.as_bytes())?;
        file.sync_all()?;
        drop(file);

        if let Err(e) = std::fs::rename(&tmp_path, path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(VerifyError::Io(e));
        }
        let _ = std::fs::File::open(dir).and_then(|f| f.sync_all());
        Ok(())
    }

    /// Load a bundle from a JSON file.
    pub fn load(path: &Path) -> Result<Self, VerifyError> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_json(&contents)
    }

    /// Number of bound certificates in the bundle.
    #[must_use]
    pub fn certificate_count(&self) -> usize {
        self.bound_certificates.len()
    }

    /// Number of bound certificates with tight (CROWN-quality) bounds.
    #[must_use]
    pub fn tight_count(&self) -> usize {
        self.bound_certificates
            .iter()
            .filter(|c| c.is_tight)
            .count()
    }
}

/// Builder for constructing a `ProofBundle` with validation.
///
/// Requires `model_name` and `model_hash` at construction. Certificates
/// and summaries can be added incrementally. `build()` validates the
/// bundle before returning it.
#[derive(Debug)]
pub struct ProofBundleBuilder {
    model_name: String,
    model_hash: String,
    bound_certificates: Vec<BoundCertificate>,
    kani_summary: Option<KaniSummary>,
    gamma_crown_summary: Option<CrownSummary>,
    nn_version: String,
}

impl ProofBundleBuilder {
    /// Create a new builder with the required model identity fields.
    ///
    /// `model_hash` should be a SHA-256 hex digest (64 hex chars) of the
    /// model weights file. Use `compute_file_hash()` from `certificate_types`.
    #[must_use]
    pub fn new(model_name: &str, model_hash: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            model_hash: model_hash.to_string(),
            bound_certificates: Vec::new(),
            kani_summary: None,
            gamma_crown_summary: None,
            nn_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Add a bound certificate to the bundle.
    #[must_use]
    pub fn add_bound_certificate(mut self, cert: BoundCertificate) -> Self {
        self.bound_certificates.push(cert);
        self
    }

    /// Set the Kani verification summary.
    #[must_use]
    pub fn set_kani_summary(
        mut self,
        total_harnesses: usize,
        passed: usize,
        failed: usize,
        timeout: usize,
    ) -> Self {
        self.kani_summary = Some(KaniSummary {
            total_harnesses,
            passed,
            failed,
            timeout,
        });
        self
    }

    /// Set the NY CROWN verification summary.
    #[must_use]
    pub fn set_crown_summary(
        mut self,
        total_entries: usize,
        sound: usize,
        heuristic: usize,
        vacuous: usize,
    ) -> Self {
        self.gamma_crown_summary = Some(CrownSummary {
            total_entries,
            sound,
            heuristic,
            vacuous,
        });
        self
    }

    /// Override the nn version string (defaults to `CARGO_PKG_VERSION`).
    #[must_use]
    pub fn with_nn_version(mut self, version: &str) -> Self {
        self.nn_version = version.to_string();
        self
    }

    /// Build the `ProofBundle`, validating all fields.
    ///
    /// # Errors
    ///
    /// Returns `ProofBundleError` if validation fails (empty hash,
    /// no certificates, NaN bounds, inconsistent summaries, etc.).
    pub fn build(self) -> Result<ProofBundle, ProofBundleError> {
        let bundle = ProofBundle {
            model_hash: self.model_hash,
            model_name: self.model_name,
            bound_certificates: self.bound_certificates,
            kani_summary: self.kani_summary,
            gamma_crown_summary: self.gamma_crown_summary,
            created_at: now_iso8601(),
            nn_version: self.nn_version,
        };
        bundle.validate()?;
        Ok(bundle)
    }
}

/// Errors specific to `ProofBundle` validation.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ProofBundleError {
    /// Model hash is not a valid SHA-256 hex digest.
    #[error("invalid model hash (expected 64 hex chars): {hash}")]
    InvalidModelHash { hash: String },

    /// Model name is empty.
    #[error("model name must not be empty")]
    EmptyModelName,

    /// Bundle contains no bound certificates.
    #[error("proof bundle must contain at least one bound certificate")]
    NoCertificates,

    /// Bound certificate contains non-finite (NaN or Inf) values.
    #[error(
        "certificate[{certificate_index}] ({kernel_name}) has non-finite {field}: \
         lower={lower}, upper={upper}"
    )]
    NonFiniteBounds {
        certificate_index: usize,
        kernel_name: String,
        field: String,
        lower: f32,
        upper: f32,
    },

    /// Bound certificate has inverted bounds (lower > upper).
    #[error(
        "certificate[{certificate_index}] ({kernel_name}) has inverted {field}: \
         lower={lower} > upper={upper}"
    )]
    InvertedBounds {
        certificate_index: usize,
        kernel_name: String,
        field: String,
        lower: f32,
        upper: f32,
    },

    /// Bound certificate has empty kernel name.
    #[error("certificate[{certificate_index}] has empty kernel name")]
    EmptyKernelName { certificate_index: usize },

    /// Kani summary counts exceed total.
    #[error("kani summary inconsistent: passed + failed + timeout ({sum}) > total ({total})")]
    InconsistentKaniSummary { total: usize, sum: usize },

    /// CROWN summary counts exceed total.
    #[error("crown summary inconsistent: sound + heuristic + vacuous ({sum}) > total ({total})")]
    InconsistentCrownSummary { total: usize, sum: usize },
}

#[cfg(kani)]
#[path = "proof_bundle_kani.rs"]
mod kani_proofs;

#[cfg(kani)]
#[path = "kani_proof_bundle.rs"]
mod kani_proof_bundle;

#[cfg(test)]
#[path = "proof_bundle_tests.rs"]
mod tests;
