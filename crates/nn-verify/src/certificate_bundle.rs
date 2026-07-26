// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Certificate bundle and error types for proof certificate validation.
//!
//! Extracted from `certificate.rs` to stay within the 500-line file limit.
//! A `CertificateBundle` groups multiple `ProofCertificate`s for a model
//! or verification run and supports atomic save/load.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::{ProofCertificate, CERTIFICATE_VERSION};
use crate::error::VerifyError;
use crate::soundness_compat::VerificationSoundnessMode;

/// A bundle of proof certificates for a model or verification run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CertificateBundle {
    /// Bundle format version.
    pub version: u32,
    /// Name of the model or verification scope.
    pub model_name: String,
    /// Per-kernel proof certificates.
    pub certificates: Vec<ProofCertificate>,
    /// ISO 8601 timestamp of bundle generation.
    pub generated_at: String,
}

impl CertificateBundle {
    /// Create a new empty bundle for the given model name.
    #[must_use]
    pub fn new(model_name: &str) -> Self {
        Self {
            version: CERTIFICATE_VERSION,
            model_name: model_name.to_string(),
            certificates: Vec::new(),
            generated_at: super::now_iso8601(),
        }
    }

    /// Add a certificate to the bundle (builder pattern).
    #[must_use]
    pub fn with_certificate(mut self, cert: ProofCertificate) -> Self {
        self.certificates.push(cert);
        self
    }

    /// Add a certificate to the bundle (mutable reference variant).
    pub fn push(&mut self, cert: ProofCertificate) {
        self.certificates.push(cert);
    }

    /// Number of certificates in the bundle.
    #[must_use]
    pub fn len(&self) -> usize {
        self.certificates.len()
    }

    /// Whether the bundle is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.certificates.is_empty()
    }

    /// Validate all certificates in the bundle.
    pub fn validate_all(&self) -> Result<(), (usize, CertificateError)> {
        for (i, cert) in self.certificates.iter().enumerate() {
            cert.validate().map_err(|e| (i, e))?;
        }
        Ok(())
    }

    /// Count certificates where `is_finite` is true.
    #[must_use]
    pub fn verified_count(&self) -> usize {
        self.certificates.iter().filter(|c| c.is_finite).count()
    }

    /// Count certificates with `Sound` soundness mode.
    #[must_use]
    pub fn sound_count(&self) -> usize {
        self.certificates
            .iter()
            .filter(|c| c.soundness_mode == VerificationSoundnessMode::Sound)
            .count()
    }

    /// Save the bundle to a JSON file using atomic write semantics.
    pub fn save(&self, path: &Path) -> Result<(), VerifyError> {
        use std::io::Write;

        let json = serde_json::to_string_pretty(self)?;
        let dir = path.parent().unwrap_or(Path::new("."));

        // Append ".tmp" to full path rather than replacing the extension.
        // `with_extension("proof.json.tmp")` would replace only the last
        // extension component, producing surprising paths like
        // `model.proof.proof.json.tmp` for `model.proof.json`.
        let tmp_path = {
            let mut s = path.as_os_str().to_owned();
            s.push(".tmp");
            std::path::PathBuf::from(s)
        };
        // Use create+truncate (not create_new) so re-runs work if a
        // prior tmp file was left behind after an interrupted save.
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
        Ok(serde_json::from_str(&contents)?)
    }

    /// Create a sub-bundle containing only certificates whose `kernel_name`
    /// is in the given set. The new bundle uses the provided `model_name`.
    ///
    /// Returns a new `CertificateBundle` with a fresh timestamp. The order
    /// of certificates matches the order they appear in the original bundle.
    #[must_use]
    pub fn filter_by_names(&self, model_name: &str, names: &[&str]) -> Self {
        let set: std::collections::HashSet<&str> = names.iter().copied().collect();
        let filtered = self
            .certificates
            .iter()
            .filter(|c| set.contains(c.kernel_name.as_str()))
            .cloned()
            .collect();
        Self {
            version: CERTIFICATE_VERSION,
            model_name: model_name.to_string(),
            certificates: filtered,
            generated_at: super::now_iso8601(),
        }
    }

    /// Whether all certificates have a non-empty `source_hash`.
    #[must_use]
    pub fn all_have_source_hash(&self) -> bool {
        self.certificates
            .iter()
            .all(|c| c.source_hash.as_ref().is_some_and(|h| !h.is_empty()))
    }

    /// Whether all certificates have `Sound` soundness mode.
    ///
    /// Returns `false` if any certificate uses `Heuristic` or `IbpValidated` mode.
    #[must_use]
    pub fn all_sound(&self) -> bool {
        self.certificates
            .iter()
            .all(|c| c.soundness_mode == VerificationSoundnessMode::Sound)
    }
}

/// Errors specific to proof certificate validation.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CertificateError {
    #[error("unsupported certificate version {version} (max supported: {max_supported})")]
    UnsupportedVersion { version: u32, max_supported: u32 },

    #[error("certificate has empty kernel name")]
    EmptyKernelName,

    #[error("is_finite=true but bounds are non-finite: lower={lower}, upper={upper}")]
    FiniteFlagMismatch { lower: f32, upper: f32 },

    #[error("inverted output bounds: lower={lower} > upper={upper}")]
    InvertedBounds { lower: f32, upper: f32 },

    #[error("layer_bounds is present but empty")]
    EmptyLayerBounds,

    #[error("layer_bounds[{expected}] has layer_index={actual}")]
    LayerIndexMismatch { expected: usize, actual: usize },

    #[error("{field} is not a valid SHA-256 hex digest: {value}")]
    InvalidHash { field: String, value: String },

    #[error(
        "output_width ({width}) inconsistent with bounds: expected {expected} (lower={lower}, upper={upper})"
    )]
    OutputWidthMismatch {
        width: f32,
        expected: f32,
        lower: f32,
        upper: f32,
    },

    #[error(
        "output_width is non-finite ({width}) but bounds are finite (lower={lower}, upper={upper})"
    )]
    NonFiniteOutputWidth { width: f32, lower: f32, upper: f32 },
}
