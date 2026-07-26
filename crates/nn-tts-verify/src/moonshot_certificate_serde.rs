// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Serde serialization and deserialization for [`MoonshotCertificate`].
//!
//! `PropertyCertificate` uses `&'static str` for `property_name`, so it cannot
//! implement `Deserialize` directly. This module provides:
//! - `MoonshotCertificate::to_json_serde()` — serde-based JSON serialization
//! - `MoonshotCertificate::from_json()` — deserialization from JSON string
//! - `MoonshotCertificate::save()` — atomic write to JSON file
//! - `MoonshotCertificate::load()` — read from JSON file
//!
//! The deserialization path uses [`PropertyCertificateOwned`] as an intermediate
//! type with `String` for `property_name`, then maps through `PROPERTY_NAMES`
//! to recover `&'static str` references.

use serde::Deserialize;

use super::{MoonshotCertificate, PropertyCertificate, SubCondition, VerificationLevel};
use crate::moonshot::PROPERTY_NAMES;

/// Owned variant of [`PropertyCertificate`] for deserialization.
///
/// Identical to `PropertyCertificate` except `property_name` is `String`
/// instead of `&'static str`, allowing serde to deserialize it from JSON.
#[derive(Debug, Clone, Deserialize)]
struct PropertyCertificateOwned {
    property_index: usize,
    property_name: String,
    level: VerificationLevel,
    proof_artifacts: Vec<String>,
    assumptions: Vec<String>,
    bound_value: Option<f64>,
    threshold: Option<f64>,
    /// Sub-condition results (v3+). Defaults to empty for v1/v2 certificates.
    #[serde(default)]
    sub_results: Vec<SubCondition>,
    /// Constructive proof Lean4 source (v4+). Defaults to None for older certs.
    #[serde(default)]
    constructive_proof_lean4: Option<String>,
}

/// Owned variant of [`MoonshotCertificate`] for deserialization.
#[derive(Debug, Clone, Deserialize)]
struct MoonshotCertificateOwned {
    /// Defaults to 1 for backward compatibility with pre-schema-version certificates.
    #[serde(default = "default_schema_version")]
    schema_version: u32,
    model_name: String,
    input_specification: String,
    properties: Vec<PropertyCertificateOwned>,
    source_hash: String,
    verification_date: String,
    verification_dim: Option<usize>,
    all_at_least_partial: bool,
    all_proven: bool,
    /// Constructive proof count (v4+). Defaults to 0 for older certs.
    #[serde(default)]
    constructive_proof_count: usize,
}

fn default_schema_version() -> u32 {
    1
}

/// Error type for certificate serialization and deserialization.
#[derive(Debug)]
pub enum CertificateDeserializeError {
    /// JSON parsing failed.
    Json(serde_json::Error),
    /// File I/O failed.
    Io(std::io::Error),
    /// Property index out of range (0-7 expected).
    PropertyIndexOutOfRange { index: usize, property_name: String },
    /// Property name in JSON doesn't match the canonical name for that index.
    PropertyNameMismatch {
        index: usize,
        expected: &'static str,
        found: String,
    },
}

impl std::fmt::Display for CertificateDeserializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(e) => write!(f, "JSON parse error: {e}"),
            Self::Io(e) => write!(f, "file I/O error: {e}"),
            Self::PropertyIndexOutOfRange {
                index,
                property_name,
            } => {
                write!(
                    f,
                    "property index {index} out of range (0-7) for property '{property_name}'"
                )
            }
            Self::PropertyNameMismatch {
                index,
                expected,
                found,
            } => {
                write!(
                    f,
                    "property name mismatch at index {index}: expected '{expected}', found '{found}'"
                )
            }
        }
    }
}

impl std::error::Error for CertificateDeserializeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(e) => Some(e),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for CertificateDeserializeError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

impl From<std::io::Error> for CertificateDeserializeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl PropertyCertificateOwned {
    /// Convert to `PropertyCertificate` by resolving `property_name` through
    /// the canonical `PROPERTY_NAMES` array.
    ///
    /// Validates that `property_index` is in range and the deserialized name
    /// matches the canonical name.
    fn into_static(self) -> Result<PropertyCertificate, CertificateDeserializeError> {
        if self.property_index >= PROPERTY_NAMES.len() {
            return Err(CertificateDeserializeError::PropertyIndexOutOfRange {
                index: self.property_index,
                property_name: self.property_name,
            });
        }

        let canonical_name = PROPERTY_NAMES[self.property_index];
        if self.property_name != canonical_name {
            return Err(CertificateDeserializeError::PropertyNameMismatch {
                index: self.property_index,
                expected: canonical_name,
                found: self.property_name,
            });
        }

        Ok(PropertyCertificate {
            property_index: self.property_index,
            property_name: canonical_name,
            level: self.level,
            proof_artifacts: self.proof_artifacts,
            assumptions: self.assumptions,
            bound_value: self.bound_value,
            threshold: self.threshold,
            sub_results: self.sub_results,
            constructive_proof_lean4: self.constructive_proof_lean4,
        })
    }
}

impl MoonshotCertificate {
    /// Serialize to JSON using serde.
    ///
    /// This produces canonical JSON output. For backward-compatible
    /// hand-formatted output, use [`to_json()`](Self::to_json).
    pub fn to_json_serde(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize a `MoonshotCertificate` from a JSON string.
    ///
    /// Property names are validated against the canonical `PROPERTY_NAMES`
    /// array to recover `&'static str` references.
    ///
    /// # Errors
    ///
    /// Returns an error if JSON parsing fails, a property index is out of
    /// range (0-7), or a property name doesn't match its canonical value.
    pub fn from_json(json: &str) -> Result<Self, CertificateDeserializeError> {
        let owned: MoonshotCertificateOwned = serde_json::from_str(json)?;

        let properties: Result<Vec<PropertyCertificate>, _> = owned
            .properties
            .into_iter()
            .map(PropertyCertificateOwned::into_static)
            .collect();

        Ok(Self {
            schema_version: owned.schema_version,
            model_name: owned.model_name,
            input_specification: owned.input_specification,
            properties: properties?,
            source_hash: owned.source_hash,
            verification_date: owned.verification_date,
            verification_dim: owned.verification_dim,
            all_at_least_partial: owned.all_at_least_partial,
            all_proven: owned.all_proven,
            constructive_proof_count: owned.constructive_proof_count,
        })
    }

    /// Save the certificate to a JSON file using atomic write.
    ///
    /// Uses write-to-tmp → fsync → rename → fsync(dir) to prevent
    /// partial writes from corrupting the certificate file.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file I/O fails.
    pub fn save(&self, path: &std::path::Path) -> Result<(), CertificateDeserializeError> {
        use std::io::Write;

        let json = self
            .to_json_serde()
            .map_err(CertificateDeserializeError::Json)?;
        let dir = path.parent().unwrap_or(std::path::Path::new("."));
        let tmp_path = path.with_extension("json.tmp");

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
            return Err(CertificateDeserializeError::Io(e));
        }

        // Best-effort directory fsync for durability.
        let _ = std::fs::File::open(dir).and_then(|f| f.sync_all());
        Ok(())
    }

    /// Load a certificate from a JSON file.
    ///
    /// Reads the file and delegates to [`from_json()`](Self::from_json)
    /// for deserialization and property name validation.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, JSON parsing fails,
    /// or property validation fails.
    pub fn load(path: &std::path::Path) -> Result<Self, CertificateDeserializeError> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json)
    }
}
