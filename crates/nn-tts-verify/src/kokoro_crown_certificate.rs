// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro CROWN certificate integration for deployment packaging.
//!
//! Links moonshot properties P1-P8 with CROWN certificates and junction
//! contracts into a single deployable certificate bundle. Supports JSON
//! serialization for deployment packaging and structural validation.
//!
//! The [`KokoroCrownCertificate`] aggregates:
//! - Per-property CROWN verification results from [`MoonshotCrownBundle`]
//! - Junction contract verification from [`VerifiedJunctionContract`]
//! - The underlying [`MoonshotCertificate`] with all 8 property proofs
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_tts_verify::kokoro_crown_certificate::KokoroCrownCertificate;
//!
//! let cert = KokoroCrownCertificate::from_components(
//!     moonshot_cert,
//!     crown_bundle,
//!     verified_junctions,
//! );
//! let json = cert.to_json()?;
//! std::fs::write("kokoro_certificate.json", &json)?;
//! ```
//!
//! Part of #4254.

use serde::{Deserialize, Serialize};

use crate::kokoro_contracts::{all_contracts, JunctionContract, VerifiedJunctionContract};
use crate::moonshot::MoonshotCertificate;
use crate::moonshot_crown::{MoonshotCrownBundle, MoonshotPropertyResult};

/// Current version of the Kokoro CROWN certificate format.
pub const KOKORO_CERTIFICATE_VERSION: u32 = 1;

/// Number of moonshot properties (P1-P8).
const NUM_PROPERTIES: usize = 8;

/// Error type for certificate operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CertificateError {
    /// JSON serialization failed.
    #[error("JSON serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Certificate validation failed.
    #[error("certificate validation failed: {reason}")]
    Validation {
        /// Human-readable reason for the validation failure.
        reason: String,
    },
}

/// Per-property CROWN result stored in the certificate.
///
/// Serializable snapshot of a [`MoonshotPropertyResult`] for deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyCrownEntry {
    /// Property index (0-7, corresponding to P1-P8).
    pub property_index: usize,
    /// Property name.
    pub property_name: String,
    /// Whether the property is proven by CROWN bounds.
    pub proven: bool,
    /// Verification level achieved.
    pub level: String,
    /// Bound value from CROWN propagation.
    pub bound_value: f64,
    /// Threshold the bound must meet.
    pub threshold: f64,
    /// Whether the underlying CROWN was sound (not IBP fallback).
    pub is_sound: bool,
    /// Human-readable explanation.
    pub explanation: String,
}

impl PropertyCrownEntry {
    /// Create from a [`MoonshotPropertyResult`].
    #[must_use]
    pub fn from_result(result: &MoonshotPropertyResult) -> Self {
        Self {
            property_index: result.property_index,
            property_name: result.property_name.to_string(),
            proven: result.proven,
            level: format!("{}", result.level),
            bound_value: result.bound_value,
            threshold: result.threshold,
            is_sound: result.is_sound,
            explanation: result.explanation.clone(),
        }
    }
}

/// Junction contract status stored in the certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JunctionContractEntry {
    /// Junction identifier (e.g., "J2_F0").
    pub name: String,
    /// Zone crossing description.
    pub zone: String,
    /// Contract lower bound.
    pub lower: f64,
    /// Contract upper bound.
    pub upper: f64,
    /// Whether proven output bounds are contained within the contract.
    pub bounds_verified: bool,
    /// Whether a Lean4 composition proof is attached.
    pub has_composition_proof: bool,
    /// Lean4 theorem name (if present).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub composition_theorem_name: Option<String>,
}

impl JunctionContractEntry {
    /// Create from a [`VerifiedJunctionContract`].
    #[must_use]
    pub fn from_verified(vjc: &VerifiedJunctionContract) -> Self {
        Self {
            name: vjc.contract.name.to_string(),
            zone: vjc.contract.zone.to_string(),
            lower: vjc.contract.lower,
            upper: vjc.contract.upper,
            bounds_verified: vjc.bounds_verified,
            has_composition_proof: vjc.has_composition_proof(),
            composition_theorem_name: vjc.composition_theorem_name.clone(),
        }
    }

    /// Create from a [`JunctionContract`] without verification status.
    #[must_use]
    pub fn from_contract(contract: &JunctionContract) -> Self {
        Self {
            name: contract.name.to_string(),
            zone: contract.zone.to_string(),
            lower: contract.lower,
            upper: contract.upper,
            bounds_verified: false,
            has_composition_proof: false,
            composition_theorem_name: None,
        }
    }
}

/// Kokoro CROWN deployment certificate.
///
/// Aggregates all verification evidence for the Kokoro TTS pipeline into
/// a single serializable structure suitable for deployment packaging.
///
/// Contains:
/// - Schema version for forward compatibility
/// - Per-property CROWN verification results (P1-P8)
/// - Junction contract verification status (J2-J5)
/// - Pipeline-level soundness and validity flags
/// - Aggregate statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KokoroCrownCertificate {
    /// Certificate format version.
    pub version: u32,
    /// Model name (e.g., "dvoice-kokoro-v1").
    pub model_name: String,
    /// ISO 8601 date of certificate generation.
    pub generated_at: String,
    /// Moonshot property results from CROWN verification.
    pub properties: Vec<PropertyCrownEntry>,
    /// Junction contract verification status.
    pub junctions: Vec<JunctionContractEntry>,
    /// Dimension used for CROWN verification.
    pub verification_dim: usize,
    /// Whether all checked properties are proven.
    pub all_properties_proven: bool,
    /// Whether all junction contracts are verified.
    pub all_junctions_verified: bool,
    /// Whether the pipeline is sound (no IBP fallback).
    pub pipeline_is_sound: bool,
    /// Number of properties proven.
    pub proven_count: usize,
    /// Total number of properties checked.
    pub total_count: usize,
    /// Number of junctions verified.
    pub junctions_verified_count: usize,
    /// Total number of junctions.
    pub junctions_total_count: usize,
    /// Number of junctions with composition proofs.
    pub composition_proof_count: usize,
}

impl KokoroCrownCertificate {
    /// Create a certificate from CROWN verification components.
    ///
    /// Collects results from the moonshot CROWN bundle and verified junction
    /// contracts into a single deployable certificate.
    #[must_use]
    pub fn from_components(
        model_name: &str,
        bundle: &MoonshotCrownBundle,
        verified_junctions: &[VerifiedJunctionContract],
    ) -> Self {
        let properties: Vec<PropertyCrownEntry> = bundle
            .results
            .iter()
            .map(PropertyCrownEntry::from_result)
            .collect();

        let junctions: Vec<JunctionContractEntry> = verified_junctions
            .iter()
            .map(JunctionContractEntry::from_verified)
            .collect();

        let proven_count = properties.iter().filter(|p| p.proven).count();
        let total_count = properties.len();
        let junctions_verified_count = junctions.iter().filter(|j| j.bounds_verified).count();
        let junctions_total_count = junctions.len();
        let composition_proof_count = junctions.iter().filter(|j| j.has_composition_proof).count();

        let all_properties_proven = bundle.all_proven;
        let all_junctions_verified = junctions.iter().all(|j| j.bounds_verified);
        let pipeline_is_sound = bundle.pipeline_cert.is_sound;

        Self {
            version: KOKORO_CERTIFICATE_VERSION,
            model_name: model_name.to_string(),
            generated_at: current_iso8601(),
            properties,
            junctions,
            verification_dim: bundle.verification_dim,
            all_properties_proven,
            all_junctions_verified,
            pipeline_is_sound,
            proven_count,
            total_count,
            junctions_verified_count,
            junctions_total_count,
            composition_proof_count,
        }
    }

    /// Create a certificate from a moonshot certificate and CROWN bundle.
    ///
    /// Uses the default junction contracts (all 6 Kokoro junctions)
    /// without verification status.
    #[must_use]
    pub fn from_moonshot_and_bundle(
        moonshot: &MoonshotCertificate,
        bundle: &MoonshotCrownBundle,
    ) -> Self {
        let default_junctions: Vec<VerifiedJunctionContract> = all_contracts()
            .into_iter()
            .map(VerifiedJunctionContract::new)
            .collect();

        let mut cert = Self::from_components(&moonshot.model_name, bundle, &default_junctions);
        // Inherit verification_dim from moonshot if available.
        if let Some(dim) = moonshot.verification_dim {
            cert.verification_dim = dim;
        }
        cert
    }

    /// Serialize the certificate to pretty-printed JSON.
    ///
    /// # Errors
    ///
    /// Returns [`CertificateError::Serialization`] if JSON serialization fails.
    pub fn to_json(&self) -> Result<String, CertificateError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize a certificate from JSON.
    ///
    /// # Errors
    ///
    /// Returns [`CertificateError::Serialization`] if JSON parsing fails.
    pub fn from_json(json: &str) -> Result<Self, CertificateError> {
        Ok(serde_json::from_str(json)?)
    }

    /// Validate the certificate's structural integrity.
    ///
    /// Checks:
    /// - Property indices are in range [0, 7]
    /// - Bound values are finite (no NaN/Inf)
    /// - Junction bounds are non-inverted (lower <= upper)
    /// - Aggregate counts are consistent with per-entry data
    ///
    /// # Errors
    ///
    /// Returns [`CertificateError::Validation`] with a description of the
    /// first validation failure found.
    pub fn validate(&self) -> Result<(), CertificateError> {
        // Check property indices.
        for entry in &self.properties {
            if entry.property_index >= NUM_PROPERTIES {
                return Err(CertificateError::Validation {
                    reason: format!(
                        "property index {} out of range [0, {})",
                        entry.property_index, NUM_PROPERTIES
                    ),
                });
            }
            if !entry.bound_value.is_finite() {
                return Err(CertificateError::Validation {
                    reason: format!(
                        "property P{} has non-finite bound_value: {}",
                        entry.property_index + 1,
                        entry.bound_value
                    ),
                });
            }
            if !entry.threshold.is_finite() {
                return Err(CertificateError::Validation {
                    reason: format!(
                        "property P{} has non-finite threshold: {}",
                        entry.property_index + 1,
                        entry.threshold
                    ),
                });
            }
        }

        // Check junction bounds.
        for entry in &self.junctions {
            if !entry.lower.is_finite() || !entry.upper.is_finite() {
                return Err(CertificateError::Validation {
                    reason: format!(
                        "junction {} has non-finite bounds: [{}, {}]",
                        entry.name, entry.lower, entry.upper
                    ),
                });
            }
            if entry.lower > entry.upper {
                return Err(CertificateError::Validation {
                    reason: format!(
                        "junction {} has inverted bounds: {} > {}",
                        entry.name, entry.lower, entry.upper
                    ),
                });
            }
        }

        // Check aggregate count consistency.
        let actual_proven = self.properties.iter().filter(|p| p.proven).count();
        if actual_proven != self.proven_count {
            return Err(CertificateError::Validation {
                reason: format!(
                    "proven_count mismatch: field says {}, actual is {}",
                    self.proven_count, actual_proven
                ),
            });
        }

        let actual_total = self.properties.len();
        if actual_total != self.total_count {
            return Err(CertificateError::Validation {
                reason: format!(
                    "total_count mismatch: field says {}, actual is {}",
                    self.total_count, actual_total
                ),
            });
        }

        let actual_junctions_verified = self.junctions.iter().filter(|j| j.bounds_verified).count();
        if actual_junctions_verified != self.junctions_verified_count {
            return Err(CertificateError::Validation {
                reason: format!(
                    "junctions_verified_count mismatch: field says {}, actual is {}",
                    self.junctions_verified_count, actual_junctions_verified
                ),
            });
        }

        let actual_junctions_total = self.junctions.len();
        if actual_junctions_total != self.junctions_total_count {
            return Err(CertificateError::Validation {
                reason: format!(
                    "junctions_total_count mismatch: field says {}, actual is {}",
                    self.junctions_total_count, actual_junctions_total
                ),
            });
        }

        let actual_composition = self
            .junctions
            .iter()
            .filter(|j| j.has_composition_proof)
            .count();
        if actual_composition != self.composition_proof_count {
            return Err(CertificateError::Validation {
                reason: format!(
                    "composition_proof_count mismatch: field says {}, actual is {}",
                    self.composition_proof_count, actual_composition
                ),
            });
        }

        Ok(())
    }

    /// Whether the certificate represents a fully verified Kokoro pipeline.
    ///
    /// All properties proven, all junctions verified, and the pipeline is sound.
    #[must_use]
    pub fn is_fully_certified(&self) -> bool {
        self.all_properties_proven && self.all_junctions_verified && self.pipeline_is_sound
    }

    /// Generate a human-readable summary report.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut out = String::with_capacity(1024);
        out.push_str(&format!(
            "=== Kokoro CROWN Certificate v{} ===\n",
            self.version
        ));
        out.push_str(&format!("Model: {}\n", self.model_name));
        out.push_str(&format!("Generated: {}\n", self.generated_at));
        out.push_str(&format!("Verification dim: {}\n", self.verification_dim));
        out.push_str(&format!("Pipeline sound: {}\n\n", self.pipeline_is_sound));

        out.push_str(&format!(
            "Properties: {}/{} proven\n",
            self.proven_count, self.total_count
        ));
        for p in &self.properties {
            let mark = if p.proven { "PROVEN" } else { "UNPROVEN" };
            out.push_str(&format!(
                "  P{}: [{}] {} (bound={:.4}, threshold={:.4})\n",
                p.property_index + 1,
                mark,
                p.property_name,
                p.bound_value,
                p.threshold,
            ));
        }

        out.push_str(&format!(
            "\nJunctions: {}/{} verified ({} with composition proofs)\n",
            self.junctions_verified_count, self.junctions_total_count, self.composition_proof_count,
        ));
        for j in &self.junctions {
            let mark = if j.bounds_verified {
                "VERIFIED"
            } else {
                "UNVERIFIED"
            };
            let proof = if j.has_composition_proof {
                " [Lean4]"
            } else {
                ""
            };
            out.push_str(&format!(
                "  {}: [{}]{} {} [{:.2}, {:.2}]\n",
                j.name, mark, proof, j.zone, j.lower, j.upper,
            ));
        }

        out.push_str(&format!(
            "\nFully certified: {}\n",
            self.is_fully_certified()
        ));
        out
    }
}

/// Generate an ISO 8601 timestamp string.
fn current_iso8601() -> String {
    // Use a simple UTC timestamp. In production this would use chrono,
    // but we avoid the dependency by using a fixed-format approach.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Format as YYYY-MM-DDTHH:MM:SSZ (approximate from epoch seconds).
    let days = secs / 86400;
    let remaining = secs % 86400;
    let hours = remaining / 3600;
    let minutes = (remaining % 3600) / 60;
    let seconds = remaining % 60;

    // Compute year/month/day from days since epoch (1970-01-01).
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Civil calendar algorithm (Howard Hinnant).
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
#[path = "kokoro_crown_certificate_tests.rs"]
mod tests;
