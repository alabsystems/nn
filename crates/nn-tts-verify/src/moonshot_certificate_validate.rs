// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Certificate validation — independently verify a [`MoonshotCertificate`]'s
//! claims against the current repository state.
//!
//! Performs structural, artifact, Kani, source-hash, level-consistency,
//! and bound-threshold-direction checks. See [`validate_certificate()`].

use std::path::Path;

use super::{MoonshotCertificate, VerificationLevel};
use crate::moonshot::PROPERTY_NAMES;

use super::source_hash::{compute_workspace_source_hash, is_valid_sha256_hex};

#[path = "moonshot_certificate_validate_bounds.rs"]
mod bounds;
use bounds::validate_bound_threshold_direction;

#[path = "moonshot_certificate_validate_evidence.rs"]
mod evidence;
use evidence::{validate_kani_consistency, validate_level_consistency};

/// Result of validating a [`MoonshotCertificate`] against repo state.
#[derive(Debug, Clone)]
pub struct CertificateValidation {
    /// Whether the certificate passes all validation checks.
    pub valid: bool,
    /// Individual validation findings (empty if valid).
    pub findings: Vec<ValidationFinding>,
    /// Number of proof artifacts that exist on disk.
    pub artifacts_found: usize,
    /// Number of proof artifacts referenced in the certificate.
    pub artifacts_total: usize,
    /// Kani harness count from workspace scan (None if scan was skipped).
    pub kani_scan_count: Option<usize>,
    /// Whether the certificate's source_hash matches the current codebase.
    /// `None` if hash verification was skipped (e.g., crates/ not found).
    pub source_hash_match: Option<bool>,
}

/// A single validation finding (warning or error).
#[derive(Debug, Clone)]
pub struct ValidationFinding {
    /// Which property (0-7) this finding relates to, or None for structural.
    pub property_index: Option<usize>,
    /// Severity of the finding.
    pub severity: FindingSeverity,
    /// Human-readable description.
    pub message: String,
}

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindingSeverity {
    /// Certificate is structurally invalid (e.g., wrong property count).
    Error,
    /// Certificate claim doesn't match repo state.
    Warning,
    /// Informational (e.g., property has no artifacts but level is None).
    Info,
}

impl CertificateValidation {
    /// Whether the certificate passed all validation checks.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Whether there are any error-level findings.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.findings
            .iter()
            .any(|f| f.severity == FindingSeverity::Error)
    }

    /// Whether there are any warning-level findings.
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        self.findings
            .iter()
            .any(|f| f.severity == FindingSeverity::Warning)
    }
}

impl std::fmt::Display for ValidationFinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = match self.severity {
            FindingSeverity::Error => "ERROR",
            FindingSeverity::Warning => "WARN",
            FindingSeverity::Info => "INFO",
        };
        if let Some(idx) = self.property_index {
            write!(f, "[{prefix}] P{}: {}", idx + 1, self.message)
        } else {
            write!(f, "[{prefix}] {}", self.message)
        }
    }
}

impl std::fmt::Display for CertificateValidation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.valid {
            writeln!(
                f,
                "Certificate VALID ({}/{} artifacts found)",
                self.artifacts_found, self.artifacts_total
            )?;
        } else {
            writeln!(f, "Certificate INVALID ({} findings)", self.findings.len())?;
        }
        match self.source_hash_match {
            Some(true) => writeln!(f, "  source_hash: MATCH")?,
            Some(false) => writeln!(f, "  source_hash: MISMATCH")?,
            None => writeln!(f, "  source_hash: SKIPPED")?,
        }
        for finding in &self.findings {
            writeln!(f, "  {finding}")?;
        }
        Ok(())
    }
}

/// Validate a certificate against the current repository state.
///
/// Performs structural validation, artifact existence checks, aggregate flag
/// consistency, and Kani harness count verification.
///
/// `repo_root` should be the path to the workspace root (the directory
/// containing `crates/`).
///
/// # Validation Checks
///
/// 1. **Structural:** 8 properties, indices 0-7, names match `PROPERTY_NAMES`.
/// 2. **Aggregate flags:** `all_at_least_partial` and `all_proven` match
///    per-property levels.
/// 3. **Artifact existence:** Each proof artifact file path exists on disk.
/// 4. **Kani consistency:** If P7 claims `KaniProven`, the workspace Kani
///    harness count matches the certificate's `bound_value`.
/// 5. **Level consistency:** Properties with no artifacts should not claim
///    CrownProven/KaniProven/SmtProven.
/// 6. **Bound-threshold direction:** Each property has a specific direction
///    in which `bound_value` must relate to `threshold` when proven.
#[must_use]
pub fn validate_certificate(cert: &MoonshotCertificate, repo_root: &Path) -> CertificateValidation {
    let mut findings = Vec::new();
    let mut artifacts_found = 0usize;
    let mut artifacts_total = 0usize;

    // 1. Structural validation
    validate_structure(cert, &mut findings);

    // 2. Aggregate flag consistency
    validate_aggregate_flags(cert, &mut findings);

    // 3. Artifact existence
    validate_artifacts(
        cert,
        repo_root,
        &mut findings,
        &mut artifacts_found,
        &mut artifacts_total,
    );

    // 4. Kani consistency
    let kani_scan_count = validate_kani_consistency(cert, repo_root, &mut findings);

    // 5. Level consistency
    validate_level_consistency(cert, &mut findings);

    // 6. Bound-threshold direction consistency
    validate_bound_threshold_direction(cert, &mut findings);

    // 7. Source hash verification
    let source_hash_match = validate_source_hash(cert, repo_root, &mut findings);

    let valid = !findings
        .iter()
        .any(|f| f.severity == FindingSeverity::Error);

    CertificateValidation {
        valid,
        findings,
        artifacts_found,
        artifacts_total,
        kani_scan_count,
        source_hash_match,
    }
}

/// Validate certificate structure: 8 properties with correct indices and names.
fn validate_structure(cert: &MoonshotCertificate, findings: &mut Vec<ValidationFinding>) {
    if cert.properties.len() != 8 {
        findings.push(ValidationFinding {
            property_index: None,
            severity: FindingSeverity::Error,
            message: format!("Expected 8 properties, found {}", cert.properties.len()),
        });
        return;
    }

    for (i, prop) in cert.properties.iter().enumerate() {
        if prop.property_index != i {
            findings.push(ValidationFinding {
                property_index: Some(i),
                severity: FindingSeverity::Error,
                message: format!(
                    "Property at position {} has index {} (expected {})",
                    i, prop.property_index, i
                ),
            });
        }

        if prop.property_name != PROPERTY_NAMES[i] {
            findings.push(ValidationFinding {
                property_index: Some(i),
                severity: FindingSeverity::Error,
                message: format!(
                    "Property name mismatch: expected '{}', found '{}'",
                    PROPERTY_NAMES[i], prop.property_name
                ),
            });
        }
    }

    if cert.model_name.is_empty() {
        findings.push(ValidationFinding {
            property_index: None,
            severity: FindingSeverity::Error,
            message: "model_name is empty".to_string(),
        });
    }

    if cert.source_hash.is_empty() {
        findings.push(ValidationFinding {
            property_index: None,
            severity: FindingSeverity::Warning,
            message:
                "source_hash is empty — certificate cannot be tied to a specific codebase version"
                    .to_string(),
        });
    }
}

/// Validate that aggregate flags match per-property levels.
fn validate_aggregate_flags(cert: &MoonshotCertificate, findings: &mut Vec<ValidationFinding>) {
    if cert.properties.len() != 8 {
        return; // Structure already flagged
    }

    let actual_all_partial = cert
        .properties
        .iter()
        .all(|p| p.level >= VerificationLevel::CrownPartial);

    if cert.all_at_least_partial != actual_all_partial {
        findings.push(ValidationFinding {
            property_index: None,
            severity: FindingSeverity::Error,
            message: format!(
                "all_at_least_partial is {} but computed value is {}",
                cert.all_at_least_partial, actual_all_partial
            ),
        });
    }

    let actual_all_proven = cert.properties.iter().all(|p| {
        matches!(
            p.level,
            VerificationLevel::CrownProven
                | VerificationLevel::KaniProven
                | VerificationLevel::SmtProven
        )
    });

    if cert.all_proven != actual_all_proven {
        findings.push(ValidationFinding {
            property_index: None,
            severity: FindingSeverity::Error,
            message: format!(
                "all_proven is {} but computed value is {}",
                cert.all_proven, actual_all_proven
            ),
        });
    }
}

/// Validate that proof artifact file paths exist on disk.
fn validate_artifacts(
    cert: &MoonshotCertificate,
    repo_root: &Path,
    findings: &mut Vec<ValidationFinding>,
    found: &mut usize,
    total: &mut usize,
) {
    for prop in &cert.properties {
        for artifact_path in &prop.proof_artifacts {
            *total += 1;
            let full_path = repo_root.join(artifact_path);
            if full_path.exists() {
                *found += 1;
            } else {
                findings.push(ValidationFinding {
                    property_index: Some(prop.property_index),
                    severity: FindingSeverity::Warning,
                    message: format!("Proof artifact not found: {artifact_path}"),
                });
            }
        }
    }
}

/// Validate the certificate's source_hash against a fresh workspace hash.
///
/// Returns `Some(true)` if hashes match, `Some(false)` if they mismatch,
/// or `None` if validation was skipped (empty hash, invalid format, or I/O error).
fn validate_source_hash(
    cert: &MoonshotCertificate,
    repo_root: &Path,
    findings: &mut Vec<ValidationFinding>,
) -> Option<bool> {
    // Skip if source_hash is empty (already flagged by validate_structure)
    if cert.source_hash.is_empty() {
        return None;
    }

    // Validate hash format
    if !is_valid_sha256_hex(&cert.source_hash) {
        findings.push(ValidationFinding {
            property_index: None,
            severity: FindingSeverity::Warning,
            message: format!(
                "source_hash is not a valid SHA-256 hex digest (expected 64 hex chars, got {} chars)",
                cert.source_hash.len()
            ),
        });
        return None;
    }

    // Recompute from workspace
    match compute_workspace_source_hash(repo_root) {
        Ok(actual_hash) => {
            if cert.source_hash == actual_hash {
                Some(true)
            } else {
                findings.push(ValidationFinding {
                    property_index: None,
                    severity: FindingSeverity::Warning,
                    message: format!(
                        "source_hash mismatch: certificate has {}, workspace computes {} \
                         (codebase changed since certificate was generated)",
                        &cert.source_hash[..16],
                        &actual_hash[..16],
                    ),
                });
                Some(false)
            }
        }
        Err(e) => {
            findings.push(ValidationFinding {
                property_index: None,
                severity: FindingSeverity::Info,
                message: format!("Could not recompute source hash: {e}"),
            });
            None
        }
    }
}
