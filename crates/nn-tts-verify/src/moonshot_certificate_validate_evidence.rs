// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Evidence validation for moonshot certificates — Kani consistency and
//! level-artifact consistency checks.
//!
//! Extracted from `moonshot_certificate_validate.rs` to stay under 500-line limit.

use std::path::Path;

use super::{FindingSeverity, ValidationFinding};
use crate::moonshot::{MoonshotCertificate, VerificationLevel};

/// Validate P7 (Kani) claims match workspace scan.
///
/// Returns the scanned harness count (or None if P7 doesn't claim KaniProven).
pub(super) fn validate_kani_consistency(
    cert: &MoonshotCertificate,
    repo_root: &Path,
    findings: &mut Vec<ValidationFinding>,
) -> Option<usize> {
    const P7_INDEX: usize = 6;
    if cert.properties.len() <= P7_INDEX {
        return None;
    }

    let p7 = &cert.properties[P7_INDEX];

    // Only validate Kani counts if P7 claims KaniProven and has bound_value
    if p7.level != VerificationLevel::KaniProven {
        return None;
    }

    let claimed_count = p7.bound_value.map(|v| v as usize);
    let claimed_total = p7.threshold.map(|v| v as usize);

    // Scan workspace for actual Kani harness count
    let crates_dir = repo_root.join("crates");
    if !crates_dir.is_dir() {
        findings.push(ValidationFinding {
            property_index: Some(P7_INDEX),
            severity: FindingSeverity::Warning,
            message: "Cannot validate Kani count: crates/ directory not found".to_string(),
        });
        return None;
    }

    let evidence =
        crate::moonshot::KaniVerificationEvidence::from_workspace_scan(&crates_dir, false);
    let actual_count = evidence.harnesses_total;

    if let Some(claimed) = claimed_count {
        // Allow tolerance: certificate may have been generated at a different commit
        // where the count was different. Flag if the actual is significantly lower.
        if actual_count < claimed {
            findings.push(ValidationFinding {
                property_index: Some(P7_INDEX),
                severity: FindingSeverity::Warning,
                message: format!(
                    "Kani harness count regressed: certificate claims {claimed} passed, workspace has {actual_count} total"
                ),
            });
        }
    }

    if let Some(total) = claimed_total {
        if actual_count < total {
            findings.push(ValidationFinding {
                property_index: Some(P7_INDEX),
                severity: FindingSeverity::Info,
                message: format!(
                    "Kani total: certificate claims {total}, workspace has {actual_count} (may differ by commit)"
                ),
            });
        }
    }

    Some(actual_count)
}

/// Validate that properties claiming proven-level have supporting evidence.
pub(super) fn validate_level_consistency(
    cert: &MoonshotCertificate,
    findings: &mut Vec<ValidationFinding>,
) {
    for prop in &cert.properties {
        let is_proven = matches!(
            prop.level,
            VerificationLevel::CrownProven
                | VerificationLevel::KaniProven
                | VerificationLevel::SmtProven
        );

        if is_proven && prop.proof_artifacts.is_empty() {
            findings.push(ValidationFinding {
                property_index: Some(prop.property_index),
                severity: FindingSeverity::Warning,
                message: format!("Claims {:?} but has no proof artifacts", prop.level),
            });
        }

        // bound_value and threshold should be set for CROWN/SMT proven properties
        if matches!(
            prop.level,
            VerificationLevel::CrownProven | VerificationLevel::SmtProven
        ) && prop.bound_value.is_none()
        {
            findings.push(ValidationFinding {
                property_index: Some(prop.property_index),
                severity: FindingSeverity::Info,
                message: format!("Claims {:?} but bound_value is None", prop.level),
            });
        }

        // Check for NaN/Inf in bound values
        if let Some(bv) = prop.bound_value {
            if !bv.is_finite() {
                findings.push(ValidationFinding {
                    property_index: Some(prop.property_index),
                    severity: FindingSeverity::Error,
                    message: format!("bound_value is non-finite: {bv}"),
                });
            }
        }

        if let Some(th) = prop.threshold {
            if !th.is_finite() {
                findings.push(ValidationFinding {
                    property_index: Some(prop.property_index),
                    severity: FindingSeverity::Error,
                    message: format!("threshold is non-finite: {th}"),
                });
            }
        }
    }
}
