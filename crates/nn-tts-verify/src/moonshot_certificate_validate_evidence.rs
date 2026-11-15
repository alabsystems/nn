// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Evidence validation for moonshot certificates — level-artifact
//! consistency checks.
//!
//! Extracted from `moonshot_certificate_validate.rs` to stay under 500-line limit.

use super::{FindingSeverity, ValidationFinding};
use crate::moonshot::{MoonshotCertificate, VerificationLevel};

/// Validate that properties claiming proven-level have supporting evidence.
pub(super) fn validate_level_consistency(
    cert: &MoonshotCertificate,
    findings: &mut Vec<ValidationFinding>,
) {
    for prop in &cert.properties {
        let is_proven = matches!(
            prop.level,
            VerificationLevel::CrownProven | VerificationLevel::SmtProven
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
