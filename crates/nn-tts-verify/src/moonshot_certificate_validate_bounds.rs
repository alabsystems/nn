// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bound-threshold direction validation for certificate properties.
//!
//! Each of the 8 moonshot properties has a specific direction in which
//! `bound_value` must relate to `threshold` for the property to be satisfied.
//! This module catches certificates that claim proven-level but have
//! internally inconsistent bound/threshold values.

use super::{FindingSeverity, MoonshotCertificate, ValidationFinding, VerificationLevel};

/// Per-property bound-threshold direction.
///
/// Derived from the `moonshot_crown` check functions:
/// - P1 (Non-silent):        bound > threshold   (max |output| must exceed RMS threshold)
/// - P2 (Non-clipping):      bound <= threshold   (worst output within [-1,1])
/// - P3 (Intelligible):      bound < threshold    (range ratio below max)
/// - P4 (Speaker-consistent): bound < threshold   (L2 distance below ε)
/// - P5 (Temporally bounded): bound <= threshold  (worst-case time within limit)
/// - P6 (Streaming-safe):    bound <= threshold   (click bound within threshold)
/// - P7 (Memory-safe):       bound >= threshold   (harnesses passed >= total)
/// - P8 (Correct impl):      bound >= threshold   (kernels proven >= total)
#[derive(Debug, Clone, Copy)]
enum BoundDirection {
    /// Property satisfied when bound > threshold (P1: non-silence).
    GreaterThan,
    /// Property satisfied when bound <= threshold (P2, P5, P6).
    LessOrEqual,
    /// Property satisfied when bound < threshold (P3, P4).
    StrictLess,
    /// Property satisfied when bound >= threshold (P7, P8).
    GreaterOrEqual,
}

/// Maps property index (0-7) to its bound-threshold direction.
const PROPERTY_DIRECTIONS: [BoundDirection; 8] = [
    BoundDirection::GreaterThan,    // P1: Non-silent
    BoundDirection::LessOrEqual,    // P2: Non-clipping
    BoundDirection::StrictLess,     // P3: Intelligible
    BoundDirection::StrictLess,     // P4: Speaker-consistent
    BoundDirection::LessOrEqual,    // P5: Temporally bounded
    BoundDirection::LessOrEqual,    // P6: Streaming-safe
    BoundDirection::GreaterOrEqual, // P7: Memory-safe (Kani)
    BoundDirection::GreaterOrEqual, // P8: Correct impl (ay)
];

/// Validate that proven properties have bound/threshold values in the correct direction.
///
/// Only checks properties at `CrownPartial` or above that have both
/// `bound_value` and `threshold` set. Non-finite values are caught
/// by `validate_level_consistency` and skipped here.
pub(super) fn validate_bound_threshold_direction(
    cert: &MoonshotCertificate,
    findings: &mut Vec<ValidationFinding>,
) {
    for prop in &cert.properties {
        // Only check properties claiming at least partial verification
        if prop.level < VerificationLevel::CrownPartial {
            continue;
        }

        let (bv, th) = match (prop.bound_value, prop.threshold) {
            (Some(b), Some(t)) => (b, t),
            _ => continue, // Missing values flagged elsewhere
        };

        // Skip non-finite — caught by validate_level_consistency
        if !bv.is_finite() || !th.is_finite() {
            continue;
        }

        if prop.property_index >= PROPERTY_DIRECTIONS.len() {
            continue;
        }

        let direction = PROPERTY_DIRECTIONS[prop.property_index];
        let violated = match direction {
            BoundDirection::GreaterThan => bv <= th,
            BoundDirection::LessOrEqual => bv > th,
            BoundDirection::StrictLess => bv >= th,
            BoundDirection::GreaterOrEqual => bv < th,
        };

        if violated {
            let dir_desc = match direction {
                BoundDirection::GreaterThan => "bound > threshold",
                BoundDirection::LessOrEqual => "bound ≤ threshold",
                BoundDirection::StrictLess => "bound < threshold",
                BoundDirection::GreaterOrEqual => "bound ≥ threshold",
            };
            findings.push(ValidationFinding {
                property_index: Some(prop.property_index),
                severity: FindingSeverity::Warning,
                message: format!(
                    "bound/threshold direction violated: bound_value={bv}, threshold={th}, \
                     expected {dir_desc}",
                ),
            });
        }
    }
}
