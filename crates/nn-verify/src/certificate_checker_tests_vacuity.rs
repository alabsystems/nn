// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for VacuityAssessment and VacuousBounds detection.
//!
//! The vacuity assessment (certificate_checker.rs:128-154) classifies
//! certificates as non-vacuous when at least 50% of layers used CROWN
//! propagation AND the output width is below DEFAULT_VACUITY_THRESHOLD (10.0).
//!
//! This was the largest coverage gap in the certificate checker suite:
//! zero tests for VacuityAssessment, VacuousBounds, or is_valid() behavior
//! when VacuousBounds is the only issue.
//!
//! the existing `tests_soundness_trace` module:
//! ```rust
//! #[cfg(test)]
//! #[path = "certificate_checker_tests_vacuity.rs"]
//! mod tests_vacuity;
//! ```

use super::checker_test_shared::{sample_input_spec, sample_verification_with_bounds};
use super::*;
use crate::certificate::ProofCertificate;
use crate::certificate_types::LayerBoundRecord;
use crate::verify_types::{KernelVerification, PropMethod};

fn sample_verification_with_width(width: f32) -> KernelVerification {
    let half = width / 2.0;
    let mut v = sample_verification_with_bounds(-half, half);
    v.kernel_name = "vacuity_test".to_string();
    v
}

/// Build layer bounds with a specific CROWN/IBP split.
///
/// `crown_count` layers use CROWN, the rest use IBP. Total = `total_layers`.
/// All layers have consistent bounds chains for the checker.
fn layer_bounds_with_crown_split(total_layers: usize, crown_count: usize) -> Vec<LayerBoundRecord> {
    assert!(crown_count <= total_layers);
    (0..total_layers)
        .map(|i| {
            let method = if i < crown_count {
                PropMethod::Crown
            } else {
                PropMethod::Ibp
            };
            let input_bounds = if i == 0 {
                vec![(-10.0, 10.0)]
            } else {
                vec![(-5.0, 5.0)]
            };
            LayerBoundRecord {
                layer_index: i,
                layer_type: if i < crown_count {
                    "Linear".to_string()
                } else {
                    "ReLU".to_string()
                },
                input_bounds,
                output_bounds: vec![(-5.0, 5.0)],
                method,
                node_name: None,
                input_sources: if i == 0 {
                    Some(vec![])
                } else {
                    Some(vec![i - 1])
                },
            }
        })
        .collect()
}

fn build_cert(width: f32, bounds: Vec<LayerBoundRecord>) -> ProofCertificate {
    ProofCertificate::from_verification(&sample_verification_with_width(width), sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64)) // satisfy source_hash presence check
}

// ---------------------------------------------------------------------------
// VacuityAssessment population
// ---------------------------------------------------------------------------

/// All-CROWN certificate with narrow bounds → non-vacuous.
#[test]
fn test_all_crown_narrow_is_non_vacuous() {
    let bounds = layer_bounds_with_crown_split(4, 4); // 100% CROWN
    let cert = build_cert(5.0, bounds); // width < 10.0 threshold
    let result = check_certificate(&cert, None, None);

    let vacuity = result.vacuity.expect("vacuity should be populated");
    assert!(
        vacuity.is_non_vacuous,
        "all-CROWN + narrow width should be non-vacuous"
    );
    assert_eq!(vacuity.crown_layers, 4);
    assert_eq!(vacuity.ibp_layers, 0);
    assert!((vacuity.crown_coverage - 1.0).abs() < 1e-6);
    assert!((vacuity.output_width - 5.0).abs() < 1e-6);

    // No VacuousBounds issue
    let vacuous_count = result
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::VacuousBounds { .. }))
        .count();
    assert_eq!(
        vacuous_count, 0,
        "non-vacuous cert should not have VacuousBounds issue"
    );
}

/// All-IBP certificate → vacuous (0% CROWN coverage < 50% threshold).
#[test]
fn test_all_ibp_is_vacuous() {
    let bounds = layer_bounds_with_crown_split(4, 0); // 0% CROWN
    let cert = build_cert(5.0, bounds);
    let result = check_certificate(&cert, None, None);

    let vacuity = result.vacuity.expect("vacuity should be populated");
    assert!(!vacuity.is_non_vacuous, "all-IBP should be vacuous");
    assert_eq!(vacuity.crown_layers, 0);
    assert_eq!(vacuity.ibp_layers, 4);
    assert!((vacuity.crown_coverage - 0.0).abs() < 1e-6);

    let vacuous_count = result
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::VacuousBounds { .. }))
        .count();
    assert_eq!(
        vacuous_count, 1,
        "vacuous cert should have VacuousBounds issue"
    );
}

/// Exactly 50% CROWN with narrow width → non-vacuous (boundary: >= 0.5).
#[test]
fn test_half_crown_narrow_is_non_vacuous() {
    let bounds = layer_bounds_with_crown_split(4, 2); // 50% CROWN
    let cert = build_cert(5.0, bounds);
    let result = check_certificate(&cert, None, None);

    let vacuity = result.vacuity.expect("vacuity should be populated");
    assert!(
        vacuity.is_non_vacuous,
        "50% CROWN + narrow width should be non-vacuous (boundary is >=)"
    );
    assert_eq!(vacuity.crown_layers, 2);
    assert_eq!(vacuity.ibp_layers, 2);
    assert!((vacuity.crown_coverage - 0.5).abs() < 1e-6);
}

/// Below 50% CROWN → vacuous even with narrow width.
#[test]
fn test_below_half_crown_is_vacuous() {
    // 1 out of 4 = 25% CROWN
    let bounds = layer_bounds_with_crown_split(4, 1);
    let cert = build_cert(2.0, bounds); // very narrow, but insufficient CROWN
    let result = check_certificate(&cert, None, None);

    let vacuity = result.vacuity.expect("vacuity should be populated");
    assert!(
        !vacuity.is_non_vacuous,
        "25% CROWN should be vacuous regardless of width"
    );
    assert!((vacuity.crown_coverage - 0.25).abs() < 1e-6);
}

/// All-CROWN but wide output bounds (>= threshold) → vacuous.
#[test]
fn test_all_crown_wide_is_vacuous() {
    let bounds = layer_bounds_with_crown_split(4, 4); // 100% CROWN
    let cert = build_cert(15.0, bounds); // width > 10.0 threshold
    let result = check_certificate(&cert, None, None);

    let vacuity = result.vacuity.expect("vacuity should be populated");
    assert!(
        !vacuity.is_non_vacuous,
        "wide bounds (15.0 > 10.0 threshold) should be vacuous even with all-CROWN"
    );
    assert!((vacuity.output_width - 15.0).abs() < 1e-6);
}

/// Exactly at threshold (10.0) → vacuous (condition is `< 10.0`, not `<=`).
#[test]
fn test_exactly_at_threshold_is_vacuous() {
    let bounds = layer_bounds_with_crown_split(4, 4); // 100% CROWN
    let cert = build_cert(10.0, bounds); // width == 10.0 = threshold
    let result = check_certificate(&cert, None, None);

    let vacuity = result.vacuity.expect("vacuity should be populated");
    assert!(
        !vacuity.is_non_vacuous,
        "width == threshold (10.0) should be vacuous (strict < comparison)"
    );
}

/// Just below threshold (9.999) → non-vacuous.
#[test]
fn test_just_below_threshold_is_non_vacuous() {
    let bounds = layer_bounds_with_crown_split(4, 4); // 100% CROWN
    let cert = build_cert(9.999, bounds);
    let result = check_certificate(&cert, None, None);

    let vacuity = result.vacuity.expect("vacuity should be populated");
    assert!(
        vacuity.is_non_vacuous,
        "width just below threshold (9.999 < 10.0) should be non-vacuous"
    );
}

// ---------------------------------------------------------------------------
// VacuousBounds as informational (not validation failure)
// ---------------------------------------------------------------------------

/// VacuousBounds is the only issue → is_valid() returns true.
///
/// VacuousBounds is informational — it warns about bound quality but does
/// not invalidate the certificate. A certificate can be structurally valid
/// but have vacuously wide bounds.
#[test]
fn test_vacuous_bounds_is_valid_informational() {
    let bounds = layer_bounds_with_crown_split(4, 0); // all IBP → vacuous
                                                      // Width must be 10.0 to match layer output_bounds (-5.0, 5.0) and avoid OutputMismatch.
    let cert = build_cert(10.0, bounds);
    let result = check_certificate(&cert, None, None);

    // VacuousBounds should be present
    let has_vacuous = result
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::VacuousBounds { .. }));
    assert!(has_vacuous, "all-IBP cert should have VacuousBounds issue");

    // But the cert is still valid (VacuousBounds is informational)
    // Note: MissingHash for source_hash is also skipped since we provide it.
    // Filter to just VacuousBounds to check the is_valid logic.
    let non_vacuity_issues: Vec<_> = result
        .issues
        .iter()
        .filter(|i| !matches!(i, CheckIssue::VacuousBounds { .. }))
        .collect();

    // If the only remaining issues are NoLayerBounds-like or MissingHash,
    // the is_valid check should pass for VacuousBounds alone.
    // With our cert construction (source_hash provided, layer_bounds present),
    // VacuousBounds should be the only issue.
    assert!(
        result.is_valid(),
        "cert with only VacuousBounds issue should be valid, but found: {non_vacuity_issues:?}"
    );
}

// ---------------------------------------------------------------------------
// No layer bounds → vacuity is None
// ---------------------------------------------------------------------------

/// Certificate without layer bounds → vacuity assessment is None.
#[test]
fn test_no_layer_bounds_vacuity_is_none() {
    let cert = ProofCertificate::from_verification(
        &sample_verification_with_width(5.0),
        sample_input_spec(),
    )
    .with_source_hash("b".repeat(64));
    // No layer_bounds attached
    let result = check_certificate(&cert, None, None);

    assert!(
        result.vacuity.is_none(),
        "cert without layer bounds should have no vacuity assessment"
    );
}
