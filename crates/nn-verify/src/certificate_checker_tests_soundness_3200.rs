// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Soundness gap tests for #3200: verify_model! vacuity threshold,
//! certify_model fusion failure recording, layer bounds failure recording.

use super::checker_test_shared::{sample_input_spec_with_bounds, sample_verification_with_bounds};
use super::*;
use crate::certificate::ProofCertificate;
use crate::certificate_types::LayerBoundRecord;
use crate::status::InputBoundsRecord;
use crate::verify_types::{KernelVerification, OutputTensorBounds, PropMethod};

fn sample_verification_wide() -> KernelVerification {
    let mut v = sample_verification_with_bounds(-1000.0, 1000.0);
    v.kernel_name = "linear".to_string();
    v
}

fn sample_input_spec() -> InputBoundsRecord {
    sample_input_spec_with_bounds(-1.0, 1.0, vec![])
}

// ---------------------------------------------------------------------------
// F1: DEFAULT_VACUITY_THRESHOLD is meaningful, not f32::MAX
// ---------------------------------------------------------------------------

/// The vacuity threshold must be a reasonable finite value, not f32::MAX.
#[test]
fn test_f1_vacuity_threshold_is_reasonable() {
    assert!(DEFAULT_VACUITY_THRESHOLD.is_finite());
    assert!(DEFAULT_VACUITY_THRESHOLD > 0.0);
    assert!(
        DEFAULT_VACUITY_THRESHOLD < 100.0,
        "threshold {DEFAULT_VACUITY_THRESHOLD} is unreasonably large",
    );
    assert!(
        DEFAULT_VACUITY_THRESHOLD < f32::MAX,
        "threshold must not be f32::MAX (no-op check)",
    );
}

/// A certificate with vacuously wide bounds (width >> threshold) must
/// report VacuousBounds in its check results.
///
/// Uses PropMethod::Crown for the layer bound so CROWN coverage = 100%
/// (passing the coverage gate). This isolates the WIDTH threshold as the
/// sole trigger for VacuousBounds, ensuring this test actually validates F1.
#[test]
fn test_f1_vacuous_bounds_detected_in_certificate() {
    let result = sample_verification_wide();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(vec![LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-1.0, 1.0)],
            output_bounds: vec![(-1000.0, 1000.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(vec![]),
        }])
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    let has_vacuous = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::VacuousBounds { .. }));
    assert!(
        has_vacuous,
        "certificate with width {} must report VacuousBounds, got: {:?}",
        result.output_width, check.issues,
    );
}

/// Width exactly at the threshold boundary should NOT be vacuous.
#[test]
fn test_f1_boundary_width_not_vacuous() {
    let mut result = sample_verification_wide();
    // Width just below threshold
    let half = DEFAULT_VACUITY_THRESHOLD / 2.0;
    result.output_lower = -half;
    result.output_upper = half;
    result.output_width = DEFAULT_VACUITY_THRESHOLD - 0.01;
    result.output_tensor = Some(OutputTensorBounds {
        lower: vec![-half],
        upper: vec![half],
        shape: vec![1],
        finite_mask: vec![true],
    });

    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(vec![LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-1.0, 1.0)],
            output_bounds: vec![(-half, half)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(vec![]),
        }])
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    let has_vacuous = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::VacuousBounds { .. }));
    assert!(
        !has_vacuous,
        "width {} < threshold {} should not be vacuous, got: {:?}",
        result.output_width, DEFAULT_VACUITY_THRESHOLD, check.issues,
    );
}

// ---------------------------------------------------------------------------
// F2/F3: CertifyResult diagnostic fields exist (structural)
// ---------------------------------------------------------------------------

/// CertifyResult must expose layer_bounds_warning and fusion_warning fields.
/// This is a compile-time structural test — verifying the fields exist
/// and have the expected types.
#[test]
fn test_f2_f3_certify_result_has_diagnostic_fields() {
    // Structural type check: if CertifyResult loses these fields,
    // this test fails to compile.
    fn _assert_fields(r: &crate::CertifyResult) {
        let _lbw: &Option<String> = &r.layer_bounds_warning;
        let _fw: &Option<String> = &r.fusion_warning;
    }
}
