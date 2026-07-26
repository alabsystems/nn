// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for independent certificate checker.

use super::checker_test_shared::{consistent_layer_bounds, sample_input_spec, sample_verification};
use super::*;
use crate::certificate::ProofCertificate;
use crate::certificate_types::{compute_bytes_hash, LayerBoundRecord};
use crate::verify_types::PropMethod;

// ---------------------------------------------------------------------------
// Valid certificate passes all checks
// ---------------------------------------------------------------------------

#[test]
fn test_check_valid_certificate_with_trace() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_weight_hash("a".repeat(64))
        .with_source_hash("b".repeat(64));

    let check = check_certificate(&cert, None, None);
    assert!(check.is_valid(), "issues: {:?}", check.issues);
    assert_eq!(check.kernel_name, "snake");
}

// ---------------------------------------------------------------------------
// Layer trace gap detection
// ---------------------------------------------------------------------------

#[test]
fn test_check_detects_layer_trace_gap() {
    let result = sample_verification();
    let mut bounds = consistent_layer_bounds();
    // Break the chain: layer 0 output != layer 1 input
    bounds[1].input_bounds = vec![(-3.0, 3.0)]; // was (-5.0, 5.0)

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);

    let check = check_certificate(&cert, None, None);
    assert!(!check.is_valid());
    let gap_issues: Vec<_> = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::LayerTraceGap { .. }))
        .collect();
    assert_eq!(gap_issues.len(), 1);
    if let CheckIssue::LayerTraceGap { layer_index, .. } = &gap_issues[0] {
        assert_eq!(*layer_index, 0);
    }
}

#[test]
fn test_check_detects_multiple_gaps() {
    let result = sample_verification();
    let mut bounds = consistent_layer_bounds();
    // Break both chains
    bounds[1].input_bounds = vec![(-3.0, 3.0)];
    bounds[2].input_bounds = vec![(1.0, 4.0)];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);

    let check = check_certificate(&cert, None, None);
    let gap_count = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::LayerTraceGap { .. }))
        .count();
    assert_eq!(gap_count, 2);
}

// ---------------------------------------------------------------------------
// Output agreement
// ---------------------------------------------------------------------------

#[test]
fn test_check_detects_output_mismatch() {
    let result = sample_verification();
    // Certificate claims [-5, 5] but trace last layer outputs [0, 3]
    // (output_lower/output_upper already set to -5.0/5.0 by sample_verification)

    let mut bounds = consistent_layer_bounds();
    // Change last layer output to mismatch certificate's claimed [-5, 5]
    bounds[2].output_bounds = vec![(0.0, 3.0)];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);

    let check = check_certificate(&cert, None, None);
    let mismatch = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::OutputMismatch { .. }));
    assert!(
        mismatch,
        "should detect output mismatch: {:?}",
        check.issues
    );
}

#[test]
fn test_check_output_agreement_within_tolerance() {
    let mut result = sample_verification();
    result.output_lower = -5.0;
    result.output_upper = 5.0;

    let mut bounds = consistent_layer_bounds();
    // Last layer output within epsilon of certificate
    bounds[2].output_bounds = vec![(-5.0 + 1e-7, 5.0 - 1e-7)];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);

    let check = check_certificate(&cert, None, None);
    let has_mismatch = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::OutputMismatch { .. }));
    assert!(
        !has_mismatch,
        "small epsilon should pass: {:?}",
        check.issues
    );
}

// ---------------------------------------------------------------------------
// Multi-element output agreement (envelope fold)
// ---------------------------------------------------------------------------

/// Multi-element output bounds: envelope min(lower)/max(upper) matches cert.
///
/// The agreement checker reduces multi-element bounds to their envelope
/// (agreement.rs:49-58). This test exercises that fold with all-finite elements
/// where the envelope matches the certificate's scalar output bounds.
#[test]
fn test_check_multi_element_output_agreement_match() {
    let mut result = sample_verification();
    // Envelope of [(-3, 2), (-5, 5)] is lower=-5, upper=5.
    result.output_lower = -5.0;
    result.output_upper = 5.0;
    result.output_width = 10.0;

    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0), (-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0), (-3.0, 3.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-5.0, 5.0), (-3.0, 3.0)],
            output_bounds: vec![(-3.0, 2.0), (-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![0]),
        },
    ];

    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    assert!(
        !check
            .issues
            .iter()
            .any(|i| matches!(i, CheckIssue::OutputMismatch { .. })),
        "multi-element envelope matching cert should not produce OutputMismatch: {:?}",
        check.issues
    );
}

/// Multi-element output bounds: envelope differs from cert → OutputMismatch.
///
/// Last layer outputs [(-3, 2), (-1, 4)]: envelope is (-3, 4).
/// Certificate claims (-5, 5). This must produce OutputMismatch.
#[test]
fn test_check_multi_element_output_agreement_mismatch() {
    let mut result = sample_verification();
    result.output_lower = -5.0;
    result.output_upper = 5.0;
    result.output_width = 10.0;

    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0), (-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0), (-3.0, 3.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-5.0, 5.0), (-3.0, 3.0)],
            // Envelope: lower=min(-3,-1)=-3, upper=max(2,4)=4
            // Certificate claims (-5, 5) → mismatch
            output_bounds: vec![(-3.0, 2.0), (-1.0, 4.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![0]),
        },
    ];

    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    let mismatch = check
        .issues
        .iter()
        .find(|i| matches!(i, CheckIssue::OutputMismatch { .. }));
    assert!(
        mismatch.is_some(),
        "multi-element envelope != cert bounds should produce OutputMismatch: {:?}",
        check.issues
    );
    if let Some(CheckIssue::OutputMismatch {
        certificate_lower,
        certificate_upper,
        trace_lower,
        trace_upper,
    }) = mismatch
    {
        assert_eq!(*certificate_lower, -5.0);
        assert_eq!(*certificate_upper, 5.0);
        assert!(
            (*trace_lower - (-3.0)).abs() < 1e-6,
            "trace_lower should be -3.0"
        );
        assert!(
            (*trace_upper - 4.0).abs() < 1e-6,
            "trace_upper should be 4.0"
        );
    }
}

// ---------------------------------------------------------------------------
// No layer bounds
// ---------------------------------------------------------------------------

#[test]
fn test_check_no_layer_bounds_reports_issue() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec());

    let check = check_certificate(&cert, None, None);
    let has_no_bounds = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NoLayerBounds));
    assert!(has_no_bounds);
}

// ---------------------------------------------------------------------------
// Hash verification
// ---------------------------------------------------------------------------

#[test]
fn test_check_weight_hash_matches() {
    let dir = std::env::temp_dir().join(format!("nn_checker_test_wh_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let weight_path = dir.join("weights.bin");
    let data = b"model weight data";
    std::fs::write(&weight_path, data).expect("write");
    let hash = compute_bytes_hash(data);

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_weight_hash(hash)
        .with_layer_bounds(consistent_layer_bounds());

    let check = check_certificate(&cert, Some(&weight_path), None);
    let hash_issues: Vec<_> = check
        .issues
        .iter()
        .filter(|i| {
            matches!(
                i,
                CheckIssue::WeightHashMismatch { .. } | CheckIssue::HashFileError { .. }
            )
        })
        .collect();
    assert!(
        hash_issues.is_empty(),
        "hash should match: {hash_issues:?}"
    );

    let _ = std::fs::remove_file(&weight_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_check_weight_hash_mismatch() {
    let dir = std::env::temp_dir().join(format!("nn_checker_test_wh_mis_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let weight_path = dir.join("weights.bin");
    std::fs::write(&weight_path, b"actual data").expect("write");

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_weight_hash("a".repeat(64)) // Wrong hash
        .with_layer_bounds(consistent_layer_bounds());

    let check = check_certificate(&cert, Some(&weight_path), None);
    let has_mismatch = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::WeightHashMismatch { .. }));
    assert!(has_mismatch);

    let _ = std::fs::remove_file(&weight_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_check_source_hash_file_error() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_source_hash("b".repeat(64))
        .with_layer_bounds(consistent_layer_bounds());

    // Provide a nonexistent path — should get HashFileError, not crash
    let check = check_certificate(&cert, None, Some(Path::new("/nonexistent/source.rs")));
    let has_file_err = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::HashFileError { .. }));
    assert!(has_file_err);
}

// ---------------------------------------------------------------------------
// Structural validation errors forwarded
// ---------------------------------------------------------------------------

#[test]
fn test_check_structural_error_forwarded() {
    let mut result = sample_verification();
    result.kernel_name = String::new(); // Invalid

    let cert = ProofCertificate::from_verification(&result, sample_input_spec());
    let check = check_certificate(&cert, None, None);
    let has_structural = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::StructuralError { .. }));
    assert!(has_structural);
}

// ---------------------------------------------------------------------------
// SmtProofMissing fails is_valid() — unverifiable "Proven" claim (#3221)
// ---------------------------------------------------------------------------

#[test]
fn test_smt_proof_missing_fails_validity() {
    let result = CheckResult {
        kernel_name: "test_kernel".to_string(),
        issues: vec![CheckIssue::SmtProofMissing],
        vacuity: None,
    };
    assert!(
        !result.is_valid(),
        "SmtProofMissing must fail is_valid() — Proven claim without proof artifact is unverifiable"
    );
}

#[test]
fn test_smt_proof_invalid_fails_validity() {
    let result = CheckResult {
        kernel_name: "test_kernel".to_string(),
        issues: vec![CheckIssue::SmtProofInvalid],
        vacuity: None,
    };
    assert!(!result.is_valid(), "SmtProofInvalid should fail is_valid()");
}
