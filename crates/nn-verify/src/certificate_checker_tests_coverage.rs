// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Coverage gap tests for certificate checker — proof_coverage audit.
//!
//! Each test exercises a previously-untested code path identified during
//! P1 proof_coverage phase.
//!
//! GAP 11 (HIGH): Mixed input_sources topology skips None layers in graph-aware mode.
//! GAP 9 (MEDIUM): Kani status enrichment through certificate_from_pipeline_enriched.
//! GAP 4 (MEDIUM): check_integrity signed cert tamper detection via checker.

use super::checker_test_shared::{sample_input_spec, sample_verification};
use super::*;
use crate::certificate::ProofCertificate;
use crate::certificate_types::LayerBoundRecord;
use crate::verify_types::PropMethod;

// ---------------------------------------------------------------------------
// GAP 11: Mixed input_sources topology — graph-aware skips None layers
// ---------------------------------------------------------------------------

/// When some layers have `input_sources` and others don't, graph-aware mode
/// is selected (because ANY record has input_sources). Layers with
/// `input_sources: None` are then skipped entirely (trace.rs line 153).
///
/// This means a certificate can have a broken sequential chain where the
/// None-sourced layer's input doesn't match the previous layer's output,
/// and the checker silently accepts it.
///
/// This test documents and exercises the gap.
#[test]
fn test_mixed_topology_none_layers_skipped_in_graph_aware_mode() {
    let result = sample_verification();

    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]), // Network input
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-5.0, 5.0)],
            output_bounds: vec![(0.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![0]), // Graph-aware: checked
        },
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-999.0, 999.0)], // DOES NOT match layer 1 output
            output_bounds: vec![(-3.0, 3.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: None, // Missing topology — graph-aware SKIPS this
        },
    ];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);
    let check = check_certificate(&cert, None, None);

    let trace_gaps: Vec<_> = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::LayerTraceGap { .. }))
        .collect();

    // Documents current behavior: the gap is NOT detected.
    // If the checker is enhanced to validate None layers in graph-aware mode,
    // update this assertion to expect trace_gaps.len() == 1.
    assert_eq!(
        trace_gaps.len(),
        0,
        "KNOWN GAP: mixed topology skips None layers in graph-aware mode. \
         Layer 2 has mismatched input_bounds but no LayerTraceGap is reported. \
         Issues: {:?}",
        check.issues
    );
}

/// Contrast test: fully graph-aware topology DOES detect the same gap.
#[test]
fn test_full_graph_aware_topology_detects_gap() {
    let result = sample_verification();

    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-5.0, 5.0)],
            output_bounds: vec![(0.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![0]),
        },
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-999.0, 999.0)], // DOES NOT match layer 1 output
            output_bounds: vec![(-3.0, 3.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![1]), // Graph-aware: CHECKED
        },
    ];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);
    let check = check_certificate(&cert, None, None);

    let trace_gaps: Vec<_> = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::LayerTraceGap { .. }))
        .collect();
    assert_eq!(
        trace_gaps.len(),
        1,
        "full graph-aware topology must detect the gap: {:?}",
        check.issues
    );
}

// ---------------------------------------------------------------------------
// GAP 9: Kani enrichment through certificate_from_pipeline_enriched
// ---------------------------------------------------------------------------

/// Verify that a valid kani_status.json file is loaded and its record
/// attached to the certificate through the pipeline enrichment path.
#[test]
fn test_kani_enrichment_pipeline_with_valid_file() {
    use crate::certificate::{certificate_from_pipeline_enriched, CertificateEnrichment};
    use crate::status::ParamInputRecord;

    let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let kani_path = repo_root.join("kani_status.json");
    if !kani_path.exists() {
        return;
    }

    let result = sample_verification();
    let variable_inputs = vec![ParamInputRecord::new(0, -1.0, 1.0)];

    let enrichment = CertificateEnrichment {
        kani_status_path: Some(kani_path),
        ..Default::default()
    };

    let cert =
        certificate_from_pipeline_enriched(&result, &variable_inputs, &[], None, Some(&enrichment));

    assert!(
        cert.kani_status.is_some(),
        "kani_status should be populated from kani_status.json for kernel 'snake'"
    );
    let kani = cert.kani_status.as_ref().unwrap();
    assert!(
        kani.harness_count >= 5,
        "snake kernel should have >= 5 Kani harnesses, got {}",
        kani.harness_count
    );
}

/// Enrichment with nonexistent kani_status.json path silently skips.
#[test]
fn test_kani_enrichment_pipeline_nonexistent_file_skipped() {
    use crate::certificate::{certificate_from_pipeline_enriched, CertificateEnrichment};
    use crate::status::ParamInputRecord;

    let result = sample_verification();
    let variable_inputs = vec![ParamInputRecord::new(0, -1.0, 1.0)];

    let enrichment = CertificateEnrichment {
        kani_status_path: Some(std::path::PathBuf::from("/nonexistent/kani_status.json")),
        ..Default::default()
    };

    let cert =
        certificate_from_pipeline_enriched(&result, &variable_inputs, &[], None, Some(&enrichment));

    assert!(
        cert.kani_status.is_none(),
        "kani_status should be None for nonexistent file"
    );
}

// ---------------------------------------------------------------------------
// GAP 4: check_integrity tampered content hash detection via checker
// ---------------------------------------------------------------------------

/// Signed cert checked without explicit key: content hash passes,
/// HMAC silently skipped (no env key).
#[test]
fn test_check_integrity_signed_cert_no_key_content_hash_passes() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(checker_test_shared::consistent_layer_bounds());

    let key = b"test-coverage-key";
    crate::certificate::integrity::sign_certificate(&mut cert, key).unwrap();

    let check = check_certificate(&cert, None, None);

    let integrity_errors: Vec<_> = check
        .issues
        .iter()
        .filter(|i| {
            matches!(
                i,
                CheckIssue::ContentHashMismatch { .. } | CheckIssue::SignatureInvalid { .. }
            )
        })
        .collect();
    assert!(
        integrity_errors.is_empty(),
        "signed cert should have no integrity errors: {integrity_errors:?}"
    );
}

/// Tampered signed cert: content hash mismatch detected by checker.
#[test]
fn test_check_integrity_tampered_content_hash_detected() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(checker_test_shared::consistent_layer_bounds());

    let key = b"test-coverage-key";
    crate::certificate::integrity::sign_certificate(&mut cert, key).unwrap();

    cert.output_width = 0.001;

    let check = check_certificate(&cert, None, None);

    let content_issues: Vec<_> = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::ContentHashMismatch { .. }))
        .collect();
    assert_eq!(
        content_issues.len(),
        1,
        "tampered cert should have ContentHashMismatch: {:?}",
        check.issues
    );
    assert!(!check.is_valid());
}
