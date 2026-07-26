// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Soundness gap tests for certificate checker.
//!
//! Each test documents a previously-known soundness gap and asserts the fix.
//! Extracted from `certificate_checker_tests.rs` — Part of #1678.
//!
//! #1692 F1/F2/F4 tests extracted to `certificate_checker_tests_soundness_1692.rs`.
//! Sequential trace tests extracted to `certificate_checker_tests_soundness_trace.rs`.

use super::checker_test_shared::{consistent_layer_bounds, sample_input_spec, sample_verification};
use super::*;
use crate::certificate::ProofCertificate;
use crate::certificate_types::LayerBoundRecord;
use crate::verify_types::PropMethod;

/// Certificates without source_hash report MissingHash.
/// weight_hash is optional v2 enrichment.
#[test]
fn test_gap_certificate_without_hashes_passes_valid() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds());
    // Deliberately do NOT attach weight_hash or source_hash

    let check = check_certificate(&cert, None, None);
    // Only source_hash absence is reported — weight_hash is optional
    assert!(
        !check.is_valid(),
        "certificates without source_hash must report issues"
    );
    let missing_hash_count = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::MissingHash { .. }))
        .count();
    assert_eq!(
        missing_hash_count, 1,
        "should report missing source_hash only (weight_hash is optional)"
    );
    // Verify it's specifically source_hash
    assert!(check.issues.iter().any(|i| matches!(
        i,
        CheckIssue::MissingHash { field } if field == "source_hash"
    )));
}

/// When a certificate HAS a hash but no file path is provided, the hash
/// cannot be verified against a file. This is a caller-side limitation,
/// not a certificate defect — the certificate has the hash.
#[test]
fn test_gap_hash_present_but_no_file_path_silently_skips() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_weight_hash("a".repeat(64)) // Hash is declared
        .with_source_hash("b".repeat(64)); // Hash is declared

    // Provide NO file paths — hash cannot be verified but MissingHash is not
    // reported because hashes ARE present.
    let check = check_certificate(&cert, None, None);
    assert!(
        check.is_valid(),
        "certificate with hashes but no file paths should pass: {:?}",
        check.issues
    );
}

/// Multi-source layers (Add, MulBinary, Concat) have structural checks only.
/// The layer's input_bounds represent the *combined* result of all sources
/// after NY bound propagation — NOT individual source contributions.
/// Element-wise containment is semantically wrong for binary ops.
/// We verify: (1) source output bounds are non-empty, (2) all elements finite.
#[test]
fn test_gap_multi_source_layer_structural_check() {
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
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-3.0, 3.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]),
        },
        // Add layer with TWO sources — input bounds [999,1000] differ from
        // individual source outputs, but this is correct NY behavior.
        // The combined input is the result of bound propagation, not a
        // concatenation of source outputs.
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Add".to_string(),
            input_bounds: vec![(999.0, 1000.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![0, 1]),
        },
    ];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);
    let check = check_certificate(&cert, None, None);

    // Structural check passes: both sources have non-empty, finite output bounds.
    // No LayerTraceGap — element-wise containment is not checked for multi-source.
    let gap_count = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::LayerTraceGap { .. }))
        .count();
    assert_eq!(
        gap_count, 0,
        "multi-source layers should not produce LayerTraceGap for valid finite bounds"
    );
}

/// Helper: build a multi-source Add layer with custom source-0 output bounds.
fn multi_source_bounds(source0_output: Vec<(f32, f32)>) -> Vec<LayerBoundRecord> {
    vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: source0_output,
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-3.0, 3.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Add".to_string(),
            input_bounds: vec![(0.0, 1.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![0, 1]),
        },
    ]
}

/// Multi-source layers with empty source output bounds produce EmptyOutputBounds.
#[test]
fn test_multi_source_empty_output_bounds_detected() {
    let cert = ProofCertificate::from_verification(&sample_verification(), sample_input_spec())
        .with_layer_bounds(multi_source_bounds(vec![]));
    let check = check_certificate(&cert, None, None);
    assert!(check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::EmptyOutputBounds { layer_index: 0 })));
}

/// Multi-source layers with non-finite source output bounds produce NonFiniteElement.
#[test]
fn test_multi_source_non_finite_detected() {
    let cert = ProofCertificate::from_verification(&sample_verification(), sample_input_spec())
        .with_layer_bounds(multi_source_bounds(vec![(f32::NAN, 5.0)]));
    let check = check_certificate(&cert, None, None);
    assert!(check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NonFiniteElement { layer_index: 0, .. })));
}

/// Dangling input_sources references produce DanglingSourceRef issues.
///
/// Source index must be < layer_index (otherwise ForwardReference fires first)
/// but not present in the bounds array. Use sparse layer indices (0, 5) with
/// layer 5 referencing non-existent source 3.
#[test]
fn test_gap_dangling_input_source_silently_skipped() {
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
        // Claims input from layer 3 which doesn't exist (but 3 < 5, so not
        // ForwardReference). This exercises the DanglingSourceRef path.
        LayerBoundRecord {
            layer_index: 5,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(999.0, 1000.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![3]), // Dangling: layer 3 not in bounds
        },
    ];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);
    let check = check_certificate(&cert, None, None);

    let dangling_count = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::DanglingSourceRef { .. }))
        .count();
    assert_eq!(
        dangling_count, 1,
        "dangling source ref must be detected: {:?}",
        check.issues
    );
    if let Some(CheckIssue::DanglingSourceRef {
        layer_index,
        dangling_source,
    }) = check
        .issues
        .iter()
        .find(|i| matches!(i, CheckIssue::DanglingSourceRef { .. }))
    {
        assert_eq!(*layer_index, 5);
        assert_eq!(*dangling_source, 3);
    }
}

/// Self-referencing input_sources (layer claims itself as source) are detected.
///
/// A layer cannot be its own input source — this would create a cycle in
/// the bound-propagation graph. The checker flags this as SelfReferenceSource.
#[test]
fn test_gap_self_referencing_input_source_detected() {
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
        // Layer 1 claims ITSELF as its input source — cycle
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-5.0, 5.0)],
            output_bounds: vec![(0.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![1]), // Self-reference!
        },
    ];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);
    let check = check_certificate(&cert, None, None);

    let self_ref_count = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::SelfReferenceSource { .. }))
        .count();
    assert_eq!(
        self_ref_count, 1,
        "self-referencing input source must be detected: {:?}",
        check.issues
    );
    assert!(check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::SelfReferenceSource { layer_index: 1 })));
}

/// FIX: Inverted bounds are now rejected regardless of is_finite flag.
/// validate() checks lower > upper with is_finite() guards for NaN safety.
#[test]
fn test_gap_non_finite_cert_allows_inverted_bounds() {
    let mut result = sample_verification();
    result.output_lower = 100.0;
    result.output_upper = -100.0; // Inverted!
    result.is_finite = false;

    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds());

    // FIX: Inverted bounds are now rejected regardless of is_finite
    assert!(
        cert.validate().is_err(),
        "inverted bounds must be rejected regardless of is_finite"
    );
}

/// FIX: NaN in output bounds now produces NanOutputBounds issue.
/// The NaN guard checks is_finite() before comparison to avoid
/// IEEE 754 pitfalls (NaN > eps is false).
#[test]
fn test_gap_nan_output_bounds_silently_pass_agreement() {
    let mut result = sample_verification();
    result.output_lower = f32::NAN;
    result.output_upper = f32::NAN;
    result.is_finite = false;
    result.output_width = f32::NAN;

    let mut bounds = consistent_layer_bounds();
    // Last layer also has NaN output
    bounds[2].output_bounds = vec![(f32::NAN, f32::NAN)];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);
    let check = check_certificate(&cert, None, None);

    // FIX: NaN is now detected — NanOutputBounds issue reported
    let has_nan = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NanOutputBounds));
    assert!(
        has_nan,
        "NaN in output bounds must be detected: {:?}",
        check.issues
    );
}

// ---------------------------------------------------------------------------
// Duplicate layer_index detection (#3020)
// ---------------------------------------------------------------------------

/// Duplicate layer_index values are detected and reported as DuplicateLayerIndex.
///
/// Previously (P3 gap): HashMap.collect() silently dropped earlier records for
/// duplicate keys. Now the checker explicitly detects duplicates during HashMap
/// construction and reports them.
#[test]
fn test_duplicate_layer_index_detected() {
    let result = sample_verification();
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-100.0, 100.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]),
        },
        // DUPLICATE layer_index=0
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-1.0, 1.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-1.0, 1.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![0]),
        },
    ];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);
    let check = check_certificate(&cert, None, None);

    // The checker now detects the duplicate layer_index=0.
    assert!(
        check
            .issues
            .iter()
            .any(|i| matches!(i, CheckIssue::DuplicateLayerIndex { layer_index: 0 })),
        "expected DuplicateLayerIndex for layer 0, got: {:?}",
        check.issues
    );
    // Certificate with duplicate layer_index should NOT be considered valid.
    assert!(
        !check.is_valid(),
        "certificate with duplicate layer_index should fail is_valid()"
    );
}

// ---------------------------------------------------------------------------
// First-layer input_bounds vs input_spec validation (#3020)
// ---------------------------------------------------------------------------

/// First layer's input_bounds are now validated against input_spec.
///
/// Previously (P3 gap): the checker never anchored the first layer's
/// input_bounds against the certificate's declared input range. A forged
/// certificate could claim arbitrary first-layer bounds.
#[test]
fn test_first_layer_input_bounds_mismatch_detected() {
    let result = sample_verification();
    // input_spec says [-10, 10]
    let input_spec = sample_input_spec();

    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            // INCONSISTENT: input_spec says [-10, 10] but first layer claims [-999, 999].
            input_bounds: vec![(-999.0, 999.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-5.0, 5.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![0]),
        },
    ];

    let cert = ProofCertificate::from_verification(&result, input_spec)
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    // The checker now detects the input_bounds vs input_spec mismatch.
    assert!(
        check.issues.iter().any(|i| matches!(
            i,
            CheckIssue::InputBoundsSpecMismatch { layer_index: 0, .. }
        )),
        "expected InputBoundsSpecMismatch for layer 0, got: {:?}",
        check.issues
    );
    // Certificate with mismatched first-layer bounds should NOT be valid.
    assert!(
        !check.is_valid(),
        "certificate with first-layer input_bounds mismatch should fail is_valid()"
    );
}

/// First layer with MATCHING input_bounds and input_spec passes cleanly.
#[test]
fn test_first_layer_input_bounds_matching_passes() {
    let result = sample_verification();
    let input_spec = sample_input_spec(); // [-10, 10]

    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)], // Matches input_spec
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-5.0, 5.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(vec![0]),
        },
    ];

    let cert = ProofCertificate::from_verification(&result, input_spec)
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    assert!(
        !check
            .issues
            .iter()
            .any(|i| matches!(i, CheckIssue::InputBoundsSpecMismatch { .. })),
        "matching first-layer input_bounds should NOT produce InputBoundsSpecMismatch: {:?}",
        check.issues
    );
}

// Sequential trace tests extracted to certificate_checker_tests_soundness_trace.rs
