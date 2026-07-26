// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! #1692 soundness fix tests for certificate checker (F1, F2, F4).
//!
//! Extracted from `certificate_checker_tests_soundness.rs` for 500-line compliance.
//! Tests the specific fixes from #1692: NaN-among-valid-elements detection (F1),
//! Concat structural validation (F2), and empty output bounds detection (F4).
//!
//! Part of #1685.

use super::checker_test_shared::{consistent_layer_bounds, sample_input_spec, sample_verification};
use super::*;
use crate::certificate::ProofCertificate;
use crate::certificate_types::LayerBoundRecord;
use crate::verify_types::PropMethod;

// ---------------------------------------------------------------------------
// #1692 soundness fixes — F1, F2, F4
// ---------------------------------------------------------------------------

/// F1 (#1692): A single NaN element among valid elements must be detected.
/// Before the fix, `if a < b { a } else { b }` in the reduce silently dropped
/// NaN (IEEE 754: `NaN < x` is false), so the NaN was lost before the
/// finiteness check on the reduced values.
#[test]
fn test_f1_nan_among_valid_elements_detected() {
    let mut result = sample_verification();
    result.output_lower = -5.0;
    result.output_upper = 10.0;
    result.output_width = 15.0;

    let mut bounds = consistent_layer_bounds();
    // Last layer has a NaN element alongside valid elements
    bounds[2].output_bounds = vec![(f32::NAN, 5.0), (1.0, 10.0)];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);
    let check = check_certificate(&cert, None, None);

    // The NaN element must be detected, not silently dropped
    let has_nan = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NanOutputBounds));
    assert!(
        has_nan,
        "NaN element among valid elements must be detected: {:?}",
        check.issues
    );

    // Should also report NonFiniteElement with the specific index
    let has_element = check.issues.iter().any(|i| {
        matches!(
            i,
            CheckIssue::NonFiniteElement {
                element_index: 0,
                ..
            }
        )
    });
    assert!(
        has_element,
        "should report specific non-finite element index: {:?}",
        check.issues
    );
}

/// F2 (#1692): Concat layers where source element counts differ from layer
/// input count are structurally valid — Concat naturally combines sources
/// with different element counts into a combined input.
/// The structural check verifies non-empty, finite source outputs only.
#[test]
fn test_f2_concat_mismatched_lengths_structural_valid() {
    let result = sample_verification();
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0)], // 1 element
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-3.0, 3.0)], // 1 element
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]),
        },
        // Concat layer: 2 sources with 1 element each → 2-element input
        // This is correct: Concat combines source outputs into combined input.
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Concat".to_string(),
            input_bounds: vec![(-5.0, 5.0), (-3.0, 3.0)], // 2 elements
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![0, 1]),
        },
    ];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);
    let check = check_certificate(&cert, None, None);

    // Structural check passes: non-empty, finite output bounds on both sources.
    // Length mismatch is expected for Concat — not a validation failure.
    let mismatch_count = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::MultiSourceLengthMismatch { .. }))
        .count();
    assert_eq!(
        mismatch_count, 0,
        "concat sources should not produce length mismatch issues: {:?}",
        check.issues
    );
}

/// F4 (#1692): Last layer with empty output_bounds must produce an
/// EmptyOutputBounds issue. Before the fix, `reduce()` returned `None`,
/// the `if let (Some, Some)` pattern didn't match, and the function
/// returned without pushing any issue.
#[test]
fn test_f4_empty_output_bounds_detected() {
    let result = sample_verification();
    let mut bounds = consistent_layer_bounds();
    // Last layer has empty output bounds
    bounds[2].output_bounds = vec![];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);
    let check = check_certificate(&cert, None, None);

    let has_empty = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::EmptyOutputBounds { .. }));
    assert!(
        has_empty,
        "empty output bounds must be detected: {:?}",
        check.issues
    );
}

/// F1 variant: Inf in output bounds element is also caught.
#[test]
fn test_f1_inf_among_valid_elements_detected() {
    let mut result = sample_verification();
    result.output_lower = 0.0;
    result.output_upper = 10.0;
    result.output_width = 10.0;

    let mut bounds = consistent_layer_bounds();
    // Last layer has an Inf element alongside valid elements
    bounds[2].output_bounds = vec![(0.0, 5.0), (f32::INFINITY, f32::NEG_INFINITY)];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);
    let check = check_certificate(&cert, None, None);

    let has_nan = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NanOutputBounds));
    assert!(
        has_nan,
        "Inf element among valid elements must be detected: {:?}",
        check.issues
    );
}
