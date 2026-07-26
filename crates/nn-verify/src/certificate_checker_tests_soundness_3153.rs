// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Soundness gap tests for #3153: is_infeasible bypass, inverted element
//! bounds, input_spec validation, and forward reference detection.

use super::checker_test_shared::{consistent_layer_bounds, sample_input_spec, sample_verification};
use super::*;
use crate::certificate::ProofCertificate;
use crate::certificate_types::LayerBoundRecord;
use crate::status::{InputBoundsRecord, OutputBoundsRecord, ParamInputRecord};
use crate::verify_types::PropMethod;

// ---------------------------------------------------------------------------
// F1: is_infeasible sentinel bounds bypass
// ---------------------------------------------------------------------------

/// Infeasible certificates must be rejected by the checker.
/// Prior bug: (0.0, 0.0) sentinels passed all checks when is_infeasible=true.
#[test]
fn test_f1_infeasible_cert_rejected() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_source_hash("a".repeat(64));
    // Simulate infeasible: (0.0, 0.0) sentinels with is_infeasible=true.
    cert.output_bounds = OutputBoundsRecord::new(0.0, 0.0);
    cert.output_bounds.is_infeasible = true;
    cert.output_width = 0.0;

    let check = check_certificate(&cert, None, None);
    assert!(
        !check.is_valid(),
        "infeasible certificate must not pass is_valid(): {:?}",
        check.issues
    );
    assert!(
        check
            .issues
            .iter()
            .any(|i| matches!(i, CheckIssue::InfeasibleBounds)),
        "must report InfeasibleBounds issue: {:?}",
        check.issues
    );
}

/// Non-infeasible certificates with (0.0, 0.0) bounds do NOT trigger
/// InfeasibleBounds — they are legitimate tight bounds.
#[test]
fn test_f1_non_infeasible_zero_bounds_ok() {
    let mut result = sample_verification();
    result.output_lower = 0.0;
    result.output_upper = 0.0;
    result.output_width = 0.0;

    let mut bounds = consistent_layer_bounds();
    bounds[2].output_bounds = vec![(0.0, 0.0)];

    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    assert!(
        !check
            .issues
            .iter()
            .any(|i| matches!(i, CheckIssue::InfeasibleBounds)),
        "non-infeasible (0,0) should not trigger InfeasibleBounds: {:?}",
        check.issues
    );
}

// ---------------------------------------------------------------------------
// F2: Per-element inverted bounds in layer trace
// ---------------------------------------------------------------------------

/// Layer output bounds with lo > hi (inverted) must be detected.
#[test]
fn test_f2_inverted_element_bounds_detected() {
    let result = sample_verification();
    let mut bounds = consistent_layer_bounds();
    // Corrupt layer 1: element has inverted bounds (5.0 > 0.0).
    bounds[1].output_bounds = vec![(5.0, 0.0)];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);
    let check = check_certificate(&cert, None, None);
    assert!(
        check.issues.iter().any(|i| matches!(
            i,
            CheckIssue::InvertedElementBounds {
                layer_index: 1,
                element_index: 0,
                ..
            }
        )),
        "inverted element bounds must be detected: {:?}",
        check.issues
    );
    assert!(!check.is_valid());
}

/// NaN element bounds should NOT trigger InvertedElementBounds (NaN comparison
/// returns false). They are caught by NonFiniteElement in the agreement checker.
#[test]
fn test_f2_nan_not_treated_as_inverted() {
    let result = sample_verification();
    let mut bounds = consistent_layer_bounds();
    bounds[1].output_bounds = vec![(f32::NAN, 5.0)];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);
    let check = check_certificate(&cert, None, None);
    assert!(
        !check
            .issues
            .iter()
            .any(|i| matches!(i, CheckIssue::InvertedElementBounds { .. })),
        "NaN should not trigger InvertedElementBounds: {:?}",
        check.issues
    );
}

/// Valid bounds (lo <= hi) should not trigger InvertedElementBounds.
#[test]
fn test_f2_valid_bounds_no_false_positive() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    assert!(
        !check
            .issues
            .iter()
            .any(|i| matches!(i, CheckIssue::InvertedElementBounds { .. })),
        "valid bounds should not trigger InvertedElementBounds: {:?}",
        check.issues
    );
}

// ---------------------------------------------------------------------------
// F3: Input specification validation
// ---------------------------------------------------------------------------

/// Empty variable_inputs means nothing was verified.
#[test]
fn test_f3_empty_input_spec_rejected() {
    let result = sample_verification();
    let input_spec = InputBoundsRecord {
        variable_inputs: vec![],
        constant_params: vec![1.0],
        input_shape: None,
        input_range: None,
    };
    let cert = ProofCertificate::from_verification(&result, input_spec)
        .with_layer_bounds(consistent_layer_bounds())
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    assert!(!check.is_valid());
    assert!(
        check
            .issues
            .iter()
            .any(|i| matches!(i, CheckIssue::InvalidInputSpec { .. })),
        "empty variable_inputs must be reported: {:?}",
        check.issues
    );
}

/// NaN in input bounds makes the proof vacuously true.
#[test]
fn test_f3_nan_input_bounds_rejected() {
    let input_spec = InputBoundsRecord {
        variable_inputs: vec![ParamInputRecord {
            param_index: 0,
            lower: f32::NAN,
            upper: 10.0,
        }],
        constant_params: vec![],
        input_shape: Some(vec![1]),
        input_range: None,
    };
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, input_spec)
        .with_layer_bounds(consistent_layer_bounds())
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    assert!(!check.is_valid());
    assert!(check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::InvalidInputSpec { .. })));
}

/// Inverted input bounds (lower > upper) make the proof vacuously true.
#[test]
fn test_f3_inverted_input_bounds_rejected() {
    let input_spec = InputBoundsRecord {
        variable_inputs: vec![ParamInputRecord {
            param_index: 0,
            lower: 10.0,
            upper: -10.0, // Inverted!
        }],
        constant_params: vec![],
        input_shape: Some(vec![1]),
        input_range: None,
    };
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, input_spec)
        .with_layer_bounds(consistent_layer_bounds())
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    assert!(!check.is_valid());
    assert!(check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::InvalidInputSpec { .. })));
}

/// Valid input spec should not trigger InvalidInputSpec.
#[test]
fn test_f3_valid_input_spec_ok() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    assert!(
        !check
            .issues
            .iter()
            .any(|i| matches!(i, CheckIssue::InvalidInputSpec { .. })),
        "valid input spec should not produce InvalidInputSpec: {:?}",
        check.issues
    );
}

// ---------------------------------------------------------------------------
// F4: Forward reference detection in graph-aware trace
// ---------------------------------------------------------------------------

/// Layer claiming input from a higher-index layer is a forward reference.
#[test]
fn test_f4_forward_reference_detected() {
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
        // Layer 1 claims input from layer 5 which doesn't exist yet
        // (even if it existed, forward reference is invalid).
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-5.0, 5.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![5]),
        },
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-5.0, 5.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![1]),
        },
    ];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);
    let check = check_certificate(&cert, None, None);

    // Forward reference from layer 1 → layer 5 detected.
    // (Also produces DanglingSourceRef since layer 5 doesn't exist in bounds.)
    let has_forward = check.issues.iter().any(|i| {
        matches!(
            i,
            CheckIssue::ForwardReference {
                layer_index: 1,
                source_index: 5
            }
        )
    });
    assert!(
        has_forward,
        "forward reference must be detected: {:?}",
        check.issues
    );
}

/// Cycle A→B→A: layer 2 references layer 0, layer 0 references layer 2.
/// Forward reference check catches the back-edge from layer 0→2.
#[test]
fn test_f4_cycle_via_forward_reference() {
    let result = sample_verification();
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            // Layer 0 claims input from layer 2 — forward reference (cycle)
            input_sources: Some(vec![2]),
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
            input_bounds: vec![(0.0, 5.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![1]),
        },
    ];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);
    let check = check_certificate(&cert, None, None);

    // Layer 0 referencing layer 2 is a forward reference.
    assert!(
        check.issues.iter().any(|i| matches!(
            i,
            CheckIssue::ForwardReference {
                layer_index: 0,
                source_index: 2
            }
        )),
        "cycle via forward reference must be detected: {:?}",
        check.issues
    );
}

/// Valid topological ordering should not produce ForwardReference.
#[test]
fn test_f4_valid_topology_no_false_positive() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    assert!(
        !check
            .issues
            .iter()
            .any(|i| matches!(i, CheckIssue::ForwardReference { .. })),
        "valid topology should not produce ForwardReference: {:?}",
        check.issues
    );
}
