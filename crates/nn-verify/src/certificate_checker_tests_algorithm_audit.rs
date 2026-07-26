// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Algorithm audit tests for certificate checker.
//!
//! Covers three findings from P10 algorithm_audit rotation.
//! F1 and F2 have production fixes; these tests verify the fixed behavior.
//!
//! F1 (FIXED): NaN in intermediate layer output bounds is now detected by
//!     `check_nonfinite_output_bounds`, which runs before sequential/graph-aware
//!     dispatch. Previously only the multi-source path had finiteness checks.
//!
//! F2 (FIXED): `check_layer_trace_sequential` now reports `bounds[i].layer_index`
//!     (the record's layer_index), consistent with the graph-aware path.
//!     Previously it reported array position `i`.
//!
//! F3 (by design): `check_inverted_element_bounds` skips NaN pairs via
//!     `is_finite()` guard. NaN in layer bounds is now caught by the F1 fix
//!     (`NonFiniteElement`), not by the inverted bounds check.
//!
//! Part of #3020 algorithm_audit phase.

use super::checker_test_shared::{sample_input_spec, sample_verification};
use super::*;
use crate::certificate::ProofCertificate;
use crate::certificate_types::LayerBoundRecord;
use crate::status::{InputBoundsRecord, ParamInputRecord};
use crate::verify_types::PropMethod;

// ---------------------------------------------------------------------------
// F1: NaN in intermediate layer bounds causes spurious LayerTraceGap
// ---------------------------------------------------------------------------

/// Helper: build layer bounds where layer 1 has NaN in output, and layer 2's
/// input matches (same NaN pattern). Sequential check should see them as
/// consistent, but `Vec<(f32, f32)>` equality fails because NaN != NaN.
fn nan_intermediate_bounds_sequential() -> Vec<LayerBoundRecord> {
    vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: None, // sequential mode: no input_sources
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "LayerNorm".to_string(),
            input_bounds: vec![(-5.0, 5.0)],
            // NaN in output bounds (e.g., from normalization layer)
            output_bounds: vec![(f32::NAN, f32::NAN)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: None,
        },
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Linear".to_string(),
            // Same NaN pattern as previous layer's output — logically consistent
            input_bounds: vec![(f32::NAN, f32::NAN)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: None,
        },
    ]
}

/// NaN in intermediate layer output bounds is now detected by
/// `check_nonfinite_output_bounds` which runs before sequential/graph-aware
/// dispatch. NonFiniteElement is emitted for the NaN layer. The sequential
/// check may also emit a LayerTraceGap (NaN != NaN per IEEE 754), but the
/// primary detection is now the non-finite check.
#[test]
fn test_nan_intermediate_bounds_sequential_detected_as_nonfinite() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(nan_intermediate_bounds_sequential())
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    // NonFiniteElement is now emitted for layer 1 (NaN output).
    let has_nonfinite = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NonFiniteElement { layer_index: 1, .. }));
    assert!(
        has_nonfinite,
        "NaN intermediate bounds detected as NonFiniteElement"
    );

    // Verify that check_inverted_element_bounds does NOT flag NaN
    // (it requires is_finite(), so NaN pairs are silently skipped).
    let has_inverted = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::InvertedElementBounds { .. }));
    assert!(
        !has_inverted,
        "NaN pairs are not flagged as inverted (is_finite() guard skips them)"
    );

    // The certificate is NOT valid.
    assert!(!check.is_valid());
}

/// NaN in the LAST layer's output bounds IS detected by check_output_agreement
/// (NanOutputBounds issue). Intermediate NaN is detected by check_nonfinite_output_bounds
/// (NonFiniteElement issue, added by F1 fix). This test verifies the output agreement path.
#[test]
fn test_nan_last_layer_detected_but_intermediate_not() {
    // Build bounds where only the last layer has NaN output
    let last_nan_bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: None,
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-5.0, 5.0)],
            output_bounds: vec![(f32::NAN, f32::NAN)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: None,
        },
    ];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(last_nan_bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    // Last layer NaN IS detected by check_output_agreement
    let has_nan_output = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NanOutputBounds));
    assert!(
        has_nan_output,
        "NaN in last layer's output detected by agreement checker"
    );
}

/// NaN in intermediate layer output bounds (not the last layer): the
/// non-finite check now correctly detects it as NonFiniteElement.
#[test]
fn test_nan_intermediate_detected_as_nonfinite() {
    // Layer 0 outputs NaN, layer 1 has finite bounds that differ from NaN
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(f32::NAN, 2.0)], // NaN lower, finite upper
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: None,
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(0.0, 2.0)], // Different from layer 0 output
            output_bounds: vec![(0.0, 2.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: None,
        },
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(0.0, 2.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: None,
        },
    ];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    // NonFiniteElement is now emitted for layer 0 (NaN lower bound).
    let has_nonfinite = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NonFiniteElement { layer_index: 0, .. }));
    assert!(
        has_nonfinite,
        "NonFiniteElement emitted for intermediate layer NaN"
    );

    // The certificate is NOT valid.
    assert!(!check.is_valid());
}

// ---------------------------------------------------------------------------
// F1 (graph-aware): Same NaN issue in single-source graph-aware path
// ---------------------------------------------------------------------------

/// Graph-aware single-source: NonFiniteElement is now detected by the
/// unified non-finite check before graph-aware dispatch.
#[test]
fn test_nan_intermediate_graph_aware_single_source_detected() {
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            // NaN output
            output_bounds: vec![(f32::NAN, f32::NAN)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]), // network input
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            // Same NaN as previous output — logically consistent
            input_bounds: vec![(f32::NAN, f32::NAN)],
            output_bounds: vec![(0.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![0]), // single source: layer 0
        },
    ];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    // NonFiniteElement is now emitted for layer 0 (NaN output).
    let has_nonfinite = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NonFiniteElement { layer_index: 0, .. }));
    assert!(
        has_nonfinite,
        "Graph-aware single-source: NaN detected as NonFiniteElement"
    );
}

/// Graph-aware MULTI-source path: NaN detected both by the unified
/// non-finite check AND the multi-source containment check (defense-in-depth).
#[test]
fn test_nan_intermediate_graph_aware_multi_source_detected() {
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(f32::NAN, 2.0)], // NaN in output
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]), // network input
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(1.0, 3.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]), // network input
        },
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Add".to_string(),
            input_bounds: vec![(0.0, 5.0)],
            output_bounds: vec![(0.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![0, 1]), // multi-source: layers 0 and 1
        },
    ];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    // Multi-source path detects NaN via explicit finiteness check.
    let has_nonfinite = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NonFiniteElement { layer_index: 0, .. }));
    assert!(
        has_nonfinite,
        "Multi-source path correctly detects NaN via NonFiniteElement"
    );
}

// ---------------------------------------------------------------------------
// F2: Sequential trace uses array position, not layer_index
// ---------------------------------------------------------------------------

/// `check_layer_trace_sequential` now reports the actual `layer_index` from
/// `LayerBoundRecord`, consistent with the graph-aware path.
#[test]
fn test_sequential_layer_index_uses_record_layer_index() {
    // Build bounds with non-sequential layer indices
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 5,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: None, // sequential mode
        },
        LayerBoundRecord {
            layer_index: 10,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(0.0, 99.0)], // Mismatched to force gap
            output_bounds: vec![(0.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: None,
        },
    ];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    // Find the LayerTraceGap and verify the reported layer_index.
    let gap = check.issues.iter().find_map(|i| {
        if let CheckIssue::LayerTraceGap { layer_index, .. } = i {
            Some(*layer_index)
        } else {
            None
        }
    });

    // Fixed: reports actual layer_index 5, not array position 0.
    assert_eq!(
        gap,
        Some(5),
        "sequential mode now reports bounds[0].layer_index (5), not array position (0)"
    );
}

// ---------------------------------------------------------------------------
// F3: check_inverted_element_bounds skips NaN (is_finite guard)
// ---------------------------------------------------------------------------

/// Verify that check_inverted_element_bounds does NOT flag NaN as inverted.
/// NaN.is_finite() returns false, so the `lo.is_finite() && hi.is_finite() && lo > hi`
/// guard correctly excludes NaN from inverted bounds detection. But this means
/// NaN in layer bounds is silently ignored by the inverted bounds check.
#[test]
fn test_inverted_check_skips_nan_deliberately() {
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-10.0, 10.0)],
        output_bounds: vec![
            (f32::NAN, 1.0),      // NaN lower
            (0.0, f32::NAN),      // NaN upper
            (f32::NAN, f32::NAN), // both NaN
            (5.0, 3.0),           // inverted (finite)
        ],
        method: PropMethod::Ibp,
        node_name: None,
        input_sources: Some(vec![]), // graph-aware mode
    }];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    // Only element 3 (5.0 > 3.0) should be flagged as inverted.
    let inverted_count = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::InvertedElementBounds { .. }))
        .count();
    assert_eq!(
        inverted_count, 1,
        "only finite inverted pair should be flagged"
    );

    // Verify the specific element flagged
    assert!(check.issues.iter().any(|i| matches!(
        i,
        CheckIssue::InvertedElementBounds {
            element_index: 3,
            lower,
            upper,
            ..
        } if (*lower - 5.0).abs() < 1e-6 && (*upper - 3.0).abs() < 1e-6
    )));

    // NaN elements are NOT flagged by check_inverted_element_bounds
    let nan_flagged_as_inverted = check.issues.iter().any(|i| {
        matches!(
            i,
            CheckIssue::InvertedElementBounds {
                element_index: 0..=2,
                ..
            }
        )
    });
    assert!(
        !nan_flagged_as_inverted,
        "NaN elements should not be flagged as inverted (is_finite guard)"
    );
}

/// Inf in output bounds: similar to NaN — Inf is not finite, so inverted
/// check skips it. But `Inf > -Inf` is true, and the guard checks
/// `is_finite()` first, so even `(Inf, -Inf)` (inverted infinite) is skipped.
#[test]
fn test_inverted_check_skips_infinite_bounds() {
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-10.0, 10.0)],
        output_bounds: vec![
            (f32::INFINITY, f32::NEG_INFINITY), // inverted infinite
            (f32::NEG_INFINITY, f32::INFINITY), // normal infinite range
        ],
        method: PropMethod::Ibp,
        node_name: None,
        input_sources: Some(vec![]),
    }];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    // Neither infinite pair should be flagged as inverted (is_finite guard)
    let inverted_count = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::InvertedElementBounds { .. }))
        .count();
    assert_eq!(
        inverted_count, 0,
        "infinite bounds are not flagged as inverted (is_finite guard)"
    );
}

// ---------------------------------------------------------------------------
// F4: NaN in first layer input_bounds — undetected when spec length differs
// ---------------------------------------------------------------------------

/// When the first layer's input_bounds has NaN but the spec has a different
/// element count (broadcast semantics), check_first_layer_input_spec previously
/// returned early at the length guard. The NaN went completely undetected.
///
/// Fix: check input_bounds finiteness BEFORE the length comparison.
#[test]
fn test_nan_first_layer_input_bounds_spec_length_mismatch() {
    // First layer has 3 input elements (NaN in first), spec has 1 variable input
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(f32::NAN, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        output_bounds: vec![(-5.0, 5.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: Some(vec![]),
    }];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    // The NaN in input_bounds must be detected as NonFiniteElement.
    let has_nonfinite = check.issues.iter().any(|i| {
        matches!(
            i,
            CheckIssue::NonFiniteElement {
                layer_index: 0,
                element_index: 0,
                ..
            }
        )
    });
    assert!(
        has_nonfinite,
        "NaN in first layer input_bounds must be detected even when spec length differs: {:?}",
        check.issues
    );
    assert!(!check.is_valid());
}

/// When spec length matches AND first layer input has NaN, both NonFiniteElement
/// and InputBoundsSpecMismatch should be emitted (defense in depth).
#[test]
fn test_nan_first_layer_input_bounds_spec_length_match() {
    // Spec has 1 variable input [-10, 10], first layer has 1 input element with NaN
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(f32::NAN, 10.0)],
        output_bounds: vec![(-5.0, 5.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: Some(vec![]),
    }];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    // NonFiniteElement for the NaN in input_bounds
    let has_nonfinite = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NonFiniteElement { layer_index: 0, .. }));
    assert!(
        has_nonfinite,
        "NonFiniteElement for NaN in first layer input: {:?}",
        check.issues
    );

    // InputBoundsSpecMismatch because (NaN, 10.0) != (-10.0, 10.0)
    let has_mismatch = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::InputBoundsSpecMismatch { .. }));
    assert!(
        has_mismatch,
        "InputBoundsSpecMismatch for NaN vs finite spec: {:?}",
        check.issues
    );
}

/// Inf in first layer input_bounds with spec length mismatch — same gap as NaN.
#[test]
fn test_inf_first_layer_input_bounds_spec_length_mismatch() {
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-1.0, f32::INFINITY), (-1.0, 1.0)],
        output_bounds: vec![(-5.0, 5.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: Some(vec![]),
    }];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    let has_nonfinite = check.issues.iter().any(|i| {
        matches!(
            i,
            CheckIssue::NonFiniteElement {
                layer_index: 0,
                element_index: 0,
                ..
            }
        )
    });
    assert!(
        has_nonfinite,
        "Inf in first layer input_bounds must be detected: {:?}",
        check.issues
    );
}

// ---------------------------------------------------------------------------
// F5: Inverted input bounds on first layer — undetected when spec length differs
// ---------------------------------------------------------------------------

/// Inverted input bounds (lo > hi) on the first layer with a spec length
/// mismatch. check_inverted_element_bounds only checks output bounds.
/// The F4 NaN check skips inverted finite pairs. Without F5, the inverted
/// input bounds go completely undetected — the certificate is vacuously true
/// (verified "for no inputs" since [5, 3] is an empty interval).
#[test]
fn test_inverted_first_layer_input_bounds_spec_length_mismatch() {
    // First layer has 2 input elements, one inverted. Spec has 1 variable input.
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(5.0, 3.0), (-1.0, 1.0)], // first element inverted
        output_bounds: vec![(-5.0, 5.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: Some(vec![]),
    }];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    let has_inverted = check.issues.iter().any(|i| {
        matches!(
            i,
            CheckIssue::InvertedElementBounds {
                layer_index: 0,
                element_index: 0,
                ..
            }
        )
    });
    assert!(
        has_inverted,
        "Inverted input bounds on first layer must be detected when spec length differs: {:?}",
        check.issues
    );
    assert!(!check.is_valid());
}

/// Inverted input bounds with matching spec length — both InvertedElementBounds
/// and InputBoundsSpecMismatch should be emitted.
#[test]
fn test_inverted_first_layer_input_bounds_spec_length_match() {
    // Spec has 1 variable input [-10, 10], first layer has 1 inverted input
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(5.0, 3.0)], // inverted: 5 > 3
        output_bounds: vec![(-5.0, 5.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: Some(vec![]),
    }];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    // InvertedElementBounds from the F5 check
    let has_inverted = check.issues.iter().any(|i| {
        matches!(
            i,
            CheckIssue::InvertedElementBounds {
                layer_index: 0,
                element_index: 0,
                ..
            }
        )
    });
    assert!(
        has_inverted,
        "InvertedElementBounds for inverted first layer input: {:?}",
        check.issues
    );

    // InputBoundsSpecMismatch because (5.0, 3.0) != (-10.0, 10.0)
    let has_mismatch = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::InputBoundsSpecMismatch { .. }));
    assert!(
        has_mismatch,
        "InputBoundsSpecMismatch for inverted vs normal spec: {:?}",
        check.issues
    );
}

// ---------------------------------------------------------------------------
// F6: Multi-source layer input_bounds not validated for NaN/Inf/inverted
// ---------------------------------------------------------------------------

/// Multi-source Add layer with NaN in its own input_bounds while both sources
/// have valid output_bounds. Before F6, check_multi_source_containment only
/// validated the SOURCE's output bounds, not the target layer's input_bounds.
/// The NaN input_bounds went completely undetected.
#[test]
fn test_multi_source_nan_input_bounds_detected() {
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0)], // valid
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(1.0, 3.0)], // valid
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Add".to_string(),
            // NaN in multi-source layer's own input_bounds — previously undetected
            input_bounds: vec![(f32::NAN, 5.0)],
            output_bounds: vec![(0.0, 5.0)], // valid output
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![0, 1]), // multi-source
        },
    ];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    // F6 fix: NonFiniteElement is now emitted for layer 2's NaN input_bounds.
    let has_nonfinite_layer2 = check.issues.iter().any(|i| {
        matches!(
            i,
            CheckIssue::NonFiniteElement {
                layer_index: 2,
                element_index: 0,
                ..
            }
        )
    });
    assert!(
        has_nonfinite_layer2,
        "NaN in multi-source layer input_bounds must be detected: {:?}",
        check.issues
    );
    assert!(!check.is_valid());
}

/// Multi-source layer with inverted (lo > hi) input_bounds. Empty interval
/// means the proof for this subgraph is vacuously true. Before F6, this
/// was undetected because check_inverted_element_bounds only checked output.
#[test]
fn test_multi_source_inverted_input_bounds_detected() {
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
            output_bounds: vec![(1.0, 3.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Add".to_string(),
            // Inverted: 8.0 > 2.0. Empty interval → vacuously true proof.
            input_bounds: vec![(8.0, 2.0)],
            output_bounds: vec![(0.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![0, 1]),
        },
    ];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    let has_inverted_layer2 = check.issues.iter().any(|i| {
        matches!(
            i,
            CheckIssue::InvertedElementBounds {
                layer_index: 2,
                element_index: 0,
                ..
            }
        )
    });
    assert!(
        has_inverted_layer2,
        "Inverted input_bounds on multi-source layer must be detected: {:?}",
        check.issues
    );
    assert!(!check.is_valid());
}

/// Second network input layer with NaN input_bounds. check_first_layer_input_spec
/// uses `bounds.iter().find()` which only validates the FIRST network input.
/// The second input layer's input_bounds were completely unchecked.
#[test]
fn test_second_network_input_nan_input_bounds_detected() {
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)], // valid first input
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]), // network input
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "Embedding".to_string(),
            // NaN on second network input — previously undetected
            input_bounds: vec![(f32::NAN, f32::NAN)],
            output_bounds: vec![(0.0, 1.0)], // valid output
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]), // second network input
        },
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Add".to_string(),
            input_bounds: vec![(0.0, 5.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![0, 1]),
        },
    ];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    // F6 fix: NonFiniteElement for layer 1's NaN input_bounds.
    let has_nonfinite_layer1 = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NonFiniteElement { layer_index: 1, .. }));
    assert!(
        has_nonfinite_layer1,
        "NaN in second network input layer's input_bounds must be detected: {:?}",
        check.issues
    );
    assert!(!check.is_valid());
}

// ---------------------------------------------------------------------------
// F7: Multi-input spec validation bypass (#3322)
// ---------------------------------------------------------------------------

/// Multi-input certificate where the second network input layer has bounds
/// that DON'T match the input_spec. The checker detects this because
/// `check_first_layer_input_spec` (#3322) iterates ALL network input layers
/// and compares each against its corresponding slice of spec_bounds.
#[test]
fn test_multi_input_spec_bypass_second_input_unchecked() {
    // Spec: 2 variable inputs, each with 1 element.
    // Input 0: [-10, 10], Input 1: [-5, 5]
    let multi_input_spec = InputBoundsRecord {
        variable_inputs: vec![
            ParamInputRecord {
                param_index: 0,
                lower: -10.0,
                upper: 10.0,
            },
            ParamInputRecord {
                param_index: 1,
                lower: -5.0,
                upper: 5.0,
            },
        ],
        constant_params: vec![],
        input_shape: Some(vec![2]),
        input_range: None,
    };

    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            // First network input: matches spec entry 0 [-10, 10]
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]), // network input
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "Embedding".to_string(),
            // Second network input: WRONG bounds [-100, 100].
            // Spec says [-5, 5] but this layer uses [-100, 100].
            // The certificate is vacuously proving bounds for a wider
            // input range than what was specified.
            input_bounds: vec![(-100.0, 100.0)],
            output_bounds: vec![(0.0, 1.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]), // second network input
        },
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Add".to_string(),
            input_bounds: vec![(0.0, 5.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![0, 1]),
        },
    ];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, multi_input_spec)
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    // #3322 FIXED: check_first_layer_input_spec now finds ALL network input
    // layers and compares each against its corresponding spec slice.
    // The second input's [-100, 100] vs spec [-5, 5] IS now detected.
    let has_spec_mismatch = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::InputBoundsSpecMismatch { .. }));

    assert!(
        has_spec_mismatch,
        "Expected InputBoundsSpecMismatch for second input layer ([-100, 100] vs spec [-5, 5]). \
         Issues: {:?}",
        check.issues
    );
}

/// Multi-input certificate where BOTH network input layers match the spec.
/// No InputBoundsSpecMismatch should be emitted.
#[test]
fn test_multi_input_spec_both_matching_passes() {
    let multi_input_spec = InputBoundsRecord {
        variable_inputs: vec![
            ParamInputRecord {
                param_index: 0,
                lower: -10.0,
                upper: 10.0,
            },
            ParamInputRecord {
                param_index: 1,
                lower: -5.0,
                upper: 5.0,
            },
        ],
        constant_params: vec![],
        input_shape: Some(vec![2]),
        input_range: None,
    };

    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)], // matches spec entry 0
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "Embedding".to_string(),
            input_bounds: vec![(-5.0, 5.0)], // matches spec entry 1
            output_bounds: vec![(0.0, 1.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Add".to_string(),
            input_bounds: vec![(0.0, 5.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![0, 1]),
        },
    ];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, multi_input_spec)
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);

    assert!(
        !check
            .issues
            .iter()
            .any(|i| matches!(i, CheckIssue::InputBoundsSpecMismatch { .. })),
        "Both inputs match spec — no InputBoundsSpecMismatch expected: {:?}",
        check.issues
    );
}

// ---------------------------------------------------------------------------
// F8: VacuityAssessment crown_layers only counted PropMethod::Crown exactly,
//     missing AlphaCrown, BetaCrown, and Analytical (all tighter than Crown).
//     A certificate verified entirely with AlphaCrown was incorrectly flagged
//     as VacuousBounds with crown_coverage=0.0.
// ---------------------------------------------------------------------------

/// AlphaCrown layers should count as "crown" for vacuity assessment.
/// Before the fix, only PropMethod::Crown was counted.
#[test]
fn test_vacuity_alpha_crown_counted_as_tight() {
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::AlphaCrown,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-5.0, 5.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::AlphaCrown,
            node_name: None,
            input_sources: Some(vec![0]),
        },
    ];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    let vacuity = check.vacuity.as_ref().expect("vacuity assessment present");

    assert_eq!(
        vacuity.crown_layers, 2,
        "AlphaCrown layers must count as crown"
    );
    assert_eq!(vacuity.ibp_layers, 0);
    assert!(
        (vacuity.crown_coverage - 1.0).abs() < 1e-6,
        "100% AlphaCrown should give crown_coverage=1.0, got {}",
        vacuity.crown_coverage
    );
}

/// BetaCrown layers should count as "crown" for vacuity assessment.
#[test]
fn test_vacuity_beta_crown_counted_as_tight() {
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-10.0, 10.0)],
        output_bounds: vec![(-5.0, 5.0)],
        method: PropMethod::BetaCrown,
        node_name: None,
        input_sources: Some(vec![]),
    }];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    let vacuity = check.vacuity.as_ref().expect("vacuity assessment present");

    assert_eq!(vacuity.crown_layers, 1, "BetaCrown must count as crown");
    assert_eq!(vacuity.ibp_layers, 0);
}

/// Analytical layers should count as "crown" (tight) for vacuity assessment.
#[test]
fn test_vacuity_analytical_counted_as_tight() {
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-10.0, 10.0)],
        output_bounds: vec![(-5.0, 5.0)],
        method: PropMethod::Analytical,
        node_name: None,
        input_sources: Some(vec![]),
    }];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    let vacuity = check.vacuity.as_ref().expect("vacuity assessment present");

    assert_eq!(vacuity.crown_layers, 1, "Analytical must count as crown");
}

/// MixedIbpCrown should NOT count as tight — it partially uses IBP.
#[test]
fn test_vacuity_mixed_ibp_crown_not_counted_as_tight() {
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-10.0, 10.0)],
        output_bounds: vec![(-5.0, 5.0)],
        method: PropMethod::MixedIbpCrown,
        node_name: None,
        input_sources: Some(vec![]),
    }];

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    let vacuity = check.vacuity.as_ref().expect("vacuity assessment present");

    assert_eq!(
        vacuity.crown_layers, 0,
        "MixedIbpCrown should NOT count as tight"
    );
    assert_eq!(vacuity.ibp_layers, 1);
}

/// Mixed PropMethod certificate: 2 Crown-family + 1 IBP = 66% coverage.
/// Uses a narrow output_width (< DEFAULT_VACUITY_THRESHOLD=10.0) so coverage
/// is the deciding factor for is_non_vacuous.
#[test]
fn test_vacuity_mixed_methods_coverage_correct() {
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-2.0, 2.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-2.0, 2.0)],
            output_bounds: vec![(0.0, 2.0)],
            method: PropMethod::AlphaCrown,
            node_name: None,
            input_sources: Some(vec![0]),
        },
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(0.0, 2.0)],
            output_bounds: vec![(-1.0, 1.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![1]),
        },
    ];

    // Use a verification with narrow output width (2.0 < threshold 10.0)
    let mut result = sample_verification();
    result.output_lower = -1.0;
    result.output_upper = 1.0;
    result.output_width = 2.0;
    if let Some(ref mut ot) = result.output_tensor {
        ot.lower = vec![-1.0];
        ot.upper = vec![1.0];
    }

    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    let vacuity = check.vacuity.as_ref().expect("vacuity assessment present");

    assert_eq!(
        vacuity.crown_layers, 2,
        "Crown + AlphaCrown = 2 tight layers"
    );
    assert_eq!(vacuity.ibp_layers, 1);
    let expected_coverage = 2.0 / 3.0;
    assert!(
        (vacuity.crown_coverage - expected_coverage).abs() < 1e-6,
        "coverage should be 2/3 ≈ 0.667, got {}",
        vacuity.crown_coverage
    );
    assert!(
        vacuity.is_non_vacuous,
        "2/3 coverage with width 2.0 should be non-vacuous"
    );
}
