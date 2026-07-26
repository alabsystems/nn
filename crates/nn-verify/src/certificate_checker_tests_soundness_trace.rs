// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! v2.0 sequential trace fallback path coverage tests (#1692 proof_coverage audit).
//!
//! Extracted from `certificate_checker_tests_soundness.rs` for 500-line compliance.
//! Tests the sequential trace validation path used when certificates lack
//! `input_sources` graph topology.

use super::checker_test_shared::{sample_input_spec, sample_verification};
use super::*;
use crate::certificate::ProofCertificate;
use crate::certificate_types::LayerBoundRecord;
use crate::verify_types::PropMethod;

// -- Sequential trace tests ---------------------------------------------------

/// v2.0 certificate without `input_sources` uses sequential trace validation.
///
/// Consistent sequential bounds (layer[i].output == layer[i+1].input)
/// should produce no LayerTraceGap issues.
#[test]
fn test_sequential_trace_consistent_bounds_no_gap() {
    let result = sample_verification();
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: None, // v2.0: no graph topology
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-5.0, 5.0)], // Matches layer 0 output
            output_bounds: vec![(0.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: None,
        },
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(0.0, 5.0)], // Matches layer 1 output
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: None,
        },
    ];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);
    let check = check_certificate(&cert, None, None);

    let trace_gaps = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::LayerTraceGap { .. }))
        .count();
    assert_eq!(
        trace_gaps, 0,
        "consistent sequential bounds should have no trace gaps: {:?}",
        check.issues
    );
}

/// v2.0 sequential trace detects a gap when layer[i].output != layer[i+1].input.
#[test]
fn test_sequential_trace_detects_gap() {
    let result = sample_verification();
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: None,
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-999.0, 999.0)], // DOES NOT match layer 0 output
            output_bounds: vec![(0.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: None,
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
        "sequential trace should detect 1 gap: {:?}",
        check.issues
    );
    if let CheckIssue::LayerTraceGap {
        layer_index,
        output_bounds,
        next_input_bounds,
    } = &trace_gaps[0]
    {
        assert_eq!(*layer_index, 0);
        assert_eq!(output_bounds, &[(-5.0, 5.0)]);
        assert_eq!(next_input_bounds, &[(-999.0, 999.0)]);
    }
}

/// v2.0 sequential trace with single-layer bounds has no gap (no pair to compare).
#[test]
fn test_sequential_trace_single_layer_no_gap() {
    let result = sample_verification();
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-10.0, 10.0)],
        output_bounds: vec![(-5.0, 5.0)],
        method: PropMethod::Ibp,
        node_name: None,
        input_sources: None,
    }];

    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);
    let check = check_certificate(&cert, None, None);

    let trace_gaps = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::LayerTraceGap { .. }))
        .count();
    assert_eq!(trace_gaps, 0, "single-layer trace should have no gaps");
}
