// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for multi-tensor trace comparison: tolerance escalation for
//! accumulated error, report generation, and empty/edge-case traces.

use crate::compare::{compare_tensors, compare_traces, ComparisonConfig};
use crate::error::ReftestError;
use crate::trace::{NamedTensor, ReferenceTrace};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tensor_1d(name: &str, data: Vec<f32>) -> NamedTensor {
    let len = data.len();
    NamedTensor::new(name, vec![len], data).expect("valid 1-D test tensor")
}

/// Build a trace from a slice of (name, data) pairs.
fn build_trace(layers: &[(&str, Vec<f32>)]) -> ReferenceTrace {
    let mut trace = ReferenceTrace::new();
    for (name, data) in layers {
        trace
            .checkpoint(name, data, &[data.len()])
            .expect("valid checkpoint");
    }
    trace
}

// ===========================================================================
// 1. Multi-tensor trace comparison (sequence of intermediate results)
// ===========================================================================

#[test]
fn test_multi_layer_trace_all_identical() {
    let layers = vec![
        ("embed", vec![1.0, 2.0, 3.0]),
        ("attn", vec![0.5, 0.6, 0.7, 0.8]),
        ("ffn", vec![10.0, 20.0]),
        ("norm", vec![0.1, -0.1]),
    ];
    let ref_trace = build_trace(&layers);
    let cand_trace = build_trace(&layers);

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("comparison should succeed");
    assert!(report.all_passed);
    assert!(report.first_failure.is_none());
    assert_eq!(report.layers.len(), 4);

    // Every layer should individually pass.
    for layer in &report.layers {
        assert!(layer.passed, "layer '{}' should pass", layer.name);
        assert_eq!(layer.max_abs_diff, 0.0);
    }
}

#[test]
fn test_multi_layer_trace_first_diverges() {
    let ref_trace = build_trace(&[
        ("layer0", vec![1.0, 2.0]),
        ("layer1", vec![3.0, 4.0]),
        ("layer2", vec![5.0, 6.0]),
    ]);
    let cand_trace = build_trace(&[
        ("layer0", vec![100.0, 200.0]), // large divergence here
        ("layer1", vec![3.0, 4.0]),
        ("layer2", vec![5.0, 6.0]),
    ]);

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("comparison should succeed");
    assert!(!report.all_passed);
    assert_eq!(report.first_failure, Some(0), "first divergence at layer 0");
    assert!(!report.layers[0].passed);
    assert!(
        report.layers[1].passed,
        "later layers may still individually pass"
    );
}

#[test]
fn test_multi_layer_trace_middle_diverges() {
    let ref_trace = build_trace(&[
        ("layer0", vec![1.0]),
        ("layer1", vec![2.0]),
        ("layer2", vec![3.0]),
    ]);
    let cand_trace = build_trace(&[
        ("layer0", vec![1.0]),
        ("layer1", vec![999.0]), // diverges
        ("layer2", vec![3.0]),
    ]);

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("comparison should succeed");
    assert!(!report.all_passed);
    assert_eq!(report.first_failure, Some(1));
    assert!(report.layers[0].passed);
    assert!(!report.layers[1].passed);
    assert!(report.layers[2].passed);
}

#[test]
fn test_multi_layer_trace_last_diverges() {
    let ref_trace = build_trace(&[
        ("encoder", vec![1.0, 2.0]),
        ("decoder", vec![3.0, 4.0]),
        ("output", vec![5.0, 6.0]),
    ]);
    let cand_trace = build_trace(&[
        ("encoder", vec![1.0, 2.0]),
        ("decoder", vec![3.0, 4.0]),
        ("output", vec![5.0, -6.0]), // sign flip at output
    ]);

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("comparison should succeed");
    assert!(!report.all_passed);
    assert_eq!(report.first_failure, Some(2));
}

#[test]
fn test_multi_layer_all_diverge() {
    let ref_trace = build_trace(&[("a", vec![1.0]), ("b", vec![2.0]), ("c", vec![3.0])]);
    let cand_trace = build_trace(&[("a", vec![99.0]), ("b", vec![99.0]), ("c", vec![99.0])]);

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("comparison should succeed");
    assert!(!report.all_passed);
    assert_eq!(
        report.first_failure,
        Some(0),
        "first_failure should be the earliest divergent layer"
    );
    // All three layers should fail.
    for layer in &report.layers {
        assert!(!layer.passed, "layer '{}' should fail", layer.name);
    }
}

// ===========================================================================
// 2. Tolerance escalation for accumulated error
// ===========================================================================

#[test]
fn test_escalating_error_strict_fails_relaxed_passes() {
    // Simulate accumulated numerical error through layers: each layer has
    // progressively larger error (as happens in deep network forward passes).
    let ref_trace = build_trace(&[
        ("layer0", vec![1.0, 2.0, 3.0]),
        ("layer1", vec![10.0, 20.0, 30.0]),
        ("layer2", vec![100.0, 200.0, 300.0]),
    ]);
    // Add escalating perturbation: 1e-7, 1e-5, 1e-3.
    // These are chosen to be clearly within relaxed (atol=1e-2, rtol=1e-1, cos=0.999)
    // but clearly outside strict (atol=1e-6, rtol=1e-5, cos=0.999999).
    let cand_trace = build_trace(&[
        ("layer0", vec![1.0 + 1e-7, 2.0 + 1e-7, 3.0 + 1e-7]),
        ("layer1", vec![10.0 + 1e-5, 20.0 + 1e-5, 30.0 + 1e-5]),
        ("layer2", vec![100.0 + 1e-3, 200.0 + 1e-3, 300.0 + 1e-3]),
    ]);

    // Strict config rejects the later layers with larger accumulated error.
    let strict_report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::strict())
        .expect("comparison should succeed");
    assert!(
        !strict_report.all_passed,
        "strict should reject accumulated error"
    );
    // Layer 2 definitely fails strict (1e-3 diff >> strict atol=1e-6).
    assert!(
        !strict_report.layers[2].passed,
        "layer2 with 1e-3 diff should fail strict"
    );

    // Relaxed config (atol=1e-2, rtol=1e-1, cos=0.999) accepts all layers.
    let relaxed_report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::relaxed())
        .expect("comparison should succeed");
    assert!(
        relaxed_report.all_passed,
        "relaxed should accept accumulated error up to 1e-3"
    );
}

#[test]
fn test_per_layer_tolerance_via_separate_comparisons() {
    // Demonstrate that users can apply different tolerances per layer
    // by running compare_tensors individually rather than compare_traces.
    let ref_layers = [tensor_1d("early", vec![1.0, 2.0]),
        tensor_1d("late", vec![100.0, 200.0])];
    let cand_layers = [
        tensor_1d("early", vec![1.0 + 1e-7, 2.0 + 1e-7]), // tight tolerance OK
        tensor_1d("late", vec![100.005, 200.005]),        // needs looser tolerance
    ];

    // Early layer: strict tolerance.
    let early_result =
        compare_tensors(&ref_layers[0], &cand_layers[0], &ComparisonConfig::strict())
            .expect("should succeed");
    assert!(early_result.passed, "early layer should pass strict");

    // Late layer: relaxed tolerance for accumulated error.
    let late_result = compare_tensors(
        &ref_layers[1],
        &cand_layers[1],
        &ComparisonConfig::relaxed(),
    )
    .expect("should succeed");
    assert!(late_result.passed, "late layer should pass relaxed");

    // Late layer with strict should fail.
    let late_strict = compare_tensors(&ref_layers[1], &cand_layers[1], &ComparisonConfig::strict())
        .expect("should succeed");
    assert!(!late_strict.passed, "late layer should fail strict");
}

// ===========================================================================
// 3. Report generation (DivergenceReport)
// ===========================================================================

#[test]
fn test_report_summary_all_passed() {
    let trace = build_trace(&[("a", vec![1.0]), ("b", vec![2.0]), ("c", vec![3.0])]);
    let report = compare_traces(&trace, &trace, &ComparisonConfig::default())
        .expect("comparison should succeed");

    let summary = report.summary();
    assert!(
        summary.contains("All 3 layers passed"),
        "summary should report all layers passed: {summary}"
    );
    // Each layer should appear as PASS.
    assert!(
        summary.contains("[PASS]"),
        "summary should contain PASS markers"
    );
    assert!(
        !summary.contains("[FAIL]"),
        "summary should not contain FAIL markers when all pass"
    );
}

#[test]
fn test_report_summary_with_failure_shows_layer_name() {
    let ref_trace = build_trace(&[("embed", vec![1.0]), ("attn.softmax", vec![0.5])]);
    let cand_trace = build_trace(&[("embed", vec![1.0]), ("attn.softmax", vec![999.0])]);

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("comparison should succeed");
    let summary = report.summary();

    assert!(
        summary.contains("First failure at layer 1"),
        "summary should identify failure index: {summary}"
    );
    assert!(
        summary.contains("attn.softmax"),
        "summary should include failing layer name: {summary}"
    );
}

#[test]
fn test_report_layers_preserve_order_and_names() {
    let ref_trace = build_trace(&[
        ("step_0", vec![1.0]),
        ("step_1", vec![2.0]),
        ("step_2", vec![3.0]),
    ]);
    let report = compare_traces(&ref_trace, &ref_trace, &ComparisonConfig::default())
        .expect("comparison should succeed");

    assert_eq!(report.layers.len(), 3);
    assert_eq!(report.layers[0].name, "step_0");
    assert_eq!(report.layers[1].name, "step_1");
    assert_eq!(report.layers[2].name, "step_2");
}

#[test]
fn test_report_metrics_reflect_divergence_magnitude() {
    let ref_trace = build_trace(&[("x", vec![0.0, 0.0, 0.0])]);
    let cand_trace = build_trace(&[("x", vec![0.1, 0.2, 0.3])]);

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::relaxed())
        .expect("comparison should succeed");

    let layer = &report.layers[0];
    assert!(
        (layer.max_abs_diff - 0.3).abs() < 1e-5,
        "max_abs should be ~0.3"
    );
    // mean_abs = (0.1 + 0.2 + 0.3) / 3 = 0.2
    assert!(
        (layer.mean_abs_diff - 0.2).abs() < 1e-5,
        "mean_abs should be ~0.2, got {}",
        layer.mean_abs_diff
    );
    assert_eq!(layer.num_elements, 3);
    assert_eq!(layer.shape, vec![3]);
}

#[test]
fn test_report_display_formatting() {
    let ref_trace = build_trace(&[("conv1", vec![1.0, 2.0, 3.0])]);
    let cand_trace = build_trace(&[("conv1", vec![1.1, 2.1, 3.1])]);

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("comparison should succeed");
    let summary = report.summary();

    // The summary should contain key metrics.
    assert!(
        summary.contains("max_abs="),
        "summary missing max_abs: {summary}"
    );
    assert!(
        summary.contains("cos="),
        "summary missing cosine: {summary}"
    );
    assert!(summary.contains("rms="), "summary missing rms: {summary}");
}

// ===========================================================================
// 4. Empty trace comparison
// ===========================================================================

#[test]
fn test_empty_traces_both_pass() {
    let ref_trace = ReferenceTrace::new();
    let cand_trace = ReferenceTrace::new();

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("comparing empty traces should succeed");
    assert!(report.all_passed);
    assert!(report.first_failure.is_none());
    assert!(report.layers.is_empty());
}

#[test]
fn test_empty_trace_summary() {
    let trace = ReferenceTrace::new();
    let report =
        compare_traces(&trace, &trace, &ComparisonConfig::default()).expect("should succeed");
    let summary = report.summary();
    assert!(
        summary.contains("All 0 layers passed"),
        "empty trace summary should say 0 layers passed: {summary}"
    );
}

#[test]
fn test_empty_vs_nonempty_trace_length_mismatch() {
    let empty = ReferenceTrace::new();
    let nonempty = build_trace(&[("layer", vec![1.0])]);

    let err = compare_traces(&empty, &nonempty, &ComparisonConfig::default())
        .expect_err("should fail on length mismatch");
    match err {
        ReftestError::TraceLengthMismatch {
            reference,
            candidate,
        } => {
            assert_eq!(reference, 0);
            assert_eq!(candidate, 1);
        }
        other => panic!("expected TraceLengthMismatch, got {other:?}"),
    }
}

#[test]
fn test_nonempty_vs_empty_trace_length_mismatch() {
    let nonempty = build_trace(&[("a", vec![1.0]), ("b", vec![2.0])]);
    let empty = ReferenceTrace::new();

    let err = compare_traces(&nonempty, &empty, &ComparisonConfig::default())
        .expect_err("should fail on length mismatch");
    match err {
        ReftestError::TraceLengthMismatch {
            reference,
            candidate,
        } => {
            assert_eq!(reference, 2);
            assert_eq!(candidate, 0);
        }
        other => panic!("expected TraceLengthMismatch, got {other:?}"),
    }
}

// ===========================================================================
// 5. Trace with shape mismatch at specific layer
// ===========================================================================

#[test]
fn test_trace_shape_mismatch_propagates_error() {
    let ref_checkpoints = vec![
        tensor_1d("ok_layer", vec![1.0, 2.0]),
        NamedTensor::new("bad_layer", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("valid"),
    ];
    let cand_checkpoints = vec![
        tensor_1d("ok_layer", vec![1.0, 2.0]),
        NamedTensor::new("bad_layer", vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("valid"),
    ];

    let ref_trace = ReferenceTrace::from_checkpoints(ref_checkpoints);
    let cand_trace = ReferenceTrace::from_checkpoints(cand_checkpoints);

    let err = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect_err("should fail on shape mismatch");
    assert!(matches!(err, ReftestError::ShapeMismatch { .. }));
}

// ===========================================================================
// 6. Trace length mismatch (different checkpoint counts)
// ===========================================================================

#[test]
fn test_trace_length_mismatch_more_reference() {
    let ref_trace = build_trace(&[("a", vec![1.0]), ("b", vec![2.0]), ("c", vec![3.0])]);
    let cand_trace = build_trace(&[("a", vec![1.0])]);

    let err = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect_err("should fail on length mismatch");
    match err {
        ReftestError::TraceLengthMismatch {
            reference,
            candidate,
        } => {
            assert_eq!(reference, 3);
            assert_eq!(candidate, 1);
        }
        other => panic!("expected TraceLengthMismatch, got {other:?}"),
    }
}

// ===========================================================================
// 7. Single-layer trace (boundary case)
// ===========================================================================

#[test]
fn test_single_layer_trace_passes() {
    let trace = build_trace(&[("only_layer", vec![1.0, 2.0, 3.0])]);
    let report =
        compare_traces(&trace, &trace, &ComparisonConfig::default()).expect("should succeed");
    assert!(report.all_passed);
    assert_eq!(report.layers.len(), 1);
    assert_eq!(report.layers[0].name, "only_layer");
}

#[test]
fn test_single_layer_trace_fails() {
    let ref_trace = build_trace(&[("loss", vec![0.5])]);
    let cand_trace = build_trace(&[("loss", vec![100.0])]);

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("should succeed");
    assert!(!report.all_passed);
    assert_eq!(report.first_failure, Some(0));
}

// ===========================================================================
// 8. assert_traces_match! macro integration
// ===========================================================================

#[test]
fn test_assert_traces_match_macro_passes() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace
        .checkpoint("layer1", &[1.0, 2.0, 3.0], &[3])
        .expect("valid");
    let mut cand_trace = ReferenceTrace::new();
    cand_trace
        .checkpoint("layer1", &[1.0, 2.0, 3.0], &[3])
        .expect("valid");

    // Should not panic.
    crate::assert_traces_match!(cand_trace, ref_trace);
}

#[test]
fn test_assert_traces_match_macro_with_custom_tolerance() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace
        .checkpoint("out", &[1.0, 2.0], &[2])
        .expect("valid");
    let mut cand_trace = ReferenceTrace::new();
    cand_trace
        .checkpoint("out", &[1.001, 2.001], &[2])
        .expect("valid");

    // Default tolerance would fail, but custom abs=0.01, rel=0.01 passes.
    crate::assert_traces_match!(cand_trace, ref_trace, abs = 0.01, rel = 0.01);
}

#[test]
#[should_panic(expected = "Tensor mismatch")]
fn test_assert_traces_match_macro_panics_on_divergence() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace
        .checkpoint("divergent", &[1.0], &[1])
        .expect("valid");
    let mut cand_trace = ReferenceTrace::new();
    cand_trace
        .checkpoint("divergent", &[999.0], &[1])
        .expect("valid");

    crate::assert_traces_match!(cand_trace, ref_trace);
}
