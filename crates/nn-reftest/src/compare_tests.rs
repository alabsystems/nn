// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for tensor comparison engine.

use super::*;
use crate::trace::NamedTensor;

fn make_tensor(name: &str, data: Vec<f32>) -> NamedTensor {
    let len = data.len();
    NamedTensor::new(name, vec![len], data).expect("valid test tensor")
}

#[test]
fn test_identical_tensors_pass() {
    let a = make_tensor("x", vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![1.0, 2.0, 3.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");

    assert!(result.passed);
    assert_eq!(result.max_abs_diff, 0.0);
    assert_eq!(result.mean_abs_diff, 0.0);
    assert!((result.cosine_similarity - 1.0).abs() < 1e-7);
}

#[test]
fn test_small_difference_passes() {
    let a = make_tensor("x", vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![1.0 + 1e-7, 2.0 - 1e-7, 3.0 + 1e-7]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");

    assert!(result.passed);
    assert!(result.max_abs_diff < 1e-5);
}

#[test]
fn test_large_difference_fails() {
    let a = make_tensor("x", vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![1.1, 2.0, 3.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");

    assert!(!result.passed);
    assert!((result.max_abs_diff - 0.1).abs() < 1e-7);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = make_tensor("x", vec![1.0, 0.0]);
    let b = make_tensor("x", vec![0.0, 1.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");

    assert!(!result.passed);
    assert!(result.cosine_similarity.abs() < 1e-7);
}

#[test]
fn test_shape_mismatch_error() {
    let a = NamedTensor::new("x", vec![2, 3], vec![0.0; 6]).expect("valid test tensor");
    let b = NamedTensor::new("x", vec![3, 2], vec![0.0; 6]).expect("valid test tensor");

    let err = compare_tensors(&a, &b, &ComparisonConfig::default())
        .expect_err("should fail on shape mismatch");
    assert!(matches!(err, ReftestError::ShapeMismatch { .. }));
}

#[test]
fn test_empty_tensor_error() {
    let a = NamedTensor::new("x", vec![0], vec![]).expect("valid test tensor");
    let b = NamedTensor::new("x", vec![0], vec![]).expect("valid test tensor");

    let err = compare_tensors(&a, &b, &ComparisonConfig::default())
        .expect_err("should fail on empty tensor");
    assert!(matches!(err, ReftestError::EmptyTensor(_)));
}

#[test]
fn test_compare_traces_matching() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace
        .checkpoint("layer1", &[1.0, 2.0], &[2])
        .expect("valid");
    ref_trace
        .checkpoint("layer2", &[3.0, 4.0, 5.0], &[3])
        .expect("valid");

    let mut cand_trace = ReferenceTrace::new();
    cand_trace
        .checkpoint("layer1", &[1.0, 2.0], &[2])
        .expect("valid");
    cand_trace
        .checkpoint("layer2", &[3.0, 4.0, 5.0], &[3])
        .expect("valid");

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("comparison should succeed");

    assert!(report.all_passed);
    assert!(report.first_failure.is_none());
    assert_eq!(report.layers.len(), 2);
}

#[test]
fn test_compare_traces_first_failure() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace
        .checkpoint("layer1", &[1.0, 2.0], &[2])
        .expect("valid");
    ref_trace
        .checkpoint("layer2", &[3.0, 4.0], &[2])
        .expect("valid");

    let mut cand_trace = ReferenceTrace::new();
    cand_trace
        .checkpoint("layer1", &[1.0, 2.0], &[2])
        .expect("valid");
    cand_trace
        .checkpoint("layer2", &[3.0, 5.0], &[2])
        .expect("valid"); // diverges here

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("comparison should succeed");

    assert!(!report.all_passed);
    assert_eq!(report.first_failure, Some(1));
}

#[test]
fn test_compare_traces_length_mismatch() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace.checkpoint("a", &[1.0], &[1]).expect("valid");

    let cand_trace = ReferenceTrace::new();

    let err = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect_err("should fail on length mismatch");
    assert!(matches!(err, ReftestError::TraceLengthMismatch { .. }));
}

#[test]
fn test_strict_config_rejects_normal_tolerance() {
    let a = make_tensor("x", vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![1.0 + 5e-6, 2.0, 3.0]);

    // Default passes
    let default_result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");
    assert!(default_result.passed);

    // Strict fails
    let strict_result =
        compare_tensors(&a, &b, &ComparisonConfig::strict()).expect("comparison should succeed");
    assert!(!strict_result.passed);
}

#[test]
fn test_relative_error_for_large_values() {
    // Large values with small absolute diff but acceptable relative diff.
    let a = make_tensor("x", vec![1000.0, 2000.0]);
    let b = make_tensor("x", vec![1000.01, 2000.01]);

    let config = ComparisonConfig {
        abs_tolerance: 0.1,
        rel_tolerance: 1e-4,
        cosine_threshold: 0.9999,
        ..ComparisonConfig::default()
    };
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(result.passed);
}

#[test]
fn test_display_layer_comparison() {
    let comp = LayerComparison {
        name: "conv1".to_string(),
        shape: vec![2, 3],
        max_abs_diff: 1e-5,
        mean_abs_diff: 5e-6,
        cosine_similarity: 0.999_999,
        max_rel_diff: 1e-4,
        num_elements: 6,
        rms_diff: 3e-6,
        peak_amplitude: 5.0,
        passed: true,
        #[cfg(feature = "spectral")]
        spectral: None,
    };
    let s = format!("{comp}");
    assert!(s.contains("PASS"));
    assert!(s.contains("conv1"));
    assert!(s.contains("rms="));
    assert!(s.contains("peak="));
}

// ---- NaN/Inf handling (IEEE 754 compliance) ----

#[test]
fn test_nan_in_candidate_fails_with_infinity_metrics() {
    let a = make_tensor("x", vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![1.0, f32::NAN, 3.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");

    assert!(!result.passed, "NaN elements must cause failure");
    assert!(
        result.max_abs_diff.is_infinite(),
        "NaN elements should report infinite max_abs_diff, got {}",
        result.max_abs_diff
    );
    assert!(
        result.max_rel_diff.is_infinite(),
        "NaN elements should report infinite max_rel_diff"
    );
}

#[test]
fn test_nan_in_reference_fails() {
    let a = make_tensor("x", vec![f32::NAN, 2.0]);
    let b = make_tensor("x", vec![1.0, 2.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");

    assert!(!result.passed, "NaN in reference must cause failure");
}

#[test]
fn test_inf_in_both_fails() {
    let a = make_tensor("x", vec![1.0, f32::INFINITY]);
    let b = make_tensor("x", vec![1.0, f32::INFINITY]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");

    assert!(!result.passed, "Inf elements must cause failure");
}

#[test]
fn test_all_nan_cosine_is_nan_not_one() {
    let a = make_tensor("x", vec![f32::NAN, f32::NAN]);
    let b = make_tensor("x", vec![f32::NAN, f32::NAN]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");

    assert!(!result.passed, "all-NaN tensors must fail");
    assert!(
        result.cosine_similarity.is_nan(),
        "all-NaN should report NaN cosine, not 1.0; got {}",
        result.cosine_similarity
    );
}

// ---- Zero-vector cosine similarity ----

#[test]
fn test_both_zero_vectors_cosine_is_one() {
    let a = make_tensor("x", vec![0.0, 0.0, 0.0]);
    let b = make_tensor("x", vec![0.0, 0.0, 0.0]);
    let config = ComparisonConfig {
        abs_tolerance: 1.0,
        rel_tolerance: 1.0,
        cosine_threshold: 0.0,
        ..ComparisonConfig::default()
    };
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(
        (result.cosine_similarity - 1.0).abs() < 1e-7,
        "both zero vectors should have cosine similarity 1.0, got {}",
        result.cosine_similarity,
    );
    assert!(result.passed);
}

#[test]
fn test_one_zero_vector_cosine_is_zero() {
    let a = make_tensor("x", vec![0.0, 0.0]);
    let b = make_tensor("x", vec![1.0, 2.0]);
    let config = ComparisonConfig {
        abs_tolerance: 100.0,
        rel_tolerance: 100.0,
        cosine_threshold: 0.5,
        ..ComparisonConfig::default()
    };
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(
        result.cosine_similarity.abs() < 1e-7,
        "one zero vector should have cosine similarity 0.0, got {}",
        result.cosine_similarity,
    );
    assert!(!result.passed, "cosine 0 < threshold 0.5 should fail");
}

// ---- Negative infinity ----

#[test]
fn test_neg_inf_in_candidate_fails() {
    let a = make_tensor("x", vec![1.0, 2.0]);
    let b = make_tensor("x", vec![1.0, f32::NEG_INFINITY]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");
    assert!(
        !result.passed,
        "NEG_INFINITY in candidate must cause failure"
    );
    assert!(
        result.max_abs_diff.is_infinite(),
        "NEG_INFINITY should report infinite abs diff"
    );
    assert!(
        result.peak_amplitude.is_infinite(),
        "NEG_INFINITY should produce infinite peak amplitude"
    );
}

#[test]
fn test_neg_inf_in_reference_fails() {
    let a = make_tensor("x", vec![f32::NEG_INFINITY, 2.0]);
    let b = make_tensor("x", vec![1.0, 2.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");
    assert!(
        !result.passed,
        "NEG_INFINITY in reference must cause failure"
    );
}

// ---- DivergenceReport summary ----

#[test]
fn test_divergence_report_summary_all_passed() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace.checkpoint("a", &[1.0], &[1]).expect("valid");
    ref_trace.checkpoint("b", &[2.0], &[1]).expect("valid");

    let report = compare_traces(&ref_trace, &ref_trace, &ComparisonConfig::default())
        .expect("comparison should succeed");
    let summary = report.summary();
    assert!(
        summary.contains("All 2 layers passed"),
        "summary should report all passed: {summary}"
    );
}

#[test]
fn test_divergence_report_summary_with_failure() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace.checkpoint("a", &[1.0], &[1]).expect("valid");
    ref_trace.checkpoint("b", &[2.0], &[1]).expect("valid");

    let mut cand_trace = ReferenceTrace::new();
    cand_trace.checkpoint("a", &[1.0], &[1]).expect("valid");
    cand_trace.checkpoint("b", &[999.0], &[1]).expect("valid");

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("comparison should succeed");
    let summary = report.summary();
    assert!(
        summary.contains("First failure at layer 1"),
        "summary should report first failure: {summary}"
    );
    assert!(
        summary.contains("'b'"),
        "summary should include failing layer name: {summary}"
    );
}

// ---- Relaxed config ----

#[test]
fn test_relaxed_config_passes_moderate_differences() {
    let a = make_tensor("x", vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![1.005, 2.005, 3.005]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::relaxed()).expect("comparison should succeed");
    assert!(
        result.passed,
        "relaxed config should pass moderate differences"
    );
}

#[test]
fn test_relaxed_vs_strict_boundary() {
    let a = make_tensor("x", vec![1.0, 2.0, 3.0]);
    // Difference of ~5e-4: passes relaxed (atol=1e-2), fails strict (atol=1e-6).
    let b = make_tensor("x", vec![1.0005, 2.0005, 3.0005]);

    let relaxed =
        compare_tensors(&a, &b, &ComparisonConfig::relaxed()).expect("comparison should succeed");
    let strict =
        compare_tensors(&a, &b, &ComparisonConfig::strict()).expect("comparison should succeed");
    assert!(relaxed.passed, "relaxed should pass");
    assert!(!strict.passed, "strict should fail");
}

// ---- Config builder methods ----

#[test]
fn test_config_new_sets_fields() {
    let config = ComparisonConfig::new(0.1, 0.2, 0.99);
    assert_eq!(config.abs_tolerance, 0.1);
    assert_eq!(config.rel_tolerance, 0.2);
    assert_eq!(config.cosine_threshold, 0.99);
    assert!(config.rms_tolerance.is_none());
    assert!(config.peak_amplitude_limit.is_none());
}

#[test]
fn test_config_builder_chain() {
    let config = ComparisonConfig::new(1e-5, 1e-4, 0.9999)
        .with_rms_tolerance(1e-3)
        .with_peak_amplitude_limit(100.0);
    assert_eq!(config.rms_tolerance, Some(1e-3));
    assert_eq!(config.peak_amplitude_limit, Some(100.0));
}

// ---- Comparison with single element tensors ----

#[test]
fn test_single_element_identical() {
    let a = make_tensor("x", vec![42.0]);
    let b = make_tensor("x", vec![42.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");
    assert!(result.passed);
    assert_eq!(result.num_elements, 1);
    assert_eq!(result.max_abs_diff, 0.0);
    assert!((result.cosine_similarity - 1.0).abs() < 1e-7);
}

#[test]
fn test_single_element_different() {
    let a = make_tensor("x", vec![1.0]);
    let b = make_tensor("x", vec![2.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");
    assert!(!result.passed);
    assert!((result.max_abs_diff - 1.0).abs() < f32::EPSILON);
}

// ---- Mixed NaN and valid elements ----

#[test]
fn test_mixed_nan_valid_still_computes_cosine() {
    // One NaN among otherwise valid and similar values.
    let a = make_tensor("x", vec![1.0, f32::NAN, 3.0, 4.0]);
    let b = make_tensor("x", vec![1.0, 2.0, 3.0, 4.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");
    assert!(!result.passed);
    assert!(result.max_abs_diff.is_infinite());
    // Cosine should still be computed from finite elements (1,3,4 dot 1,3,4).
    // But the NaN in ref is skipped, so cosine is only from finite pairs.
    // It should still be a real number (not NaN) since there are finite elements.
    // Actually: the implementation adds to dot/norm only for finite pairs, so
    // cosine is computed from {1.0, 3.0, 4.0} dot {1.0, 3.0, 4.0} = 26.
    assert!(
        result.cosine_similarity.is_finite(),
        "cosine should be finite when some elements are valid, got {}",
        result.cosine_similarity
    );
}

// ---- LayerComparison Display ----

#[test]
fn test_layer_comparison_display_fail() {
    let comp = LayerComparison {
        name: "broken_layer".to_string(),
        shape: vec![10],
        max_abs_diff: 1.0,
        mean_abs_diff: 0.5,
        cosine_similarity: 0.5,
        max_rel_diff: 0.8,
        num_elements: 10,
        rms_diff: 0.7,
        peak_amplitude: 100.0,
        passed: false,
        #[cfg(feature = "spectral")]
        spectral: None,
    };
    let s = format!("{comp}");
    assert!(s.contains("FAIL"), "display should show FAIL: {s}");
    assert!(s.contains("broken_layer"), "display should show name: {s}");
}

// ---- Trace comparison with shape mismatch in one layer ----

#[test]
fn test_compare_traces_shape_mismatch_returns_error() {
    let ref_tensor = NamedTensor::new("layer1", vec![2, 3], vec![0.0; 6]).expect("valid");
    let cand_tensor = NamedTensor::new("layer1", vec![3, 2], vec![0.0; 6]).expect("valid");

    let ref_trace = ReferenceTrace::from_checkpoints(vec![ref_tensor]);
    let cand_trace = ReferenceTrace::from_checkpoints(vec![cand_tensor]);

    let err = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect_err("should fail on shape mismatch");
    assert!(matches!(err, ReftestError::ShapeMismatch { .. }));
}

// ---- Empty trace comparison ----

#[test]
fn test_compare_empty_traces_passes() {
    let ref_trace = ReferenceTrace::new();
    let cand_trace = ReferenceTrace::new();

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("comparing empty traces should succeed");
    assert!(report.all_passed);
    assert!(report.first_failure.is_none());
    assert!(report.layers.is_empty());
}

// -- Gate tests: RMS, peak amplitude, near-zero relative tolerance --
// (extracted to compare_tests_gates.rs)
#[path = "compare_tests_gates.rs"]
mod gates;
