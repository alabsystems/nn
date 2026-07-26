// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the converter feedback loop parity diagnostics.
//!
//! Part of #4349.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;

static PARITY_FIXTURE_COUNTER: AtomicUsize = AtomicUsize::new(0);

// ---------------------------------------------------------------------------
// ParityMetric computation
// ---------------------------------------------------------------------------

#[test]
fn test_compute_parity_metric_identical() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let metric = compute_parity_metric(&data, &data).unwrap();
    assert!(
        (metric.cosine_similarity - 1.0).abs() < 1e-10,
        "identical vectors should have cosine=1.0, got {}",
        metric.cosine_similarity
    );
    assert!(
        metric.max_abs_diff < 1e-10,
        "identical vectors should have max_abs=0, got {}",
        metric.max_abs_diff
    );
    assert!(
        metric.rms_diff < 1e-10,
        "identical vectors should have rms=0, got {}",
        metric.rms_diff
    );
    assert_eq!(metric.element_count, 5);
}

#[test]
fn test_compute_parity_metric_close() {
    let reference = vec![1.0, 2.0, 3.0, 4.0];
    let candidate = vec![1.001, 2.001, 3.001, 4.001];
    let metric = compute_parity_metric(&candidate, &reference).unwrap();
    assert!(
        metric.cosine_similarity > 0.9999,
        "close vectors should have high cosine, got {}",
        metric.cosine_similarity
    );
    assert!(
        (metric.max_abs_diff - 0.001).abs() < 1e-6,
        "expected max_abs ~0.001, got {}",
        metric.max_abs_diff
    );
    assert!(metric.rms_diff < 0.002);
}

#[test]
fn test_compute_parity_metric_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let metric = compute_parity_metric(&a, &b).unwrap();
    assert!(
        metric.cosine_similarity.abs() < 1e-10,
        "orthogonal vectors should have cosine~0, got {}",
        metric.cosine_similarity
    );
}

#[test]
fn test_compute_parity_metric_opposite() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![-1.0, -2.0, -3.0];
    let metric = compute_parity_metric(&a, &b).unwrap();
    assert!(
        (metric.cosine_similarity - (-1.0)).abs() < 1e-10,
        "opposite vectors should have cosine=-1.0, got {}",
        metric.cosine_similarity
    );
}

#[test]
fn test_compute_parity_metric_length_mismatch() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0, 2.0, 3.0];
    assert!(compute_parity_metric(&a, &b).is_none());
}

#[test]
fn test_compute_parity_metric_empty() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    assert!(compute_parity_metric(&a, &b).is_none());
}

#[test]
fn test_compute_parity_metric_zeros() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![0.0, 0.0, 0.0];
    let metric = compute_parity_metric(&a, &b).unwrap();
    // Zero vectors: cosine is defined as 1.0 (convention for zero denominator).
    assert!(
        (metric.cosine_similarity - 1.0).abs() < 1e-10,
        "zero vectors should have cosine=1.0 by convention, got {}",
        metric.cosine_similarity
    );
    assert!(metric.max_abs_diff < 1e-10);
}

// ---------------------------------------------------------------------------
// ParityThresholds defaults
// ---------------------------------------------------------------------------

#[test]
fn test_parity_thresholds_defaults() {
    let t = ParityThresholds::default();
    assert!((t.cosine_min - 0.999).abs() < 1e-10);
    assert!((t.max_abs_max - 0.02).abs() < 1e-10);
    assert!((t.rms_max - 0.001).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// L3 numerical parity checks
// ---------------------------------------------------------------------------

#[test]
fn test_numerical_parity_pass() {
    let reference = vec![1.0, 2.0, 3.0, 4.0];
    let candidate = vec![1.0001, 2.0001, 3.0001, 4.0001];
    let thresholds = ParityThresholds::default();

    let checks = check_numerical_parity(&[("output", &candidate, &reference)], &thresholds);

    assert_eq!(checks.len(), 1);
    assert_eq!(checks[0].status, CheckStatus::Passed);
    assert!(checks[0].metric.is_some());
}

#[test]
fn test_numerical_parity_fail_cosine() {
    // Large divergence -> low cosine.
    let reference = vec![1.0, 0.0, 0.0];
    let candidate = vec![0.0, 1.0, 0.0];
    let thresholds = ParityThresholds::default();

    let checks = check_numerical_parity(&[("output", &candidate, &reference)], &thresholds);

    assert_eq!(checks.len(), 1);
    assert_eq!(checks[0].status, CheckStatus::Failed);
    let detail = checks[0].detail.as_ref().unwrap();
    assert!(detail.contains("cosine"), "should mention cosine: {detail}");
}

#[test]
fn test_numerical_parity_fail_max_abs() {
    let reference = vec![1.0, 2.0, 3.0];
    let candidate = vec![1.0, 2.0, 3.1]; // max_abs = 0.1 > 0.02
    let thresholds = ParityThresholds::default();

    let checks = check_numerical_parity(&[("output", &candidate, &reference)], &thresholds);

    assert_eq!(checks.len(), 1);
    assert_eq!(checks[0].status, CheckStatus::Failed);
    let detail = checks[0].detail.as_ref().unwrap();
    assert!(
        detail.contains("max_abs"),
        "should mention max_abs: {detail}"
    );
}

#[test]
fn test_numerical_parity_shape_mismatch() {
    let reference = vec![1.0, 2.0];
    let candidate = vec![1.0, 2.0, 3.0];
    let thresholds = ParityThresholds::default();

    let checks = check_numerical_parity(&[("output", &candidate, &reference)], &thresholds);

    assert_eq!(checks.len(), 1);
    assert_eq!(checks[0].status, CheckStatus::Failed);
    let detail = checks[0].detail.as_ref().unwrap();
    assert!(
        detail.contains("shape mismatch"),
        "should mention shape mismatch: {detail}"
    );
}

#[test]
fn test_numerical_parity_multiple_outputs() {
    let ref1 = vec![1.0, 2.0, 3.0];
    let cand1 = vec![1.0001, 2.0001, 3.0001];
    let ref2 = vec![10.0, 20.0];
    let cand2 = vec![10.0, 20.0];
    let thresholds = ParityThresholds::default();

    let checks = check_numerical_parity(
        &[("out_a", &cand1, &ref1), ("out_b", &cand2, &ref2)],
        &thresholds,
    );

    assert_eq!(checks.len(), 2);
    assert_eq!(checks[0].status, CheckStatus::Passed);
    assert_eq!(checks[1].status, CheckStatus::Passed);
    assert!(checks[0].name.contains("out_a"));
    assert!(checks[1].name.contains("out_b"));
}

// ---------------------------------------------------------------------------
// L0 structural checks (using real ImportedGraph from fixtures)
// ---------------------------------------------------------------------------

/// Import the MLP fixture used in convert_tests.rs.
fn import_mlp_fixture() -> ImportedGraph {
    use std::collections::HashMap;

    let id = PARITY_FIXTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("nn_parity_mlp_{}_{}", std::process::id(), id));
    std::fs::create_dir_all(&dir).unwrap();

    let graph_path = dir.join("graph.json");
    std::fs::write(&graph_path, include_str!("../test_data/e2e_mlp.json")).unwrap();

    // Write MLP weights: fc1 [8,4], fc1.bias [8], fc2 [3,8], fc2.bias [3].
    let fc1_w: Vec<u8> = (0..32)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let fc1_b: Vec<u8> = [0.0f32; 8].iter().flat_map(|f| f.to_le_bytes()).collect();
    let fc2_w: Vec<u8> = (0..24)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let fc2_b: Vec<u8> = [0.0f32; 3].iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "fc1.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8, 4], &fc1_w).unwrap(),
    );
    tensors.insert(
        "fc1.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8], &fc1_b).unwrap(),
    );
    tensors.insert(
        "fc2.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3, 8], &fc2_w).unwrap(),
    );
    tensors.insert(
        "fc2.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3], &fc2_b).unwrap(),
    );
    let weights_path = dir.join("weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&weights_path, &serialized).unwrap();

    let imported = crate::import_model(&graph_path, &weights_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    imported
}

#[test]
fn test_l0_structure_mlp_all_pass() {
    let imported = import_mlp_fixture();
    let expectation = StructuralExpectation {
        expected_op_count: Some(imported.graph.len()),
        expected_input_names: Some(vec!["x".to_string()]),
        expected_output_names: Some(vec!["linear_1".to_string()]),
    };

    let checks = check_structure(&imported, &expectation);
    for check in &checks {
        assert_eq!(
            check.status,
            CheckStatus::Passed,
            "check '{}' should pass: {:?}",
            check.name,
            check.detail
        );
    }
}

#[test]
fn test_l0_structure_wrong_op_count() {
    let imported = import_mlp_fixture();
    let expectation = StructuralExpectation {
        expected_op_count: Some(999),
        expected_input_names: None,
        expected_output_names: None,
    };

    let checks = check_structure(&imported, &expectation);
    let op_check = checks.iter().find(|c| c.name == "op_count_match").unwrap();
    assert_eq!(op_check.status, CheckStatus::Failed);
    assert!(op_check.detail.as_ref().unwrap().contains("999"));
}

#[test]
fn test_l0_structure_wrong_input_names() {
    let imported = import_mlp_fixture();
    let expectation = StructuralExpectation {
        expected_op_count: None,
        expected_input_names: Some(vec!["wrong_name".to_string()]),
        expected_output_names: None,
    };

    let checks = check_structure(&imported, &expectation);
    let name_check = checks
        .iter()
        .find(|c| c.name == "input_names_match")
        .unwrap();
    assert_eq!(name_check.status, CheckStatus::Failed);
}

#[test]
fn test_l0_structure_wrong_output_names() {
    let imported = import_mlp_fixture();
    let expectation = StructuralExpectation {
        expected_op_count: None,
        expected_input_names: None,
        expected_output_names: Some(vec!["wrong_output".to_string()]),
    };

    let checks = check_structure(&imported, &expectation);
    let name_check = checks
        .iter()
        .find(|c| c.name == "output_names_match")
        .unwrap();
    assert_eq!(name_check.status, CheckStatus::Failed);
}

#[test]
fn test_l0_structure_no_expectations() {
    let imported = import_mlp_fixture();
    let expectation = StructuralExpectation::default();

    let checks = check_structure(&imported, &expectation);
    // Should have basic checks (non_empty, has_inputs, has_output, has_compute)
    // but no expectation-specific checks.
    assert!(checks.len() >= 4);
    for check in &checks {
        assert_eq!(
            check.status,
            CheckStatus::Passed,
            "basic check '{}' should pass",
            check.name
        );
    }
}

// ---------------------------------------------------------------------------
// Full verify_parity()
// ---------------------------------------------------------------------------

#[test]
fn test_verify_parity_l0_only() {
    let imported = import_mlp_fixture();
    let expectation = StructuralExpectation::default();
    let thresholds = ParityThresholds::default();

    let report = verify_parity(&imported, "test_mlp", &expectation, None, &thresholds);

    assert!(report.overall_pass, "L0-only report should pass");
    assert_eq!(report.model_name, "test_mlp");
    // L3 should be skipped.
    let l3_skipped = report
        .checks
        .iter()
        .any(|c| c.level == ParityLevel::NumericalParity && c.status == CheckStatus::Skipped);
    assert!(l3_skipped, "L3 should be skipped when no reference");
}

#[test]
fn test_verify_parity_l0_plus_l3_pass() {
    let imported = import_mlp_fixture();
    let expectation = StructuralExpectation::default();
    let thresholds = ParityThresholds::default();

    let reference = vec![1.0, 2.0, 3.0];
    let candidate = vec![1.0001, 2.0001, 3.0001];

    let report = verify_parity(
        &imported,
        "test_mlp",
        &expectation,
        Some(&[("output", &candidate, &reference)]),
        &thresholds,
    );

    assert!(report.overall_pass, "L0+L3 should pass");
    assert!(report.failures().is_empty());
}

#[test]
fn test_verify_parity_l3_failure() {
    let imported = import_mlp_fixture();
    let expectation = StructuralExpectation::default();
    let thresholds = ParityThresholds::default();

    // Large divergence: cosine ~ 0 and max_abs >> threshold.
    let reference = vec![1.0, 0.0, 0.0];
    let candidate = vec![0.0, 100.0, 0.0];

    let report = verify_parity(
        &imported,
        "test_mlp",
        &expectation,
        Some(&[("output", &candidate, &reference)]),
        &thresholds,
    );

    assert!(!report.overall_pass, "should fail with large divergence");
    let failures = report.failures();
    assert!(!failures.is_empty());
    assert!(failures[0].name.contains("numerical_parity"));
}

#[test]
fn test_verify_parity_custom_thresholds() {
    let imported = import_mlp_fixture();
    let expectation = StructuralExpectation::default();

    // Very loose thresholds: cosine > 0.0, max_abs < 1000.
    let thresholds = ParityThresholds {
        cosine_min: 0.0,
        max_abs_max: 1000.0,
        rms_max: 1000.0,
    };

    let reference = vec![1.0, 0.0, 0.0];
    let candidate = vec![0.0, 100.0, 0.0];

    let report = verify_parity(
        &imported,
        "test_mlp",
        &expectation,
        Some(&[("output", &candidate, &reference)]),
        &thresholds,
    );

    assert!(report.overall_pass, "should pass with loose thresholds");
}

#[test]
fn test_parity_report_print() {
    let imported = import_mlp_fixture();
    let expectation = StructuralExpectation::default();
    let thresholds = ParityThresholds::default();

    let report = verify_parity(&imported, "test_mlp", &expectation, None, &thresholds);

    // print() should not panic.
    report.print();
}

#[test]
fn test_parity_report_failures_empty_when_passing() {
    let checks = vec![
        ParityCheck::passed("a", ParityLevel::Structure),
        ParityCheck::skipped("b", ParityLevel::NumericalParity, "no data"),
    ];
    let report = ParityReport::new("test".to_string(), checks);
    assert!(report.overall_pass);
    assert!(report.failures().is_empty());
}

#[test]
fn test_parity_report_failures_nonempty_when_failing() {
    let checks = vec![
        ParityCheck::passed("a", ParityLevel::Structure),
        ParityCheck::failed("b", ParityLevel::NumericalParity, "bad"),
    ];
    let report = ParityReport::new("test".to_string(), checks);
    assert!(!report.overall_pass);
    assert_eq!(report.failures().len(), 1);
}

#[test]
fn test_check_status_constructors() {
    let p = ParityCheck::passed("test", ParityLevel::Structure);
    assert_eq!(p.status, CheckStatus::Passed);
    assert!(p.detail.is_none());

    let f = ParityCheck::failed("test", ParityLevel::Bounds, "error msg");
    assert_eq!(f.status, CheckStatus::Failed);
    assert_eq!(f.detail.as_deref(), Some("error msg"));

    let s = ParityCheck::skipped("test", ParityLevel::KernelSafety, "not available");
    assert_eq!(s.status, CheckStatus::Skipped);
    assert_eq!(s.detail.as_deref(), Some("not available"));
}
