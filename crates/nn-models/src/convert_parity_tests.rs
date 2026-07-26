// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for converter feedback loop Phase 1: ParityReport + verify_parity().

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::ComputationGraph;
use nn_core::{Device, DynTensor};

use super::*;

// ---------------------------------------------------------------------------
// L0 structure checks
// ---------------------------------------------------------------------------

#[test]
fn test_parity_report_structure_pass() {
    use nn_core::dyn_tensor::trace::trace_graph;

    // Trace a simple graph to get a non-empty ComputationGraph.
    let (_, graph) = trace_graph(|| {
        let a = DynTensor::zeros(&[2, 4], nn_core::DType::F32, &Device::Cpu)?;
        let b = DynTensor::ones(&[2, 4], nn_core::DType::F32, &Device::Cpu)?;
        let c = a.add(&b)?;
        Ok(c)
    })
    .expect("trace_graph should succeed");

    let model = ConvertedModel::from_imported(
        graph,
        1,
        vec!["input".to_string()],
        vec!["output".to_string()],
        HashMap::new(),
        "test-model",
    );

    let report = model.verify_parity(None, None, None);
    assert!(report.overall_pass, "L0 checks should pass for valid model");
    assert_eq!(report.model_name, "test-model");

    // All 3 structure checks should pass.
    let structure_checks: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.level == ParityLevel::Structure)
        .collect();
    assert_eq!(structure_checks.len(), 3);
    assert!(structure_checks.iter().all(|c| c.passed));
}

#[test]
fn test_parity_report_structure_fail_empty_graph() {
    let model = ConvertedModel::new(
        ComputationGraph::from_nodes(vec![]),
        HashMap::new(),
        0,
        vec![],
        vec![],
        "empty".to_string(),
    );

    let report = model.verify_parity(None, None, None);
    assert!(!report.overall_pass, "empty model should fail L0");

    let failed: Vec<_> = report.checks.iter().filter(|c| !c.passed).collect();
    assert_eq!(failed.len(), 3, "all 3 structure checks should fail");
}

// ---------------------------------------------------------------------------
// Threshold defaults
// ---------------------------------------------------------------------------

#[test]
fn test_parity_thresholds_default() {
    let t = ParityThresholds::default();
    assert!((t.cosine_min - 0.999).abs() < 1e-12);
    assert!((t.max_abs_max - 0.02).abs() < 1e-12);
    assert!((t.rms_max - 0.001).abs() < 1e-12);
    assert!((t.bounds_width_ratio - 1.2).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Numerical helper functions
// ---------------------------------------------------------------------------

#[test]
fn test_cosine_similarity_identical() {
    let v = vec![1.0_f32, 2.0, 3.0, 4.0];
    let cos = cosine_similarity(&v, &v);
    assert!(
        (cos - 1.0).abs() < 1e-10,
        "identical vectors should have cosine 1.0, got {cos}"
    );
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0_f32, 0.0];
    let b = vec![0.0_f32, 1.0];
    let cos = cosine_similarity(&a, &b);
    assert!(
        cos.abs() < 1e-10,
        "orthogonal vectors should have cosine 0.0, got {cos}"
    );
}

#[test]
fn test_cosine_similarity_opposite() {
    let a = vec![1.0_f32, 2.0, 3.0];
    let b = vec![-1.0_f32, -2.0, -3.0];
    let cos = cosine_similarity(&a, &b);
    assert!(
        (cos + 1.0).abs() < 1e-10,
        "opposite vectors should have cosine -1.0, got {cos}"
    );
}

#[test]
fn test_cosine_similarity_zero_vector() {
    let a = vec![0.0_f32, 0.0, 0.0];
    let b = vec![1.0_f32, 2.0, 3.0];
    let cos = cosine_similarity(&a, &b);
    assert!(
        cos.abs() < 1e-10,
        "zero vector should yield cosine 0.0, got {cos}"
    );
}

#[test]
fn test_max_abs_diff_identical() {
    let v = vec![1.0_f32, 2.0, 3.0];
    let d = max_abs_diff(&v, &v);
    assert!(
        d.abs() < 1e-10,
        "identical vectors: max_abs should be 0, got {d}"
    );
}

#[test]
fn test_rms_diff_identical() {
    let v = vec![1.0_f32, 2.0, 3.0];
    let d = rms_diff(&v, &v);
    assert!(
        d.abs() < 1e-10,
        "identical vectors: rms should be 0, got {d}"
    );
}

#[test]
fn test_max_abs_diff_known() {
    let a = vec![1.0_f32, 5.0, 3.0];
    let b = vec![1.0_f32, 2.0, 3.0];
    let d = max_abs_diff(&a, &b);
    assert!((d - 3.0).abs() < 1e-10, "expected max_abs 3.0, got {d}");
}

#[test]
fn test_rms_diff_known() {
    // a - b = [0, 3, 0] => sum_sq = 9, mean = 3, sqrt = 1.732...
    let a = vec![1.0_f32, 5.0, 3.0];
    let b = vec![1.0_f32, 2.0, 3.0];
    let d = rms_diff(&a, &b);
    let expected = (9.0_f64 / 3.0).sqrt();
    assert!(
        (d - expected).abs() < 1e-10,
        "expected rms {expected}, got {d}"
    );
}

// ---------------------------------------------------------------------------
// L3 numerical parity (full pipeline)
// ---------------------------------------------------------------------------

#[test]
fn test_parity_report_numerical_match() {
    use nn_core::dyn_tensor::trace::trace_graph;

    let (_, graph) = trace_graph(|| {
        let a = DynTensor::zeros(&[2, 4], nn_core::DType::F32, &Device::Cpu)?;
        let b = DynTensor::ones(&[2, 4], nn_core::DType::F32, &Device::Cpu)?;
        let c = a.add(&b)?;
        Ok(c)
    })
    .expect("trace_graph");

    let model = ConvertedModel::from_imported(
        graph,
        1,
        vec!["input".to_string()],
        vec!["output".to_string()],
        HashMap::new(),
        "numerical-match",
    );

    let data = vec![1.0_f32; 8];
    let refs: HashMap<String, Vec<f32>> =
        [("output".to_string(), data.clone())].into_iter().collect();
    let actuals: HashMap<String, Vec<f32>> = [("output".to_string(), data)].into_iter().collect();

    let report = model.verify_parity(Some(&refs), Some(&actuals), None);
    assert!(report.overall_pass, "identical outputs should pass parity");

    let l3_checks: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.level == ParityLevel::NumericalParity)
        .collect();
    assert_eq!(l3_checks.len(), 1);
    assert!(l3_checks[0].passed);
    let metric = l3_checks[0].metric.as_ref().expect("should have metric");
    assert!((metric.cosine_similarity - 1.0).abs() < 1e-10);
    assert!(metric.max_abs_diff.abs() < 1e-10);
    assert!(metric.rms_diff.abs() < 1e-10);
    assert_eq!(metric.element_count, 8);
}

#[test]
fn test_parity_report_numerical_mismatch() {
    use nn_core::dyn_tensor::trace::trace_graph;

    let (_, graph) = trace_graph(|| {
        let a = DynTensor::zeros(&[2, 4], nn_core::DType::F32, &Device::Cpu)?;
        let b = DynTensor::ones(&[2, 4], nn_core::DType::F32, &Device::Cpu)?;
        let c = a.add(&b)?;
        Ok(c)
    })
    .expect("trace_graph");

    let model = ConvertedModel::from_imported(
        graph,
        1,
        vec!["input".to_string()],
        vec!["output".to_string()],
        HashMap::new(),
        "numerical-mismatch",
    );

    // Reference is all 1s, actual is all 100s — wildly different.
    let ref_data = vec![1.0_f32; 8];
    let actual_data = vec![100.0_f32; 8];
    let refs: HashMap<String, Vec<f32>> = [("output".to_string(), ref_data)].into_iter().collect();
    let actuals: HashMap<String, Vec<f32>> =
        [("output".to_string(), actual_data)].into_iter().collect();

    let report = model.verify_parity(Some(&refs), Some(&actuals), None);
    assert!(
        !report.overall_pass,
        "wildly different outputs should fail parity"
    );

    let l3_checks: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.level == ParityLevel::NumericalParity)
        .collect();
    assert_eq!(l3_checks.len(), 1);
    assert!(!l3_checks[0].passed);
    assert!(l3_checks[0].error.is_some());
    let metric = l3_checks[0].metric.as_ref().expect("should have metric");
    assert!(metric.max_abs_diff > 90.0, "max_abs should be large");
}
