// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for nn-reftest: reference tensor comparison.

use nn_reftest::{
    assert_traces_match, compare_tensors, compare_traces, ComparisonConfig, NamedTensor,
    ReferenceTrace,
};

// ---- Trace comparison tests ----

#[test]
fn test_identical_traces_pass() {
    let mut reference = ReferenceTrace::new();
    reference
        .checkpoint("encoder.conv1", &[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("valid");
    reference
        .checkpoint("encoder.relu1", &[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("valid");
    reference
        .checkpoint("encoder.pool1", &[2.5, 3.5], &[2])
        .expect("valid");

    let candidate = reference.clone();

    let report = compare_traces(&reference, &candidate, &ComparisonConfig::default())
        .expect("comparison should succeed");

    assert!(report.all_passed);
    assert_eq!(report.layers.len(), 3);
    assert!(report.first_failure.is_none());
}

#[test]
fn test_assert_macro_passes_on_identical() {
    let mut reference = ReferenceTrace::new();
    reference
        .checkpoint("x", &[1.0, 2.0, 3.0], &[3])
        .expect("valid");

    let candidate = reference.clone();

    assert_traces_match!(candidate, reference);
}

#[test]
fn test_assert_macro_panics_on_divergence() {
    let mut reference = ReferenceTrace::new();
    reference
        .checkpoint("x", &[1.0, 2.0, 3.0], &[3])
        .expect("valid");

    let mut candidate = ReferenceTrace::new();
    candidate
        .checkpoint("x", &[1.0, 2.0, 4.0], &[3])
        .expect("valid"); // large divergence

    let result = std::panic::catch_unwind(|| {
        assert_traces_match!(candidate, reference);
    });
    let err = result.expect_err("assert_traces_match! should panic on divergent tensors");
    let msg = err
        .downcast_ref::<String>()
        .map(String::as_str)
        .unwrap_or_else(|| {
            err.downcast_ref::<&str>()
                .copied()
                .unwrap_or("unknown panic")
        });
    assert!(
        msg.contains("Tensor mismatch at layer 'x'"),
        "expected mismatch message, got: {msg}"
    );
}

#[test]
fn test_assert_macro_custom_tolerance() {
    let mut reference = ReferenceTrace::new();
    reference
        .checkpoint("x", &[1.0, 2.0, 3.0], &[3])
        .expect("valid");

    let mut candidate = ReferenceTrace::new();
    candidate
        .checkpoint("x", &[1.001, 2.001, 3.001], &[3])
        .expect("valid");

    // Default tolerance would fail, relaxed passes.
    assert_traces_match!(candidate, reference, abs = 0.01, rel = 0.01);
}

#[test]
fn test_assert_macro_epsilon_alias() {
    let mut reference = ReferenceTrace::new();
    reference
        .checkpoint("x", &[1.0, 2.0, 3.0], &[3])
        .expect("valid");

    let mut candidate = ReferenceTrace::new();
    candidate
        .checkpoint("x", &[1.0001, 2.0001, 3.0001], &[3])
        .expect("valid");

    assert_traces_match!(candidate, reference, epsilon = 0.001);
}

// ---- Divergence detection tests ----

#[test]
fn test_first_failure_identifies_correct_layer() {
    let mut reference = ReferenceTrace::new();
    reference
        .checkpoint("layer0", &[1.0, 2.0], &[2])
        .expect("valid");
    reference
        .checkpoint("layer1", &[3.0, 4.0], &[2])
        .expect("valid");
    reference
        .checkpoint("layer2", &[5.0, 6.0], &[2])
        .expect("valid");

    let mut candidate = ReferenceTrace::new();
    candidate
        .checkpoint("layer0", &[1.0, 2.0], &[2])
        .expect("valid"); // matches
    candidate
        .checkpoint("layer1", &[3.0, 5.0], &[2])
        .expect("valid"); // diverges here
    candidate
        .checkpoint("layer2", &[5.0, 7.0], &[2])
        .expect("valid"); // also diverges

    let report = compare_traces(&reference, &candidate, &ComparisonConfig::default())
        .expect("comparison should succeed");

    assert!(!report.all_passed);
    assert_eq!(report.first_failure, Some(1));
    assert!(report.layers[0].passed);
    assert!(!report.layers[1].passed);
}

// ---- Cosine similarity edge cases ----

#[test]
fn test_parallel_vectors_high_cosine() {
    let a = NamedTensor::new("x", vec![3], vec![1.0, 2.0, 3.0]).expect("valid test tensor");
    let b = NamedTensor::new("x", vec![3], vec![2.0, 4.0, 6.0]).expect("valid test tensor"); // 2*a

    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.9999);

    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        (result.cosine_similarity - 1.0).abs() < 1e-6,
        "parallel vectors should have cos~1.0"
    );
}

#[test]
fn test_antiparallel_vectors() {
    let a = NamedTensor::new("x", vec![3], vec![1.0, 2.0, 3.0]).expect("valid test tensor");
    let b = NamedTensor::new("x", vec![3], vec![-1.0, -2.0, -3.0]).expect("valid test tensor");

    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");

    assert!((result.cosine_similarity - (-1.0)).abs() < 1e-6);
    assert!(!result.passed);
}

// ---- Summary formatting ----

#[test]
fn test_divergence_report_summary_contains_layer_names() {
    let mut reference = ReferenceTrace::new();
    reference.checkpoint("conv1", &[1.0], &[1]).expect("valid");
    reference.checkpoint("relu1", &[2.0], &[1]).expect("valid");

    let mut candidate = ReferenceTrace::new();
    candidate.checkpoint("conv1", &[1.0], &[1]).expect("valid");
    candidate.checkpoint("relu1", &[3.0], &[1]).expect("valid"); // diverges

    let report = compare_traces(&reference, &candidate, &ComparisonConfig::default())
        .expect("should succeed");

    let summary = report.summary();
    assert!(summary.contains("conv1"), "summary should mention conv1");
    assert!(summary.contains("relu1"), "summary should mention relu1");
    assert!(summary.contains("FAIL"), "summary should contain FAIL");
    assert!(
        summary.contains("First failure"),
        "summary should mention first failure"
    );
}

// ---- Safetensors roundtrip (via load module) ----

fn f32_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[test]
fn test_safetensors_roundtrip_comparison() {
    // Build a safetensors buffer with known data.
    let data_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let data_b: Vec<f32> = vec![5.0, 6.0];

    let byte_data_a = f32_to_le_bytes(&data_a);
    let byte_data_b = f32_to_le_bytes(&data_b);

    let view_a =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2], &byte_data_a)
            .expect("valid view");
    let view_b =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2], &byte_data_b)
            .expect("valid view");

    let serialized = safetensors::tensor::serialize(
        vec![
            ("layer_a".to_string(), view_a),
            ("layer_b".to_string(), view_b),
        ],
        None,
    )
    .expect("serialization should succeed");

    // Load back as a trace.
    let loaded =
        nn_reftest::load_safetensors_from_bytes(&serialized).expect("loading should succeed");

    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded.get(0).expect("exists").name, "layer_a");
    assert_eq!(
        loaded.get(0).expect("exists").data,
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(loaded.get(1).expect("exists").name, "layer_b");
    assert_eq!(loaded.get(1).expect("exists").data, vec![5.0, 6.0]);

    // Compare against itself — should pass.
    let candidate = loaded.clone();
    assert_traces_match!(candidate, loaded);
}
