// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended reference testing pipeline tests for nn-reftest.
//!
//! Covers: safetensors reference loading, NPY format loading, tolerance
//! computation, per-element comparison, trace comparison, statistical
//! comparison, shape/dtype mismatch detection, batch comparison, and
//! report generation.

use crate::compare::{compare_tensors, compare_traces, ComparisonConfig, DivergenceReport};
use crate::error::ReftestError;
use crate::load::load_safetensors_from_bytes;
use crate::npy::{read_npy_from_bytes, write_npy_to_bytes, NpyDType};
use crate::presets::TolerancePreset;
use crate::tolerance::{compare_with_tolerance, ToleranceStrategy};
use crate::trace::{NamedTensor, ReferenceTrace};

// ---------------------------------------------------------------------------
// Helper: build a minimal safetensors byte buffer from f32 tensors.
// ---------------------------------------------------------------------------

fn f32_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn build_safetensors(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
    let byte_bufs: Vec<Vec<u8>> = tensors
        .iter()
        .map(|&(_, _, data)| f32_to_le_bytes(data))
        .collect();
    let mut tensor_map: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();

    for (i, &(name, shape, _)) in tensors.iter().enumerate() {
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            shape.to_vec(),
            &byte_bufs[i],
        )
        .expect("valid tensor view");
        tensor_map.push((name.to_string(), view));
    }

    safetensors::tensor::serialize(tensor_map, None).expect("serialization should succeed")
}

// ===========================================================================
// 1. Safetensors reference loading: tensors loaded with correct shapes
// ===========================================================================

#[test]
fn test_safetensors_single_tensor_shape_preserved() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let bytes = build_safetensors(&[("weight", &[3, 4], &data)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    assert_eq!(trace.len(), 1);
    let t = trace.get(0).unwrap();
    assert_eq!(t.name, "weight");
    assert_eq!(t.shape, vec![3, 4]);
    assert_eq!(t.numel(), 12);
    assert_eq!(t.data[0], 0.0);
    assert_eq!(t.data[11], 11.0);
}

#[test]
fn test_safetensors_multiple_tensors_sorted() {
    let bytes = build_safetensors(&[
        ("z_bias", &[2], &[10.0, 20.0]),
        ("a_weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ("m_embed", &[4], &[0.1, 0.2, 0.3, 0.4]),
    ]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    assert_eq!(trace.len(), 3);
    let names: Vec<&str> = trace.names().collect();
    assert_eq!(names, vec!["a_weight", "m_embed", "z_bias"]);
}

#[test]
fn test_safetensors_scalar_tensor() {
    let bytes = build_safetensors(&[("loss", &[], &[0.5])]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    let t = trace.get(0).unwrap();
    assert!(t.shape.is_empty());
    assert_eq!(t.numel(), 1);
    assert_eq!(t.data[0], 0.5);
}

#[test]
fn test_safetensors_high_rank_tensor() {
    let data: Vec<f32> = (0..120).map(|i| i as f32 * 0.01).collect();
    let bytes = build_safetensors(&[("conv", &[2, 3, 4, 5], &data)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    let t = trace.get(0).unwrap();
    assert_eq!(t.shape, vec![2, 3, 4, 5]);
    assert_eq!(t.numel(), 120);
}

#[test]
fn test_safetensors_f16_dtype_loading() {
    // Build an f16 safetensors buffer manually.
    let f16_vals = [half::f16::from_f32(1.0),
        half::f16::from_f32(2.0),
        half::f16::from_f32(3.0)];
    let byte_data: Vec<u8> = f16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![3], &byte_data)
        .expect("valid view");
    let serialized = safetensors::tensor::serialize(vec![("f16_tensor".to_string(), view)], None)
        .expect("serialize");

    let trace = load_safetensors_from_bytes(&serialized).expect("load f16 should succeed");
    let t = trace.get(0).unwrap();
    assert_eq!(t.shape, vec![3]);
    assert!((t.data[0] - 1.0).abs() < 0.01);
    assert!((t.data[2] - 3.0).abs() < 0.01);
}

#[test]
fn test_safetensors_bf16_dtype_loading() {
    let bf16_vals = [half::bf16::from_f32(0.5), half::bf16::from_f32(-1.5)];
    let byte_data: Vec<u8> = bf16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![2], &byte_data)
        .expect("valid view");
    let serialized = safetensors::tensor::serialize(vec![("bf16_tensor".to_string(), view)], None)
        .expect("serialize");

    let trace = load_safetensors_from_bytes(&serialized).expect("load bf16 should succeed");
    let t = trace.get(0).unwrap();
    assert_eq!(t.shape, vec![2]);
    assert!((t.data[0] - 0.5).abs() < 0.1);
    assert!((t.data[1] - (-1.5)).abs() < 0.1);
}

// ===========================================================================
// 2. NPY format loading: numpy arrays imported correctly
// ===========================================================================

#[test]
fn test_npy_roundtrip_1d() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let shape = &[5];
    let npy_bytes = write_npy_to_bytes(&data, shape).expect("write should succeed");
    let tensor = read_npy_from_bytes(&npy_bytes).expect("read should succeed");

    assert_eq!(tensor.shape, vec![5]);
    assert_eq!(tensor.data, data);
    assert_eq!(tensor.dtype, NpyDType::F32);
}

#[test]
fn test_npy_roundtrip_2d() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let shape = &[3, 4];
    let npy_bytes = write_npy_to_bytes(&data, shape).expect("write should succeed");
    let tensor = read_npy_from_bytes(&npy_bytes).expect("read should succeed");

    assert_eq!(tensor.shape, vec![3, 4]);
    assert_eq!(tensor.numel(), 12);
}

#[test]
fn test_npy_roundtrip_scalar() {
    let data = vec![42.0f32];
    let shape: &[usize] = &[];
    let npy_bytes = write_npy_to_bytes(&data, shape).expect("write should succeed");
    let tensor = read_npy_from_bytes(&npy_bytes).expect("read should succeed");

    assert!(tensor.shape.is_empty());
    assert_eq!(tensor.data, vec![42.0]);
}

#[test]
fn test_npy_load_from_bytes_as_trace() {
    let data = vec![1.0f32, 2.0, 3.0];
    let npy_bytes = write_npy_to_bytes(&data, &[3]).expect("write should succeed");
    let trace = crate::load_npy_from_bytes(&npy_bytes, "nn_tensor").expect("load should succeed");

    assert_eq!(trace.len(), 1);
    let t = trace.get(0).unwrap();
    assert_eq!(t.name, "nn_tensor");
    assert_eq!(t.data, data);
}

#[test]
fn test_npy_invalid_magic_bytes() {
    let garbage = vec![0u8; 64];
    let result = read_npy_from_bytes(&garbage);
    assert!(result.is_err());
}

#[test]
fn test_npy_dtype_f32_descriptor() {
    assert_eq!(NpyDType::from_descr("<f4"), Some(NpyDType::F32));
    assert_eq!(NpyDType::from_descr(">f4"), Some(NpyDType::F32));
}

#[test]
fn test_npy_dtype_f16_descriptor() {
    assert_eq!(NpyDType::from_descr("<f2"), Some(NpyDType::F16));
    assert_eq!(NpyDType::from_descr(">f2"), Some(NpyDType::F16));
}

#[test]
fn test_npy_dtype_f64_descriptor() {
    assert_eq!(NpyDType::from_descr("<f8"), Some(NpyDType::F64));
}

#[test]
fn test_npy_dtype_integer_descriptors() {
    assert_eq!(NpyDType::from_descr("<i4"), Some(NpyDType::I32));
    assert_eq!(NpyDType::from_descr("<i8"), Some(NpyDType::I64));
    assert_eq!(NpyDType::from_descr("|u1"), Some(NpyDType::U8));
}

#[test]
fn test_npy_dtype_unknown_returns_none() {
    assert_eq!(NpyDType::from_descr("<c16"), None);
    assert_eq!(NpyDType::from_descr("garbage"), None);
}

// ===========================================================================
// 3. Tolerance computation: relative and absolute tolerance
// ===========================================================================

#[test]
fn test_tolerance_absolute_pass() {
    let actual = [1.0f32, 2.0, 3.0];
    let expected = [1.0001, 2.0001, 3.0001];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 1e-3 },
    )
    .expect("comparison should succeed");

    assert!(result.passed);
    assert!(result.max_diff < 1e-3);
    assert_eq!(result.num_mismatches, 0);
}

#[test]
fn test_tolerance_absolute_fail() {
    let actual = [1.0f32, 2.0, 3.0];
    let expected = [1.1, 2.0, 3.0];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 1e-3 },
    )
    .expect("comparison should succeed");

    assert!(!result.passed);
    assert_eq!(result.num_mismatches, 1);
    assert_eq!(result.worst_index, 0);
}

#[test]
fn test_tolerance_relative_pass() {
    let actual = [100.0f32, 200.0, 300.0];
    let expected = [100.001, 200.002, 300.003];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Relative { rtol: 1e-4 },
    )
    .expect("comparison should succeed");

    assert!(result.passed);
}

#[test]
fn test_tolerance_relative_fail() {
    let actual = [1.0f32, 2.0];
    let expected = [1.5, 2.0]; // 50% relative error
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Relative { rtol: 0.01 },
    )
    .expect("comparison should succeed");

    assert!(!result.passed);
    assert_eq!(result.num_mismatches, 1);
}

#[test]
fn test_tolerance_mixed_numpy_style() {
    let actual = [0.0f32, 100.0];
    let expected = [1e-6, 100.01];
    // NumPy: |a - b| <= atol + rtol * |b|
    // Element 0: |0.0 - 1e-6| = 1e-6 <= 1e-5 + 0.01 * 1e-6 = pass
    // Element 1: |100 - 100.01| = 0.01 <= 1e-5 + 0.01 * 100.01 = 1.00015 = pass
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Mixed {
            atol: 1e-5,
            rtol: 0.01,
        },
    )
    .expect("comparison should succeed");

    assert!(result.passed);
}

#[test]
fn test_tolerance_ulp_adjacent_floats() {
    let a = 1.0f32;
    let b = f32::from_bits(a.to_bits() + 1); // next representable float
    let result = compare_with_tolerance(&[a], &[b], &ToleranceStrategy::ULP { max_ulps: 1 })
        .expect("comparison should succeed");

    assert!(result.passed);
    assert_eq!(result.num_mismatches, 0);
}

#[test]
fn test_tolerance_ulp_far_apart() {
    let result = compare_with_tolerance(
        &[1.0f32],
        &[2.0f32],
        &ToleranceStrategy::ULP { max_ulps: 1 },
    )
    .expect("comparison should succeed");

    assert!(!result.passed);
}

#[test]
fn test_tolerance_percent_close() {
    let actual = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let expected = [1.0, 2.0, 3.0, 4.0, 100.0]; // 1 out of 5 is off
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::PercentClose {
            threshold: 0.1,
            percent: 80.0,
        },
    )
    .expect("comparison should succeed");

    // 4/5 = 80% are within threshold
    assert!(result.passed);
}

#[test]
fn test_tolerance_percent_close_fail() {
    let actual = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let expected = [100.0, 200.0, 3.0, 4.0, 5.0]; // 2 out of 5 off
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::PercentClose {
            threshold: 0.1,
            percent: 80.0,
        },
    )
    .expect("comparison should succeed");

    // 3/5 = 60% close, need 80%
    assert!(!result.passed);
}

// ===========================================================================
// 4. Per-element comparison: max absolute error tracking
// ===========================================================================

#[test]
fn test_compare_tensors_max_abs_diff_tracked() {
    let reference = NamedTensor::new("layer", vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let candidate = NamedTensor::new("layer", vec![4], vec![1.0, 2.001, 3.0, 4.005]).unwrap();

    let config = ComparisonConfig::new(0.01, 1.0, 0.0);
    let cmp = compare_tensors(&reference, &candidate, &config).expect("compare should succeed");

    // max_abs_diff should be ~0.005 (from element index 3)
    assert!((cmp.max_abs_diff - 0.005).abs() < 1e-6);
    assert!(cmp.passed);
}

#[test]
fn test_compare_tensors_mean_abs_diff() {
    let reference = NamedTensor::new("layer", vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let candidate = NamedTensor::new("layer", vec![4], vec![1.1, 2.1, 3.1, 4.1]).unwrap();

    let config = ComparisonConfig::new(1.0, 1.0, 0.0);
    let cmp = compare_tensors(&reference, &candidate, &config).expect("compare should succeed");

    assert!((cmp.mean_abs_diff - 0.1).abs() < 1e-5);
}

#[test]
fn test_compare_tensors_rms_diff() {
    let reference = NamedTensor::new("layer", vec![4], vec![0.0, 0.0, 0.0, 0.0]).unwrap();
    let candidate = NamedTensor::new("layer", vec![4], vec![1.0, 1.0, 1.0, 1.0]).unwrap();

    let config = ComparisonConfig::new(10.0, 10.0, 0.0);
    let cmp = compare_tensors(&reference, &candidate, &config).expect("compare should succeed");

    // RMS of [1, 1, 1, 1] diffs = sqrt(4/4) = 1.0
    assert!((cmp.rms_diff - 1.0).abs() < 1e-5);
}

#[test]
fn test_compare_tensors_peak_amplitude() {
    let reference = NamedTensor::new("layer", vec![3], vec![0.0, 0.0, 0.0]).unwrap();
    let candidate = NamedTensor::new("layer", vec![3], vec![1.0, -5.0, 3.0]).unwrap();

    let config = ComparisonConfig::new(100.0, 100.0, 0.0);
    let cmp = compare_tensors(&reference, &candidate, &config).expect("compare should succeed");

    assert!((cmp.peak_amplitude - 5.0).abs() < 1e-5);
}

#[test]
fn test_compare_tensors_nan_triggers_infinite_divergence() {
    let reference = NamedTensor::new("layer", vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let candidate = NamedTensor::new("layer", vec![3], vec![1.0, f32::NAN, 3.0]).unwrap();

    let config = ComparisonConfig::default();
    let cmp = compare_tensors(&reference, &candidate, &config).expect("compare should succeed");

    assert!(!cmp.passed);
    assert!(cmp.max_abs_diff.is_infinite());
}

#[test]
fn test_compare_tensors_infinity_triggers_divergence() {
    let reference = NamedTensor::new("layer", vec![2], vec![1.0, 2.0]).unwrap();
    let candidate = NamedTensor::new("layer", vec![2], vec![1.0, f32::INFINITY]).unwrap();

    let config = ComparisonConfig::default();
    let cmp = compare_tensors(&reference, &candidate, &config).expect("compare should succeed");

    assert!(!cmp.passed);
    assert!(cmp.max_abs_diff.is_infinite());
    assert!(cmp.peak_amplitude.is_infinite());
}

// ===========================================================================
// 5. Trace comparison: layer-by-layer output matching
// ===========================================================================

#[test]
fn test_trace_comparison_all_pass() {
    let mut reference = ReferenceTrace::new();
    reference
        .checkpoint("layer1", &[1.0, 2.0, 3.0], &[3])
        .unwrap();
    reference.checkpoint("layer2", &[4.0, 5.0], &[2]).unwrap();

    let mut candidate = ReferenceTrace::new();
    candidate
        .checkpoint("layer1", &[1.0, 2.0, 3.0], &[3])
        .unwrap();
    candidate.checkpoint("layer2", &[4.0, 5.0], &[2]).unwrap();

    let config = ComparisonConfig::default();
    let report = compare_traces(&reference, &candidate, &config).expect("compare should succeed");

    assert!(report.all_passed);
    assert!(report.first_failure.is_none());
    assert_eq!(report.layers.len(), 2);
}

#[test]
fn test_trace_comparison_first_failure_index() {
    let mut reference = ReferenceTrace::new();
    reference.checkpoint("layer1", &[1.0, 2.0], &[2]).unwrap();
    reference.checkpoint("layer2", &[3.0, 4.0], &[2]).unwrap();
    reference.checkpoint("layer3", &[5.0, 6.0], &[2]).unwrap();

    let mut candidate = ReferenceTrace::new();
    candidate.checkpoint("layer1", &[1.0, 2.0], &[2]).unwrap();
    candidate.checkpoint("layer2", &[30.0, 40.0], &[2]).unwrap(); // diverges here
    candidate.checkpoint("layer3", &[50.0, 60.0], &[2]).unwrap();

    let config = ComparisonConfig::strict();
    let report = compare_traces(&reference, &candidate, &config).expect("compare should succeed");

    assert!(!report.all_passed);
    assert_eq!(report.first_failure, Some(1));
    assert!(report.layers[0].passed);
    assert!(!report.layers[1].passed);
}

#[test]
fn test_trace_comparison_length_mismatch() {
    let mut reference = ReferenceTrace::new();
    reference.checkpoint("a", &[1.0], &[1]).unwrap();

    let mut candidate = ReferenceTrace::new();
    candidate.checkpoint("a", &[1.0], &[1]).unwrap();
    candidate.checkpoint("b", &[2.0], &[1]).unwrap();

    let config = ComparisonConfig::default();
    let result = compare_traces(&reference, &candidate, &config);
    assert!(matches!(
        result,
        Err(ReftestError::TraceLengthMismatch {
            reference: 1,
            candidate: 2
        })
    ));
}

#[test]
fn test_trace_comparison_empty_traces() {
    let reference = ReferenceTrace::new();
    let candidate = ReferenceTrace::new();
    let config = ComparisonConfig::default();
    let report = compare_traces(&reference, &candidate, &config).expect("compare should succeed");

    assert!(report.all_passed);
    assert_eq!(report.layers.len(), 0);
}

#[test]
fn test_trace_comparison_cosine_similarity_identical() {
    let mut reference = ReferenceTrace::new();
    reference.checkpoint("a", &[1.0, 0.0, 0.0], &[3]).unwrap();

    let mut candidate = ReferenceTrace::new();
    candidate.checkpoint("a", &[1.0, 0.0, 0.0], &[3]).unwrap();

    let config = ComparisonConfig::default();
    let report = compare_traces(&reference, &candidate, &config).expect("compare should succeed");

    assert!((report.layers[0].cosine_similarity - 1.0).abs() < 1e-6);
}

#[test]
fn test_trace_comparison_cosine_similarity_orthogonal() {
    let mut reference = ReferenceTrace::new();
    reference.checkpoint("a", &[1.0, 0.0], &[2]).unwrap();

    let mut candidate = ReferenceTrace::new();
    candidate.checkpoint("a", &[0.0, 1.0], &[2]).unwrap();

    let config = ComparisonConfig::new(10.0, 10.0, 0.5);
    let report = compare_traces(&reference, &candidate, &config).expect("compare should succeed");

    // Orthogonal vectors have cosine similarity ~0
    assert!(report.layers[0].cosine_similarity.abs() < 1e-5);
    assert!(!report.layers[0].passed); // cos 0.0 < threshold 0.5
}

// ===========================================================================
// 6. Statistical comparison: mean/std/percentile-like matching
// ===========================================================================

#[test]
fn test_compare_tensors_uniform_offset() {
    // All elements offset by the same amount
    let n = 100;
    let reference_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let offset = 0.001;
    let candidate_data: Vec<f32> = reference_data.iter().map(|&v| v + offset).collect();

    let reference = NamedTensor::new("uniform", vec![n], reference_data).unwrap();
    let candidate = NamedTensor::new("uniform", vec![n], candidate_data).unwrap();

    let config = ComparisonConfig::new(0.01, 1.0, 0.0);
    let cmp = compare_tensors(&reference, &candidate, &config).expect("compare should succeed");

    assert!(cmp.passed);
    // Mean abs diff should be ~offset
    assert!((cmp.mean_abs_diff - offset).abs() < 1e-5);
    // Max abs diff should also be ~offset (uniform)
    assert!((cmp.max_abs_diff - offset).abs() < 1e-5);
}

#[test]
fn test_compare_tensors_single_outlier_detected() {
    let reference_data = vec![1.0f32; 100];
    let mut candidate_data = vec![1.0f32; 100];
    // Single outlier at index 50
    candidate_data[50] = 100.0;

    let reference = NamedTensor::new("outlier", vec![100], reference_data).unwrap();
    let candidate = NamedTensor::new("outlier", vec![100], candidate_data).unwrap();

    let config = ComparisonConfig::new(0.01, 0.01, 0.0);
    let cmp = compare_tensors(&reference, &candidate, &config).expect("compare should succeed");

    assert!(!cmp.passed);
    assert!((cmp.max_abs_diff - 99.0).abs() < 1e-3);
    // mean_abs_diff should be ~0.99 (99.0 / 100)
    assert!((cmp.mean_abs_diff - 0.99).abs() < 0.01);
}

#[test]
fn test_tolerance_percent_close_statistical_outlier() {
    // 95% of elements match, 5% are outliers
    let n = 100;
    let actual = vec![1.0f32; n];
    let mut expected = vec![1.0f32; n];
    for e in expected.iter_mut().take(5) {
        *e = 100.0;
    }

    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::PercentClose {
            threshold: 0.1,
            percent: 90.0,
        },
    )
    .expect("comparison should succeed");

    // 95/100 = 95% close >= 90%
    assert!(result.passed);
    assert_eq!(result.num_mismatches, 5);
}

#[test]
fn test_compare_tensors_rms_gate_active() {
    let reference = NamedTensor::new("rms_test", vec![4], vec![0.0, 0.0, 0.0, 0.0]).unwrap();
    let candidate = NamedTensor::new("rms_test", vec![4], vec![0.5, 0.5, 0.5, 0.5]).unwrap();

    // Abs/rel gates pass but RMS gate should fail
    let config = ComparisonConfig::new(1.0, 1.0, 0.0).with_rms_tolerance(0.1);
    let cmp = compare_tensors(&reference, &candidate, &config).expect("compare should succeed");

    // RMS = 0.5, threshold = 0.1
    assert!(!cmp.passed);
    assert!((cmp.rms_diff - 0.5).abs() < 1e-5);
}

#[test]
fn test_compare_tensors_peak_amplitude_gate() {
    let reference = NamedTensor::new("peak_test", vec![3], vec![0.0, 0.0, 0.0]).unwrap();
    let candidate = NamedTensor::new("peak_test", vec![3], vec![0.1, 0.1, 1000.0]).unwrap();

    let config = ComparisonConfig::new(10000.0, 10000.0, 0.0).with_peak_amplitude_limit(100.0);
    let cmp = compare_tensors(&reference, &candidate, &config).expect("compare should succeed");

    assert!(!cmp.passed);
    assert!((cmp.peak_amplitude - 1000.0).abs() < 1e-2);
}

// ===========================================================================
// 7. Shape mismatch detection: different shapes reported correctly
// ===========================================================================

#[test]
fn test_compare_tensors_shape_mismatch_detected() {
    let reference = NamedTensor::new("layer", vec![2, 3], vec![1.0; 6]).unwrap();
    let candidate = NamedTensor::new("layer", vec![3, 2], vec![1.0; 6]).unwrap();

    let config = ComparisonConfig::default();
    let result = compare_tensors(&reference, &candidate, &config);
    match result {
        Err(ReftestError::ShapeMismatch {
            name,
            expected,
            actual,
        }) => {
            assert_eq!(name, "layer");
            assert_eq!(expected, vec![2, 3]);
            assert_eq!(actual, vec![3, 2]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn test_compare_tensors_rank_mismatch_detected() {
    let reference = NamedTensor::new("layer", vec![6], vec![1.0; 6]).unwrap();
    let candidate = NamedTensor::new("layer", vec![2, 3], vec![1.0; 6]).unwrap();

    let config = ComparisonConfig::default();
    let result = compare_tensors(&reference, &candidate, &config);
    assert!(matches!(result, Err(ReftestError::ShapeMismatch { .. })));
}

#[test]
fn test_compare_tensors_empty_tensor_error() {
    let reference = NamedTensor::new("empty", vec![0], vec![]).unwrap();
    let candidate = NamedTensor::new("empty", vec![0], vec![]).unwrap();

    let config = ComparisonConfig::default();
    let result = compare_tensors(&reference, &candidate, &config);
    assert!(matches!(result, Err(ReftestError::EmptyTensor(_))));
}

#[test]
fn test_named_tensor_shape_element_count_mismatch() {
    let result = NamedTensor::new("bad", vec![2, 3], vec![1.0, 2.0]);
    match result {
        Err(ReftestError::ElementCountMismatch {
            name,
            shape,
            expected,
            actual,
        }) => {
            assert_eq!(name, "bad");
            assert_eq!(shape, vec![2, 3]);
            assert_eq!(expected, 6);
            assert_eq!(actual, 2);
        }
        other => panic!("expected ElementCountMismatch, got {other:?}"),
    }
}

#[test]
fn test_trace_shape_mismatch_propagates_in_trace_comparison() {
    let mut reference = ReferenceTrace::new();
    reference.checkpoint("layer1", &[1.0, 2.0], &[2]).unwrap();

    let mut candidate = ReferenceTrace::new();
    candidate
        .checkpoint("layer1", &[1.0, 2.0, 3.0], &[3])
        .unwrap();

    let config = ComparisonConfig::default();
    let result = compare_traces(&reference, &candidate, &config);
    assert!(matches!(result, Err(ReftestError::ShapeMismatch { .. })));
}

// ===========================================================================
// 8. DType mismatch detection: different dtypes reported
// ===========================================================================

#[test]
fn test_safetensors_unsupported_dtype_error() {
    // Build a safetensors file with BOOL dtype
    let result = crate::load::convert_to_f32(&[], safetensors::Dtype::BOOL, &[1], "bool_tensor");
    assert!(matches!(result, Err(ReftestError::UnsupportedDtype(_))));
}

#[test]
fn test_safetensors_i32_dtype_error() {
    let result =
        crate::load::convert_to_f32(&[0u8; 4], safetensors::Dtype::I32, &[1], "i32_tensor");
    assert!(matches!(result, Err(ReftestError::UnsupportedDtype(_))));
}

#[test]
fn test_npy_unsupported_dtype_conversion() {
    // Directly test the conversion with an unsupported dtype
    let result = crate::npy::convert::convert_npy_to_f32(&[], "<c16", 1);
    assert!(matches!(result, Err(ReftestError::NpyUnsupportedDtype(_))));
}

#[test]
fn test_npy_dtype_to_descr_roundtrip() {
    for dtype in &[
        NpyDType::F16,
        NpyDType::F32,
        NpyDType::F64,
        NpyDType::I32,
        NpyDType::I64,
        NpyDType::U8,
    ] {
        let descr = dtype.to_descr();
        let parsed = NpyDType::from_descr(descr);
        assert_eq!(parsed, Some(*dtype), "roundtrip failed for {dtype:?}");
    }
}

#[test]
fn test_npy_dtype_display() {
    assert_eq!(format!("{}", NpyDType::F32), "<f4");
    assert_eq!(format!("{}", NpyDType::F16), "<f2");
    assert_eq!(format!("{}", NpyDType::F64), "<f8");
    assert_eq!(format!("{}", NpyDType::U8), "|u1");
}

// ===========================================================================
// 9. Batch comparison: multiple reference/actual pairs
// ===========================================================================

#[test]
fn test_batch_comparison_multiple_layers() {
    let mut reference = ReferenceTrace::new();
    let mut candidate = ReferenceTrace::new();

    for i in 0..10 {
        let data: Vec<f32> = (0..16).map(|j| (i * 16 + j) as f32 * 0.01).collect();
        let name = format!("layer_{i}");
        reference.checkpoint(&name, &data, &[4, 4]).unwrap();
        // Add a small perturbation to the candidate. 1e-7 keeps the relative
        // error within the default rel_tolerance (1e-4) even for the smallest
        // non-near-zero value (0.01): 1e-7 / 0.01 = 1e-5. A 1e-6 perturbation
        // would push 0.01's relative error to ~1.0001e-4, just over tolerance.
        let perturbed: Vec<f32> = data.iter().map(|&v| v + 1e-7).collect();
        candidate.checkpoint(&name, &perturbed, &[4, 4]).unwrap();
    }

    let config = ComparisonConfig::default();
    let report = compare_traces(&reference, &candidate, &config).expect("compare should succeed");

    assert!(report.all_passed);
    assert_eq!(report.layers.len(), 10);
    // Every layer should have very small diff
    for layer in &report.layers {
        assert!(layer.max_abs_diff < 1e-5);
        assert!(layer.passed);
    }
}

#[test]
fn test_batch_comparison_divergence_at_specific_layer() {
    let mut reference = ReferenceTrace::new();
    let mut candidate = ReferenceTrace::new();

    for i in 0..5 {
        let data: Vec<f32> = vec![i as f32; 4];
        let name = format!("layer_{i}");
        reference.checkpoint(&name, &data, &[4]).unwrap();

        if i == 3 {
            // Inject divergence at layer 3
            let bad_data: Vec<f32> = vec![999.0; 4];
            candidate.checkpoint(&name, &bad_data, &[4]).unwrap();
        } else {
            candidate.checkpoint(&name, &data, &[4]).unwrap();
        }
    }

    let config = ComparisonConfig::strict();
    let report = compare_traces(&reference, &candidate, &config).expect("compare should succeed");

    assert!(!report.all_passed);
    assert_eq!(report.first_failure, Some(3));
    // Layers 0-2 should pass, layer 3+ should fail
    assert!(report.layers[0].passed);
    assert!(report.layers[1].passed);
    assert!(report.layers[2].passed);
    assert!(!report.layers[3].passed);
}

#[test]
fn test_batch_comparison_with_different_shapes_per_layer() {
    let mut reference = ReferenceTrace::new();
    let mut candidate = ReferenceTrace::new();

    reference
        .checkpoint("embed", &[1.0, 2.0, 3.0], &[3])
        .unwrap();
    reference
        .checkpoint("hidden", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    reference.checkpoint("output", &[1.0, 2.0], &[2]).unwrap();

    candidate
        .checkpoint("embed", &[1.0, 2.0, 3.0], &[3])
        .unwrap();
    candidate
        .checkpoint("hidden", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    candidate.checkpoint("output", &[1.0, 2.0], &[2]).unwrap();

    let config = ComparisonConfig::default();
    let report = compare_traces(&reference, &candidate, &config).expect("compare should succeed");

    assert!(report.all_passed);
    assert_eq!(report.layers[0].shape, vec![3]);
    assert_eq!(report.layers[1].shape, vec![2, 3]);
    assert_eq!(report.layers[2].shape, vec![2]);
}

// ===========================================================================
// 10. Report generation: comparison results formatted correctly
// ===========================================================================

#[test]
fn test_divergence_report_summary_all_passed() {
    let mut reference = ReferenceTrace::new();
    reference.checkpoint("a", &[1.0], &[1]).unwrap();
    reference.checkpoint("b", &[2.0], &[1]).unwrap();

    let mut candidate = ReferenceTrace::new();
    candidate.checkpoint("a", &[1.0], &[1]).unwrap();
    candidate.checkpoint("b", &[2.0], &[1]).unwrap();

    let config = ComparisonConfig::default();
    let report = compare_traces(&reference, &candidate, &config).expect("compare should succeed");

    let summary = report.summary();
    assert!(summary.contains("All 2 layers passed"));
    assert!(summary.contains("[PASS]"));
    assert!(!summary.contains("[FAIL]"));
}

#[test]
fn test_divergence_report_summary_with_failure() {
    let mut reference = ReferenceTrace::new();
    reference.checkpoint("good", &[1.0], &[1]).unwrap();
    reference.checkpoint("bad", &[1.0], &[1]).unwrap();

    let mut candidate = ReferenceTrace::new();
    candidate.checkpoint("good", &[1.0], &[1]).unwrap();
    candidate.checkpoint("bad", &[100.0], &[1]).unwrap();

    let config = ComparisonConfig::strict();
    let report = compare_traces(&reference, &candidate, &config).expect("compare should succeed");

    let summary = report.summary();
    assert!(summary.contains("[FAIL]"));
    assert!(summary.contains("First failure at layer 1"));
    assert!(summary.contains("bad"));
}

#[test]
fn test_layer_comparison_display() {
    let mut reference = ReferenceTrace::new();
    reference
        .checkpoint("encoder.conv", &[1.0, 2.0, 3.0], &[3])
        .unwrap();

    let mut candidate = ReferenceTrace::new();
    candidate
        .checkpoint("encoder.conv", &[1.001, 2.001, 3.001], &[3])
        .unwrap();

    let config = ComparisonConfig::new(0.01, 0.01, 0.0);
    let report = compare_traces(&reference, &candidate, &config).expect("compare should succeed");

    let display = format!("{}", report.layers[0]);
    assert!(display.contains("[PASS]"));
    assert!(display.contains("encoder.conv"));
    assert!(display.contains("[3]"));
}

#[test]
fn test_layer_comparison_display_fail() {
    let reference = NamedTensor::new("bad_layer", vec![2], vec![1.0, 2.0]).unwrap();
    let candidate = NamedTensor::new("bad_layer", vec![2], vec![100.0, 200.0]).unwrap();

    let config = ComparisonConfig::strict();
    let cmp = compare_tensors(&reference, &candidate, &config).expect("compare should succeed");

    let display = format!("{cmp}");
    assert!(display.contains("[FAIL]"));
    assert!(display.contains("bad_layer"));
}

#[test]
fn test_divergence_report_summary_empty_trace() {
    let report = DivergenceReport {
        layers: vec![],
        first_failure: None,
        all_passed: true,
    };

    let summary = report.summary();
    assert!(summary.contains("All 0 layers passed"));
}

// ===========================================================================
// Additional cross-cutting tests
// ===========================================================================

#[test]
fn test_preset_configs_produce_valid_comparisons() {
    let mut reference = ReferenceTrace::new();
    reference
        .checkpoint("test", &[1.0, 2.0, 3.0], &[3])
        .unwrap();

    let mut candidate = ReferenceTrace::new();
    // Perturbation must be small enough to pass even the STRICTEST preset
    // (strict: abs=1e-6, rel=1e-5). A 0.001 diff cannot pass strict/standard,
    // so the perturbation here is ~1e-7 (one to two f32 ULPs above the base
    // values), which is within every preset's thresholds.
    candidate
        .checkpoint("test", &[1.0000001, 2.0000002, 3.0000002], &[3])
        .unwrap();

    for preset in TolerancePreset::ALL {
        let config = preset.to_config();
        let report = compare_traces(&reference, &candidate, &config)
            .unwrap_or_else(|e| panic!("preset '{}' comparison failed: {e}", preset.name));
        // All presets should be relaxed enough to pass this small perturbation
        assert!(
            report.all_passed,
            "preset '{}' failed on small perturbation",
            preset.name,
        );
    }
}

#[test]
fn test_preset_lookup_by_name() {
    assert_eq!(
        TolerancePreset::by_name("strict"),
        Some(TolerancePreset::STRICT)
    );
    assert_eq!(
        TolerancePreset::by_name("TRANSFORMER"),
        Some(TolerancePreset::TRANSFORMER)
    );
    assert_eq!(TolerancePreset::by_name("nonexistent"), None);
}

#[test]
fn test_comparison_config_builder_pattern() {
    let config = ComparisonConfig::new(1e-3, 1e-2, 0.99)
        .with_rms_tolerance(0.5)
        .with_peak_amplitude_limit(1000.0);

    assert_eq!(config.abs_tolerance, 1e-3);
    assert_eq!(config.rel_tolerance, 1e-2);
    assert_eq!(config.cosine_threshold, 0.99);
    assert_eq!(config.rms_tolerance, Some(0.5));
    assert_eq!(config.peak_amplitude_limit, Some(1000.0));
}

#[test]
fn test_tolerance_length_mismatch_error() {
    let result = compare_with_tolerance(
        &[1.0f32, 2.0],
        &[1.0f32, 2.0, 3.0],
        &ToleranceStrategy::Absolute { atol: 1.0 },
    );
    assert!(matches!(
        result,
        Err(ReftestError::DataLengthMismatch { .. })
    ));
}

#[test]
fn test_tolerance_empty_slices_error() {
    let result = compare_with_tolerance(&[], &[], &ToleranceStrategy::Absolute { atol: 1.0 });
    assert!(matches!(result, Err(ReftestError::EmptyTensor(_))));
}

#[test]
fn test_tolerance_nan_never_passes() {
    let actual = [f32::NAN];
    let expected = [f32::NAN];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute {
            atol: f64::INFINITY,
        },
    )
    .expect("comparison should succeed");

    assert!(!result.passed);
    assert_eq!(result.num_mismatches, 1);
}

#[test]
fn test_assert_traces_match_macro_identical() {
    let mut reference = ReferenceTrace::new();
    reference.checkpoint("x", &[1.0, 2.0], &[2]).unwrap();

    let mut candidate = ReferenceTrace::new();
    candidate.checkpoint("x", &[1.0, 2.0], &[2]).unwrap();

    crate::assert_traces_match!(candidate, reference);
}

#[test]
fn test_assert_traces_match_macro_with_tolerance() {
    let mut reference = ReferenceTrace::new();
    reference.checkpoint("x", &[1.0, 2.0], &[2]).unwrap();

    let mut candidate = ReferenceTrace::new();
    candidate.checkpoint("x", &[1.001, 2.001], &[2]).unwrap();

    crate::assert_traces_match!(candidate, reference, abs = 0.01, rel = 0.01);
}

#[test]
fn test_assert_traces_match_preset_macro() {
    let mut reference = ReferenceTrace::new();
    reference.checkpoint("x", &[1.0, 2.0], &[2]).unwrap();

    let mut candidate = ReferenceTrace::new();
    // Stay comfortably inside the audio preset's abs=1e-3 threshold. A 0.001
    // perturbation lands on the boundary, where f32 rounding of (1.001 - 1.0)
    // = 1.00005e-3 spuriously exceeds it.
    candidate.checkpoint("x", &[1.0005, 2.0005], &[2]).unwrap();

    crate::assert_traces_match_preset!(candidate, reference, TolerancePreset::AUDIO);
}

#[test]
fn test_npy_write_and_read_preserves_negative_values() {
    let data = vec![-1.5f32, -0.5, 0.0, 0.5, 1.5];
    let npy_bytes = write_npy_to_bytes(&data, &[5]).expect("write should succeed");
    let tensor = read_npy_from_bytes(&npy_bytes).expect("read should succeed");

    assert_eq!(tensor.data, data);
}

#[test]
fn test_npy_write_and_read_large_tensor() {
    let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
    let npy_bytes = write_npy_to_bytes(&data, &[10, 100]).expect("write should succeed");
    let tensor = read_npy_from_bytes(&npy_bytes).expect("read should succeed");

    assert_eq!(tensor.shape, vec![10, 100]);
    assert_eq!(tensor.numel(), 1000);
    assert!((tensor.data[999] - 0.999).abs() < 1e-5);
}

#[test]
fn test_comparison_config_strict_vs_relaxed() {
    let strict = ComparisonConfig::strict();
    let relaxed = ComparisonConfig::relaxed();

    // Strict should have tighter bounds than relaxed
    assert!(strict.abs_tolerance < relaxed.abs_tolerance);
    assert!(strict.rel_tolerance < relaxed.rel_tolerance);
    assert!(strict.cosine_threshold > relaxed.cosine_threshold);
}

#[test]
fn test_reference_trace_from_checkpoints_preserves_order() {
    let checkpoints = vec![
        NamedTensor::new("c", vec![1], vec![3.0]).unwrap(),
        NamedTensor::new("a", vec![1], vec![1.0]).unwrap(),
        NamedTensor::new("b", vec![1], vec![2.0]).unwrap(),
    ];
    let trace = ReferenceTrace::from_checkpoints(checkpoints);

    // Order should be preserved as-given (not sorted)
    let names: Vec<&str> = trace.names().collect();
    assert_eq!(names, vec!["c", "a", "b"]);
}
