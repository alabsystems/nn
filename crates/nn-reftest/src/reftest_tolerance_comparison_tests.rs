// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for nn-reftest tolerance and comparison infrastructure.
//!
//! Covers:
//! - Tolerance comparison (absolute, relative, mixed, ULP, PercentClose)
//! - Safetensors loading and comparison (f32, f16, bf16 round-trips)
//! - Shape mismatch detection (rank, dimensions, scalar edge cases)
//! - DType mismatch detection (via tolerance and NamedTensor construction)
//! - NaN/Inf handling in comparisons (scattered, all-NaN, worst_index tracking)
//! - Multi-tensor trace comparison (large traces, graduated error, interleaved)
//! - Large tolerance vs small tolerance behavior (sensitivity analysis)

use crate::compare::{compare_tensors, compare_traces, ComparisonConfig};
use crate::error::ReftestError;
use crate::load::load_safetensors_from_bytes;
use crate::presets::TolerancePreset;
use crate::tolerance::{compare_with_tolerance, ToleranceStrategy};
use crate::trace::{NamedTensor, ReferenceTrace};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tensor_1d(name: &str, data: Vec<f32>) -> NamedTensor {
    let len = data.len();
    NamedTensor::new(name, vec![len], data).expect("valid 1-D test tensor")
}

fn tensor_nd(name: &str, shape: Vec<usize>, data: Vec<f32>) -> NamedTensor {
    NamedTensor::new(name, shape, data).expect("valid N-D test tensor")
}

fn build_trace(layers: &[(&str, Vec<f32>)]) -> ReferenceTrace {
    let mut trace = ReferenceTrace::new();
    for (name, data) in layers {
        trace
            .checkpoint(name, data, &[data.len()])
            .expect("valid checkpoint");
    }
    trace
}

fn f32_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn build_safetensors_typed(tensors: &[(&str, &[usize], safetensors::Dtype, &[u8])]) -> Vec<u8> {
    let mut tensor_map: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
    for &(name, shape, dtype, data) in tensors {
        let view = safetensors::tensor::TensorView::new(dtype, shape.to_vec(), data)
            .expect("valid tensor view");
        tensor_map.push((name.to_string(), view));
    }
    safetensors::tensor::serialize(tensor_map, None).expect("serialization should succeed")
}

fn build_safetensors_f32(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
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
// 1. Absolute tolerance — monotonic sequences and boundary precision
// ===========================================================================

#[test]
fn test_absolute_tolerance_monotonic_increasing_all_within() {
    let expected: Vec<f32> = (0..50).map(|i| i as f32 * 10.0).collect();
    let actual: Vec<f32> = expected.iter().map(|&x| x + 0.005).collect();
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 0.01 },
    )
    .expect("should succeed");
    assert!(
        result.passed,
        "uniform 0.005 offset within atol=0.01 should pass"
    );
    assert_eq!(result.num_mismatches, 0);
    assert!(
        (result.max_diff - 0.005).abs() < 1e-4,
        "max_diff should be ~0.005, got {}",
        result.max_diff
    );
}

#[test]
fn test_absolute_tolerance_monotonic_decreasing_single_outlier() {
    let expected: Vec<f32> = (0..20).rev().map(|i| i as f32).collect();
    let mut actual = expected.clone();
    actual[10] += 0.5; // single outlier
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 0.1 },
    )
    .expect("should succeed");
    assert!(!result.passed, "0.5 outlier should fail atol=0.1");
    assert_eq!(result.num_mismatches, 1);
    assert_eq!(result.worst_index, 10);
}

#[test]
fn test_absolute_tolerance_very_tight_catches_float_imprecision() {
    // f32 arithmetic: 0.1 + 0.2 != 0.3 exactly.
    let a = [0.1_f32 + 0.2_f32]; // ~0.30000001192
    let b = [0.3_f32]; // ~0.30000001192 (same in f32)
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 0.0 })
        .expect("should succeed");
    // In f32, 0.1+0.2 and 0.3 are actually the same bit pattern.
    assert!(
        result.passed,
        "0.1+0.2 and 0.3 should be the same f32 representation"
    );
}

#[test]
fn test_absolute_tolerance_alternating_signs() {
    let expected: Vec<f32> = (0..10)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let actual: Vec<f32> = expected.iter().map(|&x| x + 0.001).collect();
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 0.01 },
    )
    .expect("should succeed");
    assert!(
        result.passed,
        "alternating signs with small offset should pass"
    );
}

// ===========================================================================
// 2. Relative tolerance — scale-independent comparison
// ===========================================================================

#[test]
fn test_relative_tolerance_multi_scale_uniform_error_rate() {
    // Values spanning 6 orders of magnitude, all with ~0.01% relative error.
    let expected = [1e-3_f32, 1e-1, 1e1, 1e3];
    let actual: Vec<f32> = expected.iter().map(|&x| x * 1.0001).collect();
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Relative { rtol: 0.001 },
    )
    .expect("should succeed");
    assert!(result.passed, "0.01% relative error should pass rtol=0.1%");
}

#[test]
fn test_relative_tolerance_zero_expected_uses_epsilon_floor() {
    // When expected is 0.0, relative tolerance uses epsilon floor (1e-8).
    let a = [5e-11_f32];
    let b = [0.0_f32];
    // diff = 5e-11, denom = max(5e-11, 0, 1e-8) = 1e-8, rel = 5e-11/1e-8 = 0.005.
    // Without the epsilon floor, denom would be 5e-11 and rel would be 1.0 (a
    // 100% mismatch); the floor is what lets this pass. 0.005 sits comfortably
    // below rtol=0.01, avoiding the exact-boundary case where f32->f64 widening
    // of 1e-10 would push rel just over 0.01.
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Relative { rtol: 0.01 })
        .expect("should succeed");
    assert!(
        result.passed,
        "tiny value vs zero should pass with rtol=0.01 due to epsilon floor"
    );
}

#[test]
fn test_relative_tolerance_large_values_small_absolute_diff() {
    // Large values where absolute diff is significant but relative is tiny.
    let a = [1_000_000.0_f32];
    let b = [1_000_001.0_f32];
    // Absolute diff = 1.0, relative diff = 1e-6
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Relative { rtol: 1e-5 })
        .expect("should succeed");
    assert!(result.passed, "1e-6 relative error should pass rtol=1e-5");
}

// ===========================================================================
// 3. Mixed tolerance — NumPy allclose semantics edge cases
// ===========================================================================

#[test]
fn test_mixed_tolerance_zero_expected_relies_on_atol() {
    // When expected (b) is 0: threshold = atol + rtol * 0 = atol.
    let a = [0.0005_f32];
    let b = [0.0_f32];
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::Mixed {
            atol: 0.001,
            rtol: 1e-6,
        },
    )
    .expect("should succeed");
    assert!(
        result.passed,
        "diff 0.0005 < atol 0.001 should pass for zero expected"
    );
}

#[test]
fn test_mixed_tolerance_very_large_expected_relies_on_rtol() {
    // Large expected: threshold = atol + rtol * |b| ~= rtol * |b|.
    let a = [1_000_000.5_f32];
    let b = [1_000_000.0_f32];
    // threshold = 1e-8 + 1e-6 * 1e6 = 1e-8 + 1.0 = ~1.0
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::Mixed {
            atol: 1e-8,
            rtol: 1e-6,
        },
    )
    .expect("should succeed");
    assert!(
        result.passed,
        "0.5 diff < ~1.0 threshold for large expected value"
    );
}

#[test]
fn test_mixed_tolerance_per_element_varying_scales() {
    // Each element at a different scale: mixed tolerance adapts per-element.
    let expected = [0.001_f32, 1.0, 1000.0];
    let actual = [0.0015_f32, 1.005, 1005.0]; // 50%, 0.5%, 0.5% relative diffs
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Mixed {
            atol: 0.001,
            rtol: 0.01,
        },
    )
    .expect("should succeed");
    // For 0.001: threshold = 0.001 + 0.01*0.001 = 0.00101, diff=0.0005 => pass
    // For 1.0: threshold = 0.001 + 0.01*1.0 = 0.011, diff=0.005 => pass
    // For 1000.0: threshold = 0.001 + 0.01*1000 = 10.001, diff=5.0 => pass
    assert!(
        result.passed,
        "mixed tolerance should adapt per-element scale"
    );
}

// ===========================================================================
// 4. ULP tolerance — edge cases around zero crossing and subnormals
// ===========================================================================

#[test]
fn test_ulp_zero_crossing_two_steps() {
    // +smallest_subnormal vs -smallest_subnormal = 2 ULP distance through zero.
    let pos = f32::from_bits(1);
    let neg = f32::from_bits(0x8000_0001);
    let result_1 = compare_with_tolerance(&[pos], &[neg], &ToleranceStrategy::ULP { max_ulps: 1 })
        .expect("should succeed");
    assert!(!result_1.passed, "2 ULP gap should fail max_ulps=1");

    let result_2 = compare_with_tolerance(&[pos], &[neg], &ToleranceStrategy::ULP { max_ulps: 2 })
        .expect("should succeed");
    assert!(result_2.passed, "2 ULP gap should pass max_ulps=2");
}

#[test]
fn test_ulp_large_gap_between_1_and_2() {
    let a = [1.0_f32];
    let b = [2.0_f32];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::ULP { max_ulps: 1000 })
        .expect("should succeed");
    // 1.0 to 2.0 is millions of ULPs apart.
    assert!(
        !result.passed,
        "1.0 vs 2.0 should fail even with 1000 ULP tolerance"
    );
}

#[test]
fn test_ulp_exact_match_multiple_elements() {
    let data = [0.0_f32, 1.0, -1.0, f32::MIN_POSITIVE, f32::MAX];
    let result = compare_with_tolerance(&data, &data, &ToleranceStrategy::ULP { max_ulps: 0 })
        .expect("should succeed");
    assert!(result.passed, "identical values should be 0 ULPs apart");
    assert_eq!(result.num_mismatches, 0);
}

// ===========================================================================
// 5. PercentClose — fractional outlier tolerance
// ===========================================================================

#[test]
fn test_percent_close_exactly_at_threshold_passes() {
    // 8 out of 10 within threshold = 80%, require exactly 80%.
    let expected = [0.0_f32; 10];
    let mut actual = [0.0_f32; 10];
    actual[0] = 100.0;
    actual[5] = 100.0; // 2 outliers => 80% close
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::PercentClose {
            threshold: 1.0,
            percent: 80.0,
        },
    )
    .expect("should succeed");
    assert!(
        result.passed,
        "exactly 80% close == 80% required should pass"
    );
    assert_eq!(result.num_mismatches, 2);
}

#[test]
fn test_percent_close_one_below_threshold_fails() {
    // 7 out of 10 = 70%, require 71%.
    let expected = [0.0_f32; 10];
    let mut actual = [0.0_f32; 10];
    actual[0] = 100.0;
    actual[1] = 100.0;
    actual[2] = 100.0; // 3 outliers => 70% close
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::PercentClose {
            threshold: 1.0,
            percent: 71.0,
        },
    )
    .expect("should succeed");
    assert!(!result.passed, "70% close < 71% required should fail");
}

#[test]
fn test_percent_close_100_percent_all_must_match() {
    let a = [1.0_f32, 2.0, 3.0];
    let b = [1.0_f32, 2.0, 3.001]; // one element 0.001 over threshold
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::PercentClose {
            threshold: 0.0005,
            percent: 100.0,
        },
    )
    .expect("should succeed");
    assert!(
        !result.passed,
        "100% requirement with one outlier should fail"
    );
}

// ===========================================================================
// 6. Safetensors loading — f16, bf16, multi-tensor comparison
// ===========================================================================

#[test]
fn test_safetensors_f16_load_and_compare() {
    let f16_values = [half::f16::from_f32(1.0),
        half::f16::from_f32(2.0),
        half::f16::from_f32(3.0)];
    let raw: Vec<u8> = f16_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes = build_safetensors_typed(&[("weights", &[3], safetensors::Dtype::F16, &raw)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("should load f16 safetensors");
    assert_eq!(trace.len(), 1);
    let t = trace.get(0).expect("exists");
    assert_eq!(t.shape, vec![3]);
    assert!((t.data[0] - 1.0).abs() < 0.01, "f16 1.0 should round-trip");
    assert!((t.data[1] - 2.0).abs() < 0.01, "f16 2.0 should round-trip");
    assert!((t.data[2] - 3.0).abs() < 0.01, "f16 3.0 should round-trip");
}

#[test]
fn test_safetensors_bf16_load_and_compare() {
    let bf16_values = [half::bf16::from_f32(0.5),
        half::bf16::from_f32(-1.5),
        half::bf16::from_f32(100.0)];
    let raw: Vec<u8> = bf16_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes = build_safetensors_typed(&[("bias", &[3], safetensors::Dtype::BF16, &raw)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("should load bf16 safetensors");
    assert_eq!(trace.len(), 1);
    let t = trace.get(0).expect("exists");
    assert!((t.data[0] - 0.5).abs() < 0.1);
    assert!((t.data[1] - (-1.5)).abs() < 0.1);
    assert!((t.data[2] - 100.0).abs() < 1.0);
}

#[test]
fn test_safetensors_multi_tensor_load_then_compare_trace() {
    let data_a = vec![1.0, 2.0, 3.0];
    let data_b = vec![4.0, 5.0];
    let data_c = vec![6.0, 7.0, 8.0, 9.0];
    let bytes = build_safetensors_f32(&[
        ("layer_a", &[3], &data_a),
        ("layer_b", &[2], &data_b),
        ("layer_c", &[4], &data_c),
    ]);
    let reference = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    // Build a matching candidate trace (sorted alphabetically to match safetensors).
    let mut candidate = ReferenceTrace::new();
    candidate
        .checkpoint("layer_a", &[1.0, 2.0, 3.0], &[3])
        .expect("valid");
    candidate
        .checkpoint("layer_b", &[4.0, 5.0], &[2])
        .expect("valid");
    candidate
        .checkpoint("layer_c", &[6.0, 7.0, 8.0, 9.0], &[4])
        .expect("valid");

    let report = compare_traces(&reference, &candidate, &ComparisonConfig::default())
        .expect("comparison should succeed");
    assert!(
        report.all_passed,
        "identical data from safetensors should match"
    );
    assert_eq!(report.layers.len(), 3);
}

#[test]
fn test_safetensors_multi_tensor_divergence_detected() {
    let data_a = vec![1.0, 2.0, 3.0];
    let data_b = vec![4.0, 5.0];
    let bytes = build_safetensors_f32(&[("alpha", &[3], &data_a), ("beta", &[2], &data_b)]);
    let reference = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    let mut candidate = ReferenceTrace::new();
    candidate
        .checkpoint("alpha", &[1.0, 2.0, 3.0], &[3])
        .expect("valid");
    candidate
        .checkpoint("beta", &[4.0, 999.0], &[2])
        .expect("valid"); // diverges

    let report = compare_traces(&reference, &candidate, &ComparisonConfig::default())
        .expect("comparison should succeed");
    assert!(!report.all_passed);
    assert_eq!(report.first_failure, Some(1));
    assert!(report.layers[0].passed);
    assert!(!report.layers[1].passed);
}

#[test]
fn test_safetensors_f64_load_valid() {
    let f64_values: Vec<f64> = vec![1.0, -2.5, 3.14, 0.0];
    let raw: Vec<u8> = f64_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes = build_safetensors_typed(&[("params", &[4], safetensors::Dtype::F64, &raw)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("should load f64 safetensors");
    let t = trace.get(0).expect("exists");
    assert_eq!(t.data.len(), 4);
    assert!((t.data[0] - 1.0).abs() < f32::EPSILON);
    assert!((t.data[1] - (-2.5)).abs() < f32::EPSILON);
    assert!((t.data[2] - 3.14).abs() < 1e-5);
}

// ===========================================================================
// 7. Shape mismatch detection — various rank and dimension mismatches
// ===========================================================================

#[test]
fn test_shape_mismatch_scalar_vs_vector() {
    let scalar = NamedTensor::new("x", vec![], vec![1.0]).expect("valid scalar");
    let vector = tensor_1d("x", vec![1.0]);
    let err = compare_tensors(&scalar, &vector, &ComparisonConfig::default())
        .expect_err("scalar vs vector should fail");
    assert!(matches!(err, ReftestError::ShapeMismatch { .. }));
}

#[test]
fn test_shape_mismatch_high_rank_4d_vs_2d() {
    let a = tensor_nd("x", vec![2, 3, 4, 5], vec![0.0; 120]);
    let b = tensor_nd("x", vec![24, 5], vec![0.0; 120]);
    let err = compare_tensors(&a, &b, &ComparisonConfig::default())
        .expect_err("4D vs 2D should fail on shape");
    match err {
        ReftestError::ShapeMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, vec![2, 3, 4, 5]);
            assert_eq!(actual, vec![24, 5]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn test_shape_mismatch_same_product_different_factorization() {
    // [6] vs [2,3] — same element count but different shapes.
    let a = tensor_1d("x", vec![0.0; 6]);
    let b = tensor_nd("x", vec![2, 3], vec![0.0; 6]);
    let err =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect_err("flat vs 2D should fail");
    assert!(matches!(err, ReftestError::ShapeMismatch { .. }));
}

#[test]
fn test_shape_mismatch_in_multi_layer_trace_reports_correct_layer() {
    let ref_trace = ReferenceTrace::from_checkpoints(vec![
        tensor_nd("layer_0", vec![4], vec![0.0; 4]),
        tensor_nd("layer_1", vec![2, 2], vec![0.0; 4]),
        tensor_nd("layer_2", vec![2, 3], vec![0.0; 6]),
    ]);
    let cand_trace = ReferenceTrace::from_checkpoints(vec![
        tensor_nd("layer_0", vec![4], vec![0.0; 4]),
        tensor_nd("layer_1", vec![4], vec![0.0; 4]), // shape mismatch at index 1
        tensor_nd("layer_2", vec![2, 3], vec![0.0; 6]),
    ]);
    let err = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect_err("should fail on shape mismatch");
    match err {
        ReftestError::ShapeMismatch {
            name,
            expected,
            actual,
        } => {
            assert_eq!(name, "layer_1");
            assert_eq!(expected, vec![2, 2]);
            assert_eq!(actual, vec![4]);
        }
        other => panic!("expected ShapeMismatch at layer_1, got {other:?}"),
    }
}

// ===========================================================================
// 8. DType mismatch detection — NamedTensor and tolerance level
// ===========================================================================

#[test]
fn test_named_tensor_too_many_elements_for_shape() {
    let result = NamedTensor::new("overflow_data", vec![2, 2], vec![1.0; 8]);
    match result {
        Err(ReftestError::ElementCountMismatch {
            expected, actual, ..
        }) => {
            assert_eq!(expected, 4);
            assert_eq!(actual, 8);
        }
        other => panic!("expected ElementCountMismatch, got {other:?}"),
    }
}

#[test]
fn test_named_tensor_too_few_elements_for_shape() {
    let result = NamedTensor::new("short_data", vec![10], vec![1.0; 3]);
    match result {
        Err(ReftestError::ElementCountMismatch {
            expected, actual, ..
        }) => {
            assert_eq!(expected, 10);
            assert_eq!(actual, 3);
        }
        other => panic!("expected ElementCountMismatch, got {other:?}"),
    }
}

#[test]
fn test_tolerance_data_length_mismatch_reports_correct_sizes() {
    let a = [1.0_f32; 100];
    let b = [1.0_f32; 50];
    let err = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1.0 })
        .expect_err("should fail on length mismatch");
    match err {
        ReftestError::DataLengthMismatch { expected, actual } => {
            assert_eq!(expected, 50, "expected length is from the 'expected' slice");
            assert_eq!(actual, 100, "actual length is from the 'actual' slice");
        }
        other => panic!("expected DataLengthMismatch, got {other:?}"),
    }
}

// ===========================================================================
// 9. NaN handling — scattered NaN, worst_index, all-NaN per strategy
// ===========================================================================

#[test]
fn test_nan_scattered_in_large_array_detects_all() {
    let n = 1000;
    let mut actual = vec![1.0_f32; n];
    let expected = vec![1.0_f32; n];
    // Scatter NaN at indices 100, 500, 999.
    actual[100] = f32::NAN;
    actual[500] = f32::NAN;
    actual[999] = f32::NAN;

    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 1.0 },
    )
    .expect("should succeed");
    assert!(!result.passed, "scattered NaN should cause failure");
    assert_eq!(result.num_mismatches, 3, "should detect all 3 NaN elements");
}

#[test]
fn test_nan_worst_index_is_first_nan_when_only_nans() {
    let actual = [f32::NAN, f32::NAN, f32::NAN];
    let expected = [1.0_f32, 2.0, 3.0];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 0.0 },
    )
    .expect("should succeed");
    assert!(result.max_diff.is_infinite(), "NaN diff should be infinite");
    // worst_index tracks the element with max diff; first infinite diff wins.
    assert_eq!(result.worst_index, 0, "first NaN should be worst_index");
}

#[test]
fn test_nan_in_expected_only_also_fails() {
    let actual = [1.0_f32, 2.0];
    let expected = [1.0_f32, f32::NAN];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: f64::MAX },
    )
    .expect("should succeed");
    assert!(!result.passed, "NaN in expected should also fail");
    assert_eq!(result.num_mismatches, 1);
}

#[test]
fn test_nan_compare_tensors_peak_amplitude_tracks_correctly() {
    // NaN in candidate should produce infinite peak amplitude.
    let a = tensor_1d("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = tensor_1d("x", vec![1.0, 2.0, f32::NAN, 4.0, 5.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed structurally");
    assert!(!result.passed);
    // Peak amplitude is tracked only for candidate; NaN candidate = infinite peak.
    // Actually, looking at the code: NaN is !c.is_finite() so peak_amp = f32::INFINITY.
    assert!(
        result.peak_amplitude.is_infinite(),
        "NaN in candidate should produce infinite peak amplitude"
    );
}

#[test]
fn test_inf_positive_in_reference_fails_cosine_remains_finite_for_valid() {
    let a = tensor_1d("x", vec![f32::INFINITY, 1.0, 2.0, 3.0]);
    let b = tensor_1d("x", vec![1.0, 1.0, 2.0, 3.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed structurally");
    assert!(!result.passed);
    // Cosine should be computed from finite pairs (1,2,3) dot (1,2,3) = 14.
    assert!(
        result.cosine_similarity.is_finite(),
        "cosine from valid elements should be finite, got {}",
        result.cosine_similarity
    );
}

#[test]
fn test_neg_inf_scattered_fails() {
    let actual = [1.0_f32, f32::NEG_INFINITY, 3.0, f32::NEG_INFINITY, 5.0];
    let expected = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: f64::MAX },
    )
    .expect("should succeed");
    assert!(!result.passed);
    assert_eq!(
        result.num_mismatches, 2,
        "should detect both NEG_INFINITY elements"
    );
}

// ===========================================================================
// 10. Multi-tensor trace comparison — large traces, interleaved pass/fail
// ===========================================================================

#[test]
fn test_large_trace_20_layers_all_pass() {
    let mut ref_trace = ReferenceTrace::new();
    let mut cand_trace = ReferenceTrace::new();
    for i in 0..20 {
        let data: Vec<f32> = (0..50).map(|j| (i * 50 + j) as f32 * 0.01).collect();
        ref_trace
            .checkpoint(&format!("layer_{i}"), &data, &[50])
            .expect("valid");
        cand_trace
            .checkpoint(&format!("layer_{i}"), &data, &[50])
            .expect("valid");
    }
    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("should succeed");
    assert!(report.all_passed);
    assert_eq!(report.layers.len(), 20);
}

#[test]
fn test_trace_interleaved_pass_fail_every_other() {
    let mut ref_trace = ReferenceTrace::new();
    let mut cand_trace = ReferenceTrace::new();
    for i in 0..8 {
        let ref_data = vec![1.0_f32; 10];
        let cand_data = if i % 2 == 0 {
            vec![1.0_f32; 10]
        } else {
            vec![999.0_f32; 10]
        };
        ref_trace
            .checkpoint(&format!("layer_{i}"), &ref_data, &[10])
            .expect("valid");
        cand_trace
            .checkpoint(&format!("layer_{i}"), &cand_data, &[10])
            .expect("valid");
    }
    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("should succeed");
    assert!(!report.all_passed);
    assert_eq!(
        report.first_failure,
        Some(1),
        "first odd-indexed layer fails"
    );
    // Verify interleaved pattern.
    for i in 0..8 {
        if i % 2 == 0 {
            assert!(report.layers[i].passed, "even layer {i} should pass");
        } else {
            assert!(!report.layers[i].passed, "odd layer {i} should fail");
        }
    }
}

#[test]
fn test_trace_graduated_error_increasing_per_layer() {
    // Simulate error accumulation: each layer has 10x more error than the previous.
    let mut ref_trace = ReferenceTrace::new();
    let mut cand_trace = ReferenceTrace::new();
    for i in 0..5 {
        let ref_data = vec![1.0_f32; 10];
        let perturbation = 10.0_f32.powi(i - 6); // 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
        let cand_data: Vec<f32> = ref_data.iter().map(|&x| x + perturbation).collect();
        ref_trace
            .checkpoint(&format!("layer_{i}"), &ref_data, &[10])
            .expect("valid");
        cand_trace
            .checkpoint(&format!("layer_{i}"), &cand_data, &[10])
            .expect("valid");
    }

    // Strict config (atol=1e-6) should fail from layer 1 onward.
    let strict_report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::strict())
        .expect("should succeed");
    assert!(
        !strict_report.all_passed,
        "strict should fail on accumulated error"
    );
    assert!(
        strict_report.layers[0].passed,
        "1e-6 perturbation should pass strict"
    );

    // Relaxed config (atol=1e-2) should pass all layers.
    let relaxed_report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::relaxed())
        .expect("should succeed");
    assert!(relaxed_report.all_passed, "relaxed should pass all layers");
}

#[test]
fn test_trace_only_last_layer_fails() {
    let mut ref_trace = ReferenceTrace::new();
    let mut cand_trace = ReferenceTrace::new();
    for i in 0..10 {
        let data = vec![1.0_f32; 5];
        ref_trace
            .checkpoint(&format!("layer_{i}"), &data, &[5])
            .expect("valid");
        if i == 9 {
            cand_trace
                .checkpoint(&format!("layer_{i}"), &[999.0; 5], &[5])
                .expect("valid");
        } else {
            cand_trace
                .checkpoint(&format!("layer_{i}"), &data, &[5])
                .expect("valid");
        }
    }
    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("should succeed");
    assert!(!report.all_passed);
    assert_eq!(
        report.first_failure,
        Some(9),
        "only the last layer should fail"
    );
    for i in 0..9 {
        assert!(report.layers[i].passed, "layer {i} should pass");
    }
    assert!(!report.layers[9].passed);
}

// ===========================================================================
// 11. Large tolerance vs small tolerance — sensitivity analysis
// ===========================================================================

#[test]
fn test_sensitivity_same_data_different_tolerances() {
    let ref_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let cand_data = vec![1.001, 2.001, 3.001, 4.001, 5.001];
    let ref_t = tensor_1d("x", ref_data);
    let cand_t = tensor_1d("x", cand_data);

    // Very tight tolerance: should fail (0.001 > 1e-6).
    let strict =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::strict()).expect("should succeed");
    assert!(!strict.passed, "strict should reject 0.001 diffs");

    // Default tolerance: should fail (0.001 > 1e-5).
    let default =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert!(!default.passed, "default should reject 0.001 diffs");

    // Relaxed tolerance: should pass (0.001 < 1e-2).
    let relaxed =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::relaxed()).expect("should succeed");
    assert!(relaxed.passed, "relaxed should accept 0.001 diffs");
}

#[test]
fn test_sensitivity_tolerance_ladder() {
    // Sweep tolerance from very tight to very loose; verify monotonic pass behavior.
    let ref_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
    let cand_data: Vec<f32> = ref_data.iter().map(|&x| x + 0.05).collect();
    let ref_t = tensor_1d("sweep", ref_data);
    let cand_t = tensor_1d("sweep", cand_data);

    let tolerances = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0];
    let mut first_pass = None;
    for (i, &tol) in tolerances.iter().enumerate() {
        let config = ComparisonConfig::new(tol, 1.0, 0.0); // only abs matters
        let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
        if result.passed && first_pass.is_none() {
            first_pass = Some(i);
        }
        // Once we start passing, we should keep passing (monotonic).
        if let Some(fp) = first_pass {
            if i >= fp {
                assert!(
                    result.passed,
                    "tolerance {tol} should pass (monotonicity violated at index {i})"
                );
            }
        }
    }
    // The diff is 0.05, so we should start passing around tol=0.05 => index ~4 (1e-2) or 5 (1e-1).
    assert!(
        first_pass.is_some(),
        "should eventually pass with large enough tolerance"
    );
}

#[test]
fn test_sensitivity_cosine_threshold_ladder() {
    // Two vectors at a known angle: [1,1] and [1,0].
    // cos = 1/sqrt(2) ~ 0.7071.
    let ref_t = tensor_1d("x", vec![1.0, 1.0]);
    let cand_t = tensor_1d("x", vec![1.0, 0.0]);

    let thresholds = [0.0, 0.5, 0.7, 0.707, 0.7072, 0.8, 0.9, 0.99];
    let mut first_fail = None;
    for (i, &thresh) in thresholds.iter().enumerate() {
        let config = ComparisonConfig::new(100.0, 100.0, thresh);
        let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
        if !result.passed && first_fail.is_none() {
            first_fail = Some(i);
        }
    }
    // cos ~ 0.7071, so thresholds > 0.7071 should fail.
    assert!(first_fail.is_some(), "some threshold should cause failure");
    let fi = first_fail.unwrap();
    assert!(
        thresholds[fi] > 0.7071,
        "first failure should be at threshold > 0.7071, got {}",
        thresholds[fi]
    );
}

#[test]
fn test_sensitivity_preset_ordering() {
    // Verify that presets form a strictly increasing tolerance order.
    let presets = [
        TolerancePreset::STRICT,
        TolerancePreset::STANDARD,
        TolerancePreset::TRANSFORMER,
        TolerancePreset::AUDIO,
    ];
    for window in presets.windows(2) {
        assert!(
            window[0].abs_threshold < window[1].abs_threshold,
            "preset {} abs_threshold ({}) should be < {} ({})",
            window[0].name,
            window[0].abs_threshold,
            window[1].name,
            window[1].abs_threshold,
        );
    }
}

// ===========================================================================
// 12. ComparisonConfig — gate interaction and builder patterns
// ===========================================================================

#[test]
fn test_rms_gate_does_not_affect_when_disabled() {
    let a = tensor_1d("x", vec![0.0; 100]);
    let b = tensor_1d("x", vec![0.005; 100]); // RMS = 0.005
    let config = ComparisonConfig::new(0.01, 1.0, 0.0); // no RMS gate
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(result.passed, "should pass without RMS gate");
    assert!(
        (result.rms_diff - 0.005).abs() < 1e-4,
        "RMS should still be computed"
    );
}

#[test]
fn test_peak_amplitude_gate_does_not_affect_when_disabled() {
    let a = tensor_1d("x", vec![0.0]);
    let b = tensor_1d("x", vec![1000.0]); // very large peak
    let config = ComparisonConfig::new(10000.0, 10000.0, 0.0); // no peak gate
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(result.passed, "should pass without peak gate");
    assert_eq!(
        result.peak_amplitude, 1000.0,
        "peak should still be tracked"
    );
}

#[test]
fn test_both_gates_enabled_both_must_pass() {
    let a = tensor_1d("x", vec![0.0, 0.0, 0.0, 0.0]);
    let b = tensor_1d("x", vec![0.01, 0.01, 0.01, 0.01]);

    let config = ComparisonConfig::new(0.1, 1.0, 0.0)
        .with_rms_tolerance(0.005) // RMS = 0.01 > 0.005 => fails
        .with_peak_amplitude_limit(0.1); // peak = 0.01 <= 0.1 => passes
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        !result.passed,
        "RMS gate should fail even though peak gate passes"
    );
}

// ===========================================================================
// 13. Compare tensors — various data patterns
// ===========================================================================

#[test]
fn test_compare_tensors_all_zeros_identical() {
    let a = tensor_1d("zeros", vec![0.0; 100]);
    let b = tensor_1d("zeros", vec![0.0; 100]);
    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.max_abs_diff, 0.0);
    assert_eq!(
        result.cosine_similarity, 1.0,
        "zero vs zero cosine should be 1.0"
    );
}

#[test]
fn test_compare_tensors_all_ones_with_tiny_perturbation() {
    let n = 500;
    let a_data = vec![1.0_f32; n];
    let b_data: Vec<f32> = a_data.iter().map(|&x| x + 1e-7).collect();
    let a = tensor_nd("ones", vec![n], a_data);
    let b = tensor_nd("ones", vec![n], b_data);
    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");
    assert!(
        result.passed,
        "1e-7 perturbation should pass default tolerance"
    );
    assert!(result.max_abs_diff < 1e-6);
}

#[test]
fn test_compare_tensors_negative_values_cosine_correct() {
    // Anti-parallel: [1,0] vs [-1,0] => cosine = -1.0.
    let a = tensor_1d("anti", vec![1.0, 0.0]);
    let b = tensor_1d("anti", vec![-1.0, 0.0]);
    let config = ComparisonConfig::new(100.0, 100.0, -1.0); // accept any cosine >= -1.0
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        (result.cosine_similarity - (-1.0)).abs() < 1e-5,
        "anti-parallel cosine should be -1.0, got {}",
        result.cosine_similarity
    );
    assert!(result.passed, "should pass with cosine threshold -1.0");
}

#[test]
fn test_compare_tensors_single_very_large_value() {
    let a = tensor_1d("big", vec![f32::MAX / 2.0]);
    let b = tensor_1d("big", vec![f32::MAX / 2.0]);
    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed, "identical large values should pass");
    assert_eq!(result.max_abs_diff, 0.0);
}

#[test]
fn test_compare_tensors_single_very_small_subnormal() {
    let tiny = f32::from_bits(1); // smallest positive subnormal
    let a = tensor_1d("tiny", vec![tiny]);
    let b = tensor_1d("tiny", vec![tiny]);
    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed, "identical subnormals should pass");
}

// ===========================================================================
// 14. DivergenceReport summary — edge cases
// ===========================================================================

#[test]
fn test_divergence_report_single_layer_fail_summary() {
    let ref_trace = build_trace(&[("output", vec![1.0])]);
    let cand_trace = build_trace(&[("output", vec![999.0])]);
    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("should succeed");
    let summary = report.summary();
    assert!(summary.contains("First failure at layer 0"));
    assert!(summary.contains("output"));
    assert!(summary.contains("[FAIL]"));
}

#[test]
fn test_divergence_report_all_layers_fail_summary() {
    let ref_trace = build_trace(&[("a", vec![1.0]), ("b", vec![2.0])]);
    let cand_trace = build_trace(&[("a", vec![999.0]), ("b", vec![999.0])]);
    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("should succeed");
    let summary = report.summary();
    assert!(
        summary.contains("First failure at layer 0"),
        "summary: {summary}"
    );
    // Should not contain "All N layers passed".
    assert!(
        !summary.contains("All"),
        "should not say 'All passed' when failing: {summary}"
    );
}

// ===========================================================================
// 15. Tolerance strategies — comparison result consistency checks
// ===========================================================================

#[test]
fn test_comparison_result_max_diff_always_gte_mean_diff() {
    // For any data, max_diff >= mean_diff.
    let patterns: Vec<(&str, Vec<f32>, Vec<f32>)> = vec![
        ("uniform", vec![0.0; 100], vec![0.01; 100]),
        (
            "single_outlier",
            {
                let mut v = vec![0.0; 100];
                v[50] = 10.0;
                v
            },
            vec![0.0; 100],
        ),
        (
            "gradient",
            (0..100).map(|i| i as f32 * 0.01).collect(),
            vec![0.0; 100],
        ),
    ];
    for (name, actual, expected) in &patterns {
        let result = compare_with_tolerance(
            actual,
            expected,
            &ToleranceStrategy::Absolute { atol: f64::MAX },
        )
        .expect("should succeed");
        assert!(
            result.max_diff >= result.mean_diff,
            "pattern '{name}': max_diff ({}) should be >= mean_diff ({})",
            result.max_diff,
            result.mean_diff,
        );
    }
}

#[test]
fn test_comparison_result_num_mismatches_bounded_by_length() {
    let n = 50;
    let a = vec![0.0_f32; n];
    let b: Vec<f32> = (0..n).map(|i| i as f32).collect(); // all different
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 0.0 })
        .expect("should succeed");
    // Element 0 matches (both 0.0), rest differ.
    assert!(result.num_mismatches <= n);
    assert_eq!(result.num_mismatches, n - 1, "only element 0 matches");
}

#[test]
fn test_comparison_result_worst_index_bounded_by_length() {
    let a = [0.0_f32, 0.0, 0.0, 0.0, 5.0];
    let b = [0.0_f32, 0.0, 0.0, 0.0, 0.0];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1.0 })
        .expect("should succeed");
    assert!(result.worst_index < a.len(), "worst_index must be < length");
    assert_eq!(result.worst_index, 4, "element 4 has the largest diff");
}

// ===========================================================================
// 16. Presets — by_name lookup and config conversion
// ===========================================================================

#[test]
fn test_preset_by_name_returns_all_known_presets() {
    let names = [
        "strict",
        "standard",
        "transformer",
        "audio",
        "quantized",
        "tts",
    ];
    for name in &names {
        let preset = TolerancePreset::by_name(name);
        assert!(preset.is_some(), "preset '{name}' should be found");
        assert_eq!(preset.unwrap().name, *name);
    }
}

#[test]
fn test_preset_to_config_preserves_thresholds() {
    for preset in TolerancePreset::ALL {
        let config = preset.to_config();
        assert_eq!(config.abs_tolerance, preset.abs_threshold as f32);
        assert_eq!(config.rel_tolerance, preset.rel_threshold as f32);
        assert_eq!(config.cosine_threshold, preset.cos_threshold as f32);
        assert!(
            config.rms_tolerance.is_none(),
            "preset should not enable RMS gate"
        );
        assert!(
            config.peak_amplitude_limit.is_none(),
            "preset should not enable peak gate"
        );
    }
}

#[test]
fn test_preset_unknown_name_returns_none() {
    assert!(TolerancePreset::by_name("nonexistent").is_none());
    assert!(TolerancePreset::by_name("").is_none());
    assert!(TolerancePreset::by_name("STRICT ").is_none()); // trailing space
}

// ===========================================================================
// 17. Integration — safetensors load + tolerance comparison pipeline
// ===========================================================================

#[test]
fn test_safetensors_load_compare_with_preset() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let bytes = build_safetensors_f32(&[("layer", &[4], &data)]);
    let reference = load_safetensors_from_bytes(&bytes).expect("load");

    let mut candidate = ReferenceTrace::new();
    // Perturb by 5e-4: well above the standard preset's abs=1e-5 (so it must
    // reject) yet comfortably below the audio preset's abs=1e-3 (so it must
    // accept). A 1e-3 perturbation would sit exactly on the audio boundary,
    // where f32 rounding of e.g. (1.001 - 1.0) = 1.00005e-3 spuriously exceeds
    // the 1e-3 threshold.
    candidate
        .checkpoint("layer", &[1.0005, 2.0005, 3.0005, 4.0005], &[4])
        .expect("valid");

    // Standard preset should fail (5e-4 > abs=1e-5).
    let standard_report = compare_traces(
        &reference,
        &candidate,
        &TolerancePreset::STANDARD.to_config(),
    )
    .expect("should succeed");
    assert!(
        !standard_report.all_passed,
        "standard preset should reject 5e-4 diffs"
    );

    // Audio preset should pass (5e-4 < abs=1e-3).
    let audio_report = compare_traces(&reference, &candidate, &TolerancePreset::AUDIO.to_config())
        .expect("should succeed");
    assert!(
        audio_report.all_passed,
        "audio preset should accept 5e-4 diffs"
    );
}

#[test]
fn test_end_to_end_safetensors_f16_compare_with_relaxed() {
    // Create f16 reference, compare with f32 candidate using relaxed tolerance.
    let f16_values = [half::f16::from_f32(0.1),
        half::f16::from_f32(0.2),
        half::f16::from_f32(0.3)];
    let raw: Vec<u8> = f16_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes = build_safetensors_typed(&[("weights", &[3], safetensors::Dtype::F16, &raw)]);
    let reference = load_safetensors_from_bytes(&bytes).expect("load");

    // Build candidate with exact f32 values (slight difference from f16 quantization).
    let mut candidate = ReferenceTrace::new();
    candidate
        .checkpoint("weights", &[0.1, 0.2, 0.3], &[3])
        .expect("valid");

    // f16 quantization error is ~1e-3 to 1e-4. Relaxed (atol=1e-2) should pass.
    let report = compare_traces(&reference, &candidate, &ComparisonConfig::relaxed())
        .expect("should succeed");
    assert!(
        report.all_passed,
        "f16 quantization error should be within relaxed tolerance"
    );
}
