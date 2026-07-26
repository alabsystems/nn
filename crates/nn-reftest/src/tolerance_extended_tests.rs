// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tolerance and comparison tests for nn-reftest.
//!
//! Covers: absolute/relative/mixed/ULP/PercentClose tolerance strategies,
//! NaN/Inf handling, shape mismatch detection, empty tensors, large tensors,
//! multi-dimensional comparison, tolerance report generation, error formatting,
//! edge cases (all-zeros, very small values, mixed positive/negative),
//! safetensors loading utilities, and the assert_traces_match macro.

use crate::compare::{compare_tensors, compare_traces, ComparisonConfig};
use crate::error::ReftestError;
use crate::load::load_safetensors_from_bytes;
use crate::presets::TolerancePreset;
use crate::tolerance::{compare_with_tolerance, ComparisonResult, ToleranceStrategy};
use crate::trace::{NamedTensor, ReferenceTrace};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tensor(name: &str, shape: Vec<usize>, data: Vec<f32>) -> NamedTensor {
    NamedTensor::new(name, shape, data).expect("valid test tensor")
}

fn tensor_1d(name: &str, data: Vec<f32>) -> NamedTensor {
    let len = data.len();
    NamedTensor::new(name, vec![len], data).expect("valid 1-D test tensor")
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
// 1. Absolute tolerance: exact match at zero tolerance
// ===========================================================================

#[test]
fn test_abs_exact_match_zero_tolerance() {
    let data = [42.0f32, -17.5, 0.0, 1e30, -1e-30];
    let result = compare_with_tolerance(&data, &data, &ToleranceStrategy::Absolute { atol: 0.0 })
        .expect("comparison should succeed");
    assert!(result.passed);
    assert_eq!(result.max_diff, 0.0);
    assert_eq!(result.mean_diff, 0.0);
    assert_eq!(result.num_mismatches, 0);
}

// ===========================================================================
// 2. Absolute tolerance: just within boundary
// ===========================================================================

#[test]
fn test_abs_just_within_boundary() {
    // Difference is exactly at the tolerance edge. Use a power-of-two delta
    // (2^-13) so the f32 difference is exactly representable and equals the
    // f64 atol bit-for-bit. Using `1.0f32 + 1e-4` instead would round to a
    // difference of 0.00010001659..., spuriously exceeding atol = 1e-4.
    let delta = 2.0f32.powi(-13); // 0.0001220703125, exact in f32 and f64
    let actual = [1.0f32];
    let expected = [1.0f32 + delta];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute {
            atol: f64::from(delta),
        },
    )
    .expect("comparison should succeed");
    // The difference in f64 should be <= atol.
    assert!(result.passed, "boundary difference should pass");
}

// ===========================================================================
// 3. Absolute tolerance: just exceeds boundary
// ===========================================================================

#[test]
fn test_abs_just_exceeds_boundary() {
    let actual = [0.0f32];
    let expected = [2e-4f32];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 1e-4 },
    )
    .expect("comparison should succeed");
    assert!(!result.passed, "exceeding boundary should fail");
    assert_eq!(result.num_mismatches, 1);
}

// ===========================================================================
// 4. Absolute tolerance: multiple violations tracked
// ===========================================================================

#[test]
fn test_abs_multiple_violations() {
    let actual = [0.0f32, 0.0, 0.0, 0.0];
    let expected = [0.5, 0.5, 0.001, 0.5]; // 3 violations at atol=0.1
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 0.1 },
    )
    .expect("comparison should succeed");
    assert!(!result.passed);
    assert_eq!(result.num_mismatches, 3);
}

// ===========================================================================
// 5. Relative tolerance: proportional error on large values
// ===========================================================================

#[test]
fn test_rel_proportional_error_large_values() {
    // 0.1% error on values of 1000 and 10000
    let actual = [1001.0f32, 10010.0];
    let expected = [1000.0f32, 10000.0];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Relative { rtol: 0.002 },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "0.1% error should pass 0.2% tolerance");
}

// ===========================================================================
// 6. Relative tolerance: denominator uses max(|a|, |b|, eps)
// ===========================================================================

#[test]
fn test_rel_denominator_uses_max_abs() {
    // a=2.0, b=1.0, diff=1.0, denom=max(2.0, 1.0, 1e-8)=2.0, rel=0.5
    let result = compare_with_tolerance(
        &[2.0f32],
        &[1.0f32],
        &ToleranceStrategy::Relative { rtol: 0.5 },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "50% relative diff should pass 50% tolerance");
}

// ===========================================================================
// 7. Relative tolerance: epsilon floor prevents div-by-zero
// ===========================================================================

#[test]
fn test_rel_epsilon_floor_near_zero() {
    // Both values near zero: denom is clamped to 1e-8.
    // a=1e-10, b=2e-10, diff=1e-10, denom=1e-8, rel=0.01
    let result = compare_with_tolerance(
        &[1e-10f32],
        &[2e-10f32],
        &ToleranceStrategy::Relative { rtol: 0.02 },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "near-zero with epsilon floor should pass");
}

// ===========================================================================
// 8. Combined (Mixed) tolerance: atol dominates near zero
// ===========================================================================

#[test]
fn test_mixed_atol_dominates_small_values() {
    // a=0.0, b=1e-8, diff=1e-8
    // threshold = atol + rtol * |b| = 1e-6 + 1e-3 * 1e-8 = ~1e-6
    let result = compare_with_tolerance(
        &[0.0f32],
        &[1e-8f32],
        &ToleranceStrategy::Mixed {
            atol: 1e-6,
            rtol: 1e-3,
        },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "atol should dominate for small values");
}

// ===========================================================================
// 9. Combined (Mixed) tolerance: rtol dominates for large values
// ===========================================================================

#[test]
fn test_mixed_rtol_dominates_large_values() {
    // a=10000.0, b=10000.0 + 5.0 = 10005.0, diff=5.0
    // threshold = 1e-6 + 1e-3 * 10005.0 = 10.005001
    let result = compare_with_tolerance(
        &[10000.0f32],
        &[10005.0f32],
        &ToleranceStrategy::Mixed {
            atol: 1e-6,
            rtol: 1e-3,
        },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "rtol should provide margin for large values");
}

// ===========================================================================
// 10. Combined (Mixed) tolerance: both insufficient
// ===========================================================================

#[test]
fn test_mixed_both_insufficient_for_moderate_values() {
    // a=10.0, b=11.0, diff=1.0
    // threshold = 1e-6 + 1e-4 * 11.0 = 0.001101
    let result = compare_with_tolerance(
        &[10.0f32],
        &[11.0f32],
        &ToleranceStrategy::Mixed {
            atol: 1e-6,
            rtol: 1e-4,
        },
    )
    .expect("comparison should succeed");
    assert!(
        !result.passed,
        "neither atol nor rtol should cover 10% diff"
    );
}

// ===========================================================================
// 11. NaN vs NaN fails all tolerance strategies
// ===========================================================================

#[test]
fn test_nan_vs_nan_fails_all() {
    let strategies: Vec<ToleranceStrategy> = vec![
        ToleranceStrategy::Absolute { atol: f64::MAX },
        ToleranceStrategy::Relative { rtol: f64::MAX },
        ToleranceStrategy::Mixed {
            atol: f64::MAX,
            rtol: f64::MAX,
        },
        ToleranceStrategy::ULP { max_ulps: u32::MAX },
        ToleranceStrategy::PercentClose {
            threshold: f64::MAX,
            percent: 100.0,
        },
    ];
    for strategy in &strategies {
        let result = compare_with_tolerance(&[f32::NAN], &[f32::NAN], strategy)
            .expect("comparison should succeed");
        assert!(!result.passed, "NaN vs NaN should fail for {strategy:?}");
    }
}

// ===========================================================================
// 12. NaN in one position among valid data
// ===========================================================================

#[test]
fn test_nan_single_position_in_array() {
    let actual = [1.0f32, 2.0, f32::NAN, 4.0, 5.0];
    let expected = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 1.0 },
    )
    .expect("comparison should succeed");
    assert!(!result.passed);
    assert_eq!(result.num_mismatches, 1, "only NaN element should mismatch");
    assert!(
        result.max_diff.is_infinite(),
        "max_diff should be infinite due to NaN"
    );
}

// ===========================================================================
// 13. Inf vs finite value fails
// ===========================================================================

#[test]
fn test_inf_vs_finite_fails() {
    let result = compare_with_tolerance(
        &[f32::INFINITY],
        &[1.0f32],
        &ToleranceStrategy::Absolute { atol: f64::MAX },
    )
    .expect("comparison should succeed");
    assert!(!result.passed, "+Inf vs finite should fail");
    assert!(result.max_diff.is_infinite());
}

// ===========================================================================
// 14. Neg-Inf vs Neg-Inf still fails (non-finite policy)
// ===========================================================================

#[test]
fn test_neg_inf_vs_neg_inf_fails() {
    let result = compare_with_tolerance(
        &[f32::NEG_INFINITY],
        &[f32::NEG_INFINITY],
        &ToleranceStrategy::Absolute { atol: f64::MAX },
    )
    .expect("comparison should succeed");
    assert!(
        !result.passed,
        "matching neg-infinities should fail (non-finite)"
    );
}

// ===========================================================================
// 15. Mixed NaN and Inf in same array
// ===========================================================================

#[test]
fn test_mixed_nan_and_inf_in_array() {
    let actual = [1.0f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 5.0];
    let expected = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: f64::MAX },
    )
    .expect("comparison should succeed");
    assert!(!result.passed);
    assert_eq!(
        result.num_mismatches, 3,
        "3 non-finite elements should be mismatches"
    );
}

// ===========================================================================
// 16. Shape mismatch in compare_tensors
// ===========================================================================

#[test]
fn test_shape_mismatch_detected() {
    let ref_t = tensor("layer", vec![2, 3], vec![0.0; 6]);
    let cand_t = tensor("layer", vec![3, 2], vec![0.0; 6]);
    let err =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default()).expect_err("should fail");
    match err {
        ReftestError::ShapeMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, vec![2, 3]);
            assert_eq!(actual, vec![3, 2]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

// ===========================================================================
// 17. Shape mismatch: different ranks
// ===========================================================================

#[test]
fn test_shape_mismatch_different_ranks() {
    let ref_t = tensor("r", vec![6], vec![0.0; 6]);
    let cand_t = tensor("r", vec![2, 3], vec![0.0; 6]);
    let err =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default()).expect_err("should fail");
    assert!(matches!(err, ReftestError::ShapeMismatch { .. }));
}

// ===========================================================================
// 18. Empty tensor returns error in compare_tensors
// ===========================================================================

#[test]
fn test_empty_tensor_compare_tensors_error() {
    let ref_t = tensor("empty", vec![0], vec![]);
    let cand_t = tensor("empty", vec![0], vec![]);
    let err =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default()).expect_err("should fail");
    assert!(
        matches!(err, ReftestError::EmptyTensor(_)),
        "empty tensor comparison should return EmptyTensor error"
    );
}

// ===========================================================================
// 19. Empty slices in compare_with_tolerance
// ===========================================================================

#[test]
fn test_empty_slices_tolerance_error() {
    let err = compare_with_tolerance(&[], &[], &ToleranceStrategy::Absolute { atol: 1.0 })
        .expect_err("should fail on empty slices");
    assert!(matches!(err, ReftestError::EmptyTensor(_)));
}

// ===========================================================================
// 20. Large tensor: 100k elements, uniform perturbation
// ===========================================================================

#[test]
fn test_large_tensor_100k_uniform_perturbation() {
    let n = 100_000;
    let expected: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let actual: Vec<f32> = expected.iter().map(|&x| x + 1e-6).collect();
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 1e-5 },
    )
    .expect("comparison should succeed");
    assert!(result.passed);
    assert_eq!(result.num_mismatches, 0);
    assert!(result.max_diff < 2e-6);
}

// ===========================================================================
// 21. Large tensor: single outlier detection in 50k elements
// ===========================================================================

#[test]
fn test_large_tensor_single_outlier_detected() {
    let n = 50_000;
    let expected = vec![1.0f32; n];
    let mut actual = vec![1.0f32; n];
    actual[25_000] = 999.0;
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 0.01 },
    )
    .expect("comparison should succeed");
    assert!(!result.passed);
    assert_eq!(result.num_mismatches, 1);
    assert_eq!(result.worst_index, 25_000);
    assert!((result.max_diff - 998.0).abs() < 1e-3);
}

// ===========================================================================
// 22. Multi-dimensional tensor comparison (3-D)
// ===========================================================================

#[test]
fn test_multidim_3d_tensor_comparison() {
    let data = vec![1.0f32; 2 * 3 * 4];
    let ref_t = tensor("3d", vec![2, 3, 4], data.clone());
    let cand_t = tensor("3d", vec![2, 3, 4], data);
    let result =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.num_elements, 24);
    assert_eq!(result.shape, vec![2, 3, 4]);
}

// ===========================================================================
// 23. Multi-dimensional tensor: 4-D with small perturbation
// ===========================================================================

#[test]
fn test_multidim_4d_tensor_perturbation() {
    let n = 2 * 3 * 4 * 5;
    let ref_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let cand_data: Vec<f32> = ref_data.iter().map(|x| x + 1e-6).collect();
    let ref_t = tensor("4d", vec![2, 3, 4, 5], ref_data);
    let cand_t = tensor("4d", vec![2, 3, 4, 5], cand_data);
    let result =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed, "tiny perturbation on 4-D tensor should pass");
    assert_eq!(result.num_elements, n);
}

// ===========================================================================
// 24. Tolerance report: DivergenceReport summary for passing trace
// ===========================================================================

#[test]
fn test_divergence_report_summary_all_pass() {
    let trace = build_trace(&[("a", vec![1.0]), ("b", vec![2.0])]);
    let report =
        compare_traces(&trace, &trace, &ComparisonConfig::default()).expect("should succeed");
    let summary = report.summary();
    assert!(summary.contains("All 2 layers passed"));
    assert!(summary.contains("[PASS]"));
    assert!(!summary.contains("[FAIL]"));
}

// ===========================================================================
// 25. Tolerance report: DivergenceReport summary for failing trace
// ===========================================================================

#[test]
fn test_divergence_report_summary_with_failure() {
    let ref_trace = build_trace(&[("x", vec![1.0]), ("y", vec![2.0])]);
    let cand_trace = build_trace(&[("x", vec![1.0]), ("y", vec![999.0])]);
    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("should succeed");
    let summary = report.summary();
    assert!(
        summary.contains("First failure at layer 1"),
        "should show failure: {summary}"
    );
    assert!(summary.contains("[FAIL]"));
}

// ===========================================================================
// 26. Error message formatting: ShapeMismatch Display
// ===========================================================================

#[test]
fn test_error_display_shape_mismatch() {
    let err = ReftestError::ShapeMismatch {
        name: "conv1".to_string(),
        expected: vec![2, 3],
        actual: vec![3, 2],
    };
    let msg = format!("{err}");
    assert!(msg.contains("conv1"));
    assert!(msg.contains("[2, 3]"));
    assert!(msg.contains("[3, 2]"));
}

// ===========================================================================
// 27. Error message formatting: DataLengthMismatch Display
// ===========================================================================

#[test]
fn test_error_display_data_length_mismatch() {
    let err = ReftestError::DataLengthMismatch {
        expected: 12,
        actual: 8,
    };
    let msg = format!("{err}");
    assert!(msg.contains("12"));
    assert!(msg.contains("8"));
}

// ===========================================================================
// 28. Error message formatting: TraceLengthMismatch Display
// ===========================================================================

#[test]
fn test_error_display_trace_length_mismatch() {
    let err = ReftestError::TraceLengthMismatch {
        reference: 5,
        candidate: 3,
    };
    let msg = format!("{err}");
    assert!(msg.contains("5"));
    assert!(msg.contains("3"));
}

// ===========================================================================
// 29. Edge case: all-zeros tensor comparison
// ===========================================================================

#[test]
fn test_all_zeros_tensor() {
    let n = 100;
    let zeros = vec![0.0f32; n];
    let ref_t = tensor_1d("zeros", zeros.clone());
    let cand_t = tensor_1d("zeros", zeros);
    let result =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.max_abs_diff, 0.0);
    assert_eq!(result.max_rel_diff, 0.0);
    assert!(
        (result.cosine_similarity - 1.0).abs() < 1e-6,
        "zero vs zero cosine should be 1.0"
    );
}

// ===========================================================================
// 30. Edge case: very small (subnormal) values
// ===========================================================================

#[test]
fn test_subnormal_values_comparison() {
    let a = f32::MIN_POSITIVE / 16.0;
    let b = f32::MIN_POSITIVE / 8.0;
    let result = compare_with_tolerance(
        &[a],
        &[b],
        &ToleranceStrategy::Absolute {
            atol: f64::from(f32::MIN_POSITIVE),
        },
    )
    .expect("comparison should succeed");
    assert!(
        result.passed,
        "subnormal difference within MIN_POSITIVE tolerance"
    );
}

// ===========================================================================
// 31. Edge case: mixed positive and negative values
// ===========================================================================

#[test]
fn test_mixed_positive_negative_values() {
    let actual = [-5.0f32, 3.0, -1.0, 7.0, -0.5];
    let expected = [-5.0f32, 3.0, -1.0, 7.0, -0.5];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 0.0 },
    )
    .expect("comparison should succeed");
    assert!(result.passed);
    assert_eq!(result.num_mismatches, 0);
}

// ===========================================================================
// 32. Edge case: positive vs negative (sign flip)
// ===========================================================================

#[test]
fn test_sign_flip_large_difference() {
    let actual = [5.0f32, -5.0];
    let expected = [-5.0f32, 5.0];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 5.0 },
    )
    .expect("comparison should succeed");
    assert!(
        !result.passed,
        "sign flip gives diff=10 which exceeds atol=5"
    );
    assert!((result.max_diff - 10.0).abs() < 1e-6);
}

// ===========================================================================
// 33. ComparisonConfig::strict is tighter than default
// ===========================================================================

#[test]
fn test_comparison_config_strict_tighter_than_default() {
    let strict = ComparisonConfig::strict();
    let default = ComparisonConfig::default();
    assert!(
        strict.abs_tolerance < default.abs_tolerance,
        "strict abs should be tighter"
    );
    assert!(
        strict.rel_tolerance < default.rel_tolerance,
        "strict rel should be tighter"
    );
    assert!(
        strict.cosine_threshold > default.cosine_threshold,
        "strict cosine should be higher"
    );
}

// ===========================================================================
// 34. ComparisonConfig::relaxed is looser than default
// ===========================================================================

#[test]
fn test_comparison_config_relaxed_looser_than_default() {
    let relaxed = ComparisonConfig::relaxed();
    let default = ComparisonConfig::default();
    assert!(
        relaxed.abs_tolerance > default.abs_tolerance,
        "relaxed abs should be looser"
    );
    assert!(
        relaxed.rel_tolerance > default.rel_tolerance,
        "relaxed rel should be looser"
    );
    assert!(
        relaxed.cosine_threshold < default.cosine_threshold,
        "relaxed cosine should be lower"
    );
}

// ===========================================================================
// 35. RMS tolerance gate: fails when rms exceeds limit
// ===========================================================================

#[test]
fn test_rms_tolerance_gate_fails() {
    let ref_t = tensor_1d("rms", vec![0.0, 0.0, 0.0]);
    let cand_t = tensor_1d("rms", vec![0.1, 0.1, 0.1]);
    let config = ComparisonConfig::relaxed().with_rms_tolerance(0.01);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(
        !result.passed,
        "RMS ~0.1 should fail with 0.01 rms_tolerance"
    );
}

// ===========================================================================
// 36. Peak amplitude gate: fails when candidate peak exceeds limit
// ===========================================================================

#[test]
fn test_peak_amplitude_gate_fails() {
    let ref_t = tensor_1d("peak", vec![1.0, 2.0]);
    let cand_t = tensor_1d("peak", vec![1.0, 200.0]);
    let config = ComparisonConfig::relaxed().with_peak_amplitude_limit(100.0);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(
        !result.passed,
        "peak amplitude 200 should fail with limit 100"
    );
    assert_eq!(result.peak_amplitude, 200.0);
}

// ===========================================================================
// 37. ULP: positive and negative zero are 0 ULPs apart
// ===========================================================================

#[test]
fn test_ulp_pos_neg_zero() {
    let result = compare_with_tolerance(
        &[0.0f32],
        &[-0.0f32],
        &ToleranceStrategy::ULP { max_ulps: 0 },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "+0.0 and -0.0 should be 0 ULPs apart");
}

// ===========================================================================
// 38. ULP: large distance between 1.0 and 2.0
// ===========================================================================

#[test]
fn test_ulp_large_distance_1_to_2() {
    // 1.0 and 2.0 are 2^23 ULPs apart (the mantissa range of f32).
    let result = compare_with_tolerance(
        &[1.0f32],
        &[2.0f32],
        &ToleranceStrategy::ULP { max_ulps: 100 },
    )
    .expect("comparison should succeed");
    assert!(
        !result.passed,
        "1.0 vs 2.0 is millions of ULPs apart, 100 should not suffice"
    );
}

// ===========================================================================
// 39. PercentClose: exactly at the percentage boundary
// ===========================================================================

#[test]
fn test_percent_close_exact_boundary() {
    // 9 out of 10 close = 90%, require 90%
    let actual = [0.0f32; 10];
    let mut expected = [0.0f32; 10];
    expected[9] = 100.0; // one outlier
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::PercentClose {
            threshold: 0.01,
            percent: 90.0,
        },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "90% close == 90% requirement should pass");
}

// ===========================================================================
// 40. PercentClose: just below the percentage boundary
// ===========================================================================

#[test]
fn test_percent_close_just_below_boundary() {
    // 8 out of 10 close = 80%, require 81%
    let actual = [0.0f32; 10];
    let mut expected = [0.0f32; 10];
    expected[8] = 100.0;
    expected[9] = 100.0; // two outliers
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::PercentClose {
            threshold: 0.01,
            percent: 81.0,
        },
    )
    .expect("comparison should succeed");
    assert!(!result.passed, "80% close < 81% requirement should fail");
}

// ===========================================================================
// 41. Safetensors load: verify alphabetical ordering with many tensors
// ===========================================================================

#[test]
fn test_safetensors_alphabetical_ordering_many() {
    let bytes = build_safetensors(&[
        ("z_out", &[1], &[5.0]),
        ("b_mid", &[1], &[2.0]),
        ("a_in", &[1], &[1.0]),
        ("m_hidden", &[1], &[3.0]),
        ("x_logit", &[1], &[4.0]),
    ]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");
    let names: Vec<&str> = trace.names().collect();
    assert_eq!(names, vec!["a_in", "b_mid", "m_hidden", "x_logit", "z_out"]);
}

// ===========================================================================
// 42. Safetensors load then trace comparison: identical
// ===========================================================================

#[test]
fn test_safetensors_load_compare_identical() {
    let data = vec![0.5, 1.5, 2.5];
    let bytes = build_safetensors(&[("weights", &[3], &data)]);
    let ref_trace = load_safetensors_from_bytes(&bytes).expect("load reference");
    let cand_trace = load_safetensors_from_bytes(&bytes).expect("load candidate");
    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("should succeed");
    assert!(report.all_passed);
}

// ===========================================================================
// 43. Safetensors load then trace comparison: divergent
// ===========================================================================

#[test]
fn test_safetensors_load_compare_divergent() {
    let ref_bytes = build_safetensors(&[("w", &[2], &[1.0, 2.0])]);
    let cand_bytes = build_safetensors(&[("w", &[2], &[1.0, 999.0])]);
    let ref_trace = load_safetensors_from_bytes(&ref_bytes).expect("load reference");
    let cand_trace = load_safetensors_from_bytes(&cand_bytes).expect("load candidate");
    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("should succeed");
    assert!(!report.all_passed);
    assert_eq!(report.first_failure, Some(0));
}

// ===========================================================================
// 44. Safetensors: invalid bytes rejected
// ===========================================================================

#[test]
fn test_safetensors_invalid_bytes() {
    let result = load_safetensors_from_bytes(b"garbage data not safetensors");
    assert!(matches!(result, Err(ReftestError::Safetensors(_))));
}

// ===========================================================================
// 45. TolerancePreset lookup by name
// ===========================================================================

#[test]
fn test_tolerance_preset_by_name() {
    assert_eq!(
        TolerancePreset::by_name("strict"),
        Some(TolerancePreset::STRICT)
    );
    assert_eq!(
        TolerancePreset::by_name("TRANSFORMER"),
        Some(TolerancePreset::TRANSFORMER)
    );
    assert!(TolerancePreset::by_name("nonexistent").is_none());
}

// ===========================================================================
// 46. TolerancePreset to_config produces correct thresholds
// ===========================================================================

#[test]
fn test_tolerance_preset_to_config() {
    let config = TolerancePreset::AUDIO.to_config();
    assert_eq!(config.abs_tolerance, 1e-3);
    assert_eq!(config.rel_tolerance, 1e-2);
    assert!((config.cosine_threshold - 0.99).abs() < 1e-6);
}

// ===========================================================================
// 47. ComparisonResult: Clone and Debug
// ===========================================================================

#[test]
fn test_comparison_result_clone_debug() {
    let result = compare_with_tolerance(
        &[1.0f32, 2.0],
        &[1.0f32, 2.0],
        &ToleranceStrategy::Absolute { atol: 1.0 },
    )
    .expect("should succeed");
    let cloned: ComparisonResult = result.clone();
    assert_eq!(cloned.passed, result.passed);
    assert_eq!(cloned.max_diff, result.max_diff);
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("passed"));
}

// ===========================================================================
// 48. LayerComparison Display includes key metrics
// ===========================================================================

#[test]
fn test_layer_comparison_display() {
    let ref_t = tensor_1d("test_layer", vec![1.0, 2.0, 3.0]);
    let cand_t = tensor_1d("test_layer", vec![1.1, 2.1, 3.1]);
    let result =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::relaxed()).expect("should succeed");
    let display = format!("{result}");
    assert!(display.contains("test_layer"), "should show layer name");
    assert!(display.contains("[3]"), "should show shape: {display}");
    assert!(
        display.contains("max_abs=") || display.contains("PASS") || display.contains("FAIL"),
        "should show metrics or status"
    );
}

// ===========================================================================
// 49. Trace comparison: trace length mismatch error
// ===========================================================================

#[test]
fn test_trace_length_mismatch_error_details() {
    let ref_trace = build_trace(&[("a", vec![1.0]), ("b", vec![2.0]), ("c", vec![3.0])]);
    let cand_trace = build_trace(&[("a", vec![1.0])]);
    let err = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect_err("should fail");
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
// 50. Data length mismatch in compare_with_tolerance
// ===========================================================================

#[test]
fn test_data_length_mismatch_tolerance() {
    let err = compare_with_tolerance(
        &[1.0f32, 2.0, 3.0],
        &[1.0f32, 2.0],
        &ToleranceStrategy::Absolute { atol: 1.0 },
    )
    .expect_err("should fail on length mismatch");
    match err {
        ReftestError::DataLengthMismatch { expected, actual } => {
            assert_eq!(expected, 2);
            assert_eq!(actual, 3);
        }
        other => panic!("expected DataLengthMismatch, got {other:?}"),
    }
}

// ===========================================================================
// 51. Cosine similarity: orthogonal vectors
// ===========================================================================

#[test]
fn test_cosine_orthogonal_vectors() {
    let ref_t = tensor_1d("ortho", vec![1.0, 0.0]);
    let cand_t = tensor_1d("ortho", vec![0.0, 1.0]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.5);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(
        result.cosine_similarity.abs() < 1e-6,
        "orthogonal vectors should have cosine ~0, got {}",
        result.cosine_similarity,
    );
    assert!(!result.passed, "cosine ~0 should fail threshold 0.5");
}

// ===========================================================================
// 52. Cosine similarity: anti-parallel vectors
// ===========================================================================

#[test]
fn test_cosine_anti_parallel_vectors() {
    let ref_t = tensor_1d("anti", vec![1.0, 2.0, 3.0]);
    let cand_t = tensor_1d("anti", vec![-1.0, -2.0, -3.0]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.0);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(
        (result.cosine_similarity - (-1.0)).abs() < 1e-5,
        "anti-parallel vectors should have cosine ~-1.0, got {}",
        result.cosine_similarity,
    );
}

// ===========================================================================
// 53. assert_traces_match macro: passing case
// ===========================================================================

#[test]
fn test_assert_traces_match_macro_passing() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace
        .checkpoint("layer", &[1.0, 2.0, 3.0], &[3])
        .expect("valid");
    let mut cand_trace = ReferenceTrace::new();
    cand_trace
        .checkpoint("layer", &[1.0, 2.0, 3.0], &[3])
        .expect("valid");
    // Should not panic.
    crate::assert_traces_match!(cand_trace, ref_trace);
}

// ===========================================================================
// 54. assert_traces_match macro: with custom tolerance
// ===========================================================================

#[test]
fn test_assert_traces_match_macro_custom_tolerance() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace.checkpoint("x", &[1.0, 2.0], &[2]).expect("valid");
    let mut cand_trace = ReferenceTrace::new();
    cand_trace
        .checkpoint("x", &[1.01, 2.01], &[2])
        .expect("valid");
    // With tight tolerance this would fail, but with relaxed it should pass.
    crate::assert_traces_match!(cand_trace, ref_trace, abs = 0.1, rel = 0.1, cos = 0.99);
}

// ===========================================================================
// 55. assert_traces_match macro: epsilon variant
// ===========================================================================

#[test]
fn test_assert_traces_match_macro_epsilon_variant() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace.checkpoint("v", &[10.0], &[1]).expect("valid");
    let mut cand_trace = ReferenceTrace::new();
    cand_trace
        .checkpoint("v", &[10.0 + 1e-6], &[1])
        .expect("valid");
    crate::assert_traces_match!(cand_trace, ref_trace, epsilon = 1e-4);
}

// ===========================================================================
// 56. Scalar tensor comparison (shape = [])
// ===========================================================================

#[test]
fn test_scalar_tensor_comparison() {
    let ref_t = tensor("scalar", vec![], vec![3.14]);
    let cand_t = tensor("scalar", vec![], vec![3.14]);
    let result =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.num_elements, 1);
}

// ===========================================================================
// 57. Very large values: f32 near MAX
// ===========================================================================

#[test]
fn test_very_large_f32_values() {
    let big = f32::MAX / 2.0;
    let ref_t = tensor_1d("big", vec![big, big]);
    let cand_t = tensor_1d("big", vec![big, big]);
    let config = ComparisonConfig::new(1.0, 1e-4, 0.999);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(result.passed, "identical large values should pass");
}

// ===========================================================================
// 58. Single element: worst_index is always 0
// ===========================================================================

#[test]
fn test_single_element_worst_index() {
    let result = compare_with_tolerance(
        &[5.0f32],
        &[10.0f32],
        &ToleranceStrategy::Absolute { atol: 100.0 },
    )
    .expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.worst_index, 0);
    assert!((result.max_diff - 5.0).abs() < 1e-6);
}

// ===========================================================================
// 59. Mean diff accuracy with known exact values
// ===========================================================================

#[test]
fn test_mean_diff_exact_known_values() {
    // diffs: 1.0, 2.0, 3.0, 4.0 -> mean = 2.5
    let actual = [1.0f32, 2.0, 3.0, 4.0];
    let expected = [0.0f32, 0.0, 0.0, 0.0];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 100.0 },
    )
    .expect("should succeed");
    assert!(
        (result.mean_diff - 2.5).abs() < 1e-10,
        "mean_diff should be 2.5, got {}",
        result.mean_diff,
    );
    assert!((result.max_diff - 4.0).abs() < 1e-10);
    assert_eq!(result.worst_index, 3);
}

// ===========================================================================
// 60. TolerancePreset::ALL covers all named presets
// ===========================================================================

#[test]
fn test_tolerance_preset_all_covers_all() {
    assert_eq!(TolerancePreset::ALL.len(), 6);
    let names: Vec<&str> = TolerancePreset::ALL.iter().map(|p| p.name).collect();
    assert!(names.contains(&"strict"));
    assert!(names.contains(&"standard"));
    assert!(names.contains(&"transformer"));
    assert!(names.contains(&"audio"));
    assert!(names.contains(&"quantized"));
    assert!(names.contains(&"tts"));
}

// ===========================================================================
// 61. Comparison with monotonically increasing values
// ===========================================================================

#[test]
fn test_monotonic_values_comparison() {
    let n = 200;
    let ref_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let cand_data: Vec<f32> = ref_data.iter().map(|x| x + 1e-6).collect();
    let ref_t = tensor_1d("mono", ref_data);
    let cand_t = tensor_1d("mono", cand_data);
    let result =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed);
    assert!(result.max_abs_diff < 2e-6);
}

// ===========================================================================
// 62. NamedTensor::numel matches shape product
// ===========================================================================

#[test]
fn test_named_tensor_numel_matches_shape() {
    let t = tensor("shape_check", vec![3, 4, 5], vec![0.0; 60]);
    assert_eq!(t.numel(), 60);
    assert_eq!(t.shape, vec![3, 4, 5]);
}

// ===========================================================================
// 63. ReferenceTrace capture closure returns value
// ===========================================================================

#[test]
fn test_reference_trace_capture_closure_output() {
    let (trace, output) = ReferenceTrace::capture(|cap| {
        cap.checkpoint("h1", &[1.0], &[1]).expect("valid");
        cap.checkpoint("h2", &[2.0, 3.0], &[2]).expect("valid");
        "hello"
    });
    assert_eq!(output, "hello");
    assert_eq!(trace.len(), 2);
}

// ===========================================================================
// 64. Trace names iteration order matches insertion
// ===========================================================================

#[test]
fn test_trace_names_insertion_order() {
    let trace = build_trace(&[
        ("first", vec![1.0]),
        ("second", vec![2.0]),
        ("third", vec![3.0]),
    ]);
    let names: Vec<&str> = trace.names().collect();
    assert_eq!(names, vec!["first", "second", "third"]);
}
