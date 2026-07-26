// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for nn-reftest: tolerance comparison (absolute, relative,
//! ULP, mixed, percent-close), shape/dtype mismatch detection, NaN/Inf handling,
//! safetensors loading validation, NPY format parsing, trace comparison with
//! multiple tensors, per-element error reporting, statistical comparison,
//! reference data versioning, batch comparison, empty/scalar/high-dimensional
//! tensor handling, mixed precision comparison, large tensor performance,
//! error message formatting, and comparison report generation.

use crate::compare::{compare_tensors, compare_traces, ComparisonConfig};
use crate::error::ReftestError;
use crate::load::load_safetensors_from_bytes;
use crate::npy::{read_npy_from_bytes, write_npy_to_bytes, NpyDType};
use crate::presets::TolerancePreset;
use crate::tolerance::{compare_with_tolerance, ToleranceStrategy};
use crate::trace::{NamedTensor, ReferenceTrace};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tensor(name: &str, shape: Vec<usize>, data: Vec<f32>) -> NamedTensor {
    NamedTensor::new(name, shape, data).expect("valid test tensor")
}

fn tensor_1d(name: &str, data: Vec<f32>) -> NamedTensor {
    let len = data.len();
    tensor(name, vec![len], data)
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

/// Build a minimal safetensors byte buffer from f32 tensors.
fn build_safetensors(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
    let byte_bufs: Vec<Vec<u8>> = tensors
        .iter()
        .map(|&(_, _, data)| {
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes
        })
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

/// Build a safetensors byte buffer with a specified dtype.
fn build_safetensors_typed(tensors: &[(&str, &[usize], &[u8], safetensors::Dtype)]) -> Vec<u8> {
    let mut tensor_map: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
    for &(name, shape, data, dtype) in tensors {
        let view = safetensors::tensor::TensorView::new(dtype, shape.to_vec(), data)
            .expect("valid tensor view");
        tensor_map.push((name.to_string(), view));
    }
    safetensors::tensor::serialize(tensor_map, None).expect("serialization should succeed")
}

// ===========================================================================
// 1. Absolute tolerance: exact match
// ===========================================================================

#[test]
fn test_abs_tolerance_exact_match() {
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [1.0_f32, 2.0, 3.0, 4.0];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 0.0 })
        .expect("should succeed");
    assert!(result.passed, "identical values must pass atol=0");
    assert_eq!(result.max_diff, 0.0);
    assert_eq!(result.num_mismatches, 0);
}

// ===========================================================================
// 2. Absolute tolerance: within bounds
// ===========================================================================

#[test]
fn test_abs_tolerance_within_bounds() {
    let a = [1.001_f32, 2.002, 3.003];
    let b = [1.0_f32, 2.0, 3.0];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 0.01 })
        .expect("should succeed");
    assert!(result.passed, "diffs <= 0.003 should pass atol=0.01");
    assert!(result.max_diff < 0.004);
}

// ===========================================================================
// 3. Absolute tolerance: exceeds bounds
// ===========================================================================

#[test]
fn test_abs_tolerance_exceeds_bounds() {
    let a = [1.0_f32, 2.0, 3.5];
    let b = [1.0_f32, 2.0, 3.0];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 0.1 })
        .expect("should succeed");
    assert!(!result.passed, "diff=0.5 should fail atol=0.1");
    assert_eq!(result.num_mismatches, 1);
    assert_eq!(result.worst_index, 2);
}

// ===========================================================================
// 4. Relative tolerance: exact match
// ===========================================================================

#[test]
fn test_rel_tolerance_exact_match() {
    let a = [10.0_f32, 20.0, 30.0];
    let result = compare_with_tolerance(&a, &a, &ToleranceStrategy::Relative { rtol: 0.0 })
        .expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.max_diff, 0.0);
}

// ===========================================================================
// 5. Relative tolerance: proportional difference passes
// ===========================================================================

#[test]
fn test_rel_tolerance_proportional_pass() {
    // 1% of 100.0 = 1.0; diff = 0.5 < 1.0
    let a = [100.5_f32];
    let b = [100.0_f32];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Relative { rtol: 0.01 })
        .expect("should succeed");
    assert!(result.passed, "0.5% diff should pass rtol=1%");
}

// ===========================================================================
// 6. Relative tolerance: proportional difference fails
// ===========================================================================

#[test]
fn test_rel_tolerance_proportional_fail() {
    // 1% of 100.0 = 1.0; diff = 5.0 > 1.0
    let a = [105.0_f32];
    let b = [100.0_f32];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Relative { rtol: 0.01 })
        .expect("should succeed");
    assert!(!result.passed, "5% diff should fail rtol=1%");
}

// ===========================================================================
// 7. ULP comparison: identical values
// ===========================================================================

#[test]
fn test_ulp_identical_values() {
    let a = [1.0_f32, -0.5, 0.0, 1e10];
    let result = compare_with_tolerance(&a, &a, &ToleranceStrategy::ULP { max_ulps: 0 })
        .expect("should succeed");
    assert!(result.passed, "identical values should be 0 ULPs apart");
}

// ===========================================================================
// 8. ULP comparison: adjacent floats within tolerance
// ===========================================================================

#[test]
fn test_ulp_adjacent_floats_pass() {
    let a = 1.0_f32;
    let b = f32::from_bits(a.to_bits() + 2); // 2 ULPs apart
    let result = compare_with_tolerance(&[a], &[b], &ToleranceStrategy::ULP { max_ulps: 2 })
        .expect("should succeed");
    assert!(result.passed, "2-ULP apart should pass max_ulps=2");
}

// ===========================================================================
// 9. ULP comparison: adjacent floats fail
// ===========================================================================

#[test]
fn test_ulp_adjacent_floats_fail() {
    let a = 1.0_f32;
    let b = f32::from_bits(a.to_bits() + 3); // 3 ULPs apart
    let result = compare_with_tolerance(&[a], &[b], &ToleranceStrategy::ULP { max_ulps: 2 })
        .expect("should succeed");
    assert!(!result.passed, "3-ULP apart should fail max_ulps=2");
}

// ===========================================================================
// 10. ULP comparison: NaN always fails
// ===========================================================================

#[test]
fn test_ulp_nan_always_fails() {
    let result = compare_with_tolerance(
        &[f32::NAN],
        &[f32::NAN],
        &ToleranceStrategy::ULP { max_ulps: u32::MAX },
    )
    .expect("should succeed");
    assert!(!result.passed, "NaN vs NaN should fail ULP comparison");
    assert_eq!(result.num_mismatches, 1);
}

// ===========================================================================
// 11. Mixed (NumPy-style) tolerance: combined atol+rtol
// ===========================================================================

#[test]
fn test_mixed_tolerance_combined() {
    // |a - b| <= atol + rtol * |b|
    // diff = 0.15, atol = 0.1, rtol = 0.01, |b| = 10.0
    // threshold = 0.1 + 0.01 * 10.0 = 0.2 => passes
    let result = compare_with_tolerance(
        &[10.15_f32],
        &[10.0_f32],
        &ToleranceStrategy::Mixed {
            atol: 0.1,
            rtol: 0.01,
        },
    )
    .expect("should succeed");
    assert!(
        result.passed,
        "diff=0.15 should pass mixed(0.1, 0.01) at b=10"
    );
}

// ===========================================================================
// 12. Mixed tolerance: fails when both atol and rtol insufficient
// ===========================================================================

#[test]
fn test_mixed_tolerance_fails() {
    // diff = 5.0, atol = 0.1, rtol = 0.01, |b| = 100.0
    // threshold = 0.1 + 0.01 * 100.0 = 1.1 => 5.0 > 1.1 => fails
    let result = compare_with_tolerance(
        &[105.0_f32],
        &[100.0_f32],
        &ToleranceStrategy::Mixed {
            atol: 0.1,
            rtol: 0.01,
        },
    )
    .expect("should succeed");
    assert!(
        !result.passed,
        "diff=5 should fail mixed(0.1, 0.01) at b=100"
    );
}

// ===========================================================================
// 13. PercentClose: exactly at threshold boundary
// ===========================================================================

#[test]
fn test_percent_close_boundary() {
    // 3 out of 4 close = 75%. Require exactly 75%.
    let a = [1.0_f32, 2.0, 3.0, 100.0];
    let b = [1.0_f32, 2.0, 3.0, 4.0];
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::PercentClose {
            threshold: 0.5,
            percent: 75.0,
        },
    )
    .expect("should succeed");
    assert!(result.passed, "75% close with 75% requirement should pass");
}

// ===========================================================================
// 14. Shape mismatch detection in compare_tensors
// ===========================================================================

#[test]
fn test_shape_mismatch_detection() {
    let ref_t = tensor("a", vec![2, 3], vec![0.0; 6]);
    let cand_t = tensor("a", vec![3, 2], vec![0.0; 6]);
    let err = compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default())
        .expect_err("should fail on shape mismatch");
    match err {
        ReftestError::ShapeMismatch {
            name,
            expected,
            actual,
        } => {
            assert_eq!(name, "a");
            assert_eq!(expected, vec![2, 3]);
            assert_eq!(actual, vec![3, 2]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

// ===========================================================================
// 15. Shape mismatch: rank difference
// ===========================================================================

#[test]
fn test_shape_mismatch_rank_difference() {
    let ref_t = tensor("b", vec![6], vec![0.0; 6]);
    let cand_t = tensor("b", vec![2, 3], vec![0.0; 6]);
    let err = compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default())
        .expect_err("different ranks should produce shape mismatch");
    assert!(matches!(err, ReftestError::ShapeMismatch { .. }));
}

// ===========================================================================
// 16. DType mismatch in safetensors: unsupported dtype
// ===========================================================================

#[test]
fn test_dtype_mismatch_unsupported() {
    // Build safetensors with BOOL dtype (unsupported)
    let bytes = build_safetensors_typed(&[("bad", &[1], &[1u8], safetensors::Dtype::BOOL)]);
    let err = load_safetensors_from_bytes(&bytes).expect_err("BOOL dtype should fail");
    assert!(
        matches!(err, ReftestError::UnsupportedDtype(_)),
        "expected UnsupportedDtype, got {err:?}"
    );
}

// ===========================================================================
// 17. NaN handling: NaN in reference causes failure
// ===========================================================================

#[test]
fn test_nan_in_reference_causes_failure() {
    let ref_t = tensor_1d("nan_ref", vec![f32::NAN, 1.0]);
    let cand_t = tensor_1d("nan_ref", vec![0.0, 1.0]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.0);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(!result.passed, "NaN in reference should fail");
    assert!(result.max_abs_diff.is_infinite());
}

// ===========================================================================
// 18. NaN handling: NaN in candidate causes failure
// ===========================================================================

#[test]
fn test_nan_in_candidate_causes_failure() {
    let ref_t = tensor_1d("nan_cand", vec![0.0, 1.0]);
    let cand_t = tensor_1d("nan_cand", vec![0.0, f32::NAN]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.0);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(!result.passed, "NaN in candidate should fail");
    assert!(result.peak_amplitude.is_infinite());
}

// ===========================================================================
// 19. NaN handling: NaN vs NaN is not equal (IEEE 754)
// ===========================================================================

#[test]
fn test_nan_vs_nan_not_equal() {
    let t = tensor_1d("nan_both", vec![f32::NAN]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.0);
    let result = compare_tensors(&t, &t.clone(), &config).expect("should succeed");
    assert!(!result.passed, "NaN == NaN violates IEEE 754");
}

// ===========================================================================
// 20. Inf handling: +Inf causes peak_amplitude to be infinite
// ===========================================================================

#[test]
fn test_inf_causes_infinite_peak_amplitude() {
    let ref_t = tensor_1d("inf", vec![1.0, 2.0]);
    let cand_t = tensor_1d("inf", vec![1.0, f32::INFINITY]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.0);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(result.peak_amplitude.is_infinite());
}

// ===========================================================================
// 21. Inf handling: -Inf treated as non-finite
// ===========================================================================

#[test]
fn test_neg_inf_non_finite() {
    let ref_t = tensor_1d("neginf", vec![1.0]);
    let cand_t = tensor_1d("neginf", vec![f32::NEG_INFINITY]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.0);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(!result.passed, "-Inf should fail");
    assert!(result.max_abs_diff.is_infinite());
}

// ===========================================================================
// 22. NaN in tolerance comparison: always counts as mismatch
// ===========================================================================

#[test]
fn test_nan_in_tolerance_comparison() {
    let result = compare_with_tolerance(
        &[f32::NAN, 1.0],
        &[0.0, 1.0],
        &ToleranceStrategy::Absolute { atol: f64::MAX },
    )
    .expect("should succeed");
    assert!(!result.passed, "NaN should count as mismatch");
    assert_eq!(result.num_mismatches, 1);
    assert!(result.max_diff.is_infinite());
}

// ===========================================================================
// 23. Safetensors: multiple tensors loaded in alphabetical order
// ===========================================================================

#[test]
fn test_safetensors_alphabetical_ordering() {
    let bytes = build_safetensors(&[
        ("z_weight", &[2], &[3.0, 4.0]),
        ("a_bias", &[1], &[0.5]),
        ("m_layer", &[3], &[1.0, 2.0, 3.0]),
    ]);
    let trace = load_safetensors_from_bytes(&bytes).expect("should load");
    let names: Vec<&str> = trace.names().collect();
    assert_eq!(names, vec!["a_bias", "m_layer", "z_weight"]);
}

// ===========================================================================
// 24. Safetensors: invalid bytes rejected
// ===========================================================================

#[test]
fn test_safetensors_invalid_bytes() {
    let err =
        load_safetensors_from_bytes(b"not safetensors").expect_err("invalid bytes should fail");
    assert!(matches!(err, ReftestError::Safetensors(_)));
}

// ===========================================================================
// 25. Safetensors: empty file (no tensors)
// ===========================================================================

#[test]
fn test_safetensors_empty_file() {
    let bytes = build_safetensors(&[]);
    let trace = load_safetensors_from_bytes(&bytes).expect("empty safetensors should load");
    assert!(trace.is_empty());
}

// ===========================================================================
// 26. Safetensors: BF16 tensor loaded and converted to f32
// ===========================================================================

#[test]
fn test_safetensors_bf16_conversion() {
    let bf16_vals = [
        half::bf16::from_f32(1.0),
        half::bf16::from_f32(-2.0),
        half::bf16::from_f32(0.5),
    ];
    let raw: Vec<u8> = bf16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes = build_safetensors_typed(&[("bf16_t", &[3], &raw, safetensors::Dtype::BF16)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("bf16 should load");
    let t = trace.get(0).expect("exists");
    assert!((t.data[0] - 1.0).abs() < 0.02);
    assert!((t.data[1] - (-2.0)).abs() < 0.02);
    assert!((t.data[2] - 0.5).abs() < 0.02);
}

// ===========================================================================
// 27. Safetensors: F16 tensor loaded and converted to f32
// ===========================================================================

#[test]
fn test_safetensors_f16_conversion() {
    let f16_vals = [half::f16::from_f32(3.14), half::f16::from_f32(-1.5)];
    let raw: Vec<u8> = f16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes = build_safetensors_typed(&[("f16_t", &[2], &raw, safetensors::Dtype::F16)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("f16 should load");
    let t = trace.get(0).expect("exists");
    assert!((t.data[0] - 3.14).abs() < 0.01);
    assert!((t.data[1] - (-1.5)).abs() < 0.01);
}

// ===========================================================================
// 28. NPY: roundtrip 2D tensor preserves shape and data
// ===========================================================================

#[test]
fn test_npy_roundtrip_2d() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3];
    let bytes = write_npy_to_bytes(&data, &shape).expect("write should succeed");
    let t = read_npy_from_bytes(&bytes).expect("read should succeed");
    assert_eq!(t.shape, shape);
    assert_eq!(t.data, data);
    assert_eq!(t.dtype, NpyDType::F32);
}

// ===========================================================================
// 29. NPY: bad magic rejected
// ===========================================================================

#[test]
fn test_npy_bad_magic() {
    let result = read_npy_from_bytes(b"NOT_NPY_DATA");
    assert!(result.is_err(), "bad magic should be rejected");
}

// ===========================================================================
// 30. NPY: scalar (empty shape) roundtrip
// ===========================================================================

#[test]
fn test_npy_scalar_roundtrip() {
    let data = vec![42.0_f32];
    let bytes = write_npy_to_bytes(&data, &[]).expect("write should succeed");
    let t = read_npy_from_bytes(&bytes).expect("read should succeed");
    assert!(t.shape.is_empty());
    assert_eq!(t.data, vec![42.0]);
}

// ===========================================================================
// 31. NPY: 1D roundtrip with many elements
// ===========================================================================

#[test]
fn test_npy_1d_many_elements() {
    let data: Vec<f32> = (0..500).map(|i| i as f32 * 0.01).collect();
    let bytes = write_npy_to_bytes(&data, &[500]).expect("write should succeed");
    let t = read_npy_from_bytes(&bytes).expect("read should succeed");
    assert_eq!(t.shape, vec![500]);
    assert_eq!(t.data.len(), 500);
    assert!((t.data[0]).abs() < 1e-6);
    assert!((t.data[499] - 4.99).abs() < 1e-4);
}

// ===========================================================================
// 32. Trace comparison: multiple tensors all pass
// ===========================================================================

#[test]
fn test_trace_multi_tensor_all_pass() {
    let layers = vec![
        ("embed", vec![0.1, 0.2, 0.3]),
        ("attn", vec![1.0, 2.0]),
        ("ffn", vec![5.0, 6.0, 7.0, 8.0]),
        ("norm", vec![0.0, 0.0, 0.0]),
    ];
    let ref_t = build_trace(&layers);
    let cand_t = build_trace(&layers);
    let report =
        compare_traces(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert!(report.all_passed);
    assert_eq!(report.layers.len(), 4);
    assert!(report.first_failure.is_none());
}

// ===========================================================================
// 33. Trace comparison: failure at a specific layer index
// ===========================================================================

#[test]
fn test_trace_failure_at_specific_layer() {
    let ref_t = build_trace(&[("a", vec![1.0]), ("b", vec![2.0]), ("c", vec![3.0])]);
    let cand_t = build_trace(&[("a", vec![1.0]), ("b", vec![2.0]), ("c", vec![999.0])]);
    let report =
        compare_traces(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert!(!report.all_passed);
    assert_eq!(report.first_failure, Some(2));
    assert!(report.layers[0].passed);
    assert!(report.layers[1].passed);
    assert!(!report.layers[2].passed);
}

// ===========================================================================
// 34. Trace comparison: length mismatch
// ===========================================================================

#[test]
fn test_trace_length_mismatch() {
    let ref_t = build_trace(&[("a", vec![1.0]), ("b", vec![2.0])]);
    let cand_t = build_trace(&[("a", vec![1.0])]);
    let err =
        compare_traces(&ref_t, &cand_t, &ComparisonConfig::default()).expect_err("should fail");
    match err {
        ReftestError::TraceLengthMismatch {
            reference,
            candidate,
        } => {
            assert_eq!(reference, 2);
            assert_eq!(candidate, 1);
        }
        other => panic!("expected TraceLengthMismatch, got {other:?}"),
    }
}

// ===========================================================================
// 35. Per-element error: worst_index is correct
// ===========================================================================

#[test]
fn test_per_element_worst_index() {
    let a = [0.0_f32, 0.0, 0.0, 0.0, 0.0];
    let b = [0.01, 0.02, 0.5, 0.03, 0.01];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1.0 })
        .expect("should succeed");
    assert_eq!(result.worst_index, 2, "worst_index should be 2 (diff=0.5)");
    assert!((result.max_diff - 0.5).abs() < 1e-6);
}

// ===========================================================================
// 36. Per-element error: num_mismatches counts correctly
// ===========================================================================

#[test]
fn test_per_element_num_mismatches() {
    // Diffs: 0.0, 0.5, 0.0, 1.0, 0.0 — 2 exceed atol=0.1
    let a = [1.0_f32, 1.5, 2.0, 3.0, 4.0];
    let b = [1.0_f32, 1.0, 2.0, 2.0, 4.0];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 0.1 })
        .expect("should succeed");
    assert_eq!(result.num_mismatches, 2);
    assert!(!result.passed);
}

// ===========================================================================
// 37. Statistical comparison: mean_diff computed correctly
// ===========================================================================

#[test]
fn test_statistical_mean_diff() {
    // diffs: 0.1, 0.2, 0.3 => mean = 0.2
    let a = [1.1_f32, 2.2, 3.3];
    let b = [1.0_f32, 2.0, 3.0];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1.0 })
        .expect("should succeed");
    assert!(
        (result.mean_diff - 0.2).abs() < 1e-5,
        "mean_diff should be ~0.2, got {}",
        result.mean_diff,
    );
}

// ===========================================================================
// 38. Statistical comparison: max_diff picks the largest
// ===========================================================================

#[test]
fn test_statistical_max_diff() {
    let a = [0.0_f32; 5];
    let b = [0.01_f32, 0.05, 0.1, 0.03, 0.07];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1.0 })
        .expect("should succeed");
    assert!(
        (result.max_diff - 0.1).abs() < 1e-6,
        "max_diff should be 0.1, got {}",
        result.max_diff,
    );
}

// ===========================================================================
// 39. Statistical comparison: RMS diff in compare_tensors
// ===========================================================================

#[test]
fn test_rms_diff_computed() {
    // diffs: 0.1, 0.2, 0.3 => sum_sq = 0.01+0.04+0.09 = 0.14
    // rms = sqrt(0.14/3) ~ 0.2160
    let ref_t = tensor_1d("rms", vec![1.0, 2.0, 3.0]);
    let cand_t = tensor_1d("rms", vec![1.1, 2.2, 3.3]);
    let config = ComparisonConfig::relaxed();
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    let expected_rms: f32 = (0.14_f64 / 3.0).sqrt() as f32;
    assert!(
        (result.rms_diff - expected_rms).abs() < 1e-5,
        "rms_diff should be ~{expected_rms:.6}, got {:.6}",
        result.rms_diff,
    );
}

// ===========================================================================
// 40. Reference data versioning: preset lookup by name
// ===========================================================================

#[test]
fn test_preset_lookup_by_name() {
    assert_eq!(
        TolerancePreset::by_name("transformer"),
        Some(TolerancePreset::TRANSFORMER),
    );
    assert_eq!(
        TolerancePreset::by_name("STANDARD"),
        Some(TolerancePreset::STANDARD),
    );
    assert!(TolerancePreset::by_name("nonexistent").is_none());
}

// ===========================================================================
// 41. Reference data versioning: all presets convert to valid configs
// ===========================================================================

#[test]
fn test_all_presets_produce_valid_configs() {
    for preset in TolerancePreset::ALL {
        let config = preset.to_config();
        assert!(
            config.abs_tolerance > 0.0,
            "preset '{}' has zero abs",
            preset.name
        );
        assert!(
            config.rel_tolerance > 0.0,
            "preset '{}' has zero rel",
            preset.name
        );
        assert!(
            config.cosine_threshold > 0.0,
            "preset '{}' has zero cos",
            preset.name
        );
        assert!(
            config.cosine_threshold <= 1.0,
            "preset '{}' cos > 1.0",
            preset.name
        );
    }
}

// ===========================================================================
// 42. Batch comparison: compare_traces handles 10+ layers
// ===========================================================================

#[test]
fn test_batch_comparison_many_layers() {
    let layers: Vec<(&str, Vec<f32>)> = (0..20)
        .map(|i| {
            let name: &str = Box::leak(format!("layer_{i}").into_boxed_str());
            let data = vec![i as f32; 10];
            (name, data)
        })
        .collect();
    let ref_t = build_trace(&layers);
    let cand_t = build_trace(&layers);
    let report =
        compare_traces(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert!(report.all_passed);
    assert_eq!(report.layers.len(), 20);
}

// ===========================================================================
// 43. Empty tensor: compare_tensors returns EmptyTensor error
// ===========================================================================

#[test]
fn test_empty_tensor_compare_error() {
    let ref_t = tensor("empty", vec![0], vec![]);
    let cand_t = tensor("empty", vec![0], vec![]);
    let err = compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default())
        .expect_err("empty tensors should error");
    assert!(
        matches!(err, ReftestError::EmptyTensor(_)),
        "expected EmptyTensor, got {err:?}"
    );
}

// ===========================================================================
// 44. Empty tensor: tolerance comparison returns error
// ===========================================================================

#[test]
fn test_empty_slice_tolerance_error() {
    let err = compare_with_tolerance(&[], &[], &ToleranceStrategy::Absolute { atol: 1.0 })
        .expect_err("empty slices should error");
    assert!(matches!(err, ReftestError::EmptyTensor(_)));
}

// ===========================================================================
// 45. Scalar tensor comparison: single element
// ===========================================================================

#[test]
fn test_scalar_tensor_comparison() {
    let ref_t = tensor("loss", vec![], vec![0.5]);
    let cand_t = tensor("loss", vec![], vec![0.5]);
    let result =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.num_elements, 1);
    assert_eq!(result.max_abs_diff, 0.0);
}

// ===========================================================================
// 46. Scalar tensor: slight divergence
// ===========================================================================

#[test]
fn test_scalar_tensor_slight_divergence() {
    let ref_t = tensor("loss", vec![], vec![0.5]);
    let cand_t = tensor("loss", vec![], vec![0.5001]);
    let config = ComparisonConfig::new(1e-3, 1e-3, 0.0);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(result.passed, "0.0001 diff should pass atol=1e-3");
}

// ===========================================================================
// 47. High-dimensional tensor: 5D comparison
// ===========================================================================

#[test]
fn test_high_dim_5d_comparison() {
    let shape = vec![2, 2, 2, 2, 2]; // 32 elements
    let data = vec![1.0_f32; 32];
    let ref_t = tensor("5d", shape.clone(), data.clone());
    let cand_t = tensor("5d", shape, data);
    let result =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.num_elements, 32);
}

// ===========================================================================
// 48. High-dimensional tensor: 6D comparison with divergence
// ===========================================================================

#[test]
fn test_high_dim_6d_with_divergence() {
    let shape = vec![1, 2, 2, 2, 2, 2]; // 32 elements
    let ref_data = vec![0.0_f32; 32];
    let mut cand_data = vec![0.0_f32; 32];
    cand_data[31] = 1.0; // one divergent element
    let ref_t = tensor("6d", shape.clone(), ref_data);
    let cand_t = tensor("6d", shape, cand_data);
    let config = ComparisonConfig::strict();
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(!result.passed, "1.0 diff should fail strict config");
    assert!((result.max_abs_diff - 1.0).abs() < 1e-6);
}

// ===========================================================================
// 49. Mixed precision: f32 reference vs bf16-quantized candidate
// ===========================================================================

#[test]
fn test_mixed_precision_f32_vs_bf16() {
    // Simulate bf16 quantization by round-tripping through bf16.
    let ref_data = vec![1.123_f32, 2.456, 3.789, 0.001, -5.678];
    let bf16_data: Vec<f32> = ref_data
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_f32())
        .collect();
    let ref_t = tensor_1d("mixed", ref_data);
    let cand_t = tensor_1d("mixed", bf16_data);
    // BF16 has ~7-bit mantissa, errors up to ~1% for moderate values.
    let config = ComparisonConfig::relaxed();
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(
        result.passed,
        "bf16 quantized should pass relaxed config, max_abs={}, max_rel={}",
        result.max_abs_diff, result.max_rel_diff,
    );
}

// ===========================================================================
// 50. Mixed precision: f32 reference vs f16-quantized candidate
// ===========================================================================

#[test]
fn test_mixed_precision_f32_vs_f16() {
    let ref_data = vec![0.1_f32, 0.5, 1.0, 10.0, 100.0];
    let f16_data: Vec<f32> = ref_data
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect();
    let ref_t = tensor_1d("f16mix", ref_data);
    let cand_t = tensor_1d("f16mix", f16_data);
    let config = ComparisonConfig::relaxed();
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(
        result.passed,
        "f16 quantized should pass relaxed config, max_abs={}",
        result.max_abs_diff,
    );
}

// ===========================================================================
// 51. Large tensor comparison: 10,000 elements
// ===========================================================================

#[test]
fn test_large_tensor_comparison() {
    let n = 10_000;
    let ref_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let cand_data = ref_data.clone();
    let ref_t = tensor("large", vec![n], ref_data);
    let cand_t = tensor("large", vec![n], cand_data);
    let result =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.num_elements, n);
    assert_eq!(result.max_abs_diff, 0.0);
}

// ===========================================================================
// 52. Large tensor: with sparse errors
// ===========================================================================

#[test]
fn test_large_tensor_sparse_errors() {
    let n = 10_000;
    let ref_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let mut cand_data = ref_data.clone();
    cand_data[5000] += 0.5; // single large error
    cand_data[9999] += 0.3;
    let ref_t = tensor("sparse_err", vec![n], ref_data);
    let cand_t = tensor("sparse_err", vec![n], cand_data);
    let config = ComparisonConfig::default();
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(!result.passed);
    assert!((result.max_abs_diff - 0.5).abs() < 1e-5);
}

// ===========================================================================
// 53. Error message: LayerComparison Display format
// ===========================================================================

#[test]
fn test_layer_comparison_display_format() {
    let ref_t = tensor_1d("conv1", vec![1.0, 2.0, 3.0]);
    let cand_t = tensor_1d("conv1", vec![1.1, 2.0, 3.0]);
    let result =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::relaxed()).expect("should succeed");
    let display = format!("{result}");
    assert!(
        display.contains("conv1"),
        "display should include layer name"
    );
    assert!(display.contains("[3]"), "display should include shape");
    assert!(
        display.contains("PASS") || display.contains("FAIL"),
        "display should include pass/fail: {display}"
    );
}

// ===========================================================================
// 54. Error message: DivergenceReport summary for passing trace
// ===========================================================================

#[test]
fn test_report_summary_all_passed() {
    let layers = vec![("a", vec![1.0]), ("b", vec![2.0])];
    let ref_t = build_trace(&layers);
    let cand_t = build_trace(&layers);
    let report =
        compare_traces(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    let summary = report.summary();
    assert!(
        summary.contains("All 2 layers passed"),
        "summary should indicate all passed: {summary}"
    );
}

// ===========================================================================
// 55. Error message: DivergenceReport summary with failure
// ===========================================================================

#[test]
fn test_report_summary_with_failure() {
    let ref_t = build_trace(&[("embed", vec![1.0]), ("attn", vec![3.0])]);
    let cand_t = build_trace(&[("embed", vec![1.0]), ("attn", vec![999.0])]);
    let report =
        compare_traces(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    let summary = report.summary();
    assert!(summary.contains("First failure at layer 1"));
    assert!(summary.contains("[FAIL]"));
    assert!(summary.contains("attn"));
}

// ===========================================================================
// 56. Comparison report: layer shapes recorded correctly
// ===========================================================================

#[test]
fn test_report_layer_shapes() {
    let ref_checkpoints = vec![
        tensor("v", vec![4], vec![0.0; 4]),
        tensor("m", vec![2, 3], vec![0.0; 6]),
    ];
    let cand_checkpoints = vec![
        tensor("v", vec![4], vec![0.0; 4]),
        tensor("m", vec![2, 3], vec![0.0; 6]),
    ];
    let ref_t = ReferenceTrace::from_checkpoints(ref_checkpoints);
    let cand_t = ReferenceTrace::from_checkpoints(cand_checkpoints);
    let report =
        compare_traces(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert_eq!(report.layers[0].shape, vec![4]);
    assert_eq!(report.layers[1].shape, vec![2, 3]);
}

// ===========================================================================
// 57. Config: strict vs relaxed thresholds
// ===========================================================================

#[test]
fn test_config_strict_vs_relaxed() {
    let strict = ComparisonConfig::strict();
    let relaxed = ComparisonConfig::relaxed();
    assert!(
        strict.abs_tolerance < relaxed.abs_tolerance,
        "strict abs should be tighter than relaxed"
    );
    assert!(
        strict.rel_tolerance < relaxed.rel_tolerance,
        "strict rel should be tighter than relaxed"
    );
    assert!(
        strict.cosine_threshold > relaxed.cosine_threshold,
        "strict cos should require higher similarity"
    );
}

// ===========================================================================
// 58. Config: RMS gate causes failure
// ===========================================================================

#[test]
fn test_rms_gate_causes_failure() {
    let ref_t = tensor_1d("rms_gate", vec![0.0; 4]);
    let cand_t = tensor_1d("rms_gate", vec![0.1; 4]);
    // RMS = 0.1, set RMS gate to 0.05
    let config = ComparisonConfig::relaxed().with_rms_tolerance(0.05);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(!result.passed, "RMS=0.1 should fail rms_tolerance=0.05");
}

// ===========================================================================
// 59. Config: peak amplitude gate causes failure
// ===========================================================================

#[test]
fn test_peak_amplitude_gate_causes_failure() {
    let ref_t = tensor_1d("peak", vec![0.0; 3]);
    let cand_t = tensor_1d("peak", vec![0.0, 0.0, 200.0]);
    let config = ComparisonConfig::relaxed().with_peak_amplitude_limit(100.0);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(
        !result.passed,
        "peak=200 should fail peak_amplitude_limit=100"
    );
    assert!((result.peak_amplitude - 200.0).abs() < 1e-4);
}

// ===========================================================================
// 60. Tolerance: data length mismatch returns error
// ===========================================================================

#[test]
fn test_tolerance_data_length_mismatch() {
    let err = compare_with_tolerance(
        &[1.0, 2.0],
        &[1.0],
        &ToleranceStrategy::Absolute { atol: 1.0 },
    )
    .expect_err("different lengths should error");
    assert!(matches!(err, ReftestError::DataLengthMismatch { .. }));
}

// ===========================================================================
// 61. Cosine similarity: orthogonal vectors
// ===========================================================================

#[test]
fn test_cosine_orthogonal_vectors() {
    let ref_t = tensor_1d("orth", vec![1.0, 0.0]);
    let cand_t = tensor_1d("orth", vec![0.0, 1.0]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.5);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(
        result.cosine_similarity.abs() < 1e-5,
        "orthogonal vectors should have ~0 cosine, got {}",
        result.cosine_similarity,
    );
    assert!(!result.passed, "orthogonal vectors should fail cos=0.5");
}

// ===========================================================================
// 62. Cosine similarity: parallel vectors (different magnitude)
// ===========================================================================

#[test]
fn test_cosine_parallel_vectors() {
    let ref_t = tensor_1d("par", vec![1.0, 2.0, 3.0]);
    let cand_t = tensor_1d("par", vec![2.0, 4.0, 6.0]); // 2x scale
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.999);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(
        (result.cosine_similarity - 1.0).abs() < 1e-5,
        "parallel vectors should have cosine ~1.0, got {}",
        result.cosine_similarity,
    );
}

// ===========================================================================
// 63. Cosine: zero ref vs nonzero cand
// ===========================================================================

#[test]
fn test_cosine_zero_vs_nonzero() {
    let ref_t = tensor_1d("zv", vec![0.0, 0.0]);
    let cand_t = tensor_1d("zv", vec![1.0, 2.0]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.0);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert_eq!(result.cosine_similarity, 0.0, "zero vs nonzero = 0 cosine");
}

// ===========================================================================
// 64. NPY: write_npy_to_bytes with shape mismatch
// ===========================================================================

#[test]
fn test_npy_write_shape_mismatch() {
    let err = write_npy_to_bytes(&[1.0, 2.0], &[3]).expect_err("data/shape mismatch should fail");
    assert!(
        format!("{err}").contains("data length mismatch"),
        "error should mention data length mismatch: {err}"
    );
}

// ===========================================================================
// 65. Trace: get_by_name returns first match for duplicates
// ===========================================================================

#[test]
fn test_trace_get_by_name_first_match() {
    let checkpoints = vec![tensor_1d("dup", vec![1.0]), tensor_1d("dup", vec![2.0])];
    let trace = ReferenceTrace::from_checkpoints(checkpoints);
    let found = trace.get_by_name("dup").expect("should find");
    assert_eq!(found.data, vec![1.0], "should return first match");
}

// ===========================================================================
// 66. Trace: into_checkpoints and from_checkpoints roundtrip
// ===========================================================================

#[test]
fn test_trace_checkpoint_roundtrip() {
    let mut trace = ReferenceTrace::new();
    trace.checkpoint("a", &[1.0, 2.0], &[2]).expect("valid");
    trace
        .checkpoint("b", &[3.0, 4.0, 5.0], &[3])
        .expect("valid");
    let checkpoints = trace.into_checkpoints();
    assert_eq!(checkpoints.len(), 2);
    let rebuilt = ReferenceTrace::from_checkpoints(checkpoints);
    assert_eq!(rebuilt.len(), 2);
    assert_eq!(rebuilt.get(0).expect("exists").name, "a");
}

// ===========================================================================
// 67. Trace: capture closure returns both trace and output
// ===========================================================================

#[test]
fn test_trace_capture_closure() {
    let (trace, output) = ReferenceTrace::capture(|t| {
        t.checkpoint("h", &[1.0], &[1]).expect("valid");
        42u64
    });
    assert_eq!(output, 42);
    assert_eq!(trace.len(), 1);
}

// ===========================================================================
// 68. Preset: TolerancePreset::QUANTIZED is more relaxed than STANDARD
// ===========================================================================

#[test]
fn test_preset_quantized_more_relaxed() {
    let quantized = TolerancePreset::QUANTIZED.to_config();
    let standard = TolerancePreset::STANDARD.to_config();
    assert!(quantized.abs_tolerance > standard.abs_tolerance);
    assert!(quantized.rel_tolerance > standard.rel_tolerance);
    assert!(quantized.cosine_threshold < standard.cosine_threshold);
}

// ===========================================================================
// 69. assert_traces_match macro: passing case (compile-time check)
// ===========================================================================

#[test]
fn test_assert_traces_match_macro_passes() {
    let mut ref_t = ReferenceTrace::new();
    ref_t.checkpoint("x", &[1.0, 2.0], &[2]).expect("valid");
    let mut cand_t = ReferenceTrace::new();
    cand_t.checkpoint("x", &[1.0, 2.0], &[2]).expect("valid");
    crate::assert_traces_match!(cand_t, ref_t);
}

// ===========================================================================
// 70. assert_traces_match macro with custom tolerances
// ===========================================================================

#[test]
fn test_assert_traces_match_macro_custom_tol() {
    let mut ref_t = ReferenceTrace::new();
    ref_t.checkpoint("x", &[1.0], &[1]).expect("valid");
    let mut cand_t = ReferenceTrace::new();
    cand_t.checkpoint("x", &[1.01], &[1]).expect("valid");
    crate::assert_traces_match!(cand_t, ref_t, abs = 0.1, rel = 0.1);
}

// ===========================================================================
// 71. Tolerance Inf in reference: absolute strategy
// ===========================================================================

#[test]
fn test_tolerance_inf_in_reference() {
    let result = compare_with_tolerance(
        &[1.0],
        &[f32::INFINITY],
        &ToleranceStrategy::Absolute { atol: f64::MAX },
    )
    .expect("should succeed");
    assert!(!result.passed, "Inf in expected should fail");
    assert!(result.max_diff.is_infinite());
}

// ===========================================================================
// 72. NPY: 3D tensor roundtrip
// ===========================================================================

#[test]
fn test_npy_3d_roundtrip() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let shape = vec![2, 3, 4];
    let bytes = write_npy_to_bytes(&data, &shape).expect("write should succeed");
    let t = read_npy_from_bytes(&bytes).expect("read should succeed");
    assert_eq!(t.shape, shape);
    assert_eq!(t.data, data);
}

// ===========================================================================
// 73. RMS gate passes when within tolerance
// ===========================================================================

#[test]
fn test_rms_gate_passes() {
    // Use values within relaxed abs/rel tolerance so that the RMS gate is the
    // only thing that could cause failure. Relaxed abs=1e-2, rel=1e-1.
    let ref_t = tensor_1d("rms_ok", vec![10.0; 4]);
    let cand_t = tensor_1d("rms_ok", vec![10.005; 4]); // diff=0.005 < abs=0.01
    let config = ComparisonConfig::relaxed().with_rms_tolerance(0.1);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(result.passed, "small diffs with RMS gate=0.1 should pass");
}

// ===========================================================================
// 74. Peak amplitude gate passes when within limit
// ===========================================================================

#[test]
fn test_peak_amplitude_gate_passes() {
    // Keep values close so abs/rel pass under relaxed config (abs=1e-2, rel=1e-1).
    let ref_t = tensor_1d("pk", vec![10.0, 50.0]);
    let cand_t = tensor_1d("pk", vec![10.0, 50.0]);
    let config = ComparisonConfig::relaxed().with_peak_amplitude_limit(100.0);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(result.passed, "peak=50 should pass peak_limit=100");
    assert!((result.peak_amplitude - 50.0).abs() < 1e-4);
}

// ===========================================================================
// 75. Safetensors: multi-dim tensor loaded with correct shape
// ===========================================================================

#[test]
fn test_safetensors_multidim_shape() {
    let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
    let bytes = build_safetensors(&[("w", &[3, 4, 5], &data)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("should load");
    let t = trace.get(0).expect("exists");
    assert_eq!(t.shape, vec![3, 4, 5]);
    assert_eq!(t.numel(), 60);
}

// ===========================================================================
// 76. PercentClose: 100% requirement means all must pass
// ===========================================================================

#[test]
fn test_percent_close_100_percent() {
    let a = [1.0_f32, 2.0, 3.0, 100.0];
    let b = [1.0_f32, 2.0, 3.0, 4.0];
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::PercentClose {
            threshold: 0.5,
            percent: 100.0,
        },
    )
    .expect("should succeed");
    assert!(
        !result.passed,
        "100% requirement should fail with 1 outlier"
    );
}

// ===========================================================================
// 77. PercentClose: 0% requirement always passes
// ===========================================================================

#[test]
fn test_percent_close_0_percent() {
    let a = [100.0_f32, 200.0, 300.0];
    let b = [1.0_f32, 2.0, 3.0];
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::PercentClose {
            threshold: 0.001,
            percent: 0.0,
        },
    )
    .expect("should succeed");
    assert!(result.passed, "0% requirement should always pass");
}

// ===========================================================================
// 78. Relative error: near-zero values skip relative check
// ===========================================================================

#[test]
fn test_relative_near_zero_skip() {
    let ref_t = tensor_1d("tiny", vec![1e-8, 2e-8]);
    let cand_t = tensor_1d("tiny", vec![2e-8, 3e-8]);
    // With rel_tolerance=0, near-zero values should still pass because
    // both are below abs_tolerance so relative check is skipped.
    let config = ComparisonConfig::new(1e-5, 0.0, 0.0);
    let result = compare_tensors(&ref_t, &cand_t, &config).expect("should succeed");
    assert!(
        result.passed,
        "near-zero values below abs_tolerance should skip relative check"
    );
}

// ===========================================================================
// 79. Named tensor: shape with zero dimension
// ===========================================================================

#[test]
fn test_named_tensor_zero_dim() {
    let t = NamedTensor::new("zero", vec![3, 0, 5], vec![]).expect("valid");
    assert_eq!(t.numel(), 0);
    assert_eq!(t.shape, vec![3, 0, 5]);
}

// ===========================================================================
// 80. Named tensor: shape product overflow
// ===========================================================================

#[test]
fn test_named_tensor_overflow() {
    let err = NamedTensor::new("overflow", vec![usize::MAX, 2], vec![])
        .expect_err("overflow should fail");
    assert!(matches!(err, ReftestError::ShapeProductOverflow(_)));
}

// ===========================================================================
// 81. Trace comparison: all layers fail
// ===========================================================================

#[test]
fn test_trace_all_layers_fail() {
    let ref_t = build_trace(&[("a", vec![0.0]), ("b", vec![0.0])]);
    let cand_t = build_trace(&[("a", vec![999.0]), ("b", vec![888.0])]);
    let report =
        compare_traces(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert!(!report.all_passed);
    assert_eq!(report.first_failure, Some(0));
    assert!(!report.layers[0].passed);
    assert!(!report.layers[1].passed);
}

// ===========================================================================
// 82. Config: default values match documented constants
// ===========================================================================

#[test]
fn test_config_default_values() {
    let config = ComparisonConfig::default();
    assert_eq!(config.abs_tolerance, 1e-5);
    assert_eq!(config.rel_tolerance, 1e-4);
    assert_eq!(config.cosine_threshold, 0.9999);
    assert!(config.rms_tolerance.is_none());
    assert!(config.peak_amplitude_limit.is_none());
}

// ===========================================================================
// 83. Preset: TTS preset values match documentation
// ===========================================================================

#[test]
fn test_preset_tts_values() {
    let tts = TolerancePreset::TTS;
    assert_eq!(tts.name, "tts");
    assert!((tts.abs_threshold - 5e-3).abs() < 1e-10);
    assert!((tts.rel_threshold - 1e-2).abs() < 1e-10);
    assert!((tts.cos_threshold - 0.995).abs() < 1e-10);
}

// ===========================================================================
// 84. Preset: AUDIO preset values
// ===========================================================================

#[test]
fn test_preset_audio_values() {
    let audio = TolerancePreset::AUDIO;
    assert_eq!(audio.name, "audio");
    assert!((audio.abs_threshold - 1e-3).abs() < 1e-10);
    assert!((audio.cos_threshold - 0.99).abs() < 1e-10);
}

// ===========================================================================
// 85. NpyDType: from_descr and to_descr roundtrip
// ===========================================================================

#[test]
fn test_npy_dtype_roundtrip() {
    let cases = [
        ("<f2", NpyDType::F16),
        ("<f4", NpyDType::F32),
        ("<f8", NpyDType::F64),
        ("<i4", NpyDType::I32),
        ("<i8", NpyDType::I64),
        ("|u1", NpyDType::U8),
    ];
    for (descr, expected) in cases {
        let parsed = NpyDType::from_descr(descr).unwrap_or_else(|| panic!("should parse {descr}"));
        assert_eq!(parsed, expected, "mismatch for {descr}");
    }
}

// ===========================================================================
// 86. NpyDType: unknown descriptor returns None
// ===========================================================================

#[test]
fn test_npy_dtype_unknown() {
    assert!(
        NpyDType::from_descr("<c8").is_none(),
        "complex should be None"
    );
    assert!(NpyDType::from_descr("").is_none(), "empty should be None");
}

// ===========================================================================
// 87. NpyDType: Display impl
// ===========================================================================

#[test]
fn test_npy_dtype_display() {
    assert_eq!(format!("{}", NpyDType::F32), "<f4");
    assert_eq!(format!("{}", NpyDType::F16), "<f2");
    assert_eq!(format!("{}", NpyDType::U8), "|u1");
}

// ===========================================================================
// 88. Trace: empty trace comparison (0 layers each)
// ===========================================================================

#[test]
fn test_trace_empty_comparison() {
    let ref_t = ReferenceTrace::new();
    let cand_t = ReferenceTrace::new();
    let report =
        compare_traces(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert!(report.all_passed);
    assert!(report.layers.is_empty());
    assert!(report.first_failure.is_none());
}

// ===========================================================================
// 89. Tolerance: single element with Inf diff
// ===========================================================================

#[test]
fn test_tolerance_inf_diff() {
    let result = compare_with_tolerance(
        &[f32::INFINITY],
        &[0.0],
        &ToleranceStrategy::Absolute { atol: f64::MAX },
    )
    .expect("should succeed");
    assert!(!result.passed, "Inf in actual should fail");
    assert_eq!(result.num_mismatches, 1);
}

// ===========================================================================
// 90. Layer comparison: num_elements field
// ===========================================================================

#[test]
fn test_layer_num_elements() {
    let ref_t = tensor("ne", vec![5, 4], vec![0.0; 20]);
    let cand_t = tensor("ne", vec![5, 4], vec![0.0; 20]);
    let result =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert_eq!(result.num_elements, 20);
}

// ===========================================================================
// 91. Layer comparison: name preserved
// ===========================================================================

#[test]
fn test_layer_name_preserved() {
    let ref_t = tensor_1d("encoder.block.3.norm", vec![1.0]);
    let cand_t = tensor_1d("encoder.block.3.norm", vec![1.0]);
    let result =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert_eq!(result.name, "encoder.block.3.norm");
}

// ===========================================================================
// 92. Divergence report: summary contains all layer names
// ===========================================================================

#[test]
fn test_report_summary_all_names() {
    let ref_t = build_trace(&[
        ("alpha", vec![1.0]),
        ("beta", vec![2.0]),
        ("gamma", vec![3.0]),
    ]);
    let cand_t = build_trace(&[
        ("alpha", vec![1.0]),
        ("beta", vec![2.0]),
        ("gamma", vec![3.0]),
    ]);
    let report =
        compare_traces(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    let summary = report.summary();
    assert!(summary.contains("alpha"));
    assert!(summary.contains("beta"));
    assert!(summary.contains("gamma"));
}
