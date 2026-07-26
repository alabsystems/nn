// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended comparison and tolerance tests for nn-reftest.
//!
//! Covers 12 test categories:
//! 1. Tolerance modes: absolute, relative, ULP, mixed, percent-close
//! 2. Tensor comparison: element-wise with known diffs, monotonic, scaled
//! 3. Shape mismatch detection: different ranks, transposed dims, propagation
//! 4. Dtype mismatch handling: F32 vs BF16 safetensors loading and comparison
//! 5. NaN handling: NaN==NaN for testing, NaN-immune cosine, all-NaN tensors
//! 6. Inf handling: positive/negative infinity, mixed inf/finite, symmetric inf
//! 7. Large tensor comparison: 100K+ elements, performance sanity
//! 8. Safetensors loading: shape/dtype verification, multi-tensor, sorted order
//! 9. NPY loading: f32/f16/f64/i32/i64/u8, big-endian, scalar, error paths
//! 10. Trace comparison: multi-layer, alignment, length mismatch, capture API
//! 11. Summary statistics: max/mean abs diff, percentage within tolerance
//! 12. Report generation: human-readable summaries, Display impl, pass/fail

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

fn tensor_1d(name: &str, data: Vec<f32>) -> NamedTensor {
    let len = data.len();
    NamedTensor::new(name, vec![len], data).expect("valid 1-D test tensor")
}

fn tensor_nd(name: &str, shape: Vec<usize>, data: Vec<f32>) -> NamedTensor {
    NamedTensor::new(name, shape, data).expect("valid N-D test tensor")
}

fn f32_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Build a minimal safetensors byte buffer from typed tensors.
fn build_safetensors_typed(tensors: &[(&str, &[usize], safetensors::Dtype, &[u8])]) -> Vec<u8> {
    let mut tensor_map: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
    for &(name, shape, dtype, data) in tensors {
        let view = safetensors::tensor::TensorView::new(dtype, shape.to_vec(), data)
            .expect("valid tensor view");
        tensor_map.push((name.to_string(), view));
    }
    safetensors::tensor::serialize(tensor_map, None).expect("serialization should succeed")
}

/// Build a minimal safetensors byte buffer from f32 tensors.
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

/// Build an NPY v1.0 byte buffer for a given dtype string and raw data bytes.
fn build_npy_v1(dtype: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let shape_str = if shape.is_empty() {
        "()".to_string()
    } else if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
        format!("({})", dims.join(", "))
    };

    let header = format!(
        "{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape_str}, }}",
    );

    let prefix_len = 10;
    let total_header = header.len() + 1;
    let padded_len = (prefix_len + total_header).div_ceil(64) * 64 - prefix_len;
    let padding = padded_len - header.len() - 1;

    let mut buf = Vec::new();
    buf.extend_from_slice(b"\x93NUMPY");
    buf.push(1);
    buf.push(0);
    let header_len = padded_len as u16;
    buf.extend_from_slice(&header_len.to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    buf.extend(std::iter::repeat_n(b' ', padding));
    buf.push(b'\n');
    buf.extend_from_slice(data);
    buf
}

// ===========================================================================
// 1. Tolerance modes: absolute, relative, ULP, mixed, percent-close
// ===========================================================================

#[test]
fn test_absolute_tolerance_exact_boundary_pass_and_fail() {
    // Use a power-of-two delta (2^-13) so the f32 difference is exactly
    // representable and equals the f64 atol bit-for-bit. A decimal literal
    // like 0.001f32 widens to 0.0010000000475..., which would spuriously
    // exceed the f64 atol 0.001 and break the `diff == atol` boundary.
    let delta = 2.0f32.powi(-13); // 0.0001220703125, exact in f32 and f64
    let a = [0.0f32];
    let b = [delta];
    let pass = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::Absolute {
            atol: f64::from(delta),
        },
    )
    .expect("should succeed");
    assert!(pass.passed, "diff == atol should pass");

    let fail = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::Absolute {
            atol: f64::from(delta) * 0.5,
        },
    )
    .expect("should succeed");
    assert!(!fail.passed, "diff > atol should fail");
}

#[test]
fn test_absolute_tolerance_negative_values() {
    // Verify absolute tolerance works with negative numbers.
    let a = [-5.0f32, -10.0, -0.001];
    let b = [-5.001f32, -10.002, -0.002];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 0.01 })
        .expect("should succeed");
    assert!(result.passed, "negative values within atol should pass");
}

#[test]
fn test_absolute_tolerance_zero_atol_requires_exact_match() {
    let a = [1.0f32];
    let b = [1.0f32];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 0.0 })
        .expect("should succeed");
    assert!(
        result.passed,
        "identical values should pass even with atol=0"
    );

    let b_diff = [1.0f32 + f32::EPSILON];
    let result_fail =
        compare_with_tolerance(&a, &b_diff, &ToleranceStrategy::Absolute { atol: 0.0 })
            .expect("should succeed");
    assert!(
        !result_fail.passed,
        "any difference should fail with atol=0"
    );
}

#[test]
fn test_relative_tolerance_near_zero_uses_epsilon_floor() {
    // Relative diff = |a-b| / max(|a|, |b|, 1e-8).
    let a = [1e-10f32];
    let b = [1.95e-10f32];
    // diff = 0.95e-10, denom = 1e-8 (epsilon floor dominates the tiny inputs),
    // so rel = 0.0095. This sits strictly between 0.009 and 0.01, avoiding the
    // exact-boundary case where f32->f64 widening of 1e-10 would push rel just
    // over a 0.01 threshold.
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Relative { rtol: 0.01 })
        .expect("should succeed");
    assert!(
        result.passed,
        "near-zero relative with epsilon floor should pass at rtol=0.01"
    );

    let result_tight = compare_with_tolerance(&a, &b, &ToleranceStrategy::Relative { rtol: 0.009 })
        .expect("should succeed");
    assert!(
        !result_tight.passed,
        "near-zero relative should fail at rtol=0.009"
    );
}

#[test]
fn test_relative_tolerance_large_values() {
    // For large values, relative tolerance should scale with magnitude.
    // a=1000.0, b=1001.0 => diff=1.0, rel=1.0/1001.0 ~= 0.000999
    let a = [1000.0f32];
    let b = [1001.0f32];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Relative { rtol: 0.001 })
        .expect("should succeed");
    assert!(result.passed, "1 part per 1000 should pass rtol=0.001");

    let result_tight =
        compare_with_tolerance(&a, &b, &ToleranceStrategy::Relative { rtol: 0.0009 })
            .expect("should succeed");
    assert!(
        !result_tight.passed,
        "1 part per 1000 should fail rtol=0.0009"
    );
}

#[test]
fn test_ulp_distance_adjacent_floats() {
    // Adjacent f32 values differ by exactly 1 ULP.
    let a = 1.0f32;
    let b = f32::from_bits(a.to_bits() + 1);
    let result = compare_with_tolerance(&[a], &[b], &ToleranceStrategy::ULP { max_ulps: 1 })
        .expect("should succeed");
    assert!(result.passed, "adjacent floats should be 1 ULP apart");

    let result_zero = compare_with_tolerance(&[a], &[b], &ToleranceStrategy::ULP { max_ulps: 0 })
        .expect("should succeed");
    assert!(
        !result_zero.passed,
        "adjacent floats should fail with max_ulps=0"
    );
}

#[test]
fn test_ulp_distance_symmetry_across_sign() {
    let pos = 1e-20f32;
    let neg = -1e-20f32;
    let result_pos_neg = compare_with_tolerance(
        &[pos],
        &[neg],
        &ToleranceStrategy::ULP { max_ulps: u32::MAX },
    )
    .expect("should succeed");
    let result_neg_pos = compare_with_tolerance(
        &[neg],
        &[pos],
        &ToleranceStrategy::ULP { max_ulps: u32::MAX },
    )
    .expect("should succeed");
    assert_eq!(result_pos_neg.num_mismatches, result_neg_pos.num_mismatches);
}

#[test]
fn test_ulp_comparison_at_max_positive_float() {
    let a = f32::MAX;
    let b = f32::from_bits(f32::MAX.to_bits() - 1);
    let result = compare_with_tolerance(&[a], &[b], &ToleranceStrategy::ULP { max_ulps: 1 })
        .expect("should succeed");
    assert!(
        result.passed,
        "adjacent floats at f32::MAX should be 1 ULP apart"
    );
}

#[test]
fn test_ulp_nan_always_mismatches() {
    let result = compare_with_tolerance(
        &[f32::NAN],
        &[f32::NAN],
        &ToleranceStrategy::ULP { max_ulps: u32::MAX },
    )
    .expect("should succeed");
    assert_eq!(
        result.num_mismatches, 1,
        "NaN vs NaN should be a ULP mismatch"
    );
}

#[test]
fn test_mixed_tolerance_numpy_semantics() {
    // NumPy: |a - b| <= atol + rtol * |b|
    let result = compare_with_tolerance(
        &[1.005f32],
        &[1.0f32],
        &ToleranceStrategy::Mixed {
            atol: 0.001,
            rtol: 0.01,
        },
    )
    .expect("should succeed");
    assert!(result.passed, "diff=0.005 < threshold=0.011 should pass");

    let result_fail = compare_with_tolerance(
        &[1.02f32],
        &[1.0f32],
        &ToleranceStrategy::Mixed {
            atol: 0.001,
            rtol: 0.01,
        },
    )
    .expect("should succeed");
    assert!(
        !result_fail.passed,
        "diff=0.02 > threshold=0.011 should fail"
    );
}

#[test]
fn test_mixed_tolerance_dominated_by_atol_for_small_values() {
    // For small expected values, atol dominates: threshold ~= atol.
    let a = [0.001f32];
    let b = [0.0f32]; // |b| = 0
                      // threshold = atol + rtol*0 = atol = 0.01
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::Mixed {
            atol: 0.01,
            rtol: 1e-5,
        },
    )
    .expect("should succeed");
    assert!(result.passed, "small diff dominated by atol should pass");
}

#[test]
fn test_percent_close_with_zero_percent_threshold() {
    let result = compare_with_tolerance(
        &[1.0f32, 100.0],
        &[2.0f32, 200.0],
        &ToleranceStrategy::PercentClose {
            threshold: 0.0,
            percent: 0.0,
        },
    )
    .expect("should succeed");
    assert!(result.passed, "0% requirement means all can be outliers");
}

#[test]
fn test_percent_close_with_50_percent() {
    // 4 elements: 2 within threshold, 2 outside.
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [1.0f32, 2.0, 300.0, 400.0]; // elements 2,3 are outliers
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::PercentClose {
            threshold: 0.1,
            percent: 50.0,
        },
    )
    .expect("should succeed");
    assert!(
        result.passed,
        "50% within threshold should pass when exactly 50% match"
    );

    let result_fail = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::PercentClose {
            threshold: 0.1,
            percent: 51.0,
        },
    )
    .expect("should succeed");
    assert!(
        !result_fail.passed,
        "51% required but only 50% match should fail"
    );
}

#[test]
fn test_percent_close_100_percent_is_equivalent_to_absolute() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [1.001f32, 2.001, 3.001];
    let pct = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::PercentClose {
            threshold: 0.01,
            percent: 100.0,
        },
    )
    .expect("should succeed");
    let abs = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 0.01 })
        .expect("should succeed");
    assert_eq!(
        pct.passed, abs.passed,
        "100% PercentClose should match Absolute"
    );
}

// ===========================================================================
// 2. Tensor comparison: element-wise with known diffs
// ===========================================================================

#[test]
fn test_element_wise_comparison_identical_tensors() {
    let data: Vec<f32> = (0..50).map(|i| i as f32 * 0.1).collect();
    let a = tensor_1d("x", data.clone());
    let b = tensor_1d("x", data);
    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.max_abs_diff, 0.0);
    assert_eq!(result.mean_abs_diff, 0.0);
    assert_eq!(result.cosine_similarity, 1.0);
}

#[test]
fn test_element_wise_comparison_known_uniform_perturbation() {
    // All elements differ by exactly 0.01.
    let n = 100;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = a.iter().map(|&x| x + 0.01).collect();
    let ta = tensor_1d("x", a);
    let tb = tensor_1d("x", b);
    let result =
        compare_tensors(&ta, &tb, &ComparisonConfig::new(0.02, 1.0, 0.0)).expect("should succeed");
    assert!(result.passed);
    assert!(
        (result.max_abs_diff - 0.01).abs() < 1e-5,
        "max_abs should be ~0.01"
    );
    assert!(
        (result.mean_abs_diff - 0.01).abs() < 1e-5,
        "mean_abs should be ~0.01"
    );
}

#[test]
fn test_element_wise_comparison_monotonic_sequence() {
    // Test with monotonically increasing values to verify no ordering issues.
    let a: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..1000).map(|i| i as f32 + 1e-6).collect();
    let ta = tensor_1d("mono", a);
    let tb = tensor_1d("mono", b);
    let result =
        compare_tensors(&ta, &tb, &ComparisonConfig::new(1e-4, 1e-2, 0.0)).expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.num_elements, 1000);
}

#[test]
fn test_element_wise_comparison_scaled_tensors() {
    // Identical direction but different magnitudes should fail abs but pass cos.
    let a = tensor_1d("x", vec![1.0, 2.0, 3.0]);
    let b = tensor_1d("x", vec![2.0, 4.0, 6.0]);
    let config = ComparisonConfig::new(100.0, 100.0, 0.999);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(result.passed, "same direction should pass cosine threshold");
    assert!(
        (result.cosine_similarity - 1.0).abs() < 1e-5,
        "parallel vectors should have cos=1.0"
    );
}

#[test]
fn test_element_wise_comparison_2d_tensor() {
    let a = tensor_nd("w", vec![3, 4], vec![1.0; 12]);
    let b = tensor_nd("w", vec![3, 4], vec![1.0001; 12]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::new(1e-3, 1e-2, 0.0)).expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.shape, vec![3, 4]);
}

#[test]
fn test_element_wise_comparison_single_element_divergence() {
    // All identical except one large outlier.
    let mut data_b = vec![1.0f32; 100];
    data_b[50] = 999.0;
    let a = tensor_1d("x", vec![1.0; 100]);
    let b = tensor_1d("x", data_b);
    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");
    assert!(!result.passed, "single large outlier should fail");
    assert!(
        result.max_abs_diff > 900.0,
        "max_abs should reflect the outlier"
    );
}

// ===========================================================================
// 3. Shape mismatch detection
// ===========================================================================

#[test]
fn test_shape_mismatch_reports_tensor_name() {
    let a = tensor_nd("encoder.layer3.weight", vec![4, 3], vec![0.0; 12]);
    let b = tensor_nd("encoder.layer3.weight", vec![3, 4], vec![0.0; 12]);
    let err = compare_tensors(&a, &b, &ComparisonConfig::default()).expect_err("should fail");
    match err {
        ReftestError::ShapeMismatch {
            name,
            expected,
            actual,
        } => {
            assert_eq!(name, "encoder.layer3.weight");
            assert_eq!(expected, vec![4, 3]);
            assert_eq!(actual, vec![3, 4]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn test_shape_mismatch_different_ranks() {
    let a = tensor_nd("x", vec![6], vec![0.0; 6]);
    let b = tensor_nd("x", vec![2, 3], vec![0.0; 6]);
    let err = compare_tensors(&a, &b, &ComparisonConfig::default())
        .expect_err("different rank should fail");
    assert!(matches!(err, ReftestError::ShapeMismatch { .. }));
}

#[test]
fn test_shape_mismatch_same_rank_different_dims() {
    let a = tensor_nd("x", vec![4, 5], vec![0.0; 20]);
    let b = tensor_nd("x", vec![5, 4], vec![0.0; 20]);
    let err = compare_tensors(&a, &b, &ComparisonConfig::default())
        .expect_err("transposed dims should fail");
    match err {
        ReftestError::ShapeMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, vec![4, 5]);
            assert_eq!(actual, vec![5, 4]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn test_shape_mismatch_3d_vs_2d() {
    let a = tensor_nd("x", vec![2, 3, 4], vec![0.0; 24]);
    let b = tensor_nd("x", vec![6, 4], vec![0.0; 24]);
    let err =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect_err("3D vs 2D should fail");
    assert!(matches!(err, ReftestError::ShapeMismatch { .. }));
}

#[test]
fn test_shape_mismatch_in_trace_propagates_error() {
    let ref_trace = ReferenceTrace::from_checkpoints(vec![
        tensor_nd("ok_layer", vec![2], vec![1.0, 2.0]),
        tensor_nd("bad_layer", vec![3, 2], vec![0.0; 6]),
    ]);
    let cand_trace = ReferenceTrace::from_checkpoints(vec![
        tensor_nd("ok_layer", vec![2], vec![1.0, 2.0]),
        tensor_nd("bad_layer", vec![2, 3], vec![0.0; 6]),
    ]);
    let err = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect_err("should propagate shape mismatch");
    match err {
        ReftestError::ShapeMismatch { name, .. } => {
            assert_eq!(name, "bad_layer");
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn test_element_count_mismatch_on_named_tensor_creation() {
    let result = NamedTensor::new("broken", vec![2, 3], vec![1.0; 5]);
    match result {
        Err(ReftestError::ElementCountMismatch {
            name,
            shape,
            expected,
            actual,
        }) => {
            assert_eq!(name, "broken");
            assert_eq!(shape, vec![2, 3]);
            assert_eq!(expected, 6);
            assert_eq!(actual, 5);
        }
        other => panic!("expected ElementCountMismatch, got {other:?}"),
    }
}

#[test]
fn test_shape_product_overflow_detected() {
    let result = NamedTensor::new("overflow", vec![usize::MAX, 2], vec![]);
    assert!(matches!(result, Err(ReftestError::ShapeProductOverflow(_))));
}

// ===========================================================================
// 4. Dtype mismatch handling: comparing F32 vs BF16 tensors via safetensors
// ===========================================================================

#[test]
fn test_safetensors_bf16_loaded_as_f32_for_comparison() {
    // Create a BF16 safetensors buffer and verify it loads to f32 data.
    let bf16_values: Vec<half::bf16> = vec![
        half::bf16::from_f32(1.0),
        half::bf16::from_f32(2.0),
        half::bf16::from_f32(-0.5),
    ];
    let raw: Vec<u8> = bf16_values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let bytes = build_safetensors_typed(&[("weight", &[3], safetensors::Dtype::BF16, &raw)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    assert_eq!(trace.len(), 1);
    let t = trace.get(0).expect("exists");
    assert_eq!(t.shape, vec![3]);
    // BF16 has limited precision: 1.0, 2.0 are exact; -0.5 is exact.
    assert!((t.data[0] - 1.0).abs() < 0.01);
    assert!((t.data[1] - 2.0).abs() < 0.01);
    assert!((t.data[2] - (-0.5)).abs() < 0.01);
}

#[test]
fn test_safetensors_f16_loaded_as_f32_for_comparison() {
    let f16_values: Vec<half::f16> = vec![
        half::f16::from_f32(0.0),
        half::f16::from_f32(1.5),
        half::f16::from_f32(-3.0),
    ];
    let raw: Vec<u8> = f16_values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let bytes = build_safetensors_typed(&[("bias", &[3], safetensors::Dtype::F16, &raw)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    let t = trace.get(0).expect("exists");
    assert!((t.data[0] - 0.0).abs() < 1e-3);
    assert!((t.data[1] - 1.5).abs() < 1e-3);
    assert!((t.data[2] - (-3.0)).abs() < 1e-3);
}

#[test]
fn test_compare_f32_trace_against_bf16_loaded_trace() {
    // Simulate comparing a Rust f32 model output against a BF16 reference.
    let bf16_values: Vec<half::bf16> = vec![
        half::bf16::from_f32(1.0),
        half::bf16::from_f32(2.0),
        half::bf16::from_f32(3.0),
    ];
    let raw: Vec<u8> = bf16_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes = build_safetensors_typed(&[("layer", &[3], safetensors::Dtype::BF16, &raw)]);
    let ref_trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    // Candidate trace with exact f32 values.
    let mut cand_trace = ReferenceTrace::new();
    cand_trace
        .checkpoint("layer", &[1.0, 2.0, 3.0], &[3])
        .expect("valid");

    // Use relaxed config since BF16 has ~0.4% relative error for small values.
    let config = ComparisonConfig::relaxed();
    let report =
        compare_traces(&ref_trace, &cand_trace, &config).expect("comparison should succeed");
    assert!(
        report.all_passed,
        "BF16 vs F32 with relaxed tolerance should pass"
    );
}

#[test]
fn test_safetensors_unsupported_dtype_rejected() {
    // BOOL dtype is not supported for f32 conversion.
    let raw = vec![0u8, 1u8];
    let bytes = build_safetensors_typed(&[("flags", &[2], safetensors::Dtype::BOOL, &raw)]);
    let result = load_safetensors_from_bytes(&bytes);
    assert!(matches!(result, Err(ReftestError::UnsupportedDtype(_))));
}

// ===========================================================================
// 5. NaN handling in comparisons
// ===========================================================================

#[test]
fn test_nan_in_single_element_tensor_fails_all_metrics() {
    let a = tensor_1d("x", vec![f32::NAN]);
    let b = tensor_1d("x", vec![1.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed structurally");
    assert!(!result.passed);
    assert!(result.max_abs_diff.is_infinite());
    assert!(result.max_rel_diff.is_infinite());
}

#[test]
fn test_nan_vs_nan_both_sides_fails() {
    // NaN is never equal to NaN in IEEE 754, and compare_tensors treats it as max divergence.
    let a = tensor_1d("x", vec![f32::NAN]);
    let b = tensor_1d("x", vec![f32::NAN]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed structurally");
    assert!(!result.passed, "NaN vs NaN should fail comparison");
}

#[test]
fn test_nan_in_tolerance_comparison_fails_all_strategies() {
    for strategy in &[
        ToleranceStrategy::Absolute { atol: f64::MAX },
        ToleranceStrategy::Relative { rtol: f64::MAX },
        ToleranceStrategy::Mixed {
            atol: f64::MAX,
            rtol: f64::MAX,
        },
        ToleranceStrategy::ULP { max_ulps: u32::MAX },
        ToleranceStrategy::PercentClose {
            threshold: f64::MAX,
            percent: 0.0,
        },
    ] {
        let result = compare_with_tolerance(&[f32::NAN], &[1.0], strategy).expect("should succeed");
        assert!(
            result.num_mismatches >= 1,
            "NaN in actual should be a mismatch for {strategy:?}"
        );
    }
}

#[test]
fn test_mixed_nan_and_valid_finite_cosine() {
    // NaN elements are excluded from cosine computation (valid elements only).
    let a = tensor_1d("x", vec![1.0, 2.0, f32::NAN, 4.0, 5.0]);
    let b = tensor_1d("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed structurally");
    assert!(!result.passed, "NaN element should cause failure");
    assert!(
        result.cosine_similarity.is_finite(),
        "cosine should be finite from valid elements"
    );
    assert!(
        (result.cosine_similarity - 1.0).abs() < 1e-6,
        "cosine from identical valid elements should be ~1.0"
    );
}

#[test]
fn test_all_nan_tensor_produces_nan_cosine() {
    let a = tensor_1d("x", vec![f32::NAN, f32::NAN]);
    let b = tensor_1d("x", vec![f32::NAN, f32::NAN]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed structurally");
    assert!(!result.passed);
    // All elements are non-finite, so cosine is NaN (undefined).
    assert!(
        result.cosine_similarity.is_nan(),
        "all-NaN should yield NaN cosine"
    );
}

#[test]
fn test_nan_at_various_positions() {
    // NaN at first, middle, and last position.
    for pos in [0, 2, 4] {
        let mut data = vec![1.0f32; 5];
        data[pos] = f32::NAN;
        let a = tensor_1d("x", data);
        let b = tensor_1d("x", vec![1.0; 5]);
        let result = compare_tensors(&a, &b, &ComparisonConfig::default())
            .expect("should succeed structurally");
        assert!(!result.passed, "NaN at position {pos} should fail");
    }
}

// ===========================================================================
// 6. Inf handling: positive and negative infinity comparisons
// ===========================================================================

#[test]
fn test_inf_in_candidate_produces_infinite_peak_amplitude() {
    let a = tensor_1d("x", vec![1.0, 2.0, 3.0]);
    let b = tensor_1d("x", vec![1.0, f32::INFINITY, 3.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed structurally");
    assert!(result.peak_amplitude.is_infinite());
    assert!(!result.passed);
}

#[test]
fn test_neg_inf_in_candidate_produces_infinite_peak_amplitude() {
    let a = tensor_1d("x", vec![1.0]);
    let b = tensor_1d("x", vec![f32::NEG_INFINITY]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed structurally");
    assert!(result.peak_amplitude.is_infinite());
}

#[test]
fn test_pos_inf_vs_pos_inf_still_fails() {
    // Even matching infinities should fail (non-finite).
    let a = tensor_1d("x", vec![f32::INFINITY]);
    let b = tensor_1d("x", vec![f32::INFINITY]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed structurally");
    assert!(!result.passed, "inf vs inf should fail");
    assert!(result.max_abs_diff.is_infinite());
}

#[test]
fn test_neg_inf_vs_neg_inf_still_fails() {
    let a = tensor_1d("x", vec![f32::NEG_INFINITY]);
    let b = tensor_1d("x", vec![f32::NEG_INFINITY]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed structurally");
    assert!(!result.passed, "neg_inf vs neg_inf should fail");
}

#[test]
fn test_pos_inf_vs_neg_inf_fails() {
    let a = tensor_1d("x", vec![f32::INFINITY]);
    let b = tensor_1d("x", vec![f32::NEG_INFINITY]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed structurally");
    assert!(!result.passed);
    assert!(result.max_abs_diff.is_infinite());
}

#[test]
fn test_mixed_inf_and_finite_elements() {
    // First element is inf, rest are finite and identical.
    let a = tensor_1d("x", vec![f32::INFINITY, 1.0, 2.0, 3.0]);
    let b = tensor_1d("x", vec![100.0, 1.0, 2.0, 3.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed structurally");
    assert!(!result.passed, "inf element should cause failure");
    // Cosine should still be computed from finite elements.
    assert!(result.cosine_similarity.is_finite());
}

#[test]
fn test_all_inf_tensor_comparison() {
    let a = tensor_1d("x", vec![f32::INFINITY, f32::INFINITY]);
    let b = tensor_1d("x", vec![f32::INFINITY, f32::INFINITY]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed structurally");
    assert!(!result.passed, "all-inf tensors should fail");
    assert!(result.max_abs_diff.is_infinite());
}

#[test]
fn test_negative_zero_vs_positive_zero() {
    let a = tensor_1d("x", vec![-0.0]);
    let b = tensor_1d("x", vec![0.0]);
    let result = compare_tensors(&a, &b, &ComparisonConfig::strict()).expect("should succeed");
    assert!(result.passed, "+0.0 vs -0.0 should be equal");
    assert_eq!(result.max_abs_diff, 0.0);
}

// ===========================================================================
// 7. Large tensor comparison: performance with large tensors
// ===========================================================================

#[test]
fn test_large_tensor_100k_elements_identical() {
    let n = 100_000;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let a = tensor_1d("large", data.clone());
    let b = tensor_1d("large", data);
    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.max_abs_diff, 0.0);
    assert_eq!(result.num_elements, n);
}

#[test]
fn test_large_tensor_100k_elements_small_perturbation() {
    let n = 100_000;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = a.iter().map(|&x| x + 1e-7).collect();
    let ta = tensor_1d("large", a);
    let tb = tensor_1d("large", b);
    let result = compare_tensors(&ta, &tb, &ComparisonConfig::default()).expect("should succeed");
    assert!(
        result.passed,
        "1e-7 perturbation should pass default config"
    );
}

#[test]
fn test_large_tensor_does_not_overflow_accumulation() {
    // Large identical values should not overflow sum_abs or sum_sq_diff.
    let n = 10_000;
    let a: Vec<f32> = vec![f32::MAX / 2.0; n];
    let b: Vec<f32> = vec![f32::MAX / 2.0; n];
    let result = compare_tensors(
        &tensor_1d("big", a),
        &tensor_1d("big", b),
        &ComparisonConfig::default(),
    )
    .expect("should not overflow");
    assert!(result.passed);
    assert_eq!(result.max_abs_diff, 0.0);
}

#[test]
fn test_large_tolerance_comparison_100k() {
    let n = 100_000;
    let a: Vec<f32> = vec![0.0; n];
    let b: Vec<f32> = vec![0.001; n];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 0.01 })
        .expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.num_mismatches, 0);
}

// ===========================================================================
// 8. Safetensors loading: load tensor, verify shape/dtype
// ===========================================================================

#[test]
fn test_safetensors_f32_roundtrip_shape_and_data() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes = build_safetensors(&[("weights", &[2, 3], &data)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    assert_eq!(trace.len(), 1);
    let t = trace.get(0).expect("exists");
    assert_eq!(t.name, "weights");
    assert_eq!(t.shape, vec![2, 3]);
    assert_eq!(t.data, data);
}

#[test]
fn test_safetensors_multi_tensor_sorted_order() {
    let bytes = build_safetensors(&[
        ("z_layer", &[1], &[3.0]),
        ("a_layer", &[1], &[1.0]),
        ("m_layer", &[1], &[2.0]),
    ]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");
    let names: Vec<&str> = trace.names().collect();
    assert_eq!(names, vec!["a_layer", "m_layer", "z_layer"]);
}

#[test]
fn test_safetensors_empty_file_produces_empty_trace() {
    let bytes = build_safetensors(&[]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");
    assert!(trace.is_empty());
}

#[test]
fn test_safetensors_invalid_bytes_rejected() {
    let result = load_safetensors_from_bytes(b"not a valid safetensors file");
    assert!(matches!(result, Err(ReftestError::Safetensors(_))));
}

#[test]
fn test_safetensors_scalar_tensor() {
    let bytes = build_safetensors(&[("loss", &[], &[0.5])]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");
    let t = trace.get(0).expect("exists");
    assert!(t.shape.is_empty(), "scalar should have empty shape");
    assert_eq!(t.data, vec![0.5]);
}

#[test]
fn test_safetensors_high_dimensional_tensor() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let bytes = build_safetensors(&[("4d", &[2, 3, 2, 2], &data)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");
    let t = trace.get(0).expect("exists");
    assert_eq!(t.shape, vec![2, 3, 2, 2]);
    assert_eq!(t.numel(), 24);
}

// ===========================================================================
// 9. NPY loading: various dtypes and error paths
// ===========================================================================

#[test]
fn test_npy_f32_roundtrip_preserves_data() {
    let data = vec![0.0f32, 1.0, -1.0, 3.14159, f32::MIN_POSITIVE, f32::MAX];
    let shape = vec![2, 3];
    let bytes = write_npy_to_bytes(&data, &shape).expect("write");
    let tensor = read_npy_from_bytes(&bytes).expect("read");
    assert_eq!(tensor.shape, shape);
    assert_eq!(tensor.dtype, NpyDType::F32);
    for (i, (&orig, &loaded)) in data.iter().zip(tensor.data.iter()).enumerate() {
        assert_eq!(
            orig.to_bits(),
            loaded.to_bits(),
            "element {i} should be bit-identical"
        );
    }
}

#[test]
fn test_npy_f16_load_converts_to_f32() {
    let f16_values: Vec<half::f16> = vec![
        half::f16::from_f32(0.0),
        half::f16::from_f32(1.0),
        half::f16::from_f32(-0.5),
        half::f16::from_f32(65504.0),
    ];
    let raw: Vec<u8> = f16_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<f2", &[4], &raw);
    let tensor = read_npy_from_bytes(&npy).expect("f16 read should succeed");
    assert_eq!(tensor.dtype, NpyDType::F16);
    assert_eq!(tensor.data.len(), 4);
    assert!((tensor.data[0] - 0.0).abs() < 1e-4);
    assert!((tensor.data[1] - 1.0).abs() < 1e-3);
}

#[test]
fn test_npy_f64_load_converts_to_f32() {
    let values: Vec<f64> = vec![1.0, -2.5, 0.0, 1e30];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<f8", &[4], &raw);
    let tensor = read_npy_from_bytes(&npy).expect("f64 read should succeed");
    assert_eq!(tensor.dtype, NpyDType::F64);
    assert!((tensor.data[0] - 1.0).abs() < f32::EPSILON);
    assert!((tensor.data[3] - 1e30).abs() / 1e30 < 1e-6);
}

#[test]
fn test_npy_i32_load_converts_to_f32() {
    let values: Vec<i32> = vec![0, 1, -1, 100, -32768];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<i4", &[5], &raw);
    let tensor = read_npy_from_bytes(&npy).expect("i32 read should succeed");
    assert_eq!(tensor.data.len(), 5);
    assert!((tensor.data[3] - 100.0).abs() < f32::EPSILON);
}

#[test]
fn test_npy_i64_load_converts_to_f32() {
    let values: Vec<i64> = vec![0, 1, -1, 1000];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<i8", &[4], &raw);
    let tensor = read_npy_from_bytes(&npy).expect("i64 read should succeed");
    assert!((tensor.data[3] - 1000.0).abs() < f32::EPSILON);
}

#[test]
fn test_npy_u8_load_converts_to_f32() {
    let raw: Vec<u8> = vec![0, 1, 127, 255];
    let npy = build_npy_v1("|u1", &[4], &raw);
    let tensor = read_npy_from_bytes(&npy).expect("u8 read should succeed");
    assert!((tensor.data[3] - 255.0).abs() < f32::EPSILON);
}

#[test]
fn test_npy_big_endian_f32_load() {
    let values: Vec<f32> = vec![1.0, -2.5, 3.14];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
    let npy = build_npy_v1(">f4", &[3], &raw);
    let tensor = read_npy_from_bytes(&npy).expect("big-endian f32 read should succeed");
    assert!((tensor.data[0] - 1.0).abs() < f32::EPSILON);
    assert!((tensor.data[1] - (-2.5)).abs() < f32::EPSILON);
}

#[test]
fn test_npy_big_endian_f64_load() {
    let values: Vec<f64> = vec![1.0, -2.5];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
    let npy = build_npy_v1(">f8", &[2], &raw);
    let tensor = read_npy_from_bytes(&npy).expect("big-endian f64 read should succeed");
    assert!((tensor.data[0] - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_npy_scalar_roundtrip() {
    let data = vec![42.0f32];
    let bytes = write_npy_to_bytes(&data, &[]).expect("write scalar");
    let tensor = read_npy_from_bytes(&bytes).expect("read scalar");
    assert!(tensor.shape.is_empty());
    assert_eq!(tensor.data[0], 42.0);
}

#[test]
fn test_npy_high_dimensional_roundtrip() {
    let data: Vec<f32> = (0..120).map(|i| i as f32 * 0.1).collect();
    let shape = vec![2, 3, 4, 5];
    let bytes = write_npy_to_bytes(&data, &shape).expect("write");
    let tensor = read_npy_from_bytes(&bytes).expect("read");
    assert_eq!(tensor.shape, shape);
    assert_eq!(tensor.data.len(), 120);
}

#[test]
fn test_npy_special_float_roundtrip() {
    let data = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];
    let bytes = write_npy_to_bytes(&data, &[5]).expect("write");
    let tensor = read_npy_from_bytes(&bytes).expect("read");
    assert!(tensor.data[0].is_nan());
    assert_eq!(tensor.data[1], f32::INFINITY);
    assert_eq!(tensor.data[2], f32::NEG_INFINITY);
    assert_eq!(tensor.data[3].to_bits(), 0.0f32.to_bits());
    assert_eq!(tensor.data[4].to_bits(), (-0.0f32).to_bits());
}

#[test]
fn test_npy_bad_magic_rejected() {
    let err = read_npy_from_bytes(b"NOT_NPY_DATA_AT_ALL");
    assert!(err.is_err(), "bad magic should be rejected");
}

#[test]
fn test_npy_truncated_header_rejected() {
    let err = read_npy_from_bytes(b"\x93NUMPY");
    assert!(err.is_err(), "truncated file should be rejected");
}

#[test]
fn test_npy_fortran_order_rejected() {
    let header = "{'descr': '<f4', 'fortran_order': True, 'shape': (2,), }";
    let prefix_len = 10;
    let total_header = header.len() + 1;
    let padded_len = (prefix_len + total_header).div_ceil(64) * 64 - prefix_len;
    let padding = padded_len - header.len() - 1;

    let mut buf = Vec::new();
    buf.extend_from_slice(b"\x93NUMPY");
    buf.push(1);
    buf.push(0);
    buf.extend_from_slice(&(padded_len as u16).to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    buf.extend(std::iter::repeat_n(b' ', padding));
    buf.push(b'\n');
    buf.extend_from_slice(&[0u8; 8]);
    let err = read_npy_from_bytes(&buf);
    assert!(err.is_err(), "Fortran order should be rejected");
}

#[test]
fn test_npy_dtype_descr_roundtrip_all_variants() {
    for dtype in [
        NpyDType::F16,
        NpyDType::F32,
        NpyDType::F64,
        NpyDType::I32,
        NpyDType::I64,
        NpyDType::U8,
    ] {
        let descr = dtype.to_descr();
        let parsed = NpyDType::from_descr(descr);
        assert_eq!(parsed, Some(dtype), "roundtrip failed for {dtype:?}");
    }
}

#[test]
fn test_npy_write_rejects_data_shape_mismatch() {
    let err = write_npy_to_bytes(&[1.0, 2.0, 3.0, 4.0], &[2, 3]);
    assert!(err.is_err(), "data/shape mismatch should error");
}

#[test]
fn test_npy_write_empty_tensor() {
    let bytes = write_npy_to_bytes(&[], &[0]).expect("empty tensor write");
    let tensor = read_npy_from_bytes(&bytes).expect("empty tensor read");
    assert_eq!(tensor.data.len(), 0);
    assert_eq!(tensor.shape, vec![0]);
}

#[test]
fn test_npy_write_1d_shape_has_trailing_comma() {
    let bytes = write_npy_to_bytes(&[1.0, 2.0, 3.0], &[3]).expect("write");
    // NPY v1.0 layout: 6-byte magic, 2-byte version, 2-byte little-endian
    // header length, then the ASCII header, then the raw f32 payload. Decode
    // only the header region — slicing to the end would pull in the binary
    // f32 data, which is not valid UTF-8 and makes from_utf8 fail.
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header_str = std::str::from_utf8(&bytes[10..10 + header_len]).expect("header is ASCII");
    assert!(
        header_str.contains("(3,)"),
        "1-D shape should be written as (3,)"
    );
}

// ===========================================================================
// 10. Trace comparison: compare full model traces
// ===========================================================================

#[test]
fn test_trace_alignment_by_position() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace
        .checkpoint("encoder.conv1", &[1.0, 2.0], &[2])
        .expect("valid");
    ref_trace
        .checkpoint("encoder.relu1", &[3.0, 4.0], &[2])
        .expect("valid");

    let mut cand_trace = ReferenceTrace::new();
    cand_trace
        .checkpoint("layer_0", &[1.0, 2.0], &[2])
        .expect("valid");
    cand_trace
        .checkpoint("layer_1", &[3.0, 4.0], &[2])
        .expect("valid");

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("positional comparison should succeed");
    assert!(
        report.all_passed,
        "identical data should pass regardless of names"
    );
}

#[test]
fn test_trace_divergence_detected_at_correct_index() {
    let mut ref_trace = ReferenceTrace::new();
    for i in 0..5 {
        ref_trace
            .checkpoint(&format!("layer_{i}"), &[1.0], &[1])
            .expect("valid");
    }
    let mut cand_trace = ReferenceTrace::new();
    for i in 0..5 {
        let val = if i == 3 { 999.0 } else { 1.0 };
        cand_trace
            .checkpoint(&format!("layer_{i}"), &[val], &[1])
            .expect("valid");
    }
    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("should succeed");
    assert!(!report.all_passed);
    assert_eq!(report.first_failure, Some(3));
    assert!(report.layers[0].passed);
    assert!(report.layers[1].passed);
    assert!(report.layers[2].passed);
    assert!(!report.layers[3].passed);
    assert!(report.layers[4].passed);
}

#[test]
fn test_trace_multiple_failures_reports_first() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace.checkpoint("a", &[1.0], &[1]).expect("valid");
    ref_trace.checkpoint("b", &[2.0], &[1]).expect("valid");
    ref_trace.checkpoint("c", &[3.0], &[1]).expect("valid");

    let mut cand_trace = ReferenceTrace::new();
    cand_trace.checkpoint("a", &[100.0], &[1]).expect("valid");
    cand_trace.checkpoint("b", &[200.0], &[1]).expect("valid");
    cand_trace.checkpoint("c", &[3.0], &[1]).expect("valid");

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("should succeed");
    assert_eq!(report.first_failure, Some(0));
    assert!(!report.layers[0].passed);
    assert!(!report.layers[1].passed);
    assert!(report.layers[2].passed);
}

#[test]
fn test_trace_length_mismatch_error() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace.checkpoint("a", &[1.0], &[1]).expect("valid");
    ref_trace.checkpoint("b", &[2.0], &[1]).expect("valid");

    let mut cand_trace = ReferenceTrace::new();
    cand_trace.checkpoint("a", &[1.0], &[1]).expect("valid");

    let err = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect_err("should fail");
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

#[test]
fn test_trace_capture_records_in_order() {
    let (trace, ()) = ReferenceTrace::capture(|cap| {
        cap.checkpoint("first", &[1.0], &[1]).expect("valid");
        cap.checkpoint("second", &[2.0], &[1]).expect("valid");
        cap.checkpoint("third", &[3.0], &[1]).expect("valid");
    });
    let names: Vec<&str> = trace.names().collect();
    assert_eq!(names, vec!["first", "second", "third"]);
}

#[test]
fn test_trace_10_layers_all_pass() {
    let mut ref_trace = ReferenceTrace::new();
    let mut cand_trace = ReferenceTrace::new();
    for i in 0..10 {
        let data: Vec<f32> = (0..100).map(|j| (i * 100 + j) as f32 * 0.001).collect();
        ref_trace
            .checkpoint(&format!("layer_{i}"), &data, &[10, 10])
            .expect("valid");
        cand_trace
            .checkpoint(&format!("layer_{i}"), &data, &[10, 10])
            .expect("valid");
    }
    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("should succeed");
    assert!(report.all_passed);
    assert_eq!(report.layers.len(), 10);
}

#[test]
fn test_trace_mixed_pass_fail_alternating() {
    let mut ref_trace = ReferenceTrace::new();
    let mut cand_trace = ReferenceTrace::new();
    for i in 0..5 {
        let ref_data = vec![1.0f32; 10];
        let cand_data = if i % 2 == 0 {
            vec![1.0f32; 10]
        } else {
            vec![999.0f32; 10]
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
    assert_eq!(report.first_failure, Some(1));
    assert!(report.layers[0].passed);
    assert!(!report.layers[1].passed);
    assert!(report.layers[2].passed);
    assert!(!report.layers[3].passed);
    assert!(report.layers[4].passed);
}

#[test]
fn test_trace_from_checkpoints_empty_roundtrip() {
    let trace = ReferenceTrace::from_checkpoints(vec![]);
    assert!(trace.is_empty());
    assert_eq!(trace.len(), 0);
    let checkpoints = trace.into_checkpoints();
    assert!(checkpoints.is_empty());
}

#[test]
fn test_trace_get_by_name_returns_first_match() {
    let checkpoints = vec![
        NamedTensor::new("dup", vec![1], vec![1.0]).expect("valid"),
        NamedTensor::new("dup", vec![1], vec![2.0]).expect("valid"),
    ];
    let trace = ReferenceTrace::from_checkpoints(checkpoints);
    let found = trace.get_by_name("dup").expect("should find first");
    assert_eq!(found.data, vec![1.0], "should return first match");
}

// ===========================================================================
// 11. Summary statistics: max abs diff, mean abs diff, percentage within tol
// ===========================================================================

#[test]
fn test_comparison_result_max_diff_correct() {
    let a = vec![0.0f32, 0.0, 0.0, 0.0, 0.0];
    let b = vec![0.01, 0.5, 0.001, 1.0, 0.1];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 10.0 })
        .expect("should succeed");
    assert_eq!(result.worst_index, 3, "worst index should be 3 (diff=1.0)");
    assert!((result.max_diff - 1.0).abs() < 1e-6);
}

#[test]
fn test_comparison_result_mean_diff_correct() {
    // Uniform perturbation: mean_diff should equal max_diff.
    let n = 1000;
    let a: Vec<f32> = vec![0.0; n];
    let b: Vec<f32> = vec![0.01; n];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1.0 })
        .expect("should succeed");
    assert!(
        (result.max_diff - result.mean_diff).abs() < 1e-10,
        "uniform perturbation: max_diff ({}) should equal mean_diff ({})",
        result.max_diff,
        result.mean_diff
    );
}

#[test]
fn test_comparison_result_mismatch_count_correct() {
    // 10 elements: 3 are outside tolerance.
    let a = vec![0.0f32; 10];
    let mut b = vec![0.0f32; 10];
    b[2] = 1.0; // outside
    b[5] = 2.0; // outside
    b[8] = 3.0; // outside
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 0.5 })
        .expect("should succeed");
    assert_eq!(result.num_mismatches, 3, "should have exactly 3 mismatches");
    assert!(!result.passed);
}

#[test]
fn test_percent_within_tolerance_computation() {
    // 8 of 10 elements within threshold = 80%.
    let a = vec![0.0f32; 10];
    let mut b = vec![0.001f32; 10]; // all within 0.01
    b[0] = 1.0; // outlier
    b[9] = 2.0; // outlier
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::PercentClose {
            threshold: 0.01,
            percent: 80.0,
        },
    )
    .expect("should succeed");
    assert!(result.passed, "80% within threshold, 80% required");
    assert_eq!(result.num_mismatches, 2);

    let result_fail = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::PercentClose {
            threshold: 0.01,
            percent: 81.0,
        },
    )
    .expect("should succeed");
    assert!(
        !result_fail.passed,
        "80% within threshold, 81% required should fail"
    );
}

#[test]
fn test_rms_diff_computation() {
    // diffs = [0.1, 0.0] => rms = sqrt(0.01/2) = 0.0707...
    let a = tensor_1d("x", vec![1.0, 2.0]);
    let b = tensor_1d("x", vec![1.1, 2.0]);
    let rms_expected = (0.01_f64 / 2.0).sqrt() as f32;
    let config = ComparisonConfig {
        abs_tolerance: 1.0,
        rel_tolerance: 1.0,
        cosine_threshold: 0.0,
        rms_tolerance: None,
        peak_amplitude_limit: None,
        #[cfg(feature = "spectral")]
        spectral: None,
    };
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        (result.rms_diff - rms_expected).abs() < 1e-5,
        "rms_diff should be ~{rms_expected}, got {}",
        result.rms_diff
    );
}

#[test]
fn test_cosine_similarity_orthogonal_vectors() {
    // [1, 0] and [0, 1] => cos = 0.0
    let a = tensor_1d("x", vec![1.0, 0.0]);
    let b = tensor_1d("x", vec![0.0, 1.0]);
    let config = ComparisonConfig::new(100.0, 100.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        result.cosine_similarity.abs() < 1e-5,
        "orthogonal vectors should have cos=0.0, got {}",
        result.cosine_similarity
    );
}

#[test]
fn test_cosine_both_zero_vectors_treated_as_identical() {
    let a = tensor_1d("zero", vec![0.0, 0.0, 0.0]);
    let b = tensor_1d("zero", vec![0.0, 0.0, 0.0]);
    let config = ComparisonConfig::new(0.0, 0.0, 1.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert_eq!(
        result.cosine_similarity, 1.0,
        "zero vs zero should be cos=1.0"
    );
    assert!(result.passed);
}

#[test]
fn test_cosine_one_zero_one_nonzero_returns_zero() {
    let a = tensor_1d("x", vec![0.0, 0.0]);
    let b = tensor_1d("x", vec![1.0, 2.0]);
    let config = ComparisonConfig::new(100.0, 100.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert_eq!(
        result.cosine_similarity, 0.0,
        "zero vs nonzero should be cos=0.0"
    );
}

#[test]
fn test_cosine_antiparallel_vectors() {
    let a = tensor_1d("x", vec![1.0, 0.0]);
    let b = tensor_1d("x", vec![-1.0, 0.0]);
    let config = ComparisonConfig::new(100.0, 100.0, -1.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        (result.cosine_similarity - (-1.0)).abs() < 1e-5,
        "antiparallel vectors should have cos=-1.0"
    );
}

#[test]
fn test_peak_amplitude_tracks_candidate_max() {
    let a = tensor_1d("x", vec![0.0, 0.0]);
    let b = tensor_1d("x", vec![3.0, -7.0]);
    let config = ComparisonConfig::new(100.0, 100.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        (result.peak_amplitude - 7.0).abs() < 1e-5,
        "peak amplitude should be 7.0 (max abs of candidate)"
    );
}

// ===========================================================================
// 12. Report generation: human-readable comparison reports
// ===========================================================================

#[test]
fn test_layer_comparison_display_shows_pass() {
    let a = tensor_1d("attn", vec![1.0, 2.0, 3.0]);
    let b = tensor_1d("attn", vec![1.0, 2.0, 3.0]);
    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");
    let display = format!("{result}");
    assert!(display.contains("[PASS]"), "display should contain [PASS]");
    assert!(
        display.contains("attn"),
        "display should contain layer name"
    );
}

#[test]
fn test_layer_comparison_display_shows_fail() {
    let a = tensor_1d("layer0", vec![1.0]);
    let b = tensor_1d("layer0", vec![999.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed structurally");
    let display = format!("{result}");
    assert!(display.contains("[FAIL]"), "display should contain [FAIL]");
}

#[test]
fn test_layer_comparison_display_contains_metrics() {
    let a = tensor_1d("fc1", vec![1.0, 2.0]);
    let b = tensor_1d("fc1", vec![1.1, 2.2]);
    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");
    let display = format!("{result}");
    assert!(display.contains("max_abs="), "display should show max_abs");
    assert!(
        display.contains("mean_abs="),
        "display should show mean_abs"
    );
    assert!(display.contains("rms="), "display should show rms");
    assert!(display.contains("cos="), "display should show cosine");
    assert!(display.contains("max_rel="), "display should show max_rel");
    assert!(display.contains("peak="), "display should show peak");
}

#[test]
fn test_divergence_report_summary_all_passed() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace
        .checkpoint("embedding", &[1.0, 2.0], &[2])
        .expect("valid");
    ref_trace
        .checkpoint("attention", &[3.0, 4.0], &[2])
        .expect("valid");
    ref_trace
        .checkpoint("output", &[5.0, 6.0], &[2])
        .expect("valid");

    let report = compare_traces(&ref_trace, &ref_trace, &ComparisonConfig::default())
        .expect("should succeed");
    let summary = report.summary();
    assert!(
        summary.contains("embedding"),
        "summary should mention embedding"
    );
    assert!(
        summary.contains("attention"),
        "summary should mention attention"
    );
    assert!(summary.contains("output"), "summary should mention output");
    assert!(
        summary.contains("All 3 layers passed"),
        "summary should report all passed"
    );
}

#[test]
fn test_divergence_report_summary_shows_failure_layer() {
    let ref_trace = ReferenceTrace::from_checkpoints(vec![
        tensor_1d("a", vec![1.0]),
        tensor_1d("b", vec![2.0]),
    ]);
    let cand_trace = ReferenceTrace::from_checkpoints(vec![
        tensor_1d("a", vec![1.0]),
        tensor_1d("b", vec![999.0]),
    ]);
    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("should succeed");
    let summary = report.summary();
    assert!(
        summary.contains("First failure at layer 1"),
        "summary should identify failure location, got: {summary}"
    );
}

#[test]
fn test_divergence_report_summary_single_layer_pass() {
    let ref_trace = ReferenceTrace::from_checkpoints(vec![tensor_1d("only", vec![1.0])]);
    let report = compare_traces(&ref_trace, &ref_trace, &ComparisonConfig::default())
        .expect("should succeed");
    let summary = report.summary();
    assert!(summary.contains("All 1 layers passed"));
}

#[test]
fn test_divergence_report_summary_single_layer_fail() {
    let ref_trace = ReferenceTrace::from_checkpoints(vec![tensor_1d("only", vec![1.0])]);
    let cand_trace = ReferenceTrace::from_checkpoints(vec![tensor_1d("only", vec![999.0])]);
    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("should succeed");
    let summary = report.summary();
    assert!(summary.contains("First failure at layer 0"));
    assert!(summary.contains("only"));
}

// ===========================================================================
// Additional: gate boundary checks
// ===========================================================================

#[test]
fn test_rms_gate_exact_boundary() {
    let a = tensor_1d("x", vec![1.0, 2.0]);
    let b = tensor_1d("x", vec![1.1, 2.0]);
    let rms_expected = (0.01_f64 / 2.0).sqrt() as f32;

    let config_pass = ComparisonConfig {
        abs_tolerance: 1.0,
        rel_tolerance: 1.0,
        cosine_threshold: 0.0,
        rms_tolerance: Some(rms_expected + 0.001),
        peak_amplitude_limit: None,
        #[cfg(feature = "spectral")]
        spectral: None,
    };
    let result_pass = compare_tensors(&a, &b, &config_pass).expect("should succeed");
    assert!(result_pass.passed, "RMS just under limit should pass");

    let config_fail = ComparisonConfig {
        abs_tolerance: 1.0,
        rel_tolerance: 1.0,
        cosine_threshold: 0.0,
        rms_tolerance: Some(rms_expected - 0.001),
        peak_amplitude_limit: None,
        #[cfg(feature = "spectral")]
        spectral: None,
    };
    let result_fail = compare_tensors(&a, &b, &config_fail).expect("should succeed");
    assert!(!result_fail.passed, "RMS just over limit should fail");
}

#[test]
fn test_peak_amplitude_gate_exact_boundary() {
    let a = tensor_1d("x", vec![1.0]);
    let b = tensor_1d("x", vec![5.0]);

    let config_pass = ComparisonConfig {
        abs_tolerance: 100.0,
        rel_tolerance: 100.0,
        cosine_threshold: 0.0,
        rms_tolerance: None,
        peak_amplitude_limit: Some(5.0),
        #[cfg(feature = "spectral")]
        spectral: None,
    };
    let result = compare_tensors(&a, &b, &config_pass).expect("should succeed");
    assert!(result.passed, "peak == limit should pass");

    let config_fail = ComparisonConfig {
        abs_tolerance: 100.0,
        rel_tolerance: 100.0,
        cosine_threshold: 0.0,
        rms_tolerance: None,
        peak_amplitude_limit: Some(4.99),
        #[cfg(feature = "spectral")]
        spectral: None,
    };
    let result_fail = compare_tensors(&a, &b, &config_fail).expect("should succeed");
    assert!(!result_fail.passed, "peak > limit should fail");
}

#[test]
fn test_all_gates_must_pass_for_overall_pass() {
    let a = tensor_1d("x", vec![0.0, 0.0]);
    let b = tensor_1d("x", vec![0.001, 0.0]);
    let config = ComparisonConfig {
        abs_tolerance: 0.01,
        rel_tolerance: 1.0,
        cosine_threshold: 0.0,
        rms_tolerance: Some(1e-6),
        peak_amplitude_limit: None,
        #[cfg(feature = "spectral")]
        spectral: None,
    };
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        !result.passed,
        "tight RMS gate should cause failure even when abs/rel pass"
    );
}

#[test]
fn test_cosine_threshold_exact_boundary() {
    // [1, 0] and [1, 1]: cos = 1/sqrt(2) ~ 0.7071
    let a = tensor_1d("x", vec![1.0, 0.0]);
    let b = tensor_1d("x", vec![1.0, 1.0]);

    let config_pass = ComparisonConfig {
        abs_tolerance: 100.0,
        rel_tolerance: 100.0,
        cosine_threshold: 0.707,
        ..ComparisonConfig::default()
    };
    let result = compare_tensors(&a, &b, &config_pass).expect("should succeed");
    assert!(result.passed, "cos ~0.7071 >= threshold 0.707 should pass");

    let config_fail = ComparisonConfig {
        abs_tolerance: 100.0,
        rel_tolerance: 100.0,
        cosine_threshold: 0.708,
        ..ComparisonConfig::default()
    };
    let result_fail = compare_tensors(&a, &b, &config_fail).expect("should succeed");
    assert!(
        !result_fail.passed,
        "cos ~0.7071 < threshold 0.708 should fail"
    );
}

// ===========================================================================
// Additional: preset and config tests
// ===========================================================================

#[test]
fn test_preset_configs_produce_expected_thresholds() {
    let strict = TolerancePreset::STRICT.to_config();
    assert_eq!(strict.abs_tolerance, 1e-6);
    let standard = TolerancePreset::STANDARD.to_config();
    assert_eq!(standard.abs_tolerance, 1e-5);
    let transformer = TolerancePreset::TRANSFORMER.to_config();
    assert_eq!(transformer.abs_tolerance, 1e-4);
    let audio = TolerancePreset::AUDIO.to_config();
    assert_eq!(audio.abs_tolerance, 1e-3);
    let quantized = TolerancePreset::QUANTIZED.to_config();
    assert_eq!(quantized.abs_tolerance, 1e-2);
    let tts = TolerancePreset::TTS.to_config();
    assert_eq!(tts.abs_tolerance, 5e-3);
}

#[test]
fn test_preset_by_name_case_insensitive() {
    assert!(TolerancePreset::by_name("transformer").is_some());
    assert!(TolerancePreset::by_name("TRANSFORMER").is_some());
    assert!(TolerancePreset::by_name("Transformer").is_some());
    assert!(TolerancePreset::by_name("nonexistent").is_none());
}

#[test]
fn test_comparison_config_builder_chain() {
    let config = ComparisonConfig::new(1e-4, 1e-3, 0.999)
        .with_rms_tolerance(1e-2)
        .with_peak_amplitude_limit(50.0);
    assert_eq!(config.abs_tolerance, 1e-4);
    assert_eq!(config.rel_tolerance, 1e-3);
    assert_eq!(config.cosine_threshold, 0.999);
    assert_eq!(config.rms_tolerance, Some(1e-2));
    assert_eq!(config.peak_amplitude_limit, Some(50.0));
}

#[test]
fn test_comparison_config_strict_factory() {
    let strict = ComparisonConfig::strict();
    assert_eq!(strict.abs_tolerance, 1e-6);
    assert_eq!(strict.rel_tolerance, 1e-5);
    assert_eq!(strict.cosine_threshold, 0.999_999);
}

#[test]
fn test_comparison_config_relaxed_factory() {
    let relaxed = ComparisonConfig::relaxed();
    assert_eq!(relaxed.abs_tolerance, 1e-2);
    assert_eq!(relaxed.rel_tolerance, 1e-1);
    assert_eq!(relaxed.cosine_threshold, 0.999);
}

#[test]
fn test_comparison_config_default_matches_standard_preset() {
    let default_config = ComparisonConfig::default();
    let standard = TolerancePreset::STANDARD.to_config();
    assert_eq!(default_config.abs_tolerance, standard.abs_tolerance);
    assert_eq!(default_config.rel_tolerance, standard.rel_tolerance);
    assert_eq!(default_config.cosine_threshold, standard.cosine_threshold);
}

// ===========================================================================
// Additional: tolerance error paths
// ===========================================================================

#[test]
fn test_tolerance_data_length_mismatch() {
    let a = vec![1.0f32; 3];
    let b = vec![1.0f32; 5];
    let err = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1.0 })
        .expect_err("mismatched lengths should error");
    assert!(matches!(
        err,
        ReftestError::DataLengthMismatch {
            expected: 5,
            actual: 3
        }
    ));
}

#[test]
fn test_tolerance_empty_tensor_rejected() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let err = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1.0 })
        .expect_err("empty slices should error");
    assert!(matches!(err, ReftestError::EmptyTensor(_)));
}

#[test]
fn test_named_tensor_zero_dimensional_scalar() {
    let t = NamedTensor::new("scalar", vec![], vec![3.14]).expect("valid scalar");
    assert!(t.shape.is_empty());
    assert_eq!(t.numel(), 1);
    assert_eq!(t.data[0], 3.14);
}

// ===========================================================================
// Additional: batch comparison with varying strategies
// ===========================================================================

#[test]
fn test_batch_tolerance_comparison_varying_strategies() {
    // Base values are nonzero and of similar magnitude (1.0..2.0) so that a
    // genuinely small perturbation is "small" in every sense the strategies
    // measure: absolute, relative, ULP, and percent-close. Starting at 0.0
    // would make the relative error 100% at the first element, and a 1e-5
    // absolute perturbation spans ~10^6 ULPs (far above max_ulps=100).
    let a: Vec<f32> = (0..100).map(|i| 1.0 + i as f32 * 0.01).collect();
    let b: Vec<f32> = a.iter().map(|&x| x + 1e-6).collect();

    let strategies = [
        ("absolute", ToleranceStrategy::Absolute { atol: 1e-3 }),
        ("relative", ToleranceStrategy::Relative { rtol: 1e-3 }),
        (
            "mixed",
            ToleranceStrategy::Mixed {
                atol: 1e-3,
                rtol: 1e-3,
            },
        ),
        ("ulp", ToleranceStrategy::ULP { max_ulps: 100 }),
        (
            "percent_close",
            ToleranceStrategy::PercentClose {
                threshold: 1e-3,
                percent: 99.0,
            },
        ),
    ];

    for (name, strategy) in &strategies {
        let result = compare_with_tolerance(&a, &b, strategy).expect("should succeed");
        assert!(
            result.passed,
            "strategy '{name}' should pass for small perturbation"
        );
    }
}

#[test]
fn test_comparison_without_spectral_feature_ignores_spectral_gate() {
    let a = tensor_1d("audio", vec![0.1, 0.2, 0.3, 0.4]);
    let b = tensor_1d("audio", vec![0.1, 0.2, 0.3, 0.4]);
    let config = ComparisonConfig::default();
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        result.passed,
        "identical 1-D tensors should pass without spectral gate"
    );
}
