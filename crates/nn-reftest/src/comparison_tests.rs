// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for tensor comparison infrastructure: tolerance strategies, shape/dtype
//! handling, NaN/Inf edge cases, and combined atol+rtol (NumPy-style) semantics.

use crate::compare::{compare_tensors, ComparisonConfig};
use crate::error::ReftestError;
use crate::tolerance::{compare_with_tolerance, ToleranceStrategy};
use crate::trace::NamedTensor;

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

// ===========================================================================
// 1. Exact match for integer-valued tensors
// ===========================================================================

#[test]
fn test_exact_match_integer_values_absolute() {
    // Integer-valued f32 tensors should match exactly with atol=0.
    let expected = [1.0f32, 2.0, 3.0, 4.0, 5.0, 100.0, -50.0, 0.0];
    let actual = expected;
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 0.0 },
    )
    .expect("comparison should succeed");
    assert!(
        result.passed,
        "exact integer match must pass with zero tolerance"
    );
    assert_eq!(result.max_diff, 0.0);
    assert_eq!(result.mean_diff, 0.0);
    assert_eq!(result.num_mismatches, 0);
}

#[test]
fn test_exact_match_integer_values_mixed() {
    let expected = [0.0f32, 1.0, -1.0, 255.0, -128.0];
    let result = compare_with_tolerance(
        &expected,
        &expected,
        &ToleranceStrategy::Mixed {
            atol: 0.0,
            rtol: 0.0,
        },
    )
    .expect("comparison should succeed");
    assert!(result.passed);
    assert_eq!(result.num_mismatches, 0);
}

#[test]
fn test_exact_match_integer_values_ulp() {
    let expected = [42.0f32, -99.0, 0.0, 1024.0];
    let result = compare_with_tolerance(
        &expected,
        &expected,
        &ToleranceStrategy::ULP { max_ulps: 0 },
    )
    .expect("comparison should succeed");
    assert!(
        result.passed,
        "identical integer floats should be 0 ULPs apart"
    );
}

#[test]
fn test_integer_off_by_one_fails_exact() {
    // 5.0 vs 6.0 should fail with atol=0.5 (diff=1.0 > 0.5).
    let expected = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let actual = [1.0f32, 2.0, 3.0, 4.0, 6.0];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 0.5 },
    )
    .expect("comparison should succeed");
    assert!(!result.passed);
    assert_eq!(result.num_mismatches, 1);
    assert_eq!(result.worst_index, 4);
    assert!((result.max_diff - 1.0).abs() < 1e-10);
}

// ===========================================================================
// 2. Approximate match with absolute tolerance
// ===========================================================================

#[test]
fn test_absolute_tolerance_just_within() {
    let expected = [10.0f32, 20.0, 30.0];
    let actual = [10.0005f32, 20.0005, 30.0005]; // all diffs = 0.0005
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 0.001 },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "diffs well within atol boundary should pass");
    assert_eq!(result.num_mismatches, 0);
}

#[test]
fn test_absolute_tolerance_just_beyond() {
    let expected = [10.0f32];
    let actual = [10.0011f32]; // diff = 0.0011 > 0.001
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 0.001 },
    )
    .expect("comparison should succeed");
    assert!(!result.passed, "diff beyond atol should fail");
    assert_eq!(result.num_mismatches, 1);
}

#[test]
fn test_absolute_tolerance_negative_diffs() {
    // Candidate undershoots the expected value.
    let expected = [5.0f32, 10.0];
    let actual = [4.999f32, 9.999];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 0.01 },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "negative diffs within atol should pass");
}

// ===========================================================================
// 3. Approximate match with relative tolerance
// ===========================================================================

#[test]
fn test_relative_tolerance_proportional_error() {
    // Values at different scales with the same relative error (~0.1%).
    let expected = [1.0f32, 100.0, 10000.0];
    let actual = [1.001f32, 100.1, 10010.0];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Relative { rtol: 0.002 },
    )
    .expect("comparison should succeed");
    assert!(
        result.passed,
        "0.1% relative error should pass with rtol=0.2%"
    );
}

#[test]
fn test_relative_tolerance_fails_for_small_values() {
    // Near-zero values: absolute diff is tiny but relative diff is large.
    let expected = [1e-6f32];
    let actual = [2e-6f32]; // 100% relative error
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Relative { rtol: 0.5 },
    )
    .expect("comparison should succeed");
    // Relative diff = |1e-6| / max(2e-6, 1e-6, 1e-8) = 1e-6 / 2e-6 = 0.5
    // The epsilon floor of 1e-8 means we use max(|a|, |b|, 1e-8) = 2e-6.
    assert!(
        result.passed,
        "relative diff 0.5 == rtol 0.5 should pass (boundary)"
    );
}

#[test]
fn test_relative_tolerance_rejects_20_percent_error() {
    let expected = [100.0f32];
    let actual = [120.0f32]; // 20% relative error
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Relative { rtol: 0.1 },
    )
    .expect("comparison should succeed");
    assert!(!result.passed, "20% error > 10% rtol should fail");
}

// ===========================================================================
// 4. Combined atol+rtol (NumPy-style: |a-b| <= atol + rtol*|b|)
// ===========================================================================

#[test]
fn test_mixed_numpy_semantics_small_value() {
    // For small expected value (b=0.001): threshold = atol + rtol*|b| = 1e-5 + 1e-3 * 0.001 = 1.1e-5.
    let expected = [0.001f32];
    let actual = [0.001_01f32]; // diff = 1e-5, within 1.1e-5
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Mixed {
            atol: 1e-5,
            rtol: 1e-3,
        },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "small values should rely on atol component");
}

#[test]
fn test_mixed_numpy_semantics_large_value() {
    // For large expected value (b=1000): threshold = 1e-5 + 1e-3 * 1000 = 1.00001.
    let expected = [1000.0f32];
    let actual = [1001.0f32]; // diff = 1.0, within 1.00001
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Mixed {
            atol: 1e-5,
            rtol: 1e-3,
        },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "large values should rely on rtol component");
}

#[test]
fn test_mixed_numpy_semantics_exceeds_both() {
    // diff = 2.0, threshold = 1e-5 + 1e-3 * 1.0 = 0.001001.
    let expected = [1.0f32];
    let actual = [3.0f32];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Mixed {
            atol: 1e-5,
            rtol: 1e-3,
        },
    )
    .expect("comparison should succeed");
    assert!(!result.passed);
}

#[test]
fn test_mixed_tolerance_with_negative_expected() {
    // NumPy semantics use |b|. For b=-500: threshold = 0.01 + 0.001*500 = 0.51.
    let expected = [-500.0f32];
    let actual = [-500.4f32]; // diff = 0.4 < 0.51
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Mixed {
            atol: 0.01,
            rtol: 0.001,
        },
    )
    .expect("comparison should succeed");
    assert!(
        result.passed,
        "negative expected with |b| semantics should pass"
    );
}

#[test]
fn test_mixed_zero_atol_pure_relative() {
    // With atol=0, Mixed degenerates to pure relative tolerance on |b|.
    let expected = [100.0f32];
    let actual = [100.05f32]; // diff = 0.05, threshold = 0.0 + 0.001 * 100.0 = 0.1
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Mixed {
            atol: 0.0,
            rtol: 0.001,
        },
    )
    .expect("comparison should succeed");
    assert!(
        result.passed,
        "with atol=0, should behave as pure rtol on |b|"
    );
}

#[test]
fn test_mixed_zero_rtol_pure_absolute() {
    // With rtol=0, Mixed degenerates to pure absolute tolerance.
    let expected = [100.0f32];
    let actual = [100.0005f32]; // diff = 5e-4, threshold = 1e-3 + 0 = 1e-3
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Mixed {
            atol: 1e-3,
            rtol: 0.0,
        },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "with rtol=0, should behave as pure atol");
}

// ===========================================================================
// 5. Shape mismatch detection
// ===========================================================================

#[test]
fn test_shape_mismatch_different_rank() {
    let ref_t = tensor_nd("layer", vec![2, 3], vec![0.0; 6]);
    let cand_t = tensor_1d("layer", vec![0.0; 6]);
    let err = compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default())
        .expect_err("should fail on shape mismatch");
    match err {
        ReftestError::ShapeMismatch {
            name,
            expected,
            actual,
        } => {
            assert_eq!(name, "layer");
            assert_eq!(expected, vec![2, 3]);
            assert_eq!(actual, vec![6]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn test_shape_mismatch_same_rank_different_dims() {
    let ref_t = tensor_nd("conv1", vec![4, 3], vec![0.0; 12]);
    let cand_t = tensor_nd("conv1", vec![3, 4], vec![0.0; 12]);
    let err = compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default())
        .expect_err("should fail on transposed shapes");
    assert!(matches!(err, ReftestError::ShapeMismatch { .. }));
}

#[test]
fn test_shape_mismatch_different_total_elements() {
    let ref_t = tensor_nd("fc", vec![2, 3], vec![0.0; 6]);
    let cand_t = tensor_nd("fc", vec![2, 4], vec![0.0; 8]);
    let err = compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default())
        .expect_err("should fail on different element counts via shape");
    assert!(matches!(err, ReftestError::ShapeMismatch { .. }));
}

// ===========================================================================
// 6. Dtype mismatch handling (tolerance strategy length mismatch)
// ===========================================================================

#[test]
fn test_data_length_mismatch_in_tolerance() {
    // Simulates a dtype mismatch where the data vectors have different lengths.
    let a = [1.0f32, 2.0, 3.0];
    let b = [1.0f32, 2.0];
    let err = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1.0 })
        .expect_err("should fail on length mismatch");
    match err {
        ReftestError::DataLengthMismatch { expected, actual } => {
            assert_eq!(expected, 2);
            assert_eq!(actual, 3);
        }
        other => panic!("expected DataLengthMismatch, got {other:?}"),
    }
}

#[test]
fn test_named_tensor_element_count_vs_shape() {
    // Creating a NamedTensor with mismatched element count and shape
    // simulates a dtype conversion error where the element count doesn't match.
    let result = NamedTensor::new("bad_dtype", vec![2, 3], vec![1.0; 4]);
    match result {
        Err(ReftestError::ElementCountMismatch {
            expected, actual, ..
        }) => {
            assert_eq!(expected, 6);
            assert_eq!(actual, 4);
        }
        other => panic!("expected ElementCountMismatch, got {other:?}"),
    }
}

// ===========================================================================
// 7. NaN handling in comparisons
// ===========================================================================

#[test]
fn test_nan_vs_nan_always_fails_ieee754() {
    // IEEE 754: NaN != NaN. The comparison infrastructure treats NaN as
    // infinite divergence, not as equal. This is the standard behavior.
    let a = tensor_1d("nan_layer", vec![f32::NAN, f32::NAN]);
    let b = tensor_1d("nan_layer", vec![f32::NAN, f32::NAN]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(
        !result.passed,
        "NaN vs NaN must fail under IEEE 754 semantics"
    );
    assert!(
        result.max_abs_diff.is_infinite(),
        "NaN divergence should report infinite max_abs_diff"
    );
}

#[test]
fn test_nan_only_in_reference_fails() {
    let a = tensor_1d("ref", vec![f32::NAN, 1.0, 2.0]);
    let b = tensor_1d("ref", vec![0.0, 1.0, 2.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");
    assert!(!result.passed, "NaN in reference must cause failure");
}

#[test]
fn test_nan_only_in_candidate_fails() {
    let a = tensor_1d("cand", vec![0.0, 1.0, 2.0]);
    let b = tensor_1d("cand", vec![0.0, f32::NAN, 2.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");
    assert!(!result.passed, "NaN in candidate must cause failure");
}

#[test]
fn test_nan_with_tolerance_strategy_absolute() {
    // NaN should fail even with infinite tolerance.
    let result = compare_with_tolerance(
        &[f32::NAN],
        &[1.0f32],
        &ToleranceStrategy::Absolute { atol: f64::MAX },
    )
    .expect("comparison should succeed");
    assert!(!result.passed, "NaN should fail absolute tolerance");
    assert_eq!(result.num_mismatches, 1);
}

#[test]
fn test_nan_with_tolerance_strategy_mixed() {
    let result = compare_with_tolerance(
        &[1.0f32],
        &[f32::NAN],
        &ToleranceStrategy::Mixed {
            atol: f64::MAX,
            rtol: f64::MAX,
        },
    )
    .expect("comparison should succeed");
    assert!(!result.passed, "NaN should fail mixed tolerance");
}

#[test]
fn test_nan_cosine_similarity_all_nan_is_nan() {
    // When all elements are NaN, cosine similarity should be NaN (undefined),
    // not 1.0 (which would incorrectly indicate identical tensors).
    let a = tensor_1d("all_nan", vec![f32::NAN, f32::NAN, f32::NAN]);
    let b = tensor_1d("all_nan", vec![f32::NAN, f32::NAN, f32::NAN]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(
        result.cosine_similarity.is_nan(),
        "all-NaN cosine should be NaN, got {}",
        result.cosine_similarity
    );
}

#[test]
fn test_partial_nan_cosine_still_finite() {
    // When some elements are NaN but others are finite, the cosine similarity
    // should be computed from the finite elements only and remain finite.
    let a = tensor_1d("partial", vec![1.0, f32::NAN, 3.0, 4.0]);
    let b = tensor_1d("partial", vec![1.0, 2.0, 3.0, 4.0]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(
        result.cosine_similarity.is_finite(),
        "partial NaN cosine should be finite from valid elements, got {}",
        result.cosine_similarity
    );
}

// ===========================================================================
// 8. Inf handling in comparisons
// ===========================================================================

#[test]
fn test_positive_inf_vs_positive_inf_fails() {
    // Even matching infinities should fail (non-finite defense-in-depth).
    let a = tensor_1d("inf", vec![f32::INFINITY]);
    let b = tensor_1d("inf", vec![f32::INFINITY]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(!result.passed, "matching infinities must still fail");
}

#[test]
fn test_neg_inf_vs_neg_inf_fails() {
    let a = tensor_1d("ninf", vec![f32::NEG_INFINITY]);
    let b = tensor_1d("ninf", vec![f32::NEG_INFINITY]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(!result.passed, "matching negative infinities must fail");
}

#[test]
fn test_positive_inf_vs_neg_inf_fails() {
    let a = tensor_1d("mixed_inf", vec![f32::INFINITY]);
    let b = tensor_1d("mixed_inf", vec![f32::NEG_INFINITY]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(!result.passed);
}

#[test]
fn test_inf_peak_amplitude_is_infinity() {
    let a = tensor_1d("peak", vec![1.0, 2.0]);
    let b = tensor_1d("peak", vec![1.0, f32::INFINITY]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");
    assert!(
        result.peak_amplitude.is_infinite(),
        "Inf in candidate should produce infinite peak amplitude"
    );
}

#[test]
fn test_inf_in_tolerance_compare_fails_all_strategies() {
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
            percent: 100.0,
        },
    ] {
        let result = compare_with_tolerance(&[f32::INFINITY], &[f32::INFINITY], strategy)
            .expect("comparison should succeed");
        assert!(!result.passed, "Inf vs Inf should fail for {strategy:?}");
    }
}

// ===========================================================================
// 9. Multi-dimensional tensor comparison
// ===========================================================================

#[test]
fn test_2d_tensor_comparison_passes() {
    let ref_t = tensor_nd("matrix", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let cand_t = tensor_nd("matrix", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.num_elements, 6);
}

#[test]
fn test_3d_tensor_comparison_with_small_diff() {
    let data_ref = vec![1.0f32; 2 * 3 * 4];
    let data_cand: Vec<f32> = data_ref.iter().map(|x| x + 1e-7).collect();
    let ref_t = tensor_nd("volume", vec![2, 3, 4], data_ref);
    let cand_t = tensor_nd("volume", vec![2, 3, 4], data_cand);
    let result =
        compare_tensors(&ref_t, &cand_t, &ComparisonConfig::default()).expect("should succeed");
    assert!(
        result.passed,
        "tiny perturbation should pass default tolerance"
    );
}

// ===========================================================================
// 10. Config presets affect pass/fail
// ===========================================================================

#[test]
fn test_strict_rejects_what_relaxed_accepts() {
    let a = tensor_1d("x", vec![1.0, 2.0, 3.0]);
    let b = tensor_1d("x", vec![1.001, 2.001, 3.001]);

    let strict = compare_tensors(&a, &b, &ComparisonConfig::strict()).expect("should succeed");
    let relaxed = compare_tensors(&a, &b, &ComparisonConfig::relaxed()).expect("should succeed");

    assert!(!strict.passed, "strict should reject 0.001 diffs");
    assert!(relaxed.passed, "relaxed should accept 0.001 diffs");
}

// ===========================================================================
// 11. RMS and peak amplitude gates interaction
// ===========================================================================

#[test]
fn test_rms_gate_catches_distributed_error() {
    // Small max_abs but high RMS from many small errors.
    let n = 100;
    let a_data = vec![0.0f32; n];
    let b_data = vec![0.05f32; n]; // every element off by 0.05
    let a = tensor_nd("rms_test", vec![n], a_data);
    let b = tensor_nd("rms_test", vec![n], b_data);
    let config = ComparisonConfig {
        abs_tolerance: 0.1,
        rel_tolerance: 1.0,
        cosine_threshold: 0.0,
        rms_tolerance: Some(0.01), // RMS = 0.05 > 0.01
        peak_amplitude_limit: None,
        #[cfg(feature = "spectral")]
        spectral: None,
    };
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        !result.passed,
        "RMS gate should catch distributed error even when max_abs passes"
    );
}

#[test]
fn test_peak_amplitude_catches_explosion() {
    let a = tensor_1d("peak", vec![1.0, 2.0, 3.0]);
    let b = tensor_1d("peak", vec![1.0, 2.0, 1e6]);
    let config = ComparisonConfig {
        abs_tolerance: 1e7,
        rel_tolerance: 1.0,
        cosine_threshold: 0.0,
        rms_tolerance: None,
        peak_amplitude_limit: Some(1e5),
        #[cfg(feature = "spectral")]
        spectral: None,
    };
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        !result.passed,
        "peak amplitude gate should catch output explosion"
    );
    assert_eq!(result.peak_amplitude, 1e6);
}

// ===========================================================================
// 12. Cosine similarity edge cases
// ===========================================================================

#[test]
fn test_cosine_anti_parallel_vectors() {
    // Vectors pointing in opposite directions: cosine = -1.0.
    let a = tensor_1d("anti", vec![1.0, 0.0, 0.0]);
    let b = tensor_1d("anti", vec![-1.0, 0.0, 0.0]);
    let config = ComparisonConfig::new(100.0, 100.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        (result.cosine_similarity - (-1.0)).abs() < 1e-5,
        "anti-parallel vectors should have cosine ~-1.0, got {}",
        result.cosine_similarity
    );
}

#[test]
fn test_cosine_scaled_vectors_identical_direction() {
    // Same direction, different magnitude: cosine = 1.0.
    let a = tensor_1d("scaled", vec![1.0, 2.0, 3.0]);
    let b = tensor_1d("scaled", vec![10.0, 20.0, 30.0]);
    let config = ComparisonConfig::new(100.0, 100.0, 0.999);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        (result.cosine_similarity - 1.0).abs() < 1e-5,
        "scaled vectors should have cosine ~1.0, got {}",
        result.cosine_similarity
    );
}

// ===========================================================================
// 13. Empty and single-element edge cases
// ===========================================================================

#[test]
fn test_empty_tensor_comparison_returns_error() {
    let a = NamedTensor::new("empty", vec![0], vec![]).expect("valid zero-element tensor");
    let b = NamedTensor::new("empty", vec![0], vec![]).expect("valid zero-element tensor");
    let err = compare_tensors(&a, &b, &ComparisonConfig::default())
        .expect_err("should fail on empty tensor");
    assert!(matches!(err, ReftestError::EmptyTensor(_)));
}

#[test]
fn test_single_element_scalar_shape_comparison() {
    // Scalar tensors (shape=[]) with one element.
    let a = NamedTensor::new("scalar", vec![], vec![3.14]).expect("valid scalar");
    let b = NamedTensor::new("scalar", vec![], vec![3.14]).expect("valid scalar");
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");
    assert!(result.passed);
    assert_eq!(result.num_elements, 1);
}
