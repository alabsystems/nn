// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for configurable tolerance strategies.

use super::*;

// ---- Exact match passes all strategies ----

#[test]
fn test_exact_match_passes_absolute() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let result = compare_with_tolerance(&data, &data, &ToleranceStrategy::Absolute { atol: 0.0 })
        .expect("comparison should succeed");
    assert!(result.passed);
    assert_eq!(result.max_diff, 0.0);
    assert_eq!(result.mean_diff, 0.0);
    assert_eq!(result.num_mismatches, 0);
}

#[test]
fn test_exact_match_passes_relative() {
    let data = [1.0f32, 2.0, 3.0];
    let result = compare_with_tolerance(&data, &data, &ToleranceStrategy::Relative { rtol: 0.0 })
        .expect("comparison should succeed");
    assert!(result.passed);
    assert_eq!(result.num_mismatches, 0);
}

#[test]
fn test_exact_match_passes_mixed() {
    let data = [1.0f32, -1.0, 0.0, 100.0];
    let result = compare_with_tolerance(
        &data,
        &data,
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
fn test_exact_match_passes_ulp() {
    let data = [1.0f32, 2.0, 0.5, -3.0];
    let result = compare_with_tolerance(&data, &data, &ToleranceStrategy::ULP { max_ulps: 0 })
        .expect("comparison should succeed");
    assert!(result.passed);
    assert_eq!(result.num_mismatches, 0);
}

#[test]
fn test_exact_match_passes_percent_close() {
    let data = [1.0f32, 2.0, 3.0];
    let result = compare_with_tolerance(
        &data,
        &data,
        &ToleranceStrategy::PercentClose {
            threshold: 0.0,
            percent: 100.0,
        },
    )
    .expect("comparison should succeed");
    assert!(result.passed);
    assert_eq!(result.num_mismatches, 0);
}

// ---- Absolute tolerance ----

#[test]
fn test_absolute_small_perturbation_passes() {
    let expected = [1.0f32, 2.0, 3.0];
    let actual = [1.0001, 2.0001, 3.0001];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 1e-3 },
    )
    .expect("comparison should succeed");
    assert!(result.passed);
    assert!(result.max_diff < 2e-4);
    assert_eq!(result.num_mismatches, 0);
}

#[test]
fn test_absolute_large_perturbation_fails() {
    let expected = [1.0f32, 2.0, 3.0];
    let actual = [1.1, 2.0, 3.0];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 1e-3 },
    )
    .expect("comparison should succeed");
    assert!(!result.passed);
    assert_eq!(result.num_mismatches, 1);
    assert_eq!(result.worst_index, 0);
    assert!((result.max_diff - 0.1).abs() < 1e-6);
}

#[test]
fn test_absolute_boundary_exactly_at_tolerance() {
    // Element difference is exactly at the boundary.
    let expected = [0.0f32];
    let actual = [1e-5f32];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 1e-5 },
    )
    .expect("comparison should succeed");
    // f64::from(1e-5f32) should be <= 1e-5 (they are equal as f64).
    assert!(result.passed);
}

// ---- Relative tolerance ----

#[test]
fn test_relative_small_perturbation_passes() {
    let expected = [100.0f32, 200.0, 300.0];
    let actual = [100.001, 200.002, 300.003];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Relative { rtol: 1e-4 },
    )
    .expect("comparison should succeed");
    assert!(result.passed);
}

#[test]
fn test_relative_large_perturbation_fails() {
    let expected = [100.0f32];
    let actual = [110.0f32]; // 10% relative error
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Relative { rtol: 0.01 },
    )
    .expect("comparison should succeed");
    assert!(!result.passed);
    assert_eq!(result.num_mismatches, 1);
}

// ---- Mixed tolerance (NumPy-style) ----

#[test]
fn test_mixed_atol_dominates_near_zero() {
    // Near zero: absolute tolerance matters more than relative.
    let expected = [1e-10f32];
    let actual = [2e-10f32];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Mixed {
            atol: 1e-6,
            rtol: 1e-3,
        },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "atol should dominate near zero");
}

#[test]
fn test_mixed_rtol_dominates_large_values() {
    // Large values: relative tolerance provides the margin.
    let expected = [1000.0f32];
    let actual = [1000.5f32];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Mixed {
            atol: 1e-6,
            rtol: 1e-3,
        },
    )
    .expect("comparison should succeed");
    // |1000.5 - 1000.0| = 0.5, threshold = 1e-6 + 1e-3 * 1000.0 = 1.000001
    assert!(
        result.passed,
        "rtol should provide enough margin for large values"
    );
}

#[test]
fn test_mixed_both_insufficient_fails() {
    let expected = [1.0f32];
    let actual = [2.0f32];
    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Mixed {
            atol: 1e-6,
            rtol: 1e-3,
        },
    )
    .expect("comparison should succeed");
    // |2.0 - 1.0| = 1.0, threshold = 1e-6 + 1e-3 * 1.0 = 0.001001
    assert!(!result.passed);
}

// ---- ULP comparison ----

#[test]
fn test_ulp_adjacent_floats_pass() {
    // f32 next-representable values differ by 1 ULP.
    let a = 1.0f32;
    let b = f32::from_bits(a.to_bits() + 1);
    assert_ne!(a, b);

    let result = compare_with_tolerance(&[a], &[b], &ToleranceStrategy::ULP { max_ulps: 1 })
        .expect("comparison should succeed");
    assert!(result.passed, "adjacent floats should be within 1 ULP");
    assert_eq!(result.num_mismatches, 0);
}

#[test]
fn test_ulp_zero_tolerance_rejects_adjacent() {
    let a = 1.0f32;
    let b = f32::from_bits(a.to_bits() + 1);
    let result = compare_with_tolerance(&[a], &[b], &ToleranceStrategy::ULP { max_ulps: 0 })
        .expect("comparison should succeed");
    assert!(
        !result.passed,
        "0 ULP tolerance should reject adjacent floats"
    );
    assert_eq!(result.num_mismatches, 1);
}

#[test]
fn test_ulp_nan_always_fails() {
    let result = compare_with_tolerance(
        &[f32::NAN],
        &[f32::NAN],
        &ToleranceStrategy::ULP { max_ulps: u32::MAX },
    )
    .expect("comparison should succeed");
    assert!(
        !result.passed,
        "NaN should never match even with max ULP tolerance"
    );
}

#[test]
fn test_ulp_positive_negative_zero() {
    // +0.0 and -0.0 are 0 ULPs apart in the reflected integer space
    // (both map to 0 after the sign adjustment).
    let result = compare_with_tolerance(
        &[0.0f32],
        &[-0.0f32],
        &ToleranceStrategy::ULP { max_ulps: 0 },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "+0.0 and -0.0 should be 0 ULPs apart");
}

#[test]
fn test_ulp_across_zero_boundary() {
    // Smallest positive subnormal and smallest negative subnormal are 2 ULPs apart.
    let pos = f32::from_bits(1); // smallest positive subnormal
    let neg = f32::from_bits(0x8000_0001); // smallest negative subnormal

    let result_tight =
        compare_with_tolerance(&[pos], &[neg], &ToleranceStrategy::ULP { max_ulps: 1 })
            .expect("comparison should succeed");
    assert!(
        !result_tight.passed,
        "1 ULP should not bridge across zero for 2-ULP gap"
    );

    let result_loose =
        compare_with_tolerance(&[pos], &[neg], &ToleranceStrategy::ULP { max_ulps: 2 })
            .expect("comparison should succeed");
    assert!(result_loose.passed, "2 ULP should bridge the gap");
}

#[test]
fn test_ulp_large_difference() {
    let result = compare_with_tolerance(
        &[1.0f32],
        &[2.0f32],
        &ToleranceStrategy::ULP { max_ulps: 10 },
    )
    .expect("comparison should succeed");
    assert!(!result.passed, "1.0 vs 2.0 is millions of ULPs apart");
}

// ---- PercentClose tolerance ----

#[test]
fn test_percent_close_allows_outliers() {
    // 4 out of 5 elements are close, 1 is far away.
    let expected = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let actual = [1.0, 2.0, 3.0, 4.0, 100.0]; // 80% close

    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::PercentClose {
            threshold: 0.1,
            percent: 75.0,
        },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "80% close >= 75% threshold should pass");
    assert_eq!(result.num_mismatches, 1);
}

#[test]
fn test_percent_close_too_many_outliers_fails() {
    // 2 out of 4 are far away = 50% close.
    let expected = [1.0f32, 2.0, 3.0, 4.0];
    let actual = [1.0, 200.0, 3.0, 400.0];

    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::PercentClose {
            threshold: 0.1,
            percent: 90.0,
        },
    )
    .expect("comparison should succeed");
    assert!(!result.passed, "50% close < 90% threshold should fail");
    assert_eq!(result.num_mismatches, 2);
}

#[test]
fn test_percent_close_all_within_threshold() {
    let expected = [1.0f32, 2.0, 3.0];
    let actual = [1.001, 2.001, 3.001];

    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::PercentClose {
            threshold: 0.01,
            percent: 100.0,
        },
    )
    .expect("comparison should succeed");
    assert!(result.passed);
    assert_eq!(result.num_mismatches, 0);
}

// ---- Error cases ----

#[test]
fn test_length_mismatch_returns_error() {
    let err = compare_with_tolerance(
        &[1.0f32, 2.0],
        &[1.0f32],
        &ToleranceStrategy::Absolute { atol: 1.0 },
    )
    .expect_err("should fail on length mismatch");
    assert!(matches!(err, ReftestError::DataLengthMismatch { .. }));
}

#[test]
fn test_empty_slices_returns_error() {
    let err = compare_with_tolerance(&[], &[], &ToleranceStrategy::Absolute { atol: 1.0 })
        .expect_err("should fail on empty slices");
    assert!(matches!(err, ReftestError::EmptyTensor(_)));
}

// ---- IEEE 754 edge cases ----

#[test]
fn test_infinity_in_actual_fails_all_strategies() {
    let actual = [f32::INFINITY];
    let expected = [1.0f32];

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
        let result = compare_with_tolerance(&actual, &expected, strategy)
            .expect("comparison should succeed");
        assert!(
            !result.passed,
            "Infinity should fail for strategy {strategy:?}"
        );
    }
}

#[test]
fn test_nan_in_expected_fails_all_strategies() {
    let actual = [1.0f32];
    let expected = [f32::NAN];

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
        let result = compare_with_tolerance(&actual, &expected, strategy)
            .expect("comparison should succeed");
        assert!(
            !result.passed,
            "NaN in expected should fail for strategy {strategy:?}"
        );
    }
}

// ---- Worst index tracking ----

#[test]
fn test_worst_index_points_to_largest_diff() {
    let expected = [1.0f32, 2.0, 3.0, 4.0];
    let actual = [1.0, 2.0, 3.5, 4.0]; // index 2 has largest diff

    let result = compare_with_tolerance(
        &actual,
        &expected,
        &ToleranceStrategy::Absolute { atol: 1.0 },
    )
    .expect("comparison should succeed");
    assert_eq!(result.worst_index, 2);
    assert!((result.max_diff - 0.5).abs() < 1e-6);
}

// ---- ULP distance helper ----

#[test]
fn test_ulp_distance_function() {
    assert_eq!(ulp_distance(1.0, 1.0), 0);
    assert_eq!(ulp_distance(0.0, -0.0), 0);
    assert_eq!(ulp_distance(f32::NAN, 1.0), u32::MAX);
    assert_eq!(ulp_distance(1.0, f32::NAN), u32::MAX);
    assert_eq!(ulp_distance(f32::NAN, f32::NAN), u32::MAX);

    // Adjacent floats are 1 ULP apart.
    let a = 1.0f32;
    let b = f32::from_bits(a.to_bits() + 1);
    assert_eq!(ulp_distance(a, b), 1);
    assert_eq!(ulp_distance(b, a), 1); // symmetric
}

// ---- Subnormal values (very small near zero) ----

#[test]
fn test_subnormal_absolute_within_tolerance() {
    // f32::MIN_POSITIVE is the smallest normal; values below it are subnormal.
    let a = [f32::MIN_POSITIVE / 2.0];
    let b = [f32::MIN_POSITIVE / 4.0];
    let diff = f64::from(a[0]) - f64::from(b[0]);
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::Absolute {
            atol: diff.abs() + 1e-50,
        },
    )
    .expect("comparison should succeed");
    assert!(
        result.passed,
        "subnormal difference should be within tolerance"
    );
}

#[test]
fn test_subnormal_ulp_adjacent() {
    // Two adjacent subnormal floats should be 1 ULP apart.
    let tiny = f32::from_bits(1); // smallest positive subnormal
    let next_tiny = f32::from_bits(2); // next subnormal
    assert_eq!(ulp_distance(tiny, next_tiny), 1);

    let result = compare_with_tolerance(
        &[tiny],
        &[next_tiny],
        &ToleranceStrategy::ULP { max_ulps: 1 },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "adjacent subnormals should be within 1 ULP");
}

#[test]
fn test_subnormal_vs_zero() {
    // Smallest subnormal vs zero is 1 ULP apart.
    let tiny = f32::from_bits(1);
    assert_eq!(ulp_distance(tiny, 0.0), 1);
    assert_eq!(ulp_distance(0.0, tiny), 1);
}

#[test]
fn test_subnormal_relative_with_epsilon_floor() {
    // Near-zero subnormals use the epsilon floor (1e-8) in denominator.
    let a = [f32::from_bits(100)]; // small subnormal
    let b = [f32::from_bits(101)]; // adjacent subnormal
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Relative { rtol: 1.0 })
        .expect("comparison should succeed");
    assert!(
        result.passed,
        "subnormal relative comparison should use epsilon floor"
    );
}

// ---- Negative values ----

#[test]
fn test_negative_values_absolute_within_tolerance() {
    let a = [-1.0f32, -2.0, -3.0];
    let b = [-1.0001, -2.0001, -3.0001];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1e-3 })
        .expect("comparison should succeed");
    assert!(result.passed);
    assert_eq!(result.num_mismatches, 0);
}

#[test]
fn test_negative_values_relative() {
    let a = [-100.0f32];
    let b = [-100.1f32];
    // Relative diff = 0.1 / 100.1 ~ 0.001
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Relative { rtol: 0.01 })
        .expect("comparison should succeed");
    assert!(
        result.passed,
        "small relative difference on negative values should pass"
    );
}

#[test]
fn test_opposite_signs_large_difference() {
    let a = [1.0f32];
    let b = [-1.0f32];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1.0 })
        .expect("comparison should succeed");
    // |1.0 - (-1.0)| = 2.0 > 1.0
    assert!(!result.passed);
    assert_eq!(result.num_mismatches, 1);
}

#[test]
fn test_negative_values_mixed_tolerance() {
    let a = [-500.0f32];
    let b = [-500.5f32];
    // |a - b| = 0.5, atol + rtol * |b| = 0.0 + 0.01 * 500.5 = 5.005
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::Mixed {
            atol: 0.0,
            rtol: 0.01,
        },
    )
    .expect("comparison should succeed");
    assert!(
        result.passed,
        "mixed tolerance should work for negative values"
    );
}

#[test]
fn test_negative_values_ulp() {
    let a = -1.0f32;
    let b = f32::from_bits(a.to_bits() + 1); // move 1 ULP in negative direction
    let result = compare_with_tolerance(&[a], &[b], &ToleranceStrategy::ULP { max_ulps: 1 })
        .expect("comparison should succeed");
    assert!(
        result.passed,
        "adjacent negative floats should be within 1 ULP"
    );
}

// ---- NaN handling (NaN != NaN) ----

#[test]
fn test_nan_ne_nan_all_strategies() {
    // IEEE 754: NaN != NaN. All strategies must reject NaN vs NaN.
    let a = [f32::NAN];
    let b = [f32::NAN];
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
        let result = compare_with_tolerance(&a, &b, strategy).expect("comparison should succeed");
        assert!(
            !result.passed,
            "NaN != NaN: strategy {strategy:?} should not pass"
        );
    }
}

#[test]
fn test_nan_in_mixed_array_only_nan_fails() {
    // Only the NaN element should be a mismatch; finite elements should pass.
    let a = [1.0f32, f32::NAN, 3.0];
    let b = [1.0f32, 2.0, 3.0];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1e-3 })
        .expect("comparison should succeed");
    assert!(!result.passed);
    assert_eq!(
        result.num_mismatches, 1,
        "only the NaN element should mismatch"
    );
    assert!(result.max_diff.is_infinite(), "NaN diff should be infinite");
}

// ---- Infinity edge cases ----

#[test]
fn test_matching_infinities_still_fail() {
    // Even matching infinities fail -- non-finite never passes.
    let a = [f32::INFINITY];
    let b = [f32::INFINITY];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: f64::MAX })
        .expect("comparison should succeed");
    assert!(
        !result.passed,
        "+inf == +inf should still fail (non-finite)"
    );

    let a2 = [f32::NEG_INFINITY];
    let b2 = [f32::NEG_INFINITY];
    let result2 = compare_with_tolerance(&a2, &b2, &ToleranceStrategy::Absolute { atol: f64::MAX })
        .expect("comparison should succeed");
    assert!(
        !result2.passed,
        "-inf == -inf should still fail (non-finite)"
    );
}

#[test]
fn test_mixed_finite_and_infinite_elements() {
    let a = [1.0f32, f32::INFINITY, 3.0];
    let b = [1.0f32, 2.0, 3.0];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1.0 })
        .expect("comparison should succeed");
    assert!(!result.passed);
    assert_eq!(result.num_mismatches, 1);
    assert!(result.max_diff.is_infinite());
}

// ---- Single element comparison ----

#[test]
fn test_single_element_pass() {
    let a = [42.0f32];
    let b = [42.0f32];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 0.0 })
        .expect("comparison should succeed");
    assert!(result.passed);
    assert_eq!(result.max_diff, 0.0);
    assert_eq!(result.mean_diff, 0.0);
    assert_eq!(result.worst_index, 0);
    assert_eq!(result.num_mismatches, 0);
}

#[test]
fn test_single_element_fail() {
    let a = [1.0f32];
    let b = [2.0f32];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 0.5 })
        .expect("comparison should succeed");
    assert!(!result.passed);
    assert!((result.max_diff - 1.0).abs() < 1e-10);
    assert_eq!(result.num_mismatches, 1);
    assert_eq!(result.worst_index, 0);
}

#[test]
fn test_single_element_all_strategies() {
    let a = [3.14f32];
    let b = [3.14f32];
    for strategy in &[
        ToleranceStrategy::Absolute { atol: 0.0 },
        ToleranceStrategy::Relative { rtol: 0.0 },
        ToleranceStrategy::Mixed {
            atol: 0.0,
            rtol: 0.0,
        },
        ToleranceStrategy::ULP { max_ulps: 0 },
        ToleranceStrategy::PercentClose {
            threshold: 0.0,
            percent: 100.0,
        },
    ] {
        let result = compare_with_tolerance(&a, &b, strategy).expect("comparison should succeed");
        assert!(
            result.passed,
            "single identical element should pass {strategy:?}"
        );
    }
}

// ---- Large array comparison ----

#[test]
fn test_large_array_uniform_perturbation_passes() {
    let n = 10_000;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = a.iter().map(|&x| x + 1e-5).collect();
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1e-4 })
        .expect("comparison should succeed");
    assert!(result.passed);
    assert_eq!(result.num_mismatches, 0);
    assert!(result.max_diff < 2e-5, "max_diff should be ~1e-5");
}

#[test]
fn test_large_array_single_outlier() {
    let n = 10_000;
    let mut a: Vec<f32> = vec![1.0; n];
    let b: Vec<f32> = vec![1.0; n];
    a[5000] = 100.0; // Single outlier.
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1e-3 })
        .expect("comparison should succeed");
    assert!(!result.passed);
    assert_eq!(result.num_mismatches, 1);
    assert_eq!(result.worst_index, 5000);
    assert!((result.max_diff - 99.0).abs() < 1e-3);
}

#[test]
fn test_large_array_percent_close_with_scattered_outliers() {
    let n = 1_000;
    let a: Vec<f32> = vec![1.0; n];
    let mut b: Vec<f32> = vec![1.0; n];
    // Make 5% of elements outliers (every 20th).
    for i in (0..n).step_by(20) {
        b[i] = 100.0;
    }
    // 950/1000 = 95% within threshold. Require 90%.
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::PercentClose {
            threshold: 0.01,
            percent: 90.0,
        },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "95% close >= 90% required");
}

#[test]
fn test_large_array_mean_diff_accuracy() {
    // All elements differ by exactly 0.1.
    let n = 5_000;
    let a: Vec<f32> = vec![0.0; n];
    let b: Vec<f32> = vec![0.1; n];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1.0 })
        .expect("comparison should succeed");
    assert!(result.passed);
    // mean_diff should be very close to 0.1.
    let expected_mean = 0.1_f64;
    assert!(
        (result.mean_diff - expected_mean).abs() < 1e-6,
        "mean_diff {} should be ~0.1",
        result.mean_diff
    );
}

// ---- Metric accuracy ----

#[test]
fn test_metrics_max_mean_worst_index() {
    let a = [0.0f32, 0.0, 0.0, 0.0, 0.0];
    let b = [0.1f32, 0.2, 0.3, 0.4, 0.5];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1.0 })
        .expect("comparison should succeed");
    assert!(result.passed);
    assert!(
        (result.max_diff - 0.5).abs() < 1e-6,
        "max_diff should be 0.5"
    );
    let expected_mean = (0.1 + 0.2 + 0.3 + 0.4 + 0.5) / 5.0;
    assert!(
        (result.mean_diff - expected_mean).abs() < 1e-6,
        "mean_diff should be 0.3"
    );
    assert_eq!(result.worst_index, 4, "worst_index should be element 4");
}

// ---- Strategy trait object coverage ----

#[test]
fn test_tolerance_strategy_debug_clone_eq() {
    let s1 = ToleranceStrategy::Absolute { atol: 1e-5 };
    let s2 = s1.clone();
    assert_eq!(s1, s2);
    let debug = format!("{s1:?}");
    assert!(
        debug.contains("Absolute"),
        "debug should contain variant name"
    );
}

#[test]
fn test_comparison_result_debug_clone() {
    let a = [1.0f32];
    let b = [1.0f32];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1e-3 })
        .expect("comparison should succeed");
    let cloned = result.clone();
    assert_eq!(cloned.passed, result.passed);
    assert_eq!(cloned.max_diff, result.max_diff);
    assert_eq!(cloned.mean_diff, result.mean_diff);
    assert_eq!(cloned.num_mismatches, result.num_mismatches);
    assert_eq!(cloned.worst_index, result.worst_index);
    let debug = format!("{result:?}");
    assert!(debug.contains("passed"), "debug should contain 'passed'");
}

// ---- PercentClose boundary precision ----

#[test]
fn test_percent_close_exact_boundary_percentage() {
    // 4 out of 5 = 80.0%, require exactly 80.0%.
    let a = [0.0f32; 5];
    let b = [0.0f32, 0.0, 0.0, 0.0, 1.0]; // 1 outlier
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::PercentClose {
            threshold: 0.5,
            percent: 80.0,
        },
    )
    .expect("comparison should succeed");
    assert!(result.passed, "80% close == 80% required should pass");
}

#[test]
fn test_percent_close_just_below_boundary() {
    // 3 out of 5 = 60%, require 61%.
    let a = [0.0f32; 5];
    let b = [0.0f32, 0.0, 0.0, 1.0, 1.0]; // 2 outliers
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::PercentClose {
            threshold: 0.5,
            percent: 61.0,
        },
    )
    .expect("comparison should succeed");
    assert!(!result.passed, "60% close < 61% required should fail");
}
