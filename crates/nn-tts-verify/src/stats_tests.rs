// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_welch_t_test_identical_samples() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (t, _df, p) = welch_t_test(&a, &b).unwrap();
    assert!(
        t.abs() < 1e-10,
        "t should be ~0 for identical samples, got {t}"
    );
    assert!(p > 0.99, "p should be ~1.0 for identical samples, got {p}");
}

#[test]
fn test_welch_t_test_different_means() {
    // Group A: mean ~2, Group B: mean ~10 — clearly different
    let a = vec![1.0, 2.0, 3.0, 2.0, 1.5, 2.5, 1.8, 2.2, 2.1, 1.9];
    let b = vec![9.0, 10.0, 11.0, 10.0, 9.5, 10.5, 9.8, 10.2, 10.1, 9.9];
    let (t, _df, p) = welch_t_test(&a, &b).unwrap();
    assert!(
        t.abs() > 2.0,
        "t should be large for different means, got {t}"
    );
    assert!(
        p < 0.05,
        "p should be < 0.05 for clearly different distributions, got {p}"
    );
}

#[test]
fn test_welch_t_test_min_samples() {
    let a = vec![1.0];
    let b = vec![2.0, 3.0];
    let result = welch_t_test(&a, &b);
    assert!(result.is_err(), "Should reject sample with < 2 elements");
}

#[test]
fn test_welch_t_test_zero_variance() {
    let a = vec![5.0, 5.0, 5.0];
    let b = vec![5.0, 5.0, 5.0];
    let (t, _df, p) = welch_t_test(&a, &b).unwrap();
    assert!(
        t.abs() < 1e-10,
        "t should be 0 for zero-variance identical groups"
    );
    assert!((p - 1.0).abs() < 1e-10, "p should be 1.0");
}

#[test]
fn test_cohens_d_zero_for_identical() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let d = cohens_d(&a, &b).unwrap();
    assert!(
        d.abs() < 1e-10,
        "d should be ~0 for identical samples, got {d}"
    );
}

#[test]
fn test_cohens_d_large_effect() {
    // Group A: mean=0, std~1; Group B: mean=2, std~1
    // Expected Cohen's d ≈ 2.0
    let a = vec![-0.5, 0.0, 0.5, -0.3, 0.3, -0.1, 0.1, -0.4, 0.4, 0.0];
    let b = vec![1.5, 2.0, 2.5, 1.7, 2.3, 1.9, 2.1, 1.6, 2.4, 2.0];
    let d = cohens_d(&a, &b).unwrap();
    assert!(
        d.abs() > 1.5,
        "d should be large (>1.5) for 2σ shift, got {d}"
    );
    assert!(d < 0.0, "d should be negative (group_a < group_b), got {d}");
}

#[test]
fn test_cohens_d_zero_variance() {
    let a = vec![3.0, 3.0, 3.0];
    let b = vec![3.0, 3.0, 3.0];
    let d = cohens_d(&a, &b).unwrap();
    assert_eq!(d, 0.0, "d should be 0 when both groups have zero variance");
}

#[test]
fn test_holm_bonferroni_correction() {
    // 5 p-values: one is very significant, rest are not
    let raw = vec![0.001, 0.20, 0.30, 0.50, 0.80];
    let adjusted = holm_bonferroni(&raw).unwrap();

    // The smallest p-value (0.001) is multiplied by 5 → 0.005
    assert!(
        adjusted[0] < 0.05,
        "First (smallest) should remain significant after correction"
    );

    // Non-significant p-values should be adjusted upward
    for &p in &adjusted[1..] {
        assert!(p >= raw[1], "Adjusted p-values should be >= raw values");
    }

    // All adjusted values should be <= 1.0
    for &p in &adjusted {
        assert!(p <= 1.0, "Adjusted p-values should be capped at 1.0");
    }

    // Monotonicity: adjusted values (in sorted-raw order) should be non-decreasing
    let mut indexed: Vec<(usize, f64)> = raw.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    let adj_sorted: Vec<f64> = indexed.iter().map(|&(i, _)| adjusted[i]).collect();
    for w in adj_sorted.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-15,
            "Adjusted p-values should be non-decreasing in sorted order"
        );
    }
}

#[test]
fn test_holm_bonferroni_empty() {
    let result = holm_bonferroni(&[]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_holm_bonferroni_single() {
    let adjusted = holm_bonferroni(&[0.03]).unwrap();
    assert_eq!(adjusted.len(), 1);
    assert!(
        (adjusted[0] - 0.03).abs() < 1e-15,
        "Single p-value should be unchanged"
    );
}

#[test]
fn test_student_t_cdf_symmetry() {
    // CDF should be symmetric: P(T <= -t) = 1 - P(T <= t)
    let df = 10.0;
    let t_val = 2.0;
    let left = student_t_cdf(-t_val, df);
    let right = student_t_cdf(t_val, df);
    assert!(
        (left + right - 1.0).abs() < 1e-6,
        "CDF symmetry: P(-t) + P(t) should ≈ 1.0, got {} + {} = {}",
        left,
        right,
        left + right,
    );
}

#[test]
fn test_student_t_cdf_center() {
    // CDF at t=0 should be 0.5
    let p = student_t_cdf(0.0, 10.0);
    assert!((p - 0.5).abs() < 1e-10, "CDF at t=0 should be 0.5, got {p}");
}

#[test]
fn test_student_t_cdf_known_value() {
    // For df=∞, Student's t → standard normal.
    // P(T <= 1.96) ≈ 0.975 for large df.
    let p = student_t_cdf(1.96, 1000.0);
    assert!(
        (p - 0.975).abs() < 0.005,
        "CDF(1.96, df=1000) should ≈ 0.975, got {p}"
    );
}

// ---------------------------------------------------------------------------
// NaN/Inf rejection tests
// ---------------------------------------------------------------------------

#[test]
fn test_welch_t_test_rejects_nan() {
    let a = vec![1.0, f64::NAN, 3.0];
    let b = vec![1.0, 2.0, 3.0];
    let err = welch_t_test(&a, &b).unwrap_err();
    assert!(
        matches!(err, TtsVerifyError::NonFiniteInput { .. }),
        "should reject NaN input: {err}"
    );
}

#[test]
fn test_welch_t_test_rejects_inf() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, f64::INFINITY, 3.0];
    let err = welch_t_test(&a, &b).unwrap_err();
    assert!(
        matches!(err, TtsVerifyError::NonFiniteInput { .. }),
        "should reject Inf input: {err}"
    );
}

#[test]
fn test_cohens_d_rejects_nan() {
    let a = vec![1.0, f64::NAN, 3.0];
    let b = vec![1.0, 2.0, 3.0];
    let err = cohens_d(&a, &b).unwrap_err();
    assert!(
        matches!(err, TtsVerifyError::NonFiniteInput { .. }),
        "should reject NaN input: {err}"
    );
}

#[test]
fn test_cohens_d_rejects_inf() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![f64::NEG_INFINITY, 2.0, 3.0];
    let err = cohens_d(&a, &b).unwrap_err();
    assert!(
        matches!(err, TtsVerifyError::NonFiniteInput { .. }),
        "should reject Inf input: {err}"
    );
}

#[test]
fn test_holm_bonferroni_rejects_nan() {
    let raw = vec![0.01, f64::NAN, 0.05];
    let err = holm_bonferroni(&raw).unwrap_err();
    assert!(
        matches!(err, TtsVerifyError::NonFiniteInput { .. }),
        "should reject NaN p-value: {err}"
    );
}

#[test]
fn test_holm_bonferroni_rejects_inf() {
    let raw = vec![0.01, f64::INFINITY, 0.05];
    let err = holm_bonferroni(&raw).unwrap_err();
    assert!(
        matches!(err, TtsVerifyError::NonFiniteInput { .. }),
        "should reject Inf p-value: {err}"
    );
}

// -- fold_min_propagate_nan tests -------------------------------------------

#[test]
fn test_fold_min_propagate_nan_normal() {
    let vals = vec![3.0, 1.0, 2.0];
    assert!((fold_min_propagate_nan(vals.into_iter(), f64::INFINITY) - 1.0).abs() < 1e-15);
}

#[test]
fn test_fold_min_propagate_nan_empty() {
    let vals: Vec<f64> = vec![];
    assert_eq!(
        fold_min_propagate_nan(vals.into_iter(), f64::INFINITY),
        f64::INFINITY
    );
}

#[test]
fn test_fold_min_propagate_nan_with_nan() {
    let vals = vec![3.0, f64::NAN, 1.0];
    assert!(fold_min_propagate_nan(vals.into_iter(), f64::INFINITY).is_nan());
}

#[test]
fn test_fold_min_propagate_nan_all_nan() {
    let vals = vec![f64::NAN, f64::NAN];
    assert!(fold_min_propagate_nan(vals.into_iter(), f64::INFINITY).is_nan());
}

#[test]
fn test_fold_min_propagate_nan_negative() {
    let vals = vec![-3.0, -1.0, -5.0];
    assert!((fold_min_propagate_nan(vals.into_iter(), f64::INFINITY) - (-5.0)).abs() < 1e-15);
}
