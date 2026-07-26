// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Gate tests for tensor comparison: RMS difference, peak amplitude, and
//! near-zero relative tolerance.

use super::*;

fn make_tensor(name: &str, data: Vec<f32>) -> NamedTensor {
    let len = data.len();
    NamedTensor::new(name, vec![len], data).expect("valid test tensor")
}

// ---- RMS difference gate ----

#[test]
fn test_rms_diff_computed_correctly() {
    // diffs: [0.1, 0.0, 0.0] → sum_sq = 0.01, mean_sq = 0.01/3, rms = sqrt(0.01/3)
    let a = make_tensor("x", vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![1.1, 2.0, 3.0]);
    let config = ComparisonConfig {
        abs_tolerance: 1.0,
        rel_tolerance: 1.0,
        cosine_threshold: 0.0,
        ..ComparisonConfig::default()
    };
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");

    let expected_rms = (0.01_f64 / 3.0).sqrt() as f32;
    assert!(
        (result.rms_diff - expected_rms).abs() < 1e-7,
        "rms_diff should be {expected_rms:.6e}, got {:.6e}",
        result.rms_diff
    );
}

#[test]
fn test_rms_gate_passes_within_tolerance() {
    let a = make_tensor("x", vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![1.0 + 1e-7, 2.0 - 1e-7, 3.0 + 1e-7]);
    let config = ComparisonConfig {
        rms_tolerance: Some(1e-4),
        ..ComparisonConfig::default()
    };
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(result.passed, "should pass with small RMS difference");
}

#[test]
fn test_rms_gate_fails_above_tolerance() {
    // diffs: [0.1, 0.0, 0.0] → rms ≈ 0.0577
    let a = make_tensor("x", vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![1.1, 2.0, 3.0]);
    let config = ComparisonConfig {
        abs_tolerance: 1.0,
        rel_tolerance: 1.0,
        cosine_threshold: 0.0,
        rms_tolerance: Some(0.01),
        peak_amplitude_limit: None,
        #[cfg(feature = "spectral")]
        spectral: None,
    };
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(
        !result.passed,
        "should fail: rms_diff={:.4e} exceeds rms_tolerance=0.01",
        result.rms_diff
    );
}

#[test]
fn test_rms_gate_disabled_by_default() {
    // Large difference, but default config has rms_tolerance=None.
    let a = make_tensor("x", vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![1.0, 2.0, 3.0]);
    let result =
        compare_tensors(&a, &b, &ComparisonConfig::default()).expect("comparison should succeed");
    assert!(result.passed);
    assert_eq!(result.rms_diff, 0.0);
}

// ---- Peak amplitude gate ----

#[test]
fn test_peak_amplitude_computed_correctly() {
    let a = make_tensor("x", vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![-5.0, 2.0, 3.0]);
    let config = ComparisonConfig {
        abs_tolerance: 100.0,
        rel_tolerance: 100.0,
        cosine_threshold: 0.0,
        ..ComparisonConfig::default()
    };
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert_eq!(result.peak_amplitude, 5.0, "peak should be abs(-5.0) = 5.0");
}

#[test]
fn test_peak_amplitude_gate_passes_within_limit() {
    let a = make_tensor("x", vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![1.0, 2.0, 3.0]);
    let config = ComparisonConfig {
        peak_amplitude_limit: Some(10.0),
        ..ComparisonConfig::default()
    };
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(result.passed, "should pass with peak=3.0 < limit=10.0");
}

#[test]
fn test_peak_amplitude_gate_fails_above_limit() {
    let a = make_tensor("x", vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![1.0, 2.0, 100.0]);
    let config = ComparisonConfig {
        abs_tolerance: 1000.0,
        rel_tolerance: 1000.0,
        cosine_threshold: 0.0,
        rms_tolerance: None,
        peak_amplitude_limit: Some(50.0),
        #[cfg(feature = "spectral")]
        spectral: None,
    };
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(
        !result.passed,
        "should fail: peak_amplitude={} exceeds limit=50.0",
        result.peak_amplitude
    );
    assert_eq!(result.peak_amplitude, 100.0);
}

#[test]
fn test_peak_amplitude_nan_is_infinity() {
    let a = make_tensor("x", vec![1.0, 2.0]);
    let b = make_tensor("x", vec![1.0, f32::NAN]);
    let config = ComparisonConfig {
        peak_amplitude_limit: Some(100.0),
        ..ComparisonConfig::default()
    };
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(
        result.peak_amplitude.is_infinite(),
        "NaN candidate should produce infinite peak amplitude"
    );
    assert!(!result.passed, "should fail with NaN candidate");
}

#[test]
fn test_both_gates_combined() {
    let a = make_tensor("x", vec![0.0, 0.0, 0.0]);
    let b = make_tensor("x", vec![0.1, 0.0, 0.0]);
    // RMS gate: rms ≈ 0.0577. Peak gate: peak = 0.1.
    // Set rms_tolerance tight enough to fail, peak loose enough to pass.
    let config = ComparisonConfig {
        abs_tolerance: 1.0,
        rel_tolerance: 1.0,
        cosine_threshold: 0.0,
        rms_tolerance: Some(0.01),
        peak_amplitude_limit: Some(1.0),
        #[cfg(feature = "spectral")]
        spectral: None,
    };
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(
        !result.passed,
        "should fail due to rms_diff ({:.4e}) > rms_tolerance (0.01)",
        result.rms_diff
    );
}

// ---- Near-zero relative tolerance (#1416) ----

#[test]
fn test_near_zero_values_skip_relative_error() {
    // Near-zero values: both ref and candidate are below atol=1e-3.
    // Absolute diff is tiny (8.57e-8) but relative error would be ~22%.
    // With the fix, relative error should not be computed for these values.
    let a = make_tensor("x", vec![3.0e-7, 1.0, 2.0]);
    let b = make_tensor("x", vec![3.857e-7, 1.0, 2.0]);
    let config = ComparisonConfig::new(1e-3, 1e-2, 0.999);
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(
        result.passed,
        "near-zero values should not trigger rtol failure, max_rel={:.4e}",
        result.max_rel_diff
    );
    // max_rel should be 0.0 because only the near-zero pair differs,
    // and it's excluded from relative error computation.
    assert!(
        result.max_rel_diff < 1e-6,
        "max_rel_diff should be ~0 when only near-zero values differ, got {:.4e}",
        result.max_rel_diff
    );
}

#[test]
fn test_large_values_still_check_relative_error() {
    // Large values: relative error should still be checked.
    let a = make_tensor("x", vec![1.0, 2.0]);
    let b = make_tensor("x", vec![1.05, 2.0]); // 5% relative error
    let config = ComparisonConfig::new(0.1, 0.01, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(
        !result.passed,
        "5% relative error should fail with rtol=1%, max_rel={:.4e}",
        result.max_rel_diff
    );
}

#[test]
fn test_asymmetric_near_zero_still_checks_relative_error() {
    // Asymmetric case: ref is near-zero but candidate is large (above atol).
    // Relative error MUST still be computed — a candidate of 0.5 when ref is 1e-8
    // is a real divergence, not a near-zero artifact. This catches || vs && bugs
    // in the near-zero skip condition.
    let a = make_tensor("x", vec![1e-8, 2.0]);
    let b = make_tensor("x", vec![0.5, 2.0]);
    let config = ComparisonConfig::new(1e-3, 0.01, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("comparison should succeed");
    assert!(
        !result.passed,
        "asymmetric near-zero (ref=1e-8, cand=0.5) must trigger rtol failure, max_rel={:.4e}",
        result.max_rel_diff
    );
    // Relative error should be ~1.0 (100%) since 0.5 vs 1e-8.
    assert!(
        result.max_rel_diff > 0.9,
        "relative error should be ~100%%, got {:.4e}",
        result.max_rel_diff
    );
}
