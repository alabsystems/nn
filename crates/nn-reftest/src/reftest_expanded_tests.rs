// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Expanded tests for nn-reftest: tolerance strategies, edge cases, tensor
//! comparison, float precision, batch comparison, and report generation.

use crate::compare::{compare_tensors, compare_traces, ComparisonConfig, DivergenceReport};
use crate::error::ReftestError;
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

// ===========================================================================
// 1. Tolerance strategy: machine-epsilon-level precision
// ===========================================================================

#[test]
fn test_absolute_at_f32_epsilon_boundary() {
    // f32::EPSILON ~ 1.19e-7. Two values differing by exactly one epsilon at 1.0.
    let a = [1.0f32];
    let b = [1.0f32 + f32::EPSILON];
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::Absolute {
            atol: f64::from(f32::EPSILON),
        },
    )
    .expect("should succeed");
    assert!(
        result.passed,
        "diff of exactly f32::EPSILON should pass atol=EPSILON"
    );
}

#[test]
fn test_absolute_just_above_f32_epsilon_fails() {
    let a = [1.0f32];
    let b = [1.0f32 + 2.0 * f32::EPSILON];
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::Absolute {
            atol: f64::from(f32::EPSILON),
        },
    )
    .expect("should succeed");
    assert!(!result.passed, "diff of 2*EPSILON should fail atol=EPSILON");
}

#[test]
fn test_relative_at_machine_epsilon_scale() {
    // For value=1.0, f32::EPSILON is the smallest relative perturbation.
    let a = [1.0f32];
    let b = [1.0f32 + f32::EPSILON];
    // Relative error = EPSILON / max(1.0, 1.0+eps, 1e-8) ~ EPSILON.
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::Relative {
            rtol: f64::from(f32::EPSILON) * 2.0,
        },
    )
    .expect("should succeed");
    assert!(
        result.passed,
        "EPSILON perturbation should pass with rtol=2*EPSILON"
    );
}

// ===========================================================================
// 2. Denormal (subnormal) precision
// ===========================================================================

#[test]
fn test_denormal_absolute_exact_diff() {
    let smallest = f32::from_bits(1); // smallest positive subnormal
    let a = [0.0f32];
    let b = [smallest];
    let diff = f64::from(smallest);
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: diff })
        .expect("should succeed");
    assert!(
        result.passed,
        "denormal diff should be within tolerance set to exact diff"
    );
}

#[test]
fn test_denormal_relative_epsilon_floor() {
    // Two denormals: both are tiny, so the epsilon floor (1e-8) dominates the
    // denominator in relative comparison. Diff / 1e-8 should be negligible.
    let a = [f32::from_bits(10)];
    let b = [f32::from_bits(11)];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Relative { rtol: 1.0 })
        .expect("should succeed");
    assert!(result.passed, "adjacent denormals should pass rtol=1.0");
}

#[test]
fn test_denormal_mixed_strategy() {
    let a = [f32::from_bits(100)];
    let b = [f32::from_bits(200)];
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::Mixed {
            atol: 1e-30,
            rtol: 1e-3,
        },
    )
    .expect("should succeed");
    // atol=1e-30 is generous for denormals; rtol*|b| is also tiny.
    // The diff between bits(100) and bits(200) is about 1.4e-43, so atol=1e-30 covers it.
    assert!(
        result.passed,
        "denormals should pass generous mixed tolerance"
    );
}

// ===========================================================================
// 3. Negative zero handling
// ===========================================================================

#[test]
fn test_negative_zero_absolute_zero_tolerance() {
    // -0.0 and +0.0 differ by 0.0 in absolute terms.
    let result = compare_with_tolerance(
        &[-0.0f32],
        &[0.0f32],
        &ToleranceStrategy::Absolute { atol: 0.0 },
    )
    .expect("should succeed");
    assert!(
        result.passed,
        "+0.0 and -0.0 should be equal under absolute tolerance"
    );
    assert_eq!(result.max_diff, 0.0);
}

#[test]
fn test_negative_zero_relative_zero_tolerance() {
    let result = compare_with_tolerance(
        &[-0.0f32],
        &[0.0f32],
        &ToleranceStrategy::Relative { rtol: 0.0 },
    )
    .expect("should succeed");
    assert!(
        result.passed,
        "+0.0 and -0.0 should be equal under relative tolerance"
    );
}

#[test]
fn test_negative_zero_mixed_zero_tolerance() {
    let result = compare_with_tolerance(
        &[-0.0f32],
        &[0.0f32],
        &ToleranceStrategy::Mixed {
            atol: 0.0,
            rtol: 0.0,
        },
    )
    .expect("should succeed");
    assert!(
        result.passed,
        "+0.0 and -0.0 should be equal under mixed tolerance"
    );
}

#[test]
fn test_negative_zero_ulp_distance_is_zero() {
    let result = compare_with_tolerance(
        &[-0.0f32],
        &[0.0f32],
        &ToleranceStrategy::ULP { max_ulps: 0 },
    )
    .expect("should succeed");
    assert!(result.passed, "+0.0 and -0.0 should be 0 ULPs apart");
}

#[test]
fn test_negative_zero_percent_close() {
    let result = compare_with_tolerance(
        &[-0.0f32],
        &[0.0f32],
        &ToleranceStrategy::PercentClose {
            threshold: 0.0,
            percent: 100.0,
        },
    )
    .expect("should succeed");
    assert!(
        result.passed,
        "+0.0 and -0.0 should be equal under PercentClose"
    );
}

// ===========================================================================
// 4. Multi-dimensional tensor comparison
// ===========================================================================

#[test]
fn test_compare_2d_tensor_identical() {
    let a = tensor_nd("weights", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = tensor_nd("weights", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.num_elements, 6);
    assert_eq!(result.shape, vec![2, 3]);
}

#[test]
fn test_compare_3d_tensor_with_perturbation() {
    let data_a: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
    let data_b: Vec<f32> = data_a.iter().map(|&v| v + 1e-7).collect();
    let a = tensor_nd("feat", vec![2, 3, 4], data_a);
    let b = tensor_nd("feat", vec![2, 3, 4], data_b);
    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.num_elements, 24);
}

#[test]
fn test_compare_4d_shape_mismatch() {
    let a = tensor_nd("conv", vec![1, 3, 8, 8], vec![0.0; 192]);
    let b = tensor_nd("conv", vec![1, 3, 8, 9], vec![0.0; 216]);
    let err = compare_tensors(&a, &b, &ComparisonConfig::default())
        .expect_err("should fail on shape mismatch");
    match err {
        ReftestError::ShapeMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, vec![1, 3, 8, 8]);
            assert_eq!(actual, vec![1, 3, 8, 9]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn test_compare_scalar_tensors() {
    // Scalar = shape [] with 1 element.
    let a = tensor_nd("loss", vec![], vec![0.5]);
    let b = tensor_nd("loss", vec![], vec![0.5]);
    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.num_elements, 1);
    assert!(result.shape.is_empty());
}

// ===========================================================================
// 5. Batch trace comparison: multiple failures
// ===========================================================================

#[test]
fn test_trace_multiple_failures_reports_first() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace.checkpoint("ok", &[1.0, 2.0], &[2]).unwrap();
    ref_trace.checkpoint("bad1", &[10.0], &[1]).unwrap();
    ref_trace.checkpoint("bad2", &[20.0], &[1]).unwrap();

    let mut cand_trace = ReferenceTrace::new();
    cand_trace.checkpoint("ok", &[1.0, 2.0], &[2]).unwrap();
    cand_trace.checkpoint("bad1", &[999.0], &[1]).unwrap();
    cand_trace.checkpoint("bad2", &[888.0], &[1]).unwrap();

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("should succeed");
    assert!(!report.all_passed);
    assert_eq!(
        report.first_failure,
        Some(1),
        "first failure should be at index 1"
    );
    // All layers are still compared.
    assert_eq!(report.layers.len(), 3);
    assert!(report.layers[0].passed);
    assert!(!report.layers[1].passed);
    assert!(!report.layers[2].passed);
}

#[test]
fn test_trace_all_layers_fail() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace.checkpoint("a", &[1.0], &[1]).unwrap();
    ref_trace.checkpoint("b", &[2.0], &[1]).unwrap();

    let mut cand_trace = ReferenceTrace::new();
    cand_trace.checkpoint("a", &[100.0], &[1]).unwrap();
    cand_trace.checkpoint("b", &[200.0], &[1]).unwrap();

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("should succeed");
    assert!(!report.all_passed);
    assert_eq!(report.first_failure, Some(0));
    assert!(!report.layers[0].passed);
    assert!(!report.layers[1].passed);
}

#[test]
fn test_trace_single_checkpoint_pass() {
    let mut t = ReferenceTrace::new();
    t.checkpoint("only", &[3.14], &[1]).unwrap();

    let report = compare_traces(&t, &t, &ComparisonConfig::default()).expect("should succeed");
    assert!(report.all_passed);
    assert_eq!(report.layers.len(), 1);
}

// ===========================================================================
// 6. Report generation: summary format
// ===========================================================================

#[test]
fn test_report_summary_contains_pass_status_per_layer() {
    let mut ref_trace = ReferenceTrace::new();
    ref_trace.checkpoint("layer0", &[1.0], &[1]).unwrap();
    ref_trace.checkpoint("layer1", &[2.0], &[1]).unwrap();

    let mut cand_trace = ReferenceTrace::new();
    cand_trace.checkpoint("layer0", &[1.0], &[1]).unwrap();
    cand_trace.checkpoint("layer1", &[999.0], &[1]).unwrap();

    let report = compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default())
        .expect("should succeed");
    let summary = report.summary();

    assert!(
        summary.contains("PASS"),
        "summary should contain PASS for layer0"
    );
    assert!(
        summary.contains("FAIL"),
        "summary should contain FAIL for layer1"
    );
    assert!(summary.contains("layer0"), "summary should mention layer0");
    assert!(summary.contains("layer1"), "summary should mention layer1");
    assert!(
        summary.contains("First failure at layer 1"),
        "summary should report first failure index: {summary}"
    );
}

#[test]
fn test_report_summary_empty_trace() {
    let report = DivergenceReport {
        layers: vec![],
        first_failure: None,
        all_passed: true,
    };
    let summary = report.summary();
    assert!(
        summary.contains("All 0 layers passed"),
        "empty trace should report 0 layers passed: {summary}"
    );
}

#[test]
fn test_layer_comparison_display_includes_all_metrics() {
    let a = tensor_1d("test_layer", vec![1.0, 2.0, 3.0]);
    let b = tensor_1d("test_layer", vec![1.001, 2.001, 3.001]);
    let config = ComparisonConfig::relaxed();
    let comp = compare_tensors(&a, &b, &config).expect("should succeed");

    let display = format!("{comp}");
    assert!(
        display.contains("test_layer"),
        "display should contain name"
    );
    assert!(
        display.contains("max_abs="),
        "display should contain max_abs"
    );
    assert!(
        display.contains("mean_abs="),
        "display should contain mean_abs"
    );
    assert!(display.contains("rms="), "display should contain rms");
    assert!(display.contains("cos="), "display should contain cos");
    assert!(
        display.contains("max_rel="),
        "display should contain max_rel"
    );
    assert!(display.contains("peak="), "display should contain peak");
}

// ===========================================================================
// 7. Cross-strategy consistency: identical values
// ===========================================================================

#[test]
fn test_all_strategies_agree_on_identical_large_array() {
    let n = 500;
    let data: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
    let strategies: Vec<ToleranceStrategy> = vec![
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
    ];

    for strategy in &strategies {
        let result = compare_with_tolerance(&data, &data, strategy).expect("should succeed");
        assert!(result.passed, "identical data should pass {strategy:?}");
        assert_eq!(result.num_mismatches, 0, "no mismatches for {strategy:?}");
    }
}

// ===========================================================================
// 8. ULP distance for extreme float ranges
// ===========================================================================

#[test]
fn test_ulp_f32_max_values() {
    let a = [f32::MAX];
    let b = [f32::MAX];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::ULP { max_ulps: 0 })
        .expect("should succeed");
    assert!(result.passed, "identical f32::MAX should be 0 ULPs apart");
}

#[test]
fn test_ulp_f32_max_and_adjacent() {
    let a = [f32::MAX];
    let b = [f32::from_bits(f32::MAX.to_bits() - 1)];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::ULP { max_ulps: 1 })
        .expect("should succeed");
    assert!(
        result.passed,
        "f32::MAX and its predecessor should be 1 ULP apart"
    );
}

#[test]
fn test_ulp_large_negative() {
    let a = [-f32::MAX];
    let b = [-f32::MAX];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::ULP { max_ulps: 0 })
        .expect("should succeed");
    assert!(result.passed, "identical -f32::MAX should be 0 ULPs apart");
}

#[test]
fn test_ulp_f32_min_positive_normal() {
    // f32::MIN_POSITIVE is the smallest normal number.
    let a = [f32::MIN_POSITIVE];
    let b = [f32::from_bits(f32::MIN_POSITIVE.to_bits() + 1)];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::ULP { max_ulps: 1 })
        .expect("should succeed");
    assert!(
        result.passed,
        "adjacent normals at MIN_POSITIVE should be 1 ULP apart"
    );
}

// ===========================================================================
// 9. NaN handling in compare_tensors (not just tolerance)
// ===========================================================================

#[test]
fn test_compare_tensors_both_nan_all_metrics_degraded() {
    let a = tensor_1d("x", vec![f32::NAN]);
    let b = tensor_1d("x", vec![f32::NAN]);
    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");
    assert!(!result.passed);
    assert!(result.max_abs_diff.is_infinite());
    assert!(result.max_rel_diff.is_infinite());
    assert!(result.cosine_similarity.is_nan());
    assert!(result.peak_amplitude.is_infinite());
}

#[test]
fn test_compare_tensors_nan_in_candidate_peak_is_infinite() {
    let a = tensor_1d("x", vec![1.0, 2.0]);
    let b = tensor_1d("x", vec![1.0, f32::NAN]);
    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.peak_amplitude.is_infinite());
}

#[test]
fn test_compare_tensors_inf_peak_amplitude() {
    let a = tensor_1d("x", vec![1.0]);
    let b = tensor_1d("x", vec![f32::INFINITY]);
    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");
    assert!(!result.passed);
    assert!(result.peak_amplitude.is_infinite());
}

// ===========================================================================
// 10. Config: strict vs relaxed boundary
// ===========================================================================

#[test]
fn test_strict_rejects_relaxed_accepts_same_data() {
    let a = tensor_1d("x", vec![1.0, 2.0, 3.0]);
    let b = tensor_1d("x", vec![1.001, 2.001, 3.001]);

    let strict = compare_tensors(&a, &b, &ComparisonConfig::strict()).expect("should succeed");
    let relaxed = compare_tensors(&a, &b, &ComparisonConfig::relaxed()).expect("should succeed");

    assert!(!strict.passed, "strict should reject 1e-3 perturbation");
    assert!(relaxed.passed, "relaxed should accept 1e-3 perturbation");
}

#[test]
fn test_config_with_rms_and_peak_together() {
    let a = tensor_1d("x", vec![0.0, 0.0, 0.0, 0.0]);
    // One element has a moderate deviation.
    let b = tensor_1d("x", vec![0.0, 0.0, 0.0, 0.05]);

    let config = ComparisonConfig::new(1.0, 1.0, 0.0)
        .with_rms_tolerance(0.01)
        .with_peak_amplitude_limit(0.1);

    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    // RMS = sqrt(0.05^2 / 4) = sqrt(0.000625) = 0.025 > 0.01
    assert!(
        !result.passed,
        "should fail due to RMS exceeding 0.01, rms={}",
        result.rms_diff
    );
}

// ===========================================================================
// 11. PercentClose with all failures and all passes
// ===========================================================================

#[test]
fn test_percent_close_zero_percent_always_passes() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [100.0, 200.0, 300.0];
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::PercentClose {
            threshold: 0.0,
            percent: 0.0,
        },
    )
    .expect("should succeed");
    assert!(
        result.passed,
        "0% requirement should always pass regardless of mismatches"
    );
}

#[test]
fn test_percent_close_100_percent_requires_all() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [1.0, 200.0, 3.0]; // 1 outlier out of 3 = 66.7%
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::PercentClose {
            threshold: 0.1,
            percent: 100.0,
        },
    )
    .expect("should succeed");
    assert!(
        !result.passed,
        "100% requirement should fail with any outlier"
    );
}

// ===========================================================================
// 12. Mixed tolerance: NumPy semantics verification
// ===========================================================================

#[test]
fn test_mixed_numpy_semantics_atol_plus_rtol_times_b() {
    // |a - b| <= atol + rtol * |b|
    // a=10.0, b=10.05, atol=0.01, rtol=0.001
    // threshold = 0.01 + 0.001 * 10.05 = 0.02005
    // diff = 0.05 > 0.02005 => fail
    let result = compare_with_tolerance(
        &[10.0f32],
        &[10.05],
        &ToleranceStrategy::Mixed {
            atol: 0.01,
            rtol: 0.001,
        },
    )
    .expect("should succeed");
    assert!(!result.passed, "diff=0.05 > threshold=0.02005 should fail");
}

#[test]
fn test_mixed_numpy_semantics_passes() {
    // a=10.0, b=10.005, atol=0.01, rtol=0.001
    // threshold = 0.01 + 0.001 * 10.005 = 0.020005
    // diff = 0.005 <= 0.020005 => pass
    let result = compare_with_tolerance(
        &[10.0f32],
        &[10.005],
        &ToleranceStrategy::Mixed {
            atol: 0.01,
            rtol: 0.001,
        },
    )
    .expect("should succeed");
    assert!(
        result.passed,
        "diff=0.005 <= threshold=0.020005 should pass"
    );
}

// ===========================================================================
// 13. Relative tolerance: division by zero protection
// ===========================================================================

#[test]
fn test_relative_both_zero_passes() {
    // Both values are zero. denominator = max(0, 0, 1e-8) = 1e-8.
    // diff = 0.0 / 1e-8 = 0.0 <= any rtol.
    let result = compare_with_tolerance(
        &[0.0f32],
        &[0.0],
        &ToleranceStrategy::Relative { rtol: 0.0 },
    )
    .expect("should succeed");
    assert!(result.passed, "0.0 vs 0.0 should pass even with rtol=0.0");
}

#[test]
fn test_relative_near_zero_uses_epsilon_floor() {
    // a=1e-15, b=2e-15. Without epsilon floor, rel = 0.5.
    // With floor: diff=1e-15, denom=max(2e-15, 1e-15, 1e-8)=1e-8.
    // rel = 1e-15 / 1e-8 = 1e-7.
    let result = compare_with_tolerance(
        &[1e-15f32],
        &[2e-15],
        &ToleranceStrategy::Relative { rtol: 1e-6 },
    )
    .expect("should succeed");
    assert!(
        result.passed,
        "epsilon floor should prevent false positives near zero"
    );
}

// ===========================================================================
// 14. Cosine similarity edge cases in compare_tensors
// ===========================================================================

#[test]
fn test_cosine_parallel_vectors_different_magnitude() {
    let a = tensor_1d("x", vec![1.0, 2.0, 3.0]);
    let b = tensor_1d("x", vec![2.0, 4.0, 6.0]);
    let config = ComparisonConfig::new(100.0, 100.0, 0.9999);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        (result.cosine_similarity - 1.0).abs() < 1e-6,
        "parallel vectors should have cosine~1.0, got {}",
        result.cosine_similarity
    );
}

#[test]
fn test_cosine_antiparallel_vectors() {
    let a = tensor_1d("x", vec![1.0, 2.0, 3.0]);
    let b = tensor_1d("x", vec![-1.0, -2.0, -3.0]);
    let config = ComparisonConfig::new(100.0, 100.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        (result.cosine_similarity - (-1.0)).abs() < 1e-6,
        "antiparallel vectors should have cosine~-1.0, got {}",
        result.cosine_similarity
    );
    // Cosine = -1.0 < threshold 0.0 => fail
    assert!(!result.passed);
}

#[test]
fn test_cosine_threshold_exactly_met() {
    // Two identical vectors: cosine = 1.0, threshold = 1.0.
    let a = tensor_1d("x", vec![1.0, 0.0]);
    let config = ComparisonConfig::new(1.0, 1.0, 1.0);
    let result = compare_tensors(&a, &a, &config).expect("should succeed");
    assert!(result.passed, "cosine=1.0 should meet threshold=1.0");
}

// ===========================================================================
// 15. Preset integration: all presets produce valid configs
// ===========================================================================

#[test]
fn test_all_presets_produce_usable_configs() {
    let a = tensor_1d("x", vec![1.0, 2.0, 3.0]);
    for preset in TolerancePreset::ALL {
        let config = preset.to_config();
        let result = compare_tensors(&a, &a, &config);
        assert!(
            result.is_ok(),
            "preset '{}' should produce a usable config",
            preset.name
        );
        let comp = result.unwrap();
        assert!(
            comp.passed,
            "identical tensors should pass under preset '{}'",
            preset.name
        );
    }
}

#[test]
fn test_preset_by_name_all_variants_found() {
    for preset in TolerancePreset::ALL {
        let found = TolerancePreset::by_name(preset.name);
        assert!(
            found.is_some(),
            "by_name should find preset '{}'",
            preset.name
        );
        assert_eq!(found.unwrap(), *preset);
    }
}

// ===========================================================================
// 16. assert_traces_match! macro
// ===========================================================================

#[test]
fn test_assert_traces_match_macro_passes() {
    let mut a = ReferenceTrace::new();
    a.checkpoint("l1", &[1.0, 2.0], &[2]).unwrap();
    let mut b = ReferenceTrace::new();
    b.checkpoint("l1", &[1.0, 2.0], &[2]).unwrap();
    // Should not panic.
    crate::assert_traces_match!(a, b);
}

#[test]
fn test_assert_traces_match_macro_with_epsilon_passes() {
    let mut a = ReferenceTrace::new();
    a.checkpoint("l1", &[1.0], &[1]).unwrap();
    let mut b = ReferenceTrace::new();
    b.checkpoint("l1", &[1.0 + 1e-6], &[1]).unwrap();
    crate::assert_traces_match!(a, b, epsilon = 1e-5);
}

#[test]
fn test_assert_traces_match_macro_with_custom_tolerances_passes() {
    let mut a = ReferenceTrace::new();
    a.checkpoint("l1", &[1.0, 2.0], &[2]).unwrap();
    let mut b = ReferenceTrace::new();
    b.checkpoint("l1", &[1.001, 2.001], &[2]).unwrap();
    crate::assert_traces_match!(a, b, abs = 0.01, rel = 0.01);
}

#[test]
#[should_panic(expected = "Tensor mismatch")]
fn test_assert_traces_match_macro_panics_on_failure() {
    let mut a = ReferenceTrace::new();
    a.checkpoint("l1", &[1.0], &[1]).unwrap();
    let mut b = ReferenceTrace::new();
    b.checkpoint("l1", &[100.0], &[1]).unwrap();
    crate::assert_traces_match!(a, b);
}

#[test]
fn test_assert_traces_match_macro_full_custom() {
    let mut a = ReferenceTrace::new();
    a.checkpoint("l1", &[1.0, 2.0, 3.0], &[3]).unwrap();
    let mut b = ReferenceTrace::new();
    b.checkpoint("l1", &[1.0, 2.0, 3.0], &[3]).unwrap();
    crate::assert_traces_match!(a, b, abs = 1e-5, rel = 1e-4, cos = 0.999);
}

// ===========================================================================
// 17. assert_traces_match_preset! macro
// ===========================================================================

#[test]
fn test_assert_traces_match_preset_macro_passes() {
    let mut a = ReferenceTrace::new();
    a.checkpoint("l1", &[1.0, 2.0], &[2]).unwrap();
    let mut b = ReferenceTrace::new();
    b.checkpoint("l1", &[1.0, 2.0], &[2]).unwrap();
    crate::assert_traces_match_preset!(a, b, TolerancePreset::STANDARD);
}

#[test]
#[should_panic(expected = "Tensor mismatch")]
fn test_assert_traces_match_preset_macro_panics_on_failure() {
    let mut a = ReferenceTrace::new();
    a.checkpoint("l1", &[1.0], &[1]).unwrap();
    let mut b = ReferenceTrace::new();
    b.checkpoint("l1", &[100.0], &[1]).unwrap();
    crate::assert_traces_match_preset!(a, b, TolerancePreset::STRICT);
}

// ===========================================================================
// 18. Error type coverage
// ===========================================================================

#[test]
fn test_error_trace_length_mismatch_message() {
    let mut a = ReferenceTrace::new();
    a.checkpoint("l1", &[1.0], &[1]).unwrap();
    a.checkpoint("l2", &[2.0], &[1]).unwrap();
    let b = ReferenceTrace::new();

    let err = compare_traces(&a, &b, &ComparisonConfig::default()).expect_err("should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("2") && msg.contains("0"),
        "error message should mention counts: {msg}"
    );
}

#[test]
fn test_error_empty_tensor_message() {
    let a = tensor_nd("empty", vec![0], vec![]);
    let b = tensor_nd("empty", vec![0], vec![]);
    let err = compare_tensors(&a, &b, &ComparisonConfig::default())
        .expect_err("should fail on empty tensor");
    let msg = format!("{err}");
    assert!(
        msg.contains("empty"),
        "error message should mention tensor name: {msg}"
    );
}

#[test]
fn test_error_shape_mismatch_message() {
    let a = tensor_nd("w", vec![2, 3], vec![0.0; 6]);
    let b = tensor_nd("w", vec![6], vec![0.0; 6]);
    let err = compare_tensors(&a, &b, &ComparisonConfig::default()).expect_err("should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("[2, 3]") && msg.contains("[6]"),
        "error message should describe shapes: {msg}"
    );
}

// ===========================================================================
// 19. ComparisonResult metrics accuracy
// ===========================================================================

#[test]
fn test_mean_diff_with_alternating_signs() {
    // Diffs: |+0.1|, |-0.1|, |+0.1| = all 0.1. Mean should be exactly 0.1.
    let a = [0.0f32, 0.0, 0.0];
    let b = [0.1, -0.1, 0.1];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1.0 })
        .expect("should succeed");
    assert!(
        (result.mean_diff - 0.1).abs() < 1e-7,
        "mean_diff should be 0.1, got {}",
        result.mean_diff
    );
}

#[test]
fn test_worst_index_at_end() {
    let a = [0.0f32; 5];
    let b = [0.0, 0.0, 0.0, 0.0, 1.0];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 2.0 })
        .expect("should succeed");
    assert_eq!(result.worst_index, 4);
}

#[test]
fn test_worst_index_at_beginning() {
    let a = [0.0f32; 5];
    let b = [1.0, 0.0, 0.0, 0.0, 0.0];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 2.0 })
        .expect("should succeed");
    assert_eq!(result.worst_index, 0);
}

// ===========================================================================
// 20. Tolerance strategy: large values
// ===========================================================================

#[test]
fn test_absolute_fails_on_large_values_small_atol() {
    // Large values with small absolute tolerance.
    let a = [1e6f32];
    let b = [1e6f32 + 1.0];
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 0.5 })
        .expect("should succeed");
    assert!(!result.passed, "diff=1.0 should fail atol=0.5");
}

#[test]
fn test_relative_passes_on_large_values_proportional_diff() {
    // Large values with proportional difference.
    let a = [1e6f32];
    let b = [1e6f32 + 1.0];
    // Relative diff = 1.0 / 1e6 = 1e-6.
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Relative { rtol: 1e-5 })
        .expect("should succeed");
    assert!(result.passed, "1e-6 relative diff should pass rtol=1e-5");
}

// ===========================================================================
// 21. RMS computation for known values
// ===========================================================================

#[test]
fn test_rms_diff_all_same_difference() {
    // All diffs = 0.1 => sum_sq = 4 * 0.01 = 0.04, rms = sqrt(0.01) = 0.1
    let a = tensor_1d("x", vec![0.0, 0.0, 0.0, 0.0]);
    let b = tensor_1d("x", vec![0.1, 0.1, 0.1, 0.1]);
    let config = ComparisonConfig::new(1.0, 1.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        (result.rms_diff - 0.1).abs() < 1e-6,
        "rms should be 0.1, got {}",
        result.rms_diff
    );
}

// ===========================================================================
// 22. Peak amplitude tracking
// ===========================================================================

#[test]
fn test_peak_amplitude_from_candidate_not_reference() {
    // Peak amplitude is from the candidate tensor, not the reference.
    let a = tensor_1d("x", vec![100.0, 200.0]);
    let b = tensor_1d("x", vec![1.0, 2.0]);
    let config = ComparisonConfig::new(1000.0, 1000.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert_eq!(
        result.peak_amplitude, 2.0,
        "peak_amplitude should track candidate, not reference"
    );
}

#[test]
fn test_peak_amplitude_negative_candidate() {
    let a = tensor_1d("x", vec![0.0]);
    let b = tensor_1d("x", vec![-50.0]);
    let config = ComparisonConfig::new(1000.0, 1000.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert_eq!(
        result.peak_amplitude, 50.0,
        "peak_amplitude should be abs(-50.0) = 50.0"
    );
}

// ===========================================================================
// 23. Trace: from_checkpoints and into_checkpoints
// ===========================================================================

#[test]
fn test_from_into_checkpoints_preserves_order() {
    let cps = vec![
        NamedTensor::new("c", vec![1], vec![3.0]).unwrap(),
        NamedTensor::new("a", vec![1], vec![1.0]).unwrap(),
        NamedTensor::new("b", vec![1], vec![2.0]).unwrap(),
    ];
    let trace = ReferenceTrace::from_checkpoints(cps);
    let names: Vec<&str> = trace.names().collect();
    assert_eq!(names, vec!["c", "a", "b"], "order should be preserved");

    let recovered = trace.into_checkpoints();
    assert_eq!(recovered[0].name, "c");
    assert_eq!(recovered[1].name, "a");
    assert_eq!(recovered[2].name, "b");
}

// ===========================================================================
// 24. NamedTensor: boundary shapes
// ===========================================================================

#[test]
fn test_named_tensor_1d_with_one_element() {
    let t = NamedTensor::new("s", vec![1], vec![42.0]).unwrap();
    assert_eq!(t.numel(), 1);
    assert_eq!(t.shape, vec![1]);
}

#[test]
fn test_named_tensor_high_rank_zeros() {
    // 5-D tensor of zeros.
    let t = NamedTensor::new("5d", vec![1, 1, 1, 1, 1], vec![0.0]).unwrap();
    assert_eq!(t.numel(), 1);
    assert_eq!(t.shape.len(), 5);
}

#[test]
fn test_named_tensor_large_flat() {
    let n = 10_000;
    let data: Vec<f32> = vec![1.0; n];
    let t = NamedTensor::new("big", vec![n], data).unwrap();
    assert_eq!(t.numel(), n);
}

// ===========================================================================
// 25. Infinity handling consistency across tolerance and compare
// ===========================================================================

#[test]
fn test_tolerance_neg_infinity_vs_finite_fails() {
    let result = compare_with_tolerance(
        &[f32::NEG_INFINITY],
        &[0.0],
        &ToleranceStrategy::Absolute { atol: f64::MAX },
    )
    .expect("should succeed");
    assert!(!result.passed, "NEG_INFINITY should fail");
    assert!(result.max_diff.is_infinite());
}

#[test]
fn test_tolerance_pos_neg_infinity_fails() {
    let result = compare_with_tolerance(
        &[f32::INFINITY],
        &[f32::NEG_INFINITY],
        &ToleranceStrategy::Absolute { atol: f64::MAX },
    )
    .expect("should succeed");
    assert!(!result.passed, "+inf vs -inf should fail");
}

#[test]
fn test_compare_tensors_mixed_inf_nan() {
    let a = tensor_1d("x", vec![1.0, f32::INFINITY, f32::NAN]);
    let b = tensor_1d("x", vec![1.0, f32::INFINITY, f32::NAN]);
    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");
    assert!(!result.passed, "any non-finite should cause failure");
}

// ===========================================================================
// 26. Compare: default config values are sensible
// ===========================================================================

#[test]
fn test_default_config_values() {
    let config = ComparisonConfig::default();
    assert_eq!(config.abs_tolerance, 1e-5);
    assert_eq!(config.rel_tolerance, 1e-4);
    assert_eq!(config.cosine_threshold, 0.9999);
    assert!(config.rms_tolerance.is_none());
    assert!(config.peak_amplitude_limit.is_none());
}

#[test]
fn test_strict_config_tighter_than_default() {
    let strict = ComparisonConfig::strict();
    let default = ComparisonConfig::default();
    assert!(strict.abs_tolerance < default.abs_tolerance);
    assert!(strict.rel_tolerance < default.rel_tolerance);
    assert!(strict.cosine_threshold > default.cosine_threshold);
}

#[test]
fn test_relaxed_config_looser_than_default() {
    let relaxed = ComparisonConfig::relaxed();
    let default = ComparisonConfig::default();
    assert!(relaxed.abs_tolerance > default.abs_tolerance);
    assert!(relaxed.rel_tolerance > default.rel_tolerance);
    assert!(relaxed.cosine_threshold < default.cosine_threshold);
}

// ===========================================================================
// 27. Safetensors loading integration
// ===========================================================================

/// Helper: build a minimal safetensors byte buffer from f32 tensors.
fn build_safetensors_bytes(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
    let byte_bufs: Vec<Vec<u8>> = tensors
        .iter()
        .map(|&(_, _, data)| {
            data.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>()
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

#[test]
fn test_safetensors_load_and_compare_identical() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes = build_safetensors_bytes(&[("encoder.conv1", &[2, 3], &data)]);
    let trace = crate::load_safetensors_from_bytes(&bytes).expect("load should succeed");
    assert_eq!(trace.len(), 1);
    let t = trace.get(0).unwrap();
    assert_eq!(t.name, "encoder.conv1");
    assert_eq!(t.shape, vec![2, 3]);
    assert_eq!(t.data, data);
}

#[test]
fn test_safetensors_load_multiple_tensors_sorted() {
    let bytes = build_safetensors_bytes(&[
        ("z_weight", &[2], &[9.0, 10.0]),
        ("a_bias", &[3], &[1.0, 2.0, 3.0]),
        ("m_hidden", &[1], &[5.0]),
    ]);
    let trace = crate::load_safetensors_from_bytes(&bytes).expect("load should succeed");
    let names: Vec<&str> = trace.names().collect();
    assert_eq!(names, vec!["a_bias", "m_hidden", "z_weight"]);
}

#[test]
fn test_safetensors_load_then_compare_with_candidate() {
    let ref_data = vec![1.0f32, 2.0, 3.0];
    let bytes = build_safetensors_bytes(&[("layer0", &[3], &ref_data)]);
    let reference = crate::load_safetensors_from_bytes(&bytes).expect("load ref");

    let mut candidate = ReferenceTrace::new();
    candidate
        .checkpoint("layer0", &[1.000001, 2.000001, 3.000001], &[3])
        .unwrap();

    let report =
        compare_traces(&reference, &candidate, &ComparisonConfig::default()).expect("compare");
    assert!(
        report.all_passed,
        "small perturbation should pass default config"
    );
}

#[test]
fn test_safetensors_load_f16_dtype() {
    let f16_vals = [half::f16::from_f32(1.5), half::f16::from_f32(-2.5)];
    let f16_bytes: Vec<u8> = f16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![2], &f16_bytes)
        .expect("valid f16 view");
    let serialized = safetensors::tensor::serialize(vec![("f16_tensor".to_string(), view)], None)
        .expect("serialize");

    let trace = crate::load_safetensors_from_bytes(&serialized).expect("load f16");
    let t = trace.get(0).unwrap();
    assert_eq!(t.shape, vec![2]);
    assert!((t.data[0] - 1.5).abs() < 0.01);
    assert!((t.data[1] - (-2.5)).abs() < 0.01);
}

#[test]
fn test_safetensors_load_bf16_dtype() {
    let bf16_vals = [half::bf16::from_f32(3.0), half::bf16::from_f32(0.125)];
    let bf16_bytes: Vec<u8> = bf16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![2], &bf16_bytes)
        .expect("valid bf16 view");
    let serialized = safetensors::tensor::serialize(vec![("bf16_tensor".to_string(), view)], None)
        .expect("serialize");

    let trace = crate::load_safetensors_from_bytes(&serialized).expect("load bf16");
    let t = trace.get(0).unwrap();
    assert!((t.data[0] - 3.0).abs() < 0.1);
    assert!((t.data[1] - 0.125).abs() < 0.01);
}

#[test]
fn test_safetensors_invalid_bytes_returns_error() {
    let result = crate::load_safetensors_from_bytes(b"garbage data");
    assert!(result.is_err());
    match result.unwrap_err() {
        ReftestError::Safetensors(_) => {} // expected
        other => panic!("expected Safetensors error, got {other:?}"),
    }
}

// ===========================================================================
// 28. NPY round-trip integration
// ===========================================================================

#[test]
fn test_npy_write_read_roundtrip_1d() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let bytes = crate::npy::write_npy_to_bytes(&data, &[5]).expect("write npy");
    let tensor = crate::npy::read_npy_from_bytes(&bytes).expect("read npy");
    assert_eq!(tensor.shape, vec![5]);
    assert_eq!(tensor.data, data);
    assert_eq!(tensor.dtype, crate::npy::NpyDType::F32);
}

#[test]
fn test_npy_write_read_roundtrip_2d() {
    let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
    let bytes = crate::npy::write_npy_to_bytes(&data, &[3, 4]).expect("write npy");
    let tensor = crate::npy::read_npy_from_bytes(&bytes).expect("read npy");
    assert_eq!(tensor.shape, vec![3, 4]);
    assert_eq!(tensor.data, data);
}

#[test]
fn test_npy_write_read_roundtrip_scalar() {
    let data = vec![42.0f32];
    let bytes = crate::npy::write_npy_to_bytes(&data, &[]).expect("write scalar npy");
    let tensor = crate::npy::read_npy_from_bytes(&bytes).expect("read scalar npy");
    assert!(tensor.shape.is_empty());
    assert_eq!(tensor.data, vec![42.0]);
}

#[test]
fn test_npy_write_read_roundtrip_special_values() {
    let data = vec![0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
    let bytes = crate::npy::write_npy_to_bytes(&data, &[5]).expect("write npy");
    let tensor = crate::npy::read_npy_from_bytes(&bytes).expect("read npy");
    assert_eq!(tensor.data[0], 0.0);
    assert!(tensor.data[2].is_infinite() && tensor.data[2] > 0.0);
    assert!(tensor.data[3].is_infinite() && tensor.data[3] < 0.0);
    assert!(tensor.data[4].is_nan());
}

#[test]
fn test_npy_load_from_bytes_as_trace() {
    let data = vec![10.0f32, 20.0, 30.0];
    let bytes = crate::npy::write_npy_to_bytes(&data, &[3]).expect("write npy");
    let trace = crate::load_npy_from_bytes(&bytes, "nn_tensor").expect("load npy trace");
    assert_eq!(trace.len(), 1);
    let t = trace.get(0).unwrap();
    assert_eq!(t.name, "nn_tensor");
    assert_eq!(t.data, data);
}

#[test]
fn test_npy_bad_magic_returns_error() {
    let result = crate::npy::read_npy_from_bytes(b"NOT_NPY_DATA");
    assert!(result.is_err());
}

#[test]
fn test_npy_dtype_descr_roundtrip() {
    use crate::npy::NpyDType;
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
        assert_eq!(parsed, Some(dtype), "round-trip failed for {descr}");
    }
}

// ===========================================================================
// 29. Tensor name matching and trace lookup
// ===========================================================================

#[test]
fn test_trace_get_by_name_returns_first_of_duplicates() {
    let cps = vec![
        NamedTensor::new("layer", vec![1], vec![1.0]).unwrap(),
        NamedTensor::new("layer", vec![1], vec![2.0]).unwrap(),
        NamedTensor::new("layer", vec![1], vec![3.0]).unwrap(),
    ];
    let trace = ReferenceTrace::from_checkpoints(cps);
    let found = trace.get_by_name("layer").unwrap();
    assert_eq!(found.data, vec![1.0], "should return first match");
}

#[test]
fn test_trace_get_by_name_missing_returns_none() {
    let mut trace = ReferenceTrace::new();
    trace.checkpoint("encoder", &[1.0], &[1]).unwrap();
    assert!(trace.get_by_name("decoder").is_none());
    assert!(trace.get_by_name("").is_none());
}

#[test]
fn test_trace_name_with_dots_and_slashes() {
    let mut trace = ReferenceTrace::new();
    trace
        .checkpoint("model.encoder.layer_0.self_attn.q_proj", &[1.0], &[1])
        .unwrap();
    let found = trace
        .get_by_name("model.encoder.layer_0.self_attn.q_proj")
        .unwrap();
    assert_eq!(found.data, vec![1.0]);
}

#[test]
fn test_trace_iter_order_matches_insertion() {
    let mut trace = ReferenceTrace::new();
    trace.checkpoint("third", &[3.0], &[1]).unwrap();
    trace.checkpoint("first", &[1.0], &[1]).unwrap();
    trace.checkpoint("second", &[2.0], &[1]).unwrap();

    let names: Vec<&str> = trace.names().collect();
    assert_eq!(names, vec!["third", "first", "second"]);
}

// ===========================================================================
// 30. Strict mode (atol=0, rtol=0) exact match
// ===========================================================================

#[test]
fn test_strict_zero_tolerance_identical_passes() {
    let result = compare_with_tolerance(
        &[1.0f32, 2.0, 3.0],
        &[1.0, 2.0, 3.0],
        &ToleranceStrategy::Absolute { atol: 0.0 },
    )
    .expect("should succeed");
    assert!(result.passed, "identical data should pass atol=0.0");
    assert_eq!(result.max_diff, 0.0);
    assert_eq!(result.num_mismatches, 0);
}

#[test]
fn test_strict_zero_tolerance_any_diff_fails() {
    let result = compare_with_tolerance(
        &[1.0f32],
        &[1.0 + f32::EPSILON],
        &ToleranceStrategy::Absolute { atol: 0.0 },
    )
    .expect("should succeed");
    assert!(!result.passed, "any diff should fail atol=0.0");
    assert_eq!(result.num_mismatches, 1);
}

#[test]
fn test_strict_zero_mixed_identical_passes() {
    let result = compare_with_tolerance(
        &[0.5f32, -0.5, 100.0],
        &[0.5, -0.5, 100.0],
        &ToleranceStrategy::Mixed {
            atol: 0.0,
            rtol: 0.0,
        },
    )
    .expect("should succeed");
    assert!(result.passed);
}

// ===========================================================================
// 31. Per-element error tracking depth
// ===========================================================================

#[test]
fn test_worst_index_in_middle_of_array() {
    let a = vec![0.0f32; 10];
    let mut b = vec![0.0f32; 10];
    b[5] = 100.0; // worst deviation at index 5
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 1000.0 })
        .expect("should succeed");
    assert_eq!(result.worst_index, 5);
    assert!((result.max_diff - 100.0).abs() < 1e-6);
}

#[test]
fn test_mean_diff_large_array_with_one_outlier() {
    let n = 100;
    let a = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    b[50] = 10.0; // one outlier
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 100.0 })
        .expect("should succeed");
    // mean_diff = 10.0 / 100 = 0.1
    assert!(
        (result.mean_diff - 0.1).abs() < 1e-6,
        "mean_diff should be 0.1, got {}",
        result.mean_diff
    );
    assert_eq!(result.worst_index, 50);
}

#[test]
fn test_num_mismatches_counts_multiple_outliers() {
    let a = [0.0f32, 0.0, 0.0, 0.0, 0.0];
    let b = [0.5, 0.0, 0.5, 0.0, 0.5]; // 3 outliers > 0.1
    let result = compare_with_tolerance(&a, &b, &ToleranceStrategy::Absolute { atol: 0.1 })
        .expect("should succeed");
    assert_eq!(result.num_mismatches, 3);
    assert!(!result.passed);
}

// ===========================================================================
// 32. Multi-layer report depth
// ===========================================================================

#[test]
fn test_report_many_layers_summary_counts() {
    let mut ref_trace = ReferenceTrace::new();
    let mut cand_trace = ReferenceTrace::new();
    for i in 0..10 {
        let name = format!("layer_{i}");
        ref_trace.checkpoint(&name, &[1.0], &[1]).unwrap();
        // Even layers match, odd layers diverge.
        let val = if i % 2 == 0 { 1.0 } else { 999.0 };
        cand_trace.checkpoint(&name, &[val], &[1]).unwrap();
    }

    let report =
        compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default()).expect("compare");
    assert!(!report.all_passed);
    assert_eq!(report.layers.len(), 10);
    assert_eq!(report.first_failure, Some(1));

    let pass_count = report.layers.iter().filter(|l| l.passed).count();
    let fail_count = report.layers.iter().filter(|l| !l.passed).count();
    assert_eq!(pass_count, 5);
    assert_eq!(fail_count, 5);
}

#[test]
fn test_report_summary_lists_all_layers() {
    let mut ref_trace = ReferenceTrace::new();
    let mut cand_trace = ReferenceTrace::new();
    for i in 0..5 {
        let name = format!("block_{i}");
        ref_trace.checkpoint(&name, &[1.0], &[1]).unwrap();
        cand_trace.checkpoint(&name, &[1.0], &[1]).unwrap();
    }

    let report =
        compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default()).expect("compare");
    let summary = report.summary();
    for i in 0..5 {
        assert!(
            summary.contains(&format!("block_{i}")),
            "summary should mention block_{i}"
        );
    }
    assert!(summary.contains("All 5 layers passed"));
}

#[test]
fn test_report_layers_have_correct_shapes_and_sizes() {
    let mut ref_trace = ReferenceTrace::new();
    let mut cand_trace = ReferenceTrace::new();
    ref_trace.checkpoint("small", &[1.0, 2.0], &[2]).unwrap();
    ref_trace.checkpoint("big", &[0.0; 100], &[10, 10]).unwrap();
    cand_trace.checkpoint("small", &[1.0, 2.0], &[2]).unwrap();
    cand_trace
        .checkpoint("big", &[0.0; 100], &[10, 10])
        .unwrap();

    let report =
        compare_traces(&ref_trace, &cand_trace, &ComparisonConfig::default()).expect("compare");
    assert_eq!(report.layers[0].shape, vec![2]);
    assert_eq!(report.layers[0].num_elements, 2);
    assert_eq!(report.layers[1].shape, vec![10, 10]);
    assert_eq!(report.layers[1].num_elements, 100);
}

// ===========================================================================
// 33. ComparisonConfig builder chain
// ===========================================================================

#[test]
fn test_config_builder_chain_rms_and_peak() {
    let config = ComparisonConfig::new(1e-3, 1e-2, 0.99)
        .with_rms_tolerance(0.05)
        .with_peak_amplitude_limit(10.0);
    assert_eq!(config.abs_tolerance, 1e-3);
    assert_eq!(config.rel_tolerance, 1e-2);
    assert_eq!(config.cosine_threshold, 0.99);
    assert_eq!(config.rms_tolerance, Some(0.05));
    assert_eq!(config.peak_amplitude_limit, Some(10.0));
}

#[test]
fn test_config_peak_amplitude_rejects_large_candidate() {
    let a = tensor_1d("x", vec![0.0]);
    let b = tensor_1d("x", vec![50.0]);
    let config = ComparisonConfig::new(100.0, 100.0, 0.0).with_peak_amplitude_limit(10.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        !result.passed,
        "peak_amplitude=50.0 should exceed limit=10.0"
    );
    assert_eq!(result.peak_amplitude, 50.0);
}

#[test]
fn test_config_rms_gate_independent_of_abs_gate() {
    // abs diff passes but RMS fails.
    let a = tensor_1d("x", vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let b = tensor_1d(
        "x",
        vec![
            0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
        ],
    );
    let config = ComparisonConfig::new(0.01, 1.0, 0.0).with_rms_tolerance(0.001);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    // abs passes (0.003 < 0.01), but rms = 0.003 > 0.001
    assert!(!result.passed, "rms gate should cause failure");
}

// ===========================================================================
// 34. Capture API integration
// ===========================================================================

#[test]
fn test_capture_multiple_checkpoints_and_compare() {
    let (reference, ()) = ReferenceTrace::capture(|t| {
        t.checkpoint("embed", &[1.0, 2.0, 3.0], &[3]).unwrap();
        t.checkpoint("attn", &[4.0, 5.0], &[2]).unwrap();
        t.checkpoint("ffn", &[6.0], &[1]).unwrap();
    });

    let (candidate, ()) = ReferenceTrace::capture(|t| {
        t.checkpoint("embed", &[1.0, 2.0, 3.0], &[3]).unwrap();
        t.checkpoint("attn", &[4.0, 5.0], &[2]).unwrap();
        t.checkpoint("ffn", &[6.0], &[1]).unwrap();
    });

    let report =
        compare_traces(&reference, &candidate, &ComparisonConfig::default()).expect("compare");
    assert!(report.all_passed);
    assert_eq!(report.layers.len(), 3);
}

#[test]
fn test_capture_preserves_closure_return_value() {
    let (trace, result) = ReferenceTrace::capture(|t| {
        t.checkpoint("h", &[1.0], &[1]).unwrap();
        vec![1, 2, 3]
    });
    assert_eq!(trace.len(), 1);
    assert_eq!(result, vec![1, 2, 3]);
}

// ===========================================================================
// 35. NamedTensor validation edge cases
// ===========================================================================

#[test]
fn test_named_tensor_rejects_shape_data_mismatch() {
    let err = NamedTensor::new("bad", vec![3, 3], vec![1.0, 2.0]).unwrap_err();
    match err {
        ReftestError::ElementCountMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, 9);
            assert_eq!(actual, 2);
        }
        other => panic!("expected ElementCountMismatch, got {other:?}"),
    }
}

#[test]
fn test_named_tensor_zero_in_shape() {
    // Shape [5, 0] means 0 elements.
    let t = NamedTensor::new("zero_dim", vec![5, 0], vec![]).unwrap();
    assert_eq!(t.numel(), 0);
}

#[test]
fn test_named_tensor_shape_product_overflow() {
    let err = NamedTensor::new("overflow", vec![usize::MAX, 2], vec![]).unwrap_err();
    match err {
        ReftestError::ShapeProductOverflow(_) => {}
        other => panic!("expected ShapeProductOverflow, got {other:?}"),
    }
}

// ===========================================================================
// 36. Tolerance data length mismatch
// ===========================================================================

#[test]
fn test_tolerance_length_mismatch_returns_error() {
    let err = compare_with_tolerance(
        &[1.0f32, 2.0],
        &[1.0],
        &ToleranceStrategy::Absolute { atol: 1.0 },
    )
    .unwrap_err();
    match err {
        ReftestError::DataLengthMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, 1);
            assert_eq!(actual, 2);
        }
        other => panic!("expected DataLengthMismatch, got {other:?}"),
    }
}

#[test]
fn test_tolerance_empty_slices_returns_error() {
    let err =
        compare_with_tolerance(&[], &[], &ToleranceStrategy::Absolute { atol: 1.0 }).unwrap_err();
    match err {
        ReftestError::EmptyTensor(_) => {}
        other => panic!("expected EmptyTensor, got {other:?}"),
    }
}

// ===========================================================================
// 37. Trace length mismatch error
// ===========================================================================

#[test]
fn test_trace_length_mismatch_different_sizes() {
    let mut a = ReferenceTrace::new();
    a.checkpoint("l1", &[1.0], &[1]).unwrap();
    a.checkpoint("l2", &[2.0], &[1]).unwrap();
    a.checkpoint("l3", &[3.0], &[1]).unwrap();

    let mut b = ReferenceTrace::new();
    b.checkpoint("l1", &[1.0], &[1]).unwrap();

    let err = compare_traces(&a, &b, &ComparisonConfig::default()).unwrap_err();
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
// 38. Cosine similarity: orthogonal and zero vectors
// ===========================================================================

#[test]
fn test_cosine_orthogonal_vectors() {
    let a = tensor_1d("x", vec![1.0, 0.0]);
    let b = tensor_1d("x", vec![0.0, 1.0]);
    let config = ComparisonConfig::new(100.0, 100.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        result.cosine_similarity.abs() < 1e-6,
        "orthogonal vectors should have cosine~0.0, got {}",
        result.cosine_similarity
    );
}

#[test]
fn test_cosine_both_zero_vectors_treated_as_identical() {
    let a = tensor_1d("x", vec![0.0, 0.0, 0.0]);
    let b = tensor_1d("x", vec![0.0, 0.0, 0.0]);
    let config = ComparisonConfig::new(0.0, 0.0, 1.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert_eq!(
        result.cosine_similarity, 1.0,
        "both-zero should have cosine=1.0"
    );
    assert!(result.passed);
}

#[test]
fn test_cosine_one_zero_one_nonzero() {
    let a = tensor_1d("x", vec![0.0, 0.0]);
    let b = tensor_1d("x", vec![1.0, 2.0]);
    let config = ComparisonConfig::new(100.0, 100.0, 0.5);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert_eq!(
        result.cosine_similarity, 0.0,
        "zero vs nonzero should have cosine=0.0"
    );
    assert!(!result.passed, "cosine=0.0 should fail threshold=0.5");
}

// ===========================================================================
// 39. ULP distance edge cases
// ===========================================================================

#[test]
fn test_ulp_nan_vs_anything_fails() {
    let result = compare_with_tolerance(
        &[f32::NAN],
        &[1.0],
        &ToleranceStrategy::ULP { max_ulps: u32::MAX },
    )
    .expect("should succeed");
    assert!(!result.passed, "NaN should fail ULP comparison");
}

#[test]
fn test_ulp_positive_negative_zero_is_zero_distance() {
    let result = compare_with_tolerance(
        &[0.0f32],
        &[-0.0f32],
        &ToleranceStrategy::ULP { max_ulps: 0 },
    )
    .expect("should succeed");
    assert!(result.passed, "+0.0 and -0.0 should be 0 ULPs apart");
}

#[test]
fn test_ulp_adjacent_values_at_1_dot_0() {
    let a = 1.0f32;
    let b = f32::from_bits(a.to_bits() + 1);
    let pass = compare_with_tolerance(&[a], &[b], &ToleranceStrategy::ULP { max_ulps: 1 })
        .expect("should succeed");
    assert!(pass.passed);

    let fail = compare_with_tolerance(&[a], &[b], &ToleranceStrategy::ULP { max_ulps: 0 })
        .expect("should succeed");
    assert!(!fail.passed);
}

// ===========================================================================
// 40. PercentClose with precise percentages
// ===========================================================================

#[test]
fn test_percent_close_exactly_at_threshold() {
    // 2 out of 4 elements close = 50%
    let a = [0.0f32, 0.0, 0.0, 0.0];
    let b = [0.0, 1.0, 0.0, 1.0]; // 2 outliers at threshold 0.5
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::PercentClose {
            threshold: 0.5,
            percent: 50.0,
        },
    )
    .expect("should succeed");
    assert!(result.passed, "50% close should meet 50% requirement");
}

#[test]
fn test_percent_close_just_below_threshold() {
    // 1 out of 4 close = 25%, require 50%
    let a = [0.0f32, 0.0, 0.0, 0.0];
    let b = [0.0, 1.0, 1.0, 1.0]; // 3 outliers at threshold 0.5
    let result = compare_with_tolerance(
        &a,
        &b,
        &ToleranceStrategy::PercentClose {
            threshold: 0.5,
            percent: 50.0,
        },
    )
    .expect("should succeed");
    assert!(!result.passed, "25% close should fail 50% requirement");
}
