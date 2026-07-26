// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for quantization verification certificates.

use super::*;
use crate::quality_bound::QualityMetricSpec;

// ---------------------------------------------------------------------------
// Element drift computation tests
// ---------------------------------------------------------------------------

#[test]
fn test_drift_identical_bounds() {
    let lo = [0.0f32, 1.0, -1.0];
    let hi = [1.0f32, 2.0, 0.0];
    let (max_d, mean_d, n) = compute_element_drift(&lo, &hi, &lo, &hi).unwrap();
    assert_eq!(n, 3);
    assert!(
        (max_d - 0.0).abs() < 1e-10,
        "identical bounds => zero drift"
    );
    assert!((mean_d - 0.0).abs() < 1e-10);
}

#[test]
fn test_drift_shifted_lower() {
    // bf16 lower bound shifts by 0.01 at element 0.
    let f32_lo = [0.0f32, 1.0];
    let f32_hi = [1.0f32, 2.0];
    let q_lo = [0.01f32, 1.0];
    let q_hi = [1.0f32, 2.0];
    let (max_d, _, _) = compute_element_drift(&f32_lo, &f32_hi, &q_lo, &q_hi).unwrap();
    assert!((max_d - 0.01).abs() < 1e-6, "max drift = {max_d}");
}

#[test]
fn test_drift_shifted_upper() {
    let f32_lo = [0.0f32];
    let f32_hi = [1.0f32];
    let q_lo = [0.0f32];
    let q_hi = [1.05f32];
    let (max_d, _, _) = compute_element_drift(&f32_lo, &f32_hi, &q_lo, &q_hi).unwrap();
    assert!((max_d - 0.05).abs() < 1e-6, "max drift = {max_d}");
}

#[test]
fn test_drift_mean_computation() {
    // Element 0: drift = 0.1, Element 1: drift = 0.3
    let f32_lo = [0.0f32, 0.0];
    let f32_hi = [1.0f32, 1.0];
    let q_lo = [0.1f32, 0.3];
    let q_hi = [1.0f32, 1.0];
    let (max_d, mean_d, _) = compute_element_drift(&f32_lo, &f32_hi, &q_lo, &q_hi).unwrap();
    assert!((max_d - 0.3).abs() < 1e-6, "max drift = {max_d}");
    assert!((mean_d - 0.2).abs() < 1e-6, "mean drift = {mean_d}");
}

#[test]
fn test_drift_rejects_mismatched_lengths() {
    let lo2 = [0.0f32, 1.0];
    let hi2 = [1.0f32, 2.0];
    let lo3 = [0.0f32, 1.0, 2.0];
    let hi3 = [1.0f32, 2.0, 3.0];
    assert!(compute_element_drift(&lo2, &hi2, &lo3, &hi3).is_err());
}

#[test]
fn test_drift_rejects_empty() {
    assert!(compute_element_drift(&[], &[], &[], &[]).is_err());
}

#[test]
fn test_drift_rejects_non_finite() {
    let lo = [f32::NAN];
    let hi = [1.0f32];
    assert!(compute_element_drift(&lo, &hi, &lo, &hi).is_err());
}

// ---------------------------------------------------------------------------
// Segment result tests
// ---------------------------------------------------------------------------

#[test]
fn test_build_segment_result_basic() {
    let f32_lo = [0.0f32, -1.0];
    let f32_hi = [1.0f32, 0.5];
    let q_lo = [0.01f32, -0.99];
    let q_hi = [1.02f32, 0.51];
    let seg = build_segment_result("test_seg", &f32_lo, &f32_hi, &q_lo, &q_hi).unwrap();

    assert_eq!(seg.segment_name, "test_seg");
    assert_eq!(seg.num_elements, 2);
    assert!(seg.max_element_drift > 0.0);
    assert!((seg.f32_bounds.0 - (-1.0)).abs() < 1e-6);
    assert!((seg.f32_bounds.1 - 1.0).abs() < 1e-6);
    assert!((seg.f32_output_width - 2.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Certificate construction tests
// ---------------------------------------------------------------------------

fn test_quality_specs() -> Vec<QualityMetricSpec> {
    vec![
        QualityMetricSpec {
            name: "SNR".into(),
            lipschitz_constant: 10.0,
            baseline_value: 30.0,
            threshold: 10.0,
            higher_is_better: true,
            citation: "test",
        },
        QualityMetricSpec {
            name: "MCD".into(),
            lipschitz_constant: 1.0,
            baseline_value: 3.0,
            threshold: 6.0,
            higher_is_better: false,
            citation: "test",
        },
    ]
}

#[test]
fn test_certificate_small_drift_preserves_quality() {
    let seg = build_segment_result(
        "encoder",
        &[0.0, -1.0],
        &[1.0, 0.5],
        &[0.001, -0.999],
        &[1.001, 0.501],
    )
    .unwrap();

    let cert =
        build_quantization_certificate("F32", "BF16", vec![seg], &test_quality_specs()).unwrap();

    assert!(
        cert.quality_preserved,
        "small drift should preserve quality"
    );
    assert_eq!(cert.source_dtype, "F32");
    assert_eq!(cert.target_dtype, "BF16");
    assert!(cert.max_output_drift < 0.01);
    assert!(cert.quality_certificate.all_guaranteed);
}

#[test]
fn test_certificate_large_drift_fails_quality() {
    let seg = build_segment_result(
        "decoder",
        &[0.0],
        &[1.0],
        &[5.0], // Huge shift
        &[6.0],
    )
    .unwrap();

    let cert =
        build_quantization_certificate("F32", "BF16", vec![seg], &test_quality_specs()).unwrap();

    assert!(!cert.quality_preserved, "large drift should fail quality");
    assert!(cert.max_output_drift > 4.0);
}

#[test]
fn test_certificate_multi_segment_uses_max_drift() {
    let seg1 = build_segment_result("small", &[0.0], &[1.0], &[0.001], &[1.001]).unwrap();
    let seg2 = build_segment_result("large", &[0.0], &[1.0], &[0.1], &[1.1]).unwrap();

    let cert =
        build_quantization_certificate("F32", "BF16", vec![seg1, seg2], &test_quality_specs())
            .unwrap();

    // Max drift should come from seg2.
    assert!(
        (cert.max_output_drift - 0.1).abs() < 1e-6,
        "max_output_drift = {}",
        cert.max_output_drift
    );
    assert_eq!(cert.segment_results.len(), 2);
}

#[test]
fn test_certificate_rejects_empty_segments() {
    assert!(build_quantization_certificate("F32", "BF16", vec![], &test_quality_specs()).is_err());
}
