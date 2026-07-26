// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Junction and edge-case tests for pipeline verification.
//! Extracted from pipeline_tests.rs for 500-line compliance.

use super::*;

#[test]
fn test_junction_lower_bound_violation() {
    // Stage A output lower bound (-3.0) is below Stage B input lower bound (-2.0).
    // Violation = (-2.0) - (-3.0) = 1.0.
    let stage_a = make_stage(
        "encoder",
        vec![2],
        vec![2],
        (0.0, 1.0),
        (-3.0, 1.0),
        "CROWN",
        true,
    );
    let stage_b = make_stage(
        "decoder",
        vec![2],
        vec![2],
        (-2.0, 2.0),
        (-1.0, 1.0),
        "CROWN",
        true,
    );

    let j = check_junction(&stage_a, &stage_b, 0);
    assert!(!j.bounds_contained);
    assert!((j.max_violation - 1.0).abs() < 1e-10);
    assert_eq!(j.violation_count, 2);
}

#[test]
fn test_display_impl() {
    let stage_a = make_stage(
        "a",
        vec![4],
        vec![4],
        (-1.0, 1.0),
        (-1.0, 1.0),
        "CROWN",
        true,
    );
    let stage_b = make_stage(
        "b",
        vec![4],
        vec![4],
        (-2.0, 2.0),
        (-1.0, 1.0),
        "CROWN",
        true,
    );

    let cert = verify_pipeline(&[stage_a, stage_b]).expect("valid pipeline");
    let display = format!("{cert}");
    assert!(display.contains("2 stages"));
    assert!(display.contains("valid=true"));
    assert!(display.contains("sound=true"));
}

#[test]
fn test_nan_in_bounds_treated_as_violation() {
    // NaN in output bounds must be detected as a violation, not silently skipped
    // (IEEE 754: NaN > 0.0 is false, which would bypass the violation check).
    let mut stage_a = make_stage(
        "encoder",
        vec![4],
        vec![4],
        (-1.0, 1.0),
        (-1.0, 1.0),
        "CROWN",
        true,
    );
    // Inject NaN into one output bound element.
    stage_a.output_upper[2] = f64::NAN;

    let stage_b = make_stage(
        "decoder",
        vec![4],
        vec![4],
        (-2.0, 2.0),
        (-1.0, 1.0),
        "CROWN",
        true,
    );

    let j = check_junction(&stage_a, &stage_b, 0);
    assert!(!j.bounds_contained, "NaN in bounds must cause a violation");
    assert!(j.violation_count >= 1, "at least 1 NaN element");
}

#[test]
fn test_infinity_in_bounds_treated_as_violation() {
    let mut stage_a = make_stage(
        "encoder",
        vec![4],
        vec![4],
        (-1.0, 1.0),
        (-1.0, 1.0),
        "CROWN",
        true,
    );
    stage_a.output_lower[0] = f64::NEG_INFINITY;

    let stage_b = make_stage(
        "decoder",
        vec![4],
        vec![4],
        (-2.0, 2.0),
        (-1.0, 1.0),
        "CROWN",
        true,
    );

    let j = check_junction(&stage_a, &stage_b, 0);
    assert!(!j.bounds_contained, "Inf in bounds must cause a violation");
    assert!(j.violation_count >= 1);
}

#[test]
fn test_bounds_length_mismatch_detected() {
    // Stage A has 4 output bound elements but Stage B has 8 input bound elements.
    // The 4 trailing unmatched elements should be counted as violations.
    let stage_a = VerifiedStage {
        name: "encoder".to_string(),
        input_lower: vec![-1.0; 4],
        input_upper: vec![1.0; 4],
        output_lower: vec![-1.0; 4],
        output_upper: vec![1.0; 4],
        input_shape: vec![4],
        output_shape: vec![4],
        method: "CROWN".to_string(),
        is_sound: true,
    };
    let stage_b = VerifiedStage {
        name: "decoder".to_string(),
        input_lower: vec![-2.0; 8],
        input_upper: vec![2.0; 8],
        output_lower: vec![-1.0; 8],
        output_upper: vec![1.0; 8],
        input_shape: vec![8],
        output_shape: vec![8],
        method: "CROWN".to_string(),
        is_sound: true,
    };

    let j = check_junction(&stage_a, &stage_b, 0);
    assert!(!j.bounds_contained, "length mismatch must cause violations");
    // 4 trailing unmatched elements should be violations.
    assert_eq!(j.violation_count, 4);
}

#[test]
fn test_heterogeneous_per_element_bounds() {
    // Test per-element bound checking with different bounds per element.
    // Element 0: output [-1, 0.5] ⊆ input [-2, 2] → OK
    // Element 1: output [-1, 3.0] vs input [-2, 2] → violation (3 > 2, gap=1.0)
    // Element 2: output [-1, 0.8] ⊆ input [-2, 2] → OK
    // Element 3: output [-3, 1.0] vs input [-2, 2] → violation (−2 − (−3) = 1.0)
    let stage_a = VerifiedStage {
        name: "encoder".to_string(),
        input_lower: vec![-1.0; 4],
        input_upper: vec![1.0; 4],
        output_lower: vec![-1.0, -1.0, -1.0, -3.0],
        output_upper: vec![0.5, 3.0, 0.8, 1.0],
        input_shape: vec![4],
        output_shape: vec![4],
        method: "CROWN".to_string(),
        is_sound: true,
    };
    let stage_b = VerifiedStage {
        name: "decoder".to_string(),
        input_lower: vec![-2.0; 4],
        input_upper: vec![2.0; 4],
        output_lower: vec![-1.0; 4],
        output_upper: vec![1.0; 4],
        input_shape: vec![4],
        output_shape: vec![4],
        method: "CROWN".to_string(),
        is_sound: true,
    };

    let j = check_junction(&stage_a, &stage_b, 0);
    assert!(!j.bounds_contained);
    assert_eq!(j.violation_count, 2, "elements 1 and 3 violate");
    assert!((j.max_violation - 1.0).abs() < 1e-10, "max violation = 1.0");
}
