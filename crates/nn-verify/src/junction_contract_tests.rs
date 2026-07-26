// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for junction contract verification.

use super::{verify_junction, verify_junctions, SubBlockBounds, JUNCTION_MARGIN};

fn block(name: &str, in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32) -> SubBlockBounds {
    SubBlockBounds {
        name: name.to_string(),
        input_lower: vec![in_lo],
        input_upper: vec![in_hi],
        output_lower: vec![out_lo],
        output_upper: vec![out_hi],
    }
}

#[test]
fn test_valid_junction() {
    // Upstream output [-1, 1] fits within downstream input [-2, 2].
    let a = block("a", -10.0, 10.0, -1.0, 1.0);
    let b = block("b", -2.0, 2.0, -0.5, 0.5);

    let proof = verify_junction(&a, &b).expect("should succeed");
    assert!(proof.is_valid, "output [-1,1] should fit in input [-2,2]");
    assert_eq!(proof.violation_count, 0);
    assert_eq!(proof.max_violation, 0.0);
}

#[test]
fn test_invalid_junction_upper() {
    // Upstream output [-1, 3] exceeds downstream input [-2, 2] on upper side.
    let a = block("a", -10.0, 10.0, -1.0, 3.0);
    let b = block("b", -2.0, 2.0, -0.5, 0.5);

    let proof = verify_junction(&a, &b).expect("should succeed");
    assert!(!proof.is_valid, "output upper 3.0 exceeds input upper 2.0");
    assert!(proof.violation_count > 0);
    // Violation: 3.0 - 2.0 - MARGIN ≈ 1.0
    assert!(proof.max_violation > 0.9);
}

#[test]
fn test_invalid_junction_lower() {
    // Upstream output [-3, 1] exceeds downstream input [-2, 2] on lower side.
    let a = block("a", -10.0, 10.0, -3.0, 1.0);
    let b = block("b", -2.0, 2.0, -0.5, 0.5);

    let proof = verify_junction(&a, &b).expect("should succeed");
    assert!(!proof.is_valid, "output lower -3.0 below input lower -2.0");
    assert!(proof.violation_count > 0);
}

#[test]
fn test_exact_match_with_margin() {
    // Upstream output matches downstream input exactly — should pass due to margin.
    let a = block("a", -10.0, 10.0, -2.0, 2.0);
    let b = block("b", -2.0, 2.0, -0.5, 0.5);

    let proof = verify_junction(&a, &b).expect("should succeed");
    assert!(proof.is_valid, "exact match should pass with margin");
}

#[test]
fn test_non_finite_output_is_violation() {
    let a = block("a", -10.0, 10.0, f32::NEG_INFINITY, f32::INFINITY);
    let b = block("b", -2.0, 2.0, -0.5, 0.5);

    let proof = verify_junction(&a, &b).expect("should succeed");
    assert!(
        !proof.is_valid,
        "non-finite output bounds should be violation"
    );
    assert_eq!(proof.max_violation, f32::INFINITY);
}

#[test]
fn test_dimension_mismatch() {
    let a = SubBlockBounds {
        name: "a".to_string(),
        input_lower: vec![-1.0],
        input_upper: vec![1.0],
        output_lower: vec![-1.0, -1.0], // 2 elements
        output_upper: vec![1.0, 1.0],
    };
    let b = SubBlockBounds {
        name: "b".to_string(),
        input_lower: vec![-2.0], // 1 element — mismatch
        input_upper: vec![2.0],
        output_lower: vec![-0.5],
        output_upper: vec![0.5],
    };

    let result = verify_junction(&a, &b);
    assert!(result.is_err(), "dimension mismatch should return error");
}

#[test]
fn test_verify_junctions_chain() {
    let blocks = vec![
        block("b0", -10.0, 10.0, -5.0, 5.0),
        block("b1", -6.0, 6.0, -3.0, 3.0),
        block("b2", -4.0, 4.0, -1.0, 1.0),
    ];

    let result = verify_junctions(&blocks).expect("should succeed");
    assert!(result.all_valid());
    assert_eq!(result.proofs.len(), 2);
    assert_eq!(result.invalid_count(), 0);
}

#[test]
fn test_verify_junctions_chain_with_violation() {
    let blocks = vec![
        block("b0", -10.0, 10.0, -5.0, 5.0),
        block("b1", -3.0, 3.0, -4.0, 4.0), // input [-3,3] too narrow for b0 output [-5,5]
        block("b2", -5.0, 5.0, -1.0, 1.0),
    ];

    let result = verify_junctions(&blocks).expect("should succeed");
    assert!(!result.all_valid());
    assert_eq!(result.invalid_count(), 1);
    assert!(result.max_violation() > 1.0);
}

#[test]
fn test_verify_junctions_too_few_blocks() {
    let blocks = vec![block("b0", -10.0, 10.0, -5.0, 5.0)];
    let result = verify_junctions(&blocks);
    assert!(result.is_err());
}

#[test]
fn test_multi_element_bounds() {
    let a = SubBlockBounds {
        name: "a".to_string(),
        input_lower: vec![-10.0, -10.0, -10.0],
        input_upper: vec![10.0, 10.0, 10.0],
        output_lower: vec![-1.0, -2.0, -0.5],
        output_upper: vec![1.0, 2.0, 0.5],
    };
    let b = SubBlockBounds {
        name: "b".to_string(),
        input_lower: vec![-2.0, -3.0, -1.0],
        input_upper: vec![2.0, 3.0, 1.0],
        output_lower: vec![-0.5, -1.0, -0.2],
        output_upper: vec![0.5, 1.0, 0.2],
    };

    let proof = verify_junction(&a, &b).expect("should succeed");
    assert!(proof.is_valid, "all elements should fit");
}

#[test]
fn test_junction_margin_constant() {
    // Verify the margin matches SMT_QUANTIZATION_MARGIN.
    assert!((JUNCTION_MARGIN - 1e-4).abs() < f32::EPSILON);
}
