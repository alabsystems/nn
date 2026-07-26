// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

const EPS: f32 = 1e-5;

// ---------------------------------------------------------------------------
// L2 normalize
// ---------------------------------------------------------------------------

#[test]
fn test_l2_normalize_unit_vector() {
    // A unit vector should be unchanged after L2 normalization.
    let inv_sqrt3 = 1.0_f32 / 3.0_f32.sqrt();
    let input = [inv_sqrt3, inv_sqrt3, inv_sqrt3];
    let mut output = [0.0f32; 3];
    l2_normalize(&input, &mut output);
    for (o, &x) in output.iter().zip(input.iter()) {
        assert!((o - x).abs() < EPS, "expected {x}, got {o}");
    }
}

#[test]
fn test_l2_normalize_scale() {
    // [3, 4] has L2 norm 5, so normalized = [0.6, 0.8].
    let input = [3.0f32, 4.0];
    let mut output = [0.0f32; 2];
    l2_normalize(&input, &mut output);
    assert!(
        (output[0] - 0.6).abs() < EPS,
        "expected 0.6, got {}",
        output[0]
    );
    assert!(
        (output[1] - 0.8).abs() < EPS,
        "expected 0.8, got {}",
        output[1]
    );
}

#[test]
fn test_l2_normalize_zero_vector() {
    let input = [0.0f32; 4];
    let mut output = [1.0f32; 4];
    l2_normalize(&input, &mut output);
    for &o in &output {
        assert_eq!(o, 0.0, "zero input should produce zero output");
    }
}

#[test]
fn test_l2_normalize_single_element() {
    let input = [5.0f32];
    let mut output = [0.0f32; 1];
    l2_normalize(&input, &mut output);
    assert!(
        (output[0] - 1.0).abs() < EPS,
        "single positive element should normalize to 1.0"
    );

    let input_neg = [-3.0f32];
    l2_normalize(&input_neg, &mut output);
    assert!(
        (output[0] - (-1.0)).abs() < EPS,
        "single negative element should normalize to -1.0"
    );
}

// ---------------------------------------------------------------------------
// L1 normalize
// ---------------------------------------------------------------------------

#[test]
fn test_l1_normalize_basic() {
    // After L1 normalization, the sum of absolute values should be 1.
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let mut output = [0.0f32; 4];
    l1_normalize(&input, &mut output);
    let abs_sum: f32 = output.iter().map(|x| x.abs()).sum();
    assert!(
        (abs_sum - 1.0).abs() < EPS,
        "L1 norm of output should be 1.0, got {abs_sum}"
    );
}

#[test]
fn test_l1_normalize_with_negatives() {
    let input = [-2.0f32, 3.0, -5.0];
    let mut output = [0.0f32; 3];
    l1_normalize(&input, &mut output);
    let abs_sum: f32 = output.iter().map(|x| x.abs()).sum();
    assert!(
        (abs_sum - 1.0).abs() < EPS,
        "L1 norm of output should be 1.0, got {abs_sum}"
    );
    // Signs should be preserved.
    assert!(
        output[0] < 0.0,
        "sign should be preserved for negative input"
    );
    assert!(
        output[1] > 0.0,
        "sign should be preserved for positive input"
    );
    assert!(
        output[2] < 0.0,
        "sign should be preserved for negative input"
    );
}

#[test]
fn test_l1_normalize_zero_vector() {
    let input = [0.0f32; 3];
    let mut output = [1.0f32; 3];
    l1_normalize(&input, &mut output);
    for &o in &output {
        assert_eq!(o, 0.0, "zero input should produce zero output");
    }
}

// ---------------------------------------------------------------------------
// Min-max normalize
// ---------------------------------------------------------------------------

#[test]
fn test_min_max_normalize_range() {
    // Output should be in [0, 1].
    let input = [2.0f32, 5.0, 1.0, 8.0, 3.0];
    let mut output = [0.0f32; 5];
    min_max_normalize(&input, &mut output);
    for &o in &output {
        assert!(
            (-EPS..=1.0 + EPS).contains(&o),
            "output {o} should be in [0, 1]"
        );
    }
    // Min element should map to 0, max to 1.
    assert!((output[2] - 0.0).abs() < EPS, "min element should be 0.0");
    assert!((output[3] - 1.0).abs() < EPS, "max element should be 1.0");
}

#[test]
fn test_min_max_normalize_constant() {
    // All same value: range is 0, output should be all zeros.
    let input = [3.0f32; 6];
    let mut output = [1.0f32; 6];
    min_max_normalize(&input, &mut output);
    for &o in &output {
        assert_eq!(o, 0.0, "constant input should produce all-zero output");
    }
}

#[test]
fn test_min_max_normalize_two_values() {
    let input = [10.0f32, 20.0];
    let mut output = [0.0f32; 2];
    min_max_normalize(&input, &mut output);
    assert!((output[0] - 0.0).abs() < EPS, "min should map to 0.0");
    assert!((output[1] - 1.0).abs() < EPS, "max should map to 1.0");
}

#[test]
fn test_min_max_normalize_empty() {
    let input: [f32; 0] = [];
    let mut output: [f32; 0] = [];
    min_max_normalize(&input, &mut output);
    // No panic — graceful no-op.
}

// ---------------------------------------------------------------------------
// SIMD matches reference
// ---------------------------------------------------------------------------

#[test]
fn test_l2_normalize_matches_reference() {
    let input: Vec<f32> = (0..37).map(|i| (i as f32) * 0.7 - 10.0).collect();
    let reference = l2_normalize_reference(&input);
    let mut simd_out = vec![0.0f32; input.len()];
    l2_normalize(&input, &mut simd_out);
    for (i, (&r, &s)) in reference.iter().zip(simd_out.iter()).enumerate() {
        assert!(
            (r - s).abs() < EPS,
            "l2_normalize mismatch at index {i}: reference={r}, simd={s}"
        );
    }
}

#[test]
fn test_l1_normalize_matches_reference() {
    let input: Vec<f32> = (0..37).map(|i| (i as f32) * 0.7 - 10.0).collect();
    let reference = l1_normalize_reference(&input);
    let mut simd_out = vec![0.0f32; input.len()];
    l1_normalize(&input, &mut simd_out);
    for (i, (&r, &s)) in reference.iter().zip(simd_out.iter()).enumerate() {
        assert!(
            (r - s).abs() < EPS,
            "l1_normalize mismatch at index {i}: reference={r}, simd={s}"
        );
    }
}

#[test]
fn test_min_max_normalize_matches_reference() {
    let input: Vec<f32> = (0..37).map(|i| (i as f32) * 0.7 - 10.0).collect();
    let reference = min_max_normalize_reference(&input);
    let mut simd_out = vec![0.0f32; input.len()];
    min_max_normalize(&input, &mut simd_out);
    for (i, (&r, &s)) in reference.iter().zip(simd_out.iter()).enumerate() {
        assert!(
            (r - s).abs() < EPS,
            "min_max_normalize mismatch at index {i}: reference={r}, simd={s}"
        );
    }
}
