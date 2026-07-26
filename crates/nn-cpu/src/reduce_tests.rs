// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD-optimized axis reduction operations.

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn assert_f32_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: {va} vs {vb} (diff={diff}, tol={tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// sum_f32
// ---------------------------------------------------------------------------

#[test]
fn test_sum_f32_known_values() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // two rows of 3
    let mut output = [0.0_f32; 2];
    sum_f32(&input, &mut output, 3);
    assert_f32_close(&output, &[6.0, 15.0], 1e-6, "sum_f32_known");
}

#[test]
fn test_sum_f32_single_row() {
    let input = [10.0, 20.0, 30.0, 40.0];
    let mut output = [0.0_f32; 1];
    sum_f32(&input, &mut output, 4);
    assert_f32_close(&output, &[100.0], 1e-5, "sum_f32_single_row");
}

#[test]
fn test_sum_f32_single_element() {
    let input = [42.0_f32];
    let mut output = [0.0_f32; 1];
    sum_f32(&input, &mut output, 1);
    assert_f32_close(&output, &[42.0], 1e-7, "sum_f32_single_element");
}

#[test]
fn test_sum_f32_dim_zero_noop() {
    let input: [f32; 0] = [];
    let mut output: [f32; 0] = [];
    sum_f32(&input, &mut output, 0); // no-op, should not panic
}

#[test]
fn test_sum_f32_negative_values() {
    let input = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
    let mut output = [0.0_f32; 2];
    sum_f32(&input, &mut output, 3);
    assert_f32_close(&output, &[-6.0, -15.0], 1e-6, "sum_f32_negative");
}

#[test]
fn test_sum_f32_matches_scalar() {
    let input: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.1 - 50.0).collect();
    let dim_size = 128;
    let rows = input.len() / dim_size;
    let mut scalar_out = vec![0.0_f32; rows];
    let mut simd_out = vec![0.0_f32; rows];
    sum_f32_scalar(&input, &mut scalar_out, dim_size);
    sum_f32(&input, &mut simd_out, dim_size);
    assert_f32_close(&scalar_out, &simd_out, 1e-3, "sum_f32_scalar_vs_simd");
}

#[test]
fn test_sum_f32_large_array() {
    let input: Vec<f32> = (0..4096).map(|i| ((i % 97) as f32) * 0.01).collect();
    let dim_size = 256;
    let rows = input.len() / dim_size;
    let mut output = vec![0.0_f32; rows];
    sum_f32(&input, &mut output, dim_size);
    // Verify each row against naive sum.
    for row in 0..rows {
        let start = row * dim_size;
        let expected: f32 = input[start..start + dim_size].iter().sum();
        assert!(
            (output[row] - expected).abs() < 1e-2,
            "sum_f32_large row {row}: {} vs {expected}",
            output[row]
        );
    }
}

// ---------------------------------------------------------------------------
// max_f32
// ---------------------------------------------------------------------------

#[test]
fn test_max_f32_known_values() {
    let input = [1.0, 5.0, 3.0, 2.0, 6.0, 4.0]; // two rows of 3
    let mut output = [0.0_f32; 2];
    max_f32(&input, &mut output, 3);
    assert_f32_close(&output, &[5.0, 6.0], 1e-7, "max_f32_known");
}

#[test]
fn test_max_f32_negative_values() {
    let input = [-10.0, -5.0, -8.0, -1.0, -20.0, -3.0];
    let mut output = [0.0_f32; 2];
    max_f32(&input, &mut output, 3);
    assert_f32_close(&output, &[-5.0, -1.0], 1e-7, "max_f32_negative");
}

#[test]
fn test_max_f32_single_element() {
    let input = [7.5_f32];
    let mut output = [0.0_f32; 1];
    max_f32(&input, &mut output, 1);
    assert_f32_close(&output, &[7.5], 1e-7, "max_f32_single");
}

#[test]
fn test_max_f32_matches_scalar() {
    let input: Vec<f32> = (0..1024).map(|i| (i as f32).sin() * 100.0).collect();
    let dim_size = 64;
    let rows = input.len() / dim_size;
    let mut scalar_out = vec![0.0_f32; rows];
    let mut simd_out = vec![0.0_f32; rows];
    max_f32_scalar(&input, &mut scalar_out, dim_size);
    max_f32(&input, &mut simd_out, dim_size);
    assert_f32_close(&scalar_out, &simd_out, 1e-6, "max_f32_scalar_vs_simd");
}

#[test]
fn test_max_f32_large_array() {
    let input: Vec<f32> = (0..2048).map(|i| (i as f32).cos() * 50.0).collect();
    let dim_size = 512;
    let rows = input.len() / dim_size;
    let mut output = vec![0.0_f32; rows];
    max_f32(&input, &mut output, dim_size);
    for row in 0..rows {
        let start = row * dim_size;
        let expected = input[start..start + dim_size]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (output[row] - expected).abs() < 1e-6,
            "max_f32_large row {row}: {} vs {expected}",
            output[row]
        );
    }
}

// ---------------------------------------------------------------------------
// min_f32
// ---------------------------------------------------------------------------

#[test]
fn test_min_f32_known_values() {
    let input = [5.0, 1.0, 3.0, 4.0, 2.0, 6.0]; // two rows of 3
    let mut output = [0.0_f32; 2];
    min_f32(&input, &mut output, 3);
    assert_f32_close(&output, &[1.0, 2.0], 1e-7, "min_f32_known");
}

#[test]
fn test_min_f32_negative_values() {
    let input = [-1.0, -5.0, -3.0, -10.0, -2.0, -8.0];
    let mut output = [0.0_f32; 2];
    min_f32(&input, &mut output, 3);
    assert_f32_close(&output, &[-5.0, -10.0], 1e-7, "min_f32_negative");
}

#[test]
fn test_min_f32_single_element() {
    let input = [-3.14_f32];
    let mut output = [0.0_f32; 1];
    min_f32(&input, &mut output, 1);
    assert_f32_close(&output, &[-3.14], 1e-7, "min_f32_single");
}

#[test]
fn test_min_f32_matches_scalar() {
    let input: Vec<f32> = (0..1024).map(|i| (i as f32).sin() * 100.0).collect();
    let dim_size = 64;
    let rows = input.len() / dim_size;
    let mut scalar_out = vec![0.0_f32; rows];
    let mut simd_out = vec![0.0_f32; rows];
    min_f32_scalar(&input, &mut scalar_out, dim_size);
    min_f32(&input, &mut simd_out, dim_size);
    assert_f32_close(&scalar_out, &simd_out, 1e-6, "min_f32_scalar_vs_simd");
}

#[test]
fn test_min_f32_large_array() {
    let input: Vec<f32> = (0..2048).map(|i| (i as f32).cos() * 50.0).collect();
    let dim_size = 512;
    let rows = input.len() / dim_size;
    let mut output = vec![0.0_f32; rows];
    min_f32(&input, &mut output, dim_size);
    for row in 0..rows {
        let start = row * dim_size;
        let expected = input[start..start + dim_size]
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        assert!(
            (output[row] - expected).abs() < 1e-6,
            "min_f32_large row {row}: {} vs {expected}",
            output[row]
        );
    }
}

// ---------------------------------------------------------------------------
// mean_f32
// ---------------------------------------------------------------------------

#[test]
fn test_mean_f32_known_values() {
    let input = [2.0, 4.0, 6.0, 10.0, 20.0, 30.0]; // two rows of 3
    let mut output = [0.0_f32; 2];
    mean_f32(&input, &mut output, 3);
    assert_f32_close(&output, &[4.0, 20.0], 1e-6, "mean_f32_known");
}

#[test]
fn test_mean_f32_single_element() {
    let input = [99.0_f32];
    let mut output = [0.0_f32; 1];
    mean_f32(&input, &mut output, 1);
    assert_f32_close(&output, &[99.0], 1e-7, "mean_f32_single");
}

#[test]
fn test_mean_f32_matches_scalar() {
    let input: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.3 - 150.0).collect();
    let dim_size = 128;
    let rows = input.len() / dim_size;
    let mut scalar_out = vec![0.0_f32; rows];
    let mut simd_out = vec![0.0_f32; rows];
    mean_f32_scalar(&input, &mut scalar_out, dim_size);
    mean_f32(&input, &mut simd_out, dim_size);
    assert_f32_close(&scalar_out, &simd_out, 1e-3, "mean_f32_scalar_vs_simd");
}

#[test]
fn test_mean_f32_uniform() {
    // All elements equal => mean == element value.
    let input = vec![5.0_f32; 256];
    let mut output = [0.0_f32; 1];
    mean_f32(&input, &mut output, 256);
    assert_f32_close(&output, &[5.0], 1e-5, "mean_f32_uniform");
}

#[test]
fn test_mean_f32_large_array() {
    let input: Vec<f32> = (0..4096).map(|i| ((i % 100) as f32) * 0.1).collect();
    let dim_size = 1024;
    let rows = input.len() / dim_size;
    let mut output = vec![0.0_f32; rows];
    mean_f32(&input, &mut output, dim_size);
    for row in 0..rows {
        let start = row * dim_size;
        let expected: f32 = input[start..start + dim_size].iter().sum::<f32>() / dim_size as f32;
        assert!(
            (output[row] - expected).abs() < 1e-2,
            "mean_f32_large row {row}: {} vs {expected}",
            output[row]
        );
    }
}

// ---------------------------------------------------------------------------
// argmax_f32
// ---------------------------------------------------------------------------

#[test]
fn test_argmax_f32_known_values() {
    let input = [1.0, 5.0, 3.0, 2.0, 6.0, 4.0]; // two rows of 3
    let mut output = [0_u32; 2];
    argmax_f32(&input, &mut output, 3);
    assert_eq!(output, [1, 1], "argmax_f32_known"); // max at idx 1 each row
}

#[test]
fn test_argmax_f32_first_element() {
    let input = [100.0, 1.0, 2.0, 3.0];
    let mut output = [0_u32; 1];
    argmax_f32(&input, &mut output, 4);
    assert_eq!(output[0], 0, "argmax should be 0");
}

#[test]
fn test_argmax_f32_last_element() {
    let input = [1.0, 2.0, 3.0, 100.0];
    let mut output = [0_u32; 1];
    argmax_f32(&input, &mut output, 4);
    assert_eq!(output[0], 3, "argmax should be 3");
}

#[test]
fn test_argmax_f32_ties_first_occurrence() {
    // When there are ties, first occurrence wins.
    let input = [5.0, 5.0, 5.0];
    let mut output = [0_u32; 1];
    argmax_f32(&input, &mut output, 3);
    assert_eq!(output[0], 0, "argmax ties: first occurrence");
}

#[test]
fn test_argmax_f32_single_element() {
    let input = [42.0_f32];
    let mut output = [0_u32; 1];
    argmax_f32(&input, &mut output, 1);
    assert_eq!(output[0], 0);
}

#[test]
fn test_argmax_f32_negative_values() {
    let input = [-10.0, -5.0, -8.0];
    let mut output = [0_u32; 1];
    argmax_f32(&input, &mut output, 3);
    assert_eq!(output[0], 1, "argmax of negatives: idx 1 (-5.0)");
}

#[test]
fn test_argmax_f32_matches_scalar() {
    let input: Vec<f32> = (0..1024).map(|i| (i as f32).sin() * 100.0).collect();
    let dim_size = 64;
    let rows = input.len() / dim_size;
    let mut scalar_out = vec![0_u32; rows];
    let mut simd_out = vec![0_u32; rows];
    argmax_f32_scalar(&input, &mut scalar_out, dim_size);
    argmax_f32(&input, &mut simd_out, dim_size);
    assert_eq!(scalar_out, simd_out, "argmax_f32 scalar vs simd mismatch");
}

#[test]
fn test_argmax_f32_large_array() {
    let input: Vec<f32> = (0..2048).map(|i| (i as f32).cos() * 50.0).collect();
    let dim_size = 512;
    let rows = input.len() / dim_size;
    let mut output = vec![0_u32; rows];
    argmax_f32(&input, &mut output, dim_size);
    for row in 0..rows {
        let start = row * dim_size;
        let expected = input[start..start + dim_size]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as u32;
        assert_eq!(
            output[row], expected,
            "argmax_f32_large row {row}: {} vs {expected}",
            output[row]
        );
    }
}

// ---------------------------------------------------------------------------
// argmin_f32
// ---------------------------------------------------------------------------

#[test]
fn test_argmin_f32_known_values() {
    let input = [5.0, 1.0, 3.0, 4.0, 2.0, 6.0]; // two rows of 3
    let mut output = [0_u32; 2];
    argmin_f32(&input, &mut output, 3);
    assert_eq!(output, [1, 1], "argmin_f32_known"); // min at idx 1 each row
}

#[test]
fn test_argmin_f32_first_element() {
    let input = [-100.0, 1.0, 2.0, 3.0];
    let mut output = [0_u32; 1];
    argmin_f32(&input, &mut output, 4);
    assert_eq!(output[0], 0, "argmin should be 0");
}

#[test]
fn test_argmin_f32_last_element() {
    let input = [3.0, 2.0, 1.0, -100.0];
    let mut output = [0_u32; 1];
    argmin_f32(&input, &mut output, 4);
    assert_eq!(output[0], 3, "argmin should be 3");
}

#[test]
fn test_argmin_f32_ties_first_occurrence() {
    let input = [1.0, 1.0, 1.0];
    let mut output = [0_u32; 1];
    argmin_f32(&input, &mut output, 3);
    assert_eq!(output[0], 0, "argmin ties: first occurrence");
}

#[test]
fn test_argmin_f32_single_element() {
    let input = [-7.0_f32];
    let mut output = [0_u32; 1];
    argmin_f32(&input, &mut output, 1);
    assert_eq!(output[0], 0);
}

#[test]
fn test_argmin_f32_positive_values() {
    let input = [10.0, 5.0, 8.0];
    let mut output = [0_u32; 1];
    argmin_f32(&input, &mut output, 3);
    assert_eq!(output[0], 1, "argmin of positives: idx 1 (5.0)");
}

#[test]
fn test_argmin_f32_matches_scalar() {
    let input: Vec<f32> = (0..1024).map(|i| (i as f32).sin() * 100.0).collect();
    let dim_size = 64;
    let rows = input.len() / dim_size;
    let mut scalar_out = vec![0_u32; rows];
    let mut simd_out = vec![0_u32; rows];
    argmin_f32_scalar(&input, &mut scalar_out, dim_size);
    argmin_f32(&input, &mut simd_out, dim_size);
    assert_eq!(scalar_out, simd_out, "argmin_f32 scalar vs simd mismatch");
}

#[test]
fn test_argmin_f32_large_array() {
    let input: Vec<f32> = (0..2048).map(|i| (i as f32).cos() * 50.0).collect();
    let dim_size = 512;
    let rows = input.len() / dim_size;
    let mut output = vec![0_u32; rows];
    argmin_f32(&input, &mut output, dim_size);
    for row in 0..rows {
        let start = row * dim_size;
        let expected = input[start..start + dim_size]
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as u32;
        assert_eq!(
            output[row], expected,
            "argmin_f32_large row {row}: {} vs {expected}",
            output[row]
        );
    }
}

// ---------------------------------------------------------------------------
// Edge cases: non-power-of-two dim sizes (exercise SIMD tail handling)
// ---------------------------------------------------------------------------

#[test]
fn test_reduce_non_aligned_dim_size() {
    // dim_size = 13 — not aligned to 4 (NEON) or 8 (AVX2).
    let input: Vec<f32> = (0..39).map(|i| (i as f32) * 0.5).collect(); // 3 rows of 13
    let dim_size = 13;
    let rows = 3;

    let mut sum_out = vec![0.0_f32; rows];
    let mut max_out = vec![0.0_f32; rows];
    let mut min_out = vec![0.0_f32; rows];
    let mut mean_out = vec![0.0_f32; rows];
    let mut argmax_out = vec![0_u32; rows];
    let mut argmin_out = vec![0_u32; rows];

    sum_f32(&input, &mut sum_out, dim_size);
    max_f32(&input, &mut max_out, dim_size);
    min_f32(&input, &mut min_out, dim_size);
    mean_f32(&input, &mut mean_out, dim_size);
    argmax_f32(&input, &mut argmax_out, dim_size);
    argmin_f32(&input, &mut argmin_out, dim_size);

    // Verify against scalar.
    let mut sum_scalar = vec![0.0_f32; rows];
    let mut max_scalar = vec![0.0_f32; rows];
    let mut min_scalar = vec![0.0_f32; rows];
    let mut mean_scalar = vec![0.0_f32; rows];
    let mut argmax_scalar = vec![0_u32; rows];
    let mut argmin_scalar = vec![0_u32; rows];

    sum_f32_scalar(&input, &mut sum_scalar, dim_size);
    max_f32_scalar(&input, &mut max_scalar, dim_size);
    min_f32_scalar(&input, &mut min_scalar, dim_size);
    mean_f32_scalar(&input, &mut mean_scalar, dim_size);
    argmax_f32_scalar(&input, &mut argmax_scalar, dim_size);
    argmin_f32_scalar(&input, &mut argmin_scalar, dim_size);

    assert_f32_close(&sum_out, &sum_scalar, 1e-4, "sum non-aligned");
    assert_f32_close(&max_out, &max_scalar, 1e-7, "max non-aligned");
    assert_f32_close(&min_out, &min_scalar, 1e-7, "min non-aligned");
    assert_f32_close(&mean_out, &mean_scalar, 1e-4, "mean non-aligned");
    assert_eq!(argmax_out, argmax_scalar, "argmax non-aligned");
    assert_eq!(argmin_out, argmin_scalar, "argmin non-aligned");
}

#[test]
fn test_reduce_dim_size_one() {
    // Each row is a single element.
    let input = [3.0, -1.0, 7.0, 0.5];
    let mut sum_out = [0.0_f32; 4];
    let mut max_out = [0.0_f32; 4];
    let mut min_out = [0.0_f32; 4];
    let mut mean_out = [0.0_f32; 4];
    let mut argmax_out = [0_u32; 4];
    let mut argmin_out = [0_u32; 4];

    sum_f32(&input, &mut sum_out, 1);
    max_f32(&input, &mut max_out, 1);
    min_f32(&input, &mut min_out, 1);
    mean_f32(&input, &mut mean_out, 1);
    argmax_f32(&input, &mut argmax_out, 1);
    argmin_f32(&input, &mut argmin_out, 1);

    assert_f32_close(&sum_out, &input, 1e-7, "sum dim=1");
    assert_f32_close(&max_out, &input, 1e-7, "max dim=1");
    assert_f32_close(&min_out, &input, 1e-7, "min dim=1");
    assert_f32_close(&mean_out, &input, 1e-7, "mean dim=1");
    assert_eq!(argmax_out, [0, 0, 0, 0], "argmax dim=1");
    assert_eq!(argmin_out, [0, 0, 0, 0], "argmin dim=1");
}
