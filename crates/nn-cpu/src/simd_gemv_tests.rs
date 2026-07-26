// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD-optimized GEMV with explicit tier entry points.

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Naive f64 reference GEMV for ground truth.
fn naive_gemv_f64(matrix: &[f32], vec: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m];
    for i in 0..m {
        let mut acc = 0.0_f64;
        for j in 0..k {
            acc += f64::from(matrix[i * k + j]) * f64::from(vec[j]);
        }
        out[i] = acc as f32;
    }
    out
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < tol,
            "{label} [{i}]: actual={a}, expected={e}, diff={diff}"
        );
    }
}

// ---------------------------------------------------------------------------
// Identity matrix (should copy input vector)
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_identity() {
    let matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // 3x3
    let vec = [3.0, 5.0, 7.0];
    let mut out = vec![0.0f32; 3];
    gemv_f32_scalar(&matrix, &vec, 3, 3, &mut out);
    assert_close(&out, &vec, 1e-6, "scalar identity 3x3");
}

#[test]
fn test_dispatch_identity_4x4() {
    let mut matrix = vec![0.0f32; 16];
    for i in 0..4 {
        matrix[i * 4 + i] = 1.0;
    }
    let vec = [1.0, 2.0, 3.0, 4.0];
    let mut out = vec![0.0f32; 4];
    gemv_f32(&matrix, &vec, 4, 4, &mut out);
    assert_close(&out, &vec, 1e-5, "dispatch identity 4x4");
}

#[test]
fn test_neon_identity() {
    let matrix = [1.0, 0.0, 0.0, 1.0]; // 2x2
    let vec = [42.0, -17.5];
    let mut out = vec![0.0f32; 2];
    gemv_f32_neon(&matrix, &vec, 2, 2, &mut out);
    assert_close(&out, &vec, 1e-5, "neon identity 2x2");
}

#[test]
fn test_avx2_identity() {
    let matrix = [1.0, 0.0, 0.0, 1.0]; // 2x2
    let vec = [42.0, -17.5];
    let mut out = vec![0.0f32; 2];
    gemv_f32_avx2(&matrix, &vec, 2, 2, &mut out);
    assert_close(&out, &vec, 1e-5, "avx2 identity 2x2");
}

// ---------------------------------------------------------------------------
// Known values
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_known_2x3() {
    // matrix = [[1,2,3],[4,5,6]] (2x3)
    // vec = [1, 2, 3]
    // out = [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
    let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let vec = [1.0, 2.0, 3.0];
    let expected = [14.0, 32.0];
    let mut out = vec![0.0f32; 2];
    gemv_f32_scalar(&matrix, &vec, 2, 3, &mut out);
    assert_close(&out, &expected, 1e-5, "scalar known 2x3");
}

#[test]
fn test_dispatch_known_3x2() {
    // matrix = [[1,4],[2,5],[3,6]] (3x2)
    // vec = [7, 8]
    // out = [1*7+4*8, 2*7+5*8, 3*7+6*8] = [39, 54, 69]
    let matrix = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let vec = [7.0, 8.0];
    let expected = [39.0, 54.0, 69.0];
    let mut out = vec![0.0f32; 3];
    gemv_f32(&matrix, &vec, 3, 2, &mut out);
    assert_close(&out, &expected, 1e-4, "dispatch known 3x2");
}

// ---------------------------------------------------------------------------
// With bias
// ---------------------------------------------------------------------------

#[test]
fn test_bias_known_values() {
    let matrix = [1.0, 2.0, 3.0, 4.0]; // 2x2
    let vec = [5.0, 6.0];
    let bias = [10.0, 20.0];
    // Without bias: [1*5+2*6, 3*5+4*6] = [17, 39]
    // With bias: [27, 59]
    let expected = [27.0, 59.0];
    let mut out = vec![0.0f32; 2];
    gemv_bias_f32(&matrix, &vec, &bias, 2, 2, &mut out);
    assert_close(&out, &expected, 1e-4, "bias known 2x2");
}

#[test]
fn test_bias_zero_bias_equals_no_bias() {
    let m = 5;
    let k = 7;
    let matrix: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.3).sin()).collect();
    let vec: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.7).cos()).collect();
    let zero_bias = vec![0.0f32; m];

    let mut out_no_bias = vec![0.0f32; m];
    let mut out_zero_bias = vec![0.0f32; m];
    gemv_f32(&matrix, &vec, m, k, &mut out_no_bias);
    gemv_bias_f32(&matrix, &vec, &zero_bias, m, k, &mut out_zero_bias);
    assert_close(&out_zero_bias, &out_no_bias, 1e-6, "zero bias == no bias");
}

#[test]
fn test_bias_scalar_matches_dispatch() {
    let m = 4;
    let k = 6;
    let matrix: Vec<f32> = (0..m * k).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let vec: Vec<f32> = (0..k).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let bias: Vec<f32> = (0..m).map(|i| (i as f32) * 0.3).collect();

    let mut scalar_out = vec![0.0f32; m];
    gemv_bias_f32_scalar(&matrix, &vec, &bias, m, k, &mut scalar_out);

    let mut dispatch_out = vec![0.0f32; m];
    gemv_bias_f32(&matrix, &vec, &bias, m, k, &mut dispatch_out);

    assert_close(&dispatch_out, &scalar_out, 1e-4, "bias scalar vs dispatch");
}

// ---------------------------------------------------------------------------
// Single row (1xK)
// ---------------------------------------------------------------------------

#[test]
fn test_single_row() {
    let matrix = [1.0, 2.0, 3.0, 4.0]; // 1x4
    let vec = [5.0, 6.0, 7.0, 8.0];
    let expected = [1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0]; // [70]
    let mut out = vec![0.0f32; 1];
    gemv_f32(&matrix, &vec, 1, 4, &mut out);
    assert_close(&out, &expected, 1e-4, "dispatch single row");
}

#[test]
fn test_single_element() {
    let matrix = [3.0_f32];
    let vec = [7.0_f32];
    let expected = [21.0_f32];
    let mut out = vec![0.0f32; 1];
    gemv_f32(&matrix, &vec, 1, 1, &mut out);
    assert_close(&out, &expected, 1e-6, "dispatch single element");
}

// ---------------------------------------------------------------------------
// SIMD vs reference comparison
// ---------------------------------------------------------------------------

#[test]
fn test_neon_vs_reference_8x16() {
    let m = 8;
    let k = 16;
    let matrix: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.37).sin()).collect();
    let vec: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.53).cos()).collect();
    let expected = gemv_reference(&matrix, &vec, m, k);
    let mut out = vec![0.0f32; m];
    gemv_f32_neon(&matrix, &vec, m, k, &mut out);
    assert_close(&out, &expected, 1e-4, "neon vs reference 8x16");
}

#[test]
fn test_avx2_vs_reference_8x16() {
    let m = 8;
    let k = 16;
    let matrix: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.37).sin()).collect();
    let vec: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.53).cos()).collect();
    let expected = gemv_reference(&matrix, &vec, m, k);
    let mut out = vec![0.0f32; m];
    gemv_f32_avx2(&matrix, &vec, m, k, &mut out);
    assert_close(&out, &expected, 1e-4, "avx2 vs reference 8x16");
}

#[test]
fn test_dispatch_vs_naive_32x64() {
    let m = 32;
    let k = 64;
    let matrix: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.11).sin()).collect();
    let vec: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.23).cos()).collect();
    let expected = naive_gemv_f64(&matrix, &vec, m, k);
    let mut out = vec![0.0f32; m];
    gemv_f32(&matrix, &vec, m, k, &mut out);
    assert_close(&out, &expected, 1e-3, "dispatch vs naive 32x64");
}

// ---------------------------------------------------------------------------
// Varied sizes (exercises SIMD tail handling)
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_varied_sizes() {
    for (m, k) in [
        (1, 1),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 4),
        (5, 7),
        (7, 8),
        (8, 9),
        (9, 15),
        (16, 16),
        (17, 31),
        (32, 33),
    ] {
        let matrix: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.3).sin()).collect();
        let vec: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.7).cos()).collect();
        let expected = naive_gemv_f64(&matrix, &vec, m, k);
        let mut out = vec![0.0f32; m];
        gemv_f32(&matrix, &vec, m, k, &mut out);
        assert_close(&out, &expected, 1e-4, &format!("dispatch varied {m}x{k}"));
    }
}

// ---------------------------------------------------------------------------
// Reference function returns correct output
// ---------------------------------------------------------------------------

#[test]
fn test_reference_matches_scalar() {
    let matrix = [1.0, 0.5, 0.5, 1.0]; // 2x2
    let vec = [2.0, 3.0];
    let ref_out = gemv_reference(&matrix, &vec, 2, 2);
    let mut scalar_out = vec![0.0f32; 2];
    gemv_f32_scalar(&matrix, &vec, 2, 2, &mut scalar_out);
    assert_close(&ref_out, &scalar_out, 1e-7, "reference vs scalar");
}

// ---------------------------------------------------------------------------
// Zero matrix / zero vector
// ---------------------------------------------------------------------------

#[test]
fn test_zero_matrix() {
    let matrix = vec![0.0f32; 12]; // 3x4
    let vec = [1.0, 2.0, 3.0, 4.0];
    let mut out = vec![99.0f32; 3];
    gemv_f32(&matrix, &vec, 3, 4, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.abs() < 1e-6, "zero matrix [{i}] = {v}");
    }
}

#[test]
fn test_zero_vector() {
    let matrix: Vec<f32> = (1..13).map(|i| i as f32).collect(); // 3x4
    let vec = [0.0f32; 4];
    let mut out = vec![99.0f32; 3];
    gemv_f32(&matrix, &vec, 3, 4, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.abs() < 1e-6, "zero vector [{i}] = {v}");
    }
}

// ---------------------------------------------------------------------------
// Large dimension
// ---------------------------------------------------------------------------

#[test]
fn test_large_128x256() {
    let m = 128;
    let k = 256;
    let matrix: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.01).sin()).collect();
    let vec: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.02).cos()).collect();
    let expected = naive_gemv_f64(&matrix, &vec, m, k);
    let mut out = vec![0.0f32; m];
    gemv_f32(&matrix, &vec, m, k, &mut out);
    assert_close(&out, &expected, 5e-3, "dispatch 128x256");
}

// ---------------------------------------------------------------------------
// Empty dimensions
// ---------------------------------------------------------------------------

#[test]
fn test_empty_m_dimension() {
    // m=0: no rows, vec can be non-empty, out is empty.
    let vec = [1.0, 2.0, 3.0, 4.0, 5.0]; // k=5
    let mut out = vec![0.0f32; 0];
    gemv_f32(&[], &vec, 0, 5, &mut out);
    assert!(out.is_empty());
}

#[test]
fn test_empty_k_dimension() {
    // k=0: zero inner dimension. Result should be all zeros.
    let mut out = vec![99.0f32; 3];
    gemv_f32(&[], &[], 3, 0, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.abs() < 1e-6, "k=0 [{i}] = {v}");
    }
}
