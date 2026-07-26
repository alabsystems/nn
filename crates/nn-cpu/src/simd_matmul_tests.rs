// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD-optimized matmul with explicit tier entry points.

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Naive f64 reference matmul for ground truth.
fn naive_matmul_f64(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for p in 0..k {
                acc += f64::from(a[i * k + p]) * f64::from(b[p * n + j]);
            }
            out[i * n + j] = acc as f32;
        }
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
// Identity matrix multiply
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_identity_2x2() {
    let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
    let eye = [1.0, 0.0, 0.0, 1.0]; // 2x2 identity
    let mut out = vec![0.0f32; 4];
    matmul_f32_scalar(&a, &eye, 2, 2, 2, &mut out);
    assert_close(&out, &a, 1e-6, "scalar identity 2x2");
}

#[test]
fn test_dispatch_identity_4x4() {
    let mut a = vec![0.0f32; 16];
    let mut eye = vec![0.0f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            a[i * 4 + j] = (i * 4 + j + 1) as f32;
            if i == j {
                eye[i * 4 + j] = 1.0;
            }
        }
    }
    let mut out = vec![0.0f32; 16];
    matmul_f32(&a, &eye, 4, 4, 4, &mut out);
    assert_close(&out, &a, 1e-5, "dispatch identity 4x4");
}

#[test]
fn test_neon_identity_3x3() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut out = vec![0.0f32; 9];
    matmul_f32_neon(&a, &eye, 3, 3, 3, &mut out);
    assert_close(&out, &a, 1e-5, "neon identity 3x3");
}

#[test]
fn test_avx2_identity_3x3() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut out = vec![0.0f32; 9];
    matmul_f32_avx2(&a, &eye, 3, 3, 3, &mut out);
    assert_close(&out, &a, 1e-5, "avx2 identity 3x3");
}

// ---------------------------------------------------------------------------
// Small known values
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_known_2x3_times_3x2() {
    // A = [[1,2,3],[4,5,6]] (2x3)
    // B = [[7,8],[9,10],[11,12]] (3x2)
    // C = [[1*7+2*9+3*11, 1*8+2*10+3*12],
    //      [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //   = [[58, 64], [139, 154]]
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let expected = [58.0, 64.0, 139.0, 154.0];
    let mut out = vec![0.0f32; 4];
    matmul_f32_scalar(&a, &b, 2, 3, 2, &mut out);
    assert_close(&out, &expected, 1e-5, "scalar 2x3 * 3x2");
}

#[test]
fn test_dispatch_known_2x3_times_3x2() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let expected = [58.0, 64.0, 139.0, 154.0];
    let mut out = vec![0.0f32; 4];
    matmul_f32(&a, &b, 2, 3, 2, &mut out);
    assert_close(&out, &expected, 1e-4, "dispatch 2x3 * 3x2");
}

// ---------------------------------------------------------------------------
// Non-square matrices
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_non_square_1x4_times_4x1() {
    // Row vector times column vector = scalar (1x1).
    let a = [1.0, 2.0, 3.0, 4.0]; // 1x4
    let b = [5.0, 6.0, 7.0, 8.0]; // 4x1
    let expected = [1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0]; // = 70
    let mut out = vec![0.0f32; 1];
    matmul_f32_scalar(&a, &b, 1, 4, 1, &mut out);
    assert_close(&out, &expected, 1e-5, "scalar 1x4 * 4x1");
}

#[test]
fn test_dispatch_non_square_4x1_times_1x4() {
    // Column vector times row vector = 4x4 outer product.
    let a = [1.0, 2.0, 3.0, 4.0]; // 4x1
    let b = [5.0, 6.0, 7.0, 8.0]; // 1x4
    let expected = [
        5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 15.0, 18.0, 21.0, 24.0, 20.0, 24.0, 28.0, 32.0,
    ];
    let mut out = vec![0.0f32; 16];
    matmul_f32(&a, &b, 4, 1, 4, &mut out);
    assert_close(&out, &expected, 1e-4, "dispatch 4x1 * 1x4");
}

#[test]
fn test_non_square_5x3_times_3x7() {
    let m = 5;
    let k = 3;
    let n = 7;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.2).collect();
    let expected = naive_matmul_f64(&a, &b, m, k, n);
    let mut out = vec![0.0f32; m * n];
    matmul_f32(&a, &b, m, k, n, &mut out);
    assert_close(&out, &expected, 1e-4, "dispatch 5x3 * 3x7");
}

// ---------------------------------------------------------------------------
// Single element (1x1 * 1x1)
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_single_element() {
    let a = [3.0_f32];
    let b = [7.0_f32];
    let mut out = vec![0.0f32; 1];
    matmul_f32_scalar(&a, &b, 1, 1, 1, &mut out);
    assert_close(&out, &[21.0], 1e-6, "scalar 1x1");
}

#[test]
fn test_dispatch_single_element() {
    let a = [4.5_f32];
    let b = [2.0_f32];
    let mut out = vec![0.0f32; 1];
    matmul_f32(&a, &b, 1, 1, 1, &mut out);
    assert_close(&out, &[9.0], 1e-5, "dispatch 1x1");
}

// ---------------------------------------------------------------------------
// SIMD vs reference comparison
// ---------------------------------------------------------------------------

#[test]
fn test_neon_vs_reference_8x8() {
    let m = 8;
    let k = 8;
    let n = 8;
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.37).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.53).cos()).collect();
    let expected = matmul_reference(&a, &b, m, k, n);
    let mut out = vec![0.0f32; m * n];
    matmul_f32_neon(&a, &b, m, k, n, &mut out);
    assert_close(&out, &expected, 1e-4, "neon vs reference 8x8");
}

#[test]
fn test_avx2_vs_reference_8x8() {
    let m = 8;
    let k = 8;
    let n = 8;
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.37).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.53).cos()).collect();
    let expected = matmul_reference(&a, &b, m, k, n);
    let mut out = vec![0.0f32; m * n];
    matmul_f32_avx2(&a, &b, m, k, n, &mut out);
    assert_close(&out, &expected, 1e-4, "avx2 vs reference 8x8");
}

#[test]
fn test_dispatch_vs_reference_16x16() {
    let m = 16;
    let k = 16;
    let n = 16;
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.11).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.23).cos()).collect();
    let expected = naive_matmul_f64(&a, &b, m, k, n);
    let mut out = vec![0.0f32; m * n];
    matmul_f32(&a, &b, m, k, n, &mut out);
    assert_close(&out, &expected, 1e-3, "dispatch vs naive 16x16");
}

// ---------------------------------------------------------------------------
// Varied sizes (exercises SIMD tail handling)
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_varied_sizes() {
    for (m, k, n) in [
        (1, 1, 1),
        (2, 3, 4),
        (3, 5, 2),
        (4, 4, 4),
        (5, 7, 3),
        (7, 8, 9),
        (8, 8, 8),
        (9, 11, 13),
        (15, 16, 17),
        (32, 32, 32),
    ] {
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.3).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.7).cos()).collect();
        let expected = naive_matmul_f64(&a, &b, m, k, n);
        let mut out = vec![0.0f32; m * n];
        matmul_f32(&a, &b, m, k, n, &mut out);
        assert_close(
            &out,
            &expected,
            1e-3,
            &format!("dispatch varied {m}x{k} * {k}x{n}"),
        );
    }
}

// ---------------------------------------------------------------------------
// Reference function returns correct output
// ---------------------------------------------------------------------------

#[test]
fn test_reference_matches_scalar() {
    let a = [1.0, 0.5, 0.5, 1.0]; // 2x2
    let b = [2.0, 1.0, 1.0, 2.0]; // 2x2
    let ref_out = matmul_reference(&a, &b, 2, 2, 2);
    let mut scalar_out = vec![0.0f32; 4];
    matmul_f32_scalar(&a, &b, 2, 2, 2, &mut scalar_out);
    assert_close(&ref_out, &scalar_out, 1e-7, "reference vs scalar");
}

// ---------------------------------------------------------------------------
// Zero matrices
// ---------------------------------------------------------------------------

#[test]
fn test_zero_matrix() {
    let a = vec![0.0f32; 9];
    let b: Vec<f32> = (1..10).map(|i| i as f32).collect();
    let mut out = vec![99.0f32; 9]; // pre-fill with non-zero
    matmul_f32(&a, &b, 3, 3, 3, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!((v).abs() < 1e-6, "zero matrix [{i}] = {v}");
    }
}

// ---------------------------------------------------------------------------
// Large matrix
// ---------------------------------------------------------------------------

#[test]
fn test_large_matrix_64x64() {
    let m = 64;
    let k = 64;
    let n = 64;
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.01).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.02).cos()).collect();
    let expected = naive_matmul_f64(&a, &b, m, k, n);
    let mut out = vec![0.0f32; m * n];
    matmul_f32(&a, &b, m, k, n, &mut out);
    // Larger tolerance for bigger matrices due to f32 accumulation.
    assert_close(&out, &expected, 5e-3, "dispatch 64x64");
}

// ---------------------------------------------------------------------------
// Empty dimensions are handled
// ---------------------------------------------------------------------------

#[test]
fn test_empty_m_dimension() {
    // m=0: A is empty, B can be anything, out is empty.
    let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let mut out = vec![0.0f32; 0];
    matmul_f32(&[], &b, 0, 2, 3, &mut out);
    assert!(out.is_empty());
}

#[test]
fn test_empty_k_dimension() {
    // k=0: A and B both have zero inner dim.
    let mut out = vec![0.0f32; 6]; // 2x3
    matmul_f32(&[], &[], 2, 0, 3, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.abs() < 1e-6, "k=0 [{i}] = {v}");
    }
}

#[test]
fn test_empty_n_dimension() {
    // n=0: output has zero columns.
    let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
    let mut out = vec![0.0f32; 0];
    matmul_f32(&a, &[], 2, 2, 0, &mut out);
    assert!(out.is_empty());
}
