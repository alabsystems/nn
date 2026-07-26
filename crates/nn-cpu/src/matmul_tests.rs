// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for the CPU SIMD matrix multiplication implementation.
//!
//! Covers correctness across various matrix shapes, algebraic properties,
//! edge cases, and SIMD vs scalar agreement.

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Naive O(n^3) reference matmul: a[M,K] * b[K,N] -> c[M,N].
fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for kk in 0..k {
                s += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// Transpose a row-major [R,C] matrix to row-major [C,R].
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = data[i * cols + j];
        }
    }
    out
}

/// Create an identity matrix of size n x n.
fn identity(n: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Create a zero matrix of size rows x cols.
fn zeros(rows: usize, cols: usize) -> Vec<f32> {
    vec![0.0f32; rows * cols]
}

/// Generate a deterministic pseudo-random matrix of size rows x cols.
/// Values are in [-2, 2) for reasonable numerical behaviour.
fn pseudo_random_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..rows * cols)
        .map(|_| {
            // Simple xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to [-2, 2)
            ((state as f64 / u64::MAX as f64) * 4.0 - 2.0) as f32
        })
        .collect()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch ({} vs {})", a.len(), b.len());
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: {va} vs {vb} (diff={diff}, tol={tol})"
        );
    }
}

fn assert_all_zero(data: &[f32], tol: f32, label: &str) {
    for (i, &v) in data.iter().enumerate() {
        assert!(
            v.abs() <= tol,
            "{label}[{i}]: expected ~0.0 but got {v} (tol={tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// Square matrices
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_square_2x2() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let c = matmul(&a, &b, 2, 2, 2);
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    assert_close(&c, &[19.0, 22.0, 43.0, 50.0], 1e-6, "square_2x2");
}

#[test]
fn test_matmul_square_4x4() {
    let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let b: Vec<f32> = (17..=32).map(|x| x as f32).collect();
    let expected = naive_matmul(&a, &b, 4, 4, 4);
    let c = matmul(&a, &b, 4, 4, 4);
    assert_close(&c, &expected, 1e-4, "square_4x4");
}

#[test]
fn test_matmul_square_8x8() {
    let a = pseudo_random_matrix(8, 8, 42);
    let b = pseudo_random_matrix(8, 8, 137);
    let expected = naive_matmul(&a, &b, 8, 8, 8);
    let c = matmul(&a, &b, 8, 8, 8);
    assert_close(&c, &expected, 1e-4, "square_8x8");
}

// ---------------------------------------------------------------------------
// Known result: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_known_result() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let c = matmul(&a, &b, 2, 2, 2);
    assert_eq!(c.len(), 4);
    assert!((c[0] - 19.0).abs() < 1e-6, "c[0,0] = {} (expected 19)", c[0]);
    assert!((c[1] - 22.0).abs() < 1e-6, "c[0,1] = {} (expected 22)", c[1]);
    assert!((c[2] - 43.0).abs() < 1e-6, "c[1,0] = {} (expected 43)", c[2]);
    assert!((c[3] - 50.0).abs() < 1e-6, "c[1,1] = {} (expected 50)", c[3]);
}

// ---------------------------------------------------------------------------
// Identity matrix multiplication: A * I = A and I * A = A
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_identity_2x2() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let eye = identity(2);
    let c = matmul(&a, &eye, 2, 2, 2);
    assert_close(&c, &a, 1e-6, "A*I=A 2x2");
}

#[test]
fn test_matmul_identity_left_2x2() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let eye = identity(2);
    let c = matmul(&eye, &a, 2, 2, 2);
    assert_close(&c, &a, 1e-6, "I*A=A 2x2");
}

#[test]
fn test_matmul_identity_8x8() {
    let a = pseudo_random_matrix(8, 8, 99);
    let eye = identity(8);
    let c = matmul(&a, &eye, 8, 8, 8);
    assert_close(&c, &a, 1e-5, "A*I=A 8x8");
}

#[test]
fn test_matmul_identity_large_65x65() {
    // One element beyond TILE=64 to exercise tiling remainder with identity.
    let n = 65;
    let a = pseudo_random_matrix(n, n, 300);
    let eye = identity(n);
    let c = matmul(&a, &eye, n, n, n);
    assert_close(&c, &a, 1e-3, "A*I=A 65x65");
}

// ---------------------------------------------------------------------------
// Rectangular matrices: [M,K] * [K,N]
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_rect_2x3_3x2() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let expected = naive_matmul(&a, &b, 2, 3, 2);
    let c = matmul(&a, &b, 2, 3, 2);
    assert_close(&c, &expected, 1e-4, "rect_2x3_3x2");
}

#[test]
fn test_matmul_rect_4x8_8x3() {
    let a = pseudo_random_matrix(4, 8, 10);
    let b = pseudo_random_matrix(8, 3, 20);
    let expected = naive_matmul(&a, &b, 4, 8, 3);
    let c = matmul(&a, &b, 4, 8, 3);
    assert_close(&c, &expected, 1e-4, "rect_4x8_8x3");
}

#[test]
fn test_matmul_rect_10x20_20x15() {
    let a = pseudo_random_matrix(10, 20, 55);
    let b = pseudo_random_matrix(20, 15, 66);
    let expected = naive_matmul(&a, &b, 10, 20, 15);
    let c = matmul(&a, &b, 10, 20, 15);
    assert_close(&c, &expected, 1e-3, "rect_10x20_20x15");
}

#[test]
fn test_matmul_rect_1x100_100x1() {
    // Wide times tall -> scalar result.
    let a = pseudo_random_matrix(1, 100, 71);
    let b = pseudo_random_matrix(100, 1, 72);
    let expected = naive_matmul(&a, &b, 1, 100, 1);
    let c = matmul(&a, &b, 1, 100, 1);
    assert_close(&c, &expected, 1e-3, "rect_1x100_100x1");
}

// ---------------------------------------------------------------------------
// Non-power-of-2 sizes
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_nonpow2_3x5_5x7() {
    let a = pseudo_random_matrix(3, 5, 111);
    let b = pseudo_random_matrix(5, 7, 222);
    let expected = naive_matmul(&a, &b, 3, 5, 7);
    let c = matmul(&a, &b, 3, 5, 7);
    assert_close(&c, &expected, 1e-4, "nonpow2_3x5_5x7");
}

#[test]
fn test_matmul_nonpow2_7x11_11x13() {
    let a = pseudo_random_matrix(7, 11, 333);
    let b = pseudo_random_matrix(11, 13, 444);
    let expected = naive_matmul(&a, &b, 7, 11, 13);
    let c = matmul(&a, &b, 7, 11, 13);
    assert_close(&c, &expected, 1e-3, "nonpow2_7x11_11x13");
}

#[test]
fn test_matmul_nonpow2_17x23_23x19() {
    let a = pseudo_random_matrix(17, 23, 500);
    let b = pseudo_random_matrix(23, 19, 600);
    let expected = naive_matmul(&a, &b, 17, 23, 19);
    let c = matmul(&a, &b, 17, 23, 19);
    assert_close(&c, &expected, 1e-3, "nonpow2_17x23_23x19");
}

#[test]
fn test_matmul_nonpow2_63x65_65x67() {
    // Just around the TILE=64 boundary.
    let a = pseudo_random_matrix(63, 65, 700);
    let b = pseudo_random_matrix(65, 67, 800);
    let expected = naive_matmul(&a, &b, 63, 65, 67);
    let c = matmul(&a, &b, 63, 65, 67);
    assert_close(&c, &expected, 1e-2, "nonpow2_63x65_65x67");
}

// ---------------------------------------------------------------------------
// Large matrices
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_large_128x128() {
    let (m, k, n) = (128, 128, 128);
    let a = pseudo_random_matrix(m, k, 1000);
    let b = pseudo_random_matrix(k, n, 2000);
    let expected = naive_matmul(&a, &b, m, k, n);
    let c = matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 1e-1, "large_128x128");
}

#[test]
fn test_matmul_large_256x256() {
    let (m, k, n) = (256, 256, 256);
    let a = pseudo_random_matrix(m, k, 3000);
    let b = pseudo_random_matrix(k, n, 4000);
    let expected = naive_matmul(&a, &b, m, k, n);
    let c = matmul(&a, &b, m, k, n);
    // Larger matrices accumulate more FP error; wider tolerance.
    assert_close(&c, &expected, 5e-1, "large_256x256");
}

#[test]
fn test_matmul_large_rect_200x100_100x150() {
    let (m, k, n) = (200, 100, 150);
    let a = pseudo_random_matrix(m, k, 5000);
    let b = pseudo_random_matrix(k, n, 6000);
    let expected = naive_matmul(&a, &b, m, k, n);
    let c = matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 2e-1, "large_rect_200x100x150");
}

// ---------------------------------------------------------------------------
// SIMD matches naive reference
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_tiled_matches_naive_small() {
    for size in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
        let a = pseudo_random_matrix(size, size, size as u64 * 31);
        let b = pseudo_random_matrix(size, size, size as u64 * 37);
        let expected = naive_matmul(&a, &b, size, size, size);
        let c = matmul(&a, &b, size, size, size);
        assert_close(&c, &expected, 1e-3, &format!("simd_vs_naive_{size}x{size}"));
    }
}

#[test]
fn test_matmul_tiled_matches_naive_rect_sweep() {
    let cases = [(2, 3, 4), (5, 7, 3), (8, 1, 8), (1, 8, 1), (6, 10, 4), (9, 3, 11)];
    for (m, k, n) in cases {
        let a = pseudo_random_matrix(m, k, (m * k * n) as u64);
        let b = pseudo_random_matrix(k, n, (m + k + n) as u64);
        let expected = naive_matmul(&a, &b, m, k, n);
        let c = matmul(&a, &b, m, k, n);
        assert_close(
            &c,
            &expected,
            1e-3,
            &format!("simd_vs_naive_{m}x{k}x{n}"),
        );
    }
}

#[test]
fn test_matmul_tiled_scalar_matches_naive() {
    // Directly invoke the scalar path to ensure it also matches.
    let (m, k, n) = (11, 13, 9);
    let a = pseudo_random_matrix(m, k, 77);
    let b = pseudo_random_matrix(k, n, 88);
    let expected = naive_matmul(&a, &b, m, k, n);
    let mut c = vec![0.0f32; m * n];
    matmul_tiled_scalar(&a, &b, &mut c, m, k, n);
    assert_close(&c, &expected, 1e-4, "scalar_matches_naive_11x13x9");
}

// ---------------------------------------------------------------------------
// Zero matrix: A * 0 = 0
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_zero_right() {
    let a = pseudo_random_matrix(5, 7, 12);
    let z = zeros(7, 4);
    let c = matmul(&a, &z, 5, 7, 4);
    assert_all_zero(&c, 1e-7, "A*0=0");
}

#[test]
fn test_matmul_zero_left() {
    let z = zeros(5, 7);
    let b = pseudo_random_matrix(7, 4, 34);
    let c = matmul(&z, &b, 5, 7, 4);
    assert_all_zero(&c, 1e-7, "0*B=0");
}

#[test]
fn test_matmul_zero_both() {
    let za = zeros(3, 4);
    let zb = zeros(4, 5);
    let c = matmul(&za, &zb, 3, 4, 5);
    assert_all_zero(&c, 0.0, "0*0=0");
}

// ---------------------------------------------------------------------------
// Transpose relationship: (A*B)^T = B^T * A^T
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_transpose_property_small() {
    let (m, k, n) = (3, 4, 5);
    let a = pseudo_random_matrix(m, k, 200);
    let b = pseudo_random_matrix(k, n, 201);

    let ab = matmul(&a, &b, m, k, n);
    let ab_t = transpose(&ab, m, n); // [N, M]

    let a_t = transpose(&a, m, k); // [K, M]
    let b_t = transpose(&b, k, n); // [N, K]
    let bt_at = matmul(&b_t, &a_t, n, k, m); // [N, K] * [K, M] = [N, M]

    assert_close(&ab_t, &bt_at, 1e-4, "transpose_property_3x4x5");
}

#[test]
fn test_matmul_transpose_property_square() {
    let n = 16;
    let a = pseudo_random_matrix(n, n, 400);
    let b = pseudo_random_matrix(n, n, 401);

    let ab = matmul(&a, &b, n, n, n);
    let ab_t = transpose(&ab, n, n);

    let a_t = transpose(&a, n, n);
    let b_t = transpose(&b, n, n);
    let bt_at = matmul(&b_t, &a_t, n, n, n);

    assert_close(&ab_t, &bt_at, 1e-2, "transpose_property_16x16");
}

#[test]
fn test_matmul_transpose_property_large() {
    let (m, k, n) = (33, 47, 29);
    let a = pseudo_random_matrix(m, k, 500);
    let b = pseudo_random_matrix(k, n, 501);

    let ab = matmul(&a, &b, m, k, n);
    let ab_t = transpose(&ab, m, n);

    let a_t = transpose(&a, m, k);
    let b_t = transpose(&b, k, n);
    let bt_at = matmul(&b_t, &a_t, n, k, m);

    assert_close(&ab_t, &bt_at, 1e-1, "transpose_property_33x47x29");
}

// ---------------------------------------------------------------------------
// Associativity: (A*B)*C ~= A*(B*C) within epsilon
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_associativity_small() {
    let (p, q, r, s) = (3, 4, 5, 2);
    let a = pseudo_random_matrix(p, q, 600);
    let b = pseudo_random_matrix(q, r, 601);
    let c_mat = pseudo_random_matrix(r, s, 602);

    // (A*B)*C
    let ab = matmul(&a, &b, p, q, r);
    let ab_c = matmul(&ab, &c_mat, p, r, s);

    // A*(B*C)
    let bc = matmul(&b, &c_mat, q, r, s);
    let a_bc = matmul(&a, &bc, p, q, s);

    assert_close(&ab_c, &a_bc, 1e-3, "associativity_3x4x5x2");
}

#[test]
fn test_matmul_associativity_larger() {
    let (p, q, r, s) = (16, 20, 12, 8);
    let a = pseudo_random_matrix(p, q, 700);
    let b = pseudo_random_matrix(q, r, 701);
    let c_mat = pseudo_random_matrix(r, s, 702);

    let ab = matmul(&a, &b, p, q, r);
    let ab_c = matmul(&ab, &c_mat, p, r, s);

    let bc = matmul(&b, &c_mat, q, r, s);
    let a_bc = matmul(&a, &bc, p, q, s);

    assert_close(&ab_c, &a_bc, 5e-2, "associativity_16x20x12x8");
}

// ---------------------------------------------------------------------------
// Edge case: 1x1 matrices (scalar multiply)
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_1x1_scalar_multiply() {
    let a = vec![3.0f32];
    let b = vec![7.0f32];
    let c = matmul(&a, &b, 1, 1, 1);
    assert_eq!(c.len(), 1);
    assert!((c[0] - 21.0).abs() < 1e-7, "1x1: 3*7 = {} (expected 21)", c[0]);
}

#[test]
fn test_matmul_1x1_negative() {
    let a = vec![-2.5f32];
    let b = vec![4.0f32];
    let c = matmul(&a, &b, 1, 1, 1);
    assert!((c[0] - (-10.0)).abs() < 1e-7, "1x1 negative");
}

#[test]
fn test_matmul_1x1_zero() {
    let a = vec![0.0f32];
    let b = vec![42.0f32];
    let c = matmul(&a, &b, 1, 1, 1);
    assert!((c[0]).abs() < 1e-7, "1x1 zero");
}

// ---------------------------------------------------------------------------
// Edge case: Mx1 * 1xN (outer product)
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_outer_product_3x4() {
    // a = column [1, 2, 3], b = row [4, 5, 6, 7]
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0, 7.0];
    let c = matmul(&a, &b, 3, 1, 4);
    let expected = vec![
        4.0, 5.0, 6.0, 7.0,   // 1 * [4,5,6,7]
        8.0, 10.0, 12.0, 14.0, // 2 * [4,5,6,7]
        12.0, 15.0, 18.0, 21.0, // 3 * [4,5,6,7]
    ];
    assert_close(&c, &expected, 1e-6, "outer_product_3x4");
}

#[test]
fn test_matmul_outer_product_large() {
    let m = 50;
    let n = 30;
    let a = pseudo_random_matrix(m, 1, 800);
    let b = pseudo_random_matrix(1, n, 801);
    let expected = naive_matmul(&a, &b, m, 1, n);
    let c = matmul(&a, &b, m, 1, n);
    assert_close(&c, &expected, 1e-6, "outer_product_50x30");
}

// ---------------------------------------------------------------------------
// Edge case: 1xK * Kx1 (dot product -> scalar)
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_dot_product_small() {
    // [1, 2, 3] . [4, 5, 6] = 4 + 10 + 18 = 32
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let c = matmul(&a, &b, 1, 3, 1);
    assert_eq!(c.len(), 1);
    assert!((c[0] - 32.0).abs() < 1e-6, "dot: 1x3 * 3x1 = {} (expected 32)", c[0]);
}

#[test]
fn test_matmul_dot_product_medium() {
    let k = 64;
    let a = pseudo_random_matrix(1, k, 900);
    let b = pseudo_random_matrix(k, 1, 901);
    let expected = naive_matmul(&a, &b, 1, k, 1);
    let c = matmul(&a, &b, 1, k, 1);
    assert_close(&c, &expected, 1e-3, "dot_product_k64");
}

#[test]
fn test_matmul_dot_product_large() {
    let k = 256;
    let a = pseudo_random_matrix(1, k, 950);
    let b = pseudo_random_matrix(k, 1, 951);
    let expected = naive_matmul(&a, &b, 1, k, 1);
    let c = matmul(&a, &b, 1, k, 1);
    assert_close(&c, &expected, 1e-2, "dot_product_k256");
}

// ---------------------------------------------------------------------------
// matmul_tiled: pre-zeroed output accumulation
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_tiled_accumulates() {
    // matmul_tiled accumulates into c (c must be pre-zeroed).
    // Calling twice should double the result.
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0f32; 4];
    matmul_tiled(&a, &b, &mut c, 2, 2, 2);
    matmul_tiled(&a, &b, &mut c, 2, 2, 2);
    // Expected = 2 * [[19,22],[43,50]] = [[38,44],[86,100]]
    assert_close(&c, &[38.0, 44.0, 86.0, 100.0], 1e-5, "tiled_accumulates");
}

// ---------------------------------------------------------------------------
// matmul_with_transposed_b
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_with_transposed_b_small() {
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    // B^T = [[7,9,11],[8,10,12]]
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_t = vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0];
    let mut c = vec![0.0f32; 4];
    matmul_with_transposed_b(&a, &b_t, &mut c, 2, 3, 2);
    assert_close(&c, &[58.0, 64.0, 139.0, 154.0], 1e-4, "transposed_b_small");
}

#[test]
fn test_matmul_with_transposed_b_matches_regular() {
    let (m, k, n) = (10, 15, 8);
    let a = pseudo_random_matrix(m, k, 1100);
    let b = pseudo_random_matrix(k, n, 1200);
    let b_t = transpose(&b, k, n); // [N, K]

    let regular = matmul(&a, &b, m, k, n);
    let mut transposed = vec![0.0f32; m * n];
    matmul_with_transposed_b(&a, &b_t, &mut transposed, m, k, n);

    assert_close(&regular, &transposed, 1e-3, "transposed_b_matches_regular");
}

// ---------------------------------------------------------------------------
// Degenerate dimensions: zero-sized
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_tiled_zero_m() {
    let a: Vec<f32> = vec![];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut c: Vec<f32> = vec![];
    matmul_tiled(&a, &b, &mut c, 0, 2, 2);
    assert!(c.is_empty());
}

#[test]
fn test_matmul_tiled_zero_k() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let mut c = vec![99.0f32; 4]; // Should remain unchanged since k=0.
    matmul_tiled(&a, &b, &mut c, 2, 0, 2);
    // c should be untouched because the function returns early.
    assert_close(&c, &[99.0, 99.0, 99.0, 99.0], 0.0, "zero_k_unchanged");
}

#[test]
fn test_matmul_tiled_zero_n() {
    let a = vec![1.0, 2.0];
    let b: Vec<f32> = vec![];
    let mut c: Vec<f32> = vec![];
    matmul_tiled(&a, &b, &mut c, 1, 2, 0);
    assert!(c.is_empty());
}

// ---------------------------------------------------------------------------
// Tile boundary coverage (TILE = 64)
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_exact_tile_64x64() {
    let n = 64;
    let a = pseudo_random_matrix(n, n, 1300);
    let b = pseudo_random_matrix(n, n, 1400);
    let expected = naive_matmul(&a, &b, n, n, n);
    let c = matmul(&a, &b, n, n, n);
    assert_close(&c, &expected, 5e-2, "exact_tile_64x64");
}

#[test]
fn test_matmul_two_tiles_128x64_64x128() {
    let (m, k, n) = (128, 64, 128);
    let a = pseudo_random_matrix(m, k, 1500);
    let b = pseudo_random_matrix(k, n, 1600);
    let expected = naive_matmul(&a, &b, m, k, n);
    let c = matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 2e-1, "two_tiles_128x64x128");
}

// ---------------------------------------------------------------------------
// Numerical: all-ones matrix
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_all_ones() {
    // [M,K] of ones * [K,N] of ones = [M,N] where each element = K.
    let (m, k, n) = (5, 7, 3);
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let c = matmul(&a, &b, m, k, n);
    let expected = vec![k as f32; m * n];
    assert_close(&c, &expected, 1e-6, "all_ones_5x7x3");
}

// ---------------------------------------------------------------------------
// Distributivity: A * (B + C) ~= A*B + A*C
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_distributivity() {
    let (m, k, n) = (6, 8, 5);
    let a = pseudo_random_matrix(m, k, 1700);
    let b = pseudo_random_matrix(k, n, 1701);
    let c_mat = pseudo_random_matrix(k, n, 1702);

    // B + C
    let b_plus_c: Vec<f32> = b.iter().zip(c_mat.iter()).map(|(x, y)| x + y).collect();

    let a_bpc = matmul(&a, &b_plus_c, m, k, n);
    let ab = matmul(&a, &b, m, k, n);
    let ac = matmul(&a, &c_mat, m, k, n);
    let ab_plus_ac: Vec<f32> = ab.iter().zip(ac.iter()).map(|(x, y)| x + y).collect();

    assert_close(&a_bpc, &ab_plus_ac, 1e-3, "distributivity_6x8x5");
}

// ---------------------------------------------------------------------------
// Scalar scaling: (sA) * B = s(A*B)
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_scalar_scaling() {
    let (m, k, n) = (4, 6, 3);
    let a = pseudo_random_matrix(m, k, 1800);
    let b = pseudo_random_matrix(k, n, 1801);
    let s = 2.5f32;

    let sa: Vec<f32> = a.iter().map(|x| x * s).collect();
    let sa_b = matmul(&sa, &b, m, k, n);

    let ab = matmul(&a, &b, m, k, n);
    let s_ab: Vec<f32> = ab.iter().map(|x| x * s).collect();

    assert_close(&sa_b, &s_ab, 1e-3, "scalar_scaling_4x6x3");
}

// ---------------------------------------------------------------------------
// dot_contiguous
// ---------------------------------------------------------------------------

#[test]
fn test_dot_contiguous_simple() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = dot_contiguous(&a, &b);
    assert!((result - 32.0).abs() < 1e-6, "dot_contiguous: {} (expected 32)", result);
}

#[test]
fn test_dot_contiguous_empty() {
    let result = dot_contiguous(&[], &[]);
    assert!((result - 0.0).abs() < 1e-7, "dot_contiguous empty");
}

#[test]
fn test_dot_contiguous_single() {
    let result = dot_contiguous(&[3.0], &[7.0]);
    assert!((result - 21.0).abs() < 1e-6, "dot_contiguous single");
}
