// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests verifying SIMD-dispatched operations match scalar reference implementations.

use crate::elementwise;
use crate::matmul;
use crate::reduction;
use crate::simd_detect;

// ---------------------------------------------------------------------------
// SIMD detection
// ---------------------------------------------------------------------------

#[test]
fn test_simd_detect_returns_valid_level() {
    let level = simd_detect::detect();
    // On aarch64 (Apple Silicon) we expect Neon.
    // On x86_64 with AVX2 we expect Avx2.
    // Anywhere else: Scalar.
    let lanes = simd_detect::lane_count(level);
    assert!(
        lanes >= 1,
        "lane_count must be >= 1, got {lanes} for {level:?}"
    );
}

// ---------------------------------------------------------------------------
// Elementwise: compare dispatched vs scalar
// ---------------------------------------------------------------------------

fn make_test_input() -> Vec<f32> {
    // Range covering negative, zero, positive, and large values.
    vec![
        -10.0, -5.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0,
        // Extra elements to exercise SIMD tail handling (non-multiple of 4/8).
        -3.14, 0.707, 2.718,
    ]
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, op_name: &str) {
    assert_eq!(a.len(), b.len(), "{op_name}: length mismatch");
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        assert!(
            diff <= tol,
            "{op_name}[{i}]: {va} vs {vb} (diff={diff}, tol={tol})"
        );
    }
}

#[test]
fn test_relu_simd_matches_scalar() {
    let input = make_test_input();
    let mut scalar_out = vec![0.0f32; input.len()];
    let mut simd_out = vec![0.0f32; input.len()];
    elementwise::relu_scalar(&input, &mut scalar_out);
    elementwise::relu(&input, &mut simd_out);
    assert_close(&scalar_out, &simd_out, 0.0, "relu");
}

#[test]
fn test_silu_simd_matches_scalar() {
    let input = make_test_input();
    let mut scalar_out = vec![0.0f32; input.len()];
    let mut simd_out = vec![0.0f32; input.len()];
    elementwise::silu_scalar(&input, &mut scalar_out);
    elementwise::silu(&input, &mut simd_out);
    assert_close(&scalar_out, &simd_out, 1e-6, "silu");
}

#[test]
fn test_sigmoid_simd_matches_scalar() {
    let input = make_test_input();
    let mut scalar_out = vec![0.0f32; input.len()];
    let mut simd_out = vec![0.0f32; input.len()];
    elementwise::sigmoid_scalar(&input, &mut scalar_out);
    elementwise::sigmoid(&input, &mut simd_out);
    assert_close(&scalar_out, &simd_out, 1e-6, "sigmoid");
}

#[test]
fn test_tanh_simd_matches_scalar() {
    let input = make_test_input();
    let mut scalar_out = vec![0.0f32; input.len()];
    let mut simd_out = vec![0.0f32; input.len()];
    elementwise::tanh_scalar(&input, &mut scalar_out);
    elementwise::tanh(&input, &mut simd_out);
    assert_close(&scalar_out, &simd_out, 1e-6, "tanh");
}

#[test]
fn test_gelu_simd_matches_scalar() {
    let input = make_test_input();
    let mut scalar_out = vec![0.0f32; input.len()];
    let mut simd_out = vec![0.0f32; input.len()];
    elementwise::gelu_scalar(&input, &mut scalar_out);
    elementwise::gelu(&input, &mut simd_out);
    assert_close(&scalar_out, &simd_out, 1e-6, "gelu");
}

#[test]
fn test_add_simd_matches_scalar() {
    let a = make_test_input();
    let b: Vec<f32> = a.iter().map(|x| x * 0.5 + 1.0).collect();
    let mut scalar_out = vec![0.0f32; a.len()];
    let mut simd_out = vec![0.0f32; a.len()];
    elementwise::add_scalar(&a, &b, &mut scalar_out);
    elementwise::add(&a, &b, &mut simd_out);
    assert_close(&scalar_out, &simd_out, 0.0, "add");
}

#[test]
fn test_mul_simd_matches_scalar() {
    let a = make_test_input();
    let b: Vec<f32> = a.iter().map(|x| x * 0.3 - 2.0).collect();
    let mut scalar_out = vec![0.0f32; a.len()];
    let mut simd_out = vec![0.0f32; a.len()];
    elementwise::mul_scalar(&a, &b, &mut scalar_out);
    elementwise::mul(&a, &b, &mut simd_out);
    assert_close(&scalar_out, &simd_out, 0.0, "mul");
}

#[test]
fn test_elementwise_empty_input() {
    let input: Vec<f32> = vec![];
    let mut output: Vec<f32> = vec![];
    elementwise::relu(&input, &mut output);
    elementwise::silu(&input, &mut output);
    elementwise::sigmoid(&input, &mut output);
    elementwise::tanh(&input, &mut output);
    elementwise::gelu(&input, &mut output);
    elementwise::add(&input, &input, &mut output);
    elementwise::mul(&input, &input, &mut output);
    // No panic = pass.
}

#[test]
fn test_elementwise_single_element() {
    let input = vec![1.5f32];
    let mut out = vec![0.0f32; 1];
    elementwise::relu(&input, &mut out);
    assert!((out[0] - 1.5).abs() < 1e-7, "relu(1.5) should be 1.5");
}

// ---------------------------------------------------------------------------
// Reductions: compare dispatched vs scalar
// ---------------------------------------------------------------------------

fn make_reduction_input() -> Vec<f32> {
    vec![
        1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0, 13.0, -14.0, 15.0,
        0.5, -0.25,
    ]
}

#[test]
fn test_sum_simd_matches_scalar() {
    let input = make_reduction_input();
    let scalar = reduction::sum_scalar(&input);
    let simd = reduction::sum(&input);
    assert!(
        (scalar - simd).abs() < 1e-4,
        "sum: scalar={scalar} vs simd={simd}"
    );
}

#[test]
fn test_max_simd_matches_scalar() {
    let input = make_reduction_input();
    let scalar = reduction::max_scalar(&input);
    let simd = reduction::max(&input);
    assert!(
        (scalar - simd).abs() < 1e-7,
        "max: scalar={scalar} vs simd={simd}"
    );
}

#[test]
fn test_dot_simd_matches_scalar() {
    let a = make_reduction_input();
    let b: Vec<f32> = a.iter().map(|x| x * 0.5 + 1.0).collect();
    let scalar = reduction::dot_scalar(&a, &b);
    let simd = reduction::dot(&a, &b);
    assert!(
        (scalar - simd).abs() < 1e-3,
        "dot: scalar={scalar} vs simd={simd}"
    );
}

#[test]
fn test_sum_empty() {
    assert_eq!(reduction::sum(&[]), 0.0);
}

#[test]
fn test_max_empty() {
    assert_eq!(reduction::max(&[]), f32::NEG_INFINITY);
}

#[test]
fn test_dot_empty() {
    assert_eq!(reduction::dot(&[], &[]), 0.0);
}

// ---------------------------------------------------------------------------
// Matmul
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_identity() {
    // A = [[1,2],[3,4]], B = I => C = A
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 0.0, 0.0, 1.0];
    let c = matmul::matmul(&a, &b, 2, 2, 2);
    assert_close(&c, &[1.0, 2.0, 3.0, 4.0], 1e-6, "matmul_identity");
}

#[test]
fn test_matmul_2x3_3x2() {
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    // C = [[58,64],[139,154]]
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let c = matmul::matmul(&a, &b, 2, 3, 2);
    assert_close(&c, &[58.0, 64.0, 139.0, 154.0], 1e-4, "matmul_2x3_3x2");
}

#[test]
fn test_matmul_large_tiled() {
    // 128x128 matmul to exercise tiling (TILE=64, so 4 tiles per dimension).
    let m = 128;
    let k = 128;
    let n = 128;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 13) as f32 * 0.1).collect();
    let c = matmul::matmul(&a, &b, m, k, n);

    // Verify against naive O(n^3).
    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for kk in 0..k {
                s += a[i * k + kk] * b[kk * n + j];
            }
            expected[i * n + j] = s;
        }
    }
    assert_close(&c, &expected, 1e-2, "matmul_large_tiled");
}

#[test]
fn test_matmul_transposed_b() {
    // A = [[1,2,3],[4,5,6]], B^T = [[7,9,11],[8,10,12]]
    // Same as test_matmul_2x3_3x2 but B transposed.
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_t = vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0]; // B^T stored row-major
    let mut c = vec![0.0f32; 4];
    matmul::matmul_with_transposed_b(&a, &b_t, &mut c, 2, 3, 2);
    assert_close(&c, &[58.0, 64.0, 139.0, 154.0], 1e-4, "matmul_transposed_b");
}

#[test]
fn test_matmul_non_tile_aligned() {
    // 7x5 * 5x3 — not aligned to TILE=64.
    let m = 7;
    let k = 5;
    let n = 3;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.3 - 5.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.2 - 1.0).collect();
    let c = matmul::matmul(&a, &b, m, k, n);

    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for kk in 0..k {
                s += a[i * k + kk] * b[kk * n + j];
            }
            expected[i * n + j] = s;
        }
    }
    assert_close(&c, &expected, 1e-3, "matmul_non_tile_aligned");
}
