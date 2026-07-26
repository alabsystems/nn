// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive CPU backend parity and edge-case tests.
//!
//! Validates SIMD-dispatched paths match scalar references across:
//! - All activations at multiple array sizes (tail handling)
//! - Reduction operations with dimensional variations
//! - MatMul correctness (square, rectangular, batched-via-loop, transposed)
//! - Edge cases: empty, single, denormal, NaN, Inf, large arrays

use nn_cpu::elementwise;
use nn_cpu::matmul;
use nn_cpu::reduction;
use nn_cpu::simd_detect;

// ============================================================================
// Helpers
// ============================================================================

/// Assert element-wise closeness within tolerance.
fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        // Both NaN: considered equal for propagation tests.
        if va.is_nan() && vb.is_nan() {
            continue;
        }
        let diff = (va - vb).abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: {va} vs {vb} (diff={diff}, tol={tol})"
        );
    }
}

/// Naive triple-loop matmul reference: A[M,K] * B[K,N] -> C[M,N].
fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Generate a deterministic input of `n` elements spanning negative, zero, positive.
fn make_input(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = (i as f32) / (n.max(1) as f32) * 2.0 - 1.0; // [-1, 1)
            t * 10.0 // [-10, 10)
        })
        .collect()
}

/// Run an activation through both scalar and SIMD dispatch, assert parity.
fn check_activation_parity(
    name: &str,
    scalar_fn: fn(&[f32], &mut [f32]),
    simd_fn: fn(&[f32], &mut [f32]),
    input: &[f32],
    tol: f32,
) {
    let n = input.len();
    let mut scalar_out = vec![0.0f32; n];
    let mut simd_out = vec![0.0f32; n];
    scalar_fn(input, &mut scalar_out);
    simd_fn(input, &mut simd_out);
    assert_close(&scalar_out, &simd_out, tol, name);
}

// ============================================================================
// A. Elementwise Activation Parity (25+ tests)
// ============================================================================

// -- ReLU --

#[test]
fn parity_relu_small() {
    let input = vec![-2.0, -0.5, 0.0, 1.5];
    check_activation_parity(
        "relu_small",
        elementwise::relu_scalar,
        elementwise::relu,
        &input,
        0.0,
    );
}

#[test]
fn parity_relu_simd_aligned() {
    // 256 elements: multiple of NEON(4) and AVX2(8).
    let input = make_input(256);
    check_activation_parity(
        "relu_256",
        elementwise::relu_scalar,
        elementwise::relu,
        &input,
        0.0,
    );
}

#[test]
fn parity_relu_simd_tail() {
    // 257 = 64*4 + 1 — exercises NEON tail (1 extra).
    let input = make_input(257);
    check_activation_parity(
        "relu_257",
        elementwise::relu_scalar,
        elementwise::relu,
        &input,
        0.0,
    );
}

#[test]
fn parity_relu_three_elements() {
    // Fewer than one NEON lane width.
    let input = vec![-1.0, 0.0, 1.0];
    check_activation_parity(
        "relu_3",
        elementwise::relu_scalar,
        elementwise::relu,
        &input,
        0.0,
    );
}

#[test]
fn parity_relu_five_elements() {
    // NEON: 1 chunk + 1 tail. AVX2: 0 chunks + 5 tail.
    let input = vec![-5.0, -1.0, 0.0, 1.0, 5.0];
    check_activation_parity(
        "relu_5",
        elementwise::relu_scalar,
        elementwise::relu,
        &input,
        0.0,
    );
}

// -- SiLU --

#[test]
fn parity_silu_small() {
    let input = vec![-2.0, -0.5, 0.0, 1.5];
    check_activation_parity(
        "silu_small",
        elementwise::silu_scalar,
        elementwise::silu,
        &input,
        1e-6,
    );
}

#[test]
fn parity_silu_simd_aligned() {
    let input = make_input(256);
    check_activation_parity(
        "silu_256",
        elementwise::silu_scalar,
        elementwise::silu,
        &input,
        1e-6,
    );
}

#[test]
fn parity_silu_simd_tail() {
    let input = make_input(259); // 259 = 64*4 + 3
    check_activation_parity(
        "silu_259",
        elementwise::silu_scalar,
        elementwise::silu,
        &input,
        1e-6,
    );
}

// -- Sigmoid --

#[test]
fn parity_sigmoid_small() {
    let input = vec![-10.0, -1.0, 0.0, 1.0, 10.0];
    check_activation_parity(
        "sigmoid_small",
        elementwise::sigmoid_scalar,
        elementwise::sigmoid,
        &input,
        1e-6,
    );
}

#[test]
fn parity_sigmoid_simd_aligned() {
    let input = make_input(256);
    check_activation_parity(
        "sigmoid_256",
        elementwise::sigmoid_scalar,
        elementwise::sigmoid,
        &input,
        1e-6,
    );
}

#[test]
fn parity_sigmoid_simd_tail() {
    let input = make_input(255); // 255 = 63*4 + 3
    check_activation_parity(
        "sigmoid_255",
        elementwise::sigmoid_scalar,
        elementwise::sigmoid,
        &input,
        1e-6,
    );
}

#[test]
fn parity_sigmoid_extreme_values() {
    // Sigmoid saturates near 0 and 1 for large magnitude inputs.
    let input = vec![-100.0, -50.0, -20.0, 0.0, 20.0, 50.0, 100.0];
    check_activation_parity(
        "sigmoid_extreme",
        elementwise::sigmoid_scalar,
        elementwise::sigmoid,
        &input,
        1e-6,
    );
}

// -- Tanh --

#[test]
fn parity_tanh_small() {
    let input = vec![-5.0, -1.0, 0.0, 1.0, 5.0];
    check_activation_parity(
        "tanh_small",
        elementwise::tanh_scalar,
        elementwise::tanh,
        &input,
        1e-6,
    );
}

#[test]
fn parity_tanh_simd_aligned() {
    let input = make_input(256);
    check_activation_parity(
        "tanh_256",
        elementwise::tanh_scalar,
        elementwise::tanh,
        &input,
        1e-6,
    );
}

#[test]
fn parity_tanh_simd_tail() {
    let input = make_input(258); // 258 = 64*4 + 2
    check_activation_parity(
        "tanh_258",
        elementwise::tanh_scalar,
        elementwise::tanh,
        &input,
        1e-6,
    );
}

// -- GELU --

#[test]
fn parity_gelu_small() {
    let input = vec![-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];
    check_activation_parity(
        "gelu_small",
        elementwise::gelu_scalar,
        elementwise::gelu,
        &input,
        1e-5,
    );
}

#[test]
fn parity_gelu_simd_aligned() {
    let input = make_input(256);
    check_activation_parity(
        "gelu_256",
        elementwise::gelu_scalar,
        elementwise::gelu,
        &input,
        1e-5,
    );
}

#[test]
fn parity_gelu_simd_tail() {
    let input = make_input(261); // 261 = 65*4 + 1
    check_activation_parity(
        "gelu_261",
        elementwise::gelu_scalar,
        elementwise::gelu,
        &input,
        1e-5,
    );
}

#[test]
fn parity_gelu_one_element() {
    let input = vec![0.0];
    check_activation_parity(
        "gelu_1",
        elementwise::gelu_scalar,
        elementwise::gelu,
        &input,
        1e-6,
    );
}

#[test]
fn parity_gelu_two_elements() {
    // Below NEON width.
    let input = vec![-1.0, 1.0];
    check_activation_parity(
        "gelu_2",
        elementwise::gelu_scalar,
        elementwise::gelu,
        &input,
        1e-5,
    );
}

// -- All activations at medium size --

#[test]
fn parity_all_activations_medium_1024() {
    let input = make_input(1024);
    check_activation_parity(
        "relu_1024",
        elementwise::relu_scalar,
        elementwise::relu,
        &input,
        0.0,
    );
    check_activation_parity(
        "silu_1024",
        elementwise::silu_scalar,
        elementwise::silu,
        &input,
        1e-6,
    );
    check_activation_parity(
        "sigmoid_1024",
        elementwise::sigmoid_scalar,
        elementwise::sigmoid,
        &input,
        1e-6,
    );
    check_activation_parity(
        "tanh_1024",
        elementwise::tanh_scalar,
        elementwise::tanh,
        &input,
        1e-6,
    );
    check_activation_parity(
        "gelu_1024",
        elementwise::gelu_scalar,
        elementwise::gelu,
        &input,
        1e-5,
    );
}

// -- Activation correctness spot checks --

#[test]
fn relu_known_values() {
    let input = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
    let mut out = vec![0.0f32; 5];
    elementwise::relu(&input, &mut out);
    assert_close(&out, &[0.0, 0.0, 0.0, 1.0, 3.0], 0.0, "relu_known");
}

#[test]
fn sigmoid_known_values() {
    let input = vec![0.0];
    let mut out = vec![0.0f32; 1];
    elementwise::sigmoid(&input, &mut out);
    assert!(
        (out[0] - 0.5).abs() < 1e-6,
        "sigmoid(0) should be 0.5, got {}",
        out[0]
    );
}

#[test]
fn tanh_known_values() {
    let input = vec![0.0];
    let mut out = vec![0.0f32; 1];
    elementwise::tanh(&input, &mut out);
    assert!((out[0]).abs() < 1e-6, "tanh(0) should be 0, got {}", out[0]);
}

#[test]
fn gelu_known_values() {
    // GELU(0) = 0 by symmetry.
    let input = vec![0.0];
    let mut out = vec![0.0f32; 1];
    elementwise::gelu(&input, &mut out);
    assert!((out[0]).abs() < 1e-6, "gelu(0) should be 0, got {}", out[0]);
}

// ============================================================================
// B. Reduction Parity (17+ tests)
// ============================================================================

#[test]
fn parity_sum_small() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let scalar = reduction::sum_scalar(&input);
    let simd = reduction::sum(&input);
    assert!(
        (scalar - simd).abs() < 1e-5,
        "sum_small: {scalar} vs {simd}"
    );
}

#[test]
fn parity_sum_simd_aligned() {
    let input = make_input(256);
    let scalar = reduction::sum_scalar(&input);
    let simd = reduction::sum(&input);
    assert!((scalar - simd).abs() < 1e-3, "sum_256: {scalar} vs {simd}");
}

#[test]
fn parity_sum_simd_tail() {
    let input = make_input(259);
    let scalar = reduction::sum_scalar(&input);
    let simd = reduction::sum(&input);
    assert!((scalar - simd).abs() < 1e-3, "sum_259: {scalar} vs {simd}");
}

#[test]
fn parity_sum_single_element() {
    let input = vec![42.0];
    assert!((reduction::sum(&input) - 42.0).abs() < 1e-7);
}

#[test]
fn parity_sum_many_small_values() {
    // Test accumulation order sensitivity: sum of many identical small values.
    let input = vec![0.001_f32; 10000];
    let scalar = reduction::sum_scalar(&input);
    let simd = reduction::sum(&input);
    let expected = 10.0_f32;
    // Both should be close to 10.0; SIMD may differ due to partial sums.
    assert!(
        (scalar - expected).abs() < 0.1,
        "sum_many_small scalar: {scalar}"
    );
    assert!((simd - expected).abs() < 0.1, "sum_many_small simd: {simd}");
}

#[test]
fn parity_max_small() {
    let input = vec![-5.0, 3.0, -1.0, 7.0, 2.0];
    let scalar = reduction::max_scalar(&input);
    let simd = reduction::max(&input);
    assert!(
        (scalar - simd).abs() < 1e-7,
        "max_small: {scalar} vs {simd}"
    );
    assert!((simd - 7.0).abs() < 1e-7);
}

#[test]
fn parity_max_simd_aligned() {
    let input = make_input(256);
    let scalar = reduction::max_scalar(&input);
    let simd = reduction::max(&input);
    assert!((scalar - simd).abs() < 1e-7, "max_256: {scalar} vs {simd}");
}

#[test]
fn parity_max_simd_tail() {
    let input = make_input(259);
    let scalar = reduction::max_scalar(&input);
    let simd = reduction::max(&input);
    assert!((scalar - simd).abs() < 1e-7, "max_259: {scalar} vs {simd}");
}

#[test]
fn parity_max_single_element() {
    let input = vec![-42.0];
    assert!((reduction::max(&input) - (-42.0)).abs() < 1e-7);
}

#[test]
fn parity_max_all_same() {
    let input = vec![3.14; 100];
    assert!((reduction::max(&input) - 3.14).abs() < 1e-6);
}

#[test]
fn parity_max_all_negative() {
    let input = vec![-10.0, -5.0, -1.0, -100.0];
    assert!((reduction::max(&input) - (-1.0)).abs() < 1e-7);
}

#[test]
fn parity_dot_small() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let scalar = reduction::dot_scalar(&a, &b);
    let simd = reduction::dot(&a, &b);
    // 5 + 8 + 9 + 8 + 5 = 35
    assert!(
        (scalar - simd).abs() < 1e-5,
        "dot_small: {scalar} vs {simd}"
    );
    assert!((simd - 35.0).abs() < 1e-4);
}

#[test]
fn parity_dot_simd_aligned() {
    let a = make_input(256);
    let b: Vec<f32> = a.iter().map(|x| x * 0.5 + 1.0).collect();
    let scalar = reduction::dot_scalar(&a, &b);
    let simd = reduction::dot(&a, &b);
    assert!((scalar - simd).abs() < 1.0, "dot_256: {scalar} vs {simd}");
}

#[test]
fn parity_dot_simd_tail() {
    let a = make_input(259);
    let b: Vec<f32> = a.iter().map(|x| x * 0.3 - 2.0).collect();
    let scalar = reduction::dot_scalar(&a, &b);
    let simd = reduction::dot(&a, &b);
    assert!((scalar - simd).abs() < 1.0, "dot_259: {scalar} vs {simd}");
}

#[test]
fn parity_dot_orthogonal() {
    // Orthogonal vectors: dot product should be 0.
    let a = vec![1.0, 0.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0, 0.0];
    assert!((reduction::dot(&a, &b)).abs() < 1e-7);
}

#[test]
fn parity_dot_self() {
    // dot(v, v) = ||v||^2
    let v = vec![3.0, 4.0]; // ||v||^2 = 25
    assert!((reduction::dot(&v, &v) - 25.0).abs() < 1e-5);
}

#[test]
fn reduction_sum_empty() {
    assert_eq!(reduction::sum(&[]), 0.0);
    assert_eq!(reduction::sum_scalar(&[]), 0.0);
}

#[test]
fn reduction_max_empty() {
    assert_eq!(reduction::max(&[]), f32::NEG_INFINITY);
    assert_eq!(reduction::max_scalar(&[]), f32::NEG_INFINITY);
}

#[test]
fn reduction_dot_empty() {
    assert_eq!(reduction::dot(&[], &[]), 0.0);
    assert_eq!(reduction::dot_scalar(&[], &[]), 0.0);
}

// ============================================================================
// C. MatMul Parity (17+ tests)
// ============================================================================

#[test]
fn matmul_1x1() {
    let a = vec![3.0];
    let b = vec![5.0];
    let c = matmul::matmul(&a, &b, 1, 1, 1);
    assert_close(&c, &[15.0], 1e-6, "matmul_1x1");
}

#[test]
fn matmul_4x4_identity() {
    let a: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    // 4x4 identity
    let b = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let c = matmul::matmul(&a, &b, 4, 4, 4);
    assert_close(&c, &a, 1e-5, "matmul_4x4_identity");
}

#[test]
fn matmul_4x4_known() {
    // A and B both 4x4 with known values, check against naive.
    let a: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let b: Vec<f32> = (0..16).map(|i| (i as f32) * 0.3 + 1.0).collect();
    let c = matmul::matmul(&a, &b, 4, 4, 4);
    let expected = naive_matmul(&a, &b, 4, 4, 4);
    assert_close(&c, &expected, 1e-4, "matmul_4x4_known");
}

#[test]
fn matmul_16x16() {
    let m = 16;
    let k = 16;
    let n = 16;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.2 - 0.6).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
    let c = matmul::matmul(&a, &b, m, k, n);
    let expected = naive_matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 1e-3, "matmul_16x16");
}

#[test]
fn matmul_64x64() {
    let m = 64;
    let k = 64;
    let n = 64;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 13) as f32 * 0.1).collect();
    let c = matmul::matmul(&a, &b, m, k, n);
    let expected = naive_matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 1e-2, "matmul_64x64");
}

#[test]
fn matmul_rectangular_m1() {
    // M=1 (row vector * matrix)
    let m = 1;
    let k = 8;
    let n = 4;
    let a: Vec<f32> = (0..m * k).map(|i| i as f32 + 1.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
    let c = matmul::matmul(&a, &b, m, k, n);
    let expected = naive_matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 1e-4, "matmul_m1");
}

#[test]
fn matmul_rectangular_n1() {
    // N=1 (matrix * column vector)
    let m = 6;
    let k = 4;
    let n = 1;
    let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| i as f32 + 1.0).collect();
    let c = matmul::matmul(&a, &b, m, k, n);
    let expected = naive_matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 1e-4, "matmul_n1");
}

#[test]
fn matmul_rectangular_k1() {
    // K=1 (outer product)
    let m = 4;
    let k = 1;
    let n = 5;
    let a: Vec<f32> = (0..m * k).map(|i| i as f32 + 1.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let c = matmul::matmul(&a, &b, m, k, n);
    let expected = naive_matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 1e-5, "matmul_k1");
}

#[test]
fn matmul_tall_skinny() {
    // A is 100x3, B is 3x2
    let m = 100;
    let k = 3;
    let n = 2;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 19) as f32 * 0.1 - 0.9).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5 + 0.1).collect();
    let c = matmul::matmul(&a, &b, m, k, n);
    let expected = naive_matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 1e-3, "matmul_tall_skinny");
}

#[test]
fn matmul_wide_short() {
    // A is 2x100, B is 100x3
    let m = 2;
    let k = 100;
    let n = 3;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 23) as f32 * 0.05).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 17) as f32 * 0.07 - 0.5).collect();
    let c = matmul::matmul(&a, &b, m, k, n);
    let expected = naive_matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 1e-2, "matmul_wide_short");
}

#[test]
fn matmul_batched_via_loop() {
    // Simulate batch dimension: B=3, M=4, K=5, N=6.
    let batch = 3;
    let m = 4;
    let k = 5;
    let n = 6;
    for b in 0..batch {
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i + b * 7) % 13) as f32 * 0.2 - 1.0)
            .collect();
        let bm: Vec<f32> = (0..k * n)
            .map(|i| ((i + b * 11) % 17) as f32 * 0.15 - 0.8)
            .collect();
        let c = matmul::matmul(&a, &bm, m, k, n);
        let expected = naive_matmul(&a, &bm, m, k, n);
        assert_close(&c, &expected, 1e-3, &format!("matmul_batch_{b}"));
    }
}

#[test]
fn matmul_transposed_b_small() {
    // A=[2,3], B^T=[2,3] (B is [3,2], transposed to [2,3])
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_t = vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0];
    let mut c = vec![0.0f32; 4];
    matmul::matmul_with_transposed_b(&a, &b_t, &mut c, 2, 3, 2);
    assert_close(
        &c,
        &[58.0, 64.0, 139.0, 154.0],
        1e-4,
        "matmul_transposed_b_small",
    );
}

#[test]
fn matmul_transposed_b_vs_regular() {
    // Compare matmul(A, B) with matmul_with_transposed_b(A, B^T).
    let m = 8;
    let k = 6;
    let n = 10;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 11) as f32 * 0.3 - 1.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 * 0.2 - 0.7).collect();

    // Regular matmul
    let c_regular = matmul::matmul(&a, &b, m, k, n);

    // Transpose B: B[K,N] -> B^T[N,K]
    let mut b_t = vec![0.0f32; n * k];
    for i in 0..k {
        for j in 0..n {
            b_t[j * k + i] = b[i * n + j];
        }
    }
    let mut c_transposed = vec![0.0f32; m * n];
    matmul::matmul_with_transposed_b(&a, &b_t, &mut c_transposed, m, k, n);

    assert_close(
        &c_regular,
        &c_transposed,
        1e-4,
        "matmul_transposed_vs_regular",
    );
}

#[test]
fn matmul_dot_contiguous_matches_dot() {
    let a = make_input(100);
    let b: Vec<f32> = a.iter().map(|x| x * 0.3 + 1.0).collect();
    let d1 = matmul::dot_contiguous(&a, &b);
    let d2 = reduction::dot(&a, &b);
    assert!(
        (d1 - d2).abs() < 1e-4,
        "dot_contiguous vs dot: {d1} vs {d2}"
    );
}

#[test]
fn matmul_tiled_pre_zeroed() {
    // Verify matmul_tiled accumulates into pre-zeroed buffer correctly.
    let m = 10;
    let k = 8;
    let n = 6;
    let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 + 0.1).collect();
    let mut c = vec![0.0f32; m * n];
    matmul::matmul_tiled(&a, &b, &mut c, m, k, n);
    let expected = naive_matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 1e-3, "matmul_tiled_pre_zeroed");
}

#[test]
fn matmul_128x128_cross_tile() {
    // Exercises full tiling (TILE=64): 2 tiles per dimension.
    let m = 128;
    let k = 128;
    let n = 128;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 31) as f32 * 0.03 - 0.45).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 29) as f32 * 0.04 - 0.55).collect();
    let c = matmul::matmul(&a, &b, m, k, n);
    let expected = naive_matmul(&a, &b, m, k, n);
    // Larger tolerance for 128x128 accumulation.
    assert_close(&c, &expected, 0.05, "matmul_128x128");
}

#[test]
fn matmul_non_tile_aligned_65x70() {
    // 65 and 70 are not multiples of TILE=64.
    let m = 65;
    let k = 70;
    let n = 33;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 19) as f32 * 0.1 - 0.9).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 23) as f32 * 0.08 - 0.8).collect();
    let c = matmul::matmul(&a, &b, m, k, n);
    let expected = naive_matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 0.1, "matmul_65x70x33");
}

// ============================================================================
// D. Edge Cases (18+ tests)
// ============================================================================

// -- Empty tensors --

#[test]
fn edge_empty_relu() {
    let input: Vec<f32> = vec![];
    let mut output: Vec<f32> = vec![];
    elementwise::relu(&input, &mut output);
    assert!(output.is_empty());
}

#[test]
fn edge_empty_all_activations() {
    let input: Vec<f32> = vec![];
    let mut output: Vec<f32> = vec![];
    elementwise::relu(&input, &mut output);
    elementwise::silu(&input, &mut output);
    elementwise::sigmoid(&input, &mut output);
    elementwise::tanh(&input, &mut output);
    elementwise::gelu(&input, &mut output);
    // No panic = pass.
}

// -- Single element --

#[test]
fn edge_single_relu_negative() {
    let input = vec![-1.0];
    let mut output = vec![0.0f32; 1];
    elementwise::relu(&input, &mut output);
    assert_eq!(output[0], 0.0);
}

#[test]
fn edge_single_sigmoid() {
    let input = vec![0.0];
    let mut output = vec![0.0f32; 1];
    elementwise::sigmoid(&input, &mut output);
    assert!((output[0] - 0.5).abs() < 1e-7);
}

// -- Denormal numbers --

#[test]
fn edge_denormal_inputs_relu() {
    let input = vec![1e-38, 5e-39, -1e-38, -5e-39];
    let mut output = vec![0.0f32; 4];
    elementwise::relu(&input, &mut output);
    assert_eq!(output[0], 1e-38);
    assert_eq!(output[1], 5e-39);
    assert_eq!(output[2], 0.0);
    assert_eq!(output[3], 0.0);
}

#[test]
fn edge_denormal_inputs_sigmoid() {
    let input = vec![1e-38, 5e-39, -1e-38, -5e-39];
    let mut scalar_out = vec![0.0f32; 4];
    let mut simd_out = vec![0.0f32; 4];
    elementwise::sigmoid_scalar(&input, &mut scalar_out);
    elementwise::sigmoid(&input, &mut simd_out);
    // Denormals near zero: sigmoid should be ~0.5.
    for &v in &simd_out {
        assert!(
            (v - 0.5).abs() < 1e-6,
            "sigmoid of denormal should be ~0.5, got {v}"
        );
    }
    assert_close(&scalar_out, &simd_out, 1e-6, "sigmoid_denormal");
}

#[test]
fn edge_denormal_sum() {
    let input = vec![1e-38; 100];
    let s = reduction::sum(&input);
    assert!(s.is_finite(), "sum of denormals should be finite");
    assert!(s > 0.0, "sum of positive denormals should be positive");
}

// -- NaN propagation --

#[test]
fn edge_nan_relu() {
    let input = vec![f32::NAN, 1.0, -1.0, f32::NAN];
    let mut output = vec![0.0f32; 4];
    elementwise::relu(&input, &mut output);
    assert!(output[0].is_nan(), "relu(NaN) should be NaN");
    assert_eq!(output[1], 1.0);
    assert_eq!(output[2], 0.0);
    assert!(output[3].is_nan(), "relu(NaN) should be NaN");
}

#[test]
fn edge_nan_silu() {
    let input = vec![f32::NAN, 1.0];
    let mut output = vec![0.0f32; 2];
    elementwise::silu(&input, &mut output);
    assert!(output[0].is_nan(), "silu(NaN) should be NaN");
}

#[test]
fn edge_nan_sigmoid() {
    let input = vec![f32::NAN];
    let mut output = vec![0.0f32; 1];
    elementwise::sigmoid(&input, &mut output);
    assert!(output[0].is_nan(), "sigmoid(NaN) should be NaN");
}

#[test]
fn edge_nan_tanh() {
    let input = vec![f32::NAN, 0.0];
    let mut output = vec![0.0f32; 2];
    elementwise::tanh(&input, &mut output);
    assert!(output[0].is_nan(), "tanh(NaN) should be NaN");
    assert!((output[1]).abs() < 1e-7);
}

#[test]
fn edge_nan_gelu() {
    let input = vec![f32::NAN];
    let mut output = vec![0.0f32; 1];
    elementwise::gelu(&input, &mut output);
    assert!(output[0].is_nan(), "gelu(NaN) should be NaN");
}

#[test]
fn edge_nan_sum() {
    let input = vec![1.0, f32::NAN, 3.0];
    let s = reduction::sum(&input);
    assert!(s.is_nan(), "sum with NaN should be NaN");
}

#[test]
fn edge_nan_dot() {
    let a = vec![1.0, f32::NAN, 3.0];
    let b = vec![1.0, 2.0, 3.0];
    let d = reduction::dot(&a, &b);
    assert!(d.is_nan(), "dot with NaN should be NaN");
}

// -- Inf handling --

#[test]
fn edge_inf_relu() {
    let input = vec![f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.0];
    let mut output = vec![0.0f32; 4];
    elementwise::relu(&input, &mut output);
    assert_eq!(output[0], f32::INFINITY);
    assert_eq!(output[1], 0.0); // max(neg_inf, 0) = 0
    assert_eq!(output[2], 0.0);
    assert_eq!(output[3], 1.0);
}

#[test]
fn edge_inf_sigmoid() {
    let input = vec![f32::INFINITY, f32::NEG_INFINITY];
    let mut output = vec![0.0f32; 2];
    elementwise::sigmoid(&input, &mut output);
    assert!(
        (output[0] - 1.0).abs() < 1e-6,
        "sigmoid(+inf) should be 1.0, got {}",
        output[0]
    );
    assert!(
        (output[1]).abs() < 1e-6,
        "sigmoid(-inf) should be 0.0, got {}",
        output[1]
    );
}

#[test]
fn edge_inf_sum() {
    let input = vec![1.0, f32::INFINITY, 3.0];
    let s = reduction::sum(&input);
    assert_eq!(s, f32::INFINITY, "sum with +inf should be +inf");
}

#[test]
fn edge_inf_max() {
    let input = vec![1.0, f32::NEG_INFINITY, 3.0];
    assert!((reduction::max(&input) - 3.0).abs() < 1e-7);

    let input2 = vec![f32::INFINITY, 1.0, 2.0];
    assert_eq!(reduction::max(&input2), f32::INFINITY);
}

// -- Large arrays --

#[test]
fn edge_large_array_relu() {
    let n = 1_000_000;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 500.0).collect();
    let mut output = vec![0.0f32; n];
    elementwise::relu(&input, &mut output);
    // Spot check: first 500_000 should be 0, rest should be positive.
    assert_eq!(output[0], 0.0);
    assert!(output[500_001] > 0.0);
    assert!((output[999_999] - 499.999).abs() < 0.01);
}

#[test]
fn edge_large_array_sum() {
    let n = 1_000_000;
    let input = vec![1.0f32; n];
    let s = reduction::sum(&input);
    // Should be close to 1_000_000. Floating-point accumulation error is expected.
    assert!(
        (s - 1_000_000.0).abs() < 100.0,
        "sum of 1M ones should be ~1M, got {s}"
    );
}

#[test]
fn edge_large_array_max() {
    let n = 1_000_000;
    let mut input = vec![0.0f32; n];
    input[777_777] = 42.0; // Max buried in the middle.
    assert!((reduction::max(&input) - 42.0).abs() < 1e-7);
}

// -- All-zero input --

#[test]
fn edge_all_zero_activations() {
    let input = vec![0.0f32; 32];
    let mut out = vec![0.0f32; 32];

    elementwise::relu(&input, &mut out);
    assert!(out.iter().all(|&v| v == 0.0), "relu(0) should be 0");

    elementwise::silu(&input, &mut out);
    assert!(out.iter().all(|&v| v == 0.0), "silu(0) should be 0");

    elementwise::tanh(&input, &mut out);
    assert!(out.iter().all(|&v| v.abs() < 1e-7), "tanh(0) should be 0");

    elementwise::gelu(&input, &mut out);
    assert!(out.iter().all(|&v| v.abs() < 1e-7), "gelu(0) should be 0");
}

#[test]
fn edge_all_zero_sum() {
    let input = vec![0.0f32; 1000];
    assert_eq!(reduction::sum(&input), 0.0);
}

// -- All-same-value input --

#[test]
fn edge_all_same_activations() {
    let input = vec![2.0f32; 64];
    let mut scalar_out = vec![0.0f32; 64];
    let mut simd_out = vec![0.0f32; 64];

    elementwise::relu_scalar(&input, &mut scalar_out);
    elementwise::relu(&input, &mut simd_out);
    assert!(
        simd_out.iter().all(|&v| (v - 2.0).abs() < 1e-7),
        "relu(2) should be 2"
    );
    assert_close(&scalar_out, &simd_out, 0.0, "relu_all_same");

    elementwise::sigmoid_scalar(&input, &mut scalar_out);
    elementwise::sigmoid(&input, &mut simd_out);
    assert_close(&scalar_out, &simd_out, 1e-6, "sigmoid_all_same");
}

#[test]
fn edge_all_same_max() {
    let input = vec![7.7f32; 100];
    assert!((reduction::max(&input) - 7.7).abs() < 1e-6);
}

// ============================================================================
// E. SIMD Detection (6 tests)
// ============================================================================

#[test]
fn simd_detect_returns_valid_level() {
    let level = simd_detect::detect();
    match level {
        simd_detect::SimdLevel::Scalar
        | simd_detect::SimdLevel::Neon
        | simd_detect::SimdLevel::Avx2 => {}
        // non_exhaustive: future variants are fine.
        _ => panic!("Unknown SIMD level: {level:?}"),
    }
}

#[test]
fn simd_detect_consistent_across_calls() {
    let level1 = simd_detect::detect();
    let level2 = simd_detect::detect();
    let level3 = simd_detect::detect();
    assert_eq!(level1, level2);
    assert_eq!(level2, level3);
}

#[test]
fn simd_lane_count_valid() {
    let level = simd_detect::detect();
    let lanes = simd_detect::lane_count(level);
    assert!(lanes >= 1);
    assert!(
        lanes.is_power_of_two(),
        "lane_count should be power of 2, got {lanes}"
    );
}

#[test]
#[cfg(target_arch = "aarch64")]
fn simd_detect_aarch64_is_neon() {
    // NEON is mandatory on aarch64.
    assert_eq!(simd_detect::detect(), simd_detect::SimdLevel::Neon);
    assert_eq!(simd_detect::lane_count(simd_detect::SimdLevel::Neon), 4);
}

#[test]
fn simd_lane_count_scalar_is_one() {
    assert_eq!(simd_detect::lane_count(simd_detect::SimdLevel::Scalar), 1);
}

#[test]
fn simd_lane_count_neon_is_four() {
    assert_eq!(simd_detect::lane_count(simd_detect::SimdLevel::Neon), 4);
}

#[test]
fn simd_lane_count_avx2_is_eight() {
    assert_eq!(simd_detect::lane_count(simd_detect::SimdLevel::Avx2), 8);
}

// ============================================================================
// F. Additional activation size sweep (parametric-style)
// ============================================================================

/// Tests all activations at sizes that exercise SIMD boundaries:
/// 0, 1, 2, 3, 4 (NEON boundary), 5, 7, 8 (AVX2 boundary), 9, 15, 16, 17, 31, 32, 33
#[test]
fn activation_size_sweep() {
    let sizes = [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33];

    for &sz in &sizes {
        let input = make_input(sz);
        let tol_transcendental = 1e-5;

        // ReLU (exact)
        check_activation_parity(
            &format!("relu_sz{sz}"),
            elementwise::relu_scalar,
            elementwise::relu,
            &input,
            0.0,
        );

        // SiLU
        check_activation_parity(
            &format!("silu_sz{sz}"),
            elementwise::silu_scalar,
            elementwise::silu,
            &input,
            tol_transcendental,
        );

        // Sigmoid
        check_activation_parity(
            &format!("sigmoid_sz{sz}"),
            elementwise::sigmoid_scalar,
            elementwise::sigmoid,
            &input,
            tol_transcendental,
        );

        // Tanh
        check_activation_parity(
            &format!("tanh_sz{sz}"),
            elementwise::tanh_scalar,
            elementwise::tanh,
            &input,
            tol_transcendental,
        );

        // GELU
        check_activation_parity(
            &format!("gelu_sz{sz}"),
            elementwise::gelu_scalar,
            elementwise::gelu,
            &input,
            tol_transcendental,
        );
    }
}

// ============================================================================
// G. SIMD Matmul Micro-Kernel Boundary Tests
// ============================================================================

// These tests exercise the NEON 4x4 / AVX2 4x8 micro-kernel boundaries
// and the scalar remainder paths.

#[test]
fn matmul_2x2() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let c = matmul::matmul(&a, &b, 2, 2, 2);
    // [1*5+2*7, 1*6+2*8] = [19, 22]
    // [3*5+4*7, 3*6+4*8] = [43, 50]
    assert_close(&c, &[19.0, 22.0, 43.0, 50.0], 1e-5, "matmul_2x2");
}

#[test]
fn matmul_3x3() {
    // 3x3 -- below MR_NEON=4, all remainder.
    let m = 3;
    let k = 3;
    let n = 3;
    let a: Vec<f32> = (1..=9).map(|x| x as f32).collect();
    let b: Vec<f32> = (1..=9).map(|x| x as f32 * 0.1).collect();
    let c = matmul::matmul(&a, &b, m, k, n);
    let expected = naive_matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 1e-4, "matmul_3x3");
}

#[test]
fn matmul_5x5() {
    // 5 = MR_NEON(4) + 1 remainder row. Tests micro-kernel + 1 row fallback.
    let m = 5;
    let k = 5;
    let n = 5;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.3 - 1.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.2 - 1.0).collect();
    let c = matmul::matmul(&a, &b, m, k, n);
    let expected = naive_matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 1e-3, "matmul_5x5");
}

#[test]
fn matmul_8x8() {
    // 8 = 2 * MR_NEON(4). Two full micro-kernel invocations per tile row.
    let m = 8;
    let k = 8;
    let n = 8;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.15 - 0.8).collect();
    let c = matmul::matmul(&a, &b, m, k, n);
    let expected = naive_matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 1e-3, "matmul_8x8");
}

#[test]
fn matmul_9x9() {
    // 9 = 2 * MR(4) + 1 remainder. Tests micro-kernel + remainder row/col.
    let m = 9;
    let k = 9;
    let n = 9;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.05 - 0.4).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 19) as f32 * 0.07 - 0.6).collect();
    let c = matmul::matmul(&a, &b, m, k, n);
    let expected = naive_matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 1e-3, "matmul_9x9");
}

#[test]
fn matmul_size_sweep() {
    // Sweep across sizes that exercise different micro-kernel boundaries.
    // On NEON: MR=4, NR=4. On AVX2: MR=4, NR=8.
    let sizes = [
        1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 31, 32, 33, 63, 64, 65,
    ];

    for &sz in &sizes {
        let a: Vec<f32> = (0..sz * sz)
            .map(|i| (i % 23) as f32 * 0.04 - 0.46)
            .collect();
        let b: Vec<f32> = (0..sz * sz)
            .map(|i| (i % 19) as f32 * 0.05 - 0.47)
            .collect();
        let c = matmul::matmul(&a, &b, sz, sz, sz);
        let expected = naive_matmul(&a, &b, sz, sz, sz);
        // Tolerance scales with matrix size due to FP accumulation.
        let tol = (sz as f32) * 1e-4;
        assert_close(&c, &expected, tol, &format!("matmul_sweep_{sz}x{sz}"));
    }
}

#[test]
fn matmul_rectangular_sweep() {
    // Non-square shapes exercising different M, K, N combinations.
    let shapes = [
        (1, 1, 1),
        (1, 64, 1),
        (4, 1, 4),
        (3, 7, 5),
        (5, 3, 7),
        (7, 5, 3),
        (4, 64, 8),
        (8, 64, 4),
        (13, 17, 11),
        (64, 128, 32),
        (128, 32, 64),
        (256, 1, 256),
    ];

    for (m, k, n) in shapes {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 29) as f32 * 0.03 - 0.43).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 31) as f32 * 0.04 - 0.62).collect();
        let c = matmul::matmul(&a, &b, m, k, n);
        let expected = naive_matmul(&a, &b, m, k, n);
        let tol = (k as f32).max(1.0) * 1e-4;
        assert_close(&c, &expected, tol, &format!("matmul_rect_{m}x{k}x{n}"));
    }
}

#[test]
fn matmul_256x256_vs_naive() {
    // Large enough to exercise multi-tile loops thoroughly.
    let m = 256;
    let k = 256;
    let n = 256;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 37) as f32 * 0.02 - 0.36).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 41) as f32 * 0.025 - 0.5).collect();
    let c = matmul::matmul(&a, &b, m, k, n);
    let expected = naive_matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 0.15, "matmul_256x256");
}

#[test]
fn matmul_accumulation_order_consistency() {
    // Verify that the SIMD tiled path and transposed-B path produce
    // consistent results for the same mathematical operation.
    let m = 32;
    let k = 32;
    let n = 32;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();

    // Regular matmul
    let c_regular = matmul::matmul(&a, &b, m, k, n);

    // Transpose B and use transposed-B path
    let mut b_t = vec![0.0f32; n * k];
    for i in 0..k {
        for j in 0..n {
            b_t[j * k + i] = b[i * n + j];
        }
    }
    let mut c_transposed = vec![0.0f32; m * n];
    matmul::matmul_with_transposed_b(&a, &b_t, &mut c_transposed, m, k, n);

    // Both should agree (minor FP differences from accumulation order).
    assert_close(
        &c_regular,
        &c_transposed,
        1e-3,
        "matmul_regular_vs_transposed",
    );
}

#[test]
fn matmul_zero_dimension() {
    // M=0, K=0, N=0 should not panic.
    let c = matmul::matmul(&[], &[], 0, 0, 0);
    assert!(c.is_empty());

    // M=0 with non-zero K and N
    let c = matmul::matmul(&[], &[1.0, 2.0, 3.0, 4.0], 0, 2, 2);
    assert!(c.is_empty());

    // N=0
    let c = matmul::matmul(&[1.0, 2.0], &[], 1, 2, 0);
    assert!(c.is_empty());
}

#[test]
fn matmul_all_ones() {
    // A = ones(M, K), B = ones(K, N) => C should be all K.
    let m = 16;
    let k = 32;
    let n = 8;
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let c = matmul::matmul(&a, &b, m, k, n);
    for (i, &val) in c.iter().enumerate() {
        assert!(
            (val - k as f32).abs() < 1e-3,
            "ones matmul [{i}]: expected {k}, got {val}",
        );
    }
}

#[test]
fn matmul_negative_values() {
    // All negative inputs: verifies sign handling in SIMD paths.
    let m = 8;
    let k = 8;
    let n = 8;
    let a: Vec<f32> = (0..m * k).map(|i| -(i as f32 + 1.0) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| -(i as f32 + 1.0) * 0.05).collect();
    let c = matmul::matmul(&a, &b, m, k, n);
    let expected = naive_matmul(&a, &b, m, k, n);
    assert_close(&c, &expected, 1e-3, "matmul_negative");
    // Product of two negatives should be positive.
    assert!(c.iter().all(|&v| v > 0.0), "neg * neg should be positive");
}
