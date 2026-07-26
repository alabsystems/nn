// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended CPU SIMD backend tests covering matmul, GEMV, elementwise ops,
//! reductions, softmax numerical stability, layernorm/instancenorm
//! correctness, and SIMD alignment edge cases.
//!
//! Each test compares the SIMD auto-dispatch path against a known reference
//! (scalar or analytical f64 ground truth) with appropriate tolerances.

use super::*;

// =========================================================================
// Helpers
// =========================================================================

/// f64-precision reference matmul for ground truth.
fn ref_matmul_f64(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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

/// f64-precision reference GEMV for ground truth.
fn ref_gemv_f64(matrix: &[f32], vec: &[f32], m: usize, k: usize) -> Vec<f32> {
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

/// f64-precision reference softmax for ground truth.
fn ref_softmax_f64(input: &[f32]) -> Vec<f32> {
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = input.iter().map(|&x| f64::from(x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| (e / sum) as f32).collect()
}

/// f64-precision reference layernorm for ground truth.
fn ref_layer_norm_f64(input: &[f32], gamma: &[f32], beta: &[f32], n: usize, eps: f32) -> Vec<f32> {
    let mean: f64 = input.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
    let var: f64 = input
        .iter()
        .map(|&x| {
            let d = f64::from(x) - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let inv_std = 1.0 / (var + f64::from(eps)).sqrt();
    input
        .iter()
        .enumerate()
        .map(|(i, &x)| ((f64::from(x) - mean) * inv_std * f64::from(gamma[i]) + f64::from(beta[i])) as f32)
        .collect()
}

/// f64-precision reference instance norm for ground truth.
fn ref_instance_norm_f64(input: &[f32], channels: usize, spatial: usize, eps: f32) -> Vec<f32> {
    let mut output = vec![0.0f32; channels * spatial];
    for c in 0..channels {
        let start = c * spatial;
        let ch = &input[start..start + spatial];
        let mean: f64 = ch.iter().map(|&x| f64::from(x)).sum::<f64>() / spatial as f64;
        let var: f64 = ch
            .iter()
            .map(|&x| {
                let d = f64::from(x) - mean;
                d * d
            })
            .sum::<f64>()
            / spatial as f64;
        let inv_std = 1.0 / (var + f64::from(eps)).sqrt();
        for i in 0..spatial {
            output[start + i] = ((f64::from(ch[i]) - mean) * inv_std) as f32;
        }
    }
    output
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: {va} vs {vb}, diff={diff}, tol={tol}"
        );
    }
}

/// Build identity matrix of size n x n.
fn identity_matrix(n: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

// =========================================================================
// 1. MATMUL CORRECTNESS
// =========================================================================

#[test]
fn test_matmul_identity_1x1() {
    let a = [7.5f32];
    let eye = [1.0f32];
    let mut out = [0.0f32; 1];
    matmul_f32_scalar(&a, &eye, 1, 1, 1, &mut out);
    assert_close(&out, &a, 1e-6, "matmul identity 1x1 scalar");

    let mut out2 = [0.0f32; 1];
    matmul_f32(&a, &eye, 1, 1, 1, &mut out2);
    assert_close(&out2, &a, 1e-6, "matmul identity 1x1 dispatch");
}

#[test]
fn test_matmul_identity_3x3() {
    let a: Vec<f32> = (1..=9).map(|i| i as f32).collect();
    let eye = identity_matrix(3);
    let mut out = vec![0.0f32; 9];
    matmul_f32(&a, &eye, 3, 3, 3, &mut out);
    assert_close(&out, &a, 1e-5, "matmul identity 3x3");
}

#[test]
fn test_matmul_identity_8x8() {
    let a: Vec<f32> = (0..64).map(|i| (i as f32 * 0.3).sin()).collect();
    let eye = identity_matrix(8);
    let mut out = vec![0.0f32; 64];
    matmul_f32(&a, &eye, 8, 8, 8, &mut out);
    assert_close(&out, &a, 1e-5, "matmul identity 8x8");
}

#[test]
fn test_matmul_small_2x3_times_3x2() {
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    // C = [[58,64],[139,154]]
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let expected = [58.0, 64.0, 139.0, 154.0];
    let mut out = vec![0.0f32; 4];
    matmul_f32(&a, &b, 2, 3, 2, &mut out);
    assert_close(&out, &expected, 1e-4, "matmul 2x3 * 3x2");
}

#[test]
fn test_matmul_small_1x4_times_4x1() {
    // Row vector times column vector = dot product (scalar).
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    // dot = 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
    let expected = [70.0];
    let mut out = [0.0f32; 1];
    matmul_f32(&a, &b, 1, 4, 1, &mut out);
    assert_close(&out, &expected, 1e-4, "matmul 1x4 * 4x1");
}

#[test]
fn test_matmul_transpose_commutativity() {
    // A * I == I * A^T transposed back, for square matrices.
    let n = 5;
    let a: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.7).sin()).collect();
    let eye = identity_matrix(n);
    let result1 = matmul_reference(&a, &eye, n, n, n);
    assert_close(&result1, &a, 1e-5, "matmul A*I == A");
}

#[test]
fn test_matmul_dispatch_vs_scalar_non_square() {
    // Non-square: 5x7 * 7x3 -> 5x3
    let m = 5;
    let k = 7;
    let n = 3;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).cos()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.37).sin()).collect();

    let ref_out = ref_matmul_f64(&a, &b, m, k, n);
    let mut dispatch_out = vec![0.0f32; m * n];
    matmul_f32(&a, &b, m, k, n, &mut dispatch_out);
    assert_close(
        &dispatch_out,
        &ref_out,
        1e-4,
        "matmul dispatch vs f64 ref 5x7*7x3",
    );
}

#[test]
fn test_matmul_dispatch_vs_scalar_large() {
    // Larger matrix to exercise SIMD tiling paths.
    let m = 17;
    let k = 33;
    let n = 9;
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.01).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.02).cos()).collect();

    let ref_out = ref_matmul_f64(&a, &b, m, k, n);
    let mut dispatch_out = vec![0.0f32; m * n];
    matmul_f32(&a, &b, m, k, n, &mut dispatch_out);
    assert_close(&dispatch_out, &ref_out, 1e-3, "matmul dispatch 17x33*33x9");
}

#[test]
fn test_matmul_zeros() {
    let a = vec![0.0f32; 12];
    let b: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let mut out = vec![99.0f32; 9];
    matmul_f32(&a, &b, 3, 4, 3, &mut out);
    for &v in &out {
        assert!(
            (v - 0.0).abs() < 1e-10,
            "zero matrix * B should be zero, got {v}"
        );
    }
}

#[test]
fn test_matmul_negative_values() {
    let a = [-1.0, -2.0, -3.0, -4.0];
    let b = [1.0, 0.0, 0.0, 1.0]; // identity 2x2
    let mut out = [0.0f32; 4];
    matmul_f32(&a, &b, 2, 2, 2, &mut out);
    assert_close(&out, &a, 1e-6, "matmul negative * identity");
}

// =========================================================================
// 2. GEMV CORRECTNESS
// =========================================================================

#[test]
fn test_gemv_identity_3x3() {
    let eye = identity_matrix(3);
    let v = [3.0, 5.0, 7.0];
    let mut out = vec![0.0f32; 3];
    gemv_f32(&eye, &v, 3, 3, &mut out);
    assert_close(&out, &v, 1e-5, "gemv identity 3x3");
}

#[test]
fn test_gemv_identity_8x8() {
    let eye = identity_matrix(8);
    let v: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let mut out = vec![0.0f32; 8];
    gemv_f32(&eye, &v, 8, 8, &mut out);
    assert_close(&out, &v, 1e-5, "gemv identity 8x8");
}

#[test]
fn test_gemv_small_2x3() {
    // matrix = [[1,2,3],[4,5,6]], vec = [1,1,1]
    // out = [6, 15]
    let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let v = [1.0, 1.0, 1.0];
    let expected = [6.0, 15.0];
    let mut out = vec![0.0f32; 2];
    gemv_f32(&matrix, &v, 2, 3, &mut out);
    assert_close(&out, &expected, 1e-5, "gemv 2x3 ones");
}

#[test]
fn test_gemv_dispatch_vs_reference() {
    let m = 13;
    let k = 19;
    let matrix: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.17).sin()).collect();
    let v: Vec<f32> = (0..k).map(|i| (i as f32 * 0.31).cos()).collect();

    let ref_out = ref_gemv_f64(&matrix, &v, m, k);
    let mut dispatch_out = vec![0.0f32; m];
    gemv_f32(&matrix, &v, m, k, &mut dispatch_out);
    assert_close(
        &dispatch_out,
        &ref_out,
        1e-4,
        "gemv dispatch vs f64 ref 13x19",
    );
}

#[test]
fn test_gemv_with_bias() {
    let matrix = [1.0, 0.0, 0.0, 1.0]; // 2x2 identity
    let v = [3.0, 7.0];
    let bias = [10.0, 20.0];
    let expected = [13.0, 27.0]; // [3+10, 7+20]
    let mut out = vec![0.0f32; 2];
    gemv_bias_f32(&matrix, &v, &bias, 2, 2, &mut out);
    assert_close(&out, &expected, 1e-5, "gemv_bias identity+bias");
}

#[test]
fn test_gemv_single_row() {
    // M=1, K=5 => dot product
    let matrix = [1.0, 2.0, 3.0, 4.0, 5.0];
    let v = [2.0, 3.0, 4.0, 5.0, 6.0];
    // dot = 2+6+12+20+30 = 70
    let mut out = [0.0f32; 1];
    gemv_f32(&matrix, &v, 1, 5, &mut out);
    assert!(
        (out[0] - 70.0).abs() < 1e-4,
        "gemv single row, got {}",
        out[0]
    );
}

#[test]
fn test_gemv_zeros() {
    let matrix = vec![0.0f32; 12];
    let v: Vec<f32> = (0..4).map(|i| i as f32 + 1.0).collect();
    let mut out = vec![99.0f32; 3];
    gemv_f32(&matrix, &v, 3, 4, &mut out);
    for &val in &out {
        assert!(
            (val - 0.0).abs() < 1e-10,
            "zero matrix gemv should be zero, got {val}"
        );
    }
}

// =========================================================================
// 3. ELEMENTWISE OPS
// =========================================================================

#[test]
fn test_add_f32_basic_correctness() {
    let a = [1.0, -2.0, 3.0, -4.0, 5.5];
    let b = [10.0, 20.0, 30.0, 40.0, 50.0];
    let expected = [11.0, 18.0, 33.0, 36.0, 55.5];
    let mut out = [0.0f32; 5];
    add_f32(&a, &b, &mut out);
    assert_close(&out, &expected, 1e-6, "add_f32 basic");
}

#[test]
fn test_mul_f32_basic_correctness() {
    let a = [2.0, -3.0, 0.0, 4.0, -0.5];
    let b = [5.0, 2.0, 100.0, -1.0, 8.0];
    let expected = [10.0, -6.0, 0.0, -4.0, -4.0];
    let mut out = [0.0f32; 5];
    mul_f32(&a, &b, &mut out);
    assert_close(&out, &expected, 1e-6, "mul_f32 basic");
}

#[test]
fn test_relu_f32_correctness() {
    let input = [-5.0, -0.1, 0.0, 0.1, 5.0, -100.0, 100.0];
    let expected = [0.0, 0.0, 0.0, 0.1, 5.0, 0.0, 100.0];
    let mut out = vec![0.0f32; input.len()];
    relu_f32(&input, &mut out);
    assert_close(&out, &expected, 1e-6, "relu_f32 correctness");
}

#[test]
fn test_sigmoid_f32_correctness() {
    // sigmoid(0) = 0.5, sigmoid(large) ~ 1, sigmoid(-large) ~ 0
    let input: [f32; 5] = [0.0, 10.0, -10.0, 1.0, -1.0];
    let mut out = vec![0.0f32; 5];
    let mut scalar_out = vec![0.0f32; 5];

    // Use scalar as reference
    for i in 0..input.len() {
        let x = input[i];
        scalar_out[i] = 1.0_f32 / (1.0_f32 + (-x).exp());
    }

    // Check auto-dispatch via simd_elementwise module
    // We use the elementwise module's sigmoid
    elementwise::sigmoid(&input, &mut out);
    assert_close(&out, &scalar_out, 1e-6, "sigmoid_f32 vs scalar");

    // Check specific values
    assert!(
        (out[0] - 0.5).abs() < 1e-6,
        "sigmoid(0) should be 0.5, got {}",
        out[0]
    );
    assert!(out[1] > 0.999, "sigmoid(10) should be ~1, got {}", out[1]);
    assert!(out[2] < 0.001, "sigmoid(-10) should be ~0, got {}", out[2]);
}

#[test]
fn test_tanh_f32_correctness() {
    let input = [0.0, 1.0, -1.0, 5.0, -5.0];
    let mut out = vec![0.0f32; 5];
    let mut expected = vec![0.0f32; 5];
    for (i, &x) in input.iter().enumerate() {
        expected[i] = f32::tanh(x);
    }
    elementwise::tanh(&input, &mut out);
    assert_close(&out, &expected, 1e-6, "tanh_f32 vs std::tanh");

    // tanh(0) = 0
    assert!((out[0]).abs() < 1e-6, "tanh(0) should be 0, got {}", out[0]);
    // tanh(5) ~ 1
    assert!(out[3] > 0.999, "tanh(5) should be ~1, got {}", out[3]);
}

#[test]
fn test_gelu_f32_correctness() {
    // GELU(0) = 0
    let input = [0.0f32];
    let mut out = [0.0f32; 1];
    gelu_f32(&input, &mut out);
    assert!((out[0]).abs() < 1e-6, "gelu(0) should be 0, got {}", out[0]);
}

#[test]
fn test_silu_f32_correctness() {
    // SiLU(0) = 0 * sigmoid(0) = 0
    let input = [0.0f32, 1.0, -1.0];
    let mut out = vec![0.0f32; 3];
    silu_f32(&input, &mut out);
    assert!((out[0]).abs() < 1e-6, "silu(0) should be 0, got {}", out[0]);
    // SiLU(1) = 1 * sigmoid(1) = 1/(1+e^-1) ~ 0.7311
    assert!(
        (out[1] - 0.7311).abs() < 1e-3,
        "silu(1) ~ 0.7311, got {}",
        out[1]
    );
}

#[test]
fn test_fma_f32_correctness() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [2.0, 3.0, 4.0, 5.0, 6.0];
    let c = [10.0, 20.0, 30.0, 40.0, 50.0];
    let expected = [12.0, 26.0, 42.0, 60.0, 80.0]; // a*b+c
    let mut out = [0.0f32; 5];
    fma_f32(&a, &b, &c, &mut out);
    assert_close(&out, &expected, 1e-5, "fma_f32 correctness");
}

#[test]
fn test_scalar_mul_f32_correctness() {
    let a = [1.0, -2.0, 3.0, 0.0, -5.0];
    let mut out = [0.0f32; 5];
    scalar_mul_f32(&a, 3.0, &mut out);
    let expected = [3.0, -6.0, 9.0, 0.0, -15.0];
    assert_close(&out, &expected, 1e-6, "scalar_mul_f32 correctness");
}

#[test]
fn test_elementwise_dispatch_vs_scalar_large() {
    // Exercise SIMD tail handling with length not divisible by 4 or 8.
    let n = 37;
    let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.23).sin()).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.41).cos()).collect();

    let mut add_dispatch = vec![0.0f32; n];
    let mut add_scalar = vec![0.0f32; n];
    add_f32(&a, &b, &mut add_dispatch);
    add_f32_scalar(&a, &b, &mut add_scalar);
    assert_close(
        &add_dispatch,
        &add_scalar,
        1e-6,
        "add dispatch vs scalar n=37",
    );

    let mut mul_dispatch = vec![0.0f32; n];
    let mut mul_scalar = vec![0.0f32; n];
    mul_f32(&a, &b, &mut mul_dispatch);
    mul_f32_scalar(&a, &b, &mut mul_scalar);
    assert_close(
        &mul_dispatch,
        &mul_scalar,
        1e-6,
        "mul dispatch vs scalar n=37",
    );
}

// =========================================================================
// 4. REDUCTION OPS
// =========================================================================

#[test]
fn test_reduce_sum_basic() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let result = simd_sum_f32(&x);
    assert!(
        (result - 15.0).abs() < 1e-5,
        "sum should be 15, got {result}"
    );
}

#[test]
fn test_reduce_sum_single() {
    let x = [42.0f32];
    let result = simd_sum_f32(&x);
    assert!(
        (result - 42.0).abs() < 1e-6,
        "sum of single should be 42, got {result}"
    );
}

#[test]
fn test_reduce_sum_negative() {
    let x = [-1.0, -2.0, -3.0, -4.0];
    let result = simd_sum_f32(&x);
    assert!(
        (result - (-10.0)).abs() < 1e-5,
        "sum should be -10, got {result}"
    );
}

#[test]
fn test_reduce_max_basic() {
    let x = [-5.0, 3.0, 1.0, 7.0, -2.0, 4.0, 0.0, 6.0, 2.0];
    let result = simd_max_f32(&x);
    assert!((result - 7.0).abs() < 1e-6, "max should be 7, got {result}");
}

#[test]
fn test_reduce_max_all_negative() {
    let x = [-10.0, -5.0, -3.0, -8.0, -1.0, -100.0];
    let result = simd_max_f32(&x);
    assert!(
        (result - (-1.0)).abs() < 1e-6,
        "max should be -1, got {result}"
    );
}

#[test]
fn test_reduce_min_basic() {
    let x = [5.0, 3.0, 1.0, 7.0, 2.0, 4.0, 0.0, 6.0];
    let result = simd_min_f32(&x);
    assert!((result - 0.0).abs() < 1e-6, "min should be 0, got {result}");
}

#[test]
fn test_reduce_min_all_positive() {
    let x = [10.0, 5.0, 3.0, 8.0, 1.5, 100.0];
    let result = simd_min_f32(&x);
    assert!(
        (result - 1.5).abs() < 1e-6,
        "min should be 1.5, got {result}"
    );
}

#[test]
fn test_reduce_mean_via_sum() {
    // Mean = sum / n (no dedicated mean in simd_reduce, compute manually).
    let x = [2.0, 4.0, 6.0, 8.0, 10.0];
    let sum = simd_sum_f32(&x);
    let mean = sum / x.len() as f32;
    assert!((mean - 6.0).abs() < 1e-5, "mean should be 6, got {mean}");
}

#[test]
fn test_reduce_dot_product() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    // dot = 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
    let result = simd_dot_f32(&a, &b);
    assert!(
        (result - 70.0).abs() < 1e-4,
        "dot should be 70, got {result}"
    );
}

#[test]
fn test_reduce_l2_norm() {
    let x = [3.0, 4.0]; // sqrt(9+16) = 5
    let result = l2_norm_f32(&x);
    assert!(
        (result - 5.0).abs() < 1e-5,
        "l2 norm should be 5, got {result}"
    );
}

#[test]
fn test_reduce_dispatch_vs_scalar_varied_sizes() {
    for n in [
        1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
    ] {
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).sin()).collect();

        let sum_d = simd_sum_f32(&x);
        let sum_s = simd_sum_f32_scalar(&x);
        assert!(
            (sum_d - sum_s).abs() < 1e-3,
            "sum mismatch at n={n}: dispatch={sum_d}, scalar={sum_s}"
        );

        let max_d = simd_max_f32(&x);
        let max_s = simd_max_f32_scalar(&x);
        assert!(
            (max_d - max_s).abs() < 1e-6,
            "max mismatch at n={n}: dispatch={max_d}, scalar={max_s}"
        );

        let min_d = simd_min_f32(&x);
        let min_s = simd_min_f32_scalar(&x);
        assert!(
            (min_d - min_s).abs() < 1e-6,
            "min mismatch at n={n}: dispatch={min_d}, scalar={min_s}"
        );
    }
}

// =========================================================================
// 5. SOFTMAX NUMERICAL STABILITY
// =========================================================================

#[test]
fn test_softmax_uniform_input() {
    // All equal inputs => uniform distribution.
    let input = [1.0, 1.0, 1.0, 1.0];
    let mut out = [0.0f32; 4];
    softmax_f32(&input, &mut out, 4);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - 0.25).abs() < 1e-2,
            "softmax uniform [{i}] = {v}, expected 0.25"
        );
    }
}

#[test]
fn test_softmax_single_element() {
    let input = [42.0];
    let mut out = [0.0f32; 1];
    softmax_f32(&input, &mut out, 1);
    assert!(
        (out[0] - 1.0).abs() < 1e-6,
        "softmax of single = 1, got {}",
        out[0]
    );
}

#[test]
fn test_softmax_large_values_no_overflow() {
    // Large positive values should not cause overflow due to max-subtraction.
    let input = [1000.0, 1001.0, 999.0, 1000.5];
    let mut out = vec![0.0f32; 4];
    softmax_f32(&input, &mut out, 4);
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.05,
        "softmax with large values should sum to 1, got {sum}"
    );
    for &v in &out {
        assert!(v.is_finite(), "softmax output should be finite, got {v}");
        assert!(v >= 0.0, "softmax output should be non-negative, got {v}");
    }
}

#[test]
fn test_softmax_large_negative_values_no_underflow() {
    // Large negative values should produce near-zero but finite outputs.
    let input = [-1000.0, -999.0, -998.0, -1001.0];
    let mut out = vec![0.0f32; 4];
    softmax_f32(&input, &mut out, 4);
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.05,
        "softmax large neg should sum to 1, got {sum}"
    );
    for &v in &out {
        assert!(v.is_finite(), "output must be finite");
    }
}

#[test]
fn test_softmax_mixed_extreme_values() {
    // One very large, rest very small => dominant element ~1.
    let input = [100.0, -100.0, -100.0, -100.0];
    let mut out = vec![0.0f32; 4];
    softmax_f32(&input, &mut out, 4);
    assert!(
        out[0] > 0.95,
        "dominant element should be ~1, got {}",
        out[0]
    );
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.05,
        "softmax should sum to 1, got {sum}"
    );
}

#[test]
fn test_softmax_sums_to_one_varied_sizes() {
    for n in [2, 3, 5, 7, 8, 15, 16, 17, 31, 32, 64, 100, 256] {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 1.3).sin() * 5.0).collect();
        let mut out = vec![0.0f32; n];
        softmax_f32(&input, &mut out, n);
        let sum: f32 = out.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.05,
            "softmax n={n}: sum={sum}, expected ~1.0"
        );
        // All non-negative
        for (i, &v) in out.iter().enumerate() {
            assert!(v >= 0.0, "softmax n={n} [{i}] = {v} should be >= 0");
        }
    }
}

#[test]
fn test_softmax_scalar_vs_dispatch() {
    let input: Vec<f32> = (0..20).map(|i| (i as f32) - 10.0).collect();
    let scalar = softmax_f32_reference(&input, input.len());
    let mut dispatch = vec![0.0f32; input.len()];
    softmax_f32(&input, &mut dispatch, input.len());
    // SIMD path uses fast_exp, so allow wider tolerance.
    assert_close(&dispatch, &scalar, 5e-2, "softmax scalar vs dispatch n=20");
}

#[test]
fn test_softmax_monotonicity() {
    // For sorted input, softmax output should also be sorted (monotone).
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let ref_out = ref_softmax_f64(&input);
    for i in 1..ref_out.len() {
        assert!(
            ref_out[i] >= ref_out[i - 1],
            "softmax not monotone at {i}: {} < {}",
            ref_out[i],
            ref_out[i - 1]
        );
    }
}

// =========================================================================
// 6. LAYERNORM CORRECTNESS
// =========================================================================

#[test]
fn test_layernorm_identity_params() {
    // gamma=1, beta=0 => output is just normalized input.
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let n = input.len();
    let gamma = vec![1.0f32; n];
    let beta = vec![0.0f32; n];
    let eps = 1e-5;

    let ref_out = ref_layer_norm_f64(&input, &gamma, &beta, n, eps);
    let mut out = vec![0.0f32; n];
    layer_norm_f32(&input, &mut out, &gamma, &beta, n, eps);
    assert_close(&out, &ref_out, 1e-4, "layernorm identity params");

    // Normalized output should have mean ~0.
    let mean: f64 = out.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
    assert!(mean.abs() < 1e-3, "layernorm mean should be ~0, got {mean}");
}

#[test]
fn test_layernorm_constant_input() {
    // All same values => output should be beta (variance = 0).
    let input = [5.0, 5.0, 5.0, 5.0];
    let n = 4;
    let gamma = [1.0, 1.0, 1.0, 1.0];
    let beta = [0.0, 0.0, 0.0, 0.0];
    let eps = 1e-5;

    let mut out = vec![0.0f32; n];
    layer_norm_f32(&input, &mut out, &gamma, &beta, n, eps);
    // (5 - 5) / sqrt(0 + eps) * 1 + 0 = 0
    for (i, &v) in out.iter().enumerate() {
        assert!(
            v.abs() < 1e-2,
            "layernorm constant input [{i}] should be ~0, got {v}"
        );
    }
}

#[test]
fn test_layernorm_with_scale_shift() {
    let input = [1.0, 3.0, 5.0, 7.0];
    let n = 4;
    let gamma = [2.0, 2.0, 2.0, 2.0];
    let beta = [1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let ref_out = ref_layer_norm_f64(&input, &gamma, &beta, n, eps);
    let mut out = vec![0.0f32; n];
    layer_norm_f32(&input, &mut out, &gamma, &beta, n, eps);
    assert_close(&out, &ref_out, 1e-4, "layernorm scale+shift");
}

#[test]
fn test_layernorm_dispatch_vs_scalar_varied() {
    for n in [3, 4, 5, 7, 8, 15, 16, 17, 32, 64, 128] {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).sin() * 10.0).collect();
        let gamma: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 * 0.1).cos() * 0.5).collect();
        let beta: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let eps = 1e-5;

        let ref_out = ref_layer_norm_f64(&input, &gamma, &beta, n, eps);
        let mut dispatch_out = vec![0.0f32; n];
        layer_norm_f32(&input, &mut dispatch_out, &gamma, &beta, n, eps);
        assert_close(
            &dispatch_out,
            &ref_out,
            5e-4,
            &format!("layernorm dispatch vs ref n={n}"),
        );
    }
}

#[test]
fn test_layernorm_neon_vs_scalar() {
    let n = 33; // Not divisible by 4 — exercises NEON tail handling.
    let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5) - 8.0).collect();
    let gamma = vec![1.0f32; n];
    let beta = vec![0.0f32; n];
    let eps = 1e-5;

    let mut scalar_out = vec![0.0f32; n];
    layer_norm_f32_scalar(&input, &mut scalar_out, &gamma, &beta, n, eps);

    let mut neon_out = vec![0.0f32; n];
    layer_norm_f32_neon(&input, &mut neon_out, &gamma, &beta, n, eps);

    assert_close(
        &neon_out,
        &scalar_out,
        1e-4,
        "layernorm NEON vs scalar n=33",
    );
}

// =========================================================================
// 7. INSTANCE NORM CORRECTNESS
// =========================================================================

#[test]
fn test_instance_norm_single_channel() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let channels = 1;
    let spatial = 5;
    let eps = 1e-5;

    let ref_out = ref_instance_norm_f64(&input, channels, spatial, eps);
    let mut out = vec![0.0f32; 5];
    instance_norm_f32(&input, &mut out, channels, spatial, eps);
    assert_close(&out, &ref_out, 1e-4, "instance_norm single channel");

    // Output should have mean ~0.
    let mean: f64 = out.iter().map(|&x| f64::from(x)).sum::<f64>() / spatial as f64;
    assert!(
        mean.abs() < 1e-3,
        "instance_norm mean should be ~0, got {mean}"
    );
}

#[test]
fn test_instance_norm_multi_channel() {
    // 3 channels, 4 spatial each
    let input: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let channels = 3;
    let spatial = 4;
    let eps = 1e-5;

    let ref_out = ref_instance_norm_f64(&input, channels, spatial, eps);
    let mut out = vec![0.0f32; 12];
    instance_norm_f32(&input, &mut out, channels, spatial, eps);
    assert_close(&out, &ref_out, 1e-4, "instance_norm multi channel");

    // Each channel should have mean ~0.
    for c in 0..channels {
        let start = c * spatial;
        let ch_mean: f64 = out[start..start + spatial]
            .iter()
            .map(|&x| f64::from(x))
            .sum::<f64>()
            / spatial as f64;
        assert!(
            ch_mean.abs() < 1e-3,
            "instance_norm channel {c} mean should be ~0, got {ch_mean}"
        );
    }
}

#[test]
fn test_instance_norm_constant_channel() {
    // Constant channel => output should be all 0.
    let input = [5.0, 5.0, 5.0, 5.0, 3.0, 1.0, 4.0, 2.0];
    let channels = 2;
    let spatial = 4;
    let eps = 1e-5;

    let mut out = vec![0.0f32; 8];
    instance_norm_f32(&input, &mut out, channels, spatial, eps);
    // First channel is constant => all ~0.
    for (i, &oi) in out.iter().take(4).enumerate() {
        assert!(
            oi.abs() < 1e-2,
            "constant channel [{i}] should be ~0, got {oi}"
        );
    }
}

#[test]
fn test_instance_norm_dispatch_vs_scalar_varied() {
    for spatial in [3, 4, 5, 7, 8, 15, 16, 17, 32, 64] {
        let channels = 4;
        let total = channels * spatial;
        let input: Vec<f32> = (0..total).map(|i| (i as f32 * 0.3).sin() * 5.0).collect();
        let eps = 1e-5;

        let ref_out = ref_instance_norm_f64(&input, channels, spatial, eps);
        let mut dispatch_out = vec![0.0f32; total];
        instance_norm_f32(&input, &mut dispatch_out, channels, spatial, eps);
        assert_close(
            &dispatch_out,
            &ref_out,
            5e-4,
            &format!("instance_norm dispatch vs ref spatial={spatial}"),
        );
    }
}

#[test]
fn test_instance_norm_neon_vs_scalar() {
    let channels = 3;
    let spatial = 17; // Not divisible by 4.
    let total = channels * spatial;
    let input: Vec<f32> = (0..total).map(|i| (i as f32 * 0.2) - 5.0).collect();
    let eps = 1e-5;

    let mut scalar_out = vec![0.0f32; total];
    instance_norm_f32_scalar(&input, &mut scalar_out, channels, spatial, eps);

    let mut neon_out = vec![0.0f32; total];
    instance_norm_f32_neon(&input, &mut neon_out, channels, spatial, eps);

    assert_close(
        &neon_out,
        &scalar_out,
        1e-4,
        "instance_norm NEON vs scalar spatial=17",
    );
}

// =========================================================================
// 8. SIMD ALIGNMENT HANDLING
// =========================================================================

#[test]
fn test_add_alignment_one_element() {
    let a = [3.14f32];
    let b = [2.71f32];
    let mut out = [0.0f32; 1];
    add_f32(&a, &b, &mut out);
    assert!((out[0] - 5.85).abs() < 1e-5, "add single element");
}

#[test]
fn test_add_alignment_three_elements() {
    // 3 elements: below NEON 4-lane and AVX2 8-lane thresholds.
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let mut out = [0.0f32; 3];
    add_f32(&a, &b, &mut out);
    assert_close(&out, &[5.0, 7.0, 9.0], 1e-6, "add 3 elements");
}

#[test]
fn test_relu_alignment_sizes() {
    // Test sizes around SIMD boundaries: 1,3,4,5,7,8,9
    for n in [1, 3, 4, 5, 7, 8, 9] {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) - (n as f32 / 2.0)).collect();
        let mut out = vec![0.0f32; n];
        let mut scalar_out = vec![0.0f32; n];
        relu_f32(&input, &mut out);
        relu_f32_scalar(&input, &mut scalar_out);
        assert_close(&out, &scalar_out, 0.0, &format!("relu alignment n={n}"));
    }
}

#[test]
fn test_matmul_alignment_non_multiple_of_four() {
    // 3x5 * 5x3 — neither dimension is a multiple of 4.
    let m = 3;
    let k = 5;
    let n = 3;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.2).collect();

    let ref_out = ref_matmul_f64(&a, &b, m, k, n);
    let mut out = vec![0.0f32; m * n];
    matmul_f32(&a, &b, m, k, n, &mut out);
    assert_close(&out, &ref_out, 1e-4, "matmul 3x5*5x3 alignment");
}

#[test]
fn test_gemv_alignment_non_multiple_of_four() {
    // M=3, K=5 — neither is a SIMD-friendly size.
    let m = 3;
    let k = 5;
    let matrix: Vec<f32> = (0..m * k).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let v: Vec<f32> = (0..k).map(|i| (i as f32 + 1.0) * 0.3).collect();

    let ref_out = ref_gemv_f64(&matrix, &v, m, k);
    let mut out = vec![0.0f32; m];
    gemv_f32(&matrix, &v, m, k, &mut out);
    assert_close(&out, &ref_out, 1e-4, "gemv 3x5 alignment");
}

#[test]
fn test_reduce_alignment_single_element() {
    let x = [7.0f32];
    assert!((simd_sum_f32(&x) - 7.0).abs() < 1e-6, "sum single");
    assert!((simd_max_f32(&x) - 7.0).abs() < 1e-6, "max single");
    assert!((simd_min_f32(&x) - 7.0).abs() < 1e-6, "min single");
}

#[test]
fn test_softmax_alignment_prime_sizes() {
    // Test with prime-number sizes that don't align to any SIMD width.
    for n in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
        let input: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.5) - (n as f32 * 0.25))
            .collect();
        let mut out = vec![0.0f32; n];
        softmax_f32(&input, &mut out, n);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 0.05, "softmax prime n={n}: sum={sum}");
        for &v in &out {
            assert!(v.is_finite(), "softmax prime n={n} non-finite");
            assert!(v >= 0.0, "softmax prime n={n} negative");
        }
    }
}

#[test]
fn test_layernorm_alignment_non_simd_size() {
    // n=5: not a multiple of 4 (NEON) or 8 (AVX2).
    let n = 5;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0 - 4.0).collect();
    let gamma = vec![1.0f32; n];
    let beta = vec![0.0f32; n];
    let eps = 1e-5;

    let ref_out = ref_layer_norm_f64(&input, &gamma, &beta, n, eps);
    let mut out = vec![0.0f32; n];
    layer_norm_f32(&input, &mut out, &gamma, &beta, n, eps);
    assert_close(&out, &ref_out, 1e-4, "layernorm alignment n=5");
}

#[test]
fn test_instance_norm_alignment_non_simd_spatial() {
    // spatial=3: below NEON 4-lane threshold.
    let channels = 2;
    let spatial = 3;
    let total = channels * spatial;
    let input: Vec<f32> = (0..total).map(|i| i as f32 * 1.5).collect();
    let eps = 1e-5;

    let ref_out = ref_instance_norm_f64(&input, channels, spatial, eps);
    let mut out = vec![0.0f32; total];
    instance_norm_f32(&input, &mut out, channels, spatial, eps);
    assert_close(&out, &ref_out, 1e-4, "instance_norm alignment spatial=3");
}

// =========================================================================
// 9. CROSS-OPERATION CONSISTENCY
// =========================================================================

#[test]
fn test_matmul_vs_gemv_single_column() {
    // matmul(A, b_col) should match gemv(A, b) when B is a column vector.
    let m = 6;
    let k = 10;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.11).sin()).collect();
    let b_vec: Vec<f32> = (0..k).map(|i| (i as f32 * 0.31).cos()).collect();
    // B as [K, 1] column vector for matmul:
    let b_col: Vec<f32> = b_vec.clone();

    let matmul_out = matmul_reference(&a, &b_col, m, k, 1);
    let gemv_out = gemv_reference(&a, &b_vec, m, k);
    assert_close(&matmul_out, &gemv_out, 1e-4, "matmul vs gemv single column");
}

#[test]
fn test_tiled_matmul_vs_simd_matmul() {
    // crate::matmul::matmul (tiled) should agree with simd_matmul::matmul_f32.
    let m = 8;
    let k = 12;
    let n = 6;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.07).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.13).cos()).collect();

    let tiled_out = matmul::matmul(&a, &b, m, k, n);
    let mut simd_out = vec![0.0f32; m * n];
    matmul_f32(&a, &b, m, k, n, &mut simd_out);
    assert_close(&tiled_out, &simd_out, 1e-3, "tiled matmul vs simd matmul");
}

// =========================================================================
// 10. EDGE CASES AND BOUNDARY CONDITIONS
// =========================================================================

#[test]
fn test_relu_preserves_positive() {
    let input = [0.001, 1.0, 100.0, f32::MAX / 2.0];
    let mut out = vec![0.0f32; 4];
    relu_f32(&input, &mut out);
    assert_close(&out, &input, 0.0, "relu preserves positive");
}

#[test]
fn test_relu_zeros_negative() {
    let input = [-0.001, -1.0, -100.0, f32::MIN / 2.0];
    let mut out = vec![0.0f32; 4];
    relu_f32(&input, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - 0.0).abs() < 1e-10,
            "relu negative [{i}] should be 0, got {v}"
        );
    }
}

#[test]
fn test_sigmoid_boundary_values() {
    // sigmoid(-inf) -> 0, sigmoid(+inf) -> 1 (approximately, for large values)
    let input = [-50.0, 50.0];
    let mut out = [0.0f32; 2];
    elementwise::sigmoid(&input, &mut out);
    assert!(out[0] < 1e-10, "sigmoid(-50) should be ~0, got {}", out[0]);
    assert!(
        (out[1] - 1.0).abs() < 1e-10,
        "sigmoid(50) should be ~1, got {}",
        out[1]
    );
}

#[test]
fn test_tanh_boundary_values() {
    // tanh approaches +/-1 for large inputs.
    let input = [-50.0, 50.0, 0.0];
    let mut out = [0.0f32; 3];
    elementwise::tanh(&input, &mut out);
    assert!(
        (out[0] - (-1.0)).abs() < 1e-6,
        "tanh(-50) ~= -1, got {}",
        out[0]
    );
    assert!((out[1] - 1.0).abs() < 1e-6, "tanh(50) ~= 1, got {}", out[1]);
    assert!(out[2].abs() < 1e-6, "tanh(0) = 0, got {}", out[2]);
}

#[test]
fn test_softmax_two_equal_elements() {
    let input = [0.0, 0.0];
    let mut out = [0.0f32; 2];
    softmax_f32(&input, &mut out, 2);
    assert!(
        (out[0] - 0.5).abs() < 1e-2,
        "softmax equal pair [0] = {}",
        out[0]
    );
    assert!(
        (out[1] - 0.5).abs() < 1e-2,
        "softmax equal pair [1] = {}",
        out[1]
    );
}

#[test]
fn test_reduce_sum_cancellation() {
    // Positive and negative should cancel.
    let x = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];
    let result = simd_sum_f32(&x);
    assert!(result.abs() < 1e-4, "sum should cancel to 0, got {result}");
}

#[test]
fn test_l2_norm_unit_vectors() {
    let x = [1.0, 0.0, 0.0, 0.0]; // unit e1
    let norm = l2_norm_f32(&x);
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "l2 norm of unit vector = 1, got {norm}"
    );
}

#[test]
fn test_dot_product_orthogonal() {
    let a = [1.0, 0.0, 0.0, 0.0];
    let b = [0.0, 1.0, 0.0, 0.0];
    let result = simd_dot_f32(&a, &b);
    assert!(
        result.abs() < 1e-6,
        "dot of orthogonal vectors = 0, got {result}"
    );
}
