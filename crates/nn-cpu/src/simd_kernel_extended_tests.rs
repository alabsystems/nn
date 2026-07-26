// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended CPU SIMD kernel tests covering softmax, layernorm, rmsnorm,
//! matmul, elementwise ops, reductions, transpose, conv1d, and embedding.
//!
//! Each test compares the SIMD auto-dispatch path against a known reference
//! (scalar or analytical) with appropriate tolerance for fast-exp and
//! float accumulation differences.

use super::*;

// =========================================================================
// Helpers
// =========================================================================

/// Naive f64-precision softmax for ground truth.
fn naive_softmax_f64(input: &[f32]) -> Vec<f32> {
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = input.iter().map(|&x| f64::from(x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| (e / sum) as f32).collect()
}

/// Naive f64-precision GELU for ground truth.
fn naive_gelu_f64(x: f32) -> f32 {
    let v = f64::from(x);
    let c = (2.0_f64 / std::f64::consts::PI).sqrt();
    (v * 0.5 * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh())) as f32
}

/// Naive f64-precision SiLU for ground truth.
fn naive_silu_f64(x: f32) -> f32 {
    let v = f64::from(x);
    (v / (1.0 + (-v).exp())) as f32
}

/// Assert two slices are element-wise close within `tol`.
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
            diff < tol,
            "{label}[{i}]: {va} vs {vb}, diff={diff}, tol={tol}"
        );
    }
}

// =========================================================================
// 1. SOFTMAX
// =========================================================================

#[test]
fn test_softmax_1d_small() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let expected = naive_softmax_f64(&input);
    let got = softmax_f32_reference(&input, input.len());
    assert_close(&got, &expected, 1e-5, "softmax_1d_small");
}

#[test]
fn test_softmax_1d_negative() {
    let input = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0];
    let expected = naive_softmax_f64(&input);
    let mut out = vec![0.0f32; input.len()];
    softmax_f32(&input, &mut out, input.len());
    // SIMD path uses fast_exp; allow wider tolerance.
    assert_close(&out, &expected, 5e-2, "softmax_1d_negative_dispatch");
}

#[test]
fn test_softmax_2d_rows() {
    // Apply softmax independently to each row of a 3x4 matrix.
    let data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.5 - 3.0).collect();
    for row in 0..3 {
        let start = row * 4;
        let row_slice = &data[start..start + 4];
        let expected = naive_softmax_f64(row_slice);
        let mut out = vec![0.0f32; 4];
        softmax_f32(row_slice, &mut out, 4);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-2, "softmax row {row} sum={sum}");
        assert_close(&out, &expected, 5e-2, &format!("softmax_2d_row{row}"));
    }
}

#[test]
fn test_softmax_batched_varied_sizes() {
    for n in [1, 3, 5, 7, 8, 9, 15, 16, 17, 32, 64, 128, 256] {
        let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.7).sin()).collect();
        let mut out = vec![0.0f32; n];
        softmax_f32(&input, &mut out, n);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 5e-2, "softmax batch n={n} sum={sum}");
        for (i, &v) in out.iter().enumerate() {
            assert!(v >= 0.0 && v.is_finite(), "softmax n={n} [{i}]={v}");
        }
    }
}

#[test]
fn test_softmax_single_element() {
    for val in [-100.0, -1.0, 0.0, 1.0, 100.0] {
        let input = [val];
        let mut out = vec![0.0f32; 1];
        softmax_f32(&input, &mut out, 1);
        assert!(
            (out[0] - 1.0).abs() < 1e-3,
            "softmax single({val}) = {}",
            out[0]
        );
    }
}

#[test]
fn test_softmax_empty() {
    let input: &[f32] = &[];
    let mut output: Vec<f32> = vec![];
    softmax_f32(input, &mut output, 0);
    // Should be a no-op, no crash.
}

#[test]
fn test_softmax_large_values_stability() {
    let input = [800.0, 801.0, 802.0, 803.0];
    let mut out = vec![0.0f32; 4];
    softmax_f32_scalar(&input, &mut out, 4);
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax large stability sum={sum}"
    );
    for &v in &out {
        assert!(v.is_finite() && v >= 0.0, "softmax large: {v}");
    }
}

// =========================================================================
// 2. LAYERNORM
// =========================================================================

#[test]
fn test_layernorm_basic() {
    let n = 8;
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let gamma = vec![1.0f32; n];
    let beta = vec![0.0f32; n];
    let mut out = vec![0.0f32; n];
    layer_norm_f32(&input, &mut out, &gamma, &beta, n, 1e-5);
    // Mean should be ~3.5, output should be zero-mean.
    let mean: f32 = out.iter().sum::<f32>() / n as f32;
    assert!(mean.abs() < 1e-3, "layernorm basic mean={mean}");
}

#[test]
fn test_layernorm_identity_transform() {
    // gamma=1, beta=0 should produce normalized output.
    let n = 16;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 8.0) * 0.3).collect();
    let gamma = vec![1.0f32; n];
    let beta = vec![0.0f32; n];
    let ref_out = layer_norm_f32_reference(&input, &gamma, &beta, n, 1e-5);
    let mut dispatch_out = vec![0.0f32; n];
    layer_norm_f32(&input, &mut dispatch_out, &gamma, &beta, n, 1e-5);
    assert_close(&dispatch_out, &ref_out, 1e-3, "layernorm_identity");
}

#[test]
fn test_layernorm_various_eps() {
    let n = 8;
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let gamma = vec![1.0f32; n];
    let beta = vec![0.0f32; n];
    for &eps in &[1e-12, 1e-5, 1e-3, 0.1] {
        let ref_out = layer_norm_f32_reference(&input, &gamma, &beta, n, eps);
        let mut out = vec![0.0f32; n];
        layer_norm_f32(&input, &mut out, &gamma, &beta, n, eps);
        assert_close(&out, &ref_out, 1e-3, &format!("layernorm_eps_{eps}"));
    }
}

#[test]
fn test_layernorm_with_scale_shift() {
    let n = 4;
    let input = [2.0, 4.0, 6.0, 8.0];
    let gamma = [2.0, 0.5, 1.0, 3.0];
    let beta = [1.0, -1.0, 0.0, 0.5];
    let ref_out = layer_norm_f32_reference(&input, &gamma, &beta, n, 1e-5);
    let mut out = vec![0.0f32; n];
    layer_norm_f32(&input, &mut out, &gamma, &beta, n, 1e-5);
    assert_close(&out, &ref_out, 1e-3, "layernorm_scale_shift");
}

#[test]
fn test_layernorm_constant_input() {
    // All-same values -> output == beta (since (x-mean)/std = 0).
    let n = 8;
    let input = vec![5.0f32; n];
    let gamma = vec![2.0f32; n];
    let beta: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mut out = vec![0.0f32; n];
    layer_norm_f32(&input, &mut out, &gamma, &beta, n, 1e-5);
    for (i, (&o, &b)) in out.iter().zip(beta.iter()).enumerate() {
        assert!(
            (o - b).abs() < 1e-2,
            "layernorm_const [{i}]: {o} vs expected {b}"
        );
    }
}

#[test]
fn test_layernorm_large_dimension() {
    let n = 1024;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.73).sin()).collect();
    let gamma = vec![1.0f32; n];
    let beta = vec![0.0f32; n];
    let ref_out = layer_norm_f32_reference(&input, &gamma, &beta, n, 1e-5);
    let mut out = vec![0.0f32; n];
    layer_norm_f32(&input, &mut out, &gamma, &beta, n, 1e-5);
    assert_close(&out, &ref_out, 1e-2, "layernorm_large_1024");
}

// =========================================================================
// 3. RMSNORM
// =========================================================================

#[test]
fn test_rmsnorm_basic() {
    let n = 8;
    let input: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let weight = vec![1.0f32; n];
    let ref_out = rmsnorm_reference(&input, &weight, n, 1e-5);
    let mut out = vec![0.0f32; n];
    rmsnorm(&input, &weight, &mut out, n, 1e-5);
    assert_close(&out, &ref_out, 1e-4, "rmsnorm_basic");
}

#[test]
fn test_rmsnorm_with_weights() {
    let n = 4;
    let input = [1.0, 2.0, 3.0, 4.0];
    let weight = [0.5, 2.0, 1.0, 0.1];
    let ref_out = rmsnorm_reference(&input, &weight, n, 1e-5);
    let mut out = vec![0.0f32; n];
    rmsnorm(&input, &weight, &mut out, n, 1e-5);
    assert_close(&out, &ref_out, 1e-4, "rmsnorm_weighted");
}

#[test]
fn test_rmsnorm_various_eps() {
    let n = 8;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 + 0.1) * 0.5).collect();
    let weight = vec![1.0f32; n];
    for &eps in &[1e-8, 1e-5, 1e-3, 0.1] {
        let ref_out = rmsnorm_reference(&input, &weight, n, eps);
        let mut out = vec![0.0f32; n];
        rmsnorm(&input, &weight, &mut out, n, eps);
        assert_close(&out, &ref_out, 1e-3, &format!("rmsnorm_eps_{eps}"));
    }
}

#[test]
fn test_rmsnorm_batch() {
    let batch = 3;
    let hidden = 8;
    let input: Vec<f32> = (0..(batch * hidden)).map(|i| (i as f32) * 0.1).collect();
    let weight = vec![1.0f32; hidden];
    let mut out = vec![0.0f32; batch * hidden];
    rmsnorm_batch(&input, &weight, &mut out, batch, hidden, 1e-5);
    // Each row should match independent rmsnorm.
    for b in 0..batch {
        let start = b * hidden;
        let row = &input[start..start + hidden];
        let ref_row = rmsnorm_reference(row, &weight, hidden, 1e-5);
        assert_close(
            &out[start..start + hidden],
            &ref_row,
            1e-4,
            &format!("rmsnorm_batch_row{b}"),
        );
    }
}

#[test]
fn test_rmsnorm_large_dimension() {
    let n = 1024;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.31).cos()).collect();
    let weight = vec![1.0f32; n];
    let ref_out = rmsnorm_reference(&input, &weight, n, 1e-5);
    let mut out = vec![0.0f32; n];
    rmsnorm(&input, &weight, &mut out, n, 1e-5);
    assert_close(&out, &ref_out, 1e-3, "rmsnorm_large_1024");
}

// =========================================================================
// 4. MATMUL
// =========================================================================

#[test]
fn test_matmul_square_small() {
    // 2x2 * 2x2
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let expected = [
        1.0 * 5.0 + 2.0 * 7.0,
        1.0 * 6.0 + 2.0 * 8.0,
        3.0 * 5.0 + 4.0 * 7.0,
        3.0 * 6.0 + 4.0 * 8.0,
    ];
    let mut out = vec![0.0f32; 4];
    matmul_f32(&a, &b, 2, 2, 2, &mut out);
    assert_close(&out, &expected, 1e-5, "matmul_2x2");
}

#[test]
fn test_matmul_rectangular() {
    // [2, 3] * [3, 4] -> [2, 4]
    let a: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let b: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let ref_out = matmul_reference(&a, &b, 2, 3, 4);
    let mut out = vec![0.0f32; 8];
    matmul_f32(&a, &b, 2, 3, 4, &mut out);
    assert_close(&out, &ref_out, 1e-4, "matmul_rect_2x3x4");
}

#[test]
fn test_matmul_tall_skinny() {
    // [8, 2] * [2, 1] -> [8, 1]
    let a: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    let b = [1.0, -1.0];
    let ref_out = matmul_reference(&a, &b, 8, 2, 1);
    let mut out = vec![0.0f32; 8];
    matmul_f32(&a, &b, 8, 2, 1, &mut out);
    assert_close(&out, &ref_out, 1e-4, "matmul_tall_skinny");
}

#[test]
fn test_matmul_wide() {
    // [1, 4] * [4, 8] -> [1, 8]
    let a = [1.0, 2.0, 3.0, 4.0];
    let b: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let ref_out = matmul_reference(&a, &b, 1, 4, 8);
    let mut out = vec![0.0f32; 8];
    matmul_f32(&a, &b, 1, 4, 8, &mut out);
    assert_close(&out, &ref_out, 1e-3, "matmul_wide");
}

#[test]
fn test_matmul_identity() {
    // A * I = A for 4x4.
    let n = 4;
    let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let mut eye = vec![0.0f32; 16];
    for i in 0..n {
        eye[i * n + i] = 1.0;
    }
    let mut out = vec![0.0f32; 16];
    matmul_f32(&a, &eye, n, n, n, &mut out);
    assert_close(&out, &a, 1e-5, "matmul_identity");
}

#[test]
fn test_matmul_batched_via_loop() {
    // Simulate batched matmul: batch=3, [4,4]*[4,4].
    let n = 4;
    let total = n * n;
    for batch in 0..3_usize {
        let offset = (batch as f32) * 0.1;
        let a: Vec<f32> = (0..total).map(|i| (i as f32 + offset) * 0.3).collect();
        let b: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.7).sin()).collect();
        let ref_out = matmul_reference(&a, &b, n, n, n);
        let mut out = vec![0.0f32; total];
        matmul_f32(&a, &b, n, n, n, &mut out);
        assert_close(&out, &ref_out, 1e-3, &format!("matmul_batch_{batch}"));
    }
}

#[test]
fn test_matmul_large_square() {
    let n = 32;
    let a: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.13).sin()).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.07).cos()).collect();
    let ref_out = matmul_reference(&a, &b, n, n, n);
    let mut out = vec![0.0f32; n * n];
    matmul_f32(&a, &b, n, n, n, &mut out);
    assert_close(&out, &ref_out, 1e-1, "matmul_large_32x32");
}

#[test]
fn test_matmul_zero_dimension() {
    // m=0 should produce empty output.
    let a: &[f32] = &[];
    let b = [1.0, 2.0];
    let mut out: Vec<f32> = vec![];
    matmul_f32(a, &b, 0, 1, 2, &mut out);
    assert!(out.is_empty());
}

// =========================================================================
// 5. ELEMENTWISE OPS
// =========================================================================

#[test]
fn test_add_basic() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [10.0, 20.0, 30.0, 40.0, 50.0];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
    let mut out = vec![0.0f32; 5];
    add_f32(&a, &b, &mut out);
    assert_close(&out, &expected, 1e-6, "add_basic");
}

#[test]
fn test_add_negative_values() {
    let a = [-1.0, -2.0, -3.0, 0.0];
    let b = [1.0, 2.0, 3.0, 0.0];
    let expected = [0.0, 0.0, 0.0, 0.0];
    let mut out = vec![0.0f32; 4];
    add_f32(&a, &b, &mut out);
    assert_close(&out, &expected, 1e-6, "add_negative");
}

#[test]
fn test_mul_basic() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [2.0, 3.0, 0.5, 0.25, -1.0, 0.0, 1.0, -0.5];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();
    let mut out = vec![0.0f32; 8];
    mul_f32(&a, &b, &mut out);
    assert_close(&out, &expected, 1e-5, "mul_basic");
}

#[test]
fn test_scalar_mul() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let scalar = 2.5;
    let expected: Vec<f32> = a.iter().map(|&x| x * scalar).collect();
    let mut out = vec![0.0f32; 5];
    scalar_mul_f32(&a, scalar, &mut out);
    assert_close(&out, &expected, 1e-6, "scalar_mul");
}

#[test]
fn test_fma_basic() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let c = [0.1, 0.2, 0.3, 0.4];
    let expected: Vec<f32> = (0..4).map(|i| a[i] * b[i] + c[i]).collect();
    let mut out = vec![0.0f32; 4];
    fma_f32(&a, &b, &c, &mut out);
    assert_close(&out, &expected, 1e-5, "fma_basic");
}

#[test]
fn test_relu_basic() {
    let x = [-3.0, -1.0, 0.0, 1.0, 3.0, -0.5, 0.5, 100.0];
    let expected: Vec<f32> = x.iter().map(|&v: &f32| v.max(0.0)).collect();
    let mut out = vec![0.0f32; x.len()];
    relu_f32(&x, &mut out);
    assert_close(&out, &expected, 1e-6, "relu_basic");
}

#[test]
fn test_relu_all_negative() {
    let x = [-5.0, -4.0, -3.0, -2.0, -1.0];
    let expected = vec![0.0f32; 5];
    let mut out = vec![0.0f32; 5];
    relu_f32(&x, &mut out);
    assert_close(&out, &expected, 1e-6, "relu_all_negative");
}

#[test]
fn test_gelu_basic() {
    let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let expected: Vec<f32> = x.iter().map(|&v| naive_gelu_f64(v)).collect();
    let mut out = vec![0.0f32; x.len()];
    gelu_f32(&x, &mut out);
    assert_close(&out, &expected, 1e-4, "gelu_basic");
}

#[test]
fn test_gelu_zero() {
    let x = [0.0];
    let mut out = vec![0.0f32; 1];
    gelu_f32(&x, &mut out);
    assert!(out[0].abs() < 1e-6, "gelu(0) should be ~0, got {}", out[0]);
}

#[test]
fn test_silu_basic() {
    let x = [-3.0, -1.0, 0.0, 1.0, 3.0];
    let expected: Vec<f32> = x.iter().map(|&v| naive_silu_f64(v)).collect();
    let mut out = vec![0.0f32; x.len()];
    silu_f32(&x, &mut out);
    assert_close(&out, &expected, 1e-4, "silu_basic");
}

#[test]
fn test_silu_zero() {
    let x = [0.0];
    let mut out = vec![0.0f32; 1];
    silu_f32(&x, &mut out);
    assert!(out[0].abs() < 1e-6, "silu(0) should be 0, got {}", out[0]);
}

#[test]
fn test_elementwise_varied_sizes() {
    // Exercises SIMD tail handling for add across many sizes.
    for n in [
        1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 256,
    ] {
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
        let mut out = vec![0.0f32; n];
        add_f32(&a, &b, &mut out);
        assert_close(&out, &expected, 1e-5, &format!("add_size_{n}"));
    }
}

// =========================================================================
// 6. REDUCTIONS
// =========================================================================

#[test]
fn test_sum_basic() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let expected: f32 = 15.0;
    let got = simd_sum_f32(&x);
    assert!(
        (got - expected).abs() < 1e-5,
        "sum_basic: {got} vs {expected}"
    );
}

#[test]
fn test_sum_empty() {
    let x: &[f32] = &[];
    let got = simd_sum_f32(x);
    assert!((got - 0.0).abs() < 1e-6, "sum_empty: {got}");
}

#[test]
fn test_sum_single() {
    let x = [42.0];
    let got = simd_sum_f32(&x);
    assert!((got - 42.0).abs() < 1e-5, "sum_single: {got}");
}

#[test]
fn test_sum_negative() {
    let x = [-1.0, -2.0, -3.0, -4.0];
    let got = simd_sum_f32(&x);
    assert!((got - (-10.0)).abs() < 1e-5, "sum_neg: {got}");
}

#[test]
fn test_max_basic() {
    let x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let got = simd_max_f32(&x);
    assert!((got - 9.0).abs() < 1e-6, "max_basic: {got}");
}

#[test]
fn test_max_all_negative() {
    let x = [-5.0, -4.0, -3.0, -2.0, -1.0];
    let got = simd_max_f32(&x);
    assert!((got - (-1.0)).abs() < 1e-6, "max_all_neg: {got}");
}

#[test]
fn test_max_single() {
    let x = [7.0];
    let got = simd_max_f32(&x);
    assert!((got - 7.0).abs() < 1e-6, "max_single: {got}");
}

#[test]
fn test_min_basic() {
    let x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let got = simd_min_f32(&x);
    assert!((got - 1.0).abs() < 1e-6, "min_basic: {got}");
}

#[test]
fn test_min_all_positive() {
    let x = [10.0, 20.0, 5.0, 100.0];
    let got = simd_min_f32(&x);
    assert!((got - 5.0).abs() < 1e-6, "min_all_pos: {got}");
}

#[test]
fn test_dot_product() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0;
    let got = simd_dot_f32(&a, &b);
    assert!((got - expected).abs() < 1e-4, "dot: {got} vs {expected}");
}

#[test]
fn test_dot_orthogonal() {
    let a = [1.0, 0.0, 0.0, 0.0];
    let b = [0.0, 1.0, 0.0, 0.0];
    let got = simd_dot_f32(&a, &b);
    assert!(got.abs() < 1e-6, "dot_orthogonal: {got}");
}

#[test]
fn test_l2_norm() {
    let x = [3.0, 4.0];
    let got = l2_norm_f32(&x);
    assert!((got - 5.0).abs() < 1e-5, "l2_norm: {got}");
}

#[test]
fn test_l2_norm_unit() {
    let x = [1.0, 0.0, 0.0, 0.0];
    let got = l2_norm_f32(&x);
    assert!((got - 1.0).abs() < 1e-5, "l2_norm_unit: {got}");
}

#[test]
fn test_sum_mean_via_reduction() {
    // Test computing mean = sum / n using reductions.
    let n = 100;
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let sum = simd_sum_f32(&x);
    let mean = sum / n as f32;
    let expected_mean = 49.5;
    assert!(
        (mean - expected_mean).abs() < 1e-2,
        "mean_via_reduction: {mean} vs {expected_mean}"
    );
}

#[test]
fn test_reductions_varied_sizes() {
    for n in [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 64, 128, 256] {
        let x: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        let sum = simd_sum_f32(&x);
        let expected_sum: f32 = (1..=n).map(|i| i as f32).sum();
        assert!(
            (sum - expected_sum).abs() < 1.0,
            "sum_varied n={n}: {sum} vs {expected_sum}"
        );
        let max = simd_max_f32(&x);
        assert!((max - n as f32).abs() < 1e-5, "max_varied n={n}: {max}");
    }
}

// =========================================================================
// 7. TRANSPOSE
// =========================================================================

#[test]
fn test_transpose_2d_small() {
    // 2x3 -> 3x2
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let mut out = vec![0.0f32; 6];
    transpose_2d(&input, &mut out, 2, 3);
    assert_close(&out, &expected, 1e-6, "transpose_2x3");
}

#[test]
fn test_transpose_square() {
    let n = 4;
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let expected = transpose_reference(&input, n, n);
    let mut out = vec![0.0f32; 16];
    transpose_2d(&input, &mut out, n, n);
    assert_close(&out, &expected, 1e-6, "transpose_4x4");
}

#[test]
fn test_transpose_involution() {
    // Transpose twice should give back the original.
    let rows = 5;
    let cols = 7;
    let input: Vec<f32> = (0..35).map(|i| i as f32 * 0.3).collect();
    let mut t1 = vec![0.0f32; 35];
    let mut t2 = vec![0.0f32; 35];
    transpose_2d(&input, &mut t1, rows, cols);
    transpose_2d(&t1, &mut t2, cols, rows);
    assert_close(&t2, &input, 1e-6, "transpose_involution");
}

#[test]
fn test_transpose_single_row() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let expected = input.to_vec(); // 1x5 -> 5x1 (same memory layout)
    let mut out = vec![0.0f32; 5];
    transpose_2d(&input, &mut out, 1, 5);
    assert_close(&out, &expected, 1e-6, "transpose_1xN");
}

#[test]
fn test_transpose_single_column() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let expected = input.to_vec(); // 5x1 -> 1x5 (same memory layout)
    let mut out = vec![0.0f32; 5];
    transpose_2d(&input, &mut out, 5, 1);
    assert_close(&out, &expected, 1e-6, "transpose_Nx1");
}

#[test]
fn test_transpose_large() {
    let rows = 32;
    let cols = 64;
    let input: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.17).sin()).collect();
    let expected = transpose_reference(&input, rows, cols);
    let mut out = vec![0.0f32; rows * cols];
    transpose_2d(&input, &mut out, rows, cols);
    assert_close(&out, &expected, 1e-6, "transpose_32x64");
}

#[test]
fn test_transpose_reference_matches_dispatch() {
    for (rows, cols) in [(3, 5), (8, 8), (7, 13), (16, 4)] {
        let input: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let ref_out = transpose_reference(&input, rows, cols);
        let mut out = vec![0.0f32; rows * cols];
        transpose_2d(&input, &mut out, rows, cols);
        assert_close(
            &out,
            &ref_out,
            1e-6,
            &format!("transpose_ref_{rows}x{cols}"),
        );
    }
}

// =========================================================================
// 8. CONV1D
// =========================================================================

#[test]
fn test_conv1d_basic_no_padding() {
    // 1 input channel, 1 output channel, kernel_size=3, stride=1, no padding.
    // Input: [1, 2, 3, 4, 5] -> output length = 5 - 3 + 1 = 3
    let cfg = Conv1dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: 3,
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
    };
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = [1.0, 1.0, 1.0]; // sum kernel
    let ref_out = conv1d_full_reference(&input, &weight, None, &cfg).unwrap();
    let dispatch_out = conv1d_full(&input, &weight, None, &cfg).unwrap();
    assert_close(&dispatch_out, &ref_out, 1e-4, "conv1d_basic");
    // Expected: [6.0, 9.0, 12.0]
    assert_close(&ref_out, &[6.0, 9.0, 12.0], 1e-5, "conv1d_basic_values");
}

#[test]
fn test_conv1d_with_bias() {
    let cfg = Conv1dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: 2,
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
    };
    let input = [1.0, 2.0, 3.0];
    let weight = [1.0, 1.0];
    let bias = [0.5];
    let ref_out = conv1d_full_reference(&input, &weight, Some(&bias), &cfg).unwrap();
    // [1+2+0.5, 2+3+0.5] = [3.5, 5.5]
    assert_close(&ref_out, &[3.5, 5.5], 1e-5, "conv1d_bias_values");
}

#[test]
fn test_conv1d_with_padding() {
    let cfg = Conv1dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
    };
    let input = [1.0, 2.0, 3.0];
    let weight = [1.0, 1.0, 1.0];
    let ref_out = conv1d_full_reference(&input, &weight, None, &cfg).unwrap();
    let dispatch_out = conv1d_full(&input, &weight, None, &cfg).unwrap();
    assert_eq!(ref_out.len(), 3, "conv1d_pad output length");
    assert_close(&dispatch_out, &ref_out, 1e-4, "conv1d_padding");
}

#[test]
fn test_conv1d_stride() {
    let cfg = Conv1dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: 2,
        stride: 2,
        padding: 0,
        dilation: 1,
        groups: 1,
    };
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weight = [1.0, 1.0];
    let ref_out = conv1d_full_reference(&input, &weight, None, &cfg).unwrap();
    // stride=2: [1+2, 3+4, 5+6] = [3.0, 7.0, 11.0]
    assert_close(&ref_out, &[3.0, 7.0, 11.0], 1e-5, "conv1d_stride");
}

#[test]
fn test_conv1d_multi_channel() {
    // 2 input channels, 2 output channels, kernel_size=1
    let cfg = Conv1dConfig {
        in_channels: 2,
        out_channels: 2,
        kernel_size: 1,
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
    };
    // Input: [ic=0: 1, 2, 3; ic=1: 4, 5, 6]
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // Weight: [oc=0,ic=0: 1; oc=0,ic=1: 0; oc=1,ic=0: 0; oc=1,ic=1: 1]
    let weight = [1.0, 0.0, 0.0, 1.0];
    let ref_out = conv1d_full_reference(&input, &weight, None, &cfg).unwrap();
    let dispatch_out = conv1d_full(&input, &weight, None, &cfg).unwrap();
    assert_close(&dispatch_out, &ref_out, 1e-4, "conv1d_multi_channel");
}

// =========================================================================
// 9. EMBEDDING
// =========================================================================

#[test]
fn test_embedding_basic() {
    // vocab_size=4, embed_dim=3
    let weights = [
        1.0, 2.0, 3.0, // word 0
        4.0, 5.0, 6.0, // word 1
        7.0, 8.0, 9.0, // word 2
        10.0, 11.0, 12.0, // word 3
    ];
    let indices = [2u32, 0, 3];
    let out = embedding_lookup(&weights, &indices, 3).unwrap();
    let expected = [7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
    assert_close(&out, &expected, 1e-6, "embedding_basic");
}

#[test]
fn test_embedding_single_token() {
    let weights = [1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
    let indices = [1u32];
    let out = embedding_lookup(&weights, &indices, 2).unwrap();
    assert_close(&out, &[3.0, 4.0], 1e-6, "embedding_single");
}

#[test]
fn test_embedding_repeated_indices() {
    let weights = [10.0, 20.0, 30.0, 40.0]; // vocab=2, dim=2
    let indices = [0u32, 0, 1, 1, 0];
    let out = embedding_lookup(&weights, &indices, 2).unwrap();
    let expected = [10.0, 20.0, 10.0, 20.0, 30.0, 40.0, 30.0, 40.0, 10.0, 20.0];
    assert_close(&out, &expected, 1e-6, "embedding_repeated");
}

#[test]
fn test_embedding_out_of_bounds() {
    let weights = [1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
    let indices = [5u32]; // out of bounds
    let result = embedding_lookup(&weights, &indices, 2);
    assert!(result.is_err(), "embedding should fail for OOB index");
}

#[test]
fn test_embedding_large_dim() {
    let dim = 256;
    let vocab = 10;
    let weights: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.01).collect();
    let indices = [3u32, 7, 0];
    let ref_out = embedding_reference(&weights, &indices, dim).unwrap();
    let out = embedding_lookup(&weights, &indices, dim).unwrap();
    assert_close(&out, &ref_out, 1e-6, "embedding_large_dim");
    // Verify the first element of word 3 is correct.
    let expected_first = (3 * dim) as f32 * 0.01;
    assert!(
        (out[0] - expected_first).abs() < 1e-4,
        "embedding_large_dim[0]: {} vs {}",
        out[0],
        expected_first
    );
}

// =========================================================================
// 10. EDGE CASES
// =========================================================================

#[test]
fn test_single_element_operations() {
    // Add, mul, relu, gelu, silu on single element.
    let a = [3.14];
    let b = [2.72];
    let mut out = vec![0.0f32; 1];

    add_f32(&a, &b, &mut out);
    assert!((out[0] - 5.86).abs() < 1e-4, "single_add: {}", out[0]);

    mul_f32(&a, &b, &mut out);
    assert!(
        (out[0] - 3.14 * 2.72).abs() < 1e-4,
        "single_mul: {}",
        out[0]
    );

    relu_f32(&a, &mut out);
    assert!((out[0] - 3.14).abs() < 1e-6, "single_relu_pos: {}", out[0]);

    let neg = [-3.14];
    relu_f32(&neg, &mut out);
    assert!(out[0].abs() < 1e-6, "single_relu_neg: {}", out[0]);
}

#[test]
fn test_very_large_dimension_reduce() {
    // 4096 elements to stress SIMD tail handling.
    let n = 4096;
    let x: Vec<f32> = (0..n).map(|i| 1.0 / (i as f32 + 1.0)).collect();
    let sum = simd_sum_f32(&x);
    // Harmonic series H_4096 ~ 8.89
    assert!(sum > 8.0 && sum < 10.0, "large_reduce_sum: {sum}");
    let max = simd_max_f32(&x);
    assert!((max - 1.0).abs() < 1e-5, "large_reduce_max: {max}");
}

#[test]
fn test_zero_vector_operations() {
    let zeros = vec![0.0f32; 16];
    let ones = vec![1.0f32; 16];
    let mut out = vec![0.0f32; 16];

    // add zero + ones = ones
    add_f32(&zeros, &ones, &mut out);
    assert_close(&out, &ones, 1e-6, "zero_add");

    // mul zero * ones = zero
    mul_f32(&zeros, &ones, &mut out);
    assert_close(&out, &zeros, 1e-6, "zero_mul");

    // sum of zeros = 0
    let sum = simd_sum_f32(&zeros);
    assert!(sum.abs() < 1e-6, "zero_sum: {sum}");

    // relu of zeros = zeros
    relu_f32(&zeros, &mut out);
    assert_close(&out, &zeros, 1e-6, "zero_relu");
}

#[test]
fn test_all_same_values() {
    let n = 32;
    let val = 3.0f32;
    let x = vec![val; n];

    // Softmax of all-same should be uniform.
    let mut softmax_out = vec![0.0f32; n];
    softmax_f32(&x, &mut softmax_out, n);
    let expected_prob = 1.0 / n as f32;
    for (i, &v) in softmax_out.iter().enumerate() {
        assert!(
            (v - expected_prob).abs() < 1e-2,
            "all_same_softmax[{i}]: {v} vs {expected_prob}"
        );
    }

    // LayerNorm of all-same should be beta (since normalized = 0).
    let gamma = vec![1.0f32; n];
    let beta: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let mut ln_out = vec![0.0f32; n];
    layer_norm_f32(&x, &mut ln_out, &gamma, &beta, n, 1e-5);
    for (i, (&o, &b)) in ln_out.iter().zip(beta.iter()).enumerate() {
        assert!(
            (o - b).abs() < 1e-2,
            "all_same_layernorm[{i}]: {o} vs beta={b}"
        );
    }
}

#[test]
fn test_numerical_stability_large_softmax() {
    // Large positive values should not produce NaN or Inf.
    let n = 16;
    let input: Vec<f32> = (0..n).map(|i| 500.0 + i as f32).collect();
    let mut out = vec![0.0f32; n];
    softmax_f32(&input, &mut out, n);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            v.is_finite() && v >= 0.0,
            "large_softmax_stability[{i}]={v}"
        );
    }
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 5e-2, "large_softmax_sum={sum}");
}

#[test]
fn test_numerical_stability_large_negative_softmax() {
    // Large negative values should produce near-zero probabilities without NaN.
    let n = 8;
    let input = [
        -500.0, -501.0, -502.0, -503.0, -504.0, -505.0, -506.0, -507.0,
    ];
    let mut out = vec![0.0f32; n];
    softmax_f32_scalar(&input, &mut out, n);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite() && v >= 0.0, "neg_softmax[{i}]={v}");
    }
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "neg_softmax_sum={sum}");
}

#[test]
fn test_matmul_scalar_vs_dispatch_consistency() {
    // Ensure scalar and dispatch produce very similar results.
    let m = 5;
    let k = 7;
    let n = 3;
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.37).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.53).cos()).collect();
    let mut scalar_out = vec![0.0f32; m * n];
    let mut dispatch_out = vec![0.0f32; m * n];
    matmul_f32_scalar(&a, &b, m, k, n, &mut scalar_out);
    matmul_f32(&a, &b, m, k, n, &mut dispatch_out);
    assert_close(
        &dispatch_out,
        &scalar_out,
        1e-3,
        "matmul_scalar_vs_dispatch",
    );
}

#[test]
fn test_conv1d_reference_vs_dispatch() {
    let cfg = Conv1dConfig {
        in_channels: 2,
        out_channels: 3,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
    };
    let in_len = 8;
    let input: Vec<f32> = (0..cfg.in_channels * in_len)
        .map(|i| (i as f32 * 0.2).sin())
        .collect();
    let weight: Vec<f32> = (0..cfg.out_channels * cfg.in_channels * cfg.kernel_size)
        .map(|i| (i as f32 * 0.1).cos())
        .collect();
    let ref_out = conv1d_full_reference(&input, &weight, None, &cfg).unwrap();
    let dispatch_out = conv1d_full(&input, &weight, None, &cfg).unwrap();
    assert_close(&dispatch_out, &ref_out, 1e-3, "conv1d_ref_vs_dispatch");
}
