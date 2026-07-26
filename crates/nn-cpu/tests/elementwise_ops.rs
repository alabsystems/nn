// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD-accelerated elementwise binary and unary ops.
//!
//! Validates that dispatched (NEON/AVX2) paths produce identical results
//! to scalar reference implementations across various array sizes,
//! including non-SIMD-aligned tails.

use nn_cpu::elementwise;

// ============================================================================
// Helpers
// ============================================================================

/// Assert element-wise closeness within tolerance.
fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
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

/// Generate a deterministic input of `n` elements spanning [-10, 10).
fn make_input(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = (i as f32) / (n.max(1) as f32) * 2.0 - 1.0;
            t * 10.0
        })
        .collect()
}

/// Generate a second input offset from the first to avoid trivial cancellations.
fn make_input_b(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = ((i + 7) as f32) / (n.max(1) as f32) * 2.0 - 1.0;
            t * 5.0 + 0.3
        })
        .collect()
}

// ============================================================================
// Binary ops: add
// ============================================================================

#[test]
fn test_add_exact_small() {
    let a = vec![1.0, -2.0, 3.0, -4.0, 5.0];
    let b = vec![10.0, 20.0, -30.0, 40.0, -50.0];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let mut out = vec![0.0f32; a.len()];
    elementwise::add(&a, &b, &mut out);
    assert_close(&out, &expected, 0.0, "add_exact_small");
}

#[test]
fn test_add_simd_vs_scalar_various_sizes() {
    // Sizes chosen to test: below NEON width (4), between NEON/AVX2 (4-8),
    // exact multiples, and non-aligned tails.
    for n in [
        0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 100, 1024,
    ] {
        let a = make_input(n);
        let b = make_input_b(n);
        let mut scalar_out = vec![0.0f32; n];
        let mut simd_out = vec![0.0f32; n];
        elementwise::add_scalar(&a, &b, &mut scalar_out);
        elementwise::add(&a, &b, &mut simd_out);
        assert_close(&scalar_out, &simd_out, 0.0, &format!("add_n={n}"));
    }
}

#[test]
fn test_add_zeros() {
    let a = vec![0.0f32; 17];
    let b = vec![0.0f32; 17];
    let mut out = vec![0.0f32; 17];
    elementwise::add(&a, &b, &mut out);
    assert!(
        out.iter().all(|&v| v == 0.0),
        "add zeros should be all zero"
    );
}

#[test]
fn test_add_identity() {
    let a = make_input(33);
    let zeros = vec![0.0f32; 33];
    let mut out = vec![0.0f32; 33];
    elementwise::add(&a, &zeros, &mut out);
    assert_close(&out, &a, 0.0, "add_identity");
}

// ============================================================================
// Binary ops: mul
// ============================================================================

#[test]
fn test_mul_exact_small() {
    let a = vec![1.0, -2.0, 3.0, -4.0, 5.0];
    let b = vec![10.0, 20.0, -30.0, 40.0, -50.0];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
    let mut out = vec![0.0f32; a.len()];
    elementwise::mul(&a, &b, &mut out);
    assert_close(&out, &expected, 0.0, "mul_exact_small");
}

#[test]
fn test_mul_simd_vs_scalar_various_sizes() {
    for n in [
        0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 100, 1024,
    ] {
        let a = make_input(n);
        let b = make_input_b(n);
        let mut scalar_out = vec![0.0f32; n];
        let mut simd_out = vec![0.0f32; n];
        elementwise::mul_scalar(&a, &b, &mut scalar_out);
        elementwise::mul(&a, &b, &mut simd_out);
        assert_close(&scalar_out, &simd_out, 0.0, &format!("mul_n={n}"));
    }
}

#[test]
fn test_mul_by_one() {
    let a = make_input(33);
    let ones = vec![1.0f32; 33];
    let mut out = vec![0.0f32; 33];
    elementwise::mul(&a, &ones, &mut out);
    assert_close(&out, &a, 0.0, "mul_by_one");
}

#[test]
fn test_mul_by_zero() {
    let a = make_input(33);
    let zeros = vec![0.0f32; 33];
    let mut out = vec![f32::NAN; 33];
    elementwise::mul(&a, &zeros, &mut out);
    assert!(
        out.iter().all(|&v| v == 0.0),
        "mul by zero should be all zero"
    );
}

// ============================================================================
// Unary activations: SIMD vs scalar at various sizes
// ============================================================================

#[test]
fn test_relu_simd_vs_scalar_various_sizes() {
    for n in [0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100, 1024] {
        let input = make_input(n);
        let mut scalar_out = vec![0.0f32; n];
        let mut simd_out = vec![0.0f32; n];
        elementwise::relu_scalar(&input, &mut scalar_out);
        elementwise::relu(&input, &mut simd_out);
        assert_close(&scalar_out, &simd_out, 0.0, &format!("relu_n={n}"));
    }
}

#[test]
fn test_sigmoid_simd_vs_scalar_various_sizes() {
    for n in [0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100, 1024] {
        let input = make_input(n);
        let mut scalar_out = vec![0.0f32; n];
        let mut simd_out = vec![0.0f32; n];
        elementwise::sigmoid_scalar(&input, &mut scalar_out);
        elementwise::sigmoid(&input, &mut simd_out);
        assert_close(&scalar_out, &simd_out, 1e-5, &format!("sigmoid_n={n}"));
    }
}

#[test]
fn test_silu_simd_vs_scalar_various_sizes() {
    for n in [0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100, 1024] {
        let input = make_input(n);
        let mut scalar_out = vec![0.0f32; n];
        let mut simd_out = vec![0.0f32; n];
        elementwise::silu_scalar(&input, &mut scalar_out);
        elementwise::silu(&input, &mut simd_out);
        assert_close(&scalar_out, &simd_out, 1e-5, &format!("silu_n={n}"));
    }
}

#[test]
fn test_gelu_simd_vs_scalar_various_sizes() {
    for n in [0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100, 1024] {
        let input = make_input(n);
        let mut scalar_out = vec![0.0f32; n];
        let mut simd_out = vec![0.0f32; n];
        elementwise::gelu_scalar(&input, &mut scalar_out);
        elementwise::gelu(&input, &mut simd_out);
        assert_close(&scalar_out, &simd_out, 1e-5, &format!("gelu_n={n}"));
    }
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_add_large_values() {
    let a = vec![1e30_f32; 9];
    let b = vec![1e30_f32; 9];
    let mut out = vec![0.0f32; 9];
    elementwise::add(&a, &b, &mut out);
    for &v in &out {
        assert!((v - 2e30).abs() < 1e24, "large add: got {v}");
    }
}

#[test]
fn test_mul_negative_values() {
    let a = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let b = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let expected: Vec<f32> = vec![9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0];
    let mut out = vec![0.0f32; 7];
    elementwise::mul(&a, &b, &mut out);
    assert_close(&out, &expected, 0.0, "mul_negative");
}

#[test]
fn test_relu_negative_all_zero() {
    let input = vec![-10.0, -5.0, -1.0, -0.001];
    let mut out = vec![1.0f32; 4];
    elementwise::relu(&input, &mut out);
    assert!(
        out.iter().all(|&v| v == 0.0),
        "relu of negatives should be 0"
    );
}

#[test]
fn test_sigmoid_bounds() {
    // sigmoid output must be in (0, 1) for all finite inputs.
    let input = make_input(100);
    let mut out = vec![0.0f32; 100];
    elementwise::sigmoid(&input, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            v > 0.0 && v < 1.0,
            "sigmoid[{i}]={v} out of (0,1) for input={}",
            input[i]
        );
    }
}

#[test]
fn test_silu_at_zero() {
    let input = vec![0.0f32; 5];
    let mut out = vec![1.0f32; 5];
    elementwise::silu(&input, &mut out);
    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    for &v in &out {
        assert!((v - 0.0).abs() < 1e-7, "silu(0) should be 0, got {v}");
    }
}

#[test]
fn test_gelu_at_zero() {
    let input = vec![0.0f32; 5];
    let mut out = vec![1.0f32; 5];
    elementwise::gelu(&input, &mut out);
    // gelu(0) = 0
    for &v in &out {
        assert!((v - 0.0).abs() < 1e-7, "gelu(0) should be 0, got {v}");
    }
}
