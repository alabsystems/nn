// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD-optimized LayerNorm with explicit tier entry points.

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Naive reference layernorm (f64 precision for ground truth).
fn naive_layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], n: usize, eps: f32) -> Vec<f32> {
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
        .map(|(i, &x)| {
            let normalized = (f64::from(x) - mean) * inv_std;
            (normalized * f64::from(gamma[i]) + f64::from(beta[i])) as f32
        })
        .collect()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff < tol,
            "{label}[{i}]: {x} vs {y}, diff={diff} > tol={tol}"
        );
    }
}

fn run_scalar(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0f32; n];
    layer_norm_f32_scalar(input, &mut output, gamma, beta, n, eps);
    output
}

fn run_dispatch(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0f32; n];
    layer_norm_f32(input, &mut output, gamma, beta, n, eps);
    output
}

fn run_neon(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0f32; n];
    layer_norm_f32_neon(input, &mut output, gamma, beta, n, eps);
    output
}

fn run_avx2(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0f32; n];
    layer_norm_f32_avx2(input, &mut output, gamma, beta, n, eps);
    output
}

// ---------------------------------------------------------------------------
// Basic: identity affine (gamma=1, beta=0) -> zero mean, unit variance
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_zero_mean_unit_var() {
    let n = 256;
    let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let gamma = vec![1.0; n];
    let beta = vec![0.0; n];
    let result = run_scalar(&input, &gamma, &beta, 1e-5);

    let mean: f64 = result.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
    let var: f64 = result
        .iter()
        .map(|&x| {
            let d = f64::from(x) - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;

    assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
    assert!((var - 1.0).abs() < 1e-3, "var should be ~1, got {var}");
}

#[test]
fn test_dispatch_zero_mean_unit_var() {
    let n = 256;
    let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let gamma = vec![1.0; n];
    let beta = vec![0.0; n];
    let result = run_dispatch(&input, &gamma, &beta, 1e-5);

    let mean: f64 = result.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
    let var: f64 = result
        .iter()
        .map(|&x| {
            let d = f64::from(x) - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;

    assert!(mean.abs() < 1e-4, "dispatch mean should be ~0, got {mean}");
    assert!(
        (var - 1.0).abs() < 1e-2,
        "dispatch var should be ~1, got {var}"
    );
}

// ---------------------------------------------------------------------------
// Matches naive reference
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_matches_naive() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let result = run_scalar(&input, &gamma, &beta, 1e-5);
    let expected = naive_layer_norm(&input, &gamma, &beta, 4, 1e-5);
    assert_close(&result, &expected, 1e-5, "scalar_vs_naive");
}

#[test]
fn test_dispatch_matches_naive() {
    let n = 32;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 5.0).collect();
    let gamma = vec![1.0; n];
    let beta = vec![0.0; n];
    let result = run_dispatch(&input, &gamma, &beta, 1e-5);
    let expected = naive_layer_norm(&input, &gamma, &beta, n, 1e-5);
    assert_close(&result, &expected, 1e-4, "dispatch_vs_naive");
}

#[test]
fn test_neon_matches_naive() {
    let n = 16;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 4.0).collect();
    let gamma = vec![1.0; n];
    let beta = vec![0.0; n];
    let result = run_neon(&input, &gamma, &beta, 1e-5);
    let expected = naive_layer_norm(&input, &gamma, &beta, n, 1e-5);
    assert_close(&result, &expected, 1e-4, "neon_vs_naive");
}

#[test]
fn test_avx2_matches_naive() {
    let n = 16;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 4.0).collect();
    let gamma = vec![1.0; n];
    let beta = vec![0.0; n];
    let result = run_avx2(&input, &gamma, &beta, 1e-5);
    let expected = naive_layer_norm(&input, &gamma, &beta, n, 1e-5);
    assert_close(&result, &expected, 1e-4, "avx2_vs_naive");
}

// ---------------------------------------------------------------------------
// With affine transform (non-trivial gamma and beta)
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_with_affine() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![0.5, 1.5, 2.0, 0.1];
    let beta = vec![0.1, -0.1, 0.0, 1.0];
    let result = run_scalar(&input, &gamma, &beta, 1e-5);
    let expected = naive_layer_norm(&input, &gamma, &beta, 4, 1e-5);
    assert_close(&result, &expected, 1e-5, "scalar_affine");
}

#[test]
fn test_dispatch_with_affine() {
    let n = 64;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.13).sin() * 3.0).collect();
    let gamma: Vec<f32> = (0..n).map(|i| 0.8 + (i as f32 * 0.003)).collect();
    let beta: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) - 0.5).collect();
    let result = run_dispatch(&input, &gamma, &beta, 1e-5);
    let expected = naive_layer_norm(&input, &gamma, &beta, n, 1e-5);
    assert_close(&result, &expected, 1e-4, "dispatch_affine");
}

// ---------------------------------------------------------------------------
// Uniform input -> output equals beta
// ---------------------------------------------------------------------------

#[test]
fn test_uniform_input_equals_beta() {
    let n = 32;
    let input = vec![42.0_f32; n];
    let gamma = vec![3.0; n];
    let beta: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let result = run_dispatch(&input, &gamma, &beta, 1e-5);
    assert_close(&result, &beta, 1e-4, "uniform_equals_beta");
}

// ---------------------------------------------------------------------------
// Single element -> beta (normalized = 0)
// ---------------------------------------------------------------------------

#[test]
fn test_single_element() {
    let input = vec![42.0];
    let gamma = vec![2.0];
    let beta = vec![3.0];
    let result = run_dispatch(&input, &gamma, &beta, 1e-5);
    assert_close(&result, &[3.0], 1e-5, "single_element");
}

// ---------------------------------------------------------------------------
// Varied sizes (SIMD tail handling)
// ---------------------------------------------------------------------------

#[test]
fn test_varied_sizes() {
    for &n in &[
        1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 127, 256, 1024,
    ] {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin() * 3.0).collect();
        let gamma: Vec<f32> = (0..n).map(|i| 0.8 + (i as f32 * 0.003)).collect();
        let beta: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) - 0.5).collect();
        let dispatched = run_dispatch(&input, &gamma, &beta, 1e-5);
        let naive = naive_layer_norm(&input, &gamma, &beta, n, 1e-5);
        assert_close(&dispatched, &naive, 1e-4, &format!("varied_n{n}"));
    }
}

// ---------------------------------------------------------------------------
// Numerical stability: large values
// ---------------------------------------------------------------------------

#[test]
fn test_large_values() {
    let n = 16;
    let input: Vec<f32> = (0..n).map(|i| 1e6 + (i as f32) * 100.0).collect();
    let gamma = vec![1.0; n];
    let beta = vec![0.0; n];
    let result = run_dispatch(&input, &gamma, &beta, 1e-5);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "large [{i}] = {v} not finite");
    }
    let expected = naive_layer_norm(&input, &gamma, &beta, n, 1e-5);
    assert_close(&result, &expected, 1e-3, "large_values");
}

// ---------------------------------------------------------------------------
// Numerical stability: near-zero values
// ---------------------------------------------------------------------------

#[test]
fn test_near_zero_values() {
    let n = 16;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-20).collect();
    let gamma = vec![1.0; n];
    let beta = vec![0.0; n];
    let result = run_dispatch(&input, &gamma, &beta, 1e-5);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "near-zero [{i}] = {v} not finite");
    }
}

// ---------------------------------------------------------------------------
// All-zero input -> zero output (with beta=0)
// ---------------------------------------------------------------------------

#[test]
fn test_zero_input() {
    let n = 32;
    let input = vec![0.0_f32; n];
    let gamma = vec![1.0; n];
    let beta = vec![0.0; n];
    let result = run_dispatch(&input, &gamma, &beta, 1e-5);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.abs() < 1e-5, "zero input [{i}] = {v}, expected ~0");
    }
}

// ---------------------------------------------------------------------------
// Reference matches scalar
// ---------------------------------------------------------------------------

#[test]
fn test_reference_matches_scalar() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let gamma = vec![1.0; 5];
    let beta = vec![0.0; 5];
    let ref_out = layer_norm_f32_reference(&input, &gamma, &beta, 5, 1e-5);
    let scalar_out = run_scalar(&input, &gamma, &beta, 1e-5);
    assert_close(&ref_out, &scalar_out, 1e-6, "reference_vs_scalar");
}

// ---------------------------------------------------------------------------
// Known values: input = [1,2,3,4], gamma=1, beta=0, eps=0
// ---------------------------------------------------------------------------

#[test]
fn test_known_values() {
    let input = vec![1.0_f32, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let result = run_dispatch(&input, &gamma, &beta, 0.0);

    let mean = 2.5_f64;
    let var = 1.25_f64;
    let inv_std = 1.0 / var.sqrt();
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| ((f64::from(x) - mean) * inv_std) as f32)
        .collect();
    assert_close(&result, &expected, 1e-4, "known_values");
}

// ---------------------------------------------------------------------------
// Different epsilon values
// ---------------------------------------------------------------------------

#[test]
fn test_different_eps() {
    let n = 16;
    let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let gamma = vec![1.0; n];
    let beta = vec![0.0; n];

    for &eps in &[1e-12, 1e-8, 1e-5, 1e-3, 0.1, 1.0] {
        let result = run_dispatch(&input, &gamma, &beta, eps);
        let expected = naive_layer_norm(&input, &gamma, &beta, n, eps);
        assert_close(&result, &expected, 1e-4, &format!("eps_{eps}"));
    }
}

// ---------------------------------------------------------------------------
// Typical transformer hidden sizes
// ---------------------------------------------------------------------------

#[test]
fn test_hidden_768() {
    let n = 768;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin()).collect();
    let gamma: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let beta: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0001).collect();
    let result = run_dispatch(&input, &gamma, &beta, 1e-5);
    let expected = naive_layer_norm(&input, &gamma, &beta, n, 1e-5);
    assert_close(&result, &expected, 1e-4, "hidden_768");
}

#[test]
fn test_hidden_1024() {
    let n = 1024;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).cos()).collect();
    let gamma = vec![1.0; n];
    let beta = vec![0.0; n];
    let result = run_dispatch(&input, &gamma, &beta, 1e-5);
    let expected = naive_layer_norm(&input, &gamma, &beta, n, 1e-5);
    assert_close(&result, &expected, 1e-4, "hidden_1024");
}
