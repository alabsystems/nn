// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for CPU SIMD LayerNorm and RMSNorm.

use super::*;

// ---------------------------------------------------------------------------
// Naive reference implementations (f64 precision for ground truth)
// ---------------------------------------------------------------------------

fn naive_layernorm(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    eps: f32,
    normalized_shape: usize,
) -> Vec<f32> {
    let rows = input.len() / normalized_shape;
    let mut output = vec![0.0f32; input.len()];
    for row in 0..rows {
        let start = row * normalized_shape;
        let row_in = &input[start..start + normalized_shape];
        let mean: f64 = row_in.iter().map(|&x| f64::from(x)).sum::<f64>() / normalized_shape as f64;
        let var: f64 = row_in
            .iter()
            .map(|&x| {
                let d = f64::from(x) - mean;
                d * d
            })
            .sum::<f64>()
            / normalized_shape as f64;
        let inv_std = 1.0 / (var + f64::from(eps)).sqrt();
        for i in 0..normalized_shape {
            let normalized = (f64::from(row_in[i]) - mean) * inv_std;
            output[start + i] =
                (normalized * f64::from(gamma[i]) + beta.map_or(0.0, |b| f64::from(b[i]))) as f32;
        }
    }
    output
}

fn naive_rmsnorm(input: &[f32], gamma: &[f32], eps: f32, normalized_shape: usize) -> Vec<f32> {
    let rows = input.len() / normalized_shape;
    let mut output = vec![0.0f32; input.len()];
    for row in 0..rows {
        let start = row * normalized_shape;
        let row_in = &input[start..start + normalized_shape];
        let sum_sq: f64 = row_in.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        let mean_sq = sum_sq / normalized_shape as f64;
        let inv_rms = 1.0 / (mean_sq + f64::from(eps)).sqrt();
        for i in 0..normalized_shape {
            output[start + i] = (f64::from(row_in[i]) * inv_rms * f64::from(gamma[i])) as f32;
        }
    }
    output
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

// ===========================================================================
// LayerNorm tests
// ===========================================================================

#[test]
fn test_layernorm_scalar_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let result = layernorm_scalar(&input, &gamma, Some(&beta), 1e-5, 4);
    let expected = naive_layernorm(&input, &gamma, Some(&beta), 1e-5, 4);
    assert_close(&result, &expected, 1e-5, "layernorm_scalar_basic");
}

#[test]
fn test_layernorm_scalar_with_affine() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let gamma = vec![0.5, 1.5, 2.0, 0.1];
    let beta = vec![0.1, -0.1, 0.0, 1.0];
    let result = layernorm_scalar(&input, &gamma, Some(&beta), 1e-5, 4);
    let expected = naive_layernorm(&input, &gamma, Some(&beta), 1e-5, 4);
    assert_close(&result, &expected, 1e-5, "layernorm_scalar_affine");
}

#[test]
fn test_layernorm_scalar_no_beta() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let result = layernorm_scalar(&input, &gamma, None, 1e-5, 4);
    let expected = naive_layernorm(&input, &gamma, None, 1e-5, 4);
    assert_close(&result, &expected, 1e-5, "layernorm_scalar_no_beta");
}

#[test]
fn test_layernorm_dispatch_matches_scalar() {
    let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.3 - 5.0).collect();
    let gamma = vec![1.0; 16];
    let beta = vec![0.0; 16];
    let scalar = layernorm_scalar(&input, &gamma, Some(&beta), 1e-5, 16);
    let dispatched = layernorm(&input, &gamma, Some(&beta), 1e-5, 16);
    assert_close(&scalar, &dispatched, 1e-4, "layernorm_dispatch_vs_scalar");
}

#[test]
fn test_layernorm_single_element() {
    // Single-element normalization: (x - x) / sqrt(0 + eps) * gamma + beta
    // = 0 * gamma + beta = beta
    let input = vec![42.0];
    let gamma = vec![2.0];
    let beta = vec![3.0];
    let result = layernorm(&input, &gamma, Some(&beta), 1e-5, 1);
    assert_close(&result, &[3.0], 1e-5, "layernorm_single_element");
}

#[test]
fn test_layernorm_eps_zero() {
    // eps=0 should still work for non-constant inputs (variance > 0)
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let result = layernorm(&input, &gamma, None, 0.0, 4);
    let expected = naive_layernorm(&input, &gamma, None, 0.0, 4);
    assert_close(&result, &expected, 1e-5, "layernorm_eps_zero");
}

#[test]
fn test_layernorm_typical_768() {
    // Typical transformer hidden size
    let n = 768;
    let input: Vec<f32> = (0..n * 2).map(|i| ((i as f32) * 0.01).sin()).collect();
    let gamma: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let beta: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0001).collect();
    let scalar = layernorm_scalar(&input, &gamma, Some(&beta), 1e-5, n);
    let dispatched = layernorm(&input, &gamma, Some(&beta), 1e-5, n);
    assert_close(&scalar, &dispatched, 1e-4, "layernorm_768");
}

#[test]
fn test_layernorm_typical_1024() {
    let n = 1024;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).cos()).collect();
    let gamma = vec![1.0; n];
    let beta = vec![0.0; n];
    let scalar = layernorm_scalar(&input, &gamma, Some(&beta), 1e-5, n);
    let dispatched = layernorm(&input, &gamma, Some(&beta), 1e-5, n);
    assert_close(&scalar, &dispatched, 1e-4, "layernorm_1024");
}

#[test]
fn test_layernorm_typical_4096() {
    let n = 4096;
    let input: Vec<f32> = (0..n * 3)
        .map(|i| ((i as f32) * 0.007).sin() * 2.0)
        .collect();
    let gamma = vec![1.0; n];
    let beta = vec![0.0; n];
    let scalar = layernorm_scalar(&input, &gamma, Some(&beta), 1e-5, n);
    let dispatched = layernorm(&input, &gamma, Some(&beta), 1e-5, n);
    assert_close(&scalar, &dispatched, 1e-4, "layernorm_4096");
}

#[test]
fn test_layernorm_non_simd_aligned_dim() {
    // 13 elements: not aligned to 4 (NEON) or 8 (AVX2)
    let n = 13;
    let input: Vec<f32> = (0..n).map(|i| i as f32 - 6.0).collect();
    let gamma = vec![1.0; n];
    let beta = vec![0.5; n];
    let scalar = layernorm_scalar(&input, &gamma, Some(&beta), 1e-5, n);
    let dispatched = layernorm(&input, &gamma, Some(&beta), 1e-5, n);
    assert_close(&scalar, &dispatched, 1e-4, "layernorm_non_aligned_13");
}

#[test]
fn test_layernorm_output_zero_mean_unit_var() {
    // With gamma=1, beta=0, normalized output should have mean~0 and var~1
    let n = 256;
    let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let gamma = vec![1.0; n];
    let result = layernorm(&input, &gamma, None, 1e-5, n);
    let mean: f64 = result.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
    let var: f64 = result
        .iter()
        .map(|&x| {
            let d = f64::from(x) - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    assert!(mean.abs() < 1e-5, "output mean should be ~0, got {mean}");
    assert!(
        (var - 1.0).abs() < 1e-3,
        "output var should be ~1, got {var}"
    );
}

#[test]
fn test_layernorm_multi_row() {
    // Two rows, different distributions
    let input = vec![
        1.0, 2.0, 3.0, 4.0, // row 0
        10.0, 20.0, 30.0, 40.0, // row 1 (10x scaled)
    ];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let result = layernorm(&input, &gamma, Some(&beta), 1e-5, 4);
    // Both rows should have the same normalized pattern since they differ
    // only by scale and shift.
    for i in 0..4 {
        assert!(
            (result[i] - result[4 + i]).abs() < 1e-4,
            "row0[{i}]={} vs row1[{i}]={}",
            result[i],
            result[4 + i]
        );
    }
}

// ---------------------------------------------------------------------------
// NEW: LayerNorm — output mean approx beta (bias)
// ---------------------------------------------------------------------------

#[test]
fn test_layernorm_output_mean_approx_beta() {
    // With gamma=1 and beta=b, the output mean of each row should be approx
    // mean(beta), since normalized data has mean 0 and var 1, and affine
    // transform gives gamma*normalized + beta => mean(output) ~ mean(beta).
    let n = 128;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 20.0).collect();
    let gamma = vec![1.0; n];
    let beta: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let result = layernorm(&input, &gamma, Some(&beta), 1e-5, n);

    let output_mean: f64 = result.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
    let beta_mean: f64 = beta.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
    assert!(
        (output_mean - beta_mean).abs() < 0.05,
        "output mean {output_mean} should be close to beta mean {beta_mean}"
    );
}

// ---------------------------------------------------------------------------
// NEW: LayerNorm — output std approx gamma (weight)
// ---------------------------------------------------------------------------

#[test]
fn test_layernorm_output_std_approx_gamma_rms() {
    // With beta=0, the output std should be approximately RMS(gamma),
    // since normalized data has unit variance and scaling by gamma[i]
    // gives std(output) ~ sqrt(mean(gamma^2)).
    let n = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin() * 5.0).collect();
    let gamma_val = 2.0_f32;
    let gamma = vec![gamma_val; n];
    let result = layernorm(&input, &gamma, None, 1e-5, n);

    let out_mean: f64 = result.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
    let out_var: f64 = result
        .iter()
        .map(|&x| {
            let d = f64::from(x) - out_mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let out_std = out_var.sqrt();
    // Expected std ~ gamma_val = 2.0
    assert!(
        (out_std - f64::from(gamma_val)).abs() < 0.1,
        "output std {out_std} should be close to gamma {gamma_val}"
    );
}

// ---------------------------------------------------------------------------
// NEW: LayerNorm — known values with weight=1, bias=0
// ---------------------------------------------------------------------------

#[test]
fn test_layernorm_known_values_identity_affine() {
    // input = [1, 2, 3, 4], gamma = [1,1,1,1], beta = [0,0,0,0], eps = 0
    // mean = 2.5, var = 1.25, std = sqrt(1.25)
    // normalized[i] = (x[i] - 2.5) / sqrt(1.25)
    let input = vec![1.0_f32, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let result = layernorm(&input, &gamma, Some(&beta), 0.0, 4);

    let mean = 2.5_f64;
    let var = 1.25_f64;
    let inv_std = 1.0 / var.sqrt();
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| ((f64::from(x) - mean) * inv_std) as f32)
        .collect();

    assert_close(&result, &expected, 1e-5, "layernorm_known_identity_affine");
}

// ---------------------------------------------------------------------------
// NEW: LayerNorm — uniform input gives all zeros (mean subtraction)
// ---------------------------------------------------------------------------

#[test]
fn test_layernorm_uniform_input_all_zeros() {
    // If all elements are the same, (x - mean) = 0 for each element.
    // Output = 0 * gamma + beta = beta (or 0 if no beta).
    let n = 64;
    let input = vec![7.0_f32; n];
    let gamma = vec![1.0; n];
    let result = layernorm(&input, &gamma, None, 1e-5, n);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.abs() < 1e-5,
            "uniform input: output[{i}] = {v}, expected ~0.0"
        );
    }
}

#[test]
fn test_layernorm_uniform_input_equals_beta() {
    let n = 32;
    let input = vec![42.0_f32; n];
    let gamma = vec![3.0; n];
    let beta: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let result = layernorm(&input, &gamma, Some(&beta), 1e-5, n);
    // Output should equal beta since (x - mean) = 0 => 0 * gamma + beta = beta
    assert_close(&result, &beta, 1e-5, "uniform_input_equals_beta");
}

// ---------------------------------------------------------------------------
// NEW: LayerNorm — different epsilon values
// ---------------------------------------------------------------------------

#[test]
fn test_layernorm_different_eps_values() {
    let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    let gamma = vec![1.0; 16];

    for &eps in &[1e-12, 1e-8, 1e-5, 1e-3, 0.1, 1.0] {
        let result = layernorm(&input, &gamma, None, eps, 16);
        let expected = naive_layernorm(&input, &gamma, None, eps, 16);
        assert_close(&result, &expected, 1e-4, &format!("layernorm_eps_{eps}"));
    }
}

// ---------------------------------------------------------------------------
// NEW: LayerNorm — SIMD matches naive reference (dispatch vs naive)
// ---------------------------------------------------------------------------

#[test]
fn test_layernorm_simd_matches_naive_reference() {
    // Use a variety of sizes that exercise SIMD tails and alignment
    for &n in &[
        1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256,
    ] {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin() * 3.0).collect();
        let gamma: Vec<f32> = (0..n).map(|i| 0.8 + (i as f32 * 0.003)).collect();
        let beta: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) - 0.5).collect();
        let dispatched = layernorm(&input, &gamma, Some(&beta), 1e-5, n);
        let naive = naive_layernorm(&input, &gamma, Some(&beta), 1e-5, n);
        assert_close(
            &dispatched,
            &naive,
            1e-4,
            &format!("layernorm_simd_vs_naive_n{n}"),
        );
    }
}

// ---------------------------------------------------------------------------
// NEW: LayerNorm — large arrays (1024+ elements, multi-row)
// ---------------------------------------------------------------------------

#[test]
fn test_layernorm_large_array_2048_multi_row() {
    let n = 512;
    let rows = 4;
    let total = n * rows;
    let input: Vec<f32> = (0..total)
        .map(|i| ((i as f32) * 0.0037).sin() * 10.0)
        .collect();
    let gamma: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.002).collect();
    let beta: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let result = layernorm(&input, &gamma, Some(&beta), 1e-5, n);
    let expected = naive_layernorm(&input, &gamma, Some(&beta), 1e-5, n);
    assert_close(&result, &expected, 1e-4, "layernorm_large_2048");
}

// ---------------------------------------------------------------------------
// NEW: LayerNorm — different normalized_shape sizes
// ---------------------------------------------------------------------------

#[test]
fn test_layernorm_various_normalized_shapes() {
    for &shape in &[2, 3, 5, 7, 10, 16, 48, 64, 100, 256, 512] {
        let rows = 2;
        let total = shape * rows;
        let input: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.11).cos() * 2.0)
            .collect();
        let gamma = vec![1.0; shape];
        let beta = vec![0.0; shape];
        let result = layernorm(&input, &gamma, Some(&beta), 1e-5, shape);
        let expected = naive_layernorm(&input, &gamma, Some(&beta), 1e-5, shape);
        assert_close(
            &result,
            &expected,
            1e-4,
            &format!("layernorm_shape_{shape}"),
        );
    }
}

// ---------------------------------------------------------------------------
// NEW: LayerNorm — numerical stability with large values
// ---------------------------------------------------------------------------

#[test]
fn test_layernorm_numerical_stability_large_values() {
    // Large values should not overflow thanks to Welford's algorithm
    let n = 16;
    let input: Vec<f32> = (0..n).map(|i| 1e6 + (i as f32) * 100.0).collect();
    let gamma = vec![1.0; n];
    let result = layernorm(&input, &gamma, None, 1e-5, n);

    // All outputs should be finite
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "large values: output[{i}] = {v} is not finite"
        );
    }

    // Verify against naive reference
    let expected = naive_layernorm(&input, &gamma, None, 1e-5, n);
    assert_close(&result, &expected, 1e-3, "layernorm_large_values");
}

#[test]
fn test_layernorm_numerical_stability_very_large_values() {
    // Test near the limits of f32 dynamic range
    let n = 8;
    let base = 1e30_f32;
    let input: Vec<f32> = (0..n).map(|i| base + (i as f32) * 1e24).collect();
    let gamma = vec![1.0; n];
    let result = layernorm(&input, &gamma, None, 1e-5, n);

    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "very large values: output[{i}] = {v} is not finite"
        );
    }
}

// ---------------------------------------------------------------------------
// NEW: LayerNorm — numerical stability with very small values
// ---------------------------------------------------------------------------

#[test]
fn test_layernorm_numerical_stability_small_values() {
    // Very small (subnormal-adjacent) values — eps prevents division by zero
    let n = 16;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-20).collect();
    let gamma = vec![1.0; n];
    let result = layernorm(&input, &gamma, None, 1e-5, n);

    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "small values: output[{i}] = {v} is not finite"
        );
    }

    let expected = naive_layernorm(&input, &gamma, None, 1e-5, n);
    assert_close(&result, &expected, 1e-3, "layernorm_small_values");
}

#[test]
fn test_layernorm_numerical_stability_near_zero() {
    // All-zero input: mean=0, var=0, inv_std = 1/sqrt(eps)
    // Output = 0 * gamma + beta = beta (or 0 if no beta)
    let n = 32;
    let input = vec![0.0_f32; n];
    let gamma = vec![1.0; n];
    let result = layernorm(&input, &gamma, None, 1e-5, n);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "zero input: output[{i}] = {v} is not finite");
        assert!(
            v.abs() < 1e-5,
            "zero input: output[{i}] = {v}, expected ~0.0"
        );
    }
}

// ---------------------------------------------------------------------------
// NEW: LayerNorm — scalar-only path matches naive
// ---------------------------------------------------------------------------

#[test]
fn test_layernorm_scalar_matches_naive_many_sizes() {
    for &n in &[1, 2, 3, 4, 5, 8, 13, 16, 32, 64, 100] {
        let input: Vec<f32> = (0..n * 2).map(|i| ((i as f32) * 0.23).sin()).collect();
        let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.02).collect();
        let beta: Vec<f32> = (0..n).map(|i| -0.3 + (i as f32) * 0.01).collect();
        let scalar = layernorm_scalar(&input, &gamma, Some(&beta), 1e-5, n);
        let naive = naive_layernorm(&input, &gamma, Some(&beta), 1e-5, n);
        assert_close(
            &scalar,
            &naive,
            1e-5,
            &format!("layernorm_scalar_vs_naive_n{n}"),
        );
    }
}

// ===========================================================================
// RMSNorm tests
// ===========================================================================

#[test]
fn test_rmsnorm_scalar_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let result = rmsnorm_scalar(&input, &gamma, 1e-5, 4);
    let expected = naive_rmsnorm(&input, &gamma, 1e-5, 4);
    assert_close(&result, &expected, 1e-5, "rmsnorm_scalar_basic");
}

#[test]
fn test_rmsnorm_scalar_with_gamma() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let gamma = vec![0.5, 1.5, 2.0, 0.1];
    let result = rmsnorm_scalar(&input, &gamma, 1e-5, 4);
    let expected = naive_rmsnorm(&input, &gamma, 1e-5, 4);
    assert_close(&result, &expected, 1e-5, "rmsnorm_scalar_gamma");
}

#[test]
fn test_rmsnorm_dispatch_matches_scalar() {
    let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.3 - 5.0).collect();
    let gamma = vec![1.0; 16];
    let scalar = rmsnorm_scalar(&input, &gamma, 1e-5, 16);
    let dispatched = rmsnorm(&input, &gamma, 1e-5, 16);
    assert_close(&scalar, &dispatched, 1e-4, "rmsnorm_dispatch_vs_scalar");
}

#[test]
fn test_rmsnorm_single_element() {
    // RMSNorm of single element x: x / sqrt(x^2 + eps) * gamma
    // For x=3.0, eps=1e-5, gamma=1.0:
    // = 3.0 / sqrt(9.0 + 1e-5) = 3.0 / 3.0000016... ~= 1.0
    let input = vec![3.0];
    let gamma = vec![1.0];
    let result = rmsnorm(&input, &gamma, 1e-5, 1);
    assert!(
        (result[0] - 1.0).abs() < 1e-4,
        "rmsnorm_single: got {}",
        result[0]
    );
}

#[test]
fn test_rmsnorm_eps_zero() {
    // eps=0 should work when input is non-zero
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let result = rmsnorm(&input, &gamma, 0.0, 4);
    let expected = naive_rmsnorm(&input, &gamma, 0.0, 4);
    assert_close(&result, &expected, 1e-5, "rmsnorm_eps_zero");
}

#[test]
fn test_rmsnorm_typical_768() {
    let n = 768;
    let input: Vec<f32> = (0..n * 2).map(|i| ((i as f32) * 0.01).sin()).collect();
    let gamma: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let scalar = rmsnorm_scalar(&input, &gamma, 1e-5, n);
    let dispatched = rmsnorm(&input, &gamma, 1e-5, n);
    assert_close(&scalar, &dispatched, 1e-4, "rmsnorm_768");
}

#[test]
fn test_rmsnorm_typical_1024() {
    let n = 1024;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).cos()).collect();
    let gamma = vec![1.0; n];
    let scalar = rmsnorm_scalar(&input, &gamma, 1e-5, n);
    let dispatched = rmsnorm(&input, &gamma, 1e-5, n);
    assert_close(&scalar, &dispatched, 1e-4, "rmsnorm_1024");
}

#[test]
fn test_rmsnorm_typical_4096() {
    let n = 4096;
    let input: Vec<f32> = (0..n * 3)
        .map(|i| ((i as f32) * 0.007).sin() * 2.0)
        .collect();
    let gamma = vec![1.0; n];
    let scalar = rmsnorm_scalar(&input, &gamma, 1e-5, n);
    let dispatched = rmsnorm(&input, &gamma, 1e-5, n);
    assert_close(&scalar, &dispatched, 1e-4, "rmsnorm_4096");
}

#[test]
fn test_rmsnorm_non_simd_aligned_dim() {
    let n = 13;
    let input: Vec<f32> = (0..n).map(|i| i as f32 - 6.0).collect();
    let gamma = vec![1.0; n];
    let scalar = rmsnorm_scalar(&input, &gamma, 1e-5, n);
    let dispatched = rmsnorm(&input, &gamma, 1e-5, n);
    assert_close(&scalar, &dispatched, 1e-4, "rmsnorm_non_aligned_13");
}

#[test]
fn test_rmsnorm_preserves_direction() {
    // RMSNorm with gamma=1 should preserve the sign of each element
    let input = vec![-3.0, 2.0, -1.0, 4.0];
    let gamma = vec![1.0; 4];
    let result = rmsnorm(&input, &gamma, 1e-5, 4);
    for (i, (&inp, &out)) in input.iter().zip(result.iter()).enumerate() {
        assert_eq!(
            inp.signum(),
            out.signum(),
            "[{i}]: sign changed from {inp} to {out}"
        );
    }
}

#[test]
fn test_rmsnorm_unit_vector() {
    // If input already has RMS=1, output ~= input * gamma
    // Input with RMS=1: values where mean(x^2) = 1
    let n = 4;
    // [1, 1, 1, 1] has mean(x^2) = 1
    let input = vec![1.0; n];
    let gamma = vec![2.0; n];
    let result = rmsnorm(&input, &gamma, 1e-5, n);
    // Expected: 1.0 / sqrt(1.0 + 1e-5) * 2.0 ~= 2.0
    for (i, &v) in result.iter().enumerate() {
        assert!((v - 2.0).abs() < 1e-3, "[{i}]: expected ~2.0, got {v}");
    }
}

#[test]
fn test_rmsnorm_multi_row() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let gamma = vec![1.0; 4];
    let result = rmsnorm(&input, &gamma, 1e-5, 4);
    let expected = naive_rmsnorm(&input, &gamma, 1e-5, 4);
    assert_close(&result, &expected, 1e-5, "rmsnorm_multi_row");
}

// ---------------------------------------------------------------------------
// NEW: RMSNorm — known analytical result for simple input
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_known_analytical_result() {
    // input = [3, 4], gamma = [1, 1], eps = 0
    // sum_sq = 9 + 16 = 25, mean_sq = 12.5, rms = sqrt(12.5)
    // inv_rms = 1/sqrt(12.5)
    // output = [3/sqrt(12.5), 4/sqrt(12.5)]
    let input = vec![3.0_f32, 4.0];
    let gamma = vec![1.0; 2];
    let result = rmsnorm(&input, &gamma, 0.0, 2);

    let rms = (12.5_f64).sqrt();
    let expected = vec![(3.0 / rms) as f32, (4.0 / rms) as f32];
    assert_close(&result, &expected, 1e-5, "rmsnorm_known_analytical");
}

#[test]
fn test_rmsnorm_known_analytical_with_gamma() {
    // input = [1, 2, 3], gamma = [2, 0.5, 1], eps = 0
    // sum_sq = 1 + 4 + 9 = 14, mean_sq = 14/3, rms = sqrt(14/3)
    // output[i] = input[i] / rms * gamma[i]
    let input = vec![1.0_f32, 2.0, 3.0];
    let gamma = vec![2.0, 0.5, 1.0];
    let result = rmsnorm(&input, &gamma, 0.0, 3);

    let rms = (14.0_f64 / 3.0).sqrt();
    let expected = vec![
        (1.0 / rms * 2.0) as f32,
        (2.0 / rms * 0.5) as f32,
        (3.0 / rms * 1.0) as f32,
    ];
    assert_close(&result, &expected, 1e-5, "rmsnorm_known_analytical_gamma");
}

// ---------------------------------------------------------------------------
// NEW: RMSNorm — output RMS approx 1.0 (after normalization with gamma=1)
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_output_rms_approx_one() {
    // After RMSNorm with gamma=1, the output should have RMS ~ 1.0
    // (modulo eps correction).
    let n = 256;
    let input: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.07).sin() * 5.0 + 1.0)
        .collect();
    let gamma = vec![1.0; n];
    let result = rmsnorm(&input, &gamma, 1e-5, n);

    let sum_sq: f64 = result.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let rms = (sum_sq / n as f64).sqrt();
    assert!(
        (rms - 1.0).abs() < 0.01,
        "output RMS should be ~1.0, got {rms}"
    );
}

#[test]
fn test_rmsnorm_output_rms_approx_one_large() {
    let n = 1024;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.03).cos() * 10.0).collect();
    let gamma = vec![1.0; n];
    let result = rmsnorm(&input, &gamma, 1e-5, n);

    let sum_sq: f64 = result.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let rms = (sum_sq / n as f64).sqrt();
    assert!(
        (rms - 1.0).abs() < 0.01,
        "output RMS should be ~1.0 for n=1024, got {rms}"
    );
}

// ---------------------------------------------------------------------------
// NEW: RMSNorm — different epsilon values
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_different_eps_values() {
    let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 + 0.1).collect();
    let gamma = vec![1.0; 16];

    for &eps in &[1e-12, 1e-8, 1e-5, 1e-3, 0.1, 1.0] {
        let result = rmsnorm(&input, &gamma, eps, 16);
        let expected = naive_rmsnorm(&input, &gamma, eps, 16);
        assert_close(&result, &expected, 1e-4, &format!("rmsnorm_eps_{eps}"));
    }
}

// ---------------------------------------------------------------------------
// NEW: RMSNorm — single element (detailed)
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_single_element_various() {
    // For single element x with gamma=1: output = x / sqrt(x^2 + eps)
    // = sign(x) * |x| / sqrt(x^2 + eps) ~ sign(x) for |x| >> eps
    for &x in &[0.1_f32, 1.0, 5.0, 100.0, -3.0, -0.01] {
        let input = vec![x];
        let gamma = vec![1.0];
        let result = rmsnorm(&input, &gamma, 1e-5, 1);
        let expected_f64 = f64::from(x) / (f64::from(x).powi(2) + 1e-5_f64).sqrt();
        let expected = expected_f64 as f32;
        assert!(
            (result[0] - expected).abs() < 1e-5,
            "rmsnorm single x={x}: got {}, expected {expected}",
            result[0]
        );
    }
}

// ---------------------------------------------------------------------------
// NEW: RMSNorm — uniform input
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_uniform_input() {
    // All elements equal c: RMS = |c|, output = c / |c| * gamma = sign(c) * gamma
    let n = 64;
    let c = 5.0_f32;
    let input = vec![c; n];
    let gamma = vec![1.0; n];
    let result = rmsnorm(&input, &gamma, 1e-5, n);

    // Expected: c / sqrt(c^2 + eps) ~ 1.0 (for large c)
    let expected = c / (c * c + 1e-5_f32).sqrt();
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-4,
            "uniform input: output[{i}] = {v}, expected ~{expected}"
        );
    }
}

#[test]
fn test_rmsnorm_uniform_negative_input() {
    let n = 32;
    let c = -3.0_f32;
    let input = vec![c; n];
    let gamma = vec![1.0; n];
    let result = rmsnorm(&input, &gamma, 1e-5, n);

    let expected = c / (c * c + 1e-5_f32).sqrt();
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-4,
            "uniform neg input: output[{i}] = {v}, expected ~{expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// NEW: RMSNorm — SIMD matches naive reference
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_simd_matches_naive_reference() {
    for &n in &[
        1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256,
    ] {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.17).cos() * 4.0).collect();
        let gamma: Vec<f32> = (0..n).map(|i| 0.9 + (i as f32 * 0.002)).collect();
        let dispatched = rmsnorm(&input, &gamma, 1e-5, n);
        let naive = naive_rmsnorm(&input, &gamma, 1e-5, n);
        assert_close(
            &dispatched,
            &naive,
            1e-4,
            &format!("rmsnorm_simd_vs_naive_n{n}"),
        );
    }
}

// ---------------------------------------------------------------------------
// NEW: RMSNorm — large arrays (1024+ elements)
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_large_array_2048_multi_row() {
    let n = 512;
    let rows = 4;
    let total = n * rows;
    let input: Vec<f32> = (0..total)
        .map(|i| ((i as f32) * 0.0041).sin() * 8.0)
        .collect();
    let gamma: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let result = rmsnorm(&input, &gamma, 1e-5, n);
    let expected = naive_rmsnorm(&input, &gamma, 1e-5, n);
    assert_close(&result, &expected, 1e-4, "rmsnorm_large_2048");
}

// ---------------------------------------------------------------------------
// NEW: RMSNorm — different normalized_shape sizes
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_various_normalized_shapes() {
    for &shape in &[2, 3, 5, 7, 10, 16, 48, 64, 100, 256, 512] {
        let rows = 2;
        let total = shape * rows;
        let input: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.09).sin() * 3.0 + 0.5)
            .collect();
        let gamma = vec![1.0; shape];
        let result = rmsnorm(&input, &gamma, 1e-5, shape);
        let expected = naive_rmsnorm(&input, &gamma, 1e-5, shape);
        assert_close(&result, &expected, 1e-4, &format!("rmsnorm_shape_{shape}"));
    }
}

// ---------------------------------------------------------------------------
// NEW: RMSNorm — numerical stability with large values
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_numerical_stability_large_values() {
    let n = 16;
    let input: Vec<f32> = (0..n).map(|i| 1e6 + (i as f32) * 1000.0).collect();
    let gamma = vec![1.0; n];
    let result = rmsnorm(&input, &gamma, 1e-5, n);

    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "large values: output[{i}] = {v} is not finite"
        );
    }

    let expected = naive_rmsnorm(&input, &gamma, 1e-5, n);
    assert_close(&result, &expected, 1e-3, "rmsnorm_large_values");
}

#[test]
fn test_rmsnorm_numerical_stability_very_large_values() {
    let n = 8;
    let base = 1e30_f32;
    let input: Vec<f32> = (0..n).map(|i| base + (i as f32) * 1e24).collect();
    let gamma = vec![1.0; n];
    let result = rmsnorm(&input, &gamma, 1e-5, n);

    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "very large values: output[{i}] = {v} is not finite"
        );
    }
}

// ---------------------------------------------------------------------------
// NEW: RMSNorm — numerical stability with very small values
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_numerical_stability_small_values() {
    let n = 16;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 1e-20).collect();
    let gamma = vec![1.0; n];
    let result = rmsnorm(&input, &gamma, 1e-5, n);

    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "small values: output[{i}] = {v} is not finite"
        );
    }

    let expected = naive_rmsnorm(&input, &gamma, 1e-5, n);
    assert_close(&result, &expected, 1e-3, "rmsnorm_small_values");
}

#[test]
fn test_rmsnorm_numerical_stability_near_zero() {
    // All-zero input with eps > 0: output = 0 / sqrt(0 + eps) * gamma = 0
    let n = 32;
    let input = vec![0.0_f32; n];
    let gamma = vec![1.0; n];
    let result = rmsnorm(&input, &gamma, 1e-5, n);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "zero input: output[{i}] = {v} is not finite");
        assert!(
            v.abs() < 1e-10,
            "zero input: output[{i}] = {v}, expected 0.0"
        );
    }
}

// ---------------------------------------------------------------------------
// NEW: RMSNorm — scalar-only path matches naive
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_scalar_matches_naive_many_sizes() {
    for &n in &[1, 2, 3, 4, 5, 8, 13, 16, 32, 64, 100] {
        let input: Vec<f32> = (0..n * 2)
            .map(|i| ((i as f32) * 0.19).cos() * 2.0)
            .collect();
        let gamma: Vec<f32> = (0..n).map(|i| 0.7 + (i as f32) * 0.015).collect();
        let scalar = rmsnorm_scalar(&input, &gamma, 1e-5, n);
        let naive = naive_rmsnorm(&input, &gamma, 1e-5, n);
        assert_close(
            &scalar,
            &naive,
            1e-5,
            &format!("rmsnorm_scalar_vs_naive_n{n}"),
        );
    }
}

// ===========================================================================
// Cross-cutting tests (both LayerNorm and RMSNorm)
// ===========================================================================

// ---------------------------------------------------------------------------
// NEW: Both — negative inputs
// ---------------------------------------------------------------------------

#[test]
fn test_layernorm_negative_inputs() {
    let n = 16;
    let input: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.5 - 1.0).collect();
    let gamma = vec![1.0; n];
    let result = layernorm(&input, &gamma, None, 1e-5, n);
    let expected = naive_layernorm(&input, &gamma, None, 1e-5, n);
    assert_close(&result, &expected, 1e-4, "layernorm_negative");
}

#[test]
fn test_rmsnorm_negative_inputs() {
    let n = 16;
    let input: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.5 - 1.0).collect();
    let gamma = vec![1.0; n];
    let result = rmsnorm(&input, &gamma, 1e-5, n);
    let expected = naive_rmsnorm(&input, &gamma, 1e-5, n);
    assert_close(&result, &expected, 1e-4, "rmsnorm_negative");
}

// ---------------------------------------------------------------------------
// NEW: Both — mixed positive/negative inputs
// ---------------------------------------------------------------------------

#[test]
fn test_layernorm_mixed_sign_inputs() {
    let input = vec![-5.0, 3.0, -1.0, 7.0, -2.0, 0.0, 4.0, -6.0];
    let gamma = vec![1.0; 8];
    let result = layernorm(&input, &gamma, None, 1e-5, 8);
    let expected = naive_layernorm(&input, &gamma, None, 1e-5, 8);
    assert_close(&result, &expected, 1e-4, "layernorm_mixed_sign");
}

#[test]
fn test_rmsnorm_mixed_sign_inputs() {
    let input = vec![-5.0, 3.0, -1.0, 7.0, -2.0, 0.0, 4.0, -6.0];
    let gamma = vec![1.0; 8];
    let result = rmsnorm(&input, &gamma, 1e-5, 8);
    let expected = naive_rmsnorm(&input, &gamma, 1e-5, 8);
    assert_close(&result, &expected, 1e-4, "rmsnorm_mixed_sign");
}

// ---------------------------------------------------------------------------
// NEW: Both — large eps dominates variance (smoothing effect)
// ---------------------------------------------------------------------------

#[test]
fn test_layernorm_large_eps_smoothing() {
    // With very large eps, the normalization becomes weak (inv_std -> small)
    let n = 8;
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let gamma = vec![1.0; n];
    let result_small_eps = layernorm(&input, &gamma, None, 1e-5, n);
    let result_large_eps = layernorm(&input, &gamma, None, 100.0, n);

    // With large eps, output magnitudes should be smaller
    let mag_small: f32 = result_small_eps.iter().map(|x| x.abs()).sum();
    let mag_large: f32 = result_large_eps.iter().map(|x| x.abs()).sum();
    assert!(
        mag_large < mag_small,
        "large eps should reduce output magnitude: small={mag_small}, large={mag_large}"
    );
}

#[test]
fn test_rmsnorm_large_eps_smoothing() {
    let n = 8;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
    let gamma = vec![1.0; n];
    let result_small_eps = rmsnorm(&input, &gamma, 1e-5, n);
    let result_large_eps = rmsnorm(&input, &gamma, 100.0, n);

    let mag_small: f32 = result_small_eps.iter().map(|x| x.abs()).sum();
    let mag_large: f32 = result_large_eps.iter().map(|x| x.abs()).sum();
    assert!(
        mag_large < mag_small,
        "large eps should reduce output magnitude: small={mag_small}, large={mag_large}"
    );
}

// ---------------------------------------------------------------------------
// NEW: Both — output length matches input length
// ---------------------------------------------------------------------------

#[test]
fn test_layernorm_output_length() {
    for &(total, shape) in &[(16, 4), (100, 10), (1024, 256), (5, 5), (2, 1)] {
        let input = vec![1.0_f32; total];
        let gamma = vec![1.0; shape];
        let result = layernorm(&input, &gamma, None, 1e-5, shape);
        assert_eq!(
            result.len(),
            total,
            "layernorm output length mismatch for total={total}, shape={shape}"
        );
    }
}

#[test]
fn test_rmsnorm_output_length() {
    for &(total, shape) in &[(16, 4), (100, 10), (1024, 256), (5, 5), (2, 1)] {
        let input = vec![1.0_f32; total];
        let gamma = vec![1.0; shape];
        let result = rmsnorm(&input, &gamma, 1e-5, shape);
        assert_eq!(
            result.len(),
            total,
            "rmsnorm output length mismatch for total={total}, shape={shape}"
        );
    }
}
