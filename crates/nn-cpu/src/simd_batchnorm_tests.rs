// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD-optimized BatchNorm with explicit tier entry points.

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Naive reference batchnorm (f64 precision for ground truth).
fn naive_batchnorm(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * spatial];
    for c in 0..channels {
        let inv_std = 1.0_f64 / (f64::from(var[c]) + f64::from(eps)).sqrt();
        let scale = f64::from(gamma[c]) * inv_std;
        let shift = f64::from(beta[c]) - f64::from(gamma[c]) * f64::from(mean[c]) * inv_std;
        let start = c * spatial;
        for s in 0..spatial {
            out[start + s] = (f64::from(input[start + s]) * scale + shift) as f32;
        }
    }
    out
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

fn run_scalar(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * spatial];
    batchnorm_f32_scalar(
        input, mean, var, gamma, beta, channels, spatial, eps, &mut out,
    );
    out
}

fn run_dispatch(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * spatial];
    batchnorm_f32(
        input, mean, var, gamma, beta, channels, spatial, eps, &mut out,
    );
    out
}

fn run_neon(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * spatial];
    batchnorm_f32_neon(
        input, mean, var, gamma, beta, channels, spatial, eps, &mut out,
    );
    out
}

fn run_avx2(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * spatial];
    batchnorm_f32_avx2(
        input, mean, var, gamma, beta, channels, spatial, eps, &mut out,
    );
    out
}

// ---------------------------------------------------------------------------
// Basic: identity transform (gamma=1, beta=0, mean=0, var=1)
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_identity_transform() {
    let channels = 4;
    let spatial = 64;
    let input: Vec<f32> = (0..channels * spatial).map(|i| (i as f32) * 0.1).collect();
    let mean = vec![0.0; channels];
    let var = vec![1.0; channels];
    let gamma = vec![1.0; channels];
    let beta = vec![0.0; channels];

    let result = run_scalar(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    // With mean=0, var=1, gamma=1, beta=0: output ~= input (up to eps effect on inv_std)
    assert_close(&result, &input, 2e-4, "identity");
}

// ---------------------------------------------------------------------------
// Matches naive reference
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_matches_naive() {
    let channels = 3;
    let spatial = 32;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| ((i as f32) * 0.17).sin() * 2.0)
        .collect();
    let mean = vec![0.5, -0.3, 1.0];
    let var = vec![2.0, 0.5, 1.5];
    let gamma = vec![1.5, 0.8, 2.0];
    let beta = vec![0.1, -0.2, 0.0];

    let result = run_scalar(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    let expected = naive_batchnorm(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-5, "scalar_vs_naive");
}

#[test]
fn test_dispatch_matches_naive() {
    let channels = 4;
    let spatial = 128;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| ((i as f32) * 0.07).cos() * 3.0)
        .collect();
    let mean = vec![0.1, -0.5, 0.3, 1.2];
    let var = vec![1.0, 2.5, 0.8, 3.0];
    let gamma = vec![1.0, 1.5, 0.5, 2.0];
    let beta = vec![0.0, 0.1, -0.3, 0.5];

    let result = run_dispatch(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    let expected = naive_batchnorm(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-4, "dispatch_vs_naive");
}

#[test]
fn test_neon_matches_naive() {
    let channels = 3;
    let spatial = 64;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| ((i as f32) * 0.11).sin())
        .collect();
    let mean = vec![0.2, -0.1, 0.5];
    let var = vec![1.0, 0.5, 2.0];
    let gamma = vec![1.0, 1.0, 1.0];
    let beta = vec![0.0, 0.0, 0.0];

    let result = run_neon(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    let expected = naive_batchnorm(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-4, "neon_vs_naive");
}

#[test]
fn test_avx2_matches_naive() {
    let channels = 3;
    let spatial = 64;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| ((i as f32) * 0.11).sin())
        .collect();
    let mean = vec![0.2, -0.1, 0.5];
    let var = vec![1.0, 0.5, 2.0];
    let gamma = vec![1.0, 1.0, 1.0];
    let beta = vec![0.0, 0.0, 0.0];

    let result = run_avx2(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    let expected = naive_batchnorm(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-4, "avx2_vs_naive");
}

// ---------------------------------------------------------------------------
// Reference matches scalar
// ---------------------------------------------------------------------------

#[test]
fn test_reference_matches_scalar() {
    let channels = 2;
    let spatial = 16;
    let input: Vec<f32> = (0..channels * spatial).map(|i| i as f32 * 0.3).collect();
    let mean = vec![1.0, -0.5];
    let var = vec![2.0, 1.0];
    let gamma = vec![1.5, 0.8];
    let beta = vec![0.1, -0.2];

    let ref_out = batchnorm_reference(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    let scalar_out = run_scalar(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    assert_close(&ref_out, &scalar_out, 1e-6, "reference_vs_scalar");
}

// ---------------------------------------------------------------------------
// Varied sizes (SIMD tail handling)
// ---------------------------------------------------------------------------

#[test]
fn test_varied_spatial_sizes() {
    let channels = 3;
    let mean = vec![0.5, -0.3, 1.0];
    let var = vec![2.0, 0.5, 1.5];
    let gamma = vec![1.5, 0.8, 2.0];
    let beta = vec![0.1, -0.2, 0.3];

    for &spatial in &[1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128, 256] {
        let input: Vec<f32> = (0..channels * spatial)
            .map(|i| ((i as f32) * 0.13).sin() * 2.0)
            .collect();
        let result = run_dispatch(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
        let expected = naive_batchnorm(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
        assert_close(
            &result,
            &expected,
            1e-4,
            &format!("varied_spatial_{spatial}"),
        );
    }
}

// ---------------------------------------------------------------------------
// Uniform input -> constant per channel
// ---------------------------------------------------------------------------

#[test]
fn test_uniform_input_per_channel() {
    let channels = 3;
    let spatial = 32;
    // All spatial elements for a channel are the same value.
    let mut input = vec![0.0f32; channels * spatial];
    for c in 0..channels {
        let val = (c as f32 + 1.0) * 2.0;
        for s in 0..spatial {
            input[c * spatial + s] = val;
        }
    }
    let mean = vec![2.0, 4.0, 6.0];
    let var = vec![1.0, 1.0, 1.0];
    let gamma = vec![1.0, 1.0, 1.0];
    let beta = vec![0.0, 0.0, 0.0];

    let result = run_dispatch(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    let expected = naive_batchnorm(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-5, "uniform_per_channel");
}

// ---------------------------------------------------------------------------
// Large variance (should not lose precision)
// ---------------------------------------------------------------------------

#[test]
fn test_large_variance() {
    let channels = 2;
    let spatial = 64;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| (i as f32) * 100.0)
        .collect();
    let mean = vec![0.0, 0.0];
    let var = vec![1e6, 1e6];
    let gamma = vec![1.0, 1.0];
    let beta = vec![0.0, 0.0];

    let result = run_dispatch(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "large_var [{i}] = {v} not finite");
    }
    let expected = naive_batchnorm(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-3, "large_variance");
}

// ---------------------------------------------------------------------------
// Zero variance (eps prevents division by zero)
// ---------------------------------------------------------------------------

#[test]
fn test_zero_variance() {
    let channels = 2;
    let spatial = 16;
    let input: Vec<f32> = (0..channels * spatial).map(|i| i as f32).collect();
    let mean = vec![0.0, 0.0];
    let var = vec![0.0, 0.0]; // zero variance
    let gamma = vec![1.0, 1.0];
    let beta = vec![0.0, 0.0];

    let result = run_dispatch(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "zero_var [{i}] = {v} not finite");
    }
    let expected = naive_batchnorm(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    // Large inv_std from zero variance + integer inputs = outputs ~10^4 where
    // f32 vs f64 precision naturally diverges. Use relative tolerance check.
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        let abs_diff = (r - e).abs();
        let rel_tol = e.abs() * 1e-6 + 1e-4;
        assert!(
            abs_diff < rel_tol,
            "zero_variance[{i}]: {r} vs {e}, diff={abs_diff} > rel_tol={rel_tol}"
        );
    }
}

// ---------------------------------------------------------------------------
// Different epsilon values
// ---------------------------------------------------------------------------

#[test]
fn test_different_eps() {
    let channels = 2;
    let spatial = 32;
    let input: Vec<f32> = (0..channels * spatial).map(|i| i as f32 * 0.5).collect();
    let mean = vec![1.0, -0.5];
    let var = vec![2.0, 0.1];
    let gamma = vec![1.0, 1.0];
    let beta = vec![0.0, 0.0];

    for &eps in &[1e-12, 1e-8, 1e-5, 1e-3, 0.1, 1.0] {
        let result = run_dispatch(&input, &mean, &var, &gamma, &beta, channels, spatial, eps);
        let expected = naive_batchnorm(&input, &mean, &var, &gamma, &beta, channels, spatial, eps);
        assert_close(&result, &expected, 1e-4, &format!("eps_{eps}"));
    }
}

// ---------------------------------------------------------------------------
// Single channel, single spatial element
// ---------------------------------------------------------------------------

#[test]
fn test_single_element() {
    let input = vec![5.0f32];
    let mean = vec![2.0];
    let var = vec![4.0];
    let gamma = vec![2.0];
    let beta = vec![1.0];

    let result = run_dispatch(&input, &mean, &var, &gamma, &beta, 1, 1, 1e-5);
    // (5 - 2) / sqrt(4 + 1e-5) * 2 + 1 = 3/2 * 2 + 1 = 4.0
    let expected = naive_batchnorm(&input, &mean, &var, &gamma, &beta, 1, 1, 1e-5);
    assert_close(&result, &expected, 1e-5, "single_element");
    assert!(
        (result[0] - 4.0).abs() < 1e-4,
        "expected ~4.0, got {}",
        result[0]
    );
}

// ---------------------------------------------------------------------------
// SIMD threshold behavior
// ---------------------------------------------------------------------------

#[test]
fn test_below_simd_threshold() {
    // Below BATCHNORM_SIMD_THRESHOLD, NEON/AVX2 should delegate to scalar.
    let channels = 2;
    let spatial = BATCHNORM_SIMD_THRESHOLD - 1;
    let input: Vec<f32> = (0..channels * spatial).map(|i| i as f32 * 0.1).collect();
    let mean = vec![0.5, 1.0];
    let var = vec![1.0, 2.0];
    let gamma = vec![1.0, 1.5];
    let beta = vec![0.0, 0.1];

    let scalar_out = run_scalar(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    let neon_out = run_neon(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    let avx2_out = run_avx2(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);

    assert_close(&neon_out, &scalar_out, 1e-6, "neon_below_threshold");
    assert_close(&avx2_out, &scalar_out, 1e-6, "avx2_below_threshold");
}

// ---------------------------------------------------------------------------
// Many channels (typical ResNet: 64, 128, 256, 512)
// ---------------------------------------------------------------------------

#[test]
fn test_many_channels() {
    for &channels in &[64, 128, 256] {
        let spatial = 49; // 7x7 spatial
        let input: Vec<f32> = (0..channels * spatial)
            .map(|i| ((i as f32) * 0.03).sin())
            .collect();
        let mean: Vec<f32> = (0..channels).map(|c| (c as f32) * 0.01).collect();
        let var: Vec<f32> = (0..channels).map(|c| 1.0 + (c as f32) * 0.005).collect();
        let gamma = vec![1.0; channels];
        let beta = vec![0.0; channels];

        let result = run_dispatch(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
        let expected = naive_batchnorm(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
        assert_close(
            &result,
            &expected,
            1e-4,
            &format!("many_channels_{channels}"),
        );
    }
}
