// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD-optimized GroupNorm with explicit tier entry points.

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Naive reference groupnorm (f64 precision for ground truth).
fn naive_groupnorm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    groups: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
) -> Vec<f32> {
    let cpg = channels / groups;
    let group_size = cpg * spatial;
    let mut out = vec![0.0f32; channels * spatial];

    for g in 0..groups {
        let group_start = g * cpg * spatial;
        let group_data = &input[group_start..group_start + group_size];

        let mean: f64 = group_data.iter().map(|&x| f64::from(x)).sum::<f64>() / group_size as f64;
        let var: f64 = group_data
            .iter()
            .map(|&x| {
                let d = f64::from(x) - mean;
                d * d
            })
            .sum::<f64>()
            / group_size as f64;
        let inv_std = 1.0_f64 / (var + f64::from(eps)).sqrt();

        for c_in_group in 0..cpg {
            let c = g * cpg + c_in_group;
            let ch_start = c * spatial;
            for s in 0..spatial {
                let normalized = (f64::from(input[ch_start + s]) - mean) * inv_std;
                out[ch_start + s] = (normalized * f64::from(gamma[c]) + f64::from(beta[c])) as f32;
            }
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
    gamma: &[f32],
    beta: &[f32],
    groups: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * spatial];
    groupnorm_f32_scalar(input, gamma, beta, groups, channels, spatial, eps, &mut out);
    out
}

fn run_dispatch(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    groups: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * spatial];
    groupnorm_f32(input, gamma, beta, groups, channels, spatial, eps, &mut out);
    out
}

fn run_neon(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    groups: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * spatial];
    groupnorm_f32_neon(input, gamma, beta, groups, channels, spatial, eps, &mut out);
    out
}

fn run_avx2(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    groups: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * spatial];
    groupnorm_f32_avx2(input, gamma, beta, groups, channels, spatial, eps, &mut out);
    out
}

// ---------------------------------------------------------------------------
// Basic: groups=channels (equivalent to InstanceNorm)
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_groups_eq_channels() {
    let channels = 4;
    let spatial = 16;
    let groups = 4;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| ((i as f32) * 0.1).sin())
        .collect();
    let gamma = vec![1.0; channels];
    let beta = vec![0.0; channels];

    let result = run_scalar(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    let expected = naive_groupnorm(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-5, "groups_eq_channels");
}

// ---------------------------------------------------------------------------
// Matches naive reference
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_matches_naive() {
    let channels = 8;
    let spatial = 32;
    let groups = 2;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| ((i as f32) * 0.17).sin() * 2.0)
        .collect();
    let gamma: Vec<f32> = (0..channels).map(|c| 0.8 + c as f32 * 0.1).collect();
    let beta: Vec<f32> = (0..channels).map(|c| (c as f32) * 0.05 - 0.2).collect();

    let result = run_scalar(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    let expected = naive_groupnorm(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-5, "scalar_vs_naive");
}

#[test]
fn test_dispatch_matches_naive() {
    let channels = 16;
    let spatial = 64;
    let groups = 4;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| ((i as f32) * 0.07).cos() * 3.0)
        .collect();
    let gamma = vec![1.0; channels];
    let beta = vec![0.0; channels];

    let result = run_dispatch(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    let expected = naive_groupnorm(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-4, "dispatch_vs_naive");
}

#[test]
fn test_neon_matches_naive() {
    let channels = 8;
    let spatial = 32;
    let groups = 2;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| ((i as f32) * 0.11).sin())
        .collect();
    let gamma = vec![1.0; channels];
    let beta = vec![0.0; channels];

    let result = run_neon(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    let expected = naive_groupnorm(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-4, "neon_vs_naive");
}

#[test]
fn test_avx2_matches_naive() {
    let channels = 8;
    let spatial = 32;
    let groups = 2;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| ((i as f32) * 0.11).sin())
        .collect();
    let gamma = vec![1.0; channels];
    let beta = vec![0.0; channels];

    let result = run_avx2(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    let expected = naive_groupnorm(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-4, "avx2_vs_naive");
}

// ---------------------------------------------------------------------------
// Reference matches scalar
// ---------------------------------------------------------------------------

#[test]
fn test_reference_matches_scalar() {
    let channels = 6;
    let spatial = 16;
    let groups = 3;
    let input: Vec<f32> = (0..channels * spatial).map(|i| i as f32 * 0.3).collect();
    let gamma = vec![1.0; channels];
    let beta = vec![0.0; channels];

    let ref_out = groupnorm_reference(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    let scalar_out = run_scalar(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    assert_close(&ref_out, &scalar_out, 1e-6, "reference_vs_scalar");
}

// ---------------------------------------------------------------------------
// groups=1 (equivalent to LayerNorm over C*spatial)
// ---------------------------------------------------------------------------

#[test]
fn test_single_group() {
    let channels = 4;
    let spatial = 32;
    let groups = 1;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| ((i as f32) * 0.2).sin())
        .collect();
    let gamma = vec![1.0; channels];
    let beta = vec![0.0; channels];

    let result = run_dispatch(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    let expected = naive_groupnorm(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-4, "single_group");
}

// ---------------------------------------------------------------------------
// Varied spatial sizes (SIMD tail handling)
// ---------------------------------------------------------------------------

#[test]
fn test_varied_spatial_sizes() {
    let channels = 4;
    let groups = 2;
    let gamma = vec![1.0; channels];
    let beta = vec![0.0; channels];

    for &spatial in &[1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128] {
        let input: Vec<f32> = (0..channels * spatial)
            .map(|i| ((i as f32) * 0.13).sin() * 2.0)
            .collect();
        let result = run_dispatch(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
        let expected = naive_groupnorm(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
        assert_close(
            &result,
            &expected,
            1e-4,
            &format!("varied_spatial_{spatial}"),
        );
    }
}

// ---------------------------------------------------------------------------
// Non-trivial affine parameters
// ---------------------------------------------------------------------------

#[test]
fn test_with_affine() {
    let channels = 8;
    let spatial = 64;
    let groups = 4;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| ((i as f32) * 0.05).cos() * 5.0)
        .collect();
    let gamma: Vec<f32> = (0..channels).map(|c| 0.5 + c as f32 * 0.2).collect();
    let beta: Vec<f32> = (0..channels).map(|c| c as f32 * 0.1 - 0.4).collect();

    let result = run_dispatch(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    let expected = naive_groupnorm(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-4, "with_affine");
}

// ---------------------------------------------------------------------------
// Numerical stability: near-zero inputs
// ---------------------------------------------------------------------------

#[test]
fn test_near_zero_inputs() {
    let channels = 4;
    let spatial = 16;
    let groups = 2;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| (i as f32) * 1e-20)
        .collect();
    let gamma = vec![1.0; channels];
    let beta = vec![0.0; channels];

    let result = run_dispatch(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "near_zero [{i}] = {v} not finite");
    }
}

// ---------------------------------------------------------------------------
// Uniform input within group -> output equals beta
// ---------------------------------------------------------------------------

#[test]
fn test_uniform_within_group() {
    let channels = 4;
    let spatial = 8;
    let groups = 2;
    // All elements within each group are constant.
    let mut input = vec![0.0f32; channels * spatial];
    // Group 0: channels 0,1 all = 5.0
    for c in 0..2 {
        for s in 0..spatial {
            input[c * spatial + s] = 5.0;
        }
    }
    // Group 1: channels 2,3 all = -3.0
    for c in 2..4 {
        for s in 0..spatial {
            input[c * spatial + s] = -3.0;
        }
    }
    let gamma = vec![2.0; channels];
    let beta = vec![1.0, 2.0, 3.0, 4.0];

    let result = run_dispatch(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    // Uniform within group => normalized = 0 => output = beta
    assert_close(
        &result[0..spatial],
        &vec![1.0; spatial],
        1e-4,
        "uniform_g0_c0",
    );
    assert_close(
        &result[spatial..2 * spatial],
        &vec![2.0; spatial],
        1e-4,
        "uniform_g0_c1",
    );
    assert_close(
        &result[2 * spatial..3 * spatial],
        &vec![3.0; spatial],
        1e-4,
        "uniform_g1_c2",
    );
    assert_close(
        &result[3 * spatial..4 * spatial],
        &vec![4.0; spatial],
        1e-4,
        "uniform_g1_c3",
    );
}

// ---------------------------------------------------------------------------
// Different epsilon values
// ---------------------------------------------------------------------------

#[test]
fn test_different_eps() {
    let channels = 4;
    let spatial = 32;
    let groups = 2;
    let input: Vec<f32> = (0..channels * spatial).map(|i| i as f32 * 0.5).collect();
    let gamma = vec![1.0; channels];
    let beta = vec![0.0; channels];

    for &eps in &[1e-12, 1e-8, 1e-5, 1e-3, 0.1, 1.0] {
        let result = run_dispatch(&input, &gamma, &beta, groups, channels, spatial, eps);
        let expected = naive_groupnorm(&input, &gamma, &beta, groups, channels, spatial, eps);
        assert_close(&result, &expected, 1e-4, &format!("eps_{eps}"));
    }
}

// ---------------------------------------------------------------------------
// Typical deep learning configs (ResNet GroupNorm: 32 groups)
// ---------------------------------------------------------------------------

#[test]
fn test_resnet_config() {
    let channels = 64;
    let spatial = 56 * 56; // typical first conv output
    let groups = 32;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| ((i as f32) * 0.001).sin())
        .collect();
    let gamma = vec![1.0; channels];
    let beta = vec![0.0; channels];

    let result = run_dispatch(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    let expected = naive_groupnorm(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-3, "resnet_config");
}
