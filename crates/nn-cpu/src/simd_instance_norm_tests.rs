// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD-optimized InstanceNorm with explicit tier entry points.

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Naive reference instance_norm (f64 precision for ground truth).
fn naive_instance_norm(input: &[f32], channels: usize, spatial: usize, eps: f32) -> Vec<f32> {
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
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff < tol,
            "{label}[{i}]: {x} vs {y}, diff={diff} > tol={tol}"
        );
    }
}

fn run_scalar(input: &[f32], channels: usize, spatial: usize, eps: f32) -> Vec<f32> {
    let mut output = vec![0.0f32; channels * spatial];
    instance_norm_f32_scalar(input, &mut output, channels, spatial, eps);
    output
}

fn run_dispatch(input: &[f32], channels: usize, spatial: usize, eps: f32) -> Vec<f32> {
    let mut output = vec![0.0f32; channels * spatial];
    instance_norm_f32(input, &mut output, channels, spatial, eps);
    output
}

fn run_neon(input: &[f32], channels: usize, spatial: usize, eps: f32) -> Vec<f32> {
    let mut output = vec![0.0f32; channels * spatial];
    instance_norm_f32_neon(input, &mut output, channels, spatial, eps);
    output
}

fn run_avx2(input: &[f32], channels: usize, spatial: usize, eps: f32) -> Vec<f32> {
    let mut output = vec![0.0f32; channels * spatial];
    instance_norm_f32_avx2(input, &mut output, channels, spatial, eps);
    output
}

// ---------------------------------------------------------------------------
// Basic: per-channel zero mean
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_zero_mean_per_channel() {
    let channels = 4;
    let spatial = 64;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| (i as f32 * 0.1).sin() * 5.0)
        .collect();
    let result = run_scalar(&input, channels, spatial, 1e-5);

    for c in 0..channels {
        let start = c * spatial;
        let ch = &result[start..start + spatial];
        let mean: f64 = ch.iter().map(|&x| f64::from(x)).sum::<f64>() / spatial as f64;
        assert!(
            mean.abs() < 1e-5,
            "channel {c} mean should be ~0, got {mean}"
        );
    }
}

#[test]
fn test_dispatch_zero_mean_per_channel() {
    let channels = 4;
    let spatial = 64;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| (i as f32 * 0.1).sin() * 5.0)
        .collect();
    let result = run_dispatch(&input, channels, spatial, 1e-5);

    for c in 0..channels {
        let start = c * spatial;
        let ch = &result[start..start + spatial];
        let mean: f64 = ch.iter().map(|&x| f64::from(x)).sum::<f64>() / spatial as f64;
        assert!(
            mean.abs() < 1e-4,
            "dispatch channel {c} mean should be ~0, got {mean}"
        );
    }
}

// ---------------------------------------------------------------------------
// Per-channel unit variance
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_unit_var_per_channel() {
    let channels = 3;
    let spatial = 128;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| (i as f32 * 0.07).cos() * 3.0)
        .collect();
    let result = run_scalar(&input, channels, spatial, 1e-5);

    for c in 0..channels {
        let start = c * spatial;
        let ch = &result[start..start + spatial];
        let mean: f64 = ch.iter().map(|&x| f64::from(x)).sum::<f64>() / spatial as f64;
        let var: f64 = ch
            .iter()
            .map(|&x| {
                let d = f64::from(x) - mean;
                d * d
            })
            .sum::<f64>()
            / spatial as f64;
        assert!(
            (var - 1.0).abs() < 1e-3,
            "channel {c} var should be ~1, got {var}"
        );
    }
}

// ---------------------------------------------------------------------------
// Matches naive reference
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_matches_naive() {
    let channels = 3;
    let spatial = 16;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| i as f32 * 0.3 - 7.0)
        .collect();
    let result = run_scalar(&input, channels, spatial, 1e-5);
    let expected = naive_instance_norm(&input, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-5, "scalar_vs_naive");
}

#[test]
fn test_dispatch_matches_naive() {
    let channels = 4;
    let spatial = 32;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| ((i as f32) * 0.13).sin() * 4.0)
        .collect();
    let result = run_dispatch(&input, channels, spatial, 1e-5);
    let expected = naive_instance_norm(&input, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-4, "dispatch_vs_naive");
}

#[test]
fn test_neon_matches_naive() {
    let channels = 2;
    let spatial = 16;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| (i as f32) * 0.5 - 8.0)
        .collect();
    let result = run_neon(&input, channels, spatial, 1e-5);
    let expected = naive_instance_norm(&input, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-4, "neon_vs_naive");
}

#[test]
fn test_avx2_matches_naive() {
    let channels = 2;
    let spatial = 16;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| (i as f32) * 0.5 - 8.0)
        .collect();
    let result = run_avx2(&input, channels, spatial, 1e-5);
    let expected = naive_instance_norm(&input, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-4, "avx2_vs_naive");
}

// ---------------------------------------------------------------------------
// Channels are independent
// ---------------------------------------------------------------------------

#[test]
fn test_channels_independent() {
    // Normalization of one channel should not affect another.
    let ch0 = vec![1.0, 2.0, 3.0, 4.0];
    let ch1 = vec![10.0, 20.0, 30.0, 40.0];

    let combined: Vec<f32> = ch0.iter().chain(ch1.iter()).copied().collect();
    let result_combined = run_dispatch(&combined, 2, 4, 1e-5);

    let result_ch0 = run_dispatch(&ch0, 1, 4, 1e-5);
    let result_ch1 = run_dispatch(&ch1, 1, 4, 1e-5);

    for i in 0..4 {
        assert!(
            (result_combined[i] - result_ch0[i]).abs() < 1e-6,
            "ch0[{i}] differs"
        );
        assert!(
            (result_combined[4 + i] - result_ch1[i]).abs() < 1e-6,
            "ch1[{i}] differs"
        );
    }
}

// ---------------------------------------------------------------------------
// Uniform channel -> zero output
// ---------------------------------------------------------------------------

#[test]
fn test_uniform_channel_zero_output() {
    let channels = 3;
    let spatial = 16;
    // Each channel has constant values (different per channel).
    let input: Vec<f32> = (0..channels)
        .flat_map(|c| vec![(c as f32 + 1.0) * 10.0; spatial])
        .collect();
    let result = run_dispatch(&input, channels, spatial, 1e-5);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.abs() < 1e-4,
            "uniform channel: output[{i}] = {v}, expected ~0"
        );
    }
}

// ---------------------------------------------------------------------------
// Single spatial element
// ---------------------------------------------------------------------------

#[test]
fn test_single_spatial() {
    let channels = 5;
    let spatial = 1;
    let input: Vec<f32> = (0..channels).map(|c| (c as f32) * 3.0).collect();
    let result = run_dispatch(&input, channels, spatial, 1e-5);
    // Single spatial: (x - x) / sqrt(0 + eps) = 0
    for (i, &v) in result.iter().enumerate() {
        assert!(v.abs() < 1e-4, "single spatial [{i}] = {v}, expected ~0");
    }
}

// ---------------------------------------------------------------------------
// Varied sizes (SIMD tail handling)
// ---------------------------------------------------------------------------

#[test]
fn test_varied_spatial_sizes() {
    let channels = 3;
    for &spatial in &[
        1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 127, 256,
    ] {
        let input: Vec<f32> = (0..channels * spatial)
            .map(|i| (i as f32 * 0.17).sin() * 4.0)
            .collect();
        let dispatched = run_dispatch(&input, channels, spatial, 1e-5);
        let naive = naive_instance_norm(&input, channels, spatial, 1e-5);
        assert_close(&dispatched, &naive, 1e-4, &format!("spatial_{spatial}"));
    }
}

#[test]
fn test_varied_channel_counts() {
    let spatial = 32;
    for &channels in &[1, 2, 3, 4, 8, 16, 32, 64] {
        let input: Vec<f32> = (0..channels * spatial)
            .map(|i| (i as f32 * 0.11).cos() * 3.0)
            .collect();
        let dispatched = run_dispatch(&input, channels, spatial, 1e-5);
        let naive = naive_instance_norm(&input, channels, spatial, 1e-5);
        assert_close(&dispatched, &naive, 1e-4, &format!("channels_{channels}"));
    }
}

// ---------------------------------------------------------------------------
// Numerical stability: large values
// ---------------------------------------------------------------------------

#[test]
fn test_large_values() {
    let channels = 2;
    let spatial = 16;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| 1e6 + (i as f32) * 100.0)
        .collect();
    let result = run_dispatch(&input, channels, spatial, 1e-5);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "large values [{i}] = {v} not finite");
    }
    let expected = naive_instance_norm(&input, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-3, "large_values");
}

// ---------------------------------------------------------------------------
// Numerical stability: near-zero values
// ---------------------------------------------------------------------------

#[test]
fn test_near_zero_values() {
    let channels = 2;
    let spatial = 16;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| (i as f32) * 1e-20)
        .collect();
    let result = run_dispatch(&input, channels, spatial, 1e-5);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "near-zero [{i}] = {v} not finite");
    }
}

// ---------------------------------------------------------------------------
// Reference matches scalar
// ---------------------------------------------------------------------------

#[test]
fn test_reference_matches_scalar() {
    let channels = 3;
    let spatial = 8;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| (i as f32) * 0.5 - 6.0)
        .collect();
    let ref_out = instance_norm_f32_reference(&input, channels, spatial, 1e-5);
    let scalar_out = run_scalar(&input, channels, spatial, 1e-5);
    assert_close(&ref_out, &scalar_out, 1e-6, "reference_vs_scalar");
}

// ---------------------------------------------------------------------------
// Different epsilon values
// ---------------------------------------------------------------------------

#[test]
fn test_different_eps() {
    let channels = 2;
    let spatial = 16;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| (i as f32) * 0.5 + 0.1)
        .collect();

    for &eps in &[1e-12, 1e-8, 1e-5, 1e-3, 0.1, 1.0] {
        let result = run_dispatch(&input, channels, spatial, eps);
        let expected = naive_instance_norm(&input, channels, spatial, eps);
        assert_close(&result, &expected, 1e-4, &format!("eps_{eps}"));
    }
}

// ---------------------------------------------------------------------------
// Large array: many channels, large spatial
// ---------------------------------------------------------------------------

#[test]
fn test_large_array() {
    let channels = 64;
    let spatial = 256;
    let input: Vec<f32> = (0..channels * spatial)
        .map(|i| ((i as f32) * 0.0037).sin() * 10.0)
        .collect();
    let result = run_dispatch(&input, channels, spatial, 1e-5);
    let expected = naive_instance_norm(&input, channels, spatial, 1e-5);
    assert_close(&result, &expected, 1e-4, "large_array");
}

// ---------------------------------------------------------------------------
// Empty input is a no-op
// ---------------------------------------------------------------------------

#[test]
fn test_empty_spatial() {
    let input: &[f32] = &[];
    let mut output: Vec<f32> = vec![];
    instance_norm_f32(input, &mut output, 0, 0, 1e-5);
}

#[test]
fn test_zero_channels() {
    let input: &[f32] = &[];
    let mut output: Vec<f32> = vec![];
    instance_norm_f32(input, &mut output, 0, 16, 1e-5);
}
