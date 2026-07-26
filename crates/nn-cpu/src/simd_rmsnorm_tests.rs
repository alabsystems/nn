// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD-optimized RMS Normalization.

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff < tol,
            "{label}[{i}]: {x} vs {y}, diff={diff} > tol={tol}"
        );
    }
}

// ---------------------------------------------------------------------------
// test_rmsnorm_reference_unit_weight: weight=1 should normalize
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_reference_unit_weight() {
    let hidden_size = 8;
    let input: Vec<f32> = (1..=hidden_size as i32).map(|i| i as f32).collect();
    let weight = vec![1.0f32; hidden_size];
    let eps = 1e-5;

    let result = rmsnorm_reference(&input, &weight, hidden_size, eps);

    // Verify: rms = sqrt(mean(x^2) + eps)
    let sum_sq: f64 = input.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let rms = ((sum_sq / hidden_size as f64) + f64::from(eps)).sqrt();

    // Each output = x / rms (weight=1)
    for (i, (&x, &out)) in input.iter().zip(result.iter()).enumerate() {
        let expected = (f64::from(x) / rms) as f32;
        let diff = (out - expected).abs();
        assert!(
            diff < 1e-5,
            "unit_weight[{i}]: {out} vs {expected}, diff={diff}"
        );
    }

    // Output should be finite and have a reasonable range
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "unit_weight[{i}] = {v} not finite");
    }
}

// ---------------------------------------------------------------------------
// test_rmsnorm_reference_scale: weight=2 should double normalized output
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_reference_scale() {
    let hidden_size = 8;
    let input: Vec<f32> = (1..=hidden_size as i32).map(|i| i as f32).collect();
    let weight_1 = vec![1.0f32; hidden_size];
    let weight_2 = vec![2.0f32; hidden_size];
    let eps = 1e-5;

    let result_1 = rmsnorm_reference(&input, &weight_1, hidden_size, eps);
    let result_2 = rmsnorm_reference(&input, &weight_2, hidden_size, eps);

    // result_2 should be exactly 2x result_1
    for (i, (&v1, &v2)) in result_1.iter().zip(result_2.iter()).enumerate() {
        let expected = v1 * 2.0;
        let diff = (v2 - expected).abs();
        assert!(
            diff < 1e-5,
            "scale[{i}]: {v2} vs 2*{v1}={expected}, diff={diff}"
        );
    }
}

// ---------------------------------------------------------------------------
// test_rmsnorm_simd_matches_reference: compare SIMD vs reference
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_simd_matches_reference() {
    let hidden_size = 256;
    let eps = 1e-5;

    // Generate pseudo-random data
    let mut seed: u64 = 42;
    let mut next_f32 = || -> f32 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    let input: Vec<f32> = (0..hidden_size).map(|_| next_f32()).collect();
    let weight: Vec<f32> = (0..hidden_size).map(|_| 0.5 + next_f32().abs()).collect();

    let reference = rmsnorm_reference(&input, &weight, hidden_size, eps);

    let mut output = vec![0.0f32; hidden_size];
    rmsnorm(&input, &weight, &mut output, hidden_size, eps);

    assert_close(&output, &reference, 1e-4, "rmsnorm_simd_vs_reference");
}

// ---------------------------------------------------------------------------
// test_rmsnorm_batch_multiple_rows
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_batch_multiple_rows() {
    let hidden_size = 64;
    let batch_size = 4;
    let eps = 1e-5;

    // Generate pseudo-random data
    let mut seed: u64 = 99;
    let mut next_f32 = || -> f32 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    let weight: Vec<f32> = (0..hidden_size).map(|_| 0.5 + next_f32().abs()).collect();
    let input: Vec<f32> = (0..batch_size * hidden_size).map(|_| next_f32()).collect();
    let mut output = vec![0.0f32; batch_size * hidden_size];

    rmsnorm_batch(&input, &weight, &mut output, batch_size, hidden_size, eps);

    // Verify each row independently against reference
    for b in 0..batch_size {
        let start = b * hidden_size;
        let end = start + hidden_size;
        let row_ref = rmsnorm_reference(&input[start..end], &weight, hidden_size, eps);
        assert_close(
            &output[start..end],
            &row_ref,
            1e-4,
            &format!("batch_row_{b}"),
        );
    }
}

// ---------------------------------------------------------------------------
// Varied sizes (SIMD tail handling)
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_varied_sizes() {
    let eps = 1e-5;
    for &n in &[
        1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 127, 256, 768, 1024,
    ] {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin() + 0.5).collect();
        let weight: Vec<f32> = (0..n).map(|i| 0.8 + (i as f32 * 0.003)).collect();

        let reference = rmsnorm_reference(&input, &weight, n, eps);

        let mut output = vec![0.0f32; n];
        rmsnorm(&input, &weight, &mut output, n, eps);

        assert_close(&output, &reference, 1e-4, &format!("varied_n{n}"));
    }
}

// ---------------------------------------------------------------------------
// Uniform input
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_uniform_input() {
    let n = 32;
    let val = 3.0f32;
    let input = vec![val; n];
    let weight = vec![1.0f32; n];
    let eps = 1e-5;

    let result = rmsnorm_reference(&input, &weight, n, eps);

    // For uniform input: rms = sqrt(val^2 + eps) ~ |val|
    // out = val / rms * 1.0 ~ val / |val| = sign(val) = 1.0
    for (i, &v) in result.iter().enumerate() {
        let diff = (v - 1.0).abs();
        assert!(diff < 1e-3, "uniform[{i}]: {v} vs 1.0, diff={diff}");
    }
}

// ---------------------------------------------------------------------------
// Large values: numerical stability
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_large_values() {
    let n = 16;
    let input: Vec<f32> = (0..n).map(|i| 1e6 + (i as f32) * 100.0).collect();
    let weight = vec![1.0; n];
    let eps = 1e-5;

    let mut output = vec![0.0f32; n];
    rmsnorm(&input, &weight, &mut output, n, eps);

    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "large[{i}] = {v} not finite");
    }

    let reference = rmsnorm_reference(&input, &weight, n, eps);
    assert_close(&output, &reference, 1e-3, "large_values");
}

// ---------------------------------------------------------------------------
// Typical LLM hidden sizes
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_hidden_2048() {
    let n = 2048;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin()).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.0001).collect();
    let eps = 1e-5;

    let reference = rmsnorm_reference(&input, &weight, n, eps);
    let mut output = vec![0.0f32; n];
    rmsnorm(&input, &weight, &mut output, n, eps);
    assert_close(&output, &reference, 1e-4, "hidden_2048");
}
