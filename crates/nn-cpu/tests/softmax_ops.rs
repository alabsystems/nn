// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for SIMD-accelerated softmax.
//!
//! Validates correctness against a naive reference implementation,
//! verifies output sums to 1.0, checks numerical stability with
//! extreme values, and exercises various dimension sizes including
//! non-SIMD-aligned tails.

use nn_cpu::softmax;

// ============================================================================
// Helpers
// ============================================================================

/// Naive (double-precision reference) softmax for a single row.
fn naive_softmax_f64(input: &[f32]) -> Vec<f32> {
    let max = input
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a, b| a.max(f64::from(b)));
    let exps: Vec<f64> = input.iter().map(|&x| (f64::from(x) - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| (e / sum) as f32).collect()
}

/// Assert element-wise closeness within tolerance.
fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: {va} vs {vb} (diff={diff}, tol={tol})"
        );
    }
}

// ============================================================================
// Correctness vs naive softmax
// ============================================================================

#[test]
fn test_softmax_scalar_vs_naive() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut output = vec![0.0f32; 5];
    softmax::softmax_scalar(&input, &mut output, 5);
    let expected = naive_softmax_f64(&input);
    assert_close(&output, &expected, 1e-6, "scalar_vs_naive");
}

#[test]
fn test_softmax_dispatch_vs_naive() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = vec![0.0f32; 8];
    softmax::softmax(&input, &mut output, 8);
    let expected = naive_softmax_f64(&input);
    // Fast exp approximation (Schraudolph) has ~1-2% relative error per element,
    // which compounds through the softmax normalization. Use 0.01 tolerance.
    assert_close(&output, &expected, 1e-2, "dispatch_vs_naive");
}

#[test]
fn test_softmax_scalar_vs_dispatch_multi_row() {
    let input: Vec<f32> = (0..24).map(|i| (i as f32) * 0.3 - 3.6).collect();
    let mut scalar_out = vec![0.0f32; 24];
    let mut dispatch_out = vec![0.0f32; 24];
    softmax::softmax_scalar(&input, &mut scalar_out, 8);
    softmax::softmax(&input, &mut dispatch_out, 8);
    // Tolerance for fast exp approximation in SIMD path (Schraudolph ~1-2% per element).
    assert_close(&scalar_out, &dispatch_out, 1e-2, "scalar_vs_dispatch_multi");
}

// ============================================================================
// Output sums to 1.0
// ============================================================================

#[test]
fn test_softmax_sums_to_one_small() {
    let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let mut output = vec![0.0f32; 5];
    softmax::softmax(&input, &mut output, 5);
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "small row sum = {sum}, expected 1.0"
    );
}

#[test]
fn test_softmax_sums_to_one_medium() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.2).collect();
    let mut output = vec![0.0f32; 64];
    softmax::softmax(&input, &mut output, 64);
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "medium row sum = {sum}, expected 1.0"
    );
}

#[test]
fn test_softmax_sums_to_one_multi_row() {
    let input: Vec<f32> = (0..30).map(|i| (i as f32) * 0.5 - 7.0).collect();
    let mut output = vec![0.0f32; 30];
    softmax::softmax(&input, &mut output, 10);
    for row in 0..3 {
        let start = row * 10;
        let sum: f32 = output[start..start + 10].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "row {row} sum = {sum}, expected 1.0"
        );
    }
}

// ============================================================================
// Numerical stability (large values)
// ============================================================================

#[test]
fn test_softmax_large_positive_values() {
    // Large positive values should not overflow: max subtraction prevents it.
    let input = vec![1000.0, 1001.0, 1002.0, 1003.0];
    let mut output = vec![0.0f32; 4];
    softmax::softmax_scalar(&input, &mut output, 4);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "large positive: sum = {sum}");
    // All values should be finite and non-negative.
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] is not finite: {v}");
        assert!(v >= 0.0, "output[{i}] is negative: {v}");
    }
}

#[test]
fn test_softmax_large_negative_values() {
    let input = vec![-1000.0, -1001.0, -1002.0, -1003.0];
    let mut output = vec![0.0f32; 4];
    softmax::softmax_scalar(&input, &mut output, 4);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "large negative: sum = {sum}");
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] is not finite: {v}");
        assert!(v >= 0.0, "output[{i}] is negative: {v}");
    }
}

#[test]
fn test_softmax_mixed_extreme_values() {
    // One very large value dominates; rest should be ~0.
    let input = vec![-100.0, 0.0, 100.0, -100.0];
    let mut output = vec![0.0f32; 4];
    softmax::softmax_scalar(&input, &mut output, 4);
    assert!(
        output[2] > 0.99,
        "dominant element should be ~1.0, got {}",
        output[2]
    );
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "mixed extreme: sum = {sum}");
}

#[test]
fn test_softmax_dispatch_large_values_stable() {
    // Verify the SIMD path handles large values without NaN/Inf.
    let input = vec![500.0, 501.0, 502.0, 503.0, 504.0, 505.0, 506.0, 507.0];
    let mut output = vec![0.0f32; 8];
    softmax::softmax(&input, &mut output, 8);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-3, "SIMD large values: sum = {sum}");
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "SIMD output[{i}] not finite: {v}");
        assert!(v >= 0.0, "SIMD output[{i}] negative: {v}");
    }
}

// ============================================================================
// Different dimension sizes (SIMD tail handling)
// ============================================================================

#[test]
fn test_softmax_dim_1() {
    // Single element: softmax is always 1.0.
    let input = vec![42.0];
    let mut output = vec![0.0f32; 1];
    softmax::softmax(&input, &mut output, 1);
    assert!(
        (output[0] - 1.0).abs() < 1e-6,
        "dim=1: expected 1.0, got {}",
        output[0]
    );
}

#[test]
fn test_softmax_dim_2() {
    let input = vec![1.0, 3.0];
    let mut output = vec![0.0f32; 2];
    softmax::softmax(&input, &mut output, 2);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "dim=2: sum = {sum}");
}

#[test]
fn test_softmax_dim_3_not_aligned() {
    let input = vec![1.0, 2.0, 3.0];
    let mut output = vec![0.0f32; 3];
    softmax::softmax(&input, &mut output, 3);
    let expected = naive_softmax_f64(&input);
    assert_close(&output, &expected, 1e-2, "dim=3");
}

#[test]
fn test_softmax_dim_5_not_aligned() {
    let input: Vec<f32> = (0..5).map(|i| i as f32).collect();
    let mut output = vec![0.0f32; 5];
    softmax::softmax(&input, &mut output, 5);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "dim=5: sum = {sum}");
}

#[test]
fn test_softmax_dim_7_not_aligned() {
    let input: Vec<f32> = (0..7).map(|i| (i as f32) * 0.7 - 2.1).collect();
    let mut output = vec![0.0f32; 7];
    softmax::softmax(&input, &mut output, 7);
    let expected = naive_softmax_f64(&input);
    assert_close(&output, &expected, 1e-2, "dim=7");
}

#[test]
fn test_softmax_dim_16_aligned() {
    let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 2.0).collect();
    let mut output = vec![0.0f32; 16];
    softmax::softmax(&input, &mut output, 16);
    let expected = naive_softmax_f64(&input);
    assert_close(&output, &expected, 1e-2, "dim=16");
}

#[test]
fn test_softmax_dim_17_tail_1() {
    let input: Vec<f32> = (0..17).map(|i| (i as f32) * 0.3 - 2.5).collect();
    let mut output = vec![0.0f32; 17];
    softmax::softmax(&input, &mut output, 17);
    let expected = naive_softmax_f64(&input);
    assert_close(&output, &expected, 1e-2, "dim=17");
}

#[test]
fn test_softmax_dim_128_large() {
    let input: Vec<f32> = (0..128).map(|i| (i as f32) * 0.05 - 3.2).collect();
    let mut output = vec![0.0f32; 128];
    softmax::softmax(&input, &mut output, 128);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-3, "dim=128: sum = {sum}");
    let expected = naive_softmax_f64(&input);
    assert_close(&output, &expected, 1e-2, "dim=128");
}

// ============================================================================
// Uniform input (all equal)
// ============================================================================

#[test]
fn test_softmax_uniform_input() {
    // All equal inputs => uniform output = 1/N.
    let n = 8;
    let input = vec![5.0f32; n];
    let mut output = vec![0.0f32; n];
    softmax::softmax(&input, &mut output, n);
    let expected = 1.0 / n as f32;
    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-3,
            "uniform[{i}]: expected {expected}, got {v}"
        );
    }
}

// ============================================================================
// All non-negative outputs
// ============================================================================

#[test]
fn test_softmax_outputs_non_negative() {
    let input: Vec<f32> = (0..33)
        .map(|i| ((i * 7 + 3) % 41) as f32 * 0.5 - 10.0)
        .collect();
    let mut output = vec![0.0f32; 33];
    softmax::softmax(&input, &mut output, 33);
    for (i, &v) in output.iter().enumerate() {
        assert!(v >= 0.0, "output[{i}] is negative: {v}");
        assert!(v.is_finite(), "output[{i}] is not finite: {v}");
    }
}
