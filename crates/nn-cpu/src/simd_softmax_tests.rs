// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD-optimized softmax with explicit tier entry points.

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Naive reference softmax using stdlib exp (f64 precision for ground truth).
fn naive_softmax(input: &[f32]) -> Vec<f32> {
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = input.iter().map(|&x| f64::from(x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| (e / sum) as f32).collect()
}

fn run_scalar(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0f32; n];
    softmax_f32_scalar(input, &mut output, n);
    output
}

fn run_dispatch(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0f32; n];
    softmax_f32(input, &mut output, n);
    output
}

fn run_neon(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0f32; n];
    softmax_f32_neon(input, &mut output, n);
    output
}

fn run_avx2(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0f32; n];
    softmax_f32_avx2(input, &mut output, n);
    output
}

// ---------------------------------------------------------------------------
// Basic: output sums to 1.0
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_sums_to_one() {
    let input = [1.0, 2.0, 3.0, 4.0];
    let out = run_scalar(&input);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "scalar sum = {sum}");
}

#[test]
fn test_dispatch_sums_to_one() {
    let input = [1.0, 2.0, 3.0, 4.0];
    let out = run_dispatch(&input);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-3, "dispatch sum = {sum}");
}

#[test]
fn test_neon_sums_to_one() {
    let input = [1.0, 2.0, 3.0, 4.0];
    let out = run_neon(&input);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-3, "neon sum = {sum}");
}

#[test]
fn test_avx2_sums_to_one() {
    let input = [1.0, 2.0, 3.0, 4.0];
    let out = run_avx2(&input);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-3, "avx2 sum = {sum}");
}

// ---------------------------------------------------------------------------
// Output non-negative
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_non_negative() {
    let input = [-10.0, -5.0, 0.0, 5.0, 10.0];
    let out = run_scalar(&input);
    for (i, &v) in out.iter().enumerate() {
        assert!(v >= 0.0, "scalar output[{i}] = {v} is negative");
    }
}

#[test]
fn test_dispatch_non_negative() {
    let input: Vec<f32> = (-50..50).map(|i| i as f32 * 0.3).collect();
    let out = run_dispatch(&input);
    for (i, &v) in out.iter().enumerate() {
        assert!(v >= 0.0, "dispatch output[{i}] = {v} is negative");
    }
}

// ---------------------------------------------------------------------------
// All zeros -> uniform
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_zeros_uniform() {
    let n = 8;
    let input = vec![0.0f32; n];
    let out = run_scalar(&input);
    let expected = 1.0 / n as f32;
    for (i, &v) in out.iter().enumerate() {
        assert!((v - expected).abs() < 1e-6, "scalar zeros [{i}] = {v}");
    }
}

#[test]
fn test_dispatch_zeros_uniform() {
    let n = 16;
    let input = vec![0.0f32; n];
    let out = run_dispatch(&input);
    let expected = 1.0 / n as f32;
    for (i, &v) in out.iter().enumerate() {
        assert!((v - expected).abs() < 1e-3, "dispatch zeros [{i}] = {v}");
    }
}

// ---------------------------------------------------------------------------
// Numerical stability: large positive values
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_large_positive() {
    let input = [1000.0, 1001.0, 1002.0, 1003.0];
    let out = run_scalar(&input);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "scalar large sum = {sum}");
    for &v in &out {
        assert!(v.is_finite(), "scalar large: {v} is not finite");
    }
}

#[test]
fn test_dispatch_large_positive() {
    let input = [500.0, 501.0, 502.0, 503.0, 504.0, 505.0, 506.0, 507.0];
    let out = run_dispatch(&input);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-2, "dispatch large sum = {sum}");
    for &v in &out {
        assert!(v.is_finite() && v >= 0.0, "dispatch large: {v}");
    }
}

// ---------------------------------------------------------------------------
// Single element -> 1.0
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_single_element() {
    for val in [-100.0, -1.0, 0.0, 1.0, 100.0] {
        let input = [val];
        let out = run_scalar(&input);
        assert!(
            (out[0] - 1.0).abs() < 1e-6,
            "scalar single({val}) = {}",
            out[0]
        );
    }
}

#[test]
fn test_dispatch_single_element() {
    for val in [-50.0, 0.0, 50.0] {
        let input = [val];
        let out = run_dispatch(&input);
        assert!(
            (out[0] - 1.0).abs() < 1e-3,
            "dispatch single({val}) = {}",
            out[0]
        );
    }
}

// ---------------------------------------------------------------------------
// Matches naive reference
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_matches_naive() {
    let input = [0.1, -0.5, 1.2, -1.8, 0.7, 3.0, -2.1, 0.0];
    let out = run_scalar(&input);
    let expected = naive_softmax(&input);
    for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "scalar vs naive [{i}]: {a} vs {b}");
    }
}

#[test]
fn test_dispatch_matches_naive() {
    let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 4.0).collect();
    let out = run_dispatch(&input);
    let expected = naive_softmax(&input);
    for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-2,
            "dispatch vs naive [{i}]: {a} vs {b}, diff={diff}"
        );
    }
}

// ---------------------------------------------------------------------------
// Varied sizes (exercises SIMD tail handling)
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_varied_sizes() {
    for n in [
        1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 256, 1024,
    ] {
        let input: Vec<f32> = (0..n)
            .map(|i| (i as f32) * 0.3 - (n as f32) / 2.0)
            .collect();
        let scalar_out = run_scalar(&input);
        let dispatch_out = run_dispatch(&input);
        for (i, (&a, &b)) in scalar_out.iter().zip(dispatch_out.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < 5e-2,
                "size={n} [{i}]: scalar={a} vs dispatch={b}, diff={diff}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Reference function returns correct output
// ---------------------------------------------------------------------------

#[test]
fn test_reference_matches_scalar() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let ref_out = softmax_f32_reference(&input, 5);
    let scalar_out = run_scalar(&input);
    for (i, (&a, &b)) in ref_out.iter().zip(scalar_out.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-7,
            "reference vs scalar [{i}]: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Ordering preservation
// ---------------------------------------------------------------------------

#[test]
fn test_preserves_ordering() {
    let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    let out = run_dispatch(&input);
    for i in 1..16 {
        assert!(
            out[i] >= out[i - 1] - 1e-6,
            "ordering violated at [{i}]: {} < {}",
            out[i],
            out[i - 1]
        );
    }
}

// ---------------------------------------------------------------------------
// Shift invariance
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_shift_invariance() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let shifted: Vec<f32> = input.iter().map(|&x| x + 1000.0).collect();
    let out_orig = run_scalar(&input);
    let out_shifted = run_scalar(&shifted);
    for (i, (&a, &b)) in out_orig.iter().zip(out_shifted.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "shift invariance [{i}]: {a} vs {b}");
    }
}

// ---------------------------------------------------------------------------
// Output bounded in [0, 1]
// ---------------------------------------------------------------------------

#[test]
fn test_output_bounded() {
    let inputs: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![-100.0, 0.0, 100.0],
        vec![0.0; 10],
        (0..20).map(|i| (i as f32 - 10.0) * 5.0).collect(),
    ];
    for input in &inputs {
        let out = run_dispatch(input);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (0.0..=1.0 + 1e-6).contains(&v),
                "output[{i}] = {v} not in [0, 1]"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Large array
// ---------------------------------------------------------------------------

#[test]
fn test_large_array_1024() {
    let n = 1024;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.73).sin()).collect();
    let out = run_dispatch(&input);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-2, "large array sum = {sum}");
    for (i, &v) in out.iter().enumerate() {
        assert!(v >= 0.0 && v.is_finite(), "large [{i}] = {v}");
    }
}

// ---------------------------------------------------------------------------
// Identical elements -> uniform
// ---------------------------------------------------------------------------

#[test]
fn test_identical_uniform() {
    for val in [-3.0, 0.0, 5.0] {
        for n in [4, 8, 16, 32] {
            let input = vec![val; n];
            let out = run_dispatch(&input);
            let expected = 1.0 / n as f32;
            for (i, &v) in out.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 1e-2,
                    "val={val}, n={n}: [{i}] = {v}, expected ~{expected}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Empty input is a no-op
// ---------------------------------------------------------------------------

#[test]
fn test_empty_input() {
    let input: &[f32] = &[];
    let mut output: Vec<f32> = vec![];
    softmax_f32_scalar(input, &mut output, 0);
    softmax_f32(input, &mut output, 0);
}
