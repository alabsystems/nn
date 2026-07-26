// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for CPU SIMD softmax implementation.

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Naive reference softmax for a single row (uses stdlib exp).
fn naive_softmax(input: &[f32]) -> Vec<f32> {
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Naive reference log-softmax for a single row.
fn naive_log_softmax(input: &[f32]) -> Vec<f32> {
    let sm = naive_softmax(input);
    sm.iter().map(|&x| x.ln()).collect()
}

/// Helper: run softmax_scalar on a flat slice with given dim_size,
/// return the output buffer.
fn run_scalar(input: &[f32], dim_size: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    softmax_scalar(input, &mut output, dim_size);
    output
}

/// Helper: run the auto-dispatched softmax.
fn run_dispatch(input: &[f32], dim_size: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    softmax(input, &mut output, dim_size);
    output
}

// ---------------------------------------------------------------------------
// Basic softmax: output sums to 1.0
// ---------------------------------------------------------------------------

#[test]
fn test_basic_softmax_sums_to_one() {
    let input = [1.0, 2.0, 3.0, 4.0];
    let out = run_scalar(&input, 4);
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "scalar softmax should sum to 1.0, got {sum}"
    );
}

#[test]
fn test_dispatch_softmax_sums_to_one() {
    let input = [1.0, 2.0, 3.0, 4.0];
    let out = run_dispatch(&input, 4);
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "dispatch softmax should sum to ~1.0, got {sum}"
    );
}

// ---------------------------------------------------------------------------
// Softmax preserves max element index
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_preserves_argmax() {
    let input = [1.0, 5.0, 2.0, 3.0, 0.5];
    let out = run_scalar(&input, 5);
    let argmax_in = input
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let argmax_out = out
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(
        argmax_in, argmax_out,
        "argmax should be preserved: input argmax={argmax_in}, output argmax={argmax_out}"
    );
}

#[test]
fn test_dispatch_preserves_argmax() {
    let input = [0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6];
    let out = run_dispatch(&input, 8);
    let argmax_in = input
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let argmax_out = out
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(argmax_in, argmax_out);
}

// ---------------------------------------------------------------------------
// Softmax output is non-negative
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_output_non_negative() {
    let input = [-10.0, -5.0, 0.0, 5.0, 10.0];
    let out = run_scalar(&input, 5);
    for (i, &v) in out.iter().enumerate() {
        assert!(v >= 0.0, "output[{i}] = {v} is negative");
    }
}

#[test]
fn test_dispatch_output_non_negative() {
    let input: Vec<f32> = (-50..50).map(|i| i as f32 * 0.3).collect();
    let out = run_dispatch(&input, input.len());
    for (i, &v) in out.iter().enumerate() {
        assert!(v >= 0.0, "output[{i}] = {v} is negative");
    }
}

// ---------------------------------------------------------------------------
// All-zeros input produces uniform distribution
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_all_zeros_uniform() {
    let n = 8;
    let input = vec![0.0f32; n];
    let out = run_scalar(&input, n);
    let expected = 1.0 / n as f32;
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-6,
            "output[{i}] = {v}, expected uniform = {expected}"
        );
    }
}

#[test]
fn test_dispatch_all_zeros_uniform() {
    let n = 16;
    let input = vec![0.0f32; n];
    let out = run_dispatch(&input, n);
    let expected = 1.0 / n as f32;
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-3,
            "dispatch output[{i}] = {v}, expected uniform ~{expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// Large positive values: numerical stability (max subtraction)
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_large_positive_stability() {
    // Values near exp overflow range. Softmax should still work because we
    // subtract the max before exponentiating.
    let input = [1000.0, 1001.0, 1002.0, 1003.0];
    let out = run_scalar(&input, 4);
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "large positive: sum = {sum}, expected 1.0"
    );
    // All outputs should be finite.
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] = {v} is not finite");
    }
}

#[test]
fn test_dispatch_large_positive_stability() {
    let input = [500.0, 501.0, 502.0, 503.0, 504.0, 505.0, 506.0, 507.0];
    let out = run_dispatch(&input, 8);
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-2,
        "dispatch large positive: sum = {sum}"
    );
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] = {v} is not finite");
        assert!(v >= 0.0, "output[{i}] = {v} is negative");
    }
}

// ---------------------------------------------------------------------------
// Large negative values: output near zero
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_large_negative() {
    // One normal value, rest very negative: softmax should concentrate on the
    // normal value.
    let mut input = vec![-1000.0f32; 8];
    input[3] = 0.0;
    let out = run_scalar(&input, 8);
    assert!(
        (out[3] - 1.0).abs() < 1e-6,
        "dominant element should be ~1.0, got {}",
        out[3]
    );
    for (i, &v) in out.iter().enumerate() {
        if i != 3 {
            assert!(v < 1e-6, "non-dominant output[{i}] = {v} should be ~0");
        }
    }
}

#[test]
fn test_dispatch_large_negative() {
    let mut input = vec![-500.0f32; 16];
    input[7] = 0.0;
    let out = run_dispatch(&input, 16);
    // The dominant element should be close to 1.
    assert!(
        out[7] > 0.9,
        "dominant element should be near 1.0, got {}",
        out[7]
    );
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-2, "sum = {sum}");
}

// ---------------------------------------------------------------------------
// Single element: output = 1.0
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_single_element() {
    for val in [-100.0, -1.0, 0.0, 1.0, 100.0] {
        let input = [val];
        let out = run_scalar(&input, 1);
        assert!(
            (out[0] - 1.0).abs() < 1e-6,
            "single element softmax({val}) = {}, expected 1.0",
            out[0]
        );
    }
}

#[test]
fn test_dispatch_single_element() {
    for val in [-50.0, 0.0, 50.0] {
        let input = [val];
        let out = run_dispatch(&input, 1);
        assert!(
            (out[0] - 1.0).abs() < 1e-3,
            "dispatch single element softmax({val}) = {}, expected ~1.0",
            out[0]
        );
    }
}

// ---------------------------------------------------------------------------
// Two elements: known analytic result
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_two_elements_analytic() {
    // softmax([a, b]) = [1/(1+exp(b-a)), 1/(1+exp(a-b))]
    let a = 1.0f32;
    let b = 3.0f32;
    let input = [a, b];
    let out = run_scalar(&input, 2);
    let expected_0 = 1.0 / (1.0 + (b - a).exp());
    let expected_1 = 1.0 / (1.0 + (a - b).exp());
    assert!(
        (out[0] - expected_0).abs() < 1e-6,
        "two elements: out[0]={}, expected {expected_0}",
        out[0]
    );
    assert!(
        (out[1] - expected_1).abs() < 1e-6,
        "two elements: out[1]={}, expected {expected_1}",
        out[1]
    );
}

#[test]
fn test_scalar_two_elements_symmetric() {
    // softmax([x, x]) = [0.5, 0.5]
    let input = [7.0, 7.0];
    let out = run_scalar(&input, 2);
    assert!(
        (out[0] - 0.5).abs() < 1e-6,
        "symmetric: out[0]={}, expected 0.5",
        out[0]
    );
    assert!(
        (out[1] - 0.5).abs() < 1e-6,
        "symmetric: out[1]={}, expected 0.5",
        out[1]
    );
}

// ---------------------------------------------------------------------------
// Temperature scaling verification
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_temperature_scaling() {
    let input = [1.0, 2.0, 3.0, 4.0];
    // High temperature (T=10): distribution becomes more uniform.
    let temp = 10.0f32;
    let scaled: Vec<f32> = input.iter().map(|&x| x / temp).collect();
    let out_hot = run_scalar(&scaled, 4);
    // Low temperature (T=0.1): distribution becomes more peaked.
    let temp_cold = 0.1f32;
    let scaled_cold: Vec<f32> = input.iter().map(|&x| x / temp_cold).collect();
    let out_cold = run_scalar(&scaled_cold, 4);

    // Hot distribution should be more uniform (lower max prob).
    let max_hot = out_hot.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let max_cold = out_cold.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_hot < max_cold,
        "hot max ({max_hot}) should be < cold max ({max_cold})"
    );

    // Hot distribution entropy should be higher.
    let entropy_hot: f32 = out_hot
        .iter()
        .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
        .sum();
    let entropy_cold: f32 = out_cold
        .iter()
        .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
        .sum();
    assert!(
        entropy_hot > entropy_cold,
        "hot entropy ({entropy_hot}) should be > cold entropy ({entropy_cold})"
    );

    // Both should still sum to 1.
    let sum_hot: f32 = out_hot.iter().sum();
    let sum_cold: f32 = out_cold.iter().sum();
    assert!((sum_hot - 1.0).abs() < 1e-6, "hot sum = {sum_hot}");
    assert!((sum_cold - 1.0).abs() < 1e-6, "cold sum = {sum_cold}");
}

// ---------------------------------------------------------------------------
// log_softmax: log(softmax(x)) identity
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_log_softmax_identity() {
    // log(softmax(x)) should equal x - max - log(sum(exp(x - max)))
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let sm = run_scalar(&input, 5);
    let log_sm: Vec<f32> = sm.iter().map(|&x| x.ln()).collect();
    let expected = naive_log_softmax(&input);
    for (i, (&a, &b)) in log_sm.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "log_softmax[{i}]: {a} vs expected {b}"
        );
    }
}

#[test]
fn test_log_softmax_sum_exp_is_one() {
    // exp(log_softmax(x)) should sum to 1.
    let input = [0.5, 1.5, 2.5, 3.5];
    let sm = run_scalar(&input, 4);
    let log_sm: Vec<f32> = sm.iter().map(|&x| x.ln()).collect();
    let exp_log_sm_sum: f32 = log_sm.iter().map(|&x| x.exp()).sum();
    assert!(
        (exp_log_sm_sum - 1.0).abs() < 1e-5,
        "exp(log_softmax) sum = {exp_log_sm_sum}, expected 1.0"
    );
}

// ---------------------------------------------------------------------------
// Multiple rows: each row sums to 1.0 independently
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_multi_row_independent() {
    // 3 rows of 5 elements.
    let input = [
        1.0, 2.0, 3.0, 4.0, 5.0, // row 0
        -1.0, -2.0, -3.0, -4.0, -5.0, // row 1
        0.0, 0.0, 0.0, 0.0, 0.0, // row 2
    ];
    let out = run_scalar(&input, 5);
    for row in 0..3 {
        let start = row * 5;
        let row_sum: f32 = out[start..start + 5].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-6,
            "row {row} sum = {row_sum}, expected 1.0"
        );
    }
}

#[test]
fn test_dispatch_multi_row_independent() {
    // 4 rows of 8 elements (exercises SIMD full-vector path).
    let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.6).collect();
    let out = run_dispatch(&input, 8);
    for row in 0..4 {
        let start = row * 8;
        let row_sum: f32 = out[start..start + 8].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-2,
            "dispatch row {row} sum = {row_sum}"
        );
    }
}

#[test]
fn test_multi_row_rows_are_independent() {
    // Verify that rows don't influence each other by comparing multi-row
    // output against single-row outputs.
    let row0 = [1.0, 2.0, 3.0];
    let row1 = [10.0, 20.0, 30.0];
    let combined = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];

    let out_combined = run_scalar(&combined, 3);
    let out_row0 = run_scalar(&row0, 3);
    let out_row1 = run_scalar(&row1, 3);

    for i in 0..3 {
        assert!(
            (out_combined[i] - out_row0[i]).abs() < 1e-6,
            "row0 element {i} differs in combined vs solo"
        );
        assert!(
            (out_combined[3 + i] - out_row1[i]).abs() < 1e-6,
            "row1 element {i} differs in combined vs solo"
        );
    }
}

// ---------------------------------------------------------------------------
// SIMD matches reference naive implementation
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_matches_naive() {
    let input = [0.1, -0.5, 1.2, -1.8, 0.7, 3.0, -2.1, 0.0];
    let out = run_scalar(&input, 8);
    let expected = naive_softmax(&input);
    for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "scalar vs naive [{i}]: {a} vs {b}");
    }
}

#[test]
fn test_dispatch_matches_naive_within_tolerance() {
    // SIMD fast-exp has ~1e-2 tolerance vs stdlib exp.
    let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 4.0).collect();
    let out = run_dispatch(&input, 16);
    let expected = naive_softmax(&input);
    for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-2,
            "dispatch vs naive [{i}]: {a} vs {b}, diff={diff}"
        );
    }
}

#[test]
fn test_dispatch_matches_scalar_varied_sizes() {
    // Test multiple sizes to exercise scalar tail handling in SIMD paths.
    // NEON: chunks of 4. AVX2: chunks of 8.
    // The Schraudolph fast-exp has ~3-5% relative error which compounds
    // through softmax normalization, so we use a 5e-2 tolerance.
    for n in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
        let input: Vec<f32> = (0..n)
            .map(|i| (i as f32) * 0.3 - (n as f32) / 2.0)
            .collect();
        let scalar_out = run_scalar(&input, n);
        let dispatch_out = run_dispatch(&input, n);
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
// Edge case: very large array (1000+ elements)
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_large_array() {
    let n = 1024;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.12).collect();
    let out = run_scalar(&input, n);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "large array scalar sum = {sum}");
    for (i, &v) in out.iter().enumerate() {
        assert!(v >= 0.0, "large array scalar output[{i}] = {v} is negative");
        assert!(
            v.is_finite(),
            "large array scalar output[{i}] = {v} is not finite"
        );
    }
}

#[test]
fn test_dispatch_large_array() {
    let n = 2048;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.73).sin()).collect();
    let out = run_dispatch(&input, n);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-2, "large array dispatch sum = {sum}");
    for (i, &v) in out.iter().enumerate() {
        assert!(
            v >= 0.0,
            "large array dispatch output[{i}] = {v} is negative"
        );
        assert!(
            v.is_finite(),
            "large array dispatch output[{i}] = {v} is not finite"
        );
    }
}

#[test]
fn test_large_array_multi_row() {
    // 100 rows of 128 elements each.
    let n = 128;
    let rows = 100;
    let input: Vec<f32> = (0..n * rows)
        .map(|i| ((i as f32) * 0.37).sin() * 3.0)
        .collect();
    let out = run_scalar(&input, n);
    for row in 0..rows {
        let start = row * n;
        let row_sum: f32 = out[start..start + n].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "large multi-row scalar: row {row} sum = {row_sum}"
        );
    }
}

// ---------------------------------------------------------------------------
// Edge case: identical elements produce uniform distribution
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_identical_elements_uniform() {
    for val in [-5.0, 0.0, 1.0, 42.0] {
        for n in [2, 4, 7, 16] {
            let input = vec![val; n];
            let out = run_scalar(&input, n);
            let expected = 1.0 / n as f32;
            for (i, &v) in out.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 1e-6,
                    "val={val}, n={n}: output[{i}]={v}, expected {expected}"
                );
            }
        }
    }
}

#[test]
fn test_dispatch_identical_elements_uniform() {
    for val in [-3.0, 0.0, 5.0] {
        for n in [4, 8, 16, 32] {
            let input = vec![val; n];
            let out = run_dispatch(&input, n);
            let expected = 1.0 / n as f32;
            for (i, &v) in out.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 1e-2,
                    "dispatch val={val}, n={n}: output[{i}]={v}, expected ~{expected}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// fast_exp_f32 accuracy
// ---------------------------------------------------------------------------

#[test]
fn test_fast_exp_accuracy_sweep() {
    // Sweep a range of inputs and check relative error.
    for i in -125..=125 {
        let x = i as f32 * 0.4; // range: -50..50
        let exact = x.exp();
        let approx = fast_exp_f32(x);
        if exact.is_finite() && approx.is_finite() && exact > 1e-15 {
            let rel = (exact - approx).abs() / exact;
            assert!(
                rel < 5e-2,
                "fast_exp({x}): exact={exact}, approx={approx}, rel={rel}"
            );
        }
    }
}

#[test]
fn test_fast_exp_zero() {
    let result = fast_exp_f32(0.0);
    // Schraudolph approximation has ~3% relative error.
    assert!(
        (result - 1.0).abs() < 0.05,
        "fast_exp(0.0) = {result}, expected ~1.0"
    );
}

#[test]
fn test_fast_exp_large_negative_clamps() {
    let result = fast_exp_f32(-100.0);
    assert!(
        result >= 0.0 && result.is_finite(),
        "fast_exp(-100) = {result}, expected a small non-negative finite value"
    );
}

#[test]
fn test_fast_exp_large_positive_clamps() {
    let result = fast_exp_f32(100.0);
    assert!(
        result.is_finite(),
        "fast_exp(100) = {result}, expected a finite value"
    );
}

// ---------------------------------------------------------------------------
// Softmax ordering preservation
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_preserves_ordering() {
    // Monotonically increasing input should produce monotonically increasing output.
    let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let out = run_scalar(&input, 10);
    for i in 1..10 {
        assert!(
            out[i] >= out[i - 1],
            "ordering violated at [{i}]: {} < {}",
            out[i],
            out[i - 1]
        );
    }
}

#[test]
fn test_dispatch_preserves_ordering() {
    let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    let out = run_dispatch(&input, 16);
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
// Output bounded in [0, 1]
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_output_bounded_zero_one() {
    let inputs: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![-100.0, 0.0, 100.0],
        vec![0.0; 10],
        (0..20).map(|i| (i as f32 - 10.0) * 5.0).collect(),
    ];
    for input in &inputs {
        let out = run_scalar(input, input.len());
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "output[{i}] = {v} not in [0, 1] for input {input:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Shift invariance: softmax(x) == softmax(x + c)
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_shift_invariance() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let shifted: Vec<f32> = input.iter().map(|&x| x + 1000.0).collect();
    let out_orig = run_scalar(&input, 5);
    let out_shifted = run_scalar(&shifted, 5);
    for (i, (&a, &b)) in out_orig.iter().zip(out_shifted.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "shift invariance failed at [{i}]: {a} vs {b}"
        );
    }
}

#[test]
fn test_dispatch_shift_invariance() {
    let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let shifted: Vec<f32> = input.iter().map(|&x| x + 500.0).collect();
    let out_orig = run_dispatch(&input, 8);
    let out_shifted = run_dispatch(&shifted, 8);
    for (i, (&a, &b)) in out_orig.iter().zip(out_shifted.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-2,
            "dispatch shift invariance failed at [{i}]: {a} vs {b}, diff={diff}"
        );
    }
}

// ---------------------------------------------------------------------------
// Panic on invalid inputs
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "dim_size must be > 0")]
fn test_scalar_dim_size_zero_panics() {
    let input = [1.0, 2.0];
    let mut output = [0.0f32; 2];
    softmax_scalar(&input, &mut output, 0);
}

#[test]
#[should_panic(expected = "input length must be a multiple of dim_size")]
fn test_scalar_misaligned_length_panics() {
    let input = [1.0, 2.0, 3.0];
    let mut output = [0.0f32; 3];
    softmax_scalar(&input, &mut output, 2);
}

#[test]
#[should_panic]
fn test_scalar_mismatched_lengths_panics() {
    let input = [1.0, 2.0, 3.0];
    let mut output = [0.0f32; 4];
    softmax_scalar(&input, &mut output, 3);
}

// ---------------------------------------------------------------------------
// Regression: values that previously exposed numerical issues
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_near_zero_differences() {
    // All values very close together: should produce near-uniform output
    // without catastrophic cancellation.
    let input = [1.0000001, 1.0000002, 1.0000003, 1.0000004];
    let out = run_scalar(&input, 4);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "near-zero diff sum = {sum}");
    // Distribution should be nearly uniform.
    for &v in &out {
        assert!(
            (v - 0.25).abs() < 1e-3,
            "near-zero diff: element {v}, expected ~0.25"
        );
    }
}

#[test]
fn test_scalar_mixed_positive_negative() {
    let input = [-10.0, -5.0, 0.0, 5.0, 10.0];
    let out = run_scalar(&input, 5);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "mixed sum = {sum}");
    // Verify descending input order produces descending output order.
    // Input is ascending, so output should be ascending.
    for i in 1..5 {
        assert!(
            out[i] >= out[i - 1],
            "mixed ordering: out[{i}]={} < out[{}]={}",
            out[i],
            i - 1,
            out[i - 1]
        );
    }
}

// ---------------------------------------------------------------------------
// Softmax is a proper probability distribution
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_probability_distribution_properties() {
    let input = [2.0, 1.0, 0.1, -1.0, 3.0, 0.5, -0.5, 1.5];
    let out = run_scalar(&input, 8);

    // 1. Sum to 1.
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "prob sum = {sum}");

    // 2. All non-negative.
    for &v in &out {
        assert!(v >= 0.0, "negative probability: {v}");
    }

    // 3. All <= 1.
    for &v in &out {
        assert!(v <= 1.0 + 1e-6, "probability > 1: {v}");
    }

    // 4. All finite.
    for &v in &out {
        assert!(v.is_finite(), "non-finite probability: {v}");
    }
}
