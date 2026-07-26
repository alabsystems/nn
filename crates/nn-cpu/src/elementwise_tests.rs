// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for SIMD-accelerated elementwise operations.
//!
//! Covers: add, mul, relu, silu, sigmoid, tanh, gelu.
//! Tests: known outputs, edge cases (zero, negative, large, small, NaN, Inf),
//! SIMD alignment (lengths that are multiples of 4/8 and non-aligned),
//! empty inputs, single elements, and SIMD-vs-scalar consistency.

use super::*;

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

/// Assert element-wise closeness with tolerance.
fn assert_approx(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch: got {} vs expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{label}[{i}]: expected NaN, got {a}");
        } else if e.is_infinite() {
            assert_eq!(a, e, "{label}[{i}]: expected {e}, got {a}");
        } else {
            let diff = (a - e).abs();
            assert!(
                diff <= tol,
                "{label}[{i}]: expected {e}, got {a} (diff={diff}, tol={tol})"
            );
        }
    }
}

/// Reference sigmoid for single values.
fn ref_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Reference SiLU for single values.
fn ref_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Reference GELU (tanh approximation) for single values.
fn ref_gelu(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    x * 0.5 * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

// -------------------------------------------------------------------------
// add: known outputs
// -------------------------------------------------------------------------

#[test]
fn test_add_known_values() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [10.0, 20.0, 30.0, 40.0];
    let mut out = [0.0f32; 4];
    add(&a, &b, &mut out);
    assert_approx(&out, &[11.0, 22.0, 33.0, 44.0], 0.0, "add_known");
}

#[test]
fn test_add_negative_values() {
    let a = [-1.0, -2.0, -3.0, -4.0, -5.0];
    let b = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut out = [0.0f32; 5];
    add(&a, &b, &mut out);
    assert_approx(&out, &[0.0, 0.0, 0.0, 0.0, 0.0], 0.0, "add_neg");
}

#[test]
fn test_add_zeros() {
    let a = [0.0, 0.0, 0.0, 0.0];
    let b = [0.0, 0.0, 0.0, 0.0];
    let mut out = [0.0f32; 4];
    add(&a, &b, &mut out);
    assert_approx(&out, &[0.0, 0.0, 0.0, 0.0], 0.0, "add_zeros");
}

#[test]
fn test_add_inf() {
    let a = [f32::INFINITY, f32::NEG_INFINITY, 1.0, f32::INFINITY];
    let b = [1.0, 1.0, f32::INFINITY, f32::NEG_INFINITY];
    let mut out = [0.0f32; 4];
    add(&a, &b, &mut out);
    assert_eq!(out[0], f32::INFINITY);
    assert_eq!(out[1], f32::NEG_INFINITY);
    assert_eq!(out[2], f32::INFINITY);
    // inf + neg_inf = NaN
    assert!(out[3].is_nan());
}

#[test]
fn test_add_nan_propagation() {
    let a = [f32::NAN, 1.0, 2.0, 3.0];
    let b = [1.0, f32::NAN, 3.0, 4.0];
    let mut out = [0.0f32; 4];
    add(&a, &b, &mut out);
    assert!(out[0].is_nan(), "NaN + 1.0 should be NaN");
    assert!(out[1].is_nan(), "1.0 + NaN should be NaN");
    assert_approx(&out[2..], &[5.0, 7.0], 0.0, "add_nan_rest");
}

#[test]
fn test_add_very_large_values() {
    let big = f32::MAX;
    let a = [big, big, -big, -big];
    let b = [big, 1.0, -big, 1.0];
    let mut out = [0.0f32; 4];
    add(&a, &b, &mut out);
    assert_eq!(out[0], f32::INFINITY, "overflow to +inf");
    assert_eq!(out[1], big, "MAX + 1.0 rounds to MAX");
    assert_eq!(out[2], f32::NEG_INFINITY, "overflow to -inf");
    assert_eq!(out[3], -big + 1.0, "neg MAX + 1.0");
}

#[test]
fn test_add_very_small_values() {
    let tiny = f32::MIN_POSITIVE;
    let a = [tiny, tiny, -tiny, 0.0];
    let b = [tiny, -tiny, -tiny, tiny];
    let mut out = [0.0f32; 4];
    add(&a, &b, &mut out);
    assert_approx(&out, &[2.0 * tiny, 0.0, -2.0 * tiny, tiny], 0.0, "add_tiny");
}

// -------------------------------------------------------------------------
// mul: known outputs
// -------------------------------------------------------------------------

#[test]
fn test_mul_known_values() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let mut out = [0.0f32; 4];
    mul(&a, &b, &mut out);
    assert_approx(&out, &[5.0, 12.0, 21.0, 32.0], 0.0, "mul_known");
}

#[test]
fn test_mul_by_zero() {
    let a = [1.0, -2.0, 100.0, -0.5];
    let b = [0.0, 0.0, 0.0, 0.0];
    let mut out = [0.0f32; 4];
    mul(&a, &b, &mut out);
    assert_approx(&out, &[0.0, 0.0, 0.0, 0.0], 0.0, "mul_by_zero");
}

#[test]
fn test_mul_by_one() {
    let a = [3.14, -2.71, 0.0, 42.0];
    let b = [1.0, 1.0, 1.0, 1.0];
    let mut out = [0.0f32; 4];
    mul(&a, &b, &mut out);
    assert_approx(&out, &a, 0.0, "mul_by_one");
}

#[test]
fn test_mul_negative_times_negative() {
    let a = [-1.0, -2.0, -3.0, -4.0];
    let b = [-1.0, -2.0, -3.0, -4.0];
    let mut out = [0.0f32; 4];
    mul(&a, &b, &mut out);
    assert_approx(&out, &[1.0, 4.0, 9.0, 16.0], 0.0, "mul_neg_neg");
}

#[test]
fn test_mul_inf() {
    let a = [f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, 0.0];
    let b = [2.0, 3.0, f32::NEG_INFINITY, f32::INFINITY];
    let mut out = [0.0f32; 4];
    mul(&a, &b, &mut out);
    assert_eq!(out[0], f32::INFINITY);
    assert_eq!(out[1], f32::NEG_INFINITY);
    assert_eq!(out[2], f32::NEG_INFINITY);
    // 0 * inf = NaN
    assert!(out[3].is_nan());
}

#[test]
fn test_mul_nan_propagation() {
    let a = [f32::NAN, 5.0, 0.0, f32::NAN];
    let b = [3.0, f32::NAN, f32::NAN, f32::NAN];
    let mut out = [0.0f32; 4];
    mul(&a, &b, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_nan(), "mul_nan[{i}] should be NaN, got {v}");
    }
}

// -------------------------------------------------------------------------
// relu: known outputs and edge cases
// -------------------------------------------------------------------------

#[test]
fn test_relu_positive_passthrough() {
    let input = [0.5, 1.0, 2.5, 100.0];
    let mut out = [0.0f32; 4];
    relu(&input, &mut out);
    assert_approx(&out, &input, 0.0, "relu_positive");
}

#[test]
fn test_relu_negative_zeroed() {
    let input = [-0.5, -1.0, -2.5, -100.0];
    let mut out = [0.0f32; 4];
    relu(&input, &mut out);
    assert_approx(&out, &[0.0, 0.0, 0.0, 0.0], 0.0, "relu_negative");
}

#[test]
fn test_relu_zero() {
    let input = [0.0, -0.0, 0.0, -0.0];
    let mut out = [0.0f32; 4];
    relu(&input, &mut out);
    // relu(0) = 0, relu(-0) = 0 (max(0, -0) = 0 per IEEE 754)
    for &v in &out {
        assert!(v == 0.0, "relu of zero should be 0.0");
    }
}

#[test]
fn test_relu_inf() {
    let input = [f32::INFINITY, f32::NEG_INFINITY];
    let mut out = [0.0f32; 2];
    relu(&input, &mut out);
    assert_eq!(out[0], f32::INFINITY, "relu(+inf) = +inf");
    assert_eq!(out[1], 0.0, "relu(-inf) = 0");
}

#[test]
fn test_relu_nan() {
    // relu(NaN) behavior depends on platform:
    // - Scalar f32::max(0, NaN) = NaN per IEEE 754 maxNum
    // - NEON vmaxq_f32(NaN, 0) = 0 (returns the non-NaN operand)
    // We test that the SIMD path matches the scalar fallback.
    let input = [f32::NAN];
    let mut scalar_out = [0.0f32; 1];
    let mut simd_out = [0.0f32; 1];
    relu_scalar(&input, &mut scalar_out);
    relu(&input, &mut simd_out);
    // Both should produce the same result (either NaN or 0.0 depending on arch).
    if scalar_out[0].is_nan() {
        assert!(
            simd_out[0].is_nan(),
            "relu NaN: scalar is NaN, SIMD should be too"
        );
    } else {
        assert_eq!(
            simd_out[0], scalar_out[0],
            "relu NaN: SIMD should match scalar"
        );
    }
}

// -------------------------------------------------------------------------
// sigmoid: known outputs and edge cases
// -------------------------------------------------------------------------

#[test]
fn test_sigmoid_zero() {
    let input = [0.0];
    let mut out = [0.0f32; 1];
    sigmoid(&input, &mut out);
    assert_approx(&out, &[0.5], 1e-6, "sigmoid_zero");
}

#[test]
fn test_sigmoid_known_values() {
    let input = [-10.0, -1.0, 0.0, 1.0, 10.0];
    let expected: Vec<f32> = input.iter().map(|&x| ref_sigmoid(x)).collect();
    let mut out = vec![0.0f32; 5];
    sigmoid(&input, &mut out);
    assert_approx(&out, &expected, 1e-6, "sigmoid_known");
}

#[test]
fn test_sigmoid_symmetry() {
    // sigmoid(-x) = 1 - sigmoid(x)
    let vals = [0.5, 1.0, 2.0, 5.0];
    let neg_vals: Vec<f32> = vals.iter().map(|x| -x).collect();
    let mut out_pos = vec![0.0f32; 4];
    let mut out_neg = vec![0.0f32; 4];
    sigmoid(&vals, &mut out_pos);
    sigmoid(&neg_vals, &mut out_neg);
    for i in 0..4 {
        let sum = out_pos[i] + out_neg[i];
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "sigmoid({}) + sigmoid({}) = {sum}, expected 1.0",
            vals[i],
            neg_vals[i]
        );
    }
}

#[test]
fn test_sigmoid_large_positive() {
    // sigmoid(large) -> 1.0
    let input = [50.0, 100.0, 500.0, 1000.0];
    let mut out = [0.0f32; 4];
    sigmoid(&input, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "sigmoid({}) = {v}, expected ~1.0",
            input[i]
        );
    }
}

#[test]
fn test_sigmoid_large_negative() {
    // sigmoid(very negative) -> 0.0
    let input = [-50.0, -100.0, -500.0, -1000.0];
    let mut out = [0.0f32; 4];
    sigmoid(&input, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.abs() < 1e-6, "sigmoid({}) = {v}, expected ~0.0", input[i]);
    }
}

#[test]
fn test_sigmoid_output_range() {
    // sigmoid output is always in (0, 1)
    let input = [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0];
    let mut out = vec![0.0f32; input.len()];
    sigmoid(&input, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&v),
            "sigmoid({}) = {v}, out of [0,1]",
            input[i]
        );
    }
}

// -------------------------------------------------------------------------
// tanh: known outputs and edge cases
// -------------------------------------------------------------------------

#[test]
fn test_tanh_zero() {
    let input = [0.0];
    let mut out = [0.0f32; 1];
    tanh(&input, &mut out);
    assert_approx(&out, &[0.0], 1e-7, "tanh_zero");
}

#[test]
fn test_tanh_known_values() {
    let input: [f32; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let expected: Vec<f32> = input.iter().map(|x| x.tanh()).collect();
    let mut out = vec![0.0f32; 5];
    tanh(&input, &mut out);
    assert_approx(&out, &expected, 1e-6, "tanh_known");
}

#[test]
fn test_tanh_symmetry() {
    // tanh(-x) = -tanh(x)
    let vals = [0.1, 0.5, 1.0, 3.0, 10.0];
    let neg_vals: Vec<f32> = vals.iter().map(|x| -x).collect();
    let mut out_pos = vec![0.0f32; 5];
    let mut out_neg = vec![0.0f32; 5];
    tanh(&vals, &mut out_pos);
    tanh(&neg_vals, &mut out_neg);
    for i in 0..5 {
        assert!(
            (out_pos[i] + out_neg[i]).abs() < 1e-6,
            "tanh({}) + tanh({}) should be 0",
            vals[i],
            neg_vals[i]
        );
    }
}

#[test]
fn test_tanh_large_positive() {
    // tanh(large) -> 1.0
    let input = [20.0, 50.0, 100.0, 500.0];
    let mut out = [0.0f32; 4];
    tanh(&input, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "tanh({}) = {v}, expected ~1.0",
            input[i]
        );
    }
}

#[test]
fn test_tanh_large_negative() {
    // tanh(very negative) -> -1.0
    let input = [-20.0, -50.0, -100.0, -500.0];
    let mut out = [0.0f32; 4];
    tanh(&input, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v + 1.0).abs() < 1e-6,
            "tanh({}) = {v}, expected ~-1.0",
            input[i]
        );
    }
}

#[test]
fn test_tanh_output_range() {
    // tanh output is always in [-1, 1]
    let input = [-1000.0, -1.0, 0.0, 1.0, 1000.0];
    let mut out = vec![0.0f32; input.len()];
    tanh(&input, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (-1.0..=1.0).contains(&v),
            "tanh({}) = {v}, out of [-1,1]",
            input[i]
        );
    }
}

// -------------------------------------------------------------------------
// silu (swish): known outputs and edge cases
// -------------------------------------------------------------------------

#[test]
fn test_silu_zero() {
    let input = [0.0];
    let mut out = [0.0f32; 1];
    silu(&input, &mut out);
    // silu(0) = 0 * sigmoid(0) = 0
    assert_approx(&out, &[0.0], 1e-7, "silu_zero");
}

#[test]
fn test_silu_known_values() {
    let input = [-5.0, -1.0, 0.0, 1.0, 5.0];
    let expected: Vec<f32> = input.iter().map(|&x| ref_silu(x)).collect();
    let mut out = vec![0.0f32; 5];
    silu(&input, &mut out);
    assert_approx(&out, &expected, 1e-6, "silu_known");
}

#[test]
fn test_silu_positive_one() {
    // silu(1) = 1 * sigmoid(1) = 1/(1+e^-1) ~ 0.7311
    let input = [1.0];
    let mut out = [0.0f32; 1];
    silu(&input, &mut out);
    let expected = 1.0 / (1.0 + (-1.0_f32).exp());
    assert_approx(&out, &[expected], 1e-6, "silu_one");
}

#[test]
fn test_silu_large_positive() {
    // silu(x) -> x for large positive x (sigmoid -> 1)
    let input = [50.0, 100.0];
    let mut out = [0.0f32; 2];
    silu(&input, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - input[i]).abs() < 1e-3,
            "silu({}) = {v}, expected ~{}",
            input[i],
            input[i]
        );
    }
}

#[test]
fn test_silu_large_negative() {
    // silu(x) -> 0 for large negative x (sigmoid -> 0)
    let input = [-50.0, -100.0];
    let mut out = [0.0f32; 2];
    silu(&input, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.abs() < 1e-6, "silu({}) = {v}, expected ~0.0", input[i]);
    }
}

// -------------------------------------------------------------------------
// gelu: known outputs and edge cases
// -------------------------------------------------------------------------

#[test]
fn test_gelu_zero() {
    let input = [0.0];
    let mut out = [0.0f32; 1];
    gelu(&input, &mut out);
    // gelu(0) = 0 * 0.5 * (1 + tanh(0)) = 0
    assert_approx(&out, &[0.0], 1e-7, "gelu_zero");
}

#[test]
fn test_gelu_known_values() {
    let input = [-3.0, -1.0, 0.0, 1.0, 3.0];
    let expected: Vec<f32> = input.iter().map(|&x| ref_gelu(x)).collect();
    let mut out = vec![0.0f32; 5];
    gelu(&input, &mut out);
    assert_approx(&out, &expected, 1e-6, "gelu_known");
}

#[test]
fn test_gelu_positive_one() {
    // gelu(1) = 1 * 0.5 * (1 + tanh(sqrt(2/pi) * (1 + 0.044715)))
    let input = [1.0];
    let mut out = [0.0f32; 1];
    gelu(&input, &mut out);
    let expected = ref_gelu(1.0);
    assert_approx(&out, &[expected], 1e-6, "gelu_one");
}

#[test]
fn test_gelu_large_positive() {
    // gelu(x) -> x for large positive x
    let input = [20.0, 50.0];
    let mut out = [0.0f32; 2];
    gelu(&input, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - input[i]).abs() < 1e-3,
            "gelu({}) = {v}, expected ~{}",
            input[i],
            input[i]
        );
    }
}

#[test]
fn test_gelu_large_negative() {
    // gelu(x) -> 0 for large negative x
    let input = [-20.0, -50.0];
    let mut out = [0.0f32; 2];
    gelu(&input, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.abs() < 1e-3, "gelu({}) = {v}, expected ~0.0", input[i]);
    }
}

#[test]
fn test_gelu_approximate_negative_one() {
    // gelu(-1) ~ -0.1588 (tanh approximation)
    let input = [-1.0];
    let mut out = [0.0f32; 1];
    gelu(&input, &mut out);
    let expected = ref_gelu(-1.0);
    assert_approx(&out, &[expected], 1e-5, "gelu_neg_one");
}

// -------------------------------------------------------------------------
// SIMD alignment: test at exact multiples of 4 (NEON) and 8 (AVX2)
// -------------------------------------------------------------------------

#[test]
fn test_add_length_4_neon_aligned() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [4.0, 3.0, 2.0, 1.0];
    let mut out = [0.0f32; 4];
    add(&a, &b, &mut out);
    assert_approx(&out, &[5.0, 5.0, 5.0, 5.0], 0.0, "add_len4");
}

#[test]
fn test_add_length_8_avx2_aligned() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let mut out = [0.0f32; 8];
    add(&a, &b, &mut out);
    assert_approx(&out, &[9.0; 8], 0.0, "add_len8");
}

#[test]
fn test_add_length_16_multi_chunk() {
    let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let b: Vec<f32> = (1..=16).map(|x| (17 - x) as f32).collect();
    let mut out = vec![0.0f32; 16];
    add(&a, &b, &mut out);
    // Each pair sums to 17
    assert_approx(&out, &[17.0; 16], 0.0, "add_len16");
}

#[test]
fn test_mul_length_4_neon_aligned() {
    let a = [2.0, 3.0, 4.0, 5.0];
    let b = [5.0, 4.0, 3.0, 2.0];
    let mut out = [0.0f32; 4];
    mul(&a, &b, &mut out);
    assert_approx(&out, &[10.0, 12.0, 12.0, 10.0], 0.0, "mul_len4");
}

#[test]
fn test_mul_length_8_avx2_aligned() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [0.5; 8];
    let mut out = [0.0f32; 8];
    mul(&a, &b, &mut out);
    assert_approx(
        &out,
        &[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        0.0,
        "mul_len8",
    );
}

#[test]
fn test_relu_length_8_avx2_aligned() {
    let input = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let mut out = [0.0f32; 8];
    relu(&input, &mut out);
    assert_approx(
        &out,
        &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        0.0,
        "relu_len8",
    );
}

#[test]
fn test_sigmoid_length_8_avx2_aligned() {
    let input = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let expected: Vec<f32> = input.iter().map(|&x| ref_sigmoid(x)).collect();
    let mut out = vec![0.0f32; 8];
    sigmoid(&input, &mut out);
    assert_approx(&out, &expected, 1e-6, "sigmoid_len8");
}

#[test]
fn test_tanh_length_8_avx2_aligned() {
    let input: [f32; 8] = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let expected: Vec<f32> = input.iter().map(|x| x.tanh()).collect();
    let mut out = vec![0.0f32; 8];
    tanh(&input, &mut out);
    assert_approx(&out, &expected, 1e-6, "tanh_len8");
}

#[test]
fn test_silu_length_8_avx2_aligned() {
    let input = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let expected: Vec<f32> = input.iter().map(|&x| ref_silu(x)).collect();
    let mut out = vec![0.0f32; 8];
    silu(&input, &mut out);
    assert_approx(&out, &expected, 1e-6, "silu_len8");
}

#[test]
fn test_gelu_length_8_avx2_aligned() {
    let input = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let expected: Vec<f32> = input.iter().map(|&x| ref_gelu(x)).collect();
    let mut out = vec![0.0f32; 8];
    gelu(&input, &mut out);
    assert_approx(&out, &expected, 1e-6, "gelu_len8");
}

// -------------------------------------------------------------------------
// Non-aligned lengths: exercise SIMD tail handling
// -------------------------------------------------------------------------

#[test]
fn test_add_length_1() {
    let a = [7.0];
    let b = [3.0];
    let mut out = [0.0f32; 1];
    add(&a, &b, &mut out);
    assert_approx(&out, &[10.0], 0.0, "add_len1");
}

#[test]
fn test_add_length_3() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let mut out = [0.0f32; 3];
    add(&a, &b, &mut out);
    assert_approx(&out, &[5.0, 7.0, 9.0], 0.0, "add_len3");
}

#[test]
fn test_add_length_5() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [5.0, 4.0, 3.0, 2.0, 1.0];
    let mut out = [0.0f32; 5];
    add(&a, &b, &mut out);
    assert_approx(&out, &[6.0, 6.0, 6.0, 6.0, 6.0], 0.0, "add_len5");
}

#[test]
fn test_add_length_7() {
    let a: Vec<f32> = (1..=7).map(|x| x as f32).collect();
    let b: Vec<f32> = (1..=7).map(|x| (8 - x) as f32).collect();
    let mut out = vec![0.0f32; 7];
    add(&a, &b, &mut out);
    assert_approx(&out, &[8.0; 7], 0.0, "add_len7");
}

#[test]
fn test_add_length_9() {
    let a: Vec<f32> = (0..9).map(|x| x as f32).collect();
    let b: Vec<f32> = (0..9).map(|x| (9 - x) as f32).collect();
    let mut out = vec![0.0f32; 9];
    add(&a, &b, &mut out);
    assert_approx(&out, &[9.0; 9], 0.0, "add_len9");
}

#[test]
fn test_mul_length_5() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [2.0, 2.0, 2.0, 2.0, 2.0];
    let mut out = [0.0f32; 5];
    mul(&a, &b, &mut out);
    assert_approx(&out, &[2.0, 4.0, 6.0, 8.0, 10.0], 0.0, "mul_len5");
}

#[test]
fn test_mul_length_7() {
    let a: Vec<f32> = (1..=7).map(|x| x as f32).collect();
    let b = vec![0.5f32; 7];
    let mut out = vec![0.0f32; 7];
    mul(&a, &b, &mut out);
    let expected: Vec<f32> = (1..=7).map(|x| x as f32 * 0.5).collect();
    assert_approx(&out, &expected, 0.0, "mul_len7");
}

#[test]
fn test_relu_length_5() {
    let input = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut out = [0.0f32; 5];
    relu(&input, &mut out);
    assert_approx(&out, &[0.0, 0.0, 0.0, 1.0, 2.0], 0.0, "relu_len5");
}

#[test]
fn test_relu_length_9() {
    let input: Vec<f32> = (-4..=4).map(|x| x as f32).collect();
    let mut out = vec![0.0f32; 9];
    relu(&input, &mut out);
    let expected: Vec<f32> = input.iter().map(|&x| x.max(0.0)).collect();
    assert_approx(&out, &expected, 0.0, "relu_len9");
}

#[test]
fn test_sigmoid_length_5() {
    let input = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let expected: Vec<f32> = input.iter().map(|&x| ref_sigmoid(x)).collect();
    let mut out = vec![0.0f32; 5];
    sigmoid(&input, &mut out);
    assert_approx(&out, &expected, 1e-6, "sigmoid_len5");
}

#[test]
fn test_tanh_length_5() {
    let input: [f32; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let expected: Vec<f32> = input.iter().map(|x| x.tanh()).collect();
    let mut out = vec![0.0f32; 5];
    tanh(&input, &mut out);
    assert_approx(&out, &expected, 1e-6, "tanh_len5");
}

#[test]
fn test_silu_length_7() {
    let input: Vec<f32> = (-3..=3).map(|x| x as f32).collect();
    let expected: Vec<f32> = input.iter().map(|&x| ref_silu(x)).collect();
    let mut out = vec![0.0f32; 7];
    silu(&input, &mut out);
    assert_approx(&out, &expected, 1e-6, "silu_len7");
}

#[test]
fn test_gelu_length_7() {
    let input: Vec<f32> = (-3..=3).map(|x| x as f32).collect();
    let expected: Vec<f32> = input.iter().map(|&x| ref_gelu(x)).collect();
    let mut out = vec![0.0f32; 7];
    gelu(&input, &mut out);
    assert_approx(&out, &expected, 1e-6, "gelu_len7");
}

// -------------------------------------------------------------------------
// Empty input handling
// -------------------------------------------------------------------------

#[test]
fn test_add_empty() {
    let a: [f32; 0] = [];
    let b: [f32; 0] = [];
    let mut out: [f32; 0] = [];
    add(&a, &b, &mut out);
}

#[test]
fn test_mul_empty() {
    let a: [f32; 0] = [];
    let b: [f32; 0] = [];
    let mut out: [f32; 0] = [];
    mul(&a, &b, &mut out);
}

#[test]
fn test_relu_empty() {
    let input: [f32; 0] = [];
    let mut out: [f32; 0] = [];
    relu(&input, &mut out);
}

#[test]
fn test_sigmoid_empty() {
    let input: [f32; 0] = [];
    let mut out: [f32; 0] = [];
    sigmoid(&input, &mut out);
}

#[test]
fn test_tanh_empty() {
    let input: [f32; 0] = [];
    let mut out: [f32; 0] = [];
    tanh(&input, &mut out);
}

#[test]
fn test_silu_empty() {
    let input: [f32; 0] = [];
    let mut out: [f32; 0] = [];
    silu(&input, &mut out);
}

#[test]
fn test_gelu_empty() {
    let input: [f32; 0] = [];
    let mut out: [f32; 0] = [];
    gelu(&input, &mut out);
}

// -------------------------------------------------------------------------
// Single element (pure scalar tail, no SIMD chunk)
// -------------------------------------------------------------------------

#[test]
fn test_add_single() {
    let a = [3.0];
    let b = [7.0];
    let mut out = [0.0f32; 1];
    add(&a, &b, &mut out);
    assert_approx(&out, &[10.0], 0.0, "add_single");
}

#[test]
fn test_mul_single() {
    let a = [3.0];
    let b = [7.0];
    let mut out = [0.0f32; 1];
    mul(&a, &b, &mut out);
    assert_approx(&out, &[21.0], 0.0, "mul_single");
}

#[test]
fn test_relu_single_positive() {
    let input = [5.0];
    let mut out = [0.0f32; 1];
    relu(&input, &mut out);
    assert_approx(&out, &[5.0], 0.0, "relu_single_pos");
}

#[test]
fn test_relu_single_negative() {
    let input = [-5.0];
    let mut out = [0.0f32; 1];
    relu(&input, &mut out);
    assert_approx(&out, &[0.0], 0.0, "relu_single_neg");
}

#[test]
fn test_sigmoid_single() {
    let input = [2.0];
    let mut out = [0.0f32; 1];
    sigmoid(&input, &mut out);
    assert_approx(&out, &[ref_sigmoid(2.0)], 1e-6, "sigmoid_single");
}

#[test]
fn test_tanh_single() {
    let input = [1.5];
    let mut out = [0.0f32; 1];
    tanh(&input, &mut out);
    assert_approx(&out, &[1.5_f32.tanh()], 1e-6, "tanh_single");
}

#[test]
fn test_silu_single() {
    let input = [1.0];
    let mut out = [0.0f32; 1];
    silu(&input, &mut out);
    assert_approx(&out, &[ref_silu(1.0)], 1e-6, "silu_single");
}

#[test]
fn test_gelu_single() {
    let input = [1.0];
    let mut out = [0.0f32; 1];
    gelu(&input, &mut out);
    assert_approx(&out, &[ref_gelu(1.0)], 1e-6, "gelu_single");
}

// -------------------------------------------------------------------------
// SIMD vs scalar consistency across all operations
// -------------------------------------------------------------------------

/// Generate a large input exercising multiple SIMD chunks plus tail.
fn make_large_input(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32) * 0.37 - (n as f32 / 2.0) * 0.37)
        .collect()
}

#[test]
fn test_add_simd_vs_scalar_large() {
    let n = 67; // not a multiple of 4 or 8
    let a = make_large_input(n);
    let b: Vec<f32> = a.iter().map(|x| x * 0.5 + 1.0).collect();
    let mut scalar_out = vec![0.0f32; n];
    let mut simd_out = vec![0.0f32; n];
    add_scalar(&a, &b, &mut scalar_out);
    add(&a, &b, &mut simd_out);
    assert_approx(&simd_out, &scalar_out, 0.0, "add_large_consistency");
}

#[test]
fn test_mul_simd_vs_scalar_large() {
    let n = 67;
    let a = make_large_input(n);
    let b: Vec<f32> = a.iter().map(|x| x * 0.3 - 2.0).collect();
    let mut scalar_out = vec![0.0f32; n];
    let mut simd_out = vec![0.0f32; n];
    mul_scalar(&a, &b, &mut scalar_out);
    mul(&a, &b, &mut simd_out);
    assert_approx(&simd_out, &scalar_out, 0.0, "mul_large_consistency");
}

#[test]
fn test_relu_simd_vs_scalar_large() {
    let n = 67;
    let input = make_large_input(n);
    let mut scalar_out = vec![0.0f32; n];
    let mut simd_out = vec![0.0f32; n];
    relu_scalar(&input, &mut scalar_out);
    relu(&input, &mut simd_out);
    assert_approx(&simd_out, &scalar_out, 0.0, "relu_large_consistency");
}

#[test]
fn test_sigmoid_simd_vs_scalar_large() {
    let n = 67;
    let input = make_large_input(n);
    let mut scalar_out = vec![0.0f32; n];
    let mut simd_out = vec![0.0f32; n];
    sigmoid_scalar(&input, &mut scalar_out);
    sigmoid(&input, &mut simd_out);
    assert_approx(&simd_out, &scalar_out, 1e-6, "sigmoid_large_consistency");
}

#[test]
fn test_tanh_simd_vs_scalar_large() {
    let n = 67;
    let input = make_large_input(n);
    let mut scalar_out = vec![0.0f32; n];
    let mut simd_out = vec![0.0f32; n];
    tanh_scalar(&input, &mut scalar_out);
    tanh(&input, &mut simd_out);
    assert_approx(&simd_out, &scalar_out, 1e-6, "tanh_large_consistency");
}

#[test]
fn test_silu_simd_vs_scalar_large() {
    let n = 67;
    let input = make_large_input(n);
    let mut scalar_out = vec![0.0f32; n];
    let mut simd_out = vec![0.0f32; n];
    silu_scalar(&input, &mut scalar_out);
    silu(&input, &mut simd_out);
    assert_approx(&simd_out, &scalar_out, 1e-6, "silu_large_consistency");
}

#[test]
fn test_gelu_simd_vs_scalar_large() {
    let n = 67;
    let input = make_large_input(n);
    let mut scalar_out = vec![0.0f32; n];
    let mut simd_out = vec![0.0f32; n];
    gelu_scalar(&input, &mut scalar_out);
    gelu(&input, &mut simd_out);
    assert_approx(&simd_out, &scalar_out, 1e-6, "gelu_large_consistency");
}

// -------------------------------------------------------------------------
// Length sweep: tests every alignment class from 0 through 17
// -------------------------------------------------------------------------

#[test]
fn test_relu_length_sweep() {
    for n in 0..=17 {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) - (n as f32 / 2.0)).collect();
        let mut scalar_out = vec![0.0f32; n];
        let mut simd_out = vec![0.0f32; n];
        relu_scalar(&input, &mut scalar_out);
        relu(&input, &mut simd_out);
        assert_approx(&simd_out, &scalar_out, 0.0, &format!("relu_sweep_n{n}"));
    }
}

#[test]
fn test_sigmoid_length_sweep() {
    for n in 0..=17 {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 4.0).collect();
        let mut scalar_out = vec![0.0f32; n];
        let mut simd_out = vec![0.0f32; n];
        sigmoid_scalar(&input, &mut scalar_out);
        sigmoid(&input, &mut simd_out);
        assert_approx(&simd_out, &scalar_out, 1e-6, &format!("sigmoid_sweep_n{n}"));
    }
}

#[test]
fn test_add_length_sweep() {
    for n in 0..=17 {
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let mut scalar_out = vec![0.0f32; n];
        let mut simd_out = vec![0.0f32; n];
        add_scalar(&a, &b, &mut scalar_out);
        add(&a, &b, &mut simd_out);
        assert_approx(&simd_out, &scalar_out, 0.0, &format!("add_sweep_n{n}"));
    }
}

#[test]
fn test_mul_length_sweep() {
    for n in 0..=17 {
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.3).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.7 + 1.0).collect();
        let mut scalar_out = vec![0.0f32; n];
        let mut simd_out = vec![0.0f32; n];
        mul_scalar(&a, &b, &mut scalar_out);
        mul(&a, &b, &mut simd_out);
        assert_approx(&simd_out, &scalar_out, 0.0, &format!("mul_sweep_n{n}"));
    }
}

// -------------------------------------------------------------------------
// Activation functions: verify against mathematical formulas
// -------------------------------------------------------------------------

#[test]
fn test_sigmoid_formula_verification() {
    // sigmoid(x) = 1 / (1 + exp(-x))
    // Verify at several points against manual computation.
    let test_points: [f32; 7] = [-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0];
    for &x in &test_points {
        let expected = 1.0_f32 / (1.0 + (-x).exp());
        let mut out = [0.0f32; 1];
        sigmoid(&[x], &mut out);
        assert!(
            (out[0] - expected).abs() < 1e-6,
            "sigmoid({x}): got {}, expected {expected}",
            out[0]
        );
    }
}

#[test]
fn test_silu_formula_verification() {
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    let test_points: [f32; 7] = [-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0];
    for &x in &test_points {
        let expected = x / (1.0 + (-x).exp());
        let mut out = [0.0f32; 1];
        silu(&[x], &mut out);
        assert!(
            (out[0] - expected).abs() < 1e-6,
            "silu({x}): got {}, expected {expected}",
            out[0]
        );
    }
}

#[test]
fn test_gelu_formula_verification() {
    // gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    let test_points: [f32; 7] = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];
    for &x in &test_points {
        let expected = x * 0.5 * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh());
        let mut out = [0.0f32; 1];
        gelu(&[x], &mut out);
        assert!(
            (out[0] - expected).abs() < 1e-6,
            "gelu({x}): got {}, expected {expected}",
            out[0]
        );
    }
}

#[test]
fn test_relu_formula_verification() {
    // relu(x) = max(0, x)
    let test_points: [f32; 7] = [-10.0, -1.0, -0.001, 0.0, 0.001, 1.0, 10.0];
    for &x in &test_points {
        let expected = x.max(0.0);
        let mut out = [0.0f32; 1];
        relu(&[x], &mut out);
        assert_eq!(
            out[0], expected,
            "relu({x}): got {}, expected {expected}",
            out[0]
        );
    }
}

#[test]
fn test_tanh_formula_verification() {
    // tanh(x) via standard library
    let test_points: [f32; 7] = [-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0];
    for &x in &test_points {
        let expected = x.tanh();
        let mut out = [0.0f32; 1];
        tanh(&[x], &mut out);
        assert!(
            (out[0] - expected).abs() < 1e-6,
            "tanh({x}): got {}, expected {expected}",
            out[0]
        );
    }
}

// -------------------------------------------------------------------------
// Scalar-only functions: direct tests
// -------------------------------------------------------------------------

#[test]
fn test_add_scalar_known() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let mut out = [0.0f32; 3];
    add_scalar(&a, &b, &mut out);
    assert_approx(&out, &[5.0, 7.0, 9.0], 0.0, "add_scalar_known");
}

#[test]
fn test_mul_scalar_known() {
    let a = [2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0];
    let mut out = [0.0f32; 3];
    mul_scalar(&a, &b, &mut out);
    assert_approx(&out, &[10.0, 18.0, 28.0], 0.0, "mul_scalar_known");
}

#[test]
fn test_relu_scalar_known() {
    let input = [-3.0, -1.0, 0.0, 1.0, 3.0];
    let mut out = [0.0f32; 5];
    relu_scalar(&input, &mut out);
    assert_approx(&out, &[0.0, 0.0, 0.0, 1.0, 3.0], 0.0, "relu_scalar_known");
}

#[test]
fn test_sigmoid_scalar_known() {
    let input = [0.0];
    let mut out = [0.0f32; 1];
    sigmoid_scalar(&input, &mut out);
    assert_approx(&out, &[0.5], 1e-7, "sigmoid_scalar_zero");
}

#[test]
fn test_tanh_scalar_known() {
    let input = [0.0];
    let mut out = [0.0f32; 1];
    tanh_scalar(&input, &mut out);
    assert_approx(&out, &[0.0], 1e-7, "tanh_scalar_zero");
}

#[test]
fn test_silu_scalar_known() {
    let input = [0.0];
    let mut out = [0.0f32; 1];
    silu_scalar(&input, &mut out);
    assert_approx(&out, &[0.0], 1e-7, "silu_scalar_zero");
}

#[test]
fn test_gelu_scalar_known() {
    let input = [0.0];
    let mut out = [0.0f32; 1];
    gelu_scalar(&input, &mut out);
    assert_approx(&out, &[0.0], 1e-7, "gelu_scalar_zero");
}

// -------------------------------------------------------------------------
// Larger workloads: 256 and 1024 elements
// -------------------------------------------------------------------------

#[test]
fn test_add_256_elements() {
    let n = 256;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.01).collect();
    let mut scalar_out = vec![0.0f32; n];
    let mut simd_out = vec![0.0f32; n];
    add_scalar(&a, &b, &mut scalar_out);
    add(&a, &b, &mut simd_out);
    assert_approx(&simd_out, &scalar_out, 0.0, "add_256");
}

#[test]
fn test_relu_1024_elements() {
    let n = 1024;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) - 512.0).collect();
    let mut scalar_out = vec![0.0f32; n];
    let mut simd_out = vec![0.0f32; n];
    relu_scalar(&input, &mut scalar_out);
    relu(&input, &mut simd_out);
    assert_approx(&simd_out, &scalar_out, 0.0, "relu_1024");
}

#[test]
fn test_sigmoid_1024_elements() {
    let n = 1024;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.02).collect();
    let mut scalar_out = vec![0.0f32; n];
    let mut simd_out = vec![0.0f32; n];
    sigmoid_scalar(&input, &mut scalar_out);
    sigmoid(&input, &mut simd_out);
    assert_approx(&simd_out, &scalar_out, 1e-6, "sigmoid_1024");
}

#[test]
fn test_gelu_1024_elements() {
    let n = 1024;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.01).collect();
    let mut scalar_out = vec![0.0f32; n];
    let mut simd_out = vec![0.0f32; n];
    gelu_scalar(&input, &mut scalar_out);
    gelu(&input, &mut simd_out);
    assert_approx(&simd_out, &scalar_out, 1e-6, "gelu_1024");
}

// -------------------------------------------------------------------------
// Panic tests (mismatched lengths)
// -------------------------------------------------------------------------

#[test]
#[should_panic]
fn test_add_mismatched_a_b_len() {
    let a = [1.0, 2.0];
    let b = [1.0, 2.0, 3.0];
    let mut out = [0.0f32; 2];
    add(&a, &b, &mut out);
}

#[test]
#[should_panic]
fn test_add_mismatched_a_out_len() {
    let a = [1.0, 2.0, 3.0];
    let b = [1.0, 2.0, 3.0];
    let mut out = [0.0f32; 2];
    add(&a, &b, &mut out);
}

#[test]
#[should_panic]
fn test_mul_mismatched_len() {
    let a = [1.0, 2.0];
    let b = [1.0];
    let mut out = [0.0f32; 2];
    mul(&a, &b, &mut out);
}

#[test]
#[should_panic]
fn test_relu_mismatched_len() {
    let input = [1.0, 2.0, 3.0];
    let mut out = [0.0f32; 2];
    relu(&input, &mut out);
}

#[test]
#[should_panic]
fn test_sigmoid_mismatched_len() {
    let input = [1.0, 2.0];
    let mut out = [0.0f32; 3];
    sigmoid(&input, &mut out);
}

#[test]
#[should_panic]
fn test_tanh_mismatched_len() {
    let input = [1.0];
    let mut out = [0.0f32; 2];
    tanh(&input, &mut out);
}

#[test]
#[should_panic]
fn test_silu_mismatched_len() {
    let input = [1.0, 2.0, 3.0, 4.0];
    let mut out = [0.0f32; 3];
    silu(&input, &mut out);
}

#[test]
#[should_panic]
fn test_gelu_mismatched_len() {
    let input = [1.0, 2.0];
    let mut out = [0.0f32; 5];
    gelu(&input, &mut out);
}
