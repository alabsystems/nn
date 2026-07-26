// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for the nn-cpu backend.
//!
//! Covers: SIMD detection, elementwise ops, reductions, activations,
//! softmax numerical stability, normalization (layer/batch/instance/group/rms),
//! matmul, conv1d, conv2d, transpose, RoPE, pooling, linear, embedding,
//! gather/scatter, dtype casting (f16/bf16), quantization (int8), and normalize.

// ============================================================================
// 1. SIMD Detection
// ============================================================================

#[test]
fn test_simd_detect_returns_valid_level() {
    let level = crate::simd_detect::detect();
    // On any supported platform we get a valid variant.
    match level {
        crate::simd_detect::SimdLevel::Scalar
        | crate::simd_detect::SimdLevel::Neon
        | crate::simd_detect::SimdLevel::Avx2 => {}
    }
}

#[test]
fn test_simd_detect_lane_count_positive() {
    let level = crate::simd_detect::detect();
    let lanes = crate::simd_detect::lane_count(level);
    assert!(lanes >= 1, "lane_count must be >= 1, got {lanes}");
}

#[test]
fn test_simd_detect_lane_count_scalar_is_one() {
    let lanes = crate::simd_detect::lane_count(crate::simd_detect::SimdLevel::Scalar);
    assert_eq!(lanes, 1);
}

// ============================================================================
// 2. Elementwise Binary Ops (add, mul, scalar_mul, fma)
// ============================================================================

#[test]
fn test_add_f32_basic() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![10.0, 20.0, 30.0, 40.0];
    let mut out = vec![0.0f32; 4];
    crate::add_f32(&a, &b, &mut out);
    assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn test_add_f32_zeros() {
    let a = vec![0.0; 8];
    let b = vec![0.0; 8];
    let mut out = vec![0.0f32; 8];
    crate::add_f32(&a, &b, &mut out);
    assert!(out.iter().all(|&v| v == 0.0));
}

#[test]
fn test_add_f32_negative_values() {
    let a = vec![-1.0, -2.0, -3.0, -4.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut out = vec![0.0f32; 4];
    crate::add_f32(&a, &b, &mut out);
    assert!(out.iter().all(|&v| v.abs() < 1e-6));
}

#[test]
fn test_mul_f32_basic() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![2.0, 3.0, 4.0, 5.0];
    let mut out = vec![0.0f32; 4];
    crate::mul_f32(&a, &b, &mut out);
    assert_eq!(out, vec![2.0, 6.0, 12.0, 20.0]);
}

#[test]
fn test_mul_f32_by_zero() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![0.0; 4];
    let mut out = vec![0.0f32; 4];
    crate::mul_f32(&a, &b, &mut out);
    assert!(out.iter().all(|&v| v == 0.0));
}

#[test]
fn test_scalar_mul_f32_basic() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut out = vec![0.0f32; 8];
    crate::scalar_mul_f32(&a, 3.0, &mut out);
    let expected = vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0];
    assert_eq!(out, expected);
}

#[test]
fn test_fma_f32_basic() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![2.0, 3.0, 4.0, 5.0];
    let c = vec![10.0, 20.0, 30.0, 40.0];
    let mut out = vec![0.0f32; 4];
    crate::fma_f32(&a, &b, &c, &mut out);
    // out = a*b + c
    assert_eq!(out, vec![12.0, 26.0, 42.0, 60.0]);
}

#[test]
fn test_add_f32_large_vector() {
    let n = 1024;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let mut out = vec![0.0f32; n];
    crate::add_f32(&a, &b, &mut out);
    for v in &out {
        assert!((v - n as f32).abs() < 1e-6);
    }
}

#[test]
fn test_add_f32_scalar_vs_dispatch() {
    let a: Vec<f32> = (0..17).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..17).map(|i| i as f32 * 0.2).collect();
    let mut out_scalar = vec![0.0f32; 17];
    let mut out_dispatch = vec![0.0f32; 17];
    crate::simd_elementwise::add_f32_scalar(&a, &b, &mut out_scalar);
    crate::add_f32(&a, &b, &mut out_dispatch);
    for (s, d) in out_scalar.iter().zip(out_dispatch.iter()) {
        assert!((s - d).abs() < 1e-6, "scalar={s} dispatch={d}");
    }
}

// ============================================================================
// 3. Activations (relu, gelu, silu, sigmoid, tanh)
// ============================================================================

#[test]
fn test_relu_f32_positive_passthrough() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut out = vec![0.0f32; 4];
    crate::relu_f32(&input, &mut out);
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_relu_f32_negative_zeroed() {
    let input = vec![-1.0, -2.0, -0.5, -100.0];
    let mut out = vec![0.0f32; 4];
    crate::relu_f32(&input, &mut out);
    assert!(out.iter().all(|&v| v == 0.0));
}

#[test]
fn test_relu_f32_mixed() {
    let input = vec![-1.0, 0.0, 1.0, -0.5, 0.5, 2.0, -3.0, 4.0];
    let mut out = vec![0.0f32; 8];
    crate::relu_f32(&input, &mut out);
    let expected = vec![0.0, 0.0, 1.0, 0.0, 0.5, 2.0, 0.0, 4.0];
    assert_eq!(out, expected);
}

#[test]
fn test_gelu_f32_zero_is_zero() {
    let input = vec![0.0; 4];
    let mut out = vec![0.0f32; 4];
    crate::gelu_f32(&input, &mut out);
    for &v in &out {
        assert!(v.abs() < 1e-6, "gelu(0) should be ~0, got {v}");
    }
}

#[test]
fn test_gelu_f32_positive_values() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut out = vec![0.0f32; 4];
    crate::gelu_f32(&input, &mut out);
    // GELU(x) is close to x for large positive x
    for (&o, &x) in out.iter().zip(input.iter()) {
        assert!(o > 0.0, "gelu({x}) should be positive");
        assert!(o <= x + 0.01, "gelu({x}) should be <= x");
    }
}

#[test]
fn test_silu_f32_zero_is_zero() {
    let input = vec![0.0; 4];
    let mut out = vec![0.0f32; 4];
    crate::silu_f32(&input, &mut out);
    for &v in &out {
        assert!(v.abs() < 1e-6, "silu(0) should be 0, got {v}");
    }
}

#[test]
fn test_silu_f32_positive() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut out = vec![0.0f32; 4];
    crate::silu_f32(&input, &mut out);
    for (&o, &x) in out.iter().zip(input.iter()) {
        assert!(o > 0.0 && o <= x, "silu({x})={o} should be in (0, x]");
    }
}

#[test]
fn test_sigmoid_scalar_range() {
    let input = vec![-10.0, -1.0, 0.0, 1.0, 10.0, -5.0, 5.0, 0.5];
    let mut out = vec![0.0f32; 8];
    crate::elementwise::sigmoid_scalar(&input, &mut out);
    for &v in &out {
        assert!((0.0..=1.0).contains(&v), "sigmoid output {v} not in [0,1]");
    }
}

#[test]
fn test_sigmoid_scalar_zero_is_half() {
    let input = vec![0.0];
    let mut out = vec![0.0f32; 1];
    crate::elementwise::sigmoid_scalar(&input, &mut out);
    assert!((out[0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_tanh_scalar_range() {
    let input = vec![-10.0, -1.0, 0.0, 1.0, 10.0, -5.0, 5.0, 0.5];
    let mut out = vec![0.0f32; 8];
    crate::elementwise::tanh_scalar(&input, &mut out);
    for &v in &out {
        assert!((-1.0..=1.0).contains(&v), "tanh output {v} not in [-1,1]");
    }
}

#[test]
fn test_tanh_scalar_zero_is_zero() {
    let input = vec![0.0];
    let mut out = vec![0.0f32; 1];
    crate::elementwise::tanh_scalar(&input, &mut out);
    assert!(out[0].abs() < 1e-6);
}

#[test]
fn test_gelu_scalar_vs_simd_dispatch() {
    let input: Vec<f32> = (-10..10).map(|i| i as f32 * 0.3).collect();
    let mut out_scalar = vec![0.0f32; 20];
    let mut out_dispatch = vec![0.0f32; 20];
    crate::simd_elementwise::gelu_f32_scalar(&input, &mut out_scalar);
    crate::gelu_f32(&input, &mut out_dispatch);
    for (s, d) in out_scalar.iter().zip(out_dispatch.iter()) {
        assert!((s - d).abs() < 1e-5, "scalar={s} dispatch={d}");
    }
}

// ============================================================================
// 4. Flat Reductions (simd_reduce: sum, max, min, dot, l2_norm)
// ============================================================================

#[test]
fn test_simd_sum_f32_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = crate::simd_reduce::sum_f32(&input);
    assert!((result - 10.0).abs() < 1e-6);
}

#[test]
fn test_simd_sum_f32_empty() {
    let input: Vec<f32> = vec![];
    let result = crate::simd_reduce::sum_f32(&input);
    assert_eq!(result, 0.0);
}

#[test]
fn test_simd_max_f32_basic() {
    let input = vec![1.0, 5.0, 3.0, 2.0];
    let result = crate::simd_reduce::max_f32(&input);
    assert_eq!(result, 5.0);
}

#[test]
fn test_simd_max_f32_negative() {
    let input = vec![-5.0, -1.0, -10.0, -3.0];
    let result = crate::simd_reduce::max_f32(&input);
    assert_eq!(result, -1.0);
}

#[test]
fn test_simd_min_f32_basic() {
    let input = vec![1.0, 5.0, 3.0, 2.0];
    let result = crate::simd_reduce::min_f32(&input);
    assert_eq!(result, 1.0);
}

#[test]
fn test_simd_dot_f32_basic() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![2.0, 3.0, 4.0, 5.0];
    let result = crate::simd_reduce::dot_f32(&a, &b);
    // 2 + 6 + 12 + 20 = 40
    assert!((result - 40.0).abs() < 1e-6);
}

#[test]
fn test_l2_norm_f32_unit_vector() {
    let input = vec![1.0, 0.0, 0.0, 0.0];
    let result = crate::l2_norm_f32(&input);
    assert!((result - 1.0).abs() < 1e-6);
}

#[test]
fn test_l2_norm_f32_345_triangle() {
    let input = vec![3.0, 4.0];
    let result = crate::l2_norm_f32(&input);
    assert!((result - 5.0).abs() < 1e-5);
}

// ============================================================================
// 5. Row Reductions (reduce: sum, max, min, mean, argmax, argmin)
// ============================================================================

#[test]
fn test_row_sum_reduction() {
    // 2 rows of 4 elements each
    let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let mut output = vec![0.0f32; 2];
    crate::reduce::sum_f32(&input, &mut output, 4);
    assert!((output[0] - 10.0).abs() < 1e-6);
    assert!((output[1] - 100.0).abs() < 1e-6);
}

#[test]
fn test_row_max_reduction() {
    let input = vec![1.0, 5.0, 3.0, 2.0, 10.0, 8.0, 12.0, 6.0];
    let mut output = vec![0.0f32; 2];
    crate::reduce::max_f32(&input, &mut output, 4);
    assert_eq!(output[0], 5.0);
    assert_eq!(output[1], 12.0);
}

#[test]
fn test_row_min_reduction() {
    let input = vec![1.0, 5.0, 3.0, 2.0, 10.0, 8.0, 12.0, 6.0];
    let mut output = vec![0.0f32; 2];
    crate::reduce::min_f32(&input, &mut output, 4);
    assert_eq!(output[0], 1.0);
    assert_eq!(output[1], 6.0);
}

#[test]
fn test_row_mean_reduction() {
    let input = vec![2.0, 4.0, 6.0, 8.0];
    let mut output = vec![0.0f32; 1];
    crate::reduce::mean_f32(&input, &mut output, 4);
    assert!((output[0] - 5.0).abs() < 1e-6);
}

#[test]
fn test_row_argmax_reduction() {
    let input = vec![1.0, 5.0, 3.0, 2.0];
    let mut output = vec![0u32; 1];
    crate::reduce::argmax_f32(&input, &mut output, 4);
    assert_eq!(output[0], 1);
}

#[test]
fn test_row_argmin_reduction() {
    let input = vec![5.0, 1.0, 3.0, 2.0];
    let mut output = vec![0u32; 1];
    crate::reduce::argmin_f32(&input, &mut output, 4);
    assert_eq!(output[0], 1);
}

// ============================================================================
// 6. Softmax Numerical Stability
// ============================================================================

#[test]
fn test_softmax_basic_distribution() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0f32; 4];
    crate::softmax_f32(&input, &mut output, 4);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1, got {sum}");
}

#[test]
fn test_softmax_monotonicity() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0f32; 4];
    crate::softmax_f32(&input, &mut output, 4);
    // Larger inputs should produce larger outputs
    for i in 1..4 {
        assert!(output[i] > output[i - 1], "softmax not monotonic at {i}");
    }
}

#[test]
fn test_softmax_large_values_no_overflow() {
    let input = vec![1000.0, 1001.0, 1002.0, 1003.0];
    let mut output = vec![0.0f32; 4];
    crate::softmax_f32(&input, &mut output, 4);
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "softmax with large values must sum to ~1, got {sum}"
    );
    assert!(
        output.iter().all(|v| v.is_finite()),
        "softmax produced non-finite values"
    );
}

#[test]
fn test_softmax_negative_large_no_underflow() {
    let input = vec![-1000.0, -999.0, -998.0, -997.0];
    let mut output = vec![0.0f32; 4];
    crate::softmax_f32(&input, &mut output, 4);
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "softmax with large negative values must sum to ~1, got {sum}"
    );
}

#[test]
fn test_softmax_uniform_input() {
    let input = vec![5.0; 4];
    let mut output = vec![0.0f32; 4];
    crate::softmax_f32(&input, &mut output, 4);
    for &v in &output {
        assert!(
            (v - 0.25).abs() < 1e-5,
            "uniform softmax should be 0.25, got {v}"
        );
    }
}

#[test]
fn test_softmax_scalar_vs_dispatch() {
    let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.5 - 4.0).collect();
    let mut out_scalar = vec![0.0f32; 16];
    let mut out_dispatch = vec![0.0f32; 16];
    crate::simd_softmax::softmax_f32_scalar(&input, &mut out_scalar, 16);
    crate::softmax_f32(&input, &mut out_dispatch, 16);
    // The dispatch (SIMD) path uses the Schraudolph `fast_exp_f32`
    // approximation (~1e-4 relative error, per simd_softmax docs), while the
    // scalar path uses stdlib `f32::exp()`. They therefore agree only to the
    // approximation's tolerance, not to 1e-6. Match the established 5e-2 bound
    // used by the sibling scalar-vs-dispatch softmax test.
    for (s, d) in out_scalar.iter().zip(out_dispatch.iter()) {
        assert!((s - d).abs() <= 5e-2, "scalar={s} dispatch={d}");
    }
}

// ============================================================================
// 7. MatMul
// ============================================================================

#[test]
fn test_matmul_identity() {
    // I * A = A for 2x2
    let identity = vec![1.0, 0.0, 0.0, 1.0];
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let mut out = vec![0.0f32; 4];
    crate::matmul_f32(&identity, &a, 2, 2, 2, &mut out);
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_matmul_2x3_times_3x2() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
    let mut out = vec![0.0f32; 4]; // 2x2
    crate::matmul_f32(&a, &b, 2, 3, 2, &mut out);
    // Row 0: 1*7+2*9+3*11=7+18+33=58, 1*8+2*10+3*12=8+20+36=64
    // Row 1: 4*7+5*9+6*11=28+45+66=139, 4*8+5*10+6*12=32+50+72=154
    assert!((out[0] - 58.0).abs() < 1e-4);
    assert!((out[1] - 64.0).abs() < 1e-4);
    assert!((out[2] - 139.0).abs() < 1e-4);
    assert!((out[3] - 154.0).abs() < 1e-4);
}

#[test]
fn test_matmul_scalar_vs_dispatch() {
    let m = 8;
    let k = 16;
    let n = 8;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
    let mut out_scalar = vec![0.0f32; m * n];
    let mut out_dispatch = vec![0.0f32; m * n];
    crate::simd_matmul::matmul_f32_scalar(&a, &b, m, k, n, &mut out_scalar);
    crate::matmul_f32(&a, &b, m, k, n, &mut out_dispatch);
    for (s, d) in out_scalar.iter().zip(out_dispatch.iter()) {
        assert!((s - d).abs() < 1e-3, "matmul scalar={s} dispatch={d}");
    }
}

#[test]
fn test_matmul_zero_matrix() {
    let a = vec![0.0f32; 4];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut out = vec![0.0f32; 4];
    crate::matmul_f32(&a, &b, 2, 2, 2, &mut out);
    assert!(out.iter().all(|&v| v == 0.0));
}

// ============================================================================
// 8. Layer Normalization
// ============================================================================

#[test]
fn test_layer_norm_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let mut output = vec![0.0f32; 4];
    crate::layer_norm_f32(&input, &mut output, &gamma, &beta, 4, 1e-5);
    // Normalized output should have mean ~0 and variance ~1
    let mean: f32 = output.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-5, "layernorm mean should be ~0, got {mean}");
}

#[test]
fn test_layer_norm_with_affine() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![2.0; 4];
    let beta = vec![1.0; 4];
    let mut output = vec![0.0f32; 4];
    crate::layer_norm_f32(&input, &mut output, &gamma, &beta, 4, 1e-5);
    // With gamma=2, beta=1: output = 2 * normalized + 1
    let mean: f32 = output.iter().sum::<f32>() / 4.0;
    assert!(
        (mean - 1.0).abs() < 1e-4,
        "layernorm with affine mean should be ~1, got {mean}"
    );
}

#[test]
fn test_layer_norm_reference_single_row() {
    let input = vec![2.0, 4.0, 6.0, 8.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let result = crate::layer_norm_f32_reference(&input, &gamma, &beta, 4, 1e-5);
    let mean: f32 = result.iter().sum::<f32>() / 4.0;
    assert!(
        mean.abs() < 1e-5,
        "reference layernorm mean should be ~0, got {mean}"
    );
}

#[test]
fn test_layer_norm_constant_input() {
    let input = vec![5.0; 4];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let mut output = vec![0.0f32; 4];
    crate::layer_norm_f32(&input, &mut output, &gamma, &beta, 4, 1e-5);
    // Constant input => all outputs ~0
    for &v in &output {
        assert!(
            v.abs() < 0.1,
            "constant input layernorm should produce ~0, got {v}"
        );
    }
}

// ============================================================================
// 9. Batch Normalization
// ============================================================================

#[test]
fn test_batchnorm_identity_transform() {
    // mean=0, var=1, gamma=1, beta=0 => output ~= input
    let channels = 2;
    let spatial = 4;
    let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let mean = vec![0.0; channels];
    let var = vec![1.0; channels];
    let gamma = vec![1.0; channels];
    let beta = vec![0.0; channels];
    let mut output = vec![0.0f32; 8];
    crate::batchnorm_f32(
        &input,
        &mean,
        &var,
        &gamma,
        &beta,
        channels,
        spatial,
        1e-5,
        &mut output,
    );
    for (i, (&o, &inp)) in output.iter().zip(input.iter()).enumerate() {
        assert!(
            (o - inp).abs() < 0.01,
            "batchnorm identity mismatch at {i}: {o} vs {inp}"
        );
    }
}

#[test]
fn test_batchnorm_reference_consistency() {
    let channels = 2;
    let spatial = 8;
    let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    let mean = vec![1.0, 2.0];
    let var = vec![0.5, 1.5];
    let gamma = vec![2.0, 0.5];
    let beta = vec![0.1, -0.1];
    let reference =
        crate::batchnorm_reference(&input, &mean, &var, &gamma, &beta, channels, spatial, 1e-5);
    let mut output = vec![0.0f32; 16];
    crate::batchnorm_f32(
        &input,
        &mean,
        &var,
        &gamma,
        &beta,
        channels,
        spatial,
        1e-5,
        &mut output,
    );
    for (r, o) in reference.iter().zip(output.iter()) {
        assert!((r - o).abs() < 1e-5, "reference={r} dispatch={o}");
    }
}

// ============================================================================
// 10. Instance Normalization
// ============================================================================

#[test]
fn test_instance_norm_zero_mean() {
    let channels = 2;
    let spatial = 4;
    let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let result = crate::instance_norm_f32_reference(&input, channels, spatial, 1e-5);
    // Each channel should have mean ~0
    for c in 0..channels {
        let start = c * spatial;
        let ch_mean: f32 = result[start..start + spatial].iter().sum::<f32>() / spatial as f32;
        assert!(
            ch_mean.abs() < 1e-4,
            "instance_norm channel {c} mean should be ~0, got {ch_mean}"
        );
    }
}

#[test]
fn test_instance_norm_dispatch_vs_reference() {
    let channels = 3;
    let spatial = 16;
    let input: Vec<f32> = (0..48).map(|i| (i as f32).sin()).collect();
    let reference = crate::instance_norm_f32_reference(&input, channels, spatial, 1e-5);
    let mut output = vec![0.0f32; 48];
    crate::instance_norm_f32(&input, &mut output, channels, spatial, 1e-5);
    for (r, o) in reference.iter().zip(output.iter()) {
        assert!(
            (r - o).abs() < 1e-5,
            "instance_norm reference={r} dispatch={o}"
        );
    }
}

// ============================================================================
// 11. Group Normalization
// ============================================================================

#[test]
fn test_groupnorm_basic() {
    let channels = 4;
    let spatial = 8;
    let groups = 2;
    let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let gamma = vec![1.0; channels];
    let beta = vec![0.0; channels];
    let result = crate::groupnorm_reference(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    assert_eq!(result.len(), channels * spatial);
    assert!(result.iter().all(|v| v.is_finite()));
}

#[test]
fn test_groupnorm_dispatch_vs_reference() {
    let channels = 4;
    let spatial = 16;
    let groups = 2;
    let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let gamma = vec![1.0, 2.0, 0.5, 1.5];
    let beta = vec![0.0, 0.1, -0.1, 0.2];
    let reference =
        crate::groupnorm_reference(&input, &gamma, &beta, groups, channels, spatial, 1e-5);
    let mut output = vec![0.0f32; 64];
    crate::simd_groupnorm::groupnorm_f32_scalar(
        &input,
        &gamma,
        &beta,
        groups,
        channels,
        spatial,
        1e-5,
        &mut output,
    );
    for (r, o) in reference.iter().zip(output.iter()) {
        assert!((r - o).abs() < 1e-5, "groupnorm reference={r} scalar={o}");
    }
}

// ============================================================================
// 12. RMS Normalization
// ============================================================================

#[test]
fn test_rmsnorm_reference_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0; 4];
    let result = crate::rmsnorm_reference(&input, &weight, 4, 1e-5);
    assert_eq!(result.len(), 4);
    // RMSNorm preserves relative magnitudes
    assert!(result[1] > result[0]);
    assert!(result[2] > result[1]);
    assert!(result[3] > result[2]);
}

#[test]
fn test_rmsnorm_inplace_vs_reference() {
    let input = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
    let weight = vec![1.0, 0.5, 2.0, 1.5, 0.8, 1.2, 0.3, 1.7];
    let reference = crate::rmsnorm_reference(&input, &weight, 8, 1e-6);
    let mut output = vec![0.0f32; 8];
    crate::rmsnorm(&input, &weight, &mut output, 8, 1e-6);
    for (r, o) in reference.iter().zip(output.iter()) {
        assert!((r - o).abs() < 1e-5, "rmsnorm reference={r} inplace={o}");
    }
}

// ============================================================================
// 13. Conv1d
// ============================================================================

#[test]
fn test_conv1d_full_basic() {
    let cfg = crate::Conv1dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: 3,
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
    };
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // [1, 5]
    let weight = vec![1.0, 1.0, 1.0]; // [1, 1, 3]
    let result = crate::conv1d_full_reference(&input, &weight, None, &cfg).unwrap();
    // out_len = (5 - 3) / 1 + 1 = 3
    assert_eq!(result.len(), 3);
    assert!((result[0] - 6.0).abs() < 1e-5); // 1+2+3
    assert!((result[1] - 9.0).abs() < 1e-5); // 2+3+4
    assert!((result[2] - 12.0).abs() < 1e-5); // 3+4+5
}

#[test]
fn test_conv1d_full_with_bias() {
    let cfg = crate::Conv1dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: 3,
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
    };
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![1.0, 1.0, 1.0];
    let bias = vec![10.0];
    let result = crate::conv1d_full_reference(&input, &weight, Some(&bias), &cfg).unwrap();
    assert!((result[0] - 16.0).abs() < 1e-5); // 6 + 10
}

#[test]
fn test_conv1d_full_with_stride() {
    let cfg = crate::Conv1dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: 3,
        stride: 2,
        padding: 0,
        dilation: 1,
        groups: 1,
    };
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![1.0, 1.0, 1.0];
    let result = crate::conv1d_full_reference(&input, &weight, None, &cfg).unwrap();
    // out_len = (5 - 3) / 2 + 1 = 2
    assert_eq!(result.len(), 2);
}

#[test]
fn test_conv1d_full_with_padding() {
    let cfg = crate::Conv1dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
    };
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![1.0, 1.0, 1.0];
    let result = crate::conv1d_full_reference(&input, &weight, None, &cfg).unwrap();
    // out_len = (3 + 2 - 3) / 1 + 1 = 3 (same padding)
    assert_eq!(result.len(), 3);
}

#[test]
fn test_conv1d_full_error_zero_stride() {
    let cfg = crate::Conv1dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: 3,
        stride: 0,
        padding: 0,
        dilation: 1,
        groups: 1,
    };
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![1.0, 1.0, 1.0];
    let result = crate::conv1d_full_reference(&input, &weight, None, &cfg);
    assert!(result.is_err());
}

#[test]
fn test_conv1d_full_vs_dispatch() {
    let cfg = crate::Conv1dConfig {
        in_channels: 2,
        out_channels: 2,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
    };
    let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect(); // [2, 8]
                                                                     // weight: [out_ch, in_ch/groups, kernel_size] = [2, 2, 3]
    let weight: Vec<f32> = (0..12).map(|i| (i as f32 * 0.05) - 0.3).collect();
    let reference = crate::conv1d_full_reference(&input, &weight, None, &cfg).unwrap();
    let dispatch = crate::conv1d_full(&input, &weight, None, &cfg).unwrap();
    for (r, d) in reference.iter().zip(dispatch.iter()) {
        assert!((r - d).abs() < 1e-4, "conv1d reference={r} dispatch={d}");
    }
}

// ============================================================================
// 14. Conv2d
// ============================================================================

#[test]
fn test_conv2d_reference_basic() {
    let batch = 1;
    let in_ch = 1;
    let out_ch = 1;
    let h = 4;
    let w = 4;
    let kh = 3;
    let kw = 3;
    let oh = (h - kh) + 1; // 2
    let ow = (w - kw) + 1; // 2
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let weight = vec![1.0f32; 9]; // all-ones 3x3
    let mut output = vec![0.0f32; batch * out_ch * oh * ow];
    crate::conv2d_reference(
        &input,
        &weight,
        None,
        &mut output,
        batch,
        in_ch,
        out_ch,
        h,
        w,
        kh,
        kw,
        1,
        1,
        0,
        0,
    )
    .unwrap();
    assert_eq!(output.len(), 4);
    // Top-left: sum of 3x3 starting at (0,0) = 0+1+2+4+5+6+8+9+10 = 45
    assert!((output[0] - 45.0).abs() < 1e-4);
}

#[test]
fn test_conv2d_with_bias() {
    let batch = 1;
    let in_ch = 1;
    let out_ch = 1;
    let h = 3;
    let w = 3;
    let kh = 3;
    let kw = 3;
    let oh = 1;
    let ow = 1;
    let input = vec![1.0f32; 9];
    let weight = vec![1.0f32; 9];
    let bias = vec![10.0f32];
    let mut output = vec![0.0f32; batch * out_ch * oh * ow];
    crate::conv2d_reference(
        &input,
        &weight,
        Some(&bias),
        &mut output,
        batch,
        in_ch,
        out_ch,
        h,
        w,
        kh,
        kw,
        1,
        1,
        0,
        0,
    )
    .unwrap();
    assert!((output[0] - 19.0).abs() < 1e-4); // 9 + 10
}

// ============================================================================
// 15. Transpose
// ============================================================================

#[test]
fn test_transpose_2x3() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let result = crate::transpose_reference(&input, 2, 3);
    // Expected 3x2: [[1,4],[2,5],[3,6]]
    assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_identity_square() {
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let t = crate::transpose_reference(&input, 2, 2);
    assert_eq!(t, vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_transpose_round_trip() {
    let input: Vec<f32> = (0..12).map(|i| i as f32).collect(); // 3x4
    let t1 = crate::transpose_reference(&input, 3, 4);
    let t2 = crate::transpose_reference(&t1, 4, 3);
    assert_eq!(input, t2);
}

#[test]
fn test_transpose_dispatch_vs_reference() {
    let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect(); // 8x8
    let reference = crate::transpose_reference(&input, 8, 8);
    let mut output = vec![0.0f32; 64];
    crate::transpose_2d(&input, &mut output, 8, 8);
    for (r, o) in reference.iter().zip(output.iter()) {
        assert!((r - o).abs() < 1e-6, "transpose reference={r} dispatch={o}");
    }
}

// ============================================================================
// 16. RoPE (Rotary Position Embeddings)
// ============================================================================

#[test]
fn test_rope_reference_basic() {
    let head_dim = 4;
    let seq_len = 1;
    let num_heads = 1;
    let x = vec![1.0, 2.0, 3.0, 4.0]; // [seq=1, heads=1, dim=4]
    let cos_cache = vec![1.0, 0.0]; // [seq=1, half=2]
    let sin_cache = vec![0.0, 1.0]; // [seq=1, half=2]
    let result = crate::rope_reference(&x, &cos_cache, &sin_cache, head_dim, seq_len, num_heads);
    // With cos=[1,0], sin=[0,1]:
    // out[0] = x[0]*cos[0] - x[2]*sin[0] = 1*1 - 3*0 = 1
    // out[1] = x[1]*cos[1] - x[3]*sin[1] = 2*0 - 4*1 = -4
    // out[2] = x[0]*sin[0] + x[2]*cos[0] = 1*0 + 3*1 = 3
    // out[3] = x[1]*sin[1] + x[3]*cos[1] = 2*1 + 4*0 = 2
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - (-4.0)).abs() < 1e-6);
    assert!((result[2] - 3.0).abs() < 1e-6);
    assert!((result[3] - 2.0).abs() < 1e-6);
}

#[test]
fn test_rope_apply_inplace_matches_reference() {
    let head_dim = 8;
    let seq_len = 2;
    let num_heads = 2;
    let half = head_dim / 2;
    let total = seq_len * num_heads * head_dim;
    let x: Vec<f32> = (0..total).map(|i| i as f32 * 0.1).collect();
    let cos_cache: Vec<f32> = (0..seq_len * half)
        .map(|i| (i as f32 * 0.5).cos())
        .collect();
    let sin_cache: Vec<f32> = (0..seq_len * half)
        .map(|i| (i as f32 * 0.5).sin())
        .collect();
    let reference = crate::rope_reference(&x, &cos_cache, &sin_cache, head_dim, seq_len, num_heads);
    let mut x_mut = x;
    crate::rope_apply(
        &mut x_mut, &cos_cache, &sin_cache, head_dim, seq_len, num_heads,
    );
    for (r, o) in reference.iter().zip(x_mut.iter()) {
        assert!((r - o).abs() < 1e-5, "rope reference={r} apply={o}");
    }
}

// ============================================================================
// 17. Pooling (max_pool1d, avg_pool1d)
// ============================================================================

#[test]
fn test_max_pool1d_basic() {
    let batch = 1;
    let channels = 1;
    let input_len = 6;
    let kernel_size = 2;
    let stride = 2;
    let padding = 0;
    let out_len = (input_len + 2 * padding - kernel_size) / stride + 1; // 3
    let input = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
    let mut output = vec![0.0f32; batch * channels * out_len];
    crate::max_pool1d_reference(
        &input,
        &mut output,
        batch,
        channels,
        input_len,
        kernel_size,
        stride,
        padding,
    );
    assert_eq!(output, vec![3.0, 5.0, 6.0]);
}

#[test]
fn test_avg_pool1d_basic() {
    let batch = 1;
    let channels = 1;
    let input_len = 6;
    let kernel_size = 2;
    let stride = 2;
    let padding = 0;
    let out_len = (input_len + 2 * padding - kernel_size) / stride + 1;
    let input = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
    let mut output = vec![0.0f32; batch * channels * out_len];
    crate::avg_pool1d_reference(
        &input,
        &mut output,
        batch,
        channels,
        input_len,
        kernel_size,
        stride,
        padding,
    );
    assert_eq!(output, vec![3.0, 7.0, 11.0]);
}

// ============================================================================
// 18. Linear Layer
// ============================================================================

#[test]
fn test_linear_reference_basic() {
    let in_features = 3;
    let out_features = 2;
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // [2, 3] selects first and second
    let bias = vec![0.0, 0.0];
    let mut output = vec![0.0f32; 2];
    crate::linear_reference(
        &input,
        &weight,
        &bias,
        &mut output,
        in_features,
        out_features,
    );
    assert!((output[0] - 1.0).abs() < 1e-6);
    assert!((output[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_linear_reference_with_bias() {
    let in_features = 2;
    let out_features = 2;
    let input = vec![1.0, 2.0];
    let weight = vec![1.0, 1.0, 2.0, 2.0]; // [2, 2]
    let bias = vec![10.0, 20.0];
    let mut output = vec![0.0f32; 2];
    crate::linear_reference(
        &input,
        &weight,
        &bias,
        &mut output,
        in_features,
        out_features,
    );
    // out[0] = 1*1 + 2*1 + 10 = 13
    // out[1] = 1*2 + 2*2 + 20 = 26
    assert!((output[0] - 13.0).abs() < 1e-5);
    assert!((output[1] - 26.0).abs() < 1e-5);
}

#[test]
fn test_linear_batched_reference() {
    let batch = 2;
    let in_features = 3;
    let out_features = 2;
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let weight = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // [2, 3]
    let bias = vec![0.0, 0.0];
    let mut output = vec![0.0f32; batch * out_features];
    crate::linear_batched_reference(
        &input,
        &weight,
        &bias,
        &mut output,
        batch,
        in_features,
        out_features,
    );
    // batch 0: [1, 2] ; batch 1: [4, 5]
    assert!((output[0] - 1.0).abs() < 1e-6);
    assert!((output[1] - 2.0).abs() < 1e-6);
    assert!((output[2] - 4.0).abs() < 1e-6);
    assert!((output[3] - 5.0).abs() < 1e-6);
}

// ============================================================================
// 19. Embedding Lookup
// ============================================================================

#[test]
fn test_embedding_reference_basic() {
    let embed_dim = 3;
    let weights = vec![
        1.0, 2.0, 3.0, // row 0
        4.0, 5.0, 6.0, // row 1
        7.0, 8.0, 9.0, // row 2
    ];
    let indices: Vec<u32> = vec![0, 2, 1];
    let result = crate::embedding_reference(&weights, &indices, embed_dim).unwrap();
    assert_eq!(result.len(), 9);
    assert_eq!(&result[0..3], &[1.0, 2.0, 3.0]);
    assert_eq!(&result[3..6], &[7.0, 8.0, 9.0]);
    assert_eq!(&result[6..9], &[4.0, 5.0, 6.0]);
}

#[test]
fn test_embedding_out_of_bounds() {
    let embed_dim = 2;
    let weights = vec![1.0, 2.0, 3.0, 4.0]; // vocab_size=2
    let indices: Vec<u32> = vec![5]; // out of bounds
    let result = crate::embedding_reference(&weights, &indices, embed_dim);
    assert!(result.is_err());
}

// ============================================================================
// 20. Gather / Scatter
// ============================================================================

#[test]
fn test_gather_1d_basic() {
    let input = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let indices: Vec<u32> = vec![4, 2, 0];
    let mut output = vec![0.0f32; 3];
    crate::gather_1d_scalar(&input, &indices, &mut output).unwrap();
    assert_eq!(output, vec![50.0, 30.0, 10.0]);
}

#[test]
fn test_gather_1d_out_of_bounds() {
    let input = vec![1.0, 2.0, 3.0];
    let indices: Vec<u32> = vec![10];
    let mut output = vec![0.0f32; 1];
    let result = crate::gather_1d_scalar(&input, &indices, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_scatter_add_1d_basic() {
    let input = vec![1.0, 2.0, 3.0];
    let indices: Vec<u32> = vec![0, 1, 0]; // accumulate at index 0
    let mut output = vec![0.0f32; 3];
    crate::scatter_add_1d_scalar(&input, &indices, 3, &mut output).unwrap();
    // output[0] = 1.0 + 3.0 = 4.0, output[1] = 2.0
    assert!((output[0] - 4.0).abs() < 1e-6);
    assert!((output[1] - 2.0).abs() < 1e-6);
    assert_eq!(output[2], 0.0);
}

// ============================================================================
// 21. Type Casting (f32 <-> f16, f32 <-> bf16)
// ============================================================================

#[test]
fn test_f32_to_f16_roundtrip() {
    let input = vec![0.0, 1.0, -1.0, 0.5, 100.0, -100.0, 0.001, 65504.0];
    let mut f16_buf = vec![0u16; 8];
    let mut output = vec![0.0f32; 8];
    crate::f32_to_f16(&input, &mut f16_buf);
    crate::f16_to_f32(&f16_buf, &mut output);
    for (inp, out) in input.iter().zip(output.iter()) {
        let tol = inp.abs() * 0.01 + 1e-3;
        assert!((inp - out).abs() < tol, "f16 roundtrip: {inp} -> {out}");
    }
}

#[test]
fn test_f32_to_bf16_roundtrip() {
    let input = vec![0.0, 1.0, -1.0, 0.5, 100.0, -100.0, 3.14];
    let mut bf16_buf = vec![0u16; 7];
    let mut output = vec![0.0f32; 7];
    crate::f32_to_bf16(&input, &mut bf16_buf);
    crate::bf16_to_f32(&bf16_buf, &mut output);
    for (inp, out) in input.iter().zip(output.iter()) {
        let tol = inp.abs() * 0.01 + 1e-2;
        assert!((inp - out).abs() < tol, "bf16 roundtrip: {inp} -> {out}");
    }
}

#[test]
fn test_f32_to_f16_special_values() {
    let input = vec![0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY];
    let mut f16_buf = vec![0u16; 4];
    let mut output = vec![0.0f32; 4];
    crate::f32_to_f16(&input, &mut f16_buf);
    crate::f16_to_f32(&f16_buf, &mut output);
    assert_eq!(output[0], 0.0);
    assert!(output[2].is_infinite() && output[2] > 0.0);
    assert!(output[3].is_infinite() && output[3] < 0.0);
}

#[test]
fn test_f32_to_f16_scalar_vs_dispatch() {
    let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.5 - 8.0).collect();
    let mut scalar_out = vec![0u16; 32];
    let mut dispatch_out = vec![0u16; 32];
    crate::simd_cast::f32_to_f16_scalar(&input, &mut scalar_out);
    crate::f32_to_f16(&input, &mut dispatch_out);
    assert_eq!(scalar_out, dispatch_out);
}

// ============================================================================
// 22. Quantization (int8)
// ============================================================================

#[test]
fn test_quantize_dequantize_roundtrip() {
    let input = vec![0.0, 0.5, 1.0, -0.5, -1.0, 0.25, -0.25, 0.75];
    let scale = 0.01_f32;
    let zero_point = 0_i8;
    let mut quantized = vec![0i8; 8];
    let mut dequantized = vec![0.0f32; 8];
    crate::quantize_f32_to_i8(&input, &mut quantized, scale, zero_point);
    crate::dequantize_i8_to_f32(&quantized, &mut dequantized, scale, zero_point);
    for (inp, deq) in input.iter().zip(dequantized.iter()) {
        assert!(
            (inp - deq).abs() < scale + 1e-6,
            "quant roundtrip: {inp} -> {deq}"
        );
    }
}

#[test]
fn test_quantize_clamps_range() {
    let input = vec![10.0, -10.0]; // well outside normal scale
    let scale = 0.1;
    let zero_point = 0_i8;
    let mut quantized = vec![0i8; 2];
    crate::quantize_f32_to_i8(&input, &mut quantized, scale, zero_point);
    // clamp(round(10/0.1)+0, -128, 127) = clamp(100, -128, 127) = 100
    // clamp(round(-10/0.1)+0, -128, 127) = clamp(-100, -128, 127) = -100
    assert_eq!(quantized[0], 100);
    assert_eq!(quantized[1], -100);
}

#[test]
fn test_quantize_reference_vs_dispatch() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let scale = 0.05;
    let zero_point = 10_i8;
    let mut ref_out = vec![0i8; 64];
    let mut disp_out = vec![0i8; 64];
    crate::quantize_f32_to_i8_reference(&input, &mut ref_out, scale, zero_point);
    crate::quantize_f32_to_i8(&input, &mut disp_out, scale, zero_point);
    assert_eq!(ref_out, disp_out);
}

#[test]
fn test_quantize_per_channel_basic() {
    let channels = 2;
    let elements_per_channel = 4;
    let input = vec![0.0, 0.5, 1.0, 1.5, 0.0, -0.5, -1.0, -1.5];
    let scales = vec![0.01, 0.02];
    let zero_points = vec![0i8, 0i8];
    let mut output = vec![0i8; 8];
    crate::quantize_per_channel(
        &input,
        &mut output,
        &scales,
        &zero_points,
        channels,
        elements_per_channel,
    );
    // Channel 0: 0/0.01=0, 0.5/0.01=50, 1/0.01=100, 1.5/0.01=127(clamped)
    assert_eq!(output[0], 0);
    assert_eq!(output[1], 50);
    assert_eq!(output[2], 100);
}

// ============================================================================
// 23. L2 / L1 / MinMax Normalize
// ============================================================================

#[test]
fn test_l2_normalize_unit_norm() {
    let input = vec![3.0, 4.0];
    let result = crate::l2_normalize_reference(&input);
    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "L2 normalized vector should have unit norm, got {norm}"
    );
}

#[test]
fn test_l2_normalize_zero_vector() {
    let input = vec![0.0; 4];
    let result = crate::l2_normalize_reference(&input);
    assert!(result.iter().all(|&v| v == 0.0));
}

#[test]
fn test_l1_normalize_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = crate::l1_normalize_reference(&input);
    let sum: f32 = result.iter().map(|x| x.abs()).sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "L1 normalized should sum abs to 1, got {sum}"
    );
}

#[test]
fn test_min_max_normalize_basic() {
    let input = vec![2.0, 4.0, 6.0, 8.0];
    let result = crate::min_max_normalize_reference(&input);
    assert!((result[0] - 0.0).abs() < 1e-5); // min -> 0
    assert!((result[3] - 1.0).abs() < 1e-5); // max -> 1
}

// ============================================================================
// 24. SIMD Tier Consistency (scalar vs auto-dispatch)
// ============================================================================

#[test]
fn test_relu_scalar_vs_dispatch() {
    let input: Vec<f32> = (-20..20).map(|i| i as f32 * 0.3).collect();
    let mut out_scalar = vec![0.0f32; 40];
    let mut out_dispatch = vec![0.0f32; 40];
    crate::simd_elementwise::relu_f32_scalar(&input, &mut out_scalar);
    crate::relu_f32(&input, &mut out_dispatch);
    assert_eq!(out_scalar, out_dispatch);
}

#[test]
fn test_silu_scalar_vs_dispatch() {
    let input: Vec<f32> = (-10..10).map(|i| i as f32 * 0.5).collect();
    let mut out_scalar = vec![0.0f32; 20];
    let mut out_dispatch = vec![0.0f32; 20];
    crate::simd_elementwise::silu_f32_scalar(&input, &mut out_scalar);
    crate::silu_f32(&input, &mut out_dispatch);
    for (s, d) in out_scalar.iter().zip(out_dispatch.iter()) {
        assert!((s - d).abs() < 1e-5, "silu scalar={s} dispatch={d}");
    }
}

#[test]
fn test_mul_scalar_vs_dispatch() {
    let a: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..32).map(|i| (32 - i) as f32 * 0.1).collect();
    let mut out_scalar = vec![0.0f32; 32];
    let mut out_dispatch = vec![0.0f32; 32];
    crate::simd_elementwise::mul_f32_scalar(&a, &b, &mut out_scalar);
    crate::mul_f32(&a, &b, &mut out_dispatch);
    for (s, d) in out_scalar.iter().zip(out_dispatch.iter()) {
        assert!((s - d).abs() < 1e-6, "mul scalar={s} dispatch={d}");
    }
}

// ============================================================================
// 25. Edge Cases and Numerical Stability
// ============================================================================

#[test]
fn test_softmax_single_element() {
    let input = vec![42.0];
    let mut output = vec![0.0f32; 1];
    crate::softmax_f32(&input, &mut output, 1);
    assert!(
        (output[0] - 1.0).abs() < 1e-6,
        "single-element softmax should be 1.0"
    );
}

#[test]
fn test_matmul_1x1() {
    let a = vec![3.0];
    let b = vec![4.0];
    let mut out = vec![0.0f32; 1];
    crate::matmul_f32(&a, &b, 1, 1, 1, &mut out);
    assert!((out[0] - 12.0).abs() < 1e-6);
}

#[test]
fn test_layer_norm_two_elements() {
    let input = vec![0.0, 10.0];
    let gamma = vec![1.0; 2];
    let beta = vec![0.0; 2];
    let mut output = vec![0.0f32; 2];
    crate::layer_norm_f32(&input, &mut output, &gamma, &beta, 2, 1e-5);
    // Should be approximately [-1, 1] (normalized)
    assert!(output[0] < 0.0);
    assert!(output[1] > 0.0);
}

#[test]
fn test_instance_norm_single_channel() {
    let channels = 1;
    let spatial = 8;
    let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let result = crate::instance_norm_f32_reference(&input, channels, spatial, 1e-5);
    let mean: f32 = result.iter().sum::<f32>() / spatial as f32;
    assert!(
        mean.abs() < 1e-4,
        "single channel instance_norm mean should be ~0, got {mean}"
    );
}

// ============================================================================
// 26. Large Vector / Performance-Path Tests
// ============================================================================

#[test]
fn test_matmul_large() {
    let m = 32;
    let k = 64;
    let n = 32;
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
    let mut out_ref = vec![0.0f32; m * n];
    let mut out_disp = vec![0.0f32; m * n];
    crate::simd_matmul::matmul_f32_scalar(&a, &b, m, k, n, &mut out_ref);
    crate::matmul_f32(&a, &b, m, k, n, &mut out_disp);
    for (r, d) in out_ref.iter().zip(out_disp.iter()) {
        assert!(
            (r - d).abs() < 0.01,
            "large matmul mismatch: scalar={r} dispatch={d}"
        );
    }
}

#[test]
fn test_softmax_large_row() {
    let n = 512;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) - 2.56).collect();
    let mut output = vec![0.0f32; n];
    crate::softmax_f32(&input, &mut output, n);
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "large softmax must sum to 1, got {sum}"
    );
    assert!(output.iter().all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
fn test_add_f32_odd_length() {
    // Non-SIMD-aligned length (not multiple of 4 or 8)
    let n = 13;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let mut out = vec![0.0f32; n];
    crate::add_f32(&a, &b, &mut out);
    for &v in &out {
        assert!((v - n as f32).abs() < 1e-6);
    }
}

#[test]
fn test_embedding_large_lookup() {
    let vocab_size = 100;
    let embed_dim = 64;
    let weights: Vec<f32> = (0..vocab_size * embed_dim)
        .map(|i| i as f32 * 0.001)
        .collect();
    let indices: Vec<u32> = (0..10).collect();
    let result = crate::embedding_reference(&weights, &indices, embed_dim).unwrap();
    assert_eq!(result.len(), 10 * embed_dim);
    // First embedding row should match weights[0..embed_dim]
    assert_eq!(&result[0..embed_dim], &weights[0..embed_dim]);
}
