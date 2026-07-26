// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended proc-macro expansion tests for `#[nn_macros::kernel]`.
//!
//! Complements `kernel_expansion.rs` with additional coverage:
//! - Kernels using `exp()`, `.min()`, subtraction, negation
//! - Complex multi-step let-binding chains
//! - Constants and literal propagation
//! - Bounds attribute with multiple parameters
//! - Descriptor metadata validation for multi-param kernels
//! - MSL structural checks (thread_position_in_grid, kernel attribute)
//! - Numerical edge-case sweeps

#![allow(unexpected_cfgs)]

use nn_dsl::KernelOps;

// ---------------------------------------------------------------------------
// Kernel definitions
// ---------------------------------------------------------------------------

/// Exponential kernel — tests `exp()` lowering.
#[nn_macros::kernel]
fn exp_kernel(x: f32) -> f32 {
    x.exp()
}

/// Sigmoid kernel — tests compound expression with exp + division.
#[nn_macros::kernel]
fn sigmoid_kern(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Leaky ReLU kernel — tests min/max combination.
#[nn_macros::kernel]
fn leaky_relu(x: f32) -> f32 {
    x.max(0.0) + 0.01 * x.min(0.0)
}

/// Softplus kernel — tests log-style lowering (via exp + addition).
#[nn_macros::kernel]
fn softplus(x: f32) -> f32 {
    (1.0 + x.exp()).sqrt() * (1.0 + x.exp()).sqrt()
}

/// Multi-step let-binding chain — tests IR node reuse through let bindings.
#[nn_macros::kernel]
fn multi_step(x: f32, alpha: f32) -> f32 {
    let a = x * alpha;
    let b = a.sin();
    let c = b * b;
    let d = c + x;
    d * 0.5
}

/// Kernel with constant literal propagation — tests constant folding.
#[nn_macros::kernel]
fn const_scaled(x: f32) -> f32 {
    let scale: f32 = 0.7071;
    x * scale
}

/// Kernel with bounds on both parameters.
#[nn_macros::kernel(bounds(x = "-100.0..100.0", y = "-100.0..100.0"))]
fn bounded_add(x: f32, y: f32) -> f32 {
    x + y
}

/// Three-parameter kernel — tests arity beyond 2.
#[nn_macros::kernel]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

/// Four-parameter kernel — tests higher arity.
#[nn_macros::kernel]
fn weighted_sum4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    0.25 * a + 0.25 * b + 0.25 * c + 0.25 * d
}

/// Kernel with deep nesting of method calls.
#[nn_macros::kernel]
fn deep_chain(x: f32) -> f32 {
    x.abs().sqrt().sin().abs()
}

/// Kernel using powi with different exponent.
#[nn_macros::kernel]
fn cube(x: f32) -> f32 {
    x.powi(3)
}

/// Conditional with complex branches.
#[nn_macros::kernel]
fn smooth_step(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x * x * (3.0 - 2.0 * x)
    }
}

/// Kernel using rsqrt with a guard.
#[nn_macros::kernel(bounds(x = "1e-8..1e6"))]
fn safe_rsqrt(x: f32) -> f32 {
    (x + 1e-8).rsqrt()
}

/// Single-parameter identity-ish kernel.
#[nn_macros::kernel]
fn identity_kern(x: f32) -> f32 {
    x
}

/// Subtraction-only kernel.
#[nn_macros::kernel]
fn negate(x: f32) -> f32 {
    0.0 - x
}

// ---------------------------------------------------------------------------
// Reference function tests
// ---------------------------------------------------------------------------

#[test]
fn test_exp_kernel_reference() {
    let result = exp_kernel(0.0);
    assert!((result - 1.0).abs() < 1e-6, "exp(0) = 1, got {result}");
    let result_neg = exp_kernel(-1.0);
    assert!(
        (result_neg - (-1.0_f32).exp()).abs() < 1e-6,
        "exp(-1) mismatch: got {result_neg}"
    );
}

#[test]
fn test_sigmoid_kern_reference() {
    let at_zero = sigmoid_kern(0.0);
    assert!(
        (at_zero - 0.5).abs() < 1e-6,
        "sigmoid(0) should be 0.5, got {at_zero}"
    );
    let large = sigmoid_kern(10.0);
    assert!(large > 0.999, "sigmoid(10) should be near 1, got {large}");
    let neg_large = sigmoid_kern(-10.0);
    assert!(
        neg_large < 0.001,
        "sigmoid(-10) should be near 0, got {neg_large}"
    );
}

#[test]
fn test_leaky_relu_reference() {
    assert_eq!(leaky_relu(5.0), 5.0);
    assert!((leaky_relu(-5.0) - (-0.05)).abs() < 1e-6);
    assert_eq!(leaky_relu(0.0), 0.0);
}

#[test]
fn test_multi_step_reference() {
    let result = multi_step(1.0, 2.0);
    let a: f32 = 1.0 * 2.0;
    let b = a.sin();
    let c = b * b;
    let d = c + 1.0;
    let expected = d * 0.5;
    assert!(
        (result - expected).abs() < 1e-6,
        "multi_step(1, 2) = {result}, expected {expected}"
    );
}

#[test]
fn test_const_scaled_reference() {
    let result = const_scaled(10.0);
    assert!(
        (result - 7.071).abs() < 1e-3,
        "const_scaled(10) = {result}, expected 7.071"
    );
}

#[test]
fn test_bounded_add_reference() {
    assert!((bounded_add(3.0, 4.0) - 7.0).abs() < 1e-6);
    assert!((bounded_add(-50.0, 50.0) - 0.0).abs() < 1e-6);
}

#[test]
fn test_fma_reference() {
    let result = fma(2.0, 3.0, 4.0);
    assert!((result - 10.0).abs() < 1e-6, "2*3+4 = 10, got {result}");
}

#[test]
fn test_weighted_sum4_reference() {
    let result = weighted_sum4(4.0, 8.0, 12.0, 16.0);
    let expected = 0.25 * 4.0 + 0.25 * 8.0 + 0.25 * 12.0 + 0.25 * 16.0;
    assert!(
        (result - expected).abs() < 1e-6,
        "weighted_sum4 = {result}, expected {expected}"
    );
}

#[test]
fn test_deep_chain_reference() {
    let result = deep_chain(4.0);
    let expected = 4.0_f32.abs().sqrt().sin().abs();
    assert!(
        (result - expected).abs() < 1e-6,
        "deep_chain(4) = {result}, expected {expected}"
    );
}

#[test]
fn test_cube_reference() {
    let result = cube(3.0);
    assert!((result - 27.0).abs() < 1e-4, "cube(3) = {result}");
    let neg = cube(-2.0);
    assert!((neg - (-8.0)).abs() < 1e-4, "cube(-2) = {neg}");
}

#[test]
fn test_smooth_step_reference() {
    assert_eq!(smooth_step(-1.0), 0.0);
    assert_eq!(smooth_step(2.0), 1.0);
    let mid = smooth_step(0.5);
    let expected = 0.5 * 0.5 * (3.0 - 2.0 * 0.5);
    assert!(
        (mid - expected).abs() < 1e-6,
        "smooth_step(0.5) = {mid}, expected {expected}"
    );
}

#[test]
fn test_safe_rsqrt_reference() {
    let result = safe_rsqrt(4.0);
    let expected = (4.0 + 1e-8_f32).rsqrt();
    assert!(
        (result - expected).abs() < 1e-6,
        "safe_rsqrt(4) = {result}, expected {expected}"
    );
}

#[test]
fn test_identity_kern_reference() {
    assert_eq!(identity_kern(42.0), 42.0);
    assert_eq!(identity_kern(-1.5), -1.5);
}

#[test]
fn test_negate_reference() {
    assert!((negate(5.0) - (-5.0)).abs() < 1e-6);
    assert!((negate(-3.0) - 3.0).abs() < 1e-6);
    assert!((negate(0.0) - 0.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// MSL generation structural tests
// ---------------------------------------------------------------------------

#[test]
fn test_exp_kernel_msl_has_exp() {
    assert!(
        EXP_KERNEL_MSL.contains("exp("),
        "exp kernel MSL should contain exp(), got:\n{EXP_KERNEL_MSL}"
    );
}

#[test]
fn test_sigmoid_kern_msl_has_exp() {
    assert!(
        SIGMOID_KERN_MSL.contains("exp("),
        "sigmoid MSL should use exp()"
    );
    assert!(
        SIGMOID_KERN_MSL.contains("sigmoid_kern_kernel"),
        "kernel entry point missing"
    );
}

#[test]
fn test_leaky_relu_msl_has_max_and_min() {
    assert!(
        LEAKY_RELU_MSL.contains("max("),
        "leaky_relu MSL should contain max()"
    );
    assert!(
        LEAKY_RELU_MSL.contains("min("),
        "leaky_relu MSL should contain min()"
    );
}

#[test]
fn test_fma_msl_three_params() {
    assert!(FMA_MSL.contains("fma_kernel"), "kernel entry point missing");
    assert!(FMA_MSL.contains("[[kernel]]"), "kernel attribute missing");
    // Three param buffers
    assert!(
        FMA_MSL.contains("device const float* a"),
        "param a buffer missing from MSL"
    );
    assert!(
        FMA_MSL.contains("device const float* b"),
        "param b buffer missing from MSL"
    );
    assert!(
        FMA_MSL.contains("device const float* c"),
        "param c buffer missing from MSL"
    );
}

#[test]
fn test_weighted_sum4_msl_four_params() {
    assert!(
        WEIGHTED_SUM4_MSL.contains("weighted_sum4_kernel"),
        "kernel entry point missing"
    );
    assert!(
        WEIGHTED_SUM4_MSL.contains("device const float* d"),
        "param d buffer missing from MSL"
    );
}

#[test]
fn test_deep_chain_msl_structure() {
    assert!(DEEP_CHAIN_MSL.contains("#include <metal_stdlib>"));
    assert!(DEEP_CHAIN_MSL.contains("[[kernel]]"));
    assert!(DEEP_CHAIN_MSL.contains("thread_position_in_grid"));
    assert!(DEEP_CHAIN_MSL.contains("deep_chain_kernel"));
}

#[test]
fn test_smooth_step_msl_has_ternary() {
    // if/else kernels emit ternary selects in MSL
    assert!(
        SMOOTH_STEP_MSL.contains(" ? "),
        "smooth_step MSL should use ternary for if/else, got:\n{SMOOTH_STEP_MSL}"
    );
}

#[test]
fn test_identity_msl_structure() {
    assert!(!IDENTITY_KERN_MSL.is_empty());
    assert!(IDENTITY_KERN_MSL.contains("identity_kern_kernel"));
    assert!(IDENTITY_KERN_MSL.contains("thread_position_in_grid"));
}

#[test]
fn test_all_kernels_have_metal_header() {
    let all_msls: &[(&str, &str)] = &[
        ("exp_kernel", EXP_KERNEL_MSL),
        ("sigmoid_kern", SIGMOID_KERN_MSL),
        ("leaky_relu", LEAKY_RELU_MSL),
        ("softplus", SOFTPLUS_MSL),
        ("multi_step", MULTI_STEP_MSL),
        ("const_scaled", CONST_SCALED_MSL),
        ("bounded_add", BOUNDED_ADD_MSL),
        ("fma", FMA_MSL),
        ("weighted_sum4", WEIGHTED_SUM4_MSL),
        ("deep_chain", DEEP_CHAIN_MSL),
        ("cube", CUBE_MSL),
        ("smooth_step", SMOOTH_STEP_MSL),
        ("safe_rsqrt", SAFE_RSQRT_MSL),
        ("identity_kern", IDENTITY_KERN_MSL),
        ("negate", NEGATE_MSL),
    ];

    for (name, msl) in all_msls {
        assert!(!msl.is_empty(), "{name}: MSL should not be empty");
        assert!(
            msl.contains("#include <metal_stdlib>"),
            "{name}: MSL missing metal header"
        );
        assert!(
            msl.contains("[[kernel]]"),
            "{name}: MSL missing [[kernel]] attribute"
        );
        assert!(
            msl.contains("thread_position_in_grid"),
            "{name}: MSL missing thread_position_in_grid"
        );
    }
}

// ---------------------------------------------------------------------------
// Metadata tests
// ---------------------------------------------------------------------------

#[test]
fn test_exp_kernel_metadata() {
    assert_eq!(__exp_kernel_kernel_meta::PARAM_COUNT, 1);
    assert!(__exp_kernel_kernel_meta::IR_DEBUG.contains("exp("));
}

#[test]
fn test_sigmoid_kern_metadata() {
    assert_eq!(__sigmoid_kern_kernel_meta::PARAM_COUNT, 1);
    assert!(__sigmoid_kern_kernel_meta::IR_DEBUG.contains("kernel sigmoid_kern("));
}

#[test]
fn test_fma_metadata() {
    assert_eq!(__fma_kernel_meta::PARAM_COUNT, 3);
    assert!(__fma_kernel_meta::IR_DEBUG.contains("a: f32"));
    assert!(__fma_kernel_meta::IR_DEBUG.contains("b: f32"));
    assert!(__fma_kernel_meta::IR_DEBUG.contains("c: f32"));
}

#[test]
fn test_weighted_sum4_metadata() {
    assert_eq!(__weighted_sum4_kernel_meta::PARAM_COUNT, 4);
}

#[test]
fn test_multi_step_metadata() {
    assert_eq!(__multi_step_kernel_meta::PARAM_COUNT, 2);
    // Should have more IR nodes than a simple one-op kernel
    assert!(
        __multi_step_kernel_meta::NODE_COUNT > 3,
        "multi_step should have several IR nodes, got {}",
        __multi_step_kernel_meta::NODE_COUNT,
    );
}

#[test]
fn test_identity_kern_metadata() {
    assert_eq!(__identity_kern_kernel_meta::PARAM_COUNT, 1);
    // Identity kernel should have minimal nodes (just param + return)
    assert!(
        __identity_kern_kernel_meta::NODE_COUNT <= 2,
        "identity kernel should have at most 2 IR nodes, got {}",
        __identity_kern_kernel_meta::NODE_COUNT,
    );
}

// ---------------------------------------------------------------------------
// Descriptor tests
// ---------------------------------------------------------------------------

#[test]
fn test_fma_descriptor_param_count() {
    assert_eq!(
        FMA_DESCRIPTOR.param_count, 3,
        "FMA_DESCRIPTOR.param_count must be 3 for fn fma(a, b, c)"
    );
    assert_eq!(FMA_DESCRIPTOR.entry_point, "fma_kernel");
}

#[test]
fn test_weighted_sum4_descriptor_param_count() {
    assert_eq!(
        WEIGHTED_SUM4_DESCRIPTOR.param_count, 4,
        "WEIGHTED_SUM4_DESCRIPTOR.param_count must be 4"
    );
    assert_eq!(WEIGHTED_SUM4_DESCRIPTOR.entry_point, "weighted_sum4_kernel");
}

#[test]
fn test_exp_kernel_descriptor() {
    assert_eq!(EXP_KERNEL_DESCRIPTOR.param_count, 1);
    assert_eq!(EXP_KERNEL_DESCRIPTOR.entry_point, "exp_kernel_kernel");
    assert!(!EXP_KERNEL_DESCRIPTOR.msl_source.is_empty());
}

#[test]
fn test_sigmoid_kern_descriptor() {
    assert_eq!(SIGMOID_KERN_DESCRIPTOR.param_count, 1);
    assert_eq!(SIGMOID_KERN_DESCRIPTOR.entry_point, "sigmoid_kern_kernel");
}

#[test]
fn test_bounded_add_descriptor() {
    assert_eq!(BOUNDED_ADD_DESCRIPTOR.param_count, 2);
    assert_eq!(BOUNDED_ADD_DESCRIPTOR.entry_point, "bounded_add_kernel");
}

// ---------------------------------------------------------------------------
// Precision tier metadata tests
// ---------------------------------------------------------------------------

#[test]
fn test_default_precision_is_normal() {
    assert_eq!(__exp_kernel_kernel_meta::PRECISION_TIER, "normal");
    assert_eq!(__fma_kernel_meta::PRECISION_TIER, "normal");
    assert_eq!(__multi_step_kernel_meta::PRECISION_TIER, "normal");
}

#[test]
fn test_default_precision_no_fast_math() {
    const { assert!(!__exp_kernel_kernel_meta::FAST_MATH) };
    const { assert!(!__fma_kernel_meta::FAST_MATH) };
}

// ---------------------------------------------------------------------------
// No-collision tests
// ---------------------------------------------------------------------------

#[test]
fn test_all_kernels_distinct_msl() {
    let msls = [
        EXP_KERNEL_MSL,
        SIGMOID_KERN_MSL,
        LEAKY_RELU_MSL,
        FMA_MSL,
        IDENTITY_KERN_MSL,
        NEGATE_MSL,
    ];
    for i in 0..msls.len() {
        for j in (i + 1)..msls.len() {
            assert_ne!(
                msls[i], msls[j],
                "MSL collision between kernel {i} and kernel {j}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Numerical sweep tests
// ---------------------------------------------------------------------------

#[test]
fn test_sigmoid_kern_sweep_finite() {
    for i in -200..=200 {
        let x = i as f32 * 0.1;
        let result = sigmoid_kern(x);
        assert!(result.is_finite(), "sigmoid({x}) = {result} is not finite");
        assert!(
            (0.0..=1.0).contains(&result),
            "sigmoid({x}) = {result} out of [0, 1] range"
        );
    }
}

#[test]
fn test_leaky_relu_sweep_finite() {
    for i in -100..=100 {
        let x = i as f32 * 0.5;
        let result = leaky_relu(x);
        assert!(
            result.is_finite(),
            "leaky_relu({x}) = {result} is not finite"
        );
        if x >= 0.0 {
            assert_eq!(result, x, "positive region should be identity");
        } else {
            assert!(result < 0.0, "negative region should be negative");
        }
    }
}

#[test]
fn test_smooth_step_sweep_bounds() {
    for i in -20..=30 {
        let x = i as f32 * 0.1;
        let result = smooth_step(x);
        assert!(
            result.is_finite(),
            "smooth_step({x}) = {result} is not finite"
        );
        assert!(
            (0.0..=1.0).contains(&result),
            "smooth_step({x}) = {result} out of [0, 1] range"
        );
    }
}

#[test]
fn test_deep_chain_sweep_finite() {
    for i in -100..=100 {
        let x = i as f32 * 0.1;
        let result = deep_chain(x);
        assert!(
            result.is_finite(),
            "deep_chain({x}) = {result} is not finite"
        );
        assert!(
            result >= 0.0,
            "deep_chain({x}) = {result} should be non-negative (abs output)"
        );
    }
}
