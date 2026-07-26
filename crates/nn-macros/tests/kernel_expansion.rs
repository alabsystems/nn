// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end test: apply `#[nn_macros::kernel]` to the snake function
//! and verify the expansion generates working code, MSL, and metadata.

// The kernel macro generates `#[cfg(kani)]` blocks; suppress the check-cfg warning.
#![allow(unexpected_cfgs)]

#[nn_macros::kernel(bounds(alpha = "0.1..1e6"))]
fn snake(x: f32, alpha: f32) -> f32 {
    x + (1.0 / alpha) * (alpha * x).sin().powi(2)
}

#[nn_macros::kernel]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[nn_macros::kernel]
fn gelu(x: f32) -> f32 {
    let k: f32 = 0.797_884_6;
    let inner = k * (x + 0.044715 * x * x * x);
    let e2 = (2.0 * inner).exp();
    0.5 * x * (2.0 - 2.0 / (e2 + 1.0))
}

#[nn_macros::kernel]
fn relu_if(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

#[nn_macros::kernel]
fn sum3(a: f32, b: f32, c: f32) -> f32 {
    nn_dsl::sum_reduce([a, b, c])
}

#[nn_macros::kernel(precision = "strict")]
fn strict_sin(x: f32) -> f32 {
    x.sin()
}

#[nn_macros::kernel(precision = "relaxed")]
fn relaxed_sin(x: f32) -> f32 {
    x.sin()
}

// --- Coverage kernels for remaining ops ---

#[nn_macros::kernel]
fn clamped(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

#[nn_macros::kernel]
fn cos_abs(x: f32) -> f32 {
    x.cos().abs()
}

#[nn_macros::kernel]
fn norm_factor(x: f32) -> f32 {
    x.abs().sqrt()
}

#[nn_macros::kernel]
fn inv_norm(x: f32) -> f32 {
    (x * x + 1.0).sqrt().recip()
}

use nn_dsl::KernelOps;

#[nn_macros::kernel(bounds(x = "1e-6..1e6"))]
fn inv_sqrt(x: f32) -> f32 {
    x.rsqrt()
}

#[test]
fn test_snake_reference_fn_works() {
    let result = snake(1.0, 1.0);
    let expected = 1.0 + (1.0 / 1.0) * (1.0_f32).sin().powi(2);
    assert!(
        (result - expected).abs() < 1e-6,
        "snake(1, 1) = {result}, expected {expected}"
    );
}

#[test]
fn test_snake_msl_generated() {
    assert!(!SNAKE_MSL.is_empty(), "MSL should not be empty");
    assert!(
        SNAKE_MSL.contains("#include <metal_stdlib>"),
        "MSL header missing"
    );
    assert!(
        SNAKE_MSL.contains("metal::precise::sin"),
        "sin not using precise math"
    );
    assert!(
        SNAKE_MSL.contains("snake_kernel"),
        "kernel entry point missing"
    );
    assert!(SNAKE_MSL.contains("[[kernel]]"), "kernel attribute missing");
    assert!(
        SNAKE_MSL.contains("thread_position_in_grid"),
        "thread position missing"
    );
}

#[test]
fn test_snake_ir_metadata() {
    let node_count = __snake_kernel_meta::NODE_COUNT;
    assert!(
        node_count > 2,
        "snake should have more than 2 IR nodes, got {node_count}",
    );
    assert_eq!(__snake_kernel_meta::PARAM_COUNT, 2, "snake has 2 params");
    assert!(
        __snake_kernel_meta::IR_DEBUG.contains("kernel snake(x: f32, alpha: f32) -> f32"),
        "IR debug should contain signature, got:\n{}",
        __snake_kernel_meta::IR_DEBUG,
    );
    assert!(
        __snake_kernel_meta::IR_DEBUG.contains("sin("),
        "IR debug should contain sin node",
    );
}

#[test]
fn test_precision_metadata_defaults_to_normal() {
    assert_eq!(__snake_kernel_meta::PRECISION_TIER, "normal");
    const { assert!(!__snake_kernel_meta::FAST_MATH) };
    assert_eq!(__snake_kernel_meta::DIFFERENTIAL_ABS_BUDGET, 1e-5);
    assert_eq!(__snake_kernel_meta::DIFFERENTIAL_REL_BUDGET, 1e-5);
}

#[test]
fn test_precision_metadata_strict() {
    // Exercise the generated reference function to avoid dead_code warning.
    let _ = strict_sin(0.5);
    assert_eq!(__strict_sin_kernel_meta::PRECISION_TIER, "strict");
    const { assert!(!__strict_sin_kernel_meta::FAST_MATH) };
    assert_eq!(__strict_sin_kernel_meta::DIFFERENTIAL_ABS_BUDGET, 1e-6);
    assert!(
        STRICT_SIN_MSL.contains("metal::precise::sin"),
        "strict precision should keep precise intrinsics"
    );
}

#[test]
fn test_precision_metadata_relaxed() {
    // Exercise the generated reference function to avoid dead_code warning.
    let _ = relaxed_sin(0.5);
    assert_eq!(__relaxed_sin_kernel_meta::PRECISION_TIER, "relaxed");
    const { assert!(__relaxed_sin_kernel_meta::FAST_MATH) };
    assert_eq!(__relaxed_sin_kernel_meta::DIFFERENTIAL_ABS_BUDGET, 1e-4);
    assert!(
        RELAXED_SIN_MSL.contains("metal::sin"),
        "relaxed precision should use relaxed intrinsics"
    );
    assert!(
        !RELAXED_SIN_MSL.contains("metal::precise::sin"),
        "relaxed precision should not use precise intrinsics"
    );
}

#[test]
fn test_relu_reference_fn_works() {
    assert_eq!(relu(5.0), 5.0);
    assert_eq!(relu(-3.0), 0.0);
    assert_eq!(relu(0.0), 0.0);
}

#[test]
fn test_relu_msl_generated() {
    assert!(!RELU_MSL.is_empty());
    assert!(RELU_MSL.contains("max("), "relu MSL should use max()");
    assert!(
        RELU_MSL.contains("relu_kernel"),
        "kernel entry point missing"
    );
}

#[test]
fn test_gelu_works() {
    let result_zero = gelu(0.0);
    assert!(
        result_zero.abs() < 1e-6,
        "gelu(0) should be ~0, got {result_zero}"
    );
    let result_one = gelu(1.0);
    assert!(
        (result_one - 0.8412).abs() < 0.001,
        "gelu(1) should be ~0.8412, got {result_one}"
    );
}

#[test]
fn test_gelu_msl_generated() {
    assert!(!GELU_MSL.is_empty());
    assert!(
        GELU_MSL.contains("gelu_kernel"),
        "kernel entry point missing",
    );
}

#[test]
fn test_relu_if_reference_fn_works() {
    assert_eq!(relu_if(5.0), 5.0);
    assert_eq!(relu_if(-3.0), 0.0);
}

#[test]
fn test_relu_if_msl_generated() {
    assert!(!RELU_IF_MSL.is_empty());
    assert!(
        RELU_IF_MSL.contains(" ? "),
        "if/else kernel should emit ternary select in MSL"
    );
    assert!(
        RELU_IF_MSL.contains(" > "),
        "if/else kernel should emit comparison in MSL"
    );
}

#[test]
fn test_sum3_reference_fn_works() {
    let result = sum3(1.25, -2.5, 4.0);
    let expected = 1.25 - 2.5 + 4.0;
    assert!(
        (result - expected).abs() < 1e-6,
        "sum3 mismatch: result={result}, expected={expected}"
    );
}

#[test]
fn test_sum3_msl_generated() {
    assert!(!SUM3_MSL.is_empty());
    assert!(
        SUM3_MSL.contains("sum3_kernel"),
        "kernel entry point missing"
    );
    assert!(
        SUM3_MSL.contains("a + b + c"),
        "sum_reduce lowering should emit add-chain in MSL"
    );
    assert_eq!(__sum3_kernel_meta::PARAM_COUNT, 3);
}

#[test]
fn test_snake_numerical_sweep() {
    for i in -100..=100 {
        let x = i as f32 * 0.1;
        for j in 1..=20 {
            let alpha = j as f32 * 0.5;
            let result = snake(x, alpha);
            assert!(
                result.is_finite(),
                "snake({x}, {alpha}) = {result} is not finite"
            );
        }
    }
}

// --- Coverage tests for remaining ops (clamp, cos, abs, sqrt, rsqrt) ---

#[test]
fn test_clamped_reference_fn_works() {
    assert_eq!(clamped(0.5), 0.5);
    assert_eq!(clamped(-1.0), 0.0);
    assert_eq!(clamped(2.0), 1.0);
}

#[test]
fn test_clamped_msl_generated() {
    assert!(
        CLAMPED_MSL.contains("clamp("),
        "clamp op should emit clamp() in MSL"
    );
    assert!(
        CLAMPED_MSL.contains("clamped_kernel"),
        "kernel entry point missing"
    );
}

#[test]
fn test_cos_abs_reference_fn_works() {
    let result = cos_abs(0.0);
    assert!(
        (result - 1.0).abs() < 1e-6,
        "cos(0).abs() should be 1.0, got {result}"
    );
}

#[test]
fn test_cos_abs_msl_generated() {
    assert!(
        COS_ABS_MSL.contains("metal::precise::cos"),
        "cos should use precise intrinsic"
    );
    assert!(
        COS_ABS_MSL.contains("metal::abs"),
        "abs should emit metal::abs"
    );
    assert!(
        COS_ABS_MSL.contains("cos_abs_kernel"),
        "kernel entry point missing"
    );
}

#[test]
fn test_norm_factor_reference_fn_works() {
    let result = norm_factor(4.0);
    assert!(
        (result - 2.0).abs() < 1e-6,
        "sqrt(abs(4.0)) should be 2.0, got {result}"
    );
}

#[test]
fn test_norm_factor_msl_generated() {
    assert!(
        NORM_FACTOR_MSL.contains("metal::precise::sqrt"),
        "sqrt should use precise intrinsic"
    );
    assert!(
        NORM_FACTOR_MSL.contains("metal::abs"),
        "abs should emit metal::abs in norm_factor"
    );
}

#[test]
fn test_inv_norm_reference_fn_works() {
    let result = inv_norm(0.0);
    assert!(
        (result - 1.0).abs() < 1e-6,
        "1/sqrt(0*0+1) should be 1.0, got {result}"
    );
}

#[test]
fn test_inv_norm_msl_generated() {
    assert!(
        INV_NORM_MSL.contains("metal::precise::sqrt"),
        "sqrt should use precise intrinsic in inv_norm"
    );
    // recip(x) lowers to `float(1) / x` in MSL
    assert!(
        INV_NORM_MSL.contains("float(1) /"),
        "recip should emit 1/x pattern in MSL"
    );
}

#[test]
fn test_inv_sqrt_reference_fn_works() {
    let result = inv_sqrt(4.0);
    assert!(
        (result - 0.5).abs() < 1e-6,
        "rsqrt(4.0) should be 0.5, got {result}"
    );
}

#[test]
fn test_inv_sqrt_msl_emits_rsqrt() {
    assert!(
        INV_SQRT_MSL.contains("metal::precise::rsqrt("),
        "rsqrt should emit metal::precise::rsqrt, MSL:\n{INV_SQRT_MSL}"
    );
    assert!(
        INV_SQRT_MSL.contains("inv_sqrt_kernel"),
        "kernel entry point missing"
    );
}

#[test]
fn test_multiple_kernels_no_collision() {
    // Verify that multiple kernels can coexist — each gets its own MSL const
    // and metadata module without name collisions.
    assert_ne!(SNAKE_MSL, RELU_MSL);
    assert_ne!(RELU_MSL, GELU_MSL);
    assert_ne!(SUM3_MSL, GELU_MSL);
    assert_ne!(RELU_IF_MSL, RELU_MSL);
    assert_ne!(
        __snake_kernel_meta::NODE_COUNT,
        __relu_kernel_meta::NODE_COUNT,
    );
}
