// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "macos")]
#![allow(unexpected_cfgs)]

use nn_metal::compile_msl_pipeline;

#[nn_macros::kernel(bounds(alpha = "0.1..1e6"))]
fn snake(x: f32, alpha: f32) -> f32 {
    x + (1.0 / alpha) * (alpha * x).sin().powi(2)
}

#[nn_macros::kernel(precision = "strict", bounds(alpha = "0.1..1e6"))]
fn strict_snake(x: f32, alpha: f32) -> f32 {
    x + (1.0 / alpha) * (alpha * x).sin().powi(2)
}

#[nn_macros::kernel(precision = "relaxed", bounds(alpha = "0.1..1e6"))]
fn relaxed_snake(x: f32, alpha: f32) -> f32 {
    x + (1.0 / alpha) * (alpha * x).sin().powi(2)
}

#[nn_macros::kernel]
fn clamped(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

#[nn_macros::kernel]
fn inv_norm(x: f32) -> f32 {
    (x * x + 1.0).sqrt().recip()
}

#[test]
fn test_generated_snake_msl_compiles_to_pipeline() {
    let reference = snake(0.5, 1.0);
    assert!(
        reference.is_finite(),
        "reference snake result should be finite"
    );

    let (_context, _pipeline) =
        compile_msl_pipeline(SNAKE_MSL, "snake_kernel", __snake_kernel_meta::FAST_MATH)
            .expect("snake MSL should compile and create a pipeline through nn-metal");
}

#[test]
fn test_strict_snake_msl_compiles_to_pipeline() {
    let reference = strict_snake(0.5, 1.0);
    assert!(reference.is_finite());
    assert!(
        STRICT_SNAKE_MSL.contains("metal::precise::sin"),
        "strict tier should use precise intrinsics"
    );

    let (_context, _pipeline) = compile_msl_pipeline(
        STRICT_SNAKE_MSL,
        "strict_snake_kernel",
        __strict_snake_kernel_meta::FAST_MATH,
    )
    .expect("strict-precision MSL should compile on Metal");
}

#[test]
fn test_relaxed_snake_msl_compiles_to_pipeline() {
    let reference = relaxed_snake(0.5, 1.0);
    assert!(reference.is_finite());
    assert!(
        !RELAXED_SNAKE_MSL.contains("metal::precise::sin"),
        "relaxed tier should not use precise intrinsics"
    );

    let (_context, _pipeline) = compile_msl_pipeline(
        RELAXED_SNAKE_MSL,
        "relaxed_snake_kernel",
        __relaxed_snake_kernel_meta::FAST_MATH,
    )
    .expect("relaxed-precision MSL should compile on Metal");
}

#[test]
fn test_clamp_msl_compiles_to_pipeline() {
    let reference = clamped(0.5);
    assert_eq!(reference, 0.5);

    let (_context, _pipeline) = compile_msl_pipeline(
        CLAMPED_MSL,
        "clamped_kernel",
        __clamped_kernel_meta::FAST_MATH,
    )
    .expect("clamp MSL should compile on Metal");
}

#[test]
fn test_sqrt_recip_msl_compiles_to_pipeline() {
    let reference = inv_norm(0.0);
    assert!((reference - 1.0).abs() < 1e-6);

    let (_context, _pipeline) = compile_msl_pipeline(
        INV_NORM_MSL,
        "inv_norm_kernel",
        __inv_norm_kernel_meta::FAST_MATH,
    )
    .expect("rsqrt MSL should compile on Metal");
}
