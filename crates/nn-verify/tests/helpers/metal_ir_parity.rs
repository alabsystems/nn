// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal↔IR differential testing: element-wise kernels.
//!
//! Verifies that MSL codegen from `nn-dsl` produces Metal kernels whose GPU
//! output matches the Rust reference implementations within precision budget.
//!
//! Addresses #303: dvoice differential testing bridge.
//!
//! # Coverage
//!
//! Element-wise kernels (`dispatch_elementwise`):
//! - K1 Snake: `snake_scalar()` reference
//! - K3 AdaIN: `adain_scalar()` reference
//! - K4 Fused AdaIN+Snake: `adain_snake_fused_scalar()` reference
//! - K8 SiLU-Mul: `silu_mul_scalar()` reference
//!
//! Reduction + structural kernels split to `metal_ir_parity_reduction.rs` (#548).
//!
//! All tests are gated on `target_os = "macos"` (Metal API requirement).

#![cfg(target_os = "macos")]

use nn_dsl::{
    adain_scalar, adain_snake_fused_scalar, build_adain_scalar_kernel,
    build_adain_snake_fused_kernel, build_silu_mul_kernel, build_snake_scalar_kernel,
    silu_mul_scalar, snake_scalar,
};
use nn_metal::KernelPipeline;

#[path = "../common/metal_helpers.rs"]
mod metal_helpers;
use metal_helpers::{assert_metal_cpu_parity, assert_within_budget, metal_setup, rand_f32_vec};

const SAMPLE_COUNT: usize = 1024;

// ===========================================================================
// K1 Snake: snake(x, alpha) = x + (1/alpha) * sin(alpha*x)^2
// ===========================================================================

#[test]
fn test_k1_snake_metal_ir_parity() {
    let cache = metal_setup();
    let kernel = build_snake_scalar_kernel().expect("build K1 Snake");
    let pipeline = KernelPipeline::compile(&cache, &kernel).expect("compile K1 Snake to MSL");

    let alpha = 0.5_f32;
    let x_data = rand_f32_vec(0xDEAD_BEEF, SAMPLE_COUNT, -10.0, 10.0);
    let alpha_data = vec![alpha; SAMPLE_COUNT];

    // CPU reference
    let cpu_out: Vec<f32> = x_data
        .iter()
        .map(|&x| snake_scalar(x, alpha).expect("finite test input"))
        .collect();

    // Metal GPU
    let gpu_out = pipeline
        .dispatch_elementwise(cache.context(), &[&x_data, &alpha_data])
        .expect("Metal dispatch K1 Snake");

    assert_metal_cpu_parity("snake", &gpu_out, &cpu_out);
    assert_within_budget("snake", &gpu_out, &cpu_out, &[]);
}

// ===========================================================================
// K3 AdaIN: adain(x, mu, var, gamma, beta, eps)
// ===========================================================================

#[test]
fn test_k3_adain_metal_ir_parity() {
    let cache = metal_setup();
    let kernel = build_adain_scalar_kernel().expect("build K3 AdaIN");
    let pipeline = KernelPipeline::compile(&cache, &kernel).expect("compile K3 AdaIN to MSL");

    // Constants: mu=0, var=1, gamma=1, beta=0, eps=1e-5
    let mu = 0.0_f32;
    let var_val = 1.0_f32;
    let gamma = 1.0_f32;
    let beta = 0.0_f32;
    let eps = 1e-5_f32;

    let x_data = rand_f32_vec(0xCAFE_BABE, SAMPLE_COUNT, -5.0, 5.0);
    let mu_data = vec![mu; SAMPLE_COUNT];
    let var_data = vec![var_val; SAMPLE_COUNT];
    let gamma_data = vec![gamma; SAMPLE_COUNT];
    let beta_data = vec![beta; SAMPLE_COUNT];
    let eps_data = vec![eps; SAMPLE_COUNT];

    // CPU reference
    let cpu_out: Vec<f32> = x_data
        .iter()
        .map(|&x| adain_scalar(x, mu, var_val, gamma, beta, eps).expect("adain ref"))
        .collect();

    // Metal GPU
    let gpu_out = pipeline
        .dispatch_elementwise(
            cache.context(),
            &[
                &x_data,
                &mu_data,
                &var_data,
                &gamma_data,
                &beta_data,
                &eps_data,
            ],
        )
        .expect("Metal dispatch K3 AdaIN");

    assert_metal_cpu_parity("adain", &gpu_out, &cpu_out);
    assert_within_budget("adain", &gpu_out, &cpu_out, &[]);
}

// ===========================================================================
// K4 Fused AdaIN+Snake: adain_snake(x, mu, var, gamma, beta, alpha, eps)
// ===========================================================================

#[test]
fn test_k4_fused_adain_snake_metal_ir_parity() {
    let cache = metal_setup();
    let kernel = build_adain_snake_fused_kernel().expect("build K4 Fused AdaIN+Snake");
    let pipeline = KernelPipeline::compile(&cache, &kernel).expect("compile K4 to MSL");

    // Constants: mu=0, var=1, gamma=1, beta=0, alpha=0.5, eps=1e-5
    let mu = 0.0_f32;
    let var_val = 1.0_f32;
    let gamma = 1.0_f32;
    let beta = 0.0_f32;
    let alpha = 0.5_f32;
    let eps = 1e-5_f32;

    let x_data = rand_f32_vec(0xBAAD_F00D, SAMPLE_COUNT, -5.0, 5.0);
    let mu_data = vec![mu; SAMPLE_COUNT];
    let var_data = vec![var_val; SAMPLE_COUNT];
    let gamma_data = vec![gamma; SAMPLE_COUNT];
    let beta_data = vec![beta; SAMPLE_COUNT];
    let alpha_data = vec![alpha; SAMPLE_COUNT];
    let eps_data = vec![eps; SAMPLE_COUNT];

    // CPU reference
    let cpu_out: Vec<f32> = x_data
        .iter()
        .map(|&x| {
            adain_snake_fused_scalar(x, mu, var_val, gamma, beta, alpha, eps)
                .expect("adain_snake ref")
        })
        .collect();

    // Metal GPU
    let gpu_out = pipeline
        .dispatch_elementwise(
            cache.context(),
            &[
                &x_data,
                &mu_data,
                &var_data,
                &gamma_data,
                &beta_data,
                &alpha_data,
                &eps_data,
            ],
        )
        .expect("Metal dispatch K4 Fused AdaIN+Snake");

    assert_metal_cpu_parity("adain_snake_fused", &gpu_out, &cpu_out);
    assert_within_budget("adain_snake_fused", &gpu_out, &cpu_out, &[]);
}

// ===========================================================================
// K8 SiLU-Mul: silu_mul(x, up) = silu(x) * up
// ===========================================================================

#[test]
fn test_k8_silu_mul_metal_ir_parity() {
    let cache = metal_setup();
    let kernel = build_silu_mul_kernel().expect("build K8 SiLU-Mul");
    let pipeline = KernelPipeline::compile(&cache, &kernel).expect("compile K8 SiLU-Mul to MSL");

    let up = 2.0_f32;
    let x_data = rand_f32_vec(0x1234_5678, SAMPLE_COUNT, -5.0, 5.0);
    let up_data = vec![up; SAMPLE_COUNT];

    // CPU reference
    let cpu_out: Vec<f32> = x_data
        .iter()
        .map(|&x| silu_mul_scalar(x, up).expect("finite inputs"))
        .collect();

    // Metal GPU
    let gpu_out = pipeline
        .dispatch_elementwise(cache.context(), &[&x_data, &up_data])
        .expect("Metal dispatch K8 SiLU-Mul");

    assert_metal_cpu_parity("silu_mul", &gpu_out, &cpu_out);
    assert_within_budget("silu_mul", &gpu_out, &cpu_out, &[]);
}
