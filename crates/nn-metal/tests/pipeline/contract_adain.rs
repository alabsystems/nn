// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests for AdaIN (K3) and fused AdaIN+Snake (K4).
//!
//! These are dvoice's per-element normalization + activation kernels.
//! AdaIN has 6 parameters; the fused variant adds alpha (7 parameters).
//!
//! Both use `rsqrt` (FTZ-sensitive), so denormal edge cases are skipped
//! per `KernelDef::has_ftz_sensitive_op()`. Test parameters are chosen
//! to keep the rsqrt argument well away from denormal range.
//!
//! Part of #570. Refactored to use shared harness (#700).

use nn_dsl::KernelOps;

use super::contract_harness;

// ============================================================================
// AdaIN kernel definitions for cross-backend contract testing (#570)
// ============================================================================

/// AdaIN (K3) per-element: gamma * (x - mu) * rsqrt(var_val + eps) + beta
#[nn_macros::kernel]
fn adain_k(x: f32, mu: f32, var_val: f32, gamma: f32, beta: f32, eps: f32) -> f32 {
    gamma * (x - mu) * (var_val + eps).rsqrt() + beta
}

/// Fused AdaIN+Snake (K4) per-element: AdaIN output fed into Snake activation.
/// Note: 1e-8 must match nn_dsl::SNAKE_MIN_ALPHA — see #325.
#[nn_macros::kernel]
fn adain_snake_k(
    x: f32,
    mu: f32,
    var_val: f32,
    gamma: f32,
    beta: f32,
    alpha: f32,
    eps: f32,
) -> f32 {
    let y = gamma * (x - mu) * (var_val + eps).rsqrt() + beta;
    let a = alpha.max(1e-8);
    y + (1.0 / a) * (a * y).sin().powi(2)
}

const TEST_X_9: [f32; 9] = [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0];

// ============================================================================
// Contract tests
// ============================================================================

/// AdaIN identity config: mu=0, var=1, gamma=1, beta=0, eps=1e-5.
/// Reduces to x * rsqrt(1.00001) ≈ x. Part of #570.
#[test]
fn test_adain_gpu_output_within_verified_bounds() {
    let kernel_def = nn_dsl::build_adain_scalar_kernel().expect("build adain KernelDef");
    contract_harness::assert_scalar_contract(
        &kernel_def,
        &ADAIN_K_DESCRIPTOR,
        &[0.0, 1.0, 1.0, 0.0, 1e-5], // mu, var_val, gamma, beta, eps
        (-5.0, 5.0),
        &TEST_X_9,
        "adain",
    );
}

/// AdaIN scaled config: mu=1.0, var=2.0, gamma=0.5, beta=0.1, eps=1e-5.
/// Tests proof-execution contract under realistic non-identity config. Part of #570.
#[test]
fn test_adain_scaled_gpu_output_within_verified_bounds() {
    let kernel_def = nn_dsl::build_adain_scalar_kernel().expect("build adain KernelDef");
    contract_harness::assert_scalar_contract(
        &kernel_def,
        &ADAIN_K_DESCRIPTOR,
        &[1.0, 2.0, 0.5, 0.1, 1e-5], // mu, var_val, gamma, beta, eps
        (-5.0, 5.0),
        &TEST_X_9,
        "adain (scaled)",
    );
}

/// Fused AdaIN+Snake identity config: mu=0, var=1, gamma=1, beta=0, alpha=1, eps=1e-5.
/// Dvoice K4 — most complex kernel (7 params, rsqrt + sin + powi). Part of #570.
#[test]
fn test_adain_snake_fused_gpu_output_within_verified_bounds() {
    let kernel_def =
        nn_dsl::build_adain_snake_fused_kernel().expect("build adain_snake KernelDef");
    contract_harness::assert_scalar_contract(
        &kernel_def,
        &ADAIN_SNAKE_K_DESCRIPTOR,
        &[0.0, 1.0, 1.0, 0.0, 1.0, 1e-5], // mu, var_val, gamma, beta, alpha, eps
        (-5.0, 5.0),
        &TEST_X_9,
        "adain_snake",
    );
}

/// Fused AdaIN+Snake scaled config: mu=1, var=2, gamma=0.5, beta=0.1, alpha=2, eps=1e-5.
/// Higher alpha amplifies sin² oscillations, stressing trig-heavy proof paths. Part of #570.
#[test]
fn test_adain_snake_fused_scaled_gpu_output_within_verified_bounds() {
    let kernel_def =
        nn_dsl::build_adain_snake_fused_kernel().expect("build adain_snake KernelDef");
    contract_harness::assert_scalar_contract(
        &kernel_def,
        &ADAIN_SNAKE_K_DESCRIPTOR,
        &[1.0, 2.0, 0.5, 0.1, 2.0, 1e-5], // mu, var_val, gamma, beta, alpha, eps
        (-5.0, 5.0),
        &TEST_X_9,
        "adain_snake (scaled)",
    );
}
