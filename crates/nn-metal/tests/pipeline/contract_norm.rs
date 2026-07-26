// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests for normalization kernels:
//! rms_norm_scalar, layer_norm_scalar, instance_norm_scalar, instance_norm_affine_scalar.
//!
//! These are multi-parameter scalar kernels where only x is the variable
//! and the remaining parameters (mean, var, eps, etc.) are constants.
//! Part of #506. Refactored to use shared harness (#700).

use nn_dsl::KernelOps;

use super::contract_harness;

// ============================================================================
// Normalization kernels for cross-backend contract testing (#506)
// ============================================================================

#[nn_macros::kernel]
fn rms_norm_scalar(x: f32, rms_inv: f32, weight: f32) -> f32 {
    x * rms_inv * weight
}

#[nn_macros::kernel]
fn instance_norm_scalar(x: f32, mean: f32, var_val: f32, eps: f32) -> f32 {
    (x - mean) * (var_val + eps).rsqrt()
}

#[nn_macros::kernel]
fn layer_norm_scalar(x: f32, mean: f32, var_val: f32, eps: f32, gamma: f32, beta: f32) -> f32 {
    (x - mean) * (var_val + eps).rsqrt() * gamma + beta
}

#[nn_macros::kernel]
fn instance_norm_affine_scalar(
    x: f32,
    mean: f32,
    var_val: f32,
    eps: f32,
    gamma: f32,
    beta: f32,
) -> f32 {
    (x - mean) * (var_val + eps).rsqrt() * gamma + beta
}

const TEST_X_9: [f32; 9] = [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0];

// ============================================================================
// Contract tests
// ============================================================================

/// rms_norm_scalar(x, rms_inv, weight) = x * rms_inv * weight
/// for x ∈ [-5, 5], rms_inv=1.0, weight=1.0 (identity config).
/// Part of #506.
#[test]
fn test_rms_norm_gpu_output_within_verified_bounds() {
    let kernel_def =
        nn_dsl::build_rms_norm_scalar_kernel().expect("build rms_norm_scalar KernelDef");
    contract_harness::assert_scalar_contract(
        &kernel_def,
        &RMS_NORM_SCALAR_DESCRIPTOR,
        &[1.0, 1.0], // rms_inv, weight
        (-5.0, 5.0),
        &TEST_X_9,
        "rms_norm",
    );
}

/// instance_norm_scalar(x, mean, var_val, eps) = (x - mean) * rsqrt(var_val + eps)
/// for x ∈ [-5, 5], mean=0.0, var_val=1.0, eps=1e-5 (identity config).
/// Part of #506.
#[test]
fn test_instance_norm_gpu_output_within_verified_bounds() {
    let kernel_def =
        nn_dsl::build_instance_norm_scalar_kernel().expect("build instance_norm KernelDef");
    contract_harness::assert_scalar_contract(
        &kernel_def,
        &INSTANCE_NORM_SCALAR_DESCRIPTOR,
        &[0.0, 1.0, 1e-5], // mean, var_val, eps
        (-5.0, 5.0),
        &TEST_X_9,
        "instance_norm",
    );
}

/// layer_norm_scalar(x, mean, var_val, eps, gamma, beta)
///   = (x - mean) * rsqrt(var_val + eps) * gamma + beta
/// for x ∈ [-5, 5], mean=0.0, var_val=1.0, eps=1e-5, gamma=1.0, beta=0.0.
/// Part of #506.
#[test]
fn test_layer_norm_gpu_output_within_verified_bounds() {
    let kernel_def = nn_dsl::build_layer_norm_scalar_kernel().expect("build layer_norm KernelDef");
    contract_harness::assert_scalar_contract(
        &kernel_def,
        &LAYER_NORM_SCALAR_DESCRIPTOR,
        &[0.0, 1.0, 1e-5, 1.0, 0.0], // mean, var_val, eps, gamma, beta
        (-5.0, 5.0),
        &TEST_X_9,
        "layer_norm",
    );
}

/// instance_norm_affine_scalar(x, mean, var_val, eps, gamma, beta)
///   = (x - mean) * rsqrt(var_val + eps) * gamma + beta
/// for x ∈ [-5, 5], mean=0.0, var_val=1.0, eps=1e-5, gamma=1.0, beta=0.0.
/// Part of #506.
#[test]
fn test_instance_norm_affine_gpu_output_within_verified_bounds() {
    let kernel_def = nn_dsl::build_instance_norm_affine_scalar_kernel()
        .expect("build instance_norm_affine KernelDef");
    contract_harness::assert_scalar_contract(
        &kernel_def,
        &INSTANCE_NORM_AFFINE_SCALAR_DESCRIPTOR,
        &[0.0, 1.0, 1e-5, 1.0, 0.0], // mean, var_val, eps, gamma, beta
        (-5.0, 5.0),
        &TEST_X_9,
        "instance_norm_affine",
    );
}

// ============================================================================
// Non-trivial parameter contract tests (#725)
//
// The identity-parameter tests above (mean=0, var=1, gamma=1, beta=0) mask
// codegen bugs where operations become no-ops. These tests use non-identity
// parameters so every arithmetic operation in the kernel is exercised.
// ============================================================================

/// instance_norm_scalar with non-trivial parameters:
/// mean=2.0, var_val=4.0, eps=1e-5
///
/// Computes: (x - 2.0) * rsqrt(4.0 + 1e-5) ≈ (x - 2.0) * 0.5
/// A codegen bug omitting `- mean` would produce x * 0.5 instead of (x-2)*0.5.
/// Part of #725.
#[test]
fn test_instance_norm_nontrivial_gpu_output_within_verified_bounds() {
    let kernel_def =
        nn_dsl::build_instance_norm_scalar_kernel().expect("build instance_norm KernelDef");
    contract_harness::assert_scalar_contract(
        &kernel_def,
        &INSTANCE_NORM_SCALAR_DESCRIPTOR,
        &[2.0, 4.0, 1e-5], // mean=2.0, var_val=4.0, eps=1e-5
        (-5.0, 5.0),
        &TEST_X_9,
        "instance_norm_nontrivial",
    );
}

/// layer_norm_scalar with non-trivial parameters:
/// mean=-1.0, var_val=2.0, eps=1e-5, gamma=0.5, beta=3.0
///
/// Computes: (x - (-1.0)) * rsqrt(2.0 + 1e-5) * 0.5 + 3.0
///         ≈ (x + 1.0) * 0.707 * 0.5 + 3.0
/// Every operation is exercised: subtraction (mean != 0), rsqrt (var != 1),
/// multiplication (gamma != 1), addition (beta != 0).
/// Part of #725.
#[test]
fn test_layer_norm_nontrivial_gpu_output_within_verified_bounds() {
    let kernel_def = nn_dsl::build_layer_norm_scalar_kernel().expect("build layer_norm KernelDef");
    contract_harness::assert_scalar_contract(
        &kernel_def,
        &LAYER_NORM_SCALAR_DESCRIPTOR,
        &[-1.0, 2.0, 1e-5, 0.5, 3.0], // mean=-1.0, var_val=2.0, eps=1e-5, gamma=0.5, beta=3.0
        (-5.0, 5.0),
        &TEST_X_9,
        "layer_norm_nontrivial",
    );
}

/// rms_norm_scalar with non-trivial parameters:
/// rms_inv=0.25, weight=2.0
///
/// Computes: x * 0.25 * 2.0 = x * 0.5
/// A codegen bug omitting either multiplication would change the output.
/// Part of #725.
#[test]
fn test_rms_norm_nontrivial_gpu_output_within_verified_bounds() {
    let kernel_def =
        nn_dsl::build_rms_norm_scalar_kernel().expect("build rms_norm_scalar KernelDef");
    contract_harness::assert_scalar_contract(
        &kernel_def,
        &RMS_NORM_SCALAR_DESCRIPTOR,
        &[0.25, 2.0], // rms_inv=0.25, weight=2.0
        (-5.0, 5.0),
        &TEST_X_9,
        "rms_norm_nontrivial",
    );
}
