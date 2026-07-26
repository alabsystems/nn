// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! NormActivConv1d fusion equivalence verification (#2218 F13).
//!
//! Proves that the fused per-tap computation inside `fused_norm_conv1d_*`
//! GPU kernels matches the sequential composition of InstanceNorm + affine +
//! activation + weight multiply.
//!
//! Since Conv1d is a linear operation (sum of per-tap contributions),
//! per-tap scalar equivalence implies full kernel equivalence.
//!
//! Two variants:
//! - LeakyReLU: `norm → affine → leaky_relu → weight_mul`
//! - Snake: `norm → affine → snake → weight_mul`
//!
//! Both use the Kokoro residual gamma convention: `(1 + gamma) * normed + beta`.

use crate::error::VerifyError;
use crate::fusion::{verify_fusion_equivalence, verify_fusion_equivalence_with_config};
use crate::fusion_spec::{FusionSpec, FusionVerification};
use crate::verify::VerifyConfig;

/// Verify NormActivConv1d per-tap fusion equivalence with LeakyReLU activation.
///
/// Proves: `norm_leaky_relu_mul(x, mean, inv_std, gamma, beta, slope, weight)`
/// equals `weight_mul(norm_leaky_relu(x, mean, inv_std, gamma, beta, slope), weight)`
/// for all inputs within `variable_bounds`.
///
/// Shared inputs (7): x, mean, inv_std, gamma, beta, slope, weight
///
/// - First kernel (norm_leaky_relu): params (x=0, mean=1, inv_std=2, gamma=3, beta=4, slope=5)
///   → shared indices [0, 1, 2, 3, 4, 5]
/// - Second kernel (weight_mul): params (y=0, weight=1)
///   → y from first's output, weight from shared index 6
/// - Fused kernel: params (x=0, mean=1, inv_std=2, gamma=3, beta=4, slope=5, weight=6)
///   → all 7 shared inputs
///
/// # Arguments
///
/// * `variable_bounds` — 7-element array: [x, mean, inv_std, gamma, beta, slope, weight]
/// * `epsilon` — Maximum tolerable absolute difference
///
/// # Errors
///
/// Returns [`VerifyError`] if bounds count is not 7, kernel IR fails, or
/// propagation produces non-finite results.
#[must_use = "fusion verification result is computed but not used"]
pub fn verify_norm_activ_conv1d_leaky_relu_fusion(
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
) -> Result<FusionVerification, VerifyError> {
    let fused = nn_dsl::build_norm_leaky_relu_mul_fused_kernel()?;
    let norm_activate = nn_dsl::build_norm_leaky_relu_kernel()?;
    let weight_mul = nn_dsl::build_weight_mul_kernel()?;

    let spec = FusionSpec {
        fused: &fused,
        first: &norm_activate,
        second: &weight_mul,
        num_shared_inputs: 7,
        first_param_indices: &[0, 1, 2, 3, 4, 5],
        second_param_indices: &[0, 6], // y from first output, weight from shared 6
        second_input_from_first: 0,
    };
    verify_fusion_equivalence(&spec, variable_bounds, epsilon)
}

/// Like [`verify_norm_activ_conv1d_leaky_relu_fusion`] but with custom [`VerifyConfig`].
#[must_use = "fusion verification result is computed but not used"]
pub fn verify_norm_activ_conv1d_leaky_relu_fusion_with_config(
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
    config: &VerifyConfig,
) -> Result<FusionVerification, VerifyError> {
    let fused = nn_dsl::build_norm_leaky_relu_mul_fused_kernel()?;
    let norm_activate = nn_dsl::build_norm_leaky_relu_kernel()?;
    let weight_mul = nn_dsl::build_weight_mul_kernel()?;

    let spec = FusionSpec {
        fused: &fused,
        first: &norm_activate,
        second: &weight_mul,
        num_shared_inputs: 7,
        first_param_indices: &[0, 1, 2, 3, 4, 5],
        second_param_indices: &[0, 6],
        second_input_from_first: 0,
    };
    verify_fusion_equivalence_with_config(&spec, variable_bounds, epsilon, config)
}

/// Verify NormActivConv1d per-tap fusion equivalence with Snake activation.
///
/// Proves: `norm_snake_mul(x, mean, inv_std, gamma, beta, alpha, weight)`
/// equals `weight_mul(norm_snake(x, mean, inv_std, gamma, beta, alpha), weight)`
/// for all inputs within `variable_bounds`.
///
/// Shared inputs (7): x, mean, inv_std, gamma, beta, alpha, weight
///
/// - First kernel (norm_snake): params (x=0, mean=1, inv_std=2, gamma=3, beta=4, alpha=5)
///   → shared indices [0, 1, 2, 3, 4, 5]
/// - Second kernel (weight_mul): params (y=0, weight=1)
///   → y from first's output, weight from shared index 6
/// - Fused kernel: params (x=0, mean=1, inv_std=2, gamma=3, beta=4, alpha=5, weight=6)
///   → all 7 shared inputs
///
/// # Arguments
///
/// * `variable_bounds` — 7-element array: [x, mean, inv_std, gamma, beta, alpha, weight]
/// * `epsilon` — Maximum tolerable absolute difference
///
/// # Errors
///
/// Returns [`VerifyError`] if bounds count is not 7, kernel IR fails, or
/// propagation produces non-finite results.
#[must_use = "fusion verification result is computed but not used"]
pub fn verify_norm_activ_conv1d_snake_fusion(
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
) -> Result<FusionVerification, VerifyError> {
    let fused = nn_dsl::build_norm_snake_mul_fused_kernel()?;
    let norm_activate = nn_dsl::build_norm_snake_kernel()?;
    let weight_mul = nn_dsl::build_weight_mul_kernel()?;

    let spec = FusionSpec {
        fused: &fused,
        first: &norm_activate,
        second: &weight_mul,
        num_shared_inputs: 7,
        first_param_indices: &[0, 1, 2, 3, 4, 5],
        second_param_indices: &[0, 6],
        second_input_from_first: 0,
    };
    verify_fusion_equivalence(&spec, variable_bounds, epsilon)
}

/// Like [`verify_norm_activ_conv1d_snake_fusion`] but with custom [`VerifyConfig`].
#[must_use = "fusion verification result is computed but not used"]
pub fn verify_norm_activ_conv1d_snake_fusion_with_config(
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
    config: &VerifyConfig,
) -> Result<FusionVerification, VerifyError> {
    let fused = nn_dsl::build_norm_snake_mul_fused_kernel()?;
    let norm_activate = nn_dsl::build_norm_snake_kernel()?;
    let weight_mul = nn_dsl::build_weight_mul_kernel()?;

    let spec = FusionSpec {
        fused: &fused,
        first: &norm_activate,
        second: &weight_mul,
        num_shared_inputs: 7,
        first_param_indices: &[0, 1, 2, 3, 4, 5],
        second_param_indices: &[0, 6],
        second_input_from_first: 0,
    };
    verify_fusion_equivalence_with_config(&spec, variable_bounds, epsilon, config)
}
