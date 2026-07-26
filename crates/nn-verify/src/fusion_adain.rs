// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kernel-specific fusion equivalence wrappers.
//!
//! Each function here builds the `FusionSpec` for a specific kernel pair and
//! delegates to [`super::fusion::verify_fusion_equivalence`].

use crate::error::VerifyError;
use crate::fusion::{verify_fusion_equivalence, verify_fusion_equivalence_with_config};
use crate::fusion_spec::{FusionSpec, FusionVerification};
use crate::verify::VerifyConfig;

/// Convenience: build and verify the AdaIN+Snake (K4) fusion equivalence.
///
/// Shared inputs (7): x, mu, var, gamma, beta, alpha, eps
///
/// - K3 (AdaIN): params (x=0, mu=1, var=2, gamma=3, beta=4, eps=5)
///   → shared indices [0, 1, 2, 3, 4, 6]
/// - K1 (Snake): params (y=0, alpha=1)
///   → y comes from K3 output, alpha from shared index 5
/// - K4 (fused): params (x=0, mu=1, var=2, gamma=3, beta=4, alpha=5, eps=6)
///   → all 7 shared inputs
///
/// # Arguments
///
/// * `variable_bounds` — 7-element array: [x, mu, var, gamma, beta, alpha, eps]
/// * `epsilon` — Maximum tolerable absolute difference
///
/// # Errors
///
/// Returns [`VerifyError`] if `epsilon` is NaN, `variable_bounds` length is
/// not 7, kernel IR lowering fails, or bound propagation produces non-finite
/// results.
#[must_use = "fusion verification result is computed but not used"]
pub fn verify_adain_snake_fusion(
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
) -> Result<FusionVerification, VerifyError> {
    let fused = nn_dsl::build_adain_snake_fused_kernel()?;
    let adain = nn_dsl::build_adain_scalar_kernel()?;
    let snake = nn_dsl::build_snake_scalar_kernel()?;

    // K3 (adain) params: x, mu, var_val, gamma, beta, eps
    // Map to shared: [0, 1, 2, 3, 4, 6] (eps is at shared index 6)
    let first_param_indices = &[0, 1, 2, 3, 4, 6];

    // K1 (snake) params: y, alpha
    // y comes from K3 output (index 0), alpha from shared index 5
    let second_param_indices = &[0, 5]; // index 0 is ignored (replaced by first's output)
    let second_input_from_first = 0; // snake's param 0 (y) = adain output

    let spec = FusionSpec {
        fused: &fused,
        first: &adain,
        second: &snake,
        num_shared_inputs: 7,
        first_param_indices,
        second_param_indices,
        second_input_from_first,
    };
    verify_fusion_equivalence(&spec, variable_bounds, epsilon)
}

/// Like [`verify_adain_snake_fusion`] but with custom [`VerifyConfig`].
#[must_use = "fusion verification result is computed but not used"]
pub fn verify_adain_snake_fusion_with_config(
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
    config: &VerifyConfig,
) -> Result<FusionVerification, VerifyError> {
    let fused = nn_dsl::build_adain_snake_fused_kernel()?;
    let adain = nn_dsl::build_adain_scalar_kernel()?;
    let snake = nn_dsl::build_snake_scalar_kernel()?;

    let spec = FusionSpec {
        fused: &fused,
        first: &adain,
        second: &snake,
        num_shared_inputs: 7,
        first_param_indices: &[0, 1, 2, 3, 4, 6],
        second_param_indices: &[0, 5],
        second_input_from_first: 0,
    };
    verify_fusion_equivalence_with_config(&spec, variable_bounds, epsilon, config)
}

/// Convenience: build and verify the AdaIN+LeakyReLU fusion equivalence.
///
/// Shared inputs (7): x, mu, var, gamma, beta, slope, eps
///
/// - K3 (AdaIN): params (x=0, mu=1, var=2, gamma=3, beta=4, eps=5)
///   → shared indices [0, 1, 2, 3, 4, 6]
/// - LeakyReLU: params (y=0, slope=1)
///   → y comes from K3 output, slope from shared index 5
/// - Fused: params (x=0, mu=1, var=2, gamma=3, beta=4, slope=5, eps=6)
///   → all 7 shared inputs
///
/// # Arguments
///
/// * `variable_bounds` — 7-element array: [x, mu, var, gamma, beta, slope, eps]
/// * `epsilon` — Maximum tolerable absolute difference
///
/// # Errors
///
/// Returns [`VerifyError`] if `epsilon` is NaN, `variable_bounds` length is
/// not 7, kernel IR lowering fails, or bound propagation produces non-finite
/// results.
#[must_use = "fusion verification result is computed but not used"]
pub fn verify_adain_leaky_relu_fusion(
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
) -> Result<FusionVerification, VerifyError> {
    let fused = nn_dsl::build_adain_leaky_relu_fused_kernel()?;
    let adain = nn_dsl::build_adain_scalar_kernel()?;
    let leaky_relu = nn_dsl::build_leaky_relu_scalar_kernel()?;

    // K3 (adain) params: x, mu, var_val, gamma, beta, eps
    // Map to shared: [0, 1, 2, 3, 4, 6] (eps is at shared index 6)
    let first_param_indices = &[0, 1, 2, 3, 4, 6];

    // LeakyReLU params: y, slope
    // y comes from K3 output (index 0), slope from shared index 5
    let second_param_indices = &[0, 5]; // index 0 is ignored (replaced by first's output)
    let second_input_from_first = 0; // leaky_relu's param 0 (y) = adain output

    let spec = FusionSpec {
        fused: &fused,
        first: &adain,
        second: &leaky_relu,
        num_shared_inputs: 7,
        first_param_indices,
        second_param_indices,
        second_input_from_first,
    };
    verify_fusion_equivalence(&spec, variable_bounds, epsilon)
}

/// Like [`verify_adain_leaky_relu_fusion`] but with custom [`VerifyConfig`].
#[must_use = "fusion verification result is computed but not used"]
pub fn verify_adain_leaky_relu_fusion_with_config(
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
    config: &VerifyConfig,
) -> Result<FusionVerification, VerifyError> {
    let fused = nn_dsl::build_adain_leaky_relu_fused_kernel()?;
    let adain = nn_dsl::build_adain_scalar_kernel()?;
    let leaky_relu = nn_dsl::build_leaky_relu_scalar_kernel()?;

    let spec = FusionSpec {
        fused: &fused,
        first: &adain,
        second: &leaky_relu,
        num_shared_inputs: 7,
        first_param_indices: &[0, 1, 2, 3, 4, 6],
        second_param_indices: &[0, 5],
        second_input_from_first: 0,
    };
    verify_fusion_equivalence_with_config(&spec, variable_bounds, epsilon, config)
}

/// Convenience: build and verify the RMSNorm+SiLU-Mul fusion equivalence.
///
/// Shared inputs (4): x, rms_inv, weight, up
///
/// - RMSNorm scalar: params (x=0, rms_inv=1, weight=2)
///   → shared indices [0, 1, 2]
/// - SiLU-Mul: params (x=0, up=1)
///   → x comes from RMSNorm output, up from shared index 3
/// - Fused: params (x=0, rms_inv=1, weight=2, up=3)
///   → all 4 shared inputs
///
/// # Arguments
///
/// * `variable_bounds` — 4-element array: [x, rms_inv, weight, up]
/// * `epsilon` — Maximum tolerable absolute difference
///
/// # Errors
///
/// Returns [`VerifyError`] if `epsilon` is NaN, `variable_bounds` length is
/// not 4, kernel IR lowering fails, or bound propagation produces non-finite
/// results.
#[must_use = "fusion verification result is computed but not used"]
pub fn verify_rms_norm_silu_mul_fusion(
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
) -> Result<FusionVerification, VerifyError> {
    let fused = nn_dsl::build_rms_norm_silu_mul_fused_kernel()?;
    let rms_norm = nn_dsl::build_rms_norm_scalar_kernel()?;
    let silu_mul = nn_dsl::build_silu_mul_kernel()?;

    // RMSNorm scalar: params (x, rms_inv, weight)
    // Map to shared: [0, 1, 2]
    let first_param_indices = &[0, 1, 2];

    // SiLU-Mul: params (x, up)
    // x comes from RMSNorm output, up from shared index 3
    let second_param_indices = &[0, 3]; // index 0 is ignored (replaced by first's output)
    let second_input_from_first = 0; // silu_mul's param 0 (x) = rms_norm output

    let spec = FusionSpec {
        fused: &fused,
        first: &rms_norm,
        second: &silu_mul,
        num_shared_inputs: 4,
        first_param_indices,
        second_param_indices,
        second_input_from_first,
    };
    verify_fusion_equivalence(&spec, variable_bounds, epsilon)
}

/// Like [`verify_rms_norm_silu_mul_fusion`] but with custom [`VerifyConfig`].
#[must_use = "fusion verification result is computed but not used"]
pub fn verify_rms_norm_silu_mul_fusion_with_config(
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
    config: &VerifyConfig,
) -> Result<FusionVerification, VerifyError> {
    let fused = nn_dsl::build_rms_norm_silu_mul_fused_kernel()?;
    let rms_norm = nn_dsl::build_rms_norm_scalar_kernel()?;
    let silu_mul = nn_dsl::build_silu_mul_kernel()?;

    let spec = FusionSpec {
        fused: &fused,
        first: &rms_norm,
        second: &silu_mul,
        num_shared_inputs: 4,
        first_param_indices: &[0, 1, 2],
        second_param_indices: &[0, 3],
        second_input_from_first: 0,
    };
    verify_fusion_equivalence_with_config(&spec, variable_bounds, epsilon, config)
}

/// Convenience: build and verify the LayerNorm+GELU fusion equivalence.
///
/// Shared inputs (6): x, mean, var_val, eps, gamma, beta
///
/// - LayerNorm scalar: params (x=0, mean=1, var_val=2, eps=3, gamma=4, beta=5)
///   → shared indices [0, 1, 2, 3, 4, 5]
/// - GELU: params (x=0)
///   → x comes from LayerNorm output
/// - Fused: params (x=0, mean=1, var_val=2, eps=3, gamma=4, beta=5)
///   → all 6 shared inputs (same as LayerNorm, GELU adds no new params)
///
/// # Arguments
///
/// * `variable_bounds` — 6-element array: [x, mean, var_val, eps, gamma, beta]
/// * `epsilon` — Maximum tolerable absolute difference
///
/// # Errors
///
/// Returns [`VerifyError`] if `epsilon` is NaN, `variable_bounds` length is
/// not 6, kernel IR lowering fails, or bound propagation produces non-finite
/// results.
#[must_use = "fusion verification result is computed but not used"]
pub fn verify_layer_norm_gelu_fusion(
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
) -> Result<FusionVerification, VerifyError> {
    let fused = nn_dsl::build_layer_norm_gelu_fused_kernel()?;
    let layer_norm = nn_dsl::build_layer_norm_scalar_kernel()?;
    let gelu = nn_dsl::build_gelu_kernel()?;

    // LayerNorm scalar: params (x, mean, var_val, eps, gamma, beta)
    // Map to shared: [0, 1, 2, 3, 4, 5]
    let first_param_indices = &[0, 1, 2, 3, 4, 5];

    // GELU: params (x)
    // x comes from LayerNorm output
    let second_param_indices = &[0]; // index 0 is ignored (replaced by first's output)
    let second_input_from_first = 0; // gelu's param 0 (x) = layer_norm output

    let spec = FusionSpec {
        fused: &fused,
        first: &layer_norm,
        second: &gelu,
        num_shared_inputs: 6,
        first_param_indices,
        second_param_indices,
        second_input_from_first,
    };
    verify_fusion_equivalence(&spec, variable_bounds, epsilon)
}

/// Like [`verify_layer_norm_gelu_fusion`] but with custom [`VerifyConfig`].
#[must_use = "fusion verification result is computed but not used"]
pub fn verify_layer_norm_gelu_fusion_with_config(
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
    config: &VerifyConfig,
) -> Result<FusionVerification, VerifyError> {
    let fused = nn_dsl::build_layer_norm_gelu_fused_kernel()?;
    let layer_norm = nn_dsl::build_layer_norm_scalar_kernel()?;
    let gelu = nn_dsl::build_gelu_kernel()?;

    let spec = FusionSpec {
        fused: &fused,
        first: &layer_norm,
        second: &gelu,
        num_shared_inputs: 6,
        first_param_indices: &[0, 1, 2, 3, 4, 5],
        second_param_indices: &[0],
        second_input_from_first: 0,
    };
    verify_fusion_equivalence_with_config(&spec, variable_bounds, epsilon, config)
}

/// Convenience: build and verify the AdaLayerNorm fusion equivalence.
///
/// Shared inputs (8): x, mean, var_val, eps, norm_weight, norm_bias, gamma, beta
///
/// - LayerNorm scalar: params (x=0, mean=1, var_val=2, eps=3, gamma=4, beta=5)
///   → shared indices [0, 1, 2, 3, 4, 5]
/// - Adaptive affine: params (x=0, gamma=1, beta=2)
///   → x comes from LayerNorm output, gamma from shared 6, beta from shared 7
/// - Fused: params (x=0, mean=1, var_val=2, eps=3, norm_weight=4, norm_bias=5, gamma=6, beta=7)
///   → all 8 shared inputs
///
/// # Arguments
///
/// * `variable_bounds` — 8-element array: [x, mean, var_val, eps, norm_weight, norm_bias, gamma, beta]
/// * `epsilon` — Maximum tolerable absolute difference
///
/// # Errors
///
/// Returns [`VerifyError`] if `epsilon` is NaN, `variable_bounds` length is
/// not 8, kernel IR lowering fails, or bound propagation produces non-finite
/// results.
#[must_use = "fusion verification result is computed but not used"]
pub fn verify_ada_layer_norm_fusion(
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
) -> Result<FusionVerification, VerifyError> {
    let fused = nn_dsl::build_ada_layer_norm_fused_kernel()?;
    let layer_norm = nn_dsl::build_layer_norm_scalar_kernel()?;
    let adaptive_affine = nn_dsl::build_adaptive_affine_kernel()?;

    // LayerNorm scalar: params (x, mean, var_val, eps, gamma, beta)
    // Map to shared: [0, 1, 2, 3, 4, 5]
    // Note: LayerNorm's "gamma" and "beta" are norm_weight and norm_bias in AdaLN context
    let first_param_indices = &[0, 1, 2, 3, 4, 5];

    // Adaptive affine: params (x, gamma, beta)
    // x comes from LayerNorm output, gamma from shared 6, beta from shared 7
    let second_param_indices = &[0, 6, 7]; // index 0 is ignored (replaced by first's output)
    let second_input_from_first = 0; // adaptive_affine's param 0 (x) = layer_norm output

    let spec = FusionSpec {
        fused: &fused,
        first: &layer_norm,
        second: &adaptive_affine,
        num_shared_inputs: 8,
        first_param_indices,
        second_param_indices,
        second_input_from_first,
    };
    verify_fusion_equivalence(&spec, variable_bounds, epsilon)
}

/// Like [`verify_ada_layer_norm_fusion`] but with custom [`VerifyConfig`].
#[must_use = "fusion verification result is computed but not used"]
pub fn verify_ada_layer_norm_fusion_with_config(
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
    config: &VerifyConfig,
) -> Result<FusionVerification, VerifyError> {
    let fused = nn_dsl::build_ada_layer_norm_fused_kernel()?;
    let layer_norm = nn_dsl::build_layer_norm_scalar_kernel()?;
    let adaptive_affine = nn_dsl::build_adaptive_affine_kernel()?;

    let spec = FusionSpec {
        fused: &fused,
        first: &layer_norm,
        second: &adaptive_affine,
        num_shared_inputs: 8,
        first_param_indices: &[0, 1, 2, 3, 4, 5],
        second_param_indices: &[0, 6, 7],
        second_input_from_first: 0,
    };
    verify_fusion_equivalence_with_config(&spec, variable_bounds, epsilon, config)
}

// ---------------------------------------------------------------------------
// Named fusion registry (#2931 D5)
// ---------------------------------------------------------------------------

/// Per-fusion bound arrays for [`verify_all_named_fusions`].
pub struct NamedFusionBounds<'a> {
    /// AdaIN+Snake bounds (7): x, mu, var, gamma, beta, alpha, eps
    pub adain_snake: &'a [(f32, f32)],
    /// AdaIN+LeakyReLU bounds (7): x, mu, var, gamma, beta, slope, eps
    pub adain_leaky_relu: &'a [(f32, f32)],
    /// RMSNorm+SiLU-Mul bounds (4): x, rms_inv, weight, up
    pub rms_norm_silu_mul: &'a [(f32, f32)],
    /// LayerNorm+GELU bounds (6): x, mean, var_val, eps, gamma, beta
    pub layer_norm_gelu: &'a [(f32, f32)],
    /// AdaLayerNorm bounds (8): x, mean, var_val, eps, norm_weight, norm_bias, gamma, beta
    pub ada_layer_norm: &'a [(f32, f32)],
}

/// Verify ALL named fusion pairs in a single call.
///
/// Returns a vec of `(name, result)` for each fusion. Workers and CI can call
/// this to check that no fusion has regressed after code changes.
pub fn verify_all_named_fusions(
    bounds: &NamedFusionBounds<'_>,
) -> Vec<(&'static str, Result<FusionVerification, VerifyError>)> {
    vec![
        (
            "adain_snake",
            verify_adain_snake_fusion(bounds.adain_snake, f32::MAX),
        ),
        (
            "adain_leaky_relu",
            verify_adain_leaky_relu_fusion(bounds.adain_leaky_relu, f32::MAX),
        ),
        (
            "rms_norm_silu_mul",
            verify_rms_norm_silu_mul_fusion(bounds.rms_norm_silu_mul, f32::MAX),
        ),
        (
            "layer_norm_gelu",
            verify_layer_norm_gelu_fusion(bounds.layer_norm_gelu, f32::MAX),
        ),
        (
            "ada_layer_norm",
            verify_ada_layer_norm_fusion(bounds.ada_layer_norm, f32::MAX),
        ),
    ]
}
