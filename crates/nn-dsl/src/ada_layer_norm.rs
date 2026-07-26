// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! AdaLayerNorm (AdaLN) kernel builders and reference implementations.
//!
//! AdaLN combines LayerNorm with adaptive affine modulation, used in the
//! Kokoro ProsodyPredictor and diffusion-based models:
//!
//! ```text
//! normed = (x - mean) * rsqrt(var + eps) * norm_weight + norm_bias
//! output = (1 + gamma) * normed + beta
//! ```
//!
//! The fused kernel computes both steps in one pass. The sequential path
//! runs LayerNorm (K7) followed by adaptive affine.
//!
//! # Naming convention (#336)
//!
//! - `adaptive_affine_scalar` — per-element scalar, `Result<f32, KernelError>`
//! - `ada_layer_norm_fused_scalar` — fused scalar reference, `Result<f32, KernelError>`
//! - `build_adaptive_affine_kernel` / `build_ada_layer_norm_fused_kernel` — `KernelDef` IR builders
//!
//! Part of #2714, #2701, #2218.

use crate::ir::KernelDef;
use crate::kernel_error::KernelError;
use crate::kernel_util::{build_scalar_kernel, checked_scalar_output, validate_finite_inputs};
use crate::lower::LowerError;

/// Build the adaptive affine scalar `KernelDef`.
///
/// Parameters: `x`, `gamma`, `beta` (3 params).
/// Computes: `(1.0 + gamma) * x + beta`
///
/// This is the second step of AdaLN after LayerNorm normalizes the input.
///
/// # Errors
///
/// Returns [`LowerError`] if the hardcoded kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_adaptive_affine_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn adaptive_affine(x: f32, gamma: f32, beta: f32) -> f32 {
            (1.0 + gamma) * x + beta
        }",
    )
}

/// Build the fused AdaLayerNorm scalar `KernelDef`.
///
/// Parameters: `x`, `mean`, `var_val`, `eps`, `norm_weight`, `norm_bias`,
///             `gamma`, `beta` (8 params).
/// Computes: `(1 + gamma) * ((x - mean) * rsqrt(var_val + eps) * norm_weight + norm_bias) + beta`
///
/// # Errors
///
/// Returns [`LowerError`] if the hardcoded kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_ada_layer_norm_fused_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn ada_layer_norm(x: f32, mean: f32, var_val: f32, eps: f32, norm_weight: f32, norm_bias: f32, gamma: f32, beta: f32) -> f32 {
            let normed = (x - mean) * (var_val + eps).rsqrt() * norm_weight + norm_bias;
            (1.0 + gamma) * normed + beta
        }",
    )
}

/// Reference implementation for adaptive affine scalar.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is NaN or infinite.
/// Returns [`KernelError::NonFiniteOutput`] if the computed result is non-finite.
#[must_use = "returns a Result that may contain an error"]
pub fn adaptive_affine_scalar(x: f32, gamma: f32, beta: f32) -> Result<f32, KernelError> {
    validate_finite_inputs(&[("x", x), ("gamma", gamma), ("beta", beta)])?;
    let result = (1.0 + gamma) * x + beta;
    checked_scalar_output(result)
}

/// Reference implementation for fused AdaLayerNorm scalar.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is NaN or infinite.
/// Returns [`KernelError::InvalidEps`] if `var_val + eps <= 0`.
/// Returns [`KernelError::NonFiniteOutput`] if the computed result is non-finite.
#[must_use = "returns a Result that may contain an error"]
pub fn ada_layer_norm_fused_scalar(
    x: f32,
    mean: f32,
    var_val: f32,
    eps: f32,
    norm_weight: f32,
    norm_bias: f32,
    gamma: f32,
    beta: f32,
) -> Result<f32, KernelError> {
    validate_finite_inputs(&[
        ("x", x),
        ("mean", mean),
        ("var_val", var_val),
        ("eps", eps),
        ("norm_weight", norm_weight),
        ("norm_bias", norm_bias),
        ("gamma", gamma),
        ("beta", beta),
    ])?;

    let denom_input = var_val + eps;
    if denom_input <= 0.0 {
        return Err(KernelError::InvalidEps { value: eps });
    }

    let normed = (x - mean) * denom_input.sqrt().recip() * norm_weight + norm_bias;
    let result = (1.0 + gamma) * normed + beta;
    checked_scalar_output(result)
}

#[cfg(kani)]
#[path = "ada_layer_norm_kani_tests.rs"]
mod kani_proofs;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_affine_kernel_builds() {
        let kernel = build_adaptive_affine_kernel().expect("kernel should build");
        assert_eq!(kernel.params.len(), 3);
    }

    #[test]
    fn test_ada_layer_norm_fused_kernel_builds() {
        let kernel = build_ada_layer_norm_fused_kernel().expect("kernel should build");
        assert_eq!(kernel.params.len(), 8);
    }

    #[test]
    fn test_adaptive_affine_scalar_nominal() {
        // (1 + 0.5) * 2.0 + 0.1 = 3.1
        let result = adaptive_affine_scalar(2.0, 0.5, 0.1).expect("should succeed");
        assert!((result - 3.1).abs() < 1e-6);
    }

    #[test]
    fn test_ada_layer_norm_fused_scalar_nominal() {
        // normed = (1.0 - 0.0) * rsqrt(1.0 + 1e-5) * 1.0 + 0.0 ≈ 1.0
        // output = (1 + 0.0) * 1.0 + 0.0 = 1.0
        let result = ada_layer_norm_fused_scalar(1.0, 0.0, 1.0, 1e-5, 1.0, 0.0, 0.0, 0.0)
            .expect("should succeed");
        assert!((result - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_ada_layer_norm_fused_scalar_nan_rejected() {
        let err = ada_layer_norm_fused_scalar(f32::NAN, 0.0, 1.0, 1e-5, 1.0, 0.0, 0.0, 0.0)
            .expect_err("NaN should be rejected");
        assert!(matches!(err, KernelError::NonFiniteInput { .. }));
    }

    #[test]
    fn test_ada_layer_norm_fused_scalar_invalid_eps() {
        let err = ada_layer_norm_fused_scalar(1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0)
            .expect_err("invalid eps should fail");
        assert!(matches!(err, KernelError::InvalidEps { .. }));
    }
}
