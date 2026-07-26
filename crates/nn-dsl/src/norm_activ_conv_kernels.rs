// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! NormActivConv1d per-tap scalar kernels for fusion equivalence verification.
//!
//! The fused GPU kernel `fused_norm_conv1d_{leaky_relu,snake}_f32` computes
//! InstanceNorm + affine + activation inline during Conv1d accumulation.
//! Since Conv1d is linear (sum of per-tap contributions), proving per-tap
//! scalar equivalence is sufficient for the full kernel.
//!
//! Per-tap fused computation (LeakyReLU variant):
//! ```text
//! normed = (x - mean) * inv_std
//! y = (1 + gamma) * normed + beta   // Kokoro residual gamma convention
//! activated = if y >= 0 { y } else { slope * y }
//! contribution = activated * weight
//! ```
//!
//! Note: uses `(1 + gamma)` not `gamma` — matches Kokoro's style affine
//! convention where gamma is a residual scale parameter.
//!
//! Part of #2218 F13: NormActivConv1d fusion equivalence proof.

use crate::ir::KernelDef;
use crate::kernel_error::KernelError;
use crate::kernel_util::{build_scalar_kernel, checked_scalar_output, validate_finite_inputs};
use crate::lower::LowerError;
use crate::snake::SNAKE_MIN_ALPHA;

/// Build the fused per-tap kernel for NormActivConv1d with LeakyReLU.
///
/// Parameters (7): x, mean, inv_std, gamma, beta, slope, weight.
/// Computes: `leaky_relu((1+gamma) * (x-mean) * inv_std + beta, slope) * weight`
///
/// # Errors
///
/// Returns [`LowerError`] if the kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_norm_leaky_relu_mul_fused_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn norm_leaky_relu_mul(x: f32, mean: f32, inv_std: f32, gamma: f32, beta: f32, slope: f32, weight: f32) -> f32 {
            let normed = (x - mean) * inv_std;
            let y = (1.0 + gamma) * normed + beta;
            let activated = if y >= 0.0 { y } else { slope * y };
            activated * weight
        }",
    )
}

/// Build the sequential first kernel: InstanceNorm + affine + LeakyReLU.
///
/// Parameters (6): x, mean, inv_std, gamma, beta, slope.
/// Computes: `leaky_relu((1+gamma) * (x-mean) * inv_std + beta, slope)`
///
/// # Errors
///
/// Returns [`LowerError`] if the kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_norm_leaky_relu_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn norm_leaky_relu(x: f32, mean: f32, inv_std: f32, gamma: f32, beta: f32, slope: f32) -> f32 {
            let normed = (x - mean) * inv_std;
            let y = (1.0 + gamma) * normed + beta;
            if y >= 0.0 { y } else { slope * y }
        }",
    )
}

/// Build the fused per-tap kernel for NormActivConv1d with Snake activation.
///
/// Parameters (7): x, mean, inv_std, gamma, beta, alpha, weight.
/// Computes: `snake((1+gamma) * (x-mean) * inv_std + beta, alpha) * weight`
///
/// # Errors
///
/// Returns [`LowerError`] if the kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_norm_snake_mul_fused_kernel() -> Result<KernelDef, LowerError> {
    let src = format!(
        "fn norm_snake_mul(x: f32, mean: f32, inv_std: f32, gamma: f32, beta: f32, alpha: f32, weight: f32) -> f32 {{
            let normed = (x - mean) * inv_std;
            let y = (1.0 + gamma) * normed + beta;
            let a = alpha.max({SNAKE_MIN_ALPHA:e});
            let activated = y + (1.0 / a) * (a * y).sin().powi(2);
            activated * weight
        }}"
    );
    build_scalar_kernel(&src)
}

/// Build the sequential first kernel: InstanceNorm + affine + Snake.
///
/// Parameters (6): x, mean, inv_std, gamma, beta, alpha.
/// Computes: `snake((1+gamma) * (x-mean) * inv_std + beta, alpha)`
///
/// # Errors
///
/// Returns [`LowerError`] if the kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_norm_snake_kernel() -> Result<KernelDef, LowerError> {
    let src = format!(
        "fn norm_snake(x: f32, mean: f32, inv_std: f32, gamma: f32, beta: f32, alpha: f32) -> f32 {{
            let normed = (x - mean) * inv_std;
            let y = (1.0 + gamma) * normed + beta;
            let a = alpha.max({SNAKE_MIN_ALPHA:e});
            y + (1.0 / a) * (a * y).sin().powi(2)
        }}"
    );
    build_scalar_kernel(&src)
}

/// Build the weight-multiply kernel (Conv1d per-tap weight application).
///
/// Parameters (2): y, weight.
/// Computes: `y * weight`
///
/// # Errors
///
/// Returns [`LowerError`] if the kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_weight_mul_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn weight_mul(y: f32, weight: f32) -> f32 {
            y * weight
        }",
    )
}

// --- Scalar reference implementations for Kani verification ---

/// Fused NormActivConv1d+LeakyReLU per-tap scalar reference.
///
/// Computes: `leaky_relu((1+gamma) * (x-mean) * inv_std + beta, slope) * weight`
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is non-finite.
/// Returns [`KernelError::NonFiniteOutput`] if the result overflows.
#[must_use = "returns a Result that may contain an error"]
pub fn norm_leaky_relu_mul_fused_scalar(
    x: f32,
    mean: f32,
    inv_std: f32,
    gamma: f32,
    beta: f32,
    slope: f32,
    weight: f32,
) -> Result<f32, KernelError> {
    validate_finite_inputs(&[
        ("x", x),
        ("mean", mean),
        ("inv_std", inv_std),
        ("gamma", gamma),
        ("beta", beta),
        ("slope", slope),
        ("weight", weight),
    ])?;
    let normed = (x - mean) * inv_std;
    let y = (1.0 + gamma) * normed + beta;
    let activated = if y >= 0.0 { y } else { slope * y };
    checked_scalar_output(activated * weight)
}

/// Sequential NormActivConv1d LeakyReLU scalar reference (step 1 of 2).
///
/// Computes: `leaky_relu((1+gamma) * (x-mean) * inv_std + beta, slope)`
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is non-finite.
/// Returns [`KernelError::NonFiniteOutput`] if the result overflows.
#[must_use = "returns a Result that may contain an error"]
pub fn norm_leaky_relu_scalar(
    x: f32,
    mean: f32,
    inv_std: f32,
    gamma: f32,
    beta: f32,
    slope: f32,
) -> Result<f32, KernelError> {
    validate_finite_inputs(&[
        ("x", x),
        ("mean", mean),
        ("inv_std", inv_std),
        ("gamma", gamma),
        ("beta", beta),
        ("slope", slope),
    ])?;
    let normed = (x - mean) * inv_std;
    let y = (1.0 + gamma) * normed + beta;
    let result = if y >= 0.0 { y } else { slope * y };
    checked_scalar_output(result)
}

/// Weight-multiply scalar reference (step 2 of 2, Conv1d per-tap weight).
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if either input is non-finite.
/// Returns [`KernelError::NonFiniteOutput`] if the result overflows.
#[must_use = "returns a Result that may contain an error"]
pub fn weight_mul_scalar(y: f32, weight: f32) -> Result<f32, KernelError> {
    validate_finite_inputs(&[("y", y), ("weight", weight)])?;
    checked_scalar_output(y * weight)
}

/// Fused NormActivConv1d+Snake per-tap scalar reference.
///
/// Computes: `snake((1+gamma) * (x-mean) * inv_std + beta, alpha) * weight`
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is non-finite.
/// Returns [`KernelError::NonFiniteOutput`] if the result overflows.
#[must_use = "returns a Result that may contain an error"]
pub fn norm_snake_mul_fused_scalar(
    x: f32,
    mean: f32,
    inv_std: f32,
    gamma: f32,
    beta: f32,
    alpha: f32,
    weight: f32,
) -> Result<f32, KernelError> {
    validate_finite_inputs(&[
        ("x", x),
        ("mean", mean),
        ("inv_std", inv_std),
        ("gamma", gamma),
        ("beta", beta),
        ("alpha", alpha),
        ("weight", weight),
    ])?;
    let normed = (x - mean) * inv_std;
    let y = (1.0 + gamma) * normed + beta;
    let a = alpha.max(SNAKE_MIN_ALPHA);
    let sin_val = (a * y).sin();
    let activated = y + (1.0 / a) * sin_val * sin_val;
    checked_scalar_output(activated * weight)
}

/// Sequential NormActivConv1d Snake scalar reference (step 1 of 2).
///
/// Computes: `snake((1+gamma) * (x-mean) * inv_std + beta, alpha)`
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is non-finite.
/// Returns [`KernelError::NonFiniteOutput`] if the result overflows.
#[must_use = "returns a Result that may contain an error"]
pub fn norm_snake_scalar(
    x: f32,
    mean: f32,
    inv_std: f32,
    gamma: f32,
    beta: f32,
    alpha: f32,
) -> Result<f32, KernelError> {
    validate_finite_inputs(&[
        ("x", x),
        ("mean", mean),
        ("inv_std", inv_std),
        ("gamma", gamma),
        ("beta", beta),
        ("alpha", alpha),
    ])?;
    let normed = (x - mean) * inv_std;
    let y = (1.0 + gamma) * normed + beta;
    let a = alpha.max(SNAKE_MIN_ALPHA);
    let sin_val = (a * y).sin();
    checked_scalar_output(y + (1.0 / a) * sin_val * sin_val)
}

#[cfg(kani)]
#[path = "norm_activ_conv_leaky_relu_kani.rs"]
mod kani_leaky_relu_proofs;

#[cfg(kani)]
#[path = "norm_activ_conv_snake_kani.rs"]
mod kani_snake_proofs;

#[cfg(test)]
#[path = "norm_activ_conv_kernels_tests.rs"]
mod tests;
