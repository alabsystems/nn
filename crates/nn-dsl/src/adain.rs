// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! AdaIN (K3) and fused AdaIN+Snake (K4) kernel builders and reference
//! implementations.
//!
//! These are scalar-level kernels for the per-element computation after
//! mean/variance reduction. The reduction (InstanceNorm K2) is built
//! separately via [`super::instance_norm::build_instance_norm`].
//!
//! # AdaIN formula (K3 per-element)
//!
//! ```text
//! y = gamma * (x - mu) * rsqrt(var + eps) + beta
//! ```
//!
//! # Fused AdaIN+Snake formula (K4 per-element)
//!
//! ```text
//! y = gamma * (x - mu) * rsqrt(var + eps) + beta
//! a = max(alpha, SNAKE_MIN_ALPHA)
//! out = y + (1/a) * sin²(a * y)
//! ```
//!
//! # Naming convention (#336)
//!
//! - `adain_scalar` — per-element scalar, `Result<f32, KernelError>`
//! - `adain_snake_fused_scalar` — fused scalar reference, `Result<f32, KernelError>`
//! - `build_adain_scalar_kernel` / `build_adain_snake_fused_kernel` — `KernelDef` IR builders

use crate::ir::KernelDef;
use crate::kernel_error::KernelError;
use crate::kernel_util::{
    build_scalar_kernel, checked_scalar_output, validate_finite_inputs, validate_nonzero_dims,
};
use crate::lower::LowerError;
use crate::snake::SNAKE_MIN_ALPHA;
use crate::tensor_builders::{broadcast_node, elementwise_node, input_node};
use crate::tensor_ir::{
    BroadcastAlignment, TensorIRError, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};

/// Build the AdaIN (K3) scalar KernelDef.
///
/// Parameters: `x`, `mu`, `var_val`, `gamma`, `beta`, `eps` (6 params).
/// Computes: `gamma * (x - mu) * rsqrt(var_val + eps) + beta`
///
/// # Errors
///
/// Returns [`LowerError`] if the hardcoded kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_adain_scalar_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn adain(x: f32, mu: f32, var_val: f32, gamma: f32, beta: f32, eps: f32) -> f32 {
            gamma * (x - mu) * (var_val + eps).rsqrt() + beta
        }",
    )
}

/// Build the Snake (K1) scalar KernelDef.
///
/// Parameters: `y`, `alpha` (2 params).
/// Computes: `a = max(alpha, SNAKE_MIN_ALPHA); y + (1.0 / a) * sin²(a * y)`
///
/// # Errors
///
/// Returns [`LowerError`] if the hardcoded kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_snake_scalar_kernel() -> Result<KernelDef, LowerError> {
    let src = format!(
        "fn snake(y: f32, alpha: f32) -> f32 {{
            let a = alpha.max({SNAKE_MIN_ALPHA:e});
            y + (1.0 / a) * (a * y).sin().powi(2)
        }}"
    );
    build_scalar_kernel(&src)
}

/// Build the fused AdaIN+Snake (K4) scalar KernelDef.
///
/// Parameters: `x`, `mu`, `var_val`, `gamma`, `beta`, `alpha`, `eps` (7 params).
/// Computes AdaIN then Snake in a single kernel.
///
/// # Errors
///
/// Returns [`LowerError`] if the hardcoded kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_adain_snake_fused_kernel() -> Result<KernelDef, LowerError> {
    let src = format!(
        "fn adain_snake(x: f32, mu: f32, var_val: f32, gamma: f32, beta: f32, alpha: f32, eps: f32) -> f32 {{
            let y = gamma * (x - mu) * (var_val + eps).rsqrt() + beta;
            let a = alpha.max({SNAKE_MIN_ALPHA:e});
            y + (1.0 / a) * (a * y).sin().powi(2)
        }}"
    );
    build_scalar_kernel(&src)
}

/// Reference implementation for AdaIN scalar.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any of the 6 scalar inputs
/// (`x`, `mu`, `var_val`, `gamma`, `beta`, `eps`) is NaN or infinite.
/// Returns [`KernelError::InvalidEps`] if `var_val + eps <= 0`.
/// Returns [`KernelError::NonFiniteOutput`] if the computed result is non-finite
/// despite all inputs being finite (e.g., extreme magnitudes).
#[must_use = "returns a Result that may contain an error"]
pub fn adain_scalar(
    x: f32,
    mu: f32,
    var_val: f32,
    gamma: f32,
    beta: f32,
    eps: f32,
) -> Result<f32, KernelError> {
    validate_finite_inputs(&[
        ("x", x),
        ("mu", mu),
        ("var_val", var_val),
        ("gamma", gamma),
        ("beta", beta),
        ("eps", eps),
    ])?;

    let denom_input = var_val + eps;
    if denom_input <= 0.0 {
        return Err(KernelError::InvalidEps { value: eps });
    }

    let result = gamma * (x - mu) * denom_input.sqrt().recip() + beta;
    checked_scalar_output(result)
}

/// Reference implementation for fused AdaIN+Snake scalar.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if `alpha` is NaN or infinite.
/// Propagates errors from [`adain_scalar`] for the other 6 parameters.
/// Returns [`KernelError::NonFiniteOutput`] if the fused result is non-finite
/// despite all inputs being finite.
#[must_use = "returns a Result that may contain an error"]
pub fn adain_snake_fused_scalar(
    x: f32,
    mu: f32,
    var_val: f32,
    gamma: f32,
    beta: f32,
    alpha: f32,
    eps: f32,
) -> Result<f32, KernelError> {
    validate_finite_inputs(&[("alpha", alpha)])?;
    let y = adain_scalar(x, mu, var_val, gamma, beta, eps)?;
    let a = alpha.max(SNAKE_MIN_ALPHA);
    let sin_val = (a * y).sin();
    let result = y + (1.0 / a) * sin_val * sin_val;
    checked_scalar_output(result)
}

/// Build the LeakyReLU scalar KernelDef.
///
/// Parameters: `x`, `slope` (2 params).
/// Computes: `if x >= 0.0 { x } else { slope * x }`
///
/// # Errors
///
/// Returns [`LowerError`] if the hardcoded kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_leaky_relu_scalar_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn leaky_relu(x: f32, slope: f32) -> f32 {
            if x >= 0.0 { x } else { slope * x }
        }",
    )
}

/// Build the fused AdaIN+LeakyReLU scalar KernelDef.
///
/// Parameters: `x`, `mu`, `var_val`, `gamma`, `beta`, `slope`, `eps` (7 params).
/// Computes AdaIN then LeakyReLU in a single kernel.
///
/// # Errors
///
/// Returns [`LowerError`] if the hardcoded kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_adain_leaky_relu_fused_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn adain_leaky_relu(x: f32, mu: f32, var_val: f32, gamma: f32, beta: f32, slope: f32, eps: f32) -> f32 {
            let y = gamma * (x - mu) * (var_val + eps).rsqrt() + beta;
            if y >= 0.0 { y } else { slope * y }
        }",
    )
}

/// Reference implementation for LeakyReLU scalar.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if `x` or `slope` is NaN or infinite.
/// Returns [`KernelError::NonFiniteOutput`] if the result is non-finite.
#[must_use = "returns a Result that may contain an error"]
pub fn leaky_relu_scalar(x: f32, slope: f32) -> Result<f32, KernelError> {
    validate_finite_inputs(&[("x", x), ("slope", slope)])?;
    let result = if x >= 0.0 { x } else { slope * x };
    checked_scalar_output(result)
}

/// Reference implementation for fused AdaIN+LeakyReLU scalar.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if `slope` is NaN or infinite.
/// Propagates errors from [`adain_scalar`] for the other 6 parameters.
/// Returns [`KernelError::NonFiniteOutput`] if the fused result is non-finite.
#[must_use = "returns a Result that may contain an error"]
pub fn adain_leaky_relu_fused_scalar(
    x: f32,
    mu: f32,
    var_val: f32,
    gamma: f32,
    beta: f32,
    slope: f32,
    eps: f32,
) -> Result<f32, KernelError> {
    validate_finite_inputs(&[("slope", slope)])?;
    let y = adain_scalar(x, mu, var_val, gamma, beta, eps)?;
    let result = if y >= 0.0 { y } else { slope * y };
    checked_scalar_output(result)
}

/// Build the AdaIN1d `TensorKernelDef` using the native `AdaIN1d` op.
///
/// 5 nodes: x (input), eps (input), style_gamma (input), style_beta (input),
/// adain_1d (native op).
///
/// Maps directly to NY's `AdaIN1dLayer` for tighter IBP bounds
/// compared to a decomposed InstanceNorm + elementwise scale/shift chain.
///
/// # Arguments
///
/// * `channels` — Number of channels (C dimension).
/// * `time` — Length of the time/spatial dimension (T).
///
/// # Errors
///
/// Returns [`TensorIRError::KernelValidation`] if `channels` or `time` is 0.
#[must_use = "returns a Result that may contain an error"]
pub fn build_adain1d(channels: usize, time: usize) -> Result<TensorKernelDef, TensorIRError> {
    validate_nonzero_dims(&[("channels", channels), ("time", time)])?;

    let full = vec![channels, time];

    Ok(TensorKernelDef {
        name: "adain_1d".into(),
        nodes: vec![
            input_node(0, "x", &full),
            input_node(1, "eps", &[1]),
            input_node(2, "style_gamma", &[channels]),
            input_node(3, "style_beta", &[channels]),
            TensorNode::new(
                TensorNodeId::new(4),
                TensorOpKind::AdaIN1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 1,
                    style_gamma: TensorNodeId::new(2),
                    style_beta: TensorNodeId::new(3),
                },
                full,
            ),
        ],
        output: TensorNodeId::new(4),
    })
}

/// Build the Snake (K1) `TensorKernelDef` for shape `[C, T]`.
///
/// 4 nodes: x (input `[C, T]`), alpha (input `[C]`), alpha broadcast `[C, T]`,
/// elementwise snake(x, alpha).
///
/// Snake is purely element-wise after broadcasting the per-channel alpha.
/// Maps to NY graph translation via `kernel_to_graph` on the inner
/// scalar `KernelDef`.
///
/// # Arguments
///
/// * `channels` — Number of channels (C dimension).
/// * `time` — Length of the time/spatial dimension (T).
///
/// # Errors
///
/// Returns [`TensorIRError::KernelValidation`] if `channels` or `time` is 0.
/// Returns [`TensorIRError::ScalarKernelBuild`] if the scalar kernel builder fails.
#[allow(dead_code)] // Called from #[cfg(test)] and #[cfg(kani)] only
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn build_snake_tensor(
    channels: usize,
    time: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    validate_nonzero_dims(&[("channels", channels), ("time", time)])?;

    let full = vec![channels, time];
    let snake_kernel =
        build_snake_scalar_kernel().map_err(|e| TensorIRError::ScalarKernelBuild(e.to_string()))?;

    Ok(TensorKernelDef {
        name: "snake".into(),
        nodes: vec![
            input_node(0, "x", &full),                             // x [C, T]
            input_node(1, "alpha", &[channels]),                   // alpha [C]
            broadcast_node(2, 1, &full, BroadcastAlignment::Left), // alpha → [C, T]
            elementwise_node(3, snake_kernel, &[0, 2], &full),     // snake(x, alpha)
        ],
        output: TensorNodeId::new(3),
    })
}

/// Build the fused AdaIN+Snake (K4) `TensorKernelDef` for shape `[C, T]`.
///
/// 8 nodes: x (input `[C, T]`), eps (input `[1]`), style_gamma (input `[C]`),
/// style_beta (input `[C]`), alpha (input `[C]`), AdaIN1d native op,
/// alpha broadcast `[C, T]`, elementwise snake(adain_out, alpha).
///
/// Composes the native `AdaIN1d` op (for tight NY bounds on the
/// normalization) with a Snake elementwise step. This avoids the bounds
/// blowup from decomposed InstanceNorm while keeping the Snake activation
/// as an explicit elementwise node for Kani verification.
///
/// # Arguments
///
/// * `channels` — Number of channels (C dimension).
/// * `time` — Length of the time/spatial dimension (T).
///
/// # Errors
///
/// Returns [`TensorIRError::KernelValidation`] if `channels` or `time` is 0.
/// Returns [`TensorIRError::ScalarKernelBuild`] if the scalar kernel builder fails.
#[allow(dead_code)] // Called from #[cfg(test)] and #[cfg(kani)] only
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn build_adain_snake_tensor(
    channels: usize,
    time: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    validate_nonzero_dims(&[("channels", channels), ("time", time)])?;

    let full = vec![channels, time];
    let snake_kernel =
        build_snake_scalar_kernel().map_err(|e| TensorIRError::ScalarKernelBuild(e.to_string()))?;

    Ok(TensorKernelDef {
        name: "adain_snake".into(),
        nodes: vec![
            input_node(0, "x", &full),                 // x [C, T]
            input_node(1, "eps", &[1]),                // eps scalar
            input_node(2, "style_gamma", &[channels]), // style_gamma [C]
            input_node(3, "style_beta", &[channels]),  // style_beta [C]
            input_node(4, "alpha", &[channels]),       // alpha [C]
            TensorNode::new(
                TensorNodeId::new(5),
                TensorOpKind::AdaIN1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 1,
                    style_gamma: TensorNodeId::new(2),
                    style_beta: TensorNodeId::new(3),
                },
                full.clone(),
            ), // AdaIN(x) [C, T]
            broadcast_node(6, 4, &full, BroadcastAlignment::Left), // alpha → [C, T]
            elementwise_node(7, snake_kernel, &[5, 6], &full), // snake(adain, alpha)
        ],
        output: TensorNodeId::new(7),
    })
}

#[cfg(kani)]
#[path = "adain_kani.rs"]
mod kani_proofs;

#[cfg(kani)]
#[path = "adain_kani_builder.rs"]
mod kani_builder_proofs;

#[cfg(kani)]
#[path = "adain_leaky_relu_kani.rs"]
mod kani_leaky_relu_proofs;

#[cfg(test)]
#[path = "adain_tests.rs"]
mod tests;
