// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LayerNorm (K7) kernel — `((x - mean) / sqrt(var + eps)) * gamma + beta`.
//!
//! Two-pass centered variance (like InstanceNorm K2) plus affine transform
//! (like RMSNorm K5). Layout: `[N, hidden]` where reduction is over the hidden
//! (last) axis.
//!
//! # LayerNorm formula
//!
//! ```text
//! mean   = mean(x, axis=hidden)
//! var    = mean((x - mean)², axis=hidden)
//! output = ((x - mean) / sqrt(var + eps)) * gamma + beta
//! ```
//!
//! Part of #19 (K2-K8 kernel ports).
//!
//! # Naming convention (#336)
//!
//! - `layer_norm_scalar` — per-element scalar, `Result<f32, KernelError>`
//! - `layer_norm_ref` — vector reference, `Result<Vec<f32>, KernelError>`
//! - `build_layer_norm_scalar_kernel` — `KernelDef` IR builder

use crate::ir::{BinOpKind, KernelDef, UnaryFnKind};
use crate::kernel_error::KernelError;
use crate::kernel_util::{
    affine_normalize_scalar, build_scalar_kernel, checked_slice_output, validate_eps,
    validate_finite_slice, validate_nonzero_dims, F32_PRECISION_LIMIT,
};
use crate::lower::LowerError;
use crate::tensor_builders::{
    binop_kernel, broadcast_node, elementwise_node, input_node, reduce_node, square_kernel,
    unary_kernel,
};
use crate::tensor_ir::{
    BroadcastAlignment, ReduceOp, TensorIRError, TensorKernelDef, TensorNodeId,
};

// --- K7 LayerNorm decomposed builder ---

/// Build the LayerNorm (K7) `TensorKernelDef` in decomposed form for shape `[N, hidden]`.
///
/// Decomposes into 18 nodes: 4 inputs, 2 reductions, 5 broadcasts, 7 element-wise.
/// Uses two-pass centered variance `mean((x - mean)²)` to avoid catastrophic
/// cancellation for large inputs (same approach as InstanceNorm K2).
///
/// ```text
/// Node 0:  x [N, hidden]             (Input)
/// Node 1:  eps [1]                    (Input)
/// Node 2:  gamma [hidden]             (Input)
/// Node 3:  beta [hidden]              (Input)
/// Node 4:  mean(x) [N]               (Reduce Mean, axis=1)
/// Node 5:  mean → [N, hidden]         (Broadcast Left)
/// Node 6:  x - mean [N, hidden]       (Elementwise sub)
/// Node 7:  (x-mean)² [N, hidden]      (Elementwise square)
/// Node 8:  var [N]                     (Reduce Mean, axis=1)
/// Node 9:  var → [N, hidden]           (Broadcast Left)
/// Node 10: eps → [N, hidden]           (Broadcast Left)
/// Node 11: var+eps [N, hidden]         (Elementwise add)
/// Node 12: rsqrt(var+eps) [N, hidden]  (Elementwise rsqrt)
/// Node 13: (x-mean)*rsqrt [N, hidden]  (Elementwise mul)
/// Node 14: gamma → [N, hidden]         (Broadcast Right)
/// Node 15: result*gamma [N, hidden]    (Elementwise mul)
/// Node 16: beta → [N, hidden]          (Broadcast Right)
/// Node 17: result+beta [N, hidden]     (Elementwise add)
/// ```
///
/// # Errors
///
/// Returns [`TensorIRError::KernelValidation`] if `n` or `hidden` is 0.
#[must_use = "returns a Result that may contain an error"]
pub fn build_layer_norm_decomposed(
    n: usize,
    hidden: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    validate_nonzero_dims(&[("n", n), ("hidden", hidden)])?;

    let full = vec![n, hidden];
    let reduced = vec![n];

    Ok(TensorKernelDef {
        name: "layer_norm".into(),
        nodes: vec![
            // Inputs
            input_node(0, "x", &full),         // x [N, hidden]
            input_node(1, "eps", &[1]),        // eps scalar [1]
            input_node(2, "gamma", &[hidden]), // gamma [hidden]
            input_node(3, "beta", &[hidden]),  // beta [hidden]
            // Pass 1: mean
            reduce_node(4, ReduceOp::Mean, 0, 1, &reduced), // mean(x) [N]
            broadcast_node(5, 4, &full, BroadcastAlignment::Left), // mean → [N, hidden]
            elementwise_node(6, binop_kernel("sub", BinOpKind::Sub), &[0, 5], &full), // x - mean [N, hidden]
            // Pass 2: variance
            elementwise_node(7, square_kernel(), &[6], &full), // (x-mean)² [N, hidden]
            reduce_node(8, ReduceOp::Mean, 7, 1, &reduced),    // var [N]
            broadcast_node(9, 8, &full, BroadcastAlignment::Left), // var → [N, hidden]
            broadcast_node(10, 1, &full, BroadcastAlignment::Left), // eps → [N, hidden]
            elementwise_node(11, binop_kernel("add", BinOpKind::Add), &[9, 10], &full), // var+eps [N, hidden]
            elementwise_node(12, unary_kernel("rsqrt", UnaryFnKind::Rsqrt), &[11], &full), // rsqrt(var+eps) [N, hidden]
            elementwise_node(13, binop_kernel("mul", BinOpKind::Mul), &[6, 12], &full), // (x-mean)*rsqrt [N, hidden]
            // Affine: gamma * normalized + beta
            broadcast_node(14, 2, &full, BroadcastAlignment::Right), // gamma → [N, hidden]
            elementwise_node(15, binop_kernel("mul", BinOpKind::Mul), &[13, 14], &full), // result*gamma [N, hidden]
            broadcast_node(16, 3, &full, BroadcastAlignment::Right), // beta → [N, hidden]
            elementwise_node(17, binop_kernel("add", BinOpKind::Add), &[15, 16], &full), // result+beta [N, hidden]
        ],
        output: TensorNodeId::new(17),
    })
}

/// Build the LayerNorm (K7) scalar element `KernelDef`.
///
/// Parameters: `x`, `mean`, `var_val`, `eps`, `gamma`, `beta` (6 params).
/// Computes: `(x - mean) * (var_val + eps).rsqrt() * gamma + beta`
///
/// This encodes the per-element computation after the two-pass reduction
/// computes `mean` and `var`. Uses `rsqrt` → UF approximation in ay.
///
/// # Errors
///
/// Returns [`LowerError`] if the kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_layer_norm_scalar_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn layer_norm_scalar(x: f32, mean: f32, var_val: f32, eps: f32, gamma: f32, beta: f32) -> f32 {
            (x - mean) * (var_val + eps).rsqrt() * gamma + beta
        }",
    )
}

/// Build the fused LayerNorm+GELU scalar KernelDef.
///
/// Parameters: `x`, `mean`, `var_val`, `eps`, `gamma`, `beta` (6 params).
/// Computes: `gelu(layer_norm(x, mean, var_val, eps, gamma, beta))`
///
/// This fuses the per-element LayerNorm with GELU activation, matching the
/// standard Transformer FFN pattern: LayerNorm → GELU. Uses the tanh
/// approximation form via exp (same as `build_gelu_kernel`).
///
/// # Errors
///
/// Returns [`LowerError`] if the hardcoded kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_layer_norm_gelu_fused_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn layer_norm_gelu(x: f32, mean: f32, var_val: f32, eps: f32, gamma: f32, beta: f32) -> f32 {
            let normed = (x - mean) * (var_val + eps).rsqrt() * gamma + beta;
            let k = 0.7978846;
            let inner = k * (normed + 0.044715 * normed * normed * normed);
            let e2 = (2.0 * inner).exp();
            0.5 * normed * (2.0 - 2.0 / (e2 + 1.0))
        }",
    )
}

/// Reference implementation for fused LayerNorm+GELU scalar.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is NaN or infinite.
/// Returns [`KernelError::InvalidEps`] if `var_val + eps <= 0`.
/// Returns [`KernelError::NonFiniteOutput`] if the fused result is non-finite.
#[must_use = "returns a Result that may contain an error"]
pub fn layer_norm_gelu_fused_scalar(
    x: f32,
    mean: f32,
    var_val: f32,
    eps: f32,
    gamma: f32,
    beta: f32,
) -> Result<f32, KernelError> {
    use crate::kernel_util::{checked_scalar_output, validate_finite_inputs};
    validate_finite_inputs(&[
        ("x", x),
        ("mean", mean),
        ("var_val", var_val),
        ("eps", eps),
        ("gamma", gamma),
        ("beta", beta),
    ])?;

    let denom_input = var_val + eps;
    if denom_input <= 0.0 {
        return Err(KernelError::InvalidEps { value: eps });
    }

    let normed = (x - mean) * denom_input.sqrt().recip() * gamma + beta;
    let k: f32 = 0.797_884_6;
    let inner = k * (normed + 0.044715 * normed * normed * normed);
    let e2 = (2.0 * inner).exp();
    let result = 0.5 * normed * (2.0 - 2.0 / (e2 + 1.0));
    checked_scalar_output(result)
}

// --- Reference implementation ---

/// Rust reference implementation of LayerNorm for differential testing.
///
/// Computes `((x - mean) / sqrt(var + eps)) * gamma + beta` per row.
///
/// # Errors
///
/// Returns [`KernelError::InvalidDimension`] if `n` or `hidden` is 0.
/// Returns [`KernelError::DimensionExceedsF32Precision`] if `hidden > 2^24`.
/// Returns [`KernelError::InvalidEps`] if `eps <= 0.0` or not finite.
/// Returns [`KernelError::ShapeMismatch`] if lengths don't match.
#[must_use = "returns a Result that may contain an error"]
pub fn layer_norm_ref(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    n: usize,
    hidden: usize,
    eps: f32,
) -> Result<Vec<f32>, KernelError> {
    validate_nonzero_dims(&[("n", n), ("hidden", hidden)])?;

    if hidden > F32_PRECISION_LIMIT {
        return Err(KernelError::DimensionExceedsF32Precision {
            name: "hidden",
            value: hidden,
        });
    }
    validate_eps(eps)?;
    let expected_len = n
        .checked_mul(hidden)
        .ok_or_else(|| KernelError::DimensionOverflow {
            dims: format!("{n} * {hidden}"),
        })?;
    if x.len() != expected_len {
        return Err(KernelError::ShapeMismatch {
            expected: expected_len,
            got: x.len(),
        });
    }
    if gamma.len() != hidden {
        return Err(KernelError::ShapeMismatch {
            expected: hidden,
            got: gamma.len(),
        });
    }
    if beta.len() != hidden {
        return Err(KernelError::ShapeMismatch {
            expected: hidden,
            got: beta.len(),
        });
    }
    validate_finite_slice("x", x)?;
    validate_finite_slice("gamma", gamma)?;
    validate_finite_slice("beta", beta)?;

    let hidden_f32 = hidden as f32;
    let mut output = vec![0.0f32; expected_len];

    for ni in 0..n {
        let offset = ni * hidden;
        let row = &x[offset..offset + hidden];

        // Two-pass centered variance
        let mean: f32 = row.iter().sum::<f32>() / hidden_f32;
        let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / hidden_f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for hi in 0..hidden {
            output[offset + hi] = (row[hi] - mean) * inv_std * gamma[hi] + beta[hi];
        }
    }

    checked_slice_output(&output)?;
    Ok(output)
}

// --- Per-element scalar function ---

/// Compute a single element of LayerNorm given pre-computed statistics.
///
/// `output[i] = (x - mean) / sqrt(var + eps) * gamma + beta`
///
/// This is the scalar kernel that each element of the tensor undergoes
/// after the reduction passes compute `mean` and `var`. Exposed for
/// Kani proof harnesses and differential testing.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is NaN or infinite.
/// Returns [`KernelError::NonFiniteOutput`] if the computed
/// result is non-finite despite all inputs being finite (e.g., extreme magnitudes
/// outside the Kani-proved domain).
///
/// # Safety invariants (proved by Kani)
///
/// - Finite output for finite inputs within the proved domain
///   (x ∈ [-1e3, 1e3], var ∈ [0, 1e6], eps ∈ [1e-8, 1], gamma ∈ [-10, 10]).
/// - Output is bounded when inputs are bounded.
#[must_use = "returns a Result that may contain an error"]
pub fn layer_norm_scalar(
    x: f32,
    mean: f32,
    var: f32,
    eps: f32,
    gamma: f32,
    beta: f32,
) -> Result<f32, KernelError> {
    affine_normalize_scalar(x, mean, var, eps, gamma, beta)
}

#[cfg(kani)]
#[path = "layer_norm_kani.rs"]
mod kani_proofs;

#[cfg(kani)]
#[path = "layer_norm_kani_builder_tests.rs"]
mod kani_builder_proofs;

#[cfg(test)]
#[path = "layer_norm_tests.rs"]
mod tests;
