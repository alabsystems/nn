// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! InstanceNorm (K2) with optional affine transform (gamma/beta).
//!
//! Provides affine variants of K2 InstanceNorm that match dvoice's
//! `instance_norm.metal` kernel in `has_affine=1` mode:
//! `output = gamma[c] * (x - mean) / sqrt(var + eps) + beta[c]`
//!
//! Extracted from `instance_norm.rs` to stay under the 500-line file limit.
//!
//! # Naming convention (#336)
//!
//! - `instance_norm_affine_scalar` — per-element scalar, `Result<f32, KernelError>`
//! - `instance_norm_affine_ref` — vector reference, `Result<Vec<f32>, KernelError>`
//! - `build_instance_norm_affine_scalar_kernel` — `KernelDef` IR builder

use crate::ir::{BinOpKind, KernelDef, UnaryFnKind};
use crate::kernel_error::KernelError;
use crate::kernel_util::{
    affine_normalize_scalar, build_scalar_kernel, checked_slice_output, validate_eps,
    validate_finite_slice, F32_PRECISION_LIMIT,
};
use crate::lower::LowerError;
use crate::tensor_builders::{
    binop_kernel, broadcast_node, elementwise_node, input_node, reduce_node, square_kernel,
    unary_kernel,
};
use crate::tensor_ir::{
    BroadcastAlignment, ReduceOp, TensorIRError, TensorKernelDef, TensorNode, TensorNodeId,
    TensorOpKind,
};

use super::validate_bct;

/// Build the InstanceNorm (K2) `TensorKernelDef` with affine parameters for shape `[B, C, T]`
/// using the native `InstanceNorm1d` tensor op.
///
/// Produces a 5-node IR that maps directly to NY's `InstanceNorm1dLayer::new()`.
/// Matches dvoice's `instance_norm.metal` in `has_affine=1` mode.
///
/// # Errors
///
/// Returns [`TensorIRError::KernelValidation`] if any dimension is 0.
#[allow(dead_code)] // Called from #[cfg(test)] only
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn build_instance_norm_affine(
    b: usize,
    c: usize,
    t: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    validate_bct(b, c, t)?;

    let full = vec![b, c, t];

    Ok(TensorKernelDef {
        name: "instance_norm_affine".into(),
        nodes: vec![
            input_node(0, "x", &full),
            input_node(1, "eps", &[1]),
            input_node(2, "gamma", &[c]),
            input_node(3, "beta", &[c]),
            TensorNode::new(
                TensorNodeId::new(4),
                TensorOpKind::InstanceNorm1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 2,
                    gamma: Some(TensorNodeId::new(2)),
                    beta: Some(TensorNodeId::new(3)),
                },
                full,
            ),
        ],
        output: TensorNodeId::new(4),
    })
}

/// Build the InstanceNorm (K2) `TensorKernelDef` with affine in decomposed form.
///
/// 20 nodes: 4 inputs, 2 reductions, 4 broadcasts, 2 reshapes, 8 element-wise.
/// Gamma/beta `[C]` are reshaped to `[1,C,1]` for middle-axis broadcast to `[B,C,T]`.
///
/// # Errors
///
/// Returns [`TensorIRError::KernelValidation`] if any dimension is 0.
#[allow(dead_code)] // Called from #[cfg(test)] and #[cfg(kani)] only
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn build_instance_norm_decomposed_affine(
    b: usize,
    c: usize,
    t: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    validate_bct(b, c, t)?;

    let full = vec![b, c, t];
    let reduced = vec![b, c];
    let channel_3d = vec![1, c, 1];

    Ok(TensorKernelDef {
        name: "instance_norm_affine".into(),
        nodes: vec![
            input_node(0, "x", &full),
            input_node(1, "eps", &[1]),
            input_node(2, "gamma", &[c]),
            input_node(3, "beta", &[c]),
            reduce_node(4, ReduceOp::Mean, 0, 2, &reduced),
            broadcast_node(5, 4, &full, BroadcastAlignment::Left),
            elementwise_node(6, binop_kernel("sub", BinOpKind::Sub), &[0, 5], &full),
            elementwise_node(7, square_kernel(), &[6], &full),
            reduce_node(8, ReduceOp::Mean, 7, 2, &reduced),
            broadcast_node(9, 8, &full, BroadcastAlignment::Left),
            broadcast_node(10, 1, &full, BroadcastAlignment::Left),
            elementwise_node(11, binop_kernel("add", BinOpKind::Add), &[9, 10], &full),
            elementwise_node(12, unary_kernel("rsqrt", UnaryFnKind::Rsqrt), &[11], &full),
            elementwise_node(13, binop_kernel("mul", BinOpKind::Mul), &[6, 12], &full),
            // Reshape gamma [C] → [1,C,1] for middle-axis broadcast
            TensorNode::new(
                TensorNodeId::new(14),
                TensorOpKind::Reshape {
                    input: TensorNodeId::new(2),
                    target_shape: channel_3d.clone(),
                },
                channel_3d.clone(),
            ),
            broadcast_node(15, 14, &full, BroadcastAlignment::Left),
            elementwise_node(16, binop_kernel("mul", BinOpKind::Mul), &[13, 15], &full),
            // Reshape beta [C] → [1,C,1]
            TensorNode::new(
                TensorNodeId::new(17),
                TensorOpKind::Reshape {
                    input: TensorNodeId::new(3),
                    target_shape: channel_3d.clone(),
                },
                channel_3d,
            ),
            broadcast_node(18, 17, &full, BroadcastAlignment::Left),
            elementwise_node(19, binop_kernel("add", BinOpKind::Add), &[16, 18], &full),
        ],
        output: TensorNodeId::new(19),
    })
}

/// Build the InstanceNorm (K2) scalar element `KernelDef` with affine transform.
///
/// Parameters: `x`, `mean`, `var_val`, `eps`, `gamma`, `beta` (6 params).
/// Computes: `(x - mean) * (var_val + eps).rsqrt() * gamma + beta`
///
/// # Errors
///
/// Returns [`LowerError`] if the kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_instance_norm_affine_scalar_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn instance_norm_affine_scalar(x: f32, mean: f32, var_val: f32, eps: f32, gamma: f32, beta: f32) -> f32 {
            (x - mean) * (var_val + eps).rsqrt() * gamma + beta
        }",
    )
}

/// Rust reference implementation of InstanceNorm with affine for differential testing.
///
/// Computes `gamma[c] * (x - mean) / sqrt(var + eps) + beta[c]` per channel.
///
/// # Errors
///
/// Returns [`KernelError`] on invalid dimensions, eps, or shape mismatch.
#[allow(dead_code)] // Called from #[cfg(test)] only
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn instance_norm_affine_ref(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    b: usize,
    c: usize,
    t: usize,
    eps: f32,
) -> Result<Vec<f32>, KernelError> {
    validate_bct(b, c, t)?;

    if t > F32_PRECISION_LIMIT {
        return Err(KernelError::DimensionExceedsF32Precision {
            name: "t",
            value: t,
        });
    }
    validate_eps(eps)?;
    let expected_len = b
        .checked_mul(c)
        .and_then(|bc| bc.checked_mul(t))
        .ok_or_else(|| KernelError::DimensionOverflow {
            dims: format!("{b} * {c} * {t}"),
        })?;
    if x.len() != expected_len {
        return Err(KernelError::ShapeMismatch {
            expected: expected_len,
            got: x.len(),
        });
    }
    if gamma.len() != c {
        return Err(KernelError::ShapeMismatch {
            expected: c,
            got: gamma.len(),
        });
    }
    if beta.len() != c {
        return Err(KernelError::ShapeMismatch {
            expected: c,
            got: beta.len(),
        });
    }
    validate_finite_slice("x", x)?;
    validate_finite_slice("gamma", gamma)?;
    validate_finite_slice("beta", beta)?;

    let mut output = vec![0.0f32; expected_len];

    for bi in 0..b {
        for ci in 0..c {
            let offset = (bi * c + ci) * t;
            let slice = &x[offset..offset + t];

            let t_f32 = t as f32;
            let mean: f32 = slice.iter().sum::<f32>() / t_f32;
            let var: f32 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / t_f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            for ti in 0..t {
                output[offset + ti] = gamma[ci] * (slice[ti] - mean) * inv_std + beta[ci];
            }
        }
    }

    checked_slice_output(&output)?;
    Ok(output)
}

/// Compute a single element of InstanceNorm with affine given pre-computed statistics.
///
/// `output = gamma * (x - mean) / sqrt(var + eps) + beta`
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is NaN or infinite, or if
/// the computed result overflows to infinity.
#[allow(dead_code)] // Called from #[cfg(test)] and #[cfg(kani)] only
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn instance_norm_affine_scalar(
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
mod kani_proofs {
    use super::*;
    use crate::kani_stubs::sqrt_stub;

    /// Proves `instance_norm_affine_scalar` produces finite output for bounded inputs.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_stub)]
    fn instance_norm_affine_scalar_finite_for_bounded_inputs() {
        let x: f32 = kani::any();
        let mean: f32 = kani::any();
        let var: f32 = kani::any();
        let eps: f32 = kani::any();
        let gamma: f32 = kani::any();
        let beta: f32 = kani::any();

        kani::assume(x.is_finite() && x >= -1.0e3 && x <= 1.0e3);
        kani::assume(mean.is_finite() && mean >= -1.0e3 && mean <= 1.0e3);
        kani::assume(var.is_finite() && var >= 0.0 && var <= 1.0e6);
        kani::assume(eps.is_finite() && eps >= 1.0e-8 && eps <= 1.0);
        kani::assume(gamma.is_finite() && gamma >= -10.0 && gamma <= 10.0);
        kani::assume(beta.is_finite() && beta >= -10.0 && beta <= 10.0);

        let y = instance_norm_affine_scalar(x, mean, var, eps, gamma, beta)
            .expect("must succeed for bounded finite inputs");
        assert!(y.is_finite(), "must produce finite output");
        assert!(!y.is_nan(), "must not produce NaN");
    }

    /// Proves that zero variance with positive eps still produces finite output.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_stub)]
    fn instance_norm_affine_scalar_zero_variance_safe() {
        let x: f32 = kani::any();
        let eps: f32 = kani::any();
        let gamma: f32 = kani::any();
        let beta: f32 = kani::any();

        kani::assume(x.is_finite() && x >= -1.0e3 && x <= 1.0e3);
        kani::assume(eps.is_finite() && eps >= 1.0e-8 && eps <= 1.0);
        kani::assume(gamma.is_finite() && gamma >= -10.0 && gamma <= 10.0);
        kani::assume(beta.is_finite() && beta >= -10.0 && beta <= 10.0);

        let y = instance_norm_affine_scalar(x, x, 0.0, eps, gamma, beta)
            .expect("must succeed for bounded finite inputs");
        assert!(y.is_finite(), "zero-variance output must be finite");
        assert!(
            (y - beta).abs() < 1e-3,
            "zero-variance output should equal beta"
        );
    }

    /// Proves identity affine (gamma=1, beta=0) produces bounded output.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_stub)]
    fn instance_norm_affine_scalar_identity_is_non_affine() {
        let x: f32 = kani::any();
        let mean: f32 = kani::any();
        let var: f32 = kani::any();
        let eps: f32 = kani::any();

        kani::assume(x.is_finite() && x >= -100.0 && x <= 100.0);
        kani::assume(mean.is_finite() && mean >= -100.0 && mean <= 100.0);
        kani::assume(var.is_finite() && var >= 0.0 && var <= 1.0e4);
        kani::assume(eps.is_finite() && eps >= 1.0e-5 && eps <= 1.0);

        let y = instance_norm_affine_scalar(x, mean, var, eps, 1.0, 0.0)
            .expect("must succeed for bounded finite inputs");
        assert!(y.is_finite(), "identity affine output must be finite");
        assert!(y.abs() <= 7.0e4, "identity affine output must be bounded");
    }
}
