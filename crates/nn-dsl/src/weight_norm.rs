// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Weight normalization — `w = g * v / ||v||`.
//!
//! Weight normalization is a reparameterization of weight vectors that decouples
//! the direction from the magnitude. For a weight vector `v` and scalar `g`:
//!
//! ```text
//! w = g * v / ||v||
//!   = g * v * rsqrt(sum(v²) + eps)
//! ```
//!
//! This is applied to weights *before* use in Conv1d/Linear layers. In dvoice,
//! 15 occurrences of hand-rolled weight normalization exist in `weight_norm.rs`.
//!
//! # Implementation
//!
//! Decomposed into existing tensor IR ops (no new `TensorOpKind` variant needed):
//! - `Square` (elementwise)
//! - `Reduce(Sum)` along the normalization axis
//! - `Broadcast` + `Add(eps)` for numerical stability
//! - `Rsqrt` (elementwise)
//! - `BinaryMul` for `v * rsqrt_norm`
//! - `Broadcast` + `BinaryMul` for `g * normalized_v`
//!
//! Part of #749 (dvoice weight normalization).

#[cfg(any(test, kani))]
use crate::ir::{BinOpKind, UnaryFnKind};
#[cfg(any(test, kani))]
use crate::kernel_error::KernelError;
#[cfg(any(test, kani))]
use crate::kernel_util::{
    checked_slice_output, validate_eps, validate_finite_inputs, validate_finite_slice,
    validate_nonzero_dims, F32_PRECISION_LIMIT,
};
#[cfg(test)]
use crate::tensor_builders::{
    binop_kernel, broadcast_node, elementwise_node, input_node, reduce_node, square_kernel,
    unary_kernel,
};
#[cfg(test)]
use crate::tensor_ir::{
    BroadcastAlignment, ReduceOp, TensorIRError, TensorKernelDef, TensorNodeId,
};

// --- Weight normalization decomposed builder ---

/// Build the weight normalization `TensorKernelDef` in decomposed form.
///
/// Weight shape: `[fan_out, fan_in]`. Normalization is over `axis` (typically
/// the fan_in dimension, axis=1). `g` is a per-output-channel magnitude scalar
/// with shape `[fan_out]`.
///
/// Decomposes into 12 nodes:
///
/// ```text
/// Node 0: v [fan_out, fan_in]           (Input — unnormalized weight)
/// Node 1: eps [1]                        (Input — stability epsilon)
/// Node 2: g [fan_out]                    (Input — learned magnitude)
/// Node 3: v² [fan_out, fan_in]           (Elementwise: square)
/// Node 4: sum(v²) [fan_out]              (Reduce Sum, axis=1)
/// Node 5: sum(v²) → [fan_out, fan_in]   (Broadcast Left)
/// Node 6: eps → [fan_out, fan_in]        (Broadcast Left)
/// Node 7: sum(v²)+eps [fan_out, fan_in]  (Elementwise: add)
/// Node 8: rsqrt(sum(v²)+eps)             (Elementwise: rsqrt)
/// Node 9: v * rsqrt [fan_out, fan_in]    (Elementwise: mul)
/// Node 10: g → [fan_out, fan_in]         (Broadcast Left)
/// Node 11: g * normalized_v              (Elementwise: mul)
/// ```
///
/// # Errors
///
/// Returns [`TensorIRError::KernelValidation`] if `fan_out` or `fan_in` is 0.
#[cfg(test)]
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn build_weight_norm_decomposed(
    fan_out: usize,
    fan_in: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    validate_nonzero_dims(&[("fan_out", fan_out), ("fan_in", fan_in)])?;

    let full = vec![fan_out, fan_in];
    let reduced = vec![fan_out];

    Ok(TensorKernelDef {
        name: "weight_norm".into(),
        nodes: vec![
            input_node(0, "v", &full),    // v [fan_out, fan_in]
            input_node(1, "eps", &[1]),   // eps scalar [1]
            input_node(2, "g", &reduced), // g [fan_out]
            // Square: v² [fan_out, fan_in]
            elementwise_node(3, square_kernel(), &[0], &full),
            // Reduce sum over fan_in axis: sum(v²) [fan_out]
            reduce_node(4, ReduceOp::Sum, 3, 1, &reduced),
            // Broadcast sum(v²) back to full shape
            broadcast_node(5, 4, &full, BroadcastAlignment::Left),
            // Broadcast eps to full shape
            broadcast_node(6, 1, &full, BroadcastAlignment::Left),
            // sum(v²) + eps
            elementwise_node(7, binop_kernel("add", BinOpKind::Add), &[5, 6], &full),
            // rsqrt(sum(v²) + eps)
            elementwise_node(8, unary_kernel("rsqrt", UnaryFnKind::Rsqrt), &[7], &full),
            // v * rsqrt(sum(v²) + eps) = v / ||v||
            elementwise_node(9, binop_kernel("mul", BinOpKind::Mul), &[0, 8], &full),
            // Broadcast g to full shape
            broadcast_node(10, 2, &full, BroadcastAlignment::Left),
            // g * (v / ||v||) = final weight
            elementwise_node(11, binop_kernel("mul", BinOpKind::Mul), &[10, 9], &full),
        ],
        output: TensorNodeId::new(11),
    })
}

// --- Reference implementation ---

/// Rust reference implementation of weight normalization for differential testing.
///
/// Computes `w = g * v / ||v||` where `||v|| = sqrt(sum(v²) + eps)` per row.
///
/// `v` has shape `[fan_out, fan_in]`, `g` has shape `[fan_out]`.
/// Normalization is over the fan_in (column) dimension.
///
/// # Errors
///
/// Returns [`KernelError::InvalidDimension`] if dimensions are 0.
/// Returns [`KernelError::DimensionExceedsF32Precision`] if `fan_in > 2^24`.
/// Returns [`KernelError::InvalidEps`] if `eps <= 0.0` or not finite.
/// Returns [`KernelError::ShapeMismatch`] if tensor shapes don't match.
#[cfg(test)]
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn weight_norm_ref(
    v: &[f32],
    g: &[f32],
    fan_out: usize,
    fan_in: usize,
    eps: f32,
) -> Result<Vec<f32>, KernelError> {
    validate_nonzero_dims(&[("fan_out", fan_out), ("fan_in", fan_in)])?;

    if fan_in > F32_PRECISION_LIMIT {
        return Err(KernelError::DimensionExceedsF32Precision {
            name: "fan_in",
            value: fan_in,
        });
    }
    validate_eps(eps)?;

    let expected_len =
        fan_out
            .checked_mul(fan_in)
            .ok_or_else(|| KernelError::DimensionOverflow {
                dims: format!("{fan_out} * {fan_in}"),
            })?;
    if v.len() != expected_len {
        return Err(KernelError::ShapeMismatch {
            expected: expected_len,
            got: v.len(),
        });
    }
    if g.len() != fan_out {
        return Err(KernelError::ShapeMismatch {
            expected: fan_out,
            got: g.len(),
        });
    }

    validate_finite_slice("v", v)?;
    validate_finite_slice("g", g)?;

    let mut output = vec![0.0f32; expected_len];

    for (row, &g_row) in g.iter().enumerate() {
        let offset = row * fan_in;
        let row_v = &v[offset..offset + fan_in];

        // ||v|| = sqrt(sum(v²) + eps)
        let sum_sq: f32 = row_v.iter().map(|x| x * x).sum::<f32>();
        let norm_inv = 1.0 / (sum_sq + eps).sqrt();

        for (col, &v_col) in row_v.iter().enumerate() {
            output[offset + col] = g_row * v_col * norm_inv;
        }
    }

    checked_slice_output(&output)?;
    Ok(output)
}

/// Compute a single element of weight-normalized output given pre-computed norm inverse.
///
/// `weight_norm_scalar(v, g, norm_inv) = g * v * norm_inv`
///
/// where `norm_inv = rsqrt(sum(v²) + eps)` is precomputed per row.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is NaN or infinite.
/// Returns [`KernelError::NonFiniteOutput`] if the result is non-finite.
#[cfg(any(test, kani))]
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn weight_norm_scalar(v: f32, g: f32, norm_inv: f32) -> Result<f32, KernelError> {
    validate_finite_inputs(&[("v", v), ("g", g), ("norm_inv", norm_inv)])?;

    let result = g * v * norm_inv;

    crate::kernel_util::checked_scalar_output(result)
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::kani_stubs::sqrt_stub;

    /// Proves `weight_norm_scalar` produces finite output for bounded inputs.
    ///
    /// Domain: v ∈ [-1e3, 1e3], g ∈ [-10, 10], norm_inv ∈ (0, 1e3].
    /// `norm_inv` is always positive (it's `1/sqrt(sum(v²)+eps)`).
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_stub)]
    fn weight_norm_scalar_finite_for_bounded_inputs() {
        let v: f32 = kani::any();
        let g: f32 = kani::any();
        let norm_inv: f32 = kani::any();

        kani::assume(v.is_finite());
        kani::assume(g.is_finite());
        kani::assume(norm_inv.is_finite());

        kani::assume(v >= -1.0e3 && v <= 1.0e3);
        kani::assume(g >= -10.0 && g <= 10.0);
        kani::assume(norm_inv >= 0.0 && norm_inv <= 1.0e3);

        let y = weight_norm_scalar(v, g, norm_inv)
            .expect("weight_norm_scalar must succeed for bounded finite inputs");
        assert!(
            y.is_finite(),
            "weight_norm_scalar must produce finite output"
        );
        assert!(!y.is_nan(), "weight_norm_scalar must not produce NaN");
    }

    /// Proves that zero weight vector element produces zero output.
    ///
    /// `g * 0 * norm_inv = 0` regardless of g and norm_inv.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_stub)]
    fn weight_norm_scalar_zero_v_is_zero() {
        let g: f32 = kani::any();
        let norm_inv: f32 = kani::any();

        kani::assume(g.is_finite());
        kani::assume(norm_inv.is_finite());
        kani::assume(g >= -1.0e4 && g <= 1.0e4);
        kani::assume(norm_inv >= 0.0 && norm_inv <= 1.0e4);

        let y = weight_norm_scalar(0.0, g, norm_inv)
            .expect("weight_norm_scalar must succeed for bounded finite inputs");
        assert!(y == 0.0, "weight_norm_scalar(0, _, _) must be exactly 0");
    }

    /// Proves output is bounded: `|y| <= |v| * |g| * norm_inv`.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_stub)]
    fn weight_norm_scalar_output_bounded() {
        let v: f32 = kani::any();
        let g: f32 = kani::any();
        let norm_inv: f32 = kani::any();

        kani::assume(v.is_finite());
        kani::assume(g.is_finite());
        kani::assume(norm_inv.is_finite());

        kani::assume(v >= -100.0 && v <= 100.0);
        kani::assume(g >= -10.0 && g <= 10.0);
        kani::assume(norm_inv >= 0.0 && norm_inv <= 100.0);

        let y = weight_norm_scalar(v, g, norm_inv)
            .expect("weight_norm_scalar must succeed for bounded finite inputs");
        assert!(y.is_finite(), "output must be finite");

        // |y| <= |v| * |g| * norm_inv <= 100 * 10 * 100 = 100000
        assert!(
            y.abs() <= 1.0e5 + 1.0,
            "weight_norm_scalar output must be bounded"
        );
    }
}

#[cfg(test)]
#[path = "weight_norm_tests.rs"]
mod tests;
