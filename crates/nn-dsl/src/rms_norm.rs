// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! RMSNorm (K5) kernel — `x * rsqrt(mean(x²) + eps) * weight`.
//!
//! Simpler than InstanceNorm/LayerNorm: single-pass reduction (no mean-centering).
//! Layout: `[N, hidden]` where reduction is over the hidden (last) axis.
//!
//! # RMSNorm formula
//!
//! ```text
//! rms  = sqrt(mean(x², axis=hidden) + eps)
//! output = (x / rms) * weight
//! ```
//!
//! Equivalently: `x * rsqrt(mean(x²) + eps) * weight`
//!
//! Part of #19 (K2-K8 kernel ports).
//!
//! # Naming convention (#336)
//!
//! - `rms_norm_scalar` — per-element scalar, `Result<f32, KernelError>`
//! - `rms_norm_ref` — vector reference, `Result<Vec<f32>, KernelError>`
//! - `build_rms_norm_scalar_kernel` — `KernelDef` IR builder

use crate::ir::{BinOpKind, KernelDef, UnaryFnKind};
use crate::kernel_error::KernelError;
use crate::kernel_util::{
    build_scalar_kernel, checked_scalar_output, checked_slice_output, validate_eps,
    validate_finite_inputs, validate_finite_slice, validate_nonzero_dims, F32_PRECISION_LIMIT,
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

// --- K5 RMSNorm decomposed builder ---

/// Build the RMSNorm (K5) `TensorKernelDef` in decomposed form for shape `[N, hidden]`.
///
/// Decomposes into 12 nodes: 3 inputs, 1 reduction, 3 broadcasts, 5 element-wise.
///
/// ```text
/// Node 0: x [N, hidden]              (Input)
/// Node 1: eps [1]                     (Input)
/// Node 2: weight [hidden]             (Input)
/// Node 3: x² [N, hidden]             (Elementwise: square)
/// Node 4: mean(x²) [N]               (Reduce Mean, axis=1)
/// Node 5: mean(x²) → [N, hidden]     (Broadcast Left)
/// Node 6: eps → [N, hidden]           (Broadcast Left)
/// Node 7: mean(x²)+eps [N, hidden]   (Elementwise: add)
/// Node 8: rsqrt(mean(x²)+eps)        (Elementwise: rsqrt)
/// Node 9: x * rsqrt [N, hidden]      (Elementwise: mul)
/// Node 10: weight → [N, hidden]       (Broadcast Right)
/// Node 11: result * weight            (Elementwise: mul)
/// ```
///
/// # Errors
///
/// Returns [`TensorIRError::KernelValidation`] if `n` or `hidden` is 0.
#[must_use = "returns a Result that may contain an error"]
pub fn build_rms_norm_decomposed(
    n: usize,
    hidden: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    validate_nonzero_dims(&[("n", n), ("hidden", hidden)])?;

    let full = vec![n, hidden];
    let reduced = vec![n];

    Ok(TensorKernelDef {
        name: "rms_norm".into(),
        nodes: vec![
            input_node(0, "x", &full),                             // x [N, hidden]
            input_node(1, "eps", &[1]),                            // eps scalar [1]
            input_node(2, "weight", &[hidden]),                    // weight [hidden]
            elementwise_node(3, square_kernel(), &[0], &full),     // x² [N, hidden]
            reduce_node(4, ReduceOp::Mean, 3, 1, &reduced),        // mean(x²) [N]
            broadcast_node(5, 4, &full, BroadcastAlignment::Left), // mean(x²) → [N, hidden]
            broadcast_node(6, 1, &full, BroadcastAlignment::Left), // eps → [N, hidden]
            elementwise_node(7, binop_kernel("add", BinOpKind::Add), &[5, 6], &full), // mean(x²)+eps [N, hidden]
            elementwise_node(8, unary_kernel("rsqrt", UnaryFnKind::Rsqrt), &[7], &full), // rsqrt(mean(x²)+eps) [N, hidden]
            elementwise_node(9, binop_kernel("mul", BinOpKind::Mul), &[0, 8], &full), // x * rsqrt [N, hidden]
            broadcast_node(10, 2, &full, BroadcastAlignment::Right), // weight → [N, hidden]
            elementwise_node(11, binop_kernel("mul", BinOpKind::Mul), &[9, 10], &full), // result * weight [N, hidden]
        ],
        output: TensorNodeId::new(11),
    })
}

/// Build the RMSNorm (K5) `TensorKernelDef` using the native `RmsNorm` op.
///
/// 3 nodes: x (input), eps (input), weight (input), rms_norm (native op).
/// Maps directly to NY's `RmsNormLayer` for tighter IBP bounds
/// compared to the 12-node decomposed form.
///
/// # Errors
///
/// Returns [`TensorIRError::KernelValidation`] if `n` or `hidden` is 0.
#[must_use = "returns a Result that may contain an error"]
pub fn build_rms_norm(n: usize, hidden: usize) -> Result<TensorKernelDef, TensorIRError> {
    validate_nonzero_dims(&[("n", n), ("hidden", hidden)])?;

    let full = vec![n, hidden];

    Ok(TensorKernelDef {
        name: "rms_norm".into(),
        nodes: vec![
            input_node(0, "x", &full),
            input_node(1, "eps", &[1]),
            input_node(2, "weight", &[hidden]),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::RmsNorm {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 1,
                    weight: TensorNodeId::new(2),
                },
                full,
            ),
        ],
        output: TensorNodeId::new(3),
    })
}

/// Build the RMSNorm (K5) scalar element `KernelDef`.
///
/// Parameters: `x`, `rms_inv`, `weight` (3 params).
/// Computes: `x * rms_inv * weight`
///
/// This encodes the per-element computation after the reduction pass
/// computes `rms_inv = rsqrt(mean(x²) + eps)`. Pure arithmetic — no
/// transcendentals, no UF approximation needed for ay SMT verification.
///
/// # Errors
///
/// Returns [`LowerError`] if the kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_rms_norm_scalar_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn rms_norm_scalar(x: f32, rms_inv: f32, weight: f32) -> f32 {
            x * rms_inv * weight
        }",
    )
}

// --- Reference implementation ---

/// Rust reference implementation of RMSNorm for differential testing.
///
/// Computes `x * rsqrt(mean(x²) + eps) * weight` per row.
///
/// # Errors
///
/// Returns [`KernelError::InvalidDimension`] if `n` or `hidden` is 0.
/// Returns [`KernelError::DimensionExceedsF32Precision`] if `hidden > 2^24`.
/// Returns [`KernelError::InvalidEps`] if `eps <= 0.0` or not finite.
/// Returns [`KernelError::ShapeMismatch`] if `x.len() != n * hidden` or
/// `weight.len() != hidden`.
#[must_use = "returns a Result that may contain an error"]
pub fn rms_norm_ref(
    x: &[f32],
    weight: &[f32],
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
    if weight.len() != hidden {
        return Err(KernelError::ShapeMismatch {
            expected: hidden,
            got: weight.len(),
        });
    }
    validate_finite_slice("x", x)?;
    validate_finite_slice("weight", weight)?;

    let hidden_f32 = hidden as f32;
    let mut output = vec![0.0f32; expected_len];

    for ni in 0..n {
        let offset = ni * hidden;
        let row = &x[offset..offset + hidden];

        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / hidden_f32;
        let rms_inv = 1.0 / (mean_sq + eps).sqrt();

        for hi in 0..hidden {
            output[offset + hi] = row[hi] * rms_inv * weight[hi];
        }
    }

    checked_slice_output(&output)?;
    Ok(output)
}

/// Build the fused RMSNorm+SiLU-Mul scalar KernelDef.
///
/// Parameters: `x`, `rms_inv`, `weight`, `up` (4 params).
/// Computes: `silu_mul(rms_norm(x, rms_inv, weight), up)`
///       = `(x * rms_inv * weight) / (1 + exp(-(x * rms_inv * weight))) * up`
///
/// This fuses the per-element RMSNorm with SiLU-Mul gating, matching the
/// SwiGLU pattern in LLaMA/Mistral-style models: RMSNorm → SiLU gate × up.
///
/// # Errors
///
/// Returns [`LowerError`] if the hardcoded kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_rms_norm_silu_mul_fused_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn rms_norm_silu_mul(x: f32, rms_inv: f32, weight: f32, up: f32) -> f32 {
            let normed = x * rms_inv * weight;
            (normed / (1.0 + (-normed).exp())) * up
        }",
    )
}

/// Reference implementation for fused RMSNorm+SiLU-Mul scalar.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is NaN or infinite.
/// Returns [`KernelError::NonFiniteOutput`] if the fused result is non-finite.
#[must_use = "returns a Result that may contain an error"]
pub fn rms_norm_silu_mul_fused_scalar(
    x: f32,
    rms_inv: f32,
    weight: f32,
    up: f32,
) -> Result<f32, KernelError> {
    validate_finite_inputs(&[
        ("x", x),
        ("rms_inv", rms_inv),
        ("weight", weight),
        ("up", up),
    ])?;
    let normed = x * rms_inv * weight;
    let sigmoid = 1.0 / (1.0 + (-normed).exp());
    let result = normed * sigmoid * up;
    checked_scalar_output(result)
}

/// Compute a single element of RMSNorm given pre-computed statistics.
///
/// `rms_norm_scalar(x, rms_inv, weight) = x * rms_inv * weight`
///
/// where `rms_inv = rsqrt(mean(x²) + eps)` is precomputed.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is NaN or infinite.
/// Returns [`KernelError::NonFiniteOutput`] if the computed
/// result is non-finite despite all inputs being finite (e.g., extreme magnitudes
/// outside the Kani-proved domain).
#[must_use = "returns a Result that may contain an error"]
pub fn rms_norm_scalar(x: f32, rms_inv: f32, weight: f32) -> Result<f32, KernelError> {
    validate_finite_inputs(&[("x", x), ("rms_inv", rms_inv), ("weight", weight)])?;

    let result = x * rms_inv * weight;

    checked_scalar_output(result)
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::kani_stubs::{exp_stub, sqrt_stub};

    /// Proves `rms_norm_scalar` produces finite output for bounded inputs.
    ///
    /// Domain: x ∈ [-1e3, 1e3], rms_inv ∈ [0, 1e3], weight ∈ [-10, 10].
    /// The `rms_inv` parameter is always non-negative (it's `1/sqrt(mean(x²)+eps)`).
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::stub(f32::exp, exp_stub)]
    #[kani::stub(f32::sqrt, sqrt_stub)]
    fn rms_norm_scalar_finite_for_bounded_inputs() {
        let x: f32 = kani::any();
        let rms_inv: f32 = kani::any();
        let weight: f32 = kani::any();

        kani::assume(x.is_finite());
        kani::assume(rms_inv.is_finite());
        kani::assume(weight.is_finite());

        kani::assume(x >= -1.0e3 && x <= 1.0e3);
        kani::assume(rms_inv >= 0.0 && rms_inv <= 1.0e3);
        kani::assume(weight >= -10.0 && weight <= 10.0);

        let y = rms_norm_scalar(x, rms_inv, weight)
            .expect("rms_norm_scalar must succeed for bounded finite inputs");
        assert!(y.is_finite(), "rms_norm_scalar must produce finite output");
        assert!(!y.is_nan(), "rms_norm_scalar must not produce NaN");
    }

    /// Proves that zero input always produces zero output regardless of
    /// rms_inv and weight values (within bounds).
    ///
    /// RMSNorm of zero: `0 * rms_inv * weight = 0`.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::stub(f32::exp, exp_stub)]
    #[kani::stub(f32::sqrt, sqrt_stub)]
    fn rms_norm_scalar_zero_input_is_zero() {
        let rms_inv: f32 = kani::any();
        let weight: f32 = kani::any();

        kani::assume(rms_inv.is_finite());
        kani::assume(weight.is_finite());
        kani::assume(rms_inv >= 0.0 && rms_inv <= 1.0e4);
        kani::assume(weight >= -1.0e4 && weight <= 1.0e4);

        let y = rms_norm_scalar(0.0, rms_inv, weight)
            .expect("rms_norm_scalar must succeed for bounded finite inputs");
        assert!(y == 0.0, "rms_norm_scalar(0, _, _) must be exactly 0");
    }

    /// Proves output is bounded: `|y| <= |x| * rms_inv * |weight|`.
    ///
    /// Since all three factors are finite and bounded, the product is bounded.
    /// This proves the output cannot exceed the product of the input magnitudes.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::stub(f32::exp, exp_stub)]
    #[kani::stub(f32::sqrt, sqrt_stub)]
    fn rms_norm_scalar_output_bounded() {
        let x: f32 = kani::any();
        let rms_inv: f32 = kani::any();
        let weight: f32 = kani::any();

        kani::assume(x.is_finite());
        kani::assume(rms_inv.is_finite());
        kani::assume(weight.is_finite());

        kani::assume(x >= -100.0 && x <= 100.0);
        kani::assume(rms_inv >= 0.0 && rms_inv <= 100.0);
        kani::assume(weight >= -10.0 && weight <= 10.0);

        let y = rms_norm_scalar(x, rms_inv, weight)
            .expect("rms_norm_scalar must succeed for bounded finite inputs");
        assert!(y.is_finite(), "output must be finite");

        // |y| <= |x| * rms_inv * |weight| <= 100 * 100 * 10 = 100000
        assert!(
            y.abs() <= 1.0e5 + 1.0,
            "rms_norm_scalar output must be bounded"
        );
    }
}

#[cfg(kani)]
#[path = "rms_norm_kani_builder_tests.rs"]
mod kani_builder_proofs;

#[cfg(test)]
#[path = "rms_norm_tests.rs"]
mod tests;
