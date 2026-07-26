// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! InstanceNorm (K2) kernel — the first tensor-level kernel in nn.
//!
//! Phase E of `designs/2026-02-26-kernelir-tensor-ops.md`.
//!
//! Provides:
//! - `build_instance_norm()` — native `InstanceNorm1d` op (for NY verification)
//! - `build_instance_norm_decomposed()` — 12-node form (for MSL codegen, Kani)
//! - `build_instance_norm_affine()` — native with gamma/beta (dvoice `has_affine=1`)
//! - `build_instance_norm_decomposed_affine()` — 20-node form with gamma/beta
//! - `instance_norm_ref()` — Rust reference implementation for differential testing
//! - `instance_norm_affine_ref()` — Affine reference: `gamma * norm + beta`
//!
//! # InstanceNorm formula
//!
//! ```text
//! mean   = mean(x, axis=T)
//! var    = mean((x - mean)², axis=T)
//! output = (x - mean) / sqrt(var + eps)                   [non-affine]
//! output = gamma[c] * (x - mean) / sqrt(var + eps) + beta[c]  [affine]
//! ```
//!
//! Uses two-pass centered variance to avoid catastrophic cancellation when
//! input magnitudes are large (see #102).
//!
//! # Naming convention (#336)
//!
//! - `instance_norm_scalar` — per-element scalar (via `build_instance_norm_scalar_kernel`)
//! - `instance_norm_ref` — vector reference, `Result<Vec<f32>, KernelError>`
//! - `instance_norm_affine_scalar` — per-element with affine, `Result<f32, KernelError>`
//! - `instance_norm_affine_ref` — affine vector reference, `Result<Vec<f32>, KernelError>`

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
// Re-export for test module's `use super::*;`
#[cfg(test)]
use crate::ir::ScalarType;

// --- Dimension validation ---

pub(crate) fn validate_bct(b: usize, c: usize, t: usize) -> Result<(), KernelError> {
    validate_nonzero_dims(&[("b", b), ("c", c), ("t", t)])
}

// --- K2 InstanceNorm builder ---

/// Build the InstanceNorm (K2) `TensorKernelDef` for shape `[B, C, T]` using the
/// native `InstanceNorm1d` tensor op.
///
/// This produces a compact 3-node IR (x input, eps input, InstanceNorm1d op) that
/// maps directly to NY's `InstanceNorm1dLayer` for tight bound propagation.
/// For backends that need per-op dispatch (MSL codegen, Kani), use
/// [`build_instance_norm_decomposed`] which expands into 12 scalar/reduce nodes.
///
/// # Errors
///
/// Returns [`TensorIRError::KernelValidation`] if any dimension (`b`, `c`, or `t`) is 0.
#[must_use = "returns a Result that may contain an error"]
pub fn build_instance_norm(b: usize, c: usize, t: usize) -> Result<TensorKernelDef, TensorIRError> {
    validate_bct(b, c, t)?;

    let full = vec![b, c, t];

    Ok(TensorKernelDef {
        name: "instance_norm".into(),
        nodes: vec![
            input_node(0, "x", &full),
            input_node(1, "eps", &[1]),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::InstanceNorm1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 2, // normalize over T (last axis)
                    gamma: None,
                    beta: None,
                },
                full,
            ),
        ],
        output: TensorNodeId::new(2),
    })
}

/// Build the InstanceNorm (K2) `TensorKernelDef` in decomposed form for shape `[B, C, T]`.
///
/// Decomposes into 12 nodes: 2 inputs, 2 reductions, 3 broadcasts, 5 element-wise.
/// Uses two-pass centered variance `mean((x - mean)²)` to avoid catastrophic
/// cancellation for large inputs (see #102).
///
/// This decomposed form is needed for:
/// - MSL codegen (per-op kernel dispatch)
/// - Kani verification (proving individual ops correct)
///
/// For NY verification, prefer [`build_instance_norm`] which uses the
/// native `InstanceNorm1d` op for tighter bounds.
///
/// See `designs/2026-02-26-kernelir-tensor-ops.md` Phase E for the full IR layout.
///
/// # Errors
///
/// Returns [`TensorIRError::KernelValidation`] if any dimension (`b`, `c`, or `t`) is 0.
#[must_use = "returns a Result that may contain an error"]
pub fn build_instance_norm_decomposed(
    b: usize,
    c: usize,
    t: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    validate_bct(b, c, t)?;

    let full = vec![b, c, t];
    let reduced = vec![b, c];

    // Two-pass centered variance: var = mean((x - mean)²)
    // Node 4 (x - mean) is reused for both variance and final normalization.
    Ok(TensorKernelDef {
        name: "instance_norm".into(),
        nodes: vec![
            input_node(0, "x", &full),                             // x [B,C,T]
            input_node(1, "eps", &[1]),                            // eps scalar [1]
            reduce_node(2, ReduceOp::Mean, 0, 2, &reduced),        // mean(x) [B,C]
            broadcast_node(3, 2, &full, BroadcastAlignment::Left), // mean → [B,C,T]
            elementwise_node(4, binop_kernel("sub", BinOpKind::Sub), &[0, 3], &full), // x - mean [B,C,T]
            elementwise_node(5, square_kernel(), &[4], &full), // (x-mean)² [B,C,T]
            reduce_node(6, ReduceOp::Mean, 5, 2, &reduced),    // var [B,C]
            broadcast_node(7, 6, &full, BroadcastAlignment::Left), // var → [B,C,T]
            broadcast_node(8, 1, &full, BroadcastAlignment::Left), // eps → [B,C,T]
            elementwise_node(9, binop_kernel("add", BinOpKind::Add), &[7, 8], &full), // var+eps [B,C,T]
            elementwise_node(10, unary_kernel("rsqrt", UnaryFnKind::Rsqrt), &[9], &full), // rsqrt(var+eps) [B,C,T]
            elementwise_node(11, binop_kernel("mul", BinOpKind::Mul), &[4, 10], &full), // (x-mean) * rsqrt [B,C,T]
        ],
        output: TensorNodeId::new(11),
    })
}

// --- K2 InstanceNorm scalar element kernel ---

/// Build the InstanceNorm (K2) scalar element `KernelDef`.
///
/// Parameters: `x`, `mean`, `var_val`, `eps` (4 params).
/// Computes: `(x - mean) * (var_val + eps).rsqrt()`
///
/// This encodes the per-element computation after the two-pass reduction
/// computes `mean` and `var`. Uses `rsqrt` → UF approximation in ay.
///
/// Unlike LayerNorm (K7), InstanceNorm has no affine transform (no gamma/beta).
///
/// # Errors
///
/// Returns [`LowerError`] if the kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_instance_norm_scalar_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn instance_norm_scalar(x: f32, mean: f32, var_val: f32, eps: f32) -> f32 {
            (x - mean) * (var_val + eps).rsqrt()
        }",
    )
}

/// Scalar per-element InstanceNorm: `(x - mean) * rsqrt(var_val + eps)`.
///
/// This is the non-affine variant (no gamma/beta). For the affine variant, see
/// [`instance_norm_affine_scalar`].
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is NaN or infinite.
/// Returns [`KernelError::InvalidEps`] if `var_val + eps <= 0` (division by zero/NaN).
/// Returns [`KernelError::NonFiniteOutput`] if the result overflows to infinity.
#[must_use = "returns a Result that may contain an error"]
pub fn instance_norm_scalar(x: f32, mean: f32, var_val: f32, eps: f32) -> Result<f32, KernelError> {
    validate_finite_inputs(&[("x", x), ("mean", mean), ("var_val", var_val), ("eps", eps)])?;
    // Guard against sqrt(0) → division by zero, and sqrt(negative) → NaN.
    // Matches adain_scalar pattern (adain.rs:122-125) and affine_normalize_scalar.
    let denom_input = var_val + eps;
    if denom_input <= 0.0 {
        return Err(KernelError::InvalidEps { value: eps });
    }
    checked_scalar_output((x - mean) * denom_input.sqrt().recip())
}

// --- Reference implementation ---

/// Rust reference implementation of InstanceNorm for differential testing.
///
/// Computes `(x - mean) / sqrt(var + eps)` per channel over the time axis.
///
/// # Errors
///
/// Returns [`KernelError::InvalidDimension`] if any dimension is 0.
/// Returns [`KernelError::DimensionExceedsF32Precision`] if `t > 2^24` (mean/var uses `t as f32`).
/// Returns [`KernelError::DimensionOverflow`] if `b * c * t` overflows `usize`.
/// Returns [`KernelError::InvalidEps`] if `eps <= 0.0` or `eps` is not finite.
/// Returns [`KernelError::ShapeMismatch`] if `x.len() != b * c * t`.
/// Returns [`KernelError::NonFiniteSliceElement`] if any element of `x` is NaN or Inf.
/// Returns [`KernelError::NonFiniteSliceOutput`] if any output element overflows.
#[must_use = "returns a Result that may contain an error"]
pub fn instance_norm_ref(
    x: &[f32],
    b: usize,
    c: usize,
    t: usize,
    eps: f32,
) -> Result<Vec<f32>, KernelError> {
    validate_bct(b, c, t)?;
    // usize→f32 cast in mean/variance is only lossless for t <= 2^24.
    // Above that threshold, `t as f32` silently loses precision, producing
    // incorrect mean and variance values.
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
    validate_finite_slice("x", x)?;

    let mut output = vec![0.0f32; expected_len];

    for bi in 0..b {
        for ci in 0..c {
            let offset = (bi * c + ci) * t;
            let slice = &x[offset..offset + t];

            // SAFETY: t <= F32_PRECISION_LIMIT (2^24) is enforced above.
            let t_f32 = t as f32;
            let mean: f32 = slice.iter().sum::<f32>() / t_f32;
            let var: f32 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / t_f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            for ti in 0..t {
                output[offset + ti] = (slice[ti] - mean) * inv_std;
            }
        }
    }

    checked_slice_output(&output)?;
    Ok(output)
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proves `instance_norm_scalar` produces finite output for bounded inputs.
    ///
    /// Domain: x ∈ [-1e3, 1e3], mean ∈ [-1e3, 1e3], var ∈ [0, 1e6],
    /// eps ∈ [1e-8, 1.0].
    fn sqrt_f32_stub(x: f32) -> f32 {
        let r: f32 = kani::any();
        kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
        if x > 0.0 {
            kani::assume(r > 0.0);
            kani::assume(r >= x.min(1.0));
        }
        r
    }

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn instance_norm_scalar_finite_for_bounded_inputs() {
        let x: f32 = kani::any();
        let mean: f32 = kani::any();
        let var_val: f32 = kani::any();
        let eps: f32 = kani::any();

        kani::assume(x.is_finite() && x >= -1.0e3 && x <= 1.0e3);
        kani::assume(mean.is_finite() && mean >= -1.0e3 && mean <= 1.0e3);
        kani::assume(var_val.is_finite() && var_val >= 0.0 && var_val <= 1.0e6);
        kani::assume(eps.is_finite() && eps >= 1.0e-8 && eps <= 1.0);

        let y = instance_norm_scalar(x, mean, var_val, eps)
            .expect("must succeed for bounded finite inputs");
        assert!(y.is_finite(), "must produce finite output");
        assert!(!y.is_nan(), "must not produce NaN");
    }

    /// Proves that zero variance with positive eps still produces finite output.
    ///
    /// When var_val = 0 and eps > 0, rsqrt(eps) is well-defined and finite.
    /// With mean = x, the result should be 0.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn instance_norm_scalar_zero_variance_safe() {
        let x: f32 = kani::any();
        let eps: f32 = kani::any();

        kani::assume(x.is_finite() && x >= -1.0e3 && x <= 1.0e3);
        kani::assume(eps.is_finite() && eps >= 1.0e-8 && eps <= 1.0);

        // mean = x, var = 0: all values are the same, output should be ~0
        let y =
            instance_norm_scalar(x, x, 0.0, eps).expect("must succeed for bounded finite inputs");
        assert!(y.is_finite(), "zero-variance output must be finite");
        assert!(
            y.abs() < 1e-3,
            "zero-variance output should be ~0 (x - mean = 0)"
        );
    }

    /// Proves output is bounded for positive variance inputs.
    ///
    /// With var ∈ [0, 1e4] and eps ∈ [1e-5, 1], rsqrt(var + eps) ∈ [0.01, 316].
    /// Combined with (x - mean) ∈ [-200, 200], output is bounded.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn instance_norm_scalar_positive_variance_bounded() {
        let x: f32 = kani::any();
        let mean: f32 = kani::any();
        let var_val: f32 = kani::any();
        let eps: f32 = kani::any();

        kani::assume(x.is_finite() && x >= -100.0 && x <= 100.0);
        kani::assume(mean.is_finite() && mean >= -100.0 && mean <= 100.0);
        kani::assume(var_val.is_finite() && var_val >= 0.0 && var_val <= 1.0e4);
        kani::assume(eps.is_finite() && eps >= 1.0e-5 && eps <= 1.0);

        let y = instance_norm_scalar(x, mean, var_val, eps)
            .expect("must succeed for bounded finite inputs");
        assert!(y.is_finite(), "output must be finite");
        assert!(y.abs() <= 7.0e4, "output must be bounded");
    }
}

// Affine variants extracted to instance_norm_affine.rs (500-line file limit).
#[path = "instance_norm_affine.rs"]
pub(crate) mod affine;
pub use affine::build_instance_norm_affine_scalar_kernel;
#[allow(unused_imports)]
pub(crate) use affine::{
    build_instance_norm_affine, build_instance_norm_decomposed_affine, instance_norm_affine_ref,
    instance_norm_affine_scalar,
};

#[cfg(kani)]
#[path = "instance_norm_kani_builder_tests.rs"]
mod kani_builder_proofs;

#[cfg(test)]
#[path = "instance_norm_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "instance_norm_affine_tests.rs"]
mod affine_tests;
