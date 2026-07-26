// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared kernel utility functions — validation and scalar kernel building.
//!
//! Extracts duplicated boilerplate from kernel files (layer_norm, rms_norm,
//! instance_norm, rope, silu_mul) into a single source of truth.
//!
//! Part of #340 (kernel shared utilities).

use crate::ir::KernelDef;
use crate::kernel_error::KernelError;
use crate::lower::{LowerError, Lowerer};

/// Validate that all named dimensions are nonzero.
///
/// Replaces per-kernel `validate_dims` functions that repeat the same
/// zero-check pattern with different parameter names.
///
/// # Errors
///
/// Returns [`KernelError::InvalidDimension`] for the first dimension that is zero.
pub(crate) fn validate_nonzero_dims(dims: &[(&'static str, usize)]) -> Result<(), KernelError> {
    for &(name, value) in dims {
        if value == 0 {
            return Err(KernelError::InvalidDimension { name, value: 0 });
        }
    }
    Ok(())
}

/// Build a scalar `KernelDef` from a Rust function source string.
///
/// Parses the source as a `syn::ItemFn` and lowers it to IR via [`Lowerer::lower_fn`].
/// Replaces the repeated `syn::parse_str` + `Lowerer::lower_fn` two-liner in each
/// `build_*_scalar_kernel()` function.
///
/// # Errors
///
/// Returns [`LowerError`] if the source fails to parse or lower.
pub(crate) fn build_scalar_kernel(src: &str) -> Result<KernelDef, LowerError> {
    let func: syn::ItemFn = syn::parse_str(src)?;
    Lowerer::lower_fn(&func)
}

/// Validate that all named scalar inputs are finite.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] for the first input that is NaN or infinite.
pub(crate) fn validate_finite_inputs(inputs: &[(&'static str, f32)]) -> Result<(), KernelError> {
    for &(name, value) in inputs {
        if !value.is_finite() {
            return Err(KernelError::NonFiniteInput { name, value });
        }
    }
    Ok(())
}

/// Validate that all elements of a named `f32` slice are finite.
///
/// Used by tensor-level reference implementations (instance_norm_ref, layer_norm_ref,
/// rms_norm_ref, rope_rotate_ref) to reject NaN/Inf inputs per element.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteSliceElement`] for the first element that is NaN or infinite.
pub(crate) fn validate_finite_slice(name: &'static str, values: &[f32]) -> Result<(), KernelError> {
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(KernelError::NonFiniteSliceElement {
                name,
                index: i,
                value: v,
            });
        }
    }
    Ok(())
}

/// Check that all elements of an output slice are finite.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteSliceOutput`] for the first non-finite output element.
pub(crate) fn checked_slice_output(output: &[f32]) -> Result<(), KernelError> {
    for (i, &v) in output.iter().enumerate() {
        if !v.is_finite() {
            return Err(KernelError::NonFiniteSliceOutput { index: i, value: v });
        }
    }
    Ok(())
}

/// Shared affine normalization: `(x - mean) / sqrt(var + eps) * gamma + beta`.
///
/// Used by `layer_norm_scalar` and `instance_norm_affine_scalar`, which are
/// mathematically identical at the scalar level.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is non-finite.
/// Returns [`KernelError::InvalidEps`] if `var + eps <= 0` (division by zero/NaN).
/// Returns [`KernelError::NonFiniteOutput`] if the result overflows.
pub(crate) fn affine_normalize_scalar(
    x: f32,
    mean: f32,
    var: f32,
    eps: f32,
    gamma: f32,
    beta: f32,
) -> Result<f32, KernelError> {
    validate_finite_inputs(&[
        ("x", x),
        ("mean", mean),
        ("var", var),
        ("eps", eps),
        ("gamma", gamma),
        ("beta", beta),
    ])?;
    // Guard against sqrt(0) → division by zero, and sqrt(negative) → NaN.
    // Matches the adain_scalar pattern (adain.rs:122-125).
    let denom_input = var + eps;
    if denom_input <= 0.0 {
        return Err(KernelError::InvalidEps { value: eps });
    }
    let inv_std = 1.0 / denom_input.sqrt();
    checked_scalar_output((x - mean) * inv_std * gamma + beta)
}

/// Check that a computed scalar result is finite, returning it on success.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteOutput`] if the result is NaN or infinite.
pub(crate) fn checked_scalar_output(result: f32) -> Result<f32, KernelError> {
    if result.is_finite() {
        Ok(result)
    } else {
        Err(KernelError::NonFiniteOutput {
            name: "output",
            value: result,
        })
    }
}

// --- Bounds validation helpers (Part of #391) ---

/// Maximum reduction dimension before `usize as f32` loses integer precision.
///
/// `2^24 = 16_777_216` is the largest integer exactly representable in f32.
/// Reduction dimensions beyond this threshold cause silent precision loss
/// in mean/variance computation via `t as f32`.
pub(crate) const F32_PRECISION_LIMIT: usize = 1 << 24;

/// Validate all bound values are finite and each (lo, hi) pair is non-inverted.
///
/// Combines the finiteness check and inverted-bounds check into one call,
/// replacing duplicated inline patterns in snake.rs, silu_mul.rs, rope_bounds.rs.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteBound`] for the first non-finite value, or
/// [`KernelError::InvertedBounds`] for the first pair where `lo > hi`.
pub(crate) fn validate_bounds_pairs(pairs: &[(f32, f32)]) -> Result<(), KernelError> {
    for &(lo, hi) in pairs {
        if !lo.is_finite() {
            return Err(KernelError::NonFiniteBound { value: lo });
        }
        if !hi.is_finite() {
            return Err(KernelError::NonFiniteBound { value: hi });
        }
        if lo > hi {
            return Err(KernelError::InvertedBounds {
                lower: lo,
                upper: hi,
            });
        }
    }
    Ok(())
}

/// Validate computed output bounds are finite.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteBound`] if either bound is NaN or infinite.
pub(crate) fn validate_bounds_output(lower: f32, upper: f32) -> Result<(f32, f32), KernelError> {
    if !lower.is_finite() {
        return Err(KernelError::NonFiniteBound { value: lower });
    }
    if !upper.is_finite() {
        return Err(KernelError::NonFiniteBound { value: upper });
    }
    Ok((lower, upper))
}

/// Validate epsilon is finite and strictly positive.
///
/// # Errors
///
/// Returns [`KernelError::InvalidEps`] if eps is NaN, infinite, zero, or negative.
pub(crate) fn validate_eps(eps: f32) -> Result<(), KernelError> {
    if !eps.is_finite() || eps <= 0.0 {
        return Err(KernelError::InvalidEps { value: eps });
    }
    Ok(())
}

#[cfg(test)]
#[path = "kernel_util_tests.rs"]
mod tests;
