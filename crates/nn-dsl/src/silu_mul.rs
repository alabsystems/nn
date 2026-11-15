// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SiLU-Mul (K8) kernel — elementwise `silu(x) * up`.
//!
//! The simplest remaining dvoice kernel: flat `[total]` layout, no reduction,
//! no broadcast, no paired-element access. Follows the Snake (K1) pattern.
//!
//! # SiLU-Mul formula
//!
//! ```text
//! silu(x) = x / (1 + exp(-x))     // = x * sigmoid(x)
//! out     = silu(x) * up
//! ```
//!
//! Part of #19 (K2-K8 kernel ports).
//!
//! # Naming convention (#336)
//!
//! - `silu_mul_scalar` — per-element scalar, `Result<f32, KernelError>`
//! - `silu_mul_ref` — vector reference, `Result<Vec<f32>, KernelError>`
//! - `build_silu_mul_kernel` — `KernelDef` IR builder

use crate::ir::KernelDef;
use crate::kernel_error::KernelError;
use crate::kernel_util::{
    build_scalar_kernel, checked_scalar_output, validate_bounds_output, validate_bounds_pairs,
    validate_finite_inputs, validate_nonzero_dims,
};
use crate::lower::LowerError;
use crate::tensor_builders::{elementwise_node, input_node};
use crate::tensor_ir::{TensorIRError, TensorKernelDef, TensorNodeId};

/// Build the SiLU-Mul (K8) scalar KernelDef.
///
/// Parameters: `x`, `up` (2 params).
/// Computes: `(x / (1.0 + (-x).exp())) * up`
///
/// Uses the explicit sigmoid form to match the dvoice MSL kernel
/// and avoid relying on a `sigmoid` intrinsic that may not be available.
///
/// # Errors
///
/// Returns [`LowerError`] if the hardcoded kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_silu_mul_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn silu_mul(x: f32, up: f32) -> f32 {
            (x / (1.0 + (-x).exp())) * up
        }",
    )
}

/// Scalar SiLU-Mul reference implementation.
///
/// `silu_mul(x, up) = (x * sigmoid(x)) * up`
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is NaN or infinite.
/// Returns [`KernelError::NonFiniteOutput`] if the computed result is non-finite
/// despite all inputs being finite (e.g., extreme magnitudes).
#[must_use = "returns a Result that may contain an error"]
pub fn silu_mul_scalar(x: f32, up: f32) -> Result<f32, KernelError> {
    validate_finite_inputs(&[("x", x), ("up", up)])?;

    let sigmoid = 1.0 / (1.0 + (-x).exp());
    let result = x * sigmoid * up;

    checked_scalar_output(result)
}

/// x-coordinate of the SiLU global minimum, where d/dx[silu(x)] = 0.
///
/// silu(x) decreases for x < SILU_ARGMIN, increases for x > SILU_ARGMIN.
/// The minimum value is silu(SILU_ARGMIN) ≈ -0.2785.
const SILU_ARGMIN: f32 = -1.278_464_5;

/// Compute conservative output bounds for SiLU-Mul.
///
/// `silu(x)` is **not** monotonically increasing — it has a global minimum
/// at `x ≈ -1.278` where `silu ≈ -0.278`. For `x < -1.278`, silu decreases
/// toward 0 as x → -∞.
///
/// To get sound bounds, we evaluate silu at both endpoints **and** at the
/// global minimum when the input range spans it, then take all corner
/// products with the up bounds.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteBound`] if any input is NaN or infinity,
/// or if the computed output bounds overflow to infinity.
/// Returns [`KernelError::InvertedBounds`] if `x_lo > x_hi` or `up_lo > up_hi`.
#[must_use = "returns a Result that may contain an error"]
pub fn silu_mul_scalar_bounds(
    x_lo: f32,
    x_hi: f32,
    up_lo: f32,
    up_hi: f32,
) -> Result<(f32, f32), KernelError> {
    validate_bounds_pairs(&[(x_lo, x_hi), (up_lo, up_hi)])?;

    // Collect silu extrema: endpoints plus global minimum if in range.
    let silu_lo = silu_scalar(x_lo);
    let silu_hi = silu_scalar(x_hi);
    let include_argmin = x_lo < SILU_ARGMIN && x_hi > SILU_ARGMIN;

    let mut lower = f32::INFINITY;
    let mut upper = f32::NEG_INFINITY;

    // Helper: update lower/upper from a silu value × both up bounds.
    let mut add_corners = |sv: f32| {
        for &uv in &[up_lo, up_hi] {
            let p = sv * uv;
            if p < lower {
                lower = p;
            }
            if p > upper {
                upper = p;
            }
        }
    };

    add_corners(silu_lo);
    add_corners(silu_hi);
    if include_argmin {
        add_corners(silu_scalar(SILU_ARGMIN));
    }

    // Guard: corner products can overflow to Inf for large-magnitude inputs.
    validate_bounds_output(lower, upper)
}

/// Scalar SiLU (without the multiply).
#[must_use]
fn silu_scalar(x: f32) -> f32 {
    let sigmoid = 1.0 / (1.0 + (-x).exp());
    x * sigmoid
}

/// 1d SiLU-Mul over flat arrays.
///
/// Computes `silu(x[i]) * up[i]` for all `i`.
///
/// # Errors
///
/// Returns [`KernelError`] if `x` and `up` have different lengths or are empty,
/// or if any element is non-finite.
#[allow(dead_code)] // Called from #[cfg(test)] only
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn silu_mul_ref(x: &[f32], up: &[f32]) -> Result<Vec<f32>, KernelError> {
    if x.is_empty() {
        return Err(KernelError::InvalidDimension {
            name: "total",
            value: 0,
        });
    }
    if x.len() != up.len() {
        return Err(KernelError::ShapeMismatch {
            expected: x.len(),
            got: up.len(),
        });
    }
    x.iter()
        .zip(up.iter())
        .map(|(&xi, &ui)| silu_mul_scalar(xi, ui))
        .collect()
}

/// Build the SiLU-Mul (K8) `TensorKernelDef` for shape `[N, dim]`.
///
/// 3 nodes: x (input `[N, dim]`), up (input `[N, dim]`),
/// elementwise silu_mul(x, up).
///
/// SiLU-Mul is purely element-wise with no broadcast or reduction.
/// Both inputs share the same shape. Maps to NY graph translation
/// via `kernel_to_graph` on the inner scalar `KernelDef`.
///
/// # Arguments
///
/// * `n` — Batch/sequence dimension.
/// * `dim` — Feature dimension.
///
/// # Errors
///
/// Returns [`TensorIRError::KernelValidation`] if `n` or `dim` is 0.
/// Returns [`TensorIRError::ScalarKernelBuild`] if the scalar kernel builder fails.
#[allow(dead_code)] // Called from #[cfg(test)] only
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn build_silu_mul_tensor(
    n: usize,
    dim: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    validate_nonzero_dims(&[("n", n), ("dim", dim)])?;

    let shape = vec![n, dim];
    let kernel =
        build_silu_mul_kernel().map_err(|e| TensorIRError::ScalarKernelBuild(e.to_string()))?;

    Ok(TensorKernelDef {
        name: "silu_mul".into(),
        nodes: vec![
            input_node(0, "x", &shape),                   // x [N, dim]
            input_node(1, "up", &shape),                  // up [N, dim]
            elementwise_node(2, kernel, &[0, 1], &shape), // silu_mul(x, up)
        ],
        output: TensorNodeId::new(2),
    })
}

#[cfg(test)]
#[path = "silu_mul_tests.rs"]
mod tests;
