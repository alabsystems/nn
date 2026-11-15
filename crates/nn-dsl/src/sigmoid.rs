// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Sigmoid kernel — elementwise logistic function.
//!
//! # Sigmoid formula
//!
//! ```text
//! sigmoid(x) = 1 / (1 + exp(-x))
//! ```
//!
//! Monotonically increasing everywhere, output ∈ (0, 1).
//! Used in GLU gates, binary classification, and attention mechanisms.
//!
//! Part of #645.
//!
//! # Naming convention (#336)
//!
//! - `sigmoid_scalar` — per-element scalar, `Result<f32, KernelError>`
//! - `sigmoid_ref` — vector reference, `Result<Vec<f32>, KernelError>`
//! - `build_sigmoid_kernel` — `KernelDef` IR builder
//! - `sigmoid_scalar_bounds` — analytical output bounds for NY

use crate::ir::KernelDef;
use crate::kernel_error::KernelError;
use crate::kernel_util::{
    build_scalar_kernel, checked_scalar_output, validate_bounds_output, validate_bounds_pairs,
    validate_finite_inputs,
};
use crate::lower::LowerError;

/// Build the Sigmoid scalar `KernelDef`.
///
/// Parameters: `x` (1 param).
/// Computes: `1.0 / (1.0 + (-x).exp())`
///
/// # Errors
///
/// Returns [`LowerError`] if the hardcoded kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_sigmoid_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn sigmoid(x: f32) -> f32 {
            1.0 / (1.0 + (-x).exp())
        }",
    )
}

/// Scalar sigmoid reference implementation.
///
/// `sigmoid(x) = 1 / (1 + exp(-x))`
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if input is NaN or infinite.
/// Returns [`KernelError::NonFiniteOutput`] if the computed result is non-finite.
#[must_use = "returns a Result that may contain an error"]
pub fn sigmoid_scalar(x: f32) -> Result<f32, KernelError> {
    validate_finite_inputs(&[("x", x)])?;

    let result = 1.0 / (1.0 + (-x).exp());

    checked_scalar_output(result)
}

/// Compute analytical output bounds for sigmoid.
///
/// Sigmoid is monotonically increasing, so bounds are simply
/// `(sigmoid(x_lo), sigmoid(x_hi))`.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteBound`] if any input is NaN or infinity.
/// Returns [`KernelError::InvertedBounds`] if `x_lo > x_hi`.
#[must_use = "returns a Result that may contain an error"]
pub fn sigmoid_scalar_bounds(x_lo: f32, x_hi: f32) -> Result<(f32, f32), KernelError> {
    validate_bounds_pairs(&[(x_lo, x_hi)])?;

    let sigmoid_at = |x: f32| -> f32 { 1.0 / (1.0 + (-x).exp()) };

    let lower = sigmoid_at(x_lo);
    let upper = sigmoid_at(x_hi);

    validate_bounds_output(lower, upper)
}

#[cfg(test)]
/// 1d sigmoid over a flat array.
///
/// # Errors
///
/// Returns [`KernelError`] if the array is empty or if any element is non-finite.
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn sigmoid_ref(x: &[f32]) -> Result<Vec<f32>, KernelError> {
    if x.is_empty() {
        return Err(KernelError::InvalidDimension {
            name: "total",
            value: 0,
        });
    }
    x.iter().map(|&xi| sigmoid_scalar(xi)).collect()
}

#[cfg(test)]
#[path = "sigmoid_tests.rs"]
mod tests;
