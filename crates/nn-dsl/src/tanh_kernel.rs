// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tanh kernel — elementwise hyperbolic tangent.
//!
//! # Tanh formula
//!
//! ```text
//! tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
//! ```
//!
//! Monotonically increasing, output ∈ (-1, 1).
//! Used in LSTM gate decomposition (g-candidate and cell state output) (#761).
//!
//! # Naming convention (#336)
//!
//! - `tanh_scalar` — per-element scalar, `Result<f32, KernelError>`
//! - `tanh_ref` — vector reference, `Result<Vec<f32>, KernelError>`
//! - `build_tanh_kernel` — `KernelDef` IR builder
//! - `tanh_scalar_bounds` — analytical output bounds for NY

use crate::ir::KernelDef;
use crate::kernel_error::KernelError;
use crate::kernel_util::{
    build_scalar_kernel, checked_scalar_output, validate_bounds_output, validate_bounds_pairs,
    validate_finite_inputs,
};
use crate::lower::LowerError;

/// Build the Tanh scalar `KernelDef`.
///
/// Parameters: `x` (1 param).
/// Computes: `x.tanh()`
///
/// # Errors
///
/// Returns [`LowerError`] if the hardcoded kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_tanh_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn tanh_act(x: f32) -> f32 {
            x.tanh()
        }",
    )
}

/// Scalar tanh reference implementation.
///
/// `tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)`
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if input is NaN or infinite.
/// Returns [`KernelError::NonFiniteOutput`] if the computed result is non-finite.
#[must_use = "returns a Result that may contain an error"]
pub fn tanh_scalar(x: f32) -> Result<f32, KernelError> {
    validate_finite_inputs(&[("x", x)])?;

    let result = x.tanh();

    checked_scalar_output(result)
}

/// Compute analytical output bounds for tanh.
///
/// Tanh is monotonically increasing, so bounds are simply
/// `(tanh(x_lo), tanh(x_hi))`.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteBound`] if any input is NaN or infinity.
/// Returns [`KernelError::InvertedBounds`] if `x_lo > x_hi`.
#[must_use = "returns a Result that may contain an error"]
pub fn tanh_scalar_bounds(x_lo: f32, x_hi: f32) -> Result<(f32, f32), KernelError> {
    validate_bounds_pairs(&[(x_lo, x_hi)])?;

    let lower = x_lo.tanh();
    let upper = x_hi.tanh();

    validate_bounds_output(lower, upper)
}

#[cfg(test)]
/// 1d tanh over a flat array.
///
/// # Errors
///
/// Returns [`KernelError`] if the array is empty or if any element is non-finite.
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn tanh_ref(x: &[f32]) -> Result<Vec<f32>, KernelError> {
    if x.is_empty() {
        return Err(KernelError::InvalidDimension {
            name: "total",
            value: 0,
        });
    }
    x.iter().map(|&xi| tanh_scalar(xi)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanh_scalar_positive() {
        let result = tanh_scalar(1.0).unwrap();
        assert!((result - 1.0_f32.tanh()).abs() < 1e-7);
    }

    #[test]
    fn test_tanh_scalar_negative() {
        let result = tanh_scalar(-1.0).unwrap();
        assert!((result - (-1.0_f32).tanh()).abs() < 1e-7);
    }

    #[test]
    fn test_tanh_scalar_zero() {
        let result = tanh_scalar(0.0).unwrap();
        assert!(result.abs() < 1e-7);
    }

    #[test]
    fn test_tanh_scalar_large() {
        let result = tanh_scalar(10.0).unwrap();
        assert!((result - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_tanh_scalar_nan() {
        assert!(tanh_scalar(f32::NAN).is_err());
    }

    #[test]
    fn test_tanh_scalar_inf() {
        assert!(tanh_scalar(f32::INFINITY).is_err());
    }

    #[test]
    fn test_tanh_bounds_symmetric() {
        let (lo, hi) = tanh_scalar_bounds(-2.0, 2.0).unwrap();
        assert!((lo - (-2.0_f32).tanh()).abs() < 1e-7);
        assert!((hi - 2.0_f32.tanh()).abs() < 1e-7);
    }

    #[test]
    fn test_tanh_bounds_positive() {
        let (lo, hi) = tanh_scalar_bounds(0.5, 1.5).unwrap();
        assert!((lo - 0.5_f32.tanh()).abs() < 1e-7);
        assert!((hi - 1.5_f32.tanh()).abs() < 1e-7);
    }

    #[test]
    fn test_tanh_ref_basic() {
        let input = vec![-1.0, 0.0, 1.0];
        let result = tanh_ref(&input).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - (-1.0_f32).tanh()).abs() < 1e-7);
        assert!(result[1].abs() < 1e-7);
        assert!((result[2] - 1.0_f32.tanh()).abs() < 1e-7);
    }

    #[test]
    fn test_tanh_ref_empty() {
        assert!(tanh_ref(&[]).is_err());
    }

    #[test]
    fn test_build_tanh_kernel() {
        let def = build_tanh_kernel().unwrap();
        assert_eq!(def.params.len(), 1);
        assert_eq!(def.name, "tanh_act");
    }
}
