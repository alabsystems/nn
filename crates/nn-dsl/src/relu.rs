// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ReLU kernel — elementwise rectified linear unit.
//!
//! # ReLU formula
//!
//! ```text
//! relu(x) = max(x, 0.0)
//! ```
//!
//! Piecewise linear: `0` for `x < 0`, `x` for `x >= 0`.
//! Used in Silero VAD encoder blocks and after LSTM hidden state (#761).
//!
//! # Naming convention (#336)
//!
//! - `relu_scalar` — per-element scalar, `Result<f32, KernelError>`
//! - `relu_ref` — vector reference, `Result<Vec<f32>, KernelError>`
//! - `build_relu_kernel` — `KernelDef` IR builder
//! - `relu_scalar_bounds` — analytical output bounds for NY

use crate::ir::KernelDef;
use crate::kernel_error::KernelError;
use crate::kernel_util::{
    build_scalar_kernel, checked_scalar_output, validate_bounds_output, validate_bounds_pairs,
    validate_finite_inputs,
};
use crate::lower::LowerError;

/// Build the ReLU scalar `KernelDef`.
///
/// Parameters: `x` (1 param).
/// Computes: `x.max(0.0)`
///
/// # Errors
///
/// Returns [`LowerError`] if the hardcoded kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_relu_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn relu(x: f32) -> f32 {
            x.max(0.0)
        }",
    )
}

/// Scalar ReLU reference implementation.
///
/// `relu(x) = max(x, 0.0)`
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if input is NaN or infinite.
/// Returns [`KernelError::NonFiniteOutput`] if the computed result is non-finite.
#[allow(dead_code)] // Called from #[cfg(test)] only
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn relu_scalar(x: f32) -> Result<f32, KernelError> {
    validate_finite_inputs(&[("x", x)])?;

    let result = x.max(0.0);

    checked_scalar_output(result)
}

/// Compute analytical output bounds for ReLU.
///
/// ReLU is monotonically non-decreasing, so bounds are
/// `(relu(x_lo), relu(x_hi))` = `(max(x_lo, 0), max(x_hi, 0))`.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteBound`] if any input is NaN or infinity.
/// Returns [`KernelError::InvertedBounds`] if `x_lo > x_hi`.
#[allow(dead_code)] // Called from #[cfg(test)] only
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn relu_scalar_bounds(x_lo: f32, x_hi: f32) -> Result<(f32, f32), KernelError> {
    validate_bounds_pairs(&[(x_lo, x_hi)])?;

    let lower = x_lo.max(0.0);
    let upper = x_hi.max(0.0);

    validate_bounds_output(lower, upper)
}

/// 1d ReLU over a flat array.
///
/// # Errors
///
/// Returns [`KernelError`] if the array is empty or if any element is non-finite.
#[allow(dead_code)] // Called from #[cfg(test)] only
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn relu_ref(x: &[f32]) -> Result<Vec<f32>, KernelError> {
    if x.is_empty() {
        return Err(KernelError::InvalidDimension {
            name: "total",
            value: 0,
        });
    }
    x.iter().map(|&xi| relu_scalar(xi)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_scalar_positive() {
        let result = relu_scalar(3.0).unwrap();
        assert!((result - 3.0).abs() < 1e-7);
    }

    #[test]
    fn test_relu_scalar_negative() {
        let result = relu_scalar(-2.0).unwrap();
        assert!(result.abs() < 1e-7);
    }

    #[test]
    fn test_relu_scalar_zero() {
        let result = relu_scalar(0.0).unwrap();
        assert!(result.abs() < 1e-7);
    }

    #[test]
    fn test_relu_scalar_nan() {
        assert!(relu_scalar(f32::NAN).is_err());
    }

    #[test]
    fn test_relu_scalar_inf() {
        assert!(relu_scalar(f32::INFINITY).is_err());
    }

    #[test]
    fn test_relu_bounds_both_positive() {
        let (lo, hi) = relu_scalar_bounds(1.0, 3.0).unwrap();
        assert!((lo - 1.0).abs() < 1e-7);
        assert!((hi - 3.0).abs() < 1e-7);
    }

    #[test]
    fn test_relu_bounds_both_negative() {
        let (lo, hi) = relu_scalar_bounds(-3.0, -1.0).unwrap();
        assert!(lo.abs() < 1e-7);
        assert!(hi.abs() < 1e-7);
    }

    #[test]
    fn test_relu_bounds_straddles_zero() {
        let (lo, hi) = relu_scalar_bounds(-2.0, 3.0).unwrap();
        assert!(lo.abs() < 1e-7);
        assert!((hi - 3.0).abs() < 1e-7);
    }

    #[test]
    fn test_relu_ref_basic() {
        let input = vec![-1.0, 0.0, 1.0, 2.0, -3.0];
        let result = relu_ref(&input).unwrap();
        assert_eq!(result.len(), 5);
        assert!(result[0].abs() < 1e-7);
        assert!(result[1].abs() < 1e-7);
        assert!((result[2] - 1.0).abs() < 1e-7);
        assert!((result[3] - 2.0).abs() < 1e-7);
        assert!(result[4].abs() < 1e-7);
    }

    #[test]
    fn test_relu_ref_empty() {
        assert!(relu_ref(&[]).is_err());
    }

    #[test]
    fn test_build_relu_kernel() {
        let def = build_relu_kernel().unwrap();
        assert_eq!(def.params.len(), 1);
        assert_eq!(def.name, "relu");
    }
}
