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
#[allow(dead_code)] // Called from #[cfg(test)] and #[cfg(kani)] only
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
#[allow(dead_code)] // Called from #[cfg(test)] and #[cfg(kani)] only
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

#[cfg(kani)]
mod kani_proofs {
    //! Kani proof harnesses for ReLU kernel correctness.
    //!
    //! ReLU uses only `f32::max(0.0)` — pure comparison, no transcendentals.
    //! No stubs needed (unlike sigmoid/gelu/tanh which use exp/tanh).

    use super::*;

    /// Prove ReLU produces finite non-negative output for all finite inputs.
    ///
    /// Domain: all finite f32 values.
    /// Properties: output is finite, output >= 0.0.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(3)]
    fn relu_finite_nonnegative_for_all_finite_inputs() {
        let x: f32 = kani::any();
        kani::assume(x.is_finite());

        let result = relu_scalar(x).expect("relu_scalar must succeed for finite inputs");
        assert!(result.is_finite(), "relu must produce finite output");
        assert!(result >= 0.0, "relu output must be >= 0");
    }

    /// Prove ReLU is idempotent: relu(relu(x)) == relu(x).
    ///
    /// Since relu(x) >= 0, applying relu again is a no-op.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(3)]
    fn relu_idempotent() {
        let x: f32 = kani::any();
        kani::assume(x.is_finite());

        let first = relu_scalar(x).expect("relu_scalar must succeed");
        let second = relu_scalar(first).expect("relu_scalar must succeed on non-negative");

        assert_eq!(first.to_bits(), second.to_bits(), "relu must be idempotent");
    }

    /// Prove ReLU bounds algorithm is sound: for any x in [lo, hi],
    /// relu(x) is within [relu(lo), relu(hi)].
    ///
    /// Exploits monotonicity of max(·, 0): if lo <= x <= hi,
    /// then max(lo, 0) <= max(x, 0) <= max(hi, 0).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(3)]
    fn relu_bounds_sound() {
        let x: f32 = kani::any();
        let x_lo: f32 = kani::any();
        let x_hi: f32 = kani::any();

        kani::assume(x.is_finite() && x_lo.is_finite() && x_hi.is_finite());
        kani::assume(x_lo <= x && x <= x_hi);

        let result = relu_scalar(x).expect("relu_scalar must succeed");
        let (lower, upper) = relu_scalar_bounds(x_lo, x_hi).expect("finite ordered inputs");

        assert!(
            result >= lower,
            "relu output must be >= lower bound (monotone)"
        );
        assert!(
            result <= upper,
            "relu output must be <= upper bound (monotone)"
        );
    }

    /// Prove ReLU rejects NaN input.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(3)]
    fn relu_rejects_nan() {
        let result = relu_scalar(f32::NAN);
        assert!(result.is_err(), "relu must reject NaN");
    }

    /// Prove ReLU rejects infinite input.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(3)]
    fn relu_rejects_infinity() {
        let result_pos = relu_scalar(f32::INFINITY);
        let result_neg = relu_scalar(f32::NEG_INFINITY);
        assert!(result_pos.is_err(), "relu must reject +inf");
        assert!(result_neg.is_err(), "relu must reject -inf");
    }
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
