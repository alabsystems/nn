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

#[cfg(all(kani, feature = "kani-stubbing"))]
mod kani_proofs {
    //! Kani proof harnesses for Sigmoid kernel correctness.
    //!
    //! Uses `exp_stub` / `exp_det_stub` to work around CBMC's inaccurate
    //! `f32::exp()` model (#239). Same pattern as gelu and silu_mul.

    use super::*;
    use crate::kani_stubs::{exp_det_stub, exp_stub};

    /// Prove sigmoid produces finite output in [0, 1] for bounded inputs.
    ///
    /// Domain: x in [-100, 100].
    ///
    /// Note: Uses `>=`/`<=` (closed interval) because IEEE 754 underflow in
    /// exp(-x) can produce exactly 0 or +inf, yielding sigmoid == 1.0 or 0.0.
    /// For x = 88, exp(-88) underflows to 0 → sigmoid = 1/(1+0) = 1.0 exactly.
    /// For x = -88, exp(88) overflows to inf → sigmoid = 1/inf = 0.0 exactly.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(3)]
    #[kani::stub(f32::exp, exp_stub)]
    fn sigmoid_finite_for_bounded_inputs() {
        let x: f32 = kani::any();
        kani::assume(x.is_finite() && x >= -100.0 && x <= 100.0);

        let result =
            sigmoid_scalar(x).expect("sigmoid_scalar must succeed for bounded finite inputs");
        assert!(result.is_finite(), "sigmoid must produce finite output");
        assert!(result >= 0.0, "sigmoid output must be >= 0");
        assert!(result <= 1.0, "sigmoid output must be <= 1");
    }

    /// Prove sigmoid bounds algorithm is sound: for any x in [lo, hi],
    /// sigmoid(x) is within [sigmoid(lo), sigmoid(hi)].
    ///
    /// Domain: x in [-5, 5]. Exploits monotonicity.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(3)]
    #[kani::stub(f32::exp, exp_det_stub)]
    fn sigmoid_bounds_sound() {
        let x: f32 = kani::any();
        let x_lo: f32 = kani::any();
        let x_hi: f32 = kani::any();

        kani::assume(x.is_finite() && x_lo.is_finite() && x_hi.is_finite());
        kani::assume(x >= -5.0 && x <= 5.0);
        kani::assume(x_lo >= -5.0 && x_lo <= x && x <= x_hi && x_hi <= 5.0);

        let result =
            sigmoid_scalar(x).expect("sigmoid_scalar must succeed for bounded finite inputs");
        let (lower, upper) = sigmoid_scalar_bounds(x_lo, x_hi).expect("finite inputs");

        assert!(
            result >= lower - 1e-5,
            "sigmoid output must be >= lower bound"
        );
        assert!(
            result <= upper + 1e-5,
            "sigmoid output must be <= upper bound"
        );
    }
}

#[cfg(kani)]
#[path = "sigmoid_kani_builder.rs"]
mod kani_builder_proofs;

#[cfg(test)]
#[path = "sigmoid_tests.rs"]
mod tests;
