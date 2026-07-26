// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GELU activation kernel — elementwise `gelu(x)`.
//!
//! Gaussian Error Linear Unit using the tanh approximation (matches PyTorch
//! default, TensorFlow, and dvoice/Kokoro production inference).
//!
//! # GELU formula (tanh approximation)
//!
//! ```text
//! gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
//! ```
//!
//! The tanh is expressed via exp for MSL codegen using the numerically
//! stable form: `tanh(z) = 1 - 2 / (exp(2z) + 1)`
//! (avoids inf/inf when exp overflows for large z)
//!
//! Part of #639.
//!
//! # Naming convention (#336)
//!
//! - `gelu_scalar` — per-element scalar, `Result<f32, KernelError>`
//! - `gelu_ref` — vector reference, `Result<Vec<f32>, KernelError>`
//! - `build_gelu_kernel` — `KernelDef` IR builder

use crate::ir::KernelDef;
use crate::kernel_error::KernelError;
use crate::kernel_util::{
    build_scalar_kernel, checked_scalar_output, validate_bounds_output, validate_bounds_pairs,
    validate_finite_inputs,
};
use crate::lower::LowerError;

/// Build the GELU scalar KernelDef.
///
/// Parameters: `x` (1 param).
/// Computes the tanh-approximation GELU via exp (no erf/tanh intrinsic needed).
///
/// # Errors
///
/// Returns [`LowerError`] if the hardcoded kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_gelu_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn gelu(x: f32) -> f32 {
            let k = 0.7978846;
            let inner = k * (x + 0.044715 * x * x * x);
            let e2 = (2.0 * inner).exp();
            0.5 * x * (2.0 - 2.0 / (e2 + 1.0))
        }",
    )
}

/// Scalar GELU reference implementation (tanh approximation).
///
/// `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if x is NaN or infinite.
/// Returns [`KernelError::NonFiniteOutput`] if the computed result is non-finite.
#[must_use = "returns a Result that may contain an error"]
#[cfg_attr(kani, kani::requires(
    x.is_finite() && x >= -100.0 && x <= 100.0
))]
#[cfg_attr(kani, kani::ensures(|result: &Result<f32, KernelError>|
    matches!(result, Ok(v) if v.is_finite())
))]
pub fn gelu_scalar(x: f32) -> Result<f32, KernelError> {
    validate_finite_inputs(&[("x", x)])?;
    checked_scalar_output(gelu_eval(x))
}

/// x-coordinate of the GELU global minimum (tanh approximation).
///
/// GELU decreases for x < GELU_ARGMIN, increases for x > GELU_ARGMIN.
/// The minimum value is gelu(GELU_ARGMIN) ≈ -0.1700.
/// Value from NY bisection on derivative (60 iterations).
const GELU_ARGMIN: f32 = -0.752_252_6;

/// Compute conservative output bounds for GELU.
///
/// GELU (tanh approximation) is **not** monotonically increasing — it has a
/// global minimum at `x ≈ -0.752` where `gelu ≈ -0.170`. For `x < -0.752`,
/// gelu decreases toward 0 as x → -∞.
///
/// To get sound bounds, we evaluate gelu at both endpoints **and** at the
/// global minimum when the input range spans it, then take min/max.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteBound`] if any input is NaN or infinity.
/// Returns [`KernelError::InvertedBounds`] if `x_lo > x_hi`.
#[must_use = "returns a Result that may contain an error"]
pub fn gelu_scalar_bounds(x_lo: f32, x_hi: f32) -> Result<(f32, f32), KernelError> {
    validate_bounds_pairs(&[(x_lo, x_hi)])?;

    let g_lo = gelu_eval(x_lo);
    let g_hi = gelu_eval(x_hi);

    let mut lower = g_lo.min(g_hi);
    let mut upper = g_lo.max(g_hi);

    if x_lo < GELU_ARGMIN && x_hi > GELU_ARGMIN {
        let g_min = gelu_eval(GELU_ARGMIN);
        lower = lower.min(g_min);
        upper = upper.max(g_min);
    }

    validate_bounds_output(lower, upper)
}

/// Raw GELU evaluation (tanh approximation, no validation).
///
/// Uses the numerically stable tanh form `1 - 2/(exp(2z) + 1)` to avoid
/// NaN from inf/inf when exp(2*inner) overflows for large |x|.
#[must_use]
fn gelu_eval(x: f32) -> f32 {
    let k: f32 = 0.797_884_6; // sqrt(2/pi)
    let inner = k * (x + 0.044715 * x * x * x);
    let e2 = (2.0 * inner).exp();
    0.5 * x * (2.0 - 2.0 / (e2 + 1.0))
}

#[cfg(test)]
/// 1d GELU over a flat array.
///
/// Computes `gelu(x[i])` for all `i`.
///
/// # Errors
///
/// Returns [`KernelError`] if the array is empty or any element is non-finite.
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn gelu_ref(x: &[f32]) -> Result<Vec<f32>, KernelError> {
    if x.is_empty() {
        return Err(KernelError::InvalidDimension {
            name: "total",
            value: 0,
        });
    }
    x.iter().map(|&xi| gelu_scalar(xi)).collect()
}

#[cfg(all(kani, feature = "kani-stubbing"))]
mod kani_proofs {
    //! Kani proof harnesses for GELU kernel correctness.
    //!
    //! Uses `exp_stub` / `exp_det_stub` to work around CBMC's inaccurate
    //! `f32::exp()` model (#239). Same pattern as silu_mul.

    use super::*;
    use crate::kani_stubs::{exp_det_stub, exp_stub};

    /// Prove GELU produces finite output for bounded inputs.
    ///
    /// Domain: x in [-100, 100].
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(3)]
    #[kani::stub(f32::exp, exp_stub)]
    fn gelu_finite_for_bounded_inputs() {
        let x: f32 = kani::any();
        kani::assume(x.is_finite() && x >= -100.0 && x <= 100.0);

        let result = gelu_scalar(x).expect("gelu_scalar must succeed for bounded finite inputs");
        assert!(result.is_finite(), "gelu must produce finite output");
    }

    /// Prove GELU bounds algorithm is structurally correct.
    ///
    /// Domain: x in [-5, 5]. Covers `GELU_ARGMIN` (-0.752) and the
    /// transition zone.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(3)]
    #[kani::stub(f32::exp, exp_det_stub)]
    fn gelu_bounds_sound() {
        let x: f32 = kani::any();
        let x_lo: f32 = kani::any();
        let x_hi: f32 = kani::any();

        kani::assume(x.is_finite() && x_lo.is_finite() && x_hi.is_finite());
        kani::assume(x >= -5.0 && x <= 5.0);
        kani::assume(x_lo >= -5.0 && x_lo <= x && x <= x_hi && x_hi <= 5.0);

        let result = gelu_scalar(x).expect("gelu_scalar must succeed for bounded finite inputs");
        let (lower, upper) = gelu_scalar_bounds(x_lo, x_hi).expect("finite inputs");

        assert!(result >= lower - 1e-5, "gelu output must be >= lower bound");
        assert!(result <= upper + 1e-5, "gelu output must be <= upper bound");
    }
}

#[cfg(kani)]
#[path = "gelu_kani_builder.rs"]
mod kani_builder_proofs;

#[cfg(test)]
#[path = "gelu_tests.rs"]
mod tests;
