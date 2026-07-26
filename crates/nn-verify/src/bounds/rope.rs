// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Analytical output bounds for RoPE kernels (rope_cos, rope_sin).
//!
//! Extracted from `ay/prove_bounds_rope.rs` (#859) to be always-available
//! without the `ay-smt` feature flag. Pure Rust math — no ay-bindings dependency.

use crate::error::VerifyError;
use crate::smt_error::SmtError;

/// Compute analytical output bounds for a RoPE kernel (rope_cos or rope_sin).
///
/// **#448 convention:** param 0 (x0) is the symbolic variable bounded by
/// `[x0_lower, x0_upper]`, params 1-2 (x1, freq) are constants.
///
/// Since both `rope_cos = x0 * cos(freq) - x1 * sin(freq)` and
/// `rope_sin = x0 * sin(freq) + x1 * cos(freq)` are linear in x0,
/// we evaluate the `bounds_fn` with x0 as the variable interval and
/// x1/freq as point intervals.
pub(crate) fn rope_output_bounds(
    x1_const: f32,
    freq_const: f32,
    x0_lower: f32,
    x0_upper: f32,
    bounds_fn: fn(
        f32,
        f32,
        f32,
        f32,
        f32,
        f32,
    ) -> Result<(f32, f32), nn_dsl::kernel_error::KernelError>,
) -> Result<(f64, f64), VerifyError> {
    if !x1_const.is_finite() {
        return Err(SmtError::NonFiniteConstantParam {
            index: 1,
            value: f64::from(x1_const),
        }
        .into());
    }
    if !freq_const.is_finite() {
        return Err(SmtError::NonFiniteConstantParam {
            index: 2,
            value: f64::from(freq_const),
        }
        .into());
    }
    if !x0_lower.is_finite() || !x0_upper.is_finite() {
        return Err(SmtError::NonFiniteBound {
            lower: f64::from(x0_lower),
            upper: f64::from(x0_upper),
        }
        .into());
    }
    if x0_lower > x0_upper {
        return Err(SmtError::InvertedBounds {
            lower: f64::from(x0_lower),
            upper: f64::from(x0_upper),
        }
        .into());
    }

    // bounds_fn signature: (x0_lo, x0_hi, x1_lo, x1_hi, freq_lo, freq_hi)
    // x0 is the variable interval; x1 and freq are point intervals.
    let (lo, hi) = bounds_fn(
        x0_lower, x0_upper, x1_const, x1_const, freq_const, freq_const,
    )
    .map_err(|e| SmtError::SolverError {
        reason: format!("rope bounds: {e}"),
    })?;

    let (out_lo, out_hi) = (f64::from(lo), f64::from(hi));
    if !out_lo.is_finite() || !out_hi.is_finite() {
        return Err(SmtError::NonFiniteBound {
            lower: out_lo,
            upper: out_hi,
        }
        .into());
    }

    Ok((out_lo, out_hi))
}

#[cfg(test)]
#[path = "rope_tests.rs"]
mod tests;
