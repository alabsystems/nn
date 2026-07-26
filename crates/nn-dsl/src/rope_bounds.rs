// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Interval bounds for RoPE scalar kernels.
//!
//! Computes sound output bounds for `rope_cos` and `rope_sin` using interval
//! arithmetic on the cos/sin ranges over the frequency interval.

use nn_core::{next_down_f32, next_up_f32};

use crate::kernel_error::KernelError;
use crate::kernel_util::{validate_bounds_output, validate_bounds_pairs};

/// Compute the range of `cos(x)` for `x ∈ [lo, hi]`.
///
/// Uses the periodicity of cosine: if the interval spans ≥ 2π, returns [-1, 1].
/// Otherwise, checks if any integer multiple of π falls within [lo, hi] to
/// determine whether cos achieves its extrema (-1 or +1).
///
/// Peak detection uses f64 intermediate arithmetic to avoid f32 rounding errors
/// that could cause missed peaks (producing unsound too-tight bounds).
fn cos_range(lo: f32, hi: f32) -> (f32, f32) {
    // Promote to f64 for peak detection to avoid f32 rounding in period checks.
    let lo_f64 = f64::from(lo);
    let hi_f64 = f64::from(hi);

    if hi_f64 - lo_f64 >= std::f64::consts::TAU {
        return (-1.0, 1.0);
    }
    let c_lo = lo.cos();
    let c_hi = hi.cos();
    let mut min = c_lo.min(c_hi);
    let mut max = c_lo.max(c_hi);

    // Check if any multiple of 2π (cos = 1) falls in [lo, hi].
    // Use f64 to prevent rounding from missing a peak at the interval boundary.
    let k_start = (lo_f64 / std::f64::consts::TAU).ceil();
    if k_start * std::f64::consts::TAU <= hi_f64 {
        max = 1.0;
    }

    // Check if any odd multiple of π (cos = -1) falls in [lo, hi].
    let k_start = (lo_f64 / std::f64::consts::PI).ceil();
    // Guard against i64 overflow for very large frequencies. If |k_start| exceeds
    // i64 range, the interval is so wide in terms of periods that we conservatively
    // assume the extremum is reached.
    let k_i64 = if k_start.abs() >= i64::MAX as f64 {
        min = -1.0;
        0i64 // value unused; min already set
    } else {
        k_start as i64
    };
    if min > -1.0 {
        let k = if k_i64 % 2 == 0 {
            k_start + 1.0
        } else {
            k_start
        };
        if k * std::f64::consts::PI <= hi_f64 {
            min = -1.0;
        }
    }

    (min, max)
}

/// Compute the range of `sin(x)` for `x ∈ [lo, hi]`.
///
/// Uses `sin(x) = cos(x - π/2)`. The π/2 subtraction is performed in f64 to
/// avoid f32 rounding that could shift the adjusted interval past a cos
/// peak/trough, causing `cos_range` to miss a sin extremum (#429).
///
/// The final f64→f32 cast uses round-to-nearest-even. For soundness, `adj_lo`
/// is widened downward by 1 ULP and `adj_hi` upward by 1 ULP to guarantee the
/// f32 interval contains the true mathematical interval (#518).
fn sin_range(lo: f32, hi: f32) -> (f32, f32) {
    let adj_lo = next_down_f32((f64::from(lo) - std::f64::consts::FRAC_PI_2) as f32);
    let adj_hi = next_up_f32((f64::from(hi) - std::f64::consts::FRAC_PI_2) as f32);
    cos_range(adj_lo, adj_hi)
}

/// Multiply two intervals: `[a_lo, a_hi] × [b_lo, b_hi]`.
///
/// The result is `[min(corners), max(corners)]` where corners are all four
/// products of endpoints. Uses NaN-propagating min/max so that non-finite
/// products (from 0.0 * Inf or overflow) are not silently dropped.
fn interval_mul(a_lo: f32, a_hi: f32, b_lo: f32, b_hi: f32) -> (f32, f32) {
    let p1 = a_lo * b_lo;
    let p2 = a_lo * b_hi;
    let p3 = a_hi * b_lo;
    let p4 = a_hi * b_hi;
    // Use NaN-propagating min/max: if any product is NaN or Inf,
    // it propagates through so validate_bounds_output catches it.
    let lo = nan_propagating_min_f32(
        nan_propagating_min_f32(p1, p2),
        nan_propagating_min_f32(p3, p4),
    );
    let hi = nan_propagating_max_f32(
        nan_propagating_max_f32(p1, p2),
        nan_propagating_max_f32(p3, p4),
    );
    (lo, hi)
}

/// NaN-propagating minimum: returns NaN if either argument is NaN.
///
/// Unlike `f32::min` which returns the non-NaN argument, this ensures
/// degenerate products (e.g., 0.0 * Inf = NaN) are not silently dropped.
fn nan_propagating_min_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a <= b {
        a
    } else {
        b
    }
}

/// NaN-propagating maximum: returns NaN if either argument is NaN.
fn nan_propagating_max_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a >= b {
        a
    } else {
        b
    }
}

/// Compute sound output bounds for `rope_cos(x0, x1, freq) = x0*cos(freq) - x1*sin(freq)`.
///
/// Given intervals `x0 ∈ [x0_lo, x0_hi]`, `x1 ∈ [x1_lo, x1_hi]`,
/// `freq ∈ [freq_lo, freq_hi]`, returns `(lower, upper)` such that the output
/// is guaranteed to lie within `[lower, upper]` for all inputs in the given ranges.
///
/// Uses interval arithmetic on the cos/sin ranges to produce tight bounds.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteBound`] if any input is NaN/Inf or the
/// computed output bounds overflow to infinity.
/// Returns [`KernelError::InvertedBounds`] if any `lo > hi`.
#[must_use = "returns a Result that may contain an error"]
pub fn rope_cos_scalar_bounds(
    x0_lo: f32,
    x0_hi: f32,
    x1_lo: f32,
    x1_hi: f32,
    freq_lo: f32,
    freq_hi: f32,
) -> Result<(f32, f32), KernelError> {
    validate_bounds_pairs(&[(x0_lo, x0_hi), (x1_lo, x1_hi), (freq_lo, freq_hi)])?;

    let (cos_lo, cos_hi) = cos_range(freq_lo, freq_hi);
    let (sin_lo, sin_hi) = sin_range(freq_lo, freq_hi);

    // rope_cos = x0 * cos(freq) - x1 * sin(freq)
    //          = x0 * cos(freq) + x1 * (-sin(freq))
    let (term1_lo, term1_hi) = interval_mul(x0_lo, x0_hi, cos_lo, cos_hi);
    let (term2_lo, term2_hi) = interval_mul(x1_lo, x1_hi, -sin_hi, -sin_lo);

    let lower = term1_lo + term2_lo;
    let upper = term1_hi + term2_hi;

    validate_bounds_output(lower, upper)
}

/// Compute sound output bounds for `rope_sin(x0, x1, freq) = x0*sin(freq) + x1*cos(freq)`.
///
/// Same structure as [`rope_cos_scalar_bounds`] but for the sine component.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteBound`] if any input is NaN/Inf or the
/// computed output bounds overflow to infinity.
/// Returns [`KernelError::InvertedBounds`] if any `lo > hi`.
#[must_use = "returns a Result that may contain an error"]
pub fn rope_sin_scalar_bounds(
    x0_lo: f32,
    x0_hi: f32,
    x1_lo: f32,
    x1_hi: f32,
    freq_lo: f32,
    freq_hi: f32,
) -> Result<(f32, f32), KernelError> {
    validate_bounds_pairs(&[(x0_lo, x0_hi), (x1_lo, x1_hi), (freq_lo, freq_hi)])?;

    let (cos_lo, cos_hi) = cos_range(freq_lo, freq_hi);
    let (sin_lo, sin_hi) = sin_range(freq_lo, freq_hi);

    // rope_sin = x0 * sin(freq) + x1 * cos(freq)
    let (term1_lo, term1_hi) = interval_mul(x0_lo, x0_hi, sin_lo, sin_hi);
    let (term2_lo, term2_hi) = interval_mul(x1_lo, x1_hi, cos_lo, cos_hi);

    let lower = term1_lo + term2_lo;
    let upper = term1_hi + term2_hi;

    validate_bounds_output(lower, upper)
}

#[cfg(test)]
#[path = "rope_bounds_tests.rs"]
mod tests;
