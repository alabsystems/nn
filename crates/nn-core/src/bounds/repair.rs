// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Non-finite repair and ULP rounding utilities for interval bounds.
//!
//! Extracted from `bounds.rs` to keep that module under 500 lines.
//! These functions sanitize floating-point edge cases (NaN, Inf, inverted
//! bounds) during interval arithmetic.

use ndarray::ArrayD;

/// Conservative fallback bound for NaN/Inf sanitization in bound propagation.
///
/// Matches `ny_core::FALLBACK_BOUND` so arithmetic results are identical
/// to the gamma-tensor implementation. Made `pub` so downstream crates
/// (notably `nn-verify`) can write contract tests verifying synchronization.
pub const FALLBACK_BOUND: f32 = 1.0e10;

/// Repair non-finite lower endpoint to `-FALLBACK_BOUND`.
#[cfg(any(test, kani))]
#[inline]
pub(crate) fn repair_non_finite_lower(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        -FALLBACK_BOUND
    }
}

/// Repair non-finite upper endpoint to `+FALLBACK_BOUND`.
#[cfg(any(test, kani))]
#[inline]
pub(crate) fn repair_non_finite_upper(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        FALLBACK_BOUND
    }
}

/// Fix inverted or non-finite bounds caused by asymmetric repair.
///
/// When one endpoint was very large (finite) and the other was repaired from
/// non-finite to FALLBACK_BOUND, the result can be inverted (lower > upper).
/// Also catches NaN endpoints that survived upstream repair — per IEEE 754,
/// `NaN > x` returns false, which would silently pass NaN through a bare
/// relational comparison. The explicit `!is_finite()` check prevents this.
#[cfg(any(test, kani))]
#[must_use]
pub(crate) fn enforce_bound_ordering(lower: &mut ArrayD<f32>, upper: &mut ArrayD<f32>) -> usize {
    let mut count = 0usize;
    ndarray::Zip::from(lower).and(upper).for_each(|l, u| {
        if !l.is_finite() || !u.is_finite() || *l > *u {
            *l = -FALLBACK_BOUND;
            *u = FALLBACK_BOUND;
            count += 1;
        }
    });
    count
}

/// Repair NaN-only endpoints to fallback bounds.
///
/// Unlike [`enforce_bound_ordering`], this does NOT repair infinity endpoints.
/// This preserves infeasible sentinels `(+inf, -inf)` from `mark_infeasible_all`
/// (#171) while still catching NaN that would silently bypass IEEE 754
/// relational comparisons (#66).
pub(crate) fn repair_nan_to_fallback(lower: &mut ArrayD<f32>, upper: &mut ArrayD<f32>) {
    ndarray::Zip::from(lower).and(upper).for_each(|l, u| {
        if l.is_nan() || u.is_nan() {
            *l = -FALLBACK_BOUND;
            *u = FALLBACK_BOUND;
        }
    });
}

/// Return the next representable f32 below `x` (widen lower by 1 ULP).
///
/// Canonical copy — must match `NY/crates/gamma-tensor/src/rounding.rs::next_down_f32`.
/// nn adds explicit infinity guards per design doc #171 to preserve infeasible sentinels.
pub fn next_down_f32(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    // Guard: preserve +inf sentinel (design doc #171).
    if x == f32::INFINITY {
        return f32::INFINITY;
    }
    if x == 0.0 {
        return f32::from_bits(0x8000_0001); // smallest negative subnormal
    }
    let bits = x.to_bits();
    if x.is_sign_positive() {
        f32::from_bits(bits - 1)
    } else {
        f32::from_bits(bits + 1)
    }
}

/// Return the next representable f32 above `x` (widen upper by 1 ULP).
///
/// Canonical copy — must match `NY/crates/gamma-tensor/src/rounding.rs::next_up_f32`.
/// nn adds explicit infinity guards per design doc #171 to preserve infeasible sentinels.
pub fn next_up_f32(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x == f32::INFINITY {
        return f32::INFINITY;
    }
    // Guard: preserve -inf sentinel (design doc #171).
    if x == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if x == 0.0 {
        return f32::from_bits(1); // smallest positive subnormal
    }
    let bits = x.to_bits();
    if x.is_sign_positive() {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

/// Widen `x` downward by `n` ULPs. Applies `next_down_f32` `n` times.
///
/// For non-finite or NaN inputs, returns the input unchanged (same as
/// `next_down_f32`). Saturates at `-inf` if widening exceeds `f32::MIN`.
///
/// Part of #2707 — cumulative ULP tracking for deep layer chains.
pub fn next_down_n_f32(x: f32, n: usize) -> f32 {
    let mut result = x;
    for _ in 0..n {
        result = next_down_f32(result);
    }
    result
}

/// Widen `x` upward by `n` ULPs. Applies `next_up_f32` `n` times.
///
/// For non-finite or NaN inputs, returns the input unchanged (same as
/// `next_up_f32`). Saturates at `+inf` if widening exceeds `f32::MAX`.
///
/// Part of #2707 — cumulative ULP tracking for deep layer chains.
pub fn next_up_n_f32(x: f32, n: usize) -> f32 {
    let mut result = x;
    for _ in 0..n {
        result = next_up_f32(result);
    }
    result
}
