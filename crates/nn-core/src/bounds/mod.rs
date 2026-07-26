// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Self-contained interval bounds for verification.
//!
//! [`IntervalBounds`] stores per-element `[lower, upper]` bounds as
//! `ArrayD<f32>`. Constructors reject NaN and inverted bounds.
//!
//! IBP arithmetic (add, mul, scale, shift) is provided by
//! `ny_tensor::BoundedTensor` — not duplicated here (#2005).
//!
//! Lossless conversion between `IntervalBounds` and
//! `ny_tensor::BoundedTensor` is provided by `nn-verify::bounds_bridge`.

use crate::{Result, TensorError};
use ndarray::ArrayD;

pub(crate) mod repair;

// Public re-export for cross-crate contract tests (e.g., nn-verify
// verifying nn-core and gamma-core FALLBACK_BOUND stay synchronized).
pub use repair::FALLBACK_BOUND;

// Re-export ULP functions publicly for cross-crate use (#523).
pub use repair::{next_down_f32, next_down_n_f32, next_up_f32, next_up_n_f32};

// Re-export repair_nan_to_fallback as pub(crate) under kani so ulp_round.rs can import it.
#[cfg(kani)]
pub(crate) use repair::repair_nan_to_fallback;

// Re-export repair helpers for test coverage (these are not part of the public API).
#[cfg(test)]
pub(crate) use repair::{enforce_bound_ordering, repair_non_finite_lower, repair_non_finite_upper};

/// Per-element interval bounds for tensors.
///
/// Stores `lower[i] <= upper[i]` for every element `i`. Used to track
/// value ranges through verification (NY integration).
///
/// All constructors validate inputs (NaN/Inf rejection, bound ordering).
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalBounds {
    lower: ArrayD<f32>,
    upper: ArrayD<f32>,
}

impl IntervalBounds {
    /// Create bounds, rejecting NaN, Inf, shape mismatch, and inverted bounds.
    #[must_use = "returns a Result that may contain an error"]
    pub fn new(lower: ArrayD<f32>, upper: ArrayD<f32>) -> Result<Self> {
        if lower.shape() != upper.shape() {
            return Err(TensorError::InvalidBounds(format!(
                "shape mismatch: lower {:?} vs upper {:?}",
                lower.shape(),
                upper.shape()
            )));
        }
        // Reject NaN and Inf (per design doc: check !is_finite explicitly)
        if lower.iter().any(|v| !v.is_finite()) || upper.iter().any(|v| !v.is_finite()) {
            return Err(TensorError::InvalidBounds("NaN or Inf in bounds".into()));
        }
        // Reject inverted bounds
        if ndarray::Zip::from(&lower)
            .and(&upper)
            .fold(false, |acc, &l, &u| acc || l > u)
        {
            return Err(TensorError::InvalidBounds(
                "inverted bounds: lower > upper".into(),
            ));
        }
        Ok(Self { lower, upper })
    }

    /// Create bounds allowing infinite endpoints but rejecting NaN.
    ///
    /// Use for conservative fallback bounds like `[-inf, +inf]`.
    #[must_use = "returns a Result that may contain an error"]
    pub fn new_allow_infinite(lower: ArrayD<f32>, upper: ArrayD<f32>) -> Result<Self> {
        if lower.shape() != upper.shape() {
            return Err(TensorError::InvalidBounds(format!(
                "shape mismatch: lower {:?} vs upper {:?}",
                lower.shape(),
                upper.shape()
            )));
        }
        if lower.iter().any(|v| v.is_nan()) || upper.iter().any(|v| v.is_nan()) {
            return Err(TensorError::InvalidBounds("NaN in bounds".into()));
        }
        if ndarray::Zip::from(&lower)
            .and(&upper)
            .fold(false, |acc, &l, &u| acc || l > u)
        {
            return Err(TensorError::InvalidBounds(
                "inverted bounds: lower > upper".into(),
            ));
        }
        Ok(Self { lower, upper })
    }

    /// Create concrete bounds where lower == upper. Rejects NaN and Inf.
    #[must_use = "returns a Result that may contain an error"]
    pub fn concrete(values: ArrayD<f32>) -> Result<Self> {
        if values.iter().any(|v| !v.is_finite()) {
            return Err(TensorError::InvalidBounds(
                "NaN or Inf in concrete values".into(),
            ));
        }
        Ok(Self {
            lower: values.clone(),
            upper: values,
        })
    }

    /// Create epsilon ball: `[value - epsilon, value + epsilon]`.
    ///
    /// Rejects NaN/Inf in values and negative/non-finite epsilon.
    /// Clamps overflow to `f32::MIN` / `f32::MAX`.
    #[must_use = "returns a Result that may contain an error"]
    pub fn from_epsilon(values: ArrayD<f32>, epsilon: f32) -> Result<Self> {
        if values.iter().any(|v| !v.is_finite()) {
            return Err(TensorError::InvalidBounds(
                "NaN or Inf in center values".into(),
            ));
        }
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(TensorError::InvalidBounds(
                "epsilon must be non-negative and finite".into(),
            ));
        }
        let lower = values.mapv(|v| {
            let r = v - epsilon;
            if r.is_finite() {
                r
            } else {
                f32::MIN
            }
        });
        let upper = values.mapv(|v| {
            let r = v + epsilon;
            if r.is_finite() {
                r
            } else {
                f32::MAX
            }
        });
        Ok(Self { lower, upper })
    }

    /// Lower bounds (read-only).
    #[must_use]
    pub fn lower(&self) -> &ArrayD<f32> {
        &self.lower
    }

    /// Upper bounds (read-only).
    #[must_use]
    pub fn upper(&self) -> &ArrayD<f32> {
        &self.upper
    }

    /// Both bounds at once.
    #[must_use]
    pub fn lower_upper(&self) -> (&ArrayD<f32>, &ArrayD<f32>) {
        (&self.lower, &self.upper)
    }

    /// Shape of the bounds tensor.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        self.lower.shape()
    }

    /// Maximum per-element width across all elements.
    ///
    /// Returns `f32::INFINITY` if any element has a NaN width (e.g., from
    /// `+Inf - (+Inf)` when both endpoints are the same infinity).
    #[must_use]
    pub fn max_width(&self) -> f32 {
        ndarray::Zip::from(&self.lower)
            .and(&self.upper)
            .fold(0.0f32, |acc, &l, &u| {
                let w = u - l;
                if w.is_nan() {
                    f32::INFINITY
                } else {
                    acc.max(w)
                }
            })
    }

    /// Consume and return owned `(lower, upper)` arrays.
    #[must_use]
    pub fn into_parts(self) -> (ArrayD<f32>, ArrayD<f32>) {
        (self.lower, self.upper)
    }

    /// Widen bounds by 1 ULP in each direction for soundness.
    ///
    /// Directed rounding: lower moves down, upper moves up. Guarantees the
    /// interval contains the true real-valued result despite floating-point
    /// rounding.
    ///
    /// NaN endpoints are repaired to fallback bounds (`[-FALLBACK_BOUND,
    /// +FALLBACK_BOUND]`). Infinity endpoints are preserved — they serve as
    /// infeasible sentinels from `mark_infeasible_all()` and must not be
    /// converted to finite fallback values (#171).
    #[must_use]
    pub fn round_for_soundness(&self) -> Self {
        let mut lower = self.lower.mapv(next_down_f32);
        let mut upper = self.upper.mapv(next_up_f32);
        // Repair NaN endpoints that survived ULP widening. We only repair
        // NaN here — not infinities — to preserve infeasible sentinels (#171).
        repair::repair_nan_to_fallback(&mut lower, &mut upper);
        Self { lower, upper }
    }

    /// Widen bounds by `depth` ULPs in each direction for cumulative soundness.
    ///
    /// After propagating through `depth` layers, bounds must be at least
    /// `depth` ULPs wider than the mathematically exact result to account for
    /// per-layer rounding error accumulation.
    ///
    /// `depth == 0` returns a clone. `depth == 1` is equivalent to
    /// [`round_for_soundness`](Self::round_for_soundness).
    ///
    /// Part of #2707 — sound bounds for deep layer chains (e.g., Kokoro's
    /// 58 consecutive InstanceNorm layers).
    #[must_use]
    pub fn round_for_soundness_n(&self, depth: usize) -> Self {
        if depth == 0 {
            return self.clone();
        }
        let mut lower = self.lower.mapv(|v| next_down_n_f32(v, depth));
        let mut upper = self.upper.mapv(|v| next_up_n_f32(v, depth));
        repair::repair_nan_to_fallback(&mut lower, &mut upper);
        Self { lower, upper }
    }

    /// Set all elements to infeasible sentinel `(+inf, -inf)`.
    ///
    /// Used in verification to mark infeasible branches. Call
    /// [`repair_invalid_inplace`](Self::repair_invalid_inplace) to restore
    /// valid bounds.
    pub fn mark_infeasible_all(&mut self) {
        self.lower.mapv_inplace(|_| f32::INFINITY);
        self.upper.mapv_inplace(|_| f32::NEG_INFINITY);
    }

    /// Repair elements where lower is non-finite, upper is non-finite,
    /// or `lower > upper`. Replaces with `[-inf, +inf]`.
    ///
    /// Returns count of repaired elements.
    #[must_use]
    pub fn repair_invalid_inplace(&mut self) -> usize {
        let mut count = 0;
        ndarray::Zip::from(&mut self.lower)
            .and(&mut self.upper)
            .for_each(|l, u| {
                if !l.is_finite() || !u.is_finite() || *l > *u {
                    *l = f32::NEG_INFINITY;
                    *u = f32::INFINITY;
                    count += 1;
                }
            });
        count
    }
}

#[cfg(test)]
mod tests;
