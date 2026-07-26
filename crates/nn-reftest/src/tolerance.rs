// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Configurable tolerance strategies for tensor comparison.
//!
//! Different ML operations require different comparison strategies:
//! - **Absolute:** suitable for values near zero (biases, residuals)
//! - **Relative:** suitable for large values where absolute error is misleading
//! - **Mixed:** NumPy-style `|a-b| <= atol + rtol*|b|` combining both
//! - **ULP:** bit-level floating-point comparison for near-exact results
//! - **PercentClose:** statistical tolerance allowing a fraction of outliers
//!
//! # Example
//!
//! ```rust
//! use nn_reftest::tolerance::{ToleranceStrategy, compare_with_tolerance};
//!
//! let actual = [1.0f32, 2.0, 3.0];
//! let expected = [1.0001, 2.0001, 3.0001];
//!
//! let result = compare_with_tolerance(
//!     &actual,
//!     &expected,
//!     &ToleranceStrategy::Absolute { atol: 1e-3 },
//! ).expect("comparison should succeed");
//!
//! assert!(result.passed);
//! ```

use crate::error::ReftestError;

/// Tolerance strategy for element-wise tensor comparison.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ToleranceStrategy {
    /// Maximum absolute difference: `|a - b| <= atol`.
    Absolute { atol: f64 },

    /// Maximum relative difference: `|a - b| / max(|a|, |b|, eps) <= rtol`.
    ///
    /// A small epsilon (1e-8) prevents division-by-zero for near-zero values.
    Relative { rtol: f64 },

    /// NumPy-style combined tolerance: `|a - b| <= atol + rtol * |b|`.
    ///
    /// An element passes if its absolute difference is within `atol` of zero
    /// or within `rtol` of the expected value, whichever is more generous.
    Mixed { atol: f64, rtol: f64 },

    /// Units in Last Place: two floats are close if their ULP distance
    /// is at most `max_ulps`.
    ///
    /// ULP comparison handles the density of floating-point numbers near zero
    /// and near large values uniformly. NaN never matches; opposite signs
    /// only match through zero (ULP distance through zero is well-defined).
    ULP { max_ulps: u32 },

    /// Statistical tolerance: at least `percent`% of elements must satisfy
    /// `|a - b| <= threshold`.
    ///
    /// Useful for operations where a few outliers are expected (e.g.,
    /// GPU non-determinism in reductions).
    PercentClose { threshold: f64, percent: f64 },
}

/// Result of a tolerance-aware tensor comparison.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ComparisonResult {
    /// Whether the comparison passed under the given strategy.
    pub passed: bool,
    /// Maximum absolute difference across all element pairs.
    pub max_diff: f64,
    /// Mean absolute difference across all element pairs.
    pub mean_diff: f64,
    /// Number of element pairs that exceeded the tolerance.
    pub num_mismatches: usize,
    /// Index of the element pair with the largest absolute difference.
    pub worst_index: usize,
}

/// Compare two f32 slices using the given tolerance strategy.
///
/// Returns a [`ComparisonResult`] with detailed comparison metrics.
///
/// # Errors
///
/// Returns [`ReftestError::DataLengthMismatch`] if the slices have different
/// lengths, or [`ReftestError::EmptyTensor`] if both slices are empty.
pub fn compare_with_tolerance(
    actual: &[f32],
    expected: &[f32],
    strategy: &ToleranceStrategy,
) -> Result<ComparisonResult, ReftestError> {
    if actual.len() != expected.len() {
        return Err(ReftestError::DataLengthMismatch {
            expected: expected.len(),
            actual: actual.len(),
        });
    }
    let n = actual.len();
    if n == 0 {
        return Err(ReftestError::EmptyTensor("(raw slice)".to_string()));
    }

    // First pass: compute aggregate metrics (max_diff, mean_diff, worst_index).
    let mut max_diff: f64 = 0.0;
    let mut sum_diff: f64 = 0.0;
    let mut worst_index: usize = 0;

    for i in 0..n {
        let a = f64::from(actual[i]);
        let b = f64::from(expected[i]);

        // IEEE 754: treat NaN/Inf as infinite divergence.
        let diff = if !a.is_finite() || !b.is_finite() {
            f64::INFINITY
        } else {
            (a - b).abs()
        };

        sum_diff += diff;
        if diff > max_diff || (diff.is_nan() && !max_diff.is_nan()) {
            max_diff = diff;
            worst_index = i;
        }
    }

    let mean_diff = sum_diff / n as f64;

    // Second pass: count mismatches under the chosen strategy.
    let num_mismatches = count_mismatches(actual, expected, strategy);
    let passed = match strategy {
        ToleranceStrategy::PercentClose { percent, .. } => {
            let close_fraction = (n - num_mismatches) as f64 / n as f64 * 100.0;
            close_fraction >= *percent
        }
        _ => num_mismatches == 0,
    };

    Ok(ComparisonResult {
        passed,
        max_diff,
        mean_diff,
        num_mismatches,
        worst_index,
    })
}

/// Count elements that violate the tolerance strategy.
fn count_mismatches(actual: &[f32], expected: &[f32], strategy: &ToleranceStrategy) -> usize {
    let mut mismatches = 0;
    for i in 0..actual.len() {
        if !element_passes(actual[i], expected[i], strategy) {
            mismatches += 1;
        }
    }
    mismatches
}

/// Check whether a single element pair satisfies the tolerance strategy.
fn element_passes(a: f32, b: f32, strategy: &ToleranceStrategy) -> bool {
    // Non-finite values never pass (IEEE 754 defense-in-depth).
    if !a.is_finite() || !b.is_finite() {
        return false;
    }

    match strategy {
        ToleranceStrategy::Absolute { atol } => {
            let diff = (f64::from(a) - f64::from(b)).abs();
            diff <= *atol
        }
        ToleranceStrategy::Relative { rtol } => {
            let a64 = f64::from(a);
            let b64 = f64::from(b);
            let diff = (a64 - b64).abs();
            let denom = a64.abs().max(b64.abs()).max(1e-8);
            (diff / denom) <= *rtol
        }
        ToleranceStrategy::Mixed { atol, rtol } => {
            let a64 = f64::from(a);
            let b64 = f64::from(b);
            let diff = (a64 - b64).abs();
            // NumPy semantics: |a - b| <= atol + rtol * |b|
            diff <= *atol + *rtol * b64.abs()
        }
        ToleranceStrategy::ULP { max_ulps } => ulp_distance(a, b) <= *max_ulps,
        ToleranceStrategy::PercentClose { threshold, .. } => {
            let diff = (f64::from(a) - f64::from(b)).abs();
            diff <= *threshold
        }
    }
}

/// Compute the ULP (Units in Last Place) distance between two f32 values.
///
/// Returns `u32::MAX` if either value is NaN. For values with opposite signs,
/// the distance passes through zero (the standard IEEE 754 ULP metric).
fn ulp_distance(a: f32, b: f32) -> u32 {
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }

    // Convert to signed-magnitude integer representation.
    // IEEE 754 floats in sign-magnitude map to a linear integer space
    // when negative values are reflected: if bits < 0, flip = 0x8000_0000 - bits.
    let a_bits = a.to_bits() as i32;
    let b_bits = b.to_bits() as i32;

    let a_adj = if a_bits < 0 {
        i32::MIN - a_bits
    } else {
        a_bits
    };
    let b_adj = if b_bits < 0 {
        i32::MIN - b_bits
    } else {
        b_bits
    };

    let dist = (i64::from(a_adj) - i64::from(b_adj)).unsigned_abs();
    if dist > u64::from(u32::MAX) {
        u32::MAX
    } else {
        dist as u32
    }
}

#[cfg(test)]
#[path = "tolerance_tests.rs"]
mod tests;
