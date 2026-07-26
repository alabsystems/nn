// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared utility functions for nn-verify.

#[cfg(feature = "ny")]
use ny_api::BoundedTensor;
#[cfg(feature = "ny")]
use ny_core::{nan_propagating_max, nan_propagating_min};

use crate::error::VerifyError;

/// Return `val` if finite, otherwise `fallback`.
///
/// Used to sanitize f32 values before storing them in `Serialize`-deriving
/// structs — `serde_json` panics on NaN and Infinity.
pub(crate) fn finite_or(val: f32, fallback: f32) -> f32 {
    if val.is_finite() {
        val
    } else {
        fallback
    }
}

/// Replace non-finite values in tensor bounds with `0.0` sentinels
/// (`serde_json` cannot serialize NaN/Infinity).
pub(crate) fn sanitize_tensor_bounds(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&v| finite_or(v, 0.0)).collect()
}

/// Bounds-checked access to a translation value array.
///
/// Analogous to ay's `get_node()` — returns a clean error instead of panicking
/// when the index is out of bounds. The `ctx` string identifies the call site
/// for diagnostics.
///
/// # Errors
///
/// Returns [`VerifyError::InternalTranslationError`] if `idx >= values.len()`.
pub(crate) fn get_value<'a, T>(
    values: &'a [T],
    idx: usize,
    ctx: &str,
) -> Result<&'a T, VerifyError> {
    values
        .get(idx)
        .ok_or_else(|| VerifyError::InternalTranslationError {
            context: format!(
                "{ctx}: node index {idx} out of bounds (len {})",
                values.len()
            ),
        })
}

/// Extract scalar (global min, global max) from a [`BoundedTensor`].
///
/// Folds all elements of the lower bounds with [`nan_propagating_min`] and
/// all elements of the upper bounds with [`nan_propagating_max`]. NaN values
/// propagate through the fold (they are not silently dropped).
#[cfg(feature = "ny")]
pub(crate) fn bounds_min_max(bounds: &BoundedTensor) -> (f32, f32) {
    let (lower, upper) = bounds.lower_upper();
    let scalar_lower = lower
        .iter()
        .copied()
        .fold(f32::INFINITY, nan_propagating_min);
    let scalar_upper = upper
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, nan_propagating_max);
    (scalar_lower, scalar_upper)
}

#[cfg(all(test, feature = "ny"))]
#[path = "util_tests.rs"]
mod tests;
