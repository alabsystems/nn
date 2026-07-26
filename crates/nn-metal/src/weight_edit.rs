// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Atomic weight edit operations for live model weight surgery.
//!
//! Coordinates buffer writes, GPU weight cache replacement, and KV cache
//! invalidation into a single atomic operation. This is the "apply" step
//! in the locate → compute → verify → **apply** → certify pipeline.
//!
//! # Safety Contract
//!
//! Callers must ensure no in-flight GPU command buffers reference the target
//! weight buffer before calling [`apply_weight_edit`]. Use `commit_and_wait()`
//! on all pending command buffers first.
//!
//! # Example
//!
//! ```ignore
//! use nn_metal::weight_edit::{apply_weight_edit, WeightEditSpec};
//! use nn_metal::{MetalBuffer, PipelineCache};
//!
//! let spec = WeightEditSpec {
//!     layer_name: "encoder.linear.weight",
//!     new_data: &edited_weights,
//! };
//! let result = apply_weight_edit(&mut buffer, &spec)?;
//! // result.previous_generation is the generation before the edit
//! // result.new_generation is the generation after
//! ```

use crate::buffer::MetalBuffer;
use crate::error::MetalError;

/// Specification for a single weight edit operation.
#[derive(Debug)]
pub struct WeightEditSpec<'a> {
    /// Human-readable name of the weight being edited (e.g., "encoder.linear.weight").
    pub layer_name: &'a str,
    /// New weight data to write into the buffer. Must be the same length
    /// as (or smaller than) the target buffer's allocation.
    pub new_data: &'a [f32],
}

/// Result of a successful weight edit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightEditResult {
    /// Generation counter before the edit.
    pub previous_generation: u64,
    /// Generation counter after the edit (`previous_generation + 1`).
    pub new_generation: u64,
    /// Number of f32 elements written.
    pub elements_written: usize,
}

/// Errors specific to weight edit operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WeightEditError {
    /// The new weight data is empty.
    #[error("weight edit for '{layer_name}': empty data")]
    EmptyData { layer_name: String },
    /// The new weight data contains NaN or Inf values.
    #[error("weight edit for '{layer_name}': {count} non-finite values in new data")]
    NonFiniteData { layer_name: String, count: usize },
    /// Buffer write failed.
    #[error("weight edit for '{layer_name}': buffer write failed: {source}")]
    BufferWrite {
        layer_name: String,
        #[source]
        source: MetalError,
    },
}

/// Apply a weight edit to a GPU buffer.
///
/// This is the atomic "apply" step in the weight surgery pipeline:
/// 1. Validates the new data (non-empty, finite)
/// 2. Writes the new data into the GPU buffer
///
/// After calling this, the caller should:
/// - Update any `GpuWeightCache` via [`GpuWeightCache::replace`] or
///   [`GpuWeightCache::invalidate`] to bump the generation counter
/// - Invalidate any KV caches via [`KvCacheLayer::invalidate`] to
///   discard activations computed with the old weights
///
/// # Errors
///
/// Returns [`WeightEditError`] if the data is empty, contains non-finite
/// values, or if the buffer write fails.
pub fn apply_weight_edit(
    buffer: &mut MetalBuffer,
    spec: &WeightEditSpec<'_>,
) -> Result<usize, WeightEditError> {
    // Validate: non-empty
    if spec.new_data.is_empty() {
        return Err(WeightEditError::EmptyData {
            layer_name: spec.layer_name.to_string(),
        });
    }

    // Validate: all values finite (defense-in-depth — NaN in weights
    // silently corrupts all downstream inference)
    let non_finite = crate::count_non_finite(spec.new_data);
    if non_finite > 0 {
        return Err(WeightEditError::NonFiniteData {
            layer_name: spec.layer_name.to_string(),
            count: non_finite,
        });
    }

    // Write new data into the GPU buffer
    buffer
        .write_contents(spec.new_data)
        .map_err(|e| WeightEditError::BufferWrite {
            layer_name: spec.layer_name.to_string(),
            source: e,
        })?;

    Ok(spec.new_data.len())
}

/// Apply a weight edit and update the generation counter on a
/// [`GpuWeightCache`](crate::GpuWeightCache).
///
/// Combines [`apply_weight_edit`] with cache generation tracking:
/// 1. Writes new data into the GPU buffer
/// 2. Invalidates the cache (bumps generation, forces re-init on next access)
///
/// Returns a [`WeightEditResult`] with the pre/post generation numbers.
///
/// The caller must still invalidate any KV caches separately via
/// [`KvCacheLayer::invalidate()`](nn_core::layers::KvCacheLayer::invalidate).
#[cfg(test)] // only called from weight_edit_tests.rs
pub(crate) fn apply_weight_edit_with_generation<T>(
    buffer: &mut MetalBuffer,
    spec: &WeightEditSpec<'_>,
    cache: &crate::GpuWeightCache<T>,
) -> Result<WeightEditResult, WeightEditError> {
    let elements_written = apply_weight_edit(buffer, spec)?;

    let previous_generation = cache.generation();
    cache.invalidate();
    // Derive new_generation from previous rather than re-reading the atomic.
    // A second `cache.generation()` load could observe a higher value if
    // another thread invalidated between the two loads.
    let new_generation = previous_generation + 1;

    Ok(WeightEditResult {
        previous_generation,
        new_generation,
        elements_written,
    })
}

#[cfg(test)]
#[path = "weight_edit_tests.rs"]
mod tests;
