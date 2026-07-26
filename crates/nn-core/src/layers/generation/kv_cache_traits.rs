// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! KV cache trait definitions, trait impl for [`KvCacheLayer`], and
//! validation helpers.
//!
//! Extracted from `kv_cache.rs` (#1575) to keep files under 500 lines.

use crate::dyn_tensor::DynTensor;
use crate::{Result, TensorError};

use super::{KvCacheLayer, SEQ_DIM};

/// Trait for per-layer KV cache operations.
///
/// [`KvCacheLayer`] implements this with O(1) amortized doubling buffers.
///
/// Models and `generate()` use this trait to work with any cache type.
pub trait KvCacheLayerBackend {
    /// Append new key/value tensors and return the full cached K/V.
    fn append(
        &mut self,
        new_key: &DynTensor,
        new_value: &DynTensor,
    ) -> Result<(DynTensor, DynTensor)>;

    /// Current cached sequence length.
    fn seq_len(&self) -> usize;

    /// Whether this layer has no cached entries.
    fn is_empty(&self) -> bool;

    /// Reset the cache (clear all stored K/V).
    fn reset(&mut self);

    /// Clear cached entries but preserve buffer capacity for reuse.
    ///
    /// Default implementation delegates to [`reset()`](Self::reset).
    /// [`KvCacheLayer`](super::KvCacheLayer) overrides this to retain
    /// allocated buffers, avoiding re-allocation in batch inference.
    fn clear(&mut self) {
        self.reset();
    }
}

/// Trait for multi-layer KV cache operations.
///
/// [`KvCache`] implements this with dynamically-sized per-layer buffers.
pub trait KvCacheBackend {
    /// Get a mutable reference to a cache layer (for append).
    fn layer_backend_mut(&mut self, index: usize) -> Result<&mut dyn KvCacheLayerBackend>;

    /// Number of cache layers.
    fn num_layers(&self) -> usize;

    /// Current cached sequence length (from the first non-empty layer, or 0).
    fn seq_len(&self) -> usize;

    /// Reset all layers.
    fn reset(&mut self);

    /// Clear all layers but preserve buffer capacity for reuse.
    ///
    /// Default implementation delegates to [`reset()`](Self::reset).
    fn clear(&mut self) {
        self.reset();
    }
}

impl KvCacheLayerBackend for KvCacheLayer {
    fn append(
        &mut self,
        new_key: &DynTensor,
        new_value: &DynTensor,
    ) -> Result<(DynTensor, DynTensor)> {
        self.append(new_key, new_value)
    }

    fn seq_len(&self) -> usize {
        self.seq_len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn clear(&mut self) {
        self.clear();
    }
}

/// Allocate a zero-filled buffer matching `template`'s shape but with
/// `capacity` in the sequence dimension (dim=2).
pub(in crate::layers::generation) fn alloc_buffer(
    template: &DynTensor,
    capacity: usize,
) -> Result<DynTensor> {
    let mut dims = template.dims().to_vec();
    dims[SEQ_DIM] = capacity;
    DynTensor::zeros(&dims, template.dtype(), &template.device())
}

/// Validate that key and value tensors have matching shapes.
pub(in crate::layers::generation) fn validate_kv_pair(
    key: &DynTensor,
    value: &DynTensor,
) -> Result<()> {
    if key.rank() < 3 {
        return Err(TensorError::RankMismatch {
            expected: 3,
            actual: key.rank(),
        });
    }
    if key.dims() != value.dims() {
        return Err(TensorError::shape_mismatch(
            key.dims().to_vec(),
            value.dims().to_vec(),
        ));
    }
    Ok(())
}

/// Validate that non-sequence dimensions match between cached and new tensors.
pub(in crate::layers::generation) fn validate_dims_match(
    cached: &DynTensor,
    new: &DynTensor,
    _name: &str,
) -> Result<()> {
    let cached_dims = cached.dims();
    let new_dims = new.dims();
    for (i, (c, n)) in cached_dims.iter().zip(new_dims.iter()).enumerate() {
        if i != SEQ_DIM && c != n {
            return Err(TensorError::shape_mismatch(
                cached_dims.to_vec(),
                new_dims.to_vec(),
            ));
        }
    }
    Ok(())
}
