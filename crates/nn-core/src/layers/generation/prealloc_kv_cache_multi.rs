// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-layer pre-allocated KV cache for full transformer models.
//!
//! Extracted from `prealloc_kv_cache.rs` for 500-line compliance.
//! Contains [`PreallocKvCache`] and its [`KvCacheBackend`] impl.

use crate::{Result, TensorError};

use super::prealloc_kv_cache::PreallocKvCacheLayer;
use super::{KvCacheBackend, KvCacheLayerBackend};

/// Pre-allocated KV cache for a full transformer model.
///
/// Each layer gets its own [`PreallocKvCacheLayer`] with buffers sized for
/// `max_seq_len`. All buffers are allocated lazily on first use (to infer
/// shape, dtype, and device from the input tensors).
///
/// Implements [`KvCacheBackend`] for drop-in compatibility with existing
/// generation code (`generate()`, `beam_search()`).
#[derive(Debug, Clone)]
pub struct PreallocKvCache {
    layers: Vec<PreallocKvCacheLayer>,
    max_seq_len: usize,
}

impl PreallocKvCache {
    /// Create a new pre-allocated KV cache.
    ///
    /// - `num_layers`: number of attention layers in the model
    /// - `max_seq_len`: maximum sequence positions per layer
    ///
    /// # Errors
    ///
    /// Returns an error if `max_seq_len` is zero.
    pub fn new(num_layers: usize, max_seq_len: usize) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(PreallocKvCacheLayer::new(max_seq_len)?);
        }
        Ok(Self {
            layers,
            max_seq_len,
        })
    }

    /// Number of cache layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Maximum sequence length per layer.
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get an immutable reference to a cache layer.
    pub fn layer(&self, index: usize) -> Result<&PreallocKvCacheLayer> {
        self.layers
            .get(index)
            .ok_or(TensorError::DimensionOutOfRange {
                dim: index,
                rank: self.layers.len(),
            })
    }

    /// Get a mutable reference to a cache layer (for append).
    pub fn layer_mut(&mut self, index: usize) -> Result<&mut PreallocKvCacheLayer> {
        let num = self.layers.len();
        self.layers
            .get_mut(index)
            .ok_or(TensorError::DimensionOutOfRange {
                dim: index,
                rank: num,
            })
    }

    /// Current cached sequence length (from the first non-empty layer, or 0).
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.layers
            .iter()
            .find(|l| !l.is_empty())
            .map(PreallocKvCacheLayer::seq_len)
            .unwrap_or(0)
    }

    /// Whether all layers are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.iter().all(PreallocKvCacheLayer::is_empty)
    }

    /// Reset all layers (clear the full cache, release buffers).
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Clear all layers but preserve buffer allocations for reuse.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Remaining capacity in sequence positions (from the first non-empty layer,
    /// or `max_seq_len` if all layers are empty).
    #[must_use]
    pub fn remaining_capacity(&self) -> usize {
        self.layers
            .iter()
            .find(|l| !l.is_empty())
            .map(PreallocKvCacheLayer::remaining_capacity)
            .unwrap_or(self.max_seq_len)
    }
}

impl KvCacheBackend for PreallocKvCache {
    fn layer_backend_mut(&mut self, index: usize) -> Result<&mut dyn KvCacheLayerBackend> {
        let num = self.layers.len();
        self.layers
            .get_mut(index)
            .map(|l| l as &mut dyn KvCacheLayerBackend)
            .ok_or(TensorError::DimensionOutOfRange {
                dim: index,
                rank: num,
            })
    }

    fn num_layers(&self) -> usize {
        self.num_layers()
    }

    fn seq_len(&self) -> usize {
        self.seq_len()
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn clear(&mut self) {
        self.clear();
    }
}
