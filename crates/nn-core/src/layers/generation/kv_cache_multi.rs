// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-layer KV cache for full transformer models.
//!
//! Extracted from `kv_cache.rs` (#1276) for 500-line compliance.
//! Contains [`KvCache`] (model-level cache) and its [`KvCacheBackend`] impl.

use crate::{Result, TensorError};

use super::{KvCacheLayer, KvCacheLayerBackend};

/// KV cache for a full transformer model with multiple attention layers.
///
/// Each layer gets its own [`KvCacheLayer`] indexed by layer number.
#[derive(Debug, Clone)]
pub struct KvCache {
    layers: Vec<KvCacheLayer>,
}

impl KvCache {
    /// Create a new KV cache for a model with `num_layers` attention layers.
    #[must_use]
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| KvCacheLayer::empty()).collect(),
        }
    }

    /// Number of cache layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get an immutable reference to a cache layer.
    pub fn layer(&self, index: usize) -> Result<&KvCacheLayer> {
        self.layers
            .get(index)
            .ok_or(TensorError::DimensionOutOfRange {
                dim: index,
                rank: self.layers.len(),
            })
    }

    /// Get a mutable reference to a cache layer (for append).
    pub fn layer_mut(&mut self, index: usize) -> Result<&mut KvCacheLayer> {
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
            .map(KvCacheLayer::seq_len)
            .unwrap_or(0)
    }

    /// Whether all layers are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.iter().all(KvCacheLayer::is_empty)
    }

    /// Reset all layers (clear the full cache, release buffers).
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Clear all layers but preserve buffer capacity for reuse.
    ///
    /// In batch inference (B inputs of S tokens each), using `clear()` between
    /// inputs avoids re-growing from `INITIAL_CAPACITY` each time. The doubling
    /// cascade only occurs once across all B batches.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }
}

impl super::KvCacheBackend for KvCache {
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

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove `KvCache::new` creates exactly the requested number of layers.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_kv_cache_new_layer_count() {
        let num_layers: usize = kani::any();
        kani::assume(num_layers <= 32);
        let cache = KvCache::new(num_layers);
        assert_eq!(
            cache.num_layers(),
            num_layers,
            "num_layers must match constructor arg"
        );
    }

    /// Prove `KvCache::layer` returns Ok for valid indices and Err for out-of-bounds.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_kv_cache_layer_indexing_bounds() {
        let num_layers: usize = kani::any();
        kani::assume(num_layers >= 1 && num_layers <= 16);
        let cache = KvCache::new(num_layers);
        let index: usize = kani::any();
        kani::assume(index <= num_layers + 1);
        let result = cache.layer(index);
        if index < num_layers {
            assert!(result.is_ok(), "valid index must return Ok");
        } else {
            assert!(result.is_err(), "out-of-bounds index must return Err");
        }
    }

    /// Prove `KvCache::layer_mut` returns Ok for valid indices and Err for OOB.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_kv_cache_layer_mut_indexing_bounds() {
        let num_layers: usize = kani::any();
        kani::assume(num_layers >= 1 && num_layers <= 16);
        let mut cache = KvCache::new(num_layers);
        let index: usize = kani::any();
        kani::assume(index <= num_layers + 1);
        let result = cache.layer_mut(index);
        if index < num_layers {
            assert!(result.is_ok(), "valid index must return Ok");
        } else {
            assert!(result.is_err(), "out-of-bounds index must return Err");
        }
    }

    /// Prove `KvCache::is_empty` is true for a freshly constructed cache.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_kv_cache_new_is_empty() {
        let num_layers: usize = kani::any();
        kani::assume(num_layers <= 16);
        let cache = KvCache::new(num_layers);
        assert!(cache.is_empty(), "new cache must be empty");
        assert_eq!(cache.seq_len(), 0, "new cache must have seq_len 0");
    }

    /// Prove `KvCache::reset` makes all layers empty.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_kv_cache_reset_empties() {
        let num_layers: usize = kani::any();
        kani::assume(num_layers >= 1 && num_layers <= 8);
        let mut cache = KvCache::new(num_layers);
        cache.reset();
        assert!(cache.is_empty(), "reset cache must be empty");
        assert_eq!(cache.seq_len(), 0, "reset cache must have seq_len 0");
        assert_eq!(
            cache.num_layers(),
            num_layers,
            "reset must preserve layer count"
        );
    }
}
