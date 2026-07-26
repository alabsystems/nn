// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! gpt-oss-specific KV cache with sliding window eviction awareness.
//!
//! gpt-oss alternates between sliding attention (window=128) and full attention
//! layers. The underlying [`nn_core::layers::kv_cache::KvCache`] handles per-layer
//! concatenation, but does not know about sliding window eviction. This module
//! provides [`GptOssKvCache`] which wraps the core `KvCache` and tracks
//! per-layer metadata for memory reporting and sliding window management.
//!
//! **Design rationale:** Rather than replacing the core KvCache, we compose it.
//! The core KvCache handles the actual tensor concat/append logic. This wrapper
//! adds gpt-oss-specific layer-type tracking and memory estimation.

use nn_core::layers::kv_cache::KvCache;
use nn_core::Result;

use crate::config::{GptOssConfig, LayerType};

/// KV cache for gpt-oss autoregressive generation with layer-type awareness.
///
/// Wraps the core [`KvCache`] and tracks per-layer types for sliding window
/// eviction estimation and memory reporting.
pub struct GptOssKvCache {
    /// Underlying per-layer KV cache from nn-core.
    inner: KvCache,
    /// Per-layer attention type (for sliding window awareness).
    layer_types: Vec<LayerType>,
    /// Sliding window size for `SlidingAttention` layers.
    sliding_window: usize,
}

impl GptOssKvCache {
    /// Create a new KV cache sized for the given gpt-oss config.
    #[must_use]
    pub fn new(cfg: &GptOssConfig) -> Self {
        Self {
            inner: KvCache::new(cfg.num_hidden_layers),
            layer_types: cfg.layer_types.clone(),
            sliding_window: cfg.sliding_window,
        }
    }

    /// Access the underlying core KV cache (immutable).
    #[must_use]
    pub fn inner(&self) -> &KvCache {
        &self.inner
    }

    /// Access the underlying core KV cache (mutable), e.g. for `forward_cached`.
    pub fn inner_mut(&mut self) -> &mut KvCache {
        &mut self.inner
    }

    /// Number of layers in this cache.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    /// Current sequence length (from the underlying cache).
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.inner.seq_len()
    }

    /// Sliding window size for `SlidingAttention` layers.
    #[must_use]
    pub fn sliding_window(&self) -> usize {
        self.sliding_window
    }

    /// Layer type for a given layer index.
    ///
    /// # Errors
    ///
    /// Returns an error if `layer_idx` is out of bounds.
    pub fn layer_type(&self, layer_idx: usize) -> Result<LayerType> {
        self.layer_types.get(layer_idx).copied().ok_or_else(|| {
            crate::GptOssError::InvalidInput {
                reason: format!(
                    "layer_idx {} out of bounds (num_layers={})",
                    layer_idx,
                    self.layer_types.len()
                ),
            }
            .into()
        })
    }

    /// Reset cache (start new sequence).
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Effective context length for a given layer.
    ///
    /// For `SlidingAttention` layers, this is `min(seq_len, sliding_window)`.
    /// For `FullAttention` layers, this is `seq_len`.
    #[must_use]
    pub fn effective_context_len(&self, layer_idx: usize) -> usize {
        let seq = self.seq_len();
        match self.layer_types.get(layer_idx) {
            Some(LayerType::SlidingAttention) => seq.min(self.sliding_window),
            Some(LayerType::FullAttention) | None => seq,
        }
    }

    /// Estimated memory usage across all layers (bytes).
    ///
    /// For each layer, estimates KV cache size based on effective context length,
    /// KV dimension, and F32 storage (4 bytes per element).
    ///
    /// The `kv_dim` parameter is `num_key_value_heads * head_dim`.
    #[must_use]
    pub fn estimated_memory_bytes(&self, kv_dim: usize) -> usize {
        let bpe = 4_usize; // F32
        let seq = self.seq_len();
        self.layer_types
            .iter()
            .map(|lt| {
                let effective_seq = match lt {
                    LayerType::SlidingAttention => seq.min(self.sliding_window),
                    LayerType::FullAttention => seq,
                };
                // K + V: each is [batch=1, kv_heads, seq, head_dim] = [1, kv_dim/head_dim, seq, head_dim]
                // Total elements per layer: 2 * kv_dim * effective_seq
                2 * kv_dim * effective_seq * bpe
            })
            .sum()
    }

    /// Maximum possible memory for this cache configuration (bytes).
    ///
    /// Assumes max_seq_len tokens cached. Useful for memory budget planning.
    #[must_use]
    pub fn max_memory_bytes(&self, kv_dim: usize, max_seq_len: usize) -> usize {
        let bpe = 4_usize;
        self.layer_types
            .iter()
            .map(|lt| {
                let max_effective = match lt {
                    LayerType::SlidingAttention => max_seq_len.min(self.sliding_window),
                    LayerType::FullAttention => max_seq_len,
                };
                2 * kv_dim * max_effective * bpe
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> GptOssConfig {
        GptOssConfig::gptoss_20b()
    }

    #[test]
    fn test_kv_cache_new() {
        let cfg = test_config();
        let cache = GptOssKvCache::new(&cfg);
        assert_eq!(cache.num_layers(), 24);
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.sliding_window(), 128);
    }

    #[test]
    fn test_kv_cache_layer_types() {
        let cfg = test_config();
        let cache = GptOssKvCache::new(&cfg);
        // Even layers are sliding, odd are full
        assert_eq!(cache.layer_type(0).unwrap(), LayerType::SlidingAttention);
        assert_eq!(cache.layer_type(1).unwrap(), LayerType::FullAttention);
        assert_eq!(cache.layer_type(22).unwrap(), LayerType::SlidingAttention);
        assert_eq!(cache.layer_type(23).unwrap(), LayerType::FullAttention);
    }

    #[test]
    fn test_kv_cache_layer_type_out_of_bounds() {
        let cfg = test_config();
        let cache = GptOssKvCache::new(&cfg);
        assert!(cache.layer_type(24).is_err());
    }

    #[test]
    fn test_effective_context_empty() {
        let cfg = test_config();
        let cache = GptOssKvCache::new(&cfg);
        // With seq_len=0, effective is 0 for both types
        assert_eq!(cache.effective_context_len(0), 0);
        assert_eq!(cache.effective_context_len(1), 0);
    }

    #[test]
    fn test_max_memory_sliding_vs_full() {
        let cfg = test_config();
        let cache = GptOssKvCache::new(&cfg);
        let kv_dim = cfg.kv_dim(); // 512

        // With large max_seq_len, sliding layers are capped at window=128
        let max_mem = cache.max_memory_bytes(kv_dim, 4096);
        // 12 sliding layers: each uses min(4096, 128) = 128
        // 12 full layers: each uses 4096
        let bpe = 4_usize;
        let expected_sliding = 12 * 2 * kv_dim * 128 * bpe;
        let expected_full = 12 * 2 * kv_dim * 4096 * bpe;
        assert_eq!(max_mem, expected_sliding + expected_full);
    }

    #[test]
    fn test_estimated_memory_zero_at_start() {
        let cfg = test_config();
        let cache = GptOssKvCache::new(&cfg);
        assert_eq!(cache.estimated_memory_bytes(cfg.kv_dim()), 0);
    }

    #[test]
    fn test_reset_clears_cache() {
        let cfg = test_config();
        let mut cache = GptOssKvCache::new(&cfg);
        cache.reset();
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.num_layers(), 24);
    }
}
