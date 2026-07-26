// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Flat KV cache for GLM-4/5 autoregressive generation.
//!
//! [`KVCache`] manages key-value caches for all transformer layers using
//! flat `Vec<f32>` storage. Each layer stores keys and values as contiguous
//! `f32` slices shaped `[current_len * num_heads * head_dim]`.
//!
//! This is a lightweight alternative to [`nn_core::layers::kv_cache::KvCache`]
//! (which uses `DynTensor` buffers) for scenarios that need direct `f32`
//! slice access without tensor overhead.

use crate::Glm5Error;
use nn_core::Result;

/// Flat KV cache for all transformer layers.
///
/// Stores key and value vectors as contiguous `Vec<f32>` per layer.
/// Each token contributes `num_heads * head_dim` floats to both the key
/// and value buffers for each layer.
///
/// # Capacity
///
/// The cache enforces `max_seq_len` as an upper bound. Attempting to
/// append beyond this limit returns an error.
#[derive(Debug, Clone)]
pub struct KVCache {
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    /// Per-layer key storage: `keys[layer]` has length `current_len * num_heads * head_dim`.
    keys: Vec<Vec<f32>>,
    /// Per-layer value storage: same shape as keys.
    values: Vec<Vec<f32>>,
    /// Number of tokens currently cached.
    current_len: usize,
}

impl KVCache {
    /// Create a new empty KV cache.
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers.
    /// * `num_heads` - Number of KV attention heads per layer.
    /// * `head_dim` - Dimension of each attention head.
    /// * `max_seq_len` - Maximum sequence length (append fails beyond this).
    #[must_use]
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        Self {
            num_layers,
            num_heads,
            head_dim,
            max_seq_len,
            keys: vec![Vec::new(); num_layers],
            values: vec![Vec::new(); num_layers],
            current_len: 0,
        }
    }

    /// Append key/value data for one token at the given layer.
    ///
    /// Both `key` and `value` must have exactly `num_heads * head_dim` elements.
    /// The cache must not have reached `max_seq_len`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `layer >= num_layers`
    /// - `key.len() != num_heads * head_dim`
    /// - `value.len() != num_heads * head_dim`
    /// - Appending would exceed `max_seq_len`
    pub fn append(&mut self, layer: usize, key: &[f32], value: &[f32]) -> Result<()> {
        let token_size = self.num_heads * self.head_dim;

        if layer >= self.num_layers {
            return Err(Glm5Error::InvalidInput {
                reason: format!(
                    "KVCache layer index {} out of range (num_layers={})",
                    layer, self.num_layers,
                ),
            }
            .into());
        }

        if key.len() != token_size {
            return Err(Glm5Error::InvalidInput {
                reason: format!(
                    "KVCache key length {} != expected {} (num_heads={} * head_dim={})",
                    key.len(),
                    token_size,
                    self.num_heads,
                    self.head_dim,
                ),
            }
            .into());
        }

        if value.len() != token_size {
            return Err(Glm5Error::InvalidInput {
                reason: format!(
                    "KVCache value length {} != expected {} (num_heads={} * head_dim={})",
                    value.len(),
                    token_size,
                    self.num_heads,
                    self.head_dim,
                ),
            }
            .into());
        }

        // Only check max_seq_len when appending to layer 0 (the first layer
        // call per token). All layers share `current_len`, so the check on
        // layer 0 is sufficient and avoids double-counting.
        if layer == 0 && self.current_len >= self.max_seq_len {
            return Err(Glm5Error::InvalidInput {
                reason: format!(
                    "KVCache at max sequence length ({}) — cannot append",
                    self.max_seq_len,
                ),
            }
            .into());
        }

        self.keys[layer].extend_from_slice(key);
        self.values[layer].extend_from_slice(value);

        // Increment current_len after the last layer finishes its append.
        if layer == self.num_layers - 1 {
            self.current_len += 1;
        }

        Ok(())
    }

    /// Get all cached keys for a layer.
    ///
    /// Returns a slice of length `current_len * num_heads * head_dim`.
    ///
    /// # Panics
    ///
    /// Panics if `layer >= num_layers`.
    #[must_use]
    pub fn get_keys(&self, layer: usize) -> &[f32] {
        &self.keys[layer]
    }

    /// Get all cached values for a layer.
    ///
    /// Returns a slice of length `current_len * num_heads * head_dim`.
    ///
    /// # Panics
    ///
    /// Panics if `layer >= num_layers`.
    #[must_use]
    pub fn get_values(&self, layer: usize) -> &[f32] {
        &self.values[layer]
    }

    /// Reset the cache for a new sequence.
    ///
    /// Clears all key/value data and resets the cached length to zero.
    /// Retains allocated capacity for reuse.
    pub fn clear(&mut self) {
        for k in &mut self.keys {
            k.clear();
        }
        for v in &mut self.values {
            v.clear();
        }
        self.current_len = 0;
    }

    /// Number of tokens currently cached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.current_len
    }

    /// Whether the cache is empty (no tokens cached).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    /// Maximum sequence length this cache supports.
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Number of KV heads per layer.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Head dimension.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NUM_LAYERS: usize = 2;
    const NUM_HEADS: usize = 4;
    const HEAD_DIM: usize = 8;
    const MAX_SEQ: usize = 16;

    fn make_cache() -> KVCache {
        KVCache::new(NUM_LAYERS, NUM_HEADS, HEAD_DIM, MAX_SEQ)
    }

    fn make_token_data(base: f32) -> Vec<f32> {
        (0..NUM_HEADS * HEAD_DIM)
            .map(|i| base + i as f32 * 0.01)
            .collect()
    }

    #[test]
    fn test_append_single_token_and_retrieve() {
        let mut cache = make_cache();
        let key = make_token_data(1.0);
        let val = make_token_data(2.0);

        // Append to both layers for one token.
        cache.append(0, &key, &val).unwrap();
        cache.append(1, &key, &val).unwrap();

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        let cached_keys = cache.get_keys(0);
        assert_eq!(cached_keys.len(), NUM_HEADS * HEAD_DIM);
        assert_eq!(cached_keys, key.as_slice());

        let cached_vals = cache.get_values(0);
        assert_eq!(cached_vals.len(), NUM_HEADS * HEAD_DIM);
        assert_eq!(cached_vals, val.as_slice());
    }

    #[test]
    fn test_append_multiple_tokens_concatenation() {
        let mut cache = make_cache();
        let token_size = NUM_HEADS * HEAD_DIM;

        let key1 = make_token_data(1.0);
        let val1 = make_token_data(10.0);
        let key2 = make_token_data(2.0);
        let val2 = make_token_data(20.0);

        // Token 1: append to all layers.
        for layer in 0..NUM_LAYERS {
            cache.append(layer, &key1, &val1).unwrap();
        }
        assert_eq!(cache.len(), 1);

        // Token 2: append to all layers.
        for layer in 0..NUM_LAYERS {
            cache.append(layer, &key2, &val2).unwrap();
        }
        assert_eq!(cache.len(), 2);

        // Keys for layer 0 should be key1 ++ key2.
        let all_keys = cache.get_keys(0);
        assert_eq!(all_keys.len(), 2 * token_size);
        assert_eq!(&all_keys[..token_size], key1.as_slice());
        assert_eq!(&all_keys[token_size..], key2.as_slice());

        // Values for layer 0 should be val1 ++ val2.
        let all_vals = cache.get_values(0);
        assert_eq!(all_vals.len(), 2 * token_size);
        assert_eq!(&all_vals[..token_size], val1.as_slice());
        assert_eq!(&all_vals[token_size..], val2.as_slice());
    }

    #[test]
    fn test_clear_resets_length() {
        let mut cache = make_cache();
        let key = make_token_data(1.0);
        let val = make_token_data(2.0);

        for layer in 0..NUM_LAYERS {
            cache.append(layer, &key, &val).unwrap();
        }
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert!(cache.get_keys(0).is_empty());
        assert!(cache.get_values(0).is_empty());
    }

    #[test]
    fn test_max_seq_len_enforcement() {
        let mut cache = KVCache::new(NUM_LAYERS, NUM_HEADS, HEAD_DIM, 2);
        let key = make_token_data(1.0);
        let val = make_token_data(2.0);

        // Fill to capacity (2 tokens).
        for _token in 0..2 {
            for layer in 0..NUM_LAYERS {
                cache.append(layer, &key, &val).unwrap();
            }
        }
        assert_eq!(cache.len(), 2);

        // Third token should fail on layer 0.
        let result = cache.append(0, &key, &val);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("max sequence length"),
            "expected max-seq error, got: {err_msg}"
        );
    }

    #[test]
    fn test_wrong_key_length_rejected() {
        let mut cache = make_cache();
        let short_key: Vec<f32> = vec![1.0; 3]; // too short
        let val = make_token_data(2.0);

        let result = cache.append(0, &short_key, &val);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("key length"),
            "expected key-length error, got: {err_msg}"
        );
    }

    #[test]
    fn test_wrong_value_length_rejected() {
        let mut cache = make_cache();
        let key = make_token_data(1.0);
        let short_val: Vec<f32> = vec![1.0; 5]; // too short

        let result = cache.append(0, &key, &short_val);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("value length"),
            "expected value-length error, got: {err_msg}"
        );
    }

    #[test]
    fn test_layer_out_of_range_rejected() {
        let mut cache = make_cache();
        let key = make_token_data(1.0);
        let val = make_token_data(2.0);

        let result = cache.append(NUM_LAYERS, &key, &val);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("out of range"),
            "expected layer-range error, got: {err_msg}"
        );
    }

    #[test]
    fn test_clear_then_reuse() {
        let mut cache = make_cache();
        let key = make_token_data(1.0);
        let val = make_token_data(2.0);

        for layer in 0..NUM_LAYERS {
            cache.append(layer, &key, &val).unwrap();
        }
        cache.clear();

        // Re-append after clear.
        let key2 = make_token_data(5.0);
        let val2 = make_token_data(6.0);
        for layer in 0..NUM_LAYERS {
            cache.append(layer, &key2, &val2).unwrap();
        }
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get_keys(0), key2.as_slice());
        assert_eq!(cache.get_values(0), val2.as_slice());
    }

    #[test]
    fn test_accessors() {
        let cache = make_cache();
        assert_eq!(cache.num_layers(), NUM_LAYERS);
        assert_eq!(cache.num_heads(), NUM_HEADS);
        assert_eq!(cache.head_dim(), HEAD_DIM);
        assert_eq!(cache.max_seq_len(), MAX_SEQ);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }
}
