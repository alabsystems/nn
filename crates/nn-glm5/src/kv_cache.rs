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

/// Largest layer count [`KVCache::new`] accepts.
///
/// Real GLM-4/5 checkpoints have under a hundred layers; the bound keeps the
/// per-layer buffer table (allocated in `new`) a bounded allocation.
pub const MAX_LAYERS: usize = 1 << 16;

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
    /// Floats per token per layer: `num_heads * head_dim`, checked not to
    /// overflow (nor `max_seq_len * token_size * 4` to exceed `isize::MAX`)
    /// in [`KVCache::new`].
    token_size: usize,
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
    ///
    /// # Errors
    ///
    /// Returns [`Glm5Error::InvalidConfig`] (never panics) if:
    /// - any of the four sizes is zero;
    /// - `num_heads * head_dim` (floats per token) overflows `usize`;
    /// - a full layer, `max_seq_len * num_heads * head_dim` `f32`s, would
    ///   exceed `isize::MAX` bytes (no `Vec` could hold it, so a later
    ///   `append` would panic on capacity overflow);
    /// - `num_layers` exceeds [`MAX_LAYERS`] (the per-layer buffer table is
    ///   allocated here, so its size is bounded up front).
    ///
    /// Decision 2026-10-02 (made by the orchestrator under the owner's
    /// instruction to decide and prefer reversible steps): this constructor
    /// used to be infallible and accepted any sizes, deferring the overflow
    /// to `append` and panicking on an unallocatable `num_layers`.
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        let invalid =
            |reason: String| -> nn_core::TensorError { Glm5Error::InvalidConfig { reason }.into() };
        if num_layers == 0 || num_heads == 0 || head_dim == 0 || max_seq_len == 0 {
            return Err(invalid(format!(
                "KVCache sizes must be non-zero: num_layers={num_layers}, \
                 num_heads={num_heads}, head_dim={head_dim}, max_seq_len={max_seq_len}"
            )));
        }
        let Some(token_size) = num_heads.checked_mul(head_dim) else {
            return Err(invalid(format!(
                "KVCache token size overflows: num_heads={num_heads} * head_dim={head_dim}"
            )));
        };
        let max_layer_bytes = usize::MAX >> 1; // isize::MAX, the Vec allocation limit
        let layer_bytes = max_seq_len
            .checked_mul(token_size)
            .and_then(|n| n.checked_mul(size_of::<f32>()));
        if layer_bytes.is_none_or(|b| b > max_layer_bytes) {
            return Err(invalid(format!(
                "KVCache layer size overflows: max_seq_len={max_seq_len} * \
                 num_heads={num_heads} * head_dim={head_dim} f32s exceeds isize::MAX bytes"
            )));
        }
        if num_layers > MAX_LAYERS {
            return Err(invalid(format!(
                "KVCache num_layers={num_layers} exceeds the {MAX_LAYERS}-layer limit"
            )));
        }
        let keys = vec![Vec::new(); num_layers];
        let values = vec![Vec::new(); num_layers];
        Ok(Self {
            num_layers,
            num_heads,
            head_dim,
            max_seq_len,
            token_size,
            keys,
            values,
            current_len: 0,
        })
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
        let num_layers = self.num_layers;
        let current_len = self.current_len;
        let token_size = self.token_size;

        if layer >= num_layers {
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

        // Every layer's append is checked, not only layer 0's: `current_len`
        // advances on the last layer's append, so a caller that appended to
        // the last layer alone used to grow the cache past `max_seq_len`.
        // Within one token `current_len` does not change, so for the
        // layer-0-first order this rejects exactly what the layer-0 check did.
        let Some(next_len) = current_len
            .checked_add(1)
            .filter(|&next| next <= self.max_seq_len)
        else {
            return Err(Glm5Error::InvalidInput {
                reason: format!(
                    "KVCache at max sequence length ({}) — cannot append",
                    self.max_seq_len,
                ),
            }
            .into());
        };

        // `keys` and `values` hold exactly `num_layers` buffers (built in
        // `new`), and `layer < num_layers` was checked above.
        let (Some(layer_keys), Some(layer_values)) =
            (self.keys.get_mut(layer), self.values.get_mut(layer))
        else {
            return Err(Glm5Error::InvalidInput {
                reason: format!("KVCache layer index {layer} has no buffer"),
            }
            .into());
        };
        layer_keys.extend_from_slice(key);
        layer_values.extend_from_slice(value);

        // Advance current_len after the last layer finishes its append.
        if num_layers.checked_sub(1) == Some(layer) {
            self.current_len = next_len;
        }

        Ok(())
    }

    /// Get all cached keys for a layer.
    ///
    /// Returns a slice of length `current_len * num_heads * head_dim`.
    ///
    /// # Errors
    ///
    /// Returns `Glm5Error::InvalidInput` if `layer >= num_layers` (this was a
    /// documented index panic before).
    pub fn get_keys(&self, layer: usize) -> Result<&[f32]> {
        layer_buffer(&self.keys, layer, self.num_layers)
    }

    /// Get all cached values for a layer.
    ///
    /// Returns a slice of length `current_len * num_heads * head_dim`.
    ///
    /// # Errors
    ///
    /// Returns `Glm5Error::InvalidInput` if `layer >= num_layers` (this was a
    /// documented index panic before).
    pub fn get_values(&self, layer: usize) -> Result<&[f32]> {
        layer_buffer(&self.values, layer, self.num_layers)
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

/// The cached buffer of `layer`, or the same `InvalidInput` error `append`
/// returns for an out-of-range layer.
fn layer_buffer(buffers: &[Vec<f32>], layer: usize, num_layers: usize) -> Result<&[f32]> {
    match buffers.get(layer) {
        Some(buffer) => Ok(buffer),
        None => Err(Glm5Error::InvalidInput {
            reason: format!("KVCache layer index {layer} out of range (num_layers={num_layers})"),
        }
        .into()),
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
        KVCache::new(NUM_LAYERS, NUM_HEADS, HEAD_DIM, MAX_SEQ).expect("valid KVCache sizes")
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

        let cached_keys = cache.get_keys(0).unwrap();
        assert_eq!(cached_keys.len(), NUM_HEADS * HEAD_DIM);
        assert_eq!(cached_keys, key.as_slice());

        let cached_vals = cache.get_values(0).unwrap();
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
        let all_keys = cache.get_keys(0).unwrap();
        assert_eq!(all_keys.len(), 2 * token_size);
        assert_eq!(&all_keys[..token_size], key1.as_slice());
        assert_eq!(&all_keys[token_size..], key2.as_slice());

        // Values for layer 0 should be val1 ++ val2.
        let all_vals = cache.get_values(0).unwrap();
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
        assert!(cache.get_keys(0).unwrap().is_empty());
        assert!(cache.get_values(0).unwrap().is_empty());
    }

    #[test]
    fn test_max_seq_len_enforcement() {
        let mut cache =
            KVCache::new(NUM_LAYERS, NUM_HEADS, HEAD_DIM, 2).expect("valid KVCache sizes");
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
        assert_eq!(cache.get_keys(0).unwrap(), key2.as_slice());
        assert_eq!(cache.get_values(0).unwrap(), val2.as_slice());
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

    /// Regression: `current_len` advances on the last layer's append, and
    /// `max_seq_len` used to be checked on layer 0 only, so appending to the
    /// last layer alone grew the cache past `max_seq_len`.
    #[test]
    fn test_max_seq_len_enforced_on_every_layer() {
        let mut cache =
            KVCache::new(NUM_LAYERS, NUM_HEADS, HEAD_DIM, 2).expect("valid KVCache sizes");
        let key = make_token_data(1.0);
        let val = make_token_data(2.0);
        let last = NUM_LAYERS - 1;
        cache.append(last, &key, &val).unwrap();
        cache.append(last, &key, &val).unwrap();
        assert_eq!(cache.len(), 2);
        assert!(cache.append(last, &key, &val).is_err());
        assert_eq!(cache.len(), 2);
    }

    fn new_err(num_layers: usize, num_heads: usize, head_dim: usize, max_seq: usize) -> String {
        match KVCache::new(num_layers, num_heads, head_dim, max_seq) {
            Ok(cache) => panic!("expected KVCache::new to fail, got {cache:?}"),
            Err(e) => e.to_string(),
        }
    }

    /// Regression: `num_heads * head_dim` overflowing `usize` is refused by
    /// the constructor with an error (it used to construct and defer the
    /// overflow to `append`).
    #[test]
    fn test_token_size_overflow_rejected() {
        let msg = new_err(1, usize::MAX, 2, 4);
        assert!(msg.contains("token size overflows"), "{msg}");
    }

    /// Regression: a per-layer size past `isize::MAX` bytes (which no `Vec`
    /// can hold, so `append` would eventually panic) is refused up front.
    #[test]
    fn test_layer_size_overflow_rejected() {
        let msg = new_err(1, 1, 1, usize::MAX);
        assert!(msg.contains("layer size overflows"), "{msg}");
        // `max_seq * token_size` fits in usize but `* 4` bytes does not.
        let msg = new_err(1, 1 << 31, 1 << 31, 1);
        assert!(msg.contains("layer size overflows"), "{msg}");
        // Exactly at the limit is accepted (no allocation happens in `new`).
        let max_floats = (usize::MAX >> 1) / size_of::<f32>();
        assert!(KVCache::new(1, 1, 1, max_floats).is_ok());
        assert!(KVCache::new(1, 1, 1, max_floats + 1).is_err());
    }

    /// Regression: an unallocatable layer count is an error, not a
    /// capacity-overflow panic in `vec![Vec::new(); num_layers]`.
    #[test]
    fn test_unallocatable_num_layers_rejected() {
        let msg = new_err(usize::MAX, 1, 1, 1);
        assert!(msg.contains("layer limit"), "{msg}");
        let msg = new_err(MAX_LAYERS + 1, 1, 1, 1);
        assert!(msg.contains("layer limit"), "{msg}");
        let cache = KVCache::new(MAX_LAYERS, 1, 1, 1).expect("at the limit");
        assert_eq!(cache.num_layers(), MAX_LAYERS);
    }

    /// Zero sizes are invalid configurations and are refused.
    #[test]
    fn test_zero_sizes_rejected() {
        for (l, h, d, m) in [(0, 4, 8, 16), (2, 0, 8, 16), (2, 4, 0, 16), (2, 4, 8, 0)] {
            let msg = new_err(l, h, d, m);
            assert!(msg.contains("must be non-zero"), "({l},{h},{d},{m}): {msg}");
        }
    }

    /// Regression: an out-of-range layer is an `InvalidInput` error from
    /// `get_keys` / `get_values` (it was an index-out-of-bounds panic).
    #[test]
    fn test_get_out_of_range_layer_is_error() {
        let mut cache = make_cache();
        cache
            .append(0, &make_token_data(1.0), &make_token_data(2.0))
            .unwrap();
        let err = cache.get_keys(NUM_LAYERS).unwrap_err();
        assert!(err.to_string().contains("out of range"), "{err}");
        assert!(cache.get_values(NUM_LAYERS).is_err());
        assert!(cache.get_keys(usize::MAX).is_err());
        // In-range layers are unaffected.
        assert_eq!(cache.get_keys(0).unwrap().len(), NUM_HEADS * HEAD_DIM);
        assert!(cache.get_values(1).unwrap().is_empty());
    }

    /// Realistic GLM sizes construct exactly as before.
    #[test]
    fn test_realistic_sizes_accepted() {
        let cache = KVCache::new(61, 8, 128, 131_072).expect("GLM-5-scale cache");
        assert_eq!(cache.num_layers(), 61);
        assert_eq!(cache.max_seq_len(), 131_072);
        assert!(cache.is_empty());
    }
}
