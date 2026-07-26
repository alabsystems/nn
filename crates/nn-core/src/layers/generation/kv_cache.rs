// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! KV cache for transformer autoregressive decoding.
//!
//! Provides [`KvCacheLayer`] (per-attention-layer key/value storage) and
//! [`KvCache`] (full-model cache across all layers). Cache entries are
//! [`DynTensor`]s shaped `[batch, num_kv_heads, seq_len, head_dim]`.
//!
//! # candle-nn bridge
//!
//! candle's per-layer `candle_nn::kv_cache::KvCache` maps to [`KvCacheLayer`]:
//!
//! ```ignore
//! // NOTE: ignore — consumer-side config pattern, not standalone compilable
//! // In dvoice backend.rs:
//! #[cfg(feature = "nn-backend")]
//! pub type KvCache = nn::layers::KvCacheLayer;
//! ```
//!
//! API differences: constructor returns `Result` (validates dim=2), accessors
//! have both candle names (`.k()`, `.v()`, `.current_seq_len()`, `.dim()`) and
//! nn names (`.key()`, `.value()`, `.seq_len()`).
//!
//! # Usage
//!
//! ```ignore
//! // NOTE: ignore — requires undefined model with specific forward signature
//! let mut cache = KvCache::new(num_layers);
//! for step in 0..max_tokens {
//!     let (logits, _) = model.forward(&input, &mut cache)?;
//!     let next_token = sample(&logits);
//! }
//! ```

use crate::dyn_tensor::DynTensor;
use crate::{Result, TensorError};

const SEQ_DIM: usize = 2;

// Trait definitions, KvCacheLayerBackend impl, and validation helpers
// extracted to kv_cache_traits.rs (#1575) to keep files under 500 lines.
#[path = "kv_cache_traits.rs"]
pub(super) mod traits;
use traits::{alloc_buffer, validate_dims_match, validate_kv_pair};
pub use traits::{KvCacheBackend, KvCacheLayerBackend};

/// Initial buffer capacity in sequence positions when no hint is available.
const INITIAL_CAPACITY: usize = 16;

/// Maximum sequence capacity (256K tokens). Guards against OOM from runaway
/// doubling. Sufficient for Qwen3 YaRN (131K), AI Model (200K), and similar
/// long-context models. Increase if a model legitimately needs > 256K context.
const MAX_SEQ_CAPACITY: usize = 262_144;

/// Per-layer KV cache entry for transformer attention.
///
/// Stores key and value in single `DynTensor` buffers shaped
/// `[batch, num_kv_heads, capacity, head_dim]`. Grows along dim=2 (sequence)
/// via a double-when-full strategy, giving O(1) amortized appends.
///
/// Over S single-token decode steps the total copy work is O(S) — compared
/// to O(S²) for the previous chunk-based design.
#[derive(Debug, Clone)]
pub struct KvCacheLayer {
    key_buf: Option<DynTensor>,
    value_buf: Option<DynTensor>,
    current_len: usize,
    capacity: usize,
    /// Weight generation counter for stale-cache detection.
    ///
    /// When weights are edited (via causal tracing / weight surgery), the
    /// consumer increments this counter after calling [`invalidate()`](Self::invalidate).
    /// Downstream code can compare `weight_generation()` against the model's
    /// weight generation to detect stale KV entries.
    weight_gen: u64,
}

impl KvCacheLayer {
    /// Create an empty cache layer (no cached K/V yet).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            key_buf: None,
            value_buf: None,
            current_len: 0,
            capacity: 0,
            weight_gen: 0,
        }
    }

    /// candle-nn compatibility constructor.
    ///
    /// Creates an empty cache layer equivalent to [`KvCacheLayer::empty()`].
    /// `dim` must be 2 (nn always concatenates along the sequence dimension,
    /// which is dim=2 in `[batch, num_kv_heads, seq_len, head_dim]` layout).
    /// `max_seq_len` is accepted for API compatibility but ignored — nn grows
    /// the cache dynamically.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::InvalidShape` if `dim != 2`.
    pub fn new(dim: usize, _max_seq_len: usize) -> Result<Self> {
        if dim != SEQ_DIM {
            return Err(TensorError::ValueOutOfRange {
                description: "KvCacheLayer::new requires dim=2 (sequence dimension)",
            });
        }
        Ok(Self::empty())
    }

    /// Append new key/value tensors and return the full cached K/V.
    ///
    /// `new_key` and `new_value` should have shape `[batch, num_kv_heads, new_seq, head_dim]`.
    /// Returns `(full_key, full_value)` with the filled portion
    /// along dim=2 (sequence dimension). O(1) amortized via doubling buffers.
    ///
    /// # Performance: drop returned views before next `append()`
    ///
    /// The returned `(full_key, full_value)` are zero-copy narrow views that
    /// share the backing `ArcArray` with the internal buffer. If these views
    /// are still alive when `append()` is called again, `ArcArray::into_owned()`
    /// must copy the entire buffer (COW semantics), making each step O(capacity)
    /// instead of O(new_seq) — degrading total cost from O(S) to O(S²) over S
    /// decode steps.
    ///
    /// In typical autoregressive use this is fine: the model's forward pass
    /// computes attention with the returned K/V and drops them before the next
    /// decode step. But if you store the returned views in a `Vec` or other
    /// long-lived container, you will trigger the O(S²) copy path.
    pub fn append(
        &mut self,
        new_key: &DynTensor,
        new_value: &DynTensor,
    ) -> Result<(DynTensor, DynTensor)> {
        validate_kv_pair(new_key, new_value)?;

        let new_seq = new_key.dims().get(SEQ_DIM).copied().ok_or_else(|| {
            TensorError::DimensionOutOfRange {
                dim: SEQ_DIM,
                rank: new_key.rank(),
            }
        })?;
        let needed = self
            .current_len
            .checked_add(new_seq)
            .ok_or(TensorError::InvalidShape(
                "KV cache sequence length overflow".into(),
            ))?;

        if let Some(ref kb) = self.key_buf {
            // Validate non-sequence dims match existing buffer.
            if kb.rank() != new_key.rank() {
                return Err(TensorError::RankMismatch {
                    expected: kb.rank(),
                    actual: new_key.rank(),
                });
            }
            validate_dims_match(kb, new_key, "key")?;
            if let Some(ref vb) = self.value_buf {
                validate_dims_match(vb, new_value, "value")?;
            }
        }

        if self.key_buf.is_none() {
            // First append: allocate with initial capacity.
            let cap = INITIAL_CAPACITY.max(new_seq);
            let key_buf = alloc_buffer(new_key, cap)?;
            let value_buf = alloc_buffer(new_value, cap)?;
            self.key_buf = Some(key_buf.slice_set_into(SEQ_DIM, 0, new_key)?);
            self.value_buf = Some(value_buf.slice_set_into(SEQ_DIM, 0, new_value)?);
            self.capacity = cap;
            self.current_len = new_seq;
        } else if needed > self.capacity {
            // Grow: double capacity until it fits.
            let mut new_cap = self.capacity;
            while new_cap < needed {
                new_cap = new_cap
                    .checked_mul(2)
                    .ok_or(TensorError::DimensionOverflow {
                        dims: vec![new_cap, 2],
                    })?;
            }
            if new_cap > MAX_SEQ_CAPACITY {
                return Err(TensorError::ValueOutOfRange {
                    description: "KV cache would exceed max capacity",
                });
            }
            let old_key = self.key_buf.take().ok_or(TensorError::ValueOutOfRange {
                description: "KV cache key buffer missing during grow",
            })?;
            let old_value = self.value_buf.take().ok_or(TensorError::ValueOutOfRange {
                description: "KV cache value buffer missing during grow",
            })?;
            let mut new_key_buf = alloc_buffer(&old_key, new_cap)?;
            let mut new_value_buf = alloc_buffer(&old_value, new_cap)?;
            // Copy old filled portion.
            let old_filled_k = old_key.narrow(SEQ_DIM, 0, self.current_len)?;
            let old_filled_v = old_value.narrow(SEQ_DIM, 0, self.current_len)?;
            new_key_buf = new_key_buf.slice_set_into(SEQ_DIM, 0, &old_filled_k)?;
            new_value_buf = new_value_buf.slice_set_into(SEQ_DIM, 0, &old_filled_v)?;
            // Copy new data.
            new_key_buf = new_key_buf.slice_set_into(SEQ_DIM, self.current_len, new_key)?;
            new_value_buf = new_value_buf.slice_set_into(SEQ_DIM, self.current_len, new_value)?;
            self.key_buf = Some(new_key_buf);
            self.value_buf = Some(new_value_buf);
            self.capacity = new_cap;
            self.current_len = needed;
        } else {
            // Fast path: write into existing buffer.
            let kb = self.key_buf.take().ok_or_else(|| {
                TensorError::InvalidShape("KV cache key buffer missing during append".into())
            })?;
            let vb = self.value_buf.take().ok_or_else(|| {
                TensorError::InvalidShape("KV cache value buffer missing during append".into())
            })?;
            self.key_buf = Some(kb.slice_set_into(SEQ_DIM, self.current_len, new_key)?);
            self.value_buf = Some(vb.slice_set_into(SEQ_DIM, self.current_len, new_value)?);
            self.current_len = needed;
        }

        // Return filled portion as a narrow view.
        let full_key = self
            .key_buf
            .as_ref()
            .ok_or_else(|| {
                TensorError::InvalidShape("KV cache key buffer missing after append".into())
            })?
            .narrow(SEQ_DIM, 0, self.current_len)?;
        let full_value = self
            .value_buf
            .as_ref()
            .ok_or_else(|| {
                TensorError::InvalidShape("KV cache value buffer missing after append".into())
            })?
            .narrow(SEQ_DIM, 0, self.current_len)?;
        Ok((full_key, full_value))
    }

    /// Current cached sequence length (0 if empty).
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.current_len
    }

    /// candle-nn compatibility alias for [`seq_len()`](Self::seq_len).
    #[must_use]
    pub fn current_seq_len(&self) -> usize {
        self.current_len
    }

    /// The dimension along which the cache concatenates (always `2` — sequence dim).
    ///
    /// candle-nn compatibility: `candle_nn::kv_cache::Cache::dim()`.
    #[must_use]
    pub fn dim(&self) -> usize {
        SEQ_DIM
    }

    /// candle-nn compatibility alias for [`key()`](Self::key).
    pub fn k(&self) -> Result<Option<DynTensor>> {
        self.key()
    }

    /// candle-nn compatibility alias for [`value()`](Self::value).
    pub fn v(&self) -> Result<Option<DynTensor>> {
        self.value()
    }

    /// Whether this layer has no cached entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    /// Full cached key tensor, if any.
    ///
    /// Returns `Ok(None)` if cache is empty. Returns a narrow view of the
    /// filled portion — no concatenation needed.
    pub fn key(&self) -> Result<Option<DynTensor>> {
        if self.current_len == 0 {
            return Ok(None);
        }
        match &self.key_buf {
            None => Ok(None),
            Some(buf) => Ok(Some(buf.narrow(SEQ_DIM, 0, self.current_len)?)),
        }
    }

    /// Full cached value tensor, if any.
    ///
    /// Returns `Ok(None)` if cache is empty. Returns a narrow view of the
    /// filled portion — no concatenation needed.
    pub fn value(&self) -> Result<Option<DynTensor>> {
        if self.current_len == 0 {
            return Ok(None);
        }
        match &self.value_buf {
            None => Ok(None),
            Some(buf) => Ok(Some(buf.narrow(SEQ_DIM, 0, self.current_len)?)),
        }
    }

    /// Reset the cache (clear all stored K/V, release buffers).
    ///
    /// Drops buffer memory entirely. After reset, the next append starts from
    /// `INITIAL_CAPACITY` and must re-grow through the doubling cascade.
    /// Use [`clear()`](Self::clear) to preserve buffer capacity for reuse.
    pub fn reset(&mut self) {
        self.key_buf = None;
        self.value_buf = None;
        self.current_len = 0;
        self.capacity = 0;
    }

    /// Clear cached entries but preserve buffer capacity for reuse.
    ///
    /// Unlike [`reset()`](Self::reset), this keeps the allocated buffers at
    /// their current capacity. The next append reuses existing memory without
    /// re-allocation. In batch inference (processing B inputs of S tokens each),
    /// this reduces total allocation work from O(B × S × log S) to O(S × log S)
    /// — the doubling cascade only happens once.
    pub fn clear(&mut self) {
        self.current_len = 0;
        // key_buf and value_buf retain their allocations.
        // capacity stays as-is — next append writes at offset 0.
    }

    /// Invalidate all cached KV entries due to a weight change.
    ///
    /// Clears cached sequence data and increments the weight generation counter.
    /// After a weight edit (via causal tracing / weight surgery), call this to
    /// ensure the next forward pass re-computes KV from the updated weights.
    ///
    /// Unlike [`reset()`](Self::reset), this preserves buffer allocations for reuse.
    /// Unlike [`clear()`](Self::clear), this also increments the generation counter.
    pub fn invalidate(&mut self) {
        self.current_len = 0;
        self.weight_gen = self.weight_gen.wrapping_add(1);
    }

    /// Current weight generation counter.
    ///
    /// Incremented on each [`invalidate()`](Self::invalidate) call. Consumers
    /// can compare this against a model's weight generation to detect stale
    /// KV cache entries that were computed with outdated weights.
    #[must_use]
    pub fn weight_generation(&self) -> u64 {
        self.weight_gen
    }

    /// Current buffer capacity in sequence positions.
    #[must_use]
    pub fn buffer_capacity(&self) -> usize {
        self.capacity
    }
}

// KvCache (multi-layer model cache) extracted to kv_cache_multi.rs (#1276)
#[path = "kv_cache_multi.rs"]
mod multi;
pub use multi::KvCache;

#[cfg(test)]
#[path = "kv_cache_tests.rs"]
mod tests;
