// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pre-allocated KV cache for GPU-resident compiled decoder inference.
//!
//! [`PreallocKvCacheLayer`] pre-allocates key and value buffers for `max_seq_len`
//! positions at construction time. Unlike [`KvCacheLayer`] (which doubles on
//! overflow), this avoids mid-inference GPU reallocation — critical for compiled
//! model execution where buffer addresses are baked into dispatch plans.
//!
//! # GPU residency
//!
//! Buffers are allocated on whatever device the first `append()` tensor lives on.
//! When tensors are on Metal GPU, all `slice_set_into` writes stay on-device.
//! The fixed-capacity design avoids the O(S^2) full-buffer-copy problem that
//! [`KvCacheLayer`]'s doubling strategy has on GPU (see `kv_cache_tests_perf.rs`).
//!
//! # Usage
//!
//! ```ignore
//! let mut cache = PreallocKvCache::new(num_layers, max_seq_len);
//! for step in 0..max_tokens {
//!     let (logits, _) = model.forward(&input, &mut cache)?;
//!     let next_token = sample(&logits);
//! }
//! ```
//!
//! # Compiled decoder integration
//!
//! Compiled decoders (via `trace_graph` + `CompiledModel`) trace a single forward
//! step without KV cache. At runtime, the compiled step produces new K/V tensors
//! per layer. The caller appends these to `PreallocKvCache` and passes the full
//! cached K/V back as attention context for the next step. The pre-allocated
//! buffers ensure stable GPU memory layout across all decode steps.

use crate::dyn_tensor::DynTensor;
use crate::{Result, TensorError};

use super::kv_cache::traits::{alloc_buffer, validate_dims_match, validate_kv_pair};

const SEQ_DIM: usize = 2;

/// Per-layer pre-allocated KV cache for GPU-resident decoder inference.
///
/// Allocates key and value buffers for `max_seq_len` positions on first
/// `append()`. All subsequent appends write into the pre-allocated buffer
/// at the current offset — no reallocation, no doubling, no GPU-side copies
/// of the full buffer.
///
/// Implements [`KvCacheLayerBackend`](super::KvCacheLayerBackend) for drop-in
/// compatibility with existing model code.
#[derive(Debug, Clone)]
pub struct PreallocKvCacheLayer {
    key_buf: Option<DynTensor>,
    value_buf: Option<DynTensor>,
    current_len: usize,
    max_seq_len: usize,
}

impl PreallocKvCacheLayer {
    /// Create a new pre-allocated cache layer.
    ///
    /// `max_seq_len` is the maximum number of sequence positions this layer
    /// can hold. Buffers are lazily allocated on the first `append()` call
    /// (to infer shape, dtype, and device from the input tensors).
    ///
    /// # Errors
    ///
    /// Returns `TensorError::ValueOutOfRange` if `max_seq_len` is zero.
    pub fn new(max_seq_len: usize) -> Result<Self> {
        if max_seq_len == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "PreallocKvCacheLayer: max_seq_len must be > 0",
            });
        }
        Ok(Self {
            key_buf: None,
            value_buf: None,
            current_len: 0,
            max_seq_len,
        })
    }

    /// Append new key/value tensors and return the full cached K/V.
    ///
    /// `new_key` and `new_value` should have shape `[batch, num_kv_heads, new_seq, head_dim]`.
    /// Returns `(full_key, full_value)` as narrow views of the filled portion
    /// along dim=2 (sequence dimension).
    ///
    /// On the first call, allocates buffers sized to `max_seq_len`. Subsequent
    /// calls write into the existing buffer at `current_len` offset.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - K/V shapes don't match
    /// - Appending would exceed `max_seq_len`
    /// - Non-sequence dimensions don't match the existing buffer
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
                "PreallocKvCache sequence length overflow".into(),
            ))?;

        if needed > self.max_seq_len {
            return Err(TensorError::ValueOutOfRange {
                description: "PreallocKvCache: append would exceed max_seq_len",
            });
        }

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
            // First append: allocate buffers at max_seq_len capacity.
            let key_buf = alloc_buffer(new_key, self.max_seq_len)?;
            let value_buf = alloc_buffer(new_value, self.max_seq_len)?;
            self.key_buf = Some(key_buf.slice_set_into(SEQ_DIM, 0, new_key)?);
            self.value_buf = Some(value_buf.slice_set_into(SEQ_DIM, 0, new_value)?);
            self.current_len = new_seq;
        } else {
            // Write into existing buffer at current offset.
            let kb = self.key_buf.take().ok_or_else(|| {
                TensorError::InvalidShape("PreallocKvCache key buffer missing during append".into())
            })?;
            let vb = self.value_buf.take().ok_or_else(|| {
                TensorError::InvalidShape(
                    "PreallocKvCache value buffer missing during append".into(),
                )
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
                TensorError::InvalidShape("PreallocKvCache key buffer missing after append".into())
            })?
            .narrow(SEQ_DIM, 0, self.current_len)?;
        let full_value = self
            .value_buf
            .as_ref()
            .ok_or_else(|| {
                TensorError::InvalidShape(
                    "PreallocKvCache value buffer missing after append".into(),
                )
            })?
            .narrow(SEQ_DIM, 0, self.current_len)?;
        Ok((full_key, full_value))
    }

    /// Current cached sequence length (0 if empty).
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.current_len
    }

    /// Maximum sequence length this cache can hold.
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Whether this layer has no cached entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    /// Full cached key tensor, if any.
    ///
    /// Returns `Ok(None)` if cache is empty. Returns a narrow view of the
    /// filled portion.
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
    /// filled portion.
    pub fn value(&self) -> Result<Option<DynTensor>> {
        if self.current_len == 0 {
            return Ok(None);
        }
        match &self.value_buf {
            None => Ok(None),
            Some(buf) => Ok(Some(buf.narrow(SEQ_DIM, 0, self.current_len)?)),
        }
    }

    /// Reset the cache: clear entries and release buffers.
    ///
    /// After reset, the next `append()` re-allocates buffers at `max_seq_len`.
    pub fn reset(&mut self) {
        self.key_buf = None;
        self.value_buf = None;
        self.current_len = 0;
    }

    /// Clear cached entries but preserve buffer allocations for reuse.
    ///
    /// The next `append()` writes at offset 0 of the existing buffer —
    /// no re-allocation needed. Ideal for batch inference where successive
    /// inputs reuse the same buffer.
    pub fn clear(&mut self) {
        self.current_len = 0;
    }

    /// Remaining capacity in sequence positions.
    #[must_use]
    pub fn remaining_capacity(&self) -> usize {
        self.max_seq_len - self.current_len
    }

    /// Whether the buffers have been allocated (i.e., at least one append
    /// has occurred since construction or last reset).
    #[must_use]
    pub fn is_allocated(&self) -> bool {
        self.key_buf.is_some()
    }
}

impl super::KvCacheLayerBackend for PreallocKvCacheLayer {
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

#[cfg(kani)]
#[path = "kani_prealloc_kv_cache.rs"]
mod kani_prealloc_kv_cache;

#[cfg(test)]
#[path = "prealloc_kv_cache_tests.rs"]
mod tests;
