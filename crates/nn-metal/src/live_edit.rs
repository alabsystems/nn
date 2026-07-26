// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! High-level atomic weight edit coordinator.
//!
//! [`LiveEditApply`] orchestrates the full fence → write → invalidate → resume
//! protocol for live model weight surgery. This is the "apply" step in the
//! locate → compute → verify → **apply** → certify pipeline.
//!
//! # Protocol
//!
//! 1. **Fence** — flush all pending GPU command buffers via [`gpu_scope::flush()`]
//! 2. **Apply ΔW** — write new data into Metal shared buffer via
//!    [`apply_weight_edit`](crate::weight_edit::apply_weight_edit)
//! 3. **Invalidate KV cache** — if provided, clear stale cached states
//! 4. **Resume** — next inference uses edited weights
//!
//! # Example
//!
//! ```ignore
//! use nn_metal::live_edit::LiveEditApply;
//! use nn_metal::weight_edit::WeightEditSpec;
//!
//! let spec = WeightEditSpec {
//!     layer_name: "encoder.linear.weight",
//!     new_data: &edited_weights,
//! };
//! let receipt = LiveEditApply::apply(&mut buffer, &spec, Some(&mut kv_cache))?;
//! assert!(receipt.kv_invalidated);
//! ```

use crate::buffer::MetalBuffer;
use crate::error::MetalError;
use crate::weight_edit::{apply_weight_edit, WeightEditError, WeightEditSpec};
use nn_core::layers::KvCacheLayer;

/// Receipt from a successful live weight edit.
///
/// Records what happened during the edit for audit trails and certificate
/// generation. The `previous_generation` and `new_generation` fields match
/// the KV cache weight generation counter (if a cache was invalidated).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplyReceipt {
    /// Number of f32 elements written to the buffer.
    pub elements_written: usize,
    /// Whether a KV cache was invalidated as part of this edit.
    pub kv_invalidated: bool,
    /// KV cache weight generation before the edit (0 if no cache provided).
    pub kv_generation_before: u64,
    /// KV cache weight generation after the edit (0 if no cache provided).
    pub kv_generation_after: u64,
}

/// Receipt from a successful delta weight edit.
///
/// Records the delta-apply protocol outcome for audit trails and certificate
/// generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaApplyReceipt {
    /// Number of f32 elements written to the buffer.
    pub elements_written: usize,
    /// Number of KV cache layers invalidated (0 if no cache provided).
    pub layers_invalidated: usize,
}

/// High-level atomic weight edit coordinator.
///
/// Combines GPU fence, buffer write, and KV cache invalidation into a
/// single operation. Use this instead of calling [`apply_weight_edit`]
/// directly when you need the full protocol.
pub struct LiveEditApply;

impl LiveEditApply {
    /// Apply a weight edit atomically with fence and KV cache invalidation.
    ///
    /// Protocol:
    /// 1. Flush all pending GPU command buffers (fence)
    /// 2. Write new data into the target buffer
    /// 3. Invalidate KV cache if provided (clears stale entries, bumps generation)
    ///
    /// After this call completes, the next inference forward pass will use the
    /// edited weights. On Apple Silicon unified memory, no explicit memory
    /// barrier is needed — the flush + write_contents sequence ensures coherence.
    ///
    /// # Errors
    ///
    /// Returns [`LiveEditError::Fence`] if the GPU flush fails.
    /// Returns [`LiveEditError::WeightEdit`] if the buffer write fails
    /// (empty data, non-finite values, or buffer size mismatch).
    pub fn apply(
        target_buffer: &mut MetalBuffer,
        spec: &WeightEditSpec<'_>,
        kv_cache: Option<&mut KvCacheLayer>,
    ) -> Result<ApplyReceipt, LiveEditError> {
        // Step 1: Fence — wait for all in-flight GPU command buffers.
        // This ensures no GPU kernel is reading from the target buffer.
        crate::gpu_scope::flush().map_err(|e| LiveEditError::Fence { source: e })?;

        // Step 2: Apply ΔW — write new data into the Metal buffer.
        let elements_written = apply_weight_edit(target_buffer, spec)?;

        // Step 3: Invalidate KV cache if provided.
        let (kv_invalidated, kv_generation_before, kv_generation_after) = match kv_cache {
            Some(cache) => {
                let gen_before = cache.weight_generation();
                cache.invalidate();
                let gen_after = cache.weight_generation();
                (true, gen_before, gen_after)
            }
            None => (false, 0, 0),
        };

        // Step 4: Resume — no explicit action needed. The next forward pass
        // reads the updated buffer contents directly (Apple Silicon unified memory).

        Ok(ApplyReceipt {
            elements_written,
            kv_invalidated,
            kv_generation_before,
            kv_generation_after,
        })
    }

    /// Apply a delta weight edit: `W_new = W_old + ΔW`.
    ///
    /// Full atomic delta-apply protocol:
    /// 1. **Fence** — flush all pending GPU command buffers
    /// 2. **Read** — read current buffer contents as f32
    /// 3. **Extract** — extract delta data from `DynTensor`
    /// 4. **Validate size** — delta length must match buffer length
    /// 5. **Delta add** — compute `W_new = W_old + ΔW` element-wise
    /// 6. **Finite check** — reject result containing NaN/Inf
    /// 7. **Write** — write `W_new` back into GPU buffer
    /// 8. **KV invalidate** — invalidate all KV cache layers
    ///
    /// Unlike [`apply`](Self::apply) which takes pre-computed `WeightEditSpec`,
    /// this method reads current weights from the buffer and adds the delta
    /// tensor, implementing the rank-update pattern for LoRA and fine-tuning.
    ///
    /// # Errors
    ///
    /// Returns [`LiveEditError`] variants for fence failure, buffer read
    /// failure, size mismatch, non-finite results, or buffer write failure.
    pub fn apply_delta(
        target_buffer: &mut MetalBuffer,
        delta_w: &nn_core::DynTensor,
        kv_cache: Option<&mut nn_core::layers::KvCache>,
    ) -> Result<DeltaApplyReceipt, LiveEditError> {
        // Step 1: Fence — wait for all in-flight GPU command buffers.
        crate::gpu_scope::flush().map_err(|e| LiveEditError::Fence { source: e })?;

        // Step 2: Read current buffer contents.
        let current: &[f32] = target_buffer
            .contents()
            .map_err(LiveEditError::BufferRead)?;
        let buf_len = current.len();

        // Step 3: Extract delta from DynTensor → f32 ndarray.
        let delta_arr = delta_w
            .to_f32_array()
            .map_err(LiveEditError::DeltaExtract)?;
        let delta_slice = delta_arr
            .as_slice()
            .ok_or(LiveEditError::DeltaNotContiguous)?;

        // Step 4: Size check.
        if delta_slice.len() != buf_len {
            return Err(LiveEditError::DeltaSizeMismatch {
                buffer_len: buf_len,
                delta_len: delta_slice.len(),
            });
        }

        // Step 5: Delta add — W_new = W_old + ΔW.
        let new_data: Vec<f32> = current
            .iter()
            .zip(delta_slice.iter())
            .map(|(w, d)| w + d)
            .collect();

        // Step 6: Finite check — reject NaN/Inf in result.
        let non_finite = crate::count_non_finite(&new_data);
        if non_finite > 0 {
            return Err(LiveEditError::NonFiniteResult { count: non_finite });
        }

        // Step 7: Write W_new back into the GPU buffer.
        target_buffer
            .write_contents(&new_data)
            .map_err(LiveEditError::BufferWrite)?;

        // Step 8: Invalidate all KV cache layers.
        let layers_invalidated = match kv_cache {
            Some(cache) => {
                let n = cache.num_layers();
                for i in 0..n {
                    if let Ok(layer) = cache.layer_mut(i) {
                        layer.invalidate();
                    }
                }
                n
            }
            None => 0,
        };

        Ok(DeltaApplyReceipt {
            elements_written: new_data.len(),
            layers_invalidated,
        })
    }
}

/// Errors from the live edit protocol.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LiveEditError {
    /// GPU fence (flush) failed before the edit could be applied.
    #[error("live edit fence failed: {source}")]
    Fence {
        #[source]
        source: nn_core::TensorError,
    },
    /// The underlying weight edit operation failed.
    #[error(transparent)]
    WeightEdit(#[from] WeightEditError),
    /// Buffer read failed during delta-apply (could not read current weights).
    #[error("live edit buffer read failed: {0}")]
    BufferRead(MetalError),
    /// Delta tensor element count does not match buffer element count.
    #[error("delta size mismatch: buffer has {buffer_len} elements, delta has {delta_len}")]
    DeltaSizeMismatch { buffer_len: usize, delta_len: usize },
    /// Failed to extract f32 data from delta DynTensor.
    #[error("delta tensor extraction failed: {0}")]
    DeltaExtract(nn_core::TensorError),
    /// Delta tensor data is not contiguous in memory.
    #[error("delta tensor is not contiguous")]
    DeltaNotContiguous,
    /// The result of `W_old + ΔW` contains NaN or Inf values.
    #[error("delta-apply produced {count} non-finite values")]
    NonFiniteResult { count: usize },
    /// Buffer write failed after delta computation.
    #[error("live edit buffer write failed: {0}")]
    BufferWrite(MetalError),
}

#[cfg(test)]
#[path = "live_edit_tests.rs"]
mod tests;
