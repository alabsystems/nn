// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bump-allocator arena for GPU activation tensors.
//!
//! [`ActivationArena`] pre-allocates a single large Metal buffer and
//! sub-allocates from it via bump pointer. Intermediate tensors during a
//! model forward pass share the arena buffer (via [`MetalBuffer::alias`]
//! with [`MetalTensorData::view`] offsets), eliminating per-tensor Metal
//! allocator calls.
//!
//! # Auto-grow (Part of #4289)
//!
//! When auto-grow is enabled (via [`set_auto_grow`](ActivationArena::set_auto_grow)),
//! overflow allocations trigger a new larger slab instead of returning
//! `ArenaOverflow`. Retired slabs are kept alive (ObjC ARC) and dropped
//! on [`reset`](ActivationArena::reset). The default thread-local arena
//! enables auto-grow automatically.
//!
//! # Usage pattern
//!
//! ```rust,ignore
//! let mut arena = ActivationArena::new(&ctx, 64 * 1024 * 1024)?; // 64 MB
//! arena.set_auto_grow(&ctx); // enable auto-grow for large workloads
//! loop {
//!     let output = with_arena(&mut arena, || model.forward(&input))?;
//!     let audio = output.as_cpu_f32()?;
//!     drop(output);
//!     arena.reset(); // reclaim all arena memory
//! }
//! ```
//!
//! See `designs/2026-03-12-arena-tensor-allocation.md` for full design.

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::dyn_tensor_metal::MetalTensorData;
use crate::error::MetalError;

/// Metal buffer alignment in bytes. Metal requires buffer offsets to be aligned
/// to this boundary for `set_buffer(_:offset:atIndex:)`.
const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Bump-allocator arena for GPU activation tensors.
///
/// Pre-allocates a single Metal buffer and sub-allocates regions via a bump
/// pointer. Each [`alloc`](Self::alloc) returns a [`MetalTensorData`] that
/// is an aliased view into the arena buffer at the current offset.
///
/// When auto-grow is enabled, overflow allocations trigger a new larger slab
/// instead of returning `ArenaOverflow`. Retired slabs stay alive via ObjC
/// ARC and are dropped on [`reset`](Self::reset). Part of #4289.
///
/// Call [`reset`](Self::reset) between forward passes to reclaim all arena
/// memory without Metal deallocation.
#[derive(Debug)]
pub struct ActivationArena {
    buffer: MetalBuffer,
    offset: usize,
    capacity: usize,
    peak_bytes: usize,
    /// Total bytes allocated across all slabs in the current generation.
    total_allocated: usize,
    generation: u64,
    /// When true, overflow triggers slab growth instead of `ArenaOverflow`.
    auto_grow: bool,
    /// Metal context for allocating new slabs on growth.
    ctx: Option<MetalContext>,
    /// Retired slabs kept alive for ObjC ARC. Cleared on reset.
    retired_slabs: Vec<MetalBuffer>,
    /// Growth events in the current generation (reset on each reset()).
    growth_count: usize,
    /// Total growth events since arena creation.
    total_growth_count: usize,
    /// Overflow count in the current generation (alloc requests that exceeded
    /// the current slab's remaining capacity). Part of #4289.
    overflow_count: usize,
    /// Total overflow count since arena creation. Part of #4289.
    total_overflow_count: usize,
    /// Cumulative bytes allocated via overflow in the current generation.
    /// Part of #4289.
    overflow_bytes: usize,
    /// Total bytes allocated via overflow since arena creation. Part of #4289.
    total_overflow_bytes: usize,
}

impl ActivationArena {
    /// Create a new arena with the given byte capacity.
    ///
    /// Allocates a single shared-mode Metal buffer. Returns an error if
    /// `capacity_bytes` is zero or if the Metal allocator fails.
    pub fn new(ctx: &MetalContext, capacity_bytes: usize) -> Result<Self, MetalError> {
        if capacity_bytes == 0 {
            return Err(MetalError::BufferCreate(0));
        }
        let buffer = ctx.create_buffer_zeroed(capacity_bytes)?;
        Ok(Self {
            buffer,
            offset: 0,
            capacity: capacity_bytes,
            peak_bytes: 0,
            total_allocated: 0,
            generation: 0,
            auto_grow: false,
            ctx: None,
            retired_slabs: Vec::new(),
            growth_count: 0,
            total_growth_count: 0,
            overflow_count: 0,
            total_overflow_count: 0,
            overflow_bytes: 0,
            total_overflow_bytes: 0,
        })
    }

    /// Enable auto-grow mode.
    ///
    /// When enabled, overflow in [`alloc`](Self::alloc) triggers automatic
    /// slab growth instead of returning `ArenaOverflow`. The current buffer
    /// is retired (kept alive for ObjC ARC), and a new larger buffer is
    /// allocated. Requires `ctx` to allocate new Metal buffers.
    ///
    /// Part of #4289.
    pub fn set_auto_grow(&mut self, ctx: &MetalContext) {
        self.auto_grow = true;
        self.ctx = Some(ctx.clone());
    }

    /// Sub-allocate a region of `byte_len` bytes from the arena.
    ///
    /// Returns a [`MetalTensorData`] that is a zero-copy view (via
    /// [`MetalBuffer::alias`]) into the arena buffer at an aligned offset.
    /// The offset is rounded up to [`METAL_BUFFER_ALIGNMENT`] (256 bytes).
    ///
    /// If auto-grow is enabled and the current slab overflows, a new larger
    /// slab is allocated and the request is retried. Otherwise returns
    /// `ArenaOverflow`.
    pub fn alloc(&mut self, byte_len: usize) -> Result<MetalTensorData, MetalError> {
        if byte_len == 0 {
            return Err(MetalError::BufferCreate(0));
        }
        let aligned_offset = align_up(self.offset, METAL_BUFFER_ALIGNMENT)?;
        let new_offset =
            aligned_offset
                .checked_add(byte_len)
                .ok_or(MetalError::BufferByteOverflow {
                    elems: aligned_offset,
                    elem_size: byte_len,
                })?;
        if new_offset > self.capacity {
            if self.auto_grow {
                return self.grow_and_alloc(byte_len);
            }
            return Err(MetalError::ArenaOverflow {
                requested: byte_len,
                remaining: self.capacity.saturating_sub(aligned_offset),
                capacity: self.capacity,
            });
        }
        self.offset = new_offset;
        self.total_allocated += byte_len;
        if self.total_allocated > self.peak_bytes {
            self.peak_bytes = self.total_allocated;
        }
        Ok(MetalTensorData::view_arena(
            self.buffer.alias(),
            aligned_offset,
            self.generation,
        ))
    }

    /// Grow the arena by retiring the current slab and allocating a new one.
    ///
    /// New slab capacity = max(2 * current_capacity, byte_len + alignment).
    /// The old buffer is moved to `retired_slabs` so existing views remain
    /// valid via ObjC ARC. Part of #4289.
    fn grow_and_alloc(&mut self, byte_len: usize) -> Result<MetalTensorData, MetalError> {
        let ctx = self
            .ctx
            .as_ref()
            .ok_or(MetalError::BufferCreate(0))?
            .clone();
        // New capacity: at least double, at least fits the request.
        let min_needed = byte_len
            .checked_add(METAL_BUFFER_ALIGNMENT)
            .ok_or(MetalError::BufferByteOverflow {
                elems: byte_len,
                elem_size: METAL_BUFFER_ALIGNMENT,
            })?;
        let new_capacity = self
            .capacity.saturating_mul(2)
            .max(min_needed);
        let new_buffer = ctx.create_buffer_zeroed(new_capacity)?;

        // Retire the old buffer -- existing views keep it alive via ARC.
        let old_buffer = std::mem::replace(&mut self.buffer, new_buffer);
        self.retired_slabs.push(old_buffer);
        self.capacity = new_capacity;
        self.offset = 0;
        self.growth_count += 1;
        self.total_growth_count += 1;
        // Track the overflow event and bytes. Part of #4289.
        self.overflow_count += 1;
        self.total_overflow_count += 1;
        self.overflow_bytes += byte_len;
        self.total_overflow_bytes += byte_len;

        // Allocate from the fresh slab -- guaranteed to fit.
        self.offset = byte_len;
        self.total_allocated += byte_len;
        if self.total_allocated > self.peak_bytes {
            self.peak_bytes = self.total_allocated;
        }
        Ok(MetalTensorData::view_arena(
            self.buffer.alias(),
            0, // Fresh slab starts at offset 0.
            self.generation,
        ))
    }

    /// Reset the bump pointer to zero, reclaiming all arena memory.
    ///
    /// Does NOT deallocate the underlying Metal buffer. The caller must
    /// ensure no outstanding `DynTensor` references to arena memory exist
    /// (i.e., all intermediate tensors from the forward pass are dropped).
    ///
    /// If references still exist, the aliased Metal buffer remains valid
    /// (ObjC ARC keeps it alive), but the arena will overwrite that memory
    /// on subsequent allocations.
    ///
    /// Drops retired slabs from auto-grow events. If the arena grew, the
    /// current (larger) buffer is retained for the next generation.
    pub fn reset(&mut self) {
        self.offset = 0;
        self.total_allocated = 0;
        self.growth_count = 0;
        self.overflow_count = 0;
        self.overflow_bytes = 0;
        self.retired_slabs.clear();
        self.generation += 1;
    }

    /// Peak bytes used across all forward passes since arena creation.
    ///
    /// When auto-grow is enabled, this tracks the total bytes allocated
    /// across all slabs, not just the current slab.
    #[must_use]
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }

    /// Current generation counter (incremented on each [`reset`](Self::reset)).
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Total arena capacity in bytes (current slab).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current byte offset (amount allocated in the current slab).
    #[must_use]
    pub fn used_bytes(&self) -> usize {
        self.offset
    }

    /// Remaining bytes available in the current slab.
    #[must_use]
    pub fn remaining_bytes(&self) -> usize {
        self.capacity.saturating_sub(self.offset)
    }

    /// Number of growth events in the current generation.
    #[must_use]
    pub fn growth_count(&self) -> usize {
        self.growth_count
    }

    /// Total number of growth events since arena creation.
    #[must_use]
    pub fn total_growth_count(&self) -> usize {
        self.total_growth_count
    }

    /// Number of retired slabs held alive for ObjC ARC references.
    #[must_use]
    pub fn retired_slab_count(&self) -> usize {
        self.retired_slabs.len()
    }

    /// Whether auto-grow mode is enabled.
    #[must_use]
    pub fn is_auto_grow(&self) -> bool {
        self.auto_grow
    }

    /// Number of overflow events in the current generation.
    ///
    /// An overflow occurs when an allocation request exceeds the current
    /// slab's remaining capacity. With auto-grow enabled, overflows trigger
    /// slab growth rather than errors. Part of #4289.
    #[must_use]
    pub fn overflow_count(&self) -> usize {
        self.overflow_count
    }

    /// Total number of overflow events since arena creation.
    ///
    /// Persists across resets. Use to diagnose whether the initial arena
    /// capacity is sufficient across an entire session. Part of #4289.
    #[must_use]
    pub fn total_overflow_count(&self) -> usize {
        self.total_overflow_count
    }

    /// Cumulative bytes allocated via overflow in the current generation.
    ///
    /// This is the total bytes that triggered slab growth (or would have
    /// caused `ArenaOverflow` without auto-grow). Part of #4289.
    #[must_use]
    pub fn overflow_bytes(&self) -> usize {
        self.overflow_bytes
    }

    /// Total overflow bytes since arena creation.
    ///
    /// Persists across resets. Part of #4289.
    #[must_use]
    pub fn total_overflow_bytes(&self) -> usize {
        self.total_overflow_bytes
    }

    /// Ensure the arena can hold at least `min_bytes` without growing.
    ///
    /// If the current slab's remaining capacity is less than `min_bytes`,
    /// grows to a new slab. This is a pre-sizing API: call before a
    /// workload to avoid growth during the hot path. Part of #4289.
    pub fn ensure_capacity(
        &mut self,
        ctx: &MetalContext,
        min_bytes: usize,
    ) -> Result<(), MetalError> {
        if self.remaining_bytes() >= min_bytes {
            return Ok(());
        }
        let new_capacity = min_bytes.max(self.capacity);
        let new_buffer = ctx.create_buffer_zeroed(new_capacity)?;
        let old_buffer = std::mem::replace(&mut self.buffer, new_buffer);
        if self.offset > 0 {
            self.retired_slabs.push(old_buffer);
        }
        self.capacity = new_capacity;
        self.offset = 0;
        Ok(())
    }

    /// Save the current bump pointer offset as a checkpoint.
    ///
    /// Use with [`restore_checkpoint`](Self::restore_checkpoint) to reclaim
    /// arena memory for temporary allocations within a compiled model step.
    /// See #2913.
    #[must_use]
    pub fn checkpoint(&self) -> usize {
        self.offset
    }

    /// Restore the bump pointer to a previously saved checkpoint.
    ///
    /// All allocations made after the checkpoint are logically freed -- their
    /// arena regions may be overwritten by subsequent allocations. The caller
    /// MUST ensure no outstanding references to those regions exist.
    ///
    /// Does NOT change the generation counter (no full reset). This is safe
    /// because compiled model steps extract their output via blit-copy to a
    /// contiguous buffer before the checkpoint is restored.
    ///
    /// # Metal serialization dependency
    ///
    /// The `relocate_to_planned_buffer` blit copies are GPU-side operations
    /// encoded into the same command buffer. Metal serializes encoder execution
    /// within a command buffer, so blit copies complete before this method runs
    /// on the CPU side. A future refactor splitting blit and compute into
    /// separate command buffers would need explicit synchronization here.
    /// Part of #2218 F8.
    ///
    /// # Panics
    ///
    /// Returns `Err` if `saved_offset > self.offset` (restoring to a future state).
    pub fn restore_checkpoint(&mut self, saved_offset: usize) -> Result<(), MetalError> {
        if saved_offset > self.offset {
            return Err(MetalError::ArenaCheckpoint {
                saved: saved_offset,
                current: self.offset,
            });
        }
        // Allocations made since the checkpoint are logically freed, so the
        // bytes they consumed must be returned to `total_allocated`. Otherwise
        // repeated checkpoint→temp→restore cycles (e.g. compiled-model steps
        // reusing the same arena region) would inflate `peak_bytes` linearly
        // even though concurrent usage never exceeds the per-step high-water
        // mark. Part of #2913.
        let reclaimed = self.offset - saved_offset;
        self.total_allocated = self.total_allocated.saturating_sub(reclaimed);
        self.offset = saved_offset;
        Ok(())
    }
}

/// Round `offset` up to the next multiple of `alignment`.
/// `alignment` must be a power of two and non-zero.
fn align_up(offset: usize, alignment: usize) -> Result<usize, MetalError> {
    if !alignment.is_power_of_two() {
        return Err(MetalError::InvalidArenaAlignment { alignment });
    }
    let mask = alignment - 1;
    offset
        .checked_add(mask)
        .map(|v| v & !mask)
        .ok_or(MetalError::BufferByteOverflow {
            elems: offset,
            elem_size: alignment,
        })
}

/// Result of [`estimate_arena_peak_bytes`]: the estimated peak arena usage
/// and per-step byte breakdown.
///
/// Part of #4289.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaEstimate {
    /// Estimated peak arena bytes (high-water mark across all steps).
    /// Suitable for passing to [`ActivationArena::new`] or
    /// [`ensure_default_arena_capacity`].
    pub peak_bytes: usize,
    /// Total bytes across all allocation steps (sum, not peak).
    pub total_bytes: usize,
    /// Number of allocation steps counted.
    pub step_count: usize,
}

/// Estimate peak arena bytes needed for a sequence of allocation sizes.
///
/// Walks the provided per-step byte sizes, simulating a bump allocator with
/// 256-byte alignment. Tracks the running high-water mark as a proxy for
/// peak concurrent arena usage.
///
/// # Arguments
///
/// * `step_bytes` - Iterator of byte sizes for each allocation step.
///   Each entry represents one intermediate tensor allocation.
///
/// # Returns
///
/// An [`ArenaEstimate`] with the peak byte count (suitable for
/// [`ActivationArena::new`] or [`ensure_default_arena_capacity`]) and
/// summary statistics.
///
/// # Example
///
/// ```rust,ignore
/// let sizes = vec![1024, 4096, 256, 65536];
/// let est = estimate_arena_peak_bytes(sizes.iter().copied());
/// arena.ensure_capacity(&ctx, est.peak_bytes)?;
/// ```
///
/// Part of #4289.
#[must_use]
pub fn estimate_arena_peak_bytes(step_bytes: impl IntoIterator<Item = usize>) -> ArenaEstimate {
    let mut offset: usize = 0;
    let mut peak: usize = 0;
    let mut total: usize = 0;
    let mut count: usize = 0;

    for bytes in step_bytes {
        if bytes == 0 {
            continue;
        }
        // Simulate alignment: round offset up to METAL_BUFFER_ALIGNMENT.
        let aligned = (offset + (METAL_BUFFER_ALIGNMENT - 1)) & !(METAL_BUFFER_ALIGNMENT - 1);
        let new_offset = aligned.saturating_add(bytes);
        offset = new_offset;
        total = total.saturating_add(bytes);
        count += 1;
        if offset > peak {
            peak = offset;
        }
    }

    ArenaEstimate {
        peak_bytes: peak,
        total_bytes: total,
        step_count: count,
    }
}

/// Estimate peak arena bytes from a list of named allocation entries.
///
/// Convenience wrapper around [`estimate_arena_peak_bytes`] that accepts
/// `(name, shape, elem_size)` tuples. Each entry's byte size is computed
/// as `shape.iter().product::<usize>() * elem_size`.
///
/// # Arguments
///
/// * `entries` - Slice of `(name, shape, elem_size)` tuples describing
///   each intermediate tensor allocation. `name` is for diagnostics only.
///
/// Part of #4289.
#[must_use]
pub fn estimate_arena_peak_from_shapes(
    entries: &[(&str, &[usize], usize)],
) -> ArenaEstimate {
    let sizes = entries.iter().map(|(_, shape, elem_size)| {
        shape.iter().product::<usize>().saturating_mul(*elem_size)
    });
    estimate_arena_peak_bytes(sizes)
}

// Thread-local arena scope: with_arena, without_arena, arena_alloc_or_create.
#[path = "arena_scope.rs"]
mod scope;
pub(crate) use scope::{
    arena_alloc_or_create, arm_planned_redirect_guard, checkpoint_default_arena,
    decode_scope_generation, reset_default_arena,
    restore_default_arena, PlannedRedirectGuard,
};
pub use scope::{
    default_arena_total_growth_count, ensure_default_arena_capacity, try_reset_active_arena,
    with_arena, with_decode_scope, without_arena,
};
// Test-only re-exports: used by arena_tests.rs via `super::`.
#[cfg(test)]
pub(crate) use scope::{
    clear_planned_redirect, is_arena_active, is_arena_bypassed, set_planned_redirect,
};
#[path = "arena_stats.rs"]
mod stats;
pub use pool::PoolStats;
pub use stats::{
    arena_capacity, arena_stats, default_arena_peak_bytes, reset_arena_stats, ArenaStats,
};
pub(crate) use stats::{default_arena_generation, default_arena_used_bytes, last_alloc_generation};

// Thread-local Metal buffer pool for non-arena allocations (#3079 D3).
#[path = "buffer_pool.rs"]
mod pool;
pub(crate) use pool::pool_reclaim;

#[cfg(kani)]
#[path = "arena_kani.rs"]
mod proofs;

#[cfg(test)]
#[path = "arena_tests.rs"]
mod tests;
