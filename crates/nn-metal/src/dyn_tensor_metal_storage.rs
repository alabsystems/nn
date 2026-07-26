// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU storage type for `DynTensor` on Metal.

use crate::buffer::MetalBuffer;

/// Concrete GPU storage for `DynTensor` on Metal.
///
/// Wraps a single `MetalBuffer` with optional byte offset for zero-copy views.
/// Dims and dtype live in `DynTensor` itself. `Send + Sync` is required
/// because `DynTensor::from_gpu_storage` wraps this in `Arc<dyn Any + Send + Sync>`.
///
/// `byte_offset` enables zero-copy GPU narrow: dim-0 narrow returns a view
/// sharing the parent buffer (via `MetalBuffer::alias()`) with an adjusted
/// offset, eliminating CPU memcpy and GPU kernel dispatch. Metal's
/// `set_buffer(_:offset:atIndex:)` natively supports arbitrary byte offsets
/// for buffer binding (#1945).
pub struct MetalTensorData {
    pub(crate) buffer: MetalBuffer,
    /// Byte offset within the buffer where this tensor's data starts.
    /// Zero for freshly allocated buffers; non-zero for narrow views.
    pub(crate) byte_offset: usize,
    /// Arena generation at allocation time, or `None` for non-arena buffers.
    ///
    /// When `Some(gen)`, this tensor was sub-allocated from the thread-local
    /// `ActivationArena` at generation `gen`. On CPU readback, the current
    /// arena generation is compared against this value to detect stale reads
    /// (arena was reset and memory potentially overwritten).
    ///
    /// See `designs/2026-03-14-arena-cross-thread-safety.md` (Option B).
    arena_generation: Option<u64>,
}

impl MetalTensorData {
    /// Create a new `MetalTensorData` wrapping a Metal buffer.
    ///
    /// Used by downstream crates (dvoice-metal) to construct GPU tensors
    /// from kernel output buffers without CPU round-trips.
    pub fn new(buffer: MetalBuffer) -> Self {
        Self {
            buffer,
            byte_offset: 0,
            arena_generation: None,
        }
    }

    /// Create a view into an existing buffer at the given byte offset.
    ///
    /// The buffer should be an alias (via `MetalBuffer::alias()`) of the
    /// parent tensor's buffer. The offset must be within the buffer bounds.
    /// Used by zero-copy dim-0 narrow (#1945).
    pub(crate) fn view(buffer: MetalBuffer, byte_offset: usize) -> Self {
        Self {
            buffer,
            byte_offset,
            arena_generation: None,
        }
    }

    /// Create an arena-backed view with a generation stamp.
    ///
    /// Like [`view`](Self::view), but records the arena generation at
    /// allocation time. On CPU readback, this generation is checked against
    /// the current arena generation to detect stale reads (#2328).
    pub(crate) fn view_arena(buffer: MetalBuffer, byte_offset: usize, generation: u64) -> Self {
        Self {
            buffer,
            byte_offset,
            arena_generation: Some(generation),
        }
    }

    /// Create storage for a buffer obtained from `arena_alloc_or_create`.
    ///
    /// Captures the current arena generation to detect stale reads (#2328).
    /// If no arena is active, falls back to `view()` or `new()` based on offset.
    pub(crate) fn from_arena_alloc(buffer: MetalBuffer, byte_offset: usize) -> Self {
        match crate::arena::last_alloc_generation() {
            Some(g) => Self::view_arena(buffer.alias(), byte_offset, g),
            None if byte_offset > 0 => Self::view(buffer.alias(), byte_offset),
            None => Self::new(buffer),
        }
    }

    /// Arena generation at allocation time, or `None` for non-arena buffers.
    pub(crate) fn arena_generation(&self) -> Option<u64> {
        self.arena_generation
    }

    /// Access the underlying Metal buffer.
    ///
    /// Enables downstream crates to pass GPU buffers directly to custom
    /// Metal kernels, avoiding the GPU→CPU→GPU copy in `to_flat_vec_f32()`
    /// + `create_buffer()`.
    pub fn buffer(&self) -> &MetalBuffer {
        &self.buffer
    }

    /// Byte offset within the buffer where this tensor's data starts.
    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    /// Return a [`GpuSlice`] referencing this tensor's buffer region.
    ///
    /// Pairs the buffer with its byte offset, preventing silent offset loss
    /// when passing GPU data to dispatch functions (#2175).
    pub fn as_gpu_slice(&self) -> crate::gpu_slice::GpuSlice {
        crate::gpu_slice::GpuSlice::from_ref(&self.buffer, self.byte_offset)
    }
}

// SAFETY: `MetalBuffer` wraps a Metal `Buffer` which is an Objective-C object
// with thread-safe reference counting (ARC). Metal buffers in shared storage
// mode can be read from any thread after GPU work is committed. The
// `byte_offset` and `arena_generation` fields are plain data with no
// thread-safety concerns.
unsafe impl Send for MetalTensorData {}
// SAFETY: `&MetalTensorData` only provides `&MetalBuffer` access (no interior
// mutability). Reading buffer contents is safe from multiple threads after
// GPU work is committed. `byte_offset` and `arena_generation` are immutable
// after construction.
unsafe impl Sync for MetalTensorData {}
