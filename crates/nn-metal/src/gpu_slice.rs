// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU buffer region: buffer handle + byte offset.
//!
//! `GpuSlice` pairs a [`MetalBuffer`] with its byte offset, preventing the
//! recurring bug pattern where arena byte offsets are silently lost at
//! integration boundaries (see #2176, #2167, #2009, #2175).
//!
//! Used by [`DispatchInput::Gpu`](crate::tensor_dispatch::DispatchInput::Gpu)
//! and [`CompiledModel`](crate::compiled_model::CompiledModel) to ensure the
//! offset is structurally unforgettable.

use crate::buffer::MetalBuffer;

/// A GPU buffer region: buffer handle + byte offset within the buffer.
///
/// Prevents the pattern where `(MetalBuffer, usize)` pairs are separated
/// and the byte offset is forgotten (defaulting to 0). Every construction
/// site must explicitly specify the offset.
#[derive(Debug)]
pub struct GpuSlice {
    buffer: MetalBuffer,
    byte_offset: usize,
}

impl GpuSlice {
    /// Wrap a buffer with an explicit byte offset.
    pub fn new(buffer: MetalBuffer, byte_offset: usize) -> Self {
        Self {
            buffer,
            byte_offset,
        }
    }

    /// Wrap a buffer at byte offset 0.
    ///
    /// Use for freshly allocated buffers and dedicated weight buffers
    /// (not arena-allocated).
    pub fn zero_offset(buffer: MetalBuffer) -> Self {
        Self {
            buffer,
            byte_offset: 0,
        }
    }

    /// Create a GpuSlice by aliasing (ref-count increment) an existing buffer.
    ///
    /// This is the primary constructor for building `DispatchInput::Gpu` from
    /// a borrowed `&MetalBuffer`. The alias is zero-copy (ARC increment).
    pub fn from_ref(buffer: &MetalBuffer, byte_offset: usize) -> Self {
        Self {
            buffer: buffer.alias(),
            byte_offset,
        }
    }

    /// Create a shallow alias of this slice (zero-copy, same offset).
    ///
    /// Increments the underlying Metal buffer's reference count without
    /// copying data. Used when a GpuSlice needs to be passed to multiple
    /// dispatch inputs while retaining the original.
    #[must_use]
    pub fn alias(&self) -> Self {
        Self {
            buffer: self.buffer.alias(),
            byte_offset: self.byte_offset,
        }
    }

    /// Access the underlying Metal buffer.
    pub fn buffer(&self) -> &MetalBuffer {
        &self.buffer
    }

    /// Byte offset within the buffer where the data starts.
    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    /// Consume the slice and return the underlying buffer.
    pub fn into_buffer(self) -> MetalBuffer {
        self.buffer
    }
}

#[cfg(test)]
#[path = "gpu_slice_tests.rs"]
mod tests;
