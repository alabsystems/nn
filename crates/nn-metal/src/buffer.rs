// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared CPU/GPU Metal buffer abstraction.
//!
//! [`MetalBuffer`] wraps a Metal buffer allocated in shared storage mode,
//! providing typed access via [`bytemuck::Pod`]. The [`contents_element_count`]
//! helper computes safe element counts for typed buffer views and is verified
//! by four Kani harnesses.

/// Shared CPU/GPU Metal buffer wrapper.
///
/// `MetalBuffer` is intentionally not `Clone`. Buffers created via
/// [`MetalContext::create_buffer_no_copy`] reference externally-owned memory
/// (e.g., mmap pages managed by [`WeightMap`]). A derived `Clone` would allow
/// the clone to outlive the backing memory, causing use-after-unmap UB.
/// Use [`MetalContext::clone_buffer`] for safe data-copying clones (#598).
#[derive(Debug)]
#[non_exhaustive]
pub struct MetalBuffer {
    inner: metal::Buffer,
    len_bytes: usize,
}

impl MetalBuffer {
    pub(crate) fn from_raw(inner: metal::Buffer, len_bytes: usize) -> Self {
        Self { inner, len_bytes }
    }

    /// Buffer size in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len_bytes
    }

    /// Returns `true` if the buffer has zero bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len_bytes == 0
    }

    #[must_use]
    pub(crate) fn inner(&self) -> &metal::BufferRef {
        &self.inner
    }

    /// Create a shallow alias of this buffer (zero-copy).
    ///
    /// Increments the underlying Metal buffer's reference count (ARC) without
    /// copying data. The aliased buffer shares the same GPU allocation and its
    /// data reflects any pending GPU writes, unlike `MetalContext::clone_buffer`
    /// which does a CPU-side memcpy and would read stale zeros from
    /// not-yet-committed command buffers.
    ///
    /// Used internally by reshape dispatch and externally by downstream crates
    /// (dvoice-metal) to pass nn GPU tensor buffers directly to custom
    /// Metal kernel dispatches without GPU→CPU→GPU round-trips.
    ///
    /// # Safety contract
    ///
    /// The aliased buffer is valid for as long as any reference to the
    /// underlying `metal::Buffer` exists (Objective-C ARC). For `no_copy`
    /// buffers (mmap-backed weights), the caller must ensure the backing
    /// `WeightMap` outlives the alias.
    #[must_use]
    pub fn alias(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            len_bytes: self.len_bytes,
        }
    }

    /// Returns `true` if `self` and `other` reference the same Metal allocation.
    ///
    /// Compares the CPU-mapped `contents()` pointers: aliased buffers share
    /// the same Objective-C buffer object and therefore the same pointer.
    /// Used by the blit-skip logic in `run_steps_inner` to detect when a
    /// NativeOp or Dispatch step wrote directly into the planned buffer (#3435, #3448).
    pub(crate) fn is_same_allocation(&self, other: &Self) -> bool {
        std::ptr::eq(
            self.inner.contents() as *const u8,
            other.inner.contents() as *const u8,
        )
    }

    /// Read a typed sub-slice from the buffer at a byte offset.
    ///
    /// Returns `count` elements of type `T` starting at `byte_offset` bytes
    /// from the beginning of the buffer. Returns `Err` if the offset + data
    /// would exceed the buffer length, or if the offset is not aligned for `T`.
    ///
    /// # Synchronization
    ///
    /// Same requirements as [`contents`](Self::contents): all GPU writes must
    /// be complete before reading.
    #[must_use = "returns a Result that may contain an error"]
    pub fn contents_at_offset<T: bytemuck::Pod>(
        &self,
        byte_offset: usize,
        count: usize,
    ) -> Result<&[T], crate::error::MetalError> {
        let type_size = size_of::<T>();
        if type_size == 0 || count == 0 {
            return Err(crate::error::MetalError::BufferReadback {
                reason: if type_size == 0 {
                    "zero-size type"
                } else {
                    "zero count"
                },
                buf_len: self.len_bytes,
                type_size,
            });
        }
        let data_bytes =
            count
                .checked_mul(type_size)
                .ok_or(crate::error::MetalError::BufferReadback {
                    reason: "count * type_size overflow",
                    buf_len: self.len_bytes,
                    type_size,
                })?;
        let end = byte_offset.checked_add(data_bytes).ok_or(
            crate::error::MetalError::BufferReadback {
                reason: "offset + data_bytes overflow",
                buf_len: self.len_bytes,
                type_size,
            },
        )?;
        if end > self.len_bytes {
            return Err(crate::error::MetalError::BufferReadback {
                reason: "offset + count exceeds buffer length",
                buf_len: self.len_bytes,
                type_size,
            });
        }

        // SAFETY:
        // - Buffer was created with StorageModeShared, so CPU can read it.
        // - Caller must ensure GPU writes are complete (see doc Synchronization).
        // - bytemuck::Pod guarantees `T` is plain old data.
        // - `end <= self.len_bytes` checked above, so we cannot overrun.
        // - Null and alignment are checked at runtime to prevent UB.
        unsafe {
            let base_ptr = self.inner.contents() as *const u8;
            if base_ptr.is_null() {
                return Err(crate::error::MetalError::BufferReadback {
                    reason: "null buffer pointer (private storage or released buffer)",
                    buf_len: self.len_bytes,
                    type_size,
                });
            }
            let offset_ptr = base_ptr.add(byte_offset).cast::<T>();
            if !offset_ptr.is_aligned() {
                return Err(crate::error::MetalError::BufferReadback {
                    reason: "pointer alignment failure at offset",
                    buf_len: self.len_bytes,
                    type_size,
                });
            }
            Ok(std::slice::from_raw_parts(offset_ptr, count))
        }
    }

    /// Read the buffer as a typed slice.
    ///
    /// Returns `Err` if the buffer is empty, the element type is zero-sized,
    /// or the buffer pointer is not aligned for `T`.
    ///
    /// # Synchronization
    ///
    /// The caller must ensure all GPU command buffers writing to this buffer
    /// have completed before reading. Use `CommandBuffer::wait_until_completed()`
    /// or an equivalent synchronization primitive. Reading during an active GPU
    /// write produces data races (undefined behavior).
    #[must_use = "returns a Result that may contain an error"]
    pub fn contents<T: bytemuck::Pod>(&self) -> Result<&[T], crate::error::MetalError> {
        let type_size = size_of::<T>();
        let count = contents_element_count(self.len_bytes, type_size).ok_or(
            crate::error::MetalError::BufferReadback {
                reason: if type_size == 0 {
                    "zero-size type"
                } else if self.len_bytes == 0 {
                    "empty buffer"
                } else {
                    "buffer too small for element type"
                },
                buf_len: self.len_bytes,
                type_size,
            },
        )?;

        // SAFETY:
        // - Buffer was created with StorageModeShared, so CPU can read it.
        // - Caller must ensure GPU writes are complete (see doc Synchronization).
        // - bytemuck::Pod guarantees `T` is plain old data.
        // - `count` is derived from byte length and element size, so it cannot overrun.
        // - Null and alignment are checked at runtime to prevent UB.
        unsafe {
            let ptr = self.inner.contents() as *const T;
            if ptr.is_null() {
                return Err(crate::error::MetalError::BufferReadback {
                    reason: "null buffer pointer (private storage or released buffer)",
                    buf_len: self.len_bytes,
                    type_size,
                });
            }
            if !ptr.is_aligned() {
                return Err(crate::error::MetalError::BufferReadback {
                    reason: "pointer alignment failure",
                    buf_len: self.len_bytes,
                    type_size,
                });
            }
            Ok(std::slice::from_raw_parts(ptr, count))
        }
    }

    /// Get a mutable typed slice into the buffer.
    ///
    /// Returns `Err` if the buffer is empty, the element type is zero-sized,
    /// or the buffer pointer is not aligned for `T`.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - No other references (shared or mutable) to this buffer's contents exist.
    /// - All GPU command buffers writing to this buffer have completed.
    /// - The buffer will not be submitted to the GPU while the returned slice
    ///   is alive.
    ///
    /// Typically safe when the buffer was just created (not yet submitted) or
    /// just cloned (exclusive ownership).
    #[must_use = "returns a Result that may contain an error"]
    pub(crate) unsafe fn contents_mut<T: bytemuck::Pod>(
        &mut self,
    ) -> Result<&mut [T], crate::error::MetalError> {
        let type_size = size_of::<T>();
        let count = contents_element_count(self.len_bytes, type_size).ok_or(
            crate::error::MetalError::BufferReadback {
                reason: if type_size == 0 {
                    "zero-size type"
                } else if self.len_bytes == 0 {
                    "empty buffer"
                } else {
                    "buffer too small for element type"
                },
                buf_len: self.len_bytes,
                type_size,
            },
        )?;

        // SAFETY:
        // - Buffer was created with StorageModeShared, so CPU can write it.
        // - Caller guarantees exclusive access and no active GPU work.
        // - bytemuck::Pod guarantees `T` is plain old data (no drop/init invariants).
        // - `count` is derived from byte length and element size, so it cannot overrun.
        // - Null and alignment are checked at runtime to prevent UB.
        let ptr = self.inner.contents().cast::<T>();
        if ptr.is_null() {
            return Err(crate::error::MetalError::BufferReadback {
                reason: "null buffer pointer (private storage or released buffer)",
                buf_len: self.len_bytes,
                type_size,
            });
        }
        if !ptr.is_aligned() {
            return Err(crate::error::MetalError::BufferReadback {
                reason: "pointer alignment failure",
                buf_len: self.len_bytes,
                type_size,
            });
        }
        // SAFETY: ptr is non-null, aligned, and count * size_of::<T>() <= len_bytes.
        // Caller guarantees exclusive access (no aliasing references exist).
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr, count) })
    }

    /// Write `f32` data into the buffer, overwriting existing contents.
    ///
    /// Returns `Err` if the data length exceeds the buffer capacity.
    ///
    /// # Safety
    ///
    /// The caller must ensure no GPU command buffers are reading from this
    /// buffer while the write is in progress. The buffer must have been
    /// created with `StorageModeShared`.
    pub fn write_contents(&mut self, data: &[f32]) -> Result<(), crate::error::MetalError> {
        if data.is_empty() {
            return Err(crate::error::MetalError::BufferReadback {
                reason: "empty data slice",
                buf_len: self.len_bytes,
                type_size: size_of::<f32>(),
            });
        }
        let data_bytes = size_of_val(data);
        if data_bytes > self.len_bytes {
            return Err(crate::error::MetalError::BufferReadback {
                reason: "data exceeds buffer capacity",
                buf_len: self.len_bytes,
                type_size: size_of::<f32>(),
            });
        }
        // SAFETY: Caller guarantees exclusive access (no active GPU reads).
        // contents_mut validates pointer alignment and null.
        let dst = unsafe { self.contents_mut::<f32>()? };
        dst[..data.len()].copy_from_slice(data);
        Ok(())
    }
}

/// Computes element count for a typed buffer view.
#[inline]
#[must_use]
pub(crate) fn contents_element_count(buf_len: usize, type_size: usize) -> Option<usize> {
    if type_size == 0 || buf_len == 0 {
        return None;
    }
    let count = buf_len / type_size;
    if count == 0 {
        return None;
    }
    Some(count)
}

/// Validate that a byte offset is within the buffer's allocated length.
///
/// Returns `MetalError::BufferOffsetOutOfBounds` if `byte_offset > buffer.len()`.
/// An offset equal to `buffer.len()` is technically valid (zero-length view at end)
/// but an offset exceeding it would cause Metal to read/write out-of-bounds.
///
/// Use this at FFI boundaries before binding buffers with non-zero offsets
/// to Metal compute encoders. Part of #4321.
#[inline]
pub(crate) fn validate_buffer_offset(
    buffer: &MetalBuffer,
    byte_offset: usize,
    role: &'static str,
) -> Result<(), crate::error::MetalError> {
    if byte_offset > buffer.len() {
        return Err(crate::error::MetalError::BufferOffsetOutOfBounds {
            buffer_len: buffer.len(),
            offset: byte_offset,
            role,
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "buffer_tests.rs"]
mod tests;

#[cfg(kani)]
mod proofs {
    use super::contents_element_count;

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn contents_count_never_overruns_buffer() {
        let buf_len: usize = kani::any();
        let type_size: usize = kani::any();
        kani::assume(type_size > 0 && type_size <= 32);
        kani::assume(buf_len > 0 && buf_len <= (1usize << 20));

        if let Some(count) = contents_element_count(buf_len, type_size) {
            assert!(count > 0, "Some implies at least one element");
            assert!(count.checked_mul(type_size).is_some());
            assert!(count * type_size <= buf_len);
        }
        // None is valid when buf_len < type_size (can't hold one element)
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn contents_zst_and_empty_return_none() {
        let buf_len: usize = kani::any();
        kani::assume(buf_len <= (1usize << 20));
        assert!(contents_element_count(buf_len, 0).is_none());

        let type_size: usize = kani::any();
        kani::assume(type_size <= 32);
        assert!(contents_element_count(0, type_size).is_none());
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn metal_alignment_sufficient_for_pod_types() {
        let base_addr: usize = kani::any();
        let type_align: usize = kani::any();

        kani::assume(base_addr % 16 == 0);
        kani::assume(base_addr <= (1usize << 20));
        kani::assume(type_align > 0 && type_align <= 16);
        kani::assume(type_align.is_power_of_two());

        assert_eq!(base_addr % type_align, 0);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn contents_count_is_maximal() {
        let buf_len: usize = kani::any();
        let type_size: usize = kani::any();
        kani::assume(type_size > 0 && type_size <= 32);
        kani::assume(buf_len > 0 && buf_len <= (1usize << 20));

        if let Some(count) = contents_element_count(buf_len, type_size) {
            assert!((count + 1) * type_size > buf_len);
        }
        // None when buf_len < type_size — no elements fit
    }
}
