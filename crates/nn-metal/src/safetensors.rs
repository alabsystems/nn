// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Zero-copy safetensors loading into Metal shared buffers.
//!
//! Memory-maps a safetensors file and wraps the entire mapping as a single
//! Metal buffer via `newBufferWithBytesNoCopy`. On Apple Silicon unified
//! memory (`MTLResourceStorageModeShared`), CPU and GPU access the same
//! physical pages — no data is copied.
//!
//! Individual tensors are accessed as `(offset, length)` ranges within
//! the single backing buffer. Use [`WeightMap`] with `Arc` for multi-instance
//! weight sharing (e.g., multiple model instances sharing one set of weights).

use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use memmap2::Mmap;
use nn_core::{Tensor, TensorElement};
use safetensors::SafeTensors;

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::metal_backend::{MetalBackend, MetalTensorStorage};

// WeightError, TensorInfo, convert_dtype extracted to safetensors_types.rs (#1575).
#[path = "safetensors_types.rs"]
mod types;
use types::convert_dtype;
pub use types::{TensorInfo, WeightError};

/// Page size on Apple Silicon (4 KiB).
const PAGE_SIZE: usize = 4096;

/// Round `len` up to the next page boundary.
///
/// Uses saturating arithmetic to prevent overflow for inputs near
/// `usize::MAX`. Verified by Kani proof [`page_align_never_wraps`] for all
/// file sizes up to 4 GiB: the result is always `>= len` and page-aligned.
#[inline]
#[must_use]
pub(crate) const fn page_align(len: usize) -> usize {
    len.saturating_add(PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

/// Memory-mapped safetensors file with a single Metal shared buffer.
///
/// The entire file is mmap'd and wrapped in one Metal buffer via
/// `newBufferWithBytesNoCopy`. Individual tensors are accessed as
/// (offset, length) ranges within this buffer. Truly zero-copy on
/// Apple Silicon unified memory.
///
/// Wrap in `Arc<WeightMap>` for multi-instance weight sharing.
///
/// # Drop order
///
/// The Metal buffer must be released before the mmap is unmapped,
/// otherwise the buffer briefly references unmapped memory. Both fields
/// use `ManuallyDrop` and the `Drop` impl enforces the correct order
/// (#522). This is immune to field-reorder bugs.
#[non_exhaustive]
pub struct WeightMap {
    /// Single Metal buffer wrapping the entire mmap.
    buffer: ManuallyDrop<MetalBuffer>,
    /// Keeps the mmap alive — backing memory for the Metal buffer.
    mmap: ManuallyDrop<Mmap>,
    /// Per-tensor metadata indexed by name.
    tensors: HashMap<String, TensorInfo>,
    /// Total file size in bytes.
    file_size: usize,
}

impl Drop for WeightMap {
    fn drop(&mut self) {
        // SAFETY: Drop buffer first (releases the Metal object referencing
        // the mmap'd pages), then drop the mmap (unmaps the pages). This
        // ordering is critical — reversing it would cause a use-after-unmap.
        // ManuallyDrop::drop is safe to call exactly once during Drop.
        unsafe {
            ManuallyDrop::drop(&mut self.buffer);
            ManuallyDrop::drop(&mut self.mmap);
        }
    }
}

// SAFETY: WeightMap is read-only after construction.
//
// 1. Send: Moving WeightMap between threads is safe because:
//    - Mmap is Send+Sync (read-only shared memory mapping).
//    - HashMap<String, TensorInfo> and usize are Send+Sync.
//    - MetalBuffer wraps Retained<ProtocolObject<dyn MTLBuffer>>. objc2 marks
//      this !Send conservatively, but MTLBuffer with StorageModeShared is safe
//      for concurrent access from any thread per Apple's Metal Best Practices
//      Guide: "Metal objects [...] can be shared across threads."
//
// 2. Sync: Concurrent &WeightMap access is safe because:
//    - All public methods are &self (no interior mutability).
//    - contents() returns &[u8] derived from the mmap via the Metal buffer,
//      which is valid for concurrent reads.
//    - No &mut self methods exist — WeightMap is immutable after load().
//
// 3. Drop order: explicit `Drop` impl drops buffer before mmap (#522),
//    so the Metal buffer object is released before the backing mmap is unmapped.
unsafe impl Send for WeightMap {}
unsafe impl Sync for WeightMap {}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for WeightMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightMap")
            .field("tensor_count", &self.tensors.len())
            .field("file_size", &self.file_size)
            .field("buffer_len", &self.buffer.len())
            .finish()
    }
}

impl WeightMap {
    /// Load a safetensors file into a zero-copy Metal shared buffer.
    ///
    /// The file is memory-mapped and wrapped directly as a Metal buffer
    /// on Apple Silicon unified memory. No data is copied.
    ///
    /// # Safety
    ///
    /// The returned `WeightMap` holds an mmap of the file. The file must not
    /// be modified or truncated while the `WeightMap` is alive. This is the
    /// standard mmap contract and is safe in practice for model weight files.
    #[must_use = "returns a Result that may contain an error"]
    pub unsafe fn load(path: &Path, ctx: &MetalContext) -> Result<Self, WeightError> {
        let file = std::fs::File::open(path)?;
        // SAFETY: File was just opened and is kept alive via the struct's
        // ManuallyDrop<Mmap> field. Caller guarantees no concurrent modification
        // per the function's safety contract (standard mmap invariant).
        let mmap = unsafe { Mmap::map(&file)? };
        let file_size = mmap.len();

        // Parse safetensors header to extract tensor metadata.
        let st = SafeTensors::deserialize(&mmap)?;

        // Round up to page boundary for Metal's newBufferWithBytesNoCopy.
        let aligned_len = page_align(file_size);

        // SAFETY:
        // - mmap pointer is page-aligned (OS guarantees this)
        // - aligned_len is a multiple of PAGE_SIZE
        // - mmap field keeps the mapping alive for the buffer's lifetime
        // - Mmap pages beyond file_size are zero-filled by the OS
        // - const-to-mut cast: Mmap is read-only, but Metal API requires
        //   NonNull<c_void>. The resulting Metal buffer MUST NOT be used as a
        //   GPU write target (would SIGBUS). WeightMap is a read-only store.
        let ptr =
            NonNull::new(mmap.as_ptr().cast_mut().cast()).ok_or(MetalError::NullMmapPointer)?;
        let buffer = unsafe { ctx.create_buffer_no_copy(ptr, aligned_len)? };

        // Build tensor index: name -> (offset, byte_len, dtype, shape).
        let base = mmap.as_ptr() as usize;
        let mut tensors = HashMap::new();
        for (name, view) in st.tensors() {
            let dtype = convert_dtype(view.dtype())?;
            let data = view.data();
            let offset = data.as_ptr() as usize - base;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    offset,
                    byte_len: data.len(),
                    dtype,
                    shape: view.shape().to_vec(),
                },
            );
        }

        Ok(Self {
            buffer: ManuallyDrop::new(buffer),
            mmap: ManuallyDrop::new(mmap),
            tensors,
            file_size,
        })
    }

    /// Load a safetensors file using the global Metal context.
    ///
    /// Convenience wrapper around [`WeightMap::load`] that avoids requiring
    /// consumers to manually obtain and pass a `MetalContext` reference.
    /// Requires [`MetalBackend::init`] to have been called first.
    ///
    /// # Safety
    ///
    /// Same as [`WeightMap::load`]: the file must not be modified while the
    /// `WeightMap` is alive.
    #[must_use = "returns a Result that may contain an error"]
    pub unsafe fn load_global(path: &Path) -> Result<Self, WeightError> {
        let ctx = crate::metal_backend::global_metal_context()?;
        // SAFETY: Caller guarantees the file is not modified while the
        // WeightMap is alive (forwarded from this function's own `# Safety` contract).
        unsafe { Self::load(path, ctx) }
    }

    /// Get the single Metal buffer backing all tensors.
    #[must_use]
    pub fn buffer(&self) -> &MetalBuffer {
        &self.buffer
    }

    /// Look up tensor metadata by name.
    #[must_use = "returns a Result that may contain an error"]
    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo, WeightError> {
        self.tensors
            .get(name)
            .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))
    }

    /// Iterate over all tensor names.
    #[must_use = "iterator over tensor names is computed but not used"]
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Number of tensors in the weight map.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Total file size in bytes (the mmap'd region).
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.file_size
    }

    /// Read a tensor's data as a byte slice from the mmap.
    ///
    /// Reads from CPU-accessible shared memory — useful for debugging
    /// or converting dtypes before GPU dispatch.
    #[must_use = "returns a Result that may contain an error"]
    pub fn tensor_data(&self, name: &str) -> Result<&[u8], WeightError> {
        let info = self.tensor_info(name)?;
        let all_bytes: &[u8] = self.buffer.contents().map_err(WeightError::Metal)?;
        let end = info.offset.checked_add(info.byte_len).ok_or_else(|| {
            WeightError::TensorDataOverflow {
                name: name.to_string(),
            }
        })?;
        all_bytes
            .get(info.offset..end)
            .ok_or_else(|| WeightError::TensorDataOutOfBounds {
                name: name.to_string(),
                offset: info.offset,
                byte_len: info.byte_len,
                buffer_size: all_bytes.len(),
            })
    }

    /// Load a named weight as a typed `Tensor<D, T, MetalBackend>`.
    ///
    /// Validates shape rank and dimensions against `expected_dims`, and
    /// verifies the stored dtype matches `T::dtype()`. Creates a new
    /// Metal buffer with a copy of the tensor's data (not a sub-buffer
    /// view — safe to use after the `WeightMap` is dropped).
    ///
    /// Requires `MetalBackend::init()` to have been called.
    #[must_use = "returns a Result that may contain an error"]
    pub fn load_tensor<const D: usize, T: TensorElement + bytemuck::Pod>(
        &self,
        name: &str,
        expected_dims: [usize; D],
        ctx: &MetalContext,
    ) -> Result<Tensor<D, T, MetalBackend>, WeightError> {
        let info = self.tensor_info(name)?;

        // Validate dtype matches the requested element type.
        let expected_dtype = T::dtype();
        if info.dtype != expected_dtype {
            return Err(WeightError::DtypeMismatch {
                name: name.to_string(),
                expected: expected_dtype,
                actual: info.dtype,
            });
        }

        // Validate shape rank and dimensions.
        if info.shape.len() != D || info.shape.as_slice() != expected_dims.as_slice() {
            return Err(WeightError::ShapeMismatch {
                name: name.to_string(),
                expected_rank: D,
                expected_dims: expected_dims.to_vec(),
                actual_dims: info.shape.clone(),
            });
        }

        // Read tensor bytes and create a data-owning Metal buffer.
        // Uses &[u8] directly — Metal buffers are untyped byte buffers.
        // The type T is enforced at readback time via MetalBuffer::contents::<T>().
        let tensor_bytes = self.tensor_data(name)?;
        let buffer = ctx
            .create_buffer(tensor_bytes)
            .map_err(WeightError::Metal)?;

        let numel = info.numel()?;
        let storage = MetalTensorStorage::new(Arc::new(buffer), numel);
        // Shape was validated above, so from_storage only fails on dimension
        // overflow which cannot happen with validated dims. Map to ShapeOverflow
        // as a conservative fallback.
        Tensor::from_storage(expected_dims, storage)
            .map_err(|_| WeightError::ShapeOverflow(expected_dims.to_vec()))
    }
}

#[cfg(test)]
#[path = "safetensors_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "safetensors_kani.rs"]
mod proofs;
