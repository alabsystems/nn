// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal device and command queue management.
//!
//! [`MetalContext`] holds the system default Metal device and its command
//! queue. It is the entry point for buffer allocation, MSL compilation,
//! and compute dispatch.

use std::ffi::c_void;
use std::ptr::NonNull;

use metal::{CompileOptions, MTLResourceOptions};
use objc::rc::autoreleasepool;

use crate::buffer::MetalBuffer;
use crate::dispatch::{CommandBatch, ComputeDispatch};
use crate::error::MetalError;
use crate::kernel_source::KernelSource;
use crate::pipeline::ComputePipeline;

/// Shared Metal device + queue used for kernel compilation and dispatch.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MetalContext {
    device: metal::Device,
    queue: metal::CommandQueue,
}

impl MetalContext {
    /// Create a new Metal context using the system default device.
    #[must_use = "creating a MetalContext is expensive; the result should be used"]
    pub fn new() -> Result<Self, MetalError> {
        let device = metal::Device::system_default().ok_or(MetalError::NoDevice)?;
        let queue = device.new_command_queue();
        Ok(Self { device, queue })
    }

    /// Create a shared-mode GPU buffer by copying typed input data.
    #[must_use = "returns a Result that may contain an error"]
    pub fn create_buffer<T: bytemuck::NoUninit>(
        &self,
        data: &[T],
    ) -> Result<MetalBuffer, MetalError> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let len = bytes.len();
        if len == 0 {
            return Err(MetalError::BufferCreate(0));
        }

        let buffer = self.device.new_buffer_with_data(
            bytes.as_ptr().cast(),
            len as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(MetalBuffer::from_raw(buffer, len))
    }

    /// Wrap externally-owned memory as a shared Metal buffer without copying.
    ///
    /// `ptr` must be page-aligned (4096 bytes on Apple Silicon) and `len`
    /// must be a multiple of the page size. Metal's `newBufferWithBytesNoCopy`
    /// requires both of these — misaligned inputs cause undefined behavior.
    /// These requirements are validated at runtime and return a typed error
    /// rather than proceeding into UB.
    ///
    /// # Safety
    /// The caller must ensure `ptr` stays valid and outlives the returned
    /// buffer. Metal will reference this memory directly.
    #[must_use = "returns a Result that may contain an error"]
    pub unsafe fn create_buffer_no_copy(
        &self,
        ptr: NonNull<c_void>,
        len: usize,
    ) -> Result<MetalBuffer, MetalError> {
        if len == 0 {
            return Err(MetalError::BufferCreate(0));
        }

        // Metal requires page-aligned pointer and page-multiple length.
        // Validate at runtime rather than debug_assert to prevent UB in
        // release builds (#522).
        const PAGE_SIZE: usize = 4096;
        let addr = ptr.as_ptr() as usize;
        if !addr.is_multiple_of(PAGE_SIZE) || !len.is_multiple_of(PAGE_SIZE) {
            return Err(MetalError::BufferAlignment {
                ptr: addr,
                len,
                page_size: PAGE_SIZE,
            });
        }

        let buffer = self.device.new_buffer_with_bytes_no_copy(
            ptr.as_ptr().cast_const(),
            len as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        Ok(MetalBuffer::from_raw(buffer, len))
    }

    /// Allocate a zero-initialized shared-mode GPU buffer.
    #[must_use = "returns a Result that may contain an error"]
    pub fn create_buffer_zeroed(&self, len: usize) -> Result<MetalBuffer, MetalError> {
        if len == 0 {
            return Err(MetalError::BufferCreate(0));
        }
        let buffer = self
            .device
            .new_buffer(len as u64, MTLResourceOptions::StorageModeShared);
        Ok(MetalBuffer::from_raw(buffer, len))
    }

    /// Create a new data-owning buffer by copying the contents of `src`.
    ///
    /// The returned buffer owns its own GPU-allocated memory, so it is safe
    /// to use after the source buffer (or its backing `WeightMap`) is dropped.
    /// This replaces the removed `derive(Clone)` on `MetalBuffer` (#598).
    ///
    /// **Caller must call `gpu_scope::flush()` before this function** if the
    /// source buffer may contain data from pending lazy-batch GPU dispatches.
    /// `contents()` does a CPU-side read that returns stale data without flush.
    #[must_use = "returns a Result that may contain an error"]
    pub fn clone_buffer(&self, src: &MetalBuffer) -> Result<MetalBuffer, MetalError> {
        let pending = crate::gpu_scope::pending_encoding_count();
        if pending != 0 {
            return Err(MetalError::PendingFlushRequired {
                pending_count: pending,
            });
        }
        let bytes: &[u8] = src.contents()?;
        if bytes.is_empty() {
            return Err(MetalError::BufferCreate(0));
        }
        let buffer = self.device.new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(MetalBuffer::from_raw(buffer, bytes.len()))
    }

    /// Create a new data-owning buffer by copying a byte range from `src`.
    ///
    /// Copies `len_bytes` starting at `byte_offset` in the source buffer.
    /// Used for cloning narrow-view tensors where `byte_offset > 0` means the
    /// logical data starts partway into the underlying Metal buffer (#1964).
    ///
    /// **Caller must call `gpu_scope::flush()` before this function** if the
    /// source buffer may contain data from pending lazy-batch GPU dispatches.
    /// `contents()` does a CPU-side read that returns stale data without flush.
    #[must_use = "returns a Result that may contain an error"]
    pub fn clone_buffer_range(
        &self,
        src: &MetalBuffer,
        byte_offset: usize,
        len_bytes: usize,
    ) -> Result<MetalBuffer, MetalError> {
        let pending = crate::gpu_scope::pending_encoding_count();
        if pending != 0 {
            return Err(MetalError::PendingFlushRequired {
                pending_count: pending,
            });
        }
        if len_bytes == 0 {
            return Err(MetalError::BufferCreate(0));
        }
        let bytes: &[u8] = src.contents()?;
        let end = byte_offset
            .checked_add(len_bytes)
            .ok_or(MetalError::BufferCreate(len_bytes))?;
        if end > bytes.len() {
            return Err(MetalError::BufferCreate(len_bytes));
        }
        let slice = &bytes[byte_offset..end];
        let buffer = self.device.new_buffer_with_data(
            slice.as_ptr().cast(),
            len_bytes as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(MetalBuffer::from_raw(buffer, len_bytes))
    }

    /// Returns a reference to the Metal device.
    #[must_use]
    pub fn device(&self) -> &metal::DeviceRef {
        &self.device
    }

    /// Returns a reference to the command queue.
    #[must_use]
    pub fn queue(&self) -> &metal::CommandQueueRef {
        &self.queue
    }

    /// Compile MSL source into a reusable compute pipeline.
    ///
    /// When `source` has non-empty [`KernelSource::function_constants`],
    /// the pipeline is specialized via `MTLFunctionConstantValues` (#3449).
    /// This enables the Metal compiler to unroll loops and eliminate dead
    /// code for compile-time-known parameter values.
    ///
    /// Wrapped in `autoreleasepool` because `metal-rs` internally creates
    /// autoreleased `NSString` temporaries when passing MSL source and
    /// function names to the Metal API. Without the pool, these leak on
    /// background threads (dvoice#1245).
    #[must_use = "returns a Result that may contain an error"]
    pub fn compile_pipeline(&self, source: &KernelSource) -> Result<ComputePipeline, MetalError> {
        autoreleasepool(|| {
            let options = CompileOptions::new();
            options.set_fast_math_enabled(source.fast_math());

            let library = self
                .device
                .new_library_with_source(source.msl_source(), &options)
                .map_err(MetalError::LibraryCompile)?;

            let fcv = Self::build_function_constants(source);
            let function = library
                .get_function(source.entry_point(), fcv)
                .map_err(|_| MetalError::MissingEntryPoint(source.entry_point().to_owned()))?;
            let pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(MetalError::PipelineCreate)?;

            Ok(ComputePipeline::from_raw(
                pipeline,
                source.entry_point(),
                source.fast_math(),
            ))
        })
    }

    /// Build `FunctionConstantValues` from the source's constant list.
    /// Returns `None` when the source has no function constants (common path).
    fn build_function_constants(
        source: &KernelSource,
    ) -> Option<metal::FunctionConstantValues> {
        let constants = source.function_constants();
        if constants.is_empty() {
            return None;
        }
        let fcv = metal::FunctionConstantValues::new();
        for &(index, value) in constants {
            fcv.set_constant_value_at_index(
                (&raw const value).cast::<c_void>(),
                metal::MTLDataType::UInt,
                u64::from(index),
            );
        }
        Some(fcv)
    }

    /// Compile MSL into an ICB-compatible pipeline (#3259 D3).
    #[must_use = "returns a Result that may contain an error"]
    pub fn compile_pipeline_icb(
        &self,
        source: &KernelSource,
    ) -> Result<ComputePipeline, MetalError> {
        autoreleasepool(|| {
            let options = CompileOptions::new();
            options.set_fast_math_enabled(source.fast_math());
            let library = self
                .device
                .new_library_with_source(source.msl_source(), &options)
                .map_err(MetalError::LibraryCompile)?;
            let function = library
                .get_function(source.entry_point(), None)
                .map_err(|_| MetalError::MissingEntryPoint(source.entry_point().to_owned()))?;
            let descriptor = metal::ComputePipelineDescriptor::new();
            descriptor.set_compute_function(Some(&function));
            descriptor.set_support_indirect_command_buffers(true);
            let pipeline = self
                .device
                .new_compute_pipeline_state(&descriptor)
                .map_err(MetalError::PipelineCreate)?;
            Ok(ComputePipeline::from_raw(
                pipeline,
                source.entry_point(),
                source.fast_math(),
            ))
        })
    }

    /// Start one dispatch pass in a fresh command buffer.
    ///
    /// Wrapped in `autoreleasepool` because `commandBuffer` and
    /// `computeCommandEncoder` ObjC selectors return autoreleased objects.
    /// Without the pool, these leak on background threads (dvoice#1245).
    #[must_use = "returns a Result that may contain an error"]
    pub fn create_dispatch(&self) -> Result<ComputeDispatch, MetalError> {
        autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer().to_owned();
            // Guard: check command buffer status before encoder creation (#420).
            let status = command_buffer.status();
            if status == metal::MTLCommandBufferStatus::Error {
                return Err(MetalError::EncoderCreate(format!("{status:?}")));
            }
            let encoder = command_buffer.new_compute_command_encoder().to_owned();
            Ok(ComputeDispatch::from_raw(command_buffer, encoder))
        })
    }

    /// Start a batch command buffer for multiple compute passes.
    ///
    /// Wrapped in `autoreleasepool` because `commandBuffer` ObjC selector
    /// returns an autoreleased object (dvoice#1245).
    #[must_use = "returns a Result that may contain an error"]
    pub fn begin_batch(&self) -> Result<CommandBatch, MetalError> {
        autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer().to_owned();
            // Guard: check command buffer status — mirroring create_dispatch (#420).
            // A command buffer created from a queue under GPU memory pressure or
            // after a GPU reset may start in Error state.
            let status = command_buffer.status();
            if status == metal::MTLCommandBufferStatus::Error {
                return Err(MetalError::EncoderCreate(format!(
                    "begin_batch: command buffer in error state: {status:?}"
                )));
            }
            Ok(CommandBatch::from_raw(command_buffer))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RAII guard for page-aligned allocations in tests.
    ///
    /// Ensures `dealloc` runs even if the test panics between `alloc` and
    /// the end of the test body. Without this, a failing assertion would
    /// leak the allocation. Mirrors `PageAlignedAlloc` in
    /// `tests/dispatch_cache.rs`.
    struct AllocGuard {
        ptr: *mut u8,
        layout: std::alloc::Layout,
    }

    impl Drop for AllocGuard {
        fn drop(&mut self) {
            // SAFETY: `ptr` was allocated with `std::alloc::alloc` using
            // `self.layout`, and this Drop impl is the only deallocation path.
            unsafe { std::alloc::dealloc(self.ptr, self.layout) };
        }
    }

    /// Verify that misaligned pointer is rejected (#522 AC1).
    #[test]
    fn test_create_buffer_no_copy_rejects_misaligned_ptr() {
        let ctx = MetalContext::new().expect("Metal device");
        // Intentionally misaligned: page-aligned base + 1 byte offset.
        let layout = std::alloc::Layout::from_size_align(8192, 4096).unwrap();
        // SAFETY: Layout is valid (non-zero size, power-of-two alignment).
        // AllocGuard below ensures deallocation on all exit paths.
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null());
        let _guard = AllocGuard { ptr, layout };
        // SAFETY: ptr is non-null (asserted above) and 8192-byte allocation
        // makes ptr.add(1) valid (within the allocation).
        let misaligned = unsafe { ptr.add(1) };
        let nn = NonNull::new(misaligned.cast::<c_void>()).unwrap();
        let result = unsafe { ctx.create_buffer_no_copy(nn, 4096) };
        assert!(
            matches!(result, Err(MetalError::BufferAlignment { .. })),
            "misaligned pointer should be rejected, got: {result:?}"
        );
    }

    /// Verify that non-page-multiple length is rejected (#522 AC1).
    #[test]
    fn test_create_buffer_no_copy_rejects_non_page_len() {
        let ctx = MetalContext::new().expect("Metal device");
        let layout = std::alloc::Layout::from_size_align(4096, 4096).unwrap();
        // SAFETY: Layout is valid (non-zero size, power-of-two alignment).
        // AllocGuard below ensures deallocation on all exit paths.
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null());
        let _guard = AllocGuard { ptr, layout };
        let nn = NonNull::new(ptr.cast::<c_void>()).unwrap();
        // Length 100 is not a page multiple.
        let result = unsafe { ctx.create_buffer_no_copy(nn, 100) };
        assert!(
            matches!(result, Err(MetalError::BufferAlignment { .. })),
            "non-page-multiple length should be rejected, got: {result:?}"
        );
    }
}
