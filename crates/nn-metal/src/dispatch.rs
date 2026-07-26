// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal command buffer dispatch primitives.
//!
//! Two encoder types are provided:
//! - [`ComputeDispatch`] — owns one command buffer with one compute encoder.
//!   Suitable for single kernel launches.
//! - [`CommandBatch`] + [`BatchEncoder`] — one command buffer with multiple
//!   sequential compute encoders. Suitable for multi-kernel launches that
//!   share a single commit-and-wait.
//!
//! Both encoder types expose `set_buffer`, `set_bytes`, and `encode` with
//! identical semantics; the duplication is intentional to keep each type
//! self-contained without a trait abstraction overhead.

use std::cell::Cell;
use std::ffi::c_void;
use std::time::{Duration, Instant};

use metal::MTLSize;
use objc::rc::autoreleasepool;

use crate::buffer::MetalBuffer;
use crate::error::MetalError;
use crate::pipeline::ComputePipeline;

#[path = "dispatch_pending.rs"]
mod pending;
pub use pending::PendingBatch;

/// Maximum time to wait for a GPU command buffer to complete before returning
/// an error. Prevents a hung GPU shader or wedged Metal driver from blocking
/// indefinitely and triggering a macOS watchdog kernel panic.
///
/// Set to 60 seconds — well under the macOS hardware watchdog threshold (~90s).
/// Override via `NN_GPU_TIMEOUT_SECS` environment variable for debugging.
// Keep `from_secs(60)` literally: a source-text invariant test
// (`test_gpu_timeout_under_watchdog`) pins this spelling to document the
// 60s-under-90s-watchdog contract. `from_mins(1)` is the same value but breaks it.
#[allow(clippy::duration_suboptimal_units)]
pub(crate) const GPU_TIMEOUT: Duration = Duration::from_secs(60);

/// Poll a committed command buffer for completion with a timeout.
///
/// Does NOT use `wait_until_completed()` — that call blocks indefinitely if
/// the GPU hangs, which caused the macOS watchdog kernel panic. Instead, polls
/// `status()` with exponential backoff (100µs → 10ms) and returns
/// `Err(MetalError::GpuTimeout)` if the buffer does not reach `Completed` or
/// `Error` status within the timeout.
///
/// CPU overhead from polling is negligible: most GPU work completes in < 1ms
/// (1-10 polls at 100µs). For long-running kernels (e.g., LSTM sequences),
/// the backoff caps at 10ms per poll.
fn wait_with_timeout(
    command_buffer: &metal::CommandBuffer,
    timeout: Duration,
) -> Result<(), MetalError> {
    let deadline = Instant::now() + timeout;
    let mut sleep_us = 100u64;
    loop {
        let status = command_buffer.status();
        match status {
            metal::MTLCommandBufferStatus::Completed => return Ok(()),
            metal::MTLCommandBufferStatus::Error => {
                return Err(MetalError::DispatchFailed(format!("{status:?}")));
            }
            _ => {
                if Instant::now() >= deadline {
                    return Err(MetalError::GpuTimeout(timeout));
                }
                std::thread::sleep(Duration::from_micros(sleep_us));
                sleep_us = (sleep_us * 2).min(10_000); // cap at 10ms
            }
        }
    }
}

fn to_mtl_size(size: [u32; 3]) -> MTLSize {
    MTLSize::new(u64::from(size[0]), u64::from(size[1]), u64::from(size[2]))
}

/// Validate that grid and threadgroup dimensions are non-zero.
///
/// Zero-size dimensions in Metal dispatch cause undefined behavior or GPU
/// hangs. This validation runs before every dispatch encode to catch malformed
/// tensor shapes at the FFI boundary.
fn validate_grid_dimensions(grid: [u32; 3], threads: [u32; 3]) -> Result<(), MetalError> {
    const GRID_DIM_NAMES: [&str; 3] = ["grid width", "grid height", "grid depth"];
    const THREAD_DIM_NAMES: [&str; 3] = [
        "threadgroup width",
        "threadgroup height",
        "threadgroup depth",
    ];
    for (i, &dim) in grid.iter().enumerate() {
        if dim == 0 {
            return Err(MetalError::InvalidGridDimension {
                dimension: GRID_DIM_NAMES[i],
                value: dim,
            });
        }
    }
    for (i, &dim) in threads.iter().enumerate() {
        if dim == 0 {
            return Err(MetalError::InvalidGridDimension {
                dimension: THREAD_DIM_NAMES[i],
                value: dim,
            });
        }
    }
    Ok(())
}

/// One compute dispatch encoded into one command buffer.
///
/// Implements [`Drop`] to call `end_encoding` on the encoder if the dispatch
/// is dropped without [`Self::commit_and_wait`] being called (e.g., on error
/// paths). This prevents undefined Metal behavior from uncommitted encoders.
/// (#647)
#[derive(Debug)]
#[non_exhaustive]
pub struct ComputeDispatch {
    command_buffer: metal::CommandBuffer,
    encoder: metal::ComputeCommandEncoder,
    ended: Cell<bool>,
}

impl ComputeDispatch {
    pub(crate) fn from_raw(
        command_buffer: metal::CommandBuffer,
        encoder: metal::ComputeCommandEncoder,
    ) -> Self {
        Self {
            command_buffer,
            encoder,
            ended: Cell::new(false),
        }
    }

    /// Bind a Metal buffer at the given argument index.
    pub fn set_buffer(&self, index: usize, buffer: &MetalBuffer) {
        self.encoder
            .set_buffer(index as u64, Some(buffer.inner()), 0);
    }

    /// Bind a Metal buffer at the given argument index with a byte offset.
    ///
    /// The kernel sees data starting at `byte_offset` bytes into the buffer.
    /// Used for zero-copy GPU tensor views (#1945).
    ///
    /// **Note:** This method does not validate that `byte_offset` is within
    /// the buffer bounds. Callers should use
    /// [`crate::buffer::validate_buffer_offset`] before calling this method,
    /// or use [`set_buffer_with_offset_checked`](Self::set_buffer_with_offset_checked)
    /// for automatic validation. Part of #4321.
    pub fn set_buffer_with_offset(&self, index: usize, buffer: &MetalBuffer, byte_offset: usize) {
        self.encoder
            .set_buffer(index as u64, Some(buffer.inner()), byte_offset as u64);
    }

    /// Bind a Metal buffer with a byte offset, validating bounds first.
    ///
    /// Returns `MetalError::BufferOffsetOutOfBounds` if `byte_offset` exceeds
    /// the buffer's byte length. Prefer this over the unchecked variant when
    /// the offset comes from external or computed sources. Part of #4321.
    #[must_use = "returns a Result that may contain an error"]
    pub fn set_buffer_with_offset_checked(
        &self,
        index: usize,
        buffer: &MetalBuffer,
        byte_offset: usize,
    ) -> Result<(), MetalError> {
        crate::buffer::validate_buffer_offset(buffer, byte_offset, "compute_dispatch_input")?;
        self.set_buffer_with_offset(index, buffer, byte_offset);
        Ok(())
    }

    /// Bind an inline constant value at the given argument index.
    pub fn set_bytes<T: bytemuck::NoUninit>(&self, index: usize, value: &T) {
        self.encoder.set_bytes(
            index as u64,
            size_of::<T>() as u64,
            std::ptr::from_ref::<T>(value).cast::<c_void>(),
        );
    }

    /// Allocate threadgroup shared memory at the given index.
    ///
    /// Used by reduction kernels that need scratch space for partial sums.
    pub fn set_threadgroup_memory_length(&self, index: usize, bytes: u64) {
        self.encoder
            .set_threadgroup_memory_length(index as u64, bytes);
    }

    /// Encode with `dispatch_threads` (Metal auto-computes threadgroup count).
    #[must_use = "returns a Result that may contain an error"]
    pub fn encode(
        &self,
        pipeline: &ComputePipeline,
        grid_size: [u32; 3],
        threadgroup_size: [u32; 3],
    ) -> Result<(), MetalError> {
        validate_grid_dimensions(grid_size, threadgroup_size)?;
        self.encoder.set_compute_pipeline_state(pipeline.inner());
        self.encoder
            .dispatch_threads(to_mtl_size(grid_size), to_mtl_size(threadgroup_size));
        Ok(())
    }

    /// Encode with `dispatch_thread_groups` (caller specifies threadgroup count).
    ///
    /// Unlike [`encode`], `threadgroups` is the number of threadgroups to launch,
    /// not the total thread count. Used by reduction kernels where each
    /// threadgroup processes one row/channel.
    #[must_use = "returns a Result that may contain an error"]
    pub fn encode_threadgroups(
        &self,
        pipeline: &ComputePipeline,
        threadgroups: [u32; 3],
        threads_per_group: [u32; 3],
    ) -> Result<(), MetalError> {
        validate_grid_dimensions(threadgroups, threads_per_group)?;
        self.encoder.set_compute_pipeline_state(pipeline.inner());
        self.encoder
            .dispatch_thread_groups(to_mtl_size(threadgroups), to_mtl_size(threads_per_group));
        Ok(())
    }

    /// End encoding, commit the command buffer, and block until completion.
    ///
    /// Uses [`wait_with_timeout`] to prevent indefinite blocking from a hung
    /// GPU shader or wedged Metal driver. Without a timeout, a single hung
    /// command buffer can freeze the system until the macOS hardware watchdog
    /// fires a kernel panic.
    ///
    /// Wrapped in `autoreleasepool` because `commit`, `waitUntilCompleted`,
    /// and `status` ObjC messages may create autoreleased temporary objects
    /// inside Metal's runtime. Without the pool, these accumulate on
    /// background threads (dvoice#1245).
    #[must_use = "returns a Result that may contain an error"]
    pub fn commit_and_wait(self) -> Result<(), MetalError> {
        self.ended.set(true);
        self.encoder.end_encoding();
        autoreleasepool(|| {
            self.command_buffer.commit();
            wait_with_timeout(&self.command_buffer, GPU_TIMEOUT)
        })
    }
}

impl Drop for ComputeDispatch {
    fn drop(&mut self) {
        if !self.ended.get() {
            autoreleasepool(|| {
                self.encoder.end_encoding();
            });
        }
    }
}

/// Compute encoder owned by a [`CommandBatch`].
///
/// Implements [`Drop`] to call `end_encoding` if the encoder is dropped
/// without [`Self::end_encoding`] being called (e.g., on error paths). (#647)
#[derive(Debug)]
#[non_exhaustive]
pub struct BatchEncoder {
    encoder: metal::ComputeCommandEncoder,
    ended: Cell<bool>,
}

impl BatchEncoder {
    pub(crate) fn from_raw(encoder: metal::ComputeCommandEncoder) -> Self {
        Self {
            encoder,
            ended: Cell::new(false),
        }
    }

    /// Access the underlying `ComputeCommandEncoder` for ICB resource usage
    /// declarations. Part of #3259.
    pub(crate) fn raw_encoder(&self) -> &metal::ComputeCommandEncoder {
        &self.encoder
    }

    /// Bind a Metal buffer at the given argument index.
    pub fn set_buffer(&self, index: usize, buffer: &MetalBuffer) {
        self.encoder
            .set_buffer(index as u64, Some(buffer.inner()), 0);
    }

    /// Bind a Metal buffer at the given argument index with a byte offset.
    ///
    /// The kernel sees data starting at `byte_offset` bytes into the buffer.
    /// Used for zero-copy GPU tensor views (#1945).
    ///
    /// # Safety note
    ///
    /// The caller must ensure `byte_offset <= buffer.len()`. For offsets from
    /// external or computed sources, prefer
    /// [`set_buffer_with_offset_checked`](Self::set_buffer_with_offset_checked).
    pub fn set_buffer_with_offset(&self, index: usize, buffer: &MetalBuffer, byte_offset: usize) {
        self.encoder
            .set_buffer(index as u64, Some(buffer.inner()), byte_offset as u64);
    }

    /// Bind a Metal buffer with a byte offset, validating bounds first.
    ///
    /// Returns `MetalError::BufferOffsetOutOfBounds` if `byte_offset` exceeds
    /// the buffer's byte length. Prefer this over the unchecked variant when
    /// the offset comes from external or computed sources. Part of #4321.
    #[must_use = "returns a Result that may contain an error"]
    pub fn set_buffer_with_offset_checked(
        &self,
        index: usize,
        buffer: &MetalBuffer,
        byte_offset: usize,
    ) -> Result<(), MetalError> {
        crate::buffer::validate_buffer_offset(buffer, byte_offset, "batch_encoder_input")?;
        self.set_buffer_with_offset(index, buffer, byte_offset);
        Ok(())
    }

    /// Bind an inline constant value at the given argument index.
    pub fn set_bytes<T: bytemuck::NoUninit>(&self, index: usize, value: &T) {
        self.encoder.set_bytes(
            index as u64,
            size_of::<T>() as u64,
            std::ptr::from_ref::<T>(value).cast::<c_void>(),
        );
    }

    /// Bind raw constant bytes at the given argument index.
    ///
    /// Unlike [`set_bytes`](Self::set_bytes) which takes a typed `NoUninit`
    /// reference, this accepts a pre-serialized byte slice. Used by
    /// `NativeEncoding` constant bindings where the bytes are already
    /// encoded via `bytemuck::bytes_of`.
    pub fn set_bytes_raw(&self, index: usize, raw_bytes: &[u8]) {
        self.encoder.set_bytes(
            index as u64,
            raw_bytes.len() as u64,
            raw_bytes.as_ptr().cast::<c_void>(),
        );
    }

    /// Allocate threadgroup shared memory at the given index.
    ///
    /// Used by reduction kernels that need scratch space for partial sums.
    pub fn set_threadgroup_memory_length(&self, index: usize, bytes: u64) {
        self.encoder
            .set_threadgroup_memory_length(index as u64, bytes);
    }

    /// Encode with `dispatch_threads` (Metal auto-computes threadgroup count).
    #[must_use = "returns a Result that may contain an error"]
    pub fn encode(
        &self,
        pipeline: &ComputePipeline,
        grid_size: [u32; 3],
        threadgroup_size: [u32; 3],
    ) -> Result<(), MetalError> {
        validate_grid_dimensions(grid_size, threadgroup_size)?;
        self.encoder.set_compute_pipeline_state(pipeline.inner());
        self.encoder
            .dispatch_threads(to_mtl_size(grid_size), to_mtl_size(threadgroup_size));
        Ok(())
    }

    /// Encode with `dispatch_thread_groups` (caller specifies threadgroup count).
    #[must_use = "returns a Result that may contain an error"]
    pub fn encode_threadgroups(
        &self,
        pipeline: &ComputePipeline,
        threadgroups: [u32; 3],
        threads_per_group: [u32; 3],
    ) -> Result<(), MetalError> {
        validate_grid_dimensions(threadgroups, threads_per_group)?;
        self.encoder.set_compute_pipeline_state(pipeline.inner());
        self.encoder
            .dispatch_thread_groups(to_mtl_size(threadgroups), to_mtl_size(threads_per_group));
        Ok(())
    }

    /// Finalize this encoder so the command batch can create another.
    pub fn end_encoding(self) {
        self.ended.set(true);
        self.encoder.end_encoding();
    }
}

impl Drop for BatchEncoder {
    fn drop(&mut self) {
        if !self.ended.get() {
            autoreleasepool(|| {
                self.encoder.end_encoding();
            });
        }
    }
}

/// Encodes multiple compute passes into one command buffer.
#[derive(Debug)]
#[non_exhaustive]
pub struct CommandBatch {
    command_buffer: metal::CommandBuffer,
}

impl CommandBatch {
    pub(crate) fn from_raw(command_buffer: metal::CommandBuffer) -> Self {
        Self { command_buffer }
    }

    /// Create a new compute encoder for the next kernel in this batch.
    ///
    /// Wrapped in `autoreleasepool` because `computeCommandEncoder` ObjC
    /// selector returns an autoreleased object (dvoice#1245).
    #[must_use = "returns a Result that may contain an error"]
    pub fn new_encoder(&self) -> Result<BatchEncoder, MetalError> {
        autoreleasepool(|| {
            // Guard: Metal returns nil from newComputeCommandEncoder when the
            // command buffer is in an error or already-committed state. Check
            // status before calling to prevent nil dereference (#420).
            let status = self.command_buffer.status();
            if status == metal::MTLCommandBufferStatus::Error
                || status == metal::MTLCommandBufferStatus::Completed
                || status == metal::MTLCommandBufferStatus::Committed
            {
                return Err(MetalError::EncoderCreate(format!("{status:?}")));
            }
            Ok(BatchEncoder::from_raw(
                self.command_buffer.new_compute_command_encoder().to_owned(),
            ))
        })
    }

    /// Copy bytes between Metal buffers using a blit encoder within this batch.
    ///
    /// The copy is GPU-side and executes in command buffer order, so it can
    /// safely read from buffers written by earlier compute encoders in the
    /// same batch (the Metal command buffer serializes encoder execution).
    ///
    /// Used by packed Stack/Concat dispatch (#1649) to assemble individual
    /// input buffers into a single contiguous packed buffer on GPU without
    /// CPU readback.
    ///
    /// Wrapped in `autoreleasepool` because `blitCommandEncoder` ObjC
    /// selector returns an autoreleased object (dvoice#1245).
    #[must_use = "returns a Result that may contain an error"]
    pub fn blit_copy(
        &self,
        src: &MetalBuffer,
        src_offset: usize,
        dst: &MetalBuffer,
        dst_offset: usize,
        size: usize,
    ) -> Result<(), MetalError> {
        if src_offset
            .checked_add(size)
            .map_or(true, |end| end > src.len())
        {
            return Err(MetalError::BufferBoundsExceeded {
                buffer_len: src.len(),
                offset: src_offset,
                size,
                role: "source",
            });
        }
        if dst_offset
            .checked_add(size)
            .map_or(true, |end| end > dst.len())
        {
            return Err(MetalError::BufferBoundsExceeded {
                buffer_len: dst.len(),
                offset: dst_offset,
                size,
                role: "destination",
            });
        }
        autoreleasepool(|| {
            let status = self.command_buffer.status();
            if status == metal::MTLCommandBufferStatus::Error
                || status == metal::MTLCommandBufferStatus::Completed
                || status == metal::MTLCommandBufferStatus::Committed
            {
                return Err(MetalError::EncoderCreate(format!("{status:?}")));
            }
            let blit = self.command_buffer.new_blit_command_encoder();
            blit.copy_from_buffer(
                src.inner(),
                src_offset as u64,
                dst.inner(),
                dst_offset as u64,
                size as u64,
            );
            blit.end_encoding();
            Ok(())
        })
    }

    /// Copy multiple buffer regions using a SINGLE blit encoder.
    ///
    /// Each `blit_copy` creates and destroys a blit encoder (ObjC
    /// `new_blit_command_encoder` + `end_encoding`). For N sequential
    /// relocations this incurs N ObjC method calls. `blit_copy_batch`
    /// amortizes encoder creation/destruction across all copies.
    ///
    /// Each element of `copies` is `(src, src_offset, dst, dst_offset, size)`.
    /// All bounds are validated before encoding begins; on any bounds error
    /// the entire batch is rejected.
    ///
    /// No-op (returns `Ok(())`) when `copies` is empty.
    ///
    /// Part of #4264 (R4: blit encoder batching).
    #[must_use = "returns a Result that may contain an error"]
    pub fn blit_copy_batch(
        &self,
        copies: &[(&MetalBuffer, usize, &MetalBuffer, usize, usize)],
    ) -> Result<(), MetalError> {
        if copies.is_empty() {
            return Ok(());
        }
        // Pre-validate all copies before touching the encoder.
        for &(src, src_offset, dst, dst_offset, size) in copies {
            if src_offset
                .checked_add(size)
                .map_or(true, |end| end > src.len())
            {
                return Err(MetalError::BufferBoundsExceeded {
                    buffer_len: src.len(),
                    offset: src_offset,
                    size,
                    role: "batch source",
                });
            }
            if dst_offset
                .checked_add(size)
                .map_or(true, |end| end > dst.len())
            {
                return Err(MetalError::BufferBoundsExceeded {
                    buffer_len: dst.len(),
                    offset: dst_offset,
                    size,
                    role: "batch destination",
                });
            }
        }
        autoreleasepool(|| {
            let status = self.command_buffer.status();
            if status == metal::MTLCommandBufferStatus::Error
                || status == metal::MTLCommandBufferStatus::Completed
                || status == metal::MTLCommandBufferStatus::Committed
            {
                return Err(MetalError::EncoderCreate(format!("{status:?}")));
            }
            let blit = self.command_buffer.new_blit_command_encoder();
            for &(src, src_offset, dst, dst_offset, size) in copies {
                blit.copy_from_buffer(
                    src.inner(),
                    src_offset as u64,
                    dst.inner(),
                    dst_offset as u64,
                    size as u64,
                );
            }
            blit.end_encoding();
            Ok(())
        })
    }

    /// Fill a buffer range with a single byte value using a blit encoder.
    ///
    /// Used to zero-initialize atomic counter buffers before reduction
    /// dispatches (#1815 Tier 2 conv-stats fusion).
    #[must_use = "returns a Result that may contain an error"]
    pub fn blit_fill(
        &self,
        dst: &MetalBuffer,
        dst_offset: usize,
        size: usize,
        value: u8,
    ) -> Result<(), MetalError> {
        if dst_offset
            .checked_add(size)
            .map_or(true, |end| end > dst.len())
        {
            return Err(MetalError::BufferBoundsExceeded {
                buffer_len: dst.len(),
                offset: dst_offset,
                size,
                role: "destination",
            });
        }
        autoreleasepool(|| {
            let status = self.command_buffer.status();
            if status == metal::MTLCommandBufferStatus::Error
                || status == metal::MTLCommandBufferStatus::Completed
                || status == metal::MTLCommandBufferStatus::Committed
            {
                return Err(MetalError::EncoderCreate(format!("{status:?}")));
            }
            let blit = self.command_buffer.new_blit_command_encoder();
            blit.fill_buffer(
                dst.inner(),
                metal::NSRange::new(dst_offset as u64, size as u64),
                value,
            );
            blit.end_encoding();
            Ok(())
        })
    }

    /// Commit the command buffer and block until all encoders complete.
    ///
    /// Uses [`wait_with_timeout`] to prevent indefinite blocking. See
    /// [`ComputeDispatch::commit_and_wait`] for rationale.
    ///
    /// Wrapped in `autoreleasepool` because `commit`, `waitUntilCompleted`,
    /// and `status` ObjC messages may create autoreleased temporary objects
    /// inside Metal's runtime. Without the pool, these accumulate on
    /// background threads (dvoice#1245).
    #[must_use = "returns a Result that may contain an error"]
    pub fn commit_and_wait(self) -> Result<(), MetalError> {
        autoreleasepool(|| {
            self.command_buffer.commit();
            wait_with_timeout(&self.command_buffer, GPU_TIMEOUT)
        })
    }

    /// Commit the command buffer to GPU without waiting for completion.
    ///
    /// Returns a [`PendingBatch`] handle. The GPU starts executing immediately.
    /// CPU can continue encoding a new batch or doing other work. Call
    /// [`PendingBatch::wait`] before reading any buffer written by this batch.
    ///
    /// This is the Metal equivalent of PyTorch MPS `COMMIT_AND_CONTINUE` —
    /// submit work to GPU and immediately resume CPU encoding (#2375).
    #[must_use = "store the PendingBatch and call wait() before CPU readback"]
    pub fn commit_no_wait(self) -> PendingBatch {
        autoreleasepool(|| {
            self.command_buffer.commit();
        });
        PendingBatch {
            command_buffer: self.command_buffer,
        }
    }

    /// Submit the command buffer and return a [`GpuFuture`] for async
    /// completion tracking.
    ///
    /// Like [`commit_no_wait`](Self::commit_no_wait) but returns a
    /// [`GpuFuture`] instead of a [`PendingBatch`]. The `GpuFuture` adds
    /// callback-based notification via Metal's `addCompletedHandler` in
    /// addition to polling and blocking wait.
    ///
    /// Registers the Metal `addCompletedHandler` BEFORE `commit()` (Metal
    /// requires handler registration before commit). The handler sets an
    /// internal completion flag and optionally invokes a user callback.
    ///
    /// This is the recommended non-blocking submit path for new code that
    /// needs async GPU completion notification (#4106).
    #[must_use = "store the GpuFuture and call wait() or on_complete() before CPU readback"]
    pub fn submit_async(self) -> crate::gpu_future::GpuFuture {
        // Register handler BEFORE commit (Metal requirement).
        let state = crate::gpu_future::register_completion_handler(&self.command_buffer);
        let pending = self.commit_no_wait();
        crate::gpu_future::GpuFuture::from_pending_with_state(pending, state)
    }
}
