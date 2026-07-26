// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-dispatch command batch for Vulkan compute pipelines.
//!
//! [`CommandBatch`] records multiple compute dispatches into a single command
//! buffer, separated by memory barriers. This mirrors the Metal backend's
//! `CommandBatch` + `BatchEncoder` pattern where multiple kernel launches
//! share a single commit-and-wait for throughput.
//!
//! # Lifecycle
//!
//! 1. Create via [`CommandBatch::new`].
//! 2. Record dispatches via [`record`](CommandBatch::record).
//! 3. Insert explicit barriers via [`barrier`](CommandBatch::barrier).
//! 4. Submit and wait via [`submit_and_wait`](CommandBatch::submit_and_wait).
//!
//! # Memory barriers
//!
//! Vulkan requires explicit memory barriers between dispatches that have
//! read-after-write or write-after-write dependencies. Unlike Metal (which
//! implicitly orders within an encoder), Vulkan dispatch order within a
//! command buffer provides no execution ordering guarantee without barriers.
//!
//! [`CommandBatch`] provides two barrier strategies:
//! - **Auto-barrier:** Insert a compute pipeline barrier after every dispatch
//!   (safe default, slight overhead on independent dispatches).
//! - **Manual barrier:** Caller inserts barriers explicitly via [`barrier`].

use crate::buffer::VulkanBuffer;
use crate::dispatch::ComputePipeline;
use crate::error::VulkanError;

/// Strategy for inserting memory barriers between dispatches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierStrategy {
    /// Insert a compute-to-compute memory barrier after every dispatch.
    /// Safe default -- ensures all dispatches see prior writes.
    Auto,
    /// No automatic barriers. Caller must call [`CommandBatch::barrier`]
    /// between dependent dispatches. Better throughput when dispatches
    /// are independent.
    Manual,
}

impl Default for BarrierStrategy {
    /// Auto is the safe default: barriers inserted after every dispatch.
    fn default() -> Self {
        Self::Auto
    }
}

/// A recorded dispatch within a command batch.
#[derive(Debug)]
struct RecordedDispatch {
    /// Pipeline entry point (for diagnostics).
    _entry_point: String,
    /// Number of buffers bound.
    _buffer_count: usize,
    /// Workgroup dispatch count `[x, y, z]`.
    _group_count: [u32; 3],
    /// Whether a barrier was inserted after this dispatch.
    barrier_after: bool,
}

/// Multi-dispatch command batch for Vulkan compute pipelines.
///
/// Records multiple dispatches into a single command buffer. Mirrors
/// the Metal backend's `CommandBatch` pattern.
///
/// # Example (conceptual)
///
/// ```no_run
/// use nn_vulkan::command_batch::{CommandBatch, BarrierStrategy};
/// use nn_vulkan::buffer::{VulkanBuffer, BufferUsage};
/// use nn_vulkan::dispatch::ComputePipeline;
///
/// let mut batch = CommandBatch::new(BarrierStrategy::Auto);
/// // batch.record(&pipeline, &[&buf_a, &buf_b], &push, [64, 1, 1]).unwrap();
/// // batch.record(&pipeline2, &[&buf_b, &buf_c], &push2, [32, 1, 1]).unwrap();
/// // batch.submit_and_wait().unwrap();
/// ```
#[derive(Debug)]
pub struct CommandBatch {
    /// Barrier strategy.
    strategy: BarrierStrategy,
    /// Recorded dispatches.
    dispatches: Vec<RecordedDispatch>,
    /// Number of explicit barriers inserted.
    barrier_count: u32,
    /// Opaque handle (placeholder for VkCommandBuffer).
    _command_buffer_handle: u64,
}

impl CommandBatch {
    /// Create a new command batch with the given barrier strategy.
    #[must_use]
    pub fn new(strategy: BarrierStrategy) -> Self {
        Self {
            strategy,
            dispatches: Vec::new(),
            barrier_count: 0,
            _command_buffer_handle: 0,
        }
    }

    /// Record a compute dispatch into the batch.
    ///
    /// If `BarrierStrategy::Auto`, a compute-to-compute memory barrier is
    /// automatically inserted after the dispatch.
    ///
    /// # Arguments
    ///
    /// * `pipeline` -- Compiled compute pipeline.
    /// * `buffers` -- Buffers to bind as descriptor set entries (in binding order).
    /// * `push_constants` -- Push constant data (raw bytes).
    /// * `group_count` -- Number of workgroups to dispatch `[x, y, z]`.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::CommandBufferError`] if recording fails.
    pub fn record(
        &mut self,
        pipeline: &ComputePipeline,
        buffers: &[&VulkanBuffer],
        _push_constants: &[u8],
        group_count: [u32; 3],
    ) -> Result<(), VulkanError> {
        if group_count.contains(&0) {
            return Err(VulkanError::CommandBufferError {
                reason: "workgroup count must be > 0 in all dimensions".into(),
            });
        }

        let barrier_after = self.strategy == BarrierStrategy::Auto;

        self.dispatches.push(RecordedDispatch {
            _entry_point: pipeline.entry_point().to_owned(),
            _buffer_count: buffers.len(),
            _group_count: group_count,
            barrier_after,
        });

        if barrier_after {
            self.barrier_count += 1;
        }

        // Placeholder: real implementation records vkCmdBindPipeline,
        // vkCmdBindDescriptorSets, vkCmdPushConstants, vkCmdDispatch,
        // and optionally vkCmdPipelineBarrier.
        Ok(())
    }

    /// Insert an explicit compute-to-compute memory barrier.
    ///
    /// In `BarrierStrategy::Manual` mode, call this between dispatches
    /// that have data dependencies (read-after-write or write-after-write).
    ///
    /// In `BarrierStrategy::Auto` mode, this is a no-op (barriers are
    /// already inserted after every dispatch).
    pub fn barrier(&mut self) {
        if self.strategy == BarrierStrategy::Manual {
            self.barrier_count += 1;
            // Mark the last dispatch as having a barrier after it.
            if let Some(last) = self.dispatches.last_mut() {
                last.barrier_after = true;
            }
        }
        // In Auto mode, barriers are already inserted.
    }

    /// Submit the recorded command buffer and wait for GPU completion.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::CommandBufferError`] if the batch is empty
    /// or submission fails.
    pub fn submit_and_wait(&self) -> Result<(), VulkanError> {
        if self.dispatches.is_empty() {
            return Err(VulkanError::CommandBufferError {
                reason: "no dispatches recorded in batch".into(),
            });
        }

        // Placeholder: real implementation calls vkEndCommandBuffer,
        // vkQueueSubmit, vkQueueWaitIdle.
        Ok(())
    }

    /// Submit the recorded command buffer without waiting (non-blocking).
    ///
    /// Returns a [`PendingBatch`] handle that can be polled or waited on.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::CommandBufferError`] if the batch is empty
    /// or submission fails.
    pub fn submit_async(&self) -> Result<PendingBatch, VulkanError> {
        if self.dispatches.is_empty() {
            return Err(VulkanError::CommandBufferError {
                reason: "no dispatches recorded in batch".into(),
            });
        }

        // Placeholder: real implementation calls vkEndCommandBuffer,
        // vkQueueSubmit with a VkFence for polling.
        Ok(PendingBatch {
            dispatch_count: self.dispatch_count(),
            _fence_handle: 0,
        })
    }

    /// Number of dispatches recorded.
    #[must_use]
    pub fn dispatch_count(&self) -> u32 {
        self.dispatches.len() as u32
    }

    /// Number of memory barriers inserted.
    #[must_use]
    pub fn barrier_count(&self) -> u32 {
        self.barrier_count
    }

    /// The barrier strategy used by this batch.
    #[must_use]
    pub fn strategy(&self) -> BarrierStrategy {
        self.strategy
    }

    /// Check whether a specific dispatch has a barrier after it.
    #[must_use]
    pub fn has_barrier_after(&self, dispatch_index: usize) -> bool {
        self.dispatches
            .get(dispatch_index)
            .is_some_and(|d| d.barrier_after)
    }
}

/// Handle to a submitted command batch that may still be executing on the GPU.
///
/// Mirrors Metal's `PendingBatch` pattern for non-blocking GPU submission.
#[derive(Debug)]
pub struct PendingBatch {
    /// Number of dispatches in this batch.
    dispatch_count: u32,
    /// Opaque fence handle (placeholder for VkFence).
    _fence_handle: u64,
}

impl PendingBatch {
    /// Check whether the GPU has completed all dispatches.
    ///
    /// Non-blocking: returns immediately.
    #[must_use]
    pub fn is_completed(&self) -> bool {
        // Placeholder: real implementation checks vkGetFenceStatus.
        // Returns true in scaffold mode (no real GPU work to wait for).
        true
    }

    /// Block until the GPU completes all dispatches.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::CommandBufferError`] if the wait fails or times out.
    pub fn wait(&self) -> Result<(), VulkanError> {
        // Placeholder: real implementation calls vkWaitForFences.
        Ok(())
    }

    /// Number of dispatches in this batch.
    #[must_use]
    pub fn dispatch_count(&self) -> u32 {
        self.dispatch_count
    }
}

#[cfg(test)]
#[path = "command_batch_tests.rs"]
mod command_batch_tests;
