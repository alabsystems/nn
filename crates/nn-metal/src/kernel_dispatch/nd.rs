// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-dimensional dispatch methods for [`KernelPipeline`].
//!
//! Includes 2D/3D grid dispatch, per-slice reduction, and low-level buffer
//! dispatch with explicit binding roles.

use super::{checked_output_bytes, BufferAccess, BufferBinding, KernelPipeline};
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::dispatch::BatchEncoder;
use crate::dispatch_plan::{DispatchMode, DispatchPlan};
use crate::element::MetalElement;
use crate::error::MetalError;

/// Validate that every input slice has at least `min_elems` elements.
///
/// Returns `MetalError::InputLenMismatch` for the first too-short input.
fn validate_input_lengths<T>(inputs: &[&[T]], min_elems: usize) -> Result<(), MetalError> {
    for (index, input) in inputs.iter().enumerate() {
        if input.len() < min_elems {
            return Err(MetalError::InputLenMismatch {
                expected: min_elems,
                got: input.len(),
                index,
            });
        }
    }
    Ok(())
}

impl KernelPipeline {
    /// Dispatch a kernel with explicit parameter buffer roles and no dedicated
    /// output slot.
    ///
    /// Buffer binding layout:
    /// - `buffer(0)` … `buffer(N-1)`: parameter buffers (read-only/read-write/write-only)
    /// - `buffer(N)` …: plan constants (mode-dependent)
    ///
    /// This path supports in-place kernels where writable buffers are part of
    /// the parameter list.
    #[must_use = "returns a Result that may contain an error"]
    pub fn dispatch_bindings(
        &self,
        _ctx: &MetalContext,
        bindings: &[BufferBinding<'_>],
        plan: &DispatchPlan,
    ) -> Result<(), MetalError> {
        if bindings.len() != self.param_count {
            return Err(MetalError::ParamCountMismatch {
                expected: self.param_count,
                got: bindings.len(),
            });
        }
        if bindings
            .iter()
            .all(|binding| matches!(binding.access(), BufferAccess::ReadOnly))
        {
            return Err(MetalError::InvalidDispatchBindings(
                "at least one writable parameter role is required",
            ));
        }
        // Validate that no bound buffer is empty (#4321). An empty buffer
        // passed to a Metal kernel would cause out-of-bounds GPU access.
        for (index, binding) in bindings.iter().enumerate() {
            if binding.buffer().is_empty() {
                return Err(MetalError::InputLenMismatch {
                    expected: 1,
                    got: 0,
                    index,
                });
            }
        }

        // Lazy batch (#2009): encode into the thread-local lazy batch.
        crate::gpu_scope::get_or_create_batch()
            .map_err(|e| MetalError::DispatchFailed(e.to_string()))?;
        let scope_result =
            crate::gpu_scope::encode_into_lazy_batch(|batch| -> Result<(), MetalError> {
                let enc = batch.new_encoder()?;
                for (index, binding) in bindings.iter().enumerate() {
                    enc.set_buffer(index, binding.buffer());
                }
                for (i, constant) in plan.constants().iter().enumerate() {
                    enc.set_bytes(self.param_count + i, constant);
                }
                if let Some(bytes) = plan.threadgroup_memory_bytes() {
                    enc.set_threadgroup_memory_length(0, bytes);
                }
                if plan.use_threadgroups() {
                    enc.encode_threadgroups(&self.pipeline, plan.grid(), plan.threads())?;
                } else {
                    enc.encode(&self.pipeline, plan.grid(), plan.threads())?;
                }
                enc.end_encoding();
                Ok(())
            });
        match scope_result {
            Ok(inner) => inner,
            Err(e) => Err(MetalError::DispatchFailed(e.to_string())),
        }
    }

    /// Dispatch a kernel with pre-allocated Metal buffers and an explicit plan.
    ///
    /// This is the low-level entry point for non-element-wise dispatch modes
    /// (2D/3D grids, reduction) with a dedicated output buffer. Buffer binding
    /// layout:
    /// - `buffer(0)` … `buffer(N-1)`: input buffers
    /// - `buffer(N)`: output buffer
    /// - `buffer(N+1)` …: plan constants (mode-dependent)
    ///
    /// For kernels without dedicated output buffers, use
    /// [`dispatch_bindings`](Self::dispatch_bindings).
    #[must_use = "returns a Result that may contain an error"]
    pub fn dispatch_buffers(
        &self,
        ctx: &MetalContext,
        inputs: &[&MetalBuffer],
        output: &MetalBuffer,
        plan: &DispatchPlan,
    ) -> Result<(), MetalError> {
        self.dispatch_buffers_with_offsets(ctx, inputs, &[], output, plan)
    }

    /// Dispatch a kernel with pre-allocated Metal buffers, byte offsets, and an
    /// explicit plan.
    ///
    /// Like [`dispatch_buffers`](Self::dispatch_buffers), but each input buffer
    /// can have a non-zero byte offset. Metal's `setBuffer(_:offset:atIndex:)`
    /// shifts the base pointer the kernel sees, so `data[0]` in MSL corresponds
    /// to `buffer + byte_offset`. Offsets beyond the slice length default to 0.
    ///
    /// This is required for zero-copy GPU narrow views (#1945) where
    /// `MetalTensorData::byte_offset()` is non-zero.
    #[must_use = "returns a Result that may contain an error"]
    pub fn dispatch_buffers_with_offsets(
        &self,
        ctx: &MetalContext,
        inputs: &[&MetalBuffer],
        input_offsets: &[usize],
        output: &MetalBuffer,
        plan: &DispatchPlan,
    ) -> Result<(), MetalError> {
        self.dispatch_buffers_with_all_offsets(ctx, inputs, input_offsets, output, 0, plan)
    }

    /// Like [`dispatch_buffers_with_offsets`](Self::dispatch_buffers_with_offsets),
    /// but also accepts a non-zero byte offset for the output buffer.
    ///
    /// Required for arena sub-allocation where the output buffer is a view
    /// into a larger pre-allocated Metal buffer at a non-zero byte offset.
    #[must_use = "returns a Result that may contain an error"]
    pub fn dispatch_buffers_with_all_offsets(
        &self,
        _ctx: &MetalContext,
        inputs: &[&MetalBuffer],
        input_offsets: &[usize],
        output: &MetalBuffer,
        output_offset: usize,
        plan: &DispatchPlan,
    ) -> Result<(), MetalError> {
        if inputs.len() != self.param_count {
            return Err(MetalError::ParamCountMismatch {
                expected: self.param_count,
                got: inputs.len(),
            });
        }

        // Lazy batch (#2009): encode into the thread-local lazy batch.
        crate::gpu_scope::get_or_create_batch()
            .map_err(|e| MetalError::DispatchFailed(e.to_string()))?;
        let scope_result =
            crate::gpu_scope::encode_into_lazy_batch(|batch| -> Result<(), MetalError> {
                let enc = batch.new_encoder()?;
                self.encode_into(enc, inputs, input_offsets, output, output_offset, plan)
            });
        match scope_result {
            Ok(inner) => inner,
            Err(e) => Err(MetalError::DispatchFailed(e.to_string())),
        }
    }

    /// Encode a dispatch into an existing [`BatchEncoder`] without committing.
    ///
    /// Same buffer binding layout as [`dispatch_buffers`](Self::dispatch_buffers):
    /// - `buffer(0)` … `buffer(N-1)`: input buffers
    /// - `buffer(N)`: output buffer
    /// - `buffer(N+1)` …: plan constants
    ///
    /// The encoder is consumed (end_encoding is called). The caller is
    /// responsible for committing the parent [`CommandBatch`] after all steps
    /// are encoded.
    #[must_use = "returns a Result that may contain an error"]
    pub fn encode_into(
        &self,
        encoder: BatchEncoder,
        inputs: &[&MetalBuffer],
        input_offsets: &[usize],
        output: &MetalBuffer,
        output_offset: usize,
        plan: &DispatchPlan,
    ) -> Result<(), MetalError> {
        if inputs.len() != self.param_count {
            return Err(MetalError::ParamCountMismatch {
                expected: self.param_count,
                got: inputs.len(),
            });
        }

        // Validate buffer offsets before binding to Metal encoder (#4321).
        // Out-of-bounds offsets would cause Metal to read/write past the buffer
        // allocation, producing GPU memory corruption or undefined behavior.
        for (index, buffer) in inputs.iter().enumerate() {
            let offset = input_offsets.get(index).copied().unwrap_or(0);
            if offset > 0 {
                crate::buffer::validate_buffer_offset(buffer, offset, "input")?;
            }
        }
        if output_offset > 0 {
            crate::buffer::validate_buffer_offset(output, output_offset, "output")?;
        }

        for (index, buffer) in inputs.iter().enumerate() {
            let offset = input_offsets.get(index).copied().unwrap_or(0);
            if offset > 0 {
                encoder.set_buffer_with_offset(index, buffer, offset);
            } else {
                encoder.set_buffer(index, buffer);
            }
        }
        if output_offset > 0 {
            encoder.set_buffer_with_offset(self.param_count, output, output_offset);
        } else {
            encoder.set_buffer(self.param_count, output);
        }
        for (i, constant) in plan.constants().iter().enumerate() {
            encoder.set_bytes(self.param_count + 1 + i, constant);
        }

        if let Some(bytes) = plan.threadgroup_memory_bytes() {
            encoder.set_threadgroup_memory_length(0, bytes);
        }

        if plan.use_threadgroups() {
            encoder.encode_threadgroups(&self.pipeline, plan.grid(), plan.threads())?;
        } else {
            encoder.encode(&self.pipeline, plan.grid(), plan.threads())?;
        }

        encoder.end_encoding();
        Ok(())
    }

    /// Dispatch a kernel with a 2D grid and typed inputs.
    ///
    /// Each input slice is flattened into a single buffer. The output buffer has
    /// `grid[0] * grid[1]` elements. Constants `[grid_w, grid_h]` are bound
    /// after the output buffer.
    ///
    /// Supports `f32` and [`half::f16`] via the [`MetalElement`] trait.
    #[must_use = "returns a Result that may contain an error"]
    pub fn dispatch_2d<E: MetalElement>(
        &self,
        ctx: &MetalContext,
        inputs: &[&[E]],
        grid: [u32; 2],
        threads: [u32; 2],
    ) -> Result<Vec<E>, MetalError> {
        let plan = DispatchMode::Grid2D { grid, threads }.plan()?;
        validate_input_lengths(inputs, plan.output_elems())?;

        let mut buffers = Vec::with_capacity(inputs.len());
        for input in inputs {
            buffers.push(E::create_buffer(ctx, input)?);
        }
        let buf_refs: Vec<&MetalBuffer> = buffers.iter().collect();

        let out_bytes = checked_output_bytes(plan.output_elems(), E::element_size())?;
        let (out_buf, out_offset) = crate::arena::arena_alloc_or_create(ctx, out_bytes)?;

        self.dispatch_buffers_with_all_offsets(ctx, &buf_refs, &[], &out_buf, out_offset, &plan)?;
        E::read_buffer_at_offset(&out_buf, out_offset, plan.output_elems())
    }

    /// Dispatch a kernel with a 3D grid and typed inputs.
    ///
    /// Each input slice is flattened into a single buffer. The output buffer has
    /// `grid[0] * grid[1] * grid[2]` elements. Constants `[grid_x, grid_y, grid_z]`
    /// are bound after the output buffer.
    ///
    /// Supports `f32` and [`half::f16`] via the [`MetalElement`] trait.
    #[must_use = "returns a Result that may contain an error"]
    pub fn dispatch_3d<E: MetalElement>(
        &self,
        ctx: &MetalContext,
        inputs: &[&[E]],
        grid: [u32; 3],
        threads: [u32; 3],
    ) -> Result<Vec<E>, MetalError> {
        // Lazy batch (#2009): dispatch_buffers encodes into the lazy batch;
        // read_buffer calls flush() before CPU readback.
        let plan = DispatchMode::Grid3D { grid, threads }.plan()?;
        validate_input_lengths(inputs, plan.output_elems())?;

        let mut buffers = Vec::with_capacity(inputs.len());
        for input in inputs {
            buffers.push(E::create_buffer(ctx, input)?);
        }
        let buf_refs: Vec<&MetalBuffer> = buffers.iter().collect();

        let out_bytes = checked_output_bytes(plan.output_elems(), E::element_size())?;
        let (out_buf, out_offset) = crate::arena::arena_alloc_or_create(ctx, out_bytes)?;

        self.dispatch_buffers_with_all_offsets(ctx, &buf_refs, &[], &out_buf, out_offset, &plan)?;
        E::read_buffer_at_offset(&out_buf, out_offset, plan.output_elems())
    }

    /// Dispatch a per-slice reduction kernel with typed inputs.
    ///
    /// One threadgroup per outer slice; threads within each group cooperate to
    /// reduce `reduce` elements. The output has `outer` elements. Shared memory
    /// of `shared_bytes` is allocated for partial sums.
    ///
    /// Constants bound: `[outer, reduce]`.
    ///
    /// Supports `f32` and [`half::f16`] via the [`MetalElement`] trait.
    #[must_use = "returns a Result that may contain an error"]
    pub fn dispatch_reduction<E: MetalElement>(
        &self,
        ctx: &MetalContext,
        inputs: &[&[E]],
        outer: u32,
        reduce: u32,
        threads_per_group: u32,
        shared_bytes: u32,
    ) -> Result<Vec<E>, MetalError> {
        // Lazy batch (#2009): dispatch_buffers encodes into the lazy batch;
        // read_buffer calls flush() before CPU readback.
        let plan = DispatchMode::PerSliceReduction {
            outer,
            reduce,
            threads: threads_per_group,
            shared_bytes,
        }
        .plan()?;
        let input_elems = (outer as usize)
            .checked_mul(reduce as usize)
            .ok_or(MetalError::DispatchSizeOverflow(usize::MAX))?;
        validate_input_lengths(inputs, input_elems)?;

        let mut buffers = Vec::with_capacity(inputs.len());
        for input in inputs {
            buffers.push(E::create_buffer(ctx, input)?);
        }
        let buf_refs: Vec<&MetalBuffer> = buffers.iter().collect();

        let out_bytes = checked_output_bytes(plan.output_elems(), E::element_size())?;
        let (out_buf, out_offset) = crate::arena::arena_alloc_or_create(ctx, out_bytes)?;

        self.dispatch_buffers_with_all_offsets(ctx, &buf_refs, &[], &out_buf, out_offset, &plan)?;
        E::read_buffer_at_offset(&out_buf, out_offset, plan.output_elems())
    }
}

#[cfg(test)]
#[path = "nd_tests.rs"]
mod tests;
