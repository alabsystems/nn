// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Private helper functions for tensor dispatch execution.
//!
//! Extracted from `tensor_dispatch.rs` to keep it under 400 lines (#341).
//! These functions handle MSL compilation, buffer allocation, and individual
//! dispatch step execution (reduce, elementwise).

use std::collections::HashMap;

use nn_dsl::{TensorKernelDef, TensorNodeId};

use crate::arena::arena_alloc_or_create;
use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::context::MetalContext;
use crate::dispatch::CommandBatch;
use crate::dispatch_plan::DispatchMode;
use crate::error::MetalError;
use crate::kernel_dispatch::KernelPipeline;

use super::{TensorDispatchError, REDUCE_THREADGROUP_SIZE};

#[path = "tensor_dispatch_helpers_gemm.rs"]
mod gemm;
pub(super) use gemm::{encode_simdgroup_gemm, encode_tiled_gemm, encode_tiled_transpose_step};

/// Manages buffer allocation and lookup during dispatch step encoding.
///
/// Groups the 6 parameters common to all `encode_*` functions. Constructed
/// once per step in `dispatch_one_step` and passed to each helper.
/// Part of #2981 (EncodeContext design).
pub(super) struct EncodeContext<'a> {
    pub cache: &'a PipelineCache,
    pub batch: &'a CommandBatch,
    pub ctx: &'a MetalContext,
    pub elem_size: usize,
    pub buffers: &'a mut HashMap<TensorNodeId, MetalBuffer>,
    pub offsets: &'a mut HashMap<TensorNodeId, usize>,
}

impl EncodeContext<'_> {
    /// Look up a single input buffer and its byte offset.
    pub(super) fn input(
        &self,
        id: TensorNodeId,
    ) -> Result<(&MetalBuffer, usize), TensorDispatchError> {
        let buf = self
            .buffers
            .get(&id)
            .ok_or(TensorDispatchError::MissingBuffer(id))?;
        let offset = self.offsets.get(&id).copied().unwrap_or(0);
        Ok((buf, offset))
    }

    /// Look up multiple input buffers and offsets.
    pub(super) fn inputs(
        &self,
        ids: &[TensorNodeId],
    ) -> Result<(Vec<&MetalBuffer>, Vec<usize>), TensorDispatchError> {
        let bufs: Vec<&MetalBuffer> = ids
            .iter()
            .map(|id| {
                self.buffers
                    .get(id)
                    .ok_or(TensorDispatchError::MissingBuffer(*id))
            })
            .collect::<Result<_, _>>()?;
        let offs: Vec<usize> = ids
            .iter()
            .map(|id| self.offsets.get(id).copied().unwrap_or(0))
            .collect();
        Ok((bufs, offs))
    }

    /// Allocate output buffer with checked byte size computation.
    pub(super) fn alloc_output(
        &self,
        total_elements: usize,
    ) -> Result<(MetalBuffer, usize), TensorDispatchError> {
        let out_bytes =
            total_elements
                .checked_mul(self.elem_size)
                .ok_or(MetalError::BufferByteOverflow {
                    elems: total_elements,
                    elem_size: self.elem_size,
                })?;
        Ok(arena_alloc_or_create(self.ctx, out_bytes)?)
    }

    /// Insert output buffer and offset into maps. Call after encoding.
    pub(super) fn insert_output(&mut self, id: TensorNodeId, buf: MetalBuffer, offset: usize) {
        if offset > 0 {
            self.offsets.insert(id, offset);
        }
        self.buffers.insert(id, buf);
    }

    /// Create a compiled pipeline from MSL source.
    pub(super) fn pipeline(
        &self,
        msl: &str,
        name: &str,
        param_count: usize,
    ) -> Result<KernelPipeline, TensorDispatchError> {
        Ok(KernelPipeline::from_msl(
            self.cache,
            msl,
            name,
            param_count,
            false,
        )?)
    }
}

/// Convert `usize` to `u32` for Metal dispatch grid dimensions.
pub(super) fn to_dispatch_u32(v: usize) -> Result<u32, TensorDispatchError> {
    u32::try_from(v).map_err(|_| TensorDispatchError::Metal(MetalError::DispatchSizeOverflow(v)))
}

// --- Batched helpers: encode into a CommandBatch without committing ---

/// Encode a reduction step into a [`CommandBatch`].
pub(super) fn encode_reduce(
    enc: &mut EncodeContext<'_>,
    msl: &str,
    kernel_name: &str,
    input: TensorNodeId,
    output: TensorNodeId,
    outer_size: usize,
    reduce_dim: usize,
) -> Result<(), TensorDispatchError> {
    let pipeline = enc.pipeline(msl, kernel_name, 1)?;
    let (input_buf, input_offset) = enc.input(input)?;

    let outer_u32 = to_dispatch_u32(outer_size)?;
    let reduce_u32 = to_dispatch_u32(reduce_dim)?;
    // Tree reduction requires power-of-2 threadgroup size. Round up to the
    // next power of 2 so the tree folds correctly. Threads beyond reduce_dim
    // contribute zero in Phase 1 (their index exceeds reduce_dim).
    let threads = next_power_of_2(reduce_u32).min(REDUCE_THREADGROUP_SIZE);
    let shared_bytes = threads * (enc.elem_size as u32);

    let plan = DispatchMode::PerSliceReduction {
        outer: outer_u32,
        reduce: reduce_u32,
        threads,
        shared_bytes,
    }
    .plan_cached()?
    .with_constants(vec![reduce_u32, outer_u32]);

    let (out_buf, out_offset) = enc.alloc_output(outer_size)?;
    let encoder = enc.batch.new_encoder()?;
    pipeline.encode_into(
        encoder,
        &[input_buf],
        &[input_offset],
        &out_buf,
        out_offset,
        &plan,
    )?;
    enc.insert_output(output, out_buf, out_offset);
    Ok(())
}

/// Encode an elementwise step into a [`CommandBatch`].
pub(super) fn encode_elementwise_step(
    enc: &mut EncodeContext<'_>,
    msl: &str,
    entry_point: &str,
    step_inputs: &[TensorNodeId],
    output: TensorNodeId,
    total_elements: usize,
) -> Result<(), TensorDispatchError> {
    let pipeline = enc.pipeline(msl, entry_point, step_inputs.len())?;
    let (in_bufs, in_offsets) = enc.inputs(step_inputs)?;

    let total_u32 = to_dispatch_u32(total_elements)?;
    let plan = DispatchMode::Elementwise { total: total_u32 }.plan_cached()?;

    let (out_buf, out_offset) = enc.alloc_output(total_elements)?;
    let encoder = enc.batch.new_encoder()?;
    pipeline.encode_into(encoder, &in_bufs, &in_offsets, &out_buf, out_offset, &plan)?;
    enc.insert_output(output, out_buf, out_offset);
    Ok(())
}

/// Encode a softmax step into a [`CommandBatch`].
pub(super) fn encode_softmax(
    enc: &mut EncodeContext<'_>,
    msl: &str,
    kernel_name: &str,
    input: TensorNodeId,
    output: TensorNodeId,
    outer_size: usize,
    axis_size: usize,
) -> Result<(), TensorDispatchError> {
    let pipeline = enc.pipeline(msl, kernel_name, 1)?;
    let (input_buf, input_offset) = enc.input(input)?;

    let outer_u32 = to_dispatch_u32(outer_size)?;
    let axis_u32 = to_dispatch_u32(axis_size)?;
    // Tree reduction requires power-of-2 threadgroup size (same as encode_reduce).
    let threads = next_power_of_2(axis_u32).min(REDUCE_THREADGROUP_SIZE);
    let shared_bytes = 2 * threads * (enc.elem_size as u32);

    let plan = DispatchMode::PerSliceReduction {
        outer: outer_u32,
        reduce: axis_u32,
        threads,
        shared_bytes,
    }
    .plan_cached()?
    .with_constants(vec![axis_u32, outer_u32]);

    let out_elems =
        outer_size
            .checked_mul(axis_size)
            .ok_or(TensorDispatchError::ShapeOverflow {
                shape: vec![outer_size, axis_size],
            })?;
    let (out_buf, out_offset) = enc.alloc_output(out_elems)?;
    let encoder = enc.batch.new_encoder()?;
    pipeline.encode_into(
        encoder,
        &[input_buf],
        &[input_offset],
        &out_buf,
        out_offset,
        &plan,
    )?;
    enc.insert_output(output, out_buf, out_offset);
    Ok(())
}

/// Encode a conv-like step into a [`CommandBatch`].
pub(super) fn encode_conv_like_step(
    enc: &mut EncodeContext<'_>,
    msl: &str,
    kernel_name: &str,
    input: TensorNodeId,
    weight: TensorNodeId,
    bias: Option<TensorNodeId>,
    output: TensorNodeId,
    total_elements: usize,
) -> Result<(), TensorDispatchError> {
    let mut step_inputs = vec![input, weight];
    if let Some(b) = bias {
        step_inputs.push(b);
    }
    encode_elementwise_step(enc, msl, kernel_name, &step_inputs, output, total_elements)
}

/// Round up to the next power of 2. Returns `v` unchanged if already a power of 2.
/// Returns 1 for input 0.
///
/// For values > 2^31 where the true next power of 2 (2^32) doesn't fit in u32,
/// returns 2^31 (the largest power of 2 representable in u32).
pub(crate) fn next_power_of_2(v: u32) -> u32 {
    if v <= 1 {
        return 1;
    }
    // SAFETY: checked_next_power_of_two returns None when the result would
    // overflow u32 (i.e., for v > 2^31). In that case we return 1 << 31,
    // the largest u32 power of 2. In practice this path is unreachable
    // because .min(REDUCE_THREADGROUP_SIZE) caps the result at 256, but
    // the function must be correct regardless of caller context.
    v.checked_next_power_of_two().unwrap_or(1 << 31)
}

pub(crate) fn checked_product_of_shape(shape: &[usize]) -> Result<usize, TensorDispatchError> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| TensorDispatchError::ShapeOverflow {
                shape: shape.to_vec(),
            })
    })
}

/// Look up output node shape and compute total elements.
pub(crate) fn output_elems(
    kernel: &TensorKernelDef,
    output: TensorNodeId,
) -> Result<usize, TensorDispatchError> {
    let node =
        kernel
            .nodes
            .get(output.index())
            .ok_or(TensorDispatchError::NodeIndexOutOfBounds {
                index: output.index(),
                len: kernel.nodes.len(),
            })?;
    checked_product_of_shape(&node.shape)
}
