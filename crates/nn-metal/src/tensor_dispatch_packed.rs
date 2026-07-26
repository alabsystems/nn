// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Packed buffer dispatch for kernels with >28 inputs.
//!
//! When an operation has more than `MAX_DIRECT_BINDING_INPUTS` inputs, the MSL
//! codegen emits a "packed" kernel variant that reads all inputs from a single
//! contiguous buffer with an offsets array. This module assembles the packed
//! buffer on GPU using blit copies (no CPU readback of uncommitted GPU data)
//! and dispatches the packed kernel.
//!
//! Supports:
//! - Stack/Concat (homogeneous inputs) — specialized functions
//! - General elementwise (heterogeneous inputs) — `encode_packed_elementwise_step`
//!
//! Part of #1649.

use std::collections::HashMap;

use nn_dsl::TensorNodeId;

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::context::MetalContext;
use crate::dispatch::CommandBatch;
use crate::dispatch_plan::DispatchMode;
use crate::error::MetalError;
use crate::kernel_dispatch::KernelPipeline;

use super::TensorDispatchError;

/// Encode a packed Stack step into a [`CommandBatch`].
///
/// When `n_inputs > MAX_DIRECT_BINDING_INPUTS`, the MSL kernel uses:
/// - `buffer(0)`: packed_inputs (all inputs concatenated element-wise)
/// - `buffer(1)`: offsets (element offset per input, `constant uint*`)
/// - `buffer(2)`: output
/// - `buffer(3)`: total output elements (`constant uint&`)
///
/// The packed buffer is assembled on GPU using blit copies from individual
/// input buffers into a single contiguous buffer. Offsets are computed on
/// CPU (they are cumulative element counts, known at dispatch time).
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_packed_stack_step(
    cache: &PipelineCache,
    batch: &CommandBatch,
    ctx: &MetalContext,
    msl: &str,
    kernel_name: &str,
    step_inputs: &[TensorNodeId],
    output: TensorNodeId,
    input_shape: &[usize],
    total_elements: usize,
    elem_size: usize,
    buffers: &mut HashMap<TensorNodeId, MetalBuffer>,
    offsets: &mut HashMap<TensorNodeId, usize>,
) -> Result<(), TensorDispatchError> {
    let n_inputs = step_inputs.len();

    // Each input has the same shape, so the same element count.
    let elems_per_input = super::helpers::checked_product_of_shape(input_shape)?;
    let bytes_per_input =
        elems_per_input
            .checked_mul(elem_size)
            .ok_or(MetalError::BufferByteOverflow {
                elems: elems_per_input,
                elem_size,
            })?;

    // Allocate packed buffer: n_inputs * bytes_per_input.
    let packed_bytes =
        n_inputs
            .checked_mul(bytes_per_input)
            .ok_or(MetalError::BufferByteOverflow {
                elems: n_inputs.saturating_mul(elems_per_input),
                elem_size,
            })?;
    let (packed_buf, packed_off) = crate::arena::arena_alloc_or_create(ctx, packed_bytes)?;

    // Blit-copy each input buffer into the packed buffer at its offset.
    let mut offsets_u32 = Vec::with_capacity(n_inputs);
    for (i, input_id) in step_inputs.iter().enumerate() {
        let src = buffers
            .get(input_id)
            .ok_or(TensorDispatchError::MissingBuffer(*input_id))?;
        let dst_offset = packed_off + i * bytes_per_input;
        let src_offset = offsets.get(input_id).copied().unwrap_or(0);
        batch.blit_copy(src, src_offset, &packed_buf, dst_offset, bytes_per_input)?;

        // Element offset (not byte offset) for the MSL kernel.
        let elem_offset = u32::try_from(i * elems_per_input)
            .map_err(|_| MetalError::DispatchSizeOverflow(i * elems_per_input))?;
        offsets_u32.push(elem_offset);
    }

    // Create offsets buffer from CPU data (these are known at dispatch time).
    let offsets_buf = ctx.create_buffer(bytemuck::cast_slice::<u32, u8>(&offsets_u32))?;

    // Allocate output buffer via arena (matches encode_elementwise_step pattern).
    let out_bytes =
        total_elements
            .checked_mul(elem_size)
            .ok_or(MetalError::BufferByteOverflow {
                elems: total_elements,
                elem_size,
            })?;
    let (out_buf, out_offset) = crate::arena::arena_alloc_or_create(ctx, out_bytes)?;

    // Build pipeline. fast_math=false for correctness (packed kernels are
    // structural copy operations, not math-sensitive).
    // param_count is unused for manual binding — we set buffers directly.
    let pipeline = KernelPipeline::from_msl(cache, msl, kernel_name, 0, false)?;

    let total_u32 = u32::try_from(total_elements)
        .map_err(|_| MetalError::DispatchSizeOverflow(total_elements))?;
    let plan = DispatchMode::Elementwise { total: total_u32 }.plan_cached()?;

    let encoder = batch.new_encoder()?;
    encoder.set_buffer_with_offset(0, &packed_buf, packed_off);
    encoder.set_buffer(1, &offsets_buf);
    encoder.set_buffer_with_offset(2, &out_buf, out_offset);
    encoder.set_bytes(3, &total_u32);
    encoder.encode(pipeline.pipeline(), plan.grid(), plan.threads())?;
    encoder.end_encoding();

    if out_offset > 0 {
        offsets.insert(output, out_offset);
    }
    buffers.insert(output, out_buf);
    Ok(())
}

/// Encode a packed Concat step into a [`CommandBatch`].
///
/// When `n_inputs > MAX_DIRECT_BINDING_INPUTS`, the MSL kernel uses:
/// - `buffer(0)`: packed_inputs (all inputs concatenated element-wise)
/// - `buffer(1)`: offsets (element offset per input, `constant uint*`)
/// - `buffer(2)`: input_strides (`axis_size * inner_stride` per input, `constant uint*`)
/// - `buffer(3)`: output
/// - `buffer(4)`: total output elements (`constant uint&`)
///
/// Unlike Stack where all inputs have the same shape, Concat inputs may
/// differ along the concat axis. Per-input strides are needed for index
/// computation.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_packed_concat_step(
    cache: &PipelineCache,
    batch: &CommandBatch,
    ctx: &MetalContext,
    msl: &str,
    kernel_name: &str,
    step_inputs: &[TensorNodeId],
    output: TensorNodeId,
    first_input_shape: &[usize],
    input_axis_sizes: &[usize],
    axis: usize,
    total_elements: usize,
    elem_size: usize,
    buffers: &mut HashMap<TensorNodeId, MetalBuffer>,
    offsets: &mut HashMap<TensorNodeId, usize>,
) -> Result<(), TensorDispatchError> {
    let n_inputs = step_inputs.len();

    // inner_stride = product of all dims after axis (same for all inputs).
    let inner_stride: usize =
        first_input_shape[axis + 1..]
            .iter()
            .try_fold(1usize, |acc, &d| {
                acc.checked_mul(d)
                    .ok_or_else(|| TensorDispatchError::ShapeOverflow {
                        shape: first_input_shape.to_vec(),
                    })
            })?;

    // Compute per-input element counts and byte sizes.
    // Each input has shape: [..., input_axis_sizes[i], ...] where dims
    // other than axis match first_input_shape.
    let outer_dims: usize = first_input_shape[..axis]
        .iter()
        .try_fold(1usize, |acc, &d| {
            acc.checked_mul(d)
                .ok_or_else(|| TensorDispatchError::ShapeOverflow {
                    shape: first_input_shape.to_vec(),
                })
        })?;

    let mut offsets_u32 = Vec::with_capacity(n_inputs);
    let mut strides_u32 = Vec::with_capacity(n_inputs);
    let mut per_input_bytes = Vec::with_capacity(n_inputs);
    let mut total_packed_bytes: usize = 0;
    let mut cumulative_elems: usize = 0;

    for (i, &axis_size) in input_axis_sizes.iter().enumerate() {
        let elem_offset = u32::try_from(cumulative_elems)
            .map_err(|_| MetalError::DispatchSizeOverflow(cumulative_elems))?;
        offsets_u32.push(elem_offset);

        let input_stride = axis_size.checked_mul(inner_stride).ok_or_else(|| {
            TensorDispatchError::ShapeOverflow {
                shape: first_input_shape.to_vec(),
            }
        })?;
        let stride_u32 = u32::try_from(input_stride)
            .map_err(|_| MetalError::DispatchSizeOverflow(input_stride))?;
        strides_u32.push(stride_u32);

        let input_elems = outer_dims.checked_mul(input_stride).ok_or_else(|| {
            TensorDispatchError::ShapeOverflow {
                shape: first_input_shape.to_vec(),
            }
        })?;
        let input_bytes =
            input_elems
                .checked_mul(elem_size)
                .ok_or(MetalError::BufferByteOverflow {
                    elems: input_elems,
                    elem_size,
                })?;

        per_input_bytes.push(input_bytes);
        cumulative_elems = cumulative_elems.checked_add(input_elems).ok_or_else(|| {
            TensorDispatchError::ShapeOverflow {
                shape: vec![cumulative_elems, input_elems],
            }
        })?;
        total_packed_bytes =
            total_packed_bytes
                .checked_add(input_bytes)
                .ok_or(MetalError::BufferByteOverflow {
                    elems: cumulative_elems,
                    elem_size,
                })?;

        // Validate input buffer exists before blitting.
        if !buffers.contains_key(&step_inputs[i]) {
            return Err(TensorDispatchError::MissingBuffer(step_inputs[i]));
        }
    }

    // Allocate packed buffer via arena (GPU-only intermediate).
    let (packed_buf, packed_off) = crate::arena::arena_alloc_or_create(ctx, total_packed_bytes)?;

    // Blit-copy each input buffer at its byte offset.
    let mut byte_offset: usize = packed_off;
    for (i, input_id) in step_inputs.iter().enumerate() {
        let src = buffers
            .get(input_id)
            .ok_or(TensorDispatchError::MissingBuffer(*input_id))?;
        let src_offset = offsets.get(input_id).copied().unwrap_or(0);
        batch.blit_copy(
            src,
            src_offset,
            &packed_buf,
            byte_offset,
            per_input_bytes[i],
        )?;
        byte_offset += per_input_bytes[i];
    }

    // Create metadata buffers from CPU data.
    let offsets_buf = ctx.create_buffer(bytemuck::cast_slice::<u32, u8>(&offsets_u32))?;
    let strides_buf = ctx.create_buffer(bytemuck::cast_slice::<u32, u8>(&strides_u32))?;

    // Allocate output buffer via arena (matches encode_elementwise_step pattern).
    let out_bytes =
        total_elements
            .checked_mul(elem_size)
            .ok_or(MetalError::BufferByteOverflow {
                elems: total_elements,
                elem_size,
            })?;
    let (out_buf, out_offset) = crate::arena::arena_alloc_or_create(ctx, out_bytes)?;

    // Packed concat kernel has 3 buffer params (packed_inputs, offsets, strides)
    // but we bind manually. fast_math=false for correctness.
    let pipeline = KernelPipeline::from_msl(cache, msl, kernel_name, 0, false)?;

    let total_u32 = u32::try_from(total_elements)
        .map_err(|_| MetalError::DispatchSizeOverflow(total_elements))?;
    let plan = DispatchMode::Elementwise { total: total_u32 }.plan_cached()?;

    let encoder = batch.new_encoder()?;
    encoder.set_buffer_with_offset(0, &packed_buf, packed_off);
    encoder.set_buffer(1, &offsets_buf);
    encoder.set_buffer(2, &strides_buf);
    encoder.set_buffer_with_offset(3, &out_buf, out_offset);
    encoder.set_bytes(4, &total_u32);
    encoder.encode(pipeline.pipeline(), plan.grid(), plan.threads())?;
    encoder.end_encoding();

    if out_offset > 0 {
        offsets.insert(output, out_offset);
    }
    buffers.insert(output, out_buf);
    Ok(())
}

/// Encode a packed elementwise step into a [`CommandBatch`].
///
/// For element-wise kernels where `n_inputs > MAX_DIRECT_BINDING_INPUTS`,
/// all input buffers are packed into a single contiguous buffer using GPU blit
/// copies. The packed MSL kernel reads parameters via:
/// - `buffer(0)`: packed_inputs
/// - `buffer(1)`: offsets (element offset per input)
/// - `buffer(2)`: output
/// - `buffer(3)`: total element count
///
/// All inputs must have the same element count (`total_elements`), since the
/// scalar kernel reads `packed_inputs[offsets[i] + tid]` for each parameter.
///
/// Part of #1649.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_packed_elementwise_step(
    cache: &PipelineCache,
    batch: &CommandBatch,
    ctx: &MetalContext,
    msl: &str,
    kernel_name: &str,
    step_inputs: &[TensorNodeId],
    output: TensorNodeId,
    total_elements: usize,
    elem_size: usize,
    buffers: &mut HashMap<TensorNodeId, MetalBuffer>,
    offsets: &mut HashMap<TensorNodeId, usize>,
) -> Result<(), TensorDispatchError> {
    let n_inputs = step_inputs.len();
    let bytes_per_input =
        total_elements
            .checked_mul(elem_size)
            .ok_or(MetalError::BufferByteOverflow {
                elems: total_elements,
                elem_size,
            })?;

    // Allocate packed buffer: n_inputs * bytes_per_input.
    let packed_bytes =
        n_inputs
            .checked_mul(bytes_per_input)
            .ok_or(MetalError::BufferByteOverflow {
                elems: n_inputs.saturating_mul(total_elements),
                elem_size,
            })?;
    // Allocate packed buffer via arena (GPU-only intermediate).
    let (packed_buf, packed_off) = crate::arena::arena_alloc_or_create(ctx, packed_bytes)?;

    // Blit-copy each input buffer into the packed buffer at its offset.
    let mut offsets_u32 = Vec::with_capacity(n_inputs);
    for (i, input_id) in step_inputs.iter().enumerate() {
        let src = buffers
            .get(input_id)
            .ok_or(TensorDispatchError::MissingBuffer(*input_id))?;
        let dst_offset = packed_off + i * bytes_per_input;
        let src_offset = offsets.get(input_id).copied().unwrap_or(0);
        batch.blit_copy(src, src_offset, &packed_buf, dst_offset, bytes_per_input)?;

        // Element offset for the MSL kernel.
        let elem_offset = u32::try_from(i * total_elements)
            .map_err(|_| MetalError::DispatchSizeOverflow(i * total_elements))?;
        offsets_u32.push(elem_offset);
    }

    // Create offsets buffer from CPU data.
    let offsets_buf = ctx.create_buffer(bytemuck::cast_slice::<u32, u8>(&offsets_u32))?;

    // Allocate output buffer via arena (matches encode_elementwise_step pattern).
    let (out_buf, out_offset) = crate::arena::arena_alloc_or_create(ctx, bytes_per_input)?;

    // Build pipeline. The packed kernel entry point has `_packed_kernel` suffix.
    let pipeline = KernelPipeline::from_msl(cache, msl, kernel_name, 0, false)?;

    let total_u32 = u32::try_from(total_elements)
        .map_err(|_| MetalError::DispatchSizeOverflow(total_elements))?;
    let plan = DispatchMode::Elementwise { total: total_u32 }.plan_cached()?;

    let encoder = batch.new_encoder()?;
    encoder.set_buffer_with_offset(0, &packed_buf, packed_off);
    encoder.set_buffer(1, &offsets_buf);
    encoder.set_buffer_with_offset(2, &out_buf, out_offset);
    encoder.set_bytes(3, &total_u32);
    encoder.encode(pipeline.pipeline(), plan.grid(), plan.threads())?;
    encoder.end_encoding();

    if out_offset > 0 {
        offsets.insert(output, out_offset);
    }
    buffers.insert(output, out_buf);
    Ok(())
}
