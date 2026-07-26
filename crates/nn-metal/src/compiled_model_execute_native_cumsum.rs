// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Zero-alloc Cumsum (Blelloch parallel prefix sum) executor for `CompiledModel`.
//!
//! Extracted from `compiled_model_execute_native_simple.rs` to keep under 450
//! lines. Encodes directly from GpuSlice, eliminating 2 DynTensor wraps
//! (4 heap allocs) per call. Part of #3295 D7.

use std::mem::size_of;

use nn_core::{check_dim, Result};

use crate::cache::PipelineCache;
use crate::dispatch_plan::DispatchMode;
use crate::gpu_slice::GpuSlice;
use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::checked_dim_product;

// `native_dispatch_err` is imported into `simple`'s scope from `native`
// (which re-imports it from `execute::helpers`). `CompiledModel` likewise.
use super::native_dispatch_err;
use super::CompiledModel;

/// Execute a `NativeOpKind::Cumsum` step — zero-alloc path.
///
/// Blelloch parallel prefix sum on Metal. Single-pass for axis_size <= 256,
/// three-pass for 256 < axis_size <= 65536. Encodes directly from GpuSlice
/// without DynTensor wrapping. f32-only (matches existing gpu_cumsum behavior).
/// Part of #3295 D7.
pub(in super::super) fn execute_native_cumsum(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    dim: usize,
    input_shape: &[usize],
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    let ndim = input_shape.len();
    check_dim(dim, ndim)?;

    let axis_size = input_shape[dim];
    let max_axis = crate::dyn_tensor_metal::CUMSUM_MAX_AXIS;
    if axis_size > max_axis {
        return Err(native_dispatch_err(
            step_idx,
            format!("NativeOp Cumsum: axis_size {axis_size} > max {max_axis}"),
        ));
    }

    let outer = checked_dim_product(&input_shape[..dim])
        .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Cumsum outer: {e}")))?;
    let inner = checked_dim_product(&input_shape[dim + 1..])
        .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Cumsum inner: {e}")))?;
    let total_slices = outer
        .checked_mul(inner)
        .ok_or_else(|| native_dispatch_err(step_idx, "NativeOp Cumsum: slices overflow".into()))?;

    let total_elems = checked_dim_product(input_shape)
        .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Cumsum total: {e}")))?;

    if axis_size == 0 || total_slices == 0 {
        let buf = cache
            .context()
            .create_buffer_zeroed(4)
            .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Cumsum zero: {e}")))?;
        return Ok(GpuSlice::from_ref(&buf, 0));
    }

    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;

    let block_size = crate::dyn_tensor_metal::CUMSUM_BLOCK_SIZE;
    if axis_size <= block_size {
        cumsum_single_pass(
            step_idx,
            &input_slice,
            axis_size,
            inner,
            total_slices,
            total_elems,
            cache,
        )
    } else {
        cumsum_multipass(
            step_idx,
            &input_slice,
            axis_size,
            inner,
            total_slices,
            total_elems,
            block_size,
            cache,
        )
    }
}

/// Single-threadgroup Blelloch prefix sum (axis_size <= 256).
fn cumsum_single_pass(
    step_idx: usize,
    input: &GpuSlice,
    axis_size: usize,
    inner: usize,
    total_slices: usize,
    total_elems: usize,
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    let msl_src = crate::dyn_tensor_metal::cumsum_single_pass_msl_source();
    let pipeline = KernelPipeline::from_msl(cache, &msl_src, "cumsum_f32", 1, false)
        .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Cumsum pipeline: {e}")))?;

    let out_bytes = total_elems
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| native_dispatch_err(step_idx, "NativeOp Cumsum: bytes overflow".into()))?;
    let ctx = cache.context();
    let (out_buf, out_offset) = crate::arena::arena_alloc_or_create(ctx, out_bytes)
        .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Cumsum alloc: {e}")))?;

    let axis_size_u32 = crate::to_u32(axis_size, "cumsum axis_size")?;
    let inner_u32 = crate::to_u32(inner, "cumsum inner")?;
    let total_slices_u32 = crate::to_u32(total_slices, "cumsum total_slices")?;

    let plan = DispatchMode::PerSliceReduction {
        outer: total_slices_u32,
        reduce: axis_size_u32,
        threads: 256,
        shared_bytes: 256 * size_of::<f32>() as u32,
    }
    .plan()
    .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Cumsum plan: {e}")))?
    .with_constants(vec![axis_size_u32, inner_u32]);

    pipeline
        .dispatch_buffers_with_all_offsets(
            ctx,
            &[input.buffer()],
            &[input.byte_offset()],
            &out_buf,
            out_offset,
            &plan,
        )
        .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Cumsum dispatch: {e}")))?;

    Ok(GpuSlice::from_ref(&out_buf, out_offset))
}

/// Multi-pass Blelloch prefix sum for 256 < axis_size <= 65536.
///
/// Three-pass algorithm:
/// 1. Block scan: each threadgroup scans a 256-element chunk.
/// 2. Scan block sums: single threadgroup scans the chunk totals.
/// 3. Propagate: each element adds its chunk's scanned prefix.
fn cumsum_multipass(
    step_idx: usize,
    input: &GpuSlice,
    axis_size: usize,
    inner: usize,
    total_slices: usize,
    total_elems: usize,
    block_size: usize,
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    let num_blocks = axis_size.div_ceil(block_size);

    let total_block_sums = total_slices.checked_mul(num_blocks).ok_or_else(|| {
        native_dispatch_err(step_idx, "NativeOp Cumsum: block_sums overflow".into())
    })?;

    // Compile 3 kernels.
    let p1_src = crate::dyn_tensor_metal::cumsum_block_scan_msl_source(block_size);
    let p2_src = crate::dyn_tensor_metal::cumsum_scan_block_sums_msl_source(block_size);
    let p3_src = crate::dyn_tensor_metal::cumsum_propagate_msl_source();

    let p1 = KernelPipeline::from_msl(cache, &p1_src, "cumsum_block_scan", 1, false)
        .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Cumsum p1: {e}")))?;
    let p2 = KernelPipeline::from_msl(cache, &p2_src, "cumsum_scan_block_sums", 1, false)
        .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Cumsum p2: {e}")))?;
    let p3 = KernelPipeline::from_msl(cache, p3_src, "cumsum_propagate", 1, false)
        .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Cumsum p3: {e}")))?;

    // Allocate output and temp buffers.
    let out_bytes = total_elems.checked_mul(size_of::<f32>()).ok_or_else(|| {
        native_dispatch_err(step_idx, "NativeOp Cumsum: out bytes overflow".into())
    })?;
    let block_sum_bytes = total_block_sums
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| {
            native_dispatch_err(step_idx, "NativeOp Cumsum: block_sum bytes overflow".into())
        })?;

    let ctx = cache.context();
    let (out_buf, out_off) = crate::arena::arena_alloc_or_create(ctx, out_bytes)
        .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Cumsum out alloc: {e}")))?;
    let (block_sums_buf, bs_off) = crate::arena::arena_alloc_or_create(ctx, block_sum_bytes)
        .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Cumsum bs alloc: {e}")))?;
    let (scanned_sums_buf, ss_off) = crate::arena::arena_alloc_or_create(ctx, block_sum_bytes)
        .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Cumsum ss alloc: {e}")))?;

    // Pre-compute constants.
    let axis_size_u32 = crate::to_u32(axis_size, "cumsum axis_size")?;
    let inner_u32 = crate::to_u32(inner, "cumsum inner")?;
    let num_blocks_u32 = crate::to_u32(num_blocks, "cumsum num_blocks")?;
    let bs_u32 = crate::to_u32(block_size, "cumsum block_size")?;
    let total_slices_u32 = crate::to_u32(total_slices, "cumsum total_slices")?;
    let total_groups = total_slices * num_blocks;
    let total_groups_u32 = crate::to_u32(total_groups, "cumsum total_groups")?;
    let total_threads = total_slices * axis_size;
    let total_threads_u32 = crate::to_u32(total_threads, "cumsum total_threads")?;
    let tg = 256u32.min(total_threads_u32);
    let groups = total_threads_u32.div_ceil(tg);

    let in_buf = input.buffer();
    let in_off = input.byte_offset();

    crate::gpu_scope::get_or_create_batch()?;
    crate::gpu_scope::encode_into_lazy_batch(
        |batch| -> std::result::Result<(), crate::error::MetalError> {
            // Pass 1: block-level prefix scan.
            {
                let enc = batch.new_encoder()?;
                enc.set_buffer_with_offset(0, in_buf, in_off);
                enc.set_buffer_with_offset(1, &out_buf, out_off);
                enc.set_buffer_with_offset(2, &block_sums_buf, bs_off);
                enc.set_bytes(3, &axis_size_u32);
                enc.set_bytes(4, &inner_u32);
                enc.set_bytes(5, &num_blocks_u32);
                enc.set_threadgroup_memory_length(0, (block_size * size_of::<f32>()) as u64);
                enc.encode_threadgroups(p1.pipeline(), [total_groups_u32, 1, 1], [bs_u32, 1, 1])?;
                enc.end_encoding();
            }
            // Pass 2: scan block sums.
            {
                let enc = batch.new_encoder()?;
                enc.set_buffer_with_offset(0, &block_sums_buf, bs_off);
                enc.set_buffer_with_offset(1, &scanned_sums_buf, ss_off);
                enc.set_bytes(2, &num_blocks_u32);
                enc.set_threadgroup_memory_length(0, (block_size * size_of::<f32>()) as u64);
                enc.encode_threadgroups(p2.pipeline(), [total_slices_u32, 1, 1], [bs_u32, 1, 1])?;
                enc.end_encoding();
            }
            // Pass 3: propagate scanned block sums.
            {
                let enc = batch.new_encoder()?;
                enc.set_buffer_with_offset(0, &out_buf, out_off);
                enc.set_buffer_with_offset(1, &scanned_sums_buf, ss_off);
                enc.set_bytes(2, &axis_size_u32);
                enc.set_bytes(3, &inner_u32);
                enc.set_bytes(4, &num_blocks_u32);
                enc.set_bytes(5, &bs_u32);
                enc.encode_threadgroups(p3.pipeline(), [groups, 1, 1], [tg, 1, 1])?;
                enc.end_encoding();
            }
            Ok(())
        },
    )
    .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Cumsum scope: {e}")))?
    .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Cumsum encode: {e}")))?;

    Ok(GpuSlice::from_ref(&out_buf, out_off))
}
