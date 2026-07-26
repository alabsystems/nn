// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GEMM and transpose dispatch helpers for compiled tensor pipelines.
//!
//! Extracted from `tensor_dispatch_helpers.rs` to keep files under 450 lines (#3243).

use nn_dsl::TensorNodeId;

use crate::dispatch_plan::DispatchMode;

use super::{to_dispatch_u32, EncodeContext, TensorDispatchError};

/// Encode a simdgroup-tiled GEMM step into a [`CommandBatch`](crate::dispatch::CommandBatch).
///
/// Uses 3D threadgroup dispatch with `[32, 4, 1]` threads per threadgroup
/// (128 threads, 4 SIMD groups). Grid: `[ceil(N/32), ceil(M/32), batch_count]`.
/// Shared memory: As + Bs + tile_out (see `dyn_tensor_metal_matmul_simd.rs`).
///
/// Part of #2275.
pub(in crate::tensor_dispatch) fn encode_simdgroup_gemm(
    enc: &mut EncodeContext<'_>,
    msl: &str,
    kernel_name: &str,
    step_inputs: &[TensorNodeId],
    output: TensorNodeId,
    m: usize,
    n: usize,
    batch_count: usize,
) -> Result<(), TensorDispatchError> {
    let pipeline = enc.pipeline(msl, kernel_name, step_inputs.len())?;
    let (in_bufs, in_offsets) = enc.inputs(step_inputs)?;

    let total_output = batch_count
        .checked_mul(m)
        .and_then(|v| v.checked_mul(n))
        .ok_or(TensorDispatchError::ShapeOverflow {
            shape: vec![batch_count, m, n],
        })?;
    let (out_buf, out_offset) = enc.alloc_output(total_output)?;

    let is_half = enc.elem_size == 2;
    // Threadgroup memory: As + Bs use element-sized storage, tile_out uses float.
    // f32: 3 × 32 × 33 × 4 = 12,672 bytes
    // f16: 2 × 32 × 33 × 2 + 32 × 33 × 4 = 8,448 bytes
    let tg_bytes: u64 = if is_half {
        2 * 32 * 33 * 2 + 32 * 33 * 4
    } else {
        3 * 32 * 33 * 4
    };

    let m_u32 = to_dispatch_u32(m)?;
    let n_u32 = to_dispatch_u32(n)?;
    let batch_u32 = to_dispatch_u32(batch_count)?;

    let plan = DispatchMode::Grid3D {
        grid: [n_u32.div_ceil(32), m_u32.div_ceil(32), batch_u32],
        threads: [32, 4, 1],
    }
    .plan_cached()
    .map_err(TensorDispatchError::Metal)?
    .with_output_elems(total_output)
    .with_constants(vec![])
    .with_use_threadgroups(true)
    .with_threadgroup_memory_bytes(Some(tg_bytes));

    let encoder = enc.batch.new_encoder()?;
    pipeline.encode_into(encoder, &in_bufs, &in_offsets, &out_buf, out_offset, &plan)?;
    enc.insert_output(output, out_buf, out_offset);
    Ok(())
}

/// Encode a tiled shared-memory 2D transpose into a [`CommandBatch`](crate::dispatch::CommandBatch).
///
/// Uses 2D threadgroups of (TILE, TILE) with shared memory for coalesced
/// reads and writes. Grid z-dimension handles batching. Part of #3230 (Gap 4).
pub(in crate::tensor_dispatch) fn encode_tiled_transpose_step(
    enc: &mut EncodeContext<'_>,
    msl: &str,
    kernel_name: &str,
    input: TensorNodeId,
    output: TensorNodeId,
    batch_size: usize,
    rows: usize,
    cols: usize,
) -> Result<(), TensorDispatchError> {
    let pipeline = enc.pipeline(msl, kernel_name, 1)?;
    let (input_buf, input_offset) = enc.input(input)?;

    let tile = nn_dsl::TILED_TRANSPOSE_TILE_SIZE as u32;
    let m_u32 = to_dispatch_u32(rows)?;
    let n_u32 = to_dispatch_u32(cols)?;
    let batch_u32 = to_dispatch_u32(batch_size)?;

    let total_output = batch_size
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(cols))
        .ok_or(TensorDispatchError::ShapeOverflow {
            shape: vec![batch_size, rows, cols],
        })?;
    let (out_buf, out_offset) = enc.alloc_output(total_output)?;

    // Threadgroup memory: TILE × (TILE + 1) elements (padding avoids bank conflicts).
    let tg_mem_bytes = u64::from(tile) * u64::from(tile + 1) * (enc.elem_size as u64);

    let plan = DispatchMode::Grid3D {
        grid: [n_u32.div_ceil(tile), m_u32.div_ceil(tile), batch_u32],
        threads: [tile, tile, 1],
    }
    .plan_cached()
    .map_err(TensorDispatchError::Metal)?
    .with_output_elems(total_output)
    .with_constants(vec![m_u32, n_u32])
    .with_use_threadgroups(true)
    .with_threadgroup_memory_bytes(Some(tg_mem_bytes));

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

/// Encode a tiled shared-memory GEMM step into a [`CommandBatch`](crate::dispatch::CommandBatch).
///
/// Uses 2D threadgroups of (16, 16) with shared memory for data reuse.
/// Grid z-dimension handles batching. Part of #3230 (Gap 1).
pub(in crate::tensor_dispatch) fn encode_tiled_gemm(
    enc: &mut EncodeContext<'_>,
    msl: &str,
    kernel_name: &str,
    step_inputs: &[TensorNodeId],
    output: TensorNodeId,
    m: usize,
    n: usize,
    batch_count: usize,
) -> Result<(), TensorDispatchError> {
    let pipeline = enc.pipeline(msl, kernel_name, step_inputs.len())?;
    let (in_bufs, in_offsets) = enc.inputs(step_inputs)?;

    let total_output = batch_count
        .checked_mul(m)
        .and_then(|v| v.checked_mul(n))
        .ok_or(TensorDispatchError::ShapeOverflow {
            shape: vec![batch_count, m, n],
        })?;
    let (out_buf, out_offset) = enc.alloc_output(total_output)?;

    let tile: u32 = nn_dsl::TILED_GEMM_TILE as u32;
    let m_u32 = to_dispatch_u32(m)?;
    let n_u32 = to_dispatch_u32(n)?;
    let batch_u32 = to_dispatch_u32(batch_count)?;

    // Threadgroup memory: 2 tiles of TILE × (TILE+1) float (accumulator type is always float).
    let tg_bytes: u64 = 2 * u64::from(tile) * u64::from(tile + 1) * 4;

    let plan = DispatchMode::Grid3D {
        grid: [n_u32.div_ceil(tile), m_u32.div_ceil(tile), batch_u32],
        threads: [tile, tile, 1],
    }
    .plan_cached()
    .map_err(TensorDispatchError::Metal)?
    .with_output_elems(total_output)
    .with_constants(vec![])
    .with_use_threadgroups(true)
    .with_threadgroup_memory_bytes(Some(tg_bytes));

    let encoder = enc.batch.new_encoder()?;
    pipeline.encode_into(encoder, &in_bufs, &in_offsets, &out_buf, out_offset, &plan)?;
    enc.insert_output(output, out_buf, out_offset);
    Ok(())
}
