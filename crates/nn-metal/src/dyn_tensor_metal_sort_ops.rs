// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU sort dispatch for [`MetalDynBackend`].
//!
//! Implements `gpu_sort` using a per-lane insertion sort MSL kernel. Each GPU
//! thread owns one lane (a slice along the sort dimension) and sorts it
//! independently. This is efficient for moderate axis sizes (up to ~65536)
//! typical in ML workloads (top-k preselection, NMS, attention routing).
//!
//! Issue: #3942

use std::mem::size_of;
use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{check_dim, DType, Device, Result};

use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

impl super::MetalDynBackend {
    /// GPU-native sort along a dimension using per-lane insertion sort.
    ///
    /// Returns `(values, indices)` where both have the same shape as input.
    /// `indices` is U32 dtype. When `descending` is true, largest values come
    /// first. Each GPU thread independently sorts one lane along the dimension.
    ///
    /// Falls back to CPU for axis sizes > 65536 (too much per-thread work).
    pub(super) fn gpu_sort(
        x: &DynTensor,
        dim: usize,
        descending: bool,
    ) -> Result<(DynTensor, DynTensor)> {
        // Flush pending GPU work before validation readback.
        crate::gpu_scope::flush()?;
        Self::validate_f32_buffer(x, "gpu_sort")?;

        let shape = x.dims();
        let ndim = shape.len();
        check_dim(dim, ndim)?;

        let dim_size = shape[dim];
        if dim_size <= 1 {
            // Already sorted. Return clone + identity indices.
            let idx_shape = shape.to_vec();
            let idx_data: Vec<u32> = if dim_size == 1 {
                let n_lanes = checked_dim_product(&shape[..dim])?
                    * checked_dim_product(&shape[dim + 1..])?;
                vec![0u32; n_lanes]
            } else {
                vec![]
            };
            let indices = DynTensor::from_cpu_u32(ndarray::ArrayD::<u32>::from_shape_vec(
                ndarray::IxDyn(&idx_shape),
                idx_data,
            )?)?
            .to_device(&Device::metal())?;
            return Ok((x.clone(), indices));
        }

        let inner = checked_dim_product(&shape[dim + 1..])?;
        let outer = checked_dim_product(&shape[..dim])?;
        let n_lanes = outer * inner;

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let ctx = Self::ctx()?;

        // Allocate output buffers.
        let numel = checked_dim_product(shape)?;
        let val_bytes = numel
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| metal_err("sort val bytes overflow"))?;
        let idx_bytes = numel
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| metal_err("sort idx bytes overflow"))?;

        let val_buf = ctx.create_buffer_zeroed(val_bytes).map_err(metal_err)?;
        let idx_buf = ctx.create_buffer_zeroed(idx_bytes).map_err(metal_err)?;

        let cmp_op = if descending { ">" } else { "<" };

        // Per-lane insertion sort kernel. One thread per lane.
        let msl = format!(
            r#"
#include <metal_stdlib>
using namespace metal;

kernel void sort_f32(
    device const float* input      [[buffer(0)]],
    device float*       out_vals   [[buffer(1)]],
    device uint*        out_idxs   [[buffer(2)]],
    device const uint&  dim_size_c [[buffer(3)]],
    device const uint&  inner_c    [[buffer(4)]],
    device const uint&  n_lanes_c  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= n_lanes_c) return;

    uint dim_size = dim_size_c;
    uint inner = inner_c;

    uint outer_idx = tid / inner;
    uint inner_idx = tid % inner;

    // Base offset: outer * (dim_size * inner) + inner_idx
    uint base = outer_idx * (dim_size * inner) + inner_idx;

    // Copy input to output and initialize identity indices.
    for (uint i = 0; i < dim_size; i++) {{
        uint off = base + i * inner;
        out_vals[off] = input[off];
        out_idxs[off] = i;
    }}

    // Insertion sort along the lane.
    for (uint i = 1; i < dim_size; i++) {{
        uint i_off = base + i * inner;
        float key = out_vals[i_off];
        uint key_idx = out_idxs[i_off];
        uint j = i;
        while (j > 0) {{
            uint prev_off = base + (j - 1) * inner;
            float prev = out_vals[prev_off];
            if (!(key {cmp_op} prev)) break;
            uint j_off = base + j * inner;
            out_vals[j_off] = prev;
            out_idxs[j_off] = out_idxs[prev_off];
            j--;
        }}
        uint j_off = base + j * inner;
        out_vals[j_off] = key;
        out_idxs[j_off] = key_idx;
    }}
}}
"#
        );

        super::with_pipeline_cache(|cache| {
            let pipeline = crate::kernel_dispatch::KernelPipeline::from_msl(
                cache, &msl, "sort_f32", 0, false,
            )
            .map_err(metal_err)?;

            // Custom encoding: we need 3 buffers (input, out_vals, out_idxs)
            // plus 3 scalar constants, which doesn't fit the standard
            // inputs[0..N] + output + constants layout.
            //
            // Pre-compute u32 constants outside the closure to avoid
            // TensorError→MetalError conversion inside the batch encoder.
            let dim_size_u32 = crate::to_u32(dim_size, "sort dim_size")?;
            let inner_u32 = crate::to_u32(inner, "sort inner")?;
            let n_lanes_u32 = crate::to_u32(n_lanes, "sort n_lanes")?;

            crate::gpu_scope::get_or_create_batch()
                .map_err(|e| metal_err(e.to_string()))?;
            let scope_result = crate::gpu_scope::encode_into_lazy_batch(
                |batch| -> std::result::Result<(), crate::error::MetalError> {
                    let enc = batch.new_encoder()?;
                    enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                    enc.set_buffer(1, &val_buf);
                    enc.set_buffer(2, &idx_buf);
                    enc.set_bytes(3, &dim_size_u32);
                    enc.set_bytes(4, &inner_u32);
                    enc.set_bytes(5, &n_lanes_u32);

                    // Grid: one thread per lane.
                    let grid = [n_lanes_u32, 1, 1];
                    let tg = [64.min(n_lanes_u32), 1, 1];
                    enc.encode(pipeline.pipeline(), grid, tg)?;
                    enc.end_encoding();
                    Ok(())
                },
            );
            match scope_result {
                Ok(inner_result) => inner_result.map_err(metal_err)?,
                Err(e) => return Err(metal_err(e.to_string())),
            }

            let val_storage = MetalTensorData::new(val_buf);
            let idx_storage = MetalTensorData::new(idx_buf);

            let values = DynTensor::from_gpu_storage(
                shape.to_vec(),
                x.dtype(),
                Arc::new(val_storage),
                Device::metal(),
            )?;
            let indices = DynTensor::from_gpu_storage(
                shape.to_vec(),
                DType::U32,
                Arc::new(idx_storage),
                Device::metal(),
            )?;

            Ok((values, indices))
        })
    }
}
