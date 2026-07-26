// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU data-dependent op dispatch for [`MetalDynBackend`].
//!
//! These ops use raw MSL kernels via `KernelPipeline::from_msl()` rather than
//! the `TensorBlockBuilder` pipeline, because they involve data-dependent
//! indexing (gather, scatter_add), prefix scans (cumsum), or variable-length
//! expansion (repeat_interleave) that don't decompose into element-wise scalar
//! kernels.
//!
//! Cumsum (prefix scan) extracted to `dyn_tensor_metal_cumsum_ops.rs`.
//!
//! Design: `designs/2026-03-05-metal-gpu-data-ops.md`
//! Issue: #1178

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{check_dim, DType, Device, Result, TensorError};

use crate::dispatch_plan::DispatchMode;
use crate::kernel_dispatch::KernelPipeline;

use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

impl super::MetalDynBackend {
    // ===== gather =====
    //
    // output[i][j][k] = input[i][index[i][j][k]][k]  (when dim=1)
    // General: replace the `dim` axis coordinate with the index value.

    /// GPU-native gather: indexed read along one axis.
    ///
    /// MSL kernel: each output thread decomposes its linear index into
    /// (outer, idx_axis, inner), reads the gather index, and fetches from the
    /// source at the indexed position. OOB indices produce 0.0.
    pub(super) fn gpu_gather(x: &DynTensor, ids: &DynTensor, dim: usize) -> Result<DynTensor> {
        Self::validate_f32_buffer(x, "gpu_gather")?;

        let x_shape = x.dims();
        let ids_shape = ids.dims();
        let ndim = x_shape.len();

        check_dim(dim, ndim)?;

        // Output shape = ids shape
        let out_shape = ids_shape.to_vec();
        let total = checked_dim_product(&out_shape)?;
        if total == 0 {
            return DynTensor::zeros(&out_shape, x.dtype(), &Device::metal());
        }

        // Compute strides for index decomposition.
        // ids_inner = product of ids_shape after dim (for decomposing thread index)
        // data_inner = product of x_shape after dim (for computing data offset)
        let data_dim_size = x_shape[dim];
        let ids_inner = checked_dim_product(&ids_shape[dim + 1..])?;
        let data_inner = checked_dim_product(&x_shape[dim + 1..])?;
        let ids_dim_size = ids_shape[dim];
        let ids_outer_stride =
            ids_dim_size
                .checked_mul(ids_inner)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: ids_shape.to_vec(),
                })?;

        let x_data = x.gpu_data::<MetalTensorData>()?;

        // Upload indices as native u32 Metal buffer to preserve precision
        // for indices > 2^24 (f32 has only 24-bit mantissa).
        let ids_u32 = if ids.dtype() == DType::U32 && ids.device().is_gpu() {
            ids.clone()
        } else {
            let cpu_ids = ids.to_device(&Device::Cpu)?;
            let u32_ids = cpu_ids.to_dtype(DType::U32)?;
            u32_ids.to_device(&Device::metal())?
        };
        let ids_data = ids_u32.gpu_data::<MetalTensorData>()?;

        let msl = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gather_f32(
    device const float* data       [[buffer(0)]],
    device const uint*  indices    [[buffer(1)]],
    device float* output           [[buffer(2)]],
    device const uint& total_els   [[buffer(3)]],
    device const uint& data_dim    [[buffer(4)]],
    device const uint& ids_inner_s [[buffer(5)]],
    device const uint& ids_outer_s [[buffer(6)]],
    device const uint& data_inn_s  [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_els) return;

    // Decompose tid using ids shape strides
    uint inner = tid % ids_inner_s;
    uint outer = tid / ids_outer_s;

    uint src_idx = indices[tid];
    if (src_idx >= data_dim) {
        output[tid] = 0.0f;
        return;
    }

    // Compute offset into data using data shape strides
    uint data_offset = outer * (data_dim * data_inn_s) + src_idx * data_inn_s + inner;
    output[tid] = data[data_offset];
}
"#;

        Self::dispatch_raw_msl(
            msl,
            "gather_f32",
            2, // param_count: data + indices
            &[&x_data.buffer, &ids_data.buffer],
            &[x_data.byte_offset, ids_data.byte_offset],
            total,
            &out_shape,
            x.dtype(),
            vec![
                crate::to_u32(total, "gather total")?,
                crate::to_u32(data_dim_size, "gather data_dim_size")?,
                crate::to_u32(ids_inner, "gather ids_inner")?,
                crate::to_u32(ids_outer_stride, "gather ids_outer_stride")?,
                crate::to_u32(data_inner, "gather data_inner")?,
            ],
        )
    }

    // ===== repeat_interleave =====
    //
    // Expand each element along `dim` by its corresponding repeat count.
    // E.g. [a,b,c] with counts [2,1,3] along dim=0 -> [a,a,b,c,c,c].
    // Uses a prefix-sum offsets buffer (computed on CPU) so each GPU thread
    // can binary-search to find its source element in O(log n).

    /// GPU-native repeat_interleave along one axis.
    ///
    /// CPU prepares a prefix-sum offsets buffer and uploads it to GPU.
    /// MSL kernel: each output thread binary-searches the offsets to find
    /// which source element to copy from.
    pub(super) fn gpu_repeat_interleave(
        x: &DynTensor,
        dim: usize,
        counts: &[usize],
    ) -> Result<DynTensor> {
        Self::validate_f32_buffer(x, "gpu_repeat_interleave")?;

        let shape = x.dims();
        let ndim = shape.len();

        check_dim(dim, ndim)?;

        let dim_size = shape[dim];
        if counts.len() != dim_size {
            return Err(TensorError::InvalidShape(format!(
                "gpu_repeat_interleave: counts.len()={} != dim_size={dim_size}",
                counts.len()
            )));
        }

        let total_repeats: usize = counts.iter().sum();
        if total_repeats == 0 {
            let mut out_shape = shape.to_vec();
            out_shape[dim] = 0;
            return DynTensor::zeros(&out_shape, x.dtype(), &Device::metal());
        }

        // Build output shape
        let mut out_shape = shape.to_vec();
        out_shape[dim] = total_repeats;
        let total_elems = checked_dim_product(&out_shape)?;

        // Build prefix-sum offsets on CPU: offsets[i] = sum(counts[0..i])
        // The MSL kernel binary-searches this to map output index -> source index.
        let mut offsets: Vec<u32> = Vec::with_capacity(dim_size + 1);
        offsets.push(0);
        let mut acc: u32 = 0;
        for &c in counts {
            let c_u32 = crate::to_u32(c, "repeat_interleave count")?;
            acc = acc.checked_add(c_u32).ok_or_else(|| {
                TensorError::InvalidShape("repeat_interleave: offset overflow".into())
            })?;
            offsets.push(acc);
        }

        // Upload offsets to GPU buffer
        let ctx = Self::ctx()?;
        let offsets_buf = ctx.create_buffer(&offsets).map_err(metal_err)?;

        let inner = checked_dim_product(&shape[dim + 1..])?;
        let x_data = x.gpu_data::<MetalTensorData>()?;

        let msl = r#"
#include <metal_stdlib>
using namespace metal;

kernel void repeat_interleave_f32(
    device const float* input      [[buffer(0)]],
    device const uint*  offsets    [[buffer(1)]],
    device float* output           [[buffer(2)]],
    device const uint& total_els   [[buffer(3)]],
    device const uint& dim_size    [[buffer(4)]],
    device const uint& out_dim     [[buffer(5)]],
    device const uint& inner_sz    [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_els) return;

    // Decompose tid into (outer_idx, out_axis_idx, inner_idx)
    uint out_axis_inner = out_dim * inner_sz;
    uint outer_idx = tid / out_axis_inner;
    uint rem = tid % out_axis_inner;
    uint out_axis_idx = rem / inner_sz;
    uint inner_idx = rem % inner_sz;

    // Binary search offsets to find source element index.
    // offsets[i] <= out_axis_idx < offsets[i+1] => source element is i.
    uint lo = 0, hi = dim_size;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (offsets[mid + 1] <= out_axis_idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    uint src_idx = lo;

    // Compute input offset
    uint in_offset = outer_idx * (dim_size * inner_sz) + src_idx * inner_sz + inner_idx;
    output[tid] = input[in_offset];
}
"#;

        Self::dispatch_raw_msl(
            msl,
            "repeat_interleave_f32",
            2, // param_count: input + offsets
            &[&x_data.buffer, &offsets_buf],
            &[x_data.byte_offset, 0],
            total_elems,
            &out_shape,
            x.dtype(),
            vec![
                crate::to_u32(total_elems, "repeat_interleave total_elems")?,
                crate::to_u32(dim_size, "repeat_interleave dim_size")?,
                crate::to_u32(total_repeats, "repeat_interleave total_repeats")?,
                crate::to_u32(inner, "repeat_interleave inner")?,
            ],
        )
    }

    // ===== shared raw MSL dispatch helper =====

    /// Dispatch a raw MSL kernel with explicit constant buffer bindings.
    ///
    /// Constants are bound to buffer slots after input + output buffers.
    /// Uses `DispatchMode::Elementwise` for one-thread-per-element kernels.
    pub(super) fn dispatch_raw_msl(
        msl: &str,
        entry_point: &str,
        param_count: usize,
        input_buffers: &[&crate::buffer::MetalBuffer],
        input_offsets: &[usize],
        total_elems: usize,
        out_shape: &[usize],
        out_dtype: DType,
        constants: Vec<u32>,
    ) -> Result<DynTensor> {
        let ctx = Self::ctx()?;
        super::with_pipeline_cache(|cache| {
            let pipeline = KernelPipeline::from_msl(cache, msl, entry_point, param_count, false)
                .map_err(metal_err)?;

            // Use out_dtype byte width for buffer allocation. All current callers
            // pass F32 (4 bytes), but this guard prevents silent data corruption
            // if a non-F32 dtype is ever passed.
            let elem_bytes = out_dtype.size_bytes();
            let out_bytes = total_elems.checked_mul(elem_bytes).ok_or_else(|| {
                TensorError::DimensionOverflow {
                    dims: out_shape.to_vec(),
                }
            })?;
            let (out_buf, out_offset) =
                crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;

            let plan = DispatchMode::Elementwise {
                total: crate::to_u32(total_elems, "dispatch_raw_msl total")?,
            }
            .plan()
            .map_err(metal_err)?
            .with_constants(constants);

            pipeline
                .dispatch_buffers_with_all_offsets(
                    ctx,
                    input_buffers,
                    input_offsets,
                    &out_buf,
                    out_offset,
                    &plan,
                )
                .map_err(metal_err)?;

            let storage = MetalTensorData::from_arena_alloc(out_buf, out_offset);
            DynTensor::from_gpu_storage(
                out_shape.to_vec(),
                out_dtype,
                Arc::new(storage),
                Device::metal(),
            )
        })
    }
}
