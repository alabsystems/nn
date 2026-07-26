// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU argmax/argmin dispatch for [`MetalDynBackend`].
//!
//! Returns U32 index tensors via raw MSL kernel. Each output thread reduces
//! one lane along the target dimension using a sequential scan, writing
//! the index of the extreme value.
//!
//! Part of #1147: eliminates GPU→CPU round-trip for argmax/argmin.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{check_dim, DType, Device, Result, TensorError};

use crate::dispatch_plan::DispatchMode;
use crate::kernel_dispatch::KernelPipeline;

use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

/// Generate MSL source for argmax or argmin kernel.
fn argreduce_msl(entry: &str, init_val: &str, cmp_op: &str) -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

kernel void {entry}(
    device const float* data      [[buffer(0)]],
    device uint* output           [[buffer(1)]],
    device const uint& total_els  [[buffer(2)]],
    device const uint& dim_size   [[buffer(3)]],
    device const uint& dim_stride [[buffer(4)]],
    device const uint& inner_size [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= total_els) return;
    uint inner_idx = tid % inner_size;
    uint outer_idx = tid / inner_size;
    uint base = outer_idx * dim_stride + inner_idx;
    float best_val = {init_val};
    uint best_idx = 0;
    for (uint i = 0; i < dim_size; i++) {{
        float val = data[base + i * inner_size];
        if (val {cmp_op} best_val) {{
            best_val = val;
            best_idx = i;
        }}
    }}
    output[tid] = best_idx;
}}
"#,
    )
}

/// Dispatch the argreduce MSL kernel and return a U32 result tensor.
fn dispatch_argreduce(
    x_data: &MetalTensorData,
    msl: &str,
    entry: &str,
    out_shape: &[usize],
    total: usize,
    dim_size: usize,
    dim_stride: usize,
    inner: usize,
    is_rank1: bool,
) -> Result<DynTensor> {
    let ctx = super::MetalDynBackend::ctx()?;
    super::with_pipeline_cache(|cache| {
        let pipeline = KernelPipeline::from_msl(cache, msl, entry, 1, false).map_err(metal_err)?;
        let out_bytes =
            total
                .checked_mul(size_of::<u32>())
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: out_shape.to_vec(),
                })?;
        let (out_buf, out_offset) =
            crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;

        let plan = DispatchMode::Elementwise {
            total: crate::to_u32(total, "argreduce total")?,
        }
        .plan()
        .map_err(metal_err)?
        .with_constants(vec![
            crate::to_u32(total, "argreduce total")?,
            crate::to_u32(dim_size, "argreduce dim_size")?,
            crate::to_u32(dim_stride, "argreduce dim_stride")?,
            crate::to_u32(inner, "argreduce inner")?,
        ]);

        pipeline
            .dispatch_buffers_with_all_offsets(
                ctx,
                &[&x_data.buffer],
                &[x_data.byte_offset],
                &out_buf,
                out_offset,
                &plan,
            )
            .map_err(metal_err)?;

        let storage = MetalTensorData::from_arena_alloc(out_buf, out_offset);
        let result = DynTensor::from_gpu_storage(
            out_shape.to_vec(),
            DType::U32,
            Arc::new(storage),
            Device::metal(),
        )?;
        if is_rank1 {
            result.reshape([])
        } else {
            Ok(result)
        }
    })
}

impl super::MetalDynBackend {
    /// GPU-native argmax: index of maximum value along `dim`.
    pub(super) fn gpu_argmax(x: &DynTensor, dim: usize) -> Result<DynTensor> {
        Self::gpu_argreduce(x, dim, true)
    }

    /// GPU-native argmin: index of minimum value along `dim`.
    pub(super) fn gpu_argmin(x: &DynTensor, dim: usize) -> Result<DynTensor> {
        Self::gpu_argreduce(x, dim, false)
    }

    fn gpu_argreduce(x: &DynTensor, dim: usize, find_max: bool) -> Result<DynTensor> {
        Self::validate_f32_buffer(x, "gpu_argreduce")?;
        let shape = x.dims();
        let ndim = shape.len();
        check_dim(dim, ndim)?;
        let dim_size = shape[dim];
        if dim_size == 0 {
            return Err(TensorError::ZeroLengthDimension {
                axis: dim,
                operation: "argreduce",
            });
        }

        let mut out_shape: Vec<usize> = shape.to_vec();
        out_shape.remove(dim);
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        let total = checked_dim_product(&out_shape)?;
        let x_data = x.gpu_data::<MetalTensorData>()?;
        let inner = checked_dim_product(&shape[dim + 1..])?;
        let dim_stride =
            dim_size
                .checked_mul(inner)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: shape.to_vec(),
                })?;

        let (init_val, cmp_op, entry) = if find_max {
            ("-INFINITY", ">", "argmax_f32")
        } else {
            ("INFINITY", "<", "argmin_f32")
        };
        let msl = argreduce_msl(entry, init_val, cmp_op);

        dispatch_argreduce(
            x_data,
            &msl,
            entry,
            &out_shape,
            total,
            dim_size,
            dim_stride,
            inner,
            shape.len() == 1,
        )
    }
}
