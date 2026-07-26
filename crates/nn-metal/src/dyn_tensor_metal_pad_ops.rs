// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU constant-padding dispatch for [`MetalDynBackend`].
//!
//! Implements `gpu_pad` using a raw MSL kernel. Each output thread decomposes
//! its flat index into per-dimension coordinates, checks whether the coordinate
//! falls within the source region, and either copies the source value or writes
//! the pad constant.
//!
//! Issue: #3942

use std::mem::size_of;
use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Device, Result, TensorError};

use crate::dispatch_plan::DispatchMode;
use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

impl super::MetalDynBackend {
    /// GPU-native constant padding.
    ///
    /// Each output thread decomposes its flat index into per-dimension coordinates,
    /// checks whether the coordinate falls in the source region (after subtracting
    /// left padding), and either copies the source value or writes the pad constant.
    ///
    /// Supports up to 8 dimensions (covers all practical tensor ranks).
    pub(super) fn gpu_pad(
        x: &DynTensor,
        padding: &[usize],
        value: f64,
    ) -> Result<DynTensor> {
        Self::validate_f32_buffer(x, "gpu_pad")?;

        let rank = x.rank();
        if rank == 0 || rank > 8 {
            return Err(TensorError::Unsupported(format!(
                "gpu_pad: rank {rank} not supported (1..=8)"
            )));
        }

        let n_pad_dims = padding.len() / 2;

        // Compute per-dim left padding and output shape.
        let mut pad_left = vec![0usize; rank];
        let mut out_dims = Vec::with_capacity(rank);
        #[allow(clippy::needless_range_loop)]
        for d in 0..rank {
            let pad_dim_idx = rank - 1 - d;
            let (pl, pr) = if pad_dim_idx < n_pad_dims {
                (padding[2 * pad_dim_idx], padding[2 * pad_dim_idx + 1])
            } else {
                (0, 0)
            };
            pad_left[d] = pl;
            out_dims.push(x.dims()[d] + pl + pr);
        }

        let total_out = checked_dim_product(&out_dims)?;
        if total_out == 0 {
            return DynTensor::zeros(&out_dims, x.dtype(), &Device::metal());
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let ctx = Self::ctx()?;

        // Compute strides for decomposition.
        let x_dims = x.dims();
        let mut src_strides = vec![1u32; rank];
        for d in (0..rank.saturating_sub(1)).rev() {
            src_strides[d] = src_strides[d + 1] * (x_dims[d + 1] as u32);
        }
        let mut out_strides = vec![1u32; rank];
        for d in (0..rank.saturating_sub(1)).rev() {
            out_strides[d] = out_strides[d + 1] * (out_dims[d + 1] as u32);
        }

        // Upload metadata as a single u32 buffer:
        // [out_strides(rank), src_dims(rank), pad_left(rank), src_strides(rank)]
        let mut meta: Vec<u32> = Vec::with_capacity(4 * rank);
        meta.extend(out_strides.iter());
        meta.extend(x_dims.iter().map(|&d| d as u32));
        meta.extend(pad_left.iter().map(|&d| d as u32));
        meta.extend(src_strides.iter());

        let meta_buf = ctx.create_buffer::<u32>(&meta).map_err(metal_err)?;

        // Encode pad_value as u32 bits for MSL `as_type<float>()` reinterpretation.
        let pad_bits = (value as f32).to_bits();

        let msl = format!(
            r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint RANK = {rank};

kernel void pad_f32(
    device const float* src        [[buffer(0)]],
    device const uint*  meta       [[buffer(1)]],
    device float*       output     [[buffer(2)]],
    device const uint&  total_out  [[buffer(3)]],
    device const uint&  pad_bits   [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= total_out) return;

    float pad_value = as_type<float>(pad_bits);

    // meta layout: out_strides[RANK], src_dims[RANK], pad_left[RANK], src_strides[RANK]
    device const uint* out_strides = meta;
    device const uint* src_dims    = meta + RANK;
    device const uint* pad_left_d  = meta + 2 * RANK;
    device const uint* src_strides = meta + 3 * RANK;

    uint remainder = tid;
    uint src_offset = 0;
    for (uint d = 0; d < RANK; d++) {{
        uint coord = remainder / out_strides[d];
        remainder = remainder % out_strides[d];
        uint pl = pad_left_d[d];
        if (coord < pl || coord >= pl + src_dims[d]) {{
            output[tid] = pad_value;
            return;
        }}
        src_offset += (coord - pl) * src_strides[d];
    }}

    output[tid] = src[src_offset];
}}
"#
        );

        let out_bytes = total_out
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| metal_err("pad output bytes overflow"))?;
        let (out_buf, out_byte_offset) =
            crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;

        super::with_pipeline_cache(|cache| {
            let pipeline =
                KernelPipeline::from_msl(cache, &msl, "pad_f32", 2, false).map_err(metal_err)?;

            let plan = DispatchMode::Elementwise {
                total: crate::to_u32(total_out, "pad total_out")?,
            }
            .plan()
            .map_err(metal_err)?
            .with_constants(vec![
                crate::to_u32(total_out, "pad total_out")?,
                pad_bits,
            ]);

            pipeline
                .dispatch_buffers_with_all_offsets(
                    ctx,
                    &[&x_data.buffer, &meta_buf],
                    &[x_data.byte_offset, 0],
                    &out_buf,
                    out_byte_offset,
                    &plan,
                )
                .map_err(metal_err)?;

            let storage = MetalTensorData::from_arena_alloc(out_buf, out_byte_offset);
            DynTensor::from_gpu_storage(out_dims, x.dtype(), Arc::new(storage), Device::metal())
        })
    }
}
