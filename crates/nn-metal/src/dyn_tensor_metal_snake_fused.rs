// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused per-channel Snake activation GPU kernel (#3294).
//!
//! `x + (1/alpha) * sin²(alpha * x)` in a single Metal compute dispatch.
//! Replaces ~6 separate GPU dispatches from the decomposed IR path.
//!
//! Used by standalone `snake_tensor()` calls. The Kokoro ResBlock path
//! uses `adain_snake_fused` instead (InstanceNorm + affine + Snake in 1).
//!
//! Input x: any shape. Alpha: per-channel, left-aligned broadcast.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Device, Result};

use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

#[path = "dyn_tensor_metal_snake_fused_msl.rs"]
mod msl;

/// Threadgroup size for fused Snake kernel.
const TG_SIZE: usize = 256;

impl super::MetalDynBackend {
    /// Fused per-channel Snake activation using a single Metal dispatch.
    ///
    /// `x + (1/alpha) * sin²(alpha * x)` where alpha broadcasts
    /// left-aligned over x. Alpha shape must be `[C]` or broadcastable
    /// left-aligned over x.
    ///
    /// Accepts F32, F16, and BF16 inputs. MSL kernel uses `float`
    /// intermediates for trig precision (#3294).
    pub(in super::super) fn gpu_snake_tensor_fused(
        x: &DynTensor,
        alpha: &DynTensor,
    ) -> Result<DynTensor> {
        let dtype = x.dtype();
        let (scalar_type, elem_bytes) = crate::dtype_to_msl(dtype)?;

        let dims = x.dims();
        let total_elems = checked_dim_product(dims)?;

        if total_elems == 0 {
            return DynTensor::zeros(dims, dtype, &Device::metal());
        }

        // Alpha is per-channel. Determine channel dim and stride for broadcast.
        // Alpha shape is typically [C] (rank 1) or [1, C, 1, ...] (rank matching x).
        let alpha_dims = alpha.dims();
        let channels = if alpha_dims.len() == 1 {
            alpha_dims[0]
        } else {
            // Find the non-1 dimension (channel dim).
            alpha_dims.iter().find(|&&d| d != 1).copied().unwrap_or(1)
        };

        // Channel stride = product of spatial dims (dims after channel dim).
        // For [B, C, *spatial]: channel_stride = product(spatial).
        // For [C]: channel_stride = 1 (alpha is just indexed by element).
        let channel_stride = if dims.len() >= 2 {
            checked_dim_product(&dims[2..])?
        } else {
            1
        };
        let channel_stride = channel_stride.max(1);

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let alpha_data = alpha.gpu_data::<MetalTensorData>()?;

        let ctx = Self::ctx()?;
        super::with_pipeline_cache(|cache| {
            let kernel_name = format!("fused_snake_{scalar_type}");
            let msl_src = msl::fused_snake_msl(scalar_type);
            let pipeline = KernelPipeline::from_msl(
                cache,
                &msl_src,
                &kernel_name,
                2, // 2 input buffers: x, alpha
                false,
            )
            .map_err(metal_err)?;

            let (out_buf, out_offset) =
                super::fused_helpers::alloc_output(ctx, total_elems, elem_bytes, dims)?;

            let total_elems_u32 = crate::to_u32(total_elems, "snake total_elems")?;
            let channel_stride_u32 = crate::to_u32(channel_stride, "snake channel_stride")?;
            let channels_u32 = crate::to_u32(channels, "snake channels")?;
            let num_threadgroups = total_elems_u32.div_ceil(TG_SIZE as u32);

            let encode =
                |batch: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                    let enc = batch.new_encoder()?;
                    enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                    enc.set_buffer_with_offset(1, &alpha_data.buffer, alpha_data.byte_offset);
                    enc.set_buffer_with_offset(2, &out_buf, out_offset);
                    enc.set_bytes(3, &total_elems_u32);
                    enc.set_bytes(4, &channel_stride_u32);
                    enc.set_bytes(5, &channels_u32);
                    enc.encode_threadgroups(
                        pipeline.pipeline(),
                        [num_threadgroups, 1, 1],
                        [TG_SIZE as u32, 1, 1],
                    )?;
                    enc.end_encoding();
                    Ok(())
                };

            super::fused_helpers::submit_encode(encode)?;
            super::fused_helpers::build_output(out_buf, out_offset, dims, dtype)
        })
    }
}

/// MSL source for pre-compilation: fused Snake F32 kernel.
pub(crate) fn snake_msl_source() -> String {
    msl::fused_snake_msl("float")
}

/// MSL source for pre-compilation: fused Snake F16 kernel.
pub(crate) fn snake_f16_msl_source() -> String {
    msl::fused_snake_msl("half")
}
