// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused GroupNorm GPU kernel (#3294).
//!
//! `(x - mean) / sqrt(var + eps) * weight + bias` with channel grouping,
//! in a single Metal compute dispatch. Replaces ~14 separate dispatches from
//! the decomposed IR path.
//!
//! Used by:
//! - Demucs DConv sublayers (2 group_norm_g1 calls)
//! - Any model using `layers::GroupNorm`
//!
//! Input: `[B, C, *spatial]`. Reshapes to `[B*G, (C/G)*spatial]` for
//! per-group normalization, then applies per-channel affine.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Device, Result, TensorError};

use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

#[path = "dyn_tensor_metal_group_norm_fused_msl.rs"]
mod msl;

/// Threadgroup size for fused GroupNorm kernel.
const TG_SIZE: usize = 256;

impl super::MetalDynBackend {
    /// Fused GroupNorm using a single Metal dispatch.
    ///
    /// Input `[B, C, *spatial]`, weight/bias `[C]`, num_groups divides C.
    /// Reshapes to `[B*G, (C/G)*spatial]`, normalizes over last dim,
    /// applies per-channel affine.
    ///
    /// Accepts F32, F16, and BF16 inputs. MSL kernel uses `float`
    /// accumulators internally for precision (#3294).
    pub(in super::super) fn gpu_group_norm_fused(
        x: &DynTensor,
        num_groups: usize,
        weight: &DynTensor,
        bias: &DynTensor,
        eps: f64,
    ) -> Result<DynTensor> {
        let dtype = x.dtype();
        let (scalar_type, elem_bytes) = crate::dtype_to_msl(dtype)?;

        let dims = x.dims();
        if dims.len() < 2 {
            return Err(TensorError::InvalidShape(
                "gpu_group_norm_fused requires rank >= 2".into(),
            ));
        }
        let batch = dims[0];
        let channels = dims[1];
        if !channels.is_multiple_of(num_groups) {
            return Err(TensorError::ValueOutOfRange {
                description: "gpu_group_norm_fused: channels not divisible by num_groups",
            });
        }
        let channels_per_group = channels / num_groups;
        let spatial = checked_dim_product(&dims[2..])?;

        // Reshape to [B*G, (C/G)*spatial].
        let flat_rows =
            batch
                .checked_mul(num_groups)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;
        let flat_cols = channels_per_group
            .checked_mul(spatial.max(1))
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: dims.to_vec(),
            })?;

        if flat_cols == 0 || flat_rows == 0 {
            return DynTensor::zeros(dims, dtype, &Device::metal());
        }

        let total_elems =
            flat_rows
                .checked_mul(flat_cols)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;

        let eps_f32 = super::fused_helpers::validate_eps(eps, "gpu_group_norm_fused")?;

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let w_data = weight.gpu_data::<MetalTensorData>()?;
        let b_data = bias.gpu_data::<MetalTensorData>()?;

        let ctx = Self::ctx()?;
        super::with_pipeline_cache(|cache| {
            let kernel_name = format!("fused_group_norm_{scalar_type}");
            let msl_src = msl::fused_group_norm_msl(scalar_type);
            let pipeline = KernelPipeline::from_msl(
                cache,
                &msl_src,
                &kernel_name,
                3, // 3 input buffers: x, weight, bias
                false,
            )
            .map_err(metal_err)?;

            let (out_buf, out_offset) =
                super::fused_helpers::alloc_output(ctx, total_elems, elem_bytes, dims)?;

            let flat_cols_u32 = crate::to_u32(flat_cols, "group_norm flat_cols")?;
            let flat_rows_u32 = crate::to_u32(flat_rows, "group_norm flat_rows")?;
            let cpg_u32 = crate::to_u32(channels_per_group, "group_norm cpg")?;
            let spatial_u32 = crate::to_u32(spatial.max(1), "group_norm spatial")?;
            let num_groups_u32 = crate::to_u32(num_groups, "group_norm num_groups")?;
            let tg_size_u32 = TG_SIZE as u32;

            let encode =
                |batch: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                    let enc = batch.new_encoder()?;
                    enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                    enc.set_buffer_with_offset(1, &w_data.buffer, w_data.byte_offset);
                    enc.set_buffer_with_offset(2, &b_data.buffer, b_data.byte_offset);
                    enc.set_buffer_with_offset(3, &out_buf, out_offset);
                    enc.set_bytes(4, &flat_cols_u32);
                    enc.set_bytes(5, &eps_f32);
                    enc.set_bytes(6, &cpg_u32);
                    enc.set_bytes(7, &spatial_u32);
                    enc.set_bytes(8, &num_groups_u32);
                    enc.encode_threadgroups(
                        pipeline.pipeline(),
                        [flat_rows_u32, 1, 1],
                        [tg_size_u32, 1, 1],
                    )?;
                    enc.end_encoding();
                    Ok(())
                };

            super::fused_helpers::submit_encode(encode)?;
            super::fused_helpers::build_output(out_buf, out_offset, dims, dtype)
        })
    }
}

/// MSL source for pre-compilation: fused GroupNorm F32 kernel.
pub(crate) fn group_norm_msl_source() -> String {
    msl::fused_group_norm_msl("float")
}

/// MSL source for pre-compilation: fused GroupNorm F16 kernel.
pub(crate) fn group_norm_f16_msl_source() -> String {
    msl::fused_group_norm_msl("half")
}
