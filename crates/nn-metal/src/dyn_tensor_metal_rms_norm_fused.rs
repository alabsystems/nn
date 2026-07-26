// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused RmsNorm GPU kernel (#3294).
//!
//! `x * rsqrt(mean(x²) + eps) * weight` in a single Metal compute dispatch
//! using simd-accelerated threadgroup reduction. Replaces ~8 separate Metal
//! dispatches per RmsNorm call from the decomposed IR path.
//!
//! Used by:
//! - Qwen3 (4 RmsNorm per transformer layer × N layers)
//! - Any model using `layers::RmsNorm`
//!
//! Input x: any rank >= 1, normalized over the last dimension.
//! weight: `[hidden_dim]` (RmsNorm learnable scale).
//!
//! Unlike LayerNorm, uses single-pass sum of x² (no mean-centering),
//! halving memory reads in the reduction.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Device, Result, TensorError};

use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

#[path = "dyn_tensor_metal_rms_norm_fused_msl.rs"]
mod msl;

/// Threadgroup size for fused RmsNorm kernel.
const TG_SIZE: usize = 256;

impl super::MetalDynBackend {
    /// Fused RmsNorm using a single Metal dispatch.
    ///
    /// Equivalent to `x * rsqrt(mean(x²) + eps) * weight` but uses 1
    /// dispatch instead of ~8 from the decomposed IR path.
    ///
    /// - x: any rank >= 1, normalized over the last dimension
    /// - weight: `[hidden_dim]` (matching last dim of x)
    /// - eps: RmsNorm epsilon
    ///
    /// Accepts F32, F16, and BF16 inputs. MSL kernel uses `float`
    /// accumulators internally for precision (#3294).
    pub(in super::super) fn gpu_rms_norm_fused(
        x: &DynTensor,
        weight: &DynTensor,
        eps: f64,
    ) -> Result<DynTensor> {
        let dtype = x.dtype();
        let (scalar_type, elem_bytes) = crate::dtype_to_msl(dtype)?;

        let dims = x.dims();
        let rank = dims.len();
        if rank == 0 {
            return Err(TensorError::InvalidShape(
                "gpu_rms_norm_fused requires rank >= 1".into(),
            ));
        }
        let hidden_dim = dims[rank - 1];
        let flat_rows = checked_dim_product(&dims[..rank - 1])?;
        // For rank-1 input, flat_rows is 1 (product of empty slice).
        let flat_rows = if flat_rows == 0 && rank == 1 {
            1
        } else {
            flat_rows
        };

        if hidden_dim == 0 || flat_rows == 0 {
            return DynTensor::zeros(dims, dtype, &Device::metal());
        }

        let eps_f32 = super::fused_helpers::validate_eps(eps, "gpu_rms_norm_fused")?;

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let w_data = weight.gpu_data::<MetalTensorData>()?;

        let total_elems =
            flat_rows
                .checked_mul(hidden_dim)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;

        let ctx = Self::ctx()?;
        super::with_pipeline_cache(|cache| {
            let kernel_name = format!("fused_rms_norm_{scalar_type}");
            let msl_src = msl::fused_rms_norm_msl(scalar_type);
            let pipeline = KernelPipeline::from_msl(
                cache,
                &msl_src,
                &kernel_name,
                2, // 2 input buffers: x, weight
                false,
            )
            .map_err(metal_err)?;

            let (out_buf, out_offset) =
                super::fused_helpers::alloc_output(ctx, total_elems, elem_bytes, dims)?;

            let hidden_dim_u32 = crate::to_u32(hidden_dim, "rms_norm hidden_dim")?;
            let flat_rows_u32 = crate::to_u32(flat_rows, "rms_norm flat_rows")?;
            let tg_size_u32 = TG_SIZE as u32;

            let encode =
                |batch: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                    let enc = batch.new_encoder()?;
                    enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                    enc.set_buffer_with_offset(1, &w_data.buffer, w_data.byte_offset);
                    enc.set_buffer_with_offset(2, &out_buf, out_offset);
                    enc.set_bytes(3, &hidden_dim_u32);
                    enc.set_bytes(4, &eps_f32);
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

/// MSL source for pre-compilation: fused RmsNorm F32 kernel.
pub(crate) fn rms_norm_msl_source() -> String {
    msl::fused_rms_norm_msl("float")
}

/// MSL source for pre-compilation: fused RmsNorm F16 kernel.
pub(crate) fn rms_norm_f16_msl_source() -> String {
    msl::fused_rms_norm_msl("half")
}
