// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused BatchNorm GPU kernel (#4324).
//!
//! Replaces the ~6-dispatch decomposed path (reshape + broadcast_sub +
//! add_scalar + sqrt + recip + broadcast_mul + broadcast_add) with a single
//! Metal compute dispatch. BatchNorm inference uses precomputed running
//! statistics -- no reduction needed, purely per-element with per-channel
//! parameters.
//!
//! Used by:
//! - ResNet-18 (Table Transformer backbone) -- every residual block
//! - Any CNN model with BatchNorm2d layers
//!
//! Input: `[N, C, *spatial]` (rank >= 2)
//! running_mean, running_var: `[C]`
//! weight, bias: `[C]` (optional)

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};
use nn_dsl::ir::ScalarType;

use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

#[path = "dyn_tensor_metal_batch_norm_fused_msl.rs"]
mod msl;

/// Threadgroup size for the fused BatchNorm kernel.
/// BatchNorm is purely element-wise (no reduction), so we use a standard
/// 256-thread 1D grid with dispatch_threads (Metal auto-computes threadgroups).
const TG_SIZE: usize = 256;

impl super::MetalDynBackend {
    /// Fused BatchNorm inference using a single Metal compute dispatch.
    ///
    /// Equivalent to `(x - running_mean) / sqrt(running_var + eps) * weight + bias`
    /// but uses 1 dispatch instead of ~6.
    ///
    /// - `x`: input tensor `[N, C, *spatial]` (rank >= 2, F32)
    /// - `running_mean`: `[C]` precomputed channel means (F32)
    /// - `running_var`: `[C]` precomputed channel variances (F32)
    /// - `weight`: optional `[C]` affine scale gamma (F32)
    /// - `bias`: optional `[C]` affine shift beta (F32)
    /// - `eps`: numerical stability epsilon
    pub(in super::super) fn gpu_batch_norm_fused(
        x: &DynTensor,
        running_mean: &DynTensor,
        running_var: &DynTensor,
        weight: Option<&DynTensor>,
        bias: Option<&DynTensor>,
        eps: f64,
    ) -> Result<DynTensor> {
        let dtype = x.dtype();
        let st = ScalarType::try_from(dtype)
            .map_err(|_| TensorError::dtype_mismatch(DType::F32, dtype))?;
        let scalar_type = st.msl_str();
        let elem_bytes = st.byte_size();

        let dims = x.dims();
        if dims.len() < 2 {
            return Err(TensorError::InvalidShape(
                "gpu_batch_norm_fused requires rank >= 2".into(),
            ));
        }

        let num_channels = dims[1];
        let spatial_size = checked_dim_product(&dims[2..])?;

        // Total elements in the tensor.
        let total_elems = checked_dim_product(dims)?;
        if total_elems == 0 {
            return DynTensor::zeros(dims, dtype, &Device::metal());
        }

        let eps_f32 = eps as f32;
        if !eps_f32.is_finite() || eps_f32 < 0.0 {
            return Err(TensorError::InvalidShape(format!(
                "gpu_batch_norm_fused: eps must be finite and non-negative, got {eps}"
            )));
        }

        // Validate parameter shapes.
        if running_mean.dims() != [num_channels] {
            return Err(TensorError::shape_mismatch(
                vec![num_channels],
                running_mean.dims().to_vec(),
            ));
        }
        if running_var.dims() != [num_channels] {
            return Err(TensorError::shape_mismatch(
                vec![num_channels],
                running_var.dims().to_vec(),
            ));
        }
        if let Some(w) = weight {
            if w.dims() != [num_channels] {
                return Err(TensorError::shape_mismatch(
                    vec![num_channels],
                    w.dims().to_vec(),
                ));
            }
        }
        if let Some(b) = bias {
            if b.dims() != [num_channels] {
                return Err(TensorError::shape_mismatch(
                    vec![num_channels],
                    b.dims().to_vec(),
                ));
            }
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;
        // running_mean and running_var are always F32.
        let mean_data = running_mean.gpu_data::<MetalTensorData>()?;
        let var_data = running_var.gpu_data::<MetalTensorData>()?;

        let has_weight: u32 = if weight.is_some() { 1 } else { 0 };
        let has_bias: u32 = if bias.is_some() { 1 } else { 0 };

        let ctx = Self::ctx()?;
        super::with_pipeline_cache(|cache| {
            let kernel_name = format!("fused_batch_norm_{scalar_type}");
            let msl_src = msl::fused_batch_norm_msl(scalar_type);
            // Buffer count: input + running_mean + running_var + weight + bias = 5 input buffers.
            // Weight and bias buffers are always bound (may be empty/dummy if not present).
            let pipeline = KernelPipeline::from_msl(
                cache,
                &msl_src,
                &kernel_name,
                5, // 5 input buffers
                false,
            )
            .map_err(metal_err)?;

            let out_bytes = total_elems.checked_mul(elem_bytes).ok_or_else(|| {
                TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                }
            })?;
            let (out_buf, out_offset) =
                crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;

            let num_channels_u32 = crate::to_u32(num_channels, "batch_norm num_channels")?;
            let spatial_size_u32 = crate::to_u32(spatial_size, "batch_norm spatial_size")?;
            let total_elems_u32 = crate::to_u32(total_elems, "batch_norm total_elems")?;
            let tg_size_u32 = TG_SIZE as u32;

            // Get or create dummy buffer for absent weight/bias.
            // We need a valid buffer pointer for Metal even if has_weight/has_bias is 0.
            let dummy_buf = ctx.create_buffer(&[0.0f32]).map_err(metal_err)?;

            let w_data = weight
                .map(|w| w.gpu_data::<MetalTensorData>())
                .transpose()?;
            let b_data = bias
                .map(|b| b.gpu_data::<MetalTensorData>())
                .transpose()?;

            let encode =
                |batch: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                    let enc = batch.new_encoder()?;
                    enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                    enc.set_buffer_with_offset(1, &mean_data.buffer, mean_data.byte_offset);
                    enc.set_buffer_with_offset(2, &var_data.buffer, var_data.byte_offset);
                    // Weight buffer: use actual data or dummy.
                    if let Some(wd) = w_data {
                        enc.set_buffer_with_offset(3, &wd.buffer, wd.byte_offset);
                    } else {
                        enc.set_buffer_with_offset(3, &dummy_buf, 0);
                    }
                    // Bias buffer: use actual data or dummy.
                    if let Some(bd) = b_data {
                        enc.set_buffer_with_offset(4, &bd.buffer, bd.byte_offset);
                    } else {
                        enc.set_buffer_with_offset(4, &dummy_buf, 0);
                    }
                    enc.set_buffer_with_offset(5, &out_buf, out_offset);
                    enc.set_bytes(6, &num_channels_u32);
                    enc.set_bytes(7, &spatial_size_u32);
                    enc.set_bytes(8, &eps_f32);
                    enc.set_bytes(9, &has_weight);
                    enc.set_bytes(10, &has_bias);
                    enc.set_bytes(11, &total_elems_u32);
                    enc.encode(
                        pipeline.pipeline(),
                        [total_elems_u32, 1, 1],
                        [tg_size_u32, 1, 1],
                    )?;
                    enc.end_encoding();
                    Ok(())
                };

            crate::gpu_scope::get_or_create_batch()?;
            let scope_result = crate::gpu_scope::encode_into_lazy_batch(|batch| encode(batch));
            match scope_result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(metal_err(e)),
                Err(e) => return Err(e),
            }

            let storage = MetalTensorData::from_arena_alloc(out_buf, out_offset);
            DynTensor::from_gpu_storage(dims.to_vec(), dtype, Arc::new(storage), Device::metal())
        })
    }
}

/// MSL source for pre-compilation: fused BatchNorm F32 kernel.
pub(crate) fn batch_norm_msl_source() -> String {
    msl::fused_batch_norm_msl("float")
}

/// MSL source for pre-compilation: fused BatchNorm F16 kernel.
pub(crate) fn batch_norm_f16_msl_source() -> String {
    msl::fused_batch_norm_msl("half")
}
