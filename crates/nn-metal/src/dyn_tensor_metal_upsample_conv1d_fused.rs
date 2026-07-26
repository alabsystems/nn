// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused Upsample1d + Conv1d GPU kernel dispatch (#4310).
//!
//! Single-dispatch architecture: the MSL kernel reads `[B, C_in, T]` input,
//! computes nearest-neighbor upsample inline during Conv1d accumulation,
//! and writes `[B, C_out, T_out]` output. No intermediate upsampled buffer.
//!
//! In Kokoro f0_energy, 6 upsample+conv pairs exist. Fusing each pair
//! from 3+ dispatches (upsample + conv + bias_add) into 1 dispatch saves
//! 12+ dispatches total.
//!
//! Part of #4310.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};
use nn_dsl::ir::ScalarType;

use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::metal_err;

use super::MetalTensorData;

#[path = "dyn_tensor_metal_upsample_conv1d_fused_msl.rs"]
pub(crate) mod upsample_conv1d_fused_msl;

/// Threadgroup width for the fused kernel (matches norm_conv fused pattern).
const TG_X: usize = 64;

impl super::MetalDynBackend {
    /// Fused nearest-neighbor Upsample1d + Conv1d in a single Metal dispatch.
    ///
    /// Input: `[B, C_in, T]`, Weight: `[C_out, C_in, K]`, Bias: `[C_out]`.
    /// Output: `[B, C_out, T_out]` where
    ///   `up_len = T * factor`
    ///   `T_out = (up_len + 2*padding - kernel_size) / stride + 1`
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn gpu_fused_upsample_conv1d(
        input: &DynTensor,
        weight: &DynTensor,
        bias: &DynTensor,
        upsample_factor: usize,
        padding: usize,
        stride: usize,
    ) -> Result<DynTensor> {
        let dtype = input.dtype();
        let st = ScalarType::try_from(dtype)
            .map_err(|_| TensorError::dtype_mismatch(DType::F32, dtype))?;
        let scalar_type = st.msl_str();
        let elem_bytes = st.byte_size();

        let dims = input.dims();
        if dims.len() != 3 {
            return Err(TensorError::InvalidShape(
                "gpu_fused_upsample_conv1d requires rank 3 input [B, C_in, T]".into(),
            ));
        }
        let batch = dims[0];
        let in_channels = dims[1];
        let in_len = dims[2];

        let w_dims = weight.dims();
        if w_dims.len() != 3 {
            return Err(TensorError::InvalidShape(format!(
                "gpu_fused_upsample_conv1d: weight must be rank 3, got {w_dims:?}"
            )));
        }
        let out_channels = w_dims[0];
        let kernel_size = w_dims[2];

        if upsample_factor == 0 {
            return Err(TensorError::InvalidShape(
                "gpu_fused_upsample_conv1d: upsample_factor must be > 0".into(),
            ));
        }

        let up_len = in_len
            .checked_mul(upsample_factor)
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: dims.to_vec(),
            })?;

        let padded = up_len + 2 * padding;
        if padded < kernel_size {
            return Err(TensorError::InvalidShape(format!(
                "gpu_fused_upsample_conv1d: padded length {padded} < kernel_size {kernel_size}"
            )));
        }
        let out_len = (padded - kernel_size) / stride + 1;

        if out_len == 0 {
            return DynTensor::zeros(&[batch, out_channels, 0], dtype, &Device::metal());
        }

        let input_data = input.gpu_data::<MetalTensorData>()?;
        let weight_data = weight.gpu_data::<MetalTensorData>()?;
        let bias_data = bias.gpu_data::<MetalTensorData>()?;

        let ctx = Self::ctx()?;
        super::with_pipeline_cache(|cache| {
            let kernel_name = format!("fused_upsample_conv1d_{scalar_type}");
            let msl_source = upsample_conv1d_fused_msl::fused_upsample_conv1d_msl(scalar_type);
            let pipeline =
                KernelPipeline::from_msl(cache, &msl_source, &kernel_name, 4, false)
                    .map_err(metal_err)?;

            let out_shape = vec![batch, out_channels, out_len];
            let total_out = batch
                .checked_mul(out_channels)
                .and_then(|v| v.checked_mul(out_len))
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: out_shape.clone(),
                })?;
            let out_bytes = total_out.checked_mul(elem_bytes).ok_or_else(|| {
                TensorError::DimensionOverflow {
                    dims: out_shape.clone(),
                }
            })?;
            let (out_buf, out_offset) =
                crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;

            let batch_u32 = crate::to_u32(batch, "fused_up_conv batch")?;
            let in_channels_u32 = crate::to_u32(in_channels, "fused_up_conv in_channels")?;
            let out_channels_u32 = crate::to_u32(out_channels, "fused_up_conv out_channels")?;
            let in_len_u32 = crate::to_u32(in_len, "fused_up_conv in_len")?;
            let up_len_u32 = crate::to_u32(up_len, "fused_up_conv up_len")?;
            let out_len_u32 = crate::to_u32(out_len, "fused_up_conv out_len")?;
            let kernel_size_u32 = crate::to_u32(kernel_size, "fused_up_conv kernel_size")?;
            let stride_u32 = crate::to_u32(stride, "fused_up_conv stride")?;
            let padding_u32 = crate::to_u32(padding, "fused_up_conv padding")?;
            let factor_u32 = crate::to_u32(upsample_factor, "fused_up_conv factor")?;

            let out_rows_u32 = crate::to_u32(batch * out_channels, "fused_up_conv out_rows")?;
            let grid_x = (out_len as u32).div_ceil(TG_X as u32);

            crate::gpu_scope::get_or_create_batch()?;
            let encode =
                |batch_cmd: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                    let enc = batch_cmd.new_encoder()?;
                    enc.set_buffer_with_offset(0, &input_data.buffer, input_data.byte_offset);
                    enc.set_buffer_with_offset(1, &weight_data.buffer, weight_data.byte_offset);
                    enc.set_buffer_with_offset(2, &bias_data.buffer, bias_data.byte_offset);
                    enc.set_buffer_with_offset(3, &out_buf, out_offset);
                    enc.set_bytes(4, &batch_u32);
                    enc.set_bytes(5, &in_channels_u32);
                    enc.set_bytes(6, &out_channels_u32);
                    enc.set_bytes(7, &in_len_u32);
                    enc.set_bytes(8, &up_len_u32);
                    enc.set_bytes(9, &out_len_u32);
                    enc.set_bytes(10, &kernel_size_u32);
                    enc.set_bytes(11, &stride_u32);
                    enc.set_bytes(12, &padding_u32);
                    enc.set_bytes(13, &factor_u32);
                    enc.encode_threadgroups(
                        pipeline.pipeline(),
                        [grid_x, out_rows_u32, 1],
                        [TG_X as u32, 1, 1],
                    )?;
                    enc.end_encoding();
                    Ok(())
                };

            let scope_result = crate::gpu_scope::encode_into_lazy_batch(|b| encode(b));
            match scope_result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(metal_err(e)),
                Err(e) => return Err(e),
            }

            let storage = MetalTensorData::from_arena_alloc(out_buf, out_offset);
            DynTensor::from_gpu_storage(out_shape, dtype, Arc::new(storage), Device::metal())
        })
    }
}
