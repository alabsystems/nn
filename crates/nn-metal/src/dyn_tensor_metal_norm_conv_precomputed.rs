// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv dispatch with precomputed stats — skips the stats kernel (#1815 Tier 2).
//!
//! Used by FusedResBlock phase 2: the stats were computed by phase 1's
//! conv-with-stats epilogue, so phase 2 only needs the conv dispatch (1 kernel).

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};
use nn_dsl::ir::ScalarType;

use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::metal_err;

use super::super::MetalTensorData;
use super::{PrecomputedStats, StatsActivation, CONV_TG_X};

impl super::super::MetalDynBackend {
    /// Conv with precomputed stats (skip stats dispatch). LeakyRelu variant.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super::super) fn gpu_norm_activ_conv1d_with_precomputed_stats(
        x: &DynTensor,
        gamma: &DynTensor,
        beta: &DynTensor,
        weight: &DynTensor,
        bias: &DynTensor,
        slope: f64,
        padding: usize,
        dilation: usize,
        residual: Option<super::super::norm_conv_fused::ResidualParams<'_>>,
        precomputed: &PrecomputedStats,
    ) -> Result<DynTensor> {
        let slope_f32 = slope as f32;
        if !slope_f32.is_finite() {
            return Err(TensorError::InvalidShape(format!(
                "conv_with_precomputed_stats: slope not finite: {slope}"
            )));
        }
        dispatch_conv_with_precomputed_stats(
            x,
            gamma,
            beta,
            weight,
            bias,
            StatsActivation::LeakyRelu { slope: slope_f32 },
            padding,
            dilation,
            residual,
            precomputed,
        )
    }

    /// Conv with precomputed stats (skip stats dispatch). Snake variant.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super::super) fn gpu_norm_activ_conv1d_snake_with_precomputed_stats(
        x: &DynTensor,
        gamma: &DynTensor,
        beta: &DynTensor,
        alpha: &DynTensor,
        weight: &DynTensor,
        bias: &DynTensor,
        padding: usize,
        dilation: usize,
        residual: Option<super::super::norm_conv_fused::ResidualParams<'_>>,
        precomputed: &PrecomputedStats,
    ) -> Result<DynTensor> {
        let alpha_data = alpha.gpu_data::<MetalTensorData>()?;
        dispatch_conv_with_precomputed_stats(
            x,
            gamma,
            beta,
            weight,
            bias,
            StatsActivation::Snake { alpha_data },
            padding,
            dilation,
            residual,
            precomputed,
        )
    }
}

/// 1 Metal dispatch: conv kernel with external stats at buffer(1).
#[allow(clippy::too_many_arguments)]
fn dispatch_conv_with_precomputed_stats(
    x: &DynTensor,
    gamma: &DynTensor,
    beta: &DynTensor,
    weight: &DynTensor,
    bias: &DynTensor,
    activation: StatsActivation<'_>,
    padding: usize,
    dilation: usize,
    residual: Option<super::super::norm_conv_fused::ResidualParams<'_>>,
    precomputed: &PrecomputedStats,
) -> Result<DynTensor> {
    let dtype = x.dtype();
    let st =
        ScalarType::try_from(dtype).map_err(|_| TensorError::dtype_mismatch(DType::F32, dtype))?;
    let scalar_type = st.msl_str();
    let elem_bytes = st.byte_size();

    let dims = x.dims();
    if dims.len() != 3 {
        return Err(TensorError::InvalidShape(
            "conv_with_precomputed_stats: rank 3 input required".into(),
        ));
    }
    let (batch, in_channels, in_len) = (dims[0], dims[1], dims[2]);
    let w_dims = weight.dims();
    let (out_channels, kernel_size) = (w_dims[0], w_dims[2]);

    let effective_k = (kernel_size - 1) * dilation + 1;
    let padded = in_len + 2 * padding;
    if padded < effective_k {
        return Err(TensorError::InvalidShape(format!(
            "conv_precomputed: padded {padded} < effective_k {effective_k}"
        )));
    }
    let out_len = padded - effective_k + 1;

    let x_data = x.gpu_data::<MetalTensorData>()?;
    let gamma_data = gamma.gpu_data::<MetalTensorData>()?;
    let beta_data = beta.gpu_data::<MetalTensorData>()?;
    let weight_data = weight.gpu_data::<MetalTensorData>()?;
    let bias_data = bias.gpu_data::<MetalTensorData>()?;

    let ctx = super::super::MetalDynBackend::ctx()?;
    super::super::with_pipeline_cache(|cache| {
        let conv_msl = activation.standard_msl_source(scalar_type);
        let conv_name = activation.standard_kernel_name(scalar_type);
        let ks_u32 = crate::to_u32(kernel_size, "conv_pre ks")?;
        let pad_u32 = crate::to_u32(padding, "conv_pre pad")?;
        let dil_u32 = crate::to_u32(dilation, "conv_pre dil")?;
        let fc = [(0u32, ks_u32), (1, pad_u32), (2, dil_u32)];
        let conv_pipe = KernelPipeline::from_msl_specialized(
            cache,
            &conv_msl,
            &conv_name,
            activation.input_buffer_count(),
            false,
            &fc,
        )
        .map_err(metal_err)?;

        let out_shape = vec![batch, out_channels, out_len];
        let total_out = batch
            .checked_mul(out_channels)
            .and_then(|v| v.checked_mul(out_len))
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: out_shape.clone(),
            })?;
        let out_bytes =
            total_out
                .checked_mul(elem_bytes)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: out_shape.clone(),
                })?;
        let (out_buf, out_off) =
            crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;

        let batch_u32 = crate::to_u32(batch, "conv_pre batch")?;
        let in_ch_u32 = crate::to_u32(in_channels, "conv_pre in_ch")?;
        let out_ch_u32 = crate::to_u32(out_channels, "conv_pre out_ch")?;
        let in_len_u32 = crate::to_u32(in_len, "conv_pre in_len")?;
        let out_len_u32 = crate::to_u32(out_len, "conv_pre out_len")?;
        // kernel_size, padding, dilation are function constants — baked
        // into the pipeline at creation time (#3449).
        let flat_out_rows =
            batch
                .checked_mul(out_channels)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: out_shape.clone(),
                })?;
        let out_rows_u32 = crate::to_u32(flat_out_rows, "conv_pre out_rows")?;
        let grid_x = (out_len as u32).div_ceil(CONV_TG_X as u32);

        let (has_res_u32, res_scale_f32, res_data) = match &residual {
            Some(p) => {
                let rd = p.residual.gpu_data::<MetalTensorData>()?;
                (1u32, p.scale, Some(rd))
            }
            None => (0u32, 1.0f32, None),
        };

        crate::gpu_scope::get_or_create_batch()?;
        let enc_conv = |b: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
            let enc = b.new_encoder()?;
            enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
            enc.set_buffer_with_offset(1, &precomputed.buffer, precomputed.offset);
            enc.set_buffer_with_offset(2, &gamma_data.buffer, gamma_data.byte_offset);
            enc.set_buffer_with_offset(3, &beta_data.buffer, beta_data.byte_offset);
            enc.set_buffer_with_offset(4, &weight_data.buffer, weight_data.byte_offset);
            enc.set_buffer_with_offset(5, &bias_data.buffer, bias_data.byte_offset);
            if let Some(rd) = res_data {
                enc.set_buffer_with_offset(6, &rd.buffer, rd.byte_offset);
            } else {
                enc.set_buffer_with_offset(6, &out_buf, out_off);
            }
            enc.set_buffer_with_offset(7, &out_buf, out_off);
            enc.set_bytes(8, &batch_u32);
            enc.set_bytes(9, &in_ch_u32);
            enc.set_bytes(10, &out_ch_u32);
            enc.set_bytes(11, &in_len_u32);
            enc.set_bytes(12, &out_len_u32);
            // kernel_size, padding, dilation are function constants (#3449).
            // Buffer 13: activation-specific binding.
            match &activation {
                StatsActivation::LeakyRelu { slope } => {
                    enc.set_bytes(13, slope);
                    enc.set_bytes(14, &has_res_u32);
                    enc.set_bytes(15, &res_scale_f32);
                }
                StatsActivation::Snake { alpha_data } => {
                    enc.set_buffer_with_offset(13, &alpha_data.buffer, alpha_data.byte_offset);
                    enc.set_bytes(14, &has_res_u32);
                    enc.set_bytes(15, &res_scale_f32);
                }
            }
            enc.encode_threadgroups(
                conv_pipe.pipeline(),
                [grid_x, out_rows_u32, 1],
                [CONV_TG_X as u32, 1, 1],
            )?;
            enc.end_encoding();
            Ok(())
        };
        match crate::gpu_scope::encode_into_lazy_batch(|b| enc_conv(b)) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(metal_err(e)),
            Err(e) => return Err(e),
        }

        let storage = MetalTensorData::from_arena_alloc(out_buf, out_off);
        DynTensor::from_gpu_storage(out_shape, dtype, Arc::new(storage), Device::metal())
    })
}
