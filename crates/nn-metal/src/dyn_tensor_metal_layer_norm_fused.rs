// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused LayerNorm GPU kernel.
//!
//! Combines mean+variance reduction, normalization, and affine transform
//! `weight * normed + bias` into a single Metal compute dispatch using
//! threadgroup parallel reductions. Replaces ~14 separate Metal dispatches
//! per LayerNorm call.
//!
//! Used by:
//! - `NativeOpKind::LayerNorm` in compiled model
//! - Kokoro PlBert (25 LayerNorms per forward)
//! - Kokoro TextEncoder (3 LayerNorms per forward)
//!
//! Input x: any rank >= 2, normalized over the last dimension.
//! weight/bias: `[hidden_dim]` (LayerNorm learnable params).

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};
use nn_dsl::ir::ScalarType;

use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

#[path = "dyn_tensor_metal_layer_norm_fused_msl.rs"]
mod msl;

/// Threadgroup size for fused LayerNorm kernel.
const TG_SIZE: usize = 256;

impl super::MetalDynBackend {
    /// Fused LayerNorm using a single Metal dispatch.
    ///
    /// Equivalent to `(x - mean) / sqrt(var + eps) * weight + bias`
    /// but uses 1 dispatch instead of ~14.
    ///
    /// - x: any rank >= 2 (F32), normalized over the last dimension
    /// - weight: `[hidden_dim]` (F32, LayerNorm learnable scale)
    /// - bias: `[hidden_dim]` (F32, LayerNorm learnable bias)
    /// - eps: LayerNorm epsilon
    pub(in super::super) fn gpu_layer_norm_fused(
        x: &DynTensor,
        weight: &DynTensor,
        bias: &DynTensor,
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
                "gpu_layer_norm_fused requires rank >= 2".into(),
            ));
        }
        let hidden_dim = *dims.last().unwrap_or(&0);
        let flat_rows = checked_dim_product(&dims[..dims.len() - 1])?;

        if hidden_dim == 0 || flat_rows == 0 {
            return DynTensor::zeros(dims, dtype, &Device::metal());
        }

        let eps_f32 = eps as f32;
        if !eps_f32.is_finite() || eps_f32 < 0.0 {
            return Err(TensorError::InvalidShape(format!(
                "gpu_layer_norm_fused: eps must be finite and non-negative, got {eps}"
            )));
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let w_data = weight.gpu_data::<MetalTensorData>()?;
        let b_data = bias.gpu_data::<MetalTensorData>()?;

        let total_elems =
            flat_rows
                .checked_mul(hidden_dim)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;

        let ctx = Self::ctx()?;
        super::with_pipeline_cache(|cache| {
            let kernel_name = format!("fused_layer_norm_{scalar_type}");
            let msl_src = msl::fused_layer_norm_msl(scalar_type);
            let pipeline = KernelPipeline::from_msl(
                cache,
                &msl_src,
                &kernel_name,
                3, // 3 input buffers: x, weight, bias
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

            let hidden_dim_u32 = crate::to_u32(hidden_dim, "layer_norm hidden_dim")?;
            let flat_rows_u32 = crate::to_u32(flat_rows, "layer_norm flat_rows")?;
            let tg_size_u32 = TG_SIZE as u32;

            let encode =
                |batch: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                    let enc = batch.new_encoder()?;
                    enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                    enc.set_buffer_with_offset(1, &w_data.buffer, w_data.byte_offset);
                    enc.set_buffer_with_offset(2, &b_data.buffer, b_data.byte_offset);
                    enc.set_buffer_with_offset(3, &out_buf, out_offset);
                    enc.set_bytes(4, &hidden_dim_u32);
                    enc.set_bytes(5, &eps_f32);
                    enc.encode_threadgroups(
                        pipeline.pipeline(),
                        [flat_rows_u32, 1, 1],
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

/// MSL source for pre-compilation: fused LayerNorm F32 kernel.
pub(crate) fn layer_norm_msl_source() -> String {
    msl::fused_layer_norm_msl("float")
}

/// MSL source for pre-compilation: fused LayerNorm F16 kernel.
pub(crate) fn layer_norm_f16_msl_source() -> String {
    msl::fused_layer_norm_msl("half")
}
