// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused AdaLayerNorm GPU kernel (#2482).
//!
//! Combines LayerNorm + adaptive affine `(1+gamma)*normed+beta` into a
//! single Metal compute dispatch using threadgroup parallel reductions.
//! This replaces ~6-7 separate Metal dispatches per AdaLayerNorm call.
//!
//! Used by:
//! - `NativeOpKind::AdaLayerNorm` in compiled model
//! - Kokoro ProsodyPredictor ProsodyBlock (2 per forward)
//!
//! Input x: `[B, T, C]` → reshape to `[B*T, C]`.
//! gamma/beta: `[B, 1, C]` → squeeze to `[B, C]`.
//! norm_weight/norm_bias: `[C]` (LayerNorm learnable params).

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};
use nn_dsl::ir::ScalarType;

use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

#[path = "dyn_tensor_metal_adaln_fused_msl.rs"]
mod msl;

/// Threadgroup size for fused AdaLayerNorm kernel.
const TG_SIZE: usize = 256;

impl super::MetalDynBackend {
    /// Fused AdaLayerNorm using a single Metal dispatch.
    ///
    /// Equivalent to `LayerNorm(x, w, b, eps) → (1+gamma)*normed+beta`
    /// but uses 1 dispatch instead of ~6-7.
    ///
    /// - x: `[B, T, C]` (F32)
    /// - gamma: `[B, 1, C]` (F32, adaptive scale from style projection)
    /// - beta: `[B, 1, C]` (F32, adaptive shift from style projection)
    /// - norm_weight: `[C]` (F32, LayerNorm learnable scale)
    /// - norm_bias: `[C]` (F32, LayerNorm learnable bias)
    /// - eps: LayerNorm epsilon
    /// - time_steps: T dimension (for batch index computation in kernel)
    pub(in super::super) fn gpu_ada_layer_norm_fused(
        x: &DynTensor,
        gamma: &DynTensor,
        beta: &DynTensor,
        norm_weight: &DynTensor,
        norm_bias: &DynTensor,
        eps: f64,
        time_steps: usize,
    ) -> Result<DynTensor> {
        let dtype = x.dtype();
        let st = ScalarType::try_from(dtype)
            .map_err(|_| TensorError::dtype_mismatch(DType::F32, dtype))?;
        let scalar_type = st.msl_str();
        let elem_bytes = st.byte_size();

        let dims = x.dims();
        if dims.len() < 3 {
            return Err(TensorError::InvalidShape(
                "gpu_ada_layer_norm_fused requires rank >= 3".into(),
            ));
        }
        let batch = dims[0];
        let hidden_dim = *dims.last().unwrap_or(&0);
        // Middle dimensions (T or T1*T2*...) are the time/spatial dims.
        let mid_dims = checked_dim_product(&dims[1..dims.len() - 1])?;

        if hidden_dim == 0 || mid_dims == 0 {
            return DynTensor::zeros(dims, dtype, &Device::metal());
        }

        let flat_rows =
            batch
                .checked_mul(mid_dims)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;

        let eps_f32 = eps as f32;
        if !eps_f32.is_finite() || eps_f32 <= 0.0 {
            return Err(TensorError::InvalidShape(format!(
                "gpu_ada_layer_norm_fused: eps must be finite and positive, got {eps}"
            )));
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let gamma_data = gamma.gpu_data::<MetalTensorData>()?;
        let beta_data = beta.gpu_data::<MetalTensorData>()?;
        let nw_data = norm_weight.gpu_data::<MetalTensorData>()?;
        let nb_data = norm_bias.gpu_data::<MetalTensorData>()?;

        let total_elems =
            flat_rows
                .checked_mul(hidden_dim)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;

        let ctx = Self::ctx()?;
        super::with_pipeline_cache(|cache| {
            let kernel_name = format!("fused_ada_layer_norm_{scalar_type}");
            let msl_src = msl::fused_ada_layer_norm_msl(scalar_type);
            let pipeline = KernelPipeline::from_msl(
                cache,
                &msl_src,
                &kernel_name,
                5, // 5 input buffers: x, gamma, beta, norm_weight, norm_bias
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

            let hidden_dim_u32 = crate::to_u32(hidden_dim, "adaln hidden_dim")?;
            let time_steps_u32 = crate::to_u32(time_steps, "adaln time_steps")?;
            let flat_rows_u32 = crate::to_u32(flat_rows, "adaln flat_rows")?;
            // Always dispatch full TG_SIZE threads. Threads beyond hidden_dim
            // produce zero-contribution reduction states which merge correctly.
            // This ensures the tree reduction stride-halving works (requires
            // power-of-2). #2685
            let tg_size_u32 = TG_SIZE as u32;

            let encode =
                |batch: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                    let enc = batch.new_encoder()?;
                    enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                    enc.set_buffer_with_offset(1, &gamma_data.buffer, gamma_data.byte_offset);
                    enc.set_buffer_with_offset(2, &beta_data.buffer, beta_data.byte_offset);
                    enc.set_buffer_with_offset(3, &nw_data.buffer, nw_data.byte_offset);
                    enc.set_buffer_with_offset(4, &nb_data.buffer, nb_data.byte_offset);
                    enc.set_buffer_with_offset(5, &out_buf, out_offset);
                    enc.set_bytes(6, &hidden_dim_u32);
                    enc.set_bytes(7, &time_steps_u32);
                    enc.set_bytes(8, &eps_f32);
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

/// MSL source for pre-compilation: fused AdaLayerNorm kernel (F32).
pub(crate) fn ada_layer_norm_msl_source() -> String {
    msl::fused_ada_layer_norm_msl("float")
}

/// MSL source for pre-compilation: fused AdaLayerNorm kernel (F16).
pub(crate) fn ada_layer_norm_f16_msl_source() -> String {
    msl::fused_ada_layer_norm_msl("half")
}
