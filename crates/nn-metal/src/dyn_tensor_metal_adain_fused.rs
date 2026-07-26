// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused AdaIN+Snake and AdaIN+LeakyRelu GPU kernels (#2472).
//!
//! Each kernel combines InstanceNorm + style affine + activation into a
//! single Metal compute dispatch using threadgroup parallel reductions.
//! This replaces ~20 separate Metal dispatches per AdaIN call.
//!
//! Used by:
//! - `NativeOpKind::AdainSnake` / `NativeOpKind::AdainLeakyRelu` in compiled model
//! - `gpu_adain_snake_fused` / `gpu_adain_leaky_relu_fused` in the eager path
//!
//! Input x: `[B, C, *spatial]` → reshape to `[B*C, spatial_flat]`.
//! gamma/beta: `[B, C, 1]` → flatten to `[B*C]`.
//! alpha (Snake only): `[C]` (per-channel, not per-batch).

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};
use nn_dsl::ir::ScalarType;

use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

#[path = "dyn_tensor_metal_adain_fused_msl.rs"]
mod msl;

/// Threadgroup size for fused AdaIN kernels.
const TG_SIZE: usize = 256;

impl super::MetalDynBackend {
    /// Fused AdaIN+Snake using a single Metal dispatch.
    ///
    /// Equivalent to `InstanceNorm(x) → affine(gamma, beta) → Snake(alpha)`
    /// but uses 1 dispatch instead of ~20.
    ///
    /// Supports F32 and F16 I/O: input/output buffers use the tensor's dtype,
    /// internal computation (mean, variance, activation) stays F32 for precision.
    /// Part of #3766 F16 I/O for Kokoro Generator.
    ///
    /// - x: `[B, C, *spatial]` (F32 or F16)
    /// - gamma: `[B, C, 1]` (F32 or F16, style-conditioned scale)
    /// - beta: `[B, C, 1]` (F32 or F16, style-conditioned shift)
    /// - alpha: `[C]` (F32 or F16, per-channel Snake parameter)
    /// - eps: InstanceNorm epsilon
    /// - residual_gamma: if true, `(1+g)*normed+b`; if false, `g*normed+b`. Part of #3257.
    pub(in super::super) fn gpu_adain_snake_fused(
        x: &DynTensor,
        gamma: &DynTensor,
        beta: &DynTensor,
        alpha: &DynTensor,
        eps: f64,
        residual_gamma: bool,
    ) -> Result<DynTensor> {
        let dtype = x.dtype();
        let st = ScalarType::try_from(dtype)
            .map_err(|_| TensorError::dtype_mismatch(DType::F32, dtype))?;
        let scalar_type = st.msl_str();
        let elem_bytes = st.byte_size();

        let dims = x.dims();
        if dims.len() < 3 {
            return Err(TensorError::InvalidShape(
                "gpu_adain_snake_fused requires rank >= 3".into(),
            ));
        }
        let batch = dims[0];
        let channels = dims[1];
        let spatial = checked_dim_product(&dims[2..])?;

        if spatial == 0 {
            return DynTensor::zeros(dims, dtype, &Device::metal());
        }

        let flat_rows =
            batch
                .checked_mul(channels)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;

        let eps_f32 = eps as f32;
        if !eps_f32.is_finite() || eps_f32 <= 0.0 {
            return Err(TensorError::InvalidShape(format!(
                "gpu_adain_snake_fused: eps must be finite and positive, got {eps}"
            )));
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let gamma_data = gamma.gpu_data::<MetalTensorData>()?;
        let beta_data = beta.gpu_data::<MetalTensorData>()?;
        let alpha_data = alpha.gpu_data::<MetalTensorData>()?;

        let total_elems =
            flat_rows
                .checked_mul(spatial)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;

        let ctx = Self::ctx()?;
        super::with_pipeline_cache(|cache| {
            // Entry point must match the MSL kernel function name exactly.
            // Cache differentiates via MSL source content (different for each
            // residual_gamma value), so the same entry point name is safe.
            let kernel_name = format!("fused_adain_snake_{scalar_type}");
            let msl_src = msl::fused_adain_snake_msl(scalar_type, residual_gamma);
            let pipeline = KernelPipeline::from_msl(
                cache,
                &msl_src,
                &kernel_name,
                4, // 4 input buffers: x, gamma, beta, alpha
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

            let spatial_u32 = crate::to_u32(spatial, "adain_snake spatial")?;
            let channels_u32 = crate::to_u32(channels, "adain_snake channels")?;
            let flat_rows_u32 = crate::to_u32(flat_rows, "adain_snake flat_rows")?;
            // Always dispatch full TG_SIZE threads. Threads beyond spatial_len
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
                    enc.set_buffer_with_offset(3, &alpha_data.buffer, alpha_data.byte_offset);
                    enc.set_buffer_with_offset(4, &out_buf, out_offset);
                    enc.set_bytes(5, &spatial_u32);
                    enc.set_bytes(6, &channels_u32);
                    enc.set_bytes(7, &eps_f32);
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

    /// Fused AdaIN+LeakyRelu using a single Metal dispatch.
    ///
    /// Equivalent to `InstanceNorm(x) → (1+gamma)*normed+beta → LeakyRelu(slope)`
    /// but uses 1 dispatch instead of ~20.
    ///
    /// Supports F32 and F16 I/O: input/output buffers use the tensor's dtype,
    /// internal computation (mean, variance, activation) stays F32 for precision.
    /// Part of #3766 F16 I/O for Kokoro Generator.
    ///
    /// - x: `[B, C, *spatial]` (F32 or F16)
    /// - gamma: `[B, C, 1]` (F32 or F16, style-conditioned scale)
    /// - beta: `[B, C, 1]` (F32 or F16, style-conditioned shift)
    /// - eps: InstanceNorm epsilon
    /// - slope: LeakyRelu negative slope
    pub(in super::super) fn gpu_adain_leaky_relu_fused(
        x: &DynTensor,
        gamma: &DynTensor,
        beta: &DynTensor,
        eps: f64,
        slope: f64,
    ) -> Result<DynTensor> {
        let dtype = x.dtype();
        let st = ScalarType::try_from(dtype)
            .map_err(|_| TensorError::dtype_mismatch(DType::F32, dtype))?;
        let scalar_type = st.msl_str();
        let elem_bytes = st.byte_size();

        let dims = x.dims();
        if dims.len() < 3 {
            return Err(TensorError::InvalidShape(
                "gpu_adain_leaky_relu_fused requires rank >= 3".into(),
            ));
        }
        let batch = dims[0];
        let channels = dims[1];
        let spatial = checked_dim_product(&dims[2..])?;

        if spatial == 0 {
            return DynTensor::zeros(dims, dtype, &Device::metal());
        }

        let flat_rows =
            batch
                .checked_mul(channels)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;

        let eps_f32 = eps as f32;
        if !eps_f32.is_finite() || eps_f32 <= 0.0 {
            return Err(TensorError::InvalidShape(format!(
                "gpu_adain_leaky_relu_fused: eps must be finite and positive, got {eps}"
            )));
        }
        let slope_f32 = slope as f32;
        if !slope_f32.is_finite() {
            return Err(TensorError::InvalidShape(format!(
                "gpu_adain_leaky_relu_fused: slope must be finite, got {slope}"
            )));
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let gamma_data = gamma.gpu_data::<MetalTensorData>()?;
        let beta_data = beta.gpu_data::<MetalTensorData>()?;

        let total_elems =
            flat_rows
                .checked_mul(spatial)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;

        let ctx = Self::ctx()?;
        super::with_pipeline_cache(|cache| {
            let kernel_name = format!("fused_adain_leaky_relu_{scalar_type}");
            let msl_src = msl::fused_adain_leaky_relu_msl(scalar_type);
            let pipeline = KernelPipeline::from_msl(
                cache,
                &msl_src,
                &kernel_name,
                3, // 3 input buffers: x, gamma, beta
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

            let spatial_u32 = crate::to_u32(spatial, "adain_leaky_relu spatial")?;
            let flat_rows_u32 = crate::to_u32(flat_rows, "adain_leaky_relu flat_rows")?;
            // Always dispatch full TG_SIZE threads. Threads beyond spatial_len
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
                    enc.set_buffer_with_offset(3, &out_buf, out_offset);
                    enc.set_bytes(4, &spatial_u32);
                    enc.set_bytes(5, &eps_f32);
                    enc.set_bytes(6, &slope_f32);
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

/// MSL source for pre-compilation: fused AdaIN+Snake kernel (F32, residual gamma).
pub(crate) fn adain_snake_msl_source() -> String {
    msl::fused_adain_snake_msl("float", true)
}

/// MSL source for pre-compilation: fused AdaIN+Snake kernel (F16, residual gamma).
pub(crate) fn adain_snake_f16_msl_source() -> String {
    msl::fused_adain_snake_msl("half", true)
}

/// MSL source for pre-compilation: fused AdaIN+LeakyRelu kernel (F32).
pub(crate) fn adain_leaky_relu_msl_source() -> String {
    msl::fused_adain_leaky_relu_msl("float")
}

/// MSL source for pre-compilation: fused AdaIN+LeakyRelu kernel (F16).
pub(crate) fn adain_leaky_relu_f16_msl_source() -> String {
    msl::fused_adain_leaky_relu_msl("half")
}
