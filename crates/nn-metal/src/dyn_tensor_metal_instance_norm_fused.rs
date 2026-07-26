// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused InstanceNorm GPU kernel using threadgroup parallel reduction (#2472).
//!
//! Replaces the 7-dispatch IR decomposition with a single Metal compute
//! dispatch that keeps all intermediates in threadgroup memory.
//!
//! Used by:
//! - `NativeOpKind::InstanceNorm` in the compiled model path
//! - `gpu_instance_norm_fused` in the eager DynTensor path
//!
//! Input: `[B, C, *spatial]` → reshape to `[B*C, spatial_flat]`.
//! One threadgroup per (B,C) pair. 256 threads per threadgroup.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};
use nn_dsl::ir::ScalarType;

use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

#[path = "dyn_tensor_metal_instance_norm_fused_msl.rs"]
mod msl;

/// Threadgroup size for the fused InstanceNorm kernel.
const TG_SIZE: usize = 256;

impl super::MetalDynBackend {
    /// Fused InstanceNorm using a single Metal compute dispatch with
    /// threadgroup parallel reduction.
    ///
    /// Equivalent to `gpu_instance_norm` but uses 1 dispatch instead of 7.
    /// Input `[B, C, *spatial]`. No learnable affine parameters.
    pub(in super::super) fn gpu_instance_norm_fused(x: &DynTensor, eps: f64) -> Result<DynTensor> {
        let dtype = x.dtype();
        let st = ScalarType::try_from(dtype)
            .map_err(|_| TensorError::dtype_mismatch(DType::F32, dtype))?;
        let scalar_type = st.msl_str();
        let elem_bytes = st.byte_size();

        let dims = x.dims();
        if dims.len() < 3 {
            return Err(TensorError::InvalidShape(
                "gpu_instance_norm_fused requires rank >= 3".into(),
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
                "gpu_instance_norm_fused: eps must be finite and positive, got {eps}"
            )));
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let total_elems =
            flat_rows
                .checked_mul(spatial)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;

        let ctx = Self::ctx()?;
        super::with_pipeline_cache(|cache| {
            let kernel_name = format!("fused_instance_norm_{scalar_type}");
            let msl_src = msl::fused_instance_norm_msl(scalar_type);
            let pipeline = KernelPipeline::from_msl(
                cache,
                &msl_src,
                &kernel_name,
                1, // 1 input buffer
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

            let flat_rows_u32 = crate::to_u32(flat_rows, "instance_norm flat_rows")?;
            let spatial_u32 = crate::to_u32(spatial, "instance_norm spatial")?;
            // Always dispatch full TG_SIZE threads. Threads beyond spatial_len
            // produce zero-contribution Kahan partial sums which sum to zero
            // in the simd reduction. Power-of-2 threadgroup size ensures full
            // simdgroups (32 threads each). #2685
            let tg_size_u32 = TG_SIZE as u32;

            // Encode the fused kernel directly into the lazy command batch.
            // We use manual encoding (like cumsum) because we need a float
            // constant (eps) which the DispatchPlan constants API doesn't support.
            let encode =
                |batch: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                    let enc = batch.new_encoder()?;
                    enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                    enc.set_buffer_with_offset(1, &out_buf, out_offset);
                    enc.set_bytes(2, &spatial_u32);
                    enc.set_bytes(3, &eps_f32);
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

/// MSL source for pre-compilation: fused InstanceNorm F32 kernel.
pub(crate) fn instance_norm_msl_source() -> String {
    msl::fused_instance_norm_msl("float")
}

/// MSL source for pre-compilation: fused InstanceNorm F16 kernel.
pub(crate) fn instance_norm_f16_msl_source() -> String {
    msl::fused_instance_norm_msl("half")
}
