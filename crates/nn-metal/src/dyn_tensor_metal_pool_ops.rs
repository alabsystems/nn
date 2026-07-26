// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-native 2-D pooling implementations for [`MetalDynBackend`].
//!
//! - `gpu_max_pool2d`: sliding window max over 2D spatial dims
//! - `gpu_avg_pool2d`: sliding window mean (count_include_pad=false)
//! - `gpu_adaptive_avg_pool2d`: adaptive average pool to target output dims
//!
//! Eliminates the GPU→CPU→GPU round-trip for pool operations on Metal.
//! Part of #4323.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};
use nn_dsl::ir::ScalarType;

use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

#[path = "dyn_tensor_metal_pool_ops_msl.rs"]
mod msl;

/// Pool2d parameter struct. Layout must match MSL `Pool2dParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::NoUninit)]
struct Pool2dParams {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    /// Pre-validated total output element count (batch * channels * out_h * out_w).
    /// Passed from Rust to avoid uint32 overflow in MSL multiplication.
    total_elements: u32,
}

/// Adaptive pool2d parameter struct. Layout must match MSL `AdaptivePool2dParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::NoUninit)]
struct AdaptivePool2dParams {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    /// Pre-validated total output element count (batch * channels * out_h * out_w).
    /// Passed from Rust to avoid uint32 overflow in MSL multiplication.
    total_elements: u32,
}

/// Threadgroup width for pool kernels. Each thread handles one output element.
const TG_SIZE: u32 = 256;

impl super::MetalDynBackend {
    /// GPU-native 2-D max pooling.
    ///
    /// Input shape: `[batch, channels, height, width]`
    /// Output shape: `[batch, channels, out_h, out_w]`
    pub(in super::super) fn gpu_max_pool2d(
        x: &DynTensor,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<DynTensor> {
        let dtype = x.dtype();
        let st = ScalarType::try_from(dtype)
            .map_err(|_| TensorError::dtype_mismatch(DType::F32, dtype))?;
        let scalar_type = st.msl_str();
        let elem_bytes = st.byte_size();

        let dims = x.dims();
        if dims.len() != 4 {
            return Err(TensorError::InvalidShape(format!(
                "gpu_max_pool2d requires rank 4, got {}",
                dims.len()
            )));
        }
        let (batch, channels, in_h, in_w) = (dims[0], dims[1], dims[2], dims[3]);

        let out_h = pool2d_out_len(in_h, kernel_size, padding, stride)?;
        let out_w = pool2d_out_len(in_w, kernel_size, padding, stride)?;
        let out_shape = [batch, channels, out_h, out_w];
        let total_out = checked_dim_product(&out_shape)?;

        if total_out == 0 {
            return DynTensor::zeros(&out_shape, dtype, &Device::metal());
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let ctx = Self::ctx()?;

        super::with_pipeline_cache(|cache| {
            let kernel_name = format!("max_pool2d_{scalar_type}");
            let msl_src = msl::max_pool2d_msl(scalar_type);
            let pipeline = KernelPipeline::from_msl(
                cache,
                &msl_src,
                &kernel_name,
                1, // 1 input buffer
                false,
            )
            .map_err(metal_err)?;

            let out_bytes = total_out.checked_mul(elem_bytes).ok_or_else(|| {
                TensorError::DimensionOverflow {
                    dims: out_shape.to_vec(),
                }
            })?;
            let (out_buf, out_offset) =
                crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;

            let total_threads = crate::to_u32(total_out, "pool total_threads")?;

            let params = Pool2dParams {
                batch: crate::to_u32(batch, "pool batch")?,
                channels: crate::to_u32(channels, "pool channels")?,
                in_h: crate::to_u32(in_h, "pool in_h")?,
                in_w: crate::to_u32(in_w, "pool in_w")?,
                out_h: crate::to_u32(out_h, "pool out_h")?,
                out_w: crate::to_u32(out_w, "pool out_w")?,
                kernel_size: crate::to_u32(kernel_size, "pool kernel_size")?,
                stride: crate::to_u32(stride, "pool stride")?,
                padding: crate::to_u32(padding, "pool padding")?,
                total_elements: total_threads,
            };

            let grid = [total_threads, 1, 1];
            let tg = [TG_SIZE, 1, 1];

            let encode =
                |batch_cmd: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                    let enc = batch_cmd.new_encoder()?;
                    enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                    enc.set_buffer_with_offset(1, &out_buf, out_offset);
                    enc.set_bytes(2, &params);
                    enc.encode(pipeline.pipeline(), grid, tg)?;
                    enc.end_encoding();
                    Ok(())
                };

            crate::gpu_scope::get_or_create_batch()?;
            let scope_result = crate::gpu_scope::encode_into_lazy_batch(|b| encode(b));
            match scope_result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(metal_err(e)),
                Err(e) => return Err(e),
            }

            let storage = MetalTensorData::from_arena_alloc(out_buf, out_offset);
            DynTensor::from_gpu_storage(
                out_shape.to_vec(),
                dtype,
                Arc::new(storage),
                Device::metal(),
            )
        })
    }

    /// GPU-native 2-D average pooling (count_include_pad=false).
    ///
    /// Input shape: `[batch, channels, height, width]`
    /// Output shape: `[batch, channels, out_h, out_w]`
    pub(in super::super) fn gpu_avg_pool2d(
        x: &DynTensor,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<DynTensor> {
        let dtype = x.dtype();
        let st = ScalarType::try_from(dtype)
            .map_err(|_| TensorError::dtype_mismatch(DType::F32, dtype))?;
        let scalar_type = st.msl_str();
        let elem_bytes = st.byte_size();

        let dims = x.dims();
        if dims.len() != 4 {
            return Err(TensorError::InvalidShape(format!(
                "gpu_avg_pool2d requires rank 4, got {}",
                dims.len()
            )));
        }
        let (batch, channels, in_h, in_w) = (dims[0], dims[1], dims[2], dims[3]);

        let out_h = pool2d_out_len(in_h, kernel_size, padding, stride)?;
        let out_w = pool2d_out_len(in_w, kernel_size, padding, stride)?;
        let out_shape = [batch, channels, out_h, out_w];
        let total_out = checked_dim_product(&out_shape)?;

        if total_out == 0 {
            return DynTensor::zeros(&out_shape, dtype, &Device::metal());
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let ctx = Self::ctx()?;

        super::with_pipeline_cache(|cache| {
            let kernel_name = format!("avg_pool2d_{scalar_type}");
            let msl_src = msl::avg_pool2d_msl(scalar_type);
            let pipeline = KernelPipeline::from_msl(
                cache,
                &msl_src,
                &kernel_name,
                1,
                false,
            )
            .map_err(metal_err)?;

            let out_bytes = total_out.checked_mul(elem_bytes).ok_or_else(|| {
                TensorError::DimensionOverflow {
                    dims: out_shape.to_vec(),
                }
            })?;
            let (out_buf, out_offset) =
                crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;

            let total_threads = crate::to_u32(total_out, "pool total_threads")?;

            let params = Pool2dParams {
                batch: crate::to_u32(batch, "pool batch")?,
                channels: crate::to_u32(channels, "pool channels")?,
                in_h: crate::to_u32(in_h, "pool in_h")?,
                in_w: crate::to_u32(in_w, "pool in_w")?,
                out_h: crate::to_u32(out_h, "pool out_h")?,
                out_w: crate::to_u32(out_w, "pool out_w")?,
                kernel_size: crate::to_u32(kernel_size, "pool kernel_size")?,
                stride: crate::to_u32(stride, "pool stride")?,
                padding: crate::to_u32(padding, "pool padding")?,
                total_elements: total_threads,
            };

            let grid = [total_threads, 1, 1];
            let tg = [TG_SIZE, 1, 1];

            let encode =
                |batch_cmd: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                    let enc = batch_cmd.new_encoder()?;
                    enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                    enc.set_buffer_with_offset(1, &out_buf, out_offset);
                    enc.set_bytes(2, &params);
                    enc.encode(pipeline.pipeline(), grid, tg)?;
                    enc.end_encoding();
                    Ok(())
                };

            crate::gpu_scope::get_or_create_batch()?;
            let scope_result = crate::gpu_scope::encode_into_lazy_batch(|b| encode(b));
            match scope_result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(metal_err(e)),
                Err(e) => return Err(e),
            }

            let storage = MetalTensorData::from_arena_alloc(out_buf, out_offset);
            DynTensor::from_gpu_storage(
                out_shape.to_vec(),
                dtype,
                Arc::new(storage),
                Device::metal(),
            )
        })
    }

    /// GPU-native adaptive 2-D average pooling.
    ///
    /// Input shape: `[batch, channels, height, width]`
    /// Output shape: `[batch, channels, out_h, out_w]`
    ///
    /// Uses PyTorch ATen window formula: `start = (oh * in_h) / out_h`,
    /// `end = ceil((oh + 1) * in_h / out_h)`.
    pub(in super::super) fn gpu_adaptive_avg_pool2d(
        x: &DynTensor,
        out_h: usize,
        out_w: usize,
    ) -> Result<DynTensor> {
        let dtype = x.dtype();
        let st = ScalarType::try_from(dtype)
            .map_err(|_| TensorError::dtype_mismatch(DType::F32, dtype))?;
        let scalar_type = st.msl_str();
        let elem_bytes = st.byte_size();

        let dims = x.dims();
        if dims.len() != 4 {
            return Err(TensorError::InvalidShape(format!(
                "gpu_adaptive_avg_pool2d requires rank 4, got {}",
                dims.len()
            )));
        }
        let (batch, channels, in_h, in_w) = (dims[0], dims[1], dims[2], dims[3]);

        if out_h == 0 || out_w == 0 {
            return Err(TensorError::InvalidShape(
                "gpu_adaptive_avg_pool2d: output dimensions must be > 0".into(),
            ));
        }

        let out_shape = [batch, channels, out_h, out_w];
        let total_out = checked_dim_product(&out_shape)?;

        if total_out == 0 {
            return DynTensor::zeros(&out_shape, dtype, &Device::metal());
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let ctx = Self::ctx()?;

        super::with_pipeline_cache(|cache| {
            let kernel_name = format!("adaptive_avg_pool2d_{scalar_type}");
            let msl_src = msl::adaptive_avg_pool2d_msl(scalar_type);
            let pipeline = KernelPipeline::from_msl(
                cache,
                &msl_src,
                &kernel_name,
                1,
                false,
            )
            .map_err(metal_err)?;

            let out_bytes = total_out.checked_mul(elem_bytes).ok_or_else(|| {
                TensorError::DimensionOverflow {
                    dims: out_shape.to_vec(),
                }
            })?;
            let (out_buf, out_offset) =
                crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;

            let total_threads = crate::to_u32(total_out, "pool total_threads")?;

            let params = AdaptivePool2dParams {
                batch: crate::to_u32(batch, "pool batch")?,
                channels: crate::to_u32(channels, "pool channels")?,
                in_h: crate::to_u32(in_h, "pool in_h")?,
                in_w: crate::to_u32(in_w, "pool in_w")?,
                out_h: crate::to_u32(out_h, "pool out_h")?,
                out_w: crate::to_u32(out_w, "pool out_w")?,
                total_elements: total_threads,
            };
            let grid = [total_threads, 1, 1];
            let tg = [TG_SIZE, 1, 1];

            let encode =
                |batch_cmd: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                    let enc = batch_cmd.new_encoder()?;
                    enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                    enc.set_buffer_with_offset(1, &out_buf, out_offset);
                    enc.set_bytes(2, &params);
                    enc.encode(pipeline.pipeline(), grid, tg)?;
                    enc.end_encoding();
                    Ok(())
                };

            crate::gpu_scope::get_or_create_batch()?;
            let scope_result = crate::gpu_scope::encode_into_lazy_batch(|b| encode(b));
            match scope_result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(metal_err(e)),
                Err(e) => return Err(e),
            }

            let storage = MetalTensorData::from_arena_alloc(out_buf, out_offset);
            DynTensor::from_gpu_storage(
                out_shape.to_vec(),
                dtype,
                Arc::new(storage),
                Device::metal(),
            )
        })
    }
}

/// Compute the output length for a pooling dimension (floor mode).
///
/// `out = (input + 2*padding - kernel_size) / stride + 1`
fn pool2d_out_len(
    input_len: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
) -> Result<usize> {
    if kernel_size == 0 || stride == 0 {
        return Err(TensorError::InvalidShape(
            "pool2d: kernel_size and stride must be > 0".into(),
        ));
    }
    let padded = input_len
        .checked_add(2usize.checked_mul(padding).ok_or_else(|| {
            TensorError::InvalidShape(format!("pool2d: padding overflow (padding={padding})"))
        })?)
        .ok_or_else(|| {
            TensorError::InvalidShape(format!(
                "pool2d: padded input overflow (input_len={input_len}, padding={padding})"
            ))
        })?;
    if padded < kernel_size {
        return Err(TensorError::InvalidShape(format!(
            "pool2d: padded input {padded} < kernel_size {kernel_size}"
        )));
    }
    Ok((padded - kernel_size) / stride + 1)
}
