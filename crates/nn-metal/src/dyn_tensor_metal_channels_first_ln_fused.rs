// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! Fused channels-first LayerNorm GPU kernel.
//!
//! Normalizes over dim 1 (channel dimension) of a `[B, C, T]` tensor.
//! Semantically equivalent to `Transpose(1,2) → LayerNorm → Transpose(1,2)`
//! but avoids two full-data-copy transpose dispatches.
//!
//! For each `(b, t)` position, the kernel:
//! 1. Reduces over C elements (strided by T) to compute mean and variance
//! 2. Normalizes: `(x[b,c,t] - mean) / sqrt(var + eps) * weight[c] + bias[c]`
//!
//! Input x: rank 3 `[B, C, T]`, weight/bias: `[C]`.
//! Part of #3457.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};
use nn_dsl::ir::ScalarType;

use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

/// Threadgroup size for channels-first LayerNorm kernel.
const TG_SIZE: usize = 256;

impl super::MetalDynBackend {
    /// Fused channels-first LayerNorm using a single Metal dispatch.
    ///
    /// Normalizes over dim 1 of `[B, C, T]` — equivalent to transposing to
    /// `[B, T, C]`, applying LayerNorm over the last dim, then transposing back.
    ///
    /// - x: rank 3 `[B, C, T]` (F32 or F16)
    /// - weight: `[C]` (LayerNorm scale)
    /// - bias: `[C]` (LayerNorm shift)
    /// - eps: LayerNorm epsilon
    pub(in super::super) fn gpu_channels_first_layer_norm_fused(
        x: &DynTensor,
        weight: &DynTensor,
        bias: &DynTensor,
        eps: f64,
        leaky_relu_slope: Option<f32>,
    ) -> Result<DynTensor> {
        let dtype = x.dtype();
        let st = ScalarType::try_from(dtype)
            .map_err(|_| TensorError::dtype_mismatch(DType::F32, dtype))?;
        let scalar_type = st.msl_str();
        let elem_bytes = st.byte_size();

        let dims = x.dims();
        if dims.len() != 3 {
            return Err(TensorError::InvalidShape(
                "gpu_channels_first_layer_norm_fused requires rank 3 [B, C, T]".into(),
            ));
        }
        let batch = dims[0];
        let channels = dims[1];
        let time_steps = dims[2];

        if channels == 0 || batch == 0 || time_steps == 0 {
            return DynTensor::zeros(dims, dtype, &Device::metal());
        }

        let eps_f32 = eps as f32;
        if !eps_f32.is_finite() || eps_f32 < 0.0 {
            return Err(TensorError::InvalidShape(format!(
                "gpu_channels_first_layer_norm_fused: eps must be finite and non-negative, got {eps}"
            )));
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let w_data = weight.gpu_data::<MetalTensorData>()?;
        let b_data = bias.gpu_data::<MetalTensorData>()?;

        let total_elems = checked_dim_product(dims)?;
        let flat_rows =
            batch
                .checked_mul(time_steps)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;

        let ctx = Self::ctx()?;
        let slope = leaky_relu_slope;
        super::with_pipeline_cache(|cache| {
            let (kernel_name, msl_src) = if slope.is_some() {
                (
                    format!("fused_channels_first_ln_leaky_relu_{scalar_type}"),
                    channels_first_layer_norm_leaky_relu_msl(scalar_type),
                )
            } else {
                (
                    format!("fused_channels_first_layer_norm_{scalar_type}"),
                    channels_first_layer_norm_msl(scalar_type),
                )
            };
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

            let channels_u32 = crate::to_u32(channels, "channels_first_ln channels")?;
            let time_steps_u32 = crate::to_u32(time_steps, "channels_first_ln time_steps")?;
            let flat_rows_u32 = crate::to_u32(flat_rows, "channels_first_ln flat_rows")?;
            let tg_size_u32 = TG_SIZE as u32;
            let slope_f32 = slope.unwrap_or(0.0);

            let encode =
                |batch: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                    let enc = batch.new_encoder()?;
                    enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                    enc.set_buffer_with_offset(1, &w_data.buffer, w_data.byte_offset);
                    enc.set_buffer_with_offset(2, &b_data.buffer, b_data.byte_offset);
                    enc.set_buffer_with_offset(3, &out_buf, out_offset);
                    enc.set_bytes(4, &channels_u32);
                    enc.set_bytes(5, &time_steps_u32);
                    enc.set_bytes(6, &eps_f32);
                    if slope.is_some() {
                        enc.set_bytes(7, &slope_f32);
                    }
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

/// MSL source for the fused channels-first LayerNorm kernel.
///
/// Normalizes over the channel dimension (dim 1) of a `[B, C, T]` tensor.
/// One threadgroup per `(b, t)` position. Each threadgroup reduces over C
/// elements at stride `T` (non-contiguous access pattern).
///
/// Buffers:
///   - 0: `input` — `[B, C, T]` (read-only)
///   - 1: `weight` — `[C]` (read-only, LayerNorm scale)
///   - 2: `bias` — `[C]` (read-only, LayerNorm shift)
///   - 3: `output` — `[B, C, T]` (write-only)
///
/// Constants (set_bytes):
///   - 4: `channels` — uint (C)
///   - 5: `time_steps` — uint (T)
///   - 6: `eps` — float
///
/// Dispatch: one threadgroup per (b, t) pair = B*T threadgroups.
fn channels_first_layer_norm_msl(scalar_type: &str) -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_channels_first_layer_norm_{scalar_type}(
    device const {scalar_type}* input      [[buffer(0)]],
    device const {scalar_type}* weight     [[buffer(1)]],
    device const {scalar_type}* bias       [[buffer(2)]],
    device {scalar_type}* output           [[buffer(3)]],
    constant uint& channels        [[buffer(4)]],
    constant uint& time_steps      [[buffer(5)]],
    constant float& eps            [[buffer(6)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    // gid = linear index into (B, T) space.
    // b = gid / time_steps, t = gid % time_steps.
    uint b = gid / time_steps;
    uint t = gid % time_steps;

    // Base offset for this (b, t) in the [B, C, T] tensor.
    // Element (b, c, t) is at index: b * C * T + c * T + t.
    uint bt_base = b * channels * time_steps + t;
    uint stride = time_steps; // stride between channel elements

    // --- Pass 1: Compute mean ---
    // Kahan-compensated sum for numerical stability.
    float sum_val = 0.0;
    float sum_comp = 0.0;
    for (uint c = tid; c < channels; c += tg_size) {{
        float v = float(input[bt_base + c * stride]);
        float y = v - sum_comp;
        float t_val = sum_val + y;
        sum_comp = (t_val - sum_val) - y;
        sum_val = t_val;
    }}

    // Threadgroup reduction for sum.
    threadgroup float shared_sum[256];
    shared_sum[tid] = sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            shared_sum[tid] += shared_sum[tid + s];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float mean = shared_sum[0] / float(channels);

    // --- Pass 2: Compute variance ---
    float var_sum = 0.0;
    float var_comp = 0.0;
    for (uint c = tid; c < channels; c += tg_size) {{
        float v = float(input[bt_base + c * stride]) - mean;
        float sq = v * v;
        float y = sq - var_comp;
        float t_val = var_sum + y;
        var_comp = (t_val - var_sum) - y;
        var_sum = t_val;
    }}

    threadgroup float shared_var[256];
    shared_var[tid] = var_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            shared_var[tid] += shared_var[tid + s];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float inv_std = metal::precise::rsqrt(shared_var[0] / float(channels) + eps);

    // --- Normalize + affine ---
    for (uint c = tid; c < channels; c += tg_size) {{
        float normed = (float(input[bt_base + c * stride]) - mean) * inv_std;
        output[bt_base + c * stride] = {scalar_type}(normed * float(weight[c]) + float(bias[c]));
    }}
}}
"#
    )
}

/// MSL source: fused channels-first LayerNorm + LeakyReLU.
///
/// Same as `channels_first_layer_norm_msl` but applies `val < 0 ? val * slope : val`
/// after normalization. Buffer 7 = `slope` (float).
fn channels_first_layer_norm_leaky_relu_msl(scalar_type: &str) -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_channels_first_ln_leaky_relu_{scalar_type}(
    device const {scalar_type}* input      [[buffer(0)]],
    device const {scalar_type}* weight     [[buffer(1)]],
    device const {scalar_type}* bias       [[buffer(2)]],
    device {scalar_type}* output           [[buffer(3)]],
    constant uint& channels        [[buffer(4)]],
    constant uint& time_steps      [[buffer(5)]],
    constant float& eps            [[buffer(6)]],
    constant float& slope          [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    uint b = gid / time_steps;
    uint t = gid % time_steps;
    uint bt_base = b * channels * time_steps + t;
    uint stride = time_steps;

    // --- Pass 1: Compute mean (Kahan-compensated) ---
    float sum_val = 0.0;
    float sum_comp = 0.0;
    for (uint c = tid; c < channels; c += tg_size) {{
        float v = float(input[bt_base + c * stride]);
        float y = v - sum_comp;
        float t_val = sum_val + y;
        sum_comp = (t_val - sum_val) - y;
        sum_val = t_val;
    }}
    threadgroup float shared_sum[256];
    shared_sum[tid] = sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {{
        if (tid < s) {{ shared_sum[tid] += shared_sum[tid + s]; }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float mean = shared_sum[0] / float(channels);

    // --- Pass 2: Compute variance (Kahan-compensated) ---
    float var_sum = 0.0;
    float var_comp = 0.0;
    for (uint c = tid; c < channels; c += tg_size) {{
        float v = float(input[bt_base + c * stride]) - mean;
        float sq = v * v;
        float y = sq - var_comp;
        float t_val = var_sum + y;
        var_comp = (t_val - var_sum) - y;
        var_sum = t_val;
    }}
    threadgroup float shared_var[256];
    shared_var[tid] = var_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {{
        if (tid < s) {{ shared_var[tid] += shared_var[tid + s]; }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float inv_std = metal::precise::rsqrt(shared_var[0] / float(channels) + eps);

    // --- Normalize + affine + LeakyReLU ---
    for (uint c = tid; c < channels; c += tg_size) {{
        float normed = (float(input[bt_base + c * stride]) - mean) * inv_std;
        float val = normed * float(weight[c]) + float(bias[c]);
        val = val < 0.0 ? val * slope : val;
        output[bt_base + c * stride] = {scalar_type}(val);
    }}
}}
"#
    )
}

/// MSL source for pre-compilation: fused channels-first LayerNorm F32 kernel.
pub(crate) fn channels_first_layer_norm_msl_source() -> String {
    channels_first_layer_norm_msl("float")
}

/// MSL source for pre-compilation: fused channels-first LayerNorm F16 kernel.
pub(crate) fn channels_first_layer_norm_f16_msl_source() -> String {
    channels_first_layer_norm_msl("half")
}

/// MSL source for pre-compilation: fused channels-first LayerNorm + LeakyReLU F32 kernel.
pub(crate) fn channels_first_ln_leaky_relu_msl_source() -> String {
    channels_first_layer_norm_leaky_relu_msl("float")
}

/// MSL source for pre-compilation: fused channels-first LayerNorm + LeakyReLU F16 kernel.
pub(crate) fn channels_first_ln_leaky_relu_f16_msl_source() -> String {
    channels_first_layer_norm_leaky_relu_msl("half")
}
