// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Direct Metal dispatch for FusedUpsampleConv1d -- bypasses DynTensor bridge.
//!
//! The standard path creates 3 DynTensors (x, weight, bias) from GpuSlice/
//! weight buffers, dispatches through `gpu_fused_upsample_conv1d`, then
//! extracts the output buffer back. Each DynTensor wrapping allocates an
//! `Arc<MetalTensorData>` + `Vec<usize>` shape.
//!
//! This direct path encodes the same MSL kernel
//! (`fused_upsample_conv1d_{scalar_type}`) directly on raw buffer/offset
//! pairs. Zero DynTensor allocations, zero `gpu_data()` extractions.
//!
//! Part of #4310.

use nn_core::Result;
use nn_dsl::ir::ScalarType;

use crate::cache::PipelineCache;
use crate::gpu_slice::GpuSlice;
use crate::kernel_dispatch::KernelPipeline;

use super::super::CompiledModel;
use super::native_dispatch_err;

/// Threadgroup width for the fused kernel (matches
/// `dyn_tensor_metal_upsample_conv1d_fused.rs::TG_X`).
const TG_X: u32 = 64;

/// Execute `NativeOpKind::FusedUpsampleConv1d` via direct Metal dispatch.
///
/// Dispatches the fused nearest-neighbor Upsample1d + Conv1d MSL kernel
/// directly on GpuSlice buffer/offset pairs. Eliminates 3 DynTensor
/// wrappings + 1 gpu_data extraction vs. the bridge path.
///
/// Returns the output `GpuSlice` (arena-allocated or fresh buffer).
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn execute_fused_upsample_conv1d_direct(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    upsample_factor: usize,
    _in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    input_shape: &[usize],
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    let scalar_type = model.step_scalar_type(step_idx);
    let elem_bytes = scalar_type.byte_size();
    let st_str = scalar_type.msl_str();

    if input_shape.len() != 3 {
        return Err(native_dispatch_err(
            step_idx,
            format!(
                "FusedUpsampleConv1d direct: expected rank 3 input, got {input_shape:?}"
            ),
        ));
    }
    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let in_len = input_shape[2];

    if upsample_factor == 0 || stride == 0 || kernel_size == 0 {
        return Err(native_dispatch_err(
            step_idx,
            format!(
                "FusedUpsampleConv1d direct: invalid params factor={upsample_factor} \
                 stride={stride} kernel_size={kernel_size}"
            ),
        ));
    }

    let up_len = in_len.checked_mul(upsample_factor).ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            format!(
                "FusedUpsampleConv1d direct: up_len overflow ({in_len} * {upsample_factor})"
            ),
        )
    })?;

    let padded = up_len + 2 * padding;
    if padded < kernel_size {
        return Err(native_dispatch_err(
            step_idx,
            format!(
                "FusedUpsampleConv1d direct: padded {padded} < kernel_size {kernel_size}"
            ),
        ));
    }
    let out_len = (padded - kernel_size) / stride + 1;

    if out_len == 0 || batch == 0 {
        let (out_buf, out_offset) =
            crate::arena::arena_alloc_or_create(cache.context(), 0).map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedUpsampleConv1d direct alloc (zero): {e}"),
                )
            })?;
        return Ok(GpuSlice::from_ref(&out_buf, out_offset));
    }

    // Resolve graph input: x (0) with shape [B, C_in, T].
    let x_slice = model.resolve_input_slice(step_idx, 0, buffers)?;

    // Resolve static weights: weight [C_out, C_in, K], bias [C_out].
    let weights = &model.def.weight_buffers[step_idx];
    let weight_buf = weights.get("weight").ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            "FusedUpsampleConv1d direct: missing weight 'weight'".into(),
        )
    })?;
    let bias_buf = weights.get("bias").ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            "FusedUpsampleConv1d direct: missing weight 'bias'".into(),
        )
    })?;

    // Compile (or cache-hit) the MSL kernel.
    let kernel_name = format!("fused_upsample_conv1d_{st_str}");
    let msl_src =
        crate::dyn_tensor_metal::upsample_conv1d_fused::upsample_conv1d_fused_msl::fused_upsample_conv1d_msl(st_str);
    let pipeline = KernelPipeline::from_msl(
        cache,
        &msl_src,
        &kernel_name,
        4, // 4 input buffers: input, weight, bias, output
        false,
    )
    .map_err(|e| {
        native_dispatch_err(
            step_idx,
            format!("FusedUpsampleConv1d direct pipeline: {e}"),
        )
    })?;

    // Allocate output buffer: [B, C_out, T_out].
    let total_out = batch
        .checked_mul(out_channels)
        .and_then(|v| v.checked_mul(out_len))
        .ok_or_else(|| {
            native_dispatch_err(
                step_idx,
                format!(
                    "FusedUpsampleConv1d direct: output elems overflow \
                     ({batch} * {out_channels} * {out_len})"
                ),
            )
        })?;
    let out_bytes = total_out.checked_mul(elem_bytes).ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            format!(
                "FusedUpsampleConv1d direct: output bytes overflow ({total_out} * {elem_bytes})"
            ),
        )
    })?;
    let (out_buf, out_offset) =
        crate::arena::arena_alloc_or_create(cache.context(), out_bytes).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedUpsampleConv1d direct alloc: {e}"))
        })?;

    // Encode dispatch parameters.
    let batch_u32 = crate::to_u32(batch, "fused_up_conv_direct batch")?;
    let in_channels_u32 = crate::to_u32(in_channels, "fused_up_conv_direct in_channels")?;
    let out_channels_u32 = crate::to_u32(out_channels, "fused_up_conv_direct out_channels")?;
    let in_len_u32 = crate::to_u32(in_len, "fused_up_conv_direct in_len")?;
    let up_len_u32 = crate::to_u32(up_len, "fused_up_conv_direct up_len")?;
    let out_len_u32 = crate::to_u32(out_len, "fused_up_conv_direct out_len")?;
    let kernel_size_u32 = crate::to_u32(kernel_size, "fused_up_conv_direct kernel_size")?;
    let stride_u32 = crate::to_u32(stride, "fused_up_conv_direct stride")?;
    let padding_u32 = crate::to_u32(padding, "fused_up_conv_direct padding")?;
    let factor_u32 = crate::to_u32(upsample_factor, "fused_up_conv_direct factor")?;

    let out_rows_u32 = crate::to_u32(batch * out_channels, "fused_up_conv_direct out_rows")?;
    let grid_x = out_len_u32.div_ceil(TG_X);

    // Encode Metal compute command directly on raw buffers.
    let encode =
        |batch_cmd: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
            let enc = batch_cmd.new_encoder()?;
            enc.set_buffer_with_offset(0, x_slice.buffer(), x_slice.byte_offset());
            enc.set_buffer_with_offset(1, weight_buf, 0);
            enc.set_buffer_with_offset(2, bias_buf, 0);
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
                [TG_X, 1, 1],
            )?;
            enc.end_encoding();
            Ok(())
        };

    crate::gpu_scope::get_or_create_batch()?;
    let scope_result = crate::gpu_scope::encode_into_lazy_batch(|batch_cmd| encode(batch_cmd));
    match scope_result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(native_dispatch_err(
                step_idx,
                format!("FusedUpsampleConv1d direct encode: {e}"),
            ))
        }
        Err(e) => return Err(e),
    }

    Ok(GpuSlice::from_ref(&out_buf, out_offset))
}

/// Check whether the FusedUpsampleConv1d direct path supports the given
/// scalar type.
pub(crate) fn supports_scalar_type(st: ScalarType) -> bool {
    matches!(st, ScalarType::F32 | ScalarType::F16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_f32_f16() {
        assert!(supports_scalar_type(ScalarType::F32));
        assert!(supports_scalar_type(ScalarType::F16));
        assert!(!supports_scalar_type(ScalarType::BF16));
    }
}
