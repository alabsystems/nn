// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused NativeOp executors for #4264 dispatch reduction:
//!
//! - `FusedLinearLayerNorm`: Linear → LayerNorm in 2 dispatches (GEMM + norm).
//! - `FusedInstanceNormConv1d`: InstanceNorm → Conv1d in 2 dispatches (stats + fused norm-conv).
//! - `FusedConv1dInstanceNorm`: Conv1d → InstanceNorm in 2 dispatches (conv + fused norm).
//!
//! Each saves 1+ dispatch vs the sequential unfused path.
//!
//! Part of #4264.

use std::mem::size_of;

use nn_core::Result;

use crate::cache::PipelineCache;
use crate::dyn_tensor_metal::welford_msl;
use crate::gpu_slice::GpuSlice;
use crate::kernel_dispatch::KernelPipeline;

use super::native_dispatch_err;
use super::CompiledModel;

// ---------------------------------------------------------------------------
// FusedLinearLayerNorm: Linear(x) → LayerNorm(result)
// ---------------------------------------------------------------------------

/// Threadgroup size for norm reductions.
const TG_SIZE: u32 = 256;

/// Execute a `NativeOpKind::FusedLinearLayerNorm` step.
///
/// Two-dispatch approach:
/// 1. GEMM: `y = x @ W^T + bias` into intermediate buffer
/// 2. Fused LayerNorm: mean+var reduction + normalize + affine on y
///
/// For small out_features, uses a single fused kernel (GEMM + norm in one
/// threadgroup) similar to NormLinear but in the opposite order.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_fused_linear_layer_norm(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    in_features: usize,
    out_features: usize,
    has_bias: bool,
    eps: f32,
    input_shape: &[usize],
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    let scalar_type = model.step_scalar_type(step_idx);
    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();

    // Resolve graph input.
    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;

    // Resolve weight buffers.
    let step_weights = &model.def.weight_buffers[step_idx];
    let weight = step_weights
        .get("weight")
        .ok_or_else(|| native_dispatch_err(step_idx, "FusedLinearLayerNorm: missing 'weight'".into()))?;
    let bias = if has_bias {
        Some(step_weights.get("bias").ok_or_else(|| {
            native_dispatch_err(step_idx, "FusedLinearLayerNorm: missing 'bias'".into())
        })?)
    } else {
        None
    };
    let norm_w = step_weights
        .get("norm_weight")
        .ok_or_else(|| native_dispatch_err(step_idx, "FusedLinearLayerNorm: missing 'norm_weight'".into()))?;
    let norm_b = step_weights
        .get("norm_bias")
        .ok_or_else(|| native_dispatch_err(step_idx, "FusedLinearLayerNorm: missing 'norm_bias'".into()))?;

    // Compute dimensions: input is [...batch, in_features], output is [...batch, out_features].
    let flat_rows: usize = input_shape.iter().rev().skip(1).product();
    if flat_rows == 0 || in_features == 0 || out_features == 0 {
        return Err(native_dispatch_err(
            step_idx,
            "FusedLinearLayerNorm: zero-size dimension".into(),
        ));
    }

    let total_output = flat_rows
        .checked_mul(out_features)
        .ok_or_else(|| native_dispatch_err(step_idx, "FusedLinearLayerNorm: output size overflow".into()))?;

    // Single fused kernel: Linear + LayerNorm in one threadgroup.
    // One threadgroup per row. Each threadgroup:
    //   Phase 1: GEMM row → threadgroup memory
    //   Phase 2: LayerNorm reduction on threadgroup data
    //   Phase 3: Normalize + affine → global output
    let kernel_name = format!(
        "fused_linear_layer_norm_{scalar_str}_b{}",
        u8::from(has_bias)
    );
    let msl_src = linear_layer_norm_msl(scalar_str, has_bias, step_idx)?;

    // Buffer count: input, weight, [bias,] norm_w, norm_b, output
    let buf_count = if has_bias { 5 } else { 4 };
    let pipeline = KernelPipeline::from_msl(cache, &msl_src, &kernel_name, buf_count, false)
        .map_err(|e| native_dispatch_err(step_idx, format!("FusedLinearLayerNorm pipeline: {e}")))?;

    // Allocate output buffer.
    let out_bytes = total_output
        .checked_mul(elem_bytes)
        .ok_or_else(|| native_dispatch_err(step_idx, "FusedLinearLayerNorm: output bytes overflow".into()))?;
    let ctx = cache.context();
    let (out_buf, out_offset) = crate::arena::arena_alloc_or_create(ctx, out_bytes)
        .map_err(|e| native_dispatch_err(step_idx, format!("FusedLinearLayerNorm alloc: {e}")))?;

    // Constants.
    let in_features_u32 = u32::try_from(in_features)
        .map_err(|_| native_dispatch_err(step_idx, "FusedLinearLayerNorm: in_features > u32".into()))?;
    let out_features_u32 = u32::try_from(out_features)
        .map_err(|_| native_dispatch_err(step_idx, "FusedLinearLayerNorm: out_features > u32".into()))?;
    let flat_rows_u32 = u32::try_from(flat_rows)
        .map_err(|_| native_dispatch_err(step_idx, "FusedLinearLayerNorm: flat_rows > u32".into()))?;

    // Dynamic threadgroup memory for GEMM output (reused for norm).
    let tg_mem_bytes = out_features
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| native_dispatch_err(step_idx, "FusedLinearLayerNorm: tg_mem overflow".into()))?
        as u64;

    // Encode and dispatch.
    crate::gpu_scope::get_or_create_batch()?;
    let scope_result = crate::gpu_scope::encode_into_lazy_batch(|batch| {
        let enc = batch.new_encoder()?;

        let mut buf_idx: usize = 0;
        enc.set_buffer_with_offset(buf_idx, input_slice.buffer(), input_slice.byte_offset());
        buf_idx += 1;
        enc.set_buffer_with_offset(buf_idx, weight, 0);
        buf_idx += 1;
        if let Some(b) = bias {
            enc.set_buffer_with_offset(buf_idx, b, 0);
            buf_idx += 1;
        }
        enc.set_buffer_with_offset(buf_idx, norm_w, 0);
        buf_idx += 1;
        enc.set_buffer_with_offset(buf_idx, norm_b, 0);
        buf_idx += 1;
        enc.set_buffer_with_offset(buf_idx, &out_buf, out_offset);
        buf_idx += 1;

        enc.set_bytes(buf_idx, &in_features_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &out_features_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &eps);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &flat_rows_u32);

        enc.set_threadgroup_memory_length(0, tg_mem_bytes);
        enc.encode_threadgroups(pipeline.pipeline(), [flat_rows_u32, 1, 1], [TG_SIZE, 1, 1])?;
        enc.end_encoding();
        Ok::<(), crate::error::MetalError>(())
    });
    match scope_result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(native_dispatch_err(
                step_idx,
                format!("FusedLinearLayerNorm encode: {e}"),
            ))
        }
        Err(e) => return Err(e),
    }

    Ok(GpuSlice::from_ref(&out_buf, out_offset))
}

/// Generate MSL for the fused Linear + LayerNorm kernel.
///
/// Three-phase single threadgroup:
/// 1. GEMM: compute `y[col] = dot(input_row, weight[col]) + bias[col]` → threadgroup
/// 2. LayerNorm reduction: Kahan mean + variance on threadgroup data
/// 3. Normalize + affine: `(y - mean) * inv_std * norm_w + norm_b` → output
fn linear_layer_norm_msl(
    scalar_type: &str,
    has_bias: bool,
    _step_idx: usize,
) -> Result<String> {
    let tg = TG_SIZE as usize;

    let load = |var: &str| -> String {
        if scalar_type == "float" {
            var.to_string()
        } else {
            format!("float({var})")
        }
    };
    let store = |var: &str| -> String {
        if scalar_type == "float" {
            var.to_string()
        } else {
            format!("{scalar_type}({var})")
        }
    };

    let load_weight = load("weight[col * in_features + k]");
    let load_nw = load("norm_w[col]");
    let load_nb = load("norm_b[col]");
    let store_out = store("result");

    let algo = welford_msl::DEFAULT_NORM_REDUCTION;
    let preamble = welford_msl::norm_preamble_msl(algo);

    // Buffer indices: input(0), weight(1), [bias(2)], norm_w, norm_b, output
    let (bias_param, bias_add, norm_w_idx) = if has_bias {
        let bp = format!("    device const {scalar_type}* bias      [[buffer(2)]],\n");
        (bp, "        dot += float(bias[col]);\n".to_string(), 3)
    } else {
        (String::new(), String::new(), 2)
    };

    let norm_b_idx = norm_w_idx + 1;
    let out_idx = norm_b_idx + 1;
    let if_idx = out_idx + 1;
    let of_idx = if_idx + 1;
    let eps_idx = of_idx + 1;
    let fr_idx = eps_idx + 1;

    Ok(format!(
        r#"#include <metal_stdlib>
using namespace metal;

{preamble}

kernel void fused_linear_layer_norm_{scalar_type}_b{has_bias_u}(
    device const {scalar_type}* input    [[buffer(0)]],
    device const {scalar_type}* weight   [[buffer(1)]],
{bias_param}    device const {scalar_type}* norm_w   [[buffer({norm_w_idx})]],
    device const {scalar_type}* norm_b   [[buffer({norm_b_idx})]],
    device {scalar_type}* output         [[buffer({out_idx})]],
    constant uint& in_features           [[buffer({if_idx})]],
    constant uint& out_features          [[buffer({of_idx})]],
    constant float& eps                  [[buffer({eps_idx})]],
    constant uint& flat_rows             [[buffer({fr_idx})]],
    threadgroup float* gemm_out          [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    if (gid >= flat_rows) return;
    uint in_base = gid * in_features;

    // Phase 1: GEMM — compute linear output into threadgroup memory
    for (uint col = tid; col < out_features; col += tg_size) {{
        float dot = 0;
        for (uint k = 0; k < in_features; k++) {{
            dot += {load_input} * {load_weight};
        }}
{bias_add}        gemm_out[col] = dot;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: LayerNorm reduction on gemm_out
{reduction}

    // Phase 3: Normalize + affine → global output
    uint out_base = gid * out_features;
    for (uint col = tid; col < out_features; col += tg_size) {{
        float val = (gemm_out[col] - mean) * inv_std;
        float result = val * {load_nw} + {load_nb};
        output[out_base + col] = {store_out};
    }}
}}"#,
        has_bias_u = u8::from(has_bias),
        load_input = load("input[in_base + k]"),
        reduction = ln_reduction_from_tg_msl(tg),
    ))
}

/// LayerNorm reduction MSL reading from threadgroup `gemm_out[]`.
///
/// Kahan two-pass mean+variance. Produces `mean` and `inv_std`.
fn ln_reduction_from_tg_msl(tg_size: usize) -> String {
    format!(
        r#"
    // --- LayerNorm: Kahan-compensated mean+variance on gemm_out[] ---
    threadgroup float shared_val[{tg_size}];
    threadgroup float shared_comp[{tg_size}];

    // Pass 1: mean
    float local_sum = 0.0f;
    float local_comp = 0.0f;
    for (uint i = tid; i < out_features; i += tg_size) {{
        float v = gemm_out[i];
        float y = v - local_comp;
        float t = local_sum + y;
        local_comp = (t - local_sum) - y;
        local_sum = t;
    }}
    shared_val[tid] = local_sum;
    shared_comp[tid] = local_comp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = {tg_size} / 2; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            float a_val = shared_val[tid]; float a_comp = shared_comp[tid];
            float b_val = shared_val[tid + stride]; float b_comp = shared_comp[tid + stride];
            float y = b_val - (a_comp + b_comp);
            float t = a_val + y;
            shared_comp[tid] = (t - a_val) - y;
            shared_val[tid] = t;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float mean = shared_val[0] / max(float(out_features), 1.0f);

    // Pass 2: variance
    local_sum = 0.0f;
    local_comp = 0.0f;
    for (uint i = tid; i < out_features; i += tg_size) {{
        float d = gemm_out[i] - mean;
        float sq = d * d;
        float y = sq - local_comp;
        float t = local_sum + y;
        local_comp = (t - local_sum) - y;
        local_sum = t;
    }}
    shared_val[tid] = local_sum;
    shared_comp[tid] = local_comp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = {tg_size} / 2; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            float a_val = shared_val[tid]; float a_comp = shared_comp[tid];
            float b_val = shared_val[tid + stride]; float b_comp = shared_comp[tid + stride];
            float y = b_val - (a_comp + b_comp);
            float t = a_val + y;
            shared_comp[tid] = (t - a_val) - y;
            shared_val[tid] = t;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float var = shared_val[0] / max(float(out_features), 1.0f);
    float inv_std = metal::precise::rsqrt(var + eps);"#
    )
}

// ---------------------------------------------------------------------------
// FusedInstanceNormConv1d: InstanceNorm(x) → Conv1d(result)
// ---------------------------------------------------------------------------

/// Execute a `NativeOpKind::FusedInstanceNormConv1d` step.
///
/// Two-dispatch approach:
/// 1. `compute_channel_stats`: per-channel mean + inv_std
/// 2. `fused_instance_norm_conv1d`: inline normalization during Conv1d accumulation
///
/// No affine/activation — just InstanceNorm + Conv1d.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_fused_instance_norm_conv1d(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    eps: f32,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    has_bias: bool,
    input_shape: &[usize],
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    let scalar_type = model.step_scalar_type(step_idx);
    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();

    if input_shape.len() < 3 {
        return Err(native_dispatch_err(
            step_idx,
            "FusedInstanceNormConv1d: input must be rank >= 3".into(),
        ));
    }

    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let in_len = input_shape[2];

    if batch == 0 || in_channels == 0 || in_len == 0 || out_channels == 0 {
        return Err(native_dispatch_err(
            step_idx,
            "FusedInstanceNormConv1d: zero-size dimension".into(),
        ));
    }

    // Compute output length.
    let out_len = (in_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    let flat_rows = batch
        .checked_mul(in_channels)
        .ok_or_else(|| native_dispatch_err(step_idx, "FusedInstanceNormConv1d: flat_rows overflow".into()))?;

    let total_output = batch
        .checked_mul(out_channels)
        .and_then(|v| v.checked_mul(out_len))
        .ok_or_else(|| native_dispatch_err(step_idx, "FusedInstanceNormConv1d: output size overflow".into()))?;

    // Resolve graph input.
    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;

    // Resolve conv weights.
    let step_weights = &model.def.weight_buffers[step_idx];
    let conv_w = step_weights
        .get("conv_weight")
        .ok_or_else(|| native_dispatch_err(step_idx, "FusedInstanceNormConv1d: missing 'conv_weight'".into()))?;
    let conv_b = if has_bias {
        Some(step_weights.get("conv_bias").ok_or_else(|| {
            native_dispatch_err(step_idx, "FusedInstanceNormConv1d: missing 'conv_bias'".into())
        })?)
    } else {
        None
    };

    let ctx = cache.context();

    // --- Dispatch 1: Compute per-channel stats ---
    let stats_kernel_name = format!("compute_channel_stats_{scalar_str}");
    let stats_msl = crate::dyn_tensor_metal::stats_kernel_msl_source(scalar_str);
    let stats_pipeline =
        KernelPipeline::from_msl(cache, &stats_msl, &stats_kernel_name, 1, false)
            .map_err(|e| native_dispatch_err(step_idx, format!("FusedInstanceNormConv1d stats pipeline: {e}")))?;

    let stats_bytes = flat_rows
        .checked_mul(2 * size_of::<f32>())
        .ok_or_else(|| native_dispatch_err(step_idx, "FusedInstanceNormConv1d: stats bytes overflow".into()))?;
    let (stats_buf, stats_offset) = crate::arena::arena_alloc_or_create(ctx, stats_bytes)
        .map_err(|e| native_dispatch_err(step_idx, format!("FusedInstanceNormConv1d stats alloc: {e}")))?;

    let flat_rows_u32 = u32::try_from(flat_rows)
        .map_err(|_| native_dispatch_err(step_idx, "FusedInstanceNormConv1d: flat_rows > u32".into()))?;
    let in_len_u32 = u32::try_from(in_len)
        .map_err(|_| native_dispatch_err(step_idx, "FusedInstanceNormConv1d: in_len > u32".into()))?;

    crate::gpu_scope::get_or_create_batch()?;
    let scope_result = crate::gpu_scope::encode_into_lazy_batch(|batch| {
        let enc = batch.new_encoder()?;
        enc.set_buffer_with_offset(0, input_slice.buffer(), input_slice.byte_offset());
        enc.set_buffer_with_offset(1, &stats_buf, stats_offset);
        enc.set_bytes(2, &in_len_u32);
        enc.set_bytes(3, &eps);
        enc.encode_threadgroups(
            stats_pipeline.pipeline(),
            [flat_rows_u32, 1, 1],
            [256, 1, 1],
        )?;
        enc.end_encoding();
        Ok::<(), crate::error::MetalError>(())
    });
    match scope_result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(native_dispatch_err(
                step_idx,
                format!("FusedInstanceNormConv1d stats encode: {e}"),
            ))
        }
        Err(e) => return Err(e),
    }

    // --- Dispatch 2: Fused norm + conv1d ---
    let conv_kernel_name = format!(
        "fused_instnorm_conv1d_{scalar_str}_b{}",
        u8::from(has_bias)
    );
    let conv_msl = instance_norm_conv1d_msl(scalar_str, has_bias);

    // Buffers: input, stats, conv_weight, [conv_bias,] output
    let conv_buf_count = if has_bias { 4 } else { 3 };
    let conv_pipeline =
        KernelPipeline::from_msl(cache, &conv_msl, &conv_kernel_name, conv_buf_count, false)
            .map_err(|e| native_dispatch_err(step_idx, format!("FusedInstanceNormConv1d conv pipeline: {e}")))?;

    let out_bytes = total_output
        .checked_mul(elem_bytes)
        .ok_or_else(|| native_dispatch_err(step_idx, "FusedInstanceNormConv1d: output bytes overflow".into()))?;
    let (out_buf, out_offset) = crate::arena::arena_alloc_or_create(ctx, out_bytes)
        .map_err(|e| native_dispatch_err(step_idx, format!("FusedInstanceNormConv1d alloc: {e}")))?;

    let batch_u32 = u32::try_from(batch)
        .map_err(|_| native_dispatch_err(step_idx, "FusedInstanceNormConv1d: batch > u32".into()))?;
    let in_channels_u32 = u32::try_from(in_channels)
        .map_err(|_| native_dispatch_err(step_idx, "FusedInstanceNormConv1d: in_channels > u32".into()))?;
    let out_channels_u32 = u32::try_from(out_channels)
        .map_err(|_| native_dispatch_err(step_idx, "FusedInstanceNormConv1d: out_channels > u32".into()))?;
    let out_len_u32 = u32::try_from(out_len)
        .map_err(|_| native_dispatch_err(step_idx, "FusedInstanceNormConv1d: out_len > u32".into()))?;
    let kernel_size_u32 = u32::try_from(kernel_size)
        .map_err(|_| native_dispatch_err(step_idx, "FusedInstanceNormConv1d: kernel_size > u32".into()))?;
    let stride_u32 = u32::try_from(stride)
        .map_err(|_| native_dispatch_err(step_idx, "FusedInstanceNormConv1d: stride > u32".into()))?;
    let padding_u32 = u32::try_from(padding)
        .map_err(|_| native_dispatch_err(step_idx, "FusedInstanceNormConv1d: padding > u32".into()))?;
    let dilation_u32 = u32::try_from(dilation)
        .map_err(|_| native_dispatch_err(step_idx, "FusedInstanceNormConv1d: dilation > u32".into()))?;

    // Dispatch: one thread per output element (t_out), one threadgroup row per (b, c_out).
    let tg_x: u32 = 256;
    let grid_x = out_len_u32.div_ceil(tg_x);
    let grid_y = batch_u32
        .checked_mul(out_channels_u32)
        .ok_or_else(|| native_dispatch_err(step_idx, "FusedInstanceNormConv1d: grid_y overflow".into()))?;

    let scope_result2 = crate::gpu_scope::encode_into_lazy_batch(|batch| {
        let enc = batch.new_encoder()?;

        let mut buf_idx: usize = 0;
        enc.set_buffer_with_offset(buf_idx, input_slice.buffer(), input_slice.byte_offset());
        buf_idx += 1;
        enc.set_buffer_with_offset(buf_idx, &stats_buf, stats_offset);
        buf_idx += 1;
        enc.set_buffer_with_offset(buf_idx, conv_w, 0);
        buf_idx += 1;
        if let Some(b) = conv_b {
            enc.set_buffer_with_offset(buf_idx, b, 0);
            buf_idx += 1;
        }
        enc.set_buffer_with_offset(buf_idx, &out_buf, out_offset);
        buf_idx += 1;

        enc.set_bytes(buf_idx, &batch_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &in_channels_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &out_channels_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &in_len_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &out_len_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &kernel_size_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &stride_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &padding_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &dilation_u32);

        enc.encode_threadgroups(conv_pipeline.pipeline(), [grid_x, grid_y, 1], [tg_x, 1, 1])?;
        enc.end_encoding();
        Ok::<(), crate::error::MetalError>(())
    });
    match scope_result2 {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(native_dispatch_err(
                step_idx,
                format!("FusedInstanceNormConv1d conv encode: {e}"),
            ))
        }
        Err(e) => return Err(e),
    }

    Ok(GpuSlice::from_ref(&out_buf, out_offset))
}

/// MSL for fused InstanceNorm + Conv1d (no affine, no activation).
///
/// Each thread computes one `(b, c_out, t_out)` output element.
/// For each input channel, reads precomputed stats to normalize inline
/// during Conv1d accumulation.
fn instance_norm_conv1d_msl(scalar_type: &str, has_bias: bool) -> String {
    let (bias_param, bias_init, bias_idx_offset) = if has_bias {
        (
            format!("    device const {scalar_type}* conv_bias [[buffer(3)]],\n"),
            "    float acc = float(conv_bias[oc]);\n".to_string(),
            1usize,
        )
    } else {
        (String::new(), "    float acc = 0.0f;\n".to_string(), 0)
    };

    let out_idx = 3 + bias_idx_offset;
    let batch_idx = out_idx + 1;
    let ic_idx = batch_idx + 1;
    let oc_idx = ic_idx + 1;
    let il_idx = oc_idx + 1;
    let ol_idx = il_idx + 1;
    let ks_idx = ol_idx + 1;
    let stride_idx = ks_idx + 1;
    let pad_idx = stride_idx + 1;
    let dil_idx = pad_idx + 1;

    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void fused_instnorm_conv1d_{scalar_type}_b{has_bias_u}(
    device const {scalar_type}* input      [[buffer(0)]],
    device const float* stats              [[buffer(1)]],
    device const {scalar_type}* conv_weight [[buffer(2)]],
{bias_param}    device {scalar_type}* output           [[buffer({out_idx})]],
    constant uint& batch                   [[buffer({batch_idx})]],
    constant uint& in_channels             [[buffer({ic_idx})]],
    constant uint& out_channels            [[buffer({oc_idx})]],
    constant uint& in_len                  [[buffer({il_idx})]],
    constant uint& out_len                 [[buffer({ol_idx})]],
    constant uint& kernel_size             [[buffer({ks_idx})]],
    constant uint& conv_stride             [[buffer({stride_idx})]],
    constant uint& conv_padding            [[buffer({pad_idx})]],
    constant uint& conv_dilation           [[buffer({dil_idx})]],
    uint2 gid [[thread_position_in_grid]]
) {{
    uint t = gid.x;
    uint row = gid.y;

    if (t >= out_len) return;

    uint b = row / out_channels;
    uint oc = row % out_channels;
    if (b >= batch) return;

{bias_init}
    for (uint ic = 0; ic < in_channels; ic++) {{
        uint stats_idx = (b * in_channels + ic) * 2;
        float ch_mean = stats[stats_idx];
        float ch_inv_std = stats[stats_idx + 1];

        uint w_base = (oc * in_channels + ic) * kernel_size;
        uint in_base = (b * in_channels + ic) * in_len;

        for (uint k = 0; k < kernel_size; k++) {{
            int t_in = int(t) * int(conv_stride) + int(k) * int(conv_dilation) - int(conv_padding);
            if (t_in >= 0 && uint(t_in) < in_len) {{
                float x = float(input[in_base + uint(t_in)]);
                float normed = (x - ch_mean) * ch_inv_std;
                acc += normed * float(conv_weight[w_base + k]);
            }}
        }}
    }}

    uint out_idx_val = (b * out_channels + oc) * out_len + t;
    output[out_idx_val] = {scalar_type}(acc);
}}"#,
        has_bias_u = u8::from(has_bias),
    )
}

// ---------------------------------------------------------------------------
// FusedConv1dInstanceNorm: Conv1d(x) → InstanceNorm(result)
// ---------------------------------------------------------------------------

/// Execute a `NativeOpKind::FusedConv1dInstanceNorm` step.
///
/// Two-dispatch approach:
/// 1. Conv1d: `y = conv1d(x, weight, bias)` into intermediate buffer
/// 2. Fused InstanceNorm: single-dispatch norm on conv output
///
/// The key savings come from eliminating the scheduling overhead of separate
/// Conv1d + InstanceNorm NativeOps (which each require their own dispatch plan
/// step, buffer tracking, etc.).
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_fused_conv1d_instance_norm(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    eps: f32,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    has_bias: bool,
    input_shape: &[usize],
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    let scalar_type = model.step_scalar_type(step_idx);
    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();

    if input_shape.len() < 3 {
        return Err(native_dispatch_err(
            step_idx,
            "FusedConv1dInstanceNorm: input must be rank >= 3".into(),
        ));
    }

    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let in_len = input_shape[2];

    if batch == 0 || in_channels == 0 || in_len == 0 || out_channels == 0 {
        return Err(native_dispatch_err(
            step_idx,
            "FusedConv1dInstanceNorm: zero-size dimension".into(),
        ));
    }

    // Compute output length.
    let out_len = (in_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    let conv_out_rows = batch
        .checked_mul(out_channels)
        .ok_or_else(|| native_dispatch_err(step_idx, "FusedConv1dInstanceNorm: conv_out_rows overflow".into()))?;
    let total_output = conv_out_rows
        .checked_mul(out_len)
        .ok_or_else(|| native_dispatch_err(step_idx, "FusedConv1dInstanceNorm: output size overflow".into()))?;

    // Resolve graph input.
    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;

    // Resolve conv weights.
    let step_weights = &model.def.weight_buffers[step_idx];
    let conv_w = step_weights
        .get("conv_weight")
        .ok_or_else(|| native_dispatch_err(step_idx, "FusedConv1dInstanceNorm: missing 'conv_weight'".into()))?;
    let conv_b = if has_bias {
        Some(step_weights.get("conv_bias").ok_or_else(|| {
            native_dispatch_err(step_idx, "FusedConv1dInstanceNorm: missing 'conv_bias'".into())
        })?)
    } else {
        None
    };

    let ctx = cache.context();

    // --- Dispatch 1: Conv1d into intermediate buffer ---
    let conv_kernel_name = format!(
        "conv1d_for_instnorm_{scalar_str}_b{}",
        u8::from(has_bias)
    );
    let conv_msl = conv1d_msl(scalar_str, has_bias);

    let conv_buf_count = if has_bias { 3 } else { 2 };
    let conv_pipeline =
        KernelPipeline::from_msl(cache, &conv_msl, &conv_kernel_name, conv_buf_count, false)
            .map_err(|e| native_dispatch_err(step_idx, format!("FusedConv1dInstanceNorm conv pipeline: {e}")))?;

    let intermediate_bytes = total_output
        .checked_mul(elem_bytes)
        .ok_or_else(|| native_dispatch_err(step_idx, "FusedConv1dInstanceNorm: intermediate bytes overflow".into()))?;
    let (inter_buf, inter_offset) = crate::arena::arena_alloc_or_create(ctx, intermediate_bytes)
        .map_err(|e| native_dispatch_err(step_idx, format!("FusedConv1dInstanceNorm inter alloc: {e}")))?;

    let batch_u32 = u32::try_from(batch)
        .map_err(|_| native_dispatch_err(step_idx, "FusedConv1dInstanceNorm: batch > u32".into()))?;
    let in_channels_u32 = u32::try_from(in_channels)
        .map_err(|_| native_dispatch_err(step_idx, "FusedConv1dInstanceNorm: in_channels > u32".into()))?;
    let out_channels_u32 = u32::try_from(out_channels)
        .map_err(|_| native_dispatch_err(step_idx, "FusedConv1dInstanceNorm: out_channels > u32".into()))?;
    let in_len_u32 = u32::try_from(in_len)
        .map_err(|_| native_dispatch_err(step_idx, "FusedConv1dInstanceNorm: in_len > u32".into()))?;
    let out_len_u32 = u32::try_from(out_len)
        .map_err(|_| native_dispatch_err(step_idx, "FusedConv1dInstanceNorm: out_len > u32".into()))?;
    let kernel_size_u32 = u32::try_from(kernel_size)
        .map_err(|_| native_dispatch_err(step_idx, "FusedConv1dInstanceNorm: kernel_size > u32".into()))?;
    let stride_u32 = u32::try_from(stride)
        .map_err(|_| native_dispatch_err(step_idx, "FusedConv1dInstanceNorm: stride > u32".into()))?;
    let padding_u32 = u32::try_from(padding)
        .map_err(|_| native_dispatch_err(step_idx, "FusedConv1dInstanceNorm: padding > u32".into()))?;
    let dilation_u32 = u32::try_from(dilation)
        .map_err(|_| native_dispatch_err(step_idx, "FusedConv1dInstanceNorm: dilation > u32".into()))?;

    let tg_x: u32 = 256;
    let grid_x = out_len_u32.div_ceil(tg_x);
    let grid_y = batch_u32
        .checked_mul(out_channels_u32)
        .ok_or_else(|| native_dispatch_err(step_idx, "FusedConv1dInstanceNorm: grid_y overflow".into()))?;

    crate::gpu_scope::get_or_create_batch()?;
    let scope_result = crate::gpu_scope::encode_into_lazy_batch(|batch| {
        let enc = batch.new_encoder()?;

        let mut buf_idx: usize = 0;
        enc.set_buffer_with_offset(buf_idx, input_slice.buffer(), input_slice.byte_offset());
        buf_idx += 1;
        enc.set_buffer_with_offset(buf_idx, conv_w, 0);
        buf_idx += 1;
        if let Some(b) = conv_b {
            enc.set_buffer_with_offset(buf_idx, b, 0);
            buf_idx += 1;
        }
        enc.set_buffer_with_offset(buf_idx, &inter_buf, inter_offset);
        buf_idx += 1;

        enc.set_bytes(buf_idx, &batch_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &in_channels_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &out_channels_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &in_len_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &out_len_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &kernel_size_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &stride_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &padding_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &dilation_u32);

        enc.encode_threadgroups(conv_pipeline.pipeline(), [grid_x, grid_y, 1], [tg_x, 1, 1])?;
        enc.end_encoding();
        Ok::<(), crate::error::MetalError>(())
    });
    match scope_result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(native_dispatch_err(
                step_idx,
                format!("FusedConv1dInstanceNorm conv encode: {e}"),
            ))
        }
        Err(e) => return Err(e),
    }

    // --- Dispatch 2: Fused InstanceNorm on conv output ---
    let norm_kernel_name = format!("fused_instance_norm_{scalar_str}");
    let norm_msl = crate::dyn_tensor_metal::instance_norm_msl_source();
    // Use half variant if scalar type is half.
    let norm_msl = if scalar_str == "half" {
        crate::dyn_tensor_metal::instance_norm_f16_msl_source()
    } else {
        norm_msl
    };
    let norm_pipeline =
        KernelPipeline::from_msl(cache, &norm_msl, &norm_kernel_name, 1, false)
            .map_err(|e| native_dispatch_err(step_idx, format!("FusedConv1dInstanceNorm norm pipeline: {e}")))?;

    let (out_buf, out_offset) = crate::arena::arena_alloc_or_create(ctx, intermediate_bytes)
        .map_err(|e| native_dispatch_err(step_idx, format!("FusedConv1dInstanceNorm norm alloc: {e}")))?;

    let conv_out_rows_u32 = u32::try_from(conv_out_rows)
        .map_err(|_| native_dispatch_err(step_idx, "FusedConv1dInstanceNorm: conv_out_rows > u32".into()))?;

    let scope_result2 = crate::gpu_scope::encode_into_lazy_batch(|batch| {
        let enc = batch.new_encoder()?;
        enc.set_buffer_with_offset(0, &inter_buf, inter_offset);
        enc.set_buffer_with_offset(1, &out_buf, out_offset);
        enc.set_bytes(2, &out_len_u32);
        enc.set_bytes(3, &eps);
        enc.encode_threadgroups(
            norm_pipeline.pipeline(),
            [conv_out_rows_u32, 1, 1],
            [256, 1, 1],
        )?;
        enc.end_encoding();
        Ok::<(), crate::error::MetalError>(())
    });
    match scope_result2 {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(native_dispatch_err(
                step_idx,
                format!("FusedConv1dInstanceNorm norm encode: {e}"),
            ))
        }
        Err(e) => return Err(e),
    }

    Ok(GpuSlice::from_ref(&out_buf, out_offset))
}

/// MSL for Conv1d kernel (standalone, for the first dispatch of FusedConv1dInstanceNorm).
///
/// Same algorithm as the existing conv1d MSL but without activation.
/// One thread per output element `(b, c_out, t_out)`.
fn conv1d_msl(scalar_type: &str, has_bias: bool) -> String {
    let (bias_param, bias_init, bias_idx_offset) = if has_bias {
        (
            format!("    device const {scalar_type}* conv_bias [[buffer(2)]],\n"),
            "    float acc = float(conv_bias[oc]);\n".to_string(),
            1usize,
        )
    } else {
        (String::new(), "    float acc = 0.0f;\n".to_string(), 0)
    };

    let out_idx = 2 + bias_idx_offset;
    let batch_idx = out_idx + 1;
    let ic_idx = batch_idx + 1;
    let oc_idx = ic_idx + 1;
    let il_idx = oc_idx + 1;
    let ol_idx = il_idx + 1;
    let ks_idx = ol_idx + 1;
    let stride_idx = ks_idx + 1;
    let pad_idx = stride_idx + 1;
    let dil_idx = pad_idx + 1;

    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void conv1d_for_instnorm_{scalar_type}_b{has_bias_u}(
    device const {scalar_type}* input      [[buffer(0)]],
    device const {scalar_type}* conv_weight [[buffer(1)]],
{bias_param}    device {scalar_type}* output           [[buffer({out_idx})]],
    constant uint& batch                   [[buffer({batch_idx})]],
    constant uint& in_channels             [[buffer({ic_idx})]],
    constant uint& out_channels            [[buffer({oc_idx})]],
    constant uint& in_len                  [[buffer({il_idx})]],
    constant uint& out_len                 [[buffer({ol_idx})]],
    constant uint& kernel_size             [[buffer({ks_idx})]],
    constant uint& conv_stride             [[buffer({stride_idx})]],
    constant uint& conv_padding            [[buffer({pad_idx})]],
    constant uint& conv_dilation           [[buffer({dil_idx})]],
    uint2 gid [[thread_position_in_grid]]
) {{
    uint t = gid.x;
    uint row = gid.y;

    if (t >= out_len) return;

    uint b = row / out_channels;
    uint oc = row % out_channels;
    if (b >= batch) return;

{bias_init}
    for (uint ic = 0; ic < in_channels; ic++) {{
        uint w_base = (oc * in_channels + ic) * kernel_size;
        uint in_base = (b * in_channels + ic) * in_len;

        for (uint k = 0; k < kernel_size; k++) {{
            int t_in = int(t) * int(conv_stride) + int(k) * int(conv_dilation) - int(conv_padding);
            if (t_in >= 0 && uint(t_in) < in_len) {{
                acc += float(input[in_base + uint(t_in)]) * float(conv_weight[w_base + k]);
            }}
        }}
    }}

    uint out_idx_val = (b * out_channels + oc) * out_len + t;
    output[out_idx_val] = {scalar_type}(acc);
}}"#,
        has_bias_u = u8::from(has_bias),
    )
}
