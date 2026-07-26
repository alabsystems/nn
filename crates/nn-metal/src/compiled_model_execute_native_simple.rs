// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Simple native operation helpers for `CompiledModel`: Cumsum, InstanceNorm,
//! LayerNorm, MaxPool1d, ConstantWeight, LinearActivation.
//!
//! Extracted from `compiled_model_execute_native.rs` to keep files under
//! 450 lines. See #2921.

use nn_core::Result;
use nn_dsl::{GemmActivation, PrecisionTier};

use crate::cache::PipelineCache;
use crate::dispatch_plan::DispatchMode;
use crate::gpu_slice::GpuSlice;
use crate::kernel_dispatch::KernelPipeline;

use super::CompiledModel;
use super::{dyn_to_slice, native_dispatch_err, slice_to_dyn, weight_to_dyn};

#[path = "compiled_model_execute_native_cumsum.rs"]
mod cumsum_exec;

pub(super) use cumsum_exec::execute_native_cumsum;

/// Execute a `NativeOpKind::InstanceNorm` step (fused, #2472).
pub(super) fn execute_native_instance_norm(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    eps: f32,
    input_shape: &[usize],
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);

    // Resolve the graph input.
    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;

    let input_tensor = slice_to_dyn(&input_slice, input_shape, dtype)?;

    // When the compiled model requests Strict precision, use the decomposed
    // path instead of the fused single-dispatch kernel. Both paths now use
    // Kahan-compensated reductions (#2696), so this is only needed for
    // models that explicitly require decomposed dispatch (#2704).
    let use_kahan = model
        .precision()
        .map_or(false, |c| c.tier == PrecisionTier::Strict);
    let output = if use_kahan {
        crate::dyn_tensor_metal::native_instance_norm_precise(&input_tensor, f64::from(eps))
    } else {
        crate::dyn_tensor_metal::native_instance_norm(&input_tensor, f64::from(eps))
    }
    .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp InstanceNorm: {e}")))?;

    dyn_to_slice(&output, step_idx, "InstanceNorm")
}

/// Execute a `NativeOpKind::LayerNorm` step.
///
/// Resolves the graph input and pre-uploaded weight/bias, calls
/// `gpu_layer_norm` (decomposed GPU dispatch path).
pub(super) fn execute_native_layer_norm(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    eps: f32,
    input_shape: &[usize],
    hidden_dim: usize,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);

    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;

    let weights = &model.def.weight_buffers[step_idx];
    let input_tensor = slice_to_dyn(&input_slice, input_shape, dtype)?;
    let weight = weight_to_dyn(
        weights,
        "weight",
        &[hidden_dim],
        dtype,
        step_idx,
        "LayerNorm",
    )?;
    let bias = weight_to_dyn(weights, "bias", &[hidden_dim], dtype, step_idx, "LayerNorm")?;

    let output =
        crate::dyn_tensor_metal::native_layer_norm(&input_tensor, &weight, &bias, f64::from(eps))
            .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp LayerNorm: {e}")))?;

    dyn_to_slice(&output, step_idx, "LayerNorm")
}

/// Execute a `NativeOpKind::MaxPool1d` step.
///
/// Wraps the graph input as a DynTensor, calls `max_pool1d()` via the
/// native bridge. Part of #2295 (PyanNet speaker segmentation).
pub(super) fn execute_native_max_pool1d(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    kernel_size: usize,
    stride: usize,
    padding: usize,
    input_shape: &[usize],
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);

    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;

    let input_tensor = slice_to_dyn(&input_slice, input_shape, dtype)?;

    let output =
        crate::dyn_tensor_metal::native_max_pool1d(&input_tensor, kernel_size, stride, padding)
            .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp MaxPool1d: {e}")))?;

    dyn_to_slice(&output, step_idx, "MaxPool1d")
}

/// Execute a `NativeOpKind::ConstantWeight` step.
///
/// Resolves the pre-uploaded weight buffer (e.g., from `arange`) and returns
/// it directly as the step output. No computation needed.
pub(super) fn execute_native_constant_weight(
    model: &CompiledModel,
    step_idx: usize,
    name: &str,
    shape: &[usize],
) -> Result<GpuSlice> {
    // Direct buffer pass-through: weight buffer IS the output.
    // No DynTensor wrapping needed — eliminates Arc + Vec<usize> + gpu_data
    // round-trip per ConstantWeight step. Part of #3230 Gap 5.
    let step_weights = &model.def.weight_buffers[step_idx];
    let name_data = format!("{name}_data");
    let candidates = [name_data.as_str(), name];

    for candidate in &candidates {
        if let Some(buf) = step_weights.get(*candidate) {
            return Ok(GpuSlice::from_ref(buf, 0));
        }
    }

    Err(native_dispatch_err(
        step_idx,
        format!("NativeOp ConstantWeight '{name}': missing weight buffer (shape {shape:?})"),
    ))
}

/// Execute a `NativeOpKind::LinearActivation` step.
///
/// Fused Linear + Activation: `activation(input @ weight^T + bias)`.
/// Generates and dispatches a single fused MSL kernel that computes
/// matmul + bias + activation in one Metal dispatch. The activation is
/// applied to the f32 accumulator before casting to storage type, which
/// is more precise than the 2-dispatch path.
///
/// Routes to simdgroup-tiled GEMM when dimensions conform (all % 8,
/// M×N ≥ 16384, K ≥ 128) for ~3× throughput on large Linear layers.
/// Falls back to naive per-element kernel otherwise. Part of #2256 D3+D4.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_native_linear_activation(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    activation: &GemmActivation,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
    input_shape: &[usize],
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    let scalar_type = model.step_scalar_type(step_idx);

    // Resolve the graph input.
    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;

    // Resolve weight and optional bias buffers.
    let step_weights = &model.def.weight_buffers[step_idx];
    let weight_buf = step_weights.get("weight").ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            "NativeOp LinearActivation: missing 'weight'".into(),
        )
    })?;
    let bias_buf = if has_bias {
        Some(step_weights.get("bias").ok_or_else(|| {
            native_dispatch_err(step_idx, "NativeOp LinearActivation: missing 'bias'".into())
        })?)
    } else {
        None
    };

    // Compute output element count: batch_size * out_features.
    // input_shape is [...batch, in_features]; output replaces last dim.
    let batch_size: usize = input_shape.iter().rev().skip(1).product();
    let total_output = batch_size.checked_mul(out_features).ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            "NativeOp LinearActivation: output size overflow".into(),
        )
    })?;

    // Route: simdgroup-tiled when M=batch, K=in_features, N=out_features conform.
    let use_simdgroup =
        crate::dyn_tensor_metal::should_use_simdgroup(batch_size, in_features, out_features);

    // Generate fused MSL kernel source.
    // Kernel name encodes dimensions + activation, NOT step_idx, so identical
    // LinearActivation layers at different graph positions share a single
    // compiled Metal pipeline (e.g., 12-layer transformer compiles 6 unique
    // pipelines instead of 72).
    let act_tag = activation_tag(activation);
    let kernel_name = if use_simdgroup {
        format!(
            "simd_la_{}_m{batch_size}_k{in_features}_n{out_features}_{act_tag}_b{}",
            scalar_type.msl_str(),
            u8::from(has_bias),
        )
    } else {
        format!(
            "la_{}_k{in_features}_n{out_features}_{act_tag}_b{}",
            scalar_type.msl_str(),
            u8::from(has_bias),
        )
    };
    let msl_source = if use_simdgroup {
        nn_dsl::emit_simdgroup_linear_activation_kernel(
            &kernel_name,
            scalar_type,
            in_features,
            out_features,
            batch_size,
            has_bias,
            activation,
        )
    } else {
        nn_dsl::emit_linear_activation_kernel(
            &kernel_name,
            scalar_type,
            in_features,
            out_features,
            has_bias,
            Some(activation),
            true, // include MSL prelude for standalone compilation
        )
    }
    .map_err(|e| {
        native_dispatch_err(
            step_idx,
            format!("NativeOp LinearActivation MSL codegen: {e}"),
        )
    })?;

    // Compile the kernel pipeline (cached by PipelineCache).
    let param_count = if has_bias { 3 } else { 2 };
    let pipeline = KernelPipeline::from_msl(cache, &msl_source, &kernel_name, param_count, false)
        .map_err(|e| {
        native_dispatch_err(step_idx, format!("NativeOp LinearActivation pipeline: {e}"))
    })?;

    // Allocate output buffer.
    let elem_bytes = scalar_type.byte_size();
    let out_bytes = total_output.checked_mul(elem_bytes).ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            "NativeOp LinearActivation: output bytes overflow".into(),
        )
    })?;
    let ctx = cache.context();
    let (out_buf, out_offset) =
        crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(|e| {
            native_dispatch_err(step_idx, format!("NativeOp LinearActivation alloc: {e}"))
        })?;

    // Build dispatch plan: simdgroup uses 3D threadgroup grid, naive uses 1D.
    let plan = if use_simdgroup {
        let m_u32 = u32::try_from(batch_size).map_err(|_| {
            native_dispatch_err(
                step_idx,
                "NativeOp LinearActivation: batch exceeds u32".into(),
            )
        })?;
        let n_u32 = u32::try_from(out_features).map_err(|_| {
            native_dispatch_err(
                step_idx,
                "NativeOp LinearActivation: out_features exceeds u32".into(),
            )
        })?;
        let is_half = elem_bytes == 2;
        // Threadgroup memory: As + Bs (element-sized) + tile_out (float).
        let tg_bytes: u64 = if is_half {
            2 * 32 * 33 * 2 + 32 * 33 * 4
        } else {
            3 * 32 * 33 * 4
        };
        DispatchMode::Grid3D {
            grid: [n_u32.div_ceil(32), m_u32.div_ceil(32), 1],
            threads: [32, 4, 1],
        }
        .plan()
        .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp LinearActivation plan: {e}")))?
        .with_output_elems(total_output)
        .with_constants(vec![])
        .with_use_threadgroups(true)
        .with_threadgroup_memory_bytes(Some(tg_bytes))
    } else {
        let total_u32 = u32::try_from(total_output).map_err(|_| {
            native_dispatch_err(
                step_idx,
                "NativeOp LinearActivation: total exceeds u32".into(),
            )
        })?;
        DispatchMode::Elementwise { total: total_u32 }
            .plan()
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("NativeOp LinearActivation plan: {e}"))
            })?
    };

    // Bind input buffers with offset for the graph input.
    let mut inputs: Vec<&crate::buffer::MetalBuffer> = Vec::with_capacity(param_count);
    inputs.push(input_slice.buffer());
    inputs.push(weight_buf);
    if let Some(b) = bias_buf {
        inputs.push(b);
    }
    let mut offsets = vec![input_slice.byte_offset()];
    // Weight and bias have zero offset.
    offsets.resize(param_count, 0);

    // Dispatch: single fused Metal kernel.
    pipeline
        .dispatch_buffers_with_all_offsets(ctx, &inputs, &offsets, &out_buf, out_offset, &plan)
        .map_err(|e| {
            native_dispatch_err(step_idx, format!("NativeOp LinearActivation dispatch: {e}"))
        })?;

    Ok(GpuSlice::from_ref(&out_buf, out_offset))
}

/// Execute a `NativeOpKind::Conv1dGemm` step.
///
/// Routes Conv1d through the appropriate Metal path:
/// - `groups == 1`, K=3, Kokoro shape: direct sliding-window kernel (#4264)
/// - `groups == 1` and large FLOPs: im2col + simdgroup GEMM (#3390)
/// - `groups > 1` (depthwise/grouped): generic `gpu_conv1d` (#3538)
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_native_conv1d_gemm(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    input_shape: &[usize],
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    has_bias: bool,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);

    // Resolve the graph input.
    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let input_tensor = slice_to_dyn(&input_slice, input_shape, dtype)?;

    // Resolve weight and optional bias from pre-uploaded buffers.
    // Weight shape: [C_out, C_in_per_group, K] where C_in_per_group = C_in / groups.
    let step_weights = &model.def.weight_buffers[step_idx];
    let c_in = input_shape.get(1).copied().unwrap_or(1);
    let c_in_per_group = if groups > 0 { c_in / groups } else { c_in };
    let weight = weight_to_dyn(
        step_weights,
        "weight",
        &[out_channels, c_in_per_group, kernel_size],
        dtype,
        step_idx,
        "Conv1dGemm",
    )?;
    let bias_tensor = if has_bias {
        Some(weight_to_dyn(
            step_weights,
            "bias",
            &[out_channels],
            dtype,
            step_idx,
            "Conv1dGemm",
        )?)
    } else {
        None
    };

    // Compute output shape: [B, C_out, L_out].
    let batch = input_shape.first().copied().unwrap_or(1);
    let l_in = input_shape.get(2).copied().unwrap_or(0);
    let effective_k = (kernel_size - 1) * dilation + 1;
    let padded = l_in + 2 * padding;
    let l_out = if padded >= effective_k {
        (padded - effective_k) / stride + 1
    } else {
        0
    };
    let out_shape = [batch, out_channels, l_out];

    // Route: groups == 1 → im2col+GEMM, groups > 1 → generic gpu_conv1d (#3538).
    let output = if groups == 1 {
        crate::dyn_tensor_metal::native_conv1d_gemm(
            &input_tensor,
            &weight,
            bias_tensor.as_ref(),
            padding,
            stride,
            dilation,
            &out_shape,
        )
    } else {
        crate::dyn_tensor_metal::native_conv1d(
            &input_tensor,
            &weight,
            bias_tensor.as_ref(),
            padding,
            stride,
            dilation,
            groups,
        )
    }
    .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Conv1dGemm: {e}")))?;

    dyn_to_slice(&output, step_idx, "Conv1dGemm")
}

/// Execute a `NativeOpKind::ChannelsFirstLayerNorm` step.
///
/// Normalizes over dim 1 (channel dimension) of a `[B, C, T]` tensor.
/// Equivalent to Transpose(1,2) → LayerNorm → Transpose(1,2) but avoids
/// two data-copy transpose dispatches. Part of #3457.
pub(super) fn execute_native_channels_first_layer_norm(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    eps: f32,
    input_shape: &[usize],
    channels: usize,
    leaky_relu_slope: Option<f32>,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);

    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;

    let weights = &model.def.weight_buffers[step_idx];
    let input_tensor = slice_to_dyn(&input_slice, input_shape, dtype)?;
    let weight = weight_to_dyn(
        weights,
        "weight",
        &[channels],
        dtype,
        step_idx,
        "ChannelsFirstLayerNorm",
    )?;
    let bias = weight_to_dyn(
        weights,
        "bias",
        &[channels],
        dtype,
        step_idx,
        "ChannelsFirstLayerNorm",
    )?;

    let output = crate::dyn_tensor_metal::native_channels_first_layer_norm_with_activation(
        &input_tensor,
        &weight,
        &bias,
        f64::from(eps),
        leaky_relu_slope,
    )
    .map_err(|e| {
        native_dispatch_err(
            step_idx,
            format!("NativeOp ChannelsFirstLayerNorm: {e}"),
        )
    })?;

    dyn_to_slice(&output, step_idx, "ChannelsFirstLayerNorm")
}

/// Execute a `NativeOpKind::SiluMul` step.
///
/// Computes `silu(gate) * up` in a single Metal dispatch. Gate is edge 0,
/// up is edge 1. No weights. Part of #3521.
pub(super) fn execute_native_silu_mul(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    input_shape: &[usize],
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);
    let gate_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let up_slice = model.resolve_input_slice(step_idx, 1, buffers)?;

    let gate = slice_to_dyn(&gate_slice, input_shape, dtype)?;
    let up = slice_to_dyn(&up_slice, input_shape, dtype)?;

    // silu(gate) * up = (gate * sigmoid(gate)) * up
    let silu_gate = gate.silu()?;
    let output = silu_gate.mul(&up)?;

    dyn_to_slice(&output, step_idx, "SiluMul")
}

/// Execute a `NativeOpKind::RotaryEmbedding` step.
///
/// Applies rotary position embedding to the input tensor using pre-computed
/// cos/sin caches stored as weights. Delegates to `MetalDynBackend::gpu_rope`.
///
/// Part of #3526.
pub(super) fn execute_native_rope(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    input_shape: &[usize],
    head_dim: usize,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);
    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let x = slice_to_dyn(&input_slice, input_shape, dtype)?;

    let half_dim = head_dim / 2;
    let seq_len = input_shape[input_shape.len() - 2];
    let cos_shape = &[seq_len, half_dim];
    let sin_shape = &[seq_len, half_dim];

    let step_weights = &model.def.weight_buffers[step_idx];
    let cos = weight_to_dyn(step_weights, "cos_cache", cos_shape, dtype, step_idx, "RotaryEmbedding")?;
    let sin = weight_to_dyn(step_weights, "sin_cache", sin_shape, dtype, step_idx, "RotaryEmbedding")?;

    let output = nn_core::layers::attention::rope(&x, &cos, &sin)
        .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp RotaryEmbedding: {e}")))?;

    dyn_to_slice(&output, step_idx, "RotaryEmbedding")
}

/// Execute a `NativeOpKind::Int8Gemm` step (W8A16 dequantizing GEMM, #3522).
///
/// Reads INT8 weights, per-channel F32 scale/zero_point, and dispatches the
/// `int8_matmul_dequant` simdgroup kernel. Output is F32.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_native_int8_gemm(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    in_features: usize,
    out_features: usize,
    has_bias: bool,
    input_shape: &[usize],
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    use crate::compiled_model::int8_gemm_msl;

    // Resolve F32 activation input.
    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;

    // Resolve weight buffers.
    let step_weights = &model.def.weight_buffers[step_idx];
    let weight_int8_buf = step_weights.get("weight_int8").ok_or_else(|| {
        native_dispatch_err(step_idx, "NativeOp Int8Gemm: missing 'weight_int8'".into())
    })?;
    let scale_buf = step_weights.get("scale").ok_or_else(|| {
        native_dispatch_err(step_idx, "NativeOp Int8Gemm: missing 'scale'".into())
    })?;
    let zero_point_buf = step_weights.get("zero_point").ok_or_else(|| {
        native_dispatch_err(step_idx, "NativeOp Int8Gemm: missing 'zero_point'".into())
    })?;
    let bias_buf = if has_bias {
        Some(step_weights.get("bias").ok_or_else(|| {
            native_dispatch_err(step_idx, "NativeOp Int8Gemm: missing 'bias'".into())
        })?)
    } else {
        None
    };

    // Compute output dimensions.
    let batch_size: usize = input_shape.iter().rev().skip(1).product();
    let total_output = batch_size.checked_mul(out_features).ok_or_else(|| {
        native_dispatch_err(step_idx, "NativeOp Int8Gemm: output size overflow".into())
    })?;

    // Generate INT8 dequantizing GEMM MSL.
    let info = int8_gemm_msl::Int8GemmInfo {
        m: batch_size,
        k: in_features,
        n: out_features,
        has_bias,
    };
    let msl_source = int8_gemm_msl::generate_int8_gemm_msl(&info);
    let param_count = int8_gemm_msl::int8_gemm_input_count(has_bias);

    // Fixed MSL function name — PipelineCache differentiates by MSL source hash
    // (different M/K/N constants produce different source).
    let kernel_name = "int8_matmul_dequant";

    // Compile the kernel pipeline (cached by PipelineCache).
    let pipeline =
        KernelPipeline::from_msl(cache, &msl_source, kernel_name, param_count, false).map_err(
            |e| native_dispatch_err(step_idx, format!("NativeOp Int8Gemm pipeline: {e}")),
        )?;

    // Allocate F32 output buffer.
    let out_bytes = total_output.checked_mul(4).ok_or_else(|| {
        native_dispatch_err(step_idx, "NativeOp Int8Gemm: output bytes overflow".into())
    })?;
    let ctx = cache.context();
    let (out_buf, out_offset) = crate::arena::arena_alloc_or_create(ctx, out_bytes)
        .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Int8Gemm alloc: {e}")))?;

    // Build simdgroup dispatch plan (32x32 tiles).
    let m_u32 = u32::try_from(batch_size)
        .map_err(|_| native_dispatch_err(step_idx, "NativeOp Int8Gemm: batch exceeds u32".into()))?;
    let n_u32 = u32::try_from(out_features).map_err(|_| {
        native_dispatch_err(
            step_idx,
            "NativeOp Int8Gemm: out_features exceeds u32".into(),
        )
    })?;
    let tg_bytes = int8_gemm_msl::int8_gemm_threadgroup_bytes();
    let plan = DispatchMode::Grid3D {
        grid: [n_u32.div_ceil(32), m_u32.div_ceil(32), 1],
        threads: [32, 4, 1],
    }
    .plan()
    .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Int8Gemm plan: {e}")))?
    .with_output_elems(total_output)
    .with_constants(vec![])
    .with_use_threadgroups(true)
    .with_threadgroup_memory_bytes(Some(tg_bytes));

    // Bind buffers: input, weight_int8, scale, zero_point, [bias], output.
    let mut inputs: Vec<&crate::buffer::MetalBuffer> = Vec::with_capacity(param_count);
    inputs.push(input_slice.buffer());
    inputs.push(weight_int8_buf);
    inputs.push(scale_buf);
    inputs.push(zero_point_buf);
    if let Some(b) = bias_buf {
        inputs.push(b);
    }
    let mut offsets = vec![input_slice.byte_offset()];
    // Weight, scale, zero_point, bias have zero offset.
    offsets.resize(param_count, 0);

    // Dispatch: single simdgroup Metal kernel.
    pipeline
        .dispatch_buffers_with_all_offsets(ctx, &inputs, &offsets, &out_buf, out_offset, &plan)
        .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp Int8Gemm dispatch: {e}")))?;

    Ok(GpuSlice::from_ref(&out_buf, out_offset))
}

/// Short tag for [`GemmActivation`] in kernel names.
fn activation_tag(act: &GemmActivation) -> &'static str {
    match act {
        GemmActivation::Relu => "relu",
        GemmActivation::Gelu => "gelu",
        GemmActivation::GeluErf => "geluerf",
        GemmActivation::Sigmoid => "sig",
        GemmActivation::Silu => "silu",
        GemmActivation::Tanh => "tanh",
        _ => "unsupported_activation",
    }
}

/// Execute a `NativeOpKind::BatchNorm2d` step (fused, #4324).
///
/// Resolves the graph input and pre-uploaded running_mean, running_var,
/// optional weight/bias. Calls the fused single-dispatch GPU kernel.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_native_batch_norm_2d(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    eps: f32,
    num_channels: usize,
    input_shape: &[usize],
    has_weight: bool,
    has_bias: bool,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);

    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;

    let weights = &model.def.weight_buffers[step_idx];
    let input_tensor = slice_to_dyn(&input_slice, input_shape, dtype)?;
    let running_mean = weight_to_dyn(
        weights,
        "running_mean",
        &[num_channels],
        dtype,
        step_idx,
        "BatchNorm2d",
    )?;
    let running_var = weight_to_dyn(
        weights,
        "running_var",
        &[num_channels],
        dtype,
        step_idx,
        "BatchNorm2d",
    )?;

    let weight_tensor = if has_weight {
        Some(weight_to_dyn(
            weights,
            "weight",
            &[num_channels],
            dtype,
            step_idx,
            "BatchNorm2d",
        )?)
    } else {
        None
    };
    let bias_tensor = if has_bias {
        Some(weight_to_dyn(
            weights,
            "bias",
            &[num_channels],
            dtype,
            step_idx,
            "BatchNorm2d",
        )?)
    } else {
        None
    };

    let output = crate::dyn_tensor_metal::native_batch_norm_2d(
        &input_tensor,
        &running_mean,
        &running_var,
        weight_tensor.as_ref(),
        bias_tensor.as_ref(),
        f64::from(eps),
    )
    .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp BatchNorm2d: {e}")))?;

    dyn_to_slice(&output, step_idx, "BatchNorm2d")
}
