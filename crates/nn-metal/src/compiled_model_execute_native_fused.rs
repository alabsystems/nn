// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused native operation helpers for `CompiledModel`: AdaIN variants and
//! Flash Attention.
//!
//! Extracted from `compiled_model_execute_native.rs` to keep files under
//! 450 lines. See #2544.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Result;
use nn_dsl::PrecisionTier;

use crate::cache::PipelineCache;
use crate::gpu_slice::GpuSlice;

use super::CompiledModel;
use super::{dyn_to_slice, native_dispatch_err, slice_to_dyn, weight_to_dyn};

#[path = "compiled_model_execute_native_resblock.rs"]
mod resblock_exec;

pub(super) use resblock_exec::execute_native_fused_resblock;

#[path = "compiled_model_execute_native_norm_conv.rs"]
mod norm_conv_exec;

pub(super) use norm_conv_exec::execute_native_norm_activ_conv1d;

#[path = "compiled_model_execute_native_conv1d_activation.rs"]
mod conv1d_activation_exec;

pub(crate) use conv1d_activation_exec::execute_native_conv1d_activation;

#[path = "compiled_model_execute_native_fused_adain_direct.rs"]
mod adain_direct;

#[path = "compiled_model_execute_native_fused_upsample_conv1d_direct.rs"]
mod upsample_conv1d_direct;

#[path = "compiled_model_execute_native_fused_snake_norm_direct.rs"]
mod snake_norm_direct;

/// Execute a `NativeOpKind::AdainSnake` step.
///
/// Resolves 3 graph inputs (x, gamma, beta) and 1 weight (alpha), wraps
/// as DynTensors, calls the fused AdaIN+Snake kernel. Part of #2472.
pub(super) fn execute_native_adain_snake(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    eps: f32,
    input_shape: &[usize],
    channels: usize,
    residual_gamma: bool,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);

    // Resolve graph inputs: x (0), gamma (1), beta (2).
    let x_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let gamma_slice = model.resolve_input_slice(step_idx, 1, buffers)?;
    let beta_slice = model.resolve_input_slice(step_idx, 2, buffers)?;

    let batch = input_shape[0];
    let gamma_shape = [batch, channels, 1];
    let x_tensor = slice_to_dyn(&x_slice, input_shape, dtype)?;
    let gamma_tensor = slice_to_dyn(&gamma_slice, &gamma_shape, dtype)?;
    let beta_tensor = slice_to_dyn(&beta_slice, &gamma_shape, dtype)?;

    let weights = &model.def.weight_buffers[step_idx];
    let alpha_tensor = weight_to_dyn(weights, "alpha", &[channels], dtype, step_idx, "AdainSnake")?;

    // When Strict precision is requested, route to the IR-decomposed path.
    // Both paths now use Kahan-compensated reductions (#2696), so this is
    // only needed for models that explicitly require decomposed dispatch (#2704).
    let use_kahan = model
        .precision()
        .map_or(false, |c| c.tier == PrecisionTier::Strict);
    let output = if use_kahan {
        // The decomposed path requires alpha rank to match input rank:
        // [1, C, 1] not [C] (right-aligned broadcast correctness).
        // Reshape cannot fail (same element count), but wrap in closure
        // so any error reaches the outer map_err for step_idx context.
        (|| -> Result<DynTensor> {
            let alpha_precise = alpha_tensor.reshape([1, channels, 1])?;
            crate::dyn_tensor_metal::native_adain_snake_precise(
                &x_tensor,
                &gamma_tensor,
                &beta_tensor,
                &alpha_precise,
                f64::from(eps),
                residual_gamma,
            )
        })()
    } else {
        crate::dyn_tensor_metal::native_adain_snake(
            &x_tensor,
            &gamma_tensor,
            &beta_tensor,
            &alpha_tensor,
            f64::from(eps),
            residual_gamma,
        )
    }
    .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp AdainSnake: {e}")))?;

    dyn_to_slice(&output, step_idx, "AdainSnake")
}

/// Execute a `NativeOpKind::AdainLeakyRelu` step.
///
/// Resolves 3 graph inputs (x, gamma, beta), wraps as DynTensors, calls
/// the fused AdaIN+LeakyRelu kernel. No per-channel weights. Part of #2472.
pub(super) fn execute_native_adain_leaky_relu(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    eps: f32,
    slope: f32,
    input_shape: &[usize],
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);

    // Resolve graph inputs: x (0), gamma (1), beta (2).
    let x_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let gamma_slice = model.resolve_input_slice(step_idx, 1, buffers)?;
    let beta_slice = model.resolve_input_slice(step_idx, 2, buffers)?;

    let batch = input_shape[0];
    let channels = input_shape[1];
    let gamma_shape = [batch, channels, 1];
    let x_tensor = slice_to_dyn(&x_slice, input_shape, dtype)?;
    let gamma_tensor = slice_to_dyn(&gamma_slice, &gamma_shape, dtype)?;
    let beta_tensor = slice_to_dyn(&beta_slice, &gamma_shape, dtype)?;

    // When Strict precision is requested, decompose into precise InstanceNorm
    // + manual affine + leaky_relu. Both paths now use Kahan-compensated
    // reductions (#2696); this branch is only for explicit Strict requests (#2704).
    let use_kahan = model
        .precision()
        .map_or(false, |c| c.tier == PrecisionTier::Strict);
    let output = if use_kahan {
        // Closure so `?` returns from the closure, not the function.
        // All errors reach the outer `map_err` for step_idx context.
        (|| -> Result<DynTensor> {
            let normed =
                crate::dyn_tensor_metal::native_instance_norm_precise(&x_tensor, f64::from(eps))?;
            // AdaIN affine: (1 + gamma) * normed + beta
            // gamma/beta are [B, C, 1], normed is [B, C, T] — broadcast is correct.
            let gamma_normed = normed.mul(&gamma_tensor)?;
            let affined = normed.add(&gamma_normed)?.add(&beta_tensor)?;
            affined.leaky_relu(f64::from(slope))
        })()
    } else {
        crate::dyn_tensor_metal::native_adain_leaky_relu(
            &x_tensor,
            &gamma_tensor,
            &beta_tensor,
            f64::from(eps),
            f64::from(slope),
        )
    }
    .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp AdainLeakyRelu: {e}")))?;

    dyn_to_slice(&output, step_idx, "AdainLeakyRelu")
}

/// Execute a `NativeOpKind::FusedAdainSnake` step.
///
/// Fuses InstanceNorm + affine(gamma, beta) + Snake(alpha) into a single
/// Metal dispatch. Gamma and beta are runtime graph inputs (from style
/// projections in Kokoro), resolved via `external_node_ids`. Alpha is a
/// static weight. Detected by peephole from the trace pattern:
/// `InstanceNorm(x)` -> `Mul(gamma)` -> `Add(beta)` -> `Snake(alpha)`.
///
/// Uses **direct Metal dispatch** for F32/F16: encodes the MSL kernel
/// directly on GpuSlice buffer/offset pairs, eliminating 4 DynTensor
/// wrappings + 1 gpu_data extraction. Falls back to the DynTensor bridge
/// path for unsupported dtypes (BF16).
///
/// Part of #4252 / #4449.
pub(super) fn execute_native_fused_adain_snake(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    eps: f32,
    input_shape: &[usize],
    channels: usize,
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    // Direct dispatch for F32/F16 — bypasses DynTensor bridge entirely.
    let scalar_type = model.step_scalar_type(step_idx);
    if adain_direct::supports_scalar_type(scalar_type) {
        return adain_direct::execute_fused_adain_snake_direct(
            model, step_idx, buffers, eps, input_shape, channels, cache,
        );
    }

    // Fallback: DynTensor bridge path for unsupported scalar types.
    let dtype = model.step_dtype(step_idx);
    let batch = input_shape[0];

    // Resolve graph inputs: x (0), gamma (1), beta (2).
    let x_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let gamma_slice = model.resolve_input_slice(step_idx, 1, buffers)?;
    let beta_slice = model.resolve_input_slice(step_idx, 2, buffers)?;

    let gamma_shape = [batch, channels, 1];
    let x_tensor = slice_to_dyn(&x_slice, input_shape, dtype)?;
    let gamma_tensor = slice_to_dyn(&gamma_slice, &gamma_shape, dtype)?;
    let beta_tensor = slice_to_dyn(&beta_slice, &gamma_shape, dtype)?;

    // Alpha is the only static weight (from Snake activation).
    let weights = &model.def.weight_buffers[step_idx];
    let alpha_tensor =
        weight_to_dyn(weights, "alpha", &[channels], dtype, step_idx, "FusedAdainSnake")?;

    // Delegate to existing fused AdaIN+Snake kernel.
    // residual_gamma=false: standard AdaIN convention gamma*normed+beta.
    let output = crate::dyn_tensor_metal::native_adain_snake(
        &x_tensor,
        &gamma_tensor,
        &beta_tensor,
        &alpha_tensor,
        f64::from(eps),
        false, // standard gamma*normed+beta
    )
    .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp FusedAdainSnake: {e}")))?;

    dyn_to_slice(&output, step_idx, "FusedAdainSnake")
}

/// Execute a `NativeOpKind::FusedInstanceNormMulAdd` step.
///
/// Fuses InstanceNorm + Mul(gamma) + Add(beta) into sequential DynTensor ops.
/// Gamma and beta are runtime graph inputs (from style projections in Kokoro),
/// resolved via `external_node_ids`. No static weights.
///
/// Semantically: `instance_norm(x) * gamma + beta`
///
/// Near-term: calls instance_norm then mul then add sequentially.
/// Future: a single Metal kernel that fuses all three.
/// Part of #4252.
pub(super) fn execute_native_fused_instance_norm_mul_add(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    eps: f32,
    input_shape: &[usize],
    channels: usize,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);
    let batch = input_shape[0];

    // Resolve graph inputs: x (0), gamma (1), beta (2).
    // Gamma and beta are runtime style-projection outputs, NOT static weights.
    let x_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let gamma_slice = model.resolve_input_slice(step_idx, 1, buffers)?;
    let beta_slice = model.resolve_input_slice(step_idx, 2, buffers)?;

    let gamma_shape = [batch, channels, 1];
    let x_tensor = slice_to_dyn(&x_slice, input_shape, dtype)?;
    let gamma_tensor = slice_to_dyn(&gamma_slice, &gamma_shape, dtype)?;
    let beta_tensor = slice_to_dyn(&beta_slice, &gamma_shape, dtype)?;

    // Sequential decomposition: instance_norm(x) * gamma + beta.
    // This delegates to existing GPU-fused instance_norm, then elementwise
    // mul and add. Total: 3 GPU dispatches (same as unfused, but in 1 plan step).
    // Future optimization: single Metal kernel for the full operation.
    let normed = crate::dyn_tensor_metal::native_instance_norm(&x_tensor, f64::from(eps))
        .map_err(|e| {
            native_dispatch_err(step_idx, format!("NativeOp FusedInstanceNormMulAdd norm: {e}"))
        })?;
    let scaled = normed.mul(&gamma_tensor).map_err(|e| {
        native_dispatch_err(step_idx, format!("NativeOp FusedInstanceNormMulAdd mul: {e}"))
    })?;
    let output = scaled.add(&beta_tensor).map_err(|e| {
        native_dispatch_err(step_idx, format!("NativeOp FusedInstanceNormMulAdd add: {e}"))
    })?;

    dyn_to_slice(&output, step_idx, "FusedInstanceNormMulAdd")
}

/// Execute a `NativeOpKind::FusedUpsampleConv1d` step.
///
/// Fuses nearest-neighbor upsample1d + conv1d into a single Metal dispatch.
/// The MSL kernel reads `[B, C_in, T]` and computes nearest-neighbor upsample
/// inline during Conv1d accumulation, writing `[B, C_out, T_out]` directly.
/// No intermediate upsampled buffer is materialized.
///
/// Uses **direct Metal dispatch** for F32/F16: encodes the MSL kernel
/// directly on GpuSlice buffer/offset pairs, eliminating 3 DynTensor
/// wrappings + 1 gpu_data extraction. Falls back to the DynTensor bridge
/// path for unsupported dtypes (BF16).
///
/// The f0_energy Kokoro segment has 6 pairs of upsample+conv; fusing each
/// from 3+ dispatches into 1 saves 12+ dispatches total.
/// Part of #4310.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_native_fused_upsample_conv1d(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    upsample_factor: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    input_shape: &[usize],
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    // Direct dispatch for F32/F16 — bypasses DynTensor bridge entirely.
    let scalar_type = model.step_scalar_type(step_idx);
    if upsample_conv1d_direct::supports_scalar_type(scalar_type) {
        return upsample_conv1d_direct::execute_fused_upsample_conv1d_direct(
            model,
            step_idx,
            buffers,
            upsample_factor,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            input_shape,
            cache,
        );
    }

    // Fallback: DynTensor bridge path for unsupported scalar types.
    let dtype = model.step_dtype(step_idx);

    // Resolve graph input: x (0) with shape [B, C_in, T].
    let x_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let x_tensor = slice_to_dyn(&x_slice, input_shape, dtype)?;

    // Load weights: weight [C_out, C_in, K], bias [C_out].
    let weights = &model.def.weight_buffers[step_idx];
    let weight_tensor = weight_to_dyn(
        weights,
        "weight",
        &[out_channels, input_shape[1], kernel_size],
        dtype,
        step_idx,
        "FusedUpsampleConv1d",
    )?;
    let bias_tensor = weight_to_dyn(
        weights,
        "bias",
        &[out_channels],
        dtype,
        step_idx,
        "FusedUpsampleConv1d",
    )?;

    // Single fused GPU dispatch: upsample + conv1d in one MSL kernel.
    let output = crate::dyn_tensor_metal::native_fused_upsample_conv1d(
        &x_tensor,
        &weight_tensor,
        &bias_tensor,
        upsample_factor,
        padding,
        stride,
    )
    .map_err(|e| native_dispatch_err(step_idx, format!("FusedUpsampleConv1d: {e}")))?;

    dyn_to_slice(&output, step_idx, "FusedUpsampleConv1d")
}

/// Execute a `NativeOpKind::AdaLayerNorm` step (#2482).
///
/// 3 graph inputs (x, gamma, beta) + 2 weights (norm_weight, norm_bias).
pub(super) fn execute_native_ada_layer_norm(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    eps: f32,
    input_shape: &[usize],
    hidden_dim: usize,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);
    let weights = &model.def.weight_buffers[step_idx];

    let x_tensor = slice_to_dyn(
        &model.resolve_input_slice(step_idx, 0, buffers)?,
        input_shape,
        dtype,
    )?;
    let batch = input_shape[0];
    let gb_shape = [batch, 1, hidden_dim];
    let gamma_tensor = slice_to_dyn(
        &model.resolve_input_slice(step_idx, 1, buffers)?,
        &gb_shape,
        dtype,
    )?;
    let beta_tensor = slice_to_dyn(
        &model.resolve_input_slice(step_idx, 2, buffers)?,
        &gb_shape,
        dtype,
    )?;
    let time_steps: usize = input_shape[1..input_shape.len() - 1].iter().product();
    let norm_weight = weight_to_dyn(
        weights,
        "norm_weight",
        &[hidden_dim],
        dtype,
        step_idx,
        "AdaLayerNorm",
    )?;
    let norm_bias = weight_to_dyn(
        weights,
        "norm_bias",
        &[hidden_dim],
        dtype,
        step_idx,
        "AdaLayerNorm",
    )?;

    let output = crate::dyn_tensor_metal::native_ada_layer_norm(
        &x_tensor,
        &gamma_tensor,
        &beta_tensor,
        &norm_weight,
        &norm_bias,
        f64::from(eps),
        time_steps,
    )
    .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp AdaLayerNorm: {e}")))?;

    dyn_to_slice(&output, step_idx, "AdaLayerNorm")
}

/// Execute a `NativeOpKind::FlashAttention` step (#2434).
///
/// Resolves 3 graph inputs (Q, K, V), wraps as DynTensors, calls the fused
/// Flash Attention kernel (online softmax, O(1) attention matrix memory).
/// No weight buffers needed.
pub(super) fn execute_native_flash_attention(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    scale: f32,
    causal: bool,
    q_shape: &[usize],
    k_shape: &[usize],
    input_layout: nn_dsl::AttentionLayout,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);

    // Resolve graph inputs: Q (0), K (1), V (2).
    let q_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let k_slice = model.resolve_input_slice(step_idx, 1, buffers)?;
    let v_slice = model.resolve_input_slice(step_idx, 2, buffers)?;

    let q_tensor = slice_to_dyn(&q_slice, q_shape, dtype)?;
    let k_tensor = slice_to_dyn(&k_slice, k_shape, dtype)?;
    // V shape matches K shape.
    let v_tensor = slice_to_dyn(&v_slice, k_shape, dtype)?;

    let output = match input_layout {
        nn_dsl::AttentionLayout::HeadsFirst => crate::dyn_tensor_metal::native_flash_attention(
            &q_tensor,
            &k_tensor,
            &v_tensor,
            f64::from(scale),
            causal,
        ),
        nn_dsl::AttentionLayout::SeqFirst => {
            crate::dyn_tensor_metal::native_flash_attention_seq_first(
                &q_tensor,
                &k_tensor,
                &v_tensor,
                f64::from(scale),
                causal,
            )
        }
        _ => {
            return Err(native_dispatch_err(
                step_idx,
                format!("NativeOp FlashAttention: unsupported AttentionLayout {input_layout:?}"),
            ))
        }
    }
    .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp FlashAttention: {e}")))?;

    dyn_to_slice(&output, step_idx, "FlashAttention")
}

/// Execute a `NativeOpKind::BatchedStyleProjection` step (#1815 Tier 1).
///
/// One matmul + bias_add for all FusedResBlocks in a segment. The output
/// `[B, total_out]` is sliced by each FusedResBlock via zero-copy narrow.
pub(super) fn execute_native_batched_style_projection(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    style_dim: usize,
    total_out: usize,
    style_step: usize,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);

    // Resolve style embedding input from the referenced step.
    let style_slice = buffers[style_step]
        .as_ref()
        .map(GpuSlice::alias)
        .ok_or_else(|| {
            native_dispatch_err(
                step_idx,
                format!("BatchedStyleProjection: style_step {style_step} has no buffer"),
            )
        })?;

    let batch = {
        let byte_width = dtype.size_bytes();
        // Buffer length covers the full allocation; subtract byte_offset for this slice.
        let slice_bytes = style_slice.buffer().len() - style_slice.byte_offset();
        slice_bytes / (style_dim * byte_width)
    };

    let style_tensor = slice_to_dyn(&style_slice, &[batch, style_dim], dtype)?;

    // Load concatenated weight [total_out, style_dim] and bias [total_out].
    let weights = &model.def.weight_buffers[step_idx];
    let weight_t = weight_to_dyn(
        weights,
        "weight_t",
        &[style_dim, total_out],
        dtype,
        step_idx,
        "BatchedStyleProjection",
    )?;
    let bias = weight_to_dyn(
        weights,
        "bias",
        &[total_out],
        dtype,
        step_idx,
        "BatchedStyleProjection",
    )?;

    let projected = style_tensor.matmul(&weight_t)?;
    let output = projected.broadcast_add(&bias)?;

    dyn_to_slice(&output, step_idx, "BatchedStyleProjection")
}

/// Execute a `NativeOpKind::FusedSnakeInstanceNorm` step.
///
/// Fuses Snake activation (`y = x + (1/alpha) * sin(alpha*x)^2`) with
/// InstanceNorm (per-channel normalize) into a single Metal dispatch.
/// Alpha is a static weight. Detected by peephole from the trace pattern:
/// `snake_tensor(x)` -> `InstanceNorm(snake_output)`.
///
/// Uses **direct Metal dispatch** for F32/F16: encodes the MSL kernel
/// directly on GpuSlice buffer/offset pairs. Falls back to the DynTensor
/// bridge path for unsupported dtypes (BF16).
///
/// Part of #4264.
pub(super) fn execute_native_fused_snake_instance_norm(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    eps: f32,
    input_shape: &[usize],
    channels: usize,
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    // Direct dispatch for F32/F16 — bypasses DynTensor bridge entirely.
    let scalar_type = model.step_scalar_type(step_idx);
    if snake_norm_direct::supports_scalar_type(scalar_type) {
        return snake_norm_direct::execute_fused_snake_instance_norm_direct(
            model, step_idx, buffers, eps, input_shape, channels, cache,
        );
    }

    // Fallback: DynTensor bridge path for unsupported scalar types.
    let dtype = model.step_dtype(step_idx);

    // Resolve graph input: x (0) with shape [B, C, T].
    let x_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let x_tensor = slice_to_dyn(&x_slice, input_shape, dtype)?;

    // Alpha is a static weight: [C] per-channel Snake parameter.
    let weights = &model.def.weight_buffers[step_idx];
    let alpha_tensor = weight_to_dyn(
        weights, "alpha", &[channels], dtype, step_idx, "FusedSnakeInstanceNorm",
    )?;

    // Sequential decomposition: snake(x) -> instance_norm.
    // snake(x) = x + (1/alpha) * sin(alpha * x)^2
    let alpha_3d = alpha_tensor.reshape([1, channels, 1]).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedSnakeInstanceNorm alpha reshape: {e}"))
    })?;
    let ax = x_tensor.mul(&alpha_3d).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedSnakeInstanceNorm a*x: {e}"))
    })?;
    let sin_ax = ax.sin().map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedSnakeInstanceNorm sin: {e}"))
    })?;
    let sin_sq = sin_ax.mul(&sin_ax).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedSnakeInstanceNorm sin^2: {e}"))
    })?;
    let inv_alpha = alpha_3d.recip().map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedSnakeInstanceNorm 1/a: {e}"))
    })?;
    let snake_term = sin_sq.mul(&inv_alpha).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedSnakeInstanceNorm snake_term: {e}"))
    })?;
    let snake_out = x_tensor.add(&snake_term).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedSnakeInstanceNorm snake add: {e}"))
    })?;

    // InstanceNorm on the Snake output.
    let output = crate::dyn_tensor_metal::native_instance_norm(&snake_out, f64::from(eps))
        .map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedSnakeInstanceNorm norm: {e}"))
        })?;

    dyn_to_slice(&output, step_idx, "FusedSnakeInstanceNorm")
}

/// Execute a `NativeOpKind::FusedConv1dSnakeNorm` step.
///
/// Fuses Conv1d + Snake activation + InstanceNorm into a single logical
/// NativeOp. The three operations execute sequentially via DynTensor GPU ops,
/// and the lazy command buffer batches them. This avoids materializing the
/// intermediate conv and snake output buffers as separate plan steps.
///
/// Part of #4264.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_native_conv1d_snake_norm(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    has_bias: bool,
    eps: f32,
    input_shape: &[usize],
    _cache: &PipelineCache,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);

    // Resolve graph input: x (0) with shape [B, C_in, L_in].
    let x_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let x_tensor = slice_to_dyn(&x_slice, input_shape, dtype)?;

    // Load conv weights.
    let weights = &model.def.weight_buffers[step_idx];
    let in_channels = if groups > 0 {
        input_shape[1] / groups * groups
    } else {
        input_shape[1]
    };
    let weight_shape = [out_channels, in_channels / groups, kernel_size];
    let conv_weight = weight_to_dyn(
        weights,
        "conv_weight",
        &weight_shape,
        dtype,
        step_idx,
        "FusedConv1dSnakeNorm",
    )?;

    // Phase 1: Conv1d.
    let conv_output = if has_bias {
        let conv_bias = weight_to_dyn(
            weights,
            "conv_bias",
            &[out_channels],
            dtype,
            step_idx,
            "FusedConv1dSnakeNorm",
        )?;
        let y = x_tensor
            .conv1d(&conv_weight, padding, stride, dilation, groups)
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("FusedConv1dSnakeNorm conv1d: {e}"))
            })?;
        let bias_reshaped = conv_bias.reshape([1, out_channels, 1]).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dSnakeNorm bias reshape: {e}"))
        })?;
        y.broadcast_add(&bias_reshaped).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dSnakeNorm bias add: {e}"))
        })?
    } else {
        x_tensor
            .conv1d(&conv_weight, padding, stride, dilation, groups)
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("FusedConv1dSnakeNorm conv1d: {e}"))
            })?
    };

    // Phase 2: Snake activation — x + (1/alpha) * sin(alpha*x)^2.
    let alpha = weight_to_dyn(
        weights,
        "alpha",
        &[out_channels],
        dtype,
        step_idx,
        "FusedConv1dSnakeNorm",
    )?;
    let alpha_3d = alpha.reshape([1, out_channels, 1]).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNorm alpha reshape: {e}"))
    })?;
    let ax = conv_output.broadcast_mul(&alpha_3d).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNorm alpha*x: {e}"))
    })?;
    let sin_ax = ax.sin().map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNorm sin: {e}"))
    })?;
    let sin2 = sin_ax.mul(&sin_ax).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNorm sin^2: {e}"))
    })?;
    let inv_alpha = alpha_3d.recip().map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNorm 1/alpha: {e}"))
    })?;
    let scaled = sin2.broadcast_mul(&inv_alpha).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNorm sin^2/alpha: {e}"))
    })?;
    let snake_output = conv_output.add(&scaled).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNorm snake add: {e}"))
    })?;

    // Phase 3: InstanceNorm on the Snake output.
    let output = crate::dyn_tensor_metal::native_instance_norm(&snake_output, f64::from(eps))
        .map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dSnakeNorm norm: {e}"))
        })?;

    dyn_to_slice(&output, step_idx, "FusedConv1dSnakeNorm")
}

/// Execute a `NativeOpKind::FusedConv1dSnakeNormResBlock` step.
///
/// Fuses 2x (Conv1d + Snake + InstanceNorm) + residual add into a single
/// logical NativeOp. Sequences:
///   Phase 1: conv1d(x) -> snake -> instance_norm
///   Phase 2: conv1d(phase1_out) -> snake -> instance_norm
///   Residual: phase2_out + x * residual_scale
///
/// Weight keys use `p1_` and `p2_` prefixes for the two phases.
/// Part of #4264.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_native_conv1d_snake_norm_resblock(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    p1_out_channels: usize,
    p1_kernel_size: usize,
    p1_padding: usize,
    p1_dilation: usize,
    p1_has_bias: bool,
    p2_out_channels: usize,
    p2_kernel_size: usize,
    p2_padding: usize,
    p2_dilation: usize,
    p2_has_bias: bool,
    eps: f32,
    residual_scale: f32,
    input_shape: &[usize],
    x_step: usize,
    _cache: &PipelineCache,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);
    let weights = &model.def.weight_buffers[step_idx];

    // Resolve residual x from the referenced step.
    let x_slice = buffers[x_step]
        .as_ref()
        .map(GpuSlice::alias)
        .ok_or_else(|| {
            native_dispatch_err(
                step_idx,
                format!("FusedConv1dSnakeNormResBlock: x_step {x_step} has no buffer"),
            )
        })?;
    let x_tensor = slice_to_dyn(&x_slice, input_shape, dtype)?;

    // ---- Phase 1: Conv1d -> Snake -> InstanceNorm ----
    let p1_in_channels = input_shape[1];
    let p1_weight = weight_to_dyn(
        weights,
        "p1_conv_weight",
        &[p1_out_channels, p1_in_channels, p1_kernel_size],
        dtype,
        step_idx,
        "FusedConv1dSnakeNormResBlock",
    )?;

    let p1_conv = if p1_has_bias {
        let p1_bias = weight_to_dyn(
            weights,
            "p1_conv_bias",
            &[p1_out_channels],
            dtype,
            step_idx,
            "FusedConv1dSnakeNormResBlock",
        )?;
        let y = x_tensor
            .conv1d(&p1_weight, p1_padding, 1, p1_dilation, 1)
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p1 conv: {e}"))
            })?;
        let bias_r = p1_bias.reshape([1, p1_out_channels, 1]).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p1 bias reshape: {e}"))
        })?;
        y.broadcast_add(&bias_r).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p1 bias add: {e}"))
        })?
    } else {
        x_tensor
            .conv1d(&p1_weight, p1_padding, 1, p1_dilation, 1)
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p1 conv: {e}"))
            })?
    };

    // Snake: x + (1/alpha) * sin(alpha*x)^2
    let p1_alpha = weight_to_dyn(
        weights,
        "p1_alpha",
        &[p1_out_channels],
        dtype,
        step_idx,
        "FusedConv1dSnakeNormResBlock",
    )?;
    let p1_a3d = p1_alpha.reshape([1, p1_out_channels, 1]).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p1 alpha reshape: {e}"))
    })?;
    let p1_ax = p1_conv.broadcast_mul(&p1_a3d).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p1 a*x: {e}"))
    })?;
    let p1_sin = p1_ax.sin().map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p1 sin: {e}"))
    })?;
    let p1_sin2 = p1_sin.mul(&p1_sin).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p1 sin^2: {e}"))
    })?;
    let p1_inv = p1_a3d.recip().map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p1 1/a: {e}"))
    })?;
    let p1_scaled = p1_sin2.broadcast_mul(&p1_inv).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p1 scale: {e}"))
    })?;
    let p1_snake = p1_conv.add(&p1_scaled).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p1 snake add: {e}"))
    })?;

    // InstanceNorm
    let p1_out =
        crate::dyn_tensor_metal::native_instance_norm(&p1_snake, f64::from(eps)).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p1 norm: {e}"))
        })?;

    // ---- Phase 2: Conv1d -> Snake -> InstanceNorm ----
    let p2_weight = weight_to_dyn(
        weights,
        "p2_conv_weight",
        &[p2_out_channels, p1_out_channels, p2_kernel_size],
        dtype,
        step_idx,
        "FusedConv1dSnakeNormResBlock",
    )?;

    let p2_conv = if p2_has_bias {
        let p2_bias = weight_to_dyn(
            weights,
            "p2_conv_bias",
            &[p2_out_channels],
            dtype,
            step_idx,
            "FusedConv1dSnakeNormResBlock",
        )?;
        let y = p1_out
            .conv1d(&p2_weight, p2_padding, 1, p2_dilation, 1)
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p2 conv: {e}"))
            })?;
        let bias_r = p2_bias.reshape([1, p2_out_channels, 1]).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p2 bias reshape: {e}"))
        })?;
        y.broadcast_add(&bias_r).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p2 bias add: {e}"))
        })?
    } else {
        p1_out
            .conv1d(&p2_weight, p2_padding, 1, p2_dilation, 1)
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p2 conv: {e}"))
            })?
    };

    let p2_alpha = weight_to_dyn(
        weights,
        "p2_alpha",
        &[p2_out_channels],
        dtype,
        step_idx,
        "FusedConv1dSnakeNormResBlock",
    )?;
    let p2_a3d = p2_alpha.reshape([1, p2_out_channels, 1]).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p2 alpha reshape: {e}"))
    })?;
    let p2_ax = p2_conv.broadcast_mul(&p2_a3d).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p2 a*x: {e}"))
    })?;
    let p2_sin = p2_ax.sin().map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p2 sin: {e}"))
    })?;
    let p2_sin2 = p2_sin.mul(&p2_sin).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p2 sin^2: {e}"))
    })?;
    let p2_inv = p2_a3d.recip().map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p2 1/a: {e}"))
    })?;
    let p2_scaled = p2_sin2.broadcast_mul(&p2_inv).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p2 scale: {e}"))
    })?;
    let p2_snake = p2_conv.add(&p2_scaled).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p2 snake add: {e}"))
    })?;

    let p2_out =
        crate::dyn_tensor_metal::native_instance_norm(&p2_snake, f64::from(eps)).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock p2 norm: {e}"))
        })?;

    // ---- Residual add ----
    let output = if (residual_scale - 1.0).abs() < 1e-6 {
        p2_out.add(&x_tensor).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock residual add: {e}"))
        })?
    } else {
        let scaled_x = x_tensor.affine(f64::from(residual_scale), 0.0).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock residual scale: {e}"))
        })?;
        p2_out.add(&scaled_x).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dSnakeNormResBlock residual add: {e}"))
        })?
    };

    dyn_to_slice(&output, step_idx, "FusedConv1dSnakeNormResBlock")
}

/// Execute a `NativeOpKind::FusedAddInstanceNormConv1x1` step.
///
/// Fuses Add + InstanceNorm + Conv1d(K=1) into a single logical NativeOp.
/// Sequences:
///   1. sum = x + h  (element-wise add of two [B, C_in, T] inputs)
///   2. normed = instance_norm(sum, eps)
///   3. output = conv1d(normed, weight_1x1, bias)  [B, C_out, T]
///
/// CPU fallback — decomposes to sequential DynTensor ops.
/// Part of #4264.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_native_fused_add_instance_norm_conv1x1(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    eps: f32,
    input_shape: &[usize],
    in_channels: usize,
    out_channels: usize,
    has_bias: bool,
    _cache: &PipelineCache,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);

    // Resolve graph inputs: x (0) and h (1), both [B, C_in, T].
    let x_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let x_tensor = slice_to_dyn(&x_slice, input_shape, dtype)?;

    let h_slice = model.resolve_input_slice(step_idx, 1, buffers)?;
    let h_tensor = slice_to_dyn(&h_slice, input_shape, dtype)?;

    // Step 1: Element-wise add.
    let sum = x_tensor.add(&h_tensor).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedAddInstanceNormConv1x1 add: {e}"))
    })?;

    // Step 2: InstanceNorm.
    let normed = crate::dyn_tensor_metal::native_instance_norm(&sum, f64::from(eps)).map_err(
        |e| native_dispatch_err(step_idx, format!("FusedAddInstanceNormConv1x1 norm: {e}")),
    )?;

    // Step 3: Conv1d with kernel_size=1, stride=1, padding=0, dilation=1, groups=1.
    let weights = &model.def.weight_buffers[step_idx];
    let conv_weight = weight_to_dyn(
        weights,
        "conv_weight",
        &[out_channels, in_channels, 1],
        dtype,
        step_idx,
        "FusedAddInstanceNormConv1x1",
    )?;

    let output = if has_bias {
        let conv_bias = weight_to_dyn(
            weights,
            "conv_bias",
            &[out_channels],
            dtype,
            step_idx,
            "FusedAddInstanceNormConv1x1",
        )?;
        let y = normed.conv1d(&conv_weight, 0, 1, 1, 1).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedAddInstanceNormConv1x1 conv1d: {e}"))
        })?;
        let bias_r = conv_bias.reshape([1, out_channels, 1]).map_err(|e| {
            native_dispatch_err(
                step_idx,
                format!("FusedAddInstanceNormConv1x1 bias reshape: {e}"),
            )
        })?;
        y.broadcast_add(&bias_r).map_err(|e| {
            native_dispatch_err(
                step_idx,
                format!("FusedAddInstanceNormConv1x1 bias add: {e}"),
            )
        })?
    } else {
        normed.conv1d(&conv_weight, 0, 1, 1, 1).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedAddInstanceNormConv1x1 conv1d: {e}"))
        })?
    };

    dyn_to_slice(&output, step_idx, "FusedAddInstanceNormConv1x1")
}
