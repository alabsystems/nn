// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! FusedResBlock executor: sequences 2× NormActivConv1d + residual add
//! with pre-resolved buffers.
//!
//! Part of #2218: Full ResBlock Mega-Kernel.

use nn_core::Result;
use nn_dsl::{NormActivConv1dParams, NormActivation, StyleBatchOffset, StyleProjectionParams};

use crate::cache::PipelineCache;
use crate::gpu_slice::GpuSlice;

use super::CompiledModel;
use super::{dyn_to_slice, native_dispatch_err, slice_to_dyn, weight_to_dyn};

// Mixed-activation fallback helpers (separate file for 450-line compliance).
#[path = "compiled_model_execute_native_resblock_fallback.rs"]
mod fallback;

// Weight loading and style projection helpers (extracted for 450-line compliance).
#[path = "compiled_model_execute_native_resblock_helpers.rs"]
mod rb_helpers;

// NativeEncoding plan for 3-dispatch sequence (#3472 D3 S3).
// Gated: encoding module not yet wired (#3472 in progress).
#[cfg(feature = "_native_encoding")]
#[path = "compiled_model_execute_native_resblock_plan.rs"]
mod plan;

/// Pre-computed weight lookup keys for a FusedResBlock phase.
///
/// Eliminates `format!()` allocations on every forward pass (#2935).
/// Keys match the names set during peephole fusion in
/// `trace_compile_peephole_resblock.rs`.
pub(super) struct PhaseWeightKeys {
    /// Label for error messages (e.g. "p1", "p2").
    pub(super) label: &'static str,
    /// Key for Snake alpha weight (only used when activation is Snake).
    pub(super) alpha: &'static str,
    /// Key for conv1d weight tensor.
    pub(super) conv_weight: &'static str,
    /// Key for conv1d bias tensor.
    pub(super) conv_bias: &'static str,
}

const PHASE1_KEYS: PhaseWeightKeys = PhaseWeightKeys {
    label: "p1_",
    alpha: "p1_alpha",
    conv_weight: "p1_conv_weight",
    conv_bias: "p1_conv_bias",
};

const PHASE2_KEYS: PhaseWeightKeys = PhaseWeightKeys {
    label: "p2_",
    alpha: "p2_alpha",
    conv_weight: "p2_conv_weight",
    conv_bias: "p2_conv_bias",
};

/// Execute a `NativeOpKind::FusedResBlock` step (#2218).
///
/// Sequences 2× NormActivConv1d + residual add + optional scale using
/// direct buffer access via `input_steps` (bypassing the edge_map).
///
/// **Without style_proj** (`input_steps = [x, γ1, β1, γ2, β2]`):
/// Phase 1: InstanceNorm + affine(gamma1, beta1) + activation + Conv1d
/// Phase 2: InstanceNorm + affine(gamma2, beta2) + activation + Conv1d
/// Phase 3: Residual add(x, phase2_output) * residual_scale
///
/// **With style_proj** (`input_steps = [x, style_embed]`):
/// Phase 0: Linear projections to produce gamma/beta from style embedding
/// Phase 1-3: Same as above using projected gamma/beta
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_native_fused_resblock(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    phase1: &NormActivConv1dParams,
    phase2: &NormActivConv1dParams,
    input_steps: &[usize],
    residual_scale: f32,
    style_proj: Option<&StyleProjectionParams>,
    shortcut_step: Option<usize>,
    pool_step: Option<usize>,
    style_batch_offset: Option<&StyleBatchOffset>,
    _cache: &PipelineCache,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);

    // Resolve inputs directly from buffers via input_steps.
    let resolve_step = |idx: usize| -> Result<GpuSlice> {
        let step = input_steps[idx];
        buffers[step].as_ref().map(GpuSlice::alias).ok_or_else(|| {
            native_dispatch_err(
                step_idx,
                format!("FusedResBlock: input_steps[{idx}] = {step} has no buffer"),
            )
        })
    };

    let batch = phase1.input_shape[0];
    let channels1 = phase1.input_shape[1];
    let channels2 = phase2.input_shape[1];

    // Resolve x (always input_steps[0]).
    let x_slice = resolve_step(0)?;
    let x_tensor = slice_to_dyn(&x_slice, &phase1.input_shape, dtype)?;

    // Resolve residual tensor for the final add.
    // Identity shortcut: residual = x. Conv1x1 shortcut: residual = buffers[shortcut_step].
    let residual_tensor = if let Some(sc_step) = shortcut_step {
        let sc_slice = buffers[sc_step]
            .as_ref()
            .map(GpuSlice::alias)
            .ok_or_else(|| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedResBlock: shortcut_step {sc_step} has no buffer"),
                )
            })?;
        // Conv1x1 output shape matches phase2 output: [B, C_out, T].
        let sc_shape = vec![batch, phase2.output_channels, phase1.input_shape[2]];
        slice_to_dyn(&sc_slice, &sc_shape, dtype)?
    } else {
        x_tensor.clone()
    };

    // Resolve gamma/beta pairs — 3 paths:
    // 1. style_batch_offset: narrow from pre-computed batched output (0 dispatches)
    // 2. style_proj: per-block Linear projections (4 dispatches)
    // 3. direct buffers: [x, γ1, β1, γ2, β2] (0 dispatches)
    let (gamma1, beta1, gamma2, beta2) = if let Some(sbo) = style_batch_offset {
        // Batched narrow path: input_steps = [x, batch_step].
        if input_steps.len() < 2 {
            return Err(native_dispatch_err(
                step_idx,
                format!(
                    "FusedResBlock(batch_offset): expected >=2 input_steps, got {}",
                    input_steps.len()
                ),
            ));
        }
        let batch_slice = resolve_step(1)?;
        // Use the full output dimension (total_out) from the buffer, not this
        // block's end offset. Row strides must match the actual buffer layout
        // for batch > 1. Infer total_out from buffer byte length.
        let slice_bytes = batch_slice.buffer().len() - batch_slice.byte_offset();
        let total_out_dim = slice_bytes / (batch * dtype.size_bytes());
        let batch_tensor = slice_to_dyn(&batch_slice, &[batch, total_out_dim], dtype)?;

        // Layout: [gamma1(C1), beta1(C1), gamma2(C2), beta2(C2)] at sbo.offset.
        let mut off = sbo.offset;
        let g1_2d = batch_tensor
            .narrow(1, off, sbo.channels1)
            .map_err(|e| native_dispatch_err(step_idx, format!("batch narrow g1: {e}")))?;
        off += sbo.channels1;
        let b1_2d = batch_tensor
            .narrow(1, off, sbo.channels1)
            .map_err(|e| native_dispatch_err(step_idx, format!("batch narrow b1: {e}")))?;
        off += sbo.channels1;
        let g2_2d = batch_tensor
            .narrow(1, off, sbo.channels2)
            .map_err(|e| native_dispatch_err(step_idx, format!("batch narrow g2: {e}")))?;
        off += sbo.channels2;
        let b2_2d = batch_tensor
            .narrow(1, off, sbo.channels2)
            .map_err(|e| native_dispatch_err(step_idx, format!("batch narrow b2: {e}")))?;

        // Reshape to [B, C, 1] for AdaIN compatibility (zero-copy).
        let g1 = g1_2d
            .reshape([batch, sbo.channels1, 1])
            .map_err(|e| native_dispatch_err(step_idx, format!("batch reshape g1: {e}")))?;
        let b1 = b1_2d
            .reshape([batch, sbo.channels1, 1])
            .map_err(|e| native_dispatch_err(step_idx, format!("batch reshape b1: {e}")))?;
        let g2 = g2_2d
            .reshape([batch, sbo.channels2, 1])
            .map_err(|e| native_dispatch_err(step_idx, format!("batch reshape g2: {e}")))?;
        let b2 = b2_2d
            .reshape([batch, sbo.channels2, 1])
            .map_err(|e| native_dispatch_err(step_idx, format!("batch reshape b2: {e}")))?;
        (g1, b1, g2, b2)
    } else if let Some(sp) = style_proj {
        // Style projection path: input_steps = [x, style_embed].
        if input_steps.len() < 2 {
            return Err(native_dispatch_err(
                step_idx,
                format!(
                    "FusedResBlock(style_proj): expected >=2 input_steps, got {}",
                    input_steps.len()
                ),
            ));
        }
        let style_slice = resolve_step(1)?;
        let style_tensor = slice_to_dyn(&style_slice, &[batch, sp.style_dim], dtype)?;

        let (g1, b1) = rb_helpers::run_style_projection(
            model,
            step_idx,
            &style_tensor,
            channels1,
            sp.style_dim,
            batch,
            "style1_weight",
            "style1_bias",
        )?;
        let (g2, b2) = rb_helpers::run_style_projection(
            model,
            step_idx,
            &style_tensor,
            channels2,
            sp.style_dim,
            batch,
            "style2_weight",
            "style2_bias",
        )?;
        (g1, b1, g2, b2)
    } else {
        // Direct buffer path: input_steps = [x, γ1, β1, γ2, β2].
        if input_steps.len() < 5 {
            return Err(native_dispatch_err(
                step_idx,
                format!(
                    "FusedResBlock: expected 5 input_steps, got {}",
                    input_steps.len()
                ),
            ));
        }
        let gamma1_shape = vec![batch, channels1, 1];
        let gamma2_shape = vec![batch, channels2, 1];

        let g1 = slice_to_dyn(&resolve_step(1)?, &gamma1_shape, dtype)?;
        let b1 = slice_to_dyn(&resolve_step(2)?, &gamma1_shape, dtype)?;
        let g2 = slice_to_dyn(&resolve_step(3)?, &gamma2_shape, dtype)?;
        let b2 = slice_to_dyn(&resolve_step(4)?, &gamma2_shape, dtype)?;
        (g1, b1, g2, b2)
    };

    // --- Upsample pool path (#3510) ---
    // When pool_step is set, phase1's norm+activation already ran as a standalone
    // AdainLeakyRelu/AdainSnake step, and the pool (ConvTranspose1d) already ran.
    // We read the pool output from buffers[pool_step], run Conv1d for phase1,
    // then run phase2 normally via the fallback path.
    if let Some(ps) = pool_step {
        let pool_slice = buffers[ps]
            .as_ref()
            .map(GpuSlice::alias)
            .ok_or_else(|| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedResBlock: pool_step {ps} has no buffer"),
                )
            })?;
        // Pool output shape: [B, C_in, T_pooled] — same channels as phase1 input,
        // but spatial dimension changed by the pool. Infer from buffer size.
        let pool_channels = phase1.input_shape[1];
        let pool_bytes = pool_slice.buffer().len() - pool_slice.byte_offset();
        let pool_time = pool_bytes / (batch * pool_channels * dtype.size_bytes());
        let pool_shape = vec![batch, pool_channels, pool_time];
        let pool_tensor = slice_to_dyn(&pool_slice, &pool_shape, dtype)?;

        // Phase 1: just Conv1d on pool output (norm+activation already done).
        let phase1_output = fallback::run_conv1d(
            model,
            step_idx,
            &pool_tensor,
            pool_channels,
            phase1.output_channels,
            phase1.kernel_size,
            phase1.conv_padding,
            phase1.conv_dilation,
            &PHASE1_KEYS,
        )?;

        // Phase 2: full NormActivConv1d on phase1_output.
        let phase2_activated = fallback::run_norm_activ(
            model,
            step_idx,
            &phase1_output,
            &gamma2,
            &beta2,
            &phase2.activation,
            phase2.eps,
            channels2,
            &PHASE2_KEYS,
        )?;

        let phase2_output = fallback::run_conv1d(
            model,
            step_idx,
            &phase2_activated,
            channels2,
            phase2.output_channels,
            phase2.kernel_size,
            phase2.conv_padding,
            phase2.conv_dilation,
            &PHASE2_KEYS,
        )?;

        let sum = residual_tensor.add(&phase2_output).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedResBlock(pool) residual add: {e}"))
        })?;

        let output = if (residual_scale - 1.0).abs() > f32::EPSILON {
            sum.mul_scalar(f64::from(residual_scale)).map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedResBlock(pool) residual scale: {e}"),
                )
            })?
        } else {
            sum
        };

        return dyn_to_slice(&output, step_idx, "FusedResBlock");
    }

    // --- NativeEncoding fast path (#3472 D3 S3) ---
    // Gated: encoding module not yet wired (#3472 in progress).
    // When enabled, eliminates the DynTensor bridge for LeakyRelu and Snake fast paths.

    // --- LeakyRelu fast path: 3-dispatch fused stats+norm+conv (#1815 Tier 2) ---
    // Phase 1: stats(x) + conv_with_output_stats(x) → (p1_output, p2_stats).
    // Phase 2: conv_with_precomputed_stats(p1_output, p2_stats) → output.
    // Saves 1 dispatch per FusedResBlock vs the 4-dispatch path.
    if matches!(phase1.activation, NormActivation::LeakyRelu { .. })
        && matches!(phase2.activation, NormActivation::LeakyRelu { .. })
    {
        let slope1 = match phase1.activation {
            NormActivation::LeakyRelu { slope } => slope,
            _ => {
                return Err(native_dispatch_err(
                    step_idx,
                    "phase1 activation is not LeakyRelu".into(),
                ))
            }
        };
        let slope2 = match phase2.activation {
            NormActivation::LeakyRelu { slope } => slope,
            _ => {
                return Err(native_dispatch_err(
                    step_idx,
                    "phase2 activation is not LeakyRelu".into(),
                ))
            }
        };

        let p1_conv_w =
            rb_helpers::load_conv_weight(model, step_idx, &PHASE1_KEYS, channels1, phase1)?;
        let p1_conv_b =
            rb_helpers::load_conv_bias(model, step_idx, &PHASE1_KEYS, phase1.output_channels)?;

        // Phase 1: conv + output stats epilogue (2 dispatches).
        let (phase1_output, precomputed_stats) =
            crate::dyn_tensor_metal::native_norm_activ_conv1d_with_output_stats(
                &x_tensor,
                &gamma1,
                &beta1,
                &p1_conv_w,
                &p1_conv_b,
                f64::from(phase1.eps),
                f64::from(slope1),
                phase1.conv_padding,
                phase1.conv_dilation,
                None,
                phase2.eps,
            )
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("FusedResBlock p1 conv_with_stats: {e}"))
            })?;

        let p2_conv_w =
            rb_helpers::load_conv_weight(model, step_idx, &PHASE2_KEYS, channels2, phase2)?;
        let p2_conv_b =
            rb_helpers::load_conv_bias(model, step_idx, &PHASE2_KEYS, phase2.output_channels)?;

        // Phase 2: conv with precomputed stats (1 dispatch, skips stats kernel).
        let residual_params = crate::dyn_tensor_metal::ResidualParams {
            residual: &residual_tensor,
            scale: residual_scale,
        };
        let output = crate::dyn_tensor_metal::native_norm_activ_conv1d_with_precomputed_stats(
            &phase1_output,
            &gamma2,
            &beta2,
            &p2_conv_w,
            &p2_conv_b,
            f64::from(slope2),
            phase2.conv_padding,
            phase2.conv_dilation,
            Some(residual_params),
            &precomputed_stats,
        )
        .map_err(|e| {
            native_dispatch_err(
                step_idx,
                format!("FusedResBlock p2 conv_precomputed_stats: {e}"),
            )
        })?;

        return dyn_to_slice(&output, step_idx, "FusedResBlock");
    }

    // --- Snake fast path: 3-dispatch fused (#1815 Tier 2) ---
    // Same architecture as LeakyRelu but with Snake activation + alpha device buffer.
    if matches!(phase1.activation, NormActivation::Snake)
        && matches!(phase2.activation, NormActivation::Snake)
    {
        let p1_alpha = rb_helpers::load_alpha(model, step_idx, &PHASE1_KEYS, channels1)?;
        let p1_conv_w =
            rb_helpers::load_conv_weight(model, step_idx, &PHASE1_KEYS, channels1, phase1)?;
        let p1_conv_b =
            rb_helpers::load_conv_bias(model, step_idx, &PHASE1_KEYS, phase1.output_channels)?;

        // Phase 1: conv + output stats epilogue (2 dispatches).
        let (phase1_output, precomputed_stats) =
            crate::dyn_tensor_metal::native_norm_activ_conv1d_snake_with_output_stats(
                &x_tensor,
                &gamma1,
                &beta1,
                &p1_alpha,
                &p1_conv_w,
                &p1_conv_b,
                f64::from(phase1.eps),
                phase1.conv_padding,
                phase1.conv_dilation,
                None,
                phase2.eps,
            )
            .map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedResBlock p1 Snake conv_with_stats: {e}"),
                )
            })?;

        let p2_alpha = rb_helpers::load_alpha(model, step_idx, &PHASE2_KEYS, channels2)?;
        let p2_conv_w =
            rb_helpers::load_conv_weight(model, step_idx, &PHASE2_KEYS, channels2, phase2)?;
        let p2_conv_b =
            rb_helpers::load_conv_bias(model, step_idx, &PHASE2_KEYS, phase2.output_channels)?;

        // Phase 2: conv with precomputed stats (1 dispatch).
        let residual_params = crate::dyn_tensor_metal::ResidualParams {
            residual: &residual_tensor,
            scale: residual_scale,
        };
        let output =
            crate::dyn_tensor_metal::native_norm_activ_conv1d_snake_with_precomputed_stats(
                &phase1_output,
                &gamma2,
                &beta2,
                &p2_alpha,
                &p2_conv_w,
                &p2_conv_b,
                phase2.conv_padding,
                phase2.conv_dilation,
                Some(residual_params),
                &precomputed_stats,
            )
            .map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedResBlock p2 Snake conv_precomputed: {e}"),
                )
            })?;

        return dyn_to_slice(&output, step_idx, "FusedResBlock");
    }

    // --- Mixed activation fallback: separate norm + conv dispatches ---
    // Reached only when phase1 and phase2 use different activation types.
    let phase1_activated = fallback::run_norm_activ(
        model,
        step_idx,
        &x_tensor,
        &gamma1,
        &beta1,
        &phase1.activation,
        phase1.eps,
        channels1,
        &PHASE1_KEYS,
    )?;

    let phase1_output = fallback::run_conv1d(
        model,
        step_idx,
        &phase1_activated,
        channels1,
        phase1.output_channels,
        phase1.kernel_size,
        phase1.conv_padding,
        phase1.conv_dilation,
        &PHASE1_KEYS,
    )?;

    let phase2_activated = fallback::run_norm_activ(
        model,
        step_idx,
        &phase1_output,
        &gamma2,
        &beta2,
        &phase2.activation,
        phase2.eps,
        channels2,
        &PHASE2_KEYS,
    )?;

    let phase2_output = fallback::run_conv1d(
        model,
        step_idx,
        &phase2_activated,
        channels2,
        phase2.output_channels,
        phase2.kernel_size,
        phase2.conv_padding,
        phase2.conv_dilation,
        &PHASE2_KEYS,
    )?;

    let sum = residual_tensor
        .add(&phase2_output)
        .map_err(|e| native_dispatch_err(step_idx, format!("FusedResBlock residual add: {e}")))?;

    let output = if (residual_scale - 1.0).abs() > f32::EPSILON {
        sum.mul_scalar(f64::from(residual_scale)).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedResBlock residual scale: {e}"))
        })?
    } else {
        sum
    };

    dyn_to_slice(&output, step_idx, "FusedResBlock")
}
