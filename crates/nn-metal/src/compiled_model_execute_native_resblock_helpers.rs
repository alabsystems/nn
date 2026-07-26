// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Weight loading and style projection helpers for FusedResBlock executor.
//!
//! Extracted from `compiled_model_execute_native_resblock.rs` for 450-line
//! compliance. These are shared by the main executor and the fallback module.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::{Device, Result};

use super::CompiledModel;
use super::PhaseWeightKeys;
use super::{native_dispatch_err, weight_to_dyn};

use nn_dsl::NormActivConv1dParams;

/// Linear projection: style_embed → (gamma `[B,C,1]`, beta `[B,C,1]`).
///
/// Uses pre-transposed weight (`{weight_key}_t`) when available to skip
/// the GPU transpose dispatch that `Linear::new()` would otherwise perform
/// on every forward pass. Falls back to `Linear::new()` for models compiled
/// before the pre-transpose optimization.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_style_projection(
    model: &CompiledModel,
    step_idx: usize,
    style_embed: &DynTensor,
    channels: usize,
    style_dim: usize,
    batch: usize,
    weight_key: &str,
    bias_key: &str,
) -> Result<(DynTensor, DynTensor)> {
    let dtype = model.step_dtype(step_idx);

    let rb_weights = &model.def.weight_buffers[step_idx];
    let bias = weight_to_dyn(
        rb_weights,
        bias_key,
        &[2 * channels],
        dtype,
        step_idx,
        "FusedResBlock style proj",
    )?;

    // Fast path: use pre-transposed weight to skip GPU transpose dispatch.
    let weight_t_key = format!("{weight_key}_t");
    let projected = if rb_weights.contains_key(weight_t_key.as_str()) {
        // Pre-transposed weight: [style_dim, 2*channels].
        let weight_t = weight_to_dyn(
            rb_weights,
            &weight_t_key,
            &[style_dim, 2 * channels],
            dtype,
            step_idx,
            "FusedResBlock style proj",
        )?;
        // matmul: [B, style_dim] @ [style_dim, 2*channels] = [B, 2*channels]
        let y = style_embed.matmul(&weight_t).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedResBlock style proj matmul: {e}"))
        })?;
        y.broadcast_add(&bias).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedResBlock style proj bias add: {e}"))
        })?
    } else {
        // Fallback: create Linear which transposes weight on GPU.
        let weight = weight_to_dyn(
            rb_weights,
            weight_key,
            &[2 * channels, style_dim],
            dtype,
            step_idx,
            "FusedResBlock style proj",
        )?;
        let linear = Linear::new(weight, Some(bias)).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedResBlock style proj Linear: {e}"))
        })?;
        linear.forward(style_embed).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedResBlock style proj forward: {e}"))
        })?
    };

    // Split: narrow along dim 1 into gamma [B, channels] and beta [B, channels].
    let gamma_2d = projected.narrow(1, 0, channels).map_err(|e| {
        native_dispatch_err(
            step_idx,
            format!("FusedResBlock style proj narrow gamma: {e}"),
        )
    })?;
    let beta_2d = projected.narrow(1, channels, channels).map_err(|e| {
        native_dispatch_err(
            step_idx,
            format!("FusedResBlock style proj narrow beta: {e}"),
        )
    })?;

    // Reshape to [B, channels, 1] for AdaIN compatibility.
    let gamma = gamma_2d.reshape([batch, channels, 1]).map_err(|e| {
        native_dispatch_err(
            step_idx,
            format!("FusedResBlock style proj reshape gamma: {e}"),
        )
    })?;
    let beta = beta_2d.reshape([batch, channels, 1]).map_err(|e| {
        native_dispatch_err(
            step_idx,
            format!("FusedResBlock style proj reshape beta: {e}"),
        )
    })?;

    Ok((gamma, beta))
}

/// Load a conv weight tensor `[C_out, C_in, K]` from pre-uploaded buffers.
pub(super) fn load_conv_weight(
    model: &CompiledModel,
    step_idx: usize,
    keys: &PhaseWeightKeys,
    in_channels: usize,
    params: &NormActivConv1dParams,
) -> Result<DynTensor> {
    let dtype = model.step_dtype(step_idx);
    let weights = &model.def.weight_buffers[step_idx];
    weight_to_dyn(
        weights,
        keys.conv_weight,
        &[params.output_channels, in_channels, params.kernel_size],
        dtype,
        step_idx,
        "FusedResBlock",
    )
}

/// Load a conv bias tensor `[C_out]` from pre-uploaded buffers, or create zeros.
pub(super) fn load_conv_bias(
    model: &CompiledModel,
    step_idx: usize,
    keys: &PhaseWeightKeys,
    output_channels: usize,
) -> Result<DynTensor> {
    let dtype = model.step_dtype(step_idx);
    let weights = &model.def.weight_buffers[step_idx];
    if weights.contains_key(keys.conv_bias) {
        weight_to_dyn(
            weights,
            keys.conv_bias,
            &[output_channels],
            dtype,
            step_idx,
            "FusedResBlock",
        )
    } else {
        DynTensor::zeros(&[output_channels], dtype, &Device::metal())
    }
}

/// Load a Snake alpha tensor `[C_in]` from pre-uploaded buffers.
pub(super) fn load_alpha(
    model: &CompiledModel,
    step_idx: usize,
    keys: &PhaseWeightKeys,
    channels: usize,
) -> Result<DynTensor> {
    let dtype = model.step_dtype(step_idx);
    let weights = &model.def.weight_buffers[step_idx];
    weight_to_dyn(
        weights,
        keys.alpha,
        &[channels],
        dtype,
        step_idx,
        "FusedResBlock",
    )
}
