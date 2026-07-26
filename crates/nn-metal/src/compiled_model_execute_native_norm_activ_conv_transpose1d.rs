// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Execute `NativeOpKind::NormActivConvTranspose1d` step.
//!
//! Fuses InstanceNorm + style affine + activation + ConvTranspose1d into
//! a single logical dispatch. The transposed-conv dual of NormActivConv1d.
//!
//! Sequences: AdainLeakyRelu/AdainSnake(x, gamma, beta) then
//! ConvTranspose1d(result, weight, bias). The lazy GPU command buffer
//! batches them into minimal Metal dispatches.
//!
//! Part of #4264.

use nn_core::Result;
use nn_dsl::NormActivation;

use crate::cache::PipelineCache;
use crate::gpu_slice::GpuSlice;

use super::CompiledModel;
use super::{dyn_to_slice, native_dispatch_err, slice_to_dyn, weight_to_dyn};

/// Execute a `NativeOpKind::NormActivConvTranspose1d` step.
///
/// Resolves 3 graph inputs (x, gamma, beta) and conv weights (conv_weight,
/// conv_bias, alpha), performs AdainLeakyRelu/AdainSnake then ConvTranspose1d.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_native_norm_activ_conv_transpose1d(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    activation: &NormActivation,
    eps: f32,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    output_padding: usize,
    output_channels: usize,
    input_shape: &[usize],
    _cache: &PipelineCache,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);
    let batch = input_shape[0];
    let channels = input_shape[1];

    // Resolve graph inputs: x (0), gamma (1), beta (2).
    let x_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let gamma_slice = model.resolve_input_slice(step_idx, 1, buffers)?;
    let beta_slice = model.resolve_input_slice(step_idx, 2, buffers)?;

    let gamma_shape = [batch, channels, 1];
    let x_tensor = slice_to_dyn(&x_slice, input_shape, dtype)?;
    let gamma_tensor = slice_to_dyn(&gamma_slice, &gamma_shape, dtype)?;
    let beta_tensor = slice_to_dyn(&beta_slice, &gamma_shape, dtype)?;

    // Phase 1: AdainLeakyRelu or AdainSnake.
    let activated = match activation {
        NormActivation::LeakyRelu { slope } => {
            crate::dyn_tensor_metal::native_adain_leaky_relu(
                &x_tensor,
                &gamma_tensor,
                &beta_tensor,
                f64::from(eps),
                f64::from(*slope),
            )
            .map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("NormActivConvTranspose1d adain_leaky_relu: {e}"),
                )
            })?
        }
        NormActivation::Snake => {
            let weights = &model.def.weight_buffers[step_idx];
            let alpha_tensor = weight_to_dyn(
                weights,
                "alpha",
                &[channels],
                dtype,
                step_idx,
                "NormActivConvTranspose1d",
            )?;
            crate::dyn_tensor_metal::native_adain_snake(
                &x_tensor,
                &gamma_tensor,
                &beta_tensor,
                &alpha_tensor,
                f64::from(eps),
                false, // standard gamma*normed+beta
            )
            .map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("NormActivConvTranspose1d adain_snake: {e}"),
                )
            })?
        }
        other => {
            return Err(native_dispatch_err(
                step_idx,
                format!("NormActivConvTranspose1d: unsupported activation {other:?}"),
            ));
        }
    };

    // Phase 2: ConvTranspose1d.
    let weights = &model.def.weight_buffers[step_idx];
    // ConvTranspose1d weight shape: [C_in, C_out/groups, K]
    let in_channels = channels;
    let weight_shape = [in_channels, output_channels / groups, kernel_size];
    let conv_weight = weight_to_dyn(
        weights,
        "conv_weight",
        &weight_shape,
        dtype,
        step_idx,
        "NormActivConvTranspose1d",
    )?;

    let conv_output = activated
        .conv_transpose1d(&conv_weight, padding, output_padding, stride, dilation, groups)
        .map_err(|e| {
            native_dispatch_err(
                step_idx,
                format!("NormActivConvTranspose1d conv_transpose1d: {e}"),
            )
        })?;

    // Bias add (if present).
    let output = if weights.contains_key("conv_bias") {
        let bias_tensor = weight_to_dyn(
            weights,
            "conv_bias",
            &[output_channels],
            dtype,
            step_idx,
            "NormActivConvTranspose1d",
        )?;
        let bias_reshaped = bias_tensor.reshape([1, output_channels, 1]).map_err(|e| {
            native_dispatch_err(
                step_idx,
                format!("NormActivConvTranspose1d bias reshape: {e}"),
            )
        })?;
        conv_output.broadcast_add(&bias_reshaped).map_err(|e| {
            native_dispatch_err(
                step_idx,
                format!("NormActivConvTranspose1d bias add: {e}"),
            )
        })?
    } else {
        conv_output
    };

    dyn_to_slice(&output, step_idx, "NormActivConvTranspose1d")
}
