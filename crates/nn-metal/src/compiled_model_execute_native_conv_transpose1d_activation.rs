// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Execute `NativeOpKind::FusedConvTranspose1dActivation` step.
//!
//! Fuses ConvTranspose1d + Activation (Snake, ReLU, LeakyReLU, SiLU, GELU,
//! GeluErf, Tanh) into a single logical dispatch. The conv_transpose1d and
//! activation are dispatched as DynTensor GPU ops within one NativeOp
//! execution context, enabling the lazy GPU command buffer to batch them.
//!
//! Part of #4264.

use nn_core::Result;
use nn_dsl::ConvActivation;

use crate::cache::PipelineCache;
use crate::gpu_slice::GpuSlice;

use super::CompiledModel;
use super::{dyn_to_slice, native_dispatch_err, slice_to_dyn, weight_to_dyn};

/// Execute a `NativeOpKind::FusedConvTranspose1dActivation` step.
///
/// Resolves 1 graph input (x) and conv weights (weight, bias, alpha),
/// performs ConvTranspose1d then applies the activation function.
///
/// Saves 1 dispatch per pair by merging the two-step pattern into a
/// single NativeOp that the lazy command buffer can batch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_native_conv_transpose1d_activation(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    activation: &ConvActivation,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    output_padding: usize,
    has_bias: bool,
    input_shape: &[usize],
    _cache: &PipelineCache,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);

    // Resolve graph input: x (0) with shape [B, C_in, T].
    let x_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let x_tensor = slice_to_dyn(&x_slice, input_shape, dtype)?;

    // Load conv weights.
    // ConvTranspose1d weight shape: [C_in, C_out/groups, K].
    let weights = &model.def.weight_buffers[step_idx];
    let in_channels = input_shape.get(1).copied().unwrap_or(0);
    let weight_shape = [in_channels, out_channels / groups, kernel_size];
    let weight_tensor = weight_to_dyn(
        weights,
        "weight",
        &weight_shape,
        dtype,
        step_idx,
        "FusedConvTranspose1dActivation",
    )?;

    // ConvTranspose1d.
    let conv_output = x_tensor
        .conv_transpose1d(&weight_tensor, padding, output_padding, stride, dilation, groups)
        .map_err(|e| {
            native_dispatch_err(
                step_idx,
                format!("FusedConvTranspose1dActivation conv_transpose1d: {e}"),
            )
        })?;

    // Bias add (if present).
    let conv_output = if has_bias {
        let bias_tensor = weight_to_dyn(
            weights,
            "bias",
            &[out_channels],
            dtype,
            step_idx,
            "FusedConvTranspose1dActivation",
        )?;
        // Bias broadcast: [C_out] -> [1, C_out, 1] for [B, C_out, T] tensor.
        let bias_reshaped = bias_tensor.reshape([1, out_channels, 1]).map_err(|e| {
            native_dispatch_err(
                step_idx,
                format!("FusedConvTranspose1dActivation bias reshape: {e}"),
            )
        })?;
        conv_output.broadcast_add(&bias_reshaped).map_err(|e| {
            native_dispatch_err(
                step_idx,
                format!("FusedConvTranspose1dActivation bias add: {e}"),
            )
        })?
    } else {
        conv_output
    };

    // Apply activation.
    let output = match activation {
        ConvActivation::Relu => conv_output.relu().map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConvTranspose1dActivation relu: {e}"))
        })?,
        ConvActivation::LeakyRelu { slope } => {
            conv_output.leaky_relu(f64::from(*slope)).map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedConvTranspose1dActivation leaky_relu: {e}"),
                )
            })?
        }
        ConvActivation::Silu => conv_output.silu().map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConvTranspose1dActivation silu: {e}"))
        })?,
        ConvActivation::Snake => {
            // Snake activation: x + (1/alpha) * sin(alpha * x)^2
            // Alpha weight [C_out] from weight_data.
            let alpha_tensor = weight_to_dyn(
                weights,
                "alpha",
                &[out_channels],
                dtype,
                step_idx,
                "FusedConvTranspose1dActivation",
            )?;
            // Reshape alpha: [C_out] -> [1, C_out, 1] for [B, C_out, T] broadcast.
            let alpha_3d = alpha_tensor.reshape([1, out_channels, 1]).map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedConvTranspose1dActivation alpha reshape: {e}"),
                )
            })?;
            // snake(x, alpha) = x + (1/alpha) * sin(alpha * x)^2
            let ax = conv_output.broadcast_mul(&alpha_3d).map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedConvTranspose1dActivation alpha*x: {e}"),
                )
            })?;
            let sin_ax = ax.sin().map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedConvTranspose1dActivation sin: {e}"),
                )
            })?;
            let sin2 = sin_ax.mul(&sin_ax).map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedConvTranspose1dActivation sin^2: {e}"),
                )
            })?;
            let inv_alpha = alpha_3d.recip().map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedConvTranspose1dActivation 1/alpha: {e}"),
                )
            })?;
            let scaled = sin2.broadcast_mul(&inv_alpha).map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedConvTranspose1dActivation sin^2/alpha: {e}"),
                )
            })?;
            conv_output.add(&scaled).map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedConvTranspose1dActivation add: {e}"),
                )
            })?
        }
        ConvActivation::Gelu => conv_output.gelu().map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConvTranspose1dActivation gelu: {e}"))
        })?,
        ConvActivation::GeluErf => conv_output.gelu_erf().map_err(|e| {
            native_dispatch_err(
                step_idx,
                format!("FusedConvTranspose1dActivation gelu_erf: {e}"),
            )
        })?,
        ConvActivation::Tanh => conv_output.tanh().map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConvTranspose1dActivation tanh: {e}"))
        })?,
        _ => {
            return Err(native_dispatch_err(
                step_idx,
                format!(
                    "FusedConvTranspose1dActivation: unsupported activation variant {activation:?}"
                ),
            ));
        }
    };

    dyn_to_slice(&output, step_idx, "FusedConvTranspose1dActivation")
}
