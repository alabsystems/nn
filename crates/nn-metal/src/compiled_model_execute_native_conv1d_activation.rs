// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Execute `NativeOpKind::FusedConv1dActivation` step.
//!
//! Fuses Conv1d + Activation (Snake, ReLU, LeakyReLU, SiLU, GELU, GeluErf, Tanh) into a
//! single logical dispatch. The conv1d and activation are dispatched
//! as DynTensor GPU ops within one NativeOp execution context,
//! enabling the lazy GPU command buffer to batch them.
//!
//! When `pre_activation` is true, activation is applied BEFORE conv1d
//! (Activation -> Conv1d pattern). When false, activation is applied AFTER
//! conv1d (Conv1d -> Activation pattern).
//!
//! Part of #4264.

use nn_core::Result;
use nn_dsl::ConvActivation;

use crate::cache::PipelineCache;
use crate::gpu_slice::GpuSlice;

use super::CompiledModel;
use super::{dyn_to_slice, native_dispatch_err, slice_to_dyn, weight_to_dyn};

/// Execute a `NativeOpKind::FusedConv1dActivation` step.
///
/// Resolves 1 graph input (x) and conv weights (weight, bias, alpha),
/// then either:
/// - `pre_activation == false`: Conv1d(x) then Activation (default)
/// - `pre_activation == true`: Activation(x) then Conv1d
///
/// Saves 1 dispatch per pair by merging the two-step pattern into a
/// single NativeOp that the lazy command buffer can batch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_native_conv1d_activation(
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
    has_bias: bool,
    input_shape: &[usize],
    pre_activation: bool,
    _cache: &PipelineCache,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);

    // Resolve graph input: x (0) with shape [B, C_in, T].
    let x_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let x_tensor = slice_to_dyn(&x_slice, input_shape, dtype)?;

    // Load conv weights.
    let weights = &model.def.weight_buffers[step_idx];
    let in_channels = if groups > 0 { input_shape[1] / groups * groups } else { input_shape[1] };
    let weight_shape = [out_channels, in_channels / groups, kernel_size];
    let weight_tensor = weight_to_dyn(
        weights,
        "weight",
        &weight_shape,
        dtype,
        step_idx,
        "FusedConv1dActivation",
    )?;

    // When pre_activation is true: apply activation first, then conv1d.
    // When false (default): apply conv1d first, then activation.
    let conv_input = if pre_activation {
        apply_activation(
            &x_tensor,
            activation,
            weights,
            input_shape[1], // in_channels for alpha
            dtype,
            step_idx,
        )?
    } else {
        x_tensor
    };

    // Conv1d.
    let conv_output = if has_bias {
        let bias_tensor = weight_to_dyn(
            weights,
            "bias",
            &[out_channels],
            dtype,
            step_idx,
            "FusedConv1dActivation",
        )?;
        let y = conv_input
            .conv1d(&weight_tensor, padding, stride, dilation, groups)
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("FusedConv1dActivation conv1d: {e}"))
            })?;
        // Bias broadcast: [C_out] -> [1, C_out, 1] for [B, C_out, T] tensor.
        let bias_reshaped = bias_tensor.reshape([1, out_channels, 1]).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dActivation bias reshape: {e}"))
        })?;
        y.broadcast_add(&bias_reshaped).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dActivation bias add: {e}"))
        })?
    } else {
        conv_input
            .conv1d(&weight_tensor, padding, stride, dilation, groups)
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("FusedConv1dActivation conv1d: {e}"))
            })?
    };

    // When pre_activation is false (default): apply activation after conv1d.
    let output = if pre_activation {
        conv_output
    } else {
        apply_activation(
            &conv_output,
            activation,
            weights,
            out_channels, // post-conv channels for alpha
            dtype,
            step_idx,
        )?
    };

    dyn_to_slice(&output, step_idx, "FusedConv1dActivation")
}

/// Apply the given activation function to a tensor.
///
/// Shared between pre-activation and post-activation modes.
fn apply_activation(
    input: &nn_core::DynTensor,
    activation: &ConvActivation,
    weights: &std::collections::HashMap<String, crate::buffer::MetalBuffer>,
    channels: usize,
    dtype: nn_core::DType,
    step_idx: usize,
) -> Result<nn_core::DynTensor> {
    match activation {
        ConvActivation::Relu => input.relu().map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dActivation relu: {e}"))
        }),
        ConvActivation::LeakyRelu { slope } => {
            input.leaky_relu(f64::from(*slope)).map_err(|e| {
                native_dispatch_err(step_idx, format!("FusedConv1dActivation leaky_relu: {e}"))
            })
        }
        ConvActivation::Silu => input.silu().map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dActivation silu: {e}"))
        }),
        ConvActivation::Snake => {
            // Snake activation: x + (1/alpha) * sin(alpha * x)^2
            let alpha_tensor = weight_to_dyn(
                weights,
                "alpha",
                &[channels],
                dtype,
                step_idx,
                "FusedConv1dActivation",
            )?;
            let alpha_3d = alpha_tensor.reshape([1, channels, 1]).map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedConv1dActivation alpha reshape: {e}"),
                )
            })?;
            let ax = input.broadcast_mul(&alpha_3d).map_err(|e| {
                native_dispatch_err(step_idx, format!("FusedConv1dActivation alpha*x: {e}"))
            })?;
            let sin_ax = ax.sin().map_err(|e| {
                native_dispatch_err(step_idx, format!("FusedConv1dActivation sin: {e}"))
            })?;
            let sin2 = sin_ax.mul(&sin_ax).map_err(|e| {
                native_dispatch_err(step_idx, format!("FusedConv1dActivation sin^2: {e}"))
            })?;
            let inv_alpha = alpha_3d.recip().map_err(|e| {
                native_dispatch_err(step_idx, format!("FusedConv1dActivation 1/alpha: {e}"))
            })?;
            let scaled = sin2.broadcast_mul(&inv_alpha).map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedConv1dActivation sin^2/alpha: {e}"),
                )
            })?;
            input.add(&scaled).map_err(|e| {
                native_dispatch_err(step_idx, format!("FusedConv1dActivation add: {e}"))
            })
        }
        ConvActivation::Gelu => input.gelu().map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dActivation gelu: {e}"))
        }),
        ConvActivation::GeluErf => input.gelu_erf().map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dActivation gelu_erf: {e}"))
        }),
        ConvActivation::Tanh => input.tanh().map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedConv1dActivation tanh: {e}"))
        }),
        _ => Err(native_dispatch_err(
            step_idx,
            format!("FusedConv1dActivation: unsupported activation variant {activation:?}"),
        )),
    }
}
