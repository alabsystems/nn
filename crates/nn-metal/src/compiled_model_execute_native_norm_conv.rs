// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! NormActivConv1d executor: fused AdaIN + activation + Conv1d.
//!
//! Extracted from `compiled_model_execute_native_fused.rs` to keep files
//! under 450 lines. Part of #2218.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Device, Result};
use nn_dsl::NormActivation;

use crate::dyn_tensor_metal::MetalTensorData;
use crate::gpu_slice::GpuSlice;

use super::CompiledModel;
use super::{dyn_to_slice, native_dispatch_err, slice_to_dyn, weight_to_dyn};

/// Execute a `NativeOpKind::NormActivConv1d` step (#2780).
///
/// Fuses AdaIN(InstanceNorm + style affine + activation) + Conv1d into a
/// single CompiledStep. Resolves 3 graph inputs (x, gamma, beta), 2 weights
/// (conv_weight, conv_bias), and optional alpha (Snake only).
///
/// Both LeakyRelu and Snake use fused 2-dispatch kernels (stats + inline
/// norm+activation+conv) that eliminate the intermediate activated tensor.
/// Part of #2780.
///
/// Optional `residual_params`: if provided, the fused kernel adds the
/// residual and scales in the same dispatch (used by FusedResBlock phase 2).
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_native_norm_activ_conv1d(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    activation: &NormActivation,
    eps: f32,
    conv_dilation: usize,
    conv_padding: usize,
    input_shape: &[usize],
    output_channels: usize,
    kernel_size: usize,
) -> Result<GpuSlice> {
    execute_norm_activ_conv1d_inner(
        model,
        step_idx,
        buffers,
        activation,
        eps,
        conv_dilation,
        conv_padding,
        input_shape,
        output_channels,
        kernel_size,
        None, // no residual
    )
}

/// Inner implementation shared by NormActivConv1d and FusedResBlock phase 2.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_norm_activ_conv1d_inner(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    activation: &NormActivation,
    eps: f32,
    conv_dilation: usize,
    conv_padding: usize,
    input_shape: &[usize],
    output_channels: usize,
    kernel_size: usize,
    residual: Option<(&DynTensor, f32)>,
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

    // Resolve conv weights.
    let nac_weights = &model.def.weight_buffers[step_idx];
    let conv_w = weight_to_dyn(
        nac_weights,
        "conv_weight",
        &[output_channels, channels, kernel_size],
        dtype,
        step_idx,
        "NormActivConv1d",
    )?;

    // Conv bias: use pre-uploaded buffer or create zeros.
    let conv_bias = if let Some(buf) = nac_weights.get("conv_bias") {
        DynTensor::from_gpu_storage(
            vec![output_channels],
            dtype,
            Arc::new(MetalTensorData::new(buf.alias())),
            Device::metal(),
        )?
    } else {
        DynTensor::zeros(&[output_channels], dtype, &Device::metal())?
    };

    // For LeakyRelu: use fused stats+conv kernel (eliminates intermediate).
    if let NormActivation::LeakyRelu { slope } = activation {
        let residual_params =
            residual.map(|(tensor, scale)| crate::dyn_tensor_metal::ResidualParams {
                residual: tensor,
                scale,
            });
        let output = crate::dyn_tensor_metal::native_norm_activ_conv1d(
            &x_tensor,
            &gamma_tensor,
            &beta_tensor,
            &conv_w,
            &conv_bias,
            f64::from(eps),
            f64::from(*slope),
            conv_padding,
            conv_dilation,
            residual_params,
        )
        .map_err(|e| {
            native_dispatch_err(step_idx, format!("NativeOp NormActivConv1d fused: {e}"))
        })?;

        return dyn_to_slice(&output, step_idx, "NormActivConv1d");
    }

    // Snake: fused stats+norm+conv kernel (same 2-dispatch architecture as LeakyRelu).
    let alpha_tensor = weight_to_dyn(
        nac_weights,
        "alpha",
        &[channels],
        dtype,
        step_idx,
        "NormActivConv1d",
    )?;
    let residual_params = residual.map(|(tensor, scale)| crate::dyn_tensor_metal::ResidualParams {
        residual: tensor,
        scale,
    });
    let output = crate::dyn_tensor_metal::native_norm_activ_conv1d_snake(
        &x_tensor,
        &gamma_tensor,
        &beta_tensor,
        &alpha_tensor,
        &conv_w,
        &conv_bias,
        f64::from(eps),
        conv_padding,
        conv_dilation,
        residual_params,
    )
    .map_err(|e| {
        native_dispatch_err(
            step_idx,
            format!("NativeOp NormActivConv1d Snake fused: {e}"),
        )
    })?;

    dyn_to_slice(&output, step_idx, "NormActivConv1d")
}
