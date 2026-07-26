// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! FusedResBlock mixed-activation fallback path.
//!
//! Used only when phase 1 and phase 2 have different activation types
//! (e.g. LeakyRelu + Snake). The fast path in `compiled_model_execute_native_resblock.rs`
//! handles same-activation cases with the fused NormActivConv1d kernel.
//!
//! Extracted to keep the resblock file under 450 lines (#2218).

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Conv1d, Conv1dConfig, Module};
use nn_core::Result;
use nn_dsl::NormActivation;

use crate::compiled_model::CompiledModel;

use super::rb_helpers::load_alpha;
use super::PhaseWeightKeys;
use super::{native_dispatch_err, weight_to_dyn};

/// Run the InstanceNorm + style affine + activation phase (fallback path).
///
/// Delegates to the existing fused AdaIN kernels based on activation type.
/// Uses pre-computed `keys` for weight lookup (#2935).
pub(super) fn run_norm_activ(
    model: &CompiledModel,
    step_idx: usize,
    x: &DynTensor,
    gamma: &DynTensor,
    beta: &DynTensor,
    activation: &NormActivation,
    eps: f32,
    channels: usize,
    keys: &PhaseWeightKeys,
) -> Result<DynTensor> {
    match activation {
        NormActivation::LeakyRelu { slope } => crate::dyn_tensor_metal::native_adain_leaky_relu(
            x,
            gamma,
            beta,
            f64::from(eps),
            f64::from(*slope),
        )
        .map_err(|e| {
            native_dispatch_err(
                step_idx,
                format!("FusedResBlock {}AdainLeakyRelu: {e}", keys.label),
            )
        }),
        NormActivation::Snake => {
            let alpha_tensor = load_alpha(model, step_idx, keys, channels)?;
            // FusedResBlock is Kokoro-specific: always residual_gamma=true.
            crate::dyn_tensor_metal::native_adain_snake(
                x,
                gamma,
                beta,
                &alpha_tensor,
                f64::from(eps),
                true,
            )
            .map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedResBlock {}AdainSnake: {e}", keys.label),
                )
            })
        }
        _ => Err(native_dispatch_err(
            step_idx,
            format!(
                "FusedResBlock {}unsupported NormActivation variant",
                keys.label,
            ),
        )),
    }
}

/// Run the Conv1d phase with pre-uploaded weights (fallback path).
///
/// Uses pre-computed `keys` for weight lookup (#2935).
#[allow(clippy::too_many_arguments)]
pub(super) fn run_conv1d(
    model: &CompiledModel,
    step_idx: usize,
    input: &DynTensor,
    in_channels: usize,
    output_channels: usize,
    kernel_size: usize,
    padding: usize,
    dilation: usize,
    keys: &PhaseWeightKeys,
) -> Result<DynTensor> {
    let dtype = model.step_dtype(step_idx);
    let rb_weights = &model.def.weight_buffers[step_idx];

    let conv_w = weight_to_dyn(
        rb_weights,
        keys.conv_weight,
        &[output_channels, in_channels, kernel_size],
        dtype,
        step_idx,
        "FusedResBlock Conv1d",
    )?;

    let conv_bias = if rb_weights.contains_key(keys.conv_bias) {
        Some(weight_to_dyn(
            rb_weights,
            keys.conv_bias,
            &[output_channels],
            dtype,
            step_idx,
            "FusedResBlock Conv1d",
        )?)
    } else {
        None
    };

    let config = Conv1dConfig::default()
        .with_padding(padding)
        .with_dilation(dilation);
    let conv = Conv1d::new(conv_w, conv_bias, config)?;
    conv.forward(input).map_err(|e| {
        native_dispatch_err(step_idx, format!("FusedResBlock {}Conv1d: {e}", keys.label))
    })
}
