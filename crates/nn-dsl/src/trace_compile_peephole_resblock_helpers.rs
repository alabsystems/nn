// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Helper functions for ResBlock peephole fusion.
//!
//! Extracted from `trace_compile_peephole_resblock.rs` for 450-line compliance.
//! Contains `extract_norm_activ_params` and `detect_post_add_scale`.
//!
//! Part of #2218.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;

use super::super::super::{CompiledStep, NativeOpKind, NormActivConv1dParams, NormActivation};

/// Extract NormActivConv1d params from a CompiledStep.
pub(super) fn extract_norm_activ_params(
    step: &CompiledStep,
) -> Option<(
    NormActivConv1dParams,
    HashMap<String, nn_core::dyn_tensor::trace::WeightRef>,
)> {
    match step {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::NormActivConv1d {
                    activation,
                    eps,
                    conv_dilation,
                    conv_padding,
                    input_shape,
                    output_channels,
                    kernel_size,
                    ..
                },
            weight_data,
        } => Some((
            NormActivConv1dParams {
                activation: activation.clone(),
                eps: *eps,
                conv_dilation: *conv_dilation,
                conv_padding: *conv_padding,
                input_shape: input_shape.clone(),
                output_channels: *output_channels,
                kernel_size: *kernel_size,
            },
            weight_data.clone(),
        )),
        _ => None,
    }
}

/// Extract AdaIN-only info from a standalone AdainLeakyRelu or AdainSnake NativeOp step.
///
/// Returns `(activation, eps, input_shape, weight_data)` if the step matches.
/// Used by the unfused phase1 path for upsample ResBlocks (#3510).
pub(super) fn extract_standalone_adain_params(
    step: &CompiledStep,
) -> Option<(
    NormActivation,
    f32,
    Vec<usize>,
    HashMap<String, nn_core::dyn_tensor::trace::WeightRef>,
)> {
    match step {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::AdainLeakyRelu {
                    eps,
                    slope,
                    input_shape,
                    ..
                },
            weight_data,
        } => Some((
            NormActivation::LeakyRelu { slope: *slope },
            *eps,
            input_shape.clone(),
            weight_data.clone(),
        )),
        CompiledStep::NativeOp {
            op: NativeOpKind::AdainSnake {
                eps, input_shape, ..
            },
            weight_data,
        } => Some((
            NormActivation::Snake,
            *eps,
            input_shape.clone(),
            weight_data.clone(),
        )),
        _ => None,
    }
}

/// Detect optional post-add `ConstantValue + Dispatch "mul"` pattern.
///
/// Returns `(fused_position, residual_scale, steps_to_replace)`:
/// - If pattern found: FusedResBlock goes at the mul position, scale is
///   extracted, and add + const steps are replaced.
/// - If not found: FusedResBlock goes at the add position, scale = 1.0.
pub(super) fn detect_post_add_scale(
    add_idx: usize,
    steps: &[CompiledStep],
    nodes: &[nn_core::dyn_tensor::trace::TraceNode],
    id_to_idx: &HashMap<u64, usize>,
    consumers: &HashMap<u64, Vec<usize>>,
    use_counts: &[usize],
) -> (usize, f32, Vec<usize>) {
    // The add output must have exactly 1 consumer.
    let add_id = nodes[add_idx].id();
    let add_consumers = match consumers.get(&add_id) {
        Some(c) if c.len() == 1 => c,
        _ => return (add_idx, 1.0, vec![]),
    };

    let mul_idx = add_consumers[0];
    // Consumer must be a Dispatch "mul".
    if !matches!(
        &steps[mul_idx],
        CompiledStep::Dispatch { kernel, .. } if kernel.name() == "mul"
    ) {
        return (add_idx, 1.0, vec![]);
    }

    // The mul node's inputs: one is the add output, the other is a constant.
    let mul_inputs = nodes[mul_idx].inputs();
    if mul_inputs.len() != 2 {
        return (add_idx, 1.0, vec![]);
    }

    let other_id = if mul_inputs[0] == add_id {
        mul_inputs[1]
    } else {
        mul_inputs[0]
    };

    let const_idx = match id_to_idx.get(&other_id) {
        Some(&idx) => idx,
        None => return (add_idx, 1.0, vec![]),
    };

    // The other step must be a scalar constant: either ConstantValue (from
    // TraceOp::Constant) or NativeOp::ConstantWeight (from auto-registered
    // scalar tensors, e.g. mul_scalar in F0 AdainResBlk1d).
    let scale = match (&steps[const_idx], nodes[const_idx].op()) {
        (CompiledStep::ConstantValue { .. }, TraceOp::Constant { value }) => *value as f32,
        (
            CompiledStep::NativeOp {
                op: NativeOpKind::ConstantWeight { .. },
                ..
            },
            TraceOp::ConstantWeight { weight },
        ) if weight.data().len() == 1 => weight.data()[0],
        _ => return (add_idx, 1.0, vec![]),
    };

    // Also check that the add step's use_count is 1 (only feeds into mul).
    if use_counts.get(add_idx).copied().unwrap_or(0) != 1 {
        return (add_idx, 1.0, vec![]);
    }

    // Absorb: FusedResBlock goes at mul position, replace add with IP.
    // Do NOT replace ConstantValue — it's a root node (no graph inputs),
    // so IdentityPassthrough would fail to resolve input 0 from the empty edge_map.
    // The constant buffer is tiny and harmless if unused.
    (mul_idx, scale, vec![add_idx])
}
