// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass: fuse AdainLeakyRelu/AdainSnake + ConvTranspose1d into
//! NormActivConvTranspose1d.
//!
//! Detects consecutive step pairs:
//!   Step i:   NativeOp { AdainLeakyRelu | AdainSnake }
//!   Step i+1: Dispatch { kernel: "conv_transpose1d" }
//!
//! This is the transposed-conv dual of the NormActivConv1d fusion (pass 1).
//! In Kokoro, the Generator and F0EnergyPredictor use AdainLeakyRelu followed
//! by ConvTranspose1d (stride>1) for upsampling. These fall outside the
//! NormActivConv1d peephole because it only matches regular Conv1d.
//!
//! Part of #4264.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, WeightRef};

use super::super::{CompiledStep, NativeOpKind, NormActivation};
use crate::tensor_ir::TensorOpKind;

/// Scan for AdainLeakyRelu/AdainSnake + ConvTranspose1d pairs and fuse them.
pub(super) fn fuse_norm_activ_conv_transpose1d(
    steps: &mut [CompiledStep],
    use_counts: &[usize],
    graph: &ComputationGraph,
) {
    let len = steps.len();
    if len < 2 {
        return;
    }

    let mut i = 0;
    while i + 1 < len {
        if try_fuse_pair(steps, i, use_counts, graph) {
            i += 2;
        } else {
            i += 1;
        }
    }
}

/// Try to fuse steps[i] (AdainLeakyRelu/AdainSnake) with steps[i+1] (ConvTranspose1d).
///
/// Returns `true` if the pair was fused (steps mutated in-place).
fn try_fuse_pair(
    steps: &mut [CompiledStep],
    i: usize,
    use_counts: &[usize],
    graph: &ComputationGraph,
) -> bool {
    // Step i must be a NativeOp with AdainLeakyRelu or AdainSnake.
    let adain_info = match &steps[i] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::AdainLeakyRelu {
                    eps,
                    slope,
                    input_shape,
                    ..
                },
            weight_data,
        } => Some(AdainInfo {
            activation: NormActivation::LeakyRelu { slope: *slope },
            eps: *eps,
            input_shape: input_shape.clone(),
            adain_weight_data: weight_data.clone(),
        }),
        CompiledStep::NativeOp {
            op: NativeOpKind::AdainSnake {
                eps, input_shape, ..
            },
            weight_data,
        } => Some(AdainInfo {
            activation: NormActivation::Snake,
            eps: *eps,
            input_shape: input_shape.clone(),
            adain_weight_data: weight_data.clone(),
        }),
        _ => None,
    };

    let adain_info = match adain_info {
        Some(info) => info,
        None => return false,
    };

    // Fan-out check: the AdaIN output must have exactly 1 consumer.
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }

    // Step i+1 must be a Dispatch with name "conv_transpose1d".
    let ct_info = match &steps[i + 1] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } if kernel.name() == "conv_transpose1d" => {
            extract_conv_transpose1d_params(kernel, weight_data)
        }
        _ => None,
    };

    let ct_info = match ct_info {
        Some(info) => info,
        None => return false,
    };

    // Build the fused NativeOp.
    let mut merged_weight_data = adain_info.adain_weight_data;

    // Rename conv weights with "conv_" prefix.
    if let Some(w) = ct_info.weight {
        merged_weight_data.insert("conv_weight".to_string(), w);
    }
    if let Some(b) = ct_info.bias {
        merged_weight_data.insert("conv_bias".to_string(), b);
    }
    // Alpha is already in adain_weight_data for Snake variant.

    // Capture the graph node IDs of the AdaIN step's inputs so the
    // edge_map builder can resolve edges generically.
    let ext_ids = graph.nodes().get(i).map(|node| node.inputs().to_vec());

    let fused_op = NativeOpKind::NormActivConvTranspose1d {
        activation: adain_info.activation,
        eps: adain_info.eps,
        kernel_size: ct_info.kernel_size,
        stride: ct_info.stride,
        padding: ct_info.padding,
        dilation: ct_info.dilation,
        groups: ct_info.groups,
        output_padding: ct_info.output_padding,
        output_channels: ct_info.output_channels,
        input_shape: adain_info.input_shape,
        external_node_ids: ext_ids,
    };

    // Place NormActivConvTranspose1d at step[i] (AdaIN position).
    steps[i] = CompiledStep::NativeOp {
        op: fused_op,
        weight_data: merged_weight_data,
    };

    // Replace step[i+1] with IdentityPassthrough.
    steps[i + 1] = CompiledStep::IdentityPassthrough;

    true
}

/// Extracted AdaIN info for pattern matching.
struct AdainInfo {
    activation: NormActivation,
    eps: f32,
    input_shape: Vec<usize>,
    adain_weight_data: HashMap<String, WeightRef>,
}

/// Extracted ConvTranspose1d parameters.
struct ConvTranspose1dInfo {
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    output_padding: usize,
    output_channels: usize,
    weight: Option<WeightRef>,
    bias: Option<WeightRef>,
}

/// Extract ConvTranspose1d parameters from a compiled kernel + weight data.
fn extract_conv_transpose1d_params(
    kernel: &super::super::CompiledKernel,
    weight_data: &HashMap<String, WeightRef>,
) -> Option<ConvTranspose1dInfo> {
    let def = kernel.def();

    let ct_node = def
        .nodes
        .iter()
        .find(|n| matches!(n.kind, TensorOpKind::ConvTranspose1d { .. }))?;

    let (padding, dilation, stride, groups, output_padding) = match &ct_node.kind {
        TensorOpKind::ConvTranspose1d {
            padding,
            dilation,
            stride,
            groups,
            output_padding,
            ..
        } => (*padding, *dilation, *stride, *groups, *output_padding),
        _ => return None,
    };

    // ConvTranspose1d weight shape is [C_in, C_out/groups, K].
    let weight_ref = weight_data.get("weight")?;
    let weight_shape = weight_ref.shape();
    if weight_shape.len() != 3 {
        return None;
    }
    let output_channels = weight_shape[1] * groups;
    let kernel_size = weight_shape[2];

    Some(ConvTranspose1dInfo {
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        output_padding,
        output_channels,
        weight: weight_data.get("weight").cloned(),
        bias: weight_data.get("bias").cloned(),
    })
}
