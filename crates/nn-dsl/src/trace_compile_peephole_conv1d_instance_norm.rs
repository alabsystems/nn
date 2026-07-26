// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass: fuse Conv1d + InstanceNorm → FusedConv1dInstanceNorm.
//!
//! Detects consecutive step pairs:
//!   Step i:   Dispatch { kernel: "conv1d" }
//!   Step i+1: NativeOp { InstanceNorm { eps, input_shape } }
//!
//! The Conv1d output must be single-consumer (use_counts == 1).
//!
//! Replaces the pair with:
//! - steps[i]   → NativeOp { FusedConv1dInstanceNorm { ... } }
//! - steps[i+1] → IdentityPassthrough
//!
//! Saves 1 dispatch per pair. This is the simplest conv→norm pattern,
//! without any activation between conv1d and instance_norm (unlike
//! FusedConv1dSnakeNorm which includes Snake activation in between).
//!
//! Must run AFTER FusedConv1dSnakeNorm (which handles Conv1d→Snake→InstanceNorm)
//! and FusedConv1dActivation (which handles Conv1d→Activation).
//!
//! Part of #4264.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, WeightRef};

use super::super::{CompiledStep, NativeOpKind};
use super::extract_conv1d_params;
use crate::tensor_ir::TensorOpKind;

/// Scan for Conv1d + InstanceNorm pairs and fuse them.
pub(super) fn fuse_conv1d_instance_norm(
    steps: &mut [CompiledStep],
    use_counts: &[usize],
    _graph: &ComputationGraph,
) {
    let len = steps.len();
    if len < 2 {
        return;
    }

    let mut i = 0;
    while i + 1 < len {
        if try_fuse(steps, i, use_counts) {
            i += 2;
        } else {
            i += 1;
        }
    }
}

/// Try to fuse steps[i] (Conv1d) with steps[i+1] (InstanceNorm).
fn try_fuse(steps: &mut [CompiledStep], i: usize, use_counts: &[usize]) -> bool {
    // ---- Step i: Dispatch with kernel name "conv1d" ----
    let (conv_info, conv_weight_data) = match &steps[i] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } if kernel.name() == "conv1d" => match extract_conv1d_params(kernel, weight_data) {
            Some(info) => (info, weight_data.clone()),
            None => return false,
        },
        _ => return false,
    };

    // Fan-out: conv1d output must have exactly 1 consumer.
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }

    // Only fuse groups=1 (common case).
    if conv_info.groups != 1 {
        return false;
    }

    // ---- Step i+1: NativeOp { InstanceNorm } ----
    let (eps, norm_input_shape) = match &steps[i + 1] {
        CompiledStep::NativeOp {
            op: NativeOpKind::InstanceNorm { eps, input_shape },
            ..
        } => (*eps, input_shape.clone()),
        _ => return false,
    };

    // InstanceNorm input shape must be rank >= 3 for [B, C, T].
    if norm_input_shape.len() < 3 {
        return false;
    }

    // Extract conv1d input shape for the fused op.
    let input_shape = extract_conv1d_input_shape(&steps[i]).unwrap_or_default();

    // Build merged weight_data.
    let mut merged_weight_data: HashMap<String, WeightRef> = HashMap::new();
    if let Some(w) = conv_info.weight {
        merged_weight_data.insert("conv_weight".to_string(), w);
    }
    if let Some(b) = conv_info.bias {
        merged_weight_data.insert("conv_bias".to_string(), b);
    }

    let fused_op = NativeOpKind::FusedConv1dInstanceNorm {
        eps,
        out_channels: conv_info.output_channels,
        kernel_size: conv_info.kernel_size,
        stride: conv_info.stride,
        padding: conv_info.padding,
        dilation: conv_info.dilation,
        groups: conv_info.groups,
        has_bias: conv_weight_data.contains_key("bias"),
        input_shape,
    };

    steps[i] = CompiledStep::NativeOp {
        op: fused_op,
        weight_data: merged_weight_data,
    };
    steps[i + 1] = CompiledStep::IdentityPassthrough;

    true
}

/// Extract the input shape from a conv1d Dispatch step's IR.
fn extract_conv1d_input_shape(step: &CompiledStep) -> Option<Vec<usize>> {
    match step {
        CompiledStep::Dispatch { kernel, .. } => {
            let def = kernel.def();
            let input_node = def.nodes.first()?;
            match &input_node.kind {
                TensorOpKind::Input { shape, .. } => Some(shape.clone()),
                _ => None,
            }
        }
        _ => None,
    }
}
