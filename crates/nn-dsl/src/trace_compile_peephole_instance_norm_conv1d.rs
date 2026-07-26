// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass: fuse InstanceNorm + Conv1d → FusedInstanceNormConv1d.
//!
//! Detects consecutive step pairs:
//!   Step i:   NativeOp { InstanceNorm { eps, input_shape } }
//!   Step i+1: Dispatch { kernel: "conv1d" }
//!
//! The InstanceNorm output must be single-consumer (use_counts == 1).
//!
//! Replaces the pair with:
//! - steps[i]   → NativeOp { FusedInstanceNormConv1d { ... } }
//! - steps[i+1] → IdentityPassthrough
//!
//! Saves 1 dispatch per pair. In Kokoro generator/decoder, InstanceNorm →
//! Conv1d patterns appear in channel projection layers that lack style
//! affine (no gamma/beta), unlike the NormActivConv1d pattern.
//!
//! Must run AFTER FusedResBlock, NormActivConv1d, and
//! FusedAddInstanceNormConv1x1 which handle deeper patterns.
//!
//! Part of #4264.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, WeightRef};

use super::super::{CompiledStep, NativeOpKind};
use super::extract_conv1d_params;

/// Scan for InstanceNorm + Conv1d pairs and fuse them.
pub(super) fn fuse_instance_norm_conv1d(
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

/// Try to fuse steps[i] (InstanceNorm) with steps[i+1] (Conv1d).
fn try_fuse(steps: &mut [CompiledStep], i: usize, use_counts: &[usize]) -> bool {
    // ---- Step i: NativeOp { InstanceNorm } ----
    let (eps, norm_input_shape) = match &steps[i] {
        CompiledStep::NativeOp {
            op: NativeOpKind::InstanceNorm { eps, input_shape },
            ..
        } => (*eps, input_shape.clone()),
        _ => return false,
    };

    // Fan-out: InstanceNorm output must have exactly 1 consumer.
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }

    // ---- Step i+1: Dispatch with kernel name "conv1d" ----
    let conv_info = match &steps[i + 1] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } if kernel.name() == "conv1d" => extract_conv1d_params(kernel, weight_data),
        _ => None,
    };

    let conv_info = match conv_info {
        Some(info) => info,
        None => return false,
    };

    // Build merged weight_data.
    let mut merged_weight_data: HashMap<String, WeightRef> = HashMap::new();
    if let Some(w) = conv_info.weight {
        merged_weight_data.insert("conv_weight".to_string(), w);
    }
    if let Some(b) = conv_info.bias {
        merged_weight_data.insert("conv_bias".to_string(), b);
    }

    let fused_op = NativeOpKind::FusedInstanceNormConv1d {
        eps,
        out_channels: conv_info.output_channels,
        kernel_size: conv_info.kernel_size,
        stride: conv_info.stride,
        padding: conv_info.padding,
        dilation: conv_info.dilation,
        groups: conv_info.groups,
        has_bias: merged_weight_data.contains_key("conv_bias"),
        input_shape: norm_input_shape,
    };

    steps[i] = CompiledStep::NativeOp {
        op: fused_op,
        weight_data: merged_weight_data,
    };
    steps[i + 1] = CompiledStep::IdentityPassthrough;

    true
}
