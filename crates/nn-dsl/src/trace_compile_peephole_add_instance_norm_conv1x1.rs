// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass: Add + InstanceNorm + Conv1d(K=1) → FusedAddInstanceNormConv1x1.
//!
//! Detects the 3-step pattern in compiled steps:
//!   Step i:   Dispatch { kernel: "add" }
//!   Step i+1: NativeOp { InstanceNorm { eps, input_shape } }
//!   Step i+2: Dispatch { kernel: "conv1d" } where kernel_size == 1
//!
//! Both intermediate outputs must be single-consumer (use_counts == 1).
//!
//! Replaces the 3 steps with:
//! - steps[i]   → NativeOp { FusedAddInstanceNormConv1x1 { ... } }
//! - steps[i+1] → IdentityPassthrough
//! - steps[i+2] → IdentityPassthrough
//!
//! Saves 2 Metal dispatches per Add+InstanceNorm+Conv1d(K=1) triple. In the
//! Kokoro decoder, this pattern appears when the residual sum feeds through
//! instance normalization and a 1x1 convolution for channel dimension changes.
//!
//! Runs AFTER FusedResBlock passes which consume deeper patterns first,
//! and BEFORE FusedConv1dActivation.
//!
//! Part of #4264.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, WeightRef};

use super::super::{CompiledStep, NativeOpKind};
use super::extract_conv1d_params;

/// Scan for Add + InstanceNorm + Conv1d(K=1) triples and fuse them.
pub(super) fn fuse_add_instance_norm_conv1x1(
    steps: &mut [CompiledStep],
    use_counts: &[usize],
    _graph: &ComputationGraph,
) {
    let len = steps.len();
    if len < 3 {
        return;
    }

    let mut i = 0;
    while i + 2 < len {
        if try_fuse(steps, i, use_counts) {
            // Skip past the fused triple.
            i += 3;
        } else {
            i += 1;
        }
    }
}

/// Try to fuse steps[i..i+3] as Add + InstanceNorm + Conv1d(K=1).
///
/// Returns `true` if the triple was fused (steps mutated in-place).
fn try_fuse(steps: &mut [CompiledStep], i: usize, use_counts: &[usize]) -> bool {
    // ---- Step i: Dispatch with kernel name "add" ----
    let is_add = matches!(
        &steps[i],
        CompiledStep::Dispatch { kernel, .. } if kernel.name() == "add"
    );
    if !is_add {
        return false;
    }

    // Fan-out: add output must have exactly 1 consumer.
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }

    // ---- Step i+1: NativeOp { InstanceNorm { eps, input_shape } } ----
    let (eps, input_shape) = match &steps[i + 1] {
        CompiledStep::NativeOp {
            op: NativeOpKind::InstanceNorm { eps, input_shape },
            ..
        } => (*eps, input_shape.clone()),
        _ => return false,
    };

    // InstanceNorm output must have exactly 1 consumer.
    if use_counts.get(i + 1).copied().unwrap_or(0) != 1 {
        return false;
    }

    // ---- Step i+2: Dispatch with kernel name "conv1d" and kernel_size == 1 ----
    let (conv_info, _conv_weight_data) = match &steps[i + 2] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } if kernel.name() == "conv1d" => match extract_conv1d_params(kernel, weight_data) {
            Some(info) if info.kernel_size == 1 => (info, weight_data.clone()),
            _ => return false,
        },
        _ => return false,
    };

    // Only fuse stride=1, groups=1, dilation=1, padding=0 (standard 1x1 conv).
    if conv_info.stride != 1 || conv_info.groups != 1 || conv_info.dilation != 1 {
        return false;
    }
    if conv_info.padding != 0 {
        return false;
    }

    // Input shape must be at least 3D: [B, C_in, T].
    if input_shape.len() < 3 {
        return false;
    }

    let in_channels = input_shape[1];
    let out_channels = conv_info.output_channels;
    let has_bias = conv_info.bias.is_some();

    // Build weight_data for the fused NativeOp.
    let mut fused_weights: HashMap<String, WeightRef> = HashMap::new();
    if let Some(w) = conv_info.weight {
        fused_weights.insert("conv_weight".to_string(), w);
    }
    if let Some(b) = conv_info.bias {
        fused_weights.insert("conv_bias".to_string(), b);
    }
    // Carry over InstanceNorm weights (if any, though InstanceNorm typically
    // has no learnable parameters in inference mode).
    if let CompiledStep::NativeOp { weight_data, .. } = &steps[i + 1] {
        for (k, v) in weight_data {
            fused_weights.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }

    // Replace steps: fuse into steps[i], passthrough steps[i+1] and [i+2].
    steps[i] = CompiledStep::NativeOp {
        op: NativeOpKind::FusedAddInstanceNormConv1x1 {
            eps,
            input_shape,
            in_channels,
            out_channels,
            has_bias,
        },
        weight_data: fused_weights,
    };
    steps[i + 1] = CompiledStep::IdentityPassthrough;
    steps[i + 2] = CompiledStep::IdentityPassthrough;

    true
}
