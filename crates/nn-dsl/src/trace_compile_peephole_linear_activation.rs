// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass 5: Linear + Activation → LinearActivation (GEMM epilogue).
//!
//! Fuses adjacent `Dispatch{linear}` + `Dispatch{activation}` into a single
//! `NativeOpKind::LinearActivation`. The fused kernel performs matmul with
//! activation epilogue in one Metal dispatch. Part of #2218.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::WeightRef;

use crate::tensor_ir::TensorOpKind;

use super::super::{CompiledKernel, CompiledStep, GemmActivation, NativeOpKind};

/// Scan for `Dispatch{linear}` + `Dispatch{activation}` pairs and fuse them.
pub(super) fn fuse_linear_activation(steps: &mut [CompiledStep], use_counts: &[usize]) {
    let len = steps.len();
    if len < 2 {
        return;
    }
    let mut i = 0;
    while i + 1 < len {
        if try_fuse_linear_activation(steps, i, use_counts) {
            i += 2;
        } else {
            i += 1;
        }
    }
}

/// Try to fuse steps[i] (Linear) with steps[i+1] (single activation).
fn try_fuse_linear_activation(steps: &mut [CompiledStep], i: usize, use_counts: &[usize]) -> bool {
    // Step[i] must be Dispatch with kernel name "linear".
    let linear_info = match &steps[i] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } if kernel.name() == "linear" => extract_linear_params(kernel, weight_data),
        _ => None,
    };
    let linear_info = match linear_info {
        Some(info) => info,
        None => return false,
    };

    // Fan-out: Linear output must have exactly 1 consumer.
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }

    // Step[i+1] must be a single-op activation Dispatch (not a fused chain).
    let activation = match &steps[i + 1] {
        CompiledStep::Dispatch { kernel, .. } if is_single_activation(kernel) => {
            activation_from_kernel_name(kernel.name())
        }
        _ => None,
    };
    let activation = match activation {
        Some(a) => a,
        None => return false,
    };

    // Build fused NativeOp.
    steps[i] = CompiledStep::NativeOp {
        op: NativeOpKind::LinearActivation {
            activation,
            in_features: linear_info.in_features,
            out_features: linear_info.out_features,
            has_bias: linear_info.has_bias,
            input_shape: linear_info.input_shape,
        },
        weight_data: linear_info.weight_data,
    };
    steps[i + 1] = CompiledStep::IdentityPassthrough;
    true
}

/// Extracted Linear parameters for peephole matching.
pub(crate) struct LinearInfo {
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
    pub(crate) has_bias: bool,
    pub(crate) input_shape: Vec<usize>,
    pub(crate) weight_data: HashMap<String, WeightRef>,
}

/// Extract Linear parameters from a CompiledKernel's IR nodes.
pub(crate) fn extract_linear_params(
    kernel: &CompiledKernel,
    weight_data: &HashMap<String, WeightRef>,
) -> Option<LinearInfo> {
    let def = kernel.def();

    // Find the Linear IR node.
    def.nodes
        .iter()
        .find(|n| matches!(n.kind, TensorOpKind::Linear { .. }))?;

    // Extract in/out features from weight shape [out_features, in_features].
    let weight_ref = weight_data.get("weight")?;
    let weight_shape = weight_ref.shape();
    if weight_shape.len() != 2 {
        return None;
    }
    let out_features = weight_shape[0];
    let in_features = weight_shape[1];
    let has_bias = weight_data.contains_key("bias");

    // Input shape from the first Input node's shape.
    let input_shape = def
        .nodes
        .iter()
        .find(|n| matches!(n.kind, TensorOpKind::Input { .. }))
        .map(|n| n.shape.clone())
        .unwrap_or_default();

    Some(LinearInfo {
        in_features,
        out_features,
        has_bias,
        input_shape,
        weight_data: weight_data.clone(),
    })
}

/// Check if a kernel is a single-op activation (not a fused chain).
fn is_single_activation(kernel: &CompiledKernel) -> bool {
    let name = kernel.name();
    // Fused chains are named "fused_{op}_x{N}" — exclude those.
    if name.starts_with("fused_") {
        return false;
    }
    matches!(
        name,
        "relu" | "gelu" | "gelu_erf" | "sigmoid" | "silu" | "tanh"
    )
}

/// Map kernel name to GemmActivation variant.
fn activation_from_kernel_name(name: &str) -> Option<GemmActivation> {
    match name {
        "relu" => Some(GemmActivation::Relu),
        "gelu" => Some(GemmActivation::Gelu),
        "gelu_erf" => Some(GemmActivation::GeluErf),
        "sigmoid" => Some(GemmActivation::Sigmoid),
        "silu" => Some(GemmActivation::Silu),
        "tanh" => Some(GemmActivation::Tanh),
        _ => None,
    }
}
