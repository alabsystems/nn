// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused AdaIN activation compilation helpers.
//!
//! Contains `compile_adain_snake` and `compile_adain_leaky_relu` — fused
//! InstanceNorm + style affine + activation in a single compiled step.
//!
//! Extracted from `trace_compile_activations.rs` to keep it under 450 lines.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, WeightRef};

use crate::ir::BinOpKind;
use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_builders::binop_kernel;
use crate::tensor_ir::TensorIRError;

use super::super::{resolve_input_shape, CompiledKernel, CompiledStep};

/// Fused AdaIN + Snake: `snake((1 + gamma) * InstanceNorm(x) + beta, alpha)`.
///
/// Tensor inputs: `[x, gamma, beta]` (3 trace graph inputs).
/// Weight data: `alpha` (per-channel Snake parameter), `eps` (InstanceNorm epsilon).
///
/// The computation graph:
/// 1. InstanceNorm(x, eps, axis=last) -> normed
/// 2. (1 + gamma) * normed + beta -> adain_out (style affine)
/// 3. snake(adain_out, alpha) -> output
pub(in crate::trace_compile) fn compile_adain_snake(
    node: &TraceNode,
    graph: &ComputationGraph,
    alpha: &WeightRef,
    eps: f64,
) -> Result<CompiledStep, TensorIRError> {
    let eps_f32 = eps as f32;
    if !eps_f32.is_finite() || eps_f32 < 0.0 {
        return Err(TensorIRError::NonFiniteConstant {
            name: "AdainSnake eps".into(),
            value: eps,
        });
    }
    let x_shape = resolve_input_shape(node, 0, graph)?;

    // Emit a NativeOp for rank >= 3 input (the common case: [B, C, T]).
    // The fused kernel combines InstanceNorm + affine + Snake in a single
    // Metal dispatch — replaces ~20 dispatches. Part of #2472.
    if x_shape.len() >= 3 {
        let channels = x_shape[1];
        let mut weight_data = HashMap::new();
        weight_data.insert("alpha".to_string(), alpha.clone());
        // Capture graph node IDs so the edge_map builder resolves edges
        // generically without per-NativeOp patches. Part of #3261.
        let ext_ids = Some(node.inputs().to_vec());
        return Ok(CompiledStep::NativeOp {
            op: super::super::NativeOpKind::AdainSnake {
                eps: eps_f32,
                input_shape: x_shape.to_vec(),
                channels,
                residual_gamma: true, // Kokoro convention: (1+gamma)*normed+beta
                external_node_ids: ext_ids,
            },
            weight_data,
        });
    }

    // Fallback for rank < 3 (unusual): use the IR decomposition path.
    let gamma_shape = resolve_input_shape(node, 1, graph)?;
    let beta_shape = resolve_input_shape(node, 2, graph)?;
    let shape = node.output_shape();
    let ndim = x_shape.len();
    let axis = if ndim > 0 { ndim - 1 } else { 0 };

    let mut b = TensorBlockBuilder::new("adain_snake");
    let x = b.add_input("input_0", x_shape);
    let gamma = b.add_input("input_1", gamma_shape);
    let beta = b.add_input("input_2", beta_shape);

    // eps scalar weight
    let eps_node = b.add_input("eps", &[1]);

    // 1. InstanceNorm(x) over spatial axis
    let normed = b.add_instance_norm(x, eps_node, axis, None, None, shape);

    // 2. Affine: (1 + gamma) * normed + beta
    let ones_node = b.add_input("ones", &[1]);
    let ones_bc = b.add_broadcast(ones_node, gamma_shape);
    let scale = b.add_binary_add(ones_bc, gamma, gamma_shape);
    let scale_bc = b.add_broadcast(scale, shape);
    let scaled = b.add_binary_mul(normed, scale_bc, shape);
    let beta_bc = b.add_broadcast(beta, shape);
    let adain_out = b.add_binary_add(scaled, beta_bc, shape);

    // 3. Snake(adain_out, alpha)
    let alpha_node = b.add_input("alpha", alpha.shape());
    let alpha_bc = b.add_broadcast_left(alpha_node, shape);
    let snake_kernel = crate::adain::build_snake_scalar_kernel()
        .map_err(|e| TensorIRError::ScalarKernelBuild(e.to_string()))?;
    let output = b.add_elementwise(snake_kernel, &[adain_out, alpha_bc], shape);
    let def = b.build(output)?;

    let mut weight_data = HashMap::new();
    weight_data.insert(
        "eps".to_string(),
        WeightRef::new(vec![eps_f32], vec![1]).expect("valid eps scalar"),
    );
    weight_data.insert(
        "ones".to_string(),
        WeightRef::new(vec![1.0f32], vec![1]).expect("valid scalar"),
    );
    weight_data.insert("alpha".to_string(), alpha.clone());

    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, 3),
    })
}

/// Fused AdaIN + LeakyRelu: `leaky_relu((1 + gamma) * InstanceNorm(x) + beta, slope)`.
///
/// Tensor inputs: `[x, gamma, beta]` (3 trace graph inputs).
/// Weight data: `eps` (InstanceNorm epsilon), `slope` and `zero` (LeakyRelu params).
///
/// The computation graph:
/// 1. InstanceNorm(x, eps, axis=last) -> normed
/// 2. (1 + gamma) * normed + beta -> adain_out (style affine)
/// 3. leaky_relu(adain_out, slope) -> output
pub(in crate::trace_compile) fn compile_adain_leaky_relu(
    node: &TraceNode,
    graph: &ComputationGraph,
    eps: f64,
    slope: f64,
) -> Result<CompiledStep, TensorIRError> {
    let eps_f32 = eps as f32;
    if !eps_f32.is_finite() || eps_f32 < 0.0 {
        return Err(TensorIRError::NonFiniteConstant {
            name: "AdainLeakyRelu eps".into(),
            value: eps,
        });
    }
    let slope_f32 = slope as f32;
    if !slope_f32.is_finite() {
        return Err(TensorIRError::NonFiniteConstant {
            name: "AdainLeakyRelu slope".into(),
            value: slope,
        });
    }
    let x_shape = resolve_input_shape(node, 0, graph)?;

    // Emit a NativeOp for rank >= 3 input (the common case: [B, C, T]).
    // The fused kernel combines InstanceNorm + affine + LeakyRelu in a single
    // Metal dispatch — replaces ~20 dispatches. Part of #2472.
    if x_shape.len() >= 3 {
        // Capture graph node IDs so the edge_map builder resolves edges
        // generically without per-NativeOp patches. Part of #3261.
        let ext_ids = Some(node.inputs().to_vec());
        return Ok(CompiledStep::NativeOp {
            op: super::super::NativeOpKind::AdainLeakyRelu {
                eps: eps_f32,
                slope: slope_f32,
                input_shape: x_shape.to_vec(),
                external_node_ids: ext_ids,
            },
            weight_data: HashMap::new(),
        });
    }

    // Fallback for rank < 3 (unusual): use the IR decomposition path.
    let gamma_shape = resolve_input_shape(node, 1, graph)?;
    let beta_shape = resolve_input_shape(node, 2, graph)?;
    let shape = node.output_shape();
    let ndim = x_shape.len();
    let axis = if ndim > 0 { ndim - 1 } else { 0 };

    let mut b = TensorBlockBuilder::new("adain_leaky_relu");
    let x = b.add_input("input_0", x_shape);
    let gamma = b.add_input("input_1", gamma_shape);
    let beta = b.add_input("input_2", beta_shape);

    // eps scalar weight
    let eps_node = b.add_input("eps", &[1]);

    // 1. InstanceNorm(x) over spatial axis
    let normed = b.add_instance_norm(x, eps_node, axis, None, None, shape);

    // 2. Affine: (1 + gamma) * normed + beta
    let ones_node = b.add_input("ones", &[1]);
    let ones_bc = b.add_broadcast(ones_node, gamma_shape);
    let scale = b.add_binary_add(ones_bc, gamma, gamma_shape);
    let scale_bc = b.add_broadcast(scale, shape);
    let scaled = b.add_binary_mul(normed, scale_bc, shape);
    let beta_bc = b.add_broadcast(beta, shape);
    let adain_out = b.add_binary_add(scaled, beta_bc, shape);

    // 3. LeakyRelu(adain_out, slope): relu(x) - slope * relu(-x)
    let relu_pos = b.add_relu(adain_out, shape);
    let zero_node = b.add_input("zero", &[1]);
    let zero_bc = b.add_broadcast(zero_node, shape);
    let sub_kernel = binop_kernel("sub", BinOpKind::Sub);
    let neg = b.add_elementwise(sub_kernel.clone(), &[zero_bc, adain_out], shape);
    let relu_neg = b.add_relu(neg, shape);
    let slope_node = b.add_input("slope", &[1]);
    let slope_bc = b.add_broadcast(slope_node, shape);
    let slope_relu = b.add_binary_mul(slope_bc, relu_neg, shape);
    let output = b.add_elementwise(sub_kernel, &[relu_pos, slope_relu], shape);
    let def = b.build(output)?;

    let mut weight_data = HashMap::new();
    weight_data.insert(
        "eps".to_string(),
        WeightRef::new(vec![eps_f32], vec![1]).expect("valid eps scalar"),
    );
    weight_data.insert(
        "ones".to_string(),
        WeightRef::new(vec![1.0f32], vec![1]).expect("valid scalar"),
    );
    weight_data.insert(
        "zero".to_string(),
        WeightRef::new(vec![0.0f32], vec![1]).expect("valid scalar"),
    );
    weight_data.insert(
        "slope".to_string(),
        WeightRef::new(vec![slope_f32], vec![1]).expect("valid scalar"),
    );

    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, 3),
    })
}
