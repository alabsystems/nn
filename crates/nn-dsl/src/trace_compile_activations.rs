// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Decomposed activation and binary min/max compilation helpers.
//!
//! Extracted from `trace_compile_ops.rs` to stay within the 450-line limit.
//! These ops are decomposed into simpler primitives within a single builder.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, WeightRef};

use crate::ir::{MinMaxKind, UnaryFnKind};
use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_builders::{minmax_kernel, unary_kernel};
use crate::tensor_ir::TensorIRError;

use super::super::{resolve_input_shape, CompiledKernel, CompiledStep};
use super::build_single_op;

// -- Activations (decomposed) ------------------------------------------------

/// SiLU (Swish): `silu(x) = x * sigmoid(x)`.
pub(in crate::trace_compile) fn compile_silu(
    node: &TraceNode,
    graph: &ComputationGraph,
) -> Result<CompiledStep, TensorIRError> {
    build_single_op("silu", node, graph, 1, |b, inputs| {
        let sig = b.add_sigmoid(inputs[0], node.output_shape());
        b.add_binary_mul(inputs[0], sig, node.output_shape())
    })
}

/// ELU: `elu(x, alpha) = x if x >= 0, else alpha * (exp(x) - 1)`.
///
/// Compiles to a single `TensorOpKind::Elu` IR node, which the MSL codegen
/// emits as a single elementwise kernel with alpha baked in as a compile-time
/// constant. No extra weight buffer needed.
///
/// Replaces the previous ~10 node decomposition (relu, neg, exp, broadcast,
/// sub, mul, add — 3 weight scalars), saving ~6 Metal dispatches per ELU call.
/// Part of #3230 (Gap 3).
pub(in crate::trace_compile) fn compile_elu(
    node: &TraceNode,
    graph: &ComputationGraph,
    alpha: f32,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let mut b = TensorBlockBuilder::new("elu");
    let input = b.add_input("input_0", input_shape);
    let output = b.add_elu(input, alpha, node.output_shape());
    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

/// LeakyRelu: `leaky_relu(x, slope) = select(x, slope*x, x < 0)`.
///
/// Compiles to a single `TensorOpKind::LeakyRelu` IR node, which the MSL
/// codegen emits as a single elementwise kernel with the slope baked in as a
/// compile-time constant. No extra weight buffer needed.
///
/// Replaces the previous 3-7 node decomposition, saving ~30 Metal dispatches
/// in the Kokoro model. Part of #3230 (Gap 3).
pub(in crate::trace_compile) fn compile_leaky_relu(
    node: &TraceNode,
    graph: &ComputationGraph,
    negative_slope: f32,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let mut b = TensorBlockBuilder::new("leaky_relu");
    let input = b.add_input("input_0", input_shape);
    let output = b.add_leaky_relu(input, negative_slope, node.output_shape());
    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

/// Softplus: `softplus(x) = log(1 + exp(x))`.
///
/// Compiles to a single `TensorOpKind::Softplus` IR node. The MSL codegen
/// emits `log(1.0 + exp(x))` as a single elementwise kernel.
/// Part of #3230.
pub(in crate::trace_compile) fn compile_softplus(
    node: &TraceNode,
    graph: &ComputationGraph,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let mut b = TensorBlockBuilder::new("softplus");
    let input = b.add_input("input_0", input_shape);
    let output = b.add_softplus(input, node.output_shape());
    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

/// Per-channel Snake activation: `snake(x, alpha) = x + (1/alpha) * sin²(alpha * x)`.
///
/// Alpha is a per-channel weight tensor (typically `[1, C, 1]`). The scalar
/// `build_snake_scalar_kernel()` is applied elementwise after broadcasting
/// alpha to the full input shape.
pub(in crate::trace_compile) fn compile_snake_tensor(
    node: &TraceNode,
    graph: &ComputationGraph,
    alpha: &WeightRef,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let shape = node.output_shape();
    let mut b = TensorBlockBuilder::new("snake_tensor");
    let input = b.add_input("input_0", input_shape);

    // Alpha input: per-channel weight, broadcast to full shape.
    let alpha_input = b.add_input("alpha", alpha.shape());
    let alpha_bc = b.add_broadcast_left(alpha_input, shape);

    // Apply snake scalar kernel: snake(x, alpha) = x + (1/alpha) * sin²(alpha * x)
    let snake_kernel = crate::adain::build_snake_scalar_kernel()
        .map_err(|e| TensorIRError::ScalarKernelBuild(e.to_string()))?;
    let output = b.add_elementwise(snake_kernel, &[input, alpha_bc], shape);
    let def = b.build(output)?;

    let mut weight_data = HashMap::new();
    weight_data.insert("alpha".to_string(), alpha.clone());

    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

/// LogSoftmax: `log_softmax(x, dim) = log(softmax(x, dim))`.
pub(in crate::trace_compile) fn compile_log_softmax(
    node: &TraceNode,
    graph: &ComputationGraph,
    dim: usize,
) -> Result<CompiledStep, TensorIRError> {
    let dim_i32 = i32::try_from(dim).map_err(|_| TensorIRError::SoftmaxDimOverflow { dim })?;
    let log_kernel = unary_kernel("log", UnaryFnKind::Log);
    build_single_op("log_softmax", node, graph, 1, |b, inputs| {
        let sm = b.add_softmax(inputs[0], dim_i32, node.output_shape());
        b.add_elementwise(log_kernel.clone(), &[sm], node.output_shape())
    })
}

// -- Binary min/max ----------------------------------------------------------

/// Element-wise min or max via `MinMaxKind` scalar kernel.
///
/// Handles broadcast when inputs have different shapes (e.g., `maximum(tensor, scalar)`
/// from `clamp_min` during tracing).
pub(in crate::trace_compile) fn compile_binary_minmax(
    node: &TraceNode,
    graph: &ComputationGraph,
    name: &str,
    op: MinMaxKind,
) -> Result<CompiledStep, TensorIRError> {
    let lhs_shape = resolve_input_shape(node, 0, graph)?;
    let rhs_shape = resolve_input_shape(node, 1, graph)?;
    let out = node.output_shape();
    let kernel = minmax_kernel(name, op);
    let mut b = TensorBlockBuilder::new(name);
    let lhs = b.add_input("input_0", lhs_shape);
    let rhs = b.add_input("input_1", rhs_shape);
    let lhs_bc = if lhs_shape == out {
        lhs
    } else {
        b.add_broadcast(lhs, out)
    };
    let rhs_bc = if rhs_shape == out {
        rhs
    } else {
        b.add_broadcast(rhs, out)
    };
    let output = b.add_elementwise(kernel, &[lhs_bc, rhs_bc], out);
    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: super::graph_input_ids(node, 2),
    })
}
