// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Binary op compilation helpers (Add, Sub, Mul, Div, Maximum, Minimum).
//!
//! Extracted from `trace_compile_ops.rs` to keep that file under the
//! 450-line limit. Functions use `pub(in crate::trace_compile)` visibility,
//! matching the pattern in `trace_compile_conv.rs`.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode};

use crate::ir::{BinOpKind, BinaryFnKind};
use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_builders::{binary_fn_kernel, binop_kernel};
use crate::tensor_ir::{TensorIRError, TensorNodeId};

use super::super::{resolve_input_shape, CompiledKernel, CompiledStep};
use super::BinaryMethod;

/// Resolve two binary inputs and broadcast each to `out` when shapes differ.
fn resolve_binary_broadcast(
    node: &TraceNode,
    graph: &ComputationGraph,
    b: &mut TensorBlockBuilder,
) -> Result<[TensorNodeId; 2], TensorIRError> {
    let lhs_shape = resolve_input_shape(node, 0, graph)?;
    let rhs_shape = resolve_input_shape(node, 1, graph)?;
    let out = node.output_shape();
    let lhs = b.add_input("input_0", lhs_shape);
    let rhs = b.add_input("input_1", rhs_shape);
    Ok([
        if lhs_shape == out {
            lhs
        } else {
            b.add_broadcast(lhs, out)
        },
        if rhs_shape == out {
            rhs
        } else {
            b.add_broadcast(rhs, out)
        },
    ])
}

pub(in crate::trace_compile) fn compile_binary_op(
    node: &TraceNode,
    graph: &ComputationGraph,
    name: &str,
    method: BinaryMethod,
) -> Result<CompiledStep, TensorIRError> {
    let mut b = TensorBlockBuilder::new(name);
    let [lhs, rhs] = resolve_binary_broadcast(node, graph, &mut b)?;
    let out = node.output_shape();
    let output = match method {
        BinaryMethod::BuilderAdd => b.add_binary_add(lhs, rhs, out),
        BinaryMethod::BuilderMul => b.add_binary_mul(lhs, rhs, out),
    };
    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: super::graph_input_ids(node, 2),
    })
}

pub(in crate::trace_compile) fn compile_binary_elementwise(
    node: &TraceNode,
    graph: &ComputationGraph,
    name: &str,
    op: BinOpKind,
) -> Result<CompiledStep, TensorIRError> {
    let kernel = binop_kernel(name, op);
    let mut b = TensorBlockBuilder::new(name);
    let inputs = resolve_binary_broadcast(node, graph, &mut b)?;
    let output = b.add_elementwise(kernel, &inputs, node.output_shape());
    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: super::graph_input_ids(node, 2),
    })
}

pub(in crate::trace_compile) fn compile_binary_fn_elementwise(
    node: &TraceNode,
    graph: &ComputationGraph,
    name: &str,
    op: BinaryFnKind,
) -> Result<CompiledStep, TensorIRError> {
    let kernel = binary_fn_kernel(name, op);
    let mut b = TensorBlockBuilder::new(name);
    let inputs = resolve_binary_broadcast(node, graph, &mut b)?;
    let output = b.add_elementwise(kernel, &inputs, node.output_shape());
    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: super::graph_input_ids(node, 2),
    })
}
