// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Selection/indexing op compilation: IndexSelect, Gather.
//!
//! These ops use data-dependent addressing and require dedicated DispatchStep
//! variants — they cannot be decomposed into element-wise primitives.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode};

use crate::tensor_ir::TensorIRError;

use super::super::CompiledStep;
use super::build_single_op;

/// Compile `IndexSelect { dim }`: gather slices along `dim` using 1-D indices.
///
/// Input 0: data tensor `[S0, ..., S_dim, ..., S_n]`.
/// Input 1: 1-D index tensor `[K]`.
/// Output: `[S0, ..., K, ..., S_n]`.
pub(super) fn compile_index_select(
    node: &TraceNode,
    graph: &ComputationGraph,
    dim: usize,
) -> Result<CompiledStep, TensorIRError> {
    build_single_op("index_select", node, graph, 2, |b, inputs| {
        b.add_index_select(inputs[0], inputs[1], dim, node.output_shape())
    })
}

/// Compile `Gather { dim }`: N-D index gather along `dim`.
///
/// Input 0: data tensor.
/// Input 1: index tensor (same rank as data).
/// Output shape matches index tensor shape.
pub(super) fn compile_gather(
    node: &TraceNode,
    graph: &ComputationGraph,
    dim: usize,
) -> Result<CompiledStep, TensorIRError> {
    build_single_op("gather", node, graph, 2, |b, inputs| {
        b.add_gather(inputs[0], inputs[1], dim, node.output_shape())
    })
}
