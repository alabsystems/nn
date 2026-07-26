// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shape-op dispatch: narrow, cat, transpose, permute, expand, cumsum,
//! repeat-interleave, where-cond, flip, clamp.
//!
//! Part of the category-dispatch refactor (#2305). Workers adding a new
//! shape op only touch this file, not the shared `compile_node` hub.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};

use crate::tensor_ir::TensorIRError;

use super::super::trace_compile_misc::{
    compile_cat, compile_clamp, compile_constant_pad_nd, compile_cumsum, compile_expand,
    compile_flip, compile_permute, compile_reflection_pad1d, compile_repeat_interleave,
    compile_transpose, compile_where_cond,
};
use super::super::trace_compile_ops::compile_narrow;
use super::super::CompiledStep;

/// Try to compile a shape trace op. Returns `None` for non-shape ops.
pub(in crate::trace_compile) fn try_compile(
    node: &TraceNode,
    graph: &ComputationGraph,
) -> Option<Result<CompiledStep, TensorIRError>> {
    match node.op() {
        TraceOp::Narrow { dim, start, length } => {
            Some(compile_narrow(node, graph, *dim, *start, *length))
        }
        TraceOp::Cat { dim, num_inputs: _ } => Some(compile_cat(node, graph, *dim)),
        TraceOp::Transpose { dim0, dim1 } => Some(compile_transpose(node, graph, *dim0, *dim1)),
        TraceOp::Permute { axes } => Some(compile_permute(node, graph, axes)),
        TraceOp::Expand { target_shape } => Some(compile_expand(node, graph, target_shape)),
        TraceOp::Cumsum { dim } => Some(compile_cumsum(node, graph, *dim)),
        TraceOp::RepeatInterleave { dim } => Some(compile_repeat_interleave(node, graph, *dim)),
        TraceOp::WhereCond => Some(compile_where_cond(node, graph)),
        TraceOp::Flip { dim } => Some(compile_flip(node, graph, *dim)),
        TraceOp::Clamp { min, max } => Some(compile_clamp(node, graph, min, max)),
        TraceOp::ReflectionPad1d {
            pad_left,
            pad_right,
        } => Some(compile_reflection_pad1d(node, graph, *pad_left, *pad_right)),
        TraceOp::ConstantPadNd { padding, value } => {
            Some(compile_constant_pad_nd(node, graph, padding, *value))
        }
        _ => None,
    }
}
