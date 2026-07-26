// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Graph-analysis predicates for `trace_to_graph` translation.
//!
//! Extracted from `trace_to_graph.rs` for the 450-line limit.

use std::collections::HashSet;

use nn_core::dyn_tensor::trace::{ComputationGraph, NodeId, TraceNode, TraceOp};

/// Check whether a reachable Input node is a "variable input" (data that
/// flows through the model) vs a "weight input" (parameter consumed only by
/// composite ops that embed their own weights).
///
/// Composite ops (Conv1d, Conv2d, Linear, LSTM, etc.) carry weight tensors
/// inside their `TraceOp` variant. The trace records ALL input tensor IDs
/// (data + weight), but the translator (`ny_trace_bridge::translate`) only
/// references `inputs[0]` (data) and takes weights from the variant. So
/// `inputs[1..]` are "phantom" inputs whose translated output tensors are
/// never referenced.
///
/// An Input node is "variable" if ANY reachable consumer uses it as:
/// - `inputs[0]` of a composite op (the data input), or
/// - ANY input position of a non-composite op (all inputs are variable).
pub(super) fn is_variable_input(
    graph: &ComputationGraph,
    input_id: NodeId,
    reachable: &HashSet<NodeId>,
) -> bool {
    for node in graph.nodes() {
        if !reachable.contains(&node.id()) || node.id() == input_id {
            continue;
        }
        let inputs = node.inputs();
        if !inputs.contains(&input_id) {
            continue;
        }
        // This node consumes our Input node.
        if is_composite_op(node.op()) {
            // Composite op: only inputs[0] is the data input.
            if inputs.first() == Some(&input_id) {
                return true;
            }
            // inputs[1..] are weight params embedded in TraceOp → not variable
        } else {
            // Non-composite op: all input positions are variable.
            return true;
        }
    }
    // No consumer makes this a variable input — it's weight-only.
    false
}

/// Returns true if the op is a composite op that embeds weight tensors in its
/// `TraceOp` variant. For these ops, only `inputs[0]` is the data input;
/// `inputs[1..]` are weight/parameter edges whose translated output tensors
/// are not referenced in the LayerSpec.
pub(super) fn is_composite_op(op: &TraceOp) -> bool {
    matches!(
        op,
        TraceOp::Conv1d { .. }
            | TraceOp::Conv2d { .. }
            | TraceOp::Conv3d { .. }
            | TraceOp::ConvTranspose1d { .. }
            | TraceOp::ConvTranspose2d { .. }
            | TraceOp::Linear { .. }
            | TraceOp::QLinear { .. }
            | TraceOp::Embedding { .. }
            | TraceOp::Lstm { .. }
            | TraceOp::BatchNorm { .. }
            | TraceOp::LayerNorm { .. }
            | TraceOp::RmsNorm { .. }
            | TraceOp::InstanceNorm { .. }
            | TraceOp::GroupNorm { .. }
    )
}

/// Compute the set of node IDs reachable from the output node.
///
/// Walks backward from the output (or last node) through input edges using BFS.
/// Nodes not reachable from the output are primitive ops shadowed by composite
/// ops (e.g., MatMul/Add nodes emitted by Linear::forward() before the composite
/// TraceOp::Linear node is recorded).
pub(super) fn reachable_nodes(graph: &ComputationGraph) -> HashSet<NodeId> {
    let mut reachable = HashSet::new();
    let output_id = graph
        .output_node()
        .map(TraceNode::id)
        .or_else(|| graph.nodes().last().map(TraceNode::id));
    let Some(start) = output_id else {
        return reachable;
    };

    let mut queue = std::collections::VecDeque::new();
    queue.push_back(start);
    reachable.insert(start);

    while let Some(id) = queue.pop_front() {
        if let Some(node) = graph.node(id) {
            for &input_id in node.inputs() {
                if reachable.insert(input_id) {
                    queue.push_back(input_id);
                }
            }
        }
    }

    reachable
}
