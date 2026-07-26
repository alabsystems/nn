// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Constant folding graph pass for compiled trace graphs.
//!
//! Runs before elementwise chain fusion. Folds constant subgraphs and
//! simplifies identity patterns:
//!
//! - **Constant-constant folding:** `Constant(a) + Constant(b)` → `Constant(a+b)`
//! - **Constant unary folding:** `Exp(Constant(a))` → `Constant(exp(a))`
//! - **Identity simplification:** `x + Constant(0)` → `x`, `x * Constant(1)` → `x`
//! - **Zero simplification:** `x * Constant(0)` → `Constant(0)`
//!
//! All folded values are validated for finiteness (NaN/Inf are not folded).
//! Part of #3083, #1815.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, NodeId, TraceNode, TraceOp};

/// Apply constant folding to a computation graph.
///
/// Returns a new graph with constant subgraphs folded and identity
/// patterns simplified. The original graph is not modified.
pub(crate) fn constant_fold(graph: &ComputationGraph) -> ComputationGraph {
    let nodes = graph.nodes();
    let mut new_nodes: Vec<TraceNode> = Vec::with_capacity(nodes.len());
    // Scalar constant values discovered during the pass.
    let mut const_vals: HashMap<NodeId, f64> = HashMap::new();
    // Identity simplification: node X resolved to just forwarding node Y.
    let mut remap: HashMap<NodeId, NodeId> = HashMap::new();

    for node in nodes {
        let id = node.id();

        // Remap inputs.
        let remapped_inputs: Vec<NodeId> = node
            .inputs()
            .iter()
            .map(|&inp| follow_remap(&remap, inp))
            .collect();

        // Register original constants.
        match node.op() {
            TraceOp::Constant { value } => {
                const_vals.insert(id, *value);
                new_nodes.push(node.clone());
                continue;
            }
            TraceOp::ConstantWeight { weight }
                if weight.data().len() == 1 && !weight.is_placeholder() =>
            {
                const_vals.insert(id, f64::from(weight.data()[0]));
                new_nodes.push(node.clone());
                continue;
            }
            _ => {}
        }

        // Try constant-constant folding (all inputs are known constants).
        if let Some(folded) = try_fold(node.op(), &remapped_inputs, &const_vals) {
            if folded.is_finite() {
                const_vals.insert(id, folded);
                new_nodes.push(TraceNode::new(
                    id,
                    node.name().to_string(),
                    TraceOp::Constant { value: folded },
                    vec![],
                    node.output_shape().to_vec(),
                    node.output_dtype(),
                ));
                continue;
            }
        }

        // Try identity simplification (x + 0 → x, x * 1 → x, x * 0 → 0).
        if let Some(simplified) = try_simplify(node.op(), &remapped_inputs, &const_vals) {
            match simplified {
                Simplified::Forward(target_id) => {
                    remap.insert(id, target_id);
                    // Emit as Reshape (passthrough) to preserve graph structure.
                    new_nodes.push(TraceNode::new(
                        id,
                        node.name().to_string(),
                        TraceOp::Reshape {
                            target_shape: node.output_shape().to_vec(),
                        },
                        vec![target_id],
                        node.output_shape().to_vec(),
                        node.output_dtype(),
                    ));
                    continue;
                }
                Simplified::Constant(value) => {
                    if value.is_finite() {
                        const_vals.insert(id, value);
                        new_nodes.push(TraceNode::new(
                            id,
                            node.name().to_string(),
                            TraceOp::Constant { value },
                            vec![],
                            node.output_shape().to_vec(),
                            node.output_dtype(),
                        ));
                        continue;
                    }
                }
            }
        }

        // No folding — emit with remapped inputs.
        new_nodes.push(TraceNode::new(
            id,
            node.name().to_string(),
            node.op().clone(),
            remapped_inputs,
            node.output_shape().to_vec(),
            node.output_dtype(),
        ));
    }

    // Preserve output nodes, following any remaps.
    let original_outputs: Vec<NodeId> = graph.output_nodes().iter().map(|n| n.id()).collect();
    let mut new_graph = ComputationGraph::from_nodes(new_nodes);
    for &out_id in &original_outputs {
        let remapped = follow_remap(&remap, out_id);
        let _ = new_graph.mark_output(remapped);
    }
    new_graph
}

/// Follow the remap chain to the final target.
fn follow_remap(remap: &HashMap<NodeId, NodeId>, mut id: NodeId) -> NodeId {
    // Limit iterations to prevent cycles (defensive).
    for _ in 0..64 {
        match remap.get(&id) {
            Some(&target) if target != id => id = target,
            _ => break,
        }
    }
    id
}

/// Try to fold a node whose inputs are all scalar constants.
///
/// Returns the folded value, or `None` if the op is not foldable or
/// inputs are not all constants.
fn try_fold(op: &TraceOp, inputs: &[NodeId], const_vals: &HashMap<NodeId, f64>) -> Option<f64> {
    match op {
        // Unary ops with 1 constant input.
        TraceOp::Exp => unary_const(inputs, const_vals, f64::exp),
        TraceOp::Log => unary_const(inputs, const_vals, f64::ln),
        TraceOp::Sqrt => unary_const(inputs, const_vals, f64::sqrt),
        TraceOp::Sqr => unary_const(inputs, const_vals, |x| x * x),
        TraceOp::Abs => unary_const(inputs, const_vals, f64::abs),
        TraceOp::Neg => unary_const(inputs, const_vals, |x| -x),
        TraceOp::Recip => unary_const(inputs, const_vals, |x| 1.0 / x),
        TraceOp::Sin => unary_const(inputs, const_vals, f64::sin),
        TraceOp::Cos => unary_const(inputs, const_vals, f64::cos),
        TraceOp::Tanh => unary_const(inputs, const_vals, f64::tanh),
        TraceOp::Floor => unary_const(inputs, const_vals, f64::floor),
        TraceOp::Round => unary_const(inputs, const_vals, f64::round),
        TraceOp::Fract => unary_const(inputs, const_vals, |x| x - x.floor()),
        TraceOp::Relu => unary_const(inputs, const_vals, |x| x.max(0.0)),
        TraceOp::Sigmoid => unary_const(inputs, const_vals, |x| 1.0 / (1.0 + (-x).exp())),

        // Binary ops with 2 constant inputs.
        TraceOp::Add => binary_const(inputs, const_vals, |a, b| a + b),
        TraceOp::Sub => binary_const(inputs, const_vals, |a, b| a - b),
        TraceOp::Mul => binary_const(inputs, const_vals, |a, b| a * b),
        TraceOp::Div => binary_const(inputs, const_vals, |a, b| a / b),
        TraceOp::Maximum => binary_const(inputs, const_vals, f64::max),
        TraceOp::Minimum => binary_const(inputs, const_vals, f64::min),

        _ => None,
    }
}

fn unary_const(
    inputs: &[NodeId],
    const_vals: &HashMap<NodeId, f64>,
    f: impl Fn(f64) -> f64,
) -> Option<f64> {
    let &a = const_vals.get(inputs.first()?)?;
    Some(f(a))
}

fn binary_const(
    inputs: &[NodeId],
    const_vals: &HashMap<NodeId, f64>,
    f: impl Fn(f64, f64) -> f64,
) -> Option<f64> {
    if inputs.len() < 2 {
        return None;
    }
    let &a = const_vals.get(&inputs[0])?;
    let &b = const_vals.get(&inputs[1])?;
    Some(f(a, b))
}

/// Result of identity simplification.
enum Simplified {
    /// Replace with a forward to the given node.
    Forward(NodeId),
    /// Replace with a constant value.
    Constant(f64),
}

/// Try to simplify identity patterns.
///
/// Handles: `x + 0 → x`, `0 + x → x`, `x - 0 → x`,
///          `x * 1 → x`, `1 * x → x`, `x * 0 → 0`, `0 * x → 0`,
///          `x / 1 → x`.
fn try_simplify(
    op: &TraceOp,
    inputs: &[NodeId],
    const_vals: &HashMap<NodeId, f64>,
) -> Option<Simplified> {
    if inputs.len() < 2 {
        return None;
    }
    let lhs_const = const_vals.get(&inputs[0]).copied();
    let rhs_const = const_vals.get(&inputs[1]).copied();

    match op {
        TraceOp::Add => {
            // x + 0 → x
            if rhs_const == Some(0.0) {
                return Some(Simplified::Forward(inputs[0]));
            }
            // 0 + x → x
            if lhs_const == Some(0.0) {
                return Some(Simplified::Forward(inputs[1]));
            }
            None
        }
        TraceOp::Sub => {
            // x - 0 → x
            if rhs_const == Some(0.0) {
                return Some(Simplified::Forward(inputs[0]));
            }
            None
        }
        TraceOp::Mul => {
            // x * 1 → x
            if rhs_const == Some(1.0) {
                return Some(Simplified::Forward(inputs[0]));
            }
            // 1 * x → x
            if lhs_const == Some(1.0) {
                return Some(Simplified::Forward(inputs[1]));
            }
            // x * 0 → 0
            if rhs_const == Some(0.0) {
                return Some(Simplified::Constant(0.0));
            }
            // 0 * x → 0
            if lhs_const == Some(0.0) {
                return Some(Simplified::Constant(0.0));
            }
            None
        }
        TraceOp::Div => {
            // x / 1 → x
            if rhs_const == Some(1.0) {
                return Some(Simplified::Forward(inputs[0]));
            }
            None
        }
        _ => None,
    }
}

#[cfg(test)]
#[path = "trace_compile_constant_fold_tests.rs"]
mod tests;
