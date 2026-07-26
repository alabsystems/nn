// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! DOT/Graphviz export for autodiff computation graphs.
//!
//! Converts the implicit computation graph (rooted at an `Arc<TrackedTensor>`)
//! into DOT format for visualization with Graphviz. Useful for debugging
//! gradient flow and understanding model structure.
//!
//! # Usage
//!
//! ```no_run
//! use nn_autodiff::{Var, TrackedTensor, graph_viz};
//! use nn_core::{DType, Device};
//! use std::sync::Arc;
//!
//! let x = Var::new(nn_core::DynTensor::from_vec(vec![2.0], &[1], &Device::Cpu).unwrap());
//! let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
//! let y = t.sqr().unwrap();
//! let dot = graph_viz::graph_to_dot(&y);
//! println!("{dot}");
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::path::Path;
use std::sync::Arc;

use crate::op::Op;
use crate::tracked::TrackedTensor;

/// Node classification for color coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeKind {
    /// Trainable variable (leaf with gradients).
    Input,
    /// Operation node (intermediate computation).
    Operation,
    /// The root/output node of the graph.
    Output,
    /// Constant leaf (no gradients flow through).
    NoGrad,
}

/// Collected information about a single node in the graph.
struct NodeInfo {
    kind: NodeKind,
    op_name: String,
    shape: Vec<usize>,
    is_var: bool,
}

/// Export the computation graph rooted at `root` as a DOT graph string.
///
/// Each node shows: operation name, output shape, and whether gradients are required.
/// Edges show data flow (forward direction: parent -> child).
///
/// Color coding:
/// - Green: trainable variable inputs
/// - Blue: operation nodes
/// - Red: output (root) node
/// - Gray: constant (no-grad) nodes
pub fn graph_to_dot(root: &Arc<TrackedTensor>) -> String {
    let nodes = collect_nodes(root);
    let node_map = build_node_map(&nodes, root);
    render_dot(&nodes, &node_map, root, false)
}

/// Export a simplified DOT graph showing only operation names (no shape info).
///
/// Produces a compact graph suitable for quick inspection of the computation
/// structure without shape/grad details.
pub fn graph_to_dot_minimal(root: &Arc<TrackedTensor>) -> String {
    let nodes = collect_nodes(root);
    let node_map = build_node_map(&nodes, root);
    render_dot(&nodes, &node_map, root, true)
}

/// Write the DOT graph to a file.
///
/// The output can be rendered with Graphviz: `dot -Tpng graph.dot -o graph.png`
pub fn write_dot_file(root: &Arc<TrackedTensor>, path: impl AsRef<Path>) -> std::io::Result<()> {
    let dot = graph_to_dot(root);
    std::fs::write(path, dot)
}

/// Collect all nodes in the graph via iterative DFS (post-order).
///
/// Same traversal strategy as `topological_sort` in `grad.rs`, but returns
/// nodes in DFS post-order for consistent DOT output.
fn collect_nodes(root: &Arc<TrackedTensor>) -> Vec<Arc<TrackedTensor>> {
    let mut visited = HashSet::new();
    let mut result = Vec::new();
    let mut stack: Vec<(Arc<TrackedTensor>, bool)> = vec![(Arc::clone(root), false)];

    while let Some((node, children_pushed)) = stack.pop() {
        let id = node.node_id().as_u64();

        if children_pushed {
            result.push(node);
            continue;
        }

        if !visited.insert(id) {
            continue;
        }

        stack.push((Arc::clone(&node), true));

        if let Some(op) = node.op() {
            let inputs = op_inputs(op);
            for input in inputs.into_iter().rev() {
                if !visited.contains(&input.node_id().as_u64()) {
                    stack.push((input, false));
                }
            }
        }
    }

    result
}

/// Build a map from NodeId to NodeInfo for all collected nodes.
fn build_node_map(
    nodes: &[Arc<TrackedTensor>],
    root: &Arc<TrackedTensor>,
) -> HashMap<u64, NodeInfo> {
    let root_id = root.node_id().as_u64();

    nodes
        .iter()
        .map(|node| {
            let id = node.node_id().as_u64();
            let kind = if id == root_id {
                NodeKind::Output
            } else if node.is_var() {
                NodeKind::Input
            } else if node.op().is_none() {
                NodeKind::NoGrad
            } else {
                NodeKind::Operation
            };

            let op_name = match node.op() {
                Some(op) => format!("{op:?}"),
                None if node.is_var() => "Var".to_string(),
                None => "Const".to_string(),
            };

            let info = NodeInfo {
                kind,
                op_name,
                shape: node.dims().to_vec(),
                is_var: node.is_var(),
            };

            (id, info)
        })
        .collect()
}

/// Render the DOT string from collected node information.
fn render_dot(
    nodes: &[Arc<TrackedTensor>],
    node_map: &HashMap<u64, NodeInfo>,
    root: &Arc<TrackedTensor>,
    minimal: bool,
) -> String {
    let mut dot = String::with_capacity(1024);
    let _ = writeln!(dot, "digraph computation_graph {{");
    let _ = writeln!(dot, "    rankdir=TB;");
    let _ = writeln!(
        dot,
        "    node [shape=box, style=filled, fontname=\"Helvetica\"];"
    );
    let _ = writeln!(dot);

    // Emit nodes
    for node in nodes {
        let id = node.node_id().as_u64();
        let info = match node_map.get(&id) {
            Some(i) => i,
            None => continue,
        };

        let color = match info.kind {
            NodeKind::Input => "#90EE90",     // light green
            NodeKind::Operation => "#87CEEB", // light blue
            NodeKind::Output => "#FFB6C1",    // light red/pink
            NodeKind::NoGrad => "#D3D3D3",    // light gray
        };

        let label = if minimal {
            info.op_name.clone()
        } else {
            let shape_str = format!("{:?}", info.shape);
            let grad_str = if info.is_var {
                "grad=yes"
            } else if info.kind == NodeKind::NoGrad {
                "grad=no"
            } else {
                "grad=flow"
            };
            format!("{}\\n{}\\n{}", info.op_name, shape_str, grad_str)
        };

        let _ = writeln!(dot, "    n{id} [label=\"{label}\", fillcolor=\"{color}\"];");
    }

    let _ = writeln!(dot);

    // Emit edges (forward data flow: parent -> child)
    let root_id = root.node_id().as_u64();
    let visited_ids: HashSet<u64> = nodes.iter().map(|n| n.node_id().as_u64()).collect();

    for node in nodes {
        let child_id = node.node_id().as_u64();
        if let Some(op) = node.op() {
            let inputs = op_inputs(op);
            for parent in &inputs {
                let parent_id = parent.node_id().as_u64();
                // Only emit edges for nodes in our collected set.
                if visited_ids.contains(&parent_id) {
                    // Forward edge (data flow)
                    let _ = writeln!(dot, "    n{parent_id} -> n{child_id};");
                }
            }
        }
    }

    // Add backward gradient flow edges as dashed (from output toward inputs)
    let _ = writeln!(dot);
    let _ = writeln!(dot, "    // Backward gradient flow (dashed)");
    let _ = writeln!(
        dot,
        "    edge [style=dashed, color=\"#CC0000\", constraint=false];"
    );

    for node in nodes {
        let child_id = node.node_id().as_u64();
        if let Some(op) = node.op() {
            let inputs = op_inputs(op);
            for parent in &inputs {
                let parent_id = parent.node_id().as_u64();
                if visited_ids.contains(&parent_id) {
                    // Backward edge (gradient flow): child -> parent
                    let _ = writeln!(dot, "    n{child_id} -> n{parent_id};");
                }
            }
        }
    }

    // Use `root_id` in a comment to silence unused-variable lint.
    let _ = writeln!(dot, "    // root_node={root_id}");
    let _ = writeln!(dot, "}}");
    dot
}

/// Extract input nodes from an Op (mirrors `grad_op_inputs.rs`).
///
/// This is a local copy to avoid exposing the internal `op_inputs` from `grad`.
fn op_inputs(op: &Op) -> Vec<Arc<TrackedTensor>> {
    match op {
        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) | Op::MatMul(a, b) => {
            vec![Arc::clone(a), Arc::clone(b)]
        }
        Op::Relu(x)
        | Op::Gelu(x)
        | Op::GeluErf(x)
        | Op::Silu(x)
        | Op::Tanh(x)
        | Op::Sigmoid(x)
        | Op::Exp(x)
        | Op::Log(x)
        | Op::Sqrt(x)
        | Op::Sqr(x)
        | Op::Neg(x)
        | Op::Abs(x)
        | Op::Sin(x)
        | Op::Cos(x)
        | Op::Recip(x)
        | Op::Powf(x, _)
        | Op::Clamp(x, _, _)
        | Op::Elu(x, _)
        | Op::HardSigmoid(x)
        | Op::HardSwish(x)
        | Op::Mish(x)
        | Op::Selu(x)
        | Op::Softplus(x)
        | Op::Celu(x, _) => vec![Arc::clone(x)],

        Op::SumKeepDim(x, _)
        | Op::MeanKeepDim(x, _)
        | Op::Reshape(x, _)
        | Op::Transpose(x, _, _)
        | Op::Narrow(x, _, _, _)
        | Op::Broadcast(x, _)
        | Op::Unsqueeze(x, _)
        | Op::Squeeze(x, _)
        | Op::Unfold(x, _, _, _)
        | Op::Permute(x, _)
        | Op::Softmax(x, _)
        | Op::LogSoftmax(x, _) => vec![Arc::clone(x)],

        Op::Maximum(a, b) | Op::Minimum(a, b) => vec![Arc::clone(a), Arc::clone(b)],

        Op::Conv1d { input, kernel, .. }
        | Op::Conv2d { input, kernel, .. }
        | Op::ConvTranspose1d { input, kernel, .. } => {
            vec![Arc::clone(input), Arc::clone(kernel)]
        }

        Op::Cat(inputs, _) | Op::Stack(inputs, _) => inputs.iter().map(Arc::clone).collect(),

        Op::LayerNorm {
            input,
            weight,
            bias,
            ..
        }
        | Op::GroupNorm {
            input,
            weight,
            bias,
            ..
        }
        | Op::BatchNorm {
            input,
            weight,
            bias,
            ..
        }
        | Op::InstanceNorm {
            input,
            weight,
            bias,
            ..
        } => vec![Arc::clone(input), Arc::clone(weight), Arc::clone(bias)],
        Op::RmsNorm { input, weight, .. } => {
            vec![Arc::clone(input), Arc::clone(weight)]
        }
        Op::Embedding(weight, indices) => vec![Arc::clone(weight), Arc::clone(indices)],
        Op::CrossEntropyLoss(input, targets, _)
        | Op::MseLoss(input, targets)
        | Op::L1Loss(input, targets)
        | Op::HuberLoss(input, targets, _) => {
            vec![Arc::clone(input), Arc::clone(targets)]
        }
        Op::MulScalar(x, _) | Op::AddScalar(x, _) => vec![Arc::clone(x)],
        Op::Dropout(x, mask, _) => vec![Arc::clone(x), Arc::clone(mask)],
        Op::MaxPool1d { input, .. }
        | Op::MaxPool2d { input, .. }
        | Op::AdaptiveAvgPool2d { input, .. }
        | Op::AvgPool2d { input, .. } => vec![Arc::clone(input)],
    }
}

/// Count the number of nodes in the computation graph rooted at `root`.
///
/// Useful for verifying expected graph structure in tests.
pub fn node_count(root: &Arc<TrackedTensor>) -> usize {
    collect_nodes(root).len()
}

/// Count the number of edges (forward data flow) in the computation graph.
pub fn edge_count(root: &Arc<TrackedTensor>) -> usize {
    let nodes = collect_nodes(root);
    let visited_ids: HashSet<u64> = nodes.iter().map(|n| n.node_id().as_u64()).collect();
    let mut count = 0;
    for node in &nodes {
        if let Some(op) = node.op() {
            let inputs = op_inputs(op);
            for parent in &inputs {
                if visited_ids.contains(&parent.node_id().as_u64()) {
                    count += 1;
                }
            }
        }
    }
    count
}

#[cfg(test)]
#[path = "graph_viz_tests.rs"]
mod tests;
