// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Subgraph extraction from traced computation graphs for ay SMT verification.
//!
//! Extracts small, self-contained subgraphs (3-10 layers) from a full model
//! `ComputationGraph`. Designed for ay complete (exhaustive) verification of
//! small subnetworks that are intractable for NY CROWN propagation
//! but small enough for SMT solving.
//!
//! # Target use case: Kokoro iSTFT
//!
//! The Kokoro iSTFT polar-to-cartesian conversion is a ~5-layer subgraph:
//! `(magnitude, phase) -> cos/sin -> mul -> [real, imag]`. This is fully
//! linear after the trig functions and amenable to ay QF_LRA encoding.
//!
//! # Design
//!
//! Extraction operates on `ComputationGraph` (nn-core), not `GraphNetwork`
//! (NY). This keeps the module independent of the NY feature
//! gate. The extracted subgraph is a valid `ComputationGraph` that can be:
//!
//! - Fed to `trace_to_graph_model()` for NY verification
//! - Analyzed for ay encoding feasibility (layer count, op types)
//! - Used as input to ay proof builders
//!
//! Part of #2455, #3351 (T3.6).

use std::collections::{HashMap, HashSet};

use nn_core::dyn_tensor::trace::{ComputationGraph, NodeId, TraceNode, TraceOp};
use nn_core::DType;

use crate::error::VerifyError;

/// Result of subgraph extraction.
#[derive(Debug)]
pub struct ExtractedSubgraph {
    /// The extracted subgraph as a self-contained `ComputationGraph`.
    ///
    /// External dependencies (nodes outside the extraction range that are
    /// referenced by nodes inside) are replaced with synthetic `Input` nodes.
    /// Weight references (`ConstantWeight`) within the range are preserved.
    pub graph: ComputationGraph,

    /// Number of synthetic input nodes created to replace external dependencies.
    ///
    /// Zero means the subgraph was already self-contained (all dependencies
    /// are `Input` or `ConstantWeight` nodes within the range).
    pub synthetic_input_count: usize,

    /// Number of layers in the extracted subgraph (excluding synthetic inputs).
    pub layer_count: usize,

    /// Node ID mapping: original graph node ID -> subgraph node ID.
    ///
    /// Includes both extracted nodes and synthetic input nodes.
    pub id_map: HashMap<NodeId, NodeId>,
}

/// Specification for which nodes to extract.
#[derive(Debug, Clone)]
pub enum SubgraphSpec {
    /// Extract nodes by index range (inclusive start, exclusive end).
    ///
    /// Indices refer to positions in `ComputationGraph::nodes()`.
    IndexRange { start: usize, end: usize },

    /// Extract nodes whose names contain any of the given substrings.
    ///
    /// All matching nodes plus their transitive dependencies (within the
    /// matched set) are included. External dependencies become synthetic inputs.
    NameContains { patterns: Vec<String> },

    /// Extract a specific set of node IDs.
    NodeIds { ids: Vec<NodeId> },
}

/// Extract a subgraph from a traced computation graph.
///
/// The extracted subgraph is self-contained: every node's inputs either
/// reference other nodes in the subgraph or are replaced with synthetic
/// `Input` nodes. This makes the subgraph suitable for independent
/// verification via ay or NY.
///
/// # Arguments
///
/// * `graph` - The full computation graph from tracing.
/// * `spec` - Which nodes to extract (by index range, name pattern, or IDs).
///
/// # Errors
///
/// Returns `VerifyError` if:
/// - The index range is out of bounds
/// - No nodes match the specification
/// - The resulting subgraph would be empty
///
/// # Example
///
/// ```rust,ignore
/// use nn_verify::subgraph_extract::{extract_subgraph, SubgraphSpec};
///
/// // Extract layers 10-15 from a traced model graph
/// let result = extract_subgraph(&graph, &SubgraphSpec::IndexRange {
///     start: 10,
///     end: 15,
/// }).expect("extraction");
/// assert!(result.layer_count <= 5);
/// ```
pub fn extract_subgraph(
    graph: &ComputationGraph,
    spec: &SubgraphSpec,
) -> Result<ExtractedSubgraph, VerifyError> {
    let nodes = graph.nodes();

    // Resolve spec to a set of node indices.
    let selected_indices = resolve_spec(nodes, spec)?;
    if selected_indices.is_empty() {
        return Err(VerifyError::EmptyGraph);
    }

    // Collect the selected node IDs for fast lookup.
    let selected_ids: HashSet<NodeId> = selected_indices
        .iter()
        .map(|&idx| nodes[idx].id())
        .collect();

    // Find external dependencies: input node IDs that are NOT in our selected set.
    // These need synthetic Input nodes.
    let mut external_deps: Vec<NodeId> = Vec::new();
    let mut seen_external: HashSet<NodeId> = HashSet::new();
    for &idx in &selected_indices {
        let node = &nodes[idx];
        for &input_id in node.inputs() {
            if !selected_ids.contains(&input_id) && seen_external.insert(input_id) {
                external_deps.push(input_id);
            }
        }
    }

    // Build ID remapping: assign new sequential IDs.
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut next_id: NodeId = 1;

    // First: synthetic input nodes for external dependencies.
    for &ext_id in &external_deps {
        id_map.insert(ext_id, next_id);
        next_id += 1;
    }

    // Then: selected nodes in their original order.
    for &idx in &selected_indices {
        let node_id = nodes[idx].id();
        id_map.insert(node_id, next_id);
        next_id += 1;
    }

    // Build the subgraph nodes.
    let mut sub_nodes: Vec<TraceNode> =
        Vec::with_capacity(external_deps.len() + selected_indices.len());

    // Create synthetic Input nodes for external dependencies.
    // Use the original node's output shape and dtype so downstream ops are valid.
    for &ext_id in &external_deps {
        let new_id = id_map[&ext_id];
        let (shape, dtype) = if let Some(orig_node) = graph.node(ext_id) {
            (orig_node.output_shape().to_vec(), orig_node.output_dtype())
        } else {
            // Node not found in graph -- use a scalar f32 placeholder.
            // This shouldn't happen for well-formed graphs.
            (vec![1], DType::F32)
        };
        sub_nodes.push(TraceNode::new(
            new_id,
            format!("subgraph_input_{ext_id}"),
            TraceOp::Input,
            vec![],
            shape,
            dtype,
        ));
    }

    // Copy selected nodes with remapped IDs.
    let layer_count = selected_indices.len();
    for &idx in &selected_indices {
        let orig = &nodes[idx];
        let new_id = id_map[&orig.id()];
        let new_inputs: Vec<NodeId> = orig
            .inputs()
            .iter()
            .map(|&input_id| {
                *id_map.get(&input_id).unwrap_or({
                    // This input references a node we don't have mapped.
                    // This shouldn't happen since we added all external deps above.
                    // Defensive: map to 0 (will fail topology validation).
                    &0
                })
            })
            .collect();
        sub_nodes.push(TraceNode::new(
            new_id,
            orig.name().to_string(),
            orig.op().clone(),
            new_inputs,
            orig.output_shape().to_vec(),
            orig.output_dtype(),
        ));
    }

    let sub_graph = ComputationGraph::from_nodes(sub_nodes);

    Ok(ExtractedSubgraph {
        graph: sub_graph,
        synthetic_input_count: external_deps.len(),
        layer_count,
        id_map,
    })
}

/// Validate that an extracted subgraph is self-contained.
///
/// A subgraph is self-contained if all node inputs are either:
/// - Other nodes within the subgraph, OR
/// - `Input` nodes (network inputs or synthetic inputs from extraction), OR
/// - `ConstantWeight` nodes (embedded weight tensors)
///
/// Returns `Ok(())` if valid, or a descriptive error if the subgraph has
/// dangling references.
pub fn validate_subgraph(graph: &ComputationGraph) -> Result<(), VerifyError> {
    if graph.is_empty() {
        return Err(VerifyError::EmptyGraph);
    }

    // Topology check: all input references resolve to earlier nodes.
    graph
        .validate_topology()
        .map_err(|e| VerifyError::InternalTranslationError {
            context: format!("subgraph topology validation failed: {e}"),
        })?;

    // Self-containment check: every node's inputs must be in the graph.
    let node_ids: HashSet<NodeId> = graph.nodes().iter().map(TraceNode::id).collect();
    for node in graph.nodes() {
        for &input_id in node.inputs() {
            if !node_ids.contains(&input_id) {
                return Err(VerifyError::InternalTranslationError {
                    context: format!(
                        "subgraph node '{}' (id={}) references input id={} which is not in the subgraph",
                        node.name(),
                        node.id(),
                        input_id,
                    ),
                });
            }
        }
    }

    Ok(())
}

/// Analyze a computation graph to find candidate subgraphs for ay verification.
///
/// Returns subgraph specs for contiguous regions of `min_layers..=max_layers`
/// that contain only ay-compatible operations (no LSTM, no attention, no
/// data-dependent control flow).
///
/// Each candidate includes entry/exit node analysis and an estimated SMT
/// variable count for ranking candidates by tractability.
///
/// This is a heuristic helper -- callers should inspect the returned specs
/// and select based on verification goals.
pub fn find_ay_candidates(
    graph: &ComputationGraph,
    min_layers: usize,
    max_layers: usize,
) -> Vec<AYCandidateRegion> {
    let nodes = graph.nodes();
    let mut candidates = Vec::new();

    // Sliding window: find contiguous regions of ay-compatible ops.
    let mut window_start = 0;
    while window_start < nodes.len() {
        // Skip non-compatible ops at the start.
        if !is_ay_compatible_op(nodes[window_start].op()) {
            window_start += 1;
            continue;
        }

        // Extend window as far as ay-compatible ops go.
        let mut window_end = window_start + 1;
        while window_end < nodes.len() && is_ay_compatible_op(nodes[window_end].op()) {
            window_end += 1;
        }

        let region_len = window_end - window_start;
        if region_len >= min_layers {
            // Emit candidate(s) for this contiguous compatible region.
            // If region is larger than max_layers, emit overlapping windows.
            let effective_max = max_layers.min(region_len);
            for size in min_layers..=effective_max {
                for start in window_start..=(window_end - size) {
                    let end = start + size;
                    let candidate = build_candidate_region(nodes, start, end);
                    candidates.push(candidate);
                }
            }
        }

        window_start = window_end;
    }

    candidates
}

/// Build a `AYCandidateRegion` for a contiguous slice `[start, end)` of nodes.
fn build_candidate_region(nodes: &[TraceNode], start: usize, end: usize) -> AYCandidateRegion {
    let size = end - start;

    // Collect op types and internal node IDs.
    let mut op_types = Vec::with_capacity(size);
    let mut internal_nodes = Vec::with_capacity(size);
    let region_nodes = &nodes[start..end];
    let internal_ids: HashSet<NodeId> = region_nodes.iter().map(TraceNode::id).collect();

    for node in region_nodes {
        op_types.push(trace_op_name(node.op()));
        internal_nodes.push(node.id());
    }

    // Entry nodes: external node IDs referenced by nodes inside the region.
    let mut entry_set: HashSet<NodeId> = HashSet::new();
    for node in region_nodes {
        for &input_id in node.inputs() {
            if !internal_ids.contains(&input_id) {
                entry_set.insert(input_id);
            }
        }
    }
    let entry_nodes: Vec<NodeId> = entry_set.into_iter().collect();

    // Exit nodes: internal nodes whose outputs are consumed outside the region.
    // Build a set of all node IDs referenced as inputs by nodes outside the region.
    let mut consumed_outside: HashSet<NodeId> = HashSet::new();
    for (i, node) in nodes.iter().enumerate() {
        if i < start || i >= end {
            for &input_id in node.inputs() {
                if internal_ids.contains(&input_id) {
                    consumed_outside.insert(input_id);
                }
            }
        }
    }
    // Also include the last node (always an implicit exit for verification).
    if end > start {
        consumed_outside.insert(nodes[end - 1].id());
    }
    let exit_nodes: Vec<NodeId> = consumed_outside.into_iter().collect();

    // Estimated complexity: sum per-op heuristic variable counts.
    let mut estimated_complexity: usize = 0;
    for node in region_nodes {
        estimated_complexity += estimate_op_complexity(node.op(), node.output_shape());
    }

    AYCandidateRegion {
        start_index: start,
        end_index: end,
        layer_count: size,
        op_types,
        entry_nodes,
        exit_nodes,
        internal_nodes,
        estimated_complexity,
    }
}

/// Heuristic SMT variable count for a single operation.
///
/// Element-wise ops contribute output element count. Parameterized ops
/// (Linear, Conv1d) contribute weight element count on top of output size.
fn estimate_op_complexity(op: &TraceOp, output_shape: &[usize]) -> usize {
    let output_elements: usize = output_shape.iter().product::<usize>().max(1);
    match op {
        TraceOp::Linear { weight, .. } => output_elements + weight.data().len(),
        TraceOp::Conv1d { weight, .. } => output_elements + weight.data().len(),
        TraceOp::LayerNorm { weight, bias, .. } => {
            output_elements + weight.data().len() + bias.data().len()
        }
        TraceOp::BatchNorm {
            weight,
            bias,
            running_mean,
            running_var,
            ..
        } => {
            output_elements
                + weight.data().len()
                + bias.data().len()
                + running_mean.data().len()
                + running_var.data().len()
        }
        TraceOp::Embedding { weight } => output_elements + weight.data().len(),
        TraceOp::MatMul => {
            // Bilinear: output + both input sizes (heuristic: 3x output).
            output_elements * 3
        }
        TraceOp::Softmax { .. } => {
            // exp + sum + div per element.
            output_elements * 3
        }
        _ => output_elements,
    }
}

/// A candidate region for ay SMT verification.
#[derive(Debug, Clone)]
pub struct AYCandidateRegion {
    /// Start index in `ComputationGraph::nodes()` (inclusive).
    pub start_index: usize,
    /// End index in `ComputationGraph::nodes()` (exclusive).
    pub end_index: usize,
    /// Number of layers in this candidate.
    pub layer_count: usize,
    /// Operation type names in order.
    pub op_types: Vec<String>,

    /// Node IDs at the entry boundary (inputs from outside the region).
    ///
    /// These are the original graph node IDs that nodes inside the candidate
    /// depend on but are themselves outside `[start_index, end_index)`.
    /// Empty if the region starts at the graph inputs.
    pub entry_nodes: Vec<NodeId>,

    /// Node IDs at the exit boundary (nodes inside the region whose outputs
    /// are consumed by nodes outside).
    ///
    /// For contiguous candidates this is typically the last node, but
    /// branching graphs may have multiple exit points.
    pub exit_nodes: Vec<NodeId>,

    /// Node IDs of all nodes internal to the region (between entry and exit).
    pub internal_nodes: Vec<NodeId>,

    /// Estimated SMT variable count for this region.
    ///
    /// Heuristic: sums per-op variable estimates. Element-wise ops contribute
    /// the product of their output shape dimensions. Linear/Conv1d contribute
    /// weight element count. Used to rank candidates by tractability.
    pub estimated_complexity: usize,
}

// -- Internal helpers --------------------------------------------------------

/// Resolve a `SubgraphSpec` to a sorted list of node indices.
fn resolve_spec(nodes: &[TraceNode], spec: &SubgraphSpec) -> Result<Vec<usize>, VerifyError> {
    match spec {
        SubgraphSpec::IndexRange { start, end } => {
            if *start >= nodes.len() || *end > nodes.len() || *start >= *end {
                return Err(VerifyError::InvalidInput(format!(
                    "index range [{start}, {end}) is out of bounds for graph with {} nodes",
                    nodes.len()
                )));
            }
            Ok((*start..*end).collect())
        }
        SubgraphSpec::NameContains { patterns } => {
            let indices: Vec<usize> = nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| {
                    patterns
                        .iter()
                        .any(|pat| node.name().contains(pat.as_str()))
                })
                .map(|(i, _)| i)
                .collect();
            if indices.is_empty() {
                return Err(VerifyError::InvalidInput(format!(
                    "no nodes matched name patterns: {patterns:?}"
                )));
            }
            Ok(indices)
        }
        SubgraphSpec::NodeIds { ids } => {
            let id_set: HashSet<NodeId> = ids.iter().copied().collect();
            let indices: Vec<usize> = nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| id_set.contains(&node.id()))
                .map(|(i, _)| i)
                .collect();
            if indices.is_empty() {
                return Err(VerifyError::InvalidInput(format!(
                    "no nodes matched IDs: {ids:?}"
                )));
            }
            Ok(indices)
        }
    }
}

/// Check if a `TraceOp` is compatible with ay SMT encoding.
///
/// ay handles element-wise arithmetic, constants, simple reductions,
/// and small parameterized layers (Linear, Conv1d with small kernels,
/// LayerNorm, BatchNorm, Embedding, Softmax, MatMul).
///
/// It does NOT handle: LSTM, multi-head attention, SDPA, data-dependent
/// control flow, or very large convolutions. Conv1d is only compatible
/// when `kernel_size * in_channels * out_channels <= 4096` (the SMT
/// variable count stays tractable).
///
/// # ay encoding notes
///
/// - **MatMul**: Encoded as bilinear product. Only tractable for small
///   matrices; callers should check `estimated_complexity` on candidates.
/// - **Linear**: Weight matrix encoded as constants. Tractable when
///   `in_features * out_features` is small.
/// - **Softmax**: Encoded as exp + sum + div. Exact for small dim.
/// - **LayerNorm/BatchNorm**: Encoded as mean/variance arithmetic.
/// - **Embedding**: Lookup is a conditional select over weight rows.
/// - **Conv1d**: Unrolled convolution. Small kernel constraint above.
pub fn is_ay_compatible_op(op: &TraceOp) -> bool {
    match op {
        // Conv1d: only tractable for small kernels.
        TraceOp::Conv1d { weight, groups, .. } => {
            // Weight shape is [out_channels, in_channels/groups, kernel_size].
            let weight_shape = weight.shape();
            if weight_shape.len() >= 3 {
                let out_ch = weight_shape[0];
                let in_ch_per_group = weight_shape[1];
                let kernel_size = weight_shape[2];
                // Total SMT variables for unrolled conv must be tractable.
                let complexity = out_ch * in_ch_per_group * kernel_size * groups;
                complexity <= 4096
            } else {
                false
            }
        }
        _ => matches!(
            op,
            TraceOp::Input
                | TraceOp::ConstantWeight { .. }
                | TraceOp::Constant { .. }
                | TraceOp::Add
                | TraceOp::Sub
                | TraceOp::Mul
                | TraceOp::Div
                | TraceOp::Maximum
                | TraceOp::Minimum
                | TraceOp::Neg
                | TraceOp::Abs
                | TraceOp::Exp
                | TraceOp::Log
                | TraceOp::Sqrt
                | TraceOp::Sqr
                | TraceOp::Recip
                | TraceOp::Sin
                | TraceOp::Cos
                | TraceOp::Tanh
                | TraceOp::Sigmoid
                | TraceOp::Relu
                | TraceOp::Gelu
                | TraceOp::GeluErf
                | TraceOp::Silu
                | TraceOp::MatMul
                | TraceOp::Linear { .. }
                | TraceOp::Softmax { .. }
                | TraceOp::LayerNorm { .. }
                | TraceOp::BatchNorm { .. }
                | TraceOp::Embedding { .. }
                | TraceOp::Clamp { .. }
                | TraceOp::Powf { .. }
                | TraceOp::Reshape { .. }
                | TraceOp::Transpose { .. }
                | TraceOp::Squeeze { .. }
                | TraceOp::Unsqueeze { .. }
                | TraceOp::Narrow { .. }
                | TraceOp::Cat { .. }
                | TraceOp::Flip { .. }
                | TraceOp::ReduceSum { .. }
                | TraceOp::ReduceMean { .. }
        ),
    }
}

/// Return a human-readable name for a `TraceOp` variant.
///
/// Extracts the variant name from `Debug` output, stripping field data.
/// Example: `Conv1d { weight: ..., ... }` -> `"Conv1d"`.
fn trace_op_name(op: &TraceOp) -> String {
    let debug = format!("{op:?}");
    debug
        .split(['{', '('])
        .next()
        .unwrap_or("Unknown")
        .trim()
        .to_string()
}

#[cfg(test)]
#[path = "subgraph_extract_tests.rs"]
mod tests;
