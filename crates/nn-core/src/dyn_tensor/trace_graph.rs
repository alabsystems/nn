// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `ComputationGraph` — captured computation graph from DynTensor tracing.
//!
//! Extracted from `trace.rs` to keep production files under 450 lines.

use std::collections::HashMap;

use crate::{Result, TensorError};

use super::{NodeId, TraceNode, TraceOp};

/// A captured computation graph from DynTensor tracing.
///
/// Contains all nodes in topological order (dependencies before dependents).
/// Can be converted to NY's `GraphNetwork` for formal verification.
#[derive(Debug, Clone)]
pub struct ComputationGraph {
    /// Nodes in topological (insertion) order.
    pub(super) nodes: Vec<TraceNode>,
    /// Map from node ID to index in `nodes`.
    pub(super) id_to_index: HashMap<NodeId, usize>,
    /// Output node IDs in the order they were marked.
    ///
    /// For single-output models (the common case), this has exactly one entry
    /// — the last node added during tracing. For multi-output models (e.g.,
    /// encoder-decoder, LSTM returning hidden+cell), callers use
    /// [`mark_output()`] to register additional outputs.
    pub(super) output_nodes: Vec<NodeId>,
}

impl ComputationGraph {
    /// Build a `ComputationGraph` from a pre-ordered slice of nodes.
    ///
    /// The last node is used as the output node (if any nodes exist).
    /// Callers must ensure nodes are in topological order.
    pub fn from_nodes(nodes: Vec<TraceNode>) -> Self {
        let id_to_index: HashMap<NodeId, usize> =
            nodes.iter().enumerate().map(|(i, n)| (n.id(), i)).collect();
        let output_nodes: Vec<NodeId> = nodes.last().map(TraceNode::id).into_iter().collect();
        Self {
            nodes,
            id_to_index,
            output_nodes,
        }
    }

    /// Returns all nodes in topological order.
    pub fn nodes(&self) -> &[TraceNode] {
        &self.nodes
    }

    /// Returns the primary output node (backward-compatible single-output API).
    ///
    /// For single-output graphs, this is the only output. For multi-output
    /// graphs, this returns the *last* marked output. Use [`output_nodes()`]
    /// to access all outputs.
    pub fn output_node(&self) -> Option<&TraceNode> {
        self.output_nodes
            .last()
            .and_then(|&id| self.id_to_index.get(&id))
            .map(|&idx| &self.nodes[idx])
    }

    /// Returns all output nodes in the order they were marked.
    ///
    /// For single-output graphs, this returns a slice of length 1.
    /// For multi-output graphs (encoder-decoder, LSTM hidden+cell),
    /// returns all explicitly marked outputs.
    pub fn output_nodes(&self) -> Vec<&TraceNode> {
        self.output_nodes
            .iter()
            .filter_map(|&id| self.id_to_index.get(&id).map(|&idx| &self.nodes[idx]))
            .collect()
    }

    /// Mark a node as an output of this graph.
    ///
    /// For multi-output models (encoder-decoder, LSTM), call this for each
    /// node that should be an output. The node must exist in the graph.
    /// Duplicate marks are ignored.
    ///
    /// Returns `true` if the node was found and added (or already present).
    /// Returns `false` if the node ID does not exist in the graph.
    #[must_use]
    pub fn mark_output(&mut self, id: NodeId) -> bool {
        if !self.id_to_index.contains_key(&id) {
            return false;
        }
        if !self.output_nodes.contains(&id) {
            self.output_nodes.push(id);
        }
        true
    }

    /// Replace the output list with a single primary output.
    ///
    /// Use when `trace_graph` records a default output (last traced op) but
    /// the caller needs a different node as the actual output.
    /// Returns `true` if the node was found, `false` if not in the graph.
    #[must_use]
    pub fn set_primary_output(&mut self, id: NodeId) -> bool {
        if !self.id_to_index.contains_key(&id) {
            return false;
        }
        self.output_nodes.clear();
        self.output_nodes.push(id);
        true
    }

    /// Returns a node by ID.
    pub fn node(&self, id: NodeId) -> Option<&TraceNode> {
        self.id_to_index.get(&id).map(|&idx| &self.nodes[idx])
    }

    /// Returns the number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns all input nodes (nodes with `TraceOp::Input`).
    pub fn input_nodes(&self) -> Vec<&TraceNode> {
        self.nodes
            .iter()
            .filter(|n| matches!(n.op(), TraceOp::Input))
            .collect()
    }

    /// Override the output shapes of nodes whose names appear in `overrides`.
    ///
    /// Used to update a traced graph with concrete shapes from a reference
    /// trace before compilation. The caller must ensure that overridden shapes
    /// are consistent (downstream nodes should also have updated shapes).
    ///
    /// Returns the number of nodes that were updated.
    pub fn override_node_shapes(&mut self, overrides: &HashMap<String, Vec<usize>>) -> usize {
        let mut count = 0;
        for node in &mut self.nodes {
            if let Some(new_shape) = overrides.get(&node.name) {
                node.output_shape = new_shape.clone();
                count += 1;
            }
        }
        count
    }

    /// Returns indices of nodes that are `SegmentBoundary` markers.
    pub fn segment_boundaries(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n.op(), TraceOp::SegmentBoundary { .. }))
            .map(|(i, _)| i)
            .collect()
    }

    /// Returns `true` if this graph contains at least one `SegmentBoundary`.
    pub fn has_segment_boundaries(&self) -> bool {
        self.nodes
            .iter()
            .any(|n| matches!(n.op(), TraceOp::SegmentBoundary { .. }))
    }

    /// Split this graph at `SegmentBoundary` markers into independent segments.
    ///
    /// Each segment is a self-contained `ComputationGraph` where the first
    /// segment starts with the original inputs and subsequent segments start
    /// with a synthetic `Input` node replacing the segment boundary output.
    ///
    /// Returns a `SegmentedGraph` with one segment per boundary plus one
    /// final segment. If there are no boundaries, returns the original graph
    /// as a single segment.
    pub fn split_at_segment_boundaries(&self) -> SegmentedGraph {
        let boundary_indices = self.segment_boundaries();
        if boundary_indices.is_empty() {
            return SegmentedGraph {
                segments: vec![GraphSegment {
                    graph: self.clone(),
                    boundary_reason: None,
                    boundary_bounds: None,
                }],
            };
        }

        let mut segments = Vec::new();
        let mut seg_start = 0;

        for &boundary_idx in &boundary_indices {
            // Segment: nodes [seg_start .. boundary_idx) (excludes the boundary itself)
            let seg_nodes: Vec<TraceNode> = self.nodes[seg_start..boundary_idx].to_vec();

            // Extract boundary metadata
            let (reason, bounds) = match self.nodes[boundary_idx].op() {
                TraceOp::SegmentBoundary {
                    reason,
                    input_bounds,
                } => (Some(reason.clone()), *input_bounds),
                _ => (None, None),
            };

            if !seg_nodes.is_empty() {
                let graph = Self::from_nodes(seg_nodes);
                segments.push(GraphSegment {
                    graph,
                    boundary_reason: reason,
                    boundary_bounds: bounds,
                });
            }

            // Next segment starts after the boundary node
            seg_start = boundary_idx + 1;
        }

        // Final segment: nodes after last boundary
        if seg_start < self.nodes.len() {
            let seg_nodes: Vec<TraceNode> = self.nodes[seg_start..].to_vec();
            if !seg_nodes.is_empty() {
                let graph = Self::from_nodes(seg_nodes);
                segments.push(GraphSegment {
                    graph,
                    boundary_reason: None,
                    boundary_bounds: None,
                });
            }
        }

        SegmentedGraph { segments }
    }

    /// Validate that nodes are in topological order: every node's inputs
    /// must reference nodes that appear earlier in the vector.
    ///
    /// Returns `Ok(())` if the graph is well-ordered, or
    /// `Err(TensorError::TopologyError)` with the node name, index, and
    /// missing input ID for the first out-of-order reference found.
    pub fn validate_topology(&self) -> Result<()> {
        use std::collections::HashSet;
        let mut seen: HashSet<NodeId> = HashSet::with_capacity(self.nodes.len());
        for (i, node) in self.nodes.iter().enumerate() {
            for &input_id in node.inputs() {
                if !seen.contains(&input_id) {
                    return Err(TensorError::TopologyError {
                        node_name: node.name().to_string(),
                        index: i,
                        missing_input: input_id,
                    });
                }
            }
            seen.insert(node.id());
        }
        Ok(())
    }
}

// -- Segment types for pipeline segmentation (#2378) --------------------------

/// A single segment of a computation graph split at `SegmentBoundary` markers.
///
/// Contains the sub-graph and metadata about why the split occurred.
#[derive(Debug, Clone)]
pub struct GraphSegment {
    /// The computation graph for this segment.
    pub graph: ComputationGraph,
    /// Reason for the preceding boundary (e.g., "length_regulate").
    /// `None` for the first segment (no preceding boundary) or the tail segment.
    pub boundary_reason: Option<String>,
    /// Optional (lower, upper) bounds hint from the preceding boundary.
    pub boundary_bounds: Option<(f32, f32)>,
}

/// A computation graph split at data-dependent operation boundaries.
///
/// Each segment can be independently translated to a NY `GraphNetwork`
/// and verified via IBP/CROWN. Output bounds from segment N feed as input
/// bounds to segment N+1.
#[derive(Debug, Clone)]
pub struct SegmentedGraph {
    /// Segments in order. The first segment starts with original inputs.
    /// Subsequent segments start with synthetic inputs at boundary points.
    pub segments: Vec<GraphSegment>,
}
