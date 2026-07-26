// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shape propagation through a `ComputationGraph`.
//!
//! Walks nodes topologically and recomputes output shapes from input shapes
//! using each `TraceOp`'s deterministic shape rules. This is used after
//! overriding input shapes (e.g., from a reference trace) to ensure all
//! intermediate node shapes are consistent with the new inputs.
//!
//! The per-op shape inference logic lives in `trace_shape_infer.rs` (extracted
//! to comply with the 500-line file limit).

use std::collections::HashMap;

use super::NodeId;
use crate::dyn_tensor::trace::ComputationGraph;

#[path = "trace_shape_infer.rs"]
mod shape_infer;

impl ComputationGraph {
    /// Propagate shapes through the graph from input nodes.
    ///
    /// After overriding input shapes (e.g., from a reference trace), call this
    /// to recompute all intermediate node shapes. Walks topologically and uses
    /// each op's deterministic shape rules.
    ///
    /// Returns the number of nodes whose shapes were updated.
    pub fn propagate_shapes(&mut self) -> usize {
        // Build a map of node_id -> shape as we walk.
        let mut shape_map: HashMap<NodeId, Vec<usize>> = HashMap::new();
        let mut updated = 0;

        // First pass: populate shape_map from existing node shapes.
        for node in &self.nodes {
            shape_map.insert(node.id(), node.output_shape.clone());
        }

        // Second pass: propagate shapes topologically.
        for i in 0..self.nodes.len() {
            let node = &self.nodes[i];
            let op = node.op.clone();
            let inputs = node.inputs.clone();
            let node_id = node.id();
            let old_shape = node.output_shape.clone();

            // Collect input shapes.
            let input_shapes: Vec<Vec<usize>> = inputs
                .iter()
                .filter_map(|id| shape_map.get(id).cloned())
                .collect();

            // Infer the new shape from the op and input shapes.
            let new_shape = shape_infer::infer_output_shape(&op, &input_shapes, &old_shape);

            if new_shape != old_shape {
                self.nodes[i].output_shape = new_shape.clone();
                updated += 1;
            }

            // Update shape_map with the (possibly new) shape.
            shape_map.insert(node_id, new_shape);
        }

        updated
    }
}
