// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-layer bound extraction from NY verification runs.
//!
//! Bridges NY's `collect_crown_ibp_bounds_dag_with_status()` API
//! to nn's [`LayerBoundRecord`] certificate type. This enables AC3:
//! per-layer IBP bound trace population in proof certificates.
//!
//! # Usage
//!
//! ```rust,no_run
//! use nn_verify::layer_bounds::extract_layer_bounds;
//! use nn_verify::verify::PropMethod;
//!
//! let records = extract_layer_bounds(&graph, &input_bounds)?;
//! // records is Vec<LayerBoundRecord> suitable for ProofCertificate.layer_bounds
//! ```

use ny_api::BoundedTensor;
use ny_propagate::{GraphNetwork, GraphNode, NETWORK_INPUT};

use crate::certificate_types::LayerBoundRecord;
use crate::error::VerifyError;
use crate::verify::PropMethod;

/// Extract per-layer bound records from a NY `GraphNetwork`.
///
/// Uses `collect_crown_ibp_bounds_dag_with_status()` to get per-node bounds
/// with provenance (CROWN vs IBP fallback), then converts to
/// `Vec<LayerBoundRecord>` in topological order.
///
/// The CROWN-IBP collection computes IBP bounds first, then tries to tighten
/// each node with backward CROWN propagation. Nodes where CROWN fails fall
/// back to IBP bounds. Provenance for each node indicates which method was used.
///
/// # Errors
///
/// Returns `VerifyError::Ny` if topological sort or bound collection fails.
pub fn extract_layer_bounds(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
) -> Result<Vec<LayerBoundRecord>, VerifyError> {
    // Get topological order for consistent layer indexing.
    let topo_order = graph.topological_sort()?;

    // Collect per-node CROWN-IBP bounds with provenance.
    let result = graph.collect_crown_ibp_bounds_dag_with_status(input_bounds)?;

    // Build node_name → layer_index map for input_sources resolution.
    let name_to_index: std::collections::HashMap<&str, usize> = topo_order
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    let mut records = Vec::with_capacity(topo_order.len());

    for (layer_index, node_name) in topo_order.iter().enumerate() {
        // Look up the node to get its layer type.
        let node = match graph.node(node_name) {
            Some(n) => n,
            None => continue, // Should not happen after topo sort
        };

        let layer_type = node.layer().layer_type().to_string();

        // Get the output bounds for this node.
        let output_bt = match result.bounds.get(node_name) {
            Some(bt) => bt,
            None => continue, // Node has no bounds (e.g. NETWORK_INPUT sentinel)
        };

        // Get the input bounds: either the network input or the predecessor's output.
        let input_bt = resolve_input_bounds(graph, node, &result.bounds, input_bounds);

        // Determine propagation method from provenance.
        // IBP is the safe conservative default: valid for all nodes, just wider bounds.
        // CROWN is used only when provenance confirms convergence.
        let method = match result.provenance.get(node_name) {
            Some(prov) if !prov.is_fallback() => PropMethod::Crown,
            Some(_) | None => PropMethod::Ibp,
        };

        // Resolve input source layer indices from the graph edges.
        let input_sources = resolve_input_sources(node, &name_to_index);

        // Convert BoundedTensor bounds to Vec<(f32, f32)> pairs.
        let input_pairs = bounded_tensor_to_pairs(input_bt);
        let output_pairs = bounded_tensor_to_pairs(output_bt);

        records.push(LayerBoundRecord {
            layer_index,
            layer_type,
            input_bounds: input_pairs,
            output_bounds: output_pairs,
            method,
            node_name: Some(node_name.clone()),
            input_sources: Some(input_sources),
        });
    }

    Ok(records)
}

/// Resolve the layer indices of this node's input sources.
///
/// Maps each input edge name to its layer index in the topological ordering.
/// NETWORK_INPUT edges produce no source index (empty list means "network input").
fn resolve_input_sources(
    node: &GraphNode,
    name_to_index: &std::collections::HashMap<&str, usize>,
) -> Vec<usize> {
    node.inputs()
        .iter()
        .filter(|name| name.as_str() != NETWORK_INPUT)
        .filter_map(|name| name_to_index.get(name.as_str()).copied())
        .collect()
}

/// Resolve the input bounds for a node by looking at its first input edge.
///
/// For nodes whose input is NETWORK_INPUT, returns the external input bounds.
/// For other nodes, returns the output bounds of the predecessor node.
/// Falls back to the external input if the predecessor has no cached bounds.
fn resolve_input_bounds<'a>(
    _graph: &GraphNetwork,
    node: &GraphNode,
    node_bounds: &'a std::collections::HashMap<String, BoundedTensor>,
    network_input: &'a BoundedTensor,
) -> &'a BoundedTensor {
    let inputs = node.inputs();
    if inputs.is_empty() {
        return network_input;
    }

    let first_input = &inputs[0];

    // Check for the NETWORK_INPUT sentinel.
    if first_input == NETWORK_INPUT {
        return network_input;
    }

    // Look up predecessor's output bounds.
    node_bounds.get(first_input).unwrap_or(network_input)
}

/// Convert a `BoundedTensor` to a flat Vec of (lower, upper) pairs.
///
/// Iterates over all elements in the tensor's lower and upper arrays,
/// pairing corresponding elements.
fn bounded_tensor_to_pairs(bt: &BoundedTensor) -> Vec<(f32, f32)> {
    let (lower, upper) = bt.lower_upper();
    lower
        .iter()
        .zip(upper.iter())
        .map(|(&lo, &hi)| (lo, hi))
        .collect()
}

#[cfg(test)]
#[path = "layer_bounds_tests.rs"]
mod tests;
