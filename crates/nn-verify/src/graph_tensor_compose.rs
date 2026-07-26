// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Sequential graph composition: chain and group multiple GraphNetworks.
//!
//! Extracted from graph_tensor.rs (#3350).

use ny_propagate::{GraphNetwork, GraphNode, NETWORK_INPUT};
use nn_dsl::tensor_ir::TensorKernelDef;

use crate::error::VerifyError;

use super::{tensor_kernel_to_graph_with_norm_mode, TensorParamBinding};

/// Chain multiple `GraphNetwork`s sequentially into a single combined graph.
///
/// Graph 0's `NETWORK_INPUT` is the combined graph's input.
/// For graphs 1..N, `NETWORK_INPUT` references are redirected to the
/// previous graph's output node, creating a sequential data-flow pipeline.
///
/// Node names are prefixed with `g{i}_` to avoid collisions between layers.
/// This is the building block for multi-layer CROWN sub-graphs (#2592):
/// CROWN backward propagation crosses layer boundaries within the combined
/// graph, exploiting cross-layer correlations that per-layer CROWN cannot see.
///
/// # Single-entry-point assumption
///
/// ALL `NETWORK_INPUT` references in graph N are redirected to graph N-1's
/// output. This is correct when each graph has a single variable input
/// (the flowing tensor), which is the case for layerwise pipelines where
/// constant parameters are baked into layer definitions. For graphs with
/// multiple variable inputs (multiple `SliceLayer` nodes from
/// `NETWORK_INPUT`), this function would incorrectly redirect all variable
/// entries to the same source. Use [`compose_sequential`](crate::compose_sequential)
/// for multi-variable composition.
pub fn chain_graphs(graphs: &[GraphNetwork]) -> Result<GraphNetwork, VerifyError> {
    if graphs.is_empty() {
        return Err(VerifyError::UnsupportedOp(
            "chain_graphs: empty graph list".into(),
        ));
    }

    let mut combined = GraphNetwork::new();
    let mut prev_output: Option<String> = None;

    for (i, graph) in graphs.iter().enumerate() {
        let prefix = format!("g{i}_");

        for name in graph.node_names() {
            let node = graph.node(name).ok_or_else(|| {
                VerifyError::UnsupportedOp(format!(
                    "chain_graphs: missing node '{name}' in graph {i}",
                ))
            })?;

            let prefixed_name = format!("{prefix}{name}");
            let prefixed_inputs: Vec<String> = node
                .inputs()
                .iter()
                .map(|inp| {
                    if inp == NETWORK_INPUT {
                        match &prev_output {
                            Some(prev) => prev.clone(),
                            None => NETWORK_INPUT.to_string(),
                        }
                    } else {
                        format!("{prefix}{inp}")
                    }
                })
                .collect();

            combined.add_node(GraphNode::new(
                prefixed_name,
                node.layer().clone(),
                prefixed_inputs,
            ));
        }

        prev_output = Some(format!("{prefix}{}", graph.output_name()));
    }

    if let Some(output) = prev_output {
        combined.set_output(output);
    }

    Ok(combined)
}

/// Merge multiple tensor kernels into a single `GraphNetwork` for multi-layer
/// CROWN propagation.
///
/// Each kernel is converted to its own `GraphNetwork` via
/// [`tensor_kernel_to_graph_with_norm_mode`], then chained: layer 0's output
/// feeds layer 1's input, and so on. Returns a single `GraphNetwork` where
/// CROWN can exploit cross-layer correlations for tighter bounds.
///
/// For single-layer input, returns the graph directly without prefixing.
///
/// # Errors
///
/// Returns `VerifyError` if the layer list is empty or any layer fails
/// graph translation.
pub fn tensor_kernels_to_grouped_graph(
    layers: &[(TensorKernelDef, Vec<TensorParamBinding>)],
    norm_mode: crate::verify_types::NormBoundsMode,
) -> Result<GraphNetwork, VerifyError> {
    if layers.is_empty() {
        return Err(VerifyError::UnsupportedOp(
            "tensor_kernels_to_grouped_graph: empty layer list".into(),
        ));
    }
    if layers.len() == 1 {
        return tensor_kernel_to_graph_with_norm_mode(&layers[0].0, &layers[0].1, norm_mode);
    }

    let graphs: Vec<GraphNetwork> = layers
        .iter()
        .enumerate()
        .map(|(i, (kernel, bindings))| {
            tensor_kernel_to_graph_with_norm_mode(kernel, bindings, norm_mode).map_err(|e| {
                VerifyError::UnsupportedOp(format!(
                    "tensor_kernels_to_grouped_graph: layer {i} failed: {e}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    chain_graphs(&graphs)
}
