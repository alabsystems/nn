// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Automated sub-block decomposition for verification.
//!
//! Walks a NY `GraphNetwork`, identifies normalization layers as
//! cut points, and partitions the graph into independently-verifiable
//! sub-blocks. Each sub-block has 3-5 layers — tractable for CROWN at
//! large dimensions (e.g., D=512) where monolithic propagation produces
//! vacuously wide bounds.
//!
//! # Algorithm
//!
//! 1. Walk the graph's node order
//! 2. Identify normalization boundaries (InstanceNorm, LayerNorm, RMSNorm,
//!    BatchNorm, GroupNorm, AdaIN)
//! 3. Cut the graph at each norm boundary
//! 4. Return `Vec<SubBlock>` with input/output specifications
//!
//! The key insight: normalization layers reset the scale of activations
//! (mean≈0, variance≈1 post-norm), so they are natural verification
//! boundaries. Each sub-block between norms has bounded activation growth,
//! making CROWN tractable.
//!
//! Part of #2218: Epic — Perfect Kokoro.
//! Part of #2597: Generator [-inf, inf] bounds.

use ny_propagate::layers::Layer;
use ny_propagate::GraphNetwork;

use crate::error::VerifyError;

/// A sub-block of a graph network for independent verification.
///
/// Contains the indices (into the graph's node order) defining the
/// sub-block boundary, and the normalization layer name (if any) at
/// its output boundary.
#[derive(Debug, Clone)]
pub struct SubBlock {
    /// Human-readable name for this sub-block.
    pub name: String,
    /// Index of the first node in this sub-block (inclusive).
    pub start_idx: usize,
    /// Index of the last node in this sub-block (inclusive).
    pub end_idx: usize,
    /// Number of layers in this sub-block.
    pub layer_count: usize,
    /// Whether this sub-block ends at a normalization boundary.
    pub ends_at_norm: bool,
    /// The normalization layer type name at the boundary (if any).
    pub boundary_norm_type: Option<&'static str>,
}

/// Result of sub-block decomposition.
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    /// The sub-blocks, in graph order.
    pub sub_blocks: Vec<SubBlock>,
    /// Total number of normalization boundaries found.
    pub norm_boundary_count: usize,
    /// Total number of layers in the graph.
    pub total_layers: usize,
}

impl DecompositionResult {
    /// Whether all sub-blocks have at most `max_layers` layers.
    #[must_use]
    pub fn all_tractable(&self, max_layers: usize) -> bool {
        self.sub_blocks.iter().all(|b| b.layer_count <= max_layers)
    }

    /// Maximum sub-block size (in layers).
    #[must_use]
    pub fn max_block_size(&self) -> usize {
        self.sub_blocks
            .iter()
            .map(|b| b.layer_count)
            .max()
            .unwrap_or(0)
    }
}

/// Check if a `Layer` is a normalization layer (verification boundary).
///
/// These layers reset the scale of activations (mean≈0, variance≈1),
/// making them natural cut points for sub-block decomposition.
fn is_norm_layer(layer: &Layer) -> Option<&'static str> {
    match layer {
        Layer::InstanceNorm1d(_) => Some("InstanceNorm1d"),
        Layer::LayerNorm(_) => Some("LayerNorm"),
        Layer::RmsNorm(_) => Some("RmsNorm"),
        Layer::GroupNorm(_) => Some("GroupNorm"),
        Layer::AdaIN1d(_) => Some("AdaIN1d"),
        Layer::BatchNorm(_) => Some("BatchNorm"),
        _ => None,
    }
}

/// Decompose a `GraphNetwork` into sub-blocks at normalization boundaries.
///
/// Walks the graph's node order and cuts at each normalization layer.
/// Each sub-block contains the layers from one norm boundary to the next.
///
/// # Arguments
///
/// * `graph` - The NY graph network to decompose.
/// * `max_block_size` - Maximum desired sub-block size. If a sub-block
///   exceeds this, it will be force-split to keep all blocks tractable.
///   Pass `usize::MAX` to only split at normalization boundaries.
///
/// # Errors
///
/// Returns `VerifyError::EmptyGraph` if the graph has no nodes.
pub fn decompose_at_norms(
    graph: &GraphNetwork,
    max_block_size: usize,
) -> Result<DecompositionResult, VerifyError> {
    let node_names = graph.node_names();
    let total_layers = node_names.len();

    if total_layers == 0 {
        return Err(VerifyError::EmptyGraph);
    }

    let mut sub_blocks = Vec::new();
    let mut current_start = 0;
    let mut norm_count = 0;
    let mut block_idx = 0;

    for (i, name) in node_names.iter().enumerate() {
        let norm_type = graph.node(name).and_then(|n| is_norm_layer(n.layer()));

        let at_end = i == total_layers - 1;
        let block_size = i - current_start + 1;
        let should_cut = norm_type.is_some() || at_end || block_size >= max_block_size;

        if should_cut {
            if norm_type.is_some() {
                norm_count += 1;
            }

            sub_blocks.push(SubBlock {
                name: format!("block_{block_idx}"),
                start_idx: current_start,
                end_idx: i,
                layer_count: block_size,
                ends_at_norm: norm_type.is_some(),
                boundary_norm_type: norm_type,
            });

            if !at_end {
                current_start = i + 1;
                block_idx += 1;
            }
        }
    }

    Ok(DecompositionResult {
        sub_blocks,
        norm_boundary_count: norm_count,
        total_layers,
    })
}

/// Decompose with default settings: cut only at normalization boundaries.
///
/// Convenience wrapper for `decompose_at_norms(graph, usize::MAX)`.
pub fn decompose(graph: &GraphNetwork) -> Result<DecompositionResult, VerifyError> {
    decompose_at_norms(graph, usize::MAX)
}

#[cfg(test)]
#[path = "subblock_decompose_tests.rs"]
mod tests;
