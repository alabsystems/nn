// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for sub-block decomposition.

use ny_propagate::layers::{InstanceNorm1dLayer, Layer, LinearLayer, ReLULayer, SliceLayer};
use ny_propagate::{GraphNetwork, GraphNode};
use ndarray::Array2;

use super::{decompose, decompose_at_norms, is_norm_layer};

/// Build a simple linear graph: input → Linear → ReLU → InstanceNorm → Linear → ReLU.
fn build_test_graph_with_norm() -> GraphNetwork {
    let mut graph = GraphNetwork::new();

    // Input node (SliceLayer as conventional graph input).
    graph.add_node(GraphNode::from_input(
        "input",
        Layer::Slice(SliceLayer::new(0, 0, 1)),
    ));

    // Layer 1: Linear (2→2)
    let w = Array2::eye(2);
    let b = ndarray::Array1::zeros(2);
    graph.add_node(GraphNode::new(
        "linear_0",
        Layer::Linear(LinearLayer::new(w.clone(), Some(b.clone())).unwrap()),
        vec!["input".to_string()],
    ));

    // Layer 2: ReLU
    graph.add_node(GraphNode::new(
        "relu_0",
        Layer::ReLU(ReLULayer),
        vec!["linear_0".to_string()],
    ));

    // Layer 3: InstanceNorm (norm boundary)
    graph.add_node(GraphNode::new(
        "norm_0",
        Layer::InstanceNorm1d(InstanceNorm1dLayer::new_default(2, 1e-5).expect("valid norm")),
        vec!["relu_0".to_string()],
    ));

    // Layer 4: Linear (2→2)
    graph.add_node(GraphNode::new(
        "linear_1",
        Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()),
        vec!["norm_0".to_string()],
    ));

    // Layer 5: ReLU
    graph.add_node(GraphNode::new(
        "relu_1",
        Layer::ReLU(ReLULayer),
        vec!["linear_1".to_string()],
    ));

    graph.set_output("relu_1".to_string());
    graph
}

/// Build a graph with no normalization layers.
fn build_no_norm_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();

    let w = Array2::eye(2);
    let b = ndarray::Array1::zeros(2);

    graph.add_node(GraphNode::from_input(
        "input",
        Layer::Slice(SliceLayer::new(0, 0, 1)),
    ));
    graph.add_node(GraphNode::new(
        "linear_0",
        Layer::Linear(LinearLayer::new(w.clone(), Some(b.clone())).unwrap()),
        vec!["input".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu_0",
        Layer::ReLU(ReLULayer),
        vec!["linear_0".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear_1",
        Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()),
        vec!["relu_0".to_string()],
    ));

    graph.set_output("linear_1".to_string());
    graph
}

#[test]
fn test_decompose_with_norm_boundary() {
    let graph = build_test_graph_with_norm();
    let result = decompose(&graph).expect("decompose should succeed");

    // 6 nodes (input + 5 layers), 1 norm boundary
    // Should produce 2 sub-blocks: [input..norm_0] and [linear_1..relu_1]
    assert_eq!(result.norm_boundary_count, 1);
    assert!(
        result.sub_blocks.len() >= 2,
        "expected at least 2 sub-blocks, got {}",
        result.sub_blocks.len()
    );

    // Find the block that ends at norm
    let norm_block = result
        .sub_blocks
        .iter()
        .find(|b| b.ends_at_norm)
        .expect("should have a block ending at norm");
    assert_eq!(norm_block.boundary_norm_type, Some("InstanceNorm1d"));
}

#[test]
fn test_decompose_no_norms() {
    let graph = build_no_norm_graph();
    let result = decompose(&graph).expect("decompose should succeed");

    // No norm boundaries → single block
    assert_eq!(result.norm_boundary_count, 0);
    assert_eq!(result.sub_blocks.len(), 1);
    assert!(!result.sub_blocks[0].ends_at_norm);
}

#[test]
fn test_decompose_empty_graph() {
    let graph = GraphNetwork::new();
    let result = decompose(&graph);
    assert!(result.is_err(), "empty graph should return error");
}

#[test]
fn test_max_block_size_force_split() {
    let graph = build_no_norm_graph();
    // Force split at 2 layers max
    let result = decompose_at_norms(&graph, 2).expect("decompose should succeed");

    // 4 nodes, max block size 2 → at least 2 blocks
    assert!(result.sub_blocks.len() >= 2);
    for block in &result.sub_blocks {
        assert!(
            block.layer_count <= 2,
            "block {} has {} layers, exceeds max 2",
            block.name,
            block.layer_count
        );
    }
}

#[test]
fn test_all_tractable() {
    let graph = build_test_graph_with_norm();
    let result = decompose(&graph).expect("decompose");

    assert!(result.all_tractable(10));
    let max = result.max_block_size();
    assert!(result.all_tractable(max));
    if max > 1 {
        assert!(!result.all_tractable(max - 1));
    }
}

#[test]
fn test_is_norm_layer_instance_norm() {
    let layer =
        Layer::InstanceNorm1d(InstanceNorm1dLayer::new_default(2, 1e-5).expect("valid norm"));
    assert_eq!(is_norm_layer(&layer), Some("InstanceNorm1d"));
}

#[test]
fn test_is_norm_layer_relu() {
    let layer = Layer::ReLU(ReLULayer);
    assert_eq!(is_norm_layer(&layer), None);
}

#[test]
fn test_block_indices_are_contiguous() {
    let graph = build_test_graph_with_norm();
    let result = decompose(&graph).expect("decompose");

    // Verify blocks are contiguous and cover all layers
    let mut expected_start = 0;
    for block in &result.sub_blocks {
        assert_eq!(
            block.start_idx, expected_start,
            "block {} starts at {} but expected {}",
            block.name, block.start_idx, expected_start
        );
        assert!(block.end_idx >= block.start_idx);
        assert_eq!(block.layer_count, block.end_idx - block.start_idx + 1);
        expected_start = block.end_idx + 1;
    }

    // Last block should end at the last layer
    let last = result.sub_blocks.last().unwrap();
    assert_eq!(last.end_idx, result.total_layers - 1);
}
