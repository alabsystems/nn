// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `ComputationGraph::override_node_shapes()`.
//!
//! Exercises shape overrides used by the convert pipeline to fix buffer
//! mismatches when importing PyTorch models with different tracing shapes
//! than the reference trace.

use std::collections::HashMap;

use super::*;
use crate::DType;

// -- Helpers ------------------------------------------------------------------

/// Build a simple 3-node linear graph: input -> relu -> output.
fn make_linear_graph() -> ComputationGraph {
    let nodes = vec![
        TraceNode::new(
            1,
            "input_0".to_string(),
            TraceOp::Input,
            vec![],
            vec![1, 3, 224, 224],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "relu_0".to_string(),
            TraceOp::Relu,
            vec![1],
            vec![1, 3, 224, 224],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "add_0".to_string(),
            TraceOp::Add,
            vec![1, 2],
            vec![1, 3, 224, 224],
            DType::F32,
        ),
    ];
    ComputationGraph::from_nodes(nodes)
}

// -- Basic single-node override -----------------------------------------------

#[test]
fn test_override_node_shapes_single_node_updates_shape() {
    let mut graph = make_linear_graph();
    let mut overrides = HashMap::new();
    overrides.insert("relu_0".to_string(), vec![1, 64, 56, 56]);

    let updated = graph.override_node_shapes(&overrides);

    assert_eq!(updated, 1, "should update exactly one node");
    let node = graph.node(2).unwrap();
    assert_eq!(node.output_shape(), &[1, 64, 56, 56]);
}

// -- Multiple nodes override --------------------------------------------------

#[test]
fn test_override_node_shapes_multiple_nodes_updates_all() {
    let mut graph = make_linear_graph();
    let mut overrides = HashMap::new();
    overrides.insert("input_0".to_string(), vec![2, 3, 112, 112]);
    overrides.insert("relu_0".to_string(), vec![2, 3, 112, 112]);
    overrides.insert("add_0".to_string(), vec![2, 3, 112, 112]);

    let updated = graph.override_node_shapes(&overrides);

    assert_eq!(updated, 3, "should update all three nodes");
    for node in graph.nodes() {
        assert_eq!(
            node.output_shape(),
            &[2, 3, 112, 112],
            "node {} should have updated shape",
            node.name()
        );
    }
}

// -- Non-existent node name (graceful no-op) ----------------------------------

#[test]
fn test_override_node_shapes_nonexistent_name_returns_zero() {
    let mut graph = make_linear_graph();
    let mut overrides = HashMap::new();
    overrides.insert("does_not_exist".to_string(), vec![1, 1]);

    let updated = graph.override_node_shapes(&overrides);

    assert_eq!(updated, 0, "no nodes should be updated for unknown name");
    // Verify original shapes are untouched.
    assert_eq!(graph.node(1).unwrap().output_shape(), &[1, 3, 224, 224]);
    assert_eq!(graph.node(2).unwrap().output_shape(), &[1, 3, 224, 224]);
    assert_eq!(graph.node(3).unwrap().output_shape(), &[1, 3, 224, 224]);
}

// -- Empty overrides map (no-op) ----------------------------------------------

#[test]
fn test_override_node_shapes_empty_map_is_noop() {
    let mut graph = make_linear_graph();
    let overrides: HashMap<String, Vec<usize>> = HashMap::new();

    let updated = graph.override_node_shapes(&overrides);

    assert_eq!(updated, 0, "empty overrides should update nothing");
    assert_eq!(graph.node(1).unwrap().output_shape(), &[1, 3, 224, 224]);
}

// -- Preserves other node properties ------------------------------------------

#[test]
fn test_override_node_shapes_preserves_other_properties() {
    let mut graph = make_linear_graph();
    let mut overrides = HashMap::new();
    overrides.insert("relu_0".to_string(), vec![4, 64, 28, 28]);

    graph.override_node_shapes(&overrides);

    let node = graph.node(2).unwrap();
    // Shape updated:
    assert_eq!(node.output_shape(), &[4, 64, 28, 28]);
    // Other properties preserved:
    assert_eq!(node.id(), 2);
    assert_eq!(node.name(), "relu_0");
    assert!(matches!(node.op(), TraceOp::Relu));
    assert_eq!(node.inputs(), &[1]);
    assert_eq!(node.output_dtype(), DType::F32);
}

// -- Partial match: some names exist, some don't ------------------------------

#[test]
fn test_override_node_shapes_partial_match_updates_only_existing() {
    let mut graph = make_linear_graph();
    let mut overrides = HashMap::new();
    overrides.insert("input_0".to_string(), vec![8, 3, 64, 64]);
    overrides.insert("nonexistent".to_string(), vec![1]);

    let updated = graph.override_node_shapes(&overrides);

    assert_eq!(updated, 1, "only one node name exists in graph");
    assert_eq!(graph.node(1).unwrap().output_shape(), &[8, 3, 64, 64]);
    // Other nodes unchanged:
    assert_eq!(graph.node(2).unwrap().output_shape(), &[1, 3, 224, 224]);
    assert_eq!(graph.node(3).unwrap().output_shape(), &[1, 3, 224, 224]);
}

// -- Rank change via override -------------------------------------------------

#[test]
fn test_override_node_shapes_rank_change_allowed() {
    let mut graph = make_linear_graph();
    // Override to a different rank (e.g., from 4D to 2D).
    let mut overrides = HashMap::new();
    overrides.insert("add_0".to_string(), vec![8, 512]);

    let updated = graph.override_node_shapes(&overrides);

    assert_eq!(updated, 1);
    assert_eq!(graph.node(3).unwrap().output_shape(), &[8, 512]);
}

// -- Override on empty graph --------------------------------------------------

#[test]
fn test_override_node_shapes_empty_graph_returns_zero() {
    let mut graph = ComputationGraph::from_nodes(vec![]);
    let mut overrides = HashMap::new();
    overrides.insert("anything".to_string(), vec![1, 2, 3]);

    let updated = graph.override_node_shapes(&overrides);

    assert_eq!(updated, 0);
    assert!(graph.is_empty());
}

// -- Graph remains valid after override (topology unchanged) ------------------

#[test]
fn test_override_node_shapes_graph_still_valid_after_override() {
    let mut graph = make_linear_graph();
    let mut overrides = HashMap::new();
    overrides.insert("input_0".to_string(), vec![2, 3, 128, 128]);
    overrides.insert("relu_0".to_string(), vec![2, 3, 128, 128]);
    overrides.insert("add_0".to_string(), vec![2, 3, 128, 128]);

    graph.override_node_shapes(&overrides);

    // Topology should still be valid — override only changes shapes, not edges.
    graph
        .validate_topology()
        .expect("graph should remain topologically valid after shape override");

    // Output node should still be accessible.
    let output = graph.output_node().unwrap();
    assert_eq!(output.name(), "add_0");
    assert_eq!(output.output_shape(), &[2, 3, 128, 128]);
}

// -- Override to scalar shape (empty vec) -------------------------------------

#[test]
fn test_override_node_shapes_scalar_shape() {
    let mut graph = make_linear_graph();
    let mut overrides = HashMap::new();
    overrides.insert("add_0".to_string(), vec![]);

    let updated = graph.override_node_shapes(&overrides);

    assert_eq!(updated, 1);
    assert_eq!(graph.node(3).unwrap().output_shape(), &[] as &[usize]);
}

// -- Override count matches number of matching nodes --------------------------

#[test]
fn test_override_node_shapes_return_count_accuracy() {
    let nodes = vec![
        TraceNode::new(
            10,
            "layer_a".to_string(),
            TraceOp::Input,
            vec![],
            vec![1, 768],
            DType::F32,
        ),
        TraceNode::new(
            20,
            "layer_b".to_string(),
            TraceOp::Relu,
            vec![10],
            vec![1, 768],
            DType::F32,
        ),
        TraceNode::new(
            30,
            "layer_c".to_string(),
            TraceOp::Add,
            vec![10, 20],
            vec![1, 768],
            DType::F32,
        ),
        TraceNode::new(
            40,
            "layer_d".to_string(),
            TraceOp::Relu,
            vec![30],
            vec![1, 768],
            DType::F32,
        ),
    ];
    let mut graph = ComputationGraph::from_nodes(nodes);

    let mut overrides = HashMap::new();
    overrides.insert("layer_a".to_string(), vec![2, 768]);
    overrides.insert("layer_c".to_string(), vec![2, 768]);
    overrides.insert("layer_d".to_string(), vec![2, 768]);
    overrides.insert("no_such_layer".to_string(), vec![99]);

    let updated = graph.override_node_shapes(&overrides);

    assert_eq!(updated, 3, "3 of 4 override keys exist in graph");
    assert_eq!(graph.node(10).unwrap().output_shape(), &[2, 768]);
    assert_eq!(graph.node(20).unwrap().output_shape(), &[1, 768]); // not overridden
    assert_eq!(graph.node(30).unwrap().output_shape(), &[2, 768]);
    assert_eq!(graph.node(40).unwrap().output_shape(), &[2, 768]);
}
