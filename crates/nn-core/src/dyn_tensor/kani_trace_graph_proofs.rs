// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `ComputationGraph` segmentation, topology,
//! and graph construction properties (#3737).
//!
//! Extends `kani_trace.rs` with deeper proofs for:
//!  1. Segmentation: boundary splitting produces correct segment counts
//!  2. Segmentation: metadata propagation through boundaries
//!  3. Topology: self-referencing node fails validation
//!  4. Topology: multi-input DAG validates correctly
//!  5. Graph: node() returns None for all IDs not in the graph
//!  6. Graph: len() matches nodes().len()
//!  7. Graph: input_nodes() returns empty for non-Input graphs
//!  8. Graph: from_nodes preserves topological order
//!  9. Segmentation: multiple boundaries produce correct segment count
//! 10. Segmentation: empty graph has no boundaries
//! 11. Graph: output_nodes returns correct count after mark_output
//! 12. Graph: set_primary_output then mark_output yields 2 outputs

use crate::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use crate::DType;

fn make_node(id: u64, op: TraceOp, inputs: Vec<u64>, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(id, format!("node_{id}"), op, inputs, shape, DType::F32)
}

// ===========================================================================
// Segmentation with boundaries
// ===========================================================================

/// Prove: graph with one SegmentBoundary produces exactly 2 segments.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn proof_segment_one_boundary_two_segments() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    let n1 = make_node(2, TraceOp::Relu, vec![1], vec![4]);
    let n2 = make_node(
        3,
        TraceOp::SegmentBoundary {
            reason: "test_split".to_string(),
            input_bounds: Some((-1.0, 1.0)),
        },
        vec![2],
        vec![4],
    );
    let n3 = make_node(4, TraceOp::Sigmoid, vec![3], vec![4]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1, n2, n3]);

    assert!(graph.has_segment_boundaries(), "must detect boundary");
    assert!(graph.segment_boundaries().len() == 1, "exactly 1 boundary");

    let segmented = graph.split_at_segment_boundaries();
    assert!(
        segmented.segments.len() == 2,
        "1 boundary must produce 2 segments"
    );
    // First segment: nodes before boundary (Input, Relu)
    assert!(
        segmented.segments[0].graph.len() == 2,
        "first segment has 2 nodes"
    );
    // Second segment: nodes after boundary (Sigmoid)
    assert!(
        segmented.segments[1].graph.len() == 1,
        "second segment has 1 node"
    );
}

/// Prove: boundary metadata (reason, bounds) propagates to segment.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn proof_segment_boundary_metadata_propagates() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![8]);
    let n1 = make_node(
        2,
        TraceOp::SegmentBoundary {
            reason: "length_regulate".to_string(),
            input_bounds: Some((-2.0, 2.0)),
        },
        vec![1],
        vec![8],
    );
    let n2 = make_node(3, TraceOp::Relu, vec![2], vec![8]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1, n2]);

    let segmented = graph.split_at_segment_boundaries();
    assert!(segmented.segments.len() == 2, "2 segments");

    // First segment gets the boundary reason
    let seg0 = &segmented.segments[0];
    assert!(
        seg0.boundary_reason.is_some(),
        "first segment must carry boundary reason"
    );
    let reason = seg0.boundary_reason.as_ref().unwrap();
    assert!(reason == "length_regulate", "reason must match");
    assert!(
        seg0.boundary_bounds == Some((-2.0, 2.0)),
        "bounds must match"
    );
}

/// Prove: two boundaries produce 3 segments.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(7)]
fn proof_segment_two_boundaries_three_segments() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    let n1 = make_node(
        2,
        TraceOp::SegmentBoundary {
            reason: "split1".to_string(),
            input_bounds: None,
        },
        vec![1],
        vec![4],
    );
    let n2 = make_node(3, TraceOp::Relu, vec![2], vec![4]);
    let n3 = make_node(
        4,
        TraceOp::SegmentBoundary {
            reason: "split2".to_string(),
            input_bounds: None,
        },
        vec![3],
        vec![4],
    );
    let n4 = make_node(5, TraceOp::Sigmoid, vec![4], vec![4]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1, n2, n3, n4]);

    let segmented = graph.split_at_segment_boundaries();
    assert!(
        segmented.segments.len() == 3,
        "2 boundaries must produce 3 segments"
    );
}

// ===========================================================================
// Topology validation edge cases
// ===========================================================================

/// Prove: a node referencing itself (self-loop) fails topology validation.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn proof_validate_topology_self_reference() {
    // Node 1 references itself — not a valid DAG
    let n0 = make_node(1, TraceOp::Relu, vec![1], vec![4]);
    let graph = ComputationGraph::from_nodes(vec![n0]);
    let result = graph.validate_topology();
    assert!(result.is_err(), "self-referencing node must fail topology");
}

/// Prove: a multi-input DAG with correct ordering passes topology.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn proof_validate_topology_multi_input_dag() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    let n1 = make_node(2, TraceOp::Input, vec![], vec![4]);
    let n2 = make_node(3, TraceOp::Add, vec![1, 2], vec![4]);
    let n3 = make_node(4, TraceOp::Relu, vec![3], vec![4]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1, n2, n3]);
    assert!(graph.validate_topology().is_ok(), "valid DAG must pass");
}

/// Prove: missing input reference (non-existent node) fails topology.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn proof_validate_topology_missing_input_ref() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    // Node 2 references node 999 which does not exist
    let n1 = make_node(2, TraceOp::Relu, vec![999], vec![4]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1]);
    assert!(
        graph.validate_topology().is_err(),
        "reference to non-existent node must fail"
    );
}

// ===========================================================================
// Graph query properties
// ===========================================================================

/// Prove: node() returns None for IDs not in the graph.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn proof_graph_node_missing_returns_none() {
    let n0 = make_node(100, TraceOp::Input, vec![], vec![4]);
    let graph = ComputationGraph::from_nodes(vec![n0]);

    assert!(graph.node(100).is_some(), "ID 100 exists");
    assert!(graph.node(0).is_none(), "ID 0 does not exist");
    assert!(graph.node(99).is_none(), "ID 99 does not exist");
    assert!(graph.node(101).is_none(), "ID 101 does not exist");
}

/// Prove: len() equals nodes().len().
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn proof_graph_len_matches_nodes_len() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    let n1 = make_node(2, TraceOp::Relu, vec![1], vec![4]);
    let n2 = make_node(3, TraceOp::Sigmoid, vec![2], vec![4]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1, n2]);

    assert!(
        graph.len() == graph.nodes().len(),
        "len() must match nodes().len()"
    );
    assert!(graph.len() == 3, "3 nodes");
}

/// Prove: input_nodes() returns empty when no Input ops exist.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn proof_graph_input_nodes_empty_when_no_inputs() {
    // Build a graph with only non-Input ops (synthetic, no real deps)
    let n0 = make_node(1, TraceOp::Constant { value: 1.0 }, vec![], vec![4]);
    let n1 = make_node(2, TraceOp::Relu, vec![1], vec![4]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1]);

    assert!(
        graph.input_nodes().is_empty(),
        "no Input ops means empty input_nodes()"
    );
}

/// Prove: from_nodes preserves insertion order (topological).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn proof_graph_from_nodes_preserves_order() {
    let n0 = make_node(10, TraceOp::Input, vec![], vec![4]);
    let n1 = make_node(20, TraceOp::Relu, vec![10], vec![4]);
    let n2 = make_node(30, TraceOp::Add, vec![10, 20], vec![4]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1, n2]);

    let nodes = graph.nodes();
    assert!(nodes[0].id() == 10, "first node preserves order");
    assert!(nodes[1].id() == 20, "second node preserves order");
    assert!(nodes[2].id() == 30, "third node preserves order");
}

/// Prove: empty graph has no segment boundaries.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_empty_graph_no_boundaries() {
    let graph = ComputationGraph::from_nodes(vec![]);
    assert!(
        !graph.has_segment_boundaries(),
        "empty graph has no boundaries"
    );
    assert!(graph.segment_boundaries().is_empty(), "no boundary indices");
}

/// Prove: output_nodes count increments with mark_output.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn proof_output_nodes_count_after_mark() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    let n1 = make_node(2, TraceOp::Relu, vec![1], vec![4]);
    let n2 = make_node(3, TraceOp::Sigmoid, vec![2], vec![4]);
    let mut graph = ComputationGraph::from_nodes(vec![n0, n1, n2]);

    // Default: 1 output (last node, id=3)
    assert!(graph.output_nodes().len() == 1);

    // Mark node 1
    graph.mark_output(1);
    assert!(graph.output_nodes().len() == 2);

    // Mark node 2
    graph.mark_output(2);
    assert!(graph.output_nodes().len() == 3);
}

/// Prove: set_primary_output then mark_output yields exactly 2 outputs.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn proof_set_primary_then_mark_yields_two() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    let n1 = make_node(2, TraceOp::Relu, vec![1], vec![4]);
    let mut graph = ComputationGraph::from_nodes(vec![n0, n1]);

    // Set primary to node 1, then mark node 2
    graph.set_primary_output(1);
    assert!(graph.output_nodes().len() == 1, "after set_primary: 1");

    graph.mark_output(2);
    assert!(graph.output_nodes().len() == 2, "after mark: 2");

    let outputs = graph.output_nodes();
    assert!(outputs[0].id() == 1, "first output is node 1");
    assert!(outputs[1].id() == 2, "second output is node 2");
}
