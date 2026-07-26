// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Topology validation gap proof tests for trace_to_graph_model.
//!
//! Documents and proves the F2 finding from the API health report:
//! `compile_trace()` validates topology (rejects forward references),
//! but `trace_to_graph_model()` does NOT call `validate_topology()`.
//!
//! The gap is partially mitigated by `resolve_input()` which fails when
//! a forward-referenced node hasn't been inserted into `node_names` yet,
//! but the error type is misleading (`UnsupportedOp` instead of
//! `TopologyError`).
//!
//! Part of #2146 (baseline verification).

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::{DType, TensorError};
use nn_dsl::compile_trace;
use nn_verify::trace_to_graph_model;

// ---------------------------------------------------------------------------
// Graph constructors
// ---------------------------------------------------------------------------

/// Graph with forward reference ON the output path.
///
/// Node layout (list order):
///   0: input_0  (Input, no deps)
///   1: add_0    (Add, inputs=[0, 2])  ← forward ref to node 2
///   2: relu_0   (Relu, inputs=[0])
///   3: add_1    (Add, inputs=[1, 2])  ← OUTPUT (depends on both 1 and 2)
///
/// BFS from output (node 3): reachable = {3, 1, 2, 0}.
/// All nodes are reachable. Node 1 (add_0) has a forward reference
/// to node 2, which appears after it in list order.
///
/// compile_trace rejects this (validates topology).
/// trace_to_graph_model processes nodes in list order, so when it
/// reaches node 1 (add_0), node 2 (relu_0) hasn't been inserted into
/// node_names yet → resolve_input fails with misleading UnsupportedOp.
fn make_forward_ref_on_output_path() -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![2, 3],
            DType::F32,
        ),
        // Node 1: forward reference to node 2 (not yet defined)
        TraceNode::new(
            1,
            "add_0".into(),
            TraceOp::Add,
            vec![0, 2],
            vec![2, 3],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
        // Output node: depends on nodes 1 and 2 (both reachable)
        TraceNode::new(
            3,
            "add_1".into(),
            TraceOp::Add,
            vec![1, 2],
            vec![2, 3],
            DType::F32,
        ),
    ])
}

/// Graph with forward reference OFF the output path.
///
/// Node layout (list order):
///   0: input_0   (Input, no deps)
///   1: add_0     (Add, inputs=[0, 2])  ← forward ref to node 2, but UNREACHABLE
///   2: relu_0    (Relu, inputs=[0])    ← OUTPUT
///
/// BFS from output (node 2): reachable = {2, 0}. Node 1 is unreachable.
///
/// compile_trace rejects this (validates topology on ALL nodes).
/// trace_to_graph_model silently drops node 1 (unreachable) and succeeds.
fn make_forward_ref_off_output_path() -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![2, 3],
            DType::F32,
        ),
        // Node 1: forward reference to node 2, but unreachable from output
        TraceNode::new(
            1,
            "add_0".into(),
            TraceOp::Add,
            vec![0, 2],
            vec![2, 3],
            DType::F32,
        ),
        // Node 2: output (last node)
        TraceNode::new(
            2,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
    ])
}

// ---------------------------------------------------------------------------
// Proof: validate_topology catches forward references
// ---------------------------------------------------------------------------

/// Proves that `ComputationGraph::validate_topology()` correctly rejects
/// forward references with a `TopologyError` error.
#[test]
fn test_validate_topology_catches_forward_reference() {
    let graph = make_forward_ref_on_output_path();
    let err = graph.validate_topology().unwrap_err();

    match err {
        TensorError::TopologyError {
            node_name,
            missing_input,
            ..
        } => {
            assert_eq!(node_name, "add_0");
            assert_eq!(missing_input, 2);
        }
        other => panic!("expected TopologyError, got: {other}"),
    }
}

// ---------------------------------------------------------------------------
// Proof: compile_trace catches forward references
// ---------------------------------------------------------------------------

/// Proves that `compile_trace()` rejects forward references because it
/// calls `validate_topology()` at entry.
#[test]
fn test_compile_trace_rejects_forward_reference() {
    let graph = make_forward_ref_on_output_path();
    let err = compile_trace(&graph).unwrap_err();

    let msg = err.to_string();
    assert!(
        msg.contains("missing_input=2") || msg.contains("missing input"),
        "compile_trace error should mention missing input node, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Proof: trace_to_graph_model catches forward references accidentally
// ---------------------------------------------------------------------------

/// Proves that `trace_to_graph_model()` now calls `validate_topology()`
/// and catches forward references with a topology-based error — not the
/// accidental `UnsupportedOp` from `resolve_input()`.
///
/// This is the fix for the F2 gap: intentional topology validation at entry.
#[test]
fn test_trace_to_graph_model_catches_forward_ref_with_wrong_error() {
    let graph = make_forward_ref_on_output_path();
    let err = trace_to_graph_model(&graph).unwrap_err();

    let msg = err.to_string();
    // After the fix, trace_to_graph_model calls validate_topology() at entry,
    // producing a topology-based error instead of the misleading UnsupportedOp.
    assert!(
        msg.contains("topology"),
        "trace_to_graph_model should now mention topology validation, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Proof: unreachable forward-ref nodes are silently dropped
// ---------------------------------------------------------------------------

/// Proves that `trace_to_graph_model()` now rejects graphs with forward
/// references even when the malformed node is unreachable from the output.
///
/// Previously (F2 gap), `trace_to_graph_model()` silently dropped unreachable
/// forward-ref nodes via `reachable_nodes()` BFS. Now `validate_topology()`
/// runs on ALL nodes before reachability filtering, catching the violation.
#[test]
fn test_trace_to_graph_model_silently_drops_unreachable_forward_ref() {
    let graph = make_forward_ref_off_output_path();

    // validate_topology catches the forward reference in node 1
    let topo_err = graph.validate_topology().unwrap_err();
    match topo_err {
        TensorError::TopologyError { node_name, .. } => {
            assert_eq!(node_name, "add_0");
        }
        other => panic!("expected TopologyError, got: {other}"),
    }

    // compile_trace also catches it
    assert!(compile_trace(&graph).is_err());

    // After the fix, trace_to_graph_model also rejects this graph —
    // validate_topology() runs before reachability filtering.
    let err = trace_to_graph_model(&graph).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("topology"),
        "trace_to_graph_model should now reject forward-ref graphs with topology error, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Proof: well-ordered graph accepted by both paths
// ---------------------------------------------------------------------------

/// Proves that both `compile_trace()` and `trace_to_graph_model()` accept
/// a correctly ordered graph (input -> relu -> add).
#[test]
fn test_both_paths_accept_valid_topology() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![2, 3],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "add_0".into(),
            TraceOp::Add,
            vec![0, 1], // both inputs defined before this node
            vec![2, 3],
            DType::F32,
        ),
    ]);

    // validate_topology should succeed
    graph.validate_topology().expect("valid topology");

    // compile_trace should succeed
    let steps = compile_trace(&graph).expect("compile_trace should accept valid graph");
    assert_eq!(steps.len(), 3);

    // trace_to_graph_model should succeed — 3-node graph (input + relu + add)
    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model should accept valid graph")
        .graph;
    // All 3 nodes reachable from output: input becomes NETWORK_INPUT, relu and add
    // become layers. Exact count depends on identity-wrapping but must be >= 2.
    assert!(
        gn.num_nodes() >= 2,
        "expected at least 2 translated nodes for 3-node graph, got {}",
        gn.num_nodes()
    );
}

// ---------------------------------------------------------------------------
// Proof: dangling reference (node references nonexistent node)
// ---------------------------------------------------------------------------

/// Proves that a dangling reference (referencing a node ID that doesn't
/// exist in the graph at all) is caught by both paths.
#[test]
fn test_dangling_reference_caught_by_both_paths() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![999], // references nonexistent node
            vec![4],
            DType::F32,
        ),
    ]);

    // validate_topology: catches it as TopologyError
    let topo_err = graph.validate_topology().unwrap_err();
    match topo_err {
        TensorError::TopologyError { missing_input, .. } => {
            assert_eq!(missing_input, 999);
        }
        other => panic!("expected TopologyError, got: {other}"),
    }

    // compile_trace: catches it via validate_topology
    let compile_err = compile_trace(&graph).unwrap_err();
    let msg = compile_err.to_string();
    assert!(
        msg.contains("999"),
        "compile_trace should mention dangling node ID 999, got: {msg}"
    );

    // trace_to_graph_model: catches it via validate_topology at entry
    let verify_err = trace_to_graph_model(&graph).unwrap_err();
    let msg = verify_err.to_string();
    assert!(
        msg.contains("topology"),
        "trace_to_graph_model should fail with topology error, got: {msg}"
    );
}
