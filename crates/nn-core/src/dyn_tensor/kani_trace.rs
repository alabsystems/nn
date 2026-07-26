// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional Kani proof harnesses for trace infrastructure (#3695).
//!
//! Supplements `kani_trace_types_proofs.rs` (TraceOp classification/arity)
//! and `kani_trace_op_class_proofs.rs` (canonical_name, WeightRef) with
//! proofs for the trace recorder, computation graph, and segmentation:
//!
//!  1. NameCounter: sequential names are unique
//!  2. NameCounter: names start at _0
//!  3. NameCounter: mixed prefixes have independent counters
//!  4. TraceNode: id accessor matches construction
//!  5. TraceNode: name accessor matches construction
//!  6. TraceNode: inputs accessor matches construction
//!  7. TraceNode: output_shape accessor matches construction
//!  8. TraceNode: output_dtype accessor matches construction
//!  9. ComputationGraph::from_nodes: empty graph is empty
//! 10. ComputationGraph::from_nodes: single node is output
//! 11. ComputationGraph::from_nodes: id_to_index consistent
//! 12. ComputationGraph::mark_output: existing node returns true
//! 13. ComputationGraph::mark_output: missing node returns false
//! 14. ComputationGraph::mark_output: duplicate is idempotent
//! 15. ComputationGraph::set_primary_output: replaces output list
//! 16. ComputationGraph::set_primary_output: missing returns false
//! 17. ComputationGraph::input_nodes: filters Input ops correctly
//! 18. ComputationGraph::validate_topology: valid graph passes
//! 19. ComputationGraph::validate_topology: forward ref fails
//! 20. SegmentedGraph: no boundaries yields 1 segment

use crate::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use crate::DType;

// -- Helper -------------------------------------------------------------------

fn make_node(id: u64, op: TraceOp, inputs: Vec<u64>, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(id, format!("node_{id}"), op, inputs, shape, DType::F32)
}

// ===========================================================================
// NameCounter proofs
// ===========================================================================

/// Prove: NameCounter generates sequential names starting at _0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn proof_name_counter_sequential() {
    let mut counter = super::NameCounter::new();
    let name0 = counter.next_name("relu");
    let name1 = counter.next_name("relu");
    let name2 = counter.next_name("relu");

    assert!(name0 == "relu_0", "first name must be relu_0");
    assert!(name1 == "relu_1", "second name must be relu_1");
    assert!(name2 == "relu_2", "third name must be relu_2");
}

/// Prove: NameCounter first name always ends with _0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn proof_name_counter_starts_at_zero() {
    let mut counter = super::NameCounter::new();
    let name = counter.next_name("add");
    assert!(name == "add_0", "first name must end with _0");
}

/// Prove: NameCounter maintains independent counters per prefix.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn proof_name_counter_independent_prefixes() {
    let mut counter = super::NameCounter::new();
    let relu0 = counter.next_name("relu");
    let add0 = counter.next_name("add");
    let relu1 = counter.next_name("relu");
    let add1 = counter.next_name("add");

    assert!(relu0 == "relu_0", "relu starts at 0");
    assert!(relu1 == "relu_1", "relu increments independently");
    assert!(add0 == "add_0", "add starts at 0");
    assert!(add1 == "add_1", "add increments independently");
}

// ===========================================================================
// TraceNode accessor proofs
// ===========================================================================

/// Prove: TraceNode::id() returns the constructed ID.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_trace_node_id_accessor() {
    let id: u64 = kani::any();
    kani::assume(id >= 1 && id <= 10000);

    let node = make_node(id, TraceOp::Relu, vec![1], vec![2, 3]);
    assert!(node.id() == id, "id() must return construction ID");
}

/// Prove: TraceNode::name() returns the constructed name.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_trace_node_name_accessor() {
    let node = TraceNode::new(
        42,
        "nn_relu".to_string(),
        TraceOp::Relu,
        vec![1],
        vec![4, 8],
        DType::F32,
    );
    assert!(
        node.name() == "nn_relu",
        "name() must return construction name"
    );
}

/// Prove: TraceNode::inputs() returns the constructed input IDs.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_trace_node_inputs_accessor() {
    let inputs = vec![10_u64, 20, 30];
    let node = make_node(1, TraceOp::WhereCond, inputs.clone(), vec![2, 3]);
    assert!(node.inputs().len() == 3, "inputs() must return 3 inputs");
    assert!(node.inputs()[0] == 10, "first input must be 10");
    assert!(node.inputs()[1] == 20, "second input must be 20");
    assert!(node.inputs()[2] == 30, "third input must be 30");
}

/// Prove: TraceNode::output_shape() returns the constructed shape.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_trace_node_output_shape_accessor() {
    let shape = vec![1_usize, 16, 768];
    let node = make_node(1, TraceOp::Input, vec![], shape.clone());
    assert!(
        node.output_shape().len() == 3,
        "output_shape must have rank 3"
    );
    assert!(node.output_shape()[0] == 1);
    assert!(node.output_shape()[1] == 16);
    assert!(node.output_shape()[2] == 768);
}

/// Prove: TraceNode::output_dtype() returns the constructed dtype.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_trace_node_output_dtype_accessor() {
    let node = TraceNode::new(
        1,
        "input_0".to_string(),
        TraceOp::Input,
        vec![],
        vec![4],
        DType::BF16,
    );
    assert!(
        node.output_dtype() == DType::BF16,
        "output_dtype must match construction"
    );
}

// ===========================================================================
// ComputationGraph::from_nodes proofs
// ===========================================================================

/// Prove: from_nodes with empty vec produces empty graph.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_graph_from_nodes_empty() {
    let graph = ComputationGraph::from_nodes(vec![]);
    assert!(graph.is_empty(), "empty nodes must produce empty graph");
    assert!(graph.len() == 0, "len must be 0");
    assert!(graph.output_node().is_none(), "empty graph has no output");
}

/// Prove: from_nodes with a single node sets it as the output.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_graph_from_nodes_single_output() {
    let node = make_node(42, TraceOp::Input, vec![], vec![1, 3, 224, 224]);
    let graph = ComputationGraph::from_nodes(vec![node]);

    assert!(graph.len() == 1, "graph must have 1 node");
    assert!(!graph.is_empty(), "graph must not be empty");

    let out = graph.output_node().unwrap();
    assert!(out.id() == 42, "output must be the single node");
}

/// Prove: from_nodes builds consistent id_to_index mapping.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn proof_graph_id_to_index_consistent() {
    let n0 = make_node(10, TraceOp::Input, vec![], vec![4]);
    let n1 = make_node(20, TraceOp::Relu, vec![10], vec![4]);
    let n2 = make_node(30, TraceOp::Sigmoid, vec![20], vec![4]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1, n2]);

    assert!(graph.len() == 3, "3 nodes");

    // Lookup by ID returns correct node
    let found10 = graph.node(10).unwrap();
    assert!(found10.id() == 10, "ID 10 lookup");
    let found20 = graph.node(20).unwrap();
    assert!(found20.id() == 20, "ID 20 lookup");
    let found30 = graph.node(30).unwrap();
    assert!(found30.id() == 30, "ID 30 lookup");

    // Missing ID returns None
    assert!(graph.node(99).is_none(), "missing ID returns None");
}

// ===========================================================================
// mark_output / set_primary_output proofs
// ===========================================================================

/// Prove: mark_output returns true for existing node and adds to output list.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn proof_mark_output_existing() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    let n1 = make_node(2, TraceOp::Relu, vec![1], vec![4]);
    let mut graph = ComputationGraph::from_nodes(vec![n0, n1]);

    // Default output is last node (id=2)
    let outputs_before = graph.output_nodes();
    assert!(outputs_before.len() == 1, "default: 1 output");

    // Mark node 1 as additional output
    let result = graph.mark_output(1);
    assert!(result, "mark_output must return true for existing node");

    let outputs_after = graph.output_nodes();
    assert!(outputs_after.len() == 2, "now 2 outputs");
}

/// Prove: mark_output returns false for non-existent node.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn proof_mark_output_missing() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    let mut graph = ComputationGraph::from_nodes(vec![n0]);

    let result = graph.mark_output(999);
    assert!(!result, "mark_output must return false for missing node");
}

/// Prove: mark_output is idempotent (duplicate marks don't add duplicates).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn proof_mark_output_idempotent() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    let n1 = make_node(2, TraceOp::Relu, vec![1], vec![4]);
    let mut graph = ComputationGraph::from_nodes(vec![n0, n1]);

    // Default output is id=2. Mark it again.
    graph.mark_output(2);
    graph.mark_output(2);

    let outputs = graph.output_nodes();
    assert!(
        outputs.len() == 1,
        "duplicate mark_output must not create duplicates"
    );
}

/// Prove: set_primary_output replaces entire output list.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn proof_set_primary_output_replaces() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    let n1 = make_node(2, TraceOp::Relu, vec![1], vec![4]);
    let n2 = make_node(3, TraceOp::Sigmoid, vec![2], vec![4]);
    let mut graph = ComputationGraph::from_nodes(vec![n0, n1, n2]);

    // Default output is id=3
    assert!(graph.output_node().unwrap().id() == 3);

    // Set node 1 as primary output
    let result = graph.set_primary_output(1);
    assert!(result, "must return true for existing node");

    let outputs = graph.output_nodes();
    assert!(outputs.len() == 1, "must have exactly 1 output");
    assert!(outputs[0].id() == 1, "primary output must be node 1");
}

/// Prove: set_primary_output returns false for missing node.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn proof_set_primary_output_missing() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    let mut graph = ComputationGraph::from_nodes(vec![n0]);

    let result = graph.set_primary_output(999);
    assert!(
        !result,
        "set_primary_output must return false for missing node"
    );
}

// ===========================================================================
// input_nodes filter proof
// ===========================================================================

/// Prove: input_nodes() returns exactly the nodes with TraceOp::Input.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn proof_input_nodes_filter() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    let n1 = make_node(2, TraceOp::Input, vec![], vec![8]);
    let n2 = make_node(3, TraceOp::Add, vec![1, 2], vec![4]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1, n2]);

    let inputs = graph.input_nodes();
    assert!(inputs.len() == 2, "must find exactly 2 Input nodes");
    assert!(inputs[0].id() == 1, "first input is node 1");
    assert!(inputs[1].id() == 2, "second input is node 2");
}

// ===========================================================================
// validate_topology proofs
// ===========================================================================

/// Prove: a properly ordered graph passes topology validation.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn proof_validate_topology_valid() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    let n1 = make_node(2, TraceOp::Relu, vec![1], vec![4]);
    let n2 = make_node(3, TraceOp::Sigmoid, vec![2], vec![4]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1, n2]);

    let result = graph.validate_topology();
    assert!(result.is_ok(), "valid topology must pass");
}

/// Prove: a graph with a forward reference fails topology validation.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn proof_validate_topology_forward_ref() {
    // Node 2 references node 3 which comes later -> invalid
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    let n1 = make_node(2, TraceOp::Relu, vec![3], vec![4]); // forward ref to 3!
    let n2 = make_node(3, TraceOp::Input, vec![], vec![4]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1, n2]);

    let result = graph.validate_topology();
    assert!(
        result.is_err(),
        "forward reference must fail topology check"
    );
}

// ===========================================================================
// SegmentedGraph proofs
// ===========================================================================

/// Prove: a graph with no segment boundaries produces exactly 1 segment.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn proof_segmented_no_boundaries_single_segment() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    let n1 = make_node(2, TraceOp::Relu, vec![1], vec![4]);
    let n2 = make_node(3, TraceOp::Sigmoid, vec![2], vec![4]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1, n2]);

    assert!(!graph.has_segment_boundaries(), "no boundary nodes present");

    let segmented = graph.split_at_segment_boundaries();
    assert!(
        segmented.segments.len() == 1,
        "no boundaries -> exactly 1 segment"
    );
    assert!(
        segmented.segments[0].graph.len() == 3,
        "single segment contains all nodes"
    );
    assert!(
        segmented.segments[0].boundary_reason.is_none(),
        "no preceding boundary"
    );
}
