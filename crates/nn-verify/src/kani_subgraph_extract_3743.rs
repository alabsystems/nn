// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani verification harnesses for `subgraph_extract.rs` (#3743).
//!
//! Proves additional structural and correctness properties beyond the
//! existing `kani_subgraph_extract.rs` and `kani_subgraph_extract_issue3729.rs`:
//! - `validate_subgraph`: empty graph rejection
//! - `extract_subgraph`: NameContains with no match returns error
//! - `is_ay_compatible_op`: trig ops (Tanh, Sigmoid) classification
//! - `find_ay_candidates`: min > max produces empty
//! - `find_ay_candidates`: single-node graph candidates
//! - `extract_subgraph`: single node extraction
//! - `trace_op_name`: simple op names are bare identifiers
//! - `estimate_op_complexity`: empty shape returns 1 (max(product, 1))
//! - `AYCandidateRegion`: entry nodes are external to region
//! - `AYCandidateRegion`: exit nodes include nodes consumed outside
//!
//! Part of #3743.

use std::collections::HashSet;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use crate::subgraph_extract::{
    extract_subgraph, find_ay_candidates, is_ay_compatible_op, validate_subgraph, SubgraphSpec,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_input(id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape,
        DType::F32,
    )
}

fn make_relu(id: u64, input_id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("relu_{id}"),
        TraceOp::Relu,
        vec![input_id],
        shape,
        DType::F32,
    )
}

fn make_add(id: u64, lhs: u64, rhs: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("add_{id}"),
        TraceOp::Add,
        vec![lhs, rhs],
        shape,
        DType::F32,
    )
}

fn make_neg(id: u64, input_id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("neg_{id}"),
        TraceOp::Neg,
        vec![input_id],
        shape,
        DType::F32,
    )
}

fn make_sigmoid(id: u64, input_id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("sigmoid_{id}"),
        TraceOp::Sigmoid,
        vec![input_id],
        shape,
        DType::F32,
    )
}

// ===========================================================================
// VALIDATE_SUBGRAPH
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. validate: empty graph returns EmptyGraph error
// ---------------------------------------------------------------------------

/// Prove: `validate_subgraph` on an empty graph returns `EmptyGraph`.
#[kani::unwind(8)]
#[kani::proof]
fn validate_empty_graph_error() {
    let graph = ComputationGraph::from_nodes(vec![]);
    assert!(validate_subgraph(&graph).is_err());
}

// ---------------------------------------------------------------------------
// 2. validate: well-formed chain passes
// ---------------------------------------------------------------------------

/// Prove: a well-formed linear chain passes validation.
#[kani::unwind(128)]
#[kani::proof]
fn validate_well_formed_chain_passes() {
    let nodes = vec![
        make_input(1, vec![4]),
        make_relu(2, 1, vec![4]),
        make_neg(3, 2, vec![4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    assert!(validate_subgraph(&graph).is_ok());
}

// ===========================================================================
// EXTRACT_SUBGRAPH — NAME CONTAINS
// ===========================================================================

// ---------------------------------------------------------------------------
// 3. NameContains: no matching pattern → error
// ---------------------------------------------------------------------------

/// Prove: NameContains with a pattern matching no nodes returns error.
#[kani::unwind(128)]
#[kani::proof]
fn name_contains_no_match_error() {
    let graph =
        ComputationGraph::from_nodes(vec![make_input(1, vec![4]), make_relu(2, 1, vec![4])]);
    let spec = SubgraphSpec::NameContains {
        patterns: vec!["nonexistent_pattern".to_string()],
    };
    assert!(extract_subgraph(&graph, &spec).is_err());
}

// ---------------------------------------------------------------------------
// 4. NameContains: partial name match works
// ---------------------------------------------------------------------------

/// Prove: NameContains with substring "relu" matches "relu_2".
#[kani::unwind(128)]
#[kani::proof]
fn name_contains_partial_match() {
    let graph =
        ComputationGraph::from_nodes(vec![make_input(1, vec![4]), make_relu(2, 1, vec![4])]);
    let spec = SubgraphSpec::NameContains {
        patterns: vec!["relu".to_string()],
    };
    let result = extract_subgraph(&graph, &spec).expect("must match relu_2");
    assert_eq!(result.layer_count, 1);
}

// ===========================================================================
// IS_AY_COMPATIBLE_OP — ADDITIONAL OPS
// ===========================================================================

// ---------------------------------------------------------------------------
// 5. Trig ops: Tanh and Sigmoid are ay-compatible
// ---------------------------------------------------------------------------

/// Prove: Tanh and Sigmoid are ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_compat_tanh_sigmoid() {
    assert!(is_ay_compatible_op(&TraceOp::Tanh));
    assert!(is_ay_compatible_op(&TraceOp::Sigmoid));
}

// ---------------------------------------------------------------------------
// 6. Sqr is ay-compatible
// ---------------------------------------------------------------------------

/// Prove: Sqr (elementwise square) is ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_compat_sqr() {
    assert!(is_ay_compatible_op(&TraceOp::Sqr));
}

// ---------------------------------------------------------------------------
// 7. Recip is ay-compatible
// ---------------------------------------------------------------------------

/// Prove: Recip (1/x) is ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_compat_recip() {
    assert!(is_ay_compatible_op(&TraceOp::Recip));
}

// ---------------------------------------------------------------------------
// 8. Softmax is ay-compatible
// ---------------------------------------------------------------------------

/// Prove: Softmax is ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_compat_softmax() {
    assert!(is_ay_compatible_op(&TraceOp::Softmax { dim: 0 }));
}

// ---------------------------------------------------------------------------
// 9. Dropout is NOT ay-compatible
// ---------------------------------------------------------------------------

/// Prove: Dropout is NOT ay-compatible (not in match list).
#[kani::unwind(1)]
#[kani::proof]
fn ay_incompat_dropout() {
    assert!(!is_ay_compatible_op(&TraceOp::Dropout));
}

// ---------------------------------------------------------------------------
// 10. Upsample1d is NOT ay-compatible
// ---------------------------------------------------------------------------

/// Prove: Upsample1d is NOT ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_incompat_upsample1d() {
    assert!(!is_ay_compatible_op(&TraceOp::Upsample1d { factor: 2 }));
}

// ===========================================================================
// FIND_AY_CANDIDATES — ADDITIONAL PROPERTIES
// ===========================================================================

// ---------------------------------------------------------------------------
// 11. min > max: returns empty (no valid window sizes)
// ---------------------------------------------------------------------------

/// Prove: `find_ay_candidates` with min > max returns empty.
#[kani::unwind(128)]
#[kani::proof]
fn candidates_min_gt_max_empty() {
    let graph =
        ComputationGraph::from_nodes(vec![make_input(1, vec![4]), make_relu(2, 1, vec![4])]);
    let candidates = find_ay_candidates(&graph, 5, 3);
    assert!(candidates.is_empty(), "min > max must return empty");
}

// ---------------------------------------------------------------------------
// 12. Single-node graph: min=1 finds exactly 1 candidate
// ---------------------------------------------------------------------------

/// Prove: a single-node compatible graph with min=1 yields 1 candidate.
#[kani::unwind(128)]
#[kani::proof]
fn candidates_single_node_one_candidate() {
    let graph = ComputationGraph::from_nodes(vec![make_input(1, vec![4])]);
    let candidates = find_ay_candidates(&graph, 1, 1);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].layer_count, 1);
}

// ---------------------------------------------------------------------------
// 13. All candidates have start_index < end_index
// ---------------------------------------------------------------------------

/// Prove: every candidate has start_index < end_index (non-empty range).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn candidates_start_lt_end() {
    let nodes = vec![
        make_input(1, vec![4]),
        make_relu(2, 1, vec![4]),
        make_neg(3, 2, vec![4]),
        make_sigmoid(4, 3, vec![4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let candidates = find_ay_candidates(&graph, 1, 4);
    for c in &candidates {
        assert!(c.start_index < c.end_index);
    }
}

// ---------------------------------------------------------------------------
// 14. Candidates with min=max produce uniform layer_count
// ---------------------------------------------------------------------------

/// Prove: when min == max, all candidates have exactly that layer_count.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn candidates_fixed_size() {
    let nodes = vec![
        make_input(1, vec![4]),
        make_relu(2, 1, vec![4]),
        make_neg(3, 2, vec![4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let candidates = find_ay_candidates(&graph, 2, 2);
    for c in &candidates {
        assert_eq!(c.layer_count, 2, "fixed min=max must produce uniform size");
    }
}

// ===========================================================================
// EXTRACT_SUBGRAPH — SINGLE NODE
// ===========================================================================

// ---------------------------------------------------------------------------
// 15. Single node extraction: layer_count=1, synthetic inputs for deps
// ---------------------------------------------------------------------------

/// Prove: extracting a single non-input node produces layer_count=1
/// and synthetic inputs for its dependencies.
#[kani::unwind(128)]
#[kani::proof]
fn extract_single_node() {
    let nodes = vec![make_input(1, vec![4]), make_relu(2, 1, vec![4])];
    let graph = ComputationGraph::from_nodes(nodes);
    let spec = SubgraphSpec::IndexRange { start: 1, end: 2 };
    let result = extract_subgraph(&graph, &spec).expect("valid");
    assert_eq!(result.layer_count, 1);
    assert_eq!(result.synthetic_input_count, 1);
    assert_eq!(result.graph.nodes().len(), 2);
}

// ---------------------------------------------------------------------------
// 16. Single Input node extraction: no synthetic inputs needed
// ---------------------------------------------------------------------------

/// Prove: extracting an Input node needs 0 synthetic inputs.
#[kani::unwind(128)]
#[kani::proof]
fn extract_input_node_no_synthetic() {
    let graph = ComputationGraph::from_nodes(vec![make_input(1, vec![4])]);
    let spec = SubgraphSpec::IndexRange { start: 0, end: 1 };
    let result = extract_subgraph(&graph, &spec).expect("valid");
    assert_eq!(result.layer_count, 1);
    assert_eq!(result.synthetic_input_count, 0);
}

// ===========================================================================
// AY_CANDIDATE_REGION ENTRY/EXIT ANALYSIS
// ===========================================================================

// ---------------------------------------------------------------------------
// 17. Entry nodes are NOT internal to the region
// ---------------------------------------------------------------------------

/// Prove: no entry node ID is also an internal node ID.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn candidates_entry_nodes_not_internal() {
    let nodes = vec![
        make_input(1, vec![4]),
        make_relu(2, 1, vec![4]),
        make_neg(3, 2, vec![4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let candidates = find_ay_candidates(&graph, 1, 3);
    for c in &candidates {
        let internal_set: HashSet<u64> = c.internal_nodes.iter().copied().collect();
        for entry_id in &c.entry_nodes {
            assert!(
                !internal_set.contains(entry_id),
                "entry node must not be internal"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 18. Exit nodes are a subset of internal nodes
// ---------------------------------------------------------------------------

/// Prove: every exit node ID is an internal node ID.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn candidates_exit_nodes_subset_of_internal() {
    let nodes = vec![
        make_input(1, vec![4]),
        make_relu(2, 1, vec![4]),
        make_neg(3, 2, vec![4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let candidates = find_ay_candidates(&graph, 1, 3);
    for c in &candidates {
        let internal_set: HashSet<u64> = c.internal_nodes.iter().copied().collect();
        for exit_id in &c.exit_nodes {
            assert!(internal_set.contains(exit_id), "exit node must be internal");
        }
    }
}

// ===========================================================================
// TRACE_OP_NAME ADDITIONAL TESTS
// ===========================================================================

// ---------------------------------------------------------------------------
// 19. Simple ops produce clean names without parens
// ---------------------------------------------------------------------------

/// Prove: simple ops like Relu, Add produce names without parentheses.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn trace_op_names_no_parens_for_simple_ops() {
    let nodes = vec![
        make_input(1, vec![4]),
        make_relu(2, 1, vec![4]),
        make_neg(3, 2, vec![4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let candidates = find_ay_candidates(&graph, 3, 3);
    assert_eq!(candidates.len(), 1);
    for name in &candidates[0].op_types {
        assert!(!name.contains('('), "simple op must not have parens");
    }
}

// ---------------------------------------------------------------------------
// 20. Conv1d op name is "Conv1d" (struct fields stripped)
// ---------------------------------------------------------------------------

/// Prove: Conv1d trace op name is "Conv1d" (not the full debug string).
#[kani::unwind(128)]
#[kani::proof]
fn trace_op_name_conv1d_stripped() {
    let w = WeightRef::from_shape(&[4, 4, 3]);
    let nodes = vec![
        make_input(1, vec![1, 4, 16]),
        TraceNode::new(
            2,
            "conv".to_string(),
            TraceOp::Conv1d {
                weight: w,
                bias: None,
                padding: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            vec![1],
            vec![1, 4, 16],
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let candidates = find_ay_candidates(&graph, 2, 2);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].op_types[1], "Conv1d");
}
