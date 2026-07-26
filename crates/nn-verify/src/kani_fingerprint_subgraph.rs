// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani verification harnesses for `fingerprint.rs` and `subgraph_extract.rs`.
//!
//! Proves safety properties of:
//! - Fingerprint determinism (same input => same hash)
//! - Fingerprint hash digest assembly (32-byte output)
//! - `diff_fingerprints` region detection correctness
//! - `classify_change` classification consistency
//! - `resolve_spec` index bounds validation
//! - `extract_subgraph` ID remapping correctness
//! - `validate_subgraph` self-containment checks
//! - `is_ay_compatible_op` classification completeness
//! - `find_ay_candidates` window bounds safety
//! - `estimate_op_complexity` non-zero output
//!
//! Part of #3629.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use crate::fingerprint::{
    diff_fingerprints, fingerprint_trace, fingerprint_trace_with_weights, ChangeReason,
    ChangedRegion, SubgraphFingerprint,
};
use crate::subgraph_extract::{
    extract_subgraph, find_ay_candidates, is_ay_compatible_op, validate_subgraph,
    ExtractedSubgraph, SubgraphSpec,
};

// ---------------------------------------------------------------------------
// Helper: build a small TraceNode for Kani-bounded graphs.
// ---------------------------------------------------------------------------

/// Build a simple Input TraceNode with given ID and shape.
fn make_input_node(id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape,
        DType::F32,
    )
}

/// Build a simple Relu TraceNode that depends on `input_id`.
fn make_relu_node(id: u64, input_id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("relu_{id}"),
        TraceOp::Relu,
        vec![input_id],
        shape,
        DType::F32,
    )
}

/// Build an Add TraceNode that depends on two inputs.
fn make_add_node(id: u64, lhs: u64, rhs: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("add_{id}"),
        TraceOp::Add,
        vec![lhs, rhs],
        shape,
        DType::F32,
    )
}

/// Build a Neg TraceNode (element-wise, ay-compatible).
fn make_neg_node(id: u64, input_id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("neg_{id}"),
        TraceOp::Neg,
        vec![input_id],
        shape,
        DType::F32,
    )
}

// ===========================================================================
// FINGERPRINT HARNESSES
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. fingerprint_trace determinism: same nodes => same fingerprints
// ---------------------------------------------------------------------------

/// Prove: `fingerprint_trace` is deterministic — calling it twice on the same
/// node slice produces identical fingerprints (same hash, same indices).
#[kani::unwind(128)]
#[kani::proof]
fn fingerprint_trace_deterministic_single_node() {
    let node = make_input_node(1, vec![1, 3, 224]);
    let nodes = vec![node];
    let fp1 = fingerprint_trace(&nodes);
    let fp2 = fingerprint_trace(&nodes);
    assert_eq!(fp1.len(), fp2.len(), "same input must produce same count");
    assert_eq!(fp1[0].hash, fp2[0].hash, "same node must produce same hash");
    assert_eq!(
        fp1[0].node_indices, fp2[0].node_indices,
        "same node must have same indices"
    );
}

// ---------------------------------------------------------------------------
// 2. fingerprint_trace output count equals input count
// ---------------------------------------------------------------------------

/// Prove: `fingerprint_trace` returns exactly one fingerprint per node.
///
/// Bounded to 0..=3 nodes for tractability.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn fingerprint_trace_count_equals_node_count() {
    let n: usize = kani::any();
    kani::assume(n <= 3);

    let mut nodes = Vec::new();
    for i in 0..n {
        let id = (i + 1) as u64;
        if i == 0 {
            nodes.push(make_input_node(id, vec![1]));
        } else {
            nodes.push(make_relu_node(id, i as u64, vec![1]));
        }
    }

    let fps = fingerprint_trace(&nodes);
    assert_eq!(fps.len(), n, "fingerprint count must equal node count");
}

// ---------------------------------------------------------------------------
// 3. Each fingerprint has exactly one node_index pointing to its position
// ---------------------------------------------------------------------------

/// Prove: each `SubgraphFingerprint` from `fingerprint_trace` has
/// `node_indices == [i]` for position `i`.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn fingerprint_trace_indices_are_sequential() {
    let nodes = vec![
        make_input_node(1, vec![4]),
        make_relu_node(2, 1, vec![4]),
        make_neg_node(3, 2, vec![4]),
    ];
    let fps = fingerprint_trace(&nodes);
    for (i, fp) in fps.iter().enumerate() {
        assert_eq!(fp.node_indices.len(), 1, "must have exactly one index");
        assert_eq!(fp.node_indices[0], i, "index must match position");
    }
}

// ---------------------------------------------------------------------------
// 4. fingerprint_trace_with_weights also returns correct count
// ---------------------------------------------------------------------------

/// Prove: `fingerprint_trace_with_weights` returns one fingerprint per node,
/// identical to `fingerprint_trace` in structure.
#[kani::unwind(128)]
#[kani::proof]
fn fingerprint_with_weights_count_matches() {
    let nodes = vec![
        make_input_node(1, vec![2, 3]),
        make_relu_node(2, 1, vec![2, 3]),
    ];
    let fps = fingerprint_trace_with_weights(&nodes);
    assert_eq!(fps.len(), 2, "must match node count");
    assert_eq!(fps[0].node_indices, vec![0]);
    assert_eq!(fps[1].node_indices, vec![1]);
}

// ---------------------------------------------------------------------------
// 5. Structural vs parametric: same architecture without weights => same hash
// ---------------------------------------------------------------------------

/// Prove: for `Input` nodes (no weights), structural and parametric
/// fingerprints produce the same hash.
#[kani::unwind(128)]
#[kani::proof]
fn structural_and_parametric_match_for_input_nodes() {
    let nodes = vec![make_input_node(1, vec![8])];
    let structural = fingerprint_trace(&nodes);
    let parametric = fingerprint_trace_with_weights(&nodes);
    assert_eq!(
        structural[0].hash, parametric[0].hash,
        "Input nodes have no weights; structural == parametric"
    );
}

// ---------------------------------------------------------------------------
// 6. Different ops produce different hashes
// ---------------------------------------------------------------------------

/// Prove: a Relu node and a Neg node with the same shape produce different
/// hashes (op type is included in the fingerprint).
#[kani::unwind(128)]
#[kani::proof]
fn different_ops_produce_different_hashes() {
    let relu = vec![make_relu_node(2, 1, vec![4])];
    let neg = vec![make_neg_node(2, 1, vec![4])];
    let fp_relu = fingerprint_trace(&relu);
    let fp_neg = fingerprint_trace(&neg);
    assert_ne!(
        fp_relu[0].hash, fp_neg[0].hash,
        "different ops must produce different hashes"
    );
}

// ---------------------------------------------------------------------------
// 7. Different shapes produce different hashes
// ---------------------------------------------------------------------------

/// Prove: same op type but different output shapes produce different hashes.
#[kani::unwind(128)]
#[kani::proof]
fn different_shapes_produce_different_hashes() {
    let small = vec![make_input_node(1, vec![4])];
    let large = vec![make_input_node(1, vec![8])];
    let fp_small = fingerprint_trace(&small);
    let fp_large = fingerprint_trace(&large);
    assert_ne!(
        fp_small[0].hash, fp_large[0].hash,
        "different shapes must produce different hashes"
    );
}

// ---------------------------------------------------------------------------
// 8. diff_fingerprints: identical inputs => no changes
// ---------------------------------------------------------------------------

/// Prove: `diff_fingerprints` returns empty when old and new are identical.
#[kani::unwind(128)]
#[kani::proof]
fn diff_identical_returns_empty() {
    let nodes = vec![make_input_node(1, vec![4]), make_relu_node(2, 1, vec![4])];
    let fps = fingerprint_trace(&nodes);
    let changes = diff_fingerprints(&fps, &fps);
    assert!(
        changes.is_empty(),
        "identical fingerprints must have 0 changes"
    );
}

// ---------------------------------------------------------------------------
// 9. diff_fingerprints: empty old => all inserted
// ---------------------------------------------------------------------------

/// Prove: diffing an empty old with a non-empty new reports all as Inserted.
#[kani::unwind(128)]
#[kani::proof]
fn diff_empty_old_reports_inserted() {
    let nodes = vec![make_input_node(1, vec![4])];
    let fps = fingerprint_trace(&nodes);
    let changes = diff_fingerprints(&[], &fps);
    assert_eq!(changes.len(), 1, "must have exactly one changed region");
    assert_eq!(changes[0].start, 0);
    assert_eq!(changes[0].end, 1);
    assert_eq!(changes[0].reason, ChangeReason::Inserted);
}

// ---------------------------------------------------------------------------
// 10. diff_fingerprints: empty new => all removed
// ---------------------------------------------------------------------------

/// Prove: diffing a non-empty old with an empty new reports all as Removed.
#[kani::unwind(128)]
#[kani::proof]
fn diff_empty_new_reports_removed() {
    let nodes = vec![make_input_node(1, vec![4])];
    let fps = fingerprint_trace(&nodes);
    let changes = diff_fingerprints(&fps, &[]);
    assert_eq!(changes.len(), 1, "must have exactly one changed region");
    assert_eq!(changes[0].start, 0);
    assert_eq!(changes[0].end, 1);
    assert_eq!(changes[0].reason, ChangeReason::Removed);
}

// ---------------------------------------------------------------------------
// 11. diff_fingerprints: both empty => no changes
// ---------------------------------------------------------------------------

/// Prove: diffing two empty fingerprint lists returns no changes.
#[kani::unwind(1)]
#[kani::proof]
fn diff_both_empty_returns_empty() {
    let changes = diff_fingerprints(&[], &[]);
    assert!(changes.is_empty(), "both empty => no changes");
}

// ---------------------------------------------------------------------------
// 12. diff_fingerprints: changed region has valid bounds
// ---------------------------------------------------------------------------

/// Prove: every ChangedRegion has start < end (non-empty region).
#[kani::unwind(128)]
#[kani::proof]
fn diff_changed_region_start_less_than_end() {
    // Old: Input; New: Relu (different op => different hash => changed)
    let old_nodes = vec![make_input_node(1, vec![4])];
    let new_nodes = vec![make_relu_node(1, 0, vec![4])];
    let old_fps = fingerprint_trace(&old_nodes);
    let new_fps = fingerprint_trace(&new_nodes);
    let changes = diff_fingerprints(&old_fps, &new_fps);
    for region in &changes {
        assert!(region.start < region.end, "start must be < end");
    }
}

// ---------------------------------------------------------------------------
// 13. classify_change: different op_summary => OpChanged
// ---------------------------------------------------------------------------

/// Prove: `classify_change` returns `OpChanged` when op_summary differs.
#[kani::unwind(128)]
#[kani::proof]
fn classify_change_op_changed() {
    let old_nodes = vec![make_input_node(1, vec![4])];
    let new_nodes = vec![make_relu_node(1, 0, vec![4])];
    let old_fps = fingerprint_trace(&old_nodes);
    let new_fps = fingerprint_trace(&new_nodes);
    let changes = diff_fingerprints(&old_fps, &new_fps);
    assert!(!changes.is_empty());
    assert_eq!(
        changes[0].reason,
        ChangeReason::OpChanged,
        "different ops must classify as OpChanged"
    );
}

// ---------------------------------------------------------------------------
// 14. classify_change: same op but different shape => WeightChanged
// ---------------------------------------------------------------------------

/// Prove: when op_summary is the same but shapes differ, `classify_change`
/// reports `WeightChanged` (the conservative fallback for same-op changes).
#[kani::unwind(128)]
#[kani::proof]
fn classify_change_weight_changed_for_shape_diff() {
    let old_nodes = vec![make_input_node(1, vec![4])];
    let new_nodes = vec![make_input_node(1, vec![8])];
    let old_fps = fingerprint_trace(&old_nodes);
    let new_fps = fingerprint_trace(&new_nodes);
    let changes = diff_fingerprints(&old_fps, &new_fps);
    assert!(!changes.is_empty());
    // Same canonical_name ("input") but different hash => WeightChanged
    assert_eq!(changes[0].reason, ChangeReason::WeightChanged);
}

// ===========================================================================
// SUBGRAPH EXTRACTION HARNESSES
// ===========================================================================

// ---------------------------------------------------------------------------
// 15. resolve_spec IndexRange: out-of-bounds start => error
// ---------------------------------------------------------------------------

/// Prove: `extract_subgraph` with start >= nodes.len() returns an error.
#[kani::unwind(128)]
#[kani::proof]
fn extract_subgraph_oob_start_returns_error() {
    let nodes = vec![make_input_node(1, vec![4])];
    let graph = ComputationGraph::from_nodes(nodes);
    let spec = SubgraphSpec::IndexRange { start: 5, end: 6 };
    assert!(
        extract_subgraph(&graph, &spec).is_err(),
        "start beyond graph length must fail"
    );
}

// ---------------------------------------------------------------------------
// 16. resolve_spec IndexRange: end > nodes.len() => error
// ---------------------------------------------------------------------------

/// Prove: `extract_subgraph` with end > nodes.len() returns an error.
#[kani::unwind(128)]
#[kani::proof]
fn extract_subgraph_oob_end_returns_error() {
    let nodes = vec![make_input_node(1, vec![4])];
    let graph = ComputationGraph::from_nodes(nodes);
    let spec = SubgraphSpec::IndexRange { start: 0, end: 5 };
    assert!(
        extract_subgraph(&graph, &spec).is_err(),
        "end beyond graph length must fail"
    );
}

// ---------------------------------------------------------------------------
// 17. resolve_spec IndexRange: start >= end => error
// ---------------------------------------------------------------------------

/// Prove: `extract_subgraph` with start >= end (empty range) returns an error.
#[kani::unwind(128)]
#[kani::proof]
fn extract_subgraph_empty_range_returns_error() {
    let nodes = vec![make_input_node(1, vec![4]), make_relu_node(2, 1, vec![4])];
    let graph = ComputationGraph::from_nodes(nodes);
    let spec = SubgraphSpec::IndexRange { start: 1, end: 1 };
    assert!(
        extract_subgraph(&graph, &spec).is_err(),
        "start == end (empty range) must fail"
    );
}

// ---------------------------------------------------------------------------
// 18. extract_subgraph: valid range produces correct layer_count
// ---------------------------------------------------------------------------

/// Prove: extracting a valid range produces `layer_count == end - start`.
#[kani::unwind(128)]
#[kani::proof]
fn extract_subgraph_layer_count_correct() {
    let nodes = vec![
        make_input_node(1, vec![4]),
        make_relu_node(2, 1, vec![4]),
        make_neg_node(3, 2, vec![4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let spec = SubgraphSpec::IndexRange { start: 0, end: 2 };
    let result = extract_subgraph(&graph, &spec).expect("valid range");
    assert_eq!(result.layer_count, 2, "layer_count must equal end - start");
}

// ---------------------------------------------------------------------------
// 19. extract_subgraph: synthetic input count for external dependencies
// ---------------------------------------------------------------------------

/// Prove: extracting a subgraph that depends on an external node creates
/// exactly one synthetic input.
#[kani::unwind(128)]
#[kani::proof]
fn extract_subgraph_synthetic_input_for_external_dep() {
    // Graph: Input(1) -> Relu(2) -> Neg(3)
    // Extract only Neg(3), which depends on Relu(2) not in range.
    let nodes = vec![
        make_input_node(1, vec![4]),
        make_relu_node(2, 1, vec![4]),
        make_neg_node(3, 2, vec![4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let spec = SubgraphSpec::IndexRange { start: 2, end: 3 };
    let result = extract_subgraph(&graph, &spec).expect("valid range");
    assert_eq!(
        result.synthetic_input_count, 1,
        "one external dep => one synthetic input"
    );
}

// ---------------------------------------------------------------------------
// 20. extract_subgraph: self-contained range has zero synthetic inputs
// ---------------------------------------------------------------------------

/// Prove: extracting a range starting at Input nodes has zero synthetic inputs.
#[kani::unwind(128)]
#[kani::proof]
fn extract_subgraph_self_contained_no_synthetic() {
    let nodes = vec![make_input_node(1, vec![4]), make_relu_node(2, 1, vec![4])];
    let graph = ComputationGraph::from_nodes(nodes);
    let spec = SubgraphSpec::IndexRange { start: 0, end: 2 };
    let result = extract_subgraph(&graph, &spec).expect("valid range");
    assert_eq!(
        result.synthetic_input_count, 0,
        "self-contained range must have 0 synthetic inputs"
    );
}

// ---------------------------------------------------------------------------
// 21. extract_subgraph: total subgraph node count = layers + synthetic inputs
// ---------------------------------------------------------------------------

/// Prove: the extracted subgraph has exactly `layer_count + synthetic_input_count` nodes.
#[kani::unwind(128)]
#[kani::proof]
fn extract_subgraph_total_nodes_consistent() {
    let nodes = vec![
        make_input_node(1, vec![4]),
        make_relu_node(2, 1, vec![4]),
        make_neg_node(3, 2, vec![4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let spec = SubgraphSpec::IndexRange { start: 1, end: 3 };
    let result = extract_subgraph(&graph, &spec).expect("valid range");
    let total = result.graph.nodes().len();
    assert_eq!(
        total,
        result.layer_count + result.synthetic_input_count,
        "total nodes = layers + synthetic inputs"
    );
}

// ---------------------------------------------------------------------------
// 22. extract_subgraph: ID map has entries for all nodes
// ---------------------------------------------------------------------------

/// Prove: the `id_map` covers all original node IDs in the extracted range
/// plus all synthetic input IDs.
#[kani::unwind(128)]
#[kani::proof]
fn extract_subgraph_id_map_complete() {
    let nodes = vec![
        make_input_node(1, vec![4]),
        make_relu_node(2, 1, vec![4]),
        make_neg_node(3, 2, vec![4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let spec = SubgraphSpec::IndexRange { start: 2, end: 3 };
    let result = extract_subgraph(&graph, &spec).expect("valid range");
    // Original node 3 must be in the map.
    assert!(
        result.id_map.contains_key(&3),
        "extracted node must be in id_map"
    );
    // External dep node 2 must also be in the map (as synthetic input).
    assert!(
        result.id_map.contains_key(&2),
        "external dep must be in id_map"
    );
}

// ---------------------------------------------------------------------------
// 23. validate_subgraph: empty graph returns error
// ---------------------------------------------------------------------------

/// Prove: `validate_subgraph` rejects an empty `ComputationGraph`.
#[kani::unwind(8)]
#[kani::proof]
fn validate_subgraph_rejects_empty() {
    let graph = ComputationGraph::from_nodes(vec![]);
    assert!(
        validate_subgraph(&graph).is_err(),
        "empty graph must fail validation"
    );
}

// ---------------------------------------------------------------------------
// 24. validate_subgraph: self-contained graph passes
// ---------------------------------------------------------------------------

/// Prove: a well-formed self-contained graph passes `validate_subgraph`.
#[kani::unwind(128)]
#[kani::proof]
fn validate_subgraph_accepts_self_contained() {
    let nodes = vec![make_input_node(1, vec![4]), make_relu_node(2, 1, vec![4])];
    let graph = ComputationGraph::from_nodes(nodes);
    assert!(
        validate_subgraph(&graph).is_ok(),
        "self-contained graph must pass validation"
    );
}

// ---------------------------------------------------------------------------
// 25. is_ay_compatible_op: Input is always compatible
// ---------------------------------------------------------------------------

/// Prove: `TraceOp::Input` is always ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn is_ay_compatible_input() {
    assert!(
        is_ay_compatible_op(&TraceOp::Input),
        "Input must be ay-compatible"
    );
}

// ---------------------------------------------------------------------------
// 26. is_ay_compatible_op: element-wise ops are compatible
// ---------------------------------------------------------------------------

/// Prove: core element-wise ops (Add, Sub, Mul, Relu, etc.) are ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn is_ay_compatible_elementwise_ops() {
    assert!(is_ay_compatible_op(&TraceOp::Add));
    assert!(is_ay_compatible_op(&TraceOp::Sub));
    assert!(is_ay_compatible_op(&TraceOp::Mul));
    assert!(is_ay_compatible_op(&TraceOp::Div));
    assert!(is_ay_compatible_op(&TraceOp::Neg));
    assert!(is_ay_compatible_op(&TraceOp::Abs));
    assert!(is_ay_compatible_op(&TraceOp::Relu));
    assert!(is_ay_compatible_op(&TraceOp::Exp));
    assert!(is_ay_compatible_op(&TraceOp::Log));
    assert!(is_ay_compatible_op(&TraceOp::Sqrt));
    assert!(is_ay_compatible_op(&TraceOp::Sqr));
    assert!(is_ay_compatible_op(&TraceOp::Recip));
    assert!(is_ay_compatible_op(&TraceOp::Sin));
    assert!(is_ay_compatible_op(&TraceOp::Cos));
    assert!(is_ay_compatible_op(&TraceOp::Tanh));
    assert!(is_ay_compatible_op(&TraceOp::Sigmoid));
}

// ---------------------------------------------------------------------------
// 27. is_ay_compatible_op: LSTM is NOT compatible
// ---------------------------------------------------------------------------

/// Prove: `TraceOp::Lstm` is never ay-compatible (data-dependent recurrence).
#[kani::unwind(1)]
#[kani::proof]
fn is_ay_incompatible_lstm() {
    use nn_core::dyn_tensor::trace::WeightRef;
    let w = WeightRef::from_shape(&[4, 4]);
    let op = TraceOp::Lstm {
        weight_ih: w.clone(),
        weight_hh: w.clone(),
        bias_ih: None,
        bias_hh: None,
        hidden_size: 4,
        initial_hidden: None,
        initial_cell: None,
    };
    assert!(!is_ay_compatible_op(&op), "LSTM must NOT be ay-compatible");
}

// ---------------------------------------------------------------------------
// 28. find_ay_candidates: empty graph returns no candidates
// ---------------------------------------------------------------------------

/// Prove: `find_ay_candidates` on an empty graph returns empty candidates.
#[kani::unwind(8)]
#[kani::proof]
fn find_ay_candidates_empty_graph() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let candidates = find_ay_candidates(&graph, 1, 5);
    assert!(candidates.is_empty(), "empty graph => no candidates");
}

// ---------------------------------------------------------------------------
// 29. find_ay_candidates: single incompatible node => no candidates
// ---------------------------------------------------------------------------

/// Prove: a graph with only LSTM nodes returns no ay candidates.
#[kani::unwind(64)]
#[kani::proof]
fn find_ay_candidates_all_incompatible() {
    use nn_core::dyn_tensor::trace::WeightRef;
    let w = WeightRef::from_shape(&[4, 4]);
    let lstm_node = TraceNode::new(
        1,
        "lstm_0".to_string(),
        TraceOp::Lstm {
            weight_ih: w.clone(),
            weight_hh: w.clone(),
            bias_ih: None,
            bias_hh: None,
            hidden_size: 4,
            initial_hidden: None,
            initial_cell: None,
        },
        vec![],
        vec![1, 4],
        DType::F32,
    );
    let graph = ComputationGraph::from_nodes(vec![lstm_node]);
    let candidates = find_ay_candidates(&graph, 1, 5);
    assert!(candidates.is_empty(), "all-LSTM graph => no candidates");
}

// ---------------------------------------------------------------------------
// 30. find_ay_candidates: candidate region bounds are valid
// ---------------------------------------------------------------------------

/// Prove: every ay candidate region has `start_index < end_index` and
/// `end_index <= nodes.len()`.
#[kani::unwind(128)]
#[kani::proof]
fn find_ay_candidates_valid_bounds() {
    let nodes = vec![
        make_input_node(1, vec![4]),
        make_relu_node(2, 1, vec![4]),
        make_neg_node(3, 2, vec![4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let candidates = find_ay_candidates(&graph, 1, 3);
    for c in &candidates {
        assert!(c.start_index < c.end_index, "start must be < end");
        assert!(
            c.end_index <= graph.nodes().len(),
            "end must be <= graph size"
        );
        assert_eq!(
            c.layer_count,
            c.end_index - c.start_index,
            "layer_count must equal range size"
        );
    }
}

// ---------------------------------------------------------------------------
// 31. find_ay_candidates: min_layers filter works
// ---------------------------------------------------------------------------

/// Prove: no candidate has `layer_count < min_layers`.
#[kani::unwind(128)]
#[kani::proof]
fn find_ay_candidates_respects_min_layers() {
    let nodes = vec![
        make_input_node(1, vec![4]),
        make_relu_node(2, 1, vec![4]),
        make_neg_node(3, 2, vec![4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let min = 2;
    let candidates = find_ay_candidates(&graph, min, 5);
    for c in &candidates {
        assert!(
            c.layer_count >= min,
            "no candidate should be smaller than min_layers"
        );
    }
}

// ---------------------------------------------------------------------------
// 32. estimate_op_complexity: always returns >= 1
// ---------------------------------------------------------------------------

/// Prove: `estimate_op_complexity` never returns 0 for any op with a
/// non-empty output shape (due to `.max(1)` in the implementation).
#[kani::unwind(128)]
#[kani::proof]
fn estimate_op_complexity_nonzero() {
    // Relu with shape [4] => output_elements = 4
    let nodes = vec![make_input_node(1, vec![4]), make_relu_node(2, 1, vec![4])];
    let graph = ComputationGraph::from_nodes(nodes);
    let candidates = find_ay_candidates(&graph, 1, 2);
    for c in &candidates {
        assert!(c.estimated_complexity > 0, "complexity must be > 0");
    }
}

// ---------------------------------------------------------------------------
// 33. SubgraphSpec::NodeIds: no matching IDs => error
// ---------------------------------------------------------------------------

/// Prove: `extract_subgraph` with NodeIds that don't exist returns an error.
#[kani::unwind(128)]
#[kani::proof]
fn extract_subgraph_no_matching_ids_returns_error() {
    let nodes = vec![make_input_node(1, vec![4])];
    let graph = ComputationGraph::from_nodes(nodes);
    let spec = SubgraphSpec::NodeIds {
        ids: vec![999, 1000],
    };
    assert!(
        extract_subgraph(&graph, &spec).is_err(),
        "non-existent IDs must fail"
    );
}

// ---------------------------------------------------------------------------
// 34. SubgraphSpec::NameContains: no matching names => error
// ---------------------------------------------------------------------------

/// Prove: `extract_subgraph` with name patterns that match nothing returns error.
#[kani::unwind(128)]
#[kani::proof]
fn extract_subgraph_no_matching_names_returns_error() {
    let nodes = vec![make_input_node(1, vec![4])];
    let graph = ComputationGraph::from_nodes(nodes);
    let spec = SubgraphSpec::NameContains {
        patterns: vec!["nonexistent_xyz".to_string()],
    };
    assert!(
        extract_subgraph(&graph, &spec).is_err(),
        "unmatched name patterns must fail"
    );
}

// ---------------------------------------------------------------------------
// 35. extract + validate roundtrip: extracted subgraph passes validation
// ---------------------------------------------------------------------------

/// Prove: a subgraph extracted via `extract_subgraph` always passes
/// `validate_subgraph` (extraction produces self-contained graphs).
#[kani::unwind(128)]
#[kani::proof]
fn extract_then_validate_roundtrip() {
    let nodes = vec![
        make_input_node(1, vec![4]),
        make_relu_node(2, 1, vec![4]),
        make_neg_node(3, 2, vec![4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let spec = SubgraphSpec::IndexRange { start: 1, end: 3 };
    let result = extract_subgraph(&graph, &spec).expect("valid extraction");
    assert!(
        validate_subgraph(&result.graph).is_ok(),
        "extracted subgraph must pass validation"
    );
}

// ---------------------------------------------------------------------------
// 36. ChangeReason Display: all variants produce non-empty strings
// ---------------------------------------------------------------------------

/// Prove: every `ChangeReason` variant's `Display` output is non-empty.
#[kani::unwind(64)]
#[kani::proof]
fn change_reason_display_nonempty() {
    let reasons = [
        ChangeReason::OpChanged,
        ChangeReason::ShapeChanged,
        ChangeReason::WeightChanged,
        ChangeReason::Inserted,
        ChangeReason::Removed,
    ];
    for r in &reasons {
        let s = format!("{r}");
        assert!(!s.is_empty(), "ChangeReason Display must be non-empty");
    }
}

// ---------------------------------------------------------------------------
// 37. diff_fingerprints: new longer => trailing region is Inserted
// ---------------------------------------------------------------------------

/// Prove: when new has more nodes than old, the trailing region is Inserted
/// with correct bounds (start = old.len(), end = new.len()).
#[kani::unwind(128)]
#[kani::proof]
fn diff_new_longer_has_inserted_tail() {
    let old_nodes = vec![make_input_node(1, vec![4])];
    let new_nodes = vec![make_input_node(1, vec![4]), make_relu_node(2, 1, vec![4])];
    let old_fps = fingerprint_trace(&old_nodes);
    let new_fps = fingerprint_trace(&new_nodes);
    let changes = diff_fingerprints(&old_fps, &new_fps);
    // The common part matches (both index 0 is Input with shape [4]).
    // Trailing: index 1 in new has no counterpart in old.
    assert!(!changes.is_empty());
    let last = changes.last().unwrap();
    assert_eq!(last.reason, ChangeReason::Inserted);
    assert_eq!(last.start, 1);
    assert_eq!(last.end, 2);
}

// ---------------------------------------------------------------------------
// 38. SubgraphFingerprint hash is exactly 32 bytes
// ---------------------------------------------------------------------------

/// Prove: every fingerprint hash is exactly 32 bytes (SHA-256 digest size).
#[kani::unwind(128)]
#[kani::proof]
fn fingerprint_hash_is_32_bytes() {
    let nodes = vec![make_input_node(1, vec![2, 3])];
    let fps = fingerprint_trace(&nodes);
    assert_eq!(fps[0].hash.len(), 32, "SHA-256 digest must be 32 bytes");
}

// ---------------------------------------------------------------------------
// 39. extract_subgraph: NodeIds with valid IDs succeeds
// ---------------------------------------------------------------------------

/// Prove: `extract_subgraph` with `NodeIds` that exist in the graph succeeds
/// and produces the correct layer count.
#[kani::unwind(128)]
#[kani::proof]
fn extract_subgraph_node_ids_valid() {
    let nodes = vec![
        make_input_node(1, vec![4]),
        make_relu_node(2, 1, vec![4]),
        make_neg_node(3, 2, vec![4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let spec = SubgraphSpec::NodeIds { ids: vec![1, 2] };
    let result = extract_subgraph(&graph, &spec).expect("valid IDs");
    assert_eq!(
        result.layer_count, 2,
        "two selected node IDs => layer_count 2"
    );
}

// ---------------------------------------------------------------------------
// 40. extract_subgraph: NameContains with valid pattern succeeds
// ---------------------------------------------------------------------------

/// Prove: `extract_subgraph` with a name pattern matching existing nodes
/// succeeds.
#[kani::unwind(128)]
#[kani::proof]
fn extract_subgraph_name_contains_valid() {
    let nodes = vec![make_input_node(1, vec![4]), make_relu_node(2, 1, vec![4])];
    let graph = ComputationGraph::from_nodes(nodes);
    let spec = SubgraphSpec::NameContains {
        patterns: vec!["relu".to_string()],
    };
    let result = extract_subgraph(&graph, &spec).expect("valid pattern");
    assert_eq!(result.layer_count, 1, "one relu node matched");
}

// ---------------------------------------------------------------------------
// 41. find_ay_candidates: layer_count bounded by max_layers
// ---------------------------------------------------------------------------

/// Prove: no ay candidate has `layer_count > max_layers`.
#[kani::unwind(128)]
#[kani::proof]
fn find_ay_candidates_respects_max_layers() {
    let nodes = vec![
        make_input_node(1, vec![4]),
        make_relu_node(2, 1, vec![4]),
        make_neg_node(3, 2, vec![4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let max = 2;
    let candidates = find_ay_candidates(&graph, 1, max);
    for c in &candidates {
        assert!(
            c.layer_count <= max,
            "no candidate should exceed max_layers"
        );
    }
}

// ---------------------------------------------------------------------------
// 42. extract_subgraph: id_map values are sequential from 1
// ---------------------------------------------------------------------------

/// Prove: the remapped IDs in the extracted subgraph start at 1 and are
/// sequential (no gaps), as required by NY graph translation.
#[kani::unwind(128)]
#[kani::proof]
fn extract_subgraph_id_map_sequential() {
    let nodes = vec![make_input_node(1, vec![4]), make_relu_node(2, 1, vec![4])];
    let graph = ComputationGraph::from_nodes(nodes);
    let spec = SubgraphSpec::IndexRange { start: 0, end: 2 };
    let result = extract_subgraph(&graph, &spec).expect("valid range");

    // Collect and sort the remapped IDs.
    let mut remapped: Vec<u64> = result.id_map.values().copied().collect();
    remapped.sort();

    // Must be [1, 2] for a 2-node extraction with 0 synthetic inputs.
    assert_eq!(remapped.len(), 2);
    assert_eq!(remapped[0], 1, "first remapped ID must be 1");
    assert_eq!(remapped[1], 2, "second remapped ID must be 2");
}
