// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani verification harnesses for `subgraph_extract.rs`.
//!
//! Proves safety and correctness properties of:
//! - `resolve_spec` index bounds and error conditions
//! - `extract_subgraph` ID remapping, synthetic input generation, invariants
//! - `validate_subgraph` self-containment checking
//! - `find_ay_candidates` sliding window correctness, bounds, and coverage
//! - `is_ay_compatible_op` classification for shape/reduction/trig ops
//! - `estimate_op_complexity` formula correctness for weighted ops
//! - `build_candidate_region` entry/exit node detection
//! - `trace_op_name` variant name extraction
//!
//! Part of #3682.

use std::collections::HashSet;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use crate::subgraph_extract::{
    extract_subgraph, find_ay_candidates, is_ay_compatible_op, validate_subgraph, SubgraphSpec,
    AYCandidateRegion,
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

fn make_cos(id: u64, input_id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("cos_{id}"),
        TraceOp::Cos,
        vec![input_id],
        shape,
        DType::F32,
    )
}

fn make_sin(id: u64, input_id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("sin_{id}"),
        TraceOp::Sin,
        vec![input_id],
        shape,
        DType::F32,
    )
}

fn make_mul(id: u64, lhs: u64, rhs: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("mul_{id}"),
        TraceOp::Mul,
        vec![lhs, rhs],
        shape,
        DType::F32,
    )
}

fn make_lstm(id: u64, input_id: u64, shape: Vec<usize>) -> TraceNode {
    let w_ih = WeightRef::from_shape(&[16, 4]);
    let w_hh = WeightRef::from_shape(&[16, 4]);
    TraceNode::new(
        id,
        format!("lstm_{id}"),
        TraceOp::Lstm {
            weight_ih: w_ih,
            weight_hh: w_hh,
            bias_ih: None,
            bias_hh: None,
            hidden_size: 4,
            initial_hidden: None,
            initial_cell: None,
        },
        vec![input_id],
        shape,
        DType::F32,
    )
}

/// Build a 5-node polar-to-cartesian graph for iSTFT testing.
fn build_polar_graph() -> ComputationGraph {
    let nodes = vec![
        make_input(1, vec![8]),
        make_input(2, vec![8]),
        make_cos(3, 2, vec![8]),
        make_sin(4, 2, vec![8]),
        make_mul(5, 1, 3, vec![8]),
    ];
    ComputationGraph::from_nodes(nodes)
}

// ===========================================================================
// RESOLVE_SPEC HARNESSES
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. IndexRange: start > end returns error
// ---------------------------------------------------------------------------

/// Prove: `extract_subgraph` with start > end returns an error.
#[kani::unwind(128)]
#[kani::proof]
fn resolve_spec_start_gt_end_error() {
    let graph =
        ComputationGraph::from_nodes(vec![make_input(1, vec![4]), make_relu(2, 1, vec![4])]);
    let spec = SubgraphSpec::IndexRange { start: 1, end: 0 };
    assert!(extract_subgraph(&graph, &spec).is_err());
}

// ---------------------------------------------------------------------------
// 2. IndexRange: start == 0, end == len is valid (full graph)
// ---------------------------------------------------------------------------

/// Prove: extracting the full graph by index range succeeds.
#[kani::unwind(128)]
#[kani::proof]
fn resolve_spec_full_range_valid() {
    let graph =
        ComputationGraph::from_nodes(vec![make_input(1, vec![4]), make_relu(2, 1, vec![4])]);
    let spec = SubgraphSpec::IndexRange { start: 0, end: 2 };
    let result = extract_subgraph(&graph, &spec);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().layer_count, 2);
}

// ---------------------------------------------------------------------------
// 3. NameContains: multiple patterns OR-matched
// ---------------------------------------------------------------------------

/// Prove: NameContains matches nodes matching ANY pattern, not all.
#[kani::unwind(128)]
#[kani::proof]
fn resolve_spec_name_contains_or_semantics() {
    let graph = ComputationGraph::from_nodes(vec![
        make_input(1, vec![4]),
        make_relu(2, 1, vec![4]),
        make_neg(3, 2, vec![4]),
    ]);
    let spec = SubgraphSpec::NameContains {
        patterns: vec!["relu".to_string(), "neg".to_string()],
    };
    let result = extract_subgraph(&graph, &spec).expect("valid");
    // Both relu and neg match => 2 layers extracted.
    assert_eq!(result.layer_count, 2);
}

// ---------------------------------------------------------------------------
// 4. NodeIds: duplicate IDs do not produce duplicate layers
// ---------------------------------------------------------------------------

/// Prove: passing the same NodeId twice does not extract it twice.
#[kani::unwind(128)]
#[kani::proof]
fn resolve_spec_node_ids_dedup() {
    let graph =
        ComputationGraph::from_nodes(vec![make_input(1, vec![4]), make_relu(2, 1, vec![4])]);
    let spec = SubgraphSpec::NodeIds { ids: vec![1, 1, 1] };
    let result = extract_subgraph(&graph, &spec).expect("valid");
    assert_eq!(
        result.layer_count, 1,
        "duplicate IDs must not duplicate layers"
    );
}

// ===========================================================================
// EXTRACT_SUBGRAPH INVARIANTS
// ===========================================================================

// ---------------------------------------------------------------------------
// 5. Extracted subgraph node count = layers + synthetic inputs
// ---------------------------------------------------------------------------

/// Prove: `graph.nodes().len() == layer_count + synthetic_input_count`.
#[kani::unwind(8)]
#[kani::proof]
fn extract_node_count_invariant() {
    let graph = build_polar_graph();
    // Extract Cos(3), Sin(4), Mul(5) which depend on Input(1) and Input(2).
    let spec = SubgraphSpec::IndexRange { start: 2, end: 5 };
    let result = extract_subgraph(&graph, &spec).expect("valid");
    assert_eq!(
        result.graph.nodes().len(),
        result.layer_count + result.synthetic_input_count,
    );
}

// ---------------------------------------------------------------------------
// 6. Synthetic input nodes have TraceOp::Input
// ---------------------------------------------------------------------------

/// Prove: all synthetic input nodes in the extracted subgraph are Input ops.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(6)]
fn extract_synthetic_inputs_are_input_ops() {
    let graph = build_polar_graph();
    let spec = SubgraphSpec::IndexRange { start: 2, end: 5 };
    let result = extract_subgraph(&graph, &spec).expect("valid");
    let sub_nodes = result.graph.nodes();
    for i in 0..result.synthetic_input_count {
        assert!(
            matches!(sub_nodes[i].op(), TraceOp::Input),
            "synthetic input must be Input op"
        );
    }
}

// ---------------------------------------------------------------------------
// 7. Synthetic input names start with "subgraph_input_"
// ---------------------------------------------------------------------------

/// Prove: synthetic input nodes have names prefixed with "subgraph_input_".
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(6)]
fn extract_synthetic_input_naming() {
    let graph = build_polar_graph();
    let spec = SubgraphSpec::IndexRange { start: 2, end: 5 };
    let result = extract_subgraph(&graph, &spec).expect("valid");
    let sub_nodes = result.graph.nodes();
    for i in 0..result.synthetic_input_count {
        assert!(
            sub_nodes[i].name().starts_with("subgraph_input_"),
            "synthetic input name must start with 'subgraph_input_'"
        );
    }
}

// ---------------------------------------------------------------------------
// 8. ID map has unique values (no collisions)
// ---------------------------------------------------------------------------

/// Prove: all remapped IDs in `id_map` are unique.
#[kani::unwind(8)]
#[kani::proof]
fn extract_id_map_unique_values() {
    let graph = build_polar_graph();
    let spec = SubgraphSpec::IndexRange { start: 2, end: 5 };
    let result = extract_subgraph(&graph, &spec).expect("valid");
    let values: Vec<u64> = result.id_map.values().copied().collect();
    let unique: HashSet<u64> = values.iter().copied().collect();
    assert_eq!(values.len(), unique.len(), "remapped IDs must be unique");
}

// ---------------------------------------------------------------------------
// 9. ID map values start at 1
// ---------------------------------------------------------------------------

/// Prove: the minimum remapped ID is 1 (never 0).
#[kani::unwind(128)]
#[kani::proof]
fn extract_id_map_starts_at_one() {
    let graph =
        ComputationGraph::from_nodes(vec![make_input(1, vec![4]), make_relu(2, 1, vec![4])]);
    let spec = SubgraphSpec::IndexRange { start: 0, end: 2 };
    let result = extract_subgraph(&graph, &spec).expect("valid");
    let min_id = result.id_map.values().copied().min().unwrap();
    assert_eq!(min_id, 1, "minimum remapped ID must be 1");
}

// ---------------------------------------------------------------------------
// 10. Extracted subgraph preserves output shape
// ---------------------------------------------------------------------------

/// Prove: the last extracted node has the same output shape as the
/// corresponding node in the original graph.
#[kani::unwind(8)]
#[kani::proof]
fn extract_preserves_output_shape() {
    let graph = build_polar_graph();
    let spec = SubgraphSpec::IndexRange { start: 2, end: 5 };
    let result = extract_subgraph(&graph, &spec).expect("valid");
    let orig_last = &graph.nodes()[4]; // Mul(5)
    let sub_last = result.graph.nodes().last().unwrap();
    assert_eq!(
        orig_last.output_shape(),
        sub_last.output_shape(),
        "output shape must be preserved"
    );
}

// ---------------------------------------------------------------------------
// 11. Extracted subgraph preserves op type
// ---------------------------------------------------------------------------

/// Prove: each extracted node retains its original op type.
#[kani::unwind(128)]
#[kani::proof]
fn extract_preserves_op_type() {
    let graph =
        ComputationGraph::from_nodes(vec![make_input(1, vec![4]), make_relu(2, 1, vec![4])]);
    let spec = SubgraphSpec::IndexRange { start: 0, end: 2 };
    let result = extract_subgraph(&graph, &spec).expect("valid");
    let sub_nodes = result.graph.nodes();
    assert!(matches!(sub_nodes[0].op(), TraceOp::Input));
    assert!(matches!(sub_nodes[1].op(), TraceOp::Relu));
}

// ===========================================================================
// VALIDATE_SUBGRAPH HARNESSES
// ===========================================================================

// ---------------------------------------------------------------------------
// 12. validate: extracted subgraph always passes
// ---------------------------------------------------------------------------

/// Prove: `validate_subgraph` always succeeds on the result of `extract_subgraph`
/// for a multi-dependency subgraph.
#[kani::unwind(8)]
#[kani::proof]
fn validate_extracted_subgraph_passes() {
    let graph = build_polar_graph();
    let spec = SubgraphSpec::IndexRange { start: 2, end: 5 };
    let result = extract_subgraph(&graph, &spec).expect("valid extraction");
    assert!(validate_subgraph(&result.graph).is_ok());
}

// ---------------------------------------------------------------------------
// 13. validate: graph with dangling reference fails
// ---------------------------------------------------------------------------

/// Prove: a graph where a node references a nonexistent input ID fails
/// validation.
#[kani::unwind(128)]
#[kani::proof]
fn validate_dangling_reference_fails() {
    // Node 2 references input 999, which doesn't exist.
    let nodes = vec![
        make_input(1, vec![4]),
        TraceNode::new(
            2,
            "relu_2".to_string(),
            TraceOp::Relu,
            vec![999],
            vec![4],
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    assert!(validate_subgraph(&graph).is_err());
}

// ---------------------------------------------------------------------------
// 14. validate: single Input node passes
// ---------------------------------------------------------------------------

/// Prove: a graph with just one Input node is valid.
#[kani::unwind(128)]
#[kani::proof]
fn validate_single_input_passes() {
    let graph = ComputationGraph::from_nodes(vec![make_input(1, vec![4])]);
    assert!(validate_subgraph(&graph).is_ok());
}

// ===========================================================================
// IS_AY_COMPATIBLE_OP HARNESSES
// ===========================================================================

// ---------------------------------------------------------------------------
// 15. Shape ops are ay-compatible
// ---------------------------------------------------------------------------

/// Prove: Reshape, Transpose, Squeeze, Unsqueeze, Narrow, Flip are compatible.
#[kani::unwind(8)]
#[kani::proof]
fn ay_compat_shape_ops() {
    assert!(is_ay_compatible_op(&TraceOp::Reshape {
        target_shape: vec![4, 2]
    }));
    assert!(is_ay_compatible_op(&TraceOp::Transpose {
        dim0: 0,
        dim1: 1
    }));
    assert!(is_ay_compatible_op(&TraceOp::Squeeze { dim: 0 }));
    assert!(is_ay_compatible_op(&TraceOp::Unsqueeze { dim: 0 }));
    assert!(is_ay_compatible_op(&TraceOp::Narrow {
        dim: 0,
        start: 0,
        length: 4
    }));
    assert!(is_ay_compatible_op(&TraceOp::Flip { dim: 0 }));
}

// ---------------------------------------------------------------------------
// 16. Reduction ops are ay-compatible
// ---------------------------------------------------------------------------

/// Prove: ReduceSum and ReduceMean are ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_compat_reduction_ops() {
    assert!(is_ay_compatible_op(&TraceOp::ReduceSum {
        dim: 0,
        keepdim: false
    }));
    assert!(is_ay_compatible_op(&TraceOp::ReduceMean {
        dim: 1,
        keepdim: true
    }));
}

// ---------------------------------------------------------------------------
// 17. Cat is ay-compatible
// ---------------------------------------------------------------------------

/// Prove: Cat op is ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_compat_cat() {
    assert!(is_ay_compatible_op(&TraceOp::Cat {
        dim: 0,
        num_inputs: 2
    }));
}

// ---------------------------------------------------------------------------
// 18. Clamp and Powf are ay-compatible
// ---------------------------------------------------------------------------

/// Prove: Clamp and Powf are ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_compat_clamp_powf() {
    assert!(is_ay_compatible_op(&TraceOp::Clamp {
        min: Some(0.0),
        max: Some(1.0)
    }));
    assert!(is_ay_compatible_op(&TraceOp::Powf { exponent: 2.0 }));
}

// ---------------------------------------------------------------------------
// 19. Gelu, GeluErf, Silu are ay-compatible
// ---------------------------------------------------------------------------

/// Prove: activation ops Gelu, GeluErf, Silu are ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_compat_gelu_silu() {
    assert!(is_ay_compatible_op(&TraceOp::Gelu));
    assert!(is_ay_compatible_op(&TraceOp::GeluErf));
    assert!(is_ay_compatible_op(&TraceOp::Silu));
}

// ---------------------------------------------------------------------------
// 20. SDPA is NOT ay-compatible
// ---------------------------------------------------------------------------

/// Prove: SDPA (scaled dot-product attention) is NOT ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_incompat_sdpa() {
    assert!(!is_ay_compatible_op(&TraceOp::Sdpa { scale: 1.0 }));
    assert!(!is_ay_compatible_op(&TraceOp::SdpaCausal { scale: 1.0 }));
}

// ---------------------------------------------------------------------------
// 21. MultiHeadAttention is NOT ay-compatible
// ---------------------------------------------------------------------------

/// Prove: MHA is NOT ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_incompat_mha() {
    assert!(!is_ay_compatible_op(&TraceOp::MultiHeadAttention {
        num_heads: 4,
        num_kv_heads: 4,
        head_dim: 32,
    }));
}

// ---------------------------------------------------------------------------
// 22. Conv1d: small kernel compatible, large kernel not
// ---------------------------------------------------------------------------

/// Prove: Conv1d with complexity <= 4096 is compatible, > 4096 is not.
#[kani::unwind(1)]
#[kani::proof]
fn ay_conv1d_threshold() {
    // 4 * 4 * 3 * 1 = 48 <= 4096 => compatible
    let small = TraceOp::Conv1d {
        weight: WeightRef::from_shape(&[4, 4, 3]),
        bias: None,
        padding: 1,
        stride: 1,
        dilation: 1,
        groups: 1,
    };
    assert!(is_ay_compatible_op(&small));

    // 64 * 64 * 3 * 1 = 12288 > 4096 => incompatible
    let large = TraceOp::Conv1d {
        weight: WeightRef::from_shape(&[64, 64, 3]),
        bias: None,
        padding: 1,
        stride: 1,
        dilation: 1,
        groups: 1,
    };
    assert!(!is_ay_compatible_op(&large));
}

// ---------------------------------------------------------------------------
// 23. ConvTranspose1d is NOT ay-compatible
// ---------------------------------------------------------------------------

/// Prove: ConvTranspose1d is never ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_incompat_conv_transpose1d() {
    let op = TraceOp::ConvTranspose1d {
        weight: WeightRef::from_shape(&[4, 4, 3]),
        bias: None,
        padding: 1,
        output_padding: 0,
        stride: 1,
        dilation: 1,
        groups: 1,
    };
    assert!(!is_ay_compatible_op(&op));
}

// ===========================================================================
// FIND_AY_CANDIDATES HARNESSES
// ===========================================================================

// ---------------------------------------------------------------------------
// 24. All candidates have layer_count in [min, max]
// ---------------------------------------------------------------------------

/// Prove: every candidate returned by `find_ay_candidates` has
/// `min_layers <= layer_count <= max_layers`.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn candidates_layer_count_in_range() {
    let graph = build_polar_graph();
    let min = 2_usize;
    let max = 3_usize;
    let candidates = find_ay_candidates(&graph, min, max);
    for c in &candidates {
        assert!(c.layer_count >= min, "layer_count must be >= min_layers");
        assert!(c.layer_count <= max, "layer_count must be <= max_layers");
    }
}

// ---------------------------------------------------------------------------
// 25. Candidate start_index + layer_count == end_index
// ---------------------------------------------------------------------------

/// Prove: every candidate has `end_index == start_index + layer_count`.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn candidates_index_arithmetic() {
    let graph = build_polar_graph();
    let candidates = find_ay_candidates(&graph, 1, 5);
    for c in &candidates {
        assert_eq!(
            c.end_index,
            c.start_index + c.layer_count,
            "end = start + layer_count"
        );
    }
}

// ---------------------------------------------------------------------------
// 26. All ops in candidate regions are ay-compatible
// ---------------------------------------------------------------------------

/// Prove: every node within a ay candidate region is ay-compatible.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn candidates_all_ops_compatible() {
    let graph = build_polar_graph();
    let nodes = graph.nodes();
    let candidates = find_ay_candidates(&graph, 1, 5);
    for c in &candidates {
        for i in c.start_index..c.end_index {
            assert!(
                is_ay_compatible_op(nodes[i].op()),
                "node at index {i} must be ay-compatible"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 27. LSTM breaks candidate region
// ---------------------------------------------------------------------------

/// Prove: a graph with LSTM in the middle produces candidates only
/// on compatible sides, never spanning the LSTM.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn candidates_lstm_breaks_region() {
    let nodes = vec![
        make_input(1, vec![4]),
        make_relu(2, 1, vec![4]),
        make_lstm(3, 2, vec![4, 4]), // incompatible
        make_neg(4, 3, vec![4, 4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let candidates = find_ay_candidates(&graph, 1, 4);
    for c in &candidates {
        // No candidate should span across the LSTM at index 2.
        let spans_lstm = c.start_index <= 2 && c.end_index > 2;
        assert!(!spans_lstm, "candidate must not span across LSTM");
    }
}

// ---------------------------------------------------------------------------
// 28. Empty graph: zero candidates regardless of min/max
// ---------------------------------------------------------------------------

/// Prove: `find_ay_candidates` on an empty graph always returns empty.
#[kani::unwind(8)]
#[kani::proof]
fn candidates_empty_graph_any_params() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let min: usize = kani::any();
    let max: usize = kani::any();
    kani::assume(min >= 1 && min <= 10);
    kani::assume(max >= min && max <= 10);
    let candidates = find_ay_candidates(&graph, min, max);
    assert!(candidates.is_empty());
}

// ---------------------------------------------------------------------------
// 29. Candidate op_types length matches layer_count
// ---------------------------------------------------------------------------

/// Prove: `op_types.len() == layer_count` for every candidate.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn candidates_op_types_length() {
    let graph = build_polar_graph();
    let candidates = find_ay_candidates(&graph, 2, 4);
    for c in &candidates {
        assert_eq!(c.op_types.len(), c.layer_count);
    }
}

// ---------------------------------------------------------------------------
// 30. Candidate internal_nodes length matches layer_count
// ---------------------------------------------------------------------------

/// Prove: `internal_nodes.len() == layer_count` for every candidate.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn candidates_internal_nodes_length() {
    let graph = build_polar_graph();
    let candidates = find_ay_candidates(&graph, 2, 4);
    for c in &candidates {
        assert_eq!(c.internal_nodes.len(), c.layer_count);
    }
}

// ---------------------------------------------------------------------------
// 31. Candidate estimated_complexity is always > 0
// ---------------------------------------------------------------------------

/// Prove: every candidate has non-zero estimated complexity.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn candidates_complexity_positive() {
    let graph = build_polar_graph();
    let candidates = find_ay_candidates(&graph, 1, 5);
    for c in &candidates {
        assert!(c.estimated_complexity > 0);
    }
}

// ===========================================================================
// ESTIMATE_OP_COMPLEXITY HARNESSES
// ===========================================================================

// ---------------------------------------------------------------------------
// 32. Linear complexity includes weight data length
// ---------------------------------------------------------------------------

/// Prove: Linear op complexity = output_elements + weight.data().len().
#[kani::unwind(128)]
#[kani::proof]
fn complexity_linear_includes_weights() {
    let w = WeightRef::new(vec![0.0; 32], vec![4, 8]).expect("valid");
    let nodes = vec![
        make_input(1, vec![1, 8]),
        TraceNode::new(
            2,
            "linear".to_string(),
            TraceOp::Linear {
                weight: w,
                bias: None,
            },
            vec![1],
            vec![1, 4],
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let candidates = find_ay_candidates(&graph, 2, 2);
    assert_eq!(candidates.len(), 1);
    // Input: 1*8 = 8. Linear: 1*4 + 32 (weights) = 36. Total = 44.
    assert_eq!(candidates[0].estimated_complexity, 44);
}

// ---------------------------------------------------------------------------
// 33. MatMul complexity is 3x output elements
// ---------------------------------------------------------------------------

/// Prove: MatMul op complexity = output_elements * 3.
#[kani::unwind(128)]
#[kani::proof]
fn complexity_matmul_3x() {
    let nodes = vec![
        make_input(1, vec![2, 4]),
        make_input(2, vec![4, 3]),
        TraceNode::new(
            3,
            "matmul".to_string(),
            TraceOp::MatMul,
            vec![1, 2],
            vec![2, 3],
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let candidates = find_ay_candidates(&graph, 3, 3);
    assert_eq!(candidates.len(), 1);
    // Input(1): 2*4=8. Input(2): 4*3=12. MatMul: (2*3)*3=18. Total: 38.
    assert_eq!(candidates[0].estimated_complexity, 38);
}

// ---------------------------------------------------------------------------
// 34. Softmax complexity is 3x output elements
// ---------------------------------------------------------------------------

/// Prove: Softmax op complexity = output_elements * 3.
#[kani::unwind(128)]
#[kani::proof]
fn complexity_softmax_3x() {
    let nodes = vec![
        make_input(1, vec![4]),
        TraceNode::new(
            2,
            "softmax".to_string(),
            TraceOp::Softmax { dim: 0 },
            vec![1],
            vec![4],
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let candidates = find_ay_candidates(&graph, 2, 2);
    assert_eq!(candidates.len(), 1);
    // Input: 4. Softmax: 4*3=12. Total: 16.
    assert_eq!(candidates[0].estimated_complexity, 16);
}

// ===========================================================================
// EXTRACT + VALIDATE ROUNDTRIP
// ===========================================================================

// ---------------------------------------------------------------------------
// 35. Extract by NodeIds then validate
// ---------------------------------------------------------------------------

/// Prove: extracting by NodeIds and then validating always succeeds.
#[kani::unwind(8)]
#[kani::proof]
fn extract_by_ids_then_validate() {
    let graph = build_polar_graph();
    let spec = SubgraphSpec::NodeIds { ids: vec![3, 4, 5] };
    let result = extract_subgraph(&graph, &spec).expect("valid IDs");
    assert!(validate_subgraph(&result.graph).is_ok());
}

// ---------------------------------------------------------------------------
// 36. Extract by NameContains then validate
// ---------------------------------------------------------------------------

/// Prove: extracting by name pattern and then validating always succeeds.
#[kani::unwind(128)]
#[kani::proof]
fn extract_by_name_then_validate() {
    let graph = build_polar_graph();
    let spec = SubgraphSpec::NameContains {
        patterns: vec!["cos".to_string(), "mul".to_string()],
    };
    let result = extract_subgraph(&graph, &spec).expect("valid patterns");
    assert!(validate_subgraph(&result.graph).is_ok());
}

// ===========================================================================
// TRACE_OP_NAME
// ===========================================================================

// ---------------------------------------------------------------------------
// 37. Op type names are non-empty for candidates
// ---------------------------------------------------------------------------

/// Prove: every `op_types` entry in a candidate is a non-empty string.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn op_type_names_nonempty() {
    let graph = build_polar_graph();
    let candidates = find_ay_candidates(&graph, 1, 5);
    for c in &candidates {
        for name in &c.op_types {
            assert!(!name.is_empty(), "op type name must be non-empty");
        }
    }
}

// ---------------------------------------------------------------------------
// 38. Op type names do not contain braces (field data stripped)
// ---------------------------------------------------------------------------

/// Prove: trace_op_name strips struct fields from Debug output.
#[kani::unwind(128)]
#[kani::proof]
fn op_type_names_no_braces() {
    let w = WeightRef::from_shape(&[4, 2]);
    let nodes = vec![
        make_input(1, vec![4]),
        TraceNode::new(
            2,
            "linear".to_string(),
            TraceOp::Linear {
                weight: w,
                bias: None,
            },
            vec![1],
            vec![1, 4],
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let candidates = find_ay_candidates(&graph, 2, 2);
    for c in &candidates {
        for name in &c.op_types {
            assert!(
                !name.contains('{'),
                "op type name must not contain '{{' brace"
            );
            assert!(
                !name.contains('}'),
                "op type name must not contain '}}' brace"
            );
        }
    }
}

// ===========================================================================
// SYNTHETIC INPUT SHAPE PRESERVATION
// ===========================================================================

// ---------------------------------------------------------------------------
// 39. Synthetic inputs preserve original node's output shape
// ---------------------------------------------------------------------------

/// Prove: synthetic input nodes have the same output shape as the
/// original external dependency nodes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(6)]
fn synthetic_input_shape_preservation() {
    let graph = build_polar_graph();
    // Extract Cos(3) which depends on Input(2) with shape [8].
    let spec = SubgraphSpec::IndexRange { start: 2, end: 3 };
    let result = extract_subgraph(&graph, &spec).expect("valid");
    assert_eq!(result.synthetic_input_count, 1);
    let synth = &result.graph.nodes()[0];
    assert_eq!(
        synth.output_shape(),
        &[8],
        "synthetic input shape must match original"
    );
}

// ---------------------------------------------------------------------------
// 40. Multi-input node: all external deps become synthetic inputs
// ---------------------------------------------------------------------------

/// Prove: a node with 2 external dependencies gets 2 synthetic inputs.
#[kani::unwind(128)]
#[kani::proof]
fn extract_multi_input_external_deps() {
    let nodes = vec![
        make_input(1, vec![4]),
        make_input(2, vec![4]),
        make_add(3, 1, 2, vec![4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    // Extract only the Add node.
    let spec = SubgraphSpec::IndexRange { start: 2, end: 3 };
    let result = extract_subgraph(&graph, &spec).expect("valid");
    assert_eq!(result.synthetic_input_count, 2, "both inputs are external");
    assert_eq!(result.layer_count, 1);
    assert_eq!(result.graph.nodes().len(), 3);
}

// ===========================================================================
// NEW HARNESSES -- ay compatibility for additional op types
// ===========================================================================

/// Prove: Maximum and Minimum binary ops are ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_compat_max_min_binary() {
    assert!(is_ay_compatible_op(&TraceOp::Maximum));
    assert!(is_ay_compatible_op(&TraceOp::Minimum));
}

/// Prove: Constant and ConstantWeight ops are ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_compat_constant_ops() {
    assert!(is_ay_compatible_op(&TraceOp::Constant { value: 0.0 }));
    assert!(is_ay_compatible_op(&TraceOp::ConstantWeight {
        weight: WeightRef::from_shape(&[1])
    }));
}

/// Prove: Embedding op is ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_compat_embedding() {
    let w = WeightRef::from_shape(&[100, 32]);
    assert!(is_ay_compatible_op(&TraceOp::Embedding { weight: w }));
}

/// Prove: LayerNorm and BatchNorm are ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_compat_norm_ops() {
    let w = WeightRef::from_shape(&[32]);
    let b = WeightRef::from_shape(&[32]);
    assert!(is_ay_compatible_op(&TraceOp::LayerNorm {
        eps: 1e-5,
        weight: w.clone(),
        bias: b.clone()
    }));
    let rm = WeightRef::from_shape(&[32]);
    let rv = WeightRef::from_shape(&[32]);
    assert!(is_ay_compatible_op(&TraceOp::BatchNorm {
        eps: 1e-5,
        weight: w,
        bias: b,
        running_mean: rm,
        running_var: rv
    }));
}

/// Prove: Conv2d is NOT ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_incompat_conv2d() {
    let w = WeightRef::from_shape(&[4, 4, 3, 3]);
    assert!(!is_ay_compatible_op(&TraceOp::Conv2d {
        weight: w,
        bias: None,
        padding: [1, 1],
        stride: [1, 1],
        dilation: [1, 1],
        groups: 1
    }));
}

/// Prove: ReduceMax and ReduceMin are NOT ay-compatible.
#[kani::unwind(1)]
#[kani::proof]
fn ay_incompat_reduce_max_min() {
    assert!(!is_ay_compatible_op(&TraceOp::ReduceMax {
        dim: 0,
        keepdim: false
    }));
    assert!(!is_ay_compatible_op(&TraceOp::ReduceMin {
        dim: 0,
        keepdim: false
    }));
}

// ===========================================================================
// NEW HARNESSES -- Extract subgraph additional properties
// ===========================================================================

/// Prove: nodes in the extracted subgraph are in ascending ID order.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn extract_preserves_node_order() {
    let graph = build_polar_graph();
    let spec = SubgraphSpec::IndexRange { start: 0, end: 5 };
    let result = extract_subgraph(&graph, &spec).expect("valid");
    let ids: Vec<u64> = result.graph.nodes().iter().map(|n| n.id()).collect();
    for i in 1..ids.len() {
        assert!(ids[i] > ids[i - 1], "node IDs must be ascending");
    }
}

/// Prove: synthetic input nodes preserve DType from original.
#[kani::unwind(128)]
#[kani::proof]
fn extract_synthetic_input_dtype_preserved() {
    let nodes = vec![
        TraceNode::new(
            1,
            "input_1".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::BF16,
        ),
        make_relu(2, 1, vec![4]),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let spec = SubgraphSpec::IndexRange { start: 1, end: 2 };
    let result = extract_subgraph(&graph, &spec).expect("valid");
    assert_eq!(result.synthetic_input_count, 1);
    assert_eq!(result.graph.nodes()[0].output_dtype(), DType::BF16);
}

/// Prove: all-compatible graph has a full-size candidate.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn candidates_all_compatible_has_full_size_candidate() {
    let graph = build_polar_graph();
    let n = graph.nodes().len();
    let candidates = find_ay_candidates(&graph, 1, n);
    assert!(candidates.iter().any(|c| c.layer_count == n));
}

/// Prove: no duplicate (start, end) pairs in candidates.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn candidates_overlapping_windows_unique() {
    let graph = build_polar_graph();
    let candidates = find_ay_candidates(&graph, 1, 5);
    for i in 0..candidates.len() {
        for j in (i + 1)..candidates.len() {
            let same = candidates[i].start_index == candidates[j].start_index
                && candidates[i].end_index == candidates[j].end_index;
            assert!(!same, "no duplicate (start, end) pairs");
        }
    }
}

/// Prove: elementwise op complexity equals output element count.
#[kani::unwind(128)]
#[kani::proof]
fn complexity_elementwise_is_output_elements() {
    let nodes = vec![make_input(1, vec![3, 4]), make_relu(2, 1, vec![3, 4])];
    let graph = ComputationGraph::from_nodes(nodes);
    let candidates = find_ay_candidates(&graph, 2, 2);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].estimated_complexity, 24);
}

/// Prove: last internal_node ID matches exit node.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn candidates_last_node_is_exit() {
    let graph = build_polar_graph();
    let nodes = graph.nodes();
    let candidates = find_ay_candidates(&graph, 2, 3);
    for c in &candidates {
        let last_internal = c.internal_nodes.last().expect("non-empty");
        assert_eq!(*last_internal, nodes[c.end_index - 1].id());
    }
}

/// Prove: every internal_node ID exists in the original graph.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn candidates_internal_nodes_valid_ids() {
    let graph = build_polar_graph();
    let all_ids: HashSet<u64> = graph.nodes().iter().map(|n| n.id()).collect();
    let candidates = find_ay_candidates(&graph, 1, 5);
    for c in &candidates {
        for nid in &c.internal_nodes {
            assert!(all_ids.contains(nid));
        }
    }
}

/// Prove: IndexRange with end > graph length returns error.
#[kani::unwind(128)]
#[kani::proof]
fn resolve_spec_out_of_bounds_error() {
    let graph = ComputationGraph::from_nodes(vec![make_input(1, vec![4])]);
    let spec = SubgraphSpec::IndexRange { start: 0, end: 5 };
    assert!(extract_subgraph(&graph, &spec).is_err());
}

/// Prove: NodeIds with nonexistent ID returns error.
#[kani::unwind(128)]
#[kani::proof]
fn resolve_spec_nonexistent_node_id_error() {
    let graph = ComputationGraph::from_nodes(vec![make_input(1, vec![4])]);
    let spec = SubgraphSpec::NodeIds { ids: vec![999] };
    assert!(extract_subgraph(&graph, &spec).is_err());
}
