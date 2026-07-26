// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for subgraph extraction from traced computation graphs.
//!
//! Part of #2455.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::{
    extract_subgraph, find_ay_candidates, validate_subgraph, SubgraphSpec, AYCandidateRegion,
};

/// Helper: build a small computation graph resembling a polar-to-cartesian
/// conversion (the iSTFT target for ay verification).
///
/// Graph structure:
///   0: Input (magnitude) [11]
///   1: Input (phase) [11]
///   2: Cos(phase) [11]
///   3: Sin(phase) [11]
///   4: Mul(magnitude, cos) [11]  -- real part
///   5: Mul(magnitude, sin) [11]  -- imag part
///   6: Cat(real, imag) [22]
fn build_polar_to_rect_graph() -> ComputationGraph {
    let nodes = vec![
        TraceNode::new(
            100,
            "magnitude".to_string(),
            TraceOp::Input,
            vec![],
            vec![11],
            DType::F32,
        ),
        TraceNode::new(
            101,
            "phase".to_string(),
            TraceOp::Input,
            vec![],
            vec![11],
            DType::F32,
        ),
        TraceNode::new(
            102,
            "cos_phase".to_string(),
            TraceOp::Cos,
            vec![101],
            vec![11],
            DType::F32,
        ),
        TraceNode::new(
            103,
            "sin_phase".to_string(),
            TraceOp::Sin,
            vec![101],
            vec![11],
            DType::F32,
        ),
        TraceNode::new(
            104,
            "real".to_string(),
            TraceOp::Mul,
            vec![100, 102],
            vec![11],
            DType::F32,
        ),
        TraceNode::new(
            105,
            "imag".to_string(),
            TraceOp::Mul,
            vec![100, 103],
            vec![11],
            DType::F32,
        ),
        TraceNode::new(
            106,
            "spectral".to_string(),
            TraceOp::Cat {
                dim: 0,
                num_inputs: 2,
            },
            vec![104, 105],
            vec![22],
            DType::F32,
        ),
    ];
    ComputationGraph::from_nodes(nodes)
}

/// Helper: build a larger graph with a non-ay-compatible op (LSTM) in the middle.
///
/// Graph structure:
///   0: Input [4, 8]
///   1: Relu [4, 8]
///   2: LSTM [4, 16]     -- NOT ay-compatible
///   3: Relu [4, 16]
///   4: Add(node3, node2) -- depends on LSTM output
///   5: Neg [4, 16]
fn build_mixed_graph() -> ComputationGraph {
    let nodes = vec![
        TraceNode::new(
            200,
            "input".to_string(),
            TraceOp::Input,
            vec![],
            vec![4, 8],
            DType::F32,
        ),
        TraceNode::new(
            201,
            "relu_0".to_string(),
            TraceOp::Relu,
            vec![200],
            vec![4, 8],
            DType::F32,
        ),
        TraceNode::new(
            202,
            "lstm_0".to_string(),
            TraceOp::Lstm {
                weight_ih: WeightRef::new(vec![0.0; 256], vec![64, 4]).unwrap(),
                weight_hh: WeightRef::new(vec![0.0; 1024], vec![64, 16]).unwrap(),
                bias_ih: None,
                bias_hh: None,
                hidden_size: 16,
                initial_hidden: None,
                initial_cell: None,
            },
            vec![201],
            vec![4, 16],
            DType::F32,
        ),
        TraceNode::new(
            203,
            "relu_1".to_string(),
            TraceOp::Relu,
            vec![202],
            vec![4, 16],
            DType::F32,
        ),
        TraceNode::new(
            204,
            "add_0".to_string(),
            TraceOp::Add,
            vec![203, 202],
            vec![4, 16],
            DType::F32,
        ),
        TraceNode::new(
            205,
            "neg_0".to_string(),
            TraceOp::Neg,
            vec![204],
            vec![4, 16],
            DType::F32,
        ),
    ];
    ComputationGraph::from_nodes(nodes)
}

// -- extract_subgraph tests --------------------------------------------------

#[test]
fn test_extract_full_graph_by_index_range() {
    let graph = build_polar_to_rect_graph();
    let result = extract_subgraph(&graph, &SubgraphSpec::IndexRange { start: 0, end: 7 })
        .expect("extract full graph");

    assert_eq!(result.layer_count, 7);
    assert_eq!(result.synthetic_input_count, 0);
    assert_eq!(result.graph.nodes().len(), 7);
    validate_subgraph(&result.graph).expect("full graph is self-contained");
}

#[test]
fn test_extract_middle_subgraph_creates_synthetic_inputs() {
    let graph = build_polar_to_rect_graph();
    // Extract nodes 2-6 (cos, sin, mul_real, mul_imag, cat).
    // These depend on nodes 100 (magnitude) and 101 (phase) which are outside.
    let result = extract_subgraph(&graph, &SubgraphSpec::IndexRange { start: 2, end: 7 })
        .expect("extract middle subgraph");

    assert_eq!(result.layer_count, 5);
    // Nodes 100 (magnitude) and 101 (phase) become synthetic inputs.
    assert_eq!(result.synthetic_input_count, 2);
    // Total nodes = 2 synthetic + 5 extracted
    assert_eq!(result.graph.nodes().len(), 7);
    validate_subgraph(&result.graph).expect("subgraph with synthetic inputs is self-contained");
}

#[test]
fn test_extract_single_node() {
    let graph = build_polar_to_rect_graph();
    // Extract just the Cos node (index 2). It depends on node 101 (phase).
    let result = extract_subgraph(&graph, &SubgraphSpec::IndexRange { start: 2, end: 3 })
        .expect("extract single node");

    assert_eq!(result.layer_count, 1);
    assert_eq!(result.synthetic_input_count, 1); // phase input
    assert_eq!(result.graph.nodes().len(), 2); // 1 synthetic + 1 extracted
    validate_subgraph(&result.graph).expect("single node subgraph is valid");
}

#[test]
fn test_extract_by_name_pattern() {
    let graph = build_polar_to_rect_graph();
    let result = extract_subgraph(
        &graph,
        &SubgraphSpec::NameContains {
            patterns: vec!["cos".to_string(), "sin".to_string()],
        },
    )
    .expect("extract by name");

    // Should match cos_phase and sin_phase nodes.
    assert_eq!(result.layer_count, 2);
    // Both depend on node 101 (phase) -- 1 unique external dep.
    assert_eq!(result.synthetic_input_count, 1);
    validate_subgraph(&result.graph).expect("name-matched subgraph is valid");
}

#[test]
fn test_extract_by_node_ids() {
    let graph = build_polar_to_rect_graph();
    let result = extract_subgraph(
        &graph,
        &SubgraphSpec::NodeIds {
            ids: vec![104, 105, 106],
        },
    )
    .expect("extract by node IDs");

    // Nodes 104 (real), 105 (imag), 106 (cat).
    assert_eq!(result.layer_count, 3);
    // External deps: 100 (magnitude), 102 (cos), 103 (sin) = 3 synthetic inputs.
    assert_eq!(result.synthetic_input_count, 3);
    validate_subgraph(&result.graph).expect("ID-matched subgraph is valid");
}

#[test]
fn test_extract_preserves_output_shapes() {
    let graph = build_polar_to_rect_graph();
    let result =
        extract_subgraph(&graph, &SubgraphSpec::IndexRange { start: 2, end: 7 }).expect("extract");

    // The output node should be the cat with shape [22].
    let output = result.graph.output_node().expect("has output");
    assert_eq!(output.output_shape(), &[22]);
    assert_eq!(output.name(), "spectral");
}

#[test]
fn test_extract_synthetic_inputs_have_correct_shapes() {
    let graph = build_polar_to_rect_graph();
    let result =
        extract_subgraph(&graph, &SubgraphSpec::IndexRange { start: 2, end: 7 }).expect("extract");

    // First two nodes should be synthetic inputs with shape [11].
    let sub_nodes = result.graph.nodes();
    assert!(matches!(sub_nodes[0].op(), TraceOp::Input));
    assert!(matches!(sub_nodes[1].op(), TraceOp::Input));
    assert_eq!(sub_nodes[0].output_shape(), &[11]);
    assert_eq!(sub_nodes[1].output_shape(), &[11]);
    assert!(sub_nodes[0].name().starts_with("subgraph_input_"));
    assert!(sub_nodes[1].name().starts_with("subgraph_input_"));
}

// -- Error cases -------------------------------------------------------------

#[test]
fn test_extract_out_of_bounds_index() {
    let graph = build_polar_to_rect_graph();
    let result = extract_subgraph(&graph, &SubgraphSpec::IndexRange { start: 5, end: 10 });
    assert!(result.is_err());
}

#[test]
fn test_extract_empty_range() {
    let graph = build_polar_to_rect_graph();
    let result = extract_subgraph(&graph, &SubgraphSpec::IndexRange { start: 3, end: 3 });
    assert!(result.is_err());
}

#[test]
fn test_extract_no_matching_names() {
    let graph = build_polar_to_rect_graph();
    let result = extract_subgraph(
        &graph,
        &SubgraphSpec::NameContains {
            patterns: vec!["nonexistent".to_string()],
        },
    );
    assert!(result.is_err());
}

#[test]
fn test_extract_no_matching_ids() {
    let graph = build_polar_to_rect_graph();
    let result = extract_subgraph(
        &graph,
        &SubgraphSpec::NodeIds {
            ids: vec![999, 998],
        },
    );
    assert!(result.is_err());
}

// -- validate_subgraph tests -------------------------------------------------

#[test]
fn test_validate_empty_graph() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let result = validate_subgraph(&graph);
    assert!(result.is_err());
}

#[test]
fn test_validate_well_formed_graph() {
    let graph = build_polar_to_rect_graph();
    validate_subgraph(&graph).expect("well-formed graph validates");
}

// -- find_ay_candidates tests ------------------------------------------------

#[test]
fn test_find_candidates_all_compatible() {
    let graph = build_polar_to_rect_graph();
    let candidates = find_ay_candidates(&graph, 3, 5);

    // All 7 nodes are ay-compatible, so we should find candidates.
    assert!(!candidates.is_empty());

    // Should find at least one candidate of size 3 starting at index 0.
    let has_3_at_0 = candidates
        .iter()
        .any(|c| c.start_index == 0 && c.layer_count == 3);
    assert!(has_3_at_0, "expected candidate of size 3 at index 0");

    // All candidates should have layer_count in [3, 5].
    for c in &candidates {
        assert!(
            c.layer_count >= 3 && c.layer_count <= 5,
            "layer_count={}",
            c.layer_count
        );
    }
}

#[test]
fn test_find_candidates_mixed_graph() {
    let graph = build_mixed_graph();
    let candidates = find_ay_candidates(&graph, 2, 4);

    // The LSTM at index 2 breaks ay compatibility.
    // Region 0: [Input, Relu] (indices 0-1, len=2, compatible).
    // Region 1: [Relu, Add, Neg] (indices 3-5, len=3, compatible).
    // So we should find candidates in both regions.
    let region_0: Vec<&AYCandidateRegion> =
        candidates.iter().filter(|c| c.start_index <= 1).collect();
    let region_1: Vec<&AYCandidateRegion> =
        candidates.iter().filter(|c| c.start_index >= 3).collect();

    assert!(!region_0.is_empty(), "expected candidates in region 0");
    assert!(!region_1.is_empty(), "expected candidates in region 1");
}

#[test]
fn test_find_candidates_min_larger_than_graph() {
    let graph = build_polar_to_rect_graph();
    let candidates = find_ay_candidates(&graph, 100, 200);
    assert!(
        candidates.is_empty(),
        "no candidates for min_layers > graph size"
    );
}

// -- Integration: extract + validate round-trip ------------------------------

#[test]
fn test_extract_and_validate_polar_to_rect_subgraph() {
    // This is the target use case: extract the polar-to-rect conversion
    // (cos, sin, mul, mul, cat) from a larger graph for ay verification.
    let graph = build_polar_to_rect_graph();

    // Extract the computation-only nodes (skip the two Input nodes).
    let result = extract_subgraph(&graph, &SubgraphSpec::IndexRange { start: 2, end: 7 })
        .expect("extract polar-to-rect subgraph");

    // Validate self-containment.
    validate_subgraph(&result.graph).expect("extracted subgraph is self-contained");

    // The subgraph should have 5 computation nodes + 2 synthetic inputs.
    assert_eq!(result.graph.nodes().len(), 7);
    assert_eq!(result.synthetic_input_count, 2);
    assert_eq!(result.layer_count, 5);

    // Verify all ay-compatible.
    for node in result.graph.nodes() {
        assert!(
            super::is_ay_compatible_op(node.op()),
            "node '{}' op {:?} should be ay-compatible",
            node.name(),
            node.op(),
        );
    }

    // Output should be the Cat node.
    let output = result.graph.output_node().expect("has output");
    assert_eq!(output.name(), "spectral");
    assert_eq!(output.output_shape(), &[22]);
}

// -- is_ay_compatible_op tests -----------------------------------------------

#[test]
fn test_ay_compatible_element_wise_ops() {
    // Core element-wise ops should all be compatible.
    let compatible = vec![
        TraceOp::Add,
        TraceOp::Sub,
        TraceOp::Mul,
        TraceOp::Div,
        TraceOp::Relu,
        TraceOp::Gelu,
        TraceOp::GeluErf,
        TraceOp::Silu,
        TraceOp::Tanh,
        TraceOp::Sigmoid,
        TraceOp::Exp,
        TraceOp::Log,
        TraceOp::Sqrt,
        TraceOp::Sqr,
        TraceOp::Neg,
        TraceOp::Abs,
        TraceOp::Recip,
        TraceOp::Sin,
        TraceOp::Cos,
        TraceOp::MatMul,
    ];
    for op in &compatible {
        assert!(
            super::is_ay_compatible_op(op),
            "expected {op:?} to be ay-compatible"
        );
    }
}

#[test]
fn test_ay_compatible_parameterized_ops() {
    // Linear, LayerNorm, BatchNorm, Embedding, Softmax should be compatible.
    let linear = TraceOp::Linear {
        weight: WeightRef::new(vec![0.0; 32], vec![4, 8]).unwrap(),
        bias: None,
    };
    assert!(super::is_ay_compatible_op(&linear));

    let layer_norm = TraceOp::LayerNorm {
        eps: 1e-5,
        weight: WeightRef::new(vec![1.0; 8], vec![8]).unwrap(),
        bias: WeightRef::new(vec![0.0; 8], vec![8]).unwrap(),
    };
    assert!(super::is_ay_compatible_op(&layer_norm));

    let batch_norm = TraceOp::BatchNorm {
        eps: 1e-5,
        weight: WeightRef::new(vec![1.0; 4], vec![4]).unwrap(),
        bias: WeightRef::new(vec![0.0; 4], vec![4]).unwrap(),
        running_mean: WeightRef::new(vec![0.0; 4], vec![4]).unwrap(),
        running_var: WeightRef::new(vec![1.0; 4], vec![4]).unwrap(),
    };
    assert!(super::is_ay_compatible_op(&batch_norm));

    let embedding = TraceOp::Embedding {
        weight: WeightRef::new(vec![0.0; 40], vec![5, 8]).unwrap(),
    };
    assert!(super::is_ay_compatible_op(&embedding));

    let softmax = TraceOp::Softmax { dim: 1 };
    assert!(super::is_ay_compatible_op(&softmax));
}

#[test]
fn test_ay_compatible_conv1d_small_kernel() {
    // Conv1d with small kernel (4*4*3 = 48 <= 4096) should be compatible.
    let small_conv = TraceOp::Conv1d {
        weight: WeightRef::new(vec![0.0; 48], vec![4, 4, 3]).unwrap(),
        bias: None,
        padding: 1,
        stride: 1,
        dilation: 1,
        groups: 1,
    };
    assert!(
        super::is_ay_compatible_op(&small_conv),
        "small Conv1d should be ay-compatible"
    );
}

#[test]
fn test_ay_incompatible_conv1d_large_kernel() {
    // Conv1d with large kernel (256*128*7 = 229376 > 4096) should be incompatible.
    let large_conv = TraceOp::Conv1d {
        weight: WeightRef::new(vec![0.0; 229376], vec![256, 128, 7]).unwrap(),
        bias: None,
        padding: 3,
        stride: 1,
        dilation: 1,
        groups: 1,
    };
    assert!(
        !super::is_ay_compatible_op(&large_conv),
        "large Conv1d should NOT be ay-compatible"
    );
}

#[test]
fn test_ay_incompatible_ops() {
    // LSTM, SDPA, multi-head attention should NOT be compatible.
    let lstm = TraceOp::Lstm {
        weight_ih: WeightRef::new(vec![0.0; 64], vec![16, 4]).unwrap(),
        weight_hh: WeightRef::new(vec![0.0; 256], vec![16, 16]).unwrap(),
        bias_ih: None,
        bias_hh: None,
        hidden_size: 4,
        initial_hidden: None,
        initial_cell: None,
    };
    assert!(!super::is_ay_compatible_op(&lstm), "LSTM not ay-compatible");

    let sdpa = TraceOp::Sdpa { scale: 1.0 };
    assert!(!super::is_ay_compatible_op(&sdpa), "SDPA not ay-compatible");

    let mha = TraceOp::MultiHeadAttention {
        num_heads: 8,
        num_kv_heads: 8,
        head_dim: 64,
    };
    assert!(!super::is_ay_compatible_op(&mha), "MHA not ay-compatible");

    let conv_transpose = TraceOp::ConvTranspose1d {
        weight: WeightRef::new(vec![0.0; 48], vec![4, 4, 3]).unwrap(),
        bias: None,
        padding: 1,
        output_padding: 0,
        stride: 1,
        dilation: 1,
        groups: 1,
    };
    assert!(
        !super::is_ay_compatible_op(&conv_transpose),
        "ConvTranspose1d not ay-compatible"
    );
}

// -- AYCandidateRegion field tests -------------------------------------------

#[test]
fn test_candidate_entry_exit_nodes() {
    let graph = build_polar_to_rect_graph();
    // Find candidates of size 3 to 5 in the all-compatible graph.
    let candidates = find_ay_candidates(&graph, 3, 5);

    // Find the candidate covering indices 2-6 (cos, sin, mul, mul, cat).
    let candidate = candidates
        .iter()
        .find(|c| c.start_index == 2 && c.end_index == 7)
        .expect("should find candidate [2, 7)");

    // Entry nodes: the region depends on nodes 100 (magnitude) and 101 (phase)
    // which are at indices 0 and 1, outside the region.
    assert_eq!(candidate.entry_nodes.len(), 2);
    assert!(candidate.entry_nodes.contains(&100));
    assert!(candidate.entry_nodes.contains(&101));

    // Internal nodes: all 5 nodes in the region.
    assert_eq!(candidate.internal_nodes.len(), 5);
    assert!(candidate.internal_nodes.contains(&102)); // cos
    assert!(candidate.internal_nodes.contains(&103)); // sin
    assert!(candidate.internal_nodes.contains(&104)); // mul real
    assert!(candidate.internal_nodes.contains(&105)); // mul imag
    assert!(candidate.internal_nodes.contains(&106)); // cat

    // Exit nodes: node 106 (cat) is the last node, so it's always an exit.
    // No nodes outside the region reference nodes inside (this is the end of graph).
    assert!(candidate.exit_nodes.contains(&106));
}

#[test]
fn test_candidate_estimated_complexity() {
    let graph = build_polar_to_rect_graph();
    let candidates = find_ay_candidates(&graph, 3, 5);

    // All candidates should have non-zero estimated complexity.
    for c in &candidates {
        assert!(
            c.estimated_complexity > 0,
            "candidate [{}, {}) should have non-zero complexity",
            c.start_index,
            c.end_index,
        );
    }

    // The 5-node candidate (cos, sin, mul, mul, cat) should have complexity
    // based on output shapes: 11+11+11+11+22 = 66 (all element-wise).
    let full_candidate = candidates
        .iter()
        .find(|c| c.start_index == 2 && c.end_index == 7)
        .expect("should find candidate [2, 7)");
    assert_eq!(full_candidate.estimated_complexity, 66);
}

#[test]
fn test_candidate_complexity_includes_weight_params() {
    // Build a graph with a Linear layer to verify weight params add to complexity.
    let nodes = vec![
        TraceNode::new(
            300,
            "input".to_string(),
            TraceOp::Input,
            vec![],
            vec![4, 8],
            DType::F32,
        ),
        TraceNode::new(
            301,
            "linear".to_string(),
            TraceOp::Linear {
                weight: WeightRef::new(vec![0.0; 128], vec![16, 8]).unwrap(),
                bias: None,
            },
            vec![300],
            vec![4, 16],
            DType::F32,
        ),
        TraceNode::new(
            302,
            "relu".to_string(),
            TraceOp::Relu,
            vec![301],
            vec![4, 16],
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let candidates = find_ay_candidates(&graph, 3, 3);

    assert_eq!(candidates.len(), 1);
    let c = &candidates[0];
    // Input: 4*8 = 32. Linear: 4*16 + 128 (weights) = 192. Relu: 4*16 = 64.
    // Total: 32 + 192 + 64 = 288.
    assert_eq!(c.estimated_complexity, 288);
}

#[test]
fn test_candidate_op_types_populated() {
    let graph = build_polar_to_rect_graph();
    let candidates = find_ay_candidates(&graph, 3, 3);

    // Find the candidate starting at index 0 (Input, Input, Cos).
    let c = candidates
        .iter()
        .find(|c| c.start_index == 0)
        .expect("candidate at index 0");
    assert_eq!(c.op_types.len(), 3);
    assert_eq!(c.op_types[0], "Input");
    assert_eq!(c.op_types[1], "Input");
    assert_eq!(c.op_types[2], "Cos");
}
