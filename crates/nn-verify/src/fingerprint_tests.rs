// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for subgraph fingerprinting (#2457).

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::*;

/// Helper: build a minimal trace node with the given op and output shape.
fn make_node(id: u64, name: &str, op: TraceOp, inputs: Vec<u64>, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(id, name.to_string(), op, inputs, shape, DType::F32)
}

/// Helper: build a `ComputationGraph` from a list of nodes.
fn make_graph(nodes: Vec<TraceNode>) -> ComputationGraph {
    ComputationGraph::from_nodes(nodes)
}

#[test]
fn test_fingerprint_identical_graphs_match() {
    let nodes = vec![
        make_node(1, "input_0", TraceOp::Input, vec![], vec![1, 3, 8]),
        make_node(2, "relu_0", TraceOp::Relu, vec![1], vec![1, 3, 8]),
    ];
    let fp1 = fingerprint_trace(&nodes);

    let nodes2 = vec![
        make_node(10, "input_0", TraceOp::Input, vec![], vec![1, 3, 8]),
        make_node(20, "relu_0", TraceOp::Relu, vec![10], vec![1, 3, 8]),
    ];
    let fp2 = fingerprint_trace(&nodes2);

    assert_eq!(fp1.len(), 2);
    assert_eq!(fp2.len(), 2);
    // Same ops and shapes => same hashes (node IDs are not included in hash).
    assert_eq!(fp1[0].hash, fp2[0].hash);
    assert_eq!(fp1[1].hash, fp2[1].hash);
}

#[test]
fn test_fingerprint_different_ops_differ() {
    let node_relu = make_node(1, "relu_0", TraceOp::Relu, vec![], vec![1, 3, 8]);
    let node_silu = make_node(1, "silu_0", TraceOp::Silu, vec![], vec![1, 3, 8]);

    let fp_relu = fingerprint_trace(&[node_relu]);
    let fp_silu = fingerprint_trace(&[node_silu]);

    assert_ne!(fp_relu[0].hash, fp_silu[0].hash);
}

#[test]
fn test_fingerprint_different_shapes_differ() {
    let node_a = make_node(1, "relu_0", TraceOp::Relu, vec![], vec![1, 3, 8]);
    let node_b = make_node(1, "relu_0", TraceOp::Relu, vec![], vec![1, 3, 16]);

    let fp_a = fingerprint_trace(&[node_a]);
    let fp_b = fingerprint_trace(&[node_b]);

    assert_ne!(fp_a[0].hash, fp_b[0].hash);
}

#[test]
fn test_fingerprint_different_hyperparams_differ() {
    let node_a = make_node(
        1,
        "conv1d_0",
        TraceOp::Conv1d {
            weight: WeightRef::from_shape(&[16, 3, 3]),
            bias: None,
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        },
        vec![],
        vec![1, 16, 8],
    );
    let node_b = make_node(
        1,
        "conv1d_0",
        TraceOp::Conv1d {
            weight: WeightRef::from_shape(&[16, 3, 3]),
            bias: None,
            padding: 1,
            stride: 2, // different stride
            dilation: 1,
            groups: 1,
        },
        vec![],
        vec![1, 16, 4],
    );

    let fp_a = fingerprint_trace(&[node_a]);
    let fp_b = fingerprint_trace(&[node_b]);

    assert_ne!(fp_a[0].hash, fp_b[0].hash);
}

#[test]
fn test_fingerprint_weight_content_mode() {
    let w1 = WeightRef::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("valid weight");
    let w2 = WeightRef::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).expect("valid weight");

    let node_a = make_node(
        1,
        "linear_0",
        TraceOp::Linear {
            weight: w1,
            bias: None,
        },
        vec![],
        vec![1, 2],
    );
    let node_b = make_node(
        2,
        "linear_0",
        TraceOp::Linear {
            weight: w2,
            bias: None,
        },
        vec![],
        vec![1, 2],
    );

    // Structural fingerprint: same shape => same hash
    let fp_struct_a = fingerprint_trace(std::slice::from_ref(&node_a));
    let fp_struct_b = fingerprint_trace(std::slice::from_ref(&node_b));
    assert_eq!(fp_struct_a[0].hash, fp_struct_b[0].hash);

    // Parametric fingerprint: different content => different hash
    let fp_param_a = fingerprint_trace_with_weights(&[node_a]);
    let fp_param_b = fingerprint_trace_with_weights(&[node_b]);
    assert_ne!(fp_param_a[0].hash, fp_param_b[0].hash);
}

#[test]
fn test_diff_identical() {
    let nodes = vec![
        make_node(1, "input_0", TraceOp::Input, vec![], vec![1, 3, 8]),
        make_node(2, "relu_0", TraceOp::Relu, vec![1], vec![1, 3, 8]),
    ];
    let fp = fingerprint_trace(&nodes);

    let changes = diff_fingerprints(&fp, &fp);
    assert!(
        changes.is_empty(),
        "identical graphs should have no changes"
    );
}

#[test]
fn test_diff_one_changed() {
    let old_nodes = vec![
        make_node(1, "input_0", TraceOp::Input, vec![], vec![1, 3, 8]),
        make_node(2, "relu_0", TraceOp::Relu, vec![1], vec![1, 3, 8]),
        make_node(3, "relu_1", TraceOp::Relu, vec![2], vec![1, 3, 8]),
    ];
    let new_nodes = vec![
        make_node(1, "input_0", TraceOp::Input, vec![], vec![1, 3, 8]),
        make_node(2, "silu_0", TraceOp::Silu, vec![1], vec![1, 3, 8]), // changed
        make_node(3, "relu_1", TraceOp::Relu, vec![2], vec![1, 3, 8]),
    ];

    let old_fp = fingerprint_trace(&old_nodes);
    let new_fp = fingerprint_trace(&new_nodes);
    let changes = diff_fingerprints(&old_fp, &new_fp);

    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].start, 1);
    assert_eq!(changes[0].end, 2);
    assert_eq!(changes[0].reason, ChangeReason::OpChanged);
}

#[test]
fn test_diff_inserted_nodes() {
    let old_nodes = vec![make_node(
        1,
        "input_0",
        TraceOp::Input,
        vec![],
        vec![1, 3, 8],
    )];
    let new_nodes = vec![
        make_node(1, "input_0", TraceOp::Input, vec![], vec![1, 3, 8]),
        make_node(2, "relu_0", TraceOp::Relu, vec![1], vec![1, 3, 8]),
    ];

    let old_fp = fingerprint_trace(&old_nodes);
    let new_fp = fingerprint_trace(&new_nodes);
    let changes = diff_fingerprints(&old_fp, &new_fp);

    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].start, 1);
    assert_eq!(changes[0].end, 2);
    assert_eq!(changes[0].reason, ChangeReason::Inserted);
}

#[test]
fn test_diff_removed_nodes() {
    let old_nodes = vec![
        make_node(1, "input_0", TraceOp::Input, vec![], vec![1, 3, 8]),
        make_node(2, "relu_0", TraceOp::Relu, vec![1], vec![1, 3, 8]),
    ];
    let new_nodes = vec![make_node(
        1,
        "input_0",
        TraceOp::Input,
        vec![],
        vec![1, 3, 8],
    )];

    let old_fp = fingerprint_trace(&old_nodes);
    let new_fp = fingerprint_trace(&new_nodes);
    let changes = diff_fingerprints(&old_fp, &new_fp);

    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].start, 1);
    assert_eq!(changes[0].end, 2);
    assert_eq!(changes[0].reason, ChangeReason::Removed);
}

#[test]
fn test_diff_multiple_changed_regions() {
    let old_nodes = vec![
        make_node(1, "input_0", TraceOp::Input, vec![], vec![1, 3, 8]),
        make_node(2, "relu_0", TraceOp::Relu, vec![1], vec![1, 3, 8]),
        make_node(3, "relu_1", TraceOp::Relu, vec![2], vec![1, 3, 8]),
        make_node(4, "relu_2", TraceOp::Relu, vec![3], vec![1, 3, 8]),
    ];
    let new_nodes = vec![
        make_node(1, "input_0", TraceOp::Input, vec![], vec![1, 3, 8]),
        make_node(2, "silu_0", TraceOp::Silu, vec![1], vec![1, 3, 8]), // changed
        make_node(3, "relu_1", TraceOp::Relu, vec![2], vec![1, 3, 8]),
        make_node(4, "tanh_0", TraceOp::Tanh, vec![3], vec![1, 3, 8]), // changed
    ];

    let old_fp = fingerprint_trace(&old_nodes);
    let new_fp = fingerprint_trace(&new_nodes);
    let changes = diff_fingerprints(&old_fp, &new_fp);

    assert_eq!(changes.len(), 2);
    assert_eq!(changes[0].start, 1);
    assert_eq!(changes[0].end, 2);
    assert_eq!(changes[1].start, 3);
    assert_eq!(changes[1].end, 4);
}

#[test]
fn test_fingerprint_graph_convenience() {
    let nodes = vec![
        make_node(1, "input_0", TraceOp::Input, vec![], vec![1, 3, 8]),
        make_node(2, "relu_0", TraceOp::Relu, vec![1], vec![1, 3, 8]),
    ];
    let graph = make_graph(nodes.clone());

    let fp_direct = fingerprint_trace(&nodes);
    let fp_graph = fingerprint_graph(&graph);

    assert_eq!(fp_direct.len(), fp_graph.len());
    for (a, b) in fp_direct.iter().zip(fp_graph.iter()) {
        assert_eq!(a.hash, b.hash);
    }
}

#[test]
fn test_op_summary_populated() {
    let nodes = vec![
        make_node(1, "input_0", TraceOp::Input, vec![], vec![1, 3, 8]),
        make_node(
            2,
            "linear_0",
            TraceOp::Linear {
                weight: WeightRef::from_shape(&[4, 8]),
                bias: None,
            },
            vec![1],
            vec![1, 3, 4],
        ),
    ];
    let fps = fingerprint_trace(&nodes);

    assert_eq!(fps[0].op_summary, "input");
    assert_eq!(fps[1].op_summary, "linear");
}

#[test]
fn test_empty_graph() {
    let fps = fingerprint_trace(&[]);
    assert!(fps.is_empty());

    let changes = diff_fingerprints(&[], &[]);
    assert!(changes.is_empty());
}

#[test]
fn test_change_reason_display() {
    assert_eq!(ChangeReason::OpChanged.to_string(), "op_changed");
    assert_eq!(ChangeReason::ShapeChanged.to_string(), "shape_changed");
    assert_eq!(ChangeReason::WeightChanged.to_string(), "weight_changed");
    assert_eq!(ChangeReason::Inserted.to_string(), "inserted");
    assert_eq!(ChangeReason::Removed.to_string(), "removed");
}
