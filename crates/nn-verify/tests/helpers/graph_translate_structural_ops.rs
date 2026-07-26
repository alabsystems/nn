// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: structural tensor ops (Reshape, AxisSelect, Stack, Concat)
//! translate to NY GraphNetwork nodes.
//!
//! Strengthened for #1684:
//!   - AC1: Tests verify actual layer types, not just node counts
//!   - AC4: translate_concat has translation + IBP tests
//!   - AC5: Stack node-count assertions match comments
//!
//! IBP bounds propagation tests are in `graph_translate_structural_ops_ibp.rs`.

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};

/// Helper: count occurrences of a given layer type in the graph.
fn count_layer_type(graph: &nn_verify::GraphNetwork, layer_type: &str) -> usize {
    graph
        .node_names()
        .iter()
        .filter(|name| {
            graph
                .node(name)
                .map(|n| n.layer().layer_type() == layer_type)
                .unwrap_or(false)
        })
        .count()
}

/// Helper: collect all layer types in the graph (in node order).
fn layer_types(graph: &nn_verify::GraphNetwork) -> Vec<&'static str> {
    graph
        .node_names()
        .iter()
        .filter_map(|name| graph.node(name).map(|n| n.layer().layer_type()))
        .collect()
}

// ---------------------------------------------------------------------------
// Reshape
// ---------------------------------------------------------------------------

/// Reshape: variable [2,3,4] → [2,12] produces a graph with a Reshape layer.
/// AC1: verifies the output graph contains an actual Reshape layer, not just nodes.
#[test]
fn test_reshape_variable_produces_reshape_layer() {
    let def = TensorKernelDef::new(
        "reshape_var",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![2, 3, 4],
                },
                vec![2, 3, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reshape {
                    input: TensorNodeId::new(0),
                    target_shape: vec![2, 12],
                },
                vec![2, 12],
            ),
        ],
        TensorNodeId::new(1),
    );
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("Reshape variable translation must succeed");

    assert!(graph.num_nodes() > 0, "graph must have at least one node");

    // AC1: Verify the graph contains a Reshape layer.
    let reshape_count = count_layer_type(&graph, "Reshape");
    assert!(
        reshape_count >= 1,
        "Reshape translation must produce at least one Reshape layer, got types: {:?}",
        layer_types(&graph)
    );
}

/// Reshape: constant input passes through as constant (no graph node needed).
#[test]
fn test_reshape_constant_produces_graph() {
    let def = TensorKernelDef::new(
        "reshape_const",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "c".to_string(),
                    shape: vec![6],
                },
                vec![6],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reshape {
                    input: TensorNodeId::new(0),
                    target_shape: vec![2, 3],
                },
                vec![2, 3],
            ),
        ],
        TensorNodeId::new(1),
    );
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(1.0)])
        .expect("Reshape constant translation must succeed");
    // Constant reshape: constant folds through Reshape, output becomes AddConstant.
    assert!(graph.num_nodes() > 0, "graph must have at least one node");
    let add_const_count = count_layer_type(&graph, "AddConstant");
    assert!(
        add_const_count >= 1,
        "Constant reshape must produce AddConstant output layer, got types: {:?}",
        layer_types(&graph)
    );
}

// ---------------------------------------------------------------------------
// AxisSelect
// ---------------------------------------------------------------------------

/// AxisSelect: variable [2,4,8] axis=2 index=3 → [2,4] via Slice+Squeeze.
/// AC1: verifies the graph contains both Slice and Squeeze layers.
#[test]
fn test_axis_select_variable_produces_slice_and_squeeze() {
    let def = TensorKernelDef::new(
        "axis_select_var",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![2, 4, 8],
                },
                vec![2, 4, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::AxisSelect {
                    input: TensorNodeId::new(0),
                    axis: 2,
                    index: 3,
                },
                vec![2, 4],
            ),
        ],
        TensorNodeId::new(1),
    );
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("AxisSelect variable translation must succeed");

    // Should have at least Slice + Squeeze nodes.
    assert!(
        graph.num_nodes() >= 2,
        "AxisSelect needs Slice + Squeeze (got {} nodes)",
        graph.num_nodes()
    );

    // AC1: Verify the graph contains both Slice and Squeeze layers.
    let slice_count = count_layer_type(&graph, "Slice");
    let squeeze_count = count_layer_type(&graph, "Squeeze");
    assert!(
        slice_count >= 1,
        "AxisSelect must produce a Slice layer, got types: {:?}",
        layer_types(&graph)
    );
    assert!(
        squeeze_count >= 1,
        "AxisSelect must produce a Squeeze layer, got types: {:?}",
        layer_types(&graph)
    );
}

/// AxisSelect: constant input passes through unchanged.
#[test]
fn test_axis_select_constant_produces_graph() {
    let def = TensorKernelDef::new(
        "axis_select_const",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "c".to_string(),
                    shape: vec![2, 4, 8],
                },
                vec![2, 4, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::AxisSelect {
                    input: TensorNodeId::new(0),
                    axis: 2,
                    index: 0,
                },
                vec![2, 4],
            ),
        ],
        TensorNodeId::new(1),
    );
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(5.0)])
        .expect("AxisSelect constant translation must succeed");
    // Constant AxisSelect: constant folds through AxisSelect, output becomes AddConstant.
    assert!(graph.num_nodes() > 0, "graph must have at least one node");
    let add_const_count = count_layer_type(&graph, "AddConstant");
    assert!(
        add_const_count >= 1,
        "Constant AxisSelect must produce AddConstant output layer, got types: {:?}",
        layer_types(&graph)
    );
}

/// AxisSelect: axis=1 (minimum valid axis per tensor IR validation).
/// AC1: verify Slice+Squeeze layer types regardless of axis position.
#[test]
fn test_axis_select_axis_1_produces_slice_and_squeeze() {
    let def = TensorKernelDef::new(
        "axis_select_axis1",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![4, 8],
                },
                vec![4, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::AxisSelect {
                    input: TensorNodeId::new(0),
                    axis: 1,
                    index: 5,
                },
                vec![4],
            ),
        ],
        TensorNodeId::new(1),
    );
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("AxisSelect axis=1 must succeed");

    assert!(graph.num_nodes() >= 2);
    assert!(
        count_layer_type(&graph, "Slice") >= 1,
        "AxisSelect axis=1 must have Slice, got: {:?}",
        layer_types(&graph)
    );
    assert!(
        count_layer_type(&graph, "Squeeze") >= 1,
        "AxisSelect axis=1 must have Squeeze, got: {:?}",
        layer_types(&graph)
    );
}

// ---------------------------------------------------------------------------
// Stack
// ---------------------------------------------------------------------------

/// Stack: two variable [4,6] inputs along axis=2 → [4,6,2] via Unsqueeze+Concat.
/// AC1: verifies graph contains Unsqueeze and Concat layers.
/// AC5: assertion matches the comment (5 minimum for multi-var).
#[test]
fn test_stack_two_variables_produces_unsqueeze_and_concat() {
    let def = TensorKernelDef::new(
        "stack_2var",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![4, 6],
                },
                vec![4, 6],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".to_string(),
                    shape: vec![4, 6],
                },
                vec![4, 6],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Stack {
                    inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                    axis: 2,
                },
                vec![4, 6, 2],
            ),
        ],
        TensorNodeId::new(2),
    );
    let graph = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    )
    .expect("Stack two-variable translation must succeed");

    // 2 SliceLayer (multi-var input) + 2 Unsqueeze + 1 Concat = 5 minimum.
    // AC5: assertion now matches the comment.
    assert!(
        graph.num_nodes() >= 5,
        "Stack-2 needs SliceLayer+Unsqueeze+Concat (got {} nodes, types: {:?})",
        graph.num_nodes(),
        layer_types(&graph)
    );

    // AC1: Verify the graph contains Unsqueeze and Concat layers.
    let unsqueeze_count = count_layer_type(&graph, "Unsqueeze");
    let concat_count = count_layer_type(&graph, "Concat");
    assert!(
        unsqueeze_count >= 2,
        "Stack-2 must have at least 2 Unsqueeze layers (one per input), got {unsqueeze_count}, types: {:?}",
        layer_types(&graph)
    );
    assert!(
        concat_count >= 1,
        "Stack-2 must have at least 1 Concat layer, got {concat_count}, types: {:?}",
        layer_types(&graph)
    );
}

/// Stack: three variable inputs to verify pairwise concat chain.
/// AC1: verifies 3 Unsqueeze + 2 Concat layers.
/// AC5: assertion now matches the comment (8 minimum for 3-var).
#[test]
fn test_stack_three_variables_produces_unsqueeze_concat_chain() {
    let def = TensorKernelDef::new(
        "stack_3var",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![3, 4],
                },
                vec![3, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".to_string(),
                    shape: vec![3, 4],
                },
                vec![3, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Input {
                    name: "c".to_string(),
                    shape: vec![3, 4],
                },
                vec![3, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::Stack {
                    inputs: vec![
                        TensorNodeId::new(0),
                        TensorNodeId::new(1),
                        TensorNodeId::new(2),
                    ],
                    axis: 1,
                },
                vec![3, 3, 4],
            ),
        ],
        TensorNodeId::new(3),
    );
    let graph = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::Variable,
            TensorParamBinding::Variable,
            TensorParamBinding::Variable,
        ],
    )
    .expect("Stack three-variable translation must succeed");

    // 3 SliceLayer (multi-var) + 3 Unsqueeze + 2 Concat = 8 minimum.
    // AC5: assertion now matches the comment.
    assert!(
        graph.num_nodes() >= 8,
        "Stack-3 needs SliceLayer+Unsqueeze+Concat chain (got {} nodes, types: {:?})",
        graph.num_nodes(),
        layer_types(&graph)
    );

    // AC1: Verify layer type counts.
    let unsqueeze_count = count_layer_type(&graph, "Unsqueeze");
    let concat_count = count_layer_type(&graph, "Concat");
    assert!(
        unsqueeze_count >= 3,
        "Stack-3 must have at least 3 Unsqueeze layers (one per input), got {unsqueeze_count}"
    );
    assert!(
        concat_count >= 2,
        "Stack-3 must have at least 2 Concat layers (pairwise chain), got {concat_count}"
    );
}
