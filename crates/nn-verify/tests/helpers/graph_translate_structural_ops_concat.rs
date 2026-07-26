// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Concat tensor op translation to NY GraphNetwork.
//!
//! Extracted from `graph_translate_structural_ops.rs` — Part of #1678.
//!
//! AC4 (#1684): translate_concat has translation + error tests.

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
// Concat (AC4: previously zero test coverage)
// ---------------------------------------------------------------------------

/// Concat: two variable [3,4] inputs along axis=1 → [3,8].
/// AC4: first translation test for translate_concat.
#[test]
fn test_concat_two_variables_produces_concat_layer() {
    let def = TensorKernelDef::new(
        "concat_2var",
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
                TensorOpKind::Concat {
                    inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                    axis: 1,
                },
                vec![3, 8],
            ),
        ],
        TensorNodeId::new(2),
    );
    let graph = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    )
    .expect("Concat two-variable translation must succeed");

    // Multi-var: 2 SliceLayer (input splitting) + 1 ConcatLayer = 3 minimum.
    assert!(
        graph.num_nodes() >= 3,
        "Concat-2 needs SliceLayer+Concat (got {} nodes, types: {:?})",
        graph.num_nodes(),
        layer_types(&graph)
    );

    // AC1+AC4: Verify the graph contains a Concat layer (not Unsqueeze — unlike Stack).
    let concat_count = count_layer_type(&graph, "Concat");
    assert!(
        concat_count >= 1,
        "Concat translation must produce at least 1 Concat layer, got types: {:?}",
        layer_types(&graph)
    );

    // Concat must NOT produce Unsqueeze layers (that's Stack's job).
    let unsqueeze_count = count_layer_type(&graph, "Unsqueeze");
    assert_eq!(
        unsqueeze_count,
        0,
        "Concat must not produce Unsqueeze layers (that's Stack), got types: {:?}",
        layer_types(&graph)
    );
}

/// Concat: three variable [2,5] inputs along axis=1 → [2,15].
/// Verifies pairwise concat chain for 3+ inputs (same fold pattern as Stack).
#[test]
fn test_concat_three_variables_produces_concat_chain() {
    let def = TensorKernelDef::new(
        "concat_3var",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![2, 5],
                },
                vec![2, 5],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".to_string(),
                    shape: vec![2, 5],
                },
                vec![2, 5],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Input {
                    name: "c".to_string(),
                    shape: vec![2, 5],
                },
                vec![2, 5],
            ),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::Concat {
                    inputs: vec![
                        TensorNodeId::new(0),
                        TensorNodeId::new(1),
                        TensorNodeId::new(2),
                    ],
                    axis: 1,
                },
                vec![2, 15],
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
    .expect("Concat three-variable translation must succeed");

    // Multi-var: 3 SliceLayer + 2 ConcatLayer = 5 minimum.
    assert!(
        graph.num_nodes() >= 5,
        "Concat-3 needs SliceLayer+Concat chain (got {} nodes, types: {:?})",
        graph.num_nodes(),
        layer_types(&graph)
    );

    // 2 Concat layers for pairwise fold of 3 inputs.
    let concat_count = count_layer_type(&graph, "Concat");
    assert!(
        concat_count >= 2,
        "Concat-3 must have at least 2 Concat layers (pairwise chain), got {concat_count}"
    );
    // No Unsqueeze (Concat joins along existing axis).
    assert_eq!(
        count_layer_type(&graph, "Unsqueeze"),
        0,
        "Concat must not produce Unsqueeze layers"
    );
}

/// Concat with constant input must be rejected.
#[test]
fn test_concat_rejects_constant_input() {
    let def = TensorKernelDef::new(
        "concat_const_reject",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![2, 3],
                },
                vec![2, 3],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".to_string(),
                    shape: vec![2, 3],
                },
                vec![2, 3],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Concat {
                    inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                    axis: 1,
                },
                vec![2, 6],
            ),
        ],
        TensorNodeId::new(2),
    );
    let result = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantScalar(1.0),
        ],
    );
    assert!(result.is_err(), "Concat with constant input must fail");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("constant"),
        "error should mention 'constant', got: {err_msg}"
    );
}

/// Concat with fewer than 2 inputs must be rejected.
#[test]
fn test_concat_rejects_single_input() {
    let def = TensorKernelDef::new(
        "concat_single_reject",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![2, 3],
                },
                vec![2, 3],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Concat {
                    inputs: vec![TensorNodeId::new(0)],
                    axis: 1,
                },
                vec![2, 3],
            ),
        ],
        TensorNodeId::new(1),
    );
    let result = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]);
    assert!(result.is_err(), "Concat with single input must fail");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("at least two") || err_msg.contains("fewer than 2"),
        "error should mention input count requirement, got: {err_msg}"
    );
}
