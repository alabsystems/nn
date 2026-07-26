// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Gated DeltaNet NY translation.

use super::*;
use crate::graph::FiniteF32;
use nn_dsl::tensor_ir::TensorOpKind;

/// Helper: create 6 Variable node values for q, k, v, state, gate, beta.
fn gdn_variables() -> Vec<TensorNodeValue> {
    vec![
        TensorNodeValue::Variable("q".to_string()),
        TensorNodeValue::Variable("k".to_string()),
        TensorNodeValue::Variable("v".to_string()),
        TensorNodeValue::Variable("state".to_string()),
        TensorNodeValue::Variable("gate".to_string()),
        TensorNodeValue::Variable("beta".to_string()),
    ]
}

/// Helper: build the 6 GDN `Input` tensor nodes (q,k,v,state,gate,beta) with
/// the canonical shapes for H=2, K=4, V=8 so the translator can derive dims.
fn gdn_input_nodes() -> Vec<TensorNode> {
    let (h, k, v) = (2usize, 4usize, 8usize);
    let mk = |idx: usize, name: &str, shape: Vec<usize>| {
        TensorNode::new(
            TensorNodeId::new(idx),
            TensorOpKind::Input {
                name: name.to_string(),
                shape: shape.clone(),
            },
            shape,
        )
    };
    vec![
        mk(0, "q", vec![h, k]),
        mk(1, "k", vec![h, k]),
        mk(2, "v", vec![h, v]),
        mk(3, "state", vec![h, k, v]),
        mk(4, "gate", vec![h, 1, 1]),
        mk(5, "beta", vec![h, 1]),
    ]
}

#[test]
fn test_translate_gdn_produces_variable_output() {
    let mut graph = GraphNetwork::new();
    let node_values = gdn_variables();
    let result = translate_gated_delta_net(
        TensorNodeId::new(6),
        TensorNodeId::new(0), // q
        TensorNodeId::new(1), // k
        TensorNodeId::new(2), // v
        TensorNodeId::new(3), // state
        TensorNodeId::new(4), // gate
        TensorNodeId::new(5), // beta
        0.125,
        &gdn_input_nodes(),
        &node_values,
        &mut graph,
    );
    assert!(result.is_ok(), "translation failed: {result:?}");
    match result.unwrap() {
        TensorNodeValue::Variable(name) => {
            assert_eq!(name, "t6_gdn_out");
        }
        other => panic!("expected Variable, got {other:?}"),
    }
}

#[test]
fn test_translate_gdn_node_count() {
    // The decomposition should produce exactly 9 NY nodes:
    // 1 decay (MulBinary) + 1 retrieval (MatMul) +
    // 2 beta scaling (MulBinary) + 2 outer products (MatMul) +
    // 2 state accumulation (Add) + 1 output query (MatMul) = 9
    let mut graph = GraphNetwork::new();
    let node_values = gdn_variables();
    let _ = translate_gated_delta_net(
        TensorNodeId::new(6),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        TensorNodeId::new(2),
        TensorNodeId::new(3),
        TensorNodeId::new(4),
        TensorNodeId::new(5),
        0.125,
        &gdn_input_nodes(),
        &node_values,
        &mut graph,
    )
    .expect("translation should succeed");

    // Count nodes by checking the graph has the expected node names.
    let expected_nodes = [
        "t6_gdn_decay",
        "t6_gdn_vr",
        "t6_gdn_beta_v",
        "t6_gdn_beta_vr",
        "t6_gdn_pos",
        "t6_gdn_neg",
        "t6_gdn_tmp",
        "t6_gdn_state",
        "t6_gdn_out",
    ];
    for name in expected_nodes {
        assert!(graph.node(name).is_some(), "missing expected node: {name}");
    }
}

#[test]
fn test_translate_gdn_rejects_constant_input() {
    let mut graph = GraphNetwork::new();
    let mut node_values = gdn_variables();
    // Replace q with a constant
    node_values[0] = TensorNodeValue::Constant(FiniteF32::new(1.0).unwrap());

    let result = translate_gated_delta_net(
        TensorNodeId::new(6),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        TensorNodeId::new(2),
        TensorNodeId::new(3),
        TensorNodeId::new(4),
        TensorNodeId::new(5),
        0.125,
        &gdn_input_nodes(),
        &node_values,
        &mut graph,
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("must be Variable"), "unexpected error: {err}");
}

#[test]
fn test_translate_gdn_rejects_weight_tensor_state() {
    let mut graph = GraphNetwork::new();
    let mut node_values = gdn_variables();
    // Replace state with WeightTensor
    let arr = ndarray::ArrayD::zeros(ndarray::IxDyn(&[2, 4, 8]));
    node_values[3] = TensorNodeValue::WeightTensor(arr);

    let result = translate_gated_delta_net(
        TensorNodeId::new(6),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        TensorNodeId::new(2),
        TensorNodeId::new(3),
        TensorNodeId::new(4),
        TensorNodeId::new(5),
        0.125,
        &gdn_input_nodes(),
        &node_values,
        &mut graph,
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("must be Variable"), "unexpected error: {err}");
}

#[test]
fn test_translate_gdn_different_scales() {
    // Verify translation succeeds with different scale values.
    for &scale in &[0.0625, 0.125, 0.5, 1.0] {
        let mut graph = GraphNetwork::new();
        let node_values = gdn_variables();
        let result = translate_gated_delta_net(
            TensorNodeId::new(6),
            TensorNodeId::new(0),
            TensorNodeId::new(1),
            TensorNodeId::new(2),
            TensorNodeId::new(3),
            TensorNodeId::new(4),
            TensorNodeId::new(5),
            scale,
            &gdn_input_nodes(),
            &node_values,
            &mut graph,
        );
        assert!(
            result.is_ok(),
            "translation failed for scale={scale}: {result:?}"
        );
    }
}

#[test]
fn test_translate_gdn_rejects_nan_scale() {
    let mut graph = GraphNetwork::new();
    let node_values = gdn_variables();
    let result = translate_gated_delta_net(
        TensorNodeId::new(6),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        TensorNodeId::new(2),
        TensorNodeId::new(3),
        TensorNodeId::new(4),
        TensorNodeId::new(5),
        f32::NAN,
        &gdn_input_nodes(),
        &node_values,
        &mut graph,
    );
    assert!(result.is_err(), "NaN scale should be rejected");
}

#[test]
fn test_translate_gdn_rejects_negative_scale() {
    let mut graph = GraphNetwork::new();
    let node_values = gdn_variables();
    let result = translate_gated_delta_net(
        TensorNodeId::new(6),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        TensorNodeId::new(2),
        TensorNodeId::new(3),
        TensorNodeId::new(4),
        TensorNodeId::new(5),
        -0.5,
        &gdn_input_nodes(),
        &node_values,
        &mut graph,
    );
    assert!(result.is_err(), "negative scale should be rejected");
}

#[test]
fn test_translate_gdn_rejects_zero_scale() {
    let mut graph = GraphNetwork::new();
    let node_values = gdn_variables();
    let result = translate_gated_delta_net(
        TensorNodeId::new(6),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        TensorNodeId::new(2),
        TensorNodeId::new(3),
        TensorNodeId::new(4),
        TensorNodeId::new(5),
        0.0,
        &gdn_input_nodes(),
        &node_values,
        &mut graph,
    );
    assert!(result.is_err(), "zero scale should be rejected");
}

#[test]
fn test_translate_gdn_rejects_inf_scale() {
    let mut graph = GraphNetwork::new();
    let node_values = gdn_variables();
    let result = translate_gated_delta_net(
        TensorNodeId::new(6),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        TensorNodeId::new(2),
        TensorNodeId::new(3),
        TensorNodeId::new(4),
        TensorNodeId::new(5),
        f32::INFINITY,
        &gdn_input_nodes(),
        &node_values,
        &mut graph,
    );
    assert!(result.is_err(), "infinite scale should be rejected");
}
