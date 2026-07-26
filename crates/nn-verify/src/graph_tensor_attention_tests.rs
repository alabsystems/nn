// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for attention tensor op translation to NY.

use super::*;
use crate::graph::FiniteF32;
use nn_dsl::tensor_ir::TensorOpKind;

/// Build dummy `TensorNode` entries for test fixtures.
/// All nodes get the given `shape` and a placeholder Input kind.
fn dummy_nodes(count: usize, shape: &[usize]) -> Vec<TensorNode> {
    (0..count)
        .map(|i| {
            TensorNode::new(
                TensorNodeId::new(i),
                TensorOpKind::Input {
                    name: format!("node_{i}"),
                    shape: shape.to_vec(),
                },
                shape.to_vec(),
            )
        })
        .collect()
}

#[test]
fn test_translate_attention_standard() {
    let mut graph = GraphNetwork::new();
    let nodes = dummy_nodes(4, &[2, 4, 8]);
    let node_values = vec![
        TensorNodeValue::Variable("query".to_string()),
        TensorNodeValue::Variable("key".to_string()),
        TensorNodeValue::Variable("value".to_string()),
    ];
    let result = translate_attention(
        TensorNodeId::new(3),
        &TensorNodeId::new(0),
        &TensorNodeId::new(1),
        &TensorNodeId::new(2),
        &AttentionMask::Standard,
        Some(0.125), // 1/sqrt(64)
        &node_values,
        &nodes,
        &mut graph,
    );
    assert!(result.is_ok(), "translation failed: {result:?}");
    match result.unwrap() {
        TensorNodeValue::Variable(name) => assert_eq!(name, "t3_attention"),
        other => panic!("expected Variable, got {other:?}"),
    }
}

#[test]
fn test_translate_attention_causal_no_scale() {
    let mut graph = GraphNetwork::new();
    let nodes = dummy_nodes(4, &[2, 4, 8]);
    let node_values = vec![
        TensorNodeValue::Variable("q".to_string()),
        TensorNodeValue::Variable("k".to_string()),
        TensorNodeValue::Variable("v".to_string()),
    ];
    let result = translate_attention(
        TensorNodeId::new(3),
        &TensorNodeId::new(0),
        &TensorNodeId::new(1),
        &TensorNodeId::new(2),
        &AttentionMask::Causal,
        None,
        &node_values,
        &nodes,
        &mut graph,
    );
    assert!(result.is_ok(), "translation failed: {result:?}");
}

#[test]
fn test_translate_attention_rejects_constant_q() {
    let mut graph = GraphNetwork::new();
    let nodes = dummy_nodes(4, &[2, 4, 8]);
    let node_values = vec![
        TensorNodeValue::Constant(FiniteF32::new(1.0).unwrap()),
        TensorNodeValue::Variable("key".to_string()),
        TensorNodeValue::Variable("value".to_string()),
    ];
    let result = translate_attention(
        TensorNodeId::new(3),
        &TensorNodeId::new(0),
        &TensorNodeId::new(1),
        &TensorNodeId::new(2),
        &AttentionMask::Standard,
        None,
        &node_values,
        &nodes,
        &mut graph,
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Attention Q must be Variable"),
        "error should mention Q requirement: {err}"
    );
}

#[test]
fn test_translate_attention_cross_attention_weight_tensor_kv() {
    let mut graph = GraphNetwork::new();
    // Q shape is [2, 2] — V WeightTensor must match.
    let nodes = dummy_nodes(4, &[2, 2]);
    let weight = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let node_values = vec![
        TensorNodeValue::Variable("query".to_string()),
        TensorNodeValue::Variable("key".to_string()),
        TensorNodeValue::WeightTensor(weight),
    ];
    // WeightTensor V is now accepted (cross-attention pattern).
    let result = translate_attention(
        TensorNodeId::new(3),
        &TensorNodeId::new(0),
        &TensorNodeId::new(1),
        &TensorNodeId::new(2),
        &AttentionMask::Standard,
        None,
        &node_values,
        &nodes,
        &mut graph,
    );
    assert!(
        result.is_ok(),
        "cross-attention with WeightTensor V should succeed: {result:?}"
    );
}

/// #830: Asymmetric cross-attention — a constant V whose value head dim differs
/// from Q is now ACCEPTED. The head-dim/contraction is enforced soundly by NY's
/// MatMul at propagation; only rank + leading batch dims must match at translation.
#[test]
fn test_translate_attention_accepts_asymmetric_value_head_dim() {
    let mut graph = GraphNetwork::new();
    // Q [2, 4] (seq=2, d=4); V constant [2, 8] (seq=2, value-dim=8) — value head
    // dim legitimately differs from Q.
    let nodes = dummy_nodes(4, &[2, 4]);
    let v_const = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 8]), vec![0.0; 16]).unwrap();
    let node_values = vec![
        TensorNodeValue::Variable("query".to_string()),
        TensorNodeValue::Variable("key".to_string()),
        TensorNodeValue::WeightTensor(v_const),
    ];
    let result = translate_attention(
        TensorNodeId::new(3),
        &TensorNodeId::new(0),
        &TensorNodeId::new(1),
        &TensorNodeId::new(2),
        &AttentionMask::Standard,
        None,
        &node_values,
        &nodes,
        &mut graph,
    );
    assert!(
        result.is_ok(),
        "asymmetric value head dim should be accepted: {result:?}"
    );
}

/// #830: Asymmetric cross-attention — a constant K whose sequence length (KV_SEQ)
/// differs from Q's (Q_SEQ) is now ACCEPTED (head dim matches).
#[test]
fn test_translate_attention_accepts_asymmetric_kv_seq() {
    let mut graph = GraphNetwork::new();
    // Q [2, 4] (Q_SEQ=2); K constant [3, 4] (KV_SEQ=3) — sequence lengths differ.
    let nodes = dummy_nodes(4, &[2, 4]);
    let k_const = ArrayD::from_shape_vec(ndarray::IxDyn(&[3, 4]), vec![0.0; 12]).unwrap();
    let node_values = vec![
        TensorNodeValue::Variable("query".to_string()),
        TensorNodeValue::WeightTensor(k_const),
        TensorNodeValue::Variable("value".to_string()),
    ];
    let result = translate_attention(
        TensorNodeId::new(3),
        &TensorNodeId::new(0),
        &TensorNodeId::new(1),
        &TensorNodeId::new(2),
        &AttentionMask::Standard,
        None,
        &node_values,
        &nodes,
        &mut graph,
    );
    assert!(
        result.is_ok(),
        "asymmetric KV_SEQ should be accepted: {result:?}"
    );
}

/// #830: a constant K/V whose RANK or leading batch dims (e.g. head count) differ
/// from Q is a malformed spec and must still be rejected.
#[test]
fn test_translate_attention_rejects_rank_and_batch_mismatch() {
    // (a) rank mismatch: Q rank-2, K constant rank-3.
    let mut graph = GraphNetwork::new();
    let nodes = dummy_nodes(4, &[2, 4]);
    let k_rank3 = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 4, 8]), vec![0.0; 64]).unwrap();
    let node_values = vec![
        TensorNodeValue::Variable("query".to_string()),
        TensorNodeValue::WeightTensor(k_rank3),
        TensorNodeValue::Variable("value".to_string()),
    ];
    let result = translate_attention(
        TensorNodeId::new(3),
        &TensorNodeId::new(0),
        &TensorNodeId::new(1),
        &TensorNodeId::new(2),
        &AttentionMask::Standard,
        None,
        &node_values,
        &nodes,
        &mut graph,
    );
    assert!(result.is_err(), "rank mismatch should be rejected");

    // (b) leading batch / head-count mismatch: Q [2,2,4] (heads=2) vs K [3,2,4] (heads=3).
    let mut graph2 = GraphNetwork::new();
    let nodes2 = dummy_nodes(4, &[2, 2, 4]);
    let k_badheads = ArrayD::from_shape_vec(ndarray::IxDyn(&[3, 2, 4]), vec![0.0; 24]).unwrap();
    let node_values2 = vec![
        TensorNodeValue::Variable("query".to_string()),
        TensorNodeValue::WeightTensor(k_badheads),
        TensorNodeValue::Variable("value".to_string()),
    ];
    let result2 = translate_attention(
        TensorNodeId::new(3),
        &TensorNodeId::new(0),
        &TensorNodeId::new(1),
        &TensorNodeId::new(2),
        &AttentionMask::Standard,
        None,
        &node_values2,
        &nodes2,
        &mut graph2,
    );
    assert!(
        result2.is_err(),
        "leading batch-dim (head count) mismatch should be rejected"
    );
}
