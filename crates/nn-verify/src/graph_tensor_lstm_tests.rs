// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for LSTM NY translation (`graph_tensor_lstm.rs`).
//!
//! Extracted per #175 inline test extraction pattern.

use super::*;
use ndarray::{Array1, Array2};

fn weight_tensor_2d(data: Array2<f32>) -> TensorNodeValue {
    TensorNodeValue::WeightTensor(data.into_dyn())
}

fn weight_tensor_1d(data: Array1<f32>) -> TensorNodeValue {
    TensorNodeValue::WeightTensor(data.into_dyn())
}

/// Build a simple LSTM test case: input_size=2, hidden_size=3.
/// Weight matrices are identity-ish for testability.
fn build_test_node_values(with_bias: bool) -> Vec<TensorNodeValue> {
    let input_size = 2;
    let hidden_size = 3;
    let four_h = 4 * hidden_size;

    // node 0: input variable [input_size]
    // node 1: hidden variable [hidden_size]
    // node 2: cell variable [hidden_size]
    // node 3: weight_ih [4*H, I]
    // node 4: weight_hh [4*H, H]
    // node 5: bias [4*H] (optional)

    let w_ih = Array2::from_elem((four_h, input_size), 0.1f32);
    let w_hh = Array2::from_elem((four_h, hidden_size), 0.05f32);

    let mut values = vec![
        TensorNodeValue::Variable("input".to_string()),
        TensorNodeValue::Variable("hidden".to_string()),
        TensorNodeValue::Variable("cell".to_string()),
        weight_tensor_2d(w_ih),
        weight_tensor_2d(w_hh),
    ];

    if with_bias {
        let bias = Array1::from_elem(four_h, 0.01f32);
        values.push(weight_tensor_1d(bias));
    }

    values
}

#[test]
fn test_translate_lstm_no_bias() {
    let mut graph = GraphNetwork::new();
    let node_values = build_test_node_values(false);
    let result = translate_lstm(
        TensorNodeId::new(6),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        TensorNodeId::new(2),
        TensorNodeId::new(3),
        TensorNodeId::new(4),
        None,
        &node_values,
        &mut graph,
    );
    assert!(result.is_ok(), "translation failed: {result:?}");
    match result.unwrap() {
        TensorNodeValue::Variable(name) => {
            assert!(name.contains("lstm"), "name should contain lstm: {name}");
            assert!(name.contains("h_new"), "name should contain h_new: {name}");
        }
        other => panic!("expected Variable, got {other:?}"),
    }
}

#[test]
fn test_translate_lstm_with_bias() {
    let mut graph = GraphNetwork::new();
    let node_values = build_test_node_values(true);
    let result = translate_lstm(
        TensorNodeId::new(7),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        TensorNodeId::new(2),
        TensorNodeId::new(3),
        TensorNodeId::new(4),
        Some(TensorNodeId::new(5)),
        &node_values,
        &mut graph,
    );
    assert!(result.is_ok(), "translation failed: {result:?}");
    match result.unwrap() {
        TensorNodeValue::Variable(name) => {
            assert!(name.contains("lstm"), "name should contain lstm: {name}");
        }
        other => panic!("expected Variable, got {other:?}"),
    }
}

#[test]
fn test_translate_lstm_rejects_constant_input() {
    use crate::graph::FiniteF32;
    let mut graph = GraphNetwork::new();
    let mut node_values = build_test_node_values(false);
    node_values[0] = TensorNodeValue::Constant(FiniteF32::new(1.0).unwrap());
    let result = translate_lstm(
        TensorNodeId::new(6),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        TensorNodeId::new(2),
        TensorNodeId::new(3),
        TensorNodeId::new(4),
        None,
        &node_values,
        &mut graph,
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Variable"),
        "error should mention Variable: {err}"
    );
}

#[test]
fn test_translate_lstm_rejects_non_weight_matrix() {
    let mut graph = GraphNetwork::new();
    let node_values = vec![
        TensorNodeValue::Variable("input".to_string()),
        TensorNodeValue::Variable("hidden".to_string()),
        TensorNodeValue::Variable("cell".to_string()),
        TensorNodeValue::Variable("bad_weight".to_string()), // should be WeightTensor
        weight_tensor_2d(Array2::from_elem((12, 3), 0.1f32)),
    ];
    let result = translate_lstm(
        TensorNodeId::new(5),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        TensorNodeId::new(2),
        TensorNodeId::new(3),
        TensorNodeId::new(4),
        None,
        &node_values,
        &mut graph,
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("WeightTensor"),
        "error should mention WeightTensor: {err}"
    );
}

#[test]
fn test_translate_lstm_rejects_weight_hh_row_mismatch() {
    let mut graph = GraphNetwork::new();
    let input_size = 2;
    let hidden_size = 3;
    let four_h = 4 * hidden_size;
    let node_values = vec![
        TensorNodeValue::Variable("input".to_string()),
        TensorNodeValue::Variable("hidden".to_string()),
        TensorNodeValue::Variable("cell".to_string()),
        weight_tensor_2d(Array2::from_elem((four_h, input_size), 0.1f32)),
        // weight_hh has wrong row count: 8 instead of 12
        weight_tensor_2d(Array2::from_elem((8, hidden_size), 0.05f32)),
    ];
    let result = translate_lstm(
        TensorNodeId::new(5),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        TensorNodeId::new(2),
        TensorNodeId::new(3),
        TensorNodeId::new(4),
        None,
        &node_values,
        &mut graph,
    );
    assert!(result.is_err(), "should reject weight_hh row mismatch");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("weight_hh rows"),
        "error should mention weight_hh rows: {err}"
    );
}

#[test]
fn test_translate_lstm_rejects_bias_length_mismatch() {
    let mut graph = GraphNetwork::new();
    let input_size = 2;
    let hidden_size = 3;
    let four_h = 4 * hidden_size;
    let node_values = vec![
        TensorNodeValue::Variable("input".to_string()),
        TensorNodeValue::Variable("hidden".to_string()),
        TensorNodeValue::Variable("cell".to_string()),
        weight_tensor_2d(Array2::from_elem((four_h, input_size), 0.1f32)),
        weight_tensor_2d(Array2::from_elem((four_h, hidden_size), 0.05f32)),
        // bias has wrong length: 8 instead of 12
        weight_tensor_1d(Array1::from_elem(8, 0.01f32)),
    ];
    let result = translate_lstm(
        TensorNodeId::new(7),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        TensorNodeId::new(2),
        TensorNodeId::new(3),
        TensorNodeId::new(4),
        Some(TensorNodeId::new(5)),
        &node_values,
        &mut graph,
    );
    assert!(result.is_err(), "should reject bias length mismatch");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("bias length"),
        "error should mention bias length: {err}"
    );
}
