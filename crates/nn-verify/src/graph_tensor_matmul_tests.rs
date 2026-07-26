// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `graph_tensor_matmul` — MatMul tensor-level IR → NY translation.

use super::*;
use crate::graph::FiniteF32;

#[test]
fn test_translate_matmul_variable_variable() {
    let mut graph = GraphNetwork::new();
    let node_values = vec![
        TensorNodeValue::Variable("query".to_string()),
        TensorNodeValue::Variable("key".to_string()),
    ];
    let result = translate_matmul(
        TensorNodeId::new(2),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        true,        // transpose_right for Q @ K^T
        Some(0.125), // scale = 1/sqrt(64)
        &node_values,
        &mut graph,
    );
    assert!(result.is_ok());
    match result.unwrap() {
        TensorNodeValue::Variable(name) => assert_eq!(name, "t2_matmul"),
        other => panic!("expected Variable, got {other:?}"),
    }
}

#[test]
fn test_translate_matmul_no_transpose_no_scale() {
    let mut graph = GraphNetwork::new();
    let node_values = vec![
        TensorNodeValue::Variable("attn_weights".to_string()),
        TensorNodeValue::Variable("value".to_string()),
    ];
    let result = translate_matmul(
        TensorNodeId::new(2),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        false,
        None,
        &node_values,
        &mut graph,
    );
    assert!(result.is_ok());
    match result.unwrap() {
        TensorNodeValue::Variable(name) => assert_eq!(name, "t2_matmul"),
        other => panic!("expected Variable, got {other:?}"),
    }
}

#[test]
fn test_translate_matmul_constant_constant_folds() {
    let mut graph = GraphNetwork::new();
    let node_values = vec![
        TensorNodeValue::Constant(FiniteF32::new(3.0).unwrap()),
        TensorNodeValue::Constant(FiniteF32::new(4.0).unwrap()),
    ];
    let result = translate_matmul(
        TensorNodeId::new(2),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        false,
        None,
        &node_values,
        &mut graph,
    );
    match result.expect("constant fold should succeed") {
        TensorNodeValue::Constant(c) => assert_eq!(c.get(), 12.0),
        other => panic!("expected Constant, got {other:?}"),
    }
}

#[test]
fn test_translate_matmul_constant_constant_with_scale() {
    let mut graph = GraphNetwork::new();
    let node_values = vec![
        TensorNodeValue::Constant(FiniteF32::new(2.0).unwrap()),
        TensorNodeValue::Constant(FiniteF32::new(5.0).unwrap()),
    ];
    let result = translate_matmul(
        TensorNodeId::new(2),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        false,
        Some(0.5),
        &node_values,
        &mut graph,
    );
    match result.expect("constant fold with scale should succeed") {
        TensorNodeValue::Constant(c) => assert_eq!(c.get(), 5.0), // 2*5*0.5
        other => panic!("expected Constant, got {other:?}"),
    }
}

#[test]
fn test_translate_matmul_variable_zero_constant_folds() {
    let mut graph = GraphNetwork::new();
    let node_values = vec![
        TensorNodeValue::Variable("query".to_string()),
        TensorNodeValue::Constant(FiniteF32::new(0.0).unwrap()),
    ];
    let result = translate_matmul(
        TensorNodeId::new(2),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        false,
        None,
        &node_values,
        &mut graph,
    );
    match result.expect("Variable * zero should fold") {
        TensorNodeValue::Constant(c) => assert_eq!(c.get(), 0.0),
        other => panic!("expected Constant(0.0), got {other:?}"),
    }
}

#[test]
fn test_translate_matmul_zero_constant_variable_folds() {
    let mut graph = GraphNetwork::new();
    let node_values = vec![
        TensorNodeValue::Constant(FiniteF32::new(0.0).unwrap()),
        TensorNodeValue::Variable("key".to_string()),
    ];
    let result = translate_matmul(
        TensorNodeId::new(2),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        false,
        None,
        &node_values,
        &mut graph,
    );
    match result.expect("zero * Variable should fold") {
        TensorNodeValue::Constant(c) => assert_eq!(c.get(), 0.0),
        other => panic!("expected Constant(0.0), got {other:?}"),
    }
}

#[test]
fn test_translate_matmul_rejects_nonzero_constant_variable() {
    let mut graph = GraphNetwork::new();
    let node_values = vec![
        TensorNodeValue::Constant(FiniteF32::new(1.0).unwrap()),
        TensorNodeValue::Variable("key".to_string()),
    ];
    let result = translate_matmul(
        TensorNodeId::new(2),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        true,
        None,
        &node_values,
        &mut graph,
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("unsupported operand"), "error: {err}");
}

#[test]
fn test_translate_matmul_variable_weight_tensor_no_transpose() {
    let mut graph = GraphNetwork::new();
    // W is [2, 3] — Variable([..., 2]) @ W → LinearLayer(W^T [3, 2])
    let weight =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.0, -1.0, 0.5, 0.5, 0.0]).unwrap();
    let node_values = vec![
        TensorNodeValue::Variable("query".to_string()),
        TensorNodeValue::WeightTensor(weight),
    ];
    let result = translate_matmul(
        TensorNodeId::new(2),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        false,
        None,
        &node_values,
        &mut graph,
    );
    assert!(result.is_ok(), "Variable × WeightTensor should succeed");
    match result.unwrap() {
        TensorNodeValue::Variable(name) => assert_eq!(name, "t2_matmul"),
        other => panic!("expected Variable, got {other:?}"),
    }
}

#[test]
fn test_translate_matmul_variable_weight_tensor_with_transpose_and_scale() {
    let mut graph = GraphNetwork::new();
    // W is [3, 2], transpose_right=true → x @ W^T uses W directly as LinearLayer weight
    let weight =
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.5]).unwrap();
    let node_values = vec![
        TensorNodeValue::Variable("query".to_string()),
        TensorNodeValue::WeightTensor(weight),
    ];
    let result = translate_matmul(
        TensorNodeId::new(2),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        true,
        Some(0.5),
        &node_values,
        &mut graph,
    );
    assert!(
        result.is_ok(),
        "Variable × WeightTensor^T with scale should succeed"
    );
    match result.unwrap() {
        TensorNodeValue::Variable(name) => assert_eq!(name, "t2_matmul"),
        other => panic!("expected Variable, got {other:?}"),
    }
}

#[test]
fn test_translate_matmul_weight_tensor_variable() {
    let mut graph = GraphNetwork::new();
    // W is [3, 2] — W @ Variable([..., 2]) → LinearLayer(W [3, 2])
    let weight =
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.5]).unwrap();
    let node_values = vec![
        TensorNodeValue::WeightTensor(weight),
        TensorNodeValue::Variable("key".to_string()),
    ];
    let result = translate_matmul(
        TensorNodeId::new(2),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        false,
        None,
        &node_values,
        &mut graph,
    );
    assert!(result.is_ok(), "WeightTensor × Variable should succeed");
    match result.unwrap() {
        TensorNodeValue::Variable(name) => assert_eq!(name, "t2_matmul"),
        other => panic!("expected Variable, got {other:?}"),
    }
}

#[test]
fn test_translate_matmul_weight_tensor_weight_tensor_folds() {
    let mut graph = GraphNetwork::new();
    // [2, 3] @ [3, 2] = [2, 2]
    let left = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let right = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let node_values = vec![
        TensorNodeValue::WeightTensor(left),
        TensorNodeValue::WeightTensor(right),
    ];
    let result = translate_matmul(
        TensorNodeId::new(2),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        false,
        None,
        &node_values,
        &mut graph,
    );
    match result.expect("WeightTensor × WeightTensor fold should succeed") {
        TensorNodeValue::WeightTensor(arr) => {
            assert_eq!(arr.shape(), &[2, 2]);
            // [[1,0,0] @ [[1,2],[3,4],[5,6]]] = [[1,2], [3,4]]
            let flat: Vec<f32> = arr.iter().copied().collect();
            assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0]);
        }
        other => panic!("expected WeightTensor, got {other:?}"),
    }
}

#[test]
fn test_translate_matmul_3d_variable_weight_tensor_succeeds() {
    let mut graph = GraphNetwork::new();
    // 3D weight [2, 1, 3] — batch of 2, matmul dims [1,3].
    // Variable × WT: batch linear decomposition.
    let weight =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), (1..=6).map(|x| x as f32).collect()).unwrap();
    let node_values = vec![
        TensorNodeValue::Variable("query".to_string()),
        TensorNodeValue::WeightTensor(weight),
    ];
    let result = translate_matmul(
        TensorNodeId::new(2),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        false,
        None,
        &node_values,
        &mut graph,
    );
    assert!(
        result.is_ok(),
        "3-D Variable×WeightTensor should succeed via batch decomposition: {:?}",
        result.err()
    );
    match result.unwrap() {
        TensorNodeValue::Variable(name) => assert_eq!(name, "t2_matmul"),
        other => panic!("expected Variable, got {other:?}"),
    }
}

#[test]
fn test_translate_matmul_3d_weight_tensor_weight_tensor_folds() {
    let mut graph = GraphNetwork::new();
    // [2, 1, 3] @ [2, 3, 1] = [2, 1, 1] — batched outer product
    let left =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right =
        ArrayD::from_shape_vec(IxDyn(&[2, 3, 1]), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let node_values = vec![
        TensorNodeValue::WeightTensor(left),
        TensorNodeValue::WeightTensor(right),
    ];
    let result = translate_matmul(
        TensorNodeId::new(2),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        false,
        None,
        &node_values,
        &mut graph,
    );
    match result.expect("3D WT×WT fold should succeed") {
        TensorNodeValue::WeightTensor(arr) => {
            assert_eq!(arr.shape(), &[2, 1, 1]);
            // batch 0: [1,2,3]@[1,0,0]^T = 1, batch 1: [4,5,6]@[0,1,0]^T = 5
            let flat: Vec<f32> = arr.iter().copied().collect();
            assert_eq!(flat, vec![1.0, 5.0]);
        }
        other => panic!("expected WeightTensor, got {other:?}"),
    }
}

#[test]
fn test_translate_matmul_3d_weight_tensor_variable_succeeds() {
    let mut graph = GraphNetwork::new();
    // WT[2, 3, 2] × Variable — batch linear decomposition (weight_is_left=true).
    let weight =
        ArrayD::from_shape_vec(IxDyn(&[2, 3, 2]), (1..=12).map(|x| x as f32).collect()).unwrap();
    let node_values = vec![
        TensorNodeValue::WeightTensor(weight),
        TensorNodeValue::Variable("key".to_string()),
    ];
    let result = translate_matmul(
        TensorNodeId::new(2),
        TensorNodeId::new(0),
        TensorNodeId::new(1),
        false,
        None,
        &node_values,
        &mut graph,
    );
    assert!(
        result.is_ok(),
        "3-D WeightTensor×Variable should succeed: {:?}",
        result.err()
    );
    match result.unwrap() {
        TensorNodeValue::Variable(name) => assert_eq!(name, "t2_matmul"),
        other => panic!("expected Variable, got {other:?}"),
    }
}
