// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Regression test for #2400: MissingOutputPolicy::Error soundness.
//!
//! Verifies that `build_graph_network` rejects mismatched output tensor names
//! when `MissingOutputPolicy::Error` is active (as set at trace_to_graph.rs:279).
//! With the former `WarnAndFallback`, gamma-build silently picked the last-added
//! node, which could verify bounds on the WRONG output tensor.

use std::collections::{HashMap, HashSet};

use ny_build::{
    build_graph_network, DataType, GraphBuildInputs, GraphNetworkOptions, LayerSpec,
    MissingOutputPolicy, TensorSpec, WeightStore,
};
use ny_core::LayerType;

/// MissingOutputPolicy::Error must reject an output TensorSpec whose name
/// does not match any layer output. This is the soundness defense added in
/// commit 2835d62f8 for #2400. If someone reverts trace_to_graph.rs:279 to
/// WarnAndFallback, this test documents the expected behavior.
#[test]
fn test_mismatched_output_tensor_returns_error() {
    // One layer: ReLU activation.
    let layers = vec![LayerSpec {
        name: "layer_0".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input_0".to_string()],
        outputs: vec!["layer_0_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    }];

    let inputs = vec![TensorSpec {
        name: "input_0".to_string(),
        shape: vec![2, 4],
        dtype: DataType::Float32,
    }];

    // Deliberately wrong output name — does NOT match any layer output.
    let outputs = vec![TensorSpec {
        name: "nonexistent_tensor_42".to_string(),
        shape: vec![2, 4],
        dtype: DataType::Float32,
    }];

    let weights = WeightStore::new();
    let tensor_producer: HashMap<String, String> = HashMap::new();
    let constant_tensors: HashSet<String> = HashSet::new();
    let tensor_shapes: HashMap<String, Vec<i64>> = HashMap::new();

    let data = GraphBuildInputs {
        layers: &layers,
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        tensor_producer: &tensor_producer,
        constant_tensors: &constant_tensors,
        tensor_shapes: &tensor_shapes,
    };
    let options = GraphNetworkOptions {
        missing_output_policy: MissingOutputPolicy::Error,
        ..GraphNetworkOptions::default()
    };

    let result = build_graph_network(&data, options);
    let err = result.expect_err(
        "mismatched output name with MissingOutputPolicy::Error must fail, not silently fall back",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("output") || msg.contains("resolution") || msg.contains("missing"),
        "error should mention output resolution failure, got: {msg}"
    );
}

/// Complementary positive test: a correctly-named output resolves successfully
/// with MissingOutputPolicy::Error. Proves the Error policy doesn't break
/// valid graphs — only mismatched names.
#[test]
fn test_correct_output_tensor_succeeds_with_error_policy() {
    let layers = vec![LayerSpec {
        name: "layer_0".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input_0".to_string()],
        outputs: vec!["layer_0_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    }];

    let inputs = vec![TensorSpec {
        name: "input_0".to_string(),
        shape: vec![2, 4],
        dtype: DataType::Float32,
    }];

    // Correct output name — matches layer_0's output.
    let outputs = vec![TensorSpec {
        name: "layer_0_out".to_string(),
        shape: vec![2, 4],
        dtype: DataType::Float32,
    }];

    let weights = WeightStore::new();
    let mut tensor_producer: HashMap<String, String> = HashMap::new();
    tensor_producer.insert("layer_0_out".to_string(), "input_0".to_string());
    let constant_tensors: HashSet<String> = HashSet::new();
    let tensor_shapes: HashMap<String, Vec<i64>> = HashMap::new();

    let data = GraphBuildInputs {
        layers: &layers,
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        tensor_producer: &tensor_producer,
        constant_tensors: &constant_tensors,
        tensor_shapes: &tensor_shapes,
    };
    let options = GraphNetworkOptions {
        missing_output_policy: MissingOutputPolicy::Error,
        ..GraphNetworkOptions::default()
    };

    let network = build_graph_network(&data, options)
        .expect("correctly-named output should succeed with Error policy");
    assert!(
        network.num_nodes() >= 1,
        "network should have at least 1 node"
    );
}
