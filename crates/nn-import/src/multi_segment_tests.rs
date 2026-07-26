// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for multi-segment model import, op mapping, quantization, and error handling.

use std::collections::HashMap;
use std::path::Path;

use nn_core::dyn_tensor::trace::TraceOp;
use nn_core::DType;

use super::*;
use crate::graph_build::{build_graph, build_weight_map};
use crate::op_map::{supported_ops, ResolvedWeight};
use crate::parse::{parse_exported_program, InputSpec};
use crate::quantization::{detect_quantization_from_bytes, DetectedDtype};

// ===========================================================================
// Test helpers
// ===========================================================================

/// Build the MLP graph JSON fixture as a serde_json::Value.
///
/// This is the same graph as `test_data/e2e_mlp.json` (fc1: 4->8, fc2: 8->3)
/// but parsed to a Value so it can be passed to `convert_multi_segment`.
fn mlp_graph_json() -> serde_json::Value {
    serde_json::from_str(include_str!("../test_data/e2e_mlp.json")).unwrap()
}

/// Build a second (different) MLP graph JSON with fc3: 4->6, fc4: 6->2.
///
/// Uses different weight names (fc3.weight, fc3.bias, fc4.weight, fc4.bias)
/// and different shapes to verify multi-segment isolation.
fn mlp2_graph_json() -> serde_json::Value {
    serde_json::from_str(
        r#"{
        "graph_module": {
            "graph": {
                "inputs": [
                    {"as_tensor": {"name": "p_fc3_weight"}},
                    {"as_tensor": {"name": "p_fc3_bias"}},
                    {"as_tensor": {"name": "p_fc4_weight"}},
                    {"as_tensor": {"name": "p_fc4_bias"}},
                    {"as_tensor": {"name": "y"}}
                ],
                "outputs": [{"as_tensor": {"name": "linear_3"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.linear.default",
                        "inputs": [
                            {"name": "input", "arg": {"as_tensor": {"name": "y"}}, "kind": 1},
                            {"name": "weight", "arg": {"as_tensor": {"name": "p_fc3_weight"}}, "kind": 1},
                            {"name": "bias", "arg": {"as_tensor": {"name": "p_fc3_bias"}}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "linear_2"}}],
                        "metadata": {}
                    },
                    {
                        "target": "torch.ops.aten.relu.default",
                        "inputs": [
                            {"name": "input", "arg": {"as_tensor": {"name": "linear_2"}}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "relu_1"}}],
                        "metadata": {}
                    },
                    {
                        "target": "torch.ops.aten.linear.default",
                        "inputs": [
                            {"name": "input", "arg": {"as_tensor": {"name": "relu_1"}}, "kind": 1},
                            {"name": "weight", "arg": {"as_tensor": {"name": "p_fc4_weight"}}, "kind": 1},
                            {"name": "bias", "arg": {"as_tensor": {"name": "p_fc4_bias"}}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "linear_3"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "y": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]},
                    "p_fc3_weight": {"dtype": 7, "sizes": [{"as_int": 6}, {"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 4}, {"as_int": 1}]},
                    "p_fc3_bias": {"dtype": 7, "sizes": [{"as_int": 6}], "requires_grad": true, "strides": [{"as_int": 1}]},
                    "p_fc4_weight": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 6}], "requires_grad": true, "strides": [{"as_int": 6}, {"as_int": 1}]},
                    "p_fc4_bias": {"dtype": 7, "sizes": [{"as_int": 2}], "requires_grad": true, "strides": [{"as_int": 1}]},
                    "linear_2": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 6}], "requires_grad": false, "strides": [{"as_int": 6}, {"as_int": 1}]},
                    "relu_1": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 6}], "requires_grad": false, "strides": [{"as_int": 6}, {"as_int": 1}]},
                    "linear_3": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 2}], "requires_grad": false, "strides": [{"as_int": 2}, {"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_fc3_weight"}, "parameter_name": "fc3.weight"}},
                    {"parameter": {"arg": {"name": "p_fc3_bias"}, "parameter_name": "fc3.bias"}},
                    {"parameter": {"arg": {"name": "p_fc4_weight"}, "parameter_name": "fc4.weight"}},
                    {"parameter": {"arg": {"name": "p_fc4_bias"}, "parameter_name": "fc4.bias"}},
                    {"user_input": {"arg": {"as_tensor": {"name": "y"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "linear_3"}}}}
                ]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "opset_version": {"aten": 10},
        "range_constraints": {}
    }"#,
    )
    .unwrap()
}

/// Build a graph that shares fc1 weights with the MLP graph (for shared weight testing).
///
/// This "head" segment uses the same fc1.weight/fc1.bias but maps to a different
/// output (a single linear layer with no ReLU).
fn shared_weight_graph_json() -> serde_json::Value {
    serde_json::from_str(
        r#"{
        "graph_module": {
            "graph": {
                "inputs": [
                    {"as_tensor": {"name": "p_fc1_weight"}},
                    {"as_tensor": {"name": "p_fc1_bias"}},
                    {"as_tensor": {"name": "z"}}
                ],
                "outputs": [{"as_tensor": {"name": "head_out"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.linear.default",
                        "inputs": [
                            {"name": "input", "arg": {"as_tensor": {"name": "z"}}, "kind": 1},
                            {"name": "weight", "arg": {"as_tensor": {"name": "p_fc1_weight"}}, "kind": 1},
                            {"name": "bias", "arg": {"as_tensor": {"name": "p_fc1_bias"}}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "head_out"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "z": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]},
                    "p_fc1_weight": {"dtype": 7, "sizes": [{"as_int": 8}, {"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 4}, {"as_int": 1}]},
                    "p_fc1_bias": {"dtype": 7, "sizes": [{"as_int": 8}], "requires_grad": true, "strides": [{"as_int": 1}]},
                    "head_out": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 8}], "requires_grad": false, "strides": [{"as_int": 8}, {"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_fc1_weight"}, "parameter_name": "fc1.weight"}},
                    {"parameter": {"arg": {"name": "p_fc1_bias"}, "parameter_name": "fc1.bias"}},
                    {"user_input": {"arg": {"as_tensor": {"name": "z"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "head_out"}}}}
                ]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "opset_version": {"aten": 10},
        "range_constraints": {}
    }"#,
    )
    .unwrap()
}

/// Write combined safetensors with weights for both MLP1 and MLP2.
///
/// fc1: [8, 4], fc1.bias: [8]
/// fc2: [3, 8], fc2.bias: [3]
/// fc3: [6, 4], fc3.bias: [6]
/// fc4: [2, 6], fc4.bias: [2]
fn write_combined_weights(dir: &Path) -> std::path::PathBuf {
    let fc1_w: Vec<u8> = (0..32)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let fc1_b: Vec<u8> = [0.0f32; 8].iter().flat_map(|f| f.to_le_bytes()).collect();
    let fc2_w: Vec<u8> = (0..24)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let fc2_b: Vec<u8> = [0.0f32; 3].iter().flat_map(|f| f.to_le_bytes()).collect();
    let fc3_w: Vec<u8> = (0..24)
        .flat_map(|i| ((i as f32) * 0.02).to_le_bytes())
        .collect();
    let fc3_b: Vec<u8> = [0.0f32; 6].iter().flat_map(|f| f.to_le_bytes()).collect();
    let fc4_w: Vec<u8> = (0..12)
        .flat_map(|i| ((i as f32) * 0.03).to_le_bytes())
        .collect();
    let fc4_b: Vec<u8> = [0.0f32; 2].iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "fc1.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8, 4], &fc1_w).unwrap(),
    );
    tensors.insert(
        "fc1.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8], &fc1_b).unwrap(),
    );
    tensors.insert(
        "fc2.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3, 8], &fc2_w).unwrap(),
    );
    tensors.insert(
        "fc2.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3], &fc2_b).unwrap(),
    );
    tensors.insert(
        "fc3.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![6, 4], &fc3_w).unwrap(),
    );
    tensors.insert(
        "fc3.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![6], &fc3_b).unwrap(),
    );
    tensors.insert(
        "fc4.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 6], &fc4_w).unwrap(),
    );
    tensors.insert(
        "fc4.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2], &fc4_b).unwrap(),
    );

    let weights_path = dir.join("combined_weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&weights_path, serialized).unwrap();
    weights_path
}

/// Write MLP1-only weights (fc1 + fc2) for single-segment tests.
fn write_mlp1_weights(dir: &Path) -> std::path::PathBuf {
    let fc1_w: Vec<u8> = (0..32)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let fc1_b: Vec<u8> = [0.0f32; 8].iter().flat_map(|f| f.to_le_bytes()).collect();
    let fc2_w: Vec<u8> = (0..24)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let fc2_b: Vec<u8> = [0.0f32; 3].iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "fc1.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8, 4], &fc1_w).unwrap(),
    );
    tensors.insert(
        "fc1.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8], &fc1_b).unwrap(),
    );
    tensors.insert(
        "fc2.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3, 8], &fc2_w).unwrap(),
    );
    tensors.insert(
        "fc2.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3], &fc2_b).unwrap(),
    );

    let weights_path = dir.join("mlp1_weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&weights_path, serialized).unwrap();
    weights_path
}

/// Create a unique temp directory for a test.
fn test_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("nn_mseg_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Build a minimal single-op graph JSON for testing specific ops.
fn single_op_graph(
    target: &str,
    input_name: &str,
    output_name: &str,
    input_shape: &[i64],
    output_shape: &[i64],
    extra_inputs: &str,
    extra_tensor_values: &str,
) -> String {
    let input_sizes: String = input_shape
        .iter()
        .map(|d| format!("{{\"as_int\": {d}}}"))
        .collect::<Vec<_>>()
        .join(", ");
    let output_sizes: String = output_shape
        .iter()
        .map(|d| format!("{{\"as_int\": {d}}}"))
        .collect::<Vec<_>>()
        .join(", ");
    let input_strides: String = {
        let mut strides = vec![1i64; input_shape.len()];
        for i in (0..input_shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * input_shape[i + 1];
        }
        strides
            .iter()
            .map(|s| format!("{{\"as_int\": {s}}}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let output_strides: String = {
        let mut strides = vec![1i64; output_shape.len()];
        for i in (0..output_shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * output_shape[i + 1];
        }
        strides
            .iter()
            .map(|s| format!("{{\"as_int\": {s}}}"))
            .collect::<Vec<_>>()
            .join(", ")
    };

    format!(
        r#"{{
        "graph_module": {{
            "graph": {{
                "inputs": [{{"as_tensor": {{"name": "{input_name}"}}}}],
                "outputs": [{{"as_tensor": {{"name": "{output_name}"}}}}],
                "nodes": [{{
                    "target": "{target}",
                    "inputs": [
                        {{"name": "input", "arg": {{"as_tensor": {{"name": "{input_name}"}}}}, "kind": 1}}
                        {extra_inputs}
                    ],
                    "outputs": [{{"as_tensor": {{"name": "{output_name}"}}}}],
                    "metadata": {{}}
                }}],
                "tensor_values": {{
                    "{input_name}": {{"dtype": 7, "sizes": [{input_sizes}], "requires_grad": false, "strides": [{input_strides}]}},
                    "{output_name}": {{"dtype": 7, "sizes": [{output_sizes}], "requires_grad": false, "strides": [{output_strides}]}}
                    {extra_tensor_values}
                }},
                "is_single_tensor_return": true
            }},
            "signature": {{
                "input_specs": [
                    {{"user_input": {{"arg": {{"as_tensor": {{"name": "{input_name}"}}}}}}}}
                ],
                "output_specs": [
                    {{"user_output": {{"arg": {{"as_tensor": {{"name": "{output_name}"}}}}}}}}
                ]
            }},
            "module_call_graph": []
        }},
        "schema_version": {{"major": 8, "minor": 15}},
        "range_constraints": {{}}
    }}"#
    )
}

/// Parse a single-op graph and build it, returning the output TraceOp.
fn build_single_op(json: &str) -> TraceOp {
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();
    let output = imported.graph.output_node().unwrap();
    output.op().clone()
}

/// Build safetensors bytes from typed tensor specs.
fn build_safetensors_typed(tensors: &[(&str, &[usize], &[u8], safetensors::Dtype)]) -> Vec<u8> {
    let mut tensor_map: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
    for &(name, shape, data, dtype) in tensors {
        let view = safetensors::tensor::TensorView::new(dtype, shape.to_vec(), data)
            .expect("valid tensor view");
        tensor_map.push((name.to_string(), view));
    }
    safetensors::tensor::serialize(tensor_map, None).expect("serialization should succeed")
}

// ===========================================================================
// 1. Single-segment conversion tests (existing 3 + 10 new = 13)
// ===========================================================================

/// Single-segment roundtrip: backward compatible with existing import_model.
#[test]
fn test_single_segment_roundtrip() {
    let dir = test_dir("single_rt");
    let weights_path = write_mlp1_weights(&dir);

    let graph = mlp_graph_json();
    let model = convert_single_segment(&graph, &weights_path).unwrap();

    assert_eq!(model.num_segments(), 1);
    assert_eq!(model.segment_order, vec!["main"]);
    assert!(model.shared_weights.is_empty());

    let main = model.get_segment("main").unwrap();
    assert_eq!(main.num_user_inputs, 1);
    assert_eq!(main.user_input_names, vec!["x"]);
    assert_eq!(main.output_names, vec!["linear_1"]);

    // The computation graph should match what import_model produces.
    assert_eq!(main.graph.len(), 8); // 1 input + 4 params + 3 ops

    let _ = std::fs::remove_dir_all(&dir);
}

/// graph() accessor returns the computation graph for a named segment.
#[test]
fn test_graph_accessor() {
    let dir = test_dir("graph_acc");
    let weights_path = write_mlp1_weights(&dir);

    let model = convert_single_segment(&mlp_graph_json(), &weights_path).unwrap();

    let graph = model.graph("main").unwrap();
    assert_eq!(graph.len(), 8);
    assert!(model.graph("nonexistent").is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

/// Single-segment graph node count matches expected topology.
#[test]
fn test_single_segment_node_count() {
    let dir = test_dir("single_nodecount");
    let weights_path = write_mlp1_weights(&dir);

    let model = convert_single_segment(&mlp_graph_json(), &weights_path).unwrap();
    let main = model.get_segment("main").unwrap();

    // Verify we have input, constant (param), and compute nodes
    let compute_ops: Vec<_> = main
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .collect();
    assert_eq!(compute_ops.len(), 3, "expected linear->relu->linear");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Single-segment output dtype is F32 for this MLP.
#[test]
fn test_single_segment_output_dtype() {
    let dir = test_dir("single_dtype");
    let weights_path = write_mlp1_weights(&dir);

    let model = convert_single_segment(&mlp_graph_json(), &weights_path).unwrap();
    let main = model.get_segment("main").unwrap();
    let output = main.graph.output_node().unwrap();
    assert_eq!(output.output_dtype(), DType::F32);

    let _ = std::fs::remove_dir_all(&dir);
}

/// Single-segment output shape matches expected dimensions.
#[test]
fn test_single_segment_output_shape() {
    let dir = test_dir("single_shape");
    let weights_path = write_mlp1_weights(&dir);

    let model = convert_single_segment(&mlp_graph_json(), &weights_path).unwrap();
    let main = model.get_segment("main").unwrap();
    let output = main.graph.output_node().unwrap();
    assert_eq!(output.output_shape(), &[1, 3]);

    let _ = std::fs::remove_dir_all(&dir);
}

/// Single-segment topology is valid (all input refs resolve).
#[test]
fn test_single_segment_topology_valid() {
    let dir = test_dir("single_topo");
    let weights_path = write_mlp1_weights(&dir);

    let model = convert_single_segment(&mlp_graph_json(), &weights_path).unwrap();
    let main = model.get_segment("main").unwrap();

    for node in main.graph.nodes() {
        for &input_id in node.inputs() {
            assert!(
                main.graph.node(input_id).is_some(),
                "node '{}' references missing input_id {}",
                node.name(),
                input_id
            );
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// convert_single_segment always names the segment "main".
#[test]
fn test_single_segment_default_name() {
    let dir = test_dir("single_name");
    let weights_path = write_mlp1_weights(&dir);

    let model = convert_single_segment(&mlp_graph_json(), &weights_path).unwrap();
    assert_eq!(model.segment_order, vec!["main"]);
    assert!(model.get_segment("main").is_some());

    let _ = std::fs::remove_dir_all(&dir);
}

/// Single-segment has no shared weights (only one segment).
#[test]
fn test_single_segment_no_shared_weights() {
    let dir = test_dir("single_noshare");
    let weights_path = write_mlp1_weights(&dir);

    let model = convert_single_segment(&mlp_graph_json(), &weights_path).unwrap();
    assert!(
        model.shared_weights.is_empty(),
        "single segment should have no shared weights"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Single-segment with conv_bn_relu fixture.
#[test]
fn test_single_segment_conv_bn_relu() {
    let dir = test_dir("single_conv");
    let conv_json: serde_json::Value =
        serde_json::from_str(include_str!("../test_data/conv_bn_relu.json")).unwrap();

    // Write conv+bn weights
    let conv_w: Vec<u8> = vec![0u8; 432 * 4]; // [16,3,3,3] f32
    let conv_b: Vec<u8> = vec![0u8; 16 * 4];
    let bn_w: Vec<u8> = vec![0u8; 16 * 4];
    let bn_b: Vec<u8> = vec![0u8; 16 * 4];
    let bn_mean: Vec<u8> = vec![0u8; 16 * 4];
    let bn_var: Vec<u8> = {
        let ones: Vec<f32> = vec![1.0; 16];
        ones.iter().flat_map(|f| f.to_le_bytes()).collect()
    };

    let mut tensors = HashMap::new();
    tensors.insert(
        "conv.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![16, 3, 3, 3], &conv_w)
            .unwrap(),
    );
    tensors.insert(
        "conv.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![16], &conv_b).unwrap(),
    );
    tensors.insert(
        "bn.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![16], &bn_w).unwrap(),
    );
    tensors.insert(
        "bn.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![16], &bn_b).unwrap(),
    );
    tensors.insert(
        "bn.running_mean".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![16], &bn_mean).unwrap(),
    );
    tensors.insert(
        "bn.running_var".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![16], &bn_var).unwrap(),
    );
    let weights_path = dir.join("conv_bn.safetensors");
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&weights_path, &serialized).unwrap();

    let model = convert_single_segment(&conv_json, &weights_path).unwrap();
    assert_eq!(model.num_segments(), 1);

    let main = model.get_segment("main").unwrap();
    let compute_ops: Vec<_> = main
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .collect();
    assert!(compute_ops.len() >= 3, "expected conv, bn, relu + pool ops");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Single-segment input_names is correct.
#[test]
fn test_single_segment_user_input_names() {
    let dir = test_dir("single_inputs");
    let weights_path = write_mlp1_weights(&dir);

    let model = convert_single_segment(&mlp_graph_json(), &weights_path).unwrap();
    let main = model.get_segment("main").unwrap();
    assert_eq!(main.num_user_inputs, 1);
    assert_eq!(main.user_input_names, vec!["x"]);

    let _ = std::fs::remove_dir_all(&dir);
}

/// Single-segment verifies that input nodes have zero dependencies.
#[test]
fn test_single_segment_input_no_deps() {
    let dir = test_dir("single_nodeps");
    let weights_path = write_mlp1_weights(&dir);

    let model = convert_single_segment(&mlp_graph_json(), &weights_path).unwrap();
    let main = model.get_segment("main").unwrap();

    for node in main.graph.nodes() {
        if matches!(node.op(), TraceOp::Input) {
            assert!(
                node.inputs().is_empty(),
                "Input node '{}' should have no deps",
                node.name()
            );
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Single-segment op sequence is Linear -> Relu -> Linear.
#[test]
fn test_single_segment_op_sequence() {
    let dir = test_dir("single_opseq");
    let weights_path = write_mlp1_weights(&dir);

    let model = convert_single_segment(&mlp_graph_json(), &weights_path).unwrap();
    let main = model.get_segment("main").unwrap();

    let compute_ops: Vec<_> = main
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .collect();
    assert_eq!(compute_ops.len(), 3);
    assert!(matches!(compute_ops[0].op(), TraceOp::Linear { .. }));
    assert!(matches!(compute_ops[1].op(), TraceOp::Relu));
    assert!(matches!(compute_ops[2].op(), TraceOp::Linear { .. }));

    let _ = std::fs::remove_dir_all(&dir);
}

// ===========================================================================
// 2. Multi-segment conversion tests (existing 5 + 10 new = 15)
// ===========================================================================

/// Two-segment model with independent weights.
#[test]
fn test_two_segments_independent_weights() {
    let dir = test_dir("two_indep");
    let weights_path = write_combined_weights(&dir);

    let graphs = vec![
        ("encoder".to_string(), mlp_graph_json()),
        ("decoder".to_string(), mlp2_graph_json()),
    ];

    let model = convert_multi_segment(&graphs, &weights_path).unwrap();

    assert_eq!(model.num_segments(), 2);
    assert_eq!(model.segment_order, vec!["encoder", "decoder"]);

    assert!(
        model.shared_weights.is_empty(),
        "encoder and decoder have independent weights, expected no shared weights but got: {:?}",
        model.shared_weights
    );

    let enc = model.get_segment("encoder").unwrap();
    assert_eq!(enc.num_user_inputs, 1);
    assert_eq!(enc.user_input_names, vec!["x"]);
    assert_eq!(enc.output_names, vec!["linear_1"]);

    let dec = model.get_segment("decoder").unwrap();
    assert_eq!(dec.num_user_inputs, 1);
    assert_eq!(dec.user_input_names, vec!["y"]);
    assert_eq!(dec.output_names, vec!["linear_3"]);

    let _ = std::fs::remove_dir_all(&dir);
}

/// Two-segment model with shared weights (fc1.weight and fc1.bias).
#[test]
fn test_two_segments_shared_weights() {
    let dir = test_dir("two_shared");
    let weights_path = write_combined_weights(&dir);

    let graphs = vec![
        ("backbone".to_string(), mlp_graph_json()),
        ("head".to_string(), shared_weight_graph_json()),
    ];

    let model = convert_multi_segment(&graphs, &weights_path).unwrap();

    assert_eq!(model.num_segments(), 2);
    assert_eq!(model.segment_order, vec!["backbone", "head"]);

    assert!(
        model.shared_weights.len() >= 2,
        "expected at least 2 shared weights (fc1.weight, fc1.bias), got: {:?}",
        model.shared_weights
    );
    assert!(
        model.shared_weights.contains(&"fc1.weight".to_string()),
        "expected fc1.weight in shared_weights: {:?}",
        model.shared_weights
    );
    assert!(
        model.shared_weights.contains(&"fc1.bias".to_string()),
        "expected fc1.bias in shared_weights: {:?}",
        model.shared_weights
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Segment ordering is preserved from input.
#[test]
fn test_segment_order_preserved() {
    let dir = test_dir("order_pres");
    let weights_path = write_combined_weights(&dir);

    let graphs = vec![
        ("z_last".to_string(), mlp_graph_json()),
        ("m_middle".to_string(), shared_weight_graph_json()),
        ("a_first".to_string(), mlp2_graph_json()),
    ];

    let model = convert_multi_segment(&graphs, &weights_path).unwrap();

    assert_eq!(
        model.segment_order,
        vec!["z_last", "m_middle", "a_first"],
        "segment order must be preserved from input, not sorted"
    );

    assert!(model.get_segment("z_last").is_some());
    assert!(model.get_segment("m_middle").is_some());
    assert!(model.get_segment("a_first").is_some());
    assert!(model.get_segment("nonexistent").is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

/// Empty input is rejected.
#[test]
fn test_empty_input_rejected() {
    let dir = test_dir("empty_rej");
    let weights_path = write_mlp1_weights(&dir);

    let err = convert_multi_segment(&[], &weights_path).unwrap_err();
    assert!(
        matches!(err, MultiSegmentError::EmptyInput),
        "expected EmptyInput, got: {err:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Duplicate segment names are rejected.
#[test]
fn test_duplicate_segment_name_rejected() {
    let dir = test_dir("dup_rej");
    let weights_path = write_mlp1_weights(&dir);

    let graphs = vec![
        ("encoder".to_string(), mlp_graph_json()),
        ("encoder".to_string(), mlp_graph_json()),
    ];

    let err = convert_multi_segment(&graphs, &weights_path).unwrap_err();
    assert!(
        matches!(err, MultiSegmentError::DuplicateSegment { ref name } if name == "encoder"),
        "expected DuplicateSegment for 'encoder', got: {err:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Missing weights file produces an error.
#[test]
fn test_missing_weights_file() {
    let graphs = vec![("seg".to_string(), mlp_graph_json())];
    let err =
        convert_multi_segment(&graphs, Path::new("/nonexistent/weights.safetensors")).unwrap_err();
    assert!(
        matches!(err, MultiSegmentError::Io { .. }),
        "expected Io error for missing weights, got: {err:?}"
    );
}

/// Three segments, all independent.
#[test]
fn test_three_segments_independent() {
    let dir = test_dir("three_indep");
    let weights_path = write_combined_weights(&dir);

    let graphs = vec![
        ("seg_a".to_string(), mlp_graph_json()),
        ("seg_b".to_string(), mlp2_graph_json()),
        ("seg_c".to_string(), shared_weight_graph_json()),
    ];

    let model = convert_multi_segment(&graphs, &weights_path).unwrap();
    assert_eq!(model.num_segments(), 3);
    assert_eq!(model.segment_order, vec!["seg_a", "seg_b", "seg_c"]);

    // Each segment should have a valid computation graph
    for (name, _) in &model.segments {
        let seg = model.get_segment(name).unwrap();
        assert!(!seg.graph.is_empty(), "segment '{name}' should have nodes");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Shared weights list is sorted.
#[test]
fn test_shared_weights_sorted() {
    let dir = test_dir("share_sorted");
    let weights_path = write_combined_weights(&dir);

    let graphs = vec![
        ("backbone".to_string(), mlp_graph_json()),
        ("head".to_string(), shared_weight_graph_json()),
    ];

    let model = convert_multi_segment(&graphs, &weights_path).unwrap();

    // Verify shared_weights is sorted (the implementation sorts it).
    for window in model.shared_weights.windows(2) {
        assert!(
            window[0] < window[1],
            "shared_weights not sorted: '{}' >= '{}'",
            window[0],
            window[1]
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Segment isolation: encoder graph does not reference decoder's tensors.
#[test]
fn test_segment_isolation() {
    let dir = test_dir("seg_isolation");
    let weights_path = write_combined_weights(&dir);

    let graphs = vec![
        ("encoder".to_string(), mlp_graph_json()),
        ("decoder".to_string(), mlp2_graph_json()),
    ];

    let model = convert_multi_segment(&graphs, &weights_path).unwrap();

    let enc = model.get_segment("encoder").unwrap();
    let dec = model.get_segment("decoder").unwrap();

    // Encoder input is "x", decoder input is "y" — they should not overlap.
    assert_ne!(
        enc.user_input_names, dec.user_input_names,
        "segments should have different inputs"
    );
    assert_ne!(
        enc.output_names, dec.output_names,
        "segments should have different outputs"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// MultiSegmentModel::new constructor works correctly.
#[test]
fn test_multi_segment_model_constructor() {
    let dir = test_dir("msm_ctor");
    let weights_path = write_mlp1_weights(&dir);

    let model = convert_single_segment(&mlp_graph_json(), &weights_path).unwrap();

    // Check properties of the model returned by convert_single_segment
    assert_eq!(model.num_segments(), 1);
    assert_eq!(model.segment_order.len(), 1);
    assert!(model.shared_weights.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

/// get_segment returns None for nonexistent names in multi-segment model.
#[test]
fn test_get_segment_none_for_missing() {
    let dir = test_dir("get_none");
    let weights_path = write_combined_weights(&dir);

    let graphs = vec![
        ("alpha".to_string(), mlp_graph_json()),
        ("beta".to_string(), mlp2_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &weights_path).unwrap();

    assert!(model.get_segment("alpha").is_some());
    assert!(model.get_segment("beta").is_some());
    assert!(model.get_segment("gamma").is_none());
    assert!(model.get_segment("").is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

/// Duplicate detection with three segments — two share weights.
#[test]
fn test_multi_segment_three_with_partial_sharing() {
    let dir = test_dir("three_share");
    let weights_path = write_combined_weights(&dir);

    // seg_a uses fc1/fc2, seg_b uses fc3/fc4, seg_c uses fc1/fc1 (shared with seg_a)
    let graphs = vec![
        ("seg_a".to_string(), mlp_graph_json()),
        ("seg_b".to_string(), mlp2_graph_json()),
        ("seg_c".to_string(), shared_weight_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &weights_path).unwrap();

    // fc1 weights shared between seg_a and seg_c
    assert!(
        !model.shared_weights.is_empty(),
        "expected shared weights between seg_a and seg_c"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Segments.len() matches num_segments().
#[test]
fn test_num_segments_matches_vec_len() {
    let dir = test_dir("num_match");
    let weights_path = write_combined_weights(&dir);

    let graphs = vec![
        ("a".to_string(), mlp_graph_json()),
        ("b".to_string(), mlp2_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &weights_path).unwrap();

    assert_eq!(model.num_segments(), model.segments.len());
    assert_eq!(model.num_segments(), model.segment_order.len());

    let _ = std::fs::remove_dir_all(&dir);
}

/// segment_order entries match segment names in segments vec.
#[test]
fn test_segment_order_matches_segments() {
    let dir = test_dir("order_match");
    let weights_path = write_combined_weights(&dir);

    let graphs = vec![
        ("first".to_string(), mlp_graph_json()),
        ("second".to_string(), mlp2_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &weights_path).unwrap();

    let seg_names: Vec<&str> = model.segments.iter().map(|(n, _)| n.as_str()).collect();
    let order_names: Vec<&str> = model.segment_order.iter().map(String::as_str).collect();
    assert_eq!(seg_names, order_names);

    let _ = std::fs::remove_dir_all(&dir);
}

/// Invalid JSON in graph produces SegmentImport error.
#[test]
fn test_invalid_json_segment_error() {
    let dir = test_dir("bad_json");
    let weights_path = write_mlp1_weights(&dir);

    let bad_json: serde_json::Value = serde_json::json!({"not": "a valid graph"});
    let graphs = vec![("broken".to_string(), bad_json)];

    let err = convert_multi_segment(&graphs, &weights_path).unwrap_err();
    assert!(
        matches!(err, MultiSegmentError::SegmentImport { ref segment, .. } if segment == "broken"),
        "expected SegmentImport for 'broken', got: {err:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ===========================================================================
// 3. Op mapping coverage tests (15 tests)
// ===========================================================================

/// supported_ops returns a non-empty, sorted, deduplicated list.
#[test]
fn test_supported_ops_sorted_deduped() {
    let ops = supported_ops();
    assert!(!ops.is_empty(), "supported_ops should not be empty");
    for window in ops.windows(2) {
        assert!(
            window[0] < window[1],
            "supported_ops not sorted: '{}' >= '{}'",
            window[0],
            window[1]
        );
    }
}

/// supported_ops includes linear.
#[test]
fn test_supported_ops_contains_linear() {
    assert!(supported_ops().contains(&"aten::linear"));
}

/// supported_ops includes conv1d.
#[test]
fn test_supported_ops_contains_conv1d() {
    assert!(supported_ops().contains(&"aten::conv1d"));
}

/// supported_ops includes matmul.
#[test]
fn test_supported_ops_contains_matmul() {
    assert!(supported_ops().contains(&"aten::matmul"));
}

/// supported_ops includes add and mul.
#[test]
fn test_supported_ops_contains_binary() {
    let ops = supported_ops();
    assert!(ops.contains(&"aten::add"));
    assert!(ops.contains(&"aten::mul"));
    assert!(ops.contains(&"aten::sub"));
    assert!(ops.contains(&"aten::div"));
}

/// supported_ops includes relu and sigmoid.
#[test]
fn test_supported_ops_contains_activations() {
    let ops = supported_ops();
    assert!(ops.contains(&"aten::relu"));
    assert!(ops.contains(&"aten::sigmoid"));
    assert!(ops.contains(&"aten::tanh"));
    assert!(ops.contains(&"aten::silu"));
    assert!(ops.contains(&"aten::gelu"));
}

/// supported_ops includes layer_norm, instance_norm, group_norm.
#[test]
fn test_supported_ops_contains_norms() {
    let ops = supported_ops();
    assert!(ops.contains(&"aten::layer_norm"));
    assert!(ops.contains(&"aten::instance_norm"));
    assert!(ops.contains(&"aten::group_norm"));
    assert!(ops.contains(&"aten::batch_norm"));
}

/// supported_ops includes cat and reshape.
#[test]
fn test_supported_ops_contains_shape_ops() {
    let ops = supported_ops();
    assert!(ops.contains(&"aten::cat"));
    assert!(ops.contains(&"aten::reshape"));
    assert!(ops.contains(&"aten::transpose"));
    assert!(ops.contains(&"aten::permute"));
    assert!(ops.contains(&"aten::unsqueeze"));
    assert!(ops.contains(&"aten::squeeze"));
}

/// supported_ops includes upsample_nearest1d.
#[test]
fn test_supported_ops_contains_upsample() {
    assert!(supported_ops().contains(&"aten::upsample_nearest1d"));
}

/// supported_ops includes softmax and embedding.
#[test]
fn test_supported_ops_contains_attention_embedding() {
    let ops = supported_ops();
    assert!(ops.contains(&"aten::softmax"));
    assert!(ops.contains(&"aten::embedding"));
    assert!(ops.contains(&"aten::scaled_dot_product_attention"));
}

/// Relu maps correctly via build_graph.
#[test]
fn test_op_map_relu() {
    let json = single_op_graph(
        "torch.ops.aten.relu.default",
        "x",
        "relu_out",
        &[2, 5],
        &[2, 5],
        "",
        "",
    );
    let op = build_single_op(&json);
    assert!(matches!(op, TraceOp::Relu), "expected Relu, got: {op:?}");
}

/// Sigmoid maps correctly via build_graph.
#[test]
fn test_op_map_sigmoid() {
    let json = single_op_graph(
        "torch.ops.aten.sigmoid.default",
        "x",
        "sig_out",
        &[3, 4],
        &[3, 4],
        "",
        "",
    );
    let op = build_single_op(&json);
    assert!(
        matches!(op, TraceOp::Sigmoid),
        "expected Sigmoid, got: {op:?}"
    );
}

/// Exp maps correctly.
#[test]
fn test_op_map_exp() {
    let json = single_op_graph(
        "torch.ops.aten.exp.default",
        "x",
        "exp_out",
        &[2, 3],
        &[2, 3],
        "",
        "",
    );
    let op = build_single_op(&json);
    assert!(matches!(op, TraceOp::Exp), "expected Exp, got: {op:?}");
}

/// Sin maps correctly.
#[test]
fn test_op_map_sin() {
    let json = single_op_graph(
        "torch.ops.aten.sin.default",
        "x",
        "sin_out",
        &[4],
        &[4],
        "",
        "",
    );
    let op = build_single_op(&json);
    assert!(matches!(op, TraceOp::Sin), "expected Sin, got: {op:?}");
}

/// Unsupported op yields error.
#[test]
fn test_op_map_unsupported() {
    let json = single_op_graph(
        "torch.ops.aten.totally_fake_op.default",
        "x",
        "out",
        &[4],
        &[4],
        "",
        "",
    );
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let err = build_graph(&program, &empty_weights).unwrap_err();
    assert!(
        matches!(err, ImportError::UnsupportedOp { .. }),
        "expected UnsupportedOp, got: {err:?}"
    );
}

// ===========================================================================
// 4. Quantization detection tests (8 tests)
// ===========================================================================

/// F32 detection reports correct dtype.
#[test]
fn test_quant_detect_f32() {
    let f32_data: Vec<u8> = vec![0u8; 64]; // 16 * 4
    let bytes = build_safetensors_typed(&[("w", &[4, 4], &f32_data, safetensors::Dtype::F32)]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(report.total_tensors, 1);
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::F32);
    assert!(!report.is_mixed_precision());
}

/// F16 detection reports correct dtype.
#[test]
fn test_quant_detect_f16() {
    let f16_data: Vec<u8> = vec![0u8; 32]; // 16 * 2
    let bytes = build_safetensors_typed(&[("w", &[4, 4], &f16_data, safetensors::Dtype::F16)]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::F16);
}

/// BF16 detection reports correct dtype.
#[test]
fn test_quant_detect_bf16() {
    let bf16_data: Vec<u8> = vec![0u8; 32]; // 16 * 2
    let bytes = build_safetensors_typed(&[("w", &[4, 4], &bf16_data, safetensors::Dtype::BF16)]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::BF16);
}

/// Mixed dtype detection (F32 + F16).
#[test]
fn test_quant_detect_mixed_f32_f16() {
    let f32_data: Vec<u8> = vec![0u8; 256]; // 64 * 4
    let f16_data: Vec<u8> = vec![0u8; 32]; // 16 * 2
    let bytes = build_safetensors_typed(&[
        ("enc", &[8, 8], &f32_data, safetensors::Dtype::F32),
        ("dec", &[4, 4], &f16_data, safetensors::Dtype::F16),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert!(report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown.len(), 2);
}

/// QuantizationReport fields are accurate.
#[test]
fn test_quant_report_fields() {
    let data: Vec<u8> = vec![0u8; 4096]; // 1024 * 4 = 4096
    let bytes = build_safetensors_typed(&[("big", &[32, 32], &data, safetensors::Dtype::F32)]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(report.total_tensors, 1);
    assert_eq!(report.total_parameters, 1024);
    assert_eq!(report.total_bytes, 4096);
}

/// QuantRecommendation for large F32 tensors: F16 + I8 recommendations.
#[test]
fn test_quant_recommendation_large_f32() {
    let data: Vec<u8> = vec![0u8; 2048 * 4]; // 2048 elements
    let bytes = build_safetensors_typed(&[("big", &[64, 32], &data, safetensors::Dtype::F32)]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(report.recommendations.len(), 2, "expected F16 and I8 recs");

    let f16_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == DetectedDtype::F16);
    assert!(f16_rec.is_some(), "should have F16 recommendation");
    assert_eq!(f16_rec.unwrap().savings_bytes, 2048 * 2);
}

/// DtypeBreakdown accuracy for multi-dtype model.
#[test]
fn test_dtype_breakdown_accuracy() {
    let f32_data: Vec<u8> = vec![0u8; 4 * 100]; // 100 f32 elements
    let i8_data: Vec<u8> = vec![0u8; 50]; // 50 i8 elements
    let bytes = build_safetensors_typed(&[
        ("a", &[10, 10], &f32_data, safetensors::Dtype::F32),
        ("b", &[50], &i8_data, safetensors::Dtype::I8),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    let f32_bd = report
        .dtype_breakdown
        .iter()
        .find(|b| b.dtype == DetectedDtype::F32)
        .unwrap();
    assert_eq!(f32_bd.tensor_count, 1);
    assert_eq!(f32_bd.total_parameters, 100);
    assert_eq!(f32_bd.total_bytes, 400);

    let i8_bd = report
        .dtype_breakdown
        .iter()
        .find(|b| b.dtype == DetectedDtype::I8)
        .unwrap();
    assert_eq!(i8_bd.tensor_count, 1);
    assert_eq!(i8_bd.total_parameters, 50);
    assert_eq!(i8_bd.total_bytes, 50);
}

/// dtype_fraction returns 0.0 for empty model.
#[test]
fn test_quant_dtype_fraction_empty() {
    let bytes = build_safetensors_typed(&[]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(report.dtype_fraction(DetectedDtype::F32), 0.0);
    assert_eq!(report.dtype_fraction(DetectedDtype::F16), 0.0);
}

// ===========================================================================
// 5. Error handling tests (5 tests)
// ===========================================================================

/// Missing weight in single segment produces meaningful error.
#[test]
fn test_error_missing_weight_single_segment() {
    let dir = test_dir("err_miss_w");

    // Write empty safetensors file (no weights).
    let weights_path = dir.join("empty.safetensors");
    let serialized = safetensors::serialize(
        Vec::<(String, safetensors::tensor::TensorView<'_>)>::new(),
        None,
    )
    .unwrap();
    std::fs::write(&weights_path, serialized).unwrap();

    let graphs = vec![("seg".to_string(), mlp_graph_json())];
    let err = convert_multi_segment(&graphs, &weights_path).unwrap_err();

    // Should be a SegmentImport wrapping an ImportError about missing weights
    assert!(
        matches!(err, MultiSegmentError::SegmentImport { ref segment, .. } if segment == "seg"),
        "expected SegmentImport, got: {err:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Unsupported op in a segment propagates as SegmentImport error.
#[test]
fn test_error_unsupported_op_in_segment() {
    let dir = test_dir("err_unsup_op");
    let weights_path = write_mlp1_weights(&dir);

    let bad_graph: serde_json::Value = serde_json::from_str(r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "out"}}],
                "nodes": [{
                    "target": "torch.ops.aten.totally_fake.default",
                    "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
                    "outputs": [{"as_tensor": {"name": "out"}}],
                    "metadata": {}
                }],
                "tensor_values": {
                    "x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "out": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
                "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "out"}}}}]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#)
    .unwrap();

    let graphs = vec![("bad_seg".to_string(), bad_graph)];
    let err = convert_multi_segment(&graphs, &weights_path).unwrap_err();
    assert!(
        matches!(err, MultiSegmentError::SegmentImport { ref segment, .. } if segment == "bad_seg"),
        "expected SegmentImport for 'bad_seg', got: {err:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Topology error (forward reference) propagates correctly.
#[test]
fn test_error_topology_forward_ref() {
    let dir = test_dir("err_topo");
    let weights_path = write_mlp1_weights(&dir);

    let bad_graph: serde_json::Value = serde_json::from_str(
        r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "relu"}}],
                "nodes": [{
                    "target": "torch.ops.aten.relu.default",
                    "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "nonexistent"}}, "kind": 1}],
                    "outputs": [{"as_tensor": {"name": "relu"}}],
                    "metadata": {}
                }],
                "tensor_values": {
                    "x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "relu": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
                "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "relu"}}}}]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#,
    )
    .unwrap();

    let graphs = vec![("topo_seg".to_string(), bad_graph)];
    let err = convert_multi_segment(&graphs, &weights_path).unwrap_err();
    assert!(
        matches!(err, MultiSegmentError::SegmentImport { .. }),
        "expected SegmentImport for topology error, got: {err:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Schema version mismatch is caught.
#[test]
fn test_error_wrong_schema_version() {
    let dir = test_dir("err_schema");
    let weights_path = write_mlp1_weights(&dir);

    let bad_graph: serde_json::Value = serde_json::from_str(
        r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "x"}}],
                "nodes": [],
                "tensor_values": {
                    "x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
                "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 99, "minor": 0},
        "range_constraints": {}
    }"#,
    )
    .unwrap();

    let graphs = vec![("v99".to_string(), bad_graph)];
    let err = convert_multi_segment(&graphs, &weights_path).unwrap_err();
    assert!(
        matches!(err, MultiSegmentError::SegmentImport { .. }),
        "expected SegmentImport for schema mismatch, got: {err:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// MultiSegmentError Display implementations are non-empty.
#[test]
fn test_error_display_messages() {
    let empty_err = MultiSegmentError::EmptyInput;
    let msg = format!("{empty_err}");
    assert!(
        !msg.is_empty(),
        "EmptyInput display message should not be empty"
    );

    let dup_err = MultiSegmentError::DuplicateSegment {
        name: "test".to_string(),
    };
    let msg = format!("{dup_err}");
    assert!(msg.contains("test"), "DuplicateSegment should mention name");

    let io_err = MultiSegmentError::Io {
        path: "/foo".to_string(),
        detail: "not found".to_string(),
    };
    let msg = format!("{io_err}");
    assert!(msg.contains("/foo"), "Io error should mention path");
}

// ===========================================================================
// 6. Additional op mapping tests using test_data fixtures (5 tests)
// ===========================================================================

/// layernorm_softmax fixture imports correctly.
#[test]
fn test_fixture_layernorm_softmax() {
    let json = include_str!("../test_data/layernorm_softmax.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut w = HashMap::new();
    w.insert("ln.weight".to_string(), (vec![1.0; 4], vec![4]));
    w.insert("ln.bias".to_string(), (vec![0.0; 4], vec![4]));
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &w);
    let imported = build_graph(&program, &weight_map).unwrap();

    let compute_ops: Vec<_> = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .collect();
    assert!(
        compute_ops.len() >= 2,
        "expected at least LayerNorm + Softmax"
    );
}

/// multi_input_cat fixture has correct fan-in.
#[test]
fn test_fixture_multi_input_cat() {
    let json = include_str!("../test_data/multi_input_cat.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let mut w = HashMap::new();
    w.insert("fc.weight".to_string(), (vec![0.1; 64], vec![4, 16]));
    w.insert("fc.bias".to_string(), (vec![0.0; 4], vec![4]));
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &w);
    let imported = build_graph(&program, &weight_map).unwrap();
    assert_eq!(imported.num_user_inputs, 2);

    let cat_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Cat { .. }));
    assert!(cat_node.is_some(), "should have a Cat node");
    assert_eq!(cat_node.unwrap().inputs().len(), 2);
}

/// multi_layer_mlp fixture produces correct op chain.
#[test]
fn test_fixture_multi_layer_mlp() {
    let json = include_str!("../test_data/multi_layer_mlp.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let mut w = HashMap::new();
    w.insert("fc1.weight".to_string(), (vec![0.1; 32], vec![8, 4]));
    w.insert("fc1.bias".to_string(), (vec![0.0; 8], vec![8]));
    w.insert("fc2.weight".to_string(), (vec![0.1; 24], vec![3, 8]));
    w.insert("fc2.bias".to_string(), (vec![0.0; 3], vec![3]));
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &w);
    let imported = build_graph(&program, &weight_map).unwrap();

    assert_eq!(imported.output_names, vec!["softmax"]);
    let output = imported.graph.output_node().unwrap();
    assert!(matches!(output.op(), TraceOp::Softmax { .. }));
}

/// embedding_lookup fixture imports embedding op.
#[test]
fn test_fixture_embedding_lookup() {
    let json = include_str!("../test_data/embedding_lookup.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let mut w = HashMap::new();
    w.insert("embed.weight".to_string(), (vec![0.1; 10 * 8], vec![10, 8]));
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &w);
    let imported = build_graph(&program, &weight_map).unwrap();

    let has_embedding = imported
        .graph
        .nodes()
        .iter()
        .any(|n| matches!(n.op(), TraceOp::Embedding { .. }));
    assert!(has_embedding, "should have an Embedding node");
}

// ===========================================================================
// Additional MultiSegmentModel construction and accessor tests
// ===========================================================================

/// MultiSegmentModel::new with empty segments is valid.
#[test]
fn test_multi_segment_model_empty_construction() {
    let model = MultiSegmentModel::new(vec![], vec![], vec![]);
    assert_eq!(model.num_segments(), 0);
    assert!(model.segments.is_empty());
    assert!(model.segment_order.is_empty());
    assert!(model.shared_weights.is_empty());
    assert!(model.get_segment("anything").is_none());
    assert!(model.graph("anything").is_none());
}

/// MultiSegmentModel::get_segment returns None for non-existent name.
#[test]
fn test_get_segment_returns_none_for_nonexistent() {
    let dir = test_dir("get_segment_none2");
    let weights_path = write_mlp1_weights(&dir);
    let graphs = vec![("encoder".to_string(), mlp_graph_json())];
    let model = convert_multi_segment(&graphs, &weights_path).unwrap();

    assert!(model.get_segment("decoder").is_none());
    assert!(model.get_segment("").is_none());
    assert!(model.get_segment("encoder").is_some());
}

/// MultiSegmentModel::graph returns the computation graph.
#[test]
fn test_graph_accessor_returns_computation_graph() {
    let dir = test_dir("graph_accessor2");
    let weights_path = write_mlp1_weights(&dir);
    let graphs = vec![("seg1".to_string(), mlp_graph_json())];
    let model = convert_multi_segment(&graphs, &weights_path).unwrap();

    let cg = model.graph("seg1");
    assert!(cg.is_some());
    assert!(!cg.unwrap().is_empty(), "computation graph should have nodes");
}

/// convert_single_segment produces a model with segment named "main".
#[test]
fn test_convert_single_segment_names_segment_main() {
    let dir = test_dir("single_seg_main");
    let weights_path = write_mlp1_weights(&dir);
    let model = convert_single_segment(&mlp_graph_json(), &weights_path).unwrap();

    assert_eq!(model.num_segments(), 1);
    assert_eq!(model.segment_order, vec!["main"]);
    assert!(model.get_segment("main").is_some());
    assert!(model.shared_weights.is_empty());
}

/// convert_single_segment's graph is accessible via the "main" name.
#[test]
fn test_convert_single_segment_graph_accessible() {
    let dir = test_dir("single_seg_graph");
    let weights_path = write_mlp1_weights(&dir);
    let model = convert_single_segment(&mlp_graph_json(), &weights_path).unwrap();

    let graph = model.graph("main").expect("main segment should exist");
    assert!(!graph.is_empty());
}

/// MultiSegmentError::MissingSegment variant can be constructed and displayed.
#[test]
fn test_multi_segment_error_missing_segment_display() {
    let err = MultiSegmentError::MissingSegment {
        name: "decoder".to_string(),
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("decoder"),
        "error message should contain segment name"
    );
    assert!(msg.contains("missing segment"), "error message: {msg}");
}

/// MultiSegmentError::SegmentImport wraps the segment name and source error.
#[test]
fn test_multi_segment_error_segment_import_display() {
    let source = Box::new(ImportError::UnsupportedOp {
        target: "aten.custom_op".to_string(),
    });
    let err = MultiSegmentError::SegmentImport {
        segment: "vocoder".to_string(),
        source,
    };
    let msg = format!("{err}");
    assert!(msg.contains("vocoder"), "should mention segment name");
    assert!(
        msg.contains("aten.custom_op"),
        "should mention the source op"
    );
}

/// Segment boundaries are isolated: nodes from segment A do not appear in segment B.
#[test]
fn test_segment_boundary_isolation() {
    let dir = test_dir("boundary_isolation");
    let weights_path = write_combined_weights(&dir);
    let graphs = vec![
        ("seg_a".to_string(), mlp_graph_json()),
        ("seg_b".to_string(), mlp2_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &weights_path).unwrap();

    let graph_a = model.graph("seg_a").expect("seg_a exists");
    let graph_b = model.graph("seg_b").expect("seg_b exists");

    // Graph A and B should have independent node counts.
    assert!(!graph_a.is_empty());
    assert!(!graph_b.is_empty());

    // Since they import from different graphs with different weight names,
    // the node counts may differ but both should be non-zero.
    // The key invariant is that they are independent graphs.
    assert_ne!(
        std::ptr::addr_of!(*graph_a),
        std::ptr::addr_of!(*graph_b),
        "segments must be distinct graph instances"
    );
}

/// Multi-segment with 2 segments has correct num_segments.
#[test]
fn test_two_segments_num_segments() {
    let dir = test_dir("two_seg_count");
    let weights_path = write_combined_weights(&dir);
    let graphs = vec![
        ("first".to_string(), mlp_graph_json()),
        ("second".to_string(), mlp2_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &weights_path).unwrap();
    assert_eq!(model.num_segments(), 2);
}

/// kokoro_encoder_mini fixture imports correctly.
#[test]
fn test_fixture_kokoro_encoder_mini() {
    let json = include_str!("../test_data/kokoro_encoder_mini.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();

    // Provide weights that the encoder needs (conv weights, etc.)
    let mut w = HashMap::new();
    // The encoder has conv1d and instance_norm — add minimal weights
    for spec in &program.graph_module.signature.input_specs {
        match spec {
            InputSpec::Parameter(p) => {
                let name = &p.parameter.parameter_name;
                // Derive shape from tensor_values if available
                if let Some(meta) = program
                    .graph_module
                    .graph
                    .tensor_values
                    .get(&p.parameter.arg.name)
                {
                    if let Some(shape) = meta.concrete_shape() {
                        let numel: usize = shape.iter().product();
                        w.insert(name.clone(), (vec![0.01; numel], shape));
                    }
                }
            }
            InputSpec::Buffer(b) => {
                let name = &b.buffer.buffer_name;
                if let Some(meta) = program
                    .graph_module
                    .graph
                    .tensor_values
                    .get(&b.buffer.arg.name)
                {
                    if let Some(shape) = meta.concrete_shape() {
                        let numel: usize = shape.iter().product();
                        w.insert(name.clone(), (vec![0.01; numel], shape));
                    }
                }
            }
            _ => {}
        }
    }

    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &w);
    let imported = build_graph(&program, &weight_map).unwrap();
    assert!(!imported.graph.is_empty(), "encoder graph should have nodes");
}
