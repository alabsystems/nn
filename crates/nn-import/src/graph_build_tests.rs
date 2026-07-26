// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for graph building from parsed torch.export programs.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;

use super::*;
use crate::op_map::ResolvedWeight;
use crate::parse::parse_exported_program;

/// Minimal linear model JSON (same as in parse_tests).
fn minimal_linear_json() -> &'static str {
    r#"{
        "graph_module": {
            "graph": {
                "inputs": [
                    {"as_tensor": {"name": "p_weight"}},
                    {"as_tensor": {"name": "p_bias"}},
                    {"as_tensor": {"name": "x"}}
                ],
                "outputs": [{"as_tensor": {"name": "linear"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.linear.default",
                        "inputs": [
                            {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
                            {"name": "weight", "arg": {"as_tensor": {"name": "p_weight"}}, "kind": 1},
                            {"name": "bias", "arg": {"as_tensor": {"name": "p_bias"}}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "linear"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "x": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]},
                    "p_weight": {"dtype": 7, "sizes": [{"as_int": 3}, {"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 4}, {"as_int": 1}]},
                    "p_bias": {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": true, "strides": [{"as_int": 1}]},
                    "linear": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 3}, {"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_weight"}, "parameter_name": "weight"}},
                    {"parameter": {"arg": {"name": "p_bias"}, "parameter_name": "bias"}},
                    {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "linear"}}}}
                ]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "opset_version": {"aten": 10},
        "range_constraints": {}
    }"#
}

fn make_weights() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut w = HashMap::new();
    w.insert("weight".to_string(), (vec![0.1; 12], vec![3, 4]));
    w.insert("bias".to_string(), (vec![0.0; 3], vec![3]));
    w
}

#[test]
fn test_build_graph_linear() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    let weight_data = make_weights();
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);

    let imported = build_graph(&program, &weight_map).unwrap();

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["linear"]);

    // Graph should have: 1 input (x) + 2 param placeholders + 1 linear op = 4 nodes.
    assert_eq!(imported.graph.len(), 4);

    // The last (non-placeholder) op should be Linear.
    let output = imported.graph.output_node().unwrap();
    assert!(matches!(output.op(), TraceOp::Linear { .. }));
    assert_eq!(output.output_shape(), &[2, 3]);
}

#[test]
fn test_build_graph_missing_weight() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();

    let err = build_graph(&program, &empty_weights).unwrap_err();
    assert!(
        matches!(err, ImportError::MissingWeight { .. }),
        "expected MissingWeight, got: {err:?}"
    );
}

#[test]
fn test_build_weight_map_from_param_specs() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    let weight_data = make_weights();
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);

    assert!(weight_map.contains_key("p_weight"));
    assert_eq!(weight_map["p_weight"].shape, vec![3, 4]);
    assert_eq!(weight_map["p_weight"].data.len(), 12);

    assert!(weight_map.contains_key("p_bias"));
    assert_eq!(weight_map["p_bias"].shape, vec![3]);
}

fn make_multi_layer_weights() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut w = HashMap::new();
    w.insert("fc1.weight".to_string(), (vec![0.1; 32], vec![8, 4]));
    w.insert("fc1.bias".to_string(), (vec![0.0; 8], vec![8]));
    w.insert("fc2.weight".to_string(), (vec![0.1; 24], vec![3, 8]));
    w.insert("fc2.bias".to_string(), (vec![0.0; 3], vec![3]));
    w
}

fn build_multi_layer_graph() -> ImportedGraph {
    let json = include_str!("../test_data/multi_layer_mlp.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let weight_data = make_multi_layer_weights();
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    build_graph(&program, &weight_map).unwrap()
}

#[test]
fn test_build_graph_multi_layer() {
    let imported = build_multi_layer_graph();

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["softmax"]);

    // 1 input + 4 param placeholders + 4 ops = 9 nodes
    assert_eq!(imported.graph.len(), 9);

    // Final output should be Softmax
    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(output.op(), TraceOp::Softmax { dim: 1 }),
        "expected Softmax dim=1 as output, got: {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[2, 3]);
}

#[test]
fn test_multi_layer_op_sequence() {
    let imported = build_multi_layer_graph();

    let compute_ops: Vec<_> = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .collect();

    assert_eq!(compute_ops.len(), 4, "expected 4 compute ops");
    assert!(matches!(compute_ops[0].op(), TraceOp::Linear { .. }));
    assert!(matches!(compute_ops[1].op(), TraceOp::Relu));
    assert!(matches!(compute_ops[2].op(), TraceOp::Linear { .. }));
    assert!(matches!(compute_ops[3].op(), TraceOp::Softmax { dim: 1 }));

    // Verify shapes propagate correctly through the chain
    assert_eq!(compute_ops[0].output_shape(), &[2, 8]);
    assert_eq!(compute_ops[1].output_shape(), &[2, 8]);
    assert_eq!(compute_ops[2].output_shape(), &[2, 3]);
    assert_eq!(compute_ops[3].output_shape(), &[2, 3]);
}

#[test]
fn test_multi_layer_topology_valid() {
    let imported = build_multi_layer_graph();

    for node in imported.graph.nodes() {
        for &input_id in node.inputs() {
            assert!(
                imported.graph.node(input_id).is_some(),
                "node '{}' references missing input_id {}",
                node.name(),
                input_id
            );
        }
    }
}

fn make_conv_bn_weights() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut w = HashMap::new();
    w.insert(
        "conv.weight".to_string(),
        (vec![0.1; 432], vec![16, 3, 3, 3]),
    );
    w.insert("conv.bias".to_string(), (vec![0.0; 16], vec![16]));
    w.insert("bn.weight".to_string(), (vec![1.0; 16], vec![16]));
    w.insert("bn.bias".to_string(), (vec![0.0; 16], vec![16]));
    w.insert("bn.running_mean".to_string(), (vec![0.0; 16], vec![16]));
    w.insert("bn.running_var".to_string(), (vec![1.0; 16], vec![16]));
    w
}

fn build_conv_bn_graph() -> ImportedGraph {
    let json = include_str!("../test_data/conv_bn_relu.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let weight_data = make_conv_bn_weights();
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    build_graph(&program, &weight_map).unwrap()
}

#[test]
fn test_build_graph_conv_bn_relu_pool() {
    let imported = build_conv_bn_graph();

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);

    let compute_ops: Vec<_> = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .collect();

    assert_eq!(
        compute_ops.len(),
        4,
        "expected Conv2d, BatchNorm, ReLU, AvgPool2d"
    );
    assert!(
        matches!(compute_ops[0].op(), TraceOp::Conv2d { .. }),
        "expected Conv2d, got: {:?}",
        compute_ops[0].op()
    );
    assert!(
        matches!(compute_ops[1].op(), TraceOp::BatchNorm { .. }),
        "expected BatchNorm, got: {:?}",
        compute_ops[1].op()
    );
    assert!(matches!(compute_ops[2].op(), TraceOp::Relu));
    assert!(
        matches!(compute_ops[3].op(), TraceOp::AvgPool2d { .. }),
        "expected AvgPool2d, got: {:?}",
        compute_ops[3].op()
    );

    let output = imported.graph.output_node().unwrap();
    assert_eq!(output.output_shape(), &[1, 16, 16, 16]);
}

#[test]
fn test_build_graph_buffers_registered() {
    let json = include_str!("../test_data/conv_bn_relu.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let weight_data = make_conv_bn_weights();
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);

    // Weight map should contain all 6 entries (4 params + 2 buffers)
    assert_eq!(weight_map.len(), 6);
    assert!(
        weight_map.contains_key("p_bn_mean"),
        "buffer bn.running_mean not resolved"
    );
    assert!(
        weight_map.contains_key("p_bn_var"),
        "buffer bn.running_var not resolved"
    );

    let imported = build_graph(&program, &weight_map).unwrap();
    assert!(!imported.graph.is_empty());
}

#[test]
fn test_topology_error_on_forward_reference() {
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "relu"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.relu.default",
                        "inputs": [
                            {"name": "input", "arg": {"as_tensor": {"name": "nonexistent"}}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "relu"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "relu": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [
                    {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "relu"}}}}
                ]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;

    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let err = build_graph(&program, &empty_weights).unwrap_err();
    assert!(
        matches!(err, ImportError::TopologyError { .. }),
        "expected TopologyError, got: {err:?}"
    );
}

fn make_kokoro_decoder_weights() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut w = HashMap::new();
    w.insert(
        "conv1.weight".to_string(),
        (vec![0.01; 384], vec![16, 8, 3]),
    );
    w.insert("conv1.bias".to_string(), (vec![0.0; 16], vec![16]));
    w.insert(
        "conv2.weight".to_string(),
        (vec![0.01; 768], vec![16, 16, 3]),
    );
    w.insert("conv2.bias".to_string(), (vec![0.0; 16], vec![16]));
    w
}

fn build_kokoro_decoder_graph() -> ImportedGraph {
    let json = include_str!("../test_data/kokoro_decoder_mini.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let weight_data = make_kokoro_decoder_weights();
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    build_graph(&program, &weight_map).unwrap()
}

#[test]
fn test_build_graph_kokoro_decoder_op_sequence() {
    let imported = build_kokoro_decoder_graph();

    let compute_ops: Vec<_> = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .collect();

    assert_eq!(compute_ops.len(), 10, "expected 10 compute ops");
    assert!(matches!(compute_ops[0].op(), TraceOp::Conv1d { .. }));
    assert!(matches!(compute_ops[1].op(), TraceOp::InstanceNorm { .. }));
    assert!(matches!(compute_ops[2].op(), TraceOp::LeakyRelu { .. }));
    assert!(matches!(compute_ops[3].op(), TraceOp::Conv1d { .. }));
    assert!(matches!(compute_ops[4].op(), TraceOp::Add));
    assert!(matches!(compute_ops[5].op(), TraceOp::Narrow { .. }));
    assert!(matches!(compute_ops[6].op(), TraceOp::Exp));
    assert!(matches!(compute_ops[7].op(), TraceOp::Narrow { .. }));
    assert!(matches!(compute_ops[8].op(), TraceOp::Sin));
    assert!(matches!(compute_ops[9].op(), TraceOp::Cat { .. }));
}

#[test]
fn test_build_graph_kokoro_decoder_shapes() {
    let imported = build_kokoro_decoder_graph();

    let compute_ops: Vec<_> = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .collect();

    // Conv1d(8->16, k=3, pad=1): [1,8,16] -> [1,16,16]
    assert_eq!(compute_ops[0].output_shape(), &[1, 16, 16]);
    // InstanceNorm preserves shape: [1,16,16]
    assert_eq!(compute_ops[1].output_shape(), &[1, 16, 16]);
    // LeakyReLU preserves shape: [1,16,16]
    assert_eq!(compute_ops[2].output_shape(), &[1, 16, 16]);
    // Conv1d(16->16, k=3, pad=1): [1,16,16]
    assert_eq!(compute_ops[3].output_shape(), &[1, 16, 16]);
    // Add (residual): [1,16,16]
    assert_eq!(compute_ops[4].output_shape(), &[1, 16, 16]);
    // Slice(dim=1, 0:8): [1,8,16]
    assert_eq!(compute_ops[5].output_shape(), &[1, 8, 16]);
    // Exp: [1,8,16]
    assert_eq!(compute_ops[6].output_shape(), &[1, 8, 16]);
    // Slice(dim=1, 8:16): [1,8,16]
    assert_eq!(compute_ops[7].output_shape(), &[1, 8, 16]);
    // Sin: [1,8,16]
    assert_eq!(compute_ops[8].output_shape(), &[1, 8, 16]);
    // Cat(dim=1): [1,16,16]
    assert_eq!(compute_ops[9].output_shape(), &[1, 16, 16]);
}

#[test]
fn test_build_graph_kokoro_decoder_topology_valid() {
    let imported = build_kokoro_decoder_graph();

    for node in imported.graph.nodes() {
        for &input_id in node.inputs() {
            assert!(
                imported.graph.node(input_id).is_some(),
                "node '{}' references missing input_id {}",
                node.name(),
                input_id
            );
        }
    }

    // Residual connection: Add should reference norm_0 and conv_1
    let add_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Add))
        .unwrap();
    assert_eq!(add_node.inputs().len(), 2, "Add should have 2 inputs");
}

// ---------------------------------------------------------------------------
// New tests: multi-input graph, topology, op coverage, error handling
// ---------------------------------------------------------------------------

fn make_multi_input_weights() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut w = HashMap::new();
    w.insert("fc.weight".to_string(), (vec![0.1; 64], vec![4, 16]));
    w.insert("fc.bias".to_string(), (vec![0.0; 4], vec![4]));
    w
}

fn build_multi_input_graph() -> ImportedGraph {
    let json = include_str!("../test_data/multi_input_cat.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let weight_data = make_multi_input_weights();
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    build_graph(&program, &weight_map).unwrap()
}

#[test]
fn test_build_graph_multi_input() {
    let imported = build_multi_input_graph();
    assert_eq!(imported.num_user_inputs, 2);
    assert_eq!(imported.user_input_names, vec!["a", "b"]);
    assert_eq!(imported.output_names, vec!["output"]);

    let output = imported.graph.output_node().unwrap();
    assert_eq!(output.output_shape(), &[1, 4]);
}

#[test]
fn test_multi_input_cat_fan_in() {
    let imported = build_multi_input_graph();
    let cat_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Cat { .. }))
        .expect("should have a Cat node");
    // Cat merges two inputs (a and b)
    assert_eq!(cat_node.inputs().len(), 2, "Cat should have 2 inputs");
    assert_eq!(cat_node.output_shape(), &[1, 16]);
}

#[test]
fn test_topological_order_ids_increase() {
    let imported = build_multi_layer_graph();
    let ids: Vec<_> = imported.graph.nodes().iter().map(TraceNode::id).collect();
    for window in ids.windows(2) {
        assert!(
            window[0] < window[1],
            "node ids should be strictly increasing: {} >= {}",
            window[0],
            window[1]
        );
    }
}

#[test]
fn test_input_nodes_have_no_dependencies() {
    let imported = build_multi_layer_graph();
    for node in imported.graph.nodes() {
        if matches!(node.op(), TraceOp::Input) {
            assert!(
                node.inputs().is_empty(),
                "Input node '{}' should have no dependencies but has {}",
                node.name(),
                node.inputs().len()
            );
        }
    }
}

#[test]
fn test_weight_map_ignores_extra_weights() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    let mut weight_data = make_weights();
    weight_data.insert("extra_unused".to_string(), (vec![1.0; 10], vec![10]));
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);

    // Extra weight should not appear in the map
    assert!(!weight_map.contains_key("extra_unused"));
    assert_eq!(weight_map.len(), 2);
}

#[test]
fn test_unsupported_op_error() {
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "out"}}],
                "nodes": [{
                    "target": "torch.ops.aten.fake_nonexistent_op.default",
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
    }"#;

    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let err = build_graph(&program, &empty_weights).unwrap_err();
    assert!(
        matches!(err, ImportError::UnsupportedOp { .. }),
        "expected UnsupportedOp, got: {err:?}"
    );
}

#[test]
fn test_unary_chain_exp_sin_cos() {
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "cos_out"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.exp.default",
                        "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
                        "outputs": [{"as_tensor": {"name": "exp_out"}}],
                        "metadata": {}
                    },
                    {
                        "target": "torch.ops.aten.sin.default",
                        "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "exp_out"}}, "kind": 1}],
                        "outputs": [{"as_tensor": {"name": "sin_out"}}],
                        "metadata": {}
                    },
                    {
                        "target": "torch.ops.aten.cos.default",
                        "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "sin_out"}}, "kind": 1}],
                        "outputs": [{"as_tensor": {"name": "cos_out"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "x":       {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 5}], "requires_grad": false, "strides": [{"as_int": 5}, {"as_int": 1}]},
                    "exp_out": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 5}], "requires_grad": false, "strides": [{"as_int": 5}, {"as_int": 1}]},
                    "sin_out": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 5}], "requires_grad": false, "strides": [{"as_int": 5}, {"as_int": 1}]},
                    "cos_out": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 5}], "requires_grad": false, "strides": [{"as_int": 5}, {"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
                "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "cos_out"}}}}]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;

    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    let compute_ops: Vec<_> = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .collect();

    assert_eq!(compute_ops.len(), 3);
    assert!(matches!(compute_ops[0].op(), TraceOp::Exp));
    assert!(matches!(compute_ops[1].op(), TraceOp::Sin));
    assert!(matches!(compute_ops[2].op(), TraceOp::Cos));

    // Each unary op should have exactly 1 input
    for op in &compute_ops {
        assert_eq!(op.inputs().len(), 1, "unary op should have 1 input");
    }
    // Chain connectivity: sin's input is exp's output
    assert_eq!(compute_ops[1].inputs()[0], compute_ops[0].id());
    assert_eq!(compute_ops[2].inputs()[0], compute_ops[1].id());
    // All shapes preserved
    for op in &compute_ops {
        assert_eq!(op.output_shape(), &[2, 5]);
    }
}

#[test]
fn test_diamond_topology() {
    // x -> relu -> add(relu, relu) — diamond: two edges from relu to add
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "add_out"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.relu.default",
                        "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
                        "outputs": [{"as_tensor": {"name": "relu_out"}}],
                        "metadata": {}
                    },
                    {
                        "target": "torch.ops.aten.add.Tensor",
                        "inputs": [
                            {"name": "self", "arg": {"as_tensor": {"name": "relu_out"}}, "kind": 1},
                            {"name": "other", "arg": {"as_tensor": {"name": "relu_out"}}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "add_out"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "x":        {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "relu_out": {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "add_out":  {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
                "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "add_out"}}}}]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;

    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    let add_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Add))
        .expect("should have an Add node");
    // Both inputs of Add reference the same relu output
    assert_eq!(add_node.inputs().len(), 2);
    assert_eq!(add_node.inputs()[0], add_node.inputs()[1]);
    assert_eq!(add_node.output_shape(), &[3]);
}

#[test]
fn test_supported_ops_sorted_and_deduped() {
    let ops = crate::op_map::supported_ops();
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

#[test]
fn test_supported_ops_contains_common_targets() {
    let ops = crate::op_map::supported_ops();
    for expected in &[
        "aten::linear",
        "aten::relu",
        "aten::softmax",
        "aten::cat",
        "aten::conv2d",
        "aten::embedding",
        "aten::layer_norm",
        "aten::matmul",
        "aten::add",
        "aten::mul",
    ] {
        assert!(ops.contains(expected), "supported_ops missing '{expected}'");
    }
}

#[test]
fn test_output_dtype_is_f32() {
    let imported = build_multi_layer_graph();
    let output = imported.graph.output_node().unwrap();
    assert_eq!(output.output_dtype(), DType::F32);
}

#[test]
fn test_all_nodes_have_valid_shapes() {
    let imported = build_multi_layer_graph();
    for node in imported.graph.nodes() {
        let shape = node.output_shape();
        assert!(
            !shape.is_empty() || matches!(node.op(), TraceOp::Input | TraceOp::Constant { .. }),
            "non-placeholder node '{}' has empty shape",
            node.name()
        );
    }
}

#[test]
fn test_residual_skip_connection() {
    // Build kokoro decoder graph and verify the Add node has skip-connection structure
    let imported = build_kokoro_decoder_graph();
    let add_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Add))
        .expect("should have an Add node for residual");

    let input_ids = add_node.inputs();
    assert_eq!(input_ids.len(), 2);
    // The two inputs should be different nodes (not a diamond self-add)
    assert_ne!(
        input_ids[0], input_ids[1],
        "residual Add should have two different source nodes"
    );
}

// ---------------------------------------------------------------------------
// Additional tests: weight map edge cases, embedding, reduction, layernorm,
// multiple outputs, weight data content, positional embedding
// ---------------------------------------------------------------------------

#[test]
fn test_build_weight_map_empty_specs() {
    let weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let specs: Vec<InputSpec> = Vec::new();
    let weight_map = build_weight_map(&specs, &weight_data);
    assert!(weight_map.is_empty());
}

#[test]
fn test_build_weight_map_no_matching_weights() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    // Weight data with different FQN keys than what the specs reference
    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    weight_data.insert("unrelated.weight".to_string(), (vec![0.0; 12], vec![3, 4]));
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    assert!(
        weight_map.is_empty(),
        "no specs should match unrelated FQNs"
    );
}

#[test]
fn test_build_weight_map_partial_match() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    // Only provide the weight, not the bias
    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    weight_data.insert("weight".to_string(), (vec![0.1; 12], vec![3, 4]));
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    assert_eq!(weight_map.len(), 1);
    assert!(weight_map.contains_key("p_weight"));
    assert!(!weight_map.contains_key("p_bias"));
}

#[test]
fn test_build_weight_map_preserves_data_values() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.01).collect();
    weight_data.insert("weight".to_string(), (data.clone(), vec![3, 4]));
    weight_data.insert("bias".to_string(), (vec![0.5, -0.5, 1.0], vec![3]));
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);

    assert_eq!(weight_map["p_weight"].data, data);
    assert_eq!(weight_map["p_bias"].data, vec![0.5, -0.5, 1.0]);
}

#[test]
fn test_build_graph_embedding_lookup() {
    let json = include_str!("../test_data/embedding_lookup.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    weight_data.insert("embed.weight".to_string(), (vec![0.1; 80], vec![10, 8]));
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    let imported = build_graph(&program, &weight_map).unwrap();

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["indices"]);
    assert_eq!(imported.output_names, vec!["embedding"]);

    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(output.op(), TraceOp::Embedding { .. }),
        "expected Embedding, got: {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[1, 4, 8]);
}

#[test]
fn test_build_graph_mean_sum_reduction() {
    let json = include_str!("../test_data/mean_sum_reduce.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    assert_eq!(imported.num_user_inputs, 1);
    let compute_ops: Vec<_> = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .collect();

    assert_eq!(compute_ops.len(), 2, "expected Mean and Sum");
    // Mean with keepdim=true: [1,4,8] -> [1,4,1]
    assert_eq!(compute_ops[0].output_shape(), &[1, 4, 1]);
    // Sum with keepdim=false: [1,4,1] -> [1,1]
    assert_eq!(compute_ops[1].output_shape(), &[1, 1]);
}

fn make_layernorm_weights() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut w = HashMap::new();
    w.insert("ln.weight".to_string(), (vec![1.0; 16], vec![16]));
    w.insert("ln.bias".to_string(), (vec![0.0; 16], vec![16]));
    w
}

#[test]
fn test_build_graph_layernorm_softmax() {
    let json = include_str!("../test_data/layernorm_softmax.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let weight_data = make_layernorm_weights();
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    let imported = build_graph(&program, &weight_map).unwrap();

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.output_names, vec!["softmax"]);

    let compute_ops: Vec<_> = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .collect();

    assert_eq!(compute_ops.len(), 2, "expected LayerNorm and Softmax");
    assert!(
        matches!(compute_ops[0].op(), TraceOp::LayerNorm { .. }),
        "expected LayerNorm, got: {:?}",
        compute_ops[0].op()
    );
    assert!(matches!(compute_ops[1].op(), TraceOp::Softmax { .. }));
    // Shape should be preserved: [1, 4, 16]
    assert_eq!(compute_ops[0].output_shape(), &[1, 4, 16]);
    assert_eq!(compute_ops[1].output_shape(), &[1, 4, 16]);
}

fn make_positional_embed_weights() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut w = HashMap::new();
    w.insert(
        "tok_embed.weight".to_string(),
        (vec![0.01; 1600], vec![100, 16]),
    );
    w.insert(
        "pos_embed.weight".to_string(),
        (vec![0.01; 512], vec![32, 16]),
    );
    w
}

#[test]
fn test_build_graph_positional_embedding() {
    let json = include_str!("../test_data/embedding_positional.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let weight_data = make_positional_embed_weights();
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    let imported = build_graph(&program, &weight_map).unwrap();

    assert_eq!(imported.num_user_inputs, 2);
    assert_eq!(imported.user_input_names, vec!["token_ids", "pos_ids"]);

    // Output should be the Add of two embeddings
    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(output.op(), TraceOp::Add),
        "expected Add as output, got: {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[1, 8, 16]);
    assert_eq!(output.inputs().len(), 2);
}

#[test]
fn test_build_graph_multiple_outputs() {
    // Graph with two output tensors
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [
                    {"as_tensor": {"name": "relu_out"}},
                    {"as_tensor": {"name": "neg_out"}}
                ],
                "nodes": [
                    {
                        "target": "torch.ops.aten.relu.default",
                        "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
                        "outputs": [{"as_tensor": {"name": "relu_out"}}],
                        "metadata": {}
                    },
                    {
                        "target": "torch.ops.aten.neg.default",
                        "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
                        "outputs": [{"as_tensor": {"name": "neg_out"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "x":        {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 3}, {"as_int": 1}]},
                    "relu_out": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 3}, {"as_int": 1}]},
                    "neg_out":  {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 3}, {"as_int": 1}]}
                },
                "is_single_tensor_return": false
            },
            "signature": {
                "input_specs": [
                    {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "relu_out"}}}},
                    {"user_output": {"arg": {"as_tensor": {"name": "neg_out"}}}}
                ]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;

    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    assert_eq!(imported.output_names.len(), 2);
    assert_eq!(imported.output_names, vec!["relu_out", "neg_out"]);
    assert_eq!(imported.num_user_inputs, 1);
    // 1 input + 2 compute ops = 3 nodes
    assert_eq!(imported.graph.len(), 3);
}

#[test]
fn test_build_graph_identity_passthrough() {
    // Graph where output is the same as input (no compute nodes)
    let json = r#"{
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
                "input_specs": [
                    {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "x"}}}}
                ]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;

    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.graph.len(), 1); // just the input node
    assert_eq!(imported.output_names, vec!["x"]);
}

#[test]
fn test_build_graph_bf16_dtype() {
    // Tensor with BF16 dtype (scalar_type 13)
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "relu_out"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.relu.default",
                        "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
                        "outputs": [{"as_tensor": {"name": "relu_out"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "x":        {"dtype": 13, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "relu_out": {"dtype": 13, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [
                    {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "relu_out"}}}}
                ]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;

    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    let output = imported.graph.output_node().unwrap();
    assert_eq!(output.output_dtype(), DType::BF16);
}

#[test]
fn test_build_weight_map_with_buffers() {
    let json = include_str!("../test_data/conv_bn_relu.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let weight_data = make_conv_bn_weights();
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);

    // Verify buffers (running_mean, running_var) are in the weight map
    let buffer_keys: Vec<&String> = weight_map
        .keys()
        .filter(|k| k.contains("mean") || k.contains("var"))
        .collect();
    assert_eq!(buffer_keys.len(), 2, "should have 2 buffer entries");

    // Verify buffer data content
    for key in &buffer_keys {
        let rw = &weight_map[*key];
        assert_eq!(rw.shape, vec![16]);
        assert_eq!(rw.data.len(), 16);
    }
}
