// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for graph building, weight mapping, shape inference,
//! error handling, multi-segment import, and quantization detection.
//! Part of #4186.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;
use nn_core::DType;

use crate::error::ImportError;
use crate::graph_build::{build_graph, build_weight_map, ImportedGraph};
use crate::multi_segment::{MultiSegmentError, MultiSegmentModel};
use crate::op_map::ResolvedWeight;
use crate::parse::{parse_exported_program, InputSpec};
use crate::quantization::{detect_quantization_from_bytes, DetectedDtype};

// ===========================================================================
// Helper: build safetensors bytes for quantization tests
// ===========================================================================

fn build_safetensors(tensors: &[(&str, safetensors::Dtype, &[usize])]) -> Vec<u8> {
    use safetensors::tensor::TensorView;

    let owned_data: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_name, dtype, shape)| {
            let num_elements: usize = shape.iter().product();
            let bytes_per_elem = match dtype {
                safetensors::Dtype::F32 | safetensors::Dtype::I32 | safetensors::Dtype::U32 => 4,
                safetensors::Dtype::F16
                | safetensors::Dtype::BF16
                | safetensors::Dtype::I16
                | safetensors::Dtype::U16 => 2,
                safetensors::Dtype::I8 | safetensors::Dtype::U8 | safetensors::Dtype::BOOL => 1,
                safetensors::Dtype::F64 | safetensors::Dtype::I64 | safetensors::Dtype::U64 => 8,
                _ => 4,
            };
            vec![0u8; num_elements * bytes_per_elem]
        })
        .collect();

    let views: Vec<(&str, TensorView<'_>)> = tensors
        .iter()
        .zip(owned_data.iter())
        .map(|((name, dtype, shape), data)| {
            let view = TensorView::new(*dtype, shape.to_vec(), data).unwrap();
            (*name, view)
        })
        .collect();

    safetensors::serialize(views.iter().map(|(n, v)| (*n, v)), None).unwrap()
}

// ===========================================================================
// Helper: minimal JSON builders
// ===========================================================================

/// Build a minimal torch.export JSON string with inline graph, fully
/// specified tensor_values, and input/output specs.
fn make_json_graph(
    input_specs_json: &str,
    output_specs_json: &str,
    nodes_json: &str,
    tensor_values_json: &str,
) -> String {
    format!(
        r#"{{
        "graph_module": {{
            "graph": {{
                "inputs": [],
                "outputs": [],
                "nodes": [{nodes_json}],
                "tensor_values": {{{tensor_values_json}}},
                "is_single_tensor_return": true
            }},
            "signature": {{
                "input_specs": [{input_specs_json}],
                "output_specs": [{output_specs_json}]
            }},
            "module_call_graph": []
        }},
        "schema_version": {{"major": 8, "minor": 15}},
        "range_constraints": {{}}
    }}"#
    )
}

/// Build a simple single user-input, single-output relu graph JSON.
fn single_relu_json(input_name: &str, output_name: &str, shape: &[usize]) -> String {
    let sizes = shape
        .iter()
        .map(|d| format!(r#"{{"as_int": {d}}}"#))
        .collect::<Vec<_>>()
        .join(", ");
    let strides = {
        let mut s = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s.iter()
            .map(|v| format!(r#"{{"as_int": {v}}}"#))
            .collect::<Vec<_>>()
            .join(", ")
    };

    let tv = format!(
        r#""{input_name}": {{"dtype": 7, "sizes": [{sizes}], "requires_grad": false, "strides": [{strides}]}},
        "{output_name}": {{"dtype": 7, "sizes": [{sizes}], "requires_grad": false, "strides": [{strides}]}}"#
    );

    let node = format!(
        r#"{{
            "target": "torch.ops.aten.relu.default",
            "inputs": [{{"name": "input", "arg": {{"as_tensor": {{"name": "{input_name}"}}}}, "kind": 1}}],
            "outputs": [{{"as_tensor": {{"name": "{output_name}"}}}}],
            "metadata": {{}}
        }}"#
    );

    let input_spec =
        format!(r#"{{"user_input": {{"arg": {{"as_tensor": {{"name": "{input_name}"}}}}}}}}"#);
    let output_spec =
        format!(r#"{{"user_output": {{"arg": {{"as_tensor": {{"name": "{output_name}"}}}}}}}}"#);

    make_json_graph(&input_spec, &output_spec, &node, &tv)
}

// ===========================================================================
// 1. build_graph — graph construction from parsed exported programs
// ===========================================================================

#[test]
fn test_build_graph_resnet_basic_block_structure() {
    // ResNet basic block: conv1 -> bn1 -> relu -> conv2 -> bn2 -> add(x) -> relu
    let json = include_str!("../test_data/resnet_basic_block.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    // conv1: 16 in, 16 out, 3x3 kernel
    weight_data.insert(
        "conv1.weight".to_string(),
        (vec![0.01; 16 * 16 * 3 * 3], vec![16, 16, 3, 3]),
    );
    weight_data.insert("conv1.bias".to_string(), (vec![0.0; 16], vec![16]));
    weight_data.insert("bn1.weight".to_string(), (vec![1.0; 16], vec![16]));
    weight_data.insert("bn1.bias".to_string(), (vec![0.0; 16], vec![16]));
    weight_data.insert("bn1.running_mean".to_string(), (vec![0.0; 16], vec![16]));
    weight_data.insert("bn1.running_var".to_string(), (vec![1.0; 16], vec![16]));
    weight_data.insert(
        "conv2.weight".to_string(),
        (vec![0.01; 16 * 16 * 3 * 3], vec![16, 16, 3, 3]),
    );
    weight_data.insert("conv2.bias".to_string(), (vec![0.0; 16], vec![16]));
    weight_data.insert("bn2.weight".to_string(), (vec![1.0; 16], vec![16]));
    weight_data.insert("bn2.bias".to_string(), (vec![0.0; 16], vec![16]));
    weight_data.insert("bn2.running_mean".to_string(), (vec![0.0; 16], vec![16]));
    weight_data.insert("bn2.running_var".to_string(), (vec![1.0; 16], vec![16]));

    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    let imported = build_graph(&program, &weight_map).unwrap();

    // 1 user input, 12 param/buffer placeholders, 7 compute ops = 20 nodes
    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["relu_out"]);

    let compute_ops: Vec<_> = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .collect();

    assert_eq!(compute_ops.len(), 7, "conv-bn-relu-conv-bn-add-relu");

    // Verify skip connection: the Add node should reference both bn2 output and original input x
    let add_node = compute_ops
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Add))
        .expect("should have an Add node for skip connection");
    assert_eq!(add_node.inputs().len(), 2);
    // The two inputs should be different (bn2 and x)
    assert_ne!(add_node.inputs()[0], add_node.inputs()[1]);

    // Output shape is preserved through the residual block
    let output = imported.graph.output_node().unwrap();
    assert_eq!(output.output_shape(), &[1, 16, 8, 8]);
}

#[test]
fn test_build_graph_empty_nodes_passthrough() {
    // A graph with a user input and no computation nodes; output == input.
    let json = make_json_graph(
        r#"{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}"#,
        r#"{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}"#,
        "",
        r#""x": {"dtype": 7, "sizes": [{"as_int": 5}], "requires_grad": false, "strides": [{"as_int": 1}]}"#,
    );
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    assert_eq!(imported.graph.len(), 1);
    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.output_names, vec!["x"]);
}

#[test]
fn test_build_graph_preserves_node_names() {
    // Verify that build_graph assigns the correct tensor names to graph nodes.
    let json = single_relu_json("nn_input", "nn_relu_out", &[2, 4]);
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    let names: Vec<&str> = imported.graph.nodes().iter().map(nn_core::dyn_tensor::trace::TraceNode::name).collect();
    assert!(names.contains(&"nn_input"), "should contain input name");
    assert!(names.contains(&"nn_relu_out"), "should contain output name");
}

#[test]
fn test_build_graph_getitem_index0_aliases() {
    // getitem[0] should alias to the source node, not create a new Constant.
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "getitem_0"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.relu.default",
                        "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
                        "outputs": [{"as_tensor": {"name": "relu_out"}}],
                        "metadata": {}
                    },
                    {
                        "target": "operator.getitem",
                        "inputs": [
                            {"name": "input", "arg": {"as_tensor": {"name": "relu_out"}}, "kind": 1},
                            {"name": "index", "arg": {"as_int": 0}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "getitem_0"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "x":          {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "relu_out":   {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "getitem_0":  {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
                "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "getitem_0"}}}}]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;

    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    // getitem[0] should alias to relu_out, so no new node for getitem.
    // Graph: input(x) + relu = 2 nodes.
    assert_eq!(imported.graph.len(), 2);
}

#[test]
fn test_build_graph_getitem_nonzero_creates_constant() {
    // getitem[1] should create a Constant placeholder node.
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "getitem_1"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.relu.default",
                        "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
                        "outputs": [{"as_tensor": {"name": "relu_out"}}],
                        "metadata": {}
                    },
                    {
                        "target": "operator.getitem",
                        "inputs": [
                            {"name": "input", "arg": {"as_tensor": {"name": "relu_out"}}, "kind": 1},
                            {"name": "index", "arg": {"as_int": 1}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "getitem_1"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "x":          {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "relu_out":   {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "getitem_1":  {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
                "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "getitem_1"}}}}]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;

    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    // input(x) + relu + getitem_1(Constant) = 3 nodes
    assert_eq!(imported.graph.len(), 3);
    let last_node = imported.graph.output_node().unwrap();
    assert!(
        matches!(last_node.op(), TraceOp::Constant { .. }),
        "getitem[1] should be a Constant placeholder, got: {:?}",
        last_node.op()
    );
}

// ===========================================================================
// 2. build_weight_map — weight name resolution from input specs
// ===========================================================================

#[test]
fn test_build_weight_map_resolves_buffers_and_params() {
    // Create a graph with both parameters and buffers, verify both types resolve.
    let json = include_str!("../test_data/resnet_basic_block.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    weight_data.insert(
        "conv1.weight".to_string(),
        (vec![0.01; 16 * 16 * 9], vec![16, 16, 3, 3]),
    );
    weight_data.insert("conv1.bias".to_string(), (vec![0.0; 16], vec![16]));
    weight_data.insert("bn1.weight".to_string(), (vec![1.0; 16], vec![16]));
    weight_data.insert("bn1.bias".to_string(), (vec![0.0; 16], vec![16]));
    weight_data.insert("bn1.running_mean".to_string(), (vec![0.0; 16], vec![16]));
    weight_data.insert("bn1.running_var".to_string(), (vec![1.0; 16], vec![16]));
    // Omit conv2/bn2 to test partial resolution.

    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);

    // 4 params + 2 buffers from conv1/bn1 = 6
    assert_eq!(weight_map.len(), 6);

    // Verify param-type entries
    assert!(weight_map.contains_key("p_conv1_weight"));
    assert!(weight_map.contains_key("p_conv1_bias"));
    assert!(weight_map.contains_key("p_bn1_weight"));
    assert!(weight_map.contains_key("p_bn1_bias"));

    // Verify buffer-type entries
    assert!(weight_map.contains_key("p_bn1_mean"));
    assert!(weight_map.contains_key("p_bn1_var"));
}

#[test]
fn test_build_weight_map_shape_preserved() {
    let json = include_str!("../test_data/e2e_mlp.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    weight_data.insert("fc1.weight".to_string(), (vec![0.1; 32], vec![8, 4]));
    weight_data.insert("fc1.bias".to_string(), (vec![0.0; 8], vec![8]));
    weight_data.insert("fc2.weight".to_string(), (vec![0.1; 24], vec![3, 8]));
    weight_data.insert("fc2.bias".to_string(), (vec![0.0; 3], vec![3]));

    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);

    assert_eq!(weight_map["p_fc1_weight"].shape, vec![8, 4]);
    assert_eq!(weight_map["p_fc1_weight"].data.len(), 32);
    assert_eq!(weight_map["p_fc2_weight"].shape, vec![3, 8]);
    assert_eq!(weight_map["p_fc2_bias"].shape, vec![3]);
}

#[test]
fn test_build_weight_map_user_inputs_not_included() {
    // build_weight_map should only include Parameter and Buffer entries,
    // never user inputs.
    let json = include_str!("../test_data/e2e_mlp.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    weight_data.insert("fc1.weight".to_string(), (vec![0.1; 32], vec![8, 4]));
    weight_data.insert("fc1.bias".to_string(), (vec![0.0; 8], vec![8]));
    weight_data.insert("fc2.weight".to_string(), (vec![0.1; 24], vec![3, 8]));
    weight_data.insert("fc2.bias".to_string(), (vec![0.0; 3], vec![3]));
    // Add an entry with the user input name — should NOT appear in weight_map.
    weight_data.insert("x".to_string(), (vec![0.0; 4], vec![1, 4]));

    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);

    assert!(
        !weight_map.contains_key("x"),
        "user input should not be in weight map"
    );
    assert_eq!(weight_map.len(), 4, "only 4 parameter entries expected");
}

// ===========================================================================
// 3. ImportedGraph structure — node count, edge connectivity
// ===========================================================================

#[test]
fn test_imported_graph_edge_connectivity_linear() {
    // Linear op should have exactly 1 user input dependency in its input_ids.
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [
                    {"as_tensor": {"name": "p_weight"}},
                    {"as_tensor": {"name": "x"}}
                ],
                "outputs": [{"as_tensor": {"name": "linear"}}],
                "nodes": [{
                    "target": "torch.ops.aten.linear.default",
                    "inputs": [
                        {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
                        {"name": "weight", "arg": {"as_tensor": {"name": "p_weight"}}, "kind": 1},
                        {"name": "bias", "arg": {"as_none": true}, "kind": 1}
                    ],
                    "outputs": [{"as_tensor": {"name": "linear"}}],
                    "metadata": {}
                }],
                "tensor_values": {
                    "x":        {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]},
                    "p_weight": {"dtype": 7, "sizes": [{"as_int": 3}, {"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 4}, {"as_int": 1}]},
                    "linear":   {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 3}, {"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_weight"}, "parameter_name": "weight"}},
                    {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "linear"}}}}
                ]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;

    let program = parse_exported_program(json.as_bytes()).unwrap();
    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    weight_data.insert("weight".to_string(), (vec![0.1; 12], vec![3, 4]));
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    let imported = build_graph(&program, &weight_map).unwrap();

    // 1 user input + 1 param placeholder + 1 linear op = 3 nodes
    assert_eq!(imported.graph.len(), 3);

    let linear_node = imported.graph.output_node().unwrap();
    assert!(matches!(linear_node.op(), TraceOp::Linear { .. }));
    // Linear node inputs should reference valid node IDs
    for &input_id in linear_node.inputs() {
        assert!(
            imported.graph.node(input_id).is_some(),
            "linear input_id {input_id} not found in graph"
        );
    }
}

#[test]
fn test_imported_graph_multi_output_count() {
    // Graph with two output tensors should mark both as outputs.
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
                    "x":        {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "relu_out": {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "neg_out":  {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 1}]}
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

    // Both output nodes should be marked as outputs in the graph.
    let output_nodes = imported.graph.output_nodes();
    assert_eq!(output_nodes.len(), 2);
}

#[test]
fn test_imported_graph_all_ids_are_unique() {
    let json = include_str!("../test_data/resnet_basic_block.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for (name, count, shape) in [
        ("conv1.weight", 16 * 16 * 9, vec![16, 16, 3, 3]),
        ("conv1.bias", 16, vec![16]),
        ("bn1.weight", 16, vec![16]),
        ("bn1.bias", 16, vec![16]),
        ("bn1.running_mean", 16, vec![16]),
        ("bn1.running_var", 16, vec![16]),
        ("conv2.weight", 16 * 16 * 9, vec![16, 16, 3, 3]),
        ("conv2.bias", 16, vec![16]),
        ("bn2.weight", 16, vec![16]),
        ("bn2.bias", 16, vec![16]),
        ("bn2.running_mean", 16, vec![16]),
        ("bn2.running_var", 16, vec![16]),
    ] {
        weight_data.insert(name.to_string(), (vec![0.0; count], shape));
    }

    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    let imported = build_graph(&program, &weight_map).unwrap();

    let ids: Vec<u64> = imported.graph.nodes().iter().map(nn_core::dyn_tensor::trace::TraceNode::id).collect();
    let unique: std::collections::HashSet<u64> = ids.iter().copied().collect();
    assert_eq!(ids.len(), unique.len(), "all node IDs should be unique");
}

// ===========================================================================
// 4. Weight resolution — safetensors key lookup and fallback behavior
// ===========================================================================

#[test]
fn test_weight_resolution_missing_weight_error() {
    // If a required parameter is missing from the weight map, build_graph
    // should return MissingWeight.
    let json = include_str!("../test_data/e2e_mlp.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();

    // Provide only fc1 weights, not fc2.
    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    weight_data.insert("fc1.weight".to_string(), (vec![0.1; 32], vec![8, 4]));
    weight_data.insert("fc1.bias".to_string(), (vec![0.0; 8], vec![8]));

    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    let err = build_graph(&program, &weight_map).unwrap_err();

    assert!(
        matches!(err, ImportError::MissingWeight { .. }),
        "expected MissingWeight, got: {err:?}"
    );
}

#[test]
fn test_weight_map_empty_data_allowed() {
    // Weight data with 0-length data vector should still be inserted into the map.
    let specs_json =
        r#"{"parameter": {"arg": {"name": "p_empty"}, "parameter_name": "empty_weight"}}"#;
    let specs: Vec<InputSpec> = vec![serde_json::from_str(specs_json).unwrap()];

    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    weight_data.insert("empty_weight".to_string(), (vec![], vec![0]));

    let weight_map = build_weight_map(&specs, &weight_data);
    assert_eq!(weight_map.len(), 1);
    assert!(weight_map.contains_key("p_empty"));
    assert!(weight_map["p_empty"].data.is_empty());
}

// ===========================================================================
// 5. Shape inference — output shape computation through the import graph
// ===========================================================================

#[test]
fn test_shape_inference_unary_chain_preserves_shape() {
    // Unary ops (relu, exp, sin, cos, neg) should preserve input shape.
    let shape = &[2, 3, 4];
    let json = single_relu_json("x", "relu_out", shape);
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    let output = imported.graph.output_node().unwrap();
    assert_eq!(output.output_shape(), shape);
}

#[test]
fn test_shape_inference_1d_tensor() {
    let json = single_relu_json("x", "relu_out", &[10]);
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    let output = imported.graph.output_node().unwrap();
    assert_eq!(output.output_shape(), &[10]);
}

#[test]
fn test_shape_inference_high_rank_tensor() {
    let json = single_relu_json("x", "relu_out", &[1, 2, 3, 4, 5]);
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    let output = imported.graph.output_node().unwrap();
    assert_eq!(output.output_shape(), &[1, 2, 3, 4, 5]);
}

#[test]
fn test_shape_inference_conv_bn_relu_chain() {
    // conv_bn_relu.json: x=[1,3,32,32] -> conv(3->16,k=3,p=1) -> [1,16,32,32] -> bn -> relu -> avgpool(2) -> [1,16,16,16]
    let json = include_str!("../test_data/conv_bn_relu.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    weight_data.insert(
        "conv.weight".to_string(),
        (vec![0.1; 432], vec![16, 3, 3, 3]),
    );
    weight_data.insert("conv.bias".to_string(), (vec![0.0; 16], vec![16]));
    weight_data.insert("bn.weight".to_string(), (vec![1.0; 16], vec![16]));
    weight_data.insert("bn.bias".to_string(), (vec![0.0; 16], vec![16]));
    weight_data.insert("bn.running_mean".to_string(), (vec![0.0; 16], vec![16]));
    weight_data.insert("bn.running_var".to_string(), (vec![1.0; 16], vec![16]));

    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    let imported = build_graph(&program, &weight_map).unwrap();

    let compute_ops: Vec<_> = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .collect();

    // Conv: [1,3,32,32] -> [1,16,32,32]
    assert_eq!(compute_ops[0].output_shape(), &[1, 16, 32, 32]);
    // BN preserves shape
    assert_eq!(compute_ops[1].output_shape(), &[1, 16, 32, 32]);
    // ReLU preserves shape
    assert_eq!(compute_ops[2].output_shape(), &[1, 16, 32, 32]);
    // AvgPool2d(kernel=2): halves spatial dims
    assert_eq!(compute_ops[3].output_shape(), &[1, 16, 16, 16]);
}

#[test]
fn test_shape_inference_dtype_propagation() {
    // Verify that BF16 dtype propagates through the graph.
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "relu_out"}}],
                "nodes": [{
                    "target": "torch.ops.aten.relu.default",
                    "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
                    "outputs": [{"as_tensor": {"name": "relu_out"}}],
                    "metadata": {}
                }],
                "tensor_values": {
                    "x":        {"dtype": 13, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "relu_out": {"dtype": 13, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
                "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "relu_out"}}}}]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;

    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    // Input node should be BF16
    let input_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Input))
        .unwrap();
    assert_eq!(input_node.output_dtype(), DType::BF16);

    // Output should be BF16
    let output = imported.graph.output_node().unwrap();
    assert_eq!(output.output_dtype(), DType::BF16);
}

// ===========================================================================
// 6. Error handling — missing weights, invalid graph, unsupported ops
// ===========================================================================

#[test]
fn test_error_unsupported_schema_version() {
    let json = r#"{
        "graph_module": {
            "graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}},
            "signature": {"input_specs": [], "output_specs": []},
            "module_call_graph": []
        },
        "schema_version": {"major": 9, "minor": 0},
        "range_constraints": {}
    }"#;

    let err = parse_exported_program(json.as_bytes()).unwrap_err();
    assert!(
        matches!(err, ImportError::UnsupportedSchema { major: 9, .. }),
        "expected UnsupportedSchema major=9, got: {err:?}"
    );
}

#[test]
fn test_error_invalid_json() {
    let err = parse_exported_program(b"not valid json").unwrap_err();
    assert!(
        matches!(err, ImportError::JsonParse(_)),
        "expected JsonParse, got: {err:?}"
    );
}

#[test]
fn test_error_topology_forward_reference() {
    // Node references a tensor that was never defined upstream.
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "add_out"}}],
                "nodes": [{
                    "target": "torch.ops.aten.add.Tensor",
                    "inputs": [
                        {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
                        {"name": "other", "arg": {"as_tensor": {"name": "missing_tensor"}}, "kind": 1}
                    ],
                    "outputs": [{"as_tensor": {"name": "add_out"}}],
                    "metadata": {}
                }],
                "tensor_values": {
                    "x":       {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "add_out": {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 1}]}
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
    let err = build_graph(&program, &empty_weights).unwrap_err();
    assert!(
        matches!(err, ImportError::TopologyError { .. }),
        "expected TopologyError, got: {err:?}"
    );
}

#[test]
fn test_error_unsupported_op() {
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "out"}}],
                "nodes": [{
                    "target": "torch.ops.aten.nonexistent_op_xyz123.default",
                    "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
                    "outputs": [{"as_tensor": {"name": "out"}}],
                    "metadata": {}
                }],
                "tensor_values": {
                    "x":   {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]},
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
fn test_error_unknown_tensor_in_tensor_values() {
    // A node references a tensor that exists in the graph topology but has
    // no entry in tensor_values (so shape/dtype cannot be determined).
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "relu_out"}}],
                "nodes": [{
                    "target": "torch.ops.aten.relu.default",
                    "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
                    "outputs": [{"as_tensor": {"name": "relu_out"}}],
                    "metadata": {}
                }],
                "tensor_values": {
                    "relu_out": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
                "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "relu_out"}}}}]
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
        matches!(err, ImportError::UnknownTensor { .. }),
        "expected UnknownTensor, got: {err:?}"
    );
}

// ===========================================================================
// 7. Multi-segment models — multi_segment import for models with multiple subgraphs
// ===========================================================================

#[test]
fn test_multi_segment_empty_input_error() {
    let graphs: Vec<(String, serde_json::Value)> = vec![];
    let err =
        crate::multi_segment::convert_multi_segment(&graphs, std::path::Path::new("/nonexistent"))
            .unwrap_err();
    assert!(
        matches!(err, MultiSegmentError::EmptyInput),
        "expected EmptyInput, got: {err:?}"
    );
}

#[test]
fn test_multi_segment_duplicate_name_error() {
    let val = serde_json::json!({});
    let graphs = vec![
        ("encoder".to_string(), val.clone()),
        ("encoder".to_string(), val),
    ];
    let err =
        crate::multi_segment::convert_multi_segment(&graphs, std::path::Path::new("/nonexistent"))
            .unwrap_err();
    assert!(
        matches!(err, MultiSegmentError::DuplicateSegment { ref name } if name == "encoder"),
        "expected DuplicateSegment for 'encoder', got: {err:?}"
    );
}

#[test]
fn test_multi_segment_model_api() {
    // Test MultiSegmentModel public API with manually constructed data.
    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![]);
    let ig = ImportedGraph::new(graph, 0, vec![], vec![]);
    let model = MultiSegmentModel::new(
        vec![("seg_a".to_string(), ig)],
        vec!["seg_a".to_string()],
        vec![],
    );

    assert_eq!(model.num_segments(), 1);
    assert!(model.get_segment("seg_a").is_some());
    assert!(model.get_segment("seg_b").is_none());
    assert!(model.graph("seg_a").is_some());
    assert_eq!(model.segment_order, vec!["seg_a"]);
    assert!(model.shared_weights.is_empty());
}

#[test]
fn test_multi_segment_model_two_segments() {
    // Build a MultiSegmentModel with two segments and verify isolation.
    let graph_a = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![
        nn_core::dyn_tensor::trace::TraceNode::new(
            0,
            "inp_a".to_string(),
            TraceOp::Input,
            vec![],
            vec![1, 4],
            DType::F32,
        ),
    ]);
    let ig_a = ImportedGraph::new(
        graph_a,
        1,
        vec!["inp_a".to_string()],
        vec!["inp_a".to_string()],
    );

    let graph_b = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![
        nn_core::dyn_tensor::trace::TraceNode::new(
            0,
            "inp_b".to_string(),
            TraceOp::Input,
            vec![],
            vec![2, 8],
            DType::F32,
        ),
    ]);
    let ig_b = ImportedGraph::new(
        graph_b,
        1,
        vec!["inp_b".to_string()],
        vec!["inp_b".to_string()],
    );

    let model = MultiSegmentModel::new(
        vec![("encoder".to_string(), ig_a), ("decoder".to_string(), ig_b)],
        vec!["encoder".to_string(), "decoder".to_string()],
        vec!["shared_weight_0".to_string()],
    );

    assert_eq!(model.num_segments(), 2);
    assert_eq!(model.segment_order, vec!["encoder", "decoder"]);

    let enc = model.get_segment("encoder").unwrap();
    assert_eq!(enc.num_user_inputs, 1);
    assert_eq!(enc.user_input_names, vec!["inp_a"]);

    let dec = model.get_segment("decoder").unwrap();
    assert_eq!(dec.num_user_inputs, 1);
    assert_eq!(dec.user_input_names, vec!["inp_b"]);

    assert_eq!(model.shared_weights, vec!["shared_weight_0"]);
}

// ===========================================================================
// 8. Quantization detection — detect_quantization identifies weight dtypes
// ===========================================================================

#[test]
fn test_detect_quantization_pure_f32() {
    let bytes = build_safetensors(&[
        ("layer.weight", safetensors::Dtype::F32, &[64, 32]),
        ("layer.bias", safetensors::Dtype::F32, &[64]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 2);
    assert_eq!(report.dtype_breakdown.len(), 1);
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::F32);
    assert_eq!(report.dtype_breakdown[0].tensor_count, 2);
    assert!(!report.is_mixed_precision());
    assert!((report.dtype_fraction(DetectedDtype::F32) - 1.0).abs() < 1e-9);
}

#[test]
fn test_detect_quantization_mixed_f32_bf16() {
    let bytes = build_safetensors(&[
        ("encoder.weight", safetensors::Dtype::F32, &[128, 64]),
        ("decoder.weight", safetensors::Dtype::BF16, &[64, 128]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 2);
    assert!(report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown.len(), 2);

    // F32 tensor: 128*64*4 = 32768 bytes. BF16 tensor: 64*128*2 = 16384 bytes.
    let f32_frac = report.dtype_fraction(DetectedDtype::F32);
    assert!(
        f32_frac > 0.6 && f32_frac < 0.7,
        "F32 fraction should be ~2/3, got {f32_frac}"
    );
}

#[test]
fn test_detect_quantization_f16_model() {
    let bytes = build_safetensors(&[
        ("attn.q", safetensors::Dtype::F16, &[256, 256]),
        ("attn.k", safetensors::Dtype::F16, &[256, 256]),
        ("attn.v", safetensors::Dtype::F16, &[256, 256]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 3);
    assert!(!report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::F16);
    // No F32->F16 recommendation since already F16.
    let f16_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| r.target_dtype == DetectedDtype::F16)
        .collect();
    assert!(
        f16_recs.is_empty(),
        "should not recommend F16 for already-F16 model"
    );
}

#[test]
fn test_detect_quantization_recommendations_f32_large() {
    // F32 tensors >= 1024 elements should get F16 and I8 recommendations.
    let bytes = build_safetensors(&[
        ("big_weight", safetensors::Dtype::F32, &[64, 64]), // 4096 elements
        ("small_bias", safetensors::Dtype::F32, &[16]),     // 16 elements (below threshold)
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    // Should have F16 and I8 recommendations for big_weight only.
    let f16_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == DetectedDtype::F16);
    assert!(
        f16_rec.is_some(),
        "should recommend F16 for large F32 tensors"
    );
    let f16_rec = f16_rec.unwrap();
    assert_eq!(f16_rec.tensor_names.len(), 1);
    assert_eq!(f16_rec.tensor_names[0], "big_weight");
    // Savings should be 50% of big_weight bytes
    assert_eq!(f16_rec.savings_bytes, f16_rec.current_bytes / 2);

    let i8_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == DetectedDtype::I8);
    assert!(
        i8_rec.is_some(),
        "should recommend I8 for large F32 tensors"
    );
    let i8_rec = i8_rec.unwrap();
    assert_eq!(i8_rec.savings_bytes, i8_rec.current_bytes * 3 / 4);
}

#[test]
fn test_detect_quantization_i8_weights() {
    // A model already quantized to I8 should not get further recommendations.
    let bytes = build_safetensors(&[
        ("layer.weight", safetensors::Dtype::I8, &[256, 256]),
        ("layer.scale", safetensors::Dtype::F32, &[256]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert!(report.is_mixed_precision());
    // I8 should be detected.
    let i8_breakdown = report
        .dtype_breakdown
        .iter()
        .find(|b| b.dtype == DetectedDtype::I8);
    assert!(i8_breakdown.is_some());
    assert_eq!(i8_breakdown.unwrap().tensor_count, 1);
}

#[test]
fn test_detect_quantization_empty_model() {
    let bytes = build_safetensors(&[]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 0);
    assert_eq!(report.total_parameters, 0);
    assert_eq!(report.total_bytes, 0);
    assert!(!report.is_mixed_precision());
    assert!(report.recommendations.is_empty());
    assert_eq!(report.total_savings_bytes(), 0);
}

#[test]
fn test_detect_quantization_summary_formatting() {
    let bytes = build_safetensors(&[("w1", safetensors::Dtype::F32, &[100, 100])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    let summary = report.summary();

    assert!(summary.contains("Quantization Report:"));
    assert!(summary.contains("Dtype Breakdown:"));
    assert!(summary.contains("F32"));
}

// ===========================================================================
// Additional edge cases
// ===========================================================================

#[test]
fn test_build_graph_f16_input_dtype() {
    // Verify that F16 dtype (scalar_type 6) is correctly parsed.
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "relu_out"}}],
                "nodes": [{
                    "target": "torch.ops.aten.relu.default",
                    "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
                    "outputs": [{"as_tensor": {"name": "relu_out"}}],
                    "metadata": {}
                }],
                "tensor_values": {
                    "x":        {"dtype": 6, "sizes": [{"as_int": 8}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "relu_out": {"dtype": 6, "sizes": [{"as_int": 8}], "requires_grad": false, "strides": [{"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}],
                "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "relu_out"}}}}]
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
    assert_eq!(output.output_dtype(), DType::F16);
}

#[test]
fn test_build_graph_i64_dtype() {
    // Verify that I64 dtype (scalar_type 5) is correctly parsed.
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "idx"}}],
                "outputs": [{"as_tensor": {"name": "idx"}}],
                "nodes": [],
                "tensor_values": {
                    "idx": {"dtype": 5, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "idx"}}}}],
                "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "idx"}}}}]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;

    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    let input_node = imported.graph.nodes().first().unwrap();
    assert_eq!(input_node.output_dtype(), DType::I64);
}

#[test]
fn test_build_graph_multiple_user_inputs_ordering() {
    // Verify that multiple user inputs maintain their declaration order.
    let json = include_str!("../test_data/multi_input_cat.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    weight_data.insert("fc.weight".to_string(), (vec![0.1; 64], vec![4, 16]));
    weight_data.insert("fc.bias".to_string(), (vec![0.0; 4], vec![4]));

    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    let imported = build_graph(&program, &weight_map).unwrap();

    assert_eq!(imported.num_user_inputs, 2);
    // The ordering should match the input_specs declaration order.
    assert_eq!(imported.user_input_names[0], "a");
    assert_eq!(imported.user_input_names[1], "b");
}

#[test]
fn test_weight_map_large_weight_data_preserved() {
    // Verify that large weight data (many elements) is preserved exactly.
    let json = include_str!("../test_data/e2e_mlp.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let big_data: Vec<f32> = (0..32).map(|i| i as f32 * 0.001).collect();
    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    weight_data.insert("fc1.weight".to_string(), (big_data.clone(), vec![8, 4]));
    weight_data.insert("fc1.bias".to_string(), (vec![0.0; 8], vec![8]));
    weight_data.insert("fc2.weight".to_string(), (vec![0.1; 24], vec![3, 8]));
    weight_data.insert("fc2.bias".to_string(), (vec![0.0; 3], vec![3]));

    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);

    // Verify data is exactly preserved (no rounding, no conversion).
    assert_eq!(weight_map["p_fc1_weight"].data, big_data);
}
