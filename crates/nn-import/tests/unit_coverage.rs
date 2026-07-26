// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional integration tests for nn-import public API.
//! Focuses on parse coverage, graph building, weight mapping, error handling,
//! Kokoro weights, and report/proof types.
//! Issue: #3807

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;
use nn_core::DType;
use nn_import::{
    build_graph, build_weight_map, parse_exported_program, supported_ops, ImportError,
    ResolvedWeight,
};

// ---------------------------------------------------------------------------
// 1. supported_ops() coverage
// ---------------------------------------------------------------------------

#[test]
fn test_supported_ops_non_empty_and_sorted() {
    let ops = supported_ops();
    assert!(
        ops.len() > 50,
        "expected 50+ supported ops, got {}",
        ops.len()
    );
    // Verify sorted
    for window in ops.windows(2) {
        assert!(
            window[0] <= window[1],
            "supported_ops not sorted: {:?} > {:?}",
            window[0],
            window[1]
        );
    }
    // Verify no duplicates
    let mut deduped = ops.clone();
    deduped.dedup();
    assert_eq!(ops.len(), deduped.len(), "supported_ops has duplicates");
}

#[test]
fn test_supported_ops_contains_core_ops() {
    let ops = supported_ops();
    for expected in &[
        "aten::relu",
        "aten::linear",
        "aten::softmax",
        "aten::matmul",
        "aten::conv1d",
        "aten::embedding",
        "aten::lstm",
        "aten::layer_norm",
        "aten::cat",
        "aten::reshape",
        "aten::transpose",
    ] {
        assert!(ops.contains(expected), "supported_ops missing {expected}");
    }
}

#[test]
fn test_supported_ops_contains_kokoro_ops() {
    let ops = supported_ops();
    for expected in &[
        "aten::reflection_pad1d",
        "aten::upsample_nearest1d",
        "aten::conv_transpose1d",
        "aten::atan2",
        "aten::contiguous",
        "aten::clone",
        "aten::arange",
    ] {
        assert!(
            ops.contains(expected),
            "supported_ops missing Kokoro op: {expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Parse: schema version acceptance and rejection
// ---------------------------------------------------------------------------

#[test]
fn test_parse_schema_version_8_accepted() {
    let json = r#"{
        "graph_module": {
            "graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}},
            "signature": {"input_specs": [], "output_specs": []},
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 99},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert_eq!(program.schema_version.minor, 99);
}

#[test]
fn test_parse_schema_version_9_rejected() {
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
        "expected UnsupportedSchema for major=9, got: {err:?}"
    );
}

#[test]
fn test_parse_schema_version_0_rejected() {
    let json = r#"{
        "graph_module": {
            "graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}},
            "signature": {"input_specs": [], "output_specs": []},
            "module_call_graph": []
        },
        "schema_version": {"major": 0, "minor": 1},
        "range_constraints": {}
    }"#;
    let err = parse_exported_program(json.as_bytes()).unwrap_err();
    assert!(
        matches!(err, ImportError::UnsupportedSchema { major: 0, .. }),
        "expected UnsupportedSchema for major=0, got: {err:?}"
    );
}

#[test]
fn test_parse_invalid_json_errors() {
    let err = parse_exported_program(b"not json").unwrap_err();
    assert!(
        matches!(err, ImportError::JsonParse(_)),
        "expected JsonParse error, got: {err:?}"
    );
}

#[test]
fn test_parse_empty_json_object_errors() {
    let err = parse_exported_program(b"{}").unwrap_err();
    assert!(
        matches!(err, ImportError::JsonParse(_)),
        "expected JsonParse for missing fields, got: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// 3. Parse: range_constraints, opset_version, torch_version
// ---------------------------------------------------------------------------

#[test]
fn test_parse_range_constraints_and_opset() {
    let json = r#"{
        "graph_module": {
            "graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}},
            "signature": {"input_specs": [], "output_specs": []},
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "opset_version": {"aten": 10, "custom": 1},
        "range_constraints": {"s0": {"min_val": 1, "max_val": 1024}},
        "torch_version": "2.10.0"
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert_eq!(program.opset_version.get("aten"), Some(&10));
    assert_eq!(program.opset_version.get("custom"), Some(&1));
    assert_eq!(program.range_constraints.len(), 1);
    let rc = program.range_constraints.get("s0").unwrap();
    assert_eq!(rc.min_val, 1);
    assert_eq!(rc.max_val, 1024);
    assert_eq!(program.torch_version.as_deref(), Some("2.10.0"));
}

#[test]
fn test_parse_optional_fields_default() {
    let json = r#"{
        "graph_module": {
            "graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}},
            "signature": {"input_specs": [], "output_specs": []},
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert!(program.opset_version.is_empty());
    assert!(program.torch_version.is_none());
}

// ---------------------------------------------------------------------------
// 4. Error formatting
// ---------------------------------------------------------------------------

#[test]
fn test_import_error_unsupported_op_display() {
    let err = ImportError::UnsupportedOp {
        target: "torch.ops.aten.custom.default".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("unsupported aten op"));
    assert!(msg.contains("custom"));
}

#[test]
fn test_import_error_missing_argument_display() {
    let err = ImportError::MissingArgument {
        op_target: "aten.linear".to_string(),
        arg_name: "weight".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("weight"));
    assert!(msg.contains("aten.linear"));
}

#[test]
fn test_import_error_weight_shape_mismatch_display() {
    let err = ImportError::WeightShapeMismatch {
        name: "fc1.weight".to_string(),
        shape: vec![3, 4],
        expected: 12,
        actual: 10,
    };
    let msg = err.to_string();
    assert!(msg.contains("fc1.weight"));
    assert!(msg.contains("12"));
    assert!(msg.contains("10"));
}

#[test]
fn test_import_error_topology_display() {
    let err = ImportError::TopologyError {
        node_name: "relu_0".to_string(),
        ref_name: "nonexistent".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("relu_0"));
    assert!(msg.contains("nonexistent"));
}

#[test]
fn test_import_error_negative_dimension_display() {
    let err = ImportError::NegativeDimension {
        op_target: "aten.softmax".to_string(),
        arg_name: "dim".to_string(),
        value: -3,
    };
    let msg = err.to_string();
    assert!(msg.contains("-3"));
    assert!(msg.contains("dim"));
}

#[test]
fn test_import_error_multi_axis_display() {
    let err = ImportError::MultiAxisNotSupported {
        op_target: "aten.sum".to_string(),
        op_kind: "reduction",
        dims: vec![1, 2],
    };
    let msg = err.to_string();
    assert!(msg.contains("multi-axis"));
    assert!(msg.contains("aten.sum"));
}

#[test]
fn test_convert_error_display() {
    use nn_import::ConvertError;
    let err = ConvertError::Compile("Metal pipeline creation failed".to_string());
    let msg = err.to_string();
    assert!(msg.contains("compilation error"));
    assert!(msg.contains("Metal pipeline"));

    let inner = ImportError::UnsupportedOp {
        target: "custom_op".to_string(),
    };
    let err: ConvertError = inner.into();
    let msg = err.to_string();
    assert!(msg.contains("import error"));
}

// ---------------------------------------------------------------------------
// 5. ResolvedWeight construction
// ---------------------------------------------------------------------------

#[test]
fn test_resolved_weight_new() {
    let w = ResolvedWeight::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
    assert_eq!(w.data.len(), 4);
    assert_eq!(w.shape, vec![2, 2]);
}

#[test]
fn test_resolved_weight_clone() {
    let w = ResolvedWeight::new(vec![1.0, 2.0], vec![2]);
    let w2 = w;
    assert_eq!(w2.data, vec![1.0, 2.0]);
    assert_eq!(w2.shape, vec![2]);
}

// ---------------------------------------------------------------------------
// 6. build_weight_map edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_build_weight_map_ignores_missing_fqns() {
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [
                    {"as_tensor": {"name": "p_w"}},
                    {"as_tensor": {"name": "x"}}
                ],
                "outputs": [{"as_tensor": {"name": "y"}}],
                "nodes": [],
                "tensor_values": {},
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_w"}, "parameter_name": "layer.weight"}},
                    {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "y"}}}}
                ]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut wd = HashMap::new();
    wd.insert("layer.weight".to_string(), (vec![1.0; 6], vec![2, 3]));
    wd.insert("extra.bias".to_string(), (vec![0.0; 3], vec![3]));

    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    assert!(wm.contains_key("p_w"), "mapped param should be present");
    assert_eq!(wm.len(), 1, "extra key should NOT appear in weight map");
}

#[test]
fn test_build_weight_map_empty_specs_empty_map() {
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [], "outputs": [], "nodes": [], "tensor_values": {}
            },
            "signature": {"input_specs": [], "output_specs": []},
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let wd = HashMap::new();
    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    assert!(wm.is_empty());
}

// ---------------------------------------------------------------------------
// 7. Graph building edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_build_graph_empty_no_ops() {
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
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.graph.len(), 1);
}

#[test]
fn test_build_graph_chained_unary_ops() {
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "tanh"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.relu.default",
                        "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
                        "outputs": [{"as_tensor": {"name": "relu"}}],
                        "metadata": {}
                    },
                    {
                        "target": "torch.ops.aten.tanh.default",
                        "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "relu"}}, "kind": 1}],
                        "outputs": [{"as_tensor": {"name": "tanh"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "x": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 3}, {"as_int": 1}]},
                    "relu": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 3}, {"as_int": 1}]},
                    "tanh": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 3}, {"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [
                    {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "tanh"}}}}
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
    assert_eq!(imported.graph.len(), 3);

    let compute_ops: Vec<_> = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .collect();
    assert_eq!(compute_ops.len(), 2);
    assert!(matches!(compute_ops[0].op(), TraceOp::Relu));
    assert!(matches!(compute_ops[1].op(), TraceOp::Tanh));
    assert_eq!(compute_ops[1].output_shape(), &[2, 3]);
}

#[test]
fn test_build_graph_topology_error_on_forward_reference() {
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

#[test]
fn test_build_graph_unsupported_op_errors() {
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "out"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.nonexistent_op.default",
                        "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
                        "outputs": [{"as_tensor": {"name": "out"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "out": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [
                    {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "out"}}}}
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
        matches!(err, ImportError::UnsupportedOp { .. }),
        "expected UnsupportedOp, got: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// 8. ImportedGraph constructor
// ---------------------------------------------------------------------------

#[test]
fn test_imported_graph_new() {
    use nn_core::dyn_tensor::trace::ComputationGraph;
    let cg = ComputationGraph::from_nodes(vec![]);
    let ig = nn_import::ImportedGraph::new(
        cg,
        2,
        vec!["a".to_string(), "b".to_string()],
        vec!["out".to_string()],
    );
    assert_eq!(ig.num_user_inputs, 2);
    assert_eq!(ig.user_input_names, vec!["a", "b"]);
    assert_eq!(ig.output_names, vec!["out"]);
}

// ---------------------------------------------------------------------------
// 9. Kokoro weights
// ---------------------------------------------------------------------------

#[test]
fn test_kokoro_validate_empty_keys() {
    let missing = nn_import::validate_kokoro_keys(&[]);
    assert_eq!(
        missing.len(),
        6,
        "all 6 prefixes should be missing with empty keys"
    );
}

#[test]
fn test_kokoro_map_pytorch_key_all_prefixes() {
    let keys = [
        "plbert.something",
        "bert_encoder.weight",
        "text_encoder.lstm.w",
        "prosody_predictor.shared.0.conv.weight",
        "predictor.F0.0.n1.fc.weight",
        "decoder.conv_pre.weight",
    ];
    for key in &keys {
        assert!(
            nn_import::map_pytorch_key(key).is_some(),
            "map_pytorch_key should accept '{key}'"
        );
    }
}

#[test]
fn test_kokoro_validate_safetensors_counts_only_mapped() {
    let keys: Vec<String> = vec![
        "plbert.x".to_string(),
        "bert_encoder.w".to_string(),
        "text_encoder.y".to_string(),
        "prosody_predictor.z".to_string(),
        "predictor.a".to_string(),
        "decoder.b".to_string(),
        "unknown.extra".to_string(),
    ];
    let result = nn_import::validate_kokoro_safetensors(&keys);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 6);
}

#[test]
fn test_kokoro_name_mapping_closure() {
    let mapper = nn_import::kokoro_name_mapping();
    assert_eq!(mapper("decoder.conv_pre.weight"), "decoder.conv_pre.weight");
    assert_eq!(mapper("unknown.key"), "unknown.key");
}

// ---------------------------------------------------------------------------
// 10. EquivalenceProof and report types
// ---------------------------------------------------------------------------

#[test]
fn test_equivalence_proof_all_none() {
    let proof = nn_import::EquivalenceProof::new(None, None, None);
    assert!(proof.kernel_safety.is_none());
    assert!(proof.composition_bounds.is_none());
    assert!(proof.reference_parity.is_none());
}

#[test]
fn test_equivalence_proof_with_populated_reports() {
    let proof = nn_import::EquivalenceProof::new(
        Some(nn_import::KaniSafetyReport::new(100, 98, 2)),
        Some(nn_import::CompositionBoundsReport::new(true, Some(1.5))),
        None,
    );
    let ks = proof.kernel_safety.unwrap();
    assert_eq!(ks.harness_count, 100);
    assert_eq!(ks.passed, 98);
    assert_eq!(ks.failed, 2);

    let cb = proof.composition_bounds.unwrap();
    assert!(cb.propagation_ok);
    assert!((cb.output_width.unwrap() - 1.5).abs() < f32::EPSILON);
}

#[test]
fn test_composition_bounds_report_no_width() {
    let cb = nn_import::CompositionBoundsReport::new(false, None);
    assert!(!cb.propagation_ok);
    assert!(cb.output_width.is_none());
}

// ---------------------------------------------------------------------------
// 11. ConvertReport public methods
// ---------------------------------------------------------------------------

#[test]
fn test_convert_report_verification_coverage_pct() {
    let mut vc = nn_import::VerificationCoverage::default();
    vc.gamma_crown_layers_covered = 10;
    vc.gamma_crown_layers_total = 20;
    let pct = vc.gamma_crown_coverage_pct();
    assert!((pct - 50.0).abs() < 0.01);
}

#[test]
fn test_convert_report_verification_coverage_zero_total() {
    let vc = nn_import::VerificationCoverage::default();
    assert!((vc.gamma_crown_coverage_pct()).abs() < f32::EPSILON);
}

// ---------------------------------------------------------------------------
// 12. Build graph with binary ops via JSON
// ---------------------------------------------------------------------------

#[test]
fn test_build_graph_add_binary() {
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [
                    {"as_tensor": {"name": "a"}},
                    {"as_tensor": {"name": "b"}}
                ],
                "outputs": [{"as_tensor": {"name": "sum"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.add.Tensor",
                        "inputs": [
                            {"name": "self", "arg": {"as_tensor": {"name": "a"}}, "kind": 1},
                            {"name": "other", "arg": {"as_tensor": {"name": "b"}}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "sum"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "a": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 3}, {"as_int": 1}]},
                    "b": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 3}, {"as_int": 1}]},
                    "sum": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 3}, {"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [
                    {"user_input": {"arg": {"as_tensor": {"name": "a"}}}},
                    {"user_input": {"arg": {"as_tensor": {"name": "b"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "sum"}}}}
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

    assert_eq!(imported.num_user_inputs, 2);
    assert_eq!(imported.user_input_names, vec!["a", "b"]);
    assert_eq!(imported.graph.len(), 3); // 2 inputs + 1 add

    let output = imported.graph.output_node().unwrap();
    assert!(matches!(output.op(), TraceOp::Add));
    assert_eq!(output.output_shape(), &[2, 3]);
}

// ---------------------------------------------------------------------------
// 13. Build graph: reshape op via JSON
// ---------------------------------------------------------------------------

#[test]
fn test_build_graph_reshape() {
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "reshaped"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.view.default",
                        "inputs": [
                            {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
                            {"name": "size", "arg": {"as_ints": [6, 2]}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "reshaped"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "x": {"dtype": 7, "sizes": [{"as_int": 3}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]},
                    "reshaped": {"dtype": 7, "sizes": [{"as_int": 6}, {"as_int": 2}], "requires_grad": false, "strides": [{"as_int": 2}, {"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [
                    {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "reshaped"}}}}
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
    assert!(
        matches!(output.op(), TraceOp::Reshape { target_shape } if *target_shape == vec![6, 2]),
        "expected Reshape [6,2], got: {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[6, 2]);
}

// ---------------------------------------------------------------------------
// 14. Input/Output spec classification
// ---------------------------------------------------------------------------

#[test]
fn test_parse_buffer_input_spec() {
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [
                    {"as_tensor": {"name": "p_bn_mean"}},
                    {"as_tensor": {"name": "x"}}
                ],
                "outputs": [{"as_tensor": {"name": "x"}}],
                "nodes": [],
                "tensor_values": {
                    "x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "p_bn_mean": {"dtype": 7, "sizes": [{"as_int": 8}], "requires_grad": false, "strides": [{"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [
                    {"buffer": {"arg": {"name": "p_bn_mean"}, "buffer_name": "bn.running_mean"}},
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
    let specs = &program.graph_module.signature.input_specs;
    assert_eq!(specs.len(), 2);
    assert!(
        matches!(specs[0], nn_import::InputSpec::Buffer(_)),
        "first spec should be Buffer, got: {:?}",
        specs[0]
    );
    assert!(matches!(specs[1], nn_import::InputSpec::UserInput(_)));
}

// ---------------------------------------------------------------------------
// 15. Multiple user inputs
// ---------------------------------------------------------------------------

#[test]
fn test_build_graph_multiple_user_inputs() {
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [
                    {"as_tensor": {"name": "x"}},
                    {"as_tensor": {"name": "y"}}
                ],
                "outputs": [{"as_tensor": {"name": "out"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.mul.Tensor",
                        "inputs": [
                            {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
                            {"name": "other", "arg": {"as_tensor": {"name": "y"}}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "out"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "y": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]},
                    "out": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [
                    {"user_input": {"arg": {"as_tensor": {"name": "x"}}}},
                    {"user_input": {"arg": {"as_tensor": {"name": "y"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "out"}}}}
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

    assert_eq!(imported.num_user_inputs, 2);
    assert_eq!(imported.user_input_names, vec!["x", "y"]);
    let output = imported.graph.output_node().unwrap();
    assert!(matches!(output.op(), TraceOp::Mul));
}

// ---------------------------------------------------------------------------
// 16. Tensor dtype detection through graph
// ---------------------------------------------------------------------------

#[test]
fn test_build_graph_bf16_dtype() {
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "relu"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.relu.default",
                        "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
                        "outputs": [{"as_tensor": {"name": "relu"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "x": {"dtype": 13, "sizes": [{"as_int": 2}, {"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 3}, {"as_int": 1}]},
                    "relu": {"dtype": 13, "sizes": [{"as_int": 2}, {"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 3}, {"as_int": 1}]}
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
    let imported = build_graph(&program, &empty_weights).unwrap();

    // BF16 dtype should be propagated to the input node
    let input_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Input))
        .unwrap();
    assert_eq!(input_node.output_dtype(), DType::BF16);
}
