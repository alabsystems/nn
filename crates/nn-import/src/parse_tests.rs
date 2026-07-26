// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for torch.export JSON parsing.

use super::*;

fn minimal_linear_json() -> &'static str {
    r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "p_weight"}}, {"as_tensor": {"name": "p_bias"}}, {"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "linear"}}], "nodes": [{"target": "torch.ops.aten.linear.default", "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}, {"name": "weight", "arg": {"as_tensor": {"name": "p_weight"}}, "kind": 1}, {"name": "bias", "arg": {"as_tensor": {"name": "p_bias"}}, "kind": 1}], "outputs": [{"as_tensor": {"name": "linear"}}], "metadata": {}}], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]}, "p_weight": {"dtype": 7, "sizes": [{"as_int": 3}, {"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 4}, {"as_int": 1}]}, "p_bias": {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": true, "strides": [{"as_int": 1}]}, "linear": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 3}, {"as_int": 1}]}}, "is_single_tensor_return": true}, "signature": {"input_specs": [{"parameter": {"arg": {"name": "p_weight"}, "parameter_name": "weight"}}, {"parameter": {"arg": {"name": "p_bias"}, "parameter_name": "bias"}}, {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "linear"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "opset_version": {"aten": 10}, "range_constraints": {}, "torch_version": "2.10.0"}"#
}

#[test]
fn test_parse_minimal_linear() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    assert_eq!(program.schema_version.major, 8);
    assert_eq!(program.schema_version.minor, 15);
    assert_eq!(program.graph_module.graph.nodes.len(), 1);
    assert_eq!(
        program.graph_module.graph.nodes[0].target,
        "torch.ops.aten.linear.default"
    );
    assert_eq!(program.graph_module.graph.nodes[0].inputs.len(), 3);
}

#[test]
fn test_parse_tensor_meta() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    let meta = program.graph_module.graph.tensor_values.get("x").unwrap();
    assert_eq!(meta.concrete_shape(), Some(vec![2, 4]));
    assert_eq!(meta.dtype, 7);
    assert_eq!(meta.to_dtype(), Some(nn_core::DType::F32));
}

#[test]
fn test_parse_input_specs() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    let specs = &program.graph_module.signature.input_specs;
    assert_eq!(specs.len(), 3);
    assert!(matches!(specs[0], InputSpec::Parameter(_)));
    assert!(matches!(specs[1], InputSpec::Parameter(_)));
    assert!(matches!(specs[2], InputSpec::UserInput(_)));
}

#[test]
fn test_parse_output_specs() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    assert_eq!(program.graph_module.signature.output_specs.len(), 1);
    assert!(matches!(
        program.graph_module.signature.output_specs[0],
        OutputSpec::UserOutput(_)
    ));
}

#[test]
fn test_reject_unsupported_schema() {
    let json = r#"{"graph_module": {"graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}}, "signature": {"input_specs": [], "output_specs": []}, "module_call_graph": []}, "schema_version": {"major": 7, "minor": 0}, "range_constraints": {}}"#;
    assert!(matches!(
        parse_exported_program(json.as_bytes()).unwrap_err(),
        crate::ImportError::UnsupportedSchema { major: 7, .. }
    ));
}

#[test]
fn test_argument_accessors() {
    let arg: Argument = serde_json::from_str(r#"{"as_int": 42}"#).unwrap();
    assert_eq!(arg.as_int(), Some(42));
    assert_eq!(arg.as_float(), None);
    assert!(arg.as_tensor_name().is_none());
    let arg: Argument = serde_json::from_str(r#"{"as_tensor": {"name": "foo"}}"#).unwrap();
    assert_eq!(arg.as_tensor_name(), Some("foo"));
    let arg: Argument = serde_json::from_str(r#"{"as_none": true}"#).unwrap();
    assert!(arg.is_none());
}

#[test]
fn test_sym_int_concrete() {
    assert_eq!(
        serde_json::from_str::<SymInt>(r#"{"as_int": 128}"#)
            .unwrap()
            .as_concrete(),
        Some(128)
    );
}

#[test]
fn test_parse_invalid_json() {
    assert!(matches!(
        parse_exported_program(b"bad").unwrap_err(),
        crate::ImportError::JsonParse(_)
    ));
}
#[test]
fn test_parse_truncated_json() {
    assert!(matches!(
        parse_exported_program(b"{\"graph_module\":").unwrap_err(),
        crate::ImportError::JsonParse(_)
    ));
}
#[test]
fn test_parse_empty_bytes() {
    assert!(matches!(
        parse_exported_program(b"").unwrap_err(),
        crate::ImportError::JsonParse(_)
    ));
}

#[test]
fn test_reject_future_schema() {
    let json = r#"{"graph_module": {"graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}}, "signature": {"input_specs": [], "output_specs": []}, "module_call_graph": []}, "schema_version": {"major": 9, "minor": 0}, "range_constraints": {}}"#;
    assert!(matches!(
        parse_exported_program(json.as_bytes()).unwrap_err(),
        crate::ImportError::UnsupportedSchema { major: 9, .. }
    ));
}

#[test]
fn test_parse_empty_graph() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "x"}}], "nodes": [], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}}}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    assert!(parse_exported_program(json.as_bytes())
        .unwrap()
        .graph_module
        .graph
        .nodes
        .is_empty());
}

#[test]
fn test_node_input_names() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    let names: Vec<&str> = program.graph_module.graph.nodes[0]
        .inputs
        .iter()
        .map(|na| na.name.as_str())
        .collect();
    assert_eq!(names, vec!["input", "weight", "bias"]);
}

#[test]
fn test_edge_refs_inputs() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    let refs: Vec<&str> = program.graph_module.graph.nodes[0]
        .inputs
        .iter()
        .filter_map(|na| na.arg.as_tensor_name())
        .collect();
    assert_eq!(refs, vec!["x", "p_weight", "p_bias"]);
}

#[test]
fn test_edge_refs_outputs() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    let refs: Vec<&str> = program.graph_module.graph.nodes[0]
        .outputs
        .iter()
        .filter_map(|a| a.as_tensor_name())
        .collect();
    assert_eq!(refs, vec!["linear"]);
}

#[test]
fn test_tensor_meta_strides() {
    let strides: Vec<i64> = parse_exported_program(minimal_linear_json().as_bytes())
        .unwrap()
        .graph_module
        .graph
        .tensor_values
        .get("x")
        .unwrap()
        .strides
        .iter()
        .filter_map(SymInt::as_concrete)
        .collect();
    assert_eq!(strides, vec![4, 1]);
}

#[test]
fn test_tensor_meta_requires_grad() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    assert!(
        !program
            .graph_module
            .graph
            .tensor_values
            .get("x")
            .unwrap()
            .requires_grad
    );
    assert!(
        program
            .graph_module
            .graph
            .tensor_values
            .get("p_weight")
            .unwrap()
            .requires_grad
    );
}

#[test]
fn test_all_scalar_type_mappings() {
    assert_eq!(scalar_type_to_dtype(1), Some(nn_core::DType::U8));
    assert_eq!(scalar_type_to_dtype(5), Some(nn_core::DType::I64));
    assert_eq!(scalar_type_to_dtype(6), Some(nn_core::DType::F16));
    assert_eq!(scalar_type_to_dtype(7), Some(nn_core::DType::F32));
    assert_eq!(scalar_type_to_dtype(8), Some(nn_core::DType::F64));
    assert_eq!(scalar_type_to_dtype(13), Some(nn_core::DType::BF16));
    assert_eq!(scalar_type_to_dtype(0), None);
    assert_eq!(scalar_type_to_dtype(99), None);
}

#[test]
fn test_sym_int_symbolic_returns_none() {
    assert_eq!(
        serde_json::from_str::<SymInt>(
            r#"{"as_expr": {"expr_str": "s0", "hint": {"as_int": 128}}}"#
        )
        .unwrap()
        .as_concrete(),
        None
    );
}

#[test]
fn test_argument_float() {
    assert!(
        (serde_json::from_str::<Argument>(r#"{"as_float": 3.14}"#)
            .unwrap()
            .as_float()
            .unwrap()
            - 3.14)
            .abs()
            < 1e-10
    );
}
#[test]
fn test_argument_bool() {
    assert_eq!(
        serde_json::from_str::<Argument>(r#"{"as_bool": true}"#)
            .unwrap()
            .as_bool_val(),
        Some(true)
    );
    assert_eq!(
        serde_json::from_str::<Argument>(r#"{"as_bool": false}"#)
            .unwrap()
            .as_bool_val(),
        Some(false)
    );
}
#[test]
fn test_argument_string() {
    assert_eq!(
        serde_json::from_str::<Argument>(r#"{"as_string": "hello"}"#)
            .unwrap()
            .as_string(),
        Some("hello")
    );
}
#[test]
fn test_argument_ints() {
    assert_eq!(
        serde_json::from_str::<Argument>(r#"{"as_ints": [1, 2, 3]}"#)
            .unwrap()
            .as_ints(),
        Some(&[1i64, 2, 3][..])
    );
}
#[test]
fn test_argument_tensors_list() {
    assert_eq!(
        serde_json::from_str::<Argument>(r#"{"as_tensors": [{"name": "a"}, {"name": "b"}]}"#)
            .unwrap()
            .as_tensor_names()
            .unwrap(),
        vec!["a", "b"]
    );
}
#[test]
fn test_argument_other_fallback() {
    assert!(matches!(
        serde_json::from_str::<Argument>(r#"{"unknown_key": 999}"#).unwrap(),
        Argument::Other(_)
    ));
}

#[test]
fn test_parse_buffer_input_spec() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "p_rm"}}, {"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "x"}}], "nodes": [], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "p_rm": {"dtype": 7, "sizes": [{"as_int": 16}], "requires_grad": false, "strides": [{"as_int": 1}]}}}, "signature": {"input_specs": [{"buffer": {"arg": {"name": "p_rm"}, "buffer_name": "running_mean", "persistent": true}}, {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    assert!(matches!(
        parse_exported_program(json.as_bytes())
            .unwrap()
            .graph_module
            .signature
            .input_specs[0],
        InputSpec::Buffer(_)
    ));
}

#[test]
fn test_parse_opset_version() {
    assert_eq!(
        parse_exported_program(minimal_linear_json().as_bytes())
            .unwrap()
            .opset_version
            .get("aten"),
        Some(&10)
    );
}
#[test]
fn test_parse_torch_version() {
    assert_eq!(
        parse_exported_program(minimal_linear_json().as_bytes())
            .unwrap()
            .torch_version
            .as_deref(),
        Some("2.10.0")
    );
}

#[test]
fn test_parse_range_constraints() {
    let json = r#"{"graph_module": {"graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}}, "signature": {"input_specs": [], "output_specs": []}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {"s0": {"min_val": 2, "max_val": 1024}}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let rc = program.range_constraints.get("s0").unwrap();
    assert_eq!(rc.min_val, 2);
    assert_eq!(rc.max_val, 1024);
}

#[test]
fn test_parse_multi_input_graph() {
    let json = include_str!("../test_data/multi_input_cat.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert_eq!(program.graph_module.graph.inputs.len(), 4);
    assert_eq!(program.graph_module.graph.nodes.len(), 3);
    assert_eq!(
        program
            .graph_module
            .signature
            .input_specs
            .iter()
            .filter(|s| matches!(s, InputSpec::UserInput(_)))
            .count(),
        2
    );
}

#[test]
fn test_roundtrip_structure() {
    let p1 = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    let p2 = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    assert_eq!(
        p1.graph_module.graph.nodes.len(),
        p2.graph_module.graph.nodes.len()
    );
    assert_eq!(p1.schema_version.major, p2.schema_version.major);
}

#[test]
fn test_tensor_meta_with_device() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "x"}}], "nodes": [], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}], "device": {"type": "cuda", "index": 0}}}}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let device = program
        .graph_module
        .graph
        .tensor_values
        .get("x")
        .unwrap()
        .device
        .as_ref()
        .unwrap();
    assert_eq!(device.device_type, "cuda");
    assert_eq!(device.index, Some(0));
}

#[test]
fn test_parse_missing_optional_fields() {
    let json = r#"{"graph_module": {"graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}}, "signature": {"input_specs": [], "output_specs": []}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 0}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert!(program.opset_version.is_empty());
    assert!(program.torch_version.is_none());
}

#[test]
fn test_tensor_meta_concrete_shape_1d() {
    assert_eq!(
        parse_exported_program(minimal_linear_json().as_bytes())
            .unwrap()
            .graph_module
            .graph
            .tensor_values
            .get("p_bias")
            .unwrap()
            .concrete_shape(),
        Some(vec![3])
    );
}

// ---------------------------------------------------------------------------
// Additional tests: symbolic dimensions, output variants, input spec variants,
// argument edge cases, multi-output, scalar type args
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_meta_symbolic_dimension_returns_none() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "x"}}], "nodes": [], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_expr": {"expr_str": "s0", "hint": {"as_int": 128}}}], "requires_grad": false, "strides": [{"as_int": 128}, {"as_int": 1}]}}}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let meta = program.graph_module.graph.tensor_values.get("x").unwrap();
    // Symbolic dimension should cause concrete_shape to return None
    assert_eq!(meta.concrete_shape(), None);
}

#[test]
fn test_parse_buffer_mutation_output_spec() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "x"}}], "nodes": [], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}}}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"buffer_mutation": {"arg": {"name": "p_running_mean"}, "buffer_name": "running_mean"}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert_eq!(program.graph_module.signature.output_specs.len(), 1);
    assert!(matches!(
        program.graph_module.signature.output_specs[0],
        OutputSpec::BufferMutation(_)
    ));
}

#[test]
fn test_parse_tensor_constant_input_spec() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "c0"}}, {"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "x"}}], "nodes": [], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "c0": {"dtype": 7, "sizes": [{"as_int": 8}], "requires_grad": false, "strides": [{"as_int": 1}]}}}, "signature": {"input_specs": [{"tensor_constant": {"arg": {"name": "c0"}, "tensor_constant_name": "const_tensor_0"}}, {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert!(matches!(
        program.graph_module.signature.input_specs[0],
        InputSpec::TensorConstant(_)
    ));
}

#[test]
fn test_parse_token_input_spec() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "token_0"}}, {"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "x"}}], "nodes": [], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "token_0": {"dtype": 7, "sizes": [], "requires_grad": false, "strides": []}}}, "signature": {"input_specs": [{"token": {"arg": {"name": "token_0"}}}, {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert!(matches!(
        program.graph_module.signature.input_specs[0],
        InputSpec::Token(_)
    ));
}

#[test]
fn test_parse_is_single_tensor_return_false() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "a"}}, {"as_tensor": {"name": "b"}}], "nodes": [], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "a": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "b": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}}, "is_single_tensor_return": false}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "a"}}}}, {"user_output": {"arg": {"as_tensor": {"name": "b"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert!(!program.graph_module.graph.is_single_tensor_return);
    assert_eq!(program.graph_module.signature.output_specs.len(), 2);
}

#[test]
fn test_argument_scalar_type() {
    let arg: Argument = serde_json::from_str(r#"{"as_scalar_type": 7}"#).unwrap();
    assert!(matches!(arg, Argument::ScalarType(_)));
    // ScalarType is not extractable via as_int/as_float
    assert_eq!(arg.as_int(), None);
    assert_eq!(arg.as_float(), None);
}

#[test]
fn test_argument_memory_format() {
    let arg: Argument = serde_json::from_str(r#"{"as_memory_format": 0}"#).unwrap();
    assert!(matches!(arg, Argument::MemoryFormat(_)));
    assert!(arg.as_tensor_name().is_none());
}

#[test]
fn test_argument_sym_int() {
    let arg: Argument = serde_json::from_str(r#"{"as_sym_int": {"as_int": 256}}"#).unwrap();
    assert!(matches!(arg, Argument::SymInt(_)));
}

#[test]
fn test_argument_sym_ints() {
    let arg: Argument =
        serde_json::from_str(r#"{"as_sym_ints": [{"as_int": 1}, {"as_int": 2}]}"#).unwrap();
    assert!(matches!(arg, Argument::SymInts(_)));
}

#[test]
fn test_argument_device() {
    let arg: Argument =
        serde_json::from_str(r#"{"as_device": {"type": "cpu", "index": null}}"#).unwrap();
    assert!(matches!(arg, Argument::Device(_)));
    assert!(arg.as_tensor_name().is_none());
}

#[test]
fn test_argument_optional_tensors() {
    let arg: Argument = serde_json::from_str(
        r#"{"as_optional_tensors": [{"as_tensor": {"name": "a"}}, {"as_none": true}]}"#,
    )
    .unwrap();
    assert!(matches!(arg, Argument::OptionalTensors(_)));
}

#[test]
fn test_argument_floats_list() {
    let arg: Argument = serde_json::from_str(r#"{"as_floats": [1.0, 2.5, -3.0]}"#).unwrap();
    assert!(matches!(arg, Argument::Floats(_)));
    // Floats variant does not expose via as_float (that's for single values)
    assert_eq!(arg.as_float(), None);
}

#[test]
fn test_argument_bools_list() {
    let arg: Argument = serde_json::from_str(r#"{"as_bools": [true, false, true]}"#).unwrap();
    assert!(matches!(arg, Argument::Bools(_)));
    assert_eq!(arg.as_bool_val(), None);
}

#[test]
fn test_parse_multiple_output_specs() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "out1"}}, {"as_tensor": {"name": "out2"}}, {"as_tensor": {"name": "out3"}}], "nodes": [], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "out1": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "out2": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "out3": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}}, "is_single_tensor_return": false}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "out1"}}}}, {"user_output": {"arg": {"as_tensor": {"name": "out2"}}}}, {"user_output": {"arg": {"as_tensor": {"name": "out3"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert_eq!(program.graph_module.signature.output_specs.len(), 3);
    for spec in &program.graph_module.signature.output_specs {
        assert!(matches!(spec, OutputSpec::UserOutput(_)));
    }
}

#[test]
fn test_parse_node_with_metadata() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "out"}}], "nodes": [{"target": "torch.ops.aten.relu.default", "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}], "outputs": [{"as_tensor": {"name": "out"}}], "metadata": {"source_fn_stack": "Linear.forward", "stack_trace": "file.py:10"}}], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "out": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}}}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "out"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let node = &program.graph_module.graph.nodes[0];
    assert_eq!(node.metadata.len(), 2);
    assert!(node.metadata.contains_key("source_fn_stack"));
    assert!(node.metadata.contains_key("stack_trace"));
}

#[test]
fn test_tensor_meta_dtype_f16() {
    let meta: TensorMeta = serde_json::from_str(r#"{"dtype": 6, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}"#).unwrap();
    assert_eq!(meta.to_dtype(), Some(nn_core::DType::F16));
    assert_eq!(meta.concrete_shape(), Some(vec![4]));
}

#[test]
fn test_tensor_meta_dtype_i64() {
    let meta: TensorMeta = serde_json::from_str(r#"{"dtype": 5, "sizes": [{"as_int": 2}, {"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 3}, {"as_int": 1}]}"#).unwrap();
    assert_eq!(meta.to_dtype(), Some(nn_core::DType::I64));
    assert_eq!(meta.concrete_shape(), Some(vec![2, 3]));
}

#[test]
fn test_tensor_meta_unknown_dtype() {
    let meta: TensorMeta = serde_json::from_str(r#"{"dtype": 99, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}"#).unwrap();
    assert_eq!(meta.to_dtype(), None);
}

#[test]
fn test_tensor_meta_empty_shape() {
    let meta: TensorMeta =
        serde_json::from_str(r#"{"dtype": 7, "sizes": [], "requires_grad": false, "strides": []}"#)
            .unwrap();
    assert_eq!(meta.concrete_shape(), Some(vec![]));
}

#[test]
fn test_parse_graph_output_tensors() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    let output_names: Vec<&str> = program
        .graph_module
        .graph
        .outputs
        .iter()
        .filter_map(|a| a.as_tensor_name())
        .collect();
    assert_eq!(output_names, vec!["linear"]);
}

#[test]
fn test_parse_graph_input_tensors() {
    let program = parse_exported_program(minimal_linear_json().as_bytes()).unwrap();
    let input_names: Vec<&str> = program
        .graph_module
        .graph
        .inputs
        .iter()
        .filter_map(|a| a.as_tensor_name())
        .collect();
    assert_eq!(input_names, vec!["p_weight", "p_bias", "x"]);
}
