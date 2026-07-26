// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for JSON graph parsing, aten op mapping dispatch,
//! weight mapping, input/output spec parsing, and error handling for
//! malformed graphs.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;
use nn_core::dyn_tensor::CompareOp;
use nn_core::DType;

use crate::error::ImportError;
use crate::graph_build::{build_graph, build_weight_map};
use crate::op_map::{map_node_to_trace_op, supported_ops, OpMapContext, ResolvedWeight};
use crate::parse::{
    Argument, ArgumentBool, ArgumentFloat, ArgumentInt, ArgumentInts, ArgumentNone, ArgumentString,
    ArgumentTensor, ArgumentTensors, InputSpec, NamedArgument, Node, OutputSpec, SymInt,
    SymIntConcrete, TensorArgument, TensorMeta,
};
use crate::parse_exported_program;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn empty_ctx() -> OpMapContext<'static> {
    let meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::default());
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::default());
    OpMapContext {
        tensor_meta: meta,
        weights,
    }
}

fn tensor_arg(name: &str) -> Argument {
    Argument::Tensor(ArgumentTensor {
        as_tensor: TensorArgument {
            name: name.to_string(),
        },
    })
}

fn int_arg(val: i64) -> Argument {
    Argument::Int(ArgumentInt { as_int: val })
}

fn ints_arg(vals: &[i64]) -> Argument {
    Argument::Ints(ArgumentInts {
        as_ints: vals.to_vec(),
    })
}

fn float_arg(val: f64) -> Argument {
    Argument::Float(ArgumentFloat { as_float: val })
}

fn bool_arg(val: bool) -> Argument {
    Argument::Bool(ArgumentBool { as_bool: val })
}

fn none_arg() -> Argument {
    Argument::None(ArgumentNone { as_none: true })
}

fn str_arg(val: &str) -> Argument {
    Argument::Str(ArgumentString {
        as_string: val.to_string(),
    })
}

fn named(name: &str, arg: Argument) -> NamedArgument {
    NamedArgument {
        name: name.to_string(),
        arg,
        kind: Some(1),
    }
}

fn simple_node(target: &str, inputs: Vec<NamedArgument>) -> Node {
    Node {
        target: target.to_string(),
        inputs,
        outputs: vec![tensor_arg("output")],
        metadata: HashMap::new(),
    }
}

// ===========================================================================
// Section 1: JSON graph parsing (parse_exported_program)
// ===========================================================================

#[test]
fn test_parse_graph_with_multiple_nodes() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "relu_out"}}], "nodes": [
        {"target": "torch.ops.aten.relu.default", "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}], "outputs": [{"as_tensor": {"name": "relu_out"}}], "metadata": {}},
        {"target": "torch.ops.aten.sigmoid.default", "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "relu_out"}}, "kind": 1}], "outputs": [{"as_tensor": {"name": "sig_out"}}], "metadata": {}}
    ], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]}, "relu_out": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]}, "sig_out": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]}}, "is_single_tensor_return": true}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "sig_out"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert_eq!(program.graph_module.graph.nodes.len(), 2);
    assert_eq!(
        program.graph_module.graph.nodes[0].target,
        "torch.ops.aten.relu.default"
    );
    assert_eq!(
        program.graph_module.graph.nodes[1].target,
        "torch.ops.aten.sigmoid.default"
    );
}

#[test]
fn test_parse_graph_preserves_node_order() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "z"}}], "nodes": [
        {"target": "torch.ops.aten.exp.default", "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}], "outputs": [{"as_tensor": {"name": "y"}}], "metadata": {}},
        {"target": "torch.ops.aten.log.default", "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "y"}}, "kind": 1}], "outputs": [{"as_tensor": {"name": "z"}}], "metadata": {}}
    ], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "y": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "z": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}}, "is_single_tensor_return": true}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "z"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let targets: Vec<&str> = program
        .graph_module
        .graph
        .nodes
        .iter()
        .map(|n| n.target.as_str())
        .collect();
    assert_eq!(
        targets,
        vec!["torch.ops.aten.exp.default", "torch.ops.aten.log.default"]
    );
}

#[test]
fn test_parse_graph_with_high_rank_tensor() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "x"}}], "nodes": [], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 3}, {"as_int": 4}, {"as_int": 5}, {"as_int": 6}], "requires_grad": false, "strides": [{"as_int": 360}, {"as_int": 120}, {"as_int": 30}, {"as_int": 6}, {"as_int": 1}]}}}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let meta = program.graph_module.graph.tensor_values.get("x").unwrap();
    assert_eq!(meta.concrete_shape(), Some(vec![2, 3, 4, 5, 6]));
}

#[test]
fn test_parse_graph_scalar_tensor_zero_dim() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "s"}}], "outputs": [{"as_tensor": {"name": "s"}}], "nodes": [], "tensor_values": {"s": {"dtype": 7, "sizes": [], "requires_grad": false, "strides": []}}}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "s"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "s"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let meta = program.graph_module.graph.tensor_values.get("s").unwrap();
    assert_eq!(meta.concrete_shape(), Some(vec![]));
    assert_eq!(meta.to_dtype(), Some(DType::F32));
}

#[test]
fn test_parse_graph_with_bf16_dtype() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "x"}}], "nodes": [], "tensor_values": {"x": {"dtype": 13, "sizes": [{"as_int": 8}], "requires_grad": false, "strides": [{"as_int": 1}]}}}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let meta = program.graph_module.graph.tensor_values.get("x").unwrap();
    assert_eq!(meta.to_dtype(), Some(DType::BF16));
}

#[test]
fn test_parse_graph_with_f64_dtype() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "x"}}], "nodes": [], "tensor_values": {"x": {"dtype": 8, "sizes": [{"as_int": 3}], "requires_grad": false, "strides": [{"as_int": 1}]}}}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert_eq!(
        program
            .graph_module
            .graph
            .tensor_values
            .get("x")
            .unwrap()
            .to_dtype(),
        Some(DType::F64)
    );
}

#[test]
fn test_parse_graph_u8_dtype() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "x"}}], "nodes": [], "tensor_values": {"x": {"dtype": 1, "sizes": [{"as_int": 10}], "requires_grad": false, "strides": [{"as_int": 1}]}}}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert_eq!(
        program
            .graph_module
            .graph
            .tensor_values
            .get("x")
            .unwrap()
            .to_dtype(),
        Some(DType::U8)
    );
}

#[test]
fn test_parse_graph_with_range_constraints() {
    let json = r#"{"graph_module": {"graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}}, "signature": {"input_specs": [], "output_specs": []}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {"s0": {"min_val": 1, "max_val": 512}, "s1": {"min_val": 0, "max_val": 2048}}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert_eq!(program.range_constraints.len(), 2);
    let rc = program.range_constraints.get("s0").unwrap();
    assert_eq!(rc.min_val, 1);
    assert_eq!(rc.max_val, 512);
}

#[test]
fn test_parse_graph_with_opset_versions() {
    let json = r#"{"graph_module": {"graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}}, "signature": {"input_specs": [], "output_specs": []}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "opset_version": {"aten": 10, "custom": 1}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert_eq!(program.opset_version.get("aten"), Some(&10));
    assert_eq!(program.opset_version.get("custom"), Some(&1));
}

// ===========================================================================
// Section 2: Op mapping from aten names (map_node_to_trace_op)
// ===========================================================================

#[test]
fn test_map_exp_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.exp.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Exp));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_log_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.log.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Log));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_sqrt_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sqrt.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Sqrt));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_abs_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.abs.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Abs));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_neg_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.neg.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Neg));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_sin_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sin.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Sin));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_cos_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cos.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Cos));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_floor_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.floor.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Floor));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_round_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.round.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Round));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_rsqrt_to_powf_neg_half() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.rsqrt.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match op {
        TraceOp::Powf { exponent } => {
            assert!(
                (exponent - (-0.5)).abs() < 1e-10,
                "rsqrt exponent should be -0.5"
            );
        }
        other => panic!("expected Powf, got {other:?}"),
    }
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_add_tensor() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.add.Tensor",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Add));
    assert_eq!(inputs, vec!["a", "b"]);
}

#[test]
fn test_map_sub_tensor() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sub.Tensor",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Sub));
    assert_eq!(inputs, vec!["a", "b"]);
}

#[test]
fn test_map_mul_tensor() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.mul.Tensor",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Mul));
    assert_eq!(inputs, vec!["a", "b"]);
}

#[test]
fn test_map_div_tensor() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.div.Tensor",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Div));
    assert_eq!(inputs, vec!["a", "b"]);
}

#[test]
fn test_map_maximum_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.maximum.default",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Maximum));
    assert_eq!(inputs, vec!["a", "b"]);
}

#[test]
fn test_map_minimum_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.minimum.default",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Minimum));
    assert_eq!(inputs, vec!["a", "b"]);
}

#[test]
fn test_map_mm_to_matmul() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.mm.default",
        vec![
            named("self", tensor_arg("a")),
            named("mat2", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::MatMul));
    assert_eq!(inputs, vec!["a", "b"]);
}

#[test]
fn test_map_bmm_to_matmul() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.bmm.default",
        vec![
            named("self", tensor_arg("a")),
            named("mat2", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::MatMul));
    assert_eq!(inputs, vec!["a", "b"]);
}

#[test]
fn test_map_matmul_uses_other_arg() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.matmul.default",
        vec![
            named("self", tensor_arg("q")),
            named("other", tensor_arg("k")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::MatMul));
    assert_eq!(inputs, vec!["q", "k"]);
}

#[test]
fn test_map_unsupported_op_error() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.some_imaginary_op.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ImportError::UnsupportedOp { .. }));
}

#[test]
fn test_map_reshape_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.view.default",
        vec![
            named("self", tensor_arg("x")),
            named("size", ints_arg(&[2, 3, 4])),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match op {
        TraceOp::Reshape { target_shape } => {
            assert_eq!(target_shape, vec![2, 3, 4]);
        }
        other => panic!("expected Reshape, got {other:?}"),
    }
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_transpose_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.transpose.int",
        vec![
            named("self", tensor_arg("x")),
            named("dim0", int_arg(0)),
            named("dim1", int_arg(1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match op {
        TraceOp::Transpose { dim0, dim1 } => {
            assert_eq!(dim0, 0);
            assert_eq!(dim1, 1);
        }
        other => panic!("expected Transpose, got {other:?}"),
    }
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_permute_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.permute.default",
        vec![
            named("self", tensor_arg("x")),
            named("dims", ints_arg(&[0, 2, 1])),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match op {
        TraceOp::Permute { axes } => {
            assert_eq!(axes, vec![0, 2, 1]);
        }
        other => panic!("expected Permute, got {other:?}"),
    }
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_unsqueeze_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.unsqueeze.default",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(1))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match op {
        TraceOp::Unsqueeze { dim } => assert_eq!(dim, 1),
        other => panic!("expected Unsqueeze, got {other:?}"),
    }
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_squeeze_dim_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.squeeze.dim",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(2))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match op {
        TraceOp::Squeeze { dim } => assert_eq!(dim, 2),
        other => panic!("expected Squeeze, got {other:?}"),
    }
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_dropout_is_identity() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.dropout.default",
        vec![
            named("self", tensor_arg("x")),
            named("p", float_arg(0.1)),
            named("train", bool_arg(false)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Dropout));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_relu_preserves_tensor_name() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.relu.default",
        vec![named("self", tensor_arg("nn_custom_tensor_123"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Relu));
    assert_eq!(inputs, vec!["nn_custom_tensor_123"]);
}

#[test]
fn test_map_gt_scalar() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.gt.Scalar",
        vec![
            named("self", tensor_arg("x")),
            named("other", float_arg(0.5)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match op {
        TraceOp::Compare { op, value } => {
            assert!(matches!(op, CompareOp::Gt));
            assert!((value - 0.5).abs() < 1e-10);
        }
        other => panic!("expected Compare Gt, got {other:?}"),
    }
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_lt_scalar() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.lt.Scalar",
        vec![
            named("self", tensor_arg("x")),
            named("other", float_arg(-1.0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match op {
        TraceOp::Compare { op, value } => {
            assert!(matches!(op, CompareOp::Lt));
            assert!((value - (-1.0)).abs() < 1e-10);
        }
        other => panic!("expected Compare Lt, got {other:?}"),
    }
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_eq_scalar() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.eq.Scalar",
        vec![
            named("self", tensor_arg("x")),
            named("other", float_arg(0.0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match op {
        TraceOp::Compare { op, value } => {
            assert!(matches!(op, CompareOp::Eq));
            assert!(value.abs() < 1e-10);
        }
        other => panic!("expected Compare Eq, got {other:?}"),
    }
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_silu_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.silu.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Silu));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_hardsigmoid_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.hardsigmoid.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::HardSigmoid));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_hardswish_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.hardswish.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::HardSwish));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_selu_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.selu.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Selu));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_mish_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.mish.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Mish));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_sigmoid_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sigmoid.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Sigmoid));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_recip_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.reciprocal.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Recip));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_contiguous_identity() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.contiguous.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    // contiguous should map to identity (Contiguous)
    match op {
        TraceOp::Reshape { target_shape } => assert!(
            target_shape.is_empty(),
            "identity should have empty target_shape"
        ),
        other => panic!("expected Reshape, got {other:?}"),
    }
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_clone_identity() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.clone.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match op {
        TraceOp::Reshape { target_shape } => assert!(
            target_shape.is_empty(),
            "identity should have empty target_shape"
        ),
        other => panic!("expected Reshape, got {other:?}"),
    }
    assert_eq!(inputs, vec!["x"]);
}

// ===========================================================================
// Section 3: Weight mapping (build_weight_map)
// ===========================================================================

#[test]
fn test_build_weight_map_maps_parameter_fqn_to_graph_name() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "p_weight"}}, {"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "x"}}], "nodes": [], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "p_weight": {"dtype": 7, "sizes": [{"as_int": 3}, {"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 4}, {"as_int": 1}]}}}, "signature": {"input_specs": [{"parameter": {"arg": {"name": "p_weight"}, "parameter_name": "linear.weight"}}, {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    weight_data.insert("linear.weight".to_string(), (vec![1.0; 12], vec![3, 4]));

    let wm = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    assert!(wm.contains_key("p_weight"));
    assert_eq!(wm["p_weight"].shape, vec![3, 4]);
    assert_eq!(wm["p_weight"].data.len(), 12);
}

#[test]
fn test_build_weight_map_maps_buffer() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "b_rm"}}, {"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "x"}}], "nodes": [], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "b_rm": {"dtype": 7, "sizes": [{"as_int": 16}], "requires_grad": false, "strides": [{"as_int": 1}]}}}, "signature": {"input_specs": [{"buffer": {"arg": {"name": "b_rm"}, "buffer_name": "bn.running_mean", "persistent": true}}, {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    weight_data.insert("bn.running_mean".to_string(), (vec![0.0; 16], vec![16]));

    let wm = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    assert!(wm.contains_key("b_rm"));
    assert_eq!(wm["b_rm"].shape, vec![16]);
}

#[test]
fn test_build_weight_map_missing_weight_is_absent() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "p_w"}}, {"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "x"}}], "nodes": [], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "p_w": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 1}]}}}, "signature": {"input_specs": [{"parameter": {"arg": {"name": "p_w"}, "parameter_name": "fc.weight"}}, {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let wm = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    assert!(!wm.contains_key("p_w"), "missing weight should not appear");
}

#[test]
fn test_build_weight_map_multiple_parameters() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "p_w"}}, {"as_tensor": {"name": "p_b"}}, {"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "x"}}], "nodes": [], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "p_w": {"dtype": 7, "sizes": [{"as_int": 3}, {"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 4}, {"as_int": 1}]}, "p_b": {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": true, "strides": [{"as_int": 1}]}}}, "signature": {"input_specs": [{"parameter": {"arg": {"name": "p_w"}, "parameter_name": "fc.weight"}}, {"parameter": {"arg": {"name": "p_b"}, "parameter_name": "fc.bias"}}, {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    weight_data.insert("fc.weight".to_string(), (vec![1.0; 12], vec![3, 4]));
    weight_data.insert("fc.bias".to_string(), (vec![0.5; 3], vec![3]));

    let wm = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);
    assert!(wm.contains_key("p_w"));
    assert!(wm.contains_key("p_b"));
    assert_eq!(wm["p_b"].data.len(), 3);
}

// ===========================================================================
// Section 4: Input/output spec parsing
// ===========================================================================

#[test]
fn test_input_spec_parameter_has_correct_fqn() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "p_lin_weight"}}], "outputs": [], "nodes": [], "tensor_values": {"p_lin_weight": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 1}]}}}, "signature": {"input_specs": [{"parameter": {"arg": {"name": "p_lin_weight"}, "parameter_name": "encoder.layers.0.linear.weight"}}], "output_specs": []}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    match &program.graph_module.signature.input_specs[0] {
        InputSpec::Parameter(p) => {
            assert_eq!(p.parameter.arg.name, "p_lin_weight");
            assert_eq!(p.parameter.parameter_name, "encoder.layers.0.linear.weight");
        }
        other => panic!("expected Parameter, got {other:?}"),
    }
}

#[test]
fn test_input_spec_buffer_persistent_flag() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "b"}}], "outputs": [], "nodes": [], "tensor_values": {"b": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}}}, "signature": {"input_specs": [{"buffer": {"arg": {"name": "b"}, "buffer_name": "running_var", "persistent": false}}], "output_specs": []}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    match &program.graph_module.signature.input_specs[0] {
        InputSpec::Buffer(b) => {
            assert!(!b.buffer.persistent);
            assert_eq!(b.buffer.buffer_name, "running_var");
        }
        other => panic!("expected Buffer, got {other:?}"),
    }
}

#[test]
fn test_output_spec_user_output_tensor_name() {
    let json = r#"{"graph_module": {"graph": {"inputs": [], "outputs": [{"as_tensor": {"name": "final_out"}}], "nodes": [], "tensor_values": {}}, "signature": {"input_specs": [], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "final_out"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    match &program.graph_module.signature.output_specs[0] {
        OutputSpec::UserOutput(u) => {
            assert_eq!(u.user_output.arg.as_tensor_name(), Some("final_out"));
        }
        other => panic!("expected UserOutput, got {other:?}"),
    }
}

#[test]
fn test_output_spec_buffer_mutation() {
    let json = r#"{"graph_module": {"graph": {"inputs": [], "outputs": [{"as_tensor": {"name": "bm"}}], "nodes": [], "tensor_values": {}}, "signature": {"input_specs": [], "output_specs": [{"buffer_mutation": {"arg": {"name": "bm"}, "buffer_name": "running_mean"}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    match &program.graph_module.signature.output_specs[0] {
        OutputSpec::BufferMutation(bm) => {
            assert_eq!(bm.buffer_mutation.buffer_name, "running_mean");
        }
        other => panic!("expected BufferMutation, got {other:?}"),
    }
}

#[test]
fn test_mixed_input_specs_classification() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "p_w"}}, {"as_tensor": {"name": "b_rm"}}, {"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "x"}}], "nodes": [], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "p_w": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 1}]}, "b_rm": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}}}, "signature": {"input_specs": [{"parameter": {"arg": {"name": "p_w"}, "parameter_name": "weight"}}, {"buffer": {"arg": {"name": "b_rm"}, "buffer_name": "running_mean", "persistent": true}}, {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let specs = &program.graph_module.signature.input_specs;
    assert_eq!(specs.len(), 3);
    assert!(matches!(specs[0], InputSpec::Parameter(_)));
    assert!(matches!(specs[1], InputSpec::Buffer(_)));
    assert!(matches!(specs[2], InputSpec::UserInput(_)));
}

// ===========================================================================
// Section 5: Error handling for malformed graphs
// ===========================================================================

#[test]
fn test_error_on_invalid_json_bytes() {
    let result = parse_exported_program(b"not valid json at all");
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ImportError::JsonParse(_)));
}

#[test]
fn test_error_on_empty_json_object() {
    let result = parse_exported_program(b"{}");
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ImportError::JsonParse(_)));
}

#[test]
fn test_error_on_missing_graph_module() {
    let result = parse_exported_program(br#"{"schema_version": {"major": 8, "minor": 15}}"#);
    assert!(result.is_err());
}

#[test]
fn test_error_on_schema_version_too_old() {
    let json = r#"{"graph_module": {"graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}}, "signature": {"input_specs": [], "output_specs": []}, "module_call_graph": []}, "schema_version": {"major": 6, "minor": 0}, "range_constraints": {}}"#;
    let result = parse_exported_program(json.as_bytes());
    assert!(result.is_err());
    match result.unwrap_err() {
        ImportError::UnsupportedSchema { major, .. } => assert_eq!(major, 6),
        other => panic!("expected UnsupportedSchema, got {other:?}"),
    }
}

#[test]
fn test_error_on_schema_version_too_new() {
    let json = r#"{"graph_module": {"graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}}, "signature": {"input_specs": [], "output_specs": []}, "module_call_graph": []}, "schema_version": {"major": 10, "minor": 0}, "range_constraints": {}}"#;
    let result = parse_exported_program(json.as_bytes());
    assert!(result.is_err());
    match result.unwrap_err() {
        ImportError::UnsupportedSchema { major, .. } => assert_eq!(major, 10),
        other => panic!("expected UnsupportedSchema, got {other:?}"),
    }
}

#[test]
fn test_error_on_truncated_json() {
    let result = parse_exported_program(br#"{"graph_module": {"graph":"#);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ImportError::JsonParse(_)));
}

#[test]
fn test_error_on_null_bytes() {
    let result = parse_exported_program(b"\0\0\0");
    assert!(result.is_err());
}

#[test]
fn test_error_on_missing_signature() {
    let json = r#"{"graph_module": {"graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}}}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let result = parse_exported_program(json.as_bytes());
    assert!(result.is_err());
}

#[test]
fn test_unsupported_op_error_contains_target_name() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.unknown_future_op_42.default",
        vec![named("self", tensor_arg("x"))],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    match err {
        ImportError::UnsupportedOp { target } => {
            assert!(
                target.contains("unknown_future_op_42"),
                "error should contain op name, got: {target}"
            );
        }
        other => panic!("expected UnsupportedOp, got {other:?}"),
    }
}

#[test]
fn test_error_on_json_array_not_object() {
    let result = parse_exported_program(b"[1,2,3]");
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ImportError::JsonParse(_)));
}

#[test]
fn test_error_on_schema_version_1() {
    let json = r#"{"graph_module": {"graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}}, "signature": {"input_specs": [], "output_specs": []}, "module_call_graph": []}, "schema_version": {"major": 1, "minor": 0}, "range_constraints": {}}"#;
    let result = parse_exported_program(json.as_bytes());
    assert!(result.is_err());
    match result.unwrap_err() {
        ImportError::UnsupportedSchema { major, .. } => assert_eq!(major, 1),
        other => panic!("expected UnsupportedSchema, got {other:?}"),
    }
}

// ===========================================================================
// Section 6: Argument accessor edge cases
// ===========================================================================

#[test]
fn test_argument_as_int_on_float_returns_none() {
    let arg = float_arg(3.14);
    assert_eq!(arg.as_int(), None);
}

#[test]
fn test_argument_as_float_on_int_returns_none() {
    let arg = int_arg(42);
    assert_eq!(arg.as_float(), None);
}

#[test]
fn test_argument_as_tensor_name_on_int_returns_none() {
    let arg = int_arg(42);
    assert!(arg.as_tensor_name().is_none());
}

#[test]
fn test_argument_as_bool_val_on_string_returns_none() {
    let arg = str_arg("hello");
    assert_eq!(arg.as_bool_val(), None);
}

#[test]
fn test_argument_is_none_on_int_returns_false() {
    let arg = int_arg(0);
    assert!(!arg.is_none());
}

#[test]
fn test_argument_is_none_on_none_returns_true() {
    let arg = none_arg();
    assert!(arg.is_none());
}

#[test]
fn test_argument_as_string_on_bool_returns_none() {
    let arg = bool_arg(true);
    assert_eq!(arg.as_string(), None);
}

#[test]
fn test_argument_as_ints_on_single_int_returns_none() {
    let arg = int_arg(5);
    assert!(arg.as_ints().is_none());
}

#[test]
fn test_argument_as_tensor_names_on_single_tensor_returns_none() {
    let arg = tensor_arg("x");
    assert!(arg.as_tensor_names().is_none());
}

#[test]
fn test_argument_as_tensor_names_on_tensors_list() {
    let arg = Argument::Tensors(ArgumentTensors {
        as_tensors: vec![
            TensorArgument {
                name: "a".to_string(),
            },
            TensorArgument {
                name: "b".to_string(),
            },
            TensorArgument {
                name: "c".to_string(),
            },
        ],
    });
    let names = arg.as_tensor_names().unwrap();
    assert_eq!(names, vec!["a", "b", "c"]);
}

// ===========================================================================
// Section 7: SymInt / TensorMeta edge cases
// ===========================================================================

#[test]
fn test_sym_int_concrete_value() {
    let si = SymInt::Concrete(SymIntConcrete { as_int: 256 });
    assert_eq!(si.as_concrete(), Some(256));
}

#[test]
fn test_sym_int_negative_value() {
    let si = SymInt::Concrete(SymIntConcrete { as_int: -1 });
    assert_eq!(si.as_concrete(), Some(-1));
}

#[test]
fn test_sym_int_zero() {
    let si = SymInt::Concrete(SymIntConcrete { as_int: 0 });
    assert_eq!(si.as_concrete(), Some(0));
}

#[test]
fn test_tensor_meta_concrete_shape_with_large_dims() {
    let meta: TensorMeta = serde_json::from_str(
        r#"{"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 1024}, {"as_int": 768}], "requires_grad": false, "strides": [{"as_int": 786432}, {"as_int": 768}, {"as_int": 1}]}"#,
    ).unwrap();
    assert_eq!(meta.concrete_shape(), Some(vec![1, 1024, 768]));
}

#[test]
fn test_tensor_meta_mixed_symbolic_and_concrete_returns_none() {
    let json = r#"{"dtype": 7, "sizes": [{"as_int": 1}, {"as_expr": {"expr_str": "s0", "hint": null}}], "requires_grad": false, "strides": [{"as_int": 1}, {"as_int": 1}]}"#;
    let meta: TensorMeta = serde_json::from_str(json).unwrap();
    assert_eq!(meta.concrete_shape(), None);
}

// ===========================================================================
// Section 8: supported_ops consistency
// ===========================================================================

#[test]
fn test_supported_ops_is_non_empty() {
    let ops = supported_ops();
    assert!(!ops.is_empty());
}

#[test]
fn test_supported_ops_all_have_aten_prefix() {
    let ops = supported_ops();
    for op in &ops {
        assert!(
            op.starts_with("aten::"),
            "supported op should start with 'aten::': {op}"
        );
    }
}

#[test]
fn test_supported_ops_includes_creation_ops() {
    let ops = supported_ops();
    for expected in [
        "aten::zeros",
        "aten::ones",
        "aten::full",
        "aten::arange",
        "aten::zeros_like",
        "aten::ones_like",
        "aten::full_like",
    ] {
        assert!(ops.contains(&expected), "missing creation op: {expected}");
    }
}

#[test]
fn test_supported_ops_includes_norm_ops() {
    let ops = supported_ops();
    for expected in [
        "aten::layer_norm",
        "aten::group_norm",
        "aten::batch_norm",
        "aten::instance_norm",
    ] {
        assert!(
            ops.contains(&expected),
            "missing normalization op: {expected}"
        );
    }
}

#[test]
fn test_supported_ops_includes_pooling_ops() {
    let ops = supported_ops();
    for expected in [
        "aten::max_pool1d",
        "aten::avg_pool2d",
        "aten::adaptive_avg_pool2d",
    ] {
        assert!(ops.contains(&expected), "missing pooling op: {expected}");
    }
}

#[test]
fn test_supported_ops_count_minimum() {
    let ops = supported_ops();
    assert!(
        ops.len() >= 100,
        "expected >= 100 supported ops, got {}",
        ops.len()
    );
}

// ===========================================================================
// Section 9: ResolvedWeight construction
// ===========================================================================

#[test]
fn test_resolved_weight_new() {
    let w = ResolvedWeight::new(vec![1.0, 2.0, 3.0], vec![3]);
    assert_eq!(w.data, vec![1.0, 2.0, 3.0]);
    assert_eq!(w.shape, vec![3]);
}

#[test]
fn test_resolved_weight_2d() {
    let w = ResolvedWeight::new(vec![0.0; 12], vec![3, 4]);
    assert_eq!(w.shape, vec![3, 4]);
    assert_eq!(w.data.len(), 12);
}

#[test]
fn test_resolved_weight_empty() {
    let w = ResolvedWeight::new(vec![], vec![0]);
    assert!(w.data.is_empty());
    assert_eq!(w.shape, vec![0]);
}

// ===========================================================================
// Section 10: End-to-end build_graph with simple graphs
// ===========================================================================

#[test]
fn test_build_graph_simple_relu() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "relu_out"}}], "nodes": [{"target": "torch.ops.aten.relu.default", "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}], "outputs": [{"as_tensor": {"name": "relu_out"}}], "metadata": {}}], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]}, "relu_out": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]}}, "is_single_tensor_return": true}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "relu_out"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &weights).unwrap();
    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["relu_out"]);
    // graph should have 2 nodes: input + relu
    assert!(imported.graph.nodes().len() >= 2);
}

#[test]
fn test_build_graph_empty_nodes_identity() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "x"}}], "nodes": [], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}}, "is_single_tensor_return": true}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &weights).unwrap();
    assert_eq!(imported.num_user_inputs, 1);
    // With no nodes, only input is present
    assert_eq!(imported.graph.nodes().len(), 1);
}

#[test]
fn test_build_graph_chain_two_ops() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "x"}}], "outputs": [{"as_tensor": {"name": "z"}}], "nodes": [
        {"target": "torch.ops.aten.exp.default", "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}], "outputs": [{"as_tensor": {"name": "y"}}], "metadata": {}},
        {"target": "torch.ops.aten.log.default", "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "y"}}, "kind": 1}], "outputs": [{"as_tensor": {"name": "z"}}], "metadata": {}}
    ], "tensor_values": {"x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "y": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "z": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}}, "is_single_tensor_return": true}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "z"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &weights).unwrap();
    // input + exp + log = 3 nodes
    assert_eq!(imported.graph.nodes().len(), 3);
    assert_eq!(imported.output_names, vec!["z"]);
}

#[test]
fn test_build_graph_multiple_user_inputs() {
    let json = r#"{"graph_module": {"graph": {"inputs": [{"as_tensor": {"name": "a"}}, {"as_tensor": {"name": "b"}}], "outputs": [{"as_tensor": {"name": "c"}}], "nodes": [
        {"target": "torch.ops.aten.add.Tensor", "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "a"}}, "kind": 1}, {"name": "other", "arg": {"as_tensor": {"name": "b"}}, "kind": 1}], "outputs": [{"as_tensor": {"name": "c"}}], "metadata": {}}
    ], "tensor_values": {"a": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "b": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}, "c": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]}}, "is_single_tensor_return": true}, "signature": {"input_specs": [{"user_input": {"arg": {"as_tensor": {"name": "a"}}}}, {"user_input": {"arg": {"as_tensor": {"name": "b"}}}}], "output_specs": [{"user_output": {"arg": {"as_tensor": {"name": "c"}}}}]}, "module_call_graph": []}, "schema_version": {"major": 8, "minor": 15}, "range_constraints": {}}"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &weights).unwrap();
    assert_eq!(imported.num_user_inputs, 2);
    assert_eq!(imported.user_input_names, vec!["a", "b"]);
}
