// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for the op_map operator mapping infrastructure,
//! `supported_ops()` coverage, `OpMapContext`, quantization detection,
//! and parse validation.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;

use crate::error::ImportError;
use crate::op_map::{map_node_to_trace_op, supported_ops, OpMapContext, ResolvedWeight};
use crate::parse::{
    Argument, ArgumentFloat, ArgumentInt, ArgumentInts, ArgumentNone, ArgumentString,
    ArgumentTensor, NamedArgument, Node, TensorArgument, TensorMeta,
};
use crate::quantization::{detect_quantization_from_bytes, DetectedDtype};
use crate::{parse_exported_program, ImportError as IE};

// ---------------------------------------------------------------------------
// Helpers (mirrors op_map_tests but independent module)
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

#[allow(dead_code)]
fn float_arg(val: f64) -> Argument {
    Argument::Float(ArgumentFloat { as_float: val })
}

#[allow(dead_code)]
fn none_arg() -> Argument {
    Argument::None(ArgumentNone { as_none: true })
}

#[allow(dead_code)]
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

// =========================================================================
// supported_ops() coverage
// =========================================================================

#[test]
fn test_supported_ops_not_empty() {
    let ops = supported_ops();
    assert!(
        !ops.is_empty(),
        "supported_ops() must return a non-empty list"
    );
}

#[test]
fn test_supported_ops_includes_basic_ops() {
    let ops = supported_ops();
    for expected in &["aten::add", "aten::mul", "aten::relu", "aten::linear"] {
        assert!(
            ops.contains(expected),
            "supported_ops() should contain {expected}, got: {ops:?}"
        );
    }
}

#[test]
fn test_supported_ops_includes_conv() {
    let ops = supported_ops();
    assert!(
        ops.contains(&"aten::conv1d"),
        "supported_ops() should contain aten::conv1d"
    );
    assert!(
        ops.contains(&"aten::conv2d"),
        "supported_ops() should contain aten::conv2d"
    );
    assert!(
        ops.contains(&"aten::convolution"),
        "supported_ops() should contain aten::convolution"
    );
}

#[test]
fn test_supported_ops_includes_attention() {
    let ops = supported_ops();
    assert!(
        ops.contains(&"aten::softmax"),
        "supported_ops() should contain aten::softmax"
    );
    assert!(
        ops.contains(&"aten::scaled_dot_product_attention"),
        "supported_ops() should contain aten::scaled_dot_product_attention"
    );
}

#[test]
fn test_supported_ops_count() {
    let ops = supported_ops();
    assert!(
        ops.len() > 50,
        "expected >50 supported ops, got {}",
        ops.len()
    );
}

#[test]
fn test_supported_ops_sorted_and_deduplicated() {
    let ops = supported_ops();
    // Verify sorted
    for pair in ops.windows(2) {
        assert!(
            pair[0] <= pair[1],
            "supported_ops() should be sorted, but {0} > {1}",
            pair[0],
            pair[1]
        );
    }
    // Verify deduplicated
    for pair in ops.windows(2) {
        assert!(
            pair[0] != pair[1],
            "supported_ops() should be deduplicated, but found duplicate: {}",
            pair[0]
        );
    }
}

#[test]
fn test_supported_ops_includes_normalization() {
    let ops = supported_ops();
    for expected in &[
        "aten::layer_norm",
        "aten::group_norm",
        "aten::batch_norm",
        "aten::instance_norm",
    ] {
        assert!(
            ops.contains(expected),
            "supported_ops() should contain {expected}"
        );
    }
}

#[test]
fn test_supported_ops_includes_shape_ops() {
    let ops = supported_ops();
    for expected in &[
        "aten::view",
        "aten::reshape",
        "aten::transpose",
        "aten::permute",
        "aten::squeeze",
        "aten::unsqueeze",
        "aten::cat",
        "aten::flatten",
    ] {
        assert!(
            ops.contains(expected),
            "supported_ops() should contain {expected}"
        );
    }
}

#[test]
fn test_supported_ops_includes_reductions() {
    let ops = supported_ops();
    for expected in &["aten::sum", "aten::mean", "aten::amax", "aten::amin"] {
        assert!(
            ops.contains(expected),
            "supported_ops() should contain {expected}"
        );
    }
}

#[test]
fn test_supported_ops_includes_embedding() {
    let ops = supported_ops();
    assert!(
        ops.contains(&"aten::embedding"),
        "supported_ops() should contain aten::embedding"
    );
}

// =========================================================================
// OpMapContext
// =========================================================================

#[test]
fn test_op_map_context_creation() {
    let meta: HashMap<String, TensorMeta> = HashMap::new();
    let weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let ctx = OpMapContext {
        tensor_meta: &meta,
        weights: &weights,
    };
    assert!(ctx.tensor_meta.is_empty());
    assert!(ctx.weights.is_empty());
}

#[test]
fn test_op_map_context_default() {
    let ctx = empty_ctx();
    assert!(
        ctx.tensor_meta.is_empty(),
        "default context should have empty tensor_meta"
    );
    assert!(
        ctx.weights.is_empty(),
        "default context should have empty weights"
    );
}

#[test]
fn test_op_map_context_with_weights() {
    let meta: HashMap<String, TensorMeta> = HashMap::new();
    let mut weights: HashMap<String, ResolvedWeight> = HashMap::new();
    weights.insert(
        "layer.weight".to_string(),
        ResolvedWeight::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
    );
    let ctx = OpMapContext {
        tensor_meta: &meta,
        weights: &weights,
    };
    assert_eq!(ctx.weights.len(), 1);
    let w = ctx.weights.get("layer.weight").unwrap();
    assert_eq!(w.data.len(), 4);
    assert_eq!(w.shape, vec![2, 2]);
}

// =========================================================================
// ResolvedWeight
// =========================================================================

#[test]
fn test_resolved_weight_new() {
    let w = ResolvedWeight::new(vec![0.5, 1.5], vec![1, 2]);
    assert_eq!(w.data, vec![0.5, 1.5]);
    assert_eq!(w.shape, vec![1, 2]);
}

#[test]
fn test_resolved_weight_debug() {
    let w = ResolvedWeight::new(vec![1.0], vec![1]);
    let debug = format!("{w:?}");
    assert!(debug.contains("ResolvedWeight"));
}

#[test]
fn test_resolved_weight_clone() {
    let w1 = ResolvedWeight::new(vec![3.14], vec![1]);
    let w2 = w1.clone();
    assert_eq!(w1.data, w2.data);
    assert_eq!(w1.shape, w2.shape);
}

// =========================================================================
// map_node_to_trace_op: basic mapping tests
// =========================================================================

#[test]
fn test_map_sigmoid() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sigmoid.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Sigmoid));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_tanh() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.tanh.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Tanh));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_exp() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.exp.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Exp));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_log() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.log.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Log));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_mul() {
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
fn test_map_div() {
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
fn test_map_silu() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.silu.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Silu));
}

#[test]
fn test_map_unsupported_returns_error() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.totally_fake_op.default",
        vec![named("input", tensor_arg("x"))],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::UnsupportedOp { ref target } if target.contains("totally_fake_op")),
        "expected UnsupportedOp, got: {err:?}"
    );
}

#[test]
fn test_map_unsqueeze() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.unsqueeze.default",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(0))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Unsqueeze { dim: 0 }),
        "expected Unsqueeze dim=0, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_permute() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.permute.default",
        vec![
            named("self", tensor_arg("x")),
            named("dims", ints_arg(&[2, 0, 1])),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Permute { ref axes } if axes == &[2, 0, 1]),
        "expected Permute [2,0,1], got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_flatten_needs_shape_metadata() {
    // flatten.using_ints requires input shape metadata for Reshape decomposition.
    // Without shape metadata it returns UnsupportedOp.
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.flatten.using_ints",
        vec![
            named("self", tensor_arg("x")),
            named("start_dim", int_arg(1)),
            named("end_dim", int_arg(2)),
        ],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::UnsupportedOp { .. }),
        "flatten without shape metadata should return UnsupportedOp, got: {err:?}"
    );
}

// =========================================================================
// Quantization detection
// =========================================================================

fn build_safetensors_typed(tensors: &[(&str, &[usize], &[u8], safetensors::Dtype)]) -> Vec<u8> {
    let mut tensor_map: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
    for &(name, shape, data, dtype) in tensors {
        let view = safetensors::tensor::TensorView::new(dtype, shape.to_vec(), data)
            .expect("valid tensor view");
        tensor_map.push((name.to_string(), view));
    }
    safetensors::tensor::serialize(tensor_map, None).expect("serialization should succeed")
}

#[test]
fn test_detect_quantization_empty() {
    let bytes = build_safetensors_typed(&[]);
    let report = detect_quantization_from_bytes(&bytes).expect("should parse empty model");
    assert_eq!(report.total_tensors, 0);
    assert_eq!(report.total_parameters, 0);
    assert_eq!(report.total_bytes, 0);
    assert!(report.dtype_breakdown.is_empty());
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_quantization_report_structure() {
    // Build a minimal F32 model to exercise report fields
    let w_data: Vec<u8> = vec![0u8; 16]; // 4 elements * 4 bytes
    let bytes = build_safetensors_typed(&[("weight", &[2, 2], &w_data, safetensors::Dtype::F32)]);
    let report = detect_quantization_from_bytes(&bytes).expect("should parse");

    // Verify all public fields are populated correctly
    assert_eq!(report.total_tensors, 1);
    assert_eq!(report.total_parameters, 4);
    assert_eq!(report.total_bytes, 16);
    assert_eq!(report.dtype_breakdown.len(), 1);
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::F32);
    assert_eq!(report.dtype_breakdown[0].tensor_count, 1);
    assert_eq!(report.dtype_breakdown[0].total_parameters, 4);
    assert_eq!(report.dtype_breakdown[0].total_bytes, 16);

    // Verify helper methods
    assert!(!report.is_mixed_precision());
    let frac = report.dtype_fraction(DetectedDtype::F32);
    assert!((frac - 1.0).abs() < 1e-10, "F32 fraction should be 1.0");
    assert!(
        report.dtype_fraction(DetectedDtype::BF16).abs() < 1e-10,
        "BF16 fraction should be 0.0"
    );
}

#[test]
fn test_quantization_report_mixed_precision() {
    // F32 weight + BF16 weight
    let f32_data: Vec<u8> = vec![0u8; 16]; // 4 elements * 4 bytes
    let bf16_data: Vec<u8> = vec![0u8; 8]; // 4 elements * 2 bytes
    let bytes = build_safetensors_typed(&[
        ("layer.weight", &[2, 2], &f32_data, safetensors::Dtype::F32),
        ("layer.bias", &[2, 2], &bf16_data, safetensors::Dtype::BF16),
    ]);
    let report = detect_quantization_from_bytes(&bytes).expect("should parse");

    assert_eq!(report.total_tensors, 2);
    assert!(report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown.len(), 2);

    // Summary should produce a non-empty string
    let summary = report.summary();
    assert!(!summary.is_empty());
    assert!(summary.contains("F32"));
}

#[test]
fn test_quantization_report_total_savings() {
    // Empty model => 0 savings
    let bytes = build_safetensors_typed(&[]);
    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    assert_eq!(report.total_savings_bytes(), 0);
}

#[test]
fn test_quantization_report_tensors_field() {
    let f32_data: Vec<u8> = vec![0u8; 32]; // 8 elements * 4 bytes
    let bytes = build_safetensors_typed(&[(
        "encoder.weight",
        &[2, 4],
        &f32_data,
        safetensors::Dtype::F32,
    )]);
    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    assert_eq!(report.tensors.len(), 1);
    assert_eq!(report.tensors[0].name, "encoder.weight");
    assert_eq!(report.tensors[0].dtype, DetectedDtype::F32);
    assert_eq!(report.tensors[0].shape, vec![2, 4]);
    assert_eq!(report.tensors[0].num_elements, 8);
    assert_eq!(report.tensors[0].size_bytes, 32);
}

// =========================================================================
// Parse validation
// =========================================================================

#[test]
fn test_parse_empty_json_fails() {
    let result = parse_exported_program(b"");
    assert!(result.is_err(), "empty bytes should fail to parse");
    assert!(
        matches!(result.unwrap_err(), IE::JsonParse(_)),
        "expected JsonParse error for empty input"
    );
}

#[test]
fn test_parse_invalid_json_fails() {
    let result = parse_exported_program(b"not json at all");
    assert!(result.is_err(), "invalid JSON should fail to parse");
    assert!(matches!(result.unwrap_err(), IE::JsonParse(_)));
}

#[test]
fn test_parse_minimal_program() {
    // Minimal valid structure with schema_version.major = 8
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [],
                "outputs": [],
                "nodes": [],
                "tensor_values": {}
            },
            "signature": {
                "input_specs": [],
                "output_specs": []
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 0},
        "range_constraints": {}
    }"#;
    let program =
        parse_exported_program(json.as_bytes()).expect("minimal valid program should parse");
    assert_eq!(program.schema_version.major, 8);
    assert_eq!(program.schema_version.minor, 0);
    assert!(program.graph_module.graph.nodes.is_empty());
    assert!(program.graph_module.signature.input_specs.is_empty());
    assert!(program.graph_module.signature.output_specs.is_empty());
}

#[test]
fn test_parse_wrong_schema_version_fails() {
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [],
                "outputs": [],
                "nodes": [],
                "tensor_values": {}
            },
            "signature": {
                "input_specs": [],
                "output_specs": []
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 5, "minor": 3},
        "range_constraints": {}
    }"#;
    let err = parse_exported_program(json.as_bytes()).unwrap_err();
    assert!(
        matches!(err, IE::UnsupportedSchema { major: 5, minor: 3 }),
        "expected UnsupportedSchema(5, 3), got: {err:?}"
    );
}

#[test]
fn test_parse_missing_graph_module_fails() {
    let json = r#"{"schema_version": {"major": 8, "minor": 0}}"#;
    let result = parse_exported_program(json.as_bytes());
    assert!(result.is_err(), "missing graph_module should fail");
}

// =========================================================================
// Edge cases for map_node_to_trace_op
// =========================================================================

#[test]
fn test_map_mm_matmul() {
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
fn test_map_bmm_matmul() {
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
fn test_map_reshape_view_alias() {
    // aten::_unsafe_view is an alias for view/reshape
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten._unsafe_view.default",
        vec![
            named("self", tensor_arg("x")),
            named("size", ints_arg(&[6, 8])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Reshape { ref target_shape } if target_shape == &[6, 8]),
        "expected Reshape [6, 8], got: {op:?}"
    );
}

#[test]
fn test_map_abs() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.abs.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Abs));
}

#[test]
fn test_map_neg() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.neg.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Neg));
}

#[test]
fn test_map_sqrt() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sqrt.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Sqrt));
}

#[test]
fn test_map_rsqrt_to_powf() {
    // rsqrt maps to Powf { exponent: -0.5 }
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.rsqrt.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Powf { exponent } if (exponent - (-0.5)).abs() < 1e-10),
        "expected Powf(-0.5) for rsqrt, got: {op:?}"
    );
}

#[test]
fn test_map_sin_cos() {
    let ctx = empty_ctx();
    for (target, expected_variant) in [
        ("torch.ops.aten.sin.default", "Sin"),
        ("torch.ops.aten.cos.default", "Cos"),
    ] {
        let node = simple_node(target, vec![named("input", tensor_arg("x"))]);
        let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        let variant_name = format!("{op:?}");
        assert!(
            variant_name.contains(expected_variant),
            "expected {expected_variant} for {target}, got: {variant_name}"
        );
    }
}

#[test]
fn test_map_floor_round() {
    let ctx = empty_ctx();
    for (target, expected) in [
        ("torch.ops.aten.floor.default", "Floor"),
        ("torch.ops.aten.round.default", "Round"),
    ] {
        let node = simple_node(target, vec![named("input", tensor_arg("x"))]);
        let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        let variant_name = format!("{op:?}");
        assert!(
            variant_name.contains(expected),
            "expected {expected} for {target}, got: {variant_name}"
        );
    }
}

#[test]
fn test_map_maximum_minimum() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.maximum.default",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Maximum));

    let node = simple_node(
        "torch.ops.aten.minimum.default",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Minimum));
}

#[test]
fn test_map_log_softmax() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.log_softmax.int",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(2))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::LogSoftmax { dim: 2 }),
        "expected LogSoftmax dim=2, got: {op:?}"
    );
}

#[test]
fn test_map_contiguous_is_identity() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.contiguous.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    // contiguous/clone should produce an identity-like op
    assert_eq!(inputs, vec!["x"]);
    // The exact TraceOp variant may be Identity/Contiguous/Clone — just verify success
    let _ = op;
}

#[test]
fn test_map_clone_succeeds() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.clone.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok(), "clone should map successfully");
    assert_eq!(result.unwrap().1, vec!["x"]);
}

#[test]
fn test_map_dropout_is_identity() {
    // During inference, dropout is a no-op / identity
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.dropout.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (_, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert_eq!(inputs, vec!["x"]);
}

// =========================================================================
// DetectedDtype helper methods
// =========================================================================

#[test]
fn test_detected_dtype_bytes_per_element() {
    assert_eq!(DetectedDtype::F32.bytes_per_element(), Some(4));
    assert_eq!(DetectedDtype::F16.bytes_per_element(), Some(2));
    assert_eq!(DetectedDtype::BF16.bytes_per_element(), Some(2));
    assert_eq!(DetectedDtype::F64.bytes_per_element(), Some(8));
    assert_eq!(DetectedDtype::I8.bytes_per_element(), Some(1));
    assert_eq!(DetectedDtype::U8.bytes_per_element(), Some(1));
    assert_eq!(DetectedDtype::Bool.bytes_per_element(), Some(1));
    assert_eq!(DetectedDtype::SubByte.bytes_per_element(), None);
    assert_eq!(DetectedDtype::Other.bytes_per_element(), None);
}

#[test]
fn test_detected_dtype_label() {
    assert_eq!(DetectedDtype::F32.label(), "F32");
    assert_eq!(DetectedDtype::BF16.label(), "BF16");
    assert_eq!(DetectedDtype::I8.label(), "I8");
    assert_eq!(DetectedDtype::Other.label(), "Other");
}

#[test]
fn test_detected_dtype_display() {
    let s = format!("{}", DetectedDtype::F16);
    assert_eq!(s, "F16");
}
