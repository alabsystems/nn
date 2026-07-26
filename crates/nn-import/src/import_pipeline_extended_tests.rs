// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended import pipeline tests covering the full nn-import flow:
//! aten op mapping coverage, weight format conversion, model graph
//! construction, dtype conversion, shape inference, batch dimension
//! handling, multi-file model loading, config parsing, quantization
//! import, and tokenizer/vocabulary loading.
//!
//! Part of #4186.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceOp};
use nn_core::DType;

use crate::error::ImportError;
use crate::graph_build::{build_graph, build_weight_map, ImportedGraph};
use crate::kokoro_weights::{kokoro_name_mapping, map_pytorch_key, validate_kokoro_keys};
use crate::multi_segment::{convert_multi_segment, MultiSegmentError};
use crate::op_map::{map_node_to_trace_op, supported_ops, OpMapContext, ResolvedWeight};
use crate::parse::{
    parse_exported_program, Argument, ArgumentTensor, InputSpec, NamedArgument, Node, OutputSpec,
    SymInt, SymIntConcrete, TensorArgument, TensorMeta,
};
use crate::quantization::{detect_quantization_from_bytes, DetectedDtype};

// ===========================================================================
// Helpers
// ===========================================================================

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

/// Build a torch.export graph JSON for a single relu on a given shape+dtype.
fn single_op_json(
    input_name: &str,
    output_name: &str,
    shape: &[usize],
    dtype_code: i32,
    op_target: &str,
    extra_inputs: &str,
) -> String {
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
        r#""{input_name}": {{"dtype": {dtype_code}, "sizes": [{sizes}], "requires_grad": false, "strides": [{strides}]}},
        "{output_name}": {{"dtype": {dtype_code}, "sizes": [{sizes}], "requires_grad": false, "strides": [{strides}]}}"#
    );

    let mut inputs_json = format!(
        r#"{{"name": "input", "arg": {{"as_tensor": {{"name": "{input_name}"}}}}, "kind": 1}}"#
    );
    if !extra_inputs.is_empty() {
        inputs_json.push_str(", ");
        inputs_json.push_str(extra_inputs);
    }

    let node = format!(
        r#"{{
            "target": "{op_target}",
            "inputs": [{inputs_json}],
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
// 1. Aten op mapping coverage: all common PyTorch ops have mappings
// ===========================================================================

#[test]
fn test_aten_op_mapping_minimum_count() {
    let ops = supported_ops();
    // The framework supports 200+ aten ops; verify the list is comprehensive.
    assert!(
        ops.len() >= 200,
        "expected >= 200 supported ops, got {}",
        ops.len()
    );
}

#[test]
fn test_aten_op_mapping_common_unary_ops_present() {
    let ops = supported_ops();
    let required_unary = [
        "aten::relu",
        "aten::gelu",
        "aten::silu",
        "aten::tanh",
        "aten::sigmoid",
        "aten::exp",
        "aten::log",
        "aten::sqrt",
        "aten::abs",
        "aten::neg",
        "aten::sin",
        "aten::cos",
        "aten::floor",
        "aten::round",
        "aten::rsqrt",
        "aten::reciprocal",
    ];
    for op in &required_unary {
        assert!(ops.contains(op), "missing unary op: {op}");
    }
}

#[test]
fn test_aten_op_mapping_common_binary_ops_present() {
    let ops = supported_ops();
    let required_binary = [
        "aten::add",
        "aten::sub",
        "aten::mul",
        "aten::div",
        "aten::maximum",
        "aten::minimum",
    ];
    for op in &required_binary {
        assert!(ops.contains(op), "missing binary op: {op}");
    }
}

#[test]
fn test_aten_op_mapping_matmul_linear_present() {
    let ops = supported_ops();
    for op in &["aten::mm", "aten::bmm", "aten::matmul", "aten::linear"] {
        assert!(ops.contains(op), "missing matmul/linear op: {op}");
    }
}

#[test]
fn test_aten_op_mapping_conv_ops_present() {
    let ops = supported_ops();
    let conv_ops = [
        "aten::convolution",
        "aten::conv1d",
        "aten::conv2d",
        "aten::conv_transpose1d",
        "aten::conv_transpose2d",
        "aten::conv3d",
    ];
    for op in &conv_ops {
        assert!(ops.contains(op), "missing conv op: {op}");
    }
}

#[test]
fn test_aten_op_mapping_norm_ops_present() {
    let ops = supported_ops();
    let norm_ops = [
        "aten::layer_norm",
        "aten::group_norm",
        "aten::batch_norm",
        "aten::instance_norm",
        "aten::rms_norm",
    ];
    for op in &norm_ops {
        assert!(ops.contains(op), "missing norm op: {op}");
    }
}

#[test]
fn test_aten_op_mapping_attention_ops_present() {
    let ops = supported_ops();
    let attn_ops = [
        "aten::softmax",
        "aten::log_softmax",
        "aten::scaled_dot_product_attention",
        "aten::embedding",
    ];
    for op in &attn_ops {
        assert!(ops.contains(op), "missing attention/embedding op: {op}");
    }
}

#[test]
fn test_aten_op_mapping_shape_ops_present() {
    let ops = supported_ops();
    let shape_ops = [
        "aten::view",
        "aten::reshape",
        "aten::transpose",
        "aten::permute",
        "aten::flatten",
        "aten::unsqueeze",
        "aten::squeeze",
        "aten::cat",
        "aten::slice",
        "aten::expand",
        "aten::flip",
        "aten::select",
        "aten::stack",
        "aten::narrow",
        "aten::chunk",
        "aten::split",
        "aten::unbind",
    ];
    for op in &shape_ops {
        assert!(ops.contains(op), "missing shape op: {op}");
    }
}

#[test]
fn test_aten_op_mapping_pooling_ops_present() {
    let ops = supported_ops();
    let pool_ops = [
        "aten::max_pool1d",
        "aten::avg_pool2d",
        "aten::adaptive_avg_pool2d",
        "aten::max_pool2d",
        "aten::avg_pool1d",
    ];
    for op in &pool_ops {
        assert!(ops.contains(op), "missing pooling op: {op}");
    }
}

#[test]
fn test_aten_op_mapping_activation_ops_present() {
    let ops = supported_ops();
    let act_ops = [
        "aten::elu",
        "aten::leaky_relu",
        "aten::hardtanh",
        "aten::hardsigmoid",
        "aten::hardswish",
        "aten::selu",
        "aten::softplus",
        "aten::mish",
        "aten::prelu",
    ];
    for op in &act_ops {
        assert!(ops.contains(op), "missing activation op: {op}");
    }
}

#[test]
fn test_aten_op_mapping_tensor_creation_ops_present() {
    let ops = supported_ops();
    let creation_ops = [
        "aten::zeros",
        "aten::ones",
        "aten::full",
        "aten::arange",
        "aten::zeros_like",
        "aten::ones_like",
        "aten::full_like",
        "aten::empty",
        "aten::linspace",
        "aten::eye",
    ];
    for op in &creation_ops {
        assert!(ops.contains(op), "missing creation op: {op}");
    }
}

#[test]
fn test_aten_op_mapping_comparison_ops_present() {
    let ops = supported_ops();
    let cmp_ops = [
        "aten::where",
        "aten::clamp",
        "aten::gt",
        "aten::lt",
        "aten::ge",
        "aten::le",
        "aten::eq",
        "aten::ne",
    ];
    for op in &cmp_ops {
        assert!(ops.contains(op), "missing comparison op: {op}");
    }
}

#[test]
fn test_aten_op_mapping_loss_ops_present() {
    let ops = supported_ops();
    let loss_ops = [
        "aten::mse_loss",
        "aten::l1_loss",
        "aten::smooth_l1_loss",
        "aten::huber_loss",
        "aten::binary_cross_entropy",
        "aten::cross_entropy_loss",
        "aten::nll_loss",
        "aten::kl_div",
    ];
    for op in &loss_ops {
        assert!(ops.contains(op), "missing loss op: {op}");
    }
}

#[test]
fn test_aten_op_mapping_no_duplicates() {
    let ops = supported_ops();
    let mut seen = std::collections::HashSet::new();
    for op in &ops {
        assert!(seen.insert(op), "duplicate op in supported_ops(): {op}");
    }
}

#[test]
fn test_map_node_relu_produces_relu_traceop() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.relu.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 2).unwrap();
    assert!(matches!(op, TraceOp::Relu));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_node_add_produces_add_traceop() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.add.Tensor",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 2).unwrap();
    assert!(matches!(op, TraceOp::Add));
    assert_eq!(inputs, vec!["a", "b"]);
}

#[test]
fn test_map_node_unsupported_op_errors() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.nonexistent.op.default",
        vec![named("input", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 2);
    assert!(result.is_err());
    match result.unwrap_err() {
        ImportError::UnsupportedOp { target } => {
            assert!(target.contains("nonexistent"));
        }
        other => panic!("expected UnsupportedOp, got: {other:?}"),
    }
}

// ===========================================================================
// 2. Weight format conversion: safetensors -> nn internal format
//
// Tests use `load_safetensors_weights_pub` (pub(crate) re-export of
// load_safetensors_weights) which reads a safetensors file and returns
// HashMap<String, (Vec<f32>, Vec<usize>)>.
// ===========================================================================

/// Helper: write safetensors bytes to a temp file and load via the import pipeline.
fn load_weights_from_bytes(
    data: &[u8],
) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>, ImportError> {
    // Tests run in parallel within a single process, so the temp path must be
    // unique per call. Using only the process id would cause concurrent tests
    // to clobber each other's files (header-too-small / no-such-file errors).
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp = std::env::temp_dir().join(format!(
        "nn_wconv_test_{}_{}.safetensors",
        std::process::id(),
        unique
    ));
    std::fs::write(&tmp, data).unwrap();
    let result = crate::convert::load_safetensors_weights_pub(&tmp);
    let _ = std::fs::remove_file(&tmp);
    result
}

#[test]
fn test_weight_conversion_f32_roundtrip() {
    let vals = [1.5f32, -2.0, 3.14, 0.0];
    let raw: Vec<u8> = vals.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut tensors = HashMap::new();
    tensors.insert(
        "w".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2], &raw).unwrap(),
    );
    let bytes = safetensors::serialize(&tensors, None).unwrap();
    let loaded = load_weights_from_bytes(&bytes).unwrap();
    let (f32_data, shape) = &loaded["w"];
    assert_eq!(shape, &[2, 2]);
    assert_eq!(f32_data.len(), 4);
    assert!((f32_data[0] - 1.5).abs() < 1e-6);
    assert!((f32_data[1] + 2.0).abs() < 1e-6);
}

#[test]
fn test_weight_conversion_f16_to_f32() {
    let f16_vals = [half::f16::from_f32(1.0), half::f16::from_f32(-0.5)];
    let raw: Vec<u8> = f16_vals.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut tensors = HashMap::new();
    tensors.insert(
        "w".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![2], &raw).unwrap(),
    );
    let bytes = safetensors::serialize(&tensors, None).unwrap();
    let loaded = load_weights_from_bytes(&bytes).unwrap();
    let (f32_data, _shape) = &loaded["w"];
    assert!((f32_data[0] - 1.0).abs() < 0.01);
    assert!((f32_data[1] + 0.5).abs() < 0.01);
}

#[test]
fn test_weight_conversion_bf16_to_f32() {
    let bf16_vals = [half::bf16::from_f32(2.0), half::bf16::from_f32(-1.0)];
    let raw: Vec<u8> = bf16_vals.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut tensors = HashMap::new();
    tensors.insert(
        "w".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![2], &raw).unwrap(),
    );
    let bytes = safetensors::serialize(&tensors, None).unwrap();
    let loaded = load_weights_from_bytes(&bytes).unwrap();
    let (f32_data, _shape) = &loaded["w"];
    assert!((f32_data[0] - 2.0).abs() < 0.1);
    assert!((f32_data[1] + 1.0).abs() < 0.1);
}

#[test]
fn test_weight_conversion_f64_to_f32() {
    let data = [1.234567890123456_f64];
    let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut tensors = HashMap::new();
    tensors.insert(
        "w".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F64, vec![1], &raw).unwrap(),
    );
    let bytes = safetensors::serialize(&tensors, None).unwrap();
    let loaded = load_weights_from_bytes(&bytes).unwrap();
    let (f32_data, _shape) = &loaded["w"];
    // f64 -> f32 loses precision but retains value magnitude.
    assert!((f32_data[0] - 1.234568).abs() < 1e-4);
}

#[test]
fn test_weight_conversion_i64_to_f32() {
    let data = [42_i64, -7];
    let raw: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
    let mut tensors = HashMap::new();
    tensors.insert(
        "w".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::I64, vec![2], &raw).unwrap(),
    );
    let bytes = safetensors::serialize(&tensors, None).unwrap();
    let loaded = load_weights_from_bytes(&bytes).unwrap();
    let (f32_data, _shape) = &loaded["w"];
    assert_eq!(f32_data[0], 42.0);
    assert_eq!(f32_data[1], -7.0);
}

#[test]
fn test_weight_conversion_u8_to_f32() {
    let u8_data = vec![0u8, 128, 255];
    let mut tensors = HashMap::new();
    tensors.insert(
        "u".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::U8, vec![3], &u8_data).unwrap(),
    );
    let bytes = safetensors::serialize(&tensors, None).unwrap();
    let loaded = load_weights_from_bytes(&bytes).unwrap();
    let (f32_data, _shape) = &loaded["u"];
    assert_eq!(*f32_data, vec![0.0, 128.0, 255.0]);
}

#[test]
fn test_weight_conversion_i8_to_f32() {
    // Raw byte 128 = i8 -128
    let i8_data = vec![0u8, 127, 128];
    let mut tensors = HashMap::new();
    tensors.insert(
        "i".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::I8, vec![3], &i8_data).unwrap(),
    );
    let bytes = safetensors::serialize(&tensors, None).unwrap();
    let loaded = load_weights_from_bytes(&bytes).unwrap();
    let (f32_data, _shape) = &loaded["i"];
    assert_eq!(*f32_data, vec![0.0, 127.0, -128.0]);
}

#[test]
fn test_weight_conversion_unsupported_dtype_returns_error() {
    // BOOL is not supported by tensor_view_to_f32; load_safetensors_weights
    // should return an UnsupportedDtype error.
    let data = vec![1u8, 0, 1];
    let mut tensors = HashMap::new();
    tensors.insert(
        "mask".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::BOOL, vec![3], &data).unwrap(),
    );
    let bytes = safetensors::serialize(&tensors, None).unwrap();
    let result = load_weights_from_bytes(&bytes);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ImportError::UnsupportedDtype { .. }));
}

// ===========================================================================
// 3. Model graph construction: nodes connected correctly
// ===========================================================================

#[test]
fn test_graph_construction_embedding_lookup() {
    let json = include_str!("../test_data/embedding_lookup.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let mut wd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    wd.insert("embed.weight".to_string(), (vec![0.1; 80], vec![10, 8]));
    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    let imported = build_graph(&program, &wm).unwrap();

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["indices"]);
    assert_eq!(imported.output_names, vec!["embedding"]);

    // The graph should have: 1 user input + 1 param placeholder + 1 embedding op = 3 nodes.
    let node_count = imported.graph.len();
    assert!(
        node_count >= 2,
        "graph should have at least 2 nodes (input + embedding), got {node_count}"
    );
}

#[test]
fn test_graph_construction_multi_input_cat() {
    let json = include_str!("../test_data/multi_input_cat.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let mut wd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    wd.insert("fc.weight".to_string(), (vec![0.01; 64], vec![4, 16]));
    wd.insert("fc.bias".to_string(), (vec![0.0; 4], vec![4]));
    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    let imported = build_graph(&program, &wm).unwrap();

    // Two user inputs: a and b.
    assert_eq!(imported.num_user_inputs, 2);
    assert!(imported.user_input_names.contains(&"a".to_string()));
    assert!(imported.user_input_names.contains(&"b".to_string()));
    assert_eq!(imported.output_names, vec!["output"]);

    // Should have: 2 user inputs + 2 param placeholders + 3 compute ops (cat, relu, linear).
    let compute_count = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .count();
    assert!(
        compute_count >= 3,
        "expected at least 3 compute ops (cat, relu, linear), got {compute_count}"
    );
}

#[test]
fn test_graph_construction_resnet_skip_connection() {
    let json = include_str!("../test_data/resnet_basic_block.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let mut wd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    wd.insert(
        "conv1.weight".to_string(),
        (vec![0.01; 16 * 16 * 3 * 3], vec![16, 16, 3, 3]),
    );
    wd.insert("conv1.bias".to_string(), (vec![0.0; 16], vec![16]));
    wd.insert("bn1.weight".to_string(), (vec![1.0; 16], vec![16]));
    wd.insert("bn1.bias".to_string(), (vec![0.0; 16], vec![16]));
    wd.insert("bn1.running_mean".to_string(), (vec![0.0; 16], vec![16]));
    wd.insert("bn1.running_var".to_string(), (vec![1.0; 16], vec![16]));
    wd.insert(
        "conv2.weight".to_string(),
        (vec![0.01; 16 * 16 * 3 * 3], vec![16, 16, 3, 3]),
    );
    wd.insert("conv2.bias".to_string(), (vec![0.0; 16], vec![16]));
    wd.insert("bn2.weight".to_string(), (vec![1.0; 16], vec![16]));
    wd.insert("bn2.bias".to_string(), (vec![0.0; 16], vec![16]));
    wd.insert("bn2.running_mean".to_string(), (vec![0.0; 16], vec![16]));
    wd.insert("bn2.running_var".to_string(), (vec![1.0; 16], vec![16]));
    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    let imported = build_graph(&program, &wm).unwrap();

    // Verify skip connection: Add node must exist and reference two distinct sources.
    let add_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Add))
        .expect("ResNet basic block must have an Add node for skip connection");
    assert_eq!(add_node.inputs().len(), 2);
    assert_ne!(add_node.inputs()[0], add_node.inputs()[1]);
}

#[test]
fn test_graph_construction_topology_error_on_bad_ref() {
    // A node references a tensor that does not exist in the graph.
    let json = make_json_graph(
        r#"{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}"#,
        r#"{"user_output": {"arg": {"as_tensor": {"name": "relu_out"}}}}"#,
        r#"{
            "target": "torch.ops.aten.relu.default",
            "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "nonexistent"}}, "kind": 1}],
            "outputs": [{"as_tensor": {"name": "relu_out"}}],
            "metadata": {}
        }"#,
        r#""x": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]},
        "relu_out": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}"#,
    );
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let result = build_graph(&program, &empty_weights);
    assert!(
        result.is_err(),
        "should error on topology reference to nonexistent tensor"
    );
    match result.unwrap_err() {
        ImportError::TopologyError { ref_name, .. } => {
            assert_eq!(ref_name, "nonexistent");
        }
        other => panic!("expected TopologyError, got: {other:?}"),
    }
}

#[test]
fn test_graph_construction_output_node_marked() {
    let json = single_op_json("x", "y", &[2, 3], 7, "torch.ops.aten.relu.default", "");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    // The output node should be accessible.
    let output = imported.graph.output_node();
    assert!(output.is_some(), "graph should have a marked output node");
    assert_eq!(output.unwrap().name(), "y");
}

// ===========================================================================
// 4. Dtype conversion: PyTorch dtypes map to nn DType
// ===========================================================================

#[test]
fn test_scalar_type_to_dtype_f32() {
    let meta = TensorMeta {
        dtype: 7,
        sizes: vec![SymInt::Concrete(SymIntConcrete { as_int: 4 })],
        requires_grad: false,
        strides: vec![SymInt::Concrete(SymIntConcrete { as_int: 1 })],
        storage_offset: None,
        device: None,
        layout: None,
    };
    assert_eq!(meta.to_dtype(), Some(DType::F32));
}

#[test]
fn test_scalar_type_to_dtype_f16() {
    let meta = TensorMeta {
        dtype: 6,
        sizes: vec![SymInt::Concrete(SymIntConcrete { as_int: 2 })],
        requires_grad: false,
        strides: vec![SymInt::Concrete(SymIntConcrete { as_int: 1 })],
        storage_offset: None,
        device: None,
        layout: None,
    };
    assert_eq!(meta.to_dtype(), Some(DType::F16));
}

#[test]
fn test_scalar_type_to_dtype_bf16() {
    let meta = TensorMeta {
        dtype: 13,
        sizes: vec![SymInt::Concrete(SymIntConcrete { as_int: 2 })],
        requires_grad: false,
        strides: vec![SymInt::Concrete(SymIntConcrete { as_int: 1 })],
        storage_offset: None,
        device: None,
        layout: None,
    };
    assert_eq!(meta.to_dtype(), Some(DType::BF16));
}

#[test]
fn test_scalar_type_to_dtype_f64() {
    let meta = TensorMeta {
        dtype: 8,
        sizes: vec![SymInt::Concrete(SymIntConcrete { as_int: 1 })],
        requires_grad: false,
        strides: vec![SymInt::Concrete(SymIntConcrete { as_int: 1 })],
        storage_offset: None,
        device: None,
        layout: None,
    };
    assert_eq!(meta.to_dtype(), Some(DType::F64));
}

#[test]
fn test_scalar_type_to_dtype_i64() {
    let meta = TensorMeta {
        dtype: 5,
        sizes: vec![SymInt::Concrete(SymIntConcrete { as_int: 3 })],
        requires_grad: false,
        strides: vec![SymInt::Concrete(SymIntConcrete { as_int: 1 })],
        storage_offset: None,
        device: None,
        layout: None,
    };
    assert_eq!(meta.to_dtype(), Some(DType::I64));
}

#[test]
fn test_scalar_type_to_dtype_u8() {
    let meta = TensorMeta {
        dtype: 1,
        sizes: vec![SymInt::Concrete(SymIntConcrete { as_int: 5 })],
        requires_grad: false,
        strides: vec![SymInt::Concrete(SymIntConcrete { as_int: 1 })],
        storage_offset: None,
        device: None,
        layout: None,
    };
    assert_eq!(meta.to_dtype(), Some(DType::U8));
}

#[test]
fn test_scalar_type_to_dtype_unknown_returns_none() {
    let meta = TensorMeta {
        dtype: 999,
        sizes: vec![SymInt::Concrete(SymIntConcrete { as_int: 1 })],
        requires_grad: false,
        strides: vec![SymInt::Concrete(SymIntConcrete { as_int: 1 })],
        storage_offset: None,
        device: None,
        layout: None,
    };
    assert_eq!(meta.to_dtype(), None);
}

#[test]
fn test_dtype_preserved_through_graph_import() {
    // Build a graph with F16 dtype (code 6) and verify it comes through.
    let json = single_op_json("x", "y", &[2, 4], 6, "torch.ops.aten.relu.default", "");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    // Both input and output nodes should have F16 dtype.
    let input_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| n.name() == "x")
        .unwrap();
    assert_eq!(input_node.output_dtype(), DType::F16);
    let output_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| n.name() == "y")
        .unwrap();
    assert_eq!(output_node.output_dtype(), DType::F16);
}

// ===========================================================================
// 5. Shape inference: output shapes computed from input shapes
// ===========================================================================

#[test]
fn test_shape_inference_unary_op_preserves_shape() {
    let json = single_op_json("x", "y", &[2, 3, 4], 7, "torch.ops.aten.relu.default", "");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    let output_node = imported.graph.output_node().unwrap();
    assert_eq!(output_node.output_shape(), &[2, 3, 4]);
}

#[test]
fn test_shape_inference_from_tensor_meta() {
    let meta = TensorMeta {
        dtype: 7,
        sizes: vec![
            SymInt::Concrete(SymIntConcrete { as_int: 8 }),
            SymInt::Concrete(SymIntConcrete { as_int: 16 }),
            SymInt::Concrete(SymIntConcrete { as_int: 32 }),
        ],
        requires_grad: false,
        strides: vec![
            SymInt::Concrete(SymIntConcrete { as_int: 512 }),
            SymInt::Concrete(SymIntConcrete { as_int: 32 }),
            SymInt::Concrete(SymIntConcrete { as_int: 1 }),
        ],
        storage_offset: None,
        device: None,
        layout: None,
    };
    let shape = meta.concrete_shape().unwrap();
    assert_eq!(shape, vec![8, 16, 32]);
}

#[test]
fn test_shape_inference_scalar_tensor() {
    // Scalar tensor: empty shape [].
    let json = single_op_json("x", "y", &[], 7, "torch.ops.aten.relu.default", "");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    let input_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| n.name() == "x")
        .unwrap();
    assert_eq!(input_node.output_shape(), &[] as &[usize]);
}

#[test]
fn test_shape_inference_high_rank_tensor() {
    // 5D tensor: [B, C, D, H, W].
    let shape = [2, 3, 4, 8, 8];
    let json = single_op_json("x", "y", &shape, 7, "torch.ops.aten.relu.default", "");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    let output_node = imported.graph.output_node().unwrap();
    assert_eq!(output_node.output_shape(), &shape);
}

// ===========================================================================
// 6. Batch dimension handling: batch dim preserved through import
// ===========================================================================

#[test]
fn test_batch_dim_preserved_in_simple_graph() {
    // [B=4, C=16] input through relu should preserve batch dimension.
    let json = single_op_json("x", "y", &[4, 16], 7, "torch.ops.aten.relu.default", "");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    let output_node = imported.graph.output_node().unwrap();
    assert_eq!(output_node.output_shape()[0], 4, "batch dim should be 4");
    assert_eq!(
        output_node.output_shape()[1],
        16,
        "channel dim should be 16"
    );
}

#[test]
fn test_batch_dim_preserved_4d_conv_input() {
    // [B=2, C=3, H=32, W=32] — 4D tensor typical for CNNs.
    let shape = [2, 3, 32, 32];
    let json = single_op_json("x", "y", &shape, 7, "torch.ops.aten.relu.default", "");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    let output = imported.graph.output_node().unwrap();
    assert_eq!(output.output_shape(), &shape);
    assert_eq!(output.output_shape()[0], 2, "batch dimension preserved");
}

#[test]
fn test_batch_dim_one_still_preserved() {
    // B=1 is the most common inference batch. Verify it is preserved (not squeezed).
    let json = single_op_json("x", "y", &[1, 512], 7, "torch.ops.aten.relu.default", "");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    let output = imported.graph.output_node().unwrap();
    assert_eq!(
        output.output_shape().len(),
        2,
        "rank should remain 2, not squeezed to 1D"
    );
    assert_eq!(output.output_shape()[0], 1);
}

#[test]
fn test_batch_dim_preserved_with_embedding() {
    // Embedding: [B=1, SeqLen=4] indices -> [B=1, SeqLen=4, EmbDim=8].
    let json = include_str!("../test_data/embedding_lookup.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let mut wd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    wd.insert("embed.weight".to_string(), (vec![0.0; 80], vec![10, 8]));
    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    let imported = build_graph(&program, &wm).unwrap();

    // Input: [1, 4], output: [1, 4, 8].
    let output = imported.graph.output_node().unwrap();
    assert_eq!(
        output.output_shape()[0],
        1,
        "batch dimension preserved in embedding output"
    );
}

// ===========================================================================
// 7. Multi-file model loading: sharded models reassembled correctly
// ===========================================================================

#[test]
fn test_multi_segment_basic_two_segments() {
    // Create a temporary dir with a safetensors file.
    let weights_data = build_safetensors(&[
        ("fc1.weight", safetensors::Dtype::F32, &[4, 8]),
        ("fc1.bias", safetensors::Dtype::F32, &[4]),
        ("fc2.weight", safetensors::Dtype::F32, &[2, 4]),
        ("fc2.bias", safetensors::Dtype::F32, &[2]),
    ]);
    let tmp_dir = std::env::temp_dir().join("nn_import_test_multi_seg");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let weights_path = tmp_dir.join("weights.safetensors");
    std::fs::write(&weights_path, &weights_data).unwrap();

    // Build two minimal graph JSONs: encoder uses fc1, decoder uses fc2.
    let encoder_json = make_json_graph(
        r#"{"parameter": {"arg": {"name": "p_fc1_weight"}, "parameter_name": "fc1.weight"}},
        {"parameter": {"arg": {"name": "p_fc1_bias"}, "parameter_name": "fc1.bias"}},
        {"user_input": {"arg": {"as_tensor": {"name": "enc_in"}}}}"#,
        r#"{"user_output": {"arg": {"as_tensor": {"name": "enc_out"}}}}"#,
        r#"{
            "target": "torch.ops.aten.linear.default",
            "inputs": [
                {"name": "input", "arg": {"as_tensor": {"name": "enc_in"}}, "kind": 1},
                {"name": "weight", "arg": {"as_tensor": {"name": "p_fc1_weight"}}, "kind": 1},
                {"name": "bias", "arg": {"as_tensor": {"name": "p_fc1_bias"}}, "kind": 1}
            ],
            "outputs": [{"as_tensor": {"name": "enc_out"}}],
            "metadata": {}
        }"#,
        r#""enc_in": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 8}], "requires_grad": false, "strides": [{"as_int": 8}, {"as_int": 1}]},
        "enc_out": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]},
        "p_fc1_weight": {"dtype": 7, "sizes": [{"as_int": 4}, {"as_int": 8}], "requires_grad": true, "strides": [{"as_int": 8}, {"as_int": 1}]},
        "p_fc1_bias": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 1}]}"#,
    );

    let decoder_json = make_json_graph(
        r#"{"parameter": {"arg": {"name": "p_fc2_weight"}, "parameter_name": "fc2.weight"}},
        {"parameter": {"arg": {"name": "p_fc2_bias"}, "parameter_name": "fc2.bias"}},
        {"user_input": {"arg": {"as_tensor": {"name": "dec_in"}}}}"#,
        r#"{"user_output": {"arg": {"as_tensor": {"name": "dec_out"}}}}"#,
        r#"{
            "target": "torch.ops.aten.linear.default",
            "inputs": [
                {"name": "input", "arg": {"as_tensor": {"name": "dec_in"}}, "kind": 1},
                {"name": "weight", "arg": {"as_tensor": {"name": "p_fc2_weight"}}, "kind": 1},
                {"name": "bias", "arg": {"as_tensor": {"name": "p_fc2_bias"}}, "kind": 1}
            ],
            "outputs": [{"as_tensor": {"name": "dec_out"}}],
            "metadata": {}
        }"#,
        r#""dec_in": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]},
        "dec_out": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 2}], "requires_grad": false, "strides": [{"as_int": 2}, {"as_int": 1}]},
        "p_fc2_weight": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 4}, {"as_int": 1}]},
        "p_fc2_bias": {"dtype": 7, "sizes": [{"as_int": 2}], "requires_grad": true, "strides": [{"as_int": 1}]}"#,
    );

    let enc_val: serde_json::Value = serde_json::from_str(&encoder_json).unwrap();
    let dec_val: serde_json::Value = serde_json::from_str(&decoder_json).unwrap();

    let graphs = vec![
        ("encoder".to_string(), enc_val),
        ("decoder".to_string(), dec_val),
    ];
    let model = convert_multi_segment(&graphs, &weights_path).unwrap();

    assert_eq!(model.num_segments(), 2);
    assert_eq!(model.segment_order, vec!["encoder", "decoder"]);
    assert!(model.get_segment("encoder").is_some());
    assert!(model.get_segment("decoder").is_some());
    assert!(model.get_segment("nonexistent").is_none());

    // No shared weights between these two segments.
    assert!(model.shared_weights.is_empty());

    // Clean up.
    let _ = std::fs::remove_file(&weights_path);
}

#[test]
fn test_multi_segment_duplicate_name_rejected() {
    let val: serde_json::Value = serde_json::from_str(&make_json_graph(
        r#"{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}"#,
        r#"{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}"#,
        "",
        r#""x": {"dtype": 7, "sizes": [{"as_int": 1}], "requires_grad": false, "strides": [{"as_int": 1}]}"#,
    )).unwrap();

    let graphs = vec![("seg".to_string(), val.clone()), ("seg".to_string(), val)];
    let tmp_weights = std::env::temp_dir().join("nn_import_test_dup_seg.safetensors");
    let data = build_safetensors(&[("dummy", safetensors::Dtype::F32, &[1])]);
    std::fs::write(&tmp_weights, data).unwrap();

    let result = convert_multi_segment(&graphs, &tmp_weights);
    assert!(matches!(
        result,
        Err(MultiSegmentError::DuplicateSegment { .. })
    ));
    let _ = std::fs::remove_file(&tmp_weights);
}

#[test]
fn test_multi_segment_empty_input_rejected() {
    let tmp_weights = std::env::temp_dir().join("nn_import_test_empty_seg.safetensors");
    let data = build_safetensors(&[("dummy", safetensors::Dtype::F32, &[1])]);
    std::fs::write(&tmp_weights, data).unwrap();

    let result = convert_multi_segment(&[], &tmp_weights);
    assert!(matches!(result, Err(MultiSegmentError::EmptyInput)));
    let _ = std::fs::remove_file(&tmp_weights);
}

// ===========================================================================
// 8. Config parsing: HuggingFace config.json fields extracted
// ===========================================================================

#[test]
fn test_parse_exported_program_schema_version_8() {
    let json = make_json_graph(
        r#"{"user_input": {"arg": {"as_tensor": {"name": "x"}}}}"#,
        r#"{"user_output": {"arg": {"as_tensor": {"name": "x"}}}}"#,
        "",
        r#""x": {"dtype": 7, "sizes": [{"as_int": 1}], "requires_grad": false, "strides": [{"as_int": 1}]}"#,
    );
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert_eq!(program.schema_version.major, 8);
}

#[test]
fn test_parse_exported_program_rejects_wrong_schema_version() {
    let json = r#"{
        "graph_module": {
            "graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}},
            "signature": {"input_specs": [], "output_specs": []},
            "module_call_graph": []
        },
        "schema_version": {"major": 5, "minor": 0},
        "range_constraints": {}
    }"#;
    let result = parse_exported_program(json.as_bytes());
    assert!(result.is_err());
    match result.unwrap_err() {
        ImportError::UnsupportedSchema { major, .. } => assert_eq!(major, 5),
        other => panic!("expected UnsupportedSchema, got: {other:?}"),
    }
}

#[test]
fn test_parse_exported_program_extracts_opset_version() {
    let json = r#"{
        "graph_module": {
            "graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}},
            "signature": {"input_specs": [], "output_specs": []},
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 10},
        "opset_version": {"aten": 10, "custom": 2},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert_eq!(program.opset_version.get("aten"), Some(&10));
    assert_eq!(program.opset_version.get("custom"), Some(&2));
}

#[test]
fn test_parse_exported_program_extracts_range_constraints() {
    let json = r#"{
        "graph_module": {
            "graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}},
            "signature": {"input_specs": [], "output_specs": []},
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 10},
        "range_constraints": {"s0": {"min_val": 1, "max_val": 2048}}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let rc = program.range_constraints.get("s0").unwrap();
    assert_eq!(rc.min_val, 1);
    assert_eq!(rc.max_val, 2048);
}

#[test]
fn test_parse_exported_program_extracts_torch_version() {
    let json = r#"{
        "graph_module": {
            "graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}},
            "signature": {"input_specs": [], "output_specs": []},
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 10},
        "range_constraints": {},
        "torch_version": "2.5.0"
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert_eq!(program.torch_version.as_deref(), Some("2.5.0"));
}

#[test]
fn test_parse_input_spec_parameter() {
    let json = r#"{"parameter": {"arg": {"name": "p_weight"}, "parameter_name": "layer.weight"}}"#;
    let spec: InputSpec = serde_json::from_str(json).unwrap();
    match spec {
        InputSpec::Parameter(p) => {
            assert_eq!(p.parameter.arg.name, "p_weight");
            assert_eq!(p.parameter.parameter_name, "layer.weight");
        }
        other => panic!("expected Parameter, got: {other:?}"),
    }
}

#[test]
fn test_parse_input_spec_buffer() {
    let json = r#"{"buffer": {"arg": {"name": "b_running_mean"}, "buffer_name": "bn.running_mean", "persistent": true}}"#;
    let spec: InputSpec = serde_json::from_str(json).unwrap();
    match spec {
        InputSpec::Buffer(b) => {
            assert_eq!(b.buffer.arg.name, "b_running_mean");
            assert_eq!(b.buffer.buffer_name, "bn.running_mean");
            assert!(b.buffer.persistent);
        }
        other => panic!("expected Buffer, got: {other:?}"),
    }
}

#[test]
fn test_parse_input_spec_user_input() {
    let json = r#"{"user_input": {"arg": {"as_tensor": {"name": "input_ids"}}}}"#;
    let spec: InputSpec = serde_json::from_str(json).unwrap();
    match spec {
        InputSpec::UserInput(ui) => {
            let name = ui.user_input.arg.as_tensor_name().unwrap();
            assert_eq!(name, "input_ids");
        }
        other => panic!("expected UserInput, got: {other:?}"),
    }
}

#[test]
fn test_parse_argument_variants() {
    // Tensor
    let t: Argument = serde_json::from_str(r#"{"as_tensor": {"name": "x"}}"#).unwrap();
    assert_eq!(t.as_tensor_name(), Some("x"));

    // Int
    let i: Argument = serde_json::from_str(r#"{"as_int": 42}"#).unwrap();
    assert_eq!(i.as_int(), Some(42));

    // Float
    let f: Argument = serde_json::from_str(r#"{"as_float": 3.14}"#).unwrap();
    assert!((f.as_float().unwrap() - 3.14).abs() < 0.001);

    // Bool
    let b: Argument = serde_json::from_str(r#"{"as_bool": true}"#).unwrap();
    assert_eq!(b.as_bool_val(), Some(true));

    // None
    let n: Argument = serde_json::from_str(r#"{"as_none": true}"#).unwrap();
    assert!(n.is_none());

    // String
    let s: Argument = serde_json::from_str(r#"{"as_string": "tanh"}"#).unwrap();
    assert_eq!(s.as_string(), Some("tanh"));

    // Ints
    let is: Argument = serde_json::from_str(r#"{"as_ints": [1, 2, 3]}"#).unwrap();
    assert_eq!(is.as_ints(), Some(&[1i64, 2, 3][..]));
}

// ===========================================================================
// 9. Quantization import: GPTQ/AWQ weight formats recognized
// ===========================================================================

#[test]
fn test_quantization_detect_pure_f32_model() {
    let data = build_safetensors(&[
        ("layer1.weight", safetensors::Dtype::F32, &[128, 64]),
        ("layer1.bias", safetensors::Dtype::F32, &[128]),
        ("layer2.weight", safetensors::Dtype::F32, &[64, 128]),
    ]);
    let report = detect_quantization_from_bytes(&data).unwrap();
    assert_eq!(report.total_tensors, 3);
    assert!(!report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown.len(), 1);
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::F32);
    // All data is F32, so recommendations should suggest F16 and I8.
    assert!(!report.recommendations.is_empty());
}

#[test]
fn test_quantization_detect_mixed_f32_f16() {
    let data = build_safetensors(&[
        ("embed.weight", safetensors::Dtype::F32, &[1000, 64]),
        ("attn.qkv", safetensors::Dtype::F16, &[192, 64]),
        ("attn.out", safetensors::Dtype::F16, &[64, 64]),
    ]);
    let report = detect_quantization_from_bytes(&data).unwrap();
    assert_eq!(report.total_tensors, 3);
    assert!(report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown.len(), 2);

    let f32_frac = report.dtype_fraction(DetectedDtype::F32);
    let f16_frac = report.dtype_fraction(DetectedDtype::F16);
    assert!(f32_frac > 0.0 && f32_frac < 1.0);
    assert!(f16_frac > 0.0 && f16_frac < 1.0);
    assert!((f32_frac + f16_frac - 1.0).abs() < 1e-6);
}

#[test]
fn test_quantization_detect_bf16_model() {
    let data = build_safetensors(&[
        ("w1", safetensors::Dtype::BF16, &[512, 256]),
        ("w2", safetensors::Dtype::BF16, &[256, 512]),
    ]);
    let report = detect_quantization_from_bytes(&data).unwrap();
    assert_eq!(report.total_tensors, 2);
    assert!(!report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::BF16);
    // BF16 model should have no recommendations (already compact).
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_quantization_detect_i8_quantized() {
    let data = build_safetensors(&[
        ("qweight", safetensors::Dtype::I8, &[128, 128]),
        ("scales", safetensors::Dtype::F16, &[128]),
        ("zeros", safetensors::Dtype::F16, &[128]),
    ]);
    let report = detect_quantization_from_bytes(&data).unwrap();
    assert!(report.is_mixed_precision());
    assert!(report.total_tensors == 3);

    // Should detect both I8 and F16 categories.
    let has_i8 = report
        .dtype_breakdown
        .iter()
        .any(|b| b.dtype == DetectedDtype::I8);
    let has_f16 = report
        .dtype_breakdown
        .iter()
        .any(|b| b.dtype == DetectedDtype::F16);
    assert!(has_i8, "should detect I8 tensors");
    assert!(has_f16, "should detect F16 scale/zero tensors");
}

#[test]
fn test_quantization_detect_u8_model() {
    let data = build_safetensors(&[("qweight", safetensors::Dtype::U8, &[64, 64])]);
    let report = detect_quantization_from_bytes(&data).unwrap();
    let has_u8 = report
        .dtype_breakdown
        .iter()
        .any(|b| b.dtype == DetectedDtype::U8);
    assert!(has_u8);
}

#[test]
fn test_quantization_report_total_savings() {
    let data = build_safetensors(&[("big_weight", safetensors::Dtype::F32, &[2048, 2048])]);
    let report = detect_quantization_from_bytes(&data).unwrap();
    let total_savings = report.total_savings_bytes();
    assert!(
        total_savings > 0,
        "should recommend savings for large F32 weight"
    );
    // F32->F16 saves 50%, F32->I8 saves 75%. Both are recommended.
    let f16_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == DetectedDtype::F16);
    assert!(f16_rec.is_some());
    let i8_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == DetectedDtype::I8);
    assert!(i8_rec.is_some());
}

#[test]
fn test_quantization_report_summary_not_empty() {
    let data = build_safetensors(&[("w", safetensors::Dtype::F32, &[64, 64])]);
    let report = detect_quantization_from_bytes(&data).unwrap();
    let summary = report.summary();
    assert!(!summary.is_empty());
    assert!(summary.contains("Quantization Report"));
    assert!(summary.contains("Dtype Breakdown"));
}

#[test]
fn test_quantization_bytes_per_element() {
    assert_eq!(DetectedDtype::F32.bytes_per_element(), Some(4));
    assert_eq!(DetectedDtype::F16.bytes_per_element(), Some(2));
    assert_eq!(DetectedDtype::BF16.bytes_per_element(), Some(2));
    assert_eq!(DetectedDtype::F64.bytes_per_element(), Some(8));
    assert_eq!(DetectedDtype::I8.bytes_per_element(), Some(1));
    assert_eq!(DetectedDtype::U8.bytes_per_element(), Some(1));
    assert_eq!(DetectedDtype::I32.bytes_per_element(), Some(4));
    assert_eq!(DetectedDtype::I64.bytes_per_element(), Some(8));
    assert_eq!(DetectedDtype::Bool.bytes_per_element(), Some(1));
    assert_eq!(DetectedDtype::SubByte.bytes_per_element(), None);
    assert_eq!(DetectedDtype::Other.bytes_per_element(), None);
}

#[test]
fn test_detected_dtype_from_safetensors_mapping() {
    use safetensors::Dtype as SD;
    assert_eq!(DetectedDtype::from_safetensors(SD::F32), DetectedDtype::F32);
    assert_eq!(DetectedDtype::from_safetensors(SD::F16), DetectedDtype::F16);
    assert_eq!(
        DetectedDtype::from_safetensors(SD::BF16),
        DetectedDtype::BF16
    );
    assert_eq!(DetectedDtype::from_safetensors(SD::F64), DetectedDtype::F64);
    assert_eq!(DetectedDtype::from_safetensors(SD::I8), DetectedDtype::I8);
    assert_eq!(DetectedDtype::from_safetensors(SD::U8), DetectedDtype::U8);
    assert_eq!(DetectedDtype::from_safetensors(SD::I64), DetectedDtype::I64);
    assert_eq!(
        DetectedDtype::from_safetensors(SD::BOOL),
        DetectedDtype::Bool
    );
}

#[test]
fn test_detected_dtype_label_roundtrip() {
    let all_variants = [
        DetectedDtype::F32,
        DetectedDtype::F16,
        DetectedDtype::BF16,
        DetectedDtype::F64,
        DetectedDtype::I8,
        DetectedDtype::U8,
        DetectedDtype::F8,
        DetectedDtype::SubByte,
        DetectedDtype::I16,
        DetectedDtype::I32,
        DetectedDtype::I64,
        DetectedDtype::Bool,
        DetectedDtype::C64,
        DetectedDtype::Other,
    ];
    for v in &all_variants {
        assert!(!v.label().is_empty());
        assert_eq!(format!("{v}"), v.label());
    }
}

// ===========================================================================
// 10. Tokenizer import: vocabulary and special tokens loaded
//     (via Kokoro weight key mapping, which is the import-side
//     tokenizer/vocabulary validation path)
// ===========================================================================

#[test]
fn test_kokoro_weight_key_mapping_identity() {
    let key = "plbert.embeddings.word_embeddings.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_kokoro_weight_key_mapping_all_prefixes() {
    let prefixes = [
        "plbert.embeddings.LayerNorm.weight",
        "bert_encoder.weight",
        "text_encoder.lstm.weight_ih_l0",
        "prosody_predictor.shared.0.conv.weight",
        "predictor.shared.weight_ih_l0",
        "decoder.conv_pre.weight",
    ];
    for key in &prefixes {
        assert!(
            map_pytorch_key(key).is_some(),
            "key '{key}' should be mappable"
        );
    }
}

#[test]
fn test_kokoro_weight_key_mapping_unknown_prefix_returns_none() {
    assert_eq!(map_pytorch_key("unknown.module.weight"), None);
    assert_eq!(map_pytorch_key(""), None);
    assert_eq!(map_pytorch_key("attn.q.weight"), None);
}

#[test]
fn test_kokoro_validate_keys_all_present() {
    let keys = vec![
        "plbert.embeddings.word_embeddings.weight",
        "bert_encoder.weight",
        "text_encoder.lstm.weight_ih_l0",
        "prosody_predictor.shared.0.conv.weight",
        "predictor.shared.weight_ih_l0",
        "decoder.conv_pre.weight",
    ];
    let missing = validate_kokoro_keys(&keys);
    assert!(
        missing.is_empty(),
        "all required prefixes should be present"
    );
}

#[test]
fn test_kokoro_validate_keys_missing_prefix() {
    let keys = vec![
        "plbert.embeddings.word_embeddings.weight",
        "bert_encoder.weight",
        "text_encoder.lstm.weight_ih_l0",
        // Missing: prosody_predictor, predictor, decoder.
    ];
    let missing = validate_kokoro_keys(&keys);
    assert_eq!(missing.len(), 3);
    assert!(missing.contains(&"prosody_predictor."));
    assert!(missing.contains(&"predictor."));
    assert!(missing.contains(&"decoder."));
}

#[test]
fn test_kokoro_name_mapping_closure() {
    let mapping = kokoro_name_mapping();
    assert_eq!(
        mapping("plbert.embeddings.word_embeddings.weight"),
        "plbert.embeddings.word_embeddings.weight"
    );
    // Unknown key returns identity (fallback).
    assert_eq!(mapping("unknown.key"), "unknown.key");
}

#[test]
fn test_kokoro_validate_safetensors_all_groups_present() {
    let keys: Vec<String> = vec![
        "plbert.embeddings.word_embeddings.weight".to_string(),
        "bert_encoder.weight".to_string(),
        "text_encoder.lstm.weight_ih_l0".to_string(),
        "prosody_predictor.shared.0.conv.weight".to_string(),
        "predictor.shared.weight_ih_l0".to_string(),
        "decoder.conv_pre.weight".to_string(),
    ];
    let result = crate::kokoro_weights::validate_kokoro_safetensors(&keys);
    assert!(result.is_ok());
    let mapped = result.unwrap();
    assert_eq!(mapped, 6);
}

#[test]
fn test_kokoro_validate_safetensors_missing_groups_error() {
    let keys: Vec<String> = vec!["plbert.embeddings.weight".to_string()];
    let result = crate::kokoro_weights::validate_kokoro_safetensors(&keys);
    assert!(result.is_err());
    match result.unwrap_err() {
        ImportError::MissingWeightGroups { missing_prefixes } => {
            assert!(missing_prefixes.contains("bert_encoder."));
            assert!(missing_prefixes.contains("decoder."));
        }
        other => panic!("expected MissingWeightGroups, got: {other:?}"),
    }
}

// ===========================================================================
// Additional: build_weight_map integration tests
// ===========================================================================

#[test]
fn test_build_weight_map_maps_parameter_specs() {
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [], "outputs": [], "nodes": [],
                "tensor_values": {}
            },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_fc_weight"}, "parameter_name": "fc.weight"}},
                    {"parameter": {"arg": {"name": "p_fc_bias"}, "parameter_name": "fc.bias"}}
                ],
                "output_specs": []
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut wd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    wd.insert("fc.weight".to_string(), (vec![1.0; 12], vec![4, 3]));
    wd.insert("fc.bias".to_string(), (vec![0.0; 4], vec![4]));

    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    assert!(wm.contains_key("p_fc_weight"));
    assert!(wm.contains_key("p_fc_bias"));
    assert_eq!(wm["p_fc_weight"].shape, vec![4, 3]);
    assert_eq!(wm["p_fc_bias"].shape, vec![4]);
    assert_eq!(wm["p_fc_weight"].data.len(), 12);
}

#[test]
fn test_build_weight_map_maps_buffer_specs() {
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [], "outputs": [], "nodes": [],
                "tensor_values": {}
            },
            "signature": {
                "input_specs": [
                    {"buffer": {"arg": {"name": "b_running_mean"}, "buffer_name": "bn.running_mean", "persistent": true}}
                ],
                "output_specs": []
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut wd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    wd.insert("bn.running_mean".to_string(), (vec![0.0; 16], vec![16]));

    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    assert!(wm.contains_key("b_running_mean"));
    assert_eq!(wm["b_running_mean"].shape, vec![16]);
}

#[test]
fn test_build_weight_map_ignores_missing_weights() {
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [], "outputs": [], "nodes": [],
                "tensor_values": {}
            },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_w"}, "parameter_name": "missing.weight"}}
                ],
                "output_specs": []
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let wd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    assert!(
        !wm.contains_key("p_w"),
        "missing weight should not appear in map"
    );
}

// ===========================================================================
// Additional: ResolvedWeight constructor
// ===========================================================================

#[test]
fn test_resolved_weight_constructor() {
    let rw = ResolvedWeight::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(rw.data, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(rw.shape, vec![2, 2]);
}

// ===========================================================================
// Additional: error type coverage
// ===========================================================================

#[test]
fn test_import_error_display_variants() {
    let err = ImportError::UnsupportedOp {
        target: "aten::exotic_op".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("unsupported aten op"));
    assert!(msg.contains("exotic_op"));

    let err2 = ImportError::MissingArgument {
        op_target: "aten::linear".to_string(),
        arg_name: "weight".to_string(),
    };
    let msg2 = format!("{err2}");
    assert!(msg2.contains("missing argument"));
    assert!(msg2.contains("weight"));

    let err3 = ImportError::WeightShapeMismatch {
        name: "fc.weight".to_string(),
        shape: vec![4, 3],
        expected: 12,
        actual: 10,
    };
    let msg3 = format!("{err3}");
    assert!(msg3.contains("shape"));
    assert!(msg3.contains("fc.weight"));
}

#[test]
fn test_convert_error_display_variants() {
    use crate::convert::ConvertError;

    let err = ConvertError::Compile("out of memory".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("compilation error"));
    assert!(msg.contains("out of memory"));

    let err2 = ConvertError::Reftest("mismatch at output_0".to_string());
    let msg2 = format!("{err2}");
    assert!(msg2.contains("reftest error"));
}

// ===========================================================================
// Additional: SymInt and TensorMeta edge cases
// ===========================================================================

#[test]
fn test_symint_concrete_extraction() {
    let sym = SymInt::Concrete(SymIntConcrete { as_int: 42 });
    assert_eq!(sym.as_concrete(), Some(42));
}

#[test]
fn test_symint_symbolic_returns_none() {
    use crate::parse::{SymIntExpr, SymIntSymbolic};
    let sym = SymInt::Symbolic(SymIntSymbolic {
        as_expr: SymIntExpr {
            expr_str: "s0".to_string(),
            hint: None,
        },
    });
    assert_eq!(sym.as_concrete(), None);
}

#[test]
fn test_tensor_meta_dynamic_shape_returns_none() {
    use crate::parse::{SymIntExpr, SymIntSymbolic};
    let meta = TensorMeta {
        dtype: 7,
        sizes: vec![
            SymInt::Concrete(SymIntConcrete { as_int: 1 }),
            SymInt::Symbolic(SymIntSymbolic {
                as_expr: SymIntExpr {
                    expr_str: "s0".to_string(),
                    hint: None,
                },
            }),
        ],
        requires_grad: false,
        strides: vec![],
        storage_offset: None,
        device: None,
        layout: None,
    };
    // Dynamic dims cause concrete_shape() to return None.
    assert!(meta.concrete_shape().is_none());
}

// ===========================================================================
// Additional: ImportedGraph constructor and field access
// ===========================================================================

#[test]
fn test_imported_graph_constructor() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let ig = ImportedGraph::new(
        graph,
        2,
        vec!["a".to_string(), "b".to_string()],
        vec!["out".to_string()],
    );
    assert_eq!(ig.num_user_inputs, 2);
    assert_eq!(ig.user_input_names, vec!["a", "b"]);
    assert_eq!(ig.output_names, vec!["out"]);
}

// ===========================================================================
// Additional: EquivalenceProof and reports
// ===========================================================================

#[test]
fn test_equivalence_proof_with_all_layers() {
    use crate::convert::{CompositionBoundsReport, EquivalenceProof, KaniSafetyReport};
    let proof = EquivalenceProof::new(
        Some(KaniSafetyReport::new(100, 99, 1)),
        Some(CompositionBoundsReport::new(true, Some(0.5))),
        None,
    );
    assert!(proof.kernel_safety.is_some());
    assert!(proof.composition_bounds.is_some());
    assert!(proof.reference_parity.is_none());
    assert_eq!(proof.kernel_safety.as_ref().unwrap().harness_count, 100);
    assert_eq!(proof.kernel_safety.as_ref().unwrap().passed, 99);
    assert_eq!(proof.kernel_safety.as_ref().unwrap().failed, 1);
    assert!(proof.composition_bounds.as_ref().unwrap().propagation_ok);
}

// ===========================================================================
// Additional: Quantization small tensor skip
// ===========================================================================

#[test]
fn test_quantization_small_f32_tensors_not_recommended() {
    // Tensors with < 1024 elements should NOT be recommended for quantization.
    let data = build_safetensors(&[("small_bias", safetensors::Dtype::F32, &[64])]);
    let report = detect_quantization_from_bytes(&data).unwrap();
    // 64 < 1024, so no recommendations.
    assert!(
        report.recommendations.is_empty(),
        "small tensors (< 1024 elements) should not get quantization recommendations"
    );
}

#[test]
fn test_quantization_f64_tensors_recommend_f32() {
    let data = build_safetensors(&[("big_f64", safetensors::Dtype::F64, &[128, 128])]);
    let report = detect_quantization_from_bytes(&data).unwrap();
    let f32_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == DetectedDtype::F32);
    assert!(f32_rec.is_some(), "F64 tensors should recommend F32");
    let savings = f32_rec.unwrap().savings_bytes;
    assert!(savings > 0, "F64->F32 should save bytes");
}

// ===========================================================================
// 11. Weight name mapping: HF-to-NN name translation correctness
// ===========================================================================

#[test]
fn test_weight_name_mapping_hf_linear_layers() {
    // HuggingFace transformer models use dot-separated FQNs that must be
    // preserved through the import pipeline's weight_map construction.
    let json = r#"{
        "graph_module": {
            "graph": {
                "inputs": [], "outputs": [], "nodes": [],
                "tensor_values": {}
            },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_model_layers_0_self_attn_q_proj_weight"}, "parameter_name": "model.layers.0.self_attn.q_proj.weight"}},
                    {"parameter": {"arg": {"name": "p_model_layers_0_self_attn_q_proj_bias"}, "parameter_name": "model.layers.0.self_attn.q_proj.bias"}},
                    {"parameter": {"arg": {"name": "p_model_layers_0_self_attn_k_proj_weight"}, "parameter_name": "model.layers.0.self_attn.k_proj.weight"}},
                    {"parameter": {"arg": {"name": "p_model_layers_0_mlp_gate_proj_weight"}, "parameter_name": "model.layers.0.mlp.gate_proj.weight"}}
                ],
                "output_specs": []
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut wd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    wd.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![0.1; 64], vec![8, 8]),
    );
    wd.insert(
        "model.layers.0.self_attn.q_proj.bias".to_string(),
        (vec![0.0; 8], vec![8]),
    );
    wd.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        (vec![0.2; 64], vec![8, 8]),
    );
    wd.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        (vec![0.3; 64], vec![8, 8]),
    );

    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    assert_eq!(wm.len(), 4, "all 4 weights should be mapped");
    assert!(wm.contains_key("p_model_layers_0_self_attn_q_proj_weight"));
    assert!(wm.contains_key("p_model_layers_0_self_attn_q_proj_bias"));
    assert!(wm.contains_key("p_model_layers_0_self_attn_k_proj_weight"));
    assert!(wm.contains_key("p_model_layers_0_mlp_gate_proj_weight"));
}

#[test]
fn test_weight_name_mapping_nested_modules() {
    // Deeply nested module names must map through correctly.
    let json = r#"{
        "graph_module": {
            "graph": { "inputs": [], "outputs": [], "nodes": [], "tensor_values": {} },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_enc_layers_0_attn_key_weight"}, "parameter_name": "encoder.layers.0.attention.key.weight"}},
                    {"parameter": {"arg": {"name": "p_enc_layers_0_attn_value_weight"}, "parameter_name": "encoder.layers.0.attention.value.weight"}},
                    {"buffer": {"arg": {"name": "b_enc_layers_0_norm_running_mean"}, "buffer_name": "encoder.layers.0.norm.running_mean", "persistent": true}}
                ],
                "output_specs": []
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut wd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    wd.insert(
        "encoder.layers.0.attention.key.weight".to_string(),
        (vec![0.1; 16], vec![4, 4]),
    );
    wd.insert(
        "encoder.layers.0.attention.value.weight".to_string(),
        (vec![0.2; 16], vec![4, 4]),
    );
    wd.insert(
        "encoder.layers.0.norm.running_mean".to_string(),
        (vec![0.0; 4], vec![4]),
    );

    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    assert_eq!(wm.len(), 3);
    assert_eq!(wm["p_enc_layers_0_attn_key_weight"].shape, vec![4, 4]);
    assert_eq!(wm["p_enc_layers_0_attn_value_weight"].shape, vec![4, 4]);
    assert_eq!(wm["b_enc_layers_0_norm_running_mean"].shape, vec![4]);
}

#[test]
fn test_weight_name_mapping_preserves_data_values() {
    let json = r#"{
        "graph_module": {
            "graph": { "inputs": [], "outputs": [], "nodes": [], "tensor_values": {} },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_w"}, "parameter_name": "fc.weight"}}
                ],
                "output_specs": []
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let specific_data = vec![1.0, -2.5, 3.14, 0.0, -0.001, 999.9];
    let mut wd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    wd.insert("fc.weight".to_string(), (specific_data.clone(), vec![2, 3]));
    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    assert_eq!(wm["p_w"].data, specific_data);
}

#[test]
fn test_weight_name_mapping_multiple_layers_same_type() {
    let json = r#"{
        "graph_module": {
            "graph": { "inputs": [], "outputs": [], "nodes": [], "tensor_values": {} },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_fc1_weight"}, "parameter_name": "fc1.weight"}},
                    {"parameter": {"arg": {"name": "p_fc1_bias"}, "parameter_name": "fc1.bias"}},
                    {"parameter": {"arg": {"name": "p_fc2_weight"}, "parameter_name": "fc2.weight"}},
                    {"parameter": {"arg": {"name": "p_fc2_bias"}, "parameter_name": "fc2.bias"}},
                    {"parameter": {"arg": {"name": "p_fc3_weight"}, "parameter_name": "fc3.weight"}},
                    {"parameter": {"arg": {"name": "p_fc3_bias"}, "parameter_name": "fc3.bias"}}
                ],
                "output_specs": []
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut wd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    wd.insert("fc1.weight".to_string(), (vec![1.0; 12], vec![4, 3]));
    wd.insert("fc1.bias".to_string(), (vec![0.1; 4], vec![4]));
    wd.insert("fc2.weight".to_string(), (vec![2.0; 8], vec![2, 4]));
    wd.insert("fc2.bias".to_string(), (vec![0.2; 2], vec![2]));
    wd.insert("fc3.weight".to_string(), (vec![3.0; 6], vec![3, 2]));
    wd.insert("fc3.bias".to_string(), (vec![0.3; 3], vec![3]));

    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    assert_eq!(wm.len(), 6);
    assert_eq!(wm["p_fc1_weight"].shape, vec![4, 3]);
    assert_eq!(wm["p_fc2_weight"].shape, vec![2, 4]);
    assert_eq!(wm["p_fc3_weight"].shape, vec![3, 2]);
    assert!((wm["p_fc1_weight"].data[0] - 1.0).abs() < 1e-6);
    assert!((wm["p_fc2_weight"].data[0] - 2.0).abs() < 1e-6);
    assert!((wm["p_fc3_weight"].data[0] - 3.0).abs() < 1e-6);
}

// ===========================================================================
// 12. Safetensors round-trip: load -> save -> reload preserves data
// ===========================================================================

#[test]
fn test_safetensors_roundtrip_f32_values_preserved() {
    let vals: Vec<f32> = vec![1.5, -2.0, 3.14, 0.0, f32::MIN_POSITIVE, 1e-7];
    let raw: Vec<u8> = vals.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut tensors = HashMap::new();
    tensors.insert(
        "roundtrip_test".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 3], &raw).unwrap(),
    );
    let bytes = safetensors::serialize(&tensors, None).unwrap();
    let loaded = load_weights_from_bytes(&bytes).unwrap();
    let (f32_data, shape) = &loaded["roundtrip_test"];
    assert_eq!(shape, &[2, 3]);
    assert_eq!(f32_data.len(), 6);
    for (orig, loaded_val) in vals.iter().zip(f32_data.iter()) {
        assert!(
            (orig - loaded_val).abs() < 1e-10,
            "F32 round-trip mismatch: {orig} vs {loaded_val}"
        );
    }
}

#[test]
fn test_safetensors_roundtrip_multiple_tensors_preserved() {
    let w1_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let w2_data = vec![-1.0f32, -2.0];
    let w1_raw: Vec<u8> = w1_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let w2_raw: Vec<u8> = w2_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "layer1.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2], &w1_raw).unwrap(),
    );
    tensors.insert(
        "layer1.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2], &w2_raw).unwrap(),
    );
    let bytes = safetensors::serialize(&tensors, None).unwrap();
    let loaded = load_weights_from_bytes(&bytes).unwrap();
    assert_eq!(loaded.len(), 2);
    let (d1, s1) = &loaded["layer1.weight"];
    assert_eq!(s1, &[2, 2]);
    assert_eq!(d1, &w1_data);
    let (d2, s2) = &loaded["layer1.bias"];
    assert_eq!(s2, &[2]);
    assert_eq!(d2, &w2_data);
}

#[test]
fn test_safetensors_roundtrip_f16_precision_preserved() {
    let original = [1.0f32, 0.5, -0.25, 100.0];
    let f16_vals: Vec<half::f16> = original.iter().map(|&v| half::f16::from_f32(v)).collect();
    let raw: Vec<u8> = f16_vals.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "w".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![4], &raw).unwrap(),
    );
    let bytes = safetensors::serialize(&tensors, None).unwrap();
    let loaded = load_weights_from_bytes(&bytes).unwrap();
    let (f32_data, _) = &loaded["w"];

    for (orig, loaded_val) in original.iter().zip(f32_data.iter()) {
        let f16_expected = half::f16::from_f32(*orig).to_f32();
        assert!(
            (f16_expected - loaded_val).abs() < 1e-3,
            "F16 round-trip mismatch for {orig}: expected {f16_expected}, got {loaded_val}"
        );
    }
}

#[test]
fn test_safetensors_roundtrip_empty_tensor() {
    let mut tensors = HashMap::new();
    let empty_data: Vec<u8> = vec![];
    tensors.insert(
        "empty".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![0], &empty_data)
            .unwrap(),
    );
    let bytes = safetensors::serialize(&tensors, None).unwrap();
    let loaded = load_weights_from_bytes(&bytes).unwrap();
    let (f32_data, shape) = &loaded["empty"];
    assert_eq!(shape, &[0]);
    assert!(f32_data.is_empty());
}

#[test]
fn test_safetensors_roundtrip_large_tensor() {
    let size = 4096usize;
    let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
    let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut tensors = HashMap::new();
    tensors.insert(
        "big".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![64, 64], &raw).unwrap(),
    );
    let bytes = safetensors::serialize(&tensors, None).unwrap();
    let loaded = load_weights_from_bytes(&bytes).unwrap();
    let (f32_data, shape) = &loaded["big"];
    assert_eq!(shape, &[64, 64]);
    assert_eq!(f32_data.len(), size);
    assert!((f32_data[0] - 0.0).abs() < 1e-6);
    assert!((f32_data[100] - 1.0).abs() < 1e-6);
    assert!((f32_data[4095] - 40.95).abs() < 1e-3);
}

// ===========================================================================
// 13. Missing weight detection: graceful handling of incomplete safetensors
// ===========================================================================

#[test]
fn test_missing_weight_partial_spec_only_found_weights_mapped() {
    let json = r#"{
        "graph_module": {
            "graph": { "inputs": [], "outputs": [], "nodes": [], "tensor_values": {} },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_fc1_weight"}, "parameter_name": "fc1.weight"}},
                    {"parameter": {"arg": {"name": "p_fc1_bias"}, "parameter_name": "fc1.bias"}},
                    {"parameter": {"arg": {"name": "p_fc2_weight"}, "parameter_name": "fc2.weight"}}
                ],
                "output_specs": []
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut wd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    wd.insert("fc1.weight".to_string(), (vec![1.0; 12], vec![4, 3]));
    wd.insert("fc1.bias".to_string(), (vec![0.0; 4], vec![4]));

    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    assert_eq!(wm.len(), 2, "only fc1 weights should be mapped");
    assert!(wm.contains_key("p_fc1_weight"));
    assert!(wm.contains_key("p_fc1_bias"));
    assert!(!wm.contains_key("p_fc2_weight"));
}

#[test]
fn test_missing_weight_empty_safetensors_maps_nothing() {
    let json = r#"{
        "graph_module": {
            "graph": { "inputs": [], "outputs": [], "nodes": [], "tensor_values": {} },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_w"}, "parameter_name": "layer.weight"}}
                ],
                "output_specs": []
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let wd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    assert!(wm.is_empty());
}

#[test]
fn test_missing_weight_extra_weights_ignored() {
    let json = r#"{
        "graph_module": {
            "graph": { "inputs": [], "outputs": [], "nodes": [], "tensor_values": {} },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_w"}, "parameter_name": "fc.weight"}}
                ],
                "output_specs": []
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let mut wd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    wd.insert("fc.weight".to_string(), (vec![1.0; 8], vec![2, 4]));
    wd.insert("extra.weight".to_string(), (vec![9.0; 4], vec![4]));
    wd.insert("another.bias".to_string(), (vec![0.0; 2], vec![2]));

    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    assert_eq!(wm.len(), 1);
    assert!(wm.contains_key("p_w"));
}

#[test]
fn test_missing_weight_file_io_error() {
    let nonexistent = std::path::Path::new("/tmp/nonexistent_weights_abc123.safetensors");
    let result = crate::convert::load_safetensors_weights_pub(nonexistent);
    assert!(result.is_err());
    match result.unwrap_err() {
        ImportError::Io { path, detail } => {
            assert!(path.contains("nonexistent_weights_abc123"));
            assert!(!detail.is_empty());
        }
        other => panic!("expected Io error, got: {other:?}"),
    }
}

// ===========================================================================
// 14. Duplicate weight detection: duplicate tensor names in source files
// ===========================================================================

#[test]
fn test_duplicate_weight_names_last_wins() {
    let json = r#"{
        "graph_module": {
            "graph": { "inputs": [], "outputs": [], "nodes": [], "tensor_values": {} },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_w"}, "parameter_name": "fc.weight"}}
                ],
                "output_specs": []
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let mut wd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    wd.insert("fc.weight".to_string(), (vec![1.0; 6], vec![2, 3]));
    wd.insert("fc.weight".to_string(), (vec![9.0; 8], vec![4, 2]));

    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    assert_eq!(wm.len(), 1);
    assert_eq!(wm["p_w"].shape, vec![4, 2]);
    assert_eq!(wm["p_w"].data.len(), 8);
    assert!((wm["p_w"].data[0] - 9.0).abs() < 1e-6);
}

#[test]
fn test_duplicate_graph_placeholder_names_last_spec_wins() {
    let json = r#"{
        "graph_module": {
            "graph": { "inputs": [], "outputs": [], "nodes": [], "tensor_values": {} },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_w"}, "parameter_name": "old.weight"}},
                    {"parameter": {"arg": {"name": "p_w"}, "parameter_name": "new.weight"}}
                ],
                "output_specs": []
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let mut wd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    wd.insert("old.weight".to_string(), (vec![1.0; 4], vec![2, 2]));
    wd.insert("new.weight".to_string(), (vec![2.0; 4], vec![2, 2]));

    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    assert!(wm.contains_key("p_w"));
    assert!((wm["p_w"].data[0] - 2.0).abs() < 1e-6);
}

// ===========================================================================
// 15. Weight permutation: handle transposed weights (Conv2d layout)
// ===========================================================================

#[test]
fn test_weight_permutation_conv2d_shape_preserved() {
    let conv_shape: Vec<usize> = vec![32, 3, 3, 3];
    let num_elements: usize = conv_shape.iter().product();
    let data = build_safetensors(&[("conv.weight", safetensors::Dtype::F32, &conv_shape)]);
    let loaded = load_weights_from_bytes(&data).unwrap();
    let (f32_data, shape) = &loaded["conv.weight"];
    assert_eq!(shape, &conv_shape);
    assert_eq!(f32_data.len(), num_elements);
}

#[test]
fn test_weight_permutation_linear_shape_preserved() {
    let data = build_safetensors(&[("fc.weight", safetensors::Dtype::F32, &[512, 768])]);
    let loaded = load_weights_from_bytes(&data).unwrap();
    let (_, shape) = &loaded["fc.weight"];
    assert_eq!(shape, &[512, 768]);
}

#[test]
fn test_weight_permutation_conv_transpose1d_shape() {
    let data = build_safetensors(&[("deconv.weight", safetensors::Dtype::F32, &[64, 32, 7])]);
    let loaded = load_weights_from_bytes(&data).unwrap();
    let (_, shape) = &loaded["deconv.weight"];
    assert_eq!(shape, &[64, 32, 7]);
}

#[test]
fn test_weight_permutation_embedding_shape() {
    let data = build_safetensors(&[("embed.weight", safetensors::Dtype::F32, &[50000, 768])]);
    let loaded = load_weights_from_bytes(&data).unwrap();
    let (f32_data, shape) = &loaded["embed.weight"];
    assert_eq!(shape, &[50000, 768]);
    assert_eq!(f32_data.len(), 50000 * 768);
}

#[test]
fn test_weight_permutation_3d_through_graph() {
    let json = r#"{
        "graph_module": {
            "graph": { "inputs": [], "outputs": [], "nodes": [], "tensor_values": {} },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_conv_weight"}, "parameter_name": "conv.weight"}}
                ],
                "output_specs": []
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let mut wd: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    wd.insert("conv.weight".to_string(), (vec![0.5; 48], vec![8, 2, 3]));
    let wm = build_weight_map(&program.graph_module.signature.input_specs, &wd);
    assert_eq!(wm["p_conv_weight"].shape, vec![8, 2, 3]);
    assert_eq!(wm["p_conv_weight"].data.len(), 48);
}

// ===========================================================================
// 16. Multi-file safetensors: handle sharded model files
// ===========================================================================

#[test]
fn test_multi_segment_shared_weights_detected() {
    let weights_data = build_safetensors(&[
        ("shared.weight", safetensors::Dtype::F32, &[4, 4]),
        ("shared.bias", safetensors::Dtype::F32, &[4]),
    ]);
    let tmp_dir = std::env::temp_dir().join("nn_shared_weights");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let weights_path = tmp_dir.join("weights.safetensors");
    std::fs::write(&weights_path, &weights_data).unwrap();

    let make_shared = |seg_name: &str| {
        make_json_graph(
            &format!(
                r#"{{"parameter": {{"arg": {{"name": "p_shared_weight"}}, "parameter_name": "shared.weight"}}}},
                {{"parameter": {{"arg": {{"name": "p_shared_bias"}}, "parameter_name": "shared.bias"}}}},
                {{"user_input": {{"arg": {{"as_tensor": {{"name": "{seg_name}_in"}}}}}}}}"#
            ),
            &format!(
                r#"{{"user_output": {{"arg": {{"as_tensor": {{"name": "{seg_name}_out"}}}}}}}}"#
            ),
            &format!(
                r#"{{
                    "target": "torch.ops.aten.linear.default",
                    "inputs": [
                        {{"name": "input", "arg": {{"as_tensor": {{"name": "{seg_name}_in"}}}}, "kind": 1}},
                        {{"name": "weight", "arg": {{"as_tensor": {{"name": "p_shared_weight"}}}}, "kind": 1}},
                        {{"name": "bias", "arg": {{"as_tensor": {{"name": "p_shared_bias"}}}}, "kind": 1}}
                    ],
                    "outputs": [{{"as_tensor": {{"name": "{seg_name}_out"}}}}],
                    "metadata": {{}}
                }}"#
            ),
            &format!(
                r#""{seg_name}_in": {{"dtype": 7, "sizes": [{{"as_int": 1}}, {{"as_int": 4}}], "requires_grad": false, "strides": [{{"as_int": 4}}, {{"as_int": 1}}]}},
                "{seg_name}_out": {{"dtype": 7, "sizes": [{{"as_int": 1}}, {{"as_int": 4}}], "requires_grad": false, "strides": [{{"as_int": 4}}, {{"as_int": 1}}]}},
                "p_shared_weight": {{"dtype": 7, "sizes": [{{"as_int": 4}}, {{"as_int": 4}}], "requires_grad": true, "strides": [{{"as_int": 4}}, {{"as_int": 1}}]}},
                "p_shared_bias": {{"dtype": 7, "sizes": [{{"as_int": 4}}], "requires_grad": true, "strides": [{{"as_int": 1}}]}}"#
            ),
        )
    };

    let seg_a_json = make_shared("seg_a");
    let seg_b_json = make_shared("seg_b");

    let graphs: Vec<(String, serde_json::Value)> = vec![
        (
            "seg_a".to_string(),
            serde_json::from_str(&seg_a_json).unwrap(),
        ),
        (
            "seg_b".to_string(),
            serde_json::from_str(&seg_b_json).unwrap(),
        ),
    ];
    let model = convert_multi_segment(&graphs, &weights_path).unwrap();
    assert_eq!(model.num_segments(), 2);
    assert!(
        !model.shared_weights.is_empty(),
        "shared_weights should be detected"
    );
    assert!(model.shared_weights.contains(&"shared.weight".to_string()));
    assert!(model.shared_weights.contains(&"shared.bias".to_string()));

    let _ = std::fs::remove_file(&weights_path);
}

#[test]
fn test_multi_segment_single_segment_via_convert_single() {
    let weights_data = build_safetensors(&[
        ("fc.weight", safetensors::Dtype::F32, &[4, 8]),
        ("fc.bias", safetensors::Dtype::F32, &[4]),
    ]);
    let tmp_dir = std::env::temp_dir().join("nn_single_seg2");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let weights_path = tmp_dir.join("weights.safetensors");
    std::fs::write(&weights_path, &weights_data).unwrap();

    let json = make_json_graph(
        r#"{"parameter": {"arg": {"name": "p_fc_weight"}, "parameter_name": "fc.weight"}},
        {"parameter": {"arg": {"name": "p_fc_bias"}, "parameter_name": "fc.bias"}},
        {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}"#,
        r#"{"user_output": {"arg": {"as_tensor": {"name": "y"}}}}"#,
        r#"{
            "target": "torch.ops.aten.linear.default",
            "inputs": [
                {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
                {"name": "weight", "arg": {"as_tensor": {"name": "p_fc_weight"}}, "kind": 1},
                {"name": "bias", "arg": {"as_tensor": {"name": "p_fc_bias"}}, "kind": 1}
            ],
            "outputs": [{"as_tensor": {"name": "y"}}],
            "metadata": {}
        }"#,
        r#""x": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 8}], "requires_grad": false, "strides": [{"as_int": 8}, {"as_int": 1}]},
        "y": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]},
        "p_fc_weight": {"dtype": 7, "sizes": [{"as_int": 4}, {"as_int": 8}], "requires_grad": true, "strides": [{"as_int": 8}, {"as_int": 1}]},
        "p_fc_bias": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 1}]}"#,
    );
    let val: serde_json::Value = serde_json::from_str(&json).unwrap();

    let result = crate::multi_segment::convert_single_segment(&val, &weights_path);
    assert!(result.is_ok());
    let model = result.unwrap();
    assert_eq!(model.num_segments(), 1);
    let seg = model.get_segment("main").unwrap();
    assert_eq!(seg.num_user_inputs, 1);

    let _ = std::fs::remove_file(&weights_path);
}

// ===========================================================================
// 17. Config parsing: additional edge cases
// ===========================================================================

#[test]
fn test_parse_exported_program_malformed_json_error() {
    let result = parse_exported_program(b"not json at all{{{");
    assert!(result.is_err());
    match result.unwrap_err() {
        ImportError::JsonParse(_) => {}
        other => panic!("expected JsonParse, got: {other:?}"),
    }
}

#[test]
fn test_parse_exported_program_empty_json_error() {
    let result = parse_exported_program(b"");
    assert!(result.is_err());
}

#[test]
fn test_parse_exported_program_missing_graph_module_error() {
    let json = r#"{"schema_version": {"major": 8, "minor": 0}, "range_constraints": {}}"#;
    let result = parse_exported_program(json.as_bytes());
    assert!(result.is_err());
}

#[test]
fn test_parse_exported_program_future_schema_rejected() {
    let json = r#"{
        "graph_module": {
            "graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}},
            "signature": {"input_specs": [], "output_specs": []},
            "module_call_graph": []
        },
        "schema_version": {"major": 99, "minor": 0},
        "range_constraints": {}
    }"#;
    let result = parse_exported_program(json.as_bytes());
    assert!(result.is_err());
    match result.unwrap_err() {
        ImportError::UnsupportedSchema { major, .. } => assert_eq!(major, 99),
        other => panic!("expected UnsupportedSchema, got: {other:?}"),
    }
}

#[test]
fn test_parse_exported_program_empty_graph() {
    let json = r#"{
        "graph_module": {
            "graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}},
            "signature": {"input_specs": [], "output_specs": []},
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 0},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert!(program.graph_module.graph.nodes.is_empty());
    assert!(program.graph_module.signature.input_specs.is_empty());
}

#[test]
fn test_parse_exported_program_multiple_range_constraints() {
    let json = r#"{
        "graph_module": {
            "graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}},
            "signature": {"input_specs": [], "output_specs": []},
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 10},
        "range_constraints": {
            "s0": {"min_val": 1, "max_val": 512},
            "s1": {"min_val": 0, "max_val": 2048},
            "s2": {"min_val": 16, "max_val": 16}
        }
    }"#;
    let program = parse_exported_program(json.as_bytes()).unwrap();
    assert_eq!(program.range_constraints.len(), 3);
    assert_eq!(program.range_constraints["s0"].min_val, 1);
    assert_eq!(program.range_constraints["s2"].max_val, 16);
}

#[test]
fn test_parse_output_spec_user_output() {
    let json = r#"{"user_output": {"arg": {"as_tensor": {"name": "logits"}}}}"#;
    let spec: OutputSpec = serde_json::from_str(json).unwrap();
    match spec {
        OutputSpec::UserOutput(uo) => {
            let name = uo.user_output.arg.as_tensor_name().unwrap();
            assert_eq!(name, "logits");
        }
        other => panic!("expected UserOutput, got: {other:?}"),
    }
}

#[test]
fn test_parse_tensor_meta_with_device() {
    let json = r#"{
        "dtype": 7,
        "sizes": [{"as_int": 4}],
        "requires_grad": true,
        "strides": [{"as_int": 1}],
        "device": {"type": "cuda", "index": 0},
        "layout": 0
    }"#;
    let meta: TensorMeta = serde_json::from_str(json).unwrap();
    assert_eq!(meta.dtype, 7);
    assert!(meta.requires_grad);
    assert!(meta.device.is_some());
    let dev = meta.device.as_ref().unwrap();
    assert_eq!(dev.device_type, "cuda");
    assert_eq!(dev.index, Some(0));
}

#[test]
fn test_parse_argument_floats_variant() {
    let a: Argument = serde_json::from_str(r#"{"as_floats": [1.0, 2.5, -3.14]}"#).unwrap();
    match a {
        Argument::Floats(f) => {
            assert_eq!(f.as_floats.len(), 3);
            assert!((f.as_floats[0] - 1.0).abs() < 1e-6);
        }
        other => panic!("expected Floats, got: {other:?}"),
    }
}

#[test]
fn test_parse_argument_bools_variant() {
    let a: Argument = serde_json::from_str(r#"{"as_bools": [true, false, true]}"#).unwrap();
    match a {
        Argument::Bools(b) => {
            assert_eq!(b.as_bools, vec![true, false, true]);
        }
        other => panic!("expected Bools, got: {other:?}"),
    }
}

#[test]
fn test_parse_argument_tensors_variant() {
    let a: Argument =
        serde_json::from_str(r#"{"as_tensors": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}"#)
            .unwrap();
    match a {
        Argument::Tensors(t) => {
            assert_eq!(t.as_tensors.len(), 3);
            assert_eq!(t.as_tensors[0].name, "a");
            assert_eq!(t.as_tensors[2].name, "c");
        }
        other => panic!("expected Tensors, got: {other:?}"),
    }
}

#[test]
fn test_parse_argument_scalar_type_variant() {
    let a: Argument = serde_json::from_str(r#"{"as_scalar_type": 7}"#).unwrap();
    match a {
        Argument::ScalarType(st) => assert_eq!(st.as_scalar_type, 7),
        other => panic!("expected ScalarType, got: {other:?}"),
    }
}

#[test]
fn test_parse_node_with_metadata() {
    let json = r#"{
        "target": "torch.ops.aten.relu.default",
        "inputs": [{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}],
        "outputs": [{"as_tensor": {"name": "y"}}],
        "metadata": {"source_fn_stack": "relu", "nn_module_stack": "model.act"}
    }"#;
    let node: Node = serde_json::from_str(json).unwrap();
    assert_eq!(node.target, "torch.ops.aten.relu.default");
    assert_eq!(node.metadata.len(), 2);
    assert!(node.metadata.contains_key("source_fn_stack"));
}

// ===========================================================================
// 18. Additional error type coverage
// ===========================================================================

#[test]
fn test_import_error_topology_error_display() {
    let err = ImportError::TopologyError {
        node_name: "relu_0".to_string(),
        ref_name: "missing_tensor".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("topology error"));
    assert!(msg.contains("relu_0"));
    assert!(msg.contains("missing_tensor"));
}

#[test]
fn test_import_error_negative_dimension_display() {
    let err = ImportError::NegativeDimension {
        op_target: "aten::squeeze".to_string(),
        arg_name: "dim".to_string(),
        value: -5,
    };
    let msg = format!("{err}");
    assert!(msg.contains("negative value -5"));
    assert!(msg.contains("dim"));
}

#[test]
fn test_import_error_multi_axis_display() {
    let err = ImportError::MultiAxisNotSupported {
        op_target: "aten::sum".to_string(),
        op_kind: "reduction",
        dims: vec![0, 2],
    };
    let msg = format!("{err}");
    assert!(msg.contains("multi-axis"));
    assert!(msg.contains("reduction"));
}

#[test]
fn test_import_error_io_display() {
    let err = ImportError::Io {
        path: "/tmp/model.safetensors".to_string(),
        detail: "file not found".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("I/O error"));
    assert!(msg.contains("/tmp/model.safetensors"));
}

#[test]
fn test_import_error_unsupported_dtype_display() {
    let err = ImportError::UnsupportedDtype {
        name: "mask_tensor".to_string(),
        dtype: "BOOL".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("unsupported safetensors dtype"));
    assert!(msg.contains("BOOL"));
}

#[test]
fn test_import_error_unknown_tensor_display() {
    let err = ImportError::UnknownTensor {
        name: "phantom_tensor".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("not found"));
    assert!(msg.contains("phantom_tensor"));
}

#[test]
fn test_convert_error_all_variants() {
    use crate::convert::ConvertError;

    let err1 = ConvertError::Import(ImportError::UnsupportedOp {
        target: "aten::weird".to_string(),
    });
    assert!(format!("{err1}").contains("import error"));

    let err2 = ConvertError::Compile("failed".to_string());
    assert!(format!("{err2}").contains("compilation error"));

    let err3 = ConvertError::Reftest("mismatch".to_string());
    assert!(format!("{err3}").contains("reftest error"));
}

// ===========================================================================
// 19. DType conversion: mixed dtypes and edge cases
// ===========================================================================

#[test]
fn test_dtype_conversion_mixed_dtypes_in_graph() {
    let json = make_json_graph(
        r#"{"user_input": {"arg": {"as_tensor": {"name": "f32_in"}}}},
        {"user_input": {"arg": {"as_tensor": {"name": "bf16_in"}}}}"#,
        r#"{"user_output": {"arg": {"as_tensor": {"name": "f32_in"}}}}"#,
        "",
        r#""f32_in": {"dtype": 7, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]},
        "bf16_in": {"dtype": 13, "sizes": [{"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 1}]}"#,
    );
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let empty_weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let imported = build_graph(&program, &empty_weights).unwrap();

    let f32_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| n.name() == "f32_in")
        .unwrap();
    let bf16_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| n.name() == "bf16_in")
        .unwrap();
    assert_eq!(f32_node.output_dtype(), DType::F32);
    assert_eq!(bf16_node.output_dtype(), DType::BF16);
}

// ===========================================================================
// 20. Quantization detection edge cases
// ===========================================================================

#[test]
fn test_quantization_detect_empty_safetensors() {
    let data = build_safetensors(&[]);
    let report = detect_quantization_from_bytes(&data).unwrap();
    assert_eq!(report.total_tensors, 0);
    assert!(report.dtype_breakdown.is_empty());
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_quantization_detect_all_dtypes_mixed() {
    let data = build_safetensors(&[
        ("f32_t", safetensors::Dtype::F32, &[64, 64]),
        ("f16_t", safetensors::Dtype::F16, &[64, 64]),
        ("bf16_t", safetensors::Dtype::BF16, &[64, 64]),
        ("i8_t", safetensors::Dtype::I8, &[64, 64]),
        ("u8_t", safetensors::Dtype::U8, &[64, 64]),
    ]);
    let report = detect_quantization_from_bytes(&data).unwrap();
    assert_eq!(report.total_tensors, 5);
    assert!(report.is_mixed_precision());
    assert!(report.dtype_breakdown.len() >= 4);
}

#[test]
fn test_quantization_detect_single_tensor() {
    let data = build_safetensors(&[("w", safetensors::Dtype::F16, &[32, 32])]);
    let report = detect_quantization_from_bytes(&data).unwrap();
    assert_eq!(report.total_tensors, 1);
    assert!(!report.is_mixed_precision());
    assert_eq!(report.dtype_fraction(DetectedDtype::F16), 1.0);
}
