// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Wave 16 aten op mappers.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;

use crate::op_map::{map_node_to_trace_op, supported_ops, OpMapContext, ResolvedWeight};
use crate::parse::{
    Argument, ArgumentBool, ArgumentFloat, ArgumentInt, ArgumentInts, ArgumentNone, ArgumentString,
    ArgumentTensor, NamedArgument, Node, TensorArgument, TensorMeta,
};

fn empty_ctx() -> OpMapContext<'static> {
    let meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::default());
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::default());
    OpMapContext {
        tensor_meta: meta,
        weights,
    }
}

fn ctx_with_weights(entries: Vec<(&str, Vec<f32>, Vec<usize>)>) -> OpMapContext<'static> {
    let meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::default());
    let mut weights = HashMap::new();
    for (name, data, shape) in entries {
        weights.insert(name.to_string(), ResolvedWeight::new(data, shape));
    }
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::new(weights));
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

fn float_arg(val: f64) -> Argument {
    Argument::Float(ArgumentFloat { as_float: val })
}

fn bool_arg(val: bool) -> Argument {
    Argument::Bool(ArgumentBool { as_bool: val })
}

fn none_arg() -> Argument {
    Argument::None(ArgumentNone { as_none: true })
}

fn string_arg(val: &str) -> Argument {
    Argument::Str(ArgumentString {
        as_string: val.to_string(),
    })
}

fn ints_arg(vals: Vec<i64>) -> Argument {
    Argument::Ints(ArgumentInts { as_ints: vals })
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

// =======================================================================
// relu_.default
// =======================================================================

#[test]
fn test_map_relu_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.relu_.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(&op, TraceOp::Relu), "expected Relu, got: {op:?}");
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// sigmoid_.default
// =======================================================================

#[test]
fn test_map_sigmoid_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sigmoid_.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Sigmoid),
        "expected Sigmoid, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// tanh_.default
// =======================================================================

#[test]
fn test_map_tanh_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.tanh_.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(&op, TraceOp::Tanh), "expected Tanh, got: {op:?}");
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// silu_.default
// =======================================================================

#[test]
fn test_map_silu_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.silu_.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(&op, TraceOp::Silu), "expected Silu, got: {op:?}");
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// gelu_.default
// =======================================================================

#[test]
fn test_map_gelu_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.gelu_.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(&op, TraceOp::Gelu), "expected Gelu, got: {op:?}");
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// native_layer_norm.default (with affine params)
// =======================================================================

#[test]
fn test_map_native_layer_norm_affine() {
    let ctx = ctx_with_weights(vec![
        ("ln_weight", vec![1.0; 4], vec![4]),
        ("ln_bias", vec![0.0; 4], vec![4]),
    ]);
    let node = simple_node(
        "torch.ops.aten.native_layer_norm.default",
        vec![
            named("input", tensor_arg("x")),
            named("normalized_shape", ints_arg(vec![4])),
            named("weight", tensor_arg("ln_weight")),
            named("bias", tensor_arg("ln_bias")),
            named("eps", float_arg(1e-5)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::LayerNorm { eps, .. } if (*eps - 1e-5).abs() < 1e-10),
        "expected LayerNorm with eps=1e-5, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// native_layer_norm.default (no affine)
// =======================================================================

#[test]
fn test_map_native_layer_norm_no_affine() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.native_layer_norm.default",
        vec![
            named("input", tensor_arg("x")),
            named("normalized_shape", ints_arg(vec![4])),
            named("weight", none_arg()),
            named("bias", none_arg()),
            named("eps", float_arg(1e-6)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("native_layer_norm_no_affine")),
        "expected native_layer_norm_no_affine custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// native_group_norm.default (with affine params)
// =======================================================================

#[test]
fn test_map_native_group_norm_affine() {
    let ctx = ctx_with_weights(vec![
        ("gn_weight", vec![1.0; 8], vec![8]),
        ("gn_bias", vec![0.0; 8], vec![8]),
    ]);
    let node = simple_node(
        "torch.ops.aten.native_group_norm.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("gn_weight")),
            named("bias", tensor_arg("gn_bias")),
            named("N", int_arg(2)),
            named("C", int_arg(8)),
            named("HxW", int_arg(16)),
            named("group", int_arg(4)),
            named("eps", float_arg(1e-5)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::GroupNorm { num_groups: 4, .. }),
        "expected GroupNorm with num_groups=4, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// native_group_norm.default (no affine)
// =======================================================================

#[test]
fn test_map_native_group_norm_no_affine() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.native_group_norm.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", none_arg()),
            named("bias", none_arg()),
            named("N", int_arg(2)),
            named("C", int_arg(8)),
            named("HxW", int_arg(16)),
            named("group", int_arg(4)),
            named("eps", float_arg(1e-5)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("native_group_norm_no_affine")),
        "expected native_group_norm_no_affine custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// gru.input
// =======================================================================

#[test]
fn test_map_gru_default() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.gru.input",
        vec![
            named("input", tensor_arg("x")),
            named("hx", tensor_arg("h0")),
            named("has_biases", bool_arg(true)),
            named("num_layers", int_arg(1)),
            named("dropout", float_arg(0.0)),
            named("bidirectional", bool_arg(false)),
            named("batch_first", bool_arg(true)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("gru_")),
        "expected gru custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "h0"]);
}

#[test]
fn test_map_gru_bidirectional() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.gru.input",
        vec![
            named("input", tensor_arg("x")),
            named("hx", tensor_arg("h0")),
            named("has_biases", bool_arg(true)),
            named("num_layers", int_arg(2)),
            named("dropout", float_arg(0.1)),
            named("bidirectional", bool_arg(true)),
            named("batch_first", bool_arg(false)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.contains("bidir") && name.contains("layers2")),
        "expected gru with bidirectional and 2 layers, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "h0"]);
}

// =======================================================================
// view_as_real.default
// =======================================================================

#[test]
fn test_map_view_as_real() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.view_as_real.default",
        vec![named("self", tensor_arg("z"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "view_as_real"),
        "expected view_as_real custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["z"]);
}

// =======================================================================
// view_as_complex.default
// =======================================================================

#[test]
fn test_map_view_as_complex() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.view_as_complex.default",
        vec![named("self", tensor_arg("r"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "view_as_complex"),
        "expected view_as_complex custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["r"]);
}

// =======================================================================
// fft_rfft.default
// =======================================================================

#[test]
fn test_map_fft_rfft_default() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.fft_rfft.default",
        vec![
            named("self", tensor_arg("x")),
            named("n", none_arg()),
            named("dim", int_arg(-1)),
            named("norm", none_arg()),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("fft_rfft_")),
        "expected fft_rfft custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_fft_rfft_with_n() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.fft_rfft.default",
        vec![
            named("self", tensor_arg("x")),
            named("n", int_arg(512)),
            named("dim", int_arg(-1)),
            named("norm", string_arg("ortho")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.contains("n512") && name.contains("ortho")),
        "expected fft_rfft with n=512 and ortho norm, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// fft_irfft.default
// =======================================================================

#[test]
fn test_map_fft_irfft_default() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.fft_irfft.default",
        vec![
            named("self", tensor_arg("x")),
            named("n", none_arg()),
            named("dim", int_arg(-1)),
            named("norm", none_arg()),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("fft_irfft_")),
        "expected fft_irfft custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_fft_irfft_with_n() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.fft_irfft.default",
        vec![
            named("self", tensor_arg("x")),
            named("n", int_arg(1024)),
            named("dim", int_arg(1)),
            named("norm", string_arg("forward")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.contains("n1024") && name.contains("dim1")),
        "expected fft_irfft with n=1024 and dim=1, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// feature_dropout.default
// =======================================================================

#[test]
fn test_map_feature_dropout() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.feature_dropout.default",
        vec![
            named("input", tensor_arg("x")),
            named("p", float_arg(0.5)),
            named("train", bool_arg(false)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Dropout),
        "expected Dropout (identity at inference), got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// alpha_dropout.default
// =======================================================================

#[test]
fn test_map_alpha_dropout() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.alpha_dropout.default",
        vec![
            named("input", tensor_arg("x")),
            named("p", float_arg(0.1)),
            named("train", bool_arg(false)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Dropout),
        "expected Dropout (identity at inference), got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// supported_ops includes Wave 16 entries
// =======================================================================

#[test]
fn test_supported_ops_includes_wave16() {
    let ops = supported_ops();
    for expected in &[
        "aten::relu_",
        "aten::sigmoid_",
        "aten::tanh_",
        "aten::silu_",
        "aten::gelu_",
        "aten::native_layer_norm",
        "aten::native_group_norm",
        "aten::gru",
        "aten::view_as_real",
        "aten::view_as_complex",
        "aten::fft_rfft",
        "aten::fft_irfft",
        "aten::feature_dropout",
        "aten::alpha_dropout",
    ] {
        assert!(
            ops.contains(expected),
            "supported_ops should include {expected}"
        );
    }
}
