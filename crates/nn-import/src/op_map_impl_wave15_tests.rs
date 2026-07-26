// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Wave 15 aten op mappers.

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
// clamp.Tensor
// =======================================================================

#[test]
fn test_map_clamp_tensor_min_max() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.clamp.Tensor",
        vec![
            named("self", tensor_arg("x")),
            named("min", tensor_arg("lo")),
            named("max", tensor_arg("hi")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "clamp_tensor_min_max"),
        "expected clamp_tensor_min_max, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "lo", "hi"]);
}

#[test]
fn test_map_clamp_tensor_min_only() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.clamp.Tensor",
        vec![
            named("self", tensor_arg("x")),
            named("min", tensor_arg("lo")),
            named("max", none_arg()),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "clamp_tensor_min_only"),
        "expected clamp_tensor_min_only, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "lo"]);
}

// =======================================================================
// norm.ScalarOpt_dim
// =======================================================================

#[test]
fn test_map_norm_default() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.norm.ScalarOpt_dim",
        vec![
            named("self", tensor_arg("x")),
            named("p", float_arg(2.0)),
            named("dim", ints_arg(vec![1])),
            named("keepdim", bool_arg(false)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("norm_p2")),
        "expected norm custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// einsum.default
// =======================================================================

#[test]
fn test_map_einsum() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.einsum.default",
        vec![
            named("equation", string_arg("ij,jk->ik")),
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "einsum_ij,jk->ik"),
        "expected einsum custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["a", "b"]);
}

// =======================================================================
// as_strided.default
// =======================================================================

#[test]
fn test_map_as_strided() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.as_strided.default",
        vec![
            named("self", tensor_arg("x")),
            named("size", ints_arg(vec![2, 3])),
            named("stride", ints_arg(vec![3, 1])),
            named("storage_offset", int_arg(0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "as_strided_sz2x3_st3x1_off0"),
        "expected as_strided custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// addmv.default
// =======================================================================

#[test]
fn test_map_addmv() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.addmv.default",
        vec![
            named("self", tensor_arg("bias")),
            named("mat", tensor_arg("weight")),
            named("vec", tensor_arg("input")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "addmv_b1_a1"),
        "expected addmv custom op with default scalars, got: {op:?}"
    );
    assert_eq!(inputs, vec!["bias", "weight", "input"]);
}

#[test]
fn test_map_addmv_scaled() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.addmv.default",
        vec![
            named("self", tensor_arg("bias")),
            named("mat", tensor_arg("weight")),
            named("vec", tensor_arg("input")),
            named("beta", float_arg(0.5)),
            named("alpha", float_arg(2.0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "addmv_b0.5_a2"),
        "expected addmv custom op with scaled params, got: {op:?}"
    );
    assert_eq!(inputs, vec!["bias", "weight", "input"]);
}

// =======================================================================
// addr.default
// =======================================================================

#[test]
fn test_map_addr() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.addr.default",
        vec![
            named("self", tensor_arg("m")),
            named("vec1", tensor_arg("v1")),
            named("vec2", tensor_arg("v2")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "addr_b1_a1"),
        "expected addr custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["m", "v1", "v2"]);
}

// =======================================================================
// outer.default
// =======================================================================

#[test]
fn test_map_outer() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.outer.default",
        vec![
            named("self", tensor_arg("a")),
            named("vec2", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "outer"),
        "expected outer custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["a", "b"]);
}

// =======================================================================
// bernoulli.default
// =======================================================================

#[test]
fn test_map_bernoulli() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.bernoulli.default",
        vec![named("self", tensor_arg("probs"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "bernoulli"),
        "expected bernoulli custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["probs"]);
}

// =======================================================================
// bernoulli_.float
// =======================================================================

#[test]
fn test_map_bernoulli_float() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.bernoulli_.float",
        vec![named("self", tensor_arg("x")), named("p", float_arg(0.1))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "bernoulli_p0.1"),
        "expected bernoulli_ with p=0.1, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// randn.default
// =======================================================================

#[test]
fn test_map_randn() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.randn.default",
        vec![named("size", ints_arg(vec![2, 3, 4]))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "randn_2x3x4"),
        "expected randn custom op, got: {op:?}"
    );
    assert!(inputs.is_empty(), "randn has no tensor inputs");
}

// =======================================================================
// cross.default
// =======================================================================

#[test]
fn test_map_cross() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cross.default",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
            named("dim", int_arg(1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "cross_dim1"),
        "expected cross custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["a", "b"]);
}

#[test]
fn test_map_cross_auto_dim() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cross.default",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "cross_auto"),
        "expected cross with auto dim, got: {op:?}"
    );
    assert_eq!(inputs, vec!["a", "b"]);
}

// =======================================================================
// supported_ops includes Wave 15 entries
// =======================================================================

#[test]
fn test_supported_ops_includes_wave15() {
    let ops = supported_ops();
    for expected in &[
        "aten::clamp_tensor",
        "aten::norm",
        "aten::einsum",
        "aten::as_strided",
        "aten::addmv",
        "aten::addr",
        "aten::outer",
        "aten::bernoulli",
        "aten::randn",
        "aten::cross",
    ] {
        assert!(
            ops.contains(expected),
            "supported_ops should include {expected}"
        );
    }
}
