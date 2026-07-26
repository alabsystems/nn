// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Wave 14 aten op mappers (common missing PyTorch ops).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;

use crate::op_map::{map_node_to_trace_op, supported_ops, OpMapContext, ResolvedWeight};
use crate::parse::{
    Argument, ArgumentBool, ArgumentFloat, ArgumentInt, ArgumentNone, ArgumentTensor,
    NamedArgument, Node, TensorArgument, TensorMeta,
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
// lerp.Scalar
// =======================================================================

#[test]
fn test_map_lerp_scalar() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.lerp.Scalar",
        vec![
            named("self", tensor_arg("a")),
            named("end", tensor_arg("b")),
            named("weight", float_arg(0.3)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "lerp_scalar_0.3"),
        "expected lerp_scalar custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["a", "b"]);
}

// =======================================================================
// lerp.Tensor
// =======================================================================

#[test]
fn test_map_lerp_tensor() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.lerp.Tensor",
        vec![
            named("self", tensor_arg("a")),
            named("end", tensor_arg("b")),
            named("weight", tensor_arg("w")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "lerp_tensor"),
        "expected lerp_tensor custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["a", "b", "w"]);
}

// =======================================================================
// addcmul.default
// =======================================================================

#[test]
fn test_map_addcmul() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.addcmul.default",
        vec![
            named("self", tensor_arg("x")),
            named("tensor1", tensor_arg("t1")),
            named("tensor2", tensor_arg("t2")),
            named("value", float_arg(2.0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "addcmul_v2"),
        "expected addcmul custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "t1", "t2"]);
}

#[test]
fn test_map_addcmul_default_value() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.addcmul.default",
        vec![
            named("self", tensor_arg("x")),
            named("tensor1", tensor_arg("t1")),
            named("tensor2", tensor_arg("t2")),
        ],
    );
    let (op, _inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "addcmul_v1"),
        "expected addcmul with default value=1, got: {op:?}"
    );
}

// =======================================================================
// addcdiv.default
// =======================================================================

#[test]
fn test_map_addcdiv() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.addcdiv.default",
        vec![
            named("self", tensor_arg("x")),
            named("tensor1", tensor_arg("t1")),
            named("tensor2", tensor_arg("t2")),
            named("value", float_arg(0.001)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "addcdiv_v0.001"),
        "expected addcdiv custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "t1", "t2"]);
}

// =======================================================================
// linalg_vector_norm.default
// =======================================================================

#[test]
fn test_map_linalg_vector_norm() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.linalg_vector_norm.default",
        vec![
            named("self", tensor_arg("x")),
            named("ord", float_arg(2.0)),
            named("dim", none_arg()),
            named("keepdim", bool_arg(false)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("linalg_vector_norm_")),
        "expected linalg_vector_norm custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// cdist.default
// =======================================================================

#[test]
fn test_map_cdist() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cdist.default",
        vec![
            named("x1", tensor_arg("a")),
            named("x2", tensor_arg("b")),
            named("p", float_arg(2.0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "cdist_p2"),
        "expected cdist custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["a", "b"]);
}

// =======================================================================
// multinomial.default
// =======================================================================

#[test]
fn test_map_multinomial() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.multinomial.default",
        vec![
            named("self", tensor_arg("probs")),
            named("num_samples", int_arg(10)),
            named("replacement", bool_arg(true)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "multinomial_n10_repltrue"),
        "expected multinomial custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["probs"]);
}

// =======================================================================
// searchsorted.Tensor
// =======================================================================

#[test]
fn test_map_searchsorted() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.searchsorted.Tensor",
        vec![
            named("sorted_sequence", tensor_arg("sorted")),
            named("self", tensor_arg("vals")),
            named("right", bool_arg(true)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "searchsorted_righttrue"),
        "expected searchsorted custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["sorted", "vals"]);
}

// =======================================================================
// bucketize.Tensor
// =======================================================================

#[test]
fn test_map_bucketize() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.bucketize.Tensor",
        vec![
            named("self", tensor_arg("x")),
            named("boundaries", tensor_arg("edges")),
            named("right", bool_arg(false)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "bucketize_rightfalse"),
        "expected bucketize custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "edges"]);
}

// =======================================================================
// count_nonzero.default
// =======================================================================

#[test]
fn test_map_count_nonzero() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.count_nonzero.default",
        vec![named("self", tensor_arg("x")), named("dim", none_arg())],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "count_nonzero_all"),
        "expected count_nonzero custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_count_nonzero_with_dim() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.count_nonzero.default",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(1))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "count_nonzero_dim1"),
        "expected count_nonzero with dim, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// cumprod.default
// =======================================================================

#[test]
fn test_map_cumprod() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cumprod.default",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(0))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "cumprod_dim0"),
        "expected cumprod custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// cummax.default
// =======================================================================

#[test]
fn test_map_cummax() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cummax.default",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(1))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "cummax_dim1"),
        "expected cummax custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// cummin.default
// =======================================================================

#[test]
fn test_map_cummin() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cummin.default",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(2))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "cummin_dim2"),
        "expected cummin custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// one_hot.default
// =======================================================================

#[test]
fn test_map_one_hot() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.one_hot.default",
        vec![
            named("self", tensor_arg("labels")),
            named("num_classes", int_arg(10)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "one_hot_nc10"),
        "expected one_hot custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["labels"]);
}

#[test]
fn test_map_one_hot_inferred() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.one_hot.default",
        vec![named("self", tensor_arg("labels"))],
    );
    let (op, _inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "one_hot_nc-1"),
        "expected one_hot with inferred num_classes, got: {op:?}"
    );
}

// =======================================================================
// threshold.default / threshold_.default
// =======================================================================

#[test]
fn test_map_threshold() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.threshold.default",
        vec![
            named("self", tensor_arg("x")),
            named("threshold", float_arg(0.5)),
            named("value", float_arg(-1.0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "threshold_t0.5_v-1"),
        "expected threshold custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_threshold_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.threshold_.default",
        vec![
            named("self", tensor_arg("x")),
            named("threshold", float_arg(0.0)),
            named("value", float_arg(0.0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "threshold_t0_v0"),
        "expected threshold inplace custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// count_nonzero.dim_IntList
// =======================================================================

#[test]
fn test_map_count_nonzero_dim_intlist() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.count_nonzero.dim_IntList",
        vec![
            named("self", tensor_arg("x")),
            named(
                "dim",
                Argument::Ints(crate::parse::ArgumentInts {
                    as_ints: vec![0, 2],
                }),
            ),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "count_nonzero_dim0_2"),
        "expected count_nonzero multi-dim custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// supported_ops includes Wave 14 entries
// =======================================================================

#[test]
fn test_supported_ops_includes_wave14() {
    let ops = supported_ops();
    for expected in &[
        "aten::lerp",
        "aten::addcmul",
        "aten::addcdiv",
        "aten::linalg_vector_norm",
        "aten::cdist",
        "aten::multinomial",
        "aten::searchsorted",
        "aten::bucketize",
        "aten::count_nonzero",
        "aten::cumprod",
        "aten::cummax",
        "aten::cummin",
        "aten::one_hot",
        "aten::threshold",
    ] {
        assert!(
            ops.contains(expected),
            "supported_ops should include {expected}"
        );
    }
}
