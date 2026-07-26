// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Wave 13 aten op mappers (advanced tensor manipulation and control flow).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;

use crate::op_map::{map_node_to_trace_op, supported_ops, OpMapContext, ResolvedWeight};
use crate::parse::{
    Argument, ArgumentBool, ArgumentFloat, ArgumentInt, ArgumentNone, ArgumentString,
    ArgumentTensor, ArgumentTensors, NamedArgument, Node, TensorArgument, TensorMeta,
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

fn tensor_list_arg(names: &[&str]) -> Argument {
    Argument::Tensors(ArgumentTensors {
        as_tensors: names
            .iter()
            .map(|n| TensorArgument {
                name: n.to_string(),
            })
            .collect(),
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
// index_put hacked_twin (overwrite mode)
// =======================================================================

#[test]
fn test_map_index_put_hacked_twin_overwrite() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.index_put.hacked_twin",
        vec![
            named("self", tensor_arg("x")),
            named("indices", tensor_list_arg(&["idx0"])),
            named("values", tensor_arg("vals")),
            named("accumulate", bool_arg(false)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::IndexPut { dim: 0 }),
        "expected IndexPut with dim=0, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "idx0", "vals"]);
}

// =======================================================================
// index_put hacked_twin (accumulate mode)
// =======================================================================

#[test]
fn test_map_index_put_hacked_twin_accumulate() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.index_put_.hacked_twin",
        vec![
            named("self", tensor_arg("x")),
            named("indices", tensor_list_arg(&["idx0"])),
            named("values", tensor_arg("vals")),
            named("accumulate", bool_arg(true)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("index_put_accumulate")),
        "expected index_put_accumulate custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "idx0", "vals"]);
}

// =======================================================================
// scatter_.value_reduce
// =======================================================================

#[test]
fn test_map_scatter_value_reduce() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.scatter_.value_reduce",
        vec![
            named("self", tensor_arg("x")),
            named("dim", int_arg(1)),
            named("index", tensor_arg("idx")),
            named("value", float_arg(1.0)),
            named("reduce", string_arg("add")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "scatter_value_reduce_dim1_add_v1"),
        "expected scatter_value_reduce custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "idx"]);
}

// =======================================================================
// scatter_add_ (in-place)
// =======================================================================

#[test]
fn test_map_scatter_add_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.scatter_add_.default",
        vec![
            named("self", tensor_arg("x")),
            named("dim", int_arg(0)),
            named("index", tensor_arg("idx")),
            named("src", tensor_arg("s")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::ScatterAdd { dim: 0 }),
        "expected ScatterAdd with dim=0, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "idx", "s"]);
}

// =======================================================================
// gather.out
// =======================================================================

#[test]
fn test_map_gather_out() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.gather.out",
        vec![
            named("self", tensor_arg("x")),
            named("dim", int_arg(2)),
            named("index", tensor_arg("idx")),
            named("sparse_grad", bool_arg(false)),
            named("out", tensor_arg("out_buf")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Gather { dim: 2 }),
        "expected Gather with dim=2, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "idx"]);
}

// =======================================================================
// index_select.out
// =======================================================================

#[test]
fn test_map_index_select_out() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.index_select.out",
        vec![
            named("self", tensor_arg("x")),
            named("dim", int_arg(1)),
            named("index", tensor_arg("idx")),
            named("out", tensor_arg("out_buf")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::IndexSelect { dim: 1 }),
        "expected IndexSelect with dim=1, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "idx"]);
}

// =======================================================================
// masked_fill.Tensor_Scalar
// =======================================================================

#[test]
fn test_map_masked_fill_tensor_scalar() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.masked_fill.Tensor_Scalar",
        vec![
            named("self", tensor_arg("x")),
            named("mask", tensor_arg("m")),
            named("value", float_arg(-1e9)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("masked_fill_scalar_")),
        "expected masked_fill_scalar custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "m"]);
}

// =======================================================================
// masked_select.default
// =======================================================================

#[test]
fn test_map_masked_select() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.masked_select.default",
        vec![
            named("self", tensor_arg("x")),
            named("mask", tensor_arg("m")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "masked_select"),
        "expected masked_select custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "m"]);
}

// =======================================================================
// masked_select.out
// =======================================================================

#[test]
fn test_map_masked_select_out() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.masked_select.out",
        vec![
            named("self", tensor_arg("x")),
            named("mask", tensor_arg("m")),
            named("out", tensor_arg("out_buf")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "masked_select"),
        "expected masked_select custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "m"]);
}

// =======================================================================
// nonzero.default
// =======================================================================

#[test]
fn test_map_nonzero() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.nonzero.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "nonzero"),
        "expected nonzero custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// nonzero.out
// =======================================================================

#[test]
fn test_map_nonzero_out() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.nonzero.out",
        vec![
            named("self", tensor_arg("x")),
            named("out", tensor_arg("out_buf")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "nonzero"),
        "expected nonzero custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// topk.values
// =======================================================================

#[test]
fn test_map_topk_values() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.topk.values",
        vec![
            named("self", tensor_arg("x")),
            named("k", int_arg(5)),
            named("dim", int_arg(1)),
            named("largest", bool_arg(true)),
            named("sorted", bool_arg(true)),
            named("values", tensor_arg("v_buf")),
            named("indices", tensor_arg("i_buf")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Topk { k: 5, dim: 1 }),
        "expected Topk with k=5 dim=1, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// sort.values / sort.values_stable
// =======================================================================

#[test]
fn test_map_sort_values() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sort.values",
        vec![
            named("self", tensor_arg("x")),
            named("dim", int_arg(0)),
            named("descending", bool_arg(true)),
            named("values", tensor_arg("v_buf")),
            named("indices", tensor_arg("i_buf")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            &op,
            TraceOp::Sort {
                dim: 0,
                descending: true
            }
        ),
        "expected Sort with dim=0 descending=true, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_sort_values_stable() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sort.values_stable",
        vec![
            named("self", tensor_arg("x")),
            named("dim", int_arg(2)),
            named("descending", bool_arg(false)),
            named("values", tensor_arg("v_buf")),
            named("indices", tensor_arg("i_buf")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            &op,
            TraceOp::Sort {
                dim: 2,
                descending: false
            }
        ),
        "expected Sort with dim=2 descending=false, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// _unique2.default
// =======================================================================

#[test]
fn test_map_unique2() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten._unique2.default",
        vec![
            named("self", tensor_arg("x")),
            named("sorted", bool_arg(true)),
            named("return_inverse", bool_arg(true)),
            named("return_counts", bool_arg(false)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "unique_sortedtrue_invtrue_cntfalse"),
        "expected unique custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// unique_dim.default
// =======================================================================

#[test]
fn test_map_unique_dim() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.unique_dim.default",
        vec![
            named("self", tensor_arg("x")),
            named("dim", int_arg(0)),
            named("sorted", bool_arg(true)),
            named("return_inverse", bool_arg(false)),
            named("return_counts", bool_arg(true)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "unique_dim0_sortedtrue_invfalse_cnttrue"),
        "expected unique_dim custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// unique_consecutive.default
// =======================================================================

#[test]
fn test_map_unique_consecutive() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.unique_consecutive.default",
        vec![
            named("self", tensor_arg("x")),
            named("return_inverse", bool_arg(false)),
            named("return_counts", bool_arg(false)),
            named("dim", none_arg()),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "unique_consecutive_flat_invfalse_cntfalse"),
        "expected unique_consecutive custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_unique_consecutive_with_dim() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.unique_consecutive.default",
        vec![
            named("self", tensor_arg("x")),
            named("return_inverse", bool_arg(true)),
            named("return_counts", bool_arg(true)),
            named("dim", int_arg(1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "unique_consecutive_dim1_invtrue_cnttrue"),
        "expected unique_consecutive with dim=1, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// supported_ops includes Wave 13 entries
// =======================================================================

#[test]
fn test_supported_ops_includes_wave13() {
    let ops = supported_ops();
    for expected in &[
        "aten::masked_select",
        "aten::nonzero",
        "aten::unique",
        "aten::unique_consecutive",
        "aten::index_put_hacked_twin",
    ] {
        assert!(
            ops.contains(expected),
            "supported_ops should include {expected}"
        );
    }
}
