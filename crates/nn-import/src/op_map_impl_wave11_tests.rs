// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Wave 11 aten op mappers.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;

use crate::op_map::{map_node_to_trace_op, supported_ops, OpMapContext, ResolvedWeight};
use crate::parse::{
    Argument, ArgumentBool, ArgumentFloat, ArgumentInt, ArgumentInts, ArgumentNone, ArgumentTensor,
    ArgumentTensors, NamedArgument, Node, TensorArgument, TensorMeta,
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

fn tensors_arg(names: &[&str]) -> Argument {
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

#[allow(dead_code)]
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
// affine_grid_generator
// =======================================================================

#[test]
fn test_map_affine_grid_generator() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.affine_grid_generator.default",
        vec![
            named("self", tensor_arg("theta")),
            named("size", ints_arg(&[2, 3, 64, 64])),
            named("align_corners", bool_arg(false)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "affine_grid_generator_h64_w64_alignfalse"),
        "expected affine_grid_generator custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["theta"]);
}

#[test]
fn test_map_affine_grid_generator_align_corners() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.affine_grid_generator.default",
        vec![
            named("self", tensor_arg("theta")),
            named("size", ints_arg(&[1, 1, 32, 48])),
            named("align_corners", bool_arg(true)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "affine_grid_generator_h32_w48_aligntrue"),
        "expected affine_grid_generator with align_corners=true, got: {op:?}"
    );
    assert_eq!(inputs, vec!["theta"]);
}

#[test]
fn test_map_affine_grid_generator_wrong_size_len() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.affine_grid_generator.default",
        vec![
            named("self", tensor_arg("theta")),
            named("size", ints_arg(&[2, 3, 64])), // 3D instead of 4D
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_err(), "should reject non-4D size argument");
}

// =======================================================================
// meshgrid direct mapper
// =======================================================================

#[test]
fn test_map_meshgrid_default() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.meshgrid.default",
        vec![named("tensors", tensors_arg(&["x", "y"]))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "meshgrid_ij_n2"),
        "expected meshgrid custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "y"]);
}

#[test]
fn test_map_meshgrid_xy_indexing() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.meshgrid.indexing",
        vec![
            named("tensors", tensors_arg(&["a", "b", "c"])),
            named(
                "indexing",
                Argument::Str(crate::parse::ArgumentString {
                    as_string: "xy".to_string(),
                }),
            ),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "meshgrid_xy_n3"),
        "expected meshgrid with xy indexing, got: {op:?}"
    );
    assert_eq!(inputs, vec!["a", "b", "c"]);
}

// =======================================================================
// stack direct mapper
// =======================================================================

#[test]
fn test_map_stack_default() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.stack.default",
        vec![
            named("tensors", tensors_arg(&["t0", "t1", "t2"])),
            named("dim", int_arg(0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "stack_dim0_n3"),
        "expected stack custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["t0", "t1", "t2"]);
}

#[test]
fn test_map_stack_dim1() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.stack.default",
        vec![
            named("tensors", tensors_arg(&["a", "b"])),
            named("dim", int_arg(1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "stack_dim1_n2"),
        "expected stack dim=1, got: {op:?}"
    );
    assert_eq!(inputs, vec!["a", "b"]);
}

// =======================================================================
// split / chunk direct mappers
// =======================================================================

#[test]
fn test_map_split() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.split.Tensor",
        vec![
            named("self", tensor_arg("x")),
            named("split_size", int_arg(4)),
            named("dim", int_arg(1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "split_size4_dim1"),
        "expected split custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_split_with_sizes() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.split_with_sizes.default",
        vec![
            named("self", tensor_arg("x")),
            named("split_sizes", ints_arg(&[2, 3, 5])),
            named("dim", int_arg(0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "split_sizes_2_3_5_dim0"),
        "expected split_with_sizes custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_chunk() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.chunk.default",
        vec![
            named("self", tensor_arg("x")),
            named("chunks", int_arg(3)),
            named("dim", int_arg(0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "chunk_n3_dim0"),
        "expected chunk custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// masked_fill scalar (direct mapper)
// =======================================================================

#[test]
fn test_map_masked_fill_scalar_direct() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.masked_fill.Scalar",
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

#[test]
fn test_map_masked_fill_inplace_scalar_direct() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.masked_fill_.Scalar",
        vec![
            named("self", tensor_arg("x")),
            named("mask", tensor_arg("m")),
            named("value", int_arg(0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("masked_fill_scalar_")),
        "expected masked_fill_ scalar custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "m"]);
}

// =======================================================================
// triu_ / tril_ in-place
// =======================================================================

#[test]
fn test_map_triu_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.triu_.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Triu { diagonal: 0 }),
        "expected Triu with default diagonal, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_triu_inplace_diagonal() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.triu_.default",
        vec![
            named("self", tensor_arg("x")),
            named("diagonal", int_arg(1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Triu { diagonal: 1 }),
        "expected Triu with diagonal=1, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_tril_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.tril_.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Tril { diagonal: 0 }),
        "expected Tril with default diagonal, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_tril_inplace_diagonal() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.tril_.default",
        vec![
            named("self", tensor_arg("x")),
            named("diagonal", int_arg(-2)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Tril { diagonal: -2 }),
        "expected Tril with diagonal=-2, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// arange.start_stop
// =======================================================================

#[test]
fn test_map_arange_start_stop() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.arange.start_stop",
        vec![named("start", int_arg(0)), named("end", int_arg(10))],
    );
    let (op, _inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match &op {
        TraceOp::Arange { start, end, step } => {
            assert!((start - 0.0).abs() < 1e-9);
            assert!((end - 10.0).abs() < 1e-9);
            assert!((step - 1.0).abs() < 1e-9);
        }
        _ => panic!("expected Arange, got: {op:?}"),
    }
}

#[test]
fn test_map_arange_start_stop_float() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.arange.start_stop",
        vec![named("start", float_arg(0.5)), named("end", float_arg(5.5))],
    );
    let (op, _inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match &op {
        TraceOp::Arange { start, end, step } => {
            assert!((start - 0.5).abs() < 1e-9);
            assert!((end - 5.5).abs() < 1e-9);
            assert!((step - 1.0).abs() < 1e-9);
        }
        _ => panic!("expected Arange, got: {op:?}"),
    }
}

// =======================================================================
// linspace.out
// =======================================================================

#[test]
fn test_map_linspace_out() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.linspace.out",
        vec![
            named("start", float_arg(0.0)),
            named("end", float_arg(1.0)),
            named("steps", int_arg(5)),
            named("out", tensor_arg("buf")),
        ],
    );
    let (op, _inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match &op {
        TraceOp::Arange { start, end, step } => {
            assert!((start - 0.0).abs() < 1e-9);
            // step = 1.0/4 = 0.25, end should be ~1.0 + 0.125
            assert!((step - 0.25).abs() < 1e-9);
            assert!(*end > 1.0);
        }
        _ => panic!("expected Arange for linspace.out, got: {op:?}"),
    }
}

// =======================================================================
// supported_ops includes Wave 11
// =======================================================================

#[test]
fn test_supported_ops_includes_wave11() {
    let ops = supported_ops();
    for expected in &["aten::affine_grid_generator", "aten::triu_", "aten::tril_"] {
        assert!(
            ops.contains(expected),
            "supported_ops should include {expected}"
        );
    }
}
