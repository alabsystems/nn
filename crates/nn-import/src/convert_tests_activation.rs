// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for activation and elementwise op mappings added in Wave 10.
//!
//! Covers: rsqrt, hardtanh, hardsigmoid, hardswish, selu, softplus, mish, celu.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;

use crate::op_map::{map_node_to_trace_op, OpMapContext, ResolvedWeight};
use crate::parse::{
    Argument, ArgumentFloat, ArgumentTensor, NamedArgument, Node, TensorArgument, TensorMeta,
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

fn float_arg(val: f64) -> Argument {
    Argument::Float(ArgumentFloat { as_float: val })
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

// -- rsqrt → Powf { exponent: -0.5 } --

#[test]
fn test_map_rsqrt() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.rsqrt.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Powf { exponent } if (exponent - (-0.5)).abs() < 1e-10),
        "expected Powf {{ exponent: -0.5 }}, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// -- hardtanh → Clamp { min, max } --

#[test]
fn test_map_hardtanh_defaults() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.hardtanh.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match op {
        TraceOp::Clamp {
            min: Some(min_val),
            max: Some(max_val),
        } => {
            assert!(
                (min_val - (-1.0)).abs() < 1e-10,
                "expected min=-1.0, got {min_val}"
            );
            assert!(
                (max_val - 1.0).abs() < 1e-10,
                "expected max=1.0, got {max_val}"
            );
        }
        _ => panic!("expected Clamp {{ min: Some(-1.0), max: Some(1.0) }}, got: {op:?}"),
    }
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_hardtanh_custom_bounds() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.hardtanh.default",
        vec![
            named("self", tensor_arg("x")),
            named("min_val", float_arg(-3.0)),
            named("max_val", float_arg(3.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match op {
        TraceOp::Clamp {
            min: Some(min_val),
            max: Some(max_val),
        } => {
            assert!(
                (min_val - (-3.0)).abs() < 1e-10,
                "expected min=-3.0, got {min_val}"
            );
            assert!(
                (max_val - 3.0).abs() < 1e-10,
                "expected max=3.0, got {max_val}"
            );
        }
        _ => panic!("expected Clamp with custom bounds, got: {op:?}"),
    }
}

#[test]
fn test_map_hardtanh_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.hardtanh_.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::Clamp {
                min: Some(_),
                max: Some(_)
            }
        ),
        "in-place hardtanh_ should map to Clamp, got: {op:?}"
    );
}

// -- hardsigmoid → HardSigmoid --

#[test]
fn test_map_hardsigmoid() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.hardsigmoid.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::HardSigmoid),
        "expected HardSigmoid, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// -- hardswish → HardSwish --

#[test]
fn test_map_hardswish() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.hardswish.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::HardSwish),
        "expected HardSwish, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_hardswish_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.hardswish_.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::HardSwish),
        "in-place hardswish_ should map to HardSwish, got: {op:?}"
    );
}

// -- selu → Selu --

#[test]
fn test_map_selu() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.selu.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Selu), "expected Selu, got: {op:?}");
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_selu_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.selu_.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Selu),
        "in-place selu_ should map to Selu, got: {op:?}"
    );
}

// -- softplus → Softplus --

#[test]
fn test_map_softplus() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.softplus.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Softplus),
        "expected Softplus, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// -- mish → Mish --

#[test]
fn test_map_mish() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.mish.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Mish), "expected Mish, got: {op:?}");
    assert_eq!(inputs, vec!["x"]);
}

// -- celu → Celu { alpha } --

#[test]
fn test_map_celu_default_alpha() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.celu.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Celu { alpha } if (alpha - 1.0).abs() < 1e-10),
        "expected Celu {{ alpha: 1.0 }}, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_celu_custom_alpha() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.celu.default",
        vec![
            named("self", tensor_arg("x")),
            named("alpha", float_arg(0.5)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Celu { alpha } if (alpha - 0.5).abs() < 1e-10),
        "expected Celu {{ alpha: 0.5 }}, got: {op:?}"
    );
}

#[test]
fn test_map_celu_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.celu_.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Celu { alpha } if (alpha - 1.0).abs() < 1e-10),
        "in-place celu_ should map to Celu, got: {op:?}"
    );
}
