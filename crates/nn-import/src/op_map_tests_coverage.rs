// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive op mapping coverage tests for nn-import.
//!
//! Tests every mapped aten op to verify correct TraceOp translation,
//! including unary ops, binary ops, activations, shape ops, reductions,
//! comparison ops, padding, upsampling, dpdf/kokoro-specific ops,
//! and expansion paths (flatten, select, chunk, split, unbind, stack,
//! masked_fill, index_tensor).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;
use nn_core::DType;

use super::*;
use crate::parse::{
    Argument, ArgumentBool, ArgumentFloat, ArgumentInt, ArgumentInts, ArgumentNone, ArgumentString,
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

fn str_arg(val: &str) -> Argument {
    Argument::Str(ArgumentString {
        as_string: val.to_string(),
    })
}

fn none_arg() -> Argument {
    Argument::None(ArgumentNone { as_none: true })
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

fn ctx_with_weights(entries: &[(&str, Vec<f32>, Vec<usize>)]) -> OpMapContext<'static> {
    let meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::default());
    let mut weights = HashMap::new();
    for (name, data, shape) in entries {
        weights.insert(
            name.to_string(),
            ResolvedWeight::new(data.clone(), shape.clone()),
        );
    }
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::new(weights));
    OpMapContext {
        tensor_meta: meta,
        weights,
    }
}

// ============================================================
// Unary element-wise ops
// ============================================================

#[test]
fn test_map_silu() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.silu.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Silu));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_tanh() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.tanh.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Tanh));
}

#[test]
fn test_map_sigmoid() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sigmoid.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Sigmoid));
}

#[test]
fn test_map_exp() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.exp.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Exp));
}

#[test]
fn test_map_log() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.log.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Log));
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
fn test_map_reciprocal() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.reciprocal.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Recip));
}

#[test]
fn test_map_sin() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sin.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Sin));
}

#[test]
fn test_map_cos() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cos.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Cos));
}

#[test]
fn test_map_floor() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.floor.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Floor));
}

#[test]
fn test_map_round() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.round.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Round));
}

#[test]
fn test_map_rsqrt_decomposes_to_powf() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.rsqrt.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Powf { exponent } if (exponent - (-0.5)).abs() < 1e-10),
        "rsqrt should map to Powf(-0.5), got: {op:?}"
    );
}

// ============================================================
// Binary element-wise ops
// ============================================================

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
fn test_map_maximum() {
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
}

#[test]
fn test_map_minimum() {
    let ctx = empty_ctx();
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
fn test_map_bmm() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.bmm.default",
        vec![
            named("self", tensor_arg("a")),
            named("mat2", tensor_arg("b")),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::MatMul));
}

#[test]
fn test_map_matmul() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.matmul.default",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::MatMul));
}

// ============================================================
// Shape operations
// ============================================================

#[test]
fn test_map_permute() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.permute.default",
        vec![
            named("input", tensor_arg("x")),
            named("dims", ints_arg(&[0, 2, 1])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Permute { ref axes } if *axes == vec![0, 2, 1]),
        "expected Permute([0,2,1]), got: {op:?}"
    );
}

#[test]
fn test_map_unsqueeze() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.unsqueeze.default",
        vec![named("input", tensor_arg("x")), named("dim", int_arg(1))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Unsqueeze { dim: 1 }));
}

#[test]
fn test_map_flip() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.flip.default",
        vec![
            named("input", tensor_arg("x")),
            named("dims", ints_arg(&[2])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Flip { dim: 2 }));
}

#[test]
fn test_map_slice() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.slice.Tensor",
        vec![
            named("input", tensor_arg("x")),
            named("dim", int_arg(1)),
            named("start", int_arg(2)),
            named("end", int_arg(5)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::Narrow {
                dim: 1,
                start: 2,
                length: 3
            }
        ),
        "expected Narrow(dim=1, start=2, length=3), got: {op:?}"
    );
}

#[test]
fn test_map_slice_no_end() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.slice.Tensor",
        vec![
            named("input", tensor_arg("x")),
            named("dim", int_arg(0)),
            named("start", int_arg(3)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Narrow { dim: 0, start: 3, length } if length == usize::MAX),
        "expected Narrow with max length when end is absent, got: {op:?}"
    );
}

#[test]
fn test_map_slice_i64_max_end_is_open_ended() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.slice.Tensor",
        vec![
            named("input", tensor_arg("x")),
            named("dim", int_arg(1)),
            named("start", int_arg(11)),
            named("end", int_arg(i64::MAX)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Narrow { dim: 1, start: 11, length } if length == usize::MAX),
        "expected Narrow with max length for i64::MAX end sentinel, got: {op:?}"
    );
}

#[test]
fn test_map_reshape_alias() {
    // _unsafe_view should map to the same Reshape as view.default
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten._unsafe_view.default",
        vec![
            named("input", tensor_arg("x")),
            named("size", ints_arg(&[6, 4])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Reshape { target_shape } if target_shape == vec![6, 4]));
}

// ============================================================
// Activation ops
// ============================================================

#[test]
fn test_map_elu() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.elu.default",
        vec![
            named("input", tensor_arg("x")),
            named("alpha", float_arg(1.5)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Elu { alpha } if (alpha - 1.5).abs() < 1e-10),
        "expected Elu(alpha=1.5), got: {op:?}"
    );
}

#[test]
fn test_map_elu_default_alpha() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.elu.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Elu { alpha } if (alpha - 1.0).abs() < 1e-10),
        "expected Elu(alpha=1.0 default), got: {op:?}"
    );
}

#[test]
fn test_map_leaky_relu() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.leaky_relu.default",
        vec![
            named("input", tensor_arg("x")),
            named("negative_slope", float_arg(0.2)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::LeakyRelu { slope } if (slope - 0.2).abs() < 1e-10),
        "expected LeakyRelu(slope=0.2), got: {op:?}"
    );
}

#[test]
fn test_map_hardtanh() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.hardtanh.default",
        vec![
            named("input", tensor_arg("x")),
            named("min_val", float_arg(-2.0)),
            named("max_val", float_arg(2.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Clamp { min: Some(lo), max: Some(hi) }
            if (lo - (-2.0)).abs() < 1e-10 && (hi - 2.0).abs() < 1e-10),
        "expected Clamp(-2, 2), got: {op:?}"
    );
}

#[test]
fn test_map_hardtanh_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.hardtanh_.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Clamp { min: Some(lo), max: Some(hi) }
            if (lo - (-1.0)).abs() < 1e-10 && (hi - 1.0).abs() < 1e-10),
        "expected Clamp(-1, 1) default, got: {op:?}"
    );
}

#[test]
fn test_map_softplus() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.softplus.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Softplus));
}

#[test]
fn test_map_celu() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.celu.default",
        vec![
            named("input", tensor_arg("x")),
            named("alpha", float_arg(0.75)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Celu { alpha } if (alpha - 0.75).abs() < 1e-10),
        "expected Celu(0.75), got: {op:?}"
    );
}

#[test]
fn test_map_dropout_passthrough() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.dropout.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Dropout));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_hardsigmoid() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.hardsigmoid.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::HardSigmoid));
}

#[test]
fn test_map_hardswish() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.hardswish.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::HardSwish));
}

#[test]
fn test_map_selu() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.selu.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Selu));
}

#[test]
fn test_map_mish() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.mish.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Mish));
}

// ============================================================
// Comparison / Selection ops
// ============================================================

#[test]
fn test_map_where_cond() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.where.self",
        vec![
            named("condition", tensor_arg("mask")),
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::WhereCond));
    assert_eq!(inputs, vec!["mask", "a", "b"]);
}

#[test]
fn test_map_clamp_both() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.clamp.default",
        vec![
            named("input", tensor_arg("x")),
            named("min", float_arg(-1.0)),
            named("max", float_arg(1.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Clamp { min: Some(lo), max: Some(hi) }
            if (lo - (-1.0)).abs() < 1e-10 && (hi - 1.0).abs() < 1e-10),);
}

#[test]
fn test_map_clamp_min_only() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.clamp_min.default",
        vec![
            named("input", tensor_arg("x")),
            named("min", float_arg(0.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Clamp { min: Some(lo), max: None }
            if lo.abs() < 1e-10),
        "expected Clamp(min=0, max=None), got: {op:?}"
    );
}

#[test]
fn test_map_clamp_max() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.clamp_max.default",
        vec![
            named("input", tensor_arg("x")),
            named("max", float_arg(6.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Clamp { min: None, max: Some(hi) }
            if (hi - 6.0).abs() < 1e-10),
        "expected Clamp(min=None, max=6.0), got: {op:?}"
    );
}

// ============================================================
// Attention
// ============================================================

#[test]
fn test_map_log_softmax() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.log_softmax.int",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(2))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::LogSoftmax { dim: 2 }));
}

#[test]
fn test_map_softmax_with_ndim_resolution() {
    // With known ndim, negative dim should resolve.
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten._softmax.default",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(-1))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 3).unwrap();
    assert!(
        matches!(op, TraceOp::Softmax { dim: 2 }),
        "expected dim resolved to 2 (ndim=3, dim=-1), got: {op:?}"
    );
}

// ============================================================
// Reductions
// ============================================================

#[test]
fn test_map_reduce_mean() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.mean.dim",
        vec![
            named("self", tensor_arg("x")),
            named("dim", ints_arg(&[1])),
            named("keepdim", bool_arg(true)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(
        op,
        TraceOp::ReduceMean {
            dim: 1,
            keepdim: true
        }
    ));
}

#[test]
fn test_map_reduce_max() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.amax.default",
        vec![named("self", tensor_arg("x")), named("dim", ints_arg(&[0]))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(
        op,
        TraceOp::ReduceMax {
            dim: 0,
            keepdim: false
        }
    ));
}

#[test]
fn test_map_reduce_min() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.amin.default",
        vec![named("self", tensor_arg("x")), named("dim", ints_arg(&[2]))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(
        op,
        TraceOp::ReduceMin {
            dim: 2,
            keepdim: false
        }
    ));
}

// ============================================================
// Power
// ============================================================

#[test]
fn test_map_powf_generic() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.pow.Tensor_Scalar",
        vec![
            named("self", tensor_arg("x")),
            named("exponent", float_arg(3.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Powf { exponent } if (exponent - 3.0).abs() < 1e-10),
        "expected Powf(3.0), got: {op:?}"
    );
}

#[test]
fn test_map_powf_2_becomes_sqr() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.pow.Tensor_Scalar",
        vec![
            named("self", tensor_arg("x")),
            named("exponent", float_arg(2.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Sqr), "pow(2.0) should become Sqr");
}

#[test]
fn test_map_powf_half_becomes_sqrt() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.pow.Tensor_Scalar",
        vec![
            named("self", tensor_arg("x")),
            named("exponent", float_arg(0.5)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Sqrt), "pow(0.5) should become Sqrt");
}

// ============================================================
// Misc
// ============================================================

#[test]
fn test_map_cumsum() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cumsum.default",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(1))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Cumsum { dim: 1 }));
}

#[test]
fn test_map_zeros() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.zeros.default",
        vec![named("size", ints_arg(&[2, 3]))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Constant { value } if value.abs() < 1e-10),
        "zeros should map to Constant(0.0)"
    );
    assert!(inputs.is_empty());
}

#[test]
fn test_map_zeros_like() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.zeros_like.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Constant { value } if value.abs() < 1e-10));
    assert!(inputs.is_empty());
}

// ============================================================
// Padding ops (Kokoro)
// ============================================================

#[test]
fn test_map_reflection_pad1d() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.reflection_pad1d.default",
        vec![
            named("input", tensor_arg("x")),
            named("padding", ints_arg(&[2, 3])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::ReflectionPad1d {
                pad_left: 2,
                pad_right: 3
            }
        ),
        "expected ReflectionPad1d(2, 3), got: {op:?}"
    );
}

#[test]
fn test_map_constant_pad_nd() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.constant_pad_nd.default",
        vec![
            named("input", tensor_arg("x")),
            named("pad", ints_arg(&[1, 1, 0, 0])),
            named("value", float_arg(0.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::ConstantPadNd { ref padding, value }
            if *padding == vec![1, 1, 0, 0] && value.abs() < 1e-10),
        "expected ConstantPadNd([1,1,0,0], 0.0), got: {op:?}"
    );
}

#[test]
fn test_map_pad_reflect_1d() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.pad.default",
        vec![
            named("input", tensor_arg("x")),
            named("pad", ints_arg(&[3, 3])),
            named("mode", str_arg("reflect")),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::ReflectionPad1d {
                pad_left: 3,
                pad_right: 3
            }
        ),
        "pad(reflect) with 2 values should produce ReflectionPad1d, got: {op:?}"
    );
}

#[test]
fn test_map_pad_reflect_2d() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.pad.default",
        vec![
            named("input", tensor_arg("x")),
            named("pad", ints_arg(&[1, 1, 2, 2])),
            named("mode", str_arg("reflect")),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::ReflectionPad2d {
                pad_left: 1,
                pad_right: 1,
                pad_top: 2,
                pad_bottom: 2
            }
        ),
        "pad(reflect) with 4 values should produce ReflectionPad2d, got: {op:?}"
    );
}

#[test]
fn test_map_pad_unsupported_mode() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.pad.default",
        vec![
            named("input", tensor_arg("x")),
            named("pad", ints_arg(&[1, 1])),
            named("mode", str_arg("replicate")),
        ],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(matches!(err, ImportError::UnsupportedOp { .. }));
}

// ============================================================
// Kokoro-specific ops
// ============================================================

#[test]
fn test_map_index_select() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.index_select.default",
        vec![
            named("self", tensor_arg("embed")),
            named("dim", int_arg(0)),
            named("index", tensor_arg("ids")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::IndexSelect { dim: 0 }));
    assert_eq!(inputs, vec!["embed", "ids"]);
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
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Compare { op: nn_core::dyn_tensor::CompareOp::Gt, value }
            if (value - 0.5).abs() < 1e-10),
        "expected Compare(Gt, 0.5), got: {op:?}"
    );
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
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(
        op,
        TraceOp::Compare {
            op: nn_core::dyn_tensor::CompareOp::Lt,
            ..
        }
    ));
}

#[test]
fn test_map_ge_scalar() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.ge.Scalar",
        vec![
            named("self", tensor_arg("x")),
            named("other", float_arg(0.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(
        op,
        TraceOp::Compare {
            op: nn_core::dyn_tensor::CompareOp::Ge,
            ..
        }
    ));
}

#[test]
fn test_map_le_scalar() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.le.Scalar",
        vec![
            named("self", tensor_arg("x")),
            named("other", float_arg(10.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(
        op,
        TraceOp::Compare {
            op: nn_core::dyn_tensor::CompareOp::Le,
            ..
        }
    ));
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
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(
        op,
        TraceOp::Compare {
            op: nn_core::dyn_tensor::CompareOp::Eq,
            ..
        }
    ));
}

#[test]
fn test_map_ne_scalar() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.ne.Scalar",
        vec![
            named("self", tensor_arg("x")),
            named("other", float_arg(0.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(
        op,
        TraceOp::Compare {
            op: nn_core::dyn_tensor::CompareOp::Ne,
            ..
        }
    ));
}

#[test]
fn test_map_gt_tensor() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.gt.Tensor",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(
        op,
        TraceOp::CompareTensor {
            op: nn_core::dyn_tensor::CompareOp::Gt
        }
    ));
    assert_eq!(inputs, vec!["a", "b"]);
}

#[test]
fn test_map_atan2() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.atan2.default",
        vec![
            named("self", tensor_arg("y")),
            named("other", tensor_arg("x")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Atan2));
    assert_eq!(inputs, vec!["y", "x"]);
}

#[test]
fn test_map_ones() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.ones.default",
        vec![named("size", ints_arg(&[4, 4]))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Constant { value } if (value - 1.0).abs() < 1e-10),
        "ones should map to Constant(1.0)"
    );
}

#[test]
fn test_map_full() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.full.default",
        vec![
            named("size", ints_arg(&[2, 3])),
            named("fill_value", float_arg(3.14)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Constant { value } if (value - 3.14).abs() < 1e-10),
        "expected Constant(3.14), got: {op:?}"
    );
}

#[test]
fn test_map_arange() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.arange.start_step",
        vec![
            named("start", float_arg(0.0)),
            named("end", float_arg(10.0)),
            named("step", float_arg(2.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Arange { start, end, step }
            if start.abs() < 1e-10 && (end - 10.0).abs() < 1e-10 && (step - 2.0).abs() < 1e-10),
        "expected Arange(0, 10, 2), got: {op:?}"
    );
}

#[test]
fn test_map_contiguous_identity() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.contiguous.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Reshape { ref target_shape } if target_shape.is_empty()),
        "contiguous should be identity Reshape, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_clone_identity() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.clone.default",
        vec![named("self", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Reshape { ref target_shape } if target_shape.is_empty()));
}

// ============================================================
// dpdf-specific ops
// ============================================================

#[test]
fn test_map_upsample_nearest2d() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.upsample_nearest2d.default",
        vec![
            named("input", tensor_arg("x")),
            named("scales_h", float_arg(4.0)),
            named("scales_w", float_arg(4.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Upsample2d { scale_h, scale_w, .. }
            if (scale_h - 4.0).abs() < 1e-10 && (scale_w - 4.0).abs() < 1e-10),
        "expected Upsample2d(4x4), got: {op:?}"
    );
}

#[test]
fn test_map_upsample_bilinear2d() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.upsample_bilinear2d.default",
        vec![
            named("input", tensor_arg("x")),
            named("scales_h", float_arg(2.0)),
            named("scales_w", float_arg(3.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Upsample2d { scale_h, scale_w, .. }
            if (scale_h - 2.0).abs() < 1e-10 && (scale_w - 3.0).abs() < 1e-10),);
}

#[test]
fn test_map_triu() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.triu.default",
        vec![
            named("input", tensor_arg("x")),
            named("diagonal", int_arg(1)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Triu { diagonal: 1 }));
}

#[test]
fn test_map_tril() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.tril.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Tril { diagonal: 0 }));
}

#[test]
fn test_map_gather() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.gather.default",
        vec![
            named("self", tensor_arg("src")),
            named("dim", int_arg(1)),
            named("index", tensor_arg("idx")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Gather { dim: 1 }));
    assert_eq!(inputs, vec!["src", "idx"]);
}

#[test]
fn test_map_argmax() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.argmax.default",
        vec![named("input", tensor_arg("x")), named("dim", int_arg(2))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Argmax { dim: 2 }));
}

#[test]
fn test_map_argmin() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.argmin.default",
        vec![named("input", tensor_arg("x")), named("dim", int_arg(0))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Argmin { dim: 0 }));
}

#[test]
fn test_map_pixel_shuffle() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.pixel_shuffle.default",
        vec![
            named("input", tensor_arg("x")),
            named("upscale_factor", int_arg(2)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::PixelShuffle { upscale_factor: 2 }));
}

#[test]
fn test_map_pixel_unshuffle() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.pixel_unshuffle.default",
        vec![
            named("input", tensor_arg("x")),
            named("downscale_factor", int_arg(3)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(
        op,
        TraceOp::PixelUnshuffle {
            downscale_factor: 3
        }
    ));
}

#[test]
fn test_map_repeat() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.repeat.default",
        vec![
            named("input", tensor_arg("x")),
            named("repeats", ints_arg(&[1, 2, 3])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Expand { ref target_shape } if *target_shape == vec![1, 2, 3]),
        "repeat should map to Expand, got: {op:?}"
    );
}

#[test]
fn test_map_scatter() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.scatter.src",
        vec![
            named("self", tensor_arg("dst")),
            named("dim", int_arg(1)),
            named("index", tensor_arg("idx")),
            named("src", tensor_arg("vals")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Scatter { dim: 1 }));
    assert_eq!(inputs, vec!["dst", "idx", "vals"]);
}

#[test]
fn test_map_scatter_add() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.scatter_add.default",
        vec![
            named("self", tensor_arg("dst")),
            named("dim", int_arg(0)),
            named("index", tensor_arg("idx")),
            named("src", tensor_arg("vals")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::ScatterAdd { dim: 0 }));
    assert_eq!(inputs, vec!["dst", "idx", "vals"]);
}

#[test]
fn test_map_narrow() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.narrow.default",
        vec![
            named("input", tensor_arg("x")),
            named("dim", int_arg(1)),
            named("start", int_arg(4)),
            named("length", int_arg(8)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(
        op,
        TraceOp::Narrow {
            dim: 1,
            start: 4,
            length: 8
        }
    ));
}

#[test]
fn test_map_topk() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.topk.default",
        vec![
            named("input", tensor_arg("x")),
            named("k", int_arg(5)),
            named("dim", int_arg(1)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Topk { k: 5, dim: 1 }));
}

#[test]
fn test_map_sort() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sort.default",
        vec![
            named("input", tensor_arg("x")),
            named("dim", int_arg(2)),
            named("descending", bool_arg(true)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(
        op,
        TraceOp::Sort {
            dim: 2,
            descending: true
        }
    ));
}

#[test]
fn test_map_roll() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.roll.default",
        vec![
            named("input", tensor_arg("x")),
            named("shifts", ints_arg(&[3, -2])),
            named("dims", ints_arg(&[0, 1])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Roll { ref shifts, ref dims }
            if *shifts == vec![3, -2] && *dims == vec![0, 1]),
        "expected Roll(shifts=[3,-2], dims=[0,1]), got: {op:?}"
    );
}

#[test]
fn test_map_reflection_pad2d() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.reflection_pad2d.default",
        vec![
            named("input", tensor_arg("x")),
            named("padding", ints_arg(&[1, 1, 2, 2])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(
        op,
        TraceOp::ReflectionPad2d {
            pad_left: 1,
            pad_right: 1,
            pad_top: 2,
            pad_bottom: 2
        }
    ));
}

#[test]
fn test_map_interpolate_nearest() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.interpolate.default",
        vec![
            named("input", tensor_arg("x")),
            named("mode", str_arg("nearest")),
            named("scales_h", float_arg(2.0)),
            named("scales_w", float_arg(2.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Upsample2d { .. }));
}

#[test]
fn test_map_interpolate_bilinear() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.interpolate.default",
        vec![
            named("input", tensor_arg("x")),
            named("mode", str_arg("bilinear")),
            named("scales_h", float_arg(3.0)),
            named("scales_w", float_arg(3.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Upsample2d { .. }));
}

#[test]
fn test_map_interpolate_bicubic() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.interpolate.default",
        vec![
            named("input", tensor_arg("x")),
            named("mode", str_arg("bicubic")),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Upsample2d { .. }));
}

#[test]
fn test_map_interpolate_unsupported_mode() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.interpolate.default",
        vec![
            named("input", tensor_arg("x")),
            named("mode", str_arg("area")),
        ],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(matches!(err, ImportError::UnsupportedOp { .. }));
}

// ============================================================
// Pooling ops
// ============================================================

#[test]
fn test_map_max_pool1d() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.max_pool1d.default",
        vec![
            named("input", tensor_arg("x")),
            named("kernel_size", ints_arg(&[3])),
            named("stride", ints_arg(&[2])),
            named("padding", ints_arg(&[1])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::MaxPool1d {
                kernel_size: 3,
                stride: 2,
                padding: 1
            }
        ),
        "expected MaxPool1d(3, 2, 1), got: {op:?}"
    );
}

#[test]
fn test_map_avg_pool2d() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.avg_pool2d.default",
        vec![
            named("input", tensor_arg("x")),
            named("kernel_size", ints_arg(&[2, 2])),
            named("stride", ints_arg(&[2, 2])),
            named("padding", ints_arg(&[0, 0])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(
        op,
        TraceOp::AvgPool2d {
            kernel_size: [2, 2],
            stride: [2, 2],
            padding: [0, 0]
        }
    ));
}

#[test]
fn test_map_adaptive_avg_pool2d() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.adaptive_avg_pool2d.default",
        vec![
            named("input", tensor_arg("x")),
            named("output_size", ints_arg(&[1, 1])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(
        op,
        TraceOp::AdaptiveAvgPool2d {
            output_size: [1, 1]
        }
    ));
}

// ============================================================
// Type conversion
// ============================================================

#[test]
fn test_map_to_dtype_f32() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.to.dtype",
        vec![
            named("self", tensor_arg("x")),
            named("dtype", int_arg(7)), // f32
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::ToDtype { target_dtype } if target_dtype == DType::F32),
        "expected ToDtype F32, got: {op:?}"
    );
}

#[test]
fn test_map_to_dtype_f16() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.to.dtype",
        vec![
            named("self", tensor_arg("x")),
            named("dtype", int_arg(6)), // f16
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::ToDtype { target_dtype } if target_dtype == DType::F16),);
}

#[test]
fn test_map_to_copy_works_same_as_to_dtype() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten._to_copy.default",
        vec![
            named("self", tensor_arg("x")),
            named("dtype", int_arg(8)), // f64
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::ToDtype { target_dtype } if target_dtype == DType::F64),);
}

// ============================================================
// Expansion ops (try_expand_node paths)
// ============================================================

#[test]
fn test_expand_flatten_basic() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.flatten.using_ints".to_string(),
        inputs: vec![
            named("input", tensor_arg("x")),
            named("start_dim", int_arg(1)),
            named("end_dim", int_arg(-1)),
        ],
        outputs: vec![tensor_arg("out")],
        metadata: HashMap::new(),
    };
    // Input [2, 3, 4, 5], flatten(1, -1) -> [2, 60]
    let expanded = try_expand_node(&node, &ctx, "out", &[2, 3, 4, 5]).unwrap();
    let expanded = expanded.expect("flatten should expand");
    assert_eq!(expanded.len(), 1);
    assert!(
        matches!(&expanded[0].op, TraceOp::Reshape { target_shape } if *target_shape == vec![2, 60]),
        "expected Reshape to [2, 60], got: {:?}",
        expanded[0].op
    );
}

#[test]
fn test_expand_select_int() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.select.int".to_string(),
        inputs: vec![
            named("self", tensor_arg("x")),
            named("dim", int_arg(1)),
            named("index", int_arg(2)),
        ],
        outputs: vec![tensor_arg("out")],
        metadata: HashMap::new(),
    };
    // Input [4, 8, 6], select(dim=1, index=2) -> Narrow(dim=1, 2, 1) + Reshape([4, 6])
    let expanded = try_expand_node(&node, &ctx, "out", &[4, 8, 6]).unwrap();
    let expanded = expanded.expect("select.int should expand");
    assert_eq!(expanded.len(), 2, "Narrow + Reshape");
    assert!(matches!(
        expanded[0].op,
        TraceOp::Narrow {
            dim: 1,
            start: 2,
            length: 1
        }
    ));
    assert!(
        matches!(&expanded[1].op, TraceOp::Reshape { target_shape } if *target_shape == vec![4, 6]),
    );
}

#[test]
fn test_expand_chunk() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.chunk.default".to_string(),
        inputs: vec![
            named("self", tensor_arg("x")),
            named("chunks", int_arg(3)),
            named("dim", int_arg(1)),
        ],
        outputs: vec![tensors_arg(&["c0", "c1", "c2"])],
        metadata: HashMap::new(),
    };
    // Input [1, 12, T], chunk(3, dim=1) -> 3 Narrow ops with length=4 each.
    let expanded = try_expand_node(&node, &ctx, "out", &[1, 12, 8]).unwrap();
    let expanded = expanded.expect("chunk should expand");
    assert_eq!(expanded.len(), 3);
    // Check first chunk
    assert!(
        matches!(
            expanded[0].op,
            TraceOp::Narrow {
                dim: 1,
                start: 0,
                length: 4
            }
        ),
        "chunk 0: expected Narrow(1, 0, 4), got: {:?}",
        expanded[0].op
    );
    // Check third chunk
    assert!(
        matches!(
            expanded[2].op,
            TraceOp::Narrow {
                dim: 1,
                start: 8,
                length: 4
            }
        ),
        "chunk 2: expected Narrow(1, 8, 4), got: {:?}",
        expanded[2].op
    );
}

#[test]
fn test_expand_split_uniform() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.split.Tensor".to_string(),
        inputs: vec![
            named("self", tensor_arg("x")),
            named("split_size_or_sections", ints_arg(&[5])),
            named("dim", int_arg(0)),
        ],
        outputs: vec![tensors_arg(&["s0", "s1"])],
        metadata: HashMap::new(),
    };
    // Input [10, 4], split_size=5, dim=0 -> 2 Narrow ops.
    let expanded = try_expand_node(&node, &ctx, "out", &[10, 4]).unwrap();
    let expanded = expanded.expect("split should expand");
    assert_eq!(expanded.len(), 2);
    assert!(matches!(
        expanded[0].op,
        TraceOp::Narrow {
            dim: 0,
            start: 0,
            length: 5
        }
    ));
    assert!(matches!(
        expanded[1].op,
        TraceOp::Narrow {
            dim: 0,
            start: 5,
            length: 5
        }
    ));
}

#[test]
fn test_expand_split_with_sizes() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.split.Tensor".to_string(),
        inputs: vec![
            named("self", tensor_arg("x")),
            named("split_size_or_sections", ints_arg(&[3, 7])),
            named("dim", int_arg(1)),
        ],
        outputs: vec![tensors_arg(&["a", "b"])],
        metadata: HashMap::new(),
    };
    let expanded = try_expand_node(&node, &ctx, "out", &[2, 10]).unwrap();
    let expanded = expanded.expect("split_with_sizes should expand");
    assert_eq!(expanded.len(), 2);
    assert!(matches!(
        expanded[0].op,
        TraceOp::Narrow {
            dim: 1,
            start: 0,
            length: 3
        }
    ));
    assert!(matches!(
        expanded[1].op,
        TraceOp::Narrow {
            dim: 1,
            start: 3,
            length: 7
        }
    ));
}

#[test]
fn test_expand_unbind() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.unbind.int".to_string(),
        inputs: vec![named("self", tensor_arg("x")), named("dim", int_arg(0))],
        outputs: vec![tensors_arg(&["u0", "u1"])],
        metadata: HashMap::new(),
    };
    // Input [2, 4], unbind(dim=0) -> 2 * (Narrow + Reshape)
    let expanded = try_expand_node(&node, &ctx, "out", &[2, 4]).unwrap();
    let expanded = expanded.expect("unbind should expand");
    assert_eq!(
        expanded.len(),
        4,
        "2 slices * 2 nodes each (Narrow + Reshape)"
    );
    // First slice: Narrow(dim=0, 0, 1) + Reshape([4])
    assert!(matches!(
        expanded[0].op,
        TraceOp::Narrow {
            dim: 0,
            start: 0,
            length: 1
        }
    ));
    assert!(
        matches!(&expanded[1].op, TraceOp::Reshape { target_shape } if *target_shape == vec![4]),
    );
}

#[test]
fn test_expand_stack() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.stack.default".to_string(),
        inputs: vec![
            named("tensors", tensors_arg(&["a", "b", "c"])),
            named("dim", int_arg(0)),
        ],
        outputs: vec![tensor_arg("out")],
        metadata: HashMap::new(),
    };
    // Input shape [4, 5] (shape of each tensor), stack 3 along dim 0
    let expanded = try_expand_node(&node, &ctx, "out", &[4, 5]).unwrap();
    let expanded = expanded.expect("stack should expand");
    // 3 Unsqueeze + 1 Cat = 4 nodes
    assert_eq!(expanded.len(), 4);
    assert!(matches!(expanded[0].op, TraceOp::Unsqueeze { dim: 0 }));
    assert!(matches!(expanded[1].op, TraceOp::Unsqueeze { dim: 0 }));
    assert!(matches!(expanded[2].op, TraceOp::Unsqueeze { dim: 0 }));
    assert!(matches!(
        expanded[3].op,
        TraceOp::Cat {
            dim: 0,
            num_inputs: 3
        }
    ));
    assert_eq!(expanded[3].name, "out");
}

#[test]
fn test_expand_masked_fill() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.masked_fill.Scalar".to_string(),
        inputs: vec![
            named("self", tensor_arg("x")),
            named("mask", tensor_arg("m")),
            named("value", float_arg(-1e9)),
        ],
        outputs: vec![tensor_arg("out")],
        metadata: HashMap::new(),
    };
    let expanded = try_expand_node(&node, &ctx, "out", &[2, 8]).unwrap();
    let expanded = expanded.expect("masked_fill should expand");
    assert_eq!(expanded.len(), 2, "Constant + WhereCond");
    assert!(matches!(expanded[0].op, TraceOp::Constant { .. }));
    assert!(matches!(expanded[1].op, TraceOp::WhereCond));
    assert_eq!(expanded[1].name, "out");
}

#[test]
fn test_expand_index_tensor_single() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.index.Tensor".to_string(),
        inputs: vec![
            named("self", tensor_arg("x")),
            named("indices", tensors_arg(&["idx"])),
        ],
        outputs: vec![tensor_arg("out")],
        metadata: HashMap::new(),
    };
    let expanded = try_expand_node(&node, &ctx, "out", &[10, 4]).unwrap();
    let expanded = expanded.expect("index.Tensor single should expand");
    assert_eq!(expanded.len(), 1);
    assert!(matches!(expanded[0].op, TraceOp::IndexSelect { dim: 0 }));
}

#[test]
fn test_expand_sub_scalar() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.sub.Scalar".to_string(),
        inputs: vec![
            named("self", tensor_arg("x")),
            named("other", float_arg(1.0)),
        ],
        outputs: vec![tensor_arg("y")],
        metadata: HashMap::new(),
    };
    let expanded = try_expand_node(&node, &ctx, "y", &[3, 4]).unwrap();
    let expanded = expanded.expect("sub.Scalar should expand");
    assert_eq!(expanded.len(), 2);
    assert!(matches!(expanded[1].op, TraceOp::Sub));
}

#[test]
fn test_expand_div_scalar() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.div.Scalar".to_string(),
        inputs: vec![
            named("self", tensor_arg("x")),
            named("other", float_arg(2.0)),
        ],
        outputs: vec![tensor_arg("y")],
        metadata: HashMap::new(),
    };
    let expanded = try_expand_node(&node, &ctx, "y", &[5]).unwrap();
    let expanded = expanded.expect("div.Scalar should expand");
    assert_eq!(expanded.len(), 2);
    assert!(matches!(expanded[1].op, TraceOp::Div));
}

// ============================================================
// supported_ops() coverage
// ============================================================

#[test]
fn test_supported_ops_is_sorted_and_deduped() {
    let ops = supported_ops();
    assert!(!ops.is_empty());
    for window in ops.windows(2) {
        assert!(
            window[0] <= window[1],
            "supported_ops not sorted: {:?} > {:?}",
            window[0],
            window[1]
        );
    }
}

#[test]
fn test_supported_ops_includes_key_ops() {
    let ops = supported_ops();
    let must_have = [
        "aten::relu",
        "aten::linear",
        "aten::softmax",
        "aten::embedding",
        "aten::matmul",
        "aten::conv1d",
        "aten::conv2d",
        "aten::layer_norm",
        "aten::lstm",
    ];
    for expected in &must_have {
        assert!(ops.contains(expected), "supported_ops missing {expected}");
    }
}

// ============================================================
// Grid sample
// ============================================================

#[test]
fn test_map_grid_sample_bilinear() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.grid_sample.default",
        vec![
            named("self", tensor_arg("img")),
            named("grid", tensor_arg("g")),
            named("interpolation_mode", int_arg(0)),
            named("padding_mode", int_arg(0)),
            named("align_corners", bool_arg(false)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(
        op,
        TraceOp::GridSample {
            align_corners: false,
            ..
        }
    ));
    assert_eq!(inputs, vec!["img", "g"]);
}

#[test]
fn test_map_grid_sample_non_bilinear_errors() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.grid_sample.default",
        vec![
            named("self", tensor_arg("img")),
            named("grid", tensor_arg("g")),
            named("interpolation_mode", int_arg(1)), // nearest, not supported
        ],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(matches!(err, ImportError::UnsupportedOp { .. }));
}

// ============================================================
// Error edge cases
// ============================================================

#[test]
fn test_unary_op_no_input_errors() {
    let ctx = empty_ctx();
    let node = simple_node("torch.ops.aten.relu.default", vec![]);
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::MissingArgument { .. }),
        "expected MissingArgument for empty inputs, got: {err:?}"
    );
}

#[test]
fn test_binary_op_lhs_missing() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.add.Tensor",
        vec![named("other", tensor_arg("b"))],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::MissingArgument { ref arg_name, .. } if arg_name == "self"),
        "expected MissingArgument for missing 'self', got: {err:?}"
    );
}

#[test]
fn test_cat_wrong_arg_type() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cat.default",
        vec![
            named("tensors", int_arg(42)), // wrong type
            named("dim", int_arg(0)),
        ],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::WrongArgumentType { .. }),
        "expected WrongArgumentType for non-tensor-list, got: {err:?}"
    );
}

// ============================================================
// Inplace variants
// ============================================================

#[test]
fn test_map_add_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.add_.Tensor",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Add));
}

#[test]
fn test_map_selu_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.selu_.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Selu));
}

#[test]
fn test_map_hardswish_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.hardswish_.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::HardSwish));
}

#[test]
fn test_map_celu_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.celu_.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Celu { .. }));
}

// ============================================================
// Weight-dependent ops (with ctx)
// ============================================================

#[test]
fn test_map_linear_with_weights() {
    let ctx = ctx_with_weights(&[
        ("w_linear", vec![0.1; 12], vec![4, 3]),
        ("b_linear", vec![0.0; 4], vec![4]),
    ]);
    let node = simple_node(
        "torch.ops.aten.linear.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("w_linear")),
            named("bias", tensor_arg("b_linear")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Linear { .. }));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_linear_no_bias() {
    let ctx = ctx_with_weights(&[("w_lin", vec![0.1; 6], vec![2, 3])]);
    let node = simple_node(
        "torch.ops.aten.linear.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("w_lin")),
            named("bias", none_arg()),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    if let TraceOp::Linear { bias, .. } = &op {
        assert!(bias.is_none(), "expected no bias");
    } else {
        panic!("expected Linear, got: {op:?}");
    }
}

#[test]
fn test_map_linear_missing_weight_errors() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.linear.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("nonexistent")),
        ],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::MissingWeight { .. }),
        "expected MissingWeight, got: {err:?}"
    );
}

#[test]
fn test_map_embedding_with_weights() {
    let ctx = ctx_with_weights(&[("emb_w", vec![0.1; 30], vec![10, 3])]);
    let node = simple_node(
        "torch.ops.aten.embedding.default",
        vec![
            named("weight", tensor_arg("emb_w")),
            named("indices", tensor_arg("ids")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Embedding { .. }));
    assert_eq!(inputs, vec!["ids"]);
}

#[test]
fn test_map_instance_norm() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.instance_norm.default",
        vec![
            named("input", tensor_arg("x")),
            named("eps", float_arg(1e-6)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::InstanceNorm { eps } if (eps - 1e-6).abs() < 1e-12),
        "expected InstanceNorm(eps=1e-6), got: {op:?}"
    );
}

// =========================================================================
// Attention variant tests (flash, efficient, multi_head_attention_forward)
// =========================================================================

/// Helper: create OpMapContext with tensor metadata for query shape.
fn ctx_with_query_meta(query_name: &str, shape: &[i64]) -> OpMapContext<'static> {
    use crate::parse::{SymInt, SymIntConcrete};
    let mut meta_map: HashMap<String, TensorMeta> = HashMap::new();
    meta_map.insert(
        query_name.to_string(),
        TensorMeta {
            dtype: 7,
            sizes: shape
                .iter()
                .map(|&s| SymInt::Concrete(SymIntConcrete { as_int: s }))
                .collect(),
            requires_grad: false,
            strides: vec![],
            storage_offset: None,
            device: None,
            layout: None,
        },
    );
    let meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::new(meta_map));
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::default());
    OpMapContext {
        tensor_meta: meta,
        weights,
    }
}

// --- SDPA is_causal / attn_mask tests ---

#[test]
fn test_map_sdpa_causal() {
    let ctx = ctx_with_query_meta("q", &[2, 8, 16, 64]);
    let node = simple_node(
        "torch.ops.aten.scaled_dot_product_attention.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
            named("is_causal", bool_arg(true)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::SdpaCausal { .. }),
        "expected SdpaCausal for is_causal=true, got: {op:?}"
    );
    assert_eq!(inputs, vec!["q", "k", "v"]);
}

#[test]
fn test_map_sdpa_with_mask() {
    let ctx = ctx_with_query_meta("q", &[2, 8, 16, 64]);
    let node = simple_node(
        "torch.ops.aten.scaled_dot_product_attention.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
            named("attn_mask", tensor_arg("mask")),
            named("scale", float_arg(0.125)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Sdpa { scale } if (scale - 0.125).abs() < 1e-10),
        "expected Sdpa with mask, got: {op:?}"
    );
    assert_eq!(inputs, vec!["q", "k", "v", "mask"]);
}

#[test]
fn test_map_sdpa_mask_none_no_extra_input() {
    let ctx = ctx_with_query_meta("q", &[2, 8, 16, 64]);
    let node = simple_node(
        "torch.ops.aten.scaled_dot_product_attention.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
            named("attn_mask", none_arg()),
            named("scale", float_arg(0.125)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Sdpa { .. }));
    // No mask tensor in inputs when attn_mask is None.
    assert_eq!(inputs, vec!["q", "k", "v"]);
}

// --- Flash attention tests ---

#[test]
fn test_map_flash_attention_explicit_scale() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten._scaled_dot_product_flash_attention.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
            named("scale", float_arg(0.125)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Sdpa { scale } if (scale - 0.125).abs() < 1e-10),
        "expected Sdpa with scale=0.125, got: {op:?}"
    );
    assert_eq!(inputs, vec!["q", "k", "v"]);
}

#[test]
fn test_map_flash_attention_causal() {
    let ctx = ctx_with_query_meta("q", &[2, 8, 16, 64]);
    let node = simple_node(
        "torch.ops.aten._scaled_dot_product_flash_attention.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
            named("is_causal", bool_arg(true)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::SdpaCausal { .. }),
        "expected SdpaCausal for flash attention with is_causal=true, got: {op:?}"
    );
    assert_eq!(inputs, vec!["q", "k", "v"]);
}

#[test]
fn test_map_flash_attention_auto_scale() {
    // head_dim = 64 -> scale = 1/sqrt(64) = 0.125
    let ctx = ctx_with_query_meta("q", &[2, 8, 16, 64]);
    let node = simple_node(
        "torch.ops.aten._scaled_dot_product_flash_attention.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    let expected = 1.0 / (64.0f64).sqrt();
    assert!(
        matches!(op, TraceOp::Sdpa { scale } if (scale - expected).abs() < 1e-10),
        "expected auto scale=1/sqrt(64), got: {op:?}"
    );
}

#[test]
fn test_map_flash_attention_dropout_ignored() {
    // dropout_p should be ignored at inference time.
    let ctx = ctx_with_query_meta("q", &[1, 4, 8, 32]);
    let node = simple_node(
        "torch.ops.aten._scaled_dot_product_flash_attention.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
            named("dropout_p", float_arg(0.1)),
            named("scale", float_arg(0.177)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Sdpa { scale } if (scale - 0.177).abs() < 1e-10),
        "dropout_p should be ignored, got: {op:?}"
    );
}

// --- Efficient attention tests ---

#[test]
fn test_map_efficient_attention_explicit_scale() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten._scaled_dot_product_efficient_attention.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
            named("attn_bias", none_arg()),
            named("scale", float_arg(0.25)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Sdpa { scale } if (scale - 0.25).abs() < 1e-10),
        "expected Sdpa with scale=0.25, got: {op:?}"
    );
    assert_eq!(inputs, vec!["q", "k", "v"]);
}

#[test]
fn test_map_efficient_attention_causal() {
    let ctx = ctx_with_query_meta("q", &[2, 8, 16, 64]);
    let node = simple_node(
        "torch.ops.aten._scaled_dot_product_efficient_attention.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
            named("attn_bias", none_arg()),
            named("compute_log_sumexp", bool_arg(false)),
            named("is_causal", bool_arg(true)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::SdpaCausal { .. }),
        "expected SdpaCausal for efficient attention with is_causal=true, got: {op:?}"
    );
    assert_eq!(inputs, vec!["q", "k", "v"]);
}

#[test]
fn test_map_efficient_attention_with_bias() {
    let ctx = ctx_with_query_meta("q", &[2, 8, 16, 64]);
    let node = simple_node(
        "torch.ops.aten._scaled_dot_product_efficient_attention.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
            named("attn_bias", tensor_arg("bias")),
            named("scale", float_arg(0.125)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Sdpa { scale } if (scale - 0.125).abs() < 1e-10),
        "expected Sdpa with attn_bias as 4th input, got: {op:?}"
    );
    assert_eq!(inputs, vec!["q", "k", "v", "bias"]);
}

#[test]
fn test_map_efficient_attention_auto_scale() {
    // head_dim = 128 -> scale = 1/sqrt(128)
    let ctx = ctx_with_query_meta("q", &[1, 16, 32, 128]);
    let node = simple_node(
        "torch.ops.aten._scaled_dot_product_efficient_attention.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
            named("attn_bias", none_arg()),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    let expected = 1.0 / (128.0f64).sqrt();
    assert!(
        matches!(op, TraceOp::Sdpa { scale } if (scale - expected).abs() < 1e-10),
        "expected auto scale=1/sqrt(128), got: {op:?}"
    );
}

// --- Multi-head attention forward tests ---

#[test]
fn test_map_mha_forward_explicit_dims() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.multi_head_attention_forward.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
            named("embed_dim_to_check", int_arg(512)),
            named("num_heads", int_arg(8)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    // head_dim = 512/8 = 64, scale = 1/sqrt(64) = 0.125
    let expected = 1.0 / (64.0f64).sqrt();
    assert!(
        matches!(op, TraceOp::Sdpa { scale } if (scale - expected).abs() < 1e-10),
        "expected Sdpa with scale=1/sqrt(64), got: {op:?}"
    );
    assert_eq!(inputs, vec!["q", "k", "v"]);
}

#[test]
fn test_map_mha_forward_fallback_to_meta() {
    // When embed_dim is 0, fall back to query tensor metadata.
    let ctx = ctx_with_query_meta("q", &[10, 32, 64]);
    let node = simple_node(
        "torch.ops.aten.multi_head_attention_forward.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    // Falls back to head_dim from last dim of query = 64
    let expected = 1.0 / (64.0f64).sqrt();
    assert!(
        matches!(op, TraceOp::Sdpa { scale } if (scale - expected).abs() < 1e-10),
        "expected Sdpa with scale from query meta, got: {op:?}"
    );
}

#[test]
fn test_map_mha_forward_large_dim() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.multi_head_attention_forward.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
            named("embed_dim_to_check", int_arg(1024)),
            named("num_heads", int_arg(16)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    // head_dim = 1024/16 = 64, scale = 1/sqrt(64) = 0.125
    let expected = 1.0 / (64.0f64).sqrt();
    assert!(
        matches!(op, TraceOp::Sdpa { scale } if (scale - expected).abs() < 1e-10),
        "expected Sdpa with scale=1/sqrt(64), got: {op:?}"
    );
}

// --- supported_ops includes the new attention variants ---

#[test]
fn test_supported_ops_includes_attention_variants() {
    let ops = supported_ops();
    assert!(
        ops.contains(&"aten::_scaled_dot_product_flash_attention"),
        "supported_ops should include flash attention"
    );
    assert!(
        ops.contains(&"aten::_scaled_dot_product_efficient_attention"),
        "supported_ops should include efficient attention"
    );
    assert!(
        ops.contains(&"aten::multi_head_attention_forward"),
        "supported_ops should include multi_head_attention_forward"
    );
}

// ---------------------------------------------------------------------------
// Wave 8: Vision and audio model ops
// ---------------------------------------------------------------------------

#[test]
fn test_map_upsample_bicubic2d_default() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.upsample_bicubic2d.default",
        vec![
            named("input", tensor_arg("x")),
            named("scales_h", float_arg(3.0)),
            named("scales_w", float_arg(3.0)),
        ],
    );
    let (op, deps) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Upsample2d { scale_h, scale_w, .. }
            if (scale_h - 3.0).abs() < 1e-10 && (scale_w - 3.0).abs() < 1e-10),
        "expected Upsample2d(Bicubic, 3x3), got: {op:?}"
    );
    assert_eq!(deps, vec!["x"]);
}

#[test]
fn test_map_upsample_bicubic2d_vec() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.upsample_bicubic2d.vec",
        vec![
            named("input", tensor_arg("x")),
            named("scales_h", float_arg(4.0)),
            named("scales_w", float_arg(2.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Upsample2d { scale_h, scale_w, .. }
            if (scale_h - 4.0).abs() < 1e-10 && (scale_w - 2.0).abs() < 1e-10),);
}

#[test]
fn test_map_replication_pad1d() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.replication_pad1d.default",
        vec![
            named("input", tensor_arg("x")),
            named("padding", ints_arg(&[2, 3])),
        ],
    );
    let (op, deps) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "replication_pad1d_2_3"),
        "expected replication_pad1d_2_3, got: {op:?}"
    );
    assert_eq!(deps, vec!["x"]);
}

#[test]
fn test_map_replication_pad2d() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.replication_pad2d.default",
        vec![
            named("input", tensor_arg("x")),
            named("padding", ints_arg(&[1, 1, 2, 2])),
        ],
    );
    let (op, deps) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "replication_pad2d_1_1_2_2"),
        "expected replication_pad2d_1_1_2_2, got: {op:?}"
    );
    assert_eq!(deps, vec!["x"]);
}

#[test]
fn test_map_replication_pad2d_too_few_args_errors() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.replication_pad2d.default",
        vec![
            named("input", tensor_arg("x")),
            named("padding", ints_arg(&[1, 1])),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_err(), "should error with only 2 padding elements");
}

#[test]
fn test_map_channel_shuffle() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.channel_shuffle.default",
        vec![named("input", tensor_arg("x")), named("groups", int_arg(4))],
    );
    let (op, deps) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "channel_shuffle_g4"),
        "expected channel_shuffle_g4, got: {op:?}"
    );
    assert_eq!(deps, vec!["x"]);
}

#[test]
fn test_map_adaptive_max_pool1d() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.adaptive_max_pool1d.default",
        vec![
            named("input", tensor_arg("x")),
            named("output_size", ints_arg(&[1])),
        ],
    );
    let (op, deps) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "adaptive_max_pool1d_1"),
        "expected adaptive_max_pool1d_1, got: {op:?}"
    );
    assert_eq!(deps, vec!["x"]);
}

#[test]
fn test_map_nll_loss_forward() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.nll_loss_forward.default",
        vec![
            named("input", tensor_arg("logits")),
            named("target", tensor_arg("labels")),
            named("reduction", int_arg(1)),
        ],
    );
    let (op, deps) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "nll_loss_forward_r1"),
        "expected nll_loss_forward_r1, got: {op:?}"
    );
    assert_eq!(deps, vec!["logits", "labels"]);
}

#[test]
fn test_map_mse_loss() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.mse_loss.default",
        vec![
            named("input", tensor_arg("pred")),
            named("target", tensor_arg("truth")),
            named("reduction", int_arg(2)),
        ],
    );
    let (op, deps) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "mse_loss_r2"),
        "expected mse_loss_r2, got: {op:?}"
    );
    assert_eq!(deps, vec!["pred", "truth"]);
}

#[test]
fn test_map_l1_loss() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.l1_loss.default",
        vec![
            named("input", tensor_arg("pred")),
            named("target", tensor_arg("truth")),
        ],
    );
    let (op, deps) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "l1_loss_r1"),
        "expected l1_loss_r1 (default reduction=mean), got: {op:?}"
    );
    assert_eq!(deps, vec!["pred", "truth"]);
}

#[test]
fn test_map_smooth_l1_loss() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.smooth_l1_loss.default",
        vec![
            named("input", tensor_arg("pred")),
            named("target", tensor_arg("truth")),
            named("reduction", int_arg(0)),
            named("beta", float_arg(0.5)),
        ],
    );
    let (op, deps) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "smooth_l1_loss_r0_b0.5"),
        "expected smooth_l1_loss_r0_b0.5, got: {op:?}"
    );
    assert_eq!(deps, vec!["pred", "truth"]);
}

#[test]
fn test_map_huber_loss_dispatches_to_smooth_l1() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.huber_loss.default",
        vec![
            named("input", tensor_arg("pred")),
            named("target", tensor_arg("truth")),
            named("reduction", int_arg(1)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("smooth_l1_loss")),
        "huber_loss should dispatch to smooth_l1_loss mapper, got: {op:?}"
    );
}

#[test]
fn test_map_binary_cross_entropy() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.binary_cross_entropy.default",
        vec![
            named("input", tensor_arg("probs")),
            named("target", tensor_arg("labels")),
            named("reduction", int_arg(1)),
        ],
    );
    let (op, deps) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "binary_cross_entropy_r1"),
        "expected binary_cross_entropy_r1, got: {op:?}"
    );
    assert_eq!(deps, vec!["probs", "labels"]);
}

#[test]
fn test_supported_ops_includes_wave8() {
    let ops = supported_ops();
    for expected in &[
        "aten::upsample_bicubic2d",
        "aten::replication_pad1d",
        "aten::replication_pad2d",
        "aten::channel_shuffle",
        "aten::adaptive_max_pool1d",
        "aten::nll_loss_forward",
        "aten::mse_loss",
        "aten::l1_loss",
        "aten::smooth_l1_loss",
        "aten::huber_loss",
        "aten::binary_cross_entropy",
    ] {
        assert!(
            ops.contains(expected),
            "supported_ops should include {expected}"
        );
    }
}

// =====================================================================
// Wave 9: commonly missing model patterns
// =====================================================================

// --- Unary math ---

#[test]
fn test_map_trunc() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.trunc.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "trunc"),
        "expected Custom trunc, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_expm1() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.expm1.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "expm1"),
        "expected Custom expm1, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_log1p() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.log1p.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "log1p"),
        "expected Custom log1p, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_acos() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.acos.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(&op, TraceOp::Custom { name } if name == "acos"));
}

#[test]
fn test_map_asin() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.asin.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(&op, TraceOp::Custom { name } if name == "asin"));
}

#[test]
fn test_map_atan() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.atan.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(&op, TraceOp::Custom { name } if name == "atan"));
}

#[test]
fn test_map_cosh() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cosh.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(&op, TraceOp::Custom { name } if name == "cosh"));
}

#[test]
fn test_map_sinh() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sinh.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(&op, TraceOp::Custom { name } if name == "sinh"));
}

// --- Value testing ---

#[test]
fn test_map_isinf() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.isinf.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(&op, TraceOp::Custom { name } if name == "isinf"));
}

#[test]
fn test_map_isnan() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.isnan.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(&op, TraceOp::Custom { name } if name == "isnan"));
}

#[test]
fn test_map_isfinite() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.isfinite.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(&op, TraceOp::Custom { name } if name == "isfinite"));
}

// --- Bitwise ---

#[test]
fn test_map_bitwise_not() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.bitwise_not.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(&op, TraceOp::Custom { name } if name == "bitwise_not"));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_bitwise_and() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.bitwise_and.Tensor",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(&op, TraceOp::Custom { name } if name == "bitwise_and"));
    assert_eq!(inputs, vec!["a", "b"]);
}

#[test]
fn test_map_bitwise_or() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.bitwise_or.Tensor",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(&op, TraceOp::Custom { name } if name == "bitwise_or"));
    assert_eq!(inputs, vec!["a", "b"]);
}

// --- Tensor-arg clamp variants ---

#[test]
fn test_map_clamp_min_tensor() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.clamp_min.Tensor",
        vec![
            named("self", tensor_arg("x")),
            named("min", tensor_arg("lower")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Maximum),
        "clamp_min.Tensor should map to Maximum, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "lower"]);
}

#[test]
fn test_map_clamp_max_tensor() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.clamp_max.Tensor",
        vec![
            named("self", tensor_arg("x")),
            named("max", tensor_arg("upper")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Minimum),
        "clamp_max.Tensor should map to Minimum, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "upper"]);
}

// --- Tensor creation ---

#[test]
fn test_map_tile() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.tile.default",
        vec![
            named("input", tensor_arg("x")),
            named("dims", ints_arg(&[2, 3])),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Expand { target_shape } if *target_shape == vec![2, 3]),
        "tile should map to Expand, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_arange_start() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.arange.start",
        vec![
            named("start", float_arg(1.0)),
            named("end", float_arg(10.0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Arange { start, end, step }
            if (start - 1.0).abs() < 1e-10
            && (end - 10.0).abs() < 1e-10
            && (step - 1.0).abs() < 1e-10),
        "expected Arange {{start=1, end=10, step=1}}, got: {op:?}"
    );
    assert!(inputs.is_empty());
}

#[test]
fn test_map_eye() {
    let ctx = empty_ctx();
    let node = simple_node("torch.ops.aten.eye.default", vec![named("n", int_arg(5))]);
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "eye_5_5"),
        "expected Custom eye_5_5, got: {op:?}"
    );
}

#[test]
fn test_map_eye_m() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.eye.m",
        vec![named("n", int_arg(3)), named("m", int_arg(4))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "eye_3_4"),
        "expected Custom eye_3_4, got: {op:?}"
    );
}

// --- Expand variants ---

#[test]
fn test_map_expand_as() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.expand_as.default",
        vec![
            named("self", tensor_arg("x")),
            named("other", tensor_arg("y")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Expand { target_shape } if target_shape.is_empty()),
        "expand_as should map to Expand with empty shape, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "y"]);
}

#[test]
fn test_map_broadcast_to() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.broadcast_to.default",
        vec![
            named("input", tensor_arg("x")),
            named("size", ints_arg(&[4, -1, 8])),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Expand { target_shape } if *target_shape == vec![4, usize::MAX, 8]),
        "broadcast_to should map to Expand, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// --- Loss functions ---

#[test]
fn test_map_bce_with_logits() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.binary_cross_entropy_with_logits.default",
        vec![
            named("input", tensor_arg("logits")),
            named("target", tensor_arg("labels")),
            named("reduction", int_arg(2)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "bce_with_logits_r2"),
        "expected bce_with_logits_r2, got: {op:?}"
    );
    assert_eq!(inputs, vec!["logits", "labels"]);
}

#[test]
fn test_map_cross_entropy_loss() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cross_entropy_loss.default",
        vec![
            named("input", tensor_arg("logits")),
            named("target", tensor_arg("labels")),
            named("reduction", int_arg(1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "cross_entropy_loss_r1"),
        "expected cross_entropy_loss_r1, got: {op:?}"
    );
    assert_eq!(inputs, vec!["logits", "labels"]);
}

// --- Indexing ---

#[test]
fn test_map_index_fill() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.index_fill.int_Scalar",
        vec![
            named("self", tensor_arg("x")),
            named("dim", int_arg(1)),
            named("index", tensor_arg("idx")),
            named("value", float_arg(0.0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "index_fill_dim1_v0"),
        "expected index_fill_dim1_v0, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "idx"]);
}

#[test]
fn test_map_index_copy() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.index_copy.default",
        vec![
            named("self", tensor_arg("x")),
            named("dim", int_arg(0)),
            named("index", tensor_arg("idx")),
            named("source", tensor_arg("src")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "index_copy_dim0"),
        "expected index_copy_dim0, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "idx", "src"]);
}

#[test]
fn test_map_scatter_reduce() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.scatter_reduce.two",
        vec![
            named("self", tensor_arg("x")),
            named("dim", int_arg(1)),
            named("index", tensor_arg("idx")),
            named("src", tensor_arg("src")),
            named("reduce", str_arg("sum")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "scatter_reduce_sum_dim1"),
        "expected scatter_reduce_sum_dim1, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "idx", "src"]);
}

// --- Repeat (scalar count) ---

#[test]
fn test_map_repeat_interleave_int() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.repeat_interleave.self_int",
        vec![
            named("input", tensor_arg("x")),
            named("repeats", int_arg(3)),
            named("dim", int_arg(1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "repeat_interleave_n3_dim1"),
        "expected repeat_interleave_n3_dim1, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// --- Conditional / where variants ---

#[test]
fn test_map_where_scalar_other() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.where.ScalarOther",
        vec![
            named("condition", tensor_arg("mask")),
            named("self", tensor_arg("x")),
            named("other", float_arg(-1.0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("where_scalar_other")),
        "expected where_scalar_other custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["mask", "x"]);
}

#[test]
fn test_map_where_scalar_self() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.where.ScalarSelf",
        vec![
            named("condition", tensor_arg("mask")),
            named("self", float_arg(1.0)),
            named("other", tensor_arg("x")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("where_scalar_self")),
        "expected where_scalar_self custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["mask", "x"]);
}

#[test]
fn test_map_masked_scatter() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.masked_scatter.default",
        vec![
            named("self", tensor_arg("x")),
            named("mask", tensor_arg("m")),
            named("source", tensor_arg("src")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "masked_scatter"),
        "expected masked_scatter custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "m", "src"]);
}

// --- In-place variants dispatch correctly ---

#[test]
fn test_map_index_fill_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.index_fill_.int_Scalar",
        vec![
            named("self", tensor_arg("x")),
            named("dim", int_arg(0)),
            named("index", tensor_arg("idx")),
            named("value", float_arg(1.0)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("index_fill")),
        "in-place index_fill_ should dispatch, got: {op:?}"
    );
}

#[test]
fn test_map_masked_scatter_inplace() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.masked_scatter_.default",
        vec![
            named("self", tensor_arg("x")),
            named("mask", tensor_arg("m")),
            named("source", tensor_arg("src")),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "masked_scatter"),
        "in-place masked_scatter_ should dispatch, got: {op:?}"
    );
}

// --- supported_ops includes Wave 9 ---

#[test]
fn test_supported_ops_includes_wave9() {
    let ops = supported_ops();
    for expected in &[
        "aten::trunc",
        "aten::expm1",
        "aten::log1p",
        "aten::acos",
        "aten::asin",
        "aten::atan",
        "aten::cosh",
        "aten::sinh",
        "aten::isinf",
        "aten::isnan",
        "aten::isfinite",
        "aten::bitwise_not",
        "aten::bitwise_and",
        "aten::bitwise_or",
        "aten::tile",
        "aten::eye",
        "aten::expand_as",
        "aten::broadcast_to",
        "aten::binary_cross_entropy_with_logits",
        "aten::cross_entropy_loss",
        "aten::index_fill",
        "aten::index_copy",
        "aten::scatter_reduce",
        "aten::masked_scatter",
    ] {
        assert!(
            ops.contains(expected),
            "supported_ops should include {expected}"
        );
    }
}
