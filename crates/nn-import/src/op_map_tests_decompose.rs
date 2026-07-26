// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for import-time op decomposition: scalar binary ops, squeeze.default,
//! and multi-axis reductions.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;

use super::*;
use crate::parse::{
    Argument, ArgumentFloat, ArgumentInts, ArgumentTensor, NamedArgument, Node, TensorArgument,
    TensorMeta,
};

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
        kind: None,
    }
}

fn ints_arg(vals: &[i64]) -> Argument {
    Argument::Ints(ArgumentInts {
        as_ints: vals.to_vec(),
    })
}

fn bool_arg(val: bool) -> Argument {
    Argument::Bool(crate::parse::ArgumentBool { as_bool: val })
}

fn float_arg(val: f64) -> Argument {
    Argument::Float(ArgumentFloat { as_float: val })
}

fn empty_ctx() -> OpMapContext<'static> {
    let meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::default());
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::default());
    OpMapContext {
        tensor_meta: meta,
        weights,
    }
}

// ---- Scalar binary op expansion tests ----

#[test]
fn test_expand_add_scalar() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.add.Scalar".to_string(),
        inputs: vec![
            named("self", tensor_arg("x")),
            named("other", float_arg(2.5)),
        ],
        outputs: vec![tensor_arg("y")],
        metadata: HashMap::new(),
    };
    let expanded = try_expand_node(&node, &ctx, "y", &[4, 8]).unwrap();
    let expanded = expanded.expect("add.Scalar should expand");
    assert_eq!(expanded.len(), 2, "Constant + binary op");
    assert!(matches!(expanded[0].op, TraceOp::Constant { value } if (value - 2.5).abs() < 1e-10));
    assert_eq!(expanded[0].output_shape, Vec::<usize>::new());
    assert!(matches!(expanded[1].op, TraceOp::Add));
    assert_eq!(expanded[1].name, "y");
    assert_eq!(expanded[1].input_names, vec!["x", "y_const"]);
    assert_eq!(expanded[1].output_shape, vec![4, 8]);
}

#[test]
fn test_expand_mul_scalar() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.mul.Scalar".to_string(),
        inputs: vec![
            named("self", tensor_arg("a")),
            named("other", float_arg(0.5)),
        ],
        outputs: vec![tensor_arg("result")],
        metadata: HashMap::new(),
    };
    let expanded = try_expand_node(&node, &ctx, "result", &[2, 3]).unwrap();
    let expanded = expanded.expect("mul.Scalar should expand");
    assert_eq!(expanded.len(), 2);
    assert!(matches!(expanded[1].op, TraceOp::Mul));
    assert_eq!(expanded[1].input_names, vec!["a", "result_const"]);
}

#[test]
fn test_tensor_add_not_expanded() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.add.Tensor".to_string(),
        inputs: vec![
            named("self", tensor_arg("x")),
            named("other", tensor_arg("y")),
        ],
        outputs: vec![tensor_arg("z")],
        metadata: HashMap::new(),
    };
    let result = try_expand_node(&node, &ctx, "z", &[4]).unwrap();
    assert!(result.is_none(), "add.Tensor should NOT expand");
}

// ---- squeeze.default expansion tests ----

#[test]
fn test_expand_squeeze_default() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.squeeze.default".to_string(),
        inputs: vec![named("self", tensor_arg("x"))],
        outputs: vec![tensor_arg("y")],
        metadata: HashMap::new(),
    };
    // Input [2, 1, 3, 1] → squeeze all → [2, 3]
    let expanded = try_expand_node(&node, &ctx, "y", &[2, 1, 3, 1]).unwrap();
    let expanded = expanded.expect("squeeze.default should expand when shape is known");
    assert_eq!(expanded.len(), 1, "single Reshape node");
    assert!(
        matches!(&expanded[0].op, TraceOp::Reshape { target_shape } if *target_shape == vec![2, 3]),
        "expected Reshape to [2, 3], got: {:?}",
        expanded[0].op
    );
    assert_eq!(expanded[0].name, "y");
    assert_eq!(expanded[0].output_shape, vec![2, 3]);
}

#[test]
fn test_expand_squeeze_default_no_ones() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.squeeze.default".to_string(),
        inputs: vec![named("self", tensor_arg("x"))],
        outputs: vec![tensor_arg("y")],
        metadata: HashMap::new(),
    };
    // Input [4, 8] has no size-1 dims → Reshape to same shape (no-op).
    let expanded = try_expand_node(&node, &ctx, "y", &[4, 8]).unwrap();
    let expanded = expanded.expect("squeeze.default should expand");
    assert_eq!(expanded[0].output_shape, vec![4, 8]);
}

#[test]
fn test_expand_squeeze_default_no_shape_falls_through() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.squeeze.default".to_string(),
        inputs: vec![named("self", tensor_arg("x"))],
        outputs: vec![tensor_arg("y")],
        metadata: HashMap::new(),
    };
    // Empty input shape → cannot compute output, falls through to single-op path.
    let result = try_expand_node(&node, &ctx, "y", &[]).unwrap();
    assert!(
        result.is_none(),
        "should fall through when shape is unknown"
    );
}

// ---- Multi-axis reduce expansion tests ----

#[test]
fn test_expand_multi_axis_sum_keepdim_true() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.sum.dim_IntList".to_string(),
        inputs: vec![
            named("self", tensor_arg("x")),
            named("dim", ints_arg(&[1, 2])),
            named("keepdim", bool_arg(true)),
        ],
        outputs: vec![tensor_arg("out")],
        metadata: HashMap::new(),
    };
    // Input [2, 3, 4, 5], reduce dims [1, 2], keepdim=true → [2, 1, 1, 5]
    let expanded = try_expand_node(&node, &ctx, "out", &[2, 3, 4, 5]).unwrap();
    let expanded = expanded.expect("multi-axis sum should expand");
    // 2 sequential reduce nodes (no final reshape since keepdim=true)
    assert_eq!(expanded.len(), 2, "two sequential reduces");
    assert!(matches!(
        expanded[0].op,
        TraceOp::ReduceSum {
            dim: 1,
            keepdim: true
        }
    ));
    assert_eq!(expanded[0].output_shape, vec![2, 1, 4, 5]);
    assert!(matches!(
        expanded[1].op,
        TraceOp::ReduceSum {
            dim: 2,
            keepdim: true
        }
    ));
    assert_eq!(expanded[1].output_shape, vec![2, 1, 1, 5]);
    assert_eq!(expanded[1].name, "out"); // final node gets output name
}

#[test]
fn test_expand_multi_axis_sum_keepdim_false() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.sum.dim_IntList".to_string(),
        inputs: vec![
            named("self", tensor_arg("x")),
            named("dim", ints_arg(&[1, 2])),
        ],
        outputs: vec![tensor_arg("out")],
        metadata: HashMap::new(),
    };
    // Input [2, 3, 4, 5], reduce dims [1, 2], keepdim=false → [2, 5]
    let expanded = try_expand_node(&node, &ctx, "out", &[2, 3, 4, 5]).unwrap();
    let expanded = expanded.expect("multi-axis sum should expand");
    // 2 reduce nodes + 1 reshape (keepdim=false)
    assert_eq!(expanded.len(), 3, "two reduces + one reshape");
    assert!(matches!(
        expanded[0].op,
        TraceOp::ReduceSum {
            dim: 1,
            keepdim: true
        }
    ));
    assert!(matches!(
        expanded[1].op,
        TraceOp::ReduceSum {
            dim: 2,
            keepdim: true
        }
    ));
    assert!(
        matches!(&expanded[2].op, TraceOp::Reshape { target_shape } if *target_shape == vec![2, 5]),
        "final reshape should produce [2, 5], got: {:?}",
        expanded[2].op
    );
    assert_eq!(expanded[2].name, "out");
    assert_eq!(expanded[2].output_shape, vec![2, 5]);
}

#[test]
fn test_single_axis_sum_not_expanded() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.sum.dim_IntList".to_string(),
        inputs: vec![named("self", tensor_arg("x")), named("dim", ints_arg(&[2]))],
        outputs: vec![tensor_arg("out")],
        metadata: HashMap::new(),
    };
    // Single-dim reduce should NOT expand (handled by standard single-op path).
    let result = try_expand_node(&node, &ctx, "out", &[2, 3, 4]).unwrap();
    assert!(result.is_none(), "single-axis sum should not expand");
}

#[test]
fn test_expand_multi_axis_mean() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.mean.dim".to_string(),
        inputs: vec![
            named("self", tensor_arg("x")),
            named("dim", ints_arg(&[0, 2])),
            named("keepdim", bool_arg(true)),
        ],
        outputs: vec![tensor_arg("out")],
        metadata: HashMap::new(),
    };
    let expanded = try_expand_node(&node, &ctx, "out", &[2, 3, 4]).unwrap();
    let expanded = expanded.expect("multi-axis mean should expand");
    assert_eq!(expanded.len(), 2);
    assert!(matches!(
        expanded[0].op,
        TraceOp::ReduceMean {
            dim: 0,
            keepdim: true
        }
    ));
    assert!(matches!(
        expanded[1].op,
        TraceOp::ReduceMean {
            dim: 2,
            keepdim: true
        }
    ));
}
