// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for aten op -> TraceOp mapping.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;

use super::*;
use crate::parse::{
    Argument, ArgumentInt, ArgumentInts, ArgumentNone, ArgumentTensor, NamedArgument, Node,
    TensorArgument, TensorMeta,
};

fn empty_ctx() -> OpMapContext<'static> {
    // Use leaked boxes for 'static lifetime in tests.
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

#[test]
fn test_map_relu() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.relu.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Relu));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_add() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.add.Tensor",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Add));
    assert_eq!(inputs, vec!["a", "b"]);
}

#[test]
fn test_map_sub_preserves_operand_order() {
    // Regression: binary_op used filter_map which could silently reorder
    // non-commutative op inputs if the first arg was non-tensor (#2367).
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sub.Tensor",
        vec![
            named("self", tensor_arg("minuend")),
            named("other", tensor_arg("subtrahend")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Sub));
    assert_eq!(inputs, vec!["minuend", "subtrahend"]);
}

#[test]
fn test_map_mm_uses_mat2_arg_name() {
    // mm/bmm use "mat2" not "other" in the aten schema.
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.mm.default",
        vec![
            named("self", tensor_arg("weight")),
            named("mat2", tensor_arg("input")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::MatMul));
    assert_eq!(inputs, vec!["weight", "input"]);
}

#[test]
fn test_map_binary_op_missing_other_errors() {
    // binary_op must error if "other" is missing, not silently drop inputs.
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.div.Tensor",
        vec![named("self", tensor_arg("a"))],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::MissingArgument { .. }),
        "expected MissingArgument for missing 'other', got: {err:?}"
    );
}

#[test]
fn test_map_gelu_tanh() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.gelu.default",
        vec![
            named("input", tensor_arg("x")),
            named(
                "approximate",
                Argument::Str(crate::parse::ArgumentString {
                    as_string: "tanh".to_string(),
                }),
            ),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Gelu));
}

#[test]
fn test_map_gelu_none() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.gelu.default",
        vec![
            named("input", tensor_arg("x")),
            named(
                "approximate",
                Argument::Str(crate::parse::ArgumentString {
                    as_string: "none".to_string(),
                }),
            ),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::GeluErf));
}

#[test]
fn test_map_reshape() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.view.default",
        vec![
            named("input", tensor_arg("x")),
            named("size", ints_arg(&[2, 3, 4])),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Reshape { target_shape } if target_shape == vec![2, 3, 4]));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_transpose() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.transpose.int",
        vec![
            named("input", tensor_arg("x")),
            named("dim0", int_arg(1)),
            named("dim1", int_arg(2)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Transpose { dim0: 1, dim1: 2 }));
}

#[test]
fn test_map_softmax() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.softmax.int",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(1))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Softmax { dim: 1 }));
}

#[test]
fn test_map_unsupported_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.unknown_op.default",
        vec![named("input", tensor_arg("x"))],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::UnsupportedOp { .. }),
        "expected UnsupportedOp, got: {err:?}"
    );
}

/// #2364 Bug 1: SDPA with explicit scale uses the provided value.
#[test]
fn test_map_sdpa_explicit_scale() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.scaled_dot_product_attention.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
            named(
                "scale",
                Argument::Float(crate::parse::ArgumentFloat { as_float: 0.125 }),
            ),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Sdpa { scale } if (scale - 0.125).abs() < 1e-10));
    assert_eq!(inputs, vec!["q", "k", "v"]);
}

/// #2364 Bug 1: SDPA with no scale auto-computes 1/sqrt(head_dim) from query shape.
#[test]
fn test_map_sdpa_auto_scale_from_query_shape() {
    use crate::parse::{SymInt, SymIntConcrete};
    let mut meta_map: HashMap<String, TensorMeta> = HashMap::new();
    meta_map.insert(
        "q".to_string(),
        TensorMeta {
            dtype: 7,
            sizes: vec![
                SymInt::Concrete(SymIntConcrete { as_int: 2 }),
                SymInt::Concrete(SymIntConcrete { as_int: 8 }),
                SymInt::Concrete(SymIntConcrete { as_int: 16 }),
                SymInt::Concrete(SymIntConcrete { as_int: 64 }), // head_dim = 64
            ],
            requires_grad: false,
            strides: vec![],
            storage_offset: None,
            device: None,
            layout: None,
        },
    );
    let meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::new(meta_map));
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::default());
    let ctx = OpMapContext {
        tensor_meta: meta,
        weights,
    };
    let node = simple_node(
        "torch.ops.aten.scaled_dot_product_attention.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    let expected_scale = 1.0 / (64.0f64).sqrt(); // 0.125
    assert!(
        matches!(op, TraceOp::Sdpa { scale } if (scale - expected_scale).abs() < 1e-10),
        "expected scale=1/sqrt(64), got: {op:?}"
    );
}

/// #2364 Bug 1: SDPA with no scale and no tensor_meta errors instead of defaulting.
#[test]
fn test_map_sdpa_no_scale_no_meta_errors() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.scaled_dot_product_attention.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
        ],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::MissingArgument { ref arg_name, .. } if arg_name == "scale"),
        "expected MissingArgument for scale, got: {err:?}"
    );
}

#[test]
fn test_map_cat() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cat.default",
        vec![
            named(
                "tensors",
                Argument::Tensors(crate::parse::ArgumentTensors {
                    as_tensors: vec![
                        TensorArgument {
                            name: "a".to_string(),
                        },
                        TensorArgument {
                            name: "b".to_string(),
                        },
                    ],
                }),
            ),
            named("dim", int_arg(1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(
        op,
        TraceOp::Cat {
            dim: 1,
            num_inputs: 2
        }
    ));
    assert_eq!(inputs, vec!["a", "b"]);
}

// --- #2364: Import pipeline correctness regression tests ---

/// #2364 Bug 3: squeeze.dim requires dim argument.
#[test]
fn test_map_squeeze_dim_requires_dim() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.squeeze.dim",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(2))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Squeeze { dim: 2 }),
        "expected Squeeze dim=2, got: {op:?}"
    );
}

/// #2364 Bug 3: squeeze.default (no dim) errors because TraceOp has no SqueezeAll.
#[test]
fn test_map_squeeze_default_errors() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.squeeze.default",
        vec![named("self", tensor_arg("x"))],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::UnsupportedOp { .. }),
        "expected UnsupportedOp for squeeze-all, got: {err:?}"
    );
}

/// #2364 Bug 7: to_dtype with missing dtype argument errors.
#[test]
fn test_map_to_dtype_missing_arg_errors() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.to.dtype",
        vec![named("self", tensor_arg("x"))],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::MissingArgument { ref arg_name, .. } if arg_name == "dtype"),
        "expected MissingArgument for dtype, got: {err:?}"
    );
}

/// #2364 Bug 7: to_dtype with unknown scalar type errors instead of defaulting to f32.
#[test]
fn test_map_to_dtype_unknown_scalar_type_errors() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.to.dtype",
        vec![named("self", tensor_arg("x")), named("dtype", int_arg(99))],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::WrongArgumentType { ref arg_name, .. } if arg_name == "dtype"),
        "expected WrongArgumentType for unknown ScalarType, got: {err:?}"
    );
}

/// #2364 Bug 7: to_dtype with known scalar type works correctly.
#[test]
fn test_map_to_dtype_bf16() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.to.dtype",
        vec![
            named("self", tensor_arg("x")),
            named("dtype", int_arg(13)), // bf16
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::ToDtype { target_dtype } if target_dtype == DType::BF16),
        "expected ToDtype bf16, got: {op:?}"
    );
}

// --- #2355: Safe dimension conversion tests ---

#[test]
fn test_negative_dim_rejected_softmax() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.softmax.int",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(-1))],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::NegativeDimension { value: -1, .. }),
        "expected NegativeDimension, got: {err:?}"
    );
}

#[test]
fn test_negative_dim_rejected_transpose() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.transpose.int",
        vec![
            named("input", tensor_arg("x")),
            named("dim0", int_arg(0)),
            named("dim1", int_arg(-2)),
        ],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::NegativeDimension { value: -2, .. }),
        "expected NegativeDimension, got: {err:?}"
    );
}

#[test]
fn test_reshape_neg1_sentinel_allowed() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.view.default",
        vec![
            named("input", tensor_arg("x")),
            named("size", ints_arg(&[2, -1])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Reshape { target_shape } if target_shape == vec![2, usize::MAX]),
        "reshape -1 should map to usize::MAX sentinel"
    );
}

#[test]
fn test_reshape_neg2_rejected() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.view.default",
        vec![
            named("input", tensor_arg("x")),
            named("size", ints_arg(&[2, -2])),
        ],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::NegativeDimension { value: -2, .. }),
        "expected NegativeDimension for -2, got: {err:?}"
    );
}

#[test]
fn test_expand_neg1_sentinel_allowed() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.expand.default",
        vec![
            named("input", tensor_arg("x")),
            named("size", ints_arg(&[4, -1, 8])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Expand { target_shape } if target_shape == vec![4, usize::MAX, 8]),
        "expand -1 should map to usize::MAX sentinel"
    );
}

#[test]
fn test_multi_axis_reduction_rejected() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sum.dim_IntList",
        vec![
            named("self", tensor_arg("x")),
            named("dim", ints_arg(&[1, 2])),
        ],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::MultiAxisNotSupported { .. }),
        "expected MultiAxisNotSupported, got: {err:?}"
    );
}

#[test]
fn test_single_axis_reduction_works() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sum.dim_IntList",
        vec![named("self", tensor_arg("x")), named("dim", ints_arg(&[2]))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::ReduceSum {
                dim: 2,
                keepdim: false
            }
        ),
        "expected ReduceSum with dim=2, got: {op:?}"
    );
}

// LSTM, BiLSTM expansion, and repeat_interleave tests.
#[path = "op_map_tests_expand.rs"]
mod expand;

// Scalar binary, squeeze.default, and multi-axis reduce decomposition tests.
#[path = "op_map_tests_decompose.rs"]
mod decompose;

// Comprehensive op mapping coverage: unary, binary, activations, shape,
// reductions, comparison, padding, dpdf, kokoro, expansion, pooling, etc.
#[path = "op_map_tests_coverage.rs"]
mod coverage;
