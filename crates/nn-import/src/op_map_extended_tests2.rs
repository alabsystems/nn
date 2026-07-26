// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests (batch 2) for the op_map operator mapping infrastructure.
//!
//! Covers: `map_node_to_trace_op()` for common aten ops (linear, relu, gelu,
//! conv1d, matmul, softmax, layer_norm, embedding, cat, reshape, transpose),
//! `supported_ops()` extended verification, `OpMapContext` with populated
//! weight maps, `ResolvedWeight` safetensors integration, error handling for
//! unsupported ops, and edge cases (empty inputs, single-element tensors,
//! batched operations).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;

use crate::error::ImportError;
use crate::op_map::{map_node_to_trace_op, supported_ops, OpMapContext, ResolvedWeight};
use crate::parse::{
    Argument, ArgumentBool, ArgumentFloat, ArgumentInt, ArgumentInts, ArgumentNone, ArgumentString,
    ArgumentTensor, ArgumentTensors, NamedArgument, Node, SymInt, SymIntConcrete, TensorArgument,
    TensorMeta,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

fn none_arg() -> Argument {
    Argument::None(ArgumentNone { as_none: true })
}

fn str_arg(val: &str) -> Argument {
    Argument::Str(ArgumentString {
        as_string: val.to_string(),
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

fn empty_ctx() -> OpMapContext<'static> {
    let meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::default());
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::default());
    OpMapContext {
        tensor_meta: meta,
        weights,
    }
}

/// Build an `OpMapContext` with supplied weights (leaked for 'static lifetime).
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

/// Build an `OpMapContext` with tensor metadata and weights.
fn ctx_with_meta_and_weights(
    meta_entries: Vec<(&str, Vec<i64>, i32)>,
    weight_entries: Vec<(&str, Vec<f32>, Vec<usize>)>,
) -> OpMapContext<'static> {
    let mut meta = HashMap::new();
    for (name, sizes, dtype) in meta_entries {
        meta.insert(
            name.to_string(),
            TensorMeta {
                dtype,
                sizes: sizes
                    .into_iter()
                    .map(|v| SymInt::Concrete(SymIntConcrete { as_int: v }))
                    .collect(),
                requires_grad: false,
                strides: vec![],
                storage_offset: None,
                device: None,
                layout: None,
            },
        );
    }
    let meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::new(meta));
    let mut weights = HashMap::new();
    for (name, data, shape) in weight_entries {
        weights.insert(name.to_string(), ResolvedWeight::new(data, shape));
    }
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::new(weights));
    OpMapContext {
        tensor_meta: meta,
        weights,
    }
}

// =========================================================================
// 1. map_node_to_trace_op — common aten ops
// =========================================================================

// --- aten.linear ---

#[test]
fn test_map_linear_with_bias() {
    let ctx = ctx_with_weights(vec![
        ("w", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        ("b", vec![0.1, 0.2], vec![2]),
    ]);
    let node = simple_node(
        "torch.ops.aten.linear.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("w")),
            named("bias", tensor_arg("b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Linear { ref weight, ref bias } if bias.is_some() && weight.shape() == [2, 2]),
        "expected Linear with bias and weight shape [2,2], got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_linear_without_bias() {
    let ctx = ctx_with_weights(vec![("w", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])]);
    let node = simple_node(
        "torch.ops.aten.linear.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("w")),
            named("bias", none_arg()),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Linear { ref bias, .. } if bias.is_none()),
        "expected Linear without bias, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_linear_missing_weight_returns_error() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.linear.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("missing_w")),
        ],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::MissingWeight { .. }),
        "expected MissingWeight for linear without weight data, got: {err:?}"
    );
}

// --- aten.relu ---

#[test]
fn test_map_relu_basic() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.relu.default",
        vec![named("input", tensor_arg("act"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::Relu));
    assert_eq!(inputs, vec!["act"]);
}

// --- aten.gelu ---

#[test]
fn test_map_gelu_default_approximate_none() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.gelu.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    // Default approximate="none" -> GeluErf
    assert!(
        matches!(op, TraceOp::GeluErf),
        "expected GeluErf for gelu with approximate=none, got: {op:?}"
    );
}

#[test]
fn test_map_gelu_approximate_tanh() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.gelu.default",
        vec![
            named("input", tensor_arg("x")),
            named("approximate", str_arg("tanh")),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Gelu),
        "expected Gelu (tanh approximate) for gelu with approximate=tanh, got: {op:?}"
    );
}

// --- aten.conv1d ---

#[test]
fn test_map_conv1d_standalone() {
    // conv1d.default uses standalone mapper (not convolution.default)
    let ctx = ctx_with_weights(vec![
        ("w_conv", vec![1.0; 24], vec![4, 3, 2]), // out=4, in=3, k=2
    ]);
    let node = simple_node(
        "torch.ops.aten.conv1d.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("w_conv")),
            named("bias", none_arg()),
            named("stride", ints_arg(&[1])),
            named("padding", ints_arg(&[0])),
            named("dilation", ints_arg(&[1])),
            named("groups", int_arg(1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::Conv1d {
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: 1,
                ..
            }
        ),
        "expected Conv1d with stride=1, padding=0, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_convolution_default_1d() {
    // convolution.default routes to Conv1d when weight is 3D
    let ctx = ctx_with_weights(vec![
        ("w_conv", vec![1.0; 12], vec![2, 3, 2]), // out=2, in=3, k=2
        ("b_conv", vec![0.5, 0.6], vec![2]),
    ]);
    let node = simple_node(
        "torch.ops.aten.convolution.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("w_conv")),
            named("bias", tensor_arg("b_conv")),
            named("stride", ints_arg(&[2])),
            named("padding", ints_arg(&[1])),
            named("dilation", ints_arg(&[1])),
            named("transposed", bool_arg(false)),
            named("output_padding", ints_arg(&[0])),
            named("groups", int_arg(1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Conv1d { stride: 2, padding: 1, dilation: 1, groups: 1, ref bias, .. } if bias.is_some()),
        "expected Conv1d stride=2, padding=1 with bias, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// --- aten.matmul ---

#[test]
fn test_map_matmul_default() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.matmul.default",
        vec![
            named("self", tensor_arg("q")),
            named("other", tensor_arg("k")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::MatMul));
    assert_eq!(inputs, vec!["q", "k"]);
}

#[test]
fn test_map_mm_and_bmm_both_produce_matmul() {
    let ctx = empty_ctx();
    for target in &["torch.ops.aten.mm.default", "torch.ops.aten.bmm.default"] {
        let node = simple_node(
            target,
            vec![
                named("self", tensor_arg("a")),
                named("mat2", tensor_arg("b")),
            ],
        );
        let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        assert!(
            matches!(op, TraceOp::MatMul),
            "expected MatMul for {target}"
        );
        assert_eq!(inputs, vec!["a", "b"]);
    }
}

// --- aten.softmax ---

#[test]
fn test_map_softmax_dim1() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.softmax.int",
        vec![
            named("self", tensor_arg("logits")),
            named("dim", int_arg(1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Softmax { dim: 1 }),
        "expected Softmax dim=1, got: {op:?}"
    );
    assert_eq!(inputs, vec!["logits"]);
}

#[test]
fn test_map_softmax_negative_dim_with_ndim() {
    // With input_ndim=3, dim=-1 resolves to dim=2
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.softmax.int",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(-1))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 3).unwrap();
    assert!(
        matches!(op, TraceOp::Softmax { dim: 2 }),
        "expected Softmax dim=2 (resolved from -1 with ndim=3), got: {op:?}"
    );
}

#[test]
fn test_map_softmax_underscore_variant() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten._softmax.default",
        vec![
            named("self", tensor_arg("x")),
            named("dim", int_arg(0)),
            named("half_to_float", bool_arg(false)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Softmax { dim: 0 }),
        "expected Softmax dim=0 for _softmax, got: {op:?}"
    );
}

// --- aten.layer_norm ---

#[test]
fn test_map_layer_norm() {
    let hidden = 4;
    let ctx = ctx_with_weights(vec![
        ("ln_w", vec![1.0; hidden], vec![hidden]),
        ("ln_b", vec![0.0; hidden], vec![hidden]),
    ]);
    let node = simple_node(
        "torch.ops.aten.layer_norm.default",
        vec![
            named("input", tensor_arg("x")),
            named("normalized_shape", ints_arg(&[hidden as i64])),
            named("weight", tensor_arg("ln_w")),
            named("bias", tensor_arg("ln_b")),
            named("eps", float_arg(1e-5)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::LayerNorm { eps, ref weight, ref bias }
            if (eps - 1e-5).abs() < 1e-10 && weight.shape() == [4] && bias.shape() == [4]),
        "expected LayerNorm eps=1e-5, weight/bias shape [4], got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_layer_norm_missing_weight() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.layer_norm.default",
        vec![
            named("input", tensor_arg("x")),
            named("normalized_shape", ints_arg(&[8])),
            named("weight", tensor_arg("missing_ln_w")),
            named("bias", tensor_arg("missing_ln_b")),
        ],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::MissingWeight { .. }),
        "expected MissingWeight, got: {err:?}"
    );
}

// --- aten.embedding ---

#[test]
fn test_map_embedding() {
    let vocab = 100;
    let embed_dim = 16;
    let ctx = ctx_with_weights(vec![(
        "emb_w",
        vec![0.01; vocab * embed_dim],
        vec![vocab, embed_dim],
    )]);
    let node = simple_node(
        "torch.ops.aten.embedding.default",
        vec![
            named("weight", tensor_arg("emb_w")),
            named("indices", tensor_arg("token_ids")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Embedding { ref weight } if weight.shape() == [100, 16]),
        "expected Embedding with weight shape [100,16], got: {op:?}"
    );
    // Embedding takes indices as input, not weight
    assert_eq!(inputs, vec!["token_ids"]);
}

#[test]
fn test_map_embedding_missing_weight() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.embedding.default",
        vec![
            named("weight", tensor_arg("no_such_weight")),
            named("indices", tensor_arg("ids")),
        ],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::MissingWeight { .. }),
        "expected MissingWeight for embedding, got: {err:?}"
    );
}

// --- aten.cat ---

#[test]
fn test_map_cat_two_tensors() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cat.default",
        vec![
            named("tensors", tensors_arg(&["a", "b"])),
            named("dim", int_arg(0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::Cat {
                dim: 0,
                num_inputs: 2
            }
        ),
        "expected Cat dim=0, num_inputs=2, got: {op:?}"
    );
    assert_eq!(inputs, vec!["a", "b"]);
}

#[test]
fn test_map_cat_three_tensors_dim1() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cat.default",
        vec![
            named("tensors", tensors_arg(&["x", "y", "z"])),
            named("dim", int_arg(1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::Cat {
                dim: 1,
                num_inputs: 3
            }
        ),
        "expected Cat dim=1, num_inputs=3, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "y", "z"]);
}

#[test]
fn test_map_cat_default_dim_zero() {
    // When dim argument is missing, cat defaults to dim=0
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cat.default",
        vec![named("tensors", tensors_arg(&["a", "b"]))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Cat { dim: 0, .. }),
        "expected Cat dim=0 as default, got: {op:?}"
    );
}

// --- aten.reshape ---

#[test]
fn test_map_reshape_basic() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.reshape.default",
        vec![
            named("self", tensor_arg("x")),
            named("size", ints_arg(&[2, 3])),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Reshape { ref target_shape } if target_shape == &[2, 3]),
        "expected Reshape [2,3], got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_reshape_with_neg1_infer_dim() {
    // -1 in reshape means infer that dimension. Stored as usize::MAX.
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.reshape.default",
        vec![
            named("self", tensor_arg("x")),
            named("size", ints_arg(&[-1, 4])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Reshape { ref target_shape } if target_shape == &[usize::MAX, 4]),
        "expected Reshape [usize::MAX, 4] for -1, got: {op:?}"
    );
}

#[test]
fn test_map_view_is_reshape_alias() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.view.default",
        vec![
            named("self", tensor_arg("x")),
            named("size", ints_arg(&[1, 6])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Reshape { ref target_shape } if target_shape == &[1, 6]),
        "expected Reshape [1,6] for view, got: {op:?}"
    );
}

// --- aten.transpose ---

#[test]
fn test_map_transpose_basic() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.transpose.int",
        vec![
            named("self", tensor_arg("x")),
            named("dim0", int_arg(0)),
            named("dim1", int_arg(1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Transpose { dim0: 0, dim1: 1 }),
        "expected Transpose dim0=0, dim1=1, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_transpose_higher_dims() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.transpose.int",
        vec![
            named("self", tensor_arg("x")),
            named("dim0", int_arg(1)),
            named("dim1", int_arg(3)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Transpose { dim0: 1, dim1: 3 }),
        "expected Transpose dim0=1, dim1=3, got: {op:?}"
    );
}

// =========================================================================
// 2. supported_ops() — extended verification
// =========================================================================

#[test]
fn test_supported_ops_contains_all_common_aten() {
    let ops = supported_ops();
    let expected = [
        "aten::linear",
        "aten::relu",
        "aten::gelu",
        "aten::conv1d",
        "aten::matmul",
        "aten::softmax",
        "aten::layer_norm",
        "aten::embedding",
        "aten::cat",
        "aten::reshape",
        "aten::transpose",
        "aten::silu",
        "aten::sigmoid",
        "aten::tanh",
        "aten::exp",
        "aten::log",
        "aten::add",
        "aten::sub",
        "aten::mul",
        "aten::div",
        "aten::mm",
        "aten::bmm",
    ];
    for op in &expected {
        assert!(ops.contains(op), "supported_ops() missing common op: {op}");
    }
}

#[test]
fn test_supported_ops_contains_advanced_ops() {
    let ops = supported_ops();
    let advanced = [
        "aten::scaled_dot_product_attention",
        "aten::group_norm",
        "aten::batch_norm",
        "aten::instance_norm",
        "aten::conv2d",
        "aten::conv_transpose1d",
        "aten::lstm",
        "aten::dropout",
        "aten::leaky_relu",
        "aten::elu",
        "aten::where",
        "aten::clamp",
    ];
    for op in &advanced {
        assert!(
            ops.contains(op),
            "supported_ops() missing advanced op: {op}"
        );
    }
}

#[test]
fn test_supported_ops_count_above_100() {
    // As of the current dispatch table, there are well over 100 ops.
    let ops = supported_ops();
    assert!(
        ops.len() > 100,
        "expected >100 supported ops, got {}",
        ops.len()
    );
}

#[test]
fn test_supported_ops_no_torch_prefix() {
    // All entries should use the short aten:: prefix, not the full
    // torch.ops.aten. prefix used in the dispatch table.
    let ops = supported_ops();
    for op in &ops {
        assert!(
            !op.starts_with("torch.ops."),
            "supported_ops() entry should use short aten:: form, got: {op}"
        );
        assert!(
            op.starts_with("aten::"),
            "supported_ops() entry should start with 'aten::', got: {op}"
        );
    }
}

// =========================================================================
// 3. OpMapContext — creation and weight resolution
// =========================================================================

#[test]
fn test_op_map_context_with_multiple_weights() {
    let ctx = ctx_with_weights(vec![
        ("encoder.weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        ("encoder.bias", vec![0.1, 0.2], vec![2]),
        ("decoder.weight", vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]),
    ]);
    assert_eq!(ctx.weights.len(), 3);
    assert_eq!(ctx.weights["encoder.weight"].shape, vec![2, 2]);
    assert_eq!(ctx.weights["encoder.bias"].data, vec![0.1, 0.2]);
    assert_eq!(ctx.weights["decoder.weight"].data.len(), 4);
}

#[test]
fn test_op_map_context_with_tensor_meta() {
    let ctx = ctx_with_meta_and_weights(
        vec![
            ("q", vec![1, 8, 64], 7), // F32, shape [1, 8, 64]
        ],
        vec![],
    );
    assert_eq!(ctx.tensor_meta.len(), 1);
    let q_meta = ctx.tensor_meta.get("q").unwrap();
    let shape = q_meta.concrete_shape().unwrap();
    assert_eq!(shape, vec![1, 8, 64]);
}

#[test]
fn test_op_map_context_weight_drives_linear_op() {
    // Verify that weight data from the context propagates into the TraceOp
    let ctx = ctx_with_weights(vec![("lin_w", vec![2.0, 3.0, 4.0, 5.0], vec![2, 2])]);
    let node = simple_node(
        "torch.ops.aten.linear.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("lin_w")),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match op {
        TraceOp::Linear { weight, bias } => {
            assert_eq!(weight.shape(), &[2, 2]);
            assert_eq!(weight.data(), &[2.0, 3.0, 4.0, 5.0]);
            assert!(bias.is_none());
        }
        other => panic!("expected Linear, got: {other:?}"),
    }
}

// =========================================================================
// 4. ResolvedWeight — types and safetensors integration
// =========================================================================

#[test]
fn test_resolved_weight_scalar() {
    let w = ResolvedWeight::new(vec![42.0], vec![1]);
    assert_eq!(w.data.len(), 1);
    assert_eq!(w.shape, vec![1]);
}

#[test]
fn test_resolved_weight_empty() {
    let w = ResolvedWeight::new(vec![], vec![0]);
    assert!(w.data.is_empty());
    assert_eq!(w.shape, vec![0]);
}

#[test]
fn test_resolved_weight_3d() {
    let w = ResolvedWeight::new(vec![1.0; 24], vec![2, 3, 4]);
    assert_eq!(w.data.len(), 24);
    assert_eq!(w.shape, vec![2, 3, 4]);
}

#[test]
fn test_resolved_weight_from_safetensors_bytes() {
    // Build a minimal safetensors file and verify we can read weights from it
    // and create ResolvedWeight from the data.
    let weight_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes: Vec<u8> = weight_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 3], &bytes)
        .expect("valid tensor view");
    let serialized = safetensors::tensor::serialize(vec![("test_weight".to_string(), view)], None)
        .expect("serialization");

    // Deserialize and verify
    let tensors = safetensors::SafeTensors::deserialize(&serialized).expect("deserialize");
    let t = tensors.tensor("test_weight").expect("find tensor");
    assert_eq!(t.shape(), &[2, 3]);

    // Convert to ResolvedWeight
    let f32_data: Vec<f32> = t
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let rw = ResolvedWeight::new(f32_data, t.shape().to_vec());
    assert_eq!(rw.data, weight_data);
    assert_eq!(rw.shape, vec![2, 3]);
}

// =========================================================================
// 5. Error handling — unsupported ops and missing args
// =========================================================================

#[test]
fn test_unsupported_op_returns_error_with_target() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.custom_unimplemented.default",
        vec![named("input", tensor_arg("x"))],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    match err {
        ImportError::UnsupportedOp { target } => {
            assert!(
                target.contains("custom_unimplemented"),
                "error should contain the op name, got: {target}"
            );
        }
        other => panic!("expected UnsupportedOp, got: {other:?}"),
    }
}

#[test]
fn test_multiple_unsupported_ops_each_get_different_errors() {
    let ctx = empty_ctx();
    let ops = [
        "torch.ops.aten.fake_op_alpha.default",
        "torch.ops.aten.fake_op_beta.default",
        "torch.ops.aten.fake_op_gamma.default",
    ];
    for target in &ops {
        let node = simple_node(target, vec![named("input", tensor_arg("x"))]);
        let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
        assert!(
            matches!(err, ImportError::UnsupportedOp { .. }),
            "expected UnsupportedOp for {target}, got: {err:?}"
        );
    }
}

#[test]
fn test_linear_missing_input_arg_returns_error() {
    // linear requires an "input" tensor arg
    let ctx = ctx_with_weights(vec![("w", vec![1.0, 2.0], vec![1, 2])]);
    let node = simple_node(
        "torch.ops.aten.linear.default",
        vec![
            // missing "input" named arg — provide only weight
            named("weight", tensor_arg("w")),
        ],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::MissingArgument { .. }),
        "expected MissingArgument for linear without input, got: {err:?}"
    );
}

#[test]
fn test_softmax_missing_dim_returns_error() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.softmax.int",
        vec![named("self", tensor_arg("x"))],
        // missing "dim" argument
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::MissingArgument { .. }),
        "expected MissingArgument for softmax without dim, got: {err:?}"
    );
}

#[test]
fn test_transpose_negative_dim_returns_error() {
    // Negative dimensions in transpose should produce NegativeDimension
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.transpose.int",
        vec![
            named("self", tensor_arg("x")),
            named("dim0", int_arg(-2)),
            named("dim1", int_arg(1)),
        ],
    );
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        matches!(err, ImportError::NegativeDimension { .. }),
        "expected NegativeDimension for negative dim in transpose, got: {err:?}"
    );
}

// =========================================================================
// 6. Edge cases — empty inputs, single-element tensors, batched operations
// =========================================================================

#[test]
fn test_map_reshape_single_element() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.reshape.default",
        vec![
            named("self", tensor_arg("x")),
            named("size", ints_arg(&[1])),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Reshape { ref target_shape } if target_shape == &[1]),
        "expected Reshape [1], got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_reshape_scalar_shape() {
    // Reshape to scalar shape (empty dims list -> [])
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.reshape.default",
        vec![named("self", tensor_arg("x")), named("size", ints_arg(&[]))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Reshape { ref target_shape } if target_shape.is_empty()),
        "expected Reshape [], got: {op:?}"
    );
}

#[test]
fn test_map_cat_single_tensor() {
    // Cat with a single tensor is valid (identity-like)
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cat.default",
        vec![
            named("tensors", tensors_arg(&["only"])),
            named("dim", int_arg(0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::Cat {
                dim: 0,
                num_inputs: 1
            }
        ),
        "expected Cat dim=0, num_inputs=1, got: {op:?}"
    );
    assert_eq!(inputs, vec!["only"]);
}

#[test]
fn test_map_linear_large_weight() {
    // Simulate a realistic large linear layer (e.g. 768->3072)
    let in_feat = 768;
    let out_feat = 3072;
    let data: Vec<f32> = (0..in_feat * out_feat)
        .map(|i| (i as f32) * 0.001)
        .collect();
    let bias_data: Vec<f32> = vec![0.0; out_feat];
    let ctx = ctx_with_weights(vec![
        ("big_w", data, vec![out_feat, in_feat]),
        ("big_b", bias_data, vec![out_feat]),
    ]);
    let node = simple_node(
        "torch.ops.aten.linear.default",
        vec![
            named("input", tensor_arg("hidden")),
            named("weight", tensor_arg("big_w")),
            named("bias", tensor_arg("big_b")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match op {
        TraceOp::Linear { weight, bias } => {
            assert_eq!(weight.shape(), &[3072, 768]);
            assert!(bias.is_some());
            assert_eq!(bias.as_ref().unwrap().shape(), &[3072]);
        }
        other => panic!("expected Linear, got: {other:?}"),
    }
    assert_eq!(inputs, vec!["hidden"]);
}

#[test]
fn test_map_conv1d_with_groups() {
    // Depthwise-separable conv1d: groups == in_channels
    let groups = 4;
    let ctx = ctx_with_weights(vec![
        ("dw_w", vec![1.0; groups * 3], vec![groups, 1, 3]), // groups=4, in/groups=1, k=3
    ]);
    let node = simple_node(
        "torch.ops.aten.conv1d.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("dw_w")),
            named("bias", none_arg()),
            named("stride", ints_arg(&[1])),
            named("padding", ints_arg(&[1])),
            named("dilation", ints_arg(&[1])),
            named("groups", int_arg(groups as i64)),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::Conv1d {
                groups: 4,
                padding: 1,
                ..
            }
        ),
        "expected Conv1d groups=4, padding=1, got: {op:?}"
    );
}

#[test]
fn test_map_convolution_default_2d() {
    // convolution.default routes to Conv2d when weight is 4D
    let ctx = ctx_with_weights(vec![
        ("w2d", vec![1.0; 2 * 3 * 3 * 3], vec![2, 3, 3, 3]), // out=2, in=3, kH=3, kW=3
    ]);
    let node = simple_node(
        "torch.ops.aten.convolution.default",
        vec![
            named("input", tensor_arg("img")),
            named("weight", tensor_arg("w2d")),
            named("bias", none_arg()),
            named("stride", ints_arg(&[1, 1])),
            named("padding", ints_arg(&[1, 1])),
            named("dilation", ints_arg(&[1, 1])),
            named("transposed", bool_arg(false)),
            named("output_padding", ints_arg(&[0, 0])),
            named("groups", int_arg(1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::Conv2d {
                padding: [1, 1],
                stride: [1, 1],
                dilation: [1, 1],
                groups: 1,
                ..
            }
        ),
        "expected Conv2d, got: {op:?}"
    );
    assert_eq!(inputs, vec!["img"]);
}

#[test]
fn test_map_sdpa_with_explicit_scale() {
    // SDPA with an explicit scale factor
    let ctx = ctx_with_meta_and_weights(vec![], vec![]);
    let node = simple_node(
        "torch.ops.aten.scaled_dot_product_attention.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
            named("attn_mask", none_arg()),
            named("is_causal", bool_arg(false)),
            named("scale", float_arg(0.125)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Sdpa { scale } if (scale - 0.125).abs() < 1e-10),
        "expected Sdpa scale=0.125, got: {op:?}"
    );
    assert_eq!(inputs, vec!["q", "k", "v"]);
}

#[test]
fn test_map_sdpa_causal() {
    // SDPA with is_causal=true
    let ctx = ctx_with_meta_and_weights(vec![], vec![]);
    let node = simple_node(
        "torch.ops.aten.scaled_dot_product_attention.default",
        vec![
            named("query", tensor_arg("q")),
            named("key", tensor_arg("k")),
            named("value", tensor_arg("v")),
            named("attn_mask", none_arg()),
            named("is_causal", bool_arg(true)),
            named("scale", float_arg(0.125)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::SdpaCausal { scale } if (scale - 0.125).abs() < 1e-10),
        "expected SdpaCausal scale=0.125, got: {op:?}"
    );
    assert_eq!(inputs, vec!["q", "k", "v"]);
}

#[test]
fn test_map_reshape_high_rank() {
    // 5D reshape for video/3D tensor operations
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.reshape.default",
        vec![
            named("self", tensor_arg("x")),
            named("size", ints_arg(&[1, 3, 4, 8, 8])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Reshape { ref target_shape } if target_shape == &[1, 3, 4, 8, 8]),
        "expected Reshape [1,3,4,8,8], got: {op:?}"
    );
}

#[test]
fn test_map_squeeze_basic() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.squeeze.dim",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(0))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Squeeze { dim: 0 }),
        "expected Squeeze dim=0, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_slice_with_start_end() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.slice.Tensor",
        vec![
            named("self", tensor_arg("x")),
            named("dim", int_arg(1)),
            named("start", int_arg(2)),
            named("end", int_arg(10)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::Narrow {
                dim: 1,
                start: 2,
                length: 8
            }
        ),
        "expected Narrow dim=1, start=2, length=8, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_expand_with_neg1_keep_dim() {
    // expand with -1 means keep that dimension unchanged
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.expand.default",
        vec![
            named("self", tensor_arg("x")),
            named("size", ints_arg(&[-1, 4, 8])),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Expand { ref target_shape } if target_shape == &[usize::MAX, 4, 8]),
        "expected Expand [usize::MAX, 4, 8], got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_embedding_single_token_vocab() {
    // Edge case: vocabulary of size 1
    let ctx = ctx_with_weights(vec![("tiny_emb", vec![0.5, 0.6, 0.7, 0.8], vec![1, 4])]);
    let node = simple_node(
        "torch.ops.aten.embedding.default",
        vec![
            named("weight", tensor_arg("tiny_emb")),
            named("indices", tensor_arg("ids")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Embedding { ref weight } if weight.shape() == [1, 4]),
        "expected Embedding [1,4], got: {op:?}"
    );
    assert_eq!(inputs, vec!["ids"]);
}

#[test]
fn test_map_reduce_sum_keepdim() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sum.dim_IntList",
        vec![
            named("self", tensor_arg("x")),
            named("dim", ints_arg(&[1])),
            named("keepdim", bool_arg(true)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::ReduceSum {
                dim: 1,
                keepdim: true
            }
        ),
        "expected ReduceSum dim=1 keepdim=true, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_reduce_mean_no_keepdim() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.mean.dim",
        vec![
            named("self", tensor_arg("x")),
            named("dim", ints_arg(&[2])),
            named("keepdim", bool_arg(false)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::ReduceMean {
                dim: 2,
                keepdim: false
            }
        ),
        "expected ReduceMean dim=2 keepdim=false, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_group_norm() {
    let channels = 8;
    let ctx = ctx_with_weights(vec![
        ("gn_w", vec![1.0; channels], vec![channels]),
        ("gn_b", vec![0.0; channels], vec![channels]),
    ]);
    let node = simple_node(
        "torch.ops.aten.group_norm.default",
        vec![
            named("input", tensor_arg("x")),
            named("num_groups", int_arg(4)),
            named("weight", tensor_arg("gn_w")),
            named("bias", tensor_arg("gn_b")),
            named("eps", float_arg(1e-6)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::GroupNorm { num_groups: 4, eps, .. } if (eps - 1e-6).abs() < 1e-12),
        "expected GroupNorm num_groups=4, eps=1e-6, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_instance_norm() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.instance_norm.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::InstanceNorm { eps } if (eps - 1e-5).abs() < 1e-10),
        "expected InstanceNorm eps=1e-5 (default), got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_batch_norm() {
    let ch = 4;
    let ctx = ctx_with_weights(vec![
        ("bn_w", vec![1.0; ch], vec![ch]),
        ("bn_b", vec![0.0; ch], vec![ch]),
        ("bn_rm", vec![0.5; ch], vec![ch]),
        ("bn_rv", vec![1.0; ch], vec![ch]),
    ]);
    let node = simple_node(
        "torch.ops.aten.native_batch_norm.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("bn_w")),
            named("bias", tensor_arg("bn_b")),
            named("running_mean", tensor_arg("bn_rm")),
            named("running_var", tensor_arg("bn_rv")),
            named("training", bool_arg(false)),
            named("momentum", float_arg(0.1)),
            named("eps", float_arg(1e-5)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::BatchNorm { eps, .. } if (eps - 1e-5).abs() < 1e-10),
        "expected BatchNorm eps=1e-5, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_node_index_parameter_unused() {
    // The `input_ndim` parameter (3rd arg) is only used for dim resolution.
    // Verify that the same op works with different input_ndim values when
    // the op doesn't use relative dims.
    let ctx = empty_ctx();
    for ndim in [0, 1, 2, 5, 10] {
        let node = simple_node(
            "torch.ops.aten.relu.default",
            vec![named("input", tensor_arg("x"))],
        );
        let (op, _) = map_node_to_trace_op(&node, &ctx, ndim).unwrap();
        assert!(
            matches!(op, TraceOp::Relu),
            "relu should work with any ndim={ndim}"
        );
    }
}

#[test]
fn test_map_add_in_place_variant() {
    // add_.Tensor is the in-place variant — should map identically to add.Tensor
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.add_.Tensor",
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
fn test_map_conv_transpose1d() {
    let ctx = ctx_with_weights(vec![
        ("ct_w", vec![1.0; 2 * 4 * 3], vec![2, 4, 3]), // in=2, out=4, k=3
    ]);
    let node = simple_node(
        "torch.ops.aten.conv_transpose1d.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("ct_w")),
            named("bias", none_arg()),
            named("stride", ints_arg(&[2])),
            named("padding", ints_arg(&[1])),
            named("output_padding", ints_arg(&[1])),
            named("groups", int_arg(1)),
            named("dilation", ints_arg(&[1])),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::ConvTranspose1d {
                stride: 2,
                padding: 1,
                output_padding: 1,
                groups: 1,
                ..
            }
        ),
        "expected ConvTranspose1d, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}
