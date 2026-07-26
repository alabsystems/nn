// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Wave 12 aten op mappers (normalization, embedding, loss overloads).

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

fn weight_ctx(entries: Vec<(&str, Vec<f32>, Vec<usize>)>) -> OpMapContext<'static> {
    let meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::default());
    let mut w = HashMap::new();
    for (name, data, shape) in entries {
        w.insert(name.to_string(), ResolvedWeight::new(data, shape));
    }
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::new(w));
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
// _native_batch_norm_legit
// =======================================================================

#[test]
fn test_map_native_batch_norm_legit() {
    let ctx = weight_ctx(vec![
        ("w", vec![1.0; 3], vec![3]),
        ("b", vec![0.0; 3], vec![3]),
        ("rm", vec![0.0; 3], vec![3]),
        ("rv", vec![1.0; 3], vec![3]),
    ]);
    let node = simple_node(
        "torch.ops.aten._native_batch_norm_legit.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("w")),
            named("bias", tensor_arg("b")),
            named("running_mean", tensor_arg("rm")),
            named("running_var", tensor_arg("rv")),
            named("training", bool_arg(false)),
            named("momentum", float_arg(0.1)),
            named("eps", float_arg(1e-5)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::BatchNorm { eps, .. } if (*eps - 1e-5).abs() < 1e-9),
        "expected BatchNorm with eps=1e-5, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// _native_batch_norm_legit.no_stats
// =======================================================================

#[test]
fn test_map_native_batch_norm_legit_no_stats() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten._native_batch_norm_legit.no_stats",
        vec![
            named("input", tensor_arg("x")),
            named("weight", none_arg()),
            named("bias", none_arg()),
            named("training", bool_arg(true)),
            named("momentum", float_arg(0.1)),
            named("eps", float_arg(1e-5)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("batch_norm_no_stats")),
        "expected batch_norm_no_stats custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// cudnn_batch_norm
// =======================================================================

#[test]
fn test_map_cudnn_batch_norm() {
    let ctx = weight_ctx(vec![
        ("w", vec![1.0; 4], vec![4]),
        ("b", vec![0.0; 4], vec![4]),
        ("rm", vec![0.0; 4], vec![4]),
        ("rv", vec![1.0; 4], vec![4]),
    ]);
    let node = simple_node(
        "torch.ops.aten.cudnn_batch_norm.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("w")),
            named("bias", tensor_arg("b")),
            named("running_mean", tensor_arg("rm")),
            named("running_var", tensor_arg("rv")),
            named("training", bool_arg(false)),
            named("momentum", float_arg(0.1)),
            named("eps", float_arg(1e-5)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::BatchNorm { eps, .. } if (*eps - 1e-5).abs() < 1e-9),
        "expected BatchNorm from cudnn_batch_norm, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// layer_norm with optional None weight/bias
// =======================================================================

#[test]
fn test_map_layer_norm_no_affine() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.layer_norm.no_affine",
        vec![
            named("input", tensor_arg("x")),
            named("weight", none_arg()),
            named("bias", none_arg()),
            named("eps", float_arg(1e-6)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("layer_norm_no_affine")),
        "expected layer_norm_no_affine custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_layer_norm_with_affine() {
    let ctx = weight_ctx(vec![
        ("w", vec![1.0; 8], vec![8]),
        ("b", vec![0.0; 8], vec![8]),
    ]);
    let node = simple_node(
        "torch.ops.aten.layer_norm.no_affine",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("w")),
            named("bias", tensor_arg("b")),
            named("eps", float_arg(1e-5)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::LayerNorm { .. }),
        "expected LayerNorm with affine params, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// group_norm with optional None weight/bias
// =======================================================================

#[test]
fn test_map_group_norm_no_affine() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.group_norm.no_affine",
        vec![
            named("input", tensor_arg("x")),
            named("num_groups", int_arg(4)),
            named("weight", none_arg()),
            named("bias", none_arg()),
            named("eps", float_arg(1e-5)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("group_norm_no_affine")),
        "expected group_norm_no_affine custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_group_norm_with_affine() {
    let ctx = weight_ctx(vec![
        ("w", vec![1.0; 8], vec![8]),
        ("b", vec![0.0; 8], vec![8]),
    ]);
    let node = simple_node(
        "torch.ops.aten.group_norm.no_affine",
        vec![
            named("input", tensor_arg("x")),
            named("num_groups", int_arg(2)),
            named("weight", tensor_arg("w")),
            named("bias", tensor_arg("b")),
            named("eps", float_arg(1e-5)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::GroupNorm { num_groups: 2, .. }),
        "expected GroupNorm with num_groups=2, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// instance_norm with affine parameters
// =======================================================================

#[test]
fn test_map_instance_norm_no_affine() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.instance_norm.affine",
        vec![
            named("input", tensor_arg("x")),
            named("weight", none_arg()),
            named("bias", none_arg()),
            named("eps", float_arg(1e-5)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::InstanceNorm { .. }),
        "expected InstanceNorm, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// =======================================================================
// embedding with padding_idx
// =======================================================================

#[test]
fn test_map_embedding_padding_idx() {
    let ctx = weight_ctx(vec![("emb_w", vec![0.1; 40], vec![10, 4])]);
    let node = simple_node(
        "torch.ops.aten.embedding.padding_idx",
        vec![
            named("weight", tensor_arg("emb_w")),
            named("indices", tensor_arg("ids")),
            named("padding_idx", int_arg(0)),
            named("scale_grad_by_freq", bool_arg(false)),
            named("sparse", bool_arg(false)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Embedding { .. }),
        "expected Embedding, got: {op:?}"
    );
    assert_eq!(inputs, vec!["ids"]);
}

// =======================================================================
// embedding_bag
// =======================================================================

#[test]
fn test_map_embedding_bag() {
    let ctx = weight_ctx(vec![("emb_w", vec![0.1; 40], vec![10, 4])]);
    let node = simple_node(
        "torch.ops.aten._embedding_bag.default",
        vec![
            named("weight", tensor_arg("emb_w")),
            named("indices", tensor_arg("ids")),
            named("offsets", tensor_arg("offs")),
            named("scale_grad_by_freq", bool_arg(false)),
            named("mode", int_arg(1)),
            named("sparse", bool_arg(false)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "embedding_bag_mode1"),
        "expected embedding_bag_mode1 custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["ids", "offs"]);
}

// =======================================================================
// cross_entropy_loss with full parameters
// =======================================================================

#[test]
fn test_map_cross_entropy_loss_full() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cross_entropy_loss.label_smoothing",
        vec![
            named("self", tensor_arg("logits")),
            named("target", tensor_arg("labels")),
            named("weight", none_arg()),
            named("reduction", int_arg(1)),
            named("ignore_index", int_arg(-100)),
            named("label_smoothing", float_arg(0.1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "cross_entropy_loss_r1_ig-100_ls0.1"),
        "expected cross_entropy_loss with label_smoothing=0.1, got: {op:?}"
    );
    assert_eq!(inputs, vec!["logits", "labels"]);
}

#[test]
fn test_map_cross_entropy_loss_full_defaults() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cross_entropy_loss.label_smoothing",
        vec![
            named("self", tensor_arg("logits")),
            named("target", tensor_arg("labels")),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "cross_entropy_loss_r1_ig-100_ls0"),
        "expected cross_entropy_loss with default params, got: {op:?}"
    );
    assert_eq!(inputs, vec!["logits", "labels"]);
}

// =======================================================================
// nll_loss_nd
// =======================================================================

#[test]
fn test_map_nll_loss_nd() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.nll_loss_nd.default",
        vec![
            named("self", tensor_arg("logits")),
            named("target", tensor_arg("labels")),
            named("weight", none_arg()),
            named("reduction", int_arg(1)),
            named("ignore_index", int_arg(-100)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "nll_loss_nd_r1_ig-100"),
        "expected nll_loss_nd custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["logits", "labels"]);
}

// =======================================================================
// nll_loss2d_forward
// =======================================================================

#[test]
fn test_map_nll_loss2d_forward() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.nll_loss2d_forward.default",
        vec![
            named("self", tensor_arg("pred")),
            named("target", tensor_arg("seg_map")),
            named("weight", none_arg()),
            named("reduction", int_arg(2)),
            named("ignore_index", int_arg(255)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "nll_loss2d_forward_r2_ig255"),
        "expected nll_loss2d_forward custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["pred", "seg_map"]);
}

// =======================================================================
// binary_cross_entropy with weight
// =======================================================================

#[test]
fn test_map_binary_cross_entropy_weighted() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.binary_cross_entropy.weight",
        vec![
            named("self", tensor_arg("pred")),
            named("target", tensor_arg("tgt")),
            named("weight", tensor_arg("w")),
            named("reduction", int_arg(1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "binary_cross_entropy_r1_wyes"),
        "expected bce with weight, got: {op:?}"
    );
    assert_eq!(inputs, vec!["pred", "tgt", "w"]);
}

#[test]
fn test_map_binary_cross_entropy_no_weight() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.binary_cross_entropy.weight",
        vec![
            named("self", tensor_arg("pred")),
            named("target", tensor_arg("tgt")),
            named("weight", none_arg()),
            named("reduction", int_arg(0)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "binary_cross_entropy_r0_wno"),
        "expected bce without weight, got: {op:?}"
    );
    assert_eq!(inputs, vec!["pred", "tgt"]);
}

// =======================================================================
// mse_loss backward
// =======================================================================

#[test]
fn test_map_mse_loss_backward() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.mse_loss_backward.default",
        vec![
            named("grad_output", tensor_arg("grad")),
            named("self", tensor_arg("pred")),
            named("target", tensor_arg("tgt")),
            named("reduction", int_arg(1)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "mse_loss_backward_r1"),
        "expected mse_loss_backward custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["grad", "pred", "tgt"]);
}

// =======================================================================
// l1_loss backward
// =======================================================================

#[test]
fn test_map_l1_loss_backward() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.l1_loss_backward.default",
        vec![
            named("grad_output", tensor_arg("grad")),
            named("self", tensor_arg("pred")),
            named("target", tensor_arg("tgt")),
            named("reduction", int_arg(2)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "l1_loss_backward_r2"),
        "expected l1_loss_backward custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["grad", "pred", "tgt"]);
}

// =======================================================================
// smooth_l1_loss backward
// =======================================================================

#[test]
fn test_map_smooth_l1_loss_backward() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.smooth_l1_loss_backward.default",
        vec![
            named("grad_output", tensor_arg("grad")),
            named("self", tensor_arg("pred")),
            named("target", tensor_arg("tgt")),
            named("reduction", int_arg(1)),
            named("beta", float_arg(0.5)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "smooth_l1_loss_backward_r1_b0.5"),
        "expected smooth_l1_loss_backward custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["grad", "pred", "tgt"]);
}

// =======================================================================
// kl_div backward
// =======================================================================

#[test]
fn test_map_kl_div_backward() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.kl_div_backward.default",
        vec![
            named("grad_output", tensor_arg("grad")),
            named("self", tensor_arg("log_p")),
            named("target", tensor_arg("q")),
            named("reduction", int_arg(1)),
            named("log_target", bool_arg(false)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "kl_div_backward_r1_ltfalse"),
        "expected kl_div_backward custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["grad", "log_p", "q"]);
}

#[test]
fn test_map_kl_div_backward_log_target() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.kl_div_backward.default",
        vec![
            named("grad_output", tensor_arg("grad")),
            named("self", tensor_arg("log_p")),
            named("target", tensor_arg("log_q")),
            named("reduction", int_arg(2)),
            named("log_target", bool_arg(true)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name == "kl_div_backward_r2_lttrue"),
        "expected kl_div_backward with log_target=true, got: {op:?}"
    );
    assert_eq!(inputs, vec!["grad", "log_p", "log_q"]);
}

// =======================================================================
// supported_ops includes Wave 12 entries
// =======================================================================

#[test]
fn test_supported_ops_includes_wave12() {
    let ops = supported_ops();
    for expected in &[
        "aten::_native_batch_norm_legit",
        "aten::cudnn_batch_norm",
        "aten::embedding_bag",
        "aten::nll_loss_nd",
        "aten::nll_loss2d_forward",
        "aten::mse_loss_backward",
        "aten::l1_loss_backward",
        "aten::smooth_l1_loss_backward",
        "aten::kl_div_backward",
    ] {
        assert!(
            ops.contains(expected),
            "supported_ops should include {expected}"
        );
    }
}
