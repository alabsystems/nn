// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-head attention shape op tests: reshape, transpose, permute, flatten.
//!
//! Transformer models split/merge heads via reshape+transpose+permute.
//! These tests verify the import pipeline maps these ops correctly.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;

use crate::op_map::{map_node_to_trace_op, try_expand_node, OpMapContext, ResolvedWeight};
use crate::parse::{
    Argument, ArgumentInt, ArgumentInts, ArgumentTensor, NamedArgument, Node, TensorArgument,
    TensorMeta,
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

// ---------------------------------------------------------------------------
// Reshape for multi-head attention split
// ---------------------------------------------------------------------------

/// Multi-head attention step 1: reshape [1, 4, 16] -> [1, 4, 2, 8].
///
/// In a transformer with 2 heads and head_dim=8, the hidden dimension (16)
/// is split into (num_heads=2, head_dim=8) via reshape.
#[test]
fn test_reshape_mha_split_heads() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.view.default",
        vec![
            named("input", tensor_arg("hidden")),
            named("size", ints_arg(&[1, 4, 2, 8])),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Reshape { ref target_shape } if target_shape == &[1, 4, 2, 8]),
        "expected Reshape to [1, 4, 2, 8], got: {op:?}"
    );
    assert_eq!(inputs, vec!["hidden"]);
}

/// aten::reshape maps identically to aten::view for this pattern.
#[test]
fn test_reshape_via_reshape_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.reshape.default",
        vec![
            named("input", tensor_arg("hidden")),
            named("size", ints_arg(&[1, 4, 2, 8])),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Reshape { ref target_shape } if target_shape == &[1, 4, 2, 8]),
        "expected Reshape to [1, 4, 2, 8], got: {op:?}"
    );
    assert_eq!(inputs, vec!["hidden"]);
}

/// aten::_unsafe_view is also routed to reshape (used by torch.compile).
#[test]
fn test_reshape_via_unsafe_view() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten._unsafe_view.default",
        vec![
            named("input", tensor_arg("hidden")),
            named("size", ints_arg(&[1, 4, 2, 8])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Reshape { ref target_shape } if target_shape == &[1, 4, 2, 8]),
        "expected Reshape from _unsafe_view, got: {op:?}"
    );
}

// ---------------------------------------------------------------------------
// Transpose for K^T in attention
// ---------------------------------------------------------------------------

/// Multi-head attention step 2: transpose [1, 4, 2, 8] -> [1, 2, 4, 8].
///
/// After splitting heads, we transpose to move the head dimension before
/// the sequence dimension: [B, Seq, Heads, HeadDim] -> [B, Heads, Seq, HeadDim].
#[test]
fn test_transpose_mha_heads_to_front() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.transpose.int",
        vec![
            named("input", tensor_arg("split_heads")),
            named("dim0", int_arg(1)),
            named("dim1", int_arg(2)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Transpose { dim0: 1, dim1: 2 }),
        "expected Transpose(1, 2), got: {op:?}"
    );
    assert_eq!(inputs, vec!["split_heads"]);
}

/// K^T for attention scores: transpose last two dims [1, 2, 4, 8] -> [1, 2, 8, 4].
///
/// This is the key^T transpose in scaled_dot_product_attention: swap Seq and HeadDim.
#[test]
fn test_transpose_key_t_for_attention() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.transpose.int",
        vec![
            named("input", tensor_arg("key")),
            named("dim0", int_arg(2)),
            named("dim1", int_arg(3)),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Transpose { dim0: 2, dim1: 3 }),
        "expected Transpose(2, 3) for K^T, got: {op:?}"
    );
    assert_eq!(inputs, vec!["key"]);
}

// ---------------------------------------------------------------------------
// Permute for arbitrary dimension reordering
// ---------------------------------------------------------------------------

/// Permute: full dimension reorder [B, Seq, Heads, HeadDim] -> [B, Heads, Seq, HeadDim].
///
/// Some models use permute instead of transpose for this operation.
#[test]
fn test_permute_mha_reorder() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.permute.default",
        vec![
            named("input", tensor_arg("qkv")),
            named("dims", ints_arg(&[0, 2, 1, 3])),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Permute { ref axes } if axes == &[0, 2, 1, 3]),
        "expected Permute([0, 2, 1, 3]), got: {op:?}"
    );
    assert_eq!(inputs, vec!["qkv"]);
}

/// Permute: 3D transpose equivalent [B, C, T] -> [B, T, C].
#[test]
fn test_permute_3d_bct_to_btc() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.permute.default",
        vec![
            named("input", tensor_arg("features")),
            named("dims", ints_arg(&[0, 2, 1])),
        ],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Permute { ref axes } if axes == &[0, 2, 1]),
        "expected Permute([0, 2, 1]), got: {op:?}"
    );
}

// ---------------------------------------------------------------------------
// Flatten for merging heads back
// ---------------------------------------------------------------------------

/// Flatten: merge heads [1, 2, 4, 8] -> [1, 2, 32] via flatten(start_dim=2).
///
/// After attention, the output is [B, Heads, Seq, HeadDim]. Flatten merges
/// the last two dims (Seq * HeadDim) before the output projection linear layer.
#[test]
fn test_flatten_merge_seq_and_headdim() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.flatten.using_ints",
        vec![
            named("input", tensor_arg("attn_out")),
            named("start_dim", int_arg(2)),
            named("end_dim", int_arg(3)),
        ],
    );
    let input_shape = &[1, 2, 4, 8];
    let expanded = try_expand_node(&node, &ctx, "flat_out", input_shape)
        .unwrap()
        .expect("flatten should expand");
    assert_eq!(expanded.len(), 1);
    assert!(
        matches!(expanded[0].op, TraceOp::Reshape { ref target_shape } if target_shape == &[1, 2, 32]),
        "expected Reshape to [1, 2, 32], got: {:?}",
        expanded[0].op
    );
    assert_eq!(expanded[0].output_shape, vec![1, 2, 32]);
}

/// Flatten: full flatten [1, 2, 4, 8] -> [1, 64] via flatten(start_dim=1).
///
/// Common before a final classifier linear layer.
#[test]
fn test_flatten_full_except_batch() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.flatten.using_ints",
        vec![
            named("input", tensor_arg("features")),
            named("start_dim", int_arg(1)),
            named("end_dim", int_arg(-1)),
        ],
    );
    let input_shape = &[1, 2, 4, 8];
    let expanded = try_expand_node(&node, &ctx, "flat", input_shape)
        .unwrap()
        .expect("flatten should expand");
    assert_eq!(expanded.len(), 1);
    assert!(
        matches!(expanded[0].op, TraceOp::Reshape { ref target_shape } if target_shape == &[1, 64]),
        "expected Reshape to [1, 64], got: {:?}",
        expanded[0].op
    );
    assert_eq!(expanded[0].output_shape, vec![1, 64]);
}

/// Flatten with default args (start_dim=0, end_dim=-1): total flatten.
#[test]
fn test_flatten_total_default_args() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.flatten.using_ints",
        vec![named("input", tensor_arg("x"))],
    );
    let input_shape = &[2, 3, 4];
    let expanded = try_expand_node(&node, &ctx, "flat", input_shape)
        .unwrap()
        .expect("flatten should expand");
    assert_eq!(expanded.len(), 1);
    assert!(
        matches!(expanded[0].op, TraceOp::Reshape { ref target_shape } if target_shape == &[24]),
        "expected Reshape to [24], got: {:?}",
        expanded[0].op
    );
}

// ---------------------------------------------------------------------------
// Contiguous (identity op)
// ---------------------------------------------------------------------------

/// aten::contiguous is a no-op in graph context (memory layout hint only).
///
/// Maps to Reshape with empty target_shape (identity in the trace compiler).
#[test]
fn test_contiguous_is_identity() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.contiguous.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Reshape { ref target_shape } if target_shape.is_empty()),
        "contiguous maps to Reshape(empty) identity, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

/// aten::clone is also identity in the import graph.
#[test]
fn test_clone_is_identity() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.clone.default",
        vec![named("input", tensor_arg("x"))],
    );
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Reshape { ref target_shape } if target_shape.is_empty()),
        "clone maps to Reshape(empty) identity, got: {op:?}"
    );
}

// ---------------------------------------------------------------------------
// Multi-head attention full pipeline (composed ops)
// ---------------------------------------------------------------------------

/// Full MHA reshape pipeline: test that the sequence of ops produces valid TraceOps.
///
/// Pipeline: input [1, 4, 16]
///   1. reshape [1, 4, 16] -> [1, 4, 2, 8]  (split heads)
///   2. transpose(1, 2) -> [1, 2, 4, 8]      (heads to front)
///   3. transpose(2, 3) -> [1, 2, 8, 4]      (K^T for attention scores)
///
/// This test verifies each step maps to the correct TraceOp variant.
#[test]
fn test_mha_reshape_pipeline() {
    let ctx = empty_ctx();

    // Step 1: reshape [1, 4, 16] -> [1, 4, 2, 8]
    let reshape_node = simple_node(
        "torch.ops.aten.view.default",
        vec![
            named("input", tensor_arg("hidden")),
            named("size", ints_arg(&[1, 4, 2, 8])),
        ],
    );
    let (op1, _) = map_node_to_trace_op(&reshape_node, &ctx, 0).unwrap();
    assert!(matches!(op1, TraceOp::Reshape { ref target_shape } if target_shape == &[1, 4, 2, 8]));

    // Step 2: transpose(1, 2) [1, 4, 2, 8] -> [1, 2, 4, 8]
    let transpose_node = simple_node(
        "torch.ops.aten.transpose.int",
        vec![
            named("input", tensor_arg("reshaped")),
            named("dim0", int_arg(1)),
            named("dim1", int_arg(2)),
        ],
    );
    let (op2, _) = map_node_to_trace_op(&transpose_node, &ctx, 0).unwrap();
    assert!(matches!(op2, TraceOp::Transpose { dim0: 1, dim1: 2 }));

    // Step 3: transpose(2, 3) for K^T [1, 2, 4, 8] -> [1, 2, 8, 4]
    let kt_node = simple_node(
        "torch.ops.aten.transpose.int",
        vec![
            named("input", tensor_arg("heads_first")),
            named("dim0", int_arg(2)),
            named("dim1", int_arg(3)),
        ],
    );
    let (op3, _) = map_node_to_trace_op(&kt_node, &ctx, 0).unwrap();
    assert!(matches!(op3, TraceOp::Transpose { dim0: 2, dim1: 3 }));
}

/// Flatten after attention: [1, 2, 4, 8] -> [1, 64] (flatten all but batch).
///
/// After attention output, flatten merges (heads, seq, head_dim) for the
/// output projection.
#[test]
fn test_mha_flatten_output() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.flatten.using_ints",
        vec![
            named("input", tensor_arg("attn_output")),
            named("start_dim", int_arg(1)),
            named("end_dim", int_arg(-1)),
        ],
    );
    let input_shape = &[1, 2, 4, 8];
    let expanded = try_expand_node(&node, &ctx, "proj_input", input_shape)
        .unwrap()
        .expect("flatten should expand");

    assert_eq!(expanded.len(), 1);
    assert!(
        matches!(expanded[0].op, TraceOp::Reshape { ref target_shape } if target_shape == &[1, 64]),
        "expected [1, 64], got: {:?}",
        expanded[0].op
    );
}

/// Flatten is listed in supported ops.
#[test]
fn test_flatten_in_supported_ops() {
    let ops = crate::op_map::supported_ops();
    assert!(
        ops.contains(&"aten::flatten"),
        "aten::flatten should be in supported ops list"
    );
}
