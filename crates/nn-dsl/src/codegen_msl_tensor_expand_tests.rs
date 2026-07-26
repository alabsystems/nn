// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for native norm and attention op expansion (see #667, #812).

use super::*;
use crate::adain::build_adain1d;
use crate::codegen_msl_tensor::build_dispatch_plan;
use crate::instance_norm::{build_instance_norm, build_instance_norm_decomposed};
use crate::ir::ScalarType;
use crate::layer_norm::build_layer_norm_decomposed;
use crate::rms_norm::{build_rms_norm, build_rms_norm_decomposed};
use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_builders::input_node;
use crate::tensor_ir::ReduceOp;

/// Build a standard attention graph: Q[B,T,D], K[B,T_kv,D], V[B,T_kv,D_v].
fn build_std_attention(
    b: usize,
    t: usize,
    d: usize,
    d_v: usize,
    scale: Option<f32>,
) -> TensorKernelDef {
    let mut bb = TensorBlockBuilder::new("attn_test");
    let q = bb.add_input("q", &[b, t, d]);
    let k = bb.add_input("k", &[b, t, d]);
    let v = bb.add_input("v", &[b, t, d_v]);
    let out = bb.add_attention(q, k, v, AttentionMask::Standard, scale, &[b, t, d_v]);
    bb.build(out).expect("valid attention graph")
}

#[test]
fn expand_instance_norm_non_affine_produces_correct_node_count() {
    let native = build_instance_norm(1, 4, 32).unwrap();
    let expanded = expand_norm_ops(&native);
    let decomposed = build_instance_norm_decomposed(1, 4, 32).unwrap();
    assert_eq!(
        expanded.nodes.len(),
        decomposed.nodes.len(),
        "expanded native should match decomposed node count"
    );
}

#[test]
fn expand_rms_norm_produces_correct_node_count() {
    let native = build_rms_norm(4, 32).unwrap();
    let expanded = expand_norm_ops(&native);
    let decomposed = build_rms_norm_decomposed(4, 32).unwrap();
    assert_eq!(
        expanded.nodes.len(),
        decomposed.nodes.len(),
        "expanded RmsNorm should match decomposed node count"
    );
}

#[test]
fn expand_adain1d_produces_valid_graph() {
    let native = build_adain1d(4, 32).unwrap();
    let expanded = expand_norm_ops(&native);
    let output_node = &expanded.nodes[expanded.output.index()];
    assert_eq!(output_node.shape, vec![4, 32]);
}

#[test]
fn expand_passthrough_preserves_non_norm_graph() {
    let def = TensorKernelDef::new(
        "passthrough",
        vec![
            input_node(0, "x", &[4, 32, 128]),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reduce {
                    op: ReduceOp::Mean,
                    input: TensorNodeId::new(0),
                    axis: 2,
                    keepdim: false,
                },
                vec![4, 32],
            ),
        ],
        TensorNodeId::new(1),
    );
    let expanded = expand_norm_ops(&def);
    assert_eq!(expanded.nodes.len(), 2);
    assert!(matches!(expanded.nodes[0].kind, TensorOpKind::Input { .. }));
    assert!(matches!(
        expanded.nodes[1].kind,
        TensorOpKind::Reduce { .. }
    ));
}

#[test]
fn has_norm_ops_detects_instance_norm() {
    let native = build_instance_norm(1, 4, 32).unwrap();
    assert!(has_norm_ops(&native));
    let decomposed = build_instance_norm_decomposed(1, 4, 32).unwrap();
    assert!(!has_norm_ops(&decomposed));
}

#[test]
fn has_norm_ops_detects_rms_norm() {
    let native = build_rms_norm(4, 32).unwrap();
    assert!(has_norm_ops(&native));
    let decomposed = build_rms_norm_decomposed(4, 32).unwrap();
    assert!(!has_norm_ops(&decomposed));
}

#[test]
fn has_norm_ops_detects_adain1d() {
    let native = build_adain1d(4, 32).unwrap();
    assert!(has_norm_ops(&native));
}

#[test]
fn expanded_graph_has_no_norm_ops() {
    let native_in = build_instance_norm(1, 4, 32).unwrap();
    assert!(!has_norm_ops(&expand_norm_ops(&native_in)));
    let native_rms = build_rms_norm(4, 32).unwrap();
    assert!(!has_norm_ops(&expand_norm_ops(&native_rms)));
    let native_ada = build_adain1d(4, 32).unwrap();
    assert!(!has_norm_ops(&expand_norm_ops(&native_ada)));
}

fn monolithic_layer_norm(batch: usize, hidden: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ln_expand_test");
    let x = b.add_input("x", &[batch, hidden]);
    let eps = b.add_input("eps", &[1]);
    let w = b.add_input("weight", &[hidden]);
    let bias = b.add_input("bias", &[hidden]);
    let out = b.add_layer_norm(x, eps, 1, w, bias, &[batch, hidden]);
    b.build(out).expect("valid graph")
}

#[test]
fn expand_layer_norm_produces_correct_node_count() {
    let native = monolithic_layer_norm(3, 8);
    let expanded = expand_norm_ops(&native);
    let decomposed = build_layer_norm_decomposed(3, 8).unwrap();
    assert_eq!(expanded.nodes.len(), decomposed.nodes.len());
}

#[test]
fn expand_layer_norm_output_shape_preserved() {
    let native = monolithic_layer_norm(3, 8);
    let expanded = expand_norm_ops(&native);
    assert_eq!(expanded.nodes[expanded.output.index()].shape, vec![3, 8]);
}

#[test]
fn expand_layer_norm_no_residual_norm_ops() {
    let native = monolithic_layer_norm(3, 8);
    assert!(has_norm_ops(&native));
    assert!(!has_norm_ops(&expand_norm_ops(&native)));
}

#[test]
fn expand_layer_norm_uses_right_aligned_broadcast() {
    let native = monolithic_layer_norm(3, 8);
    let expanded = expand_norm_ops(&native);
    let count = expanded
        .nodes
        .iter()
        .filter(|n| {
            matches!(&n.kind, TensorOpKind::Broadcast { alignment, .. }
                if *alignment == BroadcastAlignment::Right)
        })
        .count();
    assert_eq!(count, 2);
}

/// Structural equivalence: expanded monolithic matches decomposed K7 (#746).
#[test]
fn expand_layer_norm_structural_equivalence_with_decomposed() {
    let native = monolithic_layer_norm(3, 8);
    let expanded = expand_norm_ops(&native);
    let decomposed = build_layer_norm_decomposed(3, 8).unwrap();
    assert_eq!(expanded.nodes.len(), decomposed.nodes.len());
    for (i, (e, d)) in expanded
        .nodes
        .iter()
        .zip(decomposed.nodes.iter())
        .enumerate()
    {
        assert_eq!(e.shape, d.shape, "node {i}: shape mismatch");
        assert_eq!(
            std::mem::discriminant(&e.kind),
            std::mem::discriminant(&d.kind),
            "node {i}: op variant mismatch"
        );
        if let (
            TensorOpKind::Broadcast { alignment: ea, .. },
            TensorOpKind::Broadcast { alignment: da, .. },
        ) = (&e.kind, &d.kind)
        {
            assert_eq!(ea, da, "node {i}: broadcast alignment mismatch");
        }
    }
    assert_eq!(expanded.output, decomposed.output);
}

// --- Attention expansion tests (#812) ---

#[test]
fn has_attention_ops_detects_standard_attention() {
    let def = build_std_attention(2, 4, 8, 8, None);
    assert!(has_attention_ops(&def));
}

#[test]
fn has_attention_ops_false_for_non_attention_graph() {
    assert!(!has_attention_ops(&build_instance_norm(1, 4, 32).unwrap()));
}

#[test]
fn expand_attention_standard_produces_3_extra_nodes() {
    let def = build_std_attention(2, 4, 8, 8, None);
    assert_eq!(def.nodes.len(), 4);
    assert_eq!(expand_norm_ops(&def).nodes.len(), 6);
}

#[test]
fn expand_attention_standard_output_shape_preserved() {
    let def = build_std_attention(2, 4, 8, 8, None);
    let expanded = expand_norm_ops(&def);
    assert_eq!(expanded.nodes[expanded.output.index()].shape, vec![2, 4, 8]);
}

#[test]
fn expand_attention_standard_no_residual_ops() {
    let def = build_std_attention(2, 4, 8, 8, None);
    assert!(has_attention_ops(&def));
    assert!(!has_attention_ops(&expand_norm_ops(&def)));
}

#[test]
fn expand_attention_standard_decomposition_structure() {
    let def = build_std_attention(2, 4, 8, 8, None);
    let expanded = expand_norm_ops(&def);
    assert!(matches!(
        expanded.nodes[3].kind,
        TensorOpKind::MatMul {
            transpose_right: true,
            ..
        }
    ));
    assert_eq!(expanded.nodes[3].shape, vec![2, 4, 4]);
    assert!(matches!(
        expanded.nodes[4].kind,
        TensorOpKind::Softmax { .. }
    ));
    assert_eq!(expanded.nodes[4].shape, vec![2, 4, 4]);
    assert!(matches!(
        expanded.nodes[5].kind,
        TensorOpKind::MatMul {
            transpose_right: false,
            ..
        }
    ));
    assert_eq!(expanded.nodes[5].shape, vec![2, 4, 8]);
}

#[test]
fn expand_attention_explicit_scale_preserved() {
    let def = build_std_attention(2, 4, 8, 8, Some(0.5));
    let expanded = expand_norm_ops(&def);
    if let TensorOpKind::MatMul { scale, .. } = &expanded.nodes[3].kind {
        assert_eq!(*scale, Some(0.5));
    } else {
        panic!("node 3 should be MatMul");
    }
}

#[test]
fn expand_attention_causal_not_expanded() {
    let mut b = TensorBlockBuilder::new("causal_test");
    let q = b.add_input("q", &[2, 4, 8]);
    let k = b.add_input("k", &[2, 4, 8]);
    let v = b.add_input("v", &[2, 4, 8]);
    let out = b.add_attention(q, k, v, AttentionMask::Causal, None, &[2, 4, 8]);
    let def = b.build(out).unwrap();
    assert!(!has_attention_ops(&def));
    assert!(matches!(
        expand_norm_ops(&def).nodes[3].kind,
        TensorOpKind::Attention { .. }
    ));
}

#[test]
fn dispatch_plan_succeeds_for_standard_attention() {
    let def = build_std_attention(2, 4, 8, 8, None);
    let result = build_dispatch_plan(&def, ScalarType::F32);
    assert!(
        result.is_ok(),
        "dispatch should succeed: {:?}",
        result.err()
    );
}

#[test]
fn dispatch_plan_fails_for_causal_attention() {
    let mut b = TensorBlockBuilder::new("causal_dispatch");
    let q = b.add_input("q", &[2, 4, 8]);
    let k = b.add_input("k", &[2, 4, 8]);
    let v = b.add_input("v", &[2, 4, 8]);
    let out = b.add_attention(q, k, v, AttentionMask::Causal, None, &[2, 4, 8]);
    let def = b.build(out).unwrap();
    assert!(build_dispatch_plan(&def, ScalarType::F32).is_err());
}

// --- LSTM expansion tests (#2306) ---

fn build_lstm_monolithic(batch: usize, input_size: usize, hidden_size: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("lstm_expand_test");
    let x = b.add_input("data", &[batch, input_size]);
    let h = b.add_input("hidden", &[batch, hidden_size]);
    let c = b.add_input("cell", &[batch, hidden_size]);
    let w_ih = b.add_input("w_ih", &[4 * hidden_size, input_size]);
    let w_hh = b.add_input("w_hh", &[4 * hidden_size, hidden_size]);
    let bias = b.add_input("bias", &[4 * hidden_size]);
    let out = b.add_lstm(x, h, c, w_ih, w_hh, Some(bias), &[batch, hidden_size]);
    b.build(out).expect("valid LSTM graph")
}

#[test]
fn has_lstm_ops_detects_lstm() {
    let def = build_lstm_monolithic(1, 8, 4);
    assert!(has_lstm_ops(&def));
}

#[test]
fn has_lstm_ops_false_for_non_lstm_graph() {
    assert!(!has_lstm_ops(&build_instance_norm(1, 4, 32).unwrap()));
}

#[test]
fn expand_lstm_removes_lstm_op() {
    let def = build_lstm_monolithic(1, 8, 4);
    assert!(has_lstm_ops(&def));
    let expanded = expand_norm_ops(&def);
    assert!(!has_lstm_ops(&expanded));
}

#[test]
fn expand_lstm_output_shape_preserved() {
    let def = build_lstm_monolithic(2, 8, 4);
    let expanded = expand_norm_ops(&def);
    assert_eq!(expanded.nodes[expanded.output.index()].shape, vec![2, 4]);
}

#[test]
fn expand_lstm_structural_equivalence_with_decomposed() {
    use crate::lstm_decomposed::build_lstm_cell_decomposed;
    let monolithic = build_lstm_monolithic(2, 8, 4);
    let expanded = expand_norm_ops(&monolithic);
    let decomposed = build_lstm_cell_decomposed(8, 4, 2, true).unwrap();
    assert_eq!(
        expanded.nodes.len(),
        decomposed.nodes.len(),
        "expanded monolithic should match decomposed node count"
    );
    for (i, (e, d)) in expanded
        .nodes
        .iter()
        .zip(decomposed.nodes.iter())
        .enumerate()
    {
        assert_eq!(e.shape, d.shape, "node {i}: shape mismatch");
        assert_eq!(
            std::mem::discriminant(&e.kind),
            std::mem::discriminant(&d.kind),
            "node {i}: op variant mismatch"
        );
    }
}

#[test]
fn dispatch_plan_succeeds_for_lstm() {
    let def = build_lstm_monolithic(1, 8, 4);
    let result = build_dispatch_plan(&def, ScalarType::F32);
    assert!(
        result.is_ok(),
        "dispatch should succeed for LSTM: {:?}",
        result.err()
    );
}
