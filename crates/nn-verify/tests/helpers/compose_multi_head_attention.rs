// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: multi-head attention composite builder → NY.
//!
//! Validates that `add_multi_head_attention()` decomposes into
//! Linear(Q,K,V) → Reshape → Transpose → Attention → Transpose → Reshape → Linear(out)
//! and that the resulting `GraphNetwork` propagates bounds via IBP and CROWN.
//!
//! Consolidated: builds each config's graph ONCE and runs all checks,
//! eliminating ~7 redundant graph builds (was 10 builds, now 3).
//!
//! Part of #808.

use super::common::{assert_bounds_valid, assert_crown_tighter_than_ibp, uniform_bounds};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, BoundedTensor, PropMethod,
    TensorParamBinding,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

fn build_mha_kernel(
    name: &str,
    seq_len: usize,
    model_dim: usize,
    num_heads: usize,
    mask: AttentionMask,
) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("input", &[seq_len, model_dim]);
    let q_w = b.add_input("q_weight", &[model_dim, model_dim]);
    let k_w = b.add_input("k_weight", &[model_dim, model_dim]);
    let v_w = b.add_input("v_weight", &[model_dim, model_dim]);
    let out_w = b.add_input("out_weight", &[model_dim, model_dim]);
    let out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            out_w,
            num_heads,
            mask,
            &[seq_len, model_dim],
        )
        .expect("valid MHA");
    b.build(out).expect("valid kernel")
}

fn mha_bindings(model_dim: usize) -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[model_dim, model_dim]), 0.02f32);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w),
    ]
}

fn mha_input_bounds(seq_len: usize, model_dim: usize) -> BoundedTensor {
    uniform_bounds(&[seq_len, model_dim], 1.0)
}

// ===========================================================================
// Consolidated: 2-head MHA (t=4, d=8, h=2, Standard) — all properties
// (was: 6 separate tests, 6 graph builds → 1 test, 1 graph build)
// ===========================================================================

/// Verify MHA IBP bounds are finite and not vacuously wide.
fn check_mha_ibp_bounds(
    graph: &nn_verify::GraphNetwork,
    input: &BoundedTensor,
    t: usize,
    d: usize,
) -> BoundedTensor {
    let ibp_output = graph.propagate_ibp(input).expect("IBP through MHA block");
    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[t, d],
        "output shape [T, D]"
    );
    assert_bounds_valid(&ibp_output);

    let (lo, hi) = ibp_output.lower_upper();
    let max_bound_magnitude = 1e3_f32;
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(
            l.abs() < max_bound_magnitude,
            "lower {l} exceeds {max_bound_magnitude}"
        );
        assert!(
            u.abs() < max_bound_magnitude,
            "upper {u} exceeds {max_bound_magnitude}"
        );
    }
    ibp_output
}

/// Verify CROWN is tighter than IBP for MHA.
fn check_mha_crown_tighter(
    graph: &nn_verify::GraphNetwork,
    input: &BoundedTensor,
    ibp_output: &BoundedTensor,
    t: usize,
    d: usize,
) {
    let (method, crown_output, fallback_reason) =
        propagate_with_crown_fallback(graph, input).expect("CROWN through MHA block");
    assert!(
        matches!(method, PropMethod::Crown),
        "expected CROWN propagation, got {method:?} (fallback: {fallback_reason:?})"
    );
    assert!(
        fallback_reason.is_none(),
        "unexpected CROWN fallback: {fallback_reason:?}"
    );
    assert_eq!(
        crown_output.lower_upper().0.shape(),
        &[t, d],
        "output shape [T, D]"
    );
    assert_bounds_valid(&crown_output);

    assert_crown_tighter_than_ibp(&crown_output, ibp_output);
    let (ibp_lo, ibp_hi) = ibp_output.lower_upper();
    let (crown_lo, crown_hi) = crown_output.lower_upper();
    let ibp_width: f32 = ibp_hi.iter().zip(ibp_lo.iter()).map(|(h, l)| h - l).sum();
    let crown_width: f32 = crown_hi
        .iter()
        .zip(crown_lo.iter())
        .map(|(h, l)| h - l)
        .sum();
    assert!(ibp_width < 1e6, "IBP width should be bounded: {ibp_width}");
    assert!(
        crown_width < 1e6,
        "CROWN width should be bounded: {crown_width}"
    );
    assert!(
        crown_width <= ibp_width + 1e-3,
        "CROWN {crown_width} should be <= IBP {ibp_width}"
    );
}

#[test]
fn test_mha_2head_all_properties() {
    let (t, d, h) = (4, 8, 2);
    let def = build_mha_kernel("mha_2h", t, d, h, AttentionMask::Standard);

    // Scale factor verification
    let attn_node = def
        .nodes
        .iter()
        .find(|n| matches!(n.kind, nn_dsl::tensor_ir::TensorOpKind::Attention { .. }));
    assert!(attn_node.is_some(), "must contain an Attention node");
    if let nn_dsl::tensor_ir::TensorOpKind::Attention { scale, .. } = &attn_node.unwrap().kind {
        let head_dim = d / h;
        let expected = 1.0 / (head_dim as f32).sqrt();
        assert_eq!(
            *scale,
            Some(expected),
            "scale = 1/sqrt({head_dim}) = {expected}"
        );
    }

    assert_eq!(def.nodes.len(), 18, "5 inputs + 13 ops");
    let bindings = mha_bindings(d);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("MHA graph must build");
    assert!(graph.num_nodes() >= 5, "need multiple translation nodes");

    let input = mha_input_bounds(t, d);
    let ibp_output = check_mha_ibp_bounds(&graph, &input, t, d);
    check_mha_crown_tighter(&graph, &input, &ibp_output, t, d);
}

// ===========================================================================
// Consolidated: 4-head MHA (t=6, d=16, h=4) — graph builds + IBP
// (was: 2 separate tests, 2 graph builds → 1 test, 1 graph build)
// ===========================================================================

#[test]
fn test_mha_4head_all_properties() {
    let (t, d, h) = (6, 16, 4);
    let def = build_mha_kernel("mha_4h", t, d, h, AttentionMask::Standard);
    assert_eq!(def.nodes.len(), 18);

    let bindings = mha_bindings(d);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("4-head MHA graph");
    assert!(graph.num_nodes() >= 5);

    let input = mha_input_bounds(t, d);
    let output = graph.propagate_ibp(&input).expect("IBP through 4-head MHA");
    assert_eq!(output.lower_upper().0.shape(), &[t, d]);
    assert_bounds_valid(&output);
}

// ===========================================================================
// Consolidated: Causal MHA (t=4, d=8, h=2, Causal) — graph builds + IBP
// (was: 2 separate tests, 2 graph builds → 1 test, 1 graph build)
// ===========================================================================

#[test]
fn test_mha_causal_all_properties() {
    let (t, d, h) = (4, 8, 2);
    let def = build_mha_kernel("mha_causal", t, d, h, AttentionMask::Causal);
    let bindings = mha_bindings(d);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("causal MHA graph");
    assert!(graph.num_nodes() >= 5);

    let input = mha_input_bounds(t, d);
    let output = graph.propagate_ibp(&input).expect("IBP through causal MHA");
    assert_eq!(output.lower_upper().0.shape(), &[t, d]);
    assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// Validation error tests (no graph builds — keep as-is)
// ---------------------------------------------------------------------------

#[test]
fn test_mha_rejects_indivisible_heads() {
    let mut b = TensorBlockBuilder::new("mha_bad");
    let input = b.add_input("input", &[4, 7]);
    let q_w = b.add_input("q_weight", &[7, 7]);
    let k_w = b.add_input("k_weight", &[7, 7]);
    let v_w = b.add_input("v_weight", &[7, 7]);
    let out_w = b.add_input("out_weight", &[7, 7]);
    let result = b.add_multi_head_attention(
        input,
        q_w,
        k_w,
        v_w,
        out_w,
        2,
        AttentionMask::Standard,
        &[4, 7],
    );
    assert!(result.is_err(), "D=7 with 2 heads should fail");
}

#[test]
fn test_mha_rejects_zero_heads() {
    let mut b = TensorBlockBuilder::new("mha_zero");
    let input = b.add_input("input", &[4, 8]);
    let q_w = b.add_input("q_weight", &[8, 8]);
    let k_w = b.add_input("k_weight", &[8, 8]);
    let v_w = b.add_input("v_weight", &[8, 8]);
    let out_w = b.add_input("out_weight", &[8, 8]);
    let result = b.add_multi_head_attention(
        input,
        q_w,
        k_w,
        v_w,
        out_w,
        0,
        AttentionMask::Standard,
        &[4, 8],
    );
    assert!(result.is_err(), "0 heads should fail");
}

#[test]
fn test_mha_rejects_3d_input() {
    let mut b = TensorBlockBuilder::new("mha_3d");
    let input = b.add_input("input", &[2, 4, 8]);
    let q_w = b.add_input("q_weight", &[8, 8]);
    let k_w = b.add_input("k_weight", &[8, 8]);
    let v_w = b.add_input("v_weight", &[8, 8]);
    let out_w = b.add_input("out_weight", &[8, 8]);
    let result = b.add_multi_head_attention(
        input,
        q_w,
        k_w,
        v_w,
        out_w,
        2,
        AttentionMask::Standard,
        &[2, 4, 8],
    );
    assert!(result.is_err(), "3D input should fail");
}
