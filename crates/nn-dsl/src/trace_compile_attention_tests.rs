// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for attention, composite, and accumulation op compilation.
//!
//! Includes both direct function tests and end-to-end `compile_trace()`
//! tests verifying dispatch routing.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::trace_compile_attention::{
    compile_index_add, compile_mha, compile_rope, compile_scatter_add, compile_sdpa, compile_swiglu,
};
use super::{compile_trace, CompiledStep, NativeOpKind};

use crate::tensor_ir::TensorNode;
use crate::tensor_ir::TensorOpKind;

// -- Helpers ------------------------------------------------------------------

/// Assert a node is MatMul with the expected operand indices, transpose, and scale.
#[track_caller]
fn assert_matmul(
    node: &TensorNode,
    expected_left: usize,
    expected_right: usize,
    expected_transpose_right: bool,
    expected_has_scale: bool,
    label: &str,
) {
    match &node.kind {
        TensorOpKind::MatMul {
            left,
            right,
            transpose_right,
            scale,
        } => {
            assert_eq!(left.index(), expected_left, "{label}: left operand");
            assert_eq!(right.index(), expected_right, "{label}: right operand");
            assert_eq!(
                *transpose_right, expected_transpose_right,
                "{label}: transpose_right"
            );
            assert_eq!(scale.is_some(), expected_has_scale, "{label}: has_scale");
        }
        other => panic!("{label}: expected MatMul, got: {other:?}"),
    }
}

/// Assert a node is Softmax with the expected input node and axis.
#[track_caller]
fn assert_softmax(node: &TensorNode, expected_input: usize, expected_axis: i32, label: &str) {
    match &node.kind {
        TensorOpKind::Softmax { input, axis } => {
            assert_eq!(input.index(), expected_input, "{label}: input node");
            assert_eq!(*axis, expected_axis, "{label}: axis");
        }
        other => panic!("{label}: expected Softmax, got: {other:?}"),
    }
}

fn input_node(id: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape.to_vec(),
        DType::F32,
    )
}

fn ternary_node(
    id: u64,
    name: &str,
    op: TraceOp,
    a_id: u64,
    b_id: u64,
    c_id: u64,
    shape: &[usize],
) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        vec![a_id, b_id, c_id],
        shape.to_vec(),
        DType::F32,
    )
}

fn unary_node(id: u64, name: &str, op: TraceOp, input_id: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        vec![input_id],
        shape.to_vec(),
        DType::F32,
    )
}

// -- Sdpa tests ---------------------------------------------------------------

#[test]
fn test_compile_sdpa_basic() {
    // Q: [2, 4, 8], K: [2, 4, 8], V: [2, 4, 8] → output: [2, 4, 8]
    let scale = 1.0 / (8.0f64).sqrt();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4, 8]),
        input_node(1, &[2, 4, 8]),
        input_node(2, &[2, 4, 8]),
        ternary_node(3, "sdpa", TraceOp::Sdpa { scale }, 0, 1, 2, &[2, 4, 8]),
    ]);
    let node = &graph.nodes()[3];
    let step = compile_sdpa(node, &graph, scale).expect("sdpa should compile");
    assert!(
        matches!(step, CompiledStep::Dispatch { .. }),
        "sdpa should produce Dispatch step"
    );
}

#[test]
fn test_compile_sdpa_different_kv_length() {
    // Q: [1, 8, 64], K: [1, 16, 64], V: [1, 16, 32] → output: [1, 8, 32]
    let scale = 1.0 / (64.0f64).sqrt();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 8, 64]),
        input_node(1, &[1, 16, 64]),
        input_node(2, &[1, 16, 32]),
        ternary_node(3, "sdpa", TraceOp::Sdpa { scale }, 0, 1, 2, &[1, 8, 32]),
    ]);
    let node = &graph.nodes()[3];
    let step = compile_sdpa(node, &graph, scale).expect("sdpa should compile");
    assert!(matches!(step, CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_sdpa_4d() {
    // Q: [B, H, T, D] = [1, 4, 8, 16], K/V same → output: [1, 4, 8, 16]
    // 4D with 3 inputs and D=16 ≤ 128 → NativeOp::FlashAttention (#2434).
    let scale = 1.0 / (16.0f64).sqrt();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8, 16]),
        input_node(1, &[1, 4, 8, 16]),
        input_node(2, &[1, 4, 8, 16]),
        ternary_node(3, "sdpa", TraceOp::Sdpa { scale }, 0, 1, 2, &[1, 4, 8, 16]),
    ]);
    let node = &graph.nodes()[3];
    let step = compile_sdpa(node, &graph, scale).expect("sdpa 4d should compile");
    match &step {
        CompiledStep::NativeOp { op, weight_data } => {
            assert!(matches!(op, NativeOpKind::FlashAttention { .. }));
            assert!(weight_data.is_empty(), "flash attention has no weights");
        }
        other => panic!("4D sdpa should produce NativeOp::FlashAttention, got: {other:?}"),
    }
}

// -- Sdpa IR-level structural assertions (fixes #2331) ------------------------

#[test]
fn test_compile_sdpa_ir_operand_order_and_scale() {
    // Asymmetric shapes: Q=[1,8,64], K=[1,16,64], V=[1,16,32] → output=[1,8,32].
    // Distinct T vs T_kv (8 vs 16) and D_q vs D_v (64 vs 32) make wrong operand
    // order detectable via shape mismatches.
    let scale = 1.0 / (64.0f64).sqrt();
    let scale_f32 = scale as f32;
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 8, 64]),
        input_node(1, &[1, 16, 64]),
        input_node(2, &[1, 16, 32]),
        ternary_node(3, "sdpa", TraceOp::Sdpa { scale }, 0, 1, 2, &[1, 8, 32]),
    ]);
    let node = &graph.nodes()[3];
    let step = compile_sdpa(node, &graph, scale).expect("compile");
    let def = extract_kernel_def(&step);
    let nodes = &def.nodes;

    // 6 nodes: 3 inputs + MatMul(scores) + Softmax + MatMul(output)
    assert_eq!(nodes.len(), 6, "unmasked sdpa should have 6 IR nodes");
    for node in &nodes[..3] {
        assert!(matches!(&node.kind, TensorOpKind::Input { .. }));
    }

    // Node 3: Q @ K^T with scale — transpose_right=true
    assert_matmul(&nodes[3], 0, 1, true, true, "scores matmul");
    let actual_scale = match &nodes[3].kind {
        TensorOpKind::MatMul { scale, .. } => scale.unwrap(),
        _ => unreachable!(),
    };
    assert!(
        (actual_scale - scale_f32).abs() < 1e-7,
        "scale = 1/sqrt(64)"
    );
    assert_eq!(nodes[3].shape, vec![1, 8, 16], "scores shape [B, T, T_kv]");

    // Node 4: Softmax along last axis (rank 3 → axis 2)
    assert_softmax(&nodes[4], 3, 2, "softmax");

    // Node 5: attn @ V — no transpose, no scale
    assert_matmul(&nodes[5], 4, 2, false, false, "output matmul");
    assert_eq!(nodes[5].shape, vec![1, 8, 32], "output shape [B, T, D_v]");
    assert_eq!(def.output.index(), 5, "kernel output should be last node");
}

#[test]
fn test_compile_sdpa_4d_flash_attention_fields() {
    // 4D: Q/K/V=[1,4,8,16] → NativeOp::FlashAttention with correct shapes.
    let scale = 1.0 / (16.0f64).sqrt();
    let scale_f32 = scale as f32;
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8, 16]),
        input_node(1, &[1, 4, 8, 16]),
        input_node(2, &[1, 4, 8, 16]),
        ternary_node(3, "sdpa", TraceOp::Sdpa { scale }, 0, 1, 2, &[1, 4, 8, 16]),
    ]);
    let node = &graph.nodes()[3];
    let step = compile_sdpa(node, &graph, scale).expect("compile 4d");
    let NativeOpKind::FlashAttention {
        scale: actual_scale,
        causal,
        q_shape,
        k_shape,
        output_shape,
        ..
    } = extract_flash_attn_op(&step)
    else {
        panic!("expected FlashAttention");
    };
    assert!(
        (actual_scale - scale_f32).abs() < 1e-7,
        "scale = 1/sqrt(16)"
    );
    assert!(!causal, "non-causal for standard sdpa");
    assert_eq!(q_shape, &[1, 4, 8, 16]);
    assert_eq!(k_shape, &[1, 4, 8, 16]);
    assert_eq!(output_shape, &[1, 4, 8, 16]);
}

#[test]
fn test_compile_sdpa_4d_gqa_flash_attention() {
    // GQA: Q=[1, 8, 32, 64], K/V=[1, 2, 32, 64] (4 groups) → FlashAttention.
    let scale = 1.0 / (64.0f64).sqrt();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 8, 32, 64]),
        input_node(1, &[1, 2, 32, 64]),
        input_node(2, &[1, 2, 32, 64]),
        ternary_node(3, "sdpa", TraceOp::Sdpa { scale }, 0, 1, 2, &[1, 8, 32, 64]),
    ]);
    let node = &graph.nodes()[3];
    let step = compile_sdpa(node, &graph, scale).expect("compile gqa");
    let NativeOpKind::FlashAttention {
        q_shape, k_shape, ..
    } = extract_flash_attn_op(&step)
    else {
        panic!("expected FlashAttention");
    };
    assert_eq!(q_shape, &[1, 8, 32, 64], "Q with 8 heads");
    assert_eq!(k_shape, &[1, 2, 32, 64], "K with 2 heads");
}

#[test]
fn test_compile_sdpa_4d_large_head_dim_falls_back() {
    // D=256 > 128 → should fall back to decomposed Dispatch, not FlashAttention.
    let scale = 1.0 / (256.0f64).sqrt();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8, 256]),
        input_node(1, &[1, 4, 8, 256]),
        input_node(2, &[1, 4, 8, 256]),
        ternary_node(3, "sdpa", TraceOp::Sdpa { scale }, 0, 1, 2, &[1, 4, 8, 256]),
    ]);
    let node = &graph.nodes()[3];
    let step = compile_sdpa(node, &graph, scale).expect("compile large D");
    assert!(
        matches!(step, CompiledStep::Dispatch { .. }),
        "D>128 should fall back to decomposed Dispatch"
    );
}

#[test]
fn test_compile_sdpa_masked_ir_structure() {
    // Masked SDPA: 4 inputs → MatMul + Broadcast + BinaryAdd + Softmax + MatMul.
    // Verify mask is inserted between scores MatMul and Softmax.
    let scale = 1.0 / (8.0f64).sqrt();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4, 8]),
        input_node(1, &[2, 4, 8]),
        input_node(2, &[2, 4, 8]),
        input_node(3, &[2, 4, 4]),
        quaternary_node(
            4,
            "sdpa_masked",
            TraceOp::Sdpa { scale },
            0,
            1,
            2,
            3,
            &[2, 4, 8],
        ),
    ]);
    let node = &graph.nodes()[4];
    let step = compile_sdpa(node, &graph, scale).expect("compile masked");
    let def = extract_kernel_def(&step);

    // 9 nodes: Q(0) K(1) V(2) MatMul(3) Mask(4) Broadcast(5) BinaryAdd(6) Softmax(7) MatMul(8)
    // Mask input is added AFTER the scores MatMul in the builder (conditional block).
    assert_eq!(
        def.nodes.len(),
        9,
        "masked sdpa: 3+1 inputs + 5 ops = 9 nodes"
    );

    // Node 3: scores MatMul (Q @ K^T with scale)
    assert_matmul(&def.nodes[3], 0, 1, true, true, "scores matmul");
    // Node 4: mask input added conditionally after scores MatMul
    assert!(
        matches!(&def.nodes[4].kind, TensorOpKind::Input { .. }),
        "mask input"
    );
    // Node 7: Softmax fed by BinaryAdd(6) of scores + mask
    assert_softmax(&def.nodes[7], 6, 2, "masked softmax");
    // Node 8: final MatMul (attn @ V, no transpose, no scale)
    assert_matmul(&def.nodes[8], 7, 2, false, false, "output matmul");
}

#[test]
fn test_compile_sdpa_nan_scale_rejected() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]),
        input_node(1, &[1, 4, 8]),
        input_node(2, &[1, 4, 8]),
        ternary_node(
            3,
            "sdpa",
            TraceOp::Sdpa { scale: f64::NAN },
            0,
            1,
            2,
            &[1, 4, 8],
        ),
    ]);
    let node = &graph.nodes()[3];
    let err = compile_sdpa(node, &graph, f64::NAN).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("NonFiniteConstant") || msg.contains("Sdpa scale"),
        "NaN scale should produce NonFiniteConstant error, got: {msg}"
    );
}

#[test]
fn test_compile_sdpa_inf_scale_rejected() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]),
        input_node(1, &[1, 4, 8]),
        input_node(2, &[1, 4, 8]),
        ternary_node(
            3,
            "sdpa",
            TraceOp::Sdpa {
                scale: f64::INFINITY,
            },
            0,
            1,
            2,
            &[1, 4, 8],
        ),
    ]);
    let node = &graph.nodes()[3];
    let err = compile_sdpa(node, &graph, f64::INFINITY).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("NonFiniteConstant") || msg.contains("Sdpa scale"),
        "Inf scale should produce NonFiniteConstant error, got: {msg}"
    );
}

#[test]
fn test_compile_sdpa_rank1_rejected() {
    // Q: [8] (rank 1) — too low for matmul
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[8]),
        input_node(1, &[8]),
        input_node(2, &[8]),
        ternary_node(3, "sdpa", TraceOp::Sdpa { scale: 0.5 }, 0, 1, 2, &[8]),
    ]);
    let node = &graph.nodes()[3];
    let err = compile_sdpa(node, &graph, 0.5).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("rank too low"),
        "rank-1 should be rejected, got: {msg}"
    );
}

// -- Sdpa input count validation (fixes #2326) --------------------------------

#[test]
fn test_compile_sdpa_too_few_inputs_rejected() {
    // Only 2 inputs (Q, K) — missing V
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4, 8]),
        input_node(1, &[2, 4, 8]),
        TraceNode::new(
            2,
            "sdpa_bad".into(),
            TraceOp::Sdpa { scale: 0.5 },
            vec![0, 1],
            vec![2, 4, 8],
            DType::F32,
        ),
    ]);
    let node = &graph.nodes()[2];
    let err = compile_sdpa(node, &graph, 0.5).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("3 or 4") && msg.contains("got 2"),
        "expected input count error, got: {msg}"
    );
}

#[test]
fn test_compile_sdpa_too_many_inputs_rejected() {
    // 5 inputs — Q, K, V, mask, extra
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4, 8]),
        input_node(1, &[2, 4, 8]),
        input_node(2, &[2, 4, 8]),
        input_node(3, &[2, 4, 4]),
        input_node(4, &[2, 4, 8]),
        TraceNode::new(
            5,
            "sdpa_extra".into(),
            TraceOp::Sdpa { scale: 0.5 },
            vec![0, 1, 2, 3, 4],
            vec![2, 4, 8],
            DType::F32,
        ),
    ]);
    let node = &graph.nodes()[5];
    let err = compile_sdpa(node, &graph, 0.5).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("3 or 4") && msg.contains("got 5"),
        "expected input count error, got: {msg}"
    );
}

// -- Sdpa with mask tests (fixes #2284) ---------------------------------------

fn quaternary_node(
    id: u64,
    name: &str,
    op: TraceOp,
    a_id: u64,
    b_id: u64,
    c_id: u64,
    d_id: u64,
    shape: &[usize],
) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        vec![a_id, b_id, c_id, d_id],
        shape.to_vec(),
        DType::F32,
    )
}

#[test]
fn test_compile_sdpa_with_mask() {
    // Q: [2, 4, 8], K: [2, 4, 8], V: [2, 4, 8], mask: [2, 4, 4] → output: [2, 4, 8]
    let scale = 1.0 / (8.0f64).sqrt();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4, 8]),
        input_node(1, &[2, 4, 8]),
        input_node(2, &[2, 4, 8]),
        input_node(3, &[2, 4, 4]),
        quaternary_node(
            4,
            "sdpa_masked",
            TraceOp::Sdpa { scale },
            0,
            1,
            2,
            3,
            &[2, 4, 8],
        ),
    ]);
    let node = &graph.nodes()[4];
    let step = compile_sdpa(node, &graph, scale).expect("masked sdpa should compile");
    assert!(
        matches!(step, CompiledStep::Dispatch { .. }),
        "masked sdpa should produce Dispatch step"
    );
}

#[test]
fn test_sdpa_masked_dispatch_plan_has_more_steps_than_unmasked() {
    // Masked SDPA adds Broadcast + BinaryAdd between MatMul and Softmax.
    let scale = 1.0 / (8.0f64).sqrt();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4, 8]),
        input_node(1, &[2, 4, 8]),
        input_node(2, &[2, 4, 8]),
        input_node(3, &[2, 4, 4]),
        quaternary_node(
            4,
            "sdpa_masked",
            TraceOp::Sdpa { scale },
            0,
            1,
            2,
            3,
            &[2, 4, 8],
        ),
    ]);
    let node = &graph.nodes()[4];
    let step = compile_sdpa(node, &graph, scale).expect("compile");
    let def = extract_kernel_def(&step);
    let (plan, _) = crate::codegen_msl_tensor::build_dispatch_plan(def, crate::ir::ScalarType::F32)
        .expect("dispatch plan");
    // Unmasked: MatMul + Softmax + MatMul = 3 steps
    // Masked: MatMul + Broadcast + BinaryAdd + Softmax + MatMul = 5 steps
    assert_eq!(
        plan.len(),
        5,
        "masked sdpa should have exactly 5 dispatch steps (MatMul+Broadcast+BinaryAdd+Softmax+MatMul), got: {}",
        plan.len()
    );
}

#[test]
fn test_sdpa_masked_msl_emission_succeeds() {
    let scale = 1.0 / (8.0f64).sqrt();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4, 8]),
        input_node(1, &[2, 4, 8]),
        input_node(2, &[2, 4, 8]),
        input_node(3, &[2, 4, 4]),
        quaternary_node(
            4,
            "sdpa_masked",
            TraceOp::Sdpa { scale },
            0,
            1,
            2,
            3,
            &[2, 4, 8],
        ),
    ]);
    let node = &graph.nodes()[4];
    let step = compile_sdpa(node, &graph, scale).expect("compile");
    let def = extract_kernel_def(&step);
    let msl = crate::codegen_msl_tensor_emit::emit_tensor_msl(def, crate::ir::ScalarType::F32)
        .expect("masked MSL emission");
    assert!(
        msl.contains("[[kernel]]"),
        "masked MSL should contain kernel attribute"
    );
}

#[test]
fn test_sdpa_4d_with_mask() {
    // Q/K/V: [B, H, T, D] = [2, 4, 8, 16], mask: [1, 1, 8, 8] (broadcast over batch+heads)
    // B=2 exercises non-vacuous batch broadcast from mask dim 0 (1→2).
    let scale = 1.0 / (16.0f64).sqrt();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4, 8, 16]),
        input_node(1, &[2, 4, 8, 16]),
        input_node(2, &[2, 4, 8, 16]),
        input_node(3, &[1, 1, 8, 8]),
        quaternary_node(
            4,
            "sdpa_4d_masked",
            TraceOp::Sdpa { scale },
            0,
            1,
            2,
            3,
            &[2, 4, 8, 16],
        ),
    ]);
    let node = &graph.nodes()[4];
    let step = compile_sdpa(node, &graph, scale).expect("4d masked sdpa should compile");
    assert!(matches!(step, CompiledStep::Dispatch { .. }));
}

#[test]
fn test_sdpa_2d_mask_broadcast() {
    // Q/K/V: [2, 4, 8], mask: [4, 4] — 2D mask broadcast to 3D scores [2, 4, 4].
    // PyTorch common pattern: 2D causal mask [T, T_kv] broadcast via right-alignment.
    let scale = 1.0 / (8.0f64).sqrt();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4, 8]),
        input_node(1, &[2, 4, 8]),
        input_node(2, &[2, 4, 8]),
        input_node(3, &[4, 4]),
        quaternary_node(
            4,
            "sdpa_2d_mask",
            TraceOp::Sdpa { scale },
            0,
            1,
            2,
            3,
            &[2, 4, 8],
        ),
    ]);
    let node = &graph.nodes()[4];
    let step = compile_sdpa(node, &graph, scale).expect("2d mask broadcast sdpa should compile");
    assert!(
        matches!(step, CompiledStep::Dispatch { .. }),
        "2d mask sdpa should produce Dispatch step"
    );
}

// -- SwiGlu tests -------------------------------------------------------------

#[test]
fn test_compile_swiglu_passthrough() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 3]),
        unary_node(1, "swiglu", TraceOp::SwiGlu, 0, &[2, 3]),
    ]);
    let node = &graph.nodes()[1];
    let step = compile_swiglu(node, &graph).expect("swiglu should compile");
    assert!(matches!(step, CompiledStep::IdentityPassthrough));
}

// -- MultiHeadAttention tests -------------------------------------------------

#[test]
fn test_compile_mha_passthrough() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 8, 64]),
        unary_node(
            1,
            "mha",
            TraceOp::MultiHeadAttention {
                num_heads: 4,
                num_kv_heads: 4,
                head_dim: 16,
            },
            0,
            &[1, 8, 64],
        ),
    ]);
    let node = &graph.nodes()[1];
    let step = compile_mha(node, &graph).expect("mha should compile");
    assert!(matches!(step, CompiledStep::IdentityPassthrough));
}

// -- RotaryEmbedding tests ----------------------------------------------------

#[test]
fn test_compile_rope_success() {
    let cos = WeightRef::from_shape(&[8, 32]);
    let sin = WeightRef::from_shape(&[8, 32]);
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 8, 64]),
        unary_node(
            1,
            "rope",
            TraceOp::RotaryEmbedding {
                head_dim: 64,
                offset: 0,
                cos_cache: cos.clone(),
                sin_cache: sin.clone(),
            },
            0,
            &[1, 8, 64],
        ),
    ]);
    let node = &graph.nodes()[1];
    let step = compile_rope(node, &graph, 64, &cos, &sin).expect("rope should compile");
    match &step {
        CompiledStep::NativeOp { op, weight_data } => {
            assert!(matches!(
                op,
                NativeOpKind::RotaryEmbedding { head_dim: 64, .. }
            ));
            assert!(weight_data.contains_key("cos_cache"));
            assert!(weight_data.contains_key("sin_cache"));
        }
        other => panic!("expected NativeOp RotaryEmbedding, got {other:?}"),
    }
}

#[test]
fn test_compile_rope_odd_head_dim_rejected() {
    let cos = WeightRef::from_shape(&[8, 16]);
    let sin = WeightRef::from_shape(&[8, 16]);
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 8, 33]),
        unary_node(
            1,
            "rope",
            TraceOp::RotaryEmbedding {
                head_dim: 33,
                offset: 0,
                cos_cache: cos.clone(),
                sin_cache: sin.clone(),
            },
            0,
            &[1, 8, 33],
        ),
    ]);
    let node = &graph.nodes()[1];
    let err = compile_rope(node, &graph, 33, &cos, &sin).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("even"),
        "odd head_dim should be rejected, got: {msg}"
    );
}

// -- ScatterAdd tests ---------------------------------------------------------

#[test]
fn test_compile_scatter_add_unsupported() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 8]),
        input_node(1, &[4, 8]),
        input_node(2, &[4, 8]),
        ternary_node(
            3,
            "scatter_add",
            TraceOp::ScatterAdd { dim: 0 },
            0,
            1,
            2,
            &[4, 8],
        ),
    ]);
    let node = &graph.nodes()[3];
    let err = compile_scatter_add(node, &graph, 0).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("scatter_add") || msg.contains("atomic"),
        "scatter_add should produce informative error, got: {msg}"
    );
}

// -- IndexAdd tests -----------------------------------------------------------

#[test]
fn test_compile_index_add_unsupported() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 8]),
        input_node(1, &[4]),
        input_node(2, &[4, 8]),
        ternary_node(
            3,
            "index_add",
            TraceOp::IndexAdd { dim: 0 },
            0,
            1,
            2,
            &[4, 8],
        ),
    ]);
    let node = &graph.nodes()[3];
    let err = compile_index_add(node, &graph, 0).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("index_add") || msg.contains("atomic"),
        "index_add should produce informative error, got: {msg}"
    );
}

// -- Sdpa MSL codegen pipeline tests ------------------------------------------

fn sdpa_graph(shape: &[usize], scale: f64) -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        input_node(0, shape),
        input_node(1, shape),
        input_node(2, shape),
        ternary_node(3, "sdpa", TraceOp::Sdpa { scale }, 0, 1, 2, shape),
    ])
}

fn extract_kernel_def(step: &CompiledStep) -> &crate::tensor_ir::TensorKernelDef {
    match step {
        CompiledStep::Dispatch { kernel, .. } => kernel.def(),
        other => panic!("expected Dispatch, got: {other:?}"),
    }
}

fn extract_flash_attn_op(step: &CompiledStep) -> &NativeOpKind {
    match step {
        CompiledStep::NativeOp { op, .. } => {
            assert!(
                matches!(op, NativeOpKind::FlashAttention { .. }),
                "expected FlashAttention variant, got: {op:?}"
            );
            op
        }
        other => panic!("expected NativeOp, got: {other:?}"),
    }
}

#[test]
fn test_sdpa_dispatch_plan_has_3_steps() {
    let scale = 1.0 / (8.0f64).sqrt();
    let graph = sdpa_graph(&[2, 4, 8], scale);
    let step = compile_sdpa(&graph.nodes()[3], &graph, scale).expect("compile");
    let def = extract_kernel_def(&step);
    let (plan, _) = crate::codegen_msl_tensor::build_dispatch_plan(def, crate::ir::ScalarType::F32)
        .expect("dispatch plan");
    assert_eq!(
        plan.len(),
        3,
        "MatMul+Softmax+MatMul = 3 steps, got: {}",
        plan.len()
    );
}

#[test]
fn test_sdpa_msl_emission_succeeds() {
    let scale = 1.0 / (8.0f64).sqrt();
    let graph = sdpa_graph(&[2, 4, 8], scale);
    let step = compile_sdpa(&graph.nodes()[3], &graph, scale).expect("compile");
    let def = extract_kernel_def(&step);
    let msl = crate::codegen_msl_tensor_emit::emit_tensor_msl(def, crate::ir::ScalarType::F32)
        .expect("MSL emission");
    assert!(
        msl.starts_with("#include <metal_stdlib>"),
        "MSL should start with prelude"
    );
    assert!(
        msl.contains("[[kernel]]"),
        "MSL should contain kernel attribute"
    );
}

#[test]
fn test_sdpa_compiled_plan_generates_msl() {
    let scale = 1.0 / (8.0f64).sqrt();
    let graph = sdpa_graph(&[2, 4, 8], scale);
    let plan = super::compile_trace_to_plan(&graph).expect("plan");
    let sources = plan
        .generate_msl(crate::ir::ScalarType::F32)
        .expect("MSL gen");
    assert_eq!(sources.len(), 1, "should produce 1 MSL source for sdpa");
    assert_eq!(sources[0].kernel_name, "sdpa");
    assert!(
        sources[0].msl.contains("[[kernel]]"),
        "MSL should have kernel fn"
    );
}

#[test]
fn test_sdpa_4d_native_op_no_msl() {
    // 4D SDPA compiles to NativeOp::FlashAttention — no MSL generated.
    let scale = 1.0 / (16.0f64).sqrt();
    let graph = sdpa_graph(&[1, 4, 8, 16], scale);
    let plan = super::compile_trace_to_plan(&graph).expect("4d plan");
    let sources = plan
        .generate_msl(crate::ir::ScalarType::F32)
        .expect("4d MSL");
    // NativeOp steps don't produce MSL — the kernel is pre-compiled.
    assert_eq!(
        sources.len(),
        0,
        "NativeOp::FlashAttention produces no MSL sources"
    );
    // Verify the plan step is indeed FlashAttention.
    let sdpa_step = &plan.steps[plan.steps.len() - 1];
    assert!(
        matches!(
            sdpa_step,
            CompiledStep::NativeOp {
                op: NativeOpKind::FlashAttention { .. },
                ..
            }
        ),
        "4D SDPA should compile to NativeOp::FlashAttention, got: {:?}",
        std::mem::discriminant(sdpa_step)
    );
}

// -- End-to-end compile_trace() tests (dispatch routing) ----------------------

#[test]
fn test_e2e_sdpa_through_dispatch() {
    let scale = 1.0 / (8.0f64).sqrt();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4, 8]),
        input_node(1, &[2, 4, 8]),
        input_node(2, &[2, 4, 8]),
        ternary_node(3, "sdpa", TraceOp::Sdpa { scale }, 0, 1, 2, &[2, 4, 8]),
    ]);
    let steps = compile_trace(&graph).expect("sdpa should compile through dispatch");
    assert_eq!(steps.len(), 4);
    assert!(matches!(steps[0], CompiledStep::InputForward));
    assert!(matches!(steps[1], CompiledStep::InputForward));
    assert!(matches!(steps[2], CompiledStep::InputForward));
    assert!(
        matches!(steps[3], CompiledStep::Dispatch { .. }),
        "sdpa dispatch step expected, got: {:?}",
        std::mem::discriminant(&steps[3])
    );
}

#[test]
fn test_e2e_sdpa_masked_through_dispatch() {
    let scale = 1.0 / (8.0f64).sqrt();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4, 8]),
        input_node(1, &[2, 4, 8]),
        input_node(2, &[2, 4, 8]),
        input_node(3, &[2, 4, 4]),
        quaternary_node(
            4,
            "sdpa_masked",
            TraceOp::Sdpa { scale },
            0,
            1,
            2,
            3,
            &[2, 4, 8],
        ),
    ]);
    let steps = compile_trace(&graph).expect("masked sdpa should compile through dispatch");
    // 4 InputForward + 1 Dispatch = 5
    assert_eq!(
        steps.len(),
        5,
        "4 inputs + 1 dispatch = 5 steps, got: {}",
        steps.len()
    );
    assert!(matches!(steps[0], CompiledStep::InputForward));
    assert!(matches!(steps[1], CompiledStep::InputForward));
    assert!(matches!(steps[2], CompiledStep::InputForward));
    assert!(matches!(steps[3], CompiledStep::InputForward));
    assert!(
        matches!(steps[4], CompiledStep::Dispatch { .. }),
        "masked sdpa dispatch step expected"
    );
}

#[test]
fn test_e2e_swiglu_through_dispatch() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 3]),
        unary_node(1, "swiglu", TraceOp::SwiGlu, 0, &[2, 3]),
    ]);
    let steps = compile_trace(&graph).expect("swiglu should compile through dispatch");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
}

#[test]
fn test_e2e_mha_through_dispatch() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 8, 64]),
        unary_node(
            1,
            "mha",
            TraceOp::MultiHeadAttention {
                num_heads: 4,
                num_kv_heads: 4,
                head_dim: 16,
            },
            0,
            &[1, 8, 64],
        ),
    ]);
    let steps = compile_trace(&graph).expect("mha should compile through dispatch");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
}

#[test]
fn test_e2e_rope_native_op_through_dispatch() {
    // RoPE is now implemented as a NativeOp
    let cos = WeightRef::from_shape(&[8, 32]);
    let sin = WeightRef::from_shape(&[8, 32]);
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 8, 64]),
        unary_node(
            1,
            "rope",
            TraceOp::RotaryEmbedding {
                head_dim: 64,
                offset: 0,
                cos_cache: cos,
                sin_cache: sin,
            },
            0,
            &[1, 8, 64],
        ),
    ]);
    let steps = compile_trace(&graph).expect("rope should compile as NativeOp");
    assert_eq!(steps.len(), 2);
    assert!(
        matches!(&steps[1], CompiledStep::NativeOp { .. }),
        "rope should compile to NativeOp, got: {:?}",
        steps[1]
    );
}

#[test]
fn test_e2e_scatter_add_unsupported_through_dispatch() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 8]),
        input_node(1, &[4, 8]),
        input_node(2, &[4, 8]),
        ternary_node(
            3,
            "scatter_add",
            TraceOp::ScatterAdd { dim: 0 },
            0,
            1,
            2,
            &[4, 8],
        ),
    ]);
    let err = compile_trace(&graph).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("scatter_add") || msg.contains("atomic"),
        "scatter_add should be unsupported through dispatch, got: {msg}"
    );
}
