// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `SdpaCausal` compilation — both direct function tests and
//! end-to-end `compile_trace()` dispatch routing.
//!
//! Extracted from `trace_compile_attention_tests.rs` to stay under the
//! 1000-line file limit.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use super::trace_compile_attention::compile_sdpa_causal;
use super::{compile_trace, CompiledStep, NativeOpKind};

// -- Helpers (shared with attention_tests, duplicated for module isolation) ----

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

// -- SdpaCausal tests ---------------------------------------------------------

#[test]
fn test_compile_sdpa_causal_4d_flash_attention() {
    // 4D causal → NativeOp::FlashAttention { causal: true }
    let scale = 1.0 / (64.0f64).sqrt();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 32, 64]),
        input_node(1, &[1, 4, 32, 64]),
        input_node(2, &[1, 4, 32, 64]),
        ternary_node(
            3,
            "sdpa_causal",
            TraceOp::SdpaCausal { scale },
            0,
            1,
            2,
            &[1, 4, 32, 64],
        ),
    ]);
    let node = &graph.nodes()[3];
    let step = compile_sdpa_causal(node, &graph, scale).expect("causal 4d should compile");
    match step {
        CompiledStep::NativeOp { op, .. } => match op {
            NativeOpKind::FlashAttention {
                scale: s,
                causal,
                q_shape,
                k_shape,
                output_shape,
                ..
            } => {
                assert!((s - scale as f32).abs() < 1e-6, "scale mismatch");
                assert!(causal, "causal must be true");
                assert_eq!(q_shape, vec![1, 4, 32, 64]);
                assert_eq!(k_shape, vec![1, 4, 32, 64]);
                assert_eq!(output_shape, vec![1, 4, 32, 64]);
            }
            other => panic!("expected FlashAttention, got: {other:?}"),
        },
        other => panic!("expected NativeOp, got: {other:?}"),
    }
}

#[test]
fn test_compile_sdpa_causal_gqa() {
    // GQA causal: Q=[1,8,32,64], K/V=[1,2,32,64]
    let scale = 1.0 / (64.0f64).sqrt();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 8, 32, 64]),
        input_node(1, &[1, 2, 32, 64]),
        input_node(2, &[1, 2, 32, 64]),
        ternary_node(
            3,
            "sdpa_causal",
            TraceOp::SdpaCausal { scale },
            0,
            1,
            2,
            &[1, 8, 32, 64],
        ),
    ]);
    let node = &graph.nodes()[3];
    let step = compile_sdpa_causal(node, &graph, scale).expect("compile gqa causal");
    match step {
        CompiledStep::NativeOp {
            op: NativeOpKind::FlashAttention { causal, .. },
            ..
        } => {
            assert!(causal, "GQA causal should be true");
        }
        other => panic!("expected NativeOp::FlashAttention, got: {other:?}"),
    }
}

#[test]
fn test_compile_sdpa_causal_3d_decomposed() {
    // 3D causal → decomposed Dispatch (not eligible for FlashAttention)
    let scale = 0.5;
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4, 8]),
        input_node(1, &[2, 4, 8]),
        input_node(2, &[2, 4, 8]),
        ternary_node(
            3,
            "sdpa_causal",
            TraceOp::SdpaCausal { scale },
            0,
            1,
            2,
            &[2, 4, 8],
        ),
    ]);
    let node = &graph.nodes()[3];
    let step = compile_sdpa_causal(node, &graph, scale).expect("3d causal should compile");
    assert!(
        matches!(step, CompiledStep::Dispatch { .. }),
        "3D causal should decompose to Dispatch"
    );
}

#[test]
fn test_compile_sdpa_causal_large_head_dim_decomposed() {
    // 4D with D=256 > 128 → decomposed (not eligible for FlashAttention)
    let scale = 0.5;
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8, 256]),
        input_node(1, &[1, 4, 8, 256]),
        input_node(2, &[1, 4, 8, 256]),
        ternary_node(
            3,
            "sdpa_causal",
            TraceOp::SdpaCausal { scale },
            0,
            1,
            2,
            &[1, 4, 8, 256],
        ),
    ]);
    let node = &graph.nodes()[3];
    let step = compile_sdpa_causal(node, &graph, scale).expect("compile large D");
    assert!(
        matches!(step, CompiledStep::Dispatch { .. }),
        "D=256 should fall back to decomposed"
    );
}

#[test]
fn test_compile_sdpa_causal_nan_scale_rejected() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4, 8]),
        input_node(1, &[2, 4, 8]),
        input_node(2, &[2, 4, 8]),
        ternary_node(
            3,
            "sdpa_causal",
            TraceOp::SdpaCausal { scale: f64::NAN },
            0,
            1,
            2,
            &[2, 4, 8],
        ),
    ]);
    let node = &graph.nodes()[3];
    let err = compile_sdpa_causal(node, &graph, f64::NAN).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("NonFiniteConstant") || msg.contains("non_finite"),
        "NaN scale should be rejected: {msg}"
    );
}

#[test]
fn test_compile_sdpa_causal_rank1_rejected() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[8]),
        input_node(1, &[8]),
        input_node(2, &[8]),
        ternary_node(
            3,
            "sdpa_causal",
            TraceOp::SdpaCausal { scale: 0.5 },
            0,
            1,
            2,
            &[8],
        ),
    ]);
    let node = &graph.nodes()[3];
    let err = compile_sdpa_causal(node, &graph, 0.5).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("rank too low"),
        "rank-1 should be rejected: {msg}"
    );
}

#[test]
fn test_e2e_sdpa_causal_4d_through_dispatch() {
    // End-to-end: 4D causal SDPA → 3 InputForward + 1 NativeOp
    let scale = 1.0 / (16.0f64).sqrt();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8, 16]),
        input_node(1, &[1, 4, 8, 16]),
        input_node(2, &[1, 4, 8, 16]),
        ternary_node(
            3,
            "sdpa_causal",
            TraceOp::SdpaCausal { scale },
            0,
            1,
            2,
            &[1, 4, 8, 16],
        ),
    ]);
    let steps = compile_trace(&graph).expect("causal sdpa should compile through dispatch");
    // 3 InputForward + 1 NativeOp = 4
    assert_eq!(steps.len(), 4, "expected 4 steps, got: {}", steps.len());
    assert!(matches!(steps[0], CompiledStep::InputForward));
    assert!(matches!(steps[1], CompiledStep::InputForward));
    assert!(matches!(steps[2], CompiledStep::InputForward));
    assert!(
        matches!(
            &steps[3],
            CompiledStep::NativeOp {
                op: NativeOpKind::FlashAttention { causal: true, .. },
                ..
            }
        ),
        "expected NativeOp::FlashAttention with causal=true"
    );
}

#[test]
fn test_e2e_sdpa_causal_3d_through_dispatch() {
    // End-to-end: 3D causal SDPA → 3 InputForward + 1 Dispatch (decomposed)
    let scale = 0.5;
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4, 8]),
        input_node(1, &[2, 4, 8]),
        input_node(2, &[2, 4, 8]),
        ternary_node(
            3,
            "sdpa_causal",
            TraceOp::SdpaCausal { scale },
            0,
            1,
            2,
            &[2, 4, 8],
        ),
    ]);
    let steps = compile_trace(&graph).expect("3d causal sdpa should compile through dispatch");
    assert!(matches!(steps[0], CompiledStep::InputForward));
    assert!(matches!(steps[1], CompiledStep::InputForward));
    assert!(matches!(steps[2], CompiledStep::InputForward));
    assert!(
        matches!(steps[3], CompiledStep::Dispatch { .. }),
        "3D causal should decompose to Dispatch step"
    );
}
