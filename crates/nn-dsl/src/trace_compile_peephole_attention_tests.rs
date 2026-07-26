// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structural tests for peephole pass 9: Transpose(1,2) absorption into
//! FlashAttention SeqFirst layout.
//!
//! These are DSL-level tests — no Metal or NY dependency.
//! Part of #3088 (attention transpose elimination) and #1815 (Tier 5 D1).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use super::super::super::{AttentionLayout, CompiledStep, NativeOpKind};

fn input_node(id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape,
        DType::F32,
    )
}

fn transpose_node(
    id: u64,
    input_id: u64,
    dim0: usize,
    dim1: usize,
    shape: Vec<usize>,
) -> TraceNode {
    TraceNode::new(
        id,
        format!("transpose_{id}"),
        TraceOp::Transpose { dim0, dim1 },
        vec![input_id],
        shape,
        DType::F32,
    )
}

fn sdpa_node(id: u64, q_id: u64, k_id: u64, v_id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        "sdpa".into(),
        TraceOp::Sdpa { scale: 0.125 },
        vec![q_id, k_id, v_id],
        shape,
        DType::F32,
    )
}

/// Build a FlashAttention NativeOp step in HeadsFirst layout.
fn make_flash_attn(q_shape: &[usize], k_shape: &[usize], output_shape: &[usize]) -> CompiledStep {
    make_flash_attn_inner(q_shape, k_shape, output_shape, false)
}

/// Build a causal FlashAttention NativeOp step in HeadsFirst layout.
fn make_flash_attn_causal(
    q_shape: &[usize],
    k_shape: &[usize],
    output_shape: &[usize],
) -> CompiledStep {
    make_flash_attn_inner(q_shape, k_shape, output_shape, true)
}

fn make_flash_attn_inner(
    q_shape: &[usize],
    k_shape: &[usize],
    output_shape: &[usize],
    causal: bool,
) -> CompiledStep {
    CompiledStep::NativeOp {
        op: NativeOpKind::FlashAttention {
            scale: 0.125,
            causal,
            q_shape: q_shape.to_vec(),
            k_shape: k_shape.to_vec(),
            output_shape: output_shape.to_vec(),
            input_layout: AttentionLayout::HeadsFirst,
        },
        weight_data: HashMap::new(),
    }
}

/// Dummy Dispatch step as a placeholder for Transpose-compiled steps.
/// The peephole pass checks graph nodes (not steps) for Transpose pattern,
/// so the step type doesn't matter — it gets replaced with IdentityPassthrough.
fn dummy_dispatch() -> CompiledStep {
    use super::super::super::CompiledKernel;
    use crate::tensor_block_builder::TensorBlockBuilder;

    let mut b = TensorBlockBuilder::new("dummy");
    let a = b.add_input("x", &[4]);
    let def = b.build(a).expect("valid identity IR");
    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}

// -- Standard pattern: 3 input transposes + SDPA + 1 output transpose ---------

#[test]
fn test_peephole_absorbs_attention_transposes() {
    // B=1, S=32, H=4, D=64
    let bhsd = vec![1, 4, 32, 64]; // [B,H,S,D] post-transpose
    let bshd = vec![1, 32, 4, 64]; // [B,S,H,D] pre-transpose

    let mut steps = vec![
        CompiledStep::InputForward,           // 0: Q [B,S,H,D]
        CompiledStep::InputForward,           // 1: K [B,S,H,D]
        CompiledStep::InputForward,           // 2: V [B,S,H,D]
        dummy_dispatch(),                     // 3: Q Transpose(1,2) → [B,H,S,D]
        dummy_dispatch(),                     // 4: K Transpose(1,2) → [B,H,S,D]
        dummy_dispatch(),                     // 5: V Transpose(1,2) → [B,H,S,D]
        make_flash_attn(&bhsd, &bhsd, &bhsd), // 6: SDPA [B,H,S,D]
        dummy_dispatch(),                     // 7: Output Transpose(1,2) → [B,S,H,D]
    ];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, bshd.clone()),
        input_node(1, bshd.clone()),
        input_node(2, bshd.clone()),
        transpose_node(3, 0, 1, 2, bhsd.clone()),
        transpose_node(4, 1, 1, 2, bhsd.clone()),
        transpose_node(5, 2, 1, 2, bhsd.clone()),
        sdpa_node(6, 3, 4, 5, bhsd),
        transpose_node(7, 6, 1, 2, bshd.clone()),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // Input transposes replaced with IdentityPassthrough.
    assert!(
        matches!(&steps[3], CompiledStep::IdentityPassthrough),
        "Q transpose should be absorbed"
    );
    assert!(
        matches!(&steps[4], CompiledStep::IdentityPassthrough),
        "K transpose should be absorbed"
    );
    assert!(
        matches!(&steps[5], CompiledStep::IdentityPassthrough),
        "V transpose should be absorbed"
    );

    // Output transpose replaced with IdentityPassthrough.
    assert!(
        matches!(&steps[7], CompiledStep::IdentityPassthrough),
        "output transpose should be absorbed"
    );

    // FlashAttention switched to SeqFirst with updated shapes.
    match &steps[6] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::FlashAttention {
                    q_shape,
                    k_shape,
                    output_shape,
                    input_layout,
                    ..
                },
            ..
        } => {
            assert_eq!(
                *input_layout,
                AttentionLayout::SeqFirst,
                "layout should be SeqFirst"
            );
            assert_eq!(q_shape, &bshd, "Q shape should be [B,S,H,D]");
            assert_eq!(k_shape, &bshd, "K shape should be [B,S,H,D]");
            assert_eq!(output_shape, &bshd, "output shape should be [B,S,H,D]");
        }
        other => panic!(
            "expected FlashAttention at step 6, got: {:?}",
            std::mem::discriminant(other)
        ),
    }
}

// -- GQA pattern: different H_q and H_kv --------------------------------------

#[test]
fn test_peephole_absorbs_gqa_transposes() {
    // B=1, S=32, H_q=8, H_kv=2, D=64
    let q_bhsd = vec![1, 8, 32, 64];
    let kv_bhsd = vec![1, 2, 32, 64];
    let q_bshd = vec![1, 32, 8, 64];
    let kv_bshd = vec![1, 32, 2, 64];

    let mut steps = vec![
        CompiledStep::InputForward,                  // 0: Q [B,S,H_q,D]
        CompiledStep::InputForward,                  // 1: K [B,S,H_kv,D]
        CompiledStep::InputForward,                  // 2: V [B,S,H_kv,D]
        dummy_dispatch(),                            // 3: Q Transpose → [B,H_q,S,D]
        dummy_dispatch(),                            // 4: K Transpose → [B,H_kv,S,D]
        dummy_dispatch(),                            // 5: V Transpose → [B,H_kv,S,D]
        make_flash_attn(&q_bhsd, &kv_bhsd, &q_bhsd), // 6: SDPA
        dummy_dispatch(),                            // 7: Output Transpose → [B,S,H_q,D]
    ];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, q_bshd.clone()),
        input_node(1, kv_bshd.clone()),
        input_node(2, kv_bshd.clone()),
        transpose_node(3, 0, 1, 2, q_bhsd.clone()),
        transpose_node(4, 1, 1, 2, kv_bhsd.clone()),
        transpose_node(5, 2, 1, 2, kv_bhsd),
        sdpa_node(6, 3, 4, 5, q_bhsd),
        transpose_node(7, 6, 1, 2, q_bshd.clone()),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // All 4 transposes absorbed.
    assert!(matches!(&steps[3], CompiledStep::IdentityPassthrough));
    assert!(matches!(&steps[4], CompiledStep::IdentityPassthrough));
    assert!(matches!(&steps[5], CompiledStep::IdentityPassthrough));
    assert!(matches!(&steps[7], CompiledStep::IdentityPassthrough));

    match &steps[6] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::FlashAttention {
                    q_shape,
                    k_shape,
                    output_shape,
                    input_layout,
                    ..
                },
            ..
        } => {
            assert_eq!(*input_layout, AttentionLayout::SeqFirst);
            assert_eq!(q_shape, &q_bshd, "Q should be [B,S,H_q,D]");
            assert_eq!(k_shape, &kv_bshd, "K should be [B,S,H_kv,D]");
            assert_eq!(output_shape, &q_bshd, "output should match Q layout");
        }
        other => panic!(
            "expected FlashAttention, got: {:?}",
            std::mem::discriminant(other)
        ),
    }
}

// -- Negative: wrong transpose dims → no absorption ---------------------------

#[test]
fn test_peephole_no_absorb_wrong_transpose_dims() {
    let bhsd = vec![1, 4, 32, 64];
    let bshd = vec![1, 32, 4, 64];

    let mut steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        dummy_dispatch(), // 3: Q Transpose(0,2) — WRONG dim
        dummy_dispatch(), // 4: K Transpose(1,2)
        dummy_dispatch(), // 5: V Transpose(1,2)
        make_flash_attn(&bhsd, &bhsd, &bhsd),
        dummy_dispatch(), // 7: Output Transpose(1,2)
    ];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, bshd.clone()),
        input_node(1, bshd.clone()),
        input_node(2, bshd.clone()),
        // Q transpose uses dim0=0, dim1=2 → NOT the pattern
        transpose_node(3, 0, 0, 2, bhsd.clone()),
        transpose_node(4, 1, 1, 2, bhsd.clone()),
        transpose_node(5, 2, 1, 2, bhsd.clone()),
        sdpa_node(6, 3, 4, 5, bhsd),
        transpose_node(7, 6, 1, 2, bshd),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // Should NOT absorb — Q input is not Transpose(1,2).
    match &steps[6] {
        CompiledStep::NativeOp {
            op: NativeOpKind::FlashAttention { input_layout, .. },
            ..
        } => {
            assert_eq!(
                *input_layout,
                AttentionLayout::HeadsFirst,
                "should remain HeadsFirst"
            );
        }
        other => panic!(
            "expected FlashAttention, got: {:?}",
            std::mem::discriminant(other)
        ),
    }
    // Transpose steps should NOT be replaced.
    assert!(!matches!(&steps[3], CompiledStep::IdentityPassthrough));
}

// -- Negative: missing output transpose → no absorption -----------------------

#[test]
fn test_peephole_no_absorb_missing_output_transpose() {
    let bhsd = vec![1, 4, 32, 64];
    let bshd = vec![1, 32, 4, 64];

    let mut steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        dummy_dispatch(), // 3: Q Transpose(1,2)
        dummy_dispatch(), // 4: K Transpose(1,2)
        dummy_dispatch(), // 5: V Transpose(1,2)
        make_flash_attn(&bhsd, &bhsd, &bhsd),
        // 7: NOT a transpose — a Relu node
        dummy_dispatch(),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, bshd.clone()),
        input_node(1, bshd.clone()),
        input_node(2, bshd),
        transpose_node(3, 0, 1, 2, bhsd.clone()),
        transpose_node(4, 1, 1, 2, bhsd.clone()),
        transpose_node(5, 2, 1, 2, bhsd.clone()),
        sdpa_node(6, 3, 4, 5, bhsd.clone()),
        // Output consumer is Relu, not Transpose(1,2)
        TraceNode::new(7, "relu".into(), TraceOp::Relu, vec![6], bhsd, DType::F32),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // Should NOT absorb — no output Transpose(1,2).
    match &steps[6] {
        CompiledStep::NativeOp {
            op: NativeOpKind::FlashAttention { input_layout, .. },
            ..
        } => {
            assert_eq!(*input_layout, AttentionLayout::HeadsFirst);
        }
        other => panic!(
            "expected FlashAttention, got: {:?}",
            std::mem::discriminant(other)
        ),
    }
}

// -- Negative: SDPA output has fan-out > 1 → no absorption --------------------

#[test]
fn test_peephole_no_absorb_sdpa_fanout() {
    let bhsd = vec![1, 4, 32, 64];
    let bshd = vec![1, 32, 4, 64];

    let mut steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        dummy_dispatch(), // 3: Q Transpose(1,2)
        dummy_dispatch(), // 4: K Transpose(1,2)
        dummy_dispatch(), // 5: V Transpose(1,2)
        make_flash_attn(&bhsd, &bhsd, &bhsd),
        dummy_dispatch(), // 7: Output Transpose(1,2)
        dummy_dispatch(), // 8: Second consumer of SDPA output
    ];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, bshd.clone()),
        input_node(1, bshd.clone()),
        input_node(2, bshd.clone()),
        transpose_node(3, 0, 1, 2, bhsd.clone()),
        transpose_node(4, 1, 1, 2, bhsd.clone()),
        transpose_node(5, 2, 1, 2, bhsd.clone()),
        sdpa_node(6, 3, 4, 5, bhsd.clone()),
        transpose_node(7, 6, 1, 2, bshd),
        // Second consumer of SDPA output → fan-out > 1
        TraceNode::new(8, "other".into(), TraceOp::Relu, vec![6], bhsd, DType::F32),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // Should NOT absorb — SDPA has 2 consumers.
    match &steps[6] {
        CompiledStep::NativeOp {
            op: NativeOpKind::FlashAttention { input_layout, .. },
            ..
        } => {
            assert_eq!(*input_layout, AttentionLayout::HeadsFirst);
        }
        other => panic!(
            "expected FlashAttention, got: {:?}",
            std::mem::discriminant(other)
        ),
    }
}

// -- Reverse Transpose(2,1) is also recognized --------------------------------

#[test]
fn test_peephole_absorbs_reverse_transpose_order() {
    let bhsd = vec![1, 4, 32, 64];
    let bshd = vec![1, 32, 4, 64];

    let mut steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        dummy_dispatch(),
        dummy_dispatch(),
        dummy_dispatch(),
        make_flash_attn(&bhsd, &bhsd, &bhsd),
        dummy_dispatch(),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, bshd.clone()),
        input_node(1, bshd.clone()),
        input_node(2, bshd.clone()),
        // Transpose(2,1) instead of Transpose(1,2) — should still match.
        transpose_node(3, 0, 2, 1, bhsd.clone()),
        transpose_node(4, 1, 2, 1, bhsd.clone()),
        transpose_node(5, 2, 2, 1, bhsd.clone()),
        sdpa_node(6, 3, 4, 5, bhsd),
        transpose_node(7, 6, 2, 1, bshd),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    assert!(
        matches!(&steps[3], CompiledStep::IdentityPassthrough),
        "Transpose(2,1) should be absorbed"
    );
    assert!(matches!(&steps[7], CompiledStep::IdentityPassthrough));

    match &steps[6] {
        CompiledStep::NativeOp {
            op: NativeOpKind::FlashAttention { input_layout, .. },
            ..
        } => {
            assert_eq!(*input_layout, AttentionLayout::SeqFirst);
        }
        _ => panic!("expected FlashAttention"),
    }
}

// -- Causal FlashAttention (Qwen3/Whisper decoder) also absorbs ---------------

#[test]
fn test_peephole_absorbs_causal_attention_transposes() {
    // Causal attention (Qwen3 decoder, Whisper self-attention) uses
    // FlashAttention { causal: true }. The peephole pass must absorb
    // transposes regardless of the causal flag.
    // B=1, S=64, H=8, D=64 (Qwen3-like configuration)
    let bhsd = vec![1, 8, 64, 64]; // [B,H,S,D] post-transpose
    let bshd = vec![1, 64, 8, 64]; // [B,S,H,D] pre-transpose

    let mut steps = vec![
        CompiledStep::InputForward,                  // 0: Q [B,S,H,D]
        CompiledStep::InputForward,                  // 1: K [B,S,H,D]
        CompiledStep::InputForward,                  // 2: V [B,S,H,D]
        dummy_dispatch(),                            // 3: Q Transpose(1,2) → [B,H,S,D]
        dummy_dispatch(),                            // 4: K Transpose(1,2) → [B,H,S,D]
        dummy_dispatch(),                            // 5: V Transpose(1,2) → [B,H,S,D]
        make_flash_attn_causal(&bhsd, &bhsd, &bhsd), // 6: causal SDPA
        dummy_dispatch(),                            // 7: Output Transpose(1,2) → [B,S,H,D]
    ];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, bshd.clone()),
        input_node(1, bshd.clone()),
        input_node(2, bshd.clone()),
        transpose_node(3, 0, 1, 2, bhsd.clone()),
        transpose_node(4, 1, 1, 2, bhsd.clone()),
        transpose_node(5, 2, 1, 2, bhsd.clone()),
        sdpa_node(6, 3, 4, 5, bhsd),
        transpose_node(7, 6, 1, 2, bshd.clone()),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // All 4 transposes absorbed.
    assert!(
        matches!(&steps[3], CompiledStep::IdentityPassthrough),
        "Q transpose"
    );
    assert!(
        matches!(&steps[4], CompiledStep::IdentityPassthrough),
        "K transpose"
    );
    assert!(
        matches!(&steps[5], CompiledStep::IdentityPassthrough),
        "V transpose"
    );
    assert!(
        matches!(&steps[7], CompiledStep::IdentityPassthrough),
        "output transpose"
    );

    // FlashAttention switched to SeqFirst, causal flag preserved.
    match &steps[6] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::FlashAttention {
                    q_shape,
                    k_shape,
                    output_shape,
                    input_layout,
                    causal,
                    ..
                },
            ..
        } => {
            assert!(*causal, "causal flag must be preserved");
            assert_eq!(*input_layout, AttentionLayout::SeqFirst);
            assert_eq!(q_shape, &bshd, "Q shape [B,S,H,D]");
            assert_eq!(k_shape, &bshd, "K shape [B,S,H,D]");
            assert_eq!(output_shape, &bshd, "output shape [B,S,H,D]");
        }
        other => panic!(
            "expected FlashAttention, got: {:?}",
            std::mem::discriminant(other)
        ),
    }
}

// -- Dispatch count: 4 transposes eliminated = 4 fewer dispatches -------------

#[test]
fn test_attention_transpose_dispatch_savings() {
    // Verify the dispatch-count math: each absorbed Transpose was 1 dispatch.
    // SeqFirst FlashAttention is still 1 dispatch. Net savings = 4 per block.
    let flash = NativeOpKind::FlashAttention {
        scale: 0.125,
        causal: false,
        q_shape: vec![1, 32, 4, 64],
        k_shape: vec![1, 32, 4, 64],
        output_shape: vec![1, 32, 4, 64],
        input_layout: AttentionLayout::SeqFirst,
    };
    assert_eq!(
        flash.estimated_metal_dispatches(),
        1,
        "SeqFirst FlashAttention = 1 dispatch"
    );
}

// -- Multi-layer model: cumulative savings proof for AC2 ----------------------

#[test]
fn test_multilayer_attention_dispatch_savings() {
    // Simulate a 12-layer PlBert-like model. Each layer has:
    //   3 input Transpose(1,2) + FlashAttention + 1 output Transpose(1,2)
    // Pass 9 should absorb all 4 transposes per layer = 48 total.
    // AC2 requires >=30 dispatches reduced; this proves >=48.
    let num_layers = 12;
    let bhsd = vec![1, 4, 32, 64]; // [B,H,S,D]
    let bshd = vec![1, 32, 4, 64]; // [B,S,H,D]

    let mut steps = Vec::new();
    let mut nodes = Vec::new();
    let mut next_id: u64 = 0;

    for _ in 0..num_layers {
        let q_in = next_id;
        let k_in = next_id + 1;
        let v_in = next_id + 2;
        let q_t = next_id + 3;
        let k_t = next_id + 4;
        let v_t = next_id + 5;
        let sdpa = next_id + 6;
        let out_t = next_id + 7;

        // Steps: 3 inputs, 3 transpose dispatches, 1 flash attn, 1 output transpose
        steps.push(CompiledStep::InputForward);
        steps.push(CompiledStep::InputForward);
        steps.push(CompiledStep::InputForward);
        steps.push(dummy_dispatch());
        steps.push(dummy_dispatch());
        steps.push(dummy_dispatch());
        steps.push(make_flash_attn(&bhsd, &bhsd, &bhsd));
        steps.push(dummy_dispatch());

        // Graph nodes
        nodes.push(input_node(q_in, bshd.clone()));
        nodes.push(input_node(k_in, bshd.clone()));
        nodes.push(input_node(v_in, bshd.clone()));
        nodes.push(transpose_node(q_t, q_in, 1, 2, bhsd.clone()));
        nodes.push(transpose_node(k_t, k_in, 1, 2, bhsd.clone()));
        nodes.push(transpose_node(v_t, v_in, 1, 2, bhsd.clone()));
        nodes.push(sdpa_node(sdpa, q_t, k_t, v_t, bhsd.clone()));
        nodes.push(transpose_node(out_t, sdpa, 1, 2, bshd.clone()));

        next_id += 8;
    }

    let graph = ComputationGraph::from_nodes(nodes);
    super::super::apply_peephole(&mut steps, &graph);

    // Count absorbed transposes (now IdentityPassthrough).
    let absorbed = steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::IdentityPassthrough))
        .count();

    // 4 transposes absorbed per layer × 12 layers = 48 total.
    assert_eq!(
        absorbed,
        4 * num_layers,
        "should absorb 4 transposes per layer"
    );
    assert!(
        absorbed >= 30,
        "AC2: must reduce dispatches by >=30 (got {absorbed})"
    );

    // All FlashAttention ops should be SeqFirst.
    let seq_first_count = steps
        .iter()
        .filter(|s| {
            matches!(
                s,
                CompiledStep::NativeOp {
                    op: NativeOpKind::FlashAttention {
                        input_layout: AttentionLayout::SeqFirst,
                        ..
                    },
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        seq_first_count, num_layers,
        "all {num_layers} layers should use SeqFirst"
    );
}
