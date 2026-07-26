// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structural tests for peephole pass 13: Transpose(1,2) + LayerNorm +
//! Transpose(1,2) → ChannelsFirstLayerNorm.
//!
//! These are DSL-level tests — no Metal or NY dependency.
//! Part of #3457.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::super::super::{CompiledStep, NativeOpKind};

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

/// Dummy dispatch step — the peephole checks graph nodes, not step types,
/// so the step content doesn't matter for Transpose steps.
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

/// Build a LayerNorm NativeOp step matching the pattern the peephole scans for.
fn make_layer_norm(eps: f32, input_shape: Vec<usize>, hidden_dim: usize) -> CompiledStep {
    let mut weight_data = HashMap::new();
    weight_data.insert(
        "weight".to_string(),
        WeightRef::new(vec![1.0f32; hidden_dim], vec![hidden_dim]).unwrap(),
    );
    weight_data.insert(
        "bias".to_string(),
        WeightRef::new(vec![0.0f32; hidden_dim], vec![hidden_dim]).unwrap(),
    );
    CompiledStep::NativeOp {
        op: NativeOpKind::LayerNorm {
            eps,
            input_shape,
            hidden_dim,
        },
        weight_data,
    }
}

/// A dummy ReLU node (for conv output, etc) where we don't need a specific op.
fn relu_node(id: u64, input_id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("relu_{id}"),
        TraceOp::Relu,
        vec![input_id],
        shape,
        DType::F32,
    )
}

// --- Standard pattern: Conv1d → Transpose(1,2) → LayerNorm → Transpose(1,2) ---

#[test]
fn test_peephole_absorbs_transpose_layer_norm() {
    // Simulates one iteration of Kokoro TextEncoder conv loop:
    // Conv1d output [B=1, C=512, T=32] → Transpose(1,2) → LayerNorm → Transpose(1,2)
    let bct = vec![1, 512, 32]; // [B, C, T]
    let btc = vec![1, 32, 512]; // [B, T, C] after transpose

    let mut steps = vec![
        CompiledStep::InputForward,              // 0: input [B, C, T]
        dummy_dispatch(),                        // 1: Conv1d (produces [B, C, T])
        dummy_dispatch(),                        // 2: Transpose(1,2) → [B, T, C]
        make_layer_norm(1e-5, btc.clone(), 512), // 3: LayerNorm over last dim (C=512)
        dummy_dispatch(),                        // 4: Transpose(1,2) → [B, C, T]
    ];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, bct.clone()),
        relu_node(1, 0, bct.clone()),            // Conv1d output
        transpose_node(2, 1, 1, 2, btc.clone()), // → [B, T, C]
        relu_node(3, 2, btc),                    // LayerNorm (graph uses dummy op)
        transpose_node(4, 3, 1, 2, bct.clone()), // → [B, C, T]
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // Pre-transpose absorbed.
    assert!(
        matches!(&steps[2], CompiledStep::IdentityPassthrough),
        "pre-transpose should become IdentityPassthrough, got {:?}",
        &steps[2]
    );

    // Post-transpose absorbed.
    assert!(
        matches!(&steps[4], CompiledStep::IdentityPassthrough),
        "post-transpose should become IdentityPassthrough, got {:?}",
        &steps[4]
    );

    // LayerNorm replaced with ChannelsFirstLayerNorm.
    match &steps[3] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::ChannelsFirstLayerNorm {
                    eps,
                    input_shape,
                    channels,
                    leaky_relu_slope,
                },
            weight_data,
        } => {
            assert!((*eps - 1e-5).abs() < 1e-8, "eps should be preserved");
            assert_eq!(input_shape, &bct, "shape should be channels-first [B,C,T]");
            assert_eq!(*channels, 512, "channels should match C=512");
            assert_eq!(
                *leaky_relu_slope, None,
                "plain LayerNorm peephole should not fuse LeakyReLU"
            );
            assert!(weight_data.contains_key("weight"), "weight data preserved");
            assert!(weight_data.contains_key("bias"), "bias data preserved");
        }
        other => panic!("step[3] should be ChannelsFirstLayerNorm, got {other:?}"),
    }
}

/// Three iterations: simulates the full Kokoro TextEncoder conv loop.
#[test]
fn test_peephole_absorbs_three_conv_norm_iterations() {
    let bct = vec![1, 512, 32];
    let btc = vec![1, 32, 512];

    // 3 iterations × (Conv1d + Transpose + LayerNorm + Transpose) = 12 steps + 1 input
    let mut steps = vec![
        CompiledStep::InputForward, // 0: input
        // Iteration 1
        dummy_dispatch(),                        // 1: Conv1d
        dummy_dispatch(),                        // 2: Transpose(1,2)
        make_layer_norm(1e-5, btc.clone(), 512), // 3: LayerNorm
        dummy_dispatch(),                        // 4: Transpose(1,2)
        // Iteration 2
        dummy_dispatch(),                        // 5: Conv1d
        dummy_dispatch(),                        // 6: Transpose(1,2)
        make_layer_norm(1e-5, btc.clone(), 512), // 7: LayerNorm
        dummy_dispatch(),                        // 8: Transpose(1,2)
        // Iteration 3
        dummy_dispatch(),                        // 9: Conv1d
        dummy_dispatch(),                        // 10: Transpose(1,2)
        make_layer_norm(1e-5, btc.clone(), 512), // 11: LayerNorm
        dummy_dispatch(),                        // 12: Transpose(1,2)
    ];

    let mut nodes = vec![input_node(0, bct.clone())];
    let mut prev_id = 0u64;
    for iter in 0..3u64 {
        let base = 1 + iter * 4;
        // Conv1d output
        nodes.push(relu_node(base, prev_id, bct.clone()));
        // Transpose(1,2)
        nodes.push(transpose_node(base + 1, base, 1, 2, btc.clone()));
        // LayerNorm (dummy op)
        nodes.push(relu_node(base + 2, base + 1, btc.clone()));
        // Transpose(1,2)
        nodes.push(transpose_node(base + 3, base + 2, 1, 2, bct.clone()));
        prev_id = base + 3;
    }

    let graph = ComputationGraph::from_nodes(nodes);
    super::super::apply_peephole(&mut steps, &graph);

    // Check all 3 iterations: transposes absorbed, LayerNorm → ChannelsFirstLayerNorm
    for iter in 0..3 {
        let base = 1 + iter * 4;
        assert!(
            matches!(&steps[base + 1], CompiledStep::IdentityPassthrough),
            "iter {iter}: pre-transpose at step {} should be IdentityPassthrough",
            base + 1
        );
        assert!(
            matches!(&steps[base + 3], CompiledStep::IdentityPassthrough),
            "iter {iter}: post-transpose at step {} should be IdentityPassthrough",
            base + 3
        );
        assert!(
            matches!(
                &steps[base + 2],
                CompiledStep::NativeOp {
                    op: NativeOpKind::ChannelsFirstLayerNorm { .. },
                    ..
                }
            ),
            "iter {iter}: LayerNorm at step {} should be ChannelsFirstLayerNorm",
            base + 2
        );
    }
}

/// If the pre-transpose has multiple consumers, the pattern should NOT fire.
#[test]
fn test_peephole_skips_multi_consumer_transpose() {
    let bct = vec![1, 512, 32];
    let btc = vec![1, 32, 512];

    let mut steps = vec![
        CompiledStep::InputForward,              // 0: input
        dummy_dispatch(),                        // 1: Conv1d
        dummy_dispatch(),                        // 2: Transpose(1,2), consumed by TWO nodes
        make_layer_norm(1e-5, btc.clone(), 512), // 3: LayerNorm
        dummy_dispatch(),                        // 4: Transpose(1,2)
        dummy_dispatch(),                        // 5: another consumer of step 2
    ];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, bct.clone()),
        relu_node(1, 0, bct.clone()),
        transpose_node(2, 1, 1, 2, btc.clone()),
        relu_node(3, 2, btc.clone()), // LayerNorm
        transpose_node(4, 3, 1, 2, bct),
        relu_node(5, 2, btc), // second consumer of transpose
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // Pattern should NOT fire because step 2 has fan-out > 1.
    assert!(
        !matches!(&steps[2], CompiledStep::IdentityPassthrough),
        "multi-consumer transpose should NOT be absorbed"
    );
    assert!(
        !matches!(
            &steps[3],
            CompiledStep::NativeOp {
                op: NativeOpKind::ChannelsFirstLayerNorm { .. },
                ..
            }
        ),
        "LayerNorm should NOT be converted when transpose has multiple consumers"
    );
}

/// If the transpose uses dimensions other than (1,2), the pattern should NOT fire.
#[test]
fn test_peephole_skips_non_1_2_transpose() {
    let bct = vec![1, 512, 32];
    let btc = vec![1, 32, 512];

    let mut steps = vec![
        CompiledStep::InputForward,
        dummy_dispatch(),                        // 1: Conv1d
        dummy_dispatch(),                        // 2: Transpose(0,1) — wrong dims
        make_layer_norm(1e-5, btc.clone(), 512), // 3: LayerNorm
        dummy_dispatch(),                        // 4: Transpose(1,2)
    ];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, bct.clone()),
        relu_node(1, 0, bct.clone()),
        transpose_node(2, 1, 0, 1, btc.clone()), // Transpose(0,1), not (1,2)
        relu_node(3, 2, btc),
        transpose_node(4, 3, 1, 2, bct),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // Pattern should NOT fire because pre-transpose is (0,1) not (1,2).
    assert!(
        !matches!(&steps[2], CompiledStep::IdentityPassthrough),
        "transpose(0,1) should NOT be absorbed"
    );
}

/// If hidden_dim doesn't match channels, the pattern should NOT fire.
#[test]
fn test_peephole_skips_mismatched_hidden_dim() {
    let bct = vec![1, 512, 32];
    let btc = vec![1, 32, 512];

    let mut steps = vec![
        CompiledStep::InputForward,
        dummy_dispatch(),
        dummy_dispatch(),                        // Transpose(1,2)
        make_layer_norm(1e-5, btc.clone(), 256), // hidden_dim=256 ≠ C=512
        dummy_dispatch(),                        // Transpose(1,2)
    ];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, bct.clone()),
        relu_node(1, 0, bct.clone()),
        transpose_node(2, 1, 1, 2, btc.clone()),
        relu_node(3, 2, btc),
        transpose_node(4, 3, 1, 2, bct),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // Pattern should NOT fire — hidden_dim (256) doesn't match channels (512).
    assert!(
        !matches!(
            &steps[3],
            CompiledStep::NativeOp {
                op: NativeOpKind::ChannelsFirstLayerNorm { .. },
                ..
            }
        ),
        "mismatched hidden_dim should prevent absorption"
    );
}

/// Transpose(1,2) + LayerNorm + Transpose(1,2) + LeakyRelu → ChannelsFirstLayerNorm(slope).
#[test]
fn test_peephole_absorbs_leaky_relu_after_transpose_layer_norm() {
    let bct = vec![1, 512, 32];
    let btc = vec![1, 32, 512];

    let mut steps = vec![
        CompiledStep::InputForward,              // 0: input
        dummy_dispatch(),                        // 1: Conv1d
        dummy_dispatch(),                        // 2: Transpose(1,2)
        make_layer_norm(1e-5, btc.clone(), 512), // 3: LayerNorm
        dummy_dispatch(),                        // 4: Transpose(1,2)
        dummy_dispatch(),                        // 5: LeakyRelu(0.2)
    ];

    fn leaky_relu_node(id: u64, input_id: u64, shape: Vec<usize>) -> TraceNode {
        TraceNode::new(
            id,
            format!("leaky_relu_{id}"),
            TraceOp::LeakyRelu { slope: 0.2 },
            vec![input_id],
            shape,
            DType::F32,
        )
    }

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, bct.clone()),
        relu_node(1, 0, bct.clone()),
        transpose_node(2, 1, 1, 2, btc.clone()),
        relu_node(3, 2, btc), // LayerNorm
        transpose_node(4, 3, 1, 2, bct.clone()),
        leaky_relu_node(5, 4, bct), // LeakyRelu after post-transpose
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // Pre and post transposes absorbed.
    assert!(matches!(&steps[2], CompiledStep::IdentityPassthrough));
    assert!(matches!(&steps[4], CompiledStep::IdentityPassthrough));

    // LeakyRelu absorbed into the fused kernel.
    assert!(
        matches!(&steps[5], CompiledStep::IdentityPassthrough),
        "LeakyRelu should be absorbed into ChannelsFirstLayerNorm"
    );

    // ChannelsFirstLayerNorm with fused slope.
    match &steps[3] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::ChannelsFirstLayerNorm {
                    leaky_relu_slope, ..
                },
            ..
        } => {
            assert_eq!(
                *leaky_relu_slope,
                Some(0.2),
                "slope should be fused into ChannelsFirstLayerNorm"
            );
        }
        other => panic!("step[3] should be ChannelsFirstLayerNorm, got {other:?}"),
    }
}
