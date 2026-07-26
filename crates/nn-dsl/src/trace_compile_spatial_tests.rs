// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for spatial/norm ops added in #2113: QLinear, GroupNorm,
//! ConvTranspose2d, AvgPool2d, MaxPool2d.
//!
//! Extracted from `trace_compile_tests.rs` to keep files under 1000 lines.

use nn_core::dyn_tensor::trace::{
    ComputationGraph, TraceNode, TraceOp, TraceUpsampleMode, WeightRef,
};
use nn_core::DType;

use crate::trace_compile::{compile_trace, CompiledStep};

// -- Helpers (shared with parent test module) ----------------------------------

fn graph_from_nodes(nodes: Vec<TraceNode>) -> ComputationGraph {
    ComputationGraph::from_nodes(nodes)
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

// -- QLinear / GroupNorm (#2113) -----------------------------------------------

#[test]
fn test_compile_qlinear_with_weights() {
    let weight = WeightRef::new(vec![1.0; 12], vec![3, 4]).expect("test data");
    let bias = Some(WeightRef::new(vec![0.0; 3], vec![3]).expect("test data"));
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4]),
        TraceNode::new(
            1,
            "qlinear_0".into(),
            TraceOp::QLinear { weight, bias },
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("qlinear should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } => {
            assert!(weight_data.contains_key("weight"));
            assert_eq!(kernel.name(), "linear");
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_compile_qlinear_no_bias() {
    let weight = WeightRef::new(vec![1.0; 12], vec![3, 4]).expect("test data");
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4]),
        TraceNode::new(
            1,
            "qlinear_0".into(),
            TraceOp::QLinear { weight, bias: None },
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("qlinear without bias should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_group_norm() {
    let weight = WeightRef::new(vec![1.0; 4], vec![4]).expect("test data");
    let bias = WeightRef::new(vec![0.0; 4], vec![4]).expect("test data");
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4, 8]),
        TraceNode::new(
            1,
            "group_norm_0".into(),
            TraceOp::GroupNorm {
                num_groups: 2,
                eps: 1e-5,
                weight,
                bias,
            },
            vec![0],
            vec![2, 4, 8],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("group_norm should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } => {
            assert!(
                weight_data.contains_key("weight"),
                "group_norm should have weight data"
            );
            assert!(
                weight_data.contains_key("bias"),
                "group_norm should have bias data"
            );
            // Verify decomposition structure: the IR graph must contain
            // reshape, instance_norm, reshape-back, and affine (mul+add) nodes.
            // This catches regressions in the decomposition logic that a simple
            // "compilation succeeds" test would miss.
            assert_eq!(kernel.name(), "group_norm");
            // Must have at least: input, eps, weight, bias, reshape, instance_norm,
            // reshape-back, gamma-reshape, broadcast, mul, beta-reshape, broadcast, add
            assert!(
                kernel.def().nodes.len() >= 10,
                "group_norm decomposition should produce >= 10 IR nodes, got {}",
                kernel.def().nodes.len()
            );
            // Verify reshape and instance_norm nodes exist in the graph
            use crate::tensor_ir::TensorOpKind;
            let d = kernel.def();
            let has_reshape = d
                .nodes
                .iter()
                .any(|n| matches!(n.kind, TensorOpKind::Reshape { .. }));
            let has_instance_norm = d
                .nodes
                .iter()
                .any(|n| matches!(n.kind, TensorOpKind::InstanceNorm1d { .. }));
            let has_mul = d
                .nodes
                .iter()
                .any(|n| matches!(n.kind, TensorOpKind::BinaryMul { .. }));
            let has_add = d
                .nodes
                .iter()
                .any(|n| matches!(n.kind, TensorOpKind::BinaryAdd { .. }));
            assert!(has_reshape, "group_norm must contain Reshape node");
            assert!(
                has_instance_norm,
                "group_norm must contain InstanceNorm1d node"
            );
            assert!(has_mul, "group_norm affine must contain BinaryMul node");
            assert!(has_add, "group_norm affine must contain BinaryAdd node");
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_compile_group_norm_single_group() {
    // num_groups=1 with 2D input hits the optimized add_group_norm_g1 path.
    let weight = WeightRef::new(vec![1.0; 4], vec![4]).expect("test data");
    let bias = WeightRef::new(vec![0.0; 4], vec![4]).expect("test data");
    let graph = graph_from_nodes(vec![
        input_node(0, &[4, 8]),
        TraceNode::new(
            1,
            "group_norm_0".into(),
            TraceOp::GroupNorm {
                num_groups: 1,
                eps: 1e-5,
                weight,
                bias,
            },
            vec![0],
            vec![4, 8],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("group_norm_g1 should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch { kernel, .. } => {
            assert_eq!(kernel.name(), "group_norm");
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_compile_group_norm_rejects_indivisible_channels() {
    // 5 channels not divisible by 3 groups — must produce error.
    let weight = WeightRef::new(vec![1.0; 5], vec![5]).expect("test data");
    let bias = WeightRef::new(vec![0.0; 5], vec![5]).expect("test data");
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 5, 8]),
        TraceNode::new(
            1,
            "group_norm_0".into(),
            TraceOp::GroupNorm {
                num_groups: 3,
                eps: 1e-5,
                weight,
                bias,
            },
            vec![0],
            vec![2, 5, 8],
            DType::F32,
        ),
    ]);
    let err = compile_trace(&graph).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("not divisible"),
        "error should mention indivisibility, got: {msg}"
    );
}

#[test]
fn test_compile_group_norm_rejects_1d_input() {
    // 1D input (no spatial dim) — must produce error.
    let weight = WeightRef::new(vec![1.0; 4], vec![4]).expect("test data");
    let bias = WeightRef::new(vec![0.0; 4], vec![4]).expect("test data");
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "group_norm_0".into(),
            TraceOp::GroupNorm {
                num_groups: 2,
                eps: 1e-5,
                weight,
                bias,
            },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);
    let err = compile_trace(&graph).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("1D input"),
        "error should mention 1D input, got: {msg}"
    );
}

// -- ConvTranspose2d (#2113) ---------------------------------------------------

/// ConvTranspose2d should fail at compile time with UnsupportedTraceOp
/// because MSL codegen is not yet implemented (#2274).
#[test]
fn test_compile_conv_transpose2d_unsupported() {
    // weight: [in_ch=3, out_ch=2, kH=3, kW=3], input: [1, 3, 4, 4]
    let weight = WeightRef::new(vec![1.0; 3 * 2 * 3 * 3], vec![3, 2, 3, 3]).expect("test data");
    let bias = Some(WeightRef::new(vec![0.0; 2], vec![2]).expect("test data"));
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 3, 4, 4]),
        TraceNode::new(
            1,
            "conv_transpose2d_0".into(),
            TraceOp::ConvTranspose2d {
                weight,
                bias,
                padding: [1, 1],
                output_padding: [0, 0],
                stride: [2, 2],
                dilation: [1, 1],
                groups: 1,
            },
            vec![0],
            vec![1, 2, 7, 7],
            DType::F32,
        ),
    ]);
    let err = compile_trace(&graph).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("ConvTranspose2d"),
        "error should mention ConvTranspose2d, got: {msg}"
    );
}

// -- AvgPool2d / MaxPool2d (#2113) ---------------------------------------------

#[test]
fn test_compile_avg_pool2d() {
    // Input: [1, 3, 8, 8], kernel=2, stride=2, padding=0 → output: [1, 3, 4, 4]
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 3, 8, 8]),
        TraceNode::new(
            1,
            "avg_pool2d_0".into(),
            TraceOp::AvgPool2d {
                kernel_size: [2, 2],
                stride: [2, 2],
                padding: [0, 0],
            },
            vec![0],
            vec![1, 3, 4, 4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("avg_pool2d should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } => {
            assert!(weight_data.is_empty(), "pool ops have no weights");
            assert_eq!(kernel.name(), "avg_pool2d");
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_compile_max_pool2d() {
    // Input: [1, 3, 8, 8], kernel=2, stride=2, padding=0 → output: [1, 3, 4, 4]
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 3, 8, 8]),
        TraceNode::new(
            1,
            "max_pool2d_0".into(),
            TraceOp::MaxPool2d {
                kernel_size: [2, 2],
                stride: [2, 2],
                padding: [0, 0],
            },
            vec![0],
            vec![1, 3, 4, 4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("max_pool2d should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } => {
            assert!(weight_data.is_empty(), "pool ops have no weights");
            assert_eq!(kernel.name(), "max_pool2d");
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_compile_avg_pool2d_with_padding() {
    // Input: [1, 3, 7, 7], kernel=3, stride=2, padding=1
    // out_h = (7 + 2*1 - 3) / 2 + 1 = 6/2 + 1 = 4
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 3, 7, 7]),
        TraceNode::new(
            1,
            "avg_pool2d_0".into(),
            TraceOp::AvgPool2d {
                kernel_size: [3, 3],
                stride: [2, 2],
                padding: [1, 1],
            },
            vec![0],
            vec![1, 3, 4, 4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("avg_pool2d with padding should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

// -- LSTM tests ---------------------------------------------------------------

#[test]
fn test_compile_lstm_with_bias() {
    // LSTM: input_size=4, hidden_size=3, batch=1
    // weight_ih: [4*hidden, input] = [12, 4]
    // weight_hh: [4*hidden, hidden] = [12, 3]
    // bias_ih, bias_hh: [4*hidden] = [12]
    let input_size = 4;
    let hidden_size = 3;
    let gate_size = 4 * hidden_size; // 12

    let weight_ih = WeightRef::new(
        vec![1.0; gate_size * input_size],
        vec![gate_size, input_size],
    )
    .expect("test data");
    let weight_hh = WeightRef::new(
        vec![1.0; gate_size * hidden_size],
        vec![gate_size, hidden_size],
    )
    .expect("test data");
    let bias_ih = Some(WeightRef::new(vec![0.0; gate_size], vec![gate_size]).expect("test data"));
    let bias_hh = Some(WeightRef::new(vec![0.0; gate_size], vec![gate_size]).expect("test data"));

    let graph = graph_from_nodes(vec![
        input_node(0, &[1, input_size]),  // x
        input_node(1, &[1, hidden_size]), // h
        input_node(2, &[1, hidden_size]), // c
        TraceNode::new(
            3,
            "lstm_0".into(),
            TraceOp::Lstm {
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                hidden_size,
                initial_hidden: None,
                initial_cell: None,
            },
            vec![0, 1, 2], // 3 inputs: x, h, c
            vec![1, hidden_size],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("lstm with bias should compile");
    // 3 inputs + 1 lstm = 4 steps
    assert_eq!(steps.len(), 4);
    match &steps[3] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } => {
            assert_eq!(kernel.name(), "lstm");
            assert!(
                weight_data.contains_key("weight_ih"),
                "should have weight_ih"
            );
            assert!(
                weight_data.contains_key("weight_hh"),
                "should have weight_hh"
            );
            // With both biases, compile_lstm combines them via add
            // The combined bias is computed from bias_ih + bias_hh
            assert!(
                weight_data.contains_key("bias_ih") || weight_data.contains_key("bias_hh"),
                "should have at least one bias weight"
            );
        }
        other => panic!("expected Dispatch for lstm, got {other:?}"),
    }
}

#[test]
fn test_compile_lstm_no_bias() {
    // LSTM without bias — tests the (None, None) branch
    let input_size = 4;
    let hidden_size = 3;
    let gate_size = 4 * hidden_size;

    let weight_ih = WeightRef::new(
        vec![1.0; gate_size * input_size],
        vec![gate_size, input_size],
    )
    .expect("test data");
    let weight_hh = WeightRef::new(
        vec![1.0; gate_size * hidden_size],
        vec![gate_size, hidden_size],
    )
    .expect("test data");

    let graph = graph_from_nodes(vec![
        input_node(0, &[1, input_size]),
        input_node(1, &[1, hidden_size]),
        input_node(2, &[1, hidden_size]),
        TraceNode::new(
            3,
            "lstm_0".into(),
            TraceOp::Lstm {
                weight_ih,
                weight_hh,
                bias_ih: None,
                bias_hh: None,
                hidden_size,
                initial_hidden: None,
                initial_cell: None,
            },
            vec![0, 1, 2],
            vec![1, hidden_size],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("lstm without bias should compile");
    assert_eq!(steps.len(), 4);
    match &steps[3] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } => {
            assert_eq!(kernel.name(), "lstm");
            assert!(weight_data.contains_key("weight_ih"));
            assert!(weight_data.contains_key("weight_hh"));
        }
        other => panic!("expected Dispatch for lstm, got {other:?}"),
    }
}

#[test]
fn test_compile_lstm_partial_bias_ih_only() {
    // Tests the (Some, None) branch — only bias_ih
    let input_size = 4;
    let hidden_size = 3;
    let gate_size = 4 * hidden_size;

    let weight_ih = WeightRef::new(
        vec![1.0; gate_size * input_size],
        vec![gate_size, input_size],
    )
    .expect("test data");
    let weight_hh = WeightRef::new(
        vec![1.0; gate_size * hidden_size],
        vec![gate_size, hidden_size],
    )
    .expect("test data");
    let bias_ih = Some(WeightRef::new(vec![0.5; gate_size], vec![gate_size]).expect("test data"));

    let graph = graph_from_nodes(vec![
        input_node(0, &[1, input_size]),
        input_node(1, &[1, hidden_size]),
        input_node(2, &[1, hidden_size]),
        TraceNode::new(
            3,
            "lstm_0".into(),
            TraceOp::Lstm {
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh: None,
                hidden_size,
                initial_hidden: None,
                initial_cell: None,
            },
            vec![0, 1, 2],
            vec![1, hidden_size],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("lstm with partial bias should compile");
    assert_eq!(steps.len(), 4);
    match &steps[3] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } => {
            assert_eq!(kernel.name(), "lstm");
            assert!(weight_data.contains_key("bias_ih"));
        }
        other => panic!("expected Dispatch for lstm, got {other:?}"),
    }
}

// -- Expand (broadcast) -------------------------------------------------------

#[test]
fn test_compile_expand_produces_dispatch_not_passthrough() {
    // Expand broadcasts data to a larger shape — it must produce a Dispatch step,
    // NOT a Passthrough. Expand requires actual data movement (replicating elements),
    // unlike Reshape/Squeeze/Unsqueeze which are metadata-only.
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 3]),
        TraceNode::new(
            1,
            "expand_0".into(),
            TraceOp::Expand {
                target_shape: vec![4, 3],
            },
            vec![0],
            vec![4, 3],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("expand should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch { kernel, .. } => {
            assert_eq!(kernel.name(), "expand");
        }
        CompiledStep::Passthrough { .. } => {
            panic!("Expand must produce Dispatch (GPU kernel), not Passthrough (buffer alias)");
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

// -- WhereCond (#2177) ----------------------------------------------------------

#[test]
fn test_compile_where_cond() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 3]),
        input_node(1, &[2, 3]),
        input_node(2, &[2, 3]),
        TraceNode::new(
            3,
            "where_0".into(),
            TraceOp::WhereCond,
            vec![0, 1, 2],
            vec![2, 3],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("where_cond should compile");
    assert_eq!(steps.len(), 4); // 3 inputs + 1 dispatch
    assert!(matches!(&steps[3], CompiledStep::Dispatch { .. }));
}

// -- AdaptiveAvgPool2d / PixelShuffle / PixelUnshuffle (#2177) ----------------

#[test]
fn test_compile_adaptive_avg_pool2d() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 64, 8, 8]),
        TraceNode::new(
            1,
            "aap".into(),
            TraceOp::AdaptiveAvgPool2d {
                output_size: [1, 1],
            },
            vec![0],
            vec![1, 64, 1, 1],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("adaptive_avg_pool2d");
    assert!(matches!(&steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_pixel_shuffle() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 8, 4, 4]),
        TraceNode::new(
            1,
            "ps".into(),
            TraceOp::PixelShuffle { upscale_factor: 2 },
            vec![0],
            vec![1, 2, 8, 8],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("pixel_shuffle");
    assert!(matches!(&steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_pixel_unshuffle() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 2, 8, 8]),
        TraceNode::new(
            1,
            "pus".into(),
            TraceOp::PixelUnshuffle {
                downscale_factor: 2,
            },
            vec![0],
            vec![1, 8, 4, 4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("pixel_unshuffle");
    assert!(matches!(&steps[1], CompiledStep::Dispatch { .. }));
}

// -- Flip (#2177) ---------------------------------------------------------------

#[test]
fn test_compile_flip_dim1() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4]),
        TraceNode::new(
            1,
            "flip".into(),
            TraceOp::Flip { dim: 1 },
            vec![0],
            vec![2, 4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("flip");
    assert!(matches!(&steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_flip_dim0() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[3, 5]),
        TraceNode::new(
            1,
            "flip".into(),
            TraceOp::Flip { dim: 0 },
            vec![0],
            vec![3, 5],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("flip dim0");
    assert!(matches!(&steps[1], CompiledStep::Dispatch { .. }));
}

// -- Unfold (#2177) ---------------------------------------------------------------

#[test]
fn test_compile_unfold_stft_pattern() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 1, 8]),
        TraceNode::new(
            1,
            "unfold".into(),
            TraceOp::Unfold {
                dim: 2,
                size: 4,
                step: 2,
            },
            vec![0],
            vec![1, 1, 3, 4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("unfold stft");
    assert!(matches!(&steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_unfold_dim0() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4, 3]),
        TraceNode::new(
            1,
            "unfold".into(),
            TraceOp::Unfold {
                dim: 0,
                size: 2,
                step: 1,
            },
            vec![0],
            vec![3, 3, 2],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("unfold dim0");
    assert!(matches!(&steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_unfold_last_dim() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 6]),
        TraceNode::new(
            1,
            "unfold".into(),
            TraceOp::Unfold {
                dim: 1,
                size: 3,
                step: 2,
            },
            vec![0],
            vec![2, 2, 3],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("unfold last dim");
    assert!(matches!(&steps[1], CompiledStep::Dispatch { .. }));
}

// -- Upsample2d (#2177) -----------------------------------------------------------

#[test]
fn test_compile_upsample2d_nearest() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 3, 4, 4]),
        TraceNode::new(
            1,
            "upsample2d".into(),
            TraceOp::Upsample2d {
                mode: TraceUpsampleMode::Nearest,
                scale_h: 2.0,
                scale_w: 2.0,
            },
            vec![0],
            vec![1, 3, 8, 8],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("upsample2d nearest");
    assert!(matches!(&steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_upsample2d_bilinear_unsupported() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 3, 4, 4]),
        TraceNode::new(
            1,
            "upsample2d".into(),
            TraceOp::Upsample2d {
                mode: TraceUpsampleMode::Bilinear,
                scale_h: 2.0,
                scale_w: 2.0,
            },
            vec![0],
            vec![1, 3, 8, 8],
            DType::F32,
        ),
    ]);
    let err = compile_trace(&graph).unwrap_err();
    assert!(
        format!("{err:?}").contains("upsample2d"),
        "expected upsample2d error, got: {err:?}"
    );
}

// -- Upsample1d (#2222) -----------------------------------------------------------

#[test]
fn test_compile_upsample1d_nearest() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 3, 16]),
        TraceNode::new(
            1,
            "upsample1d".into(),
            TraceOp::Upsample1d { factor: 4 },
            vec![0],
            vec![1, 3, 64],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("upsample1d nearest");
    assert!(matches!(&steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_upsample1d_zero_factor_unsupported() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 3, 16]),
        TraceNode::new(
            1,
            "upsample1d".into(),
            TraceOp::Upsample1d { factor: 0 },
            vec![0],
            vec![1, 3, 0],
            DType::F32,
        ),
    ]);
    let err = compile_trace(&graph).unwrap_err();
    assert!(
        format!("{err:?}").contains("upsample1d"),
        "expected upsample1d error, got: {err:?}"
    );
}
