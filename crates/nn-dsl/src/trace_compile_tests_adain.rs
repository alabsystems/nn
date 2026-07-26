// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for AdaIN NativeOp compilation paths.
//!
//! Verifies that `compile_adain_snake` and `compile_adain_leaky_relu` emit
//! `CompiledStep::NativeOp` for rank >= 3 inputs and fall back to IR
//! decomposition for rank < 3.
//!
//! Part of #2472 (fused InstanceNorm MSL kernel).

use nn_core::dyn_tensor::trace::{ComputationGraph, KokoroFusedOp, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use crate::trace_compile::{compile_trace, CompiledStep, NativeOpKind};

// -- Helpers ------------------------------------------------------------------

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

// -- AdainSnake NativeOp tests ------------------------------------------------

#[test]
fn test_compile_adain_snake_native_op_rank3() {
    let alpha = WeightRef::new(vec![0.5; 4], vec![4]).expect("test alpha");
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 4, 16]), // x: [B, C, T]
        input_node(1, &[1, 4, 1]),  // gamma: [B, C, 1]
        input_node(2, &[1, 4, 1]),  // beta: [B, C, 1]
        TraceNode::new(
            3,
            "adain_snake_0".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainSnake { alpha, eps: 1e-5 }),
            vec![0, 1, 2],
            vec![1, 4, 16],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("adain_snake should compile");
    // Steps: 3 inputs (passthrough) + 1 NativeOp
    assert_eq!(steps.len(), 4);
    assert!(
        matches!(&steps[3], CompiledStep::NativeOp { .. }),
        "step 3 should be NativeOp, got {:?}",
        std::mem::discriminant(&steps[3])
    );
    let CompiledStep::NativeOp { op, weight_data } = &steps[3] else {
        unreachable!("already asserted NativeOp");
    };
    assert!(
        matches!(op, NativeOpKind::AdainSnake { .. }),
        "expected AdainSnake variant"
    );
    let NativeOpKind::AdainSnake {
        eps,
        input_shape,
        channels,
        ..
    } = op
    else {
        unreachable!("already asserted AdainSnake");
    };
    assert!((*eps - 1e-5_f32).abs() < 1e-8);
    assert_eq!(input_shape, &[1, 4, 16]);
    assert_eq!(*channels, 4);
    assert!(weight_data.contains_key("alpha"));
}

#[test]
fn test_compile_adain_snake_fallback_rank2() {
    let alpha = WeightRef::new(vec![0.5; 4], vec![4]).expect("test alpha");
    let graph = graph_from_nodes(vec![
        input_node(0, &[4, 16]), // x: rank 2, no batch dim
        input_node(1, &[4, 1]),  // gamma
        input_node(2, &[4, 1]),  // beta
        TraceNode::new(
            3,
            "adain_snake_rank2".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainSnake { alpha, eps: 1e-5 }),
            vec![0, 1, 2],
            vec![4, 16],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("adain_snake rank2 should compile");
    // Rank < 3 falls back to IR decomposition → Dispatch, not NativeOp.
    let has_native = steps
        .iter()
        .any(|s| matches!(s, CompiledStep::NativeOp { .. }));
    assert!(!has_native, "rank-2 AdainSnake should not use NativeOp");
}

// -- AdainLeakyRelu NativeOp tests --------------------------------------------

#[test]
fn test_compile_adain_leaky_relu_native_op_rank3() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 8, 32]), // x: [B, C, T]
        input_node(1, &[1, 8, 1]),  // gamma: [B, C, 1]
        input_node(2, &[1, 8, 1]),  // beta: [B, C, 1]
        TraceNode::new(
            3,
            "adain_leaky_relu_0".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainLeakyRelu {
                eps: 1e-5,
                slope: 0.2,
            }),
            vec![0, 1, 2],
            vec![1, 8, 32],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("adain_leaky_relu should compile");
    // Steps: 3 inputs (passthrough) + 1 NativeOp
    assert_eq!(steps.len(), 4);
    assert!(
        matches!(&steps[3], CompiledStep::NativeOp { .. }),
        "step 3 should be NativeOp, got {:?}",
        std::mem::discriminant(&steps[3])
    );
    let CompiledStep::NativeOp { op, weight_data } = &steps[3] else {
        unreachable!("already asserted NativeOp");
    };
    assert!(
        matches!(op, NativeOpKind::AdainLeakyRelu { .. }),
        "expected AdainLeakyRelu variant"
    );
    let NativeOpKind::AdainLeakyRelu {
        eps,
        slope,
        input_shape,
        ..
    } = op
    else {
        unreachable!("already asserted AdainLeakyRelu");
    };
    assert!((*eps - 1e-5_f32).abs() < 1e-8);
    assert!((*slope - 0.2_f32).abs() < 1e-8);
    assert_eq!(input_shape, &[1, 8, 32]);
    // AdainLeakyRelu has no weight data.
    assert!(weight_data.is_empty());
}

#[test]
fn test_compile_adain_leaky_relu_fallback_rank2() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[8, 32]), // x: rank 2
        input_node(1, &[8, 1]),  // gamma
        input_node(2, &[8, 1]),  // beta
        TraceNode::new(
            3,
            "adain_leaky_relu_rank2".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainLeakyRelu {
                eps: 1e-5,
                slope: 0.2,
            }),
            vec![0, 1, 2],
            vec![8, 32],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("adain_leaky_relu rank2 should compile");
    let has_native = steps
        .iter()
        .any(|s| matches!(s, CompiledStep::NativeOp { .. }));
    assert!(!has_native, "rank-2 AdainLeakyRelu should not use NativeOp");
}
