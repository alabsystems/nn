// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trace compilation tests for reduce/keepdim, softmax, log_softmax, and clamp.
//!
//! Extracted from `trace_compile_tests.rs` to keep files under 500 lines.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use crate::tensor_ir::TensorOpKind;
use crate::trace_compile::{compile_trace, CompiledStep};

// -- Helpers (duplicated from trace_compile_tests.rs) -------------------------

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

// -- Reduce tests -------------------------------------------------------------

#[test]
fn test_compile_reduce_sum() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4]),
        TraceNode::new(
            1,
            "reduce_sum_0".into(),
            TraceOp::ReduceSum {
                dim: 1,
                keepdim: false,
            },
            vec![0],
            vec![2],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("reduce_sum should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_reduce_max() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4]),
        TraceNode::new(
            1,
            "reduce_max_0".into(),
            TraceOp::ReduceMax {
                dim: 1,
                keepdim: false,
            },
            vec![0],
            vec![2],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("reduce_max should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

/// Reduce Sum with `keepdim: true` — the reduced axis is retained with size 1.
/// Verifies that `keepdim` propagates through `compile_reduce` to the IR node.
#[test]
fn test_compile_reduce_sum_keepdim() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4]),
        TraceNode::new(
            1,
            "reduce_sum_kd".into(),
            TraceOp::ReduceSum {
                dim: 1,
                keepdim: true,
            },
            vec![0],
            vec![2, 1],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("reduce_sum keepdim should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch { kernel, .. } => {
            let def = kernel.def();
            let reduce_node = def
                .nodes
                .iter()
                .find(|n| matches!(n.kind, TensorOpKind::Reduce { .. }))
                .expect("IR should contain a Reduce node");
            match &reduce_node.kind {
                TensorOpKind::Reduce { keepdim, .. } => {
                    assert!(*keepdim, "keepdim should be true in IR");
                }
                _ => unreachable!(),
            }
            assert_eq!(reduce_node.shape, vec![2, 1]);
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

/// Reduce Mean with `keepdim: true` on a 3D tensor.
#[test]
fn test_compile_reduce_mean_keepdim() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[3, 4, 8]),
        TraceNode::new(
            1,
            "reduce_mean_kd".into(),
            TraceOp::ReduceMean {
                dim: 2,
                keepdim: true,
            },
            vec![0],
            vec![3, 4, 1],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("reduce_mean keepdim should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch { kernel, .. } => {
            let def = kernel.def();
            let reduce_node = def
                .nodes
                .iter()
                .find(|n| matches!(n.kind, TensorOpKind::Reduce { .. }))
                .expect("IR should contain a Reduce node");
            match &reduce_node.kind {
                TensorOpKind::Reduce { keepdim, .. } => {
                    assert!(*keepdim, "keepdim should be true in IR");
                }
                _ => unreachable!(),
            }
            assert_eq!(reduce_node.shape, vec![3, 4, 1]);
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

/// Regression #2203: keepdim=true must be preserved through compilation.
#[test]
fn test_compile_reduce_keepdim_true() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4]),
        TraceNode::new(
            1,
            "reduce_sum_0".into(),
            TraceOp::ReduceSum {
                dim: 1,
                keepdim: true,
            },
            vec![0],
            vec![2, 1],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("keepdim reduce should compile");
    assert_eq!(steps.len(), 2);
}

// -- Softmax / LogSoftmax tests -----------------------------------------------

#[test]
fn test_compile_softmax() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4]),
        TraceNode::new(
            1,
            "softmax_0".into(),
            TraceOp::Softmax { dim: 1 },
            vec![0],
            vec![2, 4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("softmax should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_log_softmax() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4]),
        TraceNode::new(
            1,
            "log_softmax_0".into(),
            TraceOp::LogSoftmax { dim: 1 },
            vec![0],
            vec![2, 4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("log_softmax should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

// -- Clamp tests --------------------------------------------------------------

#[test]
fn test_compile_clamp_both() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[8]),
        TraceNode::new(
            1,
            "clamp_0".into(),
            TraceOp::Clamp {
                min: Some(-1.0),
                max: Some(1.0),
            },
            vec![0],
            vec![8],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("clamp should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch { weight_data, .. } => {
            assert!(weight_data.contains_key("clamp_min"));
            assert!(weight_data.contains_key("clamp_max"));
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_compile_clamp_min_only() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[8]),
        TraceNode::new(
            1,
            "clamp_1".into(),
            TraceOp::Clamp {
                min: Some(0.0),
                max: None,
            },
            vec![0],
            vec![8],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("clamp_min_only should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch { weight_data, .. } => {
            assert!(weight_data.contains_key("clamp_min"));
            assert!(!weight_data.contains_key("clamp_max"));
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}
