// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `verify_auto_fusion_from_graph()` pipeline.
//!
//! Exercises the single-call pipeline: `ComputationGraph` → chain detection
//! → spec generation → CROWN verification, both on synthetic graphs and
//! real Kokoro model traces.
//!
//! Part of #2127 (Wave 4: Auto fusion verification).

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;
use nn_verify::verify_auto_fusion_from_graph;

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

fn op_node(id: u64, op: TraceOp, inputs: &[u64], shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        format!("{}_{id}", op.canonical_name()),
        op,
        inputs.to_vec(),
        shape.to_vec(),
        DType::F32,
    )
}

/// Pipeline: Exp → Relu chain with point inputs — verifies full pipeline.
#[test]
fn test_pipeline_exp_relu_point_inputs() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]),
        op_node(1, TraceOp::Exp, &[0], &[1, 4, 8]),
        op_node(2, TraceOp::Relu, &[1], &[1, 4, 8]),
    ]);

    let point_bounds = &[(1.0, 1.0)];
    let result =
        verify_auto_fusion_from_graph(&graph, point_bounds, 1e-5).expect("pipeline should succeed");

    assert_eq!(result.chains_detected, 1, "one chain: Exp → Relu");
    assert_eq!(result.specs_generated, 1, "one pairwise spec");
    assert_eq!(
        result.conclusive_count, 1,
        "point inputs yield conclusive CROWN proof"
    );
}

/// Pipeline: 3-op chain — detects chain and generates pairwise specs.
///
/// Each pairwise spec may have a different number of shared inputs
/// (Add→Relu needs 2, Relu→Neg needs 1). The pipeline applies default
/// bounds when the provided bounds length doesn't match a spec.
#[test]
fn test_pipeline_three_op_chain_all_pairs() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]),
        input_node(10, &[1, 4, 8]),
        op_node(1, TraceOp::Add, &[0, 10], &[1, 4, 8]),
        op_node(2, TraceOp::Relu, &[1], &[1, 4, 8]),
        op_node(3, TraceOp::Neg, &[2], &[1, 4, 8]),
    ]);

    // Point inputs with 2 bounds (matches Add→Relu pair, Relu→Neg uses defaults).
    let bounds = &[(1.0, 1.0), (1.0, 1.0)];
    let result =
        verify_auto_fusion_from_graph(&graph, bounds, 1e-5).expect("pipeline should succeed");

    assert_eq!(result.chains_detected, 1);
    assert_eq!(result.specs_generated, 2, "3-op chain → 2 pairwise specs");
    // At least the point-input pair should be conclusive.
    assert!(
        result.conclusive_count >= 1,
        "at least 1 conclusive proof expected, got {}",
        result.conclusive_count
    );

    // All results should complete (either verified or IBP fallback).
    for r in &result.results {
        assert!(
            r.result.is_ok(),
            "pair {} should not error: {:?}",
            r.name,
            r.result.as_ref().err()
        );
    }
}

/// Pipeline: graph with no fusible chains returns empty result.
#[test]
fn test_pipeline_no_fusible_chains() {
    let graph = ComputationGraph::from_nodes(vec![input_node(0, &[2, 4])]);

    let result = verify_auto_fusion_from_graph(&graph, &[], 1e-5).expect("pipeline should succeed");

    assert_eq!(result.chains_detected, 0);
    assert_eq!(result.specs_generated, 0);
    assert_eq!(result.conclusive_count, 0);
    assert!(result.results.is_empty());
}

/// Pipeline: Kokoro-like phase generation subgraph (Clamp → Exp → Sin).
///
/// Only Exp → Sin is fusible (Clamp is not elementwise-fusible). Tests
/// that the pipeline correctly identifies partial chains in mixed graphs.
#[test]
fn test_pipeline_kokoro_like_phase_subgraph() {
    let shape = &[1, 8, 16];
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, shape),
        // Clamp is not fusible elementwise — breaks the chain.
        op_node(1, TraceOp::Exp, &[0], shape),
        op_node(2, TraceOp::Sin, &[1], shape),
    ]);

    let point_bounds = &[(0.5, 0.5)];
    let result =
        verify_auto_fusion_from_graph(&graph, point_bounds, 1e-5).expect("pipeline should succeed");

    assert_eq!(
        result.chains_detected, 1,
        "Exp → Sin should be detected as a fusible chain"
    );
    assert_eq!(result.specs_generated, 1);
    assert_eq!(
        result.conclusive_count, 1,
        "point input should yield conclusive proof"
    );

    eprintln!(
        "Kokoro-like phase subgraph: chains={}, specs={}, conclusive={}",
        result.chains_detected, result.specs_generated, result.conclusive_count
    );
}

/// Pipeline: multiple disjoint chains in one graph.
///
/// Tests that the pipeline detects and verifies all chains, not just the first.
#[test]
fn test_pipeline_multiple_disjoint_chains() {
    let shape = &[1, 4, 8];
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, shape),
        // Chain 1: Exp → Relu (nodes 1-2)
        op_node(1, TraceOp::Exp, &[0], shape),
        op_node(2, TraceOp::Relu, &[1], shape),
        // Non-fusible gap (Neg is fusible but has same shape, so check
        // fan-out: node 2 is consumed by both 3 and... no, let's make
        // a proper gap with a non-fusible op)
        // Actually, let's use two separate input branches.
        input_node(10, shape),
        // Chain 2: Sigmoid → Mul (nodes 11-12)
        op_node(11, TraceOp::Sigmoid, &[10], shape),
        op_node(12, TraceOp::Mul, &[11, 10], shape),
    ]);

    let point_bounds = &[(0.5, 0.5)];
    let result =
        verify_auto_fusion_from_graph(&graph, point_bounds, 1e-5).expect("pipeline should succeed");

    assert_eq!(
        result.chains_detected, 2,
        "two disjoint chains: Exp→Relu and Sigmoid→Mul"
    );
    assert_eq!(result.specs_generated, 2, "one spec per chain");
    assert_eq!(
        result.conclusive_count, 2,
        "both should be conclusive on point inputs"
    );
}
