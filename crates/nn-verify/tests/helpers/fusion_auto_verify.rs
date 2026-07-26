// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for auto-generated fusion verification.
//!
//! Tests the full pipeline: trace graph → chain detection → spec generation
//! → NY CROWN verification → status recording.
//!
//! CROWN relaxation on wide intervals ([-3, 3]) for nonlinear ops (exp,
//! sigmoid) produces bounded but loose diffs — this is expected. Point
//! inputs give exact zero-diff proofs.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;
use nn_dsl::detect_fusion_chains;
use nn_verify::fusion_auto::generate_fusion_specs;
use nn_verify::verify_fusion_equivalence;

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

/// Exp → Relu with point inputs: CROWN proves exact zero diff.
#[test]
fn test_auto_fusion_exp_relu_crown_proves_equivalence() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]),
        op_node(1, TraceOp::Exp, &[0], &[1, 4, 8]),
        op_node(2, TraceOp::Relu, &[1], &[1, 4, 8]),
    ]);

    let chains = detect_fusion_chains(&graph).expect("chain detection");
    assert_eq!(chains.len(), 1);

    let specs = generate_fusion_specs(&chains);
    assert_eq!(specs.len(), 1);

    let spec = &specs[0];
    // Point inputs — CROWN is exact on point intervals.
    let point_bounds = vec![(1.0, 1.0); spec.num_shared_inputs()];
    let result = verify_fusion_equivalence(&spec.as_fusion_spec(), &point_bounds, 1e-5)
        .expect("verification");

    assert!(
        result.is_conclusive(),
        "CROWN should succeed for simple elementwise fusion"
    );
    assert!(
        result.within_epsilon,
        "fused exp→relu on point inputs should have near-zero diff: [{}, {}]",
        result.diff_lower, result.diff_upper
    );
}

/// Exp → Relu with wide bounds: CROWN succeeds and produces finite diff.
///
/// CROWN relaxation on exp([-3, 3]) is loose, so diff is bounded but not
/// near-zero. This is expected — only point/narrow inputs give tight proofs.
#[test]
fn test_auto_fusion_exp_relu_wide_bounds_crown_succeeds() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]),
        op_node(1, TraceOp::Exp, &[0], &[1, 4, 8]),
        op_node(2, TraceOp::Relu, &[1], &[1, 4, 8]),
    ]);

    let chains = detect_fusion_chains(&graph).expect("chain detection");
    let specs = generate_fusion_specs(&chains);
    let spec = &specs[0];
    let bounds = vec![(-3.0, 3.0); spec.num_shared_inputs()];
    let result =
        verify_fusion_equivalence(&spec.as_fusion_spec(), &bounds, 100.0).expect("verification");

    assert!(
        result.is_conclusive(),
        "CROWN should succeed (not fall back to IBP)"
    );
    assert!(
        result.diff_lower.is_finite() && result.diff_upper.is_finite(),
        "diff bounds should be finite: [{}, {}]",
        result.diff_lower,
        result.diff_upper
    );
}

/// Sigmoid → Mul (SiLU-like pattern) with point inputs.
#[test]
fn test_auto_fusion_sigmoid_mul_crown_proves_equivalence() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]),
        op_node(1, TraceOp::Sigmoid, &[0], &[1, 4, 8]),
        op_node(2, TraceOp::Mul, &[1, 0], &[1, 4, 8]),
    ]);

    let chains = detect_fusion_chains(&graph).expect("chain detection");
    assert_eq!(chains.len(), 1);

    let specs = generate_fusion_specs(&chains);
    assert_eq!(specs.len(), 1);

    let spec = &specs[0];
    // Point inputs for exact verification.
    let point_bounds = vec![(0.5, 0.5); spec.num_shared_inputs()];
    let result = verify_fusion_equivalence(&spec.as_fusion_spec(), &point_bounds, 1e-5)
        .expect("verification");

    assert!(result.is_conclusive(), "CROWN should succeed for SiLU");
    assert!(
        result.within_epsilon,
        "fused sigmoid→mul on point inputs should have near-zero diff: max_abs_diff = {}",
        result.max_abs_diff
    );
}

/// 3-op chain: Add → Relu → Neg. Two pairwise verifications with point inputs.
#[test]
fn test_auto_fusion_three_op_chain_all_pairs_verified() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]),
        input_node(10, &[1, 4, 8]),
        op_node(1, TraceOp::Add, &[0, 10], &[1, 4, 8]),
        op_node(2, TraceOp::Relu, &[1], &[1, 4, 8]),
        op_node(3, TraceOp::Neg, &[2], &[1, 4, 8]),
    ]);

    let chains = detect_fusion_chains(&graph).expect("chain detection");
    assert_eq!(chains.len(), 1);
    assert_eq!(chains[0].chain_len, 3);

    let specs = generate_fusion_specs(&chains);
    assert_eq!(specs.len(), 2, "3-op chain → 2 pairwise specs");

    for (i, spec) in specs.iter().enumerate() {
        // Point inputs for exact verification.
        let point_bounds = vec![(1.0, 1.0); spec.num_shared_inputs()];
        let result = verify_fusion_equivalence(&spec.as_fusion_spec(), &point_bounds, 1e-5)
            .unwrap_or_else(|e| panic!("pair {i} verification failed: {e}"));

        assert!(
            result.is_conclusive(),
            "pair {i} ({}) should be CROWN-conclusive",
            spec.chain_name
        );
        assert!(
            result.within_epsilon,
            "pair {i} ({}) on point inputs should have near-zero diff: max_abs_diff = {}",
            spec.chain_name, result.max_abs_diff
        );
    }
}

/// No fusible chain → empty specs.
#[test]
fn test_auto_fusion_no_chains_returns_empty() {
    // Graph with only non-fusible ops (Linear is not fusible elementwise).
    let graph = ComputationGraph::from_nodes(vec![input_node(0, &[2, 4])]);

    let chains = detect_fusion_chains(&graph).expect("chain detection");
    assert!(chains.is_empty());

    let specs = generate_fusion_specs(&chains);
    assert!(specs.is_empty());
}
