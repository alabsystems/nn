// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Soundness tests for Select(Compare(...), then, else) → WhereLayer pipeline.
//!
//! These tests verify that the continuous approximations used for Eq and Ne
//! comparisons produce sound interval bounds when composed with WhereLayer
//! in a Select node. Also tests the Literal f64→f32 overflow guard.
//!
//! Context:
//! - Eq approximation: -(abs(lhs - rhs)), values in (-∞, 0]
//! - Ne approximation: abs(lhs - rhs), values in [0, +∞)
//! - WhereLayer uses condition ≥ 0 to include the then-branch
//!
//! Part of #40 (Compare/Select verification)
//! Part of #46 (checked_constant policy)

use nn_dsl::ir::{
    BinOpKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType,
};
use nn_verify::{kernel_to_graph, BoundedTensor};
use ndarray::{ArrayD, IxDyn};

/// Verify that Select(Eq(x, 0), then=1.0, else=0.0) produces sound bounds.
///
/// The Eq continuous approximation is -(abs(x)), producing values in (-∞, 0].
/// WhereLayer must include the then-branch (output=1.0) when the condition's
/// upper bound reaches 0 (equality is possible). If the WhereLayer excludes
/// the then-branch at condition=0, the bounds would be [0, 0] instead of
/// the correct [0, 1], making the approximation unsound.
#[test]
fn test_select_eq_where_soundness() {
    // if x == 0 { 1.0 } else { 0.0 }
    let kernel = KernelDef::new(
        "select_eq",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::Compare {
                    op: CompareOpKind::Eq,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            IRNode::new(NodeId::new(3), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(3),
                    else_val: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(4),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build Eq-Select graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -3.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 3.0f32),
    )
    .expect("bounds");
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();

    // The true function maps x=0 → 1.0 and x≠0 → 0.0.
    // Sound bounds for x in [-3, 3] must contain [0, 1].
    // Key soundness check: the upper bound must reach 1.0 (the then-branch).
    // If WhereLayer treats condition=0 as "else-only", upper would be 0.0 → UNSOUND.
    assert!(
        lo[[0]] <= 0.01,
        "Select(Eq) lower should be <= 0, got {}",
        lo[[0]]
    );
    assert!(
        hi[[0]] >= 0.99,
        "SOUNDNESS: Select(Eq) upper must be >= 1.0 to include the then-branch. \
         Got {}. This means WhereLayer excludes the equality case (condition=0), \
         making the Eq continuous approximation unsound.",
        hi[[0]]
    );
}

/// Verify that Select(Ne(x, 0), then=1.0, else=0.0) produces sound bounds.
///
/// The Ne continuous approximation is abs(x), producing values in [0, +∞).
/// At x=0, the condition is 0 (Ne is false), so else-branch should be reachable.
#[test]
fn test_select_ne_where_soundness() {
    // if x != 0 { 1.0 } else { 0.0 }
    let kernel = KernelDef::new(
        "select_ne",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::Compare {
                    op: CompareOpKind::Ne,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            IRNode::new(NodeId::new(3), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(3),
                    else_val: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(4),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build Ne-Select graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -3.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 3.0f32),
    )
    .expect("bounds");
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();

    // The true function maps x≠0 → 1.0 and x=0 → 0.0.
    // Sound bounds for x in [-3, 3] must contain [0, 1].
    assert!(
        lo[[0]] <= 0.01,
        "Select(Ne) lower should be <= 0 to include the else-branch (x=0 case), got {}",
        lo[[0]]
    );
    assert!(
        hi[[0]] >= 0.99,
        "Select(Ne) upper should be >= 1.0 to include the then-branch, got {}",
        hi[[0]]
    );
}

/// Verify that a Literal with a value exceeding f32::MAX is rejected.
///
/// IRNodeKind::Literal stores f64, but graph.rs casts to f32 for constant
/// folding. Values above f32::MAX (≈3.4e38) become f32::INFINITY, violating
/// the checked_constant policy. This test documents the expected behavior.
#[test]
fn test_literal_overflow_f32_rejected() {
    let kernel = KernelDef::new(
        "overflow_lit",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            // 3.5e38 overflows f32::MAX (≈3.4e38) → Infinity when cast to f32
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(3.5e38)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );

    let result = kernel_to_graph(&kernel, &[]);
    assert!(
        result.is_err(),
        "Literal(3.5e38) should be rejected because 3.5e38 as f32 = Inf, \
         but graph.rs:202 bypasses checked_constant for Literal nodes. \
         Got Ok instead of NonFiniteConstant error."
    );
}
