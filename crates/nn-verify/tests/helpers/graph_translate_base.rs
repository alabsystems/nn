// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for KernelIR → NY GraphNetwork translation.
//!
//! Native vs decomposed comparison tests (#338): graph_translate_native.rs
//! Piecewise activation tests (ReLU, Clamp, LeakyReLU): graph_translate_piecewise.rs

use super::common;

use nn_dsl::ir::{
    ir_pretty_print, BinOpKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, Param,
    ScalarType,
};
use nn_verify::{kernel_to_graph, BoundedTensor};
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_kernel_to_graph_snake_builds() {
    let kernel = common::snake_kernel();
    let graph = kernel_to_graph(&kernel, &[1.0]).expect("build snake graph");
    // With native SnakeLayer fast path (alpha > 0), the entire kernel maps
    // to a single NY SnakeLayer node instead of 5 decomposed nodes.
    assert_eq!(
        graph.num_nodes(),
        1,
        "snake kernel with valid alpha should use native SnakeLayer (1 node)"
    );
}

#[test]
fn test_kernel_to_graph_snake_ibp() {
    let kernel = common::snake_kernel();
    let graph = kernel_to_graph(&kernel, &[1.0]).expect("build snake graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), -10.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (out_lower, out_upper) = output.lower_upper();

    // Native SnakeLayer exploits monotonicity for exact bounds:
    // snake(x, 1) = x + sin²(x), f'(x) = 1 + sin(2x) >= 0
    // so f([-10, 10]) = [f(-10), f(10)] = [-10 + sin²(-10), 10 + sin²(10)]
    // sin²(10) ≈ sin²(-10) ≈ 0.296, giving bounds ≈ [-9.70, 10.30]
    assert!(
        out_lower[[0]].is_finite(),
        "IBP lower bound should be finite, got {}",
        out_lower[[0]]
    );
    assert!(
        out_lower[[0]] >= -10.0,
        "Snake is monotone: lower bound >= input lower (-10), got {}",
        out_lower[[0]]
    );
    assert!(
        out_lower[[0]] <= -9.0,
        "lower bound should be near snake(-10) ≈ -9.7, got {}",
        out_lower[[0]]
    );
    assert!(
        out_upper[[0]].is_finite(),
        "IBP upper bound should be finite"
    );
    assert!(
        out_upper[[0]] >= 10.0,
        "IBP upper bound should be >= 10: got {}",
        out_upper[[0]]
    );
    assert!(
        out_upper[[0]] <= 12.0,
        "Native SnakeLayer should give tight upper bound near 10.3, got {}",
        out_upper[[0]]
    );
}

#[test]
fn test_kernel_to_graph_param_mismatch() {
    let kernel = common::snake_kernel();
    let err = kernel_to_graph(&kernel, &[]).expect_err("should fail with no constant params");
    assert!(
        format!("{err:?}").contains("ParamCountMismatch"),
        "expected ParamCountMismatch, got: {err:?}"
    );
}

#[test]
fn test_constant_kernel_evaluates() {
    let kernel = KernelDef::new(
        "const_kernel",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(42.0)),
        ],
        NodeId::new(1),
    );
    let graph = kernel_to_graph(&kernel, &[]).expect("build const graph");
    assert_eq!(graph.num_nodes(), 1, "constant kernel folds to 1 node");
}

#[test]
fn test_sum_reduce_mixed() {
    // sum_reduce([x, 1.0, x]) → 2x + 1
    let kernel = KernelDef::new(
        "sum_test",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::SumReduce {
                    inputs: vec![NodeId::new(0), NodeId::new(1), NodeId::new(0)],
                },
            ),
        ],
        NodeId::new(2),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build sum_reduce graph");
    assert_eq!(
        graph.num_nodes(),
        2,
        "sum_reduce(x, 1.0, x) = 2x+1 needs 2 nodes"
    );

    let lower = ArrayD::from_elem(IxDyn(&[1]), -5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (out_lower, out_upper) = output.lower_upper();

    assert!(
        out_lower[[0]] <= -9.0 + 0.01,
        "expected lower <= -9, got {}",
        out_lower[[0]]
    );
    assert!(
        out_upper[[0]] >= 11.0 - 0.01,
        "expected upper >= 11, got {}",
        out_upper[[0]]
    );
}

#[test]
fn test_sum_reduce_all_constant() {
    let kernel = KernelDef::new(
        "const_sum",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(2.0)),
            IRNode::new(NodeId::new(2), IRNodeKind::Literal(3.0)),
            IRNode::new(NodeId::new(3), IRNodeKind::Literal(5.0)),
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::SumReduce {
                    inputs: vec![NodeId::new(1), NodeId::new(2), NodeId::new(3)],
                },
            ),
        ],
        NodeId::new(4),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build const sum graph");
    assert_eq!(
        graph.num_nodes(),
        1,
        "all-constant sum_reduce folds to 1 node"
    );
}

#[test]
fn test_sum_reduce_via_lowering() {
    let kernel = common::parse_kernel(
        "fn weighted_sum(x: f32, w1: f32, w2: f32) -> f32 { sum_reduce([w1 * x, w2 * x]) }",
    );

    let graph = kernel_to_graph(&kernel, &[2.0, 3.0]).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (out_lower, out_upper) = output.lower_upper();

    assert!(
        out_lower[[0]] <= -5.0 + 0.01,
        "expected lower <= -5, got {}",
        out_lower[[0]]
    );
    assert!(
        out_upper[[0]] >= 5.0 - 0.01,
        "expected upper >= 5, got {}",
        out_upper[[0]]
    );
}

/// With the production snake kernel (which includes `alpha.max(1e-8)`),
/// alpha=0 is clamped to `SNAKE_MIN_ALPHA` before the `1/alpha` division.
/// The graph should build successfully and produce finite bounds (#325).
///
/// With constant alpha=0 (clamped to 1e-8) and x ∈ [-1, 1]:
/// - `sin(1e-8 * x)` ≈ `1e-8 * x` (small angle), so `sin²` ≈ `1e-16 * x²`
/// - `sin²/(1e-8)` ≈ `1e-8 * x²` ≈ negligible
/// - IBP bounds ≈ [-1, 1] (barely wider than input)
#[test]
fn test_snake_alpha_zero_is_clamped() {
    let kernel = common::snake_kernel();
    let graph =
        kernel_to_graph(&kernel, &[0.0]).expect("alpha=0 should succeed — max(0, 1e-8) = 1e-8");
    // Verify the graph produces finite bounds for a simple input range.
    let (lo, hi) = common::ibp_scalar(&graph, -1.0, 1.0);
    assert!(lo.is_finite(), "lower bound must be finite, got {lo}");
    assert!(hi.is_finite(), "upper bound must be finite, got {hi}");
    // Snake output >= input (sin²/alpha >= 0), so lower bound >= -1.0.
    assert!(lo >= -1.0 - 0.01, "lower bound should be >= -1.0, got {lo}");
    // Upper bound should be close to 1.0 — sin²(1e-8*x)/1e-8 is tiny
    // for x ∈ [-1, 1]. Bound width should be modest, not blown up.
    assert!(
        hi <= 10.0,
        "upper bound should be reasonable with constant alpha, got {hi}"
    );
}

#[test]
fn test_recip_zero_constant_rejects() {
    let kernel = common::parse_kernel("fn recip_mul(x: f32, a: f32) -> f32 { (1.0 / a) * x }");
    let err = kernel_to_graph(&kernel, &[0.0]).expect_err("a=0 should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite") || msg.contains("NonFinite") || msg.contains("division by zero"),
        "expected NonFinite or division-by-zero error, got: {msg}"
    );
}

#[test]
fn test_sqrt_negative_constant_rejects() {
    let kernel = common::parse_kernel("fn sqrt_mul(x: f32, a: f32) -> f32 { a.sqrt() * x }");
    let err = kernel_to_graph(&kernel, &[-1.0]).expect_err("a=-1 should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite") || msg.contains("NonFinite"),
        "expected NonFiniteConstant error, got: {msg}"
    );
}

#[test]
fn test_non_finite_constant_param_rejected() {
    let kernel = common::snake_kernel();
    let err = kernel_to_graph(&kernel, &[f32::NAN]).expect_err("NaN param should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite") || msg.contains("NonFinite"),
        "expected NonFiniteConstant error for NaN param, got: {msg}"
    );
}

#[test]
fn test_inf_constant_param_rejected() {
    let kernel = common::snake_kernel();
    let err = kernel_to_graph(&kernel, &[f32::INFINITY]).expect_err("Inf param should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite") || msg.contains("NonFinite"),
        "expected NonFiniteConstant error for Inf param, got: {msg}"
    );
}

#[test]
fn test_neg_inf_constant_param_rejected() {
    let kernel = common::snake_kernel();
    let err =
        kernel_to_graph(&kernel, &[f32::NEG_INFINITY]).expect_err("neg Inf param should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite") || msg.contains("NonFinite"),
        "expected NonFiniteConstant error for neg Inf param, got: {msg}"
    );
}

#[test]
fn test_minmax_constant_fold_rejects_non_finite() {
    let kernel = KernelDef::new(
        "minmax_nan",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("a", ScalarType::F32),
            Param::new("b", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(NodeId::new(2), IRNodeKind::Param(2)),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::MinMax {
                    op: MinMaxKind::Min,
                    lhs: NodeId::new(1),
                    rhs: NodeId::new(2),
                },
            ),
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(3),
                },
            ),
        ],
        NodeId::new(4),
    );

    let err = kernel_to_graph(&kernel, &[f32::NAN, 1.0]).expect_err("NaN in min should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite") || msg.contains("NonFinite"),
        "expected NonFiniteConstant error from MinMax NaN, got: {msg}"
    );
}

#[test]
fn test_snake_ir_pretty_print() {
    let kernel = common::snake_kernel();
    let ir = ir_pretty_print(&kernel);
    assert!(ir.contains("snake"), "IR should contain kernel name");
    assert!(ir.contains("sin"), "IR should contain sin op");
    assert!(ir.contains("powi"), "IR should contain powi op");
}
