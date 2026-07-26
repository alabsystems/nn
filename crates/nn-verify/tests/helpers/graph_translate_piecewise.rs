// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Piecewise activation + compare/unary catch-all coverage tests.
//!
//! - Piecewise activations: ReLU (max), Clamp, LeakyReLU (select) (#40)
//! - Compare constant fold and variable coverage (#56)
//! - Unary function variable and constant-fold coverage (#56)
//!
//! BinOp, MinMax, and Select tests are in `graph_translate_catchall.rs`.
//! Core translation tests are in `graph_translate.rs`.

use super::common;

use common::{compare_var_kernel, eval_compare_fold, ibp_scalar, unary_fn_kernel};
use nn_dsl::ir::{
    BinOpKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, Param, ScalarType,
    UnaryFnKind,
};
use nn_verify::{kernel_to_graph, BoundedTensor};
use ndarray::{ArrayD, IxDyn};

// ── Piecewise activations: ReLU, Clamp, LeakyReLU ──────────────────────────

#[test]
fn test_relu_max_zero_translates() {
    let kernel = KernelDef::new(
        "relu",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::MinMax {
                    op: MinMaxKind::Max,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build relu graph");
    let lower = ArrayD::from_elem(IxDyn(&[1]), -5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (out_lower, out_upper) = output.lower_upper();

    assert!(
        out_lower[[0]] >= -0.01,
        "relu lower should be >= 0, got {}",
        out_lower[[0]]
    );
    assert!(
        out_upper[[0]] >= 4.99,
        "relu upper should be >= 5, got {}",
        out_upper[[0]]
    );
}

#[test]
fn test_relu_max_zero_negative_input() {
    let kernel = KernelDef::new(
        "relu_neg",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::MinMax {
                    op: MinMaxKind::Max,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build relu graph");
    let lower = ArrayD::from_elem(IxDyn(&[1]), -10.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), -1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (out_lower, out_upper) = output.lower_upper();

    assert!(
        out_lower[[0]] >= -0.01 && out_upper[[0]] <= 0.01,
        "relu on negative input should be [0, 0], got [{}, {}]",
        out_lower[[0]],
        out_upper[[0]]
    );
}

#[test]
fn test_clamp_constant_bounds_translates() {
    let kernel = KernelDef::new(
        "clamp_test",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(-1.0)),
            IRNode::new(NodeId::new(2), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::Clamp {
                    input: NodeId::new(0),
                    min: NodeId::new(1),
                    max: NodeId::new(2),
                },
            ),
        ],
        NodeId::new(3),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build clamp graph");
    let lower = ArrayD::from_elem(IxDyn(&[1]), -5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (out_lower, out_upper) = output.lower_upper();

    assert!(
        out_lower[[0]] >= -1.01,
        "clamp lower should be >= -1, got {}",
        out_lower[[0]]
    );
    assert!(
        out_upper[[0]] <= 1.01,
        "clamp upper should be <= 1, got {}",
        out_upper[[0]]
    );
}

#[test]
fn test_clamp_constant_fold() {
    let kernel = KernelDef::new(
        "clamp_const",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(5.0)),
            IRNode::new(NodeId::new(2), IRNodeKind::Literal(-1.0)),
            IRNode::new(NodeId::new(3), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::Clamp {
                    input: NodeId::new(1),
                    min: NodeId::new(2),
                    max: NodeId::new(3),
                },
            ),
        ],
        NodeId::new(4),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build clamp const graph");
    assert_eq!(
        graph.num_nodes(),
        1,
        "clamp(constant, min, max) folds to 1 node"
    );
}

#[test]
fn test_leaky_relu_select_translates() {
    let kernel = KernelDef::new(
        "leaky_relu",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            IRNode::new(NodeId::new(3), IRNodeKind::Literal(0.01)),
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(3),
                    rhs: NodeId::new(0),
                },
            ),
            IRNode::new(
                NodeId::new(5),
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(0),
                    else_val: NodeId::new(4),
                },
            ),
        ],
        NodeId::new(5),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build leaky_relu graph");
    let lower = ArrayD::from_elem(IxDyn(&[1]), -10.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (out_lower, out_upper) = output.lower_upper();

    assert!(
        out_lower[[0]] <= -0.09,
        "leaky_relu lower for [-10,10] should be <= -0.1, got {}",
        out_lower[[0]]
    );
    assert!(
        out_upper[[0]] >= 9.99,
        "leaky_relu upper for [-10,10] should be >= 10, got {}",
        out_upper[[0]]
    );
}

// ── Compare constant fold ───────────────────────────────────────────────────

#[test]
fn test_compare_gt_ge_lt_le_constant_fold() {
    let cases: &[(CompareOpKind, f32, f32, f32)] = &[
        (CompareOpKind::Gt, 5.0, 3.0, 1.0),
        (CompareOpKind::Gt, 3.0, 5.0, 0.0),
        (CompareOpKind::Ge, 3.0, 3.0, 1.0),
        (CompareOpKind::Ge, 2.0, 3.0, 0.0),
        (CompareOpKind::Lt, 2.0, 5.0, 1.0),
        (CompareOpKind::Lt, 5.0, 2.0, 0.0),
        (CompareOpKind::Le, 3.0, 3.0, 1.0),
        (CompareOpKind::Le, 5.0, 3.0, 0.0),
    ];

    for &(op, a, b, expected) in cases {
        let val = eval_compare_fold(op, a, b);
        assert!(
            (val - expected).abs() < 0.01,
            "{op:?}({a},{b}) should fold to {expected}, got {val}"
        );
    }
}

// ── Compare variable ────────────────────────────────────────────────────────

/// Assert that a compare-variable kernel builds and produces finite bounds.
fn assert_compare_var_translates(op: CompareOpKind) {
    let kernel = compare_var_kernel(op);
    let graph =
        kernel_to_graph(&kernel, &[]).unwrap_or_else(|e| panic!("build {op:?}(x,0) graph: {e}"));
    let (lo, hi) = ibp_scalar(&graph, -3.0, 5.0);
    assert!(
        lo.is_finite() && hi.is_finite(),
        "{op:?}(x,0) bounds should be finite, got [{lo}, {hi}]",
    );
}

#[test]
fn test_compare_gt_variable_translates() {
    assert_compare_var_translates(CompareOpKind::Gt);
}

#[test]
fn test_compare_lt_variable_translates() {
    assert_compare_var_translates(CompareOpKind::Lt);
}

#[test]
fn test_compare_ge_variable_translates() {
    assert_compare_var_translates(CompareOpKind::Ge);
}

#[test]
fn test_compare_le_variable_translates() {
    assert_compare_var_translates(CompareOpKind::Le);
}

// ── Unary variable ──────────────────────────────────────────────────────────

#[test]
fn test_unary_all_variants_variable() {
    for op in [
        UnaryFnKind::Sin,
        UnaryFnKind::Cos,
        UnaryFnKind::Exp,
        UnaryFnKind::Abs,
        UnaryFnKind::Recip,
    ] {
        let kernel = unary_fn_kernel(op);
        let graph =
            kernel_to_graph(&kernel, &[]).unwrap_or_else(|e| panic!("build {op:?}(x) graph: {e}"));
        // Use [1, 3] to keep all functions in valid domain (positive for Recip)
        let (lo, hi) = ibp_scalar(&graph, 1.0, 3.0);
        assert!(
            lo.is_finite() && hi.is_finite(),
            "{op:?}(x) bounds should be finite, got [{lo}, {hi}]",
        );
    }
}

#[test]
fn test_unary_sqrt_variable() {
    let kernel = unary_fn_kernel(UnaryFnKind::Sqrt);
    let graph = kernel_to_graph(&kernel, &[]).expect("build sqrt(x) graph");
    let (lo, hi) = ibp_scalar(&graph, 1.0, 9.0);
    assert!(
        (lo - 1.0).abs() < 0.01,
        "sqrt lower should be ~1.0, got {lo}"
    );
    assert!(
        (hi - 3.0).abs() < 0.01,
        "sqrt upper should be ~3.0, got {hi}"
    );
}

#[test]
fn test_unary_rsqrt_variable() {
    let kernel = unary_fn_kernel(UnaryFnKind::Rsqrt);
    let graph = kernel_to_graph(&kernel, &[]).expect("build rsqrt(x) graph");
    let (lo, hi) = ibp_scalar(&graph, 1.0, 4.0);
    assert!(
        (lo - 0.5).abs() < 0.01,
        "rsqrt lower should be ~0.5, got {lo}"
    );
    assert!(
        (hi - 1.0).abs() < 0.01,
        "rsqrt upper should be ~1.0, got {hi}"
    );
}

// ── Unary constant fold ─────────────────────────────────────────────────────

/// Build a kernel with constant unary: `fn f(x, a) -> f32 { a.<op>() + x }`
fn unary_const_kernel(op: UnaryFnKind) -> KernelDef {
    KernelDef::new(
        format!("unary_const_{op:?}"),
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("a", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::UnaryFn {
                    op,
                    input: NodeId::new(1),
                },
            ),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(2),
                },
            ),
        ],
        NodeId::new(3),
    )
}

#[test]
fn test_unary_all_variants_constant_fold() {
    let cases: &[(UnaryFnKind, f32, f32)] = &[
        (UnaryFnKind::Sin, 0.0, 0.0),
        (UnaryFnKind::Cos, 0.0, 1.0),
        (UnaryFnKind::Sqrt, 4.0, 2.0),
        (UnaryFnKind::Rsqrt, 4.0, 0.5),
        (UnaryFnKind::Exp, 0.0, 1.0),
        (UnaryFnKind::Abs, -3.0, 3.0),
        (UnaryFnKind::Recip, 2.0, 0.5),
    ];

    for &(op, a, expected) in cases {
        let kernel = unary_const_kernel(op);
        let graph = kernel_to_graph(&kernel, &[a])
            .unwrap_or_else(|e| panic!("build {op:?}({a}) graph: {e}"));
        let (lo, _hi) = ibp_scalar(&graph, 0.0, 0.0);
        assert!(
            (lo - expected).abs() < 0.01,
            "{op:?}({a}) should fold to {expected}, got {lo}"
        );
    }
}
