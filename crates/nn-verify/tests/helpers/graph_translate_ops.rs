// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for select, compare, and minmax graph translation ops.
//!
//! Extended tests (powi, var-var compare, ReLU via Ge): graph_translate_ops_extended.rs

use super::common;

use common::{compare_var_kernel, eval_compare_fold, ibp_scalar};
use nn_dsl::ir::{
    BinOpKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, Param, ScalarType,
};
use nn_verify::{kernel_to_graph, kernel_to_graph_multi, ParamBinding, VerifyError};

#[test]
fn test_relu_via_select_translates() {
    let kernel = KernelDef::new(
        "relu_select",
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
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(0),
                    else_val: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(3),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build relu via select");
    let (lo, hi) = ibp_scalar(&graph, -5.0, 5.0);
    assert!(
        lo >= -0.01,
        "relu via select lower should be >= 0, got {lo}"
    );
    assert!(hi >= 4.99, "relu via select upper should be >= 5, got {hi}");
}

#[test]
fn test_max_var_nonzero_const() {
    let kernel = KernelDef::new(
        "max_vc",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(3.0)),
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

    let graph = kernel_to_graph(&kernel, &[]).expect("build max(x, 3) graph");
    let (lo, hi) = ibp_scalar(&graph, 1.0, 5.0);
    assert!(
        lo >= 2.99,
        "max(x, 3) lower on [1,5] should be >= 3, got {lo}"
    );
    assert!(
        hi >= 4.99,
        "max(x, 3) upper on [1,5] should be >= 5, got {hi}"
    );
}

#[test]
fn test_min_var_const() {
    let kernel = KernelDef::new(
        "min_vc",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(2.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::MinMax {
                    op: MinMaxKind::Min,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build min(x, 2) graph");
    let (lo, hi) = ibp_scalar(&graph, -1.0, 5.0);
    assert!(
        lo <= -0.99,
        "min(x, 2) lower on [-1,5] should be <= -1, got {lo}"
    );
    assert!(
        hi <= 2.01,
        "min(x, 2) upper on [-1,5] should be <= 2, got {hi}"
    );
}

#[test]
fn test_min_var_zero() {
    let kernel = KernelDef::new(
        "min_zero",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::MinMax {
                    op: MinMaxKind::Min,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build min(x, 0) graph");
    let (lo, hi) = ibp_scalar(&graph, -5.0, 5.0);
    assert!(
        lo <= -4.99,
        "min(x, 0) lower on [-5,5] should be <= -5, got {lo}"
    );
    assert!(
        hi <= 0.01,
        "min(x, 0) upper on [-5,5] should be <= 0, got {hi}"
    );
}

/// Test max(x, 0) via MinMax IR — exercises the `c == 0.0` ReLU shortcut path
/// in `translate_minmax_var_const`. This path was untested (only max(x, 3.0)
/// was covered by `test_max_var_nonzero_const`).
#[test]
fn test_max_var_zero_relu_shortcut() {
    let kernel = KernelDef::new(
        "max_zero",
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

    let graph = kernel_to_graph(&kernel, &[]).expect("build max(x, 0) graph");

    // max(x, 0) = relu(x). On [-5, 5]: lower bound = 0, upper bound = 5.
    let (lo, hi) = ibp_scalar(&graph, -5.0, 5.0);
    assert!(
        (-0.01..=0.01).contains(&lo),
        "max(x, 0) = relu(x) lower on [-5,5] should be ≈ 0, got {lo}"
    );
    assert!(
        hi >= 4.99,
        "max(x, 0) = relu(x) upper on [-5,5] should be >= 5, got {hi}"
    );

    // All-negative input: relu on [-10, -1] should be [0, 0].
    let (lo2, hi2) = ibp_scalar(&graph, -10.0, -1.0);
    assert!(
        (-0.01..=0.01).contains(&lo2),
        "max(x, 0) lower on [-10,-1] should be ≈ 0, got {lo2}"
    );
    assert!(
        (-0.01..=0.01).contains(&hi2),
        "max(x, 0) upper on [-10,-1] should be ≈ 0, got {hi2}"
    );

    // All-positive input: relu on [2, 8] should be [2, 8].
    let (lo3, hi3) = ibp_scalar(&graph, 2.0, 8.0);
    assert!(
        lo3 >= 1.99,
        "max(x, 0) lower on [2,8] should be >= 2, got {lo3}"
    );
    assert!(
        hi3 >= 7.99,
        "max(x, 0) upper on [2,8] should be >= 8, got {hi3}"
    );
}

#[test]
fn test_select_non_activation_pattern_falls_back_to_where_layer() {
    let kernel = KernelDef::new(
        "select_where_fallback",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::Compare {
                    op: CompareOpKind::Lt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            IRNode::new(NodeId::new(3), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(3),
                },
            ),
            IRNode::new(
                NodeId::new(5),
                IRNodeKind::BinOp {
                    op: BinOpKind::Sub,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(3),
                },
            ),
            IRNode::new(
                NodeId::new(6),
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(4),
                    else_val: NodeId::new(5),
                },
            ),
        ],
        NodeId::new(6),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build select where-fallback graph");
    let (lo, hi) = ibp_scalar(&graph, -2.0, 2.0);
    assert!(
        lo.is_finite() && hi.is_finite(),
        "where-fallback bounds should be finite, got [{lo}, {hi}]"
    );
}

#[test]
fn test_clamp_variable_bounds_rejected_as_unsupported_op() {
    let kernel = KernelDef::new(
        "clamp_var_bounds",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("lo", ScalarType::F32),
            Param::new("hi", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(NodeId::new(2), IRNodeKind::Param(2)),
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

    let bindings = vec![
        ParamBinding::Variable,
        ParamBinding::Variable,
        ParamBinding::Constant(1.0),
    ];
    let err =
        kernel_to_graph_multi(&kernel, &bindings).expect_err("variable clamp bound should fail");
    assert!(
        matches!(err, VerifyError::UnsupportedOp(ref msg) if msg.contains("Clamp with variable bounds")),
        "expected UnsupportedOp for variable clamp bounds, got: {err:?}"
    );
}

// --- Compare Eq/Ne tests (#40 Finding A) ---

#[test]
fn test_compare_eq_constant_fold() {
    let val = eval_compare_fold(CompareOpKind::Eq, 3.0, 3.0);
    assert!(
        (val - 1.0).abs() < 0.01,
        "Eq(3,3) should fold to 1.0, got {val}"
    );
    let val = eval_compare_fold(CompareOpKind::Eq, 3.0, 5.0);
    assert!(val.abs() < 0.01, "Eq(3,5) should fold to 0.0, got {val}");
}

#[test]
fn test_compare_ne_constant_fold() {
    let val = eval_compare_fold(CompareOpKind::Ne, 3.0, 5.0);
    assert!(
        (val - 1.0).abs() < 0.01,
        "Ne(3,5) should fold to 1.0, got {val}"
    );
    let val = eval_compare_fold(CompareOpKind::Ne, 3.0, 3.0);
    assert!(val.abs() < 0.01, "Ne(3,3) should fold to 0.0, got {val}");
}

#[test]
fn test_compare_ne_variable_translates() {
    let kernel = compare_var_kernel(CompareOpKind::Ne);
    let graph = kernel_to_graph(&kernel, &[]).expect("build Ne(x,0) graph");
    let (lo, hi) = ibp_scalar(&graph, -3.0, 5.0);
    // Select wraps Compare Bool→F32: bounds in [0, 1]
    assert!(lo >= -0.01, "Ne lower >= 0, got {lo}");
    assert!(hi <= 1.01, "Ne upper <= 1, got {hi}");
    assert!(lo.is_finite() && hi.is_finite(), "Ne bounds finite");
}

#[test]
fn test_compare_eq_variable_translates() {
    let kernel = compare_var_kernel(CompareOpKind::Eq);
    let graph = kernel_to_graph(&kernel, &[]).expect("build Eq(x,0) graph");
    let (lo, hi) = ibp_scalar(&graph, -3.0, 5.0);
    // Select wraps Compare Bool→F32: bounds in [0, 1]
    assert!(lo >= -0.01, "Eq lower >= 0, got {lo}");
    assert!(hi <= 1.01, "Eq upper <= 1, got {hi}");
    assert!(lo.is_finite() && hi.is_finite(), "Eq bounds finite");
}
