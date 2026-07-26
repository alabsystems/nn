// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! BinOp, MinMax, and Select catch-all coverage tests for graph translation (#56).
//!
//! Tests that `BinOpKind`, `MinMaxKind`, and `Select` translate correctly
//! through the graph builder, both for variable and constant-fold paths.
//!
//! Compare and Unary tests are in `graph_translate_catchall_unary.rs`.

use super::common;

use common::{binop_const_const_kernel, binop_var_var_kernel, ibp_scalar};
use nn_dsl::ir::{
    BinOpKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, Param, ScalarType,
};
use nn_verify::{kernel_to_graph, kernel_to_graph_multi, BoundedTensor, ParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// BinOp variable-variable tests
// ---------------------------------------------------------------------------

#[test]
fn test_binop_add_var_var_translates() {
    let kernel = binop_var_var_kernel(BinOpKind::Add);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let graph = kernel_to_graph_multi(&kernel, &bindings).expect("build Add(x,y) graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[2]), 1.0f32),
    )
    .expect("bounds");
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();
    assert!(
        lo[[0]] <= -2.0 + 0.01,
        "Add lower should be <= -2, got {}",
        lo[[0]]
    );
    assert!(
        hi[[0]] >= 2.0 - 0.01,
        "Add upper should be >= 2, got {}",
        hi[[0]]
    );
}

#[test]
fn test_binop_sub_var_var_translates() {
    let kernel = binop_var_var_kernel(BinOpKind::Sub);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let graph = kernel_to_graph_multi(&kernel, &bindings).expect("build Sub(x,y) graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[2]), 1.0f32),
    )
    .expect("bounds");
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();
    assert!(
        lo[[0]] <= -2.0 + 0.01,
        "Sub lower should be <= -2, got {}",
        lo[[0]]
    );
    assert!(
        hi[[0]] >= 2.0 - 0.01,
        "Sub upper should be >= 2, got {}",
        hi[[0]]
    );
}

#[test]
fn test_binop_mul_var_var_translates() {
    let kernel = binop_var_var_kernel(BinOpKind::Mul);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let graph = kernel_to_graph_multi(&kernel, &bindings).expect("build Mul(x,y) graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -2.0f32),
        ArrayD::from_elem(IxDyn(&[2]), 2.0f32),
    )
    .expect("bounds");
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();
    assert!(
        lo[[0]] <= -4.0 + 0.01,
        "Mul lower should be <= -4, got {}",
        lo[[0]]
    );
    assert!(
        hi[[0]] >= 4.0 - 0.01,
        "Mul upper should be >= 4, got {}",
        hi[[0]]
    );
}

#[test]
fn test_binop_div_var_var_translates() {
    let kernel = binop_var_var_kernel(BinOpKind::Div);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let graph = kernel_to_graph_multi(&kernel, &bindings).expect("build Div(x,y) graph");
    // Use positive ranges to avoid division by zero
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[2]), 4.0f32),
    )
    .expect("bounds");
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();
    assert!(
        lo[[0]].is_finite(),
        "Div lower should be finite, got {}",
        lo[[0]]
    );
    assert!(
        hi[[0]].is_finite(),
        "Div upper should be finite, got {}",
        hi[[0]]
    );
}

// ---------------------------------------------------------------------------
// BinOp constant-constant fold tests
// ---------------------------------------------------------------------------

#[test]
fn test_binop_all_variants_const_const_fold() {
    let cases: &[(BinOpKind, f32, f32, f32)] = &[
        (BinOpKind::Add, 3.0, 2.0, 5.0),
        (BinOpKind::Sub, 5.0, 3.0, 2.0),
        (BinOpKind::Mul, 3.0, 4.0, 12.0),
        (BinOpKind::Div, 10.0, 2.0, 5.0),
    ];

    for &(op, a, b, expected) in cases {
        let kernel = binop_const_const_kernel(op);
        let graph = kernel_to_graph(&kernel, &[a, b])
            .unwrap_or_else(|e| panic!("build {op:?}({a},{b}) graph: {e}"));
        let (lo, _hi) = ibp_scalar(&graph, 0.0, 0.0);
        assert!(
            (lo - expected).abs() < 0.01,
            "{op:?}({a},{b}) should fold to {expected}, got {lo}"
        );
    }
}

// ---------------------------------------------------------------------------
// MinMax tests
// ---------------------------------------------------------------------------

#[test]
fn test_minmax_const_const_fold() {
    let cases: &[(MinMaxKind, f32, f32, f32)] = &[
        (MinMaxKind::Min, 3.0, 7.0, 3.0),
        (MinMaxKind::Min, 7.0, 3.0, 3.0),
        (MinMaxKind::Max, 3.0, 7.0, 7.0),
        (MinMaxKind::Max, 7.0, 3.0, 7.0),
    ];

    for &(op, a, b, expected) in cases {
        let kernel = KernelDef::new(
            format!("minmax_cc_{op:?}"),
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
                        op,
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
        let graph = kernel_to_graph(&kernel, &[a, b])
            .unwrap_or_else(|e| panic!("build {op:?}({a},{b}) graph: {e}"));
        let (lo, _hi) = ibp_scalar(&graph, 0.0, 0.0);
        assert!(
            (lo - expected).abs() < 0.01,
            "{op:?}({a},{b}) should fold to {expected}, got {lo}"
        );
    }
}

#[test]
fn test_minmax_var_var_translates() {
    for op in [MinMaxKind::Max, MinMaxKind::Min] {
        let kernel = KernelDef::new(
            format!("minmax_vv_{op:?}"),
            vec![
                Param::new("x", ScalarType::F32),
                Param::new("y", ScalarType::F32),
            ],
            ScalarType::F32,
            vec![
                IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
                IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
                IRNode::new(
                    NodeId::new(2),
                    IRNodeKind::MinMax {
                        op,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(2),
        );
        let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
        let graph = kernel_to_graph_multi(&kernel, &bindings)
            .unwrap_or_else(|e| panic!("build {op:?}(x,y) graph: {e}"));
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[2]), -3.0f32),
            ArrayD::from_elem(IxDyn(&[2]), 3.0f32),
        )
        .expect("bounds");
        let output = graph
            .propagate_ibp(&input)
            .unwrap_or_else(|e| panic!("{op:?} IBP: {e}"));
        let (lo, hi) = output.lower_upper();
        assert!(
            lo[[0]].is_finite() && hi[[0]].is_finite(),
            "{op:?}(x,y) bounds should be finite, got [{}, {}]",
            lo[[0]],
            hi[[0]]
        );
    }
}

// ---------------------------------------------------------------------------
// Select constant condition tests
// ---------------------------------------------------------------------------

/// Build a kernel with a constant boolean condition Select.
/// `cond_true`: if true, Compare(1.0 > 0.0) = true; if false, Compare(0.0 > 1.0) = false.
/// select(cond, 42.0, 99.0) → 42.0 when true, 99.0 when false.
fn select_const_cond_kernel(cond_true: bool) -> KernelDef {
    let (lit_a, lit_b) = if cond_true { (1.0, 0.0) } else { (0.0, 1.0) };
    KernelDef::new(
        format!("select_{cond_true}"),
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(lit_a)),
            IRNode::new(NodeId::new(2), IRNodeKind::Literal(lit_b)),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(1),
                    rhs: NodeId::new(2),
                },
            ),
            IRNode::new(NodeId::new(4), IRNodeKind::Literal(42.0)),
            IRNode::new(NodeId::new(5), IRNodeKind::Literal(99.0)),
            IRNode::new(
                NodeId::new(6),
                IRNodeKind::Select {
                    cond: NodeId::new(3),
                    then_val: NodeId::new(4),
                    else_val: NodeId::new(5),
                },
            ),
        ],
        NodeId::new(6),
    )
}

#[test]
fn test_select_constant_condition_true() {
    let kernel = select_const_cond_kernel(true);
    let graph = kernel_to_graph(&kernel, &[]).expect("build select(true) graph");
    let (lo, hi) = ibp_scalar(&graph, 0.0, 0.0);
    assert!(
        (lo - 42.0).abs() < 0.01,
        "select(true) lo should be 42, got {lo}"
    );
    assert!(
        (hi - 42.0).abs() < 0.01,
        "select(true) hi should be 42, got {hi}"
    );
}

#[test]
fn test_select_constant_condition_false() {
    let kernel = select_const_cond_kernel(false);
    let graph = kernel_to_graph(&kernel, &[]).expect("build select(false) graph");
    let (lo, hi) = ibp_scalar(&graph, 0.0, 0.0);
    assert!(
        (lo - 99.0).abs() < 0.01,
        "select(false) lo should be 99, got {lo}"
    );
    assert!(
        (hi - 99.0).abs() < 0.01,
        "select(false) hi should be 99, got {hi}"
    );
}

// ---------------------------------------------------------------------------
// BinOp const-var via lowering tests
// ---------------------------------------------------------------------------

#[test]
fn test_binop_sub_const_var_translates() {
    // 5 - x via lowering
    let kernel = common::parse_kernel("fn sub_cv(x: f32, c: f32) -> f32 { c - x }");
    let graph = kernel_to_graph(&kernel, &[5.0]).expect("build 5-x graph");
    let (lo, hi) = ibp_scalar(&graph, 1.0, 3.0);
    assert!(
        lo <= 2.0 + 0.01,
        "5-x lower on [1,3] should be <= 2, got {lo}"
    );
    assert!(
        hi >= 4.0 - 0.01,
        "5-x upper on [1,3] should be >= 4, got {hi}"
    );
}

#[test]
fn test_binop_div_const_var_translates() {
    // 6 / x via lowering
    let kernel = common::parse_kernel("fn div_cv(x: f32, c: f32) -> f32 { c / x }");
    let graph = kernel_to_graph(&kernel, &[6.0]).expect("build 6/x graph");
    let (lo, hi) = ibp_scalar(&graph, 2.0, 3.0);
    assert!(
        lo <= 2.0 + 0.01,
        "6/x lower on [2,3] should be <= 2, got {lo}"
    );
    assert!(
        hi >= 3.0 - 0.01,
        "6/x upper on [2,3] should be >= 3, got {hi}"
    );
}
