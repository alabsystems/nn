// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Property-based tests for KernelDef::validate().
//!
//! Tests that validate() accepts well-formed kernels and catches invalid
//! BinOp/UnaryFn references and invalid output references.
//!
//! Powi, Clamp, MinMax, param, and literal validation tests are in
//! `ir_validate_refs.rs`. Compare, Select, SumReduce, and topological
//! ordering tests are in `ir_validate_advanced.rs`.

use nn_dsl::ir::{
    BinOpKind, IRError, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, Param, ScalarType,
    UnaryFnKind,
};

// ======================== Helpers ========================

fn param(name: &str) -> Param {
    Param::new(name, ScalarType::F32)
}

fn node(id: usize, kind: IRNodeKind) -> IRNode {
    IRNode::new(NodeId::new(id), kind)
}

/// Build a minimal valid kernel: one param, one node, output = node 0.
fn minimal_kernel() -> KernelDef {
    KernelDef::new(
        "minimal",
        vec![param("x")],
        ScalarType::F32,
        vec![node(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    )
}

// ======================== Valid kernels pass ========================

#[test]
fn validate_minimal_kernel_passes() {
    let kernel = minimal_kernel();
    assert!(
        kernel.validate().is_ok(),
        "minimal valid kernel should pass"
    );
}

#[test]
fn validate_kernel_with_all_node_kinds() {
    // Build a kernel that exercises every IRNodeKind variant
    let kernel = KernelDef::new(
        "all_kinds",
        vec![param("x"), param("y")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),     // x
            node(1, IRNodeKind::Param(1)),     // y
            node(2, IRNodeKind::Literal(1.5)), // const
            node(
                3,
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            node(
                4,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(0),
                },
            ),
            node(
                5,
                IRNodeKind::Powi {
                    base: NodeId::new(4),
                    exp: 2,
                },
            ),
            node(
                6,
                IRNodeKind::Clamp {
                    input: NodeId::new(3),
                    min: NodeId::new(2),
                    max: NodeId::new(2),
                },
            ),
            node(
                7,
                IRNodeKind::MinMax {
                    op: MinMaxKind::Max,
                    lhs: NodeId::new(5),
                    rhs: NodeId::new(6),
                },
            ),
            node(
                8,
                IRNodeKind::Compare {
                    op: nn_dsl::ir::CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(2),
                },
            ),
            node(
                9,
                IRNodeKind::Select {
                    cond: NodeId::new(8),
                    then_val: NodeId::new(3),
                    else_val: NodeId::new(7),
                },
            ),
            node(
                10,
                IRNodeKind::SumReduce {
                    inputs: vec![NodeId::new(3), NodeId::new(5), NodeId::new(9)],
                },
            ),
        ],
        NodeId::new(10),
    );
    assert!(
        kernel.validate().is_ok(),
        "kernel with all node kinds should pass validation"
    );
}

#[test]
fn validate_all_unary_fn_kinds() {
    let all_unary = [
        UnaryFnKind::Sin,
        UnaryFnKind::Cos,
        UnaryFnKind::Sqrt,
        UnaryFnKind::Rsqrt,
        UnaryFnKind::Exp,
        UnaryFnKind::Abs,
        UnaryFnKind::Recip,
    ];
    for op in all_unary {
        let kernel = KernelDef::new(
            format!("test_{op}"),
            vec![param("x")],
            ScalarType::F32,
            vec![
                node(0, IRNodeKind::Param(0)),
                node(
                    1,
                    IRNodeKind::UnaryFn {
                        op,
                        input: NodeId::new(0),
                    },
                ),
            ],
            NodeId::new(1),
        );
        assert!(
            kernel.validate().is_ok(),
            "kernel with UnaryFn::{op:?} should pass"
        );
    }
}

#[test]
fn validate_all_binop_kinds() {
    let all_binops = [
        BinOpKind::Add,
        BinOpKind::Sub,
        BinOpKind::Mul,
        BinOpKind::Div,
    ];
    for op in all_binops {
        let kernel = KernelDef::new(
            format!("test_{op:?}").to_lowercase(),
            vec![param("x"), param("y")],
            ScalarType::F32,
            vec![
                node(0, IRNodeKind::Param(0)),
                node(1, IRNodeKind::Param(1)),
                node(
                    2,
                    IRNodeKind::BinOp {
                        op,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(2),
        );
        assert!(
            kernel.validate().is_ok(),
            "kernel with BinOp::{op:?} should pass"
        );
    }
}

// ======================== Invalid output ref ========================

#[test]
fn validate_invalid_output_ref_caught() {
    let kernel = KernelDef::new(
        "bad_output",
        vec![param("x")],
        ScalarType::F32,
        vec![node(0, IRNodeKind::Param(0))],
        NodeId::new(99),
    );
    let err = kernel
        .validate()
        .expect_err("should catch invalid output ref");
    assert!(
        matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(99)),
        "expected InvalidNodeRef(99), got {err:?}"
    );
}

#[test]
fn validate_output_ref_one_past_end() {
    let kernel = KernelDef::new(
        "off_by_one",
        vec![param("x")],
        ScalarType::F32,
        vec![node(0, IRNodeKind::Param(0))],
        NodeId::new(1),
    );
    let err = kernel
        .validate()
        .expect_err("off-by-one output ref should fail");
    assert!(matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(1)));
}

// ======================== Invalid node references ========================

#[test]
fn validate_binop_invalid_lhs_caught() {
    let kernel = KernelDef::new(
        "bad_lhs",
        vec![param("x")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(
                1,
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(5), // invalid
                    rhs: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    let err = kernel
        .validate()
        .expect_err("invalid BinOp lhs should fail");
    assert!(matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(5)));
}

#[test]
fn validate_binop_invalid_rhs_caught() {
    let kernel = KernelDef::new(
        "bad_rhs",
        vec![param("x")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(
                1,
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(10), // invalid
                },
            ),
        ],
        NodeId::new(1),
    );
    let err = kernel
        .validate()
        .expect_err("invalid BinOp rhs should fail");
    assert!(matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(10)));
}

#[test]
fn validate_unary_fn_invalid_input_caught() {
    let kernel = KernelDef::new(
        "bad_unary",
        vec![param("x")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(
                1,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(3), // invalid
                },
            ),
        ],
        NodeId::new(1),
    );
    let err = kernel
        .validate()
        .expect_err("invalid UnaryFn input should fail");
    assert!(matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(3)));
}
