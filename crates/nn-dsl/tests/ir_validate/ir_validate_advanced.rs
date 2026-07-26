// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Advanced validation tests for KernelDef::validate().
//!
//! Tests for Compare, Select, SumReduce nodes. Topological ordering
//! and node ID consistency tests are in `ir_validate_topology.rs`.

use nn_dsl::ir::{
    CompareOpKind, IRError, IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType,
};

// ======================== Helpers ========================

fn param(name: &str) -> Param {
    Param::new(name, ScalarType::F32)
}

fn node(id: usize, kind: IRNodeKind) -> IRNode {
    IRNode::new(NodeId::new(id), kind)
}

// ======================== Compare validation ========================

#[test]
fn validate_all_compare_op_kinds() {
    let all_cmp_ops = [
        CompareOpKind::Eq,
        CompareOpKind::Ne,
        CompareOpKind::Lt,
        CompareOpKind::Le,
        CompareOpKind::Gt,
        CompareOpKind::Ge,
    ];
    for op in all_cmp_ops {
        let kernel = KernelDef::new(
            format!("test_{op:?}").to_lowercase(),
            vec![param("x"), param("y")],
            ScalarType::F32,
            vec![
                node(0, IRNodeKind::Param(0)),
                node(1, IRNodeKind::Param(1)),
                node(
                    2,
                    IRNodeKind::Compare {
                        op,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
                // Select uses the boolean comparison result
                node(
                    3,
                    IRNodeKind::Select {
                        cond: NodeId::new(2),
                        then_val: NodeId::new(0),
                        else_val: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(3),
        );
        assert!(
            kernel.validate().is_ok(),
            "kernel with Compare::{op:?} should pass"
        );
    }
}

#[test]
fn validate_compare_invalid_lhs_caught() {
    let kernel = KernelDef::new(
        "bad_cmp_lhs",
        vec![param("x")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(
                1,
                IRNodeKind::Compare {
                    op: CompareOpKind::Lt,
                    lhs: NodeId::new(8), // invalid
                    rhs: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(0),
    );
    let err = kernel
        .validate()
        .expect_err("invalid Compare lhs should fail");
    assert!(matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(8)));
}

#[test]
fn validate_compare_invalid_rhs_caught() {
    let kernel = KernelDef::new(
        "bad_cmp_rhs",
        vec![param("x")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(
                1,
                IRNodeKind::Compare {
                    op: CompareOpKind::Ge,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(7), // invalid
                },
            ),
        ],
        NodeId::new(0),
    );
    let err = kernel
        .validate()
        .expect_err("invalid Compare rhs should fail");
    assert!(matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(7)));
}

// ======================== Select validation ========================

#[test]
fn validate_select_valid_kernel() {
    let kernel = KernelDef::new(
        "select_valid",
        vec![param("x"), param("y")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(1, IRNodeKind::Param(1)),
            node(2, IRNodeKind::Literal(0.0)),
            node(
                3,
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(2),
                },
            ),
            node(
                4,
                IRNodeKind::Select {
                    cond: NodeId::new(3),
                    then_val: NodeId::new(0),
                    else_val: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(4),
    );
    assert!(kernel.validate().is_ok(), "valid Select kernel should pass");
}

#[test]
fn validate_select_invalid_cond_caught() {
    let kernel = KernelDef::new(
        "bad_select_cond",
        vec![param("x"), param("y")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(1, IRNodeKind::Param(1)),
            node(
                2,
                IRNodeKind::Select {
                    cond: NodeId::new(20), // invalid
                    then_val: NodeId::new(0),
                    else_val: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let err = kernel
        .validate()
        .expect_err("invalid Select cond should fail");
    assert!(matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(20)));
}

#[test]
fn validate_select_invalid_then_caught() {
    let kernel = KernelDef::new(
        "bad_select_then",
        vec![param("x"), param("y")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(1, IRNodeKind::Param(1)),
            node(
                2,
                IRNodeKind::Compare {
                    op: CompareOpKind::Eq,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            node(
                3,
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(15), // invalid
                    else_val: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(3),
    );
    let err = kernel
        .validate()
        .expect_err("invalid Select then_val should fail");
    assert!(matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(15)));
}

#[test]
fn validate_select_invalid_else_caught() {
    let kernel = KernelDef::new(
        "bad_select_else",
        vec![param("x"), param("y")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(1, IRNodeKind::Param(1)),
            node(
                2,
                IRNodeKind::Compare {
                    op: CompareOpKind::Ne,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            node(
                3,
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(0),
                    else_val: NodeId::new(30), // invalid
                },
            ),
        ],
        NodeId::new(3),
    );
    let err = kernel
        .validate()
        .expect_err("invalid Select else_val should fail");
    assert!(matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(30)));
}

// ======================== SumReduce validation ========================

#[test]
fn validate_sum_reduce_empty_caught() {
    let kernel = KernelDef::new(
        "empty_reduce",
        vec![param("x")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(1, IRNodeKind::SumReduce { inputs: vec![] }),
        ],
        NodeId::new(1),
    );
    let err = kernel.validate().expect_err("empty SumReduce should fail");
    assert!(
        matches!(err, IRError::EmptySumReduce(id) if id == NodeId::new(1)),
        "expected EmptySumReduce, got {err:?}"
    );
}

#[test]
fn validate_sum_reduce_invalid_ref_caught() {
    let kernel = KernelDef::new(
        "bad_reduce_ref",
        vec![param("x")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(
                1,
                IRNodeKind::SumReduce {
                    inputs: vec![NodeId::new(0), NodeId::new(42)], // second ref invalid
                },
            ),
        ],
        NodeId::new(1),
    );
    let err = kernel
        .validate()
        .expect_err("SumReduce with invalid ref should fail");
    assert!(matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(42)));
}

#[test]
fn validate_sum_reduce_single_valid() {
    let kernel = KernelDef::new(
        "single_reduce",
        vec![param("x")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(
                1,
                IRNodeKind::SumReduce {
                    inputs: vec![NodeId::new(0)],
                },
            ),
        ],
        NodeId::new(1),
    );
    assert!(
        kernel.validate().is_ok(),
        "SumReduce with single valid ref should pass"
    );
}
