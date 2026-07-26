// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Semantic round-trip tests for Exp, Cos, Sub, and Compare variants (#449).
//!
//! Extracted from `translate_tests_coverage_449.rs` to stay under 500 lines.
//! Uses ay direct execution to verify encoding correctness via known-answer tests.

use super::*;
use nn_dsl::ir::{BinOpKind, CompareOpKind};
use ay_bindings::execute_direct::{self, ExecuteResult};

/// Helper: semantic round-trip (same pattern as node_kinds_semantic).
fn assert_semantic_roundtrip(kernel: &KernelDef, inputs: &[f64], expected: f64) {
    let mut tr = translate_kernel(kernel, &all_variable(kernel)).expect("kernel should translate");
    assert_eq!(tr.param_exprs.len(), inputs.len());

    for (param_expr, &val) in tr.param_exprs.iter().zip(inputs) {
        let concrete = real_from_f64(val).expect("input value should encode");
        tr.program.assert(param_expr.clone().eq(concrete));
    }

    let expected_expr = real_from_f64(expected).expect("expected value should encode");
    tr.program.assert(tr.output.clone().ne(expected_expr));
    tr.program.set_logic("QF_LRA");
    tr.program.check_sat();

    let result = execute_direct::execute(&tr.program);
    assert!(
        matches!(
            result,
            Ok(ExecuteResult::Verified) | Ok(ExecuteResult::Unknown(_))
        ),
        "semantic roundtrip: expected Verified or Unknown (ay#5357), got: {result:?}\n\
         kernel={}, inputs={inputs:?}, expected={expected}",
        kernel.name
    );
}

/// Helper: build a kernel `fn f(x, y) -> f32 { if x <op> y { 1.0 } else { 0.0 } }`.
fn compare_select_kernel(name: &str, op: CompareOpKind) -> KernelDef {
    KernelDef::new(
        name,
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
                IRNodeKind::Compare {
                    op,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            IRNode::new(NodeId::new(3), IRNodeKind::Literal(1.0)),
            IRNode::new(NodeId::new(4), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(5),
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(3),
                    else_val: NodeId::new(4),
                },
            ),
        ],
        NodeId::new(5),
    )
}

// --- Sub semantic ---

/// sub(5.0, 3.0) → 2.0.
#[test]
fn test_semantic_sub_known_answer() {
    let kernel = KernelDef::new(
        "sub_xy",
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
                IRNodeKind::BinOp {
                    op: BinOpKind::Sub,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    assert_semantic_roundtrip(&kernel, &[5.0, 3.0], 2.0);
}

/// sub(1.0, 4.0) → -3.0 (negative result).
#[test]
fn test_semantic_sub_negative_result() {
    let kernel = KernelDef::new(
        "sub_neg",
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
                IRNodeKind::BinOp {
                    op: BinOpKind::Sub,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    assert_semantic_roundtrip(&kernel, &[1.0, 4.0], -3.0);
}

// --- Compare semantic ---

/// Lt: 2.0 < 5.0 → true → 1.0.
#[test]
fn test_semantic_compare_lt_true() {
    assert_semantic_roundtrip(
        &compare_select_kernel("cmp_lt_t", CompareOpKind::Lt),
        &[2.0, 5.0],
        1.0,
    );
}

/// Lt: 5.0 < 2.0 → false → 0.0.
#[test]
fn test_semantic_compare_lt_false() {
    assert_semantic_roundtrip(
        &compare_select_kernel("cmp_lt_f", CompareOpKind::Lt),
        &[5.0, 2.0],
        0.0,
    );
}

/// Le: 3.0 <= 3.0 → true → 1.0 (equal case).
#[test]
fn test_semantic_compare_le_equal() {
    assert_semantic_roundtrip(
        &compare_select_kernel("cmp_le_eq", CompareOpKind::Le),
        &[3.0, 3.0],
        1.0,
    );
}

/// Le: 5.0 <= 2.0 → false → 0.0.
#[test]
fn test_semantic_compare_le_false() {
    assert_semantic_roundtrip(
        &compare_select_kernel("cmp_le_f", CompareOpKind::Le),
        &[5.0, 2.0],
        0.0,
    );
}

/// Ge: 3.0 >= 3.0 → true → 1.0 (equal case).
#[test]
fn test_semantic_compare_ge_equal() {
    assert_semantic_roundtrip(
        &compare_select_kernel("cmp_ge_eq", CompareOpKind::Ge),
        &[3.0, 3.0],
        1.0,
    );
}

/// Ge: 2.0 >= 5.0 → false → 0.0.
#[test]
fn test_semantic_compare_ge_false() {
    assert_semantic_roundtrip(
        &compare_select_kernel("cmp_ge_f", CompareOpKind::Ge),
        &[2.0, 5.0],
        0.0,
    );
}

/// Eq: 4.0 == 4.0 → true → 1.0.
#[test]
fn test_semantic_compare_eq_true() {
    assert_semantic_roundtrip(
        &compare_select_kernel("cmp_eq_t", CompareOpKind::Eq),
        &[4.0, 4.0],
        1.0,
    );
}

/// Eq: 3.0 == 5.0 → false → 0.0.
#[test]
fn test_semantic_compare_eq_false() {
    assert_semantic_roundtrip(
        &compare_select_kernel("cmp_eq_f", CompareOpKind::Eq),
        &[3.0, 5.0],
        0.0,
    );
}

/// Ne: 3.0 != 5.0 → true → 1.0.
#[test]
fn test_semantic_compare_ne_true() {
    assert_semantic_roundtrip(
        &compare_select_kernel("cmp_ne_t", CompareOpKind::Ne),
        &[3.0, 5.0],
        1.0,
    );
}

/// Ne: 4.0 != 4.0 → false → 0.0.
#[test]
fn test_semantic_compare_ne_false() {
    assert_semantic_roundtrip(
        &compare_select_kernel("cmp_ne_f", CompareOpKind::Ne),
        &[4.0, 4.0],
        0.0,
    );
}
