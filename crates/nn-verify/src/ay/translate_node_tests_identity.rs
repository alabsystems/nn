// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Identity elimination tests for translate_node.rs (#1567).
//!
//! Tests verify that x*1→x, x+0→x, x-0→x, x*0→0, x/1→x optimizations
//! are applied during SMT translation.

use super::*;
use nn_dsl::ir::{BinOpKind, IRNodeKind, NodeId};
use ay_bindings::{Expr, Sort, AYProgram};

/// Helper to create a named symbolic variable for identity elimination tests.
fn make_symbolic_var(name: &str) -> (Expr, AYProgram) {
    let mut program = AYProgram::new();
    let real_sort = Sort::real();
    let expr = program.declare_const(name, real_sort);
    (expr, program)
}

#[test]
fn test_translate_mul_by_one_eliminated_rhs() {
    // x * 1.0 → x (no real_mul emitted)
    let (x, _prog) = make_symbolic_var("x");
    let one = Expr::real(1);
    let kind = IRNodeKind::BinOp {
        op: BinOpKind::Mul,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (expr, _, _, _) =
        translate_one(&kind, &[x.clone(), one], &[], &[None, Some(1.0)], &[]).unwrap();
    // The expression should be just x, not (* x 1).
    assert_eq!(format!("{expr}"), format!("{x}"));
}

#[test]
fn test_translate_mul_by_one_eliminated_lhs() {
    // 1.0 * x → x (no real_mul emitted)
    let one = Expr::real(1);
    let (x, _prog) = make_symbolic_var("x");
    let kind = IRNodeKind::BinOp {
        op: BinOpKind::Mul,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (expr, _, _, _) =
        translate_one(&kind, &[one, x.clone()], &[], &[Some(1.0), None], &[]).unwrap();
    assert_eq!(format!("{expr}"), format!("{x}"));
}

#[test]
fn test_translate_add_zero_eliminated_rhs() {
    // x + 0.0 → x (no real_add emitted)
    let (x, _prog) = make_symbolic_var("x");
    let zero = Expr::real(0);
    let kind = IRNodeKind::BinOp {
        op: BinOpKind::Add,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (expr, _, _, _) =
        translate_one(&kind, &[x.clone(), zero], &[], &[None, Some(0.0)], &[]).unwrap();
    assert_eq!(format!("{expr}"), format!("{x}"));
}

#[test]
fn test_translate_add_zero_eliminated_lhs() {
    // 0.0 + x → x
    let zero = Expr::real(0);
    let (x, _prog) = make_symbolic_var("x");
    let kind = IRNodeKind::BinOp {
        op: BinOpKind::Add,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (expr, _, _, _) =
        translate_one(&kind, &[zero, x.clone()], &[], &[Some(0.0), None], &[]).unwrap();
    assert_eq!(format!("{expr}"), format!("{x}"));
}

#[test]
fn test_translate_sub_zero_eliminated() {
    // x - 0.0 → x
    let (x, _prog) = make_symbolic_var("x");
    let zero = Expr::real(0);
    let kind = IRNodeKind::BinOp {
        op: BinOpKind::Sub,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (expr, _, _, _) =
        translate_one(&kind, &[x.clone(), zero], &[], &[None, Some(0.0)], &[]).unwrap();
    assert_eq!(format!("{expr}"), format!("{x}"));
}

#[test]
fn test_translate_mul_by_zero_folded() {
    // x * 0.0 → 0.0
    let (x, _prog) = make_symbolic_var("x");
    let zero = Expr::real(0);
    let kind = IRNodeKind::BinOp {
        op: BinOpKind::Mul,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (expr, _, _, _) = translate_one(&kind, &[x, zero], &[], &[None, Some(0.0)], &[]).unwrap();
    let expected = Expr::real(0);
    assert_eq!(format!("{expr}"), format!("{expected}"));
}

#[test]
fn test_translate_div_by_one_eliminated() {
    // x / 1.0 → x (no div guard emitted either)
    let (x, _prog) = make_symbolic_var("x");
    let one = Expr::real(1);
    let kind = IRNodeKind::BinOp {
        op: BinOpKind::Div,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (expr, _, program, _) =
        translate_one(&kind, &[x.clone(), one], &[], &[None, Some(1.0)], &[]).unwrap();
    let smt2 = program.to_string();
    // Division by 1.0 → no division at all, no zero-guard assertion.
    assert!(
        !smt2.contains("(assert"),
        "div by 1.0 should skip zero-guard assertion, got SMT2: {smt2}"
    );
    assert_eq!(format!("{expr}"), format!("{x}"));
}

#[test]
fn test_translate_mul_nonidentity_preserved() {
    // x * 2.0 → should still produce real_mul (not eliminated)
    let (x, _prog) = make_symbolic_var("x");
    let two = Expr::real(2);
    let expected = x.clone().real_mul(two.clone());
    let kind = IRNodeKind::BinOp {
        op: BinOpKind::Mul,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (expr, _, _, _) = translate_one(&kind, &[x, two], &[], &[None, Some(2.0)], &[]).unwrap();
    // 2.0 is not identity → real_mul should be preserved.
    assert_eq!(format!("{expr}"), format!("{expected}"));
}
