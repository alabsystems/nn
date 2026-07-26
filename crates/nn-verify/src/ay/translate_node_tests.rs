// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for translate_node.rs: per-node SMT translation.
//!
//! Tests cover helpers (get_node, get_param, eval_unary_ground), literals,
//! params, and binary ops. Advanced tests (Compare, UnaryFn, Clamp, MinMax,
//! Select, SumReduce, Powi) are in translate_node_tests_advanced.rs.

use super::*;
use nn_dsl::ir::{BinOpKind, IRNodeKind, NodeId, UnaryFnKind};
use std::collections::HashSet;
use ay_bindings::{Expr, Sort, AYProgram};

pub(super) fn setup() -> (AYProgram, Sort, HashSet<String>, bool) {
    let program = AYProgram::new();
    let real_sort = Sort::real();
    let declared_ufs = HashSet::new();
    let uses_uf_approx = false;
    (program, real_sort, declared_ufs, uses_uf_approx)
}

/// Helper: translate a single node given pre-built node_exprs and param_exprs.
pub(super) fn translate_one(
    kind: &IRNodeKind,
    node_exprs: &[Expr],
    param_exprs: &[Expr],
    node_ground_values: &[Option<f64>],
    param_ground_values: &[Option<f64>],
) -> Result<(Expr, Option<f64>, AYProgram, bool), SmtError> {
    let (mut program, real_sort, mut declared_ufs, mut uses_uf_approx) = setup();
    let (expr, ground) = translate_node(
        kind,
        node_exprs,
        param_exprs,
        node_ground_values,
        param_ground_values,
        &mut program,
        &real_sort,
        &mut declared_ufs,
        &mut uses_uf_approx,
    )?;
    Ok((expr, ground, program, uses_uf_approx))
}

// ── get_node / get_param helpers ──

#[test]
fn test_get_node_valid_index() {
    let exprs = vec![Expr::real(1), Expr::real(2), Expr::real(3)];
    let result = get_node(&exprs, 1).expect("should succeed");
    assert_eq!(format!("{result}"), format!("{}", Expr::real(2)));
}

#[test]
fn test_get_node_out_of_bounds() {
    let exprs = vec![Expr::real(1)];
    let err = get_node(&exprs, 5).unwrap_err();
    match err {
        SmtError::IndexOutOfBounds {
            context,
            index,
            length,
        } => {
            assert_eq!(context, "node_exprs");
            assert_eq!(index, 5);
            assert_eq!(length, 1);
        }
        other => panic!("expected IndexOutOfBounds, got {other:?}"),
    }
}

#[test]
fn test_get_param_valid_index() {
    let exprs = vec![Expr::real(10), Expr::real(20)];
    let result = get_param(&exprs, 0).expect("should succeed");
    assert_eq!(format!("{result}"), format!("{}", Expr::real(10)));
}

#[test]
fn test_get_param_out_of_bounds() {
    let exprs: Vec<Expr> = vec![];
    let err = get_param(&exprs, 0).unwrap_err();
    match err {
        SmtError::IndexOutOfBounds { context, .. } => {
            assert_eq!(context, "param_exprs");
        }
        other => panic!("expected IndexOutOfBounds, got {other:?}"),
    }
}

// ── get_node_ground ──

#[test]
fn test_get_node_ground_present() {
    let grounds = vec![Some(1.0), None, Some(3.0)];
    assert_eq!(get_node_ground(&grounds, 0), Some(1.0));
    assert_eq!(get_node_ground(&grounds, 1), None);
    assert_eq!(get_node_ground(&grounds, 2), Some(3.0));
}

#[test]
fn test_get_node_ground_out_of_bounds_returns_none() {
    let grounds: Vec<Option<f64>> = vec![Some(1.0)];
    assert_eq!(get_node_ground(&grounds, 10), None);
}

// ── eval_unary_ground ──

#[test]
fn test_eval_unary_ground_abs() {
    assert_eq!(eval_unary_ground(UnaryFnKind::Abs, Some(-5.0)), Some(5.0));
    assert_eq!(eval_unary_ground(UnaryFnKind::Abs, Some(3.0)), Some(3.0));
}

#[test]
fn test_eval_unary_ground_recip() {
    assert_eq!(eval_unary_ground(UnaryFnKind::Recip, Some(2.0)), Some(0.5));
    // Division by zero returns None
    assert_eq!(eval_unary_ground(UnaryFnKind::Recip, Some(0.0)), None);
}

#[test]
fn test_eval_unary_ground_sqrt() {
    assert_eq!(eval_unary_ground(UnaryFnKind::Sqrt, Some(4.0)), Some(2.0));
    // Negative input returns None
    assert_eq!(eval_unary_ground(UnaryFnKind::Sqrt, Some(-1.0)), None);
}

#[test]
fn test_eval_unary_ground_rsqrt() {
    let result = eval_unary_ground(UnaryFnKind::Rsqrt, Some(4.0));
    assert!((result.unwrap() - 0.5).abs() < 1e-10);
    // Zero and negative inputs return None
    assert_eq!(eval_unary_ground(UnaryFnKind::Rsqrt, Some(0.0)), None);
    assert_eq!(eval_unary_ground(UnaryFnKind::Rsqrt, Some(-1.0)), None);
}

#[test]
fn test_eval_unary_ground_sin_cos_exp() {
    let sin_val = eval_unary_ground(UnaryFnKind::Sin, Some(0.0));
    assert!((sin_val.unwrap() - 0.0).abs() < 1e-10);

    let cos_val = eval_unary_ground(UnaryFnKind::Cos, Some(0.0));
    assert!((cos_val.unwrap() - 1.0).abs() < 1e-10);

    let exp_val = eval_unary_ground(UnaryFnKind::Exp, Some(0.0));
    assert!((exp_val.unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_eval_unary_ground_none_input() {
    assert_eq!(eval_unary_ground(UnaryFnKind::Abs, None), None);
    assert_eq!(eval_unary_ground(UnaryFnKind::Exp, None), None);
}

#[test]
fn test_eval_unary_ground_nonfinite_result_returns_none() {
    // exp(1000.0) overflows f64 → infinity → None
    assert_eq!(eval_unary_ground(UnaryFnKind::Exp, Some(1000.0)), None);
}

// ── Literal node ──

#[test]
fn test_translate_literal_positive() {
    let kind = IRNodeKind::Literal(2.5);
    let (expr, ground, _, _) = translate_one(&kind, &[], &[], &[], &[]).unwrap();
    assert_eq!(ground, Some(2.5));
    let s = format!("{expr}");
    assert!(!s.is_empty(), "literal should produce non-empty expression");
}

#[test]
fn test_translate_literal_negative() {
    let kind = IRNodeKind::Literal(-3.0);
    let (_, ground, _, _) = translate_one(&kind, &[], &[], &[], &[]).unwrap();
    assert_eq!(ground, Some(-3.0));
}

#[test]
fn test_translate_literal_zero() {
    let kind = IRNodeKind::Literal(0.0);
    let (_, ground, _, _) = translate_one(&kind, &[], &[], &[], &[]).unwrap();
    assert_eq!(ground, Some(0.0));
}

// ── Param node ──

#[test]
fn test_translate_param() {
    let param_expr = Expr::real(42);
    let kind = IRNodeKind::Param(0);
    let (expr, ground, _, _) = translate_one(
        &kind,
        &[],
        std::slice::from_ref(&param_expr),
        &[],
        &[Some(42.0)],
    )
    .unwrap();
    assert_eq!(format!("{expr}"), format!("{param_expr}"));
    assert_eq!(ground, Some(42.0));
}

#[test]
fn test_translate_param_symbolic_no_ground() {
    let (mut program, real_sort, _, _) = setup();
    let x = program.declare_const("x", real_sort);
    let kind = IRNodeKind::Param(0);
    let (expr, ground, _, _) =
        translate_one(&kind, &[], std::slice::from_ref(&x), &[], &[None]).unwrap();
    assert_eq!(format!("{expr}"), format!("{x}"));
    assert_eq!(ground, None);
}

// ── BinOp: Add, Sub, Mul ──

#[test]
fn test_translate_binop_add() {
    let a = Expr::real(3);
    let b = Expr::real(5);
    let expected = a.clone().real_add(b.clone());
    let kind = IRNodeKind::BinOp {
        op: BinOpKind::Add,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (expr, ground, _, _) =
        translate_one(&kind, &[a, b], &[], &[Some(3.0), Some(5.0)], &[]).unwrap();
    assert_eq!(format!("{expr}"), format!("{expected}"));
    assert_eq!(ground, Some(8.0));
}

#[test]
fn test_translate_binop_sub_operand_order() {
    // Sub is non-commutative: lhs - rhs, not rhs - lhs
    let a = Expr::real(10);
    let b = Expr::real(3);
    let expected = a.clone().real_sub(b.clone());
    let kind = IRNodeKind::BinOp {
        op: BinOpKind::Sub,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (expr, ground, _, _) =
        translate_one(&kind, &[a, b], &[], &[Some(10.0), Some(3.0)], &[]).unwrap();
    assert_eq!(format!("{expr}"), format!("{expected}"));
    // ground should be 10 - 3 = 7, not 3 - 10 = -7
    assert_eq!(ground, Some(7.0));
}

#[test]
fn test_translate_binop_mul() {
    let a = Expr::real(4);
    let b = Expr::real(6);
    let kind = IRNodeKind::BinOp {
        op: BinOpKind::Mul,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (_, ground, _, _) =
        translate_one(&kind, &[a, b], &[], &[Some(4.0), Some(6.0)], &[]).unwrap();
    assert_eq!(ground, Some(24.0));
}

// ── BinOp: Div with zero-guard ──

#[test]
fn test_translate_binop_div_operand_order() {
    // Div is non-commutative: lhs / rhs
    let a = Expr::real(12);
    let b = Expr::real(4);
    let expected = a.clone().real_div(b.clone());
    let kind = IRNodeKind::BinOp {
        op: BinOpKind::Div,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (expr, ground, program, _) =
        translate_one(&kind, &[a, b], &[], &[Some(12.0), Some(4.0)], &[]).unwrap();
    assert_eq!(format!("{expr}"), format!("{expected}"));
    // ground should be 12 / 4 = 3, not 4 / 12
    assert_eq!(ground, Some(3.0));
    // The program must contain the zero-divisor guard assertion
    let smt2 = program.to_string();
    assert!(
        smt2.contains("assert"),
        "div must emit zero-divisor guard assertion, got: {smt2}"
    );
}

#[test]
fn test_translate_binop_div_ground_zero_divisor() {
    // Division by zero in ground computation should yield None (not NaN)
    let a = Expr::real(5);
    let b = Expr::real(0);
    let kind = IRNodeKind::BinOp {
        op: BinOpKind::Div,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (_, ground, _, _) =
        translate_one(&kind, &[a, b], &[], &[Some(5.0), Some(0.0)], &[]).unwrap();
    assert_eq!(ground, None, "div by zero ground should be None");
}

#[test]
fn test_translate_binop_symbolic_no_ground() {
    // When one operand is symbolic (no ground), ground should be None
    let a = Expr::real(5);
    let b = Expr::real(3);
    let kind = IRNodeKind::BinOp {
        op: BinOpKind::Add,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (_, ground, _, _) = translate_one(&kind, &[a, b], &[], &[Some(5.0), None], &[]).unwrap();
    assert_eq!(ground, None);
}

// -- Identity elimination tests extracted to translate_node_tests_identity.rs (#1567) --
#[path = "translate_node_tests_identity.rs"]
mod identity;

// Advanced tests (Compare, UnaryFn, Clamp, MinMax, Select, SumReduce, Powi)
// are in translate_node_tests_advanced.rs.
#[path = "translate_node_tests_advanced.rs"]
mod advanced;
