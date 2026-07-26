// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Advanced unit tests for translate_node.rs: Compare, UnaryFn, Clamp, MinMax, Select, SumReduce, Powi.
//!
//! Split from translate_node_tests.rs to stay under the 500-line file limit.

use super::translate_one;
use nn_dsl::ir::{CompareOpKind, IRNodeKind, MinMaxKind, NodeId, UnaryFnKind};
use ay_bindings::Expr;

// ── Compare ops ──

#[test]
fn test_translate_compare_lt() {
    let a = Expr::real(1);
    let b = Expr::real(2);
    let expected = a.clone().real_lt(b.clone());
    let kind = IRNodeKind::Compare {
        op: CompareOpKind::Lt,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (expr, ground, _, _) =
        translate_one(&kind, &[a, b], &[], &[Some(1.0), Some(2.0)], &[]).unwrap();
    assert_eq!(format!("{expr}"), format!("{expected}"));
    // Compare nodes always produce None ground (boolean, not real)
    assert_eq!(ground, None);
}

#[test]
fn test_translate_compare_ge() {
    let a = Expr::real(5);
    let b = Expr::real(3);
    let expected = a.clone().real_ge(b.clone());
    let kind = IRNodeKind::Compare {
        op: CompareOpKind::Ge,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (expr, ground, _, _) =
        translate_one(&kind, &[a, b], &[], &[Some(5.0), Some(3.0)], &[]).unwrap();
    assert_eq!(format!("{expr}"), format!("{expected}"));
    assert_eq!(ground, None);
}

#[test]
fn test_translate_compare_eq_ne() {
    let a = Expr::real(7);
    let b = Expr::real(7);

    // Eq
    let kind_eq = IRNodeKind::Compare {
        op: CompareOpKind::Eq,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (expr_eq, _, _, _) =
        translate_one(&kind_eq, &[a.clone(), b.clone()], &[], &[None, None], &[]).unwrap();
    let expected_eq = a.clone().eq(b.clone());
    assert_eq!(format!("{expr_eq}"), format!("{expected_eq}"));

    // Ne
    let kind_ne = IRNodeKind::Compare {
        op: CompareOpKind::Ne,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (expr_ne, _, _, _) =
        translate_one(&kind_ne, &[a.clone(), b.clone()], &[], &[None, None], &[]).unwrap();
    let expected_ne = a.ne(b);
    assert_eq!(format!("{expr_ne}"), format!("{expected_ne}"));
}

// ── UnaryFn: Recip zero-guard ──

#[test]
fn test_translate_unary_recip_zero_guard() {
    // Recip should emit zero-guard assertion (arg != 0)
    let arg = Expr::real(5);
    let kind = IRNodeKind::UnaryFn {
        op: UnaryFnKind::Recip,
        input: NodeId::new(0),
    };
    let (_, _, program, uses_uf) = translate_one(&kind, &[arg], &[], &[None], &[]).unwrap();
    let smt2 = program.to_string();
    assert!(
        smt2.contains("assert"),
        "recip must emit zero-divisor guard, got: {smt2}"
    );
    // Recip is exact encoding (1/x), not UF approximation
    assert!(!uses_uf, "recip should not use UF approximation");
}

// ── UnaryFn: UF approximation for transcendentals ──

#[test]
fn test_translate_unary_sin_uses_uf() {
    let arg = Expr::real(1);
    let kind = IRNodeKind::UnaryFn {
        op: UnaryFnKind::Sin,
        input: NodeId::new(0),
    };
    let (_, _, _, uses_uf) = translate_one(&kind, &[arg], &[], &[None], &[]).unwrap();
    assert!(uses_uf, "sin on symbolic arg should use UF approximation");
}

#[test]
fn test_translate_unary_exp_uses_uf() {
    let arg = Expr::real(1);
    let kind = IRNodeKind::UnaryFn {
        op: UnaryFnKind::Exp,
        input: NodeId::new(0),
    };
    let (_, _, _, uses_uf) = translate_one(&kind, &[arg], &[], &[None], &[]).unwrap();
    assert!(uses_uf, "exp on symbolic arg should use UF approximation");
}

#[test]
fn test_translate_unary_abs_exact_encoding() {
    let arg = Expr::real(1);
    let kind = IRNodeKind::UnaryFn {
        op: UnaryFnKind::Abs,
        input: NodeId::new(0),
    };
    let (_, _, _, uses_uf) = translate_one(&kind, &[arg], &[], &[None], &[]).unwrap();
    assert!(!uses_uf, "abs should use exact encoding, not UF");
}

// ── UnaryFn: ground-folding ──

#[test]
fn test_translate_unary_ground_fold_sin() {
    // sin(0.0) = 0.0, should ground-fold to literal
    let arg = Expr::real(0);
    let kind = IRNodeKind::UnaryFn {
        op: UnaryFnKind::Sin,
        input: NodeId::new(0),
    };
    let (_, ground, _, uses_uf) = translate_one(&kind, &[arg], &[], &[Some(0.0)], &[]).unwrap();
    assert_eq!(ground, Some(0.0), "sin(0) should ground-fold to 0.0");
    assert!(!uses_uf, "ground-folded sin should not use UF");
}

#[test]
fn test_translate_unary_ground_fold_exp_overflow_falls_back() {
    // exp(1000.0) overflows f64 → eval_unary_ground returns None
    // → falls through to UF path
    let arg = Expr::real(1000);
    let kind = IRNodeKind::UnaryFn {
        op: UnaryFnKind::Exp,
        input: NodeId::new(0),
    };
    let (_, ground, _, uses_uf) = translate_one(&kind, &[arg], &[], &[Some(1000.0)], &[]).unwrap();
    assert_eq!(ground, None, "exp(1000) overflow should not produce ground");
    assert!(uses_uf, "exp(1000) overflow should fall back to UF");
}

// ── Clamp ──

#[test]
fn test_translate_clamp_ground() {
    let x = Expr::real(10);
    let lo = Expr::real(0);
    let hi = Expr::real(5);
    let kind = IRNodeKind::Clamp {
        input: NodeId::new(0),
        min: NodeId::new(1),
        max: NodeId::new(2),
    };
    let (_, ground, _, _) = translate_one(
        &kind,
        &[x, lo, hi],
        &[],
        &[Some(10.0), Some(0.0), Some(5.0)],
        &[],
    )
    .unwrap();
    // clamp(10, 0, 5) = 5
    assert_eq!(ground, Some(5.0));
}

#[test]
fn test_translate_clamp_symbolic_no_ground() {
    let x = Expr::real(3);
    let lo = Expr::real(0);
    let hi = Expr::real(5);
    let kind = IRNodeKind::Clamp {
        input: NodeId::new(0),
        min: NodeId::new(1),
        max: NodeId::new(2),
    };
    let (_, ground, _, _) =
        translate_one(&kind, &[x, lo, hi], &[], &[None, Some(0.0), Some(5.0)], &[]).unwrap();
    assert_eq!(ground, None, "symbolic input should yield no ground");
}

// ── MinMax ──

#[test]
fn test_translate_min_ground() {
    let a = Expr::real(3);
    let b = Expr::real(7);
    let kind = IRNodeKind::MinMax {
        op: MinMaxKind::Min,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (_, ground, _, _) =
        translate_one(&kind, &[a, b], &[], &[Some(3.0), Some(7.0)], &[]).unwrap();
    assert_eq!(ground, Some(3.0));
}

#[test]
fn test_translate_max_ground() {
    let a = Expr::real(3);
    let b = Expr::real(7);
    let kind = IRNodeKind::MinMax {
        op: MinMaxKind::Max,
        lhs: NodeId::new(0),
        rhs: NodeId::new(1),
    };
    let (_, ground, _, _) =
        translate_one(&kind, &[a, b], &[], &[Some(3.0), Some(7.0)], &[]).unwrap();
    assert_eq!(ground, Some(7.0));
}

// ── Select ──

#[test]
fn test_translate_select() {
    // ITE requires a Bool condition — construct via comparison
    let a = Expr::real(1);
    let b = Expr::real(2);
    let cond = a.real_lt(b); // Bool: 1 < 2
    let then_val = Expr::real(10);
    let else_val = Expr::real(20);
    let kind = IRNodeKind::Select {
        cond: NodeId::new(0),
        then_val: NodeId::new(1),
        else_val: NodeId::new(2),
    };
    let (expr, ground, _, _) = translate_one(
        &kind,
        &[cond.clone(), then_val.clone(), else_val.clone()],
        &[],
        &[None, None, None],
        &[],
    )
    .unwrap();
    let expected = Expr::ite(cond, then_val, else_val);
    assert_eq!(format!("{expr}"), format!("{expected}"));
    // Select always returns None ground
    assert_eq!(ground, None);
}

// ── SumReduce ──

#[test]
fn test_translate_sum_reduce_empty() {
    let kind = IRNodeKind::SumReduce { inputs: vec![] };
    let (_, ground, _, _) = translate_one(&kind, &[], &[], &[], &[]).unwrap();
    assert_eq!(ground, Some(0.0));
}

#[test]
fn test_translate_sum_reduce_three_elements() {
    let a = Expr::real(1);
    let b = Expr::real(2);
    let c = Expr::real(3);
    let kind = IRNodeKind::SumReduce {
        inputs: vec![NodeId::new(0), NodeId::new(1), NodeId::new(2)],
    };
    let (_, ground, _, _) = translate_one(
        &kind,
        &[a, b, c],
        &[],
        &[Some(1.0), Some(2.0), Some(3.0)],
        &[],
    )
    .unwrap();
    assert_eq!(ground, Some(6.0));
}

#[test]
fn test_translate_sum_reduce_partial_symbolic() {
    let a = Expr::real(1);
    let b = Expr::real(2);
    let kind = IRNodeKind::SumReduce {
        inputs: vec![NodeId::new(0), NodeId::new(1)],
    };
    let (_, ground, _, _) = translate_one(&kind, &[a, b], &[], &[Some(1.0), None], &[]).unwrap();
    assert_eq!(ground, None, "partial symbolic sum should yield no ground");
}

// ── Powi ──

#[test]
fn test_translate_powi_ground_fold() {
    // powi(3, 2) = 9, should ground-fold
    let base = Expr::real(3);
    let kind = IRNodeKind::Powi {
        base: NodeId::new(0),
        exp: 2,
    };
    let (_, ground, _, _) = translate_one(&kind, &[base], &[], &[Some(3.0)], &[]).unwrap();
    assert_eq!(ground, Some(9.0));
}

#[test]
fn test_translate_powi_zero_exponent() {
    // x^0 = 1 for any x
    let base = Expr::real(7);
    let kind = IRNodeKind::Powi {
        base: NodeId::new(0),
        exp: 0,
    };
    let (_, ground, _, _) = translate_one(&kind, &[base], &[], &[Some(7.0)], &[]).unwrap();
    assert_eq!(ground, Some(1.0));
}

#[test]
fn test_translate_powi_symbolic_small_exp_no_uf() {
    // Small exponents (<=8) use binary exponentiation (direct multiplication),
    // not UF approximation.
    let base = Expr::real(2);
    let kind = IRNodeKind::Powi {
        base: NodeId::new(0),
        exp: 3,
    };
    let (_, ground, _, uses_uf) = translate_one(&kind, &[base], &[], &[None], &[]).unwrap();
    assert_eq!(ground, None, "symbolic base should yield no ground");
    assert!(
        !uses_uf,
        "powi with small exponent should use binary exponentiation, not UF"
    );
}
