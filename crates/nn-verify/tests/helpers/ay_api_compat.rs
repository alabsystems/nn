// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay-bindings API compatibility smoke tests.
//!
//! These tests exercise the exact ay-bindings API surface that nn-verify
//! depends on (#212). If ay renames, removes, or changes the signature of
//! any of these APIs, these tests will fail at compile time or runtime,
//! providing an early warning before deeper verification code breaks.
//!
//! API surface exercised:
//! - `AYProgram`: new, set_logic, declare_const, declare_fun, assert, check_sat, Display
//! - `Expr`: real, var, ite, func_app_with_sort, real_add/sub/mul/div/neg,
//!   real_lt/le/gt/ge, eq, ne, or
//! - `Sort`: real
//! - `execute_direct::execute`, `ExecuteResult` variant matching

use ay_bindings::execute_direct::{self, ExecuteResult};
use ay_bindings::{Expr, Sort, AYProgram};

/// Helper: create a Real-sorted variable expression (matching nn's translate.rs pattern).
fn real_var(name: &str) -> Expr {
    Expr::var(name, Sort::real())
}

/// Verify AYProgram construction and basic method signatures.
#[test]
fn compat_ayprogram_core_methods() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real_sort = Sort::real();
    let _ = prog.declare_const("x", real_sort);

    // assert takes an Expr
    let x = real_var("x");
    let bound = x.real_ge(Expr::real(0));
    prog.assert(bound);

    // check_sat appends (check-sat) to the program
    prog.check_sat();

    // Display trait for SMT-LIB2 serialization
    let smt2 = prog.to_string();
    assert!(
        smt2.contains("declare-const"),
        "SMT-LIB2 output should contain declare-const"
    );
    assert!(
        smt2.contains("check-sat"),
        "SMT-LIB2 output should contain check-sat"
    );
}

/// Verify declare_fun for uninterpreted functions (used by translate_uf.rs).
#[test]
fn compat_declare_fun() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_UFNRA");

    let real_sort = Sort::real();
    prog.declare_fun("sin_approx", vec![real_sort.clone()], real_sort);

    let smt2 = prog.to_string();
    assert!(
        smt2.contains("declare-fun"),
        "SMT-LIB2 output should contain declare-fun"
    );
}

/// Verify all Expr arithmetic methods used by nn-verify.
#[test]
fn compat_expr_real_arithmetic() {
    let x = real_var("x");
    let two = Expr::real(2);
    let one = Expr::real(1);

    // All 5 arithmetic operations
    let _add = x.clone().real_add(one.clone());
    let _sub = x.clone().real_sub(one);
    let _mul = x.clone().real_mul(two.clone());
    let _div = x.clone().real_div(two);
    let _neg = x.real_neg();

    // Expr::real constructor with zero and negative
    let _zero = Expr::real(0);
    let _neg_val = Expr::real(-5);
}

/// Verify all Expr comparison methods used by nn-verify.
#[test]
fn compat_expr_real_comparisons() {
    let x = real_var("x");
    let zero = Expr::real(0);

    // All 4 real comparison operations (prefixed form, not deprecated shorthands)
    let _lt = x.clone().real_lt(zero.clone());
    let _le = x.clone().real_le(zero.clone());
    let _gt = x.clone().real_gt(zero.clone());
    let _ge = x.clone().real_ge(zero.clone());

    // eq and ne (generic, not sort-prefixed)
    let _eq = x.clone().eq(zero.clone());
    let _ne = x.ne(zero);
}

/// Verify Expr::ite (if-then-else) used for clamp, minmax, select, abs.
#[test]
fn compat_expr_ite() {
    let x = real_var("x");
    let zero = Expr::real(0);
    let one = Expr::real(1);

    let cond = x.real_ge(zero);
    let result = Expr::ite(cond, one.clone(), one.real_neg());
    // ite should produce a valid Expr (verify via SMT-LIB2 embedding)
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");
    let _ = prog.declare_const("x", Sort::real());
    prog.assert(result.real_ge(Expr::real(-1)));
    let smt2 = prog.to_string();
    assert!(smt2.contains("ite"), "SMT-LIB2 should contain ite");
}

/// Verify Expr::or (boolean combinator) used in prove.rs bounds check.
#[test]
fn compat_expr_boolean_or() {
    let x = real_var("x");
    let zero = Expr::real(0);
    let one = Expr::real(1);

    let below = x.clone().real_lt(zero);
    let above = x.real_gt(one);
    let _violation = below.or(above);
}

/// Verify func_app_with_sort for UF translation (sin, cos, exp, sqrt, powi).
#[test]
fn compat_func_app_with_sort() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_UFNRA");
    let real_sort = Sort::real();
    let _ = prog.declare_const("x", real_sort.clone());
    prog.declare_fun("sin_approx", vec![real_sort.clone()], real_sort.clone());

    let x = real_var("x");
    let result = Expr::func_app_with_sort("sin_approx", vec![x], real_sort);

    prog.assert(result.real_ge(Expr::real(-1)));
    let smt2 = prog.to_string();
    assert!(
        smt2.contains("sin_approx"),
        "SMT-LIB2 should contain UF application"
    );
}

/// Verify execute_direct::execute and all ExecuteResult variants nn matches on.
#[test]
fn compat_execute_direct_verified() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");
    let _ = prog.declare_const("x", Sort::real());

    // x >= 0 AND x <= 10 AND x < -1 (contradictory — should be UNSAT)
    let x = real_var("x");
    prog.assert(x.clone().real_ge(Expr::real(0)));
    prog.assert(x.clone().real_le(Expr::real(10)));
    prog.assert(x.real_lt(Expr::real(-1)));

    // ay#5357 fix landed (6aac039): pure integer-coefficient QF_LRA should reach Verified.
    match execute_direct::execute(&prog) {
        Ok(ExecuteResult::Verified) => {
            // Expected: UNSAT because x >= 0 contradicts x < -1
        }
        Ok(other) => {
            panic!("ay#5357 fixed: expected Verified, got: {other:?}");
        }
        Err(e) => {
            panic!("unexpected execution error: {e}");
        }
    }
}

/// Verify execute_direct with a SAT (satisfiable) case.
#[test]
fn compat_execute_direct_sat() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");
    let _ = prog.declare_const("x", Sort::real());

    // x >= 0 AND x <= 10 AND x > 5 — SAT (any x in (5, 10])
    let x = real_var("x");
    prog.assert(x.clone().real_ge(Expr::real(0)));
    prog.assert(x.clone().real_le(Expr::real(10)));
    prog.assert(x.real_gt(Expr::real(5)));

    match execute_direct::execute(&prog) {
        Ok(ExecuteResult::Counterexample { .. }) => {
            // Expected: SAT problem should produce a counterexample (model exists)
        }
        Ok(ExecuteResult::Unknown(_)) => {
            // Acceptable: solver incomplete for this instance
        }
        Ok(ExecuteResult::Verified) => {
            // Verified means UNSAT — accepting it for a SAT problem would mask a solver bug.
            panic!(
                "SAT constraint set (x >= 0 AND x <= 10 AND x > 5) returned Verified (UNSAT). \
                 This is semantically wrong — a model exists (e.g., x=6)."
            );
        }
        Ok(ExecuteResult::NeedsFallback(_)) => {
            panic!("QF_LRA should not need fallback");
        }
        Ok(other) => {
            panic!("unexpected ExecuteResult variant: {other:?}");
        }
        Err(e) => {
            panic!("unexpected execution error: {e}");
        }
    }
}

/// Verify Sort::real() is the only Sort constructor nn needs.
#[test]
fn compat_sort_real() {
    let sort = Sort::real();
    // Sort must be Clone (used with .clone() in translate.rs and translate_uf.rs).
    // The clone is the test: verifying the trait bound exists.
    #[allow(clippy::redundant_clone)]
    let _cloned = sort.clone();
}
