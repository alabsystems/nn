// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `dispatch_query` edge cases: ExecutionFailed, heuristic bounds
//! gate, UF direct execution (#2617), non-linear Unexecuted paths.
//!
//! These test the dispatch logic in `prove_exec.rs` directly by constructing
//! `SmtQuery` values, rather than going through `verify_kernel_smt` which
//! exercises the full pipeline.
//!
//! Coverage gaps addressed: #165 (ExecutionFailed path, NeedsFallback arm).

use super::*;
use ay_bindings::{Expr, Sort, AYProgram};

/// Build a minimal well-formed SmtQuery with the given overrides.
/// The program is a trivial QF_LRA program that the solver can handle.
fn trivial_query() -> SmtQuery {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");
    let x = program.declare_const("x", Sort::real());
    // x >= -10 AND x <= 10
    program.assert(x.clone().real_ge(Expr::real(-10)));
    program.assert(x.real_le(Expr::real(10)));
    program.check_sat();

    SmtQuery {
        program,
        smt2: "(set-logic QF_LRA)\n(check-sat)\n".to_string(),
        encoding: SmtEncodingKind::Exact,
        uses_nonlinear: false,
        uses_heuristic_bounds: false,
        bounds_source: BoundsSource::Analytical,
        expected_bounds: (-10.0, 10.0),
    }
}

/// Build a query whose program contains a soft assertion, which triggers
/// `NeedsFallback` in `execute_direct`. This exercises the `Err` path in
/// `dispatch_query` that produces `SmtOutcome::ExecutionFailed`.
fn fallback_triggering_query() -> SmtQuery {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");
    let x = program.declare_const("x", Sort::real());
    // The soft_assert triggers NeedsFallback in ay's direct execution.
    program.soft_assert(x.real_ge(Expr::real(0)), 1);
    program.check_sat();

    SmtQuery {
        program,
        smt2: "(set-logic QF_LRA)\n(assert-soft (>= x 0))\n(check-sat)\n".to_string(),
        encoding: SmtEncodingKind::Exact,
        uses_nonlinear: false,
        uses_heuristic_bounds: false,
        bounds_source: BoundsSource::Analytical,
        expected_bounds: (-10.0, 10.0),
    }
}

// ======================== Heuristic bounds gate ========================

#[test]
fn test_dispatch_heuristic_bounds_returns_unexecuted() {
    let mut query = trivial_query();
    query.uses_heuristic_bounds = true;

    let result = dispatch_query(query);
    assert_eq!(result.outcome, SmtOutcome::Unexecuted);
    let detail = result.detail.as_deref().unwrap_or("");
    assert!(
        detail.contains("heuristic"),
        "heuristic gate should explain reason, got: {detail}"
    );
    assert_eq!(result.bounds_source, BoundsSource::Analytical);
}

// ======================== UF encoding now uses direct execution (#2617) ========================

#[test]
fn test_dispatch_uf_encoding_attempts_direct_execution() {
    // With #2617, UfApprox linear encodings attempt direct execution with QF_UFLRA.
    // The trivial_query program has no UF declarations but is valid under UFLRA.
    // The solver returns SAT (constraints are satisfiable → Counterexample).
    let mut query = trivial_query();
    query.encoding = SmtEncodingKind::UfApprox;

    let result = dispatch_query(query);
    assert_ne!(
        result.outcome,
        SmtOutcome::Unexecuted,
        "UfApprox should no longer short-circuit to Unexecuted (#2617), got: {:?}",
        result.detail,
    );
    assert_eq!(
        result.solver, "ay-direct",
        "UfApprox should use ay-direct execution"
    );
}

// ======================== Non-linear NRA execution (#2640) ========================

#[test]
fn test_dispatch_nonlinear_reaches_solver() {
    // #2640: nonlinear queries now route to ay NRA solver via ALL logic
    // auto-detection instead of being gated as Unexecuted.
    let mut query = trivial_query();
    query.uses_nonlinear = true;

    let result = dispatch_query(query);
    assert_ne!(
        result.outcome,
        SmtOutcome::Unexecuted,
        "nonlinear query should reach solver (#2640), got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
    assert_eq!(
        result.solver, "ay-direct",
        "nonlinear query should use ay-direct with ALL logic"
    );
}

// ======================== ExecutionFailed path ========================

#[test]
fn test_dispatch_direct_execution_failure_returns_execution_failed() {
    // soft_assert triggers NeedsFallback in ay direct execution, which
    // causes try_direct_execution to return Err(SmtError::SolverError(...)).
    // dispatch_query then catches this and produces ExecutionFailed.
    let query = fallback_triggering_query();

    let result = dispatch_query(query);
    assert_eq!(
        result.outcome,
        SmtOutcome::ExecutionFailed,
        "NeedsFallback from solver should map to ExecutionFailed, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
    let detail = result.detail.as_deref().unwrap_or("");
    assert!(
        detail.contains("direct execution failed"),
        "detail should mention failure, got: {detail}"
    );
    assert!(
        detail.contains("needs fallback") || detail.contains("soft assertion"),
        "detail should include reason from solver, got: {detail}"
    );
    assert_eq!(result.solver, "ay");
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.bounds_source, BoundsSource::Analytical);
}

// ======================== Expected bounds preservation ========================

#[test]
fn test_dispatch_preserves_expected_bounds_on_all_paths() {
    let expected = (-42.5, 99.1);

    // Path 1: heuristic gate
    let mut q1 = trivial_query();
    q1.uses_heuristic_bounds = true;
    q1.expected_bounds = expected;
    let r1 = dispatch_query(q1);
    assert_eq!(r1.expected_bounds, Some(expected), "heuristic path");

    // Path 2: UF direct execution (#2617 — no longer Unexecuted)
    let mut q2 = trivial_query();
    q2.encoding = SmtEncodingKind::UfApprox;
    q2.expected_bounds = expected;
    let r2 = dispatch_query(q2);
    assert_eq!(r2.expected_bounds, Some(expected), "UF direct path");

    // Path 3: nonlinear NRA (#2640 — now reaches solver)
    let mut q3 = trivial_query();
    q3.uses_nonlinear = true;
    q3.expected_bounds = expected;
    let r3 = dispatch_query(q3);
    assert_eq!(r3.expected_bounds, Some(expected), "nonlinear path");

    // Path 4: ExecutionFailed
    let mut q4 = fallback_triggering_query();
    q4.expected_bounds = expected;
    let r4 = dispatch_query(q4);
    assert_eq!(r4.expected_bounds, Some(expected), "ExecutionFailed path");
}
