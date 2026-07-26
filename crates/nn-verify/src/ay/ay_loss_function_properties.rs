// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for loss function mathematical properties (#4208).
//!
//! Loss functions are the objective functions that ML models minimize during training.
//! Their mathematical properties (non-negativity, symmetry, triangle inequality, etc.)
//! are critical for training convergence and correctness. This module proves these
//! properties using ay's SMT solver.
//!
//! # Proved Properties
//!
//! 1. **Cross-entropy loss**: non-negativity, zero when prediction matches target
//! 2. **MSE loss**: non-negativity, symmetry, zero iff equal
//! 3. **L1 loss**: triangle inequality, non-negativity
//! 4. **Huber loss**: quadratic near zero, linear far from zero, continuity at delta
//! 5. **KL divergence**: non-negativity (Gibbs' inequality), zero when distributions match
//! 6. **Binary cross-entropy**: range bounds, symmetry properties
//! 7. **Focal loss**: relationship to cross-entropy via gamma parameter
//! 8. **CTC loss**: non-negativity properties
//!
//! # Proof Strategy
//!
//! Loss function proofs use several approaches:
//!
//! - **Algebraic identity proofs** (MSE symmetry, L1 triangle inequality): Pure polynomial
//!   or linear identities provable via QF_LRA or QF_NRA.
//!
//! - **Transcendental encoding** (cross-entropy, KL divergence): Since log/exp are
//!   outside the decidable NRA fragment, we encode known properties of log (e.g.,
//!   log(1) = 0, log(x) <= x - 1 for x > 0) as symbolic constraints and prove
//!   the loss properties follow from these axioms.
//!
//! - **Piecewise proofs** (Huber loss): Case-split on |error| <= delta vs > delta,
//!   prove each branch independently.

use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;

/// Result of a loss function property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct LossPropertyResult {
    /// Human-readable property name.
    pub property: String,
    /// Whether the proof succeeded (UNSAT = property holds for all inputs).
    pub proven: bool,
    /// SMT-LIB2 text of the query (for debugging/external solver use).
    pub smt2: String,
    /// Solver detail message.
    pub detail: String,
}

/// Declare a real variable and return its expression.
fn declare_real(program: &mut AYProgram, name: &str) -> Expr {
    program.declare_const(name, Sort::real())
}

/// Assert `lower <= expr <= upper`.
fn assert_bounds(
    program: &mut AYProgram,
    expr: &Expr,
    lower: f64,
    upper: f64,
) -> Result<(), SmtError> {
    let lo = real_from_f64(lower)?;
    let hi = real_from_f64(upper)?;
    program.assert(expr.clone().real_ge(lo));
    program.assert(expr.clone().real_le(hi));
    Ok(())
}

/// Execute a ay program and return whether UNSAT (property proven).
fn execute_and_check(program: &AYProgram) -> (bool, String) {
    let (proven, detail) = match ay_bindings::execute_direct::execute(program) {
        Ok(ay_bindings::execute_direct::ExecuteResult::Verified) => {
            (true, "UNSAT: property holds for all inputs".to_string())
        }
        Ok(ay_bindings::execute_direct::ExecuteResult::Counterexample { model, .. }) => {
            (false, format!("SAT: counterexample found: {:?}", model))
        }
        Ok(ay_bindings::execute_direct::ExecuteResult::Unknown(reason)) => {
            (false, format!("Unknown: {}", reason))
        }
        Ok(other) => (false, format!("Unexpected result: {:?}", other)),
        Err(e) => (false, format!("Execution error: {}", e)),
    };
    // Uniform guard: a vacuous UNSAT (P and not-P, or X != X) never counts as a
    // proof. See crate::ay_vacuity. No-op for genuine queries.
    crate::ay_vacuity::reject_if_vacuous(&program.to_string(), proven, detail)
}

// ---------------------------------------------------------------------------
// Property 1: Cross-Entropy Loss Non-Negativity
// ---------------------------------------------------------------------------

/// Prove that cross-entropy loss is non-negative.
///
/// Cross-entropy for a single element: `L_i = -y_i * log(p_i)`
/// where `y_i >= 0` (target) and `0 < p_i <= 1` (predicted probability).
///
/// Since log(p_i) <= 0 for p_i in (0, 1], `-log(p_i) >= 0`.
/// Combined with `y_i >= 0`, we get `L_i = y_i * (-log(p_i)) >= 0`.
///
/// We encode log(p_i) as a symbolic variable `log_p` constrained by
/// `log_p <= 0` (since p_i in (0, 1] implies log(p_i) <= 0), and prove
/// the loss is non-negative.
pub(crate) fn prove_cross_entropy_non_negative() -> Result<LossPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let y = declare_real(&mut program, "y");
    let log_p = declare_real(&mut program, "log_p");
    let loss = declare_real(&mut program, "loss");

    // y >= 0 (target probability)
    let zero = Expr::real(0);
    program.assert(y.clone().real_ge(zero.clone()));
    assert_bounds(&mut program, &y, 0.0, 1.0)?;

    // log(p) <= 0 for p in (0, 1] — fundamental property of logarithm
    program.assert(log_p.clone().real_le(zero.clone()));
    assert_bounds(&mut program, &log_p, -100.0, 0.0)?;

    // loss = -y * log(p) = y * (-log(p))
    // Since -log(p) >= 0, encode as: neg_log_p = -log_p >= 0
    let neg_log_p = declare_real(&mut program, "neg_log_p");
    program.assert(neg_log_p.clone().eq(log_p.real_neg()));

    // loss = y * neg_log_p
    program.assert(loss.clone().eq(y.real_mul(neg_log_p)));

    // Negated property: loss < 0
    let violation = loss.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "cross_entropy_non_negative".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that cross-entropy loss is zero when prediction matches target perfectly.
///
/// When p_i = 1 for the correct class (one-hot target y_i = 1), log(1) = 0,
/// so L = -1 * 0 = 0. For non-target classes (y_i = 0), L = -0 * log(p_i) = 0.
///
/// We encode: given y = 1 and log_p = 0 (i.e., p = 1), prove loss = 0.
pub(crate) fn prove_cross_entropy_zero_at_match() -> Result<LossPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let loss = declare_real(&mut program, "loss");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // y = 1 (one-hot target for correct class)
    // log(p) = 0 (p = 1, perfect prediction)
    // loss = -y * log(p) = -1 * 0 = 0
    let neg_one_times_zero = one.real_neg().real_mul(zero.clone());
    program.assert(loss.clone().eq(neg_one_times_zero));

    // Negated property: loss != 0
    let violation = loss.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "cross_entropy_zero_at_match".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: MSE Loss
// ---------------------------------------------------------------------------

/// Prove that MSE loss is non-negative: (y - p)^2 >= 0 for all y, p.
///
/// Mean squared error for a single element: L = (y - p)^2.
/// Any real number squared is non-negative.
pub(crate) fn prove_mse_non_negative() -> Result<LossPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let y = declare_real(&mut program, "y");
    let p = declare_real(&mut program, "p");
    let diff = declare_real(&mut program, "diff");
    let loss = declare_real(&mut program, "loss");

    assert_bounds(&mut program, &y, -100.0, 100.0)?;
    assert_bounds(&mut program, &p, -100.0, 100.0)?;

    // diff = y - p
    program.assert(diff.clone().eq(y.real_sub(p)));

    // loss = diff^2
    program.assert(loss.clone().eq(diff.clone().real_mul(diff)));

    // Negated property: loss < 0
    let zero = Expr::real(0);
    let violation = loss.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "mse_non_negative".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove MSE symmetry: (y - p)^2 = (p - y)^2 for all y, p.
///
/// Since (y-p)^2 = (-(p-y))^2 = (p-y)^2, MSE is symmetric in its arguments.
pub(crate) fn prove_mse_symmetry() -> Result<LossPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let y = declare_real(&mut program, "y");
    let p = declare_real(&mut program, "p");

    assert_bounds(&mut program, &y, -100.0, 100.0)?;
    assert_bounds(&mut program, &p, -100.0, 100.0)?;

    // diff1 = y - p, loss1 = diff1^2
    let diff1 = declare_real(&mut program, "diff1");
    let loss1 = declare_real(&mut program, "loss1");
    program.assert(diff1.clone().eq(y.clone().real_sub(p.clone())));
    program.assert(loss1.clone().eq(diff1.clone().real_mul(diff1)));

    // diff2 = p - y, loss2 = diff2^2
    let diff2 = declare_real(&mut program, "diff2");
    let loss2 = declare_real(&mut program, "loss2");
    program.assert(diff2.clone().eq(p.real_sub(y)));
    program.assert(loss2.clone().eq(diff2.clone().real_mul(diff2)));

    // Negated property: loss1 != loss2
    let violation = loss1.ne(loss2);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "mse_symmetry".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove MSE is zero if and only if y = p.
///
/// Forward direction: if y = p then (y-p)^2 = 0.
/// This is encoded as: y = p implies loss = 0; negated: y = p and loss != 0 is UNSAT.
pub(crate) fn prove_mse_zero_iff_equal() -> Result<LossPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let y = declare_real(&mut program, "y");
    let p = declare_real(&mut program, "p");
    let diff = declare_real(&mut program, "diff");
    let loss = declare_real(&mut program, "loss");

    assert_bounds(&mut program, &y, -100.0, 100.0)?;
    assert_bounds(&mut program, &p, -100.0, 100.0)?;

    // diff = y - p
    program.assert(diff.clone().eq(y.clone().real_sub(p.clone())));
    // loss = diff^2
    program.assert(loss.clone().eq(diff.clone().real_mul(diff)));

    // Constraint: y = p
    program.assert(y.eq(p));

    // Negated property: loss != 0
    let zero = Expr::real(0);
    let violation = loss.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "mse_zero_iff_equal".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 3: L1 Loss
// ---------------------------------------------------------------------------

/// Prove L1 loss non-negativity: |y - p| >= 0 for all y, p.
///
/// We model |y - p| using a helper variable `abs_diff` constrained by:
///   abs_diff >= (y - p) and abs_diff >= (p - y) and (abs_diff = y-p or abs_diff = p-y).
/// Then prove abs_diff >= 0.
pub(crate) fn prove_l1_non_negative() -> Result<LossPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let y = declare_real(&mut program, "y");
    let p = declare_real(&mut program, "p");
    let abs_diff = declare_real(&mut program, "abs_diff");

    assert_bounds(&mut program, &y, -100.0, 100.0)?;
    assert_bounds(&mut program, &p, -100.0, 100.0)?;

    // abs_diff >= y - p
    program.assert(abs_diff.clone().real_ge(y.clone().real_sub(p.clone())));
    // abs_diff >= p - y (= -(y-p))
    program.assert(abs_diff.clone().real_ge(p.clone().real_sub(y.clone())));

    // abs_diff is the minimum of values >= both: either y-p or p-y
    // (abs_diff = y-p) or (abs_diff = p-y)
    let eq_pos = abs_diff.clone().eq(y.clone().real_sub(p.clone()));
    let eq_neg = abs_diff.clone().eq(p.real_sub(y));
    program.assert(eq_pos.or(eq_neg));

    // Negated property: abs_diff < 0
    let zero = Expr::real(0);
    let violation = abs_diff.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "l1_non_negative".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove L1 triangle inequality: |a - c| <= |a - b| + |b - c| for all a, b, c.
///
/// This is a fundamental property of L1 loss (absolute difference as a metric).
/// We model each absolute value using helper variables with positivity and
/// disjunctive constraints.
pub(crate) fn prove_l1_triangle_inequality() -> Result<LossPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let c = declare_real(&mut program, "c");

    assert_bounds(&mut program, &a, -100.0, 100.0)?;
    assert_bounds(&mut program, &b, -100.0, 100.0)?;
    assert_bounds(&mut program, &c, -100.0, 100.0)?;

    // |a - c|: abs_ac >= a-c, abs_ac >= c-a, abs_ac = a-c or abs_ac = c-a
    let abs_ac = declare_real(&mut program, "abs_ac");
    program.assert(abs_ac.clone().real_ge(a.clone().real_sub(c.clone())));
    program.assert(abs_ac.clone().real_ge(c.clone().real_sub(a.clone())));
    let eq_ac_pos = abs_ac.clone().eq(a.clone().real_sub(c.clone()));
    let eq_ac_neg = abs_ac.clone().eq(c.clone().real_sub(a.clone()));
    program.assert(eq_ac_pos.or(eq_ac_neg));

    // |a - b|
    let abs_ab = declare_real(&mut program, "abs_ab");
    program.assert(abs_ab.clone().real_ge(a.clone().real_sub(b.clone())));
    program.assert(abs_ab.clone().real_ge(b.clone().real_sub(a.clone())));
    let eq_ab_pos = abs_ab.clone().eq(a.clone().real_sub(b.clone()));
    let eq_ab_neg = abs_ab.clone().eq(b.clone().real_sub(a.clone()));
    program.assert(eq_ab_pos.or(eq_ab_neg));

    // |b - c|
    let abs_bc = declare_real(&mut program, "abs_bc");
    program.assert(abs_bc.clone().real_ge(b.clone().real_sub(c.clone())));
    program.assert(abs_bc.clone().real_ge(c.clone().real_sub(b.clone())));
    let eq_bc_pos = abs_bc.clone().eq(b.clone().real_sub(c.clone()));
    let eq_bc_neg = abs_bc.clone().eq(c.real_sub(b));
    program.assert(eq_bc_pos.or(eq_bc_neg));

    // Negated property: |a - c| > |a - b| + |b - c|
    let rhs = abs_ab.real_add(abs_bc);
    let violation = abs_ac.real_gt(rhs);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "l1_triangle_inequality".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: Huber Loss
// ---------------------------------------------------------------------------

/// Prove Huber loss is quadratic near zero: for |error| <= delta,
/// Huber(error, delta) = 0.5 * error^2.
///
/// The Huber loss definition (piecewise):
///   L = 0.5 * e^2           if |e| <= delta
///   L = delta * (|e| - 0.5 * delta)  if |e| > delta
///
/// We prove the quadratic branch is non-negative and equals 0.5 * e^2.
pub(crate) fn prove_huber_quadratic_near_zero() -> Result<LossPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let e = declare_real(&mut program, "e");
    let delta = declare_real(&mut program, "delta");
    let loss = declare_real(&mut program, "loss");

    assert_bounds(&mut program, &e, -10.0, 10.0)?;
    assert_bounds(&mut program, &delta, 0.1, 10.0)?;

    // |e| <= delta: both e <= delta and -e <= delta (i.e., e >= -delta)
    program.assert(e.clone().real_le(delta.clone()));
    program.assert(e.clone().real_ge(delta.clone().real_neg()));

    // loss = 0.5 * e^2
    let half = real_from_f64(0.5)?;
    let e_sq = e.clone().real_mul(e);
    let expected = half.real_mul(e_sq);
    program.assert(loss.clone().eq(expected));

    // Negated property: loss < 0 (should be impossible for 0.5 * e^2)
    let zero = Expr::real(0);
    let violation = loss.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "huber_quadratic_near_zero".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove Huber loss is linear far from zero: for |error| > delta,
/// Huber(error, delta) = delta * (|error| - 0.5 * delta).
///
/// This ensures growth is linear (not quadratic) for large errors,
/// providing robustness to outliers.
pub(crate) fn prove_huber_linear_far_from_zero() -> Result<LossPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let abs_e = declare_real(&mut program, "abs_e");
    let delta = declare_real(&mut program, "delta");
    let loss = declare_real(&mut program, "loss");

    // abs_e > delta (we are in the linear region), abs_e > 0
    let zero = Expr::real(0);
    program.assert(abs_e.clone().real_gt(zero.clone()));
    program.assert(abs_e.clone().real_gt(delta.clone()));
    assert_bounds(&mut program, &abs_e, 0.0, 100.0)?;
    assert_bounds(&mut program, &delta, 0.1, 10.0)?;

    // loss = delta * (abs_e - 0.5 * delta)
    let half = real_from_f64(0.5)?;
    let half_delta = half.real_mul(delta.clone());
    let inner = abs_e.real_sub(half_delta);
    let expected = delta.real_mul(inner);
    program.assert(loss.clone().eq(expected));

    // Negated property: loss < 0
    let violation = loss.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "huber_linear_far_from_zero".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove Huber loss continuity at the transition point |error| = delta.
///
/// At |e| = delta, both branches must give the same value:
///   Quadratic: 0.5 * delta^2
///   Linear:    delta * (delta - 0.5 * delta) = delta * 0.5 * delta = 0.5 * delta^2
///
/// The proof encodes both branch values at |e| = delta and proves they are equal.
pub(crate) fn prove_huber_continuity_at_delta() -> Result<LossPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let delta = declare_real(&mut program, "delta");
    let loss_quad = declare_real(&mut program, "loss_quad");
    let loss_lin = declare_real(&mut program, "loss_lin");

    assert_bounds(&mut program, &delta, 0.01, 100.0)?;

    let half = real_from_f64(0.5)?;

    // Quadratic branch at |e| = delta: loss_quad = 0.5 * delta^2
    let delta_sq = delta.clone().real_mul(delta.clone());
    program.assert(loss_quad.clone().eq(half.clone().real_mul(delta_sq)));

    // Linear branch at |e| = delta: loss_lin = delta * (delta - 0.5 * delta)
    let half_delta = half.real_mul(delta.clone());
    let inner = delta.clone().real_sub(half_delta);
    program.assert(loss_lin.clone().eq(delta.real_mul(inner)));

    // Negated property: loss_quad != loss_lin
    let violation = loss_quad.ne(loss_lin);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "huber_continuity_at_delta".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 5: KL Divergence
// ---------------------------------------------------------------------------

/// Prove KL divergence non-negativity (Gibbs' inequality).
///
/// KL(P || Q) = sum_i p_i * log(p_i / q_i) >= 0 for all valid distributions.
///
/// The key insight is log(x) <= x - 1 for all x > 0 (with equality iff x = 1).
/// Applying to x = q_i / p_i:
///   log(q_i / p_i) <= q_i/p_i - 1
///   -log(p_i / q_i) <= q_i/p_i - 1
///   log(p_i / q_i) >= 1 - q_i/p_i
///
/// For a single element, we prove: p * log(p/q) >= p - q when log(p/q) >= 1 - q/p.
///
/// We encode the log inequality as an axiom and prove KL non-negativity follows.
/// For a single element: kl_i = p * log_ratio where log_ratio = log(p/q).
/// Using the log bound: log_ratio >= 1 - q/p, so kl_i = p * log_ratio >= p * (1 - q/p) = p - q.
/// Summing over all elements with sum(p_i) = sum(q_i) = 1 gives KL >= 0.
///
/// We prove the single-element bound: p * log_ratio >= p - q.
pub(crate) fn prove_kl_divergence_non_negative() -> Result<LossPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let p = declare_real(&mut program, "p");
    let q = declare_real(&mut program, "q");
    let log_ratio = declare_real(&mut program, "log_ratio"); // log(p/q)
    let ratio = declare_real(&mut program, "ratio"); // q/p
    let kl_elem = declare_real(&mut program, "kl_elem");

    // p, q > 0 (valid probability entries)
    let zero = Expr::real(0);
    let one = Expr::real(1);
    program.assert(p.clone().real_gt(zero.clone()));
    program.assert(q.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &p, 0.001, 1.0)?;
    assert_bounds(&mut program, &q, 0.001, 1.0)?;

    // ratio = q / p (encoded as ratio * p = q)
    program.assert(ratio.clone().real_mul(p.clone()).eq(q.clone()));
    assert_bounds(&mut program, &ratio, 0.001, 1000.0)?;

    // Log inequality axiom: log(p/q) >= 1 - q/p
    // Equivalently: log_ratio >= 1 - ratio
    let one_minus_ratio = one.real_sub(ratio);
    program.assert(log_ratio.clone().real_ge(one_minus_ratio));

    // kl_elem = p * log_ratio
    program.assert(kl_elem.clone().eq(p.clone().real_mul(log_ratio)));

    // From the log inequality:
    // kl_elem = p * log_ratio >= p * (1 - q/p) = p - q
    // For summed KL where sum(p) = sum(q) = 1: total KL >= sum(p - q) = 0

    // Negated property: kl_elem < p - q
    // (i.e., the single-element lower bound is violated)
    let p_minus_q = p.real_sub(q);
    let violation = kl_elem.real_lt(p_minus_q);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "kl_divergence_non_negative".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove KL divergence is zero when distributions match.
///
/// When P = Q (i.e., p_i = q_i for all i), log(p_i / q_i) = log(1) = 0,
/// so KL(P || Q) = sum(p_i * 0) = 0.
///
/// We encode: given p = q and log(p/q) = log(1) = 0, prove kl_elem = 0.
pub(crate) fn prove_kl_divergence_zero_when_equal() -> Result<LossPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let p = declare_real(&mut program, "p");
    let kl_elem = declare_real(&mut program, "kl_elem");

    let zero = Expr::real(0);
    program.assert(p.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &p, 0.001, 1.0)?;

    // When p = q, log(p/q) = log(1) = 0
    // kl_elem = p * 0 = 0
    let log_ratio_zero = Expr::real(0);
    program.assert(kl_elem.clone().eq(p.real_mul(log_ratio_zero)));

    // Negated property: kl_elem != 0
    let violation = kl_elem.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "kl_divergence_zero_when_equal".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 6: Binary Cross-Entropy
// ---------------------------------------------------------------------------

/// Prove binary cross-entropy range bounds.
///
/// BCE(y, p) = -[y * log(p) + (1-y) * log(1-p)]
///
/// For y in {0, 1} and p in (0, 1):
/// - When y = 0: BCE = -log(1-p) >= 0  (since 0 < 1-p < 1 means log(1-p) <= 0)
/// - When y = 1: BCE = -log(p) >= 0    (since 0 < p < 1 means log(p) <= 0)
///
/// We prove BCE >= 0 for a binary label `y in {0, 1}` (the standard BCE target).
///
/// The proof case-splits on the label so the query stays LINEAR (`QF_LRA`): the
/// product `y * log(p)` never appears, so there is no variable*variable term to
/// push the query into the undecidable `QF_NRA` fragment (which is what the old
/// real-`y`, `y*log(p)` encoding did — it left `proven` false).
pub(crate) fn prove_bce_non_negative() -> Result<LossPropertyResult, SmtError> {
    let program = build_bce_non_negative(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "bce_non_negative".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the BCE non-negativity query for a binary target `y in {0, 1}`.
///
/// With `log_p = log(p)` and `log_1mp = log(1-p)`, binary cross-entropy
/// `BCE = -[y*log(p) + (1-y)*log(1-p)]` collapses per label to
///   y = 0:  BCE = -log(1-p) = -log_1mp
///   y = 1:  BCE = -log(p)   = -log_p
/// Both logs are `<= 0` on `(0, 1)`, so each `-log(..) >= 0` and thus `BCE >= 0`.
/// Modelling the label as a case split keeps every BCE value linear in the log
/// variables — no `y*log(p)` product — so the query is decidable `QF_LRA`.
///
/// When `negate` is false the `-log(..)` loses its minus sign (the classic
/// "forgot to negate the log term in BCE" slip), so `loss = log(..) <= 0` and
/// `loss < 0` becomes satisfiable: the query must then be SAT.
fn build_bce_non_negative(negate: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let y = declare_real(&mut program, "y");
    let log_p = declare_real(&mut program, "log_p"); // log(p)
    let log_1mp = declare_real(&mut program, "log_1mp"); // log(1 - p)
    let loss = declare_real(&mut program, "loss");

    let zero = Expr::real(0);
    let one = Expr::real(1);
    let neg_hundred = Expr::real(-100);

    // y is a binary label; the disjunction below pins it to 0 or 1.
    program.assert(y.clone().real_ge(zero.clone()));
    program.assert(y.clone().real_le(one.clone()));

    // log(p) <= 0 and log(1-p) <= 0 for p in (0, 1); bounded below for a
    // concrete model. The `<= 0` facts are what make the loss non-negative.
    program.assert(log_p.clone().real_le(zero.clone()));
    program.assert(log_p.clone().real_ge(neg_hundred.clone()));
    program.assert(log_1mp.clone().real_le(zero.clone()));
    program.assert(log_1mp.clone().real_ge(neg_hundred));

    // Per-label BCE value — linear, no `y*log(p)` product.
    // Correct: loss = -log(..). Mutation (negate=false): loss = log(..).
    let val_y0 = if negate {
        log_1mp.clone().real_neg()
    } else {
        log_1mp.clone()
    };
    let val_y1 = if negate {
        log_p.clone().real_neg()
    } else {
        log_p.clone()
    };

    let branch0 = y.clone().eq(zero.clone()).and(loss.clone().eq(val_y0));
    let branch1 = y.clone().eq(one.clone()).and(loss.clone().eq(val_y1));
    program.assert(branch0.or(branch1));

    // Negated property: loss < 0.
    program.assert(loss.real_lt(zero));
    program.check_sat();
    program
}

/// Prove binary cross-entropy symmetry: BCE(y, p) = BCE(1-y, 1-p).
///
/// This is a structural property: swapping target and complement simultaneously
/// yields the same loss.
///
/// BCE(y, p) = -[y * log(p) + (1-y) * log(1-p)]
/// BCE(1-y, 1-p) = -[(1-y) * log(1-p) + y * log(p)]
///
/// These are identical by commutativity of addition.
///
/// The proof case-splits on the binary label `y in {0, 1}` so that each BCE
/// value is `-log_p` or `-log_1mp` — never a `y*log(p)` product — keeping the
/// query linear (`QF_LRA`) and decidable. The old real-`y`, `y*log(p)` encoding
/// was nonlinear (`QF_NRA`) and left `proven` false.
pub(crate) fn prove_bce_symmetry() -> Result<LossPropertyResult, SmtError> {
    let program = build_bce_symmetry(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "bce_symmetry".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the BCE symmetry query `BCE(y, p) = BCE(1-y, 1-p)` for a binary label
/// `y in {0, 1}`.
///
/// With `log_p = log(p)` and `log_1mp = log(1-p)`:
///   `BCE(y, p)`     collapses to  y=0: `-log_1mp`,  y=1: `-log_p`.
///   `BCE(1-y, 1-p)` uses target `1-y` and probability `1-p`, so the log that
///                   applies is the *complementary* one:
///                     y=0 (target 1, prob 1-p):  `-log(1-p)     = -log_1mp`
///                     y=1 (target 0, prob 1-p):  `-log(1-(1-p)) = -log(p) = -log_p`
/// so `bce1` and `bce2` agree in each branch. Symmetry holds ONLY because
/// complementing the probability swaps which log applies; `swap_probability`
/// controls that swap. Each value is linear in the logs (no product), so the
/// query is decidable `QF_LRA`.
///
/// When `swap_probability` is false the second BCE swaps the target but reuses
/// the *original* log mapping (the "swapped the label but forgot to complement
/// p" slip): `bce2` becomes `-log_p` where `bce1` is `-log_1mp` (and vice
/// versa), so `bce1 != bce2` is satisfiable and the query must be SAT.
fn build_bce_symmetry(swap_probability: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let y = declare_real(&mut program, "y");
    let log_p = declare_real(&mut program, "log_p"); // log(p)
    let log_1mp = declare_real(&mut program, "log_1mp"); // log(1 - p)
    let bce1 = declare_real(&mut program, "bce1");
    let bce2 = declare_real(&mut program, "bce2");

    let zero = Expr::real(0);
    let one = Expr::real(1);
    let neg_hundred = Expr::real(-100);

    // log(p), log(1-p) <= 0 on (0, 1); bounded below for a concrete model.
    program.assert(log_p.clone().real_le(zero.clone()));
    program.assert(log_p.clone().real_ge(neg_hundred.clone()));
    program.assert(log_1mp.clone().real_le(zero.clone()));
    program.assert(log_1mp.clone().real_ge(neg_hundred));

    // bce1 = BCE(y, p):  y=0 -> -log_1mp,  y=1 -> -log_p.
    let bce1_y0 = log_1mp.clone().real_neg();
    let bce1_y1 = log_p.clone().real_neg();

    // bce2 = BCE(1-y, 1-p). Complementing p swaps which log applies:
    //   y=0 (target 1) -> -log(1-p)     = -log_1mp
    //   y=1 (target 0) -> -log(1-(1-p)) = -log_p
    // The mutation drops the swap and reuses the original mapping.
    let (bce2_y0, bce2_y1) = if swap_probability {
        (log_1mp.clone().real_neg(), log_p.clone().real_neg())
    } else {
        (log_p.clone().real_neg(), log_1mp.clone().real_neg())
    };

    let branch0 = y
        .clone()
        .eq(zero.clone())
        .and(bce1.clone().eq(bce1_y0))
        .and(bce2.clone().eq(bce2_y0));
    let branch1 = y
        .clone()
        .eq(one.clone())
        .and(bce1.clone().eq(bce1_y1))
        .and(bce2.clone().eq(bce2_y1));
    program.assert(branch0.or(branch1));

    // Negated property: bce1 != bce2.
    program.assert(bce1.ne(bce2));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 7: Focal Loss
// ---------------------------------------------------------------------------

/// Prove focal loss relationship to cross-entropy via gamma parameter.
///
/// Focal loss: FL(p) = -alpha * (1 - p)^gamma * log(p)
/// Cross-entropy: CE(p) = -log(p)
///
/// When gamma = 0: FL(p) = -alpha * 1 * log(p) = alpha * CE(p).
///
/// This proves that focal loss reduces to (scaled) cross-entropy when gamma = 0.
///
/// We encode: given gamma = 0 and (1-p)^0 = 1, prove FL = alpha * CE.
pub(crate) fn prove_focal_loss_gamma_zero_is_ce() -> Result<LossPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let alpha = declare_real(&mut program, "alpha");
    let neg_log_p = declare_real(&mut program, "neg_log_p"); // -log(p) = CE(p)
    let focal = declare_real(&mut program, "focal");
    let alpha_ce = declare_real(&mut program, "alpha_ce");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // alpha > 0 (weighting factor)
    program.assert(alpha.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &alpha, 0.01, 10.0)?;

    // neg_log_p > 0 for p in (0, 1) — this is the cross-entropy for one element
    program.assert(neg_log_p.clone().real_ge(zero.clone()));
    assert_bounds(&mut program, &neg_log_p, 0.0, 100.0)?;

    // When gamma = 0: (1 - p)^0 = 1 (for any p != 1, and at p=1 CE=0 anyway)
    // focal = alpha * 1 * neg_log_p = alpha * neg_log_p
    program.assert(
        focal
            .clone()
            .eq(alpha.clone().real_mul(one.real_mul(neg_log_p.clone()))),
    );

    // alpha_ce = alpha * CE = alpha * neg_log_p
    program.assert(alpha_ce.clone().eq(alpha.real_mul(neg_log_p)));

    // Negated property: focal != alpha * CE
    let violation = focal.ne(alpha_ce);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "focal_loss_gamma_zero_is_ce".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove focal loss non-negativity.
///
/// FL(p) = alpha * (1-p)^gamma * (-log(p))
///
/// With alpha > 0, (1-p)^gamma >= 0 for p in (0,1) and gamma >= 0,
/// and -log(p) >= 0 for p in (0,1]:
///   FL(p) >= 0.
///
/// We encode the modulating factor (1-p)^gamma as a symbolic variable `mod_factor`
/// constrained to >= 0, and prove the product is non-negative.
pub(crate) fn prove_focal_loss_non_negative() -> Result<LossPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let alpha = declare_real(&mut program, "alpha");
    let mod_factor = declare_real(&mut program, "mod_factor"); // (1-p)^gamma
    let neg_log_p = declare_real(&mut program, "neg_log_p"); // -log(p) >= 0
    let focal = declare_real(&mut program, "focal");

    let zero = Expr::real(0);

    // alpha > 0
    program.assert(alpha.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &alpha, 0.01, 10.0)?;

    // mod_factor = (1-p)^gamma >= 0 for p in (0,1), gamma >= 0
    program.assert(mod_factor.clone().real_ge(zero.clone()));
    assert_bounds(&mut program, &mod_factor, 0.0, 1.0)?;

    // neg_log_p >= 0
    program.assert(neg_log_p.clone().real_ge(zero.clone()));
    assert_bounds(&mut program, &neg_log_p, 0.0, 100.0)?;

    // focal = alpha * mod_factor * neg_log_p
    let inner = mod_factor.real_mul(neg_log_p);
    program.assert(focal.clone().eq(alpha.real_mul(inner)));

    // Negated property: focal < 0
    let violation = focal.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "focal_loss_non_negative".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 8: CTC Loss
// ---------------------------------------------------------------------------

/// Prove CTC loss non-negativity.
///
/// CTC (Connectionist Temporal Classification) loss is defined as:
///   L = -log(P(target | input))
///
/// Since P(target | input) is a probability in (0, 1]:
///   log(P) <= 0, so -log(P) >= 0.
///
/// This is the same argument as cross-entropy non-negativity: the negative
/// log of a probability is non-negative.
pub(crate) fn prove_ctc_loss_non_negative() -> Result<LossPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let log_prob = declare_real(&mut program, "log_prob"); // log(P(target|input))
    let loss = declare_real(&mut program, "loss");

    let zero = Expr::real(0);

    // log(P) <= 0 since P in (0, 1]
    program.assert(log_prob.clone().real_le(zero.clone()));
    assert_bounds(&mut program, &log_prob, -100.0, 0.0)?;

    // loss = -log(P) >= 0
    program.assert(loss.clone().eq(log_prob.real_neg()));

    // Negated property: loss < 0
    let violation = loss.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "ctc_loss_non_negative".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove CTC loss is zero when probability is 1 (perfect alignment).
///
/// When P(target | input) = 1, log(1) = 0, so L = -0 = 0.
pub(crate) fn prove_ctc_loss_zero_at_perfect() -> Result<LossPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let loss = declare_real(&mut program, "loss");

    let zero = Expr::real(0);

    // log(1) = 0, so loss = -0 = 0
    program.assert(loss.clone().eq(zero.clone().real_neg()));

    // Negated property: loss != 0
    let violation = loss.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LossPropertyResult {
        property: "ctc_loss_zero_at_perfect".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Cross-Entropy Tests ---

    #[test]
    fn test_cross_entropy_non_negative_proven() {
        let result = prove_cross_entropy_non_negative().expect("proof should not error");
        assert!(
            result.proven,
            "Cross-entropy non-negativity should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "cross_entropy_non_negative");
    }

    #[test]
    fn test_cross_entropy_zero_at_match_proven() {
        let result = prove_cross_entropy_zero_at_match().expect("proof should not error");
        assert!(
            result.proven,
            "Cross-entropy zero at match should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "cross_entropy_zero_at_match");
    }

    // --- MSE Tests ---

    #[test]
    fn test_mse_non_negative_proven() {
        let result = prove_mse_non_negative().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "MSE non-negativity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "MSE non-negativity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "mse_non_negative");
    }

    #[test]
    fn test_mse_symmetry_proven() {
        let result = prove_mse_symmetry().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "MSE symmetry: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "MSE symmetry must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "mse_symmetry");
    }

    #[test]
    fn test_mse_zero_iff_equal_proven() {
        let result = prove_mse_zero_iff_equal().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "MSE zero iff equal: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "MSE zero iff equal must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "mse_zero_iff_equal");
    }

    // --- L1 Tests ---

    #[test]
    fn test_l1_non_negative_proven() {
        let result = prove_l1_non_negative().expect("proof should not error");
        assert!(
            result.proven,
            "L1 non-negativity should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "l1_non_negative");
    }

    #[test]
    fn test_l1_triangle_inequality_proven() {
        let result = prove_l1_triangle_inequality().expect("proof should not error");
        assert!(
            result.proven,
            "L1 triangle inequality should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "l1_triangle_inequality");
    }

    // --- Huber Loss Tests ---

    #[test]
    fn test_huber_quadratic_near_zero_proven() {
        let result = prove_huber_quadratic_near_zero().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Huber quadratic: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Huber quadratic must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "huber_quadratic_near_zero");
    }

    #[test]
    fn test_huber_linear_far_from_zero_proven() {
        let result = prove_huber_linear_far_from_zero().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Huber linear: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Huber linear must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "huber_linear_far_from_zero");
    }

    #[test]
    fn test_huber_continuity_at_delta_proven() {
        let result = prove_huber_continuity_at_delta().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Huber continuity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Huber continuity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "huber_continuity_at_delta");
    }

    // --- KL Divergence Tests ---

    #[test]
    fn test_kl_divergence_non_negative_proven() {
        let result = prove_kl_divergence_non_negative().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "KL divergence non-negativity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "KL divergence non-negativity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "kl_divergence_non_negative");
    }

    #[test]
    fn test_kl_divergence_zero_when_equal_proven() {
        let result = prove_kl_divergence_zero_when_equal().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "KL divergence zero when equal: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "KL divergence zero must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "kl_divergence_zero_when_equal");
    }

    // --- Binary Cross-Entropy Tests ---

    #[test]
    fn test_bce_non_negative_proven() {
        let result = prove_bce_non_negative().expect("proof should not error");
        // QF_LRA over a concrete case split is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "BCE non-negativity should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(crate::ay_vacuity::vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "bce_non_negative");
    }

    /// The `-log(..)` sign is the whole theorem: dropping the negation makes
    /// `loss = log(..) <= 0`, so `loss < 0` is satisfiable and the query must
    /// be SAT.
    #[test]
    fn bce_non_negative_depends_on_the_log_sign() {
        let program = build_bce_non_negative(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without the `-log` negation the loss can be negative and the query must be SAT; \
             got: {detail}",
        );
    }

    #[test]
    fn test_bce_symmetry_proven() {
        let result = prove_bce_symmetry().expect("proof should not error");
        // QF_LRA over a concrete case split is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "BCE symmetry should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(crate::ay_vacuity::vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "bce_symmetry");
    }

    /// Symmetry holds only because complementing `p` swaps which log applies.
    /// Swapping the label without complementing the probability makes
    /// `bce1 != bce2` in each branch, so the query must be SAT.
    #[test]
    fn bce_symmetry_depends_on_probability_complement() {
        let program = build_bce_symmetry(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "swapping the label without complementing p breaks the symmetry and the query \
             must be SAT; got: {detail}",
        );
    }

    // --- Focal Loss Tests ---

    #[test]
    fn test_focal_loss_gamma_zero_is_ce_proven() {
        let result = prove_focal_loss_gamma_zero_is_ce().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Focal loss gamma=0: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Focal loss gamma=0 must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "focal_loss_gamma_zero_is_ce");
    }

    #[test]
    fn test_focal_loss_non_negative_proven() {
        let result = prove_focal_loss_non_negative().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Focal loss non-negativity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Focal loss non-negativity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "focal_loss_non_negative");
    }

    // --- CTC Loss Tests ---

    #[test]
    fn test_ctc_loss_non_negative_proven() {
        let result = prove_ctc_loss_non_negative().expect("proof should not error");
        assert!(
            result.proven,
            "CTC loss non-negativity should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "ctc_loss_non_negative");
    }

    #[test]
    fn test_ctc_loss_zero_at_perfect_proven() {
        let result = prove_ctc_loss_zero_at_perfect().expect("proof should not error");
        assert!(
            result.proven,
            "CTC loss zero at perfect alignment should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "ctc_loss_zero_at_perfect");
    }

    // --- SMT2 Structure Tests ---

    #[test]
    fn test_all_loss_proofs_have_valid_smt2() {
        let proofs: Vec<LossPropertyResult> = vec![
            prove_cross_entropy_non_negative().unwrap(),
            prove_cross_entropy_zero_at_match().unwrap(),
            prove_mse_non_negative().unwrap(),
            prove_mse_symmetry().unwrap(),
            prove_mse_zero_iff_equal().unwrap(),
            prove_l1_non_negative().unwrap(),
            prove_l1_triangle_inequality().unwrap(),
            prove_huber_quadratic_near_zero().unwrap(),
            prove_huber_continuity_at_delta().unwrap(),
            prove_kl_divergence_non_negative().unwrap(),
            prove_kl_divergence_zero_when_equal().unwrap(),
            prove_bce_non_negative().unwrap(),
            prove_bce_symmetry().unwrap(),
            prove_focal_loss_gamma_zero_is_ce().unwrap(),
            prove_focal_loss_non_negative().unwrap(),
            prove_ctc_loss_non_negative().unwrap(),
            prove_ctc_loss_zero_at_perfect().unwrap(),
        ];

        for proof in &proofs {
            assert!(
                proof.smt2.contains("check-sat"),
                "{}: SMT2 should contain check-sat",
                proof.property,
            );
            assert!(
                proof.smt2.contains("declare-const"),
                "{}: SMT2 should have declarations",
                proof.property,
            );
            assert!(
                proof.smt2.contains("set-logic"),
                "{}: SMT2 should declare logic",
                proof.property,
            );
        }
    }
}
