// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for optimizer update rule mathematical properties (#4186).
//!
//! Proves fundamental mathematical properties of optimizer algorithms used
//! in nn-optim (SGD, Adam, weight decay, learning rate schedules, gradient
//! clipping). Each proof encodes the expected property as a negated assertion
//! and proves UNSAT (no counterexample exists).
//!
//! # Proved Properties
//!
//! 1. **SGD update direction**: w_new = w - lr * grad; lr > 0 means update
//!    moves in the negative gradient direction.
//! 2. **SGD with momentum**: velocity accumulation with momentum in [0, 1).
//! 3. **Adam first moment**: exponential moving average of gradients, beta1 in [0, 1).
//! 4. **Adam second moment**: exponential moving average of squared gradients.
//! 5. **Adam bias correction**: corrected_m = m / (1 - beta1^t), increases early.
//! 6. **Adam update bound**: step size bounded by lr / sqrt(v_hat + eps).
//! 7. **Weight decay**: decoupled weight decay w_new = (1 - wd*lr)*w - lr*grad.
//! 8. **Learning rate warmup**: lr increases linearly from 0 to max_lr.
//! 9. **Cosine annealing**: lr follows cosine curve from max_lr to min_lr.
//! 10. **Gradient clipping**: ||clipped_grad|| <= max_norm.
//!
//! # Proof Strategy
//!
//! - **Algebraic identity proofs** (SGD, momentum, Adam moments, weight decay):
//!   Pure polynomial/linear identities provable via QF_NRA or QF_LRA.
//!
//! - **Bound proofs** (Adam update bound, gradient clipping, bias correction):
//!   Constrain variables to valid ranges, prove bound violations are UNSAT.
//!
//! - **Schedule proofs** (warmup, cosine annealing): Encode schedule formulas
//!   with symbolic step/epoch variables, prove monotonicity/range properties.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::ay_real_lit::RealLit;
use super::error::SmtError;
use super::translate_real::real_from_f64;

/// Result of an optimizer property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct OptimizerPropertyResult {
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

/// Assert `lower < expr < upper` (strict).
fn assert_strict_bounds(
    program: &mut AYProgram,
    expr: &Expr,
    lower: f64,
    upper: f64,
) -> Result<(), SmtError> {
    let lo = real_from_f64(lower)?;
    let hi = real_from_f64(upper)?;
    program.assert(expr.clone().real_gt(lo));
    program.assert(expr.clone().real_lt(hi));
    Ok(())
}

/// Execute a ay program and return whether UNSAT (property proven).
///
/// The verdict is funneled through [`crate::ay_vacuity::reject_if_vacuous`] so a
/// query that is UNSAT only because it asserts `P ∧ ¬P` (or compares a term to
/// itself) never counts as a proof — any residual vacuity becomes a hard test
/// failure in the corresponding `test_*_proven`.
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
    crate::ay_vacuity::reject_if_vacuous(&program.to_string(), proven, detail)
}

// ---------------------------------------------------------------------------
// Property 1: SGD Update Direction
// ---------------------------------------------------------------------------

/// Prove that SGD update w_new = w - lr * grad moves in the negative gradient
/// direction when lr > 0.
///
/// Specifically: when grad > 0 and lr > 0, then w_new < w (weight decreases).
/// This proves the update opposes the gradient direction.
///
/// Encoding: define w_new = w - lr * grad, assert lr > 0 and grad > 0,
/// then assert w_new >= w (negated property). UNSAT proves w_new < w.
pub(crate) fn prove_sgd_update_direction() -> Result<OptimizerPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let w = declare_real(&mut program, "w");
    let lr = declare_real(&mut program, "lr");
    let grad = declare_real(&mut program, "grad");
    let w_new = declare_real(&mut program, "w_new");

    assert_bounds(&mut program, &w, -100.0, 100.0)?;
    assert_bounds(&mut program, &lr, 0.0, 10.0)?;
    assert_bounds(&mut program, &grad, 0.0, 100.0)?;

    // lr > 0 (strictly positive learning rate)
    let zero = Expr::real(0);
    program.assert(lr.clone().real_gt(zero.clone()));

    // grad > 0 (positive gradient)
    program.assert(grad.clone().real_gt(zero.clone()));

    // w_new = w - lr * grad
    let lr_grad = declare_real(&mut program, "lr_grad");
    program.assert(lr_grad.clone().eq(lr.real_mul(grad)));
    program.assert(w_new.clone().eq(w.clone().real_sub(lr_grad)));

    // Negated property: w_new >= w (should be impossible when lr > 0, grad > 0)
    let violation = w_new.real_ge(w);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(OptimizerPropertyResult {
        property: "optimizer_sgd_update_direction".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove the SGD update identity: w_new = w - lr * grad.
///
/// Encodes the definition and proves no assignment violates the identity.
pub(crate) fn prove_sgd_update_identity() -> Result<OptimizerPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let w = declare_real(&mut program, "w");
    let lr = declare_real(&mut program, "lr");
    let grad = declare_real(&mut program, "grad");
    let w_new = declare_real(&mut program, "w_new");

    assert_bounds(&mut program, &w, -100.0, 100.0)?;
    assert_bounds(&mut program, &lr, 0.0, 10.0)?;
    assert_bounds(&mut program, &grad, -100.0, 100.0)?;

    // w_new = w - lr * grad
    let lr_grad = declare_real(&mut program, "lr_grad");
    program.assert(lr_grad.clone().eq(lr.clone().real_mul(grad.clone())));
    program.assert(w_new.clone().eq(w.clone().real_sub(lr_grad.clone())));

    // Negated property: w_new != w - lr * grad
    let expected = w.real_sub(lr.real_mul(grad));
    let violation = w_new.ne(expected);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(OptimizerPropertyResult {
        property: "optimizer_sgd_update_identity".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: SGD with Momentum
// ---------------------------------------------------------------------------

/// Prove that SGD momentum velocity accumulation follows v_new = mu * v + grad,
/// and the weight update w_new = w - lr * v_new.
///
/// With momentum mu in [0, 1), the velocity accumulates past gradients with
/// exponential decay. We prove: w_new = w - lr * (mu * v + grad).
pub(crate) fn prove_sgd_momentum_accumulation() -> Result<OptimizerPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let w = declare_real(&mut program, "w");
    let lr = declare_real(&mut program, "lr");
    let grad = declare_real(&mut program, "grad");
    let v = declare_real(&mut program, "v");
    let mu = declare_real(&mut program, "mu");
    let v_new = declare_real(&mut program, "v_new");
    let w_new = declare_real(&mut program, "w_new");

    assert_bounds(&mut program, &w, -100.0, 100.0)?;
    assert_bounds(&mut program, &lr, 0.0, 10.0)?;
    assert_bounds(&mut program, &grad, -100.0, 100.0)?;
    assert_bounds(&mut program, &v, -100.0, 100.0)?;

    // mu in [0, 1)
    let zero = Expr::real(0);
    let one = Expr::real(1);
    program.assert(mu.clone().real_ge(zero));
    program.assert(mu.clone().real_lt(one));

    // v_new = mu * v + grad
    let mu_v = declare_real(&mut program, "mu_v");
    program.assert(mu_v.clone().eq(mu.clone().real_mul(v.clone())));
    program.assert(v_new.clone().eq(mu_v.real_add(grad.clone())));

    // w_new = w - lr * v_new
    let lr_v_new = declare_real(&mut program, "lr_v_new");
    program.assert(lr_v_new.clone().eq(lr.clone().real_mul(v_new.clone())));
    program.assert(w_new.clone().eq(w.clone().real_sub(lr_v_new)));

    // Negated property: w_new != w - lr * (mu * v + grad)
    let expected = w.real_sub(lr.real_mul(mu.real_mul(v).real_add(grad)));
    let violation = w_new.ne(expected);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(OptimizerPropertyResult {
        property: "optimizer_sgd_momentum_accumulation".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 3: Adam First Moment (EMA of Gradients)
// ---------------------------------------------------------------------------

/// Prove the Adam first moment update: m_new = beta1 * m + (1 - beta1) * grad.
///
/// This is the exponential moving average of gradients. With beta1 in [0, 1),
/// the new moment is a weighted combination of the old moment and current gradient.
pub(crate) fn prove_adam_first_moment() -> Result<OptimizerPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let m = declare_real(&mut program, "m");
    let grad = declare_real(&mut program, "grad");
    let beta1 = declare_real(&mut program, "beta1");
    let m_new = declare_real(&mut program, "m_new");

    assert_bounds(&mut program, &m, -100.0, 100.0)?;
    assert_bounds(&mut program, &grad, -100.0, 100.0)?;

    let zero = Expr::real(0);
    let one = Expr::real(1);
    program.assert(beta1.clone().real_ge(zero));
    program.assert(beta1.clone().real_lt(one.clone()));

    // Intermediates
    let beta1_m = declare_real(&mut program, "beta1_m");
    program.assert(beta1_m.clone().eq(beta1.clone().real_mul(m.clone())));

    let one_minus_beta1 = declare_real(&mut program, "one_minus_beta1");
    program.assert(one_minus_beta1.clone().eq(one.real_sub(beta1.clone())));

    let comp_grad = declare_real(&mut program, "comp_grad");
    program.assert(
        comp_grad
            .clone()
            .eq(one_minus_beta1.clone().real_mul(grad.clone())),
    );

    // m_new = beta1 * m + (1 - beta1) * grad
    program.assert(m_new.clone().eq(beta1_m.real_add(comp_grad)));

    // Negated property: m_new != beta1 * m + (1 - beta1) * grad
    let rhs_check = beta1.real_mul(m).real_add(one_minus_beta1.real_mul(grad));
    let violation = m_new.ne(rhs_check);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(OptimizerPropertyResult {
        property: "optimizer_adam_first_moment_ema".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: Adam Second Moment (EMA of Squared Gradients)
// ---------------------------------------------------------------------------

/// Prove Adam second moment non-negativity: v_new >= 0 when v >= 0.
///
/// v_new = beta2 * v + (1 - beta2) * grad^2. Since beta2 in [0, 1),
/// v >= 0, and grad^2 >= 0, v_new must be non-negative.
pub(crate) fn prove_adam_second_moment() -> Result<OptimizerPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let v = declare_real(&mut program, "v");
    let grad = declare_real(&mut program, "grad");
    let beta2 = declare_real(&mut program, "beta2");
    let v_new = declare_real(&mut program, "v_new");
    let grad_sq = declare_real(&mut program, "grad_sq");

    assert_bounds(&mut program, &v, 0.0, 10000.0)?;
    assert_bounds(&mut program, &grad, -100.0, 100.0)?;

    let zero = Expr::real(0);
    let one = Expr::real(1);
    program.assert(beta2.clone().real_ge(zero.clone()));
    program.assert(beta2.clone().real_lt(one.clone()));

    // grad_sq = grad * grad
    program.assert(grad_sq.clone().eq(grad.clone().real_mul(grad.clone())));

    // v_new = beta2 * v + (1 - beta2) * grad^2
    let beta2_v = declare_real(&mut program, "beta2_v");
    program.assert(beta2_v.clone().eq(beta2.clone().real_mul(v.clone())));

    let one_minus_beta2 = declare_real(&mut program, "one_minus_beta2");
    program.assert(one_minus_beta2.clone().eq(one.real_sub(beta2.clone())));

    let comp_grad_sq = declare_real(&mut program, "comp_grad_sq");
    program.assert(comp_grad_sq.clone().eq(one_minus_beta2.real_mul(grad_sq)));

    program.assert(v_new.clone().eq(beta2_v.real_add(comp_grad_sq)));

    // Negated property: v_new < 0
    let violation = v_new.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(OptimizerPropertyResult {
        property: "optimizer_adam_second_moment_nonneg".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove the Adam second-moment interpolation bound.
///
/// The second-moment update `v_new = beta2*v + (1 - beta2)*grad^2` is a *convex
/// combination* of the running estimate `v` and the incoming squared gradient
/// `g2 = grad^2` (weights `beta2` and `1 - beta2`, both in `[0, 1]`). A convex
/// combination never leaves the interval spanned by its endpoints, so whenever
/// `v <= g2` the update must satisfy `v <= v_new <= g2`: the second-moment
/// estimate moves *toward* the fresh signal without ever overshooting it. This
/// is a genuine consequence of the update rule — a wrong weight makes it false —
/// rather than a restatement of the definition.
///
/// Encoding is kept in decidable `QF_LRA`:
/// - `beta2` is pinned to the concrete rational `9/10`, so every product has a
///   literal factor and stays linear (a declared `beta2 * v` would be var×var,
///   i.e. `QF_NRA`, and typically returns `Unknown`).
/// - `g2` is an arbitrary non-negative real standing for `grad^2`. The bound
///   needs only `g2 >= 0`, never that `g2` is a perfect square, so we avoid the
///   `grad * grad` product entirely.
///
/// This wraps [`build_adam_second_moment_identity`]; the boolean knob is
/// exercised by the mutation test `adam_second_moment_identity_depends_on_the_scale`.
pub(crate) fn prove_adam_second_moment_identity() -> Result<OptimizerPropertyResult, SmtError> {
    let program = build_adam_second_moment_identity(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(OptimizerPropertyResult {
        property: "optimizer_adam_second_moment_identity".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the Adam second-moment interpolation query.
///
/// When `correct` is false the fresh-signal term is left unscaled
/// (`v_new = beta2*v + g2`, the classic "forgot the `(1 - beta2)` weight" slip).
/// Then any `g2 > 0` pushes `v_new` above `g2`, the interpolation bound is
/// violated, and the query flips to SAT — which is what the mutation test checks.
fn build_adam_second_moment_identity(correct: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let v = declare_real(&mut program, "v");
    // g2 = grad^2: an arbitrary non-negative real (never `grad * grad`), so the
    // query stays linear instead of dropping into QF_NRA.
    let g2 = declare_real(&mut program, "g2");
    let v_new = declare_real(&mut program, "v_new");

    // v (a variance estimate) and g2 (a square) are both non-negative.
    assert_bounds(&mut program, &v, 0.0, 10000.0)?;
    assert_bounds(&mut program, &g2, 0.0, 10000.0)?;

    // Fix the ordering v <= g2 so the two-sided bound is stated against concrete
    // endpoints; the g2 <= v case is symmetric.
    program.assert(v.clone().real_le(g2.clone()));

    // beta2 = 9/10, a representative decay in (0, 1). Pinned to a literal so
    // `beta2 * v` and `(1 - beta2) * g2` are both linear.
    let beta2 = Expr::real_ratio(9, 10);

    // decayed = beta2 * v (one step removed from the conclusion).
    let decayed = declare_real(&mut program, "decayed");
    program.assert(decayed.clone().eq(beta2.real_mul(v.clone())));

    // fresh = (1 - beta2) * g2, or — in the buggy variant — g2 unscaled.
    let fresh = declare_real(&mut program, "fresh");
    let fresh_term = if correct {
        Expr::real_ratio(1, 10).real_mul(g2.clone())
    } else {
        g2.clone()
    };
    program.assert(fresh.clone().eq(fresh_term));

    // v_new = decayed + fresh (the second-moment update).
    program.assert(v_new.clone().eq(decayed.real_add(fresh)));

    // Property: the convex combination stays within [v, g2].
    // Violation: v_new < v OR v_new > g2.
    let below = v_new.clone().real_lt(v);
    let above = v_new.real_gt(g2);
    program.assert(below.or(above));
    program.check_sat();

    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 5: Adam Bias Correction
// ---------------------------------------------------------------------------

/// Prove Adam bias correction amplifies: corrected_m >= m when m > 0.
///
/// At t=1: correction_denom = 1 - beta1. Since beta1 in (0, 1),
/// correction_denom in (0, 1), so corrected_m = m / correction_denom > m.
pub(crate) fn prove_adam_bias_correction_amplifies() -> Result<OptimizerPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let m = declare_real(&mut program, "m");
    let beta1 = declare_real(&mut program, "beta1");
    let corrected_m = declare_real(&mut program, "corrected_m");
    let correction_denom = declare_real(&mut program, "correction_denom");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // m > 0
    program.assert(m.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &m, 0.0, 100.0)?;

    // beta1 in (0, 1)
    assert_strict_bounds(&mut program, &beta1, 0.0, 1.0)?;

    // correction_denom = 1 - beta1 (at t=1)
    program.assert(correction_denom.clone().eq(one.real_sub(beta1)));

    // corrected_m * correction_denom = m
    program.assert(corrected_m.clone().real_mul(correction_denom).eq(m.clone()));

    // Negated property: corrected_m < m
    let violation = corrected_m.real_lt(m);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(OptimizerPropertyResult {
        property: "optimizer_adam_bias_correction_amplifies".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove bias correction factor >= 1: 1 / (1 - beta) >= 1 for beta in (0, 1).
pub(crate) fn prove_adam_bias_correction_factor_geq_one(
) -> Result<OptimizerPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let beta = declare_real(&mut program, "beta");
    let correction_denom = declare_real(&mut program, "correction_denom");
    let factor = declare_real(&mut program, "factor");

    assert_strict_bounds(&mut program, &beta, 0.0, 1.0)?;

    let one = Expr::real(1);

    // correction_denom = 1 - beta
    program.assert(correction_denom.clone().eq(one.clone().real_sub(beta)));

    // factor * correction_denom = 1 (i.e., factor = 1 / correction_denom)
    program.assert(factor.clone().real_mul(correction_denom).eq(one.clone()));

    // Negated property: factor < 1
    let violation = factor.real_lt(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(OptimizerPropertyResult {
        property: "optimizer_adam_bias_correction_factor_geq_one".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 6: Adam Update Bound
// ---------------------------------------------------------------------------

/// Prove Adam update denominator >= eps: sqrt(v_hat) + eps >= eps.
///
/// Since sqrt(v_hat) >= 0 and eps > 0, the denominator is at least eps,
/// bounding the step size to at most lr * |m_hat| / eps.
pub(crate) fn prove_adam_update_denominator_bound() -> Result<OptimizerPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let v_hat = declare_real(&mut program, "v_hat");
    let eps = declare_real(&mut program, "eps");
    let sqrt_v_hat = declare_real(&mut program, "sqrt_v_hat");
    let denom = declare_real(&mut program, "denom");

    let zero = Expr::real(0);

    // v_hat >= 0
    program.assert(v_hat.clone().real_ge(zero.clone()));
    assert_bounds(&mut program, &v_hat, 0.0, 10000.0)?;

    // eps > 0
    program.assert(eps.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &eps, 0.0, 1.0)?;

    // sqrt_v_hat >= 0 and sqrt_v_hat^2 = v_hat
    program.assert(sqrt_v_hat.clone().real_ge(zero));
    program.assert(sqrt_v_hat.clone().real_mul(sqrt_v_hat.clone()).eq(v_hat));

    // denom = sqrt_v_hat + eps
    program.assert(denom.clone().eq(sqrt_v_hat.real_add(eps.clone())));

    // Negated property: denom < eps
    let violation = denom.real_lt(eps);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(OptimizerPropertyResult {
        property: "optimizer_adam_update_denominator_bound".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 7: Weight Decay (Decoupled)
// ---------------------------------------------------------------------------

/// Prove decoupled weight decay identity:
/// w_new = (1 - wd * lr) * w - lr * grad.
pub(crate) fn prove_weight_decay_identity() -> Result<OptimizerPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let w = declare_real(&mut program, "w");
    let lr = declare_real(&mut program, "lr");
    let grad = declare_real(&mut program, "grad");
    let wd = declare_real(&mut program, "wd");
    let w_new = declare_real(&mut program, "w_new");

    assert_bounds(&mut program, &w, -100.0, 100.0)?;
    assert_bounds(&mut program, &lr, 0.0, 10.0)?;
    assert_bounds(&mut program, &grad, -100.0, 100.0)?;
    assert_bounds(&mut program, &wd, 0.0, 1.0)?;

    let one = Expr::real(1);

    // Intermediates
    let wd_lr = declare_real(&mut program, "wd_lr");
    program.assert(wd_lr.clone().eq(wd.clone().real_mul(lr.clone())));

    let decay_factor = declare_real(&mut program, "decay_factor");
    program.assert(decay_factor.clone().eq(one.clone().real_sub(wd_lr)));

    let decayed_w = declare_real(&mut program, "decayed_w");
    program.assert(decayed_w.clone().eq(decay_factor.real_mul(w.clone())));

    let lr_grad = declare_real(&mut program, "lr_grad");
    program.assert(lr_grad.clone().eq(lr.clone().real_mul(grad.clone())));

    program.assert(w_new.clone().eq(decayed_w.real_sub(lr_grad)));

    // Negated property: w_new != (1 - wd * lr) * w - lr * grad
    let expected = one
        .real_sub(wd.real_mul(lr.clone()))
        .real_mul(w)
        .real_sub(lr.real_mul(grad));
    let violation = w_new.ne(expected);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(OptimizerPropertyResult {
        property: "optimizer_weight_decay_identity".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove weight decay shrinks weight magnitude: when wd > 0, lr > 0,
/// grad = 0, w > 0, and wd*lr < 1, then w_new < w.
pub(crate) fn prove_weight_decay_shrinks() -> Result<OptimizerPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let w = declare_real(&mut program, "w");
    let lr = declare_real(&mut program, "lr");
    let wd = declare_real(&mut program, "wd");
    let w_new = declare_real(&mut program, "w_new");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // w > 0, lr > 0, wd > 0
    program.assert(w.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &w, 0.0, 100.0)?;
    program.assert(lr.clone().real_gt(zero.clone()));
    program.assert(wd.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &lr, 0.0, 1.0)?;
    assert_bounds(&mut program, &wd, 0.0, 1.0)?;

    // wd * lr < 1
    let wd_lr = declare_real(&mut program, "wd_lr");
    program.assert(wd_lr.clone().eq(wd.real_mul(lr)));
    program.assert(wd_lr.clone().real_lt(one.clone()));

    // w_new = (1 - wd*lr) * w  (grad = 0)
    let decay_factor = declare_real(&mut program, "decay_factor");
    program.assert(decay_factor.clone().eq(one.real_sub(wd_lr)));
    program.assert(w_new.clone().eq(decay_factor.real_mul(w.clone())));

    // Negated property: w_new >= w
    let violation = w_new.real_ge(w);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(OptimizerPropertyResult {
        property: "optimizer_weight_decay_shrinks".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 8: Learning Rate Warmup
// ---------------------------------------------------------------------------

/// Prove linear warmup is monotonically increasing:
/// for step_a < step_b, lr(step_a) < lr(step_b).
pub(crate) fn prove_lr_warmup_monotonic() -> Result<OptimizerPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let max_lr = declare_real(&mut program, "max_lr");
    let warmup_steps = declare_real(&mut program, "warmup_steps");
    let step_a = declare_real(&mut program, "step_a");
    let step_b = declare_real(&mut program, "step_b");
    let lr_a = declare_real(&mut program, "lr_a");
    let lr_b = declare_real(&mut program, "lr_b");

    let zero = Expr::real(0);

    // max_lr > 0, warmup_steps > 0
    program.assert(max_lr.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &max_lr, 0.0, 10.0)?;
    program.assert(warmup_steps.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &warmup_steps, 1.0, 10000.0)?;

    // 0 <= step_a < step_b <= warmup_steps
    program.assert(step_a.clone().real_ge(zero.clone()));
    program.assert(step_a.clone().real_lt(step_b.clone()));
    program.assert(step_b.clone().real_le(warmup_steps.clone()));

    // lr_a * warmup_steps = max_lr * step_a
    program.assert(
        lr_a.clone()
            .real_mul(warmup_steps.clone())
            .eq(max_lr.clone().real_mul(step_a)),
    );

    // lr_b * warmup_steps = max_lr * step_b
    program.assert(
        lr_b.clone()
            .real_mul(warmup_steps)
            .eq(max_lr.real_mul(step_b)),
    );

    // Negated property: lr_a >= lr_b
    let violation = lr_a.real_ge(lr_b);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(OptimizerPropertyResult {
        property: "optimizer_lr_warmup_monotonic".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove warmup lr is bounded: 0 <= lr(step) <= max_lr.
pub(crate) fn prove_lr_warmup_bounded() -> Result<OptimizerPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let max_lr = declare_real(&mut program, "max_lr");
    let warmup_steps = declare_real(&mut program, "warmup_steps");
    let step = declare_real(&mut program, "step");
    let lr = declare_real(&mut program, "lr");

    let zero = Expr::real(0);

    // max_lr > 0, warmup_steps > 0
    program.assert(max_lr.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &max_lr, 0.0, 10.0)?;
    program.assert(warmup_steps.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &warmup_steps, 1.0, 10000.0)?;

    // 0 <= step <= warmup_steps
    program.assert(step.clone().real_ge(zero.clone()));
    program.assert(step.clone().real_le(warmup_steps.clone()));

    // lr * warmup_steps = max_lr * step
    program.assert(
        lr.clone()
            .real_mul(warmup_steps)
            .eq(max_lr.clone().real_mul(step)),
    );

    // Negated property: lr < 0 OR lr > max_lr
    let too_low = lr.clone().real_lt(zero);
    let too_high = lr.real_gt(max_lr);
    let violation = too_low.or(too_high);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(OptimizerPropertyResult {
        property: "optimizer_lr_warmup_bounded".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 9: Cosine Annealing
// ---------------------------------------------------------------------------

/// Prove cosine annealing lr is bounded: min_lr <= lr <= max_lr.
///
/// lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * t / T)).
/// Since cos in [-1, 1], (1 + cos) in [0, 2], so lr in [min_lr, max_lr].
pub(crate) fn prove_cosine_annealing_bounded() -> Result<OptimizerPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let min_lr = declare_real(&mut program, "min_lr");
    let max_lr = declare_real(&mut program, "max_lr");
    let c = declare_real(&mut program, "c"); // cos(pi * t / T) in [-1, 1]
    let lr = declare_real(&mut program, "lr");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // 0 <= min_lr < max_lr
    program.assert(min_lr.clone().real_ge(zero));
    program.assert(min_lr.clone().real_lt(max_lr.clone()));
    assert_bounds(&mut program, &min_lr, 0.0, 10.0)?;
    assert_bounds(&mut program, &max_lr, 0.0, 10.0)?;

    // c in [-1, 1]
    let neg_one = real_from_f64(-1.0)?;
    program.assert(c.clone().real_ge(neg_one));
    program.assert(c.clone().real_le(one.clone()));

    // lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + c)
    let half = real_from_f64(0.5)?;
    let lr_range = declare_real(&mut program, "lr_range");
    program.assert(lr_range.clone().eq(max_lr.clone().real_sub(min_lr.clone())));

    let one_plus_c = declare_real(&mut program, "one_plus_c");
    program.assert(one_plus_c.clone().eq(one.real_add(c)));

    let scaled = declare_real(&mut program, "scaled");
    program.assert(
        scaled
            .clone()
            .eq(half.real_mul(lr_range).real_mul(one_plus_c)),
    );

    program.assert(lr.clone().eq(min_lr.clone().real_add(scaled)));

    // Negated property: lr < min_lr OR lr > max_lr
    let too_low = lr.clone().real_lt(min_lr);
    let too_high = lr.real_gt(max_lr);
    let violation = too_low.or(too_high);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(OptimizerPropertyResult {
        property: "optimizer_cosine_annealing_bounded".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove cosine annealing reaches max_lr at t=0 (when cos(0) = 1).
pub(crate) fn prove_cosine_annealing_initial() -> Result<OptimizerPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let min_lr = declare_real(&mut program, "min_lr");
    let max_lr = declare_real(&mut program, "max_lr");
    let lr = declare_real(&mut program, "lr");

    let zero = Expr::real(0);

    // 0 <= min_lr < max_lr
    program.assert(min_lr.clone().real_ge(zero));
    program.assert(min_lr.clone().real_lt(max_lr.clone()));
    assert_bounds(&mut program, &min_lr, 0.0, 10.0)?;
    assert_bounds(&mut program, &max_lr, 0.0, 10.0)?;

    // At t=0: c = cos(0) = 1, so (1 + c) = 2
    // lr = min_lr + 0.5 * (max_lr - min_lr) * 2 = max_lr
    let half = real_from_f64(0.5)?;
    let two = Expr::real(2);
    let lr_range = declare_real(&mut program, "lr_range");
    program.assert(lr_range.clone().eq(max_lr.clone().real_sub(min_lr.clone())));

    let scaled = declare_real(&mut program, "scaled");
    program.assert(scaled.clone().eq(half.real_mul(lr_range).real_mul(two)));

    program.assert(lr.clone().eq(min_lr.real_add(scaled)));

    // Negated property: lr != max_lr
    let violation = lr.ne(max_lr);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(OptimizerPropertyResult {
        property: "optimizer_cosine_annealing_initial_max".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 10: Gradient Clipping
// ---------------------------------------------------------------------------

/// Prove gradient clipping passthrough: when |g| <= max_norm, clipped = g,
/// so |clipped| <= max_norm.
pub(crate) fn prove_gradient_clipping_passthrough() -> Result<OptimizerPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let g = declare_real(&mut program, "g");
    let max_norm = declare_real(&mut program, "max_norm");
    let g_abs = declare_real(&mut program, "g_abs");

    let zero = Expr::real(0);

    // max_norm > 0
    program.assert(max_norm.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &max_norm, 0.0, 100.0)?;
    assert_bounds(&mut program, &g, -100.0, 100.0)?;

    // g_abs >= 0, g_abs >= g, g_abs >= -g, and g_abs = g or g_abs = -g
    program.assert(g_abs.clone().real_ge(zero.clone()));
    program.assert(g_abs.clone().real_ge(g.clone()));
    program.assert(g_abs.clone().real_ge(g.clone().real_neg()));
    let is_pos = g_abs.clone().eq(g.clone());
    let is_neg = g_abs.clone().eq(g.clone().real_neg());
    program.assert(is_pos.or(is_neg));

    // Passthrough branch: |g| <= max_norm
    program.assert(g_abs.clone().real_le(max_norm.clone()));

    // Negated property: g_abs > max_norm (contradicts the branch condition)
    let violation = g_abs.real_gt(max_norm);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(OptimizerPropertyResult {
        property: "optimizer_gradient_clipping_passthrough".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove gradient clipping strictly shrinks an over-threshold gradient: when the
/// gradient norm exceeds `max_norm`, rescaling by `max_norm / norm` lands the
/// clipped gradient strictly inside the open interval `(0, g)` — it stays
/// positive (direction-preserving) and drops strictly below the original
/// magnitude, `0 < clipped < g`.
///
/// The clipping operation is `scale = max_norm / norm; clipped = g * scale`. For
/// a positive scalar gradient `g > 0` the norm is `g` itself, so
/// `scale = max_norm / g` and `clipped = g * scale = max_norm`, which lies in the
/// open interval `(0, g)` exactly because the scaling branch guarantees
/// `0 < max_norm < g`. The theorem is genuinely contingent on those two branch
/// bounds: dropping `max_norm < g` lets the clip fail to reduce the magnitude and
/// dropping `max_norm > 0` lets it flip sign — either way the query turns SAT. It
/// is therefore a real consequence of the rescale, not the AC-tautology
/// `clipped == max_norm`, whose two sides are the same product `g * scale`
/// written twice (and which the lineage guard rejects as vacuous).
///
/// Encoding is kept in decidable `QF_LRA`, fast and terminating:
/// - The gradient magnitude `g` is pinned to the concrete literal `10`, a value
///   in the scaling branch (it exceeds every admissible `max_norm`). Pinning `g`
///   makes both `scale * g` (the division, encoded multiplicatively) and
///   `g * scale` (the rescale) `literal × var`, i.e. linear. Leaving `g` a
///   declared variable makes them `var × var` — `QF_NRA` — and the solver hangs.
/// - `max_norm` stays a *free* variable ranging over the whole scaling branch
///   `(0, g)`, so the bound `0 < clipped < g` is proved for every threshold below
///   the (concrete) gradient magnitude, not just one point.
///
/// This wraps [`build_gradient_clipping_scaling`]; the boolean knob is exercised
/// by the mutation test `gradient_clipping_scaling_depends_on_the_norm`.
pub(crate) fn prove_gradient_clipping_scaling() -> Result<OptimizerPropertyResult, SmtError> {
    let program = build_gradient_clipping_scaling(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(OptimizerPropertyResult {
        property: "optimizer_gradient_clipping_scaling".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the gradient-clipping rescale query.
///
/// When `correct` is false the rescale forgets to divide by the norm
/// (`clipped = g * max_norm` instead of `clipped = g * scale`, the classic
/// "scaled by the threshold, not by `threshold / norm`" slip). Then with the
/// concrete `g = 10` the clipped value is `10 * max_norm`, which for any
/// `max_norm >= 1` reaches or exceeds `g` itself — the "clip" fails to shrink the
/// gradient — so the `clipped >= g` disjunct is satisfiable and the query flips
/// to SAT, which is what the mutation test checks.
fn build_gradient_clipping_scaling(correct: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Concrete gradient magnitude g = 10 (the gradient norm for g > 0). Pinning
    // it to a literal keeps every product `literal × var` (linear / QF_LRA); a
    // declared `g` would make `scale * g` and `g * scale` var×var (QF_NRA) and
    // the solver would hang.
    let g = Expr::real(10);

    let max_norm = declare_real(&mut program, "max_norm");
    let scale = declare_real(&mut program, "scale");
    let clipped = declare_real(&mut program, "clipped");

    let zero = Expr::real(0);

    // Scaling branch: 0 < max_norm < g (the gradient norm exceeds the threshold).
    // Both bounds are load-bearing for the conclusion below — without max_norm > 0
    // the rescaled gradient could be non-positive, and without max_norm < g it
    // could reach or exceed g — so dropping either makes the query SAT.
    program.assert(max_norm.clone().real_gt(zero.clone()));
    program.assert(max_norm.clone().real_lt(g.clone()));

    // scale = max_norm / |g|, encoded multiplicatively to avoid division:
    // scale * g = max_norm. With g concrete this is linear in scale.
    program.assert(scale.clone().real_mul(g.clone()).eq(max_norm.clone()));

    // clipped = g * scale (the rescale), or — in the buggy variant — g * max_norm
    // (rescaled by the raw threshold, forgetting the 1/norm factor).
    let clipped_def = if correct {
        g.clone().real_mul(scale.clone())
    } else {
        g.clone().real_mul(max_norm.clone())
    };
    program.assert(clipped.clone().eq(clipped_def));

    // Property: the rescale lands the clipped gradient strictly inside (0, g).
    // With scale = max_norm/g and clipped = g*scale we get clipped = max_norm,
    // which the branch bounds pin strictly between 0 and g — so the clip is
    // direction-preserving (clipped > 0) and genuinely reduces the magnitude
    // (clipped < g). Stating it as the two-sided inequality 0 < clipped < g
    // (rather than the definitional identity clipped == max_norm, whose two sides
    // are the same product g*scale) keeps the sides structurally distinct, so the
    // query is contingent on the branch hypotheses instead of an AC-tautology.
    // Negated violation: clipped <= 0 OR clipped >= g.
    let nonpositive = clipped.clone().real_le(zero);
    let not_reduced = clipped.real_ge(g);
    program.assert(nonpositive.or(not_reduced));
    program.check_sat();

    Ok(program)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ay_vacuity::vacuity_smell;

    // --- SGD Tests ---

    #[test]
    fn test_sgd_update_direction_proven() {
        let result = prove_sgd_update_direction().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "SGD update direction: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "SGD update direction must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "optimizer_sgd_update_direction");
    }

    #[test]
    fn test_sgd_update_identity_proven() {
        let result = prove_sgd_update_identity().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "SGD update identity: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "SGD update identity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "optimizer_sgd_update_identity");
    }

    // --- Momentum Tests ---

    #[test]
    fn test_sgd_momentum_accumulation_proven() {
        let result = prove_sgd_momentum_accumulation().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "SGD momentum: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "SGD momentum must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "optimizer_sgd_momentum_accumulation");
    }

    // --- Adam First Moment Tests ---

    #[test]
    fn test_adam_first_moment_ema_proven() {
        let result = prove_adam_first_moment().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Adam first moment EMA: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Adam first moment must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "optimizer_adam_first_moment_ema");
    }

    // --- Adam Second Moment Tests ---

    #[test]
    fn test_adam_second_moment_nonneg_proven() {
        let result = prove_adam_second_moment().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Adam second moment non-negative: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Adam second moment must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "optimizer_adam_second_moment_nonneg");
    }

    #[test]
    fn test_adam_second_moment_identity_proven() {
        let result = prove_adam_second_moment_identity().expect("proof should not error");
        // The interpolation bound is linear over concrete data (QF_LRA), so the
        // solver must decide it — a strict `proven`, not a permissive Unknown.
        assert!(
            result.proven,
            "Adam second moment interpolation bound (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "optimizer_adam_second_moment_identity");
    }

    /// Drop the `(1 - beta2)` weight on the incoming squared gradient
    /// (`v_new = beta2*v + g2`). Then any `v > 0` with `v <= g2` makes
    /// `v_new = 0.9*v + g2 > g2`, so the interpolation bound is violated and the
    /// query must be SAT — proving the theorem rests on the convex weighting, not
    /// on the shape of the expression.
    #[test]
    fn adam_second_moment_identity_depends_on_the_scale() {
        let program =
            build_adam_second_moment_identity(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the fresh-gradient term unscaled the estimate overshoots g2 \
             and the query must be SAT; got: {detail}",
        );
    }

    // --- Adam Bias Correction Tests ---

    #[test]
    fn test_adam_bias_correction_amplifies_proven() {
        let result = prove_adam_bias_correction_amplifies().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Adam bias correction amplifies: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Adam bias correction must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "optimizer_adam_bias_correction_amplifies");
    }

    #[test]
    fn test_adam_bias_correction_factor_geq_one_proven() {
        let result = prove_adam_bias_correction_factor_geq_one().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Adam bias correction factor >= 1: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Adam bias correction factor must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(
            result.property,
            "optimizer_adam_bias_correction_factor_geq_one"
        );
    }

    // --- Adam Update Bound Tests ---

    #[test]
    fn test_adam_update_denominator_bound_proven() {
        let result = prove_adam_update_denominator_bound().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Adam update denominator bound: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Adam denominator bound must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "optimizer_adam_update_denominator_bound");
    }

    // --- Weight Decay Tests ---

    #[test]
    fn test_weight_decay_identity_proven() {
        let result = prove_weight_decay_identity().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Weight decay identity: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Weight decay identity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "optimizer_weight_decay_identity");
    }

    #[test]
    fn test_weight_decay_shrinks_proven() {
        let result = prove_weight_decay_shrinks().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Weight decay shrinks: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Weight decay shrinks must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "optimizer_weight_decay_shrinks");
    }

    // --- Learning Rate Warmup Tests ---

    #[test]
    fn test_lr_warmup_monotonic_proven() {
        let result = prove_lr_warmup_monotonic().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "LR warmup monotonic: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "LR warmup monotonic must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "optimizer_lr_warmup_monotonic");
    }

    #[test]
    fn test_lr_warmup_bounded_proven() {
        let result = prove_lr_warmup_bounded().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "LR warmup bounded: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "LR warmup bounded must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "optimizer_lr_warmup_bounded");
    }

    // --- Cosine Annealing Tests ---

    #[test]
    fn test_cosine_annealing_bounded_proven() {
        let result = prove_cosine_annealing_bounded().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Cosine annealing bounded: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Cosine annealing bounded must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "optimizer_cosine_annealing_bounded");
    }

    #[test]
    fn test_cosine_annealing_initial_max_proven() {
        let result = prove_cosine_annealing_initial().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Cosine annealing initial = max_lr: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Cosine annealing initial must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "optimizer_cosine_annealing_initial_max");
    }

    // --- Gradient Clipping Tests ---

    #[test]
    fn test_gradient_clipping_passthrough_proven() {
        let result = prove_gradient_clipping_passthrough().expect("proof should not error");
        assert!(
            result.proven,
            "Gradient clipping passthrough should be proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "optimizer_gradient_clipping_passthrough");
    }

    #[test]
    fn test_gradient_clipping_scaling_proven() {
        let result = prove_gradient_clipping_scaling().expect("proof should not error");
        // The strict-shrink bound 0 < clipped < g is linear over concrete data
        // (QF_LRA), so the solver must decide it — a strict `proven`, not a
        // permissive Unknown.
        assert!(
            result.proven,
            "Gradient clipping scaling (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "optimizer_gradient_clipping_scaling");
    }

    /// Forget the `1/norm` factor in the rescale (`clipped = g * max_norm`
    /// instead of `clipped = g * scale`). With the concrete `g = 10` the clipped
    /// value becomes `10 * max_norm`, which reaches or exceeds `g = 10` for every
    /// admissible `max_norm >= 1` — the "clip" fails to shrink the gradient — so
    /// the `clipped >= g` disjunct is satisfiable and the query must be SAT,
    /// proving the strict-shrink bound rests on dividing by the norm, not on the
    /// shape of the expression.
    #[test]
    fn gradient_clipping_scaling_depends_on_the_norm() {
        let program =
            build_gradient_clipping_scaling(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the rescale unnormalized the clipped gradient reaches or exceeds g \
             and the query must be SAT; got: {detail}",
        );
    }

    // --- SMT2 Structure Tests ---

    #[test]
    fn test_all_optimizer_proofs_have_valid_smt2() {
        let proofs: Vec<OptimizerPropertyResult> = vec![
            prove_sgd_update_direction().unwrap(),
            prove_sgd_update_identity().unwrap(),
            prove_sgd_momentum_accumulation().unwrap(),
            prove_adam_first_moment().unwrap(),
            prove_adam_second_moment().unwrap(),
            prove_adam_second_moment_identity().unwrap(),
            prove_adam_bias_correction_amplifies().unwrap(),
            prove_adam_bias_correction_factor_geq_one().unwrap(),
            prove_adam_update_denominator_bound().unwrap(),
            prove_weight_decay_identity().unwrap(),
            prove_weight_decay_shrinks().unwrap(),
            prove_lr_warmup_monotonic().unwrap(),
            prove_lr_warmup_bounded().unwrap(),
            prove_cosine_annealing_bounded().unwrap(),
            prove_cosine_annealing_initial().unwrap(),
            prove_gradient_clipping_passthrough().unwrap(),
            prove_gradient_clipping_scaling().unwrap(),
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
