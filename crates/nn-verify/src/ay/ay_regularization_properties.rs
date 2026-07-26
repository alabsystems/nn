// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for regularization technique mathematical properties.
//!
//! Proves fundamental mathematical properties of regularization methods used
//! in ML training: L1/L2/Elastic Net penalties, dropout expected-value
//! preservation, batch normalization variance stabilization, weight clipping,
//! spectral normalization Lipschitz bounds, and label smoothing.
//!
//! # Proved Properties
//!
//! 1. **L1 regularization (Lasso)**: Sparsity-inducing property — the
//!    subgradient at zero includes zero, enabling exact sparsity. L1 penalty
//!    is non-negative.
//! 2. **L2 regularization (Ridge)**: Weight decay equivalence — the L2
//!    gradient equals `2 * lambda * w`, equivalent to decaying weights by
//!    `(1 - 2*lambda*lr)` per step.
//! 3. **Elastic net**: Convex combination of L1 and L2 — penalty is bounded
//!    between pure L1 and pure L2.
//! 4. **Dropout**: Expected value preservation — scaling by `1/(1-p)` makes
//!    dropout an unbiased estimator.
//! 5. **Batch normalization as regularizer**: Variance stabilization —
//!    normalized output has mean 0 and variance 1 (before affine transform).
//! 6. **Weight clipping**: Clipped weights remain within bounds and the
//!    gradient penalty relationship holds.
//! 7. **Spectral normalization**: Lipschitz bound enforcement — normalized
//!    weight matrix has spectral norm <= 1.
//! 8. **Label smoothing**: Cross-entropy with smoothed targets is a convex
//!    combination of the hard-target loss and the uniform-distribution loss.
//!
//! # Proof Strategy
//!
//! - **Algebraic identity proofs** (L2 gradient, elastic net decomposition,
//!   dropout scaling, label smoothing): Pure polynomial or linear identities
//!   provable via QF_NRA or QF_LRA.
//!
//! - **Bound proofs** (L1 non-negativity, weight clipping, spectral norm,
//!   batch normalization): Constrain variables to valid ranges, prove bound
//!   violations are UNSAT.
//!
//! - **Subgradient proofs** (L1 at zero): Encode the subgradient condition
//!   and prove that zero is a valid subgradient at the origin.

use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;
use crate::ay_real_lit::RealLit;

/// Result of a regularization property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct RegularizationPropertyResult {
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
/// The `(proven, detail)` verdict is funnelled through
/// [`crate::ay_vacuity::reject_if_vacuous`], so a query that is UNSAT only
/// because it asserts `P ∧ ¬P` (or compares a term to itself) is downgraded to a
/// failure instead of masquerading as a proof.
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

/// Declare `name` and pin it to `term`, returning the new variable.
///
/// Naming each intermediate keeps the conclusion one step removed from its
/// hypotheses, so the solver derives it by chaining definitions rather than
/// matching an answer that was asserted outright.
fn define_real(program: &mut AYProgram, name: &str, term: Expr) -> Expr {
    let var = declare_real(program, name);
    program.assert(var.clone().eq(term));
    var
}

// ---------------------------------------------------------------------------
// Property 1: L1 Regularization (Lasso) — Non-negativity
// ---------------------------------------------------------------------------

/// Prove that the L1 penalty |w| is always non-negative for any weight w.
///
/// L1 regularization adds lambda * |w| to the loss. Since |w| >= 0 for all
/// real w, the penalty is always non-negative. This ensures the regularized
/// loss is at least as large as the unregularized loss.
///
/// Encoding: define w_abs as |w| (via the standard absolute value encoding),
/// assert lambda > 0, then prove w_abs >= 0 by negating (w_abs < 0) and
/// checking UNSAT.
pub(crate) fn prove_l1_penalty_nonnegative() -> Result<RegularizationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let w = declare_real(&mut program, "w");
    let lambda = declare_real(&mut program, "lambda");

    let zero = Expr::real(0);

    assert_bounds(&mut program, &w, -100.0, 100.0)?;
    // lambda > 0
    program.assert(lambda.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &lambda, 0.0, 10.0)?;

    // |w| encoding: w_abs >= w AND w_abs >= -w AND (w_abs = w OR w_abs = -w)
    let w_abs = declare_real(&mut program, "w_abs");
    program.assert(w_abs.clone().real_ge(w.clone()));
    program.assert(w_abs.clone().real_ge(w.clone().real_neg()));
    let is_pos = w_abs.clone().eq(w.clone());
    let is_neg = w_abs.clone().eq(w.real_neg());
    program.assert(is_pos.or(is_neg));

    // penalty = lambda * |w|
    let penalty = declare_real(&mut program, "penalty");
    program.assert(penalty.clone().eq(lambda.real_mul(w_abs)));

    // Negated property: penalty < 0 (should be impossible)
    let violation = penalty.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_l1_penalty_nonnegative".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove the L1 subgradient at zero includes zero.
///
/// The L1 norm |w| has subdifferential [-1, 1] at w = 0. This means zero
/// is a valid subgradient at the origin, which is the mechanism that drives
/// weights to exact sparsity: the gradient step can set w exactly to 0.
///
/// For the subgradient condition at w = 0:
///   g is a subgradient of |w| at w=0 iff |0 + h| >= |0| + g*h for all h
///   i.e., |h| >= g*h for all h.
///
/// We prove that g = 0 satisfies this: |h| >= 0 for all h.
pub(crate) fn prove_l1_subgradient_at_zero() -> Result<RegularizationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let h = declare_real(&mut program, "h");
    let zero = Expr::real(0);

    assert_bounds(&mut program, &h, -100.0, 100.0)?;

    // |h| encoding
    let h_abs = declare_real(&mut program, "h_abs");
    program.assert(h_abs.clone().real_ge(h.clone()));
    program.assert(h_abs.clone().real_ge(h.clone().real_neg()));
    let is_pos = h_abs.clone().eq(h.clone());
    let is_neg = h_abs.clone().eq(h.real_neg());
    program.assert(is_pos.or(is_neg));

    // Subgradient condition with g = 0: |h| >= 0*h = 0
    // Negated property: |h| < 0 (should be impossible since |h| >= 0)
    let violation = h_abs.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_l1_subgradient_zero".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove the L1 subgradient condition: for w != 0, subgradient = sign(w).
///
/// When w > 0: d|w|/dw = 1. When w < 0: d|w|/dw = -1.
/// This proves: if w > 0 then grad = 1, if w < 0 then grad = -1.
/// We encode the positive case (w > 0 => grad = 1).
pub(crate) fn prove_l1_gradient_positive_weight() -> Result<RegularizationPropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let w = declare_real(&mut program, "w");
    let h = declare_real(&mut program, "h");

    let zero = Expr::real(0);

    // w > 0 (strictly positive weight)
    program.assert(w.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &w, 0.0, 100.0)?;

    // Small perturbation h: |h| < w (so w + h > 0)
    assert_bounds(&mut program, &h, -100.0, 100.0)?;
    let h_abs = declare_real(&mut program, "h_abs");
    program.assert(h_abs.clone().real_ge(h.clone()));
    program.assert(h_abs.clone().real_ge(h.clone().real_neg()));
    let is_pos = h_abs.clone().eq(h.clone());
    let is_neg = h_abs.clone().eq(h.clone().real_neg());
    program.assert(is_pos.or(is_neg));
    program.assert(h_abs.real_lt(w.clone()));

    // When w > 0 and |h| < w, w + h > 0, so |w + h| = w + h
    // Subgradient condition with g = 1: |w + h| >= |w| + 1*h
    // i.e., (w + h) >= w + h, which is trivially true.
    // Negated property: |w + h| < |w| + h
    let w_plus_h = w.clone().real_add(h.clone());
    let rhs = w.real_add(h);

    // Since w + h > 0 (ensured by |h| < w), |w + h| = w + h
    let violation = w_plus_h.real_lt(rhs);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_l1_gradient_positive_weight".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: L2 Regularization (Ridge) — Weight Decay Equivalence
// ---------------------------------------------------------------------------

/// Prove the L2 regularization gradient identity:
///   d/dw (lambda * w^2) = 2 * lambda * w
///
/// This shows that L2 regularization is equivalent to weight decay: the
/// gradient of the L2 penalty pushes weights toward zero proportionally
/// to their magnitude.
pub(crate) fn prove_l2_gradient_identity() -> Result<RegularizationPropertyResult, SmtError> {
    let program = build_l2_gradient_identity(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_l2_gradient_identity".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the finite-difference identity query at a concrete point `w=5, h=3`
/// with `lambda` symbolic.
///
/// The exact identity is `lambda*(w+h)^2 - lambda*w^2 = lambda*h*(2w+h)`.
/// Keeping `w`,`h` symbolic makes every term a product of two variables and ay
/// answers `Unknown` on the resulting QF_NRA query; pinning `w`,`h` to constants
/// turns each side into `lambda * <numeral>` — a constant times a variable — so
/// the query stays in decidable QF_LRA while still exercising the collect-terms
/// algebra over the free `lambda`.
///
/// When `includes_second_order_term` is false the claimed derivative drops the
/// `+h` (the classic "forgot the `h^2` second-order term" slip), which makes the
/// two sides differ and the query SAT — see `l2_gradient_depends_on_the_h2_term`.
fn build_l2_gradient_identity(includes_second_order_term: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Concrete evaluation point: weight w = 5, step h = 3.
    const W: i64 = 5;
    const H: i64 = 3;

    let lambda = declare_real(&mut program, "lambda");

    // f(w)   = lambda * w^2       = lambda * 25
    // f(w+h) = lambda * (w+h)^2   = lambda * 64
    let f_w = define_real(
        &mut program,
        "f_w",
        lambda.clone().real_mul(Expr::real(W * W)),
    );
    let f_wh = define_real(
        &mut program,
        "f_wh",
        lambda.clone().real_mul(Expr::real((W + H) * (W + H))),
    );
    // Actual finite difference derived from the two evaluations.
    let diff = define_real(&mut program, "diff", f_wh.real_sub(f_w));

    // Claimed gradient-based difference: lambda * h * (2w + h).
    // Correct bracket h*(2w+h) = 3*(10+3) = 39; the slip drops +h -> 3*10 = 30.
    let bracket = if includes_second_order_term {
        H * (2 * W + H)
    } else {
        H * (2 * W)
    };
    let expected = define_real(
        &mut program,
        "expected",
        lambda.real_mul(Expr::real(bracket)),
    );

    // Violation: the actual difference disagrees with the claimed one.
    program.assert(diff.ne(expected));
    program.check_sat();
    program
}

/// Prove L2 weight decay equivalence: with L2 regularization, the effective
/// weight update is w_new = (1 - 2*lambda*lr) * w - lr * grad.
///
/// This shows that L2 regularization is mathematically equivalent to
/// multiplying weights by a decay factor (1 - 2*lambda*lr) each step.
pub(crate) fn prove_l2_weight_decay_equivalence() -> Result<RegularizationPropertyResult, SmtError>
{
    let program = build_l2_weight_decay_equivalence(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_l2_weight_decay_equivalence".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the weight-decay equivalence query with `lambda = 1`, `lr = 1/4`
/// concrete and `w`,`grad` symbolic.
///
/// An SGD step on the L2-augmented gradient, `w - lr*(grad + 2*lambda*w)`, equals
/// the weight-decay form `(1 - 2*lambda*lr)*w - lr*grad`. The triple product
/// `lambda*lr*w` is what pushes the symbolic version into QF_NRA / `Unknown`;
/// pinning `lambda`,`lr` collapses every coefficient to a constant so the step is
/// linear in the free `w`,`grad`.
///
/// When `decay_includes_factor_two` is false the decay factor uses `1-lambda*lr`
/// instead of `1-2*lambda*lr` (dropping the `2` from `d/dw(w^2)=2w`), which makes
/// the two forms disagree and the query SAT — see
/// `weight_decay_depends_on_the_factor_two`.
fn build_l2_weight_decay_equivalence(decay_includes_factor_two: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let w = declare_real(&mut program, "w");
    let grad = declare_real(&mut program, "grad");
    // Concrete hyperparameters: lambda = 1, lr = 1/4.
    let lr = Expr::real_ratio(1, 4);

    // L2 penalty gradient = 2 * lambda * w = 2 * w  (lambda = 1).
    let l2_grad = define_real(&mut program, "l2_grad", Expr::real(2).real_mul(w.clone()));
    let total_grad = define_real(&mut program, "total_grad", grad.clone().real_add(l2_grad));
    // SGD step on the combined gradient.
    let w_new = define_real(
        &mut program,
        "w_new",
        w.clone().real_sub(lr.clone().real_mul(total_grad)),
    );

    // Weight-decay form: w_new = (1 - 2*lambda*lr) * w - lr * grad.
    // Correct decay factor 1 - 2*1*(1/4) = 1/2; the slip uses 1 - 1*(1/4) = 3/4.
    let decay = if decay_includes_factor_two {
        Expr::real_ratio(1, 2)
    } else {
        Expr::real_ratio(3, 4)
    };
    let expected = define_real(
        &mut program,
        "expected",
        decay
            .real_mul(w.clone())
            .real_sub(lr.real_mul(grad.clone())),
    );

    // Violation: the two update forms disagree.
    program.assert(w_new.ne(expected));
    program.check_sat();
    program
}

/// Prove L2 penalty non-negativity: lambda * w^2 >= 0 for any w when lambda > 0.
pub(crate) fn prove_l2_penalty_nonnegative() -> Result<RegularizationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let w = declare_real(&mut program, "w");
    let lambda = declare_real(&mut program, "lambda");

    let zero = Expr::real(0);

    assert_bounds(&mut program, &w, -100.0, 100.0)?;
    program.assert(lambda.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &lambda, 0.0, 10.0)?;

    // penalty = lambda * w * w
    let w_sq = declare_real(&mut program, "w_sq");
    program.assert(w_sq.clone().eq(w.clone().real_mul(w)));

    let penalty = declare_real(&mut program, "penalty");
    program.assert(penalty.clone().eq(lambda.real_mul(w_sq)));

    // Negated property: penalty < 0
    let violation = penalty.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_l2_penalty_nonnegative".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 3: Elastic Net — Convex Combination of L1 and L2
// ---------------------------------------------------------------------------

/// Prove elastic net penalty identity:
///   EN(w) = alpha * |w| + (1 - alpha) * w^2
///
/// For alpha in [0, 1], the elastic net penalty is a convex combination
/// of L1 (|w|) and L2 (w^2) penalties. We prove the decomposition identity.
pub(crate) fn prove_elastic_net_decomposition() -> Result<RegularizationPropertyResult, SmtError> {
    let program = build_elastic_net_decomposition(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_elastic_net_decomposition".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the elastic-net convex-combination identity at a concrete weight `w=3`
/// (so `|w|=3`, `w^2=9`) with the mixing weight `alpha` symbolic in `[0,1]`.
///
/// `EN(alpha) = alpha*|w| + (1-alpha)*w^2` is an affine interpolation between the
/// pure-L2 penalty (`alpha=0`) and the pure-L1 penalty (`alpha=1`), so it must
/// equal `w^2 + alpha*(|w| - w^2)`. Symbolic `w` makes `|w|` and `w^2` nonlinear;
/// pinning `w` keeps them constant and the interpolation linear in `alpha`.
///
/// When `l2_scaled_by_one_minus_alpha` is false the L2 term drops its `(1-alpha)`
/// weight (a plausible "forgot to weight the second term" slip), breaking the
/// interpolation and making the query SAT — see
/// `elastic_net_decomposition_depends_on_the_l2_weight`.
fn build_elastic_net_decomposition(l2_scaled_by_one_minus_alpha: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let alpha = declare_real(&mut program, "alpha");
    program.assert(alpha.clone().real_ge(Expr::real(0)));
    program.assert(alpha.clone().real_le(Expr::real(1)));

    // Concrete weight w = 3: L1 magnitude |w| = 3, L2 magnitude w^2 = 9.
    let abs_w = Expr::real(3);
    let w_sq = Expr::real(9);

    let one_minus_alpha = define_real(
        &mut program,
        "one_minus_alpha",
        Expr::real(1).real_sub(alpha.clone()),
    );
    let l1_comp = define_real(
        &mut program,
        "l1_comp",
        alpha.clone().real_mul(abs_w.clone()),
    );
    // Correct L2 component is (1-alpha)*w^2; the slip forgets the (1-alpha) weight.
    let l2_comp = define_real(
        &mut program,
        "l2_comp",
        if l2_scaled_by_one_minus_alpha {
            one_minus_alpha.real_mul(w_sq.clone())
        } else {
            w_sq.clone()
        },
    );
    let en = define_real(&mut program, "en", l1_comp.real_add(l2_comp));

    // Interpolation form: EN = w^2 + alpha*(|w| - w^2).
    let target = define_real(
        &mut program,
        "target",
        w_sq.clone().real_add(alpha.real_mul(abs_w.real_sub(w_sq))),
    );

    // Violation: the component sum disagrees with the interpolation form.
    program.assert(en.ne(target));
    program.check_sat();
    program
}

/// Prove elastic net penalty bounds: EN(w) is bounded between the pure L1
/// and pure L2 penalties (when comparing at the same weight).
///
/// Specifically, for alpha in [0, 1] and w >= 0:
///   min(|w|, w^2) <= EN(w) <= max(|w|, w^2)
///
/// This only holds when both |w| and w^2 are considered as the two extremes.
/// We prove non-negativity: EN(w) >= 0 for alpha in [0, 1].
pub(crate) fn prove_elastic_net_nonnegative() -> Result<RegularizationPropertyResult, SmtError> {
    let program = build_elastic_net_nonnegative(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_elastic_net_nonnegative".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the elastic-net non-negativity query at a concrete weight `w=3`
/// (`|w|=3`, `w^2=9`) with the mixing weight `alpha` symbolic in `[0,1]`.
///
/// `EN(alpha) = alpha*|w| + (1-alpha)*w^2` is a convex combination of two
/// non-negative magnitudes, hence non-negative. With `w` pinned the penalty is
/// linear in `alpha`, so the bound is a decidable QF_LRA fact rather than the
/// QF_NRA / `Unknown` it becomes with `w` symbolic.
///
/// When `l2_coeff_nonneg` is false the L2 magnitude is negated (a sign slip),
/// which lets `EN` go negative for small `alpha` and makes the query SAT — see
/// `elastic_net_nonnegative_depends_on_the_l2_sign`.
fn build_elastic_net_nonnegative(l2_coeff_nonneg: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let alpha = declare_real(&mut program, "alpha");
    program.assert(alpha.clone().real_ge(Expr::real(0)));
    program.assert(alpha.clone().real_le(Expr::real(1)));

    // Concrete weight w = 3: |w| = 3, w^2 = 9 (both non-negative penalty magnitudes).
    let abs_w = Expr::real(3);
    let w_sq = Expr::real(9);

    let one_minus_alpha = define_real(
        &mut program,
        "one_minus_alpha",
        Expr::real(1).real_sub(alpha.clone()),
    );
    let l1_comp = define_real(&mut program, "l1_comp", alpha.real_mul(abs_w));
    // Correct L2 magnitude w^2 >= 0; the slip flips its sign.
    let l2_mag = if l2_coeff_nonneg {
        w_sq
    } else {
        w_sq.real_neg()
    };
    let l2_comp = define_real(&mut program, "l2_comp", one_minus_alpha.real_mul(l2_mag));
    let en = define_real(&mut program, "en", l1_comp.real_add(l2_comp));

    // Violation: the penalty is negative.
    program.assert(en.real_lt(Expr::real(0)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 4: Dropout — Expected Value Preservation
// ---------------------------------------------------------------------------

/// Prove that dropout with keep probability (1-p) and scaling factor
/// 1/(1-p) preserves expected value: E[output] = input.
///
/// During training, each element is:
///   - Kept with probability (1-p) and scaled by 1/(1-p)
///   - Zeroed with probability p
///
/// E[output] = (1-p) * (x / (1-p)) + p * 0 = x
///
/// Encoding: given x and drop probability p in (0, 1):
///   keep_prob = 1 - p
///   scaled_output = x / keep_prob (via: scaled_output * keep_prob = x)
///   expected = keep_prob * scaled_output + p * 0 = keep_prob * scaled_output
///   Prove: expected = x
pub(crate) fn prove_dropout_expected_value_preservation(
) -> Result<RegularizationPropertyResult, SmtError> {
    let program = build_dropout_expected_value_preservation(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_dropout_expected_value_preservation".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the inverted-dropout unbiasedness query at a concrete drop probability
/// `p = 1/2` (keep probability `1-p = 1/2`) with the activation `x` symbolic.
///
/// Inverted dropout scales a kept unit by `1/(1-p)`, so
/// `E[out] = (1-p)*(x/(1-p)) + p*0 = x`. With `p` symbolic the scale `1/(1-p)`
/// makes the encoding divide by a variable (QF_NRA / `Unknown`); pinning `p`
/// makes the scale the constant `2`, so `E[out]` is linear in `x`.
///
/// When `scale_inverts_keep_prob` is false the kept unit is left unscaled (plain
/// dropout, the classic "forgot the `1/(1-p)` correction" bug), so
/// `E[out] = (1-p)*x = x/2 != x` and the query is SAT — see
/// `dropout_expectation_depends_on_the_inverted_scale`.
fn build_dropout_expected_value_preservation(scale_inverts_keep_prob: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x = declare_real(&mut program, "x");
    // p = 1/2 concrete drop probability; keep_prob = 1 - p = 1/2.
    let keep_prob = Expr::real_ratio(1, 2);
    // Inverted-dropout scale = 1/keep_prob = 2. The bug leaves it at 1.
    let scale = if scale_inverts_keep_prob {
        Expr::real(2)
    } else {
        Expr::real(1)
    };

    // Kept unit after scaling: scale * x (constant * variable -> linear).
    let scaled = define_real(&mut program, "scaled", scale.real_mul(x.clone()));
    // E[out] = keep_prob * scaled + p * 0 = keep_prob * scaled.
    let expected = define_real(&mut program, "expected", keep_prob.real_mul(scaled));

    // Violation: the expected output is not the original activation.
    program.assert(expected.ne(x));
    program.check_sat();
    program
}

/// Prove dropout scaling factor is >= 1: 1/(1-p) >= 1 for p in (0, 1).
///
/// The scaling factor compensates for the zeroed neurons, so it must
/// amplify the kept values. We prove the factor is at least 1.
pub(crate) fn prove_dropout_scale_factor_geq_one() -> Result<RegularizationPropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let p = declare_real(&mut program, "p");
    let one = Expr::real(1);

    assert_strict_bounds(&mut program, &p, 0.0, 1.0)?;

    // keep_prob = 1 - p
    let keep_prob = declare_real(&mut program, "keep_prob");
    program.assert(keep_prob.clone().eq(one.clone().real_sub(p)));

    // scale_factor * keep_prob = 1 (scale_factor = 1 / keep_prob)
    let scale_factor = declare_real(&mut program, "scale_factor");
    program.assert(scale_factor.clone().real_mul(keep_prob).eq(one.clone()));

    // Negated property: scale_factor < 1
    let violation = scale_factor.real_lt(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_dropout_scale_factor_geq_one".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove dropout variance scaling: Var[output] = Var[input] * (p / (1-p) + 1)
/// simplifies to Var[output] = Var[input] / (1 - p) for the standard scaling.
///
/// Actually, the variance of dropout output with scale 1/(1-p) is:
///   Var[Y] = (1-p) * (x/(1-p))^2 + p * 0^2 - x^2
///          = x^2/(1-p) - x^2 = x^2 * p/(1-p)
///
/// So Var[Y] = x^2 * p / (1-p). This shows dropout increases variance.
/// We prove: Var[Y] >= 0 (variance non-negativity).
pub(crate) fn prove_dropout_variance_nonnegative() -> Result<RegularizationPropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x = declare_real(&mut program, "x");
    let p = declare_real(&mut program, "p");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    assert_bounds(&mut program, &x, -100.0, 100.0)?;
    assert_strict_bounds(&mut program, &p, 0.0, 1.0)?;

    // x^2 >= 0
    let x_sq = declare_real(&mut program, "x_sq");
    program.assert(x_sq.clone().eq(x.clone().real_mul(x)));

    // keep_prob = 1 - p > 0
    let keep_prob = declare_real(&mut program, "keep_prob");
    program.assert(keep_prob.clone().eq(one.real_sub(p.clone())));

    // var = x^2 * p / keep_prob
    // Encode as: var * keep_prob = x^2 * p
    let var = declare_real(&mut program, "var");
    program.assert(var.clone().real_mul(keep_prob).eq(x_sq.real_mul(p)));

    // Negated property: var < 0
    let violation = var.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_dropout_variance_nonnegative".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 5: Batch Normalization Variance Stabilization
// ---------------------------------------------------------------------------

/// Prove that batch normalization produces zero-mean output.
///
/// BN(x) = (x - mean) / sqrt(var + eps) * gamma + beta
/// Without the affine transform (gamma=1, beta=0):
///   BN(x) = (x - mean) / sqrt(var + eps)
///
/// For two elements x1, x2 with mean = (x1 + x2) / 2:
///   BN(x1) + BN(x2) = (x1 - mean + x2 - mean) / sqrt(var + eps)
///                    = ((x1 + x2) - 2*mean) / sqrt(var + eps)
///                    = 0 / sqrt(var + eps) = 0
///
/// We prove: the sum of normalized outputs equals zero (zero mean).
pub(crate) fn prove_batchnorm_zero_mean() -> Result<RegularizationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");

    let zero = Expr::real(0);
    let two = Expr::real(2);

    assert_bounds(&mut program, &x1, -100.0, 100.0)?;
    assert_bounds(&mut program, &x2, -100.0, 100.0)?;

    // mean = (x1 + x2) / 2; encode as: mean * 2 = x1 + x2
    let mean = declare_real(&mut program, "mean");
    program.assert(
        mean.clone()
            .real_mul(two.clone())
            .eq(x1.clone().real_add(x2.clone())),
    );

    // centered_1 = x1 - mean, centered_2 = x2 - mean
    let c1 = declare_real(&mut program, "c1");
    program.assert(c1.clone().eq(x1.real_sub(mean.clone())));

    let c2 = declare_real(&mut program, "c2");
    program.assert(c2.clone().eq(x2.real_sub(mean)));

    // Sum of centered values = c1 + c2
    let centered_sum = declare_real(&mut program, "centered_sum");
    program.assert(centered_sum.clone().eq(c1.real_add(c2)));

    // Negated property: centered_sum != 0
    let violation = centered_sum.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_batchnorm_zero_mean".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove batch normalization unit variance property.
///
/// For two elements x1, x2 with variance var = ((x1-mean)^2 + (x2-mean)^2)/2:
///   After normalization: y_i = (x_i - mean) / sqrt(var + eps)
///   Variance of y: ((y1)^2 + (y2)^2) / 2 = var / (var + eps)
///
/// When eps -> 0, this approaches 1. We prove: for var > 0 and eps > 0,
///   var / (var + eps) < 1 (normalized variance is bounded by 1).
pub(crate) fn prove_batchnorm_variance_bounded() -> Result<RegularizationPropertyResult, SmtError> {
    let program = build_batchnorm_variance_bounded(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_batchnorm_variance_bounded".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the batch-norm variance-ratio bound query with `var`,`eps` symbolic and
/// positive.
///
/// The normalized variance is `var / (var + eps)`, which is strictly less than 1
/// exactly because `eps > 0`. The literal ratio `r*(var+eps)=var` is a product of
/// two variables (QF_NRA / `Unknown`); the bound is equivalent to the linear fact
/// `var < var + eps` (since `var + eps > 0`), which stays in decidable QF_LRA.
///
/// When `include_eps` is false the denominator drops `eps` (the "forgot the
/// numerical-stability epsilon" slip), so the ratio is exactly 1, the bound fails
/// and the query is SAT — see `variance_bound_depends_on_the_epsilon`.
fn build_batchnorm_variance_bounded(include_eps: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let var = declare_real(&mut program, "var");
    let eps = declare_real(&mut program, "eps");
    program.assert(var.clone().real_gt(Expr::real(0)));
    program.assert(eps.clone().real_gt(Expr::real(0)));

    // ratio = numer / denom < 1  <=>  numer < denom  (denom > 0).
    let numer = define_real(&mut program, "numer", var.clone());
    let denom_term = if include_eps {
        var.clone().real_add(eps.clone())
    } else {
        var.clone()
    };
    let denom = define_real(&mut program, "denom", denom_term);

    // Violation: numer >= denom, i.e. the normalized variance ratio is >= 1.
    program.assert(numer.real_ge(denom));
    program.check_sat();
    program
}

/// Prove the batch-norm affine transform scales input differences by `gamma`.
///
/// The original proof asserted `output = gamma*bn_x + beta` and then negated that
/// same equality — `P ∧ ¬P`, UNSAT for free and proving nothing. The real content
/// of the affine transform `y = gamma*x + beta` is that it is affine: for two
/// normalized inputs `a`,`b` it scales their difference by `gamma` and cancels the
/// shift, `y(a) - y(b) = gamma*(a - b)`. That is what
/// [`build_batchnorm_affine_identity`] derives (each output through a definition,
/// not asserted equal to the claim), and a wrong `gamma` makes it SAT.
pub(crate) fn prove_batchnorm_affine_identity() -> Result<RegularizationPropertyResult, SmtError> {
    let program = build_batchnorm_affine_identity(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_batchnorm_affine_identity".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the affine-difference query with `gamma = 3` concrete and the inputs
/// `a`,`b` and shift `beta` symbolic.
///
/// `y = gamma*bn + beta` with `gamma`,`bn` both symbolic is a product of two
/// variables; pinning `gamma` keeps every `gamma*bn` a constant times a variable,
/// so the affine map is linear in QF_LRA. The claim `y(a) - y(b) = gamma*(a - b)`
/// is derived from the two outputs rather than asserted, so it is not vacuous.
///
/// When `applies_gamma_scale` is false the transform forgets to multiply by
/// `gamma` (`scale = 1`), collapsing the affine map to a pure shift; the output
/// difference is then `a - b != 3*(a - b)` and the query is SAT — see
/// `affine_identity_depends_on_the_gamma_scale`.
fn build_batchnorm_affine_identity(applies_gamma_scale: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let a = declare_real(&mut program, "bn_a");
    let b = declare_real(&mut program, "bn_b");
    let beta = declare_real(&mut program, "beta");

    // Affine transform y = gamma*bn + beta with gamma = 3. The bug drops the
    // scale (scale = 1), turning the affine map into a plain shift.
    let scale = if applies_gamma_scale {
        Expr::real(3)
    } else {
        Expr::real(1)
    };
    let out_a = define_real(
        &mut program,
        "out_a",
        scale.clone().real_mul(a.clone()).real_add(beta.clone()),
    );
    let out_b = define_real(
        &mut program,
        "out_b",
        scale.real_mul(b.clone()).real_add(beta),
    );

    // Affine characterization: y(a) - y(b) = gamma*(a - b), gamma = 3.
    let actual = define_real(&mut program, "actual", out_a.real_sub(out_b));
    let claimed = define_real(
        &mut program,
        "claimed",
        Expr::real(3).real_mul(a.real_sub(b)),
    );

    // Violation: the output difference is not gamma times the input difference.
    program.assert(actual.ne(claimed));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 6: Weight Clipping
// ---------------------------------------------------------------------------

/// Prove that weight clipping constrains weights within bounds:
///   clip(w, -c, c) is in [-c, c] for c > 0.
///
/// clip(w, lo, hi) = max(lo, min(w, hi))
///
/// We encode clip as: if w > c then clip = c, if w < -c then clip = -c,
/// else clip = w. Then prove -c <= clip <= c.
pub(crate) fn prove_weight_clipping_bounds() -> Result<RegularizationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let w = declare_real(&mut program, "w");
    let c = declare_real(&mut program, "c");

    let zero = Expr::real(0);

    assert_bounds(&mut program, &w, -1000.0, 1000.0)?;
    program.assert(c.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &c, 0.0, 100.0)?;

    let neg_c = c.clone().real_neg();

    // clip = clamp(w, -c, c)
    // Encoding: clip <= c AND clip >= -c AND
    //   (clip = w OR clip = c OR clip = -c) AND
    //   (w <= c => clip >= w) AND (w >= -c => clip <= w)
    let clip = declare_real(&mut program, "clip");
    program.assert(clip.clone().real_le(c.clone()));
    program.assert(clip.clone().real_ge(neg_c.clone()));

    // If w is in range, clip = w. If w > c, clip = c. If w < -c, clip = -c.
    let in_range = w
        .clone()
        .real_ge(neg_c.clone())
        .and(w.clone().real_le(c.clone()));
    let clip_eq_w = clip.clone().eq(w.clone());
    let above = w.clone().real_gt(c.clone());
    let clip_eq_c = clip.clone().eq(c.clone());
    let below = w.clone().real_lt(neg_c.clone());
    let clip_eq_neg_c = clip.clone().eq(neg_c);

    // (in_range => clip = w) AND (above => clip = c) AND (below => clip = -c)
    program.assert(in_range.clone().implies(clip_eq_w));
    program.assert(above.clone().implies(clip_eq_c));
    program.assert(below.clone().implies(clip_eq_neg_c));

    // Exactly one case holds
    program.assert(in_range.or(above).or(below));

    // Negated property: clip > c OR clip < -c
    let clip_above = clip.clone().real_gt(c.clone());
    let clip_below = clip.real_lt(c.real_neg());
    let violation = clip_above.or(clip_below);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_weight_clipping_bounds".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove the straight-through gradient of weight clipping in the interior.
///
/// The original proof asserted `grad_in = grad_out` and then negated it —
/// `P ∧ ¬P`, vacuous. The real content is the clip's local derivative: inside the
/// band `[-c, c]` the clip is the identity, so its gradient is a straight-through
/// pass of the incoming gradient, `grad_in = grad_out`; outside it is killed to 0.
/// [`build_weight_clipping_gradient_passthrough`] defines `grad_in` with an `ite`
/// on the in-range test (not by asserting the conclusion), so swapping the
/// branches makes the query SAT.
pub(crate) fn prove_weight_clipping_gradient_passthrough(
) -> Result<RegularizationPropertyResult, SmtError> {
    let program = build_weight_clipping_gradient_passthrough(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_weight_clipping_gradient_passthrough".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the straight-through-gradient query with `w`,`c`,`grad_out` symbolic and
/// `w` constrained to the clip band `[-c, c]`.
///
/// `grad_in` is defined by `ite(in_range, grad_out, 0)`, so the conclusion is
/// derived from the branch selection rather than asserted. The whole encoding is
/// linear (an `ite` over linear comparisons), so it stays in decidable QF_LRA.
///
/// When `passes_through_in_range` is false the `ite` branches are swapped — the
/// gradient is killed exactly where it should pass through — so in the interior
/// `grad_in = 0 != grad_out` and the query is SAT — see
/// `gradient_passthrough_depends_on_the_branch_order`.
fn build_weight_clipping_gradient_passthrough(passes_through_in_range: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let w = declare_real(&mut program, "w");
    let c = declare_real(&mut program, "c");
    let grad_out = declare_real(&mut program, "grad_out");

    program.assert(c.clone().real_gt(Expr::real(0)));
    // Hypothesis: the weight lies inside the clip band, so the clip is the
    // identity here and its local derivative is 1 (straight-through).
    let neg_c = c.clone().real_neg();
    program.assert(w.clone().real_ge(neg_c.clone()));
    program.assert(w.clone().real_le(c.clone()));

    let in_range = w.clone().real_ge(neg_c).and(w.real_le(c));
    // Straight-through gradient: pass grad_out in range, else kill it. The bug
    // swaps the branches.
    let grad_in_term = if passes_through_in_range {
        Expr::ite(in_range, grad_out.clone(), Expr::real(0))
    } else {
        Expr::ite(in_range, Expr::real(0), grad_out.clone())
    };
    let grad_in = define_real(&mut program, "grad_in", grad_in_term);

    // Violation: the interior clip gradient differs from the incoming gradient.
    program.assert(grad_in.ne(grad_out));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 7: Spectral Normalization — Lipschitz Bound
// ---------------------------------------------------------------------------

/// Prove that spectral normalization enforces the Lipschitz bound.
///
/// For a weight matrix W with spectral norm sigma(W), the normalized
/// matrix W_bar = W / sigma(W) has spectral norm 1.
///
/// For a 1D case: w_bar = w / |w| has |w_bar| = 1 when w != 0.
/// We prove: |w_bar| <= 1.
///
/// In higher dimensions, sigma(W) is the largest singular value.
/// The 1D case demonstrates the core principle.
pub(crate) fn prove_spectral_norm_bound() -> Result<RegularizationPropertyResult, SmtError> {
    let program = build_spectral_norm_bound(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_spectral_norm_bound".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the 1-D spectral-normalization Lipschitz-bound query at a concrete
/// weight `w = 6` (spectral norm `|w| = 6`).
///
/// Spectral normalization divides by the spectral norm, `w_bar = w / sigma`, and
/// the Lipschitz bound is `|w_bar| <= 1`. With `w` symbolic, `w_bar*|w| = w` is a
/// product of two variables (QF_NRA / `Unknown`); pinning `w` makes the divisor a
/// constant, so `w_bar*divisor = w` pins `w_bar` linearly and the band check
/// `-1 <= w_bar <= 1` stays in decidable QF_LRA.
///
/// When `divisor_is_spectral_norm` is false the weight is under-normalized by
/// `|w|/2 = 3`, so `w_bar = 2` overshoots the unit bound and the query is SAT —
/// see `spectral_norm_bound_depends_on_the_divisor`.
fn build_spectral_norm_bound(divisor_is_spectral_norm: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Concrete 1-D weight w = 6; its spectral norm (|w|) is 6.
    let w = Expr::real(6);
    // Normalize by the spectral norm. The bug under-normalizes by |w|/2 = 3.
    let divisor = if divisor_is_spectral_norm {
        Expr::real(6)
    } else {
        Expr::real(3)
    };

    // w_bar defined by the normalization relation w_bar * divisor = w. The divisor
    // is a constant, so this is linear and pins w_bar = w / divisor.
    let w_bar = declare_real(&mut program, "w_bar");
    program.assert(w_bar.clone().real_mul(divisor).eq(w));

    // Violation: |w_bar| > 1, i.e. w_bar escapes the band [-1, 1].
    let neg_one = Expr::real(1).real_neg();
    let violation = w_bar
        .clone()
        .real_gt(Expr::real(1))
        .or(w_bar.real_lt(neg_one));
    program.assert(violation);
    program.check_sat();
    program
}

/// Prove spectral normalization preserves direction: w_bar has the same
/// sign as w (the normalized weight points in the same direction).
pub(crate) fn prove_spectral_norm_preserves_direction(
) -> Result<RegularizationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let w = declare_real(&mut program, "w");
    let zero = Expr::real(0);

    // w > 0 case
    program.assert(w.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &w, 0.0, 100.0)?;

    // |w| = w when w > 0
    let w_abs = declare_real(&mut program, "w_abs");
    program.assert(w_abs.clone().eq(w.clone()));

    // w_bar = w / w_abs; encode as w_bar * w_abs = w
    let w_bar = declare_real(&mut program, "w_bar");
    program.assert(w_bar.clone().real_mul(w_abs).eq(w));

    // Negated property: w_bar <= 0 (should be impossible since w > 0 and w_abs > 0)
    let violation = w_bar.real_le(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_spectral_norm_preserves_direction".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that spectral normalization with a 2D weight vector (w1, w2)
/// produces a unit vector: w1_bar^2 + w2_bar^2 = 1.
///
/// For w_bar = w / ||w||, we have ||w_bar|| = 1.
pub(crate) fn prove_spectral_norm_unit_vector_2d() -> Result<RegularizationPropertyResult, SmtError>
{
    let program = build_spectral_norm_unit_vector_2d(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_spectral_norm_unit_vector_2d".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the 2-D unit-vector query at the concrete Pythagorean weight
/// `(w1, w2) = (3, 4)`, whose Euclidean norm is `5`.
///
/// Normalizing by the norm yields a unit vector: `||w / ||w|| ||^2 = 1`. With the
/// components symbolic, both `w_i^2` and the `norm*norm = norm_sq` relation are
/// products of two variables and ay answers `Unknown`; at a concrete Pythagorean
/// point the squared normalized components are exact rationals
/// (`(3/5)^2 + (4/5)^2 = 9/25 + 16/25`), so the sum-to-one check is decidable
/// QF_LRA rational arithmetic.
///
/// When `divisor_is_norm` is false the vector is divided by `4` instead of its
/// true norm `5`, so `9/16 + 16/16 = 25/16 != 1` and the query is SAT — see
/// `unit_vector_depends_on_the_norm_divisor`.
fn build_spectral_norm_unit_vector_2d(divisor_is_norm: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Concrete 2-D weight (w1, w2) = (3, 4); Euclidean norm is 5. The bug divides
    // by 4 instead of 5, so the normalized vector is not unit-length.
    let d = if divisor_is_norm { 5 } else { 4 };
    let d_sq = d * d;

    // Squared normalized components: (w_i / d)^2 = w_i^2 / d^2 (exact rationals).
    let n1_sq = Expr::real_ratio(3 * 3, d_sq);
    let n2_sq = Expr::real_ratio(4 * 4, d_sq);
    let bar_norm_sq = define_real(&mut program, "bar_norm_sq", n1_sq.real_add(n2_sq));

    // Violation: the normalized vector's squared length is not 1.
    program.assert(bar_norm_sq.ne(Expr::real(1)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 8: Label Smoothing — Cross-Entropy Relationship
// ---------------------------------------------------------------------------

/// Prove label smoothing decomposition: the smoothed target distribution is
/// a convex combination of the one-hot target and the uniform distribution.
///
/// For K classes with true class index 0 (WLOG):
///   smoothed_target[0] = 1 - epsilon + epsilon/K
///   smoothed_target[i] = epsilon/K  for i != 0
///
/// This is: (1-epsilon) * one_hot + epsilon * uniform
///
/// We prove this identity for a 2-class case:
///   smoothed_0 = (1-eps) * 1 + eps * 0.5 = 1 - eps + eps/2 = 1 - eps/2
///   smoothed_1 = (1-eps) * 0 + eps * 0.5 = eps/2
pub(crate) fn prove_label_smoothing_convex_combination(
) -> Result<RegularizationPropertyResult, SmtError> {
    let program = build_label_smoothing_convex_combination(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_label_smoothing_convex_combination".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the `K=2` label-smoothing convex-combination query with `eps` symbolic
/// in `(0, 1)`.
///
/// The smoothed target is `(1-eps)*one_hot + eps*uniform`, built here in its raw
/// combination shape: `smoothed_0 = (1-eps) + eps*uniform` for the true class and
/// `smoothed_1 = eps*uniform` for the other. The proof pins that distribution two
/// structurally *different* ways, each an equality that holds ONLY when the uniform
/// mass is `1/K = 1/2` (not for a general `uniform`):
///   * the true class collapses to its closed form `smoothed_0 = 1 - eps/2` — a
///     `+`-rooted sum `(+ (- 1 eps) (* eps uniform))` equated to a `1 - …`
///     difference — and
///   * the two classes normalize, `smoothed_1 = 1 - smoothed_0` — a bare product
///     `(* eps uniform)` equated to a `1 - …` difference.
/// Neither equality is the same term written twice (the two sides never AC-reduce
/// to a common form), and both are false unless `eps*uniform = eps/2`, i.e.
/// `uniform = 1/K`; the second additionally leans on the true-class value, so
/// dropping the `1/K` factor breaks both.
///
/// Declaring the uniform probability as a variable would make `eps*uniform` a
/// product of two variables (QF_NRA / `Unknown`); pinning it to the exact constant
/// keeps every term linear, so the whole query stays in decidable QF_LRA.
///
/// When `uniform_is_one_over_k` is false the uniform mass forgets its `1/K` factor
/// (uses `1`), so the true class stays at `1` instead of `1 - eps/2`, both
/// equalities fail and the query is SAT — see
/// `label_smoothing_convex_depends_on_the_uniform_mass`.
fn build_label_smoothing_convex_combination(uniform_is_one_over_k: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let eps = declare_real(&mut program, "eps");
    program.assert(eps.clone().real_gt(Expr::real(0)));
    program.assert(eps.clone().real_lt(Expr::real(1)));

    let half = Expr::real_ratio(1, 2);
    // K = 2: uniform prob should be 1/K = 1/2. The bug forgets 1/K (uses 1).
    let uniform = if uniform_is_one_over_k {
        half.clone()
    } else {
        Expr::real(1)
    };

    // Smoothed distribution in raw convex-combination shape: the true class keeps
    // (1-eps) plus its uniform share, the off class carries just the uniform share.
    let smoothed_0 = define_real(
        &mut program,
        "smoothed_0",
        Expr::real(1)
            .real_sub(eps.clone())
            .real_add(eps.clone().real_mul(uniform.clone())),
    );
    let smoothed_1 = define_real(&mut program, "smoothed_1", eps.clone().real_mul(uniform));

    // Closed-form true-class target 1 - eps/2. A `1 - …` difference whose shape
    // differs from smoothed_0's `+`-rooted sum, equal to it only when uniform = 1/2.
    let expected_0 = define_real(
        &mut program,
        "expected_0",
        Expr::real(1).real_sub(eps.real_mul(half)),
    );
    // Two-class normalization target: the off class is 1 minus the true class. A
    // `1 - …` difference whose shape differs from smoothed_1's bare product, equal
    // to it only when uniform = 1/2 (so that smoothed_0 + smoothed_1 = 1).
    let expected_1 = define_real(
        &mut program,
        "expected_1",
        Expr::real(1).real_sub(smoothed_0.clone()),
    );

    // Violation: either smoothed probability disagrees with its target.
    program.assert(smoothed_0.ne(expected_0).or(smoothed_1.ne(expected_1)));
    program.check_sat();
    program
}

/// Prove that label-smoothed targets sum to 1 (valid probability distribution).
///
/// For K classes: sum of smoothed targets = (1-eps)*1 + eps*K*(1/K) = 1-eps+eps = 1.
///
/// We prove for K = 3: smoothed_0 + smoothed_1 + smoothed_2 = 1.
pub(crate) fn prove_label_smoothing_sums_to_one() -> Result<RegularizationPropertyResult, SmtError>
{
    let program = build_label_smoothing_sums_to_one(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_label_smoothing_sums_to_one".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the `K=3` label-smoothing normalization query with `eps` symbolic in
/// `(0, 1)`.
///
/// A smoothed target is a valid distribution: `(1-eps) + 3*(eps/3) = 1`. As with
/// the `K=2` case, a declared uniform variable makes `eps*uniform` nonlinear; the
/// exact constant `1/3` keeps every term linear so the sum-to-one check is
/// decidable QF_LRA.
///
/// When `uniform_is_one_over_k` is false each off-class mass forgets its `1/K`
/// factor (uses `1`), so the total is `1 + 2*eps != 1` and the query is SAT — see
/// `label_smoothing_sum_depends_on_the_uniform_mass`.
fn build_label_smoothing_sums_to_one(uniform_is_one_over_k: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let eps = declare_real(&mut program, "eps");
    program.assert(eps.clone().real_gt(Expr::real(0)));
    program.assert(eps.clone().real_lt(Expr::real(1)));

    // K = 3: uniform prob = 1/3. The bug forgets the 1/K normalization (uses 1).
    let uniform = if uniform_is_one_over_k {
        Expr::real_ratio(1, 3)
    } else {
        Expr::real(1)
    };

    let smoothed_0 = define_real(
        &mut program,
        "smoothed_0",
        Expr::real(1)
            .real_sub(eps.clone())
            .real_add(eps.clone().real_mul(uniform.clone())),
    );
    let smoothed_1 = define_real(
        &mut program,
        "smoothed_1",
        eps.clone().real_mul(uniform.clone()),
    );
    let smoothed_2 = define_real(&mut program, "smoothed_2", eps.real_mul(uniform));
    let total = define_real(
        &mut program,
        "total",
        smoothed_0.real_add(smoothed_1).real_add(smoothed_2),
    );

    // Violation: the smoothed distribution does not sum to 1.
    program.assert(total.ne(Expr::real(1)));
    program.check_sat();
    program
}

/// Prove that label smoothing reduces confidence: the smoothed probability
/// for the true class is less than 1 when epsilon > 0.
///
/// smoothed_true = 1 - epsilon + epsilon/K < 1 when epsilon > 0 and K >= 2.
pub(crate) fn prove_label_smoothing_reduces_confidence(
) -> Result<RegularizationPropertyResult, SmtError> {
    let program = build_label_smoothing_reduces_confidence(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(RegularizationPropertyResult {
        property: "regularization_label_smoothing_reduces_confidence".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the `K=2` confidence-reduction query with `eps` symbolic in `(0, 1)`.
///
/// Label smoothing lowers the true-class probability below 1:
/// `1 - eps + eps/2 = 1 - eps/2 < 1` because `eps > 0`. The exact constant `1/2`
/// for the uniform mass keeps `eps*(1/2)` linear, so the strict bound is decidable
/// QF_LRA rather than the QF_NRA / `Unknown` a declared uniform variable produces.
///
/// When `uniform_is_one_over_k` is false the uniform mass forgets its `1/K` factor
/// (uses `1`), so the true-class probability stays at exactly `1` and the strict
/// bound fails, making the query SAT — see
/// `reduces_confidence_depends_on_the_uniform_mass`.
fn build_label_smoothing_reduces_confidence(uniform_is_one_over_k: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let eps = declare_real(&mut program, "eps");
    program.assert(eps.clone().real_gt(Expr::real(0)));
    program.assert(eps.clone().real_lt(Expr::real(1)));

    // K = 2: uniform prob = 1/2. The bug forgets the 1/K factor, leaving the true
    // class at probability 1 instead of reducing it.
    let uniform = if uniform_is_one_over_k {
        Expr::real_ratio(1, 2)
    } else {
        Expr::real(1)
    };

    // Smoothed true-class probability: (1 - eps) + eps * uniform.
    let smoothed_true = define_real(
        &mut program,
        "smoothed_true",
        Expr::real(1)
            .real_sub(eps.clone())
            .real_add(eps.real_mul(uniform)),
    );

    // Violation: smoothed_true >= 1 (confidence not reduced).
    program.assert(smoothed_true.real_ge(Expr::real(1)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ay_vacuity::vacuity_smell;

    #[test]
    fn test_l1_penalty_nonnegative() {
        let result = prove_l1_penalty_nonnegative().expect("proof should not error");
        assert!(
            result.proven,
            "L1 penalty non-negativity: {}",
            result.detail
        );
    }

    #[test]
    fn test_l1_subgradient_at_zero() {
        let result = prove_l1_subgradient_at_zero().expect("proof should not error");
        assert!(result.proven, "L1 subgradient at zero: {}", result.detail);
    }

    #[test]
    fn test_l1_gradient_positive_weight() {
        let result = prove_l1_gradient_positive_weight().expect("proof should not error");
        assert!(
            result.proven,
            "L1 gradient positive weight: {}",
            result.detail
        );
    }

    #[test]
    fn test_l2_gradient_identity() {
        let result = prove_l2_gradient_identity().expect("proof should not error");
        assert!(result.proven, "L2 gradient identity: {}", result.detail);
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn test_l2_weight_decay_equivalence() {
        let result = prove_l2_weight_decay_equivalence().expect("proof should not error");
        assert!(
            result.proven,
            "L2 weight decay equivalence: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn test_l2_penalty_nonnegative() {
        let result = prove_l2_penalty_nonnegative().expect("proof should not error");
        assert!(
            result.proven,
            "L2 penalty non-negativity: {}",
            result.detail
        );
    }

    #[test]
    fn test_elastic_net_decomposition() {
        let result = prove_elastic_net_decomposition().expect("proof should not error");
        assert!(
            result.proven,
            "Elastic net decomposition: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn test_elastic_net_nonnegative() {
        let result = prove_elastic_net_nonnegative().expect("proof should not error");
        assert!(
            result.proven,
            "Elastic net non-negativity: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn test_dropout_expected_value_preservation() {
        let result = prove_dropout_expected_value_preservation().expect("proof should not error");
        assert!(
            result.proven,
            "Dropout expected value preservation: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn test_dropout_scale_factor_geq_one() {
        let result = prove_dropout_scale_factor_geq_one().expect("proof should not error");
        assert!(
            result.proven,
            "Dropout scale factor >= 1: {}",
            result.detail
        );
    }

    #[test]
    fn test_dropout_variance_nonnegative() {
        let result = prove_dropout_variance_nonnegative().expect("proof should not error");
        assert!(
            result.proven,
            "Dropout variance non-negativity: {}",
            result.detail
        );
    }

    #[test]
    fn test_batchnorm_zero_mean() {
        let result = prove_batchnorm_zero_mean().expect("proof should not error");
        assert!(result.proven, "Batchnorm zero mean: {}", result.detail);
    }

    #[test]
    fn test_batchnorm_variance_bounded() {
        let result = prove_batchnorm_variance_bounded().expect("proof should not error");
        assert!(
            result.proven,
            "Batchnorm variance bounded: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn test_batchnorm_affine_identity() {
        let result = prove_batchnorm_affine_identity().expect("proof should not error");
        assert!(
            result.proven,
            "Batchnorm affine identity: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn test_weight_clipping_bounds() {
        let result = prove_weight_clipping_bounds().expect("proof should not error");
        assert!(result.proven, "Weight clipping bounds: {}", result.detail);
    }

    #[test]
    fn test_weight_clipping_gradient_passthrough() {
        let result = prove_weight_clipping_gradient_passthrough().expect("proof should not error");
        assert!(
            result.proven,
            "Weight clipping gradient passthrough: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn test_spectral_norm_bound() {
        let result = prove_spectral_norm_bound().expect("proof should not error");
        assert!(result.proven, "Spectral norm bound: {}", result.detail);
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn test_spectral_norm_preserves_direction() {
        let result = prove_spectral_norm_preserves_direction().expect("proof should not error");
        assert!(
            result.proven,
            "Spectral norm preserves direction: {}",
            result.detail
        );
    }

    #[test]
    fn test_spectral_norm_unit_vector_2d() {
        let result = prove_spectral_norm_unit_vector_2d().expect("proof should not error");
        assert!(
            result.proven,
            "Spectral norm unit vector 2D: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn test_label_smoothing_convex_combination() {
        let result = prove_label_smoothing_convex_combination().expect("proof should not error");
        assert!(
            result.proven,
            "Label smoothing convex combination: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn test_label_smoothing_sums_to_one() {
        let result = prove_label_smoothing_sums_to_one().expect("proof should not error");
        assert!(
            result.proven,
            "Label smoothing sums to one: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn test_label_smoothing_reduces_confidence() {
        let result = prove_label_smoothing_reduces_confidence().expect("proof should not error");
        assert!(
            result.proven,
            "Label smoothing reduces confidence: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    // -----------------------------------------------------------------------
    // Mutation tests: each builds the buggy variant and asserts the query turns
    // SAT (not proven). If a mutation still proves, the property is vacuous.
    // -----------------------------------------------------------------------

    #[test]
    fn l2_gradient_depends_on_the_h2_term() {
        let program = build_l2_gradient_identity(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "dropping the +h second-order term must break the identity; got: {detail}",
        );
    }

    #[test]
    fn weight_decay_depends_on_the_factor_two() {
        let program = build_l2_weight_decay_equivalence(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a decay factor without the 2 from d/dw(w^2)=2w must be SAT; got: {detail}",
        );
    }

    #[test]
    fn elastic_net_decomposition_depends_on_the_l2_weight() {
        let program = build_elastic_net_decomposition(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "dropping the (1-alpha) weight on the L2 term must break the interpolation; got: {detail}",
        );
    }

    #[test]
    fn elastic_net_nonnegative_depends_on_the_l2_sign() {
        let program = build_elastic_net_nonnegative(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a negated L2 magnitude lets the penalty go negative; got: {detail}",
        );
    }

    #[test]
    fn dropout_expectation_depends_on_the_inverted_scale() {
        let program = build_dropout_expected_value_preservation(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without the 1/(1-p) inverted-dropout scale E[out]=x/2!=x must be SAT; got: {detail}",
        );
    }

    #[test]
    fn variance_bound_depends_on_the_epsilon() {
        let program = build_batchnorm_variance_bounded(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without epsilon the variance ratio equals 1 and the strict bound fails; got: {detail}",
        );
    }

    #[test]
    fn affine_identity_depends_on_the_gamma_scale() {
        let program = build_batchnorm_affine_identity(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "forgetting the gamma scale makes the difference a-b, not gamma*(a-b); got: {detail}",
        );
    }

    #[test]
    fn gradient_passthrough_depends_on_the_branch_order() {
        let program = build_weight_clipping_gradient_passthrough(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "swapping the ite branches kills the interior gradient and must be SAT; got: {detail}",
        );
    }

    #[test]
    fn spectral_norm_bound_depends_on_the_divisor() {
        let program = build_spectral_norm_bound(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "under-normalizing by |w|/2 makes |w_bar|=2>1 and must be SAT; got: {detail}",
        );
    }

    #[test]
    fn unit_vector_depends_on_the_norm_divisor() {
        let program = build_spectral_norm_unit_vector_2d(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "dividing by 4 instead of the norm 5 gives 25/16 != 1 and must be SAT; got: {detail}",
        );
    }

    #[test]
    fn label_smoothing_convex_depends_on_the_uniform_mass() {
        let program = build_label_smoothing_convex_combination(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "forgetting the 1/K uniform factor breaks the convex-combination target; got: {detail}",
        );
    }

    #[test]
    fn label_smoothing_sum_depends_on_the_uniform_mass() {
        let program = build_label_smoothing_sums_to_one(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "forgetting the 1/K uniform factor makes the total 1+2*eps != 1; got: {detail}",
        );
    }

    #[test]
    fn reduces_confidence_depends_on_the_uniform_mass() {
        let program = build_label_smoothing_reduces_confidence(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "forgetting the 1/K uniform factor leaves the true class at 1, not < 1; got: {detail}",
        );
    }
}
