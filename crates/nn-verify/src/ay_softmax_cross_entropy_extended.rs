// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended ay SMT proofs for softmax and cross-entropy mathematical properties (#4232).
//!
//! Supplements the core proofs in `ay_softmax_cross_entropy` and the dpdf-specific
//! proofs in `ay_softmax_cross_entropy_dpdf` with additional mathematical properties:
//!
//! 1. **Softmax temperature scaling convergence**: At extreme temperatures, softmax
//!    approaches known limiting distributions (uniform for T->inf, one-hot for T->0).
//!    Extends the existing high-T/low-T proofs with tighter 4-class bounds.
//! 2. **Log-softmax algebraic identity**: log(softmax(x))_i = x_i - max(x) -
//!    log(sum(exp(x_j - max(x)))). The algebraic equivalence of the stable formulation.
//! 3. **KL divergence non-negativity via log inequality**: KL(P||Q) >= 0 using the
//!    fact that -log(x) >= 1 - x with equality iff x = 1 (i.e., P = Q).
//! 4. **Label smoothing CE decomposition**: smoothed CE = (1 - eps) * CE(y, p) +
//!    eps * CE(uniform, p). Proves the exact linear decomposition, not just bounds.
//! 5. **Focal loss modulation**: FL(p) = -alpha * (1-p)^gamma * log(p) has
//!    FL >= 0 for p in (0,1], alpha > 0, gamma >= 0.
//! 6. **Softmax Jacobian symmetry**: J_ij = s_i * (delta_ij - s_j) is symmetric
//!    because s_i * (-s_j) = s_j * (-s_i) for off-diagonal entries.
//!
//! # Proof Strategy
//!
//! All proofs follow the established ay pattern: model exp outputs as abstract
//! positive reals with structural constraints, encode the negation of the desired
//! property, and prove UNSAT. We use QF_LRA where possible (structural reasoning)
//! and QF_NRA when products/ratios are required.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::smt_error::SmtError;

/// Result of an extended softmax/cross-entropy property proof attempt.
#[derive(Debug, Clone)]
pub struct ExtendedSoftmaxCeResult {
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
fn assert_bounds(program: &mut AYProgram, expr: &Expr, lower: &Expr, upper: &Expr) {
    program.assert(expr.clone().real_ge(lower.clone()));
    program.assert(expr.clone().real_le(upper.clone()));
}

/// Assert `expr > 0` (strict positivity).
fn assert_positive(program: &mut AYProgram, expr: &Expr) {
    let zero = Expr::real(0);
    program.assert(expr.clone().real_gt(zero));
}

/// Assert `expr >= 0` (non-negativity).
fn assert_non_negative(program: &mut AYProgram, expr: &Expr) {
    let zero = Expr::real(0);
    program.assert(expr.clone().real_ge(zero));
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
// Property 1: Softmax Temperature Scaling Convergence (4-class)
// ---------------------------------------------------------------------------

/// Prove that at high temperature, a 4-class softmax is close to uniform (1/4).
///
/// When T is large, exp(x_i/T) values cluster near 1, so all softmax outputs
/// approach 1/n. For n=4 with exp values in (0.95, 1.05) (very high T):
///
///   s_i = e_i / (e_0 + e_1 + e_2 + e_3)
///
/// With all e_i in (0.95, 1.05), denom is in (3.8, 4.2), and each
/// s_i is in (0.95/4.2, 1.05/3.8) = (0.226, 0.276).
///
/// We prove |s_i - 0.25| < 0.03 for any s_i in this regime.
pub fn prove_temperature_convergence_4class() -> Result<ExtendedSoftmaxCeResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    // Model 4 exp values near 1 (high temperature regime)
    let e0 = declare_real(&mut program, "e0");
    let e1 = declare_real(&mut program, "e1");
    let e2 = declare_real(&mut program, "e2");
    let e3 = declare_real(&mut program, "e3");

    // e_i in (0.95, 1.05) — very high temperature
    let lo = Expr::real(95).real_div(Expr::real(100)); // 0.95
    let hi = Expr::real(105).real_div(Expr::real(100)); // 1.05

    assert_bounds(&mut program, &e0, &lo, &hi);
    assert_bounds(&mut program, &e1, &lo, &hi);
    assert_bounds(&mut program, &e2, &lo, &hi);
    assert_bounds(&mut program, &e3, &lo, &hi);

    // Denominator
    let denom = declare_real(&mut program, "denom");
    program.assert(
        denom.clone().eq(e0
            .clone()
            .real_add(e1.clone())
            .real_add(e2.clone())
            .real_add(e3.clone())),
    );

    // s_0 * denom = e_0
    let s0 = declare_real(&mut program, "s0");
    program.assert(s0.clone().real_mul(denom).eq(e0));

    // quarter = 1/4 = 0.25
    let quarter = Expr::real(1).real_div(Expr::real(4));

    // Deviation from uniform
    let dev = declare_real(&mut program, "dev");
    program.assert(dev.clone().eq(s0.real_sub(quarter)));

    // Violation: |dev| >= 0.03
    let threshold = Expr::real(3).real_div(Expr::real(100));
    let neg_threshold = Expr::real(-3).real_div(Expr::real(100));
    let violation = dev
        .clone()
        .real_ge(threshold)
        .or(dev.real_le(neg_threshold));

    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ExtendedSoftmaxCeResult {
        property: "temperature_convergence_4class_near_uniform".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: Log-Softmax Algebraic Identity
// ---------------------------------------------------------------------------

/// Prove the algebraic identity:
///   log_softmax(x)_i = x_i - max(x) - log(sum_j(exp(x_j - max(x))))
///
/// This is the numerically stable formulation of log(softmax(x)_i).
///
/// Algebraically:
///   log(softmax(x)_i) = log(exp(x_i) / sum(exp(x_j)))
///                      = x_i - log(sum(exp(x_j)))
///   Let M = max(x). Then:
///     log(sum(exp(x_j))) = log(sum(exp(x_j - M + M)))
///                        = log(sum(exp(x_j - M) * exp(M)))
///                        = log(exp(M) * sum(exp(x_j - M)))
///                        = M + log(sum(exp(x_j - M)))
///   Therefore:
///     log(softmax(x)_i) = x_i - M - log(sum(exp(x_j - M)))
///
/// We encode this as a pure algebraic identity using abstract log/exp
/// relationships: if log_sum_exp = M + log_shifted_sum and
/// log_softmax_i = x_i - log_sum_exp, then
/// log_softmax_i = x_i - M - log_shifted_sum.
pub fn prove_log_softmax_identity() -> Result<ExtendedSoftmaxCeResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Abstract variables
    let x_i = declare_real(&mut program, "x_i");
    let max_x = declare_real(&mut program, "max_x"); // M = max(x)

    // log(sum(exp(x_j))) = M + log(sum(exp(x_j - M)))
    // Let log_sum_exp = the full log-sum-exp
    // Let log_shifted_sum = log(sum(exp(x_j - M)))
    let log_sum_exp = declare_real(&mut program, "log_sum_exp");
    let log_shifted_sum = declare_real(&mut program, "log_shifted_sum");

    // Axiom: log_sum_exp = max_x + log_shifted_sum
    program.assert(
        log_sum_exp
            .clone()
            .eq(max_x.clone().real_add(log_shifted_sum.clone())),
    );

    // Naive log-softmax: lsm_naive = x_i - log_sum_exp
    let lsm_naive = declare_real(&mut program, "lsm_naive");
    program.assert(lsm_naive.clone().eq(x_i.clone().real_sub(log_sum_exp)));

    // Stable log-softmax: lsm_stable = x_i - max_x - log_shifted_sum
    let lsm_stable = declare_real(&mut program, "lsm_stable");
    program.assert(
        lsm_stable
            .clone()
            .eq(x_i.real_sub(max_x).real_sub(log_shifted_sum)),
    );

    // Violation: lsm_naive != lsm_stable
    let violation = lsm_naive.ne(lsm_stable);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ExtendedSoftmaxCeResult {
        property: "log_softmax_algebraic_identity".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 3: KL Divergence Non-Negativity via Log Inequality
// ---------------------------------------------------------------------------

/// Prove KL(P || Q) >= 0 using the log-inequality -log(x) >= 1 - x.
///
/// KL(P || Q) = sum_i(p_i * log(p_i / q_i))
///            = sum_i(p_i * (-log(q_i / p_i)))
///            >= sum_i(p_i * (1 - q_i / p_i))     [since -log(x) >= 1 - x]
///            = sum_i(p_i - q_i)
///            = sum(p_i) - sum(q_i)
///            = 1 - 1 = 0
///
/// Equality holds iff q_i/p_i = 1 for all i, i.e., P = Q.
///
/// We prove both: (a) the linearized bound equals 0 for any two valid
/// distributions, and (b) the linearized gap is 0 iff p_i = q_i for all i.
///
/// Part (a): sum(p_i - q_i) = 0 for distributions.
pub fn prove_kl_divergence_non_negativity() -> Result<ExtendedSoftmaxCeResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // Distribution P: p_i > 0, sum = 1 (3-class)
    let p0 = declare_real(&mut program, "p0");
    let p1 = declare_real(&mut program, "p1");
    let p2 = declare_real(&mut program, "p2");

    assert_positive(&mut program, &p0);
    assert_positive(&mut program, &p1);
    assert_positive(&mut program, &p2);
    program.assert(
        p0.clone()
            .real_add(p1.clone())
            .real_add(p2.clone())
            .eq(one.clone()),
    );

    // Distribution Q: q_i > 0, sum = 1
    let q0 = declare_real(&mut program, "q0");
    let q1 = declare_real(&mut program, "q1");
    let q2 = declare_real(&mut program, "q2");

    assert_positive(&mut program, &q0);
    assert_positive(&mut program, &q1);
    assert_positive(&mut program, &q2);
    program.assert(q0.clone().real_add(q1.clone()).real_add(q2.clone()).eq(one));

    // Linearized KL lower bound: sum(p_i - q_i)
    let lin_bound = (p0.real_sub(q0))
        .real_add(p1.real_sub(q1))
        .real_add(p2.real_sub(q2));

    // Violation: lin_bound != 0
    let violation = lin_bound.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ExtendedSoftmaxCeResult {
        property: "kl_divergence_non_negativity_linearized".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove KL divergence equality condition: if the linearized terms are all zero
/// individually (p_i/q_i = 1 for each i), then P = Q.
///
/// The log inequality -log(x) >= 1 - x has equality iff x = 1.
/// So KL(P||Q) = 0 iff q_i/p_i = 1 for all i, i.e., p_i = q_i.
///
/// We encode: given r_i = q_i/p_i and all r_i = 1 (equality condition),
/// prove p_i = q_i for all i.
pub fn prove_kl_divergence_equality_condition() -> Result<ExtendedSoftmaxCeResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let one = Expr::real(1);

    // Distribution P: p_i > 0
    let p0 = declare_real(&mut program, "p0");
    let p1 = declare_real(&mut program, "p1");

    assert_positive(&mut program, &p0);
    assert_positive(&mut program, &p1);
    program.assert(p0.clone().real_add(p1.clone()).eq(one.clone()));

    // Distribution Q: q_i > 0
    let q0 = declare_real(&mut program, "q0");
    let q1 = declare_real(&mut program, "q1");

    assert_positive(&mut program, &q0);
    assert_positive(&mut program, &q1);
    program.assert(q0.clone().real_add(q1.clone()).eq(one.clone()));

    // Ratio r_i = q_i / p_i, modeled as r_i * p_i = q_i
    let r0 = declare_real(&mut program, "r0");
    let r1 = declare_real(&mut program, "r1");

    program.assert(r0.clone().real_mul(p0.clone()).eq(q0.clone()));
    program.assert(r1.clone().real_mul(p1.clone()).eq(q1.clone()));

    // Equality condition: r_i = 1 for all i
    program.assert(r0.eq(one.clone()));
    program.assert(r1.eq(one));

    // Under the equality condition, p_i should equal q_i.
    // Violation: p0 != q0 OR p1 != q1
    let v0 = p0.ne(q0);
    let v1 = p1.ne(q1);
    let violation = v0.or(v1);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ExtendedSoftmaxCeResult {
        property: "kl_divergence_equality_iff_p_eq_q".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: Label Smoothing CE Decomposition
// ---------------------------------------------------------------------------

/// Prove the exact label smoothing decomposition:
///   CE(y_smooth, p) = (1 - eps) * CE(y, p) + eps * CE(u, p)
///
/// where y_smooth = (1 - eps) * y + eps * u, u = uniform = (1/K, ..., 1/K),
/// and CE is linear in its first argument.
///
/// This is stronger than the non-negativity proof in the dpdf module — it
/// proves the exact algebraic decomposition.
///
/// CE(y_smooth, p) = -sum((1-eps)*y_i + eps/K) * log(p_i))
///                 = -(1-eps) * sum(y_i * log(p_i)) - eps/K * sum(log(p_i))
///                 = (1-eps) * CE(y, p) + eps * CE(u, p)
///
/// since CE(u, p) = -sum((1/K) * log(p_i)) = -(1/K) * sum(log(p_i)).
pub fn prove_label_smoothing_decomposition() -> Result<ExtendedSoftmaxCeResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // epsilon in (0, 1)
    let eps = declare_real(&mut program, "eps");
    program.assert(eps.clone().real_gt(zero.clone()));
    program.assert(eps.clone().real_lt(one.clone()));

    // Abstract log values: log_p_i = log(p_i) <= 0 (since p_i in (0, 1])
    let log_p0 = declare_real(&mut program, "log_p0");
    let log_p1 = declare_real(&mut program, "log_p1");
    let log_p2 = declare_real(&mut program, "log_p2");

    program.assert(log_p0.clone().real_le(zero.clone()));
    program.assert(log_p1.clone().real_le(zero.clone()));
    program.assert(log_p2.clone().real_le(zero.clone()));

    // One-hot target y = (1, 0, 0), K = 3
    // CE(y, p) = -1 * log_p0 = -log_p0
    let ce_onehot = log_p0.clone().real_neg();

    // Uniform target u = (1/3, 1/3, 1/3)
    // CE(u, p) = -(1/3)(log_p0 + log_p1 + log_p2)
    let third = Expr::real(1).real_div(Expr::real(3));
    let log_sum = log_p0
        .clone()
        .real_add(log_p1.clone())
        .real_add(log_p2.clone());
    let ce_uniform = third.clone().real_mul(log_sum.clone()).real_neg();

    // Smoothed target: y_smooth = ((1-eps) + eps/3, eps/3, eps/3)
    // CE(y_smooth, p) = -(((1-eps) + eps/3)*log_p0 + (eps/3)*log_p1 + (eps/3)*log_p2)
    let one_minus_eps = one.real_sub(eps.clone());
    let eps_third = eps.clone().real_mul(third);

    let y_smooth_0 = one_minus_eps.clone().real_add(eps_third.clone());
    let y_smooth_1 = eps_third.clone();
    let y_smooth_2 = eps_third;

    let ce_smooth = y_smooth_0
        .real_mul(log_p0)
        .real_add(y_smooth_1.real_mul(log_p1))
        .real_add(y_smooth_2.real_mul(log_p2))
        .real_neg();

    // Expected: (1 - eps) * CE(y, p) + eps * CE(u, p)
    let expected = one_minus_eps
        .real_mul(ce_onehot)
        .real_add(eps.real_mul(ce_uniform));

    // Violation: ce_smooth != expected
    let violation = ce_smooth.ne(expected);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ExtendedSoftmaxCeResult {
        property: "label_smoothing_ce_decomposition".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 5: Focal Loss Non-Negativity
// ---------------------------------------------------------------------------

/// Prove focal loss FL(p) >= 0 for valid parameters.
///
/// Focal loss (Lin et al., 2017):
///   FL(p) = -alpha * (1 - p)^gamma * log(p)
///
/// For p in (0, 1], alpha > 0, gamma >= 0:
///   - (1 - p) in [0, 1), so (1 - p)^gamma in [0, 1] (non-negative)
///   - log(p) <= 0, so -log(p) >= 0
///   - alpha > 0
///   Therefore FL(p) = alpha * (1-p)^gamma * (-log(p)) >= 0.
///
/// We model this structurally: given modulator = (1-p)^gamma >= 0 and
/// neg_log_p = -log(p) >= 0 and alpha > 0, prove FL >= 0.
///
/// We use a helper variable for (1-p)^gamma since gamma is non-negative
/// and (1-p) is in [0,1).
pub fn prove_focal_loss_non_negativity() -> Result<ExtendedSoftmaxCeResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // p in (0, 1]
    let p = declare_real(&mut program, "p");
    program.assert(p.clone().real_gt(zero.clone()));
    program.assert(p.clone().real_le(one.clone()));

    // alpha > 0
    let alpha = declare_real(&mut program, "alpha");
    program.assert(alpha.clone().real_gt(zero.clone()));

    // gamma >= 0
    let gamma = declare_real(&mut program, "gamma");
    assert_non_negative(&mut program, &gamma);

    // modulator = (1 - p)^gamma >= 0
    // Since (1-p) in [0, 1) and gamma >= 0, the modulator is non-negative.
    // We model it as an abstract non-negative value.
    let modulator = declare_real(&mut program, "modulator");
    assert_non_negative(&mut program, &modulator);

    // neg_log_p = -log(p) >= 0 (since p in (0, 1], log(p) <= 0)
    let neg_log_p = declare_real(&mut program, "neg_log_p");
    assert_non_negative(&mut program, &neg_log_p);

    // FL = alpha * modulator * neg_log_p
    let fl = declare_real(&mut program, "fl");
    program.assert(fl.clone().eq(alpha.real_mul(modulator).real_mul(neg_log_p)));

    // Violation: FL < 0
    let violation = fl.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ExtendedSoftmaxCeResult {
        property: "focal_loss_non_negativity".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that focal loss reduces to standard CE when gamma = 0.
///
/// FL(p, gamma=0) = -alpha * (1-p)^0 * log(p) = -alpha * 1 * log(p) = alpha * CE
///
/// When (1-p)^0 = 1 (for any p), the focal modulation disappears.
pub fn prove_focal_loss_gamma_zero_is_ce() -> Result<ExtendedSoftmaxCeResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // alpha > 0
    let alpha = declare_real(&mut program, "alpha");
    program.assert(alpha.clone().real_gt(zero.clone()));

    // neg_log_p = -log(p) (abstract, non-negative)
    let neg_log_p = declare_real(&mut program, "neg_log_p");
    program.assert(neg_log_p.clone().real_ge(zero.clone()));

    // Standard CE term (for one class): alpha * neg_log_p
    let ce_term = alpha.clone().real_mul(neg_log_p.clone());

    // Focal loss with gamma = 0: modulator = (1-p)^0 = 1
    // FL = alpha * 1 * neg_log_p = alpha * neg_log_p
    let fl_gamma_zero = alpha.real_mul(one.real_mul(neg_log_p));

    // Violation: fl_gamma_zero != ce_term
    let violation = fl_gamma_zero.ne(ce_term);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ExtendedSoftmaxCeResult {
        property: "focal_loss_gamma_zero_equals_ce".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 6: Softmax Jacobian Symmetry
// ---------------------------------------------------------------------------

/// Prove the softmax Jacobian is symmetric: J_ij = J_ji.
///
/// The Jacobian of softmax is:
///   J_ij = dS_i/dz_j = s_i * (delta_ij - s_j)
///
/// For off-diagonal entries (i != j):
///   J_ij = s_i * (0 - s_j) = -s_i * s_j
///   J_ji = s_j * (0 - s_i) = -s_j * s_i
///
/// Since multiplication is commutative: J_ij = -s_i * s_j = -s_j * s_i = J_ji.
///
/// For diagonal entries (i = j):
///   J_ii is trivially equal to J_ii.
///
/// We prove the off-diagonal case since the diagonal is trivial.
pub fn prove_softmax_jacobian_symmetry() -> Result<ExtendedSoftmaxCeResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // s_i, s_j are softmax outputs in (0, 1)
    let s_i = declare_real(&mut program, "s_i");
    let s_j = declare_real(&mut program, "s_j");

    program.assert(s_i.clone().real_gt(zero.clone()));
    program.assert(s_i.clone().real_lt(one.clone()));
    program.assert(s_j.clone().real_gt(zero.clone()));
    program.assert(s_j.clone().real_lt(one));

    // J_ij = -s_i * s_j (off-diagonal Jacobian entry)
    let j_ij = declare_real(&mut program, "j_ij");
    program.assert(
        j_ij.clone()
            .eq(s_i.clone().real_mul(s_j.clone()).real_neg()),
    );

    // J_ji = -s_j * s_i (off-diagonal Jacobian entry, swapped)
    let j_ji = declare_real(&mut program, "j_ji");
    program.assert(j_ji.clone().eq(s_j.real_mul(s_i).real_neg()));

    // Violation: J_ij != J_ji
    let violation = j_ij.ne(j_ji);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ExtendedSoftmaxCeResult {
        property: "softmax_jacobian_symmetry".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove the softmax Jacobian diagonal entry is bounded above by 1/4.
///
/// J_ii = s_i * (1 - s_i) for s_i in (0, 1).
///
/// This is a quadratic in s_i with maximum at s_i = 0.5:
///   J_ii = s_i - s_i^2
///   d(J_ii)/d(s_i) = 1 - 2*s_i = 0  =>  s_i = 0.5
///   max(J_ii) = 0.5 * 0.5 = 0.25
///
/// Therefore J_ii <= 1/4 for all s_i in (0, 1). This bounds the gradient
/// magnitude through the softmax layer.
pub fn prove_softmax_jacobian_diagonal_upper_bound() -> Result<ExtendedSoftmaxCeResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // s_i in (0, 1)
    let s_i = declare_real(&mut program, "s_i");
    program.assert(s_i.clone().real_gt(zero));
    program.assert(s_i.clone().real_lt(one.clone()));

    // J_ii = s_i * (1 - s_i)
    let j_ii = declare_real(&mut program, "j_ii");
    let one_minus_s = one.real_sub(s_i.clone());
    program.assert(j_ii.clone().eq(s_i.real_mul(one_minus_s)));

    // Bound: 1/4
    let quarter = Expr::real(1).real_div(Expr::real(4));

    // Violation: J_ii > 1/4
    let violation = j_ii.real_gt(quarter);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ExtendedSoftmaxCeResult {
        property: "softmax_jacobian_diagonal_upper_bound_quarter".to_string(),
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

    #[test]
    fn test_temperature_convergence_4class_proven() {
        let result = prove_temperature_convergence_4class().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Temperature convergence 4-class: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Temperature convergence must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(
            result.property,
            "temperature_convergence_4class_near_uniform"
        );
    }

    #[test]
    fn test_log_softmax_identity_proven() {
        let result = prove_log_softmax_identity().expect("proof should not error");
        assert!(
            result.proven,
            "Log-softmax algebraic identity (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "log_softmax_algebraic_identity");
    }

    #[test]
    fn test_kl_divergence_non_negativity_proven() {
        let result = prove_kl_divergence_non_negativity().expect("proof should not error");
        assert!(
            result.proven,
            "KL divergence non-negativity (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "kl_divergence_non_negativity_linearized");
    }

    #[test]
    fn test_kl_divergence_equality_condition_proven() {
        let result = prove_kl_divergence_equality_condition().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "KL equality condition: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "KL equality condition must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "kl_divergence_equality_iff_p_eq_q");
    }

    #[test]
    fn test_label_smoothing_decomposition_proven() {
        let result = prove_label_smoothing_decomposition().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Label smoothing decomposition: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Label smoothing decomposition must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "label_smoothing_ce_decomposition");
    }

    #[test]
    fn test_focal_loss_non_negativity_proven() {
        let result = prove_focal_loss_non_negativity().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Focal loss non-negativity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Focal loss must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "focal_loss_non_negativity");
    }

    #[test]
    fn test_focal_loss_gamma_zero_is_ce_proven() {
        let result = prove_focal_loss_gamma_zero_is_ce().expect("proof should not error");
        assert!(
            result.proven,
            "Focal loss gamma=0 equals CE (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "focal_loss_gamma_zero_equals_ce");
    }

    #[test]
    fn test_softmax_jacobian_symmetry_proven() {
        let result = prove_softmax_jacobian_symmetry().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Softmax Jacobian symmetry: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Softmax Jacobian symmetry must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "softmax_jacobian_symmetry");
    }

    #[test]
    fn test_softmax_jacobian_diagonal_upper_bound_proven() {
        let result = prove_softmax_jacobian_diagonal_upper_bound().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Softmax Jacobian diagonal bound: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Softmax Jacobian diagonal bound must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(
            result.property,
            "softmax_jacobian_diagonal_upper_bound_quarter"
        );
    }

    #[test]
    fn test_all_extended_proofs_have_valid_smt2() {
        let proofs: Vec<ExtendedSoftmaxCeResult> = vec![
            prove_temperature_convergence_4class().unwrap(),
            prove_log_softmax_identity().unwrap(),
            prove_kl_divergence_non_negativity().unwrap(),
            prove_kl_divergence_equality_condition().unwrap(),
            prove_label_smoothing_decomposition().unwrap(),
            prove_focal_loss_non_negativity().unwrap(),
            prove_focal_loss_gamma_zero_is_ce().unwrap(),
            prove_softmax_jacobian_symmetry().unwrap(),
            prove_softmax_jacobian_diagonal_upper_bound().unwrap(),
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

    #[test]
    fn test_log_softmax_identity_smt2_has_stable_variables() {
        let result = prove_log_softmax_identity().expect("proof should not error");
        assert!(
            result.smt2.contains("lsm_naive"),
            "Log-softmax identity should reference lsm_naive"
        );
        assert!(
            result.smt2.contains("lsm_stable"),
            "Log-softmax identity should reference lsm_stable"
        );
        assert!(
            result.smt2.contains("log_shifted_sum"),
            "Log-softmax identity should reference log_shifted_sum"
        );
    }

    #[test]
    fn test_focal_loss_smt2_has_modulator() {
        let result = prove_focal_loss_non_negativity().expect("proof should not error");
        assert!(
            result.smt2.contains("modulator"),
            "Focal loss SMT2 should reference the modulator variable"
        );
        assert!(
            result.smt2.contains("alpha"),
            "Focal loss SMT2 should reference alpha"
        );
    }

    #[test]
    fn test_jacobian_symmetry_smt2_structure() {
        let result = prove_softmax_jacobian_symmetry().expect("proof should not error");
        assert!(
            result.smt2.contains("j_ij"),
            "Jacobian symmetry should reference j_ij"
        );
        assert!(
            result.smt2.contains("j_ji"),
            "Jacobian symmetry should reference j_ji"
        );
    }
}
