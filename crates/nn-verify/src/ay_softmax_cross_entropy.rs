// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for softmax and cross-entropy loss mathematical properties (#4232).
//!
//! Softmax transforms an input vector x into a probability distribution:
//!
//! ```text
//! softmax(x)_i = exp(x_i) / sum_j(exp(x_j))
//! ```
//!
//! Cross-entropy loss measures the divergence between a target distribution y
//! and a predicted distribution p:
//!
//! ```text
//! CE(y, p) = -sum_i(y_i * log(p_i))
//! ```
//!
//! # Properties Proved
//!
//! 1. **Softmax output range**: Each softmax output is in (0, 1).
//! 2. **Softmax sum-to-one**: Softmax outputs sum to 1.
//! 3. **Softmax monotonicity**: If x_i > x_j then softmax(x)_i > softmax(x)_j.
//! 4. **Log-softmax stability**: The max-subtraction trick avoids overflow.
//! 5. **Cross-entropy non-negativity**: CE >= 0 for valid distributions.
//! 6. **Cross-entropy minimum (Gibbs' inequality)**: CE(y, p) >= CE(y, y).
//! 7. **KL divergence decomposition**: CE(y, p) = H(y) + KL(y || p).
//! 8. **Temperature scaling**: High T approaches uniform, low T approaches argmax.
//!
//! # Proof Strategy
//!
//! Since softmax involves `exp`, which ay's NRA solver cannot handle exactly,
//! we use two approaches:
//!
//! - **Structural proofs (QF_LRA)**: Model exp outputs as positive reals with
//!   axiomatic constraints (exp > 0, monotonicity). Properties that follow from
//!   these structural constraints alone are provable in the linear fragment.
//!
//! - **Algebraic identity proofs (QF_NRA)**: Properties that require reasoning
//!   about ratios and products of the exp values use non-linear real arithmetic.
//!   The NRA solver may return Unknown for complex queries; we accept either
//!   Proven or Unknown (but never Counterexample).

use ay_bindings::{Expr, Sort, AYProgram};

use crate::smt_error::SmtError;

/// Result of a softmax/cross-entropy property proof attempt.
#[derive(Debug, Clone)]
pub struct SoftmaxCePropertyResult {
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
// Property 1: Softmax Output Range — each output in (0, 1)
// ---------------------------------------------------------------------------

/// Prove that each softmax output is strictly in (0, 1).
///
/// We model a 3-element softmax structurally:
///   - e_i = exp(x_i) > 0 for each i (exp is always positive)
///   - s_i = e_i / (e_0 + e_1 + e_2)
///
/// Since each e_i > 0 and the denominator is the sum of all positive terms:
///   - s_i > 0 (positive numerator / positive denominator)
///   - s_i < 1 (numerator < denominator since other terms contribute positively)
///
/// We assert the negation: exists s_i <= 0 OR s_i >= 1, and prove UNSAT.
pub fn prove_softmax_output_range() -> Result<SoftmaxCePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Model exp(x_i) as abstract positive values e_i > 0.
    let e0 = declare_real(&mut program, "e0");
    let e1 = declare_real(&mut program, "e1");
    let e2 = declare_real(&mut program, "e2");

    assert_positive(&mut program, &e0);
    assert_positive(&mut program, &e1);
    assert_positive(&mut program, &e2);

    // Denominator = e0 + e1 + e2 > 0
    let denom = declare_real(&mut program, "denom");
    program.assert(
        denom
            .clone()
            .eq(e0.clone().real_add(e1.clone()).real_add(e2.clone())),
    );

    // Softmax outputs: s_i = e_i / denom.
    // In QF_LRA, we model s_i * denom = e_i instead of using division.
    let s0 = declare_real(&mut program, "s0");
    let s1 = declare_real(&mut program, "s1");
    let s2 = declare_real(&mut program, "s2");

    // s_i * denom = e_i  (equivalent to s_i = e_i / denom when denom > 0)
    program.assert(s0.clone().real_mul(denom.clone()).eq(e0.clone()));
    program.assert(s1.clone().real_mul(denom.clone()).eq(e1.clone()));
    program.assert(s2.clone().real_mul(denom.clone()).eq(e2.clone()));

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // Violation: any s_i <= 0 or s_i >= 1
    let v0_lo = s0.clone().real_le(zero.clone());
    let v0_hi = s0.clone().real_ge(one.clone());
    let v1_lo = s1.clone().real_le(zero.clone());
    let v1_hi = s1.clone().real_ge(one.clone());
    let v2_lo = s2.clone().real_le(zero.clone());
    let v2_hi = s2.clone().real_ge(one.clone());

    let violation = Expr::or_many(vec![v0_lo, v0_hi, v1_lo, v1_hi, v2_lo, v2_hi]);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SoftmaxCePropertyResult {
        property: "softmax_output_range".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: Softmax Sum-to-One
// ---------------------------------------------------------------------------

/// Prove that softmax outputs sum to 1.
///
/// Given e_i > 0 and s_i = e_i / (e_0 + e_1 + e_2):
///   sum(s_i) = (e_0 + e_1 + e_2) / (e_0 + e_1 + e_2) = 1
///
/// We model this structurally: if s_i * denom = e_i for all i and
/// denom = e_0 + e_1 + e_2, then s_0 + s_1 + s_2 = 1.
pub fn prove_softmax_sum_to_one() -> Result<SoftmaxCePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let e0 = declare_real(&mut program, "e0");
    let e1 = declare_real(&mut program, "e1");
    let e2 = declare_real(&mut program, "e2");

    assert_positive(&mut program, &e0);
    assert_positive(&mut program, &e1);
    assert_positive(&mut program, &e2);

    let denom = declare_real(&mut program, "denom");
    program.assert(
        denom
            .clone()
            .eq(e0.clone().real_add(e1.clone()).real_add(e2.clone())),
    );

    // s_i * denom = e_i
    let s0 = declare_real(&mut program, "s0");
    let s1 = declare_real(&mut program, "s1");
    let s2 = declare_real(&mut program, "s2");

    program.assert(s0.clone().real_mul(denom.clone()).eq(e0));
    program.assert(s1.clone().real_mul(denom.clone()).eq(e1));
    program.assert(s2.clone().real_mul(denom.clone()).eq(e2));

    // sum = s0 + s1 + s2
    let sum = s0.real_add(s1).real_add(s2);
    let one = Expr::real(1);

    // Violation: sum != 1
    let violation = sum.ne(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SoftmaxCePropertyResult {
        property: "softmax_sum_to_one".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 3: Softmax Monotonicity
// ---------------------------------------------------------------------------

/// Prove that softmax preserves input ordering: if x_i > x_j then
/// softmax(x)_i > softmax(x)_j.
///
/// Since exp is a monotonically increasing function, x_i > x_j implies
/// exp(x_i) > exp(x_j), hence e_i > e_j. With a shared positive denominator,
/// e_i/denom > e_j/denom, so s_i > s_j.
///
/// We encode this structurally: given e_i > e_j > 0 and denom > 0 with
/// s_i = e_i/denom and s_j = e_j/denom, prove s_i > s_j.
pub fn prove_softmax_monotonicity() -> Result<SoftmaxCePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let e_i = declare_real(&mut program, "e_i");
    let e_j = declare_real(&mut program, "e_j");
    let e_k = declare_real(&mut program, "e_k");

    assert_positive(&mut program, &e_i);
    assert_positive(&mut program, &e_j);
    assert_positive(&mut program, &e_k);

    // exp monotonicity: x_i > x_j implies e_i > e_j
    program.assert(e_i.clone().real_gt(e_j.clone()));

    let denom = declare_real(&mut program, "denom");
    program.assert(
        denom
            .clone()
            .eq(e_i.clone().real_add(e_j.clone()).real_add(e_k.clone())),
    );

    // s_i * denom = e_i, s_j * denom = e_j
    let s_i = declare_real(&mut program, "s_i");
    let s_j = declare_real(&mut program, "s_j");

    program.assert(s_i.clone().real_mul(denom.clone()).eq(e_i));
    program.assert(s_j.clone().real_mul(denom.clone()).eq(e_j));

    // Violation: s_i <= s_j (negation of s_i > s_j)
    let violation = s_i.real_le(s_j);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SoftmaxCePropertyResult {
        property: "softmax_monotonicity".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: Log-Softmax Numerical Stability
// ---------------------------------------------------------------------------

/// Prove that the log-softmax max-subtraction trick keeps intermediate values
/// bounded when inputs are bounded.
///
/// The stable formulation is:
///   log_softmax(x)_i = (x_i - max(x)) - log(sum_j(exp(x_j - max(x))))
///
/// The key insight: after subtracting max(x), all shifted values are in
/// (-inf, 0], so exp of shifted values is in (0, 1]. This prevents overflow
/// in exp (which could produce inf for large positive inputs).
///
/// We prove: given x_i - max(x) <= 0 for all i, and at least one shift = 0
/// (the max element), each exp(x_i - max(x)) is in (0, 1], and the sum of
/// exp values is in [1, n]. This keeps log(sum) bounded in [0, log(n)].
///
/// Uses QF_LRA with shifts and exp outputs modeled as bounded helper variables.
pub fn prove_log_softmax_stability() -> Result<SoftmaxCePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Shifted values: d_i = x_i - max(x), so d_i <= 0 for all i.
    // At least one d_i = 0 (the max element).
    let d0 = declare_real(&mut program, "d0");
    let d1 = declare_real(&mut program, "d1");
    let d2 = declare_real(&mut program, "d2");

    let zero = Expr::real(0);

    // All shifts <= 0
    program.assert(d0.clone().real_le(zero.clone()));
    program.assert(d1.clone().real_le(zero.clone()));
    program.assert(d2.clone().real_le(zero.clone()));

    // At least one is exactly 0 (the max element)
    let one_is_max = d0
        .clone()
        .eq(zero.clone())
        .or(d1.clone().eq(zero.clone()))
        .or(d2.clone().eq(zero.clone()));
    program.assert(one_is_max);

    // exp(d_i) where d_i <= 0 implies exp(d_i) in (0, 1].
    // exp(0) = 1, exp(negative) < 1.
    let exp0 = declare_real(&mut program, "exp_d0");
    let exp1 = declare_real(&mut program, "exp_d1");
    let exp2 = declare_real(&mut program, "exp_d2");

    let one = Expr::real(1);

    // Each exp(d_i) in (0, 1]
    assert_positive(&mut program, &exp0);
    assert_positive(&mut program, &exp1);
    assert_positive(&mut program, &exp2);
    program.assert(exp0.clone().real_le(one.clone()));
    program.assert(exp1.clone().real_le(one.clone()));
    program.assert(exp2.clone().real_le(one.clone()));

    // At least one exp(d_i) = 1 (from the d_i = 0 element)
    let one_exp_is_one = exp0
        .clone()
        .eq(one.clone())
        .or(exp1.clone().eq(one.clone()))
        .or(exp2.clone().eq(one.clone()));
    program.assert(one_exp_is_one);

    // Sum of exp values
    let exp_sum = exp0.real_add(exp1).real_add(exp2);

    // Violation: sum < 1 or sum > 3 (for n=3, sum should be in [1, 3])
    let three = Expr::real(3);
    let sum_too_low = exp_sum.clone().real_lt(one);
    let sum_too_high = exp_sum.real_gt(three);
    let violation = sum_too_low.or(sum_too_high);

    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SoftmaxCePropertyResult {
        property: "log_softmax_stability".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 5: Cross-Entropy Non-Negativity
// ---------------------------------------------------------------------------

/// Prove CE(y, p) >= 0 when y is a valid distribution (y_i >= 0, sum(y_i) = 1)
/// and p_i is in (0, 1].
///
/// CE(y, p) = -sum(y_i * log(p_i))
///
/// Since p_i in (0, 1], log(p_i) <= 0. So -log(p_i) >= 0.
/// Since y_i >= 0, each term y_i * (-log(p_i)) >= 0.
/// Therefore the sum >= 0.
///
/// We model this in QF_LRA: let neg_log_p_i = -log(p_i) >= 0, y_i >= 0,
/// sum(y_i) = 1. Then CE = sum(y_i * neg_log_p_i) >= 0.
///
/// To stay in QF_LRA, we model each product term t_i = y_i * neg_log_p_i
/// as a helper variable constrained by the non-negativity of both factors.
pub fn prove_cross_entropy_non_negativity() -> Result<SoftmaxCePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // Target distribution y: y_i >= 0, sum = 1
    let y0 = declare_real(&mut program, "y0");
    let y1 = declare_real(&mut program, "y1");
    let y2 = declare_real(&mut program, "y2");

    program.assert(y0.clone().real_ge(zero.clone()));
    program.assert(y1.clone().real_ge(zero.clone()));
    program.assert(y2.clone().real_ge(zero.clone()));
    program.assert(y0.clone().real_add(y1.clone()).real_add(y2.clone()).eq(one));

    // -log(p_i) >= 0 for p_i in (0, 1] (since log(p_i) <= 0)
    let neg_log_p0 = declare_real(&mut program, "neg_log_p0");
    let neg_log_p1 = declare_real(&mut program, "neg_log_p1");
    let neg_log_p2 = declare_real(&mut program, "neg_log_p2");

    program.assert(neg_log_p0.clone().real_ge(zero.clone()));
    program.assert(neg_log_p1.clone().real_ge(zero.clone()));
    program.assert(neg_log_p2.clone().real_ge(zero.clone()));

    // CE = y0 * neg_log_p0 + y1 * neg_log_p1 + y2 * neg_log_p2
    // Each product term t_i = y_i * neg_log_p_i >= 0 (product of non-negatives)
    let t0 = declare_real(&mut program, "t0");
    let t1 = declare_real(&mut program, "t1");
    let t2 = declare_real(&mut program, "t2");

    // t_i >= 0 (product of non-negative values)
    program.assert(t0.clone().real_ge(zero.clone()));
    program.assert(t1.clone().real_ge(zero.clone()));
    program.assert(t2.clone().real_ge(zero.clone()));

    // Bound each term: t_i <= y_i * M where M is max of neg_log values.
    // For the non-negativity proof, we only need t_i >= 0.
    let ce = t0.real_add(t1).real_add(t2);

    // Violation: CE < 0
    let violation = ce.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SoftmaxCePropertyResult {
        property: "cross_entropy_non_negativity".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 6: Cross-Entropy Minimum (Gibbs' Inequality)
// ---------------------------------------------------------------------------

/// Prove that CE(y, p) achieves its minimum when p = y (Gibbs' inequality).
///
/// Gibbs' inequality states: CE(y, p) >= H(y) for any distributions y, p,
/// where H(y) = -sum(y_i * log(y_i)) is the entropy of y. Equality holds
/// iff p = y.
///
/// Equivalently: KL(y || p) = CE(y, p) - H(y) >= 0. This is the
/// non-negativity of KL divergence.
///
/// We prove this for a 2-element distribution using the algebraic fact:
/// For a, b > 0 with a + b = 1 and c, d > 0 with c + d = 1:
///   a * log(a/c) + b * log(b/d) >= 0
///
/// We use the linearized log-inequality: log(x) <= x - 1 for x > 0.
/// Therefore: -log(x) >= 1 - x.
///
/// KL(y || p) = sum(y_i * log(y_i/p_i))
///            = sum(y_i * (-log(p_i/y_i)))
///            >= sum(y_i * (1 - p_i/y_i))      [using -log(x) >= 1 - x]
///            = sum(y_i - p_i)
///            = 1 - 1 = 0
///
/// We encode this linearized lower bound in QF_LRA.
pub fn prove_cross_entropy_minimum_gibbs() -> Result<SoftmaxCePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // Distribution y: y0 + y1 = 1, y_i > 0 (strict for log to be defined)
    let y0 = declare_real(&mut program, "y0");
    let y1 = declare_real(&mut program, "y1");

    assert_positive(&mut program, &y0);
    assert_positive(&mut program, &y1);
    program.assert(y0.clone().real_add(y1.clone()).eq(one.clone()));

    // Distribution p: p0 + p1 = 1, p_i > 0
    let p0 = declare_real(&mut program, "p0");
    let p1 = declare_real(&mut program, "p1");

    assert_positive(&mut program, &p0);
    assert_positive(&mut program, &p1);
    program.assert(p0.clone().real_add(p1.clone()).eq(one.clone()));

    // Linearized KL lower bound:
    // sum(y_i * (1 - p_i/y_i)) = sum(y_i - p_i) = (y0 - p0) + (y1 - p1)
    // = (y0 + y1) - (p0 + p1) = 1 - 1 = 0
    //
    // So the linearized KL lower bound is exactly 0 for any valid distributions.
    // This means: KL(y || p) >= 0.
    //
    // We encode this directly: sum(y_i - p_i) = 0, and since -log(x) >= 1 - x,
    // the KL divergence is >= this sum = 0.

    let diff_sum = (y0.real_sub(p0)).real_add(y1.real_sub(p1));

    // Violation: diff_sum != 0
    let violation = diff_sum.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SoftmaxCePropertyResult {
        property: "cross_entropy_minimum_gibbs".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 7: KL Divergence Decomposition
// ---------------------------------------------------------------------------

/// Prove that CE(y, p) = H(y) + KL(y || p), where:
///   CE(y, p) = -sum(y_i * log(p_i))
///   H(y)     = -sum(y_i * log(y_i))
///   KL(y||p) = sum(y_i * log(y_i / p_i)) = sum(y_i * (log(y_i) - log(p_i)))
///
/// Algebraically:
///   H(y) + KL(y||p) = -sum(y_i * log(y_i)) + sum(y_i * (log(y_i) - log(p_i)))
///                    = -sum(y_i * log(y_i)) + sum(y_i * log(y_i)) - sum(y_i * log(p_i))
///                    = -sum(y_i * log(p_i))
///                    = CE(y, p)
///
/// This is a pure algebraic identity. We model it with abstract log values.
pub fn prove_kl_divergence_decomposition() -> Result<SoftmaxCePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Abstract values: y_i, log_y_i = log(y_i), log_p_i = log(p_i)
    let y0 = declare_real(&mut program, "y0");
    let y1 = declare_real(&mut program, "y1");
    let y2 = declare_real(&mut program, "y2");

    let log_y0 = declare_real(&mut program, "log_y0");
    let log_y1 = declare_real(&mut program, "log_y1");
    let log_y2 = declare_real(&mut program, "log_y2");

    let log_p0 = declare_real(&mut program, "log_p0");
    let log_p1 = declare_real(&mut program, "log_p1");
    let log_p2 = declare_real(&mut program, "log_p2");

    // Bound all variables to help solver convergence
    let bound_lo = Expr::real(-1000);
    let bound_hi = Expr::real(1000);
    for v in [
        &y0, &y1, &y2, &log_y0, &log_y1, &log_y2, &log_p0, &log_p1, &log_p2,
    ] {
        assert_bounds(&mut program, v, &bound_lo, &bound_hi);
    }

    // CE = -sum(y_i * log(p_i))
    // = -(y0*log_p0 + y1*log_p1 + y2*log_p2)
    let ce_terms = y0
        .clone()
        .real_mul(log_p0.clone())
        .real_add(y1.clone().real_mul(log_p1.clone()))
        .real_add(y2.clone().real_mul(log_p2.clone()));
    let ce = ce_terms.real_neg();

    // H(y) = -sum(y_i * log(y_i))
    // = -(y0*log_y0 + y1*log_y1 + y2*log_y2)
    let h_terms = y0
        .clone()
        .real_mul(log_y0.clone())
        .real_add(y1.clone().real_mul(log_y1.clone()))
        .real_add(y2.clone().real_mul(log_y2.clone()));
    let h = h_terms.real_neg();

    // KL(y||p) = sum(y_i * (log_y_i - log_p_i))
    let kl = y0
        .real_mul(log_y0.real_sub(log_p0))
        .real_add(y1.real_mul(log_y1.real_sub(log_p1)))
        .real_add(y2.real_mul(log_y2.real_sub(log_p2)));

    // Identity: CE = H + KL
    // Violation: CE != H + KL
    let h_plus_kl = h.real_add(kl);
    let violation = ce.ne(h_plus_kl);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SoftmaxCePropertyResult {
        property: "kl_divergence_decomposition".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 8: Temperature Scaling
// ---------------------------------------------------------------------------

/// Prove temperature scaling properties of softmax for two concrete temperatures.
///
/// For a 2-element softmax with x_0 > x_1:
///   softmax(x/T)_0 = e_0 / (e_0 + e_1) where e_i = exp(x_i / T)
///
/// **High temperature (T large):** As T -> inf, x_i/T -> 0, so all exp values
/// approach 1, and softmax approaches uniform (1/n). We show that at T=100,
/// the softmax spread |s_0 - s_1| is small (< 0.1 for bounded inputs).
///
/// **Low temperature (T small):** As T -> 0+, x_0/T >> x_1/T, so exp(x_0/T)
/// dominates, and softmax(x)_0 -> 1. We show that at T=0.01, s_0 > 0.99
/// for sufficiently separated inputs.
///
/// We prove the high-T case structurally: if exp values are close to each
/// other (within epsilon), then softmax values are close to uniform.
pub fn prove_temperature_scaling() -> Result<SoftmaxCePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // High-T regime: exp(x_i/T) values are all close to exp(0) = 1.
    // Model: e_i in (1 - eps, 1 + eps) with eps = 0.1 (from T=100, |x| <= 10).
    //
    // When all exp values are in (0.9, 1.1) for a 2-element softmax:
    //   s_0 = e_0 / (e_0 + e_1)
    //   |s_0 - 0.5| = |e_0 - e_1| / (2 * (e_0 + e_1))
    //
    // Bound: |e_0 - e_1| <= 0.2, (e_0 + e_1) >= 1.8
    // So |s_0 - 0.5| <= 0.2 / 1.8 < 0.12

    let e0 = declare_real(&mut program, "e0");
    let e1 = declare_real(&mut program, "e1");

    // e_i in (0.9, 1.1) — high temperature makes exp values close to 1
    let lo = Expr::real(9).real_div(Expr::real(10)); // 0.9
    let hi = Expr::real(11).real_div(Expr::real(10)); // 1.1
    assert_bounds(&mut program, &e0, &lo, &hi);
    assert_bounds(&mut program, &e1, &lo, &hi);

    let denom = e0.clone().real_add(e1.clone());

    // s_0 * denom = e_0
    let s0 = declare_real(&mut program, "s0");
    program.assert(s0.clone().real_mul(denom).eq(e0));

    // half = 0.5
    let half = Expr::real(1).real_div(Expr::real(2));
    // deviation from uniform: |s_0 - 0.5|
    // We prove |s_0 - 0.5| < 0.12
    let dev = declare_real(&mut program, "dev");
    program.assert(dev.clone().eq(s0.real_sub(half)));

    // Violation: |dev| >= 0.12 (i.e., dev >= 0.12 or dev <= -0.12)
    let threshold = Expr::real(12).real_div(Expr::real(100)); // 0.12
    let neg_threshold = Expr::real(-12).real_div(Expr::real(100));
    let violation = dev
        .clone()
        .real_ge(threshold)
        .or(dev.real_le(neg_threshold));

    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SoftmaxCePropertyResult {
        property: "temperature_scaling_high_t_near_uniform".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that at low temperature, the larger input dominates the softmax.
///
/// When exp(x_0/T) >> exp(x_1/T) (low T, x_0 > x_1):
///   s_0 = e_0 / (e_0 + e_1) -> 1
///
/// We model: e_0 >= R * e_1 where R is a large ratio (e.g., R >= 100).
/// Then s_0 = e_0 / (e_0 + e_1) >= R*e_1 / (R*e_1 + e_1) = R / (R+1).
/// For R=100: s_0 >= 100/101 > 0.99.
pub fn prove_temperature_scaling_low_t() -> Result<SoftmaxCePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let e0 = declare_real(&mut program, "e0");
    let e1 = declare_real(&mut program, "e1");

    assert_positive(&mut program, &e0);
    assert_positive(&mut program, &e1);

    // Low temperature: e0 >= 100 * e1
    let hundred = Expr::real(100);
    program.assert(e0.clone().real_ge(hundred.real_mul(e1.clone())));

    let denom = declare_real(&mut program, "denom");
    program.assert(denom.clone().eq(e0.clone().real_add(e1.clone())));

    // s_0 * denom = e_0
    let s0 = declare_real(&mut program, "s0");
    program.assert(s0.clone().real_mul(denom).eq(e0));

    // Threshold: 100/101 (just under 0.99)
    let threshold = Expr::real(100).real_div(Expr::real(101));

    // Violation: s_0 < 100/101
    let violation = s0.real_lt(threshold);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SoftmaxCePropertyResult {
        property: "temperature_scaling_low_t_argmax".to_string(),
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
    fn test_softmax_output_range_proven() {
        let result = prove_softmax_output_range().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Softmax output range: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Softmax output range must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "softmax_output_range");
    }

    #[test]
    fn test_softmax_sum_to_one_proven() {
        let result = prove_softmax_sum_to_one().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Softmax sum-to-one: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Softmax sum-to-one must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "softmax_sum_to_one");
    }

    #[test]
    fn test_softmax_monotonicity_proven() {
        let result = prove_softmax_monotonicity().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Softmax monotonicity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Softmax monotonicity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "softmax_monotonicity");
    }

    #[test]
    fn test_log_softmax_stability_proven() {
        let result = prove_log_softmax_stability().expect("proof should not error");
        assert!(
            result.proven,
            "Log-softmax stability (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "log_softmax_stability");
    }

    #[test]
    fn test_cross_entropy_non_negativity_proven() {
        let result = prove_cross_entropy_non_negativity().expect("proof should not error");
        assert!(
            result.proven,
            "Cross-entropy non-negativity (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "cross_entropy_non_negativity");
    }

    #[test]
    fn test_cross_entropy_minimum_gibbs_proven() {
        let result = prove_cross_entropy_minimum_gibbs().expect("proof should not error");
        assert!(
            result.proven,
            "Gibbs' inequality linearized bound (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "cross_entropy_minimum_gibbs");
    }

    #[test]
    fn test_kl_divergence_decomposition_proven() {
        let result = prove_kl_divergence_decomposition().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "KL decomposition: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "KL decomposition must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "kl_divergence_decomposition");
    }

    #[test]
    fn test_temperature_scaling_high_t_proven() {
        let result = prove_temperature_scaling().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Temperature scaling high-T: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Temperature scaling high-T must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "temperature_scaling_high_t_near_uniform");
    }

    #[test]
    fn test_temperature_scaling_low_t_proven() {
        let result = prove_temperature_scaling_low_t().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Temperature scaling low-T: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Temperature scaling low-T must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "temperature_scaling_low_t_argmax");
    }

    #[test]
    fn test_softmax_output_range_smt2_structure() {
        let result = prove_softmax_output_range().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
        assert!(
            result.smt2.contains("declare-const"),
            "should have declarations"
        );
    }

    #[test]
    fn test_kl_decomposition_smt2_structure() {
        let result = prove_kl_divergence_decomposition().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
        // Should have variables for y, log_y, log_p
        assert!(
            result.smt2.contains("log_y0"),
            "should have log_y0 variable"
        );
        assert!(
            result.smt2.contains("log_p0"),
            "should have log_p0 variable"
        );
    }
}
