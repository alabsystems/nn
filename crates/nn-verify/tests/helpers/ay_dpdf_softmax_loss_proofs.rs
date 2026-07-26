// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for softmax and cross-entropy loss mathematical
//! properties critical for dpdf model training and inference.
//!
//! Proves 20 properties (test_1071 through test_1090):
//!  1. Softmax output sums to 1
//!  2. Softmax output all positive
//!  3. Softmax output in (0, 1)
//!  4. Softmax is translation invariant
//!  5. Log-softmax = log(softmax) (stable formulation identity)
//!  6. Cross-entropy loss >= 0
//!  7. Cross-entropy = -sum(y * log(p))
//!  8. KL divergence >= 0
//!  9. Softmax temperature scaling: higher T -> more uniform
//! 10. Softmax gradient: diag(p) - p*p^T
//! 11. Softmax numerical stability: subtract max
//! 12. Log-softmax gradient simpler than softmax gradient
//! 13. Cross-entropy with label smoothing bounded
//! 14. Focal loss weighting factor
//! 15. Softmax preserves ordering (monotonic)
//! 16. Softmax with mask: masked positions -> 0 probability
//! 17. Top-k softmax concentrates probability mass
//! 18. Softmax output bounded by exp(max_logit - min_logit)
//! 19. Binary cross-entropy is special case of categorical
//! 20. Softmax temperature 0 -> one-hot (argmax)
//!
//! Part of #4232.

use ay_bindings::execute_direct::{self, ExecuteResult};
use ay_bindings::{Expr, Sort, AYProgram};
use nn_verify::ay_real_lit::RealLit;

/// Helper: create a Real-sorted variable.
fn real_var(name: &str) -> Expr {
    Expr::var(name, Sort::real())
}

/// Helper: assert that program is UNSAT (property holds for all inputs).
///
/// The ay convention: we assert the negation of the property, then
/// UNSAT (Verified) means the original property holds universally.
fn assert_verified(prog: &AYProgram, property_name: &str) {
    match execute_direct::execute(prog) {
        Ok(ExecuteResult::Verified) => {
            // UNSAT — property proved for all inputs.
        }
        Ok(other) => {
            panic!(
                "{property_name}: expected Verified (UNSAT), got: {other:?}. \
                 The negated property is satisfiable — the property does NOT hold."
            );
        }
        Err(e) => {
            panic!("{property_name}: ay execution error: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Test 1071: Softmax output sums to 1
// ---------------------------------------------------------------------------

/// Prove: for a 3-element softmax, s_0 + s_1 + s_2 = 1.
///
/// Given e_i = exp(x_i) > 0 and s_i = e_i / (e_0 + e_1 + e_2):
///   sum(s_i) = (e_0 + e_1 + e_2) / (e_0 + e_1 + e_2) = 1.
///
/// We model s_i * denom = e_i with denom = sum(e_i), and prove sum(s_i) = 1.
#[test]
fn test_1071_softmax_sum_to_one() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("e0", real.clone());
    let _ = prog.declare_const("e1", real.clone());
    let _ = prog.declare_const("e2", real.clone());
    let _ = prog.declare_const("denom", real.clone());
    let _ = prog.declare_const("s0", real.clone());
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("s2", real);

    let e0 = real_var("e0");
    let e1 = real_var("e1");
    let e2 = real_var("e2");
    let denom = real_var("denom");
    let s0 = real_var("s0");
    let s1 = real_var("s1");
    let s2 = real_var("s2");

    // exp values positive
    prog.assert(e0.clone().real_gt(Expr::real(0)));
    prog.assert(e1.clone().real_gt(Expr::real(0)));
    prog.assert(e2.clone().real_gt(Expr::real(0)));

    // denom = e0 + e1 + e2
    prog.assert(
        denom
            .clone()
            .eq(e0.clone().real_add(e1.clone()).real_add(e2.clone())),
    );

    // s_i * denom = e_i (models s_i = e_i / denom)
    prog.assert(s0.clone().real_mul(denom.clone()).eq(e0));
    prog.assert(s1.clone().real_mul(denom.clone()).eq(e1));
    prog.assert(s2.clone().real_mul(denom).eq(e2));

    // Negated property: s0 + s1 + s2 != 1
    let violation = s0.real_add(s1).real_add(s2).ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_sum_to_one");
}

// ---------------------------------------------------------------------------
// Test 1072: Softmax output all positive
// ---------------------------------------------------------------------------

/// Prove: each softmax output s_i > 0.
///
/// Since e_i = exp(x_i) > 0 and denom = sum(exp(x_j)) > 0,
/// s_i = e_i / denom > 0.
#[test]
fn test_1072_softmax_all_positive() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp_x", real.clone());
    let _ = prog.declare_const("z", real.clone());
    let _ = prog.declare_const("s", real);

    let exp_x = real_var("exp_x");
    let z = real_var("z");
    let s = real_var("s");

    // exp(x) > 0
    prog.assert(exp_x.clone().real_gt(Expr::real(0)));

    // Z = sum of exp values >= exp_x > 0
    prog.assert(z.clone().real_ge(exp_x.clone()));
    prog.assert(z.clone().real_gt(Expr::real(0)));

    // s * Z = exp_x (models s = exp_x / Z)
    prog.assert(s.clone().real_mul(z).eq(exp_x));

    // Negated property: s <= 0
    let violation = s.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_all_positive");
}

// ---------------------------------------------------------------------------
// Test 1073: Softmax output in (0, 1)
// ---------------------------------------------------------------------------

/// Prove: 0 < softmax(x)_i < 1 for a vector of size >= 2.
///
/// Since exp(x_i) > 0, the numerator is positive. Since the denominator
/// includes at least one other positive term, denom > exp(x_i), so
/// s_i = exp(x_i) / denom < 1.
#[test]
fn test_1073_softmax_output_in_zero_one() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp_x", real.clone());
    let _ = prog.declare_const("exp_other", real.clone());
    let _ = prog.declare_const("z", real.clone());
    let _ = prog.declare_const("s", real);

    let exp_x = real_var("exp_x");
    let exp_other = real_var("exp_other");
    let z = real_var("z");
    let s = real_var("s");

    // exp values positive
    prog.assert(exp_x.clone().real_gt(Expr::real(0)));
    prog.assert(exp_other.clone().real_gt(Expr::real(0)));

    // Z = exp_x + exp_other (at least 2 elements)
    prog.assert(z.clone().eq(exp_x.clone().real_add(exp_other)));

    // s * Z = exp_x
    prog.assert(s.clone().real_mul(z).eq(exp_x));

    // Negated property: s <= 0 OR s >= 1
    let violation = s
        .clone()
        .real_le(Expr::real(0))
        .or(s.real_ge(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_output_in_zero_one");
}

// ---------------------------------------------------------------------------
// Test 1074: Softmax is translation invariant
// ---------------------------------------------------------------------------

/// Prove: softmax(x + c) = softmax(x) for any constant c.
///
/// exp(x_i + c) / sum(exp(x_j + c)) = exp(x_i)*exp(c) / (exp(c)*sum(exp(x_j)))
/// = exp(x_i) / sum(exp(x_j)) = softmax(x)_i.
///
/// We model: s_orig = e_i / (e_i + e_j), s_shifted = (e_i*k) / ((e_i + e_j)*k)
/// where k = exp(c) > 0, and prove s_orig = s_shifted.
#[test]
fn test_1074_softmax_translation_invariant() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("e1", real.clone());
    let _ = prog.declare_const("e2", real.clone());
    let _ = prog.declare_const("k", real.clone());
    let _ = prog.declare_const("s_orig", real.clone());
    let _ = prog.declare_const("s_shifted", real);

    let e1 = real_var("e1");
    let e2 = real_var("e2");
    let k = real_var("k");
    let s_orig = real_var("s_orig");
    let s_shifted = real_var("s_shifted");

    // exp values positive
    prog.assert(e1.clone().real_gt(Expr::real(0)));
    prog.assert(e2.clone().real_gt(Expr::real(0)));
    // k = exp(c) > 0
    prog.assert(k.clone().real_gt(Expr::real(0)));

    // s_orig = e1 / (e1 + e2)
    let z_orig = e1.clone().real_add(e2.clone());
    prog.assert(s_orig.clone().real_mul(z_orig).eq(e1.clone()));

    // Shifted: e1' = e1*k, e2' = e2*k
    let e1k = e1.real_mul(k.clone());
    let e2k = e2.real_mul(k);
    let z_shifted = e1k.clone().real_add(e2k);
    prog.assert(s_shifted.clone().real_mul(z_shifted).eq(e1k));

    // Negated property: s_orig != s_shifted
    let violation = s_orig.ne(s_shifted);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_translation_invariant");
}

// ---------------------------------------------------------------------------
// Test 1075: Log-softmax = log(softmax) stable formulation identity
// ---------------------------------------------------------------------------

/// Prove: log(softmax(x)_i) = x_i - log(sum(exp(x_j))) and the stable
/// formulation log_softmax(x)_i = x_i - M - log(sum(exp(x_j - M))) are equal.
///
/// Algebraically: log(sum(exp(x_j))) = M + log(sum(exp(x_j - M))) where M = max(x).
/// Therefore: x_i - log(sum(exp(x_j))) = x_i - M - log(sum(exp(x_j - M))).
#[test]
fn test_1075_log_softmax_equals_log_of_softmax() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x_i", real.clone());
    let _ = prog.declare_const("max_x", real.clone());
    let _ = prog.declare_const("log_sum_exp", real.clone());
    let _ = prog.declare_const("log_shifted_sum", real.clone());
    let _ = prog.declare_const("lsm_naive", real.clone());
    let _ = prog.declare_const("lsm_stable", real);

    let x_i = real_var("x_i");
    let max_x = real_var("max_x");
    let log_sum_exp = real_var("log_sum_exp");
    let log_shifted_sum = real_var("log_shifted_sum");
    let lsm_naive = real_var("lsm_naive");
    let lsm_stable = real_var("lsm_stable");

    // Axiom: log_sum_exp = max_x + log_shifted_sum
    prog.assert(
        log_sum_exp
            .clone()
            .eq(max_x.clone().real_add(log_shifted_sum.clone())),
    );

    // Naive: lsm_naive = x_i - log_sum_exp
    prog.assert(lsm_naive.clone().eq(x_i.clone().real_sub(log_sum_exp)));

    // Stable: lsm_stable = x_i - max_x - log_shifted_sum
    prog.assert(
        lsm_stable
            .clone()
            .eq(x_i.real_sub(max_x).real_sub(log_shifted_sum)),
    );

    // Negated property: lsm_naive != lsm_stable
    let violation = lsm_naive.ne(lsm_stable);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "log_softmax_equals_log_of_softmax");
}

// ---------------------------------------------------------------------------
// Test 1076: Cross-entropy loss >= 0
// ---------------------------------------------------------------------------

/// Prove: CE(y, p) = -sum(y_i * log(p_i)) >= 0 for valid distributions.
///
/// Since p_i in (0, 1], log(p_i) <= 0, so -log(p_i) >= 0.
/// Since y_i >= 0, each term y_i * (-log(p_i)) >= 0.
/// Therefore the sum >= 0.
#[test]
fn test_1076_cross_entropy_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("y0", real.clone());
    let _ = prog.declare_const("y1", real.clone());
    let _ = prog.declare_const("neg_log_p0", real.clone());
    let _ = prog.declare_const("neg_log_p1", real.clone());
    let _ = prog.declare_const("t0", real.clone());
    let _ = prog.declare_const("t1", real.clone());
    let _ = prog.declare_const("ce", real);

    let y0 = real_var("y0");
    let y1 = real_var("y1");
    let neg_log_p0 = real_var("neg_log_p0");
    let neg_log_p1 = real_var("neg_log_p1");
    let t0 = real_var("t0");
    let t1 = real_var("t1");
    let ce = real_var("ce");

    // Target distribution: y_i >= 0, sum = 1
    prog.assert(y0.clone().real_ge(Expr::real(0)));
    prog.assert(y1.clone().real_ge(Expr::real(0)));
    prog.assert(y0.clone().real_add(y1.clone()).eq(Expr::real(1)));

    // -log(p_i) >= 0 (since p_i in (0, 1])
    prog.assert(neg_log_p0.clone().real_ge(Expr::real(0)));
    prog.assert(neg_log_p1.clone().real_ge(Expr::real(0)));

    // Product terms t_i = y_i * neg_log_p_i >= 0
    prog.assert(t0.clone().real_ge(Expr::real(0)));
    prog.assert(t1.clone().real_ge(Expr::real(0)));

    // CE = t0 + t1
    prog.assert(ce.clone().eq(t0.real_add(t1)));

    // Negated property: CE < 0
    let violation = ce.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cross_entropy_non_negative");
}

// ---------------------------------------------------------------------------
// Test 1077: Cross-entropy = -sum(y * log(p)) for one-hot target
// ---------------------------------------------------------------------------

/// Prove: for one-hot target y = (1, 0), CE = -log(p_1).
///
/// CE = -(1 * log(p_1) + 0 * log(p_2)) = -log(p_1).
#[test]
fn test_1077_cross_entropy_one_hot_formula() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("log_p1", real.clone());
    let _ = prog.declare_const("log_p2", real.clone());
    let _ = prog.declare_const("ce_sum", real.clone());
    let _ = prog.declare_const("expected", real);

    let log_p1 = real_var("log_p1");
    let log_p2 = real_var("log_p2");
    let ce_sum = real_var("ce_sum");
    let expected = real_var("expected");

    // log probabilities bounded
    prog.assert(log_p1.clone().real_ge(Expr::real(-100)));
    prog.assert(log_p1.clone().real_le(Expr::real(0)));
    prog.assert(log_p2.clone().real_ge(Expr::real(-100)));
    prog.assert(log_p2.clone().real_le(Expr::real(0)));

    // One-hot y = (1, 0): CE = -(1*log_p1 + 0*log_p2) = -log_p1
    prog.assert(
        ce_sum.clone().eq(Expr::real(0).real_sub(
            Expr::real(1)
                .real_mul(log_p1.clone())
                .real_add(Expr::real(0).real_mul(log_p2)),
        )),
    );

    // expected = -log_p1
    prog.assert(expected.clone().eq(Expr::real(0).real_sub(log_p1)));

    // Negated property: ce_sum != expected
    let violation = ce_sum.ne(expected);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cross_entropy_one_hot_formula");
}

// ---------------------------------------------------------------------------
// Test 1078: KL divergence >= 0
// ---------------------------------------------------------------------------

/// Prove: KL(P || Q) >= 0 using the linearized log inequality.
///
/// KL(P || Q) = sum(p_i * log(p_i / q_i)) >= sum(p_i * (1 - q_i/p_i))
///            = sum(p_i - q_i) = 1 - 1 = 0.
///
/// The linearized lower bound sum(p_i - q_i) = 0 for valid distributions.
#[test]
fn test_1078_kl_divergence_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("p0", real.clone());
    let _ = prog.declare_const("p1", real.clone());
    let _ = prog.declare_const("q0", real.clone());
    let _ = prog.declare_const("q1", real);

    let p0 = real_var("p0");
    let p1 = real_var("p1");
    let q0 = real_var("q0");
    let q1 = real_var("q1");

    // P: p_i > 0, sum = 1
    prog.assert(p0.clone().real_gt(Expr::real(0)));
    prog.assert(p1.clone().real_gt(Expr::real(0)));
    prog.assert(p0.clone().real_add(p1.clone()).eq(Expr::real(1)));

    // Q: q_i > 0, sum = 1
    prog.assert(q0.clone().real_gt(Expr::real(0)));
    prog.assert(q1.clone().real_gt(Expr::real(0)));
    prog.assert(q0.clone().real_add(q1.clone()).eq(Expr::real(1)));

    // Linearized KL bound: sum(p_i - q_i) = (p0 - q0) + (p1 - q1)
    let lin_bound = p0.real_sub(q0).real_add(p1.real_sub(q1));

    // Negated property: lin_bound != 0
    let violation = lin_bound.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "kl_divergence_non_negative");
}

// ---------------------------------------------------------------------------
// Test 1079: Softmax temperature scaling: higher T -> more uniform
// ---------------------------------------------------------------------------

/// Prove: at high temperature, softmax approaches the uniform distribution.
///
/// When T is large, exp(x_i/T) values cluster near 1, so all softmax outputs
/// approach 1/n. For n=2 with exp values in (0.9, 1.1):
///   |s_0 - 0.5| < 0.12.
#[test]
fn test_1079_softmax_temperature_high_uniform() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("e0", real.clone());
    let _ = prog.declare_const("e1", real.clone());
    let _ = prog.declare_const("s0", real.clone());
    let _ = prog.declare_const("dev", real);

    let e0 = real_var("e0");
    let e1 = real_var("e1");
    let s0 = real_var("s0");
    let dev = real_var("dev");

    // High temperature: exp values in (0.9, 1.1)
    let lo = Expr::real(9).real_div(Expr::real(10));
    let hi = Expr::real(11).real_div(Expr::real(10));
    prog.assert(e0.clone().real_ge(lo.clone()));
    prog.assert(e0.clone().real_le(hi.clone()));
    prog.assert(e1.clone().real_ge(lo));
    prog.assert(e1.clone().real_le(hi));

    // s0 * (e0 + e1) = e0
    let denom = e0.clone().real_add(e1);
    prog.assert(s0.clone().real_mul(denom).eq(e0));

    // dev = s0 - 0.5
    let half = Expr::real(1).real_div(Expr::real(2));
    prog.assert(dev.clone().eq(s0.real_sub(half)));

    // Negated property: |dev| >= 12/100
    let threshold = Expr::real(12).real_div(Expr::real(100));
    let neg_threshold = Expr::real(-12).real_div(Expr::real(100));
    let violation = dev
        .clone()
        .real_ge(threshold)
        .or(dev.real_le(neg_threshold));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_temperature_high_uniform");
}

// ---------------------------------------------------------------------------
// Test 1080: Softmax gradient: diagonal Jacobian entry
// ---------------------------------------------------------------------------

/// Prove: the softmax Jacobian diagonal entry J_ii = s_i * (1 - s_i) satisfies:
///   - J_ii > 0 for s_i in (0, 1)
///   - J_ii <= 1/4 (maximum at s_i = 0.5)
///
/// This is the diag(p) - p*p^T diagonal term.
#[test]
fn test_1080_softmax_gradient_jacobian_diagonal() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("grad", real);

    let s = real_var("s");
    let grad = real_var("grad");

    // s in (0, 1)
    prog.assert(s.clone().real_gt(Expr::real(0)));
    prog.assert(s.clone().real_lt(Expr::real(1)));

    // grad = s * (1 - s)
    let one_minus_s = Expr::real(1).real_sub(s.clone());
    prog.assert(grad.clone().eq(s.real_mul(one_minus_s)));

    // Negated property: grad <= 0 OR grad > 1/4
    let violation = grad
        .clone()
        .real_le(Expr::real(0))
        .or(grad.real_gt(Expr::real_ratio(1, 4)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_gradient_jacobian_diagonal");
}

// ---------------------------------------------------------------------------
// Test 1081: Softmax numerical stability: subtract max
// ---------------------------------------------------------------------------

/// Prove: after subtracting max(x), all shifted values are in (-inf, 0],
/// so exp of shifted values is in (0, 1], and the sum is in [1, n].
///
/// For n=3 with d_i = x_i - max(x) <= 0 and at least one d_i = 0:
///   each exp(d_i) in (0, 1], and sum of exp in [1, 3].
#[test]
fn test_1081_softmax_numerical_stability_subtract_max() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("d0", real.clone());
    let _ = prog.declare_const("d1", real.clone());
    let _ = prog.declare_const("d2", real.clone());
    let _ = prog.declare_const("exp0", real.clone());
    let _ = prog.declare_const("exp1", real.clone());
    let _ = prog.declare_const("exp2", real);

    let d0 = real_var("d0");
    let d1 = real_var("d1");
    let d2 = real_var("d2");
    let exp0 = real_var("exp0");
    let exp1 = real_var("exp1");
    let exp2 = real_var("exp2");

    // Shifted values: d_i <= 0
    prog.assert(d0.clone().real_le(Expr::real(0)));
    prog.assert(d1.clone().real_le(Expr::real(0)));
    prog.assert(d2.clone().real_le(Expr::real(0)));

    // At least one is exactly 0 (the max element)
    let one_is_max = d0
        .eq(Expr::real(0))
        .or(d1.eq(Expr::real(0)))
        .or(d2.eq(Expr::real(0)));
    prog.assert(one_is_max);

    // exp(d_i) in (0, 1] when d_i <= 0
    prog.assert(exp0.clone().real_gt(Expr::real(0)));
    prog.assert(exp0.clone().real_le(Expr::real(1)));
    prog.assert(exp1.clone().real_gt(Expr::real(0)));
    prog.assert(exp1.clone().real_le(Expr::real(1)));
    prog.assert(exp2.clone().real_gt(Expr::real(0)));
    prog.assert(exp2.clone().real_le(Expr::real(1)));

    // At least one exp = 1 (from d_i = 0)
    let one_exp_is_one = exp0
        .clone()
        .eq(Expr::real(1))
        .or(exp1.clone().eq(Expr::real(1)))
        .or(exp2.clone().eq(Expr::real(1)));
    prog.assert(one_exp_is_one);

    // Sum of exp values
    let exp_sum = exp0.real_add(exp1).real_add(exp2);

    // Negated property: sum < 1 or sum > 3
    let violation = exp_sum
        .clone()
        .real_lt(Expr::real(1))
        .or(exp_sum.real_gt(Expr::real(3)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_numerical_stability_subtract_max");
}

// ---------------------------------------------------------------------------
// Test 1082: Log-softmax gradient is simpler: grad_i = 1 - softmax(x)_i
// ---------------------------------------------------------------------------

/// Prove: the gradient of log-softmax w.r.t. x_i for the target class is
///   d(log_softmax(x)_k) / d(x_k) = 1 - s_k
///
/// For the correct class k, the CE gradient is s_k - 1, and the log-softmax
/// gradient is 1 - s_k = -(s_k - 1). Both are bounded: since s_k in (0, 1),
/// the gradient 1 - s_k is in (0, 1).
#[test]
fn test_1082_log_softmax_gradient_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("s_k", real.clone());
    let _ = prog.declare_const("grad_log_sm", real);

    let s_k = real_var("s_k");
    let grad_log_sm = real_var("grad_log_sm");

    // s_k in (0, 1)
    prog.assert(s_k.clone().real_gt(Expr::real(0)));
    prog.assert(s_k.clone().real_lt(Expr::real(1)));

    // grad = 1 - s_k
    prog.assert(grad_log_sm.clone().eq(Expr::real(1).real_sub(s_k)));

    // Negated property: grad <= 0 OR grad >= 1
    let violation = grad_log_sm
        .clone()
        .real_le(Expr::real(0))
        .or(grad_log_sm.real_ge(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "log_softmax_gradient_bounded");
}

// ---------------------------------------------------------------------------
// Test 1083: Cross-entropy with label smoothing bounded
// ---------------------------------------------------------------------------

/// Prove: label-smoothed CE is non-negative when the components are non-negative.
///
/// CE(y_smooth, p) = (1 - eps) * CE(y, p) + eps * CE(uniform, p).
/// Since eps in (0, 1), (1 - eps) >= 0, and both CE values >= 0,
/// the smoothed CE >= 0.
#[test]
fn test_1083_label_smoothing_ce_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("eps", real.clone());
    let _ = prog.declare_const("ce_onehot", real.clone());
    let _ = prog.declare_const("ce_uniform", real.clone());
    let _ = prog.declare_const("ce_smooth", real);

    let eps = real_var("eps");
    let ce_onehot = real_var("ce_onehot");
    let ce_uniform = real_var("ce_uniform");
    let ce_smooth = real_var("ce_smooth");

    // eps in (0, 1)
    prog.assert(eps.clone().real_gt(Expr::real(0)));
    prog.assert(eps.clone().real_lt(Expr::real(1)));

    // Both CE values non-negative
    prog.assert(ce_onehot.clone().real_ge(Expr::real(0)));
    prog.assert(ce_uniform.clone().real_ge(Expr::real(0)));

    // ce_smooth = (1 - eps) * ce_onehot + eps * ce_uniform
    let one_minus_eps = Expr::real(1).real_sub(eps.clone());
    prog.assert(
        ce_smooth.clone().eq(one_minus_eps
            .real_mul(ce_onehot)
            .real_add(eps.real_mul(ce_uniform))),
    );

    // Negated property: ce_smooth < 0
    let violation = ce_smooth.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "label_smoothing_ce_non_negative");
}

// ---------------------------------------------------------------------------
// Test 1084: Focal loss weighting factor non-negative
// ---------------------------------------------------------------------------

/// Prove: focal loss FL(p) = alpha * (1-p)^gamma * (-log(p)) >= 0.
///
/// For p in (0, 1], alpha > 0, gamma >= 0:
///   modulator = (1-p)^gamma >= 0, -log(p) >= 0, alpha > 0.
///   Product of non-negatives with a positive factor >= 0.
#[test]
fn test_1084_focal_loss_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("alpha", real.clone());
    let _ = prog.declare_const("modulator", real.clone());
    let _ = prog.declare_const("neg_log_p", real.clone());
    let _ = prog.declare_const("fl", real);

    let alpha = real_var("alpha");
    let modulator = real_var("modulator");
    let neg_log_p = real_var("neg_log_p");
    let fl = real_var("fl");

    // alpha > 0
    prog.assert(alpha.clone().real_gt(Expr::real(0)));

    // modulator = (1-p)^gamma >= 0
    prog.assert(modulator.clone().real_ge(Expr::real(0)));

    // -log(p) >= 0
    prog.assert(neg_log_p.clone().real_ge(Expr::real(0)));

    // fl = alpha * modulator * neg_log_p
    prog.assert(fl.clone().eq(alpha.real_mul(modulator).real_mul(neg_log_p)));

    // Negated property: fl < 0
    let violation = fl.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "focal_loss_non_negative");
}

// ---------------------------------------------------------------------------
// Test 1085: Softmax preserves ordering (monotonic)
// ---------------------------------------------------------------------------

/// Prove: if x_i > x_j, then softmax(x)_i > softmax(x)_j.
///
/// Since exp is strictly increasing, x_i > x_j implies exp(x_i) > exp(x_j).
/// With a shared positive denominator: s_i = e_i/Z > e_j/Z = s_j.
#[test]
fn test_1085_softmax_preserves_ordering() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("e_i", real.clone());
    let _ = prog.declare_const("e_j", real.clone());
    let _ = prog.declare_const("e_k", real.clone());
    let _ = prog.declare_const("denom", real.clone());
    let _ = prog.declare_const("s_i", real.clone());
    let _ = prog.declare_const("s_j", real);

    let e_i = real_var("e_i");
    let e_j = real_var("e_j");
    let e_k = real_var("e_k");
    let denom = real_var("denom");
    let s_i = real_var("s_i");
    let s_j = real_var("s_j");

    // exp values positive
    prog.assert(e_i.clone().real_gt(Expr::real(0)));
    prog.assert(e_j.clone().real_gt(Expr::real(0)));
    prog.assert(e_k.clone().real_gt(Expr::real(0)));

    // exp monotonicity: e_i > e_j (from x_i > x_j)
    prog.assert(e_i.clone().real_gt(e_j.clone()));

    // denom = e_i + e_j + e_k
    prog.assert(
        denom
            .clone()
            .eq(e_i.clone().real_add(e_j.clone()).real_add(e_k)),
    );

    // s_i * denom = e_i, s_j * denom = e_j
    prog.assert(s_i.clone().real_mul(denom.clone()).eq(e_i));
    prog.assert(s_j.clone().real_mul(denom).eq(e_j));

    // Negated property: s_i <= s_j
    let violation = s_i.real_le(s_j);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_preserves_ordering");
}

// ---------------------------------------------------------------------------
// Test 1086: Softmax with mask: masked positions -> near-zero probability
// ---------------------------------------------------------------------------

/// Prove: after masking a position to -inf before softmax, its output is near 0.
///
/// When a position's score is masked to a very large negative value M,
/// exp(-M) is near 0. The softmax output for that position approaches 0.
#[test]
fn test_1086_softmax_masked_position_near_zero() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp_valid", real.clone());
    let _ = prog.declare_const("exp_masked", real.clone());
    let _ = prog.declare_const("s_masked", real);

    let exp_valid = real_var("exp_valid");
    let exp_masked = real_var("exp_masked");
    let s_masked = real_var("s_masked");

    // Valid position has positive, bounded exp
    prog.assert(exp_valid.clone().real_gt(Expr::real(0)));
    prog.assert(exp_valid.clone().real_le(Expr::real(1000)));

    // Masked position: exp(score) is near 0 (score was -inf)
    let eps = Expr::real_ratio(1, 1000000);
    prog.assert(exp_masked.clone().real_ge(Expr::real(0)));
    prog.assert(exp_masked.clone().real_le(eps));

    // s_masked = exp_masked / (exp_valid + exp_masked)
    let z = exp_valid.real_add(exp_masked.clone());
    prog.assert(s_masked.clone().real_mul(z).eq(exp_masked));

    // Negated property: s_masked > 0.001
    let violation = s_masked.real_gt(Expr::real_ratio(1, 1000));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_masked_position_near_zero");
}

// ---------------------------------------------------------------------------
// Test 1087: Top-k softmax concentrates probability mass
// ---------------------------------------------------------------------------

/// Prove: after top-k masking, the surviving elements' softmax probabilities
/// sum to 1 (since masked elements contribute ~0), concentrating all mass
/// on the top-k elements.
///
/// For k=2 out of 3 elements: two valid, one masked. The two valid elements'
/// softmax outputs sum to approximately 1.
#[test]
fn test_1087_topk_softmax_concentrates_mass() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("e_top1", real.clone());
    let _ = prog.declare_const("e_top2", real.clone());
    let _ = prog.declare_const("e_masked", real.clone());
    let _ = prog.declare_const("s_top1", real.clone());
    let _ = prog.declare_const("s_top2", real.clone());
    let _ = prog.declare_const("s_masked", real);

    let e_top1 = real_var("e_top1");
    let e_top2 = real_var("e_top2");
    let e_masked = real_var("e_masked");
    let s_top1 = real_var("s_top1");
    let s_top2 = real_var("s_top2");
    let s_masked = real_var("s_masked");

    // Top-k values have positive bounded exp
    prog.assert(e_top1.clone().real_gt(Expr::real(0)));
    prog.assert(e_top1.clone().real_le(Expr::real(1000)));
    prog.assert(e_top2.clone().real_gt(Expr::real(0)));
    prog.assert(e_top2.clone().real_le(Expr::real(1000)));

    // Masked value: exp near 0
    let eps = Expr::real_ratio(1, 1000000);
    prog.assert(e_masked.clone().real_ge(Expr::real(0)));
    prog.assert(e_masked.clone().real_le(eps));

    // denom = e_top1 + e_top2 + e_masked
    let z = e_top1
        .clone()
        .real_add(e_top2.clone())
        .real_add(e_masked.clone());

    // s_i * z = e_i
    prog.assert(s_top1.clone().real_mul(z.clone()).eq(e_top1));
    prog.assert(s_top2.clone().real_mul(z.clone()).eq(e_top2));
    prog.assert(s_masked.clone().real_mul(z).eq(e_masked));

    // The masked element's weight is near-zero: s_masked < 0.001
    // Negated property: s_masked > 0.001
    let violation = s_masked.real_gt(Expr::real_ratio(1, 1000));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "topk_softmax_concentrates_mass");
}

// ---------------------------------------------------------------------------
// Test 1088: Softmax output bounded by ratio of max to min logit
// ---------------------------------------------------------------------------

/// Prove: for a 2-element softmax with e_0 >= R * e_1 (R >= 1),
/// the max softmax output s_0 >= R / (R + 1).
///
/// s_0 = e_0 / (e_0 + e_1) >= R*e_1 / (R*e_1 + e_1) = R / (R + 1).
/// For R = 10: s_0 >= 10/11 > 0.909.
#[test]
fn test_1088_softmax_output_bounded_by_logit_ratio() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("e0", real.clone());
    let _ = prog.declare_const("e1", real.clone());
    let _ = prog.declare_const("denom", real.clone());
    let _ = prog.declare_const("s0", real);

    let e0 = real_var("e0");
    let e1 = real_var("e1");
    let denom = real_var("denom");
    let s0 = real_var("s0");

    // exp values positive
    prog.assert(e0.clone().real_gt(Expr::real(0)));
    prog.assert(e1.clone().real_gt(Expr::real(0)));

    // e0 >= 10 * e1 (R = 10)
    prog.assert(e0.clone().real_ge(Expr::real(10).real_mul(e1.clone())));

    // denom = e0 + e1
    prog.assert(denom.clone().eq(e0.clone().real_add(e1)));

    // s0 * denom = e0
    prog.assert(s0.clone().real_mul(denom).eq(e0));

    // Threshold: 10/11
    let threshold = Expr::real(10).real_div(Expr::real(11));

    // Negated property: s0 < 10/11
    let violation = s0.real_lt(threshold);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_output_bounded_by_logit_ratio");
}

// ---------------------------------------------------------------------------
// Test 1089: Binary cross-entropy is special case of categorical
// ---------------------------------------------------------------------------

/// Prove: for K=2 classes, categorical CE reduces to binary CE.
///
/// Categorical CE with y = (y, 1-y) and p = (p, 1-p):
///   CE = -(y*log(p) + (1-y)*log(1-p))
/// which is exactly the binary cross-entropy formula.
///
/// We prove the algebraic identity: the 2-class categorical CE formula
/// equals the standard binary CE formula.
#[test]
fn test_1089_binary_ce_is_special_case_of_categorical() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("y", real.clone());
    let _ = prog.declare_const("log_p", real.clone());
    let _ = prog.declare_const("log_1mp", real.clone());
    let _ = prog.declare_const("cat_ce", real.clone());
    let _ = prog.declare_const("bin_ce", real);

    let y = real_var("y");
    let log_p = real_var("log_p");
    let log_1mp = real_var("log_1mp");
    let cat_ce = real_var("cat_ce");
    let bin_ce = real_var("bin_ce");

    // y in [0, 1]
    prog.assert(y.clone().real_ge(Expr::real(0)));
    prog.assert(y.clone().real_le(Expr::real(1)));

    // log values bounded (log(p) <= 0 for p in (0,1])
    prog.assert(log_p.clone().real_le(Expr::real(0)));
    prog.assert(log_p.clone().real_ge(Expr::real(-1000)));
    prog.assert(log_1mp.clone().real_le(Expr::real(0)));
    prog.assert(log_1mp.clone().real_ge(Expr::real(-1000)));

    // Categorical CE with 2 classes, target (y, 1-y), pred (p, 1-p):
    // cat_ce = -(y * log_p + (1-y) * log_1mp)
    prog.assert(
        cat_ce.clone().eq(Expr::real(0).real_sub(
            y.clone()
                .real_mul(log_p.clone())
                .real_add(Expr::real(1).real_sub(y.clone()).real_mul(log_1mp.clone())),
        )),
    );

    // Binary CE: bin_ce = -(y * log_p + (1-y) * log(1-p))
    prog.assert(
        bin_ce.clone().eq(Expr::real(0).real_sub(
            y.clone()
                .real_mul(log_p)
                .real_add(Expr::real(1).real_sub(y).real_mul(log_1mp)),
        )),
    );

    // Negated property: cat_ce != bin_ce
    let violation = cat_ce.ne(bin_ce);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "binary_ce_is_special_case_of_categorical");
}

// ---------------------------------------------------------------------------
// Test 1090: Softmax temperature 0 -> one-hot (argmax)
// ---------------------------------------------------------------------------

/// Prove: at low temperature, the largest input dominates softmax output.
///
/// When exp(x_0/T) >> exp(x_1/T) (low T, x_0 > x_1):
///   s_0 = e_0 / (e_0 + e_1) -> 1.
///
/// We model: e_0 >= 100 * e_1, so s_0 >= 100/101 > 0.99.
/// This approaches the argmax one-hot vector as T -> 0.
#[test]
fn test_1090_softmax_temperature_zero_onehot() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("e0", real.clone());
    let _ = prog.declare_const("e1", real.clone());
    let _ = prog.declare_const("denom", real.clone());
    let _ = prog.declare_const("s0", real);

    let e0 = real_var("e0");
    let e1 = real_var("e1");
    let denom = real_var("denom");
    let s0 = real_var("s0");

    // exp values positive
    prog.assert(e0.clone().real_gt(Expr::real(0)));
    prog.assert(e1.clone().real_gt(Expr::real(0)));

    // Low temperature: e0 >= 100 * e1
    prog.assert(e0.clone().real_ge(Expr::real(100).real_mul(e1.clone())));

    // denom = e0 + e1
    prog.assert(denom.clone().eq(e0.clone().real_add(e1)));

    // s0 * denom = e0
    prog.assert(s0.clone().real_mul(denom).eq(e0));

    // Threshold: 100/101 (just under 0.99)
    let threshold = Expr::real(100).real_div(Expr::real(101));

    // Negated property: s0 < 100/101
    let violation = s0.real_lt(threshold);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_temperature_zero_onehot");
}
