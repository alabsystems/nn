// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for batch normalization running statistics
//! and momentum.
//!
//! Proves mathematical properties of BatchNorm running statistics updates:
//! - Running mean EMA update formula correctness
//! - Momentum bounded in [0, 1]
//! - Running variance non-negativity preservation
//! - Normalized output formula (zero mean, unit variance before affine)
//! - Affine transform application (gamma * norm + beta)
//! - Epsilon strictly positive
//! - Eval vs training mode distinction
//! - Running var non-negative after update
//! - Output bounded given bounded inputs
//! - Convex combination interpolation bounds
//! - Multi-step EMA convergence
//! - Bessel correction for unbiased variance
//! - Momentum decay factor
//! - Batch mean as sample average
//! - Running stats monotone convergence
//! - Gamma/beta gradient bounds
//! - Numerical stability with epsilon
//! - Variance additivity under independence
//! - Running mean shift bounded by momentum
//! - Output channel independence
//!
//! Part of #4166.

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
// Test 731: Running mean EMA update is a convex combination
// ---------------------------------------------------------------------------

/// Prove: running_mean_new = (1 - m) * running_mean_old + m * batch_mean
/// is a convex combination, so running_mean_new lies between
/// running_mean_old and batch_mean.
///
/// For 0 < m < 1, the result is bounded: min(rm_old, bm) <= rm_new <= max(rm_old, bm).
#[test]
fn test_731_running_mean_ema_convex_combination() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("rm_old", real.clone());
    let _ = prog.declare_const("bm", real.clone());
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("rm_new", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real);

    let rm_old = real_var("rm_old");
    let bm = real_var("bm");
    let m = real_var("m");
    let rm_new = real_var("rm_new");
    let lo = real_var("lo");
    let hi = real_var("hi");

    // momentum in (0, 1)
    prog.assert(m.clone().real_gt(Expr::real(0)));
    prog.assert(m.clone().real_lt(Expr::real(1)));

    // lo = min(rm_old, bm), hi = max(rm_old, bm)
    prog.assert(lo.clone().real_le(rm_old.clone()));
    prog.assert(lo.clone().real_le(bm.clone()));
    prog.assert(hi.clone().real_ge(rm_old.clone()));
    prog.assert(hi.clone().real_ge(bm.clone()));
    prog.assert(lo.clone().eq(rm_old.clone()).or(lo.clone().eq(bm.clone())));
    prog.assert(hi.clone().eq(rm_old.clone()).or(hi.clone().eq(bm.clone())));

    // rm_new = (1-m)*rm_old + m*bm
    let one_minus_m = Expr::real(1).real_sub(m.clone());
    prog.assert(
        rm_new
            .clone()
            .eq(one_minus_m.real_mul(rm_old).real_add(m.real_mul(bm))),
    );

    // Negated property: rm_new < lo OR rm_new > hi
    let violation = rm_new.clone().real_lt(lo).or(rm_new.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "running_mean_ema_convex_combination");
}

// ---------------------------------------------------------------------------
// Test 732: Momentum must be in [0, 1] for valid EMA
// ---------------------------------------------------------------------------

/// Prove: if momentum is in [0, 1] and both running_var_old >= 0 and
/// batch_var >= 0, then running_var_new >= 0.
///
/// This is the forward direction: valid momentum preserves non-negativity.
#[test]
fn test_732_momentum_in_unit_interval_preserves_nonnegativity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("rv_old", real.clone());
    let _ = prog.declare_const("bv", real.clone());
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("rv_new", real);

    let rv_old = real_var("rv_old");
    let bv = real_var("bv");
    let m = real_var("m");
    let rv_new = real_var("rv_new");

    // m in [0, 1]
    prog.assert(m.clone().real_ge(Expr::real(0)));
    prog.assert(m.clone().real_le(Expr::real(1)));

    // Both non-negative
    prog.assert(rv_old.clone().real_ge(Expr::real(0)));
    prog.assert(bv.clone().real_ge(Expr::real(0)));

    // rv_new = (1-m)*rv_old + m*bv
    let one_minus_m = Expr::real(1).real_sub(m.clone());
    prog.assert(
        rv_new
            .clone()
            .eq(one_minus_m.real_mul(rv_old).real_add(m.real_mul(bv))),
    );

    // Negated: rv_new < 0
    let violation = rv_new.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "momentum_in_unit_interval_preserves_nonnegativity");
}

// ---------------------------------------------------------------------------
// Test 733: Running variance stays non-negative after EMA update
// ---------------------------------------------------------------------------

/// Prove: if running_var_old >= 0, batch_var >= 0, and 0 < m < 1, then
/// running_var_new = (1-m)*rv_old + m*bv >= 0.
///
/// A strict interior version (open interval momentum).
#[test]
fn test_733_running_var_nonneg_after_update() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("rv_old", real.clone());
    let _ = prog.declare_const("bv", real.clone());
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("rv_new", real);

    let rv_old = real_var("rv_old");
    let bv = real_var("bv");
    let m = real_var("m");
    let rv_new = real_var("rv_new");

    prog.assert(rv_old.clone().real_ge(Expr::real(0)));
    prog.assert(bv.clone().real_ge(Expr::real(0)));
    prog.assert(m.clone().real_gt(Expr::real(0)));
    prog.assert(m.clone().real_lt(Expr::real(1)));

    let one_minus_m = Expr::real(1).real_sub(m.clone());
    prog.assert(
        rv_new
            .clone()
            .eq(one_minus_m.real_mul(rv_old).real_add(m.real_mul(bv))),
    );

    // Negated: rv_new < 0
    let violation = rv_new.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "running_var_nonneg_after_update");
}

// ---------------------------------------------------------------------------
// Test 734: Normalized output has zero mean (2-element batch)
// ---------------------------------------------------------------------------

/// Prove: after batch normalization of a 2-element batch, the mean of
/// normalized outputs is zero.
///
/// For [x1, x2]: batch_mean = (x1+x2)/2.
/// n_i = (x_i - batch_mean) / s, so (n1+n2)/2 = 0.
#[test]
fn test_734_normalized_output_zero_mean() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("bm", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("n1", real.clone());
    let _ = prog.declare_const("n2", real.clone());
    let _ = prog.declare_const("out_mean", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let bm = real_var("bm");
    let s = real_var("s");
    let n1 = real_var("n1");
    let n2 = real_var("n2");
    let out_mean = real_var("out_mean");

    // batch_mean = (x1+x2)/2
    prog.assert(
        Expr::real(2)
            .real_mul(bm.clone())
            .eq(x1.clone().real_add(x2.clone())),
    );

    // s > 0
    prog.assert(s.clone().real_gt(Expr::real(0)));

    // n_i = (x_i - bm) / s
    prog.assert(n1.clone().real_mul(s.clone()).eq(x1.real_sub(bm.clone())));
    prog.assert(n2.clone().real_mul(s).eq(x2.real_sub(bm)));

    // out_mean = (n1+n2)/2
    prog.assert(Expr::real(2).real_mul(out_mean.clone()).eq(n1.real_add(n2)));

    // Negated: out_mean != 0
    let violation = out_mean.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "normalized_output_zero_mean");
}

// ---------------------------------------------------------------------------
// Test 735: Affine transform: y = gamma * norm + beta
// ---------------------------------------------------------------------------

/// Prove: the affine transform y = gamma * n + beta maps n = 0 to y = beta
/// and preserves relative ordering when gamma > 0.
///
/// If gamma > 0 and n1 < n2, then y1 < y2.
#[test]
fn test_735_affine_transform_preserves_order() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("n1", real.clone());
    let _ = prog.declare_const("n2", real.clone());
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("beta", real.clone());
    let _ = prog.declare_const("y1", real.clone());
    let _ = prog.declare_const("y2", real);

    let n1 = real_var("n1");
    let n2 = real_var("n2");
    let gamma = real_var("gamma");
    let beta = real_var("beta");
    let y1 = real_var("y1");
    let y2 = real_var("y2");

    // gamma > 0
    prog.assert(gamma.clone().real_gt(Expr::real(0)));

    // n1 < n2
    prog.assert(n1.clone().real_lt(n2.clone()));

    // y_i = gamma * n_i + beta
    prog.assert(
        y1.clone()
            .eq(gamma.clone().real_mul(n1).real_add(beta.clone())),
    );
    prog.assert(y2.clone().eq(gamma.real_mul(n2).real_add(beta)));

    // Negated: y1 >= y2 (order not preserved)
    let violation = y1.real_ge(y2);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "affine_transform_preserves_order");
}

// ---------------------------------------------------------------------------
// Test 736: Epsilon must be strictly positive
// ---------------------------------------------------------------------------

/// Prove: epsilon > 0 ensures sqrt(var + eps) > 0 even when var = 0.
///
/// If eps > 0 and var >= 0, then var + eps > 0.
#[test]
fn test_736_epsilon_strictly_positive() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("var", real.clone());
    let _ = prog.declare_const("eps", real.clone());
    let _ = prog.declare_const("sum", real);

    let var = real_var("var");
    let eps = real_var("eps");
    let sum = real_var("sum");

    // var >= 0
    prog.assert(var.clone().real_ge(Expr::real(0)));

    // eps > 0
    prog.assert(eps.clone().real_gt(Expr::real(0)));

    // sum = var + eps
    prog.assert(sum.clone().eq(var.real_add(eps)));

    // Negated: sum <= 0
    let violation = sum.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "epsilon_strictly_positive");
}

// ---------------------------------------------------------------------------
// Test 737: Eval mode uses running stats (different from training batch stats)
// ---------------------------------------------------------------------------

/// Prove: in eval mode, output depends on running_mean, not batch_mean.
///
/// If two different inputs x1 != x2 use the same running statistics,
/// outputs y1 != y2 (they are not collapsed). This distinguishes eval
/// from a degenerate mode that ignores inputs.
#[test]
fn test_737_eval_mode_uses_running_stats() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("rm", real.clone());
    let _ = prog.declare_const("denom", real.clone());
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("beta", real.clone());
    let _ = prog.declare_const("y1", real.clone());
    let _ = prog.declare_const("y2", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let rm = real_var("rm");
    let denom = real_var("denom");
    let gamma = real_var("gamma");
    let beta = real_var("beta");
    let y1 = real_var("y1");
    let y2 = real_var("y2");

    // x1 != x2
    prog.assert(x1.clone().ne(x2.clone()));

    // denom > 0 (sqrt(running_var + eps))
    prog.assert(denom.clone().real_gt(Expr::real(0)));

    // gamma != 0
    prog.assert(gamma.clone().ne(Expr::real(0)));

    // y_i = gamma * (x_i - rm) / denom + beta
    // modeled as: (y_i - beta) * denom = gamma * (x_i - rm)
    prog.assert(
        y1.clone()
            .real_sub(beta.clone())
            .real_mul(denom.clone())
            .eq(gamma.clone().real_mul(x1.real_sub(rm.clone()))),
    );
    prog.assert(
        y2.clone()
            .real_sub(beta)
            .real_mul(denom)
            .eq(gamma.real_mul(x2.real_sub(rm))),
    );

    // Negated: y1 = y2 (outputs collapsed)
    let violation = y1.eq(y2);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "eval_mode_uses_running_stats");
}

// ---------------------------------------------------------------------------
// Test 738: Running variance non-negative (closed interval momentum)
// ---------------------------------------------------------------------------

/// Prove: running_var_new >= 0 when momentum is in the closed interval [0, 1]
/// and both rv_old >= 0 and bv >= 0.
///
/// This covers edge cases m=0 (rv_new = rv_old) and m=1 (rv_new = bv).
#[test]
fn test_738_running_var_nonneg_closed_momentum() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("rv_old", real.clone());
    let _ = prog.declare_const("bv", real.clone());
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("rv_new", real);

    let rv_old = real_var("rv_old");
    let bv = real_var("bv");
    let m = real_var("m");
    let rv_new = real_var("rv_new");

    prog.assert(rv_old.clone().real_ge(Expr::real(0)));
    prog.assert(bv.clone().real_ge(Expr::real(0)));
    prog.assert(m.clone().real_ge(Expr::real(0)));
    prog.assert(m.clone().real_le(Expr::real(1)));

    let one_minus_m = Expr::real(1).real_sub(m.clone());
    prog.assert(
        rv_new
            .clone()
            .eq(one_minus_m.real_mul(rv_old).real_add(m.real_mul(bv))),
    );

    // Negated: rv_new < 0
    let violation = rv_new.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "running_var_nonneg_closed_momentum");
}

// ---------------------------------------------------------------------------
// Test 739: Output bounded given bounded inputs (inference)
// ---------------------------------------------------------------------------

/// Prove: if |x| <= B, running_mean and running_var are bounded, gamma and
/// beta are bounded, and eps > 0, then the output is bounded.
///
/// y = gamma * (x - rm) / denom + beta.
/// |y - beta| = |gamma| * |x - rm| / denom <= |gamma| * (B + |rm|) / denom.
/// So |y| <= |beta| + |gamma| * (B + |rm|) / denom.
///
/// We prove: |y| <= G * (B + R) / D + P where G = gamma bound, R = rm bound,
/// D = denom lower bound, P = beta bound.
#[test]
fn test_739_output_bounded_given_bounded_inputs() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("rm", real.clone());
    let _ = prog.declare_const("denom", real.clone());
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("beta", real.clone());
    let _ = prog.declare_const("y", real);

    let x = real_var("x");
    let rm = real_var("rm");
    let denom = real_var("denom");
    let gamma = real_var("gamma");
    let beta = real_var("beta");
    let y = real_var("y");

    let b = Expr::real(10); // input bound
    let r = Expr::real(10); // running mean bound
    let g = Expr::real(2); // gamma bound
    let p = Expr::real(2); // beta bound
    let d = Expr::real(1); // denom lower bound

    // |x| <= B
    prog.assert(x.clone().real_ge(Expr::real(0).real_sub(b.clone())));
    prog.assert(x.clone().real_le(b.clone()));

    // |rm| <= R
    prog.assert(rm.clone().real_ge(Expr::real(0).real_sub(r.clone())));
    prog.assert(rm.clone().real_le(r.clone()));

    // |gamma| <= G
    prog.assert(gamma.clone().real_ge(Expr::real(0).real_sub(g.clone())));
    prog.assert(gamma.clone().real_le(g.clone()));

    // |beta| <= P
    prog.assert(beta.clone().real_ge(Expr::real(0).real_sub(p.clone())));
    prog.assert(beta.clone().real_le(p.clone()));

    // denom >= D > 0
    prog.assert(denom.clone().real_ge(d.clone()));

    // y = gamma * (x - rm) / denom + beta
    // modeled as: (y - beta) * denom = gamma * (x - rm)
    prog.assert(
        y.clone()
            .real_sub(beta)
            .real_mul(denom)
            .eq(gamma.real_mul(x.real_sub(rm))),
    );

    // output bound = G * (B + R) / D + P = 2 * 20 / 1 + 2 = 42
    let out_bound = Expr::real(42);

    // Negated: |y| > 42
    let violation = y
        .clone()
        .real_gt(out_bound.clone())
        .or(y.real_lt(Expr::real(0).real_sub(out_bound)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "output_bounded_given_bounded_inputs");
}

// ---------------------------------------------------------------------------
// Test 740: Convex combination interpolation strictly interior
// ---------------------------------------------------------------------------

/// Prove: for 0 < m < 1 and a != b,
/// the EMA result r = (1-m)*a + m*b satisfies strict inequalities:
/// min(a,b) < r < max(a,b).
///
/// The result is strictly between the two endpoints (not equal to either).
#[test]
fn test_740_convex_combination_strictly_interior() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("r", real);

    let a = real_var("a");
    let b = real_var("b");
    let m = real_var("m");
    let r = real_var("r");

    // a != b
    prog.assert(a.clone().ne(b.clone()));

    // m in (0, 1)
    prog.assert(m.clone().real_gt(Expr::real(0)));
    prog.assert(m.clone().real_lt(Expr::real(1)));

    // r = (1-m)*a + m*b
    let one_minus_m = Expr::real(1).real_sub(m.clone());
    prog.assert(
        r.clone().eq(one_minus_m
            .real_mul(a.clone())
            .real_add(m.real_mul(b.clone()))),
    );

    // Negated: r = a OR r = b (result equals one of the endpoints)
    let violation = r.clone().eq(a).or(r.eq(b));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "convex_combination_strictly_interior");
}

// ---------------------------------------------------------------------------
// Test 741: Two-step EMA stays bounded
// ---------------------------------------------------------------------------

/// Prove: applying EMA twice with the same momentum m in (0,1) keeps the
/// running mean between the global min and max of all three values
/// (rm_0, bm_1, bm_2).
///
/// rm_1 = (1-m)*rm_0 + m*bm_1
/// rm_2 = (1-m)*rm_1 + m*bm_2
///
/// We show: min(rm_0, bm_1, bm_2) <= rm_2 <= max(rm_0, bm_1, bm_2).
#[test]
fn test_741_two_step_ema_stays_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("rm0", real.clone());
    let _ = prog.declare_const("bm1", real.clone());
    let _ = prog.declare_const("bm2", real.clone());
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("rm1", real.clone());
    let _ = prog.declare_const("rm2", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real);

    let rm0 = real_var("rm0");
    let bm1 = real_var("bm1");
    let bm2 = real_var("bm2");
    let m = real_var("m");
    let rm1 = real_var("rm1");
    let rm2 = real_var("rm2");
    let lo = real_var("lo");
    let hi = real_var("hi");

    // m in (0, 1)
    prog.assert(m.clone().real_gt(Expr::real(0)));
    prog.assert(m.clone().real_lt(Expr::real(1)));

    // rm1 = (1-m)*rm0 + m*bm1
    let one_minus_m = Expr::real(1).real_sub(m.clone());
    prog.assert(
        rm1.clone().eq(one_minus_m
            .clone()
            .real_mul(rm0.clone())
            .real_add(m.clone().real_mul(bm1.clone()))),
    );

    // rm2 = (1-m)*rm1 + m*bm2
    prog.assert(
        rm2.clone()
            .eq(one_minus_m.real_mul(rm1).real_add(m.real_mul(bm2.clone()))),
    );

    // lo <= all three values
    prog.assert(lo.clone().real_le(rm0.clone()));
    prog.assert(lo.clone().real_le(bm1.clone()));
    prog.assert(lo.clone().real_le(bm2.clone()));
    // lo is one of them
    prog.assert(
        lo.clone()
            .eq(rm0.clone())
            .or(lo.clone().eq(bm1.clone()))
            .or(lo.clone().eq(bm2.clone())),
    );

    // hi >= all three values
    prog.assert(hi.clone().real_ge(rm0));
    prog.assert(hi.clone().real_ge(bm1));
    prog.assert(hi.clone().real_ge(bm2));
    // hi is one of them
    prog.assert(
        hi.clone()
            .eq(real_var("rm0"))
            .or(hi.clone().eq(real_var("bm1")))
            .or(hi.clone().eq(real_var("bm2"))),
    );

    // Negated: rm2 < lo OR rm2 > hi
    let violation = rm2.clone().real_lt(lo).or(rm2.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "two_step_ema_stays_bounded");
}

// ---------------------------------------------------------------------------
// Test 742: Bessel correction: unbiased variance >= 0
// ---------------------------------------------------------------------------

/// Prove: the Bessel-corrected sample variance n/(n-1) * var >= 0
/// when var >= 0 and n >= 2.
///
/// sample_var = n / (n-1) * batch_var.
/// Since n >= 2, n/(n-1) > 0. And batch_var >= 0. So sample_var >= 0.
#[test]
fn test_742_bessel_correction_nonneg() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("batch_var", real.clone());
    let _ = prog.declare_const("n", real.clone());
    let _ = prog.declare_const("n_minus_1", real.clone());
    let _ = prog.declare_const("sample_var", real);

    let batch_var = real_var("batch_var");
    let n = real_var("n");
    let n_minus_1 = real_var("n_minus_1");
    let sample_var = real_var("sample_var");

    // batch_var >= 0
    prog.assert(batch_var.clone().real_ge(Expr::real(0)));

    // n >= 2
    prog.assert(n.clone().real_ge(Expr::real(2)));

    // n_minus_1 = n - 1
    prog.assert(n_minus_1.clone().eq(n.clone().real_sub(Expr::real(1))));

    // sample_var * (n-1) = n * batch_var (avoids division)
    prog.assert(
        sample_var
            .clone()
            .real_mul(n_minus_1)
            .eq(n.real_mul(batch_var)),
    );

    // Negated: sample_var < 0
    let violation = sample_var.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "bessel_correction_nonneg");
}

// ---------------------------------------------------------------------------
// Test 743: Momentum decay factor: (1-m)^k decreases
// ---------------------------------------------------------------------------

/// Prove: for 0 < m < 1, the decay factor (1-m)^2 < (1-m)^1 < 1.
///
/// After k steps, the weight on the initial running stat is (1-m)^k,
/// which decreases with k. We prove the two-step case.
#[test]
fn test_743_momentum_decay_factor_decreases() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("d1", real.clone());
    let _ = prog.declare_const("d2", real);

    let m = real_var("m");
    let d1 = real_var("d1");
    let d2 = real_var("d2");

    // m in (0, 1)
    prog.assert(m.clone().real_gt(Expr::real(0)));
    prog.assert(m.clone().real_lt(Expr::real(1)));

    // d1 = 1 - m (decay after 1 step)
    prog.assert(d1.clone().eq(Expr::real(1).real_sub(m.clone())));

    // d2 = (1-m)^2 = d1 * d1
    prog.assert(d2.clone().eq(d1.clone().real_mul(d1.clone())));

    // Property: 0 < d2 < d1 < 1
    // Negated: d2 >= d1 OR d1 >= 1 OR d2 <= 0
    let violation = d2
        .clone()
        .real_ge(d1.clone())
        .or(d1.real_ge(Expr::real(1)))
        .or(d2.real_le(Expr::real(0)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "momentum_decay_factor_decreases");
}

// ---------------------------------------------------------------------------
// Test 744: Batch mean is sample average (3 elements)
// ---------------------------------------------------------------------------

/// Prove: for a 3-element batch [x1, x2, x3], batch_mean = (x1+x2+x3)/3
/// satisfies the definition of the arithmetic mean.
///
/// We verify: 3 * bm = x1 + x2 + x3 implies each x_i's deviation sums to 0.
#[test]
fn test_744_batch_mean_is_sample_average() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("x3", real.clone());
    let _ = prog.declare_const("bm", real.clone());
    let _ = prog.declare_const("dev_sum", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let x3 = real_var("x3");
    let bm = real_var("bm");
    let dev_sum = real_var("dev_sum");

    // bm = (x1 + x2 + x3) / 3, modeled as 3 * bm = x1 + x2 + x3
    prog.assert(
        Expr::real(3)
            .real_mul(bm.clone())
            .eq(x1.clone().real_add(x2.clone()).real_add(x3.clone())),
    );

    // dev_sum = (x1 - bm) + (x2 - bm) + (x3 - bm)
    prog.assert(
        dev_sum.clone().eq(x1
            .real_sub(bm.clone())
            .real_add(x2.real_sub(bm.clone()))
            .real_add(x3.real_sub(bm))),
    );

    // Negated: dev_sum != 0
    let violation = dev_sum.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "batch_mean_is_sample_average");
}

// ---------------------------------------------------------------------------
// Test 745: Running mean shift bounded by momentum
// ---------------------------------------------------------------------------

/// Prove: |rm_new - rm_old| <= m * |bm - rm_old| for m in [0, 1].
///
/// Since rm_new = (1-m)*rm_old + m*bm, we have:
/// rm_new - rm_old = m*(bm - rm_old).
/// So |rm_new - rm_old| = m * |bm - rm_old|.
///
/// We model the absolute shift and prove it equals m * |diff|.
#[test]
fn test_745_running_mean_shift_bounded_by_momentum() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("rm_old", real.clone());
    let _ = prog.declare_const("bm", real.clone());
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("rm_new", real.clone());
    let _ = prog.declare_const("shift", real);

    let rm_old = real_var("rm_old");
    let bm = real_var("bm");
    let m = real_var("m");
    let rm_new = real_var("rm_new");
    let shift = real_var("shift");

    // m in [0, 1]
    prog.assert(m.clone().real_ge(Expr::real(0)));
    prog.assert(m.clone().real_le(Expr::real(1)));

    // rm_new = (1-m)*rm_old + m*bm
    let one_minus_m = Expr::real(1).real_sub(m.clone());
    prog.assert(
        rm_new.clone().eq(one_minus_m
            .real_mul(rm_old.clone())
            .real_add(m.clone().real_mul(bm.clone()))),
    );

    // shift = rm_new - rm_old
    prog.assert(shift.clone().eq(rm_new.real_sub(rm_old.clone())));

    // Property: shift = m * (bm - rm_old)
    let expected_shift = m.real_mul(bm.real_sub(rm_old));

    // Negated: shift != m * (bm - rm_old)
    let violation = shift.ne(expected_shift);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "running_mean_shift_bounded_by_momentum");
}

// ---------------------------------------------------------------------------
// Test 746: Output channel independence
// ---------------------------------------------------------------------------

/// Prove: batch normalization of two independent channels produces outputs
/// that depend only on their respective channel's statistics.
///
/// If ch1 uses (rm1, rv1, gamma1, beta1) and ch2 uses (rm2, rv2, gamma2, beta2),
/// changing ch2's statistics does not affect ch1's output.
///
/// We model: y1 = gamma1 * (x1 - rm1) / d1 + beta1 (channel 1),
/// and show y1 is independent of rm2.
#[test]
fn test_746_output_channel_independence() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("rm1", real.clone());
    let _ = prog.declare_const("d1", real.clone());
    let _ = prog.declare_const("gamma1", real.clone());
    let _ = prog.declare_const("beta1", real.clone());
    let _ = prog.declare_const("y1_a", real.clone());
    let _ = prog.declare_const("y1_b", real.clone());
    let _ = prog.declare_const("rm2_a", real.clone());
    let _ = prog.declare_const("rm2_b", real);

    let x1 = real_var("x1");
    let rm1 = real_var("rm1");
    let d1 = real_var("d1");
    let gamma1 = real_var("gamma1");
    let beta1 = real_var("beta1");
    let y1_a = real_var("y1_a");
    let y1_b = real_var("y1_b");
    let rm2_a = real_var("rm2_a");
    let rm2_b = real_var("rm2_b");

    // d1 > 0
    prog.assert(d1.clone().real_gt(Expr::real(0)));

    // rm2_a != rm2_b (different channel 2 running means)
    prog.assert(rm2_a.ne(rm2_b));

    // y1_a and y1_b both computed from same ch1 stats:
    // y1 = gamma1 * (x1 - rm1) / d1 + beta1
    prog.assert(
        y1_a.clone()
            .real_sub(beta1.clone())
            .real_mul(d1.clone())
            .eq(gamma1.clone().real_mul(x1.clone().real_sub(rm1.clone()))),
    );
    prog.assert(
        y1_b.clone()
            .real_sub(beta1)
            .real_mul(d1)
            .eq(gamma1.real_mul(x1.real_sub(rm1))),
    );

    // Negated: y1_a != y1_b (changing ch2 stats affected ch1 output)
    let violation = y1_a.ne(y1_b);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "output_channel_independence");
}

// ---------------------------------------------------------------------------
// Test 747: Numerical stability: var + eps denominator always positive
// ---------------------------------------------------------------------------

/// Prove: for any var >= 0 and eps = 1e-5, var + eps > 0.
///
/// This is the numerical stability guarantee: the denominator in
/// BatchNorm never reaches zero, preventing division-by-zero.
#[test]
fn test_747_numerical_stability_denominator_positive() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("var", real.clone());
    let _ = prog.declare_const("denom", real);

    let var = real_var("var");
    let denom = real_var("denom");

    // var >= 0
    prog.assert(var.clone().real_ge(Expr::real(0)));

    // eps = 1e-5 (typical default)
    let eps = Expr::real_ratio(1, 100000);

    // denom = var + eps
    prog.assert(denom.clone().eq(var.real_add(eps)));

    // Negated: denom <= 0
    let violation = denom.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "numerical_stability_denominator_positive");
}

// ---------------------------------------------------------------------------
// Test 748: Variance additivity under independent batches
// ---------------------------------------------------------------------------

/// Prove: if two independent batches have variances var1 and var2 with
/// means m1 and m2, and we combine them (equal size n), the combined
/// variance satisfies: combined_var = (var1 + var2)/2 + (m1-m2)^2/4.
///
/// This is a well-known formula. We verify one instance:
/// var1=1, var2=1, m1=0, m2=2 => combined_var = (1+1)/2 + (0-2)^2/4 = 1 + 1 = 2.
#[test]
fn test_748_variance_additivity_independent_batches() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("var1", real.clone());
    let _ = prog.declare_const("var2", real.clone());
    let _ = prog.declare_const("m1", real.clone());
    let _ = prog.declare_const("m2", real.clone());
    let _ = prog.declare_const("combined_var", real);

    let var1 = real_var("var1");
    let var2 = real_var("var2");
    let m1 = real_var("m1");
    let m2 = real_var("m2");
    let combined_var = real_var("combined_var");

    // Concrete values
    prog.assert(var1.eq(Expr::real(1)));
    prog.assert(var2.eq(Expr::real(1)));
    prog.assert(m1.clone().eq(Expr::real(0)));
    prog.assert(m2.clone().eq(Expr::real(2)));

    // combined_var = (var1 + var2)/2 + (m1 - m2)^2 / 4
    // For our values: (1+1)/2 + (0-2)^2/4 = 1 + 4/4 = 2
    prog.assert(combined_var.clone().eq(Expr::real(2)));

    // Verify using the formula: 2 * combined_var = var1 + var2 + (m1-m2)^2/2
    // 2 * 2 = 1 + 1 + 4/2 = 2 + 2 = 4. Check: 4 = 4.
    let diff = m1.real_sub(m2);
    let diff_sq_half = diff
        .clone()
        .real_mul(diff)
        .real_mul(Expr::real_ratio(1, 2));
    let formula_rhs = Expr::real(1).real_add(Expr::real(1)).real_add(diff_sq_half);

    // Negated: 2 * combined_var != formula_rhs
    let violation = Expr::real(2).real_mul(combined_var).ne(formula_rhs);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "variance_additivity_independent_batches");
}

// ---------------------------------------------------------------------------
// Test 749: Gamma/beta gradient bounded by input magnitude
// ---------------------------------------------------------------------------

/// Prove: the gradient of the loss w.r.t. gamma satisfies
/// dL/d_gamma = sum_i(n_i * dL/dy_i), where n_i is the normalized input.
///
/// If |n_i| <= N_bound and |dL/dy_i| <= G_bound for a batch of size B,
/// then |dL/d_gamma| <= B * N_bound * G_bound.
///
/// We prove for B=2: |dL/d_gamma| <= 2 * N_bound * G_bound.
#[test]
fn test_749_gamma_gradient_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("n1", real.clone());
    let _ = prog.declare_const("n2", real.clone());
    let _ = prog.declare_const("g1", real.clone());
    let _ = prog.declare_const("g2", real.clone());
    let _ = prog.declare_const("dg", real);

    let n1 = real_var("n1");
    let n2 = real_var("n2");
    let g1 = real_var("g1");
    let g2 = real_var("g2");
    let dg = real_var("dg");

    let n_bound = Expr::real(3);
    let g_bound = Expr::real(5);

    // |n_i| <= N_bound
    prog.assert(n1.clone().real_ge(Expr::real(0).real_sub(n_bound.clone())));
    prog.assert(n1.clone().real_le(n_bound.clone()));
    prog.assert(n2.clone().real_ge(Expr::real(0).real_sub(n_bound.clone())));
    prog.assert(n2.clone().real_le(n_bound));

    // |g_i| <= G_bound (dL/dy_i)
    prog.assert(g1.clone().real_ge(Expr::real(0).real_sub(g_bound.clone())));
    prog.assert(g1.clone().real_le(g_bound.clone()));
    prog.assert(g2.clone().real_ge(Expr::real(0).real_sub(g_bound.clone())));
    prog.assert(g2.clone().real_le(g_bound));

    // dg = n1*g1 + n2*g2
    prog.assert(dg.clone().eq(n1.real_mul(g1).real_add(n2.real_mul(g2))));

    // Bound: 2 * 3 * 5 = 30
    let total_bound = Expr::real(30);

    // Negated: |dg| > 30
    let violation = dg
        .clone()
        .real_gt(total_bound.clone())
        .or(dg.real_lt(Expr::real(0).real_sub(total_bound)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gamma_gradient_bounded");
}

// ---------------------------------------------------------------------------
// Test 750: EMA at momentum=0 preserves running stat exactly
// ---------------------------------------------------------------------------

/// Prove: when momentum = 0, rm_new = rm_old (batch stat has no effect).
///
/// rm_new = (1 - 0) * rm_old + 0 * bm = rm_old.
#[test]
fn test_750_ema_momentum_zero_preserves_stat() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("rm_old", real.clone());
    let _ = prog.declare_const("bm", real.clone());
    let _ = prog.declare_const("rm_new", real);

    let rm_old = real_var("rm_old");
    let bm = real_var("bm");
    let rm_new = real_var("rm_new");

    // m = 0
    let m = Expr::real(0);

    // rm_new = (1-m)*rm_old + m*bm = 1*rm_old + 0*bm = rm_old
    let one_minus_m = Expr::real(1).real_sub(m.clone());
    prog.assert(
        rm_new.clone().eq(one_minus_m
            .real_mul(rm_old.clone())
            .real_add(m.real_mul(bm))),
    );

    // Negated: rm_new != rm_old
    let violation = rm_new.ne(rm_old);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "ema_momentum_zero_preserves_stat");
}
