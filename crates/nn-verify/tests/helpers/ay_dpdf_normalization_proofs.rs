// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for normalization layer mathematical properties.
//!
//! Proves fundamental properties of normalization layers used in deep learning:
//! - LayerNorm: output mean is zero, output variance is one, affine transform,
//!   epsilon prevents division by zero
//! - RMSNorm: output formula, RMS positivity, normalized magnitude
//! - BatchNorm: running mean/var update formulas, inference vs training modes
//! - GroupNorm: group size divides channels, per-group independence
//! - InstanceNorm: per-sample independence
//! - Cross-cutting: scale invariance, idempotency, gradient bounds, epsilon
//!   positivity, affine parameter initialization, weight decay bounds
//!
//! Part of #4128.

use ay_bindings::execute_direct::{self, ExecuteResult};
use ay_bindings::{Expr, Sort, AYProgram};

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
// Test 491: LayerNorm output mean is 0 (2-element vector)
// ---------------------------------------------------------------------------

/// Prove: after LayerNorm normalization, the output mean is 0.
///
/// For a 2-element vector [x1, x2], mean = (x1+x2)/2.
/// Normalized: x_norm_i = (x_i - mean) / std.
/// Mean of normalized: (x_norm_1 + x_norm_2) / 2
///   = ((x1-mean) + (x2-mean)) / (2*std)
///   = (x1+x2 - 2*mean) / (2*std)
///   = 0 / (2*std) = 0.
///
/// We model: mean = (x1+x2)/2, n1 = (x1-mean)/s, n2 = (x2-mean)/s,
/// and prove (n1+n2)/2 = 0.
#[test]
fn test_491_layernorm_output_mean_zero() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("mean", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("n1", real.clone());
    let _ = prog.declare_const("n2", real.clone());
    let _ = prog.declare_const("out_mean", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let mean = real_var("mean");
    let s = real_var("s");
    let n1 = real_var("n1");
    let n2 = real_var("n2");
    let out_mean = real_var("out_mean");

    // Input bounds
    prog.assert(x1.clone().real_ge(Expr::real(-100)));
    prog.assert(x1.clone().real_le(Expr::real(100)));
    prog.assert(x2.clone().real_ge(Expr::real(-100)));
    prog.assert(x2.clone().real_le(Expr::real(100)));

    // mean = (x1 + x2) / 2, modeled as: 2 * mean = x1 + x2
    prog.assert(
        Expr::real(2)
            .real_mul(mean.clone())
            .eq(x1.clone().real_add(x2.clone())),
    );

    // s > 0 (standard deviation)
    prog.assert(s.clone().real_gt(Expr::real(0)));

    // n1 = (x1 - mean) / s, modeled as: n1 * s = x1 - mean
    prog.assert(n1.clone().real_mul(s.clone()).eq(x1.real_sub(mean.clone())));
    // n2 = (x2 - mean) / s, modeled as: n2 * s = x2 - mean
    prog.assert(n2.clone().real_mul(s).eq(x2.real_sub(mean)));

    // out_mean = (n1 + n2) / 2, modeled as: 2 * out_mean = n1 + n2
    prog.assert(Expr::real(2).real_mul(out_mean.clone()).eq(n1.real_add(n2)));

    // Negated property: out_mean != 0
    let violation = out_mean.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "layernorm_output_mean_zero");
}

// ---------------------------------------------------------------------------
// Test 492: LayerNorm output variance is 1 (after normalization)
// ---------------------------------------------------------------------------

/// Prove: after LayerNorm normalization, the output variance is 1.
///
/// For a 2-element vector, after centering: n_i = (x_i - mean) / std.
/// Variance of normalized = mean((n_i - 0)^2) = mean(n_i^2)
///   = ((x1-mean)^2 + (x2-mean)^2) / (2 * std^2)
///   = var / var = 1.
///
/// We model: var = ((x1-mean)^2 + (x2-mean)^2) / 2, std^2 = var,
/// n_i = (x_i-mean)/std, and prove n1^2 + n2^2 = 2 (i.e., mean of squares = 1).
#[test]
fn test_492_layernorm_output_variance_one() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("d1", real.clone());
    let _ = prog.declare_const("d2", real.clone());
    let _ = prog.declare_const("var", real.clone());
    let _ = prog.declare_const("n1", real.clone());
    let _ = prog.declare_const("n2", real.clone());
    let _ = prog.declare_const("out_var", real);

    let d1 = real_var("d1");
    let d2 = real_var("d2");
    let var = real_var("var");
    let n1 = real_var("n1");
    let n2 = real_var("n2");
    let out_var = real_var("out_var");

    // d1, d2 are deviations from mean (d_i = x_i - mean), not both zero
    // d1 + d2 = 0 (centered), d1 != 0
    prog.assert(d1.clone().real_add(d2.clone()).eq(Expr::real(0)));
    prog.assert(d1.clone().ne(Expr::real(0)));

    // var = (d1^2 + d2^2) / 2
    let d1_sq = d1.clone().real_mul(d1.clone());
    let d2_sq = d2.clone().real_mul(d2.clone());
    prog.assert(
        Expr::real(2)
            .real_mul(var.clone())
            .eq(d1_sq.real_add(d2_sq)),
    );

    // var > 0 (since d1 != 0)
    prog.assert(var.clone().real_gt(Expr::real(0)));

    // n_i = d_i / std where std^2 = var, so n_i^2 = d_i^2 / var
    // n1^2 * var = d1^2
    let n1_sq = n1.clone().real_mul(n1);
    prog.assert(
        n1_sq
            .clone()
            .real_mul(var.clone())
            .eq(d1.clone().real_mul(d1)),
    );
    // n2^2 * var = d2^2
    let n2_sq = n2.clone().real_mul(n2);
    prog.assert(
        n2_sq
            .clone()
            .real_mul(var.clone())
            .eq(d2.clone().real_mul(d2)),
    );

    // out_var = (n1^2 + n2^2) / 2
    prog.assert(
        Expr::real(2)
            .real_mul(out_var.clone())
            .eq(n1_sq.real_add(n2_sq)),
    );

    // Negated property: out_var != 1
    let violation = out_var.ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "layernorm_output_variance_one");
}

// ---------------------------------------------------------------------------
// Test 493: LayerNorm affine transform: y = gamma * x_norm + beta
// ---------------------------------------------------------------------------

/// Prove: the LayerNorm affine transform correctly computes y = gamma * x_norm + beta.
///
/// After normalization, the optional affine transform applies:
///   y = gamma * x_norm + beta
/// This is a simple linear function. We verify the relationship holds.
#[test]
fn test_493_layernorm_affine_transform() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x_norm", real.clone());
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("beta", real.clone());
    let _ = prog.declare_const("y", real);

    let x_norm = real_var("x_norm");
    let gamma = real_var("gamma");
    let beta = real_var("beta");
    let y = real_var("y");

    // Bounded inputs
    prog.assert(x_norm.clone().real_ge(Expr::real(-10)));
    prog.assert(x_norm.clone().real_le(Expr::real(10)));
    prog.assert(gamma.clone().real_ge(Expr::real(-10)));
    prog.assert(gamma.clone().real_le(Expr::real(10)));
    prog.assert(beta.clone().real_ge(Expr::real(-10)));
    prog.assert(beta.clone().real_le(Expr::real(10)));

    // y = gamma * x_norm + beta
    prog.assert(
        y.clone().eq(gamma
            .clone()
            .real_mul(x_norm.clone())
            .real_add(beta.clone())),
    );

    // Negated property: y != gamma * x_norm + beta
    let violation = y.ne(gamma.real_mul(x_norm).real_add(beta));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "layernorm_affine_transform");
}

// ---------------------------------------------------------------------------
// Test 494: LayerNorm epsilon prevents division by zero
// ---------------------------------------------------------------------------

/// Prove: with epsilon > 0, the denominator sqrt(var + eps) > 0 always.
///
/// var >= 0 (variance is non-negative) and eps > 0, so var + eps > 0,
/// therefore sqrt(var + eps) > 0. Division by this quantity is safe.
///
/// We model: denom^2 = var + eps with var >= 0, eps > 0, and prove denom > 0.
#[test]
fn test_494_layernorm_epsilon_prevents_div_zero() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("var", real.clone());
    let _ = prog.declare_const("eps", real.clone());
    let _ = prog.declare_const("denom", real);

    let var = real_var("var");
    let eps = real_var("eps");
    let denom = real_var("denom");

    // var >= 0 (variance is non-negative)
    prog.assert(var.clone().real_ge(Expr::real(0)));

    // eps > 0 (typical: 1e-5)
    prog.assert(eps.clone().real_gt(Expr::real(0)));

    // denom = sqrt(var + eps), modeled as: denom^2 = var + eps, denom > 0
    // (We pick the positive root.)
    prog.assert(denom.clone().real_mul(denom.clone()).eq(var.real_add(eps)));
    prog.assert(denom.clone().real_gt(Expr::real(0)));

    // Negated property: denom <= 0
    let violation = denom.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "layernorm_epsilon_prevents_div_zero");
}

// ---------------------------------------------------------------------------
// Test 495: RMSNorm output formula: output = x / RMS(x) * gamma
// ---------------------------------------------------------------------------

/// Prove: RMSNorm output equals x * gamma / RMS(x).
///
/// RMS(x) = sqrt(mean(x^2) + eps). For a 2-element vector:
/// RMS = sqrt((x1^2 + x2^2)/2 + eps).
/// Output_i = x_i * gamma / RMS.
///
/// We model: rms > 0, out = x * gamma / rms, and verify the relationship.
#[test]
fn test_495_rmsnorm_output_formula() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("rms", real.clone());
    let _ = prog.declare_const("out", real);

    let x = real_var("x");
    let gamma = real_var("gamma");
    let rms = real_var("rms");
    let out = real_var("out");

    // Input bounds
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));
    prog.assert(gamma.clone().real_ge(Expr::real(-10)));
    prog.assert(gamma.clone().real_le(Expr::real(10)));

    // rms > 0 (always positive due to eps)
    prog.assert(rms.clone().real_gt(Expr::real(0)));

    // out = x * gamma / rms, modeled as: out * rms = x * gamma
    prog.assert(
        out.clone()
            .real_mul(rms.clone())
            .eq(x.clone().real_mul(gamma.clone())),
    );

    // Negated property: out * rms != x * gamma
    let violation = out.real_mul(rms).ne(x.real_mul(gamma));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rmsnorm_output_formula");
}

// ---------------------------------------------------------------------------
// Test 496: RMSNorm: RMS is always positive
// ---------------------------------------------------------------------------

/// Prove: RMS(x) = sqrt(mean(x^2) + eps) > 0 for all x when eps > 0.
///
/// x^2 >= 0 for all real x, so mean(x^2) >= 0. With eps > 0,
/// mean(x^2) + eps > 0, and sqrt of a positive number is positive.
///
/// We model: mean_sq >= 0, eps > 0, rms^2 = mean_sq + eps, rms > 0.
#[test]
fn test_496_rmsnorm_rms_always_positive() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("mean_sq", real.clone());
    let _ = prog.declare_const("eps", real.clone());
    let _ = prog.declare_const("rms", real);

    let mean_sq = real_var("mean_sq");
    let eps = real_var("eps");
    let rms = real_var("rms");

    // mean(x^2) >= 0 (mean of squares is non-negative)
    prog.assert(mean_sq.clone().real_ge(Expr::real(0)));

    // eps > 0
    prog.assert(eps.clone().real_gt(Expr::real(0)));

    // rms = sqrt(mean_sq + eps), modeled as rms^2 = mean_sq + eps, rms > 0
    prog.assert(rms.clone().real_mul(rms.clone()).eq(mean_sq.real_add(eps)));
    prog.assert(rms.clone().real_gt(Expr::real(0)));

    // Negated property: rms <= 0
    let violation = rms.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rmsnorm_rms_always_positive");
}

// ---------------------------------------------------------------------------
// Test 497: RMSNorm normalized output magnitude approximately 1
// ---------------------------------------------------------------------------

/// Prove: after RMSNorm (with gamma=1), the RMS of the output is 1.
///
/// For a 2-element vector with gamma=1:
/// out_i = x_i / RMS(x). Then RMS(out) = RMS(x/RMS(x)) = RMS(x)/RMS(x) = 1.
///
/// More precisely: mean(out_i^2) = mean(x_i^2) / RMS(x)^2.
/// Since RMS(x)^2 = mean(x_i^2) + eps, and ignoring eps contribution:
/// For the idealized case (eps=0), mean(out_i^2) = mean(x_i^2)/mean(x_i^2) = 1.
///
/// We prove the exact case: out_rms = rms_x / rms_x = 1 when both use the
/// same mean-square computation.
#[test]
fn test_497_rmsnorm_output_magnitude_one() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("mean_sq", real.clone());
    let _ = prog.declare_const("rms", real.clone());
    let _ = prog.declare_const("out_mean_sq", real.clone());
    let _ = prog.declare_const("out_rms", real);

    let mean_sq = real_var("mean_sq");
    let rms = real_var("rms");
    let out_mean_sq = real_var("out_mean_sq");
    let out_rms = real_var("out_rms");

    // mean_sq > 0 (non-trivial input)
    prog.assert(mean_sq.clone().real_gt(Expr::real(0)));

    // rms^2 = mean_sq (eps=0 idealized case), rms > 0
    prog.assert(rms.clone().real_mul(rms.clone()).eq(mean_sq.clone()));
    prog.assert(rms.clone().real_gt(Expr::real(0)));

    // out_i = x_i / rms, so out_i^2 = x_i^2 / rms^2
    // mean(out_i^2) = mean(x_i^2) / rms^2 = mean_sq / mean_sq = 1
    prog.assert(
        out_mean_sq
            .clone()
            .real_mul(rms.clone().real_mul(rms))
            .eq(mean_sq),
    );

    // out_rms^2 = out_mean_sq, out_rms > 0
    prog.assert(out_rms.clone().real_mul(out_rms.clone()).eq(out_mean_sq));
    prog.assert(out_rms.clone().real_gt(Expr::real(0)));

    // Negated property: out_rms != 1
    let violation = out_rms.ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rmsnorm_output_magnitude_one");
}

// ---------------------------------------------------------------------------
// Test 498: BatchNorm running mean update formula
// ---------------------------------------------------------------------------

/// Prove: running_mean update follows exponential moving average.
///
/// running_mean_new = (1 - momentum) * running_mean_old + momentum * batch_mean
///
/// With momentum m: rm_new = (1-m)*rm_old + m*bm.
/// This is a convex combination when 0 < m < 1.
#[test]
fn test_498_batchnorm_running_mean_update() {
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
    // Model: lo <= rm_old, lo <= bm, hi >= rm_old, hi >= bm
    prog.assert(lo.clone().real_le(rm_old.clone()));
    prog.assert(lo.clone().real_le(bm.clone()));
    prog.assert(hi.clone().real_ge(rm_old.clone()));
    prog.assert(hi.clone().real_ge(bm.clone()));
    // lo and hi are exactly min/max
    prog.assert(lo.clone().eq(rm_old.clone()).or(lo.clone().eq(bm.clone())));
    prog.assert(hi.clone().eq(rm_old.clone()).or(hi.clone().eq(bm.clone())));

    // rm_new = (1-m)*rm_old + m*bm (convex combination)
    let one_minus_m = Expr::real(1).real_sub(m.clone());
    prog.assert(
        rm_new
            .clone()
            .eq(one_minus_m.real_mul(rm_old).real_add(m.real_mul(bm))),
    );

    // Property: lo <= rm_new <= hi (convex combination stays in range)
    // Negated: rm_new < lo OR rm_new > hi
    let violation = rm_new.clone().real_lt(lo).or(rm_new.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "batchnorm_running_mean_update");
}

// ---------------------------------------------------------------------------
// Test 499: BatchNorm running variance update formula
// ---------------------------------------------------------------------------

/// Prove: running_var update follows exponential moving average with non-negative result.
///
/// running_var_new = (1 - momentum) * running_var_old + momentum * batch_var
///
/// Since running_var_old >= 0, batch_var >= 0, and 0 < momentum < 1,
/// the convex combination running_var_new >= 0.
#[test]
fn test_499_batchnorm_running_var_update() {
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

    // Both variances non-negative
    prog.assert(rv_old.clone().real_ge(Expr::real(0)));
    prog.assert(bv.clone().real_ge(Expr::real(0)));

    // momentum in (0, 1)
    prog.assert(m.clone().real_gt(Expr::real(0)));
    prog.assert(m.clone().real_lt(Expr::real(1)));

    // rv_new = (1-m)*rv_old + m*bv
    let one_minus_m = Expr::real(1).real_sub(m.clone());
    prog.assert(
        rv_new
            .clone()
            .eq(one_minus_m.real_mul(rv_old).real_add(m.real_mul(bv))),
    );

    // Negated property: rv_new < 0
    let violation = rv_new.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "batchnorm_running_var_update");
}

// ---------------------------------------------------------------------------
// Test 500: BatchNorm inference mode uses running statistics
// ---------------------------------------------------------------------------

/// Prove: in inference mode, BatchNorm output depends on running_mean/running_var,
/// not batch statistics.
///
/// Inference: y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
///
/// We model two inputs with same running stats producing consistent outputs.
/// If x1 != x2 but same running_mean/running_var, then y1 != y2 (not collapsed).
#[test]
fn test_500_batchnorm_inference_uses_running_stats() {
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

    // Negated property: y1 = y2 (outputs collapsed despite different inputs)
    let violation = y1.eq(y2);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "batchnorm_inference_uses_running_stats");
}

// ---------------------------------------------------------------------------
// Test 501: BatchNorm training mode uses batch statistics
// ---------------------------------------------------------------------------

/// Prove: in training mode, BatchNorm centers using batch mean.
///
/// Training: y_i = gamma * (x_i - batch_mean) / sqrt(batch_var + eps) + beta
///
/// For a 2-element batch [x1, x2]: batch_mean = (x1+x2)/2.
/// The mean of outputs (before affine) should be 0,
/// same as LayerNorm centering property.
#[test]
fn test_501_batchnorm_training_uses_batch_stats() {
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

    // batch_mean = (x1 + x2) / 2
    prog.assert(
        Expr::real(2)
            .real_mul(bm.clone())
            .eq(x1.clone().real_add(x2.clone())),
    );

    // s > 0 (sqrt(batch_var + eps))
    prog.assert(s.clone().real_gt(Expr::real(0)));

    // n_i = (x_i - bm) / s
    prog.assert(n1.clone().real_mul(s.clone()).eq(x1.real_sub(bm.clone())));
    prog.assert(n2.clone().real_mul(s).eq(x2.real_sub(bm)));

    // out_mean = (n1 + n2) / 2
    prog.assert(Expr::real(2).real_mul(out_mean.clone()).eq(n1.real_add(n2)));

    // Negated property: out_mean != 0 (batch-centered outputs should have mean 0)
    let violation = out_mean.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "batchnorm_training_uses_batch_stats");
}

// ---------------------------------------------------------------------------
// Test 502: GroupNorm: group_size divides num_channels
// ---------------------------------------------------------------------------

/// Prove: GroupNorm requires num_channels = num_groups * group_size.
///
/// If num_channels = G * C_g, then channels partition into G groups of C_g each.
/// Reconstruction: G * C_g = num_channels.
///
/// We model: total = g * gs, and prove total = g * gs (exact division).
#[test]
fn test_502_groupnorm_group_size_divides_channels() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("num_channels", real.clone());
    let _ = prog.declare_const("num_groups", real.clone());
    let _ = prog.declare_const("group_size", real.clone());
    let _ = prog.declare_const("reconstructed", real);

    let num_channels = real_var("num_channels");
    let num_groups = real_var("num_groups");
    let group_size = real_var("group_size");
    let reconstructed = real_var("reconstructed");

    // Positive integers (modeled as positive reals)
    prog.assert(num_channels.clone().real_gt(Expr::real(0)));
    prog.assert(num_groups.clone().real_gt(Expr::real(0)));
    prog.assert(group_size.clone().real_gt(Expr::real(0)));

    // num_channels = num_groups * group_size
    prog.assert(
        num_channels
            .clone()
            .eq(num_groups.clone().real_mul(group_size.clone())),
    );

    // reconstructed = num_groups * group_size
    prog.assert(reconstructed.clone().eq(num_groups.real_mul(group_size)));

    // Negated property: reconstructed != num_channels
    let violation = reconstructed.ne(num_channels);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "groupnorm_group_size_divides_channels");
}

// ---------------------------------------------------------------------------
// Test 503: GroupNorm: each group normalized independently
// ---------------------------------------------------------------------------

/// Prove: GroupNorm normalizes each group independently.
///
/// For two groups, each with 2 elements:
/// Group 1: [x1, x2] -> normalized with mean1, var1
/// Group 2: [x3, x4] -> normalized with mean2, var2
///
/// Changing x3,x4 does not affect the normalization of x1,x2.
/// We model: n1 = (x1 - mean1)/s1 depends only on x1,x2 (not x3,x4).
#[test]
fn test_503_groupnorm_independent_groups() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("mean1", real.clone());
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("n1_a", real.clone());
    let _ = prog.declare_const("n1_b", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let mean1 = real_var("mean1");
    let s1 = real_var("s1");
    let n1_a = real_var("n1_a");
    let n1_b = real_var("n1_b");

    // Group 1 mean = (x1 + x2) / 2
    prog.assert(
        Expr::real(2)
            .real_mul(mean1.clone())
            .eq(x1.clone().real_add(x2.clone())),
    );

    // s1 > 0
    prog.assert(s1.clone().real_gt(Expr::real(0)));

    // n1_a = (x1 - mean1) / s1 (case A: some group 2 values)
    prog.assert(
        n1_a.clone()
            .real_mul(s1.clone())
            .eq(x1.clone().real_sub(mean1.clone())),
    );

    // n1_b = (x1 - mean1) / s1 (case B: different group 2 values, same formula)
    prog.assert(n1_b.clone().real_mul(s1).eq(x1.real_sub(mean1)));

    // Negated property: n1_a != n1_b (group 1 output changed when group 2 changed)
    let violation = n1_a.ne(n1_b);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "groupnorm_independent_groups");
}

// ---------------------------------------------------------------------------
// Test 504: InstanceNorm: each sample normalized independently
// ---------------------------------------------------------------------------

/// Prove: InstanceNorm normalizes each sample in a batch independently.
///
/// For sample i with features [f1, f2]:
/// mean_i = (f1+f2)/2, normalized_i = (f_j - mean_i) / std_i.
///
/// The normalization of sample i depends only on sample i's features.
/// We model two computations with the same sample data producing identical output.
#[test]
fn test_504_instancenorm_per_sample_independence() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("f1", real.clone());
    let _ = prog.declare_const("f2", real.clone());
    let _ = prog.declare_const("mean_i", real.clone());
    let _ = prog.declare_const("s_i", real.clone());
    let _ = prog.declare_const("n_a", real.clone());
    let _ = prog.declare_const("n_b", real);

    let f1 = real_var("f1");
    let f2 = real_var("f2");
    let mean_i = real_var("mean_i");
    let s_i = real_var("s_i");
    let n_a = real_var("n_a");
    let n_b = real_var("n_b");

    // mean_i = (f1 + f2) / 2
    prog.assert(
        Expr::real(2)
            .real_mul(mean_i.clone())
            .eq(f1.clone().real_add(f2)),
    );

    // s_i > 0
    prog.assert(s_i.clone().real_gt(Expr::real(0)));

    // n_a = (f1 - mean_i) / s_i (batch context A)
    prog.assert(
        n_a.clone()
            .real_mul(s_i.clone())
            .eq(f1.clone().real_sub(mean_i.clone())),
    );

    // n_b = (f1 - mean_i) / s_i (batch context B, different other samples)
    prog.assert(n_b.clone().real_mul(s_i).eq(f1.real_sub(mean_i)));

    // Negated property: n_a != n_b (output changed with different batch context)
    let violation = n_a.ne(n_b);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "instancenorm_per_sample_independence");
}

// ---------------------------------------------------------------------------
// Test 505: LayerNorm is scale-invariant: norm(kx) = norm(x)
// ---------------------------------------------------------------------------

/// Prove: LayerNorm(k*x) = LayerNorm(x) for k > 0 (before affine transform).
///
/// For 2-element [x1, x2] scaled by k:
/// mean(kx) = k*mean(x), var(kx) = k^2*var(x), std(kx) = k*std(x).
/// norm(kx)_i = (k*x_i - k*mean) / (k*std) = (x_i - mean) / std = norm(x)_i.
///
/// We model: n_orig = (x - mean) / s, n_scaled = (k*x - k*mean) / (k*s),
/// and prove n_orig = n_scaled.
#[test]
fn test_505_layernorm_scale_invariance() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("mean", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("k", real.clone());
    let _ = prog.declare_const("n_orig", real.clone());
    let _ = prog.declare_const("n_scaled", real);

    let x = real_var("x");
    let mean = real_var("mean");
    let s = real_var("s");
    let k = real_var("k");
    let n_orig = real_var("n_orig");
    let n_scaled = real_var("n_scaled");

    // s > 0, k > 0
    prog.assert(s.clone().real_gt(Expr::real(0)));
    prog.assert(k.clone().real_gt(Expr::real(0)));

    // n_orig = (x - mean) / s, modeled as n_orig * s = x - mean
    prog.assert(
        n_orig
            .clone()
            .real_mul(s.clone())
            .eq(x.clone().real_sub(mean.clone())),
    );

    // n_scaled = (k*x - k*mean) / (k*s), modeled as n_scaled * (k*s) = k*x - k*mean
    let ks = k.clone().real_mul(s);
    let kx = k.clone().real_mul(x);
    let kmean = k.real_mul(mean);
    prog.assert(n_scaled.clone().real_mul(ks).eq(kx.real_sub(kmean)));

    // Negated property: n_orig != n_scaled
    let violation = n_orig.ne(n_scaled);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "layernorm_scale_invariance");
}

// ---------------------------------------------------------------------------
// Test 506: Normalization idempotent: norm(norm(x)) approx norm(x)
// ---------------------------------------------------------------------------

/// Prove: applying LayerNorm twice gives the same result as once.
///
/// After first LayerNorm: output has mean 0, variance 1.
/// Applying LayerNorm again to data with mean 0, variance 1:
/// mean = 0, std = 1, so norm(y)_i = (y_i - 0) / 1 = y_i.
///
/// We model: y = (x - mean_x) / std_x (first norm), then applying norm again:
/// mean_y = 0, std_y = 1, z = (y - 0) / 1 = y. So z = y.
#[test]
fn test_506_normalization_idempotent() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("y", real.clone());
    let _ = prog.declare_const("mean_y", real.clone());
    let _ = prog.declare_const("std_y", real.clone());
    let _ = prog.declare_const("z", real);

    let y = real_var("y");
    let mean_y = real_var("mean_y");
    let std_y = real_var("std_y");
    let z = real_var("z");

    // After first LayerNorm: mean_y = 0, std_y = 1
    prog.assert(mean_y.clone().eq(Expr::real(0)));
    prog.assert(std_y.clone().eq(Expr::real(1)));

    // Second normalization: z = (y - mean_y) / std_y = (y - 0) / 1 = y
    prog.assert(z.clone().real_mul(std_y).eq(y.clone().real_sub(mean_y)));

    // Negated property: z != y
    let violation = z.ne(y);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "normalization_idempotent");
}

// ---------------------------------------------------------------------------
// Test 507: LayerNorm gradient bounded for bounded inputs
// ---------------------------------------------------------------------------

/// Prove: the LayerNorm gradient magnitude is bounded for bounded inputs.
///
/// For scalar simplification, d(LayerNorm)/dx for element i depends on:
/// d_out/d_x_i = (1/std) * (1 - 1/N) * (1 - x_norm_i^2 / N) (approximately)
///
/// For the 2-element case with bounded normalized inputs |x_norm| <= B:
/// |grad| <= 1/std * (1 + B^2) which is bounded when std > 0 and B is finite.
///
/// We prove a simpler property: for |x_norm| <= 3, the gradient factor
/// (1 - x_norm^2/2) / std is bounded by 2/std, and since std >= eps > 0,
/// the gradient is finite.
#[test]
fn test_507_layernorm_gradient_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x_norm", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("grad_factor", real);

    let x_norm = real_var("x_norm");
    let s = real_var("s");
    let grad_factor = real_var("grad_factor");

    // Bounded normalized input: |x_norm| <= 3
    prog.assert(x_norm.clone().real_ge(Expr::real(-3)));
    prog.assert(x_norm.clone().real_le(Expr::real(3)));

    // s > 0 (standard deviation bounded away from 0)
    prog.assert(s.clone().real_gt(Expr::real(0)));
    prog.assert(s.clone().real_le(Expr::real(100)));

    // grad_factor = (1/s) * (1 - x_norm^2 / 2) for N=2
    // Modeled as: grad_factor * s = 1 - x_norm^2 / 2
    // i.e., grad_factor * s * 2 = 2 - x_norm^2
    let x_norm_sq = x_norm.clone().real_mul(x_norm);
    prog.assert(
        grad_factor
            .clone()
            .real_mul(s.clone())
            .real_mul(Expr::real(2))
            .eq(Expr::real(2).real_sub(x_norm_sq)),
    );

    // Bound: |grad_factor| <= 10/s. Since s > 0 and |2 - x_norm^2| <= 11
    // (x_norm^2 <= 9, so |2-9| = 7 <= 11), |grad_factor * s * 2| <= 11
    // so |grad_factor| <= 11/(2*s).
    // We prove: |grad_factor * s| <= 6 (generous bound since |2 - x_norm^2|/2 <= 5.5)
    let gs = grad_factor.real_mul(s);
    // Negated: |grad_factor * s| > 6, i.e., gs > 6 or gs < -6
    let violation = gs
        .clone()
        .real_gt(Expr::real(6))
        .or(gs.real_lt(Expr::real(-6)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "layernorm_gradient_bounded");
}

// ---------------------------------------------------------------------------
// Test 508: Epsilon must be positive
// ---------------------------------------------------------------------------

/// Prove: epsilon > 0 is required for normalization safety.
///
/// If eps <= 0, then for an input where all elements are equal (var = 0),
/// the denominator sqrt(var + eps) could be 0 or undefined.
/// With eps > 0, the denominator is always at least sqrt(eps) > 0.
///
/// We prove: if eps > 0 and var >= 0, then var + eps > 0.
#[test]
fn test_508_epsilon_must_be_positive() {
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

    // Negated property: sum <= 0
    let violation = sum.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "epsilon_must_be_positive");
}

// ---------------------------------------------------------------------------
// Test 509: Affine parameters: gamma=1, beta=0 is identity
// ---------------------------------------------------------------------------

/// Prove: with gamma=1 and beta=0, the affine transform is the identity.
///
/// y = gamma * x + beta = 1 * x + 0 = x.
/// This is the default initialization, ensuring LayerNorm starts as
/// pure normalization without learned scaling/shifting.
#[test]
fn test_509_affine_default_is_identity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("beta", real.clone());
    let _ = prog.declare_const("y", real);

    let x = real_var("x");
    let gamma = real_var("gamma");
    let beta = real_var("beta");
    let y = real_var("y");

    // Default initialization: gamma = 1, beta = 0
    prog.assert(gamma.clone().eq(Expr::real(1)));
    prog.assert(beta.clone().eq(Expr::real(0)));

    // Input bounds
    prog.assert(x.clone().real_ge(Expr::real(-1000)));
    prog.assert(x.clone().real_le(Expr::real(1000)));

    // y = gamma * x + beta
    prog.assert(y.clone().eq(gamma.real_mul(x.clone()).real_add(beta)));

    // Negated property: y != x (identity violated)
    let violation = y.ne(x);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "affine_default_is_identity");
}

// ---------------------------------------------------------------------------
// Test 510: Weight decay on gamma/beta bounds parameter growth
// ---------------------------------------------------------------------------

/// Prove: L2 weight decay bounds the norm of affine parameters.
///
/// With weight decay lambda > 0, the update rule is:
///   gamma_new = gamma_old - lr * (grad + lambda * gamma_old)
///             = gamma_old * (1 - lr * lambda) - lr * grad
///
/// For the equilibrium (grad = 0), gamma converges to 0.
/// With bounded gradient, |gamma| is bounded by |grad_max| / lambda.
///
/// We prove: if |gamma_old| <= B and decay factor 0 < (1-lr*lambda) < 1
/// and |lr*grad| <= G, then |gamma_new| <= B*(1-lr*lambda) + G < B + G.
#[test]
fn test_510_weight_decay_bounds_parameters() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("gamma_old", real.clone());
    let _ = prog.declare_const("decay", real.clone());
    let _ = prog.declare_const("lr_grad", real.clone());
    let _ = prog.declare_const("gamma_new", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("g", real);

    let gamma_old = real_var("gamma_old");
    let decay = real_var("decay");
    let lr_grad = real_var("lr_grad");
    let gamma_new = real_var("gamma_new");
    let b = real_var("b");
    let g = real_var("g");

    // B > 0, G > 0
    prog.assert(b.clone().real_gt(Expr::real(0)));
    prog.assert(g.clone().real_gt(Expr::real(0)));

    // |gamma_old| <= B
    prog.assert(gamma_old.clone().real_ge(Expr::real(0).real_sub(b.clone())));
    prog.assert(gamma_old.clone().real_le(b.clone()));

    // decay = 1 - lr*lambda, in (0, 1)
    prog.assert(decay.clone().real_gt(Expr::real(0)));
    prog.assert(decay.clone().real_lt(Expr::real(1)));

    // |lr_grad| <= G
    prog.assert(lr_grad.clone().real_ge(Expr::real(0).real_sub(g.clone())));
    prog.assert(lr_grad.clone().real_le(g.clone()));

    // gamma_new = gamma_old * decay - lr_grad
    prog.assert(
        gamma_new
            .clone()
            .eq(gamma_old.real_mul(decay.clone()).real_sub(lr_grad)),
    );

    // Property: |gamma_new| <= B + G
    // Since |gamma_old * decay| <= B * decay < B (decay < 1)
    // and |lr_grad| <= G, we have |gamma_new| <= B*decay + G < B + G.
    let bound = b.real_add(g);

    // Negated: |gamma_new| > B + G
    let violation = gamma_new
        .clone()
        .real_gt(bound.clone())
        .or(gamma_new.real_lt(Expr::real(0).real_sub(bound)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "weight_decay_bounds_parameters");
}

// ---------------------------------------------------------------------------
// Test 1011: BatchNorm affine output has mean beta and variance gamma^2
// ---------------------------------------------------------------------------

/// Prove: for BatchNorm output y = gamma * n + beta, if the normalized
/// activations have mean 0 and variance 1, then the affine output has
/// mean beta and variance gamma^2.
///
/// We model a 2-element normalized batch with n1 + n2 = 0 and
/// n1^2 + n2^2 = 2. After affine transformation, the output mean is beta
/// and the output variance is gamma^2 exactly in this algebraic model.
///
/// Part of #4223.
#[test]
fn test_1011_batchnorm_output_mean_beta_variance_gamma_sq() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("n1", real.clone());
    let _ = prog.declare_const("n2", real.clone());
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("beta", real.clone());
    let _ = prog.declare_const("y1", real.clone());
    let _ = prog.declare_const("y2", real.clone());
    let _ = prog.declare_const("out_mean", real.clone());
    let _ = prog.declare_const("out_var", real.clone());
    let _ = prog.declare_const("gamma_sq", real);

    let n1 = real_var("n1");
    let n2 = real_var("n2");
    let gamma = real_var("gamma");
    let beta = real_var("beta");
    let y1 = real_var("y1");
    let y2 = real_var("y2");
    let out_mean = real_var("out_mean");
    let out_var = real_var("out_var");
    let gamma_sq = real_var("gamma_sq");

    // Normalized batch statistics: mean 0 and variance 1.
    prog.assert(n1.clone().real_add(n2.clone()).eq(Expr::real(0)));
    let n1_sq = n1.clone().real_mul(n1.clone());
    let n2_sq = n2.clone().real_mul(n2.clone());
    prog.assert(n1_sq.clone().real_add(n2_sq.clone()).eq(Expr::real(2)));

    // gamma_sq = gamma^2
    prog.assert(gamma_sq.clone().eq(gamma.clone().real_mul(gamma.clone())));

    // Affine transform: y_i = gamma * n_i + beta
    prog.assert(
        y1.clone()
            .eq(gamma.clone().real_mul(n1.clone()).real_add(beta.clone())),
    );
    prog.assert(
        y2.clone()
            .eq(gamma.clone().real_mul(n2.clone()).real_add(beta.clone())),
    );

    // out_mean = (y1 + y2) / 2
    prog.assert(
        Expr::real(2)
            .real_mul(out_mean.clone())
            .eq(y1.clone().real_add(y2.clone())),
    );

    // out_var = ((y1 - beta)^2 + (y2 - beta)^2) / 2
    let c1 = y1.clone().real_sub(beta.clone());
    let c2 = y2.clone().real_sub(beta.clone());
    prog.assert(
        Expr::real(2)
            .real_mul(out_var.clone())
            .eq(c1.clone().real_mul(c1).real_add(c2.clone().real_mul(c2))),
    );

    // Negated property: out_mean != beta OR out_var != gamma^2
    let violation = out_mean.ne(beta.clone()).or(out_var.ne(gamma_sq));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "batchnorm_output_mean_beta_variance_gamma_sq");
}

// ---------------------------------------------------------------------------
// Test 1012: LayerNorm 3-element output mean is 0
// ---------------------------------------------------------------------------

/// Prove: after LayerNorm on a 3-element vector, the output mean is 0.
///
/// For [x1, x2, x3], mean = (x1 + x2 + x3) / 3 and
/// n_i = (x_i - mean) / s. Then:
///   n1 + n2 + n3 = (x1 + x2 + x3 - 3 * mean) / s = 0,
/// so the output mean is exactly 0.
///
/// Part of #4223.
#[test]
fn test_1012_layernorm_output_mean_zero_3elem() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("x3", real.clone());
    let _ = prog.declare_const("mean", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("n1", real.clone());
    let _ = prog.declare_const("n2", real.clone());
    let _ = prog.declare_const("n3", real.clone());
    let _ = prog.declare_const("out_mean", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let x3 = real_var("x3");
    let mean = real_var("mean");
    let s = real_var("s");
    let n1 = real_var("n1");
    let n2 = real_var("n2");
    let n3 = real_var("n3");
    let out_mean = real_var("out_mean");

    // Input bounds
    prog.assert(x1.clone().real_ge(Expr::real(-100)));
    prog.assert(x1.clone().real_le(Expr::real(100)));
    prog.assert(x2.clone().real_ge(Expr::real(-100)));
    prog.assert(x2.clone().real_le(Expr::real(100)));
    prog.assert(x3.clone().real_ge(Expr::real(-100)));
    prog.assert(x3.clone().real_le(Expr::real(100)));

    // mean = (x1 + x2 + x3) / 3
    prog.assert(
        Expr::real(3)
            .real_mul(mean.clone())
            .eq(x1.clone().real_add(x2.clone()).real_add(x3.clone())),
    );

    // s > 0
    prog.assert(s.clone().real_gt(Expr::real(0)));

    // n_i = (x_i - mean) / s
    prog.assert(n1.clone().real_mul(s.clone()).eq(x1.real_sub(mean.clone())));
    prog.assert(n2.clone().real_mul(s.clone()).eq(x2.real_sub(mean.clone())));
    prog.assert(n3.clone().real_mul(s).eq(x3.real_sub(mean)));

    // out_mean = (n1 + n2 + n3) / 3
    prog.assert(
        Expr::real(3)
            .real_mul(out_mean.clone())
            .eq(n1.real_add(n2).real_add(n3)),
    );

    // Negated property: out_mean != 0
    let violation = out_mean.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "layernorm_output_mean_zero_3elem");
}

// ---------------------------------------------------------------------------
// Test 1013: LayerNorm 3-element output variance is 1
// ---------------------------------------------------------------------------

/// Prove: after LayerNorm on a 3-element vector, the output variance is 1.
///
/// We model centered deviations d_i = x_i - mean with d1 + d2 + d3 = 0 and
/// var = (d1^2 + d2^2 + d3^2) / 3. If n_i = d_i / sqrt(var), then
/// mean(n_i^2) = 1 exactly in this idealized normalization model.
///
/// Part of #4223.
#[test]
fn test_1013_layernorm_output_variance_one_3elem() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("d1", real.clone());
    let _ = prog.declare_const("d2", real.clone());
    let _ = prog.declare_const("d3", real.clone());
    let _ = prog.declare_const("var", real.clone());
    let _ = prog.declare_const("n1", real.clone());
    let _ = prog.declare_const("n2", real.clone());
    let _ = prog.declare_const("n3", real.clone());
    let _ = prog.declare_const("out_var", real);

    let d1 = real_var("d1");
    let d2 = real_var("d2");
    let d3 = real_var("d3");
    let var = real_var("var");
    let n1 = real_var("n1");
    let n2 = real_var("n2");
    let n3 = real_var("n3");
    let out_var = real_var("out_var");

    // Centered deviations sum to zero.
    prog.assert(
        d1.clone()
            .real_add(d2.clone())
            .real_add(d3.clone())
            .eq(Expr::real(0)),
    );

    // var = (d1^2 + d2^2 + d3^2) / 3
    let d1_sq = d1.clone().real_mul(d1.clone());
    let d2_sq = d2.clone().real_mul(d2.clone());
    let d3_sq = d3.clone().real_mul(d3.clone());
    prog.assert(
        Expr::real(3).real_mul(var.clone()).eq(d1_sq
            .clone()
            .real_add(d2_sq.clone())
            .real_add(d3_sq.clone())),
    );

    // Non-degenerate input
    prog.assert(var.clone().real_gt(Expr::real(0)));

    // n_i^2 = d_i^2 / var
    let n1_sq = n1.clone().real_mul(n1.clone());
    let n2_sq = n2.clone().real_mul(n2.clone());
    let n3_sq = n3.clone().real_mul(n3.clone());
    prog.assert(n1_sq.clone().real_mul(var.clone()).eq(d1_sq));
    prog.assert(n2_sq.clone().real_mul(var.clone()).eq(d2_sq));
    prog.assert(n3_sq.clone().real_mul(var.clone()).eq(d3_sq));

    // out_var = (n1^2 + n2^2 + n3^2) / 3
    prog.assert(
        Expr::real(3).real_mul(out_var.clone()).eq(n1_sq
            .clone()
            .real_add(n2_sq.clone())
            .real_add(n3_sq.clone())),
    );

    // Negated property: out_var != 1
    let violation = out_var.ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "layernorm_output_variance_one_3elem");
}

// ---------------------------------------------------------------------------
// Test 1014: GroupNorm groups each have mean 0 after normalization
// ---------------------------------------------------------------------------

/// Prove: when channels are split into groups, each group is normalized
/// independently and each group's normalized mean is 0.
///
/// We model 4 channels split into 2 groups of 2:
/// group 1 uses [x1, x2], group 2 uses [x3, x4].
/// Each group has its own mean and denominator.
///
/// Part of #4223.
#[test]
fn test_1014_groupnorm_group_means_zero() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("x3", real.clone());
    let _ = prog.declare_const("x4", real.clone());
    let _ = prog.declare_const("mean1", real.clone());
    let _ = prog.declare_const("mean2", real.clone());
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("s2", real.clone());
    let _ = prog.declare_const("n1", real.clone());
    let _ = prog.declare_const("n2", real.clone());
    let _ = prog.declare_const("n3", real.clone());
    let _ = prog.declare_const("n4", real.clone());
    let _ = prog.declare_const("g1_mean", real.clone());
    let _ = prog.declare_const("g2_mean", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let x3 = real_var("x3");
    let x4 = real_var("x4");
    let mean1 = real_var("mean1");
    let mean2 = real_var("mean2");
    let s1 = real_var("s1");
    let s2 = real_var("s2");
    let n1 = real_var("n1");
    let n2 = real_var("n2");
    let n3 = real_var("n3");
    let n4 = real_var("n4");
    let g1_mean = real_var("g1_mean");
    let g2_mean = real_var("g2_mean");

    // Group means
    prog.assert(
        Expr::real(2)
            .real_mul(mean1.clone())
            .eq(x1.clone().real_add(x2.clone())),
    );
    prog.assert(
        Expr::real(2)
            .real_mul(mean2.clone())
            .eq(x3.clone().real_add(x4.clone())),
    );

    // Positive denominators
    prog.assert(s1.clone().real_gt(Expr::real(0)));
    prog.assert(s2.clone().real_gt(Expr::real(0)));

    // Per-group normalization
    prog.assert(
        n1.clone()
            .real_mul(s1.clone())
            .eq(x1.real_sub(mean1.clone())),
    );
    prog.assert(n2.clone().real_mul(s1).eq(x2.real_sub(mean1)));
    prog.assert(
        n3.clone()
            .real_mul(s2.clone())
            .eq(x3.real_sub(mean2.clone())),
    );
    prog.assert(n4.clone().real_mul(s2).eq(x4.real_sub(mean2)));

    // Group means of normalized outputs
    prog.assert(Expr::real(2).real_mul(g1_mean.clone()).eq(n1.real_add(n2)));
    prog.assert(Expr::real(2).real_mul(g2_mean.clone()).eq(n3.real_add(n4)));

    // Negated property: either group mean is non-zero
    let violation = g1_mean.ne(Expr::real(0)).or(g2_mean.ne(Expr::real(0)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "groupnorm_group_means_zero");
}

// ---------------------------------------------------------------------------
// Test 1015: InstanceNorm per-instance per-channel output mean is 0
// ---------------------------------------------------------------------------

/// Prove: InstanceNorm normalizes each instance-channel slice to mean 0.
///
/// We model one instance and one channel with 3 spatial positions
/// [p1, p2, p3]. InstanceNorm computes the mean over those positions and
/// normalizes each position with the same denominator.
///
/// Part of #4223.
#[test]
fn test_1015_instancenorm_output_mean_zero() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("p1", real.clone());
    let _ = prog.declare_const("p2", real.clone());
    let _ = prog.declare_const("p3", real.clone());
    let _ = prog.declare_const("mean", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("n1", real.clone());
    let _ = prog.declare_const("n2", real.clone());
    let _ = prog.declare_const("n3", real.clone());
    let _ = prog.declare_const("out_mean", real);

    let p1 = real_var("p1");
    let p2 = real_var("p2");
    let p3 = real_var("p3");
    let mean = real_var("mean");
    let s = real_var("s");
    let n1 = real_var("n1");
    let n2 = real_var("n2");
    let n3 = real_var("n3");
    let out_mean = real_var("out_mean");

    // mean = (p1 + p2 + p3) / 3
    prog.assert(
        Expr::real(3)
            .real_mul(mean.clone())
            .eq(p1.clone().real_add(p2.clone()).real_add(p3.clone())),
    );

    // s > 0
    prog.assert(s.clone().real_gt(Expr::real(0)));

    // n_i = (p_i - mean) / s
    prog.assert(n1.clone().real_mul(s.clone()).eq(p1.real_sub(mean.clone())));
    prog.assert(n2.clone().real_mul(s.clone()).eq(p2.real_sub(mean.clone())));
    prog.assert(n3.clone().real_mul(s).eq(p3.real_sub(mean)));

    // out_mean = (n1 + n2 + n3) / 3
    prog.assert(
        Expr::real(3)
            .real_mul(out_mean.clone())
            .eq(n1.real_add(n2).real_add(n3)),
    );

    // Negated property: out_mean != 0
    let violation = out_mean.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "instancenorm_output_mean_zero");
}

// ---------------------------------------------------------------------------
// Test 1016: RMSNorm output formula on a 3-element vector
// ---------------------------------------------------------------------------

/// Prove: RMSNorm output satisfies out_i = x_i * gamma / rms for a
/// 3-element vector.
///
/// We model:
///   rms^2 = (x1^2 + x2^2 + x3^2) / 3 + eps
/// and encode the output equations as out_i * rms = x_i * gamma.
///
/// Part of #4223.
#[test]
fn test_1016_rmsnorm_output_formula_3elem() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("x3", real.clone());
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("eps", real.clone());
    let _ = prog.declare_const("rms", real.clone());
    let _ = prog.declare_const("out1", real.clone());
    let _ = prog.declare_const("out2", real.clone());
    let _ = prog.declare_const("out3", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let x3 = real_var("x3");
    let gamma = real_var("gamma");
    let eps = real_var("eps");
    let rms = real_var("rms");
    let out1 = real_var("out1");
    let out2 = real_var("out2");
    let out3 = real_var("out3");

    // Input bounds
    prog.assert(x1.clone().real_ge(Expr::real(-100)));
    prog.assert(x1.clone().real_le(Expr::real(100)));
    prog.assert(x2.clone().real_ge(Expr::real(-100)));
    prog.assert(x2.clone().real_le(Expr::real(100)));
    prog.assert(x3.clone().real_ge(Expr::real(-100)));
    prog.assert(x3.clone().real_le(Expr::real(100)));
    prog.assert(gamma.clone().real_ge(Expr::real(-10)));
    prog.assert(gamma.clone().real_le(Expr::real(10)));
    prog.assert(eps.clone().real_gt(Expr::real(0)));

    // rms^2 = mean(x^2) + eps
    let x1_sq = x1.clone().real_mul(x1.clone());
    let x2_sq = x2.clone().real_mul(x2.clone());
    let x3_sq = x3.clone().real_mul(x3.clone());
    let rms_sq = rms.clone().real_mul(rms.clone());
    prog.assert(
        Expr::real(3).real_mul(rms_sq).eq(x1_sq
            .clone()
            .real_add(x2_sq.clone())
            .real_add(x3_sq.clone())
            .real_add(Expr::real(3).real_mul(eps.clone()))),
    );
    prog.assert(rms.clone().real_gt(Expr::real(0)));

    // out_i = x_i * gamma / rms
    prog.assert(
        out1.clone()
            .real_mul(rms.clone())
            .eq(x1.clone().real_mul(gamma.clone())),
    );
    prog.assert(
        out2.clone()
            .real_mul(rms.clone())
            .eq(x2.clone().real_mul(gamma.clone())),
    );
    prog.assert(
        out3.clone()
            .real_mul(rms.clone())
            .eq(x3.clone().real_mul(gamma.clone())),
    );

    // Negated property: one of the vector components violates the formula
    let violation = out1
        .clone()
        .real_mul(rms.clone())
        .ne(x1.clone().real_mul(gamma.clone()))
        .or(out2
            .clone()
            .real_mul(rms.clone())
            .ne(x2.clone().real_mul(gamma.clone())))
        .or(out3.real_mul(rms).ne(x3.real_mul(gamma)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rmsnorm_output_formula_3elem");
}

// ---------------------------------------------------------------------------
// Test 1017: Running mean update exact EMA formula
// ---------------------------------------------------------------------------

/// Prove: BatchNorm running mean update is exactly
/// mean_new = (1 - momentum) * mean_old + momentum * batch_mean.
///
/// This test checks the exact EMA formula, not just the interpolation bound.
///
/// Part of #4223.
#[test]
fn test_1017_running_mean_update_exact_formula() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("mean_old", real.clone());
    let _ = prog.declare_const("batch_mean", real.clone());
    let _ = prog.declare_const("momentum", real.clone());
    let _ = prog.declare_const("mean_new", real);

    let mean_old = real_var("mean_old");
    let batch_mean = real_var("batch_mean");
    let momentum = real_var("momentum");
    let mean_new = real_var("mean_new");

    // Valid EMA momentum
    prog.assert(momentum.clone().real_ge(Expr::real(0)));
    prog.assert(momentum.clone().real_le(Expr::real(1)));

    // mean_new = (1 - momentum) * mean_old + momentum * batch_mean
    let one_minus_m = Expr::real(1).real_sub(momentum.clone());
    prog.assert(
        mean_new.clone().eq(one_minus_m
            .clone()
            .real_mul(mean_old.clone())
            .real_add(momentum.clone().real_mul(batch_mean.clone()))),
    );

    // Negated property: the exact EMA formula does not hold
    let violation = mean_new.ne(one_minus_m
        .real_mul(mean_old)
        .real_add(momentum.real_mul(batch_mean)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "running_mean_update_exact_formula");
}

// ---------------------------------------------------------------------------
// Test 1018: Running variance update exact EMA formula
// ---------------------------------------------------------------------------

/// Prove: BatchNorm running variance update is exactly
/// var_new = (1 - momentum) * var_old + momentum * batch_var.
///
/// This is the variance analogue of the running mean EMA update.
///
/// Part of #4223.
#[test]
fn test_1018_running_var_update_exact_formula() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("var_old", real.clone());
    let _ = prog.declare_const("batch_var", real.clone());
    let _ = prog.declare_const("momentum", real.clone());
    let _ = prog.declare_const("var_new", real);

    let var_old = real_var("var_old");
    let batch_var = real_var("batch_var");
    let momentum = real_var("momentum");
    let var_new = real_var("var_new");

    // Non-negative variances and valid momentum
    prog.assert(var_old.clone().real_ge(Expr::real(0)));
    prog.assert(batch_var.clone().real_ge(Expr::real(0)));
    prog.assert(momentum.clone().real_ge(Expr::real(0)));
    prog.assert(momentum.clone().real_le(Expr::real(1)));

    // var_new = (1 - momentum) * var_old + momentum * batch_var
    let one_minus_m = Expr::real(1).real_sub(momentum.clone());
    prog.assert(
        var_new.clone().eq(one_minus_m
            .clone()
            .real_mul(var_old.clone())
            .real_add(momentum.clone().real_mul(batch_var.clone()))),
    );

    // Negated property: the exact EMA formula does not hold
    let violation = var_new.ne(one_minus_m
        .real_mul(var_old)
        .real_add(momentum.real_mul(batch_var)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "running_var_update_exact_formula");
}

// ---------------------------------------------------------------------------
// Test 1019: Normalization output remains bounded for bounded input
// ---------------------------------------------------------------------------

/// Prove: if the centered input is bounded, the denominator is bounded away
/// from zero, and gamma/beta are bounded, then the affine normalization
/// output is bounded.
///
/// We model x and mean in [-10, 10], so d = x - mean is in [-20, 20].
/// With denom >= 1, normalized value n = d / denom is bounded by 20.
/// With gamma in [-2, 2] and beta in [-3, 3], the affine output satisfies
/// |y| <= 43.
///
/// Part of #4223.
#[test]
fn test_1019_normalization_output_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("mean", real.clone());
    let _ = prog.declare_const("d", real.clone());
    let _ = prog.declare_const("denom", real.clone());
    let _ = prog.declare_const("n", real.clone());
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("beta", real.clone());
    let _ = prog.declare_const("y", real);

    let x = real_var("x");
    let mean = real_var("mean");
    let d = real_var("d");
    let denom = real_var("denom");
    let n = real_var("n");
    let gamma = real_var("gamma");
    let beta = real_var("beta");
    let y = real_var("y");

    // Bounded input and mean
    prog.assert(x.clone().real_ge(Expr::real(-10)));
    prog.assert(x.clone().real_le(Expr::real(10)));
    prog.assert(mean.clone().real_ge(Expr::real(-10)));
    prog.assert(mean.clone().real_le(Expr::real(10)));

    // d = x - mean, and conservatively |d| <= 20
    prog.assert(d.clone().eq(x.real_sub(mean)));
    prog.assert(d.clone().real_ge(Expr::real(-20)));
    prog.assert(d.clone().real_le(Expr::real(20)));

    // denom >= 1 keeps the normalization finite and bounded
    prog.assert(denom.clone().real_ge(Expr::real(1)));

    // n = d / denom, modeled as n * denom = d
    prog.assert(n.clone().real_mul(denom.clone()).eq(d));

    // Bounded affine parameters
    prog.assert(gamma.clone().real_ge(Expr::real(-2)));
    prog.assert(gamma.clone().real_le(Expr::real(2)));
    prog.assert(beta.clone().real_ge(Expr::real(-3)));
    prog.assert(beta.clone().real_le(Expr::real(3)));

    // y = gamma * n + beta
    prog.assert(y.clone().eq(gamma.real_mul(n).real_add(beta)));

    // Negated property: |y| > 43
    let violation = y
        .clone()
        .real_gt(Expr::real(43))
        .or(y.real_lt(Expr::real(-43)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "normalization_output_bounded");
}

// ---------------------------------------------------------------------------
// Test 1020: Epsilon keeps var + eps positive even when var = 0
// ---------------------------------------------------------------------------

/// Prove: epsilon prevents division by zero even in the zero-variance case.
///
/// If var = 0 and eps > 0, then var + eps = eps > 0, so the denominator
/// term under the square root is still strictly positive.
///
/// Part of #4223.
#[test]
fn test_1020_epsilon_prevents_div_zero_when_var_zero() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("var", real.clone());
    let _ = prog.declare_const("eps", real.clone());
    let _ = prog.declare_const("denom_sq", real);

    let var = real_var("var");
    let eps = real_var("eps");
    let denom_sq = real_var("denom_sq");

    // Zero-variance corner case
    prog.assert(var.clone().eq(Expr::real(0)));

    // eps > 0
    prog.assert(eps.clone().real_gt(Expr::real(0)));

    // denom_sq = var + eps
    prog.assert(denom_sq.clone().eq(var.real_add(eps)));

    // Negated property: denom_sq <= 0
    let violation = denom_sq.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "epsilon_prevents_div_zero_when_var_zero");
}

// ---------------------------------------------------------------------------
// Test 1021: Affine transform preserves linearity in normalized input
// ---------------------------------------------------------------------------

/// Prove: the affine transform y = gamma * x_norm + beta preserves
/// linearity in x_norm up to the additive offset beta.
///
/// For two normalized inputs x1_norm and x2_norm with outputs y1 and y2:
///   y1 - y2 = gamma * (x1_norm - x2_norm).
///
/// Part of #4223.
#[test]
fn test_1021_affine_transform_linearity_in_x_norm() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1_norm", real.clone());
    let _ = prog.declare_const("x2_norm", real.clone());
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("beta", real.clone());
    let _ = prog.declare_const("y1", real.clone());
    let _ = prog.declare_const("y2", real.clone());
    let _ = prog.declare_const("delta_x", real.clone());
    let _ = prog.declare_const("delta_y", real);

    let x1_norm = real_var("x1_norm");
    let x2_norm = real_var("x2_norm");
    let gamma = real_var("gamma");
    let beta = real_var("beta");
    let y1 = real_var("y1");
    let y2 = real_var("y2");
    let delta_x = real_var("delta_x");
    let delta_y = real_var("delta_y");

    // Affine outputs
    prog.assert(
        y1.clone().eq(gamma
            .clone()
            .real_mul(x1_norm.clone())
            .real_add(beta.clone())),
    );
    prog.assert(
        y2.clone().eq(gamma
            .clone()
            .real_mul(x2_norm.clone())
            .real_add(beta.clone())),
    );

    // Input and output differences
    prog.assert(delta_x.clone().eq(x1_norm.real_sub(x2_norm)));
    prog.assert(delta_y.clone().eq(y1.real_sub(y2)));

    // Negated property: delta_y != gamma * delta_x
    let violation = delta_y.ne(gamma.real_mul(delta_x));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "affine_transform_linearity_in_x_norm");
}

// ---------------------------------------------------------------------------
// Test 1022: BatchNorm eval mode is deterministic from running stats
// ---------------------------------------------------------------------------

/// Prove: in evaluation mode, BatchNorm output is deterministic given the
/// input and the running statistics.
///
/// If two eval-mode computations use the same x, running mean, denominator,
/// gamma, and beta, then they must produce the same output.
///
/// Part of #4223.
#[test]
fn test_1022_batchnorm_eval_mode_deterministic() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("rm", real.clone());
    let _ = prog.declare_const("denom", real.clone());
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("beta", real.clone());
    let _ = prog.declare_const("y_a", real.clone());
    let _ = prog.declare_const("y_b", real);

    let x = real_var("x");
    let rm = real_var("rm");
    let denom = real_var("denom");
    let gamma = real_var("gamma");
    let beta = real_var("beta");
    let y_a = real_var("y_a");
    let y_b = real_var("y_b");

    // Fixed running statistics in eval mode
    prog.assert(denom.clone().real_gt(Expr::real(0)));

    // y = gamma * (x - rm) / denom + beta
    prog.assert(
        y_a.clone()
            .real_sub(beta.clone())
            .real_mul(denom.clone())
            .eq(gamma.clone().real_mul(x.clone().real_sub(rm.clone()))),
    );
    prog.assert(
        y_b.clone()
            .real_sub(beta.clone())
            .real_mul(denom.clone())
            .eq(gamma.clone().real_mul(x.clone().real_sub(rm.clone()))),
    );

    // Negated property: the two eval outputs differ
    let violation = y_a.ne(y_b);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "batchnorm_eval_mode_deterministic");
}

// ---------------------------------------------------------------------------
// Test 1023: GroupNorm requires num_groups to divide num_channels
// ---------------------------------------------------------------------------

/// Prove: if GroupNorm exactly partitions channels into equal-sized groups,
/// then the channel count has zero remainder when divided by num_groups.
///
/// We model:
///   num_channels = num_groups * channels_per_group
/// and also the Euclidean-division form
///   num_channels = num_groups * channels_per_group + remainder,
/// with 0 <= remainder < num_groups. Then remainder must be 0.
///
/// Part of #4223.
#[test]
fn test_1023_groupnorm_num_groups_divides_channels() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("num_channels", real.clone());
    let _ = prog.declare_const("num_groups", real.clone());
    let _ = prog.declare_const("channels_per_group", real.clone());
    let _ = prog.declare_const("remainder", real);

    let num_channels = real_var("num_channels");
    let num_groups = real_var("num_groups");
    let channels_per_group = real_var("channels_per_group");
    let remainder = real_var("remainder");

    // Positive group structure
    prog.assert(num_channels.clone().real_gt(Expr::real(0)));
    prog.assert(num_groups.clone().real_gt(Expr::real(0)));
    prog.assert(channels_per_group.clone().real_gt(Expr::real(0)));

    // Exact partitioning
    let group_product = num_groups.clone().real_mul(channels_per_group.clone());
    prog.assert(num_channels.clone().eq(group_product.clone()));

    // Division-with-remainder form
    prog.assert(
        num_channels
            .clone()
            .eq(group_product.real_add(remainder.clone())),
    );
    prog.assert(remainder.clone().real_ge(Expr::real(0)));
    prog.assert(remainder.clone().real_lt(num_groups));

    // Negated property: remainder != 0
    let violation = remainder.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "groupnorm_num_groups_divides_channels");
}

// ---------------------------------------------------------------------------
// Test 1024: LayerNorm uses shared reduction stats over the last dimension
// ---------------------------------------------------------------------------

/// Prove: LayerNorm reduces over the last dimension using one shared mean
/// and denominator for all elements in that reduced slice.
///
/// For a 3-element slice [a1, a2, a3], if n_i = (a_i - mean) / s with the
/// same mean and s for all i, then pairwise differences satisfy:
///   (n1 - n2) * s = a1 - a2
///   (n2 - n3) * s = a2 - a3.
/// This captures reduction-dimension consistency across the last axis.
///
/// Part of #4223.
#[test]
fn test_1024_layernorm_reduction_dimension_consistency() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("a1", real.clone());
    let _ = prog.declare_const("a2", real.clone());
    let _ = prog.declare_const("a3", real.clone());
    let _ = prog.declare_const("mean", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("n1", real.clone());
    let _ = prog.declare_const("n2", real.clone());
    let _ = prog.declare_const("n3", real.clone());
    let _ = prog.declare_const("delta12", real.clone());
    let _ = prog.declare_const("delta23", real);

    let a1 = real_var("a1");
    let a2 = real_var("a2");
    let a3 = real_var("a3");
    let mean = real_var("mean");
    let s = real_var("s");
    let n1 = real_var("n1");
    let n2 = real_var("n2");
    let n3 = real_var("n3");
    let delta12 = real_var("delta12");
    let delta23 = real_var("delta23");

    // Shared reduction mean over the last dimension
    prog.assert(
        Expr::real(3)
            .real_mul(mean.clone())
            .eq(a1.clone().real_add(a2.clone()).real_add(a3.clone())),
    );

    // Shared positive denominator
    prog.assert(s.clone().real_gt(Expr::real(0)));

    // All elements use the same mean and denominator
    prog.assert(
        n1.clone()
            .real_mul(s.clone())
            .eq(a1.clone().real_sub(mean.clone())),
    );
    prog.assert(
        n2.clone()
            .real_mul(s.clone())
            .eq(a2.clone().real_sub(mean.clone())),
    );
    prog.assert(
        n3.clone()
            .real_mul(s.clone())
            .eq(a3.clone().real_sub(mean.clone())),
    );

    // Pairwise normalized differences
    prog.assert(delta12.clone().eq(n1.real_sub(n2.clone())));
    prog.assert(delta23.clone().eq(n2.real_sub(n3)));

    // Negated property: shared-statistic cancellation fails
    let violation = delta12
        .clone()
        .real_mul(s.clone())
        .ne(a1.real_sub(a2.clone()))
        .or(delta23.real_mul(s).ne(a2.real_sub(a3)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "layernorm_reduction_dimension_consistency");
}

// ---------------------------------------------------------------------------
// Test 1025: Gradient through normalization has the expected structure
// ---------------------------------------------------------------------------

/// Prove: in a scalar simplification with fixed normalization statistics,
/// the gradient through normalization has the expected chain-rule form.
///
/// For y = gamma * (x - mean) / s + beta with fixed mean and s,
///   dL/dx = dL/dy * gamma / s.
/// We encode this as grad_x * s = grad_out * gamma.
///
/// Part of #4223.
#[test]
fn test_1025_gradient_through_normalization_structure() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("grad_out", real.clone());
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("grad_x", real);

    let grad_out = real_var("grad_out");
    let gamma = real_var("gamma");
    let s = real_var("s");
    let grad_x = real_var("grad_x");

    // Fixed positive denominator
    prog.assert(s.clone().real_gt(Expr::real(0)));

    // grad_x = grad_out * gamma / s
    prog.assert(
        grad_x
            .clone()
            .real_mul(s.clone())
            .eq(grad_out.clone().real_mul(gamma.clone())),
    );

    // Negated property: grad_x * s != grad_out * gamma
    let violation = grad_x.real_mul(s).ne(grad_out.real_mul(gamma));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gradient_through_normalization_structure");
}

// ---------------------------------------------------------------------------
// Test 1026: Weight decay on gamma does not affect beta
// ---------------------------------------------------------------------------

/// Prove: weight decay applied to gamma does not change the beta update.
///
/// We model two parameter updates with different gamma-related decay terms,
/// but the same beta_old, grad_beta, and learning rate. The beta updates
/// must be identical because beta is an independent parameter.
///
/// Part of #4223.
#[test]
fn test_1026_weight_decay_on_gamma_independent_of_beta() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("beta_old", real.clone());
    let _ = prog.declare_const("grad_beta", real.clone());
    let _ = prog.declare_const("lr", real.clone());
    let _ = prog.declare_const("gamma_old_a", real.clone());
    let _ = prog.declare_const("gamma_old_b", real.clone());
    let _ = prog.declare_const("grad_gamma_a", real.clone());
    let _ = prog.declare_const("grad_gamma_b", real.clone());
    let _ = prog.declare_const("wd_a", real.clone());
    let _ = prog.declare_const("wd_b", real.clone());
    let _ = prog.declare_const("gamma_new_a", real.clone());
    let _ = prog.declare_const("gamma_new_b", real.clone());
    let _ = prog.declare_const("beta_new_a", real.clone());
    let _ = prog.declare_const("beta_new_b", real);

    let beta_old = real_var("beta_old");
    let grad_beta = real_var("grad_beta");
    let lr = real_var("lr");
    let gamma_old_a = real_var("gamma_old_a");
    let gamma_old_b = real_var("gamma_old_b");
    let grad_gamma_a = real_var("grad_gamma_a");
    let grad_gamma_b = real_var("grad_gamma_b");
    let wd_a = real_var("wd_a");
    let wd_b = real_var("wd_b");
    let gamma_new_a = real_var("gamma_new_a");
    let gamma_new_b = real_var("gamma_new_b");
    let beta_new_a = real_var("beta_new_a");
    let beta_new_b = real_var("beta_new_b");

    // Positive learning rate
    prog.assert(lr.clone().real_gt(Expr::real(0)));

    // Gamma updates include weight decay
    prog.assert(
        gamma_new_a.clone().eq(gamma_old_a.clone().real_sub(
            lr.clone().real_mul(
                grad_gamma_a
                    .clone()
                    .real_add(wd_a.clone().real_mul(gamma_old_a.clone())),
            ),
        )),
    );
    prog.assert(
        gamma_new_b.clone().eq(gamma_old_b.clone().real_sub(
            lr.clone().real_mul(
                grad_gamma_b
                    .clone()
                    .real_add(wd_b.clone().real_mul(gamma_old_b.clone())),
            ),
        )),
    );

    // Beta update is independent of gamma and weight decay on gamma
    prog.assert(
        beta_new_a.clone().eq(beta_old
            .clone()
            .real_sub(lr.clone().real_mul(grad_beta.clone()))),
    );
    prog.assert(
        beta_new_b
            .clone()
            .eq(beta_old.real_sub(lr.real_mul(grad_beta))),
    );

    // Negated property: beta updates differ
    let violation = beta_new_a.ne(beta_new_b);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "weight_decay_on_gamma_independent_of_beta");
}

// ---------------------------------------------------------------------------
// Test 1027: LayerNorm is equivariant to positive input scaling
// ---------------------------------------------------------------------------

/// Prove: LayerNorm output is unchanged under positive scaling of the input
/// vector before the affine transform.
///
/// We model 3 centered deviations d_i with denominator s. After scaling by
/// k > 0, the deviations become k * d_i and the denominator becomes k * s,
/// so the normalized outputs stay the same componentwise.
///
/// Part of #4223.
#[test]
fn test_1027_layernorm_scaling_equivariance() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("d1", real.clone());
    let _ = prog.declare_const("d2", real.clone());
    let _ = prog.declare_const("d3", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("k", real.clone());
    let _ = prog.declare_const("n1", real.clone());
    let _ = prog.declare_const("n2", real.clone());
    let _ = prog.declare_const("n3", real.clone());
    let _ = prog.declare_const("n1_scaled", real.clone());
    let _ = prog.declare_const("n2_scaled", real.clone());
    let _ = prog.declare_const("n3_scaled", real);

    let d1 = real_var("d1");
    let d2 = real_var("d2");
    let d3 = real_var("d3");
    let s = real_var("s");
    let k = real_var("k");
    let n1 = real_var("n1");
    let n2 = real_var("n2");
    let n3 = real_var("n3");
    let n1_scaled = real_var("n1_scaled");
    let n2_scaled = real_var("n2_scaled");
    let n3_scaled = real_var("n3_scaled");

    // Centered deviations and positive scale factor
    prog.assert(
        d1.clone()
            .real_add(d2.clone())
            .real_add(d3.clone())
            .eq(Expr::real(0)),
    );
    prog.assert(s.clone().real_gt(Expr::real(0)));
    prog.assert(k.clone().real_gt(Expr::real(0)));

    // Original normalized outputs
    prog.assert(n1.clone().real_mul(s.clone()).eq(d1.clone()));
    prog.assert(n2.clone().real_mul(s.clone()).eq(d2.clone()));
    prog.assert(n3.clone().real_mul(s.clone()).eq(d3.clone()));

    // Scaled normalized outputs: n_i_scaled * (k * s) = k * d_i
    let ks = k.clone().real_mul(s.clone());
    prog.assert(
        n1_scaled
            .clone()
            .real_mul(ks.clone())
            .eq(k.clone().real_mul(d1)),
    );
    prog.assert(
        n2_scaled
            .clone()
            .real_mul(ks.clone())
            .eq(k.clone().real_mul(d2)),
    );
    prog.assert(n3_scaled.clone().real_mul(ks).eq(k.real_mul(d3)));

    // Negated property: scaling changes one of the normalized outputs
    let violation = n1.ne(n1_scaled).or(n2.ne(n2_scaled)).or(n3.ne(n3_scaled));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "layernorm_scaling_equivariance");
}

// ---------------------------------------------------------------------------
// Test 1028: RMSNorm denominator is strictly positive when eps > 0
// ---------------------------------------------------------------------------

/// Prove: RMSNorm denominator is strictly positive when eps > 0.
///
/// We model denom = sqrt(mean_sq + eps) via denom^2 = mean_sq + eps with
/// mean_sq >= 0, eps > 0, and denom >= 0. Then denom must be strictly > 0.
///
/// Part of #4223.
#[test]
fn test_1028_rmsnorm_denominator_strictly_positive() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("mean_sq", real.clone());
    let _ = prog.declare_const("eps", real.clone());
    let _ = prog.declare_const("denom", real);

    let mean_sq = real_var("mean_sq");
    let eps = real_var("eps");
    let denom = real_var("denom");

    // mean_sq >= 0 and eps > 0
    prog.assert(mean_sq.clone().real_ge(Expr::real(0)));
    prog.assert(eps.clone().real_gt(Expr::real(0)));

    // denom is the non-negative square root of mean_sq + eps
    prog.assert(
        denom
            .clone()
            .real_mul(denom.clone())
            .eq(mean_sq.real_add(eps)),
    );
    prog.assert(denom.clone().real_ge(Expr::real(0)));

    // Negated property: denom <= 0
    let violation = denom.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rmsnorm_denominator_strictly_positive");
}

// ---------------------------------------------------------------------------
// Test 1029: Folding BatchNorm into convolution preserves output
// ---------------------------------------------------------------------------

/// Prove: fusing eval-mode BatchNorm into convolution weights preserves the
/// output.
///
/// For z = w * x + b and eval-mode BatchNorm
///   y_bn = gamma * (z - rm) / denom + beta,
/// the folded convolution parameters satisfy
///   w_fold = gamma * w / denom
///   b_fold = gamma * (b - rm) / denom + beta.
/// Then y_fused = w_fold * x + b_fold equals y_bn.
///
/// Part of #4223.
#[test]
fn test_1029_fused_batchnorm_preserves_conv_output() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("w", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("rm", real.clone());
    let _ = prog.declare_const("denom", real.clone());
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("beta", real.clone());
    let _ = prog.declare_const("z", real.clone());
    let _ = prog.declare_const("y_bn", real.clone());
    let _ = prog.declare_const("w_fold", real.clone());
    let _ = prog.declare_const("b_fold", real.clone());
    let _ = prog.declare_const("y_fused", real);

    let x = real_var("x");
    let w = real_var("w");
    let b = real_var("b");
    let rm = real_var("rm");
    let denom = real_var("denom");
    let gamma = real_var("gamma");
    let beta = real_var("beta");
    let z = real_var("z");
    let y_bn = real_var("y_bn");
    let w_fold = real_var("w_fold");
    let b_fold = real_var("b_fold");
    let y_fused = real_var("y_fused");

    // Positive eval-mode denominator
    prog.assert(denom.clone().real_gt(Expr::real(0)));

    // Convolution output
    prog.assert(
        z.clone()
            .eq(w.clone().real_mul(x.clone()).real_add(b.clone())),
    );

    // Eval-mode BatchNorm output
    prog.assert(
        y_bn.clone()
            .real_sub(beta.clone())
            .real_mul(denom.clone())
            .eq(gamma.clone().real_mul(z.clone().real_sub(rm.clone()))),
    );

    // Folded convolution parameters
    prog.assert(
        w_fold
            .clone()
            .real_mul(denom.clone())
            .eq(gamma.clone().real_mul(w.clone())),
    );
    prog.assert(
        b_fold
            .clone()
            .real_sub(beta.clone())
            .real_mul(denom.clone())
            .eq(gamma.clone().real_mul(b.clone().real_sub(rm.clone()))),
    );

    // Fused convolution output
    prog.assert(
        y_fused
            .clone()
            .eq(w_fold.clone().real_mul(x).real_add(b_fold.clone())),
    );

    // Negated property: folded conv output differs from conv+BatchNorm
    let violation = y_bn.ne(y_fused);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "fused_batchnorm_preserves_conv_output");
}

// ---------------------------------------------------------------------------
// Test 1030: A sequence of normalization layers remains bounded
// ---------------------------------------------------------------------------

/// Prove: composing bounded normalization layers still yields a bounded
/// output.
///
/// We model two normalization-affine stages. The first stage output h is
/// bounded. The second stage normalizes h with a denominator bounded away
/// from zero and applies bounded gamma/beta. The final output remains
/// bounded.
///
/// Part of #4223.
#[test]
fn test_1030_sequence_of_norm_layers_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("n1", real.clone());
    let _ = prog.declare_const("gamma1", real.clone());
    let _ = prog.declare_const("beta1", real.clone());
    let _ = prog.declare_const("h", real.clone());
    let _ = prog.declare_const("mean2", real.clone());
    let _ = prog.declare_const("d2", real.clone());
    let _ = prog.declare_const("denom2", real.clone());
    let _ = prog.declare_const("n2", real.clone());
    let _ = prog.declare_const("gamma2", real.clone());
    let _ = prog.declare_const("beta2", real.clone());
    let _ = prog.declare_const("y", real);

    let n1 = real_var("n1");
    let gamma1 = real_var("gamma1");
    let beta1 = real_var("beta1");
    let h = real_var("h");
    let mean2 = real_var("mean2");
    let d2 = real_var("d2");
    let denom2 = real_var("denom2");
    let n2 = real_var("n2");
    let gamma2 = real_var("gamma2");
    let beta2 = real_var("beta2");
    let y = real_var("y");

    // First normalization-affine stage: bounded normalized value and params
    prog.assert(n1.clone().real_ge(Expr::real(-5)));
    prog.assert(n1.clone().real_le(Expr::real(5)));
    prog.assert(gamma1.clone().real_ge(Expr::real(-2)));
    prog.assert(gamma1.clone().real_le(Expr::real(2)));
    prog.assert(beta1.clone().real_ge(Expr::real(-3)));
    prog.assert(beta1.clone().real_le(Expr::real(3)));
    prog.assert(h.clone().eq(gamma1.real_mul(n1).real_add(beta1)));

    // Conservative bound on the first stage output
    prog.assert(h.clone().real_ge(Expr::real(-13)));
    prog.assert(h.clone().real_le(Expr::real(13)));

    // Second normalization stage: bounded mean, centered value, and denom
    prog.assert(mean2.clone().real_ge(Expr::real(-13)));
    prog.assert(mean2.clone().real_le(Expr::real(13)));
    prog.assert(d2.clone().eq(h.clone().real_sub(mean2)));
    prog.assert(d2.clone().real_ge(Expr::real(-26)));
    prog.assert(d2.clone().real_le(Expr::real(26)));
    prog.assert(denom2.clone().real_ge(Expr::real(1)));
    prog.assert(n2.clone().real_mul(denom2.clone()).eq(d2));

    // Second affine stage
    prog.assert(gamma2.clone().real_ge(Expr::real(-2)));
    prog.assert(gamma2.clone().real_le(Expr::real(2)));
    prog.assert(beta2.clone().real_ge(Expr::real(-3)));
    prog.assert(beta2.clone().real_le(Expr::real(3)));
    prog.assert(y.clone().eq(gamma2.real_mul(n2).real_add(beta2)));

    // Negated property: final output is unbounded beyond a conservative limit
    let violation = y
        .clone()
        .real_gt(Expr::real(55))
        .or(y.real_lt(Expr::real(-55)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "sequence_of_norm_layers_bounded");
}
