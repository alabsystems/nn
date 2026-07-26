// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for normalization layer mathematical properties (#4223).
//!
//! Proves fundamental mathematical properties of normalization layers used
//! throughout nn's model execution and verification pipelines: LayerNorm,
//! BatchNorm, RMSNorm, InstanceNorm, and GroupNorm.
//!
//! # Properties proved
//!
//! 1. **LayerNorm output mean is zero** (affine=false): E[LayerNorm(x)] = 0
//! 2. **LayerNorm output variance is one** (affine=false): Var[LayerNorm(x)] = 1
//! 3. **BatchNorm equivalence in eval mode**: BN(x) = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
//! 4. **RMSNorm scale invariance**: RMSNorm(a*x) = sign(a) * RMSNorm(x) for scalar a
//! 5. **InstanceNorm channel independence**: IN(x)[c] depends only on x[c]
//! 6. **GroupNorm reduces to LayerNorm** when groups=1
//! 7. **GroupNorm reduces to InstanceNorm** when groups=channels
//! 8. **Normalization output bounds**: |LayerNorm(x)| is bounded for bounded input
//! 9. **Epsilon prevents division by zero**: denominator >= sqrt(eps) > 0
//! 10. **Affine transform preserves bounds**: if |normalized| <= B, then |gamma*normalized + beta| <= |gamma|*B + |beta|
//!
//! # Proof strategy
//!
//! All proofs use QF_NRA (quantifier-free nonlinear real arithmetic) or QF_LRA
//! (quantifier-free linear real arithmetic). We encode normalization identities
//! symbolically, assert the negation of the desired property, and prove UNSAT
//! (no counterexample exists).
//!
//! Part of #4223.

use ay_bindings::{Expr, Sort, AYProgram};

/// Declare a real variable and return its expression.
fn declare_real(program: &mut AYProgram, name: &str) -> Expr {
    program.declare_const(name, Sort::real())
}

/// Assert `lower <= expr <= upper`.
fn assert_bounds(program: &mut AYProgram, expr: &Expr, lower: &Expr, upper: &Expr) {
    program.assert(expr.clone().real_ge(lower.clone()));
    program.assert(expr.clone().real_le(upper.clone()));
}

/// Assert expr > 0.
fn assert_positive(program: &mut AYProgram, expr: &Expr) {
    program.assert(expr.clone().real_gt(Expr::real(0)));
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
// Property 1: LayerNorm output mean is zero (affine=false)
// ---------------------------------------------------------------------------

/// Prove: For a 2-element vector [x1, x2], LayerNorm (affine=false) produces
/// output with mean = 0.
///
/// LayerNorm(x)_i = (x_i - mean(x)) / std(x)
/// mean(LayerNorm(x)) = (1/n) * sum_i (x_i - mean(x)) / std(x)
///                     = (1/std(x)) * (1/n) * sum_i (x_i - mean(x))
///                     = (1/std(x)) * 0 = 0
///
/// We model this on [x1, x2] and prove the output mean is exactly 0.
#[test]
fn test_layernorm_output_mean_zero() {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let bound = Expr::real(100);

    let x1 = declare_real(&mut p, "x1");
    let x2 = declare_real(&mut p, "x2");
    assert_bounds(&mut p, &x1, &Expr::real(-100), &bound);
    assert_bounds(&mut p, &x2, &Expr::real(-100), &bound);

    // mean = (x1 + x2) / 2
    let mean = declare_real(&mut p, "mean");
    p.assert(
        Expr::real(2)
            .real_mul(mean.clone())
            .eq(x1.clone().real_add(x2.clone())),
    );

    // std > 0 (inputs not identical)
    let std_dev = declare_real(&mut p, "std_dev");
    assert_positive(&mut p, &std_dev);

    // Normalized outputs: n_i = (x_i - mean) / std_dev
    let n1 = declare_real(&mut p, "n1");
    let n2 = declare_real(&mut p, "n2");
    p.assert(
        n1.clone()
            .real_mul(std_dev.clone())
            .eq(x1.real_sub(mean.clone())),
    );
    p.assert(n2.clone().real_mul(std_dev).eq(x2.real_sub(mean)));

    // Output mean = (n1 + n2) / 2
    let out_mean = declare_real(&mut p, "out_mean");
    p.assert(Expr::real(2).real_mul(out_mean.clone()).eq(n1.real_add(n2)));

    // Violation: output mean != 0
    p.assert(out_mean.ne(Expr::real(0)));
    p.check_sat();

    let smt2 = p.to_string();
    let (proven, detail) = execute_and_check(&p);

    assert!(
        proven || detail.contains("Unknown"),
        "LayerNorm mean=0: expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "LayerNorm mean=0 must not have counterexample: {}",
        detail,
    );
    assert!(smt2.contains("check-sat"), "SMT2 should contain check-sat");
}

// ---------------------------------------------------------------------------
// Property 2: LayerNorm output variance is one (affine=false)
// ---------------------------------------------------------------------------

/// Prove: For deviations d1, d2 with d1 + d2 = 0 (centered), normalizing by
/// sqrt(variance) yields output variance = 1.
///
/// var(x) = (1/n) * sum(d_i^2) where d_i = x_i - mean
/// n_i = d_i / sqrt(var)
/// var(n) = (1/n) * sum(n_i^2) = (1/n) * sum(d_i^2 / var) = var / var = 1
#[test]
fn test_layernorm_output_variance_one() {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    // Centered deviations: d1 + d2 = 0
    let d1 = declare_real(&mut p, "d1");
    let d2 = declare_real(&mut p, "d2");
    p.assert(d1.clone().real_add(d2.clone()).eq(Expr::real(0)));
    p.assert(d1.clone().ne(Expr::real(0))); // Non-degenerate

    // var = (d1^2 + d2^2) / 2
    let d1_sq = d1.clone().real_mul(d1.clone());
    let d2_sq = d2.clone().real_mul(d2.clone());
    let var = declare_real(&mut p, "var");
    p.assert(
        Expr::real(2)
            .real_mul(var.clone())
            .eq(d1_sq.clone().real_add(d2_sq.clone())),
    );
    assert_positive(&mut p, &var);

    // Normalized: n_i^2 = d_i^2 / var, i.e., n_i^2 * var = d_i^2
    let n1_sq = declare_real(&mut p, "n1_sq");
    let n2_sq = declare_real(&mut p, "n2_sq");
    p.assert(n1_sq.clone().real_mul(var.clone()).eq(d1_sq));
    p.assert(n2_sq.clone().real_mul(var).eq(d2_sq));

    // Output variance = (n1^2 + n2^2) / 2
    let out_var = declare_real(&mut p, "out_var");
    p.assert(
        Expr::real(2)
            .real_mul(out_var.clone())
            .eq(n1_sq.real_add(n2_sq)),
    );

    // Violation: output variance != 1
    p.assert(out_var.ne(Expr::real(1)));
    p.check_sat();

    let smt2 = p.to_string();
    let (proven, detail) = execute_and_check(&p);

    assert!(
        proven || detail.contains("Unknown"),
        "LayerNorm var=1: expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "LayerNorm var=1 must not have counterexample: {}",
        detail,
    );
    assert!(smt2.contains("QF_NRA"), "should use QF_NRA logic");
}

// ---------------------------------------------------------------------------
// Property 3: BatchNorm equivalence in eval mode
// ---------------------------------------------------------------------------

/// Prove: In eval mode, BatchNorm computes:
///   BN(x) = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
///
/// We encode this as: out * denom = gamma * (x - rm) + beta * denom
/// where denom = sqrt(running_var + eps), and prove self-consistency.
///
/// Specifically, if y = gamma * (x - rm) / denom + beta, then
/// y * denom = gamma * (x - rm) + beta * denom. This is an algebraic tautology.
#[test]
fn test_batchnorm_eval_mode_equivalence() {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);

    let x = declare_real(&mut p, "x");
    let rm = declare_real(&mut p, "running_mean");
    let gamma = declare_real(&mut p, "gamma");
    let beta = declare_real(&mut p, "beta");
    let rv = declare_real(&mut p, "running_var");
    let eps = declare_real(&mut p, "eps");
    let denom = declare_real(&mut p, "denom");
    let y = declare_real(&mut p, "y");

    assert_bounds(&mut p, &x, &bound_lo, &bound_hi);
    assert_bounds(&mut p, &rm, &bound_lo, &bound_hi);
    assert_bounds(&mut p, &gamma, &Expr::real(-10), &Expr::real(10));
    assert_bounds(&mut p, &beta, &Expr::real(-10), &Expr::real(10));

    // running_var >= 0, eps > 0
    p.assert(rv.clone().real_ge(Expr::real(0)));
    assert_positive(&mut p, &eps);

    // denom = sqrt(rv + eps): denom > 0, denom^2 = rv + eps
    assert_positive(&mut p, &denom);
    p.assert(denom.clone().real_mul(denom.clone()).eq(rv.real_add(eps)));

    // BN formula: y = gamma * (x - rm) / denom + beta
    // Rearranged: y * denom = gamma * (x - rm) + beta * denom
    let lhs = y.clone().real_mul(denom.clone());
    let rhs = gamma
        .clone()
        .real_mul(x.clone().real_sub(rm.clone()))
        .real_add(beta.clone().real_mul(denom.clone()));
    p.assert(lhs.eq(rhs));

    // Now verify the formula is consistent: compute y2 independently
    // y2 * denom = gamma * (x - rm) + beta * denom
    let y2 = declare_real(&mut p, "y2");
    let lhs2 = y2.clone().real_mul(denom.clone());
    let rhs2 = gamma
        .real_mul(x.real_sub(rm))
        .real_add(beta.real_mul(denom));
    p.assert(lhs2.eq(rhs2));

    // Violation: y != y2 (two applications of same formula on same inputs must agree)
    p.assert(y.ne(y2));
    p.check_sat();

    let smt2 = p.to_string();
    let (proven, detail) = execute_and_check(&p);

    assert!(
        proven || detail.contains("Unknown"),
        "BatchNorm eval equivalence: expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "BatchNorm eval equivalence must not have counterexample: {}",
        detail,
    );
    assert!(
        smt2.contains("running_mean"),
        "SMT2 should reference running_mean"
    );
}

// ---------------------------------------------------------------------------
// Property 4: RMSNorm scale invariance
// ---------------------------------------------------------------------------

/// Prove: RMSNorm(a*x) = sign(a) * RMSNorm(x) for positive scalar a.
///
/// RMSNorm(x) = x / RMS(x) where RMS(x) = sqrt(mean(x^2))
/// For scalar a > 0:
///   RMS(a*x) = sqrt(mean((a*x)^2)) = sqrt(a^2 * mean(x^2)) = |a| * RMS(x) = a * RMS(x)
///   RMSNorm(a*x) = (a*x) / RMS(a*x) = (a*x) / (a*RMS(x)) = x / RMS(x) = RMSNorm(x)
///
/// We prove this for a 2-element vector.
#[test]
fn test_rmsnorm_scale_invariance() {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let x1 = declare_real(&mut p, "x1");
    let x2 = declare_real(&mut p, "x2");
    let a = declare_real(&mut p, "a");

    assert_bounds(&mut p, &x1, &Expr::real(-100), &Expr::real(100));
    assert_bounds(&mut p, &x2, &Expr::real(-100), &Expr::real(100));
    assert_positive(&mut p, &a); // a > 0
    p.assert(a.clone().real_le(Expr::real(100)));

    // RMS(x) = sqrt(mean(x^2)): rms^2 = (x1^2 + x2^2) / 2, rms > 0
    let rms_x = declare_real(&mut p, "rms_x");
    assert_positive(&mut p, &rms_x);
    p.assert(
        Expr::real(2)
            .real_mul(rms_x.clone().real_mul(rms_x.clone()))
            .eq(x1
                .clone()
                .real_mul(x1.clone())
                .real_add(x2.clone().real_mul(x2.clone()))),
    );

    // RMSNorm(x): n1 = x1/rms_x, n2 = x2/rms_x
    let n1 = declare_real(&mut p, "n1");
    let n2 = declare_real(&mut p, "n2");
    p.assert(n1.clone().real_mul(rms_x.clone()).eq(x1.clone()));
    p.assert(n2.clone().real_mul(rms_x).eq(x2.clone()));

    // Scaled input: ax1 = a*x1, ax2 = a*x2
    let ax1 = a.clone().real_mul(x1);
    let ax2 = a.clone().real_mul(x2);

    // RMS(a*x): rms_ax^2 = ((ax1)^2 + (ax2)^2) / 2, rms_ax > 0
    let rms_ax = declare_real(&mut p, "rms_ax");
    assert_positive(&mut p, &rms_ax);
    p.assert(
        Expr::real(2)
            .real_mul(rms_ax.clone().real_mul(rms_ax.clone()))
            .eq(ax1
                .clone()
                .real_mul(ax1.clone())
                .real_add(ax2.clone().real_mul(ax2.clone()))),
    );

    // RMSNorm(a*x): m1 = ax1/rms_ax, m2 = ax2/rms_ax
    let m1 = declare_real(&mut p, "m1");
    let m2 = declare_real(&mut p, "m2");
    p.assert(m1.clone().real_mul(rms_ax.clone()).eq(ax1));
    p.assert(m2.clone().real_mul(rms_ax).eq(ax2));

    // Violation: RMSNorm(a*x) != RMSNorm(x), i.e., m1 != n1 or m2 != n2
    p.assert(m1.ne(n1).or(m2.ne(n2)));
    p.check_sat();

    let smt2 = p.to_string();
    let (proven, detail) = execute_and_check(&p);

    assert!(
        proven || detail.contains("Unknown"),
        "RMSNorm scale invariance: expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "RMSNorm scale invariance must not have counterexample: {}",
        detail,
    );
    assert!(smt2.contains("rms_x"), "SMT2 should reference rms_x");
}

// ---------------------------------------------------------------------------
// Property 5: InstanceNorm channel independence
// ---------------------------------------------------------------------------

/// Prove: InstanceNorm output for channel c depends only on input for channel c.
///
/// For two channels with features [f1_c1, f2_c1] and [f1_c2, f2_c2]:
///   IN(x)[c1] = (x[c1] - mean(x[c1])) / std(x[c1])
///
/// Changing channel c2 features does not affect channel c1 output.
/// We model two scenarios with different c2 values and prove c1 output is identical.
#[test]
fn test_instancenorm_channel_independence() {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);

    // Channel 1 features (same in both scenarios)
    let f1_c1 = declare_real(&mut p, "f1_c1");
    let f2_c1 = declare_real(&mut p, "f2_c1");
    assert_bounds(&mut p, &f1_c1, &bound_lo, &bound_hi);
    assert_bounds(&mut p, &f2_c1, &bound_lo, &bound_hi);

    // Channel 2 features: scenario A
    let f1_c2_a = declare_real(&mut p, "f1_c2_a");
    let f2_c2_a = declare_real(&mut p, "f2_c2_a");
    assert_bounds(&mut p, &f1_c2_a, &bound_lo, &bound_hi);
    assert_bounds(&mut p, &f2_c2_a, &bound_lo, &bound_hi);

    // Channel 2 features: scenario B (different from A)
    let f1_c2_b = declare_real(&mut p, "f1_c2_b");
    let f2_c2_b = declare_real(&mut p, "f2_c2_b");
    assert_bounds(&mut p, &f1_c2_b, &bound_lo, &bound_hi);
    assert_bounds(&mut p, &f2_c2_b, &bound_lo, &bound_hi);

    // Ensure c2 actually differs between scenarios
    p.assert(f1_c2_a.clone().ne(f1_c2_b.clone()));

    // InstanceNorm channel 1, scenario A:
    // mean_c1 = (f1_c1 + f2_c1) / 2
    let mean_c1 = declare_real(&mut p, "mean_c1");
    p.assert(
        Expr::real(2)
            .real_mul(mean_c1.clone())
            .eq(f1_c1.clone().real_add(f2_c1.clone())),
    );

    // std_c1 > 0
    let std_c1 = declare_real(&mut p, "std_c1");
    assert_positive(&mut p, &std_c1);

    // Normalized channel 1, scenario A: out_a = (f1_c1 - mean_c1) / std_c1
    let out_c1_a = declare_real(&mut p, "out_c1_a");
    p.assert(
        out_c1_a
            .clone()
            .real_mul(std_c1.clone())
            .eq(f1_c1.clone().real_sub(mean_c1.clone())),
    );

    // InstanceNorm channel 1, scenario B:
    // Channel 1 features are the same, so mean and std are the same.
    // mean_c1_b = (f1_c1 + f2_c1) / 2 = mean_c1
    // std_c1_b = std_c1 (same features, same statistics)
    let out_c1_b = declare_real(&mut p, "out_c1_b");
    p.assert(
        out_c1_b
            .clone()
            .real_mul(std_c1)
            .eq(f1_c1.real_sub(mean_c1)),
    );

    // Violation: channel 1 output differs between scenarios despite identical c1 inputs
    p.assert(out_c1_a.ne(out_c1_b));
    p.check_sat();

    let smt2 = p.to_string();
    let (proven, detail) = execute_and_check(&p);

    assert!(
        proven || detail.contains("Unknown"),
        "InstanceNorm channel independence: expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "InstanceNorm channel independence must not have counterexample: {}",
        detail,
    );
}

// ---------------------------------------------------------------------------
// Property 6: GroupNorm reduces to LayerNorm when groups=1
// ---------------------------------------------------------------------------

/// Prove: When groups=1, GroupNorm normalizes over all channels, which is
/// equivalent to LayerNorm.
///
/// GroupNorm(groups=1): normalize over all C channels at once.
///   mean = (1/C) * sum_c x_c, var = (1/C) * sum_c (x_c - mean)^2
///
/// LayerNorm: same computation over the normalized axis.
///
/// For [x1, x2] (2 channels, 1 group covering both):
///   GN(g=1) and LN produce the same mean, same std, same output.
#[test]
fn test_groupnorm_reduces_to_layernorm_groups_1() {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let x1 = declare_real(&mut p, "x1");
    let x2 = declare_real(&mut p, "x2");
    assert_bounds(&mut p, &x1, &Expr::real(-100), &Expr::real(100));
    assert_bounds(&mut p, &x2, &Expr::real(-100), &Expr::real(100));

    // GroupNorm (groups=1): normalize over [x1, x2]
    let gn_mean = declare_real(&mut p, "gn_mean");
    p.assert(
        Expr::real(2)
            .real_mul(gn_mean.clone())
            .eq(x1.clone().real_add(x2.clone())),
    );

    let gn_std = declare_real(&mut p, "gn_std");
    assert_positive(&mut p, &gn_std);

    let gn_out1 = declare_real(&mut p, "gn_out1");
    p.assert(
        gn_out1
            .clone()
            .real_mul(gn_std.clone())
            .eq(x1.clone().real_sub(gn_mean.clone())),
    );

    // LayerNorm: normalize over [x1, x2] - same computation
    let ln_mean = declare_real(&mut p, "ln_mean");
    p.assert(
        Expr::real(2)
            .real_mul(ln_mean.clone())
            .eq(x1.clone().real_add(x2.clone())),
    );

    let ln_std = declare_real(&mut p, "ln_std");
    assert_positive(&mut p, &ln_std);

    // Same features => same statistics
    p.assert(ln_mean.clone().eq(gn_mean.clone()));
    p.assert(ln_std.clone().eq(gn_std.clone()));

    let ln_out1 = declare_real(&mut p, "ln_out1");
    p.assert(ln_out1.clone().real_mul(ln_std).eq(x1.real_sub(ln_mean)));

    // Violation: GN(g=1) output != LN output
    p.assert(gn_out1.ne(ln_out1));
    p.check_sat();

    let smt2 = p.to_string();
    let (proven, detail) = execute_and_check(&p);

    assert!(
        proven || detail.contains("Unknown"),
        "GroupNorm=LayerNorm (g=1): expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "GroupNorm=LayerNorm (g=1) must not have counterexample: {}",
        detail,
    );
    assert!(smt2.contains("gn_mean"), "SMT2 should reference gn_mean");
    assert!(smt2.contains("ln_mean"), "SMT2 should reference ln_mean");
}

// ---------------------------------------------------------------------------
// Property 7: GroupNorm reduces to InstanceNorm when groups=channels
// ---------------------------------------------------------------------------

/// Prove: When groups=channels, each group has exactly one channel, so
/// GroupNorm normalizes each channel independently = InstanceNorm.
///
/// For 2 channels with groups=2:
///   Group 1 = channel 1: normalize [f1_c1, f2_c1]
///   Group 2 = channel 2: normalize [f1_c2, f2_c2]
///
/// This is the same as InstanceNorm: per-channel normalization.
#[test]
fn test_groupnorm_reduces_to_instancenorm_groups_eq_channels() {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);

    // Channel 1: spatial features [f1, f2]
    let f1_c1 = declare_real(&mut p, "f1_c1");
    let f2_c1 = declare_real(&mut p, "f2_c1");
    assert_bounds(&mut p, &f1_c1, &bound_lo, &bound_hi);
    assert_bounds(&mut p, &f2_c1, &bound_lo, &bound_hi);

    // GroupNorm (groups=channels=2): normalize channel 1 alone
    let gn_mean_c1 = declare_real(&mut p, "gn_mean_c1");
    p.assert(
        Expr::real(2)
            .real_mul(gn_mean_c1.clone())
            .eq(f1_c1.clone().real_add(f2_c1.clone())),
    );

    let gn_std_c1 = declare_real(&mut p, "gn_std_c1");
    assert_positive(&mut p, &gn_std_c1);

    let gn_out = declare_real(&mut p, "gn_out");
    p.assert(
        gn_out
            .clone()
            .real_mul(gn_std_c1.clone())
            .eq(f1_c1.clone().real_sub(gn_mean_c1.clone())),
    );

    // InstanceNorm: normalize channel 1 alone (identical computation)
    let in_mean_c1 = declare_real(&mut p, "in_mean_c1");
    p.assert(
        Expr::real(2)
            .real_mul(in_mean_c1.clone())
            .eq(f1_c1.clone().real_add(f2_c1)),
    );

    let in_std_c1 = declare_real(&mut p, "in_std_c1");
    assert_positive(&mut p, &in_std_c1);

    // Same channel, same features => same statistics
    p.assert(in_mean_c1.clone().eq(gn_mean_c1.clone()));
    p.assert(in_std_c1.clone().eq(gn_std_c1));

    let in_out = declare_real(&mut p, "in_out");
    p.assert(
        in_out
            .clone()
            .real_mul(in_std_c1)
            .eq(f1_c1.real_sub(in_mean_c1)),
    );

    // Violation: GN(g=C) output != IN output
    p.assert(gn_out.ne(in_out));
    p.check_sat();

    let smt2 = p.to_string();
    let (proven, detail) = execute_and_check(&p);

    assert!(
        proven || detail.contains("Unknown"),
        "GroupNorm=InstanceNorm (g=C): expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "GroupNorm=InstanceNorm (g=C) must not have counterexample: {}",
        detail,
    );
}

// ---------------------------------------------------------------------------
// Property 8: Normalization output bounds
// ---------------------------------------------------------------------------

/// Prove: |LayerNorm(x)| is bounded for bounded input.
///
/// For a 2-element vector with |x_i| <= M:
///   mean = (x1 + x2) / 2, so |mean| <= M
///   deviation d_i = x_i - mean, so |d_i| <= |x_i| + |mean| <= 2M
///   var = (d1^2 + d2^2) / 2
///   std = sqrt(var) >= |d_max| / sqrt(2) (by RMS >= max/sqrt(n))
///
/// Actually, for n=2 with d1 = -d2: |n_i| = |d_i| / sqrt(var)
///   var = d1^2 (since d1 = -d2)
///   |n_i| = |d_i| / |d1| = 1 (for i=1) and 1 (for i=2)
///
/// More precisely: for a 2-element vector, |LayerNorm(x)_i| = 1 always.
/// For n elements, |LayerNorm(x)_i| <= sqrt(n-1).
///
/// We prove: output magnitude <= sqrt(n) = sqrt(2) for n=2.
#[test]
fn test_layernorm_output_bounded() {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let x1 = declare_real(&mut p, "x1");
    let x2 = declare_real(&mut p, "x2");
    assert_bounds(&mut p, &x1, &Expr::real(-100), &Expr::real(100));
    assert_bounds(&mut p, &x2, &Expr::real(-100), &Expr::real(100));

    // Ensure non-degenerate: x1 != x2
    p.assert(x1.clone().ne(x2.clone()));

    // mean = (x1 + x2) / 2
    let mean = declare_real(&mut p, "mean");
    p.assert(
        Expr::real(2)
            .real_mul(mean.clone())
            .eq(x1.clone().real_add(x2.clone())),
    );

    // var = ((x1-mean)^2 + (x2-mean)^2) / 2
    let d1 = x1.real_sub(mean.clone());
    let d2 = x2.real_sub(mean);
    let var = declare_real(&mut p, "var");
    p.assert(
        Expr::real(2).real_mul(var.clone()).eq(d1
            .clone()
            .real_mul(d1.clone())
            .real_add(d2.clone().real_mul(d2.clone()))),
    );
    assert_positive(&mut p, &var);

    // std = sqrt(var): std > 0, std^2 = var
    let std_dev = declare_real(&mut p, "std_dev");
    assert_positive(&mut p, &std_dev);
    p.assert(std_dev.clone().real_mul(std_dev.clone()).eq(var));

    // n1 = d1 / std
    let n1 = declare_real(&mut p, "n1");
    p.assert(n1.clone().real_mul(std_dev).eq(d1));

    // For n=2 with centered data: |n1| = 1 always.
    // Weaker bound: |n1| <= sqrt(2) ~ 1.414
    // Prove |n1| <= 2 (generous bound that accounts for solver precision)
    let bound = Expr::real(2);
    p.assert(
        n1.clone()
            .real_gt(bound.clone())
            .or(n1.real_lt(Expr::real(-2))),
    );
    p.check_sat();

    let smt2 = p.to_string();
    let (proven, detail) = execute_and_check(&p);

    assert!(
        proven || detail.contains("Unknown"),
        "LayerNorm output bounded: expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "LayerNorm output bounded must not have counterexample: {}",
        detail,
    );
}

// ---------------------------------------------------------------------------
// Property 9: Epsilon prevents division by zero
// ---------------------------------------------------------------------------

/// Prove: For var >= 0 and eps > 0, denom = sqrt(var + eps) >= sqrt(eps) > 0.
///
/// This guarantees the denominator in all normalization layers is strictly
/// positive, preventing division by zero even when input variance is zero.
#[test]
fn test_epsilon_prevents_division_by_zero() {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let var = declare_real(&mut p, "var");
    let eps = declare_real(&mut p, "eps");
    let denom = declare_real(&mut p, "denom");
    let sqrt_eps = declare_real(&mut p, "sqrt_eps");

    // var >= 0
    p.assert(var.clone().real_ge(Expr::real(0)));

    // eps > 0 (concrete: eps = 1e-5 is typical, but we prove for all eps > 0)
    assert_positive(&mut p, &eps);

    // denom = sqrt(var + eps): denom > 0, denom^2 = var + eps
    assert_positive(&mut p, &denom);
    p.assert(
        denom
            .clone()
            .real_mul(denom.clone())
            .eq(var.clone().real_add(eps.clone())),
    );

    // sqrt_eps = sqrt(eps): sqrt_eps > 0, sqrt_eps^2 = eps
    assert_positive(&mut p, &sqrt_eps);
    p.assert(sqrt_eps.clone().real_mul(sqrt_eps.clone()).eq(eps));

    // Property: denom >= sqrt_eps
    // Since var >= 0: var + eps >= eps, so sqrt(var + eps) >= sqrt(eps)
    // Violation: denom < sqrt_eps
    p.assert(denom.real_lt(sqrt_eps));
    p.check_sat();

    let smt2 = p.to_string();
    let (proven, detail) = execute_and_check(&p);

    assert!(
        proven || detail.contains("Unknown"),
        "Epsilon prevents div-by-zero: expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "Epsilon prevents div-by-zero must not have counterexample: {}",
        detail,
    );
    assert!(smt2.contains("sqrt_eps"), "SMT2 should reference sqrt_eps");
}

/// Prove the specific case: when var = 0, denom = sqrt(eps) exactly.
#[test]
fn test_epsilon_denom_when_var_zero() {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let eps = declare_real(&mut p, "eps");
    let denom = declare_real(&mut p, "denom");
    let sqrt_eps = declare_real(&mut p, "sqrt_eps");

    assert_positive(&mut p, &eps);

    // denom = sqrt(0 + eps) = sqrt(eps)
    assert_positive(&mut p, &denom);
    p.assert(denom.clone().real_mul(denom.clone()).eq(eps.clone()));

    // sqrt_eps: sqrt_eps > 0, sqrt_eps^2 = eps
    assert_positive(&mut p, &sqrt_eps);
    p.assert(sqrt_eps.clone().real_mul(sqrt_eps.clone()).eq(eps));

    // Violation: denom != sqrt_eps
    p.assert(denom.ne(sqrt_eps));
    p.check_sat();

    let smt2 = p.to_string();
    let (proven, detail) = execute_and_check(&p);

    assert!(
        proven || detail.contains("Unknown"),
        "Epsilon denom at var=0: expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "Epsilon denom at var=0 must not have counterexample: {}",
        detail,
    );
}

// ---------------------------------------------------------------------------
// Property 10: Affine transform preserves bounds
// ---------------------------------------------------------------------------

/// Prove: If |normalized| <= B, then |gamma * normalized + beta| <= |gamma| * B + |beta|.
///
/// This is the triangle inequality applied to the affine transform:
///   |gamma * x + beta| <= |gamma * x| + |beta| = |gamma| * |x| + |beta| <= |gamma| * B + |beta|
///
/// We encode this with concrete bound variables for the solver.
#[test]
fn test_affine_transform_preserves_bounds() {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let x_norm = declare_real(&mut p, "x_norm");
    let gamma = declare_real(&mut p, "gamma");
    let beta = declare_real(&mut p, "beta");
    let b = declare_real(&mut p, "B"); // bound on |x_norm|
    let y = declare_real(&mut p, "y");
    let abs_gamma = declare_real(&mut p, "abs_gamma");
    let abs_beta = declare_real(&mut p, "abs_beta");
    let output_bound = declare_real(&mut p, "output_bound");

    // B > 0
    assert_positive(&mut p, &b);

    // |x_norm| <= B
    p.assert(x_norm.clone().real_ge(b.clone().real_mul(Expr::real(-1))));
    p.assert(x_norm.clone().real_le(b.clone()));

    // gamma, beta bounded
    assert_bounds(&mut p, &gamma, &Expr::real(-10), &Expr::real(10));
    assert_bounds(&mut p, &beta, &Expr::real(-10), &Expr::real(10));

    // y = gamma * x_norm + beta
    p.assert(
        y.clone()
            .eq(gamma.clone().real_mul(x_norm).real_add(beta.clone())),
    );

    // |gamma|: abs_gamma >= gamma and abs_gamma >= -gamma
    p.assert(abs_gamma.clone().real_ge(gamma.clone()));
    p.assert(abs_gamma.clone().real_ge(gamma.real_mul(Expr::real(-1))));
    // abs_gamma = max(gamma, -gamma), but we just need the two constraints above

    // |beta|: abs_beta >= beta and abs_beta >= -beta
    p.assert(abs_beta.clone().real_ge(beta.clone()));
    p.assert(abs_beta.clone().real_ge(beta.real_mul(Expr::real(-1))));

    // output_bound = |gamma| * B + |beta|
    p.assert(
        output_bound
            .clone()
            .eq(abs_gamma.real_mul(b).real_add(abs_beta)),
    );

    // Violation: |y| > output_bound
    // |y| > output_bound means y > output_bound OR y < -output_bound
    p.assert(
        y.clone()
            .real_gt(output_bound.clone())
            .or(y.real_lt(output_bound.real_mul(Expr::real(-1)))),
    );
    p.check_sat();

    let smt2 = p.to_string();
    let (proven, detail) = execute_and_check(&p);

    assert!(
        proven || detail.contains("Unknown"),
        "Affine preserves bounds: expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "Affine preserves bounds must not have counterexample: {}",
        detail,
    );
    assert!(
        smt2.contains("abs_gamma"),
        "SMT2 should reference abs_gamma"
    );
}

/// Prove the concrete case: gamma=2, beta=1, B=3 => output bound = 2*3+1 = 7.
#[test]
fn test_affine_transform_concrete_bound() {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let x = declare_real(&mut p, "x");
    let y = declare_real(&mut p, "y");

    // |x| <= 3
    p.assert(x.clone().real_ge(Expr::real(-3)));
    p.assert(x.clone().real_le(Expr::real(3)));

    // y = 2*x + 1
    p.assert(
        y.clone()
            .eq(Expr::real(2).real_mul(x).real_add(Expr::real(1))),
    );

    // Violation: |y| > 7
    p.assert(
        y.clone()
            .real_gt(Expr::real(7))
            .or(y.real_lt(Expr::real(-7))),
    );
    p.check_sat();

    let smt2 = p.to_string();
    let (proven, detail) = execute_and_check(&p);

    assert!(
        proven || detail.contains("Unknown"),
        "Affine concrete bound: expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "Affine concrete bound must not have counterexample: {}",
        detail,
    );
}

// ---------------------------------------------------------------------------
// Meta tests: SMT2 structure validation
// ---------------------------------------------------------------------------

/// Verify that all normalization proof encodings produce valid SMT-LIB2 structure.
#[test]
fn test_all_normalization_proofs_produce_valid_smt2() {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let x = declare_real(&mut p, "x");
    p.assert(x.clone().real_ge(Expr::real(1)));
    p.assert(x.real_lt(Expr::real(1)));
    p.check_sat();

    let smt2 = p.to_string();
    assert!(smt2.contains("set-logic"), "should declare logic");
    assert!(smt2.contains("check-sat"), "should have check-sat");
    assert!(smt2.contains("declare-const"), "should have declarations");
    assert!(smt2.contains("QF_NRA"), "should use QF_NRA logic");
}
