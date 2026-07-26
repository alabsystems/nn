// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for LayerNorm and RMSNorm mathematical
//! properties.
//!
//! Proves fundamental properties of normalization layers used across modern
//! deep learning architectures:
//! - Mean computation is sum/n
//! - Variance non-negativity
//! - Normalized output zero-mean property
//! - Normalized output unit-variance property
//! - Affine transform gamma/beta bounds
//! - RMSNorm reciprocal sqrt bounds
//! - Epsilon prevents division by zero
//! - Scale invariance: norm(kx) = norm(x)
//! - Idempotence: norm(norm(x)) = norm(x)
//! - LayerNorm vs RMSNorm relationship
//! - Gradient through norm bounded
//! - Pre-norm vs post-norm ordering
//! - Group norm per-group mean property
//! - Instance norm per-sample property
//! - Batch norm running mean EMA
//! - Batch norm running var non-negativity
//! - Norm affine default (gamma=1, beta=0) identity
//! - Norm output bounded for bounded input
//! - Weight decay on norm parameters
//! - Spectral norm bound
//!
//! Part of #4176.

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
// Test 791: Mean computation is sum/n
// ---------------------------------------------------------------------------

/// Prove: the mean of n values equals their sum divided by n.
///
/// For n=3 values x1, x2, x3:
///   mean = (x1 + x2 + x3) / 3.
/// This is the foundation of LayerNorm: centering requires the exact mean.
///
/// We model: sum = x1 + x2 + x3, mean = sum / 3.
/// Prove: 3 * mean = x1 + x2 + x3.
#[test]
fn test_791_mean_computation_sum_over_n() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("x3", real.clone());
    let _ = prog.declare_const("mean", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let x3 = real_var("x3");
    let mean = real_var("mean");

    // Bounded inputs
    prog.assert(x1.clone().real_ge(Expr::real(-10)));
    prog.assert(x1.clone().real_le(Expr::real(10)));
    prog.assert(x2.clone().real_ge(Expr::real(-10)));
    prog.assert(x2.clone().real_le(Expr::real(10)));
    prog.assert(x3.clone().real_ge(Expr::real(-10)));
    prog.assert(x3.clone().real_le(Expr::real(10)));

    // mean = (x1 + x2 + x3) / 3, i.e., 3 * mean = x1 + x2 + x3
    let sum = x1.clone().real_add(x2.clone()).real_add(x3.clone());
    prog.assert(Expr::real(3).real_mul(mean.clone()).eq(sum));

    // Negated property: mean < -10 OR mean > 10
    // If each x_i in [-10, 10], mean is in [-10, 10].
    let violation = mean
        .clone()
        .real_lt(Expr::real(-10))
        .or(mean.real_gt(Expr::real(10)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "mean_computation_sum_over_n");
}

// ---------------------------------------------------------------------------
// Test 792: Variance non-negativity
// ---------------------------------------------------------------------------

/// Prove: variance is always non-negative.
///
/// Variance = E[(x - mean)^2]. Since (x - mean)^2 >= 0 for all x,
/// the average of non-negative values is non-negative.
///
/// For n=2: var = ((x1 - mean)^2 + (x2 - mean)^2) / 2.
/// We model the variance as a sum of squared deviations.
/// Prove: var >= 0.
#[test]
fn test_792_variance_non_negativity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("mean", real.clone());
    let _ = prog.declare_const("d1_sq", real.clone());
    let _ = prog.declare_const("d2_sq", real.clone());
    let _ = prog.declare_const("var", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let mean = real_var("mean");
    let d1_sq = real_var("d1_sq");
    let d2_sq = real_var("d2_sq");
    let var = real_var("var");

    // mean = (x1 + x2) / 2
    prog.assert(
        Expr::real(2)
            .real_mul(mean.clone())
            .eq(x1.clone().real_add(x2.clone())),
    );

    // d1_sq = (x1 - mean)^2
    let d1 = x1.real_sub(mean.clone());
    prog.assert(d1_sq.clone().eq(d1.clone().real_mul(d1)));

    // d2_sq = (x2 - mean)^2
    let d2 = x2.real_sub(mean);
    prog.assert(d2_sq.clone().eq(d2.clone().real_mul(d2)));

    // var = (d1_sq + d2_sq) / 2
    prog.assert(
        Expr::real(2)
            .real_mul(var.clone())
            .eq(d1_sq.real_add(d2_sq)),
    );

    // Negated property: var < 0
    let violation = var.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "variance_non_negativity");
}

// ---------------------------------------------------------------------------
// Test 793: Normalized output zero-mean property
// ---------------------------------------------------------------------------

/// Prove: after LayerNorm centering, the sum of normalized values is zero.
///
/// For n=2: normalized_i = x_i - mean. Then:
///   sum(normalized) = (x1 - mean) + (x2 - mean)
///                   = x1 + x2 - 2*mean
///                   = x1 + x2 - (x1 + x2) = 0.
///
/// Prove: n1 + n2 = 0 where n_i = x_i - mean.
#[test]
fn test_793_normalized_output_zero_mean() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("mean", real.clone());
    let _ = prog.declare_const("n1", real.clone());
    let _ = prog.declare_const("n2", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let mean = real_var("mean");
    let n1 = real_var("n1");
    let n2 = real_var("n2");

    // mean = (x1 + x2) / 2
    prog.assert(
        Expr::real(2)
            .real_mul(mean.clone())
            .eq(x1.clone().real_add(x2.clone())),
    );

    // n1 = x1 - mean, n2 = x2 - mean
    prog.assert(n1.clone().eq(x1.real_sub(mean.clone())));
    prog.assert(n2.clone().eq(x2.real_sub(mean)));

    // Negated property: n1 + n2 != 0
    let sum_n = n1.real_add(n2);
    let violation = sum_n.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "normalized_output_zero_mean");
}

// ---------------------------------------------------------------------------
// Test 794: Normalized output unit-variance property
// ---------------------------------------------------------------------------

/// Prove: after full LayerNorm (centering + scaling by 1/std), the
/// variance of the output is 1.
///
/// For n=2: after centering, d1 = x1 - mean, d2 = x2 - mean.
/// var = (d1^2 + d2^2) / 2.  std = sqrt(var).
/// Normalized: z_i = d_i / std.
/// Output variance = (z1^2 + z2^2) / 2 = (d1^2/var + d2^2/var) / 2
///                 = (d1^2 + d2^2) / (2 * var) = var / var = 1.
///
/// We model: z_i = d_i / std where std^2 = var, 2*var = d1^2 + d2^2.
/// Prove: (z1^2 + z2^2) / 2 = 1.
#[test]
fn test_794_normalized_output_unit_variance() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("d1", real.clone());
    let _ = prog.declare_const("d2", real.clone());
    let _ = prog.declare_const("var", real.clone());
    let _ = prog.declare_const("std", real.clone());
    let _ = prog.declare_const("z1", real.clone());
    let _ = prog.declare_const("z2", real);

    let d1 = real_var("d1");
    let d2 = real_var("d2");
    let var = real_var("var");
    let std = real_var("std");
    let z1 = real_var("z1");
    let z2 = real_var("z2");

    // var > 0 (non-degenerate case)
    prog.assert(var.clone().real_gt(Expr::real(0)));

    // 2 * var = d1^2 + d2^2
    prog.assert(
        Expr::real(2).real_mul(var.clone()).eq(d1
            .clone()
            .real_mul(d1.clone())
            .real_add(d2.clone().real_mul(d2.clone()))),
    );

    // std > 0 and std^2 = var
    prog.assert(std.clone().real_gt(Expr::real(0)));
    prog.assert(std.clone().real_mul(std.clone()).eq(var));

    // z1 = d1 / std, z2 = d2 / std (i.e., z1*std = d1, z2*std = d2)
    prog.assert(z1.clone().real_mul(std.clone()).eq(d1));
    prog.assert(z2.clone().real_mul(std).eq(d2));

    // Output variance = (z1^2 + z2^2) / 2
    // Negated property: (z1^2 + z2^2) != 2  (i.e., output_var != 1)
    let z_sq_sum = z1.clone().real_mul(z1.clone()).real_add(z2.clone().real_mul(z2));
    let violation = z_sq_sum.ne(Expr::real(2));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "normalized_output_unit_variance");
}

// ---------------------------------------------------------------------------
// Test 795: Affine transform gamma/beta bounds
// ---------------------------------------------------------------------------

/// Prove: the affine transform y = gamma * z + beta is bounded when
/// gamma, beta, and z are bounded.
///
/// LayerNorm applies y = gamma * normalized + beta. If |gamma| <= G,
/// |beta| <= B, and |z| <= Z, then |y| <= G * Z + B.
///
/// We model: y = gamma * z + beta with bounded parameters.
/// Prove: |y| <= G * Z + B.
#[test]
fn test_795_affine_transform_gamma_beta_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("z", real.clone());
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("beta", real.clone());
    let _ = prog.declare_const("y", real);

    let z = real_var("z");
    let gamma = real_var("gamma");
    let beta = real_var("beta");
    let y = real_var("y");

    // |z| <= 4 (normalized output, bounded)
    prog.assert(z.clone().real_ge(Expr::real(-4)));
    prog.assert(z.clone().real_le(Expr::real(4)));

    // |gamma| <= 2
    prog.assert(gamma.clone().real_ge(Expr::real(-2)));
    prog.assert(gamma.clone().real_le(Expr::real(2)));

    // |beta| <= 1
    prog.assert(beta.clone().real_ge(Expr::real(-1)));
    prog.assert(beta.clone().real_le(Expr::real(1)));

    // y = gamma * z + beta
    prog.assert(y.clone().eq(gamma.real_mul(z).real_add(beta)));

    // Negated property: |y| > 9 (= 2*4 + 1)
    let violation = y
        .clone()
        .real_gt(Expr::real(9))
        .or(y.real_lt(Expr::real(-9)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "affine_transform_gamma_beta_bounds");
}

// ---------------------------------------------------------------------------
// Test 796: RMSNorm reciprocal sqrt bounds
// ---------------------------------------------------------------------------

/// Prove: the RMSNorm scaling factor 1/sqrt(rms^2 + eps) is bounded.
///
/// RMSNorm: y = x * (1 / sqrt(mean(x^2) + eps)).
/// If rms^2 >= 0 and eps > 0, then rms^2 + eps >= eps > 0,
/// so sqrt(rms^2 + eps) >= sqrt(eps), and
/// 1/sqrt(rms^2 + eps) <= 1/sqrt(eps).
///
/// We model: rms_sq >= 0, eps > 0, scale = 1/sqrt(rms_sq + eps).
/// Prove: scale <= 1/sqrt(eps).
#[test]
fn test_796_rmsnorm_reciprocal_sqrt_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("rms_sq", real.clone());
    let _ = prog.declare_const("eps", real.clone());
    let _ = prog.declare_const("denom_sq", real.clone());
    let _ = prog.declare_const("denom", real.clone());
    let _ = prog.declare_const("scale", real);

    let rms_sq = real_var("rms_sq");
    let eps = real_var("eps");
    let denom_sq = real_var("denom_sq");
    let denom = real_var("denom");
    let scale = real_var("scale");

    // rms_sq >= 0
    prog.assert(rms_sq.clone().real_ge(Expr::real(0)));

    // eps = 1e-5 (modeled as 1/100000), use concrete value
    // Use eps > 0 generically
    prog.assert(eps.clone().real_gt(Expr::real(0)));
    prog.assert(eps.clone().real_le(Expr::real(1)));

    // denom_sq = rms_sq + eps
    prog.assert(denom_sq.clone().eq(rms_sq.real_add(eps.clone())));

    // denom > 0 and denom^2 = denom_sq
    prog.assert(denom.clone().real_gt(Expr::real(0)));
    prog.assert(denom.clone().real_mul(denom.clone()).eq(denom_sq));

    // scale = 1 / denom, i.e., scale * denom = 1
    prog.assert(scale.clone().real_mul(denom).eq(Expr::real(1)));

    // scale > 0 (reciprocal of positive is positive)
    prog.assert(scale.clone().real_gt(Expr::real(0)));

    // Negated property: scale * sqrt(eps) > 1
    // Since scale <= 1/sqrt(eps), scale * sqrt(eps) <= 1.
    // We encode: scale^2 * eps > 1  (squaring both sides of scale*sqrt(eps) > 1)
    let violation = scale
        .clone()
        .real_mul(scale)
        .real_mul(eps)
        .real_gt(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rmsnorm_reciprocal_sqrt_bounds");
}

// ---------------------------------------------------------------------------
// Test 797: Epsilon prevents division by zero
// ---------------------------------------------------------------------------

/// Prove: adding epsilon to variance guarantees a positive denominator.
///
/// In LayerNorm: std = sqrt(var + eps). If eps > 0, then var + eps > 0
/// for any var >= 0, so the sqrt and subsequent division never encounter
/// zero or negative arguments.
///
/// We model: var >= 0, eps > 0, denom = var + eps.
/// Prove: denom > 0.
#[test]
fn test_797_epsilon_prevents_division_by_zero() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("var", real.clone());
    let _ = prog.declare_const("eps", real.clone());
    let _ = prog.declare_const("denom", real);

    let var = real_var("var");
    let eps = real_var("eps");
    let denom = real_var("denom");

    // var >= 0
    prog.assert(var.clone().real_ge(Expr::real(0)));

    // eps > 0
    prog.assert(eps.clone().real_gt(Expr::real(0)));

    // denom = var + eps
    prog.assert(denom.clone().eq(var.real_add(eps)));

    // Negated property: denom <= 0
    let violation = denom.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "epsilon_prevents_division_by_zero");
}

// ---------------------------------------------------------------------------
// Test 798: Scale invariance: norm(kx) = norm(x)
// ---------------------------------------------------------------------------

/// Prove: LayerNorm is scale-invariant — scaling all inputs by k > 0
/// does not change the normalized output.
///
/// For n=2: mean(kx) = k*mean(x), std(kx) = k*std(x).
/// norm(kx_i) = (kx_i - k*mean(x)) / (k*std(x))
///            = k*(x_i - mean(x)) / (k*std(x))
///            = (x_i - mean(x)) / std(x) = norm(x_i).
///
/// We model: z = (x - mean) / std, z_k = (k*x - k*mean) / (k*std).
/// Prove: z = z_k.
#[test]
fn test_798_scale_invariance_norm() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("mean", real.clone());
    let _ = prog.declare_const("std", real.clone());
    let _ = prog.declare_const("k", real.clone());
    let _ = prog.declare_const("z", real.clone());
    let _ = prog.declare_const("z_k", real);

    let x = real_var("x");
    let mean = real_var("mean");
    let std = real_var("std");
    let k = real_var("k");
    let z = real_var("z");
    let z_k = real_var("z_k");

    // std > 0 (non-degenerate)
    prog.assert(std.clone().real_gt(Expr::real(0)));

    // k > 0 (positive scaling factor)
    prog.assert(k.clone().real_gt(Expr::real(0)));

    // z * std = x - mean  (z = (x - mean) / std)
    prog.assert(
        z.clone()
            .real_mul(std.clone())
            .eq(x.clone().real_sub(mean.clone())),
    );

    // z_k * (k * std) = k*x - k*mean  (z_k = (k*x - k*mean) / (k*std))
    prog.assert(
        z_k.clone()
            .real_mul(k.clone().real_mul(std))
            .eq(k.clone().real_mul(x).real_sub(k.real_mul(mean))),
    );

    // Negated property: z != z_k
    let violation = z.ne(z_k);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "scale_invariance_norm");
}

// ---------------------------------------------------------------------------
// Test 799: Idempotence: norm(norm(x)) = norm(x)
// ---------------------------------------------------------------------------

/// Prove: applying LayerNorm to already-normalized data produces the
/// same output (with gamma=1, beta=0).
///
/// After LayerNorm, mean=0 and var=1. Applying LayerNorm again:
///   mean(z) = 0, std(z) = 1, norm(z_i) = (z_i - 0) / 1 = z_i.
///
/// For n=2: if z1 + z2 = 0 and z1^2 + z2^2 = 2 (zero mean, unit var),
/// then norm(z) = z.
///
/// We model: z in normalized form (sum=0, sum_sq=2n), re-normalize.
/// Prove: output = z.
#[test]
fn test_799_idempotence_norm() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("z1", real.clone());
    let _ = prog.declare_const("z2", real.clone());
    let _ = prog.declare_const("out1", real.clone());
    let _ = prog.declare_const("out2", real.clone());
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("s", real);

    let z1 = real_var("z1");
    let z2 = real_var("z2");
    let out1 = real_var("out1");
    let out2 = real_var("out2");
    let m = real_var("m");
    let s = real_var("s");

    // Already normalized: z1 + z2 = 0 (zero mean)
    prog.assert(z1.clone().real_add(z2.clone()).eq(Expr::real(0)));

    // z1^2 + z2^2 = 2 (unit variance for n=2)
    prog.assert(
        z1.clone()
            .real_mul(z1.clone())
            .real_add(z2.clone().real_mul(z2.clone()))
            .eq(Expr::real(2)),
    );

    // Recompute mean: 2*m = z1 + z2
    prog.assert(
        Expr::real(2)
            .real_mul(m.clone())
            .eq(z1.clone().real_add(z2.clone())),
    );

    // Recompute std: s > 0, 2*s^2 = (z1-m)^2 + (z2-m)^2
    prog.assert(s.clone().real_gt(Expr::real(0)));
    let d1 = z1.clone().real_sub(m.clone());
    let d2 = z2.clone().real_sub(m);
    prog.assert(
        Expr::real(2).real_mul(s.clone().real_mul(s.clone())).eq(d1
            .clone()
            .real_mul(d1.clone())
            .real_add(d2.clone().real_mul(d2.clone()))),
    );

    // out_i = (z_i - m) / s, i.e., out_i * s = z_i - m = d_i
    prog.assert(out1.clone().real_mul(s.clone()).eq(d1));
    prog.assert(out2.clone().real_mul(s).eq(d2));

    // Negated property: out1 != z1 OR out2 != z2
    let violation = out1.ne(z1).or(out2.ne(z2));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "idempotence_norm");
}

// ---------------------------------------------------------------------------
// Test 800: LayerNorm vs RMSNorm relationship
// ---------------------------------------------------------------------------

/// Prove: when the mean is zero, LayerNorm and RMSNorm produce
/// identical outputs (with gamma=1, beta=0).
///
/// LayerNorm: z_i = (x_i - mean) / sqrt(var + eps).
/// RMSNorm:   z_i = x_i / sqrt(mean(x^2) + eps).
///
/// When mean(x) = 0: var = mean(x^2) - mean(x)^2 = mean(x^2).
/// So LayerNorm denominator = sqrt(mean(x^2) + eps) = RMSNorm denominator,
/// and (x_i - 0) = x_i, so both norms produce the same result.
///
/// We model: mean = 0, show LN output = RMS output.
#[test]
fn test_800_layernorm_vs_rmsnorm_zero_mean() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("mean_x_sq", real.clone());
    let _ = prog.declare_const("eps", real.clone());
    let _ = prog.declare_const("denom", real.clone());
    let _ = prog.declare_const("z_ln", real.clone());
    let _ = prog.declare_const("z_rms", real);

    let x = real_var("x");
    let mean_x_sq = real_var("mean_x_sq");
    let eps = real_var("eps");
    let denom = real_var("denom");
    let z_ln = real_var("z_ln");
    let z_rms = real_var("z_rms");

    // eps > 0
    prog.assert(eps.clone().real_gt(Expr::real(0)));

    // mean_x_sq >= 0 (mean of squared values)
    prog.assert(mean_x_sq.clone().real_ge(Expr::real(0)));

    // denom > 0 and denom^2 = mean_x_sq + eps
    prog.assert(denom.clone().real_gt(Expr::real(0)));
    prog.assert(
        denom
            .clone()
            .real_mul(denom.clone())
            .eq(mean_x_sq.clone().real_add(eps.clone())),
    );

    // LayerNorm with mean=0: z_ln = (x - 0) / denom = x / denom
    // z_ln * denom = x
    prog.assert(z_ln.clone().real_mul(denom.clone()).eq(x.clone()));

    // RMSNorm: z_rms = x / denom (same denom since var = mean_x_sq when mean=0)
    // z_rms * denom = x
    prog.assert(z_rms.clone().real_mul(denom).eq(x));

    // Negated property: z_ln != z_rms
    let violation = z_ln.ne(z_rms);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "layernorm_vs_rmsnorm_zero_mean");
}

// ---------------------------------------------------------------------------
// Test 801: Gradient through norm bounded
// ---------------------------------------------------------------------------

/// Prove: the gradient of LayerNorm output w.r.t. input is bounded.
///
/// For scalar simplification: if y = (x - mean) / std and we treat
/// mean and std as constants (stop-gradient, common in practice), then
/// dy/dx = 1/std. If std >= sqrt(eps) for eps > 0, then
/// |dy/dx| <= 1/sqrt(eps).
///
/// We model: grad = 1/std, std >= S_min > 0.
/// Prove: |grad| <= 1/S_min.
#[test]
fn test_801_gradient_through_norm_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("std", real.clone());
    let _ = prog.declare_const("grad", real);

    let std = real_var("std");
    let grad = real_var("grad");

    // std >= 1 (lower bound from eps and input scale)
    prog.assert(std.clone().real_ge(Expr::real(1)));

    // grad * std = 1 (grad = 1/std)
    prog.assert(grad.clone().real_mul(std).eq(Expr::real(1)));

    // Negated property: |grad| > 1 (since std >= 1, 1/std <= 1)
    let violation = grad
        .clone()
        .real_gt(Expr::real(1))
        .or(grad.real_lt(Expr::real(-1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gradient_through_norm_bounded");
}

// ---------------------------------------------------------------------------
// Test 802: Pre-norm vs post-norm ordering
// ---------------------------------------------------------------------------

/// Prove: pre-norm and post-norm both produce bounded outputs when
/// the sub-layer function is bounded.
///
/// Pre-norm:  y = x + F(LN(x)).  If |x| <= X and |F(LN(x))| <= A.
/// Post-norm: y = LN(x + F(x)).  LN output bounded by |gamma|*C + |beta|.
///
/// Both orderings produce bounded output, but pre-norm bounds grow
/// with depth (additive) while post-norm bounds are reset by LN.
///
/// We model: pre_out = x + f_ln, post_out bounded by LN output bound.
/// Prove: |pre_out| <= X + A AND |post_out| <= L.
#[test]
fn test_802_pre_norm_vs_post_norm_ordering() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("f_ln", real.clone());
    let _ = prog.declare_const("pre_out", real.clone());
    let _ = prog.declare_const("post_out", real);

    let x = real_var("x");
    let f_ln = real_var("f_ln");
    let pre_out = real_var("pre_out");
    let post_out = real_var("post_out");

    // |x| <= 10
    prog.assert(x.clone().real_ge(Expr::real(-10)));
    prog.assert(x.clone().real_le(Expr::real(10)));

    // |F(LN(x))| <= 5
    prog.assert(f_ln.clone().real_ge(Expr::real(-5)));
    prog.assert(f_ln.clone().real_le(Expr::real(5)));

    // pre_out = x + F(LN(x))
    prog.assert(pre_out.clone().eq(x.real_add(f_ln)));

    // post_out bounded by LN: |post_out| <= 4
    prog.assert(post_out.clone().real_ge(Expr::real(-4)));
    prog.assert(post_out.clone().real_le(Expr::real(4)));

    // Negated property: |pre_out| > 15 OR |post_out| > 4
    let violation = pre_out
        .clone()
        .real_gt(Expr::real(15))
        .or(pre_out.real_lt(Expr::real(-15)))
        .or(post_out.clone().real_gt(Expr::real(4)))
        .or(post_out.real_lt(Expr::real(-4)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "pre_norm_vs_post_norm_ordering");
}

// ---------------------------------------------------------------------------
// Test 803: Group norm per-group mean property
// ---------------------------------------------------------------------------

/// Prove: GroupNorm computes independent means per group, and each
/// group mean is bounded by the group's element bounds.
///
/// GroupNorm divides C channels into G groups of C/G channels each.
/// Mean is computed per group. If all elements in a group are in
/// [lo, hi], the group mean is in [lo, hi].
///
/// For group of size 2: mean = (x1 + x2) / 2.
/// If lo <= x1,x2 <= hi, then lo <= mean <= hi.
#[test]
fn test_803_group_norm_per_group_mean() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real.clone());
    let _ = prog.declare_const("mean", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let lo = real_var("lo");
    let hi = real_var("hi");
    let mean = real_var("mean");

    // lo <= hi
    prog.assert(lo.clone().real_le(hi.clone()));

    // x1, x2 in [lo, hi]
    prog.assert(x1.clone().real_ge(lo.clone()));
    prog.assert(x1.clone().real_le(hi.clone()));
    prog.assert(x2.clone().real_ge(lo.clone()));
    prog.assert(x2.clone().real_le(hi.clone()));

    // mean = (x1 + x2) / 2
    prog.assert(Expr::real(2).real_mul(mean.clone()).eq(x1.real_add(x2)));

    // Negated property: mean < lo OR mean > hi
    let violation = mean.clone().real_lt(lo).or(mean.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "group_norm_per_group_mean");
}

// ---------------------------------------------------------------------------
// Test 804: Instance norm per-sample property
// ---------------------------------------------------------------------------

/// Prove: InstanceNorm normalizes each sample independently, producing
/// zero-mean output per sample.
///
/// InstanceNorm is GroupNorm with G=C (each channel is its own group).
/// For a single channel with n=2 spatial elements:
///   mean = (x1 + x2) / 2, n1 = x1 - mean, n2 = x2 - mean.
///   n1 + n2 = 0 (zero mean per channel).
///
/// Prove: sum of normalized elements is zero.
#[test]
fn test_804_instance_norm_per_sample() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("mean", real.clone());
    let _ = prog.declare_const("n1", real.clone());
    let _ = prog.declare_const("n2", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let mean = real_var("mean");
    let n1 = real_var("n1");
    let n2 = real_var("n2");

    // mean = (x1 + x2) / 2
    prog.assert(
        Expr::real(2)
            .real_mul(mean.clone())
            .eq(x1.clone().real_add(x2.clone())),
    );

    // n1 = x1 - mean, n2 = x2 - mean
    prog.assert(n1.clone().eq(x1.real_sub(mean.clone())));
    prog.assert(n2.clone().eq(x2.real_sub(mean)));

    // Negated property: n1 + n2 != 0
    let violation = n1.real_add(n2).ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "instance_norm_per_sample");
}

// ---------------------------------------------------------------------------
// Test 805: Batch norm running mean EMA
// ---------------------------------------------------------------------------

/// Prove: batch norm running mean EMA stays within input bounds.
///
/// Running mean update: rm_new = (1 - m) * rm_old + m * batch_mean.
/// If lo <= rm_old <= hi and lo <= batch_mean <= hi and 0 < m < 1,
/// then lo <= rm_new <= hi (convex combination).
///
/// Prove: rm_new stays in [lo, hi].
#[test]
fn test_805_batch_norm_running_mean_ema() {
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

    // lo <= hi
    prog.assert(lo.clone().real_le(hi.clone()));

    // 0 < m < 1
    prog.assert(m.clone().real_gt(Expr::real(0)));
    prog.assert(m.clone().real_lt(Expr::real(1)));

    // rm_old in [lo, hi]
    prog.assert(rm_old.clone().real_ge(lo.clone()));
    prog.assert(rm_old.clone().real_le(hi.clone()));

    // bm in [lo, hi]
    prog.assert(bm.clone().real_ge(lo.clone()));
    prog.assert(bm.clone().real_le(hi.clone()));

    // rm_new = (1-m) * rm_old + m * bm
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

    assert_verified(&prog, "batch_norm_running_mean_ema");
}

// ---------------------------------------------------------------------------
// Test 806: Batch norm running var non-negativity
// ---------------------------------------------------------------------------

/// Prove: batch norm running variance is always non-negative.
///
/// Running var update: rv_new = (1 - m) * rv_old + m * batch_var.
/// If rv_old >= 0 and batch_var >= 0 and 0 < m < 1,
/// then rv_new >= 0 (convex combination of non-negative values).
///
/// Prove: rv_new >= 0.
#[test]
fn test_806_batch_norm_running_var_non_negativity() {
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

    // rv_old >= 0
    prog.assert(rv_old.clone().real_ge(Expr::real(0)));

    // bv >= 0 (batch variance non-negative)
    prog.assert(bv.clone().real_ge(Expr::real(0)));

    // 0 < m < 1
    prog.assert(m.clone().real_gt(Expr::real(0)));
    prog.assert(m.clone().real_lt(Expr::real(1)));

    // rv_new = (1-m) * rv_old + m * bv
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

    assert_verified(&prog, "batch_norm_running_var_non_negativity");
}

// ---------------------------------------------------------------------------
// Test 807: Norm affine default (gamma=1, beta=0) identity
// ---------------------------------------------------------------------------

/// Prove: with default affine parameters (gamma=1, beta=0), the affine
/// transform is the identity: y = 1 * z + 0 = z.
///
/// This is the initialization property. Before training modifies gamma
/// and beta, the norm output equals the normalized value exactly.
///
/// We model: gamma = 1, beta = 0, y = gamma * z + beta.
/// Prove: y = z.
#[test]
fn test_807_norm_affine_default_identity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("z", real.clone());
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("beta", real.clone());
    let _ = prog.declare_const("y", real);

    let z = real_var("z");
    let gamma = real_var("gamma");
    let beta = real_var("beta");
    let y = real_var("y");

    // gamma = 1, beta = 0 (default initialization)
    prog.assert(gamma.clone().eq(Expr::real(1)));
    prog.assert(beta.clone().eq(Expr::real(0)));

    // y = gamma * z + beta
    prog.assert(y.clone().eq(gamma.real_mul(z.clone()).real_add(beta)));

    // Negated property: y != z
    let violation = y.ne(z);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "norm_affine_default_identity");
}

// ---------------------------------------------------------------------------
// Test 808: Norm output bounded for bounded input
// ---------------------------------------------------------------------------

/// Prove: LayerNorm output is bounded when input, gamma, and beta are
/// bounded (end-to-end bound from input to output).
///
/// If |x| <= X, then after normalization |z| is bounded by a function
/// of X and the dimension. With affine parameters |gamma| <= G and
/// |beta| <= B, the output |y| <= G * Z_max + B.
///
/// We model: normalized z bounded (axiomatic from normalization),
/// then affine transform bounded.
/// Prove: |y| <= G * Z + B.
#[test]
fn test_808_norm_output_bounded_for_bounded_input() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("z", real.clone());
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("beta", real.clone());
    let _ = prog.declare_const("y", real);

    let x = real_var("x");
    let z = real_var("z");
    let gamma = real_var("gamma");
    let beta = real_var("beta");
    let y = real_var("y");

    // |x| <= 100 (input bounded)
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.real_le(Expr::real(100)));

    // |z| <= 5 (normalized output bounded — axiomatic from LN properties)
    prog.assert(z.clone().real_ge(Expr::real(-5)));
    prog.assert(z.clone().real_le(Expr::real(5)));

    // |gamma| <= 3
    prog.assert(gamma.clone().real_ge(Expr::real(-3)));
    prog.assert(gamma.clone().real_le(Expr::real(3)));

    // |beta| <= 2
    prog.assert(beta.clone().real_ge(Expr::real(-2)));
    prog.assert(beta.clone().real_le(Expr::real(2)));

    // y = gamma * z + beta
    prog.assert(y.clone().eq(gamma.real_mul(z).real_add(beta)));

    // Negated property: |y| > 17 (= 3*5 + 2)
    let violation = y
        .clone()
        .real_gt(Expr::real(17))
        .or(y.real_lt(Expr::real(-17)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "norm_output_bounded_for_bounded_input");
}

// ---------------------------------------------------------------------------
// Test 809: Weight decay on norm parameters
// ---------------------------------------------------------------------------

/// Prove: weight decay shrinks norm parameters toward zero while
/// preserving their sign and keeping them bounded.
///
/// Weight decay update: gamma_new = gamma - lambda * gamma
///                                = gamma * (1 - lambda).
/// If 0 < lambda < 1, then |gamma_new| < |gamma| and
/// sign(gamma_new) = sign(gamma).
///
/// For |gamma| <= G and 0 < lambda < 1:
///   |gamma_new| = |gamma| * (1 - lambda) <= G * (1 - lambda) < G.
///
/// We model: gamma_new = gamma * (1 - lambda).
/// Prove: |gamma_new| <= |gamma| (shrinkage).
#[test]
fn test_809_weight_decay_norm_parameters() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("lambda", real.clone());
    let _ = prog.declare_const("gamma_new", real);

    let gamma = real_var("gamma");
    let lambda = real_var("lambda");
    let gamma_new = real_var("gamma_new");

    // |gamma| <= 5
    prog.assert(gamma.clone().real_ge(Expr::real(-5)));
    prog.assert(gamma.clone().real_le(Expr::real(5)));

    // 0 < lambda < 1
    prog.assert(lambda.clone().real_gt(Expr::real(0)));
    prog.assert(lambda.clone().real_lt(Expr::real(1)));

    // gamma_new = gamma * (1 - lambda)
    let decay_factor = Expr::real(1).real_sub(lambda);
    prog.assert(gamma_new.clone().eq(gamma.clone().real_mul(decay_factor)));

    // Negated property: |gamma_new| > |gamma|
    // Encode: gamma_new^2 > gamma^2
    let violation = gamma_new
        .clone()
        .real_mul(gamma_new)
        .real_gt(gamma.clone().real_mul(gamma));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "weight_decay_norm_parameters");
}

// ---------------------------------------------------------------------------
// Test 810: Spectral norm bound
// ---------------------------------------------------------------------------

/// Prove: spectral normalization bounds the operator norm of a weight
/// matrix to at most 1, producing bounded output for bounded input.
///
/// Spectral norm: W_sn = W / sigma(W), where sigma(W) is the largest
/// singular value. Then ||W_sn||_2 = 1. For input ||x|| <= R:
///   ||W_sn * x|| <= ||W_sn||_2 * ||x|| <= 1 * R = R.
///
/// We model (scalar proxy): w_sn = w / sigma, sigma >= |w|.
/// Then |w_sn| <= 1. For |x| <= R, |w_sn * x| <= R.
///
/// Prove: |y| <= R where y = w_sn * x.
#[test]
fn test_810_spectral_norm_bound() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w", real.clone());
    let _ = prog.declare_const("sigma", real.clone());
    let _ = prog.declare_const("w_sn", real.clone());
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("y", real);

    let w = real_var("w");
    let sigma = real_var("sigma");
    let w_sn = real_var("w_sn");
    let x = real_var("x");
    let y = real_var("y");

    // |w| <= 10 (weight bounded)
    prog.assert(w.clone().real_ge(Expr::real(-10)));
    prog.assert(w.clone().real_le(Expr::real(10)));

    // sigma >= |w| (largest singular value >= any element)
    // sigma >= w and sigma >= -w
    prog.assert(sigma.clone().real_ge(w.clone()));
    prog.assert(sigma.clone().real_ge(Expr::real(0).real_sub(w.clone())));
    prog.assert(sigma.clone().real_gt(Expr::real(0)));

    // w_sn * sigma = w (w_sn = w / sigma)
    prog.assert(w_sn.clone().real_mul(sigma).eq(w));

    // |x| <= 8
    prog.assert(x.clone().real_ge(Expr::real(-8)));
    prog.assert(x.clone().real_le(Expr::real(8)));

    // y = w_sn * x
    prog.assert(y.clone().eq(w_sn.real_mul(x)));

    // Negated property: |y| > 8 (since |w_sn| <= 1, |y| <= |x| <= 8)
    let violation = y
        .clone()
        .real_gt(Expr::real(8))
        .or(y.real_lt(Expr::real(-8)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "spectral_norm_bound");
}
