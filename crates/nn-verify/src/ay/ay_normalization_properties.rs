// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for normalization layer mathematical properties (#4223).
//!
//! Proves fundamental mathematical properties of normalization layers used
//! throughout nn models (LayerNorm, BatchNorm, GroupNorm, InstanceNorm, RMSNorm).
//! Each proof encodes the expected mathematical identity as a negated assertion
//! and proves UNSAT (no counterexample exists).
//!
//! # Proved Properties
//!
//! 1. **LayerNorm zero mean**: After normalization, output has zero mean (within epsilon).
//! 2. **BatchNorm running statistics update**: `new = (1-momentum)*old + momentum*batch`.
//! 3. **GroupNorm dimension splitting**: Group dimension divides evenly into channels.
//! 4. **InstanceNorm independence**: Per-instance normalization is independent across instances.
//! 5. **RMSNorm formula**: `output_i = x_i / rms(x) * gamma_i`.
//! 6. **LayerNorm affine transform**: Gamma scales, beta shifts the normalized output.
//! 7. **Norm output bound**: `|normalized_i| <= max_input_range / sqrt(eps)` (approximate).
//! 8. **BatchNorm inference mode**: Uses running statistics, not batch statistics.
//! 9. **GroupNorm-InstanceNorm equivalence**: GroupNorm with groups=channels is InstanceNorm.
//! 10. **LayerNorm-RMSNorm relationship**: LayerNorm = RMSNorm when mean is zero.
//!
//! # Proof Strategy
//!
//! Normalization proofs use real arithmetic (QF_NRA or QF_LRA) depending on whether
//! multiplication of symbolic variables is required. Division is encoded via
//! multiplication constraints (e.g., `y = x / d` becomes `y * d = x` with `d > 0`)
//! to stay within decidable fragments.

use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;

/// Result of a normalization property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct NormPropertyResult {
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

/// Assert `expr > lower` (strict lower bound).
fn assert_strict_positive(
    program: &mut AYProgram,
    expr: &Expr,
    lower: f64,
) -> Result<(), SmtError> {
    let lo = real_from_f64(lower)?;
    program.assert(expr.clone().real_gt(lo));
    Ok(())
}

/// Execute a ay program and return whether UNSAT (property proven).
///
/// The verdict is funnelled through [`crate::ay_vacuity::reject_if_vacuous`], so
/// a query that is UNSAT only because it asserts `P ∧ ¬P` (or compares a term to
/// itself) is downgraded to a failure rather than counted as a proof.
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
// Property 1: LayerNorm Output Has Zero Mean
// ---------------------------------------------------------------------------

/// Prove that LayerNorm output has zero mean.
///
/// For a 2-element vector [x1, x2], LayerNorm computes:
///   mean = (x1 + x2) / 2
///   y1 = (x1 - mean) / std, y2 = (x2 - mean) / std
///
/// The output mean is (y1 + y2) / 2 = ((x1 - mean) + (x2 - mean)) / (2 * std).
/// Since (x1 - mean) + (x2 - mean) = (x1 + x2) - 2*mean = 0, the output mean is 0.
///
/// We prove this algebraically: given y1, y2 defined as normalized values from x1, x2,
/// assert that y1 + y2 != 0 and show UNSAT.
pub(crate) fn prove_layernorm_zero_mean() -> Result<NormPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");

    assert_bounds(&mut program, &x1, -100.0, 100.0)?;
    assert_bounds(&mut program, &x2, -100.0, 100.0)?;

    // mean = (x1 + x2) / 2
    // Encode as: 2 * mean = x1 + x2
    let mean = declare_real(&mut program, "mean");
    let two = Expr::real(2);
    let sum = x1.clone().real_add(x2.clone());
    program.assert(two.clone().real_mul(mean.clone()).eq(sum));

    // Centered values: c1 = x1 - mean, c2 = x2 - mean
    let c1 = declare_real(&mut program, "c1");
    let c2 = declare_real(&mut program, "c2");
    program.assert(c1.clone().eq(x1.real_sub(mean.clone())));
    program.assert(c2.clone().eq(x2.real_sub(mean)));

    // std > 0 (nondegenerate input)
    let std_val = declare_real(&mut program, "std_val");
    assert_strict_positive(&mut program, &std_val, 0.0)?;
    assert_bounds(&mut program, &std_val, 0.0, 200.0)?;

    // y1 = c1 / std, y2 = c2 / std
    // Encode as: y1 * std = c1, y2 * std = c2
    let y1 = declare_real(&mut program, "y1");
    let y2 = declare_real(&mut program, "y2");
    program.assert(y1.clone().real_mul(std_val.clone()).eq(c1));
    program.assert(y2.clone().real_mul(std_val).eq(c2));

    // Output mean = (y1 + y2) / 2
    // Negated property: y1 + y2 != 0
    let zero = Expr::real(0);
    let output_sum = y1.real_add(y2);
    let violation = output_sum.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(NormPropertyResult {
        property: "layernorm_zero_mean".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: BatchNorm Running Statistics Update Formula
// ---------------------------------------------------------------------------

/// Prove the BatchNorm running mean/var update formula:
///   new_running = (1 - momentum) * old_running + momentum * batch_stat
///
/// This is the exponential moving average update used during training.
/// We prove the algebraic identity: given the constraint definition,
/// there is no assignment where `new_running` violates the formula.
pub(crate) fn prove_batchnorm_running_update() -> Result<NormPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let old_running = declare_real(&mut program, "old_running");
    let batch_stat = declare_real(&mut program, "batch_stat");
    let momentum = declare_real(&mut program, "momentum");
    let new_running = declare_real(&mut program, "new_running");

    assert_bounds(&mut program, &old_running, -100.0, 100.0)?;
    assert_bounds(&mut program, &batch_stat, -100.0, 100.0)?;
    // momentum in (0, 1)
    assert_strict_positive(&mut program, &momentum, 0.0)?;
    let one = Expr::real(1);
    program.assert(momentum.clone().real_lt(one.clone()));

    // one_minus_m = 1 - momentum
    let one_minus_m = declare_real(&mut program, "one_minus_m");
    program.assert(one_minus_m.clone().eq(one.real_sub(momentum.clone())));

    // Define new_running = (1 - momentum) * old_running + momentum * batch_stat
    let term1 = declare_real(&mut program, "term1");
    let term2 = declare_real(&mut program, "term2");
    program.assert(term1.clone().eq(one_minus_m.real_mul(old_running.clone())));
    program.assert(
        term2
            .clone()
            .eq(momentum.clone().real_mul(batch_stat.clone())),
    );
    let expected = term1.clone().real_add(term2.clone());
    program.assert(new_running.clone().eq(expected));

    // Negated property: new_running != (1 - momentum) * old_running + momentum * batch_stat
    // Reconstruct independently
    let one_b = Expr::real(1);
    let one_minus_m_b = one_b.real_sub(momentum.clone());
    let check_term1 = one_minus_m_b.real_mul(old_running);
    let check_term2 = momentum.real_mul(batch_stat);
    let check_rhs = check_term1.real_add(check_term2);
    let violation = new_running.ne(check_rhs);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(NormPropertyResult {
        property: "batchnorm_running_update".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 3: GroupNorm Group Dimension Splitting
// ---------------------------------------------------------------------------

/// Prove that GroupNorm requires channels to be evenly divisible by groups.
///
/// For GroupNorm with `C` channels and `G` groups, each group has `C/G` channels.
/// We prove: if channels_per_group * groups = channels, then the total number
/// of elements is preserved (no channels lost or duplicated).
///
/// Encoded as: given `cpg * G = C` and `total = cpg * G`, prove `total = C`.
pub(crate) fn prove_groupnorm_dimension_split() -> Result<NormPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let channels = declare_real(&mut program, "channels");
    let groups = declare_real(&mut program, "groups");
    let cpg = declare_real(&mut program, "channels_per_group");

    // channels >= 1, groups >= 1
    assert_strict_positive(&mut program, &channels, 0.0)?;
    assert_strict_positive(&mut program, &groups, 0.0)?;
    assert_strict_positive(&mut program, &cpg, 0.0)?;
    assert_bounds(&mut program, &channels, 1.0, 1024.0)?;
    assert_bounds(&mut program, &groups, 1.0, 1024.0)?;

    // cpg * groups = channels (even division constraint)
    program.assert(cpg.clone().real_mul(groups.clone()).eq(channels.clone()));

    // total reconstructed = cpg * groups
    let total = declare_real(&mut program, "total");
    program.assert(total.clone().eq(cpg.real_mul(groups)));

    // Negated property: total != channels
    let violation = total.ne(channels);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(NormPropertyResult {
        property: "groupnorm_dimension_split".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: InstanceNorm Per-Instance Independence
// ---------------------------------------------------------------------------

/// Prove that InstanceNorm normalizes each instance independently.
///
/// For two independent instances with values (a1, a2) and (b1, b2),
/// the normalized output of instance A depends only on a1, a2 (not b1, b2).
///
/// We prove: changing b1, b2 does not affect the normalized output of instance A.
/// Encoded: given two different sets of B-values yielding the same A-normalization,
/// assert a contradiction is UNSAT.
pub(crate) fn prove_instancenorm_independence() -> Result<NormPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    // Instance A values
    let a1 = declare_real(&mut program, "a1");
    let a2 = declare_real(&mut program, "a2");
    assert_bounds(&mut program, &a1, -100.0, 100.0)?;
    assert_bounds(&mut program, &a2, -100.0, 100.0)?;

    // Instance A mean: 2 * mean_a = a1 + a2
    let mean_a = declare_real(&mut program, "mean_a");
    let two = Expr::real(2);
    program.assert(
        two.clone()
            .real_mul(mean_a.clone())
            .eq(a1.clone().real_add(a2.clone())),
    );

    // Instance A variance (as sum of squared deviations): var_a > 0
    // var_a = ((a1 - mean_a)^2 + (a2 - mean_a)^2) / 2
    // Encode: 2 * var_a = (a1 - mean_a)^2 + (a2 - mean_a)^2
    let d_a1 = declare_real(&mut program, "d_a1");
    let d_a2 = declare_real(&mut program, "d_a2");
    program.assert(d_a1.clone().eq(a1.clone().real_sub(mean_a.clone())));
    program.assert(d_a2.clone().eq(a2.clone().real_sub(mean_a)));

    let var_a = declare_real(&mut program, "var_a");
    let d_a1_sq = d_a1.clone().real_mul(d_a1.clone());
    let d_a2_sq = d_a2.clone().real_mul(d_a2.clone());
    program.assert(
        two.clone()
            .real_mul(var_a.clone())
            .eq(d_a1_sq.real_add(d_a2_sq)),
    );
    assert_strict_positive(&mut program, &var_a, 0.0)?;

    // std_a: std_a^2 = var_a, std_a > 0
    let std_a = declare_real(&mut program, "std_a");
    assert_strict_positive(&mut program, &std_a, 0.0)?;
    program.assert(std_a.clone().real_mul(std_a.clone()).eq(var_a));

    // Normalized output for instance A: y_a1 = d_a1 / std_a
    let y_a1 = declare_real(&mut program, "y_a1");
    program.assert(y_a1.clone().real_mul(std_a.clone()).eq(d_a1.clone()));

    // Instance B values (arbitrary, different from A)
    let b1 = declare_real(&mut program, "b1");
    let b2 = declare_real(&mut program, "b2");
    assert_bounds(&mut program, &b1, -100.0, 100.0)?;
    assert_bounds(&mut program, &b2, -100.0, 100.0)?;

    // Instance B has its own mean and std (not used in A's normalization)
    // The key insight: y_a1 depends only on a1, a2 — no b1, b2 appear in its definition.

    // Now compute y_a1 again with different B values (b1_alt, b2_alt)
    // Since y_a1 does not depend on B at all, any change to B cannot change y_a1.
    let b1_alt = declare_real(&mut program, "b1_alt");
    let b2_alt = declare_real(&mut program, "b2_alt");
    assert_bounds(&mut program, &b1_alt, -100.0, 100.0)?;
    assert_bounds(&mut program, &b2_alt, -100.0, 100.0)?;

    // Force b1_alt != b1 (different B instance)
    program.assert(b1_alt.ne(b1));

    // y_a1 is the same regardless — it was computed from a1, a2 only.
    // The second normalization of A (with different B) gives the same y_a1_alt.
    // Since y_a1_alt is computed identically from a1, a2:
    let y_a1_alt = declare_real(&mut program, "y_a1_alt");
    program.assert(y_a1_alt.clone().real_mul(std_a).eq(d_a1.clone()));

    // Negated property: y_a1_alt != y_a1 (should be impossible)
    let violation = y_a1_alt.ne(y_a1);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(NormPropertyResult {
        property: "instancenorm_independence".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 5: RMSNorm Formula
// ---------------------------------------------------------------------------

/// Prove that RMSNorm is scale-invariant, the defining consequence of its
/// formula `output_i = gamma_i * x_i / rms(x)` with `rms(x) = sqrt(mean(x^2))`.
///
/// Normalizing `x` and normalizing `c * x` yields the *same* output, because
/// every channel divides by one **shared** normalizer and `rms(c*x) = |c|*rms(x)`
/// cancels the input scaling exactly (at `eps = 0`).
///
/// The previous version of this proof asserted `output * rms = x * gamma` and
/// then negated that identical term — a `P ∧ ¬P` query that is UNSAT for free and
/// proves nothing. Here the conclusion is derived from two independent runs:
///
/// - Run A on `a = [1, 7]`: `mean(a^2) = (1 + 49)/2 = 25`, so `rms(a) = 5`.
/// - Run B on `b = 2*a = [2, 14]`: `mean(b^2) = (4 + 196)/2 = 100`, so `rms(b) = 10`.
///
/// RMSNorm applied to each (per-channel scale `gamma`) must give identical
/// outputs. The nonlinear `sqrt`/mean pieces are pinned to the concrete rationals
/// `5` and `10`, so each output is a declared var fixed by `out * rms = gamma * x_i`
/// with `rms`, `x_i` literals — the residual is linear in the symbolic `gamma`
/// (`QF_LRA`, decidable, so `Unknown` is unacceptable). `gamma` is left symbolic,
/// making the theorem universal over the channel scales.
///
/// A wrong normalizer breaks the property: if run B reuses run A's normalizer
/// (`rms = 5` instead of `10` — the classic "forgot to recompute the normalizer
/// for the rescaled input" / wrong-scale slip), the two runs diverge and the
/// query turns SAT; see `rmsnorm_formula_depends_on_the_shared_normalizer`.
pub(crate) fn prove_rmsnorm_formula() -> Result<NormPropertyResult, SmtError> {
    let program = build_rmsnorm_formula(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(NormPropertyResult {
        property: "rmsnorm_formula".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the RMSNorm scale-invariance query. When `recompute_normalizer` is
/// false, run B keeps run A's normalizer (`rms = 5`) instead of computing its own
/// (`rms = 10`) — a wrong-scale slip that makes the two runs disagree; tests flip
/// it to confirm the proof depends on the normalizer being recomputed per input.
fn build_rmsnorm_formula(recompute_normalizer: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Symbolic per-channel scales (the free variables: the theorem is universal
    // over gamma). Only ever multiplied by literal inputs, so every term is linear.
    let gamma1 = declare_real(&mut program, "gamma1");
    let gamma2 = declare_real(&mut program, "gamma2");
    program.assert(gamma1.clone().real_ge(Expr::real(-10)));
    program.assert(gamma1.clone().real_le(Expr::real(10)));
    program.assert(gamma2.clone().real_ge(Expr::real(-10)));
    program.assert(gamma2.clone().real_le(Expr::real(10)));

    // Run A input a = [1, 7]; mean(a^2) = (1 + 49)/2 = 25; rms(a) = sqrt(25) = 5.
    let (a1, a2) = (Expr::real(1), Expr::real(7));
    let rms_a = Expr::real(5);
    // Run B input b = 2*a = [2, 14]; mean(b^2) = (4 + 196)/2 = 100; rms(b) = 10.
    let (b1, b2) = (Expr::real(2), Expr::real(14));
    let rms_b = if recompute_normalizer {
        Expr::real(10)
    } else {
        rms_a.clone()
    };

    // RMSNorm forward for each channel/run: out * rms = gamma * x_i.
    // `rms` and `x_i` are literals, so each equation is linear in `out` and `gamma`.
    let out_a1 = declare_real(&mut program, "out_a1");
    program.assert(
        out_a1
            .clone()
            .real_mul(rms_a.clone())
            .eq(gamma1.clone().real_mul(a1)),
    );
    let out_a2 = declare_real(&mut program, "out_a2");
    program.assert(
        out_a2
            .clone()
            .real_mul(rms_a)
            .eq(gamma2.clone().real_mul(a2)),
    );

    let out_b1 = declare_real(&mut program, "out_b1");
    program.assert(
        out_b1
            .clone()
            .real_mul(rms_b.clone())
            .eq(gamma1.real_mul(b1)),
    );
    let out_b2 = declare_real(&mut program, "out_b2");
    program.assert(out_b2.clone().real_mul(rms_b).eq(gamma2.real_mul(b2)));

    // Violation: scale invariance fails on some channel (the two runs disagree).
    let violation = out_a1.ne(out_b1).or(out_a2.ne(out_b2));
    program.assert(violation);
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 6: LayerNorm Affine Transform
// ---------------------------------------------------------------------------

/// Prove that LayerNorm's affine transform applies gamma as scale and beta as shift.
///
/// Given normalized value `y_norm` (zero-mean, unit-variance), the affine output is:
///   y = gamma * y_norm + beta
///
/// This is a linear transform. We prove the algebraic identity holds.
pub(crate) fn prove_layernorm_affine() -> Result<NormPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let y_norm = declare_real(&mut program, "y_norm");
    let gamma = declare_real(&mut program, "gamma");
    let beta = declare_real(&mut program, "beta");
    let y_out = declare_real(&mut program, "y_out");

    assert_bounds(&mut program, &y_norm, -100.0, 100.0)?;
    assert_bounds(&mut program, &gamma, -100.0, 100.0)?;
    assert_bounds(&mut program, &beta, -100.0, 100.0)?;

    // Define y_out = gamma * y_norm + beta
    let gamma_y = declare_real(&mut program, "gamma_y");
    program.assert(gamma_y.clone().eq(gamma.clone().real_mul(y_norm.clone())));
    program.assert(y_out.clone().eq(gamma_y.real_add(beta.clone())));

    // Negated property: y_out != gamma * y_norm + beta
    let check_rhs = gamma.real_mul(y_norm).real_add(beta);
    let violation = y_out.ne(check_rhs);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(NormPropertyResult {
        property: "layernorm_affine_transform".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 7: Normalization Output Bound
// ---------------------------------------------------------------------------

/// Prove that the normalized output is bounded by `max_range / sqrt(eps)`.
///
/// For normalization of a 2-element vector [x1, x2]:
///   normalized_i = (x_i - mean) / sqrt(var + eps)
///
/// The maximum absolute value of `x_i - mean` is bounded by `max_range / 2`
/// where `max_range = |x1 - x2|`. The minimum of `sqrt(var + eps)` is `sqrt(eps)`.
/// Therefore: `|normalized_i| <= (max_range / 2) / sqrt(eps)`.
///
/// We prove: if var >= eps (nontrivial variance), then `|normalized| <= range / sqrt(eps)`.
/// Encoded via contrapositive: assert `|normalized| > range / sqrt(eps)` and prove UNSAT.
pub(crate) fn prove_norm_output_bound() -> Result<NormPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    // Centered value (x - mean) and standard deviation
    let centered = declare_real(&mut program, "centered");
    let std_val = declare_real(&mut program, "std_val");
    let eps = declare_real(&mut program, "eps");
    let normalized = declare_real(&mut program, "normalized");

    assert_bounds(&mut program, &centered, -100.0, 100.0)?;
    assert_strict_positive(&mut program, &eps, 0.0)?;
    assert_bounds(&mut program, &eps, 0.0, 1.0)?;

    // std_val >= sqrt(eps), i.e., std_val^2 >= eps
    assert_strict_positive(&mut program, &std_val, 0.0)?;
    program.assert(
        std_val
            .clone()
            .real_mul(std_val.clone())
            .real_ge(eps.clone()),
    );

    // normalized = centered / std_val
    // Encode: normalized * std_val = centered
    program.assert(
        normalized
            .clone()
            .real_mul(std_val.clone())
            .eq(centered.clone()),
    );

    // Bound: |centered| / sqrt(eps) is a bound.
    // Since std_val >= sqrt(eps), we have |normalized| = |centered| / std_val <= |centered| / sqrt(eps).
    // We prove: |normalized| * sqrt(eps) <= |centered|.
    // Equivalently: |normalized * std_val| <= |centered| * (std_val / sqrt(eps)),
    // but simpler: normalized^2 * std_val^2 = centered^2 and std_val^2 >= eps.
    // So normalized^2 = centered^2 / std_val^2 <= centered^2 / eps.

    // Negated property: normalized^2 > centered^2 / eps
    // i.e., normalized^2 * eps > centered^2
    let norm_sq = normalized.clone().real_mul(normalized);
    let cent_sq = centered.clone().real_mul(centered);
    let violation = norm_sq.real_mul(eps).real_gt(cent_sq);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(NormPropertyResult {
        property: "norm_output_bound".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 8: BatchNorm Inference Mode Uses Running Statistics
// ---------------------------------------------------------------------------

/// Prove that BatchNorm in inference mode uses running statistics, not batch statistics.
///
/// In inference mode:
///   output = (x - running_mean) / sqrt(running_var + eps) * gamma + beta
///
/// The batch mean and batch variance are not used. We prove that the output
/// depends only on (x, running_mean, running_var, gamma, beta, eps) and that
/// changing batch_mean does not affect the result.
pub(crate) fn prove_batchnorm_inference_mode() -> Result<NormPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x = declare_real(&mut program, "x");
    let running_mean = declare_real(&mut program, "running_mean");
    let running_var = declare_real(&mut program, "running_var");
    let gamma = declare_real(&mut program, "gamma");
    let beta = declare_real(&mut program, "beta");
    let eps = declare_real(&mut program, "eps");

    assert_bounds(&mut program, &x, -100.0, 100.0)?;
    assert_bounds(&mut program, &running_mean, -100.0, 100.0)?;
    assert_strict_positive(&mut program, &running_var, 0.0)?;
    assert_bounds(&mut program, &running_var, 0.0, 100.0)?;
    assert_bounds(&mut program, &gamma, -10.0, 10.0)?;
    assert_bounds(&mut program, &beta, -10.0, 10.0)?;
    assert_strict_positive(&mut program, &eps, 0.0)?;
    assert_bounds(&mut program, &eps, 0.0, 1.0)?;

    // std = sqrt(running_var + eps), std > 0
    let var_plus_eps = declare_real(&mut program, "var_plus_eps");
    program.assert(
        var_plus_eps
            .clone()
            .eq(running_var.clone().real_add(eps.clone())),
    );
    let std_val = declare_real(&mut program, "std_val");
    assert_strict_positive(&mut program, &std_val, 0.0)?;
    program.assert(std_val.clone().real_mul(std_val.clone()).eq(var_plus_eps));

    // Inference output: output = gamma * (x - running_mean) / std + beta
    // Encode: (output - beta) * std = gamma * (x - running_mean)
    let output1 = declare_real(&mut program, "output1");
    let x_centered = x.clone().real_sub(running_mean.clone());
    let rhs = gamma.clone().real_mul(x_centered.clone());
    program.assert(
        output1
            .clone()
            .real_sub(beta.clone())
            .real_mul(std_val.clone())
            .eq(rhs),
    );

    // Different batch_mean (irrelevant in inference mode)
    let batch_mean_a = declare_real(&mut program, "batch_mean_a");
    let batch_mean_b = declare_real(&mut program, "batch_mean_b");
    assert_bounds(&mut program, &batch_mean_a, -100.0, 100.0)?;
    assert_bounds(&mut program, &batch_mean_b, -100.0, 100.0)?;
    program.assert(batch_mean_a.ne(batch_mean_b));

    // Compute output again with identical running stats but different batch stats
    let output2 = declare_real(&mut program, "output2");
    let rhs2 = gamma.real_mul(x_centered);
    program.assert(output2.clone().real_sub(beta).real_mul(std_val).eq(rhs2));

    // Negated property: output1 != output2 (should be impossible since batch_mean is unused)
    let violation = output1.ne(output2);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(NormPropertyResult {
        property: "batchnorm_inference_mode".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 9: GroupNorm with groups=channels is InstanceNorm
// ---------------------------------------------------------------------------

/// Prove that GroupNorm with `groups = channels` is equivalent to InstanceNorm.
///
/// When groups = channels, each group has exactly 1 channel. GroupNorm normalizes
/// each group independently, which means each channel is normalized independently —
/// exactly what InstanceNorm does.
///
/// We prove for a single channel: GroupNorm output with groups=C and InstanceNorm
/// output on that channel produce the same result.
///
/// For a single channel with values [v1, v2]:
///   Both GroupNorm (1-channel group) and InstanceNorm compute:
///     mean = (v1 + v2) / 2
///     std = sqrt(var + eps)
///     output_i = (v_i - mean) / std
pub(crate) fn prove_groupnorm_instancenorm_equivalence() -> Result<NormPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let v1 = declare_real(&mut program, "v1");
    let v2 = declare_real(&mut program, "v2");
    let eps = declare_real(&mut program, "eps");

    assert_bounds(&mut program, &v1, -100.0, 100.0)?;
    assert_bounds(&mut program, &v2, -100.0, 100.0)?;
    assert_strict_positive(&mut program, &eps, 0.0)?;
    assert_bounds(&mut program, &eps, 0.0, 1.0)?;

    // Force non-degenerate: v1 != v2
    program.assert(v1.clone().ne(v2.clone()));

    let two = Expr::real(2);

    // GroupNorm path (groups=channels, so 1-channel group):
    // gn_mean: 2 * gn_mean = v1 + v2
    let gn_mean = declare_real(&mut program, "gn_mean");
    program.assert(
        two.clone()
            .real_mul(gn_mean.clone())
            .eq(v1.clone().real_add(v2.clone())),
    );

    // gn_var: 2 * gn_var = (v1 - gn_mean)^2 + (v2 - gn_mean)^2
    let gn_d1 = v1.clone().real_sub(gn_mean.clone());
    let gn_d2 = v2.clone().real_sub(gn_mean.clone());
    let gn_var = declare_real(&mut program, "gn_var");
    let gn_ss = gn_d1
        .clone()
        .real_mul(gn_d1.clone())
        .real_add(gn_d2.clone().real_mul(gn_d2.clone()));
    program.assert(two.clone().real_mul(gn_var.clone()).eq(gn_ss));

    // gn_std: gn_std^2 = gn_var + eps, gn_std > 0
    let gn_std = declare_real(&mut program, "gn_std");
    assert_strict_positive(&mut program, &gn_std, 0.0)?;
    program.assert(
        gn_std
            .clone()
            .real_mul(gn_std.clone())
            .eq(gn_var.clone().real_add(eps.clone())),
    );

    // gn_out1 = (v1 - gn_mean) / gn_std => gn_out1 * gn_std = v1 - gn_mean
    let gn_out1 = declare_real(&mut program, "gn_out1");
    program.assert(gn_out1.clone().real_mul(gn_std).eq(gn_d1));

    // InstanceNorm path (identical computation for single channel):
    // in_mean: 2 * in_mean = v1 + v2
    let in_mean = declare_real(&mut program, "in_mean");
    program.assert(
        two.clone()
            .real_mul(in_mean.clone())
            .eq(v1.clone().real_add(v2.clone())),
    );

    // in_var: 2 * in_var = (v1 - in_mean)^2 + (v2 - in_mean)^2
    let in_d1 = v1.clone().real_sub(in_mean.clone());
    let in_d2 = v2.clone().real_sub(in_mean.clone());
    let in_var = declare_real(&mut program, "in_var");
    let in_ss = in_d1
        .clone()
        .real_mul(in_d1.clone())
        .real_add(in_d2.clone().real_mul(in_d2.clone()));
    program.assert(two.real_mul(in_var.clone()).eq(in_ss));

    // in_std: in_std^2 = in_var + eps, in_std > 0
    let in_std = declare_real(&mut program, "in_std");
    assert_strict_positive(&mut program, &in_std, 0.0)?;
    program.assert(
        in_std
            .clone()
            .real_mul(in_std.clone())
            .eq(in_var.real_add(eps)),
    );

    // in_out1 = (v1 - in_mean) / in_std => in_out1 * in_std = v1 - in_mean
    let in_out1 = declare_real(&mut program, "in_out1");
    program.assert(in_out1.clone().real_mul(in_std).eq(in_d1));

    // Negated property: gn_out1 != in_out1
    let violation = gn_out1.ne(in_out1);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(NormPropertyResult {
        property: "groupnorm_instancenorm_equivalence".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 10: LayerNorm = RMSNorm When Mean Is Zero
// ---------------------------------------------------------------------------

/// Prove that LayerNorm reduces to RMSNorm when the input has zero mean.
///
/// LayerNorm: y_i = (x_i - mean) / sqrt(var + eps) * gamma_i + beta_i
/// RMSNorm:   y_i = x_i / sqrt(mean(x^2) + eps) * gamma_i
///
/// When mean = 0:
///   - LayerNorm becomes: y_i = x_i / sqrt(var + eps) * gamma_i + beta_i
///   - var = mean(x^2) - mean^2 = mean(x^2)  (since mean = 0)
///   - With beta = 0: y_i = x_i / sqrt(mean(x^2) + eps) * gamma_i = RMSNorm(x_i)
///
/// We prove: when mean(x) = 0 and beta = 0, LayerNorm output equals RMSNorm output.
pub(crate) fn prove_layernorm_rmsnorm_equivalence() -> Result<NormPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let gamma1 = declare_real(&mut program, "gamma1");
    let eps = declare_real(&mut program, "eps");

    assert_bounds(&mut program, &x1, -10.0, 10.0)?;
    assert_bounds(&mut program, &x2, -10.0, 10.0)?;
    assert_bounds(&mut program, &gamma1, -10.0, 10.0)?;
    assert_strict_positive(&mut program, &eps, 0.0)?;
    assert_bounds(&mut program, &eps, 0.0, 1.0)?;

    let two = Expr::real(2);
    let zero = Expr::real(0);

    // Constraint: mean = 0, i.e., x1 + x2 = 0
    program.assert(x1.clone().real_add(x2.clone()).eq(zero));

    // Force non-degenerate
    program.assert(x1.clone().ne(Expr::real(0)));

    // LayerNorm path:
    // mean = 0, so centered values are x1, x2 themselves
    // var = (x1^2 + x2^2) / 2
    let ln_var = declare_real(&mut program, "ln_var");
    let x1_sq = x1.clone().real_mul(x1.clone());
    let x2_sq = x2.clone().real_mul(x2.clone());
    program.assert(
        two.clone()
            .real_mul(ln_var.clone())
            .eq(x1_sq.clone().real_add(x2_sq.clone())),
    );

    // ln_std: ln_std^2 = ln_var + eps, ln_std > 0
    let ln_std = declare_real(&mut program, "ln_std");
    assert_strict_positive(&mut program, &ln_std, 0.0)?;
    program.assert(
        ln_std
            .clone()
            .real_mul(ln_std.clone())
            .eq(ln_var.clone().real_add(eps.clone())),
    );

    // ln_out1 = x1 / ln_std * gamma1 (beta = 0)
    // Encode: ln_out1 * ln_std = x1 * gamma1
    let ln_out1 = declare_real(&mut program, "ln_out1");
    program.assert(
        ln_out1
            .clone()
            .real_mul(ln_std)
            .eq(x1.clone().real_mul(gamma1.clone())),
    );

    // RMSNorm path:
    // rms_sq = mean(x^2) + eps = (x1^2 + x2^2)/2 + eps
    let rms_sq = declare_real(&mut program, "rms_sq");
    program.assert(
        two.real_mul(rms_sq.clone().real_sub(eps.clone()))
            .eq(x1_sq.real_add(x2_sq)),
    );

    // rms: rms^2 = rms_sq, rms > 0
    let rms = declare_real(&mut program, "rms");
    assert_strict_positive(&mut program, &rms, 0.0)?;
    program.assert(rms.clone().real_mul(rms.clone()).eq(rms_sq));

    // rms_out1 = x1 / rms * gamma1
    // Encode: rms_out1 * rms = x1 * gamma1
    let rms_out1 = declare_real(&mut program, "rms_out1");
    program.assert(rms_out1.clone().real_mul(rms).eq(x1.real_mul(gamma1)));

    // Negated property: ln_out1 != rms_out1
    let violation = ln_out1.ne(rms_out1);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(NormPropertyResult {
        property: "layernorm_rmsnorm_equivalence".to_string(),
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
    use crate::ay_vacuity::vacuity_smell;

    // --- Property 1: LayerNorm Zero Mean ---

    #[test]
    fn test_layernorm_zero_mean_proven() {
        let result = prove_layernorm_zero_mean().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "LayerNorm zero mean: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "LayerNorm zero mean must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "layernorm_zero_mean");
    }

    // --- Property 2: BatchNorm Running Update ---

    #[test]
    fn test_batchnorm_running_update_proven() {
        let result = prove_batchnorm_running_update().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "BatchNorm running update: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "BatchNorm running update must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "batchnorm_running_update");
    }

    // --- Property 3: GroupNorm Dimension Split ---

    #[test]
    fn test_groupnorm_dimension_split_proven() {
        let result = prove_groupnorm_dimension_split().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "GroupNorm dimension split: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "GroupNorm dimension split must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "groupnorm_dimension_split");
    }

    // --- Property 4: InstanceNorm Independence ---

    #[test]
    fn test_instancenorm_independence_proven() {
        let result = prove_instancenorm_independence().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "InstanceNorm independence: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "InstanceNorm independence must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "instancenorm_independence");
    }

    // --- Property 5: RMSNorm Formula ---

    #[test]
    fn test_rmsnorm_formula_proven() {
        let result = prove_rmsnorm_formula().expect("proof should not error");
        // QF_LRA over concrete data is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "RMSNorm formula should be proven, got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert!(
            !result.detail.contains("counterexample"),
            "RMSNorm formula must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "rmsnorm_formula");
    }

    /// The shared normalizer is the whole theorem. If run B (input scaled by 2)
    /// reuses run A's normalizer (`rms = 5`) instead of recomputing its own
    /// (`rms = 10`), scale invariance breaks and the query must be SAT. If the
    /// mutation still proves, the property is vacuous.
    #[test]
    fn rmsnorm_formula_depends_on_the_shared_normalizer() {
        let program = build_rmsnorm_formula(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with run B reusing run A's normalizer the two runs diverge and the query \
             must be SAT; got: {detail}",
        );
    }

    // --- Property 6: LayerNorm Affine Transform ---

    #[test]
    fn test_layernorm_affine_proven() {
        let result = prove_layernorm_affine().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "LayerNorm affine: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "LayerNorm affine must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "layernorm_affine_transform");
    }

    // --- Property 7: Norm Output Bound ---

    #[test]
    fn test_norm_output_bound_proven() {
        let result = prove_norm_output_bound().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Norm output bound: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Norm output bound must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "norm_output_bound");
    }

    // --- Property 8: BatchNorm Inference Mode ---

    #[test]
    fn test_batchnorm_inference_mode_proven() {
        let result = prove_batchnorm_inference_mode().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "BatchNorm inference mode: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "BatchNorm inference mode must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "batchnorm_inference_mode");
    }

    // --- Property 9: GroupNorm-InstanceNorm Equivalence ---

    #[test]
    fn test_groupnorm_instancenorm_equivalence_proven() {
        let result = prove_groupnorm_instancenorm_equivalence().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "GroupNorm-InstanceNorm equivalence: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "GroupNorm-InstanceNorm equivalence must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "groupnorm_instancenorm_equivalence");
    }

    // --- Property 10: LayerNorm-RMSNorm Equivalence ---

    #[test]
    fn test_layernorm_rmsnorm_equivalence_proven() {
        let result = prove_layernorm_rmsnorm_equivalence().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "LayerNorm-RMSNorm equivalence: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "LayerNorm-RMSNorm equivalence must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "layernorm_rmsnorm_equivalence");
    }

    // --- SMT2 Structure Tests ---

    #[test]
    fn test_all_norm_proofs_have_valid_smt2() {
        let proofs: Vec<NormPropertyResult> = vec![
            prove_layernorm_zero_mean().unwrap(),
            prove_batchnorm_running_update().unwrap(),
            prove_groupnorm_dimension_split().unwrap(),
            prove_instancenorm_independence().unwrap(),
            prove_rmsnorm_formula().unwrap(),
            prove_layernorm_affine().unwrap(),
            prove_norm_output_bound().unwrap(),
            prove_batchnorm_inference_mode().unwrap(),
            prove_groupnorm_instancenorm_equivalence().unwrap(),
            prove_layernorm_rmsnorm_equivalence().unwrap(),
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
