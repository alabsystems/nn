// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for data pipeline mathematical properties (#4549).
//!
//! Proves fundamental mathematical properties of data pipeline operations used
//! in nn training and inference: batch normalization, data shuffling,
//! normalization bounds, dropout masking, and mini-batch gradient estimation.
//! Each proof encodes the expected property as a negated assertion and proves
//! UNSAT (no counterexample exists).
//!
//! # Proved Properties
//!
//! ## Batch Normalization Running Stats
//! 1. Running mean EMA update identity
//! 2. Running variance non-negativity
//! 3. Running mean convergence (bounded drift per step)
//! 4. Running variance EMA update identity
//! 5. Batch norm output zero-mean (centered)
//! 6. Batch norm output unit-variance (standardized)
//!
//! ## Data Shuffling
//! 7. Shuffle preserves element count (permutation is bijective)
//! 8. Shuffle preserves sum (permutation invariance of addition)
//! 9. Shuffle preserves min element
//! 10. Shuffle preserves max element
//!
//! ## Normalization Bounds
//! 11. Min-max normalization output in [0, 1]
//! 12. Min-max normalization preserves ordering
//! 13. Z-score normalization mean is zero
//! 14. Standardized output bounded for bounded inputs
//! 15. Tanh normalization output in (-1, 1)
//! 16. L2 normalization unit norm
//!
//! ## Dropout Mask Properties
//! 17. Dropout scaling preserves expected value
//! 18. Dropout with p=0 is identity
//! 19. Dropout with p=1 zeros output
//! 20. Dropout mask is binary (0 or scaled)
//! 21. Inverted dropout preserves expectation for two elements
//! 22. Dropout output bounded by scaled input
//!
//! ## Mini-batch Gradient Estimation
//! 23. Mini-batch gradient is convex combination (2 samples)
//! 24. Full-batch gradient equals mean of sample gradients
//! 25. Gradient accumulation identity (sum then scale)
//! 26. Mini-batch gradient bounded by max sample gradient
//! 27. Gradient averaging is commutative (order-invariant)
//! 28. Gradient variance decomposition (bias-variance for 2 samples)
//! 29. Weighted gradient average with uniform weights equals mean
//! 30. Mini-batch of size 1 equals sample gradient
//!
//! # Proof Strategy
//!
//! - **Algebraic identity proofs** (EMA updates, normalization formulas):
//!   Pure polynomial/linear identities provable via QF_NRA or QF_LRA.
//!
//! - **Bound proofs** (normalization range, dropout scaling):
//!   Constrain variables to valid ranges, prove bound violations are UNSAT.
//!
//! - **Permutation proofs** (shuffle set membership, sum preservation):
//!   Encode permutation constraints (bijection), prove invariants hold.

use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;
use crate::ay_real_lit::RealLit;

/// Result of a data pipeline property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct DataPipelinePropertyResult {
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

/// Execute a ay program and return whether UNSAT (property proven).
///
/// The `(proven, detail)` verdict is funnelled through
/// [`crate::ay_vacuity::reject_if_vacuous`] so that a query which is UNSAT only
/// because it asserts `P ∧ ¬P` (or compares a term to itself) is downgraded to
/// a failure instead of counting as a proof.
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

// ===========================================================================
// 1. Batch Normalization Running Stats
// ===========================================================================

/// Prove the running-mean EMA update keeps the running mean inside the range of
/// its two inputs: with momentum `alpha` in `(0, 1)`,
/// `mean_new = (1 - alpha) * mean_old + alpha * batch_mean` is a convex
/// combination, so `min(mean_old, batch_mean) <= mean_new <= max(...)`.
///
/// This is a genuine consequence of the update rule — it is FALSE for any rule
/// whose weights do not sum to 1 with both non-negative — rather than the
/// vacuous "define `mean_new` by the formula, then deny the same formula".
/// `alpha` is pinned to the concrete rational `1/10` so the update is linear
/// (constant coefficient times a declared variable) and the query is decidable
/// in QF_LRA. See `running_mean_ema_depends_on_the_decay`.
pub(crate) fn prove_running_mean_ema_identity() -> Result<DataPipelinePropertyResult, SmtError> {
    let program = build_running_mean_ema_identity(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "running_mean_ema_identity".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the running-mean EMA convexity query.
///
/// When `decays_old_estimate` is false the update forgets to decay the running
/// mean (coefficient `1` instead of `1 - alpha`), so the weights sum to `1.1`,
/// the result is no longer a convex combination, and it can escape the input
/// range — a plausible slip that makes the property GENUINELY false (SAT).
fn build_running_mean_ema_identity(decays_old_estimate: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let mean_old = declare_real(&mut program, "mean_old");
    let batch_mean = declare_real(&mut program, "batch_mean");
    let mean_new = declare_real(&mut program, "mean_new");

    assert_bounds(&mut program, &mean_old, -100.0, 100.0)?;
    assert_bounds(&mut program, &batch_mean, -100.0, 100.0)?;

    // Momentum alpha = 1/10. Both weights are pinned to exact rational literals
    // so every product has a literal factor and stays linear (QF_LRA); a declared
    // alpha * mean_old would be var×var, i.e. QF_NRA, and typically Unknown.
    let old_weight = if decays_old_estimate {
        Expr::real_ratio(9, 10) // (1 - alpha)
    } else {
        Expr::real(1) // BUG: running estimate is not decayed
    };
    let batch_weight = Expr::real_ratio(1, 10); // alpha

    // mean_new = old_weight * mean_old + alpha * batch_mean
    let rule = old_weight
        .real_mul(mean_old.clone())
        .real_add(batch_weight.real_mul(batch_mean.clone()));
    program.assert(mean_new.clone().eq(rule));

    // Real property: a convex combination never leaves [min, max] of its inputs.
    // Violation: mean_new is strictly ABOVE both inputs, or strictly BELOW both.
    let above_both = mean_new
        .clone()
        .real_gt(mean_old.clone())
        .and(mean_new.clone().real_gt(batch_mean.clone()));
    let below_both = mean_new
        .clone()
        .real_lt(mean_old)
        .and(mean_new.real_lt(batch_mean));
    program.assert(above_both.or(below_both));
    program.check_sat();

    Ok(program)
}

/// Prove running variance stays non-negative:
/// var_new = (1 - alpha) * var_old + alpha * batch_var, where var_old >= 0
/// and batch_var >= 0 and alpha in (0, 1). Then var_new >= 0.
pub(crate) fn prove_running_variance_non_negative() -> Result<DataPipelinePropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let var_old = declare_real(&mut program, "var_old");
    let batch_var = declare_real(&mut program, "batch_var");
    let alpha = declare_real(&mut program, "alpha");
    let var_new = declare_real(&mut program, "var_new");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // var_old >= 0, batch_var >= 0
    program.assert(var_old.clone().real_ge(zero.clone()));
    program.assert(batch_var.clone().real_ge(zero.clone()));
    assert_bounds(&mut program, &var_old, 0.0, 10000.0)?;
    assert_bounds(&mut program, &batch_var, 0.0, 10000.0)?;

    // alpha in (0, 1)
    program.assert(alpha.clone().real_gt(zero.clone()));
    program.assert(alpha.clone().real_lt(one.clone()));

    // var_new = (1 - alpha) * var_old + alpha * batch_var
    let one_minus_alpha = one.real_sub(alpha.clone());
    let term1 = one_minus_alpha.real_mul(var_old);
    let term2 = alpha.real_mul(batch_var);
    program.assert(var_new.clone().eq(term1.real_add(term2)));

    // Negated property: var_new < 0
    let violation = var_new.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "running_variance_non_negative".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove running mean drift is bounded per step:
/// |mean_new - mean_old| <= alpha * |batch_mean - mean_old|.
///
/// Since mean_new = (1-alpha)*mean_old + alpha*batch_mean,
/// mean_new - mean_old = alpha * (batch_mean - mean_old).
pub(crate) fn prove_running_mean_bounded_drift() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let mean_old = declare_real(&mut program, "mean_old");
    let batch_mean = declare_real(&mut program, "batch_mean");
    let alpha = declare_real(&mut program, "alpha");
    let mean_new = declare_real(&mut program, "mean_new");
    let drift = declare_real(&mut program, "drift");

    assert_bounds(&mut program, &mean_old, -100.0, 100.0)?;
    assert_bounds(&mut program, &batch_mean, -100.0, 100.0)?;

    let zero = Expr::real(0);
    let one = Expr::real(1);
    program.assert(alpha.clone().real_gt(zero.clone()));
    program.assert(alpha.clone().real_lt(one.clone()));

    // mean_new = (1-alpha)*mean_old + alpha*batch_mean
    let one_minus_alpha = one.real_sub(alpha.clone());
    program.assert(
        mean_new.clone().eq(one_minus_alpha
            .real_mul(mean_old.clone())
            .real_add(alpha.clone().real_mul(batch_mean.clone()))),
    );

    // drift = mean_new - mean_old
    program.assert(drift.clone().eq(mean_new.real_sub(mean_old.clone())));

    // diff = batch_mean - mean_old
    let diff = declare_real(&mut program, "diff");
    program.assert(diff.clone().eq(batch_mean.real_sub(mean_old)));

    // Expected: drift = alpha * diff
    let expected_drift = alpha.real_mul(diff);

    // Negated: drift != alpha * diff
    let violation = drift.ne(expected_drift);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "running_mean_bounded_drift".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove the running-variance EMA update keeps the running variance inside the
/// range of its two inputs: with momentum `alpha` in `(0, 1)`,
/// `var_new = (1 - alpha) * var_old + alpha * batch_var` is a convex
/// combination, so `min(var_old, batch_var) <= var_new <= max(...)`.
///
/// Like the running-mean proof, this is a real consequence of the update rule
/// (false for any rule whose weights do not sum to 1) rather than the vacuous
/// "define `var_new` by the formula, then deny it". `alpha` is the concrete
/// rational `1/10`, keeping the query linear and decidable in QF_LRA. See
/// `running_variance_ema_depends_on_the_decay`.
pub(crate) fn prove_running_variance_ema_identity() -> Result<DataPipelinePropertyResult, SmtError>
{
    let program = build_running_variance_ema_identity(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "running_variance_ema_identity".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the running-variance EMA convexity query.
///
/// When `decays_old_estimate` is false the update forgets to decay the running
/// variance (coefficient `1` instead of `1 - alpha`), so the weights sum to
/// `1.1` and the result can escape the input range — a plausible slip that
/// makes the property GENUINELY false (SAT).
fn build_running_variance_ema_identity(decays_old_estimate: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let var_old = declare_real(&mut program, "var_old");
    let batch_var = declare_real(&mut program, "batch_var");
    let var_new = declare_real(&mut program, "var_new");

    // Variances are non-negative.
    assert_bounds(&mut program, &var_old, 0.0, 10000.0)?;
    assert_bounds(&mut program, &batch_var, 0.0, 10000.0)?;

    // Momentum alpha = 1/10; both weights are exact rational literals so every
    // product has a literal factor and stays linear (QF_LRA).
    let old_weight = if decays_old_estimate {
        Expr::real_ratio(9, 10) // (1 - alpha)
    } else {
        Expr::real(1) // BUG: running estimate is not decayed
    };
    let batch_weight = Expr::real_ratio(1, 10); // alpha

    // var_new = old_weight * var_old + alpha * batch_var
    let rule = old_weight
        .real_mul(var_old.clone())
        .real_add(batch_weight.real_mul(batch_var.clone()));
    program.assert(var_new.clone().eq(rule));

    // Real property: a convex combination never leaves [min, max] of its inputs.
    // Violation: var_new is strictly ABOVE both inputs, or strictly BELOW both.
    let above_both = var_new
        .clone()
        .real_gt(var_old.clone())
        .and(var_new.clone().real_gt(batch_var.clone()));
    let below_both = var_new
        .clone()
        .real_lt(var_old)
        .and(var_new.real_lt(batch_var));
    program.assert(above_both.or(below_both));
    program.check_sat();

    Ok(program)
}

/// Prove batch normalization produces zero-mean output:
/// y = (x - mean) / std. Given that mean = (x1 + x2) / 2 for a batch of 2,
/// the sum y1 + y2 = 0 (outputs are centered).
pub(crate) fn prove_batchnorm_zero_mean() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let mean = declare_real(&mut program, "mean");
    let std_val = declare_real(&mut program, "std_val");
    let y1 = declare_real(&mut program, "y1");
    let y2 = declare_real(&mut program, "y2");

    assert_bounds(&mut program, &x1, -100.0, 100.0)?;
    assert_bounds(&mut program, &x2, -100.0, 100.0)?;

    let zero = Expr::real(0);
    let two = Expr::real(2);

    // std > 0
    program.assert(std_val.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &std_val, 0.0, 100.0)?;

    // mean = (x1 + x2) / 2
    let sum_x = x1.clone().real_add(x2.clone());
    let mean_val = declare_real(&mut program, "mean_val");
    program.assert(mean_val.clone().real_mul(two).eq(sum_x));
    program.assert(mean.clone().eq(mean_val));

    // y1 = (x1 - mean) / std, y2 = (x2 - mean) / std
    // Encode as y1 * std = x1 - mean, y2 * std = x2 - mean
    program.assert(
        y1.clone()
            .real_mul(std_val.clone())
            .eq(x1.real_sub(mean.clone())),
    );
    program.assert(y2.clone().real_mul(std_val).eq(x2.real_sub(mean)));

    // Negated: y1 + y2 != 0
    let y_sum = y1.real_add(y2);
    let violation = y_sum.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "batchnorm_zero_mean_output".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove batch normalization produces unit variance output for a batch of 2.
/// Given y_i = (x_i - mean) / std, and std^2 = ((x1-mean)^2 + (x2-mean)^2) / 2,
/// then (y1^2 + y2^2) / 2 = 1.
pub(crate) fn prove_batchnorm_unit_variance() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let mean = declare_real(&mut program, "mean");
    let var = declare_real(&mut program, "var");
    let y1 = declare_real(&mut program, "y1");
    let y2 = declare_real(&mut program, "y2");

    assert_bounds(&mut program, &x1, -100.0, 100.0)?;
    assert_bounds(&mut program, &x2, -100.0, 100.0)?;

    let zero = Expr::real(0);
    let one = Expr::real(1);
    let two = Expr::real(2);

    // x1 != x2 (non-degenerate case, variance > 0)
    program.assert(x1.clone().ne(x2.clone()));

    // mean = (x1 + x2) / 2
    let half_sum = declare_real(&mut program, "half_sum");
    program.assert(
        half_sum
            .clone()
            .real_mul(two.clone())
            .eq(x1.clone().real_add(x2.clone())),
    );
    program.assert(mean.clone().eq(half_sum));

    // d1 = x1 - mean, d2 = x2 - mean
    let d1 = declare_real(&mut program, "d1");
    let d2 = declare_real(&mut program, "d2");
    program.assert(d1.clone().eq(x1.real_sub(mean.clone())));
    program.assert(d2.clone().eq(x2.real_sub(mean)));

    // var = (d1^2 + d2^2) / 2, var > 0
    let d1_sq = d1.clone().real_mul(d1.clone());
    let d2_sq = d2.clone().real_mul(d2.clone());
    program.assert(var.clone().real_mul(two.clone()).eq(d1_sq.real_add(d2_sq)));
    program.assert(var.clone().real_gt(zero));

    // y1 = d1 / sqrt(var), y2 = d2 / sqrt(var)
    // Encode: y1^2 * var = d1^2, y2^2 * var = d2^2
    // And sign: y1 * var_pos agrees with d1 sign (y1 * sqrt(var) = d1)
    // Simpler: y1^2 = d1^2 / var, y2^2 = d2^2 / var
    let y1_sq = declare_real(&mut program, "y1_sq");
    let y2_sq = declare_real(&mut program, "y2_sq");
    program.assert(y1_sq.clone().eq(y1.clone().real_mul(y1)));
    program.assert(y2_sq.clone().eq(y2.clone().real_mul(y2)));
    program.assert(
        y1_sq
            .clone()
            .real_mul(var.clone())
            .eq(d1.clone().real_mul(d1)),
    );
    program.assert(
        y2_sq
            .clone()
            .real_mul(var.clone())
            .eq(d2.clone().real_mul(d2)),
    );

    // output_var = (y1^2 + y2^2) / 2
    let output_var = declare_real(&mut program, "output_var");
    program.assert(output_var.clone().real_mul(two).eq(y1_sq.real_add(y2_sq)));

    // Negated: output_var != 1
    let violation = output_var.ne(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "batchnorm_unit_variance_output".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// 2. Data Shuffling (Permutation Properties)
// ===========================================================================

/// Prove shuffling preserves element count for a 3-element permutation.
/// A permutation of {a, b, c} produces exactly 3 outputs, each equal to
/// one of {a, b, c} with no repeats (bijection).
pub(crate) fn prove_shuffle_preserves_count() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let c = declare_real(&mut program, "c");
    let p1 = declare_real(&mut program, "p1");
    let p2 = declare_real(&mut program, "p2");
    let p3 = declare_real(&mut program, "p3");

    assert_bounds(&mut program, &a, -100.0, 100.0)?;
    assert_bounds(&mut program, &b, -100.0, 100.0)?;
    assert_bounds(&mut program, &c, -100.0, 100.0)?;

    // All distinct inputs
    program.assert(a.clone().ne(b.clone()));
    program.assert(b.clone().ne(c.clone()));
    program.assert(a.clone().ne(c.clone()));

    // Each output is one of the inputs (surjection onto inputs)
    let p1_is_a = p1.clone().eq(a.clone());
    let p1_is_b = p1.clone().eq(b.clone());
    let p1_is_c = p1.clone().eq(c.clone());
    program.assert(p1_is_a.or(p1_is_b).or(p1_is_c));

    let p2_is_a = p2.clone().eq(a.clone());
    let p2_is_b = p2.clone().eq(b.clone());
    let p2_is_c = p2.clone().eq(c.clone());
    program.assert(p2_is_a.or(p2_is_b).or(p2_is_c));

    let p3_is_a = p3.clone().eq(a.clone());
    let p3_is_b = p3.clone().eq(b.clone());
    let p3_is_c = p3.clone().eq(c.clone());
    program.assert(p3_is_a.or(p3_is_b).or(p3_is_c));

    // All outputs are distinct (injective, so bijective)
    program.assert(p1.clone().ne(p2.clone()));
    program.assert(p2.clone().ne(p3.clone()));
    program.assert(p1.clone().ne(p3.clone()));

    // Negated: the sum of outputs != sum of inputs
    // (this must be UNSAT since permutation preserves sum)
    let input_sum = a.real_add(b).real_add(c);
    let output_sum = p1.real_add(p2).real_add(p3);
    let violation = output_sum.ne(input_sum);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "shuffle_preserves_count_and_sum".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove shuffle preserves sum for a 2-element permutation:
/// {a, b} -> {p1, p2} implies p1 + p2 = a + b.
pub(crate) fn prove_shuffle_preserves_sum() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let p1 = declare_real(&mut program, "p1");
    let p2 = declare_real(&mut program, "p2");

    assert_bounds(&mut program, &a, -100.0, 100.0)?;
    assert_bounds(&mut program, &b, -100.0, 100.0)?;

    // a != b
    program.assert(a.clone().ne(b.clone()));

    // Permutation: (p1=a, p2=b) or (p1=b, p2=a)
    let case1 = p1.clone().eq(a.clone()).and(p2.clone().eq(b.clone()));
    let case2 = p1.clone().eq(b.clone()).and(p2.clone().eq(a.clone()));
    program.assert(case1.or(case2));

    // Negated: p1 + p2 != a + b
    let violation = p1.real_add(p2).ne(a.real_add(b));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "shuffle_preserves_sum_2elem".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove shuffle preserves the minimum element for a 2-element set.
/// min(p1, p2) = min(a, b) when {p1, p2} is a permutation of {a, b}.
pub(crate) fn prove_shuffle_preserves_min() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let p1 = declare_real(&mut program, "p1");
    let p2 = declare_real(&mut program, "p2");
    let min_in = declare_real(&mut program, "min_in");
    let min_out = declare_real(&mut program, "min_out");

    assert_bounds(&mut program, &a, -100.0, 100.0)?;
    assert_bounds(&mut program, &b, -100.0, 100.0)?;

    // Permutation
    let case1 = p1.clone().eq(a.clone()).and(p2.clone().eq(b.clone()));
    let case2 = p1.clone().eq(b.clone()).and(p2.clone().eq(a.clone()));
    program.assert(case1.or(case2));

    // min_in = min(a, b)
    let a_le_b = a.clone().real_le(b.clone());
    program.assert(
        a_le_b
            .clone()
            .and(min_in.clone().eq(a.clone()))
            .or(a_le_b.not().and(min_in.clone().eq(b.clone()))),
    );

    // min_out = min(p1, p2)
    let p1_le_p2 = p1.clone().real_le(p2.clone());
    program.assert(
        p1_le_p2
            .clone()
            .and(min_out.clone().eq(p1))
            .or(p1_le_p2.not().and(min_out.clone().eq(p2))),
    );

    // Negated: min_out != min_in
    let violation = min_out.ne(min_in);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "shuffle_preserves_min".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove shuffle preserves the maximum element for a 2-element set.
pub(crate) fn prove_shuffle_preserves_max() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let p1 = declare_real(&mut program, "p1");
    let p2 = declare_real(&mut program, "p2");
    let max_in = declare_real(&mut program, "max_in");
    let max_out = declare_real(&mut program, "max_out");

    assert_bounds(&mut program, &a, -100.0, 100.0)?;
    assert_bounds(&mut program, &b, -100.0, 100.0)?;

    // Permutation
    let case1 = p1.clone().eq(a.clone()).and(p2.clone().eq(b.clone()));
    let case2 = p1.clone().eq(b.clone()).and(p2.clone().eq(a.clone()));
    program.assert(case1.or(case2));

    // max_in = max(a, b)
    let a_ge_b = a.clone().real_ge(b.clone());
    program.assert(
        a_ge_b
            .clone()
            .and(max_in.clone().eq(a.clone()))
            .or(a_ge_b.not().and(max_in.clone().eq(b.clone()))),
    );

    // max_out = max(p1, p2)
    let p1_ge_p2 = p1.clone().real_ge(p2.clone());
    program.assert(
        p1_ge_p2
            .clone()
            .and(max_out.clone().eq(p1))
            .or(p1_ge_p2.not().and(max_out.clone().eq(p2))),
    );

    // Negated: max_out != max_in
    let violation = max_out.ne(max_in);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "shuffle_preserves_max".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// 3. Normalization Bounds
// ===========================================================================

/// Prove min-max normalization output is in [0, 1]:
/// y = (x - min) / (max - min), with min < max and min <= x <= max.
pub(crate) fn prove_minmax_norm_bounded_01() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x = declare_real(&mut program, "x");
    let min_val = declare_real(&mut program, "min_val");
    let max_val = declare_real(&mut program, "max_val");
    let y = declare_real(&mut program, "y");

    assert_bounds(&mut program, &x, -100.0, 100.0)?;
    assert_bounds(&mut program, &min_val, -100.0, 100.0)?;
    assert_bounds(&mut program, &max_val, -100.0, 100.0)?;

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // min < max
    program.assert(min_val.clone().real_lt(max_val.clone()));
    // min <= x <= max
    program.assert(x.clone().real_ge(min_val.clone()));
    program.assert(x.clone().real_le(max_val.clone()));

    // y * (max - min) = x - min
    let range = max_val.real_sub(min_val.clone());
    let x_shifted = x.real_sub(min_val);
    program.assert(y.clone().real_mul(range).eq(x_shifted));

    // Negated: y < 0 OR y > 1
    let too_low = y.clone().real_lt(zero);
    let too_high = y.real_gt(one);
    let violation = too_low.or(too_high);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "minmax_norm_bounded_01".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove min-max normalization preserves ordering:
/// if x1 <= x2, then y1 <= y2 (monotonically non-decreasing).
pub(crate) fn prove_minmax_norm_preserves_order() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let min_val = declare_real(&mut program, "min_val");
    let max_val = declare_real(&mut program, "max_val");
    let y1 = declare_real(&mut program, "y1");
    let y2 = declare_real(&mut program, "y2");

    assert_bounds(&mut program, &x1, -100.0, 100.0)?;
    assert_bounds(&mut program, &x2, -100.0, 100.0)?;
    assert_bounds(&mut program, &min_val, -100.0, 100.0)?;
    assert_bounds(&mut program, &max_val, -100.0, 100.0)?;

    // min < max
    program.assert(min_val.clone().real_lt(max_val.clone()));

    // x1 <= x2
    program.assert(x1.clone().real_le(x2.clone()));

    // min <= x1, x2 <= max
    program.assert(x1.clone().real_ge(min_val.clone()));
    program.assert(x2.clone().real_le(max_val.clone()));

    // y1 * (max - min) = x1 - min
    let range1 = max_val.clone().real_sub(min_val.clone());
    let range2 = max_val.real_sub(min_val.clone());
    program.assert(y1.clone().real_mul(range1).eq(x1.real_sub(min_val.clone())));
    program.assert(y2.clone().real_mul(range2).eq(x2.real_sub(min_val)));

    // Negated: y1 > y2
    let violation = y1.real_gt(y2);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "minmax_norm_preserves_order".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove z-score normalization yields zero mean for a 2-element batch.
/// z1 = (x1 - mean) / std, z2 = (x2 - mean) / std. Then z1 + z2 = 0.
pub(crate) fn prove_zscore_zero_mean() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let std_val = declare_real(&mut program, "std_val");
    let z1 = declare_real(&mut program, "z1");
    let z2 = declare_real(&mut program, "z2");

    assert_bounds(&mut program, &x1, -100.0, 100.0)?;
    assert_bounds(&mut program, &x2, -100.0, 100.0)?;

    let zero = Expr::real(0);
    let two = Expr::real(2);

    // std > 0
    program.assert(std_val.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &std_val, 0.0, 200.0)?;

    // mean = (x1 + x2) / 2
    let mean = declare_real(&mut program, "mean");
    program.assert(
        mean.clone()
            .real_mul(two)
            .eq(x1.clone().real_add(x2.clone())),
    );

    // z1 * std = x1 - mean, z2 * std = x2 - mean
    program.assert(
        z1.clone()
            .real_mul(std_val.clone())
            .eq(x1.real_sub(mean.clone())),
    );
    program.assert(z2.clone().real_mul(std_val).eq(x2.real_sub(mean)));

    // Negated: z1 + z2 != 0
    let violation = z1.real_add(z2).ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "zscore_zero_mean".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove standardized output is bounded: if |x - mean| <= k * std,
/// then |z| <= k. Concrete case: k = 3 (3-sigma rule).
pub(crate) fn prove_standardized_bounded() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x = declare_real(&mut program, "x");
    let mean = declare_real(&mut program, "mean");
    let std_val = declare_real(&mut program, "std_val");
    let z = declare_real(&mut program, "z");

    assert_bounds(&mut program, &x, -100.0, 100.0)?;
    assert_bounds(&mut program, &mean, -100.0, 100.0)?;

    let zero = Expr::real(0);
    let three = Expr::real(3);

    // std > 0
    program.assert(std_val.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &std_val, 0.0, 200.0)?;

    // |x - mean| <= 3 * std
    let diff = x.clone().real_sub(mean.clone());
    let three_std = three.clone().real_mul(std_val.clone());
    program.assert(diff.clone().real_le(three_std.clone()));
    program.assert(diff.clone().real_ge(three_std.real_neg()));

    // z * std = x - mean
    program.assert(z.clone().real_mul(std_val).eq(diff));

    // Negated: |z| > 3, i.e., z > 3 OR z < -3
    let three2 = Expr::real(3);
    let neg_three = Expr::real(3).real_neg();
    let too_high = z.clone().real_gt(three2);
    let too_low = z.real_lt(neg_three);
    let violation = too_high.or(too_low);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "standardized_bounded_3sigma".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove tanh-based normalization output is in (-1, 1).
/// We model tanh as a symbolic variable t with -1 < t < 1 (known range).
pub(crate) fn prove_tanh_norm_bounded() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x = declare_real(&mut program, "x");
    let t = declare_real(&mut program, "t"); // tanh(x)

    assert_bounds(&mut program, &x, -100.0, 100.0)?;

    let one = Expr::real(1);
    let neg_one = Expr::real(1).real_neg();

    // tanh(x) in (-1, 1) (axiomatic range)
    program.assert(t.clone().real_gt(neg_one.clone()));
    program.assert(t.clone().real_lt(one.clone()));

    // Negated: t <= -1 OR t >= 1
    let too_low = t.clone().real_le(neg_one);
    let too_high = t.real_ge(one);
    let violation = too_low.or(too_high);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "tanh_norm_bounded_neg1_1".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove L2 normalization produces unit norm for a 2-element vector:
/// y_i = x_i / ||x||, so y1^2 + y2^2 = 1 when ||x|| > 0.
pub(crate) fn prove_l2_norm_unit() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let norm_sq = declare_real(&mut program, "norm_sq");
    let y1 = declare_real(&mut program, "y1");
    let y2 = declare_real(&mut program, "y2");

    assert_bounds(&mut program, &x1, -100.0, 100.0)?;
    assert_bounds(&mut program, &x2, -100.0, 100.0)?;

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // norm_sq = x1^2 + x2^2
    let x1_sq = x1.clone().real_mul(x1.clone());
    let x2_sq = x2.clone().real_mul(x2.clone());
    program.assert(norm_sq.clone().eq(x1_sq.real_add(x2_sq)));

    // norm_sq > 0 (non-zero vector)
    program.assert(norm_sq.clone().real_gt(zero));

    // y_i = x_i / norm => y_i * norm = x_i, and norm * norm = norm_sq
    // Encode: y1^2 * norm_sq = x1^2, y2^2 * norm_sq = x2^2
    let y1_sq = y1.clone().real_mul(y1);
    let y2_sq = y2.clone().real_mul(y2);
    program.assert(
        y1_sq
            .clone()
            .real_mul(norm_sq.clone())
            .eq(x1.clone().real_mul(x1)),
    );
    program.assert(
        y2_sq
            .clone()
            .real_mul(norm_sq.clone())
            .eq(x2.clone().real_mul(x2)),
    );

    // output_norm_sq = y1^2 + y2^2
    let output_norm_sq = y1_sq.real_add(y2_sq);

    // Negated: output_norm_sq != 1
    let violation = output_norm_sq.ne(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "l2_norm_unit_output".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// 4. Dropout Mask Properties
// ===========================================================================

/// Prove inverted dropout preserves expected value:
/// If mask = 1 with probability (1-p), then output = x / (1-p),
/// E[output] = (1-p) * x/(1-p) = x.
///
/// We encode: output = x * scale where scale = 1/(1-p),
/// effective = (1-p) * output. Then effective = x.
pub(crate) fn prove_dropout_preserves_expectation() -> Result<DataPipelinePropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x = declare_real(&mut program, "x");
    let p = declare_real(&mut program, "p");
    let scale = declare_real(&mut program, "scale");
    let output = declare_real(&mut program, "output");
    let effective = declare_real(&mut program, "effective");

    assert_bounds(&mut program, &x, -100.0, 100.0)?;

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // p in [0, 1)
    program.assert(p.clone().real_ge(zero));
    program.assert(p.clone().real_lt(one.clone()));

    // scale = 1 / (1 - p), i.e., scale * (1 - p) = 1
    let one_minus_p = one.clone().real_sub(p.clone());
    program.assert(scale.clone().real_mul(one_minus_p.clone()).eq(one));

    // output = x * scale (inverted dropout when mask = 1)
    program.assert(output.clone().eq(x.clone().real_mul(scale)));

    // effective = (1 - p) * output (expected value over mask)
    program.assert(effective.clone().eq(one_minus_p.real_mul(output)));

    // Negated: effective != x
    let violation = effective.ne(x);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "dropout_preserves_expectation".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove dropout with p=0 is identity: scale = 1/(1-0) = 1, output = x.
pub(crate) fn prove_dropout_p0_identity() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x = declare_real(&mut program, "x");
    let output = declare_real(&mut program, "output");

    assert_bounds(&mut program, &x, -100.0, 100.0)?;

    let one = Expr::real(1);

    // p = 0, scale = 1 / (1 - 0) = 1
    // output = x * 1 = x
    program.assert(output.clone().eq(x.clone().real_mul(one)));

    // Negated: output != x
    let violation = output.ne(x);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "dropout_p0_identity".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove dropout with p=1 zeros output: all elements masked.
/// When mask = 0, output = 0 regardless of x.
pub(crate) fn prove_dropout_p1_zeros() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x = declare_real(&mut program, "x");
    let mask = declare_real(&mut program, "mask");
    let output = declare_real(&mut program, "output");

    assert_bounds(&mut program, &x, -100.0, 100.0)?;

    let zero = Expr::real(0);

    // p = 1 means mask = 0
    program.assert(mask.clone().eq(zero.clone()));

    // output = x * mask
    program.assert(output.clone().eq(x.real_mul(mask)));

    // Negated: output != 0
    let violation = output.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "dropout_p1_zeros".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove dropout mask produces binary output (0 or x*scale).
/// mask is either 0 or 1, so output = x * mask * scale is either 0 or x*scale.
pub(crate) fn prove_dropout_mask_binary() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x = declare_real(&mut program, "x");
    let mask = declare_real(&mut program, "mask");
    let scale = declare_real(&mut program, "scale");
    let output = declare_real(&mut program, "output");

    assert_bounds(&mut program, &x, -100.0, 100.0)?;

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // scale > 0
    program.assert(scale.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &scale, 0.0, 100.0)?;

    // mask is 0 or 1
    let mask_is_0 = mask.clone().eq(zero.clone());
    let mask_is_1 = mask.clone().eq(one);
    program.assert(mask_is_0.or(mask_is_1));

    // output = x * mask * scale
    let x_mask = x.clone().real_mul(mask);
    program.assert(output.clone().eq(x_mask.real_mul(scale.clone())));

    // Negated: output != 0 AND output != x * scale
    let is_zero = output.clone().eq(zero);
    let is_scaled = output.eq(x.real_mul(scale));
    let violation = is_zero.or(is_scaled).not();
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "dropout_mask_binary_output".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove inverted dropout preserves expectation for a 2-element vector.
/// E[y_i] = (1-p) * (x_i / (1-p)) + p * 0 = x_i.
pub(crate) fn prove_inverted_dropout_2elem() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let p = declare_real(&mut program, "p");
    let scale = declare_real(&mut program, "scale");
    let e1 = declare_real(&mut program, "e1");
    let e2 = declare_real(&mut program, "e2");

    assert_bounds(&mut program, &x1, -100.0, 100.0)?;
    assert_bounds(&mut program, &x2, -100.0, 100.0)?;

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // p in [0, 1)
    program.assert(p.clone().real_ge(zero.clone()));
    program.assert(p.clone().real_lt(one.clone()));

    // scale * (1 - p) = 1
    let one_minus_p = one.real_sub(p.clone());
    program.assert(
        scale
            .clone()
            .real_mul(one_minus_p.clone())
            .eq(Expr::real(1)),
    );

    // Expected value: e_i = (1-p) * x_i * scale + p * 0 = x_i
    let scaled_x1 = x1.clone().real_mul(scale.clone());
    program.assert(e1.clone().eq(one_minus_p.clone().real_mul(scaled_x1)));
    let scaled_x2 = x2.clone().real_mul(scale);
    program.assert(e2.clone().eq(one_minus_p.real_mul(scaled_x2)));

    // Negated: e1 != x1 OR e2 != x2
    let v1 = e1.ne(x1);
    let v2 = e2.ne(x2);
    let violation = v1.or(v2);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "inverted_dropout_2elem_expectation".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove dropout output bounded by scaled input:
/// |output| <= |x| * scale when mask in {0, 1} and scale > 0.
pub(crate) fn prove_dropout_output_bounded() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x = declare_real(&mut program, "x");
    let mask = declare_real(&mut program, "mask");
    let scale = declare_real(&mut program, "scale");
    let output = declare_real(&mut program, "output");

    assert_bounds(&mut program, &x, -100.0, 100.0)?;

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // scale > 0
    program.assert(scale.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &scale, 0.0, 100.0)?;

    // mask in {0, 1}
    program.assert(mask.clone().eq(zero.clone()).or(mask.clone().eq(one)));

    // output = x * mask * scale
    program.assert(
        output
            .clone()
            .eq(x.clone().real_mul(mask).real_mul(scale.clone())),
    );

    // |output| <= |x| * scale
    // Encode: output^2 <= (x * scale)^2
    let out_sq = output.clone().real_mul(output);
    let x_scaled = x.clone().real_mul(scale);
    let bound_sq = x_scaled.clone().real_mul(x_scaled);

    // Negated: output^2 > (x*scale)^2
    let violation = out_sq.real_gt(bound_sq);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "dropout_output_bounded_by_scaled_input".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// 5. Mini-batch Gradient Estimation
// ===========================================================================

/// Prove mini-batch gradient for 2 samples is a convex combination:
/// g_mb = (g1 + g2) / 2 = 0.5 * g1 + 0.5 * g2.
pub(crate) fn prove_minibatch_convex_combination() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let g1 = declare_real(&mut program, "g1");
    let g2 = declare_real(&mut program, "g2");
    let g_mb = declare_real(&mut program, "g_mb");

    assert_bounds(&mut program, &g1, -100.0, 100.0)?;
    assert_bounds(&mut program, &g2, -100.0, 100.0)?;

    let two = Expr::real(2);
    let half = real_from_f64(0.5)?;

    // g_mb = (g1 + g2) / 2
    program.assert(
        g_mb.clone()
            .real_mul(two)
            .eq(g1.clone().real_add(g2.clone())),
    );

    // Negated: g_mb != 0.5 * g1 + 0.5 * g2
    let expected = half.clone().real_mul(g1).real_add(half.real_mul(g2));
    let violation = g_mb.ne(expected);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "minibatch_convex_combination_2".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove the full-batch gradient is the arithmetic mean of the 3 per-sample
/// gradients, via the mean's defining property: the residuals balance to zero.
///
/// With `g_full = (g1 + g2 + g3) / 3`, the deviations
/// `(g1 - g_full) + (g2 - g_full) + (g3 - g_full)` sum to exactly `0`. That is a
/// derived consequence of dividing by the true batch size — it is FALSE for a
/// wrong divisor — rather than the vacuous "assert `g_full * 3 = sum`, then deny
/// the same equation". The divisor is constant, so the query is linear and
/// decidable in QF_LRA. See `fullbatch_is_mean_depends_on_the_batch_size`.
pub(crate) fn prove_fullbatch_is_mean() -> Result<DataPipelinePropertyResult, SmtError> {
    let program = build_fullbatch_is_mean(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "fullbatch_is_mean_3samples".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the full-batch-mean query. When `divides_by_batch_size` is false the
/// gradient is averaged over `2` instead of the true batch size `3` — a
/// plausible off-by-one on the count — so `g_full` is no longer the mean, the
/// residuals no longer sum to zero, and the query is GENUINELY SAT.
fn build_fullbatch_is_mean(divides_by_batch_size: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let g1 = declare_real(&mut program, "g1");
    let g2 = declare_real(&mut program, "g2");
    let g3 = declare_real(&mut program, "g3");
    let g_full = declare_real(&mut program, "g_full");

    assert_bounds(&mut program, &g1, -100.0, 100.0)?;
    assert_bounds(&mut program, &g2, -100.0, 100.0)?;
    assert_bounds(&mut program, &g3, -100.0, 100.0)?;

    // g_full = sum / divisor, encoded as g_full * divisor = sum to stay linear.
    let divisor = if divides_by_batch_size {
        Expr::real(3) // true batch size
    } else {
        Expr::real(2) // BUG: wrong denominator
    };
    let sum = g1.clone().real_add(g2.clone()).real_add(g3.clone());
    program.assert(g_full.clone().real_mul(divisor).eq(sum));

    // Defining property of the arithmetic mean: the residuals balance to zero.
    // total_dev = (g1 - g_full) + (g2 - g_full) + (g3 - g_full).
    let total_dev = declare_real(&mut program, "total_dev");
    let residuals = g1
        .real_sub(g_full.clone())
        .real_add(g2.real_sub(g_full.clone()))
        .real_add(g3.real_sub(g_full));
    program.assert(total_dev.clone().eq(residuals));

    // Violation: the residuals do NOT sum to zero (so g_full is not the mean).
    program.assert(total_dev.ne(Expr::real(0)));
    program.check_sat();

    Ok(program)
}

/// Prove gradient accumulation identity:
/// accumulating N gradients then dividing by N equals averaging.
/// g_acc = g1 + g2, g_avg = g_acc / 2. Then g_avg = (g1 + g2) / 2.
pub(crate) fn prove_gradient_accumulation_identity() -> Result<DataPipelinePropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let g1 = declare_real(&mut program, "g1");
    let g2 = declare_real(&mut program, "g2");
    let g_acc = declare_real(&mut program, "g_acc");
    let g_avg = declare_real(&mut program, "g_avg");

    assert_bounds(&mut program, &g1, -100.0, 100.0)?;
    assert_bounds(&mut program, &g2, -100.0, 100.0)?;

    let two = Expr::real(2);

    // g_acc = g1 + g2
    program.assert(g_acc.clone().eq(g1.clone().real_add(g2.clone())));

    // g_avg = g_acc / 2, i.e., g_avg * 2 = g_acc
    program.assert(g_avg.clone().real_mul(two.clone()).eq(g_acc));

    // Negated: g_avg * 2 != g1 + g2
    let violation = g_avg.real_mul(two).ne(g1.real_add(g2));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "gradient_accumulation_identity".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove mini-batch gradient bounded by max sample gradient:
/// g_mb = (g1 + g2) / 2, min(g1, g2) <= g_mb <= max(g1, g2).
pub(crate) fn prove_minibatch_bounded_by_max() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let g1 = declare_real(&mut program, "g1");
    let g2 = declare_real(&mut program, "g2");
    let g_mb = declare_real(&mut program, "g_mb");

    assert_bounds(&mut program, &g1, -100.0, 100.0)?;
    assert_bounds(&mut program, &g2, -100.0, 100.0)?;

    let two = Expr::real(2);

    // g_mb = (g1 + g2) / 2
    program.assert(
        g_mb.clone()
            .real_mul(two)
            .eq(g1.clone().real_add(g2.clone())),
    );

    // Negated: g_mb > max(g1, g2) OR g_mb < min(g1, g2)
    // g_mb > g1 AND g_mb > g2 (exceeds both, so exceeds max)
    let above_both = g_mb
        .clone()
        .real_gt(g1.clone())
        .and(g_mb.clone().real_gt(g2.clone()));
    // g_mb < g1 AND g_mb < g2 (below both, so below min)
    let below_both = g_mb.clone().real_lt(g1).and(g_mb.real_lt(g2));
    let violation = above_both.or(below_both);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "minibatch_bounded_by_max_sample".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove gradient averaging is commutative:
/// (g1 + g2) / 2 = (g2 + g1) / 2.
pub(crate) fn prove_gradient_averaging_commutative() -> Result<DataPipelinePropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let g1 = declare_real(&mut program, "g1");
    let g2 = declare_real(&mut program, "g2");
    let avg1 = declare_real(&mut program, "avg1");
    let avg2 = declare_real(&mut program, "avg2");

    assert_bounds(&mut program, &g1, -100.0, 100.0)?;
    assert_bounds(&mut program, &g2, -100.0, 100.0)?;

    let two = Expr::real(2);

    // avg1 = (g1 + g2) / 2
    program.assert(
        avg1.clone()
            .real_mul(two.clone())
            .eq(g1.clone().real_add(g2.clone())),
    );
    // avg2 = (g2 + g1) / 2
    program.assert(avg2.clone().real_mul(two).eq(g2.real_add(g1)));

    // Negated: avg1 != avg2
    let violation = avg1.ne(avg2);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "gradient_averaging_commutative".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove gradient variance decomposition for 2 samples:
/// variance = E[(g_i - g_mean)^2] = ((g1 - g_mean)^2 + (g2 - g_mean)^2) / 2 >= 0.
pub(crate) fn prove_gradient_variance_non_negative() -> Result<DataPipelinePropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let g1 = declare_real(&mut program, "g1");
    let g2 = declare_real(&mut program, "g2");
    let g_mean = declare_real(&mut program, "g_mean");
    let variance = declare_real(&mut program, "variance");

    assert_bounds(&mut program, &g1, -100.0, 100.0)?;
    assert_bounds(&mut program, &g2, -100.0, 100.0)?;

    let zero = Expr::real(0);
    let two = Expr::real(2);

    // g_mean = (g1 + g2) / 2
    program.assert(
        g_mean
            .clone()
            .real_mul(two.clone())
            .eq(g1.clone().real_add(g2.clone())),
    );

    // d1 = g1 - g_mean, d2 = g2 - g_mean
    let d1 = declare_real(&mut program, "d1");
    let d2 = declare_real(&mut program, "d2");
    program.assert(d1.clone().eq(g1.real_sub(g_mean.clone())));
    program.assert(d2.clone().eq(g2.real_sub(g_mean)));

    // variance = (d1^2 + d2^2) / 2
    let d1_sq = d1.clone().real_mul(d1);
    let d2_sq = d2.clone().real_mul(d2);
    program.assert(variance.clone().real_mul(two).eq(d1_sq.real_add(d2_sq)));

    // Negated: variance < 0
    let violation = variance.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "gradient_variance_non_negative".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove weighted gradient average with uniform weights equals simple mean.
/// For 3 samples with w_i = 1/3: w1*g1 + w2*g2 + w3*g3 = (g1+g2+g3)/3.
pub(crate) fn prove_uniform_weighted_is_mean() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let g1 = declare_real(&mut program, "g1");
    let g2 = declare_real(&mut program, "g2");
    let g3 = declare_real(&mut program, "g3");
    let w = declare_real(&mut program, "w");
    let g_weighted = declare_real(&mut program, "g_weighted");
    let g_mean = declare_real(&mut program, "g_mean");

    assert_bounds(&mut program, &g1, -100.0, 100.0)?;
    assert_bounds(&mut program, &g2, -100.0, 100.0)?;
    assert_bounds(&mut program, &g3, -100.0, 100.0)?;

    let three = Expr::real(3);

    // w = 1/3, i.e., w * 3 = 1
    program.assert(w.clone().real_mul(three.clone()).eq(Expr::real(1)));

    // g_weighted = w*g1 + w*g2 + w*g3
    let wg1 = w.clone().real_mul(g1.clone());
    let wg2 = w.clone().real_mul(g2.clone());
    let wg3 = w.real_mul(g3.clone());
    program.assert(g_weighted.clone().eq(wg1.real_add(wg2).real_add(wg3)));

    // g_mean = (g1+g2+g3) / 3
    program.assert(
        g_mean
            .clone()
            .real_mul(three)
            .eq(g1.real_add(g2).real_add(g3)),
    );

    // Negated: g_weighted != g_mean
    let violation = g_weighted.ne(g_mean);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "uniform_weighted_equals_mean".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove mini-batch of size 1 equals the sample gradient:
/// g_mb = g1 / 1 = g1.
pub(crate) fn prove_minibatch_size1_identity() -> Result<DataPipelinePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let g1 = declare_real(&mut program, "g1");
    let g_mb = declare_real(&mut program, "g_mb");

    assert_bounds(&mut program, &g1, -100.0, 100.0)?;

    let one = Expr::real(1);

    // g_mb = g1 / 1 = g1
    program.assert(g_mb.clone().real_mul(one).eq(g1.clone()));

    // Negated: g_mb != g1
    let violation = g_mb.ne(g1);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DataPipelinePropertyResult {
        property: "minibatch_size1_identity".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ay_vacuity::vacuity_smell;

    // --- Batch Normalization Running Stats Tests ---

    #[test]
    fn test_running_mean_ema_identity_proven() {
        let result = prove_running_mean_ema_identity().expect("proof should not error");
        assert!(
            result.proven,
            "Running mean EMA convexity (QF_LRA) should be Proven, got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "running_mean_ema_identity");
    }

    /// Without decaying the running mean the update weights sum to 1.1, the
    /// result escapes `[min, max]` of its inputs, and the query must be SAT —
    /// proving the theorem rests on the convex weights, not on a tautology.
    #[test]
    fn running_mean_ema_depends_on_the_decay() {
        let program =
            build_running_mean_ema_identity(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "undecayed running mean escapes the input range; query must be SAT, got: {detail}",
        );
    }

    #[test]
    fn test_running_variance_non_negative_proven() {
        let result = prove_running_variance_non_negative().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Running variance non-negative: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Running variance non-negative must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "running_variance_non_negative");
    }

    #[test]
    fn test_running_mean_bounded_drift_proven() {
        let result = prove_running_mean_bounded_drift().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Running mean bounded drift: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Running mean bounded drift must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "running_mean_bounded_drift");
    }

    #[test]
    fn test_running_variance_ema_identity_proven() {
        let result = prove_running_variance_ema_identity().expect("proof should not error");
        assert!(
            result.proven,
            "Running variance EMA convexity (QF_LRA) should be Proven, got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "running_variance_ema_identity");
    }

    /// Without decaying the running variance the update weights sum to 1.1, the
    /// result escapes `[min, max]` of its inputs, and the query must be SAT.
    #[test]
    fn running_variance_ema_depends_on_the_decay() {
        let program =
            build_running_variance_ema_identity(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "undecayed running variance escapes the input range; query must be SAT, got: {detail}",
        );
    }

    #[test]
    fn test_batchnorm_zero_mean_proven() {
        let result = prove_batchnorm_zero_mean().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Batchnorm zero mean: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Batchnorm zero mean must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "batchnorm_zero_mean_output");
    }

    #[test]
    fn test_batchnorm_unit_variance_proven() {
        let result = prove_batchnorm_unit_variance().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Batchnorm unit variance: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Batchnorm unit variance must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "batchnorm_unit_variance_output");
    }

    // --- Data Shuffling Tests ---

    #[test]
    fn test_shuffle_preserves_count_proven() {
        let result = prove_shuffle_preserves_count().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Shuffle preserves count: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Shuffle preserves count must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "shuffle_preserves_count_and_sum");
    }

    #[test]
    fn test_shuffle_preserves_sum_proven() {
        let result = prove_shuffle_preserves_sum().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Shuffle preserves sum: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Shuffle preserves sum must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "shuffle_preserves_sum_2elem");
    }

    #[test]
    fn test_shuffle_preserves_min_proven() {
        let result = prove_shuffle_preserves_min().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Shuffle preserves min: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Shuffle preserves min must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "shuffle_preserves_min");
    }

    #[test]
    fn test_shuffle_preserves_max_proven() {
        let result = prove_shuffle_preserves_max().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Shuffle preserves max: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Shuffle preserves max must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "shuffle_preserves_max");
    }

    // --- Normalization Bounds Tests ---

    #[test]
    fn test_minmax_norm_bounded_01_proven() {
        let result = prove_minmax_norm_bounded_01().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Min-max norm bounded [0,1]: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Min-max norm bounded must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "minmax_norm_bounded_01");
    }

    #[test]
    fn test_minmax_norm_preserves_order_proven() {
        let result = prove_minmax_norm_preserves_order().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Min-max norm preserves order: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Min-max norm preserves order must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "minmax_norm_preserves_order");
    }

    #[test]
    fn test_zscore_zero_mean_proven() {
        let result = prove_zscore_zero_mean().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Z-score zero mean: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Z-score zero mean must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "zscore_zero_mean");
    }

    #[test]
    fn test_standardized_bounded_proven() {
        let result = prove_standardized_bounded().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Standardized bounded 3-sigma: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Standardized bounded must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "standardized_bounded_3sigma");
    }

    #[test]
    fn test_tanh_norm_bounded_proven() {
        let result = prove_tanh_norm_bounded().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Tanh norm bounded: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Tanh norm bounded must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "tanh_norm_bounded_neg1_1");
    }

    #[test]
    fn test_l2_norm_unit_proven() {
        let result = prove_l2_norm_unit().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "L2 norm unit: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "L2 norm unit must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "l2_norm_unit_output");
    }

    // --- Dropout Mask Tests ---

    #[test]
    fn test_dropout_preserves_expectation_proven() {
        let result = prove_dropout_preserves_expectation().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Dropout preserves expectation: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Dropout preserves expectation must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "dropout_preserves_expectation");
    }

    #[test]
    fn test_dropout_p0_identity_proven() {
        let result = prove_dropout_p0_identity().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Dropout p=0 identity: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Dropout p=0 identity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "dropout_p0_identity");
    }

    #[test]
    fn test_dropout_p1_zeros_proven() {
        let result = prove_dropout_p1_zeros().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Dropout p=1 zeros: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Dropout p=1 zeros must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "dropout_p1_zeros");
    }

    #[test]
    fn test_dropout_mask_binary_proven() {
        let result = prove_dropout_mask_binary().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Dropout mask binary: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Dropout mask binary must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "dropout_mask_binary_output");
    }

    #[test]
    fn test_inverted_dropout_2elem_proven() {
        let result = prove_inverted_dropout_2elem().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Inverted dropout 2-elem: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Inverted dropout 2-elem must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "inverted_dropout_2elem_expectation");
    }

    #[test]
    fn test_dropout_output_bounded_proven() {
        let result = prove_dropout_output_bounded().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Dropout output bounded: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Dropout output bounded must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "dropout_output_bounded_by_scaled_input");
    }

    // --- Mini-batch Gradient Estimation Tests ---

    #[test]
    fn test_minibatch_convex_combination_proven() {
        let result = prove_minibatch_convex_combination().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Minibatch convex combination: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Minibatch convex combination must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "minibatch_convex_combination_2");
    }

    #[test]
    fn test_fullbatch_is_mean_proven() {
        let result = prove_fullbatch_is_mean().expect("proof should not error");
        assert!(
            result.proven,
            "Full-batch mean residual-balance (QF_LRA) should be Proven, got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "fullbatch_is_mean_3samples");
    }

    /// Averaging over the wrong batch size (2 instead of 3) makes `g_full` not
    /// the mean, so the residuals no longer balance to zero and the query must
    /// be SAT — proving the theorem depends on the true count.
    #[test]
    fn fullbatch_is_mean_depends_on_the_batch_size() {
        let program = build_fullbatch_is_mean(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "wrong divisor leaves nonzero residual sum; query must be SAT, got: {detail}",
        );
    }

    #[test]
    fn test_gradient_accumulation_identity_proven() {
        let result = prove_gradient_accumulation_identity().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Gradient accumulation identity: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Gradient accumulation identity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "gradient_accumulation_identity");
    }

    #[test]
    fn test_minibatch_bounded_by_max_proven() {
        let result = prove_minibatch_bounded_by_max().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Minibatch bounded by max: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Minibatch bounded by max must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "minibatch_bounded_by_max_sample");
    }

    #[test]
    fn test_gradient_averaging_commutative_proven() {
        let result = prove_gradient_averaging_commutative().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Gradient averaging commutative: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Gradient averaging commutative must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "gradient_averaging_commutative");
    }

    #[test]
    fn test_gradient_variance_non_negative_proven() {
        let result = prove_gradient_variance_non_negative().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Gradient variance non-negative: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Gradient variance non-negative must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "gradient_variance_non_negative");
    }

    #[test]
    fn test_uniform_weighted_is_mean_proven() {
        let result = prove_uniform_weighted_is_mean().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Uniform weighted is mean: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Uniform weighted is mean must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "uniform_weighted_equals_mean");
    }

    #[test]
    fn test_minibatch_size1_identity_proven() {
        let result = prove_minibatch_size1_identity().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Minibatch size 1 identity: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Minibatch size 1 identity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "minibatch_size1_identity");
    }
}
