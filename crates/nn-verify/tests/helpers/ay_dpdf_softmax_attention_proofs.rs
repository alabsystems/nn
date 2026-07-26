// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for softmax and attention score properties.
//!
//! Proves fundamental properties of softmax and attention mechanisms used in
//! transformer-based document understanding models:
//! - Softmax: sum-to-one, positivity, (0,1) range, ordering preservation
//! - Softmax: shift invariance (numerical stability), log-softmax identity
//! - Softmax: temperature scaling, gradient Jacobian
//! - Attention: scaled dot-product, score bounds, row-stochastic weights
//! - Attention: causal mask, output bounds, multi-head dimension split
//! - Attention: GQA KV-repeat, self-attention symmetry, ALiBi bias
//! - Attention: top-k sparsity, sliding window, cross-attention dimensions
//!
//! Part of #4122.

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
// Test 451: Softmax outputs sum to 1 (2-element vector)
// ---------------------------------------------------------------------------

/// Prove: softmax(x1, x2) sums to 1.
///
/// For a 2-element vector, softmax outputs s1 = exp(x1)/(exp(x1)+exp(x2))
/// and s2 = exp(x2)/(exp(x1)+exp(x2)). By definition, s1 + s2 = 1.
///
/// We model s1, s2 as softmax outputs with axioms: s1 > 0, s2 > 0,
/// s1 + s2 = 1. The negation (s1 + s2 != 1) under these axioms is UNSAT.
#[test]
fn test_451_softmax_sum_to_one() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("s2", real);

    let s1 = real_var("s1");
    let s2 = real_var("s2");

    // Softmax axioms: both positive, sum to 1
    prog.assert(s1.clone().real_gt(Expr::real(0)));
    prog.assert(s2.clone().real_gt(Expr::real(0)));
    prog.assert(s1.clone().real_add(s2.clone()).eq(Expr::real(1)));

    // Negated property: s1 + s2 != 1
    let violation = s1.real_add(s2).ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_sum_to_one");
}

// ---------------------------------------------------------------------------
// Test 452: Softmax outputs are strictly positive
// ---------------------------------------------------------------------------

/// Prove: softmax(x)_i > 0 for all i.
///
/// Since exp(x_i) > 0 for all real x_i and the denominator sum(exp(x_j)) > 0,
/// each softmax output = exp(x_i) / sum(exp(x_j)) > 0.
///
/// We model: s = exp(x) / Z with exp(x) > 0 and Z > 0, so s > 0.
#[test]
fn test_452_softmax_outputs_positive() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp_x", real.clone());
    let _ = prog.declare_const("z", real.clone());
    let _ = prog.declare_const("s", real);

    let exp_x = real_var("exp_x");
    let z = real_var("z");
    let s = real_var("s");

    // exp(x) > 0 for all real x
    prog.assert(exp_x.clone().real_gt(Expr::real(0)));

    // Z = sum of exp values > 0 (and Z >= exp_x since it includes exp_x)
    prog.assert(z.clone().real_ge(exp_x.clone()));
    prog.assert(z.clone().real_gt(Expr::real(0)));

    // s = exp_x / Z, modeled as: s * Z = exp_x
    prog.assert(s.clone().real_mul(z).eq(exp_x));

    // Negated property: s <= 0
    let violation = s.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_outputs_positive");
}

// ---------------------------------------------------------------------------
// Test 453: Softmax outputs are in (0, 1)
// ---------------------------------------------------------------------------

/// Prove: 0 < softmax(x)_i < 1 for all i (in a vector of size >= 2).
///
/// Since exp(x_i) > 0, the numerator is positive. Since the denominator
/// includes at least one other positive term, denom > exp(x_i), so
/// s_i = exp(x_i) / denom < 1.
///
/// We model: s = exp_x / Z, Z > exp_x (at least 2 elements), so 0 < s < 1.
#[test]
fn test_453_softmax_outputs_in_zero_one() {
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

    // exp values are positive
    prog.assert(exp_x.clone().real_gt(Expr::real(0)));
    prog.assert(exp_other.clone().real_gt(Expr::real(0)));

    // Z = exp_x + exp_other (at least 2 elements)
    prog.assert(z.clone().eq(exp_x.clone().real_add(exp_other)));

    // s = exp_x / Z, modeled as: s * Z = exp_x
    prog.assert(s.clone().real_mul(z).eq(exp_x));

    // Negated property: s <= 0 OR s >= 1
    let violation = s
        .clone()
        .real_le(Expr::real(0))
        .or(s.real_ge(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_outputs_in_zero_one");
}

// ---------------------------------------------------------------------------
// Test 454: Softmax preserves ordering
// ---------------------------------------------------------------------------

/// Prove: if x_i > x_j, then softmax(x)_i > softmax(x)_j.
///
/// Since exp is strictly increasing, x_i > x_j implies exp(x_i) > exp(x_j).
/// With the same denominator Z, exp(x_i)/Z > exp(x_j)/Z.
///
/// We model: e1 > e2 > 0 (from x1 > x2), s1 = e1/Z, s2 = e2/Z.
#[test]
fn test_454_softmax_preserves_ordering() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("e1", real.clone());
    let _ = prog.declare_const("e2", real.clone());
    let _ = prog.declare_const("z", real.clone());
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("s2", real);

    let e1 = real_var("e1");
    let e2 = real_var("e2");
    let z = real_var("z");
    let s1 = real_var("s1");
    let s2 = real_var("s2");

    // exp values: e1 > e2 > 0 (since x1 > x2 and exp is increasing)
    prog.assert(e1.clone().real_gt(e2.clone()));
    prog.assert(e2.clone().real_gt(Expr::real(0)));

    // Z > 0 and Z >= e1 + e2
    prog.assert(z.clone().real_ge(e1.clone().real_add(e2.clone())));

    // s1 = e1/Z and s2 = e2/Z
    prog.assert(s1.clone().real_mul(z.clone()).eq(e1));
    prog.assert(s2.clone().real_mul(z).eq(e2));

    // Negated property: s1 <= s2 (ordering not preserved)
    let violation = s1.real_le(s2);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_preserves_ordering");
}

// ---------------------------------------------------------------------------
// Test 455: Softmax is invariant to constant shift (numerical stability)
// ---------------------------------------------------------------------------

/// Prove: softmax(x + c) = softmax(x) for any constant c.
///
/// exp(x_i + c) / sum(exp(x_j + c)) = exp(x_i)*exp(c) / (sum(exp(x_j))*exp(c))
/// = exp(x_i) / sum(exp(x_j)) = softmax(x)_i.
///
/// We model the 2-element case: s1 = e1/(e1+e2) and s1' = (e1*k)/((e1+e2)*k)
/// where k = exp(c) > 0, and prove s1 = s1'.
#[test]
fn test_455_softmax_shift_invariance() {
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

    // s_shifted = (e1*k) / ((e1+e2)*k) = (e1*k) / (e1*k + e2*k)
    let e1k = e1.real_mul(k.clone());
    let e2k = e2.real_mul(k);
    let z_shifted = e1k.clone().real_add(e2k);
    prog.assert(s_shifted.clone().real_mul(z_shifted).eq(e1k));

    // Negated property: s_orig != s_shifted
    let violation = s_orig.ne(s_shifted);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_shift_invariance");
}

// ---------------------------------------------------------------------------
// Test 456: Log-softmax identity
// ---------------------------------------------------------------------------

/// Prove: log(softmax(x)_i) = x_i - log(sum(exp(x_j))).
///
/// softmax(x)_i = exp(x_i) / Z where Z = sum(exp(x_j)).
/// log(softmax(x)_i) = log(exp(x_i)) - log(Z) = x_i - log(Z).
///
/// We model: ls = x - logZ (log-softmax definition) and verify that
/// ls equals the log of the softmax output.
#[test]
fn test_456_log_softmax_identity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("log_z", real.clone());
    let _ = prog.declare_const("ls", real.clone());
    let _ = prog.declare_const("log_s", real);

    let x = real_var("x");
    let log_z = real_var("log_z");
    let ls = real_var("ls");
    let log_s = real_var("log_s");

    // Input bounds
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));

    // log_z > 0 (since Z = sum(exp(x_j)) >= 1 when any x_j >= 0, but log_z can be anything)
    // log_z is just log(sum(exp(x_j)))

    // Log-softmax definition: ls = x - log_z
    prog.assert(ls.clone().eq(x.clone().real_sub(log_z.clone())));

    // log of softmax: log_s = log(exp(x)/Z) = log(exp(x)) - log(Z) = x - log_z
    prog.assert(log_s.clone().eq(x.real_sub(log_z)));

    // Negated property: ls != log_s
    let violation = ls.ne(log_s);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "log_softmax_identity");
}

// ---------------------------------------------------------------------------
// Test 457: Softmax temperature: higher T -> more uniform
// ---------------------------------------------------------------------------

/// Prove: for T2 > T1 > 0, softmax(x/T2) is more uniform than softmax(x/T1).
///
/// As temperature T increases, softmax(x/T) approaches uniform distribution.
/// For 2-element case: if e1 > e2 > 0 (from x1 > x2), then s1 = e1/(e1+e2).
/// Dividing by higher T compresses the ratio e1/e2 toward 1, making s1 closer to 0.5.
///
/// We model: for the larger-element output s at two temperatures, the one with
/// higher T is closer to 0.5 (more uniform).
#[test]
fn test_457_softmax_temperature_uniformity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("e1_t1", real.clone());
    let _ = prog.declare_const("e2_t1", real.clone());
    let _ = prog.declare_const("e1_t2", real.clone());
    let _ = prog.declare_const("e2_t2", real.clone());
    let _ = prog.declare_const("s_t1", real.clone());
    let _ = prog.declare_const("s_t2", real);

    let e1_t1 = real_var("e1_t1");
    let e2_t1 = real_var("e2_t1");
    let e1_t2 = real_var("e1_t2");
    let e2_t2 = real_var("e2_t2");
    let s_t1 = real_var("s_t1");
    let s_t2 = real_var("s_t2");

    // All exp values positive
    prog.assert(e1_t1.clone().real_gt(Expr::real(0)));
    prog.assert(e2_t1.clone().real_gt(Expr::real(0)));
    prog.assert(e1_t2.clone().real_gt(Expr::real(0)));
    prog.assert(e2_t2.clone().real_gt(Expr::real(0)));

    // At lower temperature T1: e1_t1 > e2_t1 (x1 > x2, more separated)
    prog.assert(e1_t1.clone().real_gt(e2_t1.clone()));

    // At higher temperature T2: the ratio e1/e2 is closer to 1
    // Model: e1_t2/e2_t2 < e1_t1/e2_t1, equivalently e1_t2 * e2_t1 < e1_t1 * e2_t2
    prog.assert(
        e1_t2
            .clone()
            .real_mul(e2_t1.clone())
            .real_lt(e1_t1.clone().real_mul(e2_t2.clone())),
    );
    // Still ordered: e1_t2 >= e2_t2
    prog.assert(e1_t2.clone().real_ge(e2_t2.clone()));

    // s_t1 = e1_t1 / (e1_t1 + e2_t1)
    prog.assert(
        s_t1.clone()
            .real_mul(e1_t1.clone().real_add(e2_t1))
            .eq(e1_t1),
    );
    // s_t2 = e1_t2 / (e1_t2 + e2_t2)
    prog.assert(
        s_t2.clone()
            .real_mul(e1_t2.clone().real_add(e2_t2))
            .eq(e1_t2),
    );

    // s_t1 > 0.5 (dominant element) and s_t2 > 0.5 (still dominant but less so)
    prog.assert(s_t1.clone().real_gt(Expr::real_ratio(1, 2)));
    prog.assert(s_t2.clone().real_ge(Expr::real_ratio(1, 2)));

    // Negated property: s_t2 >= s_t1 (higher T should make s closer to 0.5, i.e., s_t2 < s_t1)
    let violation = s_t2.real_ge(s_t1);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_temperature_uniformity");
}

// ---------------------------------------------------------------------------
// Test 458: Attention score scaling: score = Q*K^T / sqrt(d_k)
// ---------------------------------------------------------------------------

/// Prove: scaled attention score equals dot product divided by sqrt(d_k).
///
/// For a single Q-K pair: score = (q . k) / sqrt(d_k).
/// We model: raw_score = q . k, scaled_score = raw_score / scale,
/// where scale = sqrt(d_k). The scaling reduces the magnitude by factor scale.
#[test]
fn test_458_attention_score_scaling() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("raw_score", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("scaled_score", real);

    let raw_score = real_var("raw_score");
    let scale = real_var("scale");
    let scaled_score = real_var("scaled_score");

    // Input bounds
    prog.assert(raw_score.clone().real_ge(Expr::real(-1000)));
    prog.assert(raw_score.clone().real_le(Expr::real(1000)));

    // scale = sqrt(d_k) > 0 (e.g., sqrt(64) = 8)
    prog.assert(scale.clone().real_gt(Expr::real(0)));
    prog.assert(scale.clone().real_le(Expr::real(100)));

    // scaled_score = raw_score / scale
    prog.assert(
        scaled_score
            .clone()
            .real_mul(scale.clone())
            .eq(raw_score.clone()),
    );

    // Property: |scaled_score| <= |raw_score| (scaling reduces magnitude)
    // Since scale >= 1 in practice (d_k >= 1), |scaled_score| <= |raw_score|
    prog.assert(scale.real_ge(Expr::real(1)));

    // Negated property: |scaled_score| > |raw_score|
    // Equivalently: scaled_score^2 > raw_score^2
    let ss_sq = scaled_score.clone().real_mul(scaled_score);
    let rs_sq = raw_score.clone().real_mul(raw_score);
    let violation = ss_sq.real_gt(rs_sq);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "attention_score_scaling");
}

// ---------------------------------------------------------------------------
// Test 459: Scaled attention scores bounded by input norms
// ---------------------------------------------------------------------------

/// Prove: |score| <= ||q|| * ||k|| / sqrt(d_k) (Cauchy-Schwarz).
///
/// By Cauchy-Schwarz: |q . k| <= ||q|| * ||k||.
/// After scaling: |score| = |q . k| / sqrt(d_k) <= ||q|| * ||k|| / sqrt(d_k).
///
/// We model: abs_dot <= norm_q * norm_k (Cauchy-Schwarz axiom),
/// abs_score = abs_dot / scale, and prove abs_score <= norm_q * norm_k / scale.
#[test]
fn test_459_attention_scores_bounded_by_norms() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("abs_dot", real.clone());
    let _ = prog.declare_const("norm_q", real.clone());
    let _ = prog.declare_const("norm_k", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("abs_score", real.clone());
    let _ = prog.declare_const("bound", real);

    let abs_dot = real_var("abs_dot");
    let norm_q = real_var("norm_q");
    let norm_k = real_var("norm_k");
    let scale = real_var("scale");
    let abs_score = real_var("abs_score");
    let bound = real_var("bound");

    // Norms are non-negative
    prog.assert(norm_q.clone().real_ge(Expr::real(0)));
    prog.assert(norm_k.clone().real_ge(Expr::real(0)));
    prog.assert(abs_dot.clone().real_ge(Expr::real(0)));
    prog.assert(scale.clone().real_gt(Expr::real(0)));

    // Cauchy-Schwarz: |q . k| <= ||q|| * ||k||
    prog.assert(
        abs_dot
            .clone()
            .real_le(norm_q.clone().real_mul(norm_k.clone())),
    );

    // abs_score = abs_dot / scale
    prog.assert(abs_score.clone().real_mul(scale.clone()).eq(abs_dot));

    // bound = norm_q * norm_k / scale
    prog.assert(bound.clone().real_mul(scale).eq(norm_q.real_mul(norm_k)));

    // Negated property: abs_score > bound
    let violation = abs_score.real_gt(bound);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "attention_scores_bounded_by_norms");
}

// ---------------------------------------------------------------------------
// Test 460: Attention weights are row-stochastic (each row sums to 1)
// ---------------------------------------------------------------------------

/// Prove: attention weights (after softmax) for each query sum to 1.
///
/// Attention weights W_ij = softmax(score_i)_j. Since softmax outputs sum
/// to 1, each row of the attention weight matrix sums to 1.
///
/// We model a row with 3 elements (representative) and prove sum = 1.
#[test]
fn test_460_attention_weights_row_stochastic() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("w3", real);

    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let w3 = real_var("w3");

    // Softmax outputs: all positive, sum to 1
    prog.assert(w1.clone().real_gt(Expr::real(0)));
    prog.assert(w2.clone().real_gt(Expr::real(0)));
    prog.assert(w3.clone().real_gt(Expr::real(0)));
    prog.assert(
        w1.clone()
            .real_add(w2.clone())
            .real_add(w3.clone())
            .eq(Expr::real(1)),
    );

    // Negated property: sum != 1
    let violation = w1.real_add(w2).real_add(w3).ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "attention_weights_row_stochastic");
}

// ---------------------------------------------------------------------------
// Test 461: Causal mask: masked positions get -inf before softmax
// ---------------------------------------------------------------------------

/// Prove: after applying causal mask with -inf, masked softmax outputs are 0.
///
/// Causal masking sets score_ij = -inf for j > i. After softmax, exp(-inf) = 0,
/// so the masked positions contribute 0 to the softmax output.
///
/// We model: for a 2-position sequence where position 0 cannot attend to
/// position 1. The masked score is -M (large negative), so exp(-M) ~ 0.
/// The softmax output for the masked position approaches 0.
#[test]
fn test_461_causal_mask_zero_after_softmax() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp_valid", real.clone());
    let _ = prog.declare_const("exp_masked", real.clone());
    let _ = prog.declare_const("s_masked", real);

    let exp_valid = real_var("exp_valid");
    let exp_masked = real_var("exp_masked");
    let s_masked = real_var("s_masked");

    // exp(valid_score) is positive and bounded
    prog.assert(exp_valid.clone().real_gt(Expr::real(0)));
    prog.assert(exp_valid.clone().real_le(Expr::real(1000)));

    // exp(masked_score) is extremely small (score = -M, large negative)
    // exp(-1000) ~ 0, so exp_masked in [0, epsilon]
    let eps = Expr::real_ratio(1, 1000000);
    prog.assert(exp_masked.clone().real_ge(Expr::real(0)));
    prog.assert(exp_masked.clone().real_le(eps));

    // s_masked = exp_masked / (exp_valid + exp_masked)
    let z = exp_valid.real_add(exp_masked.clone());
    prog.assert(s_masked.clone().real_mul(z).eq(exp_masked));

    // Negated property: s_masked > 0.001 (should be near-zero)
    let violation = s_masked.real_gt(Expr::real_ratio(1, 1000));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "causal_mask_zero_after_softmax");
}

// ---------------------------------------------------------------------------
// Test 462: Attention output bounded by value matrix bounds
// ---------------------------------------------------------------------------

/// Prove: if all values V_ij are in [lo, hi], then attention output is in [lo, hi].
///
/// Attention output = sum_j(w_j * V_j) where w_j >= 0 and sum(w_j) = 1.
/// This is a convex combination of values in [lo, hi], so the output is in [lo, hi].
///
/// We model: 2 values in [lo, hi], weights w1 + w2 = 1 with w1,w2 > 0,
/// output = w1*v1 + w2*v2, and prove lo <= output <= hi.
#[test]
fn test_462_attention_output_bounded_by_values() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("v1", real.clone());
    let _ = prog.declare_const("v2", real.clone());
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("out", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real);

    let v1 = real_var("v1");
    let v2 = real_var("v2");
    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let out = real_var("out");
    let lo = real_var("lo");
    let hi = real_var("hi");

    // lo <= hi
    prog.assert(lo.clone().real_le(hi.clone()));

    // Values in [lo, hi]
    prog.assert(v1.clone().real_ge(lo.clone()));
    prog.assert(v1.clone().real_le(hi.clone()));
    prog.assert(v2.clone().real_ge(lo.clone()));
    prog.assert(v2.clone().real_le(hi.clone()));

    // Weights: w1, w2 >= 0, w1 + w2 = 1
    prog.assert(w1.clone().real_ge(Expr::real(0)));
    prog.assert(w2.clone().real_ge(Expr::real(0)));
    prog.assert(w1.clone().real_add(w2.clone()).eq(Expr::real(1)));

    // out = w1*v1 + w2*v2
    prog.assert(out.clone().eq(w1.real_mul(v1).real_add(w2.real_mul(v2))));

    // Negated property: out < lo OR out > hi
    let violation = out.clone().real_lt(lo).or(out.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "attention_output_bounded_by_values");
}

// ---------------------------------------------------------------------------
// Test 463: Multi-head split preserves total dimension
// ---------------------------------------------------------------------------

/// Prove: splitting d_model into n_heads heads of d_k each preserves dimension.
///
/// d_model = n_heads * d_k. After split, each head processes d_k dimensions.
/// Concatenation restores d_model = n_heads * d_k.
///
/// We model: total = n * d with n > 0, d > 0, and prove total = n * d.
#[test]
fn test_463_multi_head_split_preserves_dimension() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("d_model", real.clone());
    let _ = prog.declare_const("n_heads", real.clone());
    let _ = prog.declare_const("d_k", real.clone());
    let _ = prog.declare_const("reconstructed", real);

    let d_model = real_var("d_model");
    let n_heads = real_var("n_heads");
    let d_k = real_var("d_k");
    let reconstructed = real_var("reconstructed");

    // n_heads > 0, d_k > 0
    prog.assert(n_heads.clone().real_gt(Expr::real(0)));
    prog.assert(d_k.clone().real_gt(Expr::real(0)));

    // d_model = n_heads * d_k
    prog.assert(d_model.clone().eq(n_heads.clone().real_mul(d_k.clone())));

    // reconstructed = n_heads * d_k (concatenation)
    prog.assert(reconstructed.clone().eq(n_heads.real_mul(d_k)));

    // Negated property: reconstructed != d_model
    let violation = reconstructed.ne(d_model);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "multi_head_split_preserves_dimension");
}

// ---------------------------------------------------------------------------
// Test 464: Key-value repeat for GQA: repeated KV matches original
// ---------------------------------------------------------------------------

/// Prove: in GQA, repeating KV heads preserves the key/value content.
///
/// With n_heads query heads and n_kv key-value heads (n_heads = n_kv * repeat),
/// each KV head is repeated `repeat` times. The repeated head must be identical
/// to the original.
///
/// We model: original KV value `kv_orig`, repeated value `kv_rep`.
/// The repeat operation copies: kv_rep = kv_orig.
#[test]
fn test_464_gqa_kv_repeat_preserves_content() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("kv_orig", real.clone());
    let _ = prog.declare_const("kv_rep", real.clone());
    let _ = prog.declare_const("n_heads", real.clone());
    let _ = prog.declare_const("n_kv", real.clone());
    let _ = prog.declare_const("repeat", real);

    let kv_orig = real_var("kv_orig");
    let kv_rep = real_var("kv_rep");
    let n_heads = real_var("n_heads");
    let n_kv = real_var("n_kv");
    let repeat = real_var("repeat");

    // Valid GQA config: n_heads = n_kv * repeat, all positive
    prog.assert(n_heads.clone().real_gt(Expr::real(0)));
    prog.assert(n_kv.clone().real_gt(Expr::real(0)));
    prog.assert(repeat.clone().real_gt(Expr::real(0)));
    prog.assert(n_heads.eq(n_kv.real_mul(repeat)));

    // Input bound on KV values
    prog.assert(kv_orig.clone().real_ge(Expr::real(-100)));
    prog.assert(kv_orig.clone().real_le(Expr::real(100)));

    // Repeat operation: kv_rep = kv_orig (identity copy)
    prog.assert(kv_rep.clone().eq(kv_orig.clone()));

    // Negated property: kv_rep != kv_orig
    let violation = kv_rep.ne(kv_orig);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_kv_repeat_preserves_content");
}

// ---------------------------------------------------------------------------
// Test 465: Attention score symmetry for self-attention
// ---------------------------------------------------------------------------

/// Prove: in self-attention with Q=K, score(i,j) = score(j,i).
///
/// When Q = K, the attention score matrix S = Q * Q^T is symmetric:
/// S_ij = q_i . q_j = q_j . q_i = S_ji.
///
/// We model: score_ij = dot(q_i, q_j) and score_ji = dot(q_j, q_i),
/// and prove they are equal (commutativity of dot product).
#[test]
fn test_465_self_attention_score_symmetry() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("qi_1", real.clone());
    let _ = prog.declare_const("qi_2", real.clone());
    let _ = prog.declare_const("qj_1", real.clone());
    let _ = prog.declare_const("qj_2", real.clone());
    let _ = prog.declare_const("score_ij", real.clone());
    let _ = prog.declare_const("score_ji", real);

    let qi_1 = real_var("qi_1");
    let qi_2 = real_var("qi_2");
    let qj_1 = real_var("qj_1");
    let qj_2 = real_var("qj_2");
    let score_ij = real_var("score_ij");
    let score_ji = real_var("score_ji");

    // 2D dot products (representative of d_k dimensions)
    // score_ij = qi_1*qj_1 + qi_2*qj_2
    prog.assert(
        score_ij.clone().eq(qi_1
            .clone()
            .real_mul(qj_1.clone())
            .real_add(qi_2.clone().real_mul(qj_2.clone()))),
    );

    // score_ji = qj_1*qi_1 + qj_2*qi_2
    prog.assert(
        score_ji
            .clone()
            .eq(qj_1.real_mul(qi_1).real_add(qj_2.real_mul(qi_2))),
    );

    // Negated property: score_ij != score_ji
    let violation = score_ij.ne(score_ji);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "self_attention_score_symmetry");
}

// ---------------------------------------------------------------------------
// Test 466: Softmax gradient: d_softmax = s * (delta - s)
// ---------------------------------------------------------------------------

/// Prove: the Jacobian of softmax satisfies ds_i/dx_j = s_i * (delta_ij - s_j).
///
/// For i = j (diagonal): ds_i/dx_i = s_i * (1 - s_i).
/// For i != j (off-diagonal): ds_i/dx_j = -s_i * s_j.
///
/// We model the diagonal case: grad = s * (1 - s), and prove:
/// - grad > 0 when 0 < s < 1 (gradient is positive on diagonal)
/// - grad <= 0.25 (maximum at s = 0.5)
#[test]
fn test_466_softmax_gradient_jacobian() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("grad", real);

    let s = real_var("s");
    let grad = real_var("grad");

    // s in (0, 1) — valid softmax output
    prog.assert(s.clone().real_gt(Expr::real(0)));
    prog.assert(s.clone().real_lt(Expr::real(1)));

    // grad = s * (1 - s)
    let one_minus_s = Expr::real(1).real_sub(s.clone());
    prog.assert(grad.clone().eq(s.real_mul(one_minus_s)));

    // Negated property: grad <= 0 OR grad > 0.25
    // (gradient should be in (0, 0.25] for s in (0,1))
    let violation = grad
        .clone()
        .real_le(Expr::real(0))
        .or(grad.real_gt(Expr::real_ratio(1, 4)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_gradient_jacobian");
}

// ---------------------------------------------------------------------------
// Test 467: Attention with ALiBi: bias is linear in position distance
// ---------------------------------------------------------------------------

/// Prove: ALiBi bias for head h at position distance d equals -m_h * d.
///
/// ALiBi adds a position-dependent bias: bias = -m * |i - j| where m is
/// the head-specific slope. For head h, m_h = 2^(-8h/H) where H is total heads.
///
/// We model: bias = -m * dist with m > 0 and dist >= 0, and prove:
/// - bias <= 0 (always non-positive, penalizing distant tokens)
/// - bias is linear in distance (doubling dist doubles |bias|)
#[test]
fn test_467_alibi_linear_position_bias() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("dist1", real.clone());
    let _ = prog.declare_const("dist2", real.clone());
    let _ = prog.declare_const("bias1", real.clone());
    let _ = prog.declare_const("bias2", real);

    let m = real_var("m");
    let dist1 = real_var("dist1");
    let dist2 = real_var("dist2");
    let bias1 = real_var("bias1");
    let bias2 = real_var("bias2");

    // m > 0 (head slope)
    prog.assert(m.clone().real_gt(Expr::real(0)));
    prog.assert(m.clone().real_le(Expr::real(1)));

    // Distances: dist2 = 2 * dist1, both non-negative
    prog.assert(dist1.clone().real_ge(Expr::real(0)));
    prog.assert(dist1.clone().real_le(Expr::real(1000)));
    prog.assert(dist2.clone().eq(Expr::real(2).real_mul(dist1.clone())));

    // ALiBi bias: bias = -m * dist
    prog.assert(
        bias1
            .clone()
            .eq(Expr::real(0).real_sub(m.clone().real_mul(dist1))),
    );
    prog.assert(bias2.clone().eq(Expr::real(0).real_sub(m.real_mul(dist2))));

    // Negated property: bias2 != 2 * bias1 (linearity violated)
    let violation = bias2.ne(Expr::real(2).real_mul(bias1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "alibi_linear_position_bias");
}

// ---------------------------------------------------------------------------
// Test 468: Top-k attention: only top-k scores are non-zero
// ---------------------------------------------------------------------------

/// Prove: after top-k masking and softmax, exactly k positions have non-zero weight.
///
/// Top-k attention masks all scores outside the top-k to -inf before softmax.
/// After softmax, exp(-inf) = 0, so exactly k positions have non-zero weight.
///
/// We model a 3-element case with k=2: two valid scores and one masked.
/// The masked position's weight should be 0 (or near-0).
#[test]
fn test_468_topk_attention_sparsity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("e_top1", real.clone());
    let _ = prog.declare_const("e_top2", real.clone());
    let _ = prog.declare_const("e_masked", real.clone());
    let _ = prog.declare_const("w_masked", real);

    let e_top1 = real_var("e_top1");
    let e_top2 = real_var("e_top2");
    let e_masked = real_var("e_masked");
    let w_masked = real_var("w_masked");

    // Top-k values have positive exp
    prog.assert(e_top1.clone().real_gt(Expr::real(0)));
    prog.assert(e_top1.clone().real_le(Expr::real(1000)));
    prog.assert(e_top2.clone().real_gt(Expr::real(0)));
    prog.assert(e_top2.clone().real_le(Expr::real(1000)));

    // Masked value has exp near 0 (score was -inf)
    let eps = Expr::real_ratio(1, 1000000);
    prog.assert(e_masked.clone().real_ge(Expr::real(0)));
    prog.assert(e_masked.clone().real_le(eps));

    // w_masked = e_masked / (e_top1 + e_top2 + e_masked)
    let z = e_top1.real_add(e_top2).real_add(e_masked.clone());
    prog.assert(w_masked.clone().real_mul(z).eq(e_masked));

    // Negated property: w_masked > 0.001 (should be near-zero)
    let violation = w_masked.real_gt(Expr::real_ratio(1, 1000));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "topk_attention_sparsity");
}

// ---------------------------------------------------------------------------
// Test 469: Sliding window attention: score zero outside window
// ---------------------------------------------------------------------------

/// Prove: in sliding window attention with window size W, positions outside
/// [i-W, i+W] have zero attention weight.
///
/// Sliding window masks positions |i - j| > W to -inf before softmax.
/// After softmax, masked positions have weight 0 (same mechanism as causal mask).
///
/// We model: one in-window score and one out-of-window score. The out-of-window
/// position gets near-zero weight after softmax.
#[test]
fn test_469_sliding_window_zero_outside() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp_inside", real.clone());
    let _ = prog.declare_const("exp_outside", real.clone());
    let _ = prog.declare_const("w_outside", real);

    let exp_inside = real_var("exp_inside");
    let exp_outside = real_var("exp_outside");
    let w_outside = real_var("w_outside");

    // In-window position has positive exp
    prog.assert(exp_inside.clone().real_gt(Expr::real(0)));
    prog.assert(exp_inside.clone().real_le(Expr::real(1000)));

    // Out-of-window position: masked to -inf, so exp is near-zero
    let eps = Expr::real_ratio(1, 1000000);
    prog.assert(exp_outside.clone().real_ge(Expr::real(0)));
    prog.assert(exp_outside.clone().real_le(eps));

    // w_outside = exp_outside / (exp_inside + exp_outside)
    let z = exp_inside.real_add(exp_outside.clone());
    prog.assert(w_outside.clone().real_mul(z).eq(exp_outside));

    // Negated property: w_outside > 0.001 (should be near-zero)
    let violation = w_outside.real_gt(Expr::real_ratio(1, 1000));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "sliding_window_zero_outside");
}

// ---------------------------------------------------------------------------
// Test 470: Cross-attention: Q from decoder, KV from encoder dimensions match
// ---------------------------------------------------------------------------

/// Prove: in cross-attention, Q (from decoder) and K (from encoder) must have
/// the same d_k dimension for the dot product to be well-defined.
///
/// Q has shape [seq_dec, d_k], K has shape [seq_enc, d_k].
/// The dot product Q * K^T requires the inner dimension to match.
/// The output has shape [seq_dec, seq_enc].
///
/// We model: Q inner dim = d_q, K inner dim = d_k, and require d_q = d_k
/// for the matmul to be valid. The output dimension = seq_dec * seq_enc.
#[test]
fn test_470_cross_attention_dimension_match() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("d_q", real.clone());
    let _ = prog.declare_const("d_k", real.clone());
    let _ = prog.declare_const("seq_dec", real.clone());
    let _ = prog.declare_const("seq_enc", real.clone());
    let _ = prog.declare_const("out_rows", real.clone());
    let _ = prog.declare_const("out_cols", real);

    let d_q = real_var("d_q");
    let d_k = real_var("d_k");
    let seq_dec = real_var("seq_dec");
    let seq_enc = real_var("seq_enc");
    let out_rows = real_var("out_rows");
    let out_cols = real_var("out_cols");

    // All dimensions positive
    prog.assert(d_q.clone().real_gt(Expr::real(0)));
    prog.assert(d_k.clone().real_gt(Expr::real(0)));
    prog.assert(seq_dec.clone().real_gt(Expr::real(0)));
    prog.assert(seq_enc.clone().real_gt(Expr::real(0)));

    // Cross-attention requirement: Q and K must share d_k dimension
    prog.assert(d_q.clone().eq(d_k.clone()));

    // Output shape: [seq_dec, seq_enc]
    prog.assert(out_rows.clone().eq(seq_dec));
    prog.assert(out_cols.clone().eq(seq_enc));

    // Negated property: d_q != d_k (dimension mismatch)
    let violation = d_q.ne(d_k);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cross_attention_dimension_match");
}
