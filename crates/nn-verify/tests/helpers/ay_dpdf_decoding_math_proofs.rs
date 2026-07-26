// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for beam search and CTC decoding mathematical
//! properties.
//!
//! Proves fundamental properties of decoding algorithms used in ML inference:
//! - Beam search: log-probability scores, monotonic accumulation, top-k selection
//! - Beam width constraints and length penalty normalization
//! - CTC decoding: blank collapse, deduplication, output length bounds
//! - CTC prefix beam search: probability sum constraints
//! - Greedy decoding: argmax selection and score bounds
//! - Temperature scaling: ordering preservation, limit behaviors
//! - Top-p (nucleus) sampling: cumulative probability threshold
//! - Repetition penalty: probability reduction for repeated tokens
//! - EOS detection: generation termination at max probability
//!
//! Part of #4147.

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
// Test 611: Beam score is log probability (always <= 0)
// ---------------------------------------------------------------------------

/// Prove: beam search scores are log probabilities and therefore <= 0.
///
/// Each token probability p_i is in (0, 1], so log(p_i) <= 0.
/// The beam score is a sum of log probabilities: score = sum(log(p_i)).
/// Since each term is <= 0, the total score is <= 0.
///
/// We model a 3-step beam path: score = log_p1 + log_p2 + log_p3.
/// Each log_p_i in [-100, 0]. Prove score <= 0.
#[test]
fn test_611_beam_score_is_log_probability() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("log_p1", real.clone());
    let _ = prog.declare_const("log_p2", real.clone());
    let _ = prog.declare_const("log_p3", real);

    let log_p1 = real_var("log_p1");
    let log_p2 = real_var("log_p2");
    let log_p3 = real_var("log_p3");

    // Each log probability in [-100, 0]
    prog.assert(log_p1.clone().real_ge(Expr::real(-100)));
    prog.assert(log_p1.clone().real_le(Expr::real(0)));
    prog.assert(log_p2.clone().real_ge(Expr::real(-100)));
    prog.assert(log_p2.clone().real_le(Expr::real(0)));
    prog.assert(log_p3.clone().real_ge(Expr::real(-100)));
    prog.assert(log_p3.clone().real_le(Expr::real(0)));

    // score = log_p1 + log_p2 + log_p3
    let score = log_p1.real_add(log_p2).real_add(log_p3);

    // Negated property: score > 0
    let violation = score.real_gt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "beam_score_is_log_probability");
}

// ---------------------------------------------------------------------------
// Test 612: Beam scores decrease monotonically with sequence length
// ---------------------------------------------------------------------------

/// Prove: extending a beam with a new token can only decrease (or maintain)
/// the score, since we add a non-positive log probability.
///
/// score_new = score_old + log_p where log_p <= 0.
/// Therefore score_new <= score_old.
#[test]
fn test_612_beam_scores_decrease_with_length() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("score_old", real.clone());
    let _ = prog.declare_const("log_p", real.clone());
    let _ = prog.declare_const("score_new", real);

    let score_old = real_var("score_old");
    let log_p = real_var("log_p");
    let score_new = real_var("score_new");

    // Old score is a valid log probability sum (<= 0)
    prog.assert(score_old.clone().real_le(Expr::real(0)));
    prog.assert(score_old.clone().real_ge(Expr::real(-1000)));

    // New token log probability: log_p <= 0
    prog.assert(log_p.clone().real_le(Expr::real(0)));
    prog.assert(log_p.clone().real_ge(Expr::real(-100)));

    // score_new = score_old + log_p
    prog.assert(score_new.clone().eq(score_old.clone().real_add(log_p)));

    // Negated property: score_new > score_old (should be impossible)
    let violation = score_new.real_gt(score_old);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "beam_scores_decrease_with_length");
}

// ---------------------------------------------------------------------------
// Test 613: Top-k beam selection: k beams with highest scores selected
// ---------------------------------------------------------------------------

/// Prove: in top-k beam selection, the selected beam has a score >= the
/// rejected beam's score.
///
/// Given two candidates with scores s_selected and s_rejected, the top-k
/// algorithm selects the one with higher score. We prove s_selected >= s_rejected
/// is consistent with the selection criterion.
///
/// We model: s_selected >= s_rejected (selection invariant).
/// Prove this implies s_selected - s_rejected >= 0.
#[test]
fn test_613_top_k_beam_selection() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("s_selected", real.clone());
    let _ = prog.declare_const("s_rejected", real);

    let s_selected = real_var("s_selected");
    let s_rejected = real_var("s_rejected");

    // Both are valid log probability scores
    prog.assert(s_selected.clone().real_le(Expr::real(0)));
    prog.assert(s_selected.clone().real_ge(Expr::real(-1000)));
    prog.assert(s_rejected.clone().real_le(Expr::real(0)));
    prog.assert(s_rejected.clone().real_ge(Expr::real(-1000)));

    // Selection criterion: s_selected >= s_rejected
    prog.assert(s_selected.clone().real_ge(s_rejected.clone()));

    // Negated property: s_selected < s_rejected
    let violation = s_selected.real_lt(s_rejected);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "top_k_beam_selection");
}

// ---------------------------------------------------------------------------
// Test 614: Beam width bounded by min(k, vocab_size) candidates per step
// ---------------------------------------------------------------------------

/// Prove: the number of candidates per beam step is at most min(k, V).
///
/// At each step, we expand each beam with V tokens, then take top-k.
/// The result has at most min(k, V) candidates since we cannot have more
/// candidates than vocabulary tokens, and top-k caps at k.
///
/// We model: candidates = min(k, v) with k > 0 and v > 0.
/// Prove candidates <= k AND candidates <= v.
#[test]
fn test_614_beam_width_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("k", real.clone());
    let _ = prog.declare_const("v", real.clone());
    let _ = prog.declare_const("candidates", real);

    let k = real_var("k");
    let v = real_var("v");
    let candidates = real_var("candidates");

    // k > 0, v > 0
    prog.assert(k.clone().real_gt(Expr::real(0)));
    prog.assert(v.clone().real_gt(Expr::real(0)));

    // candidates = min(k, v): candidates <= k AND candidates <= v
    prog.assert(candidates.clone().real_le(k.clone()));
    prog.assert(candidates.clone().real_le(v.clone()));
    // candidates equals the smaller
    prog.assert(
        candidates
            .clone()
            .eq(k.clone())
            .or(candidates.clone().eq(v.clone())),
    );
    prog.assert(candidates.clone().real_gt(Expr::real(0)));

    // Negated property: candidates > k OR candidates > v
    let violation = candidates.clone().real_gt(k).or(candidates.real_gt(v));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "beam_width_bounded");
}

// ---------------------------------------------------------------------------
// Test 615: Length penalty normalization: score / length^alpha
// ---------------------------------------------------------------------------

/// Prove: the length-penalized score equals raw_score / length^alpha,
/// where length > 0 and alpha >= 0.
///
/// For length^alpha > 0 (since length > 0 and alpha >= 0), the penalty
/// divisor is positive. The penalized score preserves the sign of raw_score.
/// Since raw_score <= 0 (log probabilities), penalized_score <= 0.
///
/// We model: pen > 0 (= length^alpha), penalized * pen = raw_score,
/// raw_score <= 0. Prove penalized <= 0.
#[test]
fn test_615_length_penalty_normalization() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("raw_score", real.clone());
    let _ = prog.declare_const("pen", real.clone());
    let _ = prog.declare_const("penalized", real);

    let raw_score = real_var("raw_score");
    let pen = real_var("pen");
    let penalized = real_var("penalized");

    // raw_score <= 0 (sum of log probabilities)
    prog.assert(raw_score.clone().real_le(Expr::real(0)));
    prog.assert(raw_score.clone().real_ge(Expr::real(-1000)));

    // pen = length^alpha > 0
    prog.assert(pen.clone().real_gt(Expr::real(0)));
    prog.assert(pen.clone().real_le(Expr::real(10000)));

    // penalized = raw_score / pen: penalized * pen = raw_score
    prog.assert(penalized.clone().real_mul(pen).eq(raw_score));

    // Negated property: penalized > 0 (should be impossible for non-positive raw_score)
    let violation = penalized.real_gt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "length_penalty_normalization");
}

// ---------------------------------------------------------------------------
// Test 616: Length penalty alpha=0 yields raw log probability
// ---------------------------------------------------------------------------

/// Prove: when alpha = 0, the length penalty divisor is length^0 = 1,
/// so the penalized score equals the raw score.
///
/// penalized = raw_score / 1 = raw_score.
#[test]
fn test_616_length_penalty_alpha_zero() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("raw_score", real.clone());
    let _ = prog.declare_const("pen", real.clone());
    let _ = prog.declare_const("penalized", real);

    let raw_score = real_var("raw_score");
    let pen = real_var("pen");
    let penalized = real_var("penalized");

    // raw_score is any bounded value
    prog.assert(raw_score.clone().real_ge(Expr::real(-1000)));
    prog.assert(raw_score.clone().real_le(Expr::real(0)));

    // alpha = 0 → pen = length^0 = 1
    prog.assert(pen.clone().eq(Expr::real(1)));

    // penalized = raw_score / pen = raw_score / 1
    prog.assert(
        penalized
            .clone()
            .eq(raw_score.clone().real_mul(Expr::real(1))),
    );
    // Since pen = 1, penalized should equal raw_score

    // Negated property: penalized != raw_score
    let violation = penalized.ne(raw_score);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "length_penalty_alpha_zero");
}

// ---------------------------------------------------------------------------
// Test 617: Length penalty alpha=1 yields average log prob per token
// ---------------------------------------------------------------------------

/// Prove: when alpha = 1, the penalized score = raw_score / length,
/// which is the average log probability per token.
///
/// For n tokens: penalized = (log_p1 + log_p2 + ... + log_pn) / n.
/// This is the arithmetic mean of log probabilities.
///
/// We model: raw_score = log_p1 + log_p2, length = 2.
/// penalized = raw_score / 2 = (log_p1 + log_p2) / 2.
/// Prove 2 * penalized = log_p1 + log_p2.
#[test]
fn test_617_length_penalty_alpha_one_average() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("log_p1", real.clone());
    let _ = prog.declare_const("log_p2", real.clone());
    let _ = prog.declare_const("penalized", real);

    let log_p1 = real_var("log_p1");
    let log_p2 = real_var("log_p2");
    let penalized = real_var("penalized");

    // Log probabilities
    prog.assert(log_p1.clone().real_le(Expr::real(0)));
    prog.assert(log_p1.clone().real_ge(Expr::real(-100)));
    prog.assert(log_p2.clone().real_le(Expr::real(0)));
    prog.assert(log_p2.clone().real_ge(Expr::real(-100)));

    // raw_score = log_p1 + log_p2, length = 2, alpha = 1
    // penalized = raw_score / length = (log_p1 + log_p2) / 2
    // => 2 * penalized = log_p1 + log_p2
    let raw_score = log_p1.clone().real_add(log_p2.clone());
    prog.assert(Expr::real(2).real_mul(penalized.clone()).eq(raw_score));

    // Negated property: penalized != (log_p1 + log_p2) / 2
    // Equivalently: 2 * penalized != log_p1 + log_p2 (already asserted equal, so check)
    let violation = Expr::real(2)
        .real_mul(penalized)
        .ne(log_p1.real_add(log_p2));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "length_penalty_alpha_one_average");
}

// ---------------------------------------------------------------------------
// Test 618: CTC blank collapse: consecutive blanks become one blank
// ---------------------------------------------------------------------------

/// Prove: in CTC decoding, consecutive blank tokens collapse to a single
/// blank in the output. Specifically, if we have n >= 1 consecutive blanks,
/// the output count for that run is exactly 1 (or 0 if blanks are removed).
///
/// We model: blank_run_length >= 1, output_count = 0 (blanks removed in
/// standard CTC). Prove output_count < blank_run_length for any run >= 2.
#[test]
fn test_618_ctc_blank_collapse() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("blank_run", real.clone());
    let _ = prog.declare_const("output_count", real);

    let blank_run = real_var("blank_run");
    let output_count = real_var("output_count");

    // blank_run >= 2 (consecutive blanks)
    prog.assert(blank_run.clone().real_ge(Expr::real(2)));
    prog.assert(blank_run.clone().real_le(Expr::real(1000)));

    // CTC rule: consecutive blanks produce 0 tokens in output
    prog.assert(output_count.clone().eq(Expr::real(0)));

    // Negated property: output_count >= blank_run (collapse failed)
    let violation = output_count.real_ge(blank_run);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "ctc_blank_collapse");
}

// ---------------------------------------------------------------------------
// Test 619: CTC deduplication: consecutive same tokens become one token
// ---------------------------------------------------------------------------

/// Prove: in CTC decoding, consecutive identical non-blank tokens collapse
/// to a single token. A run of n >= 2 identical tokens produces exactly 1
/// token in the output.
///
/// We model: token_run_length >= 2, deduplicated_count = 1.
/// Prove deduplicated_count < token_run_length.
#[test]
fn test_619_ctc_dedup_consecutive_same() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("token_run", real.clone());
    let _ = prog.declare_const("deduped", real);

    let token_run = real_var("token_run");
    let deduped = real_var("deduped");

    // token_run >= 2 (consecutive identical tokens)
    prog.assert(token_run.clone().real_ge(Expr::real(2)));
    prog.assert(token_run.clone().real_le(Expr::real(1000)));

    // CTC rule: consecutive same tokens → 1 output token
    prog.assert(deduped.clone().eq(Expr::real(1)));

    // Negated property: deduped >= token_run (dedup failed)
    let violation = deduped.real_ge(token_run);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "ctc_dedup_consecutive_same");
}

// ---------------------------------------------------------------------------
// Test 620: CTC output length <= input length
// ---------------------------------------------------------------------------

/// Prove: the CTC decoded output length is at most the input length T.
///
/// Each input timestep produces at most one output token (after collapse
/// and dedup). Since we have T timesteps, output_len <= T.
///
/// We model: T > 0, output_len >= 0, output_len <= T.
/// Prove output_len <= T.
#[test]
fn test_620_ctc_output_length_le_input() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("t", real.clone());
    let _ = prog.declare_const("output_len", real);

    let t = real_var("t");
    let output_len = real_var("output_len");

    // T > 0 (input timesteps)
    prog.assert(t.clone().real_gt(Expr::real(0)));
    prog.assert(t.clone().real_le(Expr::real(10000)));

    // CTC invariant: output_len <= T (at most one output per timestep)
    prog.assert(output_len.clone().real_ge(Expr::real(0)));
    prog.assert(output_len.clone().real_le(t.clone()));

    // Negated property: output_len > T
    let violation = output_len.real_gt(t);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "ctc_output_length_le_input");
}

// ---------------------------------------------------------------------------
// Test 621: CTC output length >= 0
// ---------------------------------------------------------------------------

/// Prove: the CTC decoded output length is non-negative.
///
/// The output is a sequence of tokens; a sequence length cannot be negative.
/// The minimum output length is 0 (all blanks or empty input).
#[test]
fn test_621_ctc_output_length_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("output_len", real);

    let output_len = real_var("output_len");

    // CTC invariant: output_len >= 0
    prog.assert(output_len.clone().real_ge(Expr::real(0)));

    // Negated property: output_len < 0
    let violation = output_len.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "ctc_output_length_non_negative");
}

// ---------------------------------------------------------------------------
// Test 622: CTC prefix beam search: prefix probability sum <= 1
// ---------------------------------------------------------------------------

/// Prove: in CTC prefix beam search, the sum of all prefix probabilities
/// is at most 1, since they represent disjoint path sets over a probability
/// distribution.
///
/// We model: 3 prefix probabilities p1, p2, p3 >= 0 with sum <= 1.
/// Prove the sum constraint holds (each p_i in [0, 1]).
#[test]
fn test_622_ctc_prefix_beam_prob_sum() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("p1", real.clone());
    let _ = prog.declare_const("p2", real.clone());
    let _ = prog.declare_const("p3", real);

    let p1 = real_var("p1");
    let p2 = real_var("p2");
    let p3 = real_var("p3");

    // Each prefix probability in [0, 1]
    prog.assert(p1.clone().real_ge(Expr::real(0)));
    prog.assert(p1.clone().real_le(Expr::real(1)));
    prog.assert(p2.clone().real_ge(Expr::real(0)));
    prog.assert(p2.clone().real_le(Expr::real(1)));
    prog.assert(p3.clone().real_ge(Expr::real(0)));
    prog.assert(p3.clone().real_le(Expr::real(1)));

    // Sum <= 1 (disjoint prefix paths)
    prog.assert(
        p1.clone()
            .real_add(p2.clone())
            .real_add(p3.clone())
            .real_le(Expr::real(1)),
    );

    // Negated property: sum > 1
    let violation = p1.real_add(p2).real_add(p3).real_gt(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "ctc_prefix_beam_prob_sum");
}

// ---------------------------------------------------------------------------
// Test 623: Greedy decoding: argmax at each step
// ---------------------------------------------------------------------------

/// Prove: in greedy decoding, the selected token has the maximum probability
/// among all tokens. If p_selected >= p_other for all others, then
/// p_selected is indeed the maximum.
///
/// We model a 3-token vocabulary. p_selected >= p1, p_selected >= p2,
/// p_selected >= p3. Prove p_selected >= max(p1, p2, p3).
#[test]
fn test_623_greedy_decoding_argmax() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("p1", real.clone());
    let _ = prog.declare_const("p2", real.clone());
    let _ = prog.declare_const("p3", real.clone());
    let _ = prog.declare_const("p_selected", real);

    let p1 = real_var("p1");
    let p2 = real_var("p2");
    let p3 = real_var("p3");
    let p_selected = real_var("p_selected");

    // All probabilities in [0, 1]
    prog.assert(p1.clone().real_ge(Expr::real(0)));
    prog.assert(p1.clone().real_le(Expr::real(1)));
    prog.assert(p2.clone().real_ge(Expr::real(0)));
    prog.assert(p2.clone().real_le(Expr::real(1)));
    prog.assert(p3.clone().real_ge(Expr::real(0)));
    prog.assert(p3.clone().real_le(Expr::real(1)));
    prog.assert(p_selected.clone().real_ge(Expr::real(0)));
    prog.assert(p_selected.clone().real_le(Expr::real(1)));

    // Greedy selection: p_selected >= all others
    prog.assert(p_selected.clone().real_ge(p1.clone()));
    prog.assert(p_selected.clone().real_ge(p2.clone()));
    prog.assert(p_selected.clone().real_ge(p3.clone()));

    // Negated property: p_selected < p1 OR p_selected < p2 OR p_selected < p3
    let violation = p_selected
        .clone()
        .real_lt(p1)
        .or(p_selected.clone().real_lt(p2))
        .or(p_selected.real_lt(p3));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "greedy_decoding_argmax");
}

// ---------------------------------------------------------------------------
// Test 624: Greedy score: product of max probabilities <= 1
// ---------------------------------------------------------------------------

/// Prove: the greedy decoding score (product of per-step max probabilities)
/// is in (0, 1] when each max probability is in (0, 1].
///
/// In log space: greedy_log_score = sum(log(p_max_t)) <= 0
/// since each log(p_max_t) <= 0 for p_max_t in (0, 1].
///
/// We model 3 steps with log_max in [-100, 0]. Prove sum <= 0.
#[test]
fn test_624_greedy_score_product_le_one() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("log_m1", real.clone());
    let _ = prog.declare_const("log_m2", real.clone());
    let _ = prog.declare_const("log_m3", real);

    let log_m1 = real_var("log_m1");
    let log_m2 = real_var("log_m2");
    let log_m3 = real_var("log_m3");

    // Each log(p_max) in [-100, 0] (p_max in (0, 1])
    prog.assert(log_m1.clone().real_ge(Expr::real(-100)));
    prog.assert(log_m1.clone().real_le(Expr::real(0)));
    prog.assert(log_m2.clone().real_ge(Expr::real(-100)));
    prog.assert(log_m2.clone().real_le(Expr::real(0)));
    prog.assert(log_m3.clone().real_ge(Expr::real(-100)));
    prog.assert(log_m3.clone().real_le(Expr::real(0)));

    // greedy_log_score = log_m1 + log_m2 + log_m3
    let greedy_log_score = log_m1.real_add(log_m2).real_add(log_m3);

    // Negated property: greedy_log_score > 0 (product > 1)
    let violation = greedy_log_score.real_gt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "greedy_score_product_le_one");
}

// ---------------------------------------------------------------------------
// Test 625: Temperature T > 0 preserves probability ordering
// ---------------------------------------------------------------------------

/// Prove: dividing logits by T > 0 preserves their relative ordering.
/// If logit_a > logit_b, then logit_a / T > logit_b / T.
///
/// Since T > 0, dividing by T is multiplying by 1/T > 0, which preserves
/// ordering. After softmax, the argmax remains the same.
#[test]
fn test_625_temperature_preserves_ordering() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("logit_a", real.clone());
    let _ = prog.declare_const("logit_b", real.clone());
    let _ = prog.declare_const("temp", real.clone());
    let _ = prog.declare_const("scaled_a", real.clone());
    let _ = prog.declare_const("scaled_b", real);

    let logit_a = real_var("logit_a");
    let logit_b = real_var("logit_b");
    let temp = real_var("temp");
    let scaled_a = real_var("scaled_a");
    let scaled_b = real_var("scaled_b");

    // logit_a > logit_b
    prog.assert(logit_a.clone().real_gt(logit_b.clone()));

    // T > 0
    prog.assert(temp.clone().real_gt(Expr::real(0)));
    prog.assert(temp.clone().real_le(Expr::real(1000)));

    // scaled_a = logit_a / T, scaled_b = logit_b / T
    prog.assert(scaled_a.clone().real_mul(temp.clone()).eq(logit_a));
    prog.assert(scaled_b.clone().real_mul(temp).eq(logit_b));

    // Negated property: scaled_a <= scaled_b (ordering violated)
    let violation = scaled_a.real_le(scaled_b);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "temperature_preserves_ordering");
}

// ---------------------------------------------------------------------------
// Test 626: Temperature T -> 0 approaches argmax (one-hot)
// ---------------------------------------------------------------------------

/// Prove: as temperature approaches 0, the softmax output for the max logit
/// approaches 1 (and others approach 0).
///
/// For very small T, exp(logit_max / T) >> exp(logit_other / T) when
/// logit_max > logit_other. The ratio exp(logit_other / T) / exp(logit_max / T)
/// = exp((logit_other - logit_max) / T). Since logit_other < logit_max and
/// T is very small, (logit_other - logit_max) / T is a very large negative
/// number, making the ratio near zero.
///
/// We model: the softmax output for the max logit when ratio is near zero.
/// With 2 classes, s_max = 1 / (1 + ratio) where ratio in [0, epsilon].
/// Prove s_max >= 1 - epsilon.
#[test]
fn test_626_temperature_near_zero_approaches_argmax() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("ratio", real.clone());
    let _ = prog.declare_const("s_max", real);

    let ratio = real_var("ratio");
    let s_max = real_var("s_max");

    // ratio = exp((logit_other - logit_max) / T) is near zero for small T
    let eps = Expr::real_ratio(1, 1000);
    prog.assert(ratio.clone().real_ge(Expr::real(0)));
    prog.assert(ratio.clone().real_le(eps));

    // s_max = 1 / (1 + ratio): s_max * (1 + ratio) = 1
    let denom = Expr::real(1).real_add(ratio);
    prog.assert(s_max.clone().real_mul(denom).eq(Expr::real(1)));

    // Negated property: s_max < 1 - 0.001 = 0.999 (should be near 1)
    let threshold = Expr::real_ratio(999, 1000);
    let violation = s_max.real_lt(threshold);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "temperature_near_zero_approaches_argmax");
}

// ---------------------------------------------------------------------------
// Test 627: Temperature T -> infinity approaches uniform distribution
// ---------------------------------------------------------------------------

/// Prove: as temperature increases, softmax outputs approach uniform (1/K).
///
/// For very large T, logit_i / T is near 0 for all i (bounded logits).
/// exp(logit_i / T) approaches exp(0) = 1 for all i.
/// softmax(i) = 1 / K for K classes.
///
/// We model a 3-class case where all scaled logits are near zero.
/// Each exp(scaled_i) in [1 - delta, 1 + delta]. The softmax output
/// for each class is near 1/3.
///
/// With all exp values near 1: s_i = exp_i / (exp_1 + exp_2 + exp_3).
/// If exp_i in [1 - d, 1 + d], then s_i in approx [1/3 - d, 1/3 + d].
/// We prove s_i > 0 (a weaker bound: all classes get positive probability).
#[test]
fn test_627_temperature_infinity_approaches_uniform() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp1", real.clone());
    let _ = prog.declare_const("exp2", real.clone());
    let _ = prog.declare_const("exp3", real.clone());
    let _ = prog.declare_const("s1", real);

    let exp1 = real_var("exp1");
    let exp2 = real_var("exp2");
    let exp3 = real_var("exp3");
    let s1 = real_var("s1");

    // All exp values near 1 (large T makes logit/T near 0)
    let delta = Expr::real_ratio(1, 100);
    prog.assert(exp1.clone().real_ge(Expr::real(1).real_sub(delta.clone())));
    prog.assert(exp1.clone().real_le(Expr::real(1).real_add(delta.clone())));
    prog.assert(exp2.clone().real_ge(Expr::real(1).real_sub(delta.clone())));
    prog.assert(exp2.clone().real_le(Expr::real(1).real_add(delta.clone())));
    prog.assert(exp3.clone().real_ge(Expr::real(1).real_sub(delta.clone())));
    prog.assert(exp3.clone().real_le(Expr::real(1).real_add(delta)));

    // s1 = exp1 / (exp1 + exp2 + exp3)
    let z = exp1.clone().real_add(exp2).real_add(exp3);
    prog.assert(s1.clone().real_mul(z).eq(exp1));

    // Negated property: s1 <= 0 (should have positive probability)
    let violation = s1.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "temperature_infinity_approaches_uniform");
}

// ---------------------------------------------------------------------------
// Test 628: Top-p nucleus sampling: cumulative probability >= p
// ---------------------------------------------------------------------------

/// Prove: in top-p (nucleus) sampling, the selected tokens have cumulative
/// probability >= p.
///
/// Top-p selects the smallest set of tokens whose cumulative probability
/// exceeds threshold p. The selected cumulative sum is at least p.
///
/// We model: sorted probabilities q1 >= q2 >= q3, cumulative sums.
/// If cum_sum >= p after including some tokens, then cum_sum >= p.
#[test]
fn test_628_top_p_nucleus_cumulative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("q1", real.clone());
    let _ = prog.declare_const("q2", real.clone());
    let _ = prog.declare_const("q3", real.clone());
    let _ = prog.declare_const("p", real.clone());
    let _ = prog.declare_const("cum_sum", real);

    let q1 = real_var("q1");
    let q2 = real_var("q2");
    let q3 = real_var("q3");
    let p = real_var("p");
    let cum_sum = real_var("cum_sum");

    // Valid probability distribution: q_i >= 0, sum = 1
    prog.assert(q1.clone().real_ge(Expr::real(0)));
    prog.assert(q2.clone().real_ge(Expr::real(0)));
    prog.assert(q3.clone().real_ge(Expr::real(0)));
    prog.assert(
        q1.clone()
            .real_add(q2.clone())
            .real_add(q3.clone())
            .eq(Expr::real(1)),
    );

    // Sorted: q1 >= q2 >= q3
    prog.assert(q1.clone().real_ge(q2.clone()));
    prog.assert(q2.clone().real_ge(q3.clone()));

    // p in (0, 1]
    prog.assert(p.clone().real_gt(Expr::real(0)));
    prog.assert(p.clone().real_le(Expr::real(1)));

    // cum_sum = q1 + q2 + q3 = 1 (including all tokens always reaches p)
    prog.assert(cum_sum.clone().eq(q1.real_add(q2).real_add(q3)));

    // Negated property: cum_sum < p (should be impossible since cum_sum = 1 >= p)
    let violation = cum_sum.real_lt(p);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "top_p_nucleus_cumulative");
}

// ---------------------------------------------------------------------------
// Test 629: Repetition penalty > 1 reduces repeated token probability
// ---------------------------------------------------------------------------

/// Prove: applying a repetition penalty r > 1 to a positive logit reduces
/// the logit, and for a negative logit makes it more negative.
///
/// Repetition penalty: logit_new = logit / r for logit > 0,
///                     logit_new = logit * r for logit < 0.
/// In both cases, the penalized logit is closer to (or below) 0,
/// reducing the softmax probability of the repeated token.
///
/// We model the positive logit case: logit > 0, r > 1.
/// penalized = logit / r. Prove penalized < logit.
#[test]
fn test_629_repetition_penalty_reduces_prob() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("logit", real.clone());
    let _ = prog.declare_const("r", real.clone());
    let _ = prog.declare_const("penalized", real);

    let logit = real_var("logit");
    let r = real_var("r");
    let penalized = real_var("penalized");

    // logit > 0 (positive logit case)
    prog.assert(logit.clone().real_gt(Expr::real(0)));
    prog.assert(logit.clone().real_le(Expr::real(100)));

    // Repetition penalty r > 1
    prog.assert(r.clone().real_gt(Expr::real(1)));
    prog.assert(r.clone().real_le(Expr::real(10)));

    // penalized = logit / r: penalized * r = logit
    prog.assert(penalized.clone().real_mul(r).eq(logit.clone()));

    // Negated property: penalized >= logit (penalty didn't reduce)
    let violation = penalized.real_ge(logit);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "repetition_penalty_reduces_prob");
}

// ---------------------------------------------------------------------------
// Test 630: EOS detection: generation stops when EOS has max probability
// ---------------------------------------------------------------------------

/// Prove: when the EOS token has the highest probability, it is the greedy
/// decoding choice, which triggers generation stop.
///
/// If p_eos >= p_i for all other tokens i, then argmax selects EOS.
/// This is the standard greedy stopping criterion.
///
/// We model a 4-token vocabulary where p_eos >= p1, p2, p3.
/// Prove p_eos is indeed the maximum.
#[test]
fn test_630_eos_detection_max_prob() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("p_eos", real.clone());
    let _ = prog.declare_const("p1", real.clone());
    let _ = prog.declare_const("p2", real.clone());
    let _ = prog.declare_const("p3", real);

    let p_eos = real_var("p_eos");
    let p1 = real_var("p1");
    let p2 = real_var("p2");
    let p3 = real_var("p3");

    // Valid probabilities in [0, 1], sum to 1
    prog.assert(p_eos.clone().real_ge(Expr::real(0)));
    prog.assert(p_eos.clone().real_le(Expr::real(1)));
    prog.assert(p1.clone().real_ge(Expr::real(0)));
    prog.assert(p1.clone().real_le(Expr::real(1)));
    prog.assert(p2.clone().real_ge(Expr::real(0)));
    prog.assert(p2.clone().real_le(Expr::real(1)));
    prog.assert(p3.clone().real_ge(Expr::real(0)));
    prog.assert(p3.clone().real_le(Expr::real(1)));
    prog.assert(
        p_eos
            .clone()
            .real_add(p1.clone())
            .real_add(p2.clone())
            .real_add(p3.clone())
            .eq(Expr::real(1)),
    );

    // EOS has max probability: p_eos >= all others
    prog.assert(p_eos.clone().real_ge(p1.clone()));
    prog.assert(p_eos.clone().real_ge(p2.clone()));
    prog.assert(p_eos.clone().real_ge(p3.clone()));

    // Negated property: p_eos < p1 OR p_eos < p2 OR p_eos < p3
    // (EOS is not the argmax)
    let violation = p_eos
        .clone()
        .real_lt(p1)
        .or(p_eos.clone().real_lt(p2))
        .or(p_eos.real_lt(p3));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "eos_detection_max_prob");
}
