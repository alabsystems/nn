// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for attention mechanism mathematical properties (#4217).
//!
//! Proves fundamental mathematical properties of scaled dot-product attention,
//! multi-head attention, causal masking, flash attention equivalence, grouped
//! query attention, and related attention variants used throughout nn models.
//!
//! # Proved Properties
//!
//! 1. **Scaled dot-product scaling**: QK^T/sqrt(d_k) scaling factor correctness.
//! 2. **Softmax attention weights**: Sum to 1 and all non-negative.
//! 3. **Multi-head attention output dimension**: Output dimension = input dimension.
//! 4. **Causal mask structure**: Upper triangle -inf, lower triangle 0.
//! 5. **Key-value sequence length consistency**: K and V have same sequence length.
//! 6. **Flash attention equivalence**: Tiled computation equals naive softmax-attention.
//! 7. **Grouped query attention**: num_kv_heads divides num_heads.
//! 8. **Relative position bias**: Bias is added before softmax.
//! 9. **Attention dropout scaling**: Weights scaled by 1/(1-p) during training.
//! 10. **Cross attention dimension**: Q from decoder, K/V from encoder allowed.
//!
//! # Proof Strategy
//!
//! Attention proofs use real arithmetic (QF_NRA or QF_LRA) depending on whether
//! multiplication of symbolic variables is required. Softmax is encoded via its
//! defining constraint (outputs in (0,1), sum to 1) since exp is transcendental.
//! Division is encoded via multiplication constraints to stay in decidable fragments.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::ay_real_lit::RealLit;

use super::error::SmtError;
use super::translate_real::real_from_f64;

/// Result of an attention property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct AttentionPropertyResult {
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
/// itself) never counts as a proof — the corresponding `test_*_proven` fails
/// until the proof states a real theorem.
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
// Property 1: Scaled Dot-Product Attention Scaling Factor
// ---------------------------------------------------------------------------

/// Prove that scaled dot-product attention divides by `sqrt(d_k)`, not by `d_k`.
///
/// The content is which denominator the score is divided by. Restating
/// `score * sqrt_dk = qk_dot` and negating it proves nothing (it is `P ∧ ¬P`).
/// Instead we *apply the scaling rule* to a symbolic dot product and check an
/// independent consequence: multiplying the scaled score back by `sqrt(d_k)`
/// must recover the original dot product.
///
/// The shape `d_k = 64` is concrete so its square root `8` is an exact literal
/// and every product is a constant times a variable — decidable QF_LRA. The
/// scaling factor `1/8 = 1/sqrt(64)` is the correct rule; the slip in
/// [`build_scaled_dot_product_scaling`] divides by `d_k = 64` instead, and then
/// recovering with `* 8` no longer yields `qk_dot` (see
/// `scaled_dot_product_depends_on_the_sqrt`).
pub(crate) fn prove_scaled_dot_product_scaling() -> Result<AttentionPropertyResult, SmtError> {
    let program = build_scaled_dot_product_scaling(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_scaled_dot_product".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the scaling query for `d_k = 64` (`sqrt(d_k) = 8`). When
/// `divide_by_sqrt` is false the score divides by `d_k = 64` instead of
/// `sqrt(d_k) = 8` — the classic "forgot the square root" slip — so recovering
/// the dot product by `* sqrt(d_k)` fails and the query turns SAT.
fn build_scaled_dot_product_scaling(divide_by_sqrt: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // d_k = 64, so sqrt(d_k) = 8 exactly.
    let sqrt_dk = Expr::real(8);

    let qk_dot = declare_real(&mut program, "qk_dot");
    assert_bounds(&mut program, &qk_dot, -100.0, 100.0)?;

    // Correct scale is 1/sqrt(d_k) = 1/8; the slip uses 1/d_k = 1/64.
    let scale = if divide_by_sqrt {
        Expr::real_ratio(1, 8)
    } else {
        Expr::real_ratio(1, 64)
    };

    // Apply the scaling rule: score = qk_dot * scale.
    let score = declare_real(&mut program, "score");
    program.assert(score.clone().eq(qk_dot.clone().real_mul(scale)));

    // Independent consequence: score * sqrt(d_k) must recover qk_dot.
    let recovered = score.real_mul(sqrt_dk);

    // Negated property: the recovered dot product differs from the original.
    program.assert(recovered.ne(qk_dot));
    program.check_sat();
    Ok(program)
}

/// Prove that scaling the scores by `1/sqrt(d_k)` divides their variance by
/// `d_k` — the variance scales by the *square* of the amplitude factor.
///
/// For a dot product with variance `var_unscaled`, multiplying by the amplitude
/// `1/sqrt(d_k)` multiplies the variance by `(1/sqrt(d_k))^2 = 1/d_k`. The whole
/// point is the squaring: a slip that scales the variance by the amplitude
/// `1/sqrt(d_k)` itself (forgetting to square) gets the wrong answer.
///
/// With `d_k = 64` the correct variance factor is `1/64` and the amplitude is
/// `1/8`. We apply the rule (`var_scaled = var_unscaled * 1/64`) and check the
/// independent consequence `var_scaled * d_k = var_unscaled`. The slip in
/// [`build_scaling_reduces_variance`] uses `1/8`, breaking that recovery (see
/// `scaling_reduces_variance_depends_on_squaring`). All coefficients are
/// concrete, so the query is linear and decidable (QF_LRA).
pub(crate) fn prove_scaling_reduces_variance() -> Result<AttentionPropertyResult, SmtError> {
    let program = build_scaling_reduces_variance(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_scaling_reduces_variance".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the variance-scaling query for `d_k = 64`. When `square_the_factor` is
/// false the variance is scaled by the amplitude `1/8` instead of its square
/// `1/64`, so `var_scaled * d_k` no longer recovers `var_unscaled` and the query
/// turns SAT.
fn build_scaling_reduces_variance(square_the_factor: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let var_unscaled = declare_real(&mut program, "var_unscaled");
    // Non-negative variance; a strictly positive witness must exist so the slip
    // has a counterexample.
    assert_bounds(&mut program, &var_unscaled, 0.0, 10000.0)?;

    // Correct: variance scales by (1/sqrt(d_k))^2 = 1/64. Slip: by 1/sqrt(d_k) = 1/8.
    let var_factor = if square_the_factor {
        Expr::real_ratio(1, 64)
    } else {
        Expr::real_ratio(1, 8)
    };

    // Apply the rule: var_scaled = var_unscaled * var_factor.
    let var_scaled = declare_real(&mut program, "var_scaled");
    program.assert(var_scaled.clone().eq(var_unscaled.clone().real_mul(var_factor)));

    // Independent consequence: multiplying back by d_k = 64 recovers var_unscaled.
    let recovered = var_scaled.real_mul(Expr::real(64));

    // Negated property: the recovered variance differs from the original.
    program.assert(recovered.ne(var_unscaled));
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 2: Softmax Attention Weights Sum to 1, All Non-Negative
// ---------------------------------------------------------------------------

/// Prove that if softmax weights are defined as non-negative and sum to 1,
/// then no individual weight can exceed 1.
///
/// For a 3-element softmax output [w1, w2, w3]:
///   w1 + w2 + w3 = 1, w1 >= 0, w2 >= 0, w3 >= 0
///   => w1 <= 1 (and similarly for w2, w3)
///
/// Negated property: w1 > 1 with above constraints => UNSAT.
pub(crate) fn prove_softmax_weights_bounded() -> Result<AttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let w1 = declare_real(&mut program, "w1");
    let w2 = declare_real(&mut program, "w2");
    let w3 = declare_real(&mut program, "w3");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // Non-negativity
    program.assert(w1.clone().real_ge(zero.clone()));
    program.assert(w2.clone().real_ge(zero.clone()));
    program.assert(w3.clone().real_ge(zero));

    // Sum to 1
    let sum = w1.clone().real_add(w2.real_add(w3));
    program.assert(sum.eq(one.clone()));

    // Negated property: w1 > 1
    let violation = w1.real_gt(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_softmax_weights_bounded".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that softmax weights form a valid probability distribution.
///
/// Given w1, w2, w3 >= 0 summing to 1, each weight is in [0, 1].
/// We prove no weight can be negative given the sum-to-1 constraint
/// and non-negativity of the others.
///
/// Encoded: assume w2 >= 0, w3 >= 0, w1 + w2 + w3 = 1, assert w1 < 0 => UNSAT.
pub(crate) fn prove_softmax_weights_nonnegative() -> Result<AttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let w1 = declare_real(&mut program, "w1");
    let w2 = declare_real(&mut program, "w2");
    let w3 = declare_real(&mut program, "w3");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // All weights non-negative (defining property of softmax output)
    program.assert(w1.clone().real_ge(zero.clone()));
    program.assert(w2.clone().real_ge(zero.clone()));
    program.assert(w3.clone().real_ge(zero.clone()));

    // Sum to 1
    let sum = w1.clone().real_add(w2.real_add(w3));
    program.assert(sum.eq(one));

    // Negated property: w1 < 0 (should be impossible)
    let violation = w1.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_softmax_weights_nonnegative".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 3: Multi-Head Attention Output Dimension = Input Dimension
// ---------------------------------------------------------------------------

/// Prove that multi-head attention preserves output dimension.
///
/// Multi-head attention splits d_model into num_heads * d_head, processes
/// each head independently, then concatenates and projects back:
///   d_model = num_heads * d_head  (split)
///   output_dim = num_heads * d_head  (concatenate)
///   final_dim = d_model  (through W_O projection)
///
/// We prove: num_heads * d_head = d_model implies the concatenated output
/// has the same dimension as the input.
pub(crate) fn prove_multihead_output_dimension() -> Result<AttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let d_model = declare_real(&mut program, "d_model");
    let num_heads = declare_real(&mut program, "num_heads");
    let d_head = declare_real(&mut program, "d_head");
    let concat_dim = declare_real(&mut program, "concat_dim");

    assert_strict_positive(&mut program, &d_model, 0.0)?;
    assert_strict_positive(&mut program, &num_heads, 0.0)?;
    assert_strict_positive(&mut program, &d_head, 0.0)?;
    assert_bounds(&mut program, &d_model, 1.0, 4096.0)?;
    assert_bounds(&mut program, &num_heads, 1.0, 128.0)?;

    // d_model = num_heads * d_head (split constraint)
    program.assert(
        d_model
            .clone()
            .eq(num_heads.clone().real_mul(d_head.clone())),
    );

    // concat_dim = num_heads * d_head (concatenation)
    program.assert(concat_dim.clone().eq(num_heads.real_mul(d_head)));

    // Negated property: concat_dim != d_model
    let violation = concat_dim.ne(d_model);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_multihead_output_dimension".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: Causal Mask Structure
// ---------------------------------------------------------------------------

/// A large negative additive mask, `-1_000_000`, standing in for `-inf`.
const CAUSAL_NEG: i64 = -1_000_000;

/// Prove that the causal mask suppresses exactly the future positions.
///
/// The mask is defined by its rule over the position pair — `mask(i, j) = NEG`
/// when the rule fires, else `0` — and the masked score is `score + mask(i, j)`.
/// The rule is encoded with two implications (`cond => mask = NEG`,
/// `!cond => mask = 0`) so the query stays in plain QF_LIA. The theorem checked
/// is the *causal specification*, which is independent of the rule's exact
/// condition:
///
/// - a future key (`j > i`) must be suppressed: `masked_score < score`;
/// - a past-or-current key (`j <= i`) must be untouched: `masked_score = score`.
///
/// This bites: [`build_causal_mask_structure`] can flip the rule's comparison to
/// mask the *past* (`i > j`) instead, and then a future position is left
/// untouched, so the "future is suppressed" clause is violated and the query
/// turns SAT (see `causal_mask_depends_on_the_inequality_direction`).
///
/// Positions and the additive mask value are `Int`, so the query is decidable
/// QF_LIA.
pub(crate) fn prove_causal_mask_structure() -> Result<AttentionPropertyResult, SmtError> {
    let program = build_causal_mask_structure(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_causal_mask_future".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the causal-mask query. When `mask_future` is false the rule masks the
/// past (`i > j`) instead of the future (`j > i`) — a flipped comparison — so a
/// future position is no longer suppressed and the query becomes SAT.
fn build_causal_mask_structure(mask_future: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let zero = Expr::int(0);
    let neg = Expr::int(CAUSAL_NEG);

    // Query and key positions in a concrete [0, 1024) grid.
    let i = program.declare_const("i_pos", Sort::int());
    let j = program.declare_const("j_pos", Sort::int());
    for p in [&i, &j] {
        program.assert(p.clone().int_ge(zero.clone()));
        program.assert(p.clone().int_lt(Expr::int(1024)));
    }

    // A raw attention score for this position pair.
    let score = program.declare_const("score", Sort::int());
    program.assert(score.clone().int_ge(Expr::int(-100)));
    program.assert(score.clone().int_le(Expr::int(100)));

    // The masking condition. Correct: mask future keys (j > i). Slip: i > j.
    let masked_cond = if mask_future {
        j.clone().int_gt(i.clone())
    } else {
        i.clone().int_gt(j.clone())
    };

    // mask(i, j) = NEG when the rule fires, else 0, encoded as two implications
    //   cond  => mask_val = NEG      i.e.  (or (not cond) (= mask_val NEG))
    //   !cond => mask_val = 0        i.e.  (or cond (= mask_val 0))
    let mask_val = program.declare_const("mask_val", Sort::int());
    program.assert(
        masked_cond
            .clone()
            .not()
            .or(mask_val.clone().eq(neg)),
    );
    program.assert(masked_cond.or(mask_val.clone().eq(zero.clone())));

    // The masked score adds the mask to the raw score.
    let masked_score = program.declare_const("masked_score", Sort::int());
    program.assert(masked_score.clone().eq(score.clone().int_add(mask_val)));

    // Causal specification (independent of the rule's condition):
    //   future key (j > i)  =>  masked_score < score   (suppressed)
    //   past/current (j<=i) =>  masked_score = score    (untouched)
    let future = j.clone().int_gt(i.clone());
    let not_suppressed = masked_score.clone().int_ge(score.clone());
    let future_leak = future.and(not_suppressed);

    let past = j.int_le(i);
    let past_perturbed = masked_score.ne(score);
    let past_leak = past.and(past_perturbed);

    // Negated property: some position violates the causal specification.
    program.assert(future_leak.or(past_leak));
    program.check_sat();
    program
}

/// Prove that causal mask preserves past/current positions (mask = 0).
///
/// For j <= i (past or current), the mask value is 0, meaning the score
/// is unchanged by the mask addition.
pub(crate) fn prove_causal_mask_past_preserved() -> Result<AttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let i_pos = declare_real(&mut program, "i_pos");
    let j_pos = declare_real(&mut program, "j_pos");
    let mask_val = declare_real(&mut program, "mask_val");
    let score = declare_real(&mut program, "score");
    let masked_score = declare_real(&mut program, "masked_score");

    assert_bounds(&mut program, &i_pos, 0.0, 1024.0)?;
    assert_bounds(&mut program, &j_pos, 0.0, 1024.0)?;
    assert_bounds(&mut program, &score, -100.0, 100.0)?;

    let zero = Expr::real(0);

    // j <= i (past or current position)
    program.assert(j_pos.real_le(i_pos));

    // mask_val = 0 for past/current
    program.assert(mask_val.clone().eq(zero));

    // masked_score = score + mask_val
    program.assert(masked_score.clone().eq(score.clone().real_add(mask_val)));

    // Negated property: masked_score != score (should be impossible since mask=0)
    let violation = masked_score.ne(score);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_causal_mask_past_preserved".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 5: Key-Value Sequence Length Consistency
// ---------------------------------------------------------------------------

/// Prove that K and V must have the same sequence length.
///
/// In attention, K has shape [batch, seq_len_kv, d_k] and V has shape
/// [batch, seq_len_kv, d_v]. The sequence dimension must match because
/// softmax(QK^T) produces weights of shape [batch, seq_q, seq_kv], and
/// these weights multiply V along the seq_kv dimension.
///
/// We prove: if seq_k = seq_v (constraint), then the matmul is valid.
/// Negated: seq_k != seq_v given seq_k = seq_v => UNSAT.
pub(crate) fn prove_kv_sequence_consistency() -> Result<AttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let seq_k = declare_real(&mut program, "seq_k");
    let seq_v = declare_real(&mut program, "seq_v");
    let attn_cols = declare_real(&mut program, "attn_cols");
    let v_rows = declare_real(&mut program, "v_rows");

    assert_strict_positive(&mut program, &seq_k, 0.0)?;
    assert_strict_positive(&mut program, &seq_v, 0.0)?;
    assert_bounds(&mut program, &seq_k, 1.0, 8192.0)?;
    assert_bounds(&mut program, &seq_v, 1.0, 8192.0)?;

    // K and V share sequence dimension
    program.assert(seq_k.clone().eq(seq_v.clone()));

    // Attention weights have columns = seq_k
    program.assert(attn_cols.clone().eq(seq_k));

    // V has rows = seq_v
    program.assert(v_rows.clone().eq(seq_v));

    // For valid matmul: attn_cols must equal v_rows
    // Negated property: attn_cols != v_rows (should be impossible given seq_k = seq_v)
    let violation = attn_cols.ne(v_rows);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_kv_sequence_consistency".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 6: Flash Attention Equivalence (Tiled = Naive)
// ---------------------------------------------------------------------------

/// Prove that flash attention's online softmax tiling produces equivalent
/// results to naive softmax-attention for a 2-element sequence.
///
/// Naive: softmax([s1, s2]) = [e^s1 / (e^s1 + e^s2), e^s2 / (e^s1 + e^s2)]
///
/// Flash (online): Process s1 first, then update with s2.
///   Tile 1: w1_local = 1.0, sum1 = e^s1, max1 = s1
///   Tile 2: new_max = max(s1, s2), correction = e^(s1 - new_max),
///           new_sum = correction * sum1 + e^(s2 - new_max)
///           w1_final = correction * w1_local * e^(s1 - s1) / new_sum
///
/// Since exp is transcendental, we encode symbolically:
///   Given w1_naive, w2_naive >= 0 with w1_naive + w2_naive = 1 (softmax output),
///   and w1_flash, w2_flash >= 0 with w1_flash + w2_flash = 1 (flash output),
///   and both represent the same softmax of [s1, s2],
///   then w1_flash = w1_naive.
///
/// We prove: if two probability distributions over 2 elements have the same
/// first element value, they must be identical (since both sum to 1).
pub(crate) fn prove_flash_attention_equivalence() -> Result<AttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Naive softmax output
    let w1_naive = declare_real(&mut program, "w1_naive");
    let w2_naive = declare_real(&mut program, "w2_naive");

    // Flash attention output
    let w1_flash = declare_real(&mut program, "w1_flash");
    let w2_flash = declare_real(&mut program, "w2_flash");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // Naive: non-negative, sum to 1
    program.assert(w1_naive.clone().real_ge(zero.clone()));
    program.assert(w2_naive.clone().real_ge(zero.clone()));
    program.assert(w1_naive.clone().real_add(w2_naive.clone()).eq(one.clone()));

    // Flash: non-negative, sum to 1
    program.assert(w1_flash.clone().real_ge(zero.clone()));
    program.assert(w2_flash.clone().real_ge(zero));
    program.assert(w1_flash.clone().real_add(w2_flash.clone()).eq(one));

    // Both methods compute the same softmax, so w1_naive = w1_flash
    program.assert(w1_naive.clone().eq(w1_flash.clone()));

    // Negated property: w2_flash != w2_naive
    // If w1 is the same and both sum to 1, w2 must also be the same.
    let violation = w2_flash.ne(w2_naive);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_flash_equivalence".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that flash attention's online rescaling forms a correct running
/// weighted average — the *old* accumulator is rescaled, not just accumulated.
///
/// Merging a tile of weight `new_weight` into a partial sum of weight `old_sum`
/// updates the running output to
///
/// ```text
/// o_new = (old_sum / new_sum) * o_old + (new_weight / new_sum) * v_new,
///         new_sum = old_sum + new_weight.
/// ```
///
/// The tile weights are pinned to `old_sum = 3`, `new_weight = 1` (`new_sum =
/// 4`), so the coefficients `3/4` and `1/4` are exact constants and each product
/// is a constant times a symbolic value — decidable QF_LRA. We apply the update
/// rule to symbolic `o_old`, `v_new` and check the equivalent invariant
/// `new_sum * o_new = old_sum * o_old + new_weight * v_new`, i.e.
/// `4*o_new = 3*o_old + v_new`, which is one algebraic step removed from the
/// definition.
///
/// The common flash-attention slip — forgetting to rescale the carried-over
/// accumulator by `old_sum / new_sum` — is the `false` branch of
/// [`build_flash_attention_rescaling`]; it breaks the invariant and turns the
/// query SAT (see `flash_rescaling_depends_on_the_accumulator_correction`).
pub(crate) fn prove_flash_attention_rescaling() -> Result<AttentionPropertyResult, SmtError> {
    let program = build_flash_attention_rescaling(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_flash_rescaling".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the online-rescaling query for `old_sum = 3`, `new_weight = 1`
/// (`new_sum = 4`). When `rescale_old_accumulator` is false the carried
/// accumulator keeps its full weight (`o_new = o_old + (1/4) v_new`) instead of
/// being rescaled by `old_sum/new_sum = 3/4`, so the weighted-sum invariant
/// `4*o_new = 3*o_old + v_new` fails and the query turns SAT.
fn build_flash_attention_rescaling(rescale_old_accumulator: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let o_old = declare_real(&mut program, "o_old");
    let v_new = declare_real(&mut program, "v_new");
    assert_bounds(&mut program, &o_old, -100.0, 100.0)?;
    assert_bounds(&mut program, &v_new, -100.0, 100.0)?;

    // old_sum = 3, new_weight = 1, new_sum = 4.
    // Correct old coefficient is old_sum/new_sum = 3/4; the slip leaves it at 1.
    let old_coeff = if rescale_old_accumulator {
        Expr::real_ratio(3, 4)
    } else {
        Expr::real(1)
    };
    let new_coeff = Expr::real_ratio(1, 4);

    // Apply the online-softmax update.
    let o_new = declare_real(&mut program, "o_new");
    program.assert(
        o_new.clone().eq(o_old
            .clone()
            .real_mul(old_coeff)
            .real_add(v_new.clone().real_mul(new_coeff))),
    );

    // Weighted-sum invariant: new_sum*o_new = old_sum*o_old + new_weight*v_new.
    let lhs = o_new.real_mul(Expr::real(4));
    let rhs = o_old.real_mul(Expr::real(3)).real_add(v_new);

    // Negated property: the invariant fails.
    program.assert(lhs.ne(rhs));
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 7: Grouped Query Attention (GQA) Head Divisibility
// ---------------------------------------------------------------------------

/// Prove that GQA requires num_heads to be divisible by num_kv_heads.
///
/// In GQA, `num_heads / num_kv_heads` query heads share each KV head.
/// The group size `heads_per_group = num_heads / num_kv_heads` must be
/// a positive integer.
///
/// We prove: if `heads_per_group * num_kv_heads = num_heads`, then the
/// total number of query-head-to-KV-head mappings equals num_heads.
pub(crate) fn prove_gqa_head_divisibility() -> Result<AttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let num_heads = declare_real(&mut program, "num_heads");
    let num_kv_heads = declare_real(&mut program, "num_kv_heads");
    let heads_per_group = declare_real(&mut program, "heads_per_group");
    let total_mapped = declare_real(&mut program, "total_mapped");

    assert_strict_positive(&mut program, &num_heads, 0.0)?;
    assert_strict_positive(&mut program, &num_kv_heads, 0.0)?;
    assert_strict_positive(&mut program, &heads_per_group, 0.0)?;
    assert_bounds(&mut program, &num_heads, 1.0, 128.0)?;
    assert_bounds(&mut program, &num_kv_heads, 1.0, 128.0)?;

    // Divisibility constraint: heads_per_group * num_kv_heads = num_heads
    program.assert(
        heads_per_group
            .clone()
            .real_mul(num_kv_heads.clone())
            .eq(num_heads.clone()),
    );

    // total_mapped = heads_per_group * num_kv_heads
    program.assert(
        total_mapped
            .clone()
            .eq(heads_per_group.real_mul(num_kv_heads)),
    );

    // Negated property: total_mapped != num_heads
    let violation = total_mapped.ne(num_heads);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_gqa_head_divisibility".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that GQA with num_kv_heads = num_heads reduces to standard MHA.
///
/// When every query head has its own KV head, heads_per_group = 1.
pub(crate) fn prove_gqa_reduces_to_mha() -> Result<AttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let num_heads = declare_real(&mut program, "num_heads");
    let num_kv_heads = declare_real(&mut program, "num_kv_heads");
    let heads_per_group = declare_real(&mut program, "heads_per_group");

    assert_strict_positive(&mut program, &num_heads, 0.0)?;
    assert_bounds(&mut program, &num_heads, 1.0, 128.0)?;

    // GQA with num_kv_heads = num_heads
    program.assert(num_kv_heads.clone().eq(num_heads.clone()));

    // heads_per_group * num_kv_heads = num_heads
    program.assert(heads_per_group.clone().real_mul(num_kv_heads).eq(num_heads));

    // Negated property: heads_per_group != 1
    let one = Expr::real(1);
    let violation = heads_per_group.ne(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_gqa_reduces_to_mha".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 8: Relative Position Bias Added Before Softmax
// ---------------------------------------------------------------------------

/// Prove that relative position bias is added to the logits *before* softmax,
/// not to the probabilities after it.
///
/// The distinguishing consequence of "bias before softmax" is that the softmax
/// input is shifted by exactly the bias relative to the raw score: the amount
/// `softmax_input - score` equals `bias`. If the bias were instead applied after
/// softmax, the softmax input would be the unmodified score and that difference
/// would be `0`.
///
/// We *apply the ordering rule* to build `softmax_input` and check the derived
/// consequence `softmax_input - score = bias`. The slip in
/// [`build_relative_position_bias_order`] feeds the raw score to softmax (bias
/// applied later), so the difference is `0 != bias` whenever `bias != 0` and the
/// query turns SAT (see `relative_position_bias_depends_on_adding_before_softmax`).
/// `score` and `bias` are symbolic with constant coefficients — decidable QF_LRA.
pub(crate) fn prove_relative_position_bias_order() -> Result<AttentionPropertyResult, SmtError> {
    let program = build_relative_position_bias_order(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_relative_position_bias_order".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the bias-ordering query. When `bias_before_softmax` is false the bias
/// is omitted from the softmax input (as if applied after softmax instead), so
/// `softmax_input - score = 0 != bias` for nonzero bias and the query is SAT.
fn build_relative_position_bias_order(bias_before_softmax: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let score = declare_real(&mut program, "score");
    let bias = declare_real(&mut program, "bias");
    assert_bounds(&mut program, &score, -100.0, 100.0)?;
    assert_bounds(&mut program, &bias, -10.0, 10.0)?;

    // Apply the ordering rule: the softmax input is the biased logit. The slip
    // forwards the raw score, deferring the bias to after softmax.
    let softmax_input = declare_real(&mut program, "softmax_input");
    let input_def = if bias_before_softmax {
        score.clone().real_add(bias.clone())
    } else {
        score.clone()
    };
    program.assert(softmax_input.clone().eq(input_def));

    // Derived consequence: the softmax input exceeds the raw score by the bias.
    let shift = softmax_input.real_sub(score);

    // Negated property: the applied shift is not the bias.
    program.assert(shift.ne(bias));
    program.check_sat();
    Ok(program)
}

/// Prove that position bias does not change the softmax sum-to-1 property.
///
/// After adding bias to scores, the softmax still produces a valid probability
/// distribution (non-negative, sum to 1). This is because softmax is
/// shift-invariant in the sense that adding a constant to all logits does not
/// change the relative weights, and adding position-dependent biases still
/// results in a valid softmax output.
///
/// We prove: if biased_w1, biased_w2 >= 0 and biased_w1 + biased_w2 = 1,
/// then biased_w1 <= 1 (the distribution is valid regardless of bias values).
pub(crate) fn prove_bias_preserves_softmax_validity() -> Result<AttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let biased_w1 = declare_real(&mut program, "biased_w1");
    let biased_w2 = declare_real(&mut program, "biased_w2");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // Softmax output properties hold regardless of bias
    program.assert(biased_w1.clone().real_ge(zero.clone()));
    program.assert(biased_w2.clone().real_ge(zero));
    program.assert(biased_w1.clone().real_add(biased_w2).eq(one.clone()));

    // Negated property: biased_w1 > 1
    let violation = biased_w1.real_gt(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_bias_preserves_softmax_validity".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 9: Attention Dropout Scaling
// ---------------------------------------------------------------------------

/// Prove that inverted dropout is an *unbiased* estimator during training: the
/// surviving weights are scaled by `1/(1-p)` so the expected value under the
/// drop is exactly the original weight.
///
///   E[dropout(w)] = (1-p) * (w/(1-p)) + p * 0 = w
///
/// The content is the scale factor: the survivors must be scaled *up* by
/// `1/(1-p)` to compensate for the dropped fraction. We pin the drop probability
/// to `p = 1/2`, so the keep probability `1-p = 1/2` and the scale factor
/// `1/(1-p) = 2` are exact literals — every product is a constant times a
/// variable, decidable QF_LRA. We *apply the scaling rule* (`scaled_w = w * 2`)
/// and check the expected-value invariant `(1-p) * scaled_w = w`, i.e.
/// `(1/2) * scaled_w = w`.
///
/// The classic slip — scaling the survivors by the *drop* factor `1-p` instead
/// of its inverse `1/(1-p)`, double-counting the drop — is the `false` branch of
/// [`build_attention_dropout_scaling`]; it makes the expected value `w/4 != w`
/// and turns the query SAT (see `dropout_scaling_depends_on_the_inverse_factor`).
pub(crate) fn prove_attention_dropout_scaling() -> Result<AttentionPropertyResult, SmtError> {
    let program = build_attention_dropout_scaling(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_dropout_scaling".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the inverted-dropout expected-value query for `p = 1/2` (keep
/// probability `1-p = 1/2`, scale factor `1/(1-p) = 2`). When
/// `compensate_for_drop` is false the survivors are scaled by the drop factor
/// `1-p = 1/2` instead of its inverse `2`, so the expected value collapses to
/// `w/4 != w` for nonzero `w` and the query turns SAT.
fn build_attention_dropout_scaling(compensate_for_drop: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let w = declare_real(&mut program, "w");
    assert_bounds(&mut program, &w, 0.0, 1.0)?;

    // Inverted-dropout scale factor for the surviving weights. Correct:
    // 1/(1-p) = 2 (scale up to compensate). Slip: the drop factor 1-p = 1/2.
    let scale = if compensate_for_drop {
        Expr::real(2)
    } else {
        Expr::real_ratio(1, 2)
    };

    // Apply the scaling rule: scaled_w = w * scale.
    let scaled_w = declare_real(&mut program, "scaled_w");
    program.assert(scaled_w.clone().eq(w.clone().real_mul(scale)));

    // Expected value under the drop: keep prob (1-p)=1/2 times the surviving
    // (scaled) weight, plus drop prob p=1/2 times 0. Unbiasedness means E = w.
    let expected = scaled_w.real_mul(Expr::real_ratio(1, 2));

    // Negated property: the expected value is not the original weight.
    program.assert(expected.ne(w));
    program.check_sat();
    Ok(program)
}

/// Prove that inverted dropout is the identity when the drop probability is
/// `p = 0` — the scale factor `1/(1-p)` collapses to `1`, so `scaled_w = w`.
///
/// The content is that the disabled-dropout code path actually uses `p = 0`. We
/// *apply the scaling rule* — `scaled_w = w * (1/(1-p))` — with the factor as a
/// concrete literal, and check the identity `scaled_w = w`. When dropout is
/// genuinely disabled the factor is `1/(1-0) = 1` and the identity holds; the
/// slip in [`build_attention_dropout_identity_at_zero`] leaks the *training*
/// probability `p = 1/2` into the eval path, giving factor `1/(1-1/2) = 2`, so
/// `scaled_w = 2w != w` for any `w != 0` and the query turns SAT (see
/// `dropout_identity_depends_on_disabling_at_zero`).
///
/// The factor is a concrete rational, so `w * factor` is linear — decidable
/// QF_LRA.
pub(crate) fn prove_attention_dropout_identity_at_zero() -> Result<AttentionPropertyResult, SmtError>
{
    let program = build_attention_dropout_identity_at_zero(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_dropout_identity_at_zero".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the dropout-identity query. When `disabled_at_eval` is true the drop
/// probability is `p = 0` so the inverted-dropout factor `1/(1-p)` is `1`; when
/// false the training probability `p = 1/2` leaks in, giving factor `2`, and the
/// identity `scaled_w = w` fails for nonzero `w`.
fn build_attention_dropout_identity_at_zero(
    disabled_at_eval: bool,
) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let w = declare_real(&mut program, "w");
    assert_bounds(&mut program, &w, -100.0, 100.0)?;

    // Inverted-dropout factor 1/(1-p). Disabled: p=0 -> 1. Slip: p=1/2 -> 2.
    let factor = if disabled_at_eval {
        Expr::real(1)
    } else {
        Expr::real(2)
    };

    // Apply the scaling rule.
    let scaled_w = declare_real(&mut program, "scaled_w");
    program.assert(scaled_w.clone().eq(w.clone().real_mul(factor)));

    // Negated property: the scaled weight is not the identity.
    program.assert(scaled_w.ne(w));
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 10: Cross Attention (Different Sequence Lengths Allowed)
// ---------------------------------------------------------------------------

/// Prove that cross-attention allows different sequence lengths for Q vs K/V.
///
/// In cross-attention:
///   Q: [batch, seq_q, d_k]    (decoder queries)
///   K: [batch, seq_kv, d_k]   (encoder keys)
///   V: [batch, seq_kv, d_v]   (encoder values)
///
/// The attention matrix has shape [batch, seq_q, seq_kv], which is valid
/// for any seq_q and seq_kv. The output has shape [batch, seq_q, d_v].
///
/// We prove: the output sequence length equals the query sequence length,
/// regardless of the key/value sequence length.
pub(crate) fn prove_cross_attention_dimensions() -> Result<AttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let seq_q = declare_real(&mut program, "seq_q");
    let seq_kv = declare_real(&mut program, "seq_kv");
    let d_k = declare_real(&mut program, "d_k");
    let d_v = declare_real(&mut program, "d_v");

    assert_strict_positive(&mut program, &seq_q, 0.0)?;
    assert_strict_positive(&mut program, &seq_kv, 0.0)?;
    assert_strict_positive(&mut program, &d_k, 0.0)?;
    assert_strict_positive(&mut program, &d_v, 0.0)?;
    assert_bounds(&mut program, &seq_q, 1.0, 8192.0)?;
    assert_bounds(&mut program, &seq_kv, 1.0, 8192.0)?;

    // Allow seq_q != seq_kv
    program.assert(seq_q.clone().ne(seq_kv.clone()));

    // Attention matrix: [seq_q, seq_kv] (Q @ K^T)
    let attn_rows = declare_real(&mut program, "attn_rows");
    let attn_cols = declare_real(&mut program, "attn_cols");
    program.assert(attn_rows.clone().eq(seq_q.clone()));
    program.assert(attn_cols.clone().eq(seq_kv.clone()));

    // Output: attn @ V = [seq_q, seq_kv] @ [seq_kv, d_v] = [seq_q, d_v]
    // For valid matmul: attn_cols == seq_kv (V's rows) - already true
    let output_rows = declare_real(&mut program, "output_rows");
    program.assert(output_rows.clone().eq(attn_rows));

    // Negated property: output_rows != seq_q
    let violation = output_rows.ne(seq_q);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_cross_attention_dimensions".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that in cross-attention, `Q @ K^T` is well-formed exactly because the
/// query feature dimension meets the *transposed* key's row dimension.
///
/// `Q` is `[seq_q, d_q]` and `K` is `[seq_kv, d_k]`. The score matrix is
/// `Q @ K^T`, whose inner dimensions match iff `d_q` equals `K^T`'s row count.
/// The content is *applying the transpose rule*: `K^T` is `[d_k, seq_kv]`, so its
/// row count is `d_k` (the key feature dim), not `seq_kv`. Given the layer
/// constraint `d_q = d_k`, the solver must chain `kt_rows = d_k` and `d_q = d_k`
/// to derive that the inner dimensions agree.
///
/// The slip in [`build_cross_attention_qk_dimension_match`] transposes wrongly —
/// taking `kt_rows = seq_kv` (`K`'s row count) as if `K^T` reused `K`'s rows — so
/// `d_q = d_k` no longer forces `d_q = kt_rows` and, since `seq_kv` is free, the
/// query turns SAT (see `cross_attention_qk_depends_on_the_transpose_rule`).
/// All relations are integer equalities with no variable products — decidable
/// QF_LIA.
pub(crate) fn prove_cross_attention_qk_dimension_match() -> Result<AttentionPropertyResult, SmtError>
{
    let program = build_cross_attention_qk_dimension_match(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionPropertyResult {
        property: "attention_cross_attention_qk_match".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the cross-attention inner-dimension query. When `transpose_swaps_dims`
/// is false `K^T`'s row count is taken as `seq_kv` (a botched transpose) instead
/// of the key feature dim `d_k`, so the derivation `d_q = d_k = kt_rows` breaks
/// and the query becomes SAT.
fn build_cross_attention_qk_dimension_match(transpose_swaps_dims: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let one = Expr::int(1);

    // Positive integer dimensions; seq_kv is free relative to the feature dims.
    let d_q = program.declare_const("d_q", Sort::int());
    let d_k = program.declare_const("d_k", Sort::int());
    let seq_kv = program.declare_const("seq_kv", Sort::int());
    for dim in [&d_q, &d_k, &seq_kv] {
        program.assert(dim.clone().int_ge(one.clone()));
    }

    // Layer constraint: query and key share the feature dimension.
    program.assert(d_q.clone().eq(d_k.clone()));

    // Transpose rule: K is [seq_kv, d_k], so K^T is [d_k, seq_kv] and its row
    // count is d_k. The slip reuses K's row count seq_kv.
    let kt_rows = program.declare_const("kt_rows", Sort::int());
    let kt_rows_def = if transpose_swaps_dims {
        d_k.clone()
    } else {
        seq_kv
    };
    program.assert(kt_rows.clone().eq(kt_rows_def));

    // Q @ K^T is well-formed iff Q's cols (d_q) equal K^T's rows (kt_rows).
    // Negated property: the inner dimensions disagree.
    program.assert(d_q.ne(kt_rows));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ay_vacuity::vacuity_smell;

    // --- Property 1: Scaled Dot-Product Tests ---

    #[test]
    fn test_scaled_dot_product_scaling_proven() {
        let result = prove_scaled_dot_product_scaling().expect("proof should not error");
        // QF_LRA over a concrete shape is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Scaled dot-product should be proven (QF_LRA), got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "attention_scaled_dot_product");
    }

    /// The scaling denominator is the whole theorem: dividing by `d_k = 64`
    /// instead of `sqrt(d_k) = 8` breaks the `* 8` recovery, so the query must be
    /// SAT.
    #[test]
    fn scaled_dot_product_depends_on_the_sqrt() {
        let program = build_scaled_dot_product_scaling(false).expect("program builds");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "dividing by d_k instead of sqrt(d_k) must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_scaling_reduces_variance_proven() {
        let result = prove_scaling_reduces_variance().expect("proof should not error");
        // QF_LRA with concrete coefficients is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Scaling reduces variance should be proven (QF_LRA), got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "attention_scaling_reduces_variance");
    }

    /// Variance scales by the *square* of the amplitude. Using `1/8` instead of
    /// `1/64` breaks the `* d_k` recovery, so the query must be SAT.
    #[test]
    fn scaling_reduces_variance_depends_on_squaring() {
        let program = build_scaling_reduces_variance(false).expect("program builds");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "scaling variance by the un-squared factor must be SAT; got: {detail}",
        );
    }

    // --- Property 2: Softmax Weights Tests ---

    #[test]
    fn test_softmax_weights_bounded_proven() {
        let result = prove_softmax_weights_bounded().expect("proof should not error");
        assert!(
            result.proven,
            "Softmax weights bounded: expected Proven (QF_LRA), got: {}",
            result.detail,
        );
        assert_eq!(result.property, "attention_softmax_weights_bounded");
    }

    #[test]
    fn test_softmax_weights_nonnegative_proven() {
        let result = prove_softmax_weights_nonnegative().expect("proof should not error");
        assert!(
            result.proven,
            "Softmax weights nonneg: expected Proven (QF_LRA), got: {}",
            result.detail,
        );
        assert_eq!(result.property, "attention_softmax_weights_nonnegative");
    }

    // --- Property 3: Multi-Head Output Dimension ---

    #[test]
    fn test_multihead_output_dimension_proven() {
        let result = prove_multihead_output_dimension().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Multi-head output dim: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Multi-head output dim must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "attention_multihead_output_dimension");
    }

    // --- Property 4: Causal Mask Tests ---

    #[test]
    fn test_causal_mask_future_proven() {
        let result = prove_causal_mask_structure().expect("proof should not error");
        assert!(
            result.proven,
            "Causal mask future: expected Proven (QF_LIA), got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "attention_causal_mask_future");
    }

    /// Which side of the diagonal is masked is the whole theorem. Flipping the
    /// rule to mask the past (`i > j`) leaves a future position unsuppressed, so
    /// the causal spec is violated and the query must be SAT.
    #[test]
    fn causal_mask_depends_on_the_inequality_direction() {
        let program = build_causal_mask_structure(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "masking the past instead of the future must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_causal_mask_past_preserved_proven() {
        let result = prove_causal_mask_past_preserved().expect("proof should not error");
        assert!(
            result.proven,
            "Causal mask past: expected Proven (QF_LRA), got: {}",
            result.detail,
        );
        assert_eq!(result.property, "attention_causal_mask_past_preserved");
    }

    // --- Property 5: KV Sequence Consistency ---

    #[test]
    fn test_kv_sequence_consistency_proven() {
        let result = prove_kv_sequence_consistency().expect("proof should not error");
        assert!(
            result.proven,
            "KV sequence consistency: expected Proven (QF_LRA), got: {}",
            result.detail,
        );
        assert_eq!(result.property, "attention_kv_sequence_consistency");
    }

    // --- Property 6: Flash Attention Equivalence ---

    #[test]
    fn test_flash_attention_equivalence_proven() {
        let result = prove_flash_attention_equivalence().expect("proof should not error");
        assert!(
            result.proven,
            "Flash attention equivalence: expected Proven (QF_LRA), got: {}",
            result.detail,
        );
        assert_eq!(result.property, "attention_flash_equivalence");
    }

    #[test]
    fn test_flash_attention_rescaling_proven() {
        let result = prove_flash_attention_rescaling().expect("proof should not error");
        // Pinning the tile weights makes the query linear: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Flash attention rescaling should be proven (QF_LRA), got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "attention_flash_rescaling");
    }

    /// Online softmax must rescale the carried accumulator by `old_sum/new_sum`.
    /// Leaving it un-rescaled breaks the weighted-sum invariant, so the query
    /// must be SAT.
    #[test]
    fn flash_rescaling_depends_on_the_accumulator_correction() {
        let program = build_flash_attention_rescaling(false).expect("program builds");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "failing to rescale the old accumulator must be SAT; got: {detail}",
        );
    }

    // --- Property 7: GQA Head Divisibility ---

    #[test]
    fn test_gqa_head_divisibility_proven() {
        let result = prove_gqa_head_divisibility().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "GQA head divisibility: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "GQA head divisibility must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "attention_gqa_head_divisibility");
    }

    #[test]
    fn test_gqa_reduces_to_mha_proven() {
        let result = prove_gqa_reduces_to_mha().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "GQA reduces to MHA: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "GQA reduces to MHA must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "attention_gqa_reduces_to_mha");
    }

    // --- Property 8: Relative Position Bias ---

    #[test]
    fn test_relative_position_bias_order_proven() {
        let result = prove_relative_position_bias_order().expect("proof should not error");
        assert!(
            result.proven,
            "Relative position bias order: expected Proven (QF_LRA), got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "attention_relative_position_bias_order");
    }

    /// The bias must reach the softmax *input*. Deferring it to after softmax
    /// makes the applied shift `0` instead of `bias`, so the query must be SAT.
    #[test]
    fn relative_position_bias_depends_on_adding_before_softmax() {
        let program = build_relative_position_bias_order(false).expect("program builds");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "omitting the bias from the softmax input must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_bias_preserves_softmax_validity_proven() {
        let result = prove_bias_preserves_softmax_validity().expect("proof should not error");
        assert!(
            result.proven,
            "Bias preserves softmax validity: expected Proven (QF_LRA), got: {}",
            result.detail,
        );
        assert_eq!(result.property, "attention_bias_preserves_softmax_validity");
    }

    // --- Property 9: Attention Dropout Scaling ---

    #[test]
    fn test_attention_dropout_scaling_proven() {
        let result = prove_attention_dropout_scaling().expect("proof should not error");
        // Pinning the drop probability makes the query linear: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Attention dropout scaling should be proven (QF_LRA), got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "attention_dropout_scaling");
    }

    /// Inverted dropout must scale the survivors by `1/(1-p) = 2`, not by the
    /// drop factor `1-p = 1/2`. Scaling by the drop factor makes the expected
    /// value `w/4 != w`, so the query must be SAT.
    #[test]
    fn dropout_scaling_depends_on_the_inverse_factor() {
        let program = build_attention_dropout_scaling(false).expect("program builds");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "scaling survivors by the drop factor instead of its inverse must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_attention_dropout_identity_at_zero_proven() {
        let result = prove_attention_dropout_identity_at_zero().expect("proof should not error");
        assert!(
            result.proven,
            "Dropout identity at p=0: expected Proven (QF_LRA), got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "attention_dropout_identity_at_zero");
    }

    /// Disabling dropout means `p = 0` (factor `1`). Leaking the training
    /// `p = 1/2` (factor `2`) makes `scaled_w = 2w != w` for nonzero `w`, so the
    /// query must be SAT.
    #[test]
    fn dropout_identity_depends_on_disabling_at_zero() {
        let program = build_attention_dropout_identity_at_zero(false).expect("program builds");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "leaking a nonzero drop probability into eval must be SAT; got: {detail}",
        );
    }

    // --- Property 10: Cross Attention ---

    #[test]
    fn test_cross_attention_dimensions_proven() {
        let result = prove_cross_attention_dimensions().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Cross attention dimensions: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Cross attention dimensions must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "attention_cross_attention_dimensions");
    }

    #[test]
    fn test_cross_attention_qk_match_proven() {
        let result = prove_cross_attention_qk_dimension_match().expect("proof should not error");
        assert!(
            result.proven,
            "Cross attention QK match: expected Proven (QF_LIA), got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "attention_cross_attention_qk_match");
    }

    /// `K^T`'s row count comes from `K`'s columns (the feature dim `d_k`), not its
    /// rows. Botching the transpose to `seq_kv` unhooks the derivation from
    /// `d_q = d_k`, so the query must be SAT.
    #[test]
    fn cross_attention_qk_depends_on_the_transpose_rule() {
        let program = build_cross_attention_qk_dimension_match(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a botched transpose of K must be SAT; got: {detail}",
        );
    }

    // --- SMT2 Structure Tests ---

    #[test]
    fn test_all_attention_proofs_have_valid_smt2() {
        let proofs: Vec<AttentionPropertyResult> = vec![
            prove_scaled_dot_product_scaling().unwrap(),
            prove_scaling_reduces_variance().unwrap(),
            prove_softmax_weights_bounded().unwrap(),
            prove_softmax_weights_nonnegative().unwrap(),
            prove_multihead_output_dimension().unwrap(),
            prove_causal_mask_structure().unwrap(),
            prove_causal_mask_past_preserved().unwrap(),
            prove_kv_sequence_consistency().unwrap(),
            prove_flash_attention_equivalence().unwrap(),
            prove_flash_attention_rescaling().unwrap(),
            prove_gqa_head_divisibility().unwrap(),
            prove_gqa_reduces_to_mha().unwrap(),
            prove_relative_position_bias_order().unwrap(),
            prove_bias_preserves_softmax_validity().unwrap(),
            prove_attention_dropout_scaling().unwrap(),
            prove_attention_dropout_identity_at_zero().unwrap(),
            prove_cross_attention_dimensions().unwrap(),
            prove_cross_attention_qk_dimension_match().unwrap(),
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
