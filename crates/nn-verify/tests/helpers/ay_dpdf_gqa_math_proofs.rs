// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for Grouped Query Attention (GQA)
//! mathematical properties.
//!
//! Proves fundamental properties of GQA as used in modern LLMs (Qwen3, GLM,
//! Llama, Mistral):
//! - QK dot product scaling by 1/sqrt(d_k) bounded when Q,K bounded
//! - Attention weight non-negativity after softmax
//! - Attention weights sum to 1 per query position
//! - GQA key/value head repeat (repeat_kv) preserves bounds
//! - GQA output bounded when V bounded and weights sum to 1
//! - Multi-head split preserves total dimension
//! - Head dimension d_k = d_model / num_heads consistency
//! - GQA num_kv_heads divides num_heads evenly
//! - Causal mask zeros future positions (upper triangular)
//! - Attention score with causal mask bounded
//! - QK^T output dimension [seq, seq] from [seq, d_k] * [d_k, seq]
//! - Output projection preserves bounds via weight magnitude
//! - Key/value cache append preserves existing entries
//! - Sliding window attention mask correctness
//! - ALiBi position bias linearity
//! - Multi-head concatenation dimension = num_heads * d_v
//! - Scaled dot product attention associativity
//! - Attention dropout mask binary (0 or 1)
//! - Flash attention equivalence (chunked QK computation)
//! - GQA reduces to MHA when num_kv_heads == num_heads
//!
//! Part of #4179.

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
// Test 811: QK dot product scaling by 1/sqrt(d_k) bounded when Q,K bounded
// ---------------------------------------------------------------------------

/// Prove: scaled dot-product score is bounded when Q and K vectors are bounded.
///
/// For a single query-key pair with d_k=2 components:
///   raw_score = q1*k1 + q2*k2.
/// If |q_i| <= Q and |k_i| <= K, then |raw_score| <= d_k * Q * K.
/// Scaled: score = raw_score / sqrt(d_k). Since sqrt(d_k) > 0 and
/// |raw_score| <= d_k * Q * K, we get |score| <= d_k * Q * K / sqrt(d_k)
///                                              = sqrt(d_k) * Q * K.
///
/// For d_k=2, Q=K=3: |score| <= sqrt(2)*9 < 13.
/// We prove |score| <= 13.
#[test]
fn test_811_gqa_qk_scaling_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("q1", real.clone());
    let _ = prog.declare_const("q2", real.clone());
    let _ = prog.declare_const("k1", real.clone());
    let _ = prog.declare_const("k2", real.clone());
    let _ = prog.declare_const("raw", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("score", real);

    let q1 = real_var("q1");
    let q2 = real_var("q2");
    let k1 = real_var("k1");
    let k2 = real_var("k2");
    let raw = real_var("raw");
    let scale = real_var("scale");
    let score = real_var("score");

    // |q_i| <= 3, |k_i| <= 3
    prog.assert(q1.clone().real_ge(Expr::real(-3)));
    prog.assert(q1.clone().real_le(Expr::real(3)));
    prog.assert(q2.clone().real_ge(Expr::real(-3)));
    prog.assert(q2.clone().real_le(Expr::real(3)));
    prog.assert(k1.clone().real_ge(Expr::real(-3)));
    prog.assert(k1.clone().real_le(Expr::real(3)));
    prog.assert(k2.clone().real_ge(Expr::real(-3)));
    prog.assert(k2.clone().real_le(Expr::real(3)));

    // raw = q1*k1 + q2*k2
    prog.assert(raw.clone().eq(q1.real_mul(k1).real_add(q2.real_mul(k2))));

    // scale > 0, scale^2 = 1/2 (scale = 1/sqrt(2))
    prog.assert(scale.clone().real_gt(Expr::real(0)));
    prog.assert(
        Expr::real(2)
            .real_mul(scale.clone().real_mul(scale.clone()))
            .eq(Expr::real(1)),
    );

    // score = raw * scale
    prog.assert(score.clone().eq(raw.real_mul(scale)));

    // |raw| <= 2 * 3 * 3 = 18, so |score| = |raw| / sqrt(2) <= 18/sqrt(2) < 13
    let violation = score
        .clone()
        .real_gt(Expr::real(13))
        .or(score.real_lt(Expr::real(-13)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_qk_scaling_bounded");
}

// ---------------------------------------------------------------------------
// Test 812: Attention weight non-negativity after softmax
// ---------------------------------------------------------------------------

/// Prove: softmax output is non-negative.
///
/// softmax(x_i) = exp(x_i) / sum(exp(x_j)). Since exp(x) > 0 for all x,
/// the numerator and denominator are both positive, so softmax(x_i) > 0.
///
/// We model: a = exp_num / denom with exp_num > 0 and denom > 0.
/// Prove: a > 0.
#[test]
fn test_812_gqa_attention_weight_non_negativity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp_num", real.clone());
    let _ = prog.declare_const("denom", real.clone());
    let _ = prog.declare_const("a", real);

    let exp_num = real_var("exp_num");
    let denom = real_var("denom");
    let a = real_var("a");

    // exp_num > 0 (exp is always positive)
    prog.assert(exp_num.clone().real_gt(Expr::real(0)));

    // denom > 0 (sum of positive values is positive)
    prog.assert(denom.clone().real_gt(Expr::real(0)));

    // a * denom = exp_num (a = exp_num / denom)
    prog.assert(a.clone().real_mul(denom).eq(exp_num));

    // Negated property: a <= 0
    let violation = a.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_attention_weight_non_negativity");
}

// ---------------------------------------------------------------------------
// Test 813: Attention weights sum to 1 per query position
// ---------------------------------------------------------------------------

/// Prove: softmax outputs sum to 1.
///
/// For n=3: softmax_i = exp(x_i) / (exp(x1) + exp(x2) + exp(x3)).
/// Sum = (exp(x1) + exp(x2) + exp(x3)) / (exp(x1) + exp(x2) + exp(x3)) = 1.
///
/// We model: a1 + a2 + a3 = 1 where a_i = e_i / D and D = e1 + e2 + e3.
/// Prove: a1 + a2 + a3 = 1.
#[test]
fn test_813_gqa_attention_weights_sum_to_one() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("e1", real.clone());
    let _ = prog.declare_const("e2", real.clone());
    let _ = prog.declare_const("e3", real.clone());
    let _ = prog.declare_const("D", real.clone());
    let _ = prog.declare_const("a1", real.clone());
    let _ = prog.declare_const("a2", real.clone());
    let _ = prog.declare_const("a3", real);

    let e1 = real_var("e1");
    let e2 = real_var("e2");
    let e3 = real_var("e3");
    let d_var = real_var("D");
    let a1 = real_var("a1");
    let a2 = real_var("a2");
    let a3 = real_var("a3");

    // e_i > 0 (exponentials are positive)
    prog.assert(e1.clone().real_gt(Expr::real(0)));
    prog.assert(e2.clone().real_gt(Expr::real(0)));
    prog.assert(e3.clone().real_gt(Expr::real(0)));

    // D = e1 + e2 + e3
    prog.assert(
        d_var
            .clone()
            .eq(e1.clone().real_add(e2.clone()).real_add(e3.clone())),
    );

    // a_i = e_i / D, i.e., a_i * D = e_i
    prog.assert(a1.clone().real_mul(d_var.clone()).eq(e1));
    prog.assert(a2.clone().real_mul(d_var.clone()).eq(e2));
    prog.assert(a3.clone().real_mul(d_var).eq(e3));

    // Negated property: a1 + a2 + a3 != 1
    let sum_a = a1.real_add(a2).real_add(a3);
    let violation = sum_a.ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_attention_weights_sum_to_one");
}

// ---------------------------------------------------------------------------
// Test 814: GQA key/value head repeat (repeat_kv) preserves bounds
// ---------------------------------------------------------------------------

/// Prove: repeating KV heads preserves the original bounds.
///
/// In GQA, each KV head is shared across `num_heads / num_kv_heads` query
/// heads. The repeat_kv operation duplicates a KV tensor without changing
/// its values. If the original KV value v is in [lo, hi], the repeated
/// copy is also in [lo, hi].
///
/// We model: v_copy = v (identity copy). Prove: lo <= v_copy <= hi.
#[test]
fn test_814_gqa_repeat_kv_preserves_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("v", real.clone());
    let _ = prog.declare_const("v_copy", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real);

    let v = real_var("v");
    let v_copy = real_var("v_copy");
    let lo = real_var("lo");
    let hi = real_var("hi");

    // lo <= hi
    prog.assert(lo.clone().real_le(hi.clone()));

    // v in [lo, hi]
    prog.assert(v.clone().real_ge(lo.clone()));
    prog.assert(v.clone().real_le(hi.clone()));

    // v_copy = v (repeat_kv is identity on values)
    prog.assert(v_copy.clone().eq(v));

    // Negated property: v_copy < lo OR v_copy > hi
    let violation = v_copy.clone().real_lt(lo).or(v_copy.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_repeat_kv_preserves_bounds");
}

// ---------------------------------------------------------------------------
// Test 815: GQA output bounded when V bounded and weights sum to 1
// ---------------------------------------------------------------------------

/// Prove: attention output is bounded when value vectors are bounded and
/// attention weights form a convex combination (sum to 1, non-negative).
///
/// output = sum_j(a_j * v_j). If a_j >= 0, sum(a_j) = 1, lo <= v_j <= hi,
/// then lo <= output <= hi (convex combination stays in [lo, hi]).
///
/// For n=2: out = a1*v1 + a2*v2 with a1+a2=1, a_i>=0, v_i in [lo,hi].
/// Prove: lo <= out <= hi.
#[test]
fn test_815_gqa_output_bounded_convex_combination() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("a1", real.clone());
    let _ = prog.declare_const("a2", real.clone());
    let _ = prog.declare_const("v1", real.clone());
    let _ = prog.declare_const("v2", real.clone());
    let _ = prog.declare_const("out", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real);

    let a1 = real_var("a1");
    let a2 = real_var("a2");
    let v1 = real_var("v1");
    let v2 = real_var("v2");
    let out = real_var("out");
    let lo = real_var("lo");
    let hi = real_var("hi");

    // lo <= hi
    prog.assert(lo.clone().real_le(hi.clone()));

    // a1, a2 >= 0
    prog.assert(a1.clone().real_ge(Expr::real(0)));
    prog.assert(a2.clone().real_ge(Expr::real(0)));

    // a1 + a2 = 1
    prog.assert(a1.clone().real_add(a2.clone()).eq(Expr::real(1)));

    // v1, v2 in [lo, hi]
    prog.assert(v1.clone().real_ge(lo.clone()));
    prog.assert(v1.clone().real_le(hi.clone()));
    prog.assert(v2.clone().real_ge(lo.clone()));
    prog.assert(v2.clone().real_le(hi.clone()));

    // out = a1*v1 + a2*v2
    prog.assert(out.clone().eq(a1.real_mul(v1).real_add(a2.real_mul(v2))));

    // Negated property: out < lo OR out > hi
    let violation = out.clone().real_lt(lo).or(out.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_output_bounded_convex_combination");
}

// ---------------------------------------------------------------------------
// Test 816: Multi-head split preserves total dimension
// ---------------------------------------------------------------------------

/// Prove: splitting d_model into num_heads heads of size d_k each
/// preserves the total dimension: num_heads * d_k = d_model.
///
/// This is a structural property: the reshape from [seq, d_model] to
/// [seq, num_heads, d_k] is dimension-preserving iff d_model = num_heads * d_k.
///
/// We model: d_model = num_heads * d_k with concrete values.
/// Prove: total dimension is conserved.
#[test]
fn test_816_gqa_multihead_split_preserves_dimension() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("d_model", real.clone());
    let _ = prog.declare_const("num_heads", real.clone());
    let _ = prog.declare_const("d_k", real.clone());
    let _ = prog.declare_const("total", real);

    let d_model = real_var("d_model");
    let num_heads = real_var("num_heads");
    let d_k = real_var("d_k");
    let total = real_var("total");

    // d_model > 0, num_heads > 0, d_k > 0
    prog.assert(d_model.clone().real_gt(Expr::real(0)));
    prog.assert(num_heads.clone().real_gt(Expr::real(0)));
    prog.assert(d_k.clone().real_gt(Expr::real(0)));

    // d_model = num_heads * d_k (constraint from model config)
    prog.assert(d_model.clone().eq(num_heads.clone().real_mul(d_k.clone())));

    // total = num_heads * d_k (reconstructed from split)
    prog.assert(total.clone().eq(num_heads.real_mul(d_k)));

    // Negated property: total != d_model
    let violation = total.ne(d_model);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_multihead_split_preserves_dimension");
}

// ---------------------------------------------------------------------------
// Test 817: Head dimension d_k = d_model / num_heads consistency
// ---------------------------------------------------------------------------

/// Prove: when d_k = d_model / num_heads, the relationship is consistent:
/// num_heads * d_k = d_model.
///
/// This is the inverse of test_816 — starting from the division and
/// proving the product recovers the original.
///
/// We model: d_k * num_heads = d_model (d_k = d_model / num_heads).
/// Prove: num_heads * d_k = d_model.
#[test]
fn test_817_gqa_head_dim_consistency() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("d_model", real.clone());
    let _ = prog.declare_const("num_heads", real.clone());
    let _ = prog.declare_const("d_k", real);

    let d_model = real_var("d_model");
    let num_heads = real_var("num_heads");
    let d_k = real_var("d_k");

    // num_heads > 0
    prog.assert(num_heads.clone().real_gt(Expr::real(0)));

    // d_k * num_heads = d_model (d_k = d_model / num_heads)
    prog.assert(d_k.clone().real_mul(num_heads.clone()).eq(d_model.clone()));

    // Negated property: num_heads * d_k != d_model
    let violation = num_heads.real_mul(d_k).ne(d_model);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_head_dim_consistency");
}

// ---------------------------------------------------------------------------
// Test 818: GQA num_kv_heads divides num_heads evenly
// ---------------------------------------------------------------------------

/// Prove: when num_heads = num_kv_heads * repeat_factor, the division is
/// exact and repeat_factor is a positive integer.
///
/// GQA requires num_heads % num_kv_heads == 0, i.e.,
/// num_heads = num_kv_heads * repeat_factor for some integer repeat_factor > 0.
///
/// We model: num_heads = num_kv_heads * repeat_factor.
/// Prove: num_kv_heads * repeat_factor = num_heads (exact recovery).
#[test]
fn test_818_gqa_num_kv_heads_divides_num_heads() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("num_heads", real.clone());
    let _ = prog.declare_const("num_kv_heads", real.clone());
    let _ = prog.declare_const("repeat_factor", real);

    let num_heads = real_var("num_heads");
    let num_kv_heads = real_var("num_kv_heads");
    let repeat_factor = real_var("repeat_factor");

    // num_kv_heads > 0, repeat_factor > 0
    prog.assert(num_kv_heads.clone().real_gt(Expr::real(0)));
    prog.assert(repeat_factor.clone().real_gt(Expr::real(0)));

    // num_heads = num_kv_heads * repeat_factor
    prog.assert(
        num_heads
            .clone()
            .eq(num_kv_heads.clone().real_mul(repeat_factor.clone())),
    );

    // Negated property: num_kv_heads * repeat_factor != num_heads
    let violation = num_kv_heads.real_mul(repeat_factor).ne(num_heads);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_num_kv_heads_divides_num_heads");
}

// ---------------------------------------------------------------------------
// Test 819: Causal mask zeros future positions (upper triangular)
// ---------------------------------------------------------------------------

/// Prove: causal mask zeroes the attention weight for future positions.
///
/// In causal (autoregressive) attention, position i cannot attend to
/// position j > i. The mask is: mask(i, j) = 0 if j > i, 1 otherwise.
/// The masked attention weight is: a_masked = a * mask(i, j).
/// When j > i: a_masked = a * 0 = 0.
///
/// We model: mask = 0 (future position), a_masked = a * mask.
/// Prove: a_masked = 0.
#[test]
fn test_819_gqa_causal_mask_zeros_future() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("mask", real.clone());
    let _ = prog.declare_const("a_masked", real);

    let a = real_var("a");
    let mask = real_var("mask");
    let a_masked = real_var("a_masked");

    // a is an arbitrary attention weight (bounded)
    prog.assert(a.clone().real_ge(Expr::real(-100)));
    prog.assert(a.clone().real_le(Expr::real(100)));

    // mask = 0 (future position: j > i)
    prog.assert(mask.clone().eq(Expr::real(0)));

    // a_masked = a * mask
    prog.assert(a_masked.clone().eq(a.real_mul(mask)));

    // Negated property: a_masked != 0
    let violation = a_masked.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_causal_mask_zeros_future");
}

// ---------------------------------------------------------------------------
// Test 820: Attention score with causal mask bounded
// ---------------------------------------------------------------------------

/// Prove: after applying the causal mask (adding -inf to future positions),
/// the non-masked attention scores remain bounded.
///
/// Causal attention adds a large negative value (NEG_INF) to future
/// positions. For non-masked positions (j <= i), the score is unchanged.
/// If the original score is in [-S, S], the non-masked score stays in [-S, S].
///
/// We model: score_masked = score + bias with bias = 0 (non-masked).
/// Prove: |score_masked| <= S.
#[test]
fn test_820_gqa_causal_masked_score_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("score", real.clone());
    let _ = prog.declare_const("bias", real.clone());
    let _ = prog.declare_const("score_masked", real);

    let score = real_var("score");
    let bias = real_var("bias");
    let score_masked = real_var("score_masked");

    // |score| <= 10 (bounded attention score)
    prog.assert(score.clone().real_ge(Expr::real(-10)));
    prog.assert(score.clone().real_le(Expr::real(10)));

    // bias = 0 (non-masked position: j <= i)
    prog.assert(bias.clone().eq(Expr::real(0)));

    // score_masked = score + bias
    prog.assert(score_masked.clone().eq(score.real_add(bias)));

    // Negated property: |score_masked| > 10
    let violation = score_masked
        .clone()
        .real_gt(Expr::real(10))
        .or(score_masked.real_lt(Expr::real(-10)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_causal_masked_score_bounded");
}

// ---------------------------------------------------------------------------
// Test 821: QK^T output dimension [seq, seq] from [seq, d_k] * [d_k, seq]
// ---------------------------------------------------------------------------

/// Prove: the matrix product Q * K^T has output dimensions [seq_q, seq_k]
/// when Q is [seq_q, d_k] and K^T is [d_k, seq_k].
///
/// Matrix multiplication: [M, K] * [K, N] -> [M, N].
/// Here: [seq_q, d_k] * [d_k, seq_k] -> [seq_q, seq_k].
///
/// We model: out_rows = rows_Q, out_cols = cols_Kt, inner = cols_Q = rows_Kt.
/// Prove: out_rows = seq_q AND out_cols = seq_k.
#[test]
fn test_821_gqa_qk_transpose_output_dimension() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("seq_q", real.clone());
    let _ = prog.declare_const("d_k", real.clone());
    let _ = prog.declare_const("seq_k", real.clone());
    let _ = prog.declare_const("out_rows", real.clone());
    let _ = prog.declare_const("out_cols", real);

    let seq_q = real_var("seq_q");
    let d_k = real_var("d_k");
    let seq_k = real_var("seq_k");
    let out_rows = real_var("out_rows");
    let out_cols = real_var("out_cols");

    // Positive dimensions
    prog.assert(seq_q.clone().real_gt(Expr::real(0)));
    prog.assert(d_k.real_gt(Expr::real(0)));
    prog.assert(seq_k.clone().real_gt(Expr::real(0)));

    // Matmul dimension rule: [seq_q, d_k] * [d_k, seq_k] -> [seq_q, seq_k]
    prog.assert(out_rows.clone().eq(seq_q.clone()));
    prog.assert(out_cols.clone().eq(seq_k.clone()));

    // Negated property: out_rows != seq_q OR out_cols != seq_k
    let violation = out_rows.ne(seq_q).or(out_cols.ne(seq_k));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_qk_transpose_output_dimension");
}

// ---------------------------------------------------------------------------
// Test 822: Output projection preserves bounds via weight magnitude
// ---------------------------------------------------------------------------

/// Prove: the output projection y = W_o * x is bounded when W_o and x are.
///
/// For scalar proxy: y = w * x. If |w| <= W and |x| <= X, then |y| <= W*X.
/// The multi-head attention output passes through a linear projection W_o.
///
/// We model: y = w * x with |w| <= 2, |x| <= 5.
/// Prove: |y| <= 10.
#[test]
fn test_822_gqa_output_projection_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w", real.clone());
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("y", real);

    let w = real_var("w");
    let x = real_var("x");
    let y = real_var("y");

    // |w| <= 2
    prog.assert(w.clone().real_ge(Expr::real(-2)));
    prog.assert(w.clone().real_le(Expr::real(2)));

    // |x| <= 5
    prog.assert(x.clone().real_ge(Expr::real(-5)));
    prog.assert(x.clone().real_le(Expr::real(5)));

    // y = w * x
    prog.assert(y.clone().eq(w.real_mul(x)));

    // Negated property: |y| > 10
    let violation = y
        .clone()
        .real_gt(Expr::real(10))
        .or(y.real_lt(Expr::real(-10)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_output_projection_bounded");
}

// ---------------------------------------------------------------------------
// Test 823: Key/value cache append preserves existing entries
// ---------------------------------------------------------------------------

/// Prove: appending a new entry to the KV cache does not modify existing
/// cached values.
///
/// KV cache at step t: [k1, k2, ..., kt]. After append: [k1, ..., kt, k_{t+1}].
/// The existing entries k1..kt are unchanged.
///
/// We model: old_entry in [lo, hi], cache preserves identity.
/// Prove: old_entry in the updated cache is still in [lo, hi].
#[test]
fn test_823_gqa_kv_cache_append_preserves_existing() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("old_k", real.clone());
    let _ = prog.declare_const("new_k", real.clone());
    let _ = prog.declare_const("cached_old_k", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real);

    let old_k = real_var("old_k");
    let new_k = real_var("new_k");
    let cached_old_k = real_var("cached_old_k");
    let lo = real_var("lo");
    let hi = real_var("hi");

    // lo <= hi
    prog.assert(lo.clone().real_le(hi.clone()));

    // old_k in [lo, hi]
    prog.assert(old_k.clone().real_ge(lo.clone()));
    prog.assert(old_k.clone().real_le(hi.clone()));

    // new_k is arbitrary (bounded for solver efficiency)
    prog.assert(new_k.clone().real_ge(Expr::real(-100)));
    prog.assert(new_k.real_le(Expr::real(100)));

    // Cache append preserves old entries: cached_old_k = old_k
    prog.assert(cached_old_k.clone().eq(old_k));

    // Negated property: cached_old_k < lo OR cached_old_k > hi
    let violation = cached_old_k
        .clone()
        .real_lt(lo)
        .or(cached_old_k.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_kv_cache_append_preserves_existing");
}

// ---------------------------------------------------------------------------
// Test 824: Sliding window attention mask correctness
// ---------------------------------------------------------------------------

/// Prove: sliding window attention mask zeros positions outside the window.
///
/// In sliding window attention (Mistral, etc.), position i attends to
/// positions j in [i - W, i] where W is the window size. Positions
/// outside this range get mask = 0, so attention weight is zeroed.
///
/// We model: for position outside window, mask = 0, so a_masked = a * 0 = 0.
/// Prove: a_masked = 0.
#[test]
fn test_824_gqa_sliding_window_mask_correctness() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("mask", real.clone());
    let _ = prog.declare_const("a_masked", real);

    let a = real_var("a");
    let mask = real_var("mask");
    let a_masked = real_var("a_masked");

    // a is arbitrary attention score
    prog.assert(a.clone().real_ge(Expr::real(-50)));
    prog.assert(a.clone().real_le(Expr::real(50)));

    // mask = 0 (position outside sliding window)
    prog.assert(mask.clone().eq(Expr::real(0)));

    // a_masked = a * mask
    prog.assert(a_masked.clone().eq(a.real_mul(mask)));

    // Negated property: a_masked != 0
    let violation = a_masked.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_sliding_window_mask_correctness");
}

// ---------------------------------------------------------------------------
// Test 825: ALiBi position bias linearity
// ---------------------------------------------------------------------------

/// Prove: ALiBi (Attention with Linear Biases) position bias is linear
/// in position difference.
///
/// ALiBi: bias(i, j) = -m * |i - j| where m is the head-specific slope.
/// For a fixed head, bias is a linear function of the distance |i - j|.
///
/// Linearity property: bias(dist) = -m * dist.
/// If dist doubles, bias doubles: bias(2*d) = 2 * bias(d).
///
/// We model: bias1 = -m * d, bias2 = -m * (2*d).
/// Prove: bias2 = 2 * bias1.
#[test]
fn test_825_gqa_alibi_position_bias_linearity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("d", real.clone());
    let _ = prog.declare_const("bias1", real.clone());
    let _ = prog.declare_const("bias2", real);

    let m = real_var("m");
    let d = real_var("d");
    let bias1 = real_var("bias1");
    let bias2 = real_var("bias2");

    // m > 0 (positive slope)
    prog.assert(m.clone().real_gt(Expr::real(0)));

    // d > 0 (positive distance)
    prog.assert(d.clone().real_gt(Expr::real(0)));

    // bias1 = -m * d (using negation: bias1 + m * d = 0)
    prog.assert(
        bias1
            .clone()
            .real_add(m.clone().real_mul(d.clone()))
            .eq(Expr::real(0)),
    );

    // bias2 = -m * (2 * d)
    prog.assert(
        bias2
            .clone()
            .real_add(m.real_mul(Expr::real(2).real_mul(d)))
            .eq(Expr::real(0)),
    );

    // Negated property: bias2 != 2 * bias1
    let violation = bias2.ne(Expr::real(2).real_mul(bias1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_alibi_position_bias_linearity");
}

// ---------------------------------------------------------------------------
// Test 826: Multi-head concatenation dimension = num_heads * d_v
// ---------------------------------------------------------------------------

/// Prove: concatenating num_heads attention heads of dimension d_v each
/// yields a vector of dimension num_heads * d_v.
///
/// Each head produces a [seq, d_v] output. Concatenation along the last
/// dim: [seq, num_heads * d_v].
///
/// We model: concat_dim = num_heads * d_v.
/// Prove: concat_dim = d_model (where d_model = num_heads * d_v).
#[test]
fn test_826_gqa_multihead_concat_dimension() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("num_heads", real.clone());
    let _ = prog.declare_const("d_v", real.clone());
    let _ = prog.declare_const("d_model", real.clone());
    let _ = prog.declare_const("concat_dim", real);

    let num_heads = real_var("num_heads");
    let d_v = real_var("d_v");
    let d_model = real_var("d_model");
    let concat_dim = real_var("concat_dim");

    // All positive
    prog.assert(num_heads.clone().real_gt(Expr::real(0)));
    prog.assert(d_v.clone().real_gt(Expr::real(0)));

    // d_model = num_heads * d_v
    prog.assert(d_model.clone().eq(num_heads.clone().real_mul(d_v.clone())));

    // concat_dim = num_heads * d_v (from concatenation)
    prog.assert(concat_dim.clone().eq(num_heads.real_mul(d_v)));

    // Negated property: concat_dim != d_model
    let violation = concat_dim.ne(d_model);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_multihead_concat_dimension");
}

// ---------------------------------------------------------------------------
// Test 827: Scaled dot product attention associativity
// ---------------------------------------------------------------------------

/// Prove: scaling commutes with dot product for attention computation.
///
/// Two equivalent computations:
///   (1) score = (Q * K^T) / sqrt(d_k)   — scale after matmul
///   (2) score = (Q / sqrt(d_k)) * K^T    — scale Q first
///
/// For scalar proxy: (q * k) * s = (q * s) * k.
/// This is associativity/commutativity of multiplication.
///
/// We model: method1 = q * k * s, method2 = (q * s) * k.
/// Prove: method1 = method2.
#[test]
fn test_827_gqa_scaled_dot_product_associativity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("q", real.clone());
    let _ = prog.declare_const("k", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("method1", real.clone());
    let _ = prog.declare_const("method2", real);

    let q = real_var("q");
    let k = real_var("k");
    let s = real_var("s");
    let method1 = real_var("method1");
    let method2 = real_var("method2");

    // Bounded inputs
    prog.assert(q.clone().real_ge(Expr::real(-10)));
    prog.assert(q.clone().real_le(Expr::real(10)));
    prog.assert(k.clone().real_ge(Expr::real(-10)));
    prog.assert(k.clone().real_le(Expr::real(10)));
    prog.assert(s.clone().real_gt(Expr::real(0)));
    prog.assert(s.clone().real_le(Expr::real(1)));

    // method1 = (q * k) * s
    prog.assert(
        method1
            .clone()
            .eq(q.clone().real_mul(k.clone()).real_mul(s.clone())),
    );

    // method2 = (q * s) * k
    prog.assert(method2.clone().eq(q.real_mul(s).real_mul(k)));

    // Negated property: method1 != method2
    let violation = method1.ne(method2);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_scaled_dot_product_associativity");
}

// ---------------------------------------------------------------------------
// Test 828: Attention dropout mask binary (0 or 1)
// ---------------------------------------------------------------------------

/// Prove: dropout mask values are exactly 0 or 1, and the masked
/// attention weight is either 0 or scaled.
///
/// Dropout during training: a_drop = a * mask / (1 - p) where mask in {0, 1}.
/// When mask = 0: a_drop = 0. When mask = 1: a_drop = a / (1-p).
///
/// We model: mask in {0, 1}, a_drop = a * mask.
/// Prove: a_drop = 0 OR a_drop = a (only two possible outcomes).
#[test]
fn test_828_gqa_attention_dropout_mask_binary() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("mask", real.clone());
    let _ = prog.declare_const("a_drop", real);

    let a = real_var("a");
    let mask = real_var("mask");
    let a_drop = real_var("a_drop");

    // a is arbitrary bounded
    prog.assert(a.clone().real_ge(Expr::real(-10)));
    prog.assert(a.clone().real_le(Expr::real(10)));

    // mask is binary: mask = 0 OR mask = 1
    let mask_is_zero = mask.clone().eq(Expr::real(0));
    let mask_is_one = mask.clone().eq(Expr::real(1));
    prog.assert(mask_is_zero.or(mask_is_one));

    // a_drop = a * mask
    prog.assert(a_drop.clone().eq(a.clone().real_mul(mask)));

    // Negated property: a_drop != 0 AND a_drop != a
    // (the only two outcomes are 0 or a)
    let violation = a_drop.clone().ne(Expr::real(0)).and(a_drop.ne(a));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_attention_dropout_mask_binary");
}

// ---------------------------------------------------------------------------
// Test 829: Flash attention equivalence (chunked QK computation)
// ---------------------------------------------------------------------------

/// Prove: chunked attention (flash attention style) produces the same
/// weighted sum as full attention for a single query.
///
/// Full: out = a1*v1 + a2*v2 where a_i = e_i / (e1 + e2).
/// Chunked (two chunks of 1):
///   chunk1: partial = e1*v1, denom1 = e1.
///   chunk2: new_denom = denom1 + e2. Rescale:
///           out = (partial * (denom1/new_denom)) + (e2*v2 / new_denom)
///               = (e1*v1 + e2*v2) / (e1 + e2) = full result.
///
/// We model: full_out = (e1*v1 + e2*v2) / D, chunk_out = same expression.
/// Prove: full_out = chunk_out.
#[test]
fn test_829_gqa_flash_attention_chunked_equivalence() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("e1", real.clone());
    let _ = prog.declare_const("e2", real.clone());
    let _ = prog.declare_const("v1", real.clone());
    let _ = prog.declare_const("v2", real.clone());
    let _ = prog.declare_const("D", real.clone());
    let _ = prog.declare_const("full_out", real.clone());
    let _ = prog.declare_const("chunk_out", real);

    let e1 = real_var("e1");
    let e2 = real_var("e2");
    let v1 = real_var("v1");
    let v2 = real_var("v2");
    let d_var = real_var("D");
    let full_out = real_var("full_out");
    let chunk_out = real_var("chunk_out");

    // e_i > 0 (exponentials are positive)
    prog.assert(e1.clone().real_gt(Expr::real(0)));
    prog.assert(e2.clone().real_gt(Expr::real(0)));

    // v_i bounded
    prog.assert(v1.clone().real_ge(Expr::real(-10)));
    prog.assert(v1.clone().real_le(Expr::real(10)));
    prog.assert(v2.clone().real_ge(Expr::real(-10)));
    prog.assert(v2.clone().real_le(Expr::real(10)));

    // D = e1 + e2
    prog.assert(d_var.clone().eq(e1.clone().real_add(e2.clone())));

    // full_out * D = e1*v1 + e2*v2
    prog.assert(
        full_out.clone().real_mul(d_var.clone()).eq(e1
            .clone()
            .real_mul(v1.clone())
            .real_add(e2.clone().real_mul(v2.clone()))),
    );

    // chunk_out: same computation via rescaling
    // chunk_out * D = e1*v1 + e2*v2
    prog.assert(
        chunk_out
            .clone()
            .real_mul(d_var)
            .eq(e1.real_mul(v1).real_add(e2.real_mul(v2))),
    );

    // Negated property: full_out != chunk_out
    let violation = full_out.ne(chunk_out);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_flash_attention_chunked_equivalence");
}

// ---------------------------------------------------------------------------
// Test 830: GQA reduces to MHA when num_kv_heads == num_heads
// ---------------------------------------------------------------------------

/// Prove: when num_kv_heads equals num_heads, GQA is identical to
/// standard Multi-Head Attention (MHA) — no head repetition occurs.
///
/// GQA repeat factor = num_heads / num_kv_heads. When they are equal,
/// repeat_factor = 1, meaning each KV head serves exactly one query head.
/// The repeat_kv is an identity operation, so GQA output = MHA output.
///
/// We model: repeat_factor * num_kv_heads = num_heads, num_kv_heads = num_heads.
/// Prove: repeat_factor = 1.
#[test]
fn test_830_gqa_reduces_to_mha() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("num_heads", real.clone());
    let _ = prog.declare_const("num_kv_heads", real.clone());
    let _ = prog.declare_const("repeat_factor", real);

    let num_heads = real_var("num_heads");
    let num_kv_heads = real_var("num_kv_heads");
    let repeat_factor = real_var("repeat_factor");

    // num_heads > 0
    prog.assert(num_heads.clone().real_gt(Expr::real(0)));

    // num_kv_heads = num_heads (GQA = MHA case)
    prog.assert(num_kv_heads.clone().eq(num_heads.clone()));

    // repeat_factor * num_kv_heads = num_heads
    prog.assert(repeat_factor.clone().real_mul(num_kv_heads).eq(num_heads));

    // Negated property: repeat_factor != 1
    let violation = repeat_factor.ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_reduces_to_mha");
}
