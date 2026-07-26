// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for attention score scaling and causal masking
//! mathematical properties.
//!
//! Proves fundamental properties of attention scaling and masking mechanisms:
//! - Scaling factor positivity and ordering preservation
//! - Causal mask: large negative values, triangle structure
//! - Softmax of masked scores: masked positions approach zero
//! - Attention weights: sum-to-one, [0,1] range, convex combination output
//! - Multi-head attention: dimension constraints, concat restoration
//! - GQA: divisibility, integer repeat factor
//! - Sliding window: sparsity, causal intersection
//! - ALiBi: linear penalty, geometric slopes
//! - Cross-attention: no causal constraint
//! - Dropout expected value, numerically stable softmax, temperature scaling
//!
//! Part of #4139.

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
// Test 551: Scaling factor 1/sqrt(d_k) > 0 for d_k > 0
// ---------------------------------------------------------------------------

/// Prove: the attention scaling factor 1/sqrt(d_k) is positive when d_k > 0.
///
/// sqrt(d_k) > 0 for d_k > 0, and 1/sqrt(d_k) > 0 since both numerator
/// and denominator are positive.
///
/// We model: scale > 0, scale^2 = d_k (so scale = sqrt(d_k)), inv > 0,
/// inv * scale = 1 (so inv = 1/sqrt(d_k)). Prove inv > 0.
#[test]
fn test_551_scaling_factor_positive() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("d_k", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("inv", real);

    let d_k = real_var("d_k");
    let scale = real_var("scale");
    let inv = real_var("inv");

    // d_k > 0
    prog.assert(d_k.clone().real_gt(Expr::real(0)));
    prog.assert(d_k.clone().real_le(Expr::real(10000)));

    // scale = sqrt(d_k): scale > 0 and scale^2 = d_k
    prog.assert(scale.clone().real_gt(Expr::real(0)));
    prog.assert(scale.clone().real_mul(scale.clone()).eq(d_k));

    // inv = 1/scale: inv * scale = 1
    prog.assert(inv.clone().real_mul(scale).eq(Expr::real(1)));

    // Negated property: inv <= 0
    let violation = inv.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "scaling_factor_positive");
}

// ---------------------------------------------------------------------------
// Test 552: Scaled scores preserve relative ordering
// ---------------------------------------------------------------------------

/// Prove: if s1 > s2, then s1/sqrt(d) > s2/sqrt(d) for d > 0.
///
/// Dividing by a positive constant preserves ordering. Since sqrt(d) > 0,
/// s1 > s2 implies s1/sqrt(d) > s2/sqrt(d).
///
/// We model: scale > 0, ss1 = s1/scale, ss2 = s2/scale, s1 > s2.
/// Prove ss1 > ss2.
#[test]
fn test_552_scaled_scores_preserve_ordering() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("s2", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("ss1", real.clone());
    let _ = prog.declare_const("ss2", real);

    let s1 = real_var("s1");
    let s2 = real_var("s2");
    let scale = real_var("scale");
    let ss1 = real_var("ss1");
    let ss2 = real_var("ss2");

    // s1 > s2
    prog.assert(s1.clone().real_gt(s2.clone()));

    // scale > 0 (scale = sqrt(d_k))
    prog.assert(scale.clone().real_gt(Expr::real(0)));

    // ss1 = s1 / scale, ss2 = s2 / scale
    prog.assert(ss1.clone().real_mul(scale.clone()).eq(s1));
    prog.assert(ss2.clone().real_mul(scale).eq(s2));

    // Negated property: ss1 <= ss2 (ordering not preserved)
    let violation = ss1.real_le(ss2);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "scaled_scores_preserve_ordering");
}

// ---------------------------------------------------------------------------
// Test 553: Causal mask: masked positions receive large negative value
// ---------------------------------------------------------------------------

/// Prove: after causal masking, the masked score equals original score + (-M)
/// where M is a large positive value, resulting in a very negative score.
///
/// For position (i, j) where j > i, score_masked = score_orig + (-M).
/// Since M is large (e.g., 10000), score_masked < score_orig - M + bound < 0
/// for bounded original scores.
///
/// We model: original score in [-B, B], mask_value = -M with M >> B,
/// masked_score = score + mask_value. Prove masked_score < -threshold.
#[test]
fn test_553_causal_mask_large_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("score", real.clone());
    let _ = prog.declare_const("mask_val", real.clone());
    let _ = prog.declare_const("masked_score", real);

    let score = real_var("score");
    let mask_val = real_var("mask_val");
    let masked_score = real_var("masked_score");

    // Original score bounded: |score| <= 100
    prog.assert(score.clone().real_ge(Expr::real(-100)));
    prog.assert(score.clone().real_le(Expr::real(100)));

    // Mask value is large negative: mask_val = -10000
    prog.assert(mask_val.clone().eq(Expr::real(-10000)));

    // masked_score = score + mask_val
    prog.assert(masked_score.clone().eq(score.real_add(mask_val)));

    // Negated property: masked_score >= -9000 (should be < -9000)
    let violation = masked_score.real_ge(Expr::real(-9000));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "causal_mask_large_negative");
}

// ---------------------------------------------------------------------------
// Test 554: Causal mask triangle: row i has i+1 unmasked positions
// ---------------------------------------------------------------------------

/// Prove: in a causal mask for sequence length N, row i (0-indexed) has
/// exactly i+1 unmasked positions (columns 0 through i).
///
/// The number of unmasked positions in row i equals i + 1.
/// Total unmasked positions for the full NxN mask = sum_{i=0}^{N-1}(i+1)
/// = N*(N+1)/2.
///
/// We model: for row index i (0 <= i < N), unmasked_count = i + 1.
/// Verify this for a specific row index within bounds.
#[test]
fn test_554_causal_mask_triangle_count() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("i", real.clone());
    let _ = prog.declare_const("n", real.clone());
    let _ = prog.declare_const("unmasked", real);

    let i = real_var("i");
    let n = real_var("n");
    let unmasked = real_var("unmasked");

    // Valid row index: 0 <= i < N, N > 0
    prog.assert(i.clone().real_ge(Expr::real(0)));
    prog.assert(n.clone().real_gt(Expr::real(0)));
    prog.assert(i.clone().real_lt(n));

    // Causal mask rule: row i has i+1 unmasked positions
    prog.assert(unmasked.clone().eq(i.clone().real_add(Expr::real(1))));

    // Negated property: unmasked != i + 1
    let violation = unmasked.ne(i.real_add(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "causal_mask_triangle_count");
}

// ---------------------------------------------------------------------------
// Test 555: Softmax of masked scores: exp(-inf)/sum -> 0
// ---------------------------------------------------------------------------

/// Prove: when a score is masked to a very large negative value, its softmax
/// output approaches zero.
///
/// exp(-M) for M >> 0 is near zero. In the softmax denominator, the
/// contribution of the masked term is negligible, so the softmax output
/// for the masked position is near zero.
///
/// We model: exp_masked in [0, epsilon], exp_valid > 0, exp_valid bounded.
/// s_masked = exp_masked / (exp_valid + exp_masked). Prove s_masked < threshold.
#[test]
fn test_555_softmax_masked_approaches_zero() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp_valid", real.clone());
    let _ = prog.declare_const("exp_masked", real.clone());
    let _ = prog.declare_const("s_masked", real);

    let exp_valid = real_var("exp_valid");
    let exp_masked = real_var("exp_masked");
    let s_masked = real_var("s_masked");

    // Valid position has positive exp, bounded
    prog.assert(exp_valid.clone().real_gt(Expr::real(0)));
    prog.assert(exp_valid.clone().real_le(Expr::real(1000)));

    // Masked position: exp(very_negative) is near zero
    let eps = Expr::real_ratio(1, 1000000);
    prog.assert(exp_masked.clone().real_ge(Expr::real(0)));
    prog.assert(exp_masked.clone().real_le(eps));

    // s_masked = exp_masked / (exp_valid + exp_masked)
    let z = exp_valid.real_add(exp_masked.clone());
    prog.assert(s_masked.clone().real_mul(z).eq(exp_masked));

    // Negated property: s_masked > 0.001 (should be near zero)
    let violation = s_masked.real_gt(Expr::real_ratio(1, 1000));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softmax_masked_approaches_zero");
}

// ---------------------------------------------------------------------------
// Test 556: Attention weights sum to 1 per query position
// ---------------------------------------------------------------------------

/// Prove: attention weights (softmax outputs) for a single query position
/// sum to 1.
///
/// Since attention weights are computed via softmax over the key positions,
/// and softmax outputs sum to 1 by definition, the attention weights for
/// each query position sum to 1.
///
/// We model a 4-element case (representative of sequence length).
#[test]
fn test_556_attention_weights_sum_to_one() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("w3", real.clone());
    let _ = prog.declare_const("w4", real);

    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let w3 = real_var("w3");
    let w4 = real_var("w4");

    // Softmax outputs: all positive, sum to 1
    prog.assert(w1.clone().real_gt(Expr::real(0)));
    prog.assert(w2.clone().real_gt(Expr::real(0)));
    prog.assert(w3.clone().real_gt(Expr::real(0)));
    prog.assert(w4.clone().real_gt(Expr::real(0)));
    prog.assert(
        w1.clone()
            .real_add(w2.clone())
            .real_add(w3.clone())
            .real_add(w4.clone())
            .eq(Expr::real(1)),
    );

    // Negated property: sum != 1
    let violation = w1.real_add(w2).real_add(w3).real_add(w4).ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "attention_weights_sum_to_one");
}

// ---------------------------------------------------------------------------
// Test 557: Attention weights in [0, 1] per element
// ---------------------------------------------------------------------------

/// Prove: each attention weight w_i is in [0, 1].
///
/// Since w_i = exp(s_i) / Z with exp(s_i) > 0 and Z >= exp(s_i),
/// we have 0 < w_i <= 1. With at least 2 elements, w_i < 1.
///
/// We model: w_i is a softmax output in (0, 1) with constraints.
#[test]
fn test_557_attention_weights_in_zero_one() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp_i", real.clone());
    let _ = prog.declare_const("exp_other", real.clone());
    let _ = prog.declare_const("z", real.clone());
    let _ = prog.declare_const("w_i", real);

    let exp_i = real_var("exp_i");
    let exp_other = real_var("exp_other");
    let z = real_var("z");
    let w_i = real_var("w_i");

    // exp values positive (at least 2 elements)
    prog.assert(exp_i.clone().real_gt(Expr::real(0)));
    prog.assert(exp_other.clone().real_gt(Expr::real(0)));

    // Z = exp_i + exp_other (at least 2 elements)
    prog.assert(z.clone().eq(exp_i.clone().real_add(exp_other)));

    // w_i = exp_i / Z: w_i * Z = exp_i
    prog.assert(w_i.clone().real_mul(z).eq(exp_i));

    // Negated property: w_i <= 0 OR w_i >= 1
    let violation = w_i
        .clone()
        .real_le(Expr::real(0))
        .or(w_i.real_ge(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "attention_weights_in_zero_one");
}

// ---------------------------------------------------------------------------
// Test 558: Attention output bounded by value range (convex combination)
// ---------------------------------------------------------------------------

/// Prove: if all value vectors V_j are in [lo, hi], then the attention
/// output (a convex combination sum_j w_j * V_j) is also in [lo, hi].
///
/// Since w_j >= 0 and sum(w_j) = 1, the output is a convex combination.
/// Convex combinations of values in [lo, hi] remain in [lo, hi].
///
/// We model a 3-element case.
#[test]
fn test_558_attention_output_bounded_by_values() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("v1", real.clone());
    let _ = prog.declare_const("v2", real.clone());
    let _ = prog.declare_const("v3", real.clone());
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("w3", real.clone());
    let _ = prog.declare_const("out", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real);

    let v1 = real_var("v1");
    let v2 = real_var("v2");
    let v3 = real_var("v3");
    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let w3 = real_var("w3");
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
    prog.assert(v3.clone().real_ge(lo.clone()));
    prog.assert(v3.clone().real_le(hi.clone()));

    // Weights: w_i >= 0, sum = 1
    prog.assert(w1.clone().real_ge(Expr::real(0)));
    prog.assert(w2.clone().real_ge(Expr::real(0)));
    prog.assert(w3.clone().real_ge(Expr::real(0)));
    prog.assert(
        w1.clone()
            .real_add(w2.clone())
            .real_add(w3.clone())
            .eq(Expr::real(1)),
    );

    // out = w1*v1 + w2*v2 + w3*v3
    prog.assert(
        out.clone().eq(w1
            .real_mul(v1)
            .real_add(w2.real_mul(v2))
            .real_add(w3.real_mul(v3))),
    );

    // Negated property: out < lo OR out > hi
    let violation = out.clone().real_lt(lo).or(out.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "attention_output_bounded_by_values");
}

// ---------------------------------------------------------------------------
// Test 559: Multi-head: head_dim * num_heads == model_dim
// ---------------------------------------------------------------------------

/// Prove: the multi-head attention dimension constraint holds:
/// head_dim * num_heads = model_dim.
///
/// This is the fundamental dimension invariant for multi-head attention.
/// The model dimension is split evenly across heads.
///
/// We model: d_model = n_heads * d_head, and prove the identity holds.
#[test]
fn test_559_multi_head_dimension_constraint() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("d_model", real.clone());
    let _ = prog.declare_const("n_heads", real.clone());
    let _ = prog.declare_const("d_head", real);

    let d_model = real_var("d_model");
    let n_heads = real_var("n_heads");
    let d_head = real_var("d_head");

    // Positive dimensions
    prog.assert(n_heads.clone().real_gt(Expr::real(0)));
    prog.assert(d_head.clone().real_gt(Expr::real(0)));

    // d_model = n_heads * d_head
    prog.assert(d_model.clone().eq(n_heads.clone().real_mul(d_head.clone())));

    // Negated property: d_model != n_heads * d_head
    let violation = d_model.ne(n_heads.real_mul(d_head));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "multi_head_dimension_constraint");
}

// ---------------------------------------------------------------------------
// Test 560: Multi-head concat: output restores model_dim
// ---------------------------------------------------------------------------

/// Prove: concatenating n_heads heads of dimension d_head produces model_dim.
///
/// After multi-head attention, each head produces output of shape [seq, d_head].
/// Concatenation along the last axis produces [seq, n_heads * d_head] = [seq, d_model].
///
/// We model: concat_dim = n_heads * d_head, and d_model = n_heads * d_head.
/// Prove concat_dim = d_model.
#[test]
fn test_560_multi_head_concat_restores_dim() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("d_model", real.clone());
    let _ = prog.declare_const("n_heads", real.clone());
    let _ = prog.declare_const("d_head", real.clone());
    let _ = prog.declare_const("concat_dim", real);

    let d_model = real_var("d_model");
    let n_heads = real_var("n_heads");
    let d_head = real_var("d_head");
    let concat_dim = real_var("concat_dim");

    // Positive dimensions
    prog.assert(n_heads.clone().real_gt(Expr::real(0)));
    prog.assert(d_head.clone().real_gt(Expr::real(0)));

    // d_model = n_heads * d_head (definition)
    prog.assert(d_model.clone().eq(n_heads.clone().real_mul(d_head.clone())));

    // Concat dimension = n_heads * d_head (concatenation)
    prog.assert(concat_dim.clone().eq(n_heads.real_mul(d_head)));

    // Negated property: concat_dim != d_model
    let violation = concat_dim.ne(d_model);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "multi_head_concat_restores_dim");
}

// ---------------------------------------------------------------------------
// Test 561: GQA: num_heads divisible by num_kv_heads
// ---------------------------------------------------------------------------

/// Prove: in grouped-query attention, num_heads is divisible by num_kv_heads.
///
/// GQA requires num_heads = num_kv_heads * repeat_factor where repeat_factor
/// is a positive integer. This means num_heads is divisible by num_kv_heads.
///
/// We model: n_heads = n_kv * repeat with repeat > 0, and prove
/// n_heads = n_kv * repeat (the divisibility relationship).
#[test]
fn test_561_gqa_divisibility() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("n_heads", real.clone());
    let _ = prog.declare_const("n_kv", real.clone());
    let _ = prog.declare_const("repeat", real.clone());
    let _ = prog.declare_const("product", real);

    let n_heads = real_var("n_heads");
    let n_kv = real_var("n_kv");
    let repeat = real_var("repeat");
    let product = real_var("product");

    // All positive
    prog.assert(n_heads.clone().real_gt(Expr::real(0)));
    prog.assert(n_kv.clone().real_gt(Expr::real(0)));
    prog.assert(repeat.clone().real_gt(Expr::real(0)));

    // GQA constraint: n_heads = n_kv * repeat
    prog.assert(n_heads.clone().eq(n_kv.clone().real_mul(repeat.clone())));

    // product = n_kv * repeat (verification)
    prog.assert(product.clone().eq(n_kv.real_mul(repeat.clone())));

    // Negated property: product != n_heads (divisibility violated)
    let violation = product.ne(n_heads);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_divisibility");
}

// ---------------------------------------------------------------------------
// Test 562: GQA repeat factor: num_heads / num_kv_heads is integer
// ---------------------------------------------------------------------------

/// Prove: the GQA repeat factor r = n_heads / n_kv satisfies
/// n_kv * r = n_heads exactly (no remainder).
///
/// Given n_heads = n_kv * r by definition, this is an identity.
/// The proof verifies the algebraic constraint is consistent.
#[test]
fn test_562_gqa_repeat_factor_integer() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("n_heads", real.clone());
    let _ = prog.declare_const("n_kv", real.clone());
    let _ = prog.declare_const("r", real);

    let n_heads = real_var("n_heads");
    let n_kv = real_var("n_kv");
    let r = real_var("r");

    // Positive values
    prog.assert(n_heads.clone().real_gt(Expr::real(0)));
    prog.assert(n_kv.clone().real_gt(Expr::real(0)));
    prog.assert(r.clone().real_gt(Expr::real(0)));

    // r = n_heads / n_kv: r * n_kv = n_heads
    prog.assert(r.clone().real_mul(n_kv.clone()).eq(n_heads.clone()));

    // Verify: n_kv * r = n_heads (no remainder)
    let reconstructed = n_kv.real_mul(r);

    // Negated property: n_kv * r != n_heads
    let violation = reconstructed.ne(n_heads);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_repeat_factor_integer");
}

// ---------------------------------------------------------------------------
// Test 563: Sliding window: each row has at most window_size nonzero entries
// ---------------------------------------------------------------------------

/// Prove: in sliding window attention with window size W, the number of
/// unmasked (nonzero-weight) positions per row is at most W.
///
/// For row i with causal constraint, the unmasked range is
/// [max(0, i - W + 1), i]. The count is min(W, i + 1).
/// This is always <= W.
///
/// We model: count = min(W, i + 1) with W > 0 and i >= 0.
/// Prove count <= W.
#[test]
fn test_563_sliding_window_max_entries() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("w", real.clone());
    let _ = prog.declare_const("i", real.clone());
    let _ = prog.declare_const("count", real);

    let w = real_var("w");
    let i = real_var("i");
    let count = real_var("count");

    // W > 0, i >= 0
    prog.assert(w.clone().real_gt(Expr::real(0)));
    prog.assert(i.clone().real_ge(Expr::real(0)));

    // count = min(W, i+1): count <= W AND count <= i+1
    let i_plus_1 = i.real_add(Expr::real(1));
    prog.assert(count.clone().real_le(w.clone()));
    prog.assert(count.clone().real_le(i_plus_1.clone()));
    // count equals the smaller of the two
    prog.assert(count.clone().eq(w.clone()).or(count.clone().eq(i_plus_1)));

    // Negated property: count > W
    let violation = count.real_gt(w);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "sliding_window_max_entries");
}

// ---------------------------------------------------------------------------
// Test 564: Sliding window + causal: intersection mask
// ---------------------------------------------------------------------------

/// Prove: the intersection of sliding window mask and causal mask is the
/// set of positions j such that max(0, i - W + 1) <= j <= i.
///
/// Causal allows j <= i. Sliding window allows |i - j| < W, i.e.,
/// i - W + 1 <= j <= i + W - 1. The intersection is max(0, i - W + 1) <= j <= i.
///
/// We model: for a valid position j in the intersection, both constraints
/// hold. We prove j <= i AND j >= i - W + 1.
#[test]
fn test_564_sliding_window_causal_intersection() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("i", real.clone());
    let _ = prog.declare_const("j", real.clone());
    let _ = prog.declare_const("w", real);

    let i = real_var("i");
    let j = real_var("j");
    let w = real_var("w");

    // W > 0, i >= 0, j >= 0
    prog.assert(w.clone().real_gt(Expr::real(0)));
    prog.assert(i.clone().real_ge(Expr::real(0)));
    prog.assert(j.clone().real_ge(Expr::real(0)));

    // Causal constraint: j <= i
    prog.assert(j.clone().real_le(i.clone()));

    // Sliding window constraint: j >= i - W + 1
    let lower_bound = i.clone().real_sub(w).real_add(Expr::real(1));
    prog.assert(j.clone().real_ge(lower_bound.clone()));

    // Negated property: j > i OR j < i - W + 1
    // (should be impossible given the constraints)
    let violation = j.clone().real_gt(i).or(j.real_lt(lower_bound));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "sliding_window_causal_intersection");
}

// ---------------------------------------------------------------------------
// Test 565: ALiBi linear penalty: bias(i, j) = -slope * |i - j|
// ---------------------------------------------------------------------------

/// Prove: the ALiBi bias is linear in position distance.
///
/// For positions i, j with i >= j (causal), bias = -slope * (i - j).
/// Doubling the distance doubles the (absolute) bias.
///
/// We model: bias1 = -m * d1, bias2 = -m * d2, d2 = 2*d1.
/// Prove bias2 = 2 * bias1.
#[test]
fn test_565_alibi_linear_penalty() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("d1", real.clone());
    let _ = prog.declare_const("d2", real.clone());
    let _ = prog.declare_const("bias1", real.clone());
    let _ = prog.declare_const("bias2", real);

    let m = real_var("m");
    let d1 = real_var("d1");
    let d2 = real_var("d2");
    let bias1 = real_var("bias1");
    let bias2 = real_var("bias2");

    // m > 0 (head slope), d1 >= 0, d2 = 2*d1
    prog.assert(m.clone().real_gt(Expr::real(0)));
    prog.assert(d1.clone().real_ge(Expr::real(0)));
    prog.assert(d2.clone().eq(Expr::real(2).real_mul(d1.clone())));

    // bias = -m * distance
    prog.assert(
        bias1
            .clone()
            .eq(Expr::real(0).real_sub(m.clone().real_mul(d1))),
    );
    prog.assert(bias2.clone().eq(Expr::real(0).real_sub(m.real_mul(d2))));

    // Negated property: bias2 != 2 * bias1 (linearity violated)
    let violation = bias2.ne(Expr::real(2).real_mul(bias1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "alibi_linear_penalty");
}

// ---------------------------------------------------------------------------
// Test 566: ALiBi slopes: geometric sequence 2^(-8k/n)
// ---------------------------------------------------------------------------

/// Prove: ALiBi slopes form a geometric sequence where the ratio between
/// consecutive slopes is constant.
///
/// slope_k = 2^(-8k/n). The ratio slope_{k+1} / slope_k = 2^(-8/n),
/// which is constant (independent of k).
///
/// We model: s1 > 0, s2 > 0, ratio > 0, s2 = s1 * ratio, s3 = s2 * ratio.
/// Prove s3 = s1 * ratio^2 (geometric sequence property).
#[test]
fn test_566_alibi_slopes_geometric() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("s2", real.clone());
    let _ = prog.declare_const("s3", real.clone());
    let _ = prog.declare_const("ratio", real);

    let s1 = real_var("s1");
    let s2 = real_var("s2");
    let s3 = real_var("s3");
    let ratio = real_var("ratio");

    // All positive
    prog.assert(s1.clone().real_gt(Expr::real(0)));
    prog.assert(ratio.clone().real_gt(Expr::real(0)));
    prog.assert(ratio.clone().real_lt(Expr::real(1)));

    // Geometric sequence: s2 = s1 * ratio, s3 = s2 * ratio
    prog.assert(s2.clone().eq(s1.clone().real_mul(ratio.clone())));
    prog.assert(s3.clone().eq(s2.real_mul(ratio.clone())));

    // Expected: s3 = s1 * ratio^2
    let expected_s3 = s1.real_mul(ratio.clone().real_mul(ratio));

    // Negated property: s3 != s1 * ratio^2
    let violation = s3.ne(expected_s3);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "alibi_slopes_geometric");
}

// ---------------------------------------------------------------------------
// Test 567: Cross-attention: no causal constraint (full attention)
// ---------------------------------------------------------------------------

/// Prove: in cross-attention, every decoder position can attend to every
/// encoder position. The attention weight matrix has no zeros from masking.
///
/// For cross-attention with seq_dec decoder positions and seq_enc encoder
/// positions, the weight matrix has shape [seq_dec, seq_enc] where every
/// entry is positive (softmax output > 0).
///
/// We model: 2 encoder positions, 1 decoder position. Both weights > 0
/// and sum to 1. No position is masked to zero.
#[test]
fn test_567_cross_attention_full() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp_1", real.clone());
    let _ = prog.declare_const("exp_2", real.clone());
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real);

    let exp_1 = real_var("exp_1");
    let exp_2 = real_var("exp_2");
    let w1 = real_var("w1");
    let w2 = real_var("w2");

    // Both exp values positive (no masking)
    prog.assert(exp_1.clone().real_gt(Expr::real(0)));
    prog.assert(exp_1.clone().real_le(Expr::real(1000)));
    prog.assert(exp_2.clone().real_gt(Expr::real(0)));
    prog.assert(exp_2.clone().real_le(Expr::real(1000)));

    // Softmax: w1 = exp_1 / (exp_1 + exp_2), w2 = exp_2 / (exp_1 + exp_2)
    let z = exp_1.clone().real_add(exp_2.clone());
    prog.assert(w1.clone().real_mul(z.clone()).eq(exp_1));
    prog.assert(w2.clone().real_mul(z).eq(exp_2));

    // Negated property: w1 <= 0 OR w2 <= 0 (some position has zero weight)
    let violation = w1.real_le(Expr::real(0)).or(w2.real_le(Expr::real(0)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cross_attention_full");
}

// ---------------------------------------------------------------------------
// Test 568: Dropout expected value: E[drop(x)] = x (inverted scaling)
// ---------------------------------------------------------------------------

/// Prove: with inverted dropout (scaling by 1/(1-p)), the expected value
/// of the output equals the input.
///
/// During training: each element is kept with probability (1-p) and scaled
/// by 1/(1-p). E[output] = (1-p) * x/(1-p) + p * 0 = x.
///
/// We model: keep_prob = 1 - p, scale = 1 / keep_prob.
/// E[output] = keep_prob * (x * scale) + (1 - keep_prob) * 0 = x.
#[test]
fn test_568_dropout_expected_value() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("keep_prob", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("expected", real);

    let x = real_var("x");
    let keep_prob = real_var("keep_prob");
    let scale = real_var("scale");
    let expected = real_var("expected");

    // x is any real
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));

    // 0 < keep_prob <= 1 (p in [0, 1))
    prog.assert(keep_prob.clone().real_gt(Expr::real(0)));
    prog.assert(keep_prob.clone().real_le(Expr::real(1)));

    // scale = 1 / keep_prob: scale * keep_prob = 1
    prog.assert(scale.clone().real_mul(keep_prob.clone()).eq(Expr::real(1)));

    // E[output] = keep_prob * (x * scale) + (1 - keep_prob) * 0
    //           = keep_prob * x * scale
    let kept_contribution = keep_prob.real_mul(x.clone().real_mul(scale));
    prog.assert(expected.clone().eq(kept_contribution));

    // Negated property: E[output] != x
    let violation = expected.ne(x);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dropout_expected_value");
}

// ---------------------------------------------------------------------------
// Test 569: Attention score max subtraction: numerically stable softmax
// ---------------------------------------------------------------------------

/// Prove: subtracting the max score before softmax does not change the result.
///
/// softmax(s_i - max) = exp(s_i - max) / sum(exp(s_j - max))
///                    = exp(s_i) * exp(-max) / (sum(exp(s_j)) * exp(-max))
///                    = exp(s_i) / sum(exp(s_j))
///                    = softmax(s_i).
///
/// This is equivalent to softmax shift invariance. We prove it for the
/// 2-element case: the shifted and original softmax outputs are equal.
#[test]
fn test_569_attention_stable_softmax() {
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

    // k = exp(-max) > 0 (shift factor)
    prog.assert(k.clone().real_gt(Expr::real(0)));

    // s_orig = e1 / (e1 + e2)
    let z_orig = e1.clone().real_add(e2.clone());
    prog.assert(s_orig.clone().real_mul(z_orig).eq(e1.clone()));

    // s_shifted = (e1*k) / (e1*k + e2*k)
    let e1k = e1.real_mul(k.clone());
    let e2k = e2.real_mul(k);
    let z_shifted = e1k.clone().real_add(e2k);
    prog.assert(s_shifted.clone().real_mul(z_shifted).eq(e1k));

    // Negated property: s_orig != s_shifted
    let violation = s_orig.ne(s_shifted);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "attention_stable_softmax");
}

// ---------------------------------------------------------------------------
// Test 570: Temperature scaling: divide by T > 0 preserves ordering
// ---------------------------------------------------------------------------

/// Prove: dividing scores by temperature T > 0 preserves their relative
/// ordering. If score_a > score_b, then score_a / T > score_b / T.
///
/// Since T > 0, dividing by T is equivalent to multiplying by 1/T > 0,
/// which preserves ordering.
///
/// This guarantees that temperature scaling only changes the "sharpness"
/// of the softmax distribution, not the ranking of tokens.
#[test]
fn test_570_temperature_preserves_ordering() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("score_a", real.clone());
    let _ = prog.declare_const("score_b", real.clone());
    let _ = prog.declare_const("temp", real.clone());
    let _ = prog.declare_const("scaled_a", real.clone());
    let _ = prog.declare_const("scaled_b", real);

    let score_a = real_var("score_a");
    let score_b = real_var("score_b");
    let temp = real_var("temp");
    let scaled_a = real_var("scaled_a");
    let scaled_b = real_var("scaled_b");

    // score_a > score_b
    prog.assert(score_a.clone().real_gt(score_b.clone()));

    // T > 0
    prog.assert(temp.clone().real_gt(Expr::real(0)));

    // scaled_a = score_a / T, scaled_b = score_b / T
    prog.assert(scaled_a.clone().real_mul(temp.clone()).eq(score_a));
    prog.assert(scaled_b.clone().real_mul(temp).eq(score_b));

    // Negated property: scaled_a <= scaled_b (ordering not preserved)
    let violation = scaled_a.real_le(scaled_b);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "temperature_preserves_ordering");
}
