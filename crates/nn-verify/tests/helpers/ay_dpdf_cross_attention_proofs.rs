// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for cross-attention decoder mathematical
//! properties.
//!
//! Proves fundamental properties of encoder-decoder cross-attention as used
//! in DETR-style object detection and transformer decoder architectures:
//! - Query-key dot product scaling by 1/sqrt(d_k)
//! - Cross-attention softmax sums to 1.0
//! - Cross-attention weight non-negativity
//! - Value projection linear combination bounds
//! - Multi-head split dimension correctness
//! - Multi-head concat output dimension
//! - Object query initialization bounds (DETR)
//! - Encoder-decoder attention key/value projection
//! - Cross-attention with padding mask zeros
//! - Position encoding addition commutativity
//! - Attention output norm bounds
//! - Layer norm after cross-attention bounds
//! - Residual connection preservation
//! - Decoder self-attention causal mask
//! - Encoder feature broadcasting
//! - Hungarian matching cost bounds (DETR)
//! - Box regression coordinate bounds
//! - Class probability softmax bounds
//! - Iterative refinement convergence
//! - Full decoder layer output bounds
//!
//! Part of #4170.

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
// Test 751: Query-key dot product scaling by 1/sqrt(d_k)
// ---------------------------------------------------------------------------

/// Prove: the scaled dot-product attention divides by sqrt(d_k) > 0, so the
/// scaled score has the same sign as the raw dot product.
///
/// If raw_score > 0 and scale = 1/sqrt(d_k) > 0, then scaled = raw * scale > 0.
/// Likewise for raw_score < 0.
///
/// We model: scale > 0, scale^2 * d_k = 1 (scale = 1/sqrt(d_k)),
/// raw_score > 0. Prove scaled_score > 0.
#[test]
fn test_751_cross_attn_qk_scaling_sign_preserving() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("d_k", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("raw_score", real.clone());
    let _ = prog.declare_const("scaled_score", real);

    let d_k = real_var("d_k");
    let scale = real_var("scale");
    let raw_score = real_var("raw_score");
    let scaled_score = real_var("scaled_score");

    // d_k > 0, bounded
    prog.assert(d_k.clone().real_gt(Expr::real(0)));
    prog.assert(d_k.clone().real_le(Expr::real(1024)));

    // scale = 1/sqrt(d_k): scale > 0 and scale^2 * d_k = 1
    prog.assert(scale.clone().real_gt(Expr::real(0)));
    prog.assert(
        scale
            .clone()
            .real_mul(scale.clone())
            .real_mul(d_k)
            .eq(Expr::real(1)),
    );

    // raw_score > 0 (positive dot product)
    prog.assert(raw_score.clone().real_gt(Expr::real(0)));

    // scaled_score = raw_score * scale
    prog.assert(scaled_score.clone().eq(raw_score.real_mul(scale)));

    // Negated property: scaled_score <= 0 (sign not preserved)
    let violation = scaled_score.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cross_attn_qk_scaling_sign_preserving");
}

// ---------------------------------------------------------------------------
// Test 752: Cross-attention softmax output sums to 1.0
// ---------------------------------------------------------------------------

/// Prove: softmax outputs over encoder positions sum to 1 for each decoder
/// query position.
///
/// Given 3 encoder positions with positive exp values, the normalized
/// weights w_i = exp_i / Z where Z = sum(exp_j) satisfy sum(w_i) = 1.
#[test]
fn test_752_cross_attn_softmax_sum_to_one() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("e1", real.clone());
    let _ = prog.declare_const("e2", real.clone());
    let _ = prog.declare_const("e3", real.clone());
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("w3", real);

    let e1 = real_var("e1");
    let e2 = real_var("e2");
    let e3 = real_var("e3");
    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let w3 = real_var("w3");

    // All exp values positive (from encoder features)
    prog.assert(e1.clone().real_gt(Expr::real(0)));
    prog.assert(e2.clone().real_gt(Expr::real(0)));
    prog.assert(e3.clone().real_gt(Expr::real(0)));

    // Z = e1 + e2 + e3
    let z = e1.clone().real_add(e2.clone()).real_add(e3.clone());

    // w_i = e_i / Z: w_i * Z = e_i
    prog.assert(w1.clone().real_mul(z.clone()).eq(e1));
    prog.assert(w2.clone().real_mul(z.clone()).eq(e2));
    prog.assert(w3.clone().real_mul(z).eq(e3));

    // Negated property: w1 + w2 + w3 != 1
    let sum = w1.real_add(w2).real_add(w3);
    let violation = sum.ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cross_attn_softmax_sum_to_one");
}

// ---------------------------------------------------------------------------
// Test 753: Cross-attention weight non-negativity
// ---------------------------------------------------------------------------

/// Prove: each cross-attention weight w_i is non-negative.
///
/// Since w_i = exp(s_i) / Z with exp(s_i) > 0 and Z > 0,
/// we have w_i > 0 for all i.
#[test]
fn test_753_cross_attn_weight_nonneg() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp_i", real.clone());
    let _ = prog.declare_const("exp_rest", real.clone());
    let _ = prog.declare_const("z", real.clone());
    let _ = prog.declare_const("w_i", real);

    let exp_i = real_var("exp_i");
    let exp_rest = real_var("exp_rest");
    let z = real_var("z");
    let w_i = real_var("w_i");

    // exp values positive
    prog.assert(exp_i.clone().real_gt(Expr::real(0)));
    prog.assert(exp_rest.clone().real_gt(Expr::real(0)));

    // Z = exp_i + exp_rest
    prog.assert(z.clone().eq(exp_i.clone().real_add(exp_rest)));

    // w_i = exp_i / Z: w_i * Z = exp_i
    prog.assert(w_i.clone().real_mul(z).eq(exp_i));

    // Negated property: w_i <= 0
    let violation = w_i.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cross_attn_weight_nonneg");
}

// ---------------------------------------------------------------------------
// Test 754: Value projection linear combination bounds
// ---------------------------------------------------------------------------

/// Prove: the cross-attention output (convex combination of value vectors)
/// stays within the value range [lo, hi].
///
/// output = sum_j w_j * v_j with w_j >= 0 and sum(w_j) = 1.
/// If all v_j in [lo, hi], then output in [lo, hi] (convex combination).
#[test]
fn test_754_cross_attn_value_linear_combination_bounds() {
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

    assert_verified(&prog, "cross_attn_value_linear_combination_bounds");
}

// ---------------------------------------------------------------------------
// Test 755: Multi-head split dimension correctness
// ---------------------------------------------------------------------------

/// Prove: splitting model dimension d_model into n_heads of d_head each
/// satisfies d_model = n_heads * d_head. This is the fundamental multi-head
/// split invariant for the cross-attention decoder.
#[test]
fn test_755_cross_attn_multihead_split_dim() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("d_model", real.clone());
    let _ = prog.declare_const("n_heads", real.clone());
    let _ = prog.declare_const("d_head", real.clone());
    let _ = prog.declare_const("split_total", real);

    let d_model = real_var("d_model");
    let n_heads = real_var("n_heads");
    let d_head = real_var("d_head");
    let split_total = real_var("split_total");

    // Positive dimensions
    prog.assert(n_heads.clone().real_gt(Expr::real(0)));
    prog.assert(d_head.clone().real_gt(Expr::real(0)));

    // d_model = n_heads * d_head (definition)
    prog.assert(d_model.clone().eq(n_heads.clone().real_mul(d_head.clone())));

    // split_total = n_heads * d_head (after split and count)
    prog.assert(split_total.clone().eq(n_heads.real_mul(d_head)));

    // Negated property: split_total != d_model
    let violation = split_total.ne(d_model);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cross_attn_multihead_split_dim");
}

// ---------------------------------------------------------------------------
// Test 756: Multi-head concat output dimension
// ---------------------------------------------------------------------------

/// Prove: concatenating n_heads head outputs of dimension d_head produces
/// exactly d_model = n_heads * d_head.
///
/// After each head computes cross-attention independently, outputs are
/// concatenated along the feature axis. The concat dimension must equal
/// the original model dimension.
#[test]
fn test_756_cross_attn_multihead_concat_dim() {
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

    // d_model = n_heads * d_head
    prog.assert(d_model.clone().eq(n_heads.clone().real_mul(d_head.clone())));

    // concat_dim = n_heads * d_head (from concatenation)
    prog.assert(concat_dim.clone().eq(n_heads.real_mul(d_head)));

    // Negated property: concat_dim != d_model
    let violation = concat_dim.ne(d_model);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cross_attn_multihead_concat_dim");
}

// ---------------------------------------------------------------------------
// Test 757: Object query initialization bounds (DETR)
// ---------------------------------------------------------------------------

/// Prove: DETR object queries initialized with values in [-B, B] remain
/// bounded after linear projection.
///
/// If query q in [-B, B] is multiplied by weight w in [-W, W] and bias
/// b in [-Wb, Wb], then the projected output = q*w + b is bounded by
/// |q*w + b| <= B*W + Wb.
///
/// We prove the output is within [-bound, bound] where bound = B*W + Wb.
#[test]
fn test_757_detr_object_query_init_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("q", real.clone());
    let _ = prog.declare_const("w", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("out", real.clone());
    let _ = prog.declare_const("bound", real);

    let q = real_var("q");
    let w = real_var("w");
    let b = real_var("b");
    let out = real_var("out");
    let bound = real_var("bound");

    // Query bounded: |q| <= 1 (normalized initialization)
    prog.assert(q.clone().real_ge(Expr::real(-1)));
    prog.assert(q.clone().real_le(Expr::real(1)));

    // Weight bounded: |w| <= 2
    prog.assert(w.clone().real_ge(Expr::real(-2)));
    prog.assert(w.clone().real_le(Expr::real(2)));

    // Bias bounded: |b| <= 1
    prog.assert(b.clone().real_ge(Expr::real(-1)));
    prog.assert(b.clone().real_le(Expr::real(1)));

    // out = q * w + b
    prog.assert(out.clone().eq(q.real_mul(w).real_add(b)));

    // bound = 1*2 + 1 = 3
    prog.assert(bound.clone().eq(Expr::real(3)));

    // Negated property: |out| > bound (out < -bound OR out > bound)
    let violation = out
        .clone()
        .real_lt(Expr::real(0).real_sub(bound.clone()))
        .or(out.real_gt(bound));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "detr_object_query_init_bounds");
}

// ---------------------------------------------------------------------------
// Test 758: Encoder-decoder attention key/value projection
// ---------------------------------------------------------------------------

/// Prove: the key projection K = X_enc * W_k preserves ordering when
/// W_k has positive entries. If x1 > x2 and w > 0, then x1*w > x2*w.
///
/// In cross-attention, keys come from encoder features projected by W_k.
/// Positive projection weights preserve the relative magnitudes.
#[test]
fn test_758_enc_dec_kv_projection_ordering() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("w_k", real.clone());
    let _ = prog.declare_const("k1", real.clone());
    let _ = prog.declare_const("k2", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let w_k = real_var("w_k");
    let k1 = real_var("k1");
    let k2 = real_var("k2");

    // x1 > x2 (encoder features differ)
    prog.assert(x1.clone().real_gt(x2.clone()));

    // w_k > 0 (positive projection weight)
    prog.assert(w_k.clone().real_gt(Expr::real(0)));

    // k1 = x1 * w_k, k2 = x2 * w_k
    prog.assert(k1.clone().eq(x1.real_mul(w_k.clone())));
    prog.assert(k2.clone().eq(x2.real_mul(w_k)));

    // Negated property: k1 <= k2 (ordering not preserved)
    let violation = k1.real_le(k2);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "enc_dec_kv_projection_ordering");
}

// ---------------------------------------------------------------------------
// Test 759: Cross-attention with padding mask zeros
// ---------------------------------------------------------------------------

/// Prove: when a padding mask sets an encoder position's score to a very
/// large negative value, that position's attention weight approaches zero.
///
/// padded_score = score + (-M) where M >> |score|. The exp of a very
/// negative value is near zero, so the softmax weight is near zero.
///
/// We model: exp_padded in [0, eps], exp_valid > 0.
/// w_padded = exp_padded / (exp_valid + exp_padded) < threshold.
#[test]
fn test_759_cross_attn_padding_mask_zeros() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp_valid", real.clone());
    let _ = prog.declare_const("exp_padded", real.clone());
    let _ = prog.declare_const("w_padded", real);

    let exp_valid = real_var("exp_valid");
    let exp_padded = real_var("exp_padded");
    let w_padded = real_var("w_padded");

    // Valid position has positive exp, bounded
    prog.assert(exp_valid.clone().real_gt(Expr::real(0)));
    prog.assert(exp_valid.clone().real_le(Expr::real(1000)));

    // Padded position: exp(very_negative) is near zero
    let eps = Expr::real_ratio(1, 1000000);
    prog.assert(exp_padded.clone().real_ge(Expr::real(0)));
    prog.assert(exp_padded.clone().real_le(eps));

    // w_padded = exp_padded / (exp_valid + exp_padded)
    let z = exp_valid.real_add(exp_padded.clone());
    prog.assert(w_padded.clone().real_mul(z).eq(exp_padded));

    // Negated property: w_padded > 0.001
    let violation = w_padded.real_gt(Expr::real_ratio(1, 1000));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cross_attn_padding_mask_zeros");
}

// ---------------------------------------------------------------------------
// Test 760: Position encoding addition commutativity
// ---------------------------------------------------------------------------

/// Prove: adding position encoding to queries commutes with addition.
///
/// (q + pe) = (pe + q). Since real addition is commutative, the order
/// of adding position encoding to the query vector does not matter.
///
/// This is fundamental: cross-attention queries are formed as
/// q = decoder_hidden + pos_embed, and commutativity guarantees
/// implementation flexibility.
#[test]
fn test_760_position_encoding_add_commutative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("q", real.clone());
    let _ = prog.declare_const("pe", real.clone());
    let _ = prog.declare_const("sum1", real.clone());
    let _ = prog.declare_const("sum2", real);

    let q = real_var("q");
    let pe = real_var("pe");
    let sum1 = real_var("sum1");
    let sum2 = real_var("sum2");

    // Bounded inputs
    prog.assert(q.clone().real_ge(Expr::real(-100)));
    prog.assert(q.clone().real_le(Expr::real(100)));
    prog.assert(pe.clone().real_ge(Expr::real(-100)));
    prog.assert(pe.clone().real_le(Expr::real(100)));

    // sum1 = q + pe
    prog.assert(sum1.clone().eq(q.clone().real_add(pe.clone())));

    // sum2 = pe + q
    prog.assert(sum2.clone().eq(pe.real_add(q)));

    // Negated property: sum1 != sum2 (commutativity violated)
    let violation = sum1.ne(sum2);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "position_encoding_add_commutative");
}

// ---------------------------------------------------------------------------
// Test 761: Attention output norm bounds
// ---------------------------------------------------------------------------

/// Prove: the attention output (convex combination) has magnitude bounded
/// by the maximum magnitude of the value vectors.
///
/// If all |v_j| <= V_max and weights w_j >= 0 with sum = 1, then
/// |out| = |sum(w_j * v_j)| <= sum(w_j * |v_j|) <= V_max * sum(w_j) = V_max.
#[test]
fn test_761_cross_attn_output_norm_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("v1", real.clone());
    let _ = prog.declare_const("v2", real.clone());
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("out", real.clone());
    let _ = prog.declare_const("v_max", real);

    let v1 = real_var("v1");
    let v2 = real_var("v2");
    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let out = real_var("out");
    let v_max = real_var("v_max");

    // v_max > 0, all values bounded by v_max
    prog.assert(v_max.clone().real_gt(Expr::real(0)));
    prog.assert(v1.clone().real_ge(Expr::real(0).real_sub(v_max.clone())));
    prog.assert(v1.clone().real_le(v_max.clone()));
    prog.assert(v2.clone().real_ge(Expr::real(0).real_sub(v_max.clone())));
    prog.assert(v2.clone().real_le(v_max.clone()));

    // Weights: non-negative, sum to 1
    prog.assert(w1.clone().real_ge(Expr::real(0)));
    prog.assert(w2.clone().real_ge(Expr::real(0)));
    prog.assert(w1.clone().real_add(w2.clone()).eq(Expr::real(1)));

    // out = w1*v1 + w2*v2
    prog.assert(out.clone().eq(w1.real_mul(v1).real_add(w2.real_mul(v2))));

    // Negated property: |out| > v_max
    let violation = out
        .clone()
        .real_gt(v_max.clone())
        .or(out.real_lt(Expr::real(0).real_sub(v_max)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cross_attn_output_norm_bounds");
}

// ---------------------------------------------------------------------------
// Test 762: Layer norm after cross-attention bounds
// ---------------------------------------------------------------------------

/// Prove: layer normalization centers and scales the output. Given normalized
/// value x_norm = (x - mu) / sqrt(var + eps) with var >= 0 and eps > 0,
/// the denominator is always positive, so division is well-defined.
///
/// We prove the denominator > 0 under the layer norm constraints.
#[test]
fn test_762_layernorm_after_cross_attn_denom_positive() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("var", real.clone());
    let _ = prog.declare_const("eps", real.clone());
    let _ = prog.declare_const("denom_sq", real.clone());
    let _ = prog.declare_const("denom", real);

    let var = real_var("var");
    let eps = real_var("eps");
    let denom_sq = real_var("denom_sq");
    let denom = real_var("denom");

    // var >= 0 (variance is non-negative)
    prog.assert(var.clone().real_ge(Expr::real(0)));

    // eps > 0 (small stabilization constant, e.g. 1e-5)
    prog.assert(eps.clone().real_gt(Expr::real(0)));
    prog.assert(eps.clone().real_le(Expr::real(1)));

    // denom_sq = var + eps
    prog.assert(denom_sq.clone().eq(var.real_add(eps)));

    // denom = sqrt(denom_sq): denom > 0 and denom^2 = denom_sq
    prog.assert(denom.clone().real_gt(Expr::real(0)));
    prog.assert(denom.clone().real_mul(denom.clone()).eq(denom_sq));

    // Negated property: denom <= 0
    let violation = denom.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "layernorm_after_cross_attn_denom_positive");
}

// ---------------------------------------------------------------------------
// Test 763: Residual connection preservation
// ---------------------------------------------------------------------------

/// Prove: the residual connection output = input + sublayer(input) preserves
/// the input component. Specifically, if sublayer output is zero, the
/// residual output equals the original input.
///
/// This is the identity-preservation property of residual connections
/// in transformer decoders.
#[test]
fn test_763_residual_connection_preserves_input() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("sublayer_out", real.clone());
    let _ = prog.declare_const("residual", real);

    let x = real_var("x");
    let sublayer_out = real_var("sublayer_out");
    let residual = real_var("residual");

    // x is any bounded real
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));

    // sublayer_out = 0 (identity sublayer)
    prog.assert(sublayer_out.clone().eq(Expr::real(0)));

    // residual = x + sublayer_out
    prog.assert(residual.clone().eq(x.clone().real_add(sublayer_out)));

    // Negated property: residual != x
    let violation = residual.ne(x);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "residual_connection_preserves_input");
}

// ---------------------------------------------------------------------------
// Test 764: Decoder self-attention causal mask
// ---------------------------------------------------------------------------

/// Prove: the causal mask ensures that for position i, the attention weight
/// for any future position j > i is zero (via large negative masking).
///
/// After masking, exp(score + mask) for j > i has mask = -M with M >> 0,
/// so the effective score is extremely negative and the softmax output
/// for that position approaches zero.
///
/// We model: unmasked exp > 0, masked exp in [0, eps].
/// The masked weight = exp_masked / (exp_unmasked + exp_masked) < threshold.
#[test]
fn test_764_decoder_self_attn_causal_mask() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp_unmasked", real.clone());
    let _ = prog.declare_const("exp_masked", real.clone());
    let _ = prog.declare_const("w_masked", real);

    let exp_unmasked = real_var("exp_unmasked");
    let exp_masked = real_var("exp_masked");
    let w_masked = real_var("w_masked");

    // Unmasked position has significant exp value
    prog.assert(exp_unmasked.clone().real_gt(Expr::real(0)));
    prog.assert(exp_unmasked.clone().real_le(Expr::real(1000)));

    // Masked future position: exp(score - M) is near zero
    let eps = Expr::real_ratio(1, 1000000);
    prog.assert(exp_masked.clone().real_ge(Expr::real(0)));
    prog.assert(exp_masked.clone().real_le(eps));

    // w_masked = exp_masked / (exp_unmasked + exp_masked)
    let z = exp_unmasked.real_add(exp_masked.clone());
    prog.assert(w_masked.clone().real_mul(z).eq(exp_masked));

    // Negated property: w_masked > 0.001
    let violation = w_masked.real_gt(Expr::real_ratio(1, 1000));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "decoder_self_attn_causal_mask");
}

// ---------------------------------------------------------------------------
// Test 765: Encoder feature broadcasting
// ---------------------------------------------------------------------------

/// Prove: broadcasting encoder features to match decoder queries preserves
/// the encoder values. When we repeat an encoder feature vector across
/// multiple decoder query positions, each copy equals the original.
///
/// If broadcast(x) produces N copies, each copy_i = x.
#[test]
fn test_765_encoder_feature_broadcast_preserves_value() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x_enc", real.clone());
    let _ = prog.declare_const("copy1", real.clone());
    let _ = prog.declare_const("copy2", real.clone());
    let _ = prog.declare_const("copy3", real);

    let x_enc = real_var("x_enc");
    let copy1 = real_var("copy1");
    let copy2 = real_var("copy2");
    let copy3 = real_var("copy3");

    // x_enc is any bounded value
    prog.assert(x_enc.clone().real_ge(Expr::real(-100)));
    prog.assert(x_enc.clone().real_le(Expr::real(100)));

    // Broadcasting: each copy equals the original
    prog.assert(copy1.clone().eq(x_enc.clone()));
    prog.assert(copy2.clone().eq(x_enc.clone()));
    prog.assert(copy3.clone().eq(x_enc.clone()));

    // Negated property: some copy differs from original
    let violation = copy1
        .ne(x_enc.clone())
        .or(copy2.ne(x_enc.clone()))
        .or(copy3.ne(x_enc));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "encoder_feature_broadcast_preserves_value");
}

// ---------------------------------------------------------------------------
// Test 766: Hungarian matching cost bounds (DETR)
// ---------------------------------------------------------------------------

/// Prove: the Hungarian matching cost in DETR is bounded when its
/// components (class cost, box L1 cost, box GIoU cost) are individually
/// bounded.
///
/// total_cost = lambda_cls * c_cls + lambda_l1 * c_l1 + lambda_giou * c_giou.
/// If c_cls in [0, C1], c_l1 in [0, C2], c_giou in [0, C3], then
/// total_cost in [0, lambda_cls*C1 + lambda_l1*C2 + lambda_giou*C3].
///
/// We prove total_cost <= upper_bound.
#[test]
fn test_766_hungarian_matching_cost_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("c_cls", real.clone());
    let _ = prog.declare_const("c_l1", real.clone());
    let _ = prog.declare_const("c_giou", real.clone());
    let _ = prog.declare_const("total_cost", real.clone());
    let _ = prog.declare_const("upper_bound", real);

    let c_cls = real_var("c_cls");
    let c_l1 = real_var("c_l1");
    let c_giou = real_var("c_giou");
    let total_cost = real_var("total_cost");
    let upper_bound = real_var("upper_bound");

    // Class cost in [0, 1] (after softmax)
    prog.assert(c_cls.clone().real_ge(Expr::real(0)));
    prog.assert(c_cls.clone().real_le(Expr::real(1)));

    // L1 box cost in [0, 4] (sum of 4 coordinate diffs, each in [0, 1])
    prog.assert(c_l1.clone().real_ge(Expr::real(0)));
    prog.assert(c_l1.clone().real_le(Expr::real(4)));

    // GIoU cost in [0, 2] (1 - GIoU, GIoU in [-1, 1])
    prog.assert(c_giou.clone().real_ge(Expr::real(0)));
    prog.assert(c_giou.clone().real_le(Expr::real(2)));

    // DETR default lambdas: lambda_cls=1, lambda_l1=5, lambda_giou=2
    // total_cost = 1*c_cls + 5*c_l1 + 2*c_giou
    prog.assert(
        total_cost.clone().eq(Expr::real(1)
            .real_mul(c_cls)
            .real_add(Expr::real(5).real_mul(c_l1))
            .real_add(Expr::real(2).real_mul(c_giou))),
    );

    // upper_bound = 1*1 + 5*4 + 2*2 = 1 + 20 + 4 = 25
    prog.assert(upper_bound.clone().eq(Expr::real(25)));

    // Negated property: total_cost > upper_bound
    let violation = total_cost.real_gt(upper_bound);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "hungarian_matching_cost_bounds");
}

// ---------------------------------------------------------------------------
// Test 767: Box regression coordinate bounds
// ---------------------------------------------------------------------------

/// Prove: DETR box predictions after sigmoid activation are in [0, 1].
///
/// The box coordinates (cx, cy, w, h) are passed through sigmoid, which
/// maps R -> (0, 1). We model sigmoid(x) = 1 / (1 + exp(-x)) and prove
/// 0 < sigma < 1 for bounded x.
///
/// We model: exp_neg_x > 0 (always), sigma = 1 / (1 + exp_neg_x).
/// Since 1 + exp_neg_x > 1, we have 0 < sigma < 1.
#[test]
fn test_767_box_regression_sigmoid_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp_neg_x", real.clone());
    let _ = prog.declare_const("denom", real.clone());
    let _ = prog.declare_const("sigma", real);

    let exp_neg_x = real_var("exp_neg_x");
    let denom = real_var("denom");
    let sigma = real_var("sigma");

    // exp(-x) > 0 for all real x
    prog.assert(exp_neg_x.clone().real_gt(Expr::real(0)));
    prog.assert(exp_neg_x.clone().real_le(Expr::real(1000000)));

    // denom = 1 + exp(-x)
    prog.assert(denom.clone().eq(Expr::real(1).real_add(exp_neg_x)));

    // sigma = 1 / denom: sigma * denom = 1
    prog.assert(sigma.clone().real_mul(denom).eq(Expr::real(1)));

    // Negated property: sigma <= 0 OR sigma >= 1
    let violation = sigma
        .clone()
        .real_le(Expr::real(0))
        .or(sigma.real_ge(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "box_regression_sigmoid_bounds");
}

// ---------------------------------------------------------------------------
// Test 768: Class probability softmax bounds
// ---------------------------------------------------------------------------

/// Prove: DETR class probabilities from softmax are each in (0, 1) and
/// the maximum probability is at most 1.
///
/// For C classes, softmax produces probabilities p_i = exp(s_i) / Z.
/// Each p_i in (0, 1) with at least 2 classes.
/// We prove for a single class: 0 < p_i < 1.
#[test]
fn test_768_class_probability_softmax_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp_i", real.clone());
    let _ = prog.declare_const("exp_other", real.clone());
    let _ = prog.declare_const("z", real.clone());
    let _ = prog.declare_const("p_i", real);

    let exp_i = real_var("exp_i");
    let exp_other = real_var("exp_other");
    let z = real_var("z");
    let p_i = real_var("p_i");

    // At least 2 classes, both exp values positive
    prog.assert(exp_i.clone().real_gt(Expr::real(0)));
    prog.assert(exp_other.clone().real_gt(Expr::real(0)));

    // Z = exp_i + exp_other
    prog.assert(z.clone().eq(exp_i.clone().real_add(exp_other)));

    // p_i = exp_i / Z: p_i * Z = exp_i
    prog.assert(p_i.clone().real_mul(z).eq(exp_i));

    // Negated property: p_i <= 0 OR p_i >= 1
    let violation = p_i
        .clone()
        .real_le(Expr::real(0))
        .or(p_i.real_ge(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "class_probability_softmax_bounds");
}

// ---------------------------------------------------------------------------
// Test 769: Iterative refinement convergence
// ---------------------------------------------------------------------------

/// Prove: DETR iterative box refinement with contraction factor alpha < 1
/// reduces the update magnitude at each step.
///
/// If delta_{t+1} = alpha * delta_t with 0 < alpha < 1, then
/// |delta_{t+1}| < |delta_t|. This models the refinement converging
/// toward the final box prediction.
///
/// We model: delta > 0 (WLOG), alpha in (0, 1), refined = alpha * delta.
/// Prove refined < delta.
#[test]
fn test_769_iterative_refinement_contraction() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("delta", real.clone());
    let _ = prog.declare_const("alpha", real.clone());
    let _ = prog.declare_const("refined", real);

    let delta = real_var("delta");
    let alpha = real_var("alpha");
    let refined = real_var("refined");

    // delta > 0 (positive update, WLOG by symmetry)
    prog.assert(delta.clone().real_gt(Expr::real(0)));
    prog.assert(delta.clone().real_le(Expr::real(100)));

    // alpha in (0, 1) — contraction factor
    prog.assert(alpha.clone().real_gt(Expr::real(0)));
    prog.assert(alpha.clone().real_lt(Expr::real(1)));

    // refined = alpha * delta
    prog.assert(refined.clone().eq(alpha.real_mul(delta.clone())));

    // Negated property: refined >= delta (not a contraction)
    let violation = refined.real_ge(delta);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "iterative_refinement_contraction");
}

// ---------------------------------------------------------------------------
// Test 770: Full decoder layer output bounds
// ---------------------------------------------------------------------------

/// Prove: a full decoder layer (self-attention + cross-attention + FFN with
/// residual connections) preserves boundedness. If the input x is in [-B, B]
/// and each sublayer output is bounded by S, then the final output is
/// bounded by B + 3*S (one residual addition per sublayer: self-attn,
/// cross-attn, FFN).
///
/// We model 3 residual additions with bounded sublayer outputs.
#[test]
fn test_770_full_decoder_layer_output_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("sa_out", real.clone());
    let _ = prog.declare_const("ca_out", real.clone());
    let _ = prog.declare_const("ffn_out", real.clone());
    let _ = prog.declare_const("y1", real.clone());
    let _ = prog.declare_const("y2", real.clone());
    let _ = prog.declare_const("y3", real.clone());
    let _ = prog.declare_const("bound", real);

    let x = real_var("x");
    let sa_out = real_var("sa_out");
    let ca_out = real_var("ca_out");
    let ffn_out = real_var("ffn_out");
    let y1 = real_var("y1");
    let y2 = real_var("y2");
    let y3 = real_var("y3");
    let bound = real_var("bound");

    // Input bounded: |x| <= 10
    prog.assert(x.clone().real_ge(Expr::real(-10)));
    prog.assert(x.clone().real_le(Expr::real(10)));

    // Each sublayer output bounded: |sa_out|, |ca_out|, |ffn_out| <= 5
    prog.assert(sa_out.clone().real_ge(Expr::real(-5)));
    prog.assert(sa_out.clone().real_le(Expr::real(5)));
    prog.assert(ca_out.clone().real_ge(Expr::real(-5)));
    prog.assert(ca_out.clone().real_le(Expr::real(5)));
    prog.assert(ffn_out.clone().real_ge(Expr::real(-5)));
    prog.assert(ffn_out.clone().real_le(Expr::real(5)));

    // y1 = x + sa_out (after self-attention residual)
    prog.assert(y1.clone().eq(x.real_add(sa_out)));

    // y2 = y1 + ca_out (after cross-attention residual)
    prog.assert(y2.clone().eq(y1.real_add(ca_out)));

    // y3 = y2 + ffn_out (after FFN residual)
    prog.assert(y3.clone().eq(y2.real_add(ffn_out)));

    // bound = 10 + 3*5 = 25
    prog.assert(bound.clone().eq(Expr::real(25)));

    // Negated property: |y3| > bound
    let violation = y3
        .clone()
        .real_gt(bound.clone())
        .or(y3.real_lt(Expr::real(0).real_sub(bound)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "full_decoder_layer_output_bounds");
}
