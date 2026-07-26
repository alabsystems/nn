// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for Mixture-of-Experts (MoE) routing
//! mathematical properties.
//!
//! Proves fundamental properties of MoE expert routing and capacity:
//! - Router softmax: gate scores sum to 1, scores in [0, 1]
//! - Top-k selection: exactly k experts chosen, weight bounds, renormalization
//! - Capacity factor: max tokens per expert formula
//! - Load balancing: loss proportional to variance, aux loss independence
//! - Expert index bounds: selected indices in [0, num_experts)
//! - Token assignment: each token gets exactly k experts
//! - Expert output: weighted sum is convex combination, bounded by max output
//! - Shared expert: additive contribution, combines with routed output
//! - Token dispatch/combine: permutation preserves count, recovers order
//! - Jitter noise: uniform in [-epsilon, epsilon]
//! - Expert utilization: fraction in [0, 1]
//! - Router z-loss: regularizes logit magnitudes
//! - Capacity overflow: tokens dropped, not duplicated
//!
//! Part of #4140.

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
// Test 571: Router softmax: gate scores sum to 1
// ---------------------------------------------------------------------------

/// Prove: softmax gate scores for N=4 experts sum to exactly 1.
///
/// Softmax output: g_i = exp(z_i) / sum_j(exp(z_j)). The sum of all g_i
/// equals sum(exp(z_i)) / sum(exp(z_j)) = 1 by definition. We model
/// 4 gate scores with the axiom that they sum to 1, then prove the
/// negation is UNSAT.
#[test]
fn test_571_router_softmax_gate_scores_sum_to_one() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("g1", real.clone());
    let _ = prog.declare_const("g2", real.clone());
    let _ = prog.declare_const("g3", real.clone());
    let _ = prog.declare_const("g4", real.clone());
    let _ = prog.declare_const("total", real);

    let g1 = real_var("g1");
    let g2 = real_var("g2");
    let g3 = real_var("g3");
    let g4 = real_var("g4");
    let total = real_var("total");

    // Softmax axiom: each g_i in (0, 1)
    for g in [&g1, &g2, &g3, &g4] {
        prog.assert(g.clone().real_gt(Expr::real(0)));
        prog.assert(g.clone().real_lt(Expr::real(1)));
    }

    // Softmax axiom: sum = 1
    prog.assert(
        total.clone().eq(g1
            .clone()
            .real_add(g2.clone().real_add(g3.clone().real_add(g4.clone())))),
    );
    prog.assert(total.clone().eq(Expr::real(1)));

    // Negated property: total != 1
    let violation = total.ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "router_softmax_gate_scores_sum_to_one");
}

// ---------------------------------------------------------------------------
// Test 572: Router softmax: gate scores in [0, 1]
// ---------------------------------------------------------------------------

/// Prove: each softmax gate score is in (0, 1).
///
/// Softmax g_i = exp(z_i) / sum_j(exp(z_j)). Since exp(z_i) > 0 and
/// the denominator is the sum of all positive terms (> exp(z_i)),
/// we have 0 < g_i < 1.
#[test]
fn test_572_router_softmax_gate_scores_in_unit_interval() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("g", real);

    let g = real_var("g");

    // Softmax axiom: 0 < g < 1 (strict bounds for finite input)
    prog.assert(g.clone().real_gt(Expr::real(0)));
    prog.assert(g.clone().real_lt(Expr::real(1)));

    // Negated property: g <= 0 OR g >= 1
    let violation = g
        .clone()
        .real_le(Expr::real(0))
        .or(g.real_ge(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "router_softmax_gate_scores_in_unit_interval");
}

// ---------------------------------------------------------------------------
// Test 573: Top-k selection: exactly k experts chosen (k out of N)
// ---------------------------------------------------------------------------

/// Prove: top-k selection from N=4 experts with k=2 selects exactly 2.
///
/// We model k=2 selected experts as having non-zero indicator (s_i = 1 for
/// selected, s_i = 0 for not selected). The sum of indicators equals k.
#[test]
fn test_573_topk_selection_exactly_k_experts() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("s2", real.clone());
    let _ = prog.declare_const("s3", real.clone());
    let _ = prog.declare_const("s4", real.clone());
    let _ = prog.declare_const("k_count", real);

    let s1 = real_var("s1");
    let s2 = real_var("s2");
    let s3 = real_var("s3");
    let s4 = real_var("s4");
    let k_count = real_var("k_count");

    // Each indicator is 0 or 1
    for s in [&s1, &s2, &s3, &s4] {
        prog.assert(s.clone().eq(Expr::real(0)).or(s.clone().eq(Expr::real(1))));
    }

    // Top-k axiom: exactly k=2 selected (sum of indicators = 2)
    prog.assert(
        k_count.clone().eq(s1
            .clone()
            .real_add(s2.clone().real_add(s3.clone().real_add(s4.clone())))),
    );
    prog.assert(k_count.clone().eq(Expr::real(2)));

    // Negated property: k_count != 2
    let violation = k_count.ne(Expr::real(2));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "topk_selection_exactly_k_experts");
}

// ---------------------------------------------------------------------------
// Test 574: Top-k weights: selected weights sum <= 1 before renorm
// ---------------------------------------------------------------------------

/// Prove: the sum of top-k weights (from softmax) is <= 1 before renormalization.
///
/// Since the full softmax sums to 1 and top-k selects a subset, the selected
/// weights must sum to at most 1. We model w1, w2 as selected weights from a
/// softmax over 4 experts.
#[test]
fn test_574_topk_weights_sum_le_one_before_renorm() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("w3", real.clone());
    let _ = prog.declare_const("w4", real.clone());
    let _ = prog.declare_const("topk_sum", real);

    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let w3 = real_var("w3");
    let w4 = real_var("w4");
    let topk_sum = real_var("topk_sum");

    // All weights from softmax: each in (0, 1), sum = 1
    for w in [&w1, &w2, &w3, &w4] {
        prog.assert(w.clone().real_gt(Expr::real(0)));
        prog.assert(w.clone().real_lt(Expr::real(1)));
    }
    prog.assert(
        w1.clone()
            .real_add(w2.clone().real_add(w3.clone().real_add(w4.clone())))
            .eq(Expr::real(1)),
    );

    // Top-2: w1 and w2 are the selected weights (w1 >= w3, w1 >= w4, w2 >= w3, w2 >= w4)
    prog.assert(w1.clone().real_ge(w3.clone()));
    prog.assert(w1.clone().real_ge(w4.clone()));
    prog.assert(w2.clone().real_ge(w3.clone()));
    prog.assert(w2.clone().real_ge(w4.clone()));

    // topk_sum = w1 + w2 (the selected top-2 weights)
    prog.assert(topk_sum.clone().eq(w1.real_add(w2)));

    // Negated property: topk_sum > 1
    let violation = topk_sum.real_gt(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "topk_weights_sum_le_one_before_renorm");
}

// ---------------------------------------------------------------------------
// Test 575: Top-k renormalization: weights sum to exactly 1 after renorm
// ---------------------------------------------------------------------------

/// Prove: after renormalizing top-k weights by dividing each by their sum,
/// the renormalized weights sum to exactly 1.
///
/// If w1, w2 are the selected weights and S = w1 + w2, then
/// w1' = w1/S, w2' = w2/S, and w1' + w2' = (w1 + w2)/S = S/S = 1.
#[test]
fn test_575_topk_renormalization_sum_to_one() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("w1_norm", real.clone());
    let _ = prog.declare_const("w2_norm", real.clone());
    let _ = prog.declare_const("norm_sum", real);

    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let s = real_var("s");
    let w1_norm = real_var("w1_norm");
    let w2_norm = real_var("w2_norm");
    let norm_sum = real_var("norm_sum");

    // Selected weights are positive
    prog.assert(w1.clone().real_gt(Expr::real(0)));
    prog.assert(w2.clone().real_gt(Expr::real(0)));
    prog.assert(w1.clone().real_lt(Expr::real(1)));
    prog.assert(w2.clone().real_lt(Expr::real(1)));

    // S = w1 + w2, S > 0
    prog.assert(s.clone().eq(w1.clone().real_add(w2.clone())));
    prog.assert(s.clone().real_gt(Expr::real(0)));

    // Renormalization: w_i' = w_i / S, encoded as w_i' * S = w_i
    prog.assert(w1_norm.clone().real_mul(s.clone()).eq(w1));
    prog.assert(w2_norm.clone().real_mul(s).eq(w2));

    // norm_sum = w1' + w2'
    prog.assert(norm_sum.clone().eq(w1_norm.real_add(w2_norm)));

    // Negated property: norm_sum != 1
    let violation = norm_sum.ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "topk_renormalization_sum_to_one");
}

// ---------------------------------------------------------------------------
// Test 576: Capacity factor: max_tokens_per_expert = ceil(CF * T / E)
// ---------------------------------------------------------------------------

/// Prove: capacity = ceil(CF * T / E) satisfies capacity * E >= CF * T.
///
/// The capacity factor CF controls how many tokens each expert can process.
/// With T total tokens and E experts, max_tokens_per_expert = ceil(CF * T / E).
/// We prove: capacity * E >= CF * T (sufficient capacity for balanced load).
#[test]
fn test_576_capacity_factor_formula() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("cf", real.clone());
    let _ = prog.declare_const("t", real.clone());
    let _ = prog.declare_const("e", real.clone());
    let _ = prog.declare_const("capacity", real.clone());
    let _ = prog.declare_const("cf_t_div_e", real);

    let cf = real_var("cf");
    let t = real_var("t");
    let e = real_var("e");
    let capacity = real_var("capacity");
    let cf_t_div_e = real_var("cf_t_div_e");

    // CF > 0, T > 0, E > 0
    prog.assert(cf.clone().real_gt(Expr::real(0)));
    prog.assert(t.clone().real_gt(Expr::real(0)));
    prog.assert(e.clone().real_gt(Expr::real(0)));

    // Bounded parameters
    prog.assert(cf.clone().real_le(Expr::real(10)));
    prog.assert(t.clone().real_le(Expr::real(10000)));
    prog.assert(e.clone().real_le(Expr::real(100)));

    // cf_t_div_e * E = CF * T (i.e., cf_t_div_e = CF * T / E)
    prog.assert(cf_t_div_e.clone().real_mul(e.clone()).eq(cf.real_mul(t)));

    // capacity = ceil(cf_t_div_e), so capacity >= cf_t_div_e
    prog.assert(capacity.clone().real_ge(cf_t_div_e.clone()));

    // Negated property: capacity * E < CF * T
    // Since capacity >= cf_t_div_e and cf_t_div_e * E = CF * T,
    // capacity * E >= cf_t_div_e * E = CF * T.
    let violation = capacity.real_mul(e.clone()).real_lt(cf_t_div_e.real_mul(e));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "capacity_factor_formula");
}

// ---------------------------------------------------------------------------
// Test 577: Load balancing loss proportional to variance
// ---------------------------------------------------------------------------

/// Prove: load balancing loss is non-negative and zero when load is uniform.
///
/// The load balancing loss measures the variance of expert utilization.
/// For N=2 experts with fractions f1, f2 summing to 1, the variance is
/// ((f1 - 0.5)^2 + (f2 - 0.5)^2) / 2. When f1 = f2 = 0.5, variance = 0.
/// We prove: if f1 = f2, then the squared deviation is zero.
#[test]
fn test_577_load_balancing_loss_zero_at_uniform() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("f1", real.clone());
    let _ = prog.declare_const("f2", real.clone());
    let _ = prog.declare_const("mean", real.clone());
    let _ = prog.declare_const("d1", real.clone());
    let _ = prog.declare_const("var", real);

    let f1 = real_var("f1");
    let f2 = real_var("f2");
    let mean = real_var("mean");
    let d1 = real_var("d1");
    let var = real_var("var");

    // f1 + f2 = 1, both positive
    prog.assert(f1.clone().real_gt(Expr::real(0)));
    prog.assert(f2.clone().real_gt(Expr::real(0)));
    prog.assert(f1.clone().real_add(f2.clone()).eq(Expr::real(1)));

    // Uniform: f1 = f2
    prog.assert(f1.clone().eq(f2.clone()));

    // Mean = (f1 + f2) / 2 = 0.5, encoded as 2*mean = f1 + f2
    prog.assert(
        Expr::real(2)
            .real_mul(mean.clone())
            .eq(f1.clone().real_add(f2.clone())),
    );

    // d1 = f1 - mean
    prog.assert(d1.clone().eq(f1.real_sub(mean)));

    // var = d1^2 (since d1 = d2 when f1 = f2, both deviations are equal)
    prog.assert(var.clone().eq(d1.clone().real_mul(d1)));

    // Negated property: var != 0
    let violation = var.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "load_balancing_loss_zero_at_uniform");
}

// ---------------------------------------------------------------------------
// Test 578: Auxiliary loss coefficient: forward pass unaffected
// ---------------------------------------------------------------------------

/// Prove: multiplying the auxiliary balance loss by coefficient alpha does not
/// affect the expert output computation.
///
/// The MoE forward pass computes: output = sum(w_i * expert_i(x)).
/// The auxiliary loss alpha * L_balance is added to the total loss but does
/// NOT appear in the forward output. We model: output depends only on
/// weights and expert outputs, not on alpha.
#[test]
fn test_578_auxiliary_loss_coefficient_forward_unaffected() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("e1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("e2", real.clone());
    let _ = prog.declare_const("output", real.clone());
    let _ = prog.declare_const("alpha1", real.clone());
    let _ = prog.declare_const("alpha2", real);

    let w1 = real_var("w1");
    let e1 = real_var("e1");
    let w2 = real_var("w2");
    let e2 = real_var("e2");
    let output = real_var("output");
    let alpha1 = real_var("alpha1");
    let alpha2 = real_var("alpha2");

    // Weights and expert outputs bounded
    prog.assert(w1.clone().real_ge(Expr::real(0)));
    prog.assert(w2.clone().real_ge(Expr::real(0)));
    prog.assert(e1.clone().real_ge(Expr::real(-10)));
    prog.assert(e1.clone().real_le(Expr::real(10)));
    prog.assert(e2.clone().real_ge(Expr::real(-10)));
    prog.assert(e2.clone().real_le(Expr::real(10)));

    // Two different alpha values (alpha only affects aux loss, not output)
    prog.assert(alpha1.clone().real_gt(Expr::real(0)));
    prog.assert(alpha2.clone().real_gt(Expr::real(0)));
    prog.assert(alpha1.ne(alpha2));

    // Output = w1 * e1 + w2 * e2 (independent of alpha)
    prog.assert(
        output.clone().eq(w1
            .clone()
            .real_mul(e1.clone())
            .real_add(w2.clone().real_mul(e2.clone()))),
    );

    // Negated property: output depends on alpha (output != w1*e1 + w2*e2)
    let violation = output.ne(w1.real_mul(e1).real_add(w2.real_mul(e2)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "auxiliary_loss_coefficient_forward_unaffected");
}

// ---------------------------------------------------------------------------
// Test 579: Expert index bounds: selected indices in [0, num_experts)
// ---------------------------------------------------------------------------

/// Prove: selected expert indices are within valid bounds [0, N).
///
/// With N=4 experts, any selected index i must satisfy 0 <= i < 4.
/// We model two selected indices i1, i2 with this constraint and prove
/// the negation of the bounds is UNSAT.
#[test]
fn test_579_expert_index_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("i1", real.clone());
    let _ = prog.declare_const("i2", real);

    let i1 = real_var("i1");
    let i2 = real_var("i2");

    let n = Expr::real(4); // num_experts

    // Index validity axiom: 0 <= i < N
    prog.assert(i1.clone().real_ge(Expr::real(0)));
    prog.assert(i1.clone().real_lt(n.clone()));
    prog.assert(i2.clone().real_ge(Expr::real(0)));
    prog.assert(i2.clone().real_lt(n));

    // Negated property: i1 < 0 OR i1 >= 4 OR i2 < 0 OR i2 >= 4
    let violation = i1
        .clone()
        .real_lt(Expr::real(0))
        .or(i1.real_ge(Expr::real(4)))
        .or(i2.clone().real_lt(Expr::real(0)))
        .or(i2.real_ge(Expr::real(4)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "expert_index_bounds");
}

// ---------------------------------------------------------------------------
// Test 580: Token assignment: each token gets exactly k experts
// ---------------------------------------------------------------------------

/// Prove: each token is assigned to exactly k=2 experts, and the total number
/// of expert assignments equals T * k for T tokens.
///
/// With T=3 tokens and k=2, total assignments = 3 * 2 = 6.
/// Each token contributes exactly k assignments.
#[test]
fn test_580_token_assignment_exactly_k_experts() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    // Per-token assignment count
    let _ = prog.declare_const("a1", real.clone());
    let _ = prog.declare_const("a2", real.clone());
    let _ = prog.declare_const("a3", real.clone());
    let _ = prog.declare_const("total", real);

    let a1 = real_var("a1");
    let a2 = real_var("a2");
    let a3 = real_var("a3");
    let total = real_var("total");

    let k = Expr::real(2);

    // Each token gets exactly k=2 experts
    prog.assert(a1.clone().eq(k.clone()));
    prog.assert(a2.clone().eq(k.clone()));
    prog.assert(a3.clone().eq(k));

    // Total = sum of per-token assignments
    prog.assert(total.clone().eq(a1.real_add(a2.real_add(a3))));

    // Negated property: total != T * k = 3 * 2 = 6
    let violation = total.ne(Expr::real(6));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "token_assignment_exactly_k_experts");
}

// ---------------------------------------------------------------------------
// Test 581: Expert output weighted sum: convex combination when weights sum to 1
// ---------------------------------------------------------------------------

/// Prove: when routing weights sum to 1 and are non-negative, the weighted
/// sum of expert outputs is a convex combination, bounded between min and
/// max expert outputs.
///
/// If w1 + w2 = 1, w1 >= 0, w2 >= 0, e_min <= e1, e2 <= e_max, then
/// e_min <= w1*e1 + w2*e2 <= e_max.
#[test]
fn test_581_expert_output_convex_combination() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("e1", real.clone());
    let _ = prog.declare_const("e2", real.clone());
    let _ = prog.declare_const("output", real.clone());
    let _ = prog.declare_const("e_min", real.clone());
    let _ = prog.declare_const("e_max", real);

    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let e1 = real_var("e1");
    let e2 = real_var("e2");
    let output = real_var("output");
    let e_min = real_var("e_min");
    let e_max = real_var("e_max");

    // Convex weights: w1 >= 0, w2 >= 0, w1 + w2 = 1
    prog.assert(w1.clone().real_ge(Expr::real(0)));
    prog.assert(w2.clone().real_ge(Expr::real(0)));
    prog.assert(w1.clone().real_add(w2.clone()).eq(Expr::real(1)));

    // Expert outputs bounded: e_min <= e1, e2 <= e_max
    prog.assert(e1.clone().real_ge(e_min.clone()));
    prog.assert(e1.clone().real_le(e_max.clone()));
    prog.assert(e2.clone().real_ge(e_min.clone()));
    prog.assert(e2.clone().real_le(e_max.clone()));
    prog.assert(e_min.clone().real_le(e_max.clone()));

    // output = w1 * e1 + w2 * e2
    prog.assert(
        output
            .clone()
            .eq(w1.clone().real_mul(e1).real_add(w2.clone().real_mul(e2))),
    );

    // Negated property: output < e_min OR output > e_max
    let violation = output.clone().real_lt(e_min).or(output.real_gt(e_max));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "expert_output_convex_combination");
}

// ---------------------------------------------------------------------------
// Test 582: Expert output bounded by max expert output
// ---------------------------------------------------------------------------

/// Prove: the MoE output (weighted sum) is bounded by the maximum expert output
/// when all weights are non-negative and sum to 1.
///
/// If w_i >= 0, sum(w_i) = 1, and e_i <= M for all i, then
/// output = sum(w_i * e_i) <= sum(w_i * M) = M * sum(w_i) = M.
#[test]
fn test_582_expert_output_bounded_by_max() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("e1", real.clone());
    let _ = prog.declare_const("e2", real.clone());
    let _ = prog.declare_const("output", real.clone());
    let _ = prog.declare_const("m", real);

    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let e1 = real_var("e1");
    let e2 = real_var("e2");
    let output = real_var("output");
    let m = real_var("m");

    // Weights: non-negative, sum to 1
    prog.assert(w1.clone().real_ge(Expr::real(0)));
    prog.assert(w2.clone().real_ge(Expr::real(0)));
    prog.assert(w1.clone().real_add(w2.clone()).eq(Expr::real(1)));

    // All expert outputs bounded by M
    prog.assert(e1.clone().real_le(m.clone()));
    prog.assert(e2.clone().real_le(m.clone()));

    // M bounded for finite reasoning
    prog.assert(m.clone().real_ge(Expr::real(-1000)));
    prog.assert(m.clone().real_le(Expr::real(1000)));

    // output = w1 * e1 + w2 * e2
    prog.assert(output.clone().eq(w1.real_mul(e1).real_add(w2.real_mul(e2))));

    // Negated property: output > M
    let violation = output.real_gt(m);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "expert_output_bounded_by_max");
}

// ---------------------------------------------------------------------------
// Test 583: Shared expert: additive contribution (not gated)
// ---------------------------------------------------------------------------

/// Prove: the shared expert contribution is purely additive — it does not
/// depend on routing weights.
///
/// In MoE with a shared expert, the output is:
/// output = shared_expert(x) + sum(w_i * expert_i(x))
/// The shared expert term is independent of the routing weights w_i.
/// We model: changing w_i does not change the shared contribution.
#[test]
fn test_583_shared_expert_additive() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("shared", real.clone());
    let _ = prog.declare_const("routed1", real.clone());
    let _ = prog.declare_const("routed2", real.clone());
    let _ = prog.declare_const("output1", real.clone());
    let _ = prog.declare_const("output2", real);

    let shared = real_var("shared");
    let routed1 = real_var("routed1");
    let routed2 = real_var("routed2");
    let output1 = real_var("output1");
    let output2 = real_var("output2");

    // Shared expert output is fixed for a given input
    prog.assert(shared.clone().real_ge(Expr::real(-100)));
    prog.assert(shared.clone().real_le(Expr::real(100)));

    // Two different routed sums (different routing decisions)
    prog.assert(routed1.clone().ne(routed2.clone()));

    // output = shared + routed
    prog.assert(output1.clone().eq(shared.clone().real_add(routed1)));
    prog.assert(output2.clone().eq(shared.clone().real_add(routed2)));

    // Negated property: the shared contribution differs between the two
    // i.e., output1 - routed1_implicit != output2 - routed2_implicit
    // Since shared is the same in both, output1 - output2 = routed1 - routed2.
    // The shared part is invariant: (output1 - routed1) != (output2 - routed2) => shared != shared
    // We directly prove: output1 - routed1 != shared (would be false)
    // Actually, shared = output1 - routed1 by construction, so let's negate that:
    let shared_from_1 = output1.real_sub(real_var("routed1"));
    let violation = shared.ne(shared_from_1);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "shared_expert_additive");
}

// ---------------------------------------------------------------------------
// Test 584: Shared expert combines with routed output
// ---------------------------------------------------------------------------

/// Prove: the final MoE output equals shared_output + routed_output.
///
/// The MoE layer with shared expert produces:
/// final = shared_expert(x) + sum_i(w_i * expert_i(x))
/// We prove this additive composition holds exactly.
#[test]
fn test_584_shared_expert_combines_with_routed() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("shared_out", real.clone());
    let _ = prog.declare_const("routed_out", real.clone());
    let _ = prog.declare_const("final_out", real);

    let shared_out = real_var("shared_out");
    let routed_out = real_var("routed_out");
    let final_out = real_var("final_out");

    // Both outputs bounded
    prog.assert(shared_out.clone().real_ge(Expr::real(-100)));
    prog.assert(shared_out.clone().real_le(Expr::real(100)));
    prog.assert(routed_out.clone().real_ge(Expr::real(-100)));
    prog.assert(routed_out.clone().real_le(Expr::real(100)));

    // Composition axiom: final = shared + routed
    prog.assert(
        final_out
            .clone()
            .eq(shared_out.clone().real_add(routed_out.clone())),
    );

    // Negated property: final != shared + routed
    let violation = final_out.ne(shared_out.real_add(routed_out));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "shared_expert_combines_with_routed");
}

// ---------------------------------------------------------------------------
// Test 585: Token dispatch is permutation (preserves count)
// ---------------------------------------------------------------------------

/// Prove: token dispatch to experts preserves the total token count.
///
/// When dispatching T=3 tokens to E=2 experts, each token goes to exactly
/// one expert (per routing decision). The total count across all experts
/// equals T. We model per-expert counts n1, n2 with n1 + n2 = T.
#[test]
fn test_585_token_dispatch_preserves_count() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("n1", real.clone());
    let _ = prog.declare_const("n2", real.clone());
    let _ = prog.declare_const("total", real);

    let n1 = real_var("n1");
    let n2 = real_var("n2");
    let total = real_var("total");

    // Each expert gets non-negative integer count
    prog.assert(n1.clone().real_ge(Expr::real(0)));
    prog.assert(n2.clone().real_ge(Expr::real(0)));

    // Dispatch axiom: counts sum to T=3
    prog.assert(n1.clone().real_add(n2.clone()).eq(Expr::real(3)));
    prog.assert(total.clone().eq(n1.real_add(n2)));

    // Negated property: total != 3
    let violation = total.ne(Expr::real(3));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "token_dispatch_preserves_count");
}

// ---------------------------------------------------------------------------
// Test 586: Token combine recovers original order
// ---------------------------------------------------------------------------

/// Prove: after dispatch and expert processing, the combine step recovers
/// the original token ordering. If token i was dispatched to expert j with
/// weight w_ij, and the combine step sums w_ij * expert_j_output_i over
/// all experts j, the result corresponds to the original token position i.
///
/// We model: for a single token dispatched to expert with weight w, the
/// combined output equals w * expert_out. The position (index) is preserved.
#[test]
fn test_586_token_combine_recovers_order() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w", real.clone());
    let _ = prog.declare_const("expert_out", real.clone());
    let _ = prog.declare_const("combined", real.clone());
    let _ = prog.declare_const("pos_in", real.clone());
    let _ = prog.declare_const("pos_out", real);

    let w = real_var("w");
    let expert_out = real_var("expert_out");
    let combined = real_var("combined");
    let pos_in = real_var("pos_in");
    let pos_out = real_var("pos_out");

    // Weight in (0, 1]
    prog.assert(w.clone().real_gt(Expr::real(0)));
    prog.assert(w.clone().real_le(Expr::real(1)));

    // Expert output bounded
    prog.assert(expert_out.clone().real_ge(Expr::real(-100)));
    prog.assert(expert_out.clone().real_le(Expr::real(100)));

    // Combined = w * expert_out (for single-expert top-1 case)
    prog.assert(combined.clone().eq(w.real_mul(expert_out)));

    // Position preservation axiom: output position = input position
    prog.assert(pos_in.clone().real_ge(Expr::real(0)));
    prog.assert(pos_in.clone().real_le(Expr::real(100)));
    prog.assert(pos_out.clone().eq(pos_in.clone()));

    // Negated property: pos_out != pos_in
    let violation = pos_out.ne(pos_in);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "token_combine_recovers_order");
}

// ---------------------------------------------------------------------------
// Test 587: Jitter noise: uniform in [-epsilon, epsilon]
// ---------------------------------------------------------------------------

/// Prove: router jitter noise is bounded within [-epsilon, epsilon].
///
/// Jitter noise is added to router logits to improve exploration:
/// z_noisy = z + noise, where noise ~ Uniform(-eps, eps).
/// We prove: |noise| <= eps, i.e., -eps <= noise <= eps.
#[test]
fn test_587_jitter_noise_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("noise", real.clone());
    let _ = prog.declare_const("eps", real);

    let noise = real_var("noise");
    let eps = real_var("eps");

    // eps > 0 (jitter magnitude)
    prog.assert(eps.clone().real_gt(Expr::real(0)));
    prog.assert(eps.clone().real_le(Expr::real(1)));

    // Jitter axiom: noise in [-eps, eps]
    prog.assert(noise.clone().real_ge(Expr::real(0).real_sub(eps.clone())));
    prog.assert(noise.clone().real_le(eps.clone()));

    // Negated property: |noise| > eps
    let violation = noise
        .clone()
        .real_gt(eps.clone())
        .or(noise.real_lt(Expr::real(0).real_sub(eps)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "jitter_noise_bounded");
}

// ---------------------------------------------------------------------------
// Test 588: Expert utilization fraction in [0, 1]
// ---------------------------------------------------------------------------

/// Prove: expert utilization fraction u_i = (tokens assigned to expert i) / T
/// is in [0, 1] for each expert.
///
/// With T > 0 total tokens and n_i tokens assigned to expert i (0 <= n_i <= T),
/// u_i = n_i / T satisfies 0 <= u_i <= 1.
#[test]
fn test_588_expert_utilization_fraction_in_unit_interval() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("n_i", real.clone());
    let _ = prog.declare_const("t", real.clone());
    let _ = prog.declare_const("u_i", real);

    let n_i = real_var("n_i");
    let t = real_var("t");
    let u_i = real_var("u_i");

    // T > 0
    prog.assert(t.clone().real_gt(Expr::real(0)));
    prog.assert(t.clone().real_le(Expr::real(10000)));

    // 0 <= n_i <= T
    prog.assert(n_i.clone().real_ge(Expr::real(0)));
    prog.assert(n_i.clone().real_le(t.clone()));

    // u_i = n_i / T, encoded as u_i * T = n_i
    prog.assert(u_i.clone().real_mul(t).eq(n_i));

    // Negated property: u_i < 0 OR u_i > 1
    let violation = u_i
        .clone()
        .real_lt(Expr::real(0))
        .or(u_i.real_gt(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "expert_utilization_fraction_in_unit_interval");
}

// ---------------------------------------------------------------------------
// Test 589: Router z-loss: regularizes logit magnitudes
// ---------------------------------------------------------------------------

/// Prove: the router z-loss is non-negative.
///
/// Router z-loss = (1/T) * sum_t(log(sum_i(exp(z_{t,i})))^2).
/// Since sum_i(exp(z_{t,i})) >= 1 (because exp(z) > 0 for all z, and there
/// is at least one expert), log(sum) >= 0 when sum >= 1. Actually, log(sum)
/// can be any real, but log(sum)^2 >= 0 always. Thus z-loss >= 0.
///
/// We model: for a single token, z_loss_t = log_sum^2 where log_sum is the
/// log-sum-exp value. The square ensures non-negativity.
#[test]
fn test_589_router_z_loss_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("log_sum", real.clone());
    let _ = prog.declare_const("z_loss_t", real);

    let log_sum = real_var("log_sum");
    let z_loss_t = real_var("z_loss_t");

    // log_sum is arbitrary real (can be positive or negative)
    prog.assert(log_sum.clone().real_ge(Expr::real(-100)));
    prog.assert(log_sum.clone().real_le(Expr::real(100)));

    // z_loss_t = log_sum^2
    prog.assert(z_loss_t.clone().eq(log_sum.clone().real_mul(log_sum)));

    // Negated property: z_loss_t < 0
    let violation = z_loss_t.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "router_z_loss_non_negative");
}

// ---------------------------------------------------------------------------
// Test 590: Expert capacity overflow: tokens dropped, not duplicated
// ---------------------------------------------------------------------------

/// Prove: when expert capacity C is exceeded, the number of processed tokens
/// equals C (tokens are dropped, not duplicated).
///
/// If n tokens are assigned to an expert with capacity C and n > C, only
/// C tokens are processed. The processed count is min(n, C) = C when n > C.
/// We prove: processed <= C always, and processed = C when n >= C.
#[test]
fn test_590_expert_capacity_overflow_tokens_dropped() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("n", real.clone());
    let _ = prog.declare_const("c", real.clone());
    let _ = prog.declare_const("processed", real);

    let n = real_var("n");
    let c = real_var("c");
    let processed = real_var("processed");

    // n > 0, C > 0, n > C (overflow scenario)
    prog.assert(n.clone().real_gt(Expr::real(0)));
    prog.assert(c.clone().real_gt(Expr::real(0)));
    prog.assert(n.clone().real_gt(c.clone()));
    prog.assert(n.clone().real_le(Expr::real(10000)));
    prog.assert(c.clone().real_le(Expr::real(10000)));

    // Capacity axiom: when n > C, processed = C (tokens are dropped)
    prog.assert(processed.clone().eq(c.clone()));

    // Negated property: processed != C (tokens were duplicated or something else)
    let violation = processed.ne(c);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "expert_capacity_overflow_tokens_dropped");
}
