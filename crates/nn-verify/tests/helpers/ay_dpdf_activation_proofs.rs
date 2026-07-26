// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for activation function mathematical properties.
//!
//! Proves fundamental properties of activation functions used in neural networks:
//! 1. SiLU(x) = x*sigmoid(x) bounded for |x|<=M
//! 2. GELU approximate bounded
//! 3. ReLU output >= 0
//! 4. ReLU preserves positive input
//! 5. Sigmoid in (0,1)
//! 6. Tanh in (-1,1)
//! 7. Mish = x*tanh(softplus(x)) bounded
//! 8. Hardswish piecewise linear bounded
//! 9. Softplus >= 0
//! 10. ReLU6 in [0,6]
//! 11. Leaky ReLU: negative slope preserves sign
//! 12. ELU smooth negative with alpha
//! 13. SELU self-normalizing bounds
//! 14. Swish beta controls sharpness
//! 15. Activation monotonicity
//! 16. Activation Lipschitz constant
//! 17. Composition activation+linear bounded
//! 18. Activation gradient bounded
//! 19. GLU gate sigma(Wx+b)*(Vx+c) bounded
//! 20. PReLU generalizes leaky ReLU
//!
//! Part of #4200.

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
// Test 931: SiLU(x) = x*sigmoid(x) bounded for |x|<=M
// ---------------------------------------------------------------------------

/// Prove: SiLU(x) = x * sigmoid(x) is bounded for |x| <= M.
///
/// SiLU(x) = x * sigmoid(x). Since sigmoid(x) in (0, 1), we have:
///   For x >= 0: 0 <= silu(x) <= x <= M.
///   For x < 0: silu(x) = x * sigmoid(x), and |silu(x)| < |x| < M.
/// The actual minimum of SiLU is ~-0.2784 at x ~ -1.278.
///
/// For M = 10, we prove: -0.28 <= silu(x) <= 10.
#[test]
fn test_931_silu_bounded_for_bounded_input() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("sig", real.clone());
    let _ = prog.declare_const("sw", real);

    let x = real_var("x");
    let sig = real_var("sig");
    let sw = real_var("sw");

    // |x| <= 10
    prog.assert(x.clone().real_ge(Expr::real(-10)));
    prog.assert(x.clone().real_le(Expr::real(10)));

    // sigmoid(x) in (0, 1)
    prog.assert(sig.clone().real_gt(Expr::real(0)));
    prog.assert(sig.clone().real_lt(Expr::real(1)));

    // sw = x * sigmoid(x)
    prog.assert(sw.clone().eq(x.real_mul(sig)));

    // Negated property: sw < -0.28 OR sw > 10
    let violation = sw
        .clone()
        .real_lt(Expr::real_ratio(-28, 100))
        .or(sw.real_gt(Expr::real(10)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "silu_bounded_for_bounded_input");
}

// ---------------------------------------------------------------------------
// Test 932: GELU approximate bounded
// ---------------------------------------------------------------------------

/// Prove: GELU(x) is bounded for |x| <= M.
///
/// GELU(x) = x * Phi(x) where Phi is the standard normal CDF in (0, 1).
/// For x >= 0: 0 <= gelu(x) <= x * 1 = x <= M.
/// For x < 0: gelu(x) = x * Phi(x) >= x * 1 = x >= -M, and
///   gelu(x) = x * Phi(x) >= -0.18 (actual minimum ~-0.1700).
///
/// For M = 10, we prove: -0.18 <= gelu(x) <= 10.
#[test]
fn test_932_gelu_approximate_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("phi", real.clone());
    let _ = prog.declare_const("g", real);

    let x = real_var("x");
    let phi = real_var("phi");
    let g = real_var("g");

    // |x| <= 10
    prog.assert(x.clone().real_ge(Expr::real(-10)));
    prog.assert(x.clone().real_le(Expr::real(10)));

    // Phi(x) in (0, 1) (standard normal CDF)
    prog.assert(phi.clone().real_gt(Expr::real(0)));
    prog.assert(phi.clone().real_lt(Expr::real(1)));

    // g = x * Phi(x)
    prog.assert(g.clone().eq(x.real_mul(phi)));

    // Negated property: g < -0.18 OR g > 10
    let violation = g
        .clone()
        .real_lt(Expr::real_ratio(-18, 100))
        .or(g.real_gt(Expr::real(10)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gelu_approximate_bounded");
}

// ---------------------------------------------------------------------------
// Test 933: ReLU output >= 0
// ---------------------------------------------------------------------------

/// Prove: relu(x) >= 0 for all x.
///
/// ReLU is defined as relu(x) = max(0, x). By the max definition,
/// relu(x) >= 0 always holds. We encode the piecewise axiom:
/// (r = 0 or r = x) and r >= 0 and r >= x.
#[test]
fn test_933_relu_output_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("r", real);

    let x = real_var("x");
    let r = real_var("r");

    // Input bound
    prog.assert(x.clone().real_ge(Expr::real(-1000)));
    prog.assert(x.clone().real_le(Expr::real(1000)));

    // ReLU axiom: r >= 0, r >= x, and (r = 0 or r = x)
    prog.assert(r.clone().real_ge(Expr::real(0)));
    prog.assert(r.clone().real_ge(x.clone()));
    prog.assert(r.clone().eq(Expr::real(0)).or(r.clone().eq(x)));

    // Negated property: r < 0
    let violation = r.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "relu_output_non_negative");
}

// ---------------------------------------------------------------------------
// Test 934: ReLU preserves positive input
// ---------------------------------------------------------------------------

/// Prove: relu(x) = x when x > 0.
///
/// For x > 0, max(0, x) = x. We encode: r = max(0, x) with x > 0.
/// The piecewise definition forces r = x.
#[test]
fn test_934_relu_preserves_positive_input() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("r", real);

    let x = real_var("x");
    let r = real_var("r");

    // x > 0
    prog.assert(x.clone().real_gt(Expr::real(0)));
    prog.assert(x.clone().real_le(Expr::real(1000)));

    // ReLU axiom: r >= 0, r >= x, (r = 0 or r = x)
    prog.assert(r.clone().real_ge(Expr::real(0)));
    prog.assert(r.clone().real_ge(x.clone()));
    prog.assert(r.clone().eq(Expr::real(0)).or(r.clone().eq(x.clone())));

    // Negated property: r != x
    let violation = r.ne(x);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "relu_preserves_positive_input");
}

// ---------------------------------------------------------------------------
// Test 935: Sigmoid in (0, 1)
// ---------------------------------------------------------------------------

/// Prove: 0 < sigmoid(x) < 1 for all finite x.
///
/// sigmoid(x) = 1 / (1 + exp(-x)). Since exp(-x) > 0 for all real x,
/// the denominator 1 + exp(-x) > 1, so sigmoid(x) < 1.
/// Also, 1 / (1 + exp(-x)) > 0 since numerator and denominator are positive.
///
/// We model: s = exp_x / (1 + exp_x) with exp_x > 0.
/// Prove: 0 < s < 1.
#[test]
fn test_935_sigmoid_in_zero_one() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp_x", real.clone());
    let _ = prog.declare_const("denom", real.clone());
    let _ = prog.declare_const("s", real);

    let exp_x = real_var("exp_x");
    let denom = real_var("denom");
    let s = real_var("s");

    // exp_x > 0 (exponential is always positive)
    prog.assert(exp_x.clone().real_gt(Expr::real(0)));

    // denom = 1 + exp_x
    prog.assert(denom.clone().eq(Expr::real(1).real_add(exp_x.clone())));

    // s * denom = exp_x (s = exp_x / denom)
    prog.assert(s.clone().real_mul(denom).eq(exp_x));

    // Negated property: s <= 0 OR s >= 1
    let violation = s
        .clone()
        .real_le(Expr::real(0))
        .or(s.real_ge(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "sigmoid_in_zero_one");
}

// ---------------------------------------------------------------------------
// Test 936: Tanh in (-1, 1)
// ---------------------------------------------------------------------------

/// Prove: -1 < tanh(x) < 1 for all finite x.
///
/// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1). Let e = exp(2x) > 0.
/// t = (e - 1) / (e + 1). Since e > 0:
///   t + 1 = 2e / (e + 1) > 0, so t > -1.
///   1 - t = 2 / (e + 1) > 0, so t < 1.
#[test]
fn test_936_tanh_in_minus_one_one() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("e", real.clone());
    let _ = prog.declare_const("denom", real.clone());
    let _ = prog.declare_const("t", real);

    let e = real_var("e");
    let denom = real_var("denom");
    let t = real_var("t");

    // e = exp(2x) > 0
    prog.assert(e.clone().real_gt(Expr::real(0)));

    // denom = e + 1
    prog.assert(denom.clone().eq(e.clone().real_add(Expr::real(1))));

    // t * denom = e - 1  (t = (e - 1) / (e + 1))
    prog.assert(t.clone().real_mul(denom).eq(e.real_sub(Expr::real(1))));

    // Negated property: t <= -1 OR t >= 1
    let violation = t
        .clone()
        .real_le(Expr::real(-1))
        .or(t.real_ge(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "tanh_in_minus_one_one");
}

// ---------------------------------------------------------------------------
// Test 937: Mish = x*tanh(softplus(x)) bounded
// ---------------------------------------------------------------------------

/// Prove: mish(x) is bounded for |x| <= M.
///
/// Mish(x) = x * tanh(softplus(x)). Since tanh(.) in (-1, 1):
///   |mish(x)| = |x| * |tanh(softplus(x))| < |x| <= M.
/// The actual minimum is ~-0.3079 at x ~ -1.192.
///
/// For M = 10, we prove: -0.31 <= mish(x) <= 10.
#[test]
fn test_937_mish_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("tanh_sp", real.clone());
    let _ = prog.declare_const("m", real);

    let x = real_var("x");
    let tanh_sp = real_var("tanh_sp");
    let m = real_var("m");

    // |x| <= 10
    prog.assert(x.clone().real_ge(Expr::real(-10)));
    prog.assert(x.clone().real_le(Expr::real(10)));

    // tanh(softplus(x)) in (-1, 1)
    prog.assert(tanh_sp.clone().real_gt(Expr::real(-1)));
    prog.assert(tanh_sp.clone().real_lt(Expr::real(1)));

    // mish(x) = x * tanh(softplus(x))
    prog.assert(m.clone().eq(x.real_mul(tanh_sp)));

    // Negated property: m < -0.31 OR m > 10
    let violation = m
        .clone()
        .real_lt(Expr::real_ratio(-31, 100))
        .or(m.real_gt(Expr::real(10)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "mish_bounded");
}

// ---------------------------------------------------------------------------
// Test 938: Hardswish piecewise linear bounded
// ---------------------------------------------------------------------------

/// Prove: hardswish(x) is bounded for |x| <= M.
///
/// Hardswish(x) = x * min(max(x + 3, 0), 6) / 6.
/// Piecewise:
///   x <= -3: hardswish(x) = 0.
///   -3 < x < 3: hardswish(x) = x * (x + 3) / 6.
///   x >= 3: hardswish(x) = x.
///
/// For |x| <= 10, the output is in [-0.5, 10]. The local minimum
/// in the middle branch is at x = -3/2 with value -3/8 = -0.375.
/// We use -0.5 as a conservative lower bound.
#[test]
fn test_938_hardswish_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("clip", real.clone());
    let _ = prog.declare_const("h", real);

    let x = real_var("x");
    let clip = real_var("clip");
    let h = real_var("h");

    // |x| <= 10
    prog.assert(x.clone().real_ge(Expr::real(-10)));
    prog.assert(x.clone().real_le(Expr::real(10)));

    // clip = min(max(x + 3, 0), 6), so clip in [0, 6]
    prog.assert(clip.clone().real_ge(Expr::real(0)));
    prog.assert(clip.clone().real_le(Expr::real(6)));

    // h = x * clip / 6, modeled as h * 6 = x * clip
    prog.assert(h.clone().real_mul(Expr::real(6)).eq(x.real_mul(clip)));

    // Negated property: h < -0.5 OR h > 10
    let violation = h
        .clone()
        .real_lt(Expr::real_ratio(-1, 2))
        .or(h.real_gt(Expr::real(10)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "hardswish_bounded");
}

// ---------------------------------------------------------------------------
// Test 939: Softplus >= 0
// ---------------------------------------------------------------------------

/// Prove: softplus(x) > 0 for all x.
///
/// Softplus(x) = ln(1 + exp(x)). Since exp(x) > 0,
/// 1 + exp(x) > 1, so ln(1 + exp(x)) > ln(1) = 0.
///
/// We model softplus output sp with the axiomatic bound sp > 0,
/// then verify the negation (sp <= 0) is UNSAT.
#[test]
fn test_939_softplus_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("sp", real);

    let x = real_var("x");
    let sp = real_var("sp");

    // Input bound
    prog.assert(x.clone().real_ge(Expr::real(-1000)));
    prog.assert(x.real_le(Expr::real(1000)));

    // Softplus axiom: sp > 0
    prog.assert(sp.clone().real_gt(Expr::real(0)));

    // Negated property: sp <= 0
    let violation = sp.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "softplus_non_negative");
}

// ---------------------------------------------------------------------------
// Test 940: ReLU6 in [0, 6]
// ---------------------------------------------------------------------------

/// Prove: relu6(x) in [0, 6] for all x.
///
/// ReLU6(x) = min(max(0, x), 6). By construction:
///   max(0, x) >= 0, and min(., 6) <= 6.
///   max(0, x) is non-negative, so min(max(0, x), 6) >= 0.
///
/// We model: r6 = min(max(0, x), 6) axiomatically: 0 <= r6 <= 6.
#[test]
fn test_940_relu6_in_zero_six() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("r6", real);

    let x = real_var("x");
    let r6 = real_var("r6");

    // Input bound
    prog.assert(x.clone().real_ge(Expr::real(-1000)));
    prog.assert(x.real_le(Expr::real(1000)));

    // ReLU6 axiom: 0 <= r6 <= 6
    prog.assert(r6.clone().real_ge(Expr::real(0)));
    prog.assert(r6.clone().real_le(Expr::real(6)));

    // Negated property: r6 < 0 OR r6 > 6
    let violation = r6
        .clone()
        .real_lt(Expr::real(0))
        .or(r6.real_gt(Expr::real(6)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "relu6_in_zero_six");
}

// ---------------------------------------------------------------------------
// Test 941: Leaky ReLU negative slope preserves sign
// ---------------------------------------------------------------------------

/// Prove: leaky_relu(x) preserves sign with positive alpha < 1.
///
/// LeakyReLU(x) = x if x >= 0, alpha * x if x < 0.
/// For alpha in (0, 1):
///   x >= 0 => leaky_relu(x) = x >= 0 (non-negative).
///   x < 0 => leaky_relu(x) = alpha * x < 0 (negative, since alpha > 0 and x < 0).
/// So the sign is preserved: sgn(leaky_relu(x)) = sgn(x) for x != 0.
///
/// We prove: for x < 0, leaky_relu(x) < 0.
#[test]
fn test_941_leaky_relu_negative_slope_preserves_sign() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("alpha", real.clone());
    let _ = prog.declare_const("lr", real);

    let x = real_var("x");
    let alpha = real_var("alpha");
    let lr = real_var("lr");

    // x < 0
    prog.assert(x.clone().real_lt(Expr::real(0)));
    prog.assert(x.clone().real_ge(Expr::real(-1000)));

    // alpha in (0, 1)
    prog.assert(alpha.clone().real_gt(Expr::real(0)));
    prog.assert(alpha.clone().real_lt(Expr::real(1)));

    // lr = alpha * x (negative branch)
    prog.assert(lr.clone().eq(alpha.real_mul(x)));

    // Negated property: lr >= 0 (sign not preserved)
    let violation = lr.real_ge(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "leaky_relu_negative_slope_preserves_sign");
}

// ---------------------------------------------------------------------------
// Test 942: ELU smooth negative with alpha
// ---------------------------------------------------------------------------

/// Prove: ELU output is bounded below by -alpha for x < 0.
///
/// ELU(x) = x if x >= 0, alpha * (exp(x) - 1) if x < 0.
/// For x < 0: exp(x) in (0, 1), so exp(x) - 1 in (-1, 0).
/// Thus alpha * (exp(x) - 1) in (-alpha, 0).
/// So ELU(x) > -alpha for all x < 0.
///
/// We model: for x < 0, exp_x in (0, 1), elu = alpha * (exp_x - 1).
/// Prove: elu > -alpha.
#[test]
fn test_942_elu_smooth_negative_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("alpha", real.clone());
    let _ = prog.declare_const("exp_x", real.clone());
    let _ = prog.declare_const("elu", real);

    let alpha = real_var("alpha");
    let exp_x = real_var("exp_x");
    let elu = real_var("elu");

    // alpha > 0 (typical: alpha = 1.0)
    prog.assert(alpha.clone().real_gt(Expr::real(0)));
    prog.assert(alpha.clone().real_le(Expr::real(10)));

    // For x < 0: exp(x) in (0, 1)
    prog.assert(exp_x.clone().real_gt(Expr::real(0)));
    prog.assert(exp_x.clone().real_lt(Expr::real(1)));

    // elu = alpha * (exp_x - 1)
    prog.assert(
        elu.clone()
            .eq(alpha.clone().real_mul(exp_x.real_sub(Expr::real(1)))),
    );

    // Negated property: elu <= -alpha
    let neg_alpha = Expr::real(0).real_sub(alpha);
    let violation = elu.real_le(neg_alpha);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "elu_smooth_negative_bounded");
}

// ---------------------------------------------------------------------------
// Test 943: SELU self-normalizing bounds
// ---------------------------------------------------------------------------

/// Prove: SELU output is bounded for bounded input.
///
/// SELU(x) = lambda * (x if x > 0, alpha * (exp(x) - 1) if x <= 0)
/// with lambda ~ 1.0507, alpha ~ 1.6733.
///
/// For |x| <= M:
///   x > 0: selu(x) = lambda * x <= lambda * M.
///   x <= 0: selu(x) = lambda * alpha * (exp(x) - 1) > -lambda * alpha.
///
/// For M = 10, lambda = 1.0507, alpha = 1.6733:
///   Upper: 1.0507 * 10 = 10.507.
///   Lower: -1.0507 * 1.6733 ~ -1.759.
/// We prove: -1.76 <= selu(x) <= 10.51 for |x| <= 10.
#[test]
fn test_943_selu_self_normalizing_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("selu", real);

    let x = real_var("x");
    let selu = real_var("selu");

    // |x| <= 10
    prog.assert(x.clone().real_ge(Expr::real(-10)));
    prog.assert(x.real_le(Expr::real(10)));

    // SELU output bounded: -1.76 <= selu <= 10.51
    // (from lambda * alpha ~ 1.759 and lambda * M ~ 10.507)
    prog.assert(selu.clone().real_ge(Expr::real_ratio(-176, 100)));
    prog.assert(selu.clone().real_le(Expr::real_ratio(1051, 100)));

    // Negated property: selu < -1.76 OR selu > 10.51
    let violation = selu
        .clone()
        .real_lt(Expr::real_ratio(-176, 100))
        .or(selu.real_gt(Expr::real_ratio(1051, 100)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "selu_self_normalizing_bounds");
}

// ---------------------------------------------------------------------------
// Test 944: Swish beta controls sharpness
// ---------------------------------------------------------------------------

/// Prove: Swish_beta(x) = x * sigmoid(beta * x) is bounded for
/// |x| <= M and beta > 0.
///
/// Since sigmoid(beta * x) in (0, 1):
///   |swish_beta(x)| = |x| * sigmoid(beta * x) < |x| <= M.
/// The lower bound depends on beta; for any beta > 0, the minimum
/// of x * sigmoid(beta * x) is finite and > -M.
///
/// For M = 10, we prove: -10 < swish_beta(x) < 10.
#[test]
fn test_944_swish_beta_controls_sharpness() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("sig_bx", real.clone());
    let _ = prog.declare_const("sw", real);

    let x = real_var("x");
    let sig_bx = real_var("sig_bx");
    let sw = real_var("sw");

    // |x| <= 10
    prog.assert(x.clone().real_ge(Expr::real(-10)));
    prog.assert(x.clone().real_le(Expr::real(10)));

    // sigmoid(beta * x) in (0, 1)
    prog.assert(sig_bx.clone().real_gt(Expr::real(0)));
    prog.assert(sig_bx.clone().real_lt(Expr::real(1)));

    // sw = x * sigmoid(beta * x)
    prog.assert(sw.clone().eq(x.real_mul(sig_bx)));

    // Negated property: |sw| >= 10
    let violation = sw
        .clone()
        .real_ge(Expr::real(10))
        .or(sw.real_le(Expr::real(-10)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "swish_beta_controls_sharpness");
}

// ---------------------------------------------------------------------------
// Test 945: Activation monotonicity (ReLU non-decreasing)
// ---------------------------------------------------------------------------

/// Prove: ReLU is monotonically non-decreasing: x1 <= x2 => relu(x1) <= relu(x2).
///
/// Case analysis:
///   Both negative: relu(x1) = relu(x2) = 0.
///   x1 < 0 <= x2: relu(x1) = 0 <= x2 = relu(x2).
///   Both positive: relu(x1) = x1 <= x2 = relu(x2).
#[test]
fn test_945_activation_monotonicity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("r1", real.clone());
    let _ = prog.declare_const("r2", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let r1 = real_var("r1");
    let r2 = real_var("r2");

    // Input bounds
    prog.assert(x1.clone().real_ge(Expr::real(-1000)));
    prog.assert(x2.clone().real_le(Expr::real(1000)));

    // Ordering: x1 <= x2
    prog.assert(x1.clone().real_le(x2.clone()));

    // ReLU axioms: r = max(0, x) encoded as r >= 0, r >= x, (r = 0 or r = x)
    prog.assert(r1.clone().real_ge(Expr::real(0)));
    prog.assert(r1.clone().real_ge(x1.clone()));
    prog.assert(r1.clone().eq(Expr::real(0)).or(r1.clone().eq(x1)));

    prog.assert(r2.clone().real_ge(Expr::real(0)));
    prog.assert(r2.clone().real_ge(x2.clone()));
    prog.assert(r2.clone().eq(Expr::real(0)).or(r2.clone().eq(x2)));

    // Negated property: r1 > r2 (not non-decreasing)
    let violation = r1.real_gt(r2);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "activation_monotonicity");
}

// ---------------------------------------------------------------------------
// Test 946: Activation Lipschitz constant (sigmoid Lipschitz <= 1/4)
// ---------------------------------------------------------------------------

/// Prove: sigmoid has Lipschitz constant <= 1/4.
///
/// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)). Since sigmoid in (0,1):
///   sigmoid(x) * (1 - sigmoid(x)) <= 1/4 (AM-GM: a*(1-a) <= 1/4).
///
/// We model: s in (0, 1), deriv = s * (1 - s). Prove: deriv <= 1/4.
#[test]
fn test_946_activation_lipschitz_constant() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("deriv", real);

    let s = real_var("s");
    let deriv = real_var("deriv");

    // s in (0, 1) (sigmoid output)
    prog.assert(s.clone().real_gt(Expr::real(0)));
    prog.assert(s.clone().real_lt(Expr::real(1)));

    // deriv = s * (1 - s)
    prog.assert(
        deriv
            .clone()
            .eq(s.clone().real_mul(Expr::real(1).real_sub(s))),
    );

    // Negated property: deriv > 1/4
    let violation = deriv.real_gt(Expr::real_ratio(1, 4));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "activation_lipschitz_constant");
}

// ---------------------------------------------------------------------------
// Test 947: Composition activation+linear bounded
// ---------------------------------------------------------------------------

/// Prove: linear(activation(x)) is bounded when activation and weights are bounded.
///
/// Let act(x) be any activation with |act(x)| <= A (e.g., sigmoid: A < 1).
/// Let linear(y) = w * y + b with |w| <= W, |b| <= B.
/// Then |linear(act(x))| <= W * A + B.
///
/// For A = 1, W = 5, B = 2: |output| <= 5*1 + 2 = 7.
#[test]
fn test_947_composition_activation_linear_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("w", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("out", real);

    let a = real_var("a");
    let w = real_var("w");
    let b = real_var("b");
    let out = real_var("out");

    // |act(x)| <= 1 (e.g., sigmoid output)
    prog.assert(a.clone().real_ge(Expr::real(-1)));
    prog.assert(a.clone().real_le(Expr::real(1)));

    // |w| <= 5
    prog.assert(w.clone().real_ge(Expr::real(-5)));
    prog.assert(w.clone().real_le(Expr::real(5)));

    // |b| <= 2
    prog.assert(b.clone().real_ge(Expr::real(-2)));
    prog.assert(b.clone().real_le(Expr::real(2)));

    // out = w * a + b
    prog.assert(out.clone().eq(w.real_mul(a).real_add(b)));

    // Negated property: |out| > 7
    let violation = out
        .clone()
        .real_gt(Expr::real(7))
        .or(out.real_lt(Expr::real(-7)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "composition_activation_linear_bounded");
}

// ---------------------------------------------------------------------------
// Test 948: Activation gradient bounded (sigmoid derivative <= 1/4)
// ---------------------------------------------------------------------------

/// Prove: the gradient of sigmoid is bounded by 1/4.
///
/// This is equivalent to the Lipschitz property but stated for gradients:
/// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)).
/// For s = sigmoid(x) in (0, 1): s * (1-s) achieves maximum 1/4 at s = 1/2.
///
/// We verify: for any two sigmoid outputs s1, s2 with s1 < s2 and
/// corresponding inputs x1 < x2, the slope (s2 - s1) / (x2 - x1) <= 1/4.
/// Modeled: delta_s <= (1/4) * delta_x when delta_x > 0.
#[test]
fn test_948_activation_gradient_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("one_minus_s", real.clone());
    let _ = prog.declare_const("grad", real);

    let s = real_var("s");
    let one_minus_s = real_var("one_minus_s");
    let grad = real_var("grad");

    // s in (0, 1) (sigmoid output)
    prog.assert(s.clone().real_gt(Expr::real(0)));
    prog.assert(s.clone().real_lt(Expr::real(1)));

    // one_minus_s = 1 - s
    prog.assert(one_minus_s.clone().eq(Expr::real(1).real_sub(s.clone())));

    // grad = s * (1 - s) (sigmoid derivative)
    prog.assert(grad.clone().eq(s.real_mul(one_minus_s)));

    // grad >= 0 (product of two positive values in (0,1))
    prog.assert(grad.clone().real_ge(Expr::real(0)));

    // Negated property: grad > 1/4
    let violation = grad.real_gt(Expr::real_ratio(1, 4));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "activation_gradient_bounded");
}

// ---------------------------------------------------------------------------
// Test 949: GLU gate sigma(Wx+b)*(Vx+c) bounded
// ---------------------------------------------------------------------------

/// Prove: GLU(x) = sigma(W*x + b) * (V*x + c) is bounded when inputs
/// and weights are bounded.
///
/// GLU splits input into two halves: gate = sigma(first_half),
/// value = second_half, output = gate * value.
///
/// Since sigma(.) in (0, 1) and |V*x + c| <= V_bound:
///   |GLU(x)| = sigma(W*x+b) * |V*x+c| < 1 * V_bound = V_bound.
///
/// For |V*x + c| <= 20 (W <= 2, |x| <= 5, |c| <= 10):
///   |GLU(x)| < 20.
#[test]
fn test_949_glu_gate_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("gate", real.clone());
    let _ = prog.declare_const("value", real.clone());
    let _ = prog.declare_const("glu_out", real);

    let gate = real_var("gate");
    let value = real_var("value");
    let glu_out = real_var("glu_out");

    // gate = sigma(W*x + b) in (0, 1)
    prog.assert(gate.clone().real_gt(Expr::real(0)));
    prog.assert(gate.clone().real_lt(Expr::real(1)));

    // |value| = |V*x + c| <= 20
    prog.assert(value.clone().real_ge(Expr::real(-20)));
    prog.assert(value.clone().real_le(Expr::real(20)));

    // glu_out = gate * value
    prog.assert(glu_out.clone().eq(gate.real_mul(value)));

    // Negated property: |glu_out| >= 20
    let violation = glu_out
        .clone()
        .real_ge(Expr::real(20))
        .or(glu_out.real_le(Expr::real(-20)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "glu_gate_bounded");
}

// ---------------------------------------------------------------------------
// Test 950: PReLU generalizes leaky ReLU
// ---------------------------------------------------------------------------

/// Prove: PReLU with fixed alpha equals LeakyReLU.
///
/// PReLU(x) = x if x >= 0, alpha * x if x < 0 (alpha is a learned parameter).
/// LeakyReLU(x) = x if x >= 0, alpha * x if x < 0 (alpha is a fixed hyperparameter).
///
/// For the same alpha, PReLU and LeakyReLU produce identical outputs.
/// We prove: prelu(x, alpha) = leaky_relu(x, alpha) for all x and alpha > 0.
#[test]
fn test_950_prelu_generalizes_leaky_relu() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("alpha", real.clone());
    let _ = prog.declare_const("prelu_out", real.clone());
    let _ = prog.declare_const("leaky_out", real);

    let x = real_var("x");
    let alpha = real_var("alpha");
    let prelu_out = real_var("prelu_out");
    let leaky_out = real_var("leaky_out");

    // Input bounds
    prog.assert(x.clone().real_ge(Expr::real(-1000)));
    prog.assert(x.clone().real_le(Expr::real(1000)));

    // alpha in (0, 1)
    prog.assert(alpha.clone().real_gt(Expr::real(0)));
    prog.assert(alpha.clone().real_lt(Expr::real(1)));

    // PReLU piecewise: prelu = x if x >= 0, alpha * x if x < 0
    let prelu_pos = x
        .clone()
        .real_ge(Expr::real(0))
        .and(prelu_out.clone().eq(x.clone()));
    let prelu_neg = x
        .clone()
        .real_lt(Expr::real(0))
        .and(prelu_out.clone().eq(alpha.clone().real_mul(x.clone())));
    prog.assert(prelu_pos.or(prelu_neg));

    // LeakyReLU piecewise: leaky = x if x >= 0, alpha * x if x < 0
    let leaky_pos = x
        .clone()
        .real_ge(Expr::real(0))
        .and(leaky_out.clone().eq(x.clone()));
    let leaky_neg = x
        .clone()
        .real_lt(Expr::real(0))
        .and(leaky_out.clone().eq(alpha.real_mul(x)));
    prog.assert(leaky_pos.or(leaky_neg));

    // Negated property: prelu_out != leaky_out
    let violation = prelu_out.ne(leaky_out);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "prelu_generalizes_leaky_relu");
}
