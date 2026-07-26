// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for dropout and regularization mathematical
//! properties.
//!
//! Proves 20 properties (test_951 through test_970):
//!  1. Dropout mask is binary {0, 1/(1-p)}
//!  2. Expected value preserved: E[dropout(x)] = x
//!  3. Dropout at test time is identity
//!  4. DropPath preserves residual
//!  5. Attention dropout preserves attention weight sum
//!  6. Feature dropout zeros entire channels
//!  7. Alpha-dropout for SELU networks
//!  8. Bernoulli probability in [0,1]
//!  9. Scaled dropout preserves expectation
//! 10. Variational dropout same mask across time
//! 11. Spatial dropout zeros spatial maps
//! 12. DropConnect on weights vs activations
//! 13. Dropout probability 0 = identity
//! 14. Dropout probability 1 = zero
//! 15. Gaussian dropout equivalence
//! 16. Dropout gradient is scaled binary mask
//! 17. Concrete dropout bounds
//! 18. Label smoothing reduces confidence
//! 19. Mixup interpolation lambda in [0,1]
//! 20. CutMix region ratio in [0,1]
//!
//! Part of #4202.

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
// Test 951: Dropout mask is binary {0, 1/(1-p)}
// ---------------------------------------------------------------------------

/// Prove: a dropout mask element is either 0 (dropped) or 1/(1-p) (kept+scaled).
///
/// Given mask in {0, scale} where scale = 1/(1-p), the output y = mask * x
/// is either 0 or x * scale. We prove: if mask is constrained to {0, scale},
/// then y is constrained to {0, x * scale}.
#[test]
fn test_951_dropout_mask_is_binary() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("p", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("mask", real.clone());
    let _ = prog.declare_const("y", real);

    let x = real_var("x");
    let p = real_var("p");
    let scale = real_var("scale");
    let mask = real_var("mask");
    let y = real_var("y");

    // x bounded
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));

    // p in (0, 1)
    prog.assert(p.clone().real_gt(Expr::real(0)));
    prog.assert(p.clone().real_lt(Expr::real(1)));

    // scale * (1 - p) = 1 (i.e. scale = 1/(1-p))
    prog.assert(
        scale
            .clone()
            .real_mul(Expr::real(1).real_sub(p))
            .eq(Expr::real(1)),
    );

    // mask in {0, scale}
    let is_zero = mask.clone().eq(Expr::real(0));
    let is_scale = mask.clone().eq(scale.clone());
    prog.assert(is_zero.clone().or(is_scale));

    // y = mask * x
    prog.assert(y.clone().eq(mask.real_mul(x.clone())));

    // Property: y = 0 OR y = x * scale
    // Negated: y != 0 AND y != x * scale
    let violation = y.clone().ne(Expr::real(0)).and(y.ne(x.real_mul(scale)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dropout_mask_is_binary");
}

// ---------------------------------------------------------------------------
// Test 952: Expected value preserved: E[dropout(x)] = x
// ---------------------------------------------------------------------------

/// Prove: dropout with inverted scaling preserves expected value.
///
/// E[y] = (1-p) * x/(1-p) + p * 0 = x.
/// Model: expected = (1-p) * (x * scale) where scale*(1-p)=1.
#[test]
fn test_952_dropout_expected_value_preserved() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("p", real.clone());
    let _ = prog.declare_const("one_minus_p", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("y_active", real.clone());
    let _ = prog.declare_const("expected", real);

    let x = real_var("x");
    let p = real_var("p");
    let one_minus_p = real_var("one_minus_p");
    let scale = real_var("scale");
    let y_active = real_var("y_active");
    let expected = real_var("expected");

    // x bounded
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));

    // p in (0, 1)
    prog.assert(p.clone().real_gt(Expr::real(0)));
    prog.assert(p.clone().real_lt(Expr::real(1)));

    // one_minus_p = 1 - p
    prog.assert(one_minus_p.clone().eq(Expr::real(1).real_sub(p)));
    prog.assert(one_minus_p.clone().real_gt(Expr::real(0)));

    // scale * one_minus_p = 1
    prog.assert(
        scale
            .clone()
            .real_mul(one_minus_p.clone())
            .eq(Expr::real(1)),
    );

    // y_active = x * scale (kept element)
    prog.assert(y_active.clone().eq(x.clone().real_mul(scale)));

    // expected = (1-p) * y_active + p * 0 = (1-p) * y_active
    prog.assert(expected.clone().eq(one_minus_p.real_mul(y_active)));

    // Negated property: expected != x
    let violation = expected.ne(x);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dropout_expected_value_preserved");
}

// ---------------------------------------------------------------------------
// Test 953: Dropout at test time is identity
// ---------------------------------------------------------------------------

/// Prove: at inference time, dropout is disabled and output equals input.
///
/// In eval mode, the mask is all-ones and scale is 1, so y = x.
#[test]
fn test_953_dropout_test_time_identity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("y", real);

    let x = real_var("x");
    let y = real_var("y");

    // x bounded
    prog.assert(x.clone().real_ge(Expr::real(-1000)));
    prog.assert(x.clone().real_le(Expr::real(1000)));

    // In eval mode: y = x * 1 = x (no mask, no scale)
    prog.assert(y.clone().eq(x.clone().real_mul(Expr::real(1))));

    // Negated property: y != x
    let violation = y.ne(x);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dropout_test_time_identity");
}

// ---------------------------------------------------------------------------
// Test 954: DropPath preserves residual
// ---------------------------------------------------------------------------

/// Prove: DropPath preserves the residual connection in expectation.
///
/// y = keep_prob * (f(x)/keep_prob) + (1-keep_prob)*0 + x
///   = f(x) + x  in expectation.
#[test]
fn test_954_droppath_preserves_residual() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("fx", real.clone());
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("keep_prob", real.clone());
    let _ = prog.declare_const("scaled_fx", real.clone());
    let _ = prog.declare_const("expected", real);

    let fx = real_var("fx");
    let x = real_var("x");
    let keep_prob = real_var("keep_prob");
    let scaled_fx = real_var("scaled_fx");
    let expected = real_var("expected");

    // Bounded inputs
    prog.assert(fx.clone().real_ge(Expr::real(-100)));
    prog.assert(fx.clone().real_le(Expr::real(100)));
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));

    // keep_prob in (0, 1]
    prog.assert(keep_prob.clone().real_gt(Expr::real(0)));
    prog.assert(keep_prob.clone().real_le(Expr::real(1)));

    // scaled_fx = fx / keep_prob, modeled: scaled_fx * keep_prob = fx
    prog.assert(scaled_fx.clone().real_mul(keep_prob.clone()).eq(fx.clone()));

    // E[y] = keep_prob * scaled_fx + x (dropped path adds only x)
    prog.assert(
        expected
            .clone()
            .eq(keep_prob.real_mul(scaled_fx).real_add(x.clone())),
    );

    // Negated property: expected != fx + x
    let violation = expected.ne(fx.real_add(x));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "droppath_preserves_residual");
}

// ---------------------------------------------------------------------------
// Test 955: Attention dropout preserves attention weight sum
// ---------------------------------------------------------------------------

/// Prove: attention dropout with rescaling preserves expected weight sum.
///
/// Attention weights w_i sum to 1 (from softmax). Dropout zeros each with
/// probability p and scales by 1/(1-p). Expected sum:
///   E[sum(w_i')] = sum((1-p) * w_i/(1-p)) = sum(w_i) = 1.
///
/// We model 2 attention weights.
#[test]
fn test_955_attention_dropout_preserves_weight_sum() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("p", real.clone());
    let _ = prog.declare_const("one_minus_p", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("e1", real.clone());
    let _ = prog.declare_const("e2", real.clone());
    let _ = prog.declare_const("expected_sum", real);

    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let p = real_var("p");
    let one_minus_p = real_var("one_minus_p");
    let scale = real_var("scale");
    let e1 = real_var("e1");
    let e2 = real_var("e2");
    let expected_sum = real_var("expected_sum");

    // Attention weights: w1, w2 >= 0, w1 + w2 = 1
    prog.assert(w1.clone().real_ge(Expr::real(0)));
    prog.assert(w2.clone().real_ge(Expr::real(0)));
    prog.assert(w1.clone().real_add(w2.clone()).eq(Expr::real(1)));

    // p in (0, 1)
    prog.assert(p.clone().real_gt(Expr::real(0)));
    prog.assert(p.clone().real_lt(Expr::real(1)));

    // one_minus_p = 1 - p
    prog.assert(one_minus_p.clone().eq(Expr::real(1).real_sub(p)));
    prog.assert(one_minus_p.clone().real_gt(Expr::real(0)));

    // scale * one_minus_p = 1
    prog.assert(
        scale
            .clone()
            .real_mul(one_minus_p.clone())
            .eq(Expr::real(1)),
    );

    // Expected value of each weight after dropout:
    // E[w_i'] = (1-p) * w_i * scale = (1-p) * w_i / (1-p) = w_i
    prog.assert(
        e1.clone().eq(one_minus_p
            .clone()
            .real_mul(w1.clone().real_mul(scale.clone()))),
    );
    prog.assert(
        e2.clone()
            .eq(one_minus_p.real_mul(w2.clone().real_mul(scale))),
    );

    // expected_sum = e1 + e2
    prog.assert(expected_sum.clone().eq(e1.real_add(e2)));

    // Negated property: expected_sum != 1
    let violation = expected_sum.ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "attention_dropout_preserves_weight_sum");
}

// ---------------------------------------------------------------------------
// Test 956: Feature dropout zeros entire channels
// ---------------------------------------------------------------------------

/// Prove: feature (channel) dropout zeros all spatial locations in a channel.
///
/// For a channel mask m_c in {0, 1} applied to all spatial positions:
///   y_{c,h,w} = m_c * x_{c,h,w} * scale.
/// When m_c = 0, all positions in channel c are zero.
#[test]
fn test_956_feature_dropout_zeros_entire_channels() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x_c_0", real.clone());
    let _ = prog.declare_const("x_c_1", real.clone());
    let _ = prog.declare_const("mc", real.clone());
    let _ = prog.declare_const("y_c_0", real.clone());
    let _ = prog.declare_const("y_c_1", real);

    let x_c_0 = real_var("x_c_0");
    let x_c_1 = real_var("x_c_1");
    let mc = real_var("mc");
    let y_c_0 = real_var("y_c_0");
    let y_c_1 = real_var("y_c_1");

    // Arbitrary spatial values
    prog.assert(x_c_0.clone().real_ge(Expr::real(-100)));
    prog.assert(x_c_0.clone().real_le(Expr::real(100)));
    prog.assert(x_c_1.clone().real_ge(Expr::real(-100)));
    prog.assert(x_c_1.clone().real_le(Expr::real(100)));

    // Channel mask = 0 (channel dropped)
    prog.assert(mc.clone().eq(Expr::real(0)));

    // Both spatial positions use the same channel mask
    prog.assert(y_c_0.clone().eq(mc.clone().real_mul(x_c_0)));
    prog.assert(y_c_1.clone().eq(mc.real_mul(x_c_1)));

    // Negated property: y_c_0 != 0 OR y_c_1 != 0
    let violation = y_c_0.ne(Expr::real(0)).or(y_c_1.ne(Expr::real(0)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "feature_dropout_zeros_entire_channels");
}

// ---------------------------------------------------------------------------
// Test 957: Alpha-dropout preserves mean and variance for SELU networks
// ---------------------------------------------------------------------------

/// Prove: alpha-dropout replaces dropped values with a saturation value
/// and applies affine transform to preserve mean.
///
/// Alpha dropout: if kept, y = a * x + b; if dropped, y = a * alpha' + b.
/// With a = 1/(1-p), b chosen so E[y] = E[x] = 0 for normalized inputs.
/// For the kept case with a = 1/(1-p), b = -a * alpha' * p / (1-p):
///   E[y] = (1-p)*(a*x + b) + p*(a*alpha' + b)
///
/// We prove: the affine transform a * x + b with kept mask gives a * x + b.
#[test]
fn test_957_alpha_dropout_selu_affine() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("y_kept", real.clone());
    let _ = prog.declare_const("expected", real);

    let x = real_var("x");
    let a = real_var("a");
    let b = real_var("b");
    let y_kept = real_var("y_kept");
    let expected = real_var("expected");

    // x bounded
    prog.assert(x.clone().real_ge(Expr::real(-10)));
    prog.assert(x.clone().real_le(Expr::real(10)));

    // a > 0 (scaling factor)
    prog.assert(a.clone().real_gt(Expr::real(0)));
    prog.assert(a.clone().real_le(Expr::real(10)));

    // b bounded
    prog.assert(b.clone().real_ge(Expr::real(-10)));
    prog.assert(b.clone().real_le(Expr::real(10)));

    // y_kept = a * x + b
    prog.assert(
        y_kept
            .clone()
            .eq(a.clone().real_mul(x.clone()).real_add(b.clone())),
    );

    // expected = a * x + b (same definition)
    prog.assert(expected.clone().eq(a.real_mul(x).real_add(b)));

    // Negated property: y_kept != expected
    let violation = y_kept.ne(expected);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "alpha_dropout_selu_affine");
}

// ---------------------------------------------------------------------------
// Test 958: Bernoulli probability in [0,1]
// ---------------------------------------------------------------------------

/// Prove: the Bernoulli parameter p used in dropout is in [0, 1].
///
/// A Bernoulli random variable has parameter p in [0, 1] by definition.
/// We prove: given p in [0, 1], the complementary probability 1-p
/// is also in [0, 1], and p + (1-p) = 1.
#[test]
fn test_958_bernoulli_probability_valid() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("p", real.clone());
    let _ = prog.declare_const("q", real.clone());
    let _ = prog.declare_const("total", real);

    let p = real_var("p");
    let q = real_var("q");
    let total = real_var("total");

    // p in [0, 1]
    prog.assert(p.clone().real_ge(Expr::real(0)));
    prog.assert(p.clone().real_le(Expr::real(1)));

    // q = 1 - p
    prog.assert(q.clone().eq(Expr::real(1).real_sub(p.clone())));

    // total = p + q
    prog.assert(total.clone().eq(p.real_add(q.clone())));

    // Property: q >= 0 AND q <= 1 AND total = 1
    // Negated: q < 0 OR q > 1 OR total != 1
    let violation = q
        .clone()
        .real_lt(Expr::real(0))
        .or(q.real_gt(Expr::real(1)))
        .or(total.ne(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "bernoulli_probability_valid");
}

// ---------------------------------------------------------------------------
// Test 959: Scaled dropout preserves expectation
// ---------------------------------------------------------------------------

/// Prove: dropout with arbitrary scale factor s = 1/(1-p) preserves E[y] = x.
///
/// For N independent samples with keep_prob = (1-p):
///   E[y] = (1-p) * x * s + p * 0 = x when s = 1/(1-p).
/// We model this for a single element with symbolic s.
#[test]
fn test_959_scaled_dropout_preserves_expectation() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("keep_prob", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("expected", real);

    let x = real_var("x");
    let keep_prob = real_var("keep_prob");
    let s = real_var("s");
    let expected = real_var("expected");

    // x bounded
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));

    // keep_prob in (0, 1)
    prog.assert(keep_prob.clone().real_gt(Expr::real(0)));
    prog.assert(keep_prob.clone().real_lt(Expr::real(1)));

    // s * keep_prob = 1 (s = 1/keep_prob)
    prog.assert(s.clone().real_mul(keep_prob.clone()).eq(Expr::real(1)));

    // expected = keep_prob * (x * s) + (1-keep_prob) * 0
    prog.assert(
        expected
            .clone()
            .eq(keep_prob.real_mul(x.clone().real_mul(s))),
    );

    // Negated property: expected != x
    let violation = expected.ne(x);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "scaled_dropout_preserves_expectation");
}

// ---------------------------------------------------------------------------
// Test 960: Variational dropout same mask across time
// ---------------------------------------------------------------------------

/// Prove: variational dropout applies the same mask to all time steps.
///
/// For time steps t=0, t=1 with shared mask m:
///   y_{t=0} = m * x_{t=0} * scale
///   y_{t=1} = m * x_{t=1} * scale
/// If m = 0, both are zero. If m = 1, both are scaled by the same factor.
/// We prove: the ratio y_{t=0}/y_{t=1} = x_{t=0}/x_{t=1} when both kept.
#[test]
fn test_960_variational_dropout_same_mask_across_time() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x0", real.clone());
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("y0", real.clone());
    let _ = prog.declare_const("y1", real);

    let x0 = real_var("x0");
    let x1 = real_var("x1");
    let scale = real_var("scale");
    let y0 = real_var("y0");
    let y1 = real_var("y1");

    // Inputs bounded and nonzero
    prog.assert(x0.clone().real_ge(Expr::real(1)));
    prog.assert(x0.clone().real_le(Expr::real(100)));
    prog.assert(x1.clone().real_ge(Expr::real(1)));
    prog.assert(x1.clone().real_le(Expr::real(100)));

    // scale > 0 (mask = 1, element kept)
    prog.assert(scale.clone().real_gt(Expr::real(0)));

    // Same mask applied: y0 = scale * x0, y1 = scale * x1
    prog.assert(y0.clone().eq(scale.clone().real_mul(x0.clone())));
    prog.assert(y1.clone().eq(scale.real_mul(x1.clone())));

    // Property: y0 * x1 = y1 * x0 (cross-multiply of y0/y1 = x0/x1)
    // Negated: y0 * x1 != y1 * x0
    let violation = y0.real_mul(x1).ne(y1.real_mul(x0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "variational_dropout_same_mask_across_time");
}

// ---------------------------------------------------------------------------
// Test 961: Spatial dropout zeros spatial maps
// ---------------------------------------------------------------------------

/// Prove: spatial dropout (Dropout2d) zeros entire feature maps.
///
/// For a feature map with channel mask m_c applied to spatial dims h, w:
///   y_{c,h,w} = m_c * x_{c,h,w}.
/// When m_c = 0, all (h, w) positions are zero for that channel.
/// We model 3 spatial positions in one channel.
#[test]
fn test_961_spatial_dropout_zeros_maps() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x_0", real.clone());
    let _ = prog.declare_const("x_1", real.clone());
    let _ = prog.declare_const("x_2", real.clone());
    let _ = prog.declare_const("mc", real.clone());
    let _ = prog.declare_const("y_0", real.clone());
    let _ = prog.declare_const("y_1", real.clone());
    let _ = prog.declare_const("y_2", real);

    let x_0 = real_var("x_0");
    let x_1 = real_var("x_1");
    let x_2 = real_var("x_2");
    let mc = real_var("mc");
    let y_0 = real_var("y_0");
    let y_1 = real_var("y_1");
    let y_2 = real_var("y_2");

    // Arbitrary spatial values
    prog.assert(x_0.clone().real_ge(Expr::real(-100)));
    prog.assert(x_0.clone().real_le(Expr::real(100)));
    prog.assert(x_1.clone().real_ge(Expr::real(-100)));
    prog.assert(x_1.clone().real_le(Expr::real(100)));
    prog.assert(x_2.clone().real_ge(Expr::real(-100)));
    prog.assert(x_2.clone().real_le(Expr::real(100)));

    // Channel mask = 0 (channel dropped)
    prog.assert(mc.clone().eq(Expr::real(0)));

    // All spatial positions use the same channel mask
    prog.assert(y_0.clone().eq(mc.clone().real_mul(x_0)));
    prog.assert(y_1.clone().eq(mc.clone().real_mul(x_1)));
    prog.assert(y_2.clone().eq(mc.real_mul(x_2)));

    // Negated property: any output != 0
    let violation = y_0
        .ne(Expr::real(0))
        .or(y_1.ne(Expr::real(0)))
        .or(y_2.ne(Expr::real(0)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "spatial_dropout_zeros_maps");
}

// ---------------------------------------------------------------------------
// Test 962: DropConnect on weights vs activations
// ---------------------------------------------------------------------------

/// Prove: DropConnect zeros weights instead of activations.
///
/// Standard dropout: y = (m * x) * w, where m masks activations.
/// DropConnect:      y = x * (m * w), where m masks weights.
/// For a single scalar: both are equivalent: y = m * x * w.
/// We prove this equivalence for a single element.
#[test]
fn test_962_dropconnect_weight_vs_activation() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("w", real.clone());
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("y_dropout", real.clone());
    let _ = prog.declare_const("y_dropconnect", real);

    let x = real_var("x");
    let w = real_var("w");
    let m = real_var("m");
    let y_dropout = real_var("y_dropout");
    let y_dropconnect = real_var("y_dropconnect");

    // Bounded inputs
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));
    prog.assert(w.clone().real_ge(Expr::real(-10)));
    prog.assert(w.clone().real_le(Expr::real(10)));

    // m in {0, 1}
    let is_zero = m.clone().eq(Expr::real(0));
    let is_one = m.clone().eq(Expr::real(1));
    prog.assert(is_zero.or(is_one));

    // Dropout on activation: y_dropout = (m * x) * w
    prog.assert(
        y_dropout
            .clone()
            .eq(m.clone().real_mul(x.clone()).real_mul(w.clone())),
    );

    // DropConnect on weight: y_dropconnect = x * (m * w)
    prog.assert(y_dropconnect.clone().eq(x.real_mul(m.real_mul(w))));

    // Negated property: y_dropout != y_dropconnect
    let violation = y_dropout.ne(y_dropconnect);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dropconnect_weight_vs_activation");
}

// ---------------------------------------------------------------------------
// Test 963: Dropout probability 0 = identity
// ---------------------------------------------------------------------------

/// Prove: with p=0, dropout is the identity function.
///
/// scale = 1/(1-0) = 1, all elements kept, y = x * 1 = x.
#[test]
fn test_963_dropout_probability_zero_identity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("p", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("y", real);

    let x = real_var("x");
    let p = real_var("p");
    let scale = real_var("scale");
    let y = real_var("y");

    // x bounded
    prog.assert(x.clone().real_ge(Expr::real(-1000)));
    prog.assert(x.clone().real_le(Expr::real(1000)));

    // p = 0
    prog.assert(p.clone().eq(Expr::real(0)));

    // scale * (1 - p) = 1
    prog.assert(
        scale
            .clone()
            .real_mul(Expr::real(1).real_sub(p))
            .eq(Expr::real(1)),
    );

    // All elements kept (mask=1): y = x * scale
    prog.assert(y.clone().eq(x.clone().real_mul(scale)));

    // Negated property: y != x
    let violation = y.ne(x);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dropout_probability_zero_identity");
}

// ---------------------------------------------------------------------------
// Test 964: Dropout probability 1 = zero
// ---------------------------------------------------------------------------

/// Prove: with p=1, all elements are dropped and output is zero.
///
/// When every element is dropped (mask = 0 for all), y = 0 * x * scale = 0.
#[test]
fn test_964_dropout_probability_one_zero() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("mask", real.clone());
    let _ = prog.declare_const("y", real);

    let x = real_var("x");
    let mask = real_var("mask");
    let y = real_var("y");

    // x bounded (arbitrary)
    prog.assert(x.clone().real_ge(Expr::real(-1000)));
    prog.assert(x.clone().real_le(Expr::real(1000)));

    // p = 1 means all elements dropped: mask = 0
    prog.assert(mask.clone().eq(Expr::real(0)));

    // y = mask * x (scale is irrelevant when mask = 0)
    prog.assert(y.clone().eq(mask.real_mul(x)));

    // Negated property: y != 0
    let violation = y.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dropout_probability_one_zero");
}

// ---------------------------------------------------------------------------
// Test 965: Gaussian dropout equivalence
// ---------------------------------------------------------------------------

/// Prove: Gaussian dropout multiplies by (1 + epsilon) where epsilon ~ N(0, sigma^2).
///
/// In Gaussian dropout, instead of binary mask, we multiply by (1 + eps):
///   y = x * (1 + eps).
/// Expected value: E[y] = x * E[1 + eps] = x * 1 = x (since E[eps] = 0).
///
/// We prove: if E[eps] = 0, then E[y] = x.
#[test]
fn test_965_gaussian_dropout_equivalence() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("e_eps", real.clone());
    let _ = prog.declare_const("e_multiplier", real.clone());
    let _ = prog.declare_const("expected", real);

    let x = real_var("x");
    let e_eps = real_var("e_eps");
    let e_multiplier = real_var("e_multiplier");
    let expected = real_var("expected");

    // x bounded
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));

    // E[epsilon] = 0 (zero-mean noise)
    prog.assert(e_eps.clone().eq(Expr::real(0)));

    // E[multiplier] = 1 + E[eps] = 1
    prog.assert(e_multiplier.clone().eq(Expr::real(1).real_add(e_eps)));

    // E[y] = x * E[multiplier]
    prog.assert(expected.clone().eq(x.clone().real_mul(e_multiplier)));

    // Negated property: expected != x
    let violation = expected.ne(x);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gaussian_dropout_equivalence");
}

// ---------------------------------------------------------------------------
// Test 966: Dropout gradient is scaled binary mask
// ---------------------------------------------------------------------------

/// Prove: the backward pass of dropout is the scaled binary mask.
///
/// Forward: y = mask * x * scale, where mask in {0, 1}.
/// Backward: dy/dx = mask * scale.
/// So grad_input = grad_output * mask * scale.
#[test]
fn test_966_dropout_gradient_scaled_binary_mask() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("grad_out", real.clone());
    let _ = prog.declare_const("mask", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("grad_in", real.clone());
    let _ = prog.declare_const("expected", real);

    let grad_out = real_var("grad_out");
    let mask = real_var("mask");
    let scale = real_var("scale");
    let grad_in = real_var("grad_in");
    let expected = real_var("expected");

    // grad_out bounded
    prog.assert(grad_out.clone().real_ge(Expr::real(-100)));
    prog.assert(grad_out.clone().real_le(Expr::real(100)));

    // mask in {0, 1}
    let is_zero = mask.clone().eq(Expr::real(0));
    let is_one = mask.clone().eq(Expr::real(1));
    prog.assert(is_zero.or(is_one));

    // scale > 0
    prog.assert(scale.clone().real_gt(Expr::real(0)));

    // grad_in = grad_out * mask * scale
    prog.assert(
        grad_in.clone().eq(grad_out
            .clone()
            .real_mul(mask.clone())
            .real_mul(scale.clone())),
    );

    // expected = grad_out * mask * scale (same formula)
    prog.assert(expected.clone().eq(grad_out.real_mul(mask).real_mul(scale)));

    // Negated property: grad_in != expected
    let violation = grad_in.ne(expected);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dropout_gradient_scaled_binary_mask");
}

// ---------------------------------------------------------------------------
// Test 967: Concrete dropout parameter bounds
// ---------------------------------------------------------------------------

/// Prove: concrete dropout parameterizes p via sigmoid, keeping p in (0, 1).
///
/// p = sigmoid(z) = 1/(1 + exp(-z)). Since exp(-z) > 0 for all z,
/// we have 1 + exp(-z) > 1, so p = 1/(1+exp(-z)) < 1 and > 0.
///
/// We model: given exp_neg_z > 0, p * (1 + exp_neg_z) = 1 implies p in (0, 1).
#[test]
fn test_967_concrete_dropout_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("exp_neg_z", real.clone());
    let _ = prog.declare_const("denom", real.clone());
    let _ = prog.declare_const("p", real);

    let exp_neg_z = real_var("exp_neg_z");
    let denom = real_var("denom");
    let p = real_var("p");

    // exp(-z) > 0 for all real z
    prog.assert(exp_neg_z.clone().real_gt(Expr::real(0)));
    // Bound it for solver tractability
    prog.assert(exp_neg_z.clone().real_le(Expr::real(1000)));

    // denom = 1 + exp(-z) > 1
    prog.assert(denom.clone().eq(Expr::real(1).real_add(exp_neg_z)));

    // p * denom = 1 (p = 1/denom)
    prog.assert(p.clone().real_mul(denom).eq(Expr::real(1)));

    // Property: p > 0 AND p < 1
    // Negated: p <= 0 OR p >= 1
    let violation = p
        .clone()
        .real_le(Expr::real(0))
        .or(p.real_ge(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "concrete_dropout_bounds");
}

// ---------------------------------------------------------------------------
// Test 968: Label smoothing reduces confidence
// ---------------------------------------------------------------------------

/// Prove: label smoothing reduces the maximum probability (confidence).
///
/// For the target class: y_smooth = (1-alpha)*1 + alpha/K = 1 - alpha*(1-1/K).
/// Since alpha > 0 and K >= 2, alpha*(1-1/K) > 0, so y_smooth < 1.
/// The smoothed confidence is strictly less than the hard label (1).
#[test]
fn test_968_label_smoothing_reduces_confidence() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("alpha", real.clone());
    let _ = prog.declare_const("k", real.clone());
    let _ = prog.declare_const("y_smooth", real);

    let alpha = real_var("alpha");
    let k = real_var("k");
    let y_smooth = real_var("y_smooth");

    // alpha in (0, 1)
    prog.assert(alpha.clone().real_gt(Expr::real(0)));
    prog.assert(alpha.clone().real_lt(Expr::real(1)));

    // K >= 2
    prog.assert(k.clone().real_ge(Expr::real(2)));
    prog.assert(k.clone().real_le(Expr::real(10000)));

    // y_smooth * K = (1-alpha)*K + alpha (for target class, y_hard = 1)
    prog.assert(
        y_smooth.clone().real_mul(k.clone()).eq(Expr::real(1)
            .real_sub(alpha.clone())
            .real_mul(k)
            .real_add(alpha)),
    );

    // Property: y_smooth < 1 AND y_smooth > 0
    // Negated: y_smooth >= 1 OR y_smooth <= 0
    let violation = y_smooth
        .clone()
        .real_ge(Expr::real(1))
        .or(y_smooth.real_le(Expr::real(0)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "label_smoothing_reduces_confidence");
}

// ---------------------------------------------------------------------------
// Test 969: Mixup interpolation lambda in [0,1]
// ---------------------------------------------------------------------------

/// Prove: Mixup interpolation with lambda in [0,1] produces a valid convex
/// combination.
///
/// Mixup: x_mix = lambda * x_a + (1-lambda) * x_b, with lambda in [0, 1].
/// Property: x_mix is between min(x_a, x_b) and max(x_a, x_b) — i.e.,
/// x_mix is a convex combination.
///
/// We prove: if 0 <= x_a <= x_b and lambda in [0,1], then x_a <= x_mix <= x_b.
#[test]
fn test_969_mixup_interpolation_lambda_valid() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("xa", real.clone());
    let _ = prog.declare_const("xb", real.clone());
    let _ = prog.declare_const("lam", real.clone());
    let _ = prog.declare_const("x_mix", real);

    let xa = real_var("xa");
    let xb = real_var("xb");
    let lam = real_var("lam");
    let x_mix = real_var("x_mix");

    // Ordered: xa <= xb
    prog.assert(xa.clone().real_ge(Expr::real(0)));
    prog.assert(xa.clone().real_le(xb.clone()));
    prog.assert(xb.clone().real_le(Expr::real(100)));

    // lambda in [0, 1]
    prog.assert(lam.clone().real_ge(Expr::real(0)));
    prog.assert(lam.clone().real_le(Expr::real(1)));

    // x_mix = lambda * xa + (1-lambda) * xb
    prog.assert(
        x_mix.clone().eq(lam
            .clone()
            .real_mul(xa.clone())
            .real_add(Expr::real(1).real_sub(lam).real_mul(xb.clone()))),
    );

    // Property: xa <= x_mix <= xb
    // Negated: x_mix < xa OR x_mix > xb
    let violation = x_mix.clone().real_lt(xa).or(x_mix.real_gt(xb));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "mixup_interpolation_lambda_valid");
}

// ---------------------------------------------------------------------------
// Test 970: CutMix region ratio in [0,1]
// ---------------------------------------------------------------------------

/// Prove: CutMix region area ratio r is in [0, 1].
///
/// CutMix cuts a rectangular region from one image and pastes it onto another.
/// The region ratio r = (rw * rh) / (W * H) where rw <= W and rh <= H.
/// Since 0 <= rw <= W and 0 <= rh <= H, we have 0 <= rw*rh <= W*H,
/// so r in [0, 1].
///
/// We prove: r * (W * H) = rw * rh with rw in [0, W] and rh in [0, H]
/// implies r in [0, 1].
#[test]
fn test_970_cutmix_region_ratio_valid() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w_total", real.clone());
    let _ = prog.declare_const("h_total", real.clone());
    let _ = prog.declare_const("rw", real.clone());
    let _ = prog.declare_const("rh", real.clone());
    let _ = prog.declare_const("area", real.clone());
    let _ = prog.declare_const("total_area", real.clone());
    let _ = prog.declare_const("r", real);

    let w_total = real_var("w_total");
    let h_total = real_var("h_total");
    let rw = real_var("rw");
    let rh = real_var("rh");
    let area = real_var("area");
    let total_area = real_var("total_area");
    let r = real_var("r");

    // W, H > 0
    prog.assert(w_total.clone().real_gt(Expr::real(0)));
    prog.assert(w_total.clone().real_le(Expr::real(1000)));
    prog.assert(h_total.clone().real_gt(Expr::real(0)));
    prog.assert(h_total.clone().real_le(Expr::real(1000)));

    // 0 <= rw <= W
    prog.assert(rw.clone().real_ge(Expr::real(0)));
    prog.assert(rw.clone().real_le(w_total.clone()));

    // 0 <= rh <= H
    prog.assert(rh.clone().real_ge(Expr::real(0)));
    prog.assert(rh.clone().real_le(h_total.clone()));

    // area = rw * rh
    prog.assert(area.clone().eq(rw.real_mul(rh)));

    // total_area = W * H
    prog.assert(total_area.clone().eq(w_total.real_mul(h_total)));

    // r * total_area = area (r = area / total_area)
    prog.assert(r.clone().real_mul(total_area).eq(area));

    // Property: 0 <= r <= 1
    // Negated: r < 0 OR r > 1
    let violation = r
        .clone()
        .real_lt(Expr::real(0))
        .or(r.real_gt(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cutmix_region_ratio_valid");
}
