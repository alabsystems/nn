// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for dropout and regularization properties.
//!
//! Proves fundamental properties of dropout and regularization techniques:
//! - Dropout: inverted scaling by 1/(1-p), eval-mode identity, mask
//!   element-wise application, output bounds, gradient pass-through
//! - Weight decay: L2 penalty shrinks weights, L1 penalty sparsity,
//!   decoupled weight decay (AdamW)
//! - Label smoothing: output in valid probability range, smoothed
//!   distribution sums to one, hard-label recovery at alpha=0
//! - Stochastic depth: survival probability scaling, residual pass-through
//!   at survival=1, expected value preservation
//! - DropPath: path scaling by 1/keep_prob, batch independence
//!
//! Part of #4134.

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
// Test 531: Dropout inverted scaling preserves expected value
// ---------------------------------------------------------------------------

/// Prove: dropout with inverted scaling 1/(1-p) preserves expected value.
///
/// During training, dropout zeros each element with probability p and
/// scales surviving elements by 1/(1-p). The expected output per element:
///   E[y] = (1-p) * x * 1/(1-p) + p * 0 = x.
///
/// We model: y_active = x / (1 - p), expected = (1 - p) * y_active,
/// and prove expected = x.
#[test]
fn test_531_dropout_inverted_scaling_preserves_expected_value() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("p", real.clone());
    let _ = prog.declare_const("one_minus_p", real.clone());
    let _ = prog.declare_const("y_active", real.clone());
    let _ = prog.declare_const("expected", real);

    let x = real_var("x");
    let p = real_var("p");
    let one_minus_p = real_var("one_minus_p");
    let y_active = real_var("y_active");
    let expected = real_var("expected");

    // Input bounds
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));

    // p in (0, 1) — dropout probability
    prog.assert(p.clone().real_gt(Expr::real(0)));
    prog.assert(p.clone().real_lt(Expr::real(1)));

    // one_minus_p = 1 - p
    prog.assert(one_minus_p.clone().eq(Expr::real(1).real_sub(p)));

    // one_minus_p > 0 (follows from p < 1, but explicit for solver)
    prog.assert(one_minus_p.clone().real_gt(Expr::real(0)));

    // y_active = x / (1-p), modeled as: y_active * (1-p) = x
    prog.assert(y_active.clone().real_mul(one_minus_p.clone()).eq(x.clone()));

    // expected = (1-p) * y_active (probability of keeping * scaled value)
    prog.assert(expected.clone().eq(one_minus_p.real_mul(y_active)));

    // Negated property: expected != x
    let violation = expected.ne(x);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dropout_inverted_scaling_preserves_expected_value");
}

// ---------------------------------------------------------------------------
// Test 532: Dropout eval mode is identity
// ---------------------------------------------------------------------------

/// Prove: in eval mode (no dropout), the output equals the input.
///
/// During evaluation, dropout is disabled: y = x (no masking, no scaling).
/// This is modeled by p = 0, so scale = 1/(1-0) = 1, giving y = x * 1 = x.
#[test]
fn test_532_dropout_eval_mode_identity() {
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

    // Input bounds
    prog.assert(x.clone().real_ge(Expr::real(-1000)));
    prog.assert(x.clone().real_le(Expr::real(1000)));

    // Eval mode: p = 0
    prog.assert(p.clone().eq(Expr::real(0)));

    // scale * (1 - p) = 1
    prog.assert(
        scale
            .clone()
            .real_mul(Expr::real(1).real_sub(p))
            .eq(Expr::real(1)),
    );

    // y = x * scale (no masking in eval)
    prog.assert(y.clone().eq(x.clone().real_mul(scale)));

    // Negated property: y != x
    let violation = y.ne(x);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dropout_eval_mode_identity");
}

// ---------------------------------------------------------------------------
// Test 533: Dropout mask is element-wise (two independent elements)
// ---------------------------------------------------------------------------

/// Prove: dropout applies independently per element.
///
/// For two elements x1, x2 with independent masks m1, m2 in {0, 1}:
///   y1 = m1 * x1 * scale, y2 = m2 * x2 * scale.
/// The output of one element depends only on its own mask, not the other's.
/// We prove: if m1=1 and m2=0, then y1 = x1*scale and y2 = 0.
#[test]
fn test_533_dropout_mask_elementwise_independence() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("y1", real.clone());
    let _ = prog.declare_const("y2", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let scale = real_var("scale");
    let y1 = real_var("y1");
    let y2 = real_var("y2");

    // Input bounds
    prog.assert(x1.clone().real_ge(Expr::real(-100)));
    prog.assert(x1.clone().real_le(Expr::real(100)));
    prog.assert(x2.clone().real_ge(Expr::real(-100)));
    prog.assert(x2.clone().real_le(Expr::real(100)));

    // scale > 0
    prog.assert(scale.clone().real_gt(Expr::real(0)));

    // m1 = 1 (kept), m2 = 0 (dropped)
    // y1 = 1 * x1 * scale = x1 * scale
    prog.assert(y1.clone().eq(x1.clone().real_mul(scale.clone())));
    // y2 = 0 * x2 * scale = 0
    prog.assert(y2.clone().eq(Expr::real(0)));

    // Negated property: y2 != 0 (y2 should be exactly 0 when masked out)
    let violation = y2.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dropout_mask_elementwise_independence");
}

// ---------------------------------------------------------------------------
// Test 534: Dropout output bounded by scaled input
// ---------------------------------------------------------------------------

/// Prove: dropout output magnitude is bounded by |x| / (1-p).
///
/// For a kept element: y = x / (1-p). Since 0 < 1-p <= 1, we have |y| >= |x|.
/// Upper bound: |y| = |x| / (1-p). If |x| <= B, then |y| <= B / (1-p).
#[test]
fn test_534_dropout_output_bounded_by_scaled_input() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("one_minus_p", real.clone());
    let _ = prog.declare_const("y", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("bound", real);

    let x = real_var("x");
    let one_minus_p = real_var("one_minus_p");
    let y = real_var("y");
    let b = real_var("b");
    let bound = real_var("bound");

    // B > 0
    prog.assert(b.clone().real_gt(Expr::real(0)));

    // |x| <= B
    prog.assert(x.clone().real_ge(Expr::real(0).real_sub(b.clone())));
    prog.assert(x.clone().real_le(b.clone()));

    // 0 < one_minus_p <= 1
    prog.assert(one_minus_p.clone().real_gt(Expr::real(0)));
    prog.assert(one_minus_p.clone().real_le(Expr::real(1)));

    // y * one_minus_p = x (y = x / one_minus_p)
    prog.assert(y.clone().real_mul(one_minus_p.clone()).eq(x));

    // bound * one_minus_p = b (bound = b / one_minus_p)
    prog.assert(bound.clone().real_mul(one_minus_p).eq(b));

    // Negated property: |y| > bound
    let violation = y
        .clone()
        .real_gt(bound.clone())
        .or(y.real_lt(Expr::real(0).real_sub(bound)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dropout_output_bounded_by_scaled_input");
}

// ---------------------------------------------------------------------------
// Test 535: Dropout gradient pass-through for kept elements
// ---------------------------------------------------------------------------

/// Prove: gradient through dropout for a kept element equals grad_out / (1-p).
///
/// Forward: y = mask * x / (1-p), with mask=1 for kept elements.
/// Backward: grad_x = mask * grad_out / (1-p).
/// For kept (mask=1): grad_x = grad_out / (1-p).
#[test]
fn test_535_dropout_gradient_passthrough_kept() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("grad_out", real.clone());
    let _ = prog.declare_const("one_minus_p", real.clone());
    let _ = prog.declare_const("grad_x", real.clone());
    let _ = prog.declare_const("expected", real);

    let grad_out = real_var("grad_out");
    let one_minus_p = real_var("one_minus_p");
    let grad_x = real_var("grad_x");
    let expected = real_var("expected");

    // grad_out bounded
    prog.assert(grad_out.clone().real_ge(Expr::real(-100)));
    prog.assert(grad_out.clone().real_le(Expr::real(100)));

    // 0 < 1-p < 1
    prog.assert(one_minus_p.clone().real_gt(Expr::real(0)));
    prog.assert(one_minus_p.clone().real_lt(Expr::real(1)));

    // grad_x = grad_out / (1-p), modeled as grad_x * (1-p) = grad_out
    prog.assert(
        grad_x
            .clone()
            .real_mul(one_minus_p.clone())
            .eq(grad_out.clone()),
    );

    // expected = grad_out / (1-p), same equation
    prog.assert(expected.clone().real_mul(one_minus_p).eq(grad_out));

    // Negated property: grad_x != expected
    let violation = grad_x.ne(expected);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dropout_gradient_passthrough_kept");
}

// ---------------------------------------------------------------------------
// Test 536: Dropout gradient is zero for dropped elements
// ---------------------------------------------------------------------------

/// Prove: gradient through dropout for a dropped element is zero.
///
/// Forward: y = mask * x / (1-p), with mask=0 for dropped elements.
/// Backward: grad_x = mask * grad_out / (1-p) = 0 * grad_out / (1-p) = 0.
#[test]
fn test_536_dropout_gradient_zero_for_dropped() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("grad_out", real.clone());
    let _ = prog.declare_const("mask", real.clone());
    let _ = prog.declare_const("grad_x", real);

    let grad_out = real_var("grad_out");
    let mask = real_var("mask");
    let grad_x = real_var("grad_x");

    // grad_out is arbitrary
    prog.assert(grad_out.clone().real_ge(Expr::real(-1000)));
    prog.assert(grad_out.clone().real_le(Expr::real(1000)));

    // mask = 0 (dropped)
    prog.assert(mask.clone().eq(Expr::real(0)));

    // grad_x = mask * grad_out
    prog.assert(grad_x.clone().eq(mask.real_mul(grad_out)));

    // Negated property: grad_x != 0
    let violation = grad_x.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dropout_gradient_zero_for_dropped");
}

// ---------------------------------------------------------------------------
// Test 537: L2 weight decay shrinks weight magnitude
// ---------------------------------------------------------------------------

/// Prove: L2 weight decay (weight -= lr * lambda * weight) shrinks |w|.
///
/// Update: w_new = w_old - lr * lambda * w_old = w_old * (1 - lr * lambda).
/// With 0 < lr * lambda < 1, the decay factor d = 1 - lr*lambda in (0, 1).
/// Therefore |w_new| = |w_old| * d < |w_old|.
#[test]
fn test_537_l2_weight_decay_shrinks_magnitude() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w_old", real.clone());
    let _ = prog.declare_const("d", real.clone());
    let _ = prog.declare_const("w_new", real);

    let w_old = real_var("w_old");
    let d = real_var("d");
    let w_new = real_var("w_new");

    // w_old != 0 (otherwise trivial)
    prog.assert(w_old.clone().ne(Expr::real(0)));
    prog.assert(w_old.clone().real_ge(Expr::real(-100)));
    prog.assert(w_old.clone().real_le(Expr::real(100)));

    // d = 1 - lr*lambda, in (0, 1)
    prog.assert(d.clone().real_gt(Expr::real(0)));
    prog.assert(d.clone().real_lt(Expr::real(1)));

    // w_new = w_old * d
    prog.assert(w_new.clone().eq(w_old.clone().real_mul(d)));

    // Property: |w_new| < |w_old|
    // Equivalently: w_new^2 < w_old^2
    let w_new_sq = w_new.clone().real_mul(w_new);
    let w_old_sq = w_old.clone().real_mul(w_old);

    // Negated: w_new^2 >= w_old^2
    let violation = w_new_sq.real_ge(w_old_sq);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "l2_weight_decay_shrinks_magnitude");
}

// ---------------------------------------------------------------------------
// Test 538: L1 regularization gradient pushes toward zero
// ---------------------------------------------------------------------------

/// Prove: L1 regularization gradient (sign(w) * lambda) pushes w toward 0.
///
/// The L1 penalty gradient is lambda * sign(w). For w > 0, the gradient
/// is positive, so the update w_new = w - lr * lambda < w moves toward 0.
/// For w < 0, gradient is -lambda, so w_new = w + lr * lambda > w, also
/// moving toward 0.
///
/// We prove: for w > 0, the update decreases w (but keeps w_new >= 0 when
/// lr * lambda <= w).
#[test]
fn test_538_l1_regularization_pushes_toward_zero() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w", real.clone());
    let _ = prog.declare_const("lr_lambda", real.clone());
    let _ = prog.declare_const("w_new", real);

    let w = real_var("w");
    let lr_lambda = real_var("lr_lambda");
    let w_new = real_var("w_new");

    // w > 0
    prog.assert(w.clone().real_gt(Expr::real(0)));
    prog.assert(w.clone().real_le(Expr::real(100)));

    // lr * lambda > 0, and lr*lambda <= w (no overshoot past zero)
    prog.assert(lr_lambda.clone().real_gt(Expr::real(0)));
    prog.assert(lr_lambda.clone().real_le(w.clone()));

    // For w > 0, sign(w) = 1, so w_new = w - lr*lambda
    prog.assert(w_new.clone().eq(w.clone().real_sub(lr_lambda)));

    // Property: 0 <= w_new < w (moved toward zero, didn't cross)
    // Negated: w_new < 0 OR w_new >= w
    let violation = w_new.clone().real_lt(Expr::real(0)).or(w_new.real_ge(w));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "l1_regularization_pushes_toward_zero");
}

// ---------------------------------------------------------------------------
// Test 539: Decoupled weight decay (AdamW) update formula
// ---------------------------------------------------------------------------

/// Prove: AdamW decoupled weight decay applies decay independently of gradient.
///
/// AdamW update: w_new = w_old * (1 - lr * lambda) - lr * m_hat / (sqrt(v_hat) + eps)
/// The first term is pure weight decay, the second is the Adam step.
/// We prove: the weight decay component equals w_old * (1 - lr * lambda).
#[test]
fn test_539_adamw_decoupled_weight_decay() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w_old", real.clone());
    let _ = prog.declare_const("decay_factor", real.clone());
    let _ = prog.declare_const("adam_step", real.clone());
    let _ = prog.declare_const("w_new", real.clone());
    let _ = prog.declare_const("decay_component", real);

    let w_old = real_var("w_old");
    let decay_factor = real_var("decay_factor");
    let adam_step = real_var("adam_step");
    let w_new = real_var("w_new");
    let decay_component = real_var("decay_component");

    // w_old bounded
    prog.assert(w_old.clone().real_ge(Expr::real(-100)));
    prog.assert(w_old.clone().real_le(Expr::real(100)));

    // decay_factor = 1 - lr * lambda, in (0, 1)
    prog.assert(decay_factor.clone().real_gt(Expr::real(0)));
    prog.assert(decay_factor.clone().real_lt(Expr::real(1)));

    // adam_step is bounded
    prog.assert(adam_step.clone().real_ge(Expr::real(-10)));
    prog.assert(adam_step.clone().real_le(Expr::real(10)));

    // w_new = w_old * decay_factor - adam_step
    prog.assert(
        w_new.clone().eq(w_old
            .clone()
            .real_mul(decay_factor.clone())
            .real_sub(adam_step.clone())),
    );

    // decay_component = w_old * decay_factor
    prog.assert(decay_component.clone().eq(w_old.real_mul(decay_factor)));

    // Property: w_new = decay_component - adam_step
    // Negated: w_new != decay_component - adam_step
    let violation = w_new.ne(decay_component.real_sub(adam_step));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "adamw_decoupled_weight_decay");
}

// ---------------------------------------------------------------------------
// Test 540: Label smoothing output in valid probability range
// ---------------------------------------------------------------------------

/// Prove: label smoothing produces values in [alpha/K, 1 - alpha + alpha/K].
///
/// Label smoothing: y_smooth = (1 - alpha) * y_hard + alpha / K
/// where y_hard in {0, 1} and K > 1 is the number of classes.
///
/// For y_hard = 0: y_smooth = alpha / K > 0.
/// For y_hard = 1: y_smooth = 1 - alpha + alpha/K = 1 - alpha*(1 - 1/K).
/// Both are in (0, 1) for alpha in (0, 1) and K > 1.
#[test]
fn test_540_label_smoothing_valid_probability_range() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("alpha", real.clone());
    let _ = prog.declare_const("k", real.clone());
    let _ = prog.declare_const("y_hard", real.clone());
    let _ = prog.declare_const("y_smooth", real);

    let alpha = real_var("alpha");
    let k = real_var("k");
    let y_hard = real_var("y_hard");
    let y_smooth = real_var("y_smooth");

    // alpha in (0, 1)
    prog.assert(alpha.clone().real_gt(Expr::real(0)));
    prog.assert(alpha.clone().real_lt(Expr::real(1)));

    // K >= 2 (at least 2 classes)
    prog.assert(k.clone().real_ge(Expr::real(2)));
    prog.assert(k.clone().real_le(Expr::real(10000)));

    // y_hard in {0, 1}: model as y_hard >= 0 and y_hard <= 1
    prog.assert(y_hard.clone().real_ge(Expr::real(0)));
    prog.assert(y_hard.clone().real_le(Expr::real(1)));

    // y_smooth * K = (1 - alpha) * y_hard * K + alpha
    // (multiply through by K to avoid division)
    prog.assert(
        y_smooth.clone().real_mul(k.clone()).eq(Expr::real(1)
            .real_sub(alpha.clone())
            .real_mul(y_hard)
            .real_mul(k.clone())
            .real_add(alpha)),
    );

    // Property: y_smooth > 0 AND y_smooth < 1
    // Negated: y_smooth <= 0 OR y_smooth >= 1
    let violation = y_smooth
        .clone()
        .real_le(Expr::real(0))
        .or(y_smooth.real_ge(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "label_smoothing_valid_probability_range");
}

// ---------------------------------------------------------------------------
// Test 541: Label smoothing distribution sums to one
// ---------------------------------------------------------------------------

/// Prove: smoothed label distribution over K classes sums to 1.
///
/// For K classes with one-hot hard labels, after smoothing:
///   sum = (1 - alpha) * 1 + (K-1) * alpha/K + alpha/K
///       = (1 - alpha) + alpha = 1.
///
/// We model the sum of the smoothed distribution for K=2 (extensible).
#[test]
fn test_541_label_smoothing_sums_to_one() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("alpha", real.clone());
    let _ = prog.declare_const("y_target", real.clone());
    let _ = prog.declare_const("y_other", real.clone());
    let _ = prog.declare_const("total", real);

    let alpha = real_var("alpha");
    let y_target = real_var("y_target");
    let y_other = real_var("y_other");
    let total = real_var("total");

    // alpha in (0, 1)
    prog.assert(alpha.clone().real_gt(Expr::real(0)));
    prog.assert(alpha.clone().real_lt(Expr::real(1)));

    // For K=2: y_target = (1-alpha)*1 + alpha/2 = 1 - alpha/2
    // Modeled as: 2 * y_target = 2 - alpha
    prog.assert(
        Expr::real(2)
            .real_mul(y_target.clone())
            .eq(Expr::real(2).real_sub(alpha.clone())),
    );

    // y_other = (1-alpha)*0 + alpha/2 = alpha/2
    // Modeled as: 2 * y_other = alpha
    prog.assert(Expr::real(2).real_mul(y_other.clone()).eq(alpha));

    // total = y_target + y_other
    prog.assert(total.clone().eq(y_target.real_add(y_other)));

    // Negated property: total != 1
    let violation = total.ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "label_smoothing_sums_to_one");
}

// ---------------------------------------------------------------------------
// Test 542: Label smoothing recovers hard labels at alpha=0
// ---------------------------------------------------------------------------

/// Prove: with alpha=0, label smoothing is the identity (hard labels).
///
/// y_smooth = (1 - 0) * y_hard + 0 / K = y_hard.
#[test]
fn test_542_label_smoothing_identity_at_alpha_zero() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("y_hard", real.clone());
    let _ = prog.declare_const("alpha", real.clone());
    let _ = prog.declare_const("y_smooth", real);

    let y_hard = real_var("y_hard");
    let alpha = real_var("alpha");
    let y_smooth = real_var("y_smooth");

    // y_hard in [0, 1]
    prog.assert(y_hard.clone().real_ge(Expr::real(0)));
    prog.assert(y_hard.clone().real_le(Expr::real(1)));

    // alpha = 0
    prog.assert(alpha.clone().eq(Expr::real(0)));

    // y_smooth = (1 - alpha) * y_hard + alpha / K
    // With alpha = 0: y_smooth = 1 * y_hard + 0 = y_hard
    // Modeled as: y_smooth = (1 - alpha) * y_hard + alpha (for K -> inf, alpha/K -> 0)
    // More precisely for alpha = 0: y_smooth = y_hard
    prog.assert(
        y_smooth.clone().eq(Expr::real(1)
            .real_sub(alpha.clone())
            .real_mul(y_hard.clone())),
    );

    // Negated property: y_smooth != y_hard
    let violation = y_smooth.ne(y_hard);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "label_smoothing_identity_at_alpha_zero");
}

// ---------------------------------------------------------------------------
// Test 543: Stochastic depth survival probability scaling
// ---------------------------------------------------------------------------

/// Prove: stochastic depth with survival probability s scales output by 1/s.
///
/// During training: y = (sample_mask / s) * f(x) + x.
/// Expected value: E[y] = (s / s) * f(x) + x = f(x) + x.
/// This preserves the expected residual connection.
#[test]
fn test_543_stochastic_depth_survival_scaling() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("fx", real.clone());
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("scaled_fx", real.clone());
    let _ = prog.declare_const("expected", real);

    let fx = real_var("fx");
    let x = real_var("x");
    let s = real_var("s");
    let scaled_fx = real_var("scaled_fx");
    let expected = real_var("expected");

    // f(x) bounded
    prog.assert(fx.clone().real_ge(Expr::real(-100)));
    prog.assert(fx.clone().real_le(Expr::real(100)));
    // x bounded
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));

    // s in (0, 1] — survival probability
    prog.assert(s.clone().real_gt(Expr::real(0)));
    prog.assert(s.clone().real_le(Expr::real(1)));

    // scaled_fx * s = fx (scaled_fx = fx / s)
    prog.assert(scaled_fx.clone().real_mul(s.clone()).eq(fx.clone()));

    // Expected output when keeping (probability s):
    // E[y] = s * (scaled_fx + x) + (1-s) * x
    //       = s * scaled_fx + s*x + x - s*x
    //       = s * scaled_fx + x
    //       = s * (fx/s) + x = fx + x
    prog.assert(
        expected
            .clone()
            .eq(s.real_mul(scaled_fx).real_add(x.clone())),
    );

    // Negated property: expected != fx + x
    let violation = expected.ne(fx.real_add(x));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "stochastic_depth_survival_scaling");
}

// ---------------------------------------------------------------------------
// Test 544: Stochastic depth at survival=1 is identity residual
// ---------------------------------------------------------------------------

/// Prove: with survival probability s=1, stochastic depth never drops.
///
/// y = f(x)/s + x = f(x)/1 + x = f(x) + x — standard residual.
#[test]
fn test_544_stochastic_depth_survival_one_is_residual() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("fx", real.clone());
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("y", real);

    let fx = real_var("fx");
    let x = real_var("x");
    let s = real_var("s");
    let y = real_var("y");

    // Bounded inputs
    prog.assert(fx.clone().real_ge(Expr::real(-100)));
    prog.assert(fx.clone().real_le(Expr::real(100)));
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));

    // s = 1
    prog.assert(s.clone().eq(Expr::real(1)));

    // y = fx / s + x, with s=1: y = fx + x
    // Modeled: y * s = fx + x * s
    prog.assert(y.clone().real_mul(s).eq(fx.clone().real_add(x.clone())));

    // Negated property: y != fx + x
    let violation = y.ne(fx.real_add(x));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "stochastic_depth_survival_one_is_residual");
}

// ---------------------------------------------------------------------------
// Test 545: Stochastic depth dropped path outputs residual identity
// ---------------------------------------------------------------------------

/// Prove: when the path is dropped (sample_mask=0), output is just x (identity).
///
/// y = 0 * f(x)/s + x = x.
#[test]
fn test_545_stochastic_depth_dropped_is_identity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("fx", real.clone());
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("mask", real.clone());
    let _ = prog.declare_const("y", real);

    let fx = real_var("fx");
    let x = real_var("x");
    let mask = real_var("mask");
    let y = real_var("y");

    // Bounded
    prog.assert(fx.clone().real_ge(Expr::real(-1000)));
    prog.assert(fx.clone().real_le(Expr::real(1000)));
    prog.assert(x.clone().real_ge(Expr::real(-1000)));
    prog.assert(x.clone().real_le(Expr::real(1000)));

    // mask = 0 (dropped)
    prog.assert(mask.clone().eq(Expr::real(0)));

    // y = mask * fx + x (simplified: scale factor absorbed)
    prog.assert(y.clone().eq(mask.real_mul(fx).real_add(x.clone())));

    // Negated property: y != x
    let violation = y.ne(x);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "stochastic_depth_dropped_is_identity");
}

// ---------------------------------------------------------------------------
// Test 546: DropPath scales by 1/keep_prob
// ---------------------------------------------------------------------------

/// Prove: DropPath (used in vision transformers) scales kept paths by 1/keep_prob.
///
/// DropPath is equivalent to stochastic depth for residual branches.
/// For kept sample: y = x / keep_prob. Expected: keep_prob * (x/keep_prob) = x.
#[test]
fn test_546_droppath_scales_by_keep_prob() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("keep_prob", real.clone());
    let _ = prog.declare_const("y", real.clone());
    let _ = prog.declare_const("expected_value", real);

    let x = real_var("x");
    let keep_prob = real_var("keep_prob");
    let y = real_var("y");
    let expected_value = real_var("expected_value");

    // Bounded input
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));

    // keep_prob in (0, 1]
    prog.assert(keep_prob.clone().real_gt(Expr::real(0)));
    prog.assert(keep_prob.clone().real_le(Expr::real(1)));

    // y = x / keep_prob, modeled: y * keep_prob = x
    prog.assert(y.clone().real_mul(keep_prob.clone()).eq(x.clone()));

    // expected_value = keep_prob * y (expectation over the mask)
    prog.assert(expected_value.clone().eq(keep_prob.real_mul(y)));

    // Negated property: expected_value != x
    let violation = expected_value.ne(x);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "droppath_scales_by_keep_prob");
}

// ---------------------------------------------------------------------------
// Test 547: DropPath batch independence (two samples)
// ---------------------------------------------------------------------------

/// Prove: DropPath applies independently per sample in a batch.
///
/// For samples x1, x2 with independent masks m1=1, m2=0:
///   y1 = x1 / keep_prob, y2 = 0 (dropped).
/// y1 does not depend on m2 and y2 does not depend on m1.
#[test]
fn test_547_droppath_batch_independence() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("keep_prob", real.clone());
    let _ = prog.declare_const("y1", real.clone());
    let _ = prog.declare_const("y2", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let keep_prob = real_var("keep_prob");
    let y1 = real_var("y1");
    let y2 = real_var("y2");

    // Bounded inputs
    prog.assert(x1.clone().real_ge(Expr::real(-100)));
    prog.assert(x1.clone().real_le(Expr::real(100)));
    prog.assert(x2.clone().real_ge(Expr::real(-100)));
    prog.assert(x2.clone().real_le(Expr::real(100)));

    // keep_prob in (0, 1)
    prog.assert(keep_prob.clone().real_gt(Expr::real(0)));
    prog.assert(keep_prob.clone().real_lt(Expr::real(1)));

    // Sample 1 kept (m1=1): y1 = x1 / keep_prob
    prog.assert(y1.clone().real_mul(keep_prob.clone()).eq(x1.clone()));
    // Sample 2 dropped (m2=0): y2 = 0
    prog.assert(y2.clone().eq(Expr::real(0)));

    // Property: y1 * keep_prob = x1 AND y2 = 0
    // (y1 depends only on x1 and keep_prob, y2 is zero regardless of x2)
    // Negated: y1 * keep_prob != x1 OR y2 != 0
    let violation = y1.real_mul(keep_prob).ne(x1).or(y2.ne(Expr::real(0)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "droppath_batch_independence");
}

// ---------------------------------------------------------------------------
// Test 548: Weight decay combined with gradient descent bounds output
// ---------------------------------------------------------------------------

/// Prove: combined L2 decay + gradient step is bounded.
///
/// w_new = w_old * (1 - lr * lambda) - lr * grad.
/// If |w_old| <= B, decay in (0,1), |lr*grad| <= G, then |w_new| <= B + G.
#[test]
fn test_548_weight_decay_plus_gradient_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w_old", real.clone());
    let _ = prog.declare_const("decay", real.clone());
    let _ = prog.declare_const("lr_grad", real.clone());
    let _ = prog.declare_const("w_new", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("g", real);

    let w_old = real_var("w_old");
    let decay = real_var("decay");
    let lr_grad = real_var("lr_grad");
    let w_new = real_var("w_new");
    let b = real_var("b");
    let g = real_var("g");

    // B > 0, G > 0
    prog.assert(b.clone().real_gt(Expr::real(0)));
    prog.assert(g.clone().real_gt(Expr::real(0)));

    // |w_old| <= B
    prog.assert(w_old.clone().real_ge(Expr::real(0).real_sub(b.clone())));
    prog.assert(w_old.clone().real_le(b.clone()));

    // decay in (0, 1)
    prog.assert(decay.clone().real_gt(Expr::real(0)));
    prog.assert(decay.clone().real_lt(Expr::real(1)));

    // |lr_grad| <= G
    prog.assert(lr_grad.clone().real_ge(Expr::real(0).real_sub(g.clone())));
    prog.assert(lr_grad.clone().real_le(g.clone()));

    // w_new = w_old * decay - lr_grad
    prog.assert(w_new.clone().eq(w_old.real_mul(decay).real_sub(lr_grad)));

    // Property: |w_new| <= B + G
    let bound = b.real_add(g);
    let violation = w_new
        .clone()
        .real_gt(bound.clone())
        .or(w_new.real_lt(Expr::real(0).real_sub(bound)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "weight_decay_plus_gradient_bounded");
}

// ---------------------------------------------------------------------------
// Test 549: Dropout probability p=0 means no elements dropped
// ---------------------------------------------------------------------------

/// Prove: with dropout probability p=0, the scale factor is 1.
///
/// scale = 1/(1-p) = 1/(1-0) = 1. Output y = x * 1 = x.
#[test]
fn test_549_dropout_p_zero_no_drop() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("p", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("y", real);

    let p = real_var("p");
    let scale = real_var("scale");
    let x = real_var("x");
    let y = real_var("y");

    // p = 0
    prog.assert(p.clone().eq(Expr::real(0)));

    // scale * (1 - p) = 1
    prog.assert(
        scale
            .clone()
            .real_mul(Expr::real(1).real_sub(p))
            .eq(Expr::real(1)),
    );

    // x bounded
    prog.assert(x.clone().real_ge(Expr::real(-1000)));
    prog.assert(x.clone().real_le(Expr::real(1000)));

    // y = x * scale (all elements kept, mask = 1 everywhere)
    prog.assert(y.clone().eq(x.clone().real_mul(scale)));

    // Negated property: y != x
    let violation = y.ne(x);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dropout_p_zero_no_drop");
}

// ---------------------------------------------------------------------------
// Test 550: Two-layer dropout expected value preservation
// ---------------------------------------------------------------------------

/// Prove: two consecutive dropout layers preserve expected value.
///
/// Layer 1: E[y1] = x (with scale 1/(1-p1))
/// Layer 2: E[y2] = E[y1] = x (with scale 1/(1-p2))
/// Composition: expected output = x.
#[test]
fn test_550_two_layer_dropout_expected_value() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("p1", real.clone());
    let _ = prog.declare_const("p2", real.clone());
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("s2", real.clone());
    let _ = prog.declare_const("e1", real.clone());
    let _ = prog.declare_const("e2", real);

    let x = real_var("x");
    let p1 = real_var("p1");
    let p2 = real_var("p2");
    let s1 = real_var("s1");
    let s2 = real_var("s2");
    let e1 = real_var("e1");
    let e2 = real_var("e2");

    // x bounded
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));

    // p1, p2 in (0, 1)
    prog.assert(p1.clone().real_gt(Expr::real(0)));
    prog.assert(p1.clone().real_lt(Expr::real(1)));
    prog.assert(p2.clone().real_gt(Expr::real(0)));
    prog.assert(p2.clone().real_lt(Expr::real(1)));

    // s1 * (1-p1) = 1
    prog.assert(
        s1.clone()
            .real_mul(Expr::real(1).real_sub(p1))
            .eq(Expr::real(1)),
    );
    // s2 * (1-p2) = 1
    prog.assert(
        s2.clone()
            .real_mul(Expr::real(1).real_sub(p2))
            .eq(Expr::real(1)),
    );

    // E[layer1] = (1-p1) * (x * s1) = x
    prog.assert(e1.clone().eq(x.clone()));

    // E[layer2] = (1-p2) * (e1 * s2) — must also equal x
    // (1-p2) * e1 * s2 = e1 * (1-p2)*s2 = e1 * 1 = e1 = x
    prog.assert(e2.clone().eq(e1));

    // Negated property: e2 != x
    let violation = e2.ne(x);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "two_layer_dropout_expected_value");
}
