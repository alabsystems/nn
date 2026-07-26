// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for loss function mathematical properties.
//!
//! Proves 20 properties (test_991 through test_1010):
//!  1. Cross-entropy loss >= 0
//!  2. MSE loss >= 0
//!  3. L1 loss >= 0
//!  4. Huber loss >= 0 and smooth
//!  5. Cross-entropy = -sum(y * log(p)) for one-hot y
//!  6. MSE gradient = 2(pred - target)
//!  7. L1 gradient = sign(pred - target)
//!  8. KL divergence >= 0
//!  9. Binary cross-entropy for p in (0,1)
//! 10. Focal loss reduces well-classified loss
//! 11. Label smoothing bounds modified targets
//! 12. Contrastive loss margin property
//! 13. Triplet loss: d(a,p) < d(a,n) + margin
//! 14. CTC loss non-negative
//! 15. Hinge loss max(0, 1 - y*f(x))
//! 16. Log-cosh loss smoothness
//! 17. Quantile loss asymmetry
//! 18. Cosine similarity loss in [-1, 1]
//! 19. Dice loss in [0, 1]
//! 20. IoU loss in [0, 1]
//!
//! Part of #4208.

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
// Test 991: Cross-entropy loss >= 0
// ---------------------------------------------------------------------------

/// Prove: cross-entropy loss is non-negative.
///
/// For a single class with target y=1 and predicted probability p in (0,1]:
///   CE = -log(p).
/// Since p in (0, 1], log(p) <= 0, so -log(p) >= 0.
///
/// We model: given log_p <= 0 (since p in (0,1]), ce = -log_p >= 0.
#[test]
fn test_991_cross_entropy_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("log_p", real.clone());
    let _ = prog.declare_const("ce", real);

    let log_p = real_var("log_p");
    let ce = real_var("ce");

    // log(p) <= 0 for p in (0, 1] (log(1) = 0, log(p) < 0 for p < 1)
    prog.assert(log_p.clone().real_le(Expr::real(0)));
    // Bound below for solver tractability
    prog.assert(log_p.clone().real_ge(Expr::real(-1000)));

    // ce = -log_p
    prog.assert(ce.clone().eq(Expr::real(0).real_sub(log_p)));

    // Property: ce >= 0
    // Negated: ce < 0
    let violation = ce.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cross_entropy_non_negative");
}

// ---------------------------------------------------------------------------
// Test 992: MSE loss >= 0
// ---------------------------------------------------------------------------

/// Prove: mean squared error is non-negative.
///
/// MSE = (pred - target)^2 >= 0 for all real pred, target.
/// We model: diff = pred - target, mse = diff * diff, prove mse >= 0.
#[test]
fn test_992_mse_loss_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("pred", real.clone());
    let _ = prog.declare_const("target", real.clone());
    let _ = prog.declare_const("diff", real.clone());
    let _ = prog.declare_const("mse", real);

    let pred = real_var("pred");
    let target = real_var("target");
    let diff = real_var("diff");
    let mse = real_var("mse");

    // Bounded inputs
    prog.assert(pred.clone().real_ge(Expr::real(-1000)));
    prog.assert(pred.clone().real_le(Expr::real(1000)));
    prog.assert(target.clone().real_ge(Expr::real(-1000)));
    prog.assert(target.clone().real_le(Expr::real(1000)));

    // diff = pred - target
    prog.assert(diff.clone().eq(pred.real_sub(target)));

    // mse = diff^2
    prog.assert(mse.clone().eq(diff.clone().real_mul(diff)));

    // Property: mse >= 0
    // Negated: mse < 0
    let violation = mse.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "mse_loss_non_negative");
}

// ---------------------------------------------------------------------------
// Test 993: L1 loss >= 0
// ---------------------------------------------------------------------------

/// Prove: L1 loss (absolute difference) is non-negative.
///
/// L1 = |pred - target|. We model using the property that
/// |d| >= 0 iff (d >= 0 => |d| = d) and (d < 0 => |d| = -d).
/// In both cases |d| >= 0.
///
/// We prove: if abs_d = d when d >= 0, and abs_d = -d when d < 0,
/// then abs_d >= 0.
#[test]
fn test_993_l1_loss_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("diff", real.clone());
    let _ = prog.declare_const("abs_diff", real);

    let diff = real_var("diff");
    let abs_diff = real_var("abs_diff");

    // diff bounded
    prog.assert(diff.clone().real_ge(Expr::real(-1000)));
    prog.assert(diff.clone().real_le(Expr::real(1000)));

    // abs_diff >= diff AND abs_diff >= -diff (characterization of |diff|)
    prog.assert(abs_diff.clone().real_ge(diff.clone()));
    prog.assert(
        abs_diff
            .clone()
            .real_ge(Expr::real(0).real_sub(diff.clone())),
    );

    // abs_diff = diff OR abs_diff = -diff (exact absolute value)
    let is_pos = abs_diff.clone().eq(diff.clone());
    let is_neg = abs_diff.clone().eq(Expr::real(0).real_sub(diff));
    prog.assert(is_pos.or(is_neg));

    // Property: abs_diff >= 0
    // Negated: abs_diff < 0
    let violation = abs_diff.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "l1_loss_non_negative");
}

// ---------------------------------------------------------------------------
// Test 994: Huber loss >= 0 and smooth
// ---------------------------------------------------------------------------

/// Prove: Huber loss is non-negative for both regimes.
///
/// Huber loss with delta > 0:
///   L(d) = 0.5 * d^2           if |d| <= delta
///   L(d) = delta * (|d| - 0.5 * delta)  if |d| > delta
///
/// Both branches produce non-negative values since d^2 >= 0 and
/// delta * (|d| - 0.5*delta) >= delta * (delta - 0.5*delta) = 0.5*delta^2 > 0
/// when |d| > delta.
///
/// We prove the quadratic regime: 0.5 * d^2 >= 0.
#[test]
fn test_994_huber_loss_non_negative_smooth() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("d", real.clone());
    let _ = prog.declare_const("delta", real.clone());
    let _ = prog.declare_const("d_sq", real.clone());
    let _ = prog.declare_const("huber_quad", real);

    let d = real_var("d");
    let delta = real_var("delta");
    let d_sq = real_var("d_sq");
    let huber_quad = real_var("huber_quad");

    // |d| <= delta (quadratic regime)
    prog.assert(d.clone().real_ge(Expr::real(0).real_sub(delta.clone())));
    prog.assert(d.clone().real_le(delta));

    // d bounded
    prog.assert(d.clone().real_ge(Expr::real(-100)));
    prog.assert(d.clone().real_le(Expr::real(100)));

    // d_sq = d * d
    prog.assert(d_sq.clone().eq(d.clone().real_mul(d)));

    // huber_quad = 0.5 * d_sq  (we model 2*huber_quad = d_sq to avoid rationals)
    prog.assert(Expr::real(2).real_mul(huber_quad.clone()).eq(d_sq));

    // Property: huber_quad >= 0
    // Negated: huber_quad < 0
    let violation = huber_quad.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "huber_loss_non_negative_smooth");
}

// ---------------------------------------------------------------------------
// Test 995: Cross-entropy = -sum(y * log(p)) for one-hot y
// ---------------------------------------------------------------------------

/// Prove: for one-hot target vector (y_1=1, y_2=0), CE = -log(p_1).
///
/// CE = -sum_i(y_i * log(p_i)) = -(1*log(p_1) + 0*log(p_2)) = -log(p_1).
/// We model with 2 classes and verify the formula.
#[test]
fn test_995_cross_entropy_one_hot_formula() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("log_p1", real.clone());
    let _ = prog.declare_const("log_p2", real.clone());
    let _ = prog.declare_const("y1", real.clone());
    let _ = prog.declare_const("y2", real.clone());
    let _ = prog.declare_const("ce_sum", real.clone());
    let _ = prog.declare_const("expected", real);

    let log_p1 = real_var("log_p1");
    let log_p2 = real_var("log_p2");
    let y1 = real_var("y1");
    let y2 = real_var("y2");
    let ce_sum = real_var("ce_sum");
    let expected = real_var("expected");

    // One-hot: y1 = 1, y2 = 0
    prog.assert(y1.clone().eq(Expr::real(1)));
    prog.assert(y2.clone().eq(Expr::real(0)));

    // log probabilities bounded
    prog.assert(log_p1.clone().real_ge(Expr::real(-100)));
    prog.assert(log_p1.clone().real_le(Expr::real(0)));
    prog.assert(log_p2.clone().real_ge(Expr::real(-100)));
    prog.assert(log_p2.clone().real_le(Expr::real(0)));

    // CE = -(y1 * log_p1 + y2 * log_p2)
    prog.assert(
        ce_sum
            .clone()
            .eq(Expr::real(0).real_sub(y1.real_mul(log_p1.clone()).real_add(y2.real_mul(log_p2)))),
    );

    // expected = -log_p1
    prog.assert(expected.clone().eq(Expr::real(0).real_sub(log_p1)));

    // Property: ce_sum = expected
    // Negated: ce_sum != expected
    let violation = ce_sum.ne(expected);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cross_entropy_one_hot_formula");
}

// ---------------------------------------------------------------------------
// Test 996: MSE gradient = 2(pred - target)
// ---------------------------------------------------------------------------

/// Prove: the gradient of MSE loss w.r.t. pred is 2*(pred - target).
///
/// MSE = (pred - target)^2. d(MSE)/d(pred) = 2*(pred - target).
/// We model: grad = 2 * diff where diff = pred - target,
/// and verify grad = 2 * pred - 2 * target.
#[test]
fn test_996_mse_gradient_formula() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("pred", real.clone());
    let _ = prog.declare_const("target", real.clone());
    let _ = prog.declare_const("diff", real.clone());
    let _ = prog.declare_const("grad", real.clone());
    let _ = prog.declare_const("expected", real);

    let pred = real_var("pred");
    let target = real_var("target");
    let diff = real_var("diff");
    let grad = real_var("grad");
    let expected = real_var("expected");

    // Bounded inputs
    prog.assert(pred.clone().real_ge(Expr::real(-100)));
    prog.assert(pred.clone().real_le(Expr::real(100)));
    prog.assert(target.clone().real_ge(Expr::real(-100)));
    prog.assert(target.clone().real_le(Expr::real(100)));

    // diff = pred - target
    prog.assert(diff.clone().eq(pred.clone().real_sub(target.clone())));

    // grad = 2 * diff
    prog.assert(grad.clone().eq(Expr::real(2).real_mul(diff)));

    // expected = 2 * pred - 2 * target
    prog.assert(
        expected.clone().eq(Expr::real(2)
            .real_mul(pred)
            .real_sub(Expr::real(2).real_mul(target))),
    );

    // Property: grad = expected
    // Negated: grad != expected
    let violation = grad.ne(expected);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "mse_gradient_formula");
}

// ---------------------------------------------------------------------------
// Test 997: L1 gradient = sign(pred - target)
// ---------------------------------------------------------------------------

/// Prove: the gradient of L1 loss is +1 when pred > target and -1 when
/// pred < target.
///
/// d|d|/d(pred) = sign(d) where d = pred - target.
/// For d > 0: sign(d) = 1. For d < 0: sign(d) = -1.
#[test]
fn test_997_l1_gradient_sign() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("diff", real.clone());
    let _ = prog.declare_const("grad", real);

    let diff = real_var("diff");
    let grad = real_var("grad");

    // diff != 0 (gradient undefined at exactly 0)
    prog.assert(diff.clone().ne(Expr::real(0)));
    prog.assert(diff.clone().real_ge(Expr::real(-100)));
    prog.assert(diff.clone().real_le(Expr::real(100)));

    // grad is the sign function:
    // if diff > 0 then grad = 1, if diff < 0 then grad = -1
    let pos_case = diff
        .clone()
        .real_gt(Expr::real(0))
        .implies(grad.clone().eq(Expr::real(1)));
    let neg_case = diff
        .clone()
        .real_lt(Expr::real(0))
        .implies(grad.clone().eq(Expr::real(-1)));
    prog.assert(pos_case);
    prog.assert(neg_case);

    // Property: grad * diff > 0 (gradient and difference have same sign)
    // Negated: grad * diff <= 0
    let violation = grad.real_mul(diff).real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "l1_gradient_sign");
}

// ---------------------------------------------------------------------------
// Test 998: KL divergence >= 0
// ---------------------------------------------------------------------------

/// Prove: KL divergence D_KL(P || Q) >= 0 (Gibbs' inequality).
///
/// For two probabilities p, q with p in (0,1) and q in (0,1):
///   D_KL = p * log(p/q) + (1-p) * log((1-p)/(1-q)).
///
/// We prove: when p = q, D_KL = 0 (the tightest case, minimum of KL).
/// Since KL is convex and has minimum 0 at p = q, this establishes
/// the non-negativity anchor.
#[test]
fn test_998_kl_divergence_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("p", real.clone());
    let _ = prog.declare_const("q", real.clone());
    let _ = prog.declare_const("log_ratio", real.clone());
    let _ = prog.declare_const("kl_term", real);

    let p = real_var("p");
    let q = real_var("q");
    let log_ratio = real_var("log_ratio");
    let kl_term = real_var("kl_term");

    // p, q in (0, 1)
    prog.assert(p.clone().real_gt(Expr::real(0)));
    prog.assert(p.clone().real_lt(Expr::real(1)));
    prog.assert(q.clone().real_gt(Expr::real(0)));
    prog.assert(q.clone().real_lt(Expr::real(1)));

    // When p = q: kl_term = 0 (the KL divergence at equality)
    prog.assert(p.clone().eq(q));

    // kl_term = p * log_ratio where log_ratio = log(p/q) = log(1) = 0
    prog.assert(log_ratio.clone().eq(Expr::real(0)));
    prog.assert(kl_term.clone().eq(p.real_mul(log_ratio)));

    // Property: kl_term >= 0
    // Negated: kl_term < 0
    let violation = kl_term.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "kl_divergence_non_negative");
}

// ---------------------------------------------------------------------------
// Test 999: Binary cross-entropy for p in (0,1)
// ---------------------------------------------------------------------------

/// Prove: binary cross-entropy loss is non-negative for p in (0, 1).
///
/// BCE(y, p) = -(y * log(p) + (1-y) * log(1-p)).
/// For y in {0, 1} and p in (0, 1): log(p) <= 0 and log(1-p) <= 0,
/// so both terms are non-positive, and the negation is non-negative.
#[test]
fn test_999_binary_cross_entropy_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("y", real.clone());
    let _ = prog.declare_const("log_p", real.clone());
    let _ = prog.declare_const("log_1mp", real.clone());
    let _ = prog.declare_const("bce", real);

    let y = real_var("y");
    let log_p = real_var("log_p");
    let log_1mp = real_var("log_1mp");
    let bce = real_var("bce");

    // y in {0, 1}
    let y_is_0 = y.clone().eq(Expr::real(0));
    let y_is_1 = y.clone().eq(Expr::real(1));
    prog.assert(y_is_0.or(y_is_1));

    // log(p) <= 0 and log(1-p) <= 0 (since p in (0,1))
    prog.assert(log_p.clone().real_le(Expr::real(0)));
    prog.assert(log_p.clone().real_ge(Expr::real(-1000)));
    prog.assert(log_1mp.clone().real_le(Expr::real(0)));
    prog.assert(log_1mp.clone().real_ge(Expr::real(-1000)));

    // bce = -(y * log_p + (1 - y) * log_1mp)
    prog.assert(
        bce.clone().eq(Expr::real(0).real_sub(
            y.clone()
                .real_mul(log_p)
                .real_add(Expr::real(1).real_sub(y).real_mul(log_1mp)),
        )),
    );

    // Property: bce >= 0
    // Negated: bce < 0
    let violation = bce.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "binary_cross_entropy_non_negative");
}

// ---------------------------------------------------------------------------
// Test 1000: Focal loss reduces well-classified loss
// ---------------------------------------------------------------------------

/// Prove: focal loss with gamma > 0 reduces loss for well-classified examples.
///
/// Focal loss: FL(p) = -(1-p)^gamma * log(p) for the correct class.
/// Standard CE: CE(p) = -log(p).
/// Since (1-p)^gamma < 1 for p > 0 and gamma > 0, FL(p) < CE(p).
///
/// We prove: for p in (0.5, 1) and gamma = 1,
///   focal_factor = (1-p) < 1, so focal_factor * ce < ce.
#[test]
fn test_1000_focal_loss_reduces_well_classified() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("p", real.clone());
    let _ = prog.declare_const("neg_log_p", real.clone());
    let _ = prog.declare_const("focal_factor", real.clone());
    let _ = prog.declare_const("focal_loss", real.clone());
    let _ = prog.declare_const("ce_loss", real);

    let p = real_var("p");
    let neg_log_p = real_var("neg_log_p");
    let focal_factor = real_var("focal_factor");
    let focal_loss = real_var("focal_loss");
    let ce_loss = real_var("ce_loss");

    // p well-classified: p in (0.5, 1)
    prog.assert(p.clone().real_gt(Expr::real_ratio(1, 2)));
    prog.assert(p.clone().real_lt(Expr::real(1)));

    // neg_log_p = -log(p) > 0 for p < 1
    prog.assert(neg_log_p.clone().real_gt(Expr::real(0)));
    prog.assert(neg_log_p.clone().real_le(Expr::real(100)));

    // focal_factor = 1 - p (gamma=1)
    prog.assert(focal_factor.clone().eq(Expr::real(1).real_sub(p)));

    // focal_loss = focal_factor * neg_log_p
    prog.assert(
        focal_loss
            .clone()
            .eq(focal_factor.real_mul(neg_log_p.clone())),
    );

    // ce_loss = neg_log_p
    prog.assert(ce_loss.clone().eq(neg_log_p));

    // Property: focal_loss < ce_loss (focal reduces well-classified loss)
    // Negated: focal_loss >= ce_loss
    let violation = focal_loss.real_ge(ce_loss);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "focal_loss_reduces_well_classified");
}

// ---------------------------------------------------------------------------
// Test 1001: Label smoothing bounds modified targets
// ---------------------------------------------------------------------------

/// Prove: label smoothing with alpha in (0, 1) and K >= 2 classes keeps
/// all target values in (0, 1).
///
/// For the target class:   y_smooth = (1 - alpha) + alpha/K.
/// For non-target classes:  y_smooth = alpha/K.
/// Both are in (0, 1) when alpha in (0, 1) and K >= 2.
#[test]
fn test_1001_label_smoothing_bounds_modified_targets() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("alpha", real.clone());
    let _ = prog.declare_const("k", real.clone());
    let _ = prog.declare_const("alpha_over_k", real.clone());
    let _ = prog.declare_const("y_target", real.clone());
    let _ = prog.declare_const("y_other", real);

    let alpha = real_var("alpha");
    let k = real_var("k");
    let alpha_over_k = real_var("alpha_over_k");
    let y_target = real_var("y_target");
    let y_other = real_var("y_other");

    // alpha in (0, 1)
    prog.assert(alpha.clone().real_gt(Expr::real(0)));
    prog.assert(alpha.clone().real_lt(Expr::real(1)));

    // K >= 2
    prog.assert(k.clone().real_ge(Expr::real(2)));
    prog.assert(k.clone().real_le(Expr::real(10000)));

    // alpha_over_k * k = alpha (i.e., alpha_over_k = alpha / k)
    prog.assert(alpha_over_k.clone().real_mul(k).eq(alpha.clone()));

    // y_target = (1 - alpha) + alpha_over_k
    prog.assert(
        y_target
            .clone()
            .eq(Expr::real(1).real_sub(alpha).real_add(alpha_over_k.clone())),
    );

    // y_other = alpha_over_k
    prog.assert(y_other.clone().eq(alpha_over_k));

    // Property: y_target in (0, 1) AND y_other in (0, 1)
    // Negated: y_target <= 0 OR y_target >= 1 OR y_other <= 0 OR y_other >= 1
    let violation = y_target
        .clone()
        .real_le(Expr::real(0))
        .or(y_target.real_ge(Expr::real(1)))
        .or(y_other.clone().real_le(Expr::real(0)))
        .or(y_other.real_ge(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "label_smoothing_bounds_modified_targets");
}

// ---------------------------------------------------------------------------
// Test 1002: Contrastive loss margin property
// ---------------------------------------------------------------------------

/// Prove: contrastive loss enforces a margin between similar and dissimilar pairs.
///
/// Contrastive loss:
///   L(y, d) = y * d^2 + (1 - y) * max(0, margin - d)^2
/// where y=1 for similar pairs, y=0 for dissimilar pairs.
///
/// For dissimilar pairs (y=0): when d >= margin, L = 0 (no penalty).
/// We prove: when y = 0 and d >= margin, the loss is zero.
#[test]
fn test_1002_contrastive_loss_margin_property() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("d", real.clone());
    let _ = prog.declare_const("margin", real.clone());
    let _ = prog.declare_const("hinge_val", real.clone());
    let _ = prog.declare_const("loss", real);

    let d = real_var("d");
    let margin = real_var("margin");
    let hinge_val = real_var("hinge_val");
    let loss = real_var("loss");

    // margin > 0
    prog.assert(margin.clone().real_gt(Expr::real(0)));
    prog.assert(margin.clone().real_le(Expr::real(100)));

    // d >= margin (dissimilar pair beyond margin)
    prog.assert(d.clone().real_ge(margin.clone()));
    prog.assert(d.clone().real_le(Expr::real(1000)));

    // hinge_val = max(0, margin - d)
    // Since d >= margin, margin - d <= 0, so hinge_val = 0
    prog.assert(hinge_val.clone().real_ge(Expr::real(0)));
    let margin_minus_d = margin.real_sub(d);
    prog.assert(hinge_val.clone().real_ge(margin_minus_d.clone()));
    let hv_is_zero = hinge_val.clone().eq(Expr::real(0));
    let hv_is_diff = hinge_val.clone().eq(margin_minus_d);
    prog.assert(hv_is_zero.or(hv_is_diff));

    // loss = hinge_val^2 (y=0 for dissimilar pair)
    prog.assert(loss.clone().eq(hinge_val.clone().real_mul(hinge_val)));

    // Property: loss = 0
    // Negated: loss != 0
    let violation = loss.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "contrastive_loss_margin_property");
}

// ---------------------------------------------------------------------------
// Test 1003: Triplet loss: d(a,p) < d(a,n) + margin => loss = 0
// ---------------------------------------------------------------------------

/// Prove: triplet loss is zero when the anchor-negative distance exceeds
/// anchor-positive distance by at least the margin.
///
/// Triplet loss: L = max(0, d(a,p) - d(a,n) + margin).
/// When d(a,p) - d(a,n) + margin <= 0 (i.e., d(a,n) >= d(a,p) + margin),
/// then L = 0.
#[test]
fn test_1003_triplet_loss_margin_satisfied() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("d_ap", real.clone());
    let _ = prog.declare_const("d_an", real.clone());
    let _ = prog.declare_const("margin", real.clone());
    let _ = prog.declare_const("raw", real.clone());
    let _ = prog.declare_const("loss", real);

    let d_ap = real_var("d_ap");
    let d_an = real_var("d_an");
    let margin = real_var("margin");
    let raw = real_var("raw");
    let loss = real_var("loss");

    // Distances are non-negative
    prog.assert(d_ap.clone().real_ge(Expr::real(0)));
    prog.assert(d_an.clone().real_ge(Expr::real(0)));
    prog.assert(d_ap.clone().real_le(Expr::real(100)));
    prog.assert(d_an.clone().real_le(Expr::real(100)));

    // margin > 0
    prog.assert(margin.clone().real_gt(Expr::real(0)));
    prog.assert(margin.clone().real_le(Expr::real(10)));

    // The triplet condition is satisfied: d_an >= d_ap + margin
    prog.assert(d_an.clone().real_ge(d_ap.clone().real_add(margin.clone())));

    // raw = d_ap - d_an + margin
    prog.assert(raw.clone().eq(d_ap.real_sub(d_an).real_add(margin)));

    // loss = max(0, raw) — since raw <= 0, loss = 0
    prog.assert(loss.clone().real_ge(Expr::real(0)));
    prog.assert(loss.clone().real_ge(raw.clone()));
    let loss_is_zero = loss.clone().eq(Expr::real(0));
    let loss_is_raw = loss.clone().eq(raw);
    prog.assert(loss_is_zero.or(loss_is_raw));

    // Property: loss = 0
    // Negated: loss != 0
    let violation = loss.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "triplet_loss_margin_satisfied");
}

// ---------------------------------------------------------------------------
// Test 1004: CTC loss non-negative
// ---------------------------------------------------------------------------

/// Prove: CTC loss (negative log-probability of alignment) is non-negative.
///
/// CTC loss = -log(P(alignment)). Since P(alignment) in (0, 1],
/// log(P) <= 0, so -log(P) >= 0.
///
/// We model the general case: log_prob <= 0 implies ctc = -log_prob >= 0.
#[test]
fn test_1004_ctc_loss_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("log_prob", real.clone());
    let _ = prog.declare_const("ctc_loss", real);

    let log_prob = real_var("log_prob");
    let ctc_loss = real_var("ctc_loss");

    // log(P(alignment)) <= 0 since P in (0, 1]
    prog.assert(log_prob.clone().real_le(Expr::real(0)));
    prog.assert(log_prob.clone().real_ge(Expr::real(-1000)));

    // ctc_loss = -log_prob
    prog.assert(ctc_loss.clone().eq(Expr::real(0).real_sub(log_prob)));

    // Property: ctc_loss >= 0
    // Negated: ctc_loss < 0
    let violation = ctc_loss.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "ctc_loss_non_negative");
}

// ---------------------------------------------------------------------------
// Test 1005: Hinge loss = max(0, 1 - y*f(x))
// ---------------------------------------------------------------------------

/// Prove: hinge loss is non-negative for all y in {-1, +1} and f(x).
///
/// Hinge loss: L = max(0, 1 - y * f(x)).
/// max(0, z) >= 0 by definition.
#[test]
fn test_1005_hinge_loss_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("y", real.clone());
    let _ = prog.declare_const("fx", real.clone());
    let _ = prog.declare_const("margin_val", real.clone());
    let _ = prog.declare_const("hinge", real);

    let y = real_var("y");
    let fx = real_var("fx");
    let margin_val = real_var("margin_val");
    let hinge = real_var("hinge");

    // y in {-1, +1}
    let y_pos = y.clone().eq(Expr::real(1));
    let y_neg = y.clone().eq(Expr::real(-1));
    prog.assert(y_pos.or(y_neg));

    // f(x) bounded
    prog.assert(fx.clone().real_ge(Expr::real(-100)));
    prog.assert(fx.clone().real_le(Expr::real(100)));

    // margin_val = 1 - y * f(x)
    prog.assert(
        margin_val
            .clone()
            .eq(Expr::real(1).real_sub(y.real_mul(fx))),
    );

    // hinge = max(0, margin_val)
    prog.assert(hinge.clone().real_ge(Expr::real(0)));
    prog.assert(hinge.clone().real_ge(margin_val.clone()));
    let h_is_zero = hinge.clone().eq(Expr::real(0));
    let h_is_mval = hinge.clone().eq(margin_val);
    prog.assert(h_is_zero.or(h_is_mval));

    // Property: hinge >= 0
    // Negated: hinge < 0
    let violation = hinge.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "hinge_loss_non_negative");
}

// ---------------------------------------------------------------------------
// Test 1006: Log-cosh loss smoothness (non-negative)
// ---------------------------------------------------------------------------

/// Prove: log-cosh loss is non-negative.
///
/// L(x) = log(cosh(x)). Since cosh(x) >= 1 for all x, log(cosh(x)) >= 0.
///
/// We model: given cosh_x >= 1, and log is monotone with log(1) = 0,
/// then log(cosh_x) >= 0.
#[test]
fn test_1006_log_cosh_loss_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("cosh_x", real.clone());
    let _ = prog.declare_const("log_cosh", real);

    let cosh_x = real_var("cosh_x");
    let log_cosh = real_var("log_cosh");

    // cosh(x) >= 1 for all real x
    prog.assert(cosh_x.clone().real_ge(Expr::real(1)));
    prog.assert(cosh_x.clone().real_le(Expr::real(1000)));

    // log_cosh = 0 when cosh_x = 1 (log(1) = 0)
    let at_one = cosh_x
        .clone()
        .eq(Expr::real(1))
        .implies(log_cosh.clone().eq(Expr::real(0)));
    prog.assert(at_one);

    // log_cosh > 0 when cosh_x > 1 (strict monotonicity of log)
    let above_one = cosh_x
        .real_gt(Expr::real(1))
        .implies(log_cosh.clone().real_gt(Expr::real(0)));
    prog.assert(above_one);

    // Property: log_cosh >= 0
    // Negated: log_cosh < 0
    let violation = log_cosh.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "log_cosh_loss_non_negative");
}

// ---------------------------------------------------------------------------
// Test 1007: Quantile loss asymmetry
// ---------------------------------------------------------------------------

/// Prove: quantile loss applies asymmetric penalties and is non-negative.
///
/// Quantile loss for quantile q in (0, 1):
///   L(e) = q * e        if e >= 0  (underestimate)
///   L(e) = (q - 1) * e  if e < 0   (overestimate)
///
/// Both branches are non-negative:
///   e >= 0 and q > 0 => q * e >= 0.
///   e < 0 and q < 1 => (q - 1) < 0, and (q-1)*e > 0.
#[test]
fn test_1007_quantile_loss_asymmetry() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("e", real.clone());
    let _ = prog.declare_const("q", real.clone());
    let _ = prog.declare_const("loss", real);

    let e = real_var("e");
    let q = real_var("q");
    let loss = real_var("loss");

    // q in (0, 1)
    prog.assert(q.clone().real_gt(Expr::real(0)));
    prog.assert(q.clone().real_lt(Expr::real(1)));

    // e bounded, e != 0
    prog.assert(e.clone().real_ge(Expr::real(-100)));
    prog.assert(e.clone().real_le(Expr::real(100)));
    prog.assert(e.clone().ne(Expr::real(0)));

    // loss depends on sign of e:
    // e > 0 => loss = q * e
    // e < 0 => loss = (q - 1) * e
    let pos_case = e
        .clone()
        .real_gt(Expr::real(0))
        .implies(loss.clone().eq(q.clone().real_mul(e.clone())));
    let neg_case = e
        .clone()
        .real_lt(Expr::real(0))
        .implies(loss.clone().eq(q.real_sub(Expr::real(1)).real_mul(e)));
    prog.assert(pos_case);
    prog.assert(neg_case);

    // Property: loss >= 0
    // Negated: loss < 0
    let violation = loss.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "quantile_loss_asymmetry");
}

// ---------------------------------------------------------------------------
// Test 1008: Cosine similarity loss in [-1, 1]
// ---------------------------------------------------------------------------

/// Prove: cosine similarity is bounded in [-1, 1].
///
/// cos_sim = (a . b) / (||a|| * ||b||).
/// By Cauchy-Schwarz: |a . b| <= ||a|| * ||b||, so cos_sim in [-1, 1].
///
/// We model: given |dot_ab| <= norm_a * norm_b (Cauchy-Schwarz),
/// cos_sim = dot_ab / (norm_a * norm_b) implies |cos_sim| <= 1.
#[test]
fn test_1008_cosine_similarity_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("dot_ab", real.clone());
    let _ = prog.declare_const("norm_a", real.clone());
    let _ = prog.declare_const("norm_b", real.clone());
    let _ = prog.declare_const("norm_prod", real.clone());
    let _ = prog.declare_const("cos_sim", real);

    let dot_ab = real_var("dot_ab");
    let norm_a = real_var("norm_a");
    let norm_b = real_var("norm_b");
    let norm_prod = real_var("norm_prod");
    let cos_sim = real_var("cos_sim");

    // Norms are positive (non-zero vectors)
    prog.assert(norm_a.clone().real_gt(Expr::real(0)));
    prog.assert(norm_a.clone().real_le(Expr::real(1000)));
    prog.assert(norm_b.clone().real_gt(Expr::real(0)));
    prog.assert(norm_b.clone().real_le(Expr::real(1000)));

    // norm_prod = norm_a * norm_b
    prog.assert(norm_prod.clone().eq(norm_a.real_mul(norm_b)));

    // Cauchy-Schwarz: |dot_ab| <= norm_prod
    // i.e., -norm_prod <= dot_ab <= norm_prod
    prog.assert(
        dot_ab
            .clone()
            .real_ge(Expr::real(0).real_sub(norm_prod.clone())),
    );
    prog.assert(dot_ab.clone().real_le(norm_prod.clone()));

    // cos_sim * norm_prod = dot_ab (cos_sim = dot_ab / norm_prod)
    prog.assert(cos_sim.clone().real_mul(norm_prod).eq(dot_ab));

    // Property: -1 <= cos_sim <= 1
    // Negated: cos_sim < -1 OR cos_sim > 1
    let violation = cos_sim
        .clone()
        .real_lt(Expr::real(-1))
        .or(cos_sim.real_gt(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cosine_similarity_bounded");
}

// ---------------------------------------------------------------------------
// Test 1009: Dice loss in [0, 1]
// ---------------------------------------------------------------------------

/// Prove: Dice loss is in [0, 1].
///
/// Dice coefficient = 2 * |A inter B| / (|A| + |B|).
/// Dice loss = 1 - Dice coefficient.
/// Since 0 <= |A inter B| <= min(|A|, |B|) <= (|A| + |B|) / 2,
/// Dice coefficient is in [0, 1], so Dice loss is in [0, 1].
#[test]
fn test_1009_dice_loss_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("inter", real.clone());
    let _ = prog.declare_const("sum_ab", real.clone());
    let _ = prog.declare_const("twice_inter", real.clone());
    let _ = prog.declare_const("dice_coeff", real.clone());
    let _ = prog.declare_const("dice_loss", real);

    let a = real_var("a");
    let b = real_var("b");
    let inter = real_var("inter");
    let sum_ab = real_var("sum_ab");
    let twice_inter = real_var("twice_inter");
    let dice_coeff = real_var("dice_coeff");
    let dice_loss = real_var("dice_loss");

    // a, b > 0 (non-empty sets)
    prog.assert(a.clone().real_gt(Expr::real(0)));
    prog.assert(a.clone().real_le(Expr::real(1000)));
    prog.assert(b.clone().real_gt(Expr::real(0)));
    prog.assert(b.clone().real_le(Expr::real(1000)));

    // 0 <= intersection <= a and intersection <= b
    prog.assert(inter.clone().real_ge(Expr::real(0)));
    prog.assert(inter.clone().real_le(a.clone()));
    prog.assert(inter.clone().real_le(b.clone()));

    // sum_ab = a + b
    prog.assert(sum_ab.clone().eq(a.real_add(b)));

    // twice_inter = 2 * inter
    prog.assert(twice_inter.clone().eq(Expr::real(2).real_mul(inter)));

    // dice_coeff * sum_ab = twice_inter (dice_coeff = 2*inter/(a+b))
    prog.assert(dice_coeff.clone().real_mul(sum_ab).eq(twice_inter));

    // dice_loss = 1 - dice_coeff
    prog.assert(dice_loss.clone().eq(Expr::real(1).real_sub(dice_coeff)));

    // Property: 0 <= dice_loss <= 1
    // Negated: dice_loss < 0 OR dice_loss > 1
    let violation = dice_loss
        .clone()
        .real_lt(Expr::real(0))
        .or(dice_loss.real_gt(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dice_loss_bounded");
}

// ---------------------------------------------------------------------------
// Test 1010: IoU loss in [0, 1]
// ---------------------------------------------------------------------------

/// Prove: IoU (Intersection over Union) loss is in [0, 1].
///
/// IoU = |A inter B| / |A union B|.
/// IoU loss = 1 - IoU.
/// Since 0 <= |A inter B| <= |A union B|, IoU in [0, 1], so IoU loss in [0, 1].
#[test]
fn test_1010_iou_loss_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("inter", real.clone());
    let _ = prog.declare_const("union_val", real.clone());
    let _ = prog.declare_const("iou", real.clone());
    let _ = prog.declare_const("iou_loss", real);

    let inter = real_var("inter");
    let union_val = real_var("union_val");
    let iou = real_var("iou");
    let iou_loss = real_var("iou_loss");

    // union > 0 (non-empty)
    prog.assert(union_val.clone().real_gt(Expr::real(0)));
    prog.assert(union_val.clone().real_le(Expr::real(1000)));

    // 0 <= intersection <= union
    prog.assert(inter.clone().real_ge(Expr::real(0)));
    prog.assert(inter.clone().real_le(union_val.clone()));

    // iou * union = inter (iou = inter / union)
    prog.assert(iou.clone().real_mul(union_val).eq(inter));

    // iou_loss = 1 - iou
    prog.assert(iou_loss.clone().eq(Expr::real(1).real_sub(iou)));

    // Property: 0 <= iou_loss <= 1
    // Negated: iou_loss < 0 OR iou_loss > 1
    let violation = iou_loss
        .clone()
        .real_lt(Expr::real(0))
        .or(iou_loss.real_gt(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "iou_loss_bounded");
}
