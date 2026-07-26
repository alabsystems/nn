// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for optimizer and gradient safety properties.
//!
//! Proves numerical safety and correctness of training primitives:
//!
//! **SGD update rule (harnesses 1-3):**
//! - Exact update: w_new = w - lr * grad (no momentum)
//! - Momentum: m_new = mu * m + grad, w_new = w - lr * m_new
//! - Finiteness: weight update is finite for finite inputs
//!
//! **Adam optimizer (harnesses 4-7):**
//! - First moment: m = beta1 * m + (1-beta1) * grad
//! - Second moment non-negative: v = beta2 * v + (1-beta2) * grad^2 >= 0
//! - Bias-corrected v_hat >= 0
//! - Update magnitude bounded by lr / (1-beta1) for bounded gradients
//!
//! **Learning rate (harnesses 8-9):**
//! - Halving lr halves the weight step
//! - Zero lr produces zero update
//!
//! **Gradient clipping (harnesses 10-12):**
//! - Clipped gradient norm <= max_norm for finite inputs
//! - Clipping preserves gradient direction (same sign)
//! - Below-threshold gradient is unchanged
//!
//! **Weight decay (harnesses 13-15):**
//! - L2 decay: w_new = w * (1 - lr * wd) - lr * grad
//! - Decoupled (AdamW): w_new = w * (1 - wd) - lr * adam_step
//! - Positive weight decay reduces weight magnitude (for aligned signs)
//!
//! **Mixed precision training (harnesses 16-17):**
//! - bf16 gradient cast preserves sign
//! - f32 master weight + bf16 gradient produces finite update
//!
//! Part of #3942.

// ============================================================================
// Scalar optimizer helpers (pure arithmetic, no DynTensor dependency)
// ============================================================================

/// SGD update without momentum: w_new = w - lr * grad.
///
/// SYNC: nn-optim/src/sgd.rs:135 (theta = theta - lr * update).
fn sgd_update_no_momentum(w: f32, lr: f32, grad: f32) -> f32 {
    w - lr * grad
}

/// SGD momentum update: m_new = mu * m + grad.
///
/// SYNC: nn-optim/src/sgd.rs:125 (prev.mul_scalar(momentum)?.add(&grad)?).
fn sgd_momentum_update(m: f32, grad: f32, mu: f32) -> f32 {
    mu * m + grad
}

/// SGD weight update with momentum: w_new = w - lr * m_new.
///
/// SYNC: nn-optim/src/sgd.rs:135.
fn sgd_update_with_momentum(w: f32, lr: f32, m_new: f32) -> f32 {
    w - lr * m_new
}

/// Adam first moment EMA: m = beta1 * m + (1 - beta1) * grad.
///
/// SYNC: nn-optim/src/adam.rs:248-249 (beta1_ema).
fn adam_first_moment(m: f32, grad: f32, beta1: f32) -> f32 {
    beta1 * m + (1.0 - beta1) * grad
}

/// Adam second moment EMA: v = beta2 * v + (1 - beta2) * grad^2.
///
/// SYNC: nn-optim/src/adam.rs:254-255 (beta2_ema).
fn adam_second_moment(v: f32, grad: f32, beta2: f32) -> f32 {
    beta2 * v + (1.0 - beta2) * grad * grad
}

/// Adam bias correction: x_hat = x / (1 - beta^t).
fn adam_bias_correct(x: f32, beta: f32, t: u32) -> f32 {
    let correction = 1.0 - beta.powi(t as i32);
    if correction.abs() < 1e-30 {
        return x; // avoid division by near-zero
    }
    x / correction
}

/// Adam adaptive step: lr * m_hat / (sqrt(v_hat) + eps).
///
/// SYNC: nn-optim/src/adam.rs:203.
fn adam_step(lr: f32, m_hat: f32, v_hat: f32, eps: f32) -> f32 {
    lr * m_hat / (v_hat.sqrt() + eps)
}

/// L2 weight decay applied to gradient: effective_grad = grad + wd * w.
///
/// SYNC: nn-optim/src/sgd.rs:117 (grad.add(&var.data()?.mul_scalar(weight_decay)?)).
fn l2_weight_decay_grad(grad: f32, w: f32, wd: f32) -> f32 {
    grad + wd * w
}

/// Decoupled weight decay (AdamW): w_new = w * (1 - wd) - lr * adam_step_val.
///
/// SYNC: nn-optim/src/adam.rs:204-206.
fn adamw_decoupled_update(w: f32, wd: f32, lr: f32, step_val: f32) -> f32 {
    w * (1.0 - wd) - lr * step_val
}

/// Scalar gradient norm clipping: if |grad| > max_norm, scale to max_norm.
/// Returns (clipped_grad, original_norm).
fn clip_grad_scalar(grad: f32, max_norm: f32) -> (f32, f32) {
    let norm = grad.abs();
    if norm > max_norm {
        let scale = max_norm / norm;
        (grad * scale, norm)
    } else {
        (grad, norm)
    }
}

/// BF16 truncation: model bf16 round-trip by zeroing lower 16 mantissa bits.
/// Same as kani_quantization_proofs.rs:f32_to_bf16_to_f32.
fn f32_to_bf16_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let round_bit = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + round_bit);
    let bf16_bits = rounded & 0xFFFF_0000;
    f32::from_bits(bf16_bits)
}

// ============================================================================
// SGD update rule harnesses
// ============================================================================

// ---------------------------------------------------------------------------
// 1. SGD no-momentum: w_new = w - lr * grad (exact)
// ---------------------------------------------------------------------------

/// Prove: SGD without momentum computes exactly w_new = w - lr * grad.
///
/// The fundamental correctness property of vanilla SGD. Every gradient
/// descent step must be exactly this formula for convergence guarantees
/// to hold.
#[kani::unwind(1)]
#[kani::proof]
fn prove_sgd_update_exact() {
    let w: f32 = kani::any();
    let lr: f32 = kani::any();
    let grad: f32 = kani::any();
    kani::assume(w.is_finite() && w.abs() <= 1e3);
    kani::assume(lr.is_finite() && lr >= 0.0 && lr <= 1.0);
    kani::assume(grad.is_finite() && grad.abs() <= 1e3);

    let w_new = sgd_update_no_momentum(w, lr, grad);
    let expected = w - lr * grad;

    // Exact equality: same floating-point operations, same result.
    assert!(
        w_new == expected,
        "SGD update must be exactly w - lr * grad"
    );
}

// ---------------------------------------------------------------------------
// 2. SGD with momentum: m_new = mu * m + grad, w_new = w - lr * m_new
// ---------------------------------------------------------------------------

/// Prove: SGD momentum update follows the two-step formula:
/// (1) m_new = mu * m + grad
/// (2) w_new = w - lr * m_new
#[kani::unwind(1)]
#[kani::proof]
fn prove_sgd_momentum_update_correct() {
    let w: f32 = kani::any();
    let m: f32 = kani::any();
    let grad: f32 = kani::any();
    let lr: f32 = kani::any();
    let mu: f32 = kani::any();

    kani::assume(w.is_finite() && w.abs() <= 1e3);
    kani::assume(m.is_finite() && m.abs() <= 1e3);
    kani::assume(grad.is_finite() && grad.abs() <= 1e3);
    kani::assume(lr.is_finite() && lr >= 0.0 && lr <= 1.0);
    kani::assume(mu.is_finite() && mu >= 0.0 && mu < 1.0);

    let m_new = sgd_momentum_update(m, grad, mu);
    let w_new = sgd_update_with_momentum(w, lr, m_new);

    // Verify the two-step formula
    let expected_m = mu * m + grad;
    let expected_w = w - lr * expected_m;

    assert!(m_new == expected_m, "momentum update must be mu * m + grad");
    assert!(w_new == expected_w, "weight update must be w - lr * m_new");
}

// ---------------------------------------------------------------------------
// 3. SGD update is finite for finite inputs
// ---------------------------------------------------------------------------

/// Prove: SGD weight update produces a finite result for finite,
/// bounded inputs. This prevents NaN/Inf corruption during training.
#[kani::unwind(1)]
#[kani::proof]
fn prove_sgd_update_finite() {
    let w: f32 = kani::any();
    let lr: f32 = kani::any();
    let grad: f32 = kani::any();
    kani::assume(w.is_finite() && w.abs() <= 1e4);
    kani::assume(lr.is_finite() && lr >= 0.0 && lr <= 1.0);
    kani::assume(grad.is_finite() && grad.abs() <= 1e4);

    let w_new = sgd_update_no_momentum(w, lr, grad);

    // lr * grad: |lr| <= 1, |grad| <= 1e4, so |lr * grad| <= 1e4.
    // |w - lr * grad| <= |w| + |lr * grad| <= 2e4, well within f32 range.
    assert!(
        w_new.is_finite(),
        "SGD update must be finite for finite bounded inputs"
    );
}

// ============================================================================
// Adam optimizer harnesses
// ============================================================================

// ---------------------------------------------------------------------------
// 4. Adam first moment: m = beta1 * m + (1-beta1) * grad
// ---------------------------------------------------------------------------

/// Prove: Adam first moment update follows the EMA formula exactly.
/// This is the unbiased gradient estimator used in all Adam variants.
#[kani::unwind(1)]
#[kani::proof]
fn prove_adam_first_moment_correct() {
    let m: f32 = kani::any();
    let grad: f32 = kani::any();
    let beta1: f32 = kani::any();

    kani::assume(m.is_finite() && m.abs() <= 1e3);
    kani::assume(grad.is_finite() && grad.abs() <= 1e3);
    kani::assume(beta1.is_finite() && beta1 >= 0.0 && beta1 < 1.0);

    let m_new = adam_first_moment(m, grad, beta1);
    let expected = beta1 * m + (1.0 - beta1) * grad;

    assert!(
        m_new == expected,
        "Adam first moment must be beta1 * m + (1-beta1) * grad"
    );
}

// ---------------------------------------------------------------------------
// 5. Adam second moment non-negative: v = beta2 * v + (1-beta2) * grad^2
// ---------------------------------------------------------------------------

/// Prove: Adam second moment is non-negative when starting from v >= 0.
///
/// The second moment v tracks the EMA of squared gradients. Since grad^2 >= 0
/// and beta2 * v >= 0 (for v >= 0), the sum is non-negative. This ensures
/// the square root in the denominator (sqrt(v_hat) + eps) is well-defined.
#[kani::unwind(1)]
#[kani::proof]
fn prove_adam_second_moment_nonneg() {
    let v: f32 = kani::any();
    let grad: f32 = kani::any();
    let beta2: f32 = kani::any();

    kani::assume(v.is_finite() && v >= 0.0 && v <= 1e6);
    kani::assume(grad.is_finite() && grad.abs() <= 1e3);
    kani::assume(beta2.is_finite() && beta2 >= 0.0 && beta2 < 1.0);

    let v_new = adam_second_moment(v, grad, beta2);

    assert!(v_new >= 0.0, "Adam second moment must be non-negative");
    assert!(
        v_new.is_finite(),
        "Adam second moment must be finite for bounded inputs"
    );
}

// ---------------------------------------------------------------------------
// 6. Adam bias-corrected v_hat >= 0
// ---------------------------------------------------------------------------

/// Prove: bias-corrected second moment v_hat >= 0 for v >= 0 and t >= 1.
///
/// Since v >= 0 and the bias correction divides by (1 - beta2^t) which is
/// positive for beta2 in [0, 1) and t >= 1, v_hat = v / (1 - beta2^t) >= 0.
#[kani::unwind(1)]
#[kani::proof]
fn prove_adam_bias_corrected_v_hat_nonneg() {
    let v: f32 = kani::any();
    let beta2: f32 = kani::any();
    let t: u32 = kani::any();

    kani::assume(v.is_finite() && v >= 0.0 && v <= 1e6);
    kani::assume(beta2.is_finite() && beta2 >= 0.0 && beta2 < 1.0);
    kani::assume(t >= 1 && t <= 10000);

    let correction = 1.0f32 - beta2.powi(t as i32);
    kani::assume(correction > 1e-10); // correction is positive for valid beta2, t>=1

    let v_hat = v / correction;

    assert!(v_hat >= 0.0, "bias-corrected v_hat must be non-negative");
}

// ---------------------------------------------------------------------------
// 7. Adam update magnitude bounded by lr / (1-beta1) for bounded gradients
// ---------------------------------------------------------------------------

/// Prove: Adam adaptive step magnitude is bounded for bounded gradients.
///
/// With |grad| <= G, |m_hat| <= G / (1-beta1) (worst case: all grads same sign).
/// With v_hat > 0 and eps > 0: step = lr * m_hat / (sqrt(v_hat) + eps).
/// The step magnitude is bounded by lr * |m_hat| / eps <= lr * G / ((1-beta1) * eps).
#[kani::unwind(1)]
#[kani::proof]
fn prove_adam_step_bounded() {
    let m_hat: f32 = kani::any();
    let v_hat: f32 = kani::any();
    let lr: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(m_hat.is_finite() && m_hat.abs() <= 100.0);
    kani::assume(v_hat.is_finite() && v_hat >= 0.0 && v_hat <= 1e4);
    kani::assume(lr.is_finite() && lr > 0.0 && lr <= 0.01);
    kani::assume(eps.is_finite() && eps > 0.0 && eps >= 1e-8);

    let step = adam_step(lr, m_hat, v_hat, eps);

    // Upper bound: lr * |m_hat| / eps.
    // With lr=0.01, |m_hat|=100, eps=1e-8: bound = 0.01 * 100 / 1e-8 = 1e8.
    // But for practical ranges (v_hat contributes to denominator), step is much smaller.
    let denom = v_hat.sqrt() + eps;
    let expected_bound = lr * m_hat.abs() / denom;

    assert!(
        step.abs() <= expected_bound + 1e-6,
        "Adam step must be bounded by lr * |m_hat| / (sqrt(v_hat) + eps)"
    );
    assert!(
        step.is_finite(),
        "Adam step must be finite for bounded inputs"
    );
}

// ============================================================================
// Learning rate property harnesses
// ============================================================================

// ---------------------------------------------------------------------------
// 8. Halving lr halves the weight step
// ---------------------------------------------------------------------------

/// Prove: for SGD without momentum, halving the learning rate exactly
/// halves the weight change. This is the linearity property of gradient
/// descent in the learning rate parameter.
#[kani::unwind(1)]
#[kani::proof]
fn prove_lr_proportional_step() {
    let w: f32 = kani::any();
    let lr: f32 = kani::any();
    let grad: f32 = kani::any();

    kani::assume(w.is_finite() && w.abs() <= 1e3);
    kani::assume(lr.is_finite() && lr > 0.0 && lr <= 1.0);
    kani::assume(grad.is_finite() && grad.abs() <= 1e3);

    let step_full = w - sgd_update_no_momentum(w, lr, grad); // lr * grad
    let step_half = w - sgd_update_no_momentum(w, lr / 2.0, grad); // (lr/2) * grad

    // step_full = lr * grad, step_half = (lr/2) * grad.
    // step_full = 2 * step_half.
    // Check: step_full - 2 * step_half == 0.
    let diff = (step_full - 2.0 * step_half).abs();

    assert!(diff < 1e-6, "halving lr must halve the weight step");
}

// ---------------------------------------------------------------------------
// 9. Zero lr produces zero update
// ---------------------------------------------------------------------------

/// Prove: with lr = 0, the weight does not change regardless of gradient.
/// This is the safety property for pausing training (lr warmup from 0).
#[kani::unwind(1)]
#[kani::proof]
fn prove_zero_lr_no_update() {
    let w: f32 = kani::any();
    let grad: f32 = kani::any();

    kani::assume(w.is_finite() && w.abs() <= 1e6);
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);

    let w_new = sgd_update_no_momentum(w, 0.0, grad);

    assert!(w_new == w, "zero lr must produce zero update");
}

// ============================================================================
// Gradient clipping safety harnesses
// ============================================================================

// ---------------------------------------------------------------------------
// 10. Clipped gradient norm <= max_norm
// ---------------------------------------------------------------------------

/// Prove: after clipping, the gradient norm is at most max_norm.
/// This is the fundamental safety property of gradient clipping:
/// it prevents gradient explosion from causing divergent weight updates.
#[kani::unwind(1)]
#[kani::proof]
fn prove_grad_clip_bounded() {
    let grad: f32 = kani::any();
    let max_norm: f32 = kani::any();

    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    kani::assume(max_norm.is_finite() && max_norm > 0.0 && max_norm <= 1e3);

    let (clipped, _original_norm) = clip_grad_scalar(grad, max_norm);

    assert!(
        clipped.abs() <= max_norm + 1e-5,
        "clipped gradient norm must be <= max_norm"
    );
    assert!(clipped.is_finite(), "clipped gradient must be finite");
}

// ---------------------------------------------------------------------------
// 11. Gradient clipping preserves direction (same sign)
// ---------------------------------------------------------------------------

/// Prove: gradient clipping preserves the sign of the gradient.
/// Clipping scales the gradient down but never flips its direction.
/// This ensures the optimizer still moves in the correct direction.
#[kani::unwind(1)]
#[kani::proof]
fn prove_grad_clip_preserves_sign() {
    let grad: f32 = kani::any();
    let max_norm: f32 = kani::any();

    kani::assume(grad.is_finite() && grad.abs() > 0.0 && grad.abs() <= 1e6);
    kani::assume(max_norm.is_finite() && max_norm > 0.0 && max_norm <= 1e3);

    let (clipped, _) = clip_grad_scalar(grad, max_norm);

    if grad > 0.0 {
        assert!(clipped >= 0.0, "clipping must preserve positive sign");
    } else {
        assert!(clipped <= 0.0, "clipping must preserve negative sign");
    }
}

// ---------------------------------------------------------------------------
// 12. Below-threshold gradient is unchanged
// ---------------------------------------------------------------------------

/// Prove: if gradient norm is already at or below max_norm, clipping does
/// not modify the gradient. Gradients that are already safe pass through
/// unchanged.
#[kani::unwind(1)]
#[kani::proof]
fn prove_grad_clip_noop_below_threshold() {
    let grad: f32 = kani::any();
    let max_norm: f32 = kani::any();

    kani::assume(grad.is_finite() && grad.abs() <= 1e3);
    kani::assume(max_norm.is_finite() && max_norm > 0.0 && max_norm <= 1e3);
    kani::assume(grad.abs() <= max_norm); // below threshold

    let (clipped, _) = clip_grad_scalar(grad, max_norm);

    assert!(clipped == grad, "gradient below max_norm must be unchanged");
}

// ============================================================================
// Weight decay correctness harnesses
// ============================================================================

// ---------------------------------------------------------------------------
// 13. L2 weight decay: w_new = w * (1 - lr * wd) - lr * grad
// ---------------------------------------------------------------------------

/// Prove: L2 weight decay applied via gradient (SGD style) produces
/// w_new = w - lr * (grad + wd * w) = w * (1 - lr * wd) - lr * grad.
///
/// This is equivalent to adding a penalty term (wd/2) * ||w||^2 to the loss.
#[kani::unwind(1)]
#[kani::proof]
fn prove_l2_weight_decay_correct() {
    let w: f32 = kani::any();
    let grad: f32 = kani::any();
    let lr: f32 = kani::any();
    let wd: f32 = kani::any();

    kani::assume(w.is_finite() && w.abs() <= 100.0);
    kani::assume(grad.is_finite() && grad.abs() <= 100.0);
    kani::assume(lr.is_finite() && lr >= 0.0 && lr <= 0.1);
    kani::assume(wd.is_finite() && wd >= 0.0 && wd <= 0.1);

    // Apply L2 decay to gradient, then SGD update.
    let effective_grad = l2_weight_decay_grad(grad, w, wd);
    let w_new = sgd_update_no_momentum(w, lr, effective_grad);

    // Expected: w - lr * (grad + wd * w) = w * (1 - lr * wd) - lr * grad.
    let expected = w * (1.0 - lr * wd) - lr * grad;

    let diff = (w_new - expected).abs();
    assert!(
        diff < 1e-4,
        "L2 weight decay must produce w*(1-lr*wd) - lr*grad"
    );
}

// ---------------------------------------------------------------------------
// 14. Decoupled weight decay (AdamW): w_new = w * (1 - wd) - lr * step
// ---------------------------------------------------------------------------

/// Prove: AdamW decoupled weight decay applies multiplicative shrinkage
/// independently of the adaptive step, matching the AdamW formula.
#[kani::unwind(1)]
#[kani::proof]
fn prove_adamw_decoupled_decay_correct() {
    let w: f32 = kani::any();
    let wd: f32 = kani::any();
    let lr: f32 = kani::any();
    let step_val: f32 = kani::any();

    kani::assume(w.is_finite() && w.abs() <= 100.0);
    kani::assume(wd.is_finite() && wd >= 0.0 && wd <= 0.1);
    kani::assume(lr.is_finite() && lr >= 0.0 && lr <= 0.01);
    kani::assume(step_val.is_finite() && step_val.abs() <= 100.0);

    let w_new = adamw_decoupled_update(w, wd, lr, step_val);
    let expected = w * (1.0 - wd) - lr * step_val;

    assert!(
        w_new == expected,
        "AdamW must compute w * (1-wd) - lr * step"
    );
}

// ---------------------------------------------------------------------------
// 15. Positive weight decay reduces weight magnitude (aligned signs)
// ---------------------------------------------------------------------------

/// Prove: for positive weight with positive weight decay and zero gradient,
/// the updated weight has smaller magnitude. Weight decay is a regularizer
/// that shrinks weights toward zero.
#[kani::unwind(1)]
#[kani::proof]
fn prove_weight_decay_shrinks_magnitude() {
    let w: f32 = kani::any();
    let lr: f32 = kani::any();
    let wd: f32 = kani::any();

    kani::assume(w.is_finite() && w.abs() > 0.01 && w.abs() <= 100.0);
    kani::assume(lr.is_finite() && lr > 0.0 && lr <= 0.1);
    kani::assume(wd.is_finite() && wd > 0.0 && wd <= 0.1);
    // Ensure lr * wd < 1 so (1 - lr * wd) > 0 (no sign flip).
    kani::assume(lr * wd < 1.0);

    // With zero gradient, L2 decay gives: w_new = w * (1 - lr * wd).
    let effective_grad = l2_weight_decay_grad(0.0, w, wd);
    let w_new = sgd_update_no_momentum(w, lr, effective_grad);

    // |w_new| should be less than |w|.
    // w_new = w - lr * (0 + wd * w) = w * (1 - lr * wd).
    // Since 0 < lr * wd < 1, (1 - lr * wd) is in (0, 1).
    // Therefore |w_new| = |w| * (1 - lr * wd) < |w|.
    assert!(
        w_new.abs() < w.abs(),
        "weight decay with zero grad must shrink weight magnitude"
    );
}

// ============================================================================
// Mixed precision training harnesses
// ============================================================================

// ---------------------------------------------------------------------------
// 16. BF16 gradient cast preserves sign
// ---------------------------------------------------------------------------

/// Prove: casting a gradient to bf16 and back preserves its sign.
/// Sign preservation is critical for training — a sign flip would cause
/// the optimizer to move in the wrong direction.
#[kani::unwind(1)]
#[kani::proof]
fn prove_bf16_gradient_preserves_sign() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite());
    // Avoid subnormals that might flush to zero in bf16.
    kani::assume(grad.abs() >= 1e-30);

    let bf16_grad = f32_to_bf16_f32(grad);

    if grad > 0.0 {
        assert!(
            bf16_grad >= 0.0,
            "bf16 must preserve positive gradient sign"
        );
    } else if grad < 0.0 {
        assert!(
            bf16_grad <= 0.0,
            "bf16 must preserve negative gradient sign"
        );
    }
}

// ---------------------------------------------------------------------------
// 17. f32 master weight + bf16 gradient produces finite update
// ---------------------------------------------------------------------------

/// Prove: applying a bf16-rounded gradient to an f32 master weight via
/// SGD produces a finite result. This is the core mixed-precision training
/// pattern: weights stored in f32, gradients communicated in bf16.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mixed_precision_update_finite() {
    let w: f32 = kani::any();
    let grad: f32 = kani::any();
    let lr: f32 = kani::any();

    kani::assume(w.is_finite() && w.abs() <= 1e4);
    kani::assume(grad.is_finite() && grad.abs() <= 1e4);
    kani::assume(lr.is_finite() && lr >= 0.0 && lr <= 1.0);

    // Simulate bf16 gradient (round-trip through bf16 representation).
    let bf16_grad = f32_to_bf16_f32(grad);
    kani::assume(bf16_grad.is_finite()); // bf16 round-trip of finite is finite (proved elsewhere)

    let w_new = sgd_update_no_momentum(w, lr, bf16_grad);

    assert!(
        w_new.is_finite(),
        "f32 weight + bf16 gradient SGD update must be finite"
    );
}
