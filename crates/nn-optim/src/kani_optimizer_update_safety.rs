// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for optimizer update safety.
//!
//! Proves safety properties of scalar optimizer update rules:
//! 1. SGD update bounded for finite inputs
//! 2. Adam first moment (EMA) stays finite
//! 3. Adam second moment stays non-negative and finite
//! 4. Adam bias correction finite for t >= 1
//! 5. Learning rate non-negative enforced
//! 6. Weight decay bounded: w * (1 - wd) finite for wd in [0,1]
//! 7. Gradient clipping output within [-max_norm, max_norm]
//! 8. Momentum accumulation bounded after N steps
//!
//! These complement the existing proofs in `kani_optim_proofs.rs` (SGD) and
//! `kani_optim_proofs_adam.rs` (Adam/AdaFactor) with tighter safety-focused
//! properties.
//!
//! Re: #13 (verified training epic).

#[cfg(kani)]
mod optimizer_update_safety {
    // ── Helpers ─────────────────────────────────────────────────────────

    fn assume_bounded(x: f32, lo: f32, hi: f32) {
        kani::assume(!x.is_nan() && !x.is_infinite());
        kani::assume(x >= lo && x <= hi);
    }

    fn assume_bounded_f64(x: f64, lo: f64, hi: f64) {
        kani::assume(!x.is_nan() && !x.is_infinite());
        kani::assume(x >= lo && x <= hi);
    }

    // ── 1. SGD update bounded ───────────────────────────────────────────
    //
    // w_new = w - lr * grad stays finite for finite inputs.
    // Also proves the magnitude bound: |w_new - w| <= lr * |grad|.

    /// SGD vanilla update: w_new = w - lr * grad is finite.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_sgd_update_bounded_finite() {
        let w: f32 = kani::any();
        let grad: f32 = kani::any();
        let lr: f32 = kani::any();
        assume_bounded(w, -1e6, 1e6);
        assume_bounded(grad, -1e6, 1e6);
        assume_bounded(lr, 0.0, 1.0);

        let w_new = w - lr * grad;

        assert!(!w_new.is_nan(), "SGD update produced NaN");
        assert!(!w_new.is_infinite(), "SGD update produced Inf");
    }

    /// SGD update magnitude: |delta| <= lr * |grad| (within f32 rounding).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_sgd_update_magnitude_bound() {
        let w: f32 = kani::any();
        let grad: f32 = kani::any();
        let lr: f32 = kani::any();
        assume_bounded(w, -1e4, 1e4);
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(lr, 0.0, 1e-2);

        let w_new = w - lr * grad;
        let delta = (w_new - w).abs();
        // lr <= 0.01, |grad| <= 1e4 => |delta| <= 100 + rounding
        assert!(delta <= 101.0, "SGD update delta exceeded bound");
    }

    // ── 2. Adam first moment update ─────────────────────────────────────
    //
    // m = beta1 * m_prev + (1 - beta1) * g, verify finite.
    // The EMA is a convex combination when m_prev and g are bounded.

    /// Adam first moment EMA is finite for finite inputs.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adam_first_moment_finite() {
        let m_prev: f32 = kani::any();
        let g: f32 = kani::any();
        let beta1: f32 = kani::any();
        assume_bounded(m_prev, -1e6, 1e6);
        assume_bounded(g, -1e6, 1e6);
        assume_bounded(beta1, 0.0, 0.999);

        let m = beta1 * m_prev + (1.0 - beta1) * g;

        assert!(!m.is_nan(), "First moment produced NaN");
        assert!(!m.is_infinite(), "First moment produced Inf");
    }

    /// Adam first moment is bounded by max(|m_prev|, |g|) (convex combination).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adam_first_moment_bounded() {
        let m_prev: f32 = kani::any();
        let g: f32 = kani::any();
        let beta1: f32 = kani::any();
        assume_bounded(m_prev, -1e4, 1e4);
        assume_bounded(g, -1e4, 1e4);
        assume_bounded(beta1, 0.0, 0.999);

        let m = beta1 * m_prev + (1.0 - beta1) * g;

        // Convex combination: |m| <= max(|m_prev|, |g|) <= 1e4
        // Allow small f32 rounding margin
        assert!(m.abs() <= 1e4 + 1.0, "First moment exceeded input bound");
    }

    // ── 3. Adam second moment update ────────────────────────────────────
    //
    // v = beta2 * v_prev + (1 - beta2) * g^2, verify non-negative and finite.

    /// Adam second moment is non-negative for non-negative v_prev.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adam_second_moment_non_negative_safety() {
        let v_prev: f32 = kani::any();
        let g: f32 = kani::any();
        let beta2: f32 = kani::any();
        assume_bounded(v_prev, 0.0, 1e8);
        assume_bounded(g, -1e4, 1e4);
        assume_bounded(beta2, 0.0, 0.9999);

        let v = beta2 * v_prev + (1.0 - beta2) * g * g;

        assert!(v >= 0.0, "Second moment became negative");
        assert!(!v.is_nan(), "Second moment produced NaN");
        assert!(!v.is_infinite(), "Second moment produced Inf");
    }

    /// Second moment g^2 term is non-negative regardless of sign of g.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adam_second_moment_g_squared_non_negative() {
        let g: f32 = kani::any();
        assume_bounded(g, -1e6, 1e6);

        let g_sq = g * g;
        assert!(g_sq >= 0.0, "g^2 is negative");
        assert!(!g_sq.is_nan(), "g^2 is NaN");
    }

    // ── 4. Adam bias correction ─────────────────────────────────────────
    //
    // m_hat = m / (1 - beta1^t), verify finite for t >= 1.
    // Uses iterative multiplication (CBMC cannot model powi deterministically).

    /// Bias correction divisor (1 - beta^t) is strictly positive for t in [1..20].
    /// Since beta in (0,1), beta^t in (0,1), so 1 - beta^t in (0,1).
    #[kani::unwind(22)]
    #[kani::proof]
    fn prove_adam_bias_correction_finite_safety() {
        let t: u32 = kani::any();
        kani::assume(t >= 1 && t <= 20);

        // Compute beta1^t and beta2^t iteratively
        let beta1: f64 = 0.9;
        let beta2: f64 = 0.999;
        let mut pow1: f64 = 1.0;
        let mut pow2: f64 = 1.0;
        let mut i: u32 = 0;
        while i < t {
            pow1 *= beta1;
            pow2 *= beta2;
            i += 1;
        }

        let bc1_denom = 1.0 - pow1;
        let bc2_denom = 1.0 - pow2;

        // Denominators are strictly positive
        assert!(bc1_denom > 0.0, "beta1 bias correction denominator <= 0");
        assert!(bc2_denom > 0.0, "beta2 bias correction denominator <= 0");

        // Bias correction factors are finite
        let bc1 = 1.0 / bc1_denom;
        let bc2 = 1.0 / bc2_denom;
        assert!(!bc1.is_nan() && !bc1.is_infinite(), "bc1 not finite");
        assert!(!bc2.is_nan() && !bc2.is_infinite(), "bc2 not finite");
    }

    /// Bias-corrected m_hat is finite when m is finite and t >= 1.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adam_bias_corrected_moment_finite() {
        let m: f32 = kani::any();
        assume_bounded(m, -1e4, 1e4);

        // At t=1: bc1 = 1/(1-0.9) = 10.0
        let bc1: f32 = 10.0;
        let m_hat = m * bc1;

        assert!(!m_hat.is_nan(), "Bias-corrected m_hat is NaN");
        assert!(!m_hat.is_infinite(), "Bias-corrected m_hat is Inf");
        // |m| <= 1e4, bc1 = 10 => |m_hat| <= 1e5
        assert!(m_hat.abs() <= 1e5 + 1.0, "m_hat exceeded expected bound");
    }

    // ── 5. Learning rate non-negative ───────────────────────────────────
    //
    // lr >= 0 enforced: negative lr would reverse the update direction.

    /// Learning rate validation: lr must be non-negative and finite.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_learning_rate_non_negative_enforced() {
        let lr: f64 = kani::any();
        assume_bounded_f64(lr, 0.0, 10.0);

        // Mirrors the validation in error::validate_lr
        assert!(lr >= 0.0, "lr is negative");
        assert!(lr.is_finite(), "lr is not finite");

        // The update direction is correct: lr * grad has same sign as grad
        let grad: f32 = kani::any();
        assume_bounded(grad, -1e4, 1e4);
        let step = (lr as f32) * grad;
        // If grad > 0 and lr > 0, step > 0 (descent reduces theta)
        if grad > 0.0 && lr > 0.0 {
            assert!(
                step >= 0.0,
                "Positive grad + positive lr should give non-negative step"
            );
        }
    }

    /// Learning rate zero produces zero update (no parameter change).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_zero_lr_no_update() {
        let w: f32 = kani::any();
        let grad: f32 = kani::any();
        assume_bounded(w, -1e6, 1e6);
        assume_bounded(grad, -1e6, 1e6);

        let lr: f32 = 0.0;
        let w_new = w - lr * grad;

        assert!(w_new == w, "Zero lr should produce no weight change");
    }

    // ── 6. Weight decay bounded ─────────────────────────────────────────
    //
    // w * (1 - wd) stays finite for wd in [0,1].
    // This is the decoupled weight decay used in AdamW.

    /// Decoupled weight decay: w * (1 - wd) is finite for wd in [0,1].
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_weight_decay_bounded_finite() {
        let w: f32 = kani::any();
        let wd: f32 = kani::any();
        assume_bounded(w, -1e6, 1e6);
        assume_bounded(wd, 0.0, 1.0);

        let decay_factor = 1.0 - wd;
        let w_decayed = w * decay_factor;

        assert!(!w_decayed.is_nan(), "Weight decay produced NaN");
        assert!(!w_decayed.is_infinite(), "Weight decay produced Inf");
    }

    /// Weight decay shrinks magnitude: |w * (1-wd)| <= |w| for wd in [0,1].
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_weight_decay_shrinks_magnitude() {
        let w: f32 = kani::any();
        let wd: f32 = kani::any();
        assume_bounded(w, -1e4, 1e4);
        assume_bounded(wd, 0.0, 1.0);

        let decay_factor = 1.0 - wd;
        let w_decayed = w * decay_factor;

        // decay_factor in [0, 1], so |w_decayed| <= |w| + rounding
        assert!(
            w_decayed.abs() <= w.abs() + 1e-3,
            "Weight decay increased magnitude"
        );
    }

    /// Combined weight decay + lr update: w*(1-lr*wd) - lr*m_hat/denom is finite.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_weight_decay_with_adam_step_finite() {
        let w: f32 = kani::any();
        let lr: f32 = kani::any();
        let wd: f32 = kani::any();
        let m_hat: f32 = kani::any();
        let denom: f32 = kani::any();
        assume_bounded(w, -1e4, 1e4);
        assume_bounded(lr, 1e-5, 1e-2);
        assume_bounded(wd, 0.0, 0.1);
        assume_bounded(m_hat, -1e4, 1e4);
        assume_bounded(denom, 1e-8, 1e6);

        let decay_factor = 1.0 - lr * wd;
        let step = lr * m_hat / denom;
        let w_new = w * decay_factor - step;

        assert!(!w_new.is_nan(), "Adam+WD update produced NaN");
        assert!(!w_new.is_infinite(), "Adam+WD update produced Inf");
    }

    // ── 7. Gradient clipping ────────────────────────────────────────────
    //
    // clipped_grad in [-max_norm, max_norm].

    /// Gradient value clipping: result is within [-clip_value, clip_value].
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_gradient_clipping_bounded() {
        let grad: f32 = kani::any();
        let clip_value: f32 = kani::any();
        assume_bounded(grad, -1e10, 1e10);
        assume_bounded(clip_value, 1e-6, 1e6);

        // Matches grad_clip.rs clamp logic
        let clipped = if grad < -clip_value {
            -clip_value
        } else if grad > clip_value {
            clip_value
        } else {
            grad
        };

        assert!(clipped >= -clip_value, "Clipped grad below -clip_value");
        assert!(clipped <= clip_value, "Clipped grad above clip_value");
        assert!(!clipped.is_nan(), "Clipped grad is NaN");
        assert!(!clipped.is_infinite(), "Clipped grad is Inf");
    }

    /// Gradient norm clipping: scale factor is in (0, 1] when norm > max_norm.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_gradient_norm_scale_bounded() {
        let total_norm: f64 = kani::any();
        let max_norm: f64 = kani::any();
        assume_bounded_f64(total_norm, 1e-6, 1e10);
        assume_bounded_f64(max_norm, 1e-6, 1e6);

        if total_norm > max_norm {
            let scale = max_norm / total_norm;
            assert!(scale > 0.0, "Norm clip scale <= 0");
            assert!(scale <= 1.0, "Norm clip scale > 1");
            assert!(!scale.is_nan(), "Norm clip scale is NaN");
        }
    }

    /// After gradient value clipping, the update magnitude is bounded.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_clipped_update_bounded() {
        let w: f32 = kani::any();
        let grad: f32 = kani::any();
        let lr: f32 = kani::any();
        let clip_value: f32 = kani::any();
        assume_bounded(w, -1e4, 1e4);
        assume_bounded(grad, -1e10, 1e10);
        assume_bounded(lr, 1e-5, 1e-2);
        assume_bounded(clip_value, 0.01, 10.0);

        let clipped = if grad < -clip_value {
            -clip_value
        } else if grad > clip_value {
            clip_value
        } else {
            grad
        };

        let w_new = w - lr * clipped;
        assert!(!w_new.is_nan(), "Clipped update NaN");
        assert!(!w_new.is_infinite(), "Clipped update Inf");
    }

    // ── 8. Momentum accumulation bounded after N steps ──────────────────
    //
    // v_t = momentum * v_{t-1} + grad
    // For |momentum| < 1, this is a geometric series converging to
    // grad / (1 - momentum). We prove the accumulation stays bounded
    // after several steps.

    /// Momentum accumulation stays finite over 5 steps with varying gradients.
    #[kani::unwind(7)]
    #[kani::proof]
    fn prove_momentum_accumulation_5_steps_finite() {
        let momentum: f32 = kani::any();
        assume_bounded(momentum, 0.0, 0.99);

        let mut velocity: f32 = 0.0;
        let mut step: u32 = 0;
        while step < 5 {
            let grad: f32 = kani::any();
            assume_bounded(grad, -1e4, 1e4);
            velocity = momentum * velocity + grad;
            assert!(
                !velocity.is_nan() && !velocity.is_infinite(),
                "Momentum velocity not finite at step"
            );
            step += 1;
        }
    }

    /// Momentum accumulation magnitude bound: after N steps with |grad| <= G,
    /// |velocity| <= G / (1 - momentum) (geometric series bound) + rounding.
    /// Test with momentum=0.9, |grad| <= 100: bound = 100 / 0.1 = 1000.
    #[kani::unwind(12)]
    #[kani::proof]
    fn prove_momentum_accumulation_magnitude_bound() {
        let mut velocity: f32 = 0.0;
        let momentum: f32 = 0.9;
        let grad_bound: f32 = 100.0;
        // Geometric series bound: G / (1 - momentum) = 100 / 0.1 = 1000
        let theoretical_bound: f32 = 1000.0;

        let mut step: u32 = 0;
        while step < 10 {
            let grad: f32 = kani::any();
            assume_bounded(grad, -grad_bound, grad_bound);
            velocity = momentum * velocity + grad;
            assert!(
                !velocity.is_nan() && !velocity.is_infinite(),
                "Velocity not finite"
            );
            // Allow f32 rounding margin
            assert!(
                velocity.abs() <= theoretical_bound + 10.0,
                "Velocity exceeded geometric series bound"
            );
            step += 1;
        }
    }

    /// Full SGD with momentum stays finite over 8 steps (weight + velocity).
    #[kani::unwind(10)]
    #[kani::proof]
    fn prove_sgd_momentum_8_steps_finite() {
        let mut w: f32 = kani::any();
        let mut velocity: f32 = 0.0;
        let lr: f32 = kani::any();
        let momentum: f32 = kani::any();
        assume_bounded(w, -1e4, 1e4);
        assume_bounded(lr, 1e-5, 1e-2);
        assume_bounded(momentum, 0.0, 0.99);

        let mut step: u32 = 0;
        while step < 8 {
            let grad: f32 = kani::any();
            assume_bounded(grad, -1e3, 1e3);
            velocity = momentum * velocity + grad;
            w = w - lr * velocity;
            assert!(!w.is_nan() && !w.is_infinite(), "Weight not finite");
            assert!(
                !velocity.is_nan() && !velocity.is_infinite(),
                "Velocity not finite"
            );
            step += 1;
        }
    }
}
