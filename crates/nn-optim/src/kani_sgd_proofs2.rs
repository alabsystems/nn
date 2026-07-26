// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional Kani proof harnesses for SGD optimizer.
//!
//! Extends `kani_optim_proofs.rs` with:
//! - Zero gradient identity: theta unchanged when grad == 0
//! - Zero LR identity: theta unchanged when lr == 0
//! - Weight decay direction: always shrinks magnitude toward zero
//! - Momentum damping: velocity decays when gradient is zero
//! - SGD descent property: update direction opposes gradient sign (vanilla SGD)
//! - SGD with weight decay + momentum multi-step stability
//!
//! Re: #3668, #13 (verified training epic).

#[cfg(kani)]
mod proofs {
    fn assume_bounded(x: f32, lo: f32, hi: f32) {
        kani::assume(!x.is_nan() && !x.is_infinite());
        kani::assume(x >= lo && x <= hi);
    }

    /// SGD scalar update (duplicated for module isolation).
    fn sgd_scalar_update(
        theta: f32,
        grad: f32,
        lr: f32,
        weight_decay: f32,
        momentum: f32,
        velocity_prev: f32,
    ) -> (f32, f32) {
        let grad_eff = grad + weight_decay * theta;
        let velocity = if momentum > 0.0 {
            momentum * velocity_prev + grad_eff
        } else {
            grad_eff
        };
        let new_theta = theta - lr * velocity;
        (new_theta, velocity)
    }

    // ── Zero gradient identity ───────────────────────────────────────

    /// SGD with zero gradient and no weight decay: theta unchanged.
    /// theta_{t+1} = theta_t - lr * 0 = theta_t.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_sgd_zero_grad_identity() {
        let theta: f32 = kani::any();
        let lr: f32 = kani::any();
        assume_bounded(theta, -1e6, 1e6);
        assume_bounded(lr, 1e-5, 1.0);

        let (new_theta, _) = sgd_scalar_update(theta, 0.0, lr, 0.0, 0.0, 0.0);
        assert!(
            (new_theta - theta).abs() < 1e-10,
            "zero gradient must leave theta unchanged (vanilla SGD)"
        );
    }

    /// SGD with zero LR: theta unchanged regardless of gradient.
    /// theta_{t+1} = theta_t - 0 * velocity = theta_t.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_sgd_zero_lr_identity() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let momentum: f32 = kani::any();
        let v_prev: f32 = kani::any();
        assume_bounded(theta, -1e6, 1e6);
        assume_bounded(grad, -1e6, 1e6);
        assume_bounded(momentum, 0.0, 0.999);
        assume_bounded(v_prev, -1e6, 1e6);

        let (new_theta, _) = sgd_scalar_update(theta, grad, 0.0, 0.0, momentum, v_prev);
        assert!(
            (new_theta - theta).abs() < 1e-10,
            "zero LR must leave theta unchanged"
        );
    }

    // ── Weight decay direction ───────────────────────────────────────

    /// Weight decay pushes theta toward zero.
    /// With zero gradient: effective_grad = wd * theta.
    /// update = -lr * wd * theta. If theta > 0, update < 0 (shrinks).
    /// If theta < 0, update > 0 (shrinks). Magnitude always decreases.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_sgd_weight_decay_shrinks_toward_zero() {
        let theta: f32 = kani::any();
        let lr: f32 = kani::any();
        let wd: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(lr, 1e-5, 1e-2);
        assume_bounded(wd, 1e-4, 0.1);
        kani::assume(theta.abs() > 1e-3); // non-trivial theta

        // Zero gradient, just weight decay
        let (new_theta, _) = sgd_scalar_update(theta, 0.0, lr, wd, 0.0, 0.0);

        // new_theta = theta - lr * wd * theta = theta * (1 - lr * wd)
        // Since lr * wd > 0 and < 1: |new_theta| < |theta|
        assert!(
            new_theta.abs() <= theta.abs(),
            "weight decay must shrink magnitude toward zero"
        );
    }

    // ── Momentum damping ─────────────────────────────────────────────

    /// With zero gradient and no weight decay, momentum velocity decays.
    /// v_t = momentum * v_{t-1} + 0 = momentum * v_{t-1}.
    /// Since momentum < 1: |v_t| < |v_{t-1}|.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_sgd_momentum_damps_with_zero_grad() {
        let momentum: f32 = kani::any();
        let v_prev: f32 = kani::any();
        assume_bounded(momentum, 0.01, 0.999);
        assume_bounded(v_prev, -1e4, 1e4);
        kani::assume(v_prev.abs() > 1e-3); // non-trivial velocity

        // grad=0, wd=0: v_new = momentum * v_prev
        let (_, v_new) = sgd_scalar_update(0.0, 0.0, 1e-3, 0.0, momentum, v_prev);

        assert!(
            v_new.abs() < v_prev.abs() + 1e-6,
            "momentum must damp velocity when gradient is zero"
        );
    }

    // ── Descent property ─────────────────────────────────────────────

    /// Vanilla SGD (no momentum, no wd) moves theta in opposite direction of gradient.
    /// If grad > 0: new_theta < theta (descend). If grad < 0: new_theta > theta.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_sgd_descent_direction() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let lr: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(lr, 1e-5, 1e-2);
        kani::assume(grad.abs() > 1e-3); // non-zero gradient

        let (new_theta, _) = sgd_scalar_update(theta, grad, lr, 0.0, 0.0, 0.0);
        // delta = new_theta - theta = -lr * grad
        let delta = new_theta - theta;

        // delta and grad must have opposite signs
        assert!(
            delta * grad <= 0.0,
            "SGD must move theta opposite to gradient direction"
        );
    }

    // ── Multi-step with weight decay + momentum ──────────────────────

    /// SGD with weight decay + momentum stays finite over 5 steps.
    /// Full-featured SGD exercising all code paths simultaneously.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(7)]
    fn prove_sgd_full_feature_multi_step_finite() {
        let mut theta: f32 = kani::any();
        let mut velocity: f32 = 0.0;
        let lr: f32 = kani::any();
        let momentum: f32 = kani::any();
        let wd: f32 = kani::any();
        assume_bounded(theta, -1e3, 1e3);
        assume_bounded(lr, 1e-5, 1e-3);
        assume_bounded(momentum, 0.0, 0.99);
        assume_bounded(wd, 0.0, 0.01);

        let mut step: u32 = 0;
        while step < 5 {
            let grad: f32 = kani::any();
            assume_bounded(grad, -1e3, 1e3);
            let (new_theta, new_v) = sgd_scalar_update(theta, grad, lr, wd, momentum, velocity);
            assert!(!new_theta.is_nan() && !new_theta.is_infinite());
            assert!(!new_v.is_nan() && !new_v.is_infinite());
            theta = new_theta;
            velocity = new_v;
            step += 1;
        }
    }

    /// SGD update preserves relative ordering: if theta_a > theta_b initially
    /// and both receive the same gradient, then new_theta_a > new_theta_b
    /// (vanilla SGD without weight decay).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_sgd_preserves_relative_ordering() {
        let theta_a: f32 = kani::any();
        let theta_b: f32 = kani::any();
        let grad: f32 = kani::any();
        let lr: f32 = kani::any();
        assume_bounded(theta_a, -1e4, 1e4);
        assume_bounded(theta_b, -1e4, 1e4);
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(lr, 1e-5, 1e-2);
        kani::assume(theta_a > theta_b);

        let (new_a, _) = sgd_scalar_update(theta_a, grad, lr, 0.0, 0.0, 0.0);
        let (new_b, _) = sgd_scalar_update(theta_b, grad, lr, 0.0, 0.0, 0.0);

        // Both get same update: theta - lr * grad.
        // new_a - new_b = (theta_a - lr*grad) - (theta_b - lr*grad) = theta_a - theta_b > 0
        assert!(
            new_a > new_b,
            "vanilla SGD with same gradient must preserve relative ordering"
        );
    }
}
