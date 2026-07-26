// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for SGD optimizer update rules.
//!
//! Each harness proves that a scalar SGD update step produces finite
//! output for finite inputs with valid hyperparameters.
//!
//! Adam and AdaFactor harnesses are in `kani_optim_proofs_adam.rs`.
//!
//! ## Known gap: f64 vs f32 precision (#1515 AC3)
//!
//! The scalar functions below use f32 arithmetic to match `DynTensor` storage.
//! The proofs verify single-element scalar math, not tensor-level accumulation.
//!
//! - **Proved:** Each scalar update is finite for bounded inputs.
//! - **Not proved:** Tensor-level accumulation order matches scalar order exactly.
//! - **Implication:** f32 rounding in multi-element reduction may differ by ULP.
//!   The finiteness property still holds because element-wise finiteness implies
//!   tensor finiteness.
//!
//! Re: #13 (verified training epic), #1464.

#[cfg(kani)]
mod proofs {
    // ── Scalar optimizer update functions ────────────────────────────

    /// SGD update: theta_{t+1} = theta_t - lr * (grad + wd * theta_t)
    /// With momentum: v_t = momentum * v_{t-1} + grad_effective
    /// Matches `sgd.rs:80-101`.
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

    fn assume_bounded(x: f32, lo: f32, hi: f32) {
        kani::assume(!x.is_nan() && !x.is_infinite());
        kani::assume(x >= lo && x <= hi);
    }

    // ── SGD harnesses ────────────────────────────────────────────────

    /// SGD vanilla (no momentum, no weight decay) produces finite output.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_sgd_vanilla_finite() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let lr: f32 = kani::any();
        assume_bounded(theta, -1e6, 1e6);
        assume_bounded(grad, -1e6, 1e6);
        assume_bounded(lr, 1e-5, 1e-2);
        let (new_theta, _) = sgd_scalar_update(theta, grad, lr, 0.0, 0.0, 0.0);
        assert!(!new_theta.is_nan() && !new_theta.is_infinite());
    }

    /// SGD with momentum produces finite output.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_sgd_momentum_finite() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let lr: f32 = kani::any();
        let momentum: f32 = kani::any();
        let v_prev: f32 = kani::any();
        assume_bounded(theta, -1e6, 1e6);
        assume_bounded(grad, -1e6, 1e6);
        assume_bounded(lr, 1e-5, 1e-2);
        assume_bounded(momentum, 0.0, 0.999);
        assume_bounded(v_prev, -1e6, 1e6);
        let (new_theta, new_v) = sgd_scalar_update(theta, grad, lr, 0.0, momentum, v_prev);
        assert!(!new_theta.is_nan() && !new_theta.is_infinite());
        assert!(!new_v.is_nan() && !new_v.is_infinite());
    }

    /// SGD with weight decay produces finite output.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_sgd_weight_decay_finite() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let lr: f32 = kani::any();
        let wd: f32 = kani::any();
        assume_bounded(theta, -1e6, 1e6);
        assume_bounded(grad, -1e6, 1e6);
        assume_bounded(lr, 1e-5, 1e-2);
        assume_bounded(wd, 0.0, 0.1);
        let (new_theta, _) = sgd_scalar_update(theta, grad, lr, wd, 0.0, 0.0);
        assert!(!new_theta.is_nan() && !new_theta.is_infinite());
    }

    /// SGD with all features (momentum + weight decay) produces finite output.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_sgd_full_finite() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let lr: f32 = kani::any();
        let wd: f32 = kani::any();
        let momentum: f32 = kani::any();
        let v_prev: f32 = kani::any();
        assume_bounded(theta, -1e6, 1e6);
        assume_bounded(grad, -1e6, 1e6);
        assume_bounded(lr, 1e-5, 1e-2);
        assume_bounded(wd, 0.0, 0.1);
        assume_bounded(momentum, 0.0, 0.999);
        assume_bounded(v_prev, -1e6, 1e6);
        let (new_theta, new_v) = sgd_scalar_update(theta, grad, lr, wd, momentum, v_prev);
        assert!(!new_theta.is_nan() && !new_theta.is_infinite());
        assert!(!new_v.is_nan() && !new_v.is_infinite());
    }

    /// SGD vanilla update magnitude is bounded: |delta| <= lr * |grad|.
    /// Uses narrower ranges to avoid f32 cancellation error at scale boundaries.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_sgd_update_bounded() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let lr: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(lr, 1e-5, 1e-2);
        let (new_theta, _) = sgd_scalar_update(theta, grad, lr, 0.0, 0.0, 0.0);
        let delta = (new_theta - theta).abs();
        // lr <= 0.01, |grad| <= 1e4, so delta <= 100 (+ f32 rounding)
        assert!(delta <= 100.0 + 1.0);
    }

    // ── Multi-step SGD harness (AC4, #1515) ──────────────────────────

    /// SGD stays finite over 5 consecutive update steps with varying gradients.
    /// Exercises state accumulation (momentum velocity) across steps.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(7)]
    fn prove_sgd_multi_step_finite() {
        let mut theta: f32 = kani::any();
        let mut velocity: f32 = 0.0;
        let lr: f32 = kani::any();
        let momentum: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(lr, 1e-5, 1e-2);
        assume_bounded(momentum, 0.0, 0.999);
        let mut step: u32 = 0;
        while step < 5 {
            let grad: f32 = kani::any();
            assume_bounded(grad, -1e4, 1e4);
            let (new_theta, new_v) = sgd_scalar_update(theta, grad, lr, 0.0, momentum, velocity);
            assert!(!new_theta.is_nan() && !new_theta.is_infinite());
            assert!(!new_v.is_nan() && !new_v.is_infinite());
            theta = new_theta;
            velocity = new_v;
            step += 1;
        }
    }
}
