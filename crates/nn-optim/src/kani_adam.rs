// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Module-aligned Kani proof harnesses for `adam.rs`.
//!
//! These harnesses focus on scalar properties of AdamW's fused update path:
//! - bias-correction terms stay positive and finite
//! - bounded inputs do not produce NaN/Inf updates
//! - zero-gradient weight decay shrinks parameter magnitude

#[cfg(kani)]
mod proofs {
    fn assume_bounded(x: f32, lo: f32, hi: f32) {
        kani::assume(x.is_finite());
        kani::assume(x >= lo && x <= hi);
    }

    fn assume_beta(x: f32) {
        assume_bounded(x, 0.0, 0.9999);
    }

    fn sqrt_f32_stub(x: f32) -> f32 {
        let result: f32 = kani::any();
        kani::assume(result.is_finite() && result >= 0.0 && result <= 1e10);
        if x > 0.0 {
            kani::assume(result > 0.0);
        }
        result
    }

    fn powi_f64_stub(base: f64, exp: i32) -> f64 {
        let result: f64 = kani::any();
        kani::assume(result.is_finite() && result >= 0.0 && result <= 1e20);
        if base > 0.0 && base < 1.0 && exp >= 1 {
            kani::assume(result > 0.0 && result <= base);
        }
        if base > 0.0 {
            kani::assume(result > 0.0);
        }
        result
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    #[kani::stub(f64::powi, powi_f64_stub)]
    fn prove_adam_bias_corrections_positive() {
        let beta1: f32 = kani::any();
        let beta2: f32 = kani::any();
        let step_t: i32 = kani::any();
        assume_beta(beta1);
        assume_beta(beta2);
        kani::assume((1..=16).contains(&step_t));

        let bc1 = 1.0f32 / (1.0 - (beta1 as f64).powi(step_t) as f32);
        let bc2 = 1.0f32 / (1.0 - (beta2 as f64).powi(step_t) as f32);

        assert!(bc1.is_finite() && bc1 >= 1.0);
        assert!(bc2.is_finite() && bc2 >= 1.0);
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    #[kani::stub(f64::powi, powi_f64_stub)]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn prove_adam_scalar_update_finite() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let m_prev: f32 = kani::any();
        let v_prev: f32 = kani::any();
        let lr: f32 = kani::any();
        let beta1: f32 = kani::any();
        let beta2: f32 = kani::any();
        let eps: f32 = kani::any();
        let weight_decay: f32 = kani::any();
        let step_t: i32 = kani::any();

        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(m_prev, -1e4, 1e4);
        assume_bounded(v_prev, 0.0, 1e8);
        assume_bounded(lr, 1e-5, 1e-2);
        assume_beta(beta1);
        assume_beta(beta2);
        assume_bounded(eps, 1e-8, 1e-4);
        assume_bounded(weight_decay, 0.0, 0.1);
        kani::assume((1..=8).contains(&step_t));

        let m = beta1 * m_prev + (1.0 - beta1) * grad;
        let v = beta2 * v_prev + (1.0 - beta2) * grad * grad;
        let bc1 = 1.0f32 / (1.0 - (beta1 as f64).powi(step_t) as f32);
        let bc2 = 1.0f32 / (1.0 - (beta2 as f64).powi(step_t) as f32);
        let m_hat = m * bc1;
        let v_hat = v * bc2;
        let step = lr * m_hat / (v_hat.sqrt() + eps);
        let decay_factor = 1.0f32 - lr * weight_decay;
        let new_theta = theta * decay_factor - step;

        assert!(m.is_finite());
        assert!(v.is_finite() && v >= 0.0);
        assert!(step.is_finite());
        assert!(decay_factor.is_finite() && decay_factor > 0.0 && decay_factor <= 1.0);
        assert!(new_theta.is_finite());
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_adam_zero_grad_weight_decay_shrinks() {
        let theta: f32 = kani::any();
        let lr: f32 = kani::any();
        let weight_decay: f32 = kani::any();

        assume_bounded(theta, -1e6, 1e6);
        assume_bounded(lr, 1e-5, 1e-2);
        assume_bounded(weight_decay, 1e-4, 0.1);

        let decay_factor = 1.0f32 - lr * weight_decay;
        let new_theta = theta * decay_factor;

        assert!(decay_factor > 0.0 && decay_factor <= 1.0);
        assert!(new_theta.is_finite());
        assert!(new_theta.abs() <= theta.abs() + 1e-6);
    }
}
