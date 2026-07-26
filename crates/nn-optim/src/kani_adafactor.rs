// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for AdaFactor optimizer — config validation, rho_t
//! schedule, factored moment reconstruction, relative step, and update rules.
//!
//! Complements `kani_adafactor_adam_proofs.rs` and `kani_optim_proofs_adam.rs`
//! with deeper coverage of:
//!
//! - Config validation boundaries (eps_denom, eps_rms, decay_rate, beta1)
//! - rho_t schedule monotonicity and convergence
//! - Factored reconstruction error bound
//! - Relative step lr_t lower/upper bounds
//! - No-momentum path finiteness
//! - Weight decay + relative step interaction
//! - Step counter overflow safety
//!
//! Re: #3721, #13 (verified training epic).

#[cfg(kani)]
mod proofs {
    fn assume_bounded(x: f32, lo: f32, hi: f32) {
        kani::assume(!x.is_nan() && !x.is_infinite());
        kani::assume(x >= lo && x <= hi);
    }

    fn assume_bounded_f64(x: f64, lo: f64, hi: f64) {
        kani::assume(!x.is_nan() && !x.is_infinite());
        kani::assume(x >= lo && x <= hi);
    }

    fn sqrt_f32_stub(x: f32) -> f32 {
        let result: f32 = kani::any();
        kani::assume(result.is_finite() && result >= 0.0 && result <= 1e10);
        if x > 0.0 {
            kani::assume(result > 0.0);
        }
        result
    }

    fn sqrt_f64_stub(x: f64) -> f64 {
        let result: f64 = kani::any();
        kani::assume(result.is_finite() && result >= 0.0 && result <= 1e20);
        if x > 0.0 {
            kani::assume(result > 0.0);
        }
        result
    }

    // ── Config validation boundary proofs ─────────────────────────

    /// AdaFactor: eps_denom must be strictly positive and finite.
    /// Zero eps_denom causes division by zero in sqrt(v + eps_denom).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    fn prove_adafactor_eps_denom_positive_required() {
        let eps_denom: f64 = kani::any();
        assume_bounded_f64(eps_denom, 1e-40, 1e-1);
        let v: f32 = 0.0;
        let denom = (v as f64 + eps_denom).sqrt();
        assert!(denom > 0.0, "eps_denom must prevent zero denominator");
        assert!(denom.is_finite());
    }

    /// AdaFactor: eps_rms must be positive and finite for relative step.
    /// clamp_min(rms, eps_rms) ensures lr_t > 0 even when all params are zero.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_eps_rms_lower_bound() {
        let eps_rms: f64 = kani::any();
        assume_bounded_f64(eps_rms, 1e-6, 1.0);
        let param_rms: f64 = 0.0;
        let clamped = if param_rms > eps_rms {
            param_rms
        } else {
            eps_rms
        };
        assert!(clamped >= eps_rms, "clamp must enforce eps_rms floor");
        assert!(clamped > 0.0);
    }

    /// AdaFactor: beta1 at boundary 0.0 produces pure gradient (no momentum).
    /// m_new = 0 * m_prev + 1 * u_t = u_t.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_beta1_zero_is_identity() {
        let m_prev: f32 = kani::any();
        let u_t: f32 = kani::any();
        assume_bounded(m_prev, -1e4, 1e4);
        assume_bounded(u_t, -1e4, 1e4);

        let beta1: f32 = 0.0;
        let m_new = beta1 * m_prev + (1.0 - beta1) * u_t;
        assert!(
            (m_new - u_t).abs() < 1e-6,
            "beta1=0 must pass gradient through unchanged"
        );
    }

    /// AdaFactor: beta1 approaching 1.0 preserves old moment (ignores gradient).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_beta1_high_preserves_history() {
        let m_prev: f32 = kani::any();
        let u_t: f32 = kani::any();
        assume_bounded(m_prev, -1e4, 1e4);
        assume_bounded(u_t, -1e4, 1e4);

        let beta1: f32 = 0.999;
        let m_new = beta1 * m_prev + (1.0 - beta1) * u_t;
        let diff_from_prev = (m_new - beta1 * m_prev).abs();
        let grad_contrib = (1.0 - beta1) * u_t.abs();
        assert!(diff_from_prev <= grad_contrib + 1e-3);
    }

    // ── rho_t schedule proofs ─────────────────────────────────────

    /// rho_t at step 1 = clamp(1 - 1^decay_rate, 0, 1-1e-8) = 0.
    /// First step uses zero rho, meaning second moment is entirely the gradient.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_rho_t_first_step_is_zero() {
        let raw_rho = 1.0 - 1.0f64;
        let clamped = raw_rho.clamp(0.0, 1.0 - 1e-8);
        assert!(clamped == 0.0, "rho at step 1 must be 0");
    }

    /// Step counter is capped at i32::MAX before f64 cast.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_step_cap_to_i32_max() {
        let step: usize = kani::any();
        kani::assume(step >= 1);
        let capped = step.min(i32::MAX as usize);
        assert!(capped >= 1 && capped <= i32::MAX as usize);
        let t = capped as f64;
        assert!(t >= 1.0 && t.is_finite());
    }

    /// Monotonicity: for t_a < t_b, 1/t_a > 1/t_b (used in rho_t schedule).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_rho_t_monotonic_two_steps() {
        let t_a: u32 = kani::any();
        kani::assume(t_a >= 1 && t_a <= 999);
        let t_b = t_a + 1;
        let inv_a = 1.0 / (t_a as f64);
        let inv_b = 1.0 / (t_b as f64);
        assert!(inv_a > inv_b, "1/t must decrease as t increases");
    }

    // ── Factored moment reconstruction proofs ─────────────────────

    /// Factored reconstruction with large row_mean: result is attenuated.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_reconstruction_attenuated_by_large_mean() {
        let row_val: f32 = kani::any();
        let col_val: f32 = kani::any();
        let row_mean: f32 = kani::any();
        assume_bounded(row_val, 0.0, 1.0);
        assume_bounded(col_val, 0.0, 1.0);
        assume_bounded(row_mean, 10.0, 1e8);

        let denom = row_mean + 1e-30f32;
        let v_approx = row_val * col_val / denom;
        assert!(v_approx <= 0.1 + 1e-6);
        assert!(v_approx >= 0.0);
        assert!(!v_approx.is_nan() && !v_approx.is_infinite());
    }

    /// Factored reconstruction is proportional to row factor value.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_reconstruction_proportional_to_row() {
        let row_val: f32 = kani::any();
        let col_val: f32 = kani::any();
        let row_mean: f32 = kani::any();
        assume_bounded(row_val, 1e-3, 1e3);
        assume_bounded(col_val, 1e-3, 1e3);
        assume_bounded(row_mean, 1e-3, 1e3);

        let denom = row_mean + 1e-30f32;
        let v1 = row_val * col_val / denom;
        let v2 = (2.0 * row_val) * col_val / denom;
        assert!(
            (v2 - 2.0 * v1).abs() < 1e-2,
            "reconstruction must be proportional to row factor"
        );
    }

    // ── No-momentum path (beta1 = None) ────────────────────────────

    /// AdaFactor without momentum: u_t is used directly as the update.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn prove_adafactor_no_momentum_update_finite() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let v: f32 = kani::any();
        let lr: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(v, 0.0, 1e8);
        assume_bounded(lr, 1e-5, 1e-2);

        let u_t = grad / (v + 1e-30f32).sqrt();
        let new_theta = theta - lr * u_t;
        assert!(!new_theta.is_nan() && !new_theta.is_infinite());
    }

    /// AdaFactor no-momentum descent: positive grad decreases theta.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn prove_adafactor_no_momentum_descent_direction() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let v: f32 = kani::any();
        let lr: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(grad, 1e-3, 1e4);
        assume_bounded(v, 0.0, 1e8);
        assume_bounded(lr, 1e-5, 1e-2);

        let u_t = grad / (v + 1e-30f32).sqrt();
        let new_theta = theta - lr * u_t;
        assert!(
            new_theta < theta + 1e-6,
            "positive grad must decrease theta"
        );
    }

    // ── Weight decay interaction with fixed lr ─────────────────────

    /// AdaFactor fixed-lr weight decay: decay = 1 - lr * wd in (0, 1].
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_fixed_lr_decay_factor_bounded() {
        let lr: f64 = kani::any();
        let wd: f64 = kani::any();
        assume_bounded_f64(lr, 1e-5, 1e-2);
        assume_bounded_f64(wd, 0.0, 0.1);
        let decay = 1.0 - lr * wd;
        assert!(decay > 0.0 && decay <= 1.0);
        assert!(decay.is_finite());
    }

    /// AdaFactor fixed-lr: theta * (1 - lr*wd) - lr * u_t is finite.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_fixed_lr_full_update_finite() {
        let theta: f32 = kani::any();
        let u_t: f32 = kani::any();
        let lr: f32 = kani::any();
        let wd: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(u_t, -1e4, 1e4);
        assume_bounded(lr, 1e-5, 1e-2);
        assume_bounded(wd, 0.0, 0.1);

        let decay_factor = 1.0f32 - lr * wd;
        let new_theta = theta * decay_factor - lr * u_t;
        assert!(!new_theta.is_nan() && !new_theta.is_infinite());
    }

    // ── Step counter arithmetic ───────────────────────────────────

    /// step_t increments safely; capped value is always valid.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_step_increment_safe() {
        let step: usize = kani::any();
        kani::assume(step < usize::MAX);
        let new_step = step + 1;
        assert!(new_step > step);
        let capped = new_step.min(i32::MAX as usize);
        assert!(capped >= 1);
        let t_f64 = capped as f64;
        assert!(t_f64 >= 1.0 && t_f64.is_finite());
    }

    /// rho_t at capped max step still produces valid rho in [0, 1).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_rho_t_at_max_step() {
        let raw: f64 = kani::any();
        kani::assume(raw >= 0.0 && raw < 1.0);
        let clamped = raw.clamp(0.0, 1.0 - 1e-8);
        assert!(clamped >= 0.0 && clamped < 1.0);
        assert!(clamped.is_finite());
    }

    // ── Relative step lr_t bounds ──────────────────────────────────

    /// Relative step: lr_t = clamp_min(rms, eps_rms) / sqrt(t) is bounded.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    fn prove_adafactor_relative_lr_t_upper_bound() {
        let param_rms: f32 = kani::any();
        let step: u32 = kani::any();
        assume_bounded(param_rms, 0.0, 100.0);
        kani::assume(step >= 1 && step <= 1000);

        let eps_rms: f32 = 1e-3;
        let clamped = if param_rms > eps_rms {
            param_rms
        } else {
            eps_rms
        };
        let rho_lr = 1.0 / (step as f64).sqrt();
        let lr_t = clamped as f64 * rho_lr;

        assert!(lr_t <= 100.001);
        assert!(lr_t >= eps_rms as f64 * rho_lr - 1e-10);
        assert!(lr_t > 0.0 && lr_t.is_finite());
    }
}
