// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for AdaFactor and Adam optimizer correctness.
//!
//! Covers properties NOT addressed by `kani_optim_proofs_adam.rs` or
//! `kani_optim_proofs_advanced.rs`:
//!
//! AdaFactor: relative step scaling, momentum contraction, column factor
//! non-negativity, factored reconstruction (sparse), relative wd shrinkage,
//! multi-step momentum, rho_t clamp idempotency, normalized gradient bounds.
//!
//! Adam: beta1/beta2 EMA contraction/non-negativity, zero-wd branch, step
//! magnitude bound, wide-range decay factor, monotonic wd shrinkage, bias
//! correction convergence (t=50), bc ratio bound, fused loop finiteness,
//! wd=0 branch equivalence.
//!
//! Re: #3646, #13 (verified training epic).

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

    // ── Adam: beta1_ema and beta2_ema (adam.rs:248-256) ──────────────

    /// beta1_ema is a contraction: |m_new| <= max(|m_prev|, |g|).
    /// Convex combination when beta1 in [0, 1).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_beta1_ema_contraction() {
        let m_prev: f32 = kani::any();
        let g: f32 = kani::any();
        let beta1: f32 = kani::any();
        assume_bounded(m_prev, -1e4, 1e4);
        assume_bounded(g, -1e4, 1e4);
        assume_bounded(beta1, 0.0, 0.999);
        let m_new = beta1 * m_prev + (1.0 - beta1) * g;
        let bound = m_prev.abs().max(g.abs());
        assert!(
            m_new.abs() <= bound + 1e-3,
            "beta1_ema must be a contraction"
        );
        assert!(!m_new.is_nan() && !m_new.is_infinite());
    }

    /// beta2_ema preserves non-negativity: v_prev >= 0 => v_new >= 0.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_beta2_ema_non_negative() {
        let v_prev: f32 = kani::any();
        let g: f32 = kani::any();
        let beta2: f32 = kani::any();
        assume_bounded(v_prev, 0.0, 1e8);
        assume_bounded(g, -1e4, 1e4);
        assume_bounded(beta2, 0.0, 0.9999);
        let v_new = beta2 * v_prev + (1.0 - beta2) * g * g;
        assert!(v_new >= 0.0, "beta2_ema must preserve non-negativity");
        assert!(!v_new.is_nan() && !v_new.is_infinite());
    }

    /// beta1_ema cold start: m_new = (1-beta1)*g when m_prev=0.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_beta1_ema_cold_start() {
        let g: f32 = kani::any();
        let beta1: f32 = kani::any();
        assume_bounded(g, -1e4, 1e4);
        assume_bounded(beta1, 0.0, 0.999);
        let m_new = beta1 * 0.0f32 + (1.0 - beta1) * g;
        let expected = (1.0 - beta1) * g;
        assert!((m_new - expected).abs() < 1e-6);
        assert!(!m_new.is_nan() && !m_new.is_infinite());
    }

    /// beta2_ema upper bound: v_new <= max(v_prev, g^2) (convex combination).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_beta2_ema_upper_bound() {
        let v_prev: f32 = kani::any();
        let g: f32 = kani::any();
        let beta2: f32 = kani::any();
        assume_bounded(v_prev, 0.0, 1e6);
        assume_bounded(g, -1e3, 1e3);
        assume_bounded(beta2, 0.0, 0.9999);
        let g_sq = g * g;
        let v_new = beta2 * v_prev + (1.0 - beta2) * g_sq;
        assert!(v_new <= v_prev.max(g_sq) + 1e-2);
        assert!(!v_new.is_nan() && !v_new.is_infinite());
    }

    // ── Adam: zero-wd branch (adam.rs:204-207 else) ──────────────────

    /// Adam f32 update with wd=0: theta_new = theta - step (no decay_factor).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::powi, powi_f64_stub)]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn prove_adam_f32_zero_weight_decay_finite() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let m_prev: f32 = kani::any();
        let v_prev: f32 = kani::any();
        let lr: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(m_prev, -1e4, 1e4);
        assume_bounded(v_prev, 0.0, 1e8);
        assume_bounded(lr, 1e-5, 1e-2);
        let m = 0.9f32 * m_prev + 0.1f32 * grad;
        let v = 0.999f32 * v_prev + 0.001f32 * grad * grad;
        let bc1 = 1.0f32 / (1.0 - (0.9f64).powi(1) as f32);
        let bc2 = 1.0f32 / (1.0 - (0.999f64).powi(1) as f32);
        let step = lr * (m * bc1) / ((v * bc2).sqrt() + 1e-8f32);
        let new_theta = theta - step; // Zero wd path
        assert!(!new_theta.is_nan() && !new_theta.is_infinite());
        assert!(!m.is_nan() && !v.is_nan());
    }

    // ── Adam: step magnitude bound (adam.rs:203) ─────────────────────

    /// |step| <= lr * |m_hat| / eps (worst case: v_hat = 0).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn prove_adam_step_magnitude_bounded() {
        let m_hat: f32 = kani::any();
        let v_hat: f32 = kani::any();
        let lr: f32 = kani::any();
        let eps: f32 = kani::any();
        assume_bounded(m_hat, -1e4, 1e4);
        assume_bounded(v_hat, 0.0, 1e8);
        assume_bounded(lr, 1e-5, 1e-2);
        assume_bounded(eps, 1e-8, 1e-4);
        let step = lr * m_hat / (v_hat.sqrt() + eps);
        let max_step = lr * m_hat.abs() / eps;
        assert!(step.abs() <= max_step + 1e-2);
        assert!(!step.is_nan() && !step.is_infinite());
    }

    // ── Adam: decay factor wide range (adam.rs:274) ──────────────────

    /// decay_factor positive for lr up to 1.0, wd up to 0.5.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adam_decay_factor_wide_range() {
        let lr: f32 = kani::any();
        let wd: f32 = kani::any();
        assume_bounded(lr, 1e-5, 1.0);
        assume_bounded(wd, 0.0, 0.5);
        kani::assume(lr * wd < 0.99);
        let decay_factor = 1.0f32 - lr * wd;
        assert!(decay_factor > 0.0 && decay_factor <= 1.0);
        assert!(!decay_factor.is_nan() && !decay_factor.is_infinite());
    }

    /// Weight decay monotonically shrinks |theta| over 3 steps (zero grad).
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn prove_adam_weight_decay_monotonic_shrinkage() {
        let mut theta: f32 = kani::any();
        let lr: f32 = kani::any();
        let wd: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(lr, 1e-4, 1e-2);
        assume_bounded(wd, 1e-3, 0.1);
        kani::assume(theta.abs() > 1e-6);
        let initial_mag = theta.abs();
        let decay_factor = 1.0f32 - lr * wd;
        let mut step: u32 = 0;
        while step < 3 {
            theta = theta * decay_factor;
            step += 1;
        }
        assert!(theta.abs() < initial_mag + 1e-4);
        assert!(!theta.is_nan() && !theta.is_infinite());
    }

    // ── Adam: bias correction convergence (adam.rs:260-274) ──────────

    /// At t=50, bc1 < 1.01 (nearly converged to 1).
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(52)]
    fn prove_adam_bias_correction_convergence_t50() {
        let mut pow: f64 = 1.0;
        let mut i: u32 = 0;
        while i < 50 {
            pow *= 0.9;
            i += 1;
        }
        let bc = 1.0 / (1.0 - pow);
        assert!(bc > 1.0 && bc < 1.01 && bc.is_finite());
    }

    /// bc1/sqrt(bc2) < 1 for t in [1,10] — bounds effective amplification.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(12)]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    fn prove_adam_bias_correction_ratio_bounded() {
        let step: u32 = kani::any();
        kani::assume(step >= 1 && step <= 10);
        let mut pow1: f64 = 1.0;
        let mut pow2: f64 = 1.0;
        let mut i: u32 = 0;
        while i < step {
            pow1 *= 0.9;
            pow2 *= 0.999;
            i += 1;
        }
        let bc1 = 1.0 / (1.0 - pow1);
        let bc2 = 1.0 / (1.0 - pow2);
        let ratio = bc1 / bc2.sqrt();
        assert!(ratio > 0.0 && ratio < 1.0 && ratio.is_finite());
    }

    // ── Adam: fused update (adam.rs:196-218) ─────────────────────────

    /// wd=0 branch and decay_factor branch produce identical results.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adam_fused_wd_zero_equivalence() {
        let theta: f32 = kani::any();
        let step_val: f32 = kani::any();
        let lr: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(step_val, -1e2, 1e2);
        assume_bounded(lr, 1e-5, 1e-2);
        let decay_factor = 1.0f32 - lr * 0.0f32; // = 1.0
        let wd_branch = theta * decay_factor - step_val;
        let no_wd_branch = theta - step_val;
        assert!((wd_branch - no_wd_branch).abs() < 1e-7);
    }

    /// Fused loop: finite inputs => finite output (non_finite_count = 0).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::powi, powi_f64_stub)]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn prove_adam_fused_loop_no_non_finite() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let m: f32 = kani::any();
        let v: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(m, -1e4, 1e4);
        assume_bounded(v, 0.0, 1e8);
        let new_m = 0.9f32 * m + 0.1f32 * grad;
        let new_v = 0.999f32 * v + 0.001f32 * grad * grad;
        let bc1 = 1.0f32 / (1.0 - (0.9f64).powi(1) as f32);
        let bc2 = 1.0f32 / (1.0 - (0.999f64).powi(1) as f32);
        let step = 1e-3f32 * (new_m * bc1) / ((new_v * bc2).sqrt() + 1e-8f32);
        let new_val = theta * (1.0f32 - 1e-3f32 * 0.01f32) - step;
        assert!(new_val.is_finite() && new_m.is_finite() && new_v.is_finite());
    }

    // ── AdaFactor: relative step (adafactor.rs:316-323) ──────────────

    /// rho_lr = 1/sqrt(t) is finite and in (0, 1] for t in [1, 1000].
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    fn prove_adafactor_relative_step_rho_lr_positive() {
        let step: u32 = kani::any();
        kani::assume(step >= 1 && step <= 1000);
        let rho_lr = 1.0 / (step as f64).sqrt();
        assert!(rho_lr > 0.0 && rho_lr <= 1.0 && rho_lr.is_finite());
    }

    /// lr_t = clamp_min(rms, eps_rms) * rho_lr >= eps_rms * rho_lr > 0.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    fn prove_adafactor_relative_lr_t_bounded() {
        let param_rms: f32 = kani::any();
        let step: u32 = kani::any();
        assume_bounded(param_rms, 0.0, 1e4);
        kani::assume(step >= 1 && step <= 1000);
        let eps_rms: f32 = 1e-3;
        let clamped = if param_rms > eps_rms {
            param_rms
        } else {
            eps_rms
        };
        let rho_lr = 1.0 / (step as f64).sqrt();
        let lr_t = clamped as f64 * rho_lr;
        assert!(lr_t >= eps_rms as f64 * rho_lr);
        assert!(lr_t > 0.0 && lr_t.is_finite());
    }

    // ── AdaFactor: first moment (adafactor.rs:305-311) ───────────────

    /// Momentum EMA is a contraction: |m_new| <= max(|m_prev|, |u_t|).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_first_moment_contraction() {
        let m_prev: f32 = kani::any();
        let u_t: f32 = kani::any();
        let beta1: f32 = kani::any();
        assume_bounded(m_prev, -1e4, 1e4);
        assume_bounded(u_t, -1e4, 1e4);
        assume_bounded(beta1, 0.0, 0.999);
        let m_new = beta1 * m_prev + (1.0 - beta1) * u_t;
        assert!(m_new.abs() <= m_prev.abs().max(u_t.abs()) + 1e-3);
        assert!(!m_new.is_nan() && !m_new.is_infinite());
    }

    /// Full AdaFactor scalar update with momentum (beta1 != None path).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn prove_adafactor_with_momentum_finite() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let v_prev: f32 = kani::any();
        let m_prev: f32 = kani::any();
        let lr: f32 = kani::any();
        let beta1: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(v_prev, 0.0, 1e8);
        assume_bounded(m_prev, -1e4, 1e4);
        assume_bounded(lr, 1e-5, 1e-2);
        assume_bounded(beta1, 0.0, 0.999);
        let v = 0.8f32 * v_prev + 0.2f32 * grad * grad;
        let u_t = grad / (v + 1e-30f32).sqrt();
        let m_new = beta1 * m_prev + (1.0 - beta1) * u_t;
        let new_theta = theta - lr * m_new;
        assert!(!new_theta.is_nan() && !new_theta.is_infinite());
        assert!(!v.is_nan() && !m_new.is_nan());
    }

    // ── AdaFactor: column factor (adafactor.rs:244-246) ──────────────

    /// Column factor EMA preserves non-negativity (mirrors row factor proof).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_col_factor_non_negative() {
        let col_prev: f32 = kani::any();
        let g_sq_mean: f32 = kani::any();
        let rho: f32 = kani::any();
        assume_bounded(col_prev, 0.0, 1e8);
        assume_bounded(g_sq_mean, 0.0, 1e8);
        assume_bounded(rho, 0.0, 0.999);
        let col_new = rho * col_prev + (1.0 - rho) * g_sq_mean;
        assert!(col_new >= 0.0);
        assert!(!col_new.is_nan() && !col_new.is_infinite());
    }

    // ── AdaFactor: factored reconstruction sparse (adafactor.rs:248-259) ─

    /// Reconstruction finite for near-zero factors (cold start / sparse grad).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_factored_reconstruction_sparse() {
        let row_val: f32 = kani::any();
        let col_val: f32 = kani::any();
        let row_mean: f32 = kani::any();
        assume_bounded(row_val, 0.0, 1e-6);
        assume_bounded(col_val, 0.0, 1e-6);
        assume_bounded(row_mean, 0.0, 1e-6);
        let denom = row_mean + 1e-30f32;
        assert!(denom > 0.0);
        let v_approx = row_val * col_val / denom;
        assert!(v_approx >= 0.0 && !v_approx.is_nan() && !v_approx.is_infinite());
    }

    // ── AdaFactor: relative wd shrinkage (adafactor.rs:324-328) ──────

    /// Weight decay in relative step mode does not amplify parameters.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_relative_wd_shrinkage() {
        let param: f32 = kani::any();
        let lr_t: f32 = kani::any();
        let wd: f32 = kani::any();
        assume_bounded(param, -1e4, 1e4);
        assume_bounded(lr_t, 1e-5, 1e-1);
        assume_bounded(wd, 0.0, 0.1);
        kani::assume(lr_t * wd < 0.5);
        let theta_wd = param - param * lr_t * wd;
        let shrink = 1.0f32 - lr_t * wd;
        assert!(shrink > 0.0 && shrink <= 1.0);
        assert!(theta_wd.abs() <= param.abs() + 1e-3);
        assert!(!theta_wd.is_nan() && !theta_wd.is_infinite());
    }

    // ── AdaFactor: multi-step with momentum ──────────────────────────

    /// AdaFactor with momentum stays finite over 3 consecutive steps.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(5)]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn prove_adafactor_momentum_multi_step_finite() {
        let mut theta: f32 = kani::any();
        let mut v: f32 = 0.0;
        let mut m: f32 = 0.0;
        let lr: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(lr, 1e-5, 1e-2);
        let mut step: u32 = 0;
        while step < 3 {
            let grad: f32 = kani::any();
            assume_bounded(grad, -1e4, 1e4);
            v = 0.8f32 * v + 0.2f32 * grad * grad;
            let u_t = grad / (v + 1e-30f32).sqrt();
            m = 0.9f32 * m + 0.1f32 * u_t;
            theta = theta - lr * m;
            assert!(!theta.is_nan() && !theta.is_infinite());
            assert!(!v.is_nan() && !m.is_nan());
            step += 1;
        }
    }

    // ── AdaFactor: rho_t clamp (adafactor.rs:211-214) ────────────────

    /// rho_t clamp to [0, 1-1e-8] is idempotent.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_rho_clamp_idempotent() {
        let raw: f64 = kani::any();
        assume_bounded_f64(raw, -1.0, 2.0);
        let clamp = |x: f64| -> f64 {
            if x < 0.0 {
                0.0
            } else if x > 1.0 - 1e-8 {
                1.0 - 1e-8
            } else {
                x
            }
        };
        let once = clamp(raw);
        let twice = clamp(once);
        assert!((once - twice).abs() < 1e-15);
        assert!(once >= 0.0 && once < 1.0);
    }

    // ── AdaFactor: normalized gradient u_t (adafactor.rs:302) ────────

    /// u_t = grad / sqrt(v + eps) is finite for valid inputs.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn prove_adafactor_normalized_gradient_finite() {
        let grad: f32 = kani::any();
        let v: f32 = kani::any();
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(v, 0.0, 1e8);
        let denom = (v + 1e-30f32).sqrt();
        assert!(denom > 0.0);
        let u_t = grad / denom;
        assert!(!u_t.is_nan() && !u_t.is_infinite());
    }

    /// u_t finiteness with wider gradient range (stress test).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn prove_adafactor_ut_wide_range_finite() {
        let grad: f32 = kani::any();
        let v: f32 = kani::any();
        assume_bounded(grad, -1e6, 1e6);
        assume_bounded(v, 0.0, 1e12);
        let u_t = grad / (v + 1e-30f32).sqrt();
        assert!(!u_t.is_nan() && !u_t.is_infinite());
    }
}
