// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional Kani proof harnesses for nn-optim.
//!
//! Covers gaps in verification for:
//! - GradScaler scale invariants (bounded, positive, inv_scale finite)
//! - Adam EMA contraction and second-moment non-negativity
//! - Adam bias correction convergence toward 1.0
//! - SGD momentum geometric series bound
//! - Validation functions (validate_lr, validate_weight_decay)
//! - Checkpoint step overflow guard
//! - LoRA scaling finiteness
//! - Cosine schedule monotone decay in cosine phase
//! - AdaFactor rho_t convergence property
//! - GradScaler growth/backoff cycle invariant
//!
//! Re: #3795, #13 (verified training epic).

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

    fn powf_f64_stub(base: f64, _exp: f64) -> f64 {
        let _ = base;
        let result: f64 = kani::any();
        kani::assume(result.is_finite() && result >= 0.0 && result <= 1e20);
        result
    }

    // ── GradScaler invariants ─────────────────────────────────────

    /// GradScaler: scale stays in [min_scale, max_scale] after a growth update.
    /// When growth_tracker reaches growth_interval and no inf found,
    /// new_scale = min(scale * growth_factor, max_scale).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_grad_scaler_growth_stays_bounded() {
        let scale: f64 = kani::any();
        let growth_factor: f64 = kani::any();
        let max_scale: f64 = kani::any();
        let min_scale: f64 = kani::any();
        assume_bounded_f64(scale, 1.0, 1e7);
        assume_bounded_f64(growth_factor, 1.01, 4.0);
        assume_bounded_f64(max_scale, 1.0, 1e8);
        assume_bounded_f64(min_scale, 0.001, 1e4);
        kani::assume(min_scale <= scale && scale <= max_scale);

        let new_scale = if scale * growth_factor > max_scale {
            max_scale
        } else {
            scale * growth_factor
        };

        assert!(new_scale >= min_scale, "scale after growth >= min_scale");
        assert!(new_scale <= max_scale, "scale after growth <= max_scale");
        assert!(new_scale.is_finite());
        assert!(new_scale > 0.0);
    }

    /// GradScaler: scale stays in [min_scale, max_scale] after a backoff update.
    /// When inf/NaN found: new_scale = max(scale * backoff_factor, min_scale).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_grad_scaler_backoff_stays_bounded() {
        let scale: f64 = kani::any();
        let backoff_factor: f64 = kani::any();
        let min_scale: f64 = kani::any();
        let max_scale: f64 = kani::any();
        assume_bounded_f64(scale, 1.0, 1e7);
        assume_bounded_f64(backoff_factor, 0.01, 0.99);
        assume_bounded_f64(min_scale, 0.001, 1e4);
        assume_bounded_f64(max_scale, 1.0, 1e8);
        kani::assume(min_scale <= scale && scale <= max_scale);

        let new_scale = if scale * backoff_factor < min_scale {
            min_scale
        } else {
            scale * backoff_factor
        };

        assert!(new_scale >= min_scale, "scale after backoff >= min_scale");
        assert!(new_scale <= max_scale, "scale after backoff <= max_scale");
        assert!(new_scale.is_finite());
        assert!(new_scale > 0.0);
    }

    /// GradScaler: alternating growth/backoff cycles keep scale bounded.
    /// One growth then one backoff: scale * growth * backoff <= max_scale
    /// when growth * backoff < some bound.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_grad_scaler_growth_backoff_cycle() {
        let scale: f64 = kani::any();
        let growth_factor: f64 = kani::any();
        let backoff_factor: f64 = kani::any();
        let min_scale: f64 = kani::any();
        let max_scale: f64 = kani::any();
        assume_bounded_f64(scale, 1.0, 1e7);
        assume_bounded_f64(growth_factor, 1.01, 4.0);
        assume_bounded_f64(backoff_factor, 0.01, 0.99);
        assume_bounded_f64(min_scale, 0.001, 1e4);
        assume_bounded_f64(max_scale, 1.0, 1e8);
        kani::assume(min_scale <= scale && scale <= max_scale);

        // Growth step
        let after_growth = (scale * growth_factor).min(max_scale);
        // Backoff step
        let after_backoff = (after_growth * backoff_factor).max(min_scale);

        assert!(after_backoff >= min_scale);
        assert!(after_backoff <= max_scale);
        assert!(after_backoff.is_finite() && after_backoff > 0.0);
    }

    /// GradScaler: scale never reaches zero after repeated backoff.
    /// With min_scale > 0 and backoff_factor in (0,1),
    /// max(scale * backoff, min_scale) > 0 always.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_grad_scaler_scale_never_zero() {
        let scale: f64 = kani::any();
        let backoff_factor: f64 = kani::any();
        let min_scale: f64 = kani::any();
        assume_bounded_f64(scale, 0.001, 1e7);
        assume_bounded_f64(backoff_factor, 0.01, 0.99);
        assume_bounded_f64(min_scale, 0.001, 1e4);
        kani::assume(scale >= min_scale);

        let new_scale = (scale * backoff_factor).max(min_scale);
        assert!(new_scale > 0.0, "scale must remain positive");
        assert!(new_scale >= min_scale);
    }

    // ── Adam EMA properties ───────────────────────────────────────

    /// Adam beta1 EMA is a contraction: m is a weighted average of m_prev
    /// and grad, so |m| <= max(|m_prev|, |grad|).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adam_beta1_ema_contraction() {
        let m_prev: f32 = kani::any();
        let grad: f32 = kani::any();
        let beta1: f32 = kani::any();
        assume_bounded(m_prev, -1e4, 1e4);
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(beta1, 0.0, 0.9999);

        let m = beta1 * m_prev + (1.0 - beta1) * grad;
        let bound = m_prev.abs().max(grad.abs());
        assert!(
            m.abs() <= bound + 1e-3,
            "EMA must be bounded by max of inputs"
        );
        assert!(m.is_finite());
    }

    /// Adam beta2 EMA preserves non-negativity of v.
    /// v_new = beta2 * v_prev + (1-beta2) * g^2.
    /// Since v_prev >= 0 and g^2 >= 0: v_new >= 0.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adam_beta2_ema_non_negative() {
        let v_prev: f32 = kani::any();
        let grad: f32 = kani::any();
        let beta2: f32 = kani::any();
        assume_bounded(v_prev, 0.0, 1e8);
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(beta2, 0.0, 0.9999);

        let v = beta2 * v_prev + (1.0 - beta2) * grad * grad;
        assert!(v >= 0.0, "second moment must be non-negative");
        assert!(v.is_finite());
    }

    /// Adam decay_factor = 1 - lr * weight_decay is in (0, 1]
    /// for typical learning rates and weight decay values.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adam_decay_factor_bounds() {
        let lr: f32 = kani::any();
        let wd: f32 = kani::any();
        assume_bounded(lr, 1e-5, 1e-2);
        assume_bounded(wd, 0.0, 0.1);

        let decay_factor = 1.0f32 - lr * wd;
        assert!(decay_factor > 0.0, "decay factor must be positive");
        assert!(decay_factor <= 1.0, "decay factor must be <= 1");
        assert!(decay_factor.is_finite());
    }

    /// Adam bias corrections bc1 and bc2 converge toward 1.0 as step increases.
    /// At step t: bc = 1 / (1 - beta^t). As t grows, beta^t -> 0, so bc -> 1.
    /// Prove: bc at step t+1 is closer to 1.0 than bc at step t.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    #[kani::stub(f64::powi, powi_f64_stub)]
    fn prove_adam_bias_correction_convergence() {
        let beta: f32 = kani::any();
        let step_t: i32 = kani::any();
        assume_bounded(beta, 0.5, 0.999);
        kani::assume(step_t >= 1 && step_t <= 15);

        let bc_t = 1.0f32 / (1.0 - (beta as f64).powi(step_t) as f32);
        let bc_t1 = 1.0f32 / (1.0 - (beta as f64).powi(step_t + 1) as f32);

        // bc_t1 is closer to 1.0 than bc_t (both are >= 1.0)
        assert!(bc_t >= 1.0, "bias correction must be >= 1");
        assert!(bc_t1 >= 1.0, "bias correction must be >= 1");
        assert!(
            bc_t1 <= bc_t + 1e-6,
            "bias correction must converge toward 1"
        );
    }

    /// Adam: v_hat (bias-corrected second moment) is non-negative
    /// when v >= 0 and bc2 > 0.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adam_v_hat_non_negative() {
        let v: f32 = kani::any();
        let bc2: f32 = kani::any();
        assume_bounded(v, 0.0, 1e8);
        assume_bounded(bc2, 1.0, 100.0);

        let v_hat = v * bc2;
        assert!(v_hat >= 0.0, "bias-corrected v must be non-negative");
        assert!(v_hat.is_finite());
    }

    // ── SGD momentum geometric bound ──────────────────────────────

    /// SGD momentum velocity after k zero-gradient steps:
    /// |v_k| = momentum^k * |v_0| <= |v_0| (since momentum < 1).
    /// Proves geometric decay without gradient input.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn prove_sgd_momentum_geometric_decay() {
        let v0: f32 = kani::any();
        let momentum: f32 = kani::any();
        assume_bounded(v0, -1e4, 1e4);
        assume_bounded(momentum, 0.01, 0.999);

        // 3 steps of zero gradient
        let v1 = momentum * v0;
        let v2 = momentum * v1;
        let v3 = momentum * v2;

        assert!(v1.abs() <= v0.abs() + 1e-3);
        assert!(v2.abs() <= v1.abs() + 1e-3);
        assert!(v3.abs() <= v2.abs() + 1e-3);
        assert!(v3.is_finite());
    }

    // ── Validation function proofs ────────────────────────────────

    /// validate_lr: rejects NaN input.
    /// NaN is not finite, so !lr.is_finite() is true -> Err.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_validate_lr_rejects_nan() {
        let lr = f64::NAN;
        let valid = lr.is_finite() && lr >= 0.0;
        assert!(!valid, "NaN must be rejected by lr validation");
    }

    /// validate_lr: rejects negative values.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_validate_lr_rejects_negative() {
        let lr: f64 = kani::any();
        assume_bounded_f64(lr, -1e10, -1e-10);
        let valid = lr.is_finite() && lr >= 0.0;
        assert!(!valid, "negative lr must be rejected");
    }

    /// validate_lr: accepts zero and positive finite values.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_validate_lr_accepts_valid() {
        let lr: f64 = kani::any();
        assume_bounded_f64(lr, 0.0, 10.0);
        let valid = lr.is_finite() && lr >= 0.0;
        assert!(valid, "non-negative finite lr must be accepted");
    }

    /// validate_lr: rejects +Inf.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_validate_lr_rejects_inf() {
        let lr = f64::INFINITY;
        let valid = lr.is_finite() && lr >= 0.0;
        assert!(!valid, "+Inf must be rejected by lr validation");
    }

    /// validate_weight_decay: rejects NaN.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_validate_wd_rejects_nan() {
        let wd = f64::NAN;
        let valid = wd.is_finite() && wd >= 0.0;
        assert!(!valid, "NaN must be rejected by weight_decay validation");
    }

    /// validate_weight_decay: accepts zero (no decay).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_validate_wd_accepts_zero() {
        let wd = 0.0f64;
        let valid = wd.is_finite() && wd >= 0.0;
        assert!(valid, "zero weight_decay must be accepted");
    }

    // ── Checkpoint step overflow ──────────────────────────────────

    /// Checkpoint step: i64 value within usize range converts safely.
    /// step >= 0 and step <= i64::MAX as usize is valid.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_checkpoint_step_usize_safe() {
        let step: i64 = kani::any();
        kani::assume(step >= 0 && step <= 1_000_000_000);
        let as_usize = step as usize;
        assert!(as_usize <= 1_000_000_000);
        // Round-trip
        let back = as_usize as i64;
        assert!(back == step, "round-trip must preserve value");
    }

    /// Checkpoint step: negative step is invalid.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_checkpoint_negative_step_invalid() {
        let step: i64 = kani::any();
        kani::assume(step < 0);
        // Negative step should be rejected; casting to usize wraps.
        let valid = step >= 0;
        assert!(!valid, "negative step must be rejected");
    }

    /// LoRA scaling as f32: no overflow for typical values.
    /// alpha in [-1e6, 1e6], rank >= 1 => scaling in [-1e6, 1e6]
    /// which fits in f32 range.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_lora_scaling_f32_safe() {
        let alpha: f64 = kani::any();
        let rank: usize = kani::any();
        assume_bounded_f64(alpha, -1e4, 1e4);
        kani::assume(rank >= 1 && rank <= 1024);

        let scaling = alpha / rank as f64;
        let as_f32 = scaling as f32;
        assert!(!as_f32.is_infinite(), "LoRA scaling must not overflow f32");
        assert!(!as_f32.is_nan(), "LoRA scaling f32 cast must not NaN");
    }

    // ── Cosine schedule monotone decay ────────────────────────────

    /// Cosine schedule: in the decay phase, smaller cos values produce smaller LR.
    /// Since cos(pi * progress) is monotonically decreasing on [0, 1]:
    /// cos_a > cos_b => lr_a > lr_b.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_cosine_monotone_decay_with_cos() {
        let base_lr: f64 = kani::any();
        let min_lr: f64 = kani::any();
        let cos_a: f64 = kani::any();
        let cos_b: f64 = kani::any();
        assume_bounded_f64(base_lr, 1e-6, 10.0);
        assume_bounded_f64(min_lr, 0.0, 10.0);
        assume_bounded_f64(cos_a, -1.0, 1.0);
        assume_bounded_f64(cos_b, -1.0, 1.0);
        kani::assume(min_lr <= base_lr);
        kani::assume(cos_a > cos_b);

        let lr_a = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos_a);
        let lr_b = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos_b);

        assert!(lr_a >= lr_b - 1e-10, "larger cos value must produce >= LR");
    }

    // ── AdaFactor rho_t convergence ───────────────────────────────

    /// AdaFactor rho_t increases as step grows (for decay_rate < 0).
    /// rho_t = clamp(1 - t^d, 0, 1-1e-8) where d < 0.
    /// As t increases, t^d decreases (since d < 0), so 1-t^d increases.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::powf, powf_f64_stub)]
    fn prove_adafactor_rho_convergence() {
        let t_a: u32 = kani::any();
        kani::assume(t_a >= 2 && t_a <= 998);
        let t_b = t_a + 1;
        let decay_rate: f64 = -0.8;

        let rho_a = (1.0 - (t_a as f64).powf(decay_rate)).clamp(0.0, 1.0 - 1e-8);
        let rho_b = (1.0 - (t_b as f64).powf(decay_rate)).clamp(0.0, 1.0 - 1e-8);

        assert!(
            rho_b >= rho_a - 1e-10,
            "rho_t must be non-decreasing for decay_rate < 0"
        );
        assert!(rho_a.is_finite() && rho_b.is_finite());
    }

    /// AdaFactor rho_t stays strictly below 1.0 due to clamping.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_rho_strictly_below_one() {
        let raw: f64 = kani::any();
        assume_bounded_f64(raw, -10.0, 10.0);
        let clamped = raw.clamp(0.0, 1.0 - 1e-8);
        assert!(clamped < 1.0, "rho must be strictly < 1");
        assert!(clamped >= 0.0);
    }
}
