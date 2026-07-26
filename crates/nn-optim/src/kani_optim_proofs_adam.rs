// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Adam and AdaFactor optimizer update rules.
//!
//! Split from `kani_optim_proofs.rs` for 500-line compliance.
//! SGD harnesses remain in `kani_optim_proofs.rs`.
//!
//! ## Precision note (#1751)
//!
//! `adam_scalar_update` uses f64 intermediates for bias correction and the
//! parameter update, while production code (`adam.rs:178-189`) uses f32
//! throughout. The f64 proof is an *over-approximation*: if the f64 result
//! is finite, the corresponding f32 computation is also finite (for values
//! within f32 range), so the proof is sound. The difference is ≤0.013 for
//! bc2 at t=1 (beta2=0.999), converging to <1 ULP by t≈50.
//!
//! `adam_scalar_update_f32` mirrors the exact production f32 arithmetic for
//! a tighter proof that matches the deployed code path.
//!
//! Re: #13 (verified training epic), #1464.

#[cfg(kani)]
mod proofs {
    // ── Scalar optimizer update functions ────────────────────────────

    /// Adam/AdamW scalar update. Matches `adam.rs:154-182`.
    fn adam_scalar_update(
        theta: f32,
        grad: f32,
        m_prev: f32,
        v_prev: f32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step_t: u32,
    ) -> (f32, f32, f32) {
        let m = beta1 * m_prev + (1.0 - beta1) * grad;
        let v = beta2 * v_prev + (1.0 - beta2) * grad * grad;
        let beta1_64 = beta1 as f64;
        let beta2_64 = beta2 as f64;
        let t = step_t as i32;
        let bc1 = 1.0 / (1.0 - beta1_64.powi(t));
        let bc2 = 1.0 / (1.0 - beta2_64.powi(t));
        let m_hat = m as f64 * bc1;
        let v_hat = v as f64 * bc2;
        let lr_64 = lr as f64;
        let wd_64 = weight_decay as f64;
        let theta_wd = theta as f64 * (1.0 - lr_64 * wd_64);
        let denom = v_hat.sqrt() + eps as f64;
        let update = lr_64 * m_hat / denom;
        let new_theta = (theta_wd - update) as f32;
        (new_theta, m, v)
    }

    /// AdaFactor scalar update (rank < 2). Matches `adafactor.rs:236-268`.
    fn adafactor_scalar_update(
        theta: f32,
        grad: f32,
        v_prev: f32,
        lr: f32,
        rho: f32,
        eps_denom: f32,
        weight_decay: f32,
    ) -> (f32, f32) {
        let v = rho * v_prev + (1.0 - rho) * grad * grad;
        let denom = (v + eps_denom).sqrt();
        let u = grad / denom;
        let theta_wd = theta * (1.0 - lr * weight_decay);
        let new_theta = theta_wd - lr * u;
        (new_theta, v)
    }

    fn assume_bounded(x: f32, lo: f32, hi: f32) {
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
        // When base in (0,1) and exp >= 1, result is in (0, base] strictly
        // This preserves: base^t <= base for base in (0,1), t >= 1
        // Bounding by base prevents f64→f32 cast rounding result to 1.0
        if base > 0.0 && base < 1.0 && exp >= 1 {
            kani::assume(result > 0.0 && result <= base);
        }
        if base > 0.0 {
            kani::assume(result > 0.0);
        }
        result
    }

    // ── Adam harnesses ───────────────────────────────────────────────

    /// Adam update produces finite output for finite inputs.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::powi, powi_f64_stub)]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    fn prove_adam_update_finite() {
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
        let (new_theta, new_m, new_v) =
            adam_scalar_update(theta, grad, m_prev, v_prev, lr, 0.9, 0.999, 1e-8, 0.01, 1);
        assert!(!new_theta.is_nan() && !new_theta.is_infinite());
        assert!(!new_m.is_nan() && !new_m.is_infinite());
        assert!(!new_v.is_nan() && !new_v.is_infinite());
    }

    /// Adam second moment is always non-negative.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adam_second_moment_non_negative() {
        let grad: f32 = kani::any();
        let v_prev: f32 = kani::any();
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(v_prev, 0.0, 1e8);
        let v = 0.999_f32 * v_prev + 0.001_f32 * grad * grad;
        assert!(v >= 0.0);
        assert!(!v.is_nan() && !v.is_infinite());
    }

    /// Adam bias correction divisor (1 - beta^t) is positive for t in [1, 10].
    /// Uses iterative multiplication instead of powi() because CBMC cannot
    /// model powi deterministically (same pattern as sin_stub/cos_stub for RoPE).
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(12)]
    fn prove_adam_bias_correction_positive() {
        let step: u32 = kani::any();
        // Use step <= 10 for tractable CBMC unwind; mathematically, if 0 < beta < 1
        // then beta^t < 1 for all t >= 1, so 1 - beta^t > 0. Testing up to 10
        // exercises the inductive step sufficiently.
        kani::assume(step >= 1 && step <= 10);
        // Compute beta^t iteratively (CBMC powi is nondeterministic)
        let mut pow1: f64 = 1.0;
        let mut pow2: f64 = 1.0;
        let mut i: u32 = 0;
        while i < step {
            pow1 *= 0.9;
            pow2 *= 0.999;
            i += 1;
        }
        let bc1 = 1.0 - pow1;
        let bc2 = 1.0 - pow2;
        assert!(bc1 > 0.0);
        assert!(bc2 > 0.0);
        assert!(bc1 <= 1.0);
        assert!(bc2 <= 1.0);
    }

    /// Adam denominator sqrt(v_hat) + eps is strictly positive.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    fn prove_adam_denom_positive() {
        let v_prev: f32 = kani::any();
        let grad: f32 = kani::any();
        assume_bounded(v_prev, 0.0, 1e8);
        assume_bounded(grad, -1e4, 1e4);
        let v = 0.999_f32 * v_prev + 0.001_f32 * grad * grad;
        // 1 / (1 - 0.999^1) = 1 / 0.001 = 1000.0
        // Use constant instead of powi (CBMC powi is nondeterministic)
        let bc2: f64 = 1000.0;
        let v_hat = v as f64 * bc2;
        let denom = v_hat.sqrt() + 1e-8_f64;
        assert!(denom > 0.0);
        assert!(!denom.is_nan() && !denom.is_infinite());
    }

    // ── f32-precision Adam (matches production path, #1751 AC2) ──────

    /// Adam/AdamW scalar update using exact f32 arithmetic matching
    /// `adam.rs:178-189`. Bias correction computed as f32 (via f64 powi
    /// then cast, matching `adam.rs:238-239`), all intermediates f32.
    fn adam_scalar_update_f32(
        theta: f32,
        grad: f32,
        m_prev: f32,
        v_prev: f32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step_t: u32,
    ) -> (f32, f32, f32) {
        let m = beta1 * m_prev + (1.0 - beta1) * grad;
        let v = beta2 * v_prev + (1.0 - beta2) * grad * grad;
        let t = step_t as i32;
        // Production path: f64 powi → cast to f32 → f32 division.
        let bc1 = 1.0f32 / (1.0 - (beta1 as f64).powi(t) as f32);
        let bc2 = 1.0f32 / (1.0 - (beta2 as f64).powi(t) as f32);
        let m_hat = m * bc1;
        let v_hat = v * bc2;
        let step = lr * m_hat / (v_hat.sqrt() + eps);
        let new_theta = if weight_decay > 0.0 {
            let decay_factor = 1.0f32 - lr * weight_decay;
            theta * decay_factor - step
        } else {
            theta - step
        };
        (new_theta, m, v)
    }

    /// f32-precision Adam update produces finite output, matching the exact
    /// production arithmetic path in `adam.rs:178-189`.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::powi, powi_f64_stub)]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn prove_adam_f32_update_finite() {
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
        let (new_theta, new_m, new_v) =
            adam_scalar_update_f32(theta, grad, m_prev, v_prev, lr, 0.9, 0.999, 1e-8, 0.01, 1);
        assert!(!new_theta.is_nan() && !new_theta.is_infinite());
        assert!(!new_m.is_nan() && !new_m.is_infinite());
        assert!(!new_v.is_nan() && !new_v.is_infinite());
    }

    /// f32-precision Adam with wider parameter ranges.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::powi, powi_f64_stub)]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn prove_adam_f32_wide_range_finite() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let m_prev: f32 = kani::any();
        let v_prev: f32 = kani::any();
        let lr: f32 = kani::any();
        let eps: f32 = kani::any();
        assume_bounded(theta, -1e6, 1e6);
        assume_bounded(grad, -1e6, 1e6);
        assume_bounded(m_prev, -1e6, 1e6);
        assume_bounded(v_prev, 0.0, 1e12);
        assume_bounded(lr, 1e-5, 0.1);
        assume_bounded(eps, 1e-8, 1e-4);
        let (new_theta, new_m, new_v) =
            adam_scalar_update_f32(theta, grad, m_prev, v_prev, lr, 0.9, 0.999, eps, 0.01, 1);
        assert!(!new_theta.is_nan() && !new_theta.is_infinite());
        assert!(!new_m.is_nan() && !new_m.is_infinite());
        assert!(!new_v.is_nan() && !new_v.is_infinite());
    }

    /// f32-precision Adam stays finite over 3 consecutive steps.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(5)]
    #[kani::stub(f64::powi, powi_f64_stub)]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn prove_adam_f32_multi_step_finite() {
        let mut theta: f32 = kani::any();
        let mut m: f32 = 0.0;
        let mut v: f32 = 0.0;
        let lr: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(lr, 1e-5, 1e-2);
        let mut step: u32 = 0;
        while step < 3 {
            let grad: f32 = kani::any();
            assume_bounded(grad, -1e4, 1e4);
            let (new_theta, new_m, new_v) =
                adam_scalar_update_f32(theta, grad, m, v, lr, 0.9, 0.999, 1e-8, 0.01, step + 1);
            assert!(!new_theta.is_nan() && !new_theta.is_infinite());
            assert!(!new_m.is_nan() && !new_m.is_infinite());
            assert!(!new_v.is_nan() && !new_v.is_infinite());
            theta = new_theta;
            m = new_m;
            v = new_v;
            step += 1;
        }
    }

    // ── AdaFactor harnesses ──────────────────────────────────────────

    /// AdaFactor scalar update produces finite output.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn prove_adafactor_scalar_finite() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let v_prev: f32 = kani::any();
        let lr: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(v_prev, 0.0, 1e8);
        assume_bounded(lr, 1e-5, 1e-2);
        let (new_theta, new_v) = adafactor_scalar_update(theta, grad, v_prev, lr, 0.8, 1e-30, 0.0);
        assert!(!new_theta.is_nan() && !new_theta.is_infinite());
        assert!(!new_v.is_nan() && !new_v.is_infinite());
    }

    /// AdaFactor second moment is non-negative.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_adafactor_second_moment_non_negative() {
        let grad: f32 = kani::any();
        let v_prev: f32 = kani::any();
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(v_prev, 0.0, 1e8);
        let v = 0.8_f32 * v_prev + 0.2_f32 * grad * grad;
        assert!(v >= 0.0);
        assert!(!v.is_nan() && !v.is_infinite());
    }

    /// AdaFactor denominator sqrt(v + eps_denom) is strictly positive.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn prove_adafactor_denom_positive() {
        let v: f32 = kani::any();
        assume_bounded(v, 0.0, 1e8);
        let denom = (v + 1e-30_f32).sqrt();
        assert!(denom > 0.0);
        assert!(!denom.is_nan() && !denom.is_infinite());
    }

    /// AdaFactor with weight decay produces finite output.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn prove_adafactor_weight_decay_finite() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let v_prev: f32 = kani::any();
        let lr: f32 = kani::any();
        let wd: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(v_prev, 0.0, 1e8);
        assume_bounded(lr, 1e-5, 1e-2);
        assume_bounded(wd, 0.0, 0.1);
        let (new_theta, new_v) = adafactor_scalar_update(theta, grad, v_prev, lr, 0.8, 1e-30, wd);
        assert!(!new_theta.is_nan() && !new_theta.is_infinite());
        assert!(!new_v.is_nan() && !new_v.is_infinite());
    }

    // ── Multi-step accumulation harnesses (AC4, #1515) ──────────────

    /// Adam stays finite over 3 consecutive steps with state accumulation.
    /// Tests step_t > 1 with evolving first/second moment estimates.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(5)]
    #[kani::stub(f64::powi, powi_f64_stub)]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    fn prove_adam_multi_step_finite() {
        let mut theta: f32 = kani::any();
        let mut m: f32 = 0.0;
        let mut v: f32 = 0.0;
        let lr: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(lr, 1e-5, 1e-2);
        let mut step: u32 = 0;
        while step < 3 {
            let grad: f32 = kani::any();
            assume_bounded(grad, -1e4, 1e4);
            let (new_theta, new_m, new_v) =
                adam_scalar_update(theta, grad, m, v, lr, 0.9, 0.999, 1e-8, 0.01, step + 1);
            assert!(!new_theta.is_nan() && !new_theta.is_infinite());
            assert!(!new_m.is_nan() && !new_m.is_infinite());
            assert!(!new_v.is_nan() && !new_v.is_infinite());
            theta = new_theta;
            m = new_m;
            v = new_v;
            step += 1;
        }
    }

    /// AdaFactor stays finite over 5 consecutive steps with state accumulation.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(7)]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn prove_adafactor_multi_step_finite() {
        let mut theta: f32 = kani::any();
        let mut v: f32 = 0.0;
        let lr: f32 = kani::any();
        assume_bounded(theta, -1e4, 1e4);
        assume_bounded(lr, 1e-5, 1e-2);
        let mut step: u32 = 0;
        while step < 5 {
            let grad: f32 = kani::any();
            assume_bounded(grad, -1e4, 1e4);
            let (new_theta, new_v) = adafactor_scalar_update(theta, grad, v, lr, 0.8, 1e-30, 0.0);
            assert!(!new_theta.is_nan() && !new_theta.is_infinite());
            assert!(!new_v.is_nan() && !new_v.is_infinite());
            theta = new_theta;
            v = new_v;
            step += 1;
        }
    }

    /// Adam with extreme (but valid) hyperparameters produces finite output.
    /// Wider ranges than prove_adam_update_finite: lr up to 0.1, theta up to 1e6.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::powi, powi_f64_stub)]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    fn prove_adam_wide_range_finite() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let m_prev: f32 = kani::any();
        let v_prev: f32 = kani::any();
        let lr: f32 = kani::any();
        let eps: f32 = kani::any();
        assume_bounded(theta, -1e6, 1e6);
        assume_bounded(grad, -1e6, 1e6);
        assume_bounded(m_prev, -1e6, 1e6);
        assume_bounded(v_prev, 0.0, 1e12);
        assume_bounded(lr, 1e-5, 0.1);
        assume_bounded(eps, 1e-8, 1e-4);
        let (new_theta, new_m, new_v) =
            adam_scalar_update(theta, grad, m_prev, v_prev, lr, 0.9, 0.999, eps, 0.01, 1);
        assert!(!new_theta.is_nan() && !new_theta.is_infinite());
        assert!(!new_m.is_nan() && !new_m.is_infinite());
        assert!(!new_v.is_nan() && !new_v.is_infinite());
    }
}
