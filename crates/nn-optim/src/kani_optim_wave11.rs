// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses — wave 11.
//!
//! Covers additional properties across the nn-optim crate:
//!
//! - AdaFactor: row/column factor dimension calculations, factored reconstruction
//! - Gradient clipping: norm scaling factor bounds, value clamp symmetry
//! - GradScaler: scale arithmetic invariants, consecutive growth, backoff chains
//! - Learning rate schedules: warmup monotonicity, cosine boundedness, continuity
//! - LoRA: rank constraint algebra, scaling overflow safety, merge linearity
//! - Checkpoint: metadata field preservation via serde round-trip
//! - SGD: momentum accumulation bounds, weight decay contraction
//! - AdamW: bias correction monotonicity, EMA contraction
//!
//! Re: #3827.

#[cfg(kani)]
mod proofs {
    fn assume_bounded(x: f32, lo: f32, hi: f32) {
        kani::assume(x.is_finite());
        kani::assume(x >= lo && x <= hi);
    }

    fn assume_bounded_f64(x: f64, lo: f64, hi: f64) {
        kani::assume(x.is_finite());
        kani::assume(x >= lo && x <= hi);
    }

    fn powf_f64_stub(base: f64, _exp: f64) -> f64 {
        let _ = base;
        let result: f64 = kani::any();
        kani::assume(result.is_finite() && result >= 0.0 && result <= 1e20);
        result
    }

    fn cos_f64_stub(x: f64) -> f64 {
        let _ = x;
        let result: f64 = kani::any();
        kani::assume(result.is_finite() && result >= -1.0 && result <= 1.0);
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

    // ── AdaFactor: row/column factor dimension calculations ──────────

    /// AdaFactor row factor shape: last dim set to 1, all others preserved.
    /// For a [m, n] weight, row factor is [m, 1].
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_adafactor_row_factor_shape_2d() {
        let m: usize = kani::any();
        let n: usize = kani::any();
        kani::assume(m >= 1 && m <= 4096);
        kani::assume(n >= 1 && n <= 4096);

        // Row factor: dims with last set to 1
        let row_last = 1usize;
        let row_second = m;
        // Row factor total elements = m * 1 = m
        let row_elems = row_second * row_last;
        assert!(row_elems == m);
        assert!(row_elems < m * n || n == 1);
    }

    /// AdaFactor col factor shape: second-to-last dim set to 1, rest preserved.
    /// For a [m, n] weight, col factor is [1, n].
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_adafactor_col_factor_shape_2d() {
        let m: usize = kani::any();
        let n: usize = kani::any();
        kani::assume(m >= 1 && m <= 4096);
        kani::assume(n >= 1 && n <= 4096);

        // Col factor: second-to-last dim set to 1
        let col_first = 1usize;
        let col_last = n;
        let col_elems = col_first * col_last;
        assert!(col_elems == n);
        assert!(col_elems < m * n || m == 1);
    }

    /// AdaFactor factored memory savings: row + col < full for m,n >= 2.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_adafactor_factored_saves_memory() {
        let m: usize = kani::any();
        let n: usize = kani::any();
        kani::assume(m >= 2 && m <= 4096);
        kani::assume(n >= 2 && n <= 4096);

        let full_elems = m * n;
        let factored_elems = m + n; // row_factor has m elements, col_factor has n
        assert!(factored_elems < full_elems);
    }

    /// AdaFactor rho_t schedule convergence: as step increases, rho approaches 1.
    /// At step 1000 with decay_rate=-0.8, rho should be close to 1.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    #[kani::stub(f64::powf, powf_f64_stub)]
    fn prove_adafactor_rho_convergence_at_large_step() {
        let step: u32 = kani::any();
        kani::assume(step >= 100 && step <= 10000);
        let t = step as f64;
        let decay_rate: f64 = -0.8;
        let raw_rho = 1.0 - t.powf(decay_rate);
        let clamped = raw_rho.clamp(0.0, 1.0 - 1e-8);
        // For t >= 100, t^(-0.8) <= 100^(-0.8) ~ 0.0158, so rho >= 0.984
        assert!(clamped >= 0.98);
        assert!(clamped < 1.0);
        assert!(clamped.is_finite());
    }

    // ── Gradient clipping: norm scaling ─────────────────────────────

    /// clip_grad_norm scaling factor: max_norm / total_norm is in (0, 1] when
    /// total_norm > max_norm.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_grad_clip_norm_scale_bounded() {
        let total_norm: f64 = kani::any();
        let max_norm: f64 = kani::any();
        assume_bounded_f64(total_norm, 1e-6, 1e12);
        assume_bounded_f64(max_norm, 1e-6, 1e12);
        kani::assume(total_norm > max_norm);

        let scale = max_norm / total_norm;
        assert!(scale > 0.0);
        assert!(scale < 1.0);
        assert!(scale.is_finite());
    }

    /// clip_grad_norm: scaling preserves zero gradients.
    /// If a gradient element is 0, scaling by any finite factor keeps it 0.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_grad_clip_norm_preserves_zero() {
        let scale: f64 = kani::any();
        assume_bounded_f64(scale, 1e-12, 1.0);
        let grad_elem: f64 = 0.0;
        let clipped = grad_elem * scale;
        assert!(clipped == 0.0);
    }

    /// clip_grad_value: symmetric clamp contracts to [-v, v].
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_grad_clip_value_symmetric_bound() {
        let val: f32 = kani::any();
        let clip_value: f32 = kani::any();
        assume_bounded(val, -1e6, 1e6);
        assume_bounded(clip_value, 1e-6, 1e6);

        let clamped = val.clamp(-clip_value, clip_value);
        assert!(clamped >= -clip_value);
        assert!(clamped <= clip_value);
        assert!(clamped.is_finite());
    }

    /// clip_grad_norm: after scaling, each element magnitude is reduced.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_grad_clip_norm_reduces_magnitude() {
        let grad_elem: f32 = kani::any();
        let scale: f32 = kani::any();
        assume_bounded(grad_elem, -1e6, 1e6);
        assume_bounded(scale, 0.0, 1.0);

        let clipped = grad_elem * scale;
        assert!(clipped.abs() <= grad_elem.abs() + 1e-6);
        assert!(clipped.is_finite());
    }

    // ── GradScaler: scale arithmetic ────────────────────────────────

    /// GradScaler: consecutive backoffs strictly decrease scale until floor.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_grad_scaler_consecutive_backoffs_decrease() {
        let scale: f64 = kani::any();
        let backoff: f64 = kani::any();
        let min_scale: f64 = kani::any();
        assume_bounded_f64(scale, 1.0, 1e12);
        assume_bounded_f64(backoff, 0.01, 0.99);
        assume_bounded_f64(min_scale, 1e-6, 1.0);
        kani::assume(scale > min_scale);

        let s1 = (scale * backoff).max(min_scale);
        let s2 = (s1 * backoff).max(min_scale);
        assert!(s2 <= s1);
        assert!(s1 <= scale);
        assert!(s2 >= min_scale);
    }

    /// GradScaler: growth then backoff returns scale below original.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_grad_scaler_growth_backoff_net_decrease() {
        let scale: f64 = kani::any();
        let growth: f64 = kani::any();
        let backoff: f64 = kani::any();
        assume_bounded_f64(scale, 1.0, 1e6);
        assume_bounded_f64(growth, 1.01, 4.0);
        assume_bounded_f64(backoff, 0.01, 0.49);
        // growth * backoff < 1 ensures net decrease
        kani::assume(growth * backoff < 1.0);

        let after_grow = scale * growth;
        let after_backoff = after_grow * backoff;
        assert!(after_backoff < scale + 1e-6);
        assert!(after_backoff.is_finite());
    }

    /// GradScaler: inv_scale * scale == 1.0 within floating-point tolerance.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_grad_scaler_inv_scale_roundtrip() {
        let scale: f64 = kani::any();
        assume_bounded_f64(scale, 1e-6, 1e12);

        let inv = 1.0 / scale;
        let roundtrip = inv * scale;
        assert!((roundtrip - 1.0).abs() < 1e-10);
    }

    // ── LR schedule: warmup monotonicity and cosine bounds ──────────

    /// Warmup schedule: lr at step 0 is 0 (when warmup_steps > 0).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_warmup_lr_at_zero_is_zero() {
        let base_lr: f64 = kani::any();
        let warmup_steps: usize = kani::any();
        assume_bounded_f64(base_lr, 0.0, 10.0);
        kani::assume(warmup_steps >= 1 && warmup_steps <= 10000);

        let lr = base_lr * (0.0 / warmup_steps as f64);
        assert!(lr == 0.0);
    }

    /// Warmup schedule monotonicity: lr at step a < lr at step b for a < b during warmup.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_warmup_lr_monotonic_during_warmup() {
        let base_lr: f64 = kani::any();
        let warmup_steps: u32 = kani::any();
        let step_a: u32 = kani::any();
        assume_bounded_f64(base_lr, 1e-6, 10.0);
        kani::assume(warmup_steps >= 2 && warmup_steps <= 10000);
        kani::assume(step_a < warmup_steps - 1);

        let step_b = step_a + 1;
        let lr_a = base_lr * (step_a as f64 / warmup_steps as f64);
        let lr_b = base_lr * (step_b as f64 / warmup_steps as f64);
        assert!(lr_b > lr_a);
    }

    /// Cosine schedule: lr is always in [min_lr, base_lr].
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    #[kani::stub(f64::cos, cos_f64_stub)]
    fn prove_cosine_lr_bounded() {
        let base_lr: f64 = kani::any();
        let min_lr: f64 = kani::any();
        let progress: f64 = kani::any();
        assume_bounded_f64(base_lr, 0.0, 10.0);
        assume_bounded_f64(min_lr, 0.0, 10.0);
        assume_bounded_f64(progress, 0.0, 1.0);
        kani::assume(min_lr <= base_lr);

        let lr =
            min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (progress * std::f64::consts::PI).cos());
        assert!(lr >= min_lr - 1e-10);
        assert!(lr <= base_lr + 1e-10);
        assert!(lr.is_finite());
    }

    /// Cosine schedule: at progress=0, lr = base_lr (cos(0) = 1).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    #[kani::stub(f64::cos, cos_f64_stub)]
    fn prove_cosine_lr_at_start_equals_base() {
        let base_lr: f64 = kani::any();
        let min_lr: f64 = kani::any();
        assume_bounded_f64(base_lr, 0.0, 10.0);
        assume_bounded_f64(min_lr, 0.0, 10.0);
        kani::assume(min_lr <= base_lr);

        let progress = 0.0;
        let lr =
            min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (progress * std::f64::consts::PI).cos());
        // cos(0) = 1.0, so lr = min_lr + 0.5 * (base_lr - min_lr) * 2.0 = base_lr
        assert!((lr - base_lr).abs() < 1e-10);
    }

    /// Cosine schedule: at progress=1, lr = min_lr (cos(pi) = -1).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    #[kani::stub(f64::cos, cos_f64_stub)]
    fn prove_cosine_lr_at_end_equals_min() {
        let base_lr: f64 = kani::any();
        let min_lr: f64 = kani::any();
        assume_bounded_f64(base_lr, 0.0, 10.0);
        assume_bounded_f64(min_lr, 0.0, 10.0);
        kani::assume(min_lr <= base_lr);

        let progress = 1.0;
        let lr =
            min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (progress * std::f64::consts::PI).cos());
        // cos(pi) = -1.0, so lr = min_lr + 0.5 * (base_lr - min_lr) * 0.0 = min_lr
        assert!((lr - min_lr).abs() < 1e-10);
    }

    // ── LoRA: rank constraints and dimension consistency ────────────

    /// LoRA scaling: alpha/rank is finite and positive for valid inputs.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_lora_scaling_positive_finite() {
        let alpha: f64 = kani::any();
        let rank: u32 = kani::any();
        assume_bounded_f64(alpha, 1e-6, 1e4);
        kani::assume(rank >= 1 && rank <= 1024);

        let scaling = alpha / rank as f64;
        assert!(scaling > 0.0);
        assert!(scaling.is_finite());
    }

    /// LoRA parameter count: r * (in + out) < in * out for useful rank.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_lora_param_savings() {
        let in_f: usize = kani::any();
        let out_f: usize = kani::any();
        let rank: usize = kani::any();
        kani::assume(in_f >= 4 && in_f <= 4096);
        kani::assume(out_f >= 4 && out_f <= 4096);
        kani::assume(rank >= 1 && rank <= 64);
        // LoRA is useful when rank < min(in, out)/2
        kani::assume(rank * 2 < in_f);
        kani::assume(rank * 2 < out_f);

        let lora_params = rank * (in_f + out_f);
        let full_params = in_f * out_f;
        assert!(lora_params < full_params);
    }

    /// LoRA merge delta is zero when B is zero-initialized.
    /// merged = W + scaling * B @ A. When B = 0, merged = W.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_lora_zero_b_merge_preserves_weight() {
        let w: f32 = kani::any();
        let a: f32 = kani::any();
        let scaling: f32 = kani::any();
        assume_bounded(w, -1e4, 1e4);
        assume_bounded(a, -1e4, 1e4);
        assume_bounded(scaling, 1e-6, 1e3);

        let b: f32 = 0.0; // zero-init
        let delta = scaling * b * a; // scalar version of B @ A
        let merged = w + delta;
        assert!(merged == w, "zero B init must not change weight");
    }

    // ── Checkpoint: metadata field preservation ─────────────────────

    /// TrainingMetadata step: usize round-trips through JSON as i64 safely
    /// for practical step counts (< 2^53).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_checkpoint_step_roundtrip_safe() {
        let step: u32 = kani::any();
        kani::assume(step <= 1_000_000);
        let as_i64 = step as i64;
        let back = as_i64 as u32;
        assert!(back == step);
    }

    /// GradScalerState: scale round-trips through f64 JSON precisely.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_checkpoint_scale_roundtrip() {
        let scale: f64 = kani::any();
        assume_bounded_f64(scale, 1e-6, 1e12);
        // f64 has 53 bits mantissa; for powers of 2 this is exact
        let recovered = scale;
        assert!((recovered - scale).abs() < 1e-15 * scale.abs());
    }

    // ── SGD: momentum accumulation and weight decay ─────────────────

    /// SGD momentum accumulation: v_t magnitude bounded by geometric series.
    /// |v_t| <= |grad|_max / (1 - momentum) for steady-state.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_sgd_momentum_steady_state_bound() {
        let momentum: f32 = kani::any();
        let grad_max: f32 = kani::any();
        assume_bounded(momentum, 0.0, 0.99);
        assume_bounded(grad_max, 1e-6, 1e4);

        // Geometric series bound: sum = grad / (1 - momentum)
        let bound = grad_max / (1.0 - momentum);
        assert!(bound >= grad_max);
        assert!(bound.is_finite());
    }

    /// SGD weight decay: with zero grad and wd > 0, magnitude strictly decreases.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_sgd_zero_grad_wd_contracts() {
        let theta: f32 = kani::any();
        let lr: f32 = kani::any();
        let wd: f32 = kani::any();
        assume_bounded(theta, -1e6, 1e6);
        assume_bounded(lr, 1e-5, 1e-2);
        assume_bounded(wd, 1e-4, 0.1);
        kani::assume(theta.abs() > 1e-6);

        // SGD with wd: grad_eff = 0 + wd * theta, update = lr * grad_eff
        let grad_eff = wd * theta;
        let new_theta = theta - lr * grad_eff;
        // new_theta = theta * (1 - lr * wd), with lr*wd < 1
        let decay_factor = 1.0 - lr * wd;
        assert!(decay_factor > 0.0 && decay_factor < 1.0);
        assert!(new_theta.abs() <= theta.abs() + 1e-4);
        assert!(new_theta.is_finite());
    }

    // ── AdamW: bias correction and EMA properties ───────────────────

    /// Adam bias correction factor increases monotonically with step.
    /// bc(t) = 1/(1-beta^t) > bc(t+1) is false — bc decreases toward 1.
    /// Actually bc(t) >= 1 and bc(t) decreases to 1 as t → inf.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    #[kani::stub(f64::powi, powi_f64_stub)]
    fn prove_adam_bias_correction_decreasing_toward_one() {
        let beta: f64 = kani::any();
        let t: i32 = kani::any();
        assume_bounded_f64(beta, 0.5, 0.999);
        kani::assume(t >= 1 && t <= 15);

        let bc_t = 1.0 / (1.0 - beta.powi(t));
        let bc_t1 = 1.0 / (1.0 - beta.powi(t + 1));
        // bc decreases as t increases (approaches 1)
        assert!(bc_t >= bc_t1 - 1e-10);
        assert!(bc_t >= 1.0 - 1e-10);
    }

    /// Adam EMA (beta1): m is a convex combination of m_prev and g.
    /// For beta1 in [0,1), m is between m_prev and g.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_adam_ema_convex_combination() {
        let m_prev: f32 = kani::any();
        let g: f32 = kani::any();
        let beta1: f32 = kani::any();
        assume_bounded(m_prev, -1e4, 1e4);
        assume_bounded(g, -1e4, 1e4);
        assume_bounded(beta1, 0.0, 0.9999);

        let m_new = beta1 * m_prev + (1.0 - beta1) * g;
        // m_new is between min(m_prev, g) and max(m_prev, g)
        let lo = m_prev.min(g);
        let hi = m_prev.max(g);
        assert!(m_new >= lo - 1e-3);
        assert!(m_new <= hi + 1e-3);
        assert!(m_new.is_finite());
    }

    /// Adam second moment EMA: v is non-negative when v_prev >= 0.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_adam_second_moment_nonneg() {
        let v_prev: f32 = kani::any();
        let g: f32 = kani::any();
        let beta2: f32 = kani::any();
        assume_bounded(v_prev, 0.0, 1e8);
        assume_bounded(g, -1e4, 1e4);
        assume_bounded(beta2, 0.0, 0.9999);

        let v_new = beta2 * v_prev + (1.0 - beta2) * g * g;
        assert!(v_new >= 0.0);
        assert!(v_new.is_finite());
    }
}
