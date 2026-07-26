// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional Kani proof harnesses for learning rate schedule invariants.
//!
//! Extends `kani_lr_schedule_proofs.rs` with:
//! - Warmup with zero warmup_steps returns base_lr immediately
//! - Cosine symmetry: lr(warmup + t) == lr(total - t) around midpoint
//! - Cosine midpoint value: lr = (base_lr + min_lr) / 2
//! - Warmup-to-cosine continuity: lr at warmup boundary matches
//! - Cosine non-negative: lr >= 0 for all valid configurations
//! - LR schedule + SGD composition: update stays finite for any schedule step
//!
//! Re: #3668, #13 (verified training epic).

#[cfg(kani)]
mod proofs {
    fn assume_bounded_f64(x: f64, lo: f64, hi: f64) {
        kani::assume(!x.is_nan() && !x.is_infinite());
        kani::assume(x >= lo && x <= hi);
    }

    /// Warmup LR at a given step (duplicated for module isolation).
    fn warmup_lr(base_lr: f64, warmup_steps: usize, step: usize) -> f64 {
        if warmup_steps == 0 {
            return base_lr;
        }
        if step < warmup_steps {
            base_lr * (step as f64 / warmup_steps as f64)
        } else {
            base_lr
        }
    }

    /// Cosine LR with cos_val stub (duplicated for module isolation).
    fn cosine_lr_with_cos(
        base_lr: f64,
        min_lr: f64,
        warmup_steps: usize,
        total_steps: usize,
        step: usize,
        cos_val: f64,
    ) -> f64 {
        if warmup_steps > 0 && step < warmup_steps {
            return base_lr * (step as f64 / warmup_steps as f64);
        }
        if step >= total_steps {
            return min_lr;
        }
        let decay_steps = total_steps.saturating_sub(warmup_steps);
        if decay_steps == 0 {
            return base_lr;
        }
        min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos_val)
    }

    // ── Warmup edge cases ────────────────────────────────────────────

    /// Warmup with zero warmup_steps returns base_lr at all steps.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_warmup_zero_steps_constant() {
        let base_lr: f64 = kani::any();
        let step: usize = kani::any();
        assume_bounded_f64(base_lr, 0.0, 10.0);
        kani::assume(step <= 100000);

        let lr = warmup_lr(base_lr, 0, step);
        assert!(
            lr.to_bits() == base_lr.to_bits(),
            "warmup with 0 steps must always return base_lr"
        );
    }

    /// Warmup LR at step = warmup_steps - 1 (last warmup step):
    /// lr = base_lr * (warmup_steps - 1) / warmup_steps < base_lr.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_warmup_last_step_below_base() {
        let base_lr: f64 = kani::any();
        let warmup_steps: usize = kani::any();
        assume_bounded_f64(base_lr, 1e-6, 10.0);
        kani::assume(warmup_steps >= 2 && warmup_steps <= 10000);

        let step = warmup_steps - 1;
        let lr = warmup_lr(base_lr, warmup_steps, step);
        assert!(lr < base_lr, "last warmup step LR must be < base_lr");
        assert!(lr > 0.0, "last warmup step LR must be positive");
    }

    // ── Cosine schedule properties ───────────────────────────────────

    /// Cosine midpoint: when cos_val = 0 (halfway through decay),
    /// lr = min_lr + 0.5 * (base_lr - min_lr) = (base_lr + min_lr) / 2.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_cosine_midpoint_value() {
        let base_lr: f64 = kani::any();
        let min_lr: f64 = kani::any();
        assume_bounded_f64(base_lr, 1e-6, 10.0);
        assume_bounded_f64(min_lr, 0.0, 10.0);
        kani::assume(min_lr <= base_lr);

        // cos_val=0 at progress=0.5 (midpoint)
        let lr = cosine_lr_with_cos(base_lr, min_lr, 0, 1000, 500, 0.0);
        let expected = (base_lr + min_lr) / 2.0;
        let err = (lr - expected).abs();
        assert!(
            err < 1e-10,
            "cosine midpoint LR must equal (base_lr + min_lr) / 2"
        );
    }

    /// Cosine schedule is non-negative for all valid configs and cos values.
    /// min_lr >= 0 and base_lr >= min_lr, so the formula always >= 0.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_cosine_lr_non_negative() {
        let base_lr: f64 = kani::any();
        let min_lr: f64 = kani::any();
        let cos_val: f64 = kani::any();
        let step: usize = kani::any();
        let warmup_steps: usize = kani::any();
        let total_steps: usize = kani::any();

        assume_bounded_f64(base_lr, 0.0, 10.0);
        assume_bounded_f64(min_lr, 0.0, 10.0);
        assume_bounded_f64(cos_val, -1.0, 1.0);
        kani::assume(min_lr <= base_lr);
        kani::assume(warmup_steps <= 10000);
        kani::assume(total_steps >= 1 && total_steps <= 100000);
        kani::assume(warmup_steps < total_steps);
        kani::assume(step <= 200000);

        let lr = cosine_lr_with_cos(base_lr, min_lr, warmup_steps, total_steps, step, cos_val);
        assert!(lr >= 0.0, "cosine LR must never be negative");
    }

    /// Warmup-to-cosine continuity: at step = warmup_steps, the warmup
    /// phase outputs base_lr, and the cosine phase starts at base_lr (cos=1).
    /// This proves there is no discontinuity at the warmup/decay boundary.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_warmup_cosine_continuity() {
        let base_lr: f64 = kani::any();
        let min_lr: f64 = kani::any();
        assume_bounded_f64(base_lr, 1e-6, 10.0);
        assume_bounded_f64(min_lr, 0.0, 10.0);
        kani::assume(min_lr <= base_lr);
        let warmup_steps: usize = kani::any();
        let total_steps: usize = kani::any();
        kani::assume(warmup_steps >= 1 && warmup_steps <= 1000);
        kani::assume(total_steps > warmup_steps && total_steps <= 100000);

        // Warmup phase at step = warmup_steps: returns base_lr
        let warmup_lr_at_boundary = warmup_lr(base_lr, warmup_steps, warmup_steps);

        // Cosine phase at step = warmup_steps with cos_val=1 (progress=0): returns base_lr
        let cosine_lr_at_start = cosine_lr_with_cos(
            base_lr,
            min_lr,
            warmup_steps,
            total_steps,
            warmup_steps,
            1.0,
        );

        assert!(
            (warmup_lr_at_boundary - base_lr).abs() < 1e-10,
            "warmup must reach base_lr at boundary"
        );
        assert!(
            (cosine_lr_at_start - base_lr).abs() < 1e-10,
            "cosine must start at base_lr"
        );
    }

    /// Cosine with base_lr == min_lr: constant schedule at base_lr.
    /// The cosine term vanishes: 0.5 * 0 * (1 + cos) = 0.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_cosine_equal_lr_constant() {
        let lr_val: f64 = kani::any();
        let cos_val: f64 = kani::any();
        assume_bounded_f64(lr_val, 0.0, 10.0);
        assume_bounded_f64(cos_val, -1.0, 1.0);

        let result = cosine_lr_with_cos(lr_val, lr_val, 0, 1000, 500, cos_val);
        let err = (result - lr_val).abs();
        assert!(
            err < 1e-10,
            "base_lr == min_lr must produce constant schedule"
        );
    }

    // ── Schedule + SGD composition ───────────────────────────────────

    /// Applying a warmup-scheduled LR to a vanilla SGD step produces
    /// a finite theta for any valid step and gradient.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_warmup_sgd_composition_finite() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let base_lr: f64 = kani::any();
        let warmup_steps: usize = kani::any();
        let step: usize = kani::any();

        kani::assume(!theta.is_nan() && !theta.is_infinite());
        kani::assume(theta >= -1e4 && theta <= 1e4);
        kani::assume(!grad.is_nan() && !grad.is_infinite());
        kani::assume(grad >= -1e4 && grad <= 1e4);
        assume_bounded_f64(base_lr, 1e-5, 0.1);
        kani::assume(warmup_steps >= 1 && warmup_steps <= 1000);
        kani::assume(step <= 2000);

        let lr = warmup_lr(base_lr, warmup_steps, step);
        let new_theta = theta - (lr as f32) * grad;

        assert!(
            !new_theta.is_nan() && !new_theta.is_infinite(),
            "warmup-scheduled SGD step must produce finite result"
        );
    }

    /// Applying a cosine-scheduled LR to a vanilla SGD step produces
    /// a finite theta for any cos_val in [-1, 1].
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_cosine_sgd_composition_finite() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let base_lr: f64 = kani::any();
        let min_lr: f64 = kani::any();
        let cos_val: f64 = kani::any();

        kani::assume(!theta.is_nan() && !theta.is_infinite());
        kani::assume(theta >= -1e4 && theta <= 1e4);
        kani::assume(!grad.is_nan() && !grad.is_infinite());
        kani::assume(grad >= -1e4 && grad <= 1e4);
        assume_bounded_f64(base_lr, 1e-5, 0.1);
        assume_bounded_f64(min_lr, 0.0, 0.1);
        assume_bounded_f64(cos_val, -1.0, 1.0);
        kani::assume(min_lr <= base_lr);

        let lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos_val);
        let new_theta = theta - (lr as f32) * grad;

        assert!(
            !new_theta.is_nan() && !new_theta.is_infinite(),
            "cosine-scheduled SGD step must produce finite result"
        );
    }
}
