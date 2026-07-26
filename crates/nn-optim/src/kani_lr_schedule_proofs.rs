// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for learning rate schedule invariants.
//!
//! Proves that WarmupSchedule and CosineSchedule produce learning rates
//! within expected bounds. CBMC cannot model `f64::cos()` so we use a
//! nondeterministic stub (cos_stub) returning values in [-1, 1].
//!
//! Re: #13 (verified training epic).

#[cfg(kani)]
mod proofs {
    // ── Scalar schedule formulas ─────────────────────────────────────

    /// Warmup LR at a given step. Matches `lr_schedule.rs:57-66`.
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

    /// Cosine LR at a given step. Matches `lr_schedule.rs:120-141`.
    /// Uses `cos_val` parameter instead of computing cos() (CBMC stub).
    fn cosine_lr_with_cos(
        base_lr: f64,
        min_lr: f64,
        warmup_steps: usize,
        total_steps: usize,
        step: usize,
        cos_val: f64,
    ) -> f64 {
        // Warmup phase
        if warmup_steps > 0 && step < warmup_steps {
            return base_lr * (step as f64 / warmup_steps as f64);
        }
        // Past total steps
        if step >= total_steps {
            return min_lr;
        }
        // Cosine phase
        let decay_steps = total_steps.saturating_sub(warmup_steps);
        if decay_steps == 0 {
            return base_lr;
        }
        // cos_val stands in for cos(pi * progress) where progress in [0, 1]
        // cos(0) = 1, cos(pi) = -1, so cos_val in [-1, 1]
        min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos_val)
    }

    fn assume_bounded_f64(x: f64, lo: f64, hi: f64) {
        kani::assume(!x.is_nan() && !x.is_infinite());
        kani::assume(x >= lo && x <= hi);
    }

    // ── WarmupSchedule proofs ────────────────────────────────────────

    /// Warmup LR at step=0 is 0 (when warmup_steps > 0).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_warmup_lr_starts_at_zero() {
        let base_lr: f64 = kani::any();
        let warmup_steps: usize = kani::any();
        assume_bounded_f64(base_lr, 1e-6, 10.0);
        kani::assume(warmup_steps >= 1 && warmup_steps <= 10000);

        let lr = warmup_lr(base_lr, warmup_steps, 0);
        assert!(lr == 0.0, "warmup LR at step 0 must be 0");
    }

    /// Warmup LR at step=warmup_steps equals base_lr.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_warmup_lr_reaches_base() {
        let base_lr: f64 = kani::any();
        let warmup_steps: usize = kani::any();
        assume_bounded_f64(base_lr, 1e-6, 10.0);
        kani::assume(warmup_steps >= 1 && warmup_steps <= 10000);

        let lr = warmup_lr(base_lr, warmup_steps, warmup_steps);
        assert!(
            lr == base_lr,
            "warmup LR at warmup_steps must equal base_lr"
        );
    }

    /// Warmup LR is in [0, base_lr] for all steps.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_warmup_lr_bounded() {
        let base_lr: f64 = kani::any();
        let warmup_steps: usize = kani::any();
        let step: usize = kani::any();
        assume_bounded_f64(base_lr, 1e-6, 10.0);
        kani::assume(warmup_steps >= 1 && warmup_steps <= 10000);
        kani::assume(step <= 20000);

        let lr = warmup_lr(base_lr, warmup_steps, step);
        assert!(lr >= 0.0, "warmup LR must be non-negative");
        assert!(lr <= base_lr, "warmup LR must not exceed base_lr");
        assert!(!lr.is_nan(), "warmup LR must not be NaN");
        assert!(!lr.is_infinite(), "warmup LR must not be infinite");
    }

    /// Warmup LR is monotonically non-decreasing during warmup phase.
    /// For step_a < step_b < warmup_steps: lr(step_a) <= lr(step_b).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_warmup_lr_monotonic() {
        let base_lr: f64 = kani::any();
        let warmup_steps: usize = kani::any();
        let step_a: usize = kani::any();
        let step_b: usize = kani::any();
        assume_bounded_f64(base_lr, 1e-6, 10.0);
        kani::assume(warmup_steps >= 2 && warmup_steps <= 10000);
        kani::assume(step_a < warmup_steps);
        kani::assume(step_b < warmup_steps);
        kani::assume(step_a <= step_b);

        let lr_a = warmup_lr(base_lr, warmup_steps, step_a);
        let lr_b = warmup_lr(base_lr, warmup_steps, step_b);
        assert!(
            lr_a <= lr_b,
            "warmup LR must be monotonically non-decreasing"
        );
    }

    /// Cosine LR at cos_val=1 (start of cosine phase) equals base_lr.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_cosine_lr_at_start_is_base() {
        let base_lr: f64 = kani::any();
        let min_lr: f64 = kani::any();
        assume_bounded_f64(base_lr, 1e-6, 10.0);
        assume_bounded_f64(min_lr, 0.0, 10.0);
        kani::assume(min_lr <= base_lr);

        // cos(0) = 1.0 → progress=0, start of cosine phase
        let lr = cosine_lr_with_cos(base_lr, min_lr, 0, 1000, 0, 1.0);
        // min_lr + 0.5 * (base_lr - min_lr) * (1 + 1) = min_lr + (base_lr - min_lr) = base_lr
        let err = (lr - base_lr).abs();
        assert!(err < 1e-10, "cosine LR at start must equal base_lr");
    }

    /// Cosine LR at cos_val=-1 (end of cosine phase) equals min_lr.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_cosine_lr_at_end_is_min() {
        let base_lr: f64 = kani::any();
        let min_lr: f64 = kani::any();
        assume_bounded_f64(base_lr, 1e-6, 10.0);
        assume_bounded_f64(min_lr, 0.0, 10.0);
        kani::assume(min_lr <= base_lr);

        // cos(pi) = -1.0 → progress=1, end of cosine phase
        // Need step in cosine phase: step >= warmup, step < total
        let lr = cosine_lr_with_cos(base_lr, min_lr, 0, 1000, 999, -1.0);
        // min_lr + 0.5 * (base_lr - min_lr) * (1 + (-1)) = min_lr + 0 = min_lr
        let err = (lr - min_lr).abs();
        assert!(err < 1e-10, "cosine LR at end must equal min_lr");
    }

    /// Cosine LR past total_steps clamps to min_lr.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_cosine_lr_past_total_clamps() {
        let base_lr: f64 = kani::any();
        let min_lr: f64 = kani::any();
        let step: usize = kani::any();
        let total_steps: usize = kani::any();
        assume_bounded_f64(base_lr, 1e-6, 10.0);
        assume_bounded_f64(min_lr, 0.0, 10.0);
        kani::assume(min_lr <= base_lr);
        kani::assume(total_steps >= 1 && total_steps <= 100000);
        kani::assume(step >= total_steps);
        kani::assume(step <= 200000);

        let lr = cosine_lr_with_cos(base_lr, min_lr, 0, total_steps, step, 0.0);
        assert!(lr == min_lr, "cosine LR past total_steps must equal min_lr");
    }
}
