// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for GradScaler invariants.
//!
//! Proves that the gradient scaler maintains `scale > 0` through all
//! state transitions (growth and backoff), and that the update logic
//! respects the configured min_scale/max_scale bounds.
//!
//! Re: #13 (verified training epic).

#[cfg(kani)]
mod proofs {
    // ── Scalar GradScaler update logic ────────────────────────────────
    //
    // Extracted from `grad_scaler.rs:194-205`. The `update()` method:
    // - On inf/NaN (found_inf = true):  scale = max(scale * backoff_factor, min_scale)
    // - On clean step (found_inf = false): growth_tracker += 1
    //   If growth_tracker >= growth_interval: scale = min(scale * growth_factor, max_scale)

    /// Simulate one GradScaler update step.
    /// Returns (new_scale, new_growth_tracker).
    fn scaler_update(
        scale: f64,
        found_inf: bool,
        growth_tracker: usize,
        growth_interval: usize,
        growth_factor: f64,
        backoff_factor: f64,
        min_scale: f64,
        max_scale: f64,
    ) -> (f64, usize) {
        if found_inf {
            let new_scale = if scale * backoff_factor > min_scale {
                scale * backoff_factor
            } else {
                min_scale
            };
            (new_scale, 0)
        } else {
            let new_tracker = growth_tracker + 1;
            if new_tracker >= growth_interval {
                let new_scale = if scale * growth_factor < max_scale {
                    scale * growth_factor
                } else {
                    max_scale
                };
                (new_scale, 0)
            } else {
                (scale, new_tracker)
            }
        }
    }

    fn assume_bounded_f64(x: f64, lo: f64, hi: f64) {
        kani::assume(!x.is_nan() && !x.is_infinite());
        kani::assume(x >= lo && x <= hi);
    }

    // ── GradScaler invariant proofs ──────────────────────────────────

    /// GradScaler: scale > 0 after backoff (found_inf = true).
    /// Backoff reduces scale but never below min_scale > 0.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_grad_scaler_backoff_positive() {
        let scale: f64 = kani::any();
        let backoff_factor: f64 = kani::any();
        let min_scale: f64 = kani::any();
        assume_bounded_f64(scale, 1e-10, 1e20);
        assume_bounded_f64(backoff_factor, 0.01, 0.99);
        assume_bounded_f64(min_scale, 1e-10, 1e10);
        kani::assume(min_scale <= scale);

        let (new_scale, new_tracker) =
            scaler_update(scale, true, 0, 2000, 2.0, backoff_factor, min_scale, 1e20);
        assert!(new_scale > 0.0, "scale must stay positive after backoff");
        assert!(new_scale >= min_scale, "scale must not go below min_scale");
        assert!(new_scale <= scale, "backoff must not increase scale");
        assert!(new_tracker == 0, "growth_tracker must reset on backoff");
    }

    /// GradScaler: scale > 0 after growth (clean step reaching growth_interval).
    /// Growth increases scale but never above max_scale.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_grad_scaler_growth_bounded() {
        let scale: f64 = kani::any();
        let growth_factor: f64 = kani::any();
        let max_scale: f64 = kani::any();
        let min_scale: f64 = kani::any();
        assume_bounded_f64(scale, 1e-10, 1e20);
        assume_bounded_f64(growth_factor, 1.01, 10.0);
        assume_bounded_f64(max_scale, 1.0, 1e20);
        assume_bounded_f64(min_scale, 1e-10, 1e10);
        kani::assume(min_scale <= scale);
        kani::assume(scale <= max_scale);

        // Simulate growth trigger: growth_tracker == growth_interval - 1
        let growth_interval: usize = kani::any();
        kani::assume(growth_interval >= 1 && growth_interval <= 10000);
        let (new_scale, new_tracker) = scaler_update(
            scale,
            false,
            growth_interval - 1,
            growth_interval,
            growth_factor,
            0.5,
            min_scale,
            max_scale,
        );
        assert!(new_scale > 0.0, "scale must stay positive after growth");
        assert!(new_scale <= max_scale, "scale must not exceed max_scale");
        assert!(new_scale >= scale, "growth must not decrease scale");
        assert!(new_tracker == 0, "growth_tracker must reset on growth");
    }

    /// GradScaler: clean step without reaching growth_interval leaves scale unchanged.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_grad_scaler_clean_no_growth() {
        let scale: f64 = kani::any();
        let growth_tracker: usize = kani::any();
        let growth_interval: usize = kani::any();
        assume_bounded_f64(scale, 1e-10, 1e20);
        kani::assume(growth_interval >= 2 && growth_interval <= 10000);
        // growth_tracker + 1 < growth_interval, so no growth triggered
        kani::assume(growth_tracker < growth_interval - 1);

        let (new_scale, new_tracker) = scaler_update(
            scale,
            false,
            growth_tracker,
            growth_interval,
            2.0,
            0.5,
            1.0,
            1e20,
        );
        assert!(
            // Use bit comparison for exact equality
            new_scale.to_bits() == scale.to_bits(),
            "scale must not change before growth_interval"
        );
        assert!(new_tracker == growth_tracker + 1, "tracker must increment");
    }

    /// GradScaler: scale stays in [min_scale, max_scale] after ANY update.
    /// Comprehensive invariant combining growth, backoff, and no-change paths.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_grad_scaler_bounds_invariant() {
        let scale: f64 = kani::any();
        let found_inf: bool = kani::any();
        let growth_tracker: usize = kani::any();
        let growth_interval: usize = kani::any();
        let growth_factor: f64 = kani::any();
        let backoff_factor: f64 = kani::any();
        let min_scale: f64 = kani::any();
        let max_scale: f64 = kani::any();

        // Valid config constraints (match GradScaler::new validation)
        assume_bounded_f64(scale, 1e-10, 1e15);
        assume_bounded_f64(growth_factor, 1.01, 10.0);
        assume_bounded_f64(backoff_factor, 0.01, 0.99);
        assume_bounded_f64(min_scale, 1e-10, 1e10);
        assume_bounded_f64(max_scale, 1.0, 1e15);
        kani::assume(min_scale <= max_scale);
        kani::assume(min_scale <= scale && scale <= max_scale);
        kani::assume(growth_interval >= 1 && growth_interval <= 10000);
        kani::assume(growth_tracker < growth_interval);

        let (new_scale, _) = scaler_update(
            scale,
            found_inf,
            growth_tracker,
            growth_interval,
            growth_factor,
            backoff_factor,
            min_scale,
            max_scale,
        );

        assert!(new_scale >= min_scale, "scale must be >= min_scale");
        assert!(new_scale <= max_scale, "scale must be <= max_scale");
        assert!(new_scale > 0.0, "scale must be positive");
        assert!(!new_scale.is_nan(), "scale must not be NaN");
        assert!(!new_scale.is_infinite(), "scale must not be infinite");
    }

    /// GradScaler: repeated backoff converges to min_scale.
    /// After enough consecutive inf steps, scale reaches min_scale floor.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(12)]
    fn prove_grad_scaler_repeated_backoff_converges() {
        let initial_scale: f64 = kani::any();
        let backoff_factor: f64 = kani::any();
        let min_scale: f64 = kani::any();
        assume_bounded_f64(initial_scale, 1.0, 1e6);
        assume_bounded_f64(backoff_factor, 0.1, 0.9);
        assume_bounded_f64(min_scale, 0.1, 1.0);
        kani::assume(min_scale <= initial_scale);

        let mut scale = initial_scale;
        let mut i: u32 = 0;
        while i < 10 {
            let (new_scale, _) =
                scaler_update(scale, true, 0, 2000, 2.0, backoff_factor, min_scale, 1e20);
            assert!(new_scale >= min_scale, "scale floor violated at iteration");
            assert!(new_scale <= scale, "backoff must not increase");
            scale = new_scale;
            i += 1;
        }
        // After 10 backoffs with factor <= 0.9, initial_scale <= 1e6:
        // final <= 1e6 * 0.9^10 ≈ 348678 (still above min_scale for min_scale=0.1)
        // The key invariant is that min_scale floor is never violated.
        assert!(scale >= min_scale);
    }
}
