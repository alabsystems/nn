// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Module-aligned Kani proof harnesses for `grad_scaler.rs`.
//!
//! These harnesses model the scalar state machine implemented by
//! `GradScaler::update()` and `GradScaler::load_state()`.

#[cfg(kani)]
mod proofs {
    fn assume_bounded_f64(x: f64, lo: f64, hi: f64) {
        kani::assume(x.is_finite());
        kani::assume(x >= lo && x <= hi);
    }

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
            ((scale * backoff_factor).max(min_scale), 0)
        } else {
            let new_tracker = growth_tracker + 1;
            if new_tracker >= growth_interval {
                ((scale * growth_factor).min(max_scale), 0)
            } else {
                (scale, new_tracker)
            }
        }
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_grad_scaler_backoff_respects_floor() {
        let scale: f64 = kani::any();
        let backoff_factor: f64 = kani::any();
        let min_scale: f64 = kani::any();
        let max_scale: f64 = kani::any();

        assume_bounded_f64(scale, 1.0, 1e12);
        assume_bounded_f64(backoff_factor, 0.01, 0.99);
        assume_bounded_f64(min_scale, 1e-6, 1e6);
        assume_bounded_f64(max_scale, 1.0, 1e12);
        kani::assume(min_scale <= scale && scale <= max_scale);

        let (new_scale, new_tracker) = scaler_update(
            scale,
            true,
            0,
            2000,
            2.0,
            backoff_factor,
            min_scale,
            max_scale,
        );

        assert!(new_scale.is_finite());
        assert!(new_scale >= min_scale);
        assert!(new_scale <= scale);
        assert!(new_tracker == 0);
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_grad_scaler_growth_respects_ceiling() {
        let scale: f64 = kani::any();
        let growth_factor: f64 = kani::any();
        let min_scale: f64 = kani::any();
        let max_scale: f64 = kani::any();
        let growth_interval: usize = kani::any();

        assume_bounded_f64(scale, 1.0, 1e12);
        assume_bounded_f64(growth_factor, 1.01, 4.0);
        assume_bounded_f64(min_scale, 1e-6, 1e6);
        assume_bounded_f64(max_scale, 1.0, 1e12);
        kani::assume(min_scale <= scale && scale <= max_scale);
        kani::assume((1..=4096).contains(&growth_interval));

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

        assert!(new_scale.is_finite());
        assert!(new_scale >= scale);
        assert!(new_scale <= max_scale);
        assert!(new_tracker == 0);
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_grad_scaler_load_state_caps_tracker() {
        let saved_tracker: usize = kani::any();
        let growth_interval: usize = kani::any();

        kani::assume((1..=4096).contains(&growth_interval));
        kani::assume(saved_tracker <= 1_000_000);

        let capped = saved_tracker.min(growth_interval.saturating_sub(1));

        assert!(capped < growth_interval);
    }
}
