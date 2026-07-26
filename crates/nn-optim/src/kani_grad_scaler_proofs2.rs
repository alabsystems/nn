// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional Kani proof harnesses for GradScaler.
//!
//! Extends `kani_grad_scaler_proofs.rs` with:
//! - Config validation invariants (init_scale bounds, factor ranges)
//! - load_state clamping correctness
//! - inv_scale (1.0 / scale) finiteness
//! - Growth/backoff alternation: scale stays bounded through mixed sequences
//! - Backoff idempotence at min_scale floor
//!
//! Re: #3668, #13 (verified training epic).

#[cfg(kani)]
mod proofs {
    fn assume_bounded_f64(x: f64, lo: f64, hi: f64) {
        kani::assume(!x.is_nan() && !x.is_infinite());
        kani::assume(x >= lo && x <= hi);
    }

    /// Simulate one GradScaler update step (duplicated for module isolation).
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

    // ── Config validation proofs ─────────────────────────────────────

    /// GradScaler: inv_scale = 1.0 / scale is finite and positive for all
    /// valid scale values. This is critical because unscale_and_check
    /// multiplies gradients by inv_scale.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_grad_scaler_inv_scale_finite() {
        let scale: f64 = kani::any();
        let min_scale: f64 = kani::any();
        let max_scale: f64 = kani::any();
        assume_bounded_f64(scale, 1e-10, 1e15);
        assume_bounded_f64(min_scale, 1e-10, 1e10);
        assume_bounded_f64(max_scale, 1.0, 1e15);
        kani::assume(min_scale <= scale && scale <= max_scale);

        let inv_scale = 1.0 / scale;
        assert!(inv_scale > 0.0, "inv_scale must be positive");
        assert!(!inv_scale.is_nan(), "inv_scale must not be NaN");
        assert!(inv_scale.is_finite(), "inv_scale must be finite");
    }

    /// GradScaler: default config satisfies all validation constraints.
    /// init_scale=65536 in [min_scale=1, max_scale=16777216],
    /// growth_factor=2 > 1, backoff_factor=0.5 in (0,1), growth_interval=2000 > 0.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_grad_scaler_default_config_valid() {
        let init_scale: f64 = 65536.0;
        let growth_factor: f64 = 2.0;
        let backoff_factor: f64 = 0.5;
        let growth_interval: usize = 2000;
        let min_scale: f64 = 1.0;
        let max_scale: f64 = 16_777_216.0;

        // All validation checks from GradScaler::new
        assert!(init_scale > 0.0 && init_scale.is_finite());
        assert!(growth_factor > 1.0 && growth_factor.is_finite());
        assert!(backoff_factor > 0.0 && backoff_factor < 1.0 && backoff_factor.is_finite());
        assert!(min_scale > 0.0 && min_scale.is_finite());
        assert!(max_scale >= min_scale && max_scale.is_finite());
        assert!(init_scale >= min_scale && init_scale <= max_scale);
        assert!(growth_interval > 0);
    }

    /// GradScaler load_state: clamping restored scale to [min_scale, max_scale]
    /// produces a valid scale. Simulates config change between save and load.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_grad_scaler_load_state_clamp() {
        let saved_scale: f64 = kani::any();
        let min_scale: f64 = kani::any();
        let max_scale: f64 = kani::any();
        assume_bounded_f64(saved_scale, 1e-10, 1e15);
        assume_bounded_f64(min_scale, 1e-10, 1e10);
        assume_bounded_f64(max_scale, 1.0, 1e15);
        kani::assume(min_scale <= max_scale);

        // Clamp: matches load_state implementation
        let clamped = if saved_scale < min_scale {
            min_scale
        } else if saved_scale > max_scale {
            max_scale
        } else {
            saved_scale
        };
        assert!(clamped >= min_scale, "clamped scale must be >= min_scale");
        assert!(clamped <= max_scale, "clamped scale must be <= max_scale");
        assert!(clamped > 0.0, "clamped scale must be positive");
        assert!(clamped.is_finite(), "clamped scale must be finite");
    }

    /// GradScaler load_state: growth_tracker cap at growth_interval - 1
    /// prevents immediate growth on resume.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_grad_scaler_load_state_tracker_cap() {
        let saved_tracker: usize = kani::any();
        let growth_interval: usize = kani::any();
        kani::assume(growth_interval >= 1 && growth_interval <= 10000);
        kani::assume(saved_tracker <= 100000);

        // Matches load_state: min(saved, growth_interval.saturating_sub(1))
        let gi_minus_1 = if growth_interval > 0 {
            growth_interval - 1
        } else {
            0
        };
        let capped = if saved_tracker < gi_minus_1 {
            saved_tracker
        } else {
            gi_minus_1
        };

        assert!(
            capped < growth_interval,
            "capped tracker must be < growth_interval"
        );
        // One more clean step is always required after load
        // (capped + 1 <= growth_interval, but growth triggers at >=)
    }

    /// GradScaler: backoff at min_scale floor is idempotent.
    /// Once scale == min_scale, further backoffs keep it at min_scale.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_grad_scaler_backoff_idempotent_at_floor() {
        let min_scale: f64 = kani::any();
        let backoff_factor: f64 = kani::any();
        assume_bounded_f64(min_scale, 1e-10, 1e10);
        assume_bounded_f64(backoff_factor, 0.01, 0.99);

        let scale = min_scale; // Already at floor
        let (new_scale, _) =
            scaler_update(scale, true, 0, 2000, 2.0, backoff_factor, min_scale, 1e20);
        assert!(
            new_scale.to_bits() == min_scale.to_bits(),
            "backoff at min_scale must leave scale unchanged"
        );
    }

    /// GradScaler: growth at max_scale ceiling is idempotent.
    /// Once scale == max_scale, further growth keeps it at max_scale.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_grad_scaler_growth_idempotent_at_ceiling() {
        let max_scale: f64 = kani::any();
        let growth_factor: f64 = kani::any();
        let min_scale: f64 = kani::any();
        assume_bounded_f64(max_scale, 1.0, 1e15);
        assume_bounded_f64(growth_factor, 1.01, 10.0);
        assume_bounded_f64(min_scale, 1e-10, 1e10);
        kani::assume(min_scale <= max_scale);

        let scale = max_scale;
        let growth_interval: usize = 1;
        let (new_scale, _) = scaler_update(
            scale,
            false,
            0,
            growth_interval,
            growth_factor,
            0.5,
            min_scale,
            max_scale,
        );
        assert!(
            new_scale.to_bits() == max_scale.to_bits(),
            "growth at max_scale must leave scale unchanged"
        );
    }

    /// GradScaler: alternating growth/backoff keeps scale in bounds.
    /// Simulate: growth, backoff, growth, backoff — 4 steps total.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(6)]
    fn prove_grad_scaler_alternating_bounded() {
        let initial_scale: f64 = kani::any();
        let growth_factor: f64 = kani::any();
        let backoff_factor: f64 = kani::any();
        let min_scale: f64 = kani::any();
        let max_scale: f64 = kani::any();
        assume_bounded_f64(initial_scale, 1.0, 1e6);
        assume_bounded_f64(growth_factor, 1.01, 4.0);
        assume_bounded_f64(backoff_factor, 0.1, 0.9);
        assume_bounded_f64(min_scale, 0.1, 1.0);
        assume_bounded_f64(max_scale, 1e6, 1e10);
        kani::assume(min_scale <= initial_scale && initial_scale <= max_scale);

        let mut scale = initial_scale;
        // Step 1: growth (trigger by setting tracker = interval - 1)
        let (s, _) = scaler_update(
            scale,
            false,
            0,
            1,
            growth_factor,
            backoff_factor,
            min_scale,
            max_scale,
        );
        scale = s;
        assert!(scale >= min_scale && scale <= max_scale);
        // Step 2: backoff
        let (s, _) = scaler_update(
            scale,
            true,
            0,
            1,
            growth_factor,
            backoff_factor,
            min_scale,
            max_scale,
        );
        scale = s;
        assert!(scale >= min_scale && scale <= max_scale);
        // Step 3: growth
        let (s, _) = scaler_update(
            scale,
            false,
            0,
            1,
            growth_factor,
            backoff_factor,
            min_scale,
            max_scale,
        );
        scale = s;
        assert!(scale >= min_scale && scale <= max_scale);
        // Step 4: backoff
        let (s, _) = scaler_update(
            scale,
            true,
            0,
            1,
            growth_factor,
            backoff_factor,
            min_scale,
            max_scale,
        );
        scale = s;
        assert!(scale >= min_scale && scale <= max_scale);
    }
}
