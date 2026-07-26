// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for iSTFT overlap-add normalization.
//!
//! These model the sample-local COLA arithmetic used in `istft.rs`:
//! `output[i] /= window_sum[i]` when `window_sum[i] > 1e-11`.

#[cfg(kani)]
mod proofs {
    /// When two overlapping windowed frame samples carry the same underlying
    /// signal sample, COLA normalization recovers that sample.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn cola_normalization_recovers_underlying_sample() {
        let sample: f32 = kani::any();
        kani::assume(sample.is_finite());
        kani::assume(sample >= -32.0 && sample <= 32.0);

        let w0: f32 = kani::any();
        let w1: f32 = kani::any();
        kani::assume(w0.is_finite() && w1.is_finite());
        kani::assume(w0 >= 0.5 && w0 <= 1.0);
        kani::assume(w1 >= 0.0 && w1 <= 1.0);

        let window_sum = w0 * w0 + w1 * w1;
        let accum = (sample * w0) * w0 + (sample * w1) * w1;
        let normalized = accum / window_sum;

        assert!(window_sum.is_finite());
        assert!(window_sum > 1e-11);
        assert!(normalized.is_finite());
        assert!((normalized - sample).abs() <= 1e-4);
    }

    /// The `window_sum > eps` guard blocks division on near-zero overlap energy.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn cola_guard_rejects_near_zero_window_sum() {
        let w0: f32 = kani::any();
        let w1: f32 = kani::any();
        kani::assume(w0.is_finite() && w1.is_finite());
        kani::assume(w0 >= 0.0 && w0 <= 1e-6);
        kani::assume(w1 >= 0.0 && w1 <= 1e-6);

        let window_sum = w0 * w0 + w1 * w1;
        let should_normalize = window_sum > 1e-11f32;

        assert!(window_sum.is_finite());
        assert!(!should_normalize);
    }
}
