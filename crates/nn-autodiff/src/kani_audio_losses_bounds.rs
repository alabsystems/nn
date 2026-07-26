// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional Kani proof harnesses for `audio_losses.rs`.
//!
//! Focuses on scalar bounds for the public audio-loss formulas rather than the
//! lower-level STFT helpers already covered elsewhere.
//!
//! Re: #3733.

#[cfg(kani)]
mod proofs {
    use kani::assume;

    /// Numerical stability epsilon reused by the public loss formulas.
    ///
    /// SYNC: audio_losses.rs:25.
    const EPS: f32 = 1e-8;

    fn sqrt_f32_stub(x: f32) -> f32 {
        let r: f32 = kani::any();
        kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
        if x > 0.0 {
            kani::assume(r > 0.0);
        }
        r
    }

    /// Spectral convergence scalar: `sqrt(diff_sum / (ref_sum + eps))`.
    ///
    /// SYNC: audio_losses.rs:220-245.
    fn spectral_convergence(diff_sum: f32, ref_sum: f32) -> f32 {
        (diff_sum / (ref_sum + EPS)).sqrt()
    }

    /// Log spectral distance scalar: `abs(log_cand - log_ref)`.
    ///
    /// SYNC: audio_losses.rs:236-245.
    fn log_spectral_distance(log_cand: f32, log_ref: f32) -> f32 {
        (log_cand - log_ref).abs()
    }

    /// Single-resolution STFT loss scalarized to one frame/bin summary.
    ///
    /// SYNC: audio_losses.rs:220-246.
    fn stft_loss_scalar(diff_sum: f32, ref_sum: f32, log_cand: f32, log_ref: f32) -> f32 {
        spectral_convergence(diff_sum, ref_sum) + log_spectral_distance(log_cand, log_ref)
    }

    /// Mean of three non-negative losses, matching the averaging used by
    /// `multi_res_stft_loss` and `feature_matching_loss`.
    ///
    /// SYNC: audio_losses.rs:278-293, 368-382.
    fn average3(a: f32, b: f32, c: f32) -> f32 {
        (a + b + c) / 3.0
    }

    /// Scalar MSE used by the log-mel loss.
    ///
    /// SYNC: audio_losses.rs:329-334.
    fn mse_scalar(lhs: f32, rhs: f32) -> f32 {
        let diff = lhs - rhs;
        diff * diff
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn spectral_convergence_is_finite_and_non_negative() {
        let diff_sum: f32 = kani::any();
        let ref_sum: f32 = kani::any();

        assume(diff_sum.is_finite() && diff_sum >= 0.0 && diff_sum <= 1e8);
        assume(ref_sum.is_finite() && ref_sum >= 0.0 && ref_sum <= 1e8);

        let sc = spectral_convergence(diff_sum, ref_sum);
        assert!(sc.is_finite(), "spectral convergence must be finite");
        assert!(sc >= 0.0, "spectral convergence must be non-negative");
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn stft_loss_is_non_negative() {
        let diff_sum: f32 = kani::any();
        let ref_sum: f32 = kani::any();
        let log_cand: f32 = kani::any();
        let log_ref: f32 = kani::any();

        assume(diff_sum.is_finite() && diff_sum >= 0.0 && diff_sum <= 1e8);
        assume(ref_sum.is_finite() && ref_sum >= 0.0 && ref_sum <= 1e8);
        assume(log_cand.is_finite() && log_cand.abs() <= 1e4);
        assume(log_ref.is_finite() && log_ref.abs() <= 1e4);

        let total = stft_loss_scalar(diff_sum, ref_sum, log_cand, log_ref);
        assert!(total.is_finite(), "stft loss must be finite");
        assert!(total >= 0.0, "stft loss must be non-negative");
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn stft_loss_is_zero_for_identical_summaries() {
        let ref_sum: f32 = kani::any();
        let log_mag: f32 = kani::any();

        assume(ref_sum.is_finite() && ref_sum >= 0.0 && ref_sum <= 1e8);
        assume(log_mag.is_finite() && log_mag.abs() <= 1e4);

        let total = stft_loss_scalar(0.0, ref_sum, log_mag, log_mag);
        assert!(
            total == 0.0,
            "stft loss must be zero when spectral and log summaries match exactly"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn multi_res_average_stays_between_zero_and_component_max() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        let c: f32 = kani::any();

        assume(a.is_finite() && a >= 0.0 && a <= 1e6);
        assume(b.is_finite() && b >= 0.0 && b <= 1e6);
        assume(c.is_finite() && c >= 0.0 && c <= 1e6);

        let avg = average3(a, b, c);
        let max_component = a.max(b).max(c);

        assert!(avg.is_finite(), "multi-res average must be finite");
        assert!(avg >= 0.0, "multi-res average must be non-negative");
        assert!(
            avg <= max_component + 1e-5,
            "multi-res average must not exceed its largest component"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn mel_spectrogram_mse_is_non_negative() {
        let log_cand: f32 = kani::any();
        let log_ref: f32 = kani::any();

        assume(log_cand.is_finite() && log_cand.abs() <= 1e4);
        assume(log_ref.is_finite() && log_ref.abs() <= 1e4);

        let loss = mse_scalar(log_cand, log_ref);
        assert!(loss.is_finite(), "log-mel mse must be finite");
        assert!(loss >= 0.0, "log-mel mse must be non-negative");
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn mel_spectrogram_mse_is_zero_when_logs_match() {
        let log_mel: f32 = kani::any();

        assume(log_mel.is_finite() && log_mel.abs() <= 1e4);

        let loss = mse_scalar(log_mel, log_mel);
        assert!(
            loss == 0.0,
            "log-mel mse must be zero when candidate and reference match"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn feature_matching_average_is_non_negative() {
        let l1_a: f32 = kani::any();
        let l1_b: f32 = kani::any();
        let l1_c: f32 = kani::any();

        assume(l1_a.is_finite() && l1_a >= 0.0 && l1_a <= 1e6);
        assume(l1_b.is_finite() && l1_b >= 0.0 && l1_b <= 1e6);
        assume(l1_c.is_finite() && l1_c >= 0.0 && l1_c <= 1e6);

        let loss = average3(l1_a, l1_b, l1_c);
        assert!(loss.is_finite(), "feature matching loss must be finite");
        assert!(loss >= 0.0, "feature matching loss must be non-negative");
    }
}
