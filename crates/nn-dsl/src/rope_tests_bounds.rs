// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! RoPE (K6) scalar bounds, edge-case, and error-path tests.

use super::*;

// --- Scalar bounds ---

#[test]
fn test_rope_cos_scalar_bounds_contains_known_output() {
    // x0=3, x1=4, freq=0.5 → output = 3*cos(0.5) - 4*sin(0.5)
    let expected = rope_cos_scalar(3.0, 4.0, 0.5).expect("must succeed");
    let (lo, hi) = rope_cos_scalar_bounds(3.0, 3.0, 4.0, 4.0, 0.5, 0.5).expect("point bounds");
    assert!(
        lo <= expected + 1e-5 && hi >= expected - 1e-5,
        "bounds [{lo}, {hi}] must contain output {expected}"
    );
}

#[test]
fn test_rope_sin_scalar_bounds_contains_known_output() {
    let expected = rope_sin_scalar(3.0, 4.0, 0.5).expect("must succeed");
    let (lo, hi) = rope_sin_scalar_bounds(3.0, 3.0, 4.0, 4.0, 0.5, 0.5).expect("point bounds");
    assert!(
        lo <= expected + 1e-5 && hi >= expected - 1e-5,
        "bounds [{lo}, {hi}] must contain output {expected}"
    );
}

#[test]
fn test_rope_bounds_soundness_random_samples() {
    // Sample many random inputs in the bound intervals and verify every sample
    // falls within the computed bounds (empirical soundness check).
    let x0_lo = -5.0_f32;
    let x0_hi = 5.0;
    let x1_lo = -3.0;
    let x1_hi = 7.0;
    let freq_lo = -1.0;
    let freq_hi = 2.0;

    let (cos_lo, cos_hi) = rope_cos_scalar_bounds(x0_lo, x0_hi, x1_lo, x1_hi, freq_lo, freq_hi)
        .expect("finite bounds");
    let (sin_lo, sin_hi) = rope_sin_scalar_bounds(x0_lo, x0_hi, x1_lo, x1_hi, freq_lo, freq_hi)
        .expect("finite bounds");

    // Grid sample
    let steps = 20;
    for ix0 in 0..=steps {
        for ix1 in 0..=steps {
            for ifr in 0..=steps {
                let x0 = x0_lo + (x0_hi - x0_lo) * (ix0 as f32 / steps as f32);
                let x1 = x1_lo + (x1_hi - x1_lo) * (ix1 as f32 / steps as f32);
                let freq = freq_lo + (freq_hi - freq_lo) * (ifr as f32 / steps as f32);

                let yc = rope_cos_scalar(x0, x1, freq).expect("must succeed");
                let ys = rope_sin_scalar(x0, x1, freq).expect("must succeed");

                assert!(
                    yc >= cos_lo - 1e-5 && yc <= cos_hi + 1e-5,
                    "rope_cos({x0}, {x1}, {freq}) = {yc} outside [{cos_lo}, {cos_hi}]"
                );
                assert!(
                    ys >= sin_lo - 1e-5 && ys <= sin_hi + 1e-5,
                    "rope_sin({x0}, {x1}, {freq}) = {ys} outside [{sin_lo}, {sin_hi}]"
                );
            }
        }
    }
}

#[test]
fn test_rope_bounds_zero_freq_interval() {
    // freq = [0, 0] → cos(0)=1, sin(0)=0
    // rope_cos = x0*1 - x1*0 = x0, so bounds should be [x0_lo, x0_hi]
    let (lo, hi) = rope_cos_scalar_bounds(-3.0, 5.0, -2.0, 2.0, 0.0, 0.0).expect("finite");
    assert!(
        lo <= -3.0 + 1e-5,
        "cos lower should be <= x0_lo=-3, got {lo}"
    );
    assert!(hi >= 5.0 - 1e-5, "cos upper should be >= x0_hi=5, got {hi}");
}

#[test]
fn test_rope_bounds_full_freq_rotation() {
    // When freq spans a full 2π, cos and sin cover [-1,1].
    // Output range should be wide: |output| ≤ |x0| + |x1|
    let (lo, hi) =
        rope_cos_scalar_bounds(2.0, 3.0, 1.0, 4.0, 0.0, std::f32::consts::TAU).expect("finite");
    // Maximum magnitude: max(|x0|)=3 + max(|x1|)=4 = 7
    assert!(
        lo <= -0.5,
        "full rotation lower should be negative, got {lo}"
    );
    assert!(
        hi >= 0.5,
        "full rotation upper should be positive, got {hi}"
    );
}

#[test]
fn test_rope_bounds_rejects_nan() {
    let err = rope_cos_scalar_bounds(f32::NAN, 1.0, 0.0, 1.0, 0.0, 1.0)
        .expect_err("NaN should be rejected");
    assert!(
        matches!(err, KernelError::NonFiniteBound { value } if value.is_nan()),
        "expected NonFiniteBound with NaN, got {err:?}"
    );
}

#[test]
fn test_rope_bounds_rejects_infinity() {
    let err = rope_sin_scalar_bounds(0.0, 1.0, f32::INFINITY, 1.0, 0.0, 1.0)
        .expect_err("Inf should be rejected");
    assert!(
        matches!(err, KernelError::NonFiniteBound { value } if value.is_infinite()),
        "expected NonFiniteBound with Inf, got {err:?}"
    );
}

#[test]
fn test_rope_bounds_rejects_inverted_x0() {
    let err = rope_cos_scalar_bounds(5.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        .expect_err("inverted x0 should be rejected");
    assert!(
        matches!(err, KernelError::InvertedBounds { .. }),
        "expected InvertedBounds, got {err:?}"
    );
}

#[test]
fn test_rope_bounds_rejects_inverted_freq() {
    let err = rope_sin_scalar_bounds(0.0, 1.0, 0.0, 1.0, 3.0, 1.0)
        .expect_err("inverted freq should be rejected");
    assert!(
        matches!(err, KernelError::InvertedBounds { .. }),
        "expected InvertedBounds, got {err:?}"
    );
}

#[test]
fn test_rope_bounds_point_interval_matches_exact() {
    // Point interval: all inputs are exact scalars.
    let x0 = 2.5;
    let x1 = -1.3;
    let freq = 0.7;

    let expected_cos = rope_cos_scalar(x0, x1, freq).expect("must succeed");
    let expected_sin = rope_sin_scalar(x0, x1, freq).expect("must succeed");

    let (clo, chi) = rope_cos_scalar_bounds(x0, x0, x1, x1, freq, freq).expect("point bounds");
    let (slo, shi) = rope_sin_scalar_bounds(x0, x0, x1, x1, freq, freq).expect("point bounds");

    assert!(
        (clo - expected_cos).abs() < 1e-5 && (chi - expected_cos).abs() < 1e-5,
        "point cos bounds [{clo}, {chi}] should match {expected_cos}"
    );
    assert!(
        (slo - expected_sin).abs() < 1e-5 && (shi - expected_sin).abs() < 1e-5,
        "point sin bounds [{slo}, {shi}] should match {expected_sin}"
    );
}

#[test]
fn test_rope_bounds_overflow_detected() {
    // Large inputs should overflow during interval arithmetic → NonFiniteBound
    let err = rope_cos_scalar_bounds(0.0, f32::MAX, 0.0, f32::MAX, 0.0, 7.0)
        .expect_err("overflow should be detected");
    assert!(
        matches!(err, KernelError::NonFiniteBound { .. }),
        "expected NonFiniteBound, got {err:?}"
    );
}

#[test]
fn test_rope_sin_bounds_overflow_detected() {
    // Same overflow path for rope_sin_scalar_bounds.
    let err = rope_sin_scalar_bounds(0.0, f32::MAX, 0.0, f32::MAX, 0.0, 7.0)
        .expect_err("overflow should be detected");
    assert!(
        matches!(err, KernelError::NonFiniteBound { .. }),
        "expected NonFiniteBound for rope_sin bounds, got {err:?}"
    );
}

// --- cos_range / sin_range edge-case coverage via public API (P1 #275) ---

#[test]
fn test_rope_bounds_high_frequency_soundness() {
    // High-frequency interval where cos_range's `as i64` cast on
    // (freq / PI).ceil() could theoretically overflow for |freq| > ~9.2e18.
    // At f32 precision, freq=1e6 is already imprecise for trig, but bounds
    // must remain sound (not panic, not produce inverted bounds).
    let freq_hi = 1e6_f32;
    let freq_lo = freq_hi - 0.1;
    let result = rope_cos_scalar_bounds(1.0, 1.0, 1.0, 1.0, freq_lo, freq_hi);
    match result {
        Ok((lo, hi)) => {
            assert!(lo <= hi, "bounds must not be inverted: [{lo}, {hi}]");
            assert!(lo.is_finite() && hi.is_finite(), "bounds must be finite");
            // Soundness: evaluate at endpoints and midpoint.
            for &freq in &[freq_lo, freq_hi, f32::midpoint(freq_lo, freq_hi)] {
                let y = rope_cos_scalar(1.0, 1.0, freq).expect("must succeed");
                assert!(
                    y >= lo - 1e-3 && y <= hi + 1e-3,
                    "rope_cos(1, 1, {freq}) = {y} outside [{lo}, {hi}]"
                );
            }
        }
        Err(KernelError::NonFiniteBound { .. }) => {
            // Overflow is acceptable for extreme frequencies.
        }
        Err(e) => unreachable!("unexpected error for high-frequency bounds: {e:?}"),
    }
}

#[test]
fn test_rope_bounds_negative_frequency_interval() {
    // Negative frequencies: cos is even, sin is odd — test that bounds
    // are correct when freq interval is entirely negative.
    let (lo, hi) =
        rope_cos_scalar_bounds(2.0, 2.0, 1.0, 1.0, -3.0, -1.0).expect("negative freq bounds");
    assert!(lo <= hi, "bounds inverted: [{lo}, {hi}]");
    // Verify soundness at 10 sample points.
    for i in 0..=10 {
        let freq = -3.0 + 2.0 * (i as f32 / 10.0);
        let y = rope_cos_scalar(2.0, 1.0, freq).expect("must succeed");
        assert!(
            y >= lo - 1e-5 && y <= hi + 1e-5,
            "rope_cos(2, 1, {freq}) = {y} outside [{lo}, {hi}]"
        );
    }
}

#[test]
fn test_rope_bounds_freq_spanning_pi_captures_extrema() {
    // freq ∈ [0, π]: cos goes from 1 to -1, sin goes from 0 to 0 with max at π/2.
    // rope_cos = x0*cos(freq) - x1*sin(freq)
    // With x0=1, x1=0: output = cos(freq) ∈ [-1, 1].
    let (lo, hi) = rope_cos_scalar_bounds(1.0, 1.0, 0.0, 0.0, 0.0, std::f32::consts::PI)
        .expect("pi-spanning bounds");
    assert!(
        lo <= -1.0 + 1e-5,
        "should reach cos(π)=-1 at lower bound, got {lo}"
    );
    assert!(
        hi >= 1.0 - 1e-5,
        "should reach cos(0)=1 at upper bound, got {hi}"
    );
}

#[test]
fn test_rope_bounds_near_peak_f32_rounding() {
    // Regression test: interval boundary very close to a cos peak at 2π.
    // In f32, TAU ≈ 6.2831855. If peak detection used f32 arithmetic,
    // rounding in `k * TAU <= hi` could miss the peak, producing unsound
    // (too-tight) bounds. The f64 promotion in cos_range prevents this.
    let tau_f32 = std::f32::consts::TAU;
    // Interval that just barely includes 2π (one ULP above).
    let freq_lo = tau_f32 - 0.001;
    let freq_hi = tau_f32 + f32::EPSILON;
    // With x0=1, x1=0: output = cos(freq), and cos(2π) = 1.
    let (lo, hi) =
        rope_cos_scalar_bounds(1.0, 1.0, 0.0, 0.0, freq_lo, freq_hi).expect("near-peak bounds");
    assert!(
        hi >= 1.0 - 1e-5,
        "bounds must capture cos(2π)=1 peak, got hi={hi}"
    );
    assert!(lo <= hi, "bounds inverted: [{lo}, {hi}]");
}

#[test]
fn test_rope_bounds_near_trough_f32_rounding() {
    // Regression test: interval boundary very close to a cos trough at π.
    // cos(π) = -1. Same f32 rounding concern as the peak test.
    let pi_f32 = std::f32::consts::PI;
    let freq_lo = pi_f32 - 0.001;
    let freq_hi = pi_f32 + f32::EPSILON;
    // With x0=1, x1=0: output = cos(freq), and cos(π) = -1.
    let (lo, hi) =
        rope_cos_scalar_bounds(1.0, 1.0, 0.0, 0.0, freq_lo, freq_hi).expect("near-trough bounds");
    assert!(
        lo <= -1.0 + 1e-5,
        "bounds must capture cos(π)=-1 trough, got lo={lo}"
    );
    assert!(lo <= hi, "bounds inverted: [{lo}, {hi}]");
}

// --- sin_range f64 precision regression test (#429) ---

#[test]
fn test_sin_range_f64_subtraction_captures_trough() {
    // Regression test for #429: sin_range subtracts π/2 in f64, not f32.
    //
    // freq interval [4.712387561798096, 4.71238899230957] contains 3π/2 ≈ 4.71238898,
    // where sin = -1. With f32 subtraction of π/2, the shifted interval lands
    // entirely below π (both endpoints round down), causing cos_range to miss the
    // cos trough at π. With f64 subtraction, the shifted interval correctly
    // straddles π, and cos_range detects the trough.
    //
    // With x0=1, x1=0: rope_sin = sin(freq), so the bounds must contain -1.
    let freq_lo: f32 = 4.712_387_6;
    let freq_hi: f32 = 4.712_389;

    // Verify the interval actually contains 3π/2 (sin trough).
    let three_pi_2 = 3.0 * std::f64::consts::FRAC_PI_2;
    assert!(
        f64::from(freq_lo) <= three_pi_2 && three_pi_2 <= f64::from(freq_hi),
        "test precondition: freq interval must contain 3π/2"
    );

    let (lo, _hi) = rope_sin_scalar_bounds(1.0, 1.0, 0.0, 0.0, freq_lo, freq_hi)
        .expect("sin bounds near trough");
    assert!(
        lo <= -1.0 + 1e-5,
        "sin bounds must capture sin(3π/2)=-1 trough; got lo={lo} \
         (f32 π/2 subtraction would miss this — see #429)"
    );
}

// Error-path tests (NaN/Inf rejection, overflow) extracted to rope_tests_error_paths.rs
