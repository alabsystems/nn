// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dense soundness sweep for RoPE bounds functions.
//!
//! Verifies that `rope_cos_scalar_bounds` and `rope_sin_scalar_bounds` produce
//! bounds that contain the actual scalar output for a dense grid of inputs
//! across 6 interval configurations targeting specific cos_range/sin_range
//! code paths: zero-crossing, π-spanning, 2π-spanning, half-π, negative freq,
//! and asymmetric x ranges.
//!
//! Part of #659. Compensates for inherent Kani infeasibility of RoPE bounds
//! soundness proofs (trig stub limitation — see #659 finding 2).

use nn_dsl::{rope_cos_scalar, rope_cos_scalar_bounds, rope_sin_scalar, rope_sin_scalar_bounds};

/// Dense soundness sweep: 6 interval configs × 30^3 grid = 162K samples.
#[test]
fn test_rope_bounds_dense_multi_interval_soundness() {
    let configs: &[(f32, f32, f32, f32, f32, f32)] = &[
        // (x0_lo, x0_hi, x1_lo, x1_hi, freq_lo, freq_hi)
        (-5.0, 5.0, -3.0, 7.0, -0.5, 0.5),  // zero-crossing
        (-2.0, 3.0, -4.0, 1.0, 2.5, 3.8),   // π-spanning
        (-1.0, 1.0, -1.0, 1.0, 5.5, 7.0),   // 2π-spanning
        (-3.0, 3.0, -3.0, 3.0, 1.0, 2.0),   // half-π crossing
        (-4.0, 4.0, -2.0, 6.0, -4.0, -1.0), // negative freq
        (0.1, 10.0, -10.0, -0.1, 0.0, std::f32::consts::PI * 2.0), // asymmetric x
    ];
    let steps = 30;
    for (ci, &(x0_lo, x0_hi, x1_lo, x1_hi, freq_lo, freq_hi)) in configs.iter().enumerate() {
        let (cos_lo, cos_hi) =
            rope_cos_scalar_bounds(x0_lo, x0_hi, x1_lo, x1_hi, freq_lo, freq_hi).expect("bounds");
        let (sin_lo, sin_hi) =
            rope_sin_scalar_bounds(x0_lo, x0_hi, x1_lo, x1_hi, freq_lo, freq_hi).expect("bounds");
        assert!(cos_lo <= cos_hi, "config {ci}: cos bounds inverted");
        assert!(sin_lo <= sin_hi, "config {ci}: sin bounds inverted");
        for ix0 in 0..=steps {
            let x0 = x0_lo + (x0_hi - x0_lo) * (ix0 as f32 / steps as f32);
            for ix1 in 0..=steps {
                let x1 = x1_lo + (x1_hi - x1_lo) * (ix1 as f32 / steps as f32);
                for ifr in 0..=steps {
                    let freq = freq_lo + (freq_hi - freq_lo) * (ifr as f32 / steps as f32);
                    let yc = rope_cos_scalar(x0, x1, freq).expect("scalar");
                    let ys = rope_sin_scalar(x0, x1, freq).expect("scalar");
                    assert!(
                        yc >= cos_lo - 1e-5 && yc <= cos_hi + 1e-5,
                        "config {ci}: rope_cos({x0}, {x1}, {freq}) = {yc} outside bounds"
                    );
                    assert!(
                        ys >= sin_lo - 1e-5 && ys <= sin_hi + 1e-5,
                        "config {ci}: rope_sin({x0}, {x1}, {freq}) = {ys} outside bounds"
                    );
                }
            }
        }
    }
}

/// Verify bounds contain output at exact π/2 multiples (peak detection edge cases).
#[test]
fn test_rope_bounds_at_pi_multiples() {
    let pi = std::f32::consts::PI;
    let tau = std::f32::consts::TAU;
    // Test specific frequency intervals that straddle important cos/sin peaks
    let peak_configs: &[(f32, f32, f32)] = &[
        // (freq_lo, freq_hi, expected_cos_peak_or_trough)
        (pi - 0.1, pi + 0.1, -1.0),   // cos(π) = -1
        (tau - 0.1, tau + 0.1, 1.0),  // cos(2π) = 1
        (-pi - 0.1, -pi + 0.1, -1.0), // cos(-π) = -1
    ];
    for (i, &(freq_lo, freq_hi, _expected)) in peak_configs.iter().enumerate() {
        // x0=1, x1=0 reduces rope_cos to just cos(freq)
        let (lo, hi) =
            rope_cos_scalar_bounds(1.0, 1.0, 0.0, 0.0, freq_lo, freq_hi).expect("bounds");
        assert!(lo <= hi, "config {i}: bounds inverted [{lo}, {hi}]");
        assert!(lo.is_finite() && hi.is_finite(), "config {i}: non-finite");
        // Sample 100 points in the freq interval
        for j in 0..=100 {
            let freq = freq_lo + (freq_hi - freq_lo) * (j as f32 / 100.0);
            let y = rope_cos_scalar(1.0, 0.0, freq).expect("scalar");
            assert!(
                y >= lo - 1e-5 && y <= hi + 1e-5,
                "config {i}: cos(1, 0, {freq}) = {y} outside [{lo}, {hi}]"
            );
        }
    }
}
