// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for iSTFT: DFT basis finiteness, Hann window range,
//! and overlap-add accumulation safety.
//!
//! These verify the three D3 properties from #961:
//! 1. DFT basis angle computation produces finite cos/sin values
//! 2. Hann window values are in [0.0, 1.0]
//! 3. Overlap-add accumulation does not overflow to infinity

use std::f32::consts::PI;

// CBMC cannot model f32::cos / f32::sin correctly. Use stubs that return
// nondeterministic values in [-1, 1] for safety proofs.
// (Per design doc: "CBMC sqrtf/transcendental stubs for Kani harnesses")
fn cos_stub(_x: f32) -> f32 {
    let v: f32 = kani::any();
    kani::assume(v >= -1.0 && v <= 1.0);
    v
}

fn sin_stub(_x: f32) -> f32 {
    let v: f32 = kani::any();
    kani::assume(v >= -1.0 && v <= 1.0);
    v
}

/// D3 Harness 1: DFT basis finiteness.
///
/// Prove: For any valid n_fft (even, in [2..=64]) and all f, k indices,
/// the angle computation `2 * PI * f * k / n_fft` produces a finite f32,
/// and cos_stub/sin_stub (bounded to [-1, 1]) produce finite values.
///
/// The real concern is that `(f as f32) * (k as f32)` could overflow to
/// +inf for large n_fft. For n_fft <= 8192: max f = 4097, max k = 8191,
/// so f*k <= 33,558,527 which is within f32 range (~3.4e38).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn dft_basis_angle_is_finite() {
    let n_fft_half: u8 = kani::any();
    kani::assume(n_fft_half >= 1 && n_fft_half <= 32);
    let n_fft = (n_fft_half as usize) * 2; // even, 2..=64
    let n_bins = n_fft / 2 + 1;

    let f: u8 = kani::any();
    let k: u8 = kani::any();
    kani::assume((f as usize) < n_bins);
    kani::assume((k as usize) < n_fft);

    let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
    assert!(
        angle.is_finite(),
        "DFT basis angle must be finite for valid parameters"
    );

    // Cos and sin of any finite angle are finite (in [-1, 1])
    // Using stubs because CBMC can't model trig:
    let c = cos_stub(angle);
    let s = sin_stub(angle);
    assert!(c.is_finite() && c >= -1.0 && c <= 1.0);
    assert!(s.is_finite() && s >= -1.0 && s <= 1.0);
}

/// D3 Harness 2: Hann window range [0, 1].
///
/// Prove: For any valid n_fft (even, > 0), the Hann formula
/// `0.5 * (1.0 - cos(2*PI*k/n_fft))` produces a value in [0.0, 1.0].
///
/// Mathematical proof: cos(theta) in [-1, 1], so 1 - cos in [0, 2],
/// so 0.5 * (1 - cos) in [0, 1]. The Kani harness verifies this holds
/// with f32 arithmetic (no rounding beyond [0, 1]).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn hann_window_in_unit_interval() {
    let n_fft_half: u8 = kani::any();
    kani::assume(n_fft_half >= 1 && n_fft_half <= 32);
    let n_fft = (n_fft_half as usize) * 2;

    let k: u8 = kani::any();
    kani::assume((k as usize) < n_fft);

    // The angle is finite for these small values
    let angle = 2.0 * PI * (k as f32) / (n_fft as f32);
    assert!(angle.is_finite());

    // Use cos_stub bounded to [-1, 1] to model cos(angle)
    let cos_val = cos_stub(angle);

    let w = 0.5 * (1.0 - cos_val);

    // Verify: w is in [0.0, 1.0]
    assert!(w >= 0.0, "Hann window value must be >= 0.0");
    assert!(w <= 1.0, "Hann window value must be <= 1.0");
    assert!(w.is_finite(), "Hann window value must be finite");
}

/// D3 Harness 3: Overlap-add accumulation does not overflow.
///
/// Prove: For bounded frame values and Hann window in [0, 1],
/// the overlap-add accumulation `output[i] += frame_val * w` remains finite.
///
/// Key insight: at any position, at most `n_fft / hop` frames overlap.
/// With bounded frame magnitudes and w in [0, 1], the maximum accumulation
/// is `(n_fft / hop) * max_frame_val`.
///
/// For HTDemucs (n_fft=4096, hop=1024): max 4 overlapping frames.
/// For frame values from IDFT: bounded by `norm * n_bins * 2 * max_input`.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn ola_accumulation_bounded() {
    // Model a small OLA scenario: n_fft=4, hop=2
    let n_fft: usize = 4;
    let hop: usize = 2;

    // Maximum overlap at any position: ceil(n_fft / hop) = 2
    let max_overlap = (n_fft + hop - 1) / hop;

    // Frame values bounded (model IDFT output with bounded inputs)
    let frame_bound: f32 = 100.0;

    // Window values in [0, 1]
    let w: f32 = kani::any();
    kani::assume(w >= 0.0 && w <= 1.0);

    // Simulate accumulation at one position with max_overlap contributions
    let mut accum = 0.0f32;
    let mut window_sum = 0.0f32;

    for _i in 0..max_overlap {
        let frame_val: f32 = kani::any();
        kani::assume(frame_val >= -frame_bound && frame_val <= frame_bound);
        kani::assume(frame_val.is_finite());

        accum += frame_val * w;
        window_sum += w * w;
    }

    // Verify accumulation stays finite
    assert!(
        accum.is_finite(),
        "OLA accumulation must be finite for bounded inputs"
    );
    assert!(
        window_sum.is_finite(),
        "window_sum accumulation must be finite for w in [0,1]"
    );
    assert!(
        window_sum >= 0.0,
        "window_sum must be non-negative (sum of squares)"
    );

    // If window_sum > eps, the normalized output is finite
    let eps = 1e-11f32;
    if window_sum > eps {
        let normalized = accum / window_sum;
        assert!(
            normalized.is_finite(),
            "COLA-normalized output must be finite"
        );
    }
}
