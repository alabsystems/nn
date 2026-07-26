// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harness for forward_audio cos/sin STFT reconstruction.
//!
//! forward_audio (kokoro_audio.rs lines 48-51) rebuilds complex STFT from
//! (magnitude, phase) via `real = mag * cos(phase)`, `imag = mag * sin(phase)`.
//!
//! The decoder outputs phase = sin(phase_raw) in [-1, 1] (used directly as
//! radians without pi scaling). Magnitude = exp(clamp(log_mag, -88, 88))
//! (LOG_MAG_CLAMP_MAX = 88.0 in kokoro_error.rs:157), so
//! mag in [0, exp(88)] ≈ [0, 1.65e38].
//!
//! This harness proves the reconstruction products stay finite for these bounds.
//! Key safety margin: exp(88) * 1.0 = 1.65e38 < f32::MAX (3.4e38).

// CBMC cannot model transcendentals. Use nondeterministic stubs.
// (Per design doc: "CBMC transcendental stubs for Kani harnesses")

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

/// Harness: cos/sin STFT reconstruction products are finite and bounded.
///
/// forward_audio computes:
///   real_spec = magnitude * cos(phase)
///   imag_spec = magnitude * sin(phase)
///
/// Magnitude from decoder: exp(clamp(x, -88, 88)) in [0, 1.65e38].
/// (LOG_MAG_CLAMP_MAX = 88.0, defined in kokoro_error.rs:157)
/// Phase from decoder: sin(phase_raw) in [-1, 1] (used as radians directly).
///
/// Since cos/sin in [-1, 1], the products are bounded by magnitude.
/// Safety margin: exp(88) ≈ 1.65e38. Product = 1.65e38 * 1.0 < f32::MAX (3.4e38).
/// Overflow would require LOG_MAG_CLAMP_MAX > ~88.7 where exp(88.72) ≈ f32::MAX.
///
/// STRUCTURAL for cos/sin bound (stubs). SUBSTANTIVE for product finiteness:
/// proves the full production magnitude range (up to exp(88)) still yields
/// finite reconstruction values.
///
/// Covers: `kokoro_audio.rs` lines 48-51.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn audio_reconstruction_products_bounded() {
    // Magnitude range: exp(clamp(x, -88, 88)).
    // exp(-88) ≈ 6.1e-39 (subnormal → 0.0 in f32), exp(88) ≈ 1.65e38.
    // Upper bound 1.66e38 gives small margin above exp(88).
    let mag: f32 = kani::any();
    kani::assume(mag.is_finite());
    kani::assume(mag >= 0.0 && mag <= 1.66e38);

    // Phase: sin(phase_raw) output in [-1, 1], used as radians directly.
    let phase: f32 = kani::any();
    kani::assume(phase.is_finite());
    kani::assume(phase >= -1.0 && phase <= 1.0);

    let cos_val = cos_stub(phase);
    let sin_val = sin_stub(phase);

    let real = mag * cos_val;
    let imag = mag * sin_val;

    assert!(real.is_finite(), "real reconstruction must be finite");
    assert!(imag.is_finite(), "imag reconstruction must be finite");

    // Products bounded by magnitude (since |cos|, |sin| <= 1).
    // For large magnitudes, use relative tolerance: |product| <= mag.
    // f32 multiplication of finite values within f32::MAX range is exact or rounds.
    assert!(real.abs() <= mag + 1.0, "real must be bounded by magnitude");
    assert!(imag.abs() <= mag + 1.0, "imag must be bounded by magnitude");
}
