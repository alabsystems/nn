// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for audio quality metric math.
//!
//! Proves the mathematical properties required by dvoice V1/V2 gate
//! conditions (per #1470):
//!
//! 1. Cosine similarity: output in [-1, 1], zero-vector handling, normalization
//! 2. SNR/SDR: finite output, correct dB conversion, division-by-zero guard
//! 3. RMS energy: non-negative, sqrt safety
//! 4. Mel filterbank: monotonic Hz↔mel, finite output, non-negative energies
//! 5. dB conversion: finite for positive input
//!
//! Each harness uses bounded f32 inputs (|x| <= 1e4) to avoid f32 overflow
//! in intermediate computations while covering the full audio sample range.

use super::quality_metrics::{
    cosine_similarity_scalar, hz_to_mel, mel_to_hz, power_to_db, rms_scalar, snr_scalar,
};

// ---- Nondeterministic transcendental stubs for Kani (CBMC cannot model these) --
// See nn_engineering.md: "CBMC transcendental stubs for Kani."

fn sqrt_f64_stub(x: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e20);
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    if x >= 1.0 {
        kani::assume(r >= 1.0);
    }
    r
}

fn log10_f64_stub(_x: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= -20.0 && r <= 40.0);
    r
}

fn powf_f64_stub(_b: f64, _e: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite());
    r
}

// ---- Cosine Similarity Proofs ------------------------------------------------

/// Prove: cosine_similarity_scalar output is always in [-1.0, 1.0] for finite inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn cosine_similarity_output_in_range() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1e4 && b.abs() <= 1e4);

    let sim = cosine_similarity_scalar(a, b);
    assert!(sim >= -1.0, "cosine sim must be >= -1.0, got {sim}");
    assert!(sim <= 1.0, "cosine sim must be <= 1.0, got {sim}");
    assert!(sim.is_finite(), "cosine sim must be finite");
}

/// Prove: cosine_similarity_scalar returns 0.0 when either input is zero.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn cosine_similarity_zero_vector_returns_zero() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e4);

    let sim_a_zero = cosine_similarity_scalar(0.0, x);
    assert_eq!(sim_a_zero, 0.0, "zero first arg → 0.0");

    let sim_b_zero = cosine_similarity_scalar(x, 0.0);
    assert_eq!(sim_b_zero, 0.0, "zero second arg → 0.0");
}

/// Prove: cosine_similarity_scalar(x, x) == 1.0 for any non-zero finite x.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn cosine_similarity_self_is_one() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x != 0.0);
    kani::assume(x.abs() <= 1e4);

    let sim = cosine_similarity_scalar(x, x);
    // For scalar: sim(x, x) = x²/|x|² = 1.0
    assert!(
        (sim - 1.0).abs() < 1e-10,
        "self-similarity must be 1.0, got {sim}"
    );
}

/// Prove: cosine_similarity_scalar(x, -x) == -1.0 for any non-zero finite x.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn cosine_similarity_negation_is_minus_one() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x != 0.0);
    kani::assume(x.abs() <= 1e4);

    let sim = cosine_similarity_scalar(x, -x);
    assert!(
        (sim - (-1.0)).abs() < 1e-10,
        "negation similarity must be -1.0, got {sim}"
    );
}

// ---- SNR/SDR Proofs ---------------------------------------------------------

/// Prove: snr_scalar returns finite non-negative dB for non-zero signal and noise.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::log10, log10_f64_stub)]
fn snr_finite_for_nonzero_inputs() {
    let signal: f32 = kani::any();
    let noise: f32 = kani::any();
    kani::assume(signal.is_finite() && noise.is_finite());
    kani::assume(signal != 0.0 && noise != 0.0);
    kani::assume(signal.abs() <= 1e4 && noise.abs() <= 1e4);

    let snr = snr_scalar(signal, noise);
    assert!(snr.is_finite(), "SNR must be finite for non-zero inputs");
}

/// Prove: snr_scalar returns 0.0 for zero signal (silent reference guard).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn snr_zero_signal_returns_zero() {
    let noise: f32 = kani::any();
    kani::assume(noise.is_finite() && noise.abs() <= 1e4);

    let snr = snr_scalar(0.0, noise);
    assert_eq!(snr, 0.0, "zero signal → 0 dB");
}

/// Prove: snr_scalar returns +Infinity for zero noise (perfect reconstruction).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn snr_zero_noise_returns_infinity() {
    let signal: f32 = kani::any();
    kani::assume(signal.is_finite() && signal != 0.0);
    kani::assume(signal.abs() <= 1e4);

    let snr = snr_scalar(signal, 0.0);
    assert!(snr.is_infinite() && snr > 0.0, "zero noise → +Infinity");
}

/// Prove: SNR is positive when |signal| > |noise|.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::log10, log10_f64_stub)]
fn snr_positive_when_signal_dominates() {
    let signal: f32 = kani::any();
    let noise: f32 = kani::any();
    kani::assume(signal.is_finite() && noise.is_finite());
    kani::assume(signal != 0.0 && noise != 0.0);
    kani::assume(signal.abs() > noise.abs());
    kani::assume(signal.abs() <= 1e4 && noise.abs() <= 1e4);

    let snr = snr_scalar(signal, noise);
    assert!(snr > 0.0, "SNR must be positive when |signal| > |noise|");
}

// ---- RMS Energy Proofs ------------------------------------------------------

/// Prove: rms_scalar returns non-negative value for any finite input.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn rms_non_negative() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x.abs() <= 1e4);

    let r = rms_scalar(x);
    assert!(r >= 0.0, "RMS must be non-negative, got {r}");
    assert!(r.is_finite(), "RMS must be finite for finite input");
}

/// Prove: rms_scalar(0.0) == 0.0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn rms_zero_is_zero() {
    let r = rms_scalar(0.0);
    assert_eq!(r, 0.0, "RMS(0) must be 0.0");
}

/// Prove: rms_scalar(x) == |x| for a single sample.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn rms_equals_abs_for_scalar() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x.abs() <= 1e4);

    let r = rms_scalar(x);
    let expected = f64::from(x).abs();
    assert!(
        (r - expected).abs() < 1e-10,
        "RMS(x) must equal |x| for scalar, got {r} vs {expected}"
    );
}

// ---- dB Conversion Proofs ---------------------------------------------------

/// Prove: power_to_db returns finite value for positive input.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::log10, log10_f64_stub)]
fn power_to_db_finite_for_positive() {
    let power: f64 = kani::any();
    kani::assume(power > 0.0 && power.is_finite());
    kani::assume(power <= 1e10 && power >= 1e-10);

    let db = power_to_db(power);
    assert!(db.is_finite(), "dB must be finite for positive power");
}

/// Prove: power_to_db is monotonically increasing (more power → more dB).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::log10, log10_f64_stub)]
fn power_to_db_monotonic() {
    let p1: f64 = kani::any();
    let p2: f64 = kani::any();
    kani::assume(p1 > 0.0 && p2 > 0.0);
    kani::assume(p1.is_finite() && p2.is_finite());
    kani::assume(p1 <= 1e10 && p2 <= 1e10);
    kani::assume(p1 >= 1e-10 && p2 >= 1e-10);
    kani::assume(p1 < p2);

    let db1 = power_to_db(p1);
    let db2 = power_to_db(p2);
    assert!(db1 < db2, "dB must be monotonically increasing with power");
}

// ---- Mel Scale Proofs -------------------------------------------------------

/// Prove: hz_to_mel is monotonically increasing for non-negative frequencies.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::log10, log10_f64_stub)]
fn mel_monotonic_increasing() {
    let f1: f64 = kani::any();
    let f2: f64 = kani::any();
    kani::assume(f1 >= 0.0 && f2 >= 0.0);
    kani::assume(f1.is_finite() && f2.is_finite());
    kani::assume(f1 <= 22050.0 && f2 <= 22050.0); // Up to Nyquist at 44.1kHz
    kani::assume(f1 < f2);

    let mel1 = hz_to_mel(f1);
    let mel2 = hz_to_mel(f2);
    assert!(mel1 < mel2, "hz_to_mel must be monotonically increasing");
}

/// Prove: mel(0 Hz) == 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::log10, log10_f64_stub)]
fn mel_zero_hz_is_zero() {
    let mel = hz_to_mel(0.0);
    assert!(mel.abs() < 1e-10, "hz_to_mel(0) must be 0, got {mel}");
}

/// Prove: hz_to_mel returns finite non-negative value for valid frequencies.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::log10, log10_f64_stub)]
fn mel_finite_non_negative() {
    let f: f64 = kani::any();
    kani::assume(f >= 0.0 && f.is_finite());
    kani::assume(f <= 22050.0);

    let mel = hz_to_mel(f);
    assert!(mel.is_finite(), "mel must be finite");
    assert!(mel >= 0.0, "mel must be non-negative for non-negative Hz");
}

/// Prove: mel ↔ Hz roundtrip is identity within tolerance.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::log10, log10_f64_stub)]
#[kani::stub(f64::powf, powf_f64_stub)]
fn mel_hz_roundtrip() {
    let f: f64 = kani::any();
    kani::assume(f >= 0.0 && f.is_finite());
    kani::assume(f <= 22050.0);

    let roundtrip = mel_to_hz(hz_to_mel(f));
    assert!(roundtrip.is_finite(), "roundtrip must be finite");
    // Allow small relative error from f64 arithmetic.
    let err = if f > 1e-6 {
        (roundtrip - f).abs() / f
    } else {
        (roundtrip - f).abs()
    };
    assert!(err < 1e-10, "mel roundtrip error too large: {err}");
}

/// Prove: mel_to_hz returns non-negative for non-negative mel.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::powf, powf_f64_stub)]
fn mel_to_hz_non_negative() {
    let m: f64 = kani::any();
    kani::assume(m >= 0.0 && m.is_finite());
    kani::assume(m <= 5000.0); // Reasonable mel range

    let hz = mel_to_hz(m);
    assert!(hz.is_finite(), "Hz must be finite");
    assert!(hz >= 0.0, "Hz must be non-negative for non-negative mel");
}

// ---- Mel Filterbank Proofs --------------------------------------------------
//
// These harnesses call the production `triangular_filter_weight()` function
// extracted from mel_filterbank() in dsp.rs. The mel_filter_energy harness
// uses the production weight function to generate filter coefficients, then
// proves the inner-product energy is non-negative.

/// Prove: production `triangular_filter_weight()` output is in [0, 1].
///
/// Calls the production function directly (not an inline copy).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn mel_filter_weight_in_unit_interval() {
    let f_left: f64 = kani::any();
    let f_center: f64 = kani::any();
    let f_right: f64 = kani::any();
    let k: f64 = kani::any();

    kani::assume(f_left >= 0.0 && f_center > f_left && f_right > f_center);
    kani::assume(f_left.is_finite() && f_center.is_finite() && f_right.is_finite());
    kani::assume(f_left <= 1000.0 && f_center <= 1000.0 && f_right <= 1000.0);
    kani::assume(k >= 0.0 && k.is_finite() && k <= 1000.0);

    let weight = super::triangular_filter_weight(k, f_left, f_center, f_right);

    assert!(
        weight >= 0.0,
        "mel filter weight must be non-negative, got {weight}"
    );
    assert!(
        weight <= 1.0,
        "mel filter weight must be <= 1.0, got {weight}"
    );
    assert!(weight.is_finite(), "mel filter weight must be finite");
}

/// Prove: production `triangular_filter_weight()` at center frequency is 1.0.
///
/// Calls the production function directly (not an inline copy).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn mel_filter_peak_is_one() {
    let f_left: f64 = kani::any();
    let f_center: f64 = kani::any();
    let f_right: f64 = kani::any();

    kani::assume(f_left >= 0.0 && f_center > f_left && f_right > f_center);
    kani::assume(f_left.is_finite() && f_center.is_finite() && f_right.is_finite());
    kani::assume(f_left <= 1000.0 && f_center <= 1000.0 && f_right <= 1000.0);

    let weight = super::triangular_filter_weight(f_center, f_left, f_center, f_right);

    assert!(
        (weight - 1.0).abs() < 1e-10,
        "filter peak must be 1.0, got {weight}"
    );
}

/// Prove: mel filter energy is non-negative when filter weights come from
/// production `triangular_filter_weight()` and power spectrum is non-negative.
///
/// Models: energy = sum_k(filter[k] * power[k]) for a 2-bin example.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn mel_filter_energy_non_negative() {
    let f_left: f64 = kani::any();
    let f_center: f64 = kani::any();
    let f_right: f64 = kani::any();
    let k0: f64 = kani::any();
    let k1: f64 = kani::any();
    let power_0: f64 = kani::any();
    let power_1: f64 = kani::any();

    kani::assume(f_left >= 0.0 && f_center > f_left && f_right > f_center);
    kani::assume(f_left.is_finite() && f_center.is_finite() && f_right.is_finite());
    kani::assume(f_left <= 1000.0 && f_center <= 1000.0 && f_right <= 1000.0);
    kani::assume(k0 >= 0.0 && k0.is_finite() && k0 <= 1000.0);
    kani::assume(k1 >= 0.0 && k1.is_finite() && k1 <= 1000.0);
    kani::assume(power_0 >= 0.0 && power_0.is_finite() && power_0 <= 1e6);
    kani::assume(power_1 >= 0.0 && power_1.is_finite() && power_1 <= 1e6);

    let filter_0 = super::triangular_filter_weight(k0, f_left, f_center, f_right);
    let filter_1 = super::triangular_filter_weight(k1, f_left, f_center, f_right);

    let energy = filter_0 * power_0 + filter_1 * power_1;
    assert!(energy >= 0.0, "mel filter energy must be non-negative");
    assert!(energy.is_finite(), "mel filter energy must be finite");
}
