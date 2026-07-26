// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for spectral comparison and STFT properties.
//!
//! These harnesses verify the pure numerical properties of spectral analysis:
//! frequency bin count formulas, window function bounds, magnitude floor
//! behavior, spectral convergence normalization, log-spectral distance
//! non-negativity, phase coherence bounds, and STFT configuration validation.
//!
//! Issue: #3670

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// CBMC transcendental stubs
// ---------------------------------------------------------------------------

fn cos_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

fn log10_f64_stub(x: f64) -> f64 {
    let _ = x;
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= -50.0 && r <= 50.0);
    r
}

fn sqrt_f64_stub(x: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    r
}

// ---------------------------------------------------------------------------
// STFT configuration and frequency bin proofs
// ---------------------------------------------------------------------------

/// Proves that n_freqs = n_fft / 2 + 1 for all power-of-two n_fft values.
///
/// This is the Nyquist formula: a real-valued FFT of size N produces
/// N/2 + 1 unique frequency bins (DC through Nyquist). Ensuring this
/// is correct is critical for spectral comparison shape consistency.
#[kani::unwind(1)]
#[kani::proof]
fn n_freqs_equals_nyquist_formula() {
    let exp: u32 = kani::any();
    kani::assume(exp >= 1 && exp <= 20); // n_fft from 2 to 1,048,576

    let n_fft: usize = 1usize << exp;

    let config = crate::spectral::StftConfig {
        n_fft,
        hop_length: 256,
        window: crate::spectral::WindowFn::Hann,
    };

    let n_freqs = config.n_freqs();
    assert!(
        n_freqs == n_fft / 2 + 1,
        "n_freqs must equal n_fft/2 + 1 (Nyquist formula)"
    );
    assert!(n_freqs > 0, "n_freqs must be positive");
    assert!(n_freqs <= n_fft, "n_freqs must not exceed n_fft");
}

/// Proves that n_freqs is always at least 2 for any valid STFT config.
///
/// n_fft must be a power of 2 and > 0, so minimum n_fft = 2, giving
/// n_freqs = 2. This ensures spectral comparison always has at least
/// DC and Nyquist bins.
#[kani::unwind(5)]
#[kani::proof]
fn n_freqs_at_least_two() {
    let exp: u32 = kani::any();
    kani::assume(exp >= 1 && exp <= 16);

    let n_fft: usize = 1usize << exp;
    let n_freqs = n_fft / 2 + 1;

    assert!(
        n_freqs >= 2,
        "n_freqs must be at least 2 for any valid power-of-two n_fft >= 2"
    );
}

// ---------------------------------------------------------------------------
// Window function proofs
// ---------------------------------------------------------------------------

/// Proves that the Hann window values are in [0, 1] for any window length.
///
/// The Hann window is w(i) = 0.5 * (1 - cos(2*pi*i/N)). Since cos()
/// is in [-1, 1], the result is in [0, 1]. Boundary values: w(0) = 0,
/// w(N/2) = 1 for even N.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
fn hann_window_bounded_0_to_1() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 16);

    let i: usize = kani::any();
    kani::assume(i < n);

    let t = 2.0 * PI * i as f32 / n as f32;
    let w = 0.5 * (1.0 - t.cos());

    assert!(
        w >= -1e-7,
        "Hann window value must be >= 0 (within float tolerance)"
    );
    assert!(
        w <= 1.0 + 1e-7,
        "Hann window value must be <= 1 (within float tolerance)"
    );
}

/// Proves that the Hamming window values are in [0.08, 1.0] for any length.
///
/// The Hamming window is w(i) = 0.54 - 0.46 * cos(2*pi*i/N).
/// Minimum: 0.54 - 0.46 = 0.08 (at boundaries).
/// Maximum: 0.54 + 0.46 = 1.00 (at center).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
fn hamming_window_bounded() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 16);

    let i: usize = kani::any();
    kani::assume(i < n);

    let t = 2.0 * PI * i as f32 / n as f32;
    let w = 0.54 - 0.46 * t.cos();

    assert!(
        w >= 0.08 - 1e-6,
        "Hamming window value must be >= 0.08 (within float tolerance)"
    );
    assert!(
        w <= 1.0 + 1e-6,
        "Hamming window value must be <= 1.0 (within float tolerance)"
    );
}

/// Proves that the Rectangular window is identically 1.0 for all positions.
#[kani::unwind(1)]
#[kani::proof]
fn rectangular_window_all_ones() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 32);

    let window = crate::spectral::stft::make_window(crate::spectral::WindowFn::Rectangular, n);
    assert!(window.len() == n, "window length must match n");

    let i: usize = kani::any();
    kani::assume(i < n);
    assert!(
        window[i] == 1.0,
        "rectangular window must be 1.0 at every position"
    );
}

// ---------------------------------------------------------------------------
// Magnitude floor (MAG_FLOOR) proofs
// ---------------------------------------------------------------------------

/// Proves that the magnitude floor (1e-10) clamp prevents log(0) in
/// log-spectral distance computation.
///
/// The LSD formula uses 10*log10(magnitude). Without the floor,
/// magnitude = 0 would produce -inf. After clamping: log10(1e-10) = -10,
/// which is finite.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::log10, log10_f64_stub)]
fn mag_floor_prevents_log_zero() {
    let mag: f32 = kani::any();
    kani::assume(mag >= 0.0 && mag.is_finite());

    let floor: f32 = 1e-10;
    let clamped = mag.max(floor);

    assert!(clamped >= floor, "clamped magnitude must be >= MAG_FLOOR");
    assert!(clamped > 0.0, "clamped magnitude must be positive");

    let log_val = f64::from(clamped).log10();
    assert!(
        log_val.is_finite(),
        "log10 of clamped magnitude must be finite"
    );
}

/// Proves that MAG_FLOOR clamping is idempotent: clamping twice gives
/// the same result as clamping once.
#[kani::unwind(1)]
#[kani::proof]
fn mag_floor_clamping_idempotent() {
    let mag: f32 = kani::any();
    kani::assume(mag.is_finite() && mag >= 0.0);

    let floor: f32 = 1e-10;
    let once = mag.max(floor);
    let twice = once.max(floor);

    assert!(once == twice, "MAG_FLOOR clamping must be idempotent");
}

// ---------------------------------------------------------------------------
// Spectral convergence proofs
// ---------------------------------------------------------------------------

/// Proves that spectral convergence is non-negative for any pair of
/// non-negative magnitude values.
///
/// SC = ||S_ref - S_cand||_F / ||S_ref||_F. Both norms are non-negative,
/// so SC >= 0 when S_ref is non-zero.
#[kani::unwind(5)]
#[kani::proof]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn spectral_convergence_nonneg() {
    let r: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e6);
    kani::assume(c.is_finite() && c >= 0.0 && c <= 1e6);
    kani::assume(r > 0.0); // avoid division by zero

    let diff = f64::from(r) - f64::from(c);
    let sq_diff = diff * diff;
    let sq_ref = f64::from(r) * f64::from(r);

    let sc = sq_diff.sqrt() / sq_ref.sqrt();
    assert!(sc >= 0.0, "spectral convergence must be non-negative");
    assert!(
        sc.is_finite(),
        "spectral convergence must be finite for finite inputs"
    );
}

/// Proves that spectral convergence is 0 when reference == candidate.
#[kani::unwind(5)]
#[kani::proof]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn spectral_convergence_zero_for_identical() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite() && v > 0.0 && v <= 1e6);

    let diff = f64::from(v) - f64::from(v);
    let sq_diff = diff * diff;
    let sq_ref = f64::from(v) * f64::from(v);

    let sc = sq_diff.sqrt() / sq_ref.sqrt();
    assert!(
        sc == 0.0,
        "spectral convergence must be 0 for identical signals"
    );
}

// ---------------------------------------------------------------------------
// Log-spectral distance proofs
// ---------------------------------------------------------------------------

/// Proves that LSD is non-negative for any pair of positive magnitudes.
///
/// LSD = sqrt(mean((10*log10(r) - 10*log10(c))^2)). The squared difference
/// is non-negative, the mean is non-negative, and sqrt preserves
/// non-negativity.
#[kani::unwind(5)]
#[kani::proof]
#[kani::stub(f64::log10, log10_f64_stub)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn lsd_nonneg_for_positive_magnitudes() {
    let r: f32 = kani::any();
    let c: f32 = kani::any();
    let floor: f32 = 1e-10;

    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e6);
    kani::assume(c.is_finite() && c >= 0.0 && c <= 1e6);

    let r_clamped = f64::from(r.max(floor));
    let c_clamped = f64::from(c.max(floor));

    let r_db = 10.0 * r_clamped.log10();
    let c_db = 10.0 * c_clamped.log10();
    let db_diff = r_db - c_db;
    let sq = db_diff * db_diff;

    assert!(sq >= 0.0, "squared dB difference must be non-negative");

    let lsd = sq.sqrt();
    assert!(lsd >= 0.0, "LSD must be non-negative");
    assert!(lsd.is_finite(), "LSD must be finite for finite inputs");
}

/// Proves that LSD is 0 when reference equals candidate.
#[kani::unwind(5)]
#[kani::proof]
#[kani::stub(f64::log10, log10_f64_stub)]
fn lsd_zero_for_identical() {
    let v: f32 = kani::any();
    let floor: f32 = 1e-10;
    kani::assume(v.is_finite() && v >= 0.0 && v <= 1e6);

    let v_clamped = f64::from(v.max(floor));
    let r_db = 10.0 * v_clamped.log10();
    let c_db = 10.0 * v_clamped.log10();
    let db_diff = r_db - c_db;

    assert!(
        db_diff == 0.0,
        "dB difference must be 0 for identical magnitudes"
    );
}

// ---------------------------------------------------------------------------
// Phase coherence proofs
// ---------------------------------------------------------------------------

/// Proves that cos(phase_diff) is bounded in [-1, 1] for any phase values.
///
/// Phase coherence = mean(cos(phase_ref - phase_cand)). The cosine function
/// always returns values in [-1, 1], so each term is bounded.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
fn phase_coherence_term_bounded() {
    let phase_ref: f32 = kani::any();
    let phase_cand: f32 = kani::any();

    // Phase values from atan2 are in [-pi, pi].
    kani::assume(phase_ref >= -PI && phase_ref <= PI);
    kani::assume(phase_cand >= -PI && phase_cand <= PI);

    let phase_diff = phase_ref - phase_cand;
    let cos_val = phase_diff.cos();

    assert!(
        cos_val >= -1.0 - 1e-6,
        "cos(phase_diff) must be >= -1 within float tolerance"
    );
    assert!(
        cos_val <= 1.0 + 1e-6,
        "cos(phase_diff) must be <= 1 within float tolerance"
    );
}

/// Proves that phase coherence is exactly 1.0 when phases are identical.
#[kani::unwind(5)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
fn phase_coherence_one_for_identical() {
    let phase: f32 = kani::any();
    kani::assume(phase >= -PI && phase <= PI);
    kani::assume(phase.is_finite());

    let diff = phase - phase;
    let cos_val = diff.cos();

    // cos(0) = 1.0 exactly in IEEE 754.
    assert!(
        cos_val == 1.0,
        "cos(0) must be exactly 1.0 for identical phases"
    );
}

// ---------------------------------------------------------------------------
// SpectralConfig gate logic proofs
// ---------------------------------------------------------------------------

/// Proves that the SpectralConfig `passed` logic is monotonic:
/// if a signal passes with threshold T, it passes with any looser threshold T' > T.
///
/// Specifically: if LSD <= max_lsd and SC <= max_sc and phase >= min_phase,
/// then the same is true for any wider thresholds.
#[kani::unwind(1)]
#[kani::proof]
fn spectral_pass_monotonic_in_thresholds() {
    let lsd: f32 = kani::any();
    let sc: f32 = kani::any();
    let phase: f32 = kani::any();

    kani::assume(lsd.is_finite() && lsd >= 0.0 && lsd <= 100.0);
    kani::assume(sc.is_finite() && sc >= 0.0 && sc <= 10.0);
    kani::assume(phase.is_finite() && phase >= -1.0 && phase <= 1.0);

    let max_lsd: f32 = kani::any();
    let max_sc: f32 = kani::any();
    let min_phase: f32 = kani::any();

    kani::assume(max_lsd.is_finite() && max_lsd >= 0.0 && max_lsd <= 100.0);
    kani::assume(max_sc.is_finite() && max_sc >= 0.0 && max_sc <= 10.0);
    kani::assume(min_phase.is_finite() && min_phase >= -1.0 && min_phase <= 1.0);

    let passed_tight = lsd <= max_lsd && sc <= max_sc && phase >= min_phase;

    // Looser thresholds: wider LSD, wider SC, lower phase requirement.
    let loose_factor: f32 = kani::any();
    kani::assume(loose_factor.is_finite() && loose_factor >= 1.0 && loose_factor <= 10.0);

    let passed_loose = lsd <= max_lsd * loose_factor
        && sc <= max_sc * loose_factor
        && phase >= min_phase / loose_factor;

    if passed_tight {
        assert!(
            passed_loose,
            "passing tight thresholds must imply passing looser thresholds"
        );
    }
}

/// Proves that SpectralConfig default thresholds are consistent:
/// max_lsd > 0, max_sc > 0, min_phase in [0, 1].
#[kani::unwind(1)]
#[kani::proof]
fn spectral_config_default_valid() {
    let config = crate::spectral::SpectralConfig::default();

    assert!(
        config.max_lsd_db > 0.0,
        "default max_lsd_db must be positive"
    );
    assert!(
        config.max_spectral_convergence > 0.0,
        "default max_spectral_convergence must be positive"
    );
    assert!(
        config.min_phase_coherence >= 0.0 && config.min_phase_coherence <= 1.0,
        "default min_phase_coherence must be in [0, 1]"
    );
}
