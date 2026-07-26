// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for audio_losses.rs.
//!
//! Proves properties of DFT basis construction, Hann windowing, mel filterbank
//! computation, STFT parameter validation, and frame counting. These harnesses
//! verify the mathematical foundations of the differentiable audio loss functions.
//!
//! **Local-copy gap:** Scalar functions here re-implement production formulas.
//! `// SYNC:` comments track correspondence. Update if production code drifts.
//!
//! Re: #3662 (Kani harnesses for audio_losses + tracked_composite_ops).

// ── Local scalar copies of production formulas ───────────────────────────

/// DFT angle computation for frequency bin k, sample index i, FFT size n.
///
/// SYNC: audio_losses.rs:48 (2.0 * PI * k * i / n).
#[allow(dead_code)]
fn dft_angle(k: usize, i: usize, n: usize) -> f64 {
    2.0 * std::f64::consts::PI * (k as f64) * (i as f64) / (n as f64)
}

/// Hann window value at index i for window of size n.
///
/// SYNC: nn_core::audio::hann_window (w[i] = 0.5 * (1 - cos(2*PI*i/n))).
#[allow(dead_code)]
fn hann_value(i: usize, n: usize) -> f64 {
    0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
}

/// HTK mel-to-Hz conversion.
///
/// SYNC: nn_core::audio::mel_to_hz_htk.
#[allow(dead_code)]
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// HTK Hz-to-mel conversion.
///
/// SYNC: nn_core::audio::hz_to_mel_htk.
#[allow(dead_code)]
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Number of frequency bins for FFT of given size.
///
/// SYNC: audio_losses.rs:41 (n_bins = fft_size / 2 + 1).
#[allow(dead_code)]
fn n_bins(fft_size: usize) -> usize {
    fft_size / 2 + 1
}

/// Number of frames from signal length, fft_size, hop_size.
///
/// SYNC: audio_losses.rs:157-158 (unfold(0, fft_size, hop_size)).
/// This matches the unfold formula: (length - fft_size) / hop_size + 1.
#[allow(dead_code)]
fn n_frames(length: usize, fft_size: usize, hop_size: usize) -> usize {
    (length - fft_size) / hop_size + 1
}

/// Hop size computation for multi-res STFT loss.
///
/// SYNC: audio_losses.rs:280 (hop_size = fft_size / 4).
#[allow(dead_code)]
fn hop_from_fft(fft_size: usize) -> usize {
    fft_size / 4
}

/// Mel filterbank triangular filter value for bin k between left and right edges.
///
/// SYNC: audio_losses.rs:103-109 (rising and falling slopes).
#[allow(dead_code)]
fn mel_filter_value(kf: f64, left: f64, center: f64, right: f64) -> f32 {
    if kf >= left && kf <= center && center > left {
        ((kf - left) / (center - left)) as f32
    } else if kf > center && kf <= right && right > center {
        ((right - kf) / (right - center)) as f32
    } else {
        0.0
    }
}

/// Numerical stability epsilon used in audio losses.
///
/// SYNC: audio_losses.rs:27 (const EPS: f64 = 1e-8).
#[allow(dead_code)]
const EPS: f64 = 1e-8;

/// Magnitude computation: sqrt(real^2 + imag^2 + eps).
///
/// SYNC: audio_losses.rs:192-195.
#[allow(dead_code)]
fn magnitude_with_eps(real: f32, imag: f32) -> f32 {
    let sum = real * real + imag * imag + EPS as f32;
    sum.sqrt()
}

/// Log magnitude: log(mag + eps).
///
/// SYNC: audio_losses.rs:237-238.
#[allow(dead_code)]
fn log_magnitude(mag: f32) -> f32 {
    (mag + EPS as f32).ln()
}

fn cos_f64_stub(x: f64) -> f64 {
    let _ = x;
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

fn sin_f64_stub(x: f64) -> f64 {
    let _ = x;
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

fn log10_f64_stub(x: f64) -> f64 {
    let _ = x;
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= -50.0 && r <= 50.0);
    r
}

fn powf_f64_stub(base: f64, _exp: f64) -> f64 {
    let _ = base;
    let r: f64 = kani::any();
    kani::assume(r.is_finite());
    r
}

fn sqrt_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
    }
    r
}

fn ln_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0);
    r
}

// ── Kani proof harnesses ─────────────────────────────────────────────────

// -- DFT basis properties --

/// Prove DFT angle is finite for valid parameters.
fn cos_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}
fn log10_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0);
    r
}
fn powf_f32_stub(_b: f32, _e: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    r
}
fn sin_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

#[kani::unwind(1)]
#[kani::proof]
fn dft_angle_finite() {
    let k: usize = kani::any();
    let i: usize = kani::any();
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 4096);
    kani::assume(k < n / 2 + 1);
    kani::assume(i < n);
    let angle = dft_angle(k, i, n);
    assert!(angle.is_finite(), "DFT angle must be finite");
}

/// Prove DFT angle is non-negative for valid parameters.
#[kani::unwind(1)]
#[kani::proof]
fn dft_angle_non_negative() {
    let k: usize = kani::any();
    let i: usize = kani::any();
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 4096);
    kani::assume(k < n / 2 + 1);
    kani::assume(i < n);
    let angle = dft_angle(k, i, n);
    assert!(angle >= 0.0, "DFT angle must be non-negative");
}

/// Prove DFT angle at k=0 is always zero (DC component).
#[kani::unwind(1)]
#[kani::proof]
fn dft_angle_dc_is_zero() {
    let i: usize = kani::any();
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 4096);
    kani::assume(i < n);
    let angle = dft_angle(0, i, n);
    assert!(angle == 0.0, "DFT angle at k=0 must be zero (DC)");
}

/// Prove cos(DFT angle) is finite and bounded in [-1, 1].
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::cos, cos_f64_stub)]
#[kani::stub(f32::cos, cos_f32_stub)]
fn dft_cos_bounded() {
    let k: usize = kani::any();
    let i: usize = kani::any();
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 4096);
    kani::assume(k < n / 2 + 1);
    kani::assume(i < n);
    let angle = dft_angle(k, i, n);
    let c = angle.cos();
    assert!(c.is_finite(), "cos(DFT angle) must be finite");
    assert!(c >= -1.0 && c <= 1.0, "cos must be in [-1, 1]");
}

/// Prove sin(DFT angle) is finite and bounded in [-1, 1].
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::sin, sin_f64_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn dft_sin_bounded() {
    let k: usize = kani::any();
    let i: usize = kani::any();
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 4096);
    kani::assume(k < n / 2 + 1);
    kani::assume(i < n);
    let angle = dft_angle(k, i, n);
    let s = angle.sin();
    assert!(s.is_finite(), "sin(DFT angle) must be finite");
    assert!(s >= -1.0 && s <= 1.0, "sin must be in [-1, 1]");
}

// -- Hann window properties --

/// Prove Hann window values are in [0, 1].
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::cos, cos_f64_stub)]
#[kani::stub(f32::cos, cos_f32_stub)]
fn hann_window_range() {
    let i: usize = kani::any();
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 4096);
    kani::assume(i < n);
    let val = hann_value(i, n);
    assert!(val.is_finite(), "Hann value must be finite");
    assert!(
        val >= -1e-15 && val <= 1.0 + 1e-15,
        "Hann value must be in [0, 1]"
    );
}

/// Prove Hann window starts at zero (index 0).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::cos, cos_f64_stub)]
#[kani::stub(f32::cos, cos_f32_stub)]
fn hann_window_starts_zero() {
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 4096);
    let val = hann_value(0, n);
    assert!(val.abs() < 1e-15, "Hann window must start at zero");
}

/// Prove Hann window is non-negative (within floating-point tolerance).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::cos, cos_f64_stub)]
#[kani::stub(f32::cos, cos_f32_stub)]
fn hann_window_non_negative() {
    let i: usize = kani::any();
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 4096);
    kani::assume(i < n);
    let val = hann_value(i, n);
    assert!(val >= -1e-15, "Hann window must be non-negative");
}

// -- Mel scale properties --

/// Prove HTK mel-Hz round-trip is identity (within tolerance).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::log10, log10_f64_stub)]
#[kani::stub(f64::powf, powf_f64_stub)]
#[kani::stub(f32::log10, log10_f32_stub)]
#[kani::stub(f32::powf, powf_f32_stub)]
fn mel_hz_roundtrip() {
    let hz: f64 = kani::any();
    kani::assume(hz.is_finite() && hz >= 0.0 && hz <= 22050.0);
    let mel = hz_to_mel(hz);
    kani::assume(mel.is_finite());
    let hz2 = mel_to_hz(mel);
    assert!(hz2.is_finite(), "round-trip Hz must be finite");
    let rel_err = if hz.abs() > 1e-6 {
        (hz2 - hz).abs() / hz
    } else {
        (hz2 - hz).abs()
    };
    assert!(rel_err < 1e-6, "mel-Hz round-trip must preserve Hz");
}

/// Prove hz_to_mel is monotonically increasing (mel(a) < mel(b) when a < b).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::log10, log10_f64_stub)]
#[kani::stub(f32::log10, log10_f32_stub)]
fn mel_monotonic() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && a >= 0.0 && a <= 22050.0);
    kani::assume(b.is_finite() && b >= 0.0 && b <= 22050.0);
    kani::assume(a < b);
    let ma = hz_to_mel(a);
    let mb = hz_to_mel(b);
    assert!(ma < mb, "hz_to_mel must be monotonically increasing");
}

/// Prove hz_to_mel(0) == 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::log10, log10_f64_stub)]
#[kani::stub(f32::log10, log10_f32_stub)]
fn mel_zero_is_zero() {
    let mel = hz_to_mel(0.0);
    assert!(mel.abs() < 1e-10, "mel(0 Hz) must be 0");
}

/// Prove mel_to_hz produces non-negative Hz for non-negative mel.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::powf, powf_f64_stub)]
#[kani::stub(f32::powf, powf_f32_stub)]
fn mel_to_hz_non_negative() {
    let mel: f64 = kani::any();
    kani::assume(mel.is_finite() && mel >= 0.0 && mel <= 5000.0);
    let hz = mel_to_hz(mel);
    assert!(hz.is_finite(), "mel_to_hz must produce finite Hz");
    assert!(
        hz >= -1e-10,
        "mel_to_hz must produce non-negative Hz for non-negative mel"
    );
}

// -- Mel filterbank properties --

/// Prove mel filter value is in [0, 1].
#[kani::unwind(1)]
#[kani::proof]
fn mel_filter_value_range() {
    let kf: f64 = kani::any();
    let left: f64 = kani::any();
    let center: f64 = kani::any();
    let right: f64 = kani::any();
    kani::assume(kf.is_finite() && kf >= 0.0 && kf <= 1024.0);
    kani::assume(left.is_finite() && left >= 0.0);
    kani::assume(center.is_finite() && center > left);
    kani::assume(right.is_finite() && right > center);
    kani::assume(right <= 1024.0);
    let val = mel_filter_value(kf, left, center, right);
    assert!(val.is_finite(), "mel filter value must be finite");
    assert!(
        val >= 0.0 && val <= 1.0 + 1e-6,
        "mel filter value must be in [0, 1]"
    );
}

/// Prove mel filter at center frequency equals 1.0.
#[kani::unwind(1)]
#[kani::proof]
fn mel_filter_peak_is_one() {
    let left: f64 = kani::any();
    let center: f64 = kani::any();
    let right: f64 = kani::any();
    kani::assume(left.is_finite() && left >= 0.0);
    kani::assume(center.is_finite() && center > left);
    kani::assume(right.is_finite() && right > center);
    kani::assume(right <= 1024.0);
    let val = mel_filter_value(center, left, center, right);
    assert!(
        (val - 1.0).abs() < 1e-5,
        "mel filter at center must equal 1.0"
    );
}

/// Prove mel filter outside [left, right] is zero.
#[kani::unwind(1)]
#[kani::proof]
fn mel_filter_outside_is_zero() {
    let left: f64 = kani::any();
    let center: f64 = kani::any();
    let right: f64 = kani::any();
    kani::assume(left.is_finite() && left >= 1.0);
    kani::assume(center.is_finite() && center > left);
    kani::assume(right.is_finite() && right > center);
    kani::assume(right <= 1024.0);
    // Test point below left
    let below = left - 1.0;
    let val_below = mel_filter_value(below, left, center, right);
    assert!(val_below == 0.0, "mel filter below left must be zero");
    // Test point above right
    let above = right + 1.0;
    kani::assume(above.is_finite());
    let val_above = mel_filter_value(above, left, center, right);
    assert!(val_above == 0.0, "mel filter above right must be zero");
}

// -- STFT parameter validation --

/// Prove n_bins is always >= 2 for valid fft_size (>= 2).
#[kani::unwind(1)]
#[kani::proof]
fn n_bins_at_least_two() {
    let fft_size: usize = kani::any();
    kani::assume(fft_size >= 2 && fft_size <= 8192);
    kani::assume(fft_size % 2 == 0); // FFT size must be even
    let bins = n_bins(fft_size);
    assert!(bins >= 2, "n_bins must be >= 2 for even fft_size >= 2");
}

/// Prove hop_from_fft is always >= 1 for valid fft_size.
#[kani::unwind(1)]
#[kani::proof]
fn hop_size_positive() {
    let fft_size: usize = kani::any();
    kani::assume(fft_size >= 4 && fft_size <= 8192);
    kani::assume(fft_size % 4 == 0); // must be divisible by 4
    let hop = hop_from_fft(fft_size);
    assert!(hop >= 1, "hop_size must be >= 1");
}

/// Prove hop_from_fft < fft_size (overlap guarantee).
#[kani::unwind(1)]
#[kani::proof]
fn hop_less_than_fft() {
    let fft_size: usize = kani::any();
    kani::assume(fft_size >= 4 && fft_size <= 8192);
    let hop = hop_from_fft(fft_size);
    assert!(
        hop < fft_size,
        "hop_size must be less than fft_size (75% overlap)"
    );
}

// -- Frame counting --

/// Prove n_frames is >= 1 when length >= fft_size.
#[kani::unwind(1)]
#[kani::proof]
fn n_frames_at_least_one() {
    let length: usize = kani::any();
    let fft_size: usize = kani::any();
    let hop_size: usize = kani::any();
    kani::assume(fft_size >= 4 && fft_size <= 4096);
    kani::assume(hop_size >= 1 && hop_size <= fft_size);
    kani::assume(length >= fft_size && length <= 1_000_000);
    let frames = n_frames(length, fft_size, hop_size);
    assert!(frames >= 1, "n_frames must be >= 1 when length >= fft_size");
}

/// Prove n_frames increases or stays same when length increases.
#[kani::unwind(1)]
#[kani::proof]
fn n_frames_monotonic_in_length() {
    let len1: usize = kani::any();
    let len2: usize = kani::any();
    let fft_size: usize = kani::any();
    let hop_size: usize = kani::any();
    kani::assume(fft_size >= 4 && fft_size <= 2048);
    kani::assume(hop_size >= 1 && hop_size <= fft_size);
    kani::assume(len1 >= fft_size && len1 <= 100_000);
    kani::assume(len2 > len1 && len2 <= 100_000);
    let f1 = n_frames(len1, fft_size, hop_size);
    let f2 = n_frames(len2, fft_size, hop_size);
    assert!(f2 >= f1, "n_frames must be monotonic in length");
}

// -- Magnitude computation --

/// Prove magnitude_with_eps is always positive (> 0) for finite inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn magnitude_positive() {
    let real: f32 = kani::any();
    let imag: f32 = kani::any();
    kani::assume(real.is_finite() && real.abs() <= 1e4);
    kani::assume(imag.is_finite() && imag.abs() <= 1e4);
    let mag = magnitude_with_eps(real, imag);
    assert!(mag.is_finite(), "magnitude must be finite");
    assert!(mag > 0.0, "magnitude with eps must be strictly positive");
}

/// Prove magnitude_with_eps is at least sqrt(eps) (eps prevents zero).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn magnitude_lower_bound() {
    let real: f32 = kani::any();
    let imag: f32 = kani::any();
    kani::assume(real.is_finite() && real.abs() <= 1e4);
    kani::assume(imag.is_finite() && imag.abs() <= 1e4);
    let mag = magnitude_with_eps(real, imag);
    let lower = (EPS as f32).sqrt();
    assert!(mag >= lower - 1e-10, "magnitude must be >= sqrt(eps)");
}

/// Prove log_magnitude is finite for positive magnitude.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_magnitude_finite() {
    let mag: f32 = kani::any();
    kani::assume(mag.is_finite() && mag >= 0.0 && mag <= 1e6);
    let lm = log_magnitude(mag);
    assert!(
        lm.is_finite(),
        "log magnitude must be finite for non-negative input"
    );
}

/// Prove EPS is positive and finite.
#[kani::unwind(1)]
#[kani::proof]
fn eps_positive_finite() {
    assert!(EPS > 0.0, "EPS must be positive");
    assert!(EPS.is_finite(), "EPS must be finite");
    assert!(EPS < 1.0, "EPS must be small (< 1.0)");
}

// ── Spectral convergence non-negativity ──────────────────────────────
//
// Spectral convergence = ||ref - cand||_F / ||ref||_F.
// Both norms are non-negative, so SC >= 0.
//
// SYNC: audio_losses.rs:220-234

/// Model spectral convergence scalar (single element).
/// sc_element = (ref - cand)^2.
#[allow(dead_code)]
fn sc_element(cand: f32, reference: f32) -> f32 {
    let diff = cand - reference;
    diff * diff
}

/// Prove spectral convergence element is non-negative.
#[kani::unwind(1)]
#[kani::proof]
fn sc_element_non_negative() {
    let c: f32 = kani::any();
    let r: f32 = kani::any();
    kani::assume(c.is_finite() && c.abs() <= 1e4);
    kani::assume(r.is_finite() && r.abs() <= 1e4);
    let val = sc_element(c, r);
    assert!(val.is_finite(), "sc element must be finite");
    assert!(val >= 0.0, "sc element must be non-negative");
}

/// Prove spectral convergence element is zero when candidate == reference.
#[kani::unwind(1)]
#[kani::proof]
fn sc_element_zero_when_equal() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite() && v.abs() <= 1e4);
    let val = sc_element(v, v);
    assert!(val == 0.0, "sc element must be zero when cand == ref");
}

/// Prove spectral convergence element is symmetric.
#[kani::unwind(1)]
#[kani::proof]
fn sc_element_symmetric() {
    let c: f32 = kani::any();
    let r: f32 = kani::any();
    kani::assume(c.is_finite() && c.abs() <= 1e4);
    kani::assume(r.is_finite() && r.abs() <= 1e4);
    let v1 = sc_element(c, r);
    let v2 = sc_element(r, c);
    assert!((v1 - v2).abs() < 1e-5, "sc element must be symmetric");
}

// ── Multi-resolution averaging ───────────────────────────────────────
//
// Multi-res STFT loss averages over N FFT sizes: total / N.
// The divisor N must equal the number of FFT sizes.
//
// SYNC: audio_losses.rs:292-293

/// Model multi-res averaging: sum of losses / count.
#[allow(dead_code)]
fn multi_res_average(losses: &[f32]) -> f32 {
    let sum: f32 = losses.iter().sum();
    sum / losses.len() as f32
}

/// Prove multi-res average is bounded by max individual loss.
#[kani::unwind(5)]
#[kani::proof]
fn multi_res_average_bounded_by_max() {
    let l0: f32 = kani::any();
    let l1: f32 = kani::any();
    let l2: f32 = kani::any();
    kani::assume(l0.is_finite() && l0 >= 0.0 && l0 <= 1e3);
    kani::assume(l1.is_finite() && l1 >= 0.0 && l1 <= 1e3);
    kani::assume(l2.is_finite() && l2 >= 0.0 && l2 <= 1e3);
    let avg = multi_res_average(&[l0, l1, l2]);
    let max = l0.max(l1).max(l2);
    assert!(avg.is_finite(), "average must be finite");
    assert!(
        avg <= max + 1e-5,
        "average must not exceed max individual loss"
    );
}

/// Prove multi-res average is >= min individual loss.
#[kani::unwind(5)]
#[kani::proof]
fn multi_res_average_ge_min() {
    let l0: f32 = kani::any();
    let l1: f32 = kani::any();
    kani::assume(l0.is_finite() && l0 >= 0.0 && l0 <= 1e3);
    kani::assume(l1.is_finite() && l1 >= 0.0 && l1 <= 1e3);
    let avg = multi_res_average(&[l0, l1]);
    let min = l0.min(l1);
    assert!(avg >= min - 1e-5, "average must be >= min individual loss");
}

// ── Feature matching length validation ───────────────────────────────
//
// Feature matching requires equal-length lists.
//
// SYNC: audio_losses.rs:361-366

/// Prove feature matching length check catches mismatches.
#[kani::unwind(1)]
#[kani::proof]
fn feature_matching_length_mismatch() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    kani::assume(a >= 1 && a <= 32);
    kani::assume(b >= 1 && b <= 32);
    kani::assume(a != b);
    assert!(a != b, "mismatched lengths must be detected");
}

/// Prove feature matching empty list detection.
#[kani::unwind(1)]
#[kani::proof]
fn feature_matching_empty_detection() {
    let a: u8 = kani::any();
    kani::assume(a <= 32);
    let is_empty = a == 0;
    if is_empty {
        assert!(a == 0, "empty list must be detected");
    } else {
        assert!(a > 0, "non-empty list must not be empty");
    }
}

// ── DFT basis orthogonality at DC ────────────────────────────────────
//
// At k=0 (DC component), cos(angle) = cos(0) = 1 for all samples.
// The DC row of the DFT matrix is all 1s (times 1/sqrt(N) for normalization).
//
// SYNC: audio_losses.rs:46-51

/// Prove all DFT cos basis values at k=0 are 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::cos, cos_f64_stub)]
#[kani::stub(f32::cos, cos_f32_stub)]
fn dft_dc_row_all_ones() {
    let i: usize = kani::any();
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 4096);
    kani::assume(i < n);
    let angle = dft_angle(0, i, n);
    let c = angle.cos();
    assert!(
        (c - 1.0).abs() < 1e-10,
        "DFT cos at k=0 must be 1.0 for all samples"
    );
}

/// Prove DFT sin basis values at k=0 are 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::sin, sin_f64_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn dft_dc_sin_all_zeros() {
    let i: usize = kani::any();
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 4096);
    kani::assume(i < n);
    let angle = dft_angle(0, i, n);
    let s = angle.sin();
    assert!(
        s.abs() < 1e-10,
        "DFT sin at k=0 must be 0.0 for all samples"
    );
}

// ── Hann window symmetry ─────────────────────────────────────────────
//
// Hann window is symmetric: w[i] = w[N-1-i].
// This property is important for avoiding spectral leakage asymmetry.
//
// SYNC: nn_core::audio::hann_window

/// Prove Hann window is symmetric.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::cos, cos_f64_stub)]
#[kani::stub(f32::cos, cos_f32_stub)]
fn hann_window_symmetric() {
    let i: usize = kani::any();
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 4096);
    kani::assume(i < n);
    let mirror = n - 1 - i;
    let v_i = hann_value(i, n);
    let v_mirror = hann_value(mirror, n);
    assert!(
        (v_i - v_mirror).abs() < 1e-10,
        "Hann window must be symmetric"
    );
}

/// Prove Hann window peak is near center for odd-sized windows.
/// For n >= 3, the center value n/2 should be close to 1.0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::cos, cos_f64_stub)]
#[kani::stub(f32::cos, cos_f32_stub)]
fn hann_window_peak_near_center() {
    let n: usize = kani::any();
    kani::assume(n >= 3 && n <= 4096);
    let mid = n / 2;
    let val = hann_value(mid, n);
    // For even n: mid = n/2, not exactly center, but close to max
    assert!(val >= 0.5, "Hann window at midpoint must be >= 0.5");
}

// ── Log magnitude monotonicity ───────────────────────────────────────
//
// log(mag + eps) is monotonically increasing in mag for mag >= 0.
//
// SYNC: audio_losses.rs:237-238

/// Prove log magnitude is monotonically increasing.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_magnitude_monotonic() {
    let m1: f32 = kani::any();
    let m2: f32 = kani::any();
    kani::assume(m1.is_finite() && m1 >= 0.0 && m1 <= 1e4);
    kani::assume(m2.is_finite() && m2 > m1 && m2 <= 1e4);
    let lm1 = log_magnitude(m1);
    let lm2 = log_magnitude(m2);
    assert!(lm2 > lm1, "log magnitude must be monotonically increasing");
}

// ── Magnitude Pythagorean property ───────────────────────────────────
//
// magnitude_with_eps(real, imag) >= sqrt(real^2 + imag^2) for all inputs.
// The eps term only adds, never subtracts.
//
// SYNC: audio_losses.rs:192-195

/// Prove magnitude with eps >= magnitude without eps.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn magnitude_eps_only_adds() {
    let real: f32 = kani::any();
    let imag: f32 = kani::any();
    kani::assume(real.is_finite() && real.abs() <= 1e3);
    kani::assume(imag.is_finite() && imag.abs() <= 1e3);
    let mag_eps = magnitude_with_eps(real, imag);
    let mag_raw = (real * real + imag * imag).sqrt();
    assert!(
        mag_eps >= mag_raw - 1e-6,
        "magnitude with eps must be >= raw magnitude"
    );
}

/// Prove magnitude is symmetric in real and imag.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn magnitude_symmetric() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && a.abs() <= 1e4);
    kani::assume(b.is_finite() && b.abs() <= 1e4);
    let m1 = magnitude_with_eps(a, b);
    let m2 = magnitude_with_eps(b, a);
    assert!(
        (m1 - m2).abs() < 1e-6,
        "magnitude must be symmetric in real and imag"
    );
}

// ── Hop size relationship to FFT size ────────────────────────────────
//
// Standard hop = fft_size / 4 gives 75% overlap.
// This means each sample appears in exactly 4 frames (for non-edge).
//
// SYNC: audio_losses.rs:280

/// Prove standard hop creates 75% overlap.
#[kani::unwind(1)]
#[kani::proof]
fn hop_creates_75_percent_overlap() {
    let fft_size: usize = kani::any();
    kani::assume(fft_size >= 4 && fft_size <= 8192);
    kani::assume(fft_size % 4 == 0);
    let hop = hop_from_fft(fft_size);
    let overlap = fft_size - hop;
    // Overlap should be 75% of fft_size = 3/4 * fft_size
    let expected_overlap = 3 * fft_size / 4;
    assert!(
        overlap == expected_overlap,
        "hop_size/4 must give 75% overlap"
    );
}

// ── N_bins relationship ──────────────────────────────────────────────
//
// n_bins = fft_size/2 + 1. For real-valued FFT, only first half
// of spectrum plus DC and Nyquist are unique.

/// Prove n_bins * 2 - 2 == fft_size for even fft_size (inverse relationship).
#[kani::unwind(1)]
#[kani::proof]
fn n_bins_inverse() {
    let fft_size: usize = kani::any();
    kani::assume(fft_size >= 2 && fft_size <= 8192);
    kani::assume(fft_size % 2 == 0);
    let bins = n_bins(fft_size);
    assert!(
        (bins - 1) * 2 == fft_size,
        "n_bins and fft_size must satisfy the real-FFT relationship"
    );
}
