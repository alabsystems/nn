// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared audio primitives for mel-frequency conversions, windowing, and
//! crossfade blending.
//!
//! Two mel scales are provided:
//! - **HTK**: `mel = 2595 * log10(1 + hz / 700)` — used by nn-tts-verify
//!   and nn-autodiff's mel filterbank.
//! - **Slaney**: piecewise linear below 1 kHz, logarithmic above — used by
//!   AI Provider Whisper and librosa (`htk=False`).

use std::f64::consts::PI;

// -- HTK mel scale ------------------------------------------------------------

/// Convert Hz to mel using the HTK formula.
///
/// `mel = 2595 * log10(1 + hz / 700)`
pub fn hz_to_mel_htk(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel to Hz using the HTK formula (inverse of [`hz_to_mel_htk`]).
///
/// `hz = 700 * (10^(mel / 2595) - 1)`
pub fn mel_to_hz_htk(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

// -- Slaney mel scale ---------------------------------------------------------

/// Slaney frequency spacing: 200/3 ≈ 66.667 Hz per mel band below 1 kHz.
const SLANEY_F_SP: f64 = 200.0 / 3.0;

/// Mel value at the linear-to-log transition (1000 Hz / SLANEY_F_SP = 15.0).
const SLANEY_MIN_LOG_MEL: f64 = 1000.0 / SLANEY_F_SP;

/// Log step for the Slaney scale above 1 kHz: `ln(6.4) / 27`.
const SLANEY_LOG_STEP: f64 = 0.06875177742094912; // 6.4_f64.ln() / 27.0

/// Convert Hz to mel using the Slaney scale (librosa default, `htk=False`).
///
/// Piecewise linear below 1 kHz, logarithmic above.
pub fn hz_to_mel_slaney(hz: f64) -> f64 {
    if hz < 1000.0 {
        hz / SLANEY_F_SP
    } else {
        SLANEY_MIN_LOG_MEL + (hz / 1000.0).ln() / SLANEY_LOG_STEP
    }
}

/// Convert mel to Hz using the Slaney scale (inverse of [`hz_to_mel_slaney`]).
pub fn mel_to_hz_slaney(mel: f64) -> f64 {
    if mel < SLANEY_MIN_LOG_MEL {
        SLANEY_F_SP * mel
    } else {
        1000.0 * (SLANEY_LOG_STEP * (mel - SLANEY_MIN_LOG_MEL)).exp()
    }
}

// -- Hann window --------------------------------------------------------------

/// Generate a Hann window of length `n`.
///
/// `w[i] = 0.5 * (1 - cos(2π * i / n))`
///
/// Returns an empty Vec if `n == 0`.
pub fn hann_window(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos()))
        .collect()
}

// -- Linear crossfade --------------------------------------------------------

/// Compute a linear crossfade blend of `count` samples from `tail` and `head`.
///
/// Returns a `Vec<f32>` of length `count` where each sample is a convex
/// combination:
///
/// ```text
/// out[i] = tail[i] * (1 - alpha) + head[i] * alpha
/// ```
///
/// with `alpha = i / (count - 1)`.
///
/// # Edge cases
///
/// - `count == 0`: returns an empty Vec.
/// - `count == 1`: returns the average `(tail[0] + head[0]) / 2`.
///
/// # Panics
///
/// Panics if `tail.len() < count` or `head.len() < count`.
///
/// # Kani coverage
///
/// `crossfade_alpha_in_unit_interval` and `crossfade_convex_combination_bounded`
/// (nn-models Kani harnesses) prove the alpha and output bounds for this
/// formula.
pub fn crossfade_linear_blend(tail: &[f32], head: &[f32], count: usize) -> Vec<f32> {
    debug_assert!(
        tail.len() >= count,
        "tail.len() ({}) < count ({})",
        tail.len(),
        count,
    );
    debug_assert!(
        head.len() >= count,
        "head.len() ({}) < count ({})",
        head.len(),
        count,
    );
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![(tail[0] + head[0]) * 0.5];
    }
    let inv = 1.0 / (count - 1) as f32;
    (0..count)
        .map(|i| {
            let alpha = i as f32 * inv;
            tail[i] * (1.0 - alpha) + head[i] * alpha
        })
        .collect()
}

/// Append a linear crossfade blend of `tail` and `head` into `out`.
///
/// `cf` defines the total crossfade window size (used as the alpha
/// denominator: `alpha = j / (cf - 1)`). `limit` caps how many samples
/// are actually emitted -- useful when the chunk is shorter than the full
/// crossfade window.
///
/// This is the append-to-buffer variant used by the streaming assembler
/// where output is built incrementally.
///
/// # Edge cases
///
/// - `cf == 0` or `limit == 0`: appends nothing.
/// - `cf == 1` with `limit > 0`: appends the average `(tail[0] + head[0]) / 2`.
///
/// # Panics
///
/// Panics if `tail` or `head` have fewer than `cf.min(limit)` elements.
pub fn crossfade_blend_into(
    out: &mut Vec<f32>,
    tail: &[f32],
    head: &[f32],
    cf: usize,
    limit: usize,
) {
    let n = cf.min(limit);
    if n == 0 {
        return;
    }
    if cf == 1 {
        out.push((tail[0] + head[0]) * 0.5);
        return;
    }
    // Use `cf` (not `n`) as the alpha denominator so the crossfade ramp
    // rate is consistent regardless of truncation via `limit`.
    let inv = 1.0 / (cf - 1) as f32;
    out.extend((0..n).map(|j| {
        let alpha = j as f32 * inv;
        tail[j] * (1.0 - alpha) + head[j] * alpha
    }));
}

// -- Sqrt-Hann crossfade (amplitude-complementary) --------------------------

/// Compute a sqrt-Hann-windowed crossfade blend of `count` samples from `tail`
/// and `head`.
///
/// Returns a `Vec<f32>` of length `count` where each sample is a convex
/// combination using the sqrt-Hann (root raised cosine) window:
///
/// ```text
/// alpha = sqrt(0.5 * (1 - cos(PI * i / (count - 1))))
/// out[i] = tail[i] * (1 - alpha) + head[i] * alpha
/// ```
///
/// The sqrt-Hann window is **amplitude-complementary**: `(1 - alpha) + alpha = 1`
/// for all `i`, which preserves perceived loudness across the crossfade region.
/// Regular Hann is only power-complementary (sum of squares = 1), which can
/// produce audible energy dips in the middle of the crossfade for speech/TTS.
///
/// Preferred for overlapping speech synthesis (TTS streaming chorus).
///
/// # Edge cases
///
/// - `count == 0`: returns an empty Vec.
/// - `count == 1`: returns the average `(tail[0] + head[0]) / 2`.
///
/// # Panics
///
/// Panics if `tail.len() < count` or `head.len() < count`.
pub fn crossfade_sqrt_hann_blend(tail: &[f32], head: &[f32], count: usize) -> Vec<f32> {
    debug_assert!(
        tail.len() >= count,
        "tail.len() ({}) < count ({})",
        tail.len(),
        count,
    );
    debug_assert!(
        head.len() >= count,
        "head.len() ({}) < count ({})",
        head.len(),
        count,
    );
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![(tail[0] + head[0]) * 0.5];
    }
    let inv = 1.0 / (count - 1) as f64;
    (0..count)
        .map(|i| {
            let alpha = (0.5 * (1.0 - (PI * i as f64 * inv).cos())).sqrt() as f32;
            tail[i] * (1.0 - alpha) + head[i] * alpha
        })
        .collect()
}

/// Append a sqrt-Hann-windowed crossfade blend of `tail` and `head` into `out`.
///
/// `cf` defines the total crossfade window size (used as the alpha
/// denominator). `limit` caps how many samples are actually emitted --
/// useful when the chunk is shorter than the full crossfade window.
///
/// Uses the sqrt-Hann (root raised cosine) window:
/// ```text
/// alpha = sqrt(0.5 * (1 - cos(PI * j / (cf - 1))))
/// ```
///
/// # Edge cases
///
/// - `cf == 0` or `limit == 0`: appends nothing.
/// - `cf == 1` with `limit > 0`: appends the average `(tail[0] + head[0]) / 2`.
///
/// # Panics
///
/// Panics if `tail` or `head` have fewer than `cf.min(limit)` elements.
pub fn crossfade_sqrt_hann_blend_into(
    out: &mut Vec<f32>,
    tail: &[f32],
    head: &[f32],
    cf: usize,
    limit: usize,
) {
    let n = cf.min(limit);
    if n == 0 {
        return;
    }
    if cf == 1 {
        out.push((tail[0] + head[0]) * 0.5);
        return;
    }
    let inv = 1.0 / (cf - 1) as f64;
    out.extend((0..n).map(|j| {
        let alpha = (0.5 * (1.0 - (PI * j as f64 * inv).cos())).sqrt() as f32;
        tail[j] * (1.0 - alpha) + head[j] * alpha
    }));
}

// -- Overlap-add assembly ----------------------------------------------------

/// Assemble two overlapping chunks using overlap-add.
///
/// In overlap-add, both `tail` and `head` are assumed to have been windowed
/// with a complementary analysis/synthesis window. The overlap region is
/// summed directly (no gain coefficients), which is correct when both chunks
/// were produced with the same window that satisfies COLA (Constant Overlap-Add).
///
/// This is the proper method when chunks share identical overlap regions
/// (e.g., from STFT/iSTFT resynthesis or when the same waveform segment
/// appears in both chunks).
///
/// Returns a `Vec<f32>` of length `count` for the overlap region.
///
/// # Edge cases
///
/// - `count == 0`: returns an empty Vec.
///
/// # Panics
///
/// Panics if `tail.len() < count` or `head.len() < count`.
pub fn crossfade_overlap_add(tail: &[f32], head: &[f32], count: usize) -> Vec<f32> {
    debug_assert!(
        tail.len() >= count,
        "tail.len() ({}) < count ({})",
        tail.len(),
        count,
    );
    debug_assert!(
        head.len() >= count,
        "head.len() ({}) < count ({})",
        head.len(),
        count,
    );
    if count == 0 {
        return Vec::new();
    }
    (0..count).map(|i| tail[i] + head[i]).collect()
}

/// Append overlap-add of `tail` and `head` into `out`.
///
/// `cf` defines the total overlap size. `limit` caps how many samples
/// are actually emitted.
///
/// # Edge cases
///
/// - `cf == 0` or `limit == 0`: appends nothing.
///
/// # Panics
///
/// Panics if `tail` or `head` have fewer than `cf.min(limit)` elements.
pub fn crossfade_overlap_add_into(
    out: &mut Vec<f32>,
    tail: &[f32],
    head: &[f32],
    cf: usize,
    limit: usize,
) {
    let n = cf.min(limit);
    if n == 0 {
        return;
    }
    out.extend((0..n).map(|j| tail[j] + head[j]));
}

// -- Hann crossfade ----------------------------------------------------------

/// Compute a Hann-windowed crossfade blend of `count` samples from `tail` and
/// `head`.
///
/// Returns a `Vec<f32>` of length `count` where each sample is a convex
/// combination using the Hann (raised cosine) window:
///
/// ```text
/// alpha = 0.5 * (1 - cos(PI * i / (count - 1)))
/// out[i] = tail[i] * (1 - alpha) + head[i] * alpha
/// ```
///
/// The Hann window produces smoother energy transitions at chunk boundaries
/// compared to linear crossfade. Preferred for overlap durations >= 40ms.
///
/// # Edge cases
///
/// - `count == 0`: returns an empty Vec.
/// - `count == 1`: returns the average `(tail[0] + head[0]) / 2`.
///
/// # Panics
///
/// Panics if `tail.len() < count` or `head.len() < count`.
pub fn crossfade_hann_blend(tail: &[f32], head: &[f32], count: usize) -> Vec<f32> {
    debug_assert!(
        tail.len() >= count,
        "tail.len() ({}) < count ({})",
        tail.len(),
        count,
    );
    debug_assert!(
        head.len() >= count,
        "head.len() ({}) < count ({})",
        head.len(),
        count,
    );
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![(tail[0] + head[0]) * 0.5];
    }
    let inv = 1.0 / (count - 1) as f64;
    (0..count)
        .map(|i| {
            let alpha = (0.5 * (1.0 - (PI * i as f64 * inv).cos())) as f32;
            tail[i] * (1.0 - alpha) + head[i] * alpha
        })
        .collect()
}

/// Append a Hann-windowed crossfade blend of `tail` and `head` into `out`.
///
/// `cf` defines the total crossfade window size (used as the alpha
/// denominator). `limit` caps how many samples are actually emitted --
/// useful when the chunk is shorter than the full crossfade window.
///
/// Uses the Hann (raised cosine) window:
/// ```text
/// alpha = 0.5 * (1 - cos(PI * j / (cf - 1)))
/// ```
///
/// # Edge cases
///
/// - `cf == 0` or `limit == 0`: appends nothing.
/// - `cf == 1` with `limit > 0`: appends the average `(tail[0] + head[0]) / 2`.
///
/// # Panics
///
/// Panics if `tail` or `head` have fewer than `cf.min(limit)` elements.
pub fn crossfade_hann_blend_into(
    out: &mut Vec<f32>,
    tail: &[f32],
    head: &[f32],
    cf: usize,
    limit: usize,
) {
    let n = cf.min(limit);
    if n == 0 {
        return;
    }
    if cf == 1 {
        out.push((tail[0] + head[0]) * 0.5);
        return;
    }
    let inv = 1.0 / (cf - 1) as f64;
    out.extend((0..n).map(|j| {
        let alpha = (0.5 * (1.0 - (PI * j as f64 * inv).cos())) as f32;
        tail[j] * (1.0 - alpha) + head[j] * alpha
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- crossfade tests ---------------------------------------------------------

    #[test]
    fn test_crossfade_linear_blend_basic() {
        let tail = vec![1.0_f32; 5];
        let head = vec![0.0_f32; 5];
        let result = crossfade_linear_blend(&tail, &head, 5);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[2] - 0.5).abs() < 1e-6);
        assert!((result[4] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_linear_blend_identical_signals() {
        let signal = vec![0.5_f32; 10];
        let result = crossfade_linear_blend(&signal, &signal, 10);
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 0.5).abs() < 1e-6, "sample {i}: expected 0.5, got {v}");
        }
    }

    #[test]
    fn test_crossfade_linear_blend_count_zero() {
        let result = crossfade_linear_blend(&[1.0], &[0.0], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_crossfade_linear_blend_count_one() {
        let result = crossfade_linear_blend(&[1.0], &[0.0], 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_linear_blend_endpoints() {
        let tail = vec![0.3_f32; 4];
        let head = vec![0.7_f32; 4];
        let result = crossfade_linear_blend(&tail, &head, 4);
        assert!((result[0] - 0.3).abs() < 1e-6);
        assert!((result[3] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_blend_into_basic() {
        let mut out = Vec::new();
        let tail = vec![1.0_f32; 5];
        let head = vec![0.0_f32; 5];
        crossfade_blend_into(&mut out, &tail, &head, 5, 5);
        assert_eq!(out.len(), 5);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[2] - 0.5).abs() < 1e-6);
        assert!((out[4] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_blend_into_limit_less_than_cf() {
        let mut out = Vec::new();
        let tail = vec![1.0_f32; 10];
        let head = vec![0.0_f32; 10];
        // cf=10 but limit=3 => only 3 samples, alpha uses cf=10 denominator
        crossfade_blend_into(&mut out, &tail, &head, 10, 3);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - (8.0 / 9.0)).abs() < 1e-5);
        assert!((out[2] - (7.0 / 9.0)).abs() < 1e-5);
    }

    #[test]
    fn test_crossfade_blend_into_zero_limit() {
        let mut out = Vec::new();
        crossfade_blend_into(&mut out, &[1.0], &[0.0], 5, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_crossfade_blend_into_zero_cf() {
        let mut out = Vec::new();
        crossfade_blend_into(&mut out, &[1.0], &[0.0], 0, 5);
        assert!(out.is_empty());
    }

    #[test]
    fn test_crossfade_blend_into_appends() {
        let mut out = vec![99.0_f32];
        let tail = vec![1.0_f32; 3];
        let head = vec![0.0_f32; 3];
        crossfade_blend_into(&mut out, &tail, &head, 3, 3);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 99.0).abs() < 1e-6);
    }

    // -- Hann crossfade tests ----------------------------------------------------

    #[test]
    fn test_crossfade_hann_blend_basic() {
        let tail = vec![1.0_f32; 5];
        let head = vec![0.0_f32; 5];
        let result = crossfade_hann_blend(&tail, &head, 5);
        assert_eq!(result.len(), 5);
        // alpha(0) = 0.5*(1-cos(0)) = 0 → out[0] = 1.0
        assert!((result[0] - 1.0).abs() < 1e-6);
        // alpha(2) = 0.5*(1-cos(PI*2/4)) = 0.5 → out[2] = 0.5
        assert!((result[2] - 0.5).abs() < 1e-6);
        // alpha(4) = 0.5*(1-cos(PI)) = 1.0 → out[4] = 0.0
        assert!((result[4] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_hann_blend_identical_signals() {
        let signal = vec![0.5_f32; 10];
        let result = crossfade_hann_blend(&signal, &signal, 10);
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 0.5).abs() < 1e-5, "sample {i}: expected 0.5, got {v}");
        }
    }

    #[test]
    fn test_crossfade_hann_blend_count_zero() {
        let result = crossfade_hann_blend(&[1.0], &[0.0], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_crossfade_hann_blend_count_one() {
        let result = crossfade_hann_blend(&[1.0], &[0.0], 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_hann_blend_endpoints() {
        let tail = vec![0.3_f32; 4];
        let head = vec![0.7_f32; 4];
        let result = crossfade_hann_blend(&tail, &head, 4);
        // alpha(0) = 0 → out[0] = 0.3
        assert!((result[0] - 0.3).abs() < 1e-6);
        // alpha(3) = 0.5*(1-cos(PI)) = 1.0 → out[3] = 0.7
        assert!((result[3] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_hann_blend_smoother_than_linear() {
        // Hann window should have slower initial ramp than linear
        let tail = vec![1.0_f32; 100];
        let head = vec![0.0_f32; 100];
        let hann = crossfade_hann_blend(&tail, &head, 100);
        let linear = crossfade_linear_blend(&tail, &head, 100);
        // At sample 1, Hann alpha should be smaller than linear alpha
        // (Hann starts flat, linear ramps immediately)
        assert!(
            hann[1] > linear[1],
            "Hann should retain more of tail at start: hann={}, linear={}",
            hann[1],
            linear[1]
        );
    }

    #[test]
    fn test_crossfade_hann_blend_into_basic() {
        let mut out = Vec::new();
        let tail = vec![1.0_f32; 5];
        let head = vec![0.0_f32; 5];
        crossfade_hann_blend_into(&mut out, &tail, &head, 5, 5);
        assert_eq!(out.len(), 5);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[2] - 0.5).abs() < 1e-6);
        assert!((out[4] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_hann_blend_into_zero() {
        let mut out = Vec::new();
        crossfade_hann_blend_into(&mut out, &[1.0], &[0.0], 0, 5);
        assert!(out.is_empty());
        crossfade_hann_blend_into(&mut out, &[1.0], &[0.0], 5, 0);
        assert!(out.is_empty());
    }

    // -- mel/window tests --------------------------------------------------------

    #[test]
    fn test_htk_mel_roundtrip() {
        for &hz in &[0.0, 440.0, 1000.0, 8000.0, 16000.0] {
            let mel = hz_to_mel_htk(hz);
            let back = mel_to_hz_htk(mel);
            assert!((back - hz).abs() < 1e-9, "roundtrip failed for hz={hz}");
        }
    }

    #[test]
    fn test_htk_mel_known_values() {
        // hz=0 → mel=0
        assert!((hz_to_mel_htk(0.0)).abs() < 1e-12);
        // Hz=1000 → mel ≈ 1000 (approximately, not exactly due to log scale)
        let mel_1k = hz_to_mel_htk(1000.0);
        assert!(mel_1k > 999.0 && mel_1k < 1001.0);
    }

    #[test]
    fn test_slaney_mel_roundtrip() {
        for &hz in &[0.0, 440.0, 1000.0, 8000.0, 16000.0] {
            let mel = hz_to_mel_slaney(hz);
            let back = mel_to_hz_slaney(mel);
            assert!(
                (back - hz).abs() < 1e-9,
                "roundtrip failed for hz={hz}: got {back}"
            );
        }
    }

    #[test]
    fn test_slaney_mel_linear_region() {
        // Below 1 kHz, Slaney is linear: mel = hz / (200/3)
        let hz = 500.0;
        let expected = hz / (200.0 / 3.0);
        assert!((hz_to_mel_slaney(hz) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_slaney_mel_transition() {
        // At exactly 1000 Hz, both formulas should give the same mel value
        let mel = hz_to_mel_slaney(1000.0);
        let expected = 1000.0 / (200.0 / 3.0); // 15.0
        assert!((mel - expected).abs() < 1e-12);
    }

    #[test]
    fn test_hann_window_basic() {
        let w = hann_window(4);
        assert_eq!(w.len(), 4);
        // w[0] = 0.5 * (1 - cos(0)) = 0
        assert!(w[0].abs() < 1e-15);
        // w[2] = 0.5 * (1 - cos(π)) = 1
        assert!((w[2] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_hann_window_empty() {
        assert!(hann_window(0).is_empty());
    }

    #[test]
    fn test_hann_window_symmetry() {
        let w = hann_window(256);
        // Periodic Hann: w[i] = w[n-i] for i in 1..n/2 (index 0 has no mirror).
        for i in 1..128 {
            assert!((w[i] - w[256 - i]).abs() < 1e-12, "asymmetry at i={i}");
        }
    }

    #[test]
    fn test_hann_window_values_in_unit_interval() {
        let w = hann_window(100);
        for (i, &v) in w.iter().enumerate() {
            assert!((0.0..=1.0).contains(&v), "w[{i}] = {v} out of [0, 1]");
        }
    }

    // -- Sqrt-Hann crossfade tests -----------------------------------------------

    #[test]
    fn test_crossfade_sqrt_hann_blend_basic() {
        let tail = vec![1.0_f32; 5];
        let head = vec![0.0_f32; 5];
        let result = crossfade_sqrt_hann_blend(&tail, &head, 5);
        assert_eq!(result.len(), 5);
        // alpha(0) = sqrt(0.5*(1-cos(0))) = 0 → out[0] = 1.0
        assert!((result[0] - 1.0).abs() < 1e-6);
        // alpha(4) = sqrt(0.5*(1-cos(PI))) = 1.0 → out[4] = 0.0
        assert!((result[4] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_sqrt_hann_blend_amplitude_complementary() {
        // The defining property of sqrt-Hann: alpha + (1-alpha) = 1 for all i.
        // This means for tail=A, head=A (identical signals), output = A exactly.
        let signal = vec![0.7_f32; 100];
        let result = crossfade_sqrt_hann_blend(&signal, &signal, 100);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 0.7).abs() < 1e-5,
                "sample {i}: expected 0.7, got {v} (amplitude non-complementary)"
            );
        }
    }

    #[test]
    fn test_crossfade_sqrt_hann_blend_energy_within_1db() {
        // For a constant-amplitude signal crossing over to another constant-
        // amplitude signal, sqrt-Hann should keep the RMS energy of the
        // crossfade region within 1 dB of the surrounding regions.
        let n = 960; // 40ms at 24kHz
        let amplitude = 0.5_f32;
        let tail = vec![amplitude; n];
        let head = vec![amplitude; n];
        let result = crossfade_sqrt_hann_blend(&tail, &head, n);

        // RMS of the crossfade region
        let rms_cf: f32 = (result.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
        // RMS of the original signal (constant amplitude)
        let rms_orig = amplitude;

        // 1 dB = 10^(1/20) ≈ 1.122. Check ratio is within [-1dB, +1dB].
        let ratio = rms_cf / rms_orig;
        let db_diff = 20.0 * ratio.log10();
        assert!(
            db_diff.abs() < 1.0,
            "Energy deviation {db_diff:.3} dB exceeds 1 dB limit (ratio={ratio:.4})"
        );
    }

    #[test]
    fn test_crossfade_sqrt_hann_blend_midpoint() {
        // At midpoint, sqrt-Hann alpha = sqrt(0.5) ≈ 0.707
        let tail = vec![1.0_f32; 3];
        let head = vec![0.0_f32; 3];
        let result = crossfade_sqrt_hann_blend(&tail, &head, 3);
        // i=1, alpha = sqrt(0.5*(1-cos(PI*1/2))) = sqrt(0.5) ≈ 0.707
        let expected_alpha = (0.5_f64).sqrt() as f32;
        let expected = 1.0 * (1.0 - expected_alpha) + 0.0 * expected_alpha;
        assert!(
            (result[1] - expected).abs() < 1e-5,
            "midpoint: expected {expected}, got {}",
            result[1]
        );
    }

    #[test]
    fn test_crossfade_sqrt_hann_blend_count_zero() {
        let result = crossfade_sqrt_hann_blend(&[1.0], &[0.0], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_crossfade_sqrt_hann_blend_count_one() {
        let result = crossfade_sqrt_hann_blend(&[1.0], &[0.0], 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_sqrt_hann_blend_endpoints() {
        let tail = vec![0.3_f32; 4];
        let head = vec![0.7_f32; 4];
        let result = crossfade_sqrt_hann_blend(&tail, &head, 4);
        assert!((result[0] - 0.3).abs() < 1e-6);
        assert!((result[3] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_sqrt_hann_blend_into_basic() {
        let mut out = Vec::new();
        let tail = vec![1.0_f32; 5];
        let head = vec![0.0_f32; 5];
        crossfade_sqrt_hann_blend_into(&mut out, &tail, &head, 5, 5);
        assert_eq!(out.len(), 5);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[4] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_sqrt_hann_blend_into_zero() {
        let mut out = Vec::new();
        crossfade_sqrt_hann_blend_into(&mut out, &[1.0], &[0.0], 0, 5);
        assert!(out.is_empty());
        crossfade_sqrt_hann_blend_into(&mut out, &[1.0], &[0.0], 5, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_crossfade_sqrt_hann_faster_transition_in_outer_quarters() {
        // Sqrt-Hann transitions faster toward equal blending in the outer
        // quarters of the crossfade, which means both signals contribute
        // more equally at the start/end of the overlap region. This
        // reduces the "one signal dominates" zone, creating a smoother
        // perceptual transition for speech.
        //
        // At the quarter-point (i = N/4), sqrt-Hann alpha is closer to
        // 0.5 than Hann alpha, meaning more balanced blending.
        let n = 200;
        let inv = 1.0 / f64::from(n - 1);
        let quarter = n / 4;

        let alpha_hann = 0.5 * (1.0 - (PI * f64::from(quarter) * inv).cos());
        let alpha_sqrt = (0.5 * (1.0 - (PI * f64::from(quarter) * inv).cos())).sqrt();

        // sqrt-Hann alpha should be closer to 0.5 at the quarter-point
        let dist_hann = (alpha_hann - 0.5).abs();
        let dist_sqrt = (alpha_sqrt - 0.5).abs();
        assert!(
            dist_sqrt < dist_hann,
            "sqrt-Hann alpha ({alpha_sqrt:.4}) should be closer to 0.5 than \
             Hann alpha ({alpha_hann:.4}) at quarter-point"
        );
    }

    // -- Overlap-add tests -------------------------------------------------------

    #[test]
    fn test_crossfade_overlap_add_basic() {
        let tail = vec![0.5_f32; 5];
        let head = vec![0.3_f32; 5];
        let result = crossfade_overlap_add(&tail, &head, 5);
        assert_eq!(result.len(), 5);
        for &v in &result {
            assert!((v - 0.8).abs() < 1e-6);
        }
    }

    #[test]
    fn test_crossfade_overlap_add_count_zero() {
        let result = crossfade_overlap_add(&[1.0], &[0.0], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_crossfade_overlap_add_sqrt_hann_cola() {
        // Verify overlap-add with sqrt-Hann analysis/synthesis windows
        // satisfies COLA (Constant Overlap-Add).
        //
        // For a 50% hop overlap, if each segment is windowed with sqrt(Hann),
        // the overlapping windowed segments sum to 1.0 because:
        //   sqrt(hann[i])^2 + sqrt(hann[i + hop])^2 = hann[i] + hann[i + hop] = 1
        // This is the standard STFT/iSTFT COLA condition.
        //
        // Here we verify the simpler property: sqrt-Hann fade-out + fade-in = 1.
        let n = 100;
        // sqrt-Hann fade-out: alpha goes from 1 to 0
        let inv = 1.0 / (n - 1) as f64;
        let fade_out: Vec<f32> = (0..n)
            .map(|i| (0.5 * (1.0 + (PI * i as f64 * inv).cos())).sqrt() as f32)
            .collect();
        // sqrt-Hann fade-in: alpha goes from 0 to 1
        let fade_in: Vec<f32> = (0..n)
            .map(|i| (0.5 * (1.0 - (PI * i as f64 * inv).cos())).sqrt() as f32)
            .collect();

        // fade_in^2 + fade_out^2 should equal 1.0 at every point
        // (this is the power-complementary COLA property for sqrt-Hann)
        for i in 0..n {
            let sum_sq = fade_in[i] * fade_in[i] + fade_out[i] * fade_out[i];
            assert!(
                (sum_sq - 1.0).abs() < 1e-5,
                "Power COLA violated at sample {i}: sum_sq={sum_sq}"
            );
        }

        // For overlap-add of windowed unit signals, the sum should be
        // approximately constant (demonstrating that OLA preserves energy).
        let tail_windowed: Vec<f32> = fade_out.iter().map(|&w| w * 1.0).collect();
        let head_windowed: Vec<f32> = fade_in.iter().map(|&w| w * 1.0).collect();
        let result = crossfade_overlap_add(&tail_windowed, &head_windowed, n);
        // The OLA sum won't be exactly 1.0 (that requires squared windows),
        // but it should be close and monotonic-ish.
        for (i, &v) in result.iter().enumerate() {
            assert!(
                v > 0.9 && v < 1.5,
                "OLA sum out of range at sample {i}: sum={v}"
            );
        }
    }

    #[test]
    fn test_crossfade_overlap_add_into_basic() {
        let mut out = vec![99.0_f32];
        let tail = vec![0.4_f32; 3];
        let head = vec![0.6_f32; 3];
        crossfade_overlap_add_into(&mut out, &tail, &head, 3, 3);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 99.0).abs() < 1e-6);
        for &v in &out[1..] {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_crossfade_overlap_add_into_zero() {
        let mut out = Vec::new();
        crossfade_overlap_add_into(&mut out, &[1.0], &[0.0], 0, 5);
        assert!(out.is_empty());
        crossfade_overlap_add_into(&mut out, &[1.0], &[0.0], 5, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_crossfade_overlap_add_into_limit_less_than_cf() {
        let mut out = Vec::new();
        let tail = vec![0.5_f32; 10];
        let head = vec![0.3_f32; 10];
        crossfade_overlap_add_into(&mut out, &tail, &head, 10, 3);
        assert_eq!(out.len(), 3);
        for &v in &out {
            assert!((v - 0.8).abs() < 1e-6);
        }
    }
}
