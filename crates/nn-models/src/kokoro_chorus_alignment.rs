// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Adaptive voice alignment for multi-voice chorus temporal coherence.
//!
//! When multiple voices synthesize the same text, their phoneme durations may
//! differ slightly, causing temporal misalignment that sounds like sloppy timing
//! rather than natural variation. Real choirs have tight timing on stressed
//! syllables while allowing looser timing on unstressed ones.
//!
//! This module uses windowed cross-correlation to find the optimal per-window
//! shift for each voice relative to a reference (voice 0), then applies
//! fractional sample shifts with linear interpolation and crossfade blending.
//!
//! # Design
//!
//! 1. Voice 0 is the reference signal.
//! 2. For each other voice, slide a window across both signals.
//! 3. Within each window, compute normalized cross-correlation at lags
//!    `[-max_shift_samples, +max_shift_samples]`.
//! 4. The lag with the highest correlation is the optimal shift.
//! 5. Apply the shift using linear interpolation (supports fractional samples).
//! 6. Blend between original and shifted signal based on `tightness`.
//! 7. Crossfade at window boundaries to prevent clicks.
//!
//! Part of #4264.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for adaptive voice alignment.
///
/// Controls the cross-correlation window, maximum correction magnitude, and
/// how aggressively the correction is applied.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AlignmentConfig {
    /// Maximum alignment correction in samples.
    ///
    /// The cross-correlation search range is `[-max_shift_samples, +max_shift_samples]`.
    /// At 24 kHz, 480 samples = 20 ms, which is enough to correct typical
    /// phoneme-level timing drift without introducing artifacts.
    ///
    /// Range: [1, 4800]. Default: `480` (20 ms at 24 kHz).
    pub max_shift_samples: usize,

    /// Window size in samples for cross-correlation computation.
    ///
    /// Larger windows give more stable correlation estimates but reduce the
    /// temporal resolution of alignment correction. Must be at least
    /// `2 * max_shift_samples + 1` to allow the full lag search range.
    ///
    /// Range: [64, 8192]. Default: `1024` (~42 ms at 24 kHz).
    pub correlation_window: usize,

    /// Alignment tightness: 0.0 = no correction, 1.0 = full correction.
    ///
    /// At intermediate values the output is a blend between the original
    /// signal and the fully aligned signal. This preserves some natural
    /// timing variation while tightening the overall feel.
    ///
    /// Range: [0.0, 1.0]. Default: `0.6`.
    pub tightness: f32,

    /// Crossfade length in samples at window boundaries.
    ///
    /// Prevents clicks when the shift amount changes between adjacent windows.
    /// The crossfade uses a raised-cosine (Hann-style) taper.
    ///
    /// Range: [1, 1024]. Default: `64`.
    pub fade_samples: usize,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            max_shift_samples: 480,
            correlation_window: 1024,
            tightness: 0.6,
            fade_samples: 64,
        }
    }
}

impl AlignmentConfig {
    /// Create a config with the given tightness and default window/shift.
    pub fn new(tightness: f32) -> Result<Self, KokoroError> {
        let config = Self {
            tightness,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    /// Create a disabled config (passthrough, tightness = 0).
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            tightness: 0.0,
            ..Self::default()
        }
    }

    /// Set max shift samples.
    #[must_use]
    pub fn with_max_shift(mut self, max_shift_samples: usize) -> Self {
        self.max_shift_samples = max_shift_samples;
        self
    }

    /// Set correlation window size.
    #[must_use]
    pub fn with_correlation_window(mut self, correlation_window: usize) -> Self {
        self.correlation_window = correlation_window;
        self
    }

    /// Set crossfade samples.
    #[must_use]
    pub fn with_fade_samples(mut self, fade_samples: usize) -> Self {
        self.fade_samples = fade_samples;
        self
    }

    /// Validate all fields. Returns `Err` on out-of-range or non-finite values.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.max_shift_samples == 0 || self.max_shift_samples > 4800 {
            return Err(KokoroError::InvalidConfig {
                field: "max_shift_samples",
                reason: format!("must be in [1, 4800], got {}", self.max_shift_samples),
            });
        }
        if self.correlation_window < 64 || self.correlation_window > 8192 {
            return Err(KokoroError::InvalidConfig {
                field: "correlation_window",
                reason: format!("must be in [64, 8192], got {}", self.correlation_window),
            });
        }
        let min_window = 2 * self.max_shift_samples + 1;
        if self.correlation_window < min_window {
            return Err(KokoroError::InvalidConfig {
                field: "correlation_window",
                reason: format!(
                    "must be >= 2 * max_shift_samples + 1 = {}, got {}",
                    min_window, self.correlation_window
                ),
            });
        }
        if !self.tightness.is_finite() || !(0.0..=1.0).contains(&self.tightness) {
            return Err(KokoroError::InvalidConfig {
                field: "tightness",
                reason: format!("must be finite and in [0.0, 1.0], got {}", self.tightness),
            });
        }
        if self.fade_samples == 0 || self.fade_samples > 1024 {
            return Err(KokoroError::InvalidConfig {
                field: "fade_samples",
                reason: format!("must be in [1, 1024], got {}", self.fade_samples),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Cross-correlation
// ---------------------------------------------------------------------------

/// Result of a cross-correlation search within one window.
#[derive(Debug, Clone, Copy)]
pub struct CorrelationResult {
    /// Optimal lag in samples (negative = reference leads, positive = reference trails).
    pub lag: i32,
    /// Normalized cross-correlation coefficient at the optimal lag, in [-1.0, 1.0].
    pub coefficient: f32,
}

/// Compute normalized cross-correlation between two signal windows and find
/// the lag (within `[-max_lag, +max_lag]`) that maximizes correlation.
///
/// Both `reference` and `target` must have the same length and at least
/// `2 * max_lag + 1` samples. Returns the best lag and its correlation
/// coefficient.
///
/// The correlation is normalized: 1.0 = identical, -1.0 = inverted, 0.0 = uncorrelated.
pub fn cross_correlate(
    reference: &[f32],
    target: &[f32],
    max_lag: usize,
) -> Result<CorrelationResult, KokoroError> {
    if reference.len() != target.len() {
        return Err(KokoroError::InvalidInput(format!(
            "cross_correlate: reference len {} != target len {}",
            reference.len(),
            target.len()
        )));
    }
    let n = reference.len();
    let min_len = 2 * max_lag + 1;
    if n < min_len {
        return Err(KokoroError::InvalidInput(format!(
            "cross_correlate: window len {n} < 2 * max_lag + 1 = {min_len}"
        )));
    }
    if max_lag == 0 {
        return Ok(CorrelationResult {
            lag: 0,
            coefficient: 1.0,
        });
    }

    // Precompute reference energy (denominator component).
    let ref_energy: f64 = reference.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    if !ref_energy.is_finite() || ref_energy < 1e-30 {
        // Reference is silence or non-finite; no meaningful correlation.
        return Ok(CorrelationResult {
            lag: 0,
            coefficient: 0.0,
        });
    }

    let mut best_lag: i32 = 0;
    let mut best_corr: f64 = f64::NEG_INFINITY;

    let max_lag_i = max_lag as i32;
    for lag in -max_lag_i..=max_lag_i {
        // Compute cross-correlation at this lag.
        // When lag > 0, target is shifted right (we look at earlier samples of target).
        // When lag < 0, target is shifted left (we look at later samples of target).
        let mut cross: f64 = 0.0;
        let mut tgt_energy: f64 = 0.0;
        let mut count: usize = 0;

        for i in 0..n {
            let j = i as i32 + lag;
            if j < 0 || j >= n as i32 {
                continue;
            }
            let r = f64::from(reference[i]);
            let t = f64::from(target[j as usize]);
            if !r.is_finite() || !t.is_finite() {
                continue;
            }
            cross += r * t;
            tgt_energy += t * t;
            count += 1;
        }

        if count == 0 || !tgt_energy.is_finite() || tgt_energy < 1e-30 {
            continue;
        }
        if !cross.is_finite() {
            continue;
        }

        let denom = (ref_energy * tgt_energy).sqrt();
        if !denom.is_finite() || denom < 1e-30 {
            continue;
        }

        let corr = cross / denom;
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    // Clamp coefficient to valid range (floating-point can exceed slightly).
    let coefficient = if !best_corr.is_finite() {
        0.0_f32
    } else {
        (best_corr as f32).clamp(-1.0, 1.0)
    };

    Ok(CorrelationResult {
        lag: best_lag,
        coefficient,
    })
}

// ---------------------------------------------------------------------------
// Shift application
// ---------------------------------------------------------------------------

/// Shift audio by `shift` samples (can be fractional) using linear interpolation.
///
/// Positive shift delays the signal (moves it later in time); negative shift
/// advances it. Samples shifted beyond the buffer boundaries are zero-filled.
///
/// A crossfade of `fade_samples` is applied at the beginning and end to
/// avoid clicks when the shift changes between adjacent windows.
pub fn apply_shift(
    audio: &[f32],
    shift: f32,
    fade_samples: usize,
) -> Result<Vec<f32>, KokoroError> {
    if !shift.is_finite() {
        return Err(KokoroError::InvalidInput(format!(
            "apply_shift: shift must be finite, got {shift}"
        )));
    }
    let n = audio.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut out = vec![0.0f32; n];
    let shift_int = shift.floor() as i64;
    let frac = shift - shift.floor();

    for i in 0..n {
        // Source index in the original signal (floating-point).
        let src = i as i64 - shift_int;
        let src_lo = src as usize;
        let src_hi = src_lo.wrapping_add(1);

        // Bounds check for the integer part.
        if src < 0 || src >= n as i64 {
            out[i] = 0.0;
            continue;
        }

        let val_lo = audio[src_lo];
        let val_hi = if src_hi < n { audio[src_hi] } else { 0.0 };

        if !val_lo.is_finite() {
            out[i] = 0.0;
            continue;
        }
        if !val_hi.is_finite() {
            out[i] = val_lo;
            continue;
        }

        // Linear interpolation for fractional shift.
        let interpolated = val_lo + frac * (val_hi - val_lo);
        out[i] = if interpolated.is_finite() {
            interpolated
        } else {
            0.0
        };
    }

    // Apply raised-cosine fade at boundaries.
    let fade = fade_samples.min(n / 2).max(1);
    for i in 0..fade {
        let t = i as f32 / fade as f32;
        // Raised cosine: 0.5 * (1 - cos(pi * t))
        let gain = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
        let gain = if gain.is_finite() { gain } else { 0.0 };
        out[i] *= gain;
        if n - 1 - i < n {
            out[n - 1 - i] *= gain;
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Main alignment function
// ---------------------------------------------------------------------------

/// Align multiple voices to a common temporal reference.
///
/// Takes a set of same-text PCM signals (all at the same sample rate) and
/// aligns voices 1..N to voice 0 using windowed cross-correlation.
///
/// # Arguments
///
/// * `voices` - Mutable vector of per-voice PCM. Voice 0 is the reference
///   and is not modified. All voices should be the same length (shorter
///   voices are zero-padded internally; longer voices are truncated to the
///   reference length).
/// * `config` - Alignment parameters (window size, max shift, tightness).
///
/// # Returns
///
/// The aligned voices (same structure as input, voice 0 unchanged).
pub fn align_voices(
    voices: &[Vec<f32>],
    config: &AlignmentConfig,
) -> Result<Vec<Vec<f32>>, KokoroError> {
    config.validate()?;

    if voices.is_empty() {
        return Ok(Vec::new());
    }
    if voices.len() == 1 {
        return Ok(voices.to_vec());
    }

    let reference = &voices[0];
    let ref_len = reference.len();
    if ref_len == 0 {
        return Ok(voices.to_vec());
    }

    // Tightness == 0 means no correction.
    if config.tightness <= 0.0 {
        return Ok(voices.to_vec());
    }

    let window = config.correlation_window;
    let max_shift = config.max_shift_samples;
    let fade = config.fade_samples;
    let tightness = config.tightness;

    let mut result: Vec<Vec<f32>> = Vec::with_capacity(voices.len());
    // Voice 0 (reference) passes through unchanged.
    result.push(reference.clone());

    for voice_idx in 1..voices.len() {
        let target = &voices[voice_idx];
        // Normalize to reference length.
        let target_padded = normalize_length(target, ref_len);
        let aligned = align_single_voice(
            reference,
            &target_padded,
            window,
            max_shift,
            fade,
            tightness,
        )?;
        result.push(aligned);
    }

    Ok(result)
}

/// Align a single voice to the reference using windowed cross-correlation.
fn align_single_voice(
    reference: &[f32],
    target: &[f32],
    window: usize,
    max_shift: usize,
    fade: usize,
    tightness: f32,
) -> Result<Vec<f32>, KokoroError> {
    let n = reference.len();
    debug_assert_eq!(target.len(), n);

    // If the signal is shorter than one window, do a single global alignment.
    if n <= window {
        let corr = cross_correlate(
            &pad_to_min(reference, 2 * max_shift + 1),
            &pad_to_min(target, 2 * max_shift + 1),
            max_shift,
        )?;
        let effective_shift = corr.lag as f32 * tightness;
        let shifted = apply_shift(target, effective_shift, fade)?;
        return blend_signals(target, &shifted, tightness);
    }

    // Sliding window alignment.
    let hop = window / 2; // 50% overlap for smooth transitions.
    let mut aligned = vec![0.0f32; n];
    let mut weight_acc = vec![0.0f32; n];

    let mut pos = 0usize;
    while pos < n {
        let end = (pos + window).min(n);
        let ref_win = &reference[pos..end];
        let tgt_win = &target[pos..end];
        let win_len = end - pos;

        // Compute optimal shift for this window.
        let shift = if win_len > 2 * max_shift {
            let corr = cross_correlate(ref_win, tgt_win, max_shift)?;
            corr.lag as f32 * tightness
        } else {
            // Window too small for full search; use zero shift.
            0.0
        };

        // Apply shift to this window's target segment.
        let shifted_win = apply_shift(tgt_win, shift, fade.min(win_len / 2).max(1))?;

        // Blend original and shifted.
        let blended_win = blend_signals(tgt_win, &shifted_win, tightness)?;

        // Overlap-add with Hann window weighting.
        for i in 0..win_len {
            let t = i as f32 / win_len.max(1) as f32;
            let hann = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * t).cos());
            let hann = if hann.is_finite() { hann } else { 0.0 };
            let val = blended_win[i];
            if val.is_finite() {
                aligned[pos + i] += val * hann;
                weight_acc[pos + i] += hann;
            }
        }

        pos += hop;
    }

    // Normalize by accumulated Hann weights.
    for i in 0..n {
        let w = weight_acc[i];
        if w.is_finite() && w > 1e-10 {
            let val = aligned[i] / w;
            aligned[i] = if val.is_finite() { val } else { 0.0 };
        } else {
            // No window covered this sample; fall back to original.
            aligned[i] = target[i];
        }
    }

    Ok(aligned)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Blend between original and processed signal based on mix factor.
///
/// `mix = 0.0` returns original, `mix = 1.0` returns processed.
fn blend_signals(original: &[f32], processed: &[f32], mix: f32) -> Result<Vec<f32>, KokoroError> {
    if original.len() != processed.len() {
        return Err(KokoroError::InvalidInput(format!(
            "blend_signals: length mismatch {} vs {}",
            original.len(),
            processed.len()
        )));
    }
    let mix = mix.clamp(0.0, 1.0);
    let inv = 1.0 - mix;
    Ok(original
        .iter()
        .zip(processed.iter())
        .map(|(&o, &p)| {
            let val = o * inv + p * mix;
            if val.is_finite() {
                val
            } else {
                0.0
            }
        })
        .collect())
}

/// Normalize a signal to a target length (zero-pad or truncate).
fn normalize_length(signal: &[f32], target_len: usize) -> Vec<f32> {
    if signal.len() >= target_len {
        signal[..target_len].to_vec()
    } else {
        let mut padded = signal.to_vec();
        padded.resize(target_len, 0.0);
        padded
    }
}

/// Zero-pad a signal to a minimum length (used for short-signal fallback).
fn pad_to_min(signal: &[f32], min_len: usize) -> Vec<f32> {
    if signal.len() >= min_len {
        signal.to_vec()
    } else {
        let mut padded = signal.to_vec();
        padded.resize(min_len, 0.0);
        padded
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_config_default_valid() {
        let config = AlignmentConfig::default();
        config.validate().expect("default config should be valid");
    }

    #[test]
    fn test_alignment_config_new() {
        let config = AlignmentConfig::new(0.8).expect("valid tightness");
        assert!((config.tightness - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_alignment_config_invalid_tightness() {
        assert!(AlignmentConfig::new(1.5).is_err());
        assert!(AlignmentConfig::new(-0.1).is_err());
        assert!(AlignmentConfig::new(f32::NAN).is_err());
        assert!(AlignmentConfig::new(f32::INFINITY).is_err());
    }

    #[test]
    fn test_alignment_config_disabled() {
        let config = AlignmentConfig::disabled();
        assert!((config.tightness - 0.0).abs() < 1e-6);
        config.validate().expect("disabled config should be valid");
    }

    #[test]
    fn test_alignment_config_window_too_small() {
        let config = AlignmentConfig {
            max_shift_samples: 480,
            correlation_window: 64, // < 2*480+1 = 961
            tightness: 0.5,
            fade_samples: 32,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cross_correlate_identical_signals() {
        let signal = vec![0.0, 0.1, 0.5, 1.0, 0.5, 0.1, 0.0, -0.3, -0.5, -0.3, 0.0];
        let result = cross_correlate(&signal, &signal, 3).expect("should succeed");
        assert_eq!(result.lag, 0);
        assert!(
            result.coefficient > 0.99,
            "identical signals should have ~1.0 correlation"
        );
    }

    #[test]
    fn test_cross_correlate_shifted_signal() {
        // Create a signal and a version shifted by 2 samples.
        let n = 64;
        let mut reference = vec![0.0f32; n];
        let mut target = vec![0.0f32; n];
        for i in 0..n {
            let t = i as f32 / n as f32;
            let val = (2.0 * std::f32::consts::PI * 3.0 * t).sin();
            reference[i] = val;
        }
        // Shift target by +2 samples.
        target[2..n].copy_from_slice(&reference[..n - 2]);
        let result = cross_correlate(&reference, &target, 5).expect("should succeed");
        // The lag should be approximately +2 (target needs to shift left by 2).
        assert!(
            (result.lag - 2).abs() <= 1,
            "expected lag ~2, got {}",
            result.lag
        );
        assert!(
            result.coefficient > 0.8,
            "coefficient should be high for shifted sine"
        );
    }

    #[test]
    fn test_cross_correlate_silence() {
        let silence = vec![0.0f32; 32];
        let signal = vec![0.5f32; 32];
        let result = cross_correlate(&silence, &signal, 5).expect("should succeed");
        assert_eq!(result.lag, 0);
        assert!(
            (result.coefficient - 0.0).abs() < 1e-6,
            "silence should give 0 correlation"
        );
    }

    #[test]
    fn test_cross_correlate_length_mismatch() {
        let a = vec![1.0; 20];
        let b = vec![1.0; 30];
        assert!(cross_correlate(&a, &b, 3).is_err());
    }

    #[test]
    fn test_apply_shift_zero() {
        let audio = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let shifted = apply_shift(&audio, 0.0, 1).expect("zero shift");
        // With fade applied at boundaries, interior samples should be close.
        assert!(shifted.len() == audio.len());
    }

    #[test]
    fn test_apply_shift_integer() {
        // Shift right by 1 sample.
        let audio = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let shifted = apply_shift(&audio, 1.0, 1).expect("integer shift");
        // The impulse at [2] should move to approximately [3] (modulo fade).
        assert_eq!(shifted.len(), audio.len());
    }

    #[test]
    fn test_apply_shift_nan_rejected() {
        let audio = vec![1.0; 10];
        assert!(apply_shift(&audio, f32::NAN, 2).is_err());
    }

    #[test]
    fn test_apply_shift_empty() {
        let result = apply_shift(&[], 1.0, 2).expect("empty is ok");
        assert!(result.is_empty());
    }

    #[test]
    fn test_align_voices_empty() {
        let config = AlignmentConfig::default();
        let result = align_voices(&[], &config).expect("empty ok");
        assert!(result.is_empty());
    }

    #[test]
    fn test_align_voices_single() {
        let config = AlignmentConfig::default();
        let voices = vec![vec![0.5; 2048]];
        let result = align_voices(&voices, &config).expect("single voice ok");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2048);
    }

    #[test]
    fn test_align_voices_identical() {
        let config = AlignmentConfig::new(0.5).expect("valid");
        let signal: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * 5.0 * i as f32 / 2048.0).sin())
            .collect();
        let voices = vec![signal.clone(), signal];
        let result = align_voices(&voices, &config).expect("identical voices");
        assert_eq!(result.len(), 2);
        // Both voices should be very similar since they were identical.
        let max_diff: f32 = result[0]
            .iter()
            .zip(result[1].iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.5,
            "identical voices should stay similar, max_diff = {max_diff}"
        );
    }

    #[test]
    fn test_align_voices_disabled_passthrough() {
        let config = AlignmentConfig::disabled();
        let voices = vec![vec![1.0; 100], vec![0.5; 100]];
        let result = align_voices(&voices, &config).expect("disabled ok");
        assert_eq!(result[0], voices[0]);
        assert_eq!(result[1], voices[1]);
    }

    #[test]
    fn test_align_voices_with_nan_in_audio() {
        let config = AlignmentConfig::new(0.5).expect("valid");
        let reference = vec![0.5f32; 2048];
        let mut target = vec![0.5f32; 2048];
        target[100] = f32::NAN; // single NaN
        let voices = vec![reference, target];
        let result = align_voices(&voices, &config).expect("handles NaN gracefully");
        // Output should be finite everywhere.
        for &val in &result[1] {
            assert!(val.is_finite(), "output must be finite, got {val}");
        }
    }

    #[test]
    fn test_blend_signals_extremes() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let zero_mix = blend_signals(&a, &b, 0.0).expect("zero mix");
        assert_eq!(zero_mix, vec![1.0, 2.0, 3.0]);
        let full_mix = blend_signals(&a, &b, 1.0).expect("full mix");
        assert_eq!(full_mix, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_config_builder_chain() {
        let config = AlignmentConfig::new(0.7)
            .expect("valid")
            .with_max_shift(240)
            .with_correlation_window(1024)
            .with_fade_samples(128);
        assert_eq!(config.max_shift_samples, 240);
        assert_eq!(config.correlation_window, 1024);
        assert_eq!(config.fade_samples, 128);
        config.validate().expect("chained config should be valid");
    }
}
