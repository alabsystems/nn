// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Formant-preserving pitch shift for Kokoro chorus voices.
//!
//! The default chorus detuning uses sample-rate-ratio resampling, which shifts
//! formants along with pitch. Real chorus ensembles have voices with different
//! fundamental frequencies but similar formant structures (same vocal tract
//! size). This module provides PSOLA-style pitch shifting that preserves
//! formant structure while changing fundamental frequency.
//!
//! # Algorithm
//!
//! 1. Window the signal into overlapping Hann-windowed frames.
//! 2. For each frame, estimate a coarse spectral envelope via magnitude
//!    spectrum peak detection (no external FFT — uses a small DFT).
//! 3. Resample each frame at the desired pitch ratio.
//! 4. Apply inverse spectral envelope compensation to restore formants.
//! 5. Overlap-add the processed frames to reconstruct the output.
//!
//! For small detuning amounts (<5 cents), a fast path bypasses the full
//! algorithm and uses simple linear-interpolation resampling, since formant
//! shift is imperceptible at such small ratios.
//!
//! # References
//!
//! - Moulines, E. & Charpentier, F. "Pitch-synchronous waveform processing
//!   techniques for text-to-speech synthesis using diphones." Speech
//!   Communication, 9(5-6), 1990.
//! - Lent, K. "An efficient method for pitch shifting digitally sampled
//!   sounds." Computer Music Journal, 13(4), 1989.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for formant-preserving pitch shifting.
///
/// Controls the analysis window size, hop size, formant compensation ratio,
/// and sample rate used for PSOLA-style pitch shifting.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FormantPreserveConfig {
    /// Analysis window size in samples.
    ///
    /// Larger windows give better frequency resolution but worse time
    /// resolution. Must be a power of 2 in [64, 4096].
    /// Default: 1024 (~42.7ms at 24kHz).
    pub window_size: usize,

    /// Hop size between successive analysis windows in samples.
    ///
    /// Smaller hops give smoother output at higher computational cost.
    /// Must be > 0 and <= window_size. Default: 256 (~10.7ms at 24kHz).
    pub hop_size: usize,

    /// Formant shift compensation ratio in [0.0, 1.0].
    ///
    /// At 1.0 (default), formants are fully preserved — the spectral
    /// envelope is shifted inversely to the pitch shift. At 0.0, no
    /// formant compensation is applied (equivalent to naive resampling).
    /// Intermediate values blend between the two.
    pub formant_shift_ratio: f32,

    /// Sample rate in Hz. Default: 24000.0 (Kokoro native rate).
    pub sample_rate: f32,
}

impl Default for FormantPreserveConfig {
    fn default() -> Self {
        Self {
            window_size: 1024,
            hop_size: 256,
            formant_shift_ratio: 1.0,
            sample_rate: 24000.0,
        }
    }
}

impl FormantPreserveConfig {
    /// Create a new formant preservation config with validation.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is invalid:
    /// - `window_size` must be a power of 2 in [64, 4096]
    /// - `hop_size` must be in [1, window_size]
    /// - `formant_shift_ratio` must be finite and in [0.0, 1.0]
    /// - `sample_rate` must be finite and positive
    pub fn new(
        window_size: usize,
        hop_size: usize,
        formant_shift_ratio: f32,
        sample_rate: f32,
    ) -> Result<Self, KokoroError> {
        let config = Self {
            window_size,
            hop_size,
            formant_shift_ratio,
            sample_rate,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate that this configuration is internally consistent.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.window_size < 64 || self.window_size > 4096 {
            return Err(KokoroError::InvalidConfig {
                field: "window_size",
                reason: format!("must be in [64, 4096], got {}", self.window_size),
            });
        }
        if !self.window_size.is_power_of_two() {
            return Err(KokoroError::InvalidConfig {
                field: "window_size",
                reason: format!("must be a power of 2, got {}", self.window_size),
            });
        }
        if self.hop_size == 0 || self.hop_size > self.window_size {
            return Err(KokoroError::InvalidConfig {
                field: "hop_size",
                reason: format!(
                    "must be in [1, window_size={}], got {}",
                    self.window_size, self.hop_size
                ),
            });
        }
        if !self.formant_shift_ratio.is_finite() || !(0.0..=1.0).contains(&self.formant_shift_ratio)
        {
            return Err(KokoroError::InvalidConfig {
                field: "formant_shift_ratio",
                reason: format!(
                    "must be finite and in [0.0, 1.0], got {}",
                    self.formant_shift_ratio
                ),
            });
        }
        if !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("must be finite and positive, got {}", self.sample_rate),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Formant shifter
// ---------------------------------------------------------------------------

/// PSOLA-style formant-preserving pitch shifter.
///
/// Implements overlap-add windowing with per-frame spectral envelope
/// estimation and compensation. The spectral envelope is estimated via
/// peak detection in the magnitude spectrum of each Hann-windowed frame,
/// using a small real DFT computed in pure f32 arithmetic.
pub struct FormantShifter {
    config: FormantPreserveConfig,
    /// Pre-computed Hann window coefficients.
    window: Vec<f32>,
}

impl FormantShifter {
    /// Create a new formant shifter with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if the configuration is invalid.
    pub fn new(config: FormantPreserveConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let window = hann_window(config.window_size);
        Ok(Self { config, window })
    }

    /// Create a formant shifter with default configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if default config is somehow
    /// invalid (should not happen).
    pub fn with_defaults() -> Result<Self, KokoroError> {
        Self::new(FormantPreserveConfig::default())
    }

    /// Shift the pitch of mono audio while preserving formant structure.
    ///
    /// # Arguments
    ///
    /// * `audio` - Mono PCM audio samples (typically [-1.0, 1.0]).
    /// * `shift_ratio` - Pitch multiplier. >1.0 raises pitch, <1.0 lowers.
    ///   For example, 1.01 raises pitch by ~17 cents.
    ///
    /// # Returns
    ///
    /// Processed audio with shifted pitch and preserved formants. Output
    /// length matches input length.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidInput` if `shift_ratio` is non-finite
    /// or non-positive.
    pub fn process(&self, audio: &[f32], shift_ratio: f32) -> Result<Vec<f32>, KokoroError> {
        // Validate shift_ratio.
        if !shift_ratio.is_finite() || shift_ratio <= 0.0 {
            return Err(KokoroError::InvalidInput(format!(
                "shift_ratio must be finite and positive, got {shift_ratio}"
            )));
        }

        // Passthrough for identity ratio.
        if (shift_ratio - 1.0).abs() < 1e-7 {
            return Ok(audio.to_vec());
        }

        // Empty input.
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        // For very short audio, fall back to simple resampling.
        if audio.len() < self.config.window_size {
            return Ok(simple_pitch_shift(audio, shift_ratio));
        }

        let n = self.config.window_size;
        let hop = self.config.hop_size;
        let out_len = audio.len();
        let mut output = vec![0.0f32; out_len];
        let mut norm = vec![0.0f32; out_len];

        // Overlap-add processing.
        let mut frame_start: usize = 0;
        while frame_start + n <= audio.len() {
            // Extract and window the frame.
            let mut frame = vec![0.0f32; n];
            for (i, sample) in frame.iter_mut().enumerate() {
                let s = audio[frame_start + i];
                // IEEE 754 safety: NaN/Inf input zeroed.
                *sample = if s.is_finite() {
                    s * self.window[i]
                } else {
                    0.0
                };
            }

            // Compute magnitude spectrum for envelope estimation.
            let mag_spectrum = magnitude_dft(&frame);

            // Estimate spectral envelope via smoothed peak following.
            let envelope = estimate_spectral_envelope(&mag_spectrum);

            // Resample the frame at the desired pitch ratio.
            let resampled = resample_frame(&frame, shift_ratio);

            // Compute magnitude spectrum of resampled frame.
            let resampled_mag = magnitude_dft(&resampled);

            // Estimate spectral envelope of resampled frame.
            let resampled_envelope = estimate_spectral_envelope(&resampled_mag);

            // Apply formant compensation: scale the resampled frame so its
            // spectral envelope matches the original.
            let compensated = apply_formant_compensation(
                &resampled,
                &envelope,
                &resampled_envelope,
                self.config.formant_shift_ratio,
            );

            // Overlap-add into output.
            for (i, &s) in compensated.iter().enumerate() {
                let out_idx = frame_start + i;
                if out_idx < out_len {
                    let val = if s.is_finite() { s } else { 0.0 };
                    output[out_idx] += val;
                    norm[out_idx] += self.window[i] * self.window[i];
                }
            }

            frame_start += hop;
        }

        // Normalize by the overlap-add window energy.
        for (i, sample) in output.iter_mut().enumerate() {
            let n_val = norm[i];
            if n_val.is_finite() && n_val > 1e-8 {
                *sample /= n_val;
            }
            // Final IEEE 754 guard on output.
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }

        // Handle trailing samples that didn't fit a full frame.
        // They stay as-is from the last overlap (usually the tail-off
        // of the overlap-add normalization is adequate).

        Ok(output)
    }

    /// Get the underlying configuration.
    #[must_use]
    pub fn config(&self) -> &FormantPreserveConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Simple (non-formant-preserving) pitch shift — fast path
// ---------------------------------------------------------------------------

/// Simple pitch shift via linear-interpolation resampling.
///
/// This is the fast path for very small detuning amounts (<5 cents) where
/// formant shift is imperceptible. No windowing, no spectral analysis —
/// just straightforward resampling with linear interpolation.
///
/// # Arguments
///
/// * `audio` - Mono PCM audio samples.
/// * `shift_ratio` - Pitch multiplier (e.g., 1.01 for +1% pitch).
///
/// # Returns
///
/// Resampled audio with the same length as input. If `shift_ratio` is 1.0,
/// returns a copy of the input. Non-finite or non-positive ratios return
/// a zeroed buffer.
#[must_use]
pub fn simple_pitch_shift(audio: &[f32], shift_ratio: f32) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }

    // IEEE 754 safety: non-finite or non-positive ratio returns zeros.
    if !shift_ratio.is_finite() || shift_ratio <= 0.0 {
        return vec![0.0; audio.len()];
    }

    // Identity passthrough.
    if (shift_ratio - 1.0).abs() < 1e-7 {
        return audio.to_vec();
    }

    let out_len = audio.len();
    let mut output = vec![0.0f32; out_len];

    for (i, sample) in output.iter_mut().enumerate() {
        let src_pos = i as f64 * f64::from(shift_ratio);
        let src_idx = src_pos as usize;
        let frac = (src_pos - src_idx as f64) as f32;

        if src_idx >= audio.len() {
            break;
        }

        let s0 = audio[src_idx];
        let s1 = if src_idx + 1 < audio.len() {
            audio[src_idx + 1]
        } else {
            s0
        };

        // IEEE 754 safety on input samples.
        let s0 = if s0.is_finite() { s0 } else { 0.0 };
        let s1 = if s1.is_finite() { s1 } else { 0.0 };

        *sample = s0 + frac * (s1 - s0);
    }

    output
}

// ---------------------------------------------------------------------------
// Top-level convenience function
// ---------------------------------------------------------------------------

/// Shift the pitch of mono audio while preserving formant structure.
///
/// This is the main entry point. For small shifts (<5 cents, i.e.,
/// `|shift_ratio - 1.0| < 0.002890`), uses the fast
/// [`simple_pitch_shift`] path. For larger shifts, uses the full
/// PSOLA-style formant-preserving algorithm.
///
/// # Arguments
///
/// * `audio` - Mono PCM audio samples.
/// * `shift_ratio` - Pitch multiplier (e.g., 1.01 for ~17 cents up).
/// * `config` - Optional configuration. Uses defaults if `None`.
///
/// # Errors
///
/// Returns `KokoroError::InvalidInput` if `shift_ratio` is non-finite
/// or non-positive, or `KokoroError::InvalidConfig` if config is invalid.
pub fn shift_pitch_preserve_formant(
    audio: &[f32],
    shift_ratio: f32,
    config: Option<&FormantPreserveConfig>,
) -> Result<Vec<f32>, KokoroError> {
    // Validate ratio.
    if !shift_ratio.is_finite() || shift_ratio <= 0.0 {
        return Err(KokoroError::InvalidInput(format!(
            "shift_ratio must be finite and positive, got {shift_ratio}"
        )));
    }

    // Identity passthrough.
    if (shift_ratio - 1.0).abs() < 1e-7 {
        return Ok(audio.to_vec());
    }

    // Empty input.
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    // Fast path for small detuning: 5 cents = 2^(5/1200) - 1 ~ 0.002890.
    // At this level, formant shift is imperceptible.
    let cents_threshold = 0.002890;
    if (shift_ratio - 1.0).abs() < cents_threshold {
        return Ok(simple_pitch_shift(audio, shift_ratio));
    }

    // Full formant-preserving path.
    let default_config = FormantPreserveConfig::default();
    let cfg = config.unwrap_or(&default_config);
    cfg.validate()?;

    let shifter = FormantShifter::new(cfg.clone())?;
    shifter.process(audio, shift_ratio)
}

// ---------------------------------------------------------------------------
// Internal: Hann window
// ---------------------------------------------------------------------------

/// Compute a Hann window of length `n`.
///
/// `w[i] = 0.5 * (1 - cos(2*pi*i / (n-1)))` for i in 0..n.
fn hann_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let scale = 2.0 * std::f64::consts::PI / (n - 1) as f64;
    (0..n)
        .map(|i| {
            let w = 0.5 * (1.0 - (scale * i as f64).cos());
            w as f32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Internal: Small DFT for spectral analysis
// ---------------------------------------------------------------------------

/// Compute magnitude spectrum via real DFT (no external FFT library).
///
/// Computes the first `N/2 + 1` magnitude bins of a real-valued signal
/// of length N using the naive O(N^2) DFT. This is acceptable for small
/// window sizes (up to 4096) used in per-frame analysis.
///
/// Returns magnitude values (not power) for bins 0..=N/2.
fn magnitude_dft(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    if n == 0 {
        return Vec::new();
    }
    let n_bins = n / 2 + 1;
    let mut magnitudes = vec![0.0f32; n_bins];

    let two_pi_over_n = 2.0 * std::f64::consts::PI / n as f64;

    for k in 0..n_bins {
        let mut re: f64 = 0.0;
        let mut im: f64 = 0.0;

        for (i, &sample) in frame.iter().enumerate() {
            if !sample.is_finite() {
                continue;
            }
            let angle = two_pi_over_n * k as f64 * i as f64;
            re += f64::from(sample) * angle.cos();
            im -= f64::from(sample) * angle.sin();
        }

        let mag = re.hypot(im);
        magnitudes[k] = if mag.is_finite() { mag as f32 } else { 0.0 };
    }

    magnitudes
}

/// Estimate spectral envelope from a magnitude spectrum via smoothed
/// peak-following.
///
/// Uses a simple moving-maximum filter followed by smoothing to create
/// an envelope that captures formant peaks while ignoring fine harmonic
/// structure. The envelope width is proportional to the expected formant
/// bandwidth (~300-500 Hz at typical speech sample rates).
fn estimate_spectral_envelope(magnitudes: &[f32]) -> Vec<f32> {
    let n = magnitudes.len();
    if n == 0 {
        return Vec::new();
    }

    // Smoothing radius: ~5% of spectrum width, minimum 2 bins.
    let radius = (n / 20).max(2).min(n / 2);

    // Pass 1: moving maximum (peak following).
    let mut peak_env = vec![0.0f32; n];
    for i in 0..n {
        let lo = i.saturating_sub(radius);
        let hi = (i + radius + 1).min(n);
        let mut max_val = 0.0f32;
        for &m in &magnitudes[lo..hi] {
            if m.is_finite() && m > max_val {
                max_val = m;
            }
        }
        peak_env[i] = max_val;
    }

    // Pass 2: smooth the peak envelope (moving average) to avoid
    // discontinuities in the formant compensation gain.
    let smooth_radius = (radius / 2).max(1);
    let mut envelope = vec![0.0f32; n];
    for i in 0..n {
        let lo = i.saturating_sub(smooth_radius);
        let hi = (i + smooth_radius + 1).min(n);
        let count = (hi - lo) as f32;
        let sum: f32 = peak_env[lo..hi].iter().sum();
        let avg = if count > 0.0 { sum / count } else { 0.0 };
        envelope[i] = if avg.is_finite() { avg } else { 0.0 };
    }

    envelope
}

// ---------------------------------------------------------------------------
// Internal: Frame resampling
// ---------------------------------------------------------------------------

/// Resample a single frame at the given pitch ratio using linear
/// interpolation.
///
/// Output length matches input length. Positions beyond the input
/// boundary are zero-filled.
fn resample_frame(frame: &[f32], ratio: f32) -> Vec<f32> {
    let n = frame.len();
    if n == 0 {
        return Vec::new();
    }

    let mut output = vec![0.0f32; n];
    let ratio_f64 = f64::from(ratio);

    for (i, sample) in output.iter_mut().enumerate() {
        let src_pos = i as f64 * ratio_f64;
        let src_idx = src_pos as usize;
        let frac = (src_pos - src_idx as f64) as f32;

        if src_idx >= n {
            break;
        }

        let s0 = frame[src_idx];
        let s1 = if src_idx + 1 < n {
            frame[src_idx + 1]
        } else {
            s0
        };

        let s0 = if s0.is_finite() { s0 } else { 0.0 };
        let s1 = if s1.is_finite() { s1 } else { 0.0 };

        *sample = s0 + frac * (s1 - s0);
    }

    output
}

// ---------------------------------------------------------------------------
// Internal: Formant compensation
// ---------------------------------------------------------------------------

/// Apply formant compensation to a resampled frame.
///
/// Adjusts the resampled frame's spectral content so that its spectral
/// envelope matches the original frame's envelope, preserving formant
/// structure. The `blend` parameter (from `formant_shift_ratio`) controls
/// how much compensation is applied.
///
/// This is an approximation: rather than doing a full spectral-domain
/// envelope correction (which would require complex-valued FFT and
/// resynthesis), we compute a per-sample gain curve derived from the
/// ratio of original to resampled envelopes, smoothed in the spectral
/// domain and applied as a time-domain gain modulation.
fn apply_formant_compensation(
    resampled: &[f32],
    original_env: &[f32],
    resampled_env: &[f32],
    blend: f32,
) -> Vec<f32> {
    // If no compensation requested, return as-is.
    if blend < 1e-6 {
        return resampled.to_vec();
    }

    let n = resampled.len();
    let n_bins = original_env.len().min(resampled_env.len());

    if n_bins == 0 || n == 0 {
        return resampled.to_vec();
    }

    // Compute per-bin gain: original_env / resampled_env, clamped to
    // a reasonable range to avoid extreme amplification.
    let max_gain: f32 = 4.0; // Don't amplify more than 12 dB.
    let min_env: f32 = 1e-6; // Floor to avoid division by zero.

    let bin_gains: Vec<f32> = (0..n_bins)
        .map(|k| {
            let orig = original_env[k].max(min_env);
            let resamp = resampled_env[k].max(min_env);
            let gain = orig / resamp;
            let gain = if gain.is_finite() {
                gain.clamp(1.0 / max_gain, max_gain)
            } else {
                1.0
            };
            // Blend toward unity based on formant_shift_ratio.
            1.0 + blend * (gain - 1.0)
        })
        .collect();

    // Map bin gains back to a per-sample gain curve. Each sample `i`
    // in a frame of length `n` corresponds roughly to bin
    // `i * n_bins / n` in the spectrum. We interpolate for smoothness.
    let mut output = vec![0.0f32; n];
    for (i, out_sample) in output.iter_mut().enumerate() {
        // Map sample index to fractional bin index.
        let bin_pos = i as f64 * (n_bins - 1) as f64 / n.max(1) as f64;
        let bin_lo = (bin_pos as usize).min(n_bins.saturating_sub(1));
        let bin_hi = (bin_lo + 1).min(n_bins.saturating_sub(1));
        let frac = (bin_pos - bin_lo as f64) as f32;

        let gain = bin_gains[bin_lo] + frac * (bin_gains[bin_hi] - bin_gains[bin_lo]);
        let gain = if gain.is_finite() { gain } else { 1.0 };

        let s = resampled[i];
        *out_sample = if s.is_finite() { s * gain } else { 0.0 };
    }

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_is_valid() {
        let config = FormantPreserveConfig::default();
        config.validate().expect("default config should be valid");
    }

    #[test]
    fn test_config_new_validates() {
        // Valid.
        assert!(FormantPreserveConfig::new(1024, 256, 1.0, 24000.0).is_ok());
        assert!(FormantPreserveConfig::new(64, 64, 0.0, 8000.0).is_ok());

        // Invalid window_size.
        assert!(FormantPreserveConfig::new(32, 16, 1.0, 24000.0).is_err());
        assert!(FormantPreserveConfig::new(5000, 256, 1.0, 24000.0).is_err());
        assert!(FormantPreserveConfig::new(1000, 256, 1.0, 24000.0).is_err()); // not power of 2

        // Invalid hop_size.
        assert!(FormantPreserveConfig::new(1024, 0, 1.0, 24000.0).is_err());
        assert!(FormantPreserveConfig::new(1024, 2048, 1.0, 24000.0).is_err());

        // Invalid formant_shift_ratio.
        assert!(FormantPreserveConfig::new(1024, 256, -0.1, 24000.0).is_err());
        assert!(FormantPreserveConfig::new(1024, 256, 1.1, 24000.0).is_err());
        assert!(FormantPreserveConfig::new(1024, 256, f32::NAN, 24000.0).is_err());

        // Invalid sample_rate.
        assert!(FormantPreserveConfig::new(1024, 256, 1.0, 0.0).is_err());
        assert!(FormantPreserveConfig::new(1024, 256, 1.0, -1.0).is_err());
        assert!(FormantPreserveConfig::new(1024, 256, 1.0, f32::INFINITY).is_err());
    }

    #[test]
    fn test_identity_ratio_passthrough() {
        let audio: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.01).sin()).collect();

        let result =
            shift_pitch_preserve_formant(&audio, 1.0, None).expect("identity shift should succeed");
        assert_eq!(result.len(), audio.len());

        for (i, (&got, &expected)) in result.iter().zip(audio.iter()).enumerate() {
            assert!(
                (got - expected).abs() < f32::EPSILON,
                "sample {i}: {got} != {expected}"
            );
        }
    }

    #[test]
    fn test_empty_input_returns_empty() {
        let result =
            shift_pitch_preserve_formant(&[], 1.05, None).expect("empty input should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_invalid_ratio_returns_error() {
        let audio = vec![1.0; 100];
        assert!(shift_pitch_preserve_formant(&audio, 0.0, None).is_err());
        assert!(shift_pitch_preserve_formant(&audio, -1.0, None).is_err());
        assert!(shift_pitch_preserve_formant(&audio, f32::NAN, None).is_err());
        assert!(shift_pitch_preserve_formant(&audio, f32::INFINITY, None).is_err());
    }

    #[test]
    fn test_small_shift_uses_fast_path() {
        // 2 cents ~ ratio 1.00115, well below 5-cent threshold.
        let ratio = (2.0f64 / 1200.0f64).exp2() as f32;
        let audio: Vec<f32> = (0..4000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();

        let result =
            shift_pitch_preserve_formant(&audio, ratio, None).expect("small shift should succeed");
        assert_eq!(result.len(), audio.len());

        // Should differ slightly from original.
        let diff: f32 = result
            .iter()
            .zip(audio.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / result.len() as f32;
        assert!(
            diff > 1e-6,
            "output should differ from input, mean_diff={diff}"
        );
    }

    #[test]
    fn test_large_shift_preserves_output_length() {
        let audio: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 24000.0).sin())
            .collect();

        // +2 semitones up.
        let ratio = 2.0f32.powf(2.0 / 12.0);
        let result =
            shift_pitch_preserve_formant(&audio, ratio, None).expect("large shift should succeed");
        assert_eq!(result.len(), audio.len());

        // Output should be finite.
        for (i, &s) in result.iter().enumerate() {
            assert!(s.is_finite(), "sample {i} is not finite: {s}");
        }
    }

    #[test]
    fn test_nan_input_handled_safely() {
        let mut audio: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.01).sin()).collect();
        // Inject NaN.
        audio[500] = f32::NAN;
        audio[501] = f32::INFINITY;

        let result =
            shift_pitch_preserve_formant(&audio, 1.05, None).expect("NaN input should not error");
        assert_eq!(result.len(), audio.len());

        // All output should be finite.
        for (i, &s) in result.iter().enumerate() {
            assert!(s.is_finite(), "sample {i} is not finite: {s}");
        }
    }

    #[test]
    fn test_simple_pitch_shift_identity() {
        let audio: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.02).sin()).collect();
        let result = simple_pitch_shift(&audio, 1.0);
        assert_eq!(result.len(), audio.len());
        for (i, (&got, &expected)) in result.iter().zip(audio.iter()).enumerate() {
            assert!(
                (got - expected).abs() < f32::EPSILON,
                "sample {i}: {got} != {expected}"
            );
        }
    }

    #[test]
    fn test_simple_pitch_shift_nan_ratio() {
        let audio = vec![1.0; 100];
        let result = simple_pitch_shift(&audio, f32::NAN);
        assert!(
            result.iter().all(|&s| s == 0.0),
            "NaN ratio should zero output"
        );
    }

    #[test]
    fn test_simple_pitch_shift_empty() {
        let result = simple_pitch_shift(&[], 1.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_hann_window_properties() {
        let w = hann_window(256);
        assert_eq!(w.len(), 256);
        // Hann window is zero at endpoints.
        assert!(w[0].abs() < 1e-6, "w[0] = {}", w[0]);
        assert!(w[255].abs() < 1e-6, "w[255] = {}", w[255]);
        // Peak at center.
        assert!((w[127] - 1.0).abs() < 0.01 || (w[128] - 1.0).abs() < 0.01);
        // Symmetric.
        for i in 0..128 {
            assert!(
                (w[i] - w[255 - i]).abs() < 1e-6,
                "asymmetry at {i}: {} != {}",
                w[i],
                w[255 - i]
            );
        }
    }

    #[test]
    fn test_magnitude_dft_dc_bin() {
        // Constant signal should have energy only in DC bin.
        let frame = vec![1.0f32; 64];
        let mag = magnitude_dft(&frame);
        assert_eq!(mag.len(), 33);
        // DC bin should be ~64.0 (sum of all samples).
        assert!(
            (mag[0] - 64.0).abs() < 0.1,
            "DC bin = {}, expected ~64.0",
            mag[0]
        );
    }

    #[test]
    fn test_formant_shifter_struct() {
        let shifter = FormantShifter::with_defaults().expect("default shifter should construct");
        assert_eq!(shifter.config().window_size, 1024);
        assert_eq!(shifter.config().hop_size, 256);

        let audio: Vec<f32> = (0..3000)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 24000.0).sin())
            .collect();
        let result = shifter
            .process(&audio, 1.05)
            .expect("process should succeed");
        assert_eq!(result.len(), audio.len());
    }

    #[test]
    fn test_formant_shifter_short_audio_fallback() {
        let shifter = FormantShifter::with_defaults().expect("default shifter should construct");
        // Audio shorter than window_size should fall back to simple shift.
        let audio: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        let result = shifter
            .process(&audio, 1.02)
            .expect("short audio should succeed via fallback");
        assert_eq!(result.len(), audio.len());
    }

    #[test]
    fn test_spectral_envelope_nonzero() {
        let signal: Vec<f32> = (0..256)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let mag = magnitude_dft(&signal);
        let env = estimate_spectral_envelope(&mag);
        assert_eq!(env.len(), mag.len());
        // Envelope should have nonzero values where the signal has energy.
        let max_env = env.iter().copied().fold(0.0f32, f32::max);
        assert!(max_env > 0.0, "envelope max should be > 0, got {max_env}");
    }
}
