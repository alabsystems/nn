// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Stereo width enhancement with mono compatibility checking for Kokoro chorus.
//!
//! Implements mid/side stereo width control, bass mono filtering, Haas effect
//! delay for perceived width, and mono compatibility monitoring/enforcement.
//!
//! # Mid/Side technique
//!
//! ```text
//! mid  = (L + R) / 2     (center content — vocals, bass, kick)
//! side = (L - R) / 2     (stereo content — reverb, panned instruments)
//!
//! width < 1.0: attenuate side → narrower image
//! width = 1.0: unchanged
//! width > 1.0: amplify side → wider image (risk of phase cancellation)
//!
//! L_out = mid + side * width
//! R_out = mid - side * width
//! ```
//!
//! # Bass mono
//!
//! Low frequencies below `bass_mono_freq` are summed to mono. This prevents
//! phase cancellation in the bass range that causes thin/hollow sound on
//! speakers. Implemented with a one-pole lowpass crossover.
//!
//! # Haas effect
//!
//! A short delay (0.1-30ms) on one channel creates a perception of width
//! without altering frequency content. At delays <30ms the brain fuses
//! both channels into a single source localized toward the earlier channel.
//!
//! # Mono compatibility
//!
//! Pearson correlation of L and R channels measures mono compatibility:
//! - 1.0: perfectly correlated (mono-safe)
//! - 0.0: uncorrelated (some cancellation in mono)
//! - -1.0: anti-correlated (complete cancellation in mono)
//!
//! The `ensure_mono_safe` function iteratively reduces width until correlation
//! meets the `correlation_floor` threshold.
//!
//! # References
//!
//! - Haas, H. (1951). "The influence of a single echo on the audibility
//!   of speech." Acustica, 1, 49-58.
//! - Gerzon, M.A. (1992). "Signal processing for simulating realistic
//!   stereo images." AES Convention Paper 3423.

use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for stereo width processing.
///
/// Controls the amount of stereo widening, bass mono crossover, Haas delay,
/// and mono compatibility enforcement. Built via method chaining.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StereoWidthConfig {
    /// Stereo width factor in [0.0, 2.0].
    ///
    /// - 0.0: full mono (side channel zeroed)
    /// - 1.0: unchanged stereo image
    /// - 2.0: ultra-wide (side channel doubled — risk of phase issues)
    ///
    /// Default: `1.0`.
    pub width: f32,

    /// Bass mono crossover frequency in Hz, [0.0, 300.0].
    ///
    /// Frequencies below this threshold are summed to mono to prevent
    /// phase cancellation in the bass range. Set to 0.0 to disable.
    ///
    /// Default: `80.0` Hz (typical subwoofer crossover).
    pub bass_mono_freq: f32,

    /// Haas effect delay in milliseconds, [0.0, 30.0].
    ///
    /// Applied to the right channel to create perceived width. The brain
    /// fuses delays <30ms into a single spatial image. Set to 0.0 to disable.
    ///
    /// Default: `0.0` (disabled).
    pub haas_delay_ms: f32,

    /// Minimum L/R correlation for mono compatibility, [0.0, 1.0].
    ///
    /// When [`ensure_mono_safe`] is called, it reduces width until L/R
    /// correlation meets or exceeds this floor. Higher values = stricter
    /// mono compatibility at the cost of narrower stereo image.
    ///
    /// Default: `0.3`.
    pub correlation_floor: f32,
}

impl Default for StereoWidthConfig {
    fn default() -> Self {
        Self {
            width: 1.0,
            bass_mono_freq: 80.0,
            haas_delay_ms: 0.0,
            correlation_floor: 0.3,
        }
    }
}

impl StereoWidthConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set stereo width factor.
    #[must_use]
    pub fn with_width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    /// Set bass mono crossover frequency in Hz.
    #[must_use]
    pub fn with_bass_mono_freq(mut self, freq: f32) -> Self {
        self.bass_mono_freq = freq;
        self
    }

    /// Set Haas effect delay in milliseconds.
    #[must_use]
    pub fn with_haas_delay_ms(mut self, ms: f32) -> Self {
        self.haas_delay_ms = ms;
        self
    }

    /// Set minimum L/R correlation floor.
    #[must_use]
    pub fn with_correlation_floor(mut self, floor: f32) -> Self {
        self.correlation_floor = floor;
        self
    }

    /// Validate all config fields.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any field is out of range or
    /// non-finite.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.width.is_finite() || !(0.0..=2.0).contains(&self.width) {
            return Err(KokoroError::InvalidConfig {
                field: "width",
                reason: format!("width = {}: must be finite and in [0.0, 2.0]", self.width),
            });
        }
        if !self.bass_mono_freq.is_finite() || !(0.0..=300.0).contains(&self.bass_mono_freq) {
            return Err(KokoroError::InvalidConfig {
                field: "bass_mono_freq",
                reason: format!(
                    "bass_mono_freq = {}: must be finite and in [0.0, 300.0]",
                    self.bass_mono_freq,
                ),
            });
        }
        if !self.haas_delay_ms.is_finite() || !(0.0..=30.0).contains(&self.haas_delay_ms) {
            return Err(KokoroError::InvalidConfig {
                field: "haas_delay_ms",
                reason: format!(
                    "haas_delay_ms = {}: must be finite and in [0.0, 30.0]",
                    self.haas_delay_ms,
                ),
            });
        }
        if !self.correlation_floor.is_finite() || !(0.0..=1.0).contains(&self.correlation_floor) {
            return Err(KokoroError::InvalidConfig {
                field: "correlation_floor",
                reason: format!(
                    "correlation_floor = {}: must be finite and in [0.0, 1.0]",
                    self.correlation_floor,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// One-pole lowpass for bass mono crossover
// ---------------------------------------------------------------------------

/// One-pole lowpass filter for bass mono crossover.
///
/// Transfer function: `y[n] = (1 - a) * x[n] + a * y[n-1]`
/// where `a = exp(-2 * pi * fc / fs)`.
#[derive(Debug, Clone)]
struct OnePoleLP {
    /// Filter coefficient.
    coeff: f32,
    /// Previous output sample (left channel).
    state_l: f32,
    /// Previous output sample (right channel).
    state_r: f32,
}

impl OnePoleLP {
    /// Create a new one-pole lowpass at the given cutoff frequency.
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let coeff = if cutoff_hz <= 0.0 || !cutoff_hz.is_finite() {
            // Disabled: coefficient = 0 means output = input (no filtering).
            0.0
        } else {
            (-2.0 * std::f32::consts::PI * cutoff_hz / sample_rate).exp()
        };
        Self {
            coeff,
            state_l: 0.0,
            state_r: 0.0,
        }
    }

    /// Process a stereo sample pair, returning the lowpass-filtered output.
    #[inline]
    fn process(&mut self, left: f32, right: f32) -> (f32, f32) {
        let one_minus_a = 1.0 - self.coeff;
        self.state_l = one_minus_a * left + self.coeff * self.state_l;
        self.state_r = one_minus_a * right + self.coeff * self.state_r;
        (self.state_l, self.state_r)
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.state_l = 0.0;
        self.state_r = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Circular delay buffer for Haas effect
// ---------------------------------------------------------------------------

/// Circular buffer implementing a fixed-length delay line.
#[derive(Debug, Clone)]
struct DelayLine {
    buffer: Vec<f32>,
    write_pos: usize,
}

impl DelayLine {
    /// Create a delay line with the given length in samples.
    ///
    /// Length of 0 means pass-through (no delay).
    fn new(delay_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
        }
    }

    /// Push a sample and return the delayed sample.
    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        if self.buffer.len() <= 1 {
            return input;
        }
        let output = self.buffer[self.write_pos];
        self.buffer[self.write_pos] = input;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        output
    }

    /// Reset delay buffer to silence.
    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }
}

// ---------------------------------------------------------------------------
// Stereo widener processor
// ---------------------------------------------------------------------------

/// Stereo width enhancement processor with mid/side processing, bass mono
/// crossover, and Haas effect delay.
///
/// Created from a [`StereoWidthConfig`] and processes stereo audio in-place.
/// Maintains internal state for the crossover filter and Haas delay line,
/// so it should be used for a single continuous audio stream.
#[derive(Debug, Clone)]
pub struct StereoWidener {
    /// Width factor applied to the side channel.
    width: f32,
    /// One-pole lowpass for bass mono crossover.
    crossover: OnePoleLP,
    /// Whether bass mono processing is enabled (freq > 0).
    bass_mono_enabled: bool,
    /// Haas effect delay line for the right channel.
    haas_delay: DelayLine,
    /// Whether Haas delay is enabled (delay > 0).
    haas_enabled: bool,
}

impl StereoWidener {
    /// Create a new stereo widener from the given config.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any config field is invalid.
    pub fn new(config: &StereoWidthConfig) -> Result<Self, KokoroError> {
        config.validate()?;

        let bass_mono_enabled = config.bass_mono_freq > 0.0;
        let crossover = OnePoleLP::new(config.bass_mono_freq, KOKORO_SAMPLE_RATE as f32);

        let haas_samples = (config.haas_delay_ms * 0.001 * KOKORO_SAMPLE_RATE as f32) as usize;
        let haas_enabled = haas_samples > 0;
        let haas_delay = DelayLine::new(haas_samples);

        Ok(Self {
            width: config.width,
            crossover,
            bass_mono_enabled,
            haas_delay,
            haas_enabled,
        })
    }

    /// Process stereo audio in-place, applying width, bass mono, and Haas
    /// effect.
    ///
    /// Both slices must have the same length; if they differ, the shorter
    /// length is used and the remainder is untouched.
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        let len = left.len().min(right.len());

        for i in 0..len {
            let l = left[i];
            let r = right[i];

            // Mid/Side decomposition.
            let mid = (l + r) * 0.5;
            let side = (l - r) * 0.5;

            // Scale the side channel by the width factor.
            let scaled_side = side * self.width;

            // Reconstruct L/R from mid/side.
            let mut out_l = mid + scaled_side;
            let mut out_r = mid - scaled_side;

            // Bass mono: replace low-frequency stereo content with mono.
            if self.bass_mono_enabled {
                let (lp_l, lp_r) = self.crossover.process(out_l, out_r);
                // High-pass = original - lowpass.
                let hp_l = out_l - lp_l;
                let hp_r = out_r - lp_r;
                // Bass is summed to mono.
                let bass_mono = (lp_l + lp_r) * 0.5;
                out_l = bass_mono + hp_l;
                out_r = bass_mono + hp_r;
            }

            // Haas delay on the right channel.
            if self.haas_enabled {
                out_r = self.haas_delay.process(out_r);
            }

            left[i] = out_l;
            right[i] = out_r;
        }
    }

    /// Reset all internal state (filter memory and delay buffer).
    ///
    /// Call this when starting a new audio stream to avoid clicks from
    /// leftover state.
    pub fn reset(&mut self) {
        self.crossover.reset();
        self.haas_delay.reset();
    }
}

// ---------------------------------------------------------------------------
// Mono compatibility checking
// ---------------------------------------------------------------------------

/// Compute the Pearson correlation coefficient between left and right channels.
///
/// Returns a value in [-1.0, 1.0]:
/// - 1.0: perfectly correlated (mono-safe, no cancellation)
/// - 0.0: uncorrelated (partial cancellation in mono)
/// - -1.0: anti-correlated (complete cancellation in mono)
///
/// Returns 1.0 for empty or silent (zero-energy) signals (trivially mono-safe).
#[must_use]
pub fn check_mono_compatibility(left: &[f32], right: &[f32]) -> f32 {
    let len = left.len().min(right.len());
    if len == 0 {
        return 1.0;
    }

    // Compute means.
    let mut sum_l = 0.0f64;
    let mut sum_r = 0.0f64;
    for i in 0..len {
        sum_l += f64::from(left[i]);
        sum_r += f64::from(right[i]);
    }
    let mean_l = sum_l / len as f64;
    let mean_r = sum_r / len as f64;

    // Compute covariance and variances.
    let mut cov = 0.0f64;
    let mut var_l = 0.0f64;
    let mut var_r = 0.0f64;
    for i in 0..len {
        let dl = f64::from(left[i]) - mean_l;
        let dr = f64::from(right[i]) - mean_r;
        cov += dl * dr;
        var_l += dl * dl;
        var_r += dr * dr;
    }

    let denom = (var_l * var_r).sqrt();
    if denom < 1e-12 {
        // Zero or near-zero energy: trivially mono-safe.
        return 1.0;
    }

    (cov / denom) as f32
}

/// Ensure mono compatibility by reducing stereo width until L/R correlation
/// meets the given floor.
///
/// Iteratively reduces the effective width from its current level toward 0.0
/// (mono) in steps of 0.05 until `check_mono_compatibility(left, right) >= floor`.
/// The audio is modified in-place at the final width setting.
///
/// # Arguments
///
/// - `left`, `right`: stereo audio buffers (modified in-place)
/// - `floor`: minimum acceptable L/R correlation (0.0 to 1.0)
/// - `initial_width`: the starting width factor to reduce from
///
/// # Returns
///
/// The final width factor that was applied (may equal `initial_width` if
/// already mono-safe, or a reduced value).
pub fn ensure_mono_safe(
    left: &mut [f32],
    right: &mut [f32],
    floor: f32,
    initial_width: f32,
) -> f32 {
    let floor_clamped = floor.clamp(0.0, 1.0);
    let len = left.len().min(right.len());

    if len == 0 {
        return initial_width;
    }

    // Check current correlation.
    let corr = check_mono_compatibility(left, right);
    if corr >= floor_clamped {
        return initial_width;
    }

    // Need to reduce width. Keep original mid signal for reprocessing.
    let mut orig_mid = Vec::with_capacity(len);
    let mut orig_side = Vec::with_capacity(len);
    for i in 0..len {
        orig_mid.push((left[i] + right[i]) * 0.5);
        orig_side.push((left[i] - right[i]) * 0.5);
    }

    // Binary search for the minimum width that satisfies the floor.
    let mut lo = 0.0f32;
    let mut hi = initial_width;
    let mut best_width = 0.0f32;
    let iterations = 20; // Converges to ~1e-6 precision.

    for _ in 0..iterations {
        let mid_w = (lo + hi) * 0.5;

        // Reconstruct at this width.
        for i in 0..len {
            left[i] = orig_mid[i] + orig_side[i] * mid_w;
            right[i] = orig_mid[i] - orig_side[i] * mid_w;
        }

        let c = check_mono_compatibility(left, right);
        if c >= floor_clamped {
            best_width = mid_w;
            lo = mid_w;
        } else {
            hi = mid_w;
        }
    }

    // Apply the best width found.
    for i in 0..len {
        left[i] = orig_mid[i] + orig_side[i] * best_width;
        right[i] = orig_mid[i] - orig_side[i] * best_width;
    }

    best_width
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = KOKORO_SAMPLE_RATE as f32;

    // -- Config tests -------------------------------------------------------

    #[test]
    fn test_config_defaults() {
        let cfg = StereoWidthConfig::new();
        assert!((cfg.width - 1.0).abs() < f32::EPSILON);
        assert!((cfg.bass_mono_freq - 80.0).abs() < f32::EPSILON);
        assert!((cfg.haas_delay_ms - 0.0).abs() < f32::EPSILON);
        assert!((cfg.correlation_floor - 0.3).abs() < f32::EPSILON);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let cfg = StereoWidthConfig::new()
            .with_width(1.5)
            .with_bass_mono_freq(120.0)
            .with_haas_delay_ms(15.0)
            .with_correlation_floor(0.5);
        assert!((cfg.width - 1.5).abs() < f32::EPSILON);
        assert!((cfg.bass_mono_freq - 120.0).abs() < f32::EPSILON);
        assert!((cfg.haas_delay_ms - 15.0).abs() < f32::EPSILON);
        assert!((cfg.correlation_floor - 0.5).abs() < f32::EPSILON);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_width_out_of_range() {
        let cfg = StereoWidthConfig::new().with_width(3.0);
        assert!(cfg.validate().is_err());

        let cfg = StereoWidthConfig::new().with_width(-0.1);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_nan() {
        let cfg = StereoWidthConfig::new().with_width(f32::NAN);
        assert!(cfg.validate().is_err());

        let cfg = StereoWidthConfig::new().with_bass_mono_freq(f32::INFINITY);
        assert!(cfg.validate().is_err());

        let cfg = StereoWidthConfig::new().with_haas_delay_ms(f32::NAN);
        assert!(cfg.validate().is_err());

        let cfg = StereoWidthConfig::new().with_correlation_floor(f32::NAN);
        assert!(cfg.validate().is_err());
    }

    // -- Width = 0 produces mono -------------------------------------------

    #[test]
    fn test_width_zero_produces_mono() {
        let cfg = StereoWidthConfig::new()
            .with_width(0.0)
            .with_bass_mono_freq(0.0); // Disable bass mono for clean test.
        let mut widener = StereoWidener::new(&cfg).unwrap();

        let mut left = vec![1.0, 0.5, -0.3, 0.8];
        let mut right = vec![-0.2, 0.7, 0.1, -0.5];

        // Expected: mid = (L+R)/2, side = 0 => L_out == R_out == mid.
        let expected_mid: Vec<f32> = left
            .iter()
            .zip(right.iter())
            .map(|(&l, &r)| (l + r) * 0.5)
            .collect();

        widener.process(&mut left, &mut right);

        for i in 0..left.len() {
            assert!(
                (left[i] - expected_mid[i]).abs() < 1e-6,
                "left[{i}]: expected {}, got {}",
                expected_mid[i],
                left[i],
            );
            assert!(
                (right[i] - expected_mid[i]).abs() < 1e-6,
                "right[{i}]: expected {}, got {}",
                expected_mid[i],
                right[i],
            );
        }
    }

    // -- Width = 2 increases side energy -----------------------------------

    #[test]
    fn test_width_two_increases_side_energy() {
        let cfg_normal = StereoWidthConfig::new()
            .with_width(1.0)
            .with_bass_mono_freq(0.0);
        let cfg_wide = StereoWidthConfig::new()
            .with_width(2.0)
            .with_bass_mono_freq(0.0);

        // Generate a test signal with stereo content.
        let n = 1024;
        let mut left_normal = Vec::with_capacity(n);
        let mut right_normal = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / SAMPLE_RATE;
            // Different frequencies in L and R to create stereo content.
            left_normal.push((440.0 * 2.0 * std::f32::consts::PI * t).sin());
            right_normal.push((660.0 * 2.0 * std::f32::consts::PI * t).sin());
        }
        let mut left_wide = left_normal.clone();
        let mut right_wide = right_normal.clone();

        let mut widener_normal = StereoWidener::new(&cfg_normal).unwrap();
        let mut widener_wide = StereoWidener::new(&cfg_wide).unwrap();

        widener_normal.process(&mut left_normal, &mut right_normal);
        widener_wide.process(&mut left_wide, &mut right_wide);

        // Compute side energy for both.
        let side_energy_normal: f32 = left_normal
            .iter()
            .zip(right_normal.iter())
            .map(|(&l, &r)| {
                let side = (l - r) * 0.5;
                side * side
            })
            .sum();
        let side_energy_wide: f32 = left_wide
            .iter()
            .zip(right_wide.iter())
            .map(|(&l, &r)| {
                let side = (l - r) * 0.5;
                side * side
            })
            .sum();

        assert!(
            side_energy_wide > side_energy_normal * 1.5,
            "Wide side energy ({side_energy_wide}) should be significantly \
             greater than normal ({side_energy_normal})",
        );
    }

    // -- Bass mono works ---------------------------------------------------

    #[test]
    fn test_bass_mono_reduces_low_freq_stereo() {
        let n = 4096;
        let freq = 40.0; // Well below 80 Hz crossover.

        // Pure low-frequency stereo signal: L = sine, R = -sine (anti-phase).
        let mut left: Vec<f32> = (0..n)
            .map(|i| (freq * 2.0 * std::f32::consts::PI * i as f32 / SAMPLE_RATE).sin())
            .collect();
        let mut right: Vec<f32> = left.iter().map(|&s| -s).collect();

        let cfg = StereoWidthConfig::new()
            .with_width(1.0)
            .with_bass_mono_freq(80.0);
        let mut widener = StereoWidener::new(&cfg).unwrap();
        widener.process(&mut left, &mut right);

        // After bass mono, the low-frequency content should be closer to mono.
        // Measure side energy in the latter half (after filter settles).
        let start = n / 2;
        let side_energy: f32 = left[start..]
            .iter()
            .zip(right[start..].iter())
            .map(|(&l, &r)| {
                let side = (l - r) * 0.5;
                side * side
            })
            .sum();

        // Original side energy of the anti-phase signal would be high.
        // After bass mono, it should be substantially reduced.
        let original_side_energy: f32 = (start..n)
            .map(|i| {
                let s = (freq * 2.0 * std::f32::consts::PI * i as f32 / SAMPLE_RATE).sin();
                // Original: L = sin, R = -sin, side = (sin - (-sin))/2 = sin.
                s * s
            })
            .sum();

        assert!(
            side_energy < original_side_energy * 0.5,
            "Bass mono should substantially reduce low-freq side energy: \
             got {side_energy} vs original {original_side_energy}",
        );
    }

    // -- Haas delay works --------------------------------------------------

    #[test]
    fn test_haas_delay_offsets_right_channel() {
        let delay_ms = 5.0;
        let cfg = StereoWidthConfig::new()
            .with_width(1.0)
            .with_bass_mono_freq(0.0)
            .with_haas_delay_ms(delay_ms);
        let mut widener = StereoWidener::new(&cfg).unwrap();

        let delay_samples = (delay_ms * 0.001 * SAMPLE_RATE) as usize;
        let n = delay_samples * 4;

        // Impulse in both channels.
        let mut left = vec![0.0f32; n];
        let mut right = vec![0.0f32; n];
        left[0] = 1.0;
        right[0] = 1.0;

        widener.process(&mut left, &mut right);

        // Left channel should have the impulse at position 0 (mid + side).
        // Right channel impulse should be delayed by delay_samples.
        // With width=1.0 and no bass mono, mid/side is identity transform.
        assert!(
            left[0].abs() > 0.1,
            "Left should have impulse near sample 0",
        );
        assert!(
            right[delay_samples].abs() > 0.1,
            "Right should have delayed impulse at sample {delay_samples}, got {}",
            right[delay_samples],
        );
        // Early samples of right channel should be near zero (delay line was
        // initialized to silence).
        for i in 0..delay_samples.min(n) {
            assert!(
                right[i].abs() < 1e-6,
                "right[{i}] = {} should be ~0 (delay not yet reached)",
                right[i],
            );
        }
    }

    // -- Mono compatibility ------------------------------------------------

    #[test]
    fn test_mono_compatibility_identical_channels() {
        let signal: Vec<f32> = (0..1024)
            .map(|i| (440.0 * 2.0 * std::f32::consts::PI * i as f32 / SAMPLE_RATE).sin())
            .collect();
        let corr = check_mono_compatibility(&signal, &signal);
        assert!(
            (corr - 1.0).abs() < 1e-4,
            "Identical channels should have correlation ~1.0, got {corr}",
        );
    }

    #[test]
    fn test_mono_compatibility_anti_phase() {
        let signal: Vec<f32> = (0..1024)
            .map(|i| (440.0 * 2.0 * std::f32::consts::PI * i as f32 / SAMPLE_RATE).sin())
            .collect();
        let inverted: Vec<f32> = signal.iter().map(|&s| -s).collect();
        let corr = check_mono_compatibility(&signal, &inverted);
        assert!(
            (corr - (-1.0)).abs() < 1e-4,
            "Anti-phase channels should have correlation ~-1.0, got {corr}",
        );
    }

    #[test]
    fn test_mono_compatibility_empty() {
        assert!((check_mono_compatibility(&[], &[]) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mono_compatibility_silent() {
        let zeros = vec![0.0f32; 100];
        assert!((check_mono_compatibility(&zeros, &zeros) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ensure_mono_safe_reduces_width() {
        let n = 2048;
        // Create anti-phase signal (worst case for mono compatibility).
        let mut left: Vec<f32> = (0..n)
            .map(|i| (440.0 * 2.0 * std::f32::consts::PI * i as f32 / SAMPLE_RATE).sin())
            .collect();
        let mut right: Vec<f32> = left.iter().map(|&s| -s).collect();

        let floor = 0.5;
        let final_width = ensure_mono_safe(&mut left, &mut right, floor, 2.0);

        // Width should have been reduced.
        assert!(
            final_width < 2.0,
            "Width should be reduced from 2.0, got {final_width}",
        );

        // Resulting correlation should meet the floor.
        let corr = check_mono_compatibility(&left, &right);
        assert!(
            corr >= floor - 0.01, // Small tolerance for binary search precision.
            "Correlation {corr} should be >= floor {floor} (minus tolerance)",
        );
    }

    #[test]
    fn test_ensure_mono_safe_no_change_if_already_safe() {
        let n = 1024;
        // Identical channels = perfectly mono-safe.
        let signal: Vec<f32> = (0..n)
            .map(|i| (440.0 * 2.0 * std::f32::consts::PI * i as f32 / SAMPLE_RATE).sin())
            .collect();
        let mut left = signal.clone();
        let mut right = signal;

        let final_width = ensure_mono_safe(&mut left, &mut right, 0.5, 1.5);
        assert!(
            (final_width - 1.5).abs() < f32::EPSILON,
            "Width should remain 1.5 when already mono-safe, got {final_width}",
        );
    }

    // -- Reset clears state ------------------------------------------------

    #[test]
    fn test_reset_clears_state() {
        let cfg = StereoWidthConfig::new()
            .with_width(1.0)
            .with_bass_mono_freq(80.0)
            .with_haas_delay_ms(10.0);
        let mut widener = StereoWidener::new(&cfg).unwrap();

        // Process some audio to fill state.
        let mut left = vec![1.0; 256];
        let mut right = vec![-1.0; 256];
        widener.process(&mut left, &mut right);

        // Reset.
        widener.reset();

        // After reset, processing an impulse should behave like fresh state.
        let mut left2 = vec![0.0f32; 256];
        let mut right2 = vec![0.0f32; 256];
        left2[0] = 1.0;
        right2[0] = 1.0;

        let mut widener_fresh = StereoWidener::new(&cfg).unwrap();
        let mut left3 = left2.clone();
        let mut right3 = right2.clone();

        widener.process(&mut left2, &mut right2);
        widener_fresh.process(&mut left3, &mut right3);

        for i in 0..left2.len() {
            assert!(
                (left2[i] - left3[i]).abs() < 1e-6,
                "After reset, left[{i}] differs: {} vs {}",
                left2[i],
                left3[i],
            );
            assert!(
                (right2[i] - right3[i]).abs() < 1e-6,
                "After reset, right[{i}] differs: {} vs {}",
                right2[i],
                right3[i],
            );
        }
    }

    // -- Width = 1 is identity (no bass mono, no haas) ---------------------

    #[test]
    fn test_width_one_is_identity() {
        let cfg = StereoWidthConfig::new()
            .with_width(1.0)
            .with_bass_mono_freq(0.0)
            .with_haas_delay_ms(0.0);
        let mut widener = StereoWidener::new(&cfg).unwrap();

        let left_orig = vec![0.3, -0.7, 0.5, 0.1];
        let right_orig = vec![-0.2, 0.4, 0.8, -0.6];
        let mut left = left_orig.clone();
        let mut right = right_orig.clone();

        widener.process(&mut left, &mut right);

        for i in 0..left.len() {
            assert!(
                (left[i] - left_orig[i]).abs() < 1e-6,
                "Width=1 should be identity for left[{i}]: {} vs {}",
                left[i],
                left_orig[i],
            );
            assert!(
                (right[i] - right_orig[i]).abs() < 1e-6,
                "Width=1 should be identity for right[{i}]: {} vs {}",
                right[i],
                right_orig[i],
            );
        }
    }
}
