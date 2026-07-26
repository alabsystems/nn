// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Frequency-dependent stereo imaging for Kokoro chorus.
//!
//! Standard stereo processing applies a single width setting to the entire
//! signal. This produces an unsatisfying result because bass, midrange, and
//! treble have different perceptual stereo requirements:
//!
//! - **Bass (< crossover_low):** Should be narrow/mono for mono compatibility,
//!   subwoofer coherence, and phase-cancellation avoidance.
//! - **Midrange (crossover_low..crossover_high):** Carries voice body; moderate
//!   width produces a natural ensemble effect.
//! - **Highs (> crossover_high):** Air and presence; wider imaging creates
//!   the perception of space and depth.
//!
//! # Signal flow
//!
//! ```text
//! L,R input
//!   -> Linkwitz-Riley crossover (2nd order = 2 cascaded 1-pole LPF/HPF)
//!      -> low_L, low_R  (< low_crossover Hz)
//!      -> mid_L, mid_R  (low_crossover..high_crossover Hz)
//!      -> high_L, high_R (> high_crossover Hz)
//!   -> per-band mid/side width adjustment
//!   -> sum bands -> L,R output
//! ```
//!
//! # Mid/Side processing
//!
//! For each band:
//! ```text
//! mid  = (L + R) * 0.5
//! side = (L - R) * 0.5
//! side *= width
//! L = mid + side
//! R = mid - side
//! ```
//!
//! Width = 0.0 collapses to mono, 1.0 = original stereo, 2.0 = double side
//! content (hyper-stereo).
//!
//! Part of #4264, #3351.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for frequency-dependent multi-band stereo processing.
///
/// Splits the stereo signal into three frequency bands (low, mid, high) via
/// Linkwitz-Riley crossover filters and applies independent stereo width to
/// each band. This enables keeping bass mono-compatible while widening the
/// high-frequency "air" for spatial depth.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MultibandStereoConfig {
    /// Stereo width for frequencies below `low_crossover`.
    /// Range: 0.0 (mono) to 1.5 (wider than original).
    /// Typically narrow (0.0-0.5) for mono compatibility.
    pub low_width: f32,

    /// Stereo width for frequencies between `low_crossover` and `high_crossover`.
    /// Range: 0.5 to 2.0. Voice body lives here; moderate width sounds natural.
    pub mid_width: f32,

    /// Stereo width for frequencies above `high_crossover`.
    /// Range: 0.5 to 2.5. Wider settings create airy, spacious imaging.
    pub high_width: f32,

    /// Low/mid crossover frequency in Hz.
    /// Range: 80.0 to 400.0. Typically 150-300 Hz for voice content.
    pub low_crossover: f32,

    /// Mid/high crossover frequency in Hz.
    /// Range: 2000.0 to 8000.0. Typically 3000-6000 Hz for voice presence.
    pub high_crossover: f32,
}

impl MultibandStereoConfig {
    /// Create a new config with the given parameters.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range
    /// or non-finite.
    pub fn new(
        low_width: f32,
        mid_width: f32,
        high_width: f32,
        low_crossover: f32,
        high_crossover: f32,
    ) -> Result<Self, KokoroError> {
        let config = Self {
            low_width,
            mid_width,
            high_width,
            low_crossover,
            high_crossover,
        };
        config.validate()?;
        Ok(config)
    }

    /// Set stereo width for the low band.
    #[must_use]
    pub fn with_low_width(mut self, width: f32) -> Self {
        self.low_width = width;
        self
    }

    /// Set stereo width for the mid band.
    #[must_use]
    pub fn with_mid_width(mut self, width: f32) -> Self {
        self.mid_width = width;
        self
    }

    /// Set stereo width for the high band.
    #[must_use]
    pub fn with_high_width(mut self, width: f32) -> Self {
        self.high_width = width;
        self
    }

    /// Set the low/mid crossover frequency.
    #[must_use]
    pub fn with_low_crossover(mut self, freq: f32) -> Self {
        self.low_crossover = freq;
        self
    }

    /// Set the mid/high crossover frequency.
    #[must_use]
    pub fn with_high_crossover(mut self, freq: f32) -> Self {
        self.high_crossover = freq;
        self
    }

    /// Validate all parameters are within acceptable ranges.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range,
    /// non-finite, or if `low_crossover >= high_crossover`.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.low_width.is_finite() || !(0.0..=1.5).contains(&self.low_width) {
            return Err(KokoroError::InvalidConfig {
                field: "low_width",
                reason: format!(
                    "low_width = {}: must be finite and in [0.0, 1.5]",
                    self.low_width,
                ),
            });
        }
        if !self.mid_width.is_finite() || !(0.5..=2.0).contains(&self.mid_width) {
            return Err(KokoroError::InvalidConfig {
                field: "mid_width",
                reason: format!(
                    "mid_width = {}: must be finite and in [0.5, 2.0]",
                    self.mid_width,
                ),
            });
        }
        if !self.high_width.is_finite() || !(0.5..=2.5).contains(&self.high_width) {
            return Err(KokoroError::InvalidConfig {
                field: "high_width",
                reason: format!(
                    "high_width = {}: must be finite and in [0.5, 2.5]",
                    self.high_width,
                ),
            });
        }
        if !self.low_crossover.is_finite() || !(80.0..=400.0).contains(&self.low_crossover) {
            return Err(KokoroError::InvalidConfig {
                field: "low_crossover",
                reason: format!(
                    "low_crossover = {}: must be finite and in [80.0, 400.0]",
                    self.low_crossover,
                ),
            });
        }
        if !self.high_crossover.is_finite() || !(2000.0..=8000.0).contains(&self.high_crossover) {
            return Err(KokoroError::InvalidConfig {
                field: "high_crossover",
                reason: format!(
                    "high_crossover = {}: must be finite and in [2000.0, 8000.0]",
                    self.high_crossover,
                ),
            });
        }
        if self.low_crossover >= self.high_crossover {
            return Err(KokoroError::InvalidConfig {
                field: "low_crossover",
                reason: format!(
                    "low_crossover ({}) must be < high_crossover ({})",
                    self.low_crossover, self.high_crossover,
                ),
            });
        }
        Ok(())
    }
}

impl Default for MultibandStereoConfig {
    fn default() -> Self {
        Self {
            low_width: 0.3,
            mid_width: 1.0,
            high_width: 1.5,
            low_crossover: 200.0,
            high_crossover: 4000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

/// Named presets for common multi-band stereo configurations.
///
/// Each preset is tuned for a specific use case of TTS chorus mixing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MultibandStereoPreset {
    /// Vocal: Narrow low (mono-safe bass), moderate mid (natural ensemble),
    /// wide high (airy presence). Best for speech/singing.
    Vocal,
    /// Cinematic: Very narrow low, wide mid (dramatic body), ultra-wide high
    /// (spacious reverb tails). Best for dramatic narration.
    Cinematic,
    /// Radio: Mono low (FM/streaming safe), narrow mid, moderate high.
    /// Maximizes mono compatibility for lossy distribution.
    Radio,
}

impl MultibandStereoPreset {
    /// Convert this preset to a [`MultibandStereoConfig`].
    #[must_use]
    pub fn to_config(self) -> MultibandStereoConfig {
        match self {
            Self::Vocal => MultibandStereoConfig {
                low_width: 0.3,
                mid_width: 1.0,
                high_width: 1.5,
                low_crossover: 200.0,
                high_crossover: 4000.0,
            },
            Self::Cinematic => MultibandStereoConfig {
                low_width: 0.2,
                mid_width: 1.3,
                high_width: 2.0,
                low_crossover: 150.0,
                high_crossover: 5000.0,
            },
            Self::Radio => MultibandStereoConfig {
                low_width: 0.0,
                mid_width: 0.7,
                high_width: 1.0,
                low_crossover: 200.0,
                high_crossover: 3500.0,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// One-pole filter (building block for Linkwitz-Riley)
// ---------------------------------------------------------------------------

/// First-order one-pole lowpass filter.
///
/// H(z) = g / (1 - (1-g) * z^-1) where g = 1 - exp(-2*pi*fc/fs).
/// Two cascaded one-pole lowpass filters create a 2nd-order Linkwitz-Riley
/// lowpass (12 dB/oct, linear-phase summation with its complementary HPF).
#[derive(Debug, Clone)]
struct OnePole {
    /// Filter coefficient (cutoff-dependent).
    g: f32,
    /// Previous output sample (filter state).
    z1: f32,
}

impl OnePole {
    /// Create a new one-pole lowpass filter for the given cutoff and sample rate.
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        // g = 1 - exp(-2*pi*fc/fs), clamped to (0, 1) for stability.
        let g = (1.0 - (-2.0 * std::f32::consts::PI * cutoff_hz / sample_rate).exp())
            .clamp(0.001, 0.999);
        Self { g, z1: 0.0 }
    }

    /// Process one sample through the lowpass filter, returning the LP output.
    #[inline]
    fn process_lp(&mut self, input: f32) -> f32 {
        // Guard against NaN/Inf propagation.
        if !input.is_finite() {
            self.z1 = 0.0;
            return 0.0;
        }
        self.z1 += self.g * (input - self.z1);
        // Flush denormals.
        if self.z1.abs() < 1e-20 {
            self.z1 = 0.0;
        }
        self.z1
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.z1 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Linkwitz-Riley 2nd-order crossover (2 cascaded one-poles)
// ---------------------------------------------------------------------------

/// Linkwitz-Riley 2nd-order (LR2) crossover filter.
///
/// Two cascaded one-pole lowpass filters produce a 12 dB/octave rolloff.
/// The highpass output is computed as `input - lowpass`, ensuring the LP and
/// HP outputs sum to the original signal (perfect reconstruction).
///
/// This is the standard crossover topology used in professional audio for
/// band-splitting because the recombined signal has flat magnitude response
/// and linear phase at the crossover frequency.
#[derive(Debug, Clone)]
struct LinkwitzRileyCrossover {
    /// First one-pole stage.
    stage1: OnePole,
    /// Second one-pole stage (cascaded).
    stage2: OnePole,
}

impl LinkwitzRileyCrossover {
    /// Create a new LR2 crossover at the given frequency.
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        Self {
            stage1: OnePole::new(cutoff_hz, sample_rate),
            stage2: OnePole::new(cutoff_hz, sample_rate),
        }
    }

    /// Process one sample, returning `(lowpass, highpass)`.
    ///
    /// The outputs satisfy `lowpass + highpass == input` (within floating-point
    /// precision), ensuring perfect reconstruction when bands are recombined.
    #[inline]
    fn process(&mut self, input: f32) -> (f32, f32) {
        let lp1 = self.stage1.process_lp(input);
        let lp = self.stage2.process_lp(lp1);
        (lp, input - lp)
    }

    /// Reset both stages.
    fn reset(&mut self) {
        self.stage1.reset();
        self.stage2.reset();
    }
}

// ---------------------------------------------------------------------------
// Multi-band stereo processor
// ---------------------------------------------------------------------------

/// Frequency-dependent stereo width processor.
///
/// Splits stereo audio into three frequency bands using Linkwitz-Riley
/// crossovers, applies independent mid/side stereo width to each band,
/// and recombines. This allows keeping bass mono while widening highs.
///
/// # Usage
///
/// ```rust,no_run
/// use nn_models::kokoro_chorus_multiband_stereo::*;
///
/// let config = MultibandStereoPreset::Vocal.to_config();
/// let mut proc = MultibandStereoProcessor::new(&config, 24000.0).unwrap();
///
/// let mut left = vec![0.0f32; 1024];
/// let mut right = vec![0.0f32; 1024];
/// // ... fill with audio ...
/// proc.process(&mut left, &mut right);
/// ```
pub struct MultibandStereoProcessor {
    /// Low/mid crossover for left channel.
    xover_low_l: LinkwitzRileyCrossover,
    /// Low/mid crossover for right channel.
    xover_low_r: LinkwitzRileyCrossover,
    /// Mid/high crossover for left channel.
    xover_high_l: LinkwitzRileyCrossover,
    /// Mid/high crossover for right channel.
    xover_high_r: LinkwitzRileyCrossover,
    /// Per-band stereo widths.
    low_width: f32,
    mid_width: f32,
    high_width: f32,
}

impl MultibandStereoProcessor {
    /// Create a new multi-band stereo processor.
    ///
    /// # Arguments
    ///
    /// * `config` - Band configuration (widths + crossover frequencies).
    /// * `sample_rate` - Audio sample rate in Hz (e.g. 24000.0 for Kokoro).
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if config validation fails or
    /// sample rate is invalid.
    pub fn new(config: &MultibandStereoConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || !(1000.0..=192000.0).contains(&sample_rate) {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!(
                    "sample_rate = {sample_rate}: must be finite and in [1000, 192000]"
                ),
            });
        }

        Ok(Self {
            xover_low_l: LinkwitzRileyCrossover::new(config.low_crossover, sample_rate),
            xover_low_r: LinkwitzRileyCrossover::new(config.low_crossover, sample_rate),
            xover_high_l: LinkwitzRileyCrossover::new(config.high_crossover, sample_rate),
            xover_high_r: LinkwitzRileyCrossover::new(config.high_crossover, sample_rate),
            low_width: config.low_width,
            mid_width: config.mid_width,
            high_width: config.high_width,
        })
    }

    /// Process stereo audio in-place with frequency-dependent stereo width.
    ///
    /// Both slices must have the same length. If they differ, processes up to
    /// the shorter length.
    ///
    /// # Signal flow per sample
    ///
    /// 1. Split L and R into low / (mid+high) via the low crossover.
    /// 2. Split (mid+high) into mid / high via the high crossover.
    /// 3. Apply mid/side width independently to each band.
    /// 4. Sum the three bands back into L and R.
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        let len = left.len().min(right.len());

        for i in 0..len {
            let l = left[i];
            let r = right[i];

            // Guard non-finite input.
            if !l.is_finite() || !r.is_finite() {
                left[i] = 0.0;
                right[i] = 0.0;
                continue;
            }

            // --- Band splitting ---

            // Low crossover: split into low and (mid+high).
            let (low_l, midhigh_l) = self.xover_low_l.process(l);
            let (low_r, midhigh_r) = self.xover_low_r.process(r);

            // High crossover: split (mid+high) into mid and high.
            let (mid_l, high_l) = self.xover_high_l.process(midhigh_l);
            let (mid_r, high_r) = self.xover_high_r.process(midhigh_r);

            // --- Per-band mid/side processing ---

            let (out_low_l, out_low_r) = apply_midside_width(low_l, low_r, self.low_width);
            let (out_mid_l, out_mid_r) = apply_midside_width(mid_l, mid_r, self.mid_width);
            let (out_high_l, out_high_r) = apply_midside_width(high_l, high_r, self.high_width);

            // --- Recombine bands ---

            left[i] = out_low_l + out_mid_l + out_high_l;
            right[i] = out_low_r + out_mid_r + out_high_r;

            // Clamp non-finite results from floating-point edge cases.
            if !left[i].is_finite() {
                left[i] = 0.0;
            }
            if !right[i].is_finite() {
                right[i] = 0.0;
            }
        }
    }

    /// Reset all crossover filter states.
    ///
    /// Call between non-contiguous audio segments to avoid filter ringing
    /// from the previous segment bleeding into the next.
    pub fn reset(&mut self) {
        self.xover_low_l.reset();
        self.xover_low_r.reset();
        self.xover_high_l.reset();
        self.xover_high_r.reset();
    }

    /// Get the current low-band stereo width.
    #[must_use]
    pub fn low_width(&self) -> f32 {
        self.low_width
    }

    /// Get the current mid-band stereo width.
    #[must_use]
    pub fn mid_width(&self) -> f32 {
        self.mid_width
    }

    /// Get the current high-band stereo width.
    #[must_use]
    pub fn high_width(&self) -> f32 {
        self.high_width
    }
}

// ---------------------------------------------------------------------------
// Mid/side stereo width
// ---------------------------------------------------------------------------

/// Apply mid/side stereo width to a left/right pair.
///
/// - `width = 0.0`: mono (side is zeroed, only mid remains).
/// - `width = 1.0`: original stereo (no change).
/// - `width > 1.0`: hyper-stereo (side content amplified).
///
/// ```text
/// mid  = (L + R) * 0.5
/// side = (L - R) * 0.5 * width
/// L_out = mid + side
/// R_out = mid - side
/// ```
#[inline]
fn apply_midside_width(left: f32, right: f32, width: f32) -> (f32, f32) {
    let mid = (left + right) * 0.5;
    let side = (left - right) * 0.5 * width;
    (mid + side, mid - side)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 24000.0;

    // -- Config validation tests --

    #[test]
    fn test_config_default_is_valid() {
        MultibandStereoConfig::default()
            .validate()
            .expect("default config should be valid");
    }

    #[test]
    fn test_config_new_validates() {
        let cfg = MultibandStereoConfig::new(0.3, 1.0, 1.5, 200.0, 4000.0);
        assert!(cfg.is_ok());
    }

    #[test]
    fn test_config_rejects_low_width_out_of_range() {
        let cfg = MultibandStereoConfig::new(2.0, 1.0, 1.5, 200.0, 4000.0);
        assert!(cfg.is_err());
    }

    #[test]
    fn test_config_rejects_nan_mid_width() {
        let cfg = MultibandStereoConfig::new(0.3, f32::NAN, 1.5, 200.0, 4000.0);
        assert!(cfg.is_err());
    }

    #[test]
    fn test_config_rejects_crossover_inversion() {
        // low_crossover > high_crossover
        let cfg = MultibandStereoConfig::new(0.3, 1.0, 1.5, 5000.0, 3000.0);
        assert!(cfg.is_err());
    }

    #[test]
    fn test_config_rejects_inf_high_crossover() {
        let cfg = MultibandStereoConfig::new(0.3, 1.0, 1.5, 200.0, f32::INFINITY);
        assert!(cfg.is_err());
    }

    // -- Preset tests --

    #[test]
    fn test_all_presets_produce_valid_configs() {
        for preset in [
            MultibandStereoPreset::Vocal,
            MultibandStereoPreset::Cinematic,
            MultibandStereoPreset::Radio,
        ] {
            preset
                .to_config()
                .validate()
                .unwrap_or_else(|e| panic!("{preset:?} produced invalid config: {e}"));
        }
    }

    #[test]
    fn test_preset_vocal_values() {
        let cfg = MultibandStereoPreset::Vocal.to_config();
        assert!((cfg.low_width - 0.3).abs() < 1e-6);
        assert!((cfg.mid_width - 1.0).abs() < 1e-6);
        assert!((cfg.high_width - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_preset_radio_has_mono_bass() {
        let cfg = MultibandStereoPreset::Radio.to_config();
        assert!(
            (cfg.low_width - 0.0).abs() < 1e-6,
            "Radio bass should be mono"
        );
    }

    // -- Processor construction tests --

    #[test]
    fn test_processor_new_with_valid_config() {
        let cfg = MultibandStereoConfig::default();
        let proc = MultibandStereoProcessor::new(&cfg, SR);
        assert!(proc.is_ok());
    }

    #[test]
    fn test_processor_rejects_invalid_sample_rate() {
        let cfg = MultibandStereoConfig::default();
        assert!(MultibandStereoProcessor::new(&cfg, 0.0).is_err());
        assert!(MultibandStereoProcessor::new(&cfg, f32::NAN).is_err());
        assert!(MultibandStereoProcessor::new(&cfg, 500.0).is_err());
    }

    // -- Mid/side width tests --

    #[test]
    fn test_midside_width_zero_makes_mono() {
        let (l, r) = apply_midside_width(0.8, -0.3, 0.0);
        // Width 0 -> side zeroed -> L == R == mid.
        assert!(
            (l - r).abs() < 1e-7,
            "Width 0 should produce identical L and R (mono), got L={l}, R={r}"
        );
    }

    #[test]
    fn test_midside_width_one_preserves_signal() {
        let (l, r) = apply_midside_width(0.8, -0.3, 1.0);
        assert!((l - 0.8).abs() < 1e-6, "Width 1 should preserve L");
        assert!((r - (-0.3)).abs() < 1e-6, "Width 1 should preserve R");
    }

    #[test]
    fn test_midside_width_two_doubles_side() {
        let orig_l = 0.8;
        let orig_r = 0.2;
        let orig_side = (orig_l - orig_r) * 0.5; // 0.3
        let (l, r) = apply_midside_width(orig_l, orig_r, 2.0);
        let new_side = (l - r) * 0.5;
        assert!(
            (new_side - orig_side * 2.0).abs() < 1e-6,
            "Width 2 should double side content"
        );
    }

    // -- Processor behavior tests --

    #[test]
    fn test_low_width_zero_makes_bass_mono() {
        // Create a low-frequency sine (100 Hz) with stereo content.
        // With low_width=0 and high crossover frequencies set high,
        // the bass band should collapse to mono.
        let cfg = MultibandStereoConfig {
            low_width: 0.0,
            mid_width: 1.0,
            high_width: 1.0,
            low_crossover: 400.0, // Push crossover high to capture our 100Hz signal
            high_crossover: 5000.0,
            ..Default::default()
        };

        let mut proc = MultibandStereoProcessor::new(&cfg, SR).expect("valid config");

        let n = 2400; // 100ms at 24kHz — enough for filter settling
        let freq = 100.0;
        let mut left = Vec::with_capacity(n);
        let mut right = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f32 / SR;
            let phase = 2.0 * std::f32::consts::PI * freq * t;
            // Stereo content: L and R have different phases.
            left.push(phase.sin());
            right.push((phase + 0.5).sin());
        }

        proc.process(&mut left, &mut right);

        // After settling (skip first 50% for filter transient), check that
        // L and R converge toward each other in the bass band.
        let settle = n / 2;
        let mut max_diff: f32 = 0.0;
        for i in settle..n {
            let diff = (left[i] - right[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        // With low_width=0, the bass should be substantially narrower.
        // The LR2 crossover is not a brick wall: 100 Hz at a 400 Hz crossover
        // is only about 1 octave below the cutoff, so some signal leaks into
        // the mid band (which has width=1.0). Allow tolerance for this bleed.
        assert!(
            max_diff < 0.30,
            "Bass with low_width=0 should be near-mono, but max L-R diff = {max_diff}"
        );
    }

    #[test]
    fn test_high_width_widens_highs() {
        // High-frequency signal (8kHz) should be wider with high_width=2.0.
        let cfg = MultibandStereoConfig {
            low_width: 1.0,
            mid_width: 1.0,
            high_width: 2.0,
            low_crossover: 200.0,
            high_crossover: 3000.0, // Our 8kHz signal is above this
        };

        let mut proc = MultibandStereoProcessor::new(&cfg, SR).expect("valid config");

        let n = 2400;
        let freq = 8000.0;
        let mut left = Vec::with_capacity(n);
        let mut right = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f32 / SR;
            let phase = 2.0 * std::f32::consts::PI * freq * t;
            left.push(phase.sin());
            right.push((phase + 0.3).sin());
        }

        let orig_left = left.clone();
        let orig_right = right.clone();

        proc.process(&mut left, &mut right);

        // Measure stereo width as average |L-R| in the settled region.
        let settle = n / 2;
        let orig_width: f32 = (settle..n)
            .map(|i| (orig_left[i] - orig_right[i]).abs())
            .sum::<f32>()
            / (n - settle) as f32;
        let new_width: f32 =
            (settle..n).map(|i| (left[i] - right[i]).abs()).sum::<f32>() / (n - settle) as f32;

        assert!(
            new_width > orig_width * 1.3,
            "high_width=2.0 should widen high-freq stereo image: \
             original width={orig_width:.4}, processed width={new_width:.4}"
        );
    }

    #[test]
    fn test_unity_width_preserves_signal() {
        // All widths = 1.0 should be approximately pass-through
        // (within filter group delay).
        let cfg = MultibandStereoConfig {
            low_width: 1.0,
            mid_width: 1.0,
            high_width: 1.0,
            low_crossover: 200.0,
            high_crossover: 4000.0,
        };

        let mut proc = MultibandStereoProcessor::new(&cfg, SR).expect("valid config");

        let n = 4800;
        let mut left = Vec::with_capacity(n);
        let mut right = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f32 / SR;
            let phase = 2.0 * std::f32::consts::PI * 440.0 * t;
            left.push(phase.sin());
            right.push((phase * 1.5).sin());
        }

        let orig_left = left.clone();
        let orig_right = right.clone();

        proc.process(&mut left, &mut right);

        // After filter settling, energy should be preserved.
        let settle = n / 2;
        let orig_energy: f32 = (settle..n)
            .map(|i| orig_left[i].powi(2) + orig_right[i].powi(2))
            .sum();
        let new_energy: f32 = (settle..n)
            .map(|i| left[i].powi(2) + right[i].powi(2))
            .sum();

        let ratio = new_energy / orig_energy;
        assert!(
            (ratio - 1.0).abs() < 0.1,
            "Unity width should preserve energy: ratio = {ratio:.4}"
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = MultibandStereoConfig::default();
        let mut proc = MultibandStereoProcessor::new(&cfg, SR).expect("valid config");

        // Process some audio.
        let mut left = vec![1.0; 100];
        let mut right = vec![-1.0; 100];
        proc.process(&mut left, &mut right);

        // Reset and process silence — should produce silence.
        proc.reset();
        let mut left = vec![0.0; 100];
        let mut right = vec![0.0; 100];
        proc.process(&mut left, &mut right);

        let max_val = left
            .iter()
            .chain(right.iter())
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_val < 1e-10,
            "After reset + silence input, output should be silent, got max={max_val}"
        );
    }

    #[test]
    fn test_nan_input_produces_zero() {
        let cfg = MultibandStereoConfig::default();
        let mut proc = MultibandStereoProcessor::new(&cfg, SR).expect("valid config");

        let mut left = vec![f32::NAN, 0.5, f32::INFINITY];
        let mut right = vec![0.5, f32::NAN, -0.5];
        proc.process(&mut left, &mut right);

        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(
                l.is_finite() && r.is_finite(),
                "Sample {i}: non-finite output L={l}, R={r}"
            );
        }
    }

    #[test]
    fn test_builder_methods() {
        let cfg = MultibandStereoConfig::default()
            .with_low_width(0.1)
            .with_mid_width(0.8)
            .with_high_width(2.0)
            .with_low_crossover(100.0)
            .with_high_crossover(6000.0);

        assert!((cfg.low_width - 0.1).abs() < 1e-6);
        assert!((cfg.mid_width - 0.8).abs() < 1e-6);
        assert!((cfg.high_width - 2.0).abs() < 1e-6);
        assert!((cfg.low_crossover - 100.0).abs() < 1e-6);
        assert!((cfg.high_crossover - 6000.0).abs() < 1e-6);
        cfg.validate().expect("builder config should be valid");
    }

    #[test]
    fn test_accessor_methods() {
        let cfg = MultibandStereoConfig::new(0.2, 0.9, 1.8, 150.0, 5000.0).expect("valid config");
        let proc = MultibandStereoProcessor::new(&cfg, SR).expect("valid config");
        assert!((proc.low_width() - 0.2).abs() < 1e-6);
        assert!((proc.mid_width() - 0.9).abs() < 1e-6);
        assert!((proc.high_width() - 1.8).abs() < 1e-6);
    }

    #[test]
    fn test_crossover_splits_correctly() {
        // Verify that a pure low-frequency signal ends up mostly in the low
        // band by checking that low_width affects it but high_width does not.
        let n = 4800;
        let freq = 80.0; // Well below low_crossover of 300Hz

        // Config A: low_width=0 (mono), high_width=2.0
        let cfg_a = MultibandStereoConfig::new(0.0, 1.0, 2.0, 300.0, 4000.0).expect("valid");
        let mut proc_a = MultibandStereoProcessor::new(&cfg_a, SR).expect("valid");

        // Config B: low_width=1.0, high_width=0.5
        let cfg_b = MultibandStereoConfig::new(1.0, 1.0, 0.5, 300.0, 4000.0).expect("valid");
        let mut proc_b = MultibandStereoProcessor::new(&cfg_b, SR).expect("valid");

        let make_signal = || {
            let mut l = Vec::with_capacity(n);
            let mut r = Vec::with_capacity(n);
            for i in 0..n {
                let t = i as f32 / SR;
                let phase = 2.0 * std::f32::consts::PI * freq * t;
                l.push(phase.sin());
                r.push((phase + 0.7).sin());
            }
            (l, r)
        };

        let (mut la, mut ra) = make_signal();
        let (mut lb, mut rb) = make_signal();

        proc_a.process(&mut la, &mut ra);
        proc_b.process(&mut lb, &mut rb);

        // Measure stereo width after settling.
        let settle = n * 3 / 4;
        let width_a: f32 =
            (settle..n).map(|i| (la[i] - ra[i]).abs()).sum::<f32>() / (n - settle) as f32;
        let width_b: f32 =
            (settle..n).map(|i| (lb[i] - rb[i]).abs()).sum::<f32>() / (n - settle) as f32;

        // Config A has low_width=0 (mono bass), so width_a should be less
        // than config B which has low_width=1.0 (normal bass).
        assert!(
            width_a < width_b * 0.7,
            "80Hz signal should be affected by low_width, not high_width: \
             width_a(low=0)={width_a:.4}, width_b(low=1)={width_b:.4}"
        );
    }

    #[test]
    fn test_empty_buffers() {
        let cfg = MultibandStereoConfig::default();
        let mut proc = MultibandStereoProcessor::new(&cfg, SR).expect("valid config");
        let mut left: Vec<f32> = vec![];
        let mut right: Vec<f32> = vec![];
        proc.process(&mut left, &mut right);
        assert!(left.is_empty());
        assert!(right.is_empty());
    }

    #[test]
    fn test_mismatched_buffer_lengths() {
        let cfg = MultibandStereoConfig::default();
        let mut proc = MultibandStereoProcessor::new(&cfg, SR).expect("valid config");
        let mut left = vec![0.5; 100];
        let mut right = vec![-0.5; 50];
        // Should process min(100, 50) = 50 samples without panic.
        proc.process(&mut left, &mut right);
    }
}
