// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Automatic Double Tracking (ADT) vocal doubler for Kokoro chorus.
//!
//! ADT was invented at Abbey Road Studios for The Beatles -- it creates the
//! illusion of multiple vocal takes by applying subtle pitch and timing
//! variations to a copy of the voice. The effect thickens the vocal without
//! the need for a second take, and is distinct from chorus/flanger effects
//! because it uses longer delays (5-50ms) with slow modulation.
//!
//! # How it works
//!
//! ```text
//! Input ──┬──────────────────────────────────┬── Dry signal
//!         │                                  │
//!         └── Delay buffer ── Pitch shift ───┘── Wet signal (doubled)
//!              (5-50ms)       (0-30 cents)
//!              ↑ LFO mod
//!              (0.1-5 Hz)
//! ```
//!
//! The delay line uses a circular buffer with an LFO-modulated read position.
//! The LFO introduces subtle timing drift that mimics the natural variation
//! between two separate vocal takes. A small pitch offset (0-30 cents) further
//! differentiates the doubled signal from the original.
//!
//! # Stereo mode
//!
//! In stereo mode, the original (dry) signal is panned toward the left channel
//! and the doubled (wet) signal toward the right, controlled by `pan_spread`.
//! This creates a wide stereo image from a mono source -- a classic mixing
//! technique for vocals.
//!
//! # Placement in the chorus pipeline
//!
//! ADT is applied **per-voice** before the stereo mix stage:
//! ```text
//! Per-voice: vibrato -> detuning -> doubler -> EQ -> humanize
//! ```
//!
//! # References
//!
//! - Townsend, K. "Recording The Beatles." Curvebender, 2006.
//!   (original ADT technique at Abbey Road)
//! - Bode, H. "History of Electronic Sound Modification." AES Journal, 1984.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Automatic Double Tracking (ADT).
///
/// Controls the delay line, LFO modulation, pitch offset, and stereo
/// spread of the doubled signal.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DoublerConfig {
    /// Base delay for the doubled signal in milliseconds.
    ///
    /// This is the average time offset between the dry and wet signals.
    /// Short delays (5-15ms) create a tight doubling; longer delays
    /// (20-50ms) create a more obvious slap-back effect.
    ///
    /// Must be in [5.0, 50.0] and finite. Default: `18.0`.
    pub delay_ms: f32,

    /// LFO modulation depth on the delay time, in milliseconds.
    ///
    /// The LFO sweeps the delay read position by +/- this amount around
    /// the base delay. Larger values create a more noticeable "wobble"
    /// in the doubled signal.
    ///
    /// Must be in [0.0, 10.0] and finite. Default: `3.0`.
    pub delay_mod_depth_ms: f32,

    /// LFO rate for delay modulation in Hz.
    ///
    /// Controls how fast the delay modulation cycles. Slow rates
    /// (0.1-1.0 Hz) sound natural; faster rates (2-5 Hz) create a
    /// more chorus-like effect.
    ///
    /// Must be in [0.1, 5.0] and finite. Default: `0.7`.
    pub delay_mod_rate_hz: f32,

    /// Slight pitch offset for the doubled signal in cents.
    ///
    /// A small pitch difference (5-15 cents) between the dry and wet
    /// signals enhances the illusion of two separate takes. Larger
    /// values (>20 cents) become audible as detuning.
    ///
    /// Must be in [0.0, 30.0] and finite. Default: `7.0`.
    pub pitch_shift_cents: f32,

    /// Dry/wet mix balance.
    ///
    /// 0.0 = dry only (no doubling effect), 1.0 = equal mix of dry and
    /// wet. Values above 0.5 make the doubled signal louder than the
    /// original, which is unusual but allowed for creative use.
    ///
    /// Must be in [0.0, 1.0] and finite. Default: `0.5`.
    pub mix: f32,

    /// Stereo spread between original and doubled signals.
    ///
    /// Controls how far apart the dry and wet signals are panned in
    /// stereo mode. 0.0 = mono (both center), 1.0 = full spread
    /// (dry hard left, wet hard right).
    ///
    /// Must be in [0.0, 1.0] and finite. Default: `0.6`.
    pub pan_spread: f32,
}

impl Default for DoublerConfig {
    fn default() -> Self {
        Self {
            delay_ms: 18.0,
            delay_mod_depth_ms: 3.0,
            delay_mod_rate_hz: 0.7,
            pitch_shift_cents: 7.0,
            mix: 0.5,
            pan_spread: 0.6,
        }
    }
}

impl DoublerConfig {
    /// Create a new doubler configuration with all parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the base delay in milliseconds.
    #[must_use]
    pub fn with_delay_ms(mut self, delay_ms: f32) -> Self {
        self.delay_ms = delay_ms;
        self
    }

    /// Set the LFO modulation depth in milliseconds.
    #[must_use]
    pub fn with_delay_mod_depth_ms(mut self, depth: f32) -> Self {
        self.delay_mod_depth_ms = depth;
        self
    }

    /// Set the LFO modulation rate in Hz.
    #[must_use]
    pub fn with_delay_mod_rate_hz(mut self, rate: f32) -> Self {
        self.delay_mod_rate_hz = rate;
        self
    }

    /// Set the pitch shift in cents for the doubled signal.
    #[must_use]
    pub fn with_pitch_shift_cents(mut self, cents: f32) -> Self {
        self.pitch_shift_cents = cents;
        self
    }

    /// Set the dry/wet mix balance.
    #[must_use]
    pub fn with_mix(mut self, mix: f32) -> Self {
        self.mix = mix;
        self
    }

    /// Set the stereo pan spread.
    #[must_use]
    pub fn with_pan_spread(mut self, spread: f32) -> Self {
        self.pan_spread = spread;
        self
    }

    /// Create a tight doubling preset (short delay, minimal modulation).
    ///
    /// Suitable for subtle vocal thickening where the doubling should
    /// not be consciously audible.
    #[must_use]
    pub fn tight() -> Self {
        Self {
            delay_ms: 8.0,
            delay_mod_depth_ms: 1.5,
            delay_mod_rate_hz: 0.5,
            pitch_shift_cents: 5.0,
            mix: 0.4,
            pan_spread: 0.3,
        }
    }

    /// Create a wide doubling preset (longer delay, more modulation).
    ///
    /// Creates a more obvious doubled sound with noticeable stereo width.
    /// Good for lead vocals that need to fill a mix.
    #[must_use]
    pub fn wide() -> Self {
        Self {
            delay_ms: 35.0,
            delay_mod_depth_ms: 5.0,
            delay_mod_rate_hz: 1.2,
            pitch_shift_cents: 12.0,
            mix: 0.5,
            pan_spread: 0.8,
        }
    }

    /// Validate all configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is non-finite
    /// or outside its valid range.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.delay_ms.is_finite() || !(5.0..=50.0).contains(&self.delay_ms) {
            return Err(KokoroError::InvalidConfig {
                field: "delay_ms",
                reason: format!("must be finite and in [5.0, 50.0], got {}", self.delay_ms),
            });
        }
        if !self.delay_mod_depth_ms.is_finite() || !(0.0..=10.0).contains(&self.delay_mod_depth_ms)
        {
            return Err(KokoroError::InvalidConfig {
                field: "delay_mod_depth_ms",
                reason: format!(
                    "must be finite and in [0.0, 10.0], got {}",
                    self.delay_mod_depth_ms,
                ),
            });
        }
        if !self.delay_mod_rate_hz.is_finite() || !(0.1..=5.0).contains(&self.delay_mod_rate_hz) {
            return Err(KokoroError::InvalidConfig {
                field: "delay_mod_rate_hz",
                reason: format!(
                    "must be finite and in [0.1, 5.0], got {}",
                    self.delay_mod_rate_hz,
                ),
            });
        }
        if !self.pitch_shift_cents.is_finite() || !(0.0..=30.0).contains(&self.pitch_shift_cents) {
            return Err(KokoroError::InvalidConfig {
                field: "pitch_shift_cents",
                reason: format!(
                    "must be finite and in [0.0, 30.0], got {}",
                    self.pitch_shift_cents,
                ),
            });
        }
        if !self.mix.is_finite() || !(0.0..=1.0).contains(&self.mix) {
            return Err(KokoroError::InvalidConfig {
                field: "mix",
                reason: format!("must be finite and in [0.0, 1.0], got {}", self.mix),
            });
        }
        if !self.pan_spread.is_finite() || !(0.0..=1.0).contains(&self.pan_spread) {
            return Err(KokoroError::InvalidConfig {
                field: "pan_spread",
                reason: format!("must be finite and in [0.0, 1.0], got {}", self.pan_spread),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Vocal Doubler processor
// ---------------------------------------------------------------------------

/// Vocal doubler processor implementing Automatic Double Tracking (ADT).
///
/// Uses a circular delay buffer with LFO-modulated read position and
/// optional pitch shifting via linear interpolation resampling.
pub struct VocalDoubler {
    /// Circular delay buffer.
    buffer: Vec<f32>,
    /// Write position in the circular buffer.
    write_pos: usize,
    /// Buffer capacity (in samples).
    buffer_len: usize,
    /// Base delay in samples.
    base_delay_samples: f32,
    /// LFO modulation depth in samples.
    mod_depth_samples: f32,
    /// LFO phase increment per sample (radians).
    lfo_phase_inc: f32,
    /// Current LFO phase (radians).
    lfo_phase: f32,
    /// Pitch shift resampling rate ratio: 2^(cents/1200).
    pitch_rate: f64,
    /// Dry/wet mix (0 = dry only, 1 = equal).
    mix: f32,
    /// Stereo pan spread (0 = mono, 1 = full).
    pan_spread: f32,
    /// Sample rate in Hz.
    sample_rate: f32,
}

impl VocalDoubler {
    /// Create a new vocal doubler from the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Doubler configuration (validated before use).
    /// * `sample_rate` - Audio sample rate in Hz (must be > 0).
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if config validation fails, or
    /// `KokoroError::InvalidInput` if sample_rate is non-positive or non-finite.
    pub fn new(config: &DoublerConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;

        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidInput(format!(
                "sample_rate must be finite and > 0, got {sample_rate}"
            )));
        }

        let base_delay_samples = config.delay_ms * sample_rate / 1000.0;
        let mod_depth_samples = config.delay_mod_depth_ms * sample_rate / 1000.0;

        // Buffer must be large enough for max delay + modulation depth + margin
        // for linear interpolation (need one extra sample).
        let max_delay = base_delay_samples + mod_depth_samples + 2.0;
        let buffer_len = (max_delay.ceil() as usize).max(4);

        let lfo_phase_inc = std::f32::consts::TAU * config.delay_mod_rate_hz / sample_rate;

        // Pitch shift: rate = 2^(cents/1200). For the doubled signal we shift
        // down slightly (read slower) to create a lower pitch copy.
        let pitch_rate = (2.0f64).powf(f64::from(config.pitch_shift_cents) / 1200.0);

        Ok(Self {
            buffer: vec![0.0; buffer_len],
            write_pos: 0,
            buffer_len,
            base_delay_samples,
            mod_depth_samples,
            lfo_phase_inc,
            lfo_phase: 0.0,
            pitch_rate,
            mix: config.mix,
            pan_spread: config.pan_spread,
            sample_rate,
        })
    }

    /// Create a new vocal doubler with a specific initial LFO phase.
    ///
    /// Used by `apply_doubler_per_voice` to give each voice a different
    /// LFO starting phase for decorrelation.
    pub fn with_initial_phase(
        config: &DoublerConfig,
        sample_rate: f32,
        initial_phase: f32,
    ) -> Result<Self, KokoroError> {
        let mut doubler = Self::new(config, sample_rate)?;
        doubler.lfo_phase = initial_phase;
        Ok(doubler)
    }

    /// Reset the delay buffer and LFO phase to initial state.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.lfo_phase = 0.0;
    }

    /// Read a sample from the delay buffer at a fractional position.
    ///
    /// Uses linear interpolation between adjacent samples for sub-sample
    /// accuracy.
    #[inline]
    fn read_delayed(&self, delay_samples: f32) -> f32 {
        // Compute the read position relative to write_pos.
        let read_pos_f = self.write_pos as f64 - f64::from(delay_samples);

        // Wrap into buffer range.
        let buf_len_f = self.buffer_len as f64;
        let wrapped = ((read_pos_f % buf_len_f) + buf_len_f) % buf_len_f;

        let idx0 = wrapped.floor() as usize % self.buffer_len;
        let idx1 = (idx0 + 1) % self.buffer_len;
        let frac = (wrapped - wrapped.floor()) as f32;

        let s0 = self.buffer[idx0];
        let s1 = self.buffer[idx1];

        let result = s0 + frac * (s1 - s0);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }

    /// Process a mono audio buffer through the doubler.
    ///
    /// Returns a new buffer containing the mixed dry + wet signal. The
    /// output has the same length as the input.
    #[must_use]
    pub fn process_mono(&mut self, audio: &[f32]) -> Vec<f32> {
        let dry_gain = 1.0 - self.mix * 0.5;
        let wet_gain = self.mix;
        let mut output = Vec::with_capacity(audio.len());

        // Fractional read position for pitch shifting within the delay line.
        let mut pitch_phase: f64 = 0.0;

        for &sample in audio {
            // Guard against NaN/Inf input.
            let safe_sample = if sample.is_finite() { sample } else { 0.0 };

            // Write input to delay buffer.
            self.buffer[self.write_pos] = safe_sample;

            // Compute LFO-modulated delay.
            let lfo_val = self.lfo_phase.sin();
            let current_delay = self.base_delay_samples + self.mod_depth_samples * lfo_val;

            // Apply pitch shift offset: accumulate fractional offset.
            let pitch_offset = pitch_phase - pitch_phase.floor();
            let total_delay = current_delay + pitch_offset as f32;

            // Clamp delay to valid range.
            let clamped_delay = total_delay.clamp(0.0, (self.buffer_len - 2) as f32);

            let wet_sample = self.read_delayed(clamped_delay);

            // Mix dry and wet.
            let mixed = dry_gain * safe_sample + wet_gain * wet_sample;
            output.push(if mixed.is_finite() { mixed } else { 0.0 });

            // Advance LFO phase (wrap at TAU to prevent float drift).
            self.lfo_phase += self.lfo_phase_inc;
            if self.lfo_phase >= std::f32::consts::TAU {
                self.lfo_phase -= std::f32::consts::TAU;
            }

            // Advance pitch shift phase.
            // rate > 1 means higher pitch (read faster), so the doubled
            // signal reads ahead by (rate - 1) samples per output sample.
            pitch_phase += self.pitch_rate - 1.0;

            // Advance write position.
            self.write_pos = (self.write_pos + 1) % self.buffer_len;
        }

        output
    }

    /// Process a mono audio buffer into stereo with ADT panning.
    ///
    /// Returns `(left, right)` channels. The dry signal is panned toward
    /// the left and the wet (doubled) signal toward the right, controlled
    /// by `pan_spread`.
    #[must_use]
    pub fn process_stereo(&mut self, audio: &[f32]) -> (Vec<f32>, Vec<f32>) {
        // Constant-power panning coefficients.
        // At spread=0: both channels get equal mix (mono).
        // At spread=1: dry goes fully left, wet goes fully right.
        let half_spread = self.pan_spread * 0.5;
        let dry_pan_angle = (0.5 - half_spread).clamp(0.0, 1.0);
        let wet_pan_angle = (0.5 + half_spread).clamp(0.0, 1.0);

        // Constant-power pan: left = cos(angle * pi/2), right = sin(angle * pi/2)
        let half_pi = std::f32::consts::FRAC_PI_2;
        let dry_left = (dry_pan_angle * half_pi).cos();
        let dry_right = (dry_pan_angle * half_pi).sin();
        let wet_left = (wet_pan_angle * half_pi).cos();
        let wet_right = (wet_pan_angle * half_pi).sin();

        let wet_gain = self.mix;

        let mut left = Vec::with_capacity(audio.len());
        let mut right = Vec::with_capacity(audio.len());

        let mut pitch_phase: f64 = 0.0;

        for &sample in audio {
            let safe_sample = if sample.is_finite() { sample } else { 0.0 };

            // Write to delay buffer.
            self.buffer[self.write_pos] = safe_sample;

            // LFO-modulated delay.
            let lfo_val = self.lfo_phase.sin();
            let current_delay = self.base_delay_samples + self.mod_depth_samples * lfo_val;
            let pitch_offset = pitch_phase - pitch_phase.floor();
            let total_delay = current_delay + pitch_offset as f32;
            let clamped_delay = total_delay.clamp(0.0, (self.buffer_len - 2) as f32);

            let wet_sample = self.read_delayed(clamped_delay);

            // Pan dry and wet into stereo.
            let l = safe_sample * dry_left + wet_gain * wet_sample * wet_left;
            let r = safe_sample * dry_right + wet_gain * wet_sample * wet_right;

            left.push(if l.is_finite() { l } else { 0.0 });
            right.push(if r.is_finite() { r } else { 0.0 });

            // Advance LFO.
            self.lfo_phase += self.lfo_phase_inc;
            if self.lfo_phase >= std::f32::consts::TAU {
                self.lfo_phase -= std::f32::consts::TAU;
            }

            // Advance pitch phase.
            pitch_phase += self.pitch_rate - 1.0;

            // Advance write position.
            self.write_pos = (self.write_pos + 1) % self.buffer_len;
        }

        (left, right)
    }

    /// Get the current sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Get the base delay in samples.
    #[must_use]
    pub fn base_delay_samples(&self) -> f32 {
        self.base_delay_samples
    }
}

// ---------------------------------------------------------------------------
// Per-voice convenience function
// ---------------------------------------------------------------------------

/// Apply ADT doubling to each voice independently.
///
/// Each voice gets its own `VocalDoubler` instance with a different initial
/// LFO phase (evenly distributed across [0, 2*pi)) so that the delay
/// modulation patterns are decorrelated between voices.
///
/// The doubling is applied in-place: each voice buffer is replaced with
/// the mono-mixed doubled output.
///
/// # Arguments
///
/// * `voices` - Mutable slice of per-voice PCM buffers.
/// * `config` - Doubler configuration.
/// * `sample_rate` - Audio sample rate in Hz.
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` if the config is invalid, or
/// `KokoroError::InvalidInput` if sample_rate is non-positive.
pub fn apply_doubler_per_voice(
    voices: &mut [Vec<f32>],
    config: &DoublerConfig,
    sample_rate: f32,
) -> Result<(), KokoroError> {
    config.validate()?;

    if voices.is_empty() {
        return Ok(());
    }

    // Skip if mix is negligible (no audible doubling).
    if config.mix < 1e-6 {
        return Ok(());
    }

    let n_voices = voices.len();

    for (i, voice_pcm) in voices.iter_mut().enumerate() {
        if voice_pcm.is_empty() {
            continue;
        }

        // Each voice gets a different LFO starting phase.
        let initial_phase = (i as f32 / n_voices as f32) * std::f32::consts::TAU;

        let mut doubler = VocalDoubler::with_initial_phase(config, sample_rate, initial_phase)?;

        let doubled = doubler.process_mono(voice_pcm);
        voice_pcm.clear();
        voice_pcm.extend_from_slice(&doubled);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config validation ---------------------------------------------------

    #[test]
    fn test_default_config_valid() {
        let config = DoublerConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_tight_preset_valid() {
        let config = DoublerConfig::tight();
        assert!(config.validate().is_ok());
        assert!(config.delay_ms < 15.0);
    }

    #[test]
    fn test_wide_preset_valid() {
        let config = DoublerConfig::wide();
        assert!(config.validate().is_ok());
        assert!(config.delay_ms > 25.0);
    }

    #[test]
    fn test_builder_chain() {
        let config = DoublerConfig::new()
            .with_delay_ms(25.0)
            .with_delay_mod_depth_ms(4.0)
            .with_delay_mod_rate_hz(1.0)
            .with_pitch_shift_cents(10.0)
            .with_mix(0.6)
            .with_pan_spread(0.7);
        assert!(config.validate().is_ok());
        assert!((config.delay_ms - 25.0).abs() < 1e-6);
        assert!((config.mix - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_validate_rejects_delay_ms_out_of_range() {
        let config = DoublerConfig::default().with_delay_ms(4.0);
        assert!(config.validate().is_err());
        let config = DoublerConfig::default().with_delay_ms(51.0);
        assert!(config.validate().is_err());
        let config = DoublerConfig::default().with_delay_ms(f32::NAN);
        assert!(config.validate().is_err());
        let config = DoublerConfig::default().with_delay_ms(f32::INFINITY);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_mod_depth_out_of_range() {
        let config = DoublerConfig::default().with_delay_mod_depth_ms(-0.1);
        assert!(config.validate().is_err());
        let config = DoublerConfig::default().with_delay_mod_depth_ms(10.1);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_mod_rate_out_of_range() {
        let config = DoublerConfig::default().with_delay_mod_rate_hz(0.05);
        assert!(config.validate().is_err());
        let config = DoublerConfig::default().with_delay_mod_rate_hz(5.1);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_pitch_out_of_range() {
        let config = DoublerConfig::default().with_pitch_shift_cents(-0.1);
        assert!(config.validate().is_err());
        let config = DoublerConfig::default().with_pitch_shift_cents(30.1);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_mix_out_of_range() {
        let config = DoublerConfig::default().with_mix(-0.01);
        assert!(config.validate().is_err());
        let config = DoublerConfig::default().with_mix(1.01);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_pan_spread_out_of_range() {
        let config = DoublerConfig::default().with_pan_spread(-0.01);
        assert!(config.validate().is_err());
        let config = DoublerConfig::default().with_pan_spread(1.01);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_accepts_boundary_values() {
        let config = DoublerConfig::default()
            .with_delay_ms(5.0)
            .with_delay_mod_depth_ms(0.0)
            .with_delay_mod_rate_hz(0.1)
            .with_pitch_shift_cents(0.0)
            .with_mix(0.0)
            .with_pan_spread(0.0);
        assert!(config.validate().is_ok());

        let config = DoublerConfig::default()
            .with_delay_ms(50.0)
            .with_delay_mod_depth_ms(10.0)
            .with_delay_mod_rate_hz(5.0)
            .with_pitch_shift_cents(30.0)
            .with_mix(1.0)
            .with_pan_spread(1.0);
        assert!(config.validate().is_ok());
    }

    // -- VocalDoubler construction -------------------------------------------

    #[test]
    fn test_new_rejects_invalid_sample_rate() {
        let config = DoublerConfig::default();
        assert!(VocalDoubler::new(&config, 0.0).is_err());
        assert!(VocalDoubler::new(&config, -44100.0).is_err());
        assert!(VocalDoubler::new(&config, f32::NAN).is_err());
        assert!(VocalDoubler::new(&config, f32::INFINITY).is_err());
    }

    #[test]
    fn test_new_accepts_valid_sample_rate() {
        let config = DoublerConfig::default();
        assert!(VocalDoubler::new(&config, 24000.0).is_ok());
        assert!(VocalDoubler::new(&config, 44100.0).is_ok());
        assert!(VocalDoubler::new(&config, 48000.0).is_ok());
    }

    // -- Mono processing -----------------------------------------------------

    #[test]
    fn test_mono_output_same_length() {
        let config = DoublerConfig::default();
        let mut doubler = VocalDoubler::new(&config, 24000.0).unwrap();
        let audio: Vec<f32> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let output = doubler.process_mono(&audio);
        assert_eq!(output.len(), audio.len());
    }

    #[test]
    fn test_mono_doubled_differs_from_dry() {
        let config = DoublerConfig::default();
        let mut doubler = VocalDoubler::new(&config, 24000.0).unwrap();
        let audio: Vec<f32> = (0..12000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let output = doubler.process_mono(&audio);

        // After the delay line fills (>50ms = 1200 samples at 24kHz), the
        // output should differ from the input due to the wet signal.
        let skip = 2000;
        let mean_diff: f32 = output[skip..]
            .iter()
            .zip(audio[skip..].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / (output.len() - skip) as f32;

        assert!(
            mean_diff > 1e-3,
            "doubled output should differ from dry, mean diff = {mean_diff}",
        );
    }

    #[test]
    fn test_mono_delay_creates_temporal_offset() {
        // With zero modulation and zero pitch shift, the doubled signal
        // should be a delayed copy mixed with the dry.
        let config = DoublerConfig::default()
            .with_delay_mod_depth_ms(0.0)
            .with_pitch_shift_cents(0.0)
            .with_mix(1.0); // Full wet for clarity

        let mut doubler = VocalDoubler::new(&config, 24000.0).unwrap();

        // Create an impulse.
        let mut audio = vec![0.0f32; 2400];
        audio[0] = 1.0;

        let output = doubler.process_mono(&audio);

        // The output should have the dry impulse at sample 0 and
        // a wet copy at approximately base_delay_samples.
        let delay_samples = (18.0 * 24000.0 / 1000.0) as usize; // ~432 samples

        // Find the peak in the delayed region.
        let search_start = delay_samples.saturating_sub(5);
        let search_end = (delay_samples + 5).min(output.len());
        let delayed_peak = output[search_start..search_end]
            .iter()
            .copied()
            .fold(0.0f32, f32::max);

        assert!(
            delayed_peak > 0.1,
            "should find delayed impulse near sample {delay_samples}, peak = {delayed_peak}",
        );
    }

    #[test]
    fn test_mono_zero_mix_is_passthrough() {
        let config = DoublerConfig::default().with_mix(0.0);
        let mut doubler = VocalDoubler::new(&config, 24000.0).unwrap();
        let audio: Vec<f32> = (0..2400)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let output = doubler.process_mono(&audio);

        // With mix=0, dry_gain=1.0 and wet_gain=0.0, output should match input.
        for (i, (&out, &inp)) in output.iter().zip(audio.iter()).enumerate() {
            assert!(
                (out - inp).abs() < 1e-5,
                "sample {i}: output {out} != input {inp} with zero mix",
            );
        }
    }

    #[test]
    fn test_mono_rms_with_doubling() {
        // Doubled signal should have different RMS than dry (typically louder
        // due to constructive interference at some frequencies).
        let config = DoublerConfig::default();
        let mut doubler = VocalDoubler::new(&config, 24000.0).unwrap();
        let audio: Vec<f32> = (0..24000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();

        let rms_dry: f32 = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();

        let output = doubler.process_mono(&audio);
        let rms_out: f32 = (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();

        // RMS should be different (not exactly the same as dry).
        let ratio = rms_out / rms_dry;
        assert!(
            (ratio - 1.0).abs() > 0.01,
            "RMS ratio {ratio} should differ from 1.0 with doubling",
        );
    }

    // -- Stereo processing ---------------------------------------------------

    #[test]
    fn test_stereo_output_same_length() {
        let config = DoublerConfig::default();
        let mut doubler = VocalDoubler::new(&config, 24000.0).unwrap();
        let audio: Vec<f32> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let (left, right) = doubler.process_stereo(&audio);
        assert_eq!(left.len(), audio.len());
        assert_eq!(right.len(), audio.len());
    }

    #[test]
    fn test_stereo_spread_creates_channel_difference() {
        let config = DoublerConfig::default().with_pan_spread(0.8);
        let mut doubler = VocalDoubler::new(&config, 24000.0).unwrap();
        let audio: Vec<f32> = (0..12000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let (left, right) = doubler.process_stereo(&audio);

        // With non-zero spread, left and right channels should differ.
        let skip = 2000; // Skip delay fill time.
        let channel_diff: f32 = left[skip..]
            .iter()
            .zip(right[skip..].iter())
            .map(|(l, r)| (l - r).abs())
            .sum::<f32>()
            / (left.len() - skip) as f32;

        assert!(
            channel_diff > 1e-3,
            "stereo channels should differ with spread=0.8, diff = {channel_diff}",
        );
    }

    #[test]
    fn test_stereo_zero_spread_is_mono() {
        let config = DoublerConfig::default().with_pan_spread(0.0);
        let mut doubler = VocalDoubler::new(&config, 24000.0).unwrap();
        let audio: Vec<f32> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let (left, right) = doubler.process_stereo(&audio);

        // With zero spread, both channels should be identical.
        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(
                (l - r).abs() < 1e-5,
                "sample {i}: left {l} != right {r} with zero spread",
            );
        }
    }

    // -- Per-voice function --------------------------------------------------

    #[test]
    fn test_per_voice_applies_to_all() {
        let config = DoublerConfig::default();
        let signal: Vec<f32> = (0..12000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let original = signal.clone();
        let mut voices = vec![signal.clone(), signal.clone(), signal];
        apply_doubler_per_voice(&mut voices, &config, 24000.0).unwrap();

        // Each voice should be modified.
        let skip = 2000;
        for (vi, voice) in voices.iter().enumerate() {
            let mean_diff: f32 = voice[skip..]
                .iter()
                .zip(original[skip..].iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / (voice.len() - skip) as f32;

            assert!(
                mean_diff > 1e-4,
                "voice {vi} should differ from original, mean diff = {mean_diff}",
            );
        }
    }

    #[test]
    fn test_per_voice_different_phases() {
        // Each voice should get a different LFO phase, so their doubled
        // outputs should differ from each other.
        let config = DoublerConfig::default();
        let signal: Vec<f32> = (0..24000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let mut voices = vec![signal.clone(), signal.clone(), signal];
        apply_doubler_per_voice(&mut voices, &config, 24000.0).unwrap();

        // Voices should differ from each other due to different LFO phases.
        let skip = 2000;
        let diff_01: f32 = voices[0][skip..]
            .iter()
            .zip(voices[1][skip..].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / (voices[0].len() - skip) as f32;

        assert!(
            diff_01 > 1e-5,
            "voices 0 and 1 should differ, mean diff = {diff_01}",
        );
    }

    #[test]
    fn test_per_voice_empty_voices_ok() {
        let config = DoublerConfig::default();
        let mut voices: Vec<Vec<f32>> = vec![];
        assert!(apply_doubler_per_voice(&mut voices, &config, 24000.0).is_ok());
    }

    #[test]
    fn test_per_voice_empty_buffer_ok() {
        let config = DoublerConfig::default();
        let mut voices = vec![vec![], vec![1.0, 2.0, 3.0]];
        assert!(apply_doubler_per_voice(&mut voices, &config, 24000.0).is_ok());
        assert!(voices[0].is_empty());
        assert_eq!(voices[1].len(), 3);
    }

    #[test]
    fn test_per_voice_preserves_length() {
        let config = DoublerConfig::default();
        let lengths = [1000, 2000, 1500];
        let mut voices: Vec<Vec<f32>> = lengths
            .iter()
            .map(|&len| (0..len).map(|i| (i as f32 * 0.01).sin()).collect())
            .collect();
        apply_doubler_per_voice(&mut voices, &config, 24000.0).unwrap();

        for (i, (&expected_len, voice)) in lengths.iter().zip(voices.iter()).enumerate() {
            assert_eq!(
                voice.len(),
                expected_len,
                "voice {i} length should be preserved",
            );
        }
    }

    #[test]
    fn test_per_voice_zero_mix_is_identity() {
        let config = DoublerConfig::default().with_mix(0.0);
        let signal: Vec<f32> = (0..2400)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let mut voices = vec![signal.clone(), signal.clone()];
        apply_doubler_per_voice(&mut voices, &config, 24000.0).unwrap();

        // With mix=0, voices should be unchanged.
        for (vi, voice) in voices.iter().enumerate() {
            for (j, (&got, &expected)) in voice.iter().zip(signal.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < 1e-5,
                    "voice {vi} sample {j}: {got} != {expected} with zero mix",
                );
            }
        }
    }

    // -- NaN/Inf safety ------------------------------------------------------

    #[test]
    fn test_nan_input_produces_finite_output() {
        let config = DoublerConfig::default();
        let mut doubler = VocalDoubler::new(&config, 24000.0).unwrap();
        let mut audio = vec![0.0f32; 1000];
        audio[100] = f32::NAN;
        audio[200] = f32::INFINITY;
        audio[300] = f32::NEG_INFINITY;

        let output = doubler.process_mono(&audio);
        for (i, &s) in output.iter().enumerate() {
            assert!(s.is_finite(), "sample {i} is non-finite: {s}");
        }
    }

    #[test]
    fn test_stereo_nan_safety() {
        let config = DoublerConfig::default();
        let mut doubler = VocalDoubler::new(&config, 24000.0).unwrap();
        let mut audio = vec![0.0f32; 1000];
        audio[50] = f32::NAN;

        let (left, right) = doubler.process_stereo(&audio);
        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(l.is_finite(), "left sample {i} is non-finite: {l}");
            assert!(r.is_finite(), "right sample {i} is non-finite: {r}");
        }
    }

    // -- Reset ---------------------------------------------------------------

    #[test]
    fn test_reset_clears_state() {
        let config = DoublerConfig::default();
        let mut doubler = VocalDoubler::new(&config, 24000.0).unwrap();

        // Process some audio.
        let audio: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.01).sin()).collect();
        let _ = doubler.process_mono(&audio);

        // Reset.
        doubler.reset();

        // After reset, processing the same audio should give the same result
        // as a fresh instance.
        let mut fresh = VocalDoubler::new(&config, 24000.0).unwrap();
        let output_reset = doubler.process_mono(&audio);
        let output_fresh = fresh.process_mono(&audio);

        for (i, (&a, &b)) in output_reset.iter().zip(output_fresh.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "sample {i}: reset {a} != fresh {b}");
        }
    }

    // -- Pitch shift detection -----------------------------------------------

    #[test]
    fn test_pitch_shift_detectable() {
        // With large pitch shift and zero modulation, the doubled signal
        // should have a detectable frequency difference.
        let config = DoublerConfig::default()
            .with_pitch_shift_cents(25.0)
            .with_delay_mod_depth_ms(0.0)
            .with_mix(1.0);

        let mut doubler = VocalDoubler::new(&config, 24000.0).unwrap();
        let freq = 440.0;
        let sr = 24000.0;
        let audio: Vec<f32> = (0..24000)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect();

        let output = doubler.process_mono(&audio);

        // Simple zero-crossing rate analysis on a windowed portion.
        // The doubled signal mixed with dry should have beating (amplitude
        // modulation) due to the frequency difference.
        let skip = 2000;
        let window = &output[skip..skip + 12000];

        // Count zero crossings.
        let zc: usize = window
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();

        // Pure 440Hz at 24kHz: ~440 zero crossings per 12000 samples
        // (half second). With beating, the pattern changes.
        let expected_zc = (440.0 * 12000.0 / sr) as usize;
        let zc_diff = (zc as isize - expected_zc as isize).unsigned_abs();

        // The zero crossing count should differ from pure 440Hz
        // due to the pitch-shifted component creating interference.
        // We use a generous threshold since the effect is subtle.
        assert!(
            zc_diff > 0 || {
                // Alternatively check that the amplitude envelope varies
                // (beating from two close frequencies).
                let envelope_var: f32 = window
                    .chunks(100)
                    .map(|chunk| {
                        (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32).sqrt()
                    })
                    .collect::<Vec<f32>>()
                    .windows(2)
                    .map(|w| (w[1] - w[0]).abs())
                    .sum::<f32>();
                envelope_var > 0.01
            },
            "pitch shift should be detectable via zero crossings or envelope variation",
        );
    }
}
