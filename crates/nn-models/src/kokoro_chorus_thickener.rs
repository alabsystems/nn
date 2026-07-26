// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-voice micro-modulation thickener for Kokoro chorus.
//!
//! Adds subtle, LFO-driven variations in pitch, timing, and amplitude to
//! each voice in a chorus without adding new voices. The result is a thicker,
//! more lush sound that mimics the micro-imprecisions of real human singers.
//!
//! # Architecture
//!
//! ```text
//! Voice[i] ─┬─ Pitch modulation (±5-15 cents via interpolated delay line)
//!            ├─ Timing modulation (±1-3 ms via modulated delay offset)
//!            ├─ Amplitude modulation (±0.5-2 dB via smooth gain envelope)
//!            └─ Chorus delay (modulated delay line for extra width)
//!
//! Each dimension has an LFO with a per-voice phase offset derived from
//! voice index, creating natural decorrelation between voices.
//! ```
//!
//! # References
//!
//! - Dattorro, "Effect Design Part 2", J. Audio Eng. Soc., 45(10), 1997.
//! - De Sena et al., "Efficient Synthesis of Room Acoustics via Scattering
//!   Delay Networks", IEEE/ACM Trans. Audio Speech Lang. Process., 2015.
//!   (modulated delay line interpolation)

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// LFO shape
// ---------------------------------------------------------------------------

/// Low-frequency oscillator waveform shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum LfoShape {
    /// Smooth sinusoidal modulation.
    #[default]
    Sine,
    /// Softer-cornered triangle modulation for slower, more linear sweeps.
    Triangle,
}


// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the chorus voice thickener.
///
/// Controls per-voice micro-modulation of pitch, timing, and amplitude,
/// plus an optional chorus-style modulated delay for extra width.
///
/// Use builder methods or preset constructors. `#[non_exhaustive]` allows
/// future field additions without breaking downstream code.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ThickenerConfig {
    /// Pitch modulation depth in cents (1 cent = 1/100 semitone).
    /// Range: [0.0, 30.0]. Default: 8.0.
    pub pitch_depth_cents: f32,
    /// Timing modulation depth in milliseconds.
    /// Range: [0.0, 10.0]. Default: 1.5.
    pub time_depth_ms: f32,
    /// Amplitude modulation depth in decibels.
    /// Range: [0.0, 6.0]. Default: 1.0.
    pub amplitude_depth_db: f32,
    /// LFO rate in Hz (shared base rate; per-voice offsets are derived).
    /// Range: [0.01, 10.0]. Default: 0.5.
    pub lfo_rate_hz: f32,
    /// LFO waveform shape. Default: Sine.
    pub lfo_shape: LfoShape,
    /// Chorus delay base time in milliseconds.
    /// Range: [0.0, 50.0]. Default: 15.0.
    pub chorus_delay_ms: f32,
    /// Chorus delay modulation depth in milliseconds.
    /// Range: [0.0, 20.0]. Default: 5.0.
    pub chorus_depth_ms: f32,
    /// Wet/dry mix for the chorus delay. Range: [0.0, 1.0]. Default: 0.4.
    pub mix: f32,
}

impl Default for ThickenerConfig {
    fn default() -> Self {
        Self {
            pitch_depth_cents: 8.0,
            time_depth_ms: 1.5,
            amplitude_depth_db: 1.0,
            lfo_rate_hz: 0.5,
            lfo_shape: LfoShape::Sine,
            chorus_delay_ms: 15.0,
            chorus_depth_ms: 5.0,
            mix: 0.4,
        }
    }
}

impl ThickenerConfig {
    /// Create a default, validated configuration.
    pub fn new() -> Result<Self, KokoroError> {
        let c = Self::default();
        c.validate()?;
        Ok(c)
    }

    // -- Builder methods ------------------------------------------------------

    #[must_use]
    pub fn with_pitch_depth_cents(mut self, v: f32) -> Self {
        self.pitch_depth_cents = v;
        self
    }
    #[must_use]
    pub fn with_time_depth_ms(mut self, v: f32) -> Self {
        self.time_depth_ms = v;
        self
    }
    #[must_use]
    pub fn with_amplitude_depth_db(mut self, v: f32) -> Self {
        self.amplitude_depth_db = v;
        self
    }
    #[must_use]
    pub fn with_lfo_rate_hz(mut self, v: f32) -> Self {
        self.lfo_rate_hz = v;
        self
    }
    #[must_use]
    pub fn with_lfo_shape(mut self, v: LfoShape) -> Self {
        self.lfo_shape = v;
        self
    }
    #[must_use]
    pub fn with_chorus_delay_ms(mut self, v: f32) -> Self {
        self.chorus_delay_ms = v;
        self
    }
    #[must_use]
    pub fn with_chorus_depth_ms(mut self, v: f32) -> Self {
        self.chorus_depth_ms = v;
        self
    }
    #[must_use]
    pub fn with_mix(mut self, v: f32) -> Self {
        self.mix = v;
        self
    }

    /// Validate all configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any field is non-finite or
    /// outside its valid range.
    pub fn validate(&self) -> Result<(), KokoroError> {
        check_range("pitch_depth_cents", self.pitch_depth_cents, 0.0, 30.0)?;
        check_range("time_depth_ms", self.time_depth_ms, 0.0, 10.0)?;
        check_range("amplitude_depth_db", self.amplitude_depth_db, 0.0, 6.0)?;
        check_range("lfo_rate_hz", self.lfo_rate_hz, 0.01, 10.0)?;
        check_range("chorus_delay_ms", self.chorus_delay_ms, 0.0, 50.0)?;
        check_range("chorus_depth_ms", self.chorus_depth_ms, 0.0, 20.0)?;
        check_range("mix", self.mix, 0.0, 1.0)?;
        Ok(())
    }

    // -- Presets ---------------------------------------------------------------

    /// Subtle thickening: barely perceptible micro-variation.
    ///
    /// Good for solo or lead vocals where you want natural width
    /// without obvious modulation.
    #[must_use]
    pub fn subtle() -> Self {
        Self {
            pitch_depth_cents: 5.0,
            time_depth_ms: 0.8,
            amplitude_depth_db: 0.5,
            lfo_rate_hz: 0.3,
            lfo_shape: LfoShape::Sine,
            chorus_delay_ms: 10.0,
            chorus_depth_ms: 2.0,
            mix: 0.25,
        }
    }

    /// Lush thickening: rich, enveloping chorus character.
    ///
    /// Multiple modulation dimensions at moderate depth produce
    /// a warm, wide sound suitable for backing vocals.
    #[must_use]
    pub fn lush() -> Self {
        Self {
            pitch_depth_cents: 10.0,
            time_depth_ms: 2.0,
            amplitude_depth_db: 1.2,
            lfo_rate_hz: 0.5,
            lfo_shape: LfoShape::Sine,
            chorus_delay_ms: 18.0,
            chorus_depth_ms: 6.0,
            mix: 0.45,
        }
    }

    /// Dramatic thickening: bold, wide chorus effect.
    ///
    /// Higher modulation depths and faster LFO for a pronounced
    /// ensemble character. Best for multi-voice chorus arrangements.
    #[must_use]
    pub fn dramatic() -> Self {
        Self {
            pitch_depth_cents: 15.0,
            time_depth_ms: 3.0,
            amplitude_depth_db: 2.0,
            lfo_rate_hz: 0.8,
            lfo_shape: LfoShape::Triangle,
            chorus_delay_ms: 22.0,
            chorus_depth_ms: 8.0,
            mix: 0.55,
        }
    }

    /// Gentle sway: slow, deep modulation for a dreamy wash.
    ///
    /// Very slow LFO with moderate depth creates a swaying,
    /// undulating texture. Good for atmospheric pads or ambient vocals.
    #[must_use]
    pub fn gentle_sway() -> Self {
        Self {
            pitch_depth_cents: 7.0,
            time_depth_ms: 2.5,
            amplitude_depth_db: 0.8,
            lfo_rate_hz: 0.15,
            lfo_shape: LfoShape::Triangle,
            chorus_delay_ms: 20.0,
            chorus_depth_ms: 7.0,
            mix: 0.4,
        }
    }
}

fn check_range(field: &'static str, val: f32, lo: f32, hi: f32) -> Result<(), KokoroError> {
    if !val.is_finite() || val < lo || val > hi {
        Err(KokoroError::InvalidConfig {
            field,
            reason: format!("must be finite and in [{lo}, {hi}], got {val}"),
        })
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Per-voice LFO bank
// ---------------------------------------------------------------------------

/// Per-voice LFO state for a single modulation dimension.
#[derive(Debug, Clone)]
struct LfoBank {
    /// Phase accumulator per voice (radians, in [0, TAU)).
    phases: Vec<f32>,
    /// Phase increment per sample (radians).
    increment: f32,
    /// Waveform shape.
    shape: LfoShape,
}

impl LfoBank {
    /// Create a new LFO bank with per-voice phase offsets.
    ///
    /// `phase_seed` shifts the initial phase spread so that different
    /// modulation dimensions (pitch, time, amplitude) are decorrelated.
    fn new(
        n_voices: usize,
        rate_hz: f32,
        sample_rate: f32,
        shape: LfoShape,
        phase_seed: f32,
    ) -> Self {
        let increment = std::f32::consts::TAU * rate_hz / sample_rate;
        // IEEE 754 safety: if sample_rate or rate_hz produce non-finite
        // increment, clamp to zero (effectively disabling modulation).
        let increment = if increment.is_finite() {
            increment
        } else {
            0.0
        };

        let phases: Vec<f32> = (0..n_voices)
            .map(|i| {
                // Golden-ratio-based phase spread for maximal decorrelation
                // between voices, offset by phase_seed per dimension.
                let phi = phase_seed + i as f32 * std::f32::consts::TAU * 0.618_033_9;
                phi.rem_euclid(std::f32::consts::TAU)
            })
            .collect();

        Self {
            phases,
            increment,
            shape,
        }
    }

    /// Evaluate the LFO for a given voice and advance its phase by one sample.
    ///
    /// Returns a value in [-1.0, 1.0].
    #[inline]
    fn tick(&mut self, voice: usize) -> f32 {
        let phase = self.phases[voice];
        let val = match self.shape {
            LfoShape::Sine => phase.sin(),
            LfoShape::Triangle => {
                // Triangle: linear ramp, period = TAU.
                // Map phase [0, TAU) to [-1, 1]:
                //   [0, PI/2)     → 0..1
                //   [PI/2, 3PI/2) → 1..-1
                //   [3PI/2, TAU)  → -1..0
                let t = phase / std::f32::consts::TAU; // [0, 1)
                if t < 0.25 {
                    t * 4.0
                } else if t < 0.75 {
                    2.0 - t * 4.0
                } else {
                    t * 4.0 - 4.0
                }
            }
        };
        // Advance phase.
        let next = phase + self.increment;
        self.phases[voice] = next.rem_euclid(std::f32::consts::TAU);
        // IEEE 754 safety.
        if val.is_finite() {
            val
        } else {
            0.0
        }
    }

    /// Reset all phases to their initial offsets.
    fn reset(&mut self, phase_seed: f32) {
        for (i, p) in self.phases.iter_mut().enumerate() {
            let phi = phase_seed + i as f32 * std::f32::consts::TAU * 0.618_033_9;
            *p = phi.rem_euclid(std::f32::consts::TAU);
        }
    }
}

// ---------------------------------------------------------------------------
// Modulated delay line (for pitch and chorus delay)
// ---------------------------------------------------------------------------

/// Circular buffer delay line with linear-interpolated fractional read.
#[derive(Debug, Clone)]
struct DelayLine {
    buffer: Vec<f32>,
    write_pos: usize,
}

impl DelayLine {
    fn new(max_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; max_samples.max(4)],
            write_pos: 0,
        }
    }

    /// Write one sample and advance the write head.
    #[inline]
    fn write(&mut self, sample: f32) {
        let s = if sample.is_finite() { sample } else { 0.0 };
        self.buffer[self.write_pos] = s;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
    }

    /// Read at a fractional delay (in samples) behind the write head.
    ///
    /// Uses linear interpolation between the two nearest integer taps.
    #[inline]
    fn read_interpolated(&self, delay_samples: f32) -> f32 {
        let delay = delay_samples.max(0.0);
        let len = self.buffer.len();
        let int_delay = delay as usize;
        let frac = delay - int_delay as f32;

        // Two tap positions (behind write head).
        let idx0 = (self.write_pos + len - 1 - int_delay) % len;
        let idx1 = (self.write_pos + len - 2 - int_delay) % len;

        let s0 = self.buffer[idx0];
        let s1 = self.buffer[idx1];
        let out = s0 + frac * (s1 - s0);
        if out.is_finite() {
            out
        } else {
            0.0
        }
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }
}

// ---------------------------------------------------------------------------
// Processor
// ---------------------------------------------------------------------------

/// Per-voice micro-modulation processor that thickens chorus sound.
///
/// Maintains separate LFO banks for pitch, timing, and amplitude modulation,
/// plus per-voice delay lines for pitch shifting and chorus delay.
#[derive(Debug, Clone)]
pub struct ThickenerProcessor {
    config: ThickenerConfig,
    n_voices: usize,
    sample_rate: f32,

    // LFO banks — one per modulation dimension, decorrelated by phase seed.
    lfo_pitch: LfoBank,
    lfo_time: LfoBank,
    lfo_amp: LfoBank,
    lfo_chorus: LfoBank,

    // Per-voice delay lines for pitch micro-modulation.
    pitch_delays: Vec<DelayLine>,
    // Per-voice delay lines for chorus modulated delay.
    chorus_delays: Vec<DelayLine>,

    // Precomputed constants.
    pitch_max_delay_samples: f32,
    chorus_base_delay_samples: f32,
    chorus_depth_samples: f32,
}

impl ThickenerProcessor {
    /// Create a new thickener processor.
    ///
    /// # Arguments
    ///
    /// * `config` — Thickener configuration (validated on construction).
    /// * `n_voices` — Number of chorus voices to process.
    /// * `sample_rate` — Audio sample rate in Hz.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if the config is invalid or
    /// `sample_rate` is below 1000.
    pub fn new(
        config: &ThickenerConfig,
        n_voices: usize,
        sample_rate: f32,
    ) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate < 1000.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("must be >= 1000, got {sample_rate}"),
            });
        }
        if n_voices == 0 {
            return Err(KokoroError::InvalidConfig {
                field: "n_voices",
                reason: "must be >= 1".to_string(),
            });
        }

        // Pitch modulation delay: convert max cents to max delay in samples.
        // Pitch shift of C cents requires a delay modulation of approximately
        // |C/1200| * period samples. For small cents at reasonable sample rates,
        // a buffer of ~50ms is more than sufficient.
        let pitch_buf_ms = 50.0;
        let pitch_buf_samples = ((pitch_buf_ms / 1000.0) * sample_rate) as usize + 4;
        let pitch_max_delay_samples = (config.pitch_depth_cents / 1200.0) * sample_rate * 0.01;
        // Clamp to buffer size for safety.
        let pitch_max_delay_samples = pitch_max_delay_samples
            .min(pitch_buf_samples as f32 - 2.0)
            .max(0.0);

        // Chorus delay: base + modulation depth.
        let chorus_base_samples = (config.chorus_delay_ms / 1000.0) * sample_rate;
        let chorus_depth_samples = (config.chorus_depth_ms / 1000.0) * sample_rate;
        let chorus_buf_samples = ((config.chorus_delay_ms + config.chorus_depth_ms + 10.0) / 1000.0
            * sample_rate) as usize
            + 4;

        // LFO banks — each dimension gets a different phase seed for
        // decorrelation. Seeds are irrational multiples of TAU.
        let lfo_pitch = LfoBank::new(
            n_voices,
            config.lfo_rate_hz,
            sample_rate,
            config.lfo_shape,
            0.0,
        );
        let lfo_time = LfoBank::new(
            n_voices,
            config.lfo_rate_hz * 1.07, // slight rate offset
            sample_rate,
            config.lfo_shape,
            std::f32::consts::TAU * 0.333,
        );
        let lfo_amp = LfoBank::new(
            n_voices,
            config.lfo_rate_hz * 0.93, // slight rate offset
            sample_rate,
            config.lfo_shape,
            std::f32::consts::TAU * 0.667,
        );
        let lfo_chorus = LfoBank::new(
            n_voices,
            config.lfo_rate_hz * 0.87,
            sample_rate,
            config.lfo_shape,
            std::f32::consts::TAU * 0.5,
        );

        let pitch_delays = (0..n_voices)
            .map(|_| DelayLine::new(pitch_buf_samples))
            .collect();
        let chorus_delays = (0..n_voices)
            .map(|_| DelayLine::new(chorus_buf_samples))
            .collect();

        Ok(Self {
            config: config.clone(),
            n_voices,
            sample_rate,
            lfo_pitch,
            lfo_time,
            lfo_amp,
            lfo_chorus,
            pitch_delays,
            chorus_delays,
            pitch_max_delay_samples,
            chorus_base_delay_samples: chorus_base_samples,
            chorus_depth_samples,
        })
    }

    /// Process voices in-place, applying micro-modulation and chorus delay.
    ///
    /// Each element of `voices` is a mono audio buffer for one voice.
    /// All buffers must have the same length; mismatched lengths are
    /// truncated to the shortest buffer.
    pub fn process_voices(&mut self, voices: &mut [Vec<f32>]) {
        if voices.is_empty() {
            return;
        }
        let n = voices.len().min(self.n_voices);
        let len = voices.iter().take(n).map(Vec::len).min().unwrap_or(0);
        if len == 0 {
            return;
        }

        let mix = self.config.mix;
        let dry = 1.0 - mix;

        for sample_idx in 0..len {
            for voice_idx in 0..n {
                let input = voices[voice_idx][sample_idx];
                if !input.is_finite() {
                    voices[voice_idx][sample_idx] = 0.0;
                    // Still tick LFOs to keep phase aligned.
                    self.lfo_pitch.tick(voice_idx);
                    self.lfo_time.tick(voice_idx);
                    self.lfo_amp.tick(voice_idx);
                    self.lfo_chorus.tick(voice_idx);
                    self.pitch_delays[voice_idx].write(0.0);
                    self.chorus_delays[voice_idx].write(0.0);
                    continue;
                }

                // --- Pitch modulation via delay line ---
                let pitch_mod = self.lfo_pitch.tick(voice_idx);
                let pitch_delay = self.pitch_max_delay_samples * (1.0 + pitch_mod) * 0.5; // map [-1,1] to [0, max]
                self.pitch_delays[voice_idx].write(input);
                let pitch_shifted = self.pitch_delays[voice_idx].read_interpolated(pitch_delay);

                // --- Timing modulation via variable delay ---
                let time_mod = self.lfo_time.tick(voice_idx);
                let time_delay_samples = (self.config.time_depth_ms / 1000.0)
                    * self.sample_rate
                    * (1.0 + time_mod)
                    * 0.5;
                // Re-use pitch delay line for time modulation read at a
                // different tap. The pitch-shifted signal is our input.
                // For timing, we simply blend a slight shift into the signal.
                // We apply a weighted blend of the original and a time-shifted
                // version. Keep it simple: time modulation creates a slight
                // comb filtering effect that adds perceived width.
                let time_blend = if time_delay_samples > 0.5 {
                    self.pitch_delays[voice_idx].read_interpolated(time_delay_samples)
                } else {
                    pitch_shifted
                };
                // Mix pitch-shifted and time-modulated 50/50.
                let modulated = (pitch_shifted + time_blend) * 0.5;

                // --- Amplitude modulation ---
                let amp_mod = self.lfo_amp.tick(voice_idx);
                // Convert dB depth to linear gain variation.
                let db_offset = self.config.amplitude_depth_db * amp_mod;
                // 10^(dB/20) conversion: for small dB values this is
                // approximately 1 + 0.115 * dB.
                let gain = db_to_linear(db_offset);
                let amp_modulated = modulated * gain;

                // --- Chorus modulated delay for width ---
                let chorus_mod = self.lfo_chorus.tick(voice_idx);
                let chorus_delay =
                    self.chorus_base_delay_samples + self.chorus_depth_samples * chorus_mod;
                self.chorus_delays[voice_idx].write(amp_modulated);
                let chorus_wet =
                    self.chorus_delays[voice_idx].read_interpolated(chorus_delay.max(0.0));

                // --- Mix ---
                let output = dry * amp_modulated + mix * chorus_wet;
                voices[voice_idx][sample_idx] = if output.is_finite() { output } else { 0.0 };
            }
        }
    }

    /// Reset all internal state (delay lines, LFO phases).
    pub fn reset(&mut self) {
        self.lfo_pitch.reset(0.0);
        self.lfo_time.reset(std::f32::consts::TAU * 0.333);
        self.lfo_amp.reset(std::f32::consts::TAU * 0.667);
        self.lfo_chorus.reset(std::f32::consts::TAU * 0.5);
        for dl in &mut self.pitch_delays {
            dl.reset();
        }
        for dl in &mut self.chorus_delays {
            dl.reset();
        }
    }

    /// Returns the number of voices this processor was configured for.
    #[must_use]
    pub fn n_voices(&self) -> usize {
        self.n_voices
    }

    /// Returns the sample rate this processor was configured with.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Returns a reference to the active configuration.
    #[must_use]
    pub fn config(&self) -> &ThickenerConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Convert decibels to linear gain.
///
/// `db_to_linear(0.0) == 1.0`, `db_to_linear(6.0) ≈ 2.0`.
#[inline]
fn db_to_linear(db: f32) -> f32 {
    if !db.is_finite() {
        return 1.0;
    }
    let lin = 10.0_f32.powf(db / 20.0);
    if lin.is_finite() {
        lin
    } else {
        1.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "kokoro_chorus_thickener_tests.rs"]
mod tests;
