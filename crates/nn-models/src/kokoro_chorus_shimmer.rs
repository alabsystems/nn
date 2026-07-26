// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Vocal shimmer and air harmonics processor for Kokoro chorus voices.
//!
//! This module adds airy high-frequency harmonics and subtle brightness to make
//! the chorus sparkle — the "expensive microphone" effect. It adds presence,
//! air, and shimmer without harshness.
//!
//! # Processing Chain
//!
//! ```text
//! Input ─┬───────────────────────────────────────────────── dry
//!        │
//!        ├─> Peaking EQ (silk/presence at 3.5 kHz)
//!        │
//!        ├─> Peaking EQ (air band at 12 kHz)
//!        │
//!        ├─> Granular pitch shifter (+1 octave) ───┐
//!        │   with feedback loop (cascading harmonics)│
//!        │                                          v
//!        │   Bandpass (>6 kHz) ──> shimmer_amount blend
//!        │
//!        └─> High-shelf (brightness control, -3..+3 dB)
//!
//! Output = lerp(dry, processed, mix)
//! ```
//!
//! # Design Rationale
//!
//! - **Air band boost**: A peaking EQ at ~12 kHz adds the "air" band sparkle
//!   heard on expensive condenser microphones and high-end vocal chains.
//! - **Silk/presence boost**: A gentle peaking EQ at ~3.5 kHz adds vocal
//!   clarity and intelligibility without sounding harsh.
//! - **Shimmer generation**: A granular pitch shifter shifts the signal up by
//!   one octave (configurable), then feeds back with decay to create cascading
//!   harmonics reminiscent of reverb shimmer effects. Only the high-frequency
//!   portion (>6 kHz) of the shifted signal is kept, preventing muddy bass
//!   artifacts.
//! - **Brightness control**: A variable high-shelf that sweeps from -3 dB cut
//!   (dark) to +3 dB boost (bright) based on the brightness parameter.
//!
//! # References
//!
//! - Dolson, M. "The Phase Vocoder: A Tutorial." Computer Music Journal,
//!   10(4), 1986.
//! - Roads, C. "Microsound." MIT Press, 2001. Chapter 3: Granular Synthesis.
//! - Smith, J. O. "Spectral Audio Signal Processing."
//!   <https://ccrma.stanford.edu/~jos/sasp/>
//! - Zolzer, U. "DAFX: Digital Audio Effects." 2nd ed., Wiley, 2011.
//!   Chapter 7: Time-Segment Processing.
//!
//! Part of #4582, #3351.

use crate::kokoro_chorus_saturation::db_to_linear;
use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the vocal shimmer and air harmonics processor.
///
/// Constructed via [`ShimmerConfig::new`] (required for cross-crate use
/// due to `#[non_exhaustive]`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ShimmerConfig {
    /// Air band center frequency in Hz for the peaking EQ boost.
    /// Range: 8000.0 - 16000.0. Default: 12000.0.
    pub air_freq_hz: f32,
    /// Air band boost in dB.
    /// Range: 0.0 - 6.0. Default: 3.0.
    pub air_gain_db: f32,
    /// Air band Q (bandwidth) of the peaking EQ.
    /// Range: 0.3 - 3.0. Default: 0.7.
    pub air_q: f32,
    /// Shimmer intensity: 0.0 = no shimmer, 1.0 = full shimmer.
    /// Range: 0.0 - 1.0. Default: 0.3.
    pub shimmer_amount: f32,
    /// Octave shift for shimmer harmonics generation.
    /// 1.0 = up one octave, 2.0 = up two octaves.
    /// Range: 0.5 - 2.0. Default: 1.0.
    pub shimmer_octave_shift: f32,
    /// Shimmer feedback decay: controls cascading harmonic buildup.
    /// 0.0 = no feedback, 1.0 = infinite sustain (clamped below 0.95).
    /// Range: 0.0 - 0.95. Default: 0.4.
    pub shimmer_decay: f32,
    /// Silk/presence band center frequency in Hz.
    /// Range: 2000.0 - 6000.0. Default: 3500.0.
    pub silk_freq_hz: f32,
    /// Silk band gentle boost in dB.
    /// Range: 0.0 - 6.0. Default: 1.5.
    pub silk_gain_db: f32,
    /// Overall brightness control: 0.0 = dark (-3 dB shelf), 1.0 = bright
    /// (+3 dB shelf). 0.5 = neutral.
    /// Range: 0.0 - 1.0. Default: 0.5.
    pub brightness: f32,
    /// Dry/wet mix: 0.0 = fully dry (bypass), 1.0 = fully wet.
    /// Range: 0.0 - 1.0. Default: 0.3.
    pub mix: f32,
}

impl Default for ShimmerConfig {
    fn default() -> Self {
        Self {
            air_freq_hz: 12000.0,
            air_gain_db: 3.0,
            air_q: 0.7,
            shimmer_amount: 0.3,
            shimmer_octave_shift: 1.0,
            shimmer_decay: 0.4,
            silk_freq_hz: 3500.0,
            silk_gain_db: 1.5,
            brightness: 0.5,
            mix: 0.3,
        }
    }
}

impl ShimmerConfig {
    /// Create a new shimmer config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the air band center frequency in Hz.
    #[must_use]
    pub fn with_air_freq_hz(mut self, hz: f32) -> Self {
        self.air_freq_hz = hz;
        self
    }

    /// Set the air band boost in dB.
    #[must_use]
    pub fn with_air_gain_db(mut self, db: f32) -> Self {
        self.air_gain_db = db;
        self
    }

    /// Set the air band Q width.
    #[must_use]
    pub fn with_air_q(mut self, q: f32) -> Self {
        self.air_q = q;
        self
    }

    /// Set the shimmer intensity (0.0 - 1.0).
    #[must_use]
    pub fn with_shimmer_amount(mut self, amount: f32) -> Self {
        self.shimmer_amount = amount;
        self
    }

    /// Set the shimmer octave shift.
    #[must_use]
    pub fn with_shimmer_octave_shift(mut self, shift: f32) -> Self {
        self.shimmer_octave_shift = shift;
        self
    }

    /// Set the shimmer feedback decay.
    #[must_use]
    pub fn with_shimmer_decay(mut self, decay: f32) -> Self {
        self.shimmer_decay = decay;
        self
    }

    /// Set the silk/presence band center frequency in Hz.
    #[must_use]
    pub fn with_silk_freq_hz(mut self, hz: f32) -> Self {
        self.silk_freq_hz = hz;
        self
    }

    /// Set the silk band boost in dB.
    #[must_use]
    pub fn with_silk_gain_db(mut self, db: f32) -> Self {
        self.silk_gain_db = db;
        self
    }

    /// Set the brightness control (0.0 = dark, 1.0 = bright).
    #[must_use]
    pub fn with_brightness(mut self, brightness: f32) -> Self {
        self.brightness = brightness;
        self
    }

    /// Set the dry/wet mix (0.0 = dry, 1.0 = wet).
    #[must_use]
    pub fn with_mix(mut self, mix: f32) -> Self {
        self.mix = mix;
        self
    }

    /// Validate all parameters are within acceptable ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        validate_finite_range(self.air_freq_hz, 8000.0, 16000.0, "air_freq_hz")?;
        validate_finite_range(self.air_gain_db, 0.0, 6.0, "air_gain_db")?;
        validate_finite_range(self.air_q, 0.3, 3.0, "air_q")?;
        validate_finite_range(self.shimmer_amount, 0.0, 1.0, "shimmer_amount")?;
        validate_finite_range(self.shimmer_octave_shift, 0.5, 2.0, "shimmer_octave_shift")?;
        validate_finite_range(self.shimmer_decay, 0.0, 0.95, "shimmer_decay")?;
        validate_finite_range(self.silk_freq_hz, 2000.0, 6000.0, "silk_freq_hz")?;
        validate_finite_range(self.silk_gain_db, 0.0, 6.0, "silk_gain_db")?;
        validate_finite_range(self.brightness, 0.0, 1.0, "brightness")?;
        validate_finite_range(self.mix, 0.0, 1.0, "mix")?;
        Ok(())
    }

    // --- Presets ---------------------------------------------------------------

    /// Subtle: gentle air + silk, no shimmer effect.
    /// Good for spoken word and narration where clarity is needed
    /// without any ethereal character.
    #[must_use]
    pub fn subtle() -> Self {
        Self {
            air_freq_hz: 11000.0,
            air_gain_db: 2.0,
            air_q: 0.8,
            shimmer_amount: 0.0,
            shimmer_octave_shift: 1.0,
            shimmer_decay: 0.0,
            silk_freq_hz: 3500.0,
            silk_gain_db: 1.5,
            brightness: 0.55,
            mix: 0.25,
        }
    }

    /// Sparkle: moderate shimmer + air, default for chorus.
    /// Adds airy harmonics and gentle cascading shimmer that makes
    /// a multi-voice chorus sound open and spacious.
    #[must_use]
    pub fn sparkle() -> Self {
        Self {
            air_freq_hz: 12000.0,
            air_gain_db: 3.5,
            air_q: 0.7,
            shimmer_amount: 0.35,
            shimmer_octave_shift: 1.0,
            shimmer_decay: 0.4,
            silk_freq_hz: 3500.0,
            silk_gain_db: 1.5,
            brightness: 0.6,
            mix: 0.3,
        }
    }

    /// Ethereal: heavy shimmer with long decay, cathedral-like.
    /// Creates a lush, otherworldly quality with cascading octave
    /// harmonics that ring out over time.
    #[must_use]
    pub fn ethereal() -> Self {
        Self {
            air_freq_hz: 13000.0,
            air_gain_db: 4.0,
            air_q: 0.5,
            shimmer_amount: 0.7,
            shimmer_octave_shift: 1.0,
            shimmer_decay: 0.85,
            silk_freq_hz: 4000.0,
            silk_gain_db: 2.0,
            brightness: 0.7,
            mix: 0.45,
        }
    }

    /// Broadcast: minimal shimmer, just air and presence for clarity.
    /// Clean, professional sound suitable for podcast and broadcast
    /// vocal processing.
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            air_freq_hz: 10000.0,
            air_gain_db: 2.5,
            air_q: 1.0,
            shimmer_amount: 0.05,
            shimmer_octave_shift: 1.0,
            shimmer_decay: 0.2,
            silk_freq_hz: 3000.0,
            silk_gain_db: 2.0,
            brightness: 0.5,
            mix: 0.2,
        }
    }
}

/// Validate that a value is finite and within [min, max].
fn validate_finite_range(
    value: f32,
    min: f32,
    max: f32,
    field: &'static str,
) -> Result<(), KokoroError> {
    if !value.is_finite() || value < min || value > max {
        return Err(KokoroError::InvalidConfig {
            field,
            reason: format!("{field} = {value}: must be finite and in [{min}, {max}]"),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Second-order biquad filters
// ---------------------------------------------------------------------------

/// Second-order peaking (bell) EQ biquad filter.
///
/// Based on the Audio EQ Cookbook (Robert Bristow-Johnson). Used for both
/// the air band and silk/presence boosts.
#[derive(Debug, Clone)]
struct BiquadPeaking {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BiquadPeaking {
    fn new(freq_hz: f32, gain_db: f32, q: f32, sample_rate: f32) -> Self {
        if gain_db.abs() < 1e-6 {
            return Self::passthrough();
        }
        let a = 10.0_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * freq_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a;

        let inv_a0 = 1.0 / a0;
        Self {
            b0: b0 * inv_a0,
            b1: b1 * inv_a0,
            b2: b2 * inv_a0,
            a1: a1 * inv_a0,
            a2: a2 * inv_a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn passthrough() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.x1 = 0.0;
            self.x2 = 0.0;
            self.y1 = 0.0;
            self.y2 = 0.0;
            return 0.0;
        }
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = if y.is_finite() { y } else { 0.0 };
        self.y1
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// First-order high-shelf filter
// ---------------------------------------------------------------------------

/// First-order high-shelf filter for brightness control.
///
/// Boosts or cuts all frequencies above the shelf frequency. Based on
/// Zolzer's DAFX first-order shelving design.
#[derive(Debug, Clone)]
struct HighShelf {
    b0: f32,
    b1: f32,
    a1: f32,
    x_prev: f32,
    y_prev: f32,
}

impl HighShelf {
    fn new(freq_hz: f32, gain_db: f32, sample_rate: f32) -> Self {
        if gain_db.abs() < 1e-6 {
            return Self {
                b0: 1.0,
                b1: 0.0,
                a1: 0.0,
                x_prev: 0.0,
                y_prev: 0.0,
            };
        }

        let v0 = db_to_linear(gain_db);
        let k = (std::f32::consts::PI * freq_hz / sample_rate).tan();

        if v0 >= 1.0 {
            let denom = 1.0 + k;
            Self {
                b0: (v0 + k) / denom,
                b1: (k - v0) / denom,
                a1: (k - 1.0) / denom,
                x_prev: 0.0,
                y_prev: 0.0,
            }
        } else {
            let denom = v0 + k;
            Self {
                b0: v0 * (1.0 + k) / denom,
                b1: v0 * (k - 1.0) / denom,
                a1: (k - v0) / denom,
                x_prev: 0.0,
                y_prev: 0.0,
            }
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.x_prev = 0.0;
            self.y_prev = 0.0;
            return 0.0;
        }
        let y = self.b0 * x + self.b1 * self.x_prev - self.a1 * self.y_prev;
        self.x_prev = x;
        self.y_prev = if y.is_finite() { y } else { 0.0 };
        self.y_prev
    }

    fn reset(&mut self) {
        self.x_prev = 0.0;
        self.y_prev = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Single-pole highpass (for shimmer bandpass and DC blocking)
// ---------------------------------------------------------------------------

/// Single-pole highpass filter (RC time-constant derived).
#[derive(Debug, Clone)]
struct OnePoleHP {
    coeff: f32,
    x_prev: f32,
    y_prev: f32,
}

impl OnePoleHP {
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
        let dt = 1.0 / sample_rate;
        Self {
            coeff: rc / (rc + dt),
            x_prev: 0.0,
            y_prev: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.x_prev = 0.0;
            self.y_prev = 0.0;
            return 0.0;
        }
        let y = self.coeff * (self.y_prev + x - self.x_prev);
        self.x_prev = x;
        self.y_prev = if y.is_finite() { y } else { 0.0 };
        self.y_prev
    }

    fn reset(&mut self) {
        self.x_prev = 0.0;
        self.y_prev = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Granular pitch shifter
// ---------------------------------------------------------------------------

/// Grain size range in samples at 24 kHz (20ms - 40ms).
const GRAIN_MIN_MS: f32 = 20.0;
const GRAIN_MAX_MS: f32 = 40.0;
/// Overlap factor: 75% overlap means 4 overlapping grains.
const GRAIN_OVERLAP: usize = 4;

/// Granular pitch shifter for shimmer harmonic generation.
///
/// Uses overlap-add granular synthesis with Hann-windowed grains to pitch-
/// shift the input signal upward. The pitch shift is achieved by reading
/// grains at a different rate than writing them (time-stretch + resample
/// in a single pass).
///
/// Each grain is read from a circular delay buffer at a rate determined
/// by the pitch ratio. With 75% overlap and Hann windows, the grains
/// recombine smoothly with minimal artifacts.
#[derive(Debug, Clone)]
struct GranularPitchShifter {
    /// Circular delay buffer.
    buffer: Vec<f32>,
    /// Write position into the buffer.
    write_pos: usize,
    /// Per-grain read positions (fractional).
    grain_read_pos: [f64; GRAIN_OVERLAP],
    /// Per-grain phase within the current grain (0..grain_size).
    grain_phase: [usize; GRAIN_OVERLAP],
    /// Grain size in samples.
    grain_size: usize,
    /// Pitch ratio (2.0 = up one octave).
    pitch_ratio: f64,
    /// Pre-computed Hann window for one grain.
    hann_window: Vec<f32>,
}

impl GranularPitchShifter {
    fn new(octave_shift: f32, sample_rate: f32) -> Self {
        let pitch_ratio = 2.0_f64.powf(f64::from(octave_shift));

        // Use the midpoint grain size (30ms) for the default.
        let grain_ms = f32::midpoint(GRAIN_MIN_MS, GRAIN_MAX_MS);
        let grain_size = (grain_ms * sample_rate / 1000.0) as usize;
        let grain_size = grain_size.max(16);

        // Buffer needs to be large enough to hold several grains.
        let buf_size = grain_size * (GRAIN_OVERLAP + 2);

        // Pre-compute Hann window.
        let hann_window: Vec<f32> = (0..grain_size)
            .map(|i| {
                let phase = i as f32 / grain_size as f32;
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * phase).cos())
            })
            .collect();

        // Initialize grain read positions staggered across the grain.
        let hop = grain_size / GRAIN_OVERLAP;
        let mut grain_read_pos = [0.0_f64; GRAIN_OVERLAP];
        let mut grain_phase = [0usize; GRAIN_OVERLAP];
        for (i, (rp, gp)) in grain_read_pos
            .iter_mut()
            .zip(grain_phase.iter_mut())
            .enumerate()
        {
            *gp = i * hop;
            *rp = 0.0;
        }

        Self {
            buffer: vec![0.0; buf_size],
            write_pos: 0,
            grain_read_pos,
            grain_phase,
            grain_size,
            pitch_ratio,
            hann_window,
        }
    }

    /// Process one sample: write input into buffer, read pitch-shifted output
    /// from overlapping grains.
    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let input = if input.is_finite() { input } else { 0.0 };
        let buf_len = self.buffer.len();

        // Write input into circular buffer.
        self.buffer[self.write_pos] = input;
        self.write_pos = (self.write_pos + 1) % buf_len;

        let hop = self.grain_size / GRAIN_OVERLAP;
        let mut output = 0.0_f32;

        for grain_idx in 0..GRAIN_OVERLAP {
            let phase = self.grain_phase[grain_idx];

            if phase < self.grain_size {
                // Hann window amplitude for this sample within the grain.
                let window = self.hann_window[phase];

                // Read from buffer at fractional position.
                let read_pos = self.grain_read_pos[grain_idx];
                let read_int = read_pos as usize % buf_len;
                let frac = (read_pos - read_pos.floor()) as f32;

                let s0 = self.buffer[read_int];
                let s1 = self.buffer[(read_int + 1) % buf_len];
                let sample = s0 + frac * (s1 - s0);

                output += sample * window;

                // Advance read position by pitch ratio.
                self.grain_read_pos[grain_idx] += self.pitch_ratio;
            }

            // Advance grain phase.
            self.grain_phase[grain_idx] += 1;

            // Reset grain when it completes a full cycle.
            if self.grain_phase[grain_idx] >= self.grain_size {
                self.grain_phase[grain_idx] = 0;
                // Re-anchor read position to current write position offset
                // by the grain stagger.
                let offset = grain_idx * hop;
                self.grain_read_pos[grain_idx] = (self.write_pos as f64) - (offset as f64);
                if self.grain_read_pos[grain_idx] < 0.0 {
                    self.grain_read_pos[grain_idx] += buf_len as f64;
                }
            }

            // Stagger grain resets: only reset one grain per hop samples.
            // The initial stagger handles this via phase offsets.
        }

        // Normalize by overlap count (Hann windows sum to ~OVERLAP/2).
        let norm = GRAIN_OVERLAP as f32 * 0.5;
        let result = output / norm;

        if result.is_finite() {
            result
        } else {
            0.0
        }
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        let hop = self.grain_size / GRAIN_OVERLAP;
        for (i, (rp, gp)) in self
            .grain_read_pos
            .iter_mut()
            .zip(self.grain_phase.iter_mut())
            .enumerate()
        {
            *gp = i * hop;
            *rp = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// ShimmerProcessor
// ---------------------------------------------------------------------------

/// Stateful vocal shimmer and air harmonics processor.
///
/// Holds filter state for the air band EQ, silk/presence EQ, granular
/// pitch shifter, shimmer feedback buffer, bandpass filter, and brightness
/// shelf.
#[derive(Debug, Clone)]
pub struct ShimmerProcessor {
    config: ShimmerConfig,
    /// Air band peaking EQ.
    air_eq: BiquadPeaking,
    /// Silk/presence peaking EQ.
    silk_eq: BiquadPeaking,
    /// Granular pitch shifter for shimmer generation.
    pitch_shifter: GranularPitchShifter,
    /// Highpass at 6 kHz to isolate only airy shimmer harmonics.
    shimmer_hp: OnePoleHP,
    /// Shimmer feedback accumulator (single sample).
    shimmer_feedback: f32,
    /// Brightness high-shelf filter.
    brightness_shelf: HighShelf,
    /// DC blocker for the shimmer path.
    dc_blocker: OnePoleHP,
}

impl ShimmerProcessor {
    /// Create a new shimmer processor from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range,
    /// or if `sample_rate` is not finite and positive.
    pub fn new(config: &ShimmerConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("sample_rate = {sample_rate}: must be finite and positive"),
            });
        }

        let air_eq = BiquadPeaking::new(
            config.air_freq_hz,
            config.air_gain_db,
            config.air_q,
            sample_rate,
        );

        let silk_eq = BiquadPeaking::new(
            config.silk_freq_hz,
            config.silk_gain_db,
            1.2, // moderate Q for vocal presence
            sample_rate,
        );

        let pitch_shifter = GranularPitchShifter::new(config.shimmer_octave_shift, sample_rate);

        // 6 kHz highpass to keep only airy harmonics from the shimmer.
        let shimmer_hp = OnePoleHP::new(6000.0, sample_rate);

        // Brightness: map 0.0-1.0 to -3.0..+3.0 dB.
        let brightness_db = (config.brightness - 0.5) * 6.0;
        let brightness_shelf = HighShelf::new(8000.0, brightness_db, sample_rate);

        // DC blocker at 20 Hz for shimmer path.
        let dc_blocker = OnePoleHP::new(20.0, sample_rate);

        Ok(Self {
            config: *config,
            air_eq,
            silk_eq,
            pitch_shifter,
            shimmer_hp,
            shimmer_feedback: 0.0,
            brightness_shelf,
            dc_blocker,
        })
    }

    /// Create a processor using Kokoro's default 24 kHz sample rate.
    pub fn new_kokoro(config: &ShimmerConfig) -> Result<Self, KokoroError> {
        Self::new(config, KOKORO_SAMPLE_RATE as f32)
    }

    /// Process per-voice audio in-place.
    ///
    /// Fast path: returns immediately when mix is zero.
    pub fn process_voice(&mut self, audio: &mut [f32]) {
        if self.config.mix == 0.0 {
            return;
        }

        let mix = self.config.mix;
        let shimmer_amount = self.config.shimmer_amount;
        let shimmer_decay = self.config.shimmer_decay;
        let has_shimmer = shimmer_amount > 0.0;

        for sample in audio.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
                continue;
            }

            let dry = *sample;
            let mut wet = dry;

            // --- Silk/presence EQ ---
            wet = self.silk_eq.process(wet);

            // --- Air band EQ ---
            wet = self.air_eq.process(wet);

            // --- Shimmer generation ---
            if has_shimmer {
                // Feed the signal + feedback into the pitch shifter.
                let shifter_input = wet + self.shimmer_feedback;
                let shifted = self.pitch_shifter.process(shifter_input);

                // Bandpass: keep only high-frequency shimmer harmonics.
                let shimmer_hf = self.shimmer_hp.process(shifted);

                // Remove DC from shimmer.
                let shimmer_clean = self.dc_blocker.process(shimmer_hf);

                // Update feedback with decay.
                self.shimmer_feedback = shimmer_clean * shimmer_decay;
                // Prevent feedback runaway.
                if !self.shimmer_feedback.is_finite() || self.shimmer_feedback.abs() > 2.0 {
                    self.shimmer_feedback = 0.0;
                }

                // Blend shimmer into the wet signal.
                wet += shimmer_clean * shimmer_amount;
            }

            // --- Brightness shelf ---
            wet = self.brightness_shelf.process(wet);

            // --- Final dry/wet mix ---
            *sample = dry * (1.0 - mix) + wet * mix;

            // Final NaN/Inf guard.
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }
    }

    /// Reset all internal filter state (call between unrelated audio segments).
    pub fn reset(&mut self) {
        self.air_eq.reset();
        self.silk_eq.reset();
        self.pitch_shifter.reset();
        self.shimmer_hp.reset();
        self.shimmer_feedback = 0.0;
        self.brightness_shelf.reset();
        self.dc_blocker.reset();
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &ShimmerConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = KOKORO_SAMPLE_RATE as f32;

    fn sine_wave(freq: f32, n: usize, amplitude: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / SR).sin())
            .collect()
    }

    fn rms(buf: &[f32]) -> f32 {
        let sum_sq: f32 = buf.iter().map(|x| x * x).sum();
        (sum_sq / buf.len().max(1) as f32).sqrt()
    }

    fn hf_energy(buf: &[f32], cutoff_hz: f32) -> f32 {
        let mut hp = OnePoleHP::new(cutoff_hz, SR);
        let filtered: Vec<f32> = buf.iter().map(|&x| hp.process(x)).collect();
        rms(&filtered)
    }

    // --- Config validation ---

    #[test]
    fn test_config_default_valid() {
        ShimmerConfig::new()
            .validate()
            .expect("default config should be valid");
    }

    #[test]
    fn test_config_builder_roundtrip() {
        let cfg = ShimmerConfig::new()
            .with_air_freq_hz(10000.0)
            .with_air_gain_db(2.0)
            .with_air_q(1.0)
            .with_shimmer_amount(0.5)
            .with_shimmer_octave_shift(1.5)
            .with_shimmer_decay(0.6)
            .with_silk_freq_hz(4000.0)
            .with_silk_gain_db(2.0)
            .with_brightness(0.7)
            .with_mix(0.4);
        cfg.validate().expect("builder config should be valid");
        assert_eq!(cfg.air_freq_hz, 10000.0);
        assert_eq!(cfg.air_gain_db, 2.0);
        assert_eq!(cfg.air_q, 1.0);
        assert_eq!(cfg.shimmer_amount, 0.5);
        assert_eq!(cfg.shimmer_octave_shift, 1.5);
        assert_eq!(cfg.shimmer_decay, 0.6);
        assert_eq!(cfg.silk_freq_hz, 4000.0);
        assert_eq!(cfg.silk_gain_db, 2.0);
        assert_eq!(cfg.brightness, 0.7);
        assert_eq!(cfg.mix, 0.4);
    }

    #[test]
    fn test_config_invalid_air_freq() {
        assert!(ShimmerConfig::new()
            .with_air_freq_hz(5000.0)
            .validate()
            .is_err());
        assert!(ShimmerConfig::new()
            .with_air_freq_hz(20000.0)
            .validate()
            .is_err());
        assert!(ShimmerConfig::new()
            .with_air_freq_hz(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_shimmer_amount() {
        assert!(ShimmerConfig::new()
            .with_shimmer_amount(-0.1)
            .validate()
            .is_err());
        assert!(ShimmerConfig::new()
            .with_shimmer_amount(1.1)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_shimmer_decay() {
        assert!(ShimmerConfig::new()
            .with_shimmer_decay(0.96)
            .validate()
            .is_err());
        assert!(ShimmerConfig::new()
            .with_shimmer_decay(-0.1)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_mix() {
        assert!(ShimmerConfig::new().with_mix(-0.1).validate().is_err());
        assert!(ShimmerConfig::new().with_mix(1.1).validate().is_err());
        assert!(ShimmerConfig::new()
            .with_mix(f32::INFINITY)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_brightness() {
        assert!(ShimmerConfig::new()
            .with_brightness(-0.1)
            .validate()
            .is_err());
        assert!(ShimmerConfig::new()
            .with_brightness(1.1)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_silk_freq() {
        assert!(ShimmerConfig::new()
            .with_silk_freq_hz(1000.0)
            .validate()
            .is_err());
        assert!(ShimmerConfig::new()
            .with_silk_freq_hz(8000.0)
            .validate()
            .is_err());
    }

    #[test]
    fn test_presets_valid() {
        ShimmerConfig::subtle().validate().expect("subtle valid");
        ShimmerConfig::sparkle().validate().expect("sparkle valid");
        ShimmerConfig::ethereal()
            .validate()
            .expect("ethereal valid");
        ShimmerConfig::broadcast()
            .validate()
            .expect("broadcast valid");
    }

    // --- Processor behavior ---

    #[test]
    fn test_mix_zero_is_noop() {
        let mut buf = sine_wave(1000.0, 4096, 0.5);
        let original = buf.clone();
        let cfg = ShimmerConfig::new().with_mix(0.0);
        let mut proc = ShimmerProcessor::new_kokoro(&cfg).expect("valid");
        proc.process_voice(&mut buf);
        assert_eq!(buf, original, "mix=0 should be identity");
    }

    #[test]
    fn test_air_boost_increases_hf_energy() {
        let n = 8192;
        // Broadband signal with energy across the spectrum.
        let mut buf: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR;
                0.2 * (2.0 * std::f32::consts::PI * 500.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 5000.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 10000.0 * t).sin()
            })
            .collect();
        let dry_hf = hf_energy(&buf, 8000.0);

        let cfg = ShimmerConfig::new()
            .with_air_gain_db(6.0)
            .with_shimmer_amount(0.0)
            .with_silk_gain_db(0.0)
            .with_brightness(0.5)
            .with_mix(1.0);
        let mut proc = ShimmerProcessor::new_kokoro(&cfg).expect("valid");
        proc.process_voice(&mut buf);
        let wet_hf = hf_energy(&buf, 8000.0);

        assert!(
            wet_hf > dry_hf,
            "air boost should increase HF energy: dry={dry_hf}, wet={wet_hf}",
        );
    }

    #[test]
    fn test_shimmer_adds_harmonic_content() {
        let n = 8192;
        let mut buf = sine_wave(1000.0, n, 0.3);
        let dry_hf = hf_energy(&buf, 6000.0);

        let cfg = ShimmerConfig::new()
            .with_air_gain_db(0.0)
            .with_silk_gain_db(0.0)
            .with_shimmer_amount(0.8)
            .with_shimmer_decay(0.3)
            .with_brightness(0.5)
            .with_mix(1.0);
        let mut proc = ShimmerProcessor::new_kokoro(&cfg).expect("valid");
        proc.process_voice(&mut buf);
        let wet_hf = hf_energy(&buf, 6000.0);

        assert!(
            wet_hf > dry_hf,
            "shimmer should add HF harmonic content: dry={dry_hf}, wet={wet_hf}",
        );
    }

    #[test]
    fn test_brightness_bright_boosts_hf() {
        let n = 8192;
        let source: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR;
                0.3 * (2.0 * std::f32::consts::PI * 2000.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 10000.0 * t).sin()
            })
            .collect();

        // Dark: brightness = 0.0 (= -3 dB shelf)
        let mut dark = source.clone();
        let dark_cfg = ShimmerConfig::new()
            .with_air_gain_db(0.0)
            .with_silk_gain_db(0.0)
            .with_shimmer_amount(0.0)
            .with_brightness(0.0)
            .with_mix(1.0);
        let mut dark_proc = ShimmerProcessor::new_kokoro(&dark_cfg).expect("valid");
        dark_proc.process_voice(&mut dark);
        let dark_hf = hf_energy(&dark, 8000.0);

        // Bright: brightness = 1.0 (= +3 dB shelf)
        let mut bright = source;
        let bright_cfg = ShimmerConfig::new()
            .with_air_gain_db(0.0)
            .with_silk_gain_db(0.0)
            .with_shimmer_amount(0.0)
            .with_brightness(1.0)
            .with_mix(1.0);
        let mut bright_proc = ShimmerProcessor::new_kokoro(&bright_cfg).expect("valid");
        bright_proc.process_voice(&mut bright);
        let bright_hf = hf_energy(&bright, 8000.0);

        assert!(
            bright_hf > dark_hf * 1.1,
            "brightness=1.0 should have more HF than 0.0: \
             bright={bright_hf}, dark={dark_hf}",
        );
    }

    #[test]
    fn test_all_outputs_finite() {
        let inputs: Vec<f32> = vec![
            0.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            0.001,
            -0.001,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let cfg = ShimmerConfig::new()
            .with_shimmer_amount(1.0)
            .with_shimmer_decay(0.9)
            .with_air_gain_db(6.0)
            .with_mix(1.0);
        let mut proc = ShimmerProcessor::new_kokoro(&cfg).expect("valid");
        let mut buf = inputs;
        proc.process_voice(&mut buf);
        for (i, &v) in buf.iter().enumerate() {
            assert!(v.is_finite(), "sample {i} is non-finite: {v}");
        }
    }

    #[test]
    fn test_shimmer_feedback_does_not_explode() {
        // Long buffer with high feedback decay to test stability.
        let n = 48000; // 2 seconds at 24 kHz
        let mut buf = sine_wave(440.0, n, 0.5);
        let cfg = ShimmerConfig::new()
            .with_shimmer_amount(0.9)
            .with_shimmer_decay(0.9)
            .with_mix(1.0);
        let mut proc = ShimmerProcessor::new_kokoro(&cfg).expect("valid");
        proc.process_voice(&mut buf);
        let max_abs = buf.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        assert!(
            max_abs < 10.0,
            "shimmer feedback should not explode: max_abs={max_abs}",
        );
        assert!(
            buf.iter().all(|x| x.is_finite()),
            "all samples should be finite after high-feedback shimmer",
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = ShimmerConfig::new().with_shimmer_amount(0.5).with_mix(0.5);
        let mut proc = ShimmerProcessor::new_kokoro(&cfg).expect("valid");
        let mut buf = vec![0.5; 200];
        proc.process_voice(&mut buf);
        proc.reset();
        assert_eq!(proc.shimmer_feedback, 0.0);
        assert_eq!(proc.air_eq.x1, 0.0);
        assert_eq!(proc.air_eq.y1, 0.0);
        assert_eq!(proc.silk_eq.x1, 0.0);
        assert_eq!(proc.silk_eq.y1, 0.0);
        assert_eq!(proc.brightness_shelf.x_prev, 0.0);
        assert_eq!(proc.brightness_shelf.y_prev, 0.0);
        assert_eq!(proc.shimmer_hp.x_prev, 0.0);
        assert_eq!(proc.shimmer_hp.y_prev, 0.0);
    }

    #[test]
    fn test_invalid_sample_rate() {
        let cfg = ShimmerConfig::new();
        assert!(ShimmerProcessor::new(&cfg, 0.0).is_err());
        assert!(ShimmerProcessor::new(&cfg, -44100.0).is_err());
        assert!(ShimmerProcessor::new(&cfg, f32::NAN).is_err());
    }

    #[test]
    fn test_empty_buffer() {
        let cfg = ShimmerConfig::new();
        let mut proc = ShimmerProcessor::new_kokoro(&cfg).expect("valid");
        let mut buf: Vec<f32> = vec![];
        proc.process_voice(&mut buf);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_silk_boost_increases_presence_energy() {
        let n = 8192;
        // Signal at the silk frequency.
        let mut buf = sine_wave(3500.0, n, 0.3);
        let dry_rms = rms(&buf);

        let cfg = ShimmerConfig::new()
            .with_air_gain_db(0.0)
            .with_silk_gain_db(6.0)
            .with_shimmer_amount(0.0)
            .with_brightness(0.5)
            .with_mix(1.0);
        let mut proc = ShimmerProcessor::new_kokoro(&cfg).expect("valid");
        proc.process_voice(&mut buf);
        let wet_rms = rms(&buf);

        assert!(
            wet_rms > dry_rms * 1.01,
            "silk boost should increase energy at 3.5 kHz: \
             dry={dry_rms}, wet={wet_rms}",
        );
    }

    #[test]
    fn test_granular_pitch_shifter_produces_output() {
        let mut shifter = GranularPitchShifter::new(1.0, SR);
        let input = sine_wave(440.0, 2048, 0.5);
        let output: Vec<f32> = input.iter().map(|&x| shifter.process(x)).collect();
        let out_rms = rms(&output);
        assert!(
            out_rms > 0.01,
            "pitch shifter should produce non-trivial output: rms={out_rms}",
        );
        assert!(
            output.iter().all(|x| x.is_finite()),
            "all pitch shifter outputs should be finite",
        );
    }
}
