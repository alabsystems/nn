// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-tap delay and echo for spatial depth in Kokoro chorus.
//!
//! Creates discrete echoes with independent gain, pan, and filtering per tap.
//! Unlike reverb (diffuse reflections), this produces rhythmically precise
//! echoes that can be tempo-synced, filtered, and stereo-panned.
//!
//! # Architecture
//!
//! ```text
//! Input → write to circular buffer
//!       → per-tap readback (delay position, gain, pan, optional filter)
//!       → sum tap outputs → wet stereo signal
//!       → feedback path: wet → high-cut filter → add back to input
//!       → output = (1-mix) * dry + mix * wet
//! ```
//!
//! # References
//!
//! - Zolzer, U. (2011). "DAFX: Digital Audio Effects", 2nd ed., Ch. 2.
//! - Pirkle, W. (2019). "Designing Audio Effect Plugins in C++", Ch. 14.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// One-pole lowpass for feedback filtering
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct OnePoleLP {
    coeff: f32,
    state: f32,
}

impl OnePoleLP {
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let c = (-std::f32::consts::TAU * cutoff_hz / sample_rate).exp();
        Self {
            coeff: c,
            state: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        self.state = x * (1.0 - self.coeff) + self.state * self.coeff;
        if self.state.is_finite() {
            self.state
        } else {
            0.0
        }
    }

    fn reset(&mut self) {
        self.state = 0.0;
    }

    #[allow(dead_code)]
    fn set_cutoff(&mut self, cutoff_hz: f32, sample_rate: f32) {
        self.coeff = (-std::f32::consts::TAU * cutoff_hz / sample_rate).exp();
    }
}

// ---------------------------------------------------------------------------
// One-pole highpass for per-tap filtering
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct OnePoleHP {
    coeff: f32,
    prev_in: f32,
    prev_out: f32,
}

impl OnePoleHP {
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let rc = 1.0 / (std::f32::consts::TAU * cutoff_hz);
        let dt = 1.0 / sample_rate;
        let c = rc / (rc + dt);
        Self {
            coeff: c,
            prev_in: 0.0,
            prev_out: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.coeff * (self.prev_out + x - self.prev_in);
        self.prev_in = x;
        self.prev_out = if y.is_finite() { y } else { 0.0 };
        self.prev_out
    }

    fn reset(&mut self) {
        self.prev_in = 0.0;
        self.prev_out = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Per-tap filter specification
// ---------------------------------------------------------------------------

/// Optional filter applied to a single delay tap's output.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum TapFilter {
    /// One-pole lowpass at the given cutoff (Hz).
    LowPass(f32),
    /// One-pole highpass at the given cutoff (Hz).
    HighPass(f32),
}

// ---------------------------------------------------------------------------
// Delay tap definition
// ---------------------------------------------------------------------------

/// A single tap read from the shared delay buffer.
///
/// Each tap has an independent delay time, gain, stereo pan position,
/// and optional filter. Delay can be specified in milliseconds or in
/// beats (converted to ms via tempo).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DelayTap {
    /// Delay time in milliseconds (or beats if `in_beats` is true).
    pub delay_ms: f32,
    /// Whether `delay_ms` actually represents beats (converted via tempo).
    pub in_beats: bool,
    /// Gain multiplier for this tap. Range: [0.0, 2.0]. Default: 1.0.
    pub gain: f32,
    /// Stereo pan position. -1.0 = hard left, 0.0 = center, 1.0 = hard right.
    pub pan: f32,
    /// Optional per-tap filter.
    pub filter: Option<TapFilter>,
}

impl Default for DelayTap {
    fn default() -> Self {
        Self {
            delay_ms: 250.0,
            in_beats: false,
            gain: 1.0,
            pan: 0.0,
            filter: None,
        }
    }
}

impl DelayTap {
    /// Create a new tap with the given delay in milliseconds.
    #[must_use]
    pub fn ms(delay_ms: f32) -> Self {
        Self {
            delay_ms,
            ..Self::default()
        }
    }

    /// Create a tap specified in beats (requires tempo to resolve).
    #[must_use]
    pub fn beats(beats: f32) -> Self {
        Self {
            delay_ms: beats,
            in_beats: true,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn with_gain(mut self, g: f32) -> Self {
        self.gain = g;
        self
    }
    #[must_use]
    pub fn with_pan(mut self, p: f32) -> Self {
        self.pan = p;
        self
    }
    #[must_use]
    pub fn with_filter(mut self, f: TapFilter) -> Self {
        self.filter = Some(f);
        self
    }

    /// Resolve the actual delay in milliseconds, given an optional tempo.
    fn resolved_delay_ms(&self, tempo_bpm: Option<f32>) -> f32 {
        if self.in_beats {
            let bpm = tempo_bpm.unwrap_or(120.0);
            let beat_ms = 60_000.0 / bpm.max(1.0);
            self.delay_ms * beat_ms
        } else {
            self.delay_ms
        }
    }
}

// ---------------------------------------------------------------------------
// Delay configuration
// ---------------------------------------------------------------------------

/// Configuration for the multi-tap delay effect.
///
/// Built via method chaining. `#[non_exhaustive]` allows future fields.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DelayConfig {
    /// The set of delay taps. At least one required.
    pub taps: Vec<DelayTap>,
    /// Feedback amount (fraction of wet signal fed back). Range: [0.0, 0.95].
    pub feedback: f32,
    /// Wet/dry mix. 0.0 = fully dry, 1.0 = fully wet. Default: 0.2.
    pub mix: f32,
    /// High-cut filter frequency on the feedback path (Hz). Default: 6000.
    pub high_cut_hz: f32,
    /// Optional tempo in BPM for beat-synced taps.
    pub tempo_bpm: Option<f32>,
    /// Sample rate. Default: 24000 (Kokoro).
    pub sample_rate: f32,
}

impl Default for DelayConfig {
    fn default() -> Self {
        Self {
            taps: vec![DelayTap::ms(250.0)],
            feedback: 0.3,
            mix: 0.2,
            high_cut_hz: 6000.0,
            tempo_bpm: None,
            sample_rate: 24000.0,
        }
    }
}

impl DelayConfig {
    /// Create a default configuration with validation.
    pub fn new() -> Result<Self, KokoroError> {
        let c = Self::default();
        c.validate()?;
        Ok(c)
    }

    #[must_use]
    pub fn with_taps(mut self, t: Vec<DelayTap>) -> Self {
        self.taps = t;
        self
    }
    #[must_use]
    pub fn with_feedback(mut self, f: f32) -> Self {
        self.feedback = f;
        self
    }
    #[must_use]
    pub fn with_mix(mut self, m: f32) -> Self {
        self.mix = m;
        self
    }
    #[must_use]
    pub fn with_high_cut_hz(mut self, h: f32) -> Self {
        self.high_cut_hz = h;
        self
    }
    #[must_use]
    pub fn with_tempo_bpm(mut self, t: f32) -> Self {
        self.tempo_bpm = Some(t);
        self
    }
    #[must_use]
    pub fn with_sample_rate(mut self, sr: f32) -> Self {
        self.sample_rate = sr;
        self
    }

    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.taps.is_empty() {
            return Err(KokoroError::InvalidConfig {
                field: "taps",
                reason: "at least one delay tap is required".into(),
            });
        }
        if self.taps.len() > 32 {
            return Err(KokoroError::InvalidConfig {
                field: "taps",
                reason: format!("max 32 taps, got {}", self.taps.len()),
            });
        }
        check_range("feedback", self.feedback, 0.0, 0.95)?;
        check_range("mix", self.mix, 0.0, 1.0)?;
        check_range("high_cut_hz", self.high_cut_hz, 200.0, 20000.0)?;
        if !self.sample_rate.is_finite() || self.sample_rate < 1000.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("must be >= 1000, got {}", self.sample_rate),
            });
        }
        if let Some(bpm) = self.tempo_bpm {
            check_range("tempo_bpm", bpm, 20.0, 999.0)?;
        }
        for (i, tap) in self.taps.iter().enumerate() {
            let ms = tap.resolved_delay_ms(self.tempo_bpm);
            if !ms.is_finite() || !(0.1..=5000.0).contains(&ms) {
                return Err(KokoroError::InvalidConfig {
                    field: "taps[].delay_ms",
                    reason: format!("tap {i}: resolved delay {ms:.1}ms out of [0.1, 5000]"),
                });
            }
            if !tap.gain.is_finite() || tap.gain < 0.0 || tap.gain > 2.0 {
                return Err(KokoroError::InvalidConfig {
                    field: "taps[].gain",
                    reason: format!("tap {i}: gain {} out of [0.0, 2.0]", tap.gain),
                });
            }
            if !tap.pan.is_finite() || tap.pan < -1.0 || tap.pan > 1.0 {
                return Err(KokoroError::InvalidConfig {
                    field: "taps[].pan",
                    reason: format!("tap {i}: pan {} out of [-1.0, 1.0]", tap.pan),
                });
            }
        }
        Ok(())
    }

    // -- Presets ---------------------------------------------------------------

    /// Slapback echo: single short echo (~80ms), no feedback.
    ///
    /// Classic vocal doubling effect. Quick echo with no repeats.
    #[must_use]
    pub fn slapback() -> Self {
        Self {
            taps: vec![DelayTap::ms(80.0).with_gain(0.7).with_pan(0.2)],
            feedback: 0.0,
            mix: 0.25,
            high_cut_hz: 8000.0,
            tempo_bpm: None,
            sample_rate: 24000.0,
        }
    }

    /// Ping-pong delay: alternating L/R echoes with feedback.
    ///
    /// Creates a bouncing stereo echo that adds width and rhythm.
    #[must_use]
    pub fn ping_pong() -> Self {
        Self {
            taps: vec![
                DelayTap::ms(250.0).with_gain(0.8).with_pan(-0.8),
                DelayTap::ms(500.0).with_gain(0.6).with_pan(0.8),
                DelayTap::ms(750.0).with_gain(0.4).with_pan(-0.6),
            ],
            feedback: 0.35,
            mix: 0.25,
            high_cut_hz: 5000.0,
            tempo_bpm: None,
            sample_rate: 24000.0,
        }
    }

    /// Rhythmic delay: multi-tap at musical subdivisions (1/4, 1/8, 1/16).
    ///
    /// Requires tempo to be set for proper sync. Defaults to 120 BPM.
    #[must_use]
    pub fn rhythmic() -> Self {
        Self {
            taps: vec![
                DelayTap::beats(0.25)
                    .with_gain(0.5)
                    .with_pan(-0.3)
                    .with_filter(TapFilter::HighPass(200.0)),
                DelayTap::beats(0.5).with_gain(0.7).with_pan(0.3),
                DelayTap::beats(1.0).with_gain(1.0).with_pan(0.0),
            ],
            feedback: 0.3,
            mix: 0.2,
            high_cut_hz: 6000.0,
            tempo_bpm: Some(120.0),
            sample_rate: 24000.0,
        }
    }

    /// Ambient delay: long diffuse echoes with heavy filtering.
    ///
    /// Creates a shimmering, atmospheric wash behind the dry signal.
    #[must_use]
    pub fn ambient() -> Self {
        Self {
            taps: vec![
                DelayTap::ms(370.0)
                    .with_gain(0.5)
                    .with_pan(-0.4)
                    .with_filter(TapFilter::LowPass(3000.0)),
                DelayTap::ms(530.0)
                    .with_gain(0.4)
                    .with_pan(0.5)
                    .with_filter(TapFilter::LowPass(2500.0)),
                DelayTap::ms(890.0)
                    .with_gain(0.3)
                    .with_pan(-0.2)
                    .with_filter(TapFilter::LowPass(2000.0)),
                DelayTap::ms(1170.0)
                    .with_gain(0.2)
                    .with_pan(0.3)
                    .with_filter(TapFilter::LowPass(1500.0)),
            ],
            feedback: 0.55,
            mix: 0.3,
            high_cut_hz: 4000.0,
            tempo_bpm: None,
            sample_rate: 24000.0,
        }
    }

    /// Haas-effect widener: very short L/R offset for stereo widening.
    ///
    /// Sub-30ms delays fuse with the dry signal perceptually, creating
    /// a wider stereo image without audible echo.
    #[must_use]
    pub fn haas_wide() -> Self {
        Self {
            taps: vec![
                DelayTap::ms(12.0).with_gain(0.9).with_pan(-1.0),
                DelayTap::ms(18.0).with_gain(0.85).with_pan(1.0),
            ],
            feedback: 0.0,
            mix: 0.35,
            high_cut_hz: 10000.0,
            tempo_bpm: None,
            sample_rate: 24000.0,
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
// Per-tap runtime filter state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum TapFilterState {
    None,
    LowPass(OnePoleLP),
    HighPass(OnePoleHP),
}

impl TapFilterState {
    fn from_spec(spec: Option<TapFilter>, sr: f32) -> Self {
        match spec {
            None => Self::None,
            Some(TapFilter::LowPass(hz)) => Self::LowPass(OnePoleLP::new(hz, sr)),
            Some(TapFilter::HighPass(hz)) => Self::HighPass(OnePoleHP::new(hz, sr)),
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        match self {
            Self::None => x,
            Self::LowPass(f) => f.process(x),
            Self::HighPass(f) => f.process(x),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::None => {}
            Self::LowPass(f) => f.reset(),
            Self::HighPass(f) => f.reset(),
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-tap delay processor
// ---------------------------------------------------------------------------

/// Multi-tap delay processor with filtered feedback and stereo panning.
///
/// Maintains a single shared circular delay buffer. Each tap reads from
/// an independent position with gain, pan, and optional filtering.
#[derive(Debug, Clone)]
pub struct MultiTapDelay {
    config: DelayConfig,
    buffer: Vec<f32>,
    write_pos: usize,
    feedback_lp: OnePoleLP,
    tap_filters: Vec<TapFilterState>,
    feedback_accum: f32,
}

impl MultiTapDelay {
    /// Create a new multi-tap delay processor from the given configuration.
    pub fn new(config: &DelayConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let sr = config.sample_rate;

        // Buffer must be large enough for the longest tap + some headroom
        let max_delay_ms = config
            .taps
            .iter()
            .map(|t| t.resolved_delay_ms(config.tempo_bpm))
            .fold(0.0_f32, f32::max);
        // Add extra for feedback repeats: worst case feedback decays over
        // many iterations but the buffer only needs the max single-tap delay.
        let buf_samples = ((max_delay_ms / 1000.0) * sr) as usize + 4;
        // Minimum buffer size to avoid degenerate cases
        let buf_samples = buf_samples.max(16);

        let tap_filters = config
            .taps
            .iter()
            .map(|t| TapFilterState::from_spec(t.filter, sr))
            .collect();

        Ok(Self {
            config: config.clone(),
            buffer: vec![0.0; buf_samples],
            write_pos: 0,
            feedback_lp: OnePoleLP::new(config.high_cut_hz, sr),
            tap_filters,
            feedback_accum: 0.0,
        })
    }

    /// Reset all internal state (buffer, filters, feedback).
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.feedback_lp.reset();
        self.feedback_accum = 0.0;
        for f in &mut self.tap_filters {
            f.reset();
        }
    }

    /// Update the tempo for beat-synced taps.
    ///
    /// Does not reallocate the buffer; if the new tempo causes a tap delay
    /// to exceed the buffer size, the delay is clamped to the buffer length.
    pub fn set_tempo(&mut self, bpm: f32) {
        if bpm.is_finite() && (20.0..=999.0).contains(&bpm) {
            self.config.tempo_bpm = Some(bpm);
        }
    }

    /// Read from the circular buffer with linear interpolation.
    #[inline]
    fn read_interpolated(&self, delay_samples: f32) -> f32 {
        let len = self.buffer.len();
        let d = delay_samples.clamp(0.0, (len - 2) as f32);
        let d_int = d as usize;
        let frac = d - d_int as f32;

        let idx0 = (self.write_pos + len - 1 - d_int) % len;
        let idx1 = (self.write_pos + len - 2 - d_int) % len;

        let s = self.buffer[idx0] * (1.0 - frac) + self.buffer[idx1] * frac;
        if s.is_finite() {
            s
        } else {
            0.0
        }
    }

    /// Process one sample through the delay, returning (wet_left, wet_right).
    fn process_sample(&mut self, input: f32) -> (f32, f32) {
        let sr = self.config.sample_rate;
        let tempo = self.config.tempo_bpm;

        // Write input + filtered feedback to buffer
        let fb_filtered = self.feedback_lp.process(self.feedback_accum);
        let write_val = input + self.config.feedback * fb_filtered;
        self.buffer[self.write_pos] = if write_val.is_finite() {
            write_val
        } else {
            0.0
        };
        self.write_pos = (self.write_pos + 1) % self.buffer.len();

        // Sum all taps with gain and pan
        let mut wet_l = 0.0_f32;
        let mut wet_r = 0.0_f32;
        let mut wet_mono = 0.0_f32;

        for (i, tap) in self.config.taps.iter().enumerate() {
            let delay_ms = tap.resolved_delay_ms(tempo);
            let delay_samples = (delay_ms / 1000.0) * sr;
            let raw = self.read_interpolated(delay_samples);
            let filtered = self.tap_filters[i].process(raw);
            let gained = filtered * tap.gain;

            // Constant-power pan: pan in [-1, 1] mapped to angle [0, pi/2]
            let angle = (tap.pan * 0.5 + 0.5) * std::f32::consts::FRAC_PI_2;
            wet_l += gained * angle.cos();
            wet_r += gained * angle.sin();
            wet_mono += gained;
        }

        // Normalize by tap count to prevent clipping with many taps
        let norm = 1.0 / (self.config.taps.len() as f32).sqrt();
        wet_l *= norm;
        wet_r *= norm;
        self.feedback_accum = wet_mono * norm;

        (wet_l, wet_r)
    }

    /// Process stereo audio buffers in-place.
    ///
    /// Both buffers must have equal length. The delay processes the mono
    /// sum of (left+right)/2, then pans tap outputs to stereo and mixes
    /// with the dry signal.
    pub fn process_stereo(
        &mut self,
        left: &mut [f32],
        right: &mut [f32],
    ) -> Result<(), KokoroError> {
        if left.len() != right.len() {
            return Err(KokoroError::InvalidInput(format!(
                "stereo buffer length mismatch: left={}, right={}",
                left.len(),
                right.len(),
            )));
        }
        let mix = self.config.mix;
        let dry = 1.0 - mix;

        for i in 0..left.len() {
            let mono_in = (left[i] + right[i]) * 0.5;
            let (wl, wr) = self.process_sample(mono_in);
            left[i] = dry * left[i] + mix * wl;
            right[i] = dry * right[i] + mix * wr;
            if !left[i].is_finite() {
                left[i] = 0.0;
            }
            if !right[i].is_finite() {
                right[i] = 0.0;
            }
        }
        Ok(())
    }

    /// Process mono audio, returning (left, right) stereo output.
    #[must_use]
    pub fn process_mono(&mut self, audio: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mix = self.config.mix;
        let dry = 1.0 - mix;
        let mut out_l = Vec::with_capacity(audio.len());
        let mut out_r = Vec::with_capacity(audio.len());

        for &s in audio {
            let (wl, wr) = self.process_sample(s);
            let l = dry * s + mix * wl;
            let r = dry * s + mix * wr;
            out_l.push(if l.is_finite() { l } else { 0.0 });
            out_r.push(if r.is_finite() { r } else { 0.0 });
        }
        (out_l, out_r)
    }

    /// Access the current configuration (read-only).
    #[must_use]
    pub fn config(&self) -> &DelayConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests (extracted per 500-line rule)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "kokoro_chorus_delay_tests.rs"]
mod tests;
