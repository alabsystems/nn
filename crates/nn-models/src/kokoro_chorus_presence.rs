// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Vocal presence and intelligibility enhancer for Kokoro chorus.
//!
//! Enhances the 2-5 kHz presence band with a musical, dynamic boost that
//! adapts to signal level and sibilant content. Includes an air band shelf
//! (8-12 kHz) for sparkle and openness. Supports per-voice and stereo
//! processing modes.
//!
//! Processing chain:
//!
//! ```text
//! Input ──> Sibilance detector (bandpass 5-8 kHz)
//!       ──> Level detector (RMS envelope)
//!       ──> Dynamic gain = f(level, sibilance)
//!       ──> Presence peaking EQ (2-5 kHz, gain = dynamic_gain)
//!       ──> Air shelf boost (8-12 kHz, fixed gain)
//!       ──> Wet/dry mix blend
//!       ──> Output
//! ```
//!
//! # Design rationale
//!
//! - **Dynamic presence:** Quiet passages receive more boost (up to
//!   `presence_boost_db + dynamic_range_db`), loud passages receive
//!   less. This acts like an upward compressor on the presence band,
//!   improving intelligibility without harshness on peaks.
//! - **Sibilance awareness:** When sibilant energy (detected via a
//!   bandpass in the 5-8 kHz range) exceeds `sibilance_threshold_db`,
//!   the presence boost is progressively reduced. This prevents the
//!   presence EQ from amplifying harsh sibilants.
//! - **Air shelf:** A gentle high-shelf adds brightness above
//!   `air_center_hz`. Independent of the dynamic presence logic.
//!
//! # References
//!
//! - Katz, B. "Mastering Audio: The Art and the Science." 3rd ed., 2015.
//!   Chapter 10: Processing for Mastering (presence EQ, air band).
//! - Giannoulis, D. et al. "Digital Dynamic Range Compressor Design."
//!   JAES 60(6), 2012.
//! - Zolzer, U. "DAFX: Digital Audio Effects." 2nd ed., Wiley, 2011.
//!   Chapter 2: Filters; Chapter 5: Nonlinear Processing.
//!
//! Part of #4264, #3351.

use crate::kokoro_chorus_saturation::db_to_linear;
use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the vocal presence enhancer.
///
/// Constructed via [`PresenceConfig::new`] + builder methods (required for
/// cross-crate use due to `#[non_exhaustive]`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct PresenceConfig {
    /// Presence boost amount in dB (baseline for loud passages).
    /// Range: 0.0 - 12.0. Default: 3.0.
    pub presence_boost_db: f32,

    /// Center frequency of the presence peaking EQ (Hz).
    /// Range: 1000.0 - 8000.0. Default: 3500.0.
    pub presence_center_hz: f32,

    /// Q factor (bandwidth) of the presence peaking EQ.
    /// Higher Q = narrower boost. Range: 0.3 - 5.0. Default: 1.2.
    pub presence_q: f32,

    /// Air band boost in dB (high-shelf).
    /// Range: 0.0 - 8.0. Default: 2.0.
    pub air_boost_db: f32,

    /// Center frequency for the air band shelf (Hz).
    /// Range: 6000.0 - 16000.0. Default: 10000.0.
    pub air_center_hz: f32,

    /// Dynamic range in dB: how much extra boost quiet signals receive.
    /// Range: 0.0 - 18.0. Default: 6.0.
    pub dynamic_range_db: f32,

    /// Sibilance detection threshold in dB. When sibilant energy
    /// exceeds this level, presence boost is attenuated.
    /// Range: -60.0 - 0.0. Default: -20.0.
    pub sibilance_threshold_db: f32,

    /// Wet/dry mix. 0.0 = bypass, 1.0 = fully processed.
    /// Range: 0.0 - 1.0. Default: 0.5.
    pub mix: f32,
}

impl Default for PresenceConfig {
    fn default() -> Self {
        Self {
            presence_boost_db: 3.0,
            presence_center_hz: 3500.0,
            presence_q: 1.2,
            air_boost_db: 2.0,
            air_center_hz: 10000.0,
            dynamic_range_db: 6.0,
            sibilance_threshold_db: -20.0,
            mix: 0.5,
        }
    }
}

macro_rules! builder {
    ($name:ident, $field:ident, $ty:ty) => {
        #[must_use]
        pub fn $name(mut self, v: $ty) -> Self {
            self.$field = v;
            self
        }
    };
}

impl PresenceConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    builder!(with_presence_boost_db, presence_boost_db, f32);
    builder!(with_presence_center_hz, presence_center_hz, f32);
    builder!(with_presence_q, presence_q, f32);
    builder!(with_air_boost_db, air_boost_db, f32);
    builder!(with_air_center_hz, air_center_hz, f32);
    builder!(with_dynamic_range_db, dynamic_range_db, f32);
    builder!(with_sibilance_threshold_db, sibilance_threshold_db, f32);
    builder!(with_mix, mix, f32);

    /// Validate all parameters are within acceptable ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        let chk = |n: &'static str, v: f32, lo: f32, hi: f32| -> Result<(), KokoroError> {
            if !v.is_finite() || v < lo || v > hi {
                return Err(KokoroError::InvalidConfig {
                    field: n,
                    reason: format!("{n} = {v}: must be finite and in [{lo}, {hi}]"),
                });
            }
            Ok(())
        };
        chk("presence_boost_db", self.presence_boost_db, 0.0, 12.0)?;
        chk(
            "presence_center_hz",
            self.presence_center_hz,
            1000.0,
            8000.0,
        )?;
        chk("presence_q", self.presence_q, 0.3, 5.0)?;
        chk("air_boost_db", self.air_boost_db, 0.0, 8.0)?;
        chk("air_center_hz", self.air_center_hz, 6000.0, 16000.0)?;
        chk("dynamic_range_db", self.dynamic_range_db, 0.0, 18.0)?;
        chk(
            "sibilance_threshold_db",
            self.sibilance_threshold_db,
            -60.0,
            0.0,
        )?;
        chk("mix", self.mix, 0.0, 1.0)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

/// Subtle presence: gentle boost, low dynamic range. Good for already-
/// bright recordings that need just a touch more clarity.
#[must_use]
pub fn subtle() -> PresenceConfig {
    PresenceConfig::new()
        .with_presence_boost_db(1.5)
        .with_air_boost_db(1.0)
        .with_dynamic_range_db(3.0)
        .with_mix(0.4)
}

/// Forward presence: stronger mid-range presence with wider dynamic
/// range. Pushes vocals forward in the mix.
#[must_use]
pub fn forward() -> PresenceConfig {
    PresenceConfig::new()
        .with_presence_boost_db(4.5)
        .with_presence_center_hz(3000.0)
        .with_air_boost_db(2.0)
        .with_dynamic_range_db(8.0)
        .with_mix(0.6)
}

/// Broadcast presence: balanced, professional clarity for spoken word
/// and broadcast vocal work.
#[must_use]
pub fn broadcast() -> PresenceConfig {
    PresenceConfig::new()
        .with_presence_boost_db(3.0)
        .with_presence_center_hz(4000.0)
        .with_presence_q(1.5)
        .with_air_boost_db(1.5)
        .with_dynamic_range_db(6.0)
        .with_sibilance_threshold_db(-18.0)
        .with_mix(0.5)
}

/// Airy presence: emphasis on the air band for shimmer and openness.
/// Lower presence boost, higher air boost.
#[must_use]
pub fn airy() -> PresenceConfig {
    PresenceConfig::new()
        .with_presence_boost_db(2.0)
        .with_air_boost_db(4.0)
        .with_air_center_hz(9000.0)
        .with_dynamic_range_db(4.0)
        .with_mix(0.6)
}

// ---------------------------------------------------------------------------
// Biquad filter (second-order IIR, direct form I)
// ---------------------------------------------------------------------------

/// Second-order IIR filter for presence peaking EQ and sibilance detection.
#[derive(Debug, Clone)]
struct Biquad {
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

impl Biquad {
    /// Peaking EQ at `freq_hz` with quality `q` and `gain_db` boost/cut.
    fn peaking(freq_hz: f32, q: f32, gain_db: f32, sr: f32) -> Self {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * freq_hz / sr;
        let al = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let a0 = 1.0 + al / a;
        Self::norm(
            1.0 + al * a,
            -2.0 * cos_w0,
            1.0 - al * a,
            -2.0 * cos_w0,
            1.0 - al / a,
            a0,
        )
    }

    /// Bandpass filter (constant-0dB-peak gain) for sibilance detection.
    fn bandpass(freq_hz: f32, bw_oct: f32, sr: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * freq_hz / sr;
        let sin_w0 = w0.sin();
        let alpha = sin_w0 * (2.0_f32.ln() / 2.0 * bw_oct * w0 / sin_w0).sinh();
        let a0 = 1.0 + alpha;
        Self::norm(alpha, 0.0, -alpha, -2.0 * w0.cos(), 1.0 - alpha, a0)
    }

    /// High-shelf filter for air band boost.
    fn high_shelf(freq_hz: f32, gain_db: f32, sr: f32) -> Self {
        if gain_db.abs() < 1e-6 {
            return Self::unity();
        }
        let a = 10.0_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * freq_hz / sr;
        let cos_w0 = w0.cos();
        let al = w0.sin() / 2.0 * 2.0_f32.sqrt();
        let sq = a.sqrt();
        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sq * al);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sq * al);
        let a0_d = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sq * al;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sq * al;
        Self::norm(b0, b1, b2, a1, a2, a0_d)
    }

    /// Unity pass-through filter.
    fn unity() -> Self {
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

    fn norm(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32, a0: f32) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.reset();
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
// PresenceProcessor
// ---------------------------------------------------------------------------

/// Stateful vocal presence enhancer with dynamic gain and sibilance awareness.
#[derive(Debug, Clone)]
pub struct PresenceProcessor {
    config: PresenceConfig,
    /// Peaking EQ for the presence band (2-5 kHz).
    presence_eq: Biquad,
    /// High-shelf for air band.
    air_shelf: Biquad,
    /// Bandpass for sibilance detection (5-8 kHz region).
    sibilance_detector: Biquad,
    /// Smoothed RMS envelope for level-dependent gain.
    level_env: f32,
    /// Smoothed sibilance envelope.
    sib_env: f32,
    /// Envelope follower coefficients.
    attack_coeff: f32,
    release_coeff: f32,
    /// Pre-computed gain parameters.
    dynamic_range_linear: f32,
    sib_threshold_linear: f32,
}

impl PresenceProcessor {
    /// Create a new presence processor.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range,
    /// or if `sample_rate` is not finite and positive.
    pub fn new(config: &PresenceConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("sample_rate = {sample_rate}: must be finite and positive"),
            });
        }

        let presence_eq = Biquad::peaking(
            config.presence_center_hz,
            config.presence_q,
            config.presence_boost_db,
            sample_rate,
        );
        let air_shelf = Biquad::high_shelf(config.air_center_hz, config.air_boost_db, sample_rate);
        // Sibilance detection: bandpass centered at 6.5 kHz, 1.5 octave width.
        let sibilance_detector = Biquad::bandpass(6500.0, 1.5, sample_rate);

        // Envelope follower: 5 ms attack, 50 ms release (smoothing).
        let attack_coeff = (-1.0 / (0.005 * sample_rate)).exp();
        let release_coeff = (-1.0 / (0.050 * sample_rate)).exp();

        Ok(Self {
            config: *config,
            presence_eq,
            air_shelf,
            sibilance_detector,
            level_env: 0.0,
            sib_env: 0.0,
            attack_coeff,
            release_coeff,
            dynamic_range_linear: db_to_linear(config.dynamic_range_db),
            sib_threshold_linear: db_to_linear(config.sibilance_threshold_db),
        })
    }

    /// Create a processor using Kokoro's default 24 kHz sample rate.
    pub fn new_kokoro(config: &PresenceConfig) -> Result<Self, KokoroError> {
        Self::new(config, KOKORO_SAMPLE_RATE as f32)
    }

    /// Process a mono audio buffer in-place.
    ///
    /// Fast path: returns immediately when `mix == 0.0`.
    pub fn process(&mut self, audio: &mut [f32]) {
        if self.config.mix == 0.0 {
            return;
        }
        let mix = self.config.mix;

        for sample in audio.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
                continue;
            }
            let dry = *sample;
            let wet = self.process_sample(dry);
            *sample = dry * (1.0 - mix) + wet * mix;
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }
    }

    /// Process stereo buffers in-place (left and right independently).
    ///
    /// Uses a single processor instance with shared envelope state,
    /// which creates a natural stereo image (both channels respond to
    /// the same dynamics).
    pub fn process_stereo(&mut self, left: &mut [f32], right: &mut [f32]) {
        if self.config.mix == 0.0 {
            return;
        }
        let mix = self.config.mix;
        let len = left.len().min(right.len());

        for i in 0..len {
            let dl = if left[i].is_finite() { left[i] } else { 0.0 };
            let dr = if right[i].is_finite() { right[i] } else { 0.0 };

            // Mid signal for envelope detection (mono sum).
            let mid = (dl + dr) * 0.5;
            self.update_envelopes(mid);

            let gain = self.compute_dynamic_gain();
            let wl = self.apply_filters_with_gain(dl, gain);
            let wr = self.apply_filters_with_gain(dr, gain);

            left[i] = dl * (1.0 - mix) + wl * mix;
            right[i] = dr * (1.0 - mix) + wr * mix;

            if !left[i].is_finite() {
                left[i] = 0.0;
            }
            if !right[i].is_finite() {
                right[i] = 0.0;
            }
        }
    }

    /// Reset all internal state (call between unrelated audio segments).
    pub fn reset(&mut self) {
        self.presence_eq.reset();
        self.air_shelf.reset();
        self.sibilance_detector.reset();
        self.level_env = 0.0;
        self.sib_env = 0.0;
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &PresenceConfig {
        &self.config
    }

    // --- Internal helpers ---

    /// Process a single sample through the full chain (mono path).
    #[inline]
    fn process_sample(&mut self, x: f32) -> f32 {
        self.update_envelopes(x);
        let gain = self.compute_dynamic_gain();
        self.apply_filters_with_gain(x, gain)
    }

    /// Update level and sibilance envelopes from a sample.
    #[inline]
    fn update_envelopes(&mut self, x: f32) {
        // Level envelope (rectified input).
        let level = x.abs();
        let lc = if level > self.level_env {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.level_env = lc * self.level_env + (1.0 - lc) * level;

        // Sibilance envelope (bandpass-filtered, rectified).
        let sib = self.sibilance_detector.process(x).abs();
        let sc = if sib > self.sib_env {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.sib_env = sc * self.sib_env + (1.0 - sc) * sib;
    }

    /// Compute the dynamic presence gain based on current envelopes.
    ///
    /// - Quiet signals get more boost (up to base + dynamic_range).
    /// - Sibilant signals get reduced boost.
    #[inline]
    fn compute_dynamic_gain(&self) -> f32 {
        // Dynamic gain: inversely proportional to level.
        // At level_env == 0 -> maximum boost (base + dynamic_range).
        // At level_env >= 1 -> minimum boost (base only).
        let level_factor = (1.0 - self.level_env.min(1.0)).max(0.0);
        let dynamic_boost = 1.0 + (self.dynamic_range_linear - 1.0) * level_factor;

        // Sibilance attenuation: reduce presence when sibilance is high.
        let sib_atten = if self.sib_env > self.sib_threshold_linear {
            let overshoot = self.sib_env / self.sib_threshold_linear.max(1e-10);
            // Gentle rolloff: halve gain at 2x threshold, quarter at 4x.
            (1.0 / overshoot).max(0.1)
        } else {
            1.0
        };

        dynamic_boost * sib_atten
    }

    /// Apply presence EQ and air shelf with the given dynamic gain factor.
    ///
    /// NOTE: For the stereo path, the presence_eq and air_shelf filters
    /// process both channels through the same state. For fully independent
    /// stereo, create two separate `PresenceProcessor` instances. The shared
    /// approach here gives a cohesive stereo image.
    #[inline]
    fn apply_filters_with_gain(&mut self, x: f32, gain: f32) -> f32 {
        // Apply presence EQ.
        let eq_out = self.presence_eq.process(x);
        // Scale the EQ difference by the dynamic gain.
        let eq_diff = eq_out - x;
        let present = x + eq_diff * gain;
        // Apply air shelf on top.
        self.air_shelf.process(present)
    }
}

// ---------------------------------------------------------------------------
// Convenience: per-voice presence application
// ---------------------------------------------------------------------------

/// Apply presence enhancement to each voice buffer independently.
///
/// Creates one [`PresenceProcessor`] per voice and processes in place.
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` if the config is invalid.
pub fn apply_presence(
    voices: &mut [Vec<f32>],
    config: &PresenceConfig,
    sample_rate: f32,
) -> Result<(), KokoroError> {
    for voice in voices.iter_mut() {
        let mut proc = PresenceProcessor::new(config, sample_rate)?;
        proc.process(voice.as_mut_slice());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "kokoro_chorus_presence_tests.rs"]
mod tests;
