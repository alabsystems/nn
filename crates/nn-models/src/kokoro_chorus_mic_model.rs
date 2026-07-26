// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Microphone modeling and proximity effect simulation for Kokoro chorus voices.
//!
//! Simulates the acoustic characteristics of different recording setups.
//! Each chorus voice is assigned a virtual "microphone" with distinct
//! characteristics: close-mic'd voices get bass boost (proximity effect),
//! distant voices get natural high-frequency rolloff. This adds realism
//! and depth variety to the chorus.
//!
//! # Per-voice processing chain
//!
//! ```text
//! Input --> Proximity Effect (low-shelf boost) --> Mic Response Curve (EQ)
//!       --> Air Absorption (1-pole LPF) --> Self-Noise --> Output
//! ```
//!
//! # Distance assignment
//!
//! Voice 0 is closest (most proximity boost), voice N-1 is farthest
//! (most air absorption). The `per_voice_distance_spread` parameter
//! controls how much the distances differ across voices.
//!
//! # References
//!
//! - Eargle, J. "The Microphone Book." 2nd ed., Focal Press, 2004.
//! - Winer, E. "The Audio Expert." 2nd ed., Focal Press, 2018.
//! - Olson, H. F. "Acoustical Engineering." Van Nostrand, 1957.
//!   Proximity effect derivation for pressure-gradient transducers.
//!
//! Part of #4582, #3351.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// MicType enum
// ---------------------------------------------------------------------------

/// Virtual microphone model with distinct frequency response characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MicType {
    /// Large-diaphragm condenser: flat response with presence peak at ~10 kHz.
    /// The studio workhorse for vocals.
    Condenser,
    /// Dynamic (moving coil): mid-forward character, rolled-off above ~8 kHz.
    /// Classic live vocal sound (SM58-style).
    Dynamic,
    /// Ribbon: dark, smooth figure-8 response, rolled off above ~6 kHz.
    /// Vintage character with natural transient softening.
    Ribbon,
    /// Tube condenser: warm character with subtle even-harmonic saturation
    /// and gentle compression from the tube amplifier stage.
    Tube,
}

// ---------------------------------------------------------------------------
// MicModelConfig
// ---------------------------------------------------------------------------

/// Configuration for the microphone modeling and proximity effect processor.
///
/// Constructed via [`MicModelConfig::new`] (required for cross-crate use
/// due to `#[non_exhaustive]`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct MicModelConfig {
    /// Microphone model determining the characteristic frequency response.
    /// Default: [`MicType::Condenser`].
    pub mic_type: MicType,
    /// Virtual distance from source to microphone: 0.0 (touching) to
    /// 1.0 (distant). Controls proximity effect strength.
    /// Default: 0.3.
    pub proximity_distance: f32,
    /// Corner frequency (Hz) for the proximity effect low-shelf filter.
    /// Default: 200.0.
    pub proximity_freq_hz: f32,
    /// Maximum bass boost (dB) at distance 0.0 (touching the mic).
    /// Default: 6.0.
    pub proximity_boost_db: f32,
    /// Enable high-frequency air absorption rolloff with distance.
    /// Default: true.
    pub air_absorption: bool,
    /// Microphone self-noise floor in dBFS. Adds subtle analog character.
    /// Default: -80.0 (effectively silent).
    pub self_noise_db: f32,
    /// Spread of distances across voices: 0.0 = all same distance,
    /// 1.0 = maximum spread from closest to farthest.
    /// Default: 0.3.
    pub per_voice_distance_spread: f32,
    /// Sample rate in Hz. Default: 24000.0.
    pub sample_rate: f32,
}

impl Default for MicModelConfig {
    fn default() -> Self {
        Self {
            mic_type: MicType::Condenser,
            proximity_distance: 0.3,
            proximity_freq_hz: 200.0,
            proximity_boost_db: 6.0,
            air_absorption: true,
            self_noise_db: -80.0,
            per_voice_distance_spread: 0.3,
            sample_rate: 24000.0,
        }
    }
}

impl MicModelConfig {
    /// Create a new mic model config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the microphone type.
    #[must_use]
    pub fn with_mic_type(mut self, mic_type: MicType) -> Self {
        self.mic_type = mic_type;
        self
    }

    /// Set the proximity distance (0.0 = touching, 1.0 = distant).
    #[must_use]
    pub fn with_proximity_distance(mut self, distance: f32) -> Self {
        self.proximity_distance = distance;
        self
    }

    /// Set the proximity effect corner frequency in Hz.
    #[must_use]
    pub fn with_proximity_freq_hz(mut self, hz: f32) -> Self {
        self.proximity_freq_hz = hz;
        self
    }

    /// Set the maximum proximity bass boost in dB.
    #[must_use]
    pub fn with_proximity_boost_db(mut self, db: f32) -> Self {
        self.proximity_boost_db = db;
        self
    }

    /// Enable or disable air absorption HF rolloff.
    #[must_use]
    pub fn with_air_absorption(mut self, enable: bool) -> Self {
        self.air_absorption = enable;
        self
    }

    /// Set the self-noise floor in dBFS.
    #[must_use]
    pub fn with_self_noise_db(mut self, db: f32) -> Self {
        self.self_noise_db = db;
        self
    }

    /// Set the per-voice distance spread (0.0 = uniform, 1.0 = max spread).
    #[must_use]
    pub fn with_per_voice_distance_spread(mut self, spread: f32) -> Self {
        self.per_voice_distance_spread = spread;
        self
    }

    /// Set the sample rate in Hz.
    #[must_use]
    pub fn with_sample_rate(mut self, sr: f32) -> Self {
        self.sample_rate = sr;
        self
    }

    /// Validate all parameters are within acceptable ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.proximity_distance.is_finite()
            || self.proximity_distance < 0.0
            || self.proximity_distance > 1.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "proximity_distance",
                reason: format!(
                    "proximity_distance = {}: must be finite and in [0.0, 1.0]",
                    self.proximity_distance,
                ),
            });
        }
        if !self.proximity_freq_hz.is_finite()
            || self.proximity_freq_hz < 50.0
            || self.proximity_freq_hz > 500.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "proximity_freq_hz",
                reason: format!(
                    "proximity_freq_hz = {}: must be finite and in [50, 500]",
                    self.proximity_freq_hz,
                ),
            });
        }
        if !self.proximity_boost_db.is_finite()
            || self.proximity_boost_db < 0.0
            || self.proximity_boost_db > 18.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "proximity_boost_db",
                reason: format!(
                    "proximity_boost_db = {}: must be finite and in [0, 18]",
                    self.proximity_boost_db,
                ),
            });
        }
        if !self.self_noise_db.is_finite()
            || self.self_noise_db < -120.0
            || self.self_noise_db > -40.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "self_noise_db",
                reason: format!(
                    "self_noise_db = {}: must be finite and in [-120, -40]",
                    self.self_noise_db,
                ),
            });
        }
        if !self.per_voice_distance_spread.is_finite()
            || self.per_voice_distance_spread < 0.0
            || self.per_voice_distance_spread > 1.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "per_voice_distance_spread",
                reason: format!(
                    "per_voice_distance_spread = {}: must be finite and in [0.0, 1.0]",
                    self.per_voice_distance_spread,
                ),
            });
        }
        if !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!(
                    "sample_rate = {}: must be finite and positive",
                    self.sample_rate,
                ),
            });
        }
        Ok(())
    }

    // --- Presets ---------------------------------------------------------------

    /// Studio large-diaphragm condenser at moderate distance.
    /// Clean, detailed, with subtle proximity warmth.
    #[must_use]
    pub fn studio_condenser() -> Self {
        Self {
            mic_type: MicType::Condenser,
            proximity_distance: 0.25,
            proximity_freq_hz: 200.0,
            proximity_boost_db: 6.0,
            air_absorption: true,
            self_noise_db: -80.0,
            per_voice_distance_spread: 0.3,
            sample_rate: 24000.0,
        }
    }

    /// Live dynamic mic at close range. Mid-forward, controlled bass.
    #[must_use]
    pub fn live_dynamic() -> Self {
        Self {
            mic_type: MicType::Dynamic,
            proximity_distance: 0.15,
            proximity_freq_hz: 250.0,
            proximity_boost_db: 8.0,
            air_absorption: false,
            self_noise_db: -70.0,
            per_voice_distance_spread: 0.2,
            sample_rate: 24000.0,
        }
    }

    /// Vintage ribbon mic at moderate distance. Dark, smooth, lush.
    #[must_use]
    pub fn vintage_ribbon() -> Self {
        Self {
            mic_type: MicType::Ribbon,
            proximity_distance: 0.35,
            proximity_freq_hz: 180.0,
            proximity_boost_db: 5.0,
            air_absorption: true,
            self_noise_db: -65.0,
            per_voice_distance_spread: 0.25,
            sample_rate: 24000.0,
        }
    }

    /// Close intimate mic. Maximum proximity effect, minimal distance.
    #[must_use]
    pub fn close_and_intimate() -> Self {
        Self {
            mic_type: MicType::Tube,
            proximity_distance: 0.05,
            proximity_freq_hz: 220.0,
            proximity_boost_db: 10.0,
            air_absorption: false,
            self_noise_db: -72.0,
            per_voice_distance_spread: 0.1,
            sample_rate: 24000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// One-pole lowpass filter (air absorption, proximity shelf)
// ---------------------------------------------------------------------------

/// Single-pole IIR lowpass: H(z) = b / (1 - a * z^-1).
#[derive(Debug, Clone)]
struct OnePoleLP {
    a: f32,
    b: f32,
    z1: f32,
}

impl OnePoleLP {
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let w = (-2.0 * std::f32::consts::PI * cutoff_hz / sample_rate).exp();
        Self {
            a: w,
            b: 1.0 - w,
            z1: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.z1 = 0.0;
            return 0.0;
        }
        let y = self.b * x + self.a * self.z1;
        self.z1 = if y.is_finite() { y } else { 0.0 };
        self.z1
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Low-shelf filter (proximity effect)
// ---------------------------------------------------------------------------

/// First-order low-shelf filter for proximity effect bass boost.
///
/// Based on the Audio EQ Cookbook (Robert Bristow-Johnson), simplified
/// to first order for efficiency. Provides a smooth shelf boost below
/// the corner frequency.
#[derive(Debug, Clone)]
struct LowShelf {
    b0: f32,
    b1: f32,
    a1: f32,
    x1: f32,
    y1: f32,
}

impl LowShelf {
    /// Create a low-shelf filter with `gain_db` boost below `freq_hz`.
    fn new(freq_hz: f32, gain_db: f32, sample_rate: f32) -> Self {
        if gain_db.abs() < 0.01 {
            return Self::passthrough();
        }

        // First-order shelf: derived from bilinear transform of
        // H(s) = (s + w0*sqrt(A)) / (s + w0/sqrt(A))
        let a = 10.0_f32.powf(gain_db / 40.0); // sqrt of linear gain
        let w0 = 2.0 * std::f32::consts::PI * freq_hz / sample_rate;
        let tan_w0_2 = (w0 / 2.0).tan();

        let sqrt_a = a.sqrt();
        let num = tan_w0_2 * sqrt_a;
        let den = tan_w0_2 / sqrt_a;

        let b0 = (1.0 + num) / (1.0 + den);
        let b1 = (-1.0 + num) / (1.0 + den);
        let a1 = (-1.0 + den) / (1.0 + den);

        Self {
            b0,
            b1,
            a1,
            x1: 0.0,
            y1: 0.0,
        }
    }

    fn passthrough() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            a1: 0.0,
            x1: 0.0,
            y1: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.x1 = 0.0;
            self.y1 = 0.0;
            return 0.0;
        }
        let y = self.b0 * x + self.b1 * self.x1 - self.a1 * self.y1;
        self.x1 = x;
        self.y1 = if y.is_finite() { y } else { 0.0 };
        self.y1
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.y1 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Peaking EQ (mic response curve)
// ---------------------------------------------------------------------------

/// Second-order peaking (bell) EQ filter for mic response shaping.
#[derive(Debug, Clone)]
struct PeakingEQ {
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

impl PeakingEQ {
    fn new(freq_hz: f32, gain_db: f32, q: f32, sample_rate: f32) -> Self {
        if gain_db.abs() < 0.01 {
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
// Per-voice mic chain state
// ---------------------------------------------------------------------------

/// Filter state for a single voice's mic modeling chain.
#[derive(Debug, Clone)]
struct VoiceMicChain {
    /// Effective distance for this voice (0.0 = closest, 1.0 = farthest).
    distance: f32,
    /// Low-shelf filter for proximity effect.
    proximity_shelf: LowShelf,
    /// Mic response EQ (presence peak or HF rolloff).
    mic_response: PeakingEQ,
    /// Air absorption LPF (cutoff decreases with distance).
    air_lpf: OnePoleLP,
    /// Self-noise level (linear amplitude).
    noise_level: f32,
    /// Simple noise PRNG state (xorshift32).
    noise_state: u32,
}

impl VoiceMicChain {
    fn new(voice_idx: usize, n_voices: usize, config: &MicModelConfig) -> Self {
        let sr = config.sample_rate;

        // Compute per-voice distance: voice 0 is closest, voice N-1 farthest.
        let distance = compute_voice_distance(
            voice_idx,
            n_voices,
            config.proximity_distance,
            config.per_voice_distance_spread,
        );

        // Proximity effect: bass boost proportional to (1.0 - distance).
        let prox_gain_db = config.proximity_boost_db * (1.0 - distance);
        let proximity_shelf = LowShelf::new(config.proximity_freq_hz, prox_gain_db, sr);

        // Mic response curve per type.
        let mic_response = build_mic_response(config.mic_type, sr);

        // Air absorption: LPF cutoff decreasing with distance.
        // At distance 0 -> cutoff near Nyquist (no absorption).
        // At distance 1 -> cutoff at ~4 kHz (significant HF loss).
        let air_cutoff = if config.air_absorption {
            let max_cutoff = sr * 0.48;
            let min_cutoff = 4000.0_f32.min(sr * 0.3);
            max_cutoff - distance * (max_cutoff - min_cutoff)
        } else {
            sr * 0.49 // near Nyquist = effectively bypass
        };
        let air_lpf = OnePoleLP::new(air_cutoff, sr);

        // Self-noise: convert dBFS to linear amplitude.
        let noise_level = 10.0_f32.powf(config.self_noise_db / 20.0);

        // Seed PRNG with voice index for reproducible but varied noise.
        let noise_state = (voice_idx as u32).wrapping_mul(2654435761).wrapping_add(1);

        Self {
            distance,
            proximity_shelf,
            mic_response,
            air_lpf,
            noise_level,
            noise_state,
        }
    }

    /// Process one sample through the full mic chain.
    #[inline]
    fn process_sample(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            return 0.0;
        }

        // 1. Proximity effect (low-shelf boost).
        let s = self.proximity_shelf.process(x);

        // 2. Mic response curve (peaking EQ).
        let s = self.mic_response.process(s);

        // 3. Air absorption (LPF).
        let s = self.air_lpf.process(s);

        // 4. Self-noise (very low-level dithering noise).
        let noise = self.next_noise() * self.noise_level;
        let s = s + noise;

        if s.is_finite() {
            s
        } else {
            0.0
        }
    }

    /// Generate the next noise sample via xorshift32 PRNG, range [-1, 1].
    #[inline]
    fn next_noise(&mut self) -> f32 {
        // xorshift32
        let mut s = self.noise_state;
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        self.noise_state = s;
        // Map u32 to [-1.0, 1.0].
        (s as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }

    fn reset(&mut self) {
        self.proximity_shelf.reset();
        self.mic_response.reset();
        self.air_lpf.reset();
    }
}

// ---------------------------------------------------------------------------
// Distance computation
// ---------------------------------------------------------------------------

/// Compute effective distance for a given voice in the ensemble.
///
/// Voice 0 gets the closest distance, voice N-1 gets the farthest.
/// `base_distance` is the center distance, `spread` widens the range.
fn compute_voice_distance(
    voice_idx: usize,
    n_voices: usize,
    base_distance: f32,
    spread: f32,
) -> f32 {
    if n_voices <= 1 {
        return base_distance.clamp(0.0, 1.0);
    }

    // Normalized position: 0.0 for voice 0, 1.0 for voice N-1.
    let t = voice_idx as f32 / (n_voices - 1) as f32;

    // Distance range: [base - spread/2, base + spread/2], clamped to [0, 1].
    let half_spread = spread / 2.0;
    let min_dist = (base_distance - half_spread).clamp(0.0, 1.0);
    let max_dist = (base_distance + half_spread).clamp(0.0, 1.0);

    min_dist + t * (max_dist - min_dist)
}

// ---------------------------------------------------------------------------
// Mic response curve builders
// ---------------------------------------------------------------------------

/// Build the characteristic EQ for each mic type.
fn build_mic_response(mic_type: MicType, sample_rate: f32) -> PeakingEQ {
    match mic_type {
        // Condenser: presence peak at 10 kHz, +3 dB, Q=1.0
        MicType::Condenser => {
            let freq = 10000.0_f32.min(sample_rate * 0.45);
            PeakingEQ::new(freq, 3.0, 1.0, sample_rate)
        }
        // Dynamic: -3 dB shelf-like rolloff above 8 kHz (Q=0.7 for broad)
        MicType::Dynamic => {
            let freq = 8000.0_f32.min(sample_rate * 0.45);
            PeakingEQ::new(freq, -3.0, 0.7, sample_rate)
        }
        // Ribbon: -6 dB above 6 kHz (Q=0.5 for very broad)
        MicType::Ribbon => {
            let freq = 6000.0_f32.min(sample_rate * 0.45);
            PeakingEQ::new(freq, -6.0, 0.5, sample_rate)
        }
        // Tube: slight warmth boost at 300 Hz (+2 dB) + presence at 5 kHz
        // We apply the warmth boost here; tube saturation is handled
        // via the waveshaper below.
        MicType::Tube => PeakingEQ::new(300.0, 2.0, 0.8, sample_rate),
    }
}

// ---------------------------------------------------------------------------
// Tube saturation (MicType::Tube only)
// ---------------------------------------------------------------------------

/// Subtle even-harmonic saturation for tube mic modeling.
///
/// Asymmetric soft clipping: positive half driven slightly harder.
/// Applied only for `MicType::Tube`.
#[inline]
fn tube_saturate(x: f32) -> f32 {
    // Very mild drive for subtle character.
    let drive = 1.3;
    if x >= 0.0 {
        let d = x * drive * 1.1;
        d / (1.0 + d.abs())
    } else {
        let d = x * drive * 0.9;
        d / (1.0 + d.abs())
    }
}

// ---------------------------------------------------------------------------
// MicModelProcessor
// ---------------------------------------------------------------------------

/// Stateful microphone modeling processor for multi-voice chorus.
///
/// Holds per-voice filter chains that model proximity effect, mic response
/// curves, air absorption, and self-noise.
#[derive(Debug, Clone)]
pub struct MicModelProcessor {
    config: MicModelConfig,
    chains: Vec<VoiceMicChain>,
}

impl MicModelProcessor {
    /// Create a new mic model processor for `n_voices` voices.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range.
    pub fn new(config: MicModelConfig, n_voices: usize) -> Result<Self, KokoroError> {
        config.validate()?;

        let chains = (0..n_voices)
            .map(|i| VoiceMicChain::new(i, n_voices, &config))
            .collect();

        Ok(Self { config, chains })
    }

    /// Process per-voice audio buffers in-place.
    ///
    /// `voices` must have the same length as `n_voices` passed to the
    /// constructor. Voices with mismatched count are processed up to
    /// the minimum of `voices.len()` and the internal chain count.
    pub fn process_voices(&mut self, voices: &mut [Vec<f32>]) {
        let is_tube = self.config.mic_type == MicType::Tube;

        for (voice, chain) in voices.iter_mut().zip(self.chains.iter_mut()) {
            for sample in voice.iter_mut() {
                let s = chain.process_sample(*sample);

                // Tube mic: apply subtle even-harmonic saturation.
                let s = if is_tube { tube_saturate(s) } else { s };

                *sample = if s.is_finite() { s } else { 0.0 };
            }
        }
    }

    /// Reset all internal filter state (call between unrelated segments).
    pub fn reset(&mut self) {
        for chain in &mut self.chains {
            chain.reset();
        }
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &MicModelConfig {
        &self.config
    }

    /// Get the effective distance for a specific voice index.
    ///
    /// Returns `None` if `voice_idx >= n_voices`.
    #[must_use]
    pub fn voice_distance(&self, voice_idx: usize) -> Option<f32> {
        self.chains.get(voice_idx).map(|c| c.distance)
    }

    /// Number of voice chains in this processor.
    #[must_use]
    pub fn n_voices(&self) -> usize {
        self.chains.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 24000.0;

    fn sine_wave(freq: f32, n: usize, amplitude: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / SR).sin())
            .collect()
    }

    fn rms(buf: &[f32]) -> f32 {
        let sum_sq: f32 = buf.iter().map(|x| x * x).sum();
        (sum_sq / buf.len().max(1) as f32).sqrt()
    }

    // --- Config validation ---

    #[test]
    fn test_config_default_valid() {
        MicModelConfig::new()
            .validate()
            .expect("default config should be valid");
    }

    #[test]
    fn test_config_builder_roundtrip() {
        let cfg = MicModelConfig::new()
            .with_mic_type(MicType::Dynamic)
            .with_proximity_distance(0.5)
            .with_proximity_freq_hz(250.0)
            .with_proximity_boost_db(8.0)
            .with_air_absorption(false)
            .with_self_noise_db(-70.0)
            .with_per_voice_distance_spread(0.5)
            .with_sample_rate(48000.0);
        cfg.validate().expect("builder config should be valid");
        assert_eq!(cfg.mic_type, MicType::Dynamic);
        assert_eq!(cfg.proximity_distance, 0.5);
        assert_eq!(cfg.proximity_freq_hz, 250.0);
        assert_eq!(cfg.proximity_boost_db, 8.0);
        assert!(!cfg.air_absorption);
        assert_eq!(cfg.self_noise_db, -70.0);
        assert_eq!(cfg.per_voice_distance_spread, 0.5);
        assert_eq!(cfg.sample_rate, 48000.0);
    }

    #[test]
    fn test_config_invalid_proximity_distance() {
        assert!(MicModelConfig::new()
            .with_proximity_distance(1.5)
            .validate()
            .is_err());
        assert!(MicModelConfig::new()
            .with_proximity_distance(-0.1)
            .validate()
            .is_err());
        assert!(MicModelConfig::new()
            .with_proximity_distance(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_proximity_freq() {
        assert!(MicModelConfig::new()
            .with_proximity_freq_hz(30.0)
            .validate()
            .is_err());
        assert!(MicModelConfig::new()
            .with_proximity_freq_hz(600.0)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_proximity_boost() {
        assert!(MicModelConfig::new()
            .with_proximity_boost_db(-1.0)
            .validate()
            .is_err());
        assert!(MicModelConfig::new()
            .with_proximity_boost_db(20.0)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_self_noise() {
        assert!(MicModelConfig::new()
            .with_self_noise_db(-130.0)
            .validate()
            .is_err());
        assert!(MicModelConfig::new()
            .with_self_noise_db(-30.0)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_spread() {
        assert!(MicModelConfig::new()
            .with_per_voice_distance_spread(-0.1)
            .validate()
            .is_err());
        assert!(MicModelConfig::new()
            .with_per_voice_distance_spread(1.5)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_sample_rate() {
        assert!(MicModelConfig::new()
            .with_sample_rate(0.0)
            .validate()
            .is_err());
        assert!(MicModelConfig::new()
            .with_sample_rate(-1.0)
            .validate()
            .is_err());
        assert!(MicModelConfig::new()
            .with_sample_rate(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_presets_valid() {
        MicModelConfig::studio_condenser()
            .validate()
            .expect("studio_condenser");
        MicModelConfig::live_dynamic()
            .validate()
            .expect("live_dynamic");
        MicModelConfig::vintage_ribbon()
            .validate()
            .expect("vintage_ribbon");
        MicModelConfig::close_and_intimate()
            .validate()
            .expect("close_and_intimate");
    }

    // --- Distance assignment ---

    #[test]
    fn test_voice_distances_monotonic() {
        let cfg = MicModelConfig::new().with_per_voice_distance_spread(0.5);
        let proc = MicModelProcessor::new(cfg, 5).expect("valid");
        let distances: Vec<f32> = (0..5).map(|i| proc.voice_distance(i).unwrap()).collect();
        for i in 1..distances.len() {
            assert!(
                distances[i] >= distances[i - 1],
                "distances should be monotonically increasing: {distances:?}",
            );
        }
    }

    #[test]
    fn test_single_voice_uses_base_distance() {
        let cfg = MicModelConfig::new()
            .with_proximity_distance(0.4)
            .with_per_voice_distance_spread(0.5);
        let proc = MicModelProcessor::new(cfg, 1).expect("valid");
        let d = proc.voice_distance(0).unwrap();
        assert!(
            (d - 0.4).abs() < 1e-6,
            "single voice should use base distance: got {d}",
        );
    }

    #[test]
    fn test_zero_spread_uniform_distance() {
        let cfg = MicModelConfig::new()
            .with_proximity_distance(0.3)
            .with_per_voice_distance_spread(0.0);
        let proc = MicModelProcessor::new(cfg, 4).expect("valid");
        for i in 0..4 {
            let d = proc.voice_distance(i).unwrap();
            assert!(
                (d - 0.3).abs() < 1e-6,
                "zero spread: voice {i} distance {d} should be 0.3",
            );
        }
    }

    // --- Processing behavior ---

    #[test]
    fn test_proximity_boosts_bass() {
        // Close voice should have more bass energy than distant voice.
        let cfg = MicModelConfig::new()
            .with_proximity_distance(0.5)
            .with_proximity_boost_db(12.0)
            .with_per_voice_distance_spread(0.8)
            .with_air_absorption(false)
            .with_self_noise_db(-120.0);
        let n = 4096;
        let bass_signal = sine_wave(100.0, n, 0.5);
        let mut voices = vec![bass_signal.clone(), bass_signal];
        let mut proc = MicModelProcessor::new(cfg, 2).expect("valid");
        proc.process_voices(&mut voices);

        let rms_close = rms(&voices[0]);
        let rms_far = rms(&voices[1]);
        assert!(
            rms_close > rms_far,
            "close voice should have more bass: close={rms_close}, far={rms_far}",
        );
    }

    #[test]
    fn test_air_absorption_reduces_hf() {
        // Distant voice with air absorption should have less HF energy.
        let cfg = MicModelConfig::new()
            .with_proximity_distance(0.5)
            .with_proximity_boost_db(0.0) // no prox boost to isolate air effect
            .with_per_voice_distance_spread(0.9)
            .with_air_absorption(true)
            .with_self_noise_db(-120.0);
        let n = 4096;
        let hf_signal = sine_wave(8000.0, n, 0.5);
        let mut voices = vec![hf_signal.clone(), hf_signal];
        let mut proc = MicModelProcessor::new(cfg, 2).expect("valid");
        proc.process_voices(&mut voices);

        let rms_close = rms(&voices[0]);
        let rms_far = rms(&voices[1]);
        assert!(
            rms_close > rms_far * 1.01,
            "air absorption should attenuate distant HF: close={rms_close}, far={rms_far}",
        );
    }

    #[test]
    fn test_all_mic_types_produce_finite_output() {
        let types = [
            MicType::Condenser,
            MicType::Dynamic,
            MicType::Ribbon,
            MicType::Tube,
        ];
        for mic in types {
            let cfg = MicModelConfig::new().with_mic_type(mic);
            let mut proc = MicModelProcessor::new(cfg, 2).expect("valid");
            let mut voices = vec![
                vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.001],
                vec![0.3, -0.3, 0.8, -0.8, 0.0, 0.0],
            ];
            proc.process_voices(&mut voices);
            for (vi, voice) in voices.iter().enumerate() {
                for (si, &s) in voice.iter().enumerate() {
                    assert!(
                        s.is_finite(),
                        "mic={mic:?} voice={vi} sample={si}: non-finite {s}",
                    );
                }
            }
        }
    }

    #[test]
    fn test_nan_input_clamped() {
        let cfg = MicModelConfig::new();
        let mut proc = MicModelProcessor::new(cfg, 1).expect("valid");
        let mut voices = vec![vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5]];
        proc.process_voices(&mut voices);
        for (i, &s) in voices[0].iter().enumerate() {
            assert!(s.is_finite(), "sample {i} should be finite, got {s}");
        }
    }

    #[test]
    fn test_self_noise_adds_signal() {
        // With high self-noise and silence input, output should not be zero.
        let cfg = MicModelConfig::new()
            .with_self_noise_db(-40.0)
            .with_proximity_boost_db(0.0)
            .with_air_absorption(false);
        let mut proc = MicModelProcessor::new(cfg, 1).expect("valid");
        let mut voices = vec![vec![0.0; 1024]];
        proc.process_voices(&mut voices);
        let energy: f32 = voices[0].iter().map(|x| x * x).sum();
        assert!(energy > 0.0, "self-noise should add energy to silent input");
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = MicModelConfig::new();
        let mut proc = MicModelProcessor::new(cfg, 2).expect("valid");
        let mut voices = vec![vec![0.5; 100], vec![0.3; 100]];
        proc.process_voices(&mut voices);
        proc.reset();
        for chain in &proc.chains {
            assert_eq!(chain.proximity_shelf.x1, 0.0);
            assert_eq!(chain.proximity_shelf.y1, 0.0);
            assert_eq!(chain.air_lpf.z1, 0.0);
        }
    }

    #[test]
    fn test_empty_voices() {
        let cfg = MicModelConfig::new();
        let mut proc = MicModelProcessor::new(cfg, 2).expect("valid");
        let mut voices: Vec<Vec<f32>> = vec![vec![], vec![]];
        proc.process_voices(&mut voices);
        assert!(voices[0].is_empty());
        assert!(voices[1].is_empty());
    }

    #[test]
    fn test_tube_saturation_asymmetric() {
        let pos = tube_saturate(0.5);
        let neg = tube_saturate(-0.5);
        assert!(
            (pos.abs() - neg.abs()).abs() > 1e-4,
            "tube saturation should be asymmetric: |{pos}| vs |{neg}|",
        );
    }

    #[test]
    fn test_n_voices_accessor() {
        let cfg = MicModelConfig::new();
        let proc = MicModelProcessor::new(cfg, 6).expect("valid");
        assert_eq!(proc.n_voices(), 6);
    }

    #[test]
    fn test_voice_distance_out_of_range() {
        let cfg = MicModelConfig::new();
        let proc = MicModelProcessor::new(cfg, 3).expect("valid");
        assert!(proc.voice_distance(3).is_none());
        assert!(proc.voice_distance(100).is_none());
    }

    #[test]
    fn test_ribbon_darker_than_condenser() {
        // Ribbon should attenuate HF more than condenser (ribbon rolls off
        // at 6 kHz, condenser boosts at 10 kHz).
        let n = 4096;
        let hf = sine_wave(8000.0, n, 0.5);

        let cfg_ribbon = MicModelConfig::new()
            .with_mic_type(MicType::Ribbon)
            .with_proximity_boost_db(0.0)
            .with_air_absorption(false)
            .with_self_noise_db(-120.0);
        let cfg_cond = MicModelConfig::new()
            .with_mic_type(MicType::Condenser)
            .with_proximity_boost_db(0.0)
            .with_air_absorption(false)
            .with_self_noise_db(-120.0);

        let mut v_ribbon = vec![hf.clone()];
        let mut v_cond = vec![hf];
        let mut p_ribbon = MicModelProcessor::new(cfg_ribbon, 1).expect("valid");
        let mut p_cond = MicModelProcessor::new(cfg_cond, 1).expect("valid");
        p_ribbon.process_voices(&mut v_ribbon);
        p_cond.process_voices(&mut v_cond);

        let rms_ribbon = rms(&v_ribbon[0]);
        let rms_cond = rms(&v_cond[0]);
        assert!(
            rms_ribbon < rms_cond,
            "ribbon should be darker than condenser at 8kHz: ribbon={rms_ribbon}, cond={rms_cond}",
        );
    }
}
