// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Front-to-back depth staging for Kokoro chorus voices.
//!
//! Beyond left-right panning (handled by stereo and spatial modules), this
//! module positions chorus voices in a front-to-back depth field. Close voices
//! sound bright, loud, and dry; distant voices sound darker, quieter, and
//! wetter. The combination creates a 3D soundstage.
//!
//! # Processing per voice
//!
//! 1. **Distance attenuation** — gain reduction proportional to depth position.
//! 2. **Air absorption LPF** — one-pole low-pass whose cutoff decreases with
//!    depth, simulating high-frequency absorption over distance.
//! 3. **Pre-delay** — sample delay proportional to depth, simulating sound
//!    travel time at 343 m/s.
//! 4. **Early reflections** — simple comb-filter contribution scaled by depth
//!    (distant voices are wetter).
//! 5. **Proximity effect** — subtle bass boost for close voices (+2 dB below
//!    200 Hz), simulating cardioid microphone proximity effect.
//!
//! # Presets
//!
//! - [`intimate`](DepthStagingConfig::intimate) — all voices close, minimal spread.
//! - [`studio`](DepthStagingConfig::studio) — moderate depth, balanced.
//! - [`concert_hall`](DepthStagingConfig::concert_hall) — wide spread, lead close,
//!   backing far.
//! - [`cathedral`](DepthStagingConfig::cathedral) — extreme spread, heavy ER.
//!
//! Part of #4582, #3351.

use crate::kokoro_error::KokoroError;

/// Kokoro default sample rate in Hz.
const DEFAULT_SAMPLE_RATE: f32 = 24000.0;

/// Maximum pre-delay buffer size in samples. At 24 kHz, 4800 samples = 200 ms.
/// Far beyond any realistic depth pre-delay.
const MAX_PRE_DELAY_SAMPLES: usize = 4800;

/// Number of comb-filter taps for the simple early-reflection generator.
const ER_COMB_TAPS: usize = 4;

/// Prime-ish delay tap ratios for the ER comb filter, in milliseconds.
/// Chosen to avoid coloration from regular spacing.
const ER_TAP_MS: [f32; ER_COMB_TAPS] = [7.1, 11.3, 17.9, 23.7];

/// Proximity-effect bass boost frequency threshold in Hz.
const PROXIMITY_FREQ_HZ: f32 = 200.0;

/// Proximity-effect boost in dB for the closest voice (depth = 0.0).
const PROXIMITY_BOOST_DB: f32 = 2.0;

// ---------------------------------------------------------------------------
// DepthStagingConfig
// ---------------------------------------------------------------------------

/// Configuration for front-to-back depth staging of chorus voices.
///
/// Controls how voices are distributed in depth and what processing each
/// depth position receives. Built via method chaining on
/// [`DepthStagingConfig::new`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DepthStagingConfig {
    /// Number of voices to position in the depth field.
    pub n_voices: usize,

    /// How spread out voices are in the depth dimension (0.0-1.0).
    ///
    /// - `0.0` = all voices at the same depth.
    /// - `1.0` = maximum spread from front to back.
    ///
    /// Default: `0.6`.
    pub depth_spread: f32,

    /// Depth position of the lead voice (voice 0).
    ///
    /// - `0.0` = closest (front of stage).
    /// - `1.0` = farthest (back of stage).
    ///
    /// Default: `0.1` (close to listener).
    pub lead_voice_depth: f32,

    /// Depth position for backing voices.
    ///
    /// Remaining voices are distributed between `lead_voice_depth` and
    /// this value, scaled by `depth_spread`.
    ///
    /// Default: `0.6`.
    pub backing_voice_depth: f32,

    /// Maximum attenuation in dB for the farthest voice (depth = 1.0).
    ///
    /// Default: `-6.0` dB.
    pub distance_attenuation_db: f32,

    /// Low-pass filter cutoff for the farthest voice in Hz.
    ///
    /// Voices at depth 0.0 use `close_lpf_freq`; voices at depth 1.0
    /// use this value. Intermediate depths interpolate linearly.
    ///
    /// Default: `8000.0` Hz.
    pub distance_lpf_freq: f32,

    /// Low-pass filter cutoff for the closest voice in Hz.
    ///
    /// Default: `20000.0` Hz (essentially transparent).
    pub close_lpf_freq: f32,

    /// Maximum pre-delay in milliseconds for depth perception.
    ///
    /// The farthest voice (depth = 1.0) receives this much delay.
    /// Closer voices receive proportionally less.
    ///
    /// Default: `15.0` ms.
    pub pre_delay_max_ms: f32,

    /// Early-reflection wet mix for the farthest voice (0.0-1.0).
    ///
    /// Voices at depth 0.0 get no ER contribution; the farthest voice
    /// gets this amount. Intermediate depths interpolate linearly.
    ///
    /// Default: `0.3`.
    pub early_reflection_amount: f32,

    /// Whether to simulate air absorption (progressive HF rolloff with depth).
    ///
    /// When `true`, the LPF cutoff additionally decreases with depth
    /// following an exponential curve. When `false`, only the linear
    /// interpolation between `close_lpf_freq` and `distance_lpf_freq`
    /// is applied.
    ///
    /// Default: `true`.
    pub air_absorption: bool,
}

impl Default for DepthStagingConfig {
    fn default() -> Self {
        Self {
            n_voices: 4,
            depth_spread: 0.6,
            lead_voice_depth: 0.1,
            backing_voice_depth: 0.6,
            distance_attenuation_db: -6.0,
            distance_lpf_freq: 8000.0,
            close_lpf_freq: 20000.0,
            pre_delay_max_ms: 15.0,
            early_reflection_amount: 0.3,
            air_absorption: true,
        }
    }
}

impl DepthStagingConfig {
    /// Create a new depth staging config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of voices.
    #[must_use]
    pub fn with_n_voices(mut self, n: usize) -> Self {
        self.n_voices = n;
        self
    }

    /// Set the depth spread (0.0-1.0).
    #[must_use]
    pub fn with_depth_spread(mut self, spread: f32) -> Self {
        self.depth_spread = spread;
        self
    }

    /// Set the lead voice depth position (0.0-1.0).
    #[must_use]
    pub fn with_lead_voice_depth(mut self, depth: f32) -> Self {
        self.lead_voice_depth = depth;
        self
    }

    /// Set the backing voice depth position (0.0-1.0).
    #[must_use]
    pub fn with_backing_voice_depth(mut self, depth: f32) -> Self {
        self.backing_voice_depth = depth;
        self
    }

    /// Set the maximum distance attenuation in dB (should be negative).
    #[must_use]
    pub fn with_distance_attenuation_db(mut self, db: f32) -> Self {
        self.distance_attenuation_db = db;
        self
    }

    /// Set the LPF cutoff for the farthest voice in Hz.
    #[must_use]
    pub fn with_distance_lpf_freq(mut self, hz: f32) -> Self {
        self.distance_lpf_freq = hz;
        self
    }

    /// Set the LPF cutoff for the closest voice in Hz.
    #[must_use]
    pub fn with_close_lpf_freq(mut self, hz: f32) -> Self {
        self.close_lpf_freq = hz;
        self
    }

    /// Set the maximum pre-delay in milliseconds.
    #[must_use]
    pub fn with_pre_delay_max_ms(mut self, ms: f32) -> Self {
        self.pre_delay_max_ms = ms;
        self
    }

    /// Set the early-reflection wet amount for the farthest voice (0.0-1.0).
    #[must_use]
    pub fn with_early_reflection_amount(mut self, amount: f32) -> Self {
        self.early_reflection_amount = amount;
        self
    }

    /// Enable or disable air absorption simulation.
    #[must_use]
    pub fn with_air_absorption(mut self, enable: bool) -> Self {
        self.air_absorption = enable;
        self
    }

    // -- Presets --------------------------------------------------------------

    /// Intimate preset: all voices close, minimal depth spread.
    ///
    /// Suitable for solo + 1-2 backing voices that should sound like
    /// they are all at the same microphone.
    #[must_use]
    pub fn intimate() -> Self {
        Self {
            n_voices: 4,
            depth_spread: 0.15,
            lead_voice_depth: 0.05,
            backing_voice_depth: 0.2,
            distance_attenuation_db: -2.0,
            distance_lpf_freq: 16000.0,
            close_lpf_freq: 20000.0,
            pre_delay_max_ms: 3.0,
            early_reflection_amount: 0.05,
            air_absorption: false,
        }
    }

    /// Studio preset: moderate depth, balanced and natural.
    ///
    /// A good default for most chorus work. The lead is clearly in
    /// front, backing voices sit comfortably behind.
    #[must_use]
    pub fn studio() -> Self {
        Self {
            n_voices: 4,
            depth_spread: 0.5,
            lead_voice_depth: 0.1,
            backing_voice_depth: 0.5,
            distance_attenuation_db: -4.0,
            distance_lpf_freq: 10000.0,
            close_lpf_freq: 20000.0,
            pre_delay_max_ms: 10.0,
            early_reflection_amount: 0.2,
            air_absorption: true,
        }
    }

    /// Concert hall preset: wide depth spread, lead very close, backing far.
    ///
    /// Strong front-to-back separation. Distant voices have noticeable
    /// HF rolloff and early reflections.
    #[must_use]
    pub fn concert_hall() -> Self {
        Self {
            n_voices: 4,
            depth_spread: 0.8,
            lead_voice_depth: 0.05,
            backing_voice_depth: 0.75,
            distance_attenuation_db: -8.0,
            distance_lpf_freq: 6000.0,
            close_lpf_freq: 20000.0,
            pre_delay_max_ms: 20.0,
            early_reflection_amount: 0.4,
            air_absorption: true,
        }
    }

    /// Cathedral preset: extreme depth spread, heavy ER contribution.
    ///
    /// Simulates a large reverberant space. The farthest voices are
    /// dark and diffuse with significant early reflections.
    #[must_use]
    pub fn cathedral() -> Self {
        Self {
            n_voices: 4,
            depth_spread: 1.0,
            lead_voice_depth: 0.05,
            backing_voice_depth: 0.9,
            distance_attenuation_db: -10.0,
            distance_lpf_freq: 4000.0,
            close_lpf_freq: 20000.0,
            pre_delay_max_ms: 30.0,
            early_reflection_amount: 0.6,
            air_absorption: true,
        }
    }

    /// Validate that all parameters are within meaningful ranges.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range
    /// or non-finite.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.n_voices == 0 {
            return Err(KokoroError::InvalidConfig {
                field: "n_voices",
                reason: "n_voices must be >= 1".to_string(),
            });
        }
        validate_unit_range(self.depth_spread, "depth_spread")?;
        validate_unit_range(self.lead_voice_depth, "lead_voice_depth")?;
        validate_unit_range(self.backing_voice_depth, "backing_voice_depth")?;
        validate_unit_range(self.early_reflection_amount, "early_reflection_amount")?;

        if !self.distance_attenuation_db.is_finite() || self.distance_attenuation_db > 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "distance_attenuation_db",
                reason: format!(
                    "distance_attenuation_db = {}: must be finite and <= 0.0",
                    self.distance_attenuation_db,
                ),
            });
        }
        validate_positive_finite(self.distance_lpf_freq, "distance_lpf_freq")?;
        validate_positive_finite(self.close_lpf_freq, "close_lpf_freq")?;

        if !self.pre_delay_max_ms.is_finite() || self.pre_delay_max_ms < 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "pre_delay_max_ms",
                reason: format!(
                    "pre_delay_max_ms = {}: must be finite and >= 0.0",
                    self.pre_delay_max_ms,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DepthVoicePosition
// ---------------------------------------------------------------------------

/// Computed depth-stage parameters for a single voice.
///
/// Derived from the voice index and [`DepthStagingConfig`]. These values
/// control the per-voice processing chain inside [`DepthStagingProcessor`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthVoicePosition {
    /// Depth position: 0.0 (front of stage) to 1.0 (back of stage).
    pub depth: f32,

    /// Gain attenuation from distance, in dB (always <= 0.0).
    pub gain_db: f32,

    /// Low-pass filter cutoff frequency in Hz, derived from depth.
    pub lpf_cutoff: f32,

    /// Pre-delay in samples, derived from depth and sample rate.
    pub pre_delay_samples: usize,

    /// Early-reflection wet mix amount (0.0-1.0).
    pub er_mix: f32,
}

/// Compute depth positions for all voices given a config and sample rate.
///
/// Voice 0 (lead) is positioned at `lead_voice_depth`. Remaining voices
/// are distributed evenly between `lead_voice_depth` and
/// `backing_voice_depth`, scaled by `depth_spread`.
#[must_use]
pub fn compute_voice_positions(
    config: &DepthStagingConfig,
    sample_rate: f32,
) -> Vec<DepthVoicePosition> {
    let n = config.n_voices;
    if n == 0 {
        return Vec::new();
    }

    let mut positions = Vec::with_capacity(n);

    for i in 0..n {
        // Assign depth: voice 0 = lead, rest distributed toward backing.
        let depth = if n == 1 {
            config.lead_voice_depth
        } else if i == 0 {
            config.lead_voice_depth
        } else {
            let t = i as f32 / (n - 1) as f32;
            let raw = config.lead_voice_depth
                + t * (config.backing_voice_depth - config.lead_voice_depth);
            // Scale by depth_spread: 0.0 collapses all to lead depth.
            config.lead_voice_depth + (raw - config.lead_voice_depth) * config.depth_spread
        };
        let depth = depth.clamp(0.0, 1.0);

        // Gain attenuation: linear interpolation of dB based on depth.
        let gain_db = config.distance_attenuation_db * depth;

        // LPF cutoff: interpolate between close and distance, optionally
        // with additional air-absorption exponential curve.
        let base_cutoff =
            config.close_lpf_freq + depth * (config.distance_lpf_freq - config.close_lpf_freq);
        let lpf_cutoff = if config.air_absorption {
            // Exponential rolloff: at depth 1.0, reduce cutoff by an extra
            // factor. The curve goes from 1.0 at depth=0 to ~0.7 at depth=1.
            let air_factor = (-0.35 * depth).exp();
            (base_cutoff * air_factor).max(100.0)
        } else {
            base_cutoff.max(100.0)
        };

        // Pre-delay in samples.
        let delay_ms = config.pre_delay_max_ms * depth;
        let delay_samples_raw = (delay_ms * sample_rate / 1000.0).round() as usize;
        let pre_delay_samples = delay_samples_raw.min(MAX_PRE_DELAY_SAMPLES);

        // Early-reflection mix proportional to depth.
        let er_mix = config.early_reflection_amount * depth;

        positions.push(DepthVoicePosition {
            depth,
            gain_db,
            lpf_cutoff,
            pre_delay_samples,
            er_mix,
        });
    }

    positions
}

// ---------------------------------------------------------------------------
// DepthStagingProcessor
// ---------------------------------------------------------------------------

/// Per-voice depth staging processor.
///
/// Applies distance attenuation, air-absorption LPF, pre-delay, early
/// reflections, and proximity bass boost to each voice based on its
/// computed [`DepthVoicePosition`].
///
/// Create via [`DepthStagingProcessor::new`], process with
/// [`DepthStagingProcessor::process_voices`], reset between segments
/// with [`DepthStagingProcessor::reset`].
pub struct DepthStagingProcessor {
    /// Per-voice depth positions.
    positions: Vec<DepthVoicePosition>,
    /// Per-voice one-pole LPF state (previous output sample).
    lpf_states: Vec<f32>,
    /// Per-voice pre-delay circular buffers.
    delay_buffers: Vec<Vec<f32>>,
    /// Per-voice delay buffer write positions.
    delay_write_pos: Vec<usize>,
    /// Per-voice ER comb-filter circular buffers.
    er_buffers: Vec<Vec<f32>>,
    /// Per-voice ER buffer write positions.
    er_write_pos: Vec<usize>,
    /// ER comb-filter tap offsets in samples (shared across voices).
    er_tap_offsets: [usize; ER_COMB_TAPS],
    /// Per-voice proximity-effect shelf filter state.
    prox_states: Vec<f32>,
    /// Sample rate in Hz.
    sample_rate: f32,
    /// One-pole LPF coefficients per voice (alpha).
    lpf_alphas: Vec<f32>,
    /// Linear gain per voice (from gain_db).
    linear_gains: Vec<f32>,
    /// Proximity-effect one-pole LPF alpha (200 Hz shelf).
    prox_alpha: f32,
}

impl DepthStagingProcessor {
    /// Create a new depth staging processor.
    ///
    /// Computes voice positions and allocates internal state buffers.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if the config or sample rate
    /// is invalid.
    pub fn new(config: &DepthStagingConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;

        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("sample_rate = {sample_rate}: must be finite and > 0.0"),
            });
        }

        let positions = compute_voice_positions(config, sample_rate);
        let n = positions.len();

        // Compute one-pole LPF alpha for each voice.
        let lpf_alphas: Vec<f32> = positions
            .iter()
            .map(|p| one_pole_alpha(p.lpf_cutoff, sample_rate))
            .collect();

        // Compute linear gains from dB.
        let linear_gains: Vec<f32> = positions.iter().map(|p| db_to_linear(p.gain_db)).collect();

        // Allocate pre-delay buffers.
        let delay_buffers: Vec<Vec<f32>> = positions
            .iter()
            .map(|p| {
                let size = (p.pre_delay_samples + 1).max(1);
                vec![0.0f32; size]
            })
            .collect();
        let delay_write_pos = vec![0usize; n];

        // Compute ER tap offsets in samples.
        let er_tap_offsets: [usize; ER_COMB_TAPS] = {
            let mut offsets = [0usize; ER_COMB_TAPS];
            for (i, &ms) in ER_TAP_MS.iter().enumerate() {
                let samp = (ms * sample_rate / 1000.0).round() as usize;
                offsets[i] = samp.max(1);
            }
            offsets
        };

        // ER buffer size: max tap offset + 1.
        let er_buf_size = er_tap_offsets.iter().copied().max().unwrap_or(1) + 1;
        let er_buffers: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0f32; er_buf_size]).collect();
        let er_write_pos = vec![0usize; n];

        // Proximity shelf: one-pole at 200 Hz.
        let prox_alpha = one_pole_alpha(PROXIMITY_FREQ_HZ, sample_rate);

        Ok(Self {
            positions,
            lpf_states: vec![0.0f32; n],
            delay_buffers,
            delay_write_pos,
            er_buffers,
            er_write_pos,
            er_tap_offsets,
            prox_states: vec![0.0f32; n],
            sample_rate,
            lpf_alphas,
            linear_gains,
            prox_alpha,
        })
    }

    /// Process all voices in-place with depth staging effects.
    ///
    /// Each voice buffer is modified to reflect its depth position:
    /// distance attenuation, air-absorption LPF, pre-delay, early
    /// reflections, and proximity bass boost.
    ///
    /// The number of voice buffers must match `config.n_voices`.
    /// Extra or missing voices are silently handled: extra voices are
    /// left unmodified, missing voices are skipped.
    pub fn process_voices(&mut self, voices: &mut [Vec<f32>]) {
        let n = self.positions.len().min(voices.len());

        for vi in 0..n {
            let pos = self.positions[vi];
            let gain = self.linear_gains[vi];
            let alpha = self.lpf_alphas[vi];
            let voice = &mut voices[vi];

            // -- 1. Pre-delay --
            if pos.pre_delay_samples > 0 {
                self.apply_pre_delay(vi, voice);
            }

            // -- Per-sample processing --
            for sample in voice.iter_mut() {
                let s = if sample.is_finite() { *sample } else { 0.0 };

                // -- 2. Air-absorption LPF (one-pole) --
                let filtered = self.lpf_states[vi] + alpha * (s - self.lpf_states[vi]);
                let filtered = if filtered.is_finite() { filtered } else { 0.0 };
                self.lpf_states[vi] = filtered;

                // -- 3. Distance attenuation --
                let attenuated = filtered * gain;

                // -- 4. Proximity bass boost --
                // Close voices (depth near 0) get a bass shelf boost.
                let with_prox = if pos.depth < 0.5 {
                    let prox_strength = 1.0 - pos.depth * 2.0; // 1.0 at depth=0, 0.0 at depth=0.5
                    let prox_gain = db_to_linear(PROXIMITY_BOOST_DB * prox_strength);
                    // Extract bass via one-pole LPF, boost it.
                    let bass = self.prox_states[vi]
                        + self.prox_alpha * (attenuated - self.prox_states[vi]);
                    let bass = if bass.is_finite() { bass } else { 0.0 };
                    self.prox_states[vi] = bass;
                    let treble = attenuated - bass;
                    bass * prox_gain + treble
                } else {
                    // Update state even when not boosting to keep filter warm.
                    let bass = self.prox_states[vi]
                        + self.prox_alpha * (attenuated - self.prox_states[vi]);
                    self.prox_states[vi] = if bass.is_finite() { bass } else { 0.0 };
                    attenuated
                };

                *sample = if with_prox.is_finite() {
                    with_prox
                } else {
                    0.0
                };
            }

            // -- 5. Early reflections (comb filter, wet mix) --
            if pos.er_mix > 1e-6 {
                self.apply_early_reflections(vi, voice, pos.er_mix);
            }
        }
    }

    /// Apply pre-delay to a single voice using its circular buffer.
    fn apply_pre_delay(&mut self, vi: usize, voice: &mut [f32]) {
        let buf = &mut self.delay_buffers[vi];
        let buf_len = buf.len();
        let delay = self.positions[vi].pre_delay_samples;
        let wp = &mut self.delay_write_pos[vi];

        // Process in-place: read delayed, write current.
        for sample in voice.iter_mut() {
            let s = if sample.is_finite() { *sample } else { 0.0 };
            buf[*wp] = s;
            let read_pos = (*wp + buf_len - delay) % buf_len;
            *sample = buf[read_pos];
            *wp = (*wp + 1) % buf_len;
        }
    }

    /// Apply simple comb-filter early reflections to a voice.
    fn apply_early_reflections(&mut self, vi: usize, voice: &mut [f32], er_mix: f32) {
        let buf = &mut self.er_buffers[vi];
        let buf_len = buf.len();
        let wp = &mut self.er_write_pos[vi];

        let per_tap_gain = er_mix / ER_COMB_TAPS as f32;

        for sample in voice.iter_mut() {
            let dry = *sample;
            let s = if dry.is_finite() { dry } else { 0.0 };

            buf[*wp] = s;

            // Sum delayed taps.
            let mut wet = 0.0f32;
            for &offset in &self.er_tap_offsets {
                let read_pos = (*wp + buf_len - offset) % buf_len;
                wet += buf[read_pos];
            }
            wet *= per_tap_gain;

            let mixed = dry + wet;
            *sample = if mixed.is_finite() { mixed } else { 0.0 };

            *wp = (*wp + 1) % buf_len;
        }
    }

    /// Clear all internal filter states and delay buffers.
    ///
    /// Call between audio segments to prevent artifacts from stale data.
    pub fn reset(&mut self) {
        self.lpf_states.fill(0.0);
        self.prox_states.fill(0.0);
        for buf in &mut self.delay_buffers {
            buf.fill(0.0);
        }
        self.delay_write_pos.fill(0);
        for buf in &mut self.er_buffers {
            buf.fill(0.0);
        }
        self.er_write_pos.fill(0);
    }

    /// Return the computed voice positions (for diagnostics/testing).
    #[must_use]
    pub fn voice_positions(&self) -> &[DepthVoicePosition] {
        &self.positions
    }

    /// Return the number of voices configured.
    #[must_use]
    pub fn n_voices(&self) -> usize {
        self.positions.len()
    }

    /// Return the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Compute one-pole low-pass filter coefficient (alpha).
///
/// `alpha = 1 - exp(-2 * pi * cutoff / sample_rate)`
///
/// Higher alpha = higher cutoff (more HF passes through).
#[inline]
fn one_pole_alpha(cutoff_hz: f32, sample_rate: f32) -> f32 {
    let omega = 2.0 * std::f32::consts::PI * cutoff_hz / sample_rate;
    let alpha = 1.0 - (-omega).exp();
    // Clamp to valid range: 0.0 = no filtering, 1.0 = no filter (passthrough).
    if alpha.is_finite() {
        alpha.clamp(0.0, 1.0)
    } else {
        1.0 // passthrough on invalid input
    }
}

/// Convert decibels to linear amplitude.
#[inline]
fn db_to_linear(db: f32) -> f32 {
    let lin = 10.0f32.powf(db / 20.0);
    if lin.is_finite() {
        lin
    } else {
        0.0
    }
}

/// Validate that a value is in [0.0, 1.0] and finite.
fn validate_unit_range(val: f32, field: &'static str) -> Result<(), KokoroError> {
    if !val.is_finite() || !(0.0..=1.0).contains(&val) {
        return Err(KokoroError::InvalidConfig {
            field,
            reason: format!("{field} = {val}: must be finite and in [0.0, 1.0]"),
        });
    }
    Ok(())
}

/// Validate that a value is positive and finite.
fn validate_positive_finite(val: f32, field: &'static str) -> Result<(), KokoroError> {
    if !val.is_finite() || val <= 0.0 {
        return Err(KokoroError::InvalidConfig {
            field,
            reason: format!("{field} = {val}: must be finite and > 0.0"),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config validation ----------------------------------------------------

    #[test]
    fn test_default_config_valid() {
        let config = DepthStagingConfig::new();
        config.validate().expect("default config should be valid");
    }

    #[test]
    fn test_all_presets_valid() {
        for config in [
            DepthStagingConfig::intimate(),
            DepthStagingConfig::studio(),
            DepthStagingConfig::concert_hall(),
            DepthStagingConfig::cathedral(),
        ] {
            config
                .validate()
                .unwrap_or_else(|e| panic!("preset invalid: {e}"));
        }
    }

    #[test]
    fn test_invalid_zero_voices() {
        let config = DepthStagingConfig::new().with_n_voices(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_depth_spread_out_of_range() {
        let config = DepthStagingConfig::new().with_depth_spread(1.5);
        assert!(config.validate().is_err());
        let config = DepthStagingConfig::new().with_depth_spread(-0.1);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_nan_parameter() {
        let config = DepthStagingConfig::new().with_lead_voice_depth(f32::NAN);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_positive_attenuation() {
        let config = DepthStagingConfig::new().with_distance_attenuation_db(3.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_zero_lpf_freq() {
        let config = DepthStagingConfig::new().with_distance_lpf_freq(0.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_negative_pre_delay() {
        let config = DepthStagingConfig::new().with_pre_delay_max_ms(-1.0);
        assert!(config.validate().is_err());
    }

    // -- Voice position computation -------------------------------------------

    #[test]
    fn test_voice_positions_count() {
        let config = DepthStagingConfig::new().with_n_voices(6);
        let positions = compute_voice_positions(&config, DEFAULT_SAMPLE_RATE);
        assert_eq!(positions.len(), 6);
    }

    #[test]
    fn test_lead_voice_at_lead_depth() {
        let config = DepthStagingConfig::new()
            .with_n_voices(4)
            .with_lead_voice_depth(0.1);
        let positions = compute_voice_positions(&config, DEFAULT_SAMPLE_RATE);
        assert!(
            (positions[0].depth - 0.1).abs() < 1e-6,
            "lead voice depth = {}, expected 0.1",
            positions[0].depth,
        );
    }

    #[test]
    fn test_backing_voices_deeper_than_lead() {
        let config = DepthStagingConfig::new()
            .with_n_voices(4)
            .with_lead_voice_depth(0.1)
            .with_backing_voice_depth(0.7)
            .with_depth_spread(1.0);
        let positions = compute_voice_positions(&config, DEFAULT_SAMPLE_RATE);
        for (i, pos) in positions.iter().enumerate().skip(1) {
            assert!(
                pos.depth >= positions[0].depth,
                "voice {} depth {} should be >= lead depth {}",
                i,
                pos.depth,
                positions[0].depth,
            );
        }
    }

    #[test]
    fn test_depth_spread_zero_collapses() {
        let config = DepthStagingConfig::new()
            .with_n_voices(4)
            .with_depth_spread(0.0)
            .with_lead_voice_depth(0.2);
        let positions = compute_voice_positions(&config, DEFAULT_SAMPLE_RATE);
        for pos in &positions {
            assert!(
                (pos.depth - 0.2).abs() < 1e-6,
                "with spread=0, all voices should be at lead depth, got {}",
                pos.depth,
            );
        }
    }

    #[test]
    fn test_gain_db_nonpositive() {
        let config = DepthStagingConfig::new().with_n_voices(6);
        let positions = compute_voice_positions(&config, DEFAULT_SAMPLE_RATE);
        for pos in &positions {
            assert!(
                pos.gain_db <= 0.0,
                "gain_db should be <= 0.0, got {}",
                pos.gain_db,
            );
        }
    }

    #[test]
    fn test_deeper_voices_more_attenuation() {
        let config = DepthStagingConfig::new()
            .with_n_voices(4)
            .with_depth_spread(1.0)
            .with_lead_voice_depth(0.0)
            .with_backing_voice_depth(1.0);
        let positions = compute_voice_positions(&config, DEFAULT_SAMPLE_RATE);
        // Each successive voice should have equal or more attenuation.
        for i in 1..positions.len() {
            assert!(
                positions[i].gain_db <= positions[i - 1].gain_db,
                "voice {} gain {} should be <= voice {} gain {}",
                i,
                positions[i].gain_db,
                i - 1,
                positions[i - 1].gain_db,
            );
        }
    }

    #[test]
    fn test_pre_delay_increases_with_depth() {
        let config = DepthStagingConfig::new()
            .with_n_voices(4)
            .with_depth_spread(1.0)
            .with_lead_voice_depth(0.0)
            .with_backing_voice_depth(1.0)
            .with_pre_delay_max_ms(15.0);
        let positions = compute_voice_positions(&config, DEFAULT_SAMPLE_RATE);
        for i in 1..positions.len() {
            assert!(
                positions[i].pre_delay_samples >= positions[i - 1].pre_delay_samples,
                "voice {} pre_delay {} should be >= voice {} pre_delay {}",
                i,
                positions[i].pre_delay_samples,
                i - 1,
                positions[i - 1].pre_delay_samples,
            );
        }
    }

    #[test]
    fn test_er_mix_increases_with_depth() {
        let config = DepthStagingConfig::new()
            .with_n_voices(4)
            .with_depth_spread(1.0)
            .with_lead_voice_depth(0.0)
            .with_backing_voice_depth(1.0);
        let positions = compute_voice_positions(&config, DEFAULT_SAMPLE_RATE);
        for i in 1..positions.len() {
            assert!(
                positions[i].er_mix >= positions[i - 1].er_mix - 1e-6,
                "voice {} er_mix {} should be >= voice {} er_mix {}",
                i,
                positions[i].er_mix,
                i - 1,
                positions[i - 1].er_mix,
            );
        }
    }

    // -- Processor construction -----------------------------------------------

    #[test]
    fn test_processor_construction() {
        let config = DepthStagingConfig::new();
        let proc = DepthStagingProcessor::new(&config, DEFAULT_SAMPLE_RATE)
            .expect("construction should succeed");
        assert_eq!(proc.n_voices(), 4);
        assert!((proc.sample_rate() - DEFAULT_SAMPLE_RATE).abs() < 1e-6);
    }

    #[test]
    fn test_processor_invalid_sample_rate() {
        let config = DepthStagingConfig::new();
        assert!(DepthStagingProcessor::new(&config, 0.0).is_err());
        assert!(DepthStagingProcessor::new(&config, -1.0).is_err());
        assert!(DepthStagingProcessor::new(&config, f32::NAN).is_err());
    }

    #[test]
    fn test_processor_invalid_config() {
        let config = DepthStagingConfig::new().with_n_voices(0);
        assert!(DepthStagingProcessor::new(&config, DEFAULT_SAMPLE_RATE).is_err());
    }

    // -- Processing behavior --------------------------------------------------

    #[test]
    fn test_silence_produces_silence() {
        let config = DepthStagingConfig::new().with_n_voices(3);
        let mut proc = DepthStagingProcessor::new(&config, DEFAULT_SAMPLE_RATE).unwrap();
        let mut voices: Vec<Vec<f32>> = (0..3).map(|_| vec![0.0; 1000]).collect();
        proc.process_voices(&mut voices);
        for (vi, voice) in voices.iter().enumerate() {
            for &s in voice {
                assert!(s.abs() < 1e-10, "voice {vi} should be silent, got {s}");
            }
        }
    }

    #[test]
    fn test_closer_voice_louder_than_distant() {
        let config = DepthStagingConfig::new()
            .with_n_voices(2)
            .with_lead_voice_depth(0.0)
            .with_backing_voice_depth(1.0)
            .with_depth_spread(1.0)
            .with_distance_attenuation_db(-6.0)
            .with_early_reflection_amount(0.0)
            .with_pre_delay_max_ms(0.0);
        let mut proc = DepthStagingProcessor::new(&config, DEFAULT_SAMPLE_RATE).unwrap();

        // Both voices start with identical content.
        let signal: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut voices = vec![signal.clone(), signal];

        proc.process_voices(&mut voices);

        let energy_close: f32 = voices[0].iter().map(|s| s * s).sum();
        let energy_far: f32 = voices[1].iter().map(|s| s * s).sum();

        assert!(
            energy_close > energy_far,
            "close voice energy {energy_close} should exceed far voice energy {energy_far}",
        );
    }

    #[test]
    fn test_distant_voice_has_less_hf() {
        // Use a high-frequency signal to test LPF effect.
        let config = DepthStagingConfig::new()
            .with_n_voices(2)
            .with_lead_voice_depth(0.0)
            .with_backing_voice_depth(1.0)
            .with_depth_spread(1.0)
            .with_distance_lpf_freq(2000.0) // aggressive LPF for testing
            .with_close_lpf_freq(20000.0)
            .with_distance_attenuation_db(0.0) // no gain change
            .with_pre_delay_max_ms(0.0)
            .with_early_reflection_amount(0.0);
        let mut proc = DepthStagingProcessor::new(&config, DEFAULT_SAMPLE_RATE).unwrap();

        // 10 kHz tone: should be attenuated by the distant voice's LPF.
        let freq = 10000.0;
        let signal: Vec<f32> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / DEFAULT_SAMPLE_RATE).sin())
            .collect();
        let mut voices = vec![signal.clone(), signal];

        proc.process_voices(&mut voices);

        // Measure energy in the second half (after filter has settled).
        let half = voices[0].len() / 2;
        let close_energy: f32 = voices[0][half..].iter().map(|s| s * s).sum();
        let far_energy: f32 = voices[1][half..].iter().map(|s| s * s).sum();

        assert!(
            close_energy > far_energy * 1.5,
            "close voice HF energy {close_energy} should significantly exceed far voice HF energy {far_energy}",
        );
    }

    #[test]
    fn test_pre_delay_shifts_signal() {
        let config = DepthStagingConfig::new()
            .with_n_voices(1)
            .with_lead_voice_depth(0.5) // mid-depth so we get some delay
            .with_pre_delay_max_ms(10.0)
            .with_distance_attenuation_db(0.0)
            .with_close_lpf_freq(20000.0)
            .with_distance_lpf_freq(20000.0)
            .with_early_reflection_amount(0.0);
        let mut proc = DepthStagingProcessor::new(&config, DEFAULT_SAMPLE_RATE).unwrap();

        let delay_samples = proc.voice_positions()[0].pre_delay_samples;
        assert!(delay_samples > 0, "should have some pre-delay");

        // Impulse at sample 0.
        let mut signal = vec![0.0f32; 480];
        signal[0] = 1.0;
        let mut voices = vec![signal];

        proc.process_voices(&mut voices);

        // The impulse should appear at the delay offset, not at sample 0.
        // Sample 0 should be zero (delayed).
        assert!(
            voices[0][0].abs() < 0.01,
            "sample 0 should be near-zero after delay, got {}",
            voices[0][0],
        );
        // The impulse should appear around the delay position.
        // (Exact position depends on LPF smoothing, but peak should be near delay_samples.)
        let peak_pos = voices[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            (peak_pos as i64 - delay_samples as i64).unsigned_abs() <= 2,
            "peak at {peak_pos} should be near delay {delay_samples} samples",
        );
    }

    #[test]
    fn test_early_reflections_add_energy() {
        let config_dry = DepthStagingConfig::new()
            .with_n_voices(1)
            .with_lead_voice_depth(0.8) // deep enough for ER
            .with_early_reflection_amount(0.0)
            .with_distance_attenuation_db(0.0)
            .with_pre_delay_max_ms(0.0);
        let config_wet = DepthStagingConfig::new()
            .with_n_voices(1)
            .with_lead_voice_depth(0.8)
            .with_early_reflection_amount(0.5)
            .with_distance_attenuation_db(0.0)
            .with_pre_delay_max_ms(0.0);

        let mut proc_dry = DepthStagingProcessor::new(&config_dry, DEFAULT_SAMPLE_RATE).unwrap();
        let mut proc_wet = DepthStagingProcessor::new(&config_wet, DEFAULT_SAMPLE_RATE).unwrap();

        let signal: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.05).sin()).collect();

        let mut voices_dry = vec![signal.clone()];
        let mut voices_wet = vec![signal];

        proc_dry.process_voices(&mut voices_dry);
        proc_wet.process_voices(&mut voices_wet);

        // Wet version should have more energy (ER adds energy).
        let dry_energy: f32 = voices_dry[0].iter().map(|s| s * s).sum();
        let wet_energy: f32 = voices_wet[0].iter().map(|s| s * s).sum();

        assert!(
            wet_energy > dry_energy,
            "wet energy {wet_energy} should exceed dry energy {dry_energy}",
        );
    }

    #[test]
    fn test_nan_defense_in_depth() {
        let config = DepthStagingConfig::new().with_n_voices(2);
        let mut proc = DepthStagingProcessor::new(&config, DEFAULT_SAMPLE_RATE).unwrap();

        let mut voices = vec![
            vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0, 0.0],
            vec![0.5, f32::NAN, -0.5, f32::INFINITY, 0.0],
        ];

        proc.process_voices(&mut voices);

        for (vi, voice) in voices.iter().enumerate() {
            for (si, &s) in voice.iter().enumerate() {
                assert!(
                    s.is_finite(),
                    "voice {vi} sample {si} must be finite, got {s}",
                );
            }
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let config = DepthStagingConfig::new().with_n_voices(2);
        let mut proc = DepthStagingProcessor::new(&config, DEFAULT_SAMPLE_RATE).unwrap();

        // Process some audio.
        let signal: Vec<f32> = (0..480).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut voices = vec![signal.clone(), signal];
        proc.process_voices(&mut voices);

        // Reset.
        proc.reset();

        // Process silence: should produce silence.
        let mut silence = vec![vec![0.0f32; 480], vec![0.0f32; 480]];
        proc.process_voices(&mut silence);

        for (vi, voice) in silence.iter().enumerate() {
            for &s in voice {
                assert!(
                    s.abs() < 1e-10,
                    "after reset, silence should produce silence (voice {vi}, got {s})",
                );
            }
        }
    }

    #[test]
    fn test_empty_voices_no_panic() {
        let config = DepthStagingConfig::new().with_n_voices(2);
        let mut proc = DepthStagingProcessor::new(&config, DEFAULT_SAMPLE_RATE).unwrap();
        let mut voices: Vec<Vec<f32>> = vec![vec![], vec![]];
        proc.process_voices(&mut voices);
        // Should not panic.
    }

    #[test]
    fn test_fewer_voices_than_config() {
        let config = DepthStagingConfig::new().with_n_voices(4);
        let mut proc = DepthStagingProcessor::new(&config, DEFAULT_SAMPLE_RATE).unwrap();
        // Only 2 voices provided.
        let mut voices = vec![vec![0.5f32; 100], vec![0.5f32; 100]];
        proc.process_voices(&mut voices);
        // Should not panic; processes what it can.
    }

    // -- Preset property checks -----------------------------------------------

    #[test]
    fn test_cathedral_more_er_than_intimate() {
        let int_config = DepthStagingConfig::intimate();
        let cat_config = DepthStagingConfig::cathedral();
        assert!(
            cat_config.early_reflection_amount > int_config.early_reflection_amount,
            "cathedral ER {} should exceed intimate ER {}",
            cat_config.early_reflection_amount,
            int_config.early_reflection_amount,
        );
    }

    #[test]
    fn test_concert_hall_more_spread_than_studio() {
        let studio = DepthStagingConfig::studio();
        let hall = DepthStagingConfig::concert_hall();
        assert!(
            hall.depth_spread > studio.depth_spread,
            "concert hall spread {} should exceed studio spread {}",
            hall.depth_spread,
            studio.depth_spread,
        );
    }

    #[test]
    fn test_intimate_has_air_absorption_off() {
        let config = DepthStagingConfig::intimate();
        assert!(
            !config.air_absorption,
            "intimate preset should have air_absorption off",
        );
    }

    // -- Helper function tests ------------------------------------------------

    #[test]
    fn test_db_to_linear_zero_db() {
        let lin = db_to_linear(0.0);
        assert!((lin - 1.0).abs() < 1e-6, "0 dB should be 1.0, got {lin}");
    }

    #[test]
    fn test_db_to_linear_minus_6db() {
        let lin = db_to_linear(-6.0);
        assert!(
            (lin - 0.5012).abs() < 0.01,
            "-6 dB should be ~0.5012, got {lin}",
        );
    }

    #[test]
    fn test_one_pole_alpha_high_cutoff_near_one() {
        let alpha = one_pole_alpha(20000.0, 24000.0);
        assert!(
            alpha > 0.9,
            "20kHz cutoff at 24kHz SR should have alpha near 1.0, got {alpha}",
        );
    }

    #[test]
    fn test_one_pole_alpha_low_cutoff_near_zero() {
        let alpha = one_pole_alpha(10.0, 24000.0);
        assert!(
            alpha < 0.01,
            "10Hz cutoff at 24kHz SR should have alpha near 0.0, got {alpha}",
        );
    }

    #[test]
    fn test_single_voice_no_crash() {
        let config = DepthStagingConfig::new().with_n_voices(1);
        let mut proc = DepthStagingProcessor::new(&config, DEFAULT_SAMPLE_RATE).unwrap();
        let mut voices = vec![vec![0.5f32; 480]];
        proc.process_voices(&mut voices);
        assert_eq!(voices[0].len(), 480);
    }
}
