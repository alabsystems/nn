// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Spatial depth modeling for per-voice distance simulation in Kokoro chorus.
//!
//! In a real choir, front-row singers are louder and drier while back-row
//! singers are quieter and more reverberant. This module models:
//!
//! - **Distance-based attenuation** — inverse distance law (1/r), clamped.
//! - **Air absorption** — one-pole lowpass filter; more distant = more HF loss.
//! - **Propagation delay** — small delay proportional to distance at 343 m/s.
//! - **Interaural level difference (ILD)** — stereo positioning from azimuth.
//!
//! # Usage
//!
//! ```text
//! let config = SpatialConfig::new();
//! let positions = auto_layout_spatial(4, &config);
//! for (voice_pcm, pos) in voices.iter().zip(&positions) {
//!     let (left, right) = process_voice_spatial(voice_pcm, &config, pos)?;
//!     // mix into output stereo bus
//! }
//! ```
//!
//! Part of #4264, #3351.

use crate::kokoro_error::KokoroError;

/// Speed of sound in air at ~20C in meters per second.
const SPEED_OF_SOUND: f32 = 343.0;

/// Minimum clamp distance to avoid division-by-zero (meters).
const MIN_DISTANCE: f32 = 0.1;

/// Reference distance for attenuation normalization (meters).
/// At this distance, gain = 1.0.
const REF_DISTANCE: f32 = 1.0;

/// Maximum propagation delay in samples (caps memory usage).
/// At 24 kHz, 8192 samples ~ 0.34s ~ 117 meters. Far beyond any room.
const MAX_DELAY_SAMPLES: usize = 8192;

// ---------------------------------------------------------------------------
// SpatialConfig
// ---------------------------------------------------------------------------

/// Configuration for the spatial depth processor.
///
/// Controls room dimensions, listener position, air absorption frequency,
/// and sample rate. Built via method chaining on `SpatialConfig::new()`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpatialConfig {
    /// Virtual room dimension in meters. Clamps max voice distance.
    ///
    /// Default: `8.0`.
    pub room_size: f32,

    /// Listener position as distance from the stage front in meters.
    ///
    /// Default: `2.0`. Closer values make distance effects more dramatic.
    pub listener_distance: f32,

    /// Frequency (Hz) above which air absorption starts attenuating.
    ///
    /// Default: `4000.0`. Lower values simulate more absorptive environments.
    pub air_absorption_hz: f32,

    /// Audio sample rate in Hz.
    ///
    /// Default: `24000.0` (Kokoro native rate).
    pub sample_rate: f32,
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            room_size: 8.0,
            listener_distance: 2.0,
            air_absorption_hz: 4000.0,
            sample_rate: 24000.0,
        }
    }
}

impl SpatialConfig {
    /// Create a new spatial config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the virtual room size in meters.
    #[must_use]
    pub fn with_room_size(mut self, meters: f32) -> Self {
        self.room_size = meters;
        self
    }

    /// Set the listener distance from the stage front in meters.
    #[must_use]
    pub fn with_listener_distance(mut self, meters: f32) -> Self {
        self.listener_distance = meters;
        self
    }

    /// Set the air absorption cutoff frequency in Hz.
    #[must_use]
    pub fn with_air_absorption_hz(mut self, hz: f32) -> Self {
        self.air_absorption_hz = hz;
        self
    }

    /// Set the audio sample rate in Hz.
    #[must_use]
    pub fn with_sample_rate(mut self, hz: f32) -> Self {
        self.sample_rate = hz;
        self
    }

    /// Validate that all parameters are within physically meaningful ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.room_size.is_finite() || self.room_size <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "room_size",
                reason: format!("room_size = {}: must be finite and > 0.0", self.room_size),
            });
        }
        if !self.listener_distance.is_finite() || self.listener_distance < 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "listener_distance",
                reason: format!(
                    "listener_distance = {}: must be finite and >= 0.0",
                    self.listener_distance,
                ),
            });
        }
        if !self.air_absorption_hz.is_finite() || self.air_absorption_hz <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "air_absorption_hz",
                reason: format!(
                    "air_absorption_hz = {}: must be finite and > 0.0",
                    self.air_absorption_hz,
                ),
            });
        }
        if !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!(
                    "sample_rate = {}: must be finite and > 0.0",
                    self.sample_rate,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VoiceSpatialPosition
// ---------------------------------------------------------------------------

/// Spatial position of a single voice relative to the listener.
///
/// Uses a spherical coordinate system where:
/// - `distance` is the radial distance from the listener in meters.
/// - `angle` is the azimuth in radians: 0 = center, negative = left, positive = right.
/// - `elevation` is the vertical angle in radians: 0 = ear level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoiceSpatialPosition {
    /// Distance from the listener in meters.
    ///
    /// Clamped to `[MIN_DISTANCE, room_size]` during processing.
    /// Closer voices are louder and brighter.
    pub distance: f32,

    /// Azimuth angle in radians, range `[-PI, PI]`.
    ///
    /// - `0.0` = dead center
    /// - `-PI/2` = hard left
    /// - `PI/2` = hard right
    pub angle: f32,

    /// Elevation angle in radians, range `[-PI/4, PI/4]`.
    ///
    /// - `0.0` = ear level
    /// - `PI/4` = above (upper row in choir risers)
    /// - `-PI/4` = below
    pub elevation: f32,
}

impl VoiceSpatialPosition {
    /// Create a new spatial position.
    #[must_use]
    pub fn new(distance: f32, angle: f32, elevation: f32) -> Self {
        Self {
            distance,
            angle,
            elevation,
        }
    }

    /// Validate that position fields are finite and within expected ranges.
    pub fn validate(&self, room_size: f32) -> Result<(), KokoroError> {
        if !self.distance.is_finite() || self.distance < MIN_DISTANCE {
            return Err(KokoroError::InvalidConfig {
                field: "distance",
                reason: format!(
                    "distance = {}: must be finite and >= {}",
                    self.distance, MIN_DISTANCE,
                ),
            });
        }
        if !room_size.is_finite() || self.distance > room_size {
            return Err(KokoroError::InvalidConfig {
                field: "distance",
                reason: format!(
                    "distance = {} exceeds room_size = {}",
                    self.distance, room_size,
                ),
            });
        }
        if !self.angle.is_finite() {
            return Err(KokoroError::InvalidConfig {
                field: "angle",
                reason: format!("angle = {}: must be finite", self.angle),
            });
        }
        if !self.elevation.is_finite() {
            return Err(KokoroError::InvalidConfig {
                field: "elevation",
                reason: format!("elevation = {}: must be finite", self.elevation),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SpatialProcessor
// ---------------------------------------------------------------------------

/// Per-voice spatial processor that applies distance-based effects.
///
/// Holds internal state for the one-pole lowpass filter and delay buffer.
/// Create one per voice; call [`SpatialProcessor::process`] per audio chunk.
pub struct SpatialProcessor {
    /// One-pole lowpass filter state for air absorption.
    lpf_state: f32,
    /// One-pole lowpass coefficient (0..1). Higher = more filtering.
    lpf_coeff: f32,
    /// Distance-based gain (inverse distance law).
    gain: f32,
    /// Left-ear gain from ILD.
    left_gain: f32,
    /// Right-ear gain from ILD.
    right_gain: f32,
    /// Delay buffer for propagation delay.
    delay_buf: Vec<f32>,
    /// Write position in the circular delay buffer.
    delay_write: usize,
    /// Number of samples to delay.
    delay_samples: usize,
}

impl SpatialProcessor {
    /// Create a new spatial processor for a voice at the given position.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if config or position is invalid.
    pub fn new(
        config: &SpatialConfig,
        position: &VoiceSpatialPosition,
    ) -> Result<Self, KokoroError> {
        config.validate()?;
        position.validate(config.room_size)?;

        let dist = position.distance.max(MIN_DISTANCE);

        // --- Distance attenuation (inverse distance law) ---
        // gain = ref_distance / distance, clamped to [0, 1]
        let gain = (REF_DISTANCE / dist).min(1.0);
        if !gain.is_finite() {
            return Err(KokoroError::InvalidConfig {
                field: "gain",
                reason: format!("computed gain is non-finite for distance = {dist}"),
            });
        }

        // --- Air absorption lowpass coefficient ---
        // One-pole lowpass: y[n] = (1 - a) * x[n] + a * y[n-1]
        // The coefficient `a` increases with distance, meaning more HF loss.
        // At reference distance, a = 0 (no filtering). At room_size, a approaches
        // a maximum determined by air_absorption_hz.
        let normalized_dist =
            ((dist - REF_DISTANCE) / (config.room_size - REF_DISTANCE + 1e-6)).clamp(0.0, 1.0);
        // Map to cutoff: at max distance, cutoff drops to air_absorption_hz.
        // One-pole coefficient from cutoff: a = exp(-2 * pi * fc / fs).
        let cutoff_hz = config.air_absorption_hz
            + (config.sample_rate * 0.5 - config.air_absorption_hz) * (1.0 - normalized_dist);
        let cutoff_clamped = cutoff_hz.clamp(100.0, config.sample_rate * 0.5);
        let lpf_coeff = (-2.0 * std::f32::consts::PI * cutoff_clamped / config.sample_rate).exp();
        let lpf_coeff = if lpf_coeff.is_finite() {
            lpf_coeff.clamp(0.0, 0.999)
        } else {
            0.0
        };

        // --- Propagation delay ---
        // delay_seconds = distance / speed_of_sound
        let delay_seconds = dist / SPEED_OF_SOUND;
        let delay_samples_raw = (delay_seconds * config.sample_rate) as usize;
        let delay_samples = delay_samples_raw.min(MAX_DELAY_SAMPLES);
        // Buffer size: delay + 1 for circular indexing.
        let buf_size = delay_samples + 1;

        // --- ILD from azimuth angle ---
        // Simple model: the ear facing the source gets more level.
        // ILD ~ sin(angle) * max_ild_db. Typical max ILD ~6-10 dB for HF.
        // We use a simplified constant-power panning from the angle.
        let angle_clamped = position
            .angle
            .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        // Map angle to pan: -PI/2 -> pan=-1 (left), PI/2 -> pan=1 (right)
        let pan = (angle_clamped / std::f32::consts::FRAC_PI_2).clamp(-1.0, 1.0);
        // Constant-power pan law
        let pan_angle = (pan + 1.0) * 0.5 * std::f32::consts::FRAC_PI_2;
        let left_gain = pan_angle.cos();
        let right_gain = pan_angle.sin();

        // Finite checks on computed gains
        let left_gain = if left_gain.is_finite() {
            left_gain
        } else {
            0.5
        };
        let right_gain = if right_gain.is_finite() {
            right_gain
        } else {
            0.5
        };

        Ok(Self {
            lpf_state: 0.0,
            lpf_coeff,
            gain,
            left_gain,
            right_gain,
            delay_buf: vec![0.0; buf_size.max(1)],
            delay_write: 0,
            delay_samples,
        })
    }

    /// Process a mono voice buffer, returning a stereo pair `(left, right)`.
    ///
    /// Applies distance attenuation, air absorption lowpass, propagation delay,
    /// and ILD-based stereo panning. Operates sample-by-sample for correctness;
    /// the delay buffer is a simple circular buffer.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if input contains only non-finite
    /// values (empty input is fine, returns empty output).
    #[must_use]
    pub fn process(&mut self, mono: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let len = mono.len();
        let mut left = Vec::with_capacity(len);
        let mut right = Vec::with_capacity(len);

        for &sample in mono {
            // Clamp non-finite inputs to zero (defense-in-depth).
            let s = if sample.is_finite() { sample } else { 0.0 };

            // Write into delay buffer
            let buf_len = self.delay_buf.len();
            self.delay_buf[self.delay_write % buf_len] = s;

            // Read from delay buffer (propagation delay)
            let read_pos = (self.delay_write + buf_len - self.delay_samples) % buf_len;
            let delayed = self.delay_buf[read_pos];

            self.delay_write = (self.delay_write + 1) % buf_len;

            // Apply distance attenuation
            let attenuated = delayed * self.gain;

            // Apply air absorption lowpass (one-pole IIR)
            // y[n] = (1 - a) * x[n] + a * y[n-1]
            let filtered = (1.0 - self.lpf_coeff) * attenuated + self.lpf_coeff * self.lpf_state;
            self.lpf_state = if filtered.is_finite() { filtered } else { 0.0 };

            // Apply ILD stereo panning
            let l = self.lpf_state * self.left_gain;
            let r = self.lpf_state * self.right_gain;

            left.push(if l.is_finite() { l } else { 0.0 });
            right.push(if r.is_finite() { r } else { 0.0 });
        }

        (left, right)
    }

    /// Reset internal filter and delay state (e.g., between segments).
    pub fn reset(&mut self) {
        self.lpf_state = 0.0;
        self.delay_buf.fill(0.0);
        self.delay_write = 0;
    }

    /// Current distance-based gain (for diagnostics).
    #[must_use]
    pub fn gain(&self) -> f32 {
        self.gain
    }

    /// Current left/right gains from ILD (for diagnostics).
    #[must_use]
    pub fn stereo_gains(&self) -> (f32, f32) {
        (self.left_gain, self.right_gain)
    }

    /// Current delay in samples (for diagnostics).
    #[must_use]
    pub fn delay_samples(&self) -> usize {
        self.delay_samples
    }
}

// ---------------------------------------------------------------------------
// auto_layout_spatial
// ---------------------------------------------------------------------------

/// Automatically distribute N voices in a semicircle at varying distances.
///
/// Front voices are closer to the listener (louder, brighter). Back voices
/// are farther (quieter, more filtered). The layout forms concentric arcs:
///
/// ```text
///         back row (far, quiet)
///       o       o       o
///     o     o       o     o
///       front row (close, loud)
///              [listener]
/// ```
///
/// # Arguments
///
/// * `n_voices` — Number of voices to place (1..=32).
/// * `config` — Spatial config providing room_size and listener_distance.
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` if `n_voices` is 0 or > 32.
pub fn auto_layout_spatial(
    n_voices: usize,
    config: &SpatialConfig,
) -> Result<Vec<VoiceSpatialPosition>, KokoroError> {
    if n_voices == 0 || n_voices > 32 {
        return Err(KokoroError::InvalidConfig {
            field: "n_voices",
            reason: format!("n_voices = {n_voices}: must be in [1, 32]"),
        });
    }
    config.validate()?;

    let mut positions = Vec::with_capacity(n_voices);

    // Distance range: front row at listener_distance + 0.5m,
    // back row at 80% of room_size.
    let front_dist = (config.listener_distance + 0.5).max(MIN_DISTANCE);
    let back_dist = (config.room_size * 0.8).max(front_dist + 0.5);

    // Angle spread: semicircle from -60 to +60 degrees.
    let max_angle: f32 = std::f32::consts::FRAC_PI_3; // 60 degrees

    for i in 0..n_voices {
        let t = if n_voices == 1 {
            0.5
        } else {
            i as f32 / (n_voices - 1) as f32
        };

        // Alternate front/back: even indices in front row, odd in back.
        let row_t = if n_voices <= 2 {
            t
        } else if i % 2 == 0 {
            // Front row: closer distances
            (i as f32 / n_voices as f32) * 0.4
        } else {
            // Back row: farther distances
            0.5 + (i as f32 / n_voices as f32) * 0.5
        };

        let distance = front_dist + (back_dist - front_dist) * row_t;

        // Spread voices across the semicircle.
        // Center voice at angle 0, edges at +/- max_angle.
        let angle = if n_voices == 1 {
            0.0
        } else {
            let normalized = (i as f32 / (n_voices - 1) as f32) * 2.0 - 1.0;
            normalized * max_angle
        };

        // Slight elevation for back-row voices (choir risers effect).
        let elevation = row_t * 0.15; // up to ~8.6 degrees for furthest

        // Finite-check all computed values
        let distance = if distance.is_finite() {
            distance.clamp(MIN_DISTANCE, config.room_size)
        } else {
            front_dist
        };
        let angle = if angle.is_finite() { angle } else { 0.0 };
        let elevation = if elevation.is_finite() {
            elevation.clamp(-std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_4)
        } else {
            0.0
        };

        positions.push(VoiceSpatialPosition {
            distance,
            angle,
            elevation,
        });
    }

    Ok(positions)
}

// ---------------------------------------------------------------------------
// process_voice_spatial
// ---------------------------------------------------------------------------

/// Process a single mono voice through the spatial depth pipeline.
///
/// Applies all spatial effects (distance attenuation, air absorption lowpass,
/// propagation delay, ILD stereo panning) and returns a stereo pair
/// `(left, right)`.
///
/// This is a convenience function that creates a [`SpatialProcessor`] and
/// processes the entire buffer. For streaming / chunked processing, create
/// a `SpatialProcessor` directly and call [`SpatialProcessor::process`]
/// per chunk.
///
/// # Arguments
///
/// * `mono` — Mono PCM samples for one voice.
/// * `config` — Spatial configuration.
/// * `position` — This voice's spatial position.
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` if config or position is invalid.
pub fn process_voice_spatial(
    mono: &[f32],
    config: &SpatialConfig,
    position: &VoiceSpatialPosition,
) -> Result<(Vec<f32>, Vec<f32>), KokoroError> {
    let mut proc = SpatialProcessor::new(config, position)?;
    Ok(proc.process(mono))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_config_default_valid() {
        let config = SpatialConfig::new();
        config.validate().expect("default config should be valid");
    }

    #[test]
    fn test_spatial_config_invalid_room_size() {
        let config = SpatialConfig::new().with_room_size(-1.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_spatial_config_nan_rejected() {
        let config = SpatialConfig::new().with_room_size(f32::NAN);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_voice_position_validation() {
        let pos = VoiceSpatialPosition::new(2.0, 0.0, 0.0);
        pos.validate(8.0).expect("valid position");

        // Distance exceeds room size
        let pos = VoiceSpatialPosition::new(10.0, 0.0, 0.0);
        assert!(pos.validate(8.0).is_err());

        // Non-finite angle
        let pos = VoiceSpatialPosition::new(2.0, f32::INFINITY, 0.0);
        assert!(pos.validate(8.0).is_err());
    }

    #[test]
    fn test_auto_layout_single_voice() {
        let config = SpatialConfig::new();
        let positions = auto_layout_spatial(1, &config).expect("single voice layout");
        assert_eq!(positions.len(), 1);
        // Single voice should be centered
        assert!((positions[0].angle).abs() < 1e-6);
    }

    #[test]
    fn test_auto_layout_multiple_voices() {
        let config = SpatialConfig::new();
        let positions = auto_layout_spatial(4, &config).expect("4-voice layout");
        assert_eq!(positions.len(), 4);

        // All distances should be within room bounds
        for pos in &positions {
            assert!(pos.distance >= MIN_DISTANCE);
            assert!(pos.distance <= config.room_size);
            assert!(pos.angle.is_finite());
            assert!(pos.elevation.is_finite());
        }
    }

    #[test]
    fn test_auto_layout_zero_voices_rejected() {
        let config = SpatialConfig::new();
        assert!(auto_layout_spatial(0, &config).is_err());
    }

    #[test]
    fn test_auto_layout_too_many_voices_rejected() {
        let config = SpatialConfig::new();
        assert!(auto_layout_spatial(33, &config).is_err());
    }

    #[test]
    fn test_process_voice_spatial_silence() {
        let config = SpatialConfig::new();
        let pos = VoiceSpatialPosition::new(2.0, 0.0, 0.0);
        let mono = vec![0.0f32; 480];
        let (left, right) = process_voice_spatial(&mono, &config, &pos).expect("process silence");
        assert_eq!(left.len(), 480);
        assert_eq!(right.len(), 480);
        // Silence in, silence out
        for &s in &left {
            assert!((s).abs() < 1e-10);
        }
        for &s in &right {
            assert!((s).abs() < 1e-10);
        }
    }

    #[test]
    fn test_process_voice_spatial_attenuation() {
        let config = SpatialConfig::new();
        // Near voice
        let near = VoiceSpatialPosition::new(0.5, 0.0, 0.0);
        // Far voice
        let far = VoiceSpatialPosition::new(6.0, 0.0, 0.0);

        // Impulse with enough lead-in for delay to flush
        let mut mono = vec![0.0f32; 1000];
        mono[0] = 1.0;

        let (near_l, _) = process_voice_spatial(&mono, &config, &near).expect("near voice");
        let (far_l, _) = process_voice_spatial(&mono, &config, &far).expect("far voice");

        // Near voice should have higher peak energy than far voice
        let near_peak: f32 = near_l.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let far_peak: f32 = far_l.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

        assert!(
            near_peak > far_peak,
            "near peak {near_peak} should exceed far peak {far_peak}",
        );
    }

    #[test]
    fn test_process_voice_spatial_stereo_panning() {
        let config = SpatialConfig::new();
        // Voice panned hard left (negative angle), close distance to minimize delay
        let left_pos = VoiceSpatialPosition::new(0.5, -std::f32::consts::FRAC_PI_2, 0.0);
        // Use enough samples to exceed propagation delay (~35 samples at 0.5m)
        let mono = vec![1.0f32; 500];
        let (left, right) = process_voice_spatial(&mono, &config, &left_pos).expect("left pan");

        // Left channel should have more energy than right for left-panned voice
        let left_energy: f32 = left.iter().map(|s| s * s).sum();
        let right_energy: f32 = right.iter().map(|s| s * s).sum();
        assert!(
            left_energy > right_energy,
            "left energy {left_energy} should exceed right energy {right_energy} for left pan",
        );
    }

    #[test]
    fn test_process_voice_spatial_nan_defense() {
        let config = SpatialConfig::new();
        let pos = VoiceSpatialPosition::new(2.0, 0.0, 0.0);
        let mono = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0, 0.0];
        let (left, right) = process_voice_spatial(&mono, &config, &pos).expect("nan defense");

        // All outputs must be finite
        for &s in &left {
            assert!(s.is_finite(), "left sample {s} is not finite");
        }
        for &s in &right {
            assert!(s.is_finite(), "right sample {s} is not finite");
        }
    }

    #[test]
    fn test_spatial_processor_delay() {
        let config = SpatialConfig::new().with_sample_rate(24000.0);
        // At 5 meters, delay ~ 5/343 * 24000 ~ 349.9 samples ~ 350
        let pos = VoiceSpatialPosition::new(5.0, 0.0, 0.0);
        let proc = SpatialProcessor::new(&config, &pos).expect("processor creation");
        let expected_delay = (5.0 / SPEED_OF_SOUND * 24000.0) as usize;
        assert!(
            (proc.delay_samples() as i32 - expected_delay as i32).unsigned_abs() <= 1,
            "delay {} should be close to expected {}",
            proc.delay_samples(),
            expected_delay,
        );
    }

    #[test]
    fn test_spatial_processor_reset() {
        let config = SpatialConfig::new();
        let pos = VoiceSpatialPosition::new(2.0, 0.0, 0.0);
        let mut proc = SpatialProcessor::new(&config, &pos).expect("processor creation");

        // Process some audio
        let _ = proc.process(&[1.0, 0.5, 0.25]);
        // Reset clears state
        proc.reset();
        assert!((proc.lpf_state).abs() < 1e-10);
    }

    #[test]
    fn test_process_voice_spatial_empty_input() {
        let config = SpatialConfig::new();
        let pos = VoiceSpatialPosition::new(2.0, 0.0, 0.0);
        let (left, right) = process_voice_spatial(&[], &config, &pos).expect("empty input");
        assert!(left.is_empty());
        assert!(right.is_empty());
    }
}
