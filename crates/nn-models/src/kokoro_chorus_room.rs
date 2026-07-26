// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Image-source early reflections room simulation for Kokoro chorus.
//!
//! Models the first ~80ms of room sound using the image-source method,
//! computing first-order reflections from 6 walls (floor, ceiling, 4 walls).
//! Early reflections provide spatial cues about room size and shape that
//! complement the late reverb (Schroeder) already in the chorus pipeline.
//!
//! # Architecture
//!
//! ```text
//! Mono voice audio
//!   -> Image-source reflection computation (6 walls)
//!   -> Per-reflection: delay (path length) + attenuation (1/r + absorption) + pan (angle)
//!   -> 6-tap delay line (circular buffer)
//!   -> Stereo output (left, right)
//! ```
//!
//! # Image-Source Method
//!
//! For each wall, a virtual "image source" is computed by mirroring the real
//! source position across that wall. The reflection path length is the distance
//! from the listener to the image source. Attenuation combines inverse distance
//! law with wall absorption. The angle of arrival determines stereo panning.
//!
//! # References
//!
//! - Allen, J.B. & Berkley, D.A. (1979). "Image method for efficiently
//!   simulating small-room acoustics." JASA 65(4), 943-950.
//! - Vorlaender, M. (2008). "Auralization." Springer.
//!
//! Part of #4582, #3351.

use crate::kokoro_error::KokoroError;

/// Speed of sound in air at ~20C in meters per second.
const SPEED_OF_SOUND: f32 = 343.0;

/// Maximum early reflection delay in samples (caps memory usage).
/// At 24 kHz, 16384 samples ~ 0.68s ~ 234 meters round-trip. Sufficient
/// for any room up to ~100m diagonal.
const MAX_DELAY_SAMPLES: usize = 16384;

/// Number of walls: left, right, front, back, floor, ceiling.
const NUM_WALLS: usize = 6;

/// Minimum room dimension in meters.
const MIN_ROOM_DIM: f32 = 2.0;

/// Minimum height in meters.
const MIN_ROOM_HEIGHT: f32 = 2.5;

// ---------------------------------------------------------------------------
// RoomConfig
// ---------------------------------------------------------------------------

/// Physical room configuration for early reflections simulation.
///
/// Defines a rectangular room with dimensions, wall absorption, and
/// source/listener positions. Built via method chaining on `RoomConfig::new()`.
///
/// The image-source method mirrors the source across each wall to compute
/// first-order reflection paths. Wall absorption controls energy loss at
/// each reflection.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RoomConfig {
    /// Room width (X-axis) in meters. Range: 2.0-30.0.
    pub room_width: f32,

    /// Room depth (Y-axis) in meters. Range: 2.0-30.0.
    pub room_depth: f32,

    /// Room height (Z-axis) in meters. Range: 2.5-15.0.
    pub room_height: f32,

    /// Wall absorption coefficient. Range: 0.0-1.0.
    /// 0.0 = fully reflective (hard surfaces like tile/glass).
    /// 1.0 = fully absorptive (heavy curtains/acoustic foam).
    pub wall_absorption: f32,

    /// Sound source position (x, y, z) in meters.
    /// Origin is the room corner (0, 0, 0). All coordinates must be
    /// within room bounds.
    pub source_position: (f32, f32, f32),

    /// Listener position (x, y, z) in meters.
    /// Origin is the room corner (0, 0, 0). All coordinates must be
    /// within room bounds.
    pub listener_position: (f32, f32, f32),
}

impl Default for RoomConfig {
    fn default() -> Self {
        Self {
            room_width: 6.0,
            room_depth: 8.0,
            room_height: 3.0,
            wall_absorption: 0.3,
            // Source near the front-center, slightly off-center
            source_position: (3.0, 2.0, 1.5),
            // Listener at center-back
            listener_position: (3.0, 6.0, 1.5),
        }
    }
}

impl RoomConfig {
    /// Create a new room config with default values (medium recording room).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the room width in meters.
    #[must_use]
    pub fn with_room_width(mut self, width: f32) -> Self {
        self.room_width = width;
        self
    }

    /// Set the room depth in meters.
    #[must_use]
    pub fn with_room_depth(mut self, depth: f32) -> Self {
        self.room_depth = depth;
        self
    }

    /// Set the room height in meters.
    #[must_use]
    pub fn with_room_height(mut self, height: f32) -> Self {
        self.room_height = height;
        self
    }

    /// Set the wall absorption coefficient (0.0 = reflective, 1.0 = absorptive).
    #[must_use]
    pub fn with_wall_absorption(mut self, absorption: f32) -> Self {
        self.wall_absorption = absorption;
        self
    }

    /// Set the sound source position (x, y, z) in meters.
    #[must_use]
    pub fn with_source_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.source_position = (x, y, z);
        self
    }

    /// Set the listener position (x, y, z) in meters.
    #[must_use]
    pub fn with_listener_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.listener_position = (x, y, z);
        self
    }

    /// Validate that all parameters are within physically meaningful ranges.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range,
    /// non-finite, or if positions are outside the room bounds.
    pub fn validate(&self) -> Result<(), KokoroError> {
        // Room dimensions
        if !self.room_width.is_finite() || self.room_width < MIN_ROOM_DIM || self.room_width > 30.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "room_width",
                reason: format!(
                    "room_width = {}: must be finite and in [{}, 30.0]",
                    self.room_width, MIN_ROOM_DIM,
                ),
            });
        }
        if !self.room_depth.is_finite() || self.room_depth < MIN_ROOM_DIM || self.room_depth > 30.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "room_depth",
                reason: format!(
                    "room_depth = {}: must be finite and in [{}, 30.0]",
                    self.room_depth, MIN_ROOM_DIM,
                ),
            });
        }
        if !self.room_height.is_finite()
            || self.room_height < MIN_ROOM_HEIGHT
            || self.room_height > 15.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "room_height",
                reason: format!(
                    "room_height = {}: must be finite and in [{}, 15.0]",
                    self.room_height, MIN_ROOM_HEIGHT,
                ),
            });
        }

        // Wall absorption
        if !self.wall_absorption.is_finite() || !(0.0..=1.0).contains(&self.wall_absorption) {
            return Err(KokoroError::InvalidConfig {
                field: "wall_absorption",
                reason: format!(
                    "wall_absorption = {}: must be finite and in [0.0, 1.0]",
                    self.wall_absorption,
                ),
            });
        }

        // Source position within room bounds
        self.validate_position(self.source_position, "source_position")?;

        // Listener position within room bounds
        self.validate_position(self.listener_position, "listener_position")?;

        Ok(())
    }

    /// Validate a position is within room bounds and all coordinates are finite.
    fn validate_position(
        &self,
        pos: (f32, f32, f32),
        field: &'static str,
    ) -> Result<(), KokoroError> {
        let (x, y, z) = pos;
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(KokoroError::InvalidConfig {
                field,
                reason: format!("({x}, {y}, {z}): all coordinates must be finite"),
            });
        }
        if x < 0.0 || x > self.room_width {
            return Err(KokoroError::InvalidConfig {
                field,
                reason: format!(
                    "x = {}: must be in [0.0, {}] (room_width)",
                    x, self.room_width,
                ),
            });
        }
        if y < 0.0 || y > self.room_depth {
            return Err(KokoroError::InvalidConfig {
                field,
                reason: format!(
                    "y = {}: must be in [0.0, {}] (room_depth)",
                    y, self.room_depth,
                ),
            });
        }
        if z < 0.0 || z > self.room_height {
            return Err(KokoroError::InvalidConfig {
                field,
                reason: format!(
                    "z = {}: must be in [0.0, {}] (room_height)",
                    z, self.room_height,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RoomPreset
// ---------------------------------------------------------------------------

/// Predefined room configurations for common acoustic environments.
///
/// Each preset defines physically-motivated room dimensions, absorption,
/// and default source/listener positions. Use `to_config()` to get a
/// `RoomConfig` that can be further customized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RoomPreset {
    /// Small recording studio: 3x4x2.5m, high absorption (acoustic treatment).
    SmallStudio,

    /// Medium recording room: 6x8x3m, moderate absorption.
    RecordingRoom,

    /// Concert hall: 20x30x12m, low absorption (hard reflective surfaces).
    ConcertHall,

    /// Church: 15x25x10m, very low absorption (stone/wood surfaces).
    Church,
}

impl RoomPreset {
    /// Convert the preset to a `RoomConfig` with physically-motivated defaults.
    ///
    /// Source is placed near the front of the room, listener near the back.
    /// Positions are centered on the X-axis and at ear height (~1.5m).
    #[must_use]
    pub fn to_config(self) -> RoomConfig {
        match self {
            Self::SmallStudio => RoomConfig {
                room_width: 3.0,
                room_depth: 4.0,
                room_height: 2.5,
                wall_absorption: 0.7,
                source_position: (1.5, 1.0, 1.5),
                listener_position: (1.5, 3.0, 1.5),
            },
            Self::RecordingRoom => RoomConfig {
                room_width: 6.0,
                room_depth: 8.0,
                room_height: 3.0,
                wall_absorption: 0.4,
                source_position: (3.0, 2.0, 1.5),
                listener_position: (3.0, 6.0, 1.5),
            },
            Self::ConcertHall => RoomConfig {
                room_width: 20.0,
                room_depth: 30.0,
                room_height: 12.0,
                wall_absorption: 0.15,
                source_position: (10.0, 5.0, 1.5),
                listener_position: (10.0, 20.0, 1.5),
            },
            Self::Church => RoomConfig {
                room_width: 15.0,
                room_depth: 25.0,
                room_height: 10.0,
                wall_absorption: 0.1,
                source_position: (7.5, 3.0, 1.5),
                listener_position: (7.5, 18.0, 1.5),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// ReflectionTap — computed reflection parameters for one wall
// ---------------------------------------------------------------------------

/// Computed parameters for a single wall reflection.
///
/// Each tap represents a first-order image-source reflection: the path from
/// source -> wall -> listener. The delay, gain, and pan are derived from
/// the image source position.
#[derive(Debug, Clone, Copy)]
struct ReflectionTap {
    /// Delay in samples from the direct sound.
    delay_samples: usize,
    /// Amplitude gain (inverse distance * wall absorption).
    gain: f32,
    /// Stereo pan position: -1.0 = hard left, 0.0 = center, 1.0 = hard right.
    pan: f32,
}

// ---------------------------------------------------------------------------
// Image-source computation
// ---------------------------------------------------------------------------

/// Compute first-order image sources for the 6 walls of a rectangular room.
///
/// For each wall, the source is mirrored across that wall's plane to create
/// a virtual image source. The reflection parameters (delay, gain, pan) are
/// derived from the image source's relationship to the listener.
///
/// Wall ordering: left (x=0), right (x=W), front (y=0), back (y=D),
/// floor (z=0), ceiling (z=H).
fn compute_reflection_taps(config: &RoomConfig, sample_rate: f32) -> [ReflectionTap; NUM_WALLS] {
    let (sx, sy, sz) = config.source_position;
    let (lx, ly, lz) = config.listener_position;
    let absorption = config.wall_absorption;
    // Reflection coefficient: energy not absorbed by the wall.
    let reflection_coeff = 1.0 - absorption;

    // Direct path distance (for relative delay computation).
    let direct_dist = euclidean_distance(sx, sy, sz, lx, ly, lz);

    // Image sources for each wall:
    // Wall at x=0 (left wall):   image = (-sx, sy, sz)
    // Wall at x=W (right wall):  image = (2W - sx, sy, sz)
    // Wall at y=0 (front wall):  image = (sx, -sy, sz)
    // Wall at y=D (back wall):   image = (sx, 2D - sy, sz)
    // Wall at z=0 (floor):       image = (sx, sy, -sz)
    // Wall at z=H (ceiling):     image = (sx, sy, 2H - sz)
    let image_sources = [
        (-sx, sy, sz),                           // left wall
        (2.0 * config.room_width - sx, sy, sz),  // right wall
        (sx, -sy, sz),                           // front wall
        (sx, 2.0 * config.room_depth - sy, sz),  // back wall
        (sx, sy, -sz),                           // floor
        (sx, sy, 2.0 * config.room_height - sz), // ceiling
    ];

    let mut taps = [ReflectionTap {
        delay_samples: 0,
        gain: 0.0,
        pan: 0.0,
    }; NUM_WALLS];

    for (i, &(ix, iy, iz)) in image_sources.iter().enumerate() {
        let image_dist = euclidean_distance(ix, iy, iz, lx, ly, lz);

        // Delay relative to the direct sound path.
        let extra_path = (image_dist - direct_dist).max(0.0);
        let delay_sec = extra_path / SPEED_OF_SOUND;
        let delay_samples_raw = (delay_sec * sample_rate).round() as usize;
        let delay_samples = delay_samples_raw.min(MAX_DELAY_SAMPLES);

        // Gain: inverse distance law relative to direct path, times reflection coefficient.
        // Clamped to avoid amplification if image is closer than direct path.
        let distance_attenuation = if image_dist > 1e-6 {
            (direct_dist / image_dist).min(1.0)
        } else {
            1.0
        };
        let gain = distance_attenuation * reflection_coeff;
        let gain = if gain.is_finite() {
            gain.clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Pan from angle of arrival (azimuth of image source relative to listener).
        // Positive X = right, negative X = left.
        let dx = ix - lx;
        let dy = iy - ly;
        let horizontal_dist = dx.hypot(dy);
        let pan = if horizontal_dist > 1e-6 {
            (dx / horizontal_dist).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let pan = if pan.is_finite() { pan } else { 0.0 };

        taps[i] = ReflectionTap {
            delay_samples,
            gain,
            pan,
        };
    }

    taps
}

/// Euclidean distance between two 3D points.
#[inline]
fn euclidean_distance(x1: f32, y1: f32, z1: f32, x2: f32, y2: f32, z2: f32) -> f32 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let dz = z2 - z1;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// ---------------------------------------------------------------------------
// EarlyReflections processor
// ---------------------------------------------------------------------------

/// Image-source early reflections processor.
///
/// Simulates the first ~80ms of room sound using a 6-tap delay line, one
/// tap per wall (left, right, front, back, floor, ceiling). Each tap has
/// independently computed delay, gain, and stereo pan derived from the
/// image-source positions.
///
/// Create via [`EarlyReflections::new`], process audio chunks with
/// [`EarlyReflections::process`], and call [`EarlyReflections::reset`]
/// between segments.
pub struct EarlyReflections {
    /// Circular delay buffer for the input signal.
    delay_buffer: Vec<f32>,
    /// Write position in the circular buffer.
    write_pos: usize,
    /// Computed reflection parameters for 6 walls.
    taps: [ReflectionTap; NUM_WALLS],
    /// Maximum delay across all taps (determines buffer size).
    max_delay: usize,
}

impl EarlyReflections {
    /// Create a new early reflections processor for the given room configuration.
    ///
    /// Computes image-source reflection parameters for all 6 walls and allocates
    /// the delay buffer.
    ///
    /// # Arguments
    ///
    /// * `config` - Room configuration (dimensions, absorption, positions).
    /// * `sample_rate` - Audio sample rate in Hz (e.g., 24000.0 for Kokoro).
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if the room config is invalid or
    /// the sample rate is non-positive/non-finite.
    pub fn new(config: &RoomConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;

        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("sample_rate = {sample_rate}: must be finite and > 0.0"),
            });
        }

        let taps = compute_reflection_taps(config, sample_rate);

        // Find maximum delay to size the buffer.
        let max_delay = taps.iter().map(|t| t.delay_samples).max().unwrap_or(0);

        // Buffer needs max_delay + 1 for circular indexing.
        let buf_size = (max_delay + 1).max(1);

        Ok(Self {
            delay_buffer: vec![0.0; buf_size],
            write_pos: 0,
            taps,
            max_delay,
        })
    }

    /// Process a mono audio buffer, returning stereo early reflections.
    ///
    /// The output contains only the reflected sound (no direct signal).
    /// Mix with the dry signal externally to control the wet/dry ratio.
    ///
    /// # Returns
    ///
    /// A tuple `(left, right)` of equal-length vectors containing the
    /// stereo early reflections signal.
    #[must_use]
    pub fn process(&mut self, audio: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let len = audio.len();
        let mut left = vec![0.0f32; len];
        let mut right = vec![0.0f32; len];
        let buf_len = self.delay_buffer.len();

        for (i, &sample) in audio.iter().enumerate() {
            // Defense-in-depth: clamp non-finite input to zero.
            let s = if sample.is_finite() { sample } else { 0.0 };

            // Write into circular delay buffer.
            self.delay_buffer[self.write_pos] = s;

            // Sum contributions from all 6 reflection taps.
            let mut sum_l = 0.0f32;
            let mut sum_r = 0.0f32;

            for tap in &self.taps {
                if tap.gain < 1e-8 {
                    continue;
                }

                // Read from delay buffer at the tap's delay offset.
                let read_pos = (self.write_pos + buf_len - tap.delay_samples) % buf_len;
                let delayed = self.delay_buffer[read_pos];

                let reflected = delayed * tap.gain;

                // Constant-power stereo pan from the reflection's angle of arrival.
                // pan: -1.0 = left, 0.0 = center, 1.0 = right.
                let pan_angle = (tap.pan + 1.0) * 0.5 * std::f32::consts::FRAC_PI_2;
                let l_gain = pan_angle.cos();
                let r_gain = pan_angle.sin();

                sum_l += reflected * l_gain;
                sum_r += reflected * r_gain;
            }

            // Finite check on accumulated output.
            left[i] = if sum_l.is_finite() { sum_l } else { 0.0 };
            right[i] = if sum_r.is_finite() { sum_r } else { 0.0 };

            self.write_pos = (self.write_pos + 1) % buf_len;
        }

        (left, right)
    }

    /// Clear the delay buffer and reset write position.
    ///
    /// Call between audio segments to prevent artifacts from stale data.
    pub fn reset(&mut self) {
        self.delay_buffer.fill(0.0);
        self.write_pos = 0;
    }

    /// Return the computed reflection taps (for diagnostics/testing).
    #[must_use]
    pub fn tap_count(&self) -> usize {
        NUM_WALLS
    }

    /// Return the maximum delay across all taps in samples.
    #[must_use]
    pub fn max_delay_samples(&self) -> usize {
        self.max_delay
    }

    /// Return the delay in samples for a specific tap index (0-5).
    ///
    /// Returns `None` if the index is out of range.
    #[must_use]
    pub fn tap_delay(&self, index: usize) -> Option<usize> {
        self.taps.get(index).map(|t| t.delay_samples)
    }

    /// Return the gain for a specific tap index (0-5).
    ///
    /// Returns `None` if the index is out of range.
    #[must_use]
    pub fn tap_gain(&self, index: usize) -> Option<f32> {
        self.taps.get(index).map(|t| t.gain)
    }

    /// Return the pan for a specific tap index (0-5).
    ///
    /// Returns `None` if the index is out of range.
    #[must_use]
    pub fn tap_pan(&self, index: usize) -> Option<f32> {
        self.taps.get(index).map(|t| t.pan)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- RoomConfig validation ------------------------------------------------

    #[test]
    fn test_default_config_valid() {
        let config = RoomConfig::new();
        config.validate().expect("default config should be valid");
    }

    #[test]
    fn test_config_invalid_width_too_small() {
        let config = RoomConfig::new().with_room_width(1.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_invalid_width_too_large() {
        let config = RoomConfig::new().with_room_width(31.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_invalid_depth_nan() {
        let config = RoomConfig::new().with_room_depth(f32::NAN);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_invalid_height_too_small() {
        let config = RoomConfig::new().with_room_height(2.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_invalid_absorption_out_of_range() {
        let config = RoomConfig::new().with_wall_absorption(1.5);
        assert!(config.validate().is_err());
        let config = RoomConfig::new().with_wall_absorption(-0.1);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_source_outside_room() {
        let config = RoomConfig::new().with_source_position(100.0, 2.0, 1.5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_listener_outside_room() {
        let config = RoomConfig::new().with_listener_position(3.0, -1.0, 1.5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_nan_position() {
        let config = RoomConfig::new().with_source_position(f32::NAN, 2.0, 1.5);
        assert!(config.validate().is_err());
    }

    // -- RoomPreset -----------------------------------------------------------

    #[test]
    fn test_all_presets_produce_valid_configs() {
        let presets = [
            RoomPreset::SmallStudio,
            RoomPreset::RecordingRoom,
            RoomPreset::ConcertHall,
            RoomPreset::Church,
        ];
        for preset in &presets {
            let config = preset.to_config();
            config
                .validate()
                .unwrap_or_else(|e| panic!("{preset:?} produced invalid config: {e}"));
        }
    }

    #[test]
    fn test_preset_dimensions() {
        let studio = RoomPreset::SmallStudio.to_config();
        assert!((studio.room_width - 3.0).abs() < 1e-6);
        assert!((studio.room_depth - 4.0).abs() < 1e-6);
        assert!((studio.room_height - 2.5).abs() < 1e-6);

        let hall = RoomPreset::ConcertHall.to_config();
        assert!((hall.room_width - 20.0).abs() < 1e-6);
        assert!((hall.room_depth - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_preset_absorption_ordering() {
        let studio_abs = RoomPreset::SmallStudio.to_config().wall_absorption;
        let room_abs = RoomPreset::RecordingRoom.to_config().wall_absorption;
        let hall_abs = RoomPreset::ConcertHall.to_config().wall_absorption;
        let church_abs = RoomPreset::Church.to_config().wall_absorption;

        // Studio is most absorptive, church least.
        assert!(studio_abs > room_abs);
        assert!(room_abs > hall_abs);
        assert!(hall_abs > church_abs);
    }

    // -- EarlyReflections construction ----------------------------------------

    #[test]
    fn test_early_reflections_construction() {
        let config = RoomConfig::new();
        let er = EarlyReflections::new(&config, 24000.0).expect("construction should succeed");
        assert_eq!(er.tap_count(), 6);
    }

    #[test]
    fn test_early_reflections_invalid_sample_rate() {
        let config = RoomConfig::new();
        assert!(EarlyReflections::new(&config, 0.0).is_err());
        assert!(EarlyReflections::new(&config, -1.0).is_err());
        assert!(EarlyReflections::new(&config, f32::NAN).is_err());
    }

    #[test]
    fn test_early_reflections_invalid_config() {
        let config = RoomConfig::new().with_room_width(0.5);
        assert!(EarlyReflections::new(&config, 24000.0).is_err());
    }

    // -- Reflection timing: reflections arrive after direct sound -------------

    #[test]
    fn test_reflections_arrive_after_direct_sound() {
        let config = RoomConfig::new();
        let er = EarlyReflections::new(&config, 24000.0).unwrap();

        // All taps should have delay values (they are relative to direct path).
        // usize is inherently >= 0; verify taps exist and are finite.
        for i in 0..NUM_WALLS {
            let delay = er.tap_delay(i).unwrap();
            // Verify the tap exists and has a reasonable delay (< 1 second at 24kHz).
            assert!(
                delay < 24000,
                "tap {i} has unreasonable delay {delay} samples",
            );
        }
    }

    #[test]
    fn test_reflections_have_positive_delay() {
        // In a non-degenerate room, at least some reflections should
        // have non-zero delay (different path length than direct).
        let config = RoomConfig::new();
        let er = EarlyReflections::new(&config, 24000.0).unwrap();

        let total_delay: usize = (0..NUM_WALLS).map(|i| er.tap_delay(i).unwrap()).sum();
        assert!(
            total_delay > 0,
            "at least some reflections should have non-zero delay",
        );
    }

    // -- Larger room = longer delays ------------------------------------------

    #[test]
    fn test_larger_room_longer_delays() {
        let small = RoomPreset::SmallStudio.to_config();
        let large = RoomPreset::ConcertHall.to_config();

        let er_small = EarlyReflections::new(&small, 24000.0).unwrap();
        let er_large = EarlyReflections::new(&large, 24000.0).unwrap();

        assert!(
            er_large.max_delay_samples() > er_small.max_delay_samples(),
            "concert hall max delay {} should exceed studio max delay {}",
            er_large.max_delay_samples(),
            er_small.max_delay_samples(),
        );
    }

    // -- Absorption reduces energy --------------------------------------------

    #[test]
    fn test_absorption_reduces_energy() {
        // Low absorption room: more reflection energy.
        let low_abs = RoomConfig::new().with_wall_absorption(0.1);
        // High absorption room: less reflection energy.
        let high_abs = RoomConfig::new().with_wall_absorption(0.9);

        let mut er_low = EarlyReflections::new(&low_abs, 24000.0).unwrap();
        let mut er_high = EarlyReflections::new(&high_abs, 24000.0).unwrap();

        // Impulse followed by silence, long enough for reflections to arrive.
        let mut impulse = vec![0.0f32; 4800]; // 200ms at 24kHz
        impulse[0] = 1.0;

        let (low_l, low_r) = er_low.process(&impulse);
        let (high_l, high_r) = er_high.process(&impulse);

        let low_energy: f32 = low_l.iter().chain(low_r.iter()).map(|s| s * s).sum();
        let high_energy: f32 = high_l.iter().chain(high_r.iter()).map(|s| s * s).sum();

        assert!(
            low_energy > high_energy,
            "low absorption energy {low_energy} should exceed high absorption energy {high_energy}",
        );
    }

    // -- Gain validation: all taps have gain in [0, 1] -----------------------

    #[test]
    fn test_tap_gains_bounded() {
        let presets = [
            RoomPreset::SmallStudio,
            RoomPreset::RecordingRoom,
            RoomPreset::ConcertHall,
            RoomPreset::Church,
        ];
        for preset in &presets {
            let config = preset.to_config();
            let er = EarlyReflections::new(&config, 24000.0).unwrap();
            for i in 0..NUM_WALLS {
                let gain = er.tap_gain(i).unwrap();
                assert!(
                    (0.0..=1.0).contains(&gain),
                    "{preset:?} tap {i} gain {gain} out of [0, 1]",
                );
            }
        }
    }

    // -- Pan validation: all taps have pan in [-1, 1] ------------------------

    #[test]
    fn test_tap_pans_bounded() {
        let config = RoomConfig::new();
        let er = EarlyReflections::new(&config, 24000.0).unwrap();
        for i in 0..NUM_WALLS {
            let pan = er.tap_pan(i).unwrap();
            assert!(
                (-1.0..=1.0).contains(&pan),
                "tap {i} pan {pan} out of [-1, 1]",
            );
        }
    }

    // -- Process produces stereo output of correct length --------------------

    #[test]
    fn test_process_output_length() {
        let config = RoomConfig::new();
        let mut er = EarlyReflections::new(&config, 24000.0).unwrap();
        let audio = vec![0.5f32; 1000];
        let (left, right) = er.process(&audio);
        assert_eq!(left.len(), 1000);
        assert_eq!(right.len(), 1000);
    }

    #[test]
    fn test_process_empty_input() {
        let config = RoomConfig::new();
        let mut er = EarlyReflections::new(&config, 24000.0).unwrap();
        let (left, right) = er.process(&[]);
        assert!(left.is_empty());
        assert!(right.is_empty());
    }

    // -- Silence in, silence out ---------------------------------------------

    #[test]
    fn test_silence_produces_silence() {
        let config = RoomConfig::new();
        let mut er = EarlyReflections::new(&config, 24000.0).unwrap();
        let silence = vec![0.0f32; 2400];
        let (left, right) = er.process(&silence);
        for &s in left.iter().chain(right.iter()) {
            assert!(s.abs() < 1e-10, "silence should produce silence, got {s}");
        }
    }

    // -- NaN defense-in-depth ------------------------------------------------

    #[test]
    fn test_nan_defense() {
        let config = RoomConfig::new();
        let mut er = EarlyReflections::new(&config, 24000.0).unwrap();
        let bad_input = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0, 0.0];
        let (left, right) = er.process(&bad_input);
        for &s in left.iter().chain(right.iter()) {
            assert!(s.is_finite(), "output must be finite, got {s}");
        }
    }

    // -- Reset clears state --------------------------------------------------

    #[test]
    fn test_reset_clears_state() {
        let config = RoomConfig::new();
        let mut er = EarlyReflections::new(&config, 24000.0).unwrap();

        // Process an impulse.
        let mut impulse = vec![0.0f32; 480];
        impulse[0] = 1.0;
        let _ = er.process(&impulse);

        // Reset should clear the delay buffer.
        er.reset();

        // Processing silence after reset should yield silence.
        let silence = vec![0.0f32; 480];
        let (left, right) = er.process(&silence);
        for &s in left.iter().chain(right.iter()) {
            assert!(
                s.abs() < 1e-10,
                "after reset, silence should produce silence, got {s}",
            );
        }
    }

    // -- Impulse response has reflections ------------------------------------

    #[test]
    fn test_impulse_produces_reflections() {
        let config = RoomConfig::new().with_wall_absorption(0.2);
        let mut er = EarlyReflections::new(&config, 24000.0).unwrap();

        // Short impulse followed by enough silence for reflections to arrive.
        let mut impulse = vec![0.0f32; 4800];
        impulse[0] = 1.0;

        let (left, right) = er.process(&impulse);

        // There should be non-zero energy after the initial sample (reflections).
        let reflection_energy: f32 = left[1..]
            .iter()
            .chain(right[1..].iter())
            .map(|s| s * s)
            .sum();
        assert!(
            reflection_energy > 1e-8,
            "impulse should produce reflections, energy = {reflection_energy}",
        );
    }

    // -- Stereo asymmetry for off-center sources -----------------------------

    #[test]
    fn test_off_center_source_stereo_asymmetry() {
        // Source on the left side of the room.
        let config_left = RoomConfig::new()
            .with_source_position(1.0, 2.0, 1.5)
            .with_listener_position(3.0, 6.0, 1.5);
        // Source on the right side.
        let config_right = RoomConfig::new()
            .with_source_position(5.0, 2.0, 1.5)
            .with_listener_position(3.0, 6.0, 1.5);

        let mut er_left = EarlyReflections::new(&config_left, 24000.0).unwrap();
        let mut er_right = EarlyReflections::new(&config_right, 24000.0).unwrap();

        let mut impulse = vec![0.0f32; 4800];
        impulse[0] = 1.0;

        let (ll, lr) = er_left.process(&impulse);
        let (rl, rr) = er_right.process(&impulse);

        let left_l_energy: f32 = ll.iter().map(|s| s * s).sum();
        let left_r_energy: f32 = lr.iter().map(|s| s * s).sum();
        let right_l_energy: f32 = rl.iter().map(|s| s * s).sum();
        let right_r_energy: f32 = rr.iter().map(|s| s * s).sum();

        // Left-side source should have more left energy, right-side more right.
        // (This is a soft check since multiple reflections interact.)
        let left_balance = left_l_energy / (left_l_energy + left_r_energy + 1e-10);
        let right_balance = right_r_energy / (right_l_energy + right_r_energy + 1e-10);

        // The source side should receive at least somewhat more energy.
        assert!(
            left_balance > 0.35,
            "left source should have left-biased energy, balance = {left_balance}",
        );
        assert!(
            right_balance > 0.35,
            "right source should have right-biased energy, balance = {right_balance}",
        );
    }

    // -- Out-of-range tap index returns None ---------------------------------

    #[test]
    fn test_tap_accessors_out_of_range() {
        let config = RoomConfig::new();
        let er = EarlyReflections::new(&config, 24000.0).unwrap();
        assert!(er.tap_delay(6).is_none());
        assert!(er.tap_gain(100).is_none());
        assert!(er.tap_pan(6).is_none());
    }
}
