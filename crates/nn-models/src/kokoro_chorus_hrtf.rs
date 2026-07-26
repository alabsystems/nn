// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HRTF (Head-Related Transfer Function) binaural spatial audio processor
//! for the Kokoro chorus system.
//!
//! Replaces simple pan-law stereo with perceptually accurate 3D positioning
//! using interaural time delay (ITD), interaural level difference (ILD),
//! and frequency-dependent head-shadow filtering.
//!
//! # Physical Model
//!
//! Uses the Woodworth spherical-head model for ITD:
//!   `tau = (r/c)(theta + sin(theta))`
//! where `r` = head radius, `c` = speed of sound, `theta` = azimuth.
//!
//! Head shadow ILD is modeled as a low-shelf filter whose gain depends on
//! azimuth and frequency, approximating the ~6 dB attenuation at 4 kHz
//! for a 90-degree source.
//!
//! # Usage
//!
//! ```text
//! let config = HrtfConfig::new()
//!     .with_positions(semicircle(4));
//! let mut proc = HrtfProcessor::new(&config, 24000.0)?;
//! let (left, right) = proc.process_voices(&voices)?;
//! ```
//!
//! Part of #4582, #3351.

use crate::kokoro_error::KokoroError;

/// Speed of sound in dry air at ~20 C (m/s).
const DEFAULT_SPEED_OF_SOUND: f32 = 343.0;

/// Average adult human head radius in meters (8.75 cm).
const DEFAULT_HEAD_RADIUS_M: f32 = 0.0875;

/// Maximum delay line length in samples. At 48 kHz this covers ~85 ms,
/// far beyond any physical ITD (~0.7 ms max).
const MAX_DELAY_SAMPLES: usize = 4096;

/// Minimum distance clamp to avoid division-by-zero (meters).
const MIN_DISTANCE: f32 = 0.1;

// ---------------------------------------------------------------------------
// HrtfModel
// ---------------------------------------------------------------------------

/// Head model used for ITD/ILD computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum HrtfModel {
    /// Woodworth spherical head model (geometric ITD/ILD, freq-dependent).
    ///
    /// More accurate: models frequency-dependent head shadow via low-shelf
    /// biquad filter.
    #[default]
    SphericalHead,

    /// Simple delay + level difference (faster, less accurate).
    ///
    /// Uses sine-law ITD and constant ILD without frequency dependence.
    SimpleDelay,
}


// ---------------------------------------------------------------------------
// HrtfPosition
// ---------------------------------------------------------------------------

/// Spatial position of a voice source relative to the listener.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HrtfPosition {
    /// Azimuth in degrees, range `[-180, 180]`. 0 = front, 90 = right,
    /// -90 = left, 180/-180 = behind.
    pub azimuth_deg: f32,

    /// Elevation in degrees, range `[-90, 90]`. 0 = ear level,
    /// 90 = directly above.
    pub elevation_deg: f32,

    /// Distance from the listener in meters. Affects attenuation and
    /// air absorption. Clamped to `[MIN_DISTANCE, ..]` during processing.
    pub distance_m: f32,
}

impl HrtfPosition {
    /// Create a new position at the given azimuth, elevation, and distance.
    #[must_use]
    pub fn new(azimuth_deg: f32, elevation_deg: f32, distance_m: f32) -> Self {
        Self {
            azimuth_deg,
            elevation_deg,
            distance_m,
        }
    }

    /// Front-center position at the given distance.
    #[must_use]
    pub fn front(distance_m: f32) -> Self {
        Self::new(0.0, 0.0, distance_m)
    }
}

// ---------------------------------------------------------------------------
// HrtfConfig
// ---------------------------------------------------------------------------

/// Configuration for the HRTF binaural processor.
///
/// Built via method chaining on [`HrtfConfig::new()`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HrtfConfig {
    /// Whether HRTF processing is enabled. When `false`, voices are passed
    /// through as dual-mono.
    pub enabled: bool,

    /// Head radius in centimeters. Default: `8.75` (average adult).
    pub head_radius_cm: f32,

    /// Speed of sound in m/s. Default: `343.0`.
    pub speed_of_sound: f32,

    /// Per-voice spatial positions. Must match the number of voices passed
    /// to [`HrtfProcessor::process_voices`].
    pub positions: Vec<HrtfPosition>,

    /// Head model to use for ITD/ILD computation.
    pub hrtf_model: HrtfModel,

    /// Number of samples for crossfading when positions change smoothly.
    /// Default: `64`.
    pub crossfade_samples: usize,
}

impl Default for HrtfConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            head_radius_cm: DEFAULT_HEAD_RADIUS_M * 100.0,
            speed_of_sound: DEFAULT_SPEED_OF_SOUND,
            positions: Vec::new(),
            hrtf_model: HrtfModel::default(),
            crossfade_samples: 64,
        }
    }
}

impl HrtfConfig {
    /// Create a new config with default parameters and no positions.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable HRTF processing.
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set head radius in centimeters.
    #[must_use]
    pub fn with_head_radius_cm(mut self, cm: f32) -> Self {
        self.head_radius_cm = cm;
        self
    }

    /// Set speed of sound in m/s.
    #[must_use]
    pub fn with_speed_of_sound(mut self, mps: f32) -> Self {
        self.speed_of_sound = mps;
        self
    }

    /// Set per-voice spatial positions.
    #[must_use]
    pub fn with_positions(mut self, positions: Vec<HrtfPosition>) -> Self {
        self.positions = positions;
        self
    }

    /// Set the head model variant.
    #[must_use]
    pub fn with_hrtf_model(mut self, model: HrtfModel) -> Self {
        self.hrtf_model = model;
        self
    }

    /// Set crossfade length in samples.
    #[must_use]
    pub fn with_crossfade_samples(mut self, n: usize) -> Self {
        self.crossfade_samples = n;
        self
    }

    /// Validate that all parameters are physically meaningful.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.head_radius_cm.is_finite() || self.head_radius_cm <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "head_radius_cm",
                reason: format!(
                    "head_radius_cm = {}: must be finite and > 0.0",
                    self.head_radius_cm,
                ),
            });
        }
        if !self.speed_of_sound.is_finite() || self.speed_of_sound <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "speed_of_sound",
                reason: format!(
                    "speed_of_sound = {}: must be finite and > 0.0",
                    self.speed_of_sound,
                ),
            });
        }
        for (i, pos) in self.positions.iter().enumerate() {
            if !pos.azimuth_deg.is_finite() {
                return Err(KokoroError::InvalidConfig {
                    field: "azimuth_deg",
                    reason: format!("position[{i}].azimuth_deg is non-finite"),
                });
            }
            if !pos.elevation_deg.is_finite() {
                return Err(KokoroError::InvalidConfig {
                    field: "elevation_deg",
                    reason: format!("position[{i}].elevation_deg is non-finite"),
                });
            }
            if !pos.distance_m.is_finite() || pos.distance_m < MIN_DISTANCE {
                return Err(KokoroError::InvalidConfig {
                    field: "distance_m",
                    reason: format!(
                        "position[{}].distance_m = {}: must be finite and >= {}",
                        i, pos.distance_m, MIN_DISTANCE,
                    ),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Preset layouts
// ---------------------------------------------------------------------------

/// Distribute `n` voices in a semicircle in front of the listener.
///
/// Azimuth spans from `-90` to `+90` degrees, all at ear level, 1.5 m away.
#[must_use]
pub fn semicircle(n_voices: usize) -> Vec<HrtfPosition> {
    arc(n_voices, 180.0)
}

/// Distribute `n` voices in an arc of `span_deg` degrees in front.
///
/// The arc is centered at 0 degrees (front). Each voice is at 1.5 m distance.
#[must_use]
pub fn arc(n_voices: usize, span_deg: f32) -> Vec<HrtfPosition> {
    if n_voices == 0 {
        return Vec::new();
    }
    let half = span_deg * 0.5;
    (0..n_voices)
        .map(|i| {
            let t = if n_voices == 1 {
                0.5
            } else {
                i as f32 / (n_voices - 1) as f32
            };
            let az = -half + t * span_deg;
            HrtfPosition::new(az, 0.0, 1.5)
        })
        .collect()
}

/// Distribute `n` voices in a full 360-degree surround at 2.0 m distance.
#[must_use]
pub fn surround(n_voices: usize) -> Vec<HrtfPosition> {
    if n_voices == 0 {
        return Vec::new();
    }
    (0..n_voices)
        .map(|i| {
            let az = (i as f32 / n_voices as f32) * 360.0 - 180.0;
            HrtfPosition::new(az, 0.0, 2.0)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// DelayLine — fractional delay with linear interpolation
// ---------------------------------------------------------------------------

struct DelayLine {
    buffer: Vec<f32>,
    write_pos: usize,
}

impl DelayLine {
    fn new(max_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; max_samples.max(2)],
            write_pos: 0,
        }
    }

    /// Write a sample and read back at a fractional delay (linear interpolation).
    fn process(&mut self, input: f32, delay_samples: f32) -> f32 {
        let s = if input.is_finite() { input } else { 0.0 };
        let len = self.buffer.len();
        self.buffer[self.write_pos] = s;

        let delay_clamped = delay_samples.clamp(0.0, (len - 1) as f32);
        let delay_int = delay_clamped as usize;
        let frac = delay_clamped - delay_int as f32;

        let idx0 = (self.write_pos + len - delay_int) % len;
        let idx1 = (self.write_pos + len - delay_int - 1) % len;

        let out = self.buffer[idx0] * (1.0 - frac) + self.buffer[idx1] * frac;

        self.write_pos = (self.write_pos + 1) % len;
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
// BiquadFilter — second-order IIR (direct form II transposed)
// ---------------------------------------------------------------------------

struct BiquadFilter {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl BiquadFilter {
    /// Create a low-shelf biquad filter.
    ///
    /// `gain_db`: shelf gain in dB (negative = attenuation).
    /// `freq_hz`: shelf transition frequency.
    /// `sample_rate`: audio sample rate.
    fn low_shelf(gain_db: f32, freq_hz: f32, sample_rate: f32) -> Self {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * freq_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / 2.0 * (2.0_f32).sqrt(); // Q = 1/sqrt(2)
        let sqrt_a = a.sqrt();

        let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
        if !a0.is_finite() || a0.abs() < 1e-12 {
            return Self::passthrough();
        }
        let inv_a0 = 1.0 / a0;

        let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha) * inv_a0;
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0) * inv_a0;
        let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha) * inv_a0;
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0) * inv_a0;
        let a2 = ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha) * inv_a0;

        Self::from_coeffs(b0, b1, b2, a1, a2)
    }

    /// Create a high-shelf biquad filter.
    ///
    /// `gain_db`: shelf gain in dB (negative = attenuation).
    /// `freq_hz`: shelf transition frequency.
    /// `sample_rate`: audio sample rate.
    fn high_shelf(gain_db: f32, freq_hz: f32, sample_rate: f32) -> Self {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * freq_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / 2.0 * (2.0_f32).sqrt();
        let sqrt_a = a.sqrt();

        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
        if !a0.is_finite() || a0.abs() < 1e-12 {
            return Self::passthrough();
        }
        let inv_a0 = 1.0 / a0;

        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha) * inv_a0;
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0) * inv_a0;
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha) * inv_a0;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0) * inv_a0;
        let a2 = ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha) * inv_a0;

        Self::from_coeffs(b0, b1, b2, a1, a2)
    }

    fn passthrough() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn from_coeffs(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        let sanitize = |v: f32| if v.is_finite() { v } else { 0.0 };
        Self {
            b0: sanitize(b0),
            b1: sanitize(b1),
            b2: sanitize(b2),
            a1: sanitize(a1),
            a2: sanitize(a2),
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Process one sample through the biquad (direct form II transposed).
    fn process(&mut self, x: f32) -> f32 {
        let x = if x.is_finite() { x } else { 0.0 };
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        // Clamp denormals
        if self.z1.abs() < 1e-30 {
            self.z1 = 0.0;
        }
        if self.z2.abs() < 1e-30 {
            self.z2 = 0.0;
        }
        if y.is_finite() {
            y
        } else {
            0.0
        }
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// HrtfProcessor
// ---------------------------------------------------------------------------

/// Stateful HRTF binaural processor for multi-voice chorus.
///
/// Maintains per-voice delay lines for ITD and biquad filters for head
/// shadow (ILD) and air absorption. Call [`HrtfProcessor::process_voices`]
/// each audio chunk.
pub struct HrtfProcessor {
    config: HrtfConfig,
    sample_rate: f32,
    delay_lines_l: Vec<DelayLine>,
    delay_lines_r: Vec<DelayLine>,
    head_shadow_filters_l: Vec<BiquadFilter>,
    head_shadow_filters_r: Vec<BiquadFilter>,
    distance_filters: Vec<BiquadFilter>,
    itd_samples_l: Vec<f32>,
    itd_samples_r: Vec<f32>,
    distance_gains: Vec<f32>,
}

impl HrtfProcessor {
    /// Create a new HRTF processor from the given config and sample rate.
    ///
    /// # Errors
    ///
    /// Returns [`KokoroError::InvalidConfig`] if config validation fails or
    /// sample rate is non-positive.
    pub fn new(config: &HrtfConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("sample_rate = {sample_rate}: must be finite and > 0.0"),
            });
        }

        let n = config.positions.len();
        let head_radius_m = config.head_radius_cm / 100.0;
        let c = config.speed_of_sound;

        let mut delay_lines_l = Vec::with_capacity(n);
        let mut delay_lines_r = Vec::with_capacity(n);
        let mut shadow_l = Vec::with_capacity(n);
        let mut shadow_r = Vec::with_capacity(n);
        let mut dist_filters = Vec::with_capacity(n);
        let mut itd_l = Vec::with_capacity(n);
        let mut itd_r = Vec::with_capacity(n);
        let mut dist_gains = Vec::with_capacity(n);

        for pos in &config.positions {
            let az_rad = pos
                .azimuth_deg
                .to_radians()
                .clamp(-std::f32::consts::PI, std::f32::consts::PI);

            // --- ITD via Woodworth formula ---
            // tau = (r/c)(theta + sin(theta)) for the far ear
            // Near ear gets tau = 0 (or negative of far ear offset).
            let (itd_left, itd_right) =
                compute_itd(az_rad, head_radius_m, c, sample_rate, &config.hrtf_model);
            itd_l.push(itd_left);
            itd_r.push(itd_right);

            delay_lines_l.push(DelayLine::new(MAX_DELAY_SAMPLES));
            delay_lines_r.push(DelayLine::new(MAX_DELAY_SAMPLES));

            // --- Head shadow ILD (frequency-dependent) ---
            let (shadow_filter_l, shadow_filter_r) =
                compute_head_shadow(az_rad, sample_rate, &config.hrtf_model);
            shadow_l.push(shadow_filter_l);
            shadow_r.push(shadow_filter_r);

            // --- Distance attenuation ---
            let dist = pos.distance_m.max(MIN_DISTANCE);
            let gain = (1.0 / dist).min(1.0);
            dist_gains.push(if gain.is_finite() { gain } else { 1.0 });

            // --- Air absorption (high-shelf rolloff) ---
            // Approximately -1 dB per meter above 4 kHz.
            let absorption_db = -(dist - MIN_DISTANCE).max(0.0).min(20.0);
            dist_filters.push(BiquadFilter::high_shelf(absorption_db, 4000.0, sample_rate));
        }

        Ok(Self {
            config: config.clone(),
            sample_rate,
            delay_lines_l,
            delay_lines_r,
            head_shadow_filters_l: shadow_l,
            head_shadow_filters_r: shadow_r,
            distance_filters: dist_filters,
            itd_samples_l: itd_l,
            itd_samples_r: itd_r,
            distance_gains: dist_gains,
        })
    }

    /// Process all voices and sum into a stereo pair `(left, right)`.
    ///
    /// Each voice is mono PCM. The number of voice buffers must match the
    /// number of positions in the config.
    ///
    /// # Errors
    ///
    /// Returns [`KokoroError::InvalidInput`] if the voice count does not
    /// match the configured positions.
    pub fn process_voices(
        &mut self,
        voices: &[Vec<f32>],
    ) -> Result<(Vec<f32>, Vec<f32>), KokoroError> {
        if voices.len() != self.config.positions.len() {
            return Err(KokoroError::InvalidInput(format!(
                "expected {} voices, got {}",
                self.config.positions.len(),
                voices.len(),
            )));
        }

        if !self.config.enabled || voices.is_empty() {
            let max_len = voices.iter().map(Vec::len).max().unwrap_or(0);
            return Ok((vec![0.0; max_len], vec![0.0; max_len]));
        }

        let max_len = voices.iter().map(Vec::len).max().unwrap_or(0);
        let mut out_l = vec![0.0f32; max_len];
        let mut out_r = vec![0.0f32; max_len];

        for (vi, voice) in voices.iter().enumerate() {
            let itd_left = self.itd_samples_l[vi];
            let itd_right = self.itd_samples_r[vi];
            let dist_gain = self.distance_gains[vi];

            for (si, &sample) in voice.iter().enumerate() {
                let s = if sample.is_finite() { sample } else { 0.0 };

                // Apply distance attenuation
                let attenuated = s * dist_gain;

                // Apply air absorption
                let absorbed = self.distance_filters[vi].process(attenuated);

                // ITD: delay each ear differently
                let delayed_l = self.delay_lines_l[vi].process(absorbed, itd_left);
                let delayed_r = self.delay_lines_r[vi].process(absorbed, itd_right);

                // Head shadow ILD filtering
                let filtered_l = self.head_shadow_filters_l[vi].process(delayed_l);
                let filtered_r = self.head_shadow_filters_r[vi].process(delayed_r);

                out_l[si] += filtered_l;
                out_r[si] += filtered_r;
            }
        }

        // Finite-check outputs
        for s in &mut out_l {
            if !s.is_finite() {
                *s = 0.0;
            }
        }
        for s in &mut out_r {
            if !s.is_finite() {
                *s = 0.0;
            }
        }

        Ok((out_l, out_r))
    }

    /// Reset all internal state (delay lines, filters).
    pub fn reset(&mut self) {
        for dl in &mut self.delay_lines_l {
            dl.reset();
        }
        for dl in &mut self.delay_lines_r {
            dl.reset();
        }
        for f in &mut self.head_shadow_filters_l {
            f.reset();
        }
        for f in &mut self.head_shadow_filters_r {
            f.reset();
        }
        for f in &mut self.distance_filters {
            f.reset();
        }
    }

    /// Number of configured voices.
    #[must_use]
    pub fn n_voices(&self) -> usize {
        self.config.positions.len()
    }

    /// Current sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// ITD delay in samples for the left ear of voice `i`.
    #[must_use]
    pub fn itd_left(&self, i: usize) -> f32 {
        self.itd_samples_l.get(i).copied().unwrap_or(0.0)
    }

    /// ITD delay in samples for the right ear of voice `i`.
    #[must_use]
    pub fn itd_right(&self, i: usize) -> f32 {
        self.itd_samples_r.get(i).copied().unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// ITD computation
// ---------------------------------------------------------------------------

/// Compute interaural time delay in samples for left and right ears.
///
/// Woodworth formula: `tau = (r/c)(theta + sin(theta))` for the contralateral
/// (far) ear. The ipsilateral (near) ear has zero additional delay.
fn compute_itd(
    azimuth_rad: f32,
    head_radius_m: f32,
    speed_of_sound: f32,
    sample_rate: f32,
    model: &HrtfModel,
) -> (f32, f32) {
    let abs_az = azimuth_rad.abs().min(std::f32::consts::PI);

    let tau_seconds = match model {
        HrtfModel::SphericalHead => {
            // Woodworth: tau = (r/c)(theta + sin(theta))
            (head_radius_m / speed_of_sound) * (abs_az + abs_az.sin())
        }
        HrtfModel::SimpleDelay => {
            // Simple sine-law: tau = (r/c) * sin(theta)
            (head_radius_m / speed_of_sound) * abs_az.sin()
        }
    };

    let tau_samples = (tau_seconds * sample_rate).clamp(0.0, MAX_DELAY_SAMPLES as f32);

    // Source on the right (positive azimuth): left ear is far, right ear is near.
    // Source on the left (negative azimuth): right ear is far, left ear is near.
    if azimuth_rad >= 0.0 {
        // Source right: left ear delayed more
        (tau_samples, 0.0)
    } else {
        // Source left: right ear delayed more
        (0.0, tau_samples)
    }
}

/// Compute head shadow biquad filters for left and right ears.
///
/// The contralateral ear (opposite the source) gets a low-shelf cut
/// modeling the head shadow. Maximum attenuation is ~6 dB at 4 kHz for
/// a 90-degree source.
fn compute_head_shadow(
    azimuth_rad: f32,
    sample_rate: f32,
    model: &HrtfModel,
) -> (BiquadFilter, BiquadFilter) {
    let shadow_freq = 4000.0_f32;

    // Shadow intensity scales with azimuth: 0 at front, max at sides.
    let shadow_factor = azimuth_rad.abs().sin().clamp(0.0, 1.0);

    // Maximum head shadow attenuation in dB.
    let max_shadow_db: f32 = match model {
        HrtfModel::SphericalHead => -6.0,
        HrtfModel::SimpleDelay => -3.0,
    };

    let shadow_db = max_shadow_db * shadow_factor;

    if azimuth_rad >= 0.0 {
        // Source on right: left ear is shadowed
        (
            BiquadFilter::low_shelf(shadow_db, shadow_freq, sample_rate),
            BiquadFilter::passthrough(),
        )
    } else {
        // Source on left: right ear is shadowed
        (
            BiquadFilter::passthrough(),
            BiquadFilter::low_shelf(shadow_db, shadow_freq, sample_rate),
        )
    }
}

#[cfg(test)]
#[path = "kokoro_chorus_hrtf_tests.rs"]
mod tests;
