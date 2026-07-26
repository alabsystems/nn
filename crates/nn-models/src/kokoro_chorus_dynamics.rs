// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-band dynamics compressor for Kokoro chorus bus processing.
//!
//! When mixing multiple TTS voices in a chorus, dynamics vary widely: some
//! voices are louder on certain syllables, creating uneven mix energy. A
//! multi-band compressor applies different compression ratios to different
//! frequency bands, resulting in a smooth, professional-sounding mix.
//!
//! # Architecture
//!
//! ```text
//! Input --> Linkwitz-Riley Crossover (3 bands) --> Per-band RMS compressor
//!           |-- Low  (0 - 300 Hz)                  |-- BandCompressor (low)
//!           |-- Mid  (300 Hz - 4 kHz)              |-- BandCompressor (mid)
//!           +-- High (4 kHz +)                     +-- BandCompressor (high)
//!                                                          |
//!                                                          v
//!                                                   Sum bands --> BusLimiter
//!                                                                 (-0.1 dBFS)
//! ```
//!
//! # Crossover design
//!
//! Linkwitz-Riley 4th-order (LR4) crossovers are two cascaded 2nd-order
//! Butterworth filters. At the crossover frequency the response is -6 dB,
//! and the crossover is power-complementary (energy-preserving).
//!
//! # References
//!
//! - Giannoulis, D. et al. "Digital Dynamic Range Compressor Design."
//!   Journal of the Audio Engineering Society, 60(6), 2012.
//! - Linkwitz, S. "Active Crossover Networks for Noncoincident Drivers."
//!   Journal of the Audio Engineering Society, 24(1), 1976.

#[path = "kokoro_chorus_dynamics_filters.rs"]
mod filters;
use filters::ThreeBandCrossover;

use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Per-band compressor
// ---------------------------------------------------------------------------

/// Configuration for a single compressor band.
#[derive(Debug, Clone, Copy)]
pub struct BandCompressorConfig {
    /// Threshold in dBFS. Signals above this level are compressed.
    pub threshold_db: f32,
    /// Compression ratio (e.g. 2.0 means 2:1 compression).
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Soft knee width in dB (0 = hard knee).
    pub knee_db: f32,
    /// Makeup gain in dB applied after compression.
    pub makeup_gain_db: f32,
}

impl BandCompressorConfig {
    /// Validate compressor parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.threshold_db.is_finite() || self.threshold_db > 0.0 || self.threshold_db < -96.0 {
            return Err(KokoroError::InvalidConfig {
                field: "threshold_db",
                reason: format!(
                    "threshold_db = {}: must be finite and in [-96, 0]",
                    self.threshold_db,
                ),
            });
        }
        if !self.ratio.is_finite() || self.ratio < 1.0 || self.ratio > 100.0 {
            return Err(KokoroError::InvalidConfig {
                field: "ratio",
                reason: format!("ratio = {}: must be finite and in [1, 100]", self.ratio),
            });
        }
        if !self.attack_ms.is_finite() || self.attack_ms < 0.01 || self.attack_ms > 1000.0 {
            return Err(KokoroError::InvalidConfig {
                field: "attack_ms",
                reason: format!(
                    "attack_ms = {}: must be finite and in [0.01, 1000]",
                    self.attack_ms,
                ),
            });
        }
        if !self.release_ms.is_finite() || self.release_ms < 1.0 || self.release_ms > 5000.0 {
            return Err(KokoroError::InvalidConfig {
                field: "release_ms",
                reason: format!(
                    "release_ms = {}: must be finite and in [1, 5000]",
                    self.release_ms,
                ),
            });
        }
        if !self.knee_db.is_finite() || self.knee_db < 0.0 || self.knee_db > 24.0 {
            return Err(KokoroError::InvalidConfig {
                field: "knee_db",
                reason: format!("knee_db = {}: must be finite and in [0, 24]", self.knee_db),
            });
        }
        if !self.makeup_gain_db.is_finite()
            || self.makeup_gain_db < -24.0
            || self.makeup_gain_db > 24.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "makeup_gain_db",
                reason: format!(
                    "makeup_gain_db = {}: must be finite and in [-24, 24]",
                    self.makeup_gain_db,
                ),
            });
        }
        Ok(())
    }
}

/// Per-band RMS envelope compressor with soft-knee gain curve.
///
/// Tracks the RMS level of the signal using a ballistic envelope follower
/// with separate attack and release time constants. When the envelope
/// exceeds the threshold, gain is reduced according to the compression
/// ratio. A soft knee smooths the transition around the threshold.
#[derive(Debug, Clone)]
pub struct BandCompressor {
    attack_coeff: f32,
    release_coeff: f32,
    threshold_db: f32,
    ratio: f32,
    half_knee_db: f32,
    makeup_linear: f32,
    envelope_sq: f32,
}

impl BandCompressor {
    /// Create a new per-band compressor.
    pub fn new(config: &BandCompressorConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let sr = KOKORO_SAMPLE_RATE as f64;
        let attack_coeff = (-1.0 / (f64::from(config.attack_ms) * 0.001 * sr)).exp() as f32;
        let release_coeff = (-1.0 / (f64::from(config.release_ms) * 0.001 * sr)).exp() as f32;
        let makeup_linear = 10.0f64.powf(f64::from(config.makeup_gain_db) / 20.0) as f32;

        Ok(Self {
            attack_coeff,
            release_coeff,
            threshold_db: config.threshold_db,
            ratio: config.ratio,
            half_knee_db: config.knee_db * 0.5,
            makeup_linear,
            envelope_sq: 0.0,
        })
    }

    /// Compute gain reduction in dB for a given input level in dB.
    #[inline]
    fn gain_reduction_db(&self, level_db: f32) -> f32 {
        let over_db = level_db - self.threshold_db;

        if self.half_knee_db < 0.001 {
            // Hard knee.
            if over_db <= 0.0 {
                0.0
            } else {
                over_db * (1.0 - 1.0 / self.ratio)
            }
        } else if over_db < -self.half_knee_db {
            0.0
        } else if over_db > self.half_knee_db {
            over_db * (1.0 - 1.0 / self.ratio)
        } else {
            // Inside knee: quadratic interpolation.
            let x = over_db + self.half_knee_db;
            (1.0 - 1.0 / self.ratio) * x * x / (4.0 * self.half_knee_db)
        }
    }

    /// Process a single sample, returning the compressed sample.
    #[inline]
    fn process_sample(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.envelope_sq = 0.0;
            return 0.0;
        }

        let x_sq = x * x;
        let coeff = if x_sq > self.envelope_sq {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope_sq = coeff * self.envelope_sq + (1.0 - coeff) * x_sq;

        if self.envelope_sq < 1e-20 {
            self.envelope_sq = 0.0;
        }

        let rms = self.envelope_sq.sqrt();
        if rms < 1e-10 {
            return x * self.makeup_linear;
        }

        let level_db = 20.0 * rms.log10();
        if !level_db.is_finite() {
            return x * self.makeup_linear;
        }

        let reduction_db = self.gain_reduction_db(level_db);
        let gain = 10.0f32.powf(-reduction_db / 20.0);

        let out = x * gain * self.makeup_linear;
        if !out.is_finite() {
            0.0
        } else {
            out
        }
    }

    /// Process a buffer in place.
    pub fn process(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset envelope state.
    pub fn reset(&mut self) {
        self.envelope_sq = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Multi-band compressor
// ---------------------------------------------------------------------------

/// Configuration for a 3-band multi-band dynamics compressor.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MultibandCompressorConfig {
    /// Low/mid crossover frequency (Hz). Default: 300.0.
    pub low_crossover_hz: f32,
    /// Mid/high crossover frequency (Hz). Default: 4000.0.
    pub high_crossover_hz: f32,
    /// Low band compressor settings.
    pub low: BandCompressorConfig,
    /// Mid band compressor settings.
    pub mid: BandCompressorConfig,
    /// High band compressor settings.
    pub high: BandCompressorConfig,
}

impl MultibandCompressorConfig {
    /// Validate all parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        let nyquist = KOKORO_SAMPLE_RATE as f32 / 2.0;
        if !self.low_crossover_hz.is_finite()
            || self.low_crossover_hz < 20.0
            || self.low_crossover_hz >= self.high_crossover_hz
        {
            return Err(KokoroError::InvalidConfig {
                field: "low_crossover_hz",
                reason: format!(
                    "low_crossover_hz = {}: must be finite, >= 20, < high_crossover_hz ({})",
                    self.low_crossover_hz, self.high_crossover_hz,
                ),
            });
        }
        if !self.high_crossover_hz.is_finite() || self.high_crossover_hz >= nyquist {
            return Err(KokoroError::InvalidConfig {
                field: "high_crossover_hz",
                reason: format!(
                    "high_crossover_hz = {}: must be finite and < Nyquist ({})",
                    self.high_crossover_hz, nyquist,
                ),
            });
        }
        self.low.validate()?;
        self.mid.validate()?;
        self.high.validate()?;
        Ok(())
    }
}

/// 3-band multi-band dynamics compressor.
///
/// Splits the input into low, mid, and high frequency bands using
/// Linkwitz-Riley 4th-order crossover filters, applies independent
/// compression to each band, and sums the result.
pub struct MultibandCompressor {
    crossover: ThreeBandCrossover,
    comp_low: BandCompressor,
    comp_mid: BandCompressor,
    comp_high: BandCompressor,
    buf_low: Vec<f32>,
    buf_mid: Vec<f32>,
    buf_high: Vec<f32>,
}

impl MultibandCompressor {
    /// Create a new 3-band multi-band compressor.
    pub fn new(config: &MultibandCompressorConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let sr = KOKORO_SAMPLE_RATE as f32;
        Ok(Self {
            crossover: ThreeBandCrossover::new(
                config.low_crossover_hz,
                config.high_crossover_hz,
                sr,
            ),
            comp_low: BandCompressor::new(&config.low)?,
            comp_mid: BandCompressor::new(&config.mid)?,
            comp_high: BandCompressor::new(&config.high)?,
            buf_low: Vec::new(),
            buf_mid: Vec::new(),
            buf_high: Vec::new(),
        })
    }

    /// Process an audio buffer in place.
    pub fn process(&mut self, buffer: &mut [f32]) {
        let len = buffer.len();
        self.buf_low.resize(len, 0.0);
        self.buf_mid.resize(len, 0.0);
        self.buf_high.resize(len, 0.0);

        for (i, &x) in buffer.iter().enumerate() {
            let (lo, mid, hi) = self.crossover.split(x);
            self.buf_low[i] = lo;
            self.buf_mid[i] = mid;
            self.buf_high[i] = hi;
        }

        self.comp_low.process(&mut self.buf_low[..len]);
        self.comp_mid.process(&mut self.buf_mid[..len]);
        self.comp_high.process(&mut self.buf_high[..len]);

        for i in 0..len {
            let sum = self.buf_low[i] + self.buf_mid[i] + self.buf_high[i];
            buffer[i] = if sum.is_finite() { sum } else { 0.0 };
        }
    }

    /// Reset all filter and compressor states.
    pub fn reset(&mut self) {
        self.crossover.reset();
        self.comp_low.reset();
        self.comp_mid.reset();
        self.comp_high.reset();
    }
}

// ---------------------------------------------------------------------------
// Bus limiter (brick-wall at -0.1 dBFS)
// ---------------------------------------------------------------------------

/// Default ceiling: -0.1 dBFS.
const DEFAULT_CEILING_DB: f32 = -0.1;
/// Default limiter attack: 0.1 ms.
const DEFAULT_LIMITER_ATTACK_MS: f32 = 0.1;
/// Default limiter release: 50 ms.
const DEFAULT_LIMITER_RELEASE_MS: f32 = 50.0;

/// Brick-wall bus limiter at a configurable ceiling.
///
/// Uses lookahead-free peak detection with fast attack (0.1 ms) and
/// slow release (50 ms) to prevent output exceeding the ceiling.
/// The default ceiling is -0.1 dBFS (0.9886 linear), leaving headroom
/// for DAC reconstruction filters.
#[derive(Debug, Clone)]
pub struct BusLimiter {
    ceiling_linear: f32,
    envelope: f32,
    attack_coeff: f32,
    release_coeff: f32,
}

impl BusLimiter {
    /// Create a limiter with default ceiling (-0.1 dBFS).
    pub fn new() -> Self {
        Self::with_ceiling_db(DEFAULT_CEILING_DB)
    }

    /// Create a limiter with a custom ceiling in dBFS.
    pub fn with_ceiling_db(ceiling_db: f32) -> Self {
        let ceiling_linear = 10.0f64.powf(f64::from(ceiling_db) / 20.0) as f32;
        let sr = KOKORO_SAMPLE_RATE as f64;
        let attack_coeff = (-1.0 / (f64::from(DEFAULT_LIMITER_ATTACK_MS) * 0.001 * sr)).exp() as f32;
        let release_coeff = (-1.0 / (f64::from(DEFAULT_LIMITER_RELEASE_MS) * 0.001 * sr)).exp() as f32;
        Self {
            ceiling_linear,
            envelope: 0.0,
            attack_coeff,
            release_coeff,
        }
    }

    /// Process a buffer in place, ensuring no sample exceeds the ceiling.
    pub fn process(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
                self.envelope = 0.0;
                continue;
            }

            let abs_val = sample.abs();
            let coeff = if abs_val > self.envelope {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            self.envelope = coeff * self.envelope + (1.0 - coeff) * abs_val;

            if self.envelope < 1e-20 {
                self.envelope = 0.0;
            }

            if self.envelope > self.ceiling_linear {
                let gain = self.ceiling_linear / self.envelope;
                *sample *= gain;
            }

            // Hard clamp as absolute safety net.
            *sample = sample.clamp(-self.ceiling_linear, self.ceiling_linear);
        }
    }

    /// Reset the limiter state.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
    }

    /// Get the ceiling in linear amplitude.
    #[must_use]
    pub fn ceiling_linear(&self) -> f32 {
        self.ceiling_linear
    }
}

impl Default for BusLimiter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Soft-clip limiter (tanh-based, musical saturation)
// ---------------------------------------------------------------------------

/// Default soft-clip ceiling: -0.5 dBFS (0.9441 linear).
const DEFAULT_SOFT_CLIP_CEILING_DB: f32 = -0.5;
/// Default soft-clip drive: 1.5 (moderate saturation).
const DEFAULT_SOFT_CLIP_DRIVE: f32 = 1.5;

/// Configuration for the soft-clip limiter.
#[derive(Debug, Clone, Copy)]
pub struct SoftClipConfig {
    /// Ceiling in dBFS. Signals are soft-clipped to stay below this level.
    /// Default: -0.5 dBFS.
    pub ceiling_db: f32,
    /// Drive amount (>= 1.0). Higher values push more signal into the
    /// saturation curve, producing warmer harmonics but more distortion.
    /// 1.0 = no extra drive, 2.0 = moderate, 4.0 = aggressive.
    /// Default: 1.5.
    pub drive: f32,
}

impl Default for SoftClipConfig {
    fn default() -> Self {
        Self {
            ceiling_db: DEFAULT_SOFT_CLIP_CEILING_DB,
            drive: DEFAULT_SOFT_CLIP_DRIVE,
        }
    }
}

impl SoftClipConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.ceiling_db.is_finite() || self.ceiling_db > 0.0 || self.ceiling_db < -48.0 {
            return Err(KokoroError::InvalidConfig {
                field: "ceiling_db",
                reason: format!(
                    "ceiling_db = {}: must be finite and in [-48, 0]",
                    self.ceiling_db,
                ),
            });
        }
        if !self.drive.is_finite() || self.drive < 1.0 || self.drive > 20.0 {
            return Err(KokoroError::InvalidConfig {
                field: "drive",
                reason: format!("drive = {}: must be finite and in [1, 20]", self.drive),
            });
        }
        Ok(())
    }
}

/// Soft-clip limiter using `tanh` saturation for musical limiting.
///
/// Unlike [`BusLimiter`] which uses a hard clamp as its final safety net,
/// `SoftClipLimiter` applies a `tanh` waveshaping curve that smoothly
/// saturates the signal as it approaches the ceiling. This produces
/// even-order harmonics (warm distortion) rather than the abrupt
/// clipping artifacts of a brick-wall limiter.
///
/// The processing chain is:
/// 1. Scale input by `drive / ceiling` to normalize into the tanh curve.
/// 2. Apply `tanh()` waveshaping.
/// 3. Scale output back by `ceiling` to restore the target amplitude range.
#[derive(Debug, Clone)]
pub struct SoftClipLimiter {
    ceiling_linear: f32,
    drive: f32,
}

impl SoftClipLimiter {
    /// Create a soft-clip limiter with default settings (-0.5 dBFS, drive 1.5).
    pub fn new() -> Self {
        Self::from_config(&SoftClipConfig::default())
    }

    /// Create a soft-clip limiter from a configuration.
    #[must_use]
    pub fn from_config(config: &SoftClipConfig) -> Self {
        let ceiling_linear = 10.0f64.powf(f64::from(config.ceiling_db) / 20.0) as f32;
        Self {
            ceiling_linear,
            drive: config.drive,
        }
    }

    /// Create a soft-clip limiter with a custom ceiling and drive.
    pub fn with_params(ceiling_db: f32, drive: f32) -> Result<Self, KokoroError> {
        let config = SoftClipConfig { ceiling_db, drive };
        config.validate()?;
        Ok(Self::from_config(&config))
    }

    /// Process a buffer in place, applying tanh soft-clipping.
    pub fn process(&self, buffer: &mut [f32]) {
        let inv_ceiling = self.drive / self.ceiling_linear;
        for sample in buffer.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
                continue;
            }
            // Scale into tanh domain, apply saturation, scale back.
            let normalized = *sample * inv_ceiling;
            let clipped = normalized.tanh();
            *sample = clipped * self.ceiling_linear;
        }
    }

    /// Get the ceiling in linear amplitude.
    #[must_use]
    pub fn ceiling_linear(&self) -> f32 {
        self.ceiling_linear
    }

    /// Get the drive amount.
    #[must_use]
    pub fn drive(&self) -> f32 {
        self.drive
    }
}

impl Default for SoftClipLimiter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Dynamics presets
// ---------------------------------------------------------------------------

/// Named dynamics processing presets for TTS chorus bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DynamicsPreset {
    /// Gentle: light compression for natural dynamics.
    Gentle,
    /// Broadcast: medium compression for broadcast-standard loudness.
    Broadcast,
    /// Aggressive: heavy compression for dense mixes (8+ voices).
    Aggressive,
    /// Mastering: transparent limiting with gentle multiband control.
    Mastering,
}

impl DynamicsPreset {
    /// Convert this preset to a [`MultibandCompressorConfig`].
    #[must_use]
    pub fn to_config(self) -> MultibandCompressorConfig {
        match self {
            Self::Gentle => MultibandCompressorConfig {
                low_crossover_hz: 300.0,
                high_crossover_hz: 4000.0,
                low: BandCompressorConfig {
                    threshold_db: -18.0,
                    ratio: 1.5,
                    attack_ms: 20.0,
                    release_ms: 200.0,
                    knee_db: 6.0,
                    makeup_gain_db: 1.0,
                },
                mid: BandCompressorConfig {
                    threshold_db: -20.0,
                    ratio: 1.5,
                    attack_ms: 10.0,
                    release_ms: 150.0,
                    knee_db: 6.0,
                    makeup_gain_db: 1.0,
                },
                high: BandCompressorConfig {
                    threshold_db: -22.0,
                    ratio: 1.5,
                    attack_ms: 5.0,
                    release_ms: 100.0,
                    knee_db: 6.0,
                    makeup_gain_db: 0.5,
                },
            },
            Self::Broadcast => MultibandCompressorConfig {
                low_crossover_hz: 300.0,
                high_crossover_hz: 4000.0,
                low: BandCompressorConfig {
                    threshold_db: -24.0,
                    ratio: 3.0,
                    attack_ms: 15.0,
                    release_ms: 150.0,
                    knee_db: 4.0,
                    makeup_gain_db: 2.0,
                },
                mid: BandCompressorConfig {
                    threshold_db: -22.0,
                    ratio: 2.5,
                    attack_ms: 8.0,
                    release_ms: 120.0,
                    knee_db: 4.0,
                    makeup_gain_db: 2.0,
                },
                high: BandCompressorConfig {
                    threshold_db: -26.0,
                    ratio: 3.0,
                    attack_ms: 3.0,
                    release_ms: 80.0,
                    knee_db: 3.0,
                    makeup_gain_db: 1.5,
                },
            },
            Self::Aggressive => MultibandCompressorConfig {
                low_crossover_hz: 300.0,
                high_crossover_hz: 4000.0,
                low: BandCompressorConfig {
                    threshold_db: -30.0,
                    ratio: 6.0,
                    attack_ms: 5.0,
                    release_ms: 80.0,
                    knee_db: 3.0,
                    makeup_gain_db: 4.0,
                },
                mid: BandCompressorConfig {
                    threshold_db: -28.0,
                    ratio: 5.0,
                    attack_ms: 3.0,
                    release_ms: 60.0,
                    knee_db: 3.0,
                    makeup_gain_db: 3.5,
                },
                high: BandCompressorConfig {
                    threshold_db: -32.0,
                    ratio: 6.0,
                    attack_ms: 1.0,
                    release_ms: 50.0,
                    knee_db: 2.0,
                    makeup_gain_db: 3.0,
                },
            },
            Self::Mastering => MultibandCompressorConfig {
                low_crossover_hz: 300.0,
                high_crossover_hz: 4000.0,
                low: BandCompressorConfig {
                    threshold_db: -12.0,
                    ratio: 1.5,
                    attack_ms: 30.0,
                    release_ms: 300.0,
                    knee_db: 8.0,
                    makeup_gain_db: 0.5,
                },
                mid: BandCompressorConfig {
                    threshold_db: -14.0,
                    ratio: 1.3,
                    attack_ms: 15.0,
                    release_ms: 200.0,
                    knee_db: 8.0,
                    makeup_gain_db: 0.5,
                },
                high: BandCompressorConfig {
                    threshold_db: -16.0,
                    ratio: 1.5,
                    attack_ms: 8.0,
                    release_ms: 150.0,
                    knee_db: 8.0,
                    makeup_gain_db: 0.0,
                },
            },
        }
    }
}

/// Split a mono buffer into three frequency bands using an LR4 crossover.
///
/// Returns `(low, mid, high)` buffers. Convenience function for testing.
pub fn split_bands(
    input: &[f32],
    low_crossover_hz: f32,
    high_crossover_hz: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let sr = KOKORO_SAMPLE_RATE as f32;
    let mut xover = ThreeBandCrossover::new(low_crossover_hz, high_crossover_hz, sr);
    let mut lo = Vec::with_capacity(input.len());
    let mut mid = Vec::with_capacity(input.len());
    let mut hi = Vec::with_capacity(input.len());
    for &x in input {
        let (l, m, h) = xover.split(x);
        lo.push(l);
        mid.push(m);
        hi.push(h);
    }
    (lo, mid, hi)
}

#[cfg(test)]
#[path = "kokoro_chorus_dynamics_tests.rs"]
mod tests;
