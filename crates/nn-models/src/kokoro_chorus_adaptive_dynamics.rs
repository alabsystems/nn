// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Adaptive dynamics processor with psychoacoustic masking for Kokoro chorus.
//!
//! Unlike simple level-based compression, this module considers which frequency
//! bands are perceptually masked by others and only compresses content that
//! exceeds its masking threshold -- i.e., content that is actually audible.
//! The result is more transparent dynamics control that preserves micro-detail
//! in masked regions while still taming audible transients.
//!
//! # Algorithm
//!
//! 1. Split input into 8 pseudo-Bark bands (simplified from 24 critical bands).
//! 2. Compute instantaneous RMS power per band via envelope followers.
//! 3. Compute simultaneous masking thresholds using a spreading function
//!    that models how energy in one band raises the hearing threshold in
//!    adjacent bands.
//! 4. For each band: apply gain reduction only when band power exceeds its
//!    masking threshold by more than `threshold_db`.
//! 5. Recombine bands with per-band gain reductions.
//! 6. Apply lookahead delay line for transparent transient limiting.
//!
//! # References
//!
//! - Zwicker & Fastl, "Psychoacoustics," 3rd ed., Springer, 2007.
//! - ISO 226:2003, "Normal equal-loudness-level contours."
//! - Painter & Spanias, "Perceptual Coding of Digital Audio," Proc. IEEE, 2000.

use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// 8-band simplified Bark scale edges (Hz). Covers 0-15.5 kHz.
/// Derived from Zwicker critical bands, merged to reduce processing cost.
const BAND_EDGES: [f32; 9] = [
    0.0, 200.0, 510.0, 920.0, 1480.0, 2700.0, 4400.0, 7700.0, 15500.0,
];

/// Number of analysis bands.
const NUM_BANDS: usize = 8;

/// Spreading function slopes (dB/Bark): lower slope ~27 dB/Bark, upper ~-10 dB/Bark.
/// These model how a masker in one band raises hearing threshold in neighbors.
const SPREAD_LOWER_DB_PER_BARK: f32 = 27.0;
const SPREAD_UPPER_DB_PER_BARK: f32 = -10.0;

/// Absolute hearing threshold per band (dB SPL, approximate from ISO 226).
/// Indexed by band [0..8].
const ABSOLUTE_THRESHOLD_DB: [f32; NUM_BANDS] = [40.0, 20.0, 10.0, 5.0, 3.0, 5.0, 10.0, 30.0];

/// Masking offset: a tone must exceed the masked threshold by this many dB
/// to be audible (noise-masking-tone offset is ~5-6 dB, tone-masking-noise ~26 dB;
/// for broadband signals use ~10 dB).
const MASKING_OFFSET_DB: f32 = 10.0;

// ---------------------------------------------------------------------------
// Masking model selection
// ---------------------------------------------------------------------------

/// Psychoacoustic masking model variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum MaskingModel {
    /// Frequency-distance based masking with simple spreading function.
    /// Faster, suitable for real-time use.
    #[default]
    SimpleMasking,
    /// Full Bark-scale critical band masking with inter-band spreading.
    /// More accurate but slightly more expensive.
    BarkScale,
}


// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the adaptive dynamics processor.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AdaptiveDynamicsConfig {
    /// Compression threshold in dB (relative to masking threshold). Default: -12.0.
    pub threshold_db: f32,
    /// Compression ratio (e.g., 3.0 = 3:1). Default: 3.0.
    pub ratio: f32,
    /// Attack time in milliseconds. Default: 5.0.
    pub attack_ms: f32,
    /// Release time in milliseconds. Default: 50.0.
    pub release_ms: f32,
    /// Soft knee width in dB. Default: 6.0.
    pub knee_db: f32,
    /// Masking model to use. Default: SimpleMasking.
    pub masking_model: MaskingModel,
    /// Lookahead in milliseconds for transparent limiting. Default: 5.0.
    pub lookahead_ms: f32,
    /// Makeup gain in dB. 0.0 = no makeup. Default: 0.0.
    pub makeup_gain_db: f32,
}

impl Default for AdaptiveDynamicsConfig {
    fn default() -> Self {
        Self {
            threshold_db: -12.0,
            ratio: 3.0,
            attack_ms: 5.0,
            release_ms: 50.0,
            knee_db: 6.0,
            masking_model: MaskingModel::default(),
            lookahead_ms: 5.0,
            makeup_gain_db: 0.0,
        }
    }
}

impl AdaptiveDynamicsConfig {
    /// Create a new config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // Builder methods -------------------------------------------------------

    #[must_use]
    pub fn with_threshold_db(mut self, v: f32) -> Self {
        self.threshold_db = v;
        self
    }

    #[must_use]
    pub fn with_ratio(mut self, v: f32) -> Self {
        self.ratio = v;
        self
    }

    #[must_use]
    pub fn with_attack_ms(mut self, v: f32) -> Self {
        self.attack_ms = v;
        self
    }

    #[must_use]
    pub fn with_release_ms(mut self, v: f32) -> Self {
        self.release_ms = v;
        self
    }

    #[must_use]
    pub fn with_knee_db(mut self, v: f32) -> Self {
        self.knee_db = v;
        self
    }

    #[must_use]
    pub fn with_masking_model(mut self, m: MaskingModel) -> Self {
        self.masking_model = m;
        self
    }

    #[must_use]
    pub fn with_lookahead_ms(mut self, v: f32) -> Self {
        self.lookahead_ms = v;
        self
    }

    #[must_use]
    pub fn with_makeup_gain_db(mut self, v: f32) -> Self {
        self.makeup_gain_db = v;
        self
    }

    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        let check = |field: &'static str, v: f32, lo: f32, hi: f32| -> Result<(), KokoroError> {
            if !v.is_finite() || v < lo || v > hi {
                return Err(KokoroError::InvalidConfig {
                    field,
                    reason: format!("must be finite in [{lo}, {hi}], got {v}"),
                });
            }
            Ok(())
        };
        check("threshold_db", self.threshold_db, -96.0, 0.0)?;
        check("ratio", self.ratio, 1.0, 100.0)?;
        check("attack_ms", self.attack_ms, 0.01, 1000.0)?;
        check("release_ms", self.release_ms, 1.0, 5000.0)?;
        check("knee_db", self.knee_db, 0.0, 24.0)?;
        check("lookahead_ms", self.lookahead_ms, 0.0, 50.0)?;
        check("makeup_gain_db", self.makeup_gain_db, -24.0, 24.0)?;
        Ok(())
    }

    // -- Presets -------------------------------------------------------------

    /// Transparent preset: gentle compression, wide knee, only compresses
    /// clearly audible content. Ideal for solo voice or duet.
    #[must_use]
    pub fn transparent() -> Self {
        Self {
            threshold_db: -8.0,
            ratio: 2.0,
            attack_ms: 10.0,
            release_ms: 80.0,
            knee_db: 10.0,
            masking_model: MaskingModel::BarkScale,
            lookahead_ms: 5.0,
            makeup_gain_db: 0.0,
        }
    }

    /// Musical preset: moderate compression that preserves dynamics while
    /// smoothing the mix. Good for 3-6 voice chorus.
    #[must_use]
    pub fn musical() -> Self {
        Self {
            threshold_db: -14.0,
            ratio: 3.0,
            attack_ms: 5.0,
            release_ms: 60.0,
            knee_db: 6.0,
            masking_model: MaskingModel::BarkScale,
            lookahead_ms: 5.0,
            makeup_gain_db: 1.0,
        }
    }

    /// Broadcast preset: firmer compression for consistent loudness.
    /// Suitable for podcast/broadcast with chorus segments.
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            threshold_db: -18.0,
            ratio: 4.0,
            attack_ms: 3.0,
            release_ms: 40.0,
            knee_db: 4.0,
            masking_model: MaskingModel::SimpleMasking,
            lookahead_ms: 3.0,
            makeup_gain_db: 2.0,
        }
    }

    /// Aggressive preset: heavy compression for dense mixes (8+ voices).
    /// Prioritizes level consistency over natural dynamics.
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            threshold_db: -24.0,
            ratio: 8.0,
            attack_ms: 1.0,
            release_ms: 30.0,
            knee_db: 3.0,
            masking_model: MaskingModel::SimpleMasking,
            lookahead_ms: 2.0,
            makeup_gain_db: 4.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-band envelope follower
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BandEnvelope {
    attack_coeff: f32,
    release_coeff: f32,
    envelope_sq: f32,
}

impl BandEnvelope {
    fn new(attack_ms: f32, release_ms: f32) -> Self {
        let sr = KOKORO_SAMPLE_RATE as f64;
        Self {
            attack_coeff: (-1.0 / (f64::from(attack_ms) * 0.001 * sr)).exp() as f32,
            release_coeff: (-1.0 / (f64::from(release_ms) * 0.001 * sr)).exp() as f32,
            envelope_sq: 0.0,
        }
    }

    #[inline]
    fn update(&mut self, x: f32) -> f32 {
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
        self.envelope_sq
    }

    fn reset(&mut self) {
        self.envelope_sq = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Simple 2nd-order bandpass (biquad) for band splitting
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// Create a bandpass filter for the given center frequency and bandwidth.
    fn bandpass(center_hz: f32, bw_hz: f32, sr: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * center_hz / sr;
        let q = center_hz / bw_hz.max(1.0);
        let alpha = w0.sin() / (2.0 * q);
        let a0_inv = 1.0 / (1.0 + alpha);

        Self {
            b0: alpha * a0_inv,
            b1: 0.0,
            b2: -alpha * a0_inv,
            a1: -2.0 * w0.cos() * a0_inv,
            a2: (1.0 - alpha) * a0_inv,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Lowpass for the lowest band.
    fn lowpass(cutoff_hz: f32, sr: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * cutoff_hz / sr;
        let alpha = w0.sin() / (2.0 * std::f32::consts::FRAC_1_SQRT_2.recip());
        let cos_w0 = w0.cos();
        let a0_inv = 1.0 / (1.0 + alpha);

        Self {
            b0: (1.0 - cos_w0) * 0.5 * a0_inv,
            b1: (1.0 - cos_w0) * a0_inv,
            b2: (1.0 - cos_w0) * 0.5 * a0_inv,
            a1: -2.0 * cos_w0 * a0_inv,
            a2: (1.0 - alpha) * a0_inv,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Highpass for the highest band.
    fn highpass(cutoff_hz: f32, sr: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * cutoff_hz / sr;
        let alpha = w0.sin() / (2.0 * std::f32::consts::FRAC_1_SQRT_2.recip());
        let cos_w0 = w0.cos();
        let a0_inv = 1.0 / (1.0 + alpha);

        Self {
            b0: (1.0 + cos_w0) * 0.5 * a0_inv,
            b1: -(1.0 + cos_w0) * a0_inv,
            b2: (1.0 + cos_w0) * 0.5 * a0_inv,
            a1: -2.0 * cos_w0 * a0_inv,
            a2: (1.0 - alpha) * a0_inv,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.z1 = 0.0;
            self.z2 = 0.0;
            return 0.0;
        }
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        if !y.is_finite() {
            self.z1 = 0.0;
            self.z2 = 0.0;
            return 0.0;
        }
        y
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Delay line for lookahead
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct DelayLine {
    buffer: Vec<f32>,
    write_pos: usize,
}

impl DelayLine {
    fn new(delay_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let out = self.buffer[self.write_pos];
        self.buffer[self.write_pos] = if x.is_finite() { x } else { 0.0 };
        self.write_pos += 1;
        if self.write_pos >= self.buffer.len() {
            self.write_pos = 0;
        }
        out
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }
}

// ---------------------------------------------------------------------------
// Masking threshold computation
// ---------------------------------------------------------------------------

/// Compute per-band masking thresholds from per-band power levels.
/// Returns the effective hearing threshold per band in dB.
fn compute_masking_thresholds(
    band_power_db: &[f32; NUM_BANDS],
    model: MaskingModel,
) -> [f32; NUM_BANDS] {
    let mut thresholds = [0.0f32; NUM_BANDS];

    match model {
        MaskingModel::SimpleMasking => {
            // Simple model: each band's threshold is the max of absolute
            // threshold and the spreading from adjacent bands.
            for i in 0..NUM_BANDS {
                let mut max_mask = ABSOLUTE_THRESHOLD_DB[i];
                for j in 0..NUM_BANDS {
                    if i == j {
                        continue;
                    }
                    let distance = (i as f32 - j as f32).abs();
                    let spread = if j < i {
                        // Lower frequency masking higher: steeper slope
                        band_power_db[j] + SPREAD_UPPER_DB_PER_BARK * distance
                    } else {
                        // Higher frequency masking lower: shallower slope
                        band_power_db[j] - SPREAD_LOWER_DB_PER_BARK * distance
                    };
                    let masked = spread - MASKING_OFFSET_DB;
                    if masked > max_mask {
                        max_mask = masked;
                    }
                }
                thresholds[i] = max_mask;
            }
        }
        MaskingModel::BarkScale => {
            // Full Bark-scale model: uses the spreading function from
            // Schroeder et al. with asymmetric slopes.
            for i in 0..NUM_BANDS {
                let mut mask_power_sum = 0.0f64;

                for j in 0..NUM_BANDS {
                    let bark_diff = i as f32 - j as f32;
                    // Spreading function: asymmetric around masker
                    let spread_db = if bark_diff >= 0.0 {
                        // Masker below target: upper spread (gentler)
                        SPREAD_UPPER_DB_PER_BARK * bark_diff
                    } else {
                        // Masker above target: lower spread (steeper)
                        SPREAD_LOWER_DB_PER_BARK * bark_diff.abs()
                    };

                    let masked_level = band_power_db[j] + spread_db - MASKING_OFFSET_DB;
                    // Convert to linear power and sum (power addition)
                    if masked_level > -120.0 {
                        mask_power_sum += 10.0f64.powf(f64::from(masked_level) / 10.0);
                    }
                }

                // Convert summed power back to dB
                let mask_db = if mask_power_sum > 1e-20 {
                    (10.0 * mask_power_sum.log10()) as f32
                } else {
                    -120.0
                };

                // Take max of computed mask and absolute threshold
                thresholds[i] = mask_db.max(ABSOLUTE_THRESHOLD_DB[i]);
            }
        }
    }

    thresholds
}

// ---------------------------------------------------------------------------
// Adaptive dynamics processor
// ---------------------------------------------------------------------------

/// Adaptive dynamics processor using psychoacoustic masking.
///
/// Splits audio into frequency bands, computes per-band masking thresholds,
/// and only applies compression to bands where content exceeds its masking
/// threshold. This preserves perceptually masked detail while controlling
/// audible transients, resulting in more transparent dynamics processing.
pub struct AdaptiveDynamicsProcessor {
    config: AdaptiveDynamicsConfig,
    band_filters: Vec<Biquad>,
    envelopes: Vec<BandEnvelope>,
    delay: DelayLine,
    gain_reduction_db: f32,
    half_knee_db: f32,
    makeup_linear: f32,
    band_buffers: Vec<Vec<f32>>,
}

impl AdaptiveDynamicsProcessor {
    /// Create a new processor from the given configuration.
    pub fn new(config: &AdaptiveDynamicsConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let sr = KOKORO_SAMPLE_RATE as f32;
        let lookahead_samples = (config.lookahead_ms * 0.001 * sr).round() as usize;
        let makeup_linear = 10.0f64.powf(f64::from(config.makeup_gain_db) / 20.0) as f32;

        // Build band filters: lowpass, bandpass x6, highpass
        let mut filters = Vec::with_capacity(NUM_BANDS);
        filters.push(Biquad::lowpass(BAND_EDGES[1], sr));
        for i in 1..(NUM_BANDS - 1) {
            let lo = BAND_EDGES[i];
            let hi = BAND_EDGES[i + 1];
            let center = (lo + hi) * 0.5;
            let bw = hi - lo;
            filters.push(Biquad::bandpass(center, bw, sr));
        }
        filters.push(Biquad::highpass(BAND_EDGES[NUM_BANDS - 1], sr));

        let envelopes: Vec<_> = (0..NUM_BANDS)
            .map(|_| BandEnvelope::new(config.attack_ms, config.release_ms))
            .collect();

        Ok(Self {
            config: config.clone(),
            band_filters: filters,
            envelopes,
            delay: DelayLine::new(lookahead_samples),
            gain_reduction_db: 0.0,
            half_knee_db: config.knee_db * 0.5,
            makeup_linear,
            band_buffers: vec![Vec::new(); NUM_BANDS],
        })
    }

    /// Create a processor with default configuration.
    pub fn with_defaults() -> Result<Self, KokoroError> {
        Self::new(&AdaptiveDynamicsConfig::default())
    }

    /// Soft-knee gain reduction curve. Returns positive dB of gain reduction.
    #[inline]
    fn gain_reduction_for_over(&self, over_db: f32) -> f32 {
        let ratio_factor = 1.0 - 1.0 / self.config.ratio;
        if self.half_knee_db < 0.001 {
            if over_db <= 0.0 {
                0.0
            } else {
                over_db * ratio_factor
            }
        } else if over_db < -self.half_knee_db {
            0.0
        } else if over_db > self.half_knee_db {
            over_db * ratio_factor
        } else {
            let x = over_db + self.half_knee_db;
            ratio_factor * x * x / (4.0 * self.half_knee_db)
        }
    }

    /// Process audio buffer in place.
    pub fn process(&mut self, audio: &mut [f32]) {
        if audio.is_empty() {
            return;
        }
        let len = audio.len();

        // Resize band buffers
        for buf in &mut self.band_buffers {
            buf.resize(len, 0.0);
        }

        // Split into bands
        for (band_idx, (filter, buf)) in self
            .band_filters
            .iter_mut()
            .zip(self.band_buffers.iter_mut())
            .enumerate()
        {
            for (i, &x) in audio.iter().enumerate() {
                buf[i] = filter.process(x);
            }
            let _ = band_idx; // used implicitly via zip
        }

        // Process sample-by-sample with masking-aware compression
        let mut band_power_db = [0.0f32; NUM_BANDS];

        for i in 0..len {
            // Compute per-band envelope levels
            for (band, env) in self.envelopes.iter_mut().enumerate() {
                let sample = self.band_buffers[band][i];
                let env_sq = env.update(sample);
                let rms = env_sq.sqrt();
                band_power_db[band] = if rms > 1e-10 {
                    let db = 20.0 * rms.log10();
                    if db.is_finite() {
                        db
                    } else {
                        -120.0
                    }
                } else {
                    -120.0
                };
            }

            // Compute masking thresholds
            let thresholds = compute_masking_thresholds(&band_power_db, self.config.masking_model);

            // Compute per-band gain reduction, take max across bands
            let mut max_reduction_db = 0.0f32;
            for band in 0..NUM_BANDS {
                // Only compress if band level exceeds its masking threshold
                let excess_db = band_power_db[band] - thresholds[band];
                if excess_db <= 0.0 {
                    continue; // Band is masked -- no compression needed
                }
                // Apply compression curve relative to threshold
                let over_db = excess_db + self.config.threshold_db;
                let reduction = self.gain_reduction_for_over(over_db);
                if reduction > max_reduction_db {
                    max_reduction_db = reduction;
                }
            }

            // Track the reported gain reduction (smoothed)
            self.gain_reduction_db = 0.95 * self.gain_reduction_db + 0.05 * max_reduction_db;

            // Apply gain to the delayed (lookahead) signal
            let delayed = self.delay.process(audio[i]);
            let gain_linear = if max_reduction_db > 0.001 {
                10.0f32.powf(-max_reduction_db / 20.0)
            } else {
                1.0
            };

            let out = delayed * gain_linear * self.makeup_linear;
            audio[i] = if out.is_finite() { out } else { 0.0 };
        }
    }

    /// Get the current smoothed gain reduction in dB.
    #[must_use]
    pub fn get_gain_reduction_db(&self) -> f32 {
        self.gain_reduction_db
    }

    /// Get a reference to the active configuration.
    #[must_use]
    pub fn config(&self) -> &AdaptiveDynamicsConfig {
        &self.config
    }

    /// Reset all internal state (filters, envelopes, delay line, metering).
    pub fn reset(&mut self) {
        for f in &mut self.band_filters {
            f.reset();
        }
        for e in &mut self.envelopes {
            e.reset();
        }
        self.delay.reset();
        self.gain_reduction_db = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validates() {
        AdaptiveDynamicsConfig::default().validate().unwrap();
    }

    #[test]
    fn test_preset_configs_validate() {
        AdaptiveDynamicsConfig::transparent().validate().unwrap();
        AdaptiveDynamicsConfig::musical().validate().unwrap();
        AdaptiveDynamicsConfig::broadcast().validate().unwrap();
        AdaptiveDynamicsConfig::aggressive().validate().unwrap();
    }

    #[test]
    fn test_invalid_threshold_rejected() {
        let cfg = AdaptiveDynamicsConfig::new().with_threshold_db(1.0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_ratio_rejected() {
        let cfg = AdaptiveDynamicsConfig::new().with_ratio(0.5);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_builder_chain() {
        let cfg = AdaptiveDynamicsConfig::new()
            .with_threshold_db(-18.0)
            .with_ratio(4.0)
            .with_attack_ms(3.0)
            .with_release_ms(40.0)
            .with_knee_db(4.0)
            .with_masking_model(MaskingModel::BarkScale)
            .with_lookahead_ms(3.0)
            .with_makeup_gain_db(2.0);
        cfg.validate().unwrap();
        assert_eq!(cfg.ratio, 4.0);
        assert_eq!(cfg.masking_model, MaskingModel::BarkScale);
    }

    #[test]
    fn test_processor_creation() {
        let proc = AdaptiveDynamicsProcessor::with_defaults();
        assert!(proc.is_ok());
    }

    #[test]
    fn test_silence_passthrough() {
        let mut proc = AdaptiveDynamicsProcessor::with_defaults().unwrap();
        let mut audio = vec![0.0f32; 1000];
        proc.process(&mut audio);
        // Silence should remain silence (within float precision)
        assert!(audio.iter().all(|&s| s.abs() < 1e-6));
    }

    #[test]
    fn test_empty_buffer_no_crash() {
        let mut proc = AdaptiveDynamicsProcessor::with_defaults().unwrap();
        let mut audio = Vec::new();
        proc.process(&mut audio);
        assert!(audio.is_empty());
    }

    #[test]
    fn test_nan_sanitization() {
        let mut proc = AdaptiveDynamicsProcessor::with_defaults().unwrap();
        let mut audio = vec![0.5, f32::NAN, 0.3, f32::INFINITY, -0.2];
        proc.process(&mut audio);
        assert!(audio.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_gain_reduction_reports_nonzero_for_loud_signal() {
        let mut proc =
            AdaptiveDynamicsProcessor::new(&AdaptiveDynamicsConfig::aggressive()).unwrap();
        // Feed a loud sine-like signal
        let mut audio: Vec<f32> = (0..4800)
            .map(|i| 0.9 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        proc.process(&mut audio);
        // After processing a loud signal, gain reduction should be non-trivial
        assert!(proc.get_gain_reduction_db() > 0.0);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut proc = AdaptiveDynamicsProcessor::with_defaults().unwrap();
        let mut audio: Vec<f32> = (0..2400)
            .map(|i| 0.8 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 24000.0).sin())
            .collect();
        proc.process(&mut audio);
        proc.reset();
        assert!(proc.get_gain_reduction_db().abs() < 1e-6);
    }

    #[test]
    fn test_masking_threshold_simple() {
        let power = [
            -120.0, -120.0, -120.0, -10.0, -120.0, -120.0, -120.0, -120.0,
        ];
        let thresh = compute_masking_thresholds(&power, MaskingModel::SimpleMasking);
        // Band 3 has energy; neighboring bands should have elevated thresholds
        assert!(thresh[2] > ABSOLUTE_THRESHOLD_DB[2]);
        assert!(thresh[4] > ABSOLUTE_THRESHOLD_DB[4]);
    }

    #[test]
    fn test_masking_threshold_bark() {
        let power = [
            -120.0, -120.0, -120.0, -10.0, -120.0, -120.0, -120.0, -120.0,
        ];
        let thresh = compute_masking_thresholds(&power, MaskingModel::BarkScale);
        // Band 3 has energy; nearby bands should have elevated thresholds
        assert!(thresh[2] > ABSOLUTE_THRESHOLD_DB[2]);
    }

    #[test]
    fn test_all_presets_produce_valid_processors() {
        for cfg in [
            AdaptiveDynamicsConfig::transparent(),
            AdaptiveDynamicsConfig::musical(),
            AdaptiveDynamicsConfig::broadcast(),
            AdaptiveDynamicsConfig::aggressive(),
        ] {
            let proc = AdaptiveDynamicsProcessor::new(&cfg);
            assert!(proc.is_ok(), "Failed for preset: {cfg:?}");
        }
    }
}
