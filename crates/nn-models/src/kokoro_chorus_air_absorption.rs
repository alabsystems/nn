// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Frequency-dependent air absorption modeling for Kokoro chorus voices.
//!
//! Models how sound propagates through air over distance: high frequencies
//! are absorbed more than low frequencies, so distant voices sound darker
//! and less present. This is one of the primary depth cues the human ear
//! uses to judge distance.
//!
//! # Physical basis (simplified ISO 9613-1)
//!
//! Atmospheric absorption coefficient alpha(f) increases roughly as f^1.7
//! at typical room conditions (20 C, 50% RH). At 1 kHz the attenuation
//! is negligible over a few meters, but at 10 kHz it is clearly audible.
//! This module uses cascaded one-pole lowpass filters whose cutoff
//! decreases with distance to approximate the effect.
//!
//! # Per-voice processing
//!
//! ```text
//! Input --> Distance-dependent LPF (cascaded) --> Presence shelf (optional)
//!       --> Wet/dry mix --> Output
//! ```
//!
//! Voice 0 is closest (least absorption), voice N-1 is farthest (most
//! absorption). The `per_voice_spread` parameter controls how much the
//! distances differ across voices.
//!
//! # References
//!
//! - ISO 9613-1:1993, "Acoustics — Attenuation of sound during
//!   propagation outdoors — Part 1: Calculation of the absorption of
//!   sound by the atmosphere."
//! - Bass, H.E. et al., "Atmospheric absorption of sound: Further
//!   developments," JASA 97(1), 1995.
//!
//! Part of #3351.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// AirAbsorptionConfig
// ---------------------------------------------------------------------------

/// Configuration for frequency-dependent air absorption modeling.
///
/// Constructed via [`AirAbsorptionConfig::new`] (required for cross-crate
/// use due to `#[non_exhaustive]`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct AirAbsorptionConfig {
    /// Ambient temperature in degrees Celsius. Affects absorption rate:
    /// colder air absorbs HF more aggressively. Default: 20.0.
    pub temperature_c: f32,
    /// Relative humidity as a percentage (0-100). Higher humidity slightly
    /// reduces HF absorption at moderate frequencies but increases it at
    /// very high frequencies. Default: 50.0.
    pub humidity_percent: f32,
    /// Maximum virtual distance from source to listener in meters.
    /// Controls the farthest voice's absorption depth. Default: 3.0.
    pub max_distance_m: f32,
    /// How much the distances vary across voices: 0.0 = all same distance,
    /// 1.0 = full spread from 0 to `max_distance_m`. Default: 0.5.
    pub per_voice_spread: f32,
    /// Wet/dry mix: 0.0 = bypass (dry), 1.0 = full absorption effect.
    /// Default: 0.5.
    pub mix: f32,
    /// Number of cascaded one-pole lowpass filter stages per voice.
    /// More stages produce a steeper rolloff curve. Default: 2.
    pub filter_stages: usize,
    /// Enable a subtle high-shelf presence compensation for moderate
    /// distances. This prevents voices from sounding too dull when the
    /// absorption is mild. Default: true.
    pub presence_compensation: bool,
    /// Sample rate in Hz. Default: 24000.0.
    pub sample_rate: f32,
}

impl Default for AirAbsorptionConfig {
    fn default() -> Self {
        Self {
            temperature_c: 20.0,
            humidity_percent: 50.0,
            max_distance_m: 3.0,
            per_voice_spread: 0.5,
            mix: 0.5,
            filter_stages: 2,
            presence_compensation: true,
            sample_rate: 24000.0,
        }
    }
}

impl AirAbsorptionConfig {
    /// Create a new air absorption config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the ambient temperature in Celsius.
    #[must_use]
    pub fn with_temperature_c(mut self, temp: f32) -> Self {
        self.temperature_c = temp;
        self
    }

    /// Set the relative humidity (0-100%).
    #[must_use]
    pub fn with_humidity_percent(mut self, rh: f32) -> Self {
        self.humidity_percent = rh;
        self
    }

    /// Set the maximum virtual distance in meters.
    #[must_use]
    pub fn with_max_distance_m(mut self, dist: f32) -> Self {
        self.max_distance_m = dist;
        self
    }

    /// Set the per-voice distance spread (0.0 = uniform, 1.0 = max spread).
    #[must_use]
    pub fn with_per_voice_spread(mut self, spread: f32) -> Self {
        self.per_voice_spread = spread;
        self
    }

    /// Set the wet/dry mix (0.0 = bypass, 1.0 = full effect).
    #[must_use]
    pub fn with_mix(mut self, mix: f32) -> Self {
        self.mix = mix;
        self
    }

    /// Set the number of cascaded filter stages.
    #[must_use]
    pub fn with_filter_stages(mut self, stages: usize) -> Self {
        self.filter_stages = stages;
        self
    }

    /// Enable or disable presence compensation high-shelf.
    #[must_use]
    pub fn with_presence_compensation(mut self, enable: bool) -> Self {
        self.presence_compensation = enable;
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
        if !self.temperature_c.is_finite()
            || self.temperature_c < -40.0
            || self.temperature_c > 60.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "temperature_c",
                reason: format!(
                    "temperature_c = {}: must be finite and in [-40, 60]",
                    self.temperature_c,
                ),
            });
        }
        if !self.humidity_percent.is_finite()
            || self.humidity_percent < 0.0
            || self.humidity_percent > 100.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "humidity_percent",
                reason: format!(
                    "humidity_percent = {}: must be finite and in [0, 100]",
                    self.humidity_percent,
                ),
            });
        }
        if !self.max_distance_m.is_finite()
            || self.max_distance_m < 0.0
            || self.max_distance_m > 100.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "max_distance_m",
                reason: format!(
                    "max_distance_m = {}: must be finite and in [0, 100]",
                    self.max_distance_m,
                ),
            });
        }
        if !self.per_voice_spread.is_finite()
            || self.per_voice_spread < 0.0
            || self.per_voice_spread > 1.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "per_voice_spread",
                reason: format!(
                    "per_voice_spread = {}: must be finite and in [0.0, 1.0]",
                    self.per_voice_spread,
                ),
            });
        }
        if !self.mix.is_finite() || self.mix < 0.0 || self.mix > 1.0 {
            return Err(KokoroError::InvalidConfig {
                field: "mix",
                reason: format!("mix = {}: must be finite and in [0.0, 1.0]", self.mix),
            });
        }
        if self.filter_stages == 0 || self.filter_stages > 8 {
            return Err(KokoroError::InvalidConfig {
                field: "filter_stages",
                reason: format!("filter_stages = {}: must be in [1, 8]", self.filter_stages),
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

    /// Studio preset: close distances, minimal absorption.
    /// Voices are tightly grouped with subtle HF differences.
    #[must_use]
    pub fn studio() -> Self {
        Self {
            temperature_c: 22.0,
            humidity_percent: 45.0,
            max_distance_m: 1.5,
            per_voice_spread: 0.3,
            mix: 0.3,
            filter_stages: 1,
            presence_compensation: true,
            sample_rate: 24000.0,
        }
    }

    /// Concert hall preset: moderate distances, noticeable absorption.
    /// Voices are spread across a wider stage with clear depth separation.
    #[must_use]
    pub fn concert_hall() -> Self {
        Self {
            temperature_c: 20.0,
            humidity_percent: 55.0,
            max_distance_m: 8.0,
            per_voice_spread: 0.6,
            mix: 0.6,
            filter_stages: 2,
            presence_compensation: true,
            sample_rate: 24000.0,
        }
    }

    /// Outdoor preset: maximum distance effect with dry air.
    /// Farthest voices are dramatically darker than close ones.
    #[must_use]
    pub fn outdoor() -> Self {
        Self {
            temperature_c: 25.0,
            humidity_percent: 35.0,
            max_distance_m: 20.0,
            per_voice_spread: 0.8,
            mix: 0.8,
            filter_stages: 3,
            presence_compensation: false,
            sample_rate: 24000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// One-pole lowpass filter
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
        let cutoff = cutoff_hz.clamp(1.0, sample_rate * 0.499);
        let w = (-2.0 * std::f32::consts::PI * cutoff / sample_rate).exp();
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
// High-shelf filter (presence compensation)
// ---------------------------------------------------------------------------

/// First-order high-shelf filter for subtle presence compensation.
///
/// Adds a small boost above the shelf frequency to counteract mild
/// air absorption at moderate distances, preventing voices from
/// sounding excessively dull.
#[derive(Debug, Clone)]
struct HighShelf {
    b0: f32,
    b1: f32,
    a1: f32,
    x1: f32,
    y1: f32,
}

impl HighShelf {
    /// Create a high-shelf filter with `gain_db` boost above `freq_hz`.
    fn new(freq_hz: f32, gain_db: f32, sample_rate: f32) -> Self {
        if gain_db.abs() < 0.01 {
            return Self::passthrough();
        }

        let a = 10.0_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * freq_hz / sample_rate;
        let tan_w0_2 = (w0 / 2.0).tan();

        let sqrt_a = a.sqrt();
        // High-shelf: swap numerator and denominator sqrt(A) relative to low-shelf.
        let num = tan_w0_2 / sqrt_a;
        let den = tan_w0_2 * sqrt_a;

        let denom = 1.0 + num;
        let b0 = (1.0 + den) / denom;
        let b1 = (-1.0 + den) / denom;
        let a1 = (-1.0 + num) / denom;

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
// Per-voice absorption chain
// ---------------------------------------------------------------------------

/// Filter state for a single voice's air absorption chain.
#[derive(Debug, Clone)]
struct VoiceAbsorptionChain {
    /// Virtual distance in meters for this voice.
    distance_m: f32,
    /// Cascaded one-pole lowpass filters (steeper rolloff with more stages).
    lpf_stages: Vec<OnePoleLP>,
    /// Optional presence compensation shelf.
    presence_shelf: HighShelf,
    /// Wet/dry mix ratio.
    mix: f32,
}

impl VoiceAbsorptionChain {
    fn new(voice_idx: usize, n_voices: usize, config: &AirAbsorptionConfig) -> Self {
        let sr = config.sample_rate;

        let distance_m = compute_voice_distance_m(
            voice_idx,
            n_voices,
            config.max_distance_m,
            config.per_voice_spread,
        );

        // Compute absorption cutoff frequency based on distance and atmosphere.
        let cutoff_hz = absorption_cutoff(
            distance_m,
            config.temperature_c,
            config.humidity_percent,
            sr,
        );

        let lpf_stages = (0..config.filter_stages)
            .map(|_| OnePoleLP::new(cutoff_hz, sr))
            .collect();

        // Presence compensation: small HF boost at moderate distances.
        let presence_shelf = if config.presence_compensation && distance_m > 0.3 && distance_m < 5.0
        {
            // Scale compensation: peaks around 1-2m, tapers at extremes.
            let norm_dist = ((distance_m - 0.3) / 4.7).clamp(0.0, 1.0);
            let gain_db = 1.5 * (1.0 - (norm_dist * 2.0 - 1.0).powi(2));
            let shelf_freq = 6000.0_f32.min(sr * 0.4);
            HighShelf::new(shelf_freq, gain_db, sr)
        } else {
            HighShelf::passthrough()
        };

        Self {
            distance_m,
            lpf_stages,
            presence_shelf,
            mix: config.mix,
        }
    }

    /// Process one sample through the absorption chain with wet/dry mix.
    #[inline]
    fn process_sample(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            return 0.0;
        }

        let dry = x;

        // Cascaded lowpass (air absorption).
        let mut wet = x;
        for lpf in &mut self.lpf_stages {
            wet = lpf.process(wet);
        }

        // Presence compensation.
        wet = self.presence_shelf.process(wet);

        // Wet/dry mix.
        let out = dry * (1.0 - self.mix) + wet * self.mix;
        if out.is_finite() {
            out
        } else {
            0.0
        }
    }

    fn reset(&mut self) {
        for lpf in &mut self.lpf_stages {
            lpf.reset();
        }
        self.presence_shelf.reset();
    }
}

// ---------------------------------------------------------------------------
// Distance computation
// ---------------------------------------------------------------------------

/// Compute virtual distance in meters for a voice in the ensemble.
///
/// Voice 0 is closest, voice N-1 is farthest. The center of the range
/// is always `max_distance_m / 2`. `spread` controls how wide the range
/// is around that center (0 = all at center, 1 = full [0, max_distance_m]).
fn compute_voice_distance_m(
    voice_idx: usize,
    n_voices: usize,
    max_distance_m: f32,
    spread: f32,
) -> f32 {
    let center = max_distance_m * 0.5;

    if n_voices <= 1 {
        return center.clamp(0.0, max_distance_m);
    }

    let half_range = center * spread;
    let min_dist = (center - half_range).clamp(0.0, max_distance_m);
    let max_dist = (center + half_range).clamp(0.0, max_distance_m);

    let t = voice_idx as f32 / (n_voices - 1) as f32;
    min_dist + t * (max_dist - min_dist)
}

// ---------------------------------------------------------------------------
// Absorption coefficient computation
// ---------------------------------------------------------------------------

/// Compute the effective lowpass cutoff frequency for a given distance
/// and atmospheric conditions.
///
/// Simplified model based on ISO 9613-1: absorption increases roughly
/// as f^1.7. We invert this to find the frequency at which absorption
/// reaches a perceptual threshold, then use that as the LPF cutoff.
///
/// Temperature and humidity modulate the absorption rate:
/// - Higher temperature -> slightly less absorption (faster molecular relaxation)
/// - Higher humidity -> more complex: reduces absorption 1-4 kHz but
///   increases it above ~10 kHz. We use a simplified net effect.
fn absorption_cutoff(
    distance_m: f32,
    temperature_c: f32,
    humidity_percent: f32,
    sample_rate: f32,
) -> f32 {
    let nyquist = sample_rate * 0.499;

    if distance_m <= 0.0 {
        return nyquist;
    }

    // Base absorption coefficient at 10 kHz, 20C, 50% RH.
    // ISO 9613-1 gives ~0.02 dB/m for pure atmospheric absorption, but
    // for artistic chorus depth modeling we use an exaggerated coefficient
    // (0.3 dB/m) to produce perceptible differences over the 1-20m range
    // typical in chorus voice placement.
    let base_alpha_10k = 0.3_f32;

    // Temperature correction: absorption decreases ~1% per degree above 20C.
    let temp_factor = 1.0 - (temperature_c - 20.0) * 0.01;
    let temp_factor = temp_factor.clamp(0.5, 2.0);

    // Humidity correction: simplified net effect.
    // Very dry air (<20%) absorbs more; humid air (>70%) absorbs slightly less
    // at mid frequencies but more at very high frequencies.
    let humidity_factor = if humidity_percent < 20.0 {
        1.3
    } else if humidity_percent > 70.0 {
        0.9
    } else {
        1.0 - (humidity_percent - 50.0) * 0.002
    };
    let humidity_factor = humidity_factor.clamp(0.5, 2.0);

    let alpha = base_alpha_10k * temp_factor * humidity_factor;

    // Total absorption at 10 kHz over the given distance (dB).
    let total_db = alpha * distance_m;

    // Find the frequency where absorption reaches a perceptual threshold
    // (3 dB). Using the f^1.7 power law:
    //   absorption(f) = total_db * (f / 10000)^1.7
    //   Solve for f where absorption(f) = threshold_db:
    //   f = 10000 * (threshold_db / total_db)^(1/1.7)
    let threshold_db = 3.0;

    if total_db <= 0.001 {
        return nyquist;
    }

    let ratio = (threshold_db / total_db).clamp(0.0, 100.0);
    let cutoff = 10000.0 * ratio.powf(1.0 / 1.7);

    cutoff.clamp(500.0, nyquist)
}

// ---------------------------------------------------------------------------
// AirAbsorptionProcessor
// ---------------------------------------------------------------------------

/// Stateful per-voice air absorption processor for multi-voice chorus.
///
/// Models frequency-dependent high-frequency attenuation that increases
/// with distance, providing natural depth cues across chorus voices.
#[derive(Debug, Clone)]
pub struct AirAbsorptionProcessor {
    config: AirAbsorptionConfig,
    chains: Vec<VoiceAbsorptionChain>,
}

impl AirAbsorptionProcessor {
    /// Create a new air absorption processor for `n_voices` voices.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range.
    pub fn new(config: &AirAbsorptionConfig, n_voices: usize) -> Result<Self, KokoroError> {
        config.validate()?;

        let chains = (0..n_voices)
            .map(|i| VoiceAbsorptionChain::new(i, n_voices, config))
            .collect();

        Ok(Self {
            config: *config,
            chains,
        })
    }

    /// Process per-voice audio buffers in-place.
    ///
    /// `voices` must have the same length as `n_voices` passed to the
    /// constructor. Voices beyond the chain count are left unchanged.
    pub fn process_voices(&mut self, voices: &mut [Vec<f32>]) {
        for (voice, chain) in voices.iter_mut().zip(self.chains.iter_mut()) {
            for sample in voice.iter_mut() {
                *sample = chain.process_sample(*sample);
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
    pub fn config(&self) -> &AirAbsorptionConfig {
        &self.config
    }

    /// Get the effective distance in meters for a specific voice index.
    ///
    /// Returns `None` if `voice_idx >= n_voices`.
    #[must_use]
    pub fn voice_distance_m(&self, voice_idx: usize) -> Option<f32> {
        self.chains.get(voice_idx).map(|c| c.distance_m)
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
        AirAbsorptionConfig::new()
            .validate()
            .expect("default config should be valid");
    }

    #[test]
    fn test_config_builder_roundtrip() {
        let cfg = AirAbsorptionConfig::new()
            .with_temperature_c(25.0)
            .with_humidity_percent(60.0)
            .with_max_distance_m(5.0)
            .with_per_voice_spread(0.7)
            .with_mix(0.8)
            .with_filter_stages(3)
            .with_presence_compensation(false)
            .with_sample_rate(48000.0);
        cfg.validate().expect("builder config should be valid");
        assert_eq!(cfg.temperature_c, 25.0);
        assert_eq!(cfg.humidity_percent, 60.0);
        assert_eq!(cfg.max_distance_m, 5.0);
        assert_eq!(cfg.per_voice_spread, 0.7);
        assert_eq!(cfg.mix, 0.8);
        assert_eq!(cfg.filter_stages, 3);
        assert!(!cfg.presence_compensation);
        assert_eq!(cfg.sample_rate, 48000.0);
    }

    #[test]
    fn test_config_invalid_temperature() {
        assert!(AirAbsorptionConfig::new()
            .with_temperature_c(-50.0)
            .validate()
            .is_err());
        assert!(AirAbsorptionConfig::new()
            .with_temperature_c(70.0)
            .validate()
            .is_err());
        assert!(AirAbsorptionConfig::new()
            .with_temperature_c(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_humidity() {
        assert!(AirAbsorptionConfig::new()
            .with_humidity_percent(-1.0)
            .validate()
            .is_err());
        assert!(AirAbsorptionConfig::new()
            .with_humidity_percent(101.0)
            .validate()
            .is_err());
        assert!(AirAbsorptionConfig::new()
            .with_humidity_percent(f32::INFINITY)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_max_distance() {
        assert!(AirAbsorptionConfig::new()
            .with_max_distance_m(-1.0)
            .validate()
            .is_err());
        assert!(AirAbsorptionConfig::new()
            .with_max_distance_m(101.0)
            .validate()
            .is_err());
        assert!(AirAbsorptionConfig::new()
            .with_max_distance_m(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_spread() {
        assert!(AirAbsorptionConfig::new()
            .with_per_voice_spread(-0.1)
            .validate()
            .is_err());
        assert!(AirAbsorptionConfig::new()
            .with_per_voice_spread(1.1)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_mix() {
        assert!(AirAbsorptionConfig::new()
            .with_mix(-0.1)
            .validate()
            .is_err());
        assert!(AirAbsorptionConfig::new().with_mix(1.1).validate().is_err());
        assert!(AirAbsorptionConfig::new()
            .with_mix(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_filter_stages() {
        assert!(AirAbsorptionConfig::new()
            .with_filter_stages(0)
            .validate()
            .is_err());
        assert!(AirAbsorptionConfig::new()
            .with_filter_stages(9)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_sample_rate() {
        assert!(AirAbsorptionConfig::new()
            .with_sample_rate(0.0)
            .validate()
            .is_err());
        assert!(AirAbsorptionConfig::new()
            .with_sample_rate(-1.0)
            .validate()
            .is_err());
        assert!(AirAbsorptionConfig::new()
            .with_sample_rate(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_presets_valid() {
        AirAbsorptionConfig::studio().validate().expect("studio");
        AirAbsorptionConfig::concert_hall()
            .validate()
            .expect("concert_hall");
        AirAbsorptionConfig::outdoor().validate().expect("outdoor");
    }

    // --- Distance assignment ---

    #[test]
    fn test_voice_distances_monotonic() {
        let cfg = AirAbsorptionConfig::new().with_per_voice_spread(0.8);
        let proc = AirAbsorptionProcessor::new(&cfg, 5).expect("valid");
        let distances: Vec<f32> = (0..5).map(|i| proc.voice_distance_m(i).unwrap()).collect();
        for i in 1..distances.len() {
            assert!(
                distances[i] >= distances[i - 1],
                "distances should be monotonically increasing: {distances:?}",
            );
        }
    }

    #[test]
    fn test_single_voice_distance() {
        let cfg = AirAbsorptionConfig::new()
            .with_max_distance_m(4.0)
            .with_per_voice_spread(0.5);
        let proc = AirAbsorptionProcessor::new(&cfg, 1).expect("valid");
        let d = proc.voice_distance_m(0).unwrap();
        assert!(d >= 0.0, "distance must be non-negative: {d}");
        assert!(d <= 4.0, "distance must not exceed max: {d}");
    }

    #[test]
    fn test_zero_spread_uniform_distance() {
        let cfg = AirAbsorptionConfig::new()
            .with_max_distance_m(3.0)
            .with_per_voice_spread(0.0);
        let proc = AirAbsorptionProcessor::new(&cfg, 4).expect("valid");
        let d0 = proc.voice_distance_m(0).unwrap();
        for i in 1..4 {
            let d = proc.voice_distance_m(i).unwrap();
            assert!(
                (d - d0).abs() < 1e-6,
                "zero spread: voice {i} distance {d} should equal voice 0 distance {d0}",
            );
        }
    }

    // --- Energy non-increase ---

    #[test]
    fn test_energy_non_increase_full_wet() {
        // With full wet mix and no presence compensation, output energy
        // should not exceed input energy (lowpass only removes energy).
        let cfg = AirAbsorptionConfig::new()
            .with_mix(1.0)
            .with_max_distance_m(5.0)
            .with_per_voice_spread(0.8)
            .with_presence_compensation(false);
        let n = 4096;
        let signal = sine_wave(8000.0, n, 0.5);
        let input_energy: f32 = signal.iter().map(|x| x * x).sum();

        let mut voices = vec![signal; 4];
        let mut proc = AirAbsorptionProcessor::new(&cfg, 4).expect("valid");
        proc.process_voices(&mut voices);

        for (vi, voice) in voices.iter().enumerate() {
            let output_energy: f32 = voice.iter().map(|x| x * x).sum();
            assert!(
                output_energy <= input_energy * 1.01,
                "voice {vi}: output energy {output_energy} should not exceed input {input_energy}",
            );
        }
    }

    // --- HF rolloff verification ---

    #[test]
    fn test_hf_rolloff_increases_with_distance() {
        // Farthest voice should attenuate HF more than closest voice.
        let cfg = AirAbsorptionConfig::new()
            .with_mix(1.0)
            .with_max_distance_m(10.0)
            .with_per_voice_spread(0.9)
            .with_presence_compensation(false);
        let n = 4096;
        let hf = sine_wave(8000.0, n, 0.5);

        let mut voices = vec![hf.clone(), hf];
        let mut proc = AirAbsorptionProcessor::new(&cfg, 2).expect("valid");
        proc.process_voices(&mut voices);

        let rms_close = rms(&voices[0]);
        let rms_far = rms(&voices[1]);
        assert!(
            rms_close > rms_far * 1.01,
            "close voice should have more HF energy than far: close={rms_close}, far={rms_far}",
        );
    }

    #[test]
    fn test_lf_passes_through_mostly() {
        // Low frequencies should be minimally affected by air absorption.
        let cfg = AirAbsorptionConfig::new()
            .with_mix(1.0)
            .with_max_distance_m(5.0)
            .with_per_voice_spread(0.8)
            .with_presence_compensation(false);
        let n = 4096;
        let lf = sine_wave(200.0, n, 0.5);
        let input_rms = rms(&lf);

        let mut voices = vec![lf];
        let mut proc = AirAbsorptionProcessor::new(&cfg, 1).expect("valid");
        proc.process_voices(&mut voices);

        let output_rms = rms(&voices[0]);
        let ratio = output_rms / input_rms;
        assert!(
            ratio > 0.90,
            "200 Hz should pass through with minimal loss: ratio={ratio}",
        );
    }

    // --- Distance monotonicity of absorption ---

    #[test]
    fn test_more_distance_means_more_absorption() {
        // With more stages and more distance, a HF signal should be
        // attenuated more.
        let n = 4096;
        let hf = sine_wave(10000.0, n, 0.5);

        let cfg_close = AirAbsorptionConfig::new()
            .with_mix(1.0)
            .with_max_distance_m(1.0)
            .with_per_voice_spread(0.0)
            .with_presence_compensation(false);
        let cfg_far = AirAbsorptionConfig::new()
            .with_mix(1.0)
            .with_max_distance_m(15.0)
            .with_per_voice_spread(0.0)
            .with_presence_compensation(false);

        let mut v_close = vec![hf.clone()];
        let mut v_far = vec![hf];
        let mut p_close = AirAbsorptionProcessor::new(&cfg_close, 1).expect("valid");
        let mut p_far = AirAbsorptionProcessor::new(&cfg_far, 1).expect("valid");
        p_close.process_voices(&mut v_close);
        p_far.process_voices(&mut v_far);

        let rms_close = rms(&v_close[0]);
        let rms_far = rms(&v_far[0]);
        assert!(
            rms_close > rms_far,
            "closer distance should preserve more HF: close={rms_close}, far={rms_far}",
        );
    }

    // --- NaN safety ---

    #[test]
    fn test_nan_input_produces_finite_output() {
        let cfg = AirAbsorptionConfig::new();
        let mut proc = AirAbsorptionProcessor::new(&cfg, 1).expect("valid");
        let mut voices = vec![vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5, -0.5]];
        proc.process_voices(&mut voices);
        for (i, &s) in voices[0].iter().enumerate() {
            assert!(s.is_finite(), "sample {i} should be finite, got {s}");
        }
    }

    #[test]
    fn test_all_zeros_stays_zero() {
        let cfg = AirAbsorptionConfig::new()
            .with_mix(1.0)
            .with_presence_compensation(false);
        let mut proc = AirAbsorptionProcessor::new(&cfg, 2).expect("valid");
        let mut voices = vec![vec![0.0; 256], vec![0.0; 256]];
        proc.process_voices(&mut voices);
        for (vi, voice) in voices.iter().enumerate() {
            for (si, &s) in voice.iter().enumerate() {
                assert_eq!(s, 0.0, "voice {vi} sample {si} should be 0.0, got {s}");
            }
        }
    }

    // --- Mix parameter ---

    #[test]
    fn test_zero_mix_is_bypass() {
        let cfg = AirAbsorptionConfig::new().with_mix(0.0);
        let n = 1024;
        let signal = sine_wave(8000.0, n, 0.5);
        let expected = signal.clone();

        let mut voices = vec![signal];
        let mut proc = AirAbsorptionProcessor::new(&cfg, 1).expect("valid");
        proc.process_voices(&mut voices);

        for (i, (&out, &exp)) in voices[0].iter().zip(expected.iter()).enumerate() {
            assert!(
                (out - exp).abs() < 1e-6,
                "sample {i}: expected {exp}, got {out} (mix=0 should be bypass)",
            );
        }
    }

    // --- Reset clears state ---

    #[test]
    fn test_reset_clears_filter_state() {
        let cfg = AirAbsorptionConfig::new();
        let mut proc = AirAbsorptionProcessor::new(&cfg, 2).expect("valid");
        let mut voices = vec![vec![0.5; 100], vec![0.3; 100]];
        proc.process_voices(&mut voices);
        proc.reset();
        for chain in &proc.chains {
            for lpf in &chain.lpf_stages {
                assert_eq!(lpf.z1, 0.0, "lpf state should be cleared after reset");
            }
            assert_eq!(chain.presence_shelf.x1, 0.0, "shelf x1 should be cleared");
            assert_eq!(chain.presence_shelf.y1, 0.0, "shelf y1 should be cleared");
        }
    }

    // --- Accessor tests ---

    #[test]
    fn test_n_voices_accessor() {
        let cfg = AirAbsorptionConfig::new();
        let proc = AirAbsorptionProcessor::new(&cfg, 6).expect("valid");
        assert_eq!(proc.n_voices(), 6);
    }

    #[test]
    fn test_voice_distance_out_of_range() {
        let cfg = AirAbsorptionConfig::new();
        let proc = AirAbsorptionProcessor::new(&cfg, 3).expect("valid");
        assert!(proc.voice_distance_m(3).is_none());
        assert!(proc.voice_distance_m(100).is_none());
    }

    // --- Empty voices ---

    #[test]
    fn test_empty_voices() {
        let cfg = AirAbsorptionConfig::new();
        let mut proc = AirAbsorptionProcessor::new(&cfg, 2).expect("valid");
        let mut voices: Vec<Vec<f32>> = vec![vec![], vec![]];
        proc.process_voices(&mut voices);
        assert!(voices[0].is_empty());
        assert!(voices[1].is_empty());
    }

    // --- Temperature and humidity effects ---

    #[test]
    fn test_cold_air_absorbs_more() {
        // Cold air should produce more HF absorption than warm air.
        // Use large distance (30m center) so the temperature difference
        // produces clearly different cutoff frequencies.
        let n = 4096;
        let hf = sine_wave(10000.0, n, 0.5);

        let cfg_cold = AirAbsorptionConfig::new()
            .with_temperature_c(0.0)
            .with_mix(1.0)
            .with_max_distance_m(60.0)
            .with_per_voice_spread(0.0)
            .with_filter_stages(3)
            .with_presence_compensation(false);
        let cfg_warm = AirAbsorptionConfig::new()
            .with_temperature_c(35.0)
            .with_mix(1.0)
            .with_max_distance_m(60.0)
            .with_per_voice_spread(0.0)
            .with_filter_stages(3)
            .with_presence_compensation(false);

        let mut v_cold = vec![hf.clone()];
        let mut v_warm = vec![hf];
        let mut p_cold = AirAbsorptionProcessor::new(&cfg_cold, 1).expect("valid");
        let mut p_warm = AirAbsorptionProcessor::new(&cfg_warm, 1).expect("valid");
        p_cold.process_voices(&mut v_cold);
        p_warm.process_voices(&mut v_warm);

        let rms_cold = rms(&v_cold[0]);
        let rms_warm = rms(&v_warm[0]);
        assert!(
            rms_warm > rms_cold,
            "warm air should absorb less HF: warm={rms_warm}, cold={rms_cold}",
        );
    }

    // --- Absorption cutoff sanity ---

    #[test]
    fn test_absorption_cutoff_zero_distance() {
        let cutoff = absorption_cutoff(0.0, 20.0, 50.0, 24000.0);
        assert!(
            cutoff > 11000.0,
            "zero distance should give near-Nyquist cutoff: {cutoff}",
        );
    }

    #[test]
    fn test_absorption_cutoff_large_distance() {
        let cutoff = absorption_cutoff(50.0, 20.0, 50.0, 24000.0);
        assert!(
            cutoff < 8000.0,
            "large distance should give low cutoff: {cutoff}",
        );
    }

    #[test]
    fn test_absorption_cutoff_always_finite() {
        // Various edge cases should all produce finite, positive cutoffs.
        let cases = [
            (0.0, 20.0, 50.0),
            (0.001, -40.0, 0.0),
            (100.0, 60.0, 100.0),
            (50.0, 0.0, 15.0),
        ];
        for (dist, temp, hum) in cases {
            let cutoff = absorption_cutoff(dist, temp, hum, 24000.0);
            assert!(
                cutoff.is_finite() && cutoff > 0.0,
                "cutoff should be finite and positive for dist={dist}, temp={temp}, hum={hum}: got {cutoff}",
            );
        }
    }
}
