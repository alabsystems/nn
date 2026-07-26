// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Vocal presence and intelligibility optimizer for Kokoro chorus.
//!
//! Protects critical speech frequency bands (2-4 kHz formant transitions,
//! 5-8 kHz sibilance) from masking during chorus processing. Dynamically
//! boosts the intelligibility band when masked, enhances consonant transients,
//! and reports an STI-like metric (Speech Transmission Index approximation).
//!
//! # References
//!
//! - Steeneken & Houtgast, "A physical method for measuring speech-transmission
//!   quality," JASA 67(1), 1980.
//! - ANSI/ASA S3.5-1997, "Methods for Calculation of the Speech Intelligibility
//!   Index."
//!
//! Part of #4582, #3351.

use crate::kokoro_chorus_saturation::db_to_linear;
use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the vocal intelligibility optimizer.
///
/// Constructed via [`IntelligibilityConfig::new`] (required for cross-crate
/// use due to `#[non_exhaustive]`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct IntelligibilityConfig {
    /// Protection strength: 0.0 = bypass, 1.0 = maximum. Default: 0.5.
    pub protection_amount: f32,
    /// Low edge of critical band (Hz). Range: 1000-3000. Default: 2000.
    pub critical_band_low_hz: f32,
    /// High edge of critical band (Hz). Range: 3000-6000. Default: 4000.
    pub critical_band_high_hz: f32,
    /// Sibilance protection center (Hz). Range: 4000-10000. Default: 6000.
    pub sibilance_band_hz: f32,
    /// Consonant transient boost (dB). Range: 0.0-6.0. Default: 2.0.
    pub consonant_boost_db: f32,
    /// Enable multiband sibilance analysis. Default: true.
    pub multiband_analysis: bool,
}

impl Default for IntelligibilityConfig {
    fn default() -> Self {
        Self {
            protection_amount: 0.5,
            critical_band_low_hz: 2000.0,
            critical_band_high_hz: 4000.0,
            sibilance_band_hz: 6000.0,
            consonant_boost_db: 2.0,
            multiband_analysis: true,
        }
    }
}

impl IntelligibilityConfig {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    #[must_use]
    pub fn with_protection_amount(mut self, v: f32) -> Self {
        self.protection_amount = v;
        self
    }
    #[must_use]
    pub fn with_critical_band_low_hz(mut self, v: f32) -> Self {
        self.critical_band_low_hz = v;
        self
    }
    #[must_use]
    pub fn with_critical_band_high_hz(mut self, v: f32) -> Self {
        self.critical_band_high_hz = v;
        self
    }
    #[must_use]
    pub fn with_sibilance_band_hz(mut self, v: f32) -> Self {
        self.sibilance_band_hz = v;
        self
    }
    #[must_use]
    pub fn with_consonant_boost_db(mut self, v: f32) -> Self {
        self.consonant_boost_db = v;
        self
    }
    #[must_use]
    pub fn with_multiband_analysis(mut self, v: bool) -> Self {
        self.multiband_analysis = v;
        self
    }

    /// Validate all parameters are within acceptable ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        let err =
            |field: &'static str, reason: String| Err(KokoroError::InvalidConfig { field, reason });
        if !self.protection_amount.is_finite() || !(0.0..=1.0).contains(&self.protection_amount) {
            return err(
                "protection_amount",
                format!("{}: must be in [0, 1]", self.protection_amount),
            );
        }
        if !self.critical_band_low_hz.is_finite()
            || !(1000.0..=3000.0).contains(&self.critical_band_low_hz)
        {
            return err(
                "critical_band_low_hz",
                format!("{}: must be in [1000, 3000]", self.critical_band_low_hz),
            );
        }
        if !self.critical_band_high_hz.is_finite()
            || !(3000.0..=6000.0).contains(&self.critical_band_high_hz)
        {
            return err(
                "critical_band_high_hz",
                format!("{}: must be in [3000, 6000]", self.critical_band_high_hz),
            );
        }
        if self.critical_band_low_hz >= self.critical_band_high_hz {
            return err(
                "critical_band_high_hz",
                format!(
                    "low ({}) must be < high ({})",
                    self.critical_band_low_hz, self.critical_band_high_hz
                ),
            );
        }
        if !self.sibilance_band_hz.is_finite()
            || !(4000.0..=10000.0).contains(&self.sibilance_band_hz)
        {
            return err(
                "sibilance_band_hz",
                format!("{}: must be in [4000, 10000]", self.sibilance_band_hz),
            );
        }
        if !self.consonant_boost_db.is_finite() || !(0.0..=6.0).contains(&self.consonant_boost_db) {
            return err(
                "consonant_boost_db",
                format!("{}: must be in [0, 6]", self.consonant_boost_db),
            );
        }
        Ok(())
    }

    /// Gentle protection: subtle boost to maintain clarity.
    #[must_use]
    pub fn protect() -> Self {
        Self::new()
            .with_protection_amount(0.3)
            .with_consonant_boost_db(1.0)
    }
    /// Enhanced intelligibility: more aggressive presence boost.
    #[must_use]
    pub fn enhance() -> Self {
        Self::new()
            .with_protection_amount(0.7)
            .with_consonant_boost_db(3.0)
    }
    /// Broadcast standard: EBU-R128 style, maximum clarity.
    #[must_use]
    pub fn broadcast_standard() -> Self {
        Self::new()
            .with_protection_amount(0.8)
            .with_consonant_boost_db(4.0)
            .with_critical_band_low_hz(1800.0)
            .with_critical_band_high_hz(4500.0)
    }
    /// Singing: wider band, less consonant boost for natural tone.
    #[must_use]
    pub fn singing() -> Self {
        Self::new()
            .with_protection_amount(0.4)
            .with_consonant_boost_db(1.5)
            .with_critical_band_low_hz(1500.0)
            .with_critical_band_high_hz(5000.0)
            .with_sibilance_band_hz(7000.0)
    }
}

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

/// State-variable bandpass filter for spectral band isolation.
#[derive(Debug, Clone)]
struct BandpassFilter {
    f: f32,
    q_inv: f32,
    bp: f32,
    lp: f32,
}

impl BandpassFilter {
    fn new(center_hz: f32, q: f32, sample_rate: f32) -> Self {
        Self {
            f: 2.0 * (std::f32::consts::PI * center_hz / sample_rate).sin(),
            q_inv: 1.0 / q.max(0.1),
            bp: 0.0,
            lp: 0.0,
        }
    }
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.bp = 0.0;
            self.lp = 0.0;
            return 0.0;
        }
        let hp = x - self.lp - self.q_inv * self.bp;
        self.bp += self.f * hp;
        self.lp += self.f * self.bp;
        if !self.bp.is_finite() {
            self.bp = 0.0;
        }
        if !self.lp.is_finite() {
            self.lp = 0.0;
        }
        self.bp
    }
    fn reset(&mut self) {
        self.bp = 0.0;
        self.lp = 0.0;
    }
}

/// One-pole envelope follower for energy tracking.
#[derive(Debug, Clone)]
struct EnvelopeFollower {
    attack_coeff: f32,
    release_coeff: f32,
    envelope: f32,
}

impl EnvelopeFollower {
    fn new(attack_ms: f32, release_ms: f32, sample_rate: f32) -> Self {
        Self {
            attack_coeff: Self::coeff(attack_ms, sample_rate),
            release_coeff: Self::coeff(release_ms, sample_rate),
            envelope: 0.0,
        }
    }
    #[inline]
    fn coeff(time_ms: f32, sr: f32) -> f32 {
        let c = (-1.0 / (f64::from(time_ms) * 0.001 * f64::from(sr))).exp() as f32;
        if c.is_finite() {
            c
        } else {
            0.0
        }
    }
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let abs_x = x.abs();
        let c = if abs_x > self.envelope {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope = c * self.envelope + (1.0 - c) * abs_x;
        if self.envelope < 1e-20 {
            self.envelope = 0.0;
        }
        self.envelope
    }
    fn reset(&mut self) {
        self.envelope = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------

/// Stateful vocal intelligibility optimizer.
///
/// Analyzes spectral content in the intelligibility band, detects consonant
/// transients, protects sibilance, and reports an STI-like metric.
#[derive(Debug, Clone)]
pub struct IntelligibilityOptimizer {
    config: IntelligibilityConfig,
    intelli_bp: BandpassFilter,
    sibilance_bp: BandpassFilter,
    broadband_env: EnvelopeFollower,
    intelli_env: EnvelopeFollower,
    sibilance_env: EnvelopeFollower,
    fast_env: EnvelopeFollower,
    slow_env: EnvelopeFollower,
    consonant_gain_linear: f32,
    sti_sum: f64,
    sti_count: u64,
}

impl IntelligibilityOptimizer {
    /// Create a new optimizer. Returns error if config or sample_rate invalid.
    pub fn new(config: &IntelligibilityConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("{sample_rate}: must be finite and positive"),
            });
        }
        let center = f32::midpoint(config.critical_band_low_hz, config.critical_band_high_hz);
        let bw = config.critical_band_high_hz - config.critical_band_low_hz;
        let q = (center / bw).max(0.3);
        Ok(Self {
            config: *config,
            intelli_bp: BandpassFilter::new(center, q, sample_rate),
            sibilance_bp: BandpassFilter::new(config.sibilance_band_hz, 1.0, sample_rate),
            broadband_env: EnvelopeFollower::new(5.0, 50.0, sample_rate),
            intelli_env: EnvelopeFollower::new(5.0, 50.0, sample_rate),
            sibilance_env: EnvelopeFollower::new(2.0, 30.0, sample_rate),
            fast_env: EnvelopeFollower::new(1.0, 30.0, sample_rate),
            slow_env: EnvelopeFollower::new(20.0, 50.0, sample_rate),
            consonant_gain_linear: db_to_linear(config.consonant_boost_db),
            sti_sum: 0.0,
            sti_count: 0,
        })
    }

    /// Create using Kokoro's default 24 kHz sample rate.
    pub fn new_kokoro(config: &IntelligibilityConfig) -> Result<Self, KokoroError> {
        Self::new(config, KOKORO_SAMPLE_RATE as f32)
    }

    /// Process a single voice audio buffer in-place.
    /// Fast path: returns immediately when `protection_amount == 0.0`.
    pub fn process_voice(&mut self, audio: &mut [f32]) {
        if self.config.protection_amount == 0.0 {
            return;
        }
        let protection = self.config.protection_amount;
        let mask_threshold = 0.1 + protection * 0.4;

        for sample in audio.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
                continue;
            }
            let dry = *sample;

            // Spectral analysis
            let intelli_sig = self.intelli_bp.process(dry);
            let sibilance_sig = self.sibilance_bp.process(dry);
            let broadband_level = self.broadband_env.process(dry);
            let intelli_level = self.intelli_env.process(intelli_sig);
            let sibilance_level = self.sibilance_env.process(sibilance_sig);

            // Masking detection: boost intelligibility band when masked
            let masking_ratio = if broadband_level > 1e-10 {
                intelli_level / broadband_level
            } else {
                1.0
            };
            if masking_ratio < mask_threshold && broadband_level > 1e-8 {
                let deficit = ((mask_threshold - masking_ratio) / mask_threshold).clamp(0.0, 1.0);
                let boost = db_to_linear(6.0 * protection * deficit);
                *sample = dry + intelli_sig * (boost - 1.0) * protection;
            }

            // Consonant transient detection
            let fast = self.fast_env.process(intelli_sig);
            let slow = self.slow_env.process(intelli_sig);
            let transient = (fast - slow).max(0.0);
            if transient > slow * 0.5 && self.config.consonant_boost_db > 0.0 {
                let strength = (transient / (slow + 1e-10)).min(2.0) / 2.0;
                *sample *= 1.0 + (self.consonant_gain_linear - 1.0) * strength * protection;
            }

            // Sibilance protection
            if self.config.multiband_analysis
                && sibilance_level < broadband_level * 0.05
                && broadband_level > 1e-8
            {
                *sample += sibilance_sig * 0.3 * protection;
            }

            // STI metric update
            if broadband_level > 1e-8 {
                self.sti_sum += f64::from((intelli_level / broadband_level).min(1.0));
                self.sti_count += 1;
            }

            if !sample.is_finite() {
                *sample = 0.0;
            }
        }
    }

    /// STI-like intelligibility score (0.0-1.0). >0.7 = good, <0.5 = poor.
    #[must_use]
    pub fn get_intelligibility_score(&self) -> f32 {
        if self.sti_count == 0 {
            return 1.0;
        }
        ((self.sti_sum / self.sti_count as f64) as f32).clamp(0.0, 1.0)
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.intelli_bp.reset();
        self.sibilance_bp.reset();
        self.broadband_env.reset();
        self.intelli_env.reset();
        self.sibilance_env.reset();
        self.fast_env.reset();
        self.slow_env.reset();
        self.sti_sum = 0.0;
        self.sti_count = 0;
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &IntelligibilityConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Per-voice convenience
// ---------------------------------------------------------------------------

/// Apply intelligibility optimization to each voice independently.
/// Returns average STI-like score across all voices.
pub fn optimize_intelligibility(
    voices: &mut [Vec<f32>],
    config: &IntelligibilityConfig,
    sample_rate: f32,
) -> Result<f32, KokoroError> {
    if voices.is_empty() {
        return Ok(1.0);
    }
    let mut total = 0.0_f32;
    for voice in voices.iter_mut() {
        let mut opt = IntelligibilityOptimizer::new(config, sample_rate)?;
        opt.process_voice(voice);
        total += opt.get_intelligibility_score();
    }
    Ok(total / voices.len() as f32)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    const SR: f32 = KOKORO_SAMPLE_RATE as f32;

    fn sine_wave(freq: f32, n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / SR).sin())
            .collect()
    }
    fn rms(buf: &[f32]) -> f32 {
        if buf.is_empty() {
            return 0.0;
        }
        (buf.iter().map(|x| x * x).sum::<f32>() / buf.len() as f32).sqrt()
    }

    #[test]
    fn test_config_default_valid() {
        IntelligibilityConfig::new()
            .validate()
            .expect("default valid");
    }
    #[test]
    fn test_config_builder_roundtrip() {
        let c = IntelligibilityConfig::new()
            .with_protection_amount(0.8)
            .with_critical_band_low_hz(1800.0)
            .with_critical_band_high_hz(4500.0)
            .with_sibilance_band_hz(7000.0)
            .with_consonant_boost_db(3.0)
            .with_multiband_analysis(false);
        c.validate().expect("builder valid");
        assert_eq!(c.protection_amount, 0.8);
        assert!(!c.multiband_analysis);
    }
    #[test]
    fn test_config_invalid_protection_amount() {
        assert!(IntelligibilityConfig::new()
            .with_protection_amount(-0.1)
            .validate()
            .is_err());
        assert!(IntelligibilityConfig::new()
            .with_protection_amount(1.1)
            .validate()
            .is_err());
        assert!(IntelligibilityConfig::new()
            .with_protection_amount(f32::NAN)
            .validate()
            .is_err());
    }
    #[test]
    fn test_config_invalid_critical_bands() {
        assert!(IntelligibilityConfig::new()
            .with_critical_band_low_hz(500.0)
            .validate()
            .is_err());
        assert!(IntelligibilityConfig::new()
            .with_critical_band_high_hz(7000.0)
            .validate()
            .is_err());
    }
    #[test]
    fn test_config_band_ordering() {
        let c = IntelligibilityConfig {
            critical_band_low_hz: 3000.0,
            critical_band_high_hz: 3000.0,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }
    #[test]
    fn test_config_invalid_sibilance() {
        assert!(IntelligibilityConfig::new()
            .with_sibilance_band_hz(3000.0)
            .validate()
            .is_err());
    }
    #[test]
    fn test_config_invalid_consonant_boost() {
        assert!(IntelligibilityConfig::new()
            .with_consonant_boost_db(-0.1)
            .validate()
            .is_err());
        assert!(IntelligibilityConfig::new()
            .with_consonant_boost_db(7.0)
            .validate()
            .is_err());
    }
    #[test]
    fn test_presets_valid() {
        IntelligibilityConfig::protect()
            .validate()
            .expect("protect");
        IntelligibilityConfig::enhance()
            .validate()
            .expect("enhance");
        IntelligibilityConfig::broadcast_standard()
            .validate()
            .expect("broadcast");
        IntelligibilityConfig::singing()
            .validate()
            .expect("singing");
    }
    #[test]
    fn test_zero_protection_is_bypass() {
        let cfg = IntelligibilityConfig::new().with_protection_amount(0.0);
        let mut opt = IntelligibilityOptimizer::new_kokoro(&cfg).unwrap();
        let mut buf = sine_wave(1000.0, 4096, 0.5);
        let orig = buf.clone();
        opt.process_voice(&mut buf);
        assert_eq!(buf, orig);
    }
    #[test]
    fn test_processing_modifies_signal() {
        let cfg = IntelligibilityConfig::new().with_protection_amount(1.0);
        let mut opt = IntelligibilityOptimizer::new_kokoro(&cfg).unwrap();
        let n = 8192;
        let mut buf: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR;
                0.3 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 3000.0 * t).sin()
            })
            .collect();
        let orig_rms = rms(&buf);
        opt.process_voice(&mut buf);
        assert!((rms(&buf) - orig_rms).abs() > 1e-5);
    }
    #[test]
    fn test_sti_score_in_range() {
        let cfg = IntelligibilityConfig::new();
        let mut opt = IntelligibilityOptimizer::new_kokoro(&cfg).unwrap();
        let mut buf = sine_wave(3000.0, 8192, 0.5);
        opt.process_voice(&mut buf);
        let s = opt.get_intelligibility_score();
        assert!((0.0..=1.0).contains(&s), "score {s} out of range");
    }
    #[test]
    fn test_sti_no_frames() {
        let opt = IntelligibilityOptimizer::new_kokoro(&IntelligibilityConfig::new()).unwrap();
        assert_eq!(opt.get_intelligibility_score(), 1.0);
    }
    #[test]
    fn test_all_outputs_finite() {
        let cfg = IntelligibilityConfig::new().with_protection_amount(1.0);
        let mut opt = IntelligibilityOptimizer::new_kokoro(&cfg).unwrap();
        let mut buf = vec![
            0.0,
            0.5,
            -0.5,
            1.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        opt.process_voice(&mut buf);
        for (i, &v) in buf.iter().enumerate() {
            assert!(v.is_finite(), "sample {i}: {v}");
        }
    }
    #[test]
    fn test_reset_clears_state() {
        let cfg = IntelligibilityConfig::new().with_protection_amount(0.8);
        let mut opt = IntelligibilityOptimizer::new_kokoro(&cfg).unwrap();
        let mut buf = sine_wave(3000.0, 4096, 0.5);
        opt.process_voice(&mut buf);
        assert!(opt.sti_count > 0);
        opt.reset();
        assert_eq!(opt.sti_count, 0);
        assert_eq!(opt.get_intelligibility_score(), 1.0);
    }
    #[test]
    fn test_optimize_per_voice() {
        let mut voices = vec![sine_wave(1000.0, 4096, 0.5), sine_wave(3000.0, 4096, 0.5)];
        let s = optimize_intelligibility(
            &mut voices,
            &IntelligibilityConfig::new().with_protection_amount(0.6),
            SR,
        )
        .unwrap();
        assert!((0.0..=1.0).contains(&s));
    }
    #[test]
    fn test_optimize_empty() {
        assert_eq!(
            optimize_intelligibility(&mut [], &IntelligibilityConfig::new(), SR).unwrap(),
            1.0
        );
    }
    #[test]
    fn test_optimize_invalid_config() {
        assert!(optimize_intelligibility(
            &mut [vec![0.0; 100]],
            &IntelligibilityConfig::new().with_protection_amount(2.0),
            SR
        )
        .is_err());
    }
    #[test]
    fn test_invalid_sample_rate() {
        let cfg = IntelligibilityConfig::new();
        assert!(IntelligibilityOptimizer::new(&cfg, 0.0).is_err());
        assert!(IntelligibilityOptimizer::new(&cfg, -1.0).is_err());
        assert!(IntelligibilityOptimizer::new(&cfg, f32::NAN).is_err());
    }
}
