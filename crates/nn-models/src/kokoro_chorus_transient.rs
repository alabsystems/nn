// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Transient shaper for Kokoro chorus consonant and sustain control.
//!
//! In multi-voice TTS chorus, consonant onsets (plosives, fricatives) and
//! sustained vowel portions benefit from independent gain control. Boosting
//! transient attacks makes consonants pop and improves intelligibility in
//! dense chorus textures, while sustain control shapes the body of held
//! notes and vowels.
//!
//! # Architecture
//!
//! ```text
//! Input --> Bandpass detection --> Fast envelope (1ms attack)
//!                                 Slow envelope (20ms attack)
//!       --> Transient = fast_env - slow_env  (onset energy)
//!       --> Sustain   = slow_env             (held energy)
//!       --> gain = attack_gain^transient * sustain_gain^sustain
//!       --> Output = Input * gain
//! ```
//!
//! The bandpass filter on the detection sidechain focuses transient
//! detection on the speech consonant range (1-8 kHz by default), where
//! plosive bursts and fricative noise carry the most energy. This avoids
//! false transient triggers from low-frequency content like room rumble.
//!
//! # Envelope followers
//!
//! Two one-pole envelope followers track the detection signal:
//! - **Fast** (1ms attack, 50ms release): responds instantly to onsets.
//! - **Slow** (20ms attack, 50ms release): tracks the sustain envelope.
//!
//! The difference `fast - slow` is positive during transient onsets and
//! decays to zero during sustained portions. The slow envelope by itself
//! is positive during held notes.
//!
//! # Gain application
//!
//! Transient and sustain gains are applied multiplicatively:
//! - `transient_gain = 10^(attack_gain_db/20)` applied proportionally to
//!   the normalized transient signal.
//! - `sustain_gain = 10^(sustain_gain_db/20)` applied proportionally to
//!   the normalized sustain signal.
//! - At 0 dB for both, the output equals the input (identity).
//!
//! # References
//!
//! - Giannoulis, D. et al. "Digital Dynamic Range Compressor Design."
//!   Journal of the Audio Engineering Society, 60(6), 2012.
//! - Verfaille, V. & Arfib, D. "Adaptive Digital Audio Effects."
//!   Proc. DAFx-02, Hamburg, 2002.
//!
//! Part of #4581, #3351.

use crate::kokoro_chorus_saturation::db_to_linear;
use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the transient shaper.
///
/// Constructed via [`TransientConfig::new`] (required for cross-crate use
/// due to `#[non_exhaustive]`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct TransientConfig {
    /// Attack gain in dB: boost (+) or cut (-) transient onsets (consonants).
    /// Range: -12.0 to 12.0. Default: 0.0 (neutral).
    pub attack_gain_db: f32,
    /// Sustain gain in dB: boost (+) or cut (-) sustained portions (vowels).
    /// Range: -12.0 to 12.0. Default: 0.0 (neutral).
    pub sustain_gain_db: f32,
    /// Transient detection sensitivity. Higher values detect subtler onsets.
    /// Range: 0.1 to 10.0. Default: 1.0.
    pub sensitivity: f32,
    /// Center frequency for the bandpass detection filter (Hz).
    /// Focuses transient detection on the speech consonant range.
    /// Range: 1000.0 to 8000.0. Default: 3000.0.
    pub detection_freq: f32,
}

impl Default for TransientConfig {
    fn default() -> Self {
        Self {
            attack_gain_db: 0.0,
            sustain_gain_db: 0.0,
            sensitivity: 1.0,
            detection_freq: 3000.0,
        }
    }
}

impl TransientConfig {
    /// Create a new transient config with default values (neutral, identity).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the attack (transient onset) gain in dB.
    #[must_use]
    pub fn with_attack(mut self, gain_db: f32) -> Self {
        self.attack_gain_db = gain_db;
        self
    }

    /// Set the sustain (held portion) gain in dB.
    #[must_use]
    pub fn with_sustain(mut self, gain_db: f32) -> Self {
        self.sustain_gain_db = gain_db;
        self
    }

    /// Set the detection sensitivity.
    #[must_use]
    pub fn with_sensitivity(mut self, sensitivity: f32) -> Self {
        self.sensitivity = sensitivity;
        self
    }

    /// Set the bandpass detection center frequency in Hz.
    #[must_use]
    pub fn with_detection_freq(mut self, freq: f32) -> Self {
        self.detection_freq = freq;
        self
    }

    /// Validate all parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.attack_gain_db.is_finite()
            || self.attack_gain_db < -12.0
            || self.attack_gain_db > 12.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "attack_gain_db",
                reason: format!(
                    "attack_gain_db = {}: must be finite and in [-12, 12]",
                    self.attack_gain_db,
                ),
            });
        }
        if !self.sustain_gain_db.is_finite()
            || self.sustain_gain_db < -12.0
            || self.sustain_gain_db > 12.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "sustain_gain_db",
                reason: format!(
                    "sustain_gain_db = {}: must be finite and in [-12, 12]",
                    self.sustain_gain_db,
                ),
            });
        }
        if !self.sensitivity.is_finite() || self.sensitivity < 0.1 || self.sensitivity > 10.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sensitivity",
                reason: format!(
                    "sensitivity = {}: must be finite and in [0.1, 10.0]",
                    self.sensitivity,
                ),
            });
        }
        if !self.detection_freq.is_finite()
            || self.detection_freq < 1000.0
            || self.detection_freq > 8000.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "detection_freq",
                reason: format!(
                    "detection_freq = {}: must be finite and in [1000, 8000]",
                    self.detection_freq,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// One-pole envelope follower
// ---------------------------------------------------------------------------

/// Single-pole envelope follower with separate attack and release coefficients.
///
/// Tracks the absolute value of the input signal with asymmetric time
/// constants: fast attack to catch transients, slower release for decay.
#[derive(Debug, Clone)]
struct EnvelopeFollower {
    attack_coeff: f32,
    release_coeff: f32,
    envelope: f32,
}

impl EnvelopeFollower {
    /// Create a new envelope follower from time constants in milliseconds.
    fn new(attack_ms: f32, release_ms: f32, sample_rate: f32) -> Self {
        let attack_coeff = Self::time_constant(attack_ms, sample_rate);
        let release_coeff = Self::time_constant(release_ms, sample_rate);
        Self {
            attack_coeff,
            release_coeff,
            envelope: 0.0,
        }
    }

    /// Compute one-pole coefficient from time constant in ms.
    ///
    /// coefficient = exp(-1 / (time_ms * 0.001 * sample_rate))
    #[inline]
    fn time_constant(time_ms: f32, sample_rate: f32) -> f32 {
        let tc = (-1.0 / (f64::from(time_ms) * 0.001 * f64::from(sample_rate))).exp() as f32;
        if !tc.is_finite() {
            0.0
        } else {
            tc
        }
    }

    /// Process one sample and return the current envelope value.
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let abs_x = x.abs();
        let coeff = if abs_x > self.envelope {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope = coeff * self.envelope + (1.0 - coeff) * abs_x;
        // Flush denormals.
        if self.envelope < 1e-20 {
            self.envelope = 0.0;
        }
        self.envelope
    }

    /// Reset envelope state.
    fn reset(&mut self) {
        self.envelope = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Bandpass detection filter (2nd-order state-variable)
// ---------------------------------------------------------------------------

/// State-variable bandpass filter for the detection sidechain.
///
/// A 2nd-order resonant bandpass centered at `detection_freq`. The Q is
/// set to produce a moderately wide passband (~2 octaves) covering the
/// speech consonant energy region.
#[derive(Debug, Clone)]
struct DetectionBandpass {
    /// Filter coefficient: 2*sin(pi*fc/fs).
    f: f32,
    /// Damping: 1/Q.
    q_inv: f32,
    /// State variables.
    bp: f32,
    lp: f32,
}

impl DetectionBandpass {
    /// Create a new bandpass centered at `center_hz` with moderate Q.
    fn new(center_hz: f32, sample_rate: f32) -> Self {
        // State-variable filter coefficient.
        let f = 2.0 * (std::f32::consts::PI * center_hz / sample_rate).sin();
        // Q ~ 0.7 gives a ~2-octave bandwidth — good for speech transients.
        let q_inv = 1.0 / 0.7;
        Self {
            f,
            q_inv,
            bp: 0.0,
            lp: 0.0,
        }
    }

    /// Process one sample, returning the bandpassed output.
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
        // Flush denormals / NaN guard.
        if !self.bp.is_finite() {
            self.bp = 0.0;
        }
        if !self.lp.is_finite() {
            self.lp = 0.0;
        }
        self.bp
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.bp = 0.0;
        self.lp = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Transient shaper processor
// ---------------------------------------------------------------------------

/// Stateful transient shaper with dual envelope followers and bandpass
/// detection sidechain.
///
/// Detects transient onsets by comparing a fast envelope follower against
/// a slow one. The difference identifies attack portions while the slow
/// envelope identifies sustained portions. Independent dB gain is applied
/// to each.
#[derive(Debug, Clone)]
pub struct TransientShaper {
    config: TransientConfig,
    /// Bandpass filter on the detection sidechain.
    bandpass: DetectionBandpass,
    /// Fast envelope follower (1ms attack).
    fast_env: EnvelopeFollower,
    /// Slow envelope follower (20ms attack).
    slow_env: EnvelopeFollower,
    /// Linear gain for transient portions.
    attack_gain_linear: f32,
    /// Linear gain for sustained portions.
    sustain_gain_linear: f32,
}

/// Fast envelope attack time in milliseconds.
const FAST_ATTACK_MS: f32 = 1.0;
/// Slow envelope attack time in milliseconds.
const SLOW_ATTACK_MS: f32 = 20.0;
/// Shared release time in milliseconds.
const RELEASE_MS: f32 = 50.0;

impl TransientShaper {
    /// Create a new transient shaper from config and sample rate.
    pub fn new(config: &TransientConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("sample_rate = {sample_rate}: must be finite and positive"),
            });
        }

        let bandpass = DetectionBandpass::new(config.detection_freq, sample_rate);
        let fast_env = EnvelopeFollower::new(FAST_ATTACK_MS, RELEASE_MS, sample_rate);
        let slow_env = EnvelopeFollower::new(SLOW_ATTACK_MS, RELEASE_MS, sample_rate);
        let attack_gain_linear = db_to_linear(config.attack_gain_db);
        let sustain_gain_linear = db_to_linear(config.sustain_gain_db);

        Ok(Self {
            config: *config,
            bandpass,
            fast_env,
            slow_env,
            attack_gain_linear,
            sustain_gain_linear,
        })
    }

    /// Create a processor using Kokoro's default 24 kHz sample rate.
    pub fn new_kokoro(config: &TransientConfig) -> Result<Self, KokoroError> {
        use crate::kokoro_tts::KOKORO_SAMPLE_RATE;
        Self::new(config, KOKORO_SAMPLE_RATE as f32)
    }

    /// Process an audio buffer in-place through the transient shaper.
    ///
    /// Fast path: returns immediately when both gains are 0 dB (identity).
    pub fn process(&mut self, audio: &mut [f32]) {
        // Fast path: neutral config is identity.
        if self.config.attack_gain_db == 0.0 && self.config.sustain_gain_db == 0.0 {
            return;
        }

        let sensitivity = self.config.sensitivity;

        for sample in audio.iter_mut() {
            // Guard non-finite input.
            if !sample.is_finite() {
                *sample = 0.0;
                continue;
            }

            let dry = *sample;

            // Detection sidechain: bandpass filter the input.
            let detected = self.bandpass.process(dry);

            // Dual envelope followers on the detected signal.
            let fast = self.fast_env.process(detected);
            let slow = self.slow_env.process(detected);

            // Transient signal: fast - slow (positive during onsets).
            let transient_raw = (fast - slow).max(0.0) * sensitivity;
            // Sustain signal: slow envelope (positive during held notes).
            let sustain_raw = slow * sensitivity;

            // Normalize to [0, 1] range for gain blending.
            // Use the peak of both signals as the normalization reference.
            let peak = (transient_raw + sustain_raw).max(1e-10);
            let transient_amount = (transient_raw / peak).min(1.0);
            let sustain_amount = (sustain_raw / peak).min(1.0);

            // Compute gain: blend between attack_gain and sustain_gain
            // based on the transient/sustain amounts.
            // At neutral (0 dB), both gains are 1.0, so output = input.
            let gain = transient_amount * self.attack_gain_linear
                + sustain_amount * self.sustain_gain_linear
                + (1.0 - transient_amount - sustain_amount).max(0.0);

            let out = dry * gain;

            // Final NaN/Inf guard.
            *sample = if out.is_finite() { out } else { 0.0 };
        }
    }

    /// Reset all internal state (call between unrelated audio segments).
    pub fn reset(&mut self) {
        self.bandpass.reset();
        self.fast_env.reset();
        self.slow_env.reset();
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &TransientConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Per-voice convenience function
// ---------------------------------------------------------------------------

/// Apply transient shaping to each voice independently.
///
/// Each voice gets its own [`TransientShaper`] instance so envelope state
/// does not leak between voices. This is useful for making consonants
/// pop or smoothing harsh plosives across a chorus ensemble.
///
/// # Errors
///
/// Returns an error if the config is invalid.
pub fn apply_transient_shaping(
    voices: &mut [Vec<f32>],
    config: &TransientConfig,
    sample_rate: f32,
) -> Result<(), KokoroError> {
    for voice in voices.iter_mut() {
        let mut shaper = TransientShaper::new(config, sample_rate)?;
        shaper.process(voice);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 24000.0;

    /// Generate a sine wave at the given frequency and amplitude.
    fn sine_wave(freq: f32, amplitude: f32, duration_ms: f32, sr: f32) -> Vec<f32> {
        let n = (duration_ms * 0.001 * sr) as usize;
        (0..n)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect()
    }

    /// Generate a signal with a sharp transient onset followed by sustain.
    fn transient_then_sustain(sr: f32) -> Vec<f32> {
        let mut signal = Vec::new();
        // 10ms silence.
        signal.extend(vec![0.0; (0.01 * sr) as usize]);
        // 5ms sharp transient (broadband click: alternating +/- 0.8).
        let transient_len = (0.005 * sr) as usize;
        for i in 0..transient_len {
            signal.push(if i % 2 == 0 { 0.8 } else { -0.8 });
        }
        // 100ms sustain (smooth sine at 440 Hz, amplitude 0.3).
        let sustain = sine_wave(440.0, 0.3, 100.0, sr);
        signal.extend(sustain);
        // 10ms silence tail.
        signal.extend(vec![0.0; (0.01 * sr) as usize]);
        signal
    }

    // -- Config tests --------------------------------------------------------

    #[test]
    fn test_config_default_valid() {
        TransientConfig::new()
            .validate()
            .expect("default should be valid");
    }

    #[test]
    fn test_config_builder() {
        let cfg = TransientConfig::new()
            .with_attack(6.0)
            .with_sustain(-3.0)
            .with_sensitivity(2.0)
            .with_detection_freq(4000.0);
        cfg.validate().expect("builder config should be valid");
        assert_eq!(cfg.attack_gain_db, 6.0);
        assert_eq!(cfg.sustain_gain_db, -3.0);
        assert_eq!(cfg.sensitivity, 2.0);
        assert_eq!(cfg.detection_freq, 4000.0);
    }

    #[test]
    fn test_config_invalid_attack_gain() {
        assert!(TransientConfig::new().with_attack(13.0).validate().is_err());
        assert!(TransientConfig::new()
            .with_attack(-13.0)
            .validate()
            .is_err());
        assert!(TransientConfig::new()
            .with_attack(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_sustain_gain() {
        assert!(TransientConfig::new()
            .with_sustain(12.1)
            .validate()
            .is_err());
        assert!(TransientConfig::new()
            .with_sustain(-12.1)
            .validate()
            .is_err());
        assert!(TransientConfig::new()
            .with_sustain(f32::INFINITY)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_sensitivity() {
        assert!(TransientConfig::new()
            .with_sensitivity(0.05)
            .validate()
            .is_err());
        assert!(TransientConfig::new()
            .with_sensitivity(11.0)
            .validate()
            .is_err());
        assert!(TransientConfig::new()
            .with_sensitivity(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_detection_freq() {
        assert!(TransientConfig::new()
            .with_detection_freq(500.0)
            .validate()
            .is_err());
        assert!(TransientConfig::new()
            .with_detection_freq(9000.0)
            .validate()
            .is_err());
        assert!(TransientConfig::new()
            .with_detection_freq(f32::NEG_INFINITY)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_boundary_values_valid() {
        TransientConfig::new()
            .with_attack(-12.0)
            .with_sustain(12.0)
            .with_sensitivity(0.1)
            .with_detection_freq(1000.0)
            .validate()
            .expect("boundary min/max should be valid");

        TransientConfig::new()
            .with_attack(12.0)
            .with_sustain(-12.0)
            .with_sensitivity(10.0)
            .with_detection_freq(8000.0)
            .validate()
            .expect("boundary max/min should be valid");
    }

    // -- Neutral config is identity ------------------------------------------

    #[test]
    fn test_neutral_config_is_identity() {
        let cfg = TransientConfig::new(); // 0 dB / 0 dB
        let mut shaper = TransientShaper::new(&cfg, SR).expect("valid");
        let original = transient_then_sustain(SR);
        let mut processed = original.clone();
        shaper.process(&mut processed);
        // With both gains at 0 dB, output should equal input exactly.
        assert_eq!(
            processed, original,
            "neutral config (0dB/0dB) should be identity",
        );
    }

    // -- Attack boost increases transient peak --------------------------------

    #[test]
    fn test_attack_boost_increases_transient_peak() {
        let cfg = TransientConfig::new()
            .with_attack(6.0)
            .with_sustain(0.0)
            .with_sensitivity(3.0);
        let mut shaper = TransientShaper::new(&cfg, SR).expect("valid");

        let original = transient_then_sustain(SR);
        let mut processed = original.clone();
        shaper.process(&mut processed);

        // Measure peak in the transient region (samples 240-360 at 24kHz
        // for the 10ms-15ms window).
        let transient_start = (0.01 * SR) as usize;
        let transient_end = transient_start + (0.005 * SR) as usize;

        let orig_peak: f32 = original[transient_start..transient_end]
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);

        let proc_peak: f32 = processed[transient_start..transient_end]
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);

        assert!(
            proc_peak > orig_peak,
            "attack boost should increase transient peak: orig={orig_peak}, proc={proc_peak}",
        );
    }

    // -- Sustain boost increases RMS of held portion --------------------------

    #[test]
    fn test_sustain_boost_increases_held_rms() {
        let cfg = TransientConfig::new()
            .with_attack(0.0)
            .with_sustain(6.0)
            .with_sensitivity(3.0);
        let mut shaper = TransientShaper::new(&cfg, SR).expect("valid");

        let original = transient_then_sustain(SR);
        let mut processed = original.clone();
        shaper.process(&mut processed);

        // Measure RMS in the sustain region (after transient, ~15ms-115ms).
        let sustain_start = (0.015 * SR) as usize + (0.02 * SR) as usize;
        let sustain_end = sustain_start + (0.08 * SR) as usize;

        let orig_rms = rms(&original[sustain_start..sustain_end]);
        let proc_rms = rms(&processed[sustain_start..sustain_end]);

        assert!(
            proc_rms > orig_rms,
            "sustain boost should increase held-portion RMS: orig={orig_rms}, proc={proc_rms}",
        );
    }

    // -- NaN/Inf input safety -------------------------------------------------

    #[test]
    fn test_nan_inf_input_clamped_to_zero() {
        let cfg = TransientConfig::new().with_attack(6.0).with_sustain(-3.0);
        let mut shaper = TransientShaper::new(&cfg, SR).expect("valid");
        let mut buf = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5, -0.3];
        shaper.process(&mut buf);
        for (i, &v) in buf.iter().enumerate() {
            assert!(v.is_finite(), "sample {i} should be finite, got {v}");
        }
    }

    // -- Finite output for all parameter combinations -------------------------

    #[test]
    fn test_all_outputs_finite() {
        let configs = [
            TransientConfig::new().with_attack(12.0).with_sustain(12.0),
            TransientConfig::new()
                .with_attack(-12.0)
                .with_sustain(-12.0),
            TransientConfig::new().with_attack(12.0).with_sustain(-12.0),
            TransientConfig::new().with_attack(-12.0).with_sustain(12.0),
        ];
        for cfg in &configs {
            let mut shaper = TransientShaper::new(cfg, SR).expect("valid");
            let mut buf = transient_then_sustain(SR);
            shaper.process(&mut buf);
            for (i, &v) in buf.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "non-finite at sample {i} for config {cfg:?}: {v}",
                );
            }
        }
    }

    // -- Reset clears state ---------------------------------------------------

    #[test]
    fn test_reset_clears_state() {
        let cfg = TransientConfig::new().with_attack(6.0);
        let mut shaper = TransientShaper::new(&cfg, SR).expect("valid");
        let mut buf = transient_then_sustain(SR);
        shaper.process(&mut buf);
        shaper.reset();
        assert_eq!(
            shaper.fast_env.envelope, 0.0,
            "fast_env should be zero after reset"
        );
        assert_eq!(
            shaper.slow_env.envelope, 0.0,
            "slow_env should be zero after reset"
        );
        assert_eq!(
            shaper.bandpass.bp, 0.0,
            "bandpass bp should be zero after reset"
        );
        assert_eq!(
            shaper.bandpass.lp, 0.0,
            "bandpass lp should be zero after reset"
        );
    }

    // -- Per-voice convenience function ----------------------------------------

    #[test]
    fn test_apply_transient_shaping_per_voice() {
        let cfg = TransientConfig::new().with_attack(3.0).with_sustain(-2.0);
        let voice = transient_then_sustain(SR);
        let mut voices = vec![voice.clone(), voice.clone(), voice];
        apply_transient_shaping(&mut voices, &cfg, SR).expect("should succeed");
        for (vi, v) in voices.iter().enumerate() {
            for (si, &s) in v.iter().enumerate() {
                assert!(s.is_finite(), "voice {vi} sample {si} non-finite: {s}");
            }
        }
    }

    #[test]
    fn test_apply_transient_shaping_empty_voices() {
        let cfg = TransientConfig::new();
        let mut voices: Vec<Vec<f32>> = vec![];
        apply_transient_shaping(&mut voices, &cfg, SR).expect("empty voices should succeed");
    }

    #[test]
    fn test_apply_transient_shaping_invalid_config() {
        let cfg = TransientConfig::new().with_attack(20.0); // out of range
        let mut voices = vec![vec![0.0; 100]];
        assert!(
            apply_transient_shaping(&mut voices, &cfg, SR).is_err(),
            "should reject invalid config",
        );
    }

    // -- Invalid sample rate ---------------------------------------------------

    #[test]
    fn test_invalid_sample_rate() {
        let cfg = TransientConfig::new();
        assert!(TransientShaper::new(&cfg, 0.0).is_err());
        assert!(TransientShaper::new(&cfg, -1.0).is_err());
        assert!(TransientShaper::new(&cfg, f32::NAN).is_err());
        assert!(TransientShaper::new(&cfg, f32::INFINITY).is_err());
    }

    // -- Helper ---------------------------------------------------------------

    fn rms(buf: &[f32]) -> f32 {
        if buf.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = buf.iter().map(|x| x * x).sum();
        (sum_sq / buf.len() as f32).sqrt()
    }
}
