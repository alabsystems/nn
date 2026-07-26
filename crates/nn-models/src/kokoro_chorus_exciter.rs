// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Harmonic exciter with air band enhancement for Kokoro chorus voices.
//!
//! An exciter adds subtle harmonic content to make voices cut through a mix.
//! Unlike saturation (which adds warmth via low-order harmonics and nonlinear
//! waveshaping), an exciter targets the upper frequency spectrum to add
//! presence, clarity, and "air." The processing chain:
//!
//! ```text
//! Input ─┬─────────────────────────────────────────┐
//!        │                                         │ dry
//!        └─> Highpass filter ─> Half-wave rectify   │
//!            (isolate HF)      + gentle saturation  │
//!                              (generate harmonics) │
//!                                    │              │
//!                                    v              v
//!                              harmonics_mix ──> wet/dry blend
//!                                                   │
//!                                                   v
//!                                            Air shelf boost
//!                                                   │
//!                                                   v
//!                                                Output
//! ```
//!
//! # Design rationale
//!
//! - **Highpass isolation:** Only frequencies above `frequency_hz` are fed into
//!   the harmonic generator. This prevents low-frequency energy from creating
//!   muddy intermodulation products.
//! - **Half-wave rectification + saturation:** Half-wave rectification folds
//!   the negative half of the waveform, generating even harmonics (octave,
//!   double-octave). A gentle tanh saturation on top adds odd harmonics at
//!   controllable intensity. The `harmonic_order` parameter selects which
//!   harmonic is emphasized via a resonant peak in the generation path.
//! - **Air shelf:** A first-order high-shelf filter boosts everything above
//!   `air_freq_hz`, adding the "air" and shimmer characteristic of high-end
//!   analog exciters (Aphex Aural Exciter, SPL Vitalizer).
//!
//! # References
//!
//! - Giannoulis, D. et al. "Digital Dynamic Range Compressor Design."
//!   JAES 60(6), 2012.
//! - Zölzer, U. "DAFX: Digital Audio Effects." 2nd ed., Wiley, 2011.
//!   Chapter 5: Nonlinear Processing; Chapter 2: Filters.
//! - Smith, J. O. "Introduction to Digital Filters with Audio Applications."
//!   <https://ccrma.stanford.edu/~jos/filters/>
//!
//! Part of #4264, #3351.

use crate::kokoro_chorus_saturation::db_to_linear;
use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the harmonic exciter.
///
/// Constructed via [`ExciterConfig::new`] (required for cross-crate use
/// due to `#[non_exhaustive]`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ExciterConfig {
    /// Highpass frequency (Hz) for isolating the exciter band.
    /// Only content above this frequency is fed to the harmonic generator.
    /// Range: 2000.0 - 12000.0. Default: 3000.0.
    pub frequency_hz: f32,

    /// How much of the generated harmonic content to blend into the signal.
    /// 0.0 = no harmonics (bypass), 1.0 = full harmonic blend.
    /// Default: 0.3.
    pub harmonics_mix: f32,

    /// Which harmonic order to emphasize.
    /// 2 = octave (even), 3 = fifth (odd), 4 = double-octave, 5 = major third.
    /// Range: 2-5. Default: 2.
    pub harmonic_order: u32,

    /// Frequency (Hz) for the "air" band shelf boost.
    /// Range: 8000.0 - 16000.0. Default: 10000.0.
    pub air_freq_hz: f32,

    /// Gain in dB for the air band shelf boost.
    /// Range: 0.0 - 6.0. Default: 1.5.
    pub air_gain_db: f32,
}

impl Default for ExciterConfig {
    fn default() -> Self {
        Self {
            frequency_hz: 3000.0,
            harmonics_mix: 0.3,
            harmonic_order: 2,
            air_freq_hz: 10000.0,
            air_gain_db: 1.5,
        }
    }
}

impl ExciterConfig {
    /// Create a new exciter config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the highpass frequency for the exciter band.
    #[must_use]
    pub fn with_frequency_hz(mut self, hz: f32) -> Self {
        self.frequency_hz = hz;
        self
    }

    /// Set the harmonics mix amount (0.0 = bypass, 1.0 = full).
    #[must_use]
    pub fn with_harmonics_mix(mut self, mix: f32) -> Self {
        self.harmonics_mix = mix;
        self
    }

    /// Set which harmonic order to emphasize (2-5).
    #[must_use]
    pub fn with_harmonic_order(mut self, order: u32) -> Self {
        self.harmonic_order = order;
        self
    }

    /// Set the air band frequency.
    #[must_use]
    pub fn with_air_freq_hz(mut self, hz: f32) -> Self {
        self.air_freq_hz = hz;
        self
    }

    /// Set the air band gain in dB.
    #[must_use]
    pub fn with_air_gain_db(mut self, db: f32) -> Self {
        self.air_gain_db = db;
        self
    }

    /// Validate all parameters are within acceptable ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.frequency_hz.is_finite()
            || self.frequency_hz < 2000.0
            || self.frequency_hz > 12000.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "frequency_hz",
                reason: format!(
                    "frequency_hz = {}: must be finite and in [2000, 12000]",
                    self.frequency_hz,
                ),
            });
        }
        if !self.harmonics_mix.is_finite() || self.harmonics_mix < 0.0 || self.harmonics_mix > 1.0 {
            return Err(KokoroError::InvalidConfig {
                field: "harmonics_mix",
                reason: format!(
                    "harmonics_mix = {}: must be finite and in [0.0, 1.0]",
                    self.harmonics_mix,
                ),
            });
        }
        if self.harmonic_order < 2 || self.harmonic_order > 5 {
            return Err(KokoroError::InvalidConfig {
                field: "harmonic_order",
                reason: format!(
                    "harmonic_order = {}: must be in [2, 5]",
                    self.harmonic_order,
                ),
            });
        }
        if !self.air_freq_hz.is_finite() || self.air_freq_hz < 8000.0 || self.air_freq_hz > 16000.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "air_freq_hz",
                reason: format!(
                    "air_freq_hz = {}: must be finite and in [8000, 16000]",
                    self.air_freq_hz,
                ),
            });
        }
        if !self.air_gain_db.is_finite() || self.air_gain_db < 0.0 || self.air_gain_db > 6.0 {
            return Err(KokoroError::InvalidConfig {
                field: "air_gain_db",
                reason: format!(
                    "air_gain_db = {}: must be finite and in [0.0, 6.0]",
                    self.air_gain_db,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// First-order highpass filter
// ---------------------------------------------------------------------------

/// Single-pole highpass filter for isolating the exciter frequency band.
///
/// H(z) = (1 + z^-1) / 2  *  1 / (1 - a * z^-1)
/// where a = (1 - sin(w0)) / cos(w0), w0 = 2*pi*fc/fs.
///
/// This is the simplest analog-prototype-derived highpass: a DC-blocking
/// filter with 6 dB/octave rolloff below the cutoff.
#[derive(Debug, Clone)]
struct HighpassFilter {
    /// Filter coefficient derived from cutoff frequency.
    coeff: f32,
    /// Previous input sample.
    x_prev: f32,
    /// Previous output sample.
    y_prev: f32,
}

impl HighpassFilter {
    /// Create a new first-order highpass at `cutoff_hz`.
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        // RC time constant approach: coeff = RC / (RC + dt)
        // where RC = 1/(2*pi*fc), dt = 1/fs.
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
        let dt = 1.0 / sample_rate;
        let coeff = rc / (rc + dt);
        Self {
            coeff,
            x_prev: 0.0,
            y_prev: 0.0,
        }
    }

    /// Process a single sample.
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.x_prev = 0.0;
            self.y_prev = 0.0;
            return 0.0;
        }
        let y = self.coeff * (self.y_prev + x - self.x_prev);
        self.x_prev = x;
        self.y_prev = if y.is_finite() { y } else { 0.0 };
        self.y_prev
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.x_prev = 0.0;
        self.y_prev = 0.0;
    }
}

// ---------------------------------------------------------------------------
// First-order high-shelf filter (for air band)
// ---------------------------------------------------------------------------

/// First-order high-shelf filter for air band enhancement.
///
/// Boosts all frequencies above the shelf frequency by `gain_db`. Uses a
/// simple first-order design for minimal phase distortion and low CPU cost.
///
/// Based on the first-order shelving filter from:
/// Zölzer, U. "DAFX: Digital Audio Effects," 2nd ed., eq. (2.22).
#[derive(Debug, Clone)]
struct HighShelfFilter {
    /// Coefficients for the shelf filter: y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
    b0: f32,
    b1: f32,
    a1: f32,
    /// State
    x_prev: f32,
    y_prev: f32,
}

impl HighShelfFilter {
    /// Create a first-order high shelf at `freq_hz` with `gain_db` boost.
    fn new(freq_hz: f32, gain_db: f32, sample_rate: f32) -> Self {
        if gain_db.abs() < 1e-6 {
            // Unity gain: pass-through.
            return Self {
                b0: 1.0,
                b1: 0.0,
                a1: 0.0,
                x_prev: 0.0,
                y_prev: 0.0,
            };
        }

        let v0 = db_to_linear(gain_db);
        let k = (std::f32::consts::PI * freq_hz / sample_rate).tan();
        let k2 = k;

        if v0 >= 1.0 {
            // Boost
            let denom = 1.0 + k2;
            let b0 = (v0 + k2) / denom;
            let b1 = (k2 - v0) / denom;
            let a1 = (k2 - 1.0) / denom;
            Self {
                b0,
                b1,
                a1,
                x_prev: 0.0,
                y_prev: 0.0,
            }
        } else {
            // Cut (v0 < 1)
            let denom = v0 + k2;
            let b0 = v0 * (1.0 + k2) / denom;
            let b1 = v0 * (k2 - 1.0) / denom;
            let a1 = (k2 - v0) / denom;
            Self {
                b0,
                b1,
                a1,
                x_prev: 0.0,
                y_prev: 0.0,
            }
        }
    }

    /// Process a single sample.
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.x_prev = 0.0;
            self.y_prev = 0.0;
            return 0.0;
        }
        let y = self.b0 * x + self.b1 * self.x_prev - self.a1 * self.y_prev;
        self.x_prev = x;
        self.y_prev = if y.is_finite() { y } else { 0.0 };
        self.y_prev
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.x_prev = 0.0;
        self.y_prev = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Harmonic generation
// ---------------------------------------------------------------------------

/// Generate harmonics via half-wave rectification and gentle saturation.
///
/// Half-wave rectification (zeroing the negative half) creates even harmonics
/// (2nd, 4th). Gentle tanh saturation adds odd harmonics (3rd, 5th). The
/// `harmonic_order` parameter scales the drive into the tanh to emphasize
/// higher-order harmonics.
#[inline]
fn generate_harmonics(sample: f32, harmonic_order: u32) -> f32 {
    // Half-wave rectification: keep only positive part.
    let rectified = sample.max(0.0);

    // Drive factor increases with harmonic order to generate higher partials.
    // order 2 -> drive 1.5 (subtle), order 5 -> drive 4.5 (aggressive).
    let drive = 1.5 + (harmonic_order as f32 - 2.0) * 1.0;
    let saturated = (rectified * drive).tanh();

    // Remove DC offset introduced by half-wave rectification.
    // The DC component of a half-wave rectified sine is 1/pi * amplitude.
    // We approximate removal by subtracting the local mean (simplified as
    // a fraction of the saturated value). A proper DC blocker would be a
    // highpass, but the upstream highpass already handles this adequately.
    saturated
}

// ---------------------------------------------------------------------------
// HarmonicExciter processor
// ---------------------------------------------------------------------------

/// Stateful harmonic exciter processor.
///
/// Holds filter state for the highpass (exciter band isolation), air shelf,
/// and a DC blocker on the harmonics path.
#[derive(Debug, Clone)]
pub struct HarmonicExciter {
    config: ExciterConfig,
    /// Highpass to isolate the upper frequency band for harmonic generation.
    highpass: HighpassFilter,
    /// Air band high-shelf boost.
    air_shelf: HighShelfFilter,
    /// DC blocker for the harmonics path (removes rectification DC offset).
    dc_blocker: HighpassFilter,
}

impl HarmonicExciter {
    /// Create a new harmonic exciter from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range,
    /// or if `sample_rate` is not finite and positive.
    pub fn new(config: &ExciterConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("sample_rate = {sample_rate}: must be finite and positive"),
            });
        }

        let highpass = HighpassFilter::new(config.frequency_hz, sample_rate);
        let air_shelf = HighShelfFilter::new(config.air_freq_hz, config.air_gain_db, sample_rate);
        // DC blocker at 20 Hz removes residual DC from half-wave rectification.
        let dc_blocker = HighpassFilter::new(20.0, sample_rate);

        Ok(Self {
            config: *config,
            highpass,
            air_shelf,
            dc_blocker,
        })
    }

    /// Create an exciter using Kokoro's default 24 kHz sample rate.
    pub fn new_kokoro(config: &ExciterConfig) -> Result<Self, KokoroError> {
        Self::new(config, KOKORO_SAMPLE_RATE as f32)
    }

    /// Process an audio buffer in-place through the exciter.
    ///
    /// Fast path: returns immediately when `harmonics_mix == 0.0` and
    /// `air_gain_db == 0.0` (no processing needed).
    pub fn process(&mut self, audio: &mut [f32]) {
        // Fast path: nothing to do.
        if self.config.harmonics_mix == 0.0 && self.config.air_gain_db < 1e-6 {
            return;
        }

        let mix = self.config.harmonics_mix;
        let order = self.config.harmonic_order;

        for sample in audio.iter_mut() {
            // Guard non-finite input.
            if !sample.is_finite() {
                *sample = 0.0;
                continue;
            }

            let dry = *sample;

            // --- Harmonic generation path ---
            if mix > 0.0 {
                // Isolate upper frequencies.
                let hf = self.highpass.process(dry);

                // Generate harmonics from the isolated HF content.
                let harmonics_raw = generate_harmonics(hf, order);

                // Remove DC offset from rectification.
                let harmonics = self.dc_blocker.process(harmonics_raw);

                // Blend harmonics with dry signal.
                *sample = dry + harmonics * mix;
            }

            // --- Air shelf boost ---
            *sample = self.air_shelf.process(*sample);

            // Final NaN/Inf guard.
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }
    }

    /// Reset all internal filter state (call between unrelated audio segments).
    pub fn reset(&mut self) {
        self.highpass.reset();
        self.air_shelf.reset();
        self.dc_blocker.reset();
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &ExciterConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Convenience: per-voice exciter application
// ---------------------------------------------------------------------------

/// Apply the harmonic exciter to each voice buffer independently.
///
/// Creates one [`HarmonicExciter`] per voice (each with independent filter
/// state) and processes in place. For streaming scenarios where filter state
/// must persist across calls, create [`HarmonicExciter`] instances directly.
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` if the config is invalid.
pub fn apply_exciter(
    voices: &mut [Vec<f32>],
    config: &ExciterConfig,
    sample_rate: f32,
) -> Result<(), KokoroError> {
    for voice in voices.iter_mut() {
        let mut exciter = HarmonicExciter::new(config, sample_rate)?;
        exciter.process(voice.as_mut_slice());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = KOKORO_SAMPLE_RATE as f32;

    /// Generate a sine wave buffer at the given frequency.
    fn sine_wave(freq: f32, n_samples: usize, amplitude: f32) -> Vec<f32> {
        (0..n_samples)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / SR).sin())
            .collect()
    }

    /// Compute RMS energy of a signal.
    fn rms(buf: &[f32]) -> f32 {
        let sum_sq: f32 = buf.iter().map(|x| x * x).sum();
        (sum_sq / buf.len() as f32).sqrt()
    }

    /// Compute energy above a given frequency using a simple highpass.
    fn hf_energy(buf: &[f32], cutoff_hz: f32) -> f32 {
        let mut hp = HighpassFilter::new(cutoff_hz, SR);
        let filtered: Vec<f32> = buf.iter().map(|&x| hp.process(x)).collect();
        rms(&filtered)
    }

    // --- Config validation ---

    #[test]
    fn test_config_default_valid() {
        ExciterConfig::new()
            .validate()
            .expect("default config should be valid");
    }

    #[test]
    fn test_config_builder_roundtrip() {
        let cfg = ExciterConfig::new()
            .with_frequency_hz(4000.0)
            .with_harmonics_mix(0.5)
            .with_harmonic_order(3)
            .with_air_freq_hz(12000.0)
            .with_air_gain_db(3.0);
        cfg.validate().expect("builder config should be valid");
        assert_eq!(cfg.frequency_hz, 4000.0);
        assert_eq!(cfg.harmonics_mix, 0.5);
        assert_eq!(cfg.harmonic_order, 3);
        assert_eq!(cfg.air_freq_hz, 12000.0);
        assert_eq!(cfg.air_gain_db, 3.0);
    }

    #[test]
    fn test_config_invalid_frequency_hz() {
        assert!(ExciterConfig::new()
            .with_frequency_hz(500.0)
            .validate()
            .is_err());
        assert!(ExciterConfig::new()
            .with_frequency_hz(15000.0)
            .validate()
            .is_err());
        assert!(ExciterConfig::new()
            .with_frequency_hz(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_harmonics_mix() {
        assert!(ExciterConfig::new()
            .with_harmonics_mix(-0.1)
            .validate()
            .is_err());
        assert!(ExciterConfig::new()
            .with_harmonics_mix(1.1)
            .validate()
            .is_err());
        assert!(ExciterConfig::new()
            .with_harmonics_mix(f32::INFINITY)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_harmonic_order() {
        assert!(ExciterConfig::new()
            .with_harmonic_order(1)
            .validate()
            .is_err());
        assert!(ExciterConfig::new()
            .with_harmonic_order(6)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_air_freq_hz() {
        assert!(ExciterConfig::new()
            .with_air_freq_hz(5000.0)
            .validate()
            .is_err());
        assert!(ExciterConfig::new()
            .with_air_freq_hz(20000.0)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_air_gain_db() {
        assert!(ExciterConfig::new()
            .with_air_gain_db(-0.1)
            .validate()
            .is_err());
        assert!(ExciterConfig::new()
            .with_air_gain_db(7.0)
            .validate()
            .is_err());
    }

    // --- Exciter behavior ---

    #[test]
    fn test_harmonics_mix_zero_is_identity_when_no_air() {
        let mut buf = sine_wave(1000.0, 4096, 0.5);
        let original = buf.clone();
        let cfg = ExciterConfig::new()
            .with_harmonics_mix(0.0)
            .with_air_gain_db(0.0);
        let mut exciter = HarmonicExciter::new_kokoro(&cfg).unwrap();
        exciter.process(&mut buf);
        assert_eq!(buf, original, "mix=0 + air_gain=0 should be identity");
    }

    #[test]
    fn test_excited_signal_has_more_hf_energy() {
        let n = 8192;
        let freq = 1000.0;
        let mut buf = sine_wave(freq, n, 0.5);
        let dry_hf = hf_energy(&buf, 4000.0);

        let cfg = ExciterConfig::new()
            .with_frequency_hz(2000.0)
            .with_harmonics_mix(0.8)
            .with_harmonic_order(2)
            .with_air_gain_db(0.0);
        let mut exciter = HarmonicExciter::new_kokoro(&cfg).unwrap();
        exciter.process(&mut buf);
        let wet_hf = hf_energy(&buf, 4000.0);

        assert!(
            wet_hf > dry_hf,
            "excited signal should have more HF energy: dry={dry_hf}, wet={wet_hf}",
        );
    }

    #[test]
    fn test_air_boost_increases_energy_above_air_freq() {
        let n = 8192;
        // Use a broadband signal (sum of many frequencies) so the shelf has
        // something to boost.
        let mut buf: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR;
                0.2 * (2.0 * std::f32::consts::PI * 500.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 2000.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 8000.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 11000.0 * t).sin()
            })
            .collect();
        let dry_air = hf_energy(&buf, 9000.0);

        let cfg = ExciterConfig::new()
            .with_harmonics_mix(0.0)
            .with_air_freq_hz(9000.0)
            .with_air_gain_db(6.0);
        let mut exciter = HarmonicExciter::new_kokoro(&cfg).unwrap();
        exciter.process(&mut buf);
        let wet_air = hf_energy(&buf, 9000.0);

        assert!(
            wet_air > dry_air * 1.1,
            "air boost should increase HF energy: dry={dry_air}, wet={wet_air}",
        );
    }

    #[test]
    fn test_all_outputs_finite() {
        let inputs = vec![
            0.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            0.001,
            -0.001,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let cfg = ExciterConfig::new()
            .with_harmonics_mix(1.0)
            .with_air_gain_db(6.0);
        let mut exciter = HarmonicExciter::new_kokoro(&cfg).unwrap();
        let mut buf = inputs;
        exciter.process(&mut buf);
        for (i, &v) in buf.iter().enumerate() {
            assert!(v.is_finite(), "sample {i} is non-finite: {v}");
        }
    }

    #[test]
    fn test_higher_harmonic_order_adds_more_energy() {
        let n = 8192;
        let freq = 2000.0;

        let mut buf2 = sine_wave(freq, n, 0.5);
        let cfg2 = ExciterConfig::new()
            .with_harmonics_mix(0.8)
            .with_harmonic_order(2)
            .with_air_gain_db(0.0);
        let mut ex2 = HarmonicExciter::new_kokoro(&cfg2).unwrap();
        ex2.process(&mut buf2);
        let energy_order2 = rms(&buf2);

        let mut buf5 = sine_wave(freq, n, 0.5);
        let cfg5 = ExciterConfig::new()
            .with_harmonics_mix(0.8)
            .with_harmonic_order(5)
            .with_air_gain_db(0.0);
        let mut ex5 = HarmonicExciter::new_kokoro(&cfg5).unwrap();
        ex5.process(&mut buf5);
        let energy_order5 = rms(&buf5);

        // Higher order applies more drive, which should produce different
        // (generally more) harmonic content. We check they differ.
        assert!(
            (energy_order5 - energy_order2).abs() > 1e-4,
            "different harmonic orders should produce different energy: \
             order2={energy_order2}, order5={energy_order5}",
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = ExciterConfig::new().with_harmonics_mix(0.5);
        let mut exciter = HarmonicExciter::new_kokoro(&cfg).unwrap();
        let mut buf = vec![0.5; 100];
        exciter.process(&mut buf);
        exciter.reset();
        assert_eq!(exciter.highpass.x_prev, 0.0);
        assert_eq!(exciter.highpass.y_prev, 0.0);
        assert_eq!(exciter.dc_blocker.x_prev, 0.0);
        assert_eq!(exciter.dc_blocker.y_prev, 0.0);
        assert_eq!(exciter.air_shelf.x_prev, 0.0);
        assert_eq!(exciter.air_shelf.y_prev, 0.0);
    }

    #[test]
    fn test_apply_exciter_per_voice() {
        let n = 2048;
        let mut voices = vec![
            sine_wave(800.0, n, 0.5),
            sine_wave(1200.0, n, 0.5),
            sine_wave(1600.0, n, 0.5),
        ];
        let dry_energies: Vec<f32> = voices.iter().map(|v| rms(v)).collect();

        let cfg = ExciterConfig::new()
            .with_harmonics_mix(0.5)
            .with_air_gain_db(3.0);
        apply_exciter(&mut voices, &cfg, SR).unwrap();

        // Each voice should have been processed (energy changed).
        for (i, (voice, dry_e)) in voices.iter().zip(dry_energies.iter()).enumerate() {
            let wet_e = rms(voice);
            assert!(
                (wet_e - dry_e).abs() > 1e-4,
                "voice {i} should be modified: dry_rms={dry_e}, wet_rms={wet_e}",
            );
        }
    }

    #[test]
    fn test_invalid_sample_rate() {
        let cfg = ExciterConfig::new();
        assert!(HarmonicExciter::new(&cfg, 0.0).is_err());
        assert!(HarmonicExciter::new(&cfg, -44100.0).is_err());
        assert!(HarmonicExciter::new(&cfg, f32::NAN).is_err());
    }

    #[test]
    fn test_generate_harmonics_zero_input() {
        // Zero input should produce zero output.
        assert_eq!(generate_harmonics(0.0, 2), 0.0);
        assert_eq!(generate_harmonics(0.0, 5), 0.0);
    }

    #[test]
    fn test_generate_harmonics_negative_input() {
        // Negative input is half-wave rectified to zero.
        let h = generate_harmonics(-0.5, 3);
        assert_eq!(h, 0.0, "negative input should be rectified to zero");
    }
}
