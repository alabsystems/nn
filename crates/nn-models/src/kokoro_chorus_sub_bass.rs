// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Sub-harmonic bass enhancement for Kokoro chorus depth and warmth.
//!
//! Synthesizes octave-below harmonics from the lowest frequency content of
//! the chorus mix, adding a richer, fuller foundation especially useful for
//! male voices and bass sections. Unlike the warmth module (body-band
//! saturation) or the exciter (HF harmonic generation), this module targets
//! content *below* the fundamental voice range and generates new energy one
//! octave lower.
//!
//! # Algorithm
//!
//! ```text
//! Input ──┬──────────────────────────────────────────── dry
//!         │
//!         └─> LP filter (crossover) ─> bass signal
//!                                         │
//!                                         v
//!                           Full-wave rectify (doubles freq)
//!                                         │
//!                                         v
//!                           LP filter (crossover / 2) ─> sub-octave
//!                                         │
//!                                         v
//!                              Soft-saturation (drive)
//!                                         │
//!                                         v
//!                               DC blocker (20 Hz HP)
//!                                         │
//!                                         v
//! Output = dry + amount * sub_signal ──────┘
//! ```
//!
//! 1. Low-pass the input at the crossover frequency to isolate bass content.
//! 2. Full-wave rectify the bass signal. Rectification doubles the
//!    fundamental frequency (e.g., 100 Hz becomes 200 Hz) but also creates
//!    a strong DC component and sub-harmonics via the resulting envelope.
//! 3. Low-pass the rectified signal at half the crossover frequency. This
//!    recovers the sub-octave content (the envelope of the rectified bass)
//!    while removing the doubled-frequency artifact.
//! 4. Apply soft saturation for harmonic richness and warmth.
//! 5. Remove residual DC via a 20 Hz highpass.
//! 6. Mix into the original signal: `out = input + amount * sub`.
//!
//! Multi-pass LP filtering (configurable `filter_order`) increases the
//! steepness of the crossover, producing a cleaner sub-octave signal.
//!
//! # References
//!
//! - Zolzer, U. "DAFX: Digital Audio Effects." 2nd ed., Wiley, 2011.
//!   Chapter 5: Nonlinear Processing (sub-harmonic synthesis).
//! - Classic hardware subharmonic synthesizers for sub-bass generation.
//! - Waves MaxxBass — psychoacoustic bass enhancement via harmonics.
//!
//! Part of #4582, #3351.

use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the sub-harmonic bass enhancer.
///
/// Constructed via [`SubBassConfig::new`] (required for cross-crate use
/// due to `#[non_exhaustive]`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct SubBassConfig {
    /// Crossover frequency (Hz) separating bass from the rest of the signal.
    /// Content below this frequency is used to generate sub-harmonics.
    /// Range: 40.0 - 250.0. Default: 120.0.
    pub frequency_hz: f32,

    /// Sub-bass mix amount: 0.0 = bypass, 1.0 = full sub signal.
    /// Default: 0.3.
    pub amount: f32,

    /// Blend between the original bass and the synthesized sub-octave.
    /// 0.0 = pure sub-octave only, 1.0 = equal blend.
    /// Range: 0.0 - 1.0. Default: 0.5.
    pub octave_mix: f32,

    /// Saturation drive applied to the sub signal for harmonic richness.
    /// 0.0 = clean, 1.0 = heavy saturation.
    /// Range: 0.0 - 1.0. Default: 0.2.
    pub drive: f32,

    /// Number of cascaded LP filter passes for crossover steepness.
    /// Each pass adds ~6 dB/octave of rolloff.
    /// Range: 1 - 8. Default: 4.
    pub filter_order: u32,
}

impl Default for SubBassConfig {
    fn default() -> Self {
        Self {
            frequency_hz: 120.0,
            amount: 0.3,
            octave_mix: 0.5,
            drive: 0.2,
            filter_order: 4,
        }
    }
}

impl SubBassConfig {
    /// Create a new sub-bass config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the crossover frequency in Hz.
    #[must_use]
    pub fn with_frequency_hz(mut self, hz: f32) -> Self {
        self.frequency_hz = hz;
        self
    }

    /// Set the sub-bass mix amount (0.0 = bypass, 1.0 = full).
    #[must_use]
    pub fn with_amount(mut self, amount: f32) -> Self {
        self.amount = amount;
        self
    }

    /// Set the octave blend (0.0 = sub-only, 1.0 = equal blend).
    #[must_use]
    pub fn with_octave_mix(mut self, mix: f32) -> Self {
        self.octave_mix = mix;
        self
    }

    /// Set the saturation drive (0.0 = clean, 1.0 = heavy).
    #[must_use]
    pub fn with_drive(mut self, drive: f32) -> Self {
        self.drive = drive;
        self
    }

    /// Set the LP filter order (cascaded passes, 1-8).
    #[must_use]
    pub fn with_filter_order(mut self, order: u32) -> Self {
        self.filter_order = order;
        self
    }

    /// Validate all parameters are within acceptable ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.frequency_hz.is_finite() || self.frequency_hz < 40.0 || self.frequency_hz > 250.0 {
            return Err(KokoroError::InvalidConfig {
                field: "frequency_hz",
                reason: format!(
                    "frequency_hz = {}: must be finite and in [40, 250]",
                    self.frequency_hz,
                ),
            });
        }
        if !self.amount.is_finite() || self.amount < 0.0 || self.amount > 1.0 {
            return Err(KokoroError::InvalidConfig {
                field: "amount",
                reason: format!("amount = {}: must be finite and in [0.0, 1.0]", self.amount),
            });
        }
        if !self.octave_mix.is_finite() || self.octave_mix < 0.0 || self.octave_mix > 1.0 {
            return Err(KokoroError::InvalidConfig {
                field: "octave_mix",
                reason: format!(
                    "octave_mix = {}: must be finite and in [0.0, 1.0]",
                    self.octave_mix,
                ),
            });
        }
        if !self.drive.is_finite() || self.drive < 0.0 || self.drive > 1.0 {
            return Err(KokoroError::InvalidConfig {
                field: "drive",
                reason: format!("drive = {}: must be finite and in [0.0, 1.0]", self.drive),
            });
        }
        if self.filter_order < 1 || self.filter_order > 8 {
            return Err(KokoroError::InvalidConfig {
                field: "filter_order",
                reason: format!("filter_order = {}: must be in [1, 8]", self.filter_order),
            });
        }
        Ok(())
    }

    // --- Presets ---------------------------------------------------------------

    /// Subtle warmth — barely perceptible low-end thickening.
    /// Good for female voices and bright content that needs a touch of body.
    #[must_use]
    pub fn subtle_warmth() -> Self {
        Self {
            frequency_hz: 100.0,
            amount: 0.15,
            octave_mix: 0.3,
            drive: 0.1,
            filter_order: 4,
        }
    }

    /// Deep bass — pronounced sub-bass for rich male voice foundation.
    /// Adds noticeable low-end weight without muddiness.
    #[must_use]
    pub fn deep_bass() -> Self {
        Self {
            frequency_hz: 140.0,
            amount: 0.45,
            octave_mix: 0.6,
            drive: 0.25,
            filter_order: 4,
        }
    }

    /// Sub rumble — heavy, felt-not-heard sub-bass. Cinematic depth.
    /// Best used sparingly; can overwhelm on small speakers.
    #[must_use]
    pub fn sub_rumble() -> Self {
        Self {
            frequency_hz: 80.0,
            amount: 0.6,
            octave_mix: 0.2,
            drive: 0.4,
            filter_order: 6,
        }
    }

    /// Vocal body — tuned for speech, adds chest resonance without boom.
    /// Balances the sub-octave with the original bass for natural fullness.
    #[must_use]
    pub fn vocal_body() -> Self {
        Self {
            frequency_hz: 120.0,
            amount: 0.3,
            octave_mix: 0.5,
            drive: 0.15,
            filter_order: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Single-pole lowpass filter
// ---------------------------------------------------------------------------

/// Single-pole IIR lowpass: y[n] = b*x[n] + a*y[n-1].
///
/// Used in cascaded chains for the crossover and sub-octave extraction.
#[derive(Debug, Clone)]
struct OnePoleLP {
    a: f32,
    b: f32,
    z1: f32,
}

impl OnePoleLP {
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let w = (-2.0 * std::f32::consts::PI * cutoff_hz / sample_rate).exp();
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
// Single-pole highpass (DC blocker)
// ---------------------------------------------------------------------------

/// Single-pole highpass for DC blocking on the sub-bass path.
#[derive(Debug, Clone)]
struct OnePoleHP {
    coeff: f32,
    x_prev: f32,
    y_prev: f32,
}

impl OnePoleHP {
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
        let dt = 1.0 / sample_rate;
        Self {
            coeff: rc / (rc + dt),
            x_prev: 0.0,
            y_prev: 0.0,
        }
    }

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

    fn reset(&mut self) {
        self.x_prev = 0.0;
        self.y_prev = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Soft saturation
// ---------------------------------------------------------------------------

/// Soft-clip waveshaper for the sub-bass signal.
///
/// Uses tanh saturation with configurable drive. Produces warm, even-harmonic
/// distortion that thickens the sub-octave without harsh clipping artifacts.
#[inline]
fn soft_saturate(x: f32, drive: f32) -> f32 {
    if drive < 1e-6 {
        return x;
    }
    let gain = 1.0 + drive * 4.0;
    (x * gain).tanh()
}

// ---------------------------------------------------------------------------
// SubBassEnhancer
// ---------------------------------------------------------------------------

/// Stateful sub-harmonic bass enhancer.
///
/// Holds filter state for the crossover lowpass cascade, the sub-octave
/// extraction lowpass cascade, and a DC blocker.
#[derive(Debug, Clone)]
pub struct SubBassEnhancer {
    config: SubBassConfig,
    /// Cascaded LP filters for crossover isolation (bass extraction).
    crossover_lp: Vec<OnePoleLP>,
    /// Cascaded LP filters for sub-octave extraction (half crossover freq).
    sub_octave_lp: Vec<OnePoleLP>,
    /// DC blocker (20 Hz highpass) to remove rectification DC offset.
    dc_blocker: OnePoleHP,
}

impl SubBassEnhancer {
    /// Create a new sub-bass enhancer from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range,
    /// or if `sample_rate` is not finite and positive.
    pub fn new(config: SubBassConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("sample_rate = {sample_rate}: must be finite and positive"),
            });
        }

        let order = config.filter_order as usize;

        // Crossover LP at the configured frequency.
        let crossover_lp = (0..order)
            .map(|_| OnePoleLP::new(config.frequency_hz, sample_rate))
            .collect();

        // Sub-octave LP at half the crossover (extracts the envelope).
        let sub_freq = (config.frequency_hz / 2.0).max(15.0);
        let sub_octave_lp = (0..order)
            .map(|_| OnePoleLP::new(sub_freq, sample_rate))
            .collect();

        // DC blocker at 20 Hz.
        let dc_blocker = OnePoleHP::new(20.0, sample_rate);

        Ok(Self {
            config,
            crossover_lp,
            sub_octave_lp,
            dc_blocker,
        })
    }

    /// Create an enhancer using Kokoro's default 24 kHz sample rate.
    pub fn new_kokoro(config: SubBassConfig) -> Result<Self, KokoroError> {
        Self::new(config, KOKORO_SAMPLE_RATE as f32)
    }

    /// Process a stereo bus (left and right channels) in-place.
    ///
    /// Both channels are processed through the same filter chain for a
    /// mono-compatible sub-bass signal (sub-bass is typically centered).
    /// The left channel drives the sub-bass generation; the same sub signal
    /// is mixed into both channels for phase coherence.
    ///
    /// Fast path: returns immediately when `amount == 0.0`.
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        if self.config.amount == 0.0 {
            return;
        }

        let len = left.len().min(right.len());
        let amount = self.config.amount;
        let drive = self.config.drive;
        let octave_mix = self.config.octave_mix;

        for i in 0..len {
            // Use mid signal (mono sum) for sub generation to keep phase coherent.
            let mid = (left[i] + right[i]) * 0.5;
            if !mid.is_finite() {
                // Sanitize non-finite inputs so the output stays finite rather
                // than propagating NaN/Inf through the channels.
                if !left[i].is_finite() {
                    left[i] = 0.0;
                }
                if !right[i].is_finite() {
                    right[i] = 0.0;
                }
                continue;
            }

            // --- Crossover: isolate bass via cascaded LP ---
            let mut bass = mid;
            for lp in &mut self.crossover_lp {
                bass = lp.process(bass);
            }

            // --- Full-wave rectification ---
            // Rectifying doubles the frequency of the fundamental but creates
            // a strong DC + envelope component at the original sub-frequency.
            let rectified = bass.abs();

            // --- Sub-octave extraction via LP at half crossover ---
            let mut sub = rectified;
            for lp in &mut self.sub_octave_lp {
                sub = lp.process(sub);
            }

            // --- Blend original bass envelope with sub-octave ---
            // octave_mix=0 gives pure sub-octave, octave_mix=1 blends bass in.
            let blended = sub * (1.0 - octave_mix) + bass * octave_mix;

            // --- Soft saturation for harmonic richness ---
            let saturated = soft_saturate(blended, drive);

            // --- DC blocker ---
            let sub_signal = self.dc_blocker.process(saturated);

            // --- Mix into both channels ---
            left[i] += amount * sub_signal;
            right[i] += amount * sub_signal;

            // Final NaN/Inf guard.
            if !left[i].is_finite() {
                left[i] = 0.0;
            }
            if !right[i].is_finite() {
                right[i] = 0.0;
            }
        }
    }

    /// Reset all internal filter state (call between unrelated audio segments).
    pub fn reset(&mut self) {
        for lp in &mut self.crossover_lp {
            lp.reset();
        }
        for lp in &mut self.sub_octave_lp {
            lp.reset();
        }
        self.dc_blocker.reset();
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &SubBassConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = KOKORO_SAMPLE_RATE as f32;

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
        SubBassConfig::new()
            .validate()
            .expect("default config should be valid");
    }

    #[test]
    fn test_config_builder_roundtrip() {
        let cfg = SubBassConfig::new()
            .with_frequency_hz(100.0)
            .with_amount(0.5)
            .with_octave_mix(0.7)
            .with_drive(0.3)
            .with_filter_order(6);
        cfg.validate().expect("builder config should be valid");
        assert_eq!(cfg.frequency_hz, 100.0);
        assert_eq!(cfg.amount, 0.5);
        assert_eq!(cfg.octave_mix, 0.7);
        assert_eq!(cfg.drive, 0.3);
        assert_eq!(cfg.filter_order, 6);
    }

    #[test]
    fn test_config_invalid_frequency_hz() {
        assert!(SubBassConfig::new()
            .with_frequency_hz(20.0)
            .validate()
            .is_err());
        assert!(SubBassConfig::new()
            .with_frequency_hz(300.0)
            .validate()
            .is_err());
        assert!(SubBassConfig::new()
            .with_frequency_hz(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_amount() {
        assert!(SubBassConfig::new().with_amount(-0.1).validate().is_err());
        assert!(SubBassConfig::new().with_amount(1.5).validate().is_err());
        assert!(SubBassConfig::new()
            .with_amount(f32::INFINITY)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_octave_mix() {
        assert!(SubBassConfig::new()
            .with_octave_mix(-0.1)
            .validate()
            .is_err());
        assert!(SubBassConfig::new()
            .with_octave_mix(1.1)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_drive() {
        assert!(SubBassConfig::new().with_drive(-0.1).validate().is_err());
        assert!(SubBassConfig::new().with_drive(1.1).validate().is_err());
        assert!(SubBassConfig::new()
            .with_drive(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_filter_order() {
        assert!(SubBassConfig::new()
            .with_filter_order(0)
            .validate()
            .is_err());
        assert!(SubBassConfig::new()
            .with_filter_order(9)
            .validate()
            .is_err());
    }

    #[test]
    fn test_presets_valid() {
        SubBassConfig::subtle_warmth()
            .validate()
            .expect("subtle_warmth valid");
        SubBassConfig::deep_bass()
            .validate()
            .expect("deep_bass valid");
        SubBassConfig::sub_rumble()
            .validate()
            .expect("sub_rumble valid");
        SubBassConfig::vocal_body()
            .validate()
            .expect("vocal_body valid");
    }

    // --- Processor behavior ---

    #[test]
    fn test_zero_amount_is_noop() {
        let mut left = sine_wave(100.0, 2048, 0.5);
        let mut right = sine_wave(100.0, 2048, 0.5);
        let orig_left = left.clone();
        let orig_right = right.clone();
        let cfg = SubBassConfig::new().with_amount(0.0);
        let mut proc = SubBassEnhancer::new_kokoro(cfg).expect("valid");
        proc.process(&mut left, &mut right);
        assert_eq!(left, orig_left, "zero amount should not modify left");
        assert_eq!(right, orig_right, "zero amount should not modify right");
    }

    #[test]
    fn test_sub_bass_adds_energy() {
        let n = 8192;
        let mut left = sine_wave(100.0, n, 0.5);
        let mut right = sine_wave(100.0, n, 0.5);
        let dry_rms = rms(&left);

        let cfg = SubBassConfig::new()
            .with_frequency_hz(150.0)
            .with_amount(0.8)
            .with_drive(0.3);
        let mut proc = SubBassEnhancer::new_kokoro(cfg).expect("valid");
        proc.process(&mut left, &mut right);
        let wet_rms = rms(&left);

        assert!(
            wet_rms > dry_rms,
            "sub-bass should add energy: dry={dry_rms}, wet={wet_rms}",
        );
    }

    #[test]
    fn test_stereo_coherence() {
        // Both channels should receive the same sub-bass addition.
        let n = 4096;
        let mut left = sine_wave(80.0, n, 0.4);
        let mut right = sine_wave(80.0, n, 0.4);

        let cfg = SubBassConfig::new().with_amount(0.5);
        let mut proc = SubBassEnhancer::new_kokoro(cfg).expect("valid");
        proc.process(&mut left, &mut right);

        // Since both channels started identical, they should remain identical
        // (sub-bass is derived from mono mid signal).
        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(
                (l - r).abs() < 1e-6,
                "sample {i}: left={l}, right={r} should be equal",
            );
        }
    }

    #[test]
    fn test_all_outputs_finite() {
        let mut left = vec![
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
        let mut right = vec![
            0.5,
            -0.5,
            0.0,
            -1.0,
            1.0,
            -0.001,
            0.001,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ];
        let cfg = SubBassConfig::new().with_amount(1.0).with_drive(1.0);
        let mut proc = SubBassEnhancer::new_kokoro(cfg).expect("valid");
        proc.process(&mut left, &mut right);
        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(l.is_finite(), "left sample {i} non-finite: {l}");
            assert!(r.is_finite(), "right sample {i} non-finite: {r}");
        }
    }

    #[test]
    fn test_drive_affects_output() {
        let n = 4096;
        let mut left_clean = sine_wave(100.0, n, 0.5);
        let mut right_clean = sine_wave(100.0, n, 0.5);
        let cfg_clean = SubBassConfig::new().with_amount(0.5).with_drive(0.0);
        let mut proc_clean = SubBassEnhancer::new_kokoro(cfg_clean).expect("valid");
        proc_clean.process(&mut left_clean, &mut right_clean);
        let rms_clean = rms(&left_clean);

        let mut left_driven = sine_wave(100.0, n, 0.5);
        let mut right_driven = sine_wave(100.0, n, 0.5);
        let cfg_driven = SubBassConfig::new().with_amount(0.5).with_drive(1.0);
        let mut proc_driven = SubBassEnhancer::new_kokoro(cfg_driven).expect("valid");
        proc_driven.process(&mut left_driven, &mut right_driven);
        let rms_driven = rms(&left_driven);

        assert!(
            (rms_driven - rms_clean).abs() > 1e-4,
            "drive should change character: clean={rms_clean}, driven={rms_driven}",
        );
    }

    #[test]
    fn test_soft_saturate_zero_drive_is_identity() {
        assert_eq!(soft_saturate(0.5, 0.0), 0.5);
        assert_eq!(soft_saturate(-0.3, 0.0), -0.3);
        assert_eq!(soft_saturate(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_soft_saturate_bounded() {
        // tanh output is in (-1, 1) mathematically; in f32 it saturates to
        // exactly +/-1.0 for large inputs (e.g. tanh(50.0) rounds to 1.0), so
        // the bound is |sat| <= 1.0.
        for &x in &[0.5, 1.0, 2.0, 10.0, -0.5, -1.0, -2.0, -10.0] {
            let sat = soft_saturate(x, 1.0);
            assert!(
                sat.abs() <= 1.0,
                "soft_saturate({x}, 1.0) = {sat}, should be in [-1, 1]",
            );
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = SubBassConfig::new().with_amount(0.5);
        let mut proc = SubBassEnhancer::new_kokoro(cfg).expect("valid");
        let mut left = vec![0.5; 100];
        let mut right = vec![0.5; 100];
        proc.process(&mut left, &mut right);
        proc.reset();
        for lp in &proc.crossover_lp {
            assert_eq!(lp.z1, 0.0);
        }
        for lp in &proc.sub_octave_lp {
            assert_eq!(lp.z1, 0.0);
        }
        assert_eq!(proc.dc_blocker.x_prev, 0.0);
        assert_eq!(proc.dc_blocker.y_prev, 0.0);
    }

    #[test]
    fn test_invalid_sample_rate() {
        let cfg = SubBassConfig::new();
        assert!(SubBassEnhancer::new(cfg, 0.0).is_err());
        assert!(SubBassEnhancer::new(cfg, -44100.0).is_err());
        assert!(SubBassEnhancer::new(cfg, f32::NAN).is_err());
    }

    #[test]
    fn test_empty_buffers() {
        let cfg = SubBassConfig::new();
        let mut proc = SubBassEnhancer::new_kokoro(cfg).expect("valid");
        let mut left: Vec<f32> = vec![];
        let mut right: Vec<f32> = vec![];
        proc.process(&mut left, &mut right);
        assert!(left.is_empty());
        assert!(right.is_empty());
    }

    #[test]
    fn test_mismatched_lengths_uses_minimum() {
        let cfg = SubBassConfig::new().with_amount(0.5);
        let mut proc = SubBassEnhancer::new_kokoro(cfg).expect("valid");
        let mut left = vec![0.3; 100];
        let mut right = vec![0.3; 50];
        proc.process(&mut left, &mut right);
        // Should not panic; processes min(100, 50) = 50 samples.
        // Samples 50..100 in left should be unchanged.
        for &v in &left[50..] {
            assert_eq!(v, 0.3, "samples beyond min length should be untouched");
        }
    }
}
