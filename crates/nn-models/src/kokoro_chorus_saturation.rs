// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Harmonic warmth and saturation for Kokoro chorus bus processing.
//!
//! Professional chorus production adds subtle harmonic distortion to bind
//! voices together and add warmth — the same character that analog mixing
//! consoles, tape machines, and tube preamps impart. This module provides
//! four saturation modes modeling different analog characteristics.
//!
//! # Architecture
//!
//! ```text
//! Input --> 2x upsample (linear interp) --> waveshaper(mode) --> 1-pole LP
//!           decimation --> wet/dry mix --> output gain compensation
//! ```
//!
//! Simple 2x oversampling reduces aliasing artifacts from the nonlinear
//! waveshaping stage. The decimation filter is a one-pole lowpass at
//! Nyquist/2 to remove mirror images introduced by the nonlinearity.
//!
//! # Saturation modes
//!
//! - **Tape** — asymmetric soft clipping modeling magnetic tape saturation.
//!   Uses `tanh` with a positive/negative asymmetry factor, producing
//!   predominantly even harmonics. The classic "warm analog" sound.
//! - **Tube** — even-harmonic-rich distortion with `x / (1 + |x|)` curve.
//!   Models the transfer characteristic of a tube stage operated below
//!   clipping. Smooth, musical compression.
//! - **Console** — symmetric `tanh` saturation with very mild drive.
//!   Models the subtle odd-harmonic coloring of transformer-coupled
//!   mixing consoles. Barely audible, but adds cohesion.
//! - **Warm** — gentle blend of even and odd harmonics via a polynomial
//!   waveshaper. The lightest touch — adds subtle color without
//!   perceptible distortion.
//!
//! # References
//!
//! - Välimäki, V. & Reiss, J. D. "All About Audio Equalization:
//!   Solutions and Frontiers." Applied Sciences, 6(5), 2016.
//! - Zölzer, U. "DAFX: Digital Audio Effects." 2nd ed., Wiley, 2011.
//!   Chapter 5: Nonlinear Processing.
//! - Smith, J. O. "Physical Audio Signal Processing."
//!   <https://ccrma.stanford.edu/~jos/pasp/> — oversampling for
//!   waveshaping.
//!
//! Part of #4264, #3351.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// dB <-> linear conversions
// ---------------------------------------------------------------------------

/// Convert decibels to linear amplitude.
///
/// Returns 0.0 for -infinity dB. Checks finiteness of input.
#[inline]
#[must_use]
pub fn db_to_linear(db: f32) -> f32 {
    if !db.is_finite() {
        return 0.0;
    }
    // 10^(db/20) = e^(db * ln(10)/20)
    let lin = 10.0_f32.powf(db / 20.0);
    if !lin.is_finite() {
        return 0.0;
    }
    lin
}

/// Convert linear amplitude to decibels.
///
/// Returns `-f32::INFINITY` for zero or negative input. Checks finiteness.
#[inline]
#[must_use]
pub fn linear_to_db(lin: f32) -> f32 {
    if !lin.is_finite() || lin <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let db = 20.0 * lin.log10();
    if !db.is_finite() {
        return f32::NEG_INFINITY;
    }
    db
}

// ---------------------------------------------------------------------------
// Saturation mode
// ---------------------------------------------------------------------------

/// Type of saturation curve applied to the audio signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaturationMode {
    /// Asymmetric soft clipping modeling magnetic tape (tanh-based with
    /// positive/negative asymmetry). Predominantly even harmonics.
    Tape,
    /// Even-harmonic-rich distortion using `x / (1 + |x|)` waveshaper.
    /// Models a tube stage operated below clipping.
    Tube,
    /// Symmetric tanh saturation with very mild drive. Models subtle
    /// odd-harmonic coloring of transformer-coupled consoles.
    Console,
    /// Gentle blend of even and odd harmonics via polynomial waveshaper.
    /// Lightest touch — adds color without perceptible distortion.
    Warm,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the saturation processor.
///
/// Constructed via [`SaturationConfig::new`] (required for cross-crate use
/// due to `#[non_exhaustive]`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct SaturationConfig {
    /// Saturation amount: 0.0 = completely clean, 1.0 = heavy distortion.
    /// Default: 0.2 (subtle warmth).
    pub drive: f32,
    /// Wet/dry mix: 0.0 = all dry (bypass), 1.0 = all wet (fully saturated).
    /// Default: 0.5.
    pub mix: f32,
    /// Type of saturation curve.
    /// Default: [`SaturationMode::Tape`].
    pub mode: SaturationMode,
    /// Output gain compensation in dB (saturation tends to increase level).
    /// Default: -1.0 dB.
    pub output_gain_db: f32,
}

impl Default for SaturationConfig {
    fn default() -> Self {
        Self {
            drive: 0.2,
            mix: 0.5,
            mode: SaturationMode::Tape,
            output_gain_db: -1.0,
        }
    }
}

impl SaturationConfig {
    /// Create a new saturation config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the drive amount (0.0 = clean, 1.0 = heavy).
    #[must_use]
    pub fn with_drive(mut self, drive: f32) -> Self {
        self.drive = drive;
        self
    }

    /// Set the wet/dry mix (0.0 = bypass, 1.0 = fully saturated).
    #[must_use]
    pub fn with_mix(mut self, mix: f32) -> Self {
        self.mix = mix;
        self
    }

    /// Set the saturation mode.
    #[must_use]
    pub fn with_mode(mut self, mode: SaturationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the output gain compensation in dB.
    #[must_use]
    pub fn with_output_gain_db(mut self, db: f32) -> Self {
        self.output_gain_db = db;
        self
    }

    /// Validate all parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.drive.is_finite() || self.drive < 0.0 || self.drive > 1.0 {
            return Err(KokoroError::InvalidConfig {
                field: "drive",
                reason: format!("drive = {}: must be finite and in [0.0, 1.0]", self.drive),
            });
        }
        if !self.mix.is_finite() || self.mix < 0.0 || self.mix > 1.0 {
            return Err(KokoroError::InvalidConfig {
                field: "mix",
                reason: format!("mix = {}: must be finite and in [0.0, 1.0]", self.mix),
            });
        }
        if !self.output_gain_db.is_finite()
            || self.output_gain_db < -24.0
            || self.output_gain_db > 24.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "output_gain_db",
                reason: format!(
                    "output_gain_db = {}: must be finite and in [-24, 24]",
                    self.output_gain_db,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// One-pole lowpass filter for decimation
// ---------------------------------------------------------------------------

/// Single-pole IIR lowpass used for oversampled decimation.
///
/// H(z) = b / (1 - a * z^-1) where a = e^(-2*pi*fc/fs).
#[derive(Debug, Clone)]
struct OnePoleLP {
    a: f32,
    b: f32,
    z1: f32,
}

impl OnePoleLP {
    /// Create a new one-pole lowpass at `cutoff_hz` given `sample_rate`.
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let w = (-2.0 * std::f32::consts::PI * cutoff_hz / sample_rate).exp();
        Self {
            a: w,
            b: 1.0 - w,
            z1: 0.0,
        }
    }

    /// Process one sample through the filter.
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

    /// Reset filter state.
    fn reset(&mut self) {
        self.z1 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Waveshaping functions
// ---------------------------------------------------------------------------

/// Tape saturation: asymmetric tanh with even-harmonic bias.
///
/// Positive half driven slightly harder than negative, producing
/// asymmetry that generates even harmonics (2nd, 4th, ...).
#[inline]
fn waveshape_tape(x: f32, drive: f32) -> f32 {
    // Drive maps [0,1] -> [1, 6] gain into the tanh
    let gain = 1.0 + drive * 5.0;
    let pos_gain = gain * 1.1; // slight asymmetry: positive driven harder
    let neg_gain = gain * 0.9;
    if x >= 0.0 {
        (x * pos_gain).tanh()
    } else {
        (x * neg_gain).tanh()
    }
}

/// Tube saturation: x / (1 + |x|) waveshaper — smooth, even-harmonic.
///
/// This sigmoid-like curve provides softer clipping than tanh and has
/// a broader "linear region" before saturation onset.
#[inline]
fn waveshape_tube(x: f32, drive: f32) -> f32 {
    let gain = 1.0 + drive * 4.0;
    let driven = x * gain;
    driven / (1.0 + driven.abs())
}

/// Console saturation: symmetric tanh with mild drive — odd harmonics.
///
/// Very subtle coloring. The drive range is intentionally compressed
/// so even drive=1.0 stays mild.
#[inline]
fn waveshape_console(x: f32, drive: f32) -> f32 {
    let gain = 1.0 + drive * 2.0; // gentler range than tape
    (x * gain).tanh()
}

/// Warm saturation: polynomial waveshaper blending even + odd harmonics.
///
/// `f(x) = x + c * (x^2 - x^3)` where `c` is derived from drive.
/// The x^2 term produces even harmonics; x^3 produces odd.
/// Lightest touch of all modes.
#[inline]
fn waveshape_warm(x: f32, drive: f32) -> f32 {
    let c = drive * 0.3; // very gentle coefficient
    let x2 = x * x;
    let x3 = x2 * x;
    x + c * (x2 - x3)
}

/// Select and apply the waveshaping function for the given mode.
#[inline]
fn apply_waveshaper(x: f32, drive: f32, mode: SaturationMode) -> f32 {
    match mode {
        SaturationMode::Tape => waveshape_tape(x, drive),
        SaturationMode::Tube => waveshape_tube(x, drive),
        SaturationMode::Console => waveshape_console(x, drive),
        SaturationMode::Warm => waveshape_warm(x, drive),
    }
}

// ---------------------------------------------------------------------------
// Saturation processor
// ---------------------------------------------------------------------------

/// Stateful saturation processor with 2x oversampling.
///
/// Holds the one-pole lowpass filter state used for decimation after
/// waveshaping at the oversampled rate.
#[derive(Debug, Clone)]
pub struct SaturationProcessor {
    config: SaturationConfig,
    /// Output gain as linear multiplier (derived from `output_gain_db`).
    output_gain_linear: f32,
    /// One-pole lowpass for decimation after 2x oversampled waveshaping.
    decimation_lp: OnePoleLP,
}

impl SaturationProcessor {
    /// Create a new saturation processor from the given config.
    ///
    /// `sample_rate` is the base (non-oversampled) sample rate.
    pub fn new(config: SaturationConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("sample_rate = {sample_rate}: must be finite and positive"),
            });
        }

        let output_gain_linear = db_to_linear(config.output_gain_db);

        // Decimation lowpass at Nyquist/2 of the *oversampled* rate.
        // Oversampled rate = 2 * sample_rate. Nyquist = sample_rate.
        // Cut at sample_rate * 0.45 to allow gentle rolloff.
        let oversampled_rate = sample_rate * 2.0;
        let cutoff = sample_rate * 0.45;
        let decimation_lp = OnePoleLP::new(cutoff, oversampled_rate);

        Ok(Self {
            config,
            output_gain_linear,
            decimation_lp,
        })
    }

    /// Create a processor using Kokoro's default 24 kHz sample rate.
    pub fn new_kokoro(config: SaturationConfig) -> Result<Self, KokoroError> {
        use crate::kokoro_tts::KOKORO_SAMPLE_RATE;
        Self::new(config, KOKORO_SAMPLE_RATE as f32)
    }

    /// Reset internal filter state (call between unrelated audio segments).
    pub fn reset(&mut self) {
        self.decimation_lp.reset();
    }

    /// Process an audio buffer in-place through the saturation stage.
    ///
    /// Fast path: returns immediately when `drive == 0.0` (no processing).
    pub fn process(&mut self, buf: &mut [f32]) {
        // Fast path: drive=0 means no saturation.
        if self.config.drive == 0.0 {
            return;
        }

        let drive = self.config.drive;
        let mix = self.config.mix;
        let mode = self.config.mode;
        let gain = self.output_gain_linear;

        for sample in buf.iter_mut() {
            // Guard non-finite input.
            if !sample.is_finite() {
                *sample = 0.0;
                continue;
            }

            let dry = *sample;

            // --- 2x oversample: linear interpolation upsample ---
            // Produce two oversampled values: midpoint (interpolated) and
            // the original sample. We process both through the waveshaper
            // and decimation filter, keeping only every other output.

            // First oversampled sample: midpoint between previous output
            // of the decimation filter and current input.
            let mid = (self.decimation_lp.z1 + dry) * 0.5;
            let shaped_mid = apply_waveshaper(mid, drive, mode);
            let _ = self.decimation_lp.process(shaped_mid);

            // Second oversampled sample: the actual input.
            let shaped = apply_waveshaper(dry, drive, mode);
            let decimated = self.decimation_lp.process(shaped);

            // --- Wet/dry mix ---
            let wet = decimated;
            let mixed = dry * (1.0 - mix) + wet * mix;

            // --- Output gain compensation ---
            let out = mixed * gain;

            // Final NaN/Inf guard.
            *sample = if out.is_finite() { out } else { 0.0 };
        }
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &SaturationConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Standalone processing function
// ---------------------------------------------------------------------------

/// Apply saturation to a buffer using default Kokoro sample rate (24 kHz).
///
/// Convenience wrapper that creates a temporary [`SaturationProcessor`],
/// processes the buffer, and returns. For processing multiple buffers
/// with shared state (e.g., streaming), prefer creating a
/// [`SaturationProcessor`] and calling [`SaturationProcessor::process`].
pub fn process_saturation(buf: &mut [f32], config: &SaturationConfig) -> Result<(), KokoroError> {
    let mut proc = SaturationProcessor::new_kokoro(*config)?;
    proc.process(buf);
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_to_linear_unity() {
        let lin = db_to_linear(0.0);
        assert!(
            (lin - 1.0).abs() < 1e-6,
            "0 dB should be unity gain, got {lin}"
        );
    }

    #[test]
    fn test_db_to_linear_minus6() {
        let lin = db_to_linear(-6.0206);
        assert!((lin - 0.5).abs() < 1e-3, "-6 dB should be ~0.5, got {lin}");
    }

    #[test]
    fn test_db_to_linear_nan() {
        assert_eq!(db_to_linear(f32::NAN), 0.0);
        assert_eq!(db_to_linear(f32::INFINITY), 0.0);
        assert_eq!(db_to_linear(f32::NEG_INFINITY), 0.0);
    }

    #[test]
    fn test_linear_to_db_unity() {
        let db = linear_to_db(1.0);
        assert!(db.abs() < 1e-6, "unity should be 0 dB, got {db}");
    }

    #[test]
    fn test_linear_to_db_zero() {
        assert_eq!(linear_to_db(0.0), f32::NEG_INFINITY);
    }

    #[test]
    fn test_linear_to_db_negative() {
        assert_eq!(linear_to_db(-1.0), f32::NEG_INFINITY);
    }

    #[test]
    fn test_linear_to_db_nan() {
        assert_eq!(linear_to_db(f32::NAN), f32::NEG_INFINITY);
    }

    #[test]
    fn test_roundtrip_db_linear() {
        for db_val in [-20.0, -12.0, -6.0, -3.0, 0.0, 3.0, 6.0] {
            let roundtrip = linear_to_db(db_to_linear(db_val));
            assert!(
                (roundtrip - db_val).abs() < 0.01,
                "roundtrip failed for {db_val}: got {roundtrip}",
            );
        }
    }

    #[test]
    fn test_config_default_valid() {
        SaturationConfig::new()
            .validate()
            .expect("default config should be valid");
    }

    #[test]
    fn test_config_builder() {
        let cfg = SaturationConfig::new()
            .with_drive(0.5)
            .with_mix(0.8)
            .with_mode(SaturationMode::Tube)
            .with_output_gain_db(-2.0);
        cfg.validate().expect("builder config should be valid");
        assert_eq!(cfg.drive, 0.5);
        assert_eq!(cfg.mix, 0.8);
        assert_eq!(cfg.mode, SaturationMode::Tube);
        assert_eq!(cfg.output_gain_db, -2.0);
    }

    #[test]
    fn test_config_invalid_drive() {
        let r = SaturationConfig::new().with_drive(1.5).validate();
        assert!(r.is_err(), "drive > 1.0 should be invalid");
        let r = SaturationConfig::new().with_drive(-0.1).validate();
        assert!(r.is_err(), "drive < 0.0 should be invalid");
        let r = SaturationConfig::new().with_drive(f32::NAN).validate();
        assert!(r.is_err(), "NaN drive should be invalid");
    }

    #[test]
    fn test_config_invalid_mix() {
        let r = SaturationConfig::new().with_mix(1.1).validate();
        assert!(r.is_err(), "mix > 1.0 should be invalid");
        let r = SaturationConfig::new().with_mix(f32::INFINITY).validate();
        assert!(r.is_err(), "Inf mix should be invalid");
    }

    #[test]
    fn test_config_invalid_output_gain() {
        let r = SaturationConfig::new().with_output_gain_db(50.0).validate();
        assert!(r.is_err(), "output_gain_db > 24 should be invalid");
    }

    #[test]
    fn test_zero_drive_is_noop() {
        let mut buf = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let original = buf.clone();
        let cfg = SaturationConfig::new().with_drive(0.0);
        let mut proc = SaturationProcessor::new_kokoro(cfg).expect("valid config");
        proc.process(&mut buf);
        assert_eq!(buf, original, "drive=0 should not modify signal");
    }

    #[test]
    fn test_tape_adds_harmonics() {
        // A pure sine processed through tape saturation should have
        // more energy (higher peaks or different shape) than the input.
        let n = 1024;
        let freq = 440.0;
        let sr = 24000.0;
        let mut buf: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin() * 0.5)
            .collect();
        let dry_energy: f32 = buf.iter().map(|x| x * x).sum();

        let cfg = SaturationConfig::new()
            .with_drive(0.5)
            .with_mix(1.0)
            .with_mode(SaturationMode::Tape)
            .with_output_gain_db(0.0);
        let mut proc = SaturationProcessor::new(cfg, sr).expect("valid");
        proc.process(&mut buf);

        // The saturated signal will have different shape (harmonics added)
        // but similar energy level. We just verify it was actually modified.
        let wet_energy: f32 = buf.iter().map(|x| x * x).sum();
        assert!(
            (wet_energy - dry_energy).abs() / dry_energy.max(1e-10) > 0.001,
            "tape saturation should change signal energy",
        );
    }

    #[test]
    fn test_all_modes_produce_finite_output() {
        let modes = [
            SaturationMode::Tape,
            SaturationMode::Tube,
            SaturationMode::Console,
            SaturationMode::Warm,
        ];
        for mode in modes {
            let mut buf = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.001, -0.001];
            let cfg = SaturationConfig::new()
                .with_drive(1.0)
                .with_mix(1.0)
                .with_mode(mode);
            let mut proc = SaturationProcessor::new_kokoro(cfg).expect("valid");
            proc.process(&mut buf);
            for (i, &v) in buf.iter().enumerate() {
                assert!(v.is_finite(), "mode {mode:?} sample {i} is non-finite: {v}");
            }
        }
    }

    #[test]
    fn test_nan_input_clamped_to_zero() {
        let mut buf = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5, -0.3];
        let cfg = SaturationConfig::new().with_drive(0.5).with_mix(1.0);
        let mut proc = SaturationProcessor::new_kokoro(cfg).expect("valid");
        proc.process(&mut buf);
        for (i, &v) in buf.iter().enumerate() {
            assert!(v.is_finite(), "sample {i} should be finite, got {v}");
        }
    }

    #[test]
    fn test_mix_blending() {
        // mix=0.0 should preserve the input exactly.
        let mut buf = vec![0.1, -0.2, 0.3];
        let original = buf.clone();
        let cfg = SaturationConfig::new().with_drive(0.8).with_mix(0.0);
        let mut proc = SaturationProcessor::new_kokoro(cfg).expect("valid");
        proc.process(&mut buf);
        // With mix=0, output = dry * gain. Check proportionality.
        let gain = db_to_linear(cfg.output_gain_db);
        for (i, (&out, &orig)) in buf.iter().zip(original.iter()).enumerate() {
            let expected = orig * gain;
            assert!(
                (out - expected).abs() < 1e-5,
                "mix=0 sample {i}: expected {expected}, got {out}",
            );
        }
    }

    #[test]
    fn test_process_saturation_convenience() {
        let mut buf = vec![0.1, 0.2, 0.3, 0.4];
        let cfg = SaturationConfig::new().with_drive(0.3);
        process_saturation(&mut buf, &cfg).expect("should succeed");
        for &v in &buf {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_waveshape_tape_asymmetry() {
        // Tape mode should produce different magnitudes for positive vs negative.
        let pos = waveshape_tape(0.5, 0.5);
        let neg = waveshape_tape(-0.5, 0.5);
        assert!(
            (pos.abs() - neg.abs()).abs() > 1e-4,
            "tape should be asymmetric: |{pos}| vs |{neg}|",
        );
    }

    #[test]
    fn test_waveshape_console_symmetry() {
        // Console mode should be symmetric.
        let pos = waveshape_console(0.5, 0.5);
        let neg = waveshape_console(-0.5, 0.5);
        assert!(
            (pos + neg).abs() < 1e-6,
            "console should be symmetric: {pos} vs {neg}",
        );
    }

    #[test]
    fn test_waveshape_tube_bounded() {
        // Tube mode x/(1+|x|) is bounded in (-1, 1).
        for &drive in &[0.0, 0.5, 1.0] {
            for &x in &[-100.0, -1.0, 0.0, 1.0, 100.0] {
                let y = waveshape_tube(x, drive);
                assert!(
                    y.abs() < 1.0 + 1e-6,
                    "tube should be bounded: waveshape_tube({x}, {drive}) = {y}",
                );
            }
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = SaturationConfig::new().with_drive(0.5);
        let mut proc = SaturationProcessor::new_kokoro(cfg).expect("valid");
        let mut buf1 = vec![0.5; 100];
        proc.process(&mut buf1);
        proc.reset();
        // After reset, filter state should be zeroed.
        assert_eq!(proc.decimation_lp.z1, 0.0);
    }
}
