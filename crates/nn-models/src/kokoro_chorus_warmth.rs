// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Vocal warmth and presence processor for Kokoro chorus voices.
//!
//! This module adds analog-style warmth (subtle even-harmonic saturation in
//! the 200-800 Hz body range) and presence (gentle peaking boost at 2-5 kHz
//! for intelligibility). It targets the specific frequency ranges that make
//! voices sound warm and clear, unlike the saturation module (full-band) or
//! the exciter (high-frequency harmonics and air).
//!
//! # Architecture
//!
//! ```text
//! Input ─┬─────────────────────────────────────────────── dry
//!        │
//!        ├─> Bandpass (body) ─> Waveshaper(mode) ─> wet_body
//!        │
//!        └─> Peaking EQ (presence) ──────────────── eq_out
//!
//! Output = eq_out * (1 - warmth) + (eq_out + wet_body * warmth) * warmth
//!        = eq_out + wet_body * warmth_amount
//! ```
//!
//! The body band is isolated via a cascaded highpass+lowpass bandpass filter,
//! then waveshaped to add even harmonics. The presence peaking EQ boosts a
//! narrow band around the presence frequency. The two paths combine
//! additively with independent amount controls.
//!
//! # Warmth modes
//!
//! - **Tube** — Asymmetric soft clipping producing predominantly even
//!   harmonics. Models a single-ended tube stage where positive and negative
//!   half-cycles are amplified differently.
//! - **Tape** — Symmetric saturation with a gentle high-frequency rolloff
//!   that models the magnetic hysteresis of analog tape.
//! - **Transformer** — Subtle even-harmonic enhancement with a low-frequency
//!   bump, modeling the inductance saturation of iron-core transformers.
//!
//! # References
//!
//! - Valimaki, V. & Reiss, J. D. "All About Audio Equalization."
//!   Applied Sciences, 6(5), 2016.
//! - Zolzer, U. "DAFX: Digital Audio Effects." 2nd ed., Wiley, 2011.
//! - Smith, J. O. "Physical Audio Signal Processing."
//!   <https://ccrma.stanford.edu/~jos/pasp/>
//!
//! Part of #4582, #3351.

use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Warmth mode
// ---------------------------------------------------------------------------

/// Analog warmth character applied to the vocal body band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WarmthMode {
    /// Asymmetric soft clipping (even harmonics dominant). Models a
    /// single-ended tube stage operated below clipping threshold.
    Tube,
    /// Symmetric saturation with gentle HF rolloff. Models magnetic
    /// tape hysteresis — warm, slightly compressed character.
    Tape,
    /// Subtle even-harmonic enhancement with LF bump. Models the
    /// inductance saturation of iron-core audio transformers.
    Transformer,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the vocal warmth and presence processor.
///
/// Constructed via [`WarmthConfig::new`] (required for cross-crate use
/// due to `#[non_exhaustive]`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct WarmthConfig {
    /// Warmth (body saturation) amount: 0.0 = bypass, 1.0 = full.
    /// Default: 0.3.
    pub warmth_amount: f32,
    /// Presence (clarity boost) amount: 0.0 = bypass, 1.0 = full.
    /// Default: 0.25.
    pub presence_amount: f32,
    /// Center frequency (Hz) of the body band for warmth saturation.
    /// Range: 100.0 - 1000.0. Default: 400.0.
    pub body_freq_hz: f32,
    /// Center frequency (Hz) of the presence peaking EQ.
    /// Range: 1500.0 - 6000.0. Default: 3000.0.
    pub presence_freq_hz: f32,
    /// Analog warmth character.
    /// Default: [`WarmthMode::Tube`].
    pub warmth_mode: WarmthMode,
}

impl Default for WarmthConfig {
    fn default() -> Self {
        Self {
            warmth_amount: 0.3,
            presence_amount: 0.25,
            body_freq_hz: 400.0,
            presence_freq_hz: 3000.0,
            warmth_mode: WarmthMode::Tube,
        }
    }
}

impl WarmthConfig {
    /// Create a new warmth config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the warmth (body saturation) amount.
    #[must_use]
    pub fn with_warmth_amount(mut self, amount: f32) -> Self {
        self.warmth_amount = amount;
        self
    }

    /// Set the presence (clarity boost) amount.
    #[must_use]
    pub fn with_presence_amount(mut self, amount: f32) -> Self {
        self.presence_amount = amount;
        self
    }

    /// Set the body band center frequency in Hz.
    #[must_use]
    pub fn with_body_freq_hz(mut self, hz: f32) -> Self {
        self.body_freq_hz = hz;
        self
    }

    /// Set the presence peaking EQ center frequency in Hz.
    #[must_use]
    pub fn with_presence_freq_hz(mut self, hz: f32) -> Self {
        self.presence_freq_hz = hz;
        self
    }

    /// Set the analog warmth mode.
    #[must_use]
    pub fn with_warmth_mode(mut self, mode: WarmthMode) -> Self {
        self.warmth_mode = mode;
        self
    }

    /// Validate all parameters are within acceptable ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.warmth_amount.is_finite() || self.warmth_amount < 0.0 || self.warmth_amount > 1.0 {
            return Err(KokoroError::InvalidConfig {
                field: "warmth_amount",
                reason: format!(
                    "warmth_amount = {}: must be finite and in [0.0, 1.0]",
                    self.warmth_amount,
                ),
            });
        }
        if !self.presence_amount.is_finite()
            || self.presence_amount < 0.0
            || self.presence_amount > 1.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "presence_amount",
                reason: format!(
                    "presence_amount = {}: must be finite and in [0.0, 1.0]",
                    self.presence_amount,
                ),
            });
        }
        if !self.body_freq_hz.is_finite() || self.body_freq_hz < 100.0 || self.body_freq_hz > 1000.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "body_freq_hz",
                reason: format!(
                    "body_freq_hz = {}: must be finite and in [100, 1000]",
                    self.body_freq_hz,
                ),
            });
        }
        if !self.presence_freq_hz.is_finite()
            || self.presence_freq_hz < 1500.0
            || self.presence_freq_hz > 6000.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "presence_freq_hz",
                reason: format!(
                    "presence_freq_hz = {}: must be finite and in [1500, 6000]",
                    self.presence_freq_hz,
                ),
            });
        }
        Ok(())
    }

    // --- Presets ---------------------------------------------------------------

    /// Subtle warmth — barely perceptible body thickening, gentle presence.
    #[must_use]
    pub fn subtle() -> Self {
        Self {
            warmth_amount: 0.15,
            presence_amount: 0.15,
            body_freq_hz: 350.0,
            presence_freq_hz: 3200.0,
            warmth_mode: WarmthMode::Transformer,
        }
    }

    /// Warm broadcast voice — full body with clear presence for speech.
    #[must_use]
    pub fn warm_broadcast() -> Self {
        Self {
            warmth_amount: 0.45,
            presence_amount: 0.35,
            body_freq_hz: 400.0,
            presence_freq_hz: 3000.0,
            warmth_mode: WarmthMode::Tube,
        }
    }

    /// Vintage radio — heavy tube warmth, rolled-off presence for a
    /// retro AM/FM broadcast character.
    #[must_use]
    pub fn vintage_radio() -> Self {
        Self {
            warmth_amount: 0.7,
            presence_amount: 0.2,
            body_freq_hz: 500.0,
            presence_freq_hz: 2500.0,
            warmth_mode: WarmthMode::Tape,
        }
    }

    /// Intimate close-mic vocal — gentle tape warmth with moderate
    /// presence for a whispered or ASMR-style character.
    #[must_use]
    pub fn intimate() -> Self {
        Self {
            warmth_amount: 0.35,
            presence_amount: 0.4,
            body_freq_hz: 300.0,
            presence_freq_hz: 3500.0,
            warmth_mode: WarmthMode::Tape,
        }
    }
}

// ---------------------------------------------------------------------------
// One-pole filters (highpass + lowpass for bandpass isolation)
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

/// Single-pole highpass derived from RC time constant.
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
// Peaking EQ for presence
// ---------------------------------------------------------------------------

/// Second-order peaking (bell) EQ filter.
///
/// Based on the Audio EQ Cookbook (Robert Bristow-Johnson). Provides a
/// narrow boost around the center frequency with configurable gain.
#[derive(Debug, Clone)]
struct PeakingEQ {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl PeakingEQ {
    /// Create a peaking EQ at `freq_hz` with `gain_db` boost and `q` factor.
    fn new(freq_hz: f32, gain_db: f32, q: f32, sample_rate: f32) -> Self {
        if gain_db.abs() < 1e-6 {
            return Self::passthrough();
        }

        let a = 10.0_f32.powf(gain_db / 40.0); // sqrt of linear gain
        let w0 = 2.0 * std::f32::consts::PI * freq_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a;

        // Normalize by a0.
        let inv_a0 = 1.0 / a0;
        Self {
            b0: b0 * inv_a0,
            b1: b1 * inv_a0,
            b2: b2 * inv_a0,
            a1: a1 * inv_a0,
            a2: a2 * inv_a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn passthrough() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.x1 = 0.0;
            self.x2 = 0.0;
            self.y1 = 0.0;
            self.y2 = 0.0;
            return 0.0;
        }
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = if y.is_finite() { y } else { 0.0 };
        self.y1
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Waveshaping functions (body band only)
// ---------------------------------------------------------------------------

/// Tube warmth: asymmetric soft clipping producing even harmonics.
///
/// Positive half is driven harder than negative, creating the characteristic
/// second-harmonic content of single-ended tube amplifiers.
#[inline]
fn waveshape_tube(x: f32, drive: f32) -> f32 {
    let gain = 1.0 + drive * 4.0;
    if x >= 0.0 {
        let driven = x * gain * 1.15; // positive driven harder
        driven / (1.0 + driven.abs())
    } else {
        let driven = x * gain * 0.85; // negative softer
        driven / (1.0 + driven.abs())
    }
}

/// Tape warmth: symmetric saturation with gentle HF rolloff.
///
/// Models magnetic tape hysteresis — the signal is compressed symmetrically
/// via tanh, then a one-pole lowpass simulates the tape head's inherent
/// HF loss. The `lp_state` parameter carries the lowpass filter state.
#[inline]
fn waveshape_tape(x: f32, drive: f32) -> f32 {
    let gain = 1.0 + drive * 3.0;
    (x * gain).tanh()
}

/// Transformer warmth: even-harmonic enhancement with LF emphasis.
///
/// Models iron-core transformer saturation. The polynomial waveshaper
/// (x + c*x^2) generates predominantly even harmonics. The coefficient
/// is kept small for subtlety.
#[inline]
fn waveshape_transformer(x: f32, drive: f32) -> f32 {
    let c = drive * 0.25;
    let y = x + c * x * x;
    // Soft-limit to prevent runaway at high drive.
    y / (1.0 + y.abs() * 0.1)
}

/// Apply the body-band waveshaper for the given mode.
#[inline]
fn apply_body_waveshaper(x: f32, drive: f32, mode: WarmthMode) -> f32 {
    match mode {
        WarmthMode::Tube => waveshape_tube(x, drive),
        WarmthMode::Tape => waveshape_tape(x, drive),
        WarmthMode::Transformer => waveshape_transformer(x, drive),
    }
}

// ---------------------------------------------------------------------------
// WarmthProcessor
// ---------------------------------------------------------------------------

/// Stateful vocal warmth and presence processor.
///
/// Holds filter state for the bandpass (body isolation), peaking EQ
/// (presence boost), and an optional tape HF rolloff lowpass.
#[derive(Debug, Clone)]
pub struct WarmthProcessor {
    config: WarmthConfig,
    /// Highpass leg of the body bandpass.
    body_hp: OnePoleHP,
    /// Lowpass leg of the body bandpass.
    body_lp: OnePoleLP,
    /// Tape-mode HF rolloff (only active in Tape mode).
    tape_lp: OnePoleLP,
    /// Presence peaking EQ.
    presence_eq: PeakingEQ,
}

impl WarmthProcessor {
    /// Create a new warmth processor from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range,
    /// or if `sample_rate` is not finite and positive.
    pub fn new(config: WarmthConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("sample_rate = {sample_rate}: must be finite and positive"),
            });
        }

        // Body bandpass: highpass at body_freq / 2, lowpass at body_freq * 2.
        // This gives a ~2-octave-wide band centered on body_freq_hz.
        let bp_low = (config.body_freq_hz / 2.0).max(20.0);
        let bp_high = (config.body_freq_hz * 2.0).min(sample_rate * 0.45);
        let body_hp = OnePoleHP::new(bp_low, sample_rate);
        let body_lp = OnePoleLP::new(bp_high, sample_rate);

        // Tape mode HF rolloff at 4 kHz (simulates tape head loss).
        let tape_lp = OnePoleLP::new(4000.0, sample_rate);

        // Presence peaking EQ: gain proportional to presence_amount,
        // max +6 dB, Q = 1.5 for a musically broad bell.
        let pres_gain_db = config.presence_amount * 6.0;
        let presence_eq = PeakingEQ::new(config.presence_freq_hz, pres_gain_db, 1.5, sample_rate);

        Ok(Self {
            config,
            body_hp,
            body_lp,
            tape_lp,
            presence_eq,
        })
    }

    /// Create a processor using Kokoro's default 24 kHz sample rate.
    pub fn new_kokoro(config: WarmthConfig) -> Result<Self, KokoroError> {
        Self::new(config, KOKORO_SAMPLE_RATE as f32)
    }

    /// Process per-voice audio in-place.
    ///
    /// Fast path: returns immediately when both amounts are zero.
    pub fn process_voice(&mut self, audio: &mut [f32]) {
        if self.config.warmth_amount == 0.0 && self.config.presence_amount == 0.0 {
            return;
        }

        let warmth = self.config.warmth_amount;
        let mode = self.config.warmth_mode;
        let has_warmth = warmth > 0.0;

        for sample in audio.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
                continue;
            }

            let input = *sample;

            // --- Presence EQ path (applied to full signal) ---
            let eq_out = self.presence_eq.process(input);

            // --- Body warmth path ---
            if has_warmth {
                // Isolate the body band.
                let hp_out = self.body_hp.process(input);
                let body = self.body_lp.process(hp_out);

                // Apply waveshaper to the isolated body band.
                let mut shaped = apply_body_waveshaper(body, warmth, mode);

                // Tape mode: apply additional HF rolloff after waveshaping.
                if mode == WarmthMode::Tape {
                    shaped = self.tape_lp.process(shaped);
                }

                // Blend: EQ'd signal + shaped body harmonics scaled by warmth.
                *sample = eq_out + shaped * warmth;
            } else {
                *sample = eq_out;
            }

            // Final NaN/Inf guard.
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }
    }

    /// Reset all internal filter state (call between unrelated audio segments).
    pub fn reset(&mut self) {
        self.body_hp.reset();
        self.body_lp.reset();
        self.tape_lp.reset();
        self.presence_eq.reset();
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &WarmthConfig {
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
        WarmthConfig::new()
            .validate()
            .expect("default config should be valid");
    }

    #[test]
    fn test_config_builder_roundtrip() {
        let cfg = WarmthConfig::new()
            .with_warmth_amount(0.5)
            .with_presence_amount(0.4)
            .with_body_freq_hz(500.0)
            .with_presence_freq_hz(4000.0)
            .with_warmth_mode(WarmthMode::Tape);
        cfg.validate().expect("builder config should be valid");
        assert_eq!(cfg.warmth_amount, 0.5);
        assert_eq!(cfg.presence_amount, 0.4);
        assert_eq!(cfg.body_freq_hz, 500.0);
        assert_eq!(cfg.presence_freq_hz, 4000.0);
        assert_eq!(cfg.warmth_mode, WarmthMode::Tape);
    }

    #[test]
    fn test_config_invalid_warmth_amount() {
        assert!(WarmthConfig::new()
            .with_warmth_amount(1.5)
            .validate()
            .is_err());
        assert!(WarmthConfig::new()
            .with_warmth_amount(-0.1)
            .validate()
            .is_err());
        assert!(WarmthConfig::new()
            .with_warmth_amount(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_presence_amount() {
        assert!(WarmthConfig::new()
            .with_presence_amount(1.1)
            .validate()
            .is_err());
        assert!(WarmthConfig::new()
            .with_presence_amount(f32::INFINITY)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_body_freq() {
        assert!(WarmthConfig::new()
            .with_body_freq_hz(50.0)
            .validate()
            .is_err());
        assert!(WarmthConfig::new()
            .with_body_freq_hz(2000.0)
            .validate()
            .is_err());
        assert!(WarmthConfig::new()
            .with_body_freq_hz(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_presence_freq() {
        assert!(WarmthConfig::new()
            .with_presence_freq_hz(500.0)
            .validate()
            .is_err());
        assert!(WarmthConfig::new()
            .with_presence_freq_hz(8000.0)
            .validate()
            .is_err());
    }

    #[test]
    fn test_presets_valid() {
        WarmthConfig::subtle().validate().expect("subtle valid");
        WarmthConfig::warm_broadcast()
            .validate()
            .expect("broadcast valid");
        WarmthConfig::vintage_radio()
            .validate()
            .expect("vintage valid");
        WarmthConfig::intimate().validate().expect("intimate valid");
    }

    // --- Processor behavior ---

    #[test]
    fn test_zero_amounts_is_noop() {
        let mut buf = sine_wave(440.0, 2048, 0.5);
        let original = buf.clone();
        let cfg = WarmthConfig::new()
            .with_warmth_amount(0.0)
            .with_presence_amount(0.0);
        let mut proc = WarmthProcessor::new_kokoro(cfg).expect("valid");
        proc.process_voice(&mut buf);
        assert_eq!(buf, original, "zero amounts should not modify signal");
    }

    #[test]
    fn test_warmth_modifies_signal() {
        let mut buf = sine_wave(400.0, 4096, 0.5);
        let dry_rms = rms(&buf);
        let cfg = WarmthConfig::new()
            .with_warmth_amount(0.8)
            .with_presence_amount(0.0);
        let mut proc = WarmthProcessor::new_kokoro(cfg).expect("valid");
        proc.process_voice(&mut buf);
        let wet_rms = rms(&buf);
        assert!(
            (wet_rms - dry_rms).abs() > 1e-4,
            "warmth should modify signal: dry={dry_rms}, wet={wet_rms}",
        );
    }

    #[test]
    fn test_presence_modifies_signal() {
        // Use a signal with energy near the presence frequency.
        let mut buf = sine_wave(3000.0, 4096, 0.3);
        let dry_rms = rms(&buf);
        let cfg = WarmthConfig::new()
            .with_warmth_amount(0.0)
            .with_presence_amount(0.8);
        let mut proc = WarmthProcessor::new_kokoro(cfg).expect("valid");
        proc.process_voice(&mut buf);
        let wet_rms = rms(&buf);
        assert!(
            wet_rms > dry_rms * 1.01,
            "presence boost should increase energy at 3 kHz: \
             dry={dry_rms}, wet={wet_rms}",
        );
    }

    #[test]
    fn test_all_modes_produce_finite_output() {
        let modes = [WarmthMode::Tube, WarmthMode::Tape, WarmthMode::Transformer];
        for mode in modes {
            let mut buf = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.001, -0.001];
            let cfg = WarmthConfig::new()
                .with_warmth_amount(1.0)
                .with_presence_amount(1.0)
                .with_warmth_mode(mode);
            let mut proc = WarmthProcessor::new_kokoro(cfg).expect("valid");
            proc.process_voice(&mut buf);
            for (i, &v) in buf.iter().enumerate() {
                assert!(v.is_finite(), "mode {mode:?} sample {i} is non-finite: {v}");
            }
        }
    }

    #[test]
    fn test_nan_input_clamped_to_zero() {
        let mut buf = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5];
        let cfg = WarmthConfig::new()
            .with_warmth_amount(0.5)
            .with_presence_amount(0.5);
        let mut proc = WarmthProcessor::new_kokoro(cfg).expect("valid");
        proc.process_voice(&mut buf);
        for (i, &v) in buf.iter().enumerate() {
            assert!(v.is_finite(), "sample {i} should be finite, got {v}");
        }
    }

    #[test]
    fn test_tube_asymmetry() {
        let pos = waveshape_tube(0.5, 0.5);
        let neg = waveshape_tube(-0.5, 0.5);
        assert!(
            (pos.abs() - neg.abs()).abs() > 1e-4,
            "tube should be asymmetric: |{pos}| vs |{neg}|",
        );
    }

    #[test]
    fn test_tape_symmetry() {
        let pos = waveshape_tape(0.5, 0.5);
        let neg = waveshape_tape(-0.5, 0.5);
        assert!(
            (pos + neg).abs() < 1e-6,
            "tape should be symmetric: {pos} vs {neg}",
        );
    }

    #[test]
    fn test_transformer_even_harmonics() {
        // Transformer x + c*x^2 should produce different magnitude for +/-
        let pos = waveshape_transformer(0.5, 0.5);
        let neg = waveshape_transformer(-0.5, 0.5);
        assert!(
            (pos.abs() - neg.abs()).abs() > 1e-4,
            "transformer should have asymmetric magnitude: |{pos}| vs |{neg}|",
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = WarmthConfig::new()
            .with_warmth_amount(0.5)
            .with_presence_amount(0.5);
        let mut proc = WarmthProcessor::new_kokoro(cfg).expect("valid");
        let mut buf = vec![0.5; 100];
        proc.process_voice(&mut buf);
        proc.reset();
        assert_eq!(proc.body_hp.x_prev, 0.0);
        assert_eq!(proc.body_hp.y_prev, 0.0);
        assert_eq!(proc.body_lp.z1, 0.0);
        assert_eq!(proc.tape_lp.z1, 0.0);
        assert_eq!(proc.presence_eq.x1, 0.0);
        assert_eq!(proc.presence_eq.y1, 0.0);
    }

    #[test]
    fn test_invalid_sample_rate() {
        let cfg = WarmthConfig::new();
        assert!(WarmthProcessor::new(cfg, 0.0).is_err());
        assert!(WarmthProcessor::new(cfg, -44100.0).is_err());
        assert!(WarmthProcessor::new(cfg, f32::NAN).is_err());
    }

    #[test]
    fn test_empty_buffer() {
        let cfg = WarmthConfig::new();
        let mut proc = WarmthProcessor::new_kokoro(cfg).expect("valid");
        let mut buf: Vec<f32> = vec![];
        proc.process_voice(&mut buf);
        assert!(buf.is_empty());
    }
}
