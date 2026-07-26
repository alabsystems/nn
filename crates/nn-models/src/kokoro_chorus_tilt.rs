// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Spectral tilt and brightness control for Kokoro chorus output.
//!
//! A tilt filter adjusts the overall spectral slope with a single intuitive
//! control. Positive `tilt_db` makes the output brighter (boost HF, cut LF);
//! negative makes it darker (boost LF, cut HF). This is simpler than a full
//! parametric EQ and maps naturally to the perceptual bright-vs-dark axis.
//!
//! # Architecture
//!
//! ```text
//! Input ──> 1st-order allpass-based tilt ──> [optional air/body shelf] ──> Output
//!
//! Tilt core:
//!   ap[n] = allpass(x[n], pole)     -- 1st-order allpass at pivot freq
//!   lp[n] = (x[n] + ap[n]) / 2      -- low-pass half (ap = +1 at DC)
//!   hp[n] = (x[n] - ap[n]) / 2      -- high-pass half (ap = -1 at Nyquist)
//!   y[n]  = (1-k) * lp[n]  +  (1+k) * hp[n]  =  x[n] - k * ap[n]
//!
//!   k > 0: HF boost + LF cut  (brighter)
//!   k < 0: LF boost + HF cut  (darker)
//!   k = 0: unity gain          (neutral)
//! ```
//!
//! The allpass-based tilt filter is extremely efficient -- one allpass state
//! variable plus a weighted sum. The allpass is tuned to the pivot frequency
//! so that the gain crossover (0 dB) occurs at the pivot.
//!
//! # Tilt modes
//!
//! - **TiltOnly** -- Pure spectral tilt, no additional shaping.
//! - **TiltPlusAir** -- Tilt + gentle high-shelf at 10 kHz for airy shimmer.
//! - **TiltPlusBody** -- Tilt + gentle low-shelf at 200 Hz for chest warmth.
//!
//! # References
//!
//! - Pirkle, W. C. "Designing Audio Effect Plugins in C++." 2nd ed., 2019.
//!   Chapter 11: First-order allpass-based tilt filter.
//! - Valimaki, V. & Reiss, J. D. "All About Audio Equalization."
//!   Applied Sciences, 6(5), 2016.
//! - Smith, J. O. "Introduction to Digital Filters with Audio Applications."
//!   <https://ccrma.stanford.edu/~jos/filters/>
//!
//! Part of #4582, #3351.

use crate::kokoro_chorus_saturation::db_to_linear;
use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Tilt mode
// ---------------------------------------------------------------------------

/// Spectral tilt processing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TiltMode {
    /// Pure spectral tilt only.
    TiltOnly,
    /// Tilt plus a gentle high-shelf "air" boost at 10 kHz.
    TiltPlusAir,
    /// Tilt plus a gentle low-shelf "body" boost at 200 Hz.
    TiltPlusBody,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the spectral tilt and brightness processor.
///
/// Constructed via [`TiltConfig::new`] (required for cross-crate use
/// due to `#[non_exhaustive]`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct TiltConfig {
    /// Tilt amount in dB. Positive = brighter, negative = darker.
    /// Range: -12.0 to +12.0. Default: 0.0 (neutral).
    pub tilt_db: f32,
    /// Pivot frequency in Hz where the tilt crosses 0 dB.
    /// Range: 200.0 to 5000.0. Default: 1000.0.
    pub pivot_freq_hz: f32,
    /// Processing mode controlling optional air/body shelves.
    /// Default: [`TiltMode::TiltOnly`].
    pub mode: TiltMode,
}

impl Default for TiltConfig {
    fn default() -> Self {
        Self {
            tilt_db: 0.0,
            pivot_freq_hz: 1000.0,
            mode: TiltMode::TiltOnly,
        }
    }
}

impl TiltConfig {
    /// Create a new tilt config with default values (neutral, 1 kHz pivot).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the tilt amount in dB.
    #[must_use]
    pub fn with_tilt_db(mut self, db: f32) -> Self {
        self.tilt_db = db;
        self
    }

    /// Set the pivot frequency in Hz.
    #[must_use]
    pub fn with_pivot_freq_hz(mut self, hz: f32) -> Self {
        self.pivot_freq_hz = hz;
        self
    }

    /// Set the processing mode.
    #[must_use]
    pub fn with_mode(mut self, mode: TiltMode) -> Self {
        self.mode = mode;
        self
    }

    /// Validate all parameters are within acceptable ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.tilt_db.is_finite() || self.tilt_db < -12.0 || self.tilt_db > 12.0 {
            return Err(KokoroError::InvalidConfig {
                field: "tilt_db",
                reason: format!(
                    "tilt_db = {}: must be finite and in [-12.0, 12.0]",
                    self.tilt_db,
                ),
            });
        }
        if !self.pivot_freq_hz.is_finite()
            || self.pivot_freq_hz < 200.0
            || self.pivot_freq_hz > 5000.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "pivot_freq_hz",
                reason: format!(
                    "pivot_freq_hz = {}: must be finite and in [200.0, 5000.0]",
                    self.pivot_freq_hz,
                ),
            });
        }
        Ok(())
    }

    // --- Presets ---------------------------------------------------------------

    /// Bright preset: +3 dB tilt for added sparkle and presence.
    #[must_use]
    pub fn bright() -> Self {
        Self {
            tilt_db: 3.0,
            pivot_freq_hz: 1000.0,
            mode: TiltMode::TiltOnly,
        }
    }

    /// Dark preset: -3 dB tilt for a warmer, darker tone.
    #[must_use]
    pub fn dark() -> Self {
        Self {
            tilt_db: -3.0,
            pivot_freq_hz: 1000.0,
            mode: TiltMode::TiltOnly,
        }
    }

    /// Airy preset: +2 dB tilt with additional air shelf boost.
    #[must_use]
    pub fn airy() -> Self {
        Self {
            tilt_db: 2.0,
            pivot_freq_hz: 1000.0,
            mode: TiltMode::TiltPlusAir,
        }
    }

    /// Warm preset: -2 dB tilt with additional body shelf boost.
    #[must_use]
    pub fn warm() -> Self {
        Self {
            tilt_db: -2.0,
            pivot_freq_hz: 1000.0,
            mode: TiltMode::TiltPlusBody,
        }
    }

    /// Neutral preset: 0 dB tilt, pure pass-through.
    #[must_use]
    pub fn neutral() -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// First-order allpass filter
// ---------------------------------------------------------------------------

/// First-order allpass filter tuned to a given frequency.
///
/// H(z) = (a + z^-1) / (1 + a * z^-1)
///
/// At the tuning frequency the allpass has 90 degrees of phase shift.
/// Below that frequency it tends toward 0 degrees; above, toward 180.
#[derive(Debug, Clone)]
struct Allpass1 {
    /// Allpass coefficient.
    a: f32,
    /// Previous input sample.
    x_prev: f32,
    /// Previous output sample.
    y_prev: f32,
}

impl Allpass1 {
    /// Create a first-order allpass tuned so the 90-degree phase point
    /// sits at `freq_hz`.
    fn new(freq_hz: f32, sample_rate: f32) -> Self {
        // Bilinear-transform allpass coefficient:
        //   a = (tan(pi*fc/fs) - 1) / (tan(pi*fc/fs) + 1)
        let t = (std::f32::consts::PI * freq_hz / sample_rate).tan();
        let a = (t - 1.0) / (t + 1.0);
        Self {
            a,
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
        let y = self.a * x + self.x_prev - self.a * self.y_prev;
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
// First-order shelving filters (for air / body modes)
// ---------------------------------------------------------------------------

/// First-order high-shelf filter for air band enhancement.
///
/// Based on Zolzer, "DAFX," 2nd ed., eq. (2.22). Boosts frequencies
/// above `freq_hz` by `gain_db`.
#[derive(Debug, Clone)]
struct HighShelf1 {
    b0: f32,
    b1: f32,
    a1: f32,
    x_prev: f32,
    y_prev: f32,
}

impl HighShelf1 {
    fn new(freq_hz: f32, gain_db: f32, sample_rate: f32) -> Self {
        if gain_db.abs() < 1e-6 {
            return Self::passthrough();
        }
        let v0 = db_to_linear(gain_db);
        let k = (std::f32::consts::PI * freq_hz / sample_rate).tan();
        if v0 >= 1.0 {
            let denom = 1.0 + k;
            Self {
                b0: (v0 + k) / denom,
                b1: (k - v0) / denom,
                a1: (k - 1.0) / denom,
                x_prev: 0.0,
                y_prev: 0.0,
            }
        } else {
            let denom = v0 + k;
            Self {
                b0: v0 * (1.0 + k) / denom,
                b1: v0 * (k - 1.0) / denom,
                a1: (k - v0) / denom,
                x_prev: 0.0,
                y_prev: 0.0,
            }
        }
    }

    fn passthrough() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            a1: 0.0,
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
        let y = self.b0 * x + self.b1 * self.x_prev - self.a1 * self.y_prev;
        self.x_prev = x;
        self.y_prev = if y.is_finite() { y } else { 0.0 };
        self.y_prev
    }

    fn reset(&mut self) {
        self.x_prev = 0.0;
        self.y_prev = 0.0;
    }
}

/// First-order low-shelf filter for body enhancement.
///
/// Boosts frequencies below `freq_hz` by `gain_db`.
#[derive(Debug, Clone)]
struct LowShelf1 {
    b0: f32,
    b1: f32,
    a1: f32,
    x_prev: f32,
    y_prev: f32,
}

impl LowShelf1 {
    fn new(freq_hz: f32, gain_db: f32, sample_rate: f32) -> Self {
        if gain_db.abs() < 1e-6 {
            return Self::passthrough();
        }
        let v0 = db_to_linear(gain_db);
        let k = (std::f32::consts::PI * freq_hz / sample_rate).tan();
        if v0 >= 1.0 {
            // Boost
            let denom = 1.0 + k;
            Self {
                b0: (1.0 + v0 * k) / denom,
                b1: (v0 * k - 1.0) / denom,
                a1: (k - 1.0) / denom,
                x_prev: 0.0,
                y_prev: 0.0,
            }
        } else {
            // Cut
            let denom = 1.0 + k / v0;
            Self {
                b0: (1.0 + k) / denom,
                b1: (k - 1.0) / denom,
                a1: (k / v0 - 1.0) / denom,
                x_prev: 0.0,
                y_prev: 0.0,
            }
        }
    }

    fn passthrough() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            a1: 0.0,
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
        let y = self.b0 * x + self.b1 * self.x_prev - self.a1 * self.y_prev;
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
// TiltProcessor
// ---------------------------------------------------------------------------

/// Stateful spectral tilt processor.
///
/// Applies a single-knob brightness control to audio using a 1st-order
/// allpass-based tilt filter, with optional air or body shelf enhancement.
#[derive(Debug, Clone)]
pub struct TiltProcessor {
    config: TiltConfig,
    /// Tilt mix parameter k: positive = bright, negative = dark.
    k: f32,
    /// Core allpass filter at the pivot frequency.
    allpass: Allpass1,
    /// Optional high-shelf for TiltPlusAir mode.
    air_shelf: Option<HighShelf1>,
    /// Optional low-shelf for TiltPlusBody mode.
    body_shelf: Option<LowShelf1>,
}

/// Air shelf: gentle boost at 10 kHz, fixed at +2 dB.
const AIR_SHELF_FREQ_HZ: f32 = 10000.0;
const AIR_SHELF_GAIN_DB: f32 = 2.0;

/// Body shelf: gentle boost at 200 Hz, fixed at +2 dB.
const BODY_SHELF_FREQ_HZ: f32 = 200.0;
const BODY_SHELF_GAIN_DB: f32 = 2.0;

impl TiltProcessor {
    /// Create a new tilt processor from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range,
    /// or if `sample_rate` is not finite and positive.
    pub fn new(config: TiltConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("sample_rate = {sample_rate}: must be finite and positive"),
            });
        }

        // Convert tilt_db to the mixing coefficient k.
        // k maps linearly: tilt_db / max_tilt_db so that +/-12 dB -> +/-1.
        let k = config.tilt_db / 12.0;

        let allpass = Allpass1::new(config.pivot_freq_hz, sample_rate);

        let air_shelf = match config.mode {
            TiltMode::TiltPlusAir => Some(HighShelf1::new(
                AIR_SHELF_FREQ_HZ,
                AIR_SHELF_GAIN_DB,
                sample_rate,
            )),
            _ => None,
        };

        let body_shelf = match config.mode {
            TiltMode::TiltPlusBody => Some(LowShelf1::new(
                BODY_SHELF_FREQ_HZ,
                BODY_SHELF_GAIN_DB,
                sample_rate,
            )),
            _ => None,
        };

        Ok(Self {
            config,
            k,
            allpass,
            air_shelf,
            body_shelf,
        })
    }

    /// Create a processor using Kokoro's default 24 kHz sample rate.
    pub fn new_kokoro(config: TiltConfig) -> Result<Self, KokoroError> {
        Self::new(config, KOKORO_SAMPLE_RATE as f32)
    }

    /// Process a single channel of audio in-place.
    ///
    /// Fast path: returns immediately when `tilt_db == 0.0` and mode is
    /// `TiltOnly` (pure pass-through).
    pub fn process(&mut self, audio: &mut [f32]) {
        let is_noop = self.config.tilt_db == 0.0 && self.config.mode == TiltMode::TiltOnly;
        if is_noop {
            return;
        }

        let k = self.k;

        for sample in audio.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
                continue;
            }

            let x = *sample;

            // Core tilt. Decompose the input into low- and high-pass halves
            // using the allpass: lp = (x + ap)/2, hp = (x - ap)/2. The allpass
            // is +1 at DC and -1 at Nyquist, so lp passes lows and hp passes
            // highs. Weight them by (1-k) and (1+k):
            //   y = (1-k)*lp + (1+k)*hp = x - k*ap
            // For k > 0 this boosts HF and cuts LF (brighter); k < 0 darkens;
            // k = 0 is unity gain.
            let ap = self.allpass.process(x);
            let tilted = x - k * ap;

            *sample = tilted;

            // Optional air shelf (TiltPlusAir mode).
            if let Some(ref mut shelf) = self.air_shelf {
                *sample = shelf.process(*sample);
            }

            // Optional body shelf (TiltPlusBody mode).
            if let Some(ref mut shelf) = self.body_shelf {
                *sample = shelf.process(*sample);
            }

            // Final NaN/Inf guard.
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }
    }

    /// Process stereo audio (left and right channels) in-place.
    ///
    /// Both channels share the same tilt configuration but use the
    /// processor's single set of filter state. For independent per-channel
    /// state, create two `TiltProcessor` instances.
    ///
    /// Note: this processes left then right sequentially. The filter state
    /// carries across channels, which is acceptable for matched-length
    /// stereo buffers from the same source.
    pub fn process_stereo(&mut self, left: &mut Vec<f32>, right: &mut Vec<f32>) {
        self.process(left);
        self.process(right);
    }

    /// Reset all internal filter state (call between unrelated audio segments).
    pub fn reset(&mut self) {
        self.allpass.reset();
        if let Some(ref mut shelf) = self.air_shelf {
            shelf.reset();
        }
        if let Some(ref mut shelf) = self.body_shelf {
            shelf.reset();
        }
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &TiltConfig {
        &self.config
    }

    /// The effective tilt mix coefficient k (in [-1, +1]).
    #[must_use]
    pub fn tilt_k(&self) -> f32 {
        self.k
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

    /// Measure energy through a simple one-pole highpass at the given cutoff.
    fn hf_energy(buf: &[f32], cutoff_hz: f32) -> f32 {
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
        let dt = 1.0 / SR;
        let coeff = rc / (rc + dt);
        let mut x_prev = 0.0_f32;
        let mut y_prev = 0.0_f32;
        let filtered: Vec<f32> = buf
            .iter()
            .map(|&x| {
                let y = coeff * (y_prev + x - x_prev);
                x_prev = x;
                y_prev = y;
                y
            })
            .collect();
        rms(&filtered)
    }

    /// Measure energy through a simple one-pole lowpass at the given cutoff.
    fn lf_energy(buf: &[f32], cutoff_hz: f32) -> f32 {
        let w = (-2.0 * std::f32::consts::PI * cutoff_hz / SR).exp();
        let b = 1.0 - w;
        let mut z1 = 0.0_f32;
        let filtered: Vec<f32> = buf
            .iter()
            .map(|&x| {
                let y = b * x + w * z1;
                z1 = y;
                y
            })
            .collect();
        rms(&filtered)
    }

    // --- Config validation ---

    #[test]
    fn test_config_default_valid() {
        TiltConfig::new()
            .validate()
            .expect("default config should be valid");
    }

    #[test]
    fn test_config_builder_roundtrip() {
        let cfg = TiltConfig::new()
            .with_tilt_db(6.0)
            .with_pivot_freq_hz(2000.0)
            .with_mode(TiltMode::TiltPlusAir);
        cfg.validate().expect("builder config should be valid");
        assert_eq!(cfg.tilt_db, 6.0);
        assert_eq!(cfg.pivot_freq_hz, 2000.0);
        assert_eq!(cfg.mode, TiltMode::TiltPlusAir);
    }

    #[test]
    fn test_config_invalid_tilt_db() {
        assert!(TiltConfig::new().with_tilt_db(13.0).validate().is_err());
        assert!(TiltConfig::new().with_tilt_db(-13.0).validate().is_err());
        assert!(TiltConfig::new().with_tilt_db(f32::NAN).validate().is_err());
        assert!(TiltConfig::new()
            .with_tilt_db(f32::INFINITY)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_pivot_freq() {
        assert!(TiltConfig::new()
            .with_pivot_freq_hz(50.0)
            .validate()
            .is_err());
        assert!(TiltConfig::new()
            .with_pivot_freq_hz(6000.0)
            .validate()
            .is_err());
        assert!(TiltConfig::new()
            .with_pivot_freq_hz(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_boundary_values_valid() {
        TiltConfig::new()
            .with_tilt_db(-12.0)
            .validate()
            .expect("-12 valid");
        TiltConfig::new()
            .with_tilt_db(12.0)
            .validate()
            .expect("+12 valid");
        TiltConfig::new()
            .with_pivot_freq_hz(200.0)
            .validate()
            .expect("200 Hz valid");
        TiltConfig::new()
            .with_pivot_freq_hz(5000.0)
            .validate()
            .expect("5000 Hz valid");
    }

    #[test]
    fn test_presets_valid() {
        TiltConfig::bright().validate().expect("bright valid");
        TiltConfig::dark().validate().expect("dark valid");
        TiltConfig::airy().validate().expect("airy valid");
        TiltConfig::warm().validate().expect("warm valid");
        TiltConfig::neutral().validate().expect("neutral valid");
    }

    // --- Processor behavior ---

    #[test]
    fn test_neutral_is_noop() {
        let mut buf = sine_wave(440.0, 4096, 0.5);
        let original = buf.clone();
        let cfg = TiltConfig::neutral();
        let mut proc = TiltProcessor::new_kokoro(cfg).expect("valid");
        proc.process(&mut buf);
        assert_eq!(buf, original, "neutral tilt should not modify signal");
    }

    #[test]
    fn test_bright_tilt_boosts_hf() {
        let n = 8192;
        // Broadband signal with energy across the spectrum.
        let mut buf: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR;
                0.3 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 3000.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 8000.0 * t).sin()
            })
            .collect();
        let dry_hf = hf_energy(&buf, 2000.0);

        let cfg = TiltConfig::new().with_tilt_db(6.0);
        let mut proc = TiltProcessor::new_kokoro(cfg).expect("valid");
        proc.process(&mut buf);
        let wet_hf = hf_energy(&buf, 2000.0);

        assert!(
            wet_hf > dry_hf,
            "bright tilt should boost HF energy: dry={dry_hf}, wet={wet_hf}",
        );
    }

    #[test]
    fn test_dark_tilt_boosts_lf() {
        let n = 8192;
        let mut buf: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR;
                0.3 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 3000.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 8000.0 * t).sin()
            })
            .collect();
        let dry_lf = lf_energy(&buf, 500.0);

        let cfg = TiltConfig::new().with_tilt_db(-6.0);
        let mut proc = TiltProcessor::new_kokoro(cfg).expect("valid");
        proc.process(&mut buf);
        let wet_lf = lf_energy(&buf, 500.0);

        assert!(
            wet_lf > dry_lf,
            "dark tilt should boost LF energy: dry={dry_lf}, wet={wet_lf}",
        );
    }

    #[test]
    fn test_tilt_symmetry() {
        // Bright and dark with equal magnitude should produce equal but
        // opposite spectral changes (mirrored around pivot).
        let n = 8192;
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR;
                0.3 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 5000.0 * t).sin()
            })
            .collect();
        let dry_hf = hf_energy(&signal, 2000.0);

        let mut bright_buf = signal.clone();
        let mut bright_proc =
            TiltProcessor::new_kokoro(TiltConfig::new().with_tilt_db(6.0)).unwrap();
        bright_proc.process(&mut bright_buf);
        let bright_hf = hf_energy(&bright_buf, 2000.0);

        let mut dark_buf = signal;
        let mut dark_proc =
            TiltProcessor::new_kokoro(TiltConfig::new().with_tilt_db(-6.0)).unwrap();
        dark_proc.process(&mut dark_buf);
        let dark_hf = hf_energy(&dark_buf, 2000.0);

        // Bright should have more HF than dry, dark should have less.
        assert!(
            bright_hf > dry_hf && dark_hf < dry_hf,
            "tilt should be symmetric: bright_hf={bright_hf}, dry_hf={dry_hf}, dark_hf={dark_hf}",
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
        let cfg = TiltConfig::new().with_tilt_db(12.0);
        let mut proc = TiltProcessor::new_kokoro(cfg).unwrap();
        let mut buf = inputs;
        proc.process(&mut buf);
        for (i, &v) in buf.iter().enumerate() {
            assert!(v.is_finite(), "sample {i} is non-finite: {v}");
        }
    }

    #[test]
    fn test_all_modes_produce_finite_output() {
        let modes = [
            TiltMode::TiltOnly,
            TiltMode::TiltPlusAir,
            TiltMode::TiltPlusBody,
        ];
        for mode in modes {
            let mut buf = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.001, -0.001];
            let cfg = TiltConfig::new().with_tilt_db(6.0).with_mode(mode);
            let mut proc = TiltProcessor::new_kokoro(cfg).expect("valid");
            proc.process(&mut buf);
            for (i, &v) in buf.iter().enumerate() {
                assert!(v.is_finite(), "mode {mode:?} sample {i} is non-finite: {v}");
            }
        }
    }

    #[test]
    fn test_air_mode_adds_hf_shimmer() {
        let n = 8192;
        let mut buf: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR;
                0.3 * (2.0 * std::f32::consts::PI * 500.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 8000.0 * t).sin()
            })
            .collect();

        // Process with TiltOnly at same tilt.
        let mut tilt_only_buf = buf.clone();
        let cfg_only = TiltConfig::new()
            .with_tilt_db(2.0)
            .with_mode(TiltMode::TiltOnly);
        let mut proc_only = TiltProcessor::new_kokoro(cfg_only).unwrap();
        proc_only.process(&mut tilt_only_buf);
        let tilt_only_hf = hf_energy(&tilt_only_buf, 8000.0);

        // Process with TiltPlusAir at same tilt.
        let cfg_air = TiltConfig::new()
            .with_tilt_db(2.0)
            .with_mode(TiltMode::TiltPlusAir);
        let mut proc_air = TiltProcessor::new_kokoro(cfg_air).unwrap();
        proc_air.process(&mut buf);
        let air_hf = hf_energy(&buf, 8000.0);

        assert!(
            air_hf > tilt_only_hf,
            "TiltPlusAir should have more HF energy than TiltOnly: \
             air={air_hf}, tilt_only={tilt_only_hf}",
        );
    }

    #[test]
    fn test_body_mode_adds_lf_warmth() {
        let n = 8192;
        let mut buf: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR;
                0.3 * (2.0 * std::f32::consts::PI * 150.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 3000.0 * t).sin()
            })
            .collect();

        let mut tilt_only_buf = buf.clone();
        let cfg_only = TiltConfig::new()
            .with_tilt_db(-2.0)
            .with_mode(TiltMode::TiltOnly);
        let mut proc_only = TiltProcessor::new_kokoro(cfg_only).unwrap();
        proc_only.process(&mut tilt_only_buf);
        let tilt_only_lf = lf_energy(&tilt_only_buf, 300.0);

        let cfg_body = TiltConfig::new()
            .with_tilt_db(-2.0)
            .with_mode(TiltMode::TiltPlusBody);
        let mut proc_body = TiltProcessor::new_kokoro(cfg_body).unwrap();
        proc_body.process(&mut buf);
        let body_lf = lf_energy(&buf, 300.0);

        assert!(
            body_lf > tilt_only_lf,
            "TiltPlusBody should have more LF energy than TiltOnly: \
             body={body_lf}, tilt_only={tilt_only_lf}",
        );
    }

    #[test]
    fn test_stereo_processes_both_channels() {
        let n = 4096;
        let mut left = sine_wave(440.0, n, 0.5);
        let mut right = sine_wave(880.0, n, 0.5);
        let dry_left_rms = rms(&left);
        let dry_right_rms = rms(&right);

        let cfg = TiltConfig::new().with_tilt_db(6.0);
        let mut proc = TiltProcessor::new_kokoro(cfg).unwrap();
        proc.process_stereo(&mut left, &mut right);

        // Both channels should be modified.
        let wet_left_rms = rms(&left);
        let wet_right_rms = rms(&right);
        assert!(
            (wet_left_rms - dry_left_rms).abs() > 1e-4,
            "left channel should be modified",
        );
        assert!(
            (wet_right_rms - dry_right_rms).abs() > 1e-4,
            "right channel should be modified",
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = TiltConfig::new()
            .with_tilt_db(6.0)
            .with_mode(TiltMode::TiltPlusAir);
        let mut proc = TiltProcessor::new_kokoro(cfg).expect("valid");
        let mut buf = vec![0.5; 100];
        proc.process(&mut buf);
        proc.reset();
        assert_eq!(proc.allpass.x_prev, 0.0);
        assert_eq!(proc.allpass.y_prev, 0.0);
        if let Some(ref shelf) = proc.air_shelf {
            assert_eq!(shelf.x_prev, 0.0);
            assert_eq!(shelf.y_prev, 0.0);
        }
    }

    #[test]
    fn test_invalid_sample_rate() {
        let cfg = TiltConfig::new();
        assert!(TiltProcessor::new(cfg, 0.0).is_err());
        assert!(TiltProcessor::new(cfg, -44100.0).is_err());
        assert!(TiltProcessor::new(cfg, f32::NAN).is_err());
    }

    #[test]
    fn test_empty_buffer() {
        let cfg = TiltConfig::new().with_tilt_db(6.0);
        let mut proc = TiltProcessor::new_kokoro(cfg).expect("valid");
        let mut buf: Vec<f32> = vec![];
        proc.process(&mut buf);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_tilt_k_accessor() {
        let cfg = TiltConfig::new().with_tilt_db(6.0);
        let proc = TiltProcessor::new_kokoro(cfg).unwrap();
        assert!(
            (proc.tilt_k() - 0.5).abs() < 1e-6,
            "k should be 0.5 for +6 dB"
        );

        let cfg_neg = TiltConfig::new().with_tilt_db(-12.0);
        let proc_neg = TiltProcessor::new_kokoro(cfg_neg).unwrap();
        assert!(
            (proc_neg.tilt_k() + 1.0).abs() < 1e-6,
            "k should be -1.0 for -12 dB"
        );
    }
}
