// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fletcher-Munson equal-loudness contour compensation for Kokoro chorus.
//!
//! Human hearing is not flat: we perceive bass and treble as quieter at
//! lower SPL (ISO 226:2003). When a mix mastered at studio monitor levels
//! (~85 phon) is played back at casual headphone levels (~65 phon), bass
//! and treble appear to recede. This module computes the perceptual
//! difference between a reference monitoring level and the actual playback
//! level, then applies a multi-band shelving correction so the chorus
//! sounds spectrally balanced at any volume.
//!
//! # Algorithm
//!
//! 1. Evaluate ISO 226:2003 equal-loudness contours at the reference and
//!    target phon levels for each 1/3-octave band center (20 Hz--12.5 kHz).
//! 2. Compute the correction curve: `reference_spl - target_spl` at each
//!    band. This is the dB boost needed so the listener perceives the same
//!    spectral balance at the lower playback level.
//! 3. Approximate the correction curve via cascaded low-shelf and
//!    high-shelf biquad filters, blended by the `strength` parameter.
//!
//! # References
//!
//! - ISO 226:2003, "Acoustics -- Normal equal-loudness-level contours."
//! - Suzuki, Y. & Takeshima, H. "Equal-loudness-level contours for
//!   pure tones." JASA 116(2), 918--933, 2004.
//! - Bristow-Johnson, R. "Audio EQ Cookbook." (2005).

use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// ISO 226:2003 equal-loudness contour data
// ---------------------------------------------------------------------------

/// Standard 1/3-octave band center frequencies from 20 Hz to 12.5 kHz
/// (ISO 266:1997). 31 bands covering the audible range relevant to the
/// equal-loudness contour standard.
const THIRD_OCTAVE_CENTERS: [f32; 31] = [
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
    500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0,
    8000.0, 10000.0, 12500.0, 16000.0, 20000.0,
];

/// SPL values (dB) for the equal-loudness contour at selected phon levels.
///
/// Each row is the SPL required at each 1/3-octave center to produce the
/// given phon-level loudness perception. Derived from ISO 226:2003 Table 1
/// and Suzuki & Takeshima (2004) interpolation.
///
/// Phon levels: 20, 30, 40, 50, 60, 70, 80, 90.
/// Frequencies: 31 bands from 20 Hz to 20 kHz.
const CONTOUR_PHON_LEVELS: [f32; 8] = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];

/// SPL in dB at each 1/3-octave center for the equal-loudness contours.
/// Row index = phon level index (into `CONTOUR_PHON_LEVELS`).
/// Column index = frequency band index (into `THIRD_OCTAVE_CENTERS`).
///
/// Values approximate ISO 226:2003 data. Below 25 Hz and above 12.5 kHz
/// the standard does not define contours precisely; values are
/// extrapolated conservatively.
const CONTOUR_SPL: [[f32; 31]; 8] = [
    // 20 phon
    [
        78.5, 74.0, 68.7, 63.0, 58.0, 53.5, 49.5, 46.5, 44.0, 42.0, 40.5, 38.5, 36.5, 34.0, 31.5,
        29.0, 26.5, 25.0, 24.5, 24.5, 25.0, 25.5, 26.0, 27.0, 29.0, 32.5, 38.0, 45.5, 55.0, 65.0,
        75.0,
    ],
    // 30 phon
    [
        74.0, 70.0, 65.0, 59.5, 55.0, 51.0, 47.5, 44.5, 42.0, 40.0, 38.5, 37.0, 35.5, 34.0, 32.5,
        31.0, 30.0, 30.0, 29.5, 29.5, 30.0, 30.5, 31.0, 31.5, 33.0, 36.0, 41.0, 47.5, 56.0, 66.0,
        76.0,
    ],
    // 40 phon
    [
        70.0, 66.5, 62.0, 57.0, 52.5, 49.0, 46.0, 43.5, 41.5, 40.0, 39.0, 38.0, 37.0, 36.0, 35.0,
        34.0, 33.5, 33.5, 33.5, 34.0, 34.5, 35.0, 35.5, 36.0, 37.5, 39.5, 44.0, 50.0, 58.0, 67.0,
        77.0,
    ],
    // 50 phon
    [
        66.0, 63.0, 59.0, 55.0, 51.0, 47.5, 45.0, 43.0, 41.5, 40.5, 40.0, 39.5, 39.0, 38.5, 38.0,
        37.5, 37.5, 37.5, 37.5, 38.0, 38.5, 39.0, 39.5, 40.5, 42.0, 44.0, 48.0, 53.0, 60.0, 69.0,
        78.0,
    ],
    // 60 phon
    [
        63.0, 60.0, 57.0, 53.5, 50.5, 47.5, 45.0, 43.5, 42.5, 41.5, 41.0, 41.0, 41.0, 41.0, 41.0,
        41.0, 41.0, 41.0, 41.0, 41.5, 42.0, 42.5, 43.0, 44.0, 46.0, 48.5, 52.0, 57.0, 63.0, 71.0,
        80.0,
    ],
    // 70 phon
    [
        60.0, 57.5, 55.0, 52.0, 50.0, 48.0, 46.0, 44.5, 43.5, 43.0, 42.5, 42.5, 42.5, 43.0, 43.5,
        44.0, 44.5, 45.0, 45.0, 45.5, 46.0, 46.5, 47.0, 48.0, 50.0, 52.5, 56.0, 61.0, 67.0, 74.0,
        82.0,
    ],
    // 80 phon
    [
        57.5, 55.5, 53.5, 51.5, 50.0, 48.5, 47.5, 46.5, 45.5, 45.0, 44.5, 44.5, 44.5, 45.0, 46.0,
        47.0, 47.5, 48.5, 49.0, 49.5, 50.0, 50.5, 51.0, 52.0, 54.0, 56.5, 60.0, 65.0, 71.0, 78.0,
        85.0,
    ],
    // 90 phon
    [
        56.0, 54.5, 53.0, 51.5, 50.5, 49.5, 49.0, 48.5, 48.0, 47.5, 47.5, 47.5, 47.5, 48.0, 49.0,
        50.0, 51.0, 52.0, 52.5, 53.5, 54.0, 54.5, 55.0, 56.0, 58.0, 60.5, 64.0, 69.0, 75.0, 82.0,
        89.0,
    ],
];

/// Interpolate the equal-loudness contour SPL at a given phon level and
/// frequency band index.
///
/// Linearly interpolates between the two nearest contour rows. Clamps
/// to the 20--90 phon range defined by the lookup table.
fn contour_spl_at(phon: f32, band: usize) -> f32 {
    let phon = phon.clamp(CONTOUR_PHON_LEVELS[0], *CONTOUR_PHON_LEVELS.last().unwrap());
    // Find the two bracketing phon levels.
    let mut lo_idx = 0;
    for i in 0..CONTOUR_PHON_LEVELS.len() - 1 {
        if phon >= CONTOUR_PHON_LEVELS[i] {
            lo_idx = i;
        }
    }
    let hi_idx = (lo_idx + 1).min(CONTOUR_PHON_LEVELS.len() - 1);
    if lo_idx == hi_idx {
        return CONTOUR_SPL[lo_idx][band];
    }
    let lo_phon = CONTOUR_PHON_LEVELS[lo_idx];
    let hi_phon = CONTOUR_PHON_LEVELS[hi_idx];
    let t = (phon - lo_phon) / (hi_phon - lo_phon);
    let lo_spl = CONTOUR_SPL[lo_idx][band];
    let hi_spl = CONTOUR_SPL[hi_idx][band];
    lo_spl + t * (hi_spl - lo_spl)
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Fletcher-Munson equal-loudness contour correction.
///
/// Constructed via [`LoudnessCurveConfig::new`] and builder methods
/// (required for cross-crate use due to `#[non_exhaustive]`).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LoudnessCurveConfig {
    /// Target playback loudness in phons (the level the listener hears at).
    /// Range: 20.0--90.0. Default: 70.0 (moderate casual listening).
    pub target_phon: f32,
    /// Reference monitoring level in phons (the level the mix was mastered at).
    /// Range: 20.0--90.0. Default: 85.0 (studio nearfield monitors).
    pub reference_phon: f32,
    /// Whether to apply low-frequency (bass) compensation.
    pub bass_boost_enabled: bool,
    /// Whether to apply high-frequency (treble) compensation.
    pub treble_boost_enabled: bool,
    /// Correction strength: 0.0 = flat (no correction), 1.0 = full contour
    /// compensation. Default: 0.5.
    pub strength: f32,
    /// Number of correction bands (1--31, one-third octave). Default: 31.
    pub n_bands: usize,
}

impl Default for LoudnessCurveConfig {
    fn default() -> Self {
        Self {
            target_phon: 70.0,
            reference_phon: 85.0,
            bass_boost_enabled: true,
            treble_boost_enabled: true,
            strength: 0.5,
            n_bands: 31,
        }
    }
}

impl LoudnessCurveConfig {
    /// Create a config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set target playback phon level.
    #[must_use]
    pub fn with_target_phon(mut self, v: f32) -> Self {
        self.target_phon = v;
        self
    }

    /// Set reference monitoring phon level.
    #[must_use]
    pub fn with_reference_phon(mut self, v: f32) -> Self {
        self.reference_phon = v;
        self
    }

    /// Enable or disable bass (low-frequency) compensation.
    #[must_use]
    pub fn with_bass_boost_enabled(mut self, v: bool) -> Self {
        self.bass_boost_enabled = v;
        self
    }

    /// Enable or disable treble (high-frequency) compensation.
    #[must_use]
    pub fn with_treble_boost_enabled(mut self, v: bool) -> Self {
        self.treble_boost_enabled = v;
        self
    }

    /// Set correction strength (0.0--1.0).
    #[must_use]
    pub fn with_strength(mut self, v: f32) -> Self {
        self.strength = v;
        self
    }

    /// Set number of correction bands (1--31).
    #[must_use]
    pub fn with_n_bands(mut self, v: usize) -> Self {
        self.n_bands = v;
        self
    }

    /// Validate all parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        let err = |field: &'static str, reason: String| -> Result<(), KokoroError> {
            Err(KokoroError::InvalidConfig { field, reason })
        };
        if !self.target_phon.is_finite() || self.target_phon < 20.0 || self.target_phon > 90.0 {
            return err(
                "target_phon",
                format!("must be finite in [20.0, 90.0], got {}", self.target_phon),
            );
        }
        if !self.reference_phon.is_finite()
            || self.reference_phon < 20.0
            || self.reference_phon > 90.0
        {
            return err(
                "reference_phon",
                format!(
                    "must be finite in [20.0, 90.0], got {}",
                    self.reference_phon
                ),
            );
        }
        if !self.strength.is_finite() || self.strength < 0.0 || self.strength > 1.0 {
            return err(
                "strength",
                format!("must be finite in [0.0, 1.0], got {}", self.strength),
            );
        }
        if self.n_bands == 0 || self.n_bands > 31 {
            return err(
                "n_bands",
                format!("must be in [1, 31], got {}", self.n_bands),
            );
        }
        Ok(())
    }

    // --- Presets ---------------------------------------------------------------

    /// Casual headphone listening: target 65 phon, reference 85, strength 0.6.
    ///
    /// Significant bass and treble boost to compensate for the large
    /// perceptual difference between studio and headphone levels.
    #[must_use]
    pub fn headphone_casual() -> Self {
        Self {
            target_phon: 65.0,
            reference_phon: 85.0,
            bass_boost_enabled: true,
            treble_boost_enabled: true,
            strength: 0.6,
            n_bands: 31,
        }
    }

    /// Studio monitor reference: target and reference both at 85 phon.
    ///
    /// Produces a flat (no correction) curve since the playback and
    /// monitoring levels match. Strength 0.0 for true bypass.
    #[must_use]
    pub fn studio_monitor() -> Self {
        Self {
            target_phon: 85.0,
            reference_phon: 85.0,
            bass_boost_enabled: true,
            treble_boost_enabled: true,
            strength: 0.0,
            n_bands: 31,
        }
    }

    /// Quiet late-night listening: target 50 phon, reference 85, strength 0.8.
    ///
    /// Strong compensation for the substantial bass and treble loss
    /// perceived at very low listening levels.
    #[must_use]
    pub fn quiet_listening() -> Self {
        Self {
            target_phon: 50.0,
            reference_phon: 85.0,
            bass_boost_enabled: true,
            treble_boost_enabled: true,
            strength: 0.8,
            n_bands: 31,
        }
    }

    /// Broadcast delivery: target 70 phon, reference 85, strength 0.4.
    ///
    /// Mild compensation appropriate for unknown playback environments.
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            target_phon: 70.0,
            reference_phon: 85.0,
            bass_boost_enabled: true,
            treble_boost_enabled: true,
            strength: 0.4,
            n_bands: 31,
        }
    }
}

// ---------------------------------------------------------------------------
// Biquad filter (Direct Form II Transposed)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct BiquadCoeffs {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

#[derive(Debug, Clone)]
struct BiquadFilter {
    coeffs: BiquadCoeffs,
    z1: f32,
    z2: f32,
}

impl BiquadFilter {
    fn new(coeffs: BiquadCoeffs) -> Self {
        Self {
            coeffs,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        if !input.is_finite() {
            self.z1 = 0.0;
            self.z2 = 0.0;
            return 0.0;
        }
        let c = &self.coeffs;
        let output = c.b0 * input + self.z1;
        self.z1 = c.b1 * input - c.a1 * output + self.z2;
        self.z2 = c.b2 * input - c.a2 * output;
        if !output.is_finite() {
            self.z1 = 0.0;
            self.z2 = 0.0;
            0.0
        } else {
            output
        }
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Shelving filter coefficient design (Audio EQ Cookbook, Bristow-Johnson)
// ---------------------------------------------------------------------------

/// Low-shelf biquad coefficients.
///
/// Boosts or cuts frequencies below `freq_hz` by `gain_db`.
fn low_shelf_coeffs(freq_hz: f32, gain_db: f32, q: f32, sr: f32) -> BiquadCoeffs {
    let a = 10.0f32.powf(gain_db / 40.0);
    let w0 = std::f32::consts::TAU * freq_hz / sr;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);
    let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

    let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
    BiquadCoeffs {
        b0: (a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha)) / a0,
        b1: (2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0)) / a0,
        b2: (a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha)) / a0,
        a1: (-2.0 * ((a - 1.0) + (a + 1.0) * cos_w0)) / a0,
        a2: ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha) / a0,
    }
}

/// High-shelf biquad coefficients.
///
/// Boosts or cuts frequencies above `freq_hz` by `gain_db`.
fn high_shelf_coeffs(freq_hz: f32, gain_db: f32, q: f32, sr: f32) -> BiquadCoeffs {
    let a = 10.0f32.powf(gain_db / 40.0);
    let w0 = std::f32::consts::TAU * freq_hz / sr;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);
    let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
    BiquadCoeffs {
        b0: (a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha)) / a0,
        b1: (-2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)) / a0,
        b2: (a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha)) / a0,
        a1: (2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)) / a0,
        a2: ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha) / a0,
    }
}

/// Peaking EQ biquad coefficients.
///
/// Boosts or cuts a band centered at `freq_hz` by `gain_db` with quality
/// factor `q`.
fn peaking_eq_coeffs(freq_hz: f32, gain_db: f32, q: f32, sr: f32) -> BiquadCoeffs {
    let a = 10.0f32.powf(gain_db / 40.0);
    let w0 = std::f32::consts::TAU * freq_hz / sr;
    let cos_w0 = w0.cos();
    let alpha = w0.sin() / (2.0 * q);
    let a0 = 1.0 + alpha / a;
    BiquadCoeffs {
        b0: (1.0 + alpha * a) / a0,
        b1: (-2.0 * cos_w0) / a0,
        b2: (1.0 - alpha * a) / a0,
        a1: (-2.0 * cos_w0) / a0,
        a2: (1.0 - alpha / a) / a0,
    }
}

// ---------------------------------------------------------------------------
// Correction band
// ---------------------------------------------------------------------------

/// A single correction band: frequency, raw gain, and biquad filter.
#[derive(Debug, Clone)]
struct CorrectionBand {
    /// Center frequency in Hz.
    freq_hz: f32,
    /// Correction gain in dB (positive = boost, negative = cut).
    gain_db: f32,
    /// Biquad filter implementing this band's correction.
    filter: BiquadFilter,
}

// ---------------------------------------------------------------------------
// LoudnessCurveProcessor
// ---------------------------------------------------------------------------

/// Multi-band Fletcher-Munson equal-loudness contour correction processor.
///
/// Pre-computes the correction curve at construction time by evaluating
/// the ISO 226 contours at the reference and target phon levels. The
/// difference is decomposed into a bank of shelving and peaking biquad
/// filters that approximate the correction curve.
pub struct LoudnessCurveProcessor {
    config: LoudnessCurveConfig,
    sample_rate: f32,
    /// Per-band correction filters, ordered by frequency.
    bands: Vec<CorrectionBand>,
    /// The raw correction curve in dB at each band center (before strength
    /// scaling), for inspection/debug.
    raw_correction_db: Vec<f32>,
}

impl LoudnessCurveProcessor {
    /// Create a new processor.
    ///
    /// Pre-computes the contour difference and designs the correction
    /// filter bank. The `sample_rate` must be finite and positive.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is invalid.
    pub fn new(config: &LoudnessCurveConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("must be finite and positive, got {sample_rate}"),
            });
        }

        let nyquist = sample_rate / 2.0;
        let n_bands = config.n_bands.min(31);

        // Compute the correction curve: reference contour SPL minus target
        // contour SPL. A positive value means we need to boost that band
        // so the listener at the target level perceives the same balance
        // as at the reference level.
        let mut raw_correction_db = Vec::with_capacity(n_bands);
        let mut active_bands: Vec<(f32, f32)> = Vec::with_capacity(n_bands);

        for i in 0..31 {
            if active_bands.len() >= n_bands {
                break;
            }
            let freq = THIRD_OCTAVE_CENTERS[i];
            if freq >= nyquist {
                break;
            }

            let ref_spl = contour_spl_at(config.reference_phon, i);
            let tgt_spl = contour_spl_at(config.target_phon, i);
            // The correction needed: at the target level, the listener
            // needs `ref_spl - tgt_spl` additional dB to perceive the
            // same loudness balance as at the reference level.
            let mut correction = ref_spl - tgt_spl;

            // Apply bass/treble enable flags. The crossover between
            // "bass" and "treble" is 1 kHz (band index 17).
            let is_bass = freq < 1000.0;
            let is_treble = freq > 1000.0;
            if is_bass && !config.bass_boost_enabled {
                correction = 0.0;
            }
            if is_treble && !config.treble_boost_enabled {
                correction = 0.0;
            }

            // Scale by strength.
            let scaled = correction * config.strength;

            raw_correction_db.push(correction);
            active_bands.push((freq, scaled));
        }

        // Design the filter bank. Use:
        // - A low-shelf for the lowest active band (bass correction).
        // - A high-shelf for the highest active band (treble correction).
        // - Peaking EQ for all mid bands.
        // Q of 0.707 (Butterworth) for shelves gives smooth transitions.
        // Q of ~2.0 for peaking gives moderate bandwidth overlap.
        let shelf_q: f32 = 0.707;
        let peak_q: f32 = 2.0;

        let mut bands = Vec::with_capacity(active_bands.len());
        for (idx, &(freq, gain_db)) in active_bands.iter().enumerate() {
            // Skip bands with negligible correction.
            if gain_db.abs() < 0.05 {
                bands.push(CorrectionBand {
                    freq_hz: freq,
                    gain_db: 0.0,
                    filter: BiquadFilter::new(BiquadCoeffs {
                        b0: 1.0,
                        b1: 0.0,
                        b2: 0.0,
                        a1: 0.0,
                        a2: 0.0,
                    }),
                });
                continue;
            }

            let coeffs = if idx == 0 {
                // Lowest band: low-shelf.
                low_shelf_coeffs(freq, gain_db, shelf_q, sample_rate)
            } else if idx == active_bands.len() - 1 {
                // Highest band: high-shelf.
                high_shelf_coeffs(freq, gain_db, shelf_q, sample_rate)
            } else {
                // Mid bands: peaking EQ.
                peaking_eq_coeffs(freq, gain_db, peak_q, sample_rate)
            };

            bands.push(CorrectionBand {
                freq_hz: freq,
                gain_db,
                filter: BiquadFilter::new(coeffs),
            });
        }

        Ok(Self {
            config: config.clone(),
            sample_rate,
            bands,
            raw_correction_db,
        })
    }

    /// Create a processor at Kokoro's default 24 kHz sample rate.
    pub fn new_kokoro(config: &LoudnessCurveConfig) -> Result<Self, KokoroError> {
        Self::new(config, KOKORO_SAMPLE_RATE as f32)
    }

    /// Apply the equal-loudness correction to audio in-place.
    ///
    /// Each sample passes through the cascaded filter bank. Processing is
    /// a no-op when `strength == 0.0` or all correction gains are zero.
    pub fn process(&mut self, audio: &mut [f32]) {
        if audio.is_empty() || self.config.strength == 0.0 {
            return;
        }

        for sample in audio.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
                continue;
            }
            let mut x = *sample;
            for band in &mut self.bands {
                x = band.filter.process(x);
            }
            *sample = if x.is_finite() { x } else { 0.0 };
        }
    }

    /// Clear all internal filter states (call between unrelated audio segments).
    pub fn reset(&mut self) {
        for band in &mut self.bands {
            band.filter.reset();
        }
    }

    /// Read-only access to the configuration.
    #[must_use]
    pub fn config(&self) -> &LoudnessCurveConfig {
        &self.config
    }

    /// The sample rate this processor was constructed for.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// The raw (un-strength-scaled) correction curve in dB at each active
    /// band center. Positive = boost needed, negative = cut needed.
    #[must_use]
    pub fn raw_correction_db(&self) -> &[f32] {
        &self.raw_correction_db
    }

    /// The per-band correction gains actually applied (strength-scaled) in dB.
    #[must_use]
    pub fn band_gains_db(&self) -> Vec<f32> {
        self.bands.iter().map(|b| b.gain_db).collect()
    }

    /// The center frequencies of the active correction bands.
    #[must_use]
    pub fn band_frequencies(&self) -> Vec<f32> {
        self.bands.iter().map(|b| b.freq_hz).collect()
    }

    /// Number of active correction bands.
    #[must_use]
    pub fn n_active_bands(&self) -> usize {
        self.bands.len()
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
            .map(|i| amplitude * (std::f32::consts::TAU * freq * i as f32 / SR).sin())
            .collect()
    }

    fn rms(buf: &[f32]) -> f32 {
        let sum_sq: f32 = buf.iter().map(|x| x * x).sum();
        (sum_sq / buf.len().max(1) as f32).sqrt()
    }

    // --- Config validation ---

    #[test]
    fn test_default_config_valid() {
        LoudnessCurveConfig::new()
            .validate()
            .expect("default config should validate");
    }

    #[test]
    fn test_builder_roundtrip() {
        let cfg = LoudnessCurveConfig::new()
            .with_target_phon(60.0)
            .with_reference_phon(80.0)
            .with_bass_boost_enabled(false)
            .with_treble_boost_enabled(true)
            .with_strength(0.7)
            .with_n_bands(15);
        cfg.validate().expect("builder config should validate");
        assert_eq!(cfg.target_phon, 60.0);
        assert_eq!(cfg.reference_phon, 80.0);
        assert!(!cfg.bass_boost_enabled);
        assert!(cfg.treble_boost_enabled);
        assert!((cfg.strength - 0.7).abs() < 1e-6);
        assert_eq!(cfg.n_bands, 15);
    }

    #[test]
    fn test_invalid_target_phon() {
        assert!(LoudnessCurveConfig::new()
            .with_target_phon(10.0)
            .validate()
            .is_err());
        assert!(LoudnessCurveConfig::new()
            .with_target_phon(100.0)
            .validate()
            .is_err());
        assert!(LoudnessCurveConfig::new()
            .with_target_phon(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_invalid_reference_phon() {
        assert!(LoudnessCurveConfig::new()
            .with_reference_phon(10.0)
            .validate()
            .is_err());
        assert!(LoudnessCurveConfig::new()
            .with_reference_phon(100.0)
            .validate()
            .is_err());
        assert!(LoudnessCurveConfig::new()
            .with_reference_phon(f32::INFINITY)
            .validate()
            .is_err());
    }

    #[test]
    fn test_invalid_strength() {
        assert!(LoudnessCurveConfig::new()
            .with_strength(-0.1)
            .validate()
            .is_err());
        assert!(LoudnessCurveConfig::new()
            .with_strength(1.1)
            .validate()
            .is_err());
        assert!(LoudnessCurveConfig::new()
            .with_strength(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_invalid_n_bands() {
        assert!(LoudnessCurveConfig::new()
            .with_n_bands(0)
            .validate()
            .is_err());
        assert!(LoudnessCurveConfig::new()
            .with_n_bands(32)
            .validate()
            .is_err());
    }

    #[test]
    fn test_boundary_values_valid() {
        LoudnessCurveConfig::new()
            .with_target_phon(20.0)
            .validate()
            .expect("20 phon valid");
        LoudnessCurveConfig::new()
            .with_target_phon(90.0)
            .validate()
            .expect("90 phon valid");
        LoudnessCurveConfig::new()
            .with_strength(0.0)
            .validate()
            .expect("0.0 strength valid");
        LoudnessCurveConfig::new()
            .with_strength(1.0)
            .validate()
            .expect("1.0 strength valid");
        LoudnessCurveConfig::new()
            .with_n_bands(1)
            .validate()
            .expect("1 band valid");
        LoudnessCurveConfig::new()
            .with_n_bands(31)
            .validate()
            .expect("31 bands valid");
    }

    // --- Presets ---

    #[test]
    fn test_all_presets_valid() {
        LoudnessCurveConfig::headphone_casual()
            .validate()
            .expect("headphone_casual valid");
        LoudnessCurveConfig::studio_monitor()
            .validate()
            .expect("studio_monitor valid");
        LoudnessCurveConfig::quiet_listening()
            .validate()
            .expect("quiet_listening valid");
        LoudnessCurveConfig::broadcast()
            .validate()
            .expect("broadcast valid");
    }

    #[test]
    fn test_preset_parameters() {
        let hp = LoudnessCurveConfig::headphone_casual();
        assert_eq!(hp.target_phon, 65.0);
        assert_eq!(hp.reference_phon, 85.0);
        assert!((hp.strength - 0.6).abs() < 1e-6);

        let sm = LoudnessCurveConfig::studio_monitor();
        assert_eq!(sm.target_phon, 85.0);
        assert_eq!(sm.reference_phon, 85.0);
        assert_eq!(sm.strength, 0.0);

        let ql = LoudnessCurveConfig::quiet_listening();
        assert_eq!(ql.target_phon, 50.0);
        assert!((ql.strength - 0.8).abs() < 1e-6);

        let bc = LoudnessCurveConfig::broadcast();
        assert_eq!(bc.target_phon, 70.0);
        assert!((bc.strength - 0.4).abs() < 1e-6);
    }

    // --- Contour interpolation ---

    #[test]
    fn test_contour_spl_at_exact_levels() {
        // At exactly 70 phon, band 17 (1 kHz) should return the table value.
        let spl = contour_spl_at(70.0, 17);
        assert!(
            (spl - CONTOUR_SPL[5][17]).abs() < 0.01,
            "70 phon @ 1 kHz: expected {}, got {spl}",
            CONTOUR_SPL[5][17],
        );
    }

    #[test]
    fn test_contour_spl_interpolates() {
        // 55 phon should be between 50 and 60 phon contours at band 0.
        let spl_55 = contour_spl_at(55.0, 0);
        let spl_50 = contour_spl_at(50.0, 0);
        let spl_60 = contour_spl_at(60.0, 0);
        assert!(
            spl_55 >= spl_60.min(spl_50) && spl_55 <= spl_60.max(spl_50),
            "55 phon should interpolate between 50 and 60: {spl_50} <= {spl_55} <= {spl_60}",
        );
    }

    #[test]
    fn test_contour_spl_clamps() {
        // Below 20 phon should clamp to 20.
        let spl_10 = contour_spl_at(10.0, 17);
        let spl_20 = contour_spl_at(20.0, 17);
        assert!(
            (spl_10 - spl_20).abs() < 0.01,
            "10 phon should clamp to 20: {spl_10} vs {spl_20}",
        );
        // Above 90 phon should clamp to 90.
        let spl_100 = contour_spl_at(100.0, 17);
        let spl_90 = contour_spl_at(90.0, 17);
        assert!(
            (spl_100 - spl_90).abs() < 0.01,
            "100 phon should clamp to 90: {spl_100} vs {spl_90}",
        );
    }

    // --- Processor construction ---

    #[test]
    fn test_processor_construction_default() {
        let cfg = LoudnessCurveConfig::new();
        let proc = LoudnessCurveProcessor::new_kokoro(&cfg).expect("should construct");
        assert!(proc.n_active_bands() > 0);
        assert_eq!(proc.sample_rate(), SR);
    }

    #[test]
    fn test_processor_invalid_sample_rate() {
        let cfg = LoudnessCurveConfig::new();
        assert!(LoudnessCurveProcessor::new(&cfg, 0.0).is_err());
        assert!(LoudnessCurveProcessor::new(&cfg, -44100.0).is_err());
        assert!(LoudnessCurveProcessor::new(&cfg, f32::NAN).is_err());
    }

    #[test]
    fn test_correction_curve_direction() {
        // When target < reference (listening quieter than mastered), the
        // correction should be positive at low frequencies (bass boost)
        // because we perceive less bass at lower SPL.
        let cfg = LoudnessCurveConfig::new()
            .with_target_phon(50.0)
            .with_reference_phon(85.0)
            .with_strength(1.0);
        let proc = LoudnessCurveProcessor::new_kokoro(&cfg).expect("should construct");
        let raw = proc.raw_correction_db();
        // Band 0 is 20 Hz -- should have positive correction.
        assert!(
            raw[0] > 0.0,
            "low-freq correction should be positive (bass boost): got {}",
            raw[0],
        );
    }

    #[test]
    fn test_equal_phon_produces_zero_correction() {
        // When target == reference, all corrections should be zero.
        let cfg = LoudnessCurveConfig::new()
            .with_target_phon(70.0)
            .with_reference_phon(70.0)
            .with_strength(1.0);
        let proc = LoudnessCurveProcessor::new_kokoro(&cfg).expect("should construct");
        for &g in proc.raw_correction_db() {
            assert!(
                g.abs() < 0.01,
                "same phon levels should produce zero correction, got {g}",
            );
        }
    }

    // --- Audio processing behavior ---

    #[test]
    fn test_studio_monitor_is_passthrough() {
        let cfg = LoudnessCurveConfig::studio_monitor();
        let mut proc = LoudnessCurveProcessor::new_kokoro(&cfg).expect("should construct");
        let original = sine_wave(440.0, 4096, 0.5);
        let mut buf = original.clone();
        proc.process(&mut buf);
        // strength == 0.0 should be a no-op.
        assert_eq!(buf, original, "studio_monitor preset should be passthrough");
    }

    #[test]
    fn test_headphone_casual_boosts_bass() {
        let n = 8192;
        let cfg = LoudnessCurveConfig::headphone_casual();
        let mut proc = LoudnessCurveProcessor::new_kokoro(&cfg).expect("should construct");

        let mut bass = sine_wave(80.0, n, 0.3);
        let dry_rms = rms(&bass);
        proc.process(&mut bass);
        let wet_rms = rms(&bass);

        assert!(
            wet_rms > dry_rms,
            "headphone_casual should boost 80 Hz bass: dry={dry_rms}, wet={wet_rms}",
        );
    }

    #[test]
    fn test_quiet_listening_strong_correction() {
        let cfg = LoudnessCurveConfig::quiet_listening();
        let proc = LoudnessCurveProcessor::new_kokoro(&cfg).expect("should construct");
        let gains = proc.band_gains_db();
        // At least some bands should have significant non-zero gains.
        let max_abs = gains.iter().map(|g| g.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs > 0.5,
            "quiet_listening should have significant correction, max |gain| = {max_abs}",
        );
    }

    #[test]
    fn test_all_outputs_finite() {
        let cfg = LoudnessCurveConfig::headphone_casual();
        let mut proc = LoudnessCurveProcessor::new_kokoro(&cfg).expect("should construct");
        let mut buf = vec![
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
        proc.process(&mut buf);
        for (i, &v) in buf.iter().enumerate() {
            assert!(v.is_finite(), "sample {i} is non-finite: {v}");
        }
    }

    #[test]
    fn test_empty_buffer() {
        let cfg = LoudnessCurveConfig::new();
        let mut proc = LoudnessCurveProcessor::new_kokoro(&cfg).expect("should construct");
        let mut buf: Vec<f32> = vec![];
        proc.process(&mut buf);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = LoudnessCurveConfig::headphone_casual();
        let mut proc = LoudnessCurveProcessor::new_kokoro(&cfg).expect("should construct");
        let mut buf = vec![0.5; 100];
        proc.process(&mut buf);
        proc.reset();
        // After reset, all filter states should be zero.
        for band in &proc.bands {
            assert_eq!(band.filter.z1, 0.0, "z1 should be zero after reset");
            assert_eq!(band.filter.z2, 0.0, "z2 should be zero after reset");
        }
    }

    #[test]
    fn test_bass_disable_zeroes_low_bands() {
        let cfg = LoudnessCurveConfig::new()
            .with_target_phon(50.0)
            .with_reference_phon(85.0)
            .with_bass_boost_enabled(false)
            .with_strength(1.0);
        let proc = LoudnessCurveProcessor::new_kokoro(&cfg).expect("should construct");
        // All bands below 1 kHz should have zero raw correction.
        let freqs = proc.band_frequencies();
        let raw = proc.raw_correction_db();
        for (i, &freq) in freqs.iter().enumerate() {
            if freq < 1000.0 {
                assert!(
                    raw[i].abs() < 0.01,
                    "bass disabled: band {i} ({freq} Hz) should be ~0, got {}",
                    raw[i],
                );
            }
        }
    }

    #[test]
    fn test_treble_disable_zeroes_high_bands() {
        let cfg = LoudnessCurveConfig::new()
            .with_target_phon(50.0)
            .with_reference_phon(85.0)
            .with_treble_boost_enabled(false)
            .with_strength(1.0);
        let proc = LoudnessCurveProcessor::new_kokoro(&cfg).expect("should construct");
        let freqs = proc.band_frequencies();
        let raw = proc.raw_correction_db();
        for (i, &freq) in freqs.iter().enumerate() {
            if freq > 1000.0 {
                assert!(
                    raw[i].abs() < 0.01,
                    "treble disabled: band {i} ({freq} Hz) should be ~0, got {}",
                    raw[i],
                );
            }
        }
    }

    #[test]
    fn test_n_bands_respected() {
        let cfg = LoudnessCurveConfig::new().with_n_bands(10);
        let proc = LoudnessCurveProcessor::new_kokoro(&cfg).expect("should construct");
        assert!(
            proc.n_active_bands() <= 10,
            "expected <= 10 active bands, got {}",
            proc.n_active_bands(),
        );
    }

    #[test]
    fn test_band_frequencies_below_nyquist() {
        let cfg = LoudnessCurveConfig::new();
        let proc = LoudnessCurveProcessor::new(&cfg, 8000.0).expect("should construct");
        let nyquist = 4000.0;
        for &freq in &proc.band_frequencies() {
            assert!(
                freq < nyquist,
                "band freq {freq} Hz exceeds Nyquist {nyquist} Hz",
            );
        }
    }

    #[test]
    fn test_all_presets_construct_and_process() {
        let presets = [
            LoudnessCurveConfig::headphone_casual(),
            LoudnessCurveConfig::studio_monitor(),
            LoudnessCurveConfig::quiet_listening(),
            LoudnessCurveConfig::broadcast(),
        ];
        for (i, cfg) in presets.iter().enumerate() {
            let mut proc = LoudnessCurveProcessor::new_kokoro(cfg)
                .unwrap_or_else(|e| panic!("preset {i} failed to construct: {e}"));
            let mut buf = sine_wave(440.0, 2048, 0.5);
            proc.process(&mut buf);
            for (j, &v) in buf.iter().enumerate() {
                assert!(v.is_finite(), "preset {i} sample {j} is non-finite: {v}");
            }
        }
    }
}
