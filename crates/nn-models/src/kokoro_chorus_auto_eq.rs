// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Automatic spectral-analysis EQ for Kokoro chorus voice balancing.
//!
//! Analyzes voice audio via windowed FFT, compares against a target
//! frequency-response curve, and applies corrective peaking EQ filters at
//! ISO 1/3-octave band center frequencies.
//!
//! # References
//!
//! - Bristow-Johnson, R. "Audio EQ Cookbook." (2005).
//! - ISO 266:1997 -- 1/3-octave band center frequencies.

use crate::kokoro_error::KokoroError;

/// Standard ISO 266 1/3-octave band center frequencies (20 Hz -- 20 kHz).
const THIRD_OCTAVE_CENTERS: [f32; 31] = [
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
    500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0,
    8000.0, 10000.0, 12500.0, 16000.0, 20000.0,
];

/// Q for 1/3-octave peaking EQ bandwidth (Q ~ 4.318).
const THIRD_OCTAVE_Q: f32 = 4.318;

// ---------------------------------------------------------------------------
// Target curves
// ---------------------------------------------------------------------------

/// Reference frequency response curve for automatic EQ correction.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TargetCurve {
    /// Flat response (broadcast standard): 0 dB at all bands.
    Flat,
    /// Natural speech: slight low-mid warmth, gentle presence boost, HF rolloff.
    Speech,
    /// Singing voice: mid scoop, high presence, low warmth.
    Singing,
    /// Custom target from (frequency_hz, gain_db) pairs, linearly interpolated.
    Custom(Vec<(f32, f32)>),
}

impl TargetCurve {
    /// Evaluate the target curve at each 1/3-octave band center (31 dB values).
    fn evaluate(&self) -> [f32; 31] {
        match self {
            Self::Flat => [0.0; 31],
            Self::Speech => Self::from_sparse(&[
                (0, -2.0),
                (1, -1.5),
                (2, -1.0),
                (3, -0.5), // sub rolloff
                (6, 1.0),
                (7, 1.5),
                (8, 2.0),
                (9, 1.5), // warmth
                (10, 1.0),
                (11, 0.5),
                (20, 1.5),
                (21, 2.0),
                (22, 1.5),
                (23, 1.0), // presence
                (27, -1.0),
                (28, -2.0),
                (29, -3.0),
                (30, -4.0), // HF rolloff
            ]),
            Self::Singing => Self::from_sparse(&[
                (6, 1.5),
                (7, 2.0),
                (8, 1.5), // warmth
                (13, -1.0),
                (14, -1.5),
                (15, -1.0), // mid scoop
                (21, 2.0),
                (22, 2.5),
                (23, 2.0),
                (24, 1.5), // presence
                (27, 1.0),
                (28, 0.5), // air
                (29, -1.0),
                (30, -2.0), // HF rolloff
            ]),
            Self::Custom(pairs) => {
                let mut curve = [0.0f32; 31];
                if !pairs.is_empty() {
                    for (i, &freq) in THIRD_OCTAVE_CENTERS.iter().enumerate() {
                        curve[i] = interpolate_pairs(pairs, freq);
                    }
                }
                curve
            }
        }
    }

    fn from_sparse(entries: &[(usize, f32)]) -> [f32; 31] {
        let mut curve = [0.0f32; 31];
        for &(idx, val) in entries {
            curve[idx] = val;
        }
        curve
    }
}

/// Linear interpolation of (freq, dB) pairs at a given frequency.
fn interpolate_pairs(pairs: &[(f32, f32)], freq: f32) -> f32 {
    if pairs.is_empty() {
        return 0.0;
    }
    if pairs.len() == 1 || freq <= pairs[0].0 {
        return pairs[0].1;
    }
    if freq >= pairs[pairs.len() - 1].0 {
        return pairs[pairs.len() - 1].1;
    }
    for w in pairs.windows(2) {
        let (f0, db0) = w[0];
        let (f1, db1) = w[1];
        if freq >= f0 && freq <= f1 {
            let t = if (f1 - f0).abs() < 1e-10 {
                0.0
            } else {
                (freq - f0) / (f1 - f0)
            };
            return db0 + t * (db1 - db0);
        }
    }
    0.0
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the automatic spectral EQ processor.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AutoEqConfig {
    /// Whether auto-EQ is enabled.
    pub enabled: bool,
    /// FFT size for spectral analysis (power of 2, 512-4096).
    pub analysis_window: usize,
    /// Target frequency response curve.
    pub target_curve: TargetCurve,
    /// Correction strength (0.0 = none, 1.0 = full).
    pub correction_strength: f32,
    /// Maximum boost in dB (positive). Default: 6.0.
    pub max_boost_db: f32,
    /// Maximum cut in dB (positive value, applied as negative). Default: 12.0.
    pub max_cut_db: f32,
    /// Smoothing width in octaves. Default: 1/3.
    pub smoothing_octaves: f32,
    /// Number of correction bands (max 31). Default: 31.
    pub n_bands: usize,
}

impl Default for AutoEqConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            analysis_window: 2048,
            target_curve: TargetCurve::Flat,
            correction_strength: 0.5,
            max_boost_db: 6.0,
            max_cut_db: 12.0,
            smoothing_octaves: 1.0 / 3.0,
            n_bands: 31,
        }
    }
}

impl AutoEqConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Set whether auto-EQ is enabled.
    #[must_use]
    pub fn with_enabled(mut self, v: bool) -> Self {
        self.enabled = v;
        self
    }
    /// Set the FFT analysis window size.
    #[must_use]
    pub fn with_analysis_window(mut self, v: usize) -> Self {
        self.analysis_window = v;
        self
    }
    /// Set the target frequency response curve.
    #[must_use]
    pub fn with_target_curve(mut self, v: TargetCurve) -> Self {
        self.target_curve = v;
        self
    }
    /// Set correction strength (0.0-1.0).
    #[must_use]
    pub fn with_correction_strength(mut self, v: f32) -> Self {
        self.correction_strength = v;
        self
    }
    /// Set maximum boost in dB.
    #[must_use]
    pub fn with_max_boost_db(mut self, v: f32) -> Self {
        self.max_boost_db = v;
        self
    }
    /// Set maximum cut in dB.
    #[must_use]
    pub fn with_max_cut_db(mut self, v: f32) -> Self {
        self.max_cut_db = v;
        self
    }
    /// Set smoothing width in octaves.
    #[must_use]
    pub fn with_smoothing_octaves(mut self, v: f32) -> Self {
        self.smoothing_octaves = v;
        self
    }

    /// Validate that all parameters are within valid ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        let err =
            |field: &'static str, reason: String| Err(KokoroError::InvalidConfig { field, reason });
        if !self.analysis_window.is_power_of_two()
            || self.analysis_window < 512
            || self.analysis_window > 4096
        {
            return err(
                "analysis_window",
                format!(
                    "analysis_window = {}: must be power of 2 in [512, 4096]",
                    self.analysis_window,
                ),
            );
        }
        if !self.correction_strength.is_finite() || !(0.0..=1.0).contains(&self.correction_strength)
        {
            return err(
                "correction_strength",
                format!(
                    "correction_strength = {}: must be finite in [0.0, 1.0]",
                    self.correction_strength,
                ),
            );
        }
        if !self.max_boost_db.is_finite() || self.max_boost_db < 0.0 || self.max_boost_db > 24.0 {
            return err(
                "max_boost_db",
                format!(
                    "max_boost_db = {}: must be finite in [0.0, 24.0]",
                    self.max_boost_db,
                ),
            );
        }
        if !self.max_cut_db.is_finite() || self.max_cut_db < 0.0 || self.max_cut_db > 48.0 {
            return err(
                "max_cut_db",
                format!(
                    "max_cut_db = {}: must be finite in [0.0, 48.0]",
                    self.max_cut_db,
                ),
            );
        }
        if !self.smoothing_octaves.is_finite()
            || self.smoothing_octaves < 0.05
            || self.smoothing_octaves > 3.0
        {
            return err(
                "smoothing_octaves",
                format!(
                    "smoothing_octaves = {}: must be finite in [0.05, 3.0]",
                    self.smoothing_octaves,
                ),
            );
        }
        if self.n_bands == 0 || self.n_bands > 31 {
            return err(
                "n_bands",
                format!("n_bands = {}: must be in [1, 31]", self.n_bands),
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Radix-2 DIT FFT (self-contained, matches kokoro_chorus_freeze.rs)
// ---------------------------------------------------------------------------

/// In-place radix-2 decimation-in-time FFT. `data` length MUST be power of 2.
fn fft(data: &mut [(f32, f32)]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    if n <= 1 {
        return;
    }
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            data.swap(i, j);
        }
    }
    let mut stage_len = 2;
    while stage_len <= n {
        let half = stage_len / 2;
        let angle_step = -std::f32::consts::TAU / stage_len as f32;
        for k in (0..n).step_by(stage_len) {
            for j in 0..half {
                let angle = angle_step * j as f32;
                let (tw_re, tw_im) = (angle.cos(), angle.sin());
                let (a_re, a_im) = data[k + j];
                let (b_re, b_im) = data[k + j + half];
                let t_re = b_re * tw_re - b_im * tw_im;
                let t_im = b_re * tw_im + b_im * tw_re;
                data[k + j] = (a_re + t_re, a_im + t_im);
                data[k + j + half] = (a_re - t_re, a_im - t_im);
            }
        }
        stage_len *= 2;
    }
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            0.5 * (1.0 - (std::f32::consts::TAU * t).cos())
        })
        .collect()
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

    fn process_buffer(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            *s = self.process(*s);
        }
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

/// Peaking EQ biquad coefficients (Audio EQ Cookbook, Bristow-Johnson).
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
// AutoEqProcessor
// ---------------------------------------------------------------------------

/// Automatic spectral EQ processor.
///
/// Analyzes input audio spectrum, compares to a target curve, designs
/// corrective peaking EQ filters at 1/3-octave band centers, and applies
/// them to the audio signal.
pub struct AutoEqProcessor {
    config: AutoEqConfig,
    filters: Vec<BiquadFilter>,
    band_frequencies: Vec<f32>,
    band_gains_db: Vec<f32>,
    window: Vec<f32>,
    target_db: Vec<f32>,
    sample_rate: f32,
}

impl AutoEqProcessor {
    /// Create a new auto-EQ processor.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if parameters are out of range.
    pub fn new(config: &AutoEqConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("sample_rate = {sample_rate}: must be finite and positive"),
            });
        }

        let n_bands = config.n_bands.min(31);
        let target_full = config.target_curve.evaluate();
        let nyquist = sample_rate / 2.0;

        let mut band_frequencies = Vec::with_capacity(n_bands);
        let mut target_db = Vec::with_capacity(n_bands);
        for i in 0..31 {
            if band_frequencies.len() >= n_bands {
                break;
            }
            if THIRD_OCTAVE_CENTERS[i] < nyquist {
                band_frequencies.push(THIRD_OCTAVE_CENTERS[i]);
                target_db.push(target_full[i]);
            }
        }

        let mut filters = Vec::with_capacity(band_frequencies.len());
        for &freq in &band_frequencies {
            filters.push(BiquadFilter::new(peaking_eq_coeffs(
                freq,
                0.0,
                THIRD_OCTAVE_Q,
                sample_rate,
            )));
        }

        Ok(Self {
            config: config.clone(),
            filters,
            band_gains_db: vec![0.0f32; band_frequencies.len()],
            band_frequencies,
            window: hann_window(config.analysis_window),
            target_db,
            sample_rate,
        })
    }

    /// Analyze the spectrum of `audio`, returning measured dB per band.
    #[must_use]
    pub fn analyze_spectrum(&self, audio: &[f32]) -> Vec<f32> {
        let n = self.config.analysis_window;
        let mut frame = vec![0.0f32; n];
        let src_len = audio.len().min(n);
        let src_start = audio.len().saturating_sub(n);
        frame[..src_len].copy_from_slice(&audio[src_start..src_start + src_len]);

        for (s, &w) in frame.iter_mut().zip(self.window.iter()) {
            *s *= w;
        }

        let mut spectrum: Vec<(f32, f32)> = frame.iter().map(|&s| (s, 0.0)).collect();
        fft(&mut spectrum);

        let n_bins = n / 2 + 1;
        let bin_width = self.sample_rate / n as f32;
        let factor = 2.0f32.powf(1.0 / 6.0);
        let mut band_db = vec![-96.0f32; self.band_frequencies.len()];

        for (band_idx, &center) in self.band_frequencies.iter().enumerate() {
            let bin_low = ((center / factor) / bin_width).floor() as usize;
            let bin_high = ((center * factor) / bin_width).ceil() as usize;
            let bin_low = bin_low.max(1);
            let bin_high = bin_high.min(n_bins - 1);
            if bin_low > bin_high {
                continue;
            }

            let mut energy = 0.0f32;
            let mut count = 0usize;
            for bin in bin_low..=bin_high {
                let (re, im) = spectrum[bin];
                energy += re * re + im * im;
                count += 1;
            }
            if count > 0 && energy > 0.0 {
                let db = 20.0 * (energy / count as f32).sqrt().log10();
                band_db[band_idx] = db.max(-96.0);
            }
        }
        band_db
    }

    /// Analyze the audio spectrum and apply corrective EQ in place.
    pub fn analyze_and_correct(&mut self, audio: &mut [f32], sample_rate: f32) {
        if !self.config.enabled || audio.is_empty() {
            return;
        }

        let measured_db = self.analyze_spectrum(audio);
        let sr = if sample_rate > 0.0 && sample_rate.is_finite() {
            sample_rate
        } else {
            self.sample_rate
        };

        for (i, &measured) in measured_db.iter().enumerate() {
            if i >= self.target_db.len() || i >= self.band_frequencies.len() {
                break;
            }
            let raw = (self.target_db[i] - measured) * self.config.correction_strength;
            let clamped = raw
                .max(-self.config.max_cut_db)
                .min(self.config.max_boost_db);
            self.band_gains_db[i] = clamped;

            let gain = if clamped.abs() > 0.01 { clamped } else { 0.0 };
            self.filters[i] = BiquadFilter::new(peaking_eq_coeffs(
                self.band_frequencies[i],
                gain,
                THIRD_OCTAVE_Q,
                sr,
            ));
        }

        for filter in &mut self.filters {
            filter.process_buffer(audio);
        }
    }

    /// Get the current correction gains (dB) at each band.
    #[must_use]
    pub fn band_gains(&self) -> &[f32] {
        &self.band_gains_db
    }

    /// Get the band center frequencies.
    #[must_use]
    pub fn band_frequencies(&self) -> &[f32] {
        &self.band_frequencies
    }

    /// Reset all filter states (e.g., between audio segments).
    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
        self.band_gains_db.fill(0.0);
    }
}

#[cfg(test)]
#[path = "kokoro_chorus_auto_eq_tests.rs"]
mod tests;
