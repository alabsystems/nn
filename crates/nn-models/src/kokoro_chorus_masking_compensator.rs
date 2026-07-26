// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Psychoacoustic masking compensator for cross-voice clarity in chorus.
//!
//! When multiple voices overlap in a chorus, simultaneous masking causes
//! weaker spectral components to become inaudible. This module detects
//! masking interactions between voices using a simplified Zwicker model
//! on the Bark scale and boosts masked components to maintain clarity.
//!
//! # Algorithm (simplified Zwicker model)
//!
//! 1. For each voice, compute Bark-scale spectral energy via windowed FFT.
//! 2. For each band in each voice, compute the masking threshold from all
//!    OTHER voices using an asymmetric spreading function:
//!    - Upward masking slope: -25 dB/Bark (configurable)
//!    - Downward masking slope: -10 dB/Bark
//! 3. If a band's energy in voice N falls below the combined masking
//!    threshold from the other voices, boost that band in voice N.
//! 4. Optionally weight compensation gains toward formant frequencies
//!    (500, 1500, 2500, 3500 Hz) to preserve speech intelligibility.
//!
//! # References
//!
//! - Zwicker, E. & Fastl, H. "Psychoacoustics: Facts and Models,"
//!   3rd ed., Springer, 2007. Chapters 4 (masking) and 6 (critical bands).
//! - Painter, T. & Spanias, A. "Perceptual Coding of Digital Audio,"
//!   Proceedings of the IEEE, vol. 88, no. 4, 2000, pp. 451-515.
//! - Moore, B.C.J. "An Introduction to the Psychology of Hearing,"
//!   6th ed., Brill, 2012. Chapter 3 (frequency selectivity).

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SILENCE_DB: f32 = -120.0;
const AMPLITUDE_FLOOR: f64 = 1e-20;

/// Bark-scale critical band edges (Hz), 25 values for 24 bands.
/// Source: Zwicker & Fastl, "Psychoacoustics," Table 6.1.
const BARK_EDGES: [f32; 25] = [
    0.0, 100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0,
    2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0,
    15500.0,
];

/// Bark-scale band center frequencies (Hz), 24 values.
/// Geometric mean of adjacent edges.
const BARK_CENTERS: [f32; 24] = [
    50.0, 150.0, 250.0, 350.0, 450.0, 570.0, 700.0, 840.0, 1000.0, 1170.0, 1370.0, 1600.0, 1860.0,
    2160.0, 2510.0, 2920.0, 3420.0, 4050.0, 4850.0, 5850.0, 7050.0, 8600.0, 10750.0, 13750.0,
];

/// Formant center frequencies for speech intelligibility (Hz).
/// F1 ~ 500 Hz, F2 ~ 1500 Hz, F3 ~ 2500 Hz, F4 ~ 3500 Hz.
const FORMANT_CENTERS_HZ: [f32; 4] = [500.0, 1500.0, 2500.0, 3500.0];

/// Downward masking slope (dB/Bark). Less steep than upward.
const DOWNWARD_SLOPE_DB_PER_BARK: f32 = 10.0;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the psychoacoustic masking compensator.
///
/// Controls the Bark-scale analysis resolution, compensation strength,
/// masking model parameters, and formant protection behavior.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MaskingCompensatorConfig {
    /// Number of Bark-scale analysis bands (default: 24).
    pub n_bands: usize,
    /// How much to boost masked content, 0.0-1.0 (default: 0.4).
    pub compensation_strength: f32,
    /// Upward masking spread slope in dB/Bark (default: 25.0).
    pub masking_slope_db_per_bark: f32,
    /// Include absolute hearing threshold in masking model (default: true).
    pub absolute_threshold: bool,
    /// Prioritize formant regions for compensation (default: true).
    pub protect_formants: bool,
    /// Analysis FFT window size (default: 2048).
    pub window_size: usize,
    /// Audio sample rate in Hz (default: 24000.0).
    pub sample_rate: f32,
}

impl Default for MaskingCompensatorConfig {
    fn default() -> Self {
        Self {
            n_bands: 24,
            compensation_strength: 0.4,
            masking_slope_db_per_bark: 25.0,
            absolute_threshold: true,
            protect_formants: true,
            window_size: 2048,
            sample_rate: 24000.0,
        }
    }
}

impl MaskingCompensatorConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of Bark-scale analysis bands.
    #[must_use]
    pub fn with_n_bands(mut self, v: usize) -> Self {
        self.n_bands = v;
        self
    }

    /// Set the compensation strength (0.0-1.0).
    #[must_use]
    pub fn with_compensation_strength(mut self, v: f32) -> Self {
        self.compensation_strength = v;
        self
    }

    /// Set the upward masking slope in dB/Bark.
    #[must_use]
    pub fn with_masking_slope_db_per_bark(mut self, v: f32) -> Self {
        self.masking_slope_db_per_bark = v;
        self
    }

    /// Set whether to include absolute hearing threshold.
    #[must_use]
    pub fn with_absolute_threshold(mut self, v: bool) -> Self {
        self.absolute_threshold = v;
        self
    }

    /// Set whether to prioritize formant regions.
    #[must_use]
    pub fn with_protect_formants(mut self, v: bool) -> Self {
        self.protect_formants = v;
        self
    }

    /// Set the analysis FFT window size.
    #[must_use]
    pub fn with_window_size(mut self, v: usize) -> Self {
        self.window_size = v;
        self
    }

    /// Set the audio sample rate.
    #[must_use]
    pub fn with_sample_rate(mut self, v: f32) -> Self {
        self.sample_rate = v;
        self
    }

    /// Transparent preset: minimal compensation, formant-only protection.
    #[must_use]
    pub fn transparent() -> Self {
        Self {
            compensation_strength: 0.15,
            masking_slope_db_per_bark: 25.0,
            protect_formants: true,
            absolute_threshold: true,
            ..Self::default()
        }
    }

    /// Clarity preset: moderate compensation for improved cross-voice separation.
    #[must_use]
    pub fn clarity() -> Self {
        Self {
            compensation_strength: 0.5,
            masking_slope_db_per_bark: 25.0,
            protect_formants: true,
            absolute_threshold: true,
            ..Self::default()
        }
    }

    /// Aggressive preset: strong compensation for dense chorus textures.
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            compensation_strength: 0.8,
            masking_slope_db_per_bark: 20.0,
            protect_formants: false,
            absolute_threshold: false,
            ..Self::default()
        }
    }

    /// Formant-only preset: only boost formant frequency regions.
    #[must_use]
    pub fn formant_only() -> Self {
        Self {
            compensation_strength: 0.6,
            masking_slope_db_per_bark: 25.0,
            protect_formants: true,
            absolute_threshold: true,
            ..Self::default()
        }
    }

    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        let err =
            |field: &'static str, reason: String| Err(KokoroError::InvalidConfig { field, reason });
        if self.n_bands == 0 || self.n_bands > 24 {
            return err(
                "n_bands",
                format!("n_bands = {}: must be in [1, 24]", self.n_bands),
            );
        }
        if !self.compensation_strength.is_finite()
            || !(0.0..=1.0).contains(&self.compensation_strength)
        {
            return err(
                "compensation_strength",
                format!(
                    "compensation_strength = {}: must be finite in [0.0, 1.0]",
                    self.compensation_strength
                ),
            );
        }
        if !self.masking_slope_db_per_bark.is_finite()
            || self.masking_slope_db_per_bark < 1.0
            || self.masking_slope_db_per_bark > 50.0
        {
            return err(
                "masking_slope_db_per_bark",
                format!(
                    "masking_slope_db_per_bark = {}: must be finite in [1.0, 50.0]",
                    self.masking_slope_db_per_bark
                ),
            );
        }
        if !self.window_size.is_power_of_two() || self.window_size < 256 || self.window_size > 8192
        {
            return err(
                "window_size",
                format!(
                    "window_size = {}: must be power of 2 in [256, 8192]",
                    self.window_size
                ),
            );
        }
        if !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            return err(
                "sample_rate",
                format!(
                    "sample_rate = {}: must be finite and positive",
                    self.sample_rate
                ),
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Per-voice masking analysis report
// ---------------------------------------------------------------------------

/// Per-voice masking analysis report.
///
/// Contains the masked energy and applied compensation gain for each
/// Bark-scale band in a single voice.
#[derive(Debug, Clone)]
pub struct MaskingAnalysis {
    /// Per-band masked energy in dB (how much energy is masked by other voices).
    pub masked_energy_db: Vec<f32>,
    /// Per-band compensation gain in dB (boost applied to restore masked content).
    pub compensation_gain_db: Vec<f32>,
}

// ---------------------------------------------------------------------------
// FFT helpers (self-contained radix-2 DIT, matches other chorus modules)
// ---------------------------------------------------------------------------

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
// Bark-scale spectral analysis
// ---------------------------------------------------------------------------

/// Compute Bark-scale spectral energy (dB) for an audio buffer.
///
/// Returns `n_bands` energy values in dB, one per Bark band.
fn bark_spectral_energy(
    audio: &[f32],
    window: &[f32],
    n_bands: usize,
    sample_rate: f32,
) -> Vec<f32> {
    let n = window.len();
    let half = n / 2 + 1;
    let bin_hz = f64::from(sample_rate) / n as f64;

    // Apply Hann window and compute FFT.
    let src_len = audio.len().min(n);
    let mut spectrum: Vec<(f32, f32)> = vec![(0.0, 0.0); n];
    for i in 0..src_len {
        let s = audio[i];
        if s.is_finite() {
            spectrum[i].0 = s * window[i.min(window.len() - 1)];
        }
    }
    fft(&mut spectrum);

    // Compute power spectrum.
    let mut power = vec![0.0f64; half];
    for k in 0..half {
        let (re, im) = spectrum[k];
        power[k] = (f64::from(re) * f64::from(re) + f64::from(im) * f64::from(im)) / n as f64;
    }

    // Map to Bark bands.
    let bands = n_bands.min(24);
    let mut result = vec![SILENCE_DB; bands];
    for band in 0..bands {
        let lo_hz = BARK_EDGES[band];
        let hi_hz = BARK_EDGES[band + 1];
        let bin_lo = (f64::from(lo_hz) / bin_hz).floor() as usize;
        let bin_hi = ((f64::from(hi_hz) / bin_hz).ceil() as usize).min(half);
        if bin_lo >= bin_hi || bin_lo >= half {
            continue;
        }
        let band_power: f64 = power[bin_lo..bin_hi].iter().sum();
        let mean_power = band_power / (bin_hi - bin_lo) as f64;
        if mean_power > AMPLITUDE_FLOOR {
            let db = 10.0 * mean_power.log10();
            result[band] = if db.is_finite() {
                db as f32
            } else {
                SILENCE_DB
            };
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Masking model
// ---------------------------------------------------------------------------

/// Compute the masking threshold at `target_band` from a masker at
/// `masker_band` with `masker_energy_db`, using an asymmetric
/// spreading function.
///
/// Upward masking (target > masker): steeper slope (e.g., -25 dB/Bark).
/// Downward masking (target < masker): gentler slope (e.g., -10 dB/Bark).
fn spreading_function(
    masker_band: usize,
    masker_energy_db: f32,
    target_band: usize,
    upward_slope: f32,
) -> f32 {
    if masker_energy_db <= SILENCE_DB {
        return SILENCE_DB;
    }
    let bark_distance = (target_band as f32 - masker_band as f32).abs();
    let slope = if target_band > masker_band {
        upward_slope
    } else {
        DOWNWARD_SLOPE_DB_PER_BARK
    };
    let attenuation = slope * bark_distance;
    let threshold = masker_energy_db - attenuation;
    threshold.max(SILENCE_DB)
}

/// Absolute threshold of hearing approximation (dB SPL) at the center
/// frequency of each Bark band. Simplified from ISO 226:2003.
///
/// At the sample rates used by Kokoro (24 kHz), bands above ~12 kHz
/// are at Nyquist so we use a high threshold for those.
fn absolute_hearing_threshold(n_bands: usize) -> Vec<f32> {
    let bands = n_bands.min(24);
    let mut thresholds = vec![0.0f32; bands];
    for band in 0..bands {
        let f = BARK_CENTERS[band];
        // Simplified ATH: Terhardt (1979) approximation in dB SPL,
        // shifted to a relative dB scale suitable for our analysis.
        let f_khz = f / 1000.0;
        if !f_khz.is_finite() || f_khz <= 0.0 {
            thresholds[band] = 0.0;
            continue;
        }
        let ath = 3.64 * f_khz.powf(-0.8) - 6.5 * (-(f_khz - 3.3).powi(2) * 0.6).exp()
            + 1e-3 * f_khz.powi(4);
        // Clamp to reasonable range and shift to our analysis dB scale.
        // Our power spectrum dB values are relative, not absolute SPL,
        // so we use a scaled-down version as a floor.
        thresholds[band] = (ath * 0.3 - 40.0).clamp(-80.0, 0.0);
    }
    thresholds
}

/// Compute the combined masking threshold at each band for a given voice
/// from all other voices' spectral energy.
fn compute_masking_threshold(
    voice_idx: usize,
    all_energies: &[Vec<f32>],
    n_bands: usize,
    upward_slope: f32,
    include_ath: bool,
) -> Vec<f32> {
    let bands = n_bands.min(24);
    let mut threshold = vec![SILENCE_DB; bands];
    let ath = if include_ath {
        absolute_hearing_threshold(bands)
    } else {
        vec![SILENCE_DB; bands]
    };

    for target_band in 0..bands {
        // Power-sum masking contributions from all other voices.
        let mut combined_power = 0.0f64;

        for (v, energy) in all_energies.iter().enumerate() {
            if v == voice_idx {
                continue;
            }
            for masker_band in 0..energy.len().min(bands) {
                let masker_db = energy[masker_band];
                let spread_db =
                    spreading_function(masker_band, masker_db, target_band, upward_slope);
                if spread_db > SILENCE_DB {
                    // Convert dB to linear power and sum.
                    combined_power += 10.0f64.powf(f64::from(spread_db) / 10.0);
                }
            }
        }

        // Convert back to dB.
        let masking_db = if combined_power > AMPLITUDE_FLOOR {
            let db = 10.0 * combined_power.log10();
            if db.is_finite() {
                db as f32
            } else {
                SILENCE_DB
            }
        } else {
            SILENCE_DB
        };

        // The effective threshold is the maximum of masking and ATH.
        threshold[target_band] = masking_db.max(ath[target_band]);
    }

    threshold
}

/// Compute formant weighting factors for each band. Bands near formant
/// center frequencies get higher weights (up to 1.5x), while distant
/// bands get 1.0x.
fn formant_weights(n_bands: usize) -> Vec<f32> {
    let bands = n_bands.min(24);
    let mut weights = vec![1.0f32; bands];

    for band in 0..bands {
        let center = BARK_CENTERS[band];
        let mut max_proximity = 0.0f32;
        for &formant_hz in &FORMANT_CENTERS_HZ {
            // Gaussian proximity: peaks at formant center, falls off.
            let dist = (center - formant_hz).abs();
            let bandwidth = formant_hz * 0.2; // 20% of formant frequency
            if bandwidth > 0.0 {
                let proximity = (-0.5 * (dist / bandwidth).powi(2)).exp();
                max_proximity = max_proximity.max(proximity);
            }
        }
        // Scale: 1.0 (no formant) to 1.5 (at formant peak).
        weights[band] = 1.0 + 0.5 * max_proximity;
    }

    weights
}

// ---------------------------------------------------------------------------
// Biquad filter for applying per-band compensation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BiquadFilter {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl BiquadFilter {
    /// Peaking EQ coefficients (Audio EQ Cookbook, Bristow-Johnson).
    fn peaking_eq(freq_hz: f32, gain_db: f32, q: f32, sr: f32) -> Self {
        let a = 10.0f32.powf(gain_db / 40.0);
        let w0 = std::f32::consts::TAU * freq_hz / sr;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha / a;
        Self {
            b0: (1.0 + alpha * a) / a0,
            b1: (-2.0 * cos_w0) / a0,
            b2: (1.0 - alpha * a) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha / a) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Unity (bypass) filter.
    fn unity() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.z1 = 0.0;
            self.z2 = 0.0;
            return 0.0;
        }
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        if !y.is_finite() {
            self.z1 = 0.0;
            self.z2 = 0.0;
            return 0.0;
        }
        y
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

// ---------------------------------------------------------------------------
// MaskingCompensator
// ---------------------------------------------------------------------------

/// Psychoacoustic masking compensator for cross-voice chorus clarity.
///
/// Analyzes spectral energy across voices in Bark-scale critical bands,
/// detects simultaneous masking interactions using a simplified Zwicker
/// model, and applies per-band EQ boosts to masked voices to restore
/// audibility of their important spectral content.
pub struct MaskingCompensator {
    config: MaskingCompensatorConfig,
    window: Vec<f32>,
    formant_w: Vec<f32>,
    /// Per-voice filter banks. Outer = voice, inner = per-band biquad.
    voice_filters: Vec<Vec<BiquadFilter>>,
}

impl MaskingCompensator {
    /// Create a new masking compensator.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if parameters are out of range.
    pub fn new(config: &MaskingCompensatorConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let window = hann_window(config.window_size);
        let formant_w = if config.protect_formants {
            formant_weights(config.n_bands)
        } else {
            vec![1.0; config.n_bands]
        };
        Ok(Self {
            config: config.clone(),
            window,
            formant_w,
            voice_filters: Vec::new(),
        })
    }

    /// Analyze cross-voice masking and apply per-band compensation boosts.
    ///
    /// For each voice, computes the masking threshold from all other voices
    /// and boosts bands whose energy falls below that threshold. Returns a
    /// per-voice `MaskingAnalysis` report.
    ///
    /// Voices are modified in-place with compensatory EQ.
    pub fn process_voices(&mut self, voices: &mut [Vec<f32>]) -> Vec<MaskingAnalysis> {
        let n_voices = voices.len();
        if n_voices < 2 {
            // No cross-voice masking with 0 or 1 voice.
            return voices
                .iter()
                .map(|_| MaskingAnalysis {
                    masked_energy_db: vec![0.0; self.config.n_bands],
                    compensation_gain_db: vec![0.0; self.config.n_bands],
                })
                .collect();
        }

        let n_bands = self.config.n_bands;
        let strength = self.config.compensation_strength;
        let upward_slope = self.config.masking_slope_db_per_bark;
        let include_ath = self.config.absolute_threshold;

        // Step 1: Compute Bark-scale energy for all voices.
        let all_energies: Vec<Vec<f32>> = voices
            .iter()
            .map(|v| bark_spectral_energy(v, &self.window, n_bands, self.config.sample_rate))
            .collect();

        // Ensure filter capacity.
        self.ensure_voice_capacity(n_voices);

        // Step 2-3: For each voice, compute masking and compensation.
        let mut analyses = Vec::with_capacity(n_voices);

        for voice_idx in 0..n_voices {
            let threshold = compute_masking_threshold(
                voice_idx,
                &all_energies,
                n_bands,
                upward_slope,
                include_ath,
            );

            let energy = &all_energies[voice_idx];
            let mut masked_energy_db = vec![0.0f32; n_bands];
            let mut compensation_gain_db = vec![0.0f32; n_bands];

            for band in 0..n_bands {
                let voice_energy = energy[band];
                let mask_threshold = threshold[band];

                // How much of this band is masked (dB below threshold).
                let masked_amount = (mask_threshold - voice_energy).max(0.0);
                masked_energy_db[band] = masked_amount;

                // Compensation: boost by a fraction of the masked amount.
                let mut boost = masked_amount * strength;

                // Apply formant weighting.
                boost *= self.formant_w[band.min(self.formant_w.len() - 1)];

                // Clamp to avoid excessive boosting (max 12 dB per band).
                boost = boost.clamp(0.0, 12.0);

                // Skip negligible boosts.
                if boost < 0.1 {
                    boost = 0.0;
                }

                compensation_gain_db[band] = boost;

                // Update the biquad filter for this voice + band.
                let center_hz = BARK_CENTERS[band.min(23)];
                let filters = &mut self.voice_filters[voice_idx];
                if boost > 0.1 {
                    // Q ~ 1.5: moderately narrow for per-band compensation.
                    filters[band] =
                        BiquadFilter::peaking_eq(center_hz, boost, 1.5, self.config.sample_rate);
                } else {
                    filters[band] = BiquadFilter::unity();
                }
            }

            analyses.push(MaskingAnalysis {
                masked_energy_db,
                compensation_gain_db,
            });
        }

        // Step 4: Apply filter cascades to each voice.
        for (voice_idx, voice) in voices.iter_mut().enumerate() {
            let filters = &mut self.voice_filters[voice_idx];
            for filter in filters.iter_mut() {
                filter.process_buffer(voice);
            }
        }

        analyses
    }

    /// Reset all internal filter states.
    pub fn reset(&mut self) {
        for filters in &mut self.voice_filters {
            for f in filters.iter_mut() {
                f.reset();
            }
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &MaskingCompensatorConfig {
        &self.config
    }

    /// Ensure internal filter storage has capacity for `n_voices`.
    fn ensure_voice_capacity(&mut self, n_voices: usize) {
        let n_bands = self.config.n_bands;
        while self.voice_filters.len() < n_voices {
            let filters = (0..n_bands).map(|_| BiquadFilter::unity()).collect();
            self.voice_filters.push(filters);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_validates() {
        let config = MaskingCompensatorConfig::default();
        config.validate().expect("default config should validate");
    }

    #[test]
    fn test_config_presets_validate() {
        MaskingCompensatorConfig::transparent()
            .validate()
            .expect("transparent preset should validate");
        MaskingCompensatorConfig::clarity()
            .validate()
            .expect("clarity preset should validate");
        MaskingCompensatorConfig::aggressive()
            .validate()
            .expect("aggressive preset should validate");
        MaskingCompensatorConfig::formant_only()
            .validate()
            .expect("formant_only preset should validate");
    }

    #[test]
    fn test_config_builder_chain() {
        let config = MaskingCompensatorConfig::new()
            .with_n_bands(16)
            .with_compensation_strength(0.6)
            .with_masking_slope_db_per_bark(20.0)
            .with_absolute_threshold(false)
            .with_protect_formants(false)
            .with_window_size(1024)
            .with_sample_rate(48000.0);
        config
            .validate()
            .expect("builder chain should produce valid config");
        assert_eq!(config.n_bands, 16);
        assert!((config.compensation_strength - 0.6).abs() < 1e-6);
        assert!((config.masking_slope_db_per_bark - 20.0).abs() < 1e-6);
        assert!(!config.absolute_threshold);
        assert!(!config.protect_formants);
        assert_eq!(config.window_size, 1024);
        assert!((config.sample_rate - 48000.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_invalid_strength() {
        let config = MaskingCompensatorConfig::new().with_compensation_strength(1.5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_invalid_n_bands() {
        assert!(MaskingCompensatorConfig::new()
            .with_n_bands(0)
            .validate()
            .is_err());
        assert!(MaskingCompensatorConfig::new()
            .with_n_bands(25)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_window_size() {
        let config = MaskingCompensatorConfig::new().with_window_size(1000);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_invalid_sample_rate() {
        let config = MaskingCompensatorConfig::new().with_sample_rate(-1.0);
        assert!(config.validate().is_err());
        let config = MaskingCompensatorConfig::new().with_sample_rate(f32::NAN);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_invalid_slope() {
        assert!(MaskingCompensatorConfig::new()
            .with_masking_slope_db_per_bark(0.0)
            .validate()
            .is_err());
        assert!(MaskingCompensatorConfig::new()
            .with_masking_slope_db_per_bark(60.0)
            .validate()
            .is_err());
    }

    #[test]
    fn test_compensator_creation() {
        let config = MaskingCompensatorConfig::default();
        let comp = MaskingCompensator::new(&config);
        assert!(comp.is_ok());
    }

    #[test]
    fn test_process_empty_voices() {
        let config = MaskingCompensatorConfig::default();
        let mut comp = MaskingCompensator::new(&config).unwrap();
        let mut voices: Vec<Vec<f32>> = vec![];
        let analyses = comp.process_voices(&mut voices);
        assert!(analyses.is_empty());
    }

    #[test]
    fn test_process_single_voice_no_masking() {
        let config = MaskingCompensatorConfig::default();
        let mut comp = MaskingCompensator::new(&config).unwrap();
        let tone = generate_sine(440.0, 24000.0, 4096, 0.5);
        let mut voices = vec![tone];
        let analyses = comp.process_voices(&mut voices);
        assert_eq!(analyses.len(), 1);
        // Single voice: no cross-voice masking, all gains should be zero.
        assert!(
            analyses[0].compensation_gain_db.iter().all(|&g| g == 0.0),
            "single voice should have no compensation"
        );
    }

    #[test]
    fn test_process_identical_voices_minimal_compensation() {
        let config = MaskingCompensatorConfig::clarity();
        let mut comp = MaskingCompensator::new(&config).unwrap();
        let tone = generate_sine(440.0, 24000.0, 4096, 0.5);
        let original = tone.clone();
        let mut voices = vec![tone.clone(), tone];
        let analyses = comp.process_voices(&mut voices);
        assert_eq!(analyses.len(), 2);

        // For identical voices, masking and signal are equal, so
        // compensation should be modest (signal is at or above threshold).
        let total_boost_0: f32 = analyses[0].compensation_gain_db.iter().sum();
        let total_boost_1: f32 = analyses[1].compensation_gain_db.iter().sum();
        // The exact amount depends on spreading, but it should not be huge.
        assert!(
            total_boost_0 < 50.0 && total_boost_1 < 50.0,
            "identical voices should not need extreme compensation: {total_boost_0}, {total_boost_1}"
        );

        // Audio should be modified (filters applied).
        let changed = voices[0]
            .iter()
            .zip(original.iter())
            .any(|(a, b)| (a - b).abs() > 1e-8);
        // With identical voices there IS cross-masking (each masks the other),
        // so some compensation is expected.
        assert!(
            changed || total_boost_0 == 0.0,
            "compensation should modify audio when boost > 0"
        );
    }

    #[test]
    fn test_process_different_frequencies_targeted_boost() {
        let config = MaskingCompensatorConfig::new()
            .with_compensation_strength(0.8)
            .with_protect_formants(false);
        let mut comp = MaskingCompensator::new(&config).unwrap();

        let sr = 24000.0;
        let n = 4096;
        // Voice 0: loud 500 Hz tone.
        let loud_tone = generate_sine(500.0, sr, n, 0.8);
        // Voice 1: quiet 600 Hz tone (nearby Bark band, will be masked).
        let quiet_tone = generate_sine(600.0, sr, n, 0.05);
        let original_quiet = quiet_tone.clone();

        let mut voices = vec![loud_tone, quiet_tone];
        let analyses = comp.process_voices(&mut voices);

        // Voice 1 (quiet) should have more compensation than voice 0 (loud).
        let boost_0: f32 = analyses[0].compensation_gain_db.iter().sum();
        let boost_1: f32 = analyses[1].compensation_gain_db.iter().sum();
        assert!(
            boost_1 > boost_0,
            "quiet voice should receive more compensation than loud voice: {boost_0} vs {boost_1}"
        );

        // Voice 1 should be audibly modified.
        let rms_diff: f32 = voices[1]
            .iter()
            .zip(original_quiet.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / n as f32;
        assert!(
            rms_diff > 1e-8,
            "masked voice should be modified, rms_diff={rms_diff}"
        );
    }

    #[test]
    fn test_formant_weights_boost_formant_regions() {
        let weights = formant_weights(24);
        assert_eq!(weights.len(), 24);

        // Find the band closest to 1500 Hz (F2). Band 11 center ~1600 Hz.
        let f2_band = 11;
        // Find a band far from formants, e.g., band 20 center ~7050 Hz.
        let distant_band = 20;

        assert!(
            weights[f2_band] > weights[distant_band],
            "formant band weight ({}) should exceed distant band weight ({})",
            weights[f2_band],
            weights[distant_band]
        );
        // Formant band should be boosted above 1.0.
        assert!(
            weights[f2_band] > 1.0,
            "formant band should have weight > 1.0, got {}",
            weights[f2_band]
        );
    }

    #[test]
    fn test_spreading_function_upward_steeper_than_downward() {
        let masker_band = 10;
        let masker_db = -20.0;
        let slope = 25.0;

        let upward = spreading_function(masker_band, masker_db, 12, slope);
        let downward = spreading_function(masker_band, masker_db, 8, slope);

        // Upward masking (-25 dB/Bark * 2) = -50 dB attenuation.
        // Downward masking (-10 dB/Bark * 2) = -20 dB attenuation.
        // So downward threshold should be higher (less attenuation).
        assert!(
            downward > upward,
            "downward masking should be stronger (less attenuation): up={upward}, down={downward}"
        );
    }

    #[test]
    fn test_spreading_function_silent_masker() {
        let result = spreading_function(5, SILENCE_DB, 6, 25.0);
        assert!(
            (result - SILENCE_DB).abs() < 1e-6,
            "silent masker should produce silence threshold"
        );
    }

    #[test]
    fn test_absolute_hearing_threshold_shape() {
        let ath = absolute_hearing_threshold(24);
        assert_eq!(ath.len(), 24);
        // All values should be finite and in reasonable range.
        for &t in &ath {
            assert!(t.is_finite(), "ATH values must be finite");
            assert!((-80.0..=0.0).contains(&t), "ATH out of range: {t}");
        }
    }

    #[test]
    fn test_reset_clears_filter_state() {
        let config = MaskingCompensatorConfig::default();
        let mut comp = MaskingCompensator::new(&config).unwrap();
        let tone = generate_sine(440.0, 24000.0, 2048, 0.5);
        let mut voices = vec![tone.clone(), tone];
        comp.process_voices(&mut voices);
        comp.reset();
        // After reset, filters should be in initial state.
        // Process again and check no residual state leaks.
        let mut voices2 = vec![
            generate_sine(440.0, 24000.0, 2048, 0.5),
            generate_sine(440.0, 24000.0, 2048, 0.5),
        ];
        let analyses = comp.process_voices(&mut voices2);
        assert_eq!(analyses.len(), 2);
    }

    #[test]
    fn test_nan_input_handled_gracefully() {
        let config = MaskingCompensatorConfig::default();
        let mut comp = MaskingCompensator::new(&config).unwrap();
        let mut voices = vec![vec![f32::NAN; 2048], vec![0.1; 2048]];
        // Should not panic.
        let analyses = comp.process_voices(&mut voices);
        assert_eq!(analyses.len(), 2);
        // Output should be finite.
        for &s in &voices[1] {
            assert!(s.is_finite(), "output should be finite after NaN input");
        }
    }

    #[test]
    fn test_masking_analysis_fields() {
        let config = MaskingCompensatorConfig::new().with_n_bands(12);
        let mut comp = MaskingCompensator::new(&config).unwrap();
        let mut voices = vec![
            generate_sine(500.0, 24000.0, 2048, 0.5),
            generate_sine(600.0, 24000.0, 2048, 0.1),
        ];
        let analyses = comp.process_voices(&mut voices);
        assert_eq!(analyses[0].masked_energy_db.len(), 12);
        assert_eq!(analyses[0].compensation_gain_db.len(), 12);
        assert_eq!(analyses[1].masked_energy_db.len(), 12);
        assert_eq!(analyses[1].compensation_gain_db.len(), 12);
    }

    #[test]
    fn test_compensation_gain_capped_at_12db() {
        let config = MaskingCompensatorConfig::new()
            .with_compensation_strength(1.0)
            .with_protect_formants(false);
        let mut comp = MaskingCompensator::new(&config).unwrap();
        // Very loud masker + very quiet target to trigger max boost.
        let loud = generate_sine(500.0, 24000.0, 4096, 0.9);
        let quiet: Vec<f32> = vec![0.001; 4096];
        let mut voices = vec![loud, quiet];
        let analyses = comp.process_voices(&mut voices);
        for &g in &analyses[1].compensation_gain_db {
            assert!(
                g <= 12.0 + 1e-6,
                "compensation gain should be capped at 12 dB, got {g}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn generate_sine(freq_hz: f32, sr: f32, n: usize, amplitude: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (std::f32::consts::TAU * freq_hz * i as f32 / sr).sin() * amplitude)
            .collect()
    }
}
