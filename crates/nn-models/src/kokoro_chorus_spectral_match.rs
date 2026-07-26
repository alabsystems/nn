// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Spectral envelope matching for Kokoro chorus voice consistency.
//!
//! When multiple TTS voices are mixed in a chorus, each voice may have a
//! different timbral character -- one brighter, another darker. This module
//! analyzes the spectral envelope of a reference voice and gently adjusts
//! all other voices to match, producing a cohesive ensemble.
//!
//! # Algorithm
//!
//! 1. Compute the band-averaged spectral envelope of the reference voice
//!    using a windowed FFT with logarithmically-spaced bands.
//! 2. Compute the same envelope for each target voice.
//! 3. Calculate per-band correction ratios (target vs. reference).
//! 4. Optionally smooth the correction curve to preserve formant structure.
//! 5. Apply corrections as a cascade of peaking EQ biquad filters.
//!
//! # References
//!
//! - Bristow-Johnson, R. "Audio EQ Cookbook." (2005).
//! - ITU-R BS.1770-4 loudness measurement (band weighting concept).
//! - Puckette, M. "The Theory and Technique of Electronic Music." (2007).

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for spectral envelope matching across chorus voices.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpectralMatchConfig {
    /// Index of the reference voice (0-based). Default: 0.
    pub reference_voice: usize,
    /// Correction strength (0.0 = none, 1.0 = full match). Default: 0.5.
    pub match_strength: f32,
    /// Number of analysis bands (logarithmically spaced). Default: 24.
    pub n_bands: usize,
    /// FFT window size (power of 2). Default: 2048.
    pub window_size: usize,
    /// Preserve formant structure by smoothing the correction curve. Default: true.
    pub preserve_formants: bool,
    /// Maximum per-band correction in dB (positive). Default: 6.0.
    pub max_correction_db: f32,
    /// Sample rate in Hz. Default: 24000.0.
    pub sample_rate: f32,
}

impl Default for SpectralMatchConfig {
    fn default() -> Self {
        Self {
            reference_voice: 0,
            match_strength: 0.5,
            n_bands: 24,
            window_size: 2048,
            preserve_formants: true,
            max_correction_db: 6.0,
            sample_rate: 24000.0,
        }
    }
}

impl SpectralMatchConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the reference voice index.
    #[must_use]
    pub fn with_reference_voice(mut self, v: usize) -> Self {
        self.reference_voice = v;
        self
    }

    /// Set the match strength (0.0-1.0).
    #[must_use]
    pub fn with_match_strength(mut self, v: f32) -> Self {
        self.match_strength = v;
        self
    }

    /// Set the number of analysis bands.
    #[must_use]
    pub fn with_n_bands(mut self, v: usize) -> Self {
        self.n_bands = v;
        self
    }

    /// Set the FFT window size.
    #[must_use]
    pub fn with_window_size(mut self, v: usize) -> Self {
        self.window_size = v;
        self
    }

    /// Set whether to preserve formants during correction.
    #[must_use]
    pub fn with_preserve_formants(mut self, v: bool) -> Self {
        self.preserve_formants = v;
        self
    }

    /// Set the maximum correction in dB.
    #[must_use]
    pub fn with_max_correction_db(mut self, v: f32) -> Self {
        self.max_correction_db = v;
        self
    }

    /// Set the sample rate.
    #[must_use]
    pub fn with_sample_rate(mut self, v: f32) -> Self {
        self.sample_rate = v;
        self
    }

    /// Subtle preset: gentle spectral nudging with formant preservation.
    #[must_use]
    pub fn subtle() -> Self {
        Self {
            match_strength: 0.25,
            max_correction_db: 3.0,
            preserve_formants: true,
            ..Self::default()
        }
    }

    /// Tight match preset: strong spectral alignment for unison passages.
    #[must_use]
    pub fn tight_match() -> Self {
        Self {
            match_strength: 0.85,
            max_correction_db: 9.0,
            preserve_formants: false,
            n_bands: 32,
            ..Self::default()
        }
    }

    /// Formant-aware preset: strong matching that preserves vocal identity.
    #[must_use]
    pub fn formant_aware() -> Self {
        Self {
            match_strength: 0.65,
            max_correction_db: 6.0,
            preserve_formants: true,
            n_bands: 24,
            ..Self::default()
        }
    }

    /// Validate all parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        let err =
            |field: &'static str, reason: String| Err(KokoroError::InvalidConfig { field, reason });
        if !self.match_strength.is_finite() || !(0.0..=1.0).contains(&self.match_strength) {
            return err(
                "match_strength",
                format!(
                    "match_strength = {}: must be finite in [0.0, 1.0]",
                    self.match_strength
                ),
            );
        }
        if self.n_bands == 0 || self.n_bands > 64 {
            return err(
                "n_bands",
                format!("n_bands = {}: must be in [1, 64]", self.n_bands),
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
        if !self.max_correction_db.is_finite()
            || self.max_correction_db < 0.0
            || self.max_correction_db > 24.0
        {
            return err(
                "max_correction_db",
                format!(
                    "max_correction_db = {}: must be finite in [0.0, 24.0]",
                    self.max_correction_db
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
// Radix-2 DIT FFT (self-contained, matches kokoro_chorus_auto_eq.rs)
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
// Band analysis
// ---------------------------------------------------------------------------

/// Compute logarithmically-spaced band center frequencies from low_hz to
/// the Nyquist limit.
fn compute_band_centers(n_bands: usize, sample_rate: f32) -> Vec<f32> {
    let low_hz = 80.0f32;
    let nyquist = sample_rate / 2.0;
    let high_hz = nyquist.min(12000.0);
    if n_bands <= 1 {
        return vec![(low_hz * high_hz).sqrt()];
    }
    let log_low = low_hz.ln();
    let log_high = high_hz.ln();
    (0..n_bands)
        .map(|i| {
            let t = i as f32 / (n_bands - 1) as f32;
            (log_low + t * (log_high - log_low)).exp()
        })
        .collect()
}

/// Compute the spectral envelope (dB per band) of an audio buffer.
fn analyze_envelope(
    audio: &[f32],
    window: &[f32],
    band_centers: &[f32],
    sample_rate: f32,
) -> Vec<f32> {
    let n = window.len();
    let mut frame = vec![0.0f32; n];
    let src_len = audio.len().min(n);
    let src_start = audio.len().saturating_sub(n);
    frame[..src_len].copy_from_slice(&audio[src_start..src_start + src_len]);

    // Apply Hann window.
    for (s, &w) in frame.iter_mut().zip(window.iter()) {
        *s *= w;
    }

    // FFT.
    let mut spectrum: Vec<(f32, f32)> = frame.iter().map(|&s| (s, 0.0)).collect();
    fft(&mut spectrum);

    let n_bins = n / 2 + 1;
    let bin_width = sample_rate / n as f32;
    // Use 1/3-octave bandwidth around each center for averaging.
    let factor = 2.0f32.powf(1.0 / 6.0);

    let mut band_db = vec![-96.0f32; band_centers.len()];
    for (idx, &center) in band_centers.iter().enumerate() {
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
            band_db[idx] = db.max(-96.0);
        }
    }
    band_db
}

/// Smooth a correction curve with a 3-tap triangular kernel to preserve
/// formant structure while eliminating sharp per-band jumps.
fn smooth_corrections(corrections: &mut [f32], iterations: usize) {
    if corrections.len() < 3 {
        return;
    }
    let mut temp = vec![0.0f32; corrections.len()];
    for _ in 0..iterations {
        temp[0] = corrections[0] * 0.667 + corrections[1] * 0.333;
        let last = corrections.len() - 1;
        temp[last] = corrections[last - 1] * 0.333 + corrections[last] * 0.667;
        for i in 1..last {
            temp[i] = corrections[i - 1] * 0.25 + corrections[i] * 0.5 + corrections[i + 1] * 0.25;
        }
        corrections.copy_from_slice(&temp);
    }
}

// ---------------------------------------------------------------------------
// SpectralMatcher
// ---------------------------------------------------------------------------

/// Spectral envelope matcher for chorus voice consistency.
///
/// Analyzes a reference voice's spectral envelope and applies gentle
/// corrective EQ to other voices so that all share a consistent timbral
/// character. The correction is applied via a cascade of peaking EQ
/// biquad filters, one per analysis band.
pub struct SpectralMatcher {
    config: SpectralMatchConfig,
    band_centers: Vec<f32>,
    window: Vec<f32>,
    /// Per-voice filter banks. Outer index = voice, inner = per-band biquad.
    voice_filters: Vec<Vec<BiquadFilter>>,
    /// Last computed per-band correction in dB for each voice.
    voice_corrections_db: Vec<Vec<f32>>,
    /// Cached reference envelope (dB per band).
    reference_envelope: Vec<f32>,
}

impl SpectralMatcher {
    /// Create a new spectral matcher.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if parameters are out of range.
    pub fn new(config: &SpectralMatchConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let band_centers = compute_band_centers(config.n_bands, config.sample_rate);
        let window = hann_window(config.window_size);
        // Q for each band: moderate bandwidth (1 octave).
        let n = band_centers.len();
        Ok(Self {
            config: config.clone(),
            band_centers,
            window,
            voice_filters: Vec::new(),
            voice_corrections_db: Vec::new(),
            reference_envelope: vec![-96.0; n],
        })
    }

    /// Match all voices to the reference voice's spectral envelope.
    ///
    /// `voices` is a mutable slice of per-voice audio buffers. The reference
    /// voice (by index from config) is analyzed but not modified. All other
    /// voices are gently corrected to match the reference's timbral profile.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if the reference voice index
    /// is out of range.
    pub fn match_to_reference(&mut self, voices: &mut [Vec<f32>]) -> Result<(), KokoroError> {
        if voices.is_empty() {
            return Ok(());
        }
        let ref_idx = self.config.reference_voice;
        if ref_idx >= voices.len() {
            return Err(KokoroError::InvalidConfig {
                field: "reference_voice",
                reason: format!(
                    "reference_voice = {} but only {} voices provided",
                    ref_idx,
                    voices.len()
                ),
            });
        }

        // Analyze reference envelope.
        self.reference_envelope = analyze_envelope(
            &voices[ref_idx],
            &self.window,
            &self.band_centers,
            self.config.sample_rate,
        );

        // Ensure we have filter banks for all voices.
        self.ensure_voice_capacity(voices.len());

        let strength = self.config.match_strength;
        let max_db = self.config.max_correction_db;
        let q = 2.0; // Moderate Q for spectral shaping.

        for voice_idx in 0..voices.len() {
            if voice_idx == ref_idx {
                continue;
            }
            if voices[voice_idx].is_empty() {
                continue;
            }

            // Analyze this voice's envelope.
            let voice_env = analyze_envelope(
                &voices[voice_idx],
                &self.window,
                &self.band_centers,
                self.config.sample_rate,
            );

            // Compute per-band corrections.
            let n = self.band_centers.len();
            let corrections = &mut self.voice_corrections_db[voice_idx];
            for band in 0..n {
                let diff = self.reference_envelope[band] - voice_env[band];
                // Scale by strength and clamp.
                let raw = diff * strength;
                corrections[band] = raw.clamp(-max_db, max_db);
            }

            // Smooth if formant preservation is enabled.
            if self.config.preserve_formants {
                smooth_corrections(corrections, 2);
            }

            // Update filter coefficients.
            let filters = &mut self.voice_filters[voice_idx];
            for band in 0..n {
                let gain_db = corrections[band];
                let effective = if gain_db.abs() > 0.05 { gain_db } else { 0.0 };
                filters[band] = BiquadFilter::new(peaking_eq_coeffs(
                    self.band_centers[band],
                    effective,
                    q,
                    self.config.sample_rate,
                ));
            }

            // Apply filter cascade to voice audio.
            for filter in filters.iter_mut() {
                filter.process_buffer(&mut voices[voice_idx]);
            }
        }

        Ok(())
    }

    /// Get the last computed reference envelope (dB per band).
    #[must_use]
    pub fn reference_envelope(&self) -> &[f32] {
        &self.reference_envelope
    }

    /// Get the per-band corrections (dB) applied to a given voice.
    ///
    /// Returns `None` if the voice index is out of range or no corrections
    /// have been computed yet.
    #[must_use]
    pub fn voice_corrections(&self, voice_idx: usize) -> Option<&[f32]> {
        self.voice_corrections_db
            .get(voice_idx)
            .map(Vec::as_slice)
    }

    /// Get the analysis band center frequencies.
    #[must_use]
    pub fn band_centers(&self) -> &[f32] {
        &self.band_centers
    }

    /// Reset all filter states and cached envelopes.
    pub fn reset(&mut self) {
        for filters in &mut self.voice_filters {
            for f in filters.iter_mut() {
                f.reset();
            }
        }
        for corrections in &mut self.voice_corrections_db {
            corrections.fill(0.0);
        }
        self.reference_envelope.fill(-96.0);
    }

    /// Ensure internal storage has capacity for `n_voices`.
    fn ensure_voice_capacity(&mut self, n_voices: usize) {
        let n_bands = self.band_centers.len();
        while self.voice_filters.len() < n_voices {
            let mut filters = Vec::with_capacity(n_bands);
            for &freq in &self.band_centers {
                filters.push(BiquadFilter::new(peaking_eq_coeffs(
                    freq,
                    0.0,
                    2.0,
                    self.config.sample_rate,
                )));
            }
            self.voice_filters.push(filters);
        }
        while self.voice_corrections_db.len() < n_voices {
            self.voice_corrections_db.push(vec![0.0; n_bands]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_validates() {
        let config = SpectralMatchConfig::default();
        config.validate().expect("default config should validate");
    }

    #[test]
    fn test_config_presets_validate() {
        SpectralMatchConfig::subtle()
            .validate()
            .expect("subtle preset should validate");
        SpectralMatchConfig::tight_match()
            .validate()
            .expect("tight_match preset should validate");
        SpectralMatchConfig::formant_aware()
            .validate()
            .expect("formant_aware preset should validate");
    }

    #[test]
    fn test_config_invalid_strength() {
        let config = SpectralMatchConfig::new().with_match_strength(1.5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_invalid_window_size() {
        let config = SpectralMatchConfig::new().with_window_size(1000);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_invalid_max_correction() {
        let config = SpectralMatchConfig::new().with_max_correction_db(-1.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_matcher_creation() {
        let config = SpectralMatchConfig::default();
        let matcher = SpectralMatcher::new(&config);
        assert!(matcher.is_ok());
        let m = matcher.unwrap();
        assert_eq!(m.band_centers().len(), 24);
    }

    #[test]
    fn test_match_empty_voices() {
        let config = SpectralMatchConfig::default();
        let mut matcher = SpectralMatcher::new(&config).unwrap();
        let mut voices: Vec<Vec<f32>> = vec![];
        assert!(matcher.match_to_reference(&mut voices).is_ok());
    }

    #[test]
    fn test_match_reference_out_of_range() {
        let config = SpectralMatchConfig::new().with_reference_voice(5);
        let mut matcher = SpectralMatcher::new(&config).unwrap();
        let mut voices = vec![vec![0.0; 1024], vec![0.0; 1024]];
        assert!(matcher.match_to_reference(&mut voices).is_err());
    }

    #[test]
    fn test_match_identical_voices_no_change() {
        let config = SpectralMatchConfig::new().with_match_strength(1.0);
        let mut matcher = SpectralMatcher::new(&config).unwrap();

        // Two identical sine waves at 440 Hz.
        let sr = 24000.0;
        let n = 4096;
        let tone: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr).sin() * 0.5)
            .collect();
        let mut voices = vec![tone.clone(), tone];
        let original = voices[1].clone();

        matcher
            .match_to_reference(&mut voices)
            .expect("matching should succeed");

        // Corrections should be near zero for identical signals.
        let corrections = matcher.voice_corrections(1).unwrap();
        let max_correction = corrections.iter().map(|c| c.abs()).fold(0.0f32, f32::max);
        // Allow a small tolerance due to filter transients.
        assert!(
            max_correction < 1.0,
            "identical voices should need minimal correction, got {max_correction} dB"
        );

        // Audio should be very close to original (filters at ~0 dB gain).
        let rms_diff: f32 = voices[1]
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / n as f32;
        assert!(
            rms_diff < 0.01,
            "identical voices should produce near-identical output, rms_diff={rms_diff}"
        );
    }

    #[test]
    fn test_match_different_voices_applies_correction() {
        let config = SpectralMatchConfig::new().with_match_strength(0.8);
        let mut matcher = SpectralMatcher::new(&config).unwrap();

        let sr = 24000.0;
        let n = 4096;
        // Reference: 440 Hz tone.
        let ref_tone: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr).sin() * 0.5)
            .collect();
        // Voice 1: 2000 Hz tone (brighter).
        let bright_tone: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 2000.0 * i as f32 / sr).sin() * 0.5)
            .collect();

        let mut voices = vec![ref_tone, bright_tone.clone()];
        matcher
            .match_to_reference(&mut voices)
            .expect("matching should succeed");

        // Voice 1 should have been modified.
        let changed = voices[1]
            .iter()
            .zip(bright_tone.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "different voices should be corrected");
    }

    #[test]
    fn test_smooth_corrections_preserves_broad_trends() {
        let mut corrections = vec![0.0, 0.0, 6.0, 0.0, 0.0];
        smooth_corrections(&mut corrections, 2);
        // The spike at index 2 should be spread out.
        assert!(corrections[2] < 6.0, "smoothing should reduce sharp peaks");
        assert!(
            corrections[1] > 0.0,
            "smoothing should spread energy to neighbors"
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let config = SpectralMatchConfig::default();
        let mut matcher = SpectralMatcher::new(&config).unwrap();
        let sr = 24000.0;
        let n = 2048;
        let tone: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr).sin() * 0.5)
            .collect();
        let mut voices = vec![tone.clone(), tone];
        matcher.match_to_reference(&mut voices).unwrap();
        matcher.reset();
        assert!(
            matcher
                .reference_envelope()
                .iter()
                .all(|&v| (v - (-96.0)).abs() < 1e-6),
            "reset should clear reference envelope"
        );
    }

    #[test]
    fn test_band_centers_logarithmically_spaced() {
        let centers = compute_band_centers(24, 24000.0);
        assert_eq!(centers.len(), 24);
        // Should be monotonically increasing.
        for w in centers.windows(2) {
            assert!(w[1] > w[0], "bands should be monotonically increasing");
        }
        // First should be around 80 Hz, last around 12000 Hz.
        assert!(centers[0] > 70.0 && centers[0] < 90.0);
        assert!(centers[23] > 11000.0 && centers[23] < 12500.0);
    }

    #[test]
    fn test_nan_in_voice_handled_gracefully() {
        let config = SpectralMatchConfig::default();
        let mut matcher = SpectralMatcher::new(&config).unwrap();
        let mut voices = vec![vec![0.1; 2048], vec![f32::NAN; 2048]];
        // Should not panic.
        let result = matcher.match_to_reference(&mut voices);
        assert!(result.is_ok());
    }
}
